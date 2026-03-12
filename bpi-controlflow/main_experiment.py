# main.py
# 200 samples — sequence-level Levenshtein, no binned analysis

import random
import pickle
import pandas as pd
import numpy as np
import editdistance
from anthropic import Anthropic
from openai import OpenAI


# Load data

with open("traces.pkl", "rb") as f:
    traces = pickle.load(f)

with open("clean_rules.pkl", "rb") as f:
    clean_rules = pickle.load(f)

traces = traces.tolist()
valid_edges = set(clean_rules.keys())

EVENTS_LIST = sorted(list({e for t in traces for e in t}))

# Map each event to a fixed-length ID so edit distance operates on uniform tokens
event_to_id = {e: f"E{str(i).zfill(3)}" for i, e in enumerate(EVENTS_LIST)}


# Config

MODELS = {
    "haiku": ("anthropic", "claude-haiku-4-5"),
    "sonnet": ("anthropic", "claude-sonnet-4-5"),
    "gpt4nano": ("openai", "gpt-4.1-nano")
}

FEWSHOT_SETTINGS = [0, 5, 15]
SEEDS = [0, 1, 2]
RETRIEVAL_MODES = ["levenshtein", "random"]

N_SAMPLES = 200

anthropic_client = Anthropic()
openai_client = OpenAI()

CACHE = {}


# Sequence-level Levenshtein: each event ID is one indivisible token
# E001->E002->E003 vs E001->E002->E017 = distance 1
# E001->E002->E003 vs E001->E004->E005 = distance 2

def lev_similarity_seq(seq1, seq2):
    dist = editdistance.eval(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 1.0
    return 1 - dist / max_len


# Model call with caching

def query_model(provider, model_id, prompt):

    cache_key = (provider, model_id, prompt)
    if cache_key in CACHE:
        return CACHE[cache_key]

    if provider == "anthropic":
        response = anthropic_client.messages.create(
            model=model_id,
            max_tokens=40,
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.content[0].text.strip()

    elif provider == "openai":
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content.strip()

    CACHE[cache_key] = output
    return output


def clean_prediction(pred):
    pred = pred.split("\n")[0].strip()
    for e in EVENTS_LIST:
        if e in pred:
            return e
    return pred


# Main evaluation loop

all_metrics = []

for model_name, (provider, model_id) in MODELS.items():

    print("\n====================")
    print("MODEL:", model_name)
    print("====================")

    for retrieval_mode in RETRIEVAL_MODES:

        for seed in SEEDS:

            random.seed(seed)

            shuffled = traces.copy()
            random.shuffle(shuffled)

            split_idx = int(0.8 * len(shuffled))
            train_traces = shuffled[:split_idx]
            test_traces = shuffled[split_idx:]

            # Build train candidates and their ID token lists once per seed
            train_candidates = []
            train_candidate_id_seqs = []

            for t in train_traces:
                for i in range(1, len(t)):
                    p = t[:i]
                    nxt = t[i]
                    train_candidates.append((p, nxt))
                    train_candidate_id_seqs.append([event_to_id[e] for e in p])

            # Build test examples once per seed
            examples = []

            for trace in test_traces:
                if len(trace) < 2:
                    continue
                idx = random.randint(1, len(trace) - 1)
                prefix = trace[:idx]
                target = trace[idx]
                examples.append((prefix, target))

            random.shuffle(examples)
            examples = examples[:N_SAMPLES]

            # Precompute prefix ID token lists for sequence-level comparison
            prefix_id_seqs = [[event_to_id[e] for e in p] for p, _ in examples]

            for N_FEWSHOT in FEWSHOT_SETTINGS:

                correct = 0
                violations = 0

                print(f"Seed {seed} | {retrieval_mode} | {N_FEWSHOT}-shot")

                for idx_ex, (prefix, target) in enumerate(examples):

                    prefix_id_seq = prefix_id_seqs[idx_ex]

                    if N_FEWSHOT == 0:
                        retrieved = []

                    elif retrieval_mode == "random":
                        retrieved = random.sample(
                            train_candidates,
                            min(N_FEWSHOT, len(train_candidates))
                        )

                    else:
                        # Retrieve by sequence-level Levenshtein on ID token lists
                        sims = [
                            lev_similarity_seq(prefix_id_seq, cand_seq)
                            for cand_seq in train_candidate_id_seqs
                        ]
                        top_idx = np.argsort(sims)[-N_FEWSHOT:][::-1]
                        retrieved = [train_candidates[i] for i in top_idx]

                    shots = ""
                    for i, (p, nxt) in enumerate(retrieved):
                        shots += f"""
Example {i+1}
Sequence:
{" -> ".join(p)}
Next event:
{nxt}
"""

                    prompt = f"""
You are predicting the next event in a loan application process.

Possible events:
{chr(10).join(EVENTS_LIST)}

Here are similar examples:
{shots}

Sequence:
{" -> ".join(prefix)}

Predict the SINGLE most likely next event.

Rules:
- Answer with EXACTLY ONE event name.
- Do not explain.
"""

                    raw_pred = query_model(provider, model_id, prompt)
                    pred = clean_prediction(raw_pred)

                    is_correct = (pred == target)
                    is_valid = ((prefix[-1], pred) in valid_edges)

                    if is_correct:
                        correct += 1
                    if not is_valid:
                        violations += 1

                accuracy = correct / len(examples)
                gvr = violations / len(examples)

                all_metrics.append({
                    "model": model_name,
                    "retrieval": retrieval_mode,
                    "seed": seed,
                    "fewshot": N_FEWSHOT,
                    "accuracy": accuracy,
                    "gvr": gvr
                })


# Aggregate results across seeds

df = pd.DataFrame(all_metrics)

agg = (
    df.groupby(["model", "retrieval", "fewshot"])
    .agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        gvr_mean=("gvr", "mean"),
        gvr_std=("gvr", "std")
    )
    .reset_index()
)

print("\nAggregated Results:")
print(agg)

agg.to_csv("cross_model_results.csv", index=False)
print("\nSaved: cross_model_results.csv")
print("\nDone.")