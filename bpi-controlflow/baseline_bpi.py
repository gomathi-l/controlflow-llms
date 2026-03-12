# baselines_bpi.py
# Baselines for BPI next-event prediction
# 1) Markov model
# 2) LSTM
# 3) 1-NN sequence matching (Levenshtein)

import random
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import editdistance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Config

SEEDS = [0, 1, 2]
N_SAMPLES = 200
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1          # fraction of training traces held out for LSTM early stopping

EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 3
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Read data

with open("traces.pkl", "rb") as f:
    traces = pickle.load(f)

with open("clean_rules.pkl", "rb") as f:
    clean_rules = pickle.load(f)

traces = traces.tolist()
valid_edges = set(clean_rules.keys())

EVENTS_LIST = sorted(list({e for t in traces for e in t}))
NUM_EVENTS = len(EVENTS_LIST)

print("Loaded traces:", len(traces))
print("Unique events:", NUM_EVENTS)


# Event mappings

event_to_id = {e: i for i, e in enumerate(EVENTS_LIST)}
id_to_event = {i: e for e, i in event_to_id.items()}

event_to_lev_token = {e: f"E{str(i).zfill(3)}" for i, e in enumerate(EVENTS_LIST)}

vocab = [PAD_TOKEN, UNK_TOKEN] + EVENTS_LIST
token_to_idx = {tok: i for i, tok in enumerate(vocab)}

PAD_IDX = token_to_idx[PAD_TOKEN]
UNK_IDX = token_to_idx[UNK_TOKEN]


# Utility

def clean_prediction(pred):
    if pred is None:
        return None

    pred = str(pred).strip()

    if pred in event_to_id:
        return pred

    for e in EVENTS_LIST:
        if e in pred:
            return e

    return None


# Prefix sampling

def build_examples_from_traces(trace_list):

    all_prefixes = []

    for trace in trace_list:

        if len(trace) < 2:
            continue

        for i in range(1, len(trace)):
            all_prefixes.append((trace[:i], trace[i]))

    print("Total prefix prediction tasks:", len(all_prefixes))

    random.shuffle(all_prefixes)

    if N_SAMPLES is not None:
        return all_prefixes[:N_SAMPLES]

    return all_prefixes


# Metrics

def accuracy_from_predictions(examples, predictions):

    correct = 0

    for (_, target), pred in zip(examples, predictions):
        if pred == target:
            correct += 1

    return correct / len(examples)


def gvr_from_predictions(examples, predictions):

    violations = 0

    for (prefix, _), pred in zip(examples, predictions):
        if pred is None or (prefix[-1], pred) not in valid_edges:
            violations += 1

    return violations / len(examples)


# Markov model

class MarkovNextEventModel:

    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.global_counts = Counter()

    def fit(self, traces):

        for trace in traces:
            for i in range(len(trace) - 1):
                a = trace[i]
                b = trace[i + 1]
                self.transitions[a][b] += 1
                self.global_counts[b] += 1

    def predict_one(self, prefix):

        last = prefix[-1]

        if self.transitions[last]:
            pred = self.transitions[last].most_common(1)[0][0]
        elif self.global_counts:
            pred = self.global_counts.most_common(1)[0][0]
        else:
            pred = None

        return clean_prediction(pred)

    def predict(self, examples):
        return [self.predict_one(prefix) for prefix, _ in examples]

# 1-NN matching

def lev_similarity_seq(seq1, seq2):

    dist = editdistance.eval(seq1, seq2)
    max_len = max(len(seq1), len(seq2))

    if max_len == 0:
        return 1.0

    return 1 - dist / max_len


class OneNNSequenceMatcher:

    def fit(self, traces):

        self.train_candidates = []

        for t in traces:
            for i in range(1, len(t)):
                self.train_candidates.append((t[:i], t[i]))

        self.candidate_tokens = [
            tuple(event_to_lev_token[e] for e in p)
            for p, _ in self.train_candidates
        ]

    def predict_one(self, prefix):

        prefix_tokens = tuple(event_to_lev_token[e] for e in prefix)

        best_idx = max(
            range(len(self.candidate_tokens)),
            key=lambda i: lev_similarity_seq(prefix_tokens, self.candidate_tokens[i])
        )

        return clean_prediction(self.train_candidates[best_idx][1])

    def predict(self, examples):
        return [self.predict_one(p) for p, _ in examples]


# LSTM model

def encode_prefix(prefix):
    return [token_to_idx.get(e, UNK_IDX) for e in prefix]


def encode_target(event):
    return event_to_id[event]


class PrefixDataset(Dataset):

    def __init__(self, traces):

        self.samples = []

        for trace in traces:
            for i in range(1, len(trace)):
                target = trace[i]
                if target not in event_to_id:   # FIX: skip unseen targets to avoid KeyError
                    continue
                self.samples.append((encode_prefix(trace[:i]), encode_target(target)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):

    seqs, labels = zip(*batch)

    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    padded = [s + [PAD_IDX]*(max_len-len(s)) for s in seqs]

    return (
        torch.tensor(padded),
        torch.tensor(lengths),
        torch.tensor(labels)
    )


class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), EMBED_DIM, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_EVENTS)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


class LSTMBaseline:

    def __init__(self):
        self.model = LSTMModel().to(DEVICE)
        self.best_state = None

    def fit(self, traces):

        # FIX: VAL_RATIO is now actually used — split off a val set for early stopping
        val_size   = max(1, int(VAL_RATIO * len(traces)))
        val_traces = traces[:val_size]
        fit_traces = traces[val_size:]

        train_loader = DataLoader(PrefixDataset(fit_traces),
                                  batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
        val_loader   = DataLoader(PrefixDataset(val_traces),
                                  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

        optimizer        = torch.optim.Adam(self.model.parameters(), lr=LR)
        loss_fn          = nn.CrossEntropyLoss()
        best_val_loss    = float("inf")
        patience_counter = 0

        for epoch in range(EPOCHS):

            # train
            self.model.train()
            train_loss, n = 0.0, 0
            for x, lengths, y in train_loader:
                x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = loss_fn(self.model(x, lengths), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n += 1

            # validate
            self.model.eval()
            val_loss, nv = 0.0, 0
            with torch.no_grad():
                for x, lengths, y in val_loader:
                    x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
                    val_loss += loss_fn(self.model(x, lengths), y).item()
                    nv += 1

            avg_train = train_loss / max(n,  1)
            avg_val   = val_loss   / max(nv, 1)

            print(f"LSTM epoch {epoch+1}/{EPOCHS} | train={avg_train:.4f}  val={avg_val:.4f}")

            # early stopping
            if avg_val < best_val_loss:
                best_val_loss    = avg_val
                patience_counter = 0
                self.best_state  = {k: v.cpu().clone()
                                    for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def predict(self, examples):

        self.model.eval()

        seqs    = [encode_prefix(p) for p, _ in examples]
        lengths = [max(len(s), 1) for s in seqs]
        max_len = max(lengths)
        padded  = [s + [PAD_IDX]*(max_len-len(s)) for s in seqs]

        x = torch.tensor(padded).to(DEVICE)
        l = torch.tensor(lengths).to(DEVICE)

        with torch.no_grad():
            preds = torch.argmax(self.model(x, l), dim=1).cpu().numpy()

        return [clean_prediction(id_to_event[i]) for i in preds]


# Main evaluation

results = []

for seed in SEEDS:

    print("\nSeed", seed)

    set_seed(seed)

    shuffled = traces.copy()
    random.shuffle(shuffled)

    split = int(TRAIN_RATIO * len(shuffled))
    train = shuffled[:split]
    test  = shuffled[split:]

    examples = build_examples_from_traces(test)
    print("Evaluation examples:", len(examples))

    # Markov
    markov = MarkovNextEventModel()
    markov.fit(train)
    preds = markov.predict(examples)
    acc, gvr = accuracy_from_predictions(examples, preds), gvr_from_predictions(examples, preds)
    results.append({"model": "markov", "seed": seed, "accuracy": acc, "gvr": gvr})
    print("Markov  ACC", acc, "GVR", gvr)

    # 1-NN
    nn1 = OneNNSequenceMatcher()
    nn1.fit(train)
    preds = nn1.predict(examples)
    acc, gvr = accuracy_from_predictions(examples, preds), gvr_from_predictions(examples, preds)
    results.append({"model": "1nn", "seed": seed, "accuracy": acc, "gvr": gvr})
    print("1-NN    ACC", acc, "GVR", gvr)

    # LSTM — re-seed so weight init is reproducible
    set_seed(seed)
    lstm = LSTMBaseline()
    lstm.fit(train)
    preds = lstm.predict(examples)
    acc, gvr = accuracy_from_predictions(examples, preds), gvr_from_predictions(examples, preds)
    results.append({"model": "lstm", "seed": seed, "accuracy": acc, "gvr": gvr})
    print("LSTM    ACC", acc, "GVR", gvr)

df = pd.DataFrame(results)
df.to_csv("baseline_results.csv", index=False)

agg = (
    df.groupby("model")
    .agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy",  "std"),
        gvr_mean=("gvr",      "mean"),
        gvr_std=("gvr",       "std")
    )
    .reset_index()
)
agg.to_csv("all_baseline_results.csv", index=False)

print("\nAggregated results")
print(agg)
print("\nDONE with baseline_results.csv and all_baseline_results.csv")