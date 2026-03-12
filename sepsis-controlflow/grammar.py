# SEPSIS EVENT LOG
# CONTROL-FLOW GRAMMAR EXTRACTION PIPELINE

import pm4py
import pandas as pd
from collections import Counter
import pickle

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_vis

print("Loading Sepsis XES log:")

log = pm4py.read_xes("data/Sepsis Cases - Event Log.xes")
df = pm4py.convert_to_dataframe(log)

print("Dataset shape:", df.shape)

# Convert timestamps
df["time:timestamp"] = pd.to_datetime(
    df["time:timestamp"].astype(str),
    errors="coerce"
)

# Sort chronologically
df = df.sort_values(["case:concept:name", "time:timestamp"])

print("\nColumns:")
print(df.columns)

print("\nActivity counts:")
print(df["concept:name"].value_counts())

events_per_case = df.groupby("case:concept:name").size()

print("\nEvents per case summary:")
print(events_per_case.describe())

# Build traces

print("\nCreating traces:")

traces = (
    df.groupby("case:concept:name")["concept:name"]
    .apply(list)
)

print(traces.head())

# Extract transition grammar

print("\nExtracting transition grammar:")

rules = Counter()

for trace in traces:
    for i in range(len(trace)-1):
        rules[(trace[i], trace[i+1])] += 1

print("\nTop transitions:")
print(rules.most_common(20))

# Clean and remove noise

MIN_COUNT = 0   # lower than BPI (smaller dataset)

clean_rules = {
    pair: c
    for pair, c in rules.items()
    if c >= MIN_COUNT
}

print("\nClean grammar rules:")
for k,v in clean_rules.items():
    print(k, "->", v)

# Hierarchical Grammar

print("\nDiscovering hierarchical grammar (process tree):")

df_pm = df.dropna(subset=[
    "case:concept:name",
    "concept:name",
    "time:timestamp"
])

event_log = log_converter.apply(
    df_pm,
    variant=log_converter.Variants.TO_EVENT_LOG
)

process_tree = inductive_miner.apply(event_log)

print("\nProcess Tree (Hierarchical Grammar):")
print(process_tree)

# Visualize

print("\nSaving process tree visualization:")

gviz = pt_vis.apply(process_tree)
pt_vis.save(gviz, "process_tree.png")

print("Saved: process_tree.png")

# Save grammars

rules_df = pd.DataFrame([
    (a, b, c)
    for (a,b), c in rules.items()
], columns=["from", "to", "count"])

rules_df.to_csv("transition_grammar.csv", index=False)

with open("traces.pkl", "wb") as f:
    pickle.dump(traces, f)

with open("clean_rules.pkl", "wb") as f:
    pickle.dump(clean_rules, f)

print("\nSaved traces.pkl + clean_rules.pkl")


# Valid edges for GVR

valid_edges = set(clean_rules.keys())

print("\nValid edges used for GVR:\n")

for src, dst in sorted(valid_edges):
    print(f"{src} -> {dst}")

print(f"\nTotal valid edges: {len(valid_edges)}")

# Trace Length Distribution

import matplotlib.pyplot as plt
import numpy as np

# Scatter Plot

print("\nScatter plot of trace lengths:")

case_ids = events_per_case.index
trace_lengths = events_per_case.values

plt.figure(figsize=(8,6))
plt.scatter(range(len(trace_lengths)), trace_lengths)

plt.xlabel("Case Index")
plt.ylabel("Trace Length (# Events)")
plt.title("Scatter Plot of Trace Length per Case")

plt.tight_layout()
plt.savefig("trace_length_scatter.png")
plt.show()

print("Saved: trace_length_scatter.png")

print("\nRobust length statistics:")
print("Median:", events_per_case.median())
print("Q1:", events_per_case.quantile(0.25))
print("Q3:", events_per_case.quantile(0.75))
print("90th percentile:", events_per_case.quantile(0.90))
print("95th percentile:", events_per_case.quantile(0.95))
print("Max:", events_per_case.max())