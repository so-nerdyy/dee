#!/usr/bin/env python3
"""Generate plots from results/ (optional deliverable)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LABEL = "v65-terminal-fetch-20260903T0202Z"

t = json.load(open(RES / f"tournament_{LABEL}.json"))
p = json.load(open(RES / f"predictor_{LABEL}.json"))
f = json.load(open(RES / f"prefetch_{LABEL}.json"))

# ---------------- plot 1: misses vs capacity ----------------
pols = ["lru", "arc", "recency_freq", "least_stale", "keepvalue", "belady"]
caps = sorted({r["capacity_experts_per_layer"] for r in t})
fig, ax = plt.subplots(figsize=(7, 4.5))
for pol in pols:
    ys = [next(r["misses"] for r in t
               if r["policy"] == pol
               and r["capacity_experts_per_layer"] == c
               and r["tier2_experts_per_layer"] == 0) for c in caps]
    ax.plot(caps, ys, marker="o", ms=3,
            label="Belady (offline oracle)" if pol == "belady" else pol)
ax.set_xlabel("VRAM capacity (experts per layer)")
ax.set_ylabel("demand misses (43 layers x 16 forwards)")
ax.set_title("DSV4 expert cache: policy tournament (sealed v65 trace)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RES / "misses_vs_capacity.png", dpi=150)

# ---------------- plot 2: predictor recall ----------------
fig, ax = plt.subplots(figsize=(6, 4))
ks = [6, 8, 12, 16]
for name in ("freq_expert", "persist_layer", "cross_union", "combo"):
    ys = [p["results"][name][f"recall@{k}"] for k in ks]
    ax.plot(ks, ys, marker="o", label=name)
ax.set_xlabel("candidate set size k")
ax.set_ylabel("recall of true layer bundle")
ax.set_title("Causal cross-layer predictor recall (DSV4)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RES / "predictor_recall.png", dpi=150)

# ---------------- plot 3: prefetch effect ----------------
fig, ax = plt.subplots(figsize=(6, 4))
modes = defaultdict(dict)
for r in f["rows"]:
    if r["max_prefetch_per_bundle"] == 6:
        modes[r["capacity_experts_per_layer"]][r["mode"]] = r["misses"]
caps_f = sorted(modes)
x = range(len(caps_f))
w = 0.35
off = [modes[c].get("off", 0) for c in caps_f]
rep = [modes[c].get("replace", 0) for c in caps_f]
ax.bar([i - w / 2 for i in x], off, w, label="demand-only")
ax.bar([i + w / 2 for i in x], rep, w, label="prefetch+replace (max_pf=6)")
ax.set_xticks(list(x))
ax.set_xticklabels([f"cap-{c}" for c in caps_f])
ax.set_ylabel("demand misses")
ax.set_title("Prefetch effect by capacity (DSV4, causal predictor)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(RES / "prefetch_effect.png", dpi=150)

print("wrote plots to", RES)
