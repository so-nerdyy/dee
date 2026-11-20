"""Regenerate the paper's figures from numbers committed in dee.cpp/runs and
PREDICTIONS_PHASE6.md (values copied below with their evidence paths).
Run: python paper/figures/make_figures.py  (needs matplotlib)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9, "font.family": "serif"})

# r7 decode tok/s (aggregate at b=8), runs/p6m-r7-<gpu>-<row>/scorecard.json
GPUS = ["2xT4", "A10", "L4", "2xL4", "L40S", "RTX PRO\n6000"]
r7 = {
    "b=1, host 16 GiB": [0.696, 0.411, 0.766, 0.703, 1.552, 2.219],
    "b=1, host 64 GiB": [0.632, 0.756, 0.732, 0.672, 1.351, 1.948],
    "b=8, host 16 GiB": [None, 2.470, 1.204, 1.452, 2.475, 4.856],
    "b=8, host 64 GiB": [None, 3.230, 1.866, 1.826, 3.867, 6.274],
}
colors = ["#9ecae1", "#3182bd", "#fdae6b", "#e6550d"]
fig, ax = plt.subplots(figsize=(6.6, 2.6))
w = 0.2
for i, (lab, vals) in enumerate(r7.items()):
    xs = [g + (i - 1.5) * w for g in range(len(GPUS))]
    for x, v in zip(xs, vals):
        if v is None:
            ax.text(x, 0.08, "n/r", ha="center", fontsize=6, rotation=90)
    ax.bar([x for x, v in zip(xs, vals) if v is not None],
           [v for v in vals if v is not None], w, label=lab, color=colors[i],
           edgecolor="black", linewidth=0.3)
ax.set_xticks(range(len(GPUS)))
ax.set_xticklabels(GPUS)
ax.set_ylabel("decode tokens/s")
ax.legend(ncol=2, frameon=False, fontsize=7, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "r7_tps.pdf")

# Addendum-3 predicted nominal vs measured, testable r5+r7 tok/s rows
rows = [  # (label, measured, nominal, lo, hi)
    ("L4 b1 h16", 0.766, 1.722, 1.342, 2.996), ("L4 b1 h64", 0.732, 1.968, 1.543, 3.540),
    ("L4 b8 h16", 1.204, 4.375, 3.465, 8.117), ("L4 b8 h64", 1.866, 9.006, 7.586, 24.890),
    ("2xL4 b1 h16", 0.703, 1.624, 1.265, 2.765), ("2xL4 b1 h64", 0.672, 1.865, 1.460, 3.272),
    ("2xL4 b8 h16", 1.452, 4.620, 3.671, 8.649), ("2xL4 b8 h64", 1.826, 11.831, 10.370, 44.511),
    ("A10 b1 h16", 0.411, 0.733, 0.559, 1.128), ("A10 b1 h64", 0.756, 0.882, 0.675, 1.382),
    ("A10 b8 h16", 2.470, 2.068, 1.590, 3.299), ("A10 b8 h64", 3.230, 9.029, 7.603, 25.071),
]
fig, ax = plt.subplots(figsize=(3.3, 3.0))
for lab, m, n, lo, hi in rows:
    inside = lo <= m <= hi
    ax.errorbar(n, m, xerr=[[n - lo], [hi - n]], fmt="o" if inside else "x",
                color="#31a354" if inside else "#de2d26", ms=4, lw=0.6, capsize=1.5)
lim = [0.3, 50]
ax.plot(lim, lim, "k--", lw=0.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim); ax.set_ylim(0.3, 10)
ax.set_xlabel("pre-registered nominal tok/s (Addendum 3)")
ax.set_ylabel("measured tok/s (r7)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "pred_vs_meas.pdf")
print("wrote", OUT)


# ---------------------------------------------------------------------------
# Eviction policy vs host-tier size: simulator (theory/NEXT_USE_SIM.md, b=8,
# last 32 of 64 decode steps) with hardware points (Addenda 4 and 6, r7 L4).
sizes = [16, 24, 32, 64]
sim = {"LRU": [0.000, 0.460, 0.578, 0.824],
       "next-use": [0.418, 0.529, 0.614, 0.833],
       "Belady (offline bound)": [0.607, 0.716, 0.786, 0.931]}
style = {"LRU": ("#de2d26", "s"), "next-use": ("#3182bd", "o"),
         "Belady (offline bound)": ("#636363", "^")}
fig, ax = plt.subplots(figsize=(3.3, 2.5))
for k, v in sim.items():
    c, m = style[k]
    ax.plot(sizes, v, color=c, marker=m, ms=3.5, lw=1,
            ls="--" if "Belady" in k else "-", label=f"{k} (sim)")
ax.scatter([16, 64], [0.000, 0.835], marker="s", s=40, facecolor="none",
           edgecolor="#de2d26", lw=1.2, label="LRU, measured (r5)", zorder=5)
ax.scatter([16, 64], [0.439, 0.848], marker="o", s=40, facecolor="none",
           edgecolor="#3182bd", lw=1.2, label="next-use, measured (r6/r7)", zorder=5)
ax.set_xlabel("host tier (GiB)")
ax.set_ylabel("host hit rate, b=8")
ax.set_xticks(sizes)
ax.set_ylim(-0.03, 1.0)
ax.legend(frameon=False, fontsize=6, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "eviction_sizes.pdf")

# r6 rows (Addendum 6): tok/s and host hit per placement policy
pol = ["LRU", "next-use", "+prev-step\nhints", "+router\nhints"]
tps = [1.037, 1.348, 1.419, 1.269]
hit = [0.000, 0.439, 0.503, 0.543]
ret = [0.000, 0.439, 0.437, 0.436]  # retention-only host hit (hits minus hint hits)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
a1.bar(pol, tps, color=["#de2d26", "#3182bd", "#9ecae1", "#9ecae1"],
       edgecolor="black", lw=0.3)
for i, v in enumerate(tps):
    a1.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
a1.set_ylabel("decode tok/s (aggregate)")
a1.set_ylim(0, 1.7)
a1.set_title("(a) throughput, 1xL4, b=8, host 16 GiB", fontsize=8)
x = range(len(pol))
a2.bar([i - 0.18 for i in x], ret, 0.36, color="#3182bd", edgecolor="black",
       lw=0.3, label="kept by eviction")
a2.bar([i + 0.18 for i in x], hit, 0.36, color="#fdae6b", edgecolor="black",
       lw=0.3, label="total incl. hint hits")
a2.set_xticks(list(x)); a2.set_xticklabels(pol)
a2.set_ylabel("host hit rate")
a2.set_ylim(0, 1.2)
a2.axhline(0.87, color="#e6550d", ls=":", lw=1)
a2.text(3.45, 0.90, "A5 predicted 0.87", fontsize=6.5, ha="right", color="#e6550d")
a2.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)
a2.set_title("(b) where the host hits came from", fontsize=8)
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "r6_policies.pdf")

# L4 b=8 host-16 throughput across runs (Addenda 4, 6; r7 scorecards)
prog = [("r5\nLRU", 0.591), ("r6\nLRU", 1.037), ("r6\nnext-use", 1.348),
        ("r7\nnext-use\n(repeat)", 1.230), ("r7\nnext-use", 1.204)]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ax.bar([p[0] for p in prog], [p[1] for p in prog],
       color=["#fcbba1", "#de2d26", "#3182bd", "#6baed6", "#6baed6"],
       edgecolor="black", lw=0.3)
for i, (_, v) in enumerate(prog):
    ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
ax.annotate("host-path\nfixes", xy=(0.5, 0.85), xytext=(-0.45, 1.25),
            fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_ylabel("decode tok/s")
ax.set_title("1xL4, b=8, host 16 GiB", fontsize=8)
ax.set_ylim(0, 1.6)
ax.tick_params(axis="x", labelsize=6.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "l4_progress.pdf")

# Host tier 16 -> 64 GiB per GPU at b=8 (r7 scorecards): cold reads per token
gp = ["A10", "L4", "2xL4", "L40S", "RTX PRO\n6000"]
cold16 = [107.7, 108.4, 109.3, 109.8, 106.3]
cold64 = [27.9, 29.4, 29.8, 29.3, 27.9]
gain = [3.230 / 2.470, 1.866 / 1.204, 1.826 / 1.452, 3.867 / 2.475, 6.274 / 4.856]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.2))
xs = range(len(gp))
a1.bar([i - 0.18 for i in xs], cold16, 0.36, color="#fdae6b", edgecolor="black",
       lw=0.3, label="host 16 GiB")
a1.bar([i + 0.18 for i in xs], cold64, 0.36, color="#e6550d", edgecolor="black",
       lw=0.3, label="host 64 GiB")
a1.set_xticks(list(xs)); a1.set_xticklabels(gp, fontsize=7)
a1.set_ylabel("SSD reads per token")
a1.set_ylim(0, 140)
a1.legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
a1.set_title("(a) cold reads, b=8", fontsize=8)
a2.bar(list(xs), gain, color="#756bb1", edgecolor="black", lw=0.3)
for i, v in enumerate(gain):
    a2.text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=7)
a2.set_xticks(list(xs)); a2.set_xticklabels(gp, fontsize=7)
a2.set_ylabel("tok/s, 64 GiB / 16 GiB")
a2.set_ylim(1.0, 1.75)
a2.set_title("(b) speedup from the larger host tier", fontsize=8)
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "host_tier.pdf")

# Post-hoc step-time model (PR #26 theory/r7/tps_fit.json)
fit = {  # gpu: (t_fixed s, c_read ms, cold/step h16, cold/step h64, mem BW GB/s)
    "L4": (3.409, 3.733, 867.5, 235.6, 300),
    "2xL4": (3.958, 1.776, 874.5, 238.2, None),
    "A10": (2.211, 1.194, 861.4, 223.3, 600),
    "L40S": (1.645, 1.807, 878.3, 234.6, 864),
    "RTX PRO 6000": (1.143, 0.593, 850.3, 222.8, 1792),
}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})
labels, fixed, rd = [], [], []
for g, (tf, cr, c16, c64, _) in fit.items():
    for h, c in (("16", c16), ("64", c64)):
        labels.append(f"{g}\n{h} GiB" if h == "16" else f"\n{h}")
        fixed.append(tf); rd.append(c * cr / 1000)
xs = range(len(labels))
a1.bar(xs, fixed, color="#9ecae1", edgecolor="black", lw=0.3, label="fixed per step")
a1.bar(xs, rd, bottom=fixed, color="#e6550d", edgecolor="black", lw=0.3,
       label="cold reads x cost per read")
a1.set_xticks(list(xs)); a1.set_xticklabels(labels, fontsize=5.5)
a1.set_ylabel("seconds per decode step, b=8")
a1.legend(frameon=False, fontsize=6.5)
a1.set_title("(a) fitted step time", fontsize=8)
bw = [(v[4], v[0], g) for g, v in fit.items() if v[4]]
for b_, t_, g in bw:
    a2.scatter(b_, t_, color="#3182bd", s=14, zorder=3)
    a2.annotate(g, (b_, t_), fontsize=6, xytext=(3, 3), textcoords="offset points")
xx = [250, 2000]
a2.plot(xx, [114 * v ** -0.62 for v in xx], "k--", lw=0.7)
a2.text(260, 1.0, r"$t \approx 114\,\mathrm{BW}^{-0.62}$", fontsize=7)
a2.set_xscale("log"); a2.set_yscale("log")
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
a2.xaxis.set_major_locator(FixedLocator([300, 600, 1000, 1800])); a2.xaxis.set_minor_locator(NullLocator())
a2.yaxis.set_major_locator(FixedLocator([1, 2, 3])); a2.yaxis.set_minor_locator(NullLocator())
a2.xaxis.set_major_formatter(ScalarFormatter()); a2.yaxis.set_major_formatter(ScalarFormatter())
a2.set_xlabel("GPU memory bandwidth (GB/s)")
a2.set_ylabel("fixed time per step (s)")
a2.set_title("(b) fixed time vs bandwidth", fontsize=8)
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "tps_model.pdf")

# Cross-GPU token agreement (harness comparisons reported by the r7 runs)
import numpy as np
from matplotlib.colors import ListedColormap
G1 = ["T4", "A10", "L4", "2xL4", "L40S", "RTX"]
# 0 = not compared, 1 = identical tokens, 2 = different tokens
grp1 = {"T4": 0, "A10": 1, "L4": 2, "2xL4": 2, "L40S": 2, "RTX": 2}
M1 = np.array([[1 if grp1[a] == grp1[b] else 2 for b in G1] for a in G1])
G8 = ["A10", "L4", "2xL4", "L40S", "RTX"]
M8 = np.zeros((5, 5), int)
for i in range(5):
    M8[i, i] = 1
pairs = {("L4", "2xL4"): 1, ("L4", "L40S"): 2, ("L4", "RTX"): 2, ("L4", "A10"): 2}
for (a, b), v in pairs.items():
    i, j = G8.index(a), G8.index(b)
    M8[i, j] = M8[j, i] = v
cm = ListedColormap(["#f0f0f0", "#74c476", "#fb6a4a"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.7))
for a, M, G, t in ((a1, M1, G1, "(a) b=1, sealed prompt"),
                   (a2, M8, G8, "(b) b=8, 8-prompt cohort")):
    a.imshow(M, cmap=cm, vmin=0, vmax=2)
    a.set_xticks(range(len(G))); a.set_xticklabels(G, fontsize=7, rotation=45)
    a.set_yticks(range(len(G))); a.set_yticklabels(G, fontsize=7)
    for i in range(len(G)):
        for j in range(len(G)):
            a.text(j, i, {0: "?", 1: "=", 2: "x"}[M[i, j]], ha="center",
                   va="center", fontsize=8)
    a.set_title(t, fontsize=8)
fig.text(0.5, 0.01, "= identical token ids    x different    ? not compared",
         ha="center", fontsize=7)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "cross_gpu.pdf")

# Cost per 1k tokens at b=8, host 64 GiB, all-in list rate (r7_environment.json)
rate = {"A10": 0.00062392, "L4": 0.00053992, "2xL4": 0.00076192,
        "L40S": 0.00093096, "RTX PRO\n6000": 0.00137304}
t64 = {"A10": 3.230, "L4": 1.866, "2xL4": 1.826, "L40S": 3.867, "RTX PRO\n6000": 6.274}
fig, ax = plt.subplots(figsize=(3.3, 2.2))
cost = [1000 * rate[g] / t64[g] for g in rate]
ax.bar(list(rate), cost, color="#31a354", edgecolor="black", lw=0.3)
for i, v in enumerate(cost):
    ax.text(i, v + 0.005, f"${v:.2f}", ha="center", fontsize=7)
ax.set_ylabel("USD per 1,000 tokens")
ax.tick_params(axis="x", labelsize=6.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "cost.pdf")
print("wrote extra figures")
