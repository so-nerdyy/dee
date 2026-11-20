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
