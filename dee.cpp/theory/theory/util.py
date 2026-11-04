"""Shared utilities: deterministic RNG, CSV/JSON writers, bootstrap, plotting."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import paths

RNG_SEED = 20260915


def rng(seed: int = RNG_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def write_csv(name: str, rows: Sequence[Dict], header: Optional[Sequence[str]] = None) -> Path:
    paths.ensure_outputs()
    out = paths.DATA_DIR / name
    if not rows and not header:
        out.write_text("", encoding="utf-8")
        return out
    if header is None:
        # union of all row keys, first-seen order (heterogeneous rows keep
        # their family-specific columns)
        header = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    header.append(k)
    header = list(header)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def write_json(name: str, obj) -> Path:
    paths.ensure_outputs()
    out = paths.DATA_DIR / name
    out.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float), encoding="utf-8")
    return out


def write_md(name: str, text: str) -> Path:
    paths.ensure_outputs()
    out = paths.DATA_DIR / name
    out.write_text(text, encoding="utf-8")
    return out


def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = RNG_SEED) -> Dict[str, float]:
    """Percentile bootstrap CI over an i.i.d. sample of ``values``."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    g = rng(seed)
    idx = g.integers(0, x.size, size=(n_boot, x.size))
    boot = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(stat(x)), "lo": float(lo), "hi": float(hi), "n": int(x.size)}


def figure(name: str, figsize=(7.2, 4.6)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths.ensure_outputs()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def savefig(fig, name: str) -> Path:
    import matplotlib.pyplot as plt

    paths.ensure_outputs()
    out = paths.FIGS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def dlog(*a) -> None:
    print("[theory]", *a, flush=True)
