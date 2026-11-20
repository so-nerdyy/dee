"""Evaluation metrics for the predictor lab.

Per arm x trace x task x m:

  * recall     = |announced n target| / |target|          (target records covered)
  * precision  = |announced n target| / |announced|       (announced records that land)
  * announced records per source call
  * byte-value accounting: announced records are STAGED as prefetch hints at
    the source call.  A hint is VALUABLE only if it converts a
    would-have-been COLD load at the target call into a hit.  The baseline
    hierarchy is replayed WITHOUT hints (causal LRU at a stated capacity);
    for each landed hint we ask the baseline: would this record have been a
    cold load at this call?

        saved_cold_records = landed hints where the baseline said COLD
        stale_records      = landed hints where the baseline said RESIDENT
        wasted_records     = announced records that never land

  * calibration: equal-count reliability bins over (score, lands) pairs + ECE.

Bootstrap CIs resample whole tokens (T2) / forward steps (T1) -- clusters --
so within-token layer correlation is respected.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

Rec = Tuple[int, int]

# Reference capacities for the would-have-been-cold baseline (combined
# device+host residency expressed in slots):
BASELINE_SCENARIOS = (
    ("always_cold", 0),
    ("vram281_host682", 281 + 682),
    ("vram1024_host4096", 1024 + 4096),
)


class LRUReplay:
    """Causal LRU demand replay over (layer, expert) records."""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache: "OrderedDict[Rec, None]" = OrderedDict()

    def access(self, rec: Rec) -> bool:
        """Return True iff this access is a cold load under this replay."""
        if rec in self.cache:
            self.cache.move_to_end(rec)
            return False
        if self.cap > 0 and len(self.cache) >= self.cap:
            self.cache.popitem(last=False)
        self.cache[rec] = None
        return True


def evaluate_arm(examples: Sequence,
                 ranked: Dict[int, Dict[int, List[Tuple[Rec, float]]]],
                 first_touch: Set[Rec], n_boot: int = 400, seed: int = 20260915):
    """Full evaluation of one arm on one example list.

    ``ranked``: example index -> {m -> ranked [(rec, score)] announced list at
    that hint budget m}.  Under the per-call semantics the list is the top-m
    of one global ranking; under the per-source-expert semantics it is the
    union of each source expert's top-m (up to |S_src| x m records).
    Returns (metric_rows, byte_rows, per_call) for the m sweep.
    """
    per_call = []
    for idx, ex in enumerate(examples):
        by_m = ranked.get(idx, {})
        tgt = set(ex.target)
        per_call.append({
            "cluster": ex.cluster,
            "ann_by_m": by_m,
            "n_tgt": len(tgt),
            "tgt": tgt,
        })

    rows = []
    for m in (1, 2, 4, 8, 16, 32):
        n_ann = n_tgt = n_hit = 0
        cluster_stat: Dict[int, List[int]] = {}
        for c in per_call:
            ann = [r for r, _s in c["ann_by_m"].get(m, [])]
            h = len(set(ann) & c["tgt"])
            n_ann += len(ann)
            n_tgt += c["n_tgt"]
            n_hit += h
            cs = cluster_stat.setdefault(c["cluster"], [0, 0, 0])
            cs[0] += len(ann)
            cs[1] += c["n_tgt"]
            cs[2] += h
            cs = cluster_stat.setdefault(c["cluster"], [0, 0, 0])
            cs[0] += len(ann)
            cs[1] += c["n_tgt"]
            cs[2] += h
        prec = n_hit / n_ann if n_ann else 0.0
        rec = n_hit / n_tgt if n_tgt else 0.0
        row = {
            "m": m,
            "n_calls": len(per_call),
            "n_clusters": len(cluster_stat),
            "n_announced_total": n_ann,
            "n_target_total": n_tgt,
            "n_hit_total": n_hit,
            "recall": rec,
            "precision": prec,
            "announced_per_call": n_ann / max(1, len(per_call)),
        }
        if n_boot > 0 and cluster_stat:
            ci_r = _cluster_bootstrap(cluster_stat, 2, 1, n_boot, seed)
            ci_p = _cluster_bootstrap(cluster_stat, 2, 0, n_boot, seed + 1)
            row.update({"recall_lo": ci_r[0], "recall_hi": ci_r[1],
                        "precision_lo": ci_p[0], "precision_hi": ci_p[1]})
        else:
            row.update({"recall_lo": float("nan"), "recall_hi": float("nan"),
                        "precision_lo": float("nan"), "precision_hi": float("nan")})
        rows.append(row)

    byte_rows = []
    for label, cap in BASELINE_SCENARIOS:
        for m in (4, 8, 16):
            byte_rows.append(_byte_row(label, cap, m, per_call))
    return rows, byte_rows, per_call


def _byte_row(label: str, cap: int, m: int, per_call):
    """Byte accounting at hint budget m under one baseline residency scenario."""
    baseline = LRUReplay(cap)
    saved = stale = wasted = 0
    n_ann = n_tgt = n_hit = 0
    for c in per_call:
        ann = [r for r, _s in c["ann_by_m"].get(m, [])]
        ann_set = set(ann)
        # baseline demand stream at the TARGET call (no hints staged)
        cold_at: Set[Rec] = set()
        for r in sorted(c["tgt"]):
            if baseline.access(r):
                cold_at.add(r)
        for r in ann:
            if r in c["tgt"]:
                n_hit += 1
                if r in cold_at:
                    saved += 1
                else:
                    stale += 1
            else:
                wasted += 1
        n_ann += len(ann)
        n_tgt += c["n_tgt"]
    return {
        "scenario": label, "baseline_capacity_slots": cap, "m": m,
        "n_calls": len(per_call),
        "announced_total": n_ann,
        "hits_total": n_hit,
        "saved_cold_records": saved,
        "stale_records": stale,
        "wasted_records": wasted,
        "precision": n_hit / n_ann if n_ann else 0.0,
        "recall": n_hit / n_tgt if n_tgt else 0.0,
    }


def _cluster_bootstrap(cluster_stat: Dict[int, List[int]], hit_i: int, den_i: int,
                       n_boot: int, seed: int) -> Tuple[float, float]:
    vals = np.asarray(list(cluster_stat.values()), dtype=float)   # (ann, tgt, hit)
    if vals.shape[0] == 0:
        return (float("nan"), float("nan"))
    g = np.random.default_rng(seed)
    n = vals.shape[0]
    idx = g.integers(0, n, size=(n_boot, n))
    num = vals[:, hit_i][idx].sum(axis=1)
    den = vals[:, den_i][idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    lo, hi = np.quantile(stat, [0.025, 0.975])
    return (float(lo), float(hi))


def calibration_bins(per_call, m: int, n_bins: int = 8):
    """Equal-count reliability bins over per-candidate (score, lands) pairs at
    budget m.  Returns (rows, ECE, n_pairs)."""
    pairs = []
    for c in per_call:
        tgt = c["tgt"]
        for r, s in c["ann_by_m"].get(m, []):
            pairs.append((float(s), 1 if r in tgt else 0))
    if not pairs:
        return [], float("nan"), 0
    pairs.sort(key=lambda t: t[0])
    n = len(pairs)
    bounds = np.linspace(0, n, n_bins + 1).astype(int)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        seg = pairs[bounds[b]:bounds[b + 1]]
        if not seg:
            continue
        conf = float(np.mean([s for s, _y in seg]))
        freq = float(np.mean([y for _s, y in seg]))
        rows.append({"bin": b, "lo": seg[0][0], "hi": seg[-1][0],
                     "n": len(seg), "mean_predicted": conf, "observed_freq": freq,
                     "gap": freq - conf})
        ece += len(seg) / n * abs(freq - conf)
    return rows, float(ece), n
