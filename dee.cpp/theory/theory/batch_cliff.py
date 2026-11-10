"""Phase-7 correction: host-tier hit rate under batched decode (the b>1 capacity cliff).

What Phase 6 killed (PREDICTIONS_PHASE6.md Addendum 4, run p6m-l4-r5-b8-h16): with 8 rows
decoding together on a 16 GiB host tier, the measured host hit rate was 0 of 49,463 lookups,
while both the frozen and the Addendum-3 rows predicted 0.51. The registered host-hit term
(predecl.HOST_HIT_CURVE / PREDECL_FINITE_HIT) depends on host capacity only. Under the
Independent Reference Model (IRM) behind Che's approximation, that is correct, because batching
only rescales time. But the engine does not issue an IRM stream. Each decode step sweeps layers
0..L-1 in order and looks up the step's de-duplicated (layer, expert) set once. A record used at
step t can next be used at step t+k, and between the two lookups LRU sees (about) every record
of the k steps in between. That is the classic loop-larger-than-cache case: if one step's set
does not fit, LRU evicts every record before its next use, and the hit rate collapses to ~0
whatever the popularity skew.

Step-granular LRU model (this module):

    u_i(b)    = 1 - (1 - pi_i)^b          P(record i is in one step's set), batch b
    U_k(b)    = sum_i 1 - (1 - u_i)^k     E[distinct records over k consecutive steps]
    K(b, C)   = max { k >= 0 : U_k(b) <= C }   reuse gaps (in steps) that still hit
    H_step    = sum_i u_i (1 - (1 - u_i)^K) / sum_i u_i      steady-state hit share

K = 0 (no reuse gap fits) exactly when U_1(b) > C: one step's set is larger than the cache.
That is the cliff. It is a hard edge in the model, because lookups are de-duplicated per step,
so no record can hit inside its own step.

Corrected term (a cap, not a replacement):

    host_hit(b, C) = min( H_registered(C), H_step(b, C) )

Away from the cliff H_step is >= the registered short-window prior, so the registered numbers
are unchanged there, and the four r5 rows that passed still pass. Only near or below the cliff
does the cap bind.

Known limits, stated before any new run:
- pi_i comes from the sealed b=1 trace (one prompt, 2,364 of 11,776 records ever seen). It has
  no tail, and 8 different prompts touch more distinct experts than one does. U_1(8) is
  1,317 in the model, but r5 measured a mean of 1,521 (915-1,644) per decode step. So the
  model places the b=8 cliff at about 16.4 GiB, while r5's per-step sets put it at about 19 GiB.
  The cliff is real either way. Its exact position is uncertain by about 15%.
- H_step is steady-state. It says nothing about compulsory misses in a short window, which is
  why it is used only as a cap on the registered short-window prior.
- The VRAM tier in front of the host tier (3.5 GiB on L4) is ignored. It would move the cliff by
  at most ~270 records.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

GIB = float(1 << 30)


def step_inclusion(pi: np.ndarray, b: int) -> np.ndarray:
    """u_i(b): probability that record i is in one decode step's de-duplicated set."""
    pi = np.asarray(pi, dtype=float).reshape(-1)
    pi = pi[pi > 0]
    return 1.0 - (1.0 - pi) ** int(b)


def union_over_steps(u: np.ndarray, k: int) -> float:
    """U_k: expected distinct records touched over k consecutive steps."""
    if k <= 0:
        return 0.0
    return float(np.sum(1.0 - (1.0 - u) ** int(k)))


def reuse_horizon(u: np.ndarray, capacity: float, k_max: int = 100_000) -> int:
    """K: the largest step gap k with U_k <= capacity (0 means even one step does not fit)."""
    if union_over_steps(u, 1) > capacity:
        return 0
    if float(u.size) <= capacity:
        return k_max          # every record ever touched fits: no gap is too long
    lo, hi = 1, 2
    while hi < k_max and union_over_steps(u, hi) <= capacity:
        lo, hi = hi, hi * 2
    hi = min(hi, k_max)
    while lo + 1 < hi:        # invariant: U_lo <= C < U_hi
        mid = (lo + hi) // 2
        if union_over_steps(u, mid) <= capacity:
            lo = mid
        else:
            hi = mid
    return lo


def step_lru_hit(pi: np.ndarray, b: int, capacity: float) -> float:
    """H_step(b, C): steady-state host hit share under step-granular LRU."""
    u = step_inclusion(pi, b)
    if u.size == 0:
        return 0.0
    k = reuse_horizon(u, capacity)
    if k == 0:
        return 0.0
    return float(np.sum(u * (1.0 - (1.0 - u) ** k)) / np.sum(u))


def cliff_records(pi: np.ndarray, b: int) -> float:
    """Host capacity (records) below which the hit rate collapses: U_1(b)."""
    return union_over_steps(step_inclusion(pi, b), 1)


def corrected_host_hit(registered: float, pi: np.ndarray, b: int, capacity: float) -> float:
    """min(registered short-window prior, step-granular steady-state hit)."""
    return min(float(registered), step_lru_hit(pi, b, capacity))


# ---------------------------------------------------------------------------
# Driver: grid + the r5 check (no network, deterministic)
# ---------------------------------------------------------------------------
# Measured on p6m-l4-r5 (Addendum 4 and its follow-up; runs/p6m-l4-r5-*/scorecard.json).
R5_ROWS = [
    {"run": "p6m-l4-r5-b1-h64", "b": 1, "host_gib": 64, "measured_hit": 0.5745},
    {"run": "p6m-l4-r5-b1-h16", "b": 1, "host_gib": 16, "measured_hit": 0.5578},
    {"run": "p6m-l4-r5-b8-h64", "b": 8, "host_gib": 64, "measured_hit": 0.8351},
    {"run": "p6m-l4-r5-b8-h16", "b": 8, "host_gib": 16, "measured_hit": 0.0},
]
R5_B8_STEP_DISTINCT_MEAN = 1521   # measured, b8-h16 decode steps (range 915-1,644)


def r5_check(pi: np.ndarray, record_bytes: int, registered) -> List[Dict[str, object]]:
    """Registered vs corrected host-hit nominal on the four r5 rows.

    ``registered(host_gib) -> float`` is the registered short-window prior
    (predecl._finite_hit for dsv4_flash).
    """
    rows = []
    for r in R5_ROWS:
        cap = r["host_gib"] * GIB // record_bytes
        reg = float(registered(r["host_gib"]))
        u = step_inclusion(pi, r["b"])
        rows.append({
            **r,
            "capacity_records": int(cap),
            "step_set_U1": round(union_over_steps(u, 1), 1),
            "reuse_horizon_K": reuse_horizon(u, cap),
            "registered_nominal": reg,
            "step_lru_hit": round(step_lru_hit(pi, r["b"], cap), 4),
            "corrected_nominal": round(corrected_host_hit(reg, pi, r["b"], cap), 4),
        })
    return rows


def grid(pi: np.ndarray, record_bytes: int, registered,
         batches: Sequence[int] = (1, 2, 4, 8, 16),
         gibs: Sequence[float] = (4, 8, 12, 16, 20, 24, 32, 48, 64)) -> List[Dict[str, object]]:
    rows = []
    for b in batches:
        cliff = cliff_records(pi, b)
        for g in gibs:
            cap = g * GIB // record_bytes
            reg = float(registered(g))
            rows.append({"b": b, "host_gib": g, "capacity_records": int(cap),
                         "cliff_records": round(cliff, 1),
                         "cliff_gib": round(cliff * record_bytes / GIB, 2),
                         "registered_nominal": round(reg, 4),
                         "step_lru_hit": round(step_lru_hit(pi, b, cap), 4),
                         "corrected_nominal": round(corrected_host_hit(reg, pi, b, cap), 4)})
    return rows


def run(write: bool = True) -> Dict[str, object]:
    from .constants import D4_RECORD
    from .predecl import _finite_hit
    from .roofline import build_pop, load_geoms

    pop = build_pop(load_geoms()["dsv4_flash"])
    rec = int(D4_RECORD.value)
    reg = lambda g: _finite_hit(g, "dsv4_flash")  # noqa: E731
    check = r5_check(pop.pi, rec, reg)
    table = grid(pop.pi, rec, reg)
    b8_measured_cliff_gib = R5_B8_STEP_DISTINCT_MEAN * rec / GIB
    summary = {
        "popularity_source": pop.source,
        "records_with_nonzero_pi": int(np.count_nonzero(pop.pi)),
        "b8_cliff_records_model": round(cliff_records(pop.pi, 8), 1),
        "b8_cliff_gib_model": round(cliff_records(pop.pi, 8) * rec / GIB, 2),
        "b8_step_distinct_measured_mean": R5_B8_STEP_DISTINCT_MEAN,
        "b8_cliff_gib_measured_sets": round(b8_measured_cliff_gib, 2),
        "r5_check": check,
    }
    if write:
        from .util import write_csv, write_json
        write_csv("batch_cliff_grid.csv", table)
        write_json("batch_cliff_summary.json", summary)
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
