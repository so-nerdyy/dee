"""Phase-7 follow-up: where the b=8 host-hit cliff sits, from the real 8-prompt route journals.

`batch_cliff.py` (PR #18) explains the r5 b8-h16 kill with a step-granular LRU model, but its
popularity comes from the sealed b=1 trace: one prompt, 16 tokens, 2,364 of 11,776 records ever
seen. That popularity has no tail, so it puts the b=8 cliff at 16.4 GiB while r5's measured
per-step sets point to about 19 GiB.

The r5 b=8 rows ran the cohort in `aws/phase6/prompts_phase4_8.json`. Those are the same eight
prompts as the Phase-4 2xT4 cache campaign (same prompt sha256 for q0..q7; q7 repeats q0), whose
full 128-token route journals are committed under
`benchmark_reports/deepseek-v4-flash-0731-t4/gpu2-phase4-cache-hierarchy/evidence/a1_asis/`.
So the b=8 decode stream r5 saw can be rebuilt in-repo, with no Modal access:

    step_set(s) = sorted union over prompts q of {(layer, e) : e routed for q at forward step s}

sweeping layers 0..42 in order, de-duplicated per step (the engine looks each (layer, expert)
up once per step). This module

1. rebuilds that stream and checks it against what r5 measured (per-step set sizes, window
   lookups, host hit at 16 and 64 GiB),
2. replays the engine's byte-budget LRU host tier over it at any capacity (exact trace replay,
   all records the same 12.75 MiB size), and
3. rebuilds expert popularity from all eight prompts and re-evaluates the closed-form cliff from
   `batch_cliff.py` with it, so the model and the replay can be compared.

Inputs are read-only. Deterministic, no RNG, no network.

Known limits (stated before any new run):
- The journals are b=1 runs on 2xT4. r5 ran the cohort batched on 1xL4, and Addendum 4 found the
  L4 flips one near-tie expert at layer 6 on the b=1 prompt. The rebuilt stream reproduces r5's
  per-step set sizes to within about 1% and its window lookups to within 2.5%, so routing
  differences are small but not zero.
- The VRAM tier in front of the host tier (3,584 MiB = 268 records on L4) is not modelled. It
  absorbs some lookups before they reach the host tier, which may be part of the 2.5% gap.
- The rebuilt popularity is in-sample for this cohort. It is the right input for predicting a
  rerun of the same eight prompts, and says nothing about other prompts.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import paths
from .batch_cliff import step_lru_hit, union_over_steps, step_inclusion

GIB = float(1 << 30)
N_LAYERS = 43
N_EXPERTS = 256

COHORT_DIR = paths.DEE_ROOT / (
    "benchmark_reports/deepseek-v4-flash-0731-t4/gpu2-phase4-cache-hierarchy/evidence/a1_asis")
N_PROMPTS = 8
DUPLICATE_PROMPTS = {7: 0}     # q7 is a verbatim repeat of q0 (same prompt sha256)

# r5 b=8 rows: 65 tokens = prefill (forward step 0) + 64 decode steps. score.py's window is the
# checkpoint delta from step 32 to step 64, i.e. the lookups of forward steps 33..64.
R5_LAST_STEP = 64
R5_WINDOW = (33, 64)           # inclusive forward steps

# Measured on p6m-l4-r5 (PREDICTIONS_PHASE6.md Addendum 4; runs/p6m-l4-r5-b8-*/scorecard.json).
R5_MEASURED = {
    "step_set_mean": 1521, "step_set_min": 915, "step_set_max": 1644, "prefill_set": 6147,
    "window_lookups": 49463,
    "rows": [{"run": "p6m-l4-r5-b8-h16", "host_gib": 16, "measured_hit": 0.0},
             {"run": "p6m-l4-r5-b8-h64", "host_gib": 64, "measured_hit": 0.8351}],
}

Rec = Tuple[int, int]
Journal = Dict[int, Dict[int, frozenset]]    # forward_step -> layer -> expert ids


def load_journal(path: Path) -> Journal:
    out: Journal = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            es = frozenset(int(e) for row in r["expert_ids_rank_order"] for e in row)
            out.setdefault(int(r["forward_step"]), {})[int(r["layer"])] = es
    return out


def load_cohort(cohort_dir: Path = COHORT_DIR, n: int = N_PROMPTS) -> List[Journal]:
    return [load_journal(cohort_dir / f"routed_experts-q{q}.jsonl") for q in range(n)]


def step_set(cohort: Sequence[Journal], step: int) -> List[Rec]:
    """The step's de-duplicated (layer, expert) lookups, in the engine's layer order."""
    out: List[Rec] = []
    for layer in range(N_LAYERS):
        u = set()
        for j in cohort:
            u |= j[step].get(layer, frozenset())
        out.extend((layer, e) for e in sorted(u))
    return out


def stream(cohort: Sequence[Journal], last_step: int = R5_LAST_STEP) -> List[List[Rec]]:
    return [step_set(cohort, s) for s in range(last_step + 1)]


def replay_lru(steps: Sequence[Sequence[Rec]], capacity: int,
               window: Tuple[int, int] = R5_WINDOW) -> Dict[str, float]:
    """Exact LRU replay of a host tier holding ``capacity`` equal-size records.

    Mirrors HostPackCache: a hit refreshes the entry, a miss inserts it at the front and evicts
    from the back. Returns hits / lookups counted over the inclusive ``window`` of steps.
    """
    cache: "OrderedDict[Rec, None]" = OrderedDict()
    hits = looks = 0
    lo, hi = window
    for s, recs in enumerate(steps):
        inw = lo <= s <= hi
        for k in recs:
            if k in cache:
                cache.move_to_end(k)
                hits += inw
            else:
                cache[k] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
            looks += inw
    return {"hits": hits, "lookups": looks, "hit": hits / looks if looks else 0.0}


def pooled_popularity(cohort: Sequence[Journal], first: int = 1, last: int = 127) -> np.ndarray:
    """pi_i: per-layer-call inclusion probability over decode steps of the DISTINCT prompts."""
    cnt = np.zeros((N_LAYERS, N_EXPERTS))
    calls = 0
    for q, j in enumerate(cohort):
        if q in DUPLICATE_PROMPTS:
            continue
        for s in range(first, last + 1):
            for layer in range(N_LAYERS):
                for e in j[s].get(layer, ()):
                    cnt[layer, e] += 1
            calls += 1
    return (cnt / calls).reshape(-1)


def capacity_records(host_gib: float, record_bytes: int) -> int:
    return int(host_gib * GIB // record_bytes)


def replay_band(steps, host_gib: float, record_bytes: int, rel: float = 0.03) -> Tuple[float, float]:
    """Replay hit at capacity x (1 - rel) and x (1 + rel): the routing-difference band."""
    c = capacity_records(host_gib, record_bytes)
    lo = replay_lru(steps, int(c * (1 - rel)))["hit"]
    hi = replay_lru(steps, int(c * (1 + rel)))["hit"]
    return min(lo, hi), max(lo, hi)


def edge_gib(steps, record_bytes: int, lo_gib: float = 14.0, hi_gib: float = 40.0,
             thresh: float = 0.01) -> float:
    """Smallest host size (0.1 GiB grid) whose window replay hit exceeds ``thresh``."""
    g = lo_gib
    while g <= hi_gib:
        if replay_lru(steps, capacity_records(g, record_bytes))["hit"] > thresh:
            return round(g, 1)
        g = round(g + 0.1, 1)
    return float("nan")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
SWEEP_GIB = (16, 17, 18, 18.5, 19, 19.5, 20, 20.5, 21, 22, 24, 26, 28, 30, 32, 40, 48, 64)

# Addendum 5 (registered 2026-11-11, before run p6m-l4-r6): the rows the next run scores.
ADD5_HOSTS = (18, 24, 32)
ADD5_REGISTERED = "2026-11-11"
HIT_PAD, HIT_KILL_PAD = 0.02, 0.04     # absolute pad on the +-3% / +-5% capacity bands
COLD_PAD, COLD_KILL_PAD = 0.10, 0.20   # relative pad (r5 h64 cold read 9% under the replay)


def add5_rows(steps, record_bytes: int, hosts: Sequence[float] = ADD5_HOSTS) -> List[Dict]:
    """Registered host-hit and cold rec/tok rows for b=8 on 1xL4 at each host size.

    nominal = exact replay at the host's capacity. range = the replay over capacity x [0.97, 1.03]
    padded by HIT_PAD; kill = the replay over x [0.95, 1.05] padded by HIT_KILL_PAD. Cold rec/tok
    = window misses / (8 x 32) on the same capacities, padded relatively.
    """
    lo_s, hi_s = R5_WINDOW
    n_tok = 8 * (hi_s - lo_s + 1)

    def at(c):
        r = replay_lru(steps, c)
        return r["hit"], (r["lookups"] - r["hits"]) / n_tok

    rows = []
    for g in hosts:
        c = capacity_records(g, record_bytes)
        h0, k0 = at(c)
        h3 = [at(int(c * f)) for f in (0.97, 1.03)]
        h5 = [at(int(c * f)) for f in (0.95, 1.05)]
        hits3, hits5 = [h for h, _ in h3 + [(h0, k0)]], [h for h, _ in h5 + h3 + [(h0, k0)]]
        cold3, cold5 = [k for _, k in h3 + [(h0, k0)]], [k for _, k in h5 + h3 + [(h0, k0)]]
        clip = lambda x: min(1.0, max(0.0, x))  # noqa: E731
        rows.append({
            "gpu": "L4", "model": "dsv4_flash", "b": 8, "host_gib": g, "capacity_records": c,
            "host_hit_rate": {
                "nominal": round(h0, 4),
                "lo": round(clip(min(hits3) - HIT_PAD), 4), "hi": round(clip(max(hits3) + HIT_PAD), 4),
                "kill_lo": round(clip(min(hits5) - HIT_KILL_PAD), 4),
                "kill_hi": round(clip(max(hits5) + HIT_KILL_PAD), 4)},
            "cold_records_per_tok": {
                "nominal": round(k0, 2),
                "lo": round(min(cold3) * (1 - COLD_PAD), 2), "hi": round(max(cold3) * (1 + COLD_PAD), 2),
                "kill_lo": round(min(cold5) * (1 - COLD_KILL_PAD), 2),
                "kill_hi": round(max(cold5) * (1 + COLD_KILL_PAD), 2)},
        })
    return rows


def run(write: bool = True) -> Dict[str, object]:
    from .constants import D4_RECORD
    from .predecl import _finite_hit
    from .roofline import build_pop, load_geoms

    rec = int(D4_RECORD.value)
    cohort = load_cohort()
    steps = stream(cohort)
    decode = [len(x) for x in steps[1:R5_LAST_STEP + 1]]
    lo, hi = R5_WINDOW
    pi_all = pooled_popularity(cohort)
    pi_sealed = build_pop(load_geoms()["dsv4_flash"]).pi
    b_distinct = N_PROMPTS - len(DUPLICATE_PROMPTS)

    check = []
    for r in R5_MEASURED["rows"]:
        cap = capacity_records(r["host_gib"], rec)
        rp = replay_lru(steps, cap)
        check.append({**r, "capacity_records": cap, "replay_hit": round(rp["hit"], 4),
                      "replay_lookups": rp["lookups"]})

    table = []
    for g in SWEEP_GIB:
        cap = capacity_records(g, rec)
        rp = replay_lru(steps, cap)
        table.append({
            "host_gib": g, "capacity_records": cap,
            "replay_hit": round(rp["hit"], 4),
            "model_rebuilt_pop_b7": round(step_lru_hit(pi_all, b_distinct, cap), 4),
            "model_sealed_pop_b8": round(step_lru_hit(pi_sealed, 8, cap), 4),
            "registered_nominal": round(float(_finite_hit(g, "dsv4_flash")), 4)
            if g == int(g) else None,
        })

    u_sealed = step_inclusion(pi_sealed, 8)
    u_all7 = step_inclusion(pi_all, b_distinct)
    u_all8 = step_inclusion(pi_all, 8)
    summary = {
        "cohort": "aws/phase6/prompts_phase4_8.json == Phase-4 q0..q7 (q7 repeats q0)",
        "journals": paths.rel(COHORT_DIR),
        "record_bytes": rec,
        "stream_check": {
            "decode_step_set_mean": round(float(np.mean(decode)), 1),
            "decode_step_set_min": int(min(decode)), "decode_step_set_max": int(max(decode)),
            "prefill_set": len(steps[0]),
            "window_lookups": int(sum(len(x) for x in steps[lo:hi + 1])),
            "window_step_set_max": int(max(len(x) for x in steps[lo:hi + 1])),
            "r5_measured": {k: v for k, v in R5_MEASURED.items() if k != "rows"},
        },
        "r5_check": check,
        "cliff_gib": {
            "sealed_pop_b8_model": round(union_over_steps(u_sealed, 1) * rec / GIB, 2),
            "rebuilt_pop_b7_model": round(union_over_steps(u_all7, 1) * rec / GIB, 2),
            "rebuilt_pop_b8_model_iid": round(union_over_steps(u_all8, 1) * rec / GIB, 2),
            "replay_first_hit_gib": edge_gib(steps, rec),
        },
        "records_seen_rebuilt": int(np.count_nonzero(pi_all)),
        "records_seen_sealed": int(np.count_nonzero(pi_sealed)),
    }
    add5 = {
        "addendum": "PREDICTIONS_PHASE6.md Addendum 5", "registered": ADD5_REGISTERED,
        "source": "theory.batch_cliff_replay (exact LRU replay of the rebuilt r5 b=8 stream)",
        "requires": {"gpu": "L4", "b": 8, "min_decode_steps": 64, "cohort": "0-7",
                     "prompts_file": "dee.cpp/aws/phase6/prompts_phase4_8.json",
                     "budget_mib": 3584, "window": "last 32 decode steps (score.py)"},
        "graded_metrics": ["host_hit_rate", "cold_records_per_tok"],
        "not_registered": ["decode_tps", "usd_per_1k_tok"],
        "rows": add5_rows(steps, rec),
    }
    summary["addendum5"] = add5
    if write:
        from .util import write_csv, write_json
        write_csv("batch_cliff_replay_grid.csv", table)
        write_json("batch_cliff_replay_summary.json", summary)
        write_json("predecl_add5.json", add5)
    summary["grid"] = table
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=float))
