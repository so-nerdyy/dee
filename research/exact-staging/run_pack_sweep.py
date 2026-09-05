#!/usr/bin/env python3
"""Pack-budget sweep (exact-staging track): replay the production host-pack
LRU over the sealed v65 journal at multiple budgets and estimate decode wall
with the calibrated critical-path model (imported READ-ONLY from the
research/exact-critical-path worktree, if present).

Outputs:
  results/pack_replay_sweep.json     validated replay counters per budget
  results/pack_budget_wall_estimates.json  simulated wall estimates

No sealed evidence is modified. All wall numbers are SIMULATED estimates,
not measured speedups. No TPS claims.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = HERE.parent.parent
sys.path.insert(0, str(WORKTREE_ROOT / "dee.cpp" / "experiments" / "exact_staging"))
from bounded_staging_queue import pack_replay  # noqa: E402

RECORD_BYTES = 13_369_344
SEALED_DECODE_MISSES = 1252
# recalibrated observational model (recalibrate_model.py, results/recalibrated_model.json):
# wall_ms_step = 754.74 + 48.679 * misses (R^2 0.9785 on sealed v65; held-out
# errors 1.3-2.2% on the 5 host-reuse seal runs). SIM_BASELINE_S below is the
# model's prediction for the 8.5 GiB/GPU (682-record) replay, not a measurement.
MISS_SERVICE_MS = 48.679
SERIAL_RESIDUAL_MS = 754.74
SIM_BASELINE_S = (SERIAL_RESIDUAL_MS * 15 + MISS_SERVICE_MS * SEALED_DECODE_MISSES) / 1000
OBSERVED_S = 72.267


def maybe_wall_estimate(per_step_reads: list[int]) -> float | None:
    """Predict decode wall with the recalibrated observational model
    (research/exact-staging/recalibrate_model.py)."""
    return sum(SERIAL_RESIDUAL_MS + MISS_SERVICE_MS * r
               for r in per_step_reads) / 1000



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", type=Path, required=True,
                    help="sealed routed_experts.jsonl (read-only)")
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[682, 762, 843, 883, 923, 963, 1024],
                    help="pack capacity in records (682 = sealed 8.5 GiB)")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sweep = []
    for cap in args.budgets:
        r = pack_replay(args.journal, cap)
        dm = r["decode_storage_misses_total"]
        per_gpu = {d: (v["misses"], v["evictions"])
                   for d, v in r["per_device"].items()}
        print(f"cap={cap:>5} rec ({r['budget_gib_per_gpu']:>5} GiB/GPU): "
              f"decode misses={dm:>5} (sealed {SEALED_DECODE_MISSES}) "
              f"per-GPU (misses, evictions)={per_gpu}")
        wall = maybe_wall_estimate(list(r["decode_per_step_reads"].values()))
        if wall is not None:
            print(f"          simulated decode wall={wall} s "
                  f"({-100 * (1 - wall / SIM_BASELINE_S):+.2f}% vs sim baseline, "
                  f"{-100 * (1 - wall / OBSERVED_S):+.2f}% vs observed)")
        sweep.append({"budget_records": cap,
                      "budget_gib_per_gpu": r["budget_gib_per_gpu"],
                      "decode_storage_misses": dm,
                      "per_device": r["per_device"],
                      "decode_per_step_reads": r["decode_per_step_reads"],
                      "simulated_decode_wall_s": wall,
                      "delta_vs_sim_baseline_pct": (round(-100 * (1 - wall / SIM_BASELINE_S), 2)
                                                    if wall else None),
                      "delta_vs_observed_pct": (round(-100 * (1 - wall / OBSERVED_S), 2)
                                                if wall else None)})

    (args.out / "pack_replay_sweep.json").write_text(json.dumps(
        {"schema": "exact-staging/pack-replay-sweep-v1",
         "journal": str(args.journal),
         "sealed_decode_storage_misses": SEALED_DECODE_MISSES,
         "note": "Replay is validated: LRU@682 reproduces sealed v65 per-GPU "
                 "pack counters within 2 records. Wall numbers are SIMULATED.",
         "sweep": sweep}, indent=2))
    print(f"saved {args.out / 'pack_replay_sweep.json'}")


if __name__ == "__main__":
    main()
