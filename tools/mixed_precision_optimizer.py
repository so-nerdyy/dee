#!/usr/bin/env python3
"""mixed_precision_optimizer.py — Phase E: expert codec-assignment optimizer.

MEMORY-ONLY SIMULATION unless an explicit sensitivity file is provided.
No quality data is fabricated anywhere in this tool.

Objective (v1, storage-side):
    minimize  expected SSD bytes/token  +  cache occupancy pressure
    subject to  sum_e sensitivity_weight(e) * quality_penalty(e, codec_e) <= B

quality_penalty comes ONLY from the explicit sensitivity file
(--sensitivity-json, {"layer,expert": weight>0}); without it the run is
labeled MEMORY-ONLY SIMULATION and the budget constraint is disabled
(every weight treated as unknown -> uniform assignment only).

Design for the future scoring function (documented, not fabricated):
    v1:  value(e) = expected_reuse_saved_bytes / resident_bytes
    v2:  value(e, codec) = latency_saved - quality_penalty - eviction_cost

Outputs the assignment JSON consumed by expert_hierarchy_sim.py --mixed-map.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_hierarchy_sim import (  # noqa: E402
    DEFAULT_TRACE, GIB, expert_bytes, load_trace, simulate,
)

CODEC_ORDER = ["MXFP4_CURRENT", "IQ2_XXS_APPROX", "STQ1_0_IDEALIZED"]
# byte cost per expert relative to MXFP4, computed once
REL_COST = {c: expert_bytes(c) / expert_bytes("MXFP4_CURRENT") for c in CODEC_ORDER}
# synthetic sensitivity budget prices (NOT quality data; optimizer internals
# used only when an explicit sensitivity file is supplied)
PENALTY_TABLE = {  # illustrative unitless penalties per codec tier
    "MXFP4_CURRENT": 0.0,
    "IQ2_XXS_APPROX": 1.0,
    "STQ1_0_IDEALIZED": 2.5,
}


def value_v1(freq: int, eb: int, resident_eb: int) -> float:
    """v1 value: expected reuse saved bytes / resident bytes.

    expected_reuse_saved_bytes ~ freq * (MXFP4 bytes - codec bytes)
    resident_bytes ~ codec bytes (cache occupancy charge)
    """
    return freq * eb / resident_eb


def optimize(freq: Counter, sensitivity: dict | None, budget: float | None,
             n_mxfp4: int, n_iq2: int) -> dict:
    """Greedy tiered assignment.

    Without sensitivity: rank purely by reuse value density
    (freq * saved_bytes / resident_bytes) and cut tiers at n_mxfp4/n_iq2.

    With sensitivity + budget: greedy knapsack — walk experts in decreasing
    sensitivity-weighted value, assign the cheapest codec that fits the
    remaining budget, escalating precision for high-sensitivity experts.
    """
    keys = sorted(freq, key=lambda k: -freq[k])
    total_w = sum(sensitivity.values()) if sensitivity else 0.0
    assign = {}
    spent = 0.0
    if sensitivity and budget is not None:
        # Budgeted knapsack (correct direction):
        #   minimize total bytes  s.t.  sum_e sens(e) * penalty(codec_e) <= B.
        # Start at the min-penalty assignment (all MXFP4, penalty 0, max bytes)
        # and downgrade the LEAST sensitive experts first (IQ2, then STQ) while
        # the accumulated penalty stays within the budget.
        ranked = sorted(keys, key=lambda k: sensitivity.get(k, 0.0))  # least sensitive first
        for k in keys:
            assign[k] = "MXFP4_CURRENT"
        max_cost = sum(sensitivity.get(k, 0.0) for k in keys) * PENALTY_TABLE["STQ1_0_IDEALIZED"]
        if budget >= max_cost:
            # budget cannot constrain anything: take the all-cheapest assignment
            for k in keys:
                assign[k] = "STQ1_0_IDEALIZED"
            spent = max_cost
        else:
            for k in ranked:
                w = sensitivity.get(k, 0.0)
                # try STQ directly (largest byte saving)
                delta_stq = w * PENALTY_TABLE["STQ1_0_IDEALIZED"]
                if spent + delta_stq <= budget:
                    assign[k] = "STQ1_0_IDEALIZED"
                    spent += delta_stq
                    continue
                # else try IQ2
                delta_iq2 = w * PENALTY_TABLE["IQ2_XXS_APPROX"]
                if spent + delta_iq2 <= budget:
                    assign[k] = "IQ2_XXS_APPROX"
                    spent += delta_iq2
            # second pass: experts still at MXFP4 that fit IQ2
            for k in ranked:
                if assign[k] != "MXFP4_CURRENT":
                    continue
                w = sensitivity.get(k, 0.0)
                delta_iq2 = w * PENALTY_TABLE["IQ2_XXS_APPROX"]
                if spent + delta_iq2 <= budget:
                    assign[k] = "IQ2_XXS_APPROX"
                    spent += delta_iq2
        stats = {"mode": "budgeted", "budget": budget, "spent": round(spent, 3),
                 "max_possible_cost_all_stq": round(max_cost, 3),
                 "sensitivity_weight_total": round(total_w, 3)}
    else:
        for i, k in enumerate(keys):
            if i < n_mxfp4:
                assign[k] = "MXFP4_CURRENT"
            elif i < n_mxfp4 + n_iq2:
                assign[k] = "IQ2_XXS_APPROX"
            else:
                assign[k] = "STQ1_0_IDEALIZED"
        stats = {"mode": "MEMORY-ONLY SIMULATION", "budget": None, "spent": None}

    if assign is not None:
        counts = Counter(assign.values())
        stats["assignment_counts"] = dict(counts)
        stats["bytes_total_assigned"] = sum(expert_bytes(assign[k]) for k in assign)
    return assign, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase E codec-assignment optimizer")
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    ap.add_argument("--sensitivity-json", default=None,
                    help='explicit file {"layer,expert": weight}; REQUIRED for budgeted runs')
    ap.add_argument("--budget", type=float, default=None,
                    help="total sensitivity-weighted penalty units (budgeted mode)")
    ap.add_argument("--frac-mxfp4", type=float, default=0.10)
    ap.add_argument("--frac-iq2", type=float, default=0.30)
    ap.add_argument("--vram-gib", type=float, default=8.0)
    ap.add_argument("--ram-gib", type=float, default=16.0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--label", default="MIXED-OPT")
    args = ap.parse_args()

    trace = load_trace(args.trace)
    freq = Counter()
    for step in trace:
        for key in step.experts:
            freq[key] += 1

    sensitivity = None
    if args.sensitivity_json:
        with open(args.sensitivity_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        sensitivity = {tuple(int(x) for x in k.split(",")): float(v)
                       for k, v in raw.items()}

    n = len(freq)
    assign, stats = optimize(freq, sensitivity, args.budget,
                             int(round(args.frac_mxfp4 * n)),
                             int(round(args.frac_iq2 * n)))

    if assign is None:
        # infeasible budget: emit stats only, no sim
        out = {"label": args.label, "optimizer": stats,
               "label_note": "budget infeasible; no simulation run"}
        print(json.dumps(out, indent=2))
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        return 1

    def eb_for(k, _a=assign):
        return expert_bytes(_a.get(k, "STQ1_0_IDEALIZED"))

    res = simulate(trace, eb_for, int(args.vram_gib * GIB), int(args.ram_gib * GIB),
                   policy="LRU", trace_label=args.label, warm_start=True)

    out = {
        "label": args.label,
        "optimizer": stats,
        "sim": {k: res[k] for k in (
            "decode_combined_hit_rate", "decode_ssd_bytes_per_token",
            "decode_h2d_bytes_per_token", "ssd_bytes_per_token",
            "h2d_bytes_per_token", "vram_evictions", "ram_evictions")},
        "label_note": "MEMORY-ONLY SIMULATION — no quality claims"
                      if not sensitivity else "budgeted simulation with explicit sensitivity file",
    }
    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"assignment": {f"{l},{e}": c for (l, e), c in assign.items()},
                       **out}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
