#!/usr/bin/env python3
"""Phase C/D/E: what-if sweeps, sensitivity, and interaction analysis.

Uses the calibrated simulator (tools/exact_critical_path_sim.py) with the
parameters fitted to sealed v65 (see results/calibration.json).

Every number produced here is a SIMULATED RESULT, not a measured speedup.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from exact_critical_path_sim import (  # noqa: E402
    default_params, calibrate, simulate, gpu_layer_uniques, load_route_bundles,
    find_trace, V65,
)

OUT_DIR = HERE.parent / "research" / "exact-critical-path" / "results"


def load_calibrated() -> tuple[dict, list]:
    params = default_params()
    cal_file = OUT_DIR / "calibration.json"
    if cal_file.exists():
        params = json.loads(cal_file.read_text())["params"]
    else:
        trace = find_trace()
        uniques = gpu_layer_uniques(load_route_bundles(trace))
        params, _ = calibrate(uniques, params)
    trace = find_trace()
    uniques = gpu_layer_uniques(load_route_bundles(trace))
    return params, uniques


def wall_of(params, uniques, overrides: dict) -> float:
    p = dict(params)
    p.update(overrides)
    res = simulate(p, uniques)
    return res["predicted_wall_s"]


# ---------------------------------------------------------------------------
# Phase C: exact-preserving changes
# ---------------------------------------------------------------------------

SCENARIOS = {
    # label -> parameter overrides
    "baseline_v65": {},
    "buffer_reuse": {"host_buffer_reuse": True},
    "grouped_h2d": {"grouped_h2d": True},
    "grouped_dispatch": {"grouped_dispatch": True},
    "host_orch_-50%": {"host_orch_reduction": 0.5},
    "host_orch_-100%": {"host_orch_reduction": 1.0},
    "staging_lead_8": {"staging_lead_layers": 8},
    "staging_lead_16": {"staging_lead_layers": 16},
    "staging_lead_30": {"staging_lead_layers": 30},
    "staging_lead_full": {"staging_lead_layers": 42},
    "h2d_2x_rate": {"h2d_gbps": None},   # filled dynamically (2x fitted)
    "lanes_3": {"read_lanes": 3},
    "lanes_12": {"read_lanes": 12},
    "no_overlap": {"compute_overlaps_copies": False},
}


def phase_c(params, uniques) -> list[dict]:
    rows = []
    base = wall_of(params, uniques, {})
    for label, ov in SCENARIOS.items():
        if label == "baseline_v65":
            w = base
        else:
            o = dict(ov)
            if label == "h2d_2x_rate":
                o["h2d_gbps"] = params["h2d_gbps"] * 2
            w = wall_of(params, uniques, o)
        rows.append({
            "scenario": label,
            "predicted_wall_s": round(w, 3),
            "delta_s_vs_baseline": round(w - base, 3),
            "delta_pct": round((w - base) / base * 100, 2),
        })
    rows.sort(key=lambda r: r["predicted_wall_s"])
    return rows


# ---------------------------------------------------------------------------
# Phase D: sensitivity sweeps (SSD BW, H2D rate, GPU compute, host overhead)
# ---------------------------------------------------------------------------

def phase_d(params, uniques) -> dict:
    out = {}
    base = wall_of(params, uniques, {})
    sweeps = {
        "ssd_aggregate_mb_s": [0.75, 1.0, 1.25, 1.5, 2.0],
        "h2d_gbps": [0.75, 1.0, 1.25, 1.5, 2.0],
        "gpu_compute_per_record_ms": [0.75, 1.0, 1.25, 1.5, 2.0],
        "host_orchestration_per_layer_ms": [0.5, 1.0, 2.0, 4.0],
        "staging_lead_layers": [0, 4, 8, 16, 30, 42],
    }
    for key, mults in sweeps.items():
        rows = []
        for m in mults:
            if key == "staging_lead_layers":
                v = m
            elif key == "host_orchestration_per_layer_ms":
                v = params[key] * m
            else:
                v = params[key] * m
            w = wall_of(params, uniques, {key: v})
            rows.append({"param": key, "value": round(v, 4),
                         "multiplier": m, "wall_s": round(w, 3),
                         "delta_pct": round((w - base) / base * 100, 2)})
        out[key] = rows
    return out


# ---------------------------------------------------------------------------
# Phase E: interactions (pruned: only combos touching distinct resources)
# ---------------------------------------------------------------------------

def phase_e(params, uniques) -> list[dict]:
    combos = [
        ("buffer_reuse+lead16", {"host_buffer_reuse": True, "staging_lead_layers": 16}),
        ("buffer_reuse+grouped_h2d", {"host_buffer_reuse": True, "grouped_h2d": True}),
        ("grouped_h2d+grouped_dispatch", {"grouped_h2d": True, "grouped_dispatch": True}),
        ("grouped_h2d+lead16", {"grouped_h2d": True, "staging_lead_layers": 16}),
        ("orch-100%+lead16", {"host_orch_reduction": 1.0, "staging_lead_layers": 16}),
        ("buffer_reuse+grouped_h2d+lead16",
         {"host_buffer_reuse": True, "grouped_h2d": True, "staging_lead_layers": 16}),
        ("orch-100%+grouped_dispatch",
         {"host_orch_reduction": 1.0, "grouped_dispatch": True}),
    ]
    rows = []
    base = wall_of(params, uniques, {})
    for label, ov in combos:
        w = wall_of(params, uniques, ov)
        rows.append({"combo": label, "wall_s": round(w, 3),
                     "delta_pct_vs_baseline": round((w - base) / base * 100, 2)})
    rows.sort(key=lambda r: r["wall_s"])
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params, uniques = load_calibrated()
    base = wall_of(params, uniques, {})
    print(f"calibrated baseline wall: {base:.2f} s (observed {V65['decode_wall_s']} s)")

    c_rows = phase_c(params, uniques)
    d_rows = phase_d(params, uniques)
    e_rows = phase_e(params, uniques)

    payload = {
        "note": "SIMULATED RESULTS — not measured speedups. Calibrated to sealed v65.",
        "calibrated_params": params,
        "baseline_wall_s": round(base, 3),
        "observed_wall_s": V65["decode_wall_s"],
        "phase_c_scenarios": c_rows,
        "phase_d_sweeps": d_rows,
        "phase_e_interactions": e_rows,
    }
    (OUT_DIR / "sweeps.json").write_text(json.dumps(payload, indent=2))

    with (OUT_DIR / "phase_c_scenarios.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(c_rows[0].keys()))
        w.writeheader()
        w.writerows(c_rows)
    with (OUT_DIR / "phase_e_interactions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(e_rows[0].keys()))
        w.writeheader()
        w.writerows(e_rows)

    print("\nPhase C (scenarios):")
    for r in c_rows:
        print(f"  {r['scenario']:<28} {r['predicted_wall_s']:>8.2f} s  {r['delta_pct']:>+7.2f}%")
    print("\nPhase E (interactions):")
    for r in e_rows:
        print(f"  {r['combo']:<36} {r['wall_s']:>8.2f} s  {r['delta_pct_vs_baseline']:>+7.2f}%")
    print(f"\nwrote {OUT_DIR}/sweeps.json, phase_c_scenarios.csv, phase_e_interactions.csv")


if __name__ == "__main__":
    main()
