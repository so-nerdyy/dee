#!/usr/bin/env python3
"""Phase F: ingest Codex's sealed host-buffer-reuse A/B into the simulator.

Usage:
    python tools/ingest_ab.py --ab path/to/ab_artifact.json [--out PATH]

Accepted artifact shapes (auto-detected):
  1. A/B summary schema (see research/exact-critical-path/AB_SCHEMA.json):
     {"baseline": {...run fields..., "reuse_telemetry": {...}},
      "candidate": {...run fields..., "reuse_telemetry": {...}}}
  2. A single sealed run in the standard v65 result.json shape plus
     "variant": "baseline" | "candidate" (ingest twice, once per arm).

Run fields consumed: decode_wall_s (required), decode_tokens, run_id, commit.
Unknown fields are echoed, never invented. If a metric is absent it is
reported as UNKNOWN — nothing is inferred.

The simulator then:
  - predicts each arm with host_buffer_reuse False/True,
  - checks whether the predicted direction matches the measured direction,
  - re-fits the pack-copy cost under reuse from the candidate arm if the
    measured wall is available (calibration update, reported explicitly).

Simulated speedup != measured speedup. Direction agreement is the claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from exact_critical_path_sim import (  # noqa: E402
    default_params, simulate, gpu_layer_uniques, load_route_bundles, find_trace,
    V65,
)

OUT_DIR = HERE.parent / "research" / "exact-critical-path" / "results"


def extract_run(obj: dict) -> dict:
    """Pull the fields the model needs; mark everything else UNKNOWN."""
    return {
        "run_id": obj.get("run_id", "UNKNOWN"),
        "commit": obj.get("commit", "UNKNOWN"),
        "variant": obj.get("variant", "UNKNOWN"),
        "decode_wall_s": obj.get("decode_wall_s"),          # required
        "decode_tokens": obj.get("decode_tokens", V65["tokens"]),
        "reuse_telemetry": obj.get("reuse_telemetry", {}),
        "unknown_fields": sorted(
            k for k in obj
            if k not in {"run_id", "commit", "variant", "decode_wall_s",
                         "decode_tokens", "reuse_telemetry"}
        ),
    }


def load_ab(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "baseline" in data and "candidate" in data:
        return {"baseline": extract_run(data["baseline"]),
                "candidate": extract_run(data["candidate"])}
    if "baseline" in data or "candidate" in data:
        raise SystemExit(f"{path}: A/B artifact has only one arm; ingest each "
                         "arm separately with 'variant' set, or complete the pair.")
    # single sealed run shape
    run = extract_run(data)
    if run["variant"] == "UNKNOWN":
        raise SystemExit(f"{path}: single-run artifact needs a 'variant' field "
                         "('baseline' or 'candidate').")
    return {run["variant"]: run}


def predict_arm(params: dict, uniques: list, reuse: bool) -> float:
    p = dict(params)
    p["host_buffer_reuse"] = reuse
    return simulate(p, uniques)["predicted_wall_s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ab", type=Path, required=True, nargs="+",
                    help="A/B artifact JSON path(s).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ab: dict = {}
    for p in args.ab:
        for k, v in load_ab(p).items():
            if k in ab:
                raise SystemExit(f"duplicate variant '{k}' across artifacts")
            ab[k] = v
    if "baseline" not in ab or "candidate" not in ab:
        raise SystemExit(f"need both arms; have {sorted(ab)}")

    trace = find_trace()
    uniques = gpu_layer_uniques(load_route_bundles(trace))
    params = default_params()
    cal = OUT_DIR / "calibration.json"
    if cal.exists():
        params = json.loads(cal.read_text())["params"]

    pred_base = predict_arm(params, uniques, reuse=False)
    pred_cand = predict_arm(params, uniques, reuse=True)

    report = {
        "note": "SIMULATED vs MEASURED comparison. Direction agreement only; "
                "simulated speedup != measured speedup.",
        "arms": ab,
        "measured": {},
        "simulated": {
            "baseline_wall_s": round(pred_base, 3),
            "candidate_wall_s": round(pred_cand, 3),
            "predicted_delta_pct": round((pred_cand - pred_base) / pred_base * 100, 2),
        },
        "direction_agrees": None,
        "calibration_update": None,
    }

    mb, mc = ab["baseline"], ab["candidate"]
    for name, arm in (("baseline", mb), ("candidate", mc)):
        if arm["decode_wall_s"] is None:
            report["measured"][name] = "UNKNOWN (decode_wall_s missing)"
        else:
            report["measured"][name] = {
                "wall_s": arm["decode_wall_s"],
                "tokens": arm["decode_tokens"],
                "tok_s": round(arm["decode_tokens"] / arm["decode_wall_s"], 4)
                if arm["decode_wall_s"] else "UNKNOWN",
            }
    if all(isinstance(report["measured"][n], dict) for n in ("baseline", "candidate")):
        wb = report["measured"]["baseline"]["wall_s"]
        wc = report["measured"]["candidate"]["wall_s"]
        meas_delta = (wc - wb) / wb * 100
        report["measured"]["delta_pct"] = round(meas_delta, 2)
        report["direction_agrees"] = bool(
            (pred_cand - pred_base) * (wc - wb) >= 0
        ) if meas_delta != 0 else None
        # calibration update: effective pack cost under reuse that would make
        # the model match the candidate arm exactly (linear in pack_ms).
        if wc != wb:
            p = dict(params)
            p["host_buffer_reuse"] = True
            p["host_pack_ms_per_record"] = 0.0
            w0 = simulate(p, uniques)["predicted_wall_s"]
            p["host_pack_ms_per_record"] = params["host_pack_ms_per_record"]
            w1 = simulate(p, uniques)["predicted_wall_s"]
            if abs(w1 - w0) > 1e-9:
                frac = (wc - w0) / (w1 - w0)
                fitted = max(0.0, params["host_pack_ms_per_record"] * frac)
                report["calibration_update"] = {
                    "kind": "pack_ms_under_reuse",
                    "fitted_ms_per_record": round(fitted, 4),
                    "reuse_model": "0.05 ms pointer-swap (current)",
                    "caveat": "single-point fit on the A/B candidate arm; "
                              "trace-local, no cross-trace claim",
                }
    out = args.out or (OUT_DIR / "ab_ingestion.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("measured", "simulated", "direction_agrees",
                       "calibration_update")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
