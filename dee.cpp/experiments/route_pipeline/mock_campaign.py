#!/usr/bin/env python3
"""Deterministic mock canonical campaign (no CUDA, no checkpoint).

Behaves like the canonical decode interface for the evidence pipeline:
writes profile-run/{result.json, host-profile.jsonl, stage-profile.json,
correctness.json} (+ optional mock ABC payload) for --out DIR, honoring
--profile-stages. All values are FIXED per scenario (no randomness); timings
are illustrative constants for exercising ingestion/closure/ranking logic,
never T4 measurements.

Scenarios:
  1 sync-dominant      -> EVENT_HANDOFF promoted
  2 shared-dominant    -> SHARED_OVERLAP promoted
  3 d2h-dominant       -> ROUTE_D2H promoted
  4 nothing material   -> NO_OVERLAP_OPTIMIZATION_JUSTIFIED
  5 incomplete closure -> PROFILE_INCOMPLETE (all HOLD)
  6 pair mismatch      -> INVALID_PROFILE_PAIR (on-run IDs differ)
  7 malformed          -> INVALID_PROFILE_EVIDENCE (corrupt jsonl)

Geometry: 2 tokens x 4 layers (shape-agnostic ingestion; real runs use 43).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

TOKENS, LAYERS = 2, 4
TOPK = 6
IDS = [list(range(100, 106)), list(range(200, 206))]
TEXT = "mock decode text (deterministic)"
SHA = "mock-source-sha"
HARDWARE = {"count": 2, "names": ["Tesla T4", "Tesla T4"], "compute_cap": "7.5"}

# Per-row span tables. Children of native_call sum to <= native.
BASE_CHILDREN = {"source_wait_ms": 1.0, "fill_wait_ms": 1.0, "h2d_ms": 2.0,
                 "decode_ms": 0.5, "expert_compute_ms": 1.0,
                 "native_output_sync_wait_ms": 0.2}
BASE_TOP = {"route_d2h_host_wait_ms": 0.05, "shared_expert_ms": 0.05,
            "combine_ms": 0.2, "orchestration_ms": 0.1}


def _row_spans(children: dict, top: dict) -> dict:
    """C++ host_span_name() keys (faithful to host_layer_records_json)."""
    cpp = {"source_wait_ms": "source_lookup_wait",
           "fill_wait_ms": "fill_wait",
           "decode_ms": "decode_wait",
           "expert_compute_ms": "expert_compute_wait",
           "native_output_sync_wait_ms": "native_output_sync"}
    native = round(sum(children.values()), 6)
    spans = {"native_call_wall": {"ms": native, "provenance": "HOST_WALL"}}
    for schema_name, ms in children.items():
        if schema_name == "h2d_ms":
            continue  # H2D device time is a GpuStage, not a host span
        spans[cpp[schema_name]] = {"ms": ms, "provenance": "HOST_WALL"}
    return spans


def scenario_spans(number: int) -> tuple[dict, dict]:
    children = dict(BASE_CHILDREN)
    top = dict(BASE_TOP)
    if number == 1:
        children["native_output_sync_wait_ms"] = 6.0
    elif number == 2:
        children["decode_ms"] = 0.2
        children["expert_compute_ms"] = 0.3
        top["shared_expert_ms"] = 3.0
    elif number == 3:
        children["decode_ms"] = 0.2
        children["expert_compute_ms"] = 0.3
        top["route_d2h_host_wait_ms"] = 2.0
    elif number == 4:
        children["decode_ms"] = 0.2
        children["expert_compute_ms"] = 0.3
    elif number == 5:
        children["decode_ms"] = 0.2
        children["expert_compute_ms"] = 0.3
    return children, top


def scenario_abc(number: int) -> dict:
    hidden = 5.0 if number in (1, 3) else 0.2
    return {
        "gpu": "mock-T4", "status": "ok",
        "A_hash_staging": {
            "status": "ok", "reads": 18, "bytes": 18 * 13369344,
            "route_table_realism": "ROUTE_TABLE_REALISM_NOT_MEASURED",
            "serial_ms": {"p50": 12.0, "mean": 12.0, "min": 11.0, "n": 5},
            "batched_ms": {"p50": 12.0 - hidden, "mean": 12.0 - hidden,
                           "min": 11.0 - hidden, "n": 5}},
        "B_contention": {
            "status": "ok", "alone_ms": {"p50": 1.0}, "loaded_ms": {"p50": 1.2},
            "efficiency_c": 0.83},
        "C_barrier": {
            "status": "ok", "sync_ms": {"p50": 0.05}, "event_ms": {"p50": 0.01}},
    }


def build_run(out: Path, scenario: int, profile_stages: bool,
              mode: str, wall_scale: float = 1.0) -> dict:
    children, top = scenario_spans(scenario)
    cpp_records, py_lines = [], []
    for token in range(TOKENS):
        for layer in range(LAYERS):
            device = 0 if layer < 2 else 1
            cpp_records.append({
                "token": token, "layer": layer, "device": device,
                "spans": _row_spans(children, top),
                "counters": {"ids_bytes": 24, "copies": 1, "syncs": 1, "events": 0}})
            native = round(sum(children.values()), 6)
            py_lines.append(json.dumps({
                "token": token, "layer": layer, "device": device,
                "route_d2h_host_wait_ms": top["route_d2h_host_wait_ms"],
                "route_d2h_provenance": "HOST_WALL",
                "native_call_wall_ms": native,  # duplicate: dropped on merge
                "combine_ms": top["combine_ms"],
                "combine_provenance": "HOST_WALL",
                "shared_host_wall_ms": top["shared_expert_ms"],
                "shared_device_ms": None,
                "shared_device_provenance": "UNKNOWN",
                "ids_bytes": 24}))
    row_wall = (top["route_d2h_host_wait_ms"] + sum(children.values())
                + top["shared_expert_ms"] + top["combine_ms"] + top["orchestration_ms"])
    wall_ms = round(row_wall * TOKENS * LAYERS * wall_scale, 6)
    ids = IDS if not (scenario == 6 and mode == "on") else [[1, 2, 3, 4, 5, 6]] * 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage-profile.json").write_text(json.dumps(
        {"stage_profile": {"note": "mock", "profile_stages": profile_stages},
         "host_layer_records": {"records": cpp_records}}, indent=2), encoding="utf-8")
    host_path = out / "host-profile.jsonl"
    if scenario == 7 and mode == "on":
        host_path.write_text('{"layer": 0, "broken": \n', encoding="utf-8")
    else:
        host_path.write_text("\n".join(py_lines) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps({
        "status": "ok",
        "metrics": {"decode_wall_s": wall_ms / 1000.0, "tokens": TOKENS},
        "generated_ids": ids, "decoded_text": TEXT,
        "source_sha": SHA, "prompt_hash": "mock-prompt-hash",
        "config": {"model": "mock-dsv4", "topk": TOPK,
                   "profile_stages": profile_stages,
                   "DEE_HOST_PROFILE": profile_stages},
        "hardware": HARDWARE}, indent=2), encoding="utf-8")
    (out / "correctness.json").write_text(json.dumps(
        {"classification": "ACCEPT_CORRECTNESS",
         "gates": {"exact_ids": True, "exact_text": True, "sealed": True}},
        indent=2), encoding="utf-8")
    return {"wall_ms": wall_ms, "dir": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", type=int, default=1, choices=range(1, 8))
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--mode", default=None, choices=("off", "on"),
                        help="default: derived from DEE_HOST_PROFILE env (1->on)")
    parser.add_argument("--emit-abc", type=Path, default=None)
    parser.add_argument("--wall-scale", type=float, default=1.0)
    args = parser.parse_args()
    mode = args.mode or ("on" if os.environ.get("DEE_HOST_PROFILE") == "1" else "off")
    info = build_run(args.out, args.scenario, args.profile_stages,
                     mode, args.wall_scale)
    if args.emit_abc is not None:
        wall_scale_note = ("scenario 5 uses --wall-scale 3.0 via run_evidence; "
                           "mock ABC is scenario-shaped only")
        payload = scenario_abc(args.scenario)
        payload["note"] = wall_scale_note
        args.emit_abc.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"mock": "ok", **info}))


if __name__ == "__main__":
    raise SystemExit(main())
