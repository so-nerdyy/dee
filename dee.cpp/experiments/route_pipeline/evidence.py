#!/usr/bin/env python3
"""End-to-end evidence ingestion, ranking, and next-A/B generation.

Companion to host_profiler.py (schema, closure, base decisions). This module
owns the canonical profile-run/ contract, record merging without duplicate
accounting, off/on pair validation, ABC mechanics ingestion, evidence-fed
ranking, and single-candidate next-A/B derivation. No torch/CUDA needed.

Failure taxonomy (never silent empty success):
  EvidenceError -> INVALID_PROFILE_EVIDENCE (malformed/missing telemetry)
  PairError     -> INVALID_PROFILE_PAIR (mismatched off/on runs)
"""

from __future__ import annotations

import json
from pathlib import Path

from host_profiler import (
    NESTED_IN_NATIVE_CALL,
    SPAN_FIELDS,
    SchemaError,
    compute_closure,
    evaluate_decisions,
    validate_record,
)

PROFILE_RUN_FILES = ("result.json", "host-profile.jsonl",
                     "stage-profile.json", "correctness.json")


class EvidenceError(ValueError):
    """Malformed/missing evidence -> INVALID_PROFILE_EVIDENCE (fail closed)."""
    code = "INVALID_PROFILE_EVIDENCE"


class PairError(ValueError):
    """Mismatched off/on pair -> INVALID_PROFILE_PAIR (fail closed)."""
    code = "INVALID_PROFILE_PAIR"


# C++ host_span_name() -> schema field. LayerWall (native-side sub-wall),
# StageEnqueueWait, ReadinessWait, GatherScatterWait are kept as extras:
# real attribution detail that must never join closure sums (the parent
# native_call_wall_ms already contains them).
CPP_SPAN_MAP = {
    "native_call_wall": "native_call_wall_ms",
    "source_lookup_wait": "source_wait_ms",
    "fill_wait": "fill_wait_ms",
    "decode_wait": "decode_ms",
    "expert_compute_wait": "expert_compute_ms",
    "native_output_sync": "native_output_sync_wait_ms",
    "shared_expert": "shared_expert_ms",
    "combine": "combine_ms",
    "orchestration": "orchestration_ms",
    "handoff": "handoff_ms",
}
CPP_EXTRA_SPANS = ("layer_wall", "stage_enqueue_wait", "readiness_wait",
                   "gather_scatter_wait")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvidenceError(f"missing required profile file: {path}")
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"malformed JSON in {path}: {exc}")


def load_profile_run(run_dir) -> dict:
    """Read and structurally validate one profile-run/ directory."""
    run_dir = Path(run_dir)
    result = _read_json(run_dir / "result.json")
    if not isinstance(result, dict):
        raise EvidenceError("result.json must be an object")
    for key in ("status", "metrics", "source_sha", "hardware"):
        if key not in result:
            raise EvidenceError(f"result.json missing required key: {key}")
    correctness = _read_json(run_dir / "correctness.json")
    stage = _read_json(run_dir / "stage-profile.json")
    try:
        lines = (run_dir / "host-profile.jsonl").read_text(
            encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise EvidenceError(f"missing required profile file: {run_dir / 'host-profile.jsonl'}")
    py_rows = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"host-profile.jsonl line {lineno}: {exc}")
        if not isinstance(row, dict):
            raise EvidenceError(f"host-profile.jsonl line {lineno}: not an object")
        py_rows.append(row)
    return {"result": result, "correctness": correctness, "stage": stage,
            "py_rows": py_rows, "dir": str(run_dir)}


def _normalize_cpp_record(rec: dict) -> dict:
    try:
        token, layer, device = rec["token"], rec["layer"], rec["device"]
        spans = rec["spans"]
    except (KeyError, TypeError):
        raise EvidenceError("C++ record needs token/layer/device/spans")
    if not isinstance(spans, dict):
        raise EvidenceError("C++ record spans must be an object")
    out: dict = {"token": token, "layer": layer, "device": device,
                 "provenance": {}, "extras": {}}
    for name, entry in spans.items():
        if not isinstance(entry, dict) or "ms" not in entry:
            raise EvidenceError(f"C++ span {name!r} needs an ms field")
        ms, prov = entry["ms"], entry.get("provenance", "UNKNOWN")
        if name in CPP_SPAN_MAP:
            out[CPP_SPAN_MAP[name]] = ms
            out["provenance"][CPP_SPAN_MAP[name]] = prov
        elif name in CPP_EXTRA_SPANS:
            out["extras"][name + "_ms"] = ms
        else:
            raise EvidenceError(f"unknown C++ span {name!r}")
    counters = rec.get("counters", {})
    if not isinstance(counters, dict):
        raise EvidenceError("C++ counters must be an object")
    out["counters"] = dict(counters)
    return out


_PY_FIELD_MAP = {
    "route_d2h_host_wait_ms": "route_d2h_host_wait_ms",
    "native_call_wall_ms": "native_call_wall_ms",
    "combine_ms": "combine_ms",
    "shared_host_wall_ms": "shared_expert_ms",
}
_PY_PROVENANCE_KEYS = ("route_d2h_provenance", "combine_provenance")


def _normalize_py_row(row: dict) -> dict:
    if "layer" not in row:
        raise EvidenceError("Python row needs layer")
    out: dict = {"token": row.get("token", -1), "layer": row["layer"],
                 "device": row.get("device", -1),
                 "provenance": {}, "extras": {}}
    for src, dst in _PY_FIELD_MAP.items():
        if src in row and row[src] is not None:
            out[dst] = row[src]
            label = "HOST_WALL"
            for cand in (src + "_provenance", dst + "_provenance",
                         *_PY_PROVENANCE_KEYS):
                if cand in row and isinstance(row[cand], str):
                    label = row[cand]
                    break
            out["provenance"][dst] = label
    if "shared_device_ms" in row:
        out["extras"]["shared_device_ms"] = row["shared_device_ms"]
        out["extras"]["shared_device_provenance"] = row.get(
            "shared_device_provenance", "UNKNOWN")
    if "ids_bytes" in row:
        out["counters"] = {"ids_bytes": row["ids_bytes"]}
    return out


def merge_records(cpp_records: list[dict], py_rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Merge by (token, layer, device). Precedence (no double accounting):
    C++ owns native spans (native_call_wall + nested detail); Python owns
    route_d2h/shared/combine/orchestration; a Python native_call_wall is
    DROPPED when a C++ record exists for the same key (noted); kept
    otherwise. Nulls preserved as UNKNOWN."""
    merged: dict[tuple, dict] = {}
    notes: list[str] = []
    for rec in cpp_records:
        norm = _normalize_cpp_record(rec)
        key = (norm["token"], norm["layer"], norm["device"])
        if key in merged:
            raise EvidenceError(f"duplicate C++ record for {key}")
        merged[key] = norm
    for row in py_rows:
        norm = _normalize_py_row(row)
        key = (norm["token"], norm["layer"], norm["device"])
        base = merged.get(key)
        if base is None:
            merged[key] = norm
            notes.append(f"{key}: python-only record (no C++ spans)")
            continue
        for field in SPAN_FIELDS:
            if field in norm and norm[field] is not None and field not in base:
                base[field] = norm[field]
                base["provenance"][field] = norm["provenance"].get(field, "HOST_WALL")
        if "native_call_wall_ms" in norm and "native_call_wall_ms" in base:
            notes.append(f"{key}: python native_call_wall dropped (C++ owns native spans)")
        base.get("extras", {}).update(norm.get("extras", {}))
        if "counters" in norm:
            base.setdefault("counters", {}).update(norm["counters"])
    records = []
    for (token, layer, device) in sorted(merged):
        rec = merged[(token, layer, device)]
        full = {"token": token, "layer": layer, "device": device}
        for field in SPAN_FIELDS:
            full[field] = rec.get(field)
        full["provenance"] = {f: rec.get("provenance", {}).get(f, "UNKNOWN")
                              for f in SPAN_FIELDS}
        full["extras"] = rec.get("extras", {})
        if "counters" in rec:
            full["counters"] = rec["counters"]
        records.append(validate_record(full))
    return records, notes


def records_to_csv(records: list[dict]) -> str:
    lines = ["token,layer,device," + ",".join(SPAN_FIELDS) + ",provenance"]
    for rec in records:
        cells = [str(rec[k]) for k in ("token", "layer", "device")]
        for field in SPAN_FIELDS:
            value = rec.get(field)
            cells.append("" if value is None else f"{value:.6f}")
        cells.append(";".join(f"{f}={rec['provenance'].get(f, 'UNKNOWN')}"
                              for f in SPAN_FIELDS))
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def rollup_per_token(records: list[dict]) -> list[dict]:
    by_token: dict = {}
    for rec in records:
        by_token.setdefault(rec["token"], []).append(rec)
    out = []
    for token in sorted(by_token):
        rows = by_token[token]
        entry: dict = {"token": token, "layers": len(rows), "by_field_ms": {}}
        for field in SPAN_FIELDS:
            if field in NESTED_IN_NATIVE_CALL:
                continue
            entry["by_field_ms"][field] = round(
                sum(r[field] for r in rows if r.get(field) is not None), 6)
        out.append(entry)
    return out


def validate_pair(off: dict, on: dict) -> dict:
    """Matched off/on validation. Walls come from result.json metrics ONLY
    (never stdout tails). Any mismatch -> PairError (INVALID_PROFILE_PAIR)."""
    ro, rn = off["result"], on["result"]
    problems = []
    for key in ("source_sha", "prompt_hash", "hardware"):
        if ro.get(key) != rn.get(key):
            problems.append(f"{key} differs off/on")
    if ro.get("tokens") != rn.get("tokens"):
        problems.append("token count differs off/on")
    co, cn = dict(ro.get("config", {})), dict(rn.get("config", {}))
    for cfg in (co, cn):
        cfg.pop("profile_stages", None)
        cfg.pop("DEE_HOST_PROFILE", None)
    if co != cn:
        problems.append("config differs off/on beyond profiling flags")
    if ro.get("generated_ids") != rn.get("generated_ids"):
        problems.append("generated IDs differ off/on")
    if ro.get("decoded_text") != rn.get("decoded_text"):
        problems.append("decoded text differs off/on")
    for tag, correctness in (("off", off["correctness"]), ("on", on["correctness"])):
        gates = (correctness.get("gates", {}) if isinstance(correctness, dict) else {})
        if not gates or not all(gates.values()):
            problems.append(f"{tag} correctness gates not all passing")
    if problems:
        raise PairError("INVALID_PROFILE_PAIR: " + "; ".join(problems))
    try:
        off_wall = float(ro["metrics"]["decode_wall_s"])
        on_wall = float(rn["metrics"]["decode_wall_s"])
    except (KeyError, TypeError, ValueError):
        raise PairError("INVALID_PROFILE_PAIR: decode_wall_s missing/non-numeric")
    perturbation = on_wall - off_wall
    return {"pair": "VALID",
            "off_wall_s": off_wall, "on_wall_s": on_wall,
            "perturbation_s": round(perturbation, 6),
            "perturbation_fraction": (round(perturbation / off_wall, 6)
                                      if off_wall else None),
            "tokens": ro.get("tokens"),
            "source_sha": ro.get("source_sha")}


def ingest_abc(payload: dict) -> dict:
    """Normalize ABC runner output from parsed JSON fields only (never stdout
    tails). Missing/failing cases stay UNKNOWN."""
    if not isinstance(payload, dict):
        raise EvidenceError("ABC payload must be an object")
    out: dict = {"gpu": payload.get("gpu", "UNKNOWN"),
                 "cases": {}, "status": payload.get("status", "UNKNOWN")}
    for name in ("A_hash_staging", "B_contention", "C_barrier"):
        case = payload.get(name)
        if not isinstance(case, dict) or case.get("status") != "ok":
            out["cases"][name] = {"status": "UNKNOWN",
                                  "reason": "missing or failing case output"}
            continue
        out["cases"][name] = {"status": "ok", "metrics": {
            k: v for k, v in case.items() if k != "status"}}
    try:
        a = out["cases"]["A_hash_staging"]["metrics"]
        out["cases"]["A_hash_staging"]["metrics"]["hidden_ms"] = (
            a["serial_ms"]["p50"] - a["batched_ms"]["p50"])
    except (KeyError, TypeError):
        pass
    return out


def rank_from_evidence(closure: dict, mechanics: dict,
                       thresholds: dict | None = None) -> dict:
    """Ranking fed by PARSED closure + mechanics. HASH promotes on measured
    hidden ms; C1/C2 promotes only on measured compute share; everything
    else follows the closure-fraction gates. Never UNMEASURED when the
    required fields exist."""
    th = {"hash_hidden_ms_min": 1.0, "compute_material_fraction": 0.20}
    if thresholds:
        th.update(thresholds)
    ranking = evaluate_decisions(closure, th)
    ranking["closure_status"] = ("OK" if ranking.get("closure_ok")
                                 else "PROFILE_INCOMPLETE")
    cases = mechanics.get("cases", {})

    hidden = cases.get("A_hash_staging", {}).get("metrics", {}).get("hidden_ms")
    if hidden is not None and hidden >= th["hash_hidden_ms_min"]:
        ranking["candidates"]["HASH_EARLY_STAGING"] = {
            "rank": "PROMOTE",
            "reason": f"batched-submit hides {hidden:.3f} ms p50 "
                      f"(need >={th['hash_hidden_ms_min']}); legality closed host-side",
        }
    elif hidden is not None:
        ranking["candidates"]["HASH_EARLY_STAGING"] = {
            "rank": "HOLD",
            "reason": f"hidden {hidden:.3f} ms below {th['hash_hidden_ms_min']} ms",
        }

    efficiency = cases.get("B_contention", {}).get("metrics", {}).get("efficiency_c")
    if efficiency is not None:
        ranking["candidates"]["SHARED_OVERLAP"]["reason"] += (
            f"; measured contention c={efficiency}")

    wall = closure.get("wall_ms") or 0.0
    nested = closure.get("by_field_nested_ms", {})
    compute_share = ((nested.get("decode_ms", 0.0)
                      + nested.get("expert_compute_ms", 0.0)) / wall
                     if wall > 0 else None)
    if (ranking.get("closure_ok") and compute_share is not None
            and compute_share >= th["compute_material_fraction"]):
        ranking["candidates"]["C1_C2"] = {
            "rank": "PROMOTE",
            "reason": f"decode+compute share={compute_share:.3f} "
                      f"(need >={th['compute_material_fraction']})",
        }
    ranking["fractions"] = {"compute_share": compute_share,
                            "hidden_ms_A": hidden,
                            "efficiency_c_B": efficiency}
    return ranking


_AB_SPECS = {
    "HASH_EARLY_STAGING": {
        "mechanism": "submit L0-L2 packed reads at token start; consume gated on weights",
        "files": ["dee.cpp/scripts/deepseek_v4_layer_candidate.py",
                  "dee.cpp/src/engine.cpp (submit path only)"],
        "telemetry": ["ready_before_demand per (layer,expert)", "hidden ms/token"],
        "gates": ["exact IDs", "exact weights", "identical bytes", "identical output"],
    },
    "SHARED_OVERLAP": {
        "mechanism": "shared expert on overlapped stream; buffered exact join",
        "files": ["dee.cpp/scripts/deepseek_v4_layer_candidate.py"],
        "telemetry": ["shared host/device ms", "join order proof"],
        "gates": ["identical branch outputs", "identical join", "exact final output"],
    },
    "EVENT_HANDOFF": {
        "mechanism": "per-layer event handoff replacing full stream sync",
        "files": ["dee.cpp/src/engine.cpp", "dee.cpp/pydee/pydee.cpp"],
        "telemetry": ["wait ms eliminated", "lifetime proofs"],
        "gates": ["identical output", "no premature reuse", "deterministic completion"],
    },
    "ROUTE_D2H_NARROW": {
        "mechanism": "pinned copy + copy-event host wait instead of full sync",
        "files": ["dee.cpp/scripts/deepseek_v4_layer_candidate.py"],
        "telemetry": ["copy ms vs wait ms", "unrelated-work estimate"],
        "gates": ["identical IDs", "no missed readiness"],
    },
}

_AB_PRIORITY = ("EVENT_HANDOFF", "SHARED_OVERLAP", "ROUTE_D2H_NARROW", "HASH_EARLY_STAGING")


def build_next_ab(ranking: dict) -> dict:
    """At most ONE candidate, derived from ranking only. Never hardcoded."""
    promoted = [name for name in _AB_PRIORITY
                if ranking.get("candidates", {}).get(name, {}).get("rank") == "PROMOTE"]
    if not promoted:
        reason = "no candidate cleared its evidence gate"
        if ranking.get("closure_status") == "PROFILE_INCOMPLETE":
            reason += " (closure insufficient: PROFILE_INCOMPLETE — "
            reason += "do not select an optimization from poor attribution)"
        return {"classification": "NO_OVERLAP_OPTIMIZATION_JUSTIFIED",
                "reason": reason,
                "closure_status": ranking.get("closure_status", "UNKNOWN"),
                "ranking": {k: v.get("rank") for k, v in
                            ranking.get("candidates", {}).items()}}
    name = promoted[0]
    spec = _AB_SPECS[name]
    return {"classification": f"{name}_PROMOTED",
            "candidate": name,
            "mechanism": spec["mechanism"],
            "files_affected": spec["files"],
            "telemetry_required": spec["telemetry"],
            "correctness_gates": spec["gates"],
            "ab_shape": {"baseline": "current exact runtime",
                         "candidate": f"current exact runtime + {name} ONLY"},
            "replication": "matched off/on, same prompt/tokens/hardware; "
                           "abort on any gate failure or output mismatch",
            "runner_up": promoted[1:]}
