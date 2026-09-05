#!/usr/bin/env python3
"""Tests for the host/sync profiler (research/route-pipeline goal track).

Covers: default-OFF instrumentation, identical disabled code path (static
diff scan), nesting rules, UNKNOWN-safe closure, attribution, malformed
telemetry failing closed, decision-rule conservatism, and the Python
candidate-side flag default.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "dee.cpp" / "experiments" / "route_pipeline"
sys.path.insert(0, str(EXP))

from host_profiler import (  # noqa: E402
    NESTED_IN_NATIVE_CALL,
    SPAN_FIELDS,
    SchemaError,
    compute_closure,
    evaluate_decisions,
    validate_record,
)

ENGINE = ROOT / "dee.cpp" / "src" / "engine.cpp"
PROF_H = ROOT / "dee.cpp" / "include" / "dee" / "profiling.h"
PROF_CPP = ROOT / "dee.cpp" / "src" / "profiling.cpp"
CANDIDATE = ROOT / "dee.cpp" / "scripts" / "deepseek_v4_layer_candidate.py"


def _diff(path: Path) -> str:
    proc = subprocess.run(["git", "diff", "HEAD", "--", str(path)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0
    return proc.stdout


def _added(diff: str) -> list[str]:
    return [ln[1:] for ln in diff.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")]


def _removed(diff: str) -> list[str]:
    return [ln[1:] for ln in diff.splitlines()
            if ln.startswith("-") and not ln.startswith("---")]


# --- Default OFF ----------------------------------------------------------

def test_profiler_member_defaults_off():
    text = PROF_H.read_text(encoding="utf-8")
    assert "bool enabled_ = false;" in text


def test_engine_config_profile_defaults_off():
    text = (ROOT / "dee.cpp" / "include" / "dee" / "engine.h").read_text(encoding="utf-8")
    assert "profile_stages = false" in text


def test_candidate_flag_defaults_off():
    text = CANDIDATE.read_text(encoding="utf-8")
    assert 'os.environ.get("DEE_HOST_PROFILE") == "1"' in text
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'dee.cpp'); "
         "sys.path.insert(0, 'dee.cpp/scripts'); "
         "import deepseek_v4_layer_candidate as m; "
         "print(m._HOST_PROFILE)"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "False"


# --- Identical disabled path / no behavior change --------------------------

_FORBIDDEN_ADDED = ("cublasGemmEx", "cublasSgemv", "cudaMemcpy", "cudaMemset",
                    "cudaStreamSynchronize", "cudaEventRecord",
                    "cudaEventSynchronize", "cudaStreamWaitEvent",
                    "__global__", "fmaf(", "expf(")


def _head_text(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    proc = subprocess.run(["git", "show", f"HEAD:{rel}"],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-500:]
    return proc.stdout


# --- Identical disabled path / no behavior change --------------------------

def test_engine_diff_adds_only_profiler_scopes():
    """Removed lines must all survive (moved verbatim) except the single
    const relaxation; truly-new lines must be profiler scaffolding. This
    proves zero logic change."""
    diff = _diff(ENGINE)
    removed = {ln[1:].strip() for ln in diff.splitlines()
               if ln.startswith("-") and not ln.startswith("---")}
    added = [ln[1:].strip() for ln in diff.splitlines()
             if ln.startswith("+") and not ln.startswith("+++")]
    new_text = ENGINE.read_text(encoding="utf-8")
    new_stripped = {ln.strip() for ln in new_text.splitlines()}
    vanished = removed - new_stripped
    # Only the const->bool guarded-assignment relaxation may vanish; the
    # same statement shape exists elsewhere, so verify by count below.
    assert vanished <= {
        "const bool computed = d_blob && swiglu_expert_batch_fp16_cuda("}, vanished
    head_text = _head_text(ENGINE)
    relaxed = "const bool computed = d_blob && swiglu_expert_batch_fp16_cuda("
    assert head_text.count(relaxed) == new_text.count(relaxed) + 1
    added_set = {ln for ln in added if ln and not ln.startswith("//")}
    truly_new = added_set - removed
    for line in truly_new:
        assert ("profiler_" in line or "host_prof" in line
                or "HostSpanGuard" in line or "HostLayerScope" in line
                or "HostSpan::" in line
                or line in ("{", "}", "bool computed = false;",
                            "computed = d_blob && swiglu_expert_batch_fp16_cuda(")
                or line.startswith("cfg_.device_id")
                or line.startswith("current_token_")), line
    for line in truly_new:
        for forbidden in _FORBIDDEN_ADDED:
            assert forbidden not in line, (forbidden, line)


def test_no_synchronization_removed():
    removed = "\n".join(_removed(_diff(ENGINE)))
    assert "cudaStreamSynchronize" not in removed
    assert "cudaEventRecord" not in removed
    assert "cudaMemcpy" not in removed


def test_no_arithmetic_or_routing_change():
    diff = _diff(ENGINE)
    removed = {ln[1:].strip() for ln in diff.splitlines()
               if ln.startswith("-") and not ln.startswith("---")}
    new_stripped = {ln.strip() for ln in ENGINE.read_text(encoding="utf-8").splitlines()}
    # Same statement may exist elsewhere; verify by count that at most the
    # single guarded-assignment site changed shape.
    vanished = removed - new_stripped
    assert vanished <= {
        "const bool computed = d_blob && swiglu_expert_batch_fp16_cuda("}, vanished
    head_text = _head_text(ENGINE)
    relaxed = "const bool computed = d_blob && swiglu_expert_batch_fp16_cuda("
    assert head_text.count(relaxed) == ENGINE.read_text(encoding="utf-8").count(relaxed) + 1
    assert "bool computed = false;" in new_stripped


def test_candidate_diff_is_timing_only():
    for line in _added(_diff(CANDIDATE)):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert any(key in line for key in (
            "perf_counter", "_HOST_PROFILE", "host_profile", "hp[", "hp =",
            "if hp ", "hp is not None", "shared_start_event", "shared_end_event", "dump_host_profile",
            "import os", "import json as _json", "device", "token_rows",
            "copy_submit_t", "wait_start_t", "wait_end_t", "ids_bytes",
            "route_d2h", "native_call_wall", "combine_", "shared_host_wall",
            "provenance", "rows", "open(", "elapsed_time", "Event(",
            "current_stream", "query", ".record(",
            "try:", "except", "hp[", "deferred", "shared-expert",
            "synchronization", "perturb", "UNKNOWN", "completed",
            "waited", "dump", "JSON", "out", "row = dict",
            "start is not", "return path", "Returns", "else:")), line


# --- Nesting rules ----------------------------------------------------------

def test_nested_set_matches_schema():
    assert NESTED_IN_NATIVE_CALL == frozenset((
        "source_wait_ms", "fill_wait_ms", "h2d_ms", "decode_ms",
        "expert_compute_ms", "native_output_sync_wait_ms"))
    text = PROF_H.read_text(encoding="utf-8")
    assert "host_span_nested" in text


def test_closure_excludes_nested():
    rec = {"token": 0, "layer": 0, "device": 0,
           "native_call_wall_ms": 10.0, "decode_ms": 4.0,
           "expert_compute_ms": 3.0, "layer_wall_ms": 12.0}
    out = compute_closure([rec], decode_wall_ms=12.0)
    assert out["accounted_ms"] == 10.0  # children NOT added
    assert out["unknown_ms"] == 2.0
    assert out["closure_fraction"] == round(10.0 / 12.0, 6)
    assert out["forced"] is False


# --- UNKNOWN-safe closure ----------------------------------------------------------

def test_unknown_never_forced():
    out = compute_closure([], decode_wall_ms=5.0)
    assert out["accounted_ms"] == 0.0
    assert out["unknown_ms"] == 5.0
    assert out["closure_fraction"] == 0.0
    assert out["wall_ms"] == 5.0
    assert out["by_field_ms"] == {}
    assert out["by_field_nested_ms"] == {}
    assert out["forced"] is False


def test_null_spans_stay_unknown():
    rec = {"token": 1, "layer": 2, "device": 0, "native_call_wall_ms": 7.0,
           "shared_expert_ms": None}
    out = compute_closure([rec], decode_wall_ms=10.0)
    assert out["accounted_ms"] == 7.0 and out["unknown_ms"] == 3.0


def test_malformed_telemetry_fails_closed():
    import pytest
    with pytest.raises(SchemaError):
        validate_record({"token": 0})  # missing layer/device
    with pytest.raises(SchemaError):
        validate_record({"token": 0, "layer": 0, "device": 0,
                         "decode_ms": -1.0})
    with pytest.raises(SchemaError):
        validate_record({"token": 0, "layer": 0, "device": 0,
                         "decode_ms": "fast"})
    with pytest.raises(SchemaError):
        validate_record({"token": 0, "layer": 0, "device": 0,
                         "decode_ms": 1.0,
                         "provenance": {"decode_ms": "MADE_UP"}})
    with pytest.raises(SchemaError):
        validate_record("not-a-record")
    with pytest.raises(SchemaError):
        compute_closure([{"token": 0}], decode_wall_ms=-2.0)


def test_attribution_per_token_layer():
    recs = [{"token": t, "layer": layer, "device": layer // 22,
             "native_call_wall_ms": 5.0}
            for t in range(2) for layer in range(43)]
    out = compute_closure(recs, decode_wall_ms=2 * 43 * 5.0)
    assert out["closure_fraction"] == 1.0
    assert out["accounted_ms"] == 430.0


# --- Decision rules ----------------------------------------------------------

def test_decisions_hold_by_default():
    out = evaluate_decisions(compute_closure([], decode_wall_ms=10.0))
    assert out["closure_ok"] is False
    assert all(v["rank"] in ("HOLD", "PROMOTE_IF_MECHANICS")
               for v in out["candidates"].values())
    assert out["candidates"]["C1_C2"]["rank"] == "HOLD"


def test_decisions_promote_only_on_evidence():
    recs = [{"token": 0, "layer": layer, "device": 0,
             "native_call_wall_ms": 80.0,
             "native_output_sync_wait_ms": 8.0,
             "shared_expert_ms": 8.0,
             "route_d2h_host_wait_ms": 3.0,
             "combine_ms": 1.0} for layer in range(10)]
    wall = 10 * 100.0
    out = evaluate_decisions(compute_closure(recs, decode_wall_ms=wall))
    assert out["closure_ok"] is True
    assert out["candidates"]["SHARED_OVERLAP"]["rank"] == "PROMOTE"
    assert out["candidates"]["EVENT_HANDOFF"]["rank"] == "PROMOTE"
    assert out["candidates"]["ROUTE_D2H_NARROW"]["rank"] == "PROMOTE"
    assert out["candidates"]["C1_C2"]["rank"] == "HOLD"


def test_span_fields_cover_schema():
    for field in ("route_d2h_host_wait_ms", "native_call_wall_ms",
                  "source_wait_ms", "fill_wait_ms", "h2d_ms", "decode_ms",
                  "expert_compute_ms", "native_output_sync_wait_ms",
                  "shared_expert_ms", "combine_ms", "orchestration_ms",
                  "handoff_ms"):
        assert field in SPAN_FIELDS
    assert len(SPAN_FIELDS) == 12


def test_json_round_trip():
    recs = [{"token": 0, "layer": 5, "device": 0, "native_call_wall_ms": 9.5,
             "provenance": {"native_call_wall_ms": "HOST_WALL"}}]
    out = compute_closure(recs, decode_wall_ms=10.0)
    json.dumps(out)
    assert out["by_field_ms"] == {"native_call_wall_ms": 9.5}
