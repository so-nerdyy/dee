"""Self-contained unit test for Phase 1 pre-launch verification.

Exercises:
  * run_milestone4_ledger_seal.py — full PASS/FAIL cascade of validate_row
    + Rule-3 schema fields + 3 branches (PASS, legacy-pattern, new-regression)
    + run-id typo guard.
  * tmp/m4_supervisor.py — the FAIL-path of `_stage_kernel`
    (literal misscount + stage cleanup invariant).

Each profiled-row fixture writes ALL SIX sibling files the production
cascade reads:

    run-report.json           (correctness + lifetime + generation fields)
    path-proof.json           (device_path_share=1.0, host_fallback=0)
    expert-trace.jsonl        (non-empty)
    warmup-expert-trace.jsonl (non-empty)
    timing-raw.json           (timing_events_allocated / dropped)
    transfer-ledger.json      (by-phase warmup / measured observed/launched)

NO NETWORK, NO KAGGLE.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import uuid
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "dee.cpp" / "scripts" / "run_milestone4_ledger_seal.py"
SUPER = REPO / "tmp" / "m4_supervisor.py"

EXPECTED_GENERATED_TOKEN_IDS = [11, 271, 40, 1044]


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _build_minimal_row(
    output_dir: Path,
    *,
    run_id: str,
    warmup_launched: int,
    warmup_observed: int,
    measured_launched: int,
    measured_observed: int,
    timing_allocated: int,
    timing_dropped: int,
    decode_tps: float = 6.5,
) -> None:
    """Write the six sibling files for one profiled row so the production
    cascade can pass all six gates."""
    row_dir = output_dir / "runs" / run_id
    row_dir.mkdir(parents=True, exist_ok=True)

    run_report = {
        "run_id": run_id,
        "result": "PASS",
        "correctness": {
            "passed": True,
            "all_layers": True,
            "all_40_layers_executed": True,
            "baseline_tokens_exact": True,
            "router_parity": True,
        },
        "lifetime": {
            "no_trace_abort": True,
            "invalid_lifetime_ops": 0,
        },
        "generation": {
            "generated_token_ids": list(EXPECTED_GENERATED_TOKEN_IDS),
            "max_new_tokens": 4,
            "single_stream_decode_tokens_per_second": decode_tps,
        },
        "peak_process_vram_per_gpu_bytes": 7_500_000_000,
    }
    (row_dir / "run-report.json").write_text(
        json.dumps(run_report, indent=2), encoding="utf-8"
    )

    path_proof = {
        "device_path_share": 1.0,
        "host_path_fallback_calls": 0,
        "summary": "100% device execution; zero host fallback",
    }
    (row_dir / "path-proof.json").write_text(
        json.dumps(path_proof, indent=2), encoding="utf-8"
    )

    expert_event = (
        json.dumps({
            "step": 0,
            "expert_id": 0,
            "cache_state": "resident",
        })
        + "\n"
    )
    (row_dir / "expert-trace.jsonl").write_text(
        expert_event, encoding="utf-8"
    )
    (row_dir / "warmup-expert-trace.jsonl").write_text(
        expert_event, encoding="utf-8"
    )

    timing_raw = {
        "profile_snapshots": [
            {
                "layers": [
                    {
                        "profile": {
                            "operations": {
                                "timing_events_allocated": timing_allocated,
                                "timing_events_dropped": timing_dropped,
                            }
                        }
                    }
                ]
            }
        ]
    }
    (row_dir / "timing-raw.json").write_text(
        json.dumps(timing_raw, indent=2), encoding="utf-8"
    )

    transfer_ledger = {
        "coverage": {
            "transfer_timing": {
                "by_phase": {
                    "warmup": {
                        "launched": warmup_launched,
                        "observed": warmup_observed,
                    },
                    "measured": {
                        "launched": measured_launched,
                        "observed": measured_observed,
                    },
                }
            }
        },
        "classification": {
            "compulsory_cold_miss": 0,
            "capacity_miss": 0,
            "policy_miss": 0,
            "duplicate_transfer": 0,
            "stale_prefetch": 0,
            "unused_prefetch": 0,
            "premature_eviction": 0,
            "metadata_inconsistency": 0,
            "wrong_gpu_transfer": 0,
            "unclassified": 0,
        },
    }
    (row_dir / "transfer-ledger.json").write_text(
        json.dumps(transfer_ledger, indent=2), encoding="utf-8"
    )


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1: PASS row — both phases complete.
# ---------------------------------------------------------------------------


def test_harness_rule3_pass_row() -> bool:
    h = _load(HARNESS)
    td = Path(tempfile.mkdtemp(prefix="r3_pass_"))
    _build_minimal_row(
        td,
        run_id="capacity-8-profiled",
        warmup_launched=12,
        warmup_observed=12,
        measured_launched=40,
        measured_observed=40,
        timing_allocated=16,
        timing_dropped=0,
    )
    res = h.validate_row(td, "capacity-8-profiled")
    print(
        f"PASS-row: status={res.get('status')!r} "
        f"terminal_reason={res.get('terminal_reason')!r}"
    )
    ok = (
        res.get("result") == "PASS"
        and res.get("warmup_timing_complete") is True
        and res.get("measured_timing_complete") is True
        and res.get("missing_warmup_event_count") == 0
        and res.get("new_timing_regression") is False
        and res.get("timing_events_allocated") == 16
        and res.get("timing_events_dropped") == 0
        and res.get("timing_pool_capacity") == 16
        and res.get("timing_pool_capacity_source")
        == "derived_from_allocated_plus_dropped"
        and res.get("legacy_warmup_exception_used") is False
    )
    if not ok:
        print(f"  detailed fields: {json.dumps(res, sort_keys=True)}")
    return ok


# ---------------------------------------------------------------------------
# Test 2: Legacy-pattern row — warmup incomplete, measured complete.
# ---------------------------------------------------------------------------


def test_harness_rule3_legacy_pattern_row() -> bool:
    h = _load(HARNESS)
    td = Path(tempfile.mkdtemp(prefix="r3_legacy_"))
    _build_minimal_row(
        td,
        run_id="capacity-16-profiled",
        warmup_launched=12,
        warmup_observed=6,
        measured_launched=40,
        measured_observed=40,
        timing_allocated=12,
        timing_dropped=0,
    )
    res = h.validate_row(td, "capacity-16-profiled")
    print(
        f"LEGACY-row: status={res.get('status')!r} "
        f"terminal_reason={res.get('terminal_reason')!r}"
    )
    safe = res.get("safe_explanation") or ""
    ok = (
        res.get("result") == "PASS"
        and res.get("warmup_timing_complete") is False
        and res.get("measured_timing_complete") is True
        and res.get("missing_warmup_event_count") == 6
        and res.get("legacy_warmup_exception_used") is True
        and res.get("new_timing_regression") is False
        and res.get("timing_pool_capacity") == 12
        and isinstance(safe, str)
        and "legacy" in safe.lower()
    )
    if not ok:
        print(f"  detailed fields: {json.dumps(res, sort_keys=True)}")
    return ok


# ---------------------------------------------------------------------------
# Test 3: New-regression row — both phases incomplete → fail closed.
# ---------------------------------------------------------------------------


def test_harness_rule3_new_regression_row() -> bool:
    h = _load(HARNESS)
    td = Path(tempfile.mkdtemp(prefix="r3_reg_"))
    _build_minimal_row(
        td,
        run_id="capacity-32-profiled",
        warmup_launched=12,
        warmup_observed=8,
        measured_launched=40,
        measured_observed=37,
        timing_allocated=8,
        timing_dropped=0,
    )
    res = h.validate_row(td, "capacity-32-profiled")
    print(
        f"REG-row: status={res.get('status')!r} "
        f"terminal_reason={res.get('terminal_reason')!r}"
    )
    reason = res.get("terminal_reason") or ""
    ok = (
        res.get("result") == "FAIL"
        and res.get("new_timing_regression") is True
        and res.get("warmup_timing_complete") is False
        and res.get("measured_timing_complete") is False
        and "newly" in reason.lower()
    )
    if not ok:
        print(f"  detailed fields: {json.dumps(res, sort_keys=True)}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: Unknown run_id → guard raises ValueError before any IO.
# ---------------------------------------------------------------------------


def test_harness_unknown_run_id() -> bool:
    h = _load(HARNESS)
    td = Path(tempfile.mkdtemp(prefix="r3_uid_"))
    try:
        h.validate_row(td, "capacity-32-profild")  # typo'd
    except ValueError as exc:
        msg = str(exc)
        ok = (
            "validate_row called with unknown run_id" in msg
            and "capacity-32-profiled" in msg
        )
        print(f"UNKNOWN-run: raised cleanly: {ok}")
        return ok
    print("UNKNOWN-run: did not raise")
    return False


# ---------------------------------------------------------------------------
# Test 5: Supervisor `_stage_kernel` FAIL-path on missing literal + cleanup.
# ---------------------------------------------------------------------------


def test_supervisor_literal_staging_misscount() -> bool:
    s = _load(SUPER)
    fake = Path(tempfile.mkdtemp(prefix="sup_fake_"))
    src = fake / "src"
    src.mkdir()
    # Production reads `ornith_milestone4.py` from the staged tree.
    (src / "ornith_milestone4.py").write_text(
        "print('no staging literals here')\n", encoding="utf-8"
    )
    test_run_id = "test-staging-" + uuid.uuid4().hex[:8]
    expected_stage = s.ROOT / ("m4_kernel_stage_" + test_run_id)
    try:
        s._stage_kernel(
            kernel_dir=src,
            run_id=test_run_id,
            integration_sha="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            integration_branch="freeze/phase1-integration-testaaaa",
            harness_nonce="nonce-123",
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        textual_ok = (
            "literal" in msg or "override" in msg or "missing" in msg
        )
        cleanup_ok = not expected_stage.exists()
        ok = textual_ok and cleanup_ok
        print(
            f"SUPERVISOR-misscount: textual_ok={textual_ok} "
            f"cleanup_ok={cleanup_ok}"
        )
        return ok
    print("SUPERVISOR-misscount: did not raise")
    return False


def main() -> int:
    results = {
        "harness_rule3_pass_row": test_harness_rule3_pass_row(),
        "harness_rule3_legacy_pattern_row": test_harness_rule3_legacy_pattern_row(),
        "harness_rule3_new_regression_row": test_harness_rule3_new_regression_row(),
        "harness_unknown_run_id": test_harness_unknown_run_id(),
        "supervisor_literal_staging_misscount": test_supervisor_literal_staging_misscount(),
    }
    print()
    print("--- summary ---")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_pass = all(results.values())
    print(f"FINAL={'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
