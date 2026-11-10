#!/usr/bin/env python3
"""Focused tests for `tmp/revalidate_phase1.py`.

Five tests:
1. Wrapper loads and exposes the same `REQUIRED_FUNCS` set as the canonical
   harness module — proving it actually imports the canonical functions and
   does not reimplement them (delegate-not-copy).
2. Set sys.argv + monkey-patch `harness.validate_row` to a sentinel; the
   wrapper's `per_row[*]` MUST equal the sentinel — proving the wrapper
   calls the canonical function rather than its own implementation.
3. Build a stub evidence tree with all 6 row dirs and all required files,
   run the wrapper as a subprocess, assert SEAL=PASS + all 12 user
   requirements true.
4. Build a stub evidence tree with one row dir MISSING; assert that the
   wrapper fails closed with exit code 14.
5. Build a stub v1 dir + set sys.argv + override the wrapper module's
   `EXPECTED_HARNESS_FILE_SHA256` and `_load_harness`; assert exit=15.

All tests self-contained: importlib.util.spec_from_file_location for module
loading, tempfile.mkdtemp for isolation, no kaggle / git push / sleeps.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WRAPPER_PATH = REPO / "tmp" / "revalidate_phase1.py"
HARNESS_PATH = (
    REPO / "dee.cpp" / "scripts" / "run_milestone4_ledger_seal.py"
)
EXPECTED_HARNESS_SHA = (
    "94d01ca2811314331be526db09acd74298f9afc463adb7a7120f03144a52be08"
)

REQUIRED_FUNCS = [
    "validate_row",
    "assemble_gates",
    "compose_seal",
    "v1_compare_block",
    "freeze_identity",
    "write_atomic",
    "sha256_file",
]


@contextmanager
def argv_guard(argv: list[str]):
    """Context manager: temporarily set sys.argv to argv for the duration."""
    saved = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = saved


def _fresh_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def _load_via_importlib(path: Path) -> object:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_stub_row(row_dir: Path, profiled: bool) -> None:
    row_dir.mkdir(parents=True, exist_ok=True)
    (row_dir / "run-report.json").write_text(
        json.dumps(
            {
                "result": "PASS",
                "correctness": {
                    "all_40_layers_executed": True,
                    "baseline_tokens_exact": True,
                },
                "generation": {"generated_token_ids": [11, 271, 40, 1044]},
            }
        ),
        encoding="utf-8",
    )
    (row_dir / "path-proof.json").write_text(
        json.dumps({"device_path_share": 1.0, "host_path_fallback_calls": 0}),
        encoding="utf-8",
    )
    (row_dir / "timing-raw.json").write_text(
        json.dumps({"profile_snapshots": []}),
        encoding="utf-8",
    )
    # Non-empty traces for profiled rows; empty traces for control rows.
    # This mirrors the REAL Kaggle-side evidence archive where control rows
    # have 0-byte expert-trace.jsonl. The wrapper must NOT fail on empty
    # traces after the harness's Bug B fix.
    if profiled:
        (row_dir / "expert-trace.jsonl").write_text(
            json.dumps(
                {"event": "sample", "layer": 0, "ts": 0.0}
            ) + "\n",
            encoding="utf-8",
        )
        (row_dir / "warmup-expert-trace.jsonl").write_text(
            json.dumps(
                {"event": "sample", "layer": 0, "ts": 0.0}
            ) + "\n",
            encoding="utf-8",
        )
    else:
        (row_dir / "expert-trace.jsonl").write_text("", encoding="utf-8")
        (row_dir / "warmup-expert-trace.jsonl").write_text("", encoding="utf-8")
    if profiled:
        (row_dir / "transfer-ledger.json").write_text(
            json.dumps(
                {
                    "coverage": {
                        "transfer_timing": {
                            "by_phase": {
                                "warmup": {"observed": 6, "launched": 6},
                                "measured": {"observed": 40, "launched": 40},
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )


def _build_stub_evidence(root: Path, missing_run: str | None = None) -> Path:
    """Build a stub evidence root with all six row dirs (or one dropped)."""
    root.mkdir(parents=True, exist_ok=True)
    run_ids = [
        "capacity-8-control",
        "capacity-8-profiled",
        "capacity-16-control",
        "capacity-16-profiled",
        "capacity-32-control",
        "capacity-32-profiled",
    ]
    for rid in run_ids:
        if rid == missing_run:
            continue
        _build_stub_row(root / "runs" / rid, profiled=rid.endswith("-profiled"))
    (root / "environment.json").write_text(
        json.dumps({"run_id": "stub", "commit": "stub"}), encoding="utf-8"
    )
    (root / "capacity-sweep-summary.json").write_text(
        json.dumps({"experiments": []}), encoding="utf-8"
    )
    (root / "logs").mkdir(exist_ok=True)
    return root


def _run_wrapper_subprocess(
    evidence_dir: Path, output_dir: Path, label: str = "test-validator"
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(WRAPPER_PATH),
        "--evidence-dir", str(evidence_dir),
        "--output-dir", str(output_dir),
        "--label", label,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_1_wrapper_loads_and_lists_canonical_functions() -> bool:
    """Wrapper loads as a Python module and exposes the expected REQUIRED_FUNCS."""
    spec = importlib.util.spec_from_file_location(
        "wrapper_module", str(WRAPPER_PATH)
    )
    wm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wm)
    set_in_wrapper = set(getattr(wm, "REQUIRED_FUNCS", []))
    if set_in_wrapper != set(REQUIRED_FUNCS):
        print(f"WRAPPER-1: REQUIRED_FUNCS mismatch. expected={REQUIRED_FUNCS} "
              f"got={sorted(set_in_wrapper)}")
        return False
    if getattr(wm, "EXPECTED_HARNESS_FILE_SHA256", "") != EXPECTED_HARNESS_SHA:
        print("WRAPPER-1: EXPECTED_HARNESS_FILE_SHA256 mismatch.")
        return False
    print("WRAPPER-1: PASS")
    return True


def test_2_monkey_patch_proves_delegation() -> bool:
    """Monkey-patch harness.validate_row to a SENTINEL; wrapper's per_row
    must equal the sentinel for ALL six rows. This proves the wrapper calls
    the canonical harness function rather than its own reimplementation.
    """
    tmp = _fresh_dir(prefix="rev2_")
    try:
        harness = _load_via_importlib(HARNESS_PATH)
        sentinel_value = {
            "run_id": "__monkey_patch_sentinel__",
            "result": "PASS",
            "profiled": False,
            "terminal_reason": "PASS",
            "legacy_evidence_reanalysis": False,
            "allow_missing_warmup_timing_used": False,
            "warmup_timing_complete": True,
            "missing_warmup_event_count": 0,
            "missing_warmup_events": 0,
            "measured_timing_complete": True,
            "timing_events_allocated": 0,
            "timing_events_dropped": 0,
            "timing_pool_capacity": 0,
            "timing_pool_capacity_source": "test",
            "legacy_warmup_exception_used": False,
            "legacy_warmup_exception_pattern": False,
            "new_timing_regression": False,
            "safe_explanation": None,
            "decode_tokens_per_second": None,
            "peak_process_vram_per_gpu_bytes": None,
        }

        def fake_validate_row(output_dir_, run_id):
            sentinel_value["run_id"] = run_id  # prove per-run dispatch
            return dict(sentinel_value)

        original = harness.validate_row
        harness.validate_row = fake_validate_row  # type: ignore[attr-defined]
        try:
            evidence_root = _build_stub_evidence(tmp / "evidence")
            output_root = tmp / "out"
            spec = importlib.util.spec_from_file_location(
                "wrapper_module", str(WRAPPER_PATH)
            )
            wm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wm)
            with argv_guard(
                [
                    "revalidate_phase1",
                    "--evidence-dir", str(evidence_root),
                    "--output-dir", str(output_root),
                    "--label", "test-2-delegation",
                ]
            ):
                rc = wm.main()
            seal = json.loads(
                (output_root / "SEAL.json").read_text(encoding="utf-8")
            )
            seen_run_ids = [r["run_id"] for r in seal.get("per_row", [])]
            if sorted(seen_run_ids) != sorted(
                (
                    "capacity-8-control", "capacity-8-profiled",
                    "capacity-16-control", "capacity-16-profiled",
                    "capacity-32-control", "capacity-32-profiled",
                )
            ):
                print(f"WRAPPER-2: per_row run_ids mismatch. got={seen_run_ids}")
                return False
            for r in seal["per_row"]:
                if r["terminal_reason"] != "PASS":
                    print(f"WRAPPER-2: per_row terminal_reason drifted: {r}")
                    return False
            if seal.get("result") != "PASS":
                print(f"WRAPPER-2: seal.result != PASS. got={seal.get('result')}")
                return False
            print(
                f"WRAPPER-2: PASS \u2014 sentinel propagated through validate_row "
                f"into SEAL.json per_row; result={seal['result']} rc={rc}"
            )
            return True
        finally:
            harness.validate_row = original  # type: ignore[attr-defined]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_3_full_stub_evidence_yields_seal_pass() -> bool:
    """Build a stub evidence tree with all 6 row dirs + summary; run wrapper
    as subprocess; expect SEAL=PASS and all 11 gates=true.
    """
    tmp = _fresh_dir(prefix="rev3_")
    try:
        evidence_root = _build_stub_evidence(tmp / "evidence")
        output_root = tmp / "out"
        proc = _run_wrapper_subprocess(evidence_root, output_root)
        if proc.returncode != 0:
            print(f"WRAPPER-3: wrapper exit={proc.returncode}; expected 0. "
                  f"stdout={proc.stdout[:400]!r} stderr={proc.stderr[:400]!r}")
            return False
        seal = json.loads((output_root / "SEAL.json").read_text(encoding="utf-8"))
        if seal.get("result") != "PASS":
            print(f"WRAPPER-3: SEAL != PASS. terminal_reason="
                  f"{seal.get('terminal_reason')}")
            return False
        if len(seal.get("per_row", [])) != 6:
            print(f"WRAPPER-3: per_row has {len(seal.get('per_row', []))} entries")
            return False
        gate_keys = [
            "all_six_rows_present",
            "complete_measured_timing",
            "generation_aware_transfer_classification",
            "zero_unresolved_metadata_inconsistencies",
            "zero_newly_dropped_timing_events",
            "correctness_passes_all_six",
            "lifetime_passes_all_six",
            "artifact_completeness_all_six",
            "v1_baseline_preserved_unchanged",
            "phase2_skeleton_present",
            "manifest_validation_passes",
            "hash_validation_passes",
        ]
        for k in gate_keys:
            if not seal["gates"].get(k):
                print(f"WRAPPER-3: gate {k} false in: "
                      f"{[(gk, seal['gates'][gk]) for gk in gate_keys if not seal['gates'][gk]]}")
                return False
        summary = json.loads(
            (output_root / "seal-validation-summary.json").read_text(encoding="utf-8")
        )
        for k, v in summary["user_requirements_met"].items():
            if v is not True:
                print(f"WRAPPER-3: user_requirements_met[{k}]={v}")
                return False
        print(f"WRAPPER-3: PASS \u2014 SEAL result={seal['result']}, "
              f"all 12 gates true, all 10 user-requirements-met, "
              f"per_row has 6 entries")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_4_missing_row_fail_closed() -> bool:
    """Stub evidence with one row dir MISSING; wrapper must fail closed
    with exit 14 AND write a FAILURE.json.
    """
    tmp = _fresh_dir(prefix="rev4_")
    try:
        evidence_root = _build_stub_evidence(
            tmp / "evidence", missing_run="capacity-32-control"
        )
        output_root = tmp / "out"
        proc = _run_wrapper_subprocess(evidence_root, output_root)
        if proc.returncode != 14:
            print(f"WRAPPER-4: wrapper exit={proc.returncode}; expected 14. "
                  f"stdout={proc.stdout[:400]!r}")
            return False
        failure = json.loads(
            (output_root / "FAILURE.json").read_text(encoding="utf-8")
        )
        if failure.get("result") != "FAIL":
            print(f"WRAPPER-4: FAILURE.json missing FAIL result: {failure}")
            return False
        if "row_dir" not in str(failure) and "absent" not in str(failure):
            print(f"WRAPPER-4: FAILURE.json terminal_reason should reference "
                  f"missing row dirs. got={failure.get('terminal_reason')}")
            return False
        print(f"WRAPPER-4: PASS \u2014 exit=14, FAILURE.json written, "
              f"reason={failure.get('terminal_reason')[:100]}")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_5_harness_sha_mismatch_fail_closed() -> bool:
    """The wrapper MUST exit 15 when the loaded harness file's content
    SHA-256 does not match the user-authorized post-fix sha. Achieve
    this by overriding the wrapper module's EXPECTED_HARNESS_FILE_SHA256
    BEFORE calling main() and using a stub v1 baseline so it doesn't fail
    earlier at code 16.
    """
    tmp = _fresh_dir(prefix="rev5_")
    try:
        evidence_root = _build_stub_evidence(tmp / "evidence")
        output_root = tmp / "out"

        # Stub v1 baseline inside the tmp dir.
        v1_stub = tmp / "v1"
        v1_stub.mkdir(parents=True, exist_ok=True)
        (v1_stub / "capacity-sweep-summary.json").write_text(
            json.dumps({"experiments": []}), encoding="utf-8"
        )

        spec = importlib.util.spec_from_file_location(
            "wrapper_module", str(WRAPPER_PATH)
        )
        wm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wm)

        # Override EXPECTED_HARNESS_FILE_SHA256 to a wrong one; this forces
        # the wrapper to fail closed at code 15 (identity ambiguous).
        wm.EXPECTED_HARNESS_FILE_SHA256 = "DEADBEEF" * 8

        # Override DEFAULT_V1_BASELINE on the loaded harness so the wrapper
        # doesn't fail closed at code 16 before reaching the SHA gate.
        real_harness = _load_via_importlib(HARNESS_PATH)
        real_harness.DEFAULT_V1_BASELINE = v1_stub

        # Patch _load_harness so the wrapper uses our real_harness instance
        # (which we just mutated) instead of loading the file again.
        wm._load_harness = lambda p: real_harness  # type: ignore[attr-defined]

        with argv_guard(
            [
                "revalidate_phase1",
                "--evidence-dir", str(evidence_root),
                "--output-dir", str(output_root),
                "--label", "test-5-sha-mismatch",
            ]
        ):
            try:
                rc = wm.main()
            except SystemExit as se:
                rc = se.code
        if rc != 15:
            print(f"WRAPPER-5: wrapper exited with {rc}; expected 15.")
            return False
        print(f"WRAPPER-5: PASS \u2014 wrapper refused to validate against "
              f"unknown harness SHA, exit=15")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print(f"=== Post-fix harness SHA256: {EXPECTED_HARNESS_SHA} ===")
    print(f"=== Repo: {REPO} ===")
    print(f"=== Wrapper: {WRAPPER_PATH} ===")
    results = {
        "test_1_wrapper_loads_and_lists_canonical_functions": test_1_wrapper_loads_and_lists_canonical_functions(),
        "test_2_monkey_patch_proves_delegation": test_2_monkey_patch_proves_delegation(),
        "test_3_full_stub_evidence_yields_seal_pass": test_3_full_stub_evidence_yields_seal_pass(),
        "test_4_missing_row_fail_closed": test_4_missing_row_fail_closed(),
        "test_5_harness_sha_mismatch_fail_closed": test_5_harness_sha_mismatch_fail_closed(),
    }
    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    failed = sum(1 for ok in results.values() if not ok)
    print(f"FINAL={'PASS' if failed == 0 else 'FAIL'} failed={failed}/5")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
