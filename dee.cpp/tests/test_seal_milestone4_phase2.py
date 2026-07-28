from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "seal_milestone4_phase2.py"
)
SPEC = importlib.util.spec_from_file_location("seal_m4_phase2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_verifier_accepts_exact_file_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "runs" / "a.json"
    artifact.parent.mkdir()
    artifact.write_text('{"result":"PASS"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "result": "PASS",
                "artifacts": [
                    {
                        "path": "runs/a.json",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": digest(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = MODULE.verify_manifest(
        tmp_path, manifest, path_key="path", size_key="size_bytes"
    )
    assert result["verified_paths"] == 1
    artifact.write_text('{"result":"FAIL"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash changed"):
        MODULE.verify_manifest(
            tmp_path, manifest, path_key="path", size_key="size_bytes"
        )


def test_final_report_checks_all_seven_semantic_fingerprints() -> None:
    commit = "1" * 40
    contracts = MODULE.variant_contracts()
    rows = []
    for contract in contracts:
        run_id = contract["run_id"]
        rows.append(
            {
                "run_id": run_id,
                "result": "PASS",
                "git_commit": commit,
                "configuration_fingerprint_sha256": contract[
                    "configuration_fingerprint_sha256"
                ],
                "full_model_single_sequence_decode_tps": 6.5,
                "peak_process_vram_per_gpu_bytes": {
                    "cuda:0": 7_000_000_000,
                    "cuda:1": 7_100_000_000,
                },
                "correctness": {
                    "all_40_layers_executed": True,
                    "expected_tokens_exact": True,
                    "generated_token_ids": MODULE.EXPECTED_GENERATED_TOKEN_IDS[
                        run_id
                    ],
                    "device_path_share": 1.0,
                    "host_path_fallback_calls": 0,
                },
                "lifetime": {
                    "result": "NO_TRACE_ABORT",
                    "process_exit_code": 0,
                    "critical_markers": [],
                },
                "h2d_bytes_per_token": None,
                "attention_time": None,
                "expert_time": None,
                "synchronization": None,
            }
        )
    report = {
        "result": "PASS",
        "terminal_reason": "ALL_SEVEN_UNIQUE_CAP32_MATRIX_ROWS_PASS",
        "git_commit": commit,
        "selected_run_ids": list(MODULE.SEVEN_VARIANTS),
        "configuration_fingerprints_unique": True,
        "production_cache_capacity": 32,
        "explicit_diagnostic_override": {
            "run_id": "dual-cache-capacity-4",
            "cache_experts_per_layer": 4,
        },
        "rows": rows,
    }
    result = MODULE.validate_final_report(report, commit)
    assert result["row_count"] == 7
    assert result["unique_fingerprint_count"] == 7
    rows[0]["configuration_fingerprint_sha256"] = "bad"
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        MODULE.validate_final_report(report, commit)
