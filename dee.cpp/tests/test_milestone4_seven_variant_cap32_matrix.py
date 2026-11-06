from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_milestone4_seven_variant_cap32_matrix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_milestone4_seven_variant_cap32_matrix",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
PHASE2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE2)


def test_exact_seven_variants_have_unique_semantic_fingerprints() -> None:
    contracts = PHASE2.variant_contracts()

    assert [item["run_id"] for item in contracts] == list(PHASE2.SEVEN_VARIANTS)
    assert len(contracts) == 7
    assert len({
        item["configuration_fingerprint_sha256"] for item in contracts
    }) == 7
    assert all(
        len(item["configuration_fingerprint_sha256"]) == 64
        for item in contracts
    )


def test_cap32_production_rows_preserve_explicit_cap4_diagnostic() -> None:
    configurations = PHASE2.semantic_configurations()

    assert all(
        configurations[run_id]["cache_experts_per_layer"] == 32
        for run_id in PHASE2.CAP32_VARIANTS
    )
    cap4 = configurations["dual-cache-capacity-4"]
    assert cap4["cache_experts_per_layer"] == 4
    assert cap4["allow_sub_topk_cache"] is True


def test_duplicate_semantic_configuration_fails_closed(monkeypatch) -> None:
    configurations = PHASE2.semantic_configurations()
    configurations["dual-long-prompt"] = dict(
        configurations["dual-cache-disabled"]
    )
    monkeypatch.setattr(
        PHASE2,
        "semantic_configurations",
        lambda: configurations,
    )

    with pytest.raises(RuntimeError, match="duplicate semantic configurations"):
        PHASE2.variant_contracts()


def test_phase1_gate_requires_terminal_pass(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="SEAL.json not found"):
        PHASE2.gate(tmp_path)

    (tmp_path / "SEAL.json").write_text(json.dumps({
        "result": "FAIL",
        "terminal_reason": "fixture failure",
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="fixture failure"):
        PHASE2.gate(tmp_path)

    (tmp_path / "SEAL.json").write_text(json.dumps({
        "result": "PASS",
        "identity": {"run_id": "phase1-fixture"},
        "source_seal_sha256": PHASE2.CANONICAL_PHASE1_SEAL_SHA256,
    }), encoding="utf-8")
    assert PHASE2.gate(tmp_path)["result"] == "PASS"


def test_lifetime_gate_accepts_clean_exit_and_rejects_abort(
    tmp_path: Path,
) -> None:
    log = tmp_path / "row.log"
    log.write_text("generation PASS\ncleanup PASS\n", encoding="utf-8")
    lifetime = PHASE2.validate_lifetime_log(log, "row")
    assert lifetime["result"] == "NO_TRACE_ABORT"
    assert len(lifetime["log_sha256"]) == 64

    log.write_text(
        "generation PASS\n[ta_free_DOUBLE_FREE_ABORT] ptr=0x1\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lifetime gate failed"):
        PHASE2.validate_lifetime_log(log, "row")


def test_runtime_configuration_validation_fails_on_capacity_drift() -> None:
    expected = PHASE2.semantic_configurations()["dual-warm-control"]
    actual = {
        field: expected[field]
        for field in (
            "classification",
            "prompt",
            "max_new_tokens",
            "warmup_generation",
            "cache_experts_per_layer",
            "cache_disabled",
            "profile_enabled",
            "trace_requests",
            "profile_timeline",
            "reference_parity",
        )
    }
    PHASE2.validate_runtime_configuration(actual, expected, "dual-warm-control")

    actual["cache_experts_per_layer"] = 8
    with pytest.raises(RuntimeError, match="configuration drifted"):
        PHASE2.validate_runtime_configuration(
            actual,
            expected,
            "dual-warm-control",
        )
