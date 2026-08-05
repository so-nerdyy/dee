from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.seal_deepseek_v4_ds10 import stage_gates, validate_manifest


def _trace(n_layers: int) -> list[dict]:
    return [{"layer": layer, "order": layer, "finite": True,
             "selected_experts": [[0, 1, 2, 3, 4, 5]]}
            for layer in range(n_layers)]


def _common(stage: str, verdict: str, gates: dict) -> dict:
    return {"stage": stage, "verdict": verdict, "cuda_available": True,
            "gpu_count": 2, "performance_comparable": False, "gates": gates}


def test_v1_gate_uses_canonical_coverage_fields() -> None:
    evidence = _common("v1", "ACCEPT_MODEL_LOAD", {
        "coverage_audit": True,
        "coverage": {"all_resolved": True, "tensor_count": 72_317,
                     "shard_count": 48},
        "memory_ceilings_ok": True,
        "model_load_ok": True,
        "model_load": {"state_count": 43, "actual_memory_ceilings_ok": True},
    })
    assert all(stage_gates(evidence, "v1").values())
    evidence["gates"]["coverage"].pop("all_resolved")
    evidence["gates"]["coverage"]["ok"] = True
    assert stage_gates(evidence, "v1")["coverage"] is False


def test_final_gate_requires_16_complete_layer_traces() -> None:
    tokens = list(range(16))
    token_trace = {f"token_{idx}": {"layers": _trace(43)}
                   for idx in range(16)}
    evidence = _common("final", "ACCEPT_DUAL_T4_DECODE", {
        "tokens": tokens, "token_count": 16, "token_ids_in_vocab": True,
        "token_trace": token_trace, "deterministic_rerun": True,
        "cold_warm_equal": True, "cache_capacity_variation_equal": True,
    })
    assert all(stage_gates(evidence, "final").values())
    token_trace["token_7"]["layers"].pop()
    assert stage_gates(evidence, "final")["token_layers"] is False


def test_manifest_rehashes_every_artifact(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    payload = evidence / "payload.json"
    payload.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {
        "required_paths": ["payload.json"],
        "artifacts": [{"path": "payload.json", "bytes": payload.stat().st_size,
                       "sha256": digest}],
    }
    (evidence / "ds10-artifact-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    assert validate_manifest(evidence)["artifact_count"] == 1
    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        validate_manifest(evidence)
