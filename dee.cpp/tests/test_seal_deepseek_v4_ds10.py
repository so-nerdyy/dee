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


def _host_memory(current: int = 4_000_000_000,
                 peak: int = 7_000_000_000) -> dict:
    return {"current_rss_bytes": current, "peak_rss_bytes": peak,
            "ceiling_bytes": 12_000_000_000}


def _build_memory(cache_budget: int) -> dict:
    return {"cuda0_reserved_gib": 6.0, "cuda1_reserved_gib": 6.0,
            "cache_budget_bytes": cache_budget,
            "host_memory": _host_memory()}


def _runtime() -> dict:
    cache = {"resident_bytes": 1 << 30, "fallbacks": 0,
             "checksum_failures": 0}
    return {"backends": {"cpu_expert_execution": False,
                          "routed_experts": "freebuff_ds8_cache_fp16_cuda"},
            "cache0": dict(cache), "cache1": dict(cache)}


def _generation_memory() -> dict:
    after_release = _host_memory(current=2_000_000_000)
    return {
        "memory": _build_memory(2 << 30),
        "memory2": _build_memory(1536 << 20),
        "host_memory_after_primary": _host_memory(),
        "allocator_release": {"attempted": True,
                              "before": _host_memory(),
                              "after": after_release},
        "host_memory_after_release": after_release,
        "host_memory_after_alternate": _host_memory(),
        "peak_memory_primary_gib": {"cuda0": 8.0, "cuda1": 9.0},
        "peak_memory_alternate_gib": {"cuda0": 8.0, "cuda1": 9.0},
        "generation_memory_ceilings_ok": True,
        "runtime_after_warm": _runtime(),
        "runtime_alternate_budget": _runtime(),
        "alternate_cache_budget_bytes": 1536 << 20,
    }


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
        "warm_tokens": tokens, "rerun_tokens": tokens,
        **_generation_memory(),
    })
    assert all(stage_gates(evidence, "final").values())
    token_trace["token_7"]["layers"].pop()
    assert stage_gates(evidence, "final")["token_layers"] is False


def test_generation_gate_rejects_recorded_host_memory_over_ceiling() -> None:
    memory = _generation_memory()
    memory["memory2"]["host_memory"] = _host_memory(
        current=13_977_284_608, peak=17_153_708_032)
    evidence = _common("v5", "ACCEPT_DUAL_T4_FIRST_TOKEN", {
        "tokens": [666], "token_count": 1, "token_ids_in_vocab": True,
        "token_trace": {"token_0": {"layers": _trace(43)}},
        "deterministic_rerun": True, "cold_warm_equal": True,
        "cache_capacity_variation_equal": True,
        "warm_tokens": [666], "rerun_tokens": [666], **memory,
    })
    assert stage_gates(evidence, "v5")["generation_memory"] is False


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
