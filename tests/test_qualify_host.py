#!/usr/bin/env python3
"""Tests for tools/qualify_host.py (deterministic structure + claim discipline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import qualify_host  # noqa: E402
from qualify_host import (  # noqa: E402
    build_cost_model,
    build_verdicts,
    cuda_feature_availability,
    probe_system,
)


def _fake_system():
    return {
        "cpu": {"cpu_model": "Fake CPU", "status": "measured"},
        "counts": {"logical": 8, "physical_cores": 4, "status": "measured"},
        "isa": {"status": "measured", "avx2": True},
        "ram": {"status": "measured", "total_bytes": 32 * 1024 ** 3,
                "available_bytes": 24 * 1024 ** 3},
    }


def _fake_budget():
    return {"vram_slots": 623, "ram_slots": 1927, "total_experts": 11008,
            "cache_coverage_fraction": 0.23, "expert_bytes_mxfp4": 13369344}


def test_feature_availability_derived_not_measured():
    feat = cuda_feature_availability("12.0")
    assert feat["status"] == "derived"
    assert feat["fp4_tensor_cores"] is True
    assert "do NOT imply native FP4 execution" in feat["note"]
    old = cuda_feature_availability("7.5")
    assert old["bf16_cuda"] is False and old["fp4_tensor_cores"] is False
    assert cuda_feature_availability(None)["status"] == "unknown"
    assert cuda_feature_availability("bogus")["status"] == "unknown"


def test_verdicts_separate_unknown_and_make_no_tps_claim():
    gpu = {"cuda": {"available": False, "reason": "fake"},
           "pcie": {"status": "unknown", "reason": "fake"},
           "features_by_device": []}
    verdicts, unknown = build_verdicts(_fake_system(), gpu, None, None, _fake_budget())
    assert isinstance(unknown, list) and len(unknown) >= 2
    blob = json.dumps(verdicts).lower()
    # No throughput *prediction*: the disclaimer may name tok/s only to refuse it.
    assert "tokens per second" not in blob
    assert "decode_tokens_per_s" not in blob
    assert "20 tps feasibility as proven" not in blob
    assert "not claimed here" in blob
    assert "UNPROVEN" in verdicts["no_performance_prediction"]
    assert "5070" in verdicts["no_performance_prediction"]


def test_verdicts_with_measured_transfer():
    mib = 1024 * 1024
    transfer = {"results": {"pinned_h2d": [
        {"size_bytes": 13 * mib, "size_mib": 13, "status": "measured",
         "latency_ms_median": 1.5, "bandwidth_gbps": 9.0}]}}
    gpu = {"cuda": {"available": True,
                    "devices": [{"free_vram_bytes": 10 * 1024 ** 3}]},
           "pcie": {"status": "measured"},
           "features_by_device": [{"index": 0, "name": "FakeGPU",
                                   "bf16_cuda": True, "fp8_tensor_cores": True,
                                   "fp4_tensor_cores": False}]}
    storage = {"disk": {"free_bytes": 500 * 1024 ** 3}}
    verdicts, _ = build_verdicts(_fake_system(), gpu, storage, transfer, _fake_budget())
    assert "measured pinned-H2D" in verdicts["h2d_cost_scheduler_input"]
    assert "not a TPS claim" in verdicts["h2d_cost_scheduler_input"]


def test_cost_model_export_only():
    cost = build_cost_model(None, None, _fake_system(),
                            {"cuda": {"available": False}, "pcie": {}})
    assert cost["integration_status"] == "EXPORT_ONLY_NOT_INTEGRATED"
    assert cost["gpu_execution"] is None
    assert "Not wired into scheduling" in cost["gpu_execution_note"]
    assert cost["ssd"]["status"] == "UNKNOWN"


def test_cost_model_t_h2d_from_measured():
    mib = 1024 * 1024
    transfer = {"results": {"pinned_h2d": [
        {"size_bytes": 13 * mib, "status": "measured", "latency_ms_median": 2.0}]}}
    cost = build_cost_model(transfer, None, _fake_system(), {})
    assert cost["t_h2d_ms_pinned_by_size_bytes"] == {str(13 * mib): 2.0}
    assert cost["t_h2d_source"] == "measured pinned_h2d median"


def test_probe_system_never_fabricates():
    system = probe_system()
    for section in ("cpu", "isa", "ram"):
        assert system[section]["status"] in ("measured", "unknown")
        if system[section]["status"] == "unknown":
            assert "reason" in system[section]
    json.dumps(system)


def test_qualify_structure_skipping_probes(tmp_path):
    report = qualify_host.qualify(tmp_path, run_storage=False, run_h2d=False,
                                  file_mib=1, blocks="256KiB", storage_iters=1,
                                  random_ops=1, h2d_sizes=[1.0], h2d_iters=1,
                                  h2d_warmup=0, vram_override=None,
                                  reserved_override=None, ram_cache_override=None)
    assert set(report) >= {"tool", "version", "measured", "derived", "unknown",
                           "cost_model", "claim_tiers"}
    assert set(report["measured"]) == {"system", "gpu", "storage", "transfer"}
    assert report["measured"]["storage"] is None
    assert "derived" in report and "memory_budget_mxfp4" in report["derived"]
    assert isinstance(report["unknown"], list)
    blob = json.dumps(report).lower()
    assert "tokens per second" not in blob
    json.dumps(report)
