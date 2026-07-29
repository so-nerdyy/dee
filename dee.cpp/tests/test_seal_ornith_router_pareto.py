from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "seal_ornith_router_pareto.py"
)
SPEC = importlib.util.spec_from_file_location("seal_router_pareto", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def summary(backend: str, rate: float) -> dict:
    optimized = backend == "torch-device"
    return {
        "trial_count": 3,
        "tokens_per_second": {
            "minimum": rate - 0.1,
            "median": rate,
            "maximum": rate + 0.1,
            "all": [rate - 0.1, rate, rate + 0.1],
        },
        "all_tokens_exact": True,
        "all_40_layers_executed": True,
        "all_expert_calls_device_resident": True,
        "host_fallback_calls": 0,
        "router_native_host_calls": 0 if optimized else 480,
        "router_torch_device_calls": 480 if optimized else 0,
        "router_hidden_d2h_total_bytes": 0 if optimized else 1_000,
        "router_outputs_h2d_total_bytes": 0 if optimized else 2_000,
        "router_scalar_sync_calls": 0 if optimized else 480,
        "peak_process_vram_per_gpu_bytes": {
            "cuda:0": 7_000_000_000,
            "cuda:1": 7_100_000_000,
        },
    }


def valid_report() -> dict:
    baseline = 6.5
    optimized = 7.2
    speedup = optimized / baseline
    gates = {
        "throughput_improved_by_at_least_2_percent": True,
        "tokens_and_text_exact": True,
        "all_trace_categories_passed": True,
        "all_expert_calls_device_resident": True,
        "zero_host_fallback_calls": True,
        "optimized_router_has_zero_host_roundtrips_and_scalar_syncs": True,
        "process_vram_at_most_8_gib_per_gpu": True,
    }
    return {
        "result": "PASS",
        "git_commit": "a" * 40,
        "environment": {
            "gpus": [
                {"device": 0, "name": "Tesla T4"},
                {"device": 1, "name": "Tesla T4"},
            ]
        },
        "configuration": {
            "cache_experts_per_layer": 32,
            "split_layer": 20,
            "trials_per_backend": 3,
            "measurement_order": MODULE.EXPECTED_MEASUREMENT_ORDER,
            "prompt": "Hello",
            "greedy": True,
            "load_once": True,
            "warmup_per_backend": 1,
        },
        "correctness": {
            "native_tokens": MODULE.EXPECTED_TOKENS,
            "optimized_tokens": MODULE.EXPECTED_TOKENS,
            "generated_token_ids_exact": True,
            "decoded_text_exact": True,
            "all_trace_categories_passed": True,
        },
        "backend_summaries": {
            "native-host": summary("native-host", baseline),
            "torch-device": summary("torch-device", optimized),
        },
        "pareto": {
            "baseline_native_host_median_tps": baseline,
            "optimized_torch_device_median_tps": optimized,
            "speedup_ratio": speedup,
            "speedup_percent": (speedup - 1.0) * 100.0,
            "gates": gates,
        },
    }


def test_router_pareto_report_passes_all_gates() -> None:
    report = valid_report()
    result = MODULE.validate_report(report, "a" * 40)
    assert result["speedup_ratio"] > 1.02
    assert result["trial_count_per_backend"] == 3
    assert result["generated_token_ids"] == MODULE.EXPECTED_TOKENS
    assert result["optimized_torch_device"]["router_scalar_sync_calls"] == 0


def test_router_pareto_report_rejects_optimized_host_sync() -> None:
    report = valid_report()
    report["backend_summaries"]["torch-device"]["router_scalar_sync_calls"] = 1
    with pytest.raises(RuntimeError, match="counter is nonzero"):
        MODULE.validate_report(report, "a" * 40)
