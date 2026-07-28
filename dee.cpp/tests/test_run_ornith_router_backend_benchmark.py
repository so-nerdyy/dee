from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_ornith_router_backend_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("router_backend_benchmark", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(rate: float, device_calls: int = 160) -> dict:
    return {
        "tokens_per_second": rate,
        "per_token_decode_seconds": [0.1, 0.11, 0.09],
        "peak_vram_bytes": {"cuda:0": 7_000_000_000, "cuda:1": 7_100_000_000},
        "tokens_exact": True,
        "all_40_layers_executed": True,
        "path_proof": {
            "device_path_share": 1.0,
            "host_path_fallback_calls": 0,
            "router_native_host_calls": 160,
            "router_torch_device_calls": 0,
            "router_hidden_d2h_total_bytes": 1_000,
            "router_outputs_h2d_total_bytes": 2_000,
            "router_scalar_sync_calls": 160,
            "device_path_calls": device_calls,
        },
    }


def test_trial_summary_reports_median_and_path_evidence() -> None:
    summary = MODULE.summarize_trials([row(6.0), row(7.0), row(6.5)])
    assert summary["tokens_per_second"]["median"] == 6.5
    assert summary["decode_seconds"]["sample_count"] == 9
    assert summary["peak_process_vram_per_gpu_bytes"]["cuda:1"] == 7_100_000_000
    assert summary["all_tokens_exact"] is True
    assert summary["all_expert_calls_device_resident"] is True
    assert summary["router_native_host_calls"] == 480
    assert summary["router_scalar_sync_calls"] == 480


def test_empty_trial_summary_fails_closed() -> None:
    try:
        MODULE.summarize_trials([])
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty trials must fail")
