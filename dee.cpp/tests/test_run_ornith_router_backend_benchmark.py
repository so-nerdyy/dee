from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_ornith_router_backend_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("router_backend_benchmark", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

M5B_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_ornith_m5b_execution_benchmark.py"
)
M5B_SPEC = importlib.util.spec_from_file_location("m5b_execution_benchmark", M5B_SCRIPT)
assert M5B_SPEC and M5B_SPEC.loader
M5B_MODULE = importlib.util.module_from_spec(M5B_SPEC)
M5B_SPEC.loader.exec_module(M5B_MODULE)

M5C_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_ornith_m5c_combined_benchmark.py"
)
M5C_SPEC = importlib.util.spec_from_file_location(
    "m5c_combined_benchmark", M5C_SCRIPT
)
assert M5C_SPEC and M5C_SPEC.loader
M5C_MODULE = importlib.util.module_from_spec(M5C_SPEC)
M5C_SPEC.loader.exec_module(M5C_MODULE)

M5D_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_ornith_m5d_direct_row_benchmark.py"
)
M5D_SPEC = importlib.util.spec_from_file_location(
    "m5d_direct_row_benchmark", M5D_SCRIPT
)
assert M5D_SPEC and M5D_SPEC.loader
M5D_MODULE = importlib.util.module_from_spec(M5D_SPEC)
M5D_SPEC.loader.exec_module(M5D_MODULE)


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


def test_m5b_balanced_order_is_paired_and_bounded() -> None:
    order = M5B_MODULE.balanced_order(3)
    assert order == [
        "debug-full-logit",
        "production",
        "production",
        "debug-full-logit",
        "debug-full-logit",
        "production",
    ]
    assert order.count("debug-full-logit") == 3
    assert order.count("production") == 3


def test_m5b_paired_analysis_requires_majority_real_wins() -> None:
    sequence = [
        {"execution_mode": "debug-full-logit", "tokens_per_second": 8.0},
        {"execution_mode": "production", "tokens_per_second": 8.2},
        {"execution_mode": "production", "tokens_per_second": 8.1},
        {"execution_mode": "debug-full-logit", "tokens_per_second": 8.0},
        {"execution_mode": "debug-full-logit", "tokens_per_second": 8.0},
        {"execution_mode": "production", "tokens_per_second": 7.9},
    ]
    result = M5B_MODULE.paired_trial_analysis(sequence)
    assert result["pair_count"] == 3
    assert result["production_wins"] == 2
    assert result["minimum_speedup_ratio"] == 7.9 / 8.0


def test_m5b_exact_trace_comparison_rejects_any_numeric_drift() -> None:
    records = {
        category: [("operation", np.asarray([1.0], dtype=np.float32))]
        for category in set(M5B_MODULE.TOLERANCES) | {"expert_ids"}
    }
    expected = SimpleNamespace(records=records)
    exact = SimpleNamespace(records={
        category: [(label, array.copy()) for label, array in rows]
        for category, rows in records.items()
    })
    drifted_records = {
        category: [(label, array.copy()) for label, array in rows]
        for category, rows in records.items()
    }
    drifted_records["final_hidden_state"][0][1][0] += 1e-6
    drifted = SimpleNamespace(records=drifted_records)
    assert M5B_MODULE.exact_trace_comparison(
        expected, exact
    )["all_categories_bitwise_exact"]
    assert not M5B_MODULE.exact_trace_comparison(
        expected, drifted
    )["all_categories_bitwise_exact"]


def test_m5b_exact_trace_comparison_fails_closed_on_missing_category() -> None:
    partial = SimpleNamespace(records={
        "final_hidden_state": [(
            "layer=39", np.asarray([1.0], dtype=np.float32)
        )]
    })
    result = M5B_MODULE.exact_trace_comparison(partial, partial)
    assert not result["all_required_categories_present"]
    assert not result["all_categories_bitwise_exact"]


def test_m5b_thermal_clock_analysis_is_fail_closed() -> None:
    assert M5B_MODULE.thermal_clock_analysis([])["anomaly_detected"]
    rows = [{
        "thermal_clock": {
            "nvml_error": None,
            "by_device": {
                "cuda:0": {
                    "sample_count": 4,
                    "maximum_temperature_c": 71,
                    "median_sm_clock_mhz": 1500,
                },
                "cuda:1": {
                    "sample_count": 4,
                    "maximum_temperature_c": 72,
                    "median_sm_clock_mhz": 1485,
                },
            },
        },
    }]
    result = M5B_MODULE.thermal_clock_analysis(rows)
    assert result["samples_present"]
    assert not result["anomaly_detected"]
    assert not result["absolute_clock_floor_warning_observed"]


def test_m5b_thermal_clock_analysis_separates_floor_warning_from_parity() -> None:
    rows = []
    for mode in ("production", "debug-full-logit"):
        rows.append({
            "execution_mode": mode,
            "thermal_clock": {
                "nvml_error": None,
                "by_device": {
                    "cuda:0": {
                        "sample_count": 8,
                        "maximum_temperature_c": 52,
                        "median_temperature_c": 51,
                        "median_sm_clock_mhz": 585,
                    },
                    "cuda:1": {
                        "sample_count": 8,
                        "maximum_temperature_c": 49,
                        "median_temperature_c": 48,
                        "median_sm_clock_mhz": 585,
                    },
                },
            },
        })
    result = M5B_MODULE.thermal_clock_analysis(rows)
    assert result["absolute_clock_floor_warning_observed"]
    assert not result["anomaly_detected"]
    rows[1]["thermal_clock"]["by_device"]["cuda:0"][
        "median_sm_clock_mhz"
    ] = 1000
    result = M5B_MODULE.thermal_clock_analysis(rows)
    assert result["anomaly_detected"]
    assert result["by_device"]["cuda:0"]["clock_parity_anomaly"]


def test_m5c_balanced_order_is_paired_and_bounded() -> None:
    order = M5C_MODULE.balanced_order(3)
    assert order == [
        "production",
        "native-combined",
        "native-combined",
        "production",
        "production",
        "native-combined",
    ]
    assert order.count("production") == 3
    assert order.count("native-combined") == 3


def test_m5c_paired_analysis_handles_reversed_pair_order() -> None:
    sequence = [
        {"execution_mode": "production", "tokens_per_second": 8.0},
        {"execution_mode": "native-combined", "tokens_per_second": 8.4},
        {"execution_mode": "native-combined", "tokens_per_second": 8.2},
        {"execution_mode": "production", "tokens_per_second": 8.0},
    ]
    result = M5C_MODULE.paired_trial_analysis(sequence)
    assert result["pair_count"] == 2
    assert result["candidate_wins"] == 2
    assert result["minimum_speedup_ratio"] == 8.2 / 8.0


def test_m5c_workspace_proof_is_exact_and_fail_closed() -> None:
    stats = {
        "aggregate": {
            "host_moe_dispatch_bytes": 128,
            "device_moe_raw_workspace_bytes": 256,
        },
        "by_layer": [
            {
                "host_moe_dispatch_bytes": 64,
                "device_moe_raw_workspace_bytes": 128,
            },
            {
                "host_moe_dispatch_bytes": 64,
                "device_moe_raw_workspace_bytes": 128,
            },
        ],
    }
    proof = M5C_MODULE.analyze_workspace(
        stats, layers=2, topk=8, hidden=4
    )
    assert proof["passed"]
    stats["by_layer"][1]["device_moe_raw_workspace_bytes"] += 4
    assert not M5C_MODULE.analyze_workspace(
        stats, layers=2, topk=8, hidden=4
    )["passed"]


def test_m5d_balanced_order_and_paired_analysis() -> None:
    order = M5D_MODULE.balanced_order(3)
    assert order == [
        "native-combined",
        "native-combined-direct",
        "native-combined-direct",
        "native-combined",
        "native-combined",
        "native-combined-direct",
    ]
    sequence = [
        {"execution_mode": order[0], "tokens_per_second": 8.0},
        {"execution_mode": order[1], "tokens_per_second": 8.1},
        {"execution_mode": order[2], "tokens_per_second": 8.2},
        {"execution_mode": order[3], "tokens_per_second": 8.0},
    ]
    result = M5D_MODULE.paired_trial_analysis(sequence)
    assert result["pair_count"] == 2
    assert result["candidate_wins"] == 2
    assert result["minimum_speedup_ratio"] == 8.1 / 8.0
