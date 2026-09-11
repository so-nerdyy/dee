from __future__ import annotations

import inspect

from scripts import run_ornith_generation as generation
from scripts import run_ornith_m5f_pointer_batched_benchmark as benchmark


def test_m5f_balanced_order_has_three_alternating_pairs() -> None:
    order = benchmark.balanced_order(3)
    assert order == [
        benchmark.CONTROL,
        benchmark.CANDIDATE,
        benchmark.CANDIDATE,
        benchmark.CONTROL,
        benchmark.CONTROL,
        benchmark.CANDIDATE,
    ]
    assert all(
        set(order[index:index + 2]) == set(benchmark.MODES)
        for index in range(0, len(order), 2)
    )


def test_m5f_paired_analysis_uses_candidate_over_control() -> None:
    sequence = [
        {"execution_mode": benchmark.CONTROL, "tokens_per_second": 8.0},
        {"execution_mode": benchmark.CANDIDATE, "tokens_per_second": 12.0},
        {"execution_mode": benchmark.CANDIDATE, "tokens_per_second": 10.0},
        {"execution_mode": benchmark.CONTROL, "tokens_per_second": 8.0},
    ]
    report = benchmark.paired_trial_analysis(sequence)
    assert report["pair_count"] == 2
    assert report["candidate_wins"] == 2
    assert report["pairs"][0]["speedup_ratio"] == 1.5
    assert report["pairs"][1]["speedup_ratio"] == 1.25


def test_pointer_workspace_contract_uses_fixed_eight_arrays() -> None:
    layers = 2
    topk = 8
    inter = 512
    pointer_bytes = 8
    host = benchmark.POINTER_ARRAYS * topk * pointer_bytes
    device = 3 * topk * inter * 2 + host
    stats = {
        "aggregate": {
            "host_moe_pointer_table_bytes": layers * host,
            "device_moe_pointer_batch_workspace_bytes": layers * device,
        },
        "by_layer": [
            {
                "host_moe_pointer_table_bytes": host,
                "device_moe_pointer_batch_workspace_bytes": device,
            }
            for _ in range(layers)
        ],
    }
    contract = benchmark.analyze_pointer_workspace(
        stats,
        layers=layers,
        topk=topk,
        inter=inter,
        pointer_bytes=pointer_bytes,
    )
    assert contract["passed"]
    assert contract["expected_device_workspace_bytes_per_layer"] == 25088


def test_pointer_workspace_contract_rejects_extra_allocation() -> None:
    stats = {
        "aggregate": {
            "host_moe_pointer_table_bytes": 512,
            "device_moe_pointer_batch_workspace_bytes": 25089,
        },
        "by_layer": [
            {
                "host_moe_pointer_table_bytes": 512,
                "device_moe_pointer_batch_workspace_bytes": 25089,
            }
        ],
    }
    assert not benchmark.analyze_pointer_workspace(
        stats, layers=1, topk=8, inter=512, pointer_bytes=8
    )["passed"]


def test_pointer_batched_mode_is_explicit_and_fail_closed() -> None:
    assert (
        generation.POINTER_BATCHED_EXECUTION_MODE
        in generation.EXECUTION_MODES
    )
    source = inspect.getsource(generation.HybridExperts.forward)
    assert "moe_forward_combined_pointer_batched_device" in source
    assert "native_pointer_batched_fallback_calls" in source
    assert "raise RuntimeError" in source


def test_pointer_batched_proof_schema_has_acceptance_counters() -> None:
    proof = generation.fresh_engine_path_proof()
    assert proof["native_pointer_batched_calls"] == 0
    assert proof["native_pointer_batched_fallback_calls"] == 0
