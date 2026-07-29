from __future__ import annotations

import inspect

from scripts import run_ornith_generation as generation
from scripts import run_ornith_m5e_fused_norm_benchmark as benchmark


def test_m5e_balanced_order_has_three_alternating_pairs() -> None:
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


def test_m5e_paired_analysis_uses_candidate_over_control() -> None:
    sequence = [
        {"execution_mode": benchmark.CONTROL, "tokens_per_second": 8.0},
        {"execution_mode": benchmark.CANDIDATE, "tokens_per_second": 10.0},
        {"execution_mode": benchmark.CANDIDATE, "tokens_per_second": 9.0},
        {"execution_mode": benchmark.CONTROL, "tokens_per_second": 8.0},
    ]
    report = benchmark.paired_trial_analysis(sequence)
    assert report["pair_count"] == 2
    assert report["candidate_wins"] == 2
    assert report["pairs"][0]["speedup_ratio"] == 1.25
    assert report["pairs"][1]["speedup_ratio"] == 1.125


def test_fused_norm_mode_is_explicit_and_fail_closed() -> None:
    assert generation.FUSED_NORM_EXECUTION_MODE in generation.EXECUTION_MODES
    regular = inspect.getsource(generation.HybridRMSNorm.forward)
    gated = inspect.getsource(generation.HybridRMSNormGated.forward)
    assert "qwen_rms_norm_device" in regular
    assert "fused Qwen RMSNorm failed" in regular
    assert "qwen_rms_norm_gated_device" in gated
    assert "fused Qwen gated RMSNorm failed" in gated


def test_fused_norm_proof_schema_has_all_acceptance_counters() -> None:
    proof = generation.fresh_engine_path_proof()
    assert proof["fused_rms_norm_calls"] == 0
    assert proof["fused_rms_norm_gated_calls"] == 0
    assert proof["fused_norm_output_allocations"] == 0
    assert proof["fused_norm_failures"] == 0


def test_m5e_inventory_contract_is_independent_and_complete() -> None:
    inventory = {
        "wrapped_regular": [
            label
            for layer in range(benchmark.ORNITH_LAYER_COUNT)
            for label in (
                f"layer={layer}:input_layernorm",
                f"layer={layer}:post_attention_layernorm",
            )
        ] + ["final_norm"],
        "wrapped_gated": [
            f"layer={layer}:linear_attn.norm"
            for layer in range(benchmark.ORNITH_LINEAR_ATTN_LAYER_COUNT)
        ],
        "intentionally_unwrapped_full_attention_head_norms": [
            label
            for layer in range(
                benchmark.ORNITH_LINEAR_ATTN_LAYER_COUNT,
                benchmark.ORNITH_LAYER_COUNT,
            )
            for label in (
                f"layer={layer}:self_attn.q_norm",
                f"layer={layer}:self_attn.k_norm",
            )
        ],
    }
    contract = benchmark.normalization_inventory_contract(inventory)
    assert contract["passed"]
    assert all(contract["checks"].values())


def test_m5e_inventory_contract_rejects_a_missed_head_norm() -> None:
    inventory = {
        "wrapped_regular": [
            label
            for layer in range(benchmark.ORNITH_LAYER_COUNT)
            for label in (
                f"layer={layer}:input_layernorm",
                f"layer={layer}:post_attention_layernorm",
            )
        ] + ["final_norm"],
        "wrapped_gated": [
            f"layer={layer}:linear_attn.norm"
            for layer in range(benchmark.ORNITH_LINEAR_ATTN_LAYER_COUNT)
        ],
        "intentionally_unwrapped_full_attention_head_norms": [
            label
            for layer in range(
                benchmark.ORNITH_LINEAR_ATTN_LAYER_COUNT,
                benchmark.ORNITH_LAYER_COUNT,
            )
            for label in (
                f"layer={layer}:self_attn.q_norm",
                f"layer={layer}:self_attn.k_norm",
            )
        ][:-1],
    }
    assert not benchmark.normalization_inventory_contract(inventory)["passed"]


def test_m5e_source_contains_real_shape_dual_stream_contract_probe() -> None:
    source = inspect.getsource(benchmark.native_contract_probe)
    assert "wrapper.reference_forward" in source
    assert "torch.cuda.default_stream" in source
    assert "torch.cuda.Stream" in source
    assert "view(torch.int16)" in source
    assert '"regular", "gated"' in source
