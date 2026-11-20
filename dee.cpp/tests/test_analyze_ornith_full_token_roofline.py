from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_ornith_full_token_roofline.py"
)
SPEC = importlib.util.spec_from_file_location("ornith_roofline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def tensor(
    name: str,
    component: str,
    device: str,
    byte_size: int,
    shape: list[int],
    dtype: str = "BF16",
) -> dict:
    return {
        "tensor_name": name,
        "target_runtime_component": component,
        "target_device": device,
        "byte_size": byte_size,
        "shape": shape,
        "dtype": dtype,
        "expected_lifetime": "generation_resident",
    }


def test_tensor_floor_uses_one_embedding_row_and_topk_experts() -> None:
    tensors = [
        tensor("model.embed_tokens.weight", "token_embedding", "cuda:0", 200, [10, 10]),
        tensor("lm_head.weight", "lm_head", "cuda:1", 200, [10, 10]),
    ]
    for layer, device in ((0, "cuda:0"), (1, "cuda:1")):
        for expert in range(3):
            tensors.extend(
                [
                    tensor(
                        f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                        "routed_expert",
                        device,
                        40,
                        [2, 10],
                    ),
                    tensor(
                        f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                        "routed_expert",
                        device,
                        40,
                        [2, 10],
                    ),
                    tensor(
                        f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                        "routed_expert",
                        device,
                        40,
                        [10, 2],
                    ),
                ]
            )
    result = MODULE.analyze_tensor_map({"tensors": tensors}, topk=2)
    assert result["bytes_per_expert"] == 120
    assert result["embedding_row_bytes_by_device"] == {"cuda:0": 20}
    assert result["mandatory_weight_bytes_by_device"] == {
        "cuda:0": 260,
        "cuda:1": 440,
    }
    assert result["mandatory_weight_bytes_per_token"] == 700


def test_trace_union_and_runtime_counts() -> None:
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemvx::kernel",
                "ts": 0,
                "dur": 10,
                "args": {"device": 0},
            },
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy DtoD (Device -> Device)",
                "ts": 5,
                "dur": 10,
                "args": {"device": 1, "bytes": 4096},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemv2T_kernel",
                "ts": 25,
                "dur": 5,
                "args": {"device": 1},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 1,
                "dur": 1,
                "args": {},
            },
        ]
    }
    result = MODULE.analyze_trace(trace)
    assert result["kernel_count"] == 2
    assert result["gpu_memcpy_count"] == 1
    assert result["gpu_timeline"]["span_ms"] == 0.03
    assert result["gpu_timeline"]["busy_ms"] == 0.02
    assert result["gpu_timeline"]["idle_ms"] == 0.01
    assert result["gpu_timeline"]["positive_gap_count"] == 1
    assert result["cuda_runtime_call_counts"]["cudaLaunchKernel"] == 1


def test_layer_differential_is_explicitly_inferred() -> None:
    rows = []
    for layer, layer_type, residual, attention in (
        (0, "linear_attention", 6.0, 0.0),
        (1, "full_attention", 1.0, 2.0),
    ):
        row = {
            "phase": "decode",
            "step": 3,
            "layer": layer,
            "layer_type": layer_type,
            "total_layer_wall_ms": 10.0,
            "residual_and_unattributed_wall_ms": residual,
            "attention_wall_ms": attention,
        }
        rows.append(row)
    result = MODULE.analyze_layer_timing({"rows": rows})
    differential = result["linear_attention_differential"]
    assert differential["inferred_linear_attention_ms"] == 5.0
    assert differential["evidence_class"] == "inferred_differential"
    matching = [
        row
        for row in result["derived_ranked_critical_path"]
        if row["component"] == "linear_attention_inferred"
    ]
    assert matching[0]["evidence_class"] == "inferred_differential"


def test_roofline_keeps_floor_distinct_from_measurement() -> None:
    tensor_analysis = {
        "mandatory_weight_bytes_per_token": 6_000_000_000,
        "max_device_mandatory_weight_bytes": 3_500_000_000,
    }
    control = {"single_stream_decode_tokens_per_second": 6.5}
    result = MODULE.build_roofline(tensor_analysis, control, 270.0, 320.0)
    assert result["evidence_class"] == "calculated_roofline_not_hardware_counter"
    assert result["achieved_effective_mandatory_weight_bandwidth_gb_per_second"] == 39.0
    assert result["ceilings"]["realistic_sustained"][
        "single_sequence_sequential_device_tps"
    ] == 45.0
    assert result["mtp_or_speculative_decoding"]["projection"] is None
