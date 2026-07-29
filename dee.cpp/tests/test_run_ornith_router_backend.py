from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_ornith_generation.py"
)
SPEC = importlib.util.spec_from_file_location("run_ornith_generation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReferenceRouter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            requires_grad=False,
        )

    def forward(self, hidden_states):
        logits = torch.nn.functional.linear(hidden_states, self.weight)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float)
        weights, experts = torch.topk(probabilities, 2, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return probabilities, weights, experts


class EngineMustNotRun:
    def route_topk_batch(self, *_args, **_kwargs):
        raise AssertionError("torch-device backend called the native-host router")


def test_torch_device_router_uses_official_resident_module() -> None:
    context = MODULE.ExecutionContext()
    context.router_backend = "torch-device"
    reference = ReferenceRouter()
    hybrid = MODULE.HybridRouter(
        torch.nn, reference, EngineMustNotRun(), 7, context, top_k=2
    )
    hidden = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
    expected = reference(hidden.reshape(-1, 4))
    actual = hybrid.module(hidden)
    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(actual, expected))
    assert context.executed_router_layers == {7}
    proof = context.engine_path_proof
    assert proof["router_torch_device_calls"] == 1
    assert proof["router_native_host_calls"] == 0
    assert proof["router_hidden_d2h_total_bytes"] == 0
    assert proof["router_outputs_h2d_total_bytes"] == 0
    assert proof["router_scalar_sync_calls"] == 0


def test_native_host_remains_the_default_backend() -> None:
    context = MODULE.ExecutionContext()
    assert context.router_backend == "native-host"
    assert context.execution_mode == "production"
    proof = MODULE.fresh_engine_path_proof()
    assert proof["router_native_host_calls"] == 0
    assert proof["router_torch_device_calls"] == 0
    assert proof["pybind_device_calls"] == 0
    assert proof["python_combine_calls"] == 0
    assert proof["raw_output_allocations"] == 0
    assert proof["native_combined_calls"] == 0
    assert proof["native_combined_stream_handoffs"] == 0


def test_native_combined_mode_attempts_native_api_before_python_d2h() -> None:
    source = inspect.getsource(MODULE.HybridExperts.forward)
    combined = source.index("self.engine.moe_forward_combined_device(")
    python_d2h = source.index(
        'with forensic_span(self.context, "expert_ids_gpu_to_cpu"'
    )
    assert combined < python_d2h
    assert "native_combined_stream_handoffs" in source


def test_unknown_execution_mode_fails_before_runtime_access() -> None:
    with pytest.raises(ValueError, match="unsupported execution mode"):
        MODULE.run_generation(
            {}, None, "Hello", 4, False, "dee", False,
            execution_mode="not-a-mode",
        )


def test_decode_timer_includes_m5b_post_model_work() -> None:
    source = inspect.getsource(MODULE.run_generation)
    elapsed = source.index(
        "full_token_elapsed = time.perf_counter() - step_start"
    )
    assert elapsed > source.index("logits_records.append")
    assert elapsed > source.index("torch.argmax(logits")
    assert elapsed > source.index("next_token_buffer.fill_")
    assert elapsed > source.index("attention_mask = torch.cat")


def test_stable_fp16_combine_matches_legacy_with_duplicate_ids() -> None:
    raw = torch.tensor(
        [[
            [0.3333, -1.25, 2.0],
            [1.5, 0.25, -0.75],
            [-2.0, 0.5, 0.125],
            [0.75, -0.5, 1.25],
        ]],
        dtype=torch.float16,
    )
    weights = torch.tensor(
        [[0.125, 0.375, 0.25, 0.25]], dtype=torch.float32
    )
    # Duplicate expert 3 verifies stable order among equal IDs: positions
    # [1, 3, 0, 2] for expert IDs [1, 2, 3, 3].
    ids = MODULE.np.asarray([[3, 1, 3, 2]], dtype=MODULE.np.int32)
    legacy = MODULE.stable_combine_selected_experts(
        torch.zeros((1, 3), dtype=torch.float16),
        raw,
        weights,
        ids,
        legacy_accumulator=True,
    )
    production = MODULE.stable_combine_selected_experts(
        torch.zeros((1, 3), dtype=torch.float16),
        raw,
        weights,
        ids,
        legacy_accumulator=False,
    )
    assert torch.equal(production, legacy)


def test_eager_mixed_dtype_combine_rounds_product_before_add() -> None:
    raw = torch.tensor([[[-3.041015625], [3.09765625]]], dtype=torch.float16)
    weights = torch.tensor(
        [[1.0, 0.891144335269928]], dtype=torch.float32
    )
    ids = MODULE.np.asarray([[0, 1]], dtype=MODULE.np.int32)
    actual = MODULE.stable_combine_selected_experts(
        torch.zeros((1, 1), dtype=torch.float16),
        raw,
        weights,
        ids,
        legacy_accumulator=False,
    )
    assert actual.item() == -0.28125
    assert actual.view(torch.int16).item() == -19328


def test_native_combined_mode_is_fail_closed() -> None:
    source = inspect.getsource(MODULE.HybridExperts.forward)
    assert "native-combined binding unavailable" in source
    assert "native-combined execution failed" in source
