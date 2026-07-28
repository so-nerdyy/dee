from __future__ import annotations

import importlib.util
from pathlib import Path

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
    proof = MODULE.fresh_engine_path_proof()
    assert proof["router_native_host_calls"] == 0
    assert proof["router_torch_device_calls"] == 0
