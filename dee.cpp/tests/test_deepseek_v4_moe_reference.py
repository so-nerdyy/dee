"""DS8 MoE reference tests (shared-expert FP8, weighted top-6 routing)."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    sys.stderr.write("tests/test_deepseek_v4_moe_reference.py requires pytest: pip install pytest\n")
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from scripts import deepseek_v4_expert_reference as ds7  # noqa: E402
from scripts import deepseek_v4_moe_reference as moe  # noqa: E402


def test_dequantize_fp8_e4m3_block_tiling() -> None:
    out, inp = 128, 256  # 1x2 scale tiles
    weight = (torch.rand(out, inp) * 10 - 5).to(torch.float8_e4m3fn)
    scale = torch.zeros((out // 128, inp // 128), dtype=torch.uint8)
    scale[0, 0] = 0x7F  # 1.0
    scale[0, 1] = 0x80  # 2.0
    w = moe.dequantize_fp8_e4m3(weight, scale)
    assert w.shape == (out, inp)
    # second 128-wide tile scaled by 2.0
    assert torch.allclose(w[:, 128:], weight.float()[:, 128:] * 2.0, atol=1e-3)
    assert torch.allclose(w[:, :128], weight.float()[:, :128], atol=1e-3)


def test_dequantize_fp8_e4m3_rejects_wrong_scale_shape() -> None:
    weight = torch.zeros(200, 200, dtype=torch.float8_e4m3fn)
    scale = torch.zeros((1, 1), dtype=torch.uint8)  # 200//128 = 1 tile but ceil(200/128)=2
    with pytest.raises(ValueError, match="block-128"):
        moe.dequantize_fp8_e4m3(weight, scale)


def test_shared_expert_forward_matches_direct_fp32() -> None:
    torch.manual_seed(1)
    hidden, inter = 128, 128
    shared = {
        "w1.weight": (torch.rand(inter, hidden) * 2 - 1).to(torch.float8_e4m3fn),
        "w1.scale": torch.full((1, 1), 0x80, dtype=torch.uint8),  # 2.0
        "w2.weight": (torch.rand(hidden, inter) * 2 - 1).to(torch.float8_e4m3fn),
        "w2.scale": torch.full((1, 1), 0x80, dtype=torch.uint8),
        "w3.weight": (torch.rand(inter, hidden) * 2 - 1).to(torch.float8_e4m3fn),
        "w3.scale": torch.full((1, 1), 0x80, dtype=torch.uint8),
    }
    x = torch.randn(3, hidden)
    got = moe.shared_expert_forward(x, shared)
    # Direct reference: dequant then expert math.
    w1 = moe.dequantize_fp8_e4m3(shared["w1.weight"], shared["w1.scale"])
    w2 = moe.dequantize_fp8_e4m3(shared["w2.weight"], shared["w2.scale"])
    w3 = moe.dequantize_fp8_e4m3(shared["w3.weight"], shared["w3.scale"])
    gate = torch.clamp(x @ w1.t(), max=10.0)
    up = torch.clamp(x @ w3.t(), min=-10.0, max=10.0)
    expected = (torch.nn.functional.silu(gate) * up) @ w2.t()
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_weighted_routed_expert_places_weight_before_w2() -> None:
    torch.manual_seed(2)
    hidden, inter = 64, 32
    t = {
        "w1.weight": torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8),
        "w1.scale": torch.randint(*ds7.REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8),
        "w2.weight": torch.randint(0, 256, (hidden, inter // 2), dtype=torch.uint8),
        "w2.scale": torch.randint(*ds7.REALISTIC_SCALE_BYTES, (hidden, inter // 32), dtype=torch.uint8),
        "w3.weight": torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8),
        "w3.scale": torch.randint(*ds7.REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8),
    }
    x = torch.randn(2, hidden)
    weights = torch.tensor([[0.3], [0.7]])
    got = moe.routed_expert_forward_weighted(x, weights, t)
    # Manual: weight applied to intermediate h BEFORE w2 (official placement).
    w1 = ds7.dequantize_expert_weight(t["w1.weight"], t["w1.scale"])
    w2 = ds7.dequantize_expert_weight(t["w2.weight"], t["w2.scale"])
    w3 = ds7.dequantize_expert_weight(t["w3.weight"], t["w3.scale"])
    gate = torch.clamp(x @ w1.t(), max=10.0)
    up = torch.clamp(x @ w3.t(), min=-10.0, max=10.0)
    h = torch.nn.functional.silu(gate) * up
    expected = (weights * h) @ w2.t()
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)
    # The routing weight is a per-token SCALAR, so weight-before-w2 and
    # weight-after-w2 are mathematically identical ((w*h)@W2^T == w*(h@W2^T)).
    # What must differ is the UNWEIGHTED output: the reference must actually
    # apply the routing weight (this is the semantic the DS8 candidate must
    # match).
    unweighted = h @ w2.t()
    assert not torch.allclose(got, unweighted, atol=1e-2, rtol=1e-2)


def test_moe_layer_routing_and_combination() -> None:
    torch.manual_seed(3)
    hidden, inter, n_experts, topk = 64, 32, 8, 3
    routed = {}
    for eid in range(n_experts):
        routed[eid] = {
            "w1.weight": torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8),
            "w1.scale": torch.randint(*ds7.REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8),
            "w2.weight": torch.randint(0, 256, (hidden, inter // 2), dtype=torch.uint8),
            "w2.scale": torch.randint(*ds7.REALISTIC_SCALE_BYTES, (hidden, inter // 32), dtype=torch.uint8),
            "w3.weight": torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8),
            "w3.scale": torch.randint(*ds7.REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8),
        }
    shared = {
        "w1.weight": (torch.rand(inter, hidden) * 2 - 1).to(torch.float8_e4m3fn),
        "w1.scale": torch.full((1, 1), 0x7F, dtype=torch.uint8),
        "w2.weight": (torch.rand(hidden, inter) * 2 - 1).to(torch.float8_e4m3fn),
        "w2.scale": torch.full((1, 1), 0x7F, dtype=torch.uint8),
        "w3.weight": (torch.rand(inter, hidden) * 2 - 1).to(torch.float8_e4m3fn),
        "w3.scale": torch.full((1, 1), 0x7F, dtype=torch.uint8),
    }
    gate = torch.randn(n_experts, hidden) * 0.5
    bias = torch.randn(n_experts) * 0.1
    x = torch.randn(4, hidden)

    out = moe.moe_layer_forward(x, gate, bias, routed, shared,
                                topk=topk, keep_per_expert=True)
    # Routing weights sum to route_scale (non-softmax normalization).
    assert torch.allclose(out["routing_weights"].sum(dim=-1),
                          torch.full((4,), 1.5))
    # Top-k uses BIASED scores (selection), weights from UN-shifted scores.
    scores = torch.nn.functional.softplus(x @ gate.t()).sqrt()
    biased = scores + bias
    expected_ids = biased.topk(topk, dim=-1)[1]
    assert torch.equal(out["expert_ids"], expected_ids)
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 1.5
    assert torch.allclose(out["routing_weights"], expected_weights, atol=1e-5)
    # moe_output == weighted routed sum + shared.
    routed_sum = torch.zeros_like(out["moe_output"])
    for contrib in out["per_expert"].values():
        routed_sum += contrib
    assert torch.allclose(routed_sum + out["shared_output"], out["moe_output"],
                          atol=1e-4)
    # shared_output == shared_expert_forward(x).
    assert torch.allclose(out["shared_output"],
                          moe.shared_expert_forward(x, shared), atol=1e-4)


def test_moe_layer_without_shared_and_zero_weights() -> None:
    torch.manual_seed(4)
    hidden, inter, n_experts, topk = 32, 32, 4, 2
    routed = {}
    for eid in range(n_experts):
        routed[eid] = {
            "w1.weight": torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8),
            "w1.scale": torch.randint(*ds7.REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8),
            "w2.weight": torch.randint(0, 256, (hidden, inter // 2), dtype=torch.uint8),
            "w2.scale": torch.randint(*ds7.REALISTIC_SCALE_BYTES, (hidden, inter // 32), dtype=torch.uint8),
            "w3.weight": torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8),
            "w3.scale": torch.randint(*ds7.REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8),
        }
    gate = torch.zeros(n_experts, hidden)
    bias = None
    x = torch.randn(2, hidden)
    out = moe.moe_layer_forward(x, gate, bias, routed, None, topk=topk)
    assert torch.isfinite(out["moe_output"]).all()
    assert torch.allclose(out["shared_output"], torch.zeros_like(out["shared_output"]))
