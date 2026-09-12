from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    sys.stderr.write("tests/test_deepseek_v4_expert_reference.py requires pytest: pip install pytest\n")
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts import deepseek_v4_expert_reference as ref  # noqa: E402


def test_e8m0_decode_formula() -> None:
    # ue8m0: value = 2^(bits - 127). Constants verified against
    # torch.float8_e8m0fnu round-trips on torch 2.7.1; the dtype test below
    # exercises the live torch round-trip path.
    sample_bytes = [0x7E, 0x7F, 0x80, 0x81, 0x82, 0x86, 0x88, 0x76, 0x7B, 0x00]
    bits = torch.tensor(sample_bytes, dtype=torch.uint8)
    got = ref.decode_e8m0(bits)
    expected = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0, 128.0, 512.0, 2.0 ** -9,
                             0.0625, 2.0 ** -127], dtype=torch.float32)
    assert torch.allclose(got, expected, rtol=1e-6, atol=1e-30)


def test_official_constants_pinned() -> None:
    # Pin the official model constants the reference relies on, so they are
    # exercised (not dead module state) and any future drift is caught.
    assert ref.FP4_BLOCK_SIZE == 32  # weight scale block on K (official)
    assert ref.FP8_MAX == 448.0 and ref.FP8_MIN == -448.0  # fp8 e4m3 clamp
    assert ref.E8M0_BIAS == 127  # ue8m0 exponent bias
    assert ref.REALISTIC_SCALE_BYTES == (0x7D, 0x83)
    # swiglu_limit=10.0 is the model's default clamp (inference/config.json).
    # Dimensions must be self-consistent: hidden=32, inter=32 (a multiple of
    # the 32-element scale block, like the real moe_inter_dim=2048).
    hidden = inter = 32
    y = ref.expert_forward(
        torch.zeros(1, hidden),
        torch.zeros(inter, hidden // 2, dtype=torch.uint8),
        torch.full((inter, hidden // 32), 0x80, dtype=torch.uint8),
        torch.zeros(hidden, inter // 2, dtype=torch.uint8),
        torch.full((hidden, inter // 32), 0x80, dtype=torch.uint8),
        torch.zeros(inter, hidden // 2, dtype=torch.uint8),
        torch.full((inter, hidden // 32), 0x80, dtype=torch.uint8),
        swiglu_limit=10.0,
    )
    assert y.shape == (1, hidden)
    assert torch.isfinite(y).all()


def test_decode_e8m0_bit_reinterpretation_across_dtypes() -> None:
    # Real checkpoint dtypes: weights load as int8 (I8), scales as
    # float8_e8m0fnu (F8_E8M0). decode_e8m0 must reinterpret raw bits, not
    # numerically cast (byte 0x81 -> 4.0 as float8 -> int 4 would be wrong).
    bytes_ = torch.tensor([0x7F, 0x80, 0x81, 0x86, 0x88], dtype=torch.uint8)
    expected = torch.tensor([1.0, 2.0, 4.0, 128.0, 512.0], dtype=torch.float32)
    assert torch.allclose(ref.decode_e8m0(bytes_), expected)
    # int8 (two's complement bit pattern, e.g. 0xFF -> -1).
    i8 = bytes_.to(torch.int8)
    assert torch.allclose(ref.decode_e8m0(i8), expected)
    # float8_e8m0fnu.
    f8 = bytes_.view(torch.float8_e8m0fnu)
    assert torch.allclose(ref.decode_e8m0(f8), expected)


def test_unpack_fp4_official_nibble_order() -> None:
    # Official convert.py order: low nibble -> element 2i, high nibble -> 2i+1.
    # 0x12: low=2 -> FP4_TABLE[2]=1.0, high=1 -> FP4_TABLE[1]=0.5.
    packed = torch.tensor([[0x12, 0xF0]], dtype=torch.uint8)  # [1, 2] -> [1, 4]
    vals = ref.unpack_fp4(packed)
    expected = torch.tensor([[1.0, 0.5, 0.0, -6.0]], dtype=torch.float32)
    assert torch.equal(vals, expected)
    # 0x14: low=4 -> 2.0, high=1 -> 0.5 (the value set the earlier wrong
    # expectation implied).
    vals2 = ref.unpack_fp4(torch.tensor([[0x14]], dtype=torch.uint8))
    assert torch.equal(vals2, torch.tensor([[2.0, 0.5]], dtype=torch.float32))
    # Table round-trip over all 256 bytes.
    all_bytes = torch.arange(256, dtype=torch.uint8).reshape(1, 256)
    vals = ref.unpack_fp4(all_bytes)
    assert vals.shape == (1, 512)
    lo = (all_bytes & 0x0F).to(torch.int64)
    hi = ((all_bytes >> 4) & 0x0F).to(torch.int64)
    exp_lo = ref.FP4_TABLE[lo]
    exp_hi = ref.FP4_TABLE[hi]
    assert torch.equal(vals[0, 0::2], exp_lo[0])
    assert torch.equal(vals[0, 1::2], exp_hi[0])


def test_dequantize_weight_shape_and_scaling() -> None:
    out, inp = 8, 64  # in//32 = 2 scale blocks
    packed = torch.randint(0, 256, (out, inp // 2), dtype=torch.uint8)
    scale = torch.zeros((out, inp // 32), dtype=torch.uint8)
    scale[:, 0] = 0x7F  # 1.0 for first 32 inputs
    scale[:, 1] = 0x80  # 2.0 for next 32 inputs
    w = ref.dequantize_expert_weight(packed, scale)
    assert w.shape == (out, inp)
    assert torch.equal(w[:, 0], ref.unpack_fp4(packed)[:, 0])
    assert torch.allclose(w[:, 32:], ref.unpack_fp4(packed)[:, 32:] * 2.0)


def _make_expert(hidden: int, inter: int, seed: int = 1):
    torch.manual_seed(seed)
    packed_w1 = torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8)
    scale_w1 = torch.randint(*ref.REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8)
    packed_w2 = torch.randint(0, 256, (hidden, inter // 2), dtype=torch.uint8)
    scale_w2 = torch.randint(*ref.REALISTIC_SCALE_BYTES, (hidden, inter // 32), dtype=torch.uint8)
    packed_w3 = torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8)
    scale_w3 = torch.randint(*ref.REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8)
    return packed_w1, scale_w1, packed_w2, scale_w2, packed_w3, scale_w3


def test_expert_forward_clamps_gate_and_up() -> None:
    hidden, inter = 64, 32
    (packed_w1, scale_w1, packed_w2, scale_w2,
     packed_w3, scale_w3) = _make_expert(hidden, inter)
    x = torch.randn(3, hidden) * 3.0  # amplify so clamps engage

    # The official Expert.forward clamps asymmetrically:
    #   gate = clamp(gate, max=limit)      (no lower bound!)
    #   up   = clamp(up,  min=-limit, max=limit)
    limit = 10.0
    w1 = ref.dequantize_expert_weight(packed_w1, scale_w1)
    w2 = ref.dequantize_expert_weight(packed_w2, scale_w2)
    w3 = ref.dequantize_expert_weight(packed_w3, scale_w3)
    gate = x @ w1.t()
    up = x @ w3.t()
    gate_c = torch.clamp(gate, max=limit)
    up_c = torch.clamp(up, min=-limit, max=limit)
    expected = (torch.nn.functional.silu(gate_c) * up_c) @ w2.t()

    got = ref.expert_forward(x, packed_w1, scale_w1, packed_w2, scale_w2,
                             packed_w3, scale_w3, swiglu_limit=limit)
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)

    # With amplification some gate/up values exceed the limit, so clamping
    # must have changed something (guard against a no-op implementation).
    unclamped = (torch.nn.functional.silu(gate) * up) @ w2.t()
    assert not torch.allclose(got, unclamped, atol=1e-2, rtol=1e-2)

    # limit=0 disables clamping entirely and must match the unclamped math.
    y_free = ref.expert_forward(x, packed_w1, scale_w1, packed_w2, scale_w2,
                                packed_w3, scale_w3, swiglu_limit=0.0)
    assert torch.allclose(y_free, unclamped, atol=1e-4, rtol=1e-4)


def test_validate_fp4_consistency_metrics() -> None:
    out, inp = 16, 64
    packed = torch.randint(0, 256, (out, inp // 2), dtype=torch.uint8)
    scale = torch.randint(*ref.REALISTIC_SCALE_BYTES, (out, inp // 32), dtype=torch.uint8)
    w = ref.dequantize_expert_weight(packed, scale)
    metrics = ref.validate_fp4_consistency(packed, scale)
    assert metrics["shape"] == [out, inp]
    # A random byte with a zero nibble decodes to exactly 0.0 (FP4_TABLE[0]),
    # so the nonzero fraction is strictly between 0 and 1 for random input.
    assert 0.0 < metrics["nonzeros"] < 1.0
    assert metrics["min"] == float(w.min())
    assert metrics["max"] == float(w.max())
    assert metrics["abs_mean"] == float(w.abs().mean())
    assert torch.isfinite(w).all()


def test_router_sqrtsoftplus_normalized_scaled() -> None:
    torch.manual_seed(2)
    n_tokens, dim, n_experts, topk = 4, 128, 256, 6
    x = torch.randn(n_tokens, dim)
    gate = torch.randn(n_experts, dim) * 0.5
    scores, ids, weights = ref.router_scores(x, gate, topk=topk)
    assert ids.shape == (n_tokens, topk)
    assert weights.shape == (n_tokens, topk)
    # Non-softmax: weights sum to 1 then scaled by route_scale.
    assert torch.allclose(weights.sum(dim=-1), torch.full((n_tokens,), 1.5))
    # Top-k ids must be the k largest biased scores.
    expected_scores = torch.nn.functional.softplus(x @ gate.t()).sqrt()
    topk_ids = expected_scores.topk(topk, dim=-1)[1]
    assert torch.equal(ids, topk_ids)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
