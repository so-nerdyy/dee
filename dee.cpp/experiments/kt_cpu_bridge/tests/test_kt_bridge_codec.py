"""Codec tests: nibble order / e8m0 / dequant / clamp plumbing (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from kt_cpu_bridge.codec import (  # noqa: E402
    REALISTIC_SCALE_BYTES,
    compact_scales_e8,
    decode_e8m0,
    dequantize_weight,
    unpack_fp4,
    ue8m0_to_bf16_bits,
)


def test_unpack_fp4_official_nibble_order():
    packed = torch.tensor([[0x12, 0xF0]], dtype=torch.uint8)
    vals = unpack_fp4(packed)
    assert torch.equal(vals, torch.tensor([[1.0, 0.5, 0.0, -6.0]]))
    vals2 = unpack_fp4(torch.tensor([[0x14]], dtype=torch.uint8))
    assert torch.equal(vals2, torch.tensor([[2.0, 0.5]]))


def test_e8m0_decode_formula():
    bits = torch.tensor([0x7E, 0x7F, 0x80, 0x81, 0x82, 0x86, 0x88, 0x76], dtype=torch.uint8)
    got = decode_e8m0(bits)
    exp = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0, 128.0, 512.0, 2.0**-9])
    assert torch.allclose(got, exp, rtol=1e-6, atol=1e-30)


def test_e8m0_rejects_ff():
    import pytest

    with pytest.raises(ValueError):
        decode_e8m0(torch.tensor([0x7F, 0xFF], dtype=torch.uint8))


def test_e8m0_cross_dtype_reinterpret():
    base = torch.tensor([0x7F, 0x80, 0x81], dtype=torch.uint8)
    exp = torch.tensor([1.0, 2.0, 4.0])
    assert torch.allclose(decode_e8m0(base), exp)
    assert torch.allclose(decode_e8m0(base.to(torch.int8)), exp)
    assert torch.allclose(decode_e8m0(base.view(torch.float8_e8m0fnu)), exp)


def test_dequantize_shape_and_scaling():
    out, inp = 8, 64
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (out, inp // 2), dtype=torch.uint8)
    scale = torch.zeros((out, inp // 32), dtype=torch.uint8)
    scale[:, 0] = 0x7F
    scale[:, 1] = 0x80
    w = dequantize_weight(packed, scale)
    assert w.shape == (out, inp)
    assert torch.equal(w[:, 0], unpack_fp4(packed)[:, 0])
    assert torch.allclose(w[:, 32:], unpack_fp4(packed)[:, 32:] * 2.0)


def test_compact_e8_lossless_and_rejects_ff():
    import pytest

    torch.manual_seed(1)
    s = torch.randint(*REALISTIC_SCALE_BYTES, (16, 4), dtype=torch.uint8)
    assert torch.equal(compact_scales_e8(s), s)
    with pytest.raises(ValueError):
        compact_scales_e8(torch.tensor([[0xFF]], dtype=torch.uint8))


def test_ue8m0_to_bf16_lossless_spot():
    # (0x7F << 7) as bf16 == 1.0; (0x80 << 7) == 2.0
    import struct

    for byte, val in ((0x7F, 1.0), (0x80, 2.0), (0x81, 4.0)):
        bits = ue8m0_to_bf16_bits(byte)
        as_u32 = bits << 16
        assert struct.unpack("<f", struct.pack("<I", as_u32))[0] == val
