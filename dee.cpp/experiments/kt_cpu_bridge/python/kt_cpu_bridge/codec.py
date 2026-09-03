"""E2M1/E8M0 codec mirroring dee weight_mmap semantics exactly.

Isolated prototype (dee.cpp/experiments/kt_cpu_bridge). No production import.
"""
from __future__ import annotations

import struct

import torch

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
FP4_BLOCK = 32
E8M0_BIAS = 127
REALISTIC_SCALE_BYTES = (0x7D, 0x83)


def decode_e8m0(bits: torch.Tensor) -> torch.Tensor:
    if bits.dtype != torch.uint8:
        bits = bits.view(torch.uint8)
    if bool(((bits == 0xFF).any())):
        raise ValueError("0xFF scale byte rejected (KT +inf vs dee clamp; fail closed)")
    out = torch.empty(bits.shape, dtype=torch.float32)
    flat_in = bits.reshape(-1).to(torch.int64)
    flat_out = out.reshape(-1)
    for i, b in enumerate(flat_in.tolist()):
        if b == 0:
            flat_out[i] = 2.0 ** -127
        else:
            flat_out[i] = struct.unpack("<f", struct.pack("<I", b << 23))[0]
    return out


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    raw = packed if packed.dtype == torch.uint8 else packed.view(torch.uint8)
    low = (raw & 0x0F).to(torch.int64)
    high = ((raw >> 4) & 0x0F).to(torch.int64)
    out, packed_in = raw.shape
    lo_vals = FP4_TABLE[low]
    hi_vals = FP4_TABLE[high]
    return torch.stack([lo_vals, hi_vals], dim=-1).reshape(out, packed_in * 2)


def dequantize_weight(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    values = unpack_fp4(packed)
    out, in_dim = values.shape
    assert in_dim % 32 == 0, "in must be a multiple of 32 (MXFP4 block)"
    assert packed.shape == (out, in_dim // 2)
    assert scale.shape == (out, in_dim // 32)
    scales = decode_e8m0(scale)
    expanded = scales.unsqueeze(-1).expand(out, in_dim // 32, 32).reshape(out, in_dim)
    return values * expanded


def ue8m0_to_bf16_bits(b: int) -> int:
    """Lossless KT loader step: (u8 << 7) viewed as bf16 (exp=e, mant=0)."""
    if not 0 <= b <= 255:
        raise ValueError("scale byte out of range")
    if b == 0xFF:
        raise ValueError("0xFF rejected")
    return (b << 7) & 0xFFFF


def compact_scales_e8(scale_u8: torch.Tensor) -> torch.Tensor:
    """KT finalize_scale_e8 emulation: validate positive-pow2, return e8 bytes.

    Lossless for real checkpoints: compacted bytes equal the source bytes.
    """
    if scale_u8.dtype != torch.uint8:
        scale_u8 = scale_u8.view(torch.uint8)
    if bool((scale_u8 == 0xFF).any()):
        raise ValueError("0xFF rejected")
    f = decode_e8m0(scale_u8)
    # validate positive pow2 for nonzero bytes (mirrors KT sign/mant/exp check)
    for b, v in zip(scale_u8.reshape(-1).tolist(), f.reshape(-1).tolist()):
        if b == 0:
            continue
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
        assert (bits & 0x80000000) == 0 and (bits & 0x007FFFFF) == 0, (b, v)
        assert ((bits >> 23) & 0xFF) not in (0, 0xFF), (b, v)
    return scale_u8.clone()


def expand_e8_scales(e8: torch.Tensor) -> torch.Tensor:
    return decode_e8m0(e8)
