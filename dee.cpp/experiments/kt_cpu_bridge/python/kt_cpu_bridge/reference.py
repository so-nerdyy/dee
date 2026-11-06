"""Reference (exact fp32) + KT-emulated (bf16-roundtrip) expert forward.

Mirrors scripts/deepseek_v4_expert_reference.py semantics; the KT path
reproduces KTransformers' numeric boundaries (bf16 wire, late weighting) so
Phase D can quantify the delta on CPU without AVX512 hardware.
"""
from __future__ import annotations

import torch

from .codec import decode_e8m0, dequantize_weight


def _bf16_roundtrip(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def expert_forward_reference(
    x: torch.Tensor,
    packed_w1: torch.Tensor, scale_w1: torch.Tensor,
    packed_w2: torch.Tensor, scale_w2: torch.Tensor,
    packed_w3: torch.Tensor, scale_w3: torch.Tensor,
    routing_weight: float = 1.0,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """Exact dee semantics: weight on intermediate BEFORE w2, full fp32."""
    w1 = dequantize_weight(packed_w1, scale_w1)
    w2 = dequantize_weight(packed_w2, scale_w2)
    w3 = dequantize_weight(packed_w3, scale_w3)
    gate = x.float() @ w1.transpose(0, 1)
    up = x.float() @ w3.transpose(0, 1)
    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    h = torch.nn.functional.silu(gate) * up
    h = routing_weight * h
    return h @ w2.transpose(0, 1)


def kt_emulated_forward(
    x: torch.Tensor,
    packed_w1: torch.Tensor, scale_w1: torch.Tensor,
    packed_w2: torch.Tensor, scale_w2: torch.Tensor,
    packed_w3: torch.Tensor, scale_w3: torch.Tensor,
    routing_weight: float = 1.0,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """KT-faithful numeric path: bf16 wire + late (post-w2) weighting."""
    w1 = dequantize_weight(packed_w1, scale_w1)
    w2 = dequantize_weight(packed_w2, scale_w2)
    w3 = dequantize_weight(packed_w3, scale_w3)
    xb = _bf16_roundtrip(x.float())
    gate = _bf16_roundtrip(xb @ w1.transpose(0, 1))
    up = _bf16_roundtrip(xb @ w3.transpose(0, 1))
    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    h = _bf16_roundtrip(torch.nn.functional.silu(gate) * up)
    y = h @ w2.transpose(0, 1)
    y = routing_weight * y  # KT late weighting (algebraically equivalent)
    return _bf16_roundtrip(y)


def error_metrics(ref: torch.Tensor, got: torch.Tensor) -> dict:
    r = ref.float().reshape(-1)
    g = got.float().reshape(-1)
    assert r.shape == g.shape
    abs_err = (r - g).abs()
    denom = r.abs().clamp_min(1e-3)
    rel = abs_err / denom
    finite = bool(torch.isfinite(r).all() and torch.isfinite(g).all())
    cos = float(torch.nn.functional.cosine_similarity(r.unsqueeze(0), g.unsqueeze(0))[0]) if finite else float("nan")
    import math

    def q(p: float) -> float:
        s, _ = torch.sort(rel)
        idx = min(len(s) - 1, int(math.ceil(p * len(s)) - 1))
        return float(s[idx])

    return {
        "max_abs": float(abs_err.max()),
        "mean_abs": float(abs_err.mean()),
        "mean_rel": float(rel.mean()),
        "p95_rel": q(0.95),
        "p99_rel": q(0.99),
        "cosine": cos,
        "finite": finite,
        "n": int(r.numel()),
    }
