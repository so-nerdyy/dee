"""Trusted reference for ONE DeepSeek-V4-Flash-0731 routed expert (DS7).

Implements the exact mathematical semantics of the official pinned inference
code (benchmark_reports/deepseek-v4-flash-0731-t4/official-source/inference/):

- FP4 expert weights are stored packed as I8 ``[out, in//2]`` (two e2m1fn
  values per byte).  Byte order follows convert.py's cast_e2m1fn_to_e4m3fn:
  ``low  = byte & 0x0F`` -> first value, ``high = (byte >> 4) & 0x0F`` ->
  second value, decoded through the official 16-entry FP4_TABLE
  (indices 0..7 positive, 8..15 negative).
- Scales are ``F8_E8M0`` (ue8m0) ``[out, in//32]``; decode is
  ``2^(bits - 127)`` (verified against torch.float8_e8m0fnu round trips).
- Weight dequantization: ``w[o, i] = fp4_value * scale[o, i//32]``.
- Expert forward (model.py Expert): ``up`` and ``gate`` are clamped to
  ``[-swiglu_limit, swiglu_limit]`` (gate max-only), then ``silu(gate) * up``,
  then the down projection.
- Router (model.py Gate): scores via sqrt(softplus(linear(x))), top-k
  selection, weights normalized (non-softmax) and multiplied by route_scale.Everything is computed in FP32 and requires only torch. This is the
trusted reference for DS7: the T4 candidate kernels are validated against it,
not the other way around. It is deliberately NOT bitwise-tied to any specific
CUDA reduction order.

Note on activation quantization: the official ``fp4_gemm`` path first runs
``act_quant`` (block-128 FP8) on the activation before the GEMM; this
reference intentionally computes in full FP32. The candidate side therefore
carries activation-quantization error that must fall inside the predeclared
DS7 tolerance -- near-bitwise agreement is NOT expected here.
"""

from __future__ import annotations

import torch

# Official 16-entry FP4 e2m1fn lookup table (convert.py).
FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,  # positive half
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative half
    ],
    dtype=torch.float32,
)

FP4_BLOCK_SIZE = 32  # elements per scale block on K
E8M0_BIAS = 127  # ue8m0: value = 2^(bits - 127)
FP8_MAX = 448.0
FP8_MIN = -448.0


def decode_e8m0(bits: torch.Tensor) -> torch.Tensor:
    """Decode F8_E8M0 (ue8m0) bytes to float32: 2^(bits - 127).

    Verified against torch.float8_e8m0fnu round-trips:
    0x7f -> 1.0, 0x80 -> 2.0, 0x86 -> 128.0, 0x88 -> 512.0, 0x76 -> 2^-9.

    The input may arrive as uint8, int8 (safetensors ``I8`` weight bytes), or
    float8_e8m0fnu (safetensors ``F8_E8M0`` scales). Always reinterpret the
    raw bits as uint8 -- a numeric cast (``.to(int64)``) would decode e.g.
    byte 0x81 (-> 4.0 as float8) as the integer 4 instead of 2^(129-127).
    """
    if bits.dtype != torch.uint8:
        bits = bits.view(torch.uint8)
    return torch.pow(2.0, (bits.to(torch.int64) - E8M0_BIAS).to(torch.float32))


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack an I8 [out, in//2] packed FP4 tensor to FP32 [out, in].

    Uses the official nibble order: low nibble -> element 2i,
    high nibble -> element 2i+1, decoded via FP4_TABLE.
    """
    if packed.dtype == torch.uint8:
        raw = packed
    else:
        raw = packed.view(torch.uint8)
    low = (raw & 0x0F).to(torch.int64)
    high = ((raw >> 4) & 0x0F).to(torch.int64)
    out, packed_in = raw.shape
    lo_vals = FP4_TABLE[low]  # [out, packed_in]
    hi_vals = FP4_TABLE[high]
    # Interleave: [out, packed_in, 2] -> [out, packed_in*2]
    values = torch.stack([lo_vals, hi_vals], dim=-1).reshape(out, packed_in * 2)
    return values


def dequantize_expert_weight(
    packed: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize one packed FP4 expert weight to FP32 [out, in].

    packed: I8 [out, in//2] (two FP4 per byte).
    scale:  F8_E8M0 [out, in//32] (one power-of-two scale per 32 inputs).
    Returns [out, in] FP32.
    """
    values = unpack_fp4(packed)  # [out, in]
    out, in_dim = values.shape
    scales = decode_e8m0(scale)  # [out, in//32]
    # Expand along the in dimension: w[o, i] = fp4 * scale[o, i//32].
    expanded = scales.unsqueeze(-1).expand(out, in_dim // 32, 32).reshape(out, in_dim)
    return values * expanded


def expert_forward(
    x: torch.Tensor,
    packed_w1: torch.Tensor,
    scale_w1: torch.Tensor,
    packed_w2: torch.Tensor,
    scale_w2: torch.Tensor,
    packed_w3: torch.Tensor,
    scale_w3: torch.Tensor,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """One routed expert: down(silu(gate(x)) * up(x)) with official clamps.

    x: [..., hidden] (last dim = 4096).
    Returns [..., moe_inter_dim] FP32.
    """
    w1 = dequantize_expert_weight(packed_w1, scale_w1)  # [inter, hidden]
    w2 = dequantize_expert_weight(packed_w2, scale_w2)  # [hidden, inter]
    w3 = dequantize_expert_weight(packed_w3, scale_w3)  # [inter, hidden]

    gate = x @ w1.transpose(0, 1)  # [..., inter]
    up = x @ w3.transpose(0, 1)  # [..., inter]

    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

    hidden = torch.nn.functional.silu(gate) * up  # [..., inter]
    return hidden @ w2.transpose(0, 1)  # [..., hidden]


def router_scores(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    score_func: str = "sqrtsoftplus",
    topk: int = 6,
    route_scale: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official routing: sqrt(softplus(x @ W^T)), top-k, normalize, scale.

    x: [n_tokens, dim].
    gate_weight: [n_routed_experts, dim].
    Returns (scores, selected_expert_ids, routing_weights).
    """
    scores = x.float() @ gate_weight.float().transpose(0, 1)
    if score_func == "softmax":
        scores = torch.softmax(scores, dim=-1)
    elif score_func == "sigmoid":
        scores = torch.sigmoid(scores)
    elif score_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(scores).sqrt()
    else:
        raise ValueError(f"unknown score_func {score_func!r}")
    original_scores = scores
    if bias is not None:
        scores = scores + bias
    indices = scores.topk(topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    if score_func != "softmax":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * route_scale
    return scores, indices, weights


def validate_fp4_consistency(packed: torch.Tensor, scale: torch.Tensor) -> dict[str, float]:
    """Sanity metrics for a dequantized expert weight (DS7 diagnostics)."""
    w = dequantize_expert_weight(packed, scale)
    return {
        "shape": list(w.shape),
        "min": float(w.min()),
        "max": float(w.max()),
        "mean": float(w.mean()),
        "abs_mean": float(w.abs().mean()),
        "nonzeros": float((w != 0).float().mean()),
    }


# Realistic ue8m0 scale bytes: 0x7d..0x82 -> scales 0.25..8.0. Random bytes
# up to 255 decode to 2^128, which overflows FP32 to inf and turns the output
# NaN, so self-tests must stay inside the representable range.
REALISTIC_SCALE_BYTES = (0x7D, 0x83)


def main() -> None:
    """Self-test against the official semantics (no checkpoint needed)."""
    torch.manual_seed(0)
    hidden, inter, out = 4096, 2048, 2048
    packed_w1 = torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8)
    scale_w1 = torch.randint(*REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8)
    packed_w2 = torch.randint(0, 256, (out, inter // 2), dtype=torch.uint8)
    scale_w2 = torch.randint(*REALISTIC_SCALE_BYTES, (out, inter // 32), dtype=torch.uint8)
    packed_w3 = torch.randint(0, 256, (inter, hidden // 2), dtype=torch.uint8)
    scale_w3 = torch.randint(*REALISTIC_SCALE_BYTES, (inter, hidden // 32), dtype=torch.uint8)
    x = torch.randn(2, hidden)

    y = expert_forward(x, packed_w1, scale_w1, packed_w2, scale_w2, packed_w3, scale_w3)
    print("expert_forward out:", tuple(y.shape), "mean", float(y.mean().abs()))
    assert torch.isfinite(y).all(), "expert_forward must not produce NaN/inf on realistic scales"
    print("self-test OK")


if __name__ == "__main__":
    main()
