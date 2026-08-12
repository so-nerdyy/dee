"""DS9 building blocks: exact official DeepSeek-V4-Flash-0731 layer primitives.

Pure-torch ports of the pinned official inference code
(``official-source/inference/model.py`` + ``kernel.py``) for the DS9
"one complete official layer" milestone.  Everything here is shape- and
device-agnostic and requires only torch; every function is unit-tested
against independent implementations in ``tests/test_deepseek_v4_layer.py``.

Official semantics preserved exactly:

- RMSNorm: fp32 mean-of-squares, ``rsqrt(var + eps)``, learned weight, cast
  back to the input dtype (model.py RMSNorm.forward).
- YaRN rotary: ``precompute_freqs_cis`` + ``apply_rotary_emb`` (complex
  multiply, conjugate for inverse) ported line-for-line.
- FWHT: natural-order Walsh-Hadamard transform matching the
  ``fast_hadamard_transform`` package convention used by ``rotate_activation``
  (``y = scale * H @ x``, unnormalized H, caller supplies ``n ** -0.5``).
- e8m0 power-of-2 scales: the kernel's ``fast_round_scale`` = ceil-to-next-
  power-of-two via IEEE-754 bit manipulation (NOT round-to-nearest; torch's
  native float8_e8m0fnu cast is RNE and does NOT match, so the bit-exact
  implementation is used).
- FP8 act-quant simulation (``act_quant_kernel``, inplace=True): per-128 (or
  per-64) block amax -> scale -> clamp [-448, 448] -> e4m3 cast (RNE) ->
  dequantize -> cast back to bf16.
- FP4 act-quant simulation (``fp4_quant_kernel``, inplace=True): per-32
  block power-of-2 scale, clamp [-6, 6], round-to-nearest-even on the
  official 16-entry e2m1fn grid (FP4_TABLE), dequantize -> bf16.
- Sparse attention (``sparse_attn_kernel``): index-gathered top-k attention
  with the learnable ``attn_sink`` entering ONLY the softmax denominator
  (``sum_exp += exp(sink - max)`` after the loop, no numerator
  contribution), online-softmax equivalent accumulated in FP32.
- ``get_window_topk_idxs`` / ``get_compress_topk_idxs``: exact sliding-window
  and compression index matrices.
- ``hc_split_sinkhorn``: pre/post sigmoid gates + comb softmax, +eps, one
  initial column normalization, then ``sinkhorn_iters - 1`` row+column
  normalization rounds.

All functions are deterministic and differentiable-free; they operate on
whatever device the tensors are on.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch

# Official 16-entry FP4 e2m1fn lookup table (convert.py).
FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,  # positive half
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative half
    ],
    dtype=torch.float32,
)
FP8_MAX = 448.0
FP8_MIN = -448.0
FP4_MAX = 6.0


# ---------------------------------------------------------------------------
# Norms / rotary
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Official RMSNorm.forward: fp32 math, cast back to input dtype."""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    return (weight.float() * out).to(dtype)


@lru_cache(maxsize=8)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """YaRN-scaled rotary complex exponentials (model.py precompute_freqs_cis)."""
    dim = int(dim)
    seqlen = int(seqlen)

    def find_correction_dim(num_rotations: int, d: int, b: float, max_seq_len: int) -> float:
        return d * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(b))

    def find_correction_range(low_rot: int, high_rot: int, d: int, b: float, max_seq_len: int) -> tuple[int, int]:
        low = math.floor(find_correction_dim(low_rot, d, b, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, d, b, max_seq_len))
        return max(low, 0), min(high, d - 1)

    def linear_ramp_factor(min_v: float, max_v: float, d: int) -> torch.Tensor:
        if min_v == max_v:
            max_v += 0.001
        linear_func = (torch.arange(d, dtype=torch.float32) - min_v) / (max_v - min_v)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place rotary embedding via complex multiply (model.py)."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    xc = torch.view_as_real(xc * freqs_cis).flatten(-2)
    y.copy_(xc)
    return y


def hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Natural-order Walsh-Hadamard transform, ``y = scale * H @ x``.

    Matches ``fast_hadamard_transform.hadamard_transform(x, scale=...)`` with
    default ``normalize=None`` (unnormalized H, caller applies the scale).
    Requires the last dim to be a power of two.
    """
    n = x.size(-1)
    if n & (n - 1):
        raise ValueError(f"hadamard last dim must be power of two, got {n}")
    y = x.float().clone()
    h = 1
    while h < n:
        y = y.reshape(-1, 2 * h)
        a, b = y[:, :h], y[:, h:]
        y = torch.cat([a + b, a - b], dim=-1)
        h *= 2
    y = y.reshape(*x.shape)
    return (y * scale).to(x.dtype)


# ---------------------------------------------------------------------------
# Quantization simulations (bit-exact with the tilelang kernels)
# ---------------------------------------------------------------------------


def fast_log2_ceil(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) via IEEE-754 bit manipulation (kernel.py fast_log2_ceil).

    The kernel always computes in FP32, so the input is widened to FP32
    first (a bf16/fp16 input cannot be viewed as int32).
    """
    x = x.float()
    b = x.view(torch.int32)
    exp = (b >> 23) & 0xFF
    man = b & 0x7FFFFF
    return exp - 127 + (man != 0).to(torch.int32)


def fast_pow2(e: torch.Tensor) -> torch.Tensor:
    """2**e for integer e via IEEE-754 bit manipulation (kernel.py fast_pow2)."""
    bits = ((e.to(torch.int64) + 127) << 23).to(torch.int32)
    return bits.view(torch.float32)


def fast_round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """Ceil-to-next-power-of-two scale (kernel.py fast_round_scale)."""
    return fast_pow2(fast_log2_ceil(amax * max_inv))


def round_e2m1_grid(a: torch.Tensor) -> torch.Tensor:
    """Round non-negative values to the official e2m1fn grid, round-to-nearest-even.

    Grid (positive half of FP4_TABLE): [0, 0.5, 1, 1.5, 2, 3, 4, 6].
    Ties resolve to the even-indexed grid entry (IEEE RNE on the e2m1
    significand): 0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5.0->4.
    """
    a = a.clamp(0.0, FP4_MAX)
    return torch.where(
        a <= 0.25, 0.0,
        torch.where(
            a < 0.75, 0.5,
            torch.where(
                a <= 1.25, 1.0,
                torch.where(
                    a < 1.75, 1.5,
                    torch.where(
                        a <= 2.5, 2.0,
                        torch.where(
                            a < 3.5, 3.0,
                            torch.where(a <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )


def act_quant_inplace(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = "ue8m0",
) -> torch.Tensor:
    """In-place FP8 (e4m3) quantize+dequantize round trip, back to x.dtype.

    Mirrors kernel.py ``act_quant(x, block_size, scale_fmt, scale_dtype,
    inplace=True)``: per-row block amax (floor 1e-4), scale = power-of-two
    ceil (ue8m0) or linear ``amax/448``, clamp to [-448, 448], e4m3 cast
    (RNE), dequantize, cast back to the input dtype (bf16 in the model).
    """
    if x.dtype == torch.float32:
        raise ValueError("act_quant_inplace expects a half-precision tensor")
    n = x.size(-1)
    if n % block_size:
        raise ValueError(f"act_quant block {block_size} does not divide {n}")
    flat = x.reshape(-1, n)
    view = flat.unflatten(-1, (-1, block_size))  # [M, nb, block]
    amax = view.abs().amax(dim=-1).clamp_min(1e-4)  # [M, nb]
    if scale_fmt == "ue8m0":
        s = fast_round_scale(amax, 1.0 / FP8_MAX)
    else:
        s = amax * (1.0 / FP8_MAX)
    q = (view / s.unsqueeze(-1)).clamp(FP8_MIN, FP8_MAX).contiguous()
    q = q.to(torch.float8_e4m3fn).float() * s.unsqueeze(-1)
    x.copy_(q.reshape_as(flat).reshape_as(x).to(x.dtype))
    return x


def fp4_act_quant_inplace(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """In-place FP4 (e2m1fn) quantize+dequantize round trip, back to x.dtype.

    Mirrors kernel.py ``fp4_act_quant(x, block_size, inplace=True)``:
    per-row 32-block power-of-two scale (floor 6*2**-126), clamp [-6, 6],
    RNE rounding on the e2m1fn grid, dequantize, cast back to bf16.
    """
    n = x.size(-1)
    if n % block_size:
        raise ValueError(f"fp4_act_quant block {block_size} does not divide {n}")
    flat = x.reshape(-1, n)
    view = flat.unflatten(-1, (-1, block_size))
    amax = view.abs().amax(dim=-1).clamp_min(6.0 * (2.0 ** -126))
    s = fast_round_scale(amax, 1.0 / FP4_MAX)
    q = (view / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX)
    mag = round_e2m1_grid(q.abs())
    q = mag * q.sign()
    x.copy_((q * s.unsqueeze(-1)).reshape_as(flat).reshape_as(x).to(x.dtype))
    return x


# ---------------------------------------------------------------------------
# Sparse attention (official sparse_attn_kernel semantics, FP32 reference)
# ---------------------------------------------------------------------------


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Index-gathered sparse attention with sink-in-denominator only.

    q: [b, m, h, d]; kv: [b, n, d]; attn_sink: [h]; topk_idxs: [b, m, k]
    int32 with -1 marking invalid (masked) positions.

    Faithful FP32 equivalent of the official kernel:
      scores = scale * q @ kv[gathered]; invalid -> -inf
      m_h = row max; e = exp(scores - m_h); invalid -> 0
      sum_exp = sum_t e + exp(sink_h - m_h)     <- sink only in the denominator
      o = sum_t (e * kv) / sum_exp
    """
    b, m, h, d = q.shape
    k = topk_idxs.shape[-1]
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp(min=0)
    # gather over dim 1 of kv [b, n, d]; the index must share kv's ndim,
    # so flatten (m, k) into the gather rows and reshape afterwards
    flat_idx = idx.reshape(b, m * k).long()  # gather requires int64 index
    gathered = kv.gather(1, flat_idx.unsqueeze(-1).expand(b, m * k, d))
    gathered = gathered.reshape(b, m, k, d) * valid.unsqueeze(-1).float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered) * softmax_scale
    # valid is [b, m, k]; unsqueeze(2) -> [b, m, 1, k] to broadcast vs [b, m, h, k]
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    safe = scores.masked_fill(~valid.unsqueeze(2), -1e30)
    m_h = safe.amax(dim=-1, keepdim=True)  # [b, m, h, 1]
    e = torch.exp(scores - m_h)  # invalid -> exp(-inf - m) = 0
    sum_exp = e.sum(dim=-1, keepdim=True) + torch.exp(
        attn_sink[None, None, :, None] - m_h)
    o = (e.unsqueeze(-1) * gathered.unsqueeze(2)).sum(dim=3) / sum_exp
    return o.to(q.dtype)


# ---------------------------------------------------------------------------
# Index matrices (official get_window_topk_idxs / get_compress_topk_idxs)
# ---------------------------------------------------------------------------


def get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int
) -> torch.Tensor:
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [torch.arange(start_pos + 1, window_size),
             torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = torch.nn.functional.pad(
            torch.arange(start_pos + 1), (0, window_size - start_pos - 1),
            value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def get_compress_topk_idxs(
    ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int
) -> torch.Tensor:
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# Router selection (official Gate.forward, both score and hash variants)
# ---------------------------------------------------------------------------


def router_select(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    *,
    tid2eid: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    topk: int = 6,
    route_scale: float = 1.5,
    score_func: str = "sqrtsoftplus",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official Gate.forward for both routing variants.

    x: [n, dim].  Score layers: scores = sqrt(softplus(x @ W^T)); the bias
    shifts ONLY the selection; weights come from the un-shifted scores,
    normalized (non-softmax) and scaled by route_scale.

    Hash layers (tid2eid given): selection = tid2eid[input_ids] (I64
    [vocab, topk]); weights still come from the same score function on the
    gate weight, gathered at the hash-selected indices.

    Returns (selection_scores, selected_ids [n, topk], routing_weights
    [n, topk]).  ``selection_scores`` is the biased scores for score layers
    and the raw scores for hash layers.
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
    if tid2eid is not None:
        if input_ids is None:
            raise ValueError("hash routing requires input_ids")
        indices = tid2eid[input_ids].to(torch.long)  # [n, topk]
        selection_scores = scores
    else:
        if gate_bias is not None:
            scores = scores + gate_bias
        indices = scores.topk(topk, dim=-1)[1]
        selection_scores = scores
    weights = original_scores.gather(1, indices)
    if score_func != "softmax":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * route_scale
    return selection_scores, indices, weights


# ---------------------------------------------------------------------------
# Hyper-Connections split (official hc_split_sinkhorn_kernel, FP32 reference)
# ---------------------------------------------------------------------------


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pre/post/comb split with the official Sinkhorn normalization order.

    mixes: [..., mix_hc] with mix_hc = (2 + hc_mult) * hc_mult.
    Layout: [pre (hc) | post (hc) | comb (hc*hc)].
    """
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.size(-1) != mix_hc:
        raise ValueError(f"mixes last dim {mixes.size(-1)} != mix_hc {mix_hc}")
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (
        mixes[..., 2 * hc :].unflatten(-1, (hc, hc)) * hc_scale[2]
        + hc_base[2 * hc :].view(hc, hc)
    )
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
