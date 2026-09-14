"""DS8 trusted reference: full official MoE FFN for one DeepSeek-V4 layer.

Extends the DS7 single-expert reference (``deepseek_v4_expert_reference``,
left byte-identical) with the exact official semantics for a complete layer
FFN, verified against ``official-source/inference/model.py``:

- Router (``Gate.forward``): scores = sqrt(softplus(x @ gate_w^T)); the bias
  shifts only the SELECTION; routing weights come from the un-shifted scores,
  are normalized (non-softmax) and multiplied by ``route_scale`` (1.5).
- Routed expert (``Expert.forward``): gate=clamp(w1(x), max=limit),
  up=clamp(w3(x), -limit..limit), h=silu(gate)*up, then the routing weight is
  applied to ``h`` BEFORE the down projection, then ``w2``.
- Shared expert (``shared_experts``): a SEPARATE expert whose weights are
  stored F8_E4M3 with F8_E8M0 block-128 scales (NOT packed FP4). Applied with
  NO routing weight.
- Combination (``MoE.forward``): y = sum over selected experts of
  weight*expert(x) for routed tokens, then y += shared_experts(x). All in
  FP32.

Everything is FP32 except the storage dtypes (F8_E4M3 shared weights,
F8_E8M0 scales), which are dequantized here. This is the trusted reference
for DS8; the T4 candidate kernels are validated against it.
"""

from __future__ import annotations

import torch

from scripts import deepseek_v4_expert_reference as ds7
from scripts import deepseek_v4_layer_common as common

FP8_E4M3_BLOCK = 128  # official fp8 weight block size (per 128x128 scale tile)


def dequantize_fp8_e4m3(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an F8_E4M3 [out, in] weight with F8_E8M0 [out//128, in//128] scales.

    Returns FP32 [out, in].  ``weight`` may arrive as float8_e4m3fn
    (safetensors F8_E4M3) or raw int8 bytes (view-reinterpreted).
    """
    if weight.dtype != torch.float8_e4m3fn:
        weight = weight.view(torch.float8_e4m3fn)
    values = weight.float()
    out, in_dim = values.shape
    scales = ds7.decode_e8m0(scale)  # [ceil(out/128), ceil(in/128)]
    if scales.shape != ((out + FP8_E4M3_BLOCK - 1) // FP8_E4M3_BLOCK,
                        (in_dim + FP8_E4M3_BLOCK - 1) // FP8_E4M3_BLOCK):
        raise ValueError(
            f"scale shape {scales.shape} does not match official block-128 "
            f"tiling for weight {list(values.shape)}"
        )
    # Expand per 128x128 tile: w[o, i] = fp8[o, i] * scale[o//128, i//128].
    expanded = scales.repeat_interleave(
        FP8_E4M3_BLOCK, dim=0).repeat_interleave(FP8_E4M3_BLOCK, dim=1)
    expanded = expanded[:out, :in_dim]
    return values * expanded


def shared_expert_forward(
    x: torch.Tensor,
    t: dict[str, torch.Tensor],
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """One shared expert (F8_E4M3 storage): down(silu(gate)*up), no weights.

    x: [..., hidden].  ``t`` keys: w1.weight/w1.scale, w2.weight/w2.scale,
    w3.weight/w3.scale (the official shared_experts naming).
    Returns [..., hidden] FP32.

    NOTE: the shared expert's weights are dequantized to FP32 here and the
    SwiGLU math runs directly -- the shared path must NOT be routed through
    ``ds7.expert_forward``, which re-interprets its weight arguments as
    packed-FP4 I8 bytes.
    """
    w1 = dequantize_fp8_e4m3(t["w1.weight"], t["w1.scale"])
    w2 = dequantize_fp8_e4m3(t["w2.weight"], t["w2.scale"])
    w3 = dequantize_fp8_e4m3(t["w3.weight"], t["w3.scale"])
    gate = x @ w1.transpose(0, 1)  # [.., inter]
    up = x @ w3.transpose(0, 1)  # [.., inter]
    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    h = torch.nn.functional.silu(gate) * up
    return h @ w2.transpose(0, 1)  # [.., hidden]


def routed_expert_forward_weighted(
    x: torch.Tensor,
    weights: torch.Tensor,
    t: dict[str, torch.Tensor],
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """One routed expert with the official weight placement (before w2).

    x: [n, hidden].  weights: [n, 1] routing weights for this expert.
    ``t`` keys are the packed FP4 routed-expert tensors (w1/w2/w3 + scales).
    Returns [n, hidden] FP32.
    """
    w1 = ds7.dequantize_expert_weight(t["w1.weight"], t["w1.scale"])
    w2 = ds7.dequantize_expert_weight(t["w2.weight"], t["w2.scale"])
    w3 = ds7.dequantize_expert_weight(t["w3.weight"], t["w3.scale"])

    gate = x @ w1.transpose(0, 1)  # [n, inter]
    up = x @ w3.transpose(0, 1)
    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    h = torch.nn.functional.silu(gate) * up  # [n, inter]
    h = weights * h  # official: routing weight applied to the intermediate
    return h @ w2.transpose(0, 1)  # [n, hidden]


def moe_layer_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor | None,
    routed: dict[int, dict[str, torch.Tensor]],
    shared: dict[str, torch.Tensor] | None,
    *,
    topk: int = 6,
    route_scale: float = 1.5,
    score_func: str = "sqrtsoftplus",
    swiglu_limit: float = 10.0,
    keep_per_expert: bool = False,
    tid2eid: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Official MoE FFN for one layer, in FP32.

    x: [n_tokens, hidden].  ``routed`` maps expert_id -> dict of packed
    FP4 tensors.  ``shared`` is the F8_E4M3 shared-expert tensors (may be
    None for tests without a shared expert).

    Hash layers: pass ``tid2eid`` (I64 [vocab, topk]) and ``input_ids``
    [n_tokens]; selection follows the hash table, weights come from the
    sqrtsoftplus scores gathered at the hash-selected indices.

    Returns:
      scores        [n, n_routed]  selection scores (biased for score
                                    layers, raw for hash layers)
      bias_scores   [n, n_routed]  scores + bias (== scores when no bias)
      expert_ids    [n, topk]
      routing_weights [n, topk]
      per_expert    dict[expert_id -> [n, hidden]] (only when keep_per_expert)
      shared_output [n, hidden]     (zeros if no shared expert)
      moe_output    [n, hidden]     weighted routed sum + shared
    """
    n = x.shape[0]
    if tid2eid is not None:
        # Hash routing: selection from the learned table, weights from the
        # official score function (no bias in hash layers).
        if input_ids is None:
            raise ValueError("hash routing requires input_ids")
        _, indices, weights = common.router_select(
            x, gate_weight, None, tid2eid=tid2eid, input_ids=input_ids,
            topk=topk, route_scale=route_scale, score_func=score_func)
        scores = torch.nn.functional.softplus(
            x.float() @ gate_weight.float().transpose(0, 1)).sqrt()
        bias_scores = scores
    else:
        scores, indices, weights = ds7.router_scores(
            x, gate_weight, bias=gate_bias, score_func=score_func,
            topk=topk, route_scale=route_scale)
        bias_scores = scores + gate_bias if gate_bias is not None else scores

    # Group tokens by selected expert (official MoE.forward pattern), so each
    # expert's weights are dequantized once and executed on all its tokens.
    groups: dict[int, list[tuple[int, float]]] = {}
    for tok in range(n):
        for pos in range(topk):
            eid = int(indices[tok, pos])
            w = float(weights[tok, pos])
            if w == 0.0:
                continue
            groups.setdefault(eid, []).append((tok, w))

    per_expert: dict[int, torch.Tensor] = {}
    moe = torch.zeros(n, x.shape[1], dtype=torch.float32)
    for eid, pairs in groups.items():
        toks = [pair[0] for pair in pairs]
        ws = torch.tensor([[pair[1]] for pair in pairs], dtype=torch.float32)
        out = routed_expert_forward_weighted(
            x[toks], ws, routed[eid], swiglu_limit=swiglu_limit)
        moe[toks] += out
        if keep_per_expert:
            bucket = per_expert.setdefault(
                eid, torch.zeros(n, x.shape[1], dtype=torch.float32))
            bucket[toks] += out

    shared_output = torch.zeros(n, x.shape[1], dtype=torch.float32)
    if shared is not None:
        shared_output = shared_expert_forward(x, shared, swiglu_limit=swiglu_limit)
    moe = moe + shared_output

    return {
        "scores": scores,
        "bias_scores": bias_scores,
        "expert_ids": indices,
        "routing_weights": weights,
        "per_expert": per_expert,
        "shared_output": shared_output,
        "moe_output": moe,
    }


def main() -> None:
    """Self-test against the official semantics (no checkpoint needed)."""
    torch.manual_seed(0)
    hidden, inter = 64, 32  # small multiples of 32 so FP4/FP8 blocks divide
    n, n_experts, topk = 4, 8, 2

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
        "w1.scale": torch.full((inter // 128 + 1, hidden // 128 + 1), 0x7F, dtype=torch.uint8),
        "w2.weight": (torch.rand(hidden, inter) * 2 - 1).to(torch.float8_e4m3fn),
        "w2.scale": torch.full((hidden // 128 + 1, inter // 128 + 1), 0x7F, dtype=torch.uint8),
        "w3.weight": (torch.rand(inter, hidden) * 2 - 1).to(torch.float8_e4m3fn),
        "w3.scale": torch.full((inter // 128 + 1, hidden // 128 + 1), 0x7F, dtype=torch.uint8),
    }
    gate = torch.randn(n_experts, hidden) * 0.5
    bias = torch.randn(n_experts) * 0.1
    x = torch.randn(n, hidden)

    out = moe_layer_forward(x, gate, bias, routed, shared,
                            topk=topk, keep_per_expert=True)
    assert out["moe_output"].shape == (n, hidden)
    assert torch.isfinite(out["moe_output"]).all()
    assert out["expert_ids"].shape == (n, topk)
    assert torch.allclose(out["routing_weights"].sum(dim=-1),
                          torch.full((n,), route_scale := 1.5))
    assert set(out["per_expert"]) <= set(range(n_experts))
    # weighted routed sum + shared == moe_output
    routed_sum = torch.zeros_like(out["moe_output"])
    for eid, contrib in out["per_expert"].items():
        routed_sum += contrib
    assert torch.allclose(routed_sum + out["shared_output"], out["moe_output"])
    # selection uses biased scores
    bias_scores = torch.nn.functional.softplus(x @ gate.t()).sqrt() + bias
    assert torch.equal(out["expert_ids"], bias_scores.topk(topk, dim=-1)[1])
    print("moe self-test OK")


if __name__ == "__main__":
    main()
