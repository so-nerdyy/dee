"""DS9 Freebuff candidate: one official layer on T4 through the DS8 runtime.

The candidate executes the SAME official layer math as the trusted reference
(``deepseek_v4_layer_reference.DeepseekV4Layer``) but:

- attention / compressor / indexer / mHC run as native torch CUDA kernels on
  the T4 (hybrid bring-up bridge; a fully native Freebuff kernel set is a
  later milestone and must NOT be compared for latency against this);
- the FFN (top-6 routed experts + shared expert) runs through the sealed
  DS8 ``DeepSeekExpertCache`` + ``DeepSeekExpertLoader`` with FP16-expanded
  payloads staged on a side stream and guarded by per-expert completion
  events -- exactly the runtime accepted at DS8.

Modes stay separated:
- reference mode: ``ffn_backend="fp32"`` (the trusted FP32 oracle);
- candidate mode: ``ffn_backend="cache_fp16"`` (this module).

No latency comparison is made between the two.  ``performance_comparable``
stays false for DS9.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from scripts import deepseek_v4_expert_reference as ds7
from scripts import deepseek_v4_layer_reference as layer_ref
from scripts import deepseek_v4_moe_reference as moe


class DeepseekV4CacheFfn:
    """Cache-backed top-6 routed + shared FFN executed on the T4 (DS8 path).

    Callable: (x_bf16 [b, s, hidden], input_ids, capture) -> fp32 [b, s, hidden].
    """

    def __init__(self, *, cache: Any, loader: Any, layer_id: int,
                 fp16_payloads: dict[int, dict[str, torch.Tensor]],
                 shared_payload: dict[str, torch.Tensor],
                 cfg: layer_ref.LayerConfig, device: str = "cuda",
                 provider: Any = None):
        self.cache = cache
        self.loader = loader
        self.layer_id = layer_id
        self.fp16_payloads = fp16_payloads
        self.shared_payload = shared_payload
        self.cfg = cfg
        self.device = device
        self.provider = provider
        self.tid2eid: Optional[torch.Tensor] = None
        self.stats = {"requests": 0, "hits": 0, "misses": 0, "staging_waits": 0}
        self.last_route: dict[str, Any] = {}

    def __call__(self, x: torch.Tensor, input_ids: torch.Tensor,
                 capture: Optional[dict[str, Any]]) -> torch.Tensor:
        cfg = self.cfg
        b, s, d = x.shape
        xf = x.reshape(-1, d).float()
        if self.tid2eid is not None:
            # Hash-routed layer: selection from the learned table, weights
            # from the official score function (no bias).
            from scripts import deepseek_v4_layer_common as common
            _, ids, weights = common.router_select(
                xf, self.gate_w, None, tid2eid=self.tid2eid,
                input_ids=input_ids.flatten(), topk=cfg.topk,
                route_scale=cfg.route_scale, score_func="sqrtsoftplus")
            scores = torch.nn.functional.softplus(
                xf @ self.gate_w.float().transpose(0, 1)).sqrt()
            if capture is not None:
                capture["router_scores"] = scores
                capture["router_bias_scores"] = scores
                capture["expert_ids"] = ids
                capture["routing_weights"] = weights
        else:
            scores, ids, weights = ds7.router_scores(
                xf, self.gate_w, bias=self.gate_b, score_func="sqrtsoftplus",
                topk=cfg.topk, route_scale=cfg.route_scale)
            if capture is not None:
                capture["router_scores"] = scores
                capture["router_bias_scores"] = (
                    scores + self.gate_b if self.gate_b is not None else scores)
                capture["expert_ids"] = ids
                capture["routing_weights"] = weights
        self.last_route = {
            "expert_ids": ids.detach().cpu().tolist(),
            "routing_weights": weights.detach().float().cpu().tolist(),
        }
        moe_out, shared_out = self._run_experts(xf, ids, weights)
        if capture is not None:
            # Same labels/units as the reference _ffn_fp32_direct: moe_out is
            # the COMBINED (routed + shared) fp32 output; shared_out is the
            # shared-expert-only fp32 contribution.
            capture["moe_out"] = moe_out.reshape(b, s, d)
            capture["shared_out"] = shared_out.reshape(b, s, d)
        # Official MoE.forward returns y.type_as(x): cast the fp32
        # accumulation back to the input dtype (bf16) before hc_post.
        return moe_out.reshape(b, s, d).to(x.dtype)

    # -- wiring ------------------------------------------------------------
    def attach_gate(self, gate_w: torch.Tensor, gate_b: Optional[torch.Tensor]) -> None:
        self.gate_w = gate_w
        self.gate_b = gate_b

    def attach_hash(self, tid2eid: Optional[torch.Tensor]) -> None:
        self.tid2eid = tid2eid

    # -- expert execution --------------------------------------------------
    def _run_experts(self, xf: torch.Tensor, ids: torch.Tensor,
                     weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        n, d = xf.shape
        # Keep the accumulator on the execution device: v5 crashed at the
        # post-FFN _hc_post with 'found at least two devices, cuda:0 and
        # cpu!' because the FFN returned CPU tensors (the DS8 harness needed
        # .cpu() for host-side comparison, but here the FFN output feeds the
        # CUDA layer's hc_post).
        moe = torch.zeros(n, d, dtype=torch.float32, device=self.device)
        groups: dict[int, list[tuple[int, float]]] = {}
        for tok in range(n):
            for pos in range(cfg.topk):
                eid = int(ids[tok, pos])
                w = float(weights[tok, pos])
                if w == 0.0:
                    continue
                groups.setdefault(eid, []).append((tok, w))
        for eid, pairs in groups.items():
            self.stats["requests"] += 1
            entry = self.cache.get(self.layer_id, eid)
            if entry is None:
                self.stats["misses"] += 1
                payload = self.fp16_payloads.get(eid)
                if payload is None and self.provider is not None:
                    # Dynamic DS10 experts retain only compact official bytes
                    # in the provider's bounded per-layer LRU.  Do not insert
                    # the expanded FP16 payload here: this dict is unbounded
                    # across prefill routes and previously leaked host RSS.
                    payload = self.provider.get_fp16_payload(
                        self.layer_id, eid)
                if payload is None:
                    raise RuntimeError(
                        f"candidate routed to expert {eid} outside the "
                        f"reference-discovered union "
                        f"{sorted(self.fp16_payloads)}; route divergence "
                        f"between candidate and reference")
                entry = self.loader.stage(self.layer_id, eid,
                                          payload,
                                          metadata={"expert_type": "routed"})
            else:
                self.stats["hits"] += 1
            if entry.ready_event is not None:
                self.stats["staging_waits"] += 1
            self.loader.wait(entry)
            payload = entry.payload
            toks = [pair[0] for pair in pairs]
            ws = torch.tensor([[pair[1]] for pair in pairs],
                              dtype=torch.float32).reshape(-1, 1)
            xc = xf[toks].half().to(self.device)
            gate = torch.clamp((xc @ payload["w1.weight"].t()).float(),
                               max=cfg.swiglu_limit)
            up = torch.clamp((xc @ payload["w3.weight"].t()).float(),
                             min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
            h = torch.nn.functional.silu(gate) * up
            h = ws.to(self.device) * h  # official: weight before w2
            out = (h.half() @ payload["w2.weight"].t()).float()
            moe[toks] += out
        # shared expert (unweighted).  CACHE1: the shared expert runs every
        # token, every layer (10.8% of all accesses on the sealed trace), so
        # pin it permanently after the first stage -- the pin/unpin machinery
        # already exists in DeepSeekExpertCache but was never called, which
        # left the 43 shared experts (48 MiB each) churning through the budget
        # and being re-fetched over HTTP every single token.
        self.stats["requests"] += 1
        skey = -1
        s_entry = self.cache.get(self.layer_id, skey)
        if s_entry is None:
            self.stats["misses"] += 1
            if not self.shared_payload and self.provider is not None:
                self.shared_payload = self.provider.get_shared_fp16_payload(
                    self.layer_id)
            s_entry = self.loader.stage(self.layer_id, skey, self.shared_payload,
                                        metadata={"expert_type": "shared"})
            if not self.cache.pin(self.layer_id, skey):
                self.stats["misses"] += 1  # pin failed: not resident
        else:
            self.stats["hits"] += 1
        if s_entry.ready_event is not None:
            self.stats["staging_waits"] += 1
        self.loader.wait(s_entry)
        sp = s_entry.payload
        xc = xf.half().to(self.device)
        gate = torch.clamp((xc @ sp["w1.weight"].t()).float(),
                           max=cfg.swiglu_limit)
        up = torch.clamp((xc @ sp["w3.weight"].t()).float(),
                         min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
        h = torch.nn.functional.silu(gate) * up
        shared_out = (h.half() @ sp["w2.weight"].t()).float()
        return moe + shared_out, shared_out


def build_fp16_payloads(
    routed_raw: dict[int, dict[str, torch.Tensor]],
) -> dict[int, dict[str, torch.Tensor]]:
    """Dequantize packed-FP4 routed experts to FP16 payloads (DS8 family C)."""
    return {
        eid: {
            "w1.weight": ds7.dequantize_expert_weight(
                t["w1.weight"], t["w1.scale"]).half(),
            "w2.weight": ds7.dequantize_expert_weight(
                t["w2.weight"], t["w2.scale"]).half(),
            "w3.weight": ds7.dequantize_expert_weight(
                t["w3.weight"], t["w3.scale"]).half(),
        }
        for eid, t in routed_raw.items()
    }


def build_shared_fp16_payload(
    shared_raw: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Dequantize the F8_E4M3 shared expert to an FP16 payload."""
    return {
        "w1.weight": moe.dequantize_fp8_e4m3(
            shared_raw["w1.weight"], shared_raw["w1.scale"]).half(),
        "w2.weight": moe.dequantize_fp8_e4m3(
            shared_raw["w2.weight"], shared_raw["w2.scale"]).half(),
        "w3.weight": moe.dequantize_fp8_e4m3(
            shared_raw["w3.weight"], shared_raw["w3.scale"]).half(),
    }


def make_candidate_layer(
    cfg: layer_ref.LayerConfig,
    w: dict[str, torch.Tensor],
    *,
    device: str = "cuda",
    max_batch: int = 1,
    cache: Any,
    loader: Any,
    layer_id: int,
    fp16_payloads: dict[int, dict[str, torch.Tensor]],
    shared_payload: dict[str, torch.Tensor],
    provider: Any = None,
) -> layer_ref.DeepseekV4Layer:
    """Build the Freebuff candidate layer (cache-fp16 FFN backend).

    Hash-routed layers (``w["ffn"]["tid2eid"]`` set) use the learned table
    for expert selection; ``provider`` enables on-demand expert fetches for
    cache misses (DS10 full model).
    """
    ffn = DeepseekV4CacheFfn(cache=cache, loader=loader, layer_id=layer_id,
                             fp16_payloads=fp16_payloads,
                             shared_payload=shared_payload, cfg=cfg,
                             device=device, provider=provider)
    ffn.attach_gate(w["ffn"]["gate_w"], w["ffn"]["gate_b"])
    ffn.attach_hash(w["ffn"].get("tid2eid"))
    return layer_ref.DeepseekV4Layer(cfg, w, device=device, max_batch=max_batch,
                                     ffn_fn=ffn)
