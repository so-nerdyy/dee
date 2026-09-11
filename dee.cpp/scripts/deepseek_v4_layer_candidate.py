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

import time
from typing import Any, Optional

import numpy as np
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
                 provider: Any = None, diagnostics: bool = True):
        self.cache = cache
        self.loader = loader
        self.layer_id = layer_id
        self.fp16_payloads = fp16_payloads
        self.shared_payload = shared_payload
        self.cfg = cfg
        self.device = device
        self.provider = provider
        self.diagnostics = bool(diagnostics)
        self.tid2eid: Optional[torch.Tensor] = None
        self.stats = {"requests": 0, "hits": 0, "misses": 0, "staging_waits": 0}
        self.last_route: dict[str, Any] = {}

    # Profiling hooks are deliberately no-ops for the legacy cache backend.
    # The native subclass records CUDA events without synchronizing the hot
    # path, then resolves them after measured generation.
    def _profile_ffn_begin(self, x: torch.Tensor) -> None:
        del x

    def _profile_ffn_mark(self) -> None:
        pass

    def _profile_ffn_finish(self, *, rows: int) -> None:
        del rows

    def __call__(self, x: torch.Tensor, input_ids: torch.Tensor,
                 capture: Optional[dict[str, Any]]) -> torch.Tensor:
        cfg = self.cfg
        b, s, d = x.shape
        self._profile_ffn_begin(x)
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
        if self.diagnostics or capture is not None:
            self.last_route = {
                "expert_ids": ids.detach().cpu().tolist(),
                "routing_weights": weights.detach().float().cpu().tolist(),
            }
        else:
            self.last_route = {}
        self._profile_ffn_mark()
        moe_out, shared_out = self._run_experts(xf, ids, weights)
        if capture is not None:
            # Same labels/units as the reference _ffn_fp32_direct: moe_out is
            # the COMBINED (routed + shared) fp32 output; shared_out is the
            # shared-expert-only fp32 contribution.
            capture["moe_out"] = moe_out.reshape(b, s, d)
            capture["shared_out"] = shared_out.reshape(b, s, d)
        # Official MoE.forward returns y.type_as(x): cast the fp32
        # accumulation back to the input dtype (bf16) before hc_post.
        result = moe_out.reshape(b, s, d).to(x.dtype)
        self._profile_ffn_finish(rows=b * s)
        return result

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
            # Pin the freshly-staged entry.  It is resident by construction
            # (stage -> reserve -> resident), so a failure here is an
            # accounting anomaly, not a cache miss -- count it separately so
            # miss/hit totals stay exact.
            if not self.cache.pin(self.layer_id, skey):
                self.stats.setdefault("pin_failures", 0)
                self.stats["pin_failures"] += 1
            else:
                # CACHE1d: the GPU entry is now pinned permanently (survives
                # reset_state), so the host FP16 copy is dead weight.  Free it
                # to keep host RSS inside the 12 GB ceiling: 43 layers x 48 MiB
                # of eager shared copies were the second-largest RSS driver.
                self.stats.setdefault("shared_host_freed", 0)
                self.stats["shared_host_freed"] += 1
                self.shared_payload = None
                if self.provider is not None:
                    self.provider.shared_payloads.pop(self.layer_id, None)
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


class DeepseekV4NativeFfn(DeepseekV4CacheFfn):
    """Routed experts via pydee.Engine.moe_forward_experts (native FP4 decode
    + SwiGLU on CUDA); router + shared expert follow the DS8 torch path.

    The engine mmaps the same safetensors shards the harness loads, decodes the
    packed I8 + per-block F8_E8M0 weights on the transfer stream, and runs the
    SwiGLU expert forward natively.  Per-expert outputs are returned unweighted;
    the combine here applies the routing weight AFTER the down projection, which
    is exact because the official weight-before-w2 placement commutes with the
    linear w2 (``w * (h @ w2.T) == (w * h) @ w2.T``).
    """

    def __init__(self, *, engine: Any, layer_id: int, cfg: layer_ref.LayerConfig,
                 device: str = "cuda", shared_payload: Optional[dict[str, torch.Tensor]] = None,
                 provider: Any = None, diagnostics: bool = True,
                 profile_stages: bool = False):
        # Deliberately bypass the parent __init__: the native backend needs no
        # cache/loader/fp16_payloads for routed experts (the engine owns those).
        self.engine = engine
        self.layer_id = layer_id
        self.cfg = cfg
        self.device = device
        self.shared_payload = shared_payload
        self.provider = provider
        self.diagnostics = bool(diagnostics)
        self.profile_stages = bool(profile_stages)
        self.tid2eid: Optional[torch.Tensor] = None
        self.stats = {"requests": 0, "hits": 0, "misses": 0,
                      "native_fwd_ms": 0.0, "native_calls": 0,
                      "native_batch_calls": 0, "route_id_d2h_copies": 0,
                      "route_id_d2h_bytes": 0, "full_hidden_d2h_copies": 0,
                      "raw_expert_output_d2h_copies": 0,
                      "hidden_h2d_copies": 0, "numpy_bridge_calls": 0,
                      "host_synchronizations": 0,
                      "route_id_host_synchronizations": 0,
                      "native_output_synchronizations": 0}
        self.last_route: dict[str, Any] = {}
        self._native_hidden_fp16: Optional[torch.Tensor] = None
        self._native_raw_output: Optional[torch.Tensor] = None
        self._native_moe_output: Optional[torch.Tensor] = None
        self._native_route_ids_host: Optional[torch.Tensor] = None
        # Device-resident shared-expert payload (materialized lazily on the
        # first forward; the provider hands back CPU FP16 tensors).
        self._shared_dev: Optional[dict[str, torch.Tensor]] = None
        self._active_profile_events: Optional[list[torch.cuda.Event]] = None
        self._active_profile_boundary = 1
        self._profile_start_pos = -1
        self._profile_rows: list[dict[str, Any]] = []

    _PROFILE_PHASES = (
        "router",
        "routed_dispatch_and_native",
        "routed_combine",
        "shared_expert",
        "output_cast",
    )

    def _profile_ffn_begin(self, x: torch.Tensor) -> None:
        self._active_profile_events = None
        # Begin is also the recovery boundary for a previous call that raised
        # after recording only some of its events.  Without resetting the
        # cursor here, a later call starts at the stale boundary and leaves
        # one or more newly allocated events unrecorded.
        self._active_profile_boundary = 1
        if not self.profile_stages or not x.is_cuda:
            return
        stream = torch.cuda.current_stream(x.device)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        events[0].record(stream)
        self._active_profile_events = events

    def set_profile_start_pos(self, start_pos: int) -> None:
        """Tag the next recorded FFN row with the caller's decode position."""
        self._profile_start_pos = int(start_pos)

    def _profile_ffn_mark(self) -> None:
        events = self._active_profile_events
        if events is None:
            return
        boundary = self._active_profile_boundary
        if boundary >= len(events):
            raise RuntimeError("native FFN profiler recorded too many boundaries")
        events[boundary].record(torch.cuda.current_stream(self.device))
        self._active_profile_boundary = boundary + 1

    def _profile_ffn_finish(self, *, rows: int) -> None:
        events = self._active_profile_events
        if events is None:
            return
        self._profile_ffn_mark()
        if getattr(self, "_active_profile_boundary", 0) != len(events):
            raise RuntimeError("native FFN profiler recorded incomplete boundaries")
        self._profile_rows.append({
            "rows": int(rows),
            "events": events,
            "start_pos": int(self._profile_start_pos),
        })
        self._active_profile_events = None
        self._active_profile_boundary = 1

    def reset_stage_profile(self) -> None:
        self._active_profile_events = None
        self._active_profile_boundary = 1
        self._profile_start_pos = -1
        self._profile_rows = []

    def stage_profile_snapshot(self) -> dict[str, Any]:
        if not self.profile_stages:
            return {
                "enabled": False,
                "layer": self.layer_id,
                "calls": 0,
                "event_interval_count": 0,
                "totals_ms": {},
            }
        totals = {name: 0.0 for name in self._PROFILE_PHASES}
        calls = []
        per_start_pos: dict[str, dict[str, float]] = {}
        for index, stored in enumerate(self._profile_rows):
            events = stored["events"]
            durations = {
                name: float(events[phase].elapsed_time(events[phase + 1]))
                for phase, name in enumerate(self._PROFILE_PHASES)
            }
            for name, milliseconds in durations.items():
                totals[name] += milliseconds
            start_key = str(stored["start_pos"])
            position_totals = per_start_pos.setdefault(
                start_key, {name: 0.0 for name in self._PROFILE_PHASES})
            for name, milliseconds in durations.items():
                position_totals[name] += milliseconds
            calls.append({
                "call": index,
                "start_pos": stored["start_pos"],
                "rows": stored["rows"],
                "durations_ms": {
                    name: round(milliseconds, 6)
                    for name, milliseconds in durations.items()
                },
            })
        return {
            "enabled": True,
            "layer": self.layer_id,
            "device": str(self.device),
            "calls": len(calls),
            "event_interval_count": len(calls) * len(self._PROFILE_PHASES),
            "totals_ms": {
                name: round(milliseconds, 6)
                for name, milliseconds in totals.items()
            },
            "per_start_pos_ms": {
                key: {
                    name: round(milliseconds, 6)
                    for name, milliseconds in values.items()
                }
                for key, values in per_start_pos.items()
            },
            "calls_detail": calls,
        }

    def _run_experts(self, xf: torch.Tensor, ids: torch.Tensor,
                     weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a whole layer's routed experts without the NumPy bridge.

        The only host boundary is the compact int32 route-ID matrix required by
        the existing native scheduler. Hidden states, raw expert outputs, and
        the weighted FP32 reduction remain device-resident. The native API is
        deliberately required here: a missing device method is a hard error,
        never a silent fallback to the old per-token path.
        """
        cfg = self.cfg
        n, d = xf.shape
        batch_device = getattr(self.engine, "moe_forward_batch_device", None)
        if batch_device is None:
            raise RuntimeError(
                "native device MoE API unavailable; refusing host bridge fallback")

        if (self._native_hidden_fp16 is None or
                tuple(self._native_hidden_fp16.shape) != (n, d)):
            self._native_hidden_fp16 = torch.empty(
                (n, d), dtype=torch.float16, device=self.device)
        # copy_ performs the required FP32->FP16 conversion on the GPU while
        # reusing the same allocation on every layer/token call.
        self._native_hidden_fp16.copy_(xf)
        hidden_fp16 = self._native_hidden_fp16

        if (self._native_raw_output is None or
                tuple(self._native_raw_output.shape) != (n, cfg.topk, d)):
            self._native_raw_output = torch.empty(
                (n, cfg.topk, d), dtype=torch.float32, device=self.device)
        raw = self._native_raw_output

        # Exactly one compact route-ID transfer per layer. Reuse the host
        # buffer; the native scheduler only needs this small dispatch matrix.
        # Keep it alive through the pybind call because NumPy is a zero-copy
        # view of it.
        ids_shape = (n, cfg.topk)
        if (self._native_route_ids_host is None or
                tuple(self._native_route_ids_host.shape) != ids_shape):
            self._native_route_ids_host = torch.empty(
                ids_shape, dtype=torch.int32, device="cpu",
                pin_memory=bool(ids.is_cuda))
        ids_host_tensor = self._native_route_ids_host
        ids_host_tensor.copy_(ids.detach(), non_blocking=bool(ids.is_cuda))
        if ids.is_cuda:
            # The scheduler consumes the host array synchronously. This also
            # orders the preceding FP32->FP16 hidden copy before the native
            # engine's independent compute stream.
            torch.cuda.current_stream(ids.device).synchronize()
            self.stats["host_synchronizations"] += 1
            self.stats["route_id_host_synchronizations"] += 1
        ids_host = ids_host_tensor.numpy()
        self.stats["route_id_d2h_copies"] += 1
        self.stats["route_id_d2h_bytes"] += int(ids_host_tensor.numel() * 4)
        # The NumPy object is only a zero-copy view of the compact route IDs;
        # hidden states and expert outputs never cross through NumPy.
        self.stats["numpy_bridge_calls"] += 0

        t0 = time.perf_counter()
        ok = bool(batch_device(
            self.layer_id, hidden_fp16.data_ptr(), n, ids_host, cfg.topk,
            raw.data_ptr()))
        self.stats["native_fwd_ms"] += (time.perf_counter() - t0) * 1000.0
        self.stats["native_calls"] += 1
        self.stats["native_batch_calls"] += 1
        self.stats["requests"] += int(ids.numel())
        # moe_forward_batch_device is the non-pointer-batched exact API and
        # currently synchronizes its native compute stream before returning.
        self.stats["host_synchronizations"] += 1
        self.stats["native_output_synchronizations"] += 1
        if not ok:
            detail = ""
            if hasattr(self.engine, "last_error_message"):
                detail = self.engine.last_error_message()
            raise RuntimeError(
                f"native moe_forward_batch_device failed layer={self.layer_id}"
                + (f": {detail}" if detail else ""))
        self._profile_ffn_mark()

        if (self._native_moe_output is None or
                tuple(self._native_moe_output.shape) != (n, d)):
            self._native_moe_output = torch.empty(
                (n, d), dtype=torch.float32, device=self.device)
        moe = self._native_moe_output
        moe.zero_()
        weights_f32 = weights if weights.dtype == torch.float32 else weights.float()
        # Preserve the established rank-order FP32 reduction instead of using
        # an opaque reduction kernel that might reorder the top-K summation.
        for rank in range(cfg.topk):
            moe.addcmul_(
                raw[:, rank, :], weights_f32[:, rank].unsqueeze(1))
        self._profile_ffn_mark()

        shared_out = self._shared_forward(xf, hidden_fp16)
        self._profile_ffn_mark()
        return moe + shared_out, shared_out

    def _shared_forward(self, xf: torch.Tensor,
                        hidden_fp16: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shared expert (F8_E4M3) stays on the torch path (already FP16-expanded)."""
        n, d = xf.shape
        if not self.shared_payload and self.provider is not None:
            self.shared_payload = self.provider.get_shared_fp16_payload(self.layer_id)
        if not self.shared_payload:
            return torch.zeros(n, d, dtype=torch.float32, device=self.device)
        if self._shared_dev is None:
            # The provider hands back CPU FP16 tensors; move them to the
            # execution device once and reuse across every token/layer call.
            self._shared_dev = {k: v.to(self.device)
                                for k, v in self.shared_payload.items()}
        w1 = self._shared_dev["w1.weight"]
        w2 = self._shared_dev["w2.weight"]
        w3 = self._shared_dev["w3.weight"]
        xc = hidden_fp16 if hidden_fp16 is not None else xf.half().to(self.device)
        gate = torch.clamp((xc @ w1.t()).float(),
                           max=self.cfg.swiglu_limit)
        up = torch.clamp((xc @ w3.t()).float(),
                         min=-self.cfg.swiglu_limit, max=self.cfg.swiglu_limit)
        h = torch.nn.functional.silu(gate) * up
        return (h.half() @ w2.t()).float()


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
    diagnostics: bool = True,
    profile_stages: bool = False,
) -> layer_ref.DeepseekV4Layer:
    """Build the Freebuff candidate layer (cache-fp16 FFN backend).

    Hash-routed layers (``w["ffn"]["tid2eid"]`` set) use the learned table
    for expert selection; ``provider`` enables on-demand expert fetches for
    cache misses (DS10 full model).
    """
    ffn = DeepseekV4CacheFfn(cache=cache, loader=loader, layer_id=layer_id,
                             fp16_payloads=fp16_payloads,
                             shared_payload=shared_payload, cfg=cfg,
                             device=device, provider=provider,
                             diagnostics=diagnostics)
    ffn.attach_gate(w["ffn"]["gate_w"], w["ffn"]["gate_b"])
    ffn.attach_hash(w["ffn"].get("tid2eid"))
    return layer_ref.DeepseekV4Layer(
        cfg, w, device=device, max_batch=max_batch, ffn_fn=ffn,
        layer_id=layer_id, profile_stages=profile_stages)


def make_native_candidate_layer(
    cfg: layer_ref.LayerConfig,
    w: dict[str, torch.Tensor],
    *,
    engine: Any,
    layer_id: int,
    device: str = "cuda",
    max_batch: int = 1,
    shared_payload: Optional[dict[str, torch.Tensor]] = None,
    provider: Any = None,
    diagnostics: bool = True,
    profile_stages: bool = False,
) -> layer_ref.DeepseekV4Layer:
    """Build the native-engine candidate layer (routed experts via
    pydee.Engine.moe_forward_experts; router + shared expert on torch)."""
    ffn = DeepseekV4NativeFfn(engine=engine, layer_id=layer_id, cfg=cfg,
                              device=device, shared_payload=shared_payload,
                              provider=provider, diagnostics=diagnostics,
                              profile_stages=profile_stages)
    ffn.attach_gate(w["ffn"]["gate_w"], w["ffn"]["gate_b"])
    ffn.attach_hash(w["ffn"].get("tid2eid"))
    return layer_ref.DeepseekV4Layer(
        cfg, w, device=device, max_batch=max_batch, ffn_fn=ffn,
        layer_id=layer_id, profile_stages=profile_stages)
