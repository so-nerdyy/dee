#!/usr/bin/env python3
"""Isolated route-pipeline discrete-event simulator (NOT production).

Models one decode token through the 43-layer DSV4 stack as an event DAG and
compares LEGAL schedules only. Dependency rules enforced fail-closed
(violations raise instead of simulating):

  R1 route_known(L) requires combine_done(L-1), EXCEPT hash layers {0,1,2}
     whose expert IDs come from tid2eid[input_ids] (source:
     deepseek_v4_layer_common.router_select) and are known at token start.
     Hash routing WEIGHTS still need that layer's hidden state.
  R2 stage_submit(expert, L) requires route_known(L) (official IDs only;
     predictor routes are never admitted — there is no knob for them).
  R3 consume(expert, L) requires stage_done(expert, L).
  R4 combine_done(L) requires all consume(*, L) + shared_done(L).
  R5 route_known(L+1) (score layers) requires combine_done(L) — the router
     input is the post-residual hidden state.
  R6 Nothing for token t+1 exists before sample(t) (sampling is a data
     dependency on the final logits). The simulator models one token.

Latencies are explicit parameters (defaults labeled ASSUMED for structure,
calibrate from host.json/profiler). Millisecond outputs are SIMULATED, never
measurements; the structural outputs (legal edge lists, which rule binds)
are the actual product.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path

HASH_LAYERS = frozenset((0, 1, 2))  # layer_id < n_hash_layers (model.py)


@dataclass
class Latency:
    """Per-operation milliseconds. Defaults are ASSUMED placeholders."""
    dense_attn_norm: float = 2.0     # attention+norms before the router
    router: float = 0.4              # router GEMM + topk + route-ID D2H
    route_sync: float = 0.3          # host stream sync for route IDs
    h2d_per_expert: float = 1.0      # 12.75 MiB packed miss transfer
    decode_per_expert: float = 0.6   # full FP16 materialization
    gemm_per_expert: float = 0.5     # gate+up+down at batch-1
    combine: float = 0.2             # host-driven weighted reduction
    shared_expert: float = 1.5       # shared expert forward (indep. of routed)
    layer_sync: float = 0.5          # cudaStreamSynchronize per layer batch
    handoff_gpu: float = 0.4         # hidden-state A->host->B at layer 22
    source: str = "ASSUMED (calibrate from profiler/host.json)"


@dataclass
class ScheduleFlags:
    hash_prefetch: bool = False   # submit L0-2 expert reads at token start
    shared_overlap: bool = False  # run shared expert across routed staging
    no_layer_sync: bool = False   # one sync per token (ASSUMED event handoff)
    batch_submit: bool = False    # all of a layer's reads at route-known
    # NOTE: batch_submit changes only host submission order; device work per
    # expert is unchanged, so it is modeled as removing host-serial gaps.
    host_serial_gap: float = 0.05


@dataclass
class SimResult:
    critical_ms: float
    schedule: str
    edges: list = field(default_factory=list)   # (op, layer, expert, t0, t1)
    bound_by: list = field(default_factory=list)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"dependency violation: {msg}")


def simulate(n_layers: int = 43, topk: int = 6, hit_rate: float = 0.0,
             lat: Latency | None = None,
             flags: ScheduleFlags | None = None,
             schedule: str = "baseline") -> SimResult:
    lat = lat or Latency()
    flags = flags or ScheduleFlags()
    _check(0.0 <= hit_rate <= 1.0, "hit_rate in [0,1]")
    _check(n_layers > 0 and topk > 0, "positive geometry")

    now: dict[str, float] = {}
    edges: list = []
    bound: list = []

    def at(key: str) -> float:
        _check(key in now, f"{key} used before defined")
        return now[key]

    def put(key: str, t: float, label: str, layer: int, expert: int = -1) -> None:
        now[key] = t
        edges.append((label, layer, expert, round(t, 6)))

    # Token start: embedding + first-token sampling dependency noted.
    put("token_start", 0.0, "token_start", -1)
    hash_ids_known = 0.0  # R1 exception: tid2eid needs only input_ids.

    if flags.hash_prefetch:
        for layer in sorted(HASH_LAYERS):
            if layer >= n_layers:
                continue
            for rank in range(topk):
                # Staging submit needs no hidden state for hash IDs (R1/R2).
                done = hash_ids_known + lat.h2d_per_expert
                put(f"stage_done:{layer}:{rank}", done, "stage_prefetch",
                    layer, rank)
        bound.append("hash_prefetch:submitted L0-2 reads at token start")

    combine_prev = 0.0
    syncs = 0
    for layer in range(n_layers):
        is_hash = layer in HASH_LAYERS
        # All layer compute chains after the previous combine (residual data
        # dependency); ONLY staging submits for hash layers may go early.
        compute_start = combine_prev if layer > 0 else 0.0
        if layer > 0 and not is_hash:
            bound.append(f"route_known:{layer} bound by R5 combine_done:{layer-1}")
        if layer > 0 and is_hash:
            bound.append(f"layer:{layer} compute bound by combine_done:{layer-1}; "
                         "only staging submits lead")
        t = compute_start + lat.dense_attn_norm + lat.router + lat.route_sync
        put(f"route_known:{layer}", t, "route_known", layer)

        # Stage this layer's experts (misses only transfer; hits are cached).
        # Single transfer stream: completions are serial; batch_submit only
        # removes host-side submission gaps, not device serialization.
        misses = [r for r in range(topk)
                  if r >= int(topk * hit_rate)]
        last_done = t
        for i, rank in enumerate(misses):
            pre = f"stage_done:{layer}:{rank}"
            if flags.hash_prefetch and is_hash and pre in now:
                last_done = max(last_done, now[pre])
                continue  # already prefetched; R2/R3 satisfied early
            gap = 0.0 if flags.batch_submit else flags.host_serial_gap
            submit = t + (0.0 if flags.batch_submit else gap * i)
            last_done = max(last_done, submit) + lat.h2d_per_expert
            put(pre, last_done, "stage", layer, rank)
        # Compute each expert after its staging (R3); serial on one stream.
        comp = max(last_done, t)
        for rank in range(topk):
            if rank in misses:
                comp = max(comp, at(f"stage_done:{layer}:{rank}"))
            comp = comp + lat.decode_per_expert + lat.gemm_per_expert
            put(f"consume_done:{layer}:{rank}", comp, "consume", layer, rank)
        # Shared expert: needs only this layer's hidden state (known at
        # route_known time), independent of routed staging (R4 joins later).
        if flags.shared_overlap:
            shared_done = t + lat.shared_expert  # overlaps staging/compute
        else:
            shared_done = comp + lat.shared_expert  # current serial order
        put(f"shared_done:{layer}", shared_done, "shared", layer)
        # R4 combine joins routed + shared.
        combine = max(comp, shared_done) + lat.combine
        put(f"combine_done:{layer}", combine, "combine", layer)
        if layer == 21:  # dual-GPU handoff before layer 22 (P2.3 split)
            combine = combine + lat.handoff_gpu
            put(f"handoff_done:{layer}", combine, "handoff", layer)
            bound.append("handoff: A->host->B after layer 21 (no P2P assumed)")
        if not flags.no_layer_sync:
            combine = combine + lat.layer_sync
            syncs += 1
        combine_prev = combine
        put(f"layer_done:{layer}", combine, "layer_done", layer)

    if flags.no_layer_sync:
        combine_prev += lat.layer_sync  # single token-end sync (ASSUMED)
        bound.append("no_layer_sync: single token-end sync is ASSUMED "
                     "(needs event-handoff proof, see DUAL_GPU_PIPELINE.md)")
    return SimResult(critical_ms=combine_prev, schedule=schedule,
                     edges=edges, bound_by=bound)


def handoff(result: SimResult, **extra) -> dict:
    """Phase G machine-readable causal-lead summary for Flash's track."""
    return {
        "future_layer_official_lead": 0,
        "lead_basis": ("score-layer routes require post-residual hidden "
                       "states (R5); hash IDs excepted below"),
        "hash_layer_early_ids": sorted(HASH_LAYERS),
        "hash_basis": "tid2eid[input_ids]; weights still need hidden (R1)",
        "token_level_lead": 0,
        "token_basis": "R6 sampling data dependency; nothing stageable for t+1",
        "cross_gpu_official_overlap": False,
        "cross_gpu_basis": "routes for B-side layers unknown while A computes",
        "schedule": result.schedule,
        "critical_ms_simulated": round(result.critical_ms, 3),
        "latency_source": "ASSUMED (see Latency.source)",
        "bound_by": result.bound_by,
        **extra,
    }


if __name__ == "__main__":  # pragma: no cover
    base = simulate()
    print(json.dumps(handoff(base), indent=2))
