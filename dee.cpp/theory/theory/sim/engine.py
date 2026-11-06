"""Discrete-event digital twin of the dee hierarchy (NVMe -> RAM -> VRAM -> GPU).

EVENT-DRIVEN, not a reimplementation of the closed forms in ``theory.cache`` /
``theory.roofline``.  Entities and their real behaviour are each transcribed
from a named artifact line (declarations in ``theory/sim/constants.py``):

  request arrivals -> token decode steps -> layer router decisions ->
  expert demands -> VRAM slots (priority-LRU / recency) -> host tier (bounded
  LRU, batch-protected) -> SSD service queue (lanes + per-request service) ->
  H2D transfers (async copy engine) -> compute stages.

Pipeline mechanics reproduced (verified to the counter against the anchor run
by ``theory/sim/replay.py``):

  * per-layer-call deduplicated staging: each distinct expert is staged once
    per layer call (``RouterCall.experts`` is the dedup set);
  * staging order is ASCENDING EXPERT ID with ``priority = K - i``
    (engine.cpp:565-567 builds ``active_experts`` in id order; :592 and :4626
    pass ``K - i``).  The RankPriority eviction score ``last_used +
    priority*2^20`` (vram_cache.h:244-247) acts on exactly this priority;
  * fill coalescing: an in-batch duplicate is one fill (host_pack_cache.cpp
    :245-256) and an in-flight request coalesces (async_prefetcher.cpp:455-473);
  * EAGER WHOLE-CALL HOST CONSULT: the host pack is consulted and, on a miss,
    FILLED for every requested record even when the record is already VRAM
    resident (engine.cpp:3120-3253 ``prepare_fp4_experts`` runs before any
    staging; :3267 ``get_staging_fp4`` re-consults the pack on both paths);
  * blocking fill sub-batches of ``queue_depth`` requests served by
    ``fill_lanes`` concurrent pread workers (engine.cpp:3142 loops ``get_batch``
    in sub-batches and blocks on each one);
  * async H2D on one copy engine with the GEMM stage pipelined behind it
    (async_prefetcher submit-then-``wait_on_stream``, engine.cpp:1842-1849);
  * demand-gated fills along the layer dependency chain: a layer's demand is
    unknown until the previous layer's router has run, so no fill can be
    issued before its own layer-call arrives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Key = Tuple[int, int]


# ==========================================================================
# Request-stream entities
# ==========================================================================
@dataclass(frozen=True)
class RouterCall:
    """One layer-call: the deduplicated expert demand the engine stages."""
    token: int
    layer: int
    device: str
    experts: Tuple[int, ...]      # DISTINCT ids, ASCENDING (engine staging order)
    phase: str = "decode"
    token_rows: int = 1

    @property
    def key_set(self) -> Tuple[Key, ...]:
        return tuple((self.layer, e) for e in self.experts)


@dataclass
class Stream:
    """A request stream = layer-calls in layer-dependency order."""
    calls: List[RouterCall]
    n_layers: int
    experts_per_layer: int
    topk: int
    label: str = "stream"

    @property
    def n_requests(self) -> int:
        return sum(len(c.experts) for c in self.calls)

    def keys(self) -> List[Key]:
        out: List[Key] = []
        for c in self.calls:
            out.extend(c.key_set)
        return out

    def token_ids(self) -> List[int]:
        out: List[int] = []
        for c in self.calls:
            if not out or out[-1] != c.token:
                out.append(c.token)
        return out


def layer_device_map(n_layers: int, split: int) -> Dict[int, str]:
    """Static model-parallel layer -> GPU split (MEASURED: 22 / 21 on 2xT4)."""
    return {l: ("cuda:0" if l < split else "cuda:1") for l in range(n_layers)}


def stream_from_sealed(sealed, device_map: Dict[int, str]) -> Stream:
    """Build a Stream from ``theory.sources.load_sealed_stream()``."""
    calls: List[RouterCall] = []
    for (tok, layer) in sealed.keys:
        rows = sealed.raw[(tok, layer)]
        experts = tuple(sorted({e for row in rows for e in row}))
        calls.append(RouterCall(
            tok, layer, device_map[layer], experts,
            phase=sealed.phases.get((tok, layer), "decode") or
                  ("prefill" if tok == 0 else "decode"),
            token_rows=len(rows)))
    return Stream(calls, sealed.n_layers, sealed.experts_per_layer,
                  sealed.topk, "sealed_dsv4")


# ==========================================================================
# Per-record pread service time: inverse-CDF over the MEASURED quantiles
# ==========================================================================
class ReadService:
    """Piecewise-linear quantile function through the measured pread quantiles.

    Anchors (result.json ``expert_store.cuda0``): mean 103.5997, p50 104.49955,
    p95 163.912439, max 168.568131 ms.  The implied floor (DERIVED) is the
    fastest read consistent with those four numbers.
    """

    def __init__(self, t_floor: float, p50: float, p95: float, t_max: float,
                 seed: int = 20260916, deterministic: bool = False,
                 mean_decode: float = 83.156, mean_prefill: float = 124.417):
        self.q = np.array([0.0, 0.50, 0.95, 1.0])
        self.t = np.array([t_floor, p50, p95, t_max], dtype=float)
        self.deterministic = deterministic
        self.mean_decode = mean_decode
        self.mean_prefill = mean_prefill
        self.rng = np.random.default_rng(seed)
        # analytic mean of a piecewise-LINEAR CDF read the other way is not the
        # trapezoid mean; the quantile function here is piecewise linear in u,
        # so its mean IS the trapezoid integral over u in [0,1].
        self.mean_model = float(sum(
            0.5 * (self.t[i] + self.t[i + 1]) * (self.q[i + 1] - self.q[i])
            for i in range(len(self.q) - 1)))

    def draw(self, phase: str = "decode") -> float:
        if self.deterministic:
            return self.mean_prefill if phase == "prefill" else self.mean_decode
        return float(np.interp(self.rng.random(), self.q, self.t))

    def draws(self, n: int, phase: str = "decode") -> np.ndarray:
        if n <= 0:
            return np.zeros(0)
        if self.deterministic:
            m = self.mean_prefill if phase == "prefill" else self.mean_decode
            return np.full(n, m)
        return np.interp(self.rng.random(n), self.q, self.t)


# ==========================================================================
# Cache tiers
# ==========================================================================
class HostPack:
    """Bounded host pack: LRU with in-batch protection (host_pack_cache.cpp).

    ``get_batch`` semantics transcribed from ``src/host_pack_cache.cpp:224-314``:
    scan the request list in order — an in-batch duplicate counts as a hit and
    does not move LRU; a present key counts as a hit and moves to MRU; a unique
    miss is counted and reserved.  Victims come from the LRU back, skipping
    every key of the incoming batch.  Unique misses are admitted at the MRU
    front in request order BEFORE the fills run.
    """

    def __init__(self, capacity: int):
        self.cap = int(capacity)
        self.lru: Dict[Key, None] = {}       # dict order == LRU (old) -> MRU (new)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self.lru)

    def get_batch(self, keys: Sequence[Key]) -> List[bool]:
        out = [False] * len(keys)
        seen = set()
        unique_misses: List[Key] = []
        for i, k in enumerate(keys):
            if k in seen:
                self.hits += 1            # in-batch duplicate -> counted hit
                out[i] = True
                continue
            seen.add(k)
            if k in self.lru:
                self.lru.pop(k)
                self.lru[k] = None
                self.hits += 1
                out[i] = True
                continue
            self.misses += 1
            unique_misses.append(k)
        batch_keys = set(keys)
        need = len(unique_misses)
        while len(self.lru) + need > self.cap and self.lru:
            victim = None
            for kk in self.lru:
                if kk not in batch_keys:
                    victim = kk
                    break
            if victim is None:
                break                      # budget rejection path
            self.lru.pop(victim)
            self.evictions += 1
        for k in unique_misses:
            self.lru.pop(k, None)
            self.lru[k] = None
        return out


class VramCache:
    """Device expert tier: bounded packed residency, pin-protected eviction.

    ``policy='priority_lru'`` reproduces the anchor run's RankPriority score
    ``last_used + priority * 2^20`` (vram_cache.h:244-247) — min score evicted
    first, pinned blocks skipped.  ``policy='recency'`` is the Phase-2
    recommended repair (pure ``last_used``).
    """

    def __init__(self, capacity: int, policy: str = "priority_lru",
                 priority_weight: int = 1 << 20):
        self.cap = int(capacity)
        self.policy = policy
        self.pw = priority_weight
        self.blocks: Dict[Key, List[int]] = {}   # key -> [last_used, priority, pins]
        self.tick = 0
        self.hits = 0
        self.cold = 0
        self.evictions = 0
        self.pinned_skipped = 0

    def __len__(self) -> int:
        return len(self.blocks)

    def _score(self, b: List[int]) -> int:
        return b[0] if self.policy == "recency" else b[0] + b[1] * self.pw

    def _evict_one(self) -> None:
        victim = None
        vs = None
        for k, b in self.blocks.items():
            if b[2] != 0:
                self.pinned_skipped += 1
                continue
            s = self._score(b)
            if victim is None or s < vs:
                victim, vs = k, s
        if victim is None:
            raise RuntimeError("evict_until_free: no evictable victim (all pinned)")
        del self.blocks[victim]
        self.evictions += 1

    def ensure(self, key: Key, priority: int) -> str:
        self.tick += 1
        b = self.blocks.get(key)
        if b is not None:
            b[0] = self.tick
            b[1] = priority
            b[2] += 1
            self.hits += 1
            return "hit"
        while len(self.blocks) >= self.cap:
            self._evict_one()
        self.blocks[key] = [self.tick, priority, 1]
        self.cold += 1
        return "cold"

    def unpin(self, key: Key) -> None:
        b = self.blocks.get(key)
        if b is not None and b[2] > 0:
            b[2] -= 1


# ==========================================================================
# Discrete-event resources
# ==========================================================================
class LaneFarm:
    """``n`` concurrent pread workers draining one FIFO queue (list scheduling).

    A job submitted at ``t`` starts on the earliest-free lane
    ``max(t, lane_free)`` — the makespan a pool of worker threads over one
    shared request queue realises.  ``HostPackCache`` runs exactly one such
    pool per ``get_batch`` call and blocks on it.
    """

    def __init__(self, n_lanes: int):
        self.n = max(1, int(n_lanes))
        self.free = [0.0] * self.n
        self.busy_ms = 0.0
        self.jobs = 0

    def submit(self, t: float, service_ms: float) -> float:
        i = min(range(self.n), key=lambda j: (self.free[j], j))
        end = max(t, self.free[i]) + service_ms
        self.free[i] = end
        self.busy_ms += service_ms
        self.jobs += 1
        return end

    def submit_batch(self, t: float, services: Sequence[float]) -> float:
        """Dispatch one blocking fill sub-batch; return its makespan."""
        end = t
        for s in services:
            e = self.submit(t, s)
            if e > end:
                end = e
        return end

    @property
    def utilization(self) -> float:
        end = max(self.free) if self.free else 0.0
        return self.busy_ms / (self.n * end) if end > 0 else 0.0


class SerialPipe:
    """One FIFO single-server resource (H2D copy engine, GPU compute engine)."""

    def __init__(self, name: str):
        self.name = name
        self.free = 0.0
        self.busy_ms = 0.0
        self.jobs = 0

    def submit(self, t: float, service_ms: float) -> float:
        end = max(t, self.free) + service_ms
        self.free = end
        self.busy_ms += service_ms
        self.jobs += 1
        return end


# ==========================================================================
# Configuration + results
# ==========================================================================
@dataclass
class HierarchyConfig:
    host_slots: int                       # per GPU
    vram_slots: int                       # per GPU
    record_bytes: int
    fill_lanes: int = 3
    queue_depth: int = 6
    vram_policy: str = "priority_lru"
    b_ssd: float = 0.33 * (1 << 30)       # aggregate bank ceiling B/s
    b_h2d: float = 11.44e9                # aggregate H2D B/s (MEASURED)
    touch_ms: float = 0.39369             # per expert-record GEMM touch
    dense_call_ms: float = 13.272         # non-routed per layer-call (decode)
    dense_prefill_call_ms: float = 486.835  # non-routed per layer-call (prefill)
    orch_ms: float = 0.0                  # CALIBRATED orchestration slack/call
    read_seed: int = 20260916
    read_t_floor: float = 86.9
    read_p50: float = 104.49955
    read_p95: float = 163.912439
    read_t_max: float = 168.568131
    read_ms_decode: float = 83.156        # DERIVED phase means (deterministic)
    read_ms_prefill: float = 124.417
    host_shared: bool = False
    eager_host_consult: bool = True
    deterministic_read: bool = False   # phase-mean service (REPLAY gate mode)

    def read_service(self) -> ReadService:
        return ReadService(self.read_t_floor, self.read_p50, self.read_p95,
                           self.read_t_max, self.read_seed,
                           self.deterministic_read,
                           self.read_ms_decode, self.read_ms_prefill)


@dataclass
class TokenStats:
    token: int
    phase: str
    requests: int = 0
    host_hits: int = 0
    host_misses: int = 0
    resident_hits: int = 0
    cold_loads: int = 0
    h2d_copies: int = 0
    h2d_bytes: int = 0
    read_jobs: int = 0
    read_bytes: int = 0
    vram_evictions: int = 0
    host_evictions: int = 0
    wall_ms: float = 0.0
    fill_ms: float = 0.0
    compute_ms: float = 0.0
    dense_ms: float = 0.0

    @property
    def storage_requests(self) -> int:
        """Host-pack misses == the engine's ``storage_requests`` counter
        (verified token-for-token on the anchor run)."""
        return self.host_misses


@dataclass
class SimResult:
    tokens: List[TokenStats]
    host_hits: int
    host_misses: int
    resident_hits: int
    cold_loads: int
    h2d_copies: int
    read_jobs: int
    vram_evictions: int
    host_evictions: int
    wall_ms: float
    lane_util: float
    read_mean_ms: float
    per_call_fill_ms: List[float] = field(default_factory=list)
    label: str = ""
    host_hits_by_dev: Dict[str, int] = field(default_factory=dict)
    host_misses_by_dev: Dict[str, int] = field(default_factory=dict)
    requests_by_dev: Dict[str, int] = field(default_factory=dict)

    @property
    def n_requests(self) -> int:
        return self.host_hits + self.host_misses

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def rates(self) -> Dict[str, float]:
        req = self.n_requests
        vram_missed = self.cold_loads
        host_hits_on_vram_miss = vram_missed - self.read_jobs
        return {
            "requests": req,
            "H_v": self.resident_hits / req if req else 0.0,
            "H_h_conditional": (host_hits_on_vram_miss / vram_missed
                                if vram_missed else 0.0),
            "host_share_eager": self.host_hits / req if req else 0.0,
            "cold_per_request": self.read_jobs / req if req else 0.0,
            "cold_records_per_token": self.read_jobs / max(1, self.n_tokens),
            "fills_per_token": self.cold_loads / max(1, self.n_tokens),
            "token_ms": self.wall_ms / max(1, self.n_tokens),
        }


# ==========================================================================
# The hierarchy simulation
# ==========================================================================
class HierarchySim:
    """Event-driven digital twin of one (cell, model) hierarchy instance.

    Event flow per layer-call, released along the layer dependency chain::

        CALL_ARRIVE
          -> host consult + blocking fill sub-batches over the SSD lane farm
          -> VRAM ensure (evict/pin) + async H2D submission on the copy engine
          -> GEMM touches on the GPU compute engine
          -> dense stage -> CALL_DONE (releases the next layer-call)

    The host consult is *eager and whole-call*: the live engine fills the host
    pack for every requested record regardless of VRAM residency, so a
    VRAM-resident record still costs a host lookup and, on a pack miss, a full
    pread.  That is the single largest sim-vs-closed-form divergence and it is
    reported with magnitude, never smoothed away.
    """

    def __init__(self, cfg: HierarchyConfig):
        self.cfg = cfg
        self.read = cfg.read_service()
        # each GPU runs its own host pack + its own fill lane pool
        # (result.json expert_store.cuda0/.cuda1 each report 3 lanes)
        self.lanes: Dict[str, LaneFarm] = {}
        self.h2d: Dict[str, SerialPipe] = {}
        self.host: Dict[str, HostPack] = {}
        self.vram: Dict[str, VramCache] = {}
        self.compute: Dict[str, SerialPipe] = {}
        self.host_hits_by_dev: Dict[str, int] = {}
        self.host_misses_by_dev: Dict[str, int] = {}
        self.requests_by_dev: Dict[str, int] = {}

    def _lanes_for(self, device: str) -> LaneFarm:
        key = "shared" if self.cfg.host_shared else device
        if key not in self.lanes:
            self.lanes[key] = LaneFarm(self.cfg.fill_lanes)
        return self.lanes[key]

    def _h2d_for(self, device: str) -> SerialPipe:
        if device not in self.h2d:
            self.h2d[device] = SerialPipe("h2d:" + device)
        return self.h2d[device]

    def _host_for(self, device: str) -> HostPack:
        key = "shared" if self.cfg.host_shared else device
        if key not in self.host:
            self.host[key] = HostPack(self.cfg.host_slots)
        return self.host[key]

    def _vram_for(self, device: str) -> VramCache:
        if device not in self.vram:
            self.vram[device] = VramCache(self.cfg.vram_slots,
                                          self.cfg.vram_policy)
        return self.vram[device]

    def _compute_for(self, device: str) -> SerialPipe:
        if device not in self.compute:
            self.compute[device] = SerialPipe("compute:" + device)
        return self.compute[device]

    def run(self, stream: Stream) -> SimResult:
        """Discrete-event run over the stream.

        Per layer-call the event graph is::

            t_arrive  (released by the previous layer-call's combine)
              -> fill makespan   (lane farm; blocking sub-batches of
                                  queue_depth requests, demand-gated)
              -> VRAM ensure     (evict + pin; a miss schedules an H2D copy)
              -> H2D burst       (per-GPU copy engine, 1 record per copy)
              -> compute         (per-GPU engine: dense stage + K GEMM touches)
              -> combine -> t_done  (releases the next layer-call)

        Everything between ``t_arrive`` and ``t_done`` is on the critical path
        of the layer dependency chain, so the token wall is the sum of its
        calls' spans (plus whatever the copy engine still owes after the last
        call).  No fill can be issued before its own layer-call arrives: the
        next layer's demand is produced by this layer's router, so the fills
        are demand-gated along the chain — the exact property that makes the
        closed form's ``t_storage = M_h * P_rec / B_SSD`` a bound rather than
        a prediction.
        """
        cfg = self.cfg
        rec = float(cfg.record_bytes)
        h2d_svc = rec / max(1.0, cfg.b_h2d) * 1e3
        toks: Dict[int, TokenStats] = {}
        order: List[int] = []
        token_start: Dict[int, float] = {}
        token_end: Dict[int, float] = {}
        per_call_fill: List[float] = []
        ev0: Dict[str, int] = {}

        chain_t = 0.0
        cur_token: Optional[int] = None
        for call in stream.calls:
            if call.token != cur_token:
                if cur_token is not None:
                    token_end[cur_token] = chain_t
                cur_token = call.token
                token_start[call.token] = chain_t
                ts = TokenStats(call.token, call.phase)
                toks[call.token] = ts
                order.append(call.token)
            ts = toks[call.token]
            hp = self._host_for(call.device)
            vc = self._vram_for(call.device)
            cmp_pipe = self._compute_for(call.device)
            h2d_pipe = self._h2d_for(call.device)
            lanes = self._lanes_for(call.device)
            keys = list(call.key_set)
            K = len(keys)

            # ---- (1) eager whole-call host consult (prepare_fp4_experts) ----
            outcomes = hp.get_batch(keys) if cfg.eager_host_consult else [True] * K
            ts.requests += K
            ts.host_hits += sum(1 for o in outcomes if o)
            ts.host_misses += sum(1 for o in outcomes if not o)
            self.host_hits_by_dev[call.device] = \
                self.host_hits_by_dev.get(call.device, 0) + \
                sum(1 for o in outcomes if o)
            self.host_misses_by_dev[call.device] = \
                self.host_misses_by_dev.get(call.device, 0) + \
                sum(1 for o in outcomes if not o)
            self.requests_by_dev[call.device] = \
                self.requests_by_dev.get(call.device, 0) + K

            # ---- (2) blocking fill sub-batches ------------------------------
            # engine.cpp:3142 loops ``get_batch`` over the chunk's REQUESTS in
            # sub-batches of ``queue_depth`` and blocks on each one
            # (HostSpanGuard FillWait); only the misses inside a sub-batch are
            # served by the lane pool.
            t = chain_t
            fill_ms = 0.0
            for s0 in range(0, K, cfg.queue_depth):
                sub = [k for k, o in zip(keys[s0:s0 + cfg.queue_depth],
                                         outcomes[s0:s0 + cfg.queue_depth])
                       if not o]
                if not sub:
                    continue
                t0 = t
                t = lanes.submit_batch(t0, self.read.draws(len(sub),
                                                           call.phase))
                fill_ms += t - t0
                ts.read_jobs += len(sub)
                ts.read_bytes += len(sub) * int(rec)
            per_call_fill.append(fill_ms)
            ts.fill_ms += fill_ms
            t_fill_end = t

            # ---- (3) VRAM stage: ensure + pin, ascending id, prio = K - i ---
            cold_keys: List[Key] = []
            for i, k in enumerate(keys):
                if vc.ensure(k, K - i) == "hit":
                    ts.resident_hits += 1
                else:
                    ts.cold_loads += 1
                    ts.h2d_copies += 1
                    ts.h2d_bytes += int(rec)
                    cold_keys.append(k)
            for k in keys:
                vc.unpin(k)

            # ---- (4) async H2D on the copy engine ---------------------------
            # one device fill == one record H2D (anchor counters:
            # cold_loads == h2d_copies == evictions, token for token).  The copy
            # engine and the compute engine are DIFFERENT resources running
            # concurrently from ``t_fill_end``: a GEMM waits only on its own
            # record's H2D (engine.cpp:1842-1849 wait_on_stream per record), so
            # the two engines OVERLAP and the call advances at their max.
            t_h2d = t_fill_end
            for _ in cold_keys:
                t_h2d = h2d_pipe.submit(t_h2d, h2d_svc)

            # ---- (5) compute: dense stage + K GEMM touches ------------------
            dense = (cfg.dense_prefill_call_ms if call.phase == "prefill"
                     else cfg.dense_call_ms)
            ts.dense_ms += dense
            ts.compute_ms += K * cfg.touch_ms
            t_cmp = cmp_pipe.submit(t_fill_end, dense + K * cfg.touch_ms
                                    + cfg.orch_ms)
            chain_t = t_cmp if t_cmp > t_h2d else t_h2d

            ts.vram_evictions += vc.evictions - ev0.get(call.device, 0)
            ev0[call.device] = vc.evictions
            hkey = "shared" if cfg.host_shared else call.device
            ts.host_evictions += hp.evictions - ev0.get("h:" + hkey, 0)
            ev0["h:" + hkey] = hp.evictions

        token_end[cur_token] = chain_t
        for tok in order:
            toks[tok].wall_ms = token_end[tok] - token_start[tok]

        n_read = sum(t.read_jobs for t in toks.values())
        return SimResult(
            tokens=[toks[k] for k in order],
            host_hits=sum(t.host_hits for t in toks.values()),
            host_misses=sum(t.host_misses for t in toks.values()),
            resident_hits=sum(t.resident_hits for t in toks.values()),
            cold_loads=sum(t.cold_loads for t in toks.values()),
            h2d_copies=sum(t.h2d_copies for t in toks.values()),
            read_jobs=n_read,
            vram_evictions=sum(v.evictions for v in self.vram.values()),
            host_evictions=sum(h.evictions for h in self.host.values()),
            wall_ms=chain_t,
            lane_util=max((f.utilization for f in self.lanes.values()),
                          default=0.0),
            read_mean_ms=self.read.mean_model,
            per_call_fill_ms=per_call_fill,
            label=stream.label,
            host_hits_by_dev=dict(self.host_hits_by_dev),
            host_misses_by_dev=dict(self.host_misses_by_dev),
            requests_by_dev=dict(self.requests_by_dev),
        )
