"""Offline cache simulation engine.

Simulates dee's expert residency cache over a canonical demand sequence.
Costs are byte-based and (optionally) latency-based using measured
trace-run storage statistics (expert_store read p50/p95 from the sealed
v65 run) -- clearly labeled as trace-derived, not re-measured here.

Two tiers are supported:
  tier 1 (VRAM): capacity_bytes, policy-driven
  tier 2 (host RAM): tier2_bytes, LRU, populated automatically on
      tier-1 miss from SSD (mirrors dee's host_pack cache behavior).

Phase E prefetch is provided by PrefetchEngine (prefetch.py), which
subclasses this engine.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace


def _reindex(demands):
    """Return demands with seq reindexed 0..n-1 (slice-local positions).

    Works with frozen dataclasses (trace.Demand) and duck-typed
    stand-ins used in tests.
    """
    out = []
    for i, d in enumerate(demands):
        try:
            nd = replace(d, seq=i)
        except TypeError:
            nd = copy.copy(d)
            nd.seq = i
        out.append(nd)
    return out

# Measured storage-read latency statistics from the sealed v65 run
# (dee.cpp/tmp/v65-terminal-fetch-20260903T0202Z, expert_store.cuda0):
# these are TRACE-DERIVED constants, not re-measured in this project.
MEASURED_READ_MS = {
    "p50": 256.986,
    "p95": 266.417,
    "mean": 235.208,
    "bandwidth_mib_s": 54.21,
    # source_read_overlap_percent ~82.3%: share of fill time overlapped
    # with compute in the sealed run (used ONLY for a labeled
    # theoretical exposed-stall estimate).
    "fill_overlap_percent": 82.32,
}


@dataclass
class Metrics:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    compulsory_misses: int = 0
    capacity_misses: int = 0
    evictions: int = 0
    insertions: int = 0
    miss_bytes: int = 0
    ssd_bytes: int = 0
    tier2_h2d_bytes: int = 0
    useful_residency_ticks: int = 0
    wasted_residency_ticks: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TwoTierStats:
    t1_hits: int = 0
    t2_hits: int = 0
    t1_miss_from_t2: int = 0
    t1_miss_from_ssd: int = 0
    t2_evictions: int = 0


class Engine:
    """Tier-1 policy cache with optional tier-2 LRU backing.

    The tier-1 policy must expose: resident (set), resident_bytes,
    has(), admit_new(key, nbytes, tick), on_hit(key, tick),
    note_tick(tick) (optional), last_evictions (list, reset per admit).
    """

    def __init__(self, policy, record_bytes: int = 13_369_344,
                 tier2_bytes: int = 0):
        self.policy = policy
        self.record_bytes = record_bytes
        self.tier2_bytes = tier2_bytes
        self.t2_resident: set[int] = set()
        self.t2_last: dict[int, int] = {}  # key -> last-touch tick (LRU)
        self.m = Metrics()
        self.t = TwoTierStats()
        self._known: set[int] = set()      # keys ever demanded
        self._insert_tick: dict[int, int] = {}   # key -> tier-1 insert tick
        self._last_access: dict[int, int] = {}   # key -> last demand tick
        self._ever_t1: set[int] = set()

    # ------------------------------------------------------------------
    def run(self, demands) -> Metrics:
        for d in demands:
            self.access_demand(d.scope, d.seq, d.nbytes)
        self._close_out(demands[-1].seq if demands else 0)
        return self.m

    def _close_out(self, last_tick: int) -> None:
        """Still-resident keys: residency up to their last use is useful,
        any tail beyond it is wasted (never demanded again)."""
        m = self.m
        for key, it in self._insert_tick.items():
            la = self._last_access.get(key, it)
            if la >= it:
                m.useful_residency_ticks += la - it
                m.wasted_residency_ticks += max(0, last_tick - la)
            else:
                m.wasted_residency_ticks += last_tick - it
        self._insert_tick.clear()

    # ------------------------------------------------------------------
    def access_demand(self, key: int, tick: int, nbytes: int) -> None:
        m = self.m
        pol = self.policy
        m.accesses += 1
        if hasattr(pol, "note_tick"):
            pol.note_tick(tick)
        self._last_access[key] = tick

        if key in pol.resident:
            m.hits += 1
            self.t.t1_hits += 1
            it = self._insert_tick.get(key)
            if it is not None:
                m.useful_residency_ticks += tick - it
            # residency is continuous: re-anchor at this touch so the
            # post-hit interval is counted at the next hit/eviction.
            self._insert_tick[key] = tick
            pol.on_hit(key, tick)
            return

        # ---- miss ----
        m.misses += 1
        m.miss_bytes += nbytes
        if key not in self._known:
            m.compulsory_misses += 1
        else:
            m.capacity_misses += 1
        self._known.add(key)

        if self.tier2_bytes > 0:
            if key in self.t2_resident:
                self.t.t2_hits += 1
                self.t.t1_miss_from_t2 += 1
                m.tier2_h2d_bytes += nbytes
            else:
                self.t.t1_miss_from_ssd += 1
                m.ssd_bytes += nbytes
                self._t2_insert(key, tick, nbytes)
        else:
            m.ssd_bytes += nbytes

        self._admit(key, tick, nbytes)

    def _admit(self, key: int, tick: int, nbytes: int) -> None:
        """Insert into tier 1 via the policy; bookkeep evictions."""
        m = self.m
        pol = self.policy
        pol.last_evictions = []
        pol.admit_new(key, nbytes, tick)
        victims = pol.last_evictions
        m.evictions += len(victims)
        for v in victims:
            it = self._insert_tick.pop(v, None)
            la = self._last_access.get(v)
            if it is not None:
                if la is not None and la >= it:
                    m.useful_residency_ticks += la - it
                    m.wasted_residency_ticks += tick - la
                else:
                    m.wasted_residency_ticks += tick - it
            self._on_evict_hook(v, tick)
        if key in pol.resident:
            m.insertions += 1
            self._insert_tick[key] = tick
            self._ever_t1.add(key)
        # tier-2 mirror on tier-1 eviction (dee host_pack behavior)
        for v in victims:
            if self.tier2_bytes > 0 and v not in self.t2_resident:
                self._t2_insert(v, tick, self._nbytes_of(v))

    def _nbytes_of(self, key: int) -> int:
        bmap = getattr(self.policy, "_bytes", None)
        if bmap and key in bmap:
            return bmap[key]
        return self.record_bytes

    def _on_evict_hook(self, key: int, tick: int) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------
    def _t2_insert(self, key: int, tick: int, nbytes: int) -> None:
        while self.t2_resident and (
            (len(self.t2_resident) + 1) * nbytes > self.tier2_bytes
        ):
            victim = min(self.t2_last, key=self.t2_last.get)  # type: ignore[arg-type]
            self.t2_resident.discard(victim)
            del self.t2_last[victim]
            self.t.t2_evictions += 1
        self.t2_resident.add(key)
        self.t2_last[key] = tick


class LayeredEngine:
    """One independent cache per transformer layer (dee semantics).

    dee's expert cache capacity is PER LAYER (e.g. cap-32 = 32
    (layer,expert) records resident per GPU per layer), so the correct
    model is 43 independent tier-1 caches over per-layer demand
    subsequences. Tier-2 (host pack) is global in dee; modeling it as
    an equal per-layer slice is an approximation, noted in outputs.
    """

    def __init__(self, policy_factory, n_layers: int = 43,
                 record_bytes: int = 13_369_344,
                 tier2_bytes_per_layer: int = 0):
        self.n_layers = n_layers
        self.record_bytes = record_bytes
        self.tier2_bytes_per_layer = tier2_bytes_per_layer
        self.policy_factory = policy_factory
        self.engines: dict[int, Engine] = {}

    def run(self, demands) -> Metrics:
        per: dict[int, list] = {}
        for d in demands:
            per.setdefault(d.layer, []).append(d)
        for layer, ds in per.items():
            # Reindex seq slice-locally: policies (esp. Belady's
            # next-use chain) index positions within THIS layer's
            # demand sequence, and the global trace seq would
            # overflow/hit the wrong positions.
            local = _reindex(ds)
            eng = Engine(self.policy_factory(layer),
                         record_bytes=self.record_bytes,
                         tier2_bytes=self.tier2_bytes_per_layer)
            self.engines[layer] = eng
            eng.run(local)
        return self.aggregate(list(per.keys()))

    def aggregate(self, layers: list[int]) -> Metrics:
        m = Metrics()
        t_hits = t2_hits = miss_t2 = miss_ssd = t2_ev = 0
        for layer in layers:
            e = self.engines[layer]
            em, et = e.m, e.t
            m.accesses += em.accesses
            m.hits += em.hits
            m.misses += em.misses
            m.compulsory_misses += em.compulsory_misses
            m.capacity_misses += em.capacity_misses
            m.evictions += em.evictions
            m.insertions += em.insertions
            m.miss_bytes += em.miss_bytes
            m.ssd_bytes += em.ssd_bytes
            m.tier2_h2d_bytes += em.tier2_h2d_bytes
            m.useful_residency_ticks += em.useful_residency_ticks
            m.wasted_residency_ticks += em.wasted_residency_ticks
            t_hits += et.t1_hits
            t2_hits += et.t2_hits
            miss_t2 += et.t1_miss_from_t2
            miss_ssd += et.t1_miss_from_ssd
            t2_ev += et.t2_evictions
        self.t = TwoTierStats(t1_hits=t_hits, t2_hits=t2_hits,
                              t1_miss_from_t2=miss_t2,
                              t1_miss_from_ssd=miss_ssd,
                              t2_evictions=t2_ev)
        return m
