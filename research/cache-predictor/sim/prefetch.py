"""Phase E: prefetch / proactive-replacement simulation (offline, causal).

Model per layer (dee semantics):
  - At the START of each new step's bundle at layer L, the causal
    predictor (cross-token recurrence) emits a ranked candidate list.
  - Candidates not resident are prefetched into FREE slots ("safe"
    mode) or additionally by replacing the lowest-value resident
    ("replace" mode, gated by hysteresis).
  - Hysteresis / minimum dwell: a resident expert cannot be evicted
    before `min_dwell` demands have passed since its insertion, and
    only `max_prefetch_per_bundle` prefetches are issued per bundle.

Prefetches are inserted into the SAME policy cache as demand (as
"hits" of prediction, marked via _prefetched set) so pollution is
measured exactly: a prefetched expert that is never demanded before
eviction contributes wasted bytes and wasted residency.

No future routing information is used: candidate lists for step t
use only step < t bundles. The official router remains authoritative
(demand sequence is unchanged; only residency timing changes).

All stall-time results are THEORETICAL: they use the trace-derived
read-latency statistics (MEASURED_READ_MS) and a bandwidth sweep, NOT
a runtime benchmark.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policies import OnlinePolicy  # noqa: E402
from engine import Engine, Metrics, MEASURED_READ_MS  # noqa: E402


class PrefetchPolicy(OnlinePolicy):
    """LRU base with prediction-aware PREFETCH victim selection.

    Demand-path evictions are PURE LRU (last-touch recency) so that
    mode="off" reproduces the tournament LRU baseline exactly and the
    prefetch deltas isolate the prefetch effect. Only the prefetch
    replacement path (_hysteresis_victim in PrefetchEngine) uses
    prediction-aware scoring (protect predicted keys, skip
    young/min-dwell residents). Prefetched experts count as resident;
    whether they were useful is tracked by the engine via the
    `_prefetched` marker.
    """

    name = "prefetch_lru"

    def __init__(self, capacity_bytes: int, default_nbytes: int = 1,
                 ranked_fn=None):
        super().__init__(capacity_bytes)
        self._default_nbytes = default_nbytes
        self.ranked_fn = ranked_fn
        self.last_touch: dict[int, int] = {}  # key -> last touch tick (LRU)
        self.order: dict[int, int] = {}       # key -> insert tick (dwell)
        self.freq: dict[int, int] = {}
        self._tick = 0
        self._p: set[int] = set()
        self._prefetched: set[int] = set()

    def note_tick(self, tick: int) -> None:
        self._tick = tick
        if self.ranked_fn is not None:
            self._p = {k for k, _ in self.ranked_fn(tick)}
        else:
            self._p = set()

    def on_hit(self, key, tick):
        self.freq[key] = self.freq.get(key, 0) + 1
        self.last_touch[key] = tick

    def _insert(self, key, tick):
        self.order[key] = tick
        self.last_touch[key] = tick
        self.freq[key] = 0

    def _remove_meta(self, key):
        self.order.pop(key, None)
        self.last_touch.pop(key, None)
        self.freq.pop(key, None)
        self._prefetched.discard(key)

    def admit_new(self, key: int, nbytes: int, tick: int) -> None:
        self.last_evictions = []
        self._evict_until(key, nbytes)
        if self.resident_bytes + nbytes <= self.capacity_bytes:
            self.resident.add(key)
            self.resident_bytes += nbytes
            self._insert(key, tick)

    def _evict_until(self, incoming_key: int, nbytes: int) -> None:
        victims = []
        while (self.resident
               and self.resident_bytes + nbytes > self.capacity_bytes):
            v = self._pick_victim(exclude=incoming_key)
            if v is None:
                break
            self.resident.discard(v)
            self.resident_bytes -= self._default_nbytes
            self._remove_meta(v)
            victims.append(v)
        self.last_evictions = victims

    def _pick_victim(self, exclude: int) -> int | None:
        # PURE LRU on the demand path: evict least-recently-touched.
        candidates = [k for k in self.resident if k != exclude]
        if not candidates:
            return None
        return min(candidates, key=lambda k: self.last_touch[k])

    def mark_prefetched(self, key: int) -> None:
        self._prefetched.add(key)

    def free_slots(self, record_bytes: int) -> int:
        return (self.capacity_bytes - self.resident_bytes) // record_bytes


class PrefetchEngine(Engine):
    """Demand engine + causal prefetch injection at bundle starts.

    prefetch_mode:
      "off"      baseline (pure demand)
      "safe"     prefetch only into free slots
      "replace"  additionally evict lowest-value resident beyond
                 hysteresis (min_dwell), bounded per bundle
    """

    def __init__(self, policy, record_bytes: int,
                 ranked_fn, prefetch_mode: str = "off",
                 max_prefetch_per_bundle: int = 6,
                 min_dwell: int = 6):
        super().__init__(policy, record_bytes=record_bytes)
        self.ranked_fn = ranked_fn
        self.mode = prefetch_mode
        self.max_prefetch = max_prefetch_per_bundle
        self.min_dwell = min_dwell
        # extra stats
        self.prefetch_issued = 0
        self.prefetch_useful = 0        # demanded while still resident
        self.prefetch_wasted = 0        # evicted before any demand
        self.prefetch_bytes = 0
        self.wasted_prefetch_bytes = 0
        self.demand_misses_converted = 0  # demand miss avoided by prefetch
        self.pollution_events = 0         # demand miss caused by a victim
        # that was prefetched-but-unused or displaced by prefetch
        self._prefetched_live: set[int] = set()

    # ------------------------------------------------------------------
    def run(self, demands) -> Metrics:
        # Reindex to slice-local ticks for consistency with the ranked
        # provider (positions within this layer's demand sequence).
        from engine import _reindex
        demands = _reindex(demands)
        seen_bundle: set[tuple[int, int]] = set()
        for d in demands:
            key = (d.step, d.layer)
            if key not in seen_bundle:
                seen_bundle.add(key)
                if (d.step > 0 and self.mode != "off"
                        and self.ranked_fn is not None):
                    self._prefetch_bundle(d)
            self.access_demand(d.scope, d.seq, d.nbytes)
        self._close_out_prefetches()
        return self.m

    def _prefetch_bundle(self, d) -> None:
        pol = self.policy
        if not hasattr(pol, "mark_prefetched"):
            return
        self._note_tick(d.seq)
        candidates = self.ranked_fn(d.seq)
        issued = 0
        for key, _score in candidates:
            if issued >= self.max_prefetch:
                break
            if pol.has(key):
                continue
            free = pol.free_slots(self.record_bytes)
            if free > 0:
                self._prefetch_insert(key, d)
                issued += 1
                continue
            if self.mode != "replace":
                continue
            victim = self._hysteresis_victim()
            if victim is None:
                continue
            pol.remove(victim)
            self.m.evictions += 1
            self._record_eviction(victim, d.seq, by_prefetch=True)
            self._prefetch_insert(key, d)
            issued += 1

    def _hysteresis_victim(self) -> int | None:
        pol = self.policy
        tick = self._current_tick()
        cands = []
        for k in pol.resident:
            it = self._insert_tick.get(k)
            if it is None or tick - it < self.min_dwell:
                continue
            if k in pol._p:  # predicted to be reused: protect
                continue
            cands.append(k)
        if not cands:
            return None

        def score(k):
            pred = 0 if k in pol._p else 1   # evict unpredicted first
            fr = pol.freq.get(k, 0)
            age = tick - pol.order[k]
            return (pred, fr, -age)          # lowest wins

        return min(cands, key=score)

    def _prefetch_insert(self, key: int, d) -> None:
        pol = self.policy
        pol.admit_new(key, self.record_bytes, d.seq)
        if key in pol.resident:
            # admit_new may have internally evicted victims
            # (pol.last_evictions is set by _evict_until); account for
            # them exactly like the demand path does.
            for v in list(pol.last_evictions):
                self.m.evictions += 1
                self._record_eviction(v, d.seq, by_prefetch=True)
            self.prefetch_issued += 1
            self.prefetch_bytes += self.record_bytes
            self._insert_tick[key] = d.seq
            self._prefetched_live.add(key)
            pol.mark_prefetched(key)

    def _record_eviction(self, key: int, tick: int,
                         by_prefetch: bool = False) -> None:
        m = self.m
        it = self._insert_tick.pop(key, None)
        la = self._last_access.get(key)
        if key in self._prefetched_live:
            # prefetched and never demanded while resident with THIS
            # prefetch -> wasted, even if an earlier residency of the
            # same key was useful (re-prefetch case).
            self._prefetched_live.discard(key)
            self.prefetch_wasted += 1
            self.wasted_prefetch_bytes += self.record_bytes
        if it is not None:
            if la is not None and la >= it:
                m.useful_residency_ticks += la - it
                m.wasted_residency_ticks += tick - la
            else:
                m.wasted_residency_ticks += tick - it

    def _note_tick(self, tick: int) -> None:
        if hasattr(self.policy, "note_tick"):
            self.policy.note_tick(tick)

    def _current_tick(self) -> int:
        return self._last_tick if hasattr(self, "_last_tick") else 0

    # access_demand override: track prefetch usefulness + last_tick
    def access_demand(self, key: int, tick: int, nbytes: int) -> None:
        self._last_tick = tick
        was_prefetched = key in self._prefetched_live
        super().access_demand(key, tick, nbytes)
        if was_prefetched:
            # arrived while still prefetched-live: a demand miss avoided
            self.prefetch_useful += 1
            self.demand_misses_converted += 1
            self._prefetched_live.discard(key)

    def _on_evict_hook(self, key: int, tick: int) -> None:
        """Called by base Engine._admit for each demand-path eviction.

        A prefetched-live key evicted here was never demanded since its
        prefetch -> count as wasted prefetch (residency accounting was
        already done by the base engine; do not repeat it).
        """
        if key in self._prefetched_live:
            self._prefetched_live.discard(key)
            self.prefetch_wasted += 1
            self.wasted_prefetch_bytes += self.record_bytes

    def _close_out_prefetches(self) -> None:
        for key in list(self._prefetched_live):
            self.prefetch_wasted += 1
            self.wasted_prefetch_bytes += self.record_bytes
            self._prefetched_live.discard(key)



def theoretical_stall_sweep(misses_by_config, record_bytes,
                            bandwidths_gib_s=(3.0, 6.0, 12.0, 25.0),
                            overlap_fraction=None):
    """THEORETICAL only: exposed-stall estimate per config.

    exposed_ms = misses * read_ms / (1 + prefetch_lead_useful_ratio)
    where read_ms uses measured p50 and bandwidth scales linearly.
    overlap_fraction defaults to the trace-measured 82.3%.
    """
    if overlap_fraction is None:
        overlap_fraction = MEASURED_READ_MS["fill_overlap_percent"] / 100.0
    p50 = MEASURED_READ_MS["p50"]
    rows = []
    for label, misses in misses_by_config:
        row = {"config": label, "misses": misses}
        for bw in bandwidths_gib_s:
            # p50 was measured at ~54.2 MiB/s effective per-record read;
            # scale inversely with bandwidth for the theoretical sweep.
            read_ms = p50 * (MEASURED_READ_MS["bandwidth_mib_s"] / (bw * 1024))
            exposed = misses * read_ms * (1.0 - overlap_fraction)
            row[f"exposed_stall_ms@{bw}GiB_s"] = round(exposed, 1)
        rows.append(row)
    return rows
