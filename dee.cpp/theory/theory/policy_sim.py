"""Trace-driven cache-policy and prefetch simulator (free: replays committed route journals).

What it is for.  Phase 6 (run p6m-l4-r5) measured a 0% host-tier hit rate at b=8 on a 16 GiB
host tier: one decode step asks for more (layer, expert) records than the tier holds, and LRU
then evicts every record just before its next use (``batch_cliff.py`` explains the kill).  This
module scores replacement policies and prefetch schedules against the SAME request streams the
engine issued, before any billed run:

  * eviction: ``lru`` (the engine today), ``next_use`` (evict the record whose predicted next
    use is furthest away; what the engine now implements), ``layer_distance`` (next_use without
    the reuse-probability term) and ``belady`` (true furthest-next-use with future knowledge:
    an upper bound no online policy can beat);
  * lookahead prefetch: hints for layer L+1 issued while layer L runs, with a FIFO or a
    deadline-ordered read queue.

Fidelity to the engine (src/engine.cpp, src/host_pack_cache.cpp, src/vram_cache.cpp):

  * a layer call looks up its de-duplicated expert set in ascending expert-id order;
  * the host tier is consulted for EVERY active expert, VRAM-resident or not (the engine needs
    the packed host bytes to build the staging request), unless ``vram_bypass`` is on;
  * host lookups go in sub-batches of ``queue_depth`` (the engine's source_read_queue_depth);
    a sub-batch protects only its own keys from eviction;
  * VRAM ensures are sequential and pin as they go, so a resident block of the current layer
    that has not been reached yet can be evicted by an earlier cold load in the same call.

Exactness.  Nothing here touches routing: a policy only decides where a record sits and when it
moves.  The simulator consumes the authoritative route journal and never rewrites it.

Credits.  Next-use ("probabilistic Belady") eviction and the deadline-ordered queue follow
SeqMoE (arXiv 2609.12978); the layer-ahead hint follows Pre-gated MoE (Hwang et al., ISCA 2024)
and ProMoE.  Belady's MIN is the classical optimal offline policy (Belady, 1966).
"""
from __future__ import annotations

import heapq
import json
import random
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Key = Tuple[int, int]            # (layer, expert)

RECORD_BYTES = 13_369_344        # one packed DSv4 expert record (PREDICTIONS_PHASE6.md)
GIB = 1 << 30


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LayerCall:
    step: int
    layer: int
    experts: Tuple[int, ...]     # de-duplicated, ascending (engine dispatch order)

    @property
    def keys(self) -> List[Key]:
        return [(self.layer, e) for e in self.experts]


def load_journal(path: Path | str) -> List[LayerCall]:
    """One routed_experts.jsonl (RoutedExpertJournal schema 1) as layer calls."""
    calls: List[LayerCall] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            experts = sorted({int(e) for row in d["expert_ids_rank_order"] for e in row})
            calls.append(LayerCall(int(d["forward_step"]), int(d["layer"]), tuple(experts)))
    return calls


def merge_cohort(journals: Sequence[Sequence[LayerCall]]) -> List[LayerCall]:
    """Decode b prompts together: one call per (step, layer) with the union of the rows.

    Rows of a cohort route independently (exactness), so the cohort's request set at a step is
    the union of the single-prompt sets at that step.  Steps a prompt does not have are skipped
    for that prompt.
    """
    merged: Dict[Tuple[int, int], Set[int]] = {}
    for calls in journals:
        for c in calls:
            merged.setdefault((c.step, c.layer), set()).update(c.experts)
    return [LayerCall(s, l, tuple(sorted(e))) for (s, l), e in sorted(merged.items())]


def n_layers_of(calls: Sequence[LayerCall]) -> int:
    return max(c.layer for c in calls) + 1


# ---------------------------------------------------------------------------
# Policies.  ``victim(candidates)`` returns the key to evict among ``candidates``.
# ---------------------------------------------------------------------------
class Policy:
    """Replacement policy for one tier.  ``rank(key)``: larger = evict sooner."""
    name = "base"

    def __init__(self) -> None:
        self._order: List[Key] = []      # residents by rank, ascending; rebuilt per layer call
        self._order_valid = False
        self._dirty: Set[Key] = set()    # keys whose rank changed since the rebuild

    def begin_layer(self, step: int, layer: int, call_index: int) -> None:
        self._order_valid = False

    def on_use(self, key: Key) -> None:
        self._dirty.add(key)

    def forget(self, key: Key) -> None:
        pass

    def rank(self, key: Key) -> float:
        raise NotImplementedError

    def victims(self, resident: Set[Key], protected: Set[Key], k: int) -> List[Key]:
        """The k unprotected residents with the largest rank, worst first.

        Exactly a full scan per eviction, but done incrementally: ranks only change for keys
        that were used (or inserted) since the layer call began, so those are re-ranked on
        every pick and everything else comes off one sorted list.
        """
        if not self._order_valid:
            self._order = sorted(resident, key=self.rank)
            self._order_valid = True
            self._dirty = set()
        out: List[Key] = []
        taken: Set[Key] = set()
        order = self._order
        while len(out) < k:
            while order and (order[-1] not in resident or order[-1] in self._dirty
                             or order[-1] in taken):
                order.pop()
            i = len(order) - 1
            while i >= 0 and order[i] in protected:
                i -= 1
            best: Optional[Key] = order[i] if i >= 0 else None
            best_rank = self.rank(best) if best is not None else float("-inf")
            for d in self._dirty:
                if d in resident and d not in protected and d not in taken:
                    r = self.rank(d)
                    if r > best_rank:
                        best, best_rank = d, r
            if best is None:
                break
            out.append(best)
            taken.add(best)
        return out


class Lru(Policy):
    name = "lru"

    def __init__(self) -> None:
        super().__init__()
        self.tick = 0
        self.last: Dict[Key, int] = {}

    def on_use(self, key: Key) -> None:
        super().on_use(key)
        self.tick += 1
        self.last[key] = self.tick

    def forget(self, key: Key) -> None:
        self.last.pop(key, None)

    def rank(self, key: Key) -> float:
        return -float(self.last.get(key, 0))


class NextUse(Policy):
    """Evict the record whose expected next use is furthest away.

    Decode sweeps the layers in a fixed cycle, so a record of layer L can next be used when the
    sweep reaches L again: ``d = (L - current_layer) mod n_layers`` layer calls from now.  It is
    used at that visit with probability p, else one cycle later, and so on, which gives

        E[next use] = d + n_layers * (1/p - 1)        (geometric over visits)

    p is an exponentially weighted reuse estimate per record: each visit of its layer multiplies
    it by (1 - alpha), each use adds alpha.  A record never seen before starts at ``p0``.
    ``use_probability=False`` drops the second term (pure cyclic distance, LRU tie-break).

    A record of the layer running now has d = 0 and is never preferred as a victim: the engine
    may still need it later in this same call.
    """
    name = "next_use"

    def __init__(self, n_layers: int, alpha: float = 0.2, p0: float = 0.1,
                 p_floor: float = 1e-3, use_probability: bool = True) -> None:
        super().__init__()
        self.n = int(n_layers)
        self.alpha = float(alpha)
        self.p0 = float(p0)
        self.p_floor = float(p_floor)
        self.use_probability = bool(use_probability)
        if not use_probability:
            self.name = "layer_distance"
        self.cur = 0
        self.visits = [0] * self.n
        self.ewma: Dict[Key, float] = {}
        self.last_visit: Dict[Key, int] = {}
        self.tick = 0
        self.last_tick: Dict[Key, int] = {}

    def begin_layer(self, step: int, layer: int, call_index: int) -> None:
        super().begin_layer(step, layer, call_index)
        self.cur = layer
        self.visits[layer] += 1

    def on_use(self, key: Key) -> None:
        super().on_use(key)
        self.tick += 1
        self.last_tick[key] = self.tick
        v = self.visits[key[0]]
        if key in self.ewma:
            gap = v - self.last_visit[key]
            if gap <= 0:
                return          # already counted in this visit (C++: idempotent)
            self.ewma[key] = self.ewma[key] * (1.0 - self.alpha) ** gap + self.alpha
        else:
            self.ewma[key] = self.p0 * (1.0 - self.alpha) + self.alpha
        self.last_visit[key] = v

    def probability(self, key: Key) -> float:
        layer = key[0]
        e = self.ewma.get(key)
        if e is None:
            return self.p0
        # visits of this layer since the record's last use that are already known misses;
        # the visit in progress is not known yet for records of the current layer
        elapsed = self.visits[layer] - self.last_visit[key]
        if layer == self.cur:
            elapsed -= 1
        if elapsed > 0:
            e *= (1.0 - self.alpha) ** elapsed
        return max(self.p_floor, min(1.0, e))

    def rank(self, key: Key) -> float:
        d = (key[0] - self.cur) % self.n
        if d == 0 and self.use_probability:
            return -1.0         # the running layer leaves last (C++: victim_rank)
        if not self.use_probability:
            # break ties toward the least recently used
            return d - 1e-9 * self.last_tick.get(key, 0)
        return d + self.n * (1.0 / self.probability(key) - 1.0)


class Belady(Policy):
    """Furthest true next use (offline optimum).  Needs the whole stream up front."""
    name = "belady"

    def __init__(self, calls: Sequence[LayerCall]) -> None:
        super().__init__()
        self.uses: Dict[Key, List[int]] = {}
        for i, c in enumerate(calls):
            for k in c.keys:
                self.uses.setdefault(k, []).append(i)
        self.now = 0

    def begin_layer(self, step: int, layer: int, call_index: int) -> None:
        super().begin_layer(step, layer, call_index)
        self.now = call_index

    def rank(self, key: Key) -> float:
        u = self.uses.get(key)
        if not u:
            return float("inf")
        # a use in the call running now still counts (the call may not have reached it)
        i = bisect_right(u, self.now - 1)
        return u[i] if i < len(u) else float("inf")


def make_policy(name: str, calls: Sequence[LayerCall], n_layers: int, **kw) -> Policy:
    if name == "lru":
        return Lru()
    if name == "next_use":
        return NextUse(n_layers, **kw)
    if name == "layer_distance":
        return NextUse(n_layers, use_probability=False, **kw)
    if name == "belady":
        return Belady(calls)
    raise ValueError(f"unknown policy {name!r}")


# ---------------------------------------------------------------------------
# One bounded tier
# ---------------------------------------------------------------------------
class Tier:
    def __init__(self, capacity: int, policy: Policy) -> None:
        self.capacity = int(capacity)
        self.policy = policy
        self.resident: Set[Key] = set()
        self.evictions = 0

    def __contains__(self, key: Key) -> bool:
        return key in self.resident

    def make_room(self, need: int, protected: Set[Key]) -> bool:
        """Evict until ``need`` more records fit.  False when protection blocks it."""
        over = len(self.resident) + need - self.capacity
        if over <= 0:
            return True
        chosen = self.policy.victims(self.resident, protected, over)
        for victim in chosen:
            self.resident.discard(victim)
            self.policy.forget(victim)
            self.evictions += 1
        return len(chosen) == over

    def insert(self, key: Key) -> None:
        self.resident.add(key)
        self.policy._dirty.add(key)


# ---------------------------------------------------------------------------
# Counting simulation (no clock): hits, source reads, VRAM loads per step
# ---------------------------------------------------------------------------
@dataclass
class StepCounts:
    step: int
    lookups: int = 0          # active (layer, expert) pairs this step
    host_lookups: int = 0
    host_hits: int = 0
    source_reads: int = 0
    vram_hits: int = 0
    vram_loads: int = 0


@dataclass
class SimResult:
    steps: List[StepCounts]
    config: Dict[str, object] = field(default_factory=dict)

    def window(self, from_step: int = 1, to_step: Optional[int] = None) -> Dict[str, float]:
        """Pooled counts over steps [from_step, to_step] (inclusive; default: all decode)."""
        rows = [s for s in self.steps
                if s.step >= from_step and (to_step is None or s.step <= to_step)]
        tot = {f: sum(getattr(s, f) for s in rows)
               for f in ("lookups", "host_lookups", "host_hits", "source_reads",
                         "vram_hits", "vram_loads")}
        n = max(1, len(rows))
        return {
            **tot,
            "steps": len(rows),
            "host_hit_rate": tot["host_hits"] / tot["host_lookups"] if tot["host_lookups"] else 0.0,
            "vram_hit_rate": tot["vram_hits"] / tot["lookups"] if tot["lookups"] else 0.0,
            "source_reads_per_step": tot["source_reads"] / n,
            "vram_loads_per_step": tot["vram_loads"] / n,
            "lookups_per_step": tot["lookups"] / n,
        }


def simulate(calls: Sequence[LayerCall], *, host_records: int, vram_records: int,
             host_policy: str = "lru", vram_policy: str = "lru",
             vram_bypass: bool = False, queue_depth: int = 6,
             policy_kw: Optional[dict] = None) -> SimResult:
    """Replay ``calls`` through a VRAM tier in front of a host tier."""
    n_layers = n_layers_of(calls)
    kw = policy_kw or {}
    host = Tier(host_records, make_policy(host_policy, calls, n_layers,
                                          **(kw if host_policy != "lru" and host_policy != "belady" else {})))
    vram = Tier(vram_records, make_policy(vram_policy, calls, n_layers,
                                          **(kw if vram_policy != "lru" and vram_policy != "belady" else {})))
    steps: Dict[int, StepCounts] = {}
    for index, call in enumerate(calls):
        sc = steps.setdefault(call.step, StepCounts(call.step))
        host.policy.begin_layer(call.step, call.layer, index)
        vram.policy.begin_layer(call.step, call.layer, index)
        keys = call.keys
        sc.lookups += len(keys)
        chunk = max(1, vram.capacity)
        for first in range(0, len(keys), chunk):
            part = keys[first:first + chunk]
            # ---- host tier: every active expert, in sub-batches of queue_depth
            wanted = [k for k in part if not (vram_bypass and k in vram)]
            for q in range(0, len(wanted), max(1, queue_depth)):
                batch = wanted[q:q + max(1, queue_depth)]
                protected = set(batch)
                misses = [k for k in batch if k not in host]
                sc.host_lookups += len(batch)
                sc.host_hits += len(batch) - len(misses)
                sc.source_reads += len(misses)
                for k in batch:
                    host.policy.on_use(k)
                if misses:
                    host.make_room(len(misses), protected)
                    for k in misses:
                        host.insert(k)
            # ---- VRAM tier: sequential ensure, pinning as it goes
            pinned: Set[Key] = set()
            for k in part:
                vram.policy.on_use(k)
                if k in vram:
                    sc.vram_hits += 1
                else:
                    sc.vram_loads += 1
                    vram.make_room(1, pinned)
                    vram.insert(k)
                pinned.add(k)
    cfg = {"host_records": host_records, "vram_records": vram_records,
           "host_policy": host_policy, "vram_policy": vram_policy,
           "vram_bypass": vram_bypass, "queue_depth": queue_depth,
           "n_layers": n_layers, "policy_kw": kw}
    return SimResult([steps[s] for s in sorted(steps)], cfg)


# ---------------------------------------------------------------------------
# Lookahead prefetch with a read queue (adds a clock)
# ---------------------------------------------------------------------------
@dataclass
class TimeModel:
    """Per-step wall-time model.  All values are milliseconds.

    read_ms   one source read on one lane (record bytes / per-lane read rate)
    lanes     concurrent read lanes (the engine's source_read_lanes)
    stage_ms  host->VRAM staging of one record that is not VRAM-resident (gather + H2D)
    layer_ms  everything else in one layer call (dense blocks, expert compute)
    """
    read_ms: float
    lanes: int = 4
    stage_ms: float = 0.0
    layer_ms: float = 0.0


class ReadPool:
    """``lanes`` non-preemptive readers pulling from one queue.

    order='deadline': the pending read with the earliest deadline goes first, so a read the
    model is waiting on (deadline = now) always beats a speculative one.
    order='fifo': strict arrival order; a speculative read queued earlier delays a demand read.
    """

    def __init__(self, lanes: int, read_ms: float, order: str = "deadline") -> None:
        if order not in ("deadline", "fifo"):
            raise ValueError(order)
        self.order = order
        self.read_ms = float(read_ms)
        self.free_at = [0.0] * int(lanes)
        self.pending: List[Tuple[float, int, Key]] = []    # (sort key, seq, key)
        self.entry: Dict[Key, Tuple[float, int]] = {}
        self.done_at: Dict[Key, float] = {}                # started (in flight or finished)
        self.seq = 0
        self.reads = 0

    def submit(self, key: Key, deadline: float) -> None:
        if key in self.done_at:
            return
        old = self.entry.get(key)
        if old is not None:
            if self.order == "fifo" or old[0] <= deadline:
                return
        self.seq += 1
        rank = deadline if self.order == "deadline" else float(self.seq)
        self.entry[key] = (rank, self.seq)
        heapq.heappush(self.pending, (rank, self.seq, key))

    def cancel(self, key: Key) -> None:
        self.entry.pop(key, None)

    def _pop(self) -> Optional[Key]:
        while self.pending:
            rank, seq, key = heapq.heappop(self.pending)
            if self.entry.get(key) == (rank, seq):
                del self.entry[key]
                return key
        return None

    def run_until(self, t: float, now: float) -> None:
        """Start queued reads on lanes that come free before ``t`` (never before ``now``)."""
        while self.pending:
            lane = min(range(len(self.free_at)), key=self.free_at.__getitem__)
            start = max(self.free_at[lane], now)
            if start >= t:
                return
            key = self._pop()
            if key is None:
                return
            self.free_at[lane] = start + self.read_ms
            self.done_at[key] = start + self.read_ms
            self.reads += 1

    def wait_for(self, keys: Sequence[Key], now: float) -> float:
        """Run the queue until every key in ``keys`` has been read; return that time."""
        need = {k for k in keys if k not in self.done_at}
        while need:
            lane = min(range(len(self.free_at)), key=self.free_at.__getitem__)
            start = max(self.free_at[lane], now)
            key = self._pop()
            if key is None:
                raise RuntimeError("demand key was never submitted")
            self.free_at[lane] = start + self.read_ms
            self.done_at[key] = start + self.read_ms
            self.reads += 1
            need.discard(key)
        return max([now] + [self.done_at[k] for k in keys])

    def take(self, key: Key) -> None:
        self.done_at.pop(key, None)


def hint_set(true_next: Sequence[Key], *, recall: float, precision: float,
             n_experts: int, rng: random.Random) -> List[Key]:
    """A synthetic layer-ahead hint with the given recall and precision vs the true set."""
    if not true_next:
        return []
    layer = true_next[0][0]
    hits = [k for k in true_next if rng.random() < recall]
    wrong_n = int(round(len(hits) * (1.0 / max(precision, 1e-6) - 1.0)))
    truth = set(true_next)
    wrong: List[Key] = []
    while len(wrong) < wrong_n and len(wrong) + len(truth) < n_experts:
        k = (layer, rng.randrange(n_experts))
        if k not in truth and k not in wrong:
            wrong.append(k)
    return hits + wrong


@dataclass
class TimedResult:
    counts: SimResult
    step_ms: Dict[int, float]
    hint_reads: int = 0          # hints queued
    hint_reads_used: int = 0     # hinted records the model then asked for
    demand_wait_ms: float = 0.0  # time the model stood waiting on source reads
    reads_started: int = 0       # every source read that ran, demand and hint

    def tokens_per_second(self, rows: int, from_step: int = 1,
                          to_step: Optional[int] = None) -> float:
        ms = [v for s, v in self.step_ms.items()
              if s >= from_step and (to_step is None or s <= to_step)]
        return rows * len(ms) / (sum(ms) / 1000.0) if ms else 0.0


def simulate_timed(calls: Sequence[LayerCall], *, host_records: int, vram_records: int,
                   time_model: TimeModel, host_policy: str = "lru", vram_policy: str = "lru",
                   vram_bypass: bool = False, lookahead: Optional[str] = None,
                   recall: float = 1.0, precision: float = 1.0, queue: str = "deadline",
                   depth: int = 1, n_experts: int = 256, seed: int = 0,
                   policy_kw: Optional[dict] = None) -> TimedResult:
    """Replay with a clock.  ``lookahead``: None | 'synthetic' | 'prev_step'.

    While layer L stages and computes, the read pool keeps working on hints for layer L+1:
    'synthetic' draws them from the true next set at the given recall/precision (the real
    router-ahead hint has to be measured on a GPU; this sweeps what it could be), 'prev_step'
    uses the experts layer L+1 used on the previous step (computable from the journal alone).
    ``depth`` hints that many layer calls ahead.  Hints that are still queued when their layer
    starts are dropped under queue='deadline'; under 'fifo' they stay in line.
    """
    n_layers = n_layers_of(calls)
    kw = policy_kw or {}

    def mk(name: str) -> Policy:
        return make_policy(name, calls, n_layers, **(kw if name in ("next_use", "layer_distance") else {}))

    host, vram = Tier(host_records, mk(host_policy)), Tier(vram_records, mk(vram_policy))
    pool = ReadPool(time_model.lanes, time_model.read_ms, queue)
    rng = random.Random(seed)
    steps: Dict[int, StepCounts] = {}
    step_start: Dict[int, float] = {}
    step_end: Dict[int, float] = {}
    prev_sets: Dict[int, Tuple[Key, ...]] = {}
    hinted: Set[Key] = set()
    out = TimedResult(SimResult([]), {})
    now = 0.0
    for index, call in enumerate(calls):
        sc = steps.setdefault(call.step, StepCounts(call.step))
        step_start.setdefault(call.step, now)
        host.policy.begin_layer(call.step, call.layer, index)
        vram.policy.begin_layer(call.step, call.layer, index)
        keys = call.keys
        sc.lookups += len(keys)

        # hints for this layer that never started: drop (deadline) or leave in line (fifo)
        if queue == "deadline":
            for k in [k for k in hinted if k[0] == call.layer and k not in pool.done_at]:
                if k not in keys:
                    pool.cancel(k)
        hinted = {k for k in hinted if k[0] != call.layer}

        wanted = [k for k in keys if not (vram_bypass and k in vram)]
        protected = set(wanted)
        sc.host_lookups += len(wanted)
        demand: List[Key] = []
        for k in wanted:
            host.policy.on_use(k)
            if k in host:
                sc.host_hits += 1
            elif k in pool.done_at and pool.done_at[k] <= now:
                # a hint read finished early: it is host-resident by now
                out.hint_reads_used += 1
                sc.host_hits += 1
                host.make_room(1, protected)
                host.insert(k)
                pool.take(k)
            else:
                if k in pool.done_at:
                    out.hint_reads_used += 1          # in flight: wait for it, no new read
                else:
                    sc.source_reads += 1
                pool.submit(k, deadline=now)
                demand.append(k)

        # lookahead hints for the next ``depth`` layer calls, issued now (nearest first)
        if lookahead:
            for ahead in range(1, depth + 1):
                if index + ahead >= len(calls):
                    break
                nxt = calls[index + ahead]
                if lookahead == "synthetic":
                    guess = hint_set(nxt.keys, recall=recall, precision=precision,
                                     n_experts=n_experts, rng=rng)
                elif lookahead == "prev_step":
                    guess = list(prev_sets.get(nxt.layer, ()))
                else:
                    raise ValueError(lookahead)
                for k in guess:
                    if k in host or k in pool.done_at or k in pool.entry \
                            or (vram_bypass and k in vram):
                        continue
                    # after every demand read of this call, nearer layers first
                    pool.submit(k, deadline=now + ahead)
                    hinted.add(k)
                    out.hint_reads += 1

        t_ready = pool.wait_for(demand, now) if demand else now
        out.demand_wait_ms += t_ready - now
        for k in demand:
            host.make_room(1, protected)
            host.insert(k)
            pool.take(k)

        pinned: Set[Key] = set()
        loads = 0
        for k in keys:
            vram.policy.on_use(k)
            if k in vram:
                sc.vram_hits += 1
            else:
                loads += 1
                vram.make_room(1, pinned)
                vram.insert(k)
            pinned.add(k)
        sc.vram_loads += loads
        t_end = t_ready + loads * time_model.stage_ms + time_model.layer_ms
        pool.run_until(t_end, t_ready)
        # hint reads that finished become host-resident without a wait (they are one layer
        # ahead, so next_use ranks them as the soonest-needed records)
        for k in [k for k in hinted if k in pool.done_at and pool.done_at[k] <= t_end]:
            if k not in host:
                host.make_room(1, set(keys))
                host.insert(k)
                if isinstance(host.policy, Lru):
                    host.policy.on_use(k)
        prev_sets[call.layer] = tuple(keys)
        now = t_end
        step_end[call.step] = now

    out.counts = SimResult([steps[s] for s in sorted(steps)],
                           {"host_records": host_records, "vram_records": vram_records,
                            "host_policy": host_policy, "vram_policy": vram_policy,
                            "vram_bypass": vram_bypass, "lookahead": lookahead,
                            "recall": recall, "precision": precision, "queue": queue})
    out.step_ms = {s: step_end[s] - step_start[s] for s in step_start}
    out.reads_started = pool.reads
    return out


# ---------------------------------------------------------------------------
# Command line: the tables in theory/NEXT_USE_SIM.md
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("journals", nargs="+",
                    help="route journals (routed_experts-*.jsonl); several = one merged cohort")
    ap.add_argument("--host-gib", type=float, nargs="+", default=[16.0])
    ap.add_argument("--vram-records", type=int, default=281)
    ap.add_argument("--max-step", type=int, default=64)
    ap.add_argument("--window", type=int, nargs=2, default=[33, 64], metavar=("FROM", "TO"))
    ap.add_argument("--policies", nargs="+",
                    default=["lru", "layer_distance", "next_use", "belady"])
    ap.add_argument("--timed", action="store_true",
                    help="also replay with a clock (lookahead + queue order)")
    ap.add_argument("--read-ms", type=float, default=12.3)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--stage-ms", type=float, default=5.0)
    ap.add_argument("--layer-ms", type=float, default=20.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    journals = [load_journal(p) for p in args.journals]
    calls = [c for c in merge_cohort(journals) if c.step <= args.max_step]
    rows = len(journals)
    lo, hi = args.window
    out: Dict[str, object] = {"rows": rows, "window": [lo, hi],
                              "vram_records": args.vram_records, "tables": []}
    for gib in args.host_gib:
        host_records = int(gib * GIB // RECORD_BYTES)
        print(f"\nb={rows}, host {gib:g} GiB ({host_records} records), "
              f"VRAM {args.vram_records} records, steps {lo}-{hi}")
        print("| policy | host hit | VRAM hit | source reads/step | lookups/step |")
        print("|---|---|---|---|---|")
        for name in args.policies:
            w = simulate(calls, host_records=host_records, vram_records=args.vram_records,
                         host_policy=name, vram_policy=name).window(lo, hi)
            print(f"| {name} | {w['host_hit_rate']:.3f} | {w['vram_hit_rate']:.3f} | "
                  f"{w['source_reads_per_step']:.0f} | {w['lookups_per_step']:.0f} |")
            out["tables"].append({"host_gib": gib, "host_records": host_records,
                                  "policy": name, **w})
        if not args.timed:
            continue
        tm = TimeModel(read_ms=args.read_ms, lanes=args.lanes,
                       stage_ms=args.stage_ms, layer_ms=args.layer_ms)
        print("\n| policy | hints | queue | tok/s (model) | host hit | reads/step incl. hints |")
        print("|---|---|---|---|---|---|")
        steps = hi - lo + 1
        for policy, lookahead, queue in (
                ("lru", None, "deadline"), ("next_use", None, "deadline"),
                ("next_use", "prev_step", "deadline"), ("next_use", "prev_step", "fifo")):
            r = simulate_timed(calls, host_records=host_records,
                               vram_records=args.vram_records, time_model=tm,
                               host_policy=policy, vram_policy=policy,
                               lookahead=lookahead, queue=queue)
            w = r.counts.window(lo, hi)
            tps = r.tokens_per_second(rows, lo, hi)
            print(f"| {policy} | {lookahead or 'off'} | {queue} | {tps:.2f} | "
                  f"{w['host_hit_rate']:.3f} | {r.reads_started / max(1, len(r.step_ms)):.0f} |")
            out["tables"].append({"host_gib": gib, "policy": policy, "timed": True,
                                  "lookahead": lookahead, "queue": queue,
                                  "tokens_per_second_model": tps, "steps": steps, **w})
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
