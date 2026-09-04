"""Cache replacement policies for the offline DSV4 tournament.

All policies operate on (layer, expert) "scope" keys with uniform
record bytes today; variable-size records are supported through an
optional per-key byte map (byte-aware capacity).

Offline policies (Belady) may inspect the future demand list; online
policies may not.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from typing import Callable


class Policy:
    """Base class. Capacity is expressed in BYTES (capacity_bytes)."""

    name = "base"

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.resident_bytes = 0
        self.resident: set[int] = set()
        self.last_evictions: list[int] = []  # victims of the last admit_new

    def resident_count(self) -> int:
        return len(self.resident)

    def has(self, key: int) -> bool:
        return key in self.resident

    def admit_new(self, key: int, nbytes: int, tick: int) -> None:
        """A compulsory access just missed; policy may insert or bypass."""
        raise NotImplementedError

    def on_hit(self, key: int, tick: int) -> None:
        pass

    def note_tick(self, tick: int) -> None:
        """Called before each access with the demand index."""

    def victims_for_insert(self, nbytes: int) -> list[int]:
        """Keys this policy would evict to make room for nbytes."""
        return []


def _evict_until_fits(policy: Policy, nbytes: int) -> list[int]:
    """Shared byte-accounting eviction loop driven by pick_victim()."""
    victims = []
    while policy.resident and policy.resident_bytes + nbytes > policy.capacity_bytes:
        v = policy.pick_victim()
        if v is None:
            break
        policy.remove(v)
        victims.append(v)
    policy.last_evictions = victims
    return victims


class OnlinePolicy(Policy):
    """Online policies pick victims from internal metadata only."""

    def pick_victim(self) -> int | None:
        raise NotImplementedError

    def admit_new(self, key: int, nbytes: int, tick: int) -> None:
        for v in _evict_until_fits(self, nbytes):
            pass
        if self.resident_bytes + nbytes <= self.capacity_bytes:
            self.resident.add(key)
            self.resident_bytes += nbytes
            self._insert(key, tick)

    def _insert(self, key: int, tick: int) -> None:
        pass

    def remove(self, key: int) -> None:
        self.resident.discard(key)
        self.resident_bytes -= self._nbytes_of(key)
        self._remove_meta(key)

    # byte map hook (uniform by default)
    _bytes: dict[int, int] = {}

    def set_bytes(self, bmap: dict[int, int]) -> None:
        self._bytes = bmap

    def _nbytes_of(self, key: int) -> int:
        return self._bytes.get(key, self._default_nbytes)

    _default_nbytes: int = 0


class LRU(OnlinePolicy):
    name = "lru"

    def __init__(self, capacity_bytes: int, default_nbytes: int = 1):
        super().__init__(capacity_bytes)
        self._default_nbytes = default_nbytes
        self.order: OrderedDict[int, None] = OrderedDict()

    def _insert(self, key, tick):
        self.order[key] = None

    def on_hit(self, key, tick):
        self.order.move_to_end(key)

    def _remove_meta(self, key):
        self.order.pop(key, None)

    def pick_victim(self):
        return next(iter(self.order), None)


class LFU(OnlinePolicy):
    name = "lfu"

    def __init__(self, capacity_bytes: int, default_nbytes: int = 1):
        super().__init__(capacity_bytes)
        self._default_nbytes = default_nbytes
        self.freq: defaultdict[int, int] = defaultdict(int)

    def _insert(self, key, tick):
        self.freq[key] = 0

    def on_hit(self, key, tick):
        self.freq[key] += 1

    def _remove_meta(self, key):
        self.freq.pop(key, None)

    def pick_victim(self):
        return min(self.order_keys(), key=lambda k: self.freq[k])

    def order_keys(self):
        return list(self.resident)


class LeastStale(OnlinePolicy):
    """SpecMD-inspired: evict the expert whose last use is the furthest in
    the past *relative to its typical inter-use gap* (staleness ratio).
    A key not used for longer than its usual gap is 'stale'.
    Falls back to plain oldest-last-use when gaps are unknown.
    """

    name = "least_stale"

    def __init__(self, capacity_bytes: int, default_nbytes: int = 1):
        super().__init__(capacity_bytes)
        self._default_nbytes = default_nbytes
        self.last_used: dict[int, int] = {}
        self.gap_est: defaultdict[int, float] = defaultdict(float)
        self.seen_count: defaultdict[int, int] = defaultdict(int)

    def _insert(self, key, tick):
        self.last_used[key] = tick

    def on_hit(self, key, tick):
        prev = self.last_used.get(key)
        if prev is not None:
            n = self.seen_count[key]
            gap = tick - prev
            # EWMA of inter-use gap
            self.gap_est[key] = (self.gap_est[key] * n + gap) / (n + 1)
            self.seen_count[key] = n + 1
        self.last_used[key] = tick

    def _remove_meta(self, key):
        self.last_used.pop(key, None)
        self.gap_est.pop(key, None)
        self.seen_count.pop(key, None)

    def pick_victim(self):
        def stale_score(k):
            age = self._tick - self.last_used[k]
            gap = self.gap_est.get(k, 0.0)
            if gap <= 0:
                return age  # unknown gap -> plain age
            return age / gap  # >1 means overdue; evict most-overdue

        return max(self.resident, key=stale_score)

    _tick: int = 0

    def note_tick(self, tick: int) -> None:
        self._tick = tick


class ARCLike(OnlinePolicy):
    """Adaptive Replacement Cache (Megiddo & Modha, FAST 2003).

    T1: recency, T2: frequency; B1/B2 ghost lists adapt the pivot p.
    Implemented in ITEM COUNTS (paper-faithful), which is exact for
    today's uniform-size DSV4 expert records; the byte accounting is
    count * record_bytes.
    """

    name = "arc"

    def __init__(self, capacity_bytes: int, default_nbytes: int = 1):
        super().__init__(capacity_bytes)
        self._default_nbytes = default_nbytes
        self.c = max(1, capacity_bytes // default_nbytes)
        self.p = 0.0
        self.t1: OrderedDict[int, None] = OrderedDict()
        self.t2: OrderedDict[int, None] = OrderedDict()
        self.b1: OrderedDict[int, None] = OrderedDict()
        self.b2: OrderedDict[int, None] = OrderedDict()

    def _replace(self, key: int, nbytes: int) -> None:
        if self.t1 and (
            (key in self.b2 and len(self.t1) == 1)
            or (len(self.t1) > self.p)
        ):
            old, _ = self.t1.popitem(last=False)
            self.resident.discard(old)
            self.resident_bytes -= self._default_nbytes
            self.b1[old] = None
            self.last_evictions.append(old)
        elif self.t2:
            old, _ = self.t2.popitem(last=False)
            self.resident.discard(old)
            self.resident_bytes -= self._default_nbytes
            self.b2[old] = None
            self.last_evictions.append(old)
        elif self.t1:
            old, _ = self.t1.popitem(last=False)
            self.resident.discard(old)
            self.resident_bytes -= self._default_nbytes
            self.b1[old] = None
            self.last_evictions.append(old)

    def on_hit(self, key, tick):
        if key in self.t1:
            self.t1.move_to_end(key)
            del self.t1[key]
            self.t2[key] = None
        else:
            self.t2.move_to_end(key)

    def admit_new(self, key: int, nbytes: int, tick: int) -> None:
        self.last_evictions = []
        if key in self.b1:
            # CASE II: ghost hit in B1 (frequent-in-past) -> favor T2
            delta = max(len(self.b2) / max(1, len(self.b1)), 1.0)
            self.p = min(float(self.c), self.p + delta)
            self._replace(key, nbytes)
            self.b1.pop(key, None)
            self.t2[key] = None
        elif key in self.b2:
            # CASE III: ghost hit in B2 (recency-in-past) -> favor T1
            delta = max(len(self.b1) / max(1, len(self.b2)), 1.0)
            self.p = max(0.0, self.p - delta)
            self._replace(key, nbytes)
            self.b2.pop(key, None)
            self.t2[key] = None
        else:
            # CASE IV: fresh miss
            L1 = len(self.t1) + len(self.b1)
            L2 = len(self.t2) + len(self.b2)
            if L1 == self.c and len(self.t1) < self.c:
                # (i): B1 full-ish, T1 has room -> drop a B1 ghost, replace
                if self.b1:
                    self.b1.popitem(last=False)
                self._replace(key, nbytes)
            elif L1 == self.c:
                # (i-alt): T1 == C -> drop T1 LRU entirely (no ghost)
                old, _ = self.t1.popitem(last=False)
                self.resident.discard(old)
                self.resident_bytes -= self._default_nbytes
                self.last_evictions.append(old)
            elif L1 < self.c and L1 + L2 >= self.c:
                # (ii): REPLACE unconditionally (paper case iv-2)
                if L1 + L2 >= 2 * self.c and self.b2:
                    self.b2.popitem(last=False)
                self._replace(key, nbytes)
            self.t1[key] = None
        self.resident.add(key)
        self.resident_bytes += nbytes
        # ghost trim safety (paper keeps |B1|+|B2| <= 2C)
        while len(self.b1) + len(self.b2) > 2 * self.c:
            if self.b1:
                self.b1.popitem(last=False)
            elif self.b2:
                self.b2.popitem(last=False)
            else:
                break


class Belady(OnlinePolicy):
    """Offline oracle: evict the resident key whose next use is furthest
    away (never used again = immediate victim)."""

    name = "belady"

    def __init__(self, capacity_bytes: int, default_nbytes: int = 1,
                 future: list[int] | None = None):
        super().__init__(capacity_bytes)
        self._default_nbytes = default_nbytes
        assert future is not None
        n = len(future)
        # next_occ[i] = next index j > i with future[j] == future[i]
        nxt = [1 << 60] * n
        last: dict[int, int] = {}
        for i in range(n - 1, -1, -1):
            nxt[i] = last.get(future[i], 1 << 60)
            last[future[i]] = i
        self.next_occ = nxt
        self.last_pos: dict[int, int] = {}  # key -> most recent demand pos
        self._pos = -1

    def _insert(self, key, tick):
        self.last_pos[key] = tick

    def _remove_meta(self, key):
        self.last_pos.pop(key, None)

    def note_tick(self, tick: int) -> None:
        self._pos += 1  # engine ticks are demand indices

    def pick_victim(self):
        nxt = self.next_occ
        lp = self.last_pos
        return max(self.resident,
                   key=lambda k: nxt[lp[k]] if k in lp else (1 << 60))

    def on_hit(self, key, tick):
        self.last_pos[key] = tick


class RecencyFreqHybrid(OnlinePolicy):
    """Simple recency-frequency blend: score = freq / (1 + age_in_ticks)."""

    name = "recency_freq"

    def __init__(self, capacity_bytes: int, default_nbytes: int = 1):
        super().__init__(capacity_bytes)
        self._default_nbytes = default_nbytes
        self.freq: defaultdict[int, int] = defaultdict(int)
        self.last_used: dict[int, int] = {}
        self._tick = 0

    def note_tick(self, tick: int) -> None:
        self._tick = tick

    def _insert(self, key, tick):
        self.freq[key] = 0
        self.last_used[key] = tick

    def on_hit(self, key, tick):
        self.freq[key] += 1
        self.last_used[key] = tick

    def _remove_meta(self, key):
        self.freq.pop(key, None)
        self.last_used.pop(key, None)

    def pick_victim(self):
        return min(
            self.resident,
            key=lambda k: (self.freq[k] + 1) / (1 + self._tick - self.last_used[k]),
        )


class PredictionAware(OnlinePolicy):
    """Keeps experts the predictor expects to reuse soon.

    score(key) = predicted_next_use_distance (smaller = keep).
    Experts not in the prediction window are evicted first, LRU order
    as tie-break. The predictor is causal: it only sees demands before
    the current tick.
    """

    name = "pred_aware"

    def __init__(self, capacity_bytes: int, default_nbytes: int = 1,
                 predictor: Callable[[int], dict[int, int]] | None = None):
        super().__init__(capacity_bytes)
        self._default_nbytes = default_nbytes
        self.predictor = predictor  # tick -> {key: predicted_next_tick}
        self._pred: dict[int, int] = {}
        self._tick = 0
        self.order: OrderedDict[int, None] = OrderedDict()

    def note_tick(self, tick: int) -> None:
        self._tick = tick
        if self.predictor is not None:
            self._pred = self.predictor(tick)

    def _insert(self, key, tick):
        self.order[key] = None

    def on_hit(self, key, tick):
        self.order.move_to_end(key)

    def _remove_meta(self, key):
        self.order.pop(key, None)

    def pick_victim(self):
        def rank(k):
            d = self._pred.get(k)
            if d is None:
                return (1, 0)  # not predicted
            return (0, d)

        return max(self.resident, key=rank)


def make_policy(name: str, capacity_bytes: int, default_nbytes: int,
                future: list[int] | None = None,
                predictor: Callable[[int], dict[int, int]] | None = None) -> Policy:
    if name == "lru":
        return LRU(capacity_bytes, default_nbytes)
    if name == "lfu":
        return LFU(capacity_bytes, default_nbytes)
    if name == "least_stale":
        return LeastStale(capacity_bytes, default_nbytes)
    if name == "arc":
        return ARCLike(capacity_bytes, default_nbytes)
    if name == "belady":
        return Belady(capacity_bytes, default_nbytes, future=future)
    if name == "recency_freq":
        return RecencyFreqHybrid(capacity_bytes, default_nbytes)
    if name == "pred_aware":
        return PredictionAware(capacity_bytes, default_nbytes, predictor=predictor)
    raise ValueError(f"unknown policy {name}")
