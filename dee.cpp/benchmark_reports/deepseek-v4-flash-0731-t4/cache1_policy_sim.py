#!/usr/bin/env python3
"""CACHE1.1b — cache policy simulation on the sealed DS10 v12 access stream.

Replays the exact per-token, per-layer expert access sequence from the
evidence, split by the DS10 partition (GPU0 = layers 0..21, GPU1 = 22..42,
each with its own cache budget).  Models BOTH cache levels of the DS10
runtime:

  Level 1 — GPU DeepSeekExpertCache: LRU (score = last_use +
            priority*PRIORITY_WEIGHT), 48 MiB FP16 payloads, shared expert
            key = (layer,-1), optional pins.
  Level 2 — host ExpertProvider raw LRU (raw_experts_per_layer compact
            tensors/layer).  A GPU miss served by the provider is a host
            hit (no HTTP fetch, but dequant + H2D staging still happen);
            a provider miss is an HTTP range fetch.

Counters reported per policy:
  gpu_hit%   — fraction served entirely from GPU cache
  prov_hit%  — GPU miss but provider host hit (no HTTP)
  http_fetch — total expert fetch events (w1+w2+w3 group counted once)
  and the total fraction of accesses needing ANY network round trip.

Run: python cache1_policy_sim.py <evidence.json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

EXPERT_BYTES = 50331648  # 48 MiB FP16 payload
PIN = 10 ** 12


def flatten_expert_groups(sel: list) -> list[int]:
    out: list[int] = []

    def rec(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                rec(y)
        else:
            out.append(int(x))

    rec(sel)
    return out


class GpuCache:
    def __init__(self, budget_bytes: int, *, pin_shared: bool,
                 pinned_popular: set = frozenset()):
        self.budget = budget_bytes
        self.pin_shared = pin_shared
        self.pinned_popular = set(pinned_popular)
        self.entries: dict[tuple[int, int], tuple[int, int]] = {}
        self.tick = 0
        self.hits = 0
        self.misses = 0
        self.used = 0

    def __init__(self, budget_bytes: int, *, pin_shared: bool,
                 pinned_popular: set = frozenset()):
        self.budget = budget_bytes
        self.pin_shared = pin_shared
        self.pinned_popular = set(pinned_popular)
        self.entries: dict[tuple[int, int], tuple[int, int]] = {}
        self.tick = 0
        self.hits = 0
        self.misses = 0
        self.used = 0
        self.lfu = False

    def _score(self, key) -> int:
        last, freq = self.entries[key]
        if key[1] == -1 and self.pin_shared:
            return PIN  # max => never the min victim
        if key in self.pinned_popular:
            return PIN
        if self.lfu:
            return -freq  # min => evict lowest frequency
        return last

    def evict_until_free(self, need: int) -> None:
        while self.used + need > self.budget and self.entries:
            victim = min(self.entries, key=self._score)
            self.used -= EXPERT_BYTES
            del self.entries[victim]

    def access(self, key: tuple[int, int]) -> bool:
        self.tick += 1
        if key in self.entries:
            self.hits += 1
            last, freq = self.entries[key]
            self.entries[key] = (self.tick, freq + 1)
            return True
        self.misses += 1
        self.evict_until_free(EXPERT_BYTES)
        if self.used + EXPERT_BYTES > self.budget:
            return False  # cannot fit even alone (budget < one expert)
        self.entries[key] = (self.tick, 1)
        self.used += EXPERT_BYTES
        return False


class ProviderLru:
    """Host raw-payload LRU per layer (raw_experts_per_layer)."""

    def __init__(self, per_layer: int):
        self.per_layer = per_layer
        self.entries: dict[int, dict[int, int]] = defaultdict(dict)  # layer -> eid -> tick
        self.tick = 0
        self.hits = 0
        self.misses = 0

    def access(self, layer: int, eid: int) -> bool:
        self.tick += 1
        m = self.entries[layer]
        if eid in m:
            self.hits += 1
            m[eid] = self.tick
            return True
        self.misses += 1
        m[eid] = self.tick
        while len(m) > self.per_layer:
            # evict smallest tick (oldest)
            victim = min(m, key=m.get)
            del m[victim]
        return False


def build_stream(d: dict) -> tuple[list, int, int]:
    """Emit one access per DISTINCT expert per (token, layer) call, plus the
    shared expert — exactly what the real FFN does (it groups duplicate
    expert selections within a call before hitting the cache)."""
    g = d.get("gates", d)
    tt = g.get("token_trace", {})
    tokens = sorted(tt.keys(), key=lambda k: int(k.split("_")[1]))
    stream: list[tuple[int, tuple[int, int]]] = []
    n_layers = 0
    for tname in tokens:
        tok = tt[tname]
        idx = int(tname.split("_")[1])
        for row in tok.get("layers", []):
            lid = int(row["layer"])
            n_layers = max(n_layers, lid + 1)
            for e in dict.fromkeys(
                    flatten_expert_groups(row.get("selected_experts", []))):
                stream.append((idx, (lid, e)))
            stream.append((idx, (lid, -1)))  # shared expert per layer call
    return stream, len(tokens), n_layers


def run_policy(stream: list, split: int, *, label: str,
               budget: int, pin_shared: bool, provider_per_layer: int,
               popular_per_layer: int = 0, lfu: bool = False) -> dict:
    # popularity from full trace (oracle info; only used if popular_per_layer)
    _ = None
    freq: dict[tuple[int, int], int] = defaultdict(int)
    for _, key in stream:
        freq[key] += 1
    pinned_pop: dict[int, set] = defaultdict(set)
    if popular_per_layer:
        per_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (lid, eid), c in freq.items():
            if eid == -1:
                continue
            per_layer[lid].append(((lid, eid), c))
        for lid, items in per_layer.items():
            items.sort(key=lambda x: -x[1])
            pinned_pop[lid] = {k for k, _ in items[:popular_per_layer]}

    # pinned_pop keyed by layer id; each GPU cache gets only its own layers.
    def pop_set(gpu_layers: range) -> set:
        s: set = set()
        for lid in gpu_layers:
            s |= pinned_pop.get(lid, set())
        return s

    c0 = GpuCache(budget, pin_shared=pin_shared,
                  pinned_popular=pop_set(range(0, split)))
    c1 = GpuCache(budget, pin_shared=pin_shared,
                  pinned_popular=pop_set(range(split, split + 100)))
    if lfu:
        # LFU among unpinned: override scoring by using a negative-frequency
        # key trick in GpuCache._score via a flag.
        for c in (c0, c1):
            c.lfu = True
    prov0 = ProviderLru(provider_per_layer)
    prov1 = ProviderLru(provider_per_layer)
    gpu_hits = gpu_misses = prov_hits = http = 0
    for _, key in stream:
        lid, eid = key
        gpu = c0 if lid < split else c1
        prov = prov0 if lid < split else prov1
        if gpu.access(key):
            gpu_hits += 1
        else:
            gpu_misses += 1
            if prov.access(lid, eid):
                prov_hits += 1
            else:
                http += 1
    total = len(stream)
    print(f"{label:40s}: gpu_hit={100.0*gpu_hits/total:5.2f}%  "
          f"provider_hit={100.0*prov_hits/total:5.2f}%  "
          f"http_fetch={http} ({100.0*http/total:4.1f}%)")
    return {"gpu_hit_pct": 100.0 * gpu_hits / total,
            "provider_hit_pct": 100.0 * prov_hits / total,
            "http": http, "http_pct": 100.0 * http / total}


def belady_per_gpu(stream: list, split: int, capacity: int) -> tuple[float, float]:
    results = []
    for lo, hi in ((0, split), (split, 10 ** 9)):
        sub = [(tok, key) for tok, key in stream if lo <= key[0] < hi]
        nxt = defaultdict(list)
        for i, (_, key) in enumerate(sub):
            nxt[key].append(i)
        for k in nxt:
            nxt[k].append(10 ** 9)
        ptr = {k: 0 for k in nxt}
        cache: dict = {}
        hits = 0
        for i, (_, key) in enumerate(sub):
            if key in cache:
                hits += 1
            else:
                if len(cache) >= capacity:
                    victim = max(cache, key=lambda k: nxt[k][ptr[k]])
                    del cache[victim]
                cache[key] = i
            ptr[key] += 1
        results.append(100.0 * hits / max(1, len(sub)))
    return tuple(results)  # type: ignore[return-value]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    stream, n_tokens, n_layers = build_stream(d)
    split = (n_layers + 1) // 2
    print(f"tokens={n_tokens} layers={n_layers} split={split} "
          f"(GPU0={split} layers, GPU1={n_layers-split})  "
          f"accesses(+shared)={len(stream)}")

    print("\n== GPU-cache + provider two-level policy matrix ==")
    # DS10 as-run: 2 GiB GPU LRU, NO provider retention (raw=0)
    run_policy(stream, split, label="ds10_current(2GiB, prov_raw=0)",
               budget=2 << 30, pin_shared=False, provider_per_layer=0)
    # provider retention only (still 2 GiB GPU LRU)
    run_policy(stream, split, label="provider_raw8 (2GiB GPU LRU)",
               budget=2 << 30, pin_shared=False, provider_per_layer=8)
    run_policy(stream, split, label="provider_raw16 (2GiB GPU LRU)",
               budget=2 << 30, pin_shared=False, provider_per_layer=16)
    # pin shared + provider
    run_policy(stream, split, label="pin_shared+prov8 (2GiB)",
               budget=2 << 30, pin_shared=True, provider_per_layer=8)
    run_policy(stream, split, label="pin_shared+prov8 (4GiB)",
               budget=4 << 30, pin_shared=True, provider_per_layer=8)
    run_policy(stream, split, label="pin_shared+prov16 (8GiB)",
               budget=8 << 30, pin_shared=True, provider_per_layer=16)
    # eviction-policy variants at 2 GiB (capacity-bound regime)
    run_policy(stream, split, label="pin_shared+LFU+prov8 (2GiB)",
               budget=2 << 30, pin_shared=True, provider_per_layer=8, lfu=True)
    run_policy(stream, split, label="pin_shared+LFU+prov8 (4GiB)",
               budget=4 << 30, pin_shared=True, provider_per_layer=8, lfu=True)
    run_policy(stream, split, label="pin_shared+LFU+prov16 (8GiB)",
               budget=8 << 30, pin_shared=True, provider_per_layer=16, lfu=True)
    # popularity pinning (oracle-informed)
    run_policy(stream, split, label="pin_shared+pop2+prov8 (4GiB)",
               budget=4 << 30, pin_shared=True, provider_per_layer=8,
               popular_per_layer=2)
    run_policy(stream, split, label="pin_shared+pop4+prov8 (8GiB)",
               budget=8 << 30, pin_shared=True, provider_per_layer=8,
               popular_per_layer=4)

    print("\n== Belady (oracle) GPU-cache ceilings per GPU ==")
    for budget_gib in (2, 4, 8, 12):
        cap = int(budget_gib << 30) // EXPERT_BYTES
        h0, h1 = belady_per_gpu(stream, split, cap)
        print(f"  {budget_gib:2d} GiB (~{cap:3d} experts/GPU): "
              f"GPU0={h0:5.2f}%  GPU1={h1:5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
