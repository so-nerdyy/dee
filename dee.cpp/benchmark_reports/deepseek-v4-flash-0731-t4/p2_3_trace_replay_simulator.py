#!/usr/bin/env python3
"""P2.3 — Trace-replay cache simulator: packed FP4 VRAM residency vs FP16.

Replays the sealed DS10 v12 `token_trace` (16 tokens x 43 layers x top-6
routed experts) through an LRU VRAM expert cache and reports, for both the
expanded FP16 and packed FP4 representations:

  - resident experts at a given budget
  - routed hit rate (GPU cache hits)
  - per-GPU and aggregate routed hit rate
  - reuse-distance coverage (accesses whose distance fits in the cache)
  - packed H2D bytes/token
  - evictions/token
  - Belady (oracle) hit ceiling

This answers the P2.3 research question quantitatively: does 3.76x packed
capacity cross the working-set threshold enough to capture the ~37.5%
consecutive-token reuse, or is the access pattern too spread out for LRU at
any affordable budget?

Trace provenance: benchmark_reports/deepseek-v4-flash-0731-t4/
  ds10-v12-accept-dual-t4-decode/evidence/ds10-evidence.json  (gates/token_trace)
"""

import json
import os
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Sizes (real DeepSeek-V4-Flash-0731 expert: gate/up [4096,2048], down
# [2048,4096], e2m1fn packed I8 + e8m0 per-block scales).
# ---------------------------------------------------------------------------
FP16_EXPERT_MIB = 48.00
FP4_EXPERT_MIB = 12.75          # 12.00 MiB packed + 0.75 MiB scales
FP4_TRANSFER_MIB = 12.75        # both cache modes transfer packed FP4
MIB = 1 << 20
LAYER_SPLIT = 22                # cuda0 owns 0..21; cuda1 owns 22..42
V44_H2D_BYTES = 65_509_785_600

EV_PATH = os.path.join(
    os.path.dirname(__file__),
    "ds10-v12-accept-dual-t4-decode", "evidence", "ds10-evidence.json")


def load_trace():
    """Flatten token_trace into (token, layer, [expert,...]) per layer-call.

    ``selected_experts`` is a list of per-position top-6 lists: prefill
    token_0 has 7 positions (7 routed MoE calls for that layer), decode
    tokens have 1 position each.  Each position contributes its 6 experts.
    """
    with open(EV_PATH, encoding="utf-8") as f:
        ev = json.load(f)
    trace = ev["gates"]["token_trace"]
    tokens = []
    for t in sorted(trace.keys(), key=lambda s: int(s.split("_")[1])):
        layers = trace[t]["layers"]
        for ly in layers:
            sel = ly["selected_experts"]
            if sel and isinstance(sel[0], list):
                flattened = [int(e) for pos in sel for e in pos]
            else:
                flattened = [int(e) for e in sel]
            # The engine groups all positions by expert and stages each unique
            # expert once per layer call. Preserve first-seen rank order.
            experts = list(dict.fromkeys(flattened))
            tokens.append((int(t.split("_")[1]), int(ly["layer"]), experts))
    return tokens


class LRUCache:
    """Simple LRU with O(1) touch/evict and hit accounting."""

    def __init__(self, capacity_experts: int):
        self.cap = capacity_experts
        self.map = OrderedDict()          # key -> None (order = recency)
        self.hits = 0
        self.evictions = 0
        self.total = 0

    def access(self, key):
        self.total += 1
        if key in self.map:
            self.map.move_to_end(key)
            self.hits += 1
            return True
        if len(self.map) >= self.cap and self.cap > 0:
            self.map.popitem(last=False)
            self.evictions += 1
        self.map[key] = None
        return False


class BeladyCache:
    """Oracle (Belady MIN): evict the resident key whose NEXT use is farthest."""

    def __init__(self, capacity_experts: int, future_uses):
        self.cap = capacity_experts
        self.future = future_uses          # list of lists: next-use index per key
        self.map = {}                       # key -> next use index (None = never)
        self.hits = 0
        self.total = 0
        self.evictions = 0

    def access(self, key, pos):
        self.total += 1
        if key in self.map:
            self.hits += 1
            self.map[key] = self._next_use(key, pos)
            return True
        if len(self.map) >= self.cap and self.cap > 0:
            # evict key with the farthest (or never) next use
            victim = max(self.map, key=lambda k: (
                self.map[k] if self.map[k] is not None else 10**12))
            del self.map[victim]
            self.evictions += 1
        self.map[key] = self._next_use(key, pos)
        return False

    def _next_use(self, key, pos):
        nxt = self.future.get(key, [])
        for n in nxt:
            if n > pos:
                return n
        return None


def reuse_distance_coverage(accesses, capacity):
    """Fraction of accesses whose previous occurrence is within `capacity`
    distinct-key steps — the reuse LRU *can* capture."""
    stack = OrderedDict()
    covered = 0
    for key in accesses:
        if key in stack:
            ordered = list(stack.keys())
            distance = len(ordered) - 1 - ordered.index(key)
            if distance < capacity:
                covered += 1
            stack.move_to_end(key)
        else:
            stack[key] = None
    return covered / max(1, len(accesses))


def simulate(budget_gib, expert_mib, trace):
    cap = int(budget_gib * 1024 / expert_mib)
    if cap <= 0:
        cap = 1
    caches = [LRUCache(cap), LRUCache(cap)]
    for _token, layer, experts in trace:
        gpu = 0 if layer < LAYER_SPLIT else 1
        for expert in experts:
            caches[gpu].access((layer, expert))
    hits = sum(cache.hits for cache in caches)
    total = sum(cache.total for cache in caches)
    return {
        "budget_gib": budget_gib,
        "expert_mib": expert_mib,
        "residents_per_gpu": cap,
        "hits": hits,
        "total_hits": hits,
        "total": total,
        "hit_rate": hits / max(1, total),
        "cold_loads": total - hits,
        "evictions": sum(cache.evictions for cache in caches),
        "per_gpu": [
            {
                "requests": cache.total,
                "hits": cache.hits,
                "cold_loads": cache.total - cache.hits,
                "evictions": cache.evictions,
                "resident_experts": len(cache.map),
            }
            for cache in caches
        ],
    }


def belady(budget_gib, expert_mib, trace):
    cap = int(budget_gib * 1024 / expert_mib)
    if cap <= 0:
        cap = 1
    streams = [[], []]
    for _token, layer, experts in trace:
        gpu = 0 if layer < LAYER_SPLIT else 1
        streams[gpu].extend((layer, expert) for expert in experts)
    hits = 0
    total = 0
    for stream in streams:
        future_uses = {}
        for pos, key in enumerate(stream):
            future_uses.setdefault(key, []).append(pos)
        cache = BeladyCache(cap, future_uses)
        for pos, key in enumerate(stream):
            hits += int(cache.access(key, pos))
        total += len(stream)
    return hits / max(1, total)


def main():
    if not os.path.exists(EV_PATH):
        print(f"sealed DS10 evidence not found at {EV_PATH}; run the v15/v16 "
              f"generation first or point EV_PATH at a local trace dump")
        return 1
    trace = load_trace()
    print(f"=== P2.3 packed-FP4 trace replay ({len(trace)} layer-calls, "
          f"{len(set((t, l) for t, l, _ in trace))} token-layer pairs) ===")

    streams = [[], []]
    for _token, layer, experts in trace:
        gpu = 0 if layer < LAYER_SPLIT else 1
        streams[gpu].extend((layer, expert) for expert in experts)
    total = sum(len(stream) for stream in streams)
    print(f"unique staged requests={total} "
          f"(cuda0={len(streams[0])}, cuda1={len(streams[1])}); "
          "shared experts excluded (separate persistent path)")

    for budget_gib in (1.0, 2.0, 4.0, 6.0, 8.0):
        row = []
        for label, mib in (("FP16", FP16_EXPERT_MIB), ("FP4 ", FP4_EXPERT_MIB)):
            r = simulate(budget_gib, mib, trace)
            row.append((label, r))
        fp16 = row[0][1]
        fp4 = row[1][1]
        print(f"\n--- budget {budget_gib:.0f} GiB ---")
        print(f"  FP16: {fp16['residents_per_gpu']:>4} residents/GPU  "
              f"routed hit={fp16['hit_rate']*100:5.1f}%  "
              f"evicts={fp16['evictions']}")
        print(f"  FP4 : {fp4['residents_per_gpu']:>4} residents/GPU  "
              f"routed hit={fp4['hit_rate']*100:5.1f}%  "
              f"evicts={fp4['evictions']}")
        bel = belady(budget_gib, FP4_EXPERT_MIB, trace)
        print(f"  Belady ceiling (FP4 size): {bel*100:5.1f}%")

    # Bytes per token at 3.5 GiB (the v15/v16 primary budget)
    print("\n=== bytes/token at 3.5 GiB (v15/v16 primary) ===")
    for label, mib in (("FP16", FP16_EXPERT_MIB), ("FP4 ", FP4_EXPERT_MIB)):
        r = simulate(3.5, mib, trace)
        h2d_gib = (r["cold_loads"] / 16.0 * FP4_TRANSFER_MIB
                   * MIB / (1 << 30))
        print(f"  {label}: routed hit={r['hit_rate']*100:5.1f}%  "
              f"packed H2D={h2d_gib:.2f} GiB/token  "
              f"cold={r['cold_loads']}")

    replay_v44 = simulate(3.5, FP16_EXPERT_MIB, trace)
    observed = {
        "hits": [104, 95],
        "cold_loads": [2509, 2391],
    }
    print("\n=== v44 replay cross-check ===")
    for gpu, row in enumerate(replay_v44["per_gpu"]):
        print(f"  cuda{gpu}: replay hits/cold={row['hits']}/{row['cold_loads']} "
              f"observed={observed['hits'][gpu]}/{observed['cold_loads'][gpu]}")
    print(f"  measured v44 H2D={V44_H2D_BYTES / 16 / (1 << 30):.3f} "
          "GiB/emitted token (authoritative; replay is directional)")

    # Reuse-distance coverage: how much reuse is even capturable?
    print("\n=== reuse-distance coverage (LRU-capturable reuse) ===")
    for budget_gib in (1.0, 2.0, 4.0, 6.0, 8.0):
        for label, mib in (("FP16", FP16_EXPERT_MIB), ("FP4 ", FP4_EXPERT_MIB)):
            cap = int(budget_gib * 1024 / mib)
            covered = sum(
                reuse_distance_coverage(stream, cap) * len(stream)
                for stream in streams)
            cov = covered / max(1, total)
            print(f"  {budget_gib:>3.0f} GiB {label}: coverage={cov*100:5.1f}%")

    print("\n=== conclusion ===")
    print("Belady ceiling at 8 GiB is the hard reuse bound; LRU approaches it "
          "only when the working set fits. Compare FP4 vs FP16 at the same "
          "budget: FP4 gains 3.76x capacity, capturing more of the structural "
          "37.5% consecutive-token reuse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
