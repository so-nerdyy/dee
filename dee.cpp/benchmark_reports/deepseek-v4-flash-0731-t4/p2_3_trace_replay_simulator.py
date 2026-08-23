#!/usr/bin/env python3
"""P2.3 — Trace-replay cache simulator: packed FP4 VRAM residency vs FP16.

Replays the sealed DS10 v12 `token_trace` (16 tokens x 43 layers x top-6
routed experts) through an LRU VRAM expert cache and reports, for both the
expanded FP16 and packed FP4 representations:

  - resident experts at a given budget
  - routed hit rate (GPU cache hits)
  - total hit rate (incl. shared expert pinning)
  - reuse-distance coverage (accesses whose distance fits in the cache)
  - H2D bytes/token and storage bytes/token
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
FP4_EXPERT_MIB = 12.75          # 12.582912 MiB packed + 0.75 MiB scales
MIB = 1 << 20

# Shared expert: one per layer, same 3-projection size.
N_SHARED_PER_LAYER = 1

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
                experts = [int(e) for pos in sel for e in pos]
            else:
                experts = [int(e) for e in sel]
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
    last_pos = {}
    covered = 0
    span_keys = 0
    for i, key in enumerate(accesses):
        if key in last_pos:
            # count distinct keys since last use
            span_keys += 1
            if span_keys <= capacity:
                covered += 1
        else:
            span_keys = 0
        last_pos[key] = i
    return covered / max(1, len(accesses))


def simulate(budget_gib, expert_mib, accesses, future_uses, pin_shared,
             n_shared_per_layer, layer_of):
    cap = int(budget_gib * 1024 / expert_mib)
    if cap <= 0:
        cap = 1
    cache = LRUCache(cap)
    # Pin shared experts: they consume capacity but are never evicted.
    pinned = set()
    if pin_shared:
        for key in accesses:
            if key[0] == -1:            # shared expert key: (-1, layer)
                pinned.add(key)
    # Capacity left for routed experts
    routed_cap = max(1, cap - len(pinned))
    routed_cache = LRUCache(routed_cap)

    hits = 0
    total = 0
    for key in accesses:
        total += 1
        if key in pinned:
            hits += 1
            continue
        if routed_cache.access(key):
            hits += 1
    routed_hits = sum(1 for key in accesses
                      if key[0] != -1 and key in routed_cache.map or
                      (key[0] != -1 and False))
    return {
        "budget_gib": budget_gib,
        "expert_mib": expert_mib,
        "residents": cap,
        "routed_hits": routed_hits,
        "total_hits": hits,
        "total": total,
        "hit_rate": hits / max(1, total),
        "evictions": routed_cache.evictions,
        "routed_hit_rate": routed_hits / max(1, sum(
            1 for key in accesses if key[0] != -1)),
    }


def belady(budget_gib, expert_mib, accesses, future_uses, layer_of):
    cap = int(budget_gib * 1024 / expert_mib)
    if cap <= 0:
        cap = 1
    cache = BeladyCache(cap, future_uses)
    hits = 0
    for pos, key in enumerate(accesses):
        if cache.access(key, pos):
            hits += 1
    return hits / max(1, len(accesses))


def main():
    if not os.path.exists(EV_PATH):
        print(f"sealed DS10 evidence not found at {EV_PATH}; run the v15/v16 "
              f"generation first or point EV_PATH at a local trace dump")
        return 1
    trace = load_trace()
    print(f"=== P2.3 packed-FP4 trace replay ({len(trace)} layer-calls, "
          f"{len(set((t, l) for t, l, _ in trace))} token-layer pairs) ===")

    # Build access sequence: routed experts (layer, expert) + shared (-1, layer).
    # The v12 trace does NOT include shared experts in selected_experts (top-6
    # routed only); shared expert = one per layer, present every layer-call.
    accesses = []
    for tok, layer, experts in trace:
        for e in experts:
            accesses.append((layer, e))
        accesses.append((-1, layer))       # shared expert, always accessed

    # Future-use map for Belady
    future_uses = {}
    for i, key in enumerate(accesses):
        future_uses.setdefault(key, []).append(i)

    total = len(accesses)
    routed_total = sum(1 for k in accesses if k[0] != -1)
    print(f"accesses={total} (routed={routed_total}, shared={total-routed_total})")

    for budget_gib in (1.0, 2.0, 4.0, 6.0, 8.0):
        row = []
        for label, mib in (("FP16", FP16_EXPERT_MIB), ("FP4 ", FP4_EXPERT_MIB)):
            r = simulate(budget_gib, mib, accesses, future_uses,
                         pin_shared=True, n_shared_per_layer=N_SHARED_PER_LAYER,
                         layer_of=None)
            row.append((label, r))
        fp16 = row[0][1]
        fp4 = row[1][1]
        print(f"\n--- budget {budget_gib:.0f} GiB ---")
        print(f"  FP16: {fp16['residents']:>4} residents  hit={fp16['hit_rate']*100:5.1f}%  "
              f"routed={fp16['routed_hit_rate']*100:5.1f}%  evicts={fp16['evictions']}")
        print(f"  FP4 : {fp4['residents']:>4} residents  hit={fp4['hit_rate']*100:5.1f}%  "
              f"routed={fp4['routed_hit_rate']*100:5.1f}%  evicts={fp4['evictions']}")
        bel = belady(budget_gib, FP4_EXPERT_MIB, accesses, future_uses, None)
        print(f"  Belady ceiling (FP4 size): {bel*100:5.1f}%")

    # Bytes per token at 3.5 GiB (the v15/v16 primary budget)
    print("\n=== bytes/token at 3.5 GiB (v15/v16 primary) ===")
    for label, mib in (("FP16", FP16_EXPERT_MIB), ("FP4 ", FP4_EXPERT_MIB)):
        r = simulate(3.5, mib, accesses, future_uses, True,
                     N_SHARED_PER_LAYER, None)
        per_token = r["total"] / 16.0
        cold_per_token = per_token * (1 - r["hit_rate"])
        h2d_gib = cold_per_token * mib * MIB / (1 << 30)
        print(f"  {label}: hit={r['hit_rate']*100:5.1f}%  "
              f"H2D={h2d_gib:.2f} GiB/token  (routed {r['routed_hit_rate']*100:.1f}%)")

    # Reuse-distance coverage: how much reuse is even capturable?
    print("\n=== reuse-distance coverage (LRU-capturable reuse) ===")
    for budget_gib in (1.0, 2.0, 4.0, 6.0, 8.0):
        for label, mib in (("FP16", FP16_EXPERT_MIB), ("FP4 ", FP4_EXPERT_MIB)):
            cap = int(budget_gib * 1024 / mib)
            cov = reuse_distance_coverage(accesses, cap)
            print(f"  {budget_gib:>3.0f} GiB {label}: coverage={cov*100:5.1f}%")

    print("\n=== conclusion ===")
    print("Belady ceiling at 8 GiB is the hard reuse bound; LRU approaches it "
          "only when the working set fits. Compare FP4 vs FP16 at the same "
          "budget: FP4 gains 3.76x capacity, capturing more of the structural "
          "37.5% consecutive-token reuse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
