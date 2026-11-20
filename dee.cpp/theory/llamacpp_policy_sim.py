#!/usr/bin/env python3
"""Score eviction policies for llama.cpp's MoE cache (--moe-cache-mib) on recorded routing.

The cache under test is llama.cpp's own: ClassLRU is imported unchanged from
llama.cpp's scripts/moe-cache-tune.py (its Python mirror of src/llama-moe-cache.cpp),
so its numbers are upstream's, not a re-implementation.  The other policies use the
same single slot pool shared by all cached layers, the same per-lookup rule (the
experts selected by the current lookup cannot be evicted), and the same skip of
ubatches above 32 tokens.

Policies
  lru       plain least recently used (llama.cpp's first policy, PR #29887)
  classlru  upstream ClassLRU (PR #31233), imported from moe-cache-tune.py
  next_use  DEE's next-use estimator (include/dee/next_use.h), cycle = cached layers
  belady    offline optimum: evict the expert used furthest in the future

Inputs
  --dee-journal   DEE route journals (routed_experts*.jsonl), concatenated in order,
                  as one long session with requests back to back
  --mct           llama.cpp traces written with LLAMA_MOE_CACHE_TRACE (MCT1 format)

Metric: hit rate = hits / expert lookups, over lookups of ubatches the cache serves,
after --skip-tokens tokens of warm-up.  Misses are expert uploads, the cost the
cache exists to cut.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

MAX_BATCH = 32  # llama_moe_cache::impl::max_batch

Record = Tuple[int, int, Tuple[int, ...]]  # layer, experts per token, ids (n_tokens * n_used)


# ---------------------------------------------------------------- traces

def read_dee_journals(paths: Sequence[str]) -> Tuple[List[int], int, int, List[Record]]:
    records: List[Record] = []
    layers, n_expert, n_used = set(), 0, 0
    for p in paths:
        rows = [json.loads(line) for line in open(p)]
        rows.sort(key=lambda r: (r["forward_step"], r["layer"]))
        for r in rows:
            toks = r["expert_ids_rank_order"]
            n_used = len(toks[0])
            ids = tuple(int(e) for t in toks for e in t)
            n_expert = max(n_expert, max(ids) + 1)
            layers.add(int(r["layer"]))
            records.append((int(r["layer"]), n_used, ids))
    n_expert = 1 << max(0, (n_expert - 1).bit_length())  # 256 for DSv4
    return sorted(layers), n_expert, n_used, records


def read_mct(paths: Sequence[str], tune) -> Tuple[List[int], int, int, List[Record]]:
    n_layer, n_used, groups, records = tune.concat_traces(list(paths))
    if len(groups) != 1:
        sys.exit("only single-group (one GPU) traces are supported")
    g = groups[0]
    return list(g.layers), g.n_expert, n_used, [(il, nu, tuple(ids)) for il, nu, ids in records]


# ---------------------------------------------------------------- policies

class SlotPool:
    """A pool of n_slots shared by all cached layers, keys = (layer, expert)."""

    def __init__(self, n_slots: int) -> None:
        self.n_slots = n_slots
        self.slot_of: Dict[Tuple[int, int], int] = {}
        self.key_of: List[Tuple[int, int] | None] = [None] * n_slots
        self.free = list(range(n_slots - 1, -1, -1))

    def evict_candidates(self, protected: set) -> np.ndarray:
        """mask of slots that may be evicted now"""
        mask = np.ones(self.n_slots, dtype=bool)
        for k in protected:
            s = self.slot_of.get(k)
            if s is not None:
                mask[s] = False
        return mask


class LRU(SlotPool):
    name = "lru"

    def __init__(self, n_slots: int, **_) -> None:
        super().__init__(n_slots)
        self.last = np.full(n_slots, -1, dtype=np.int64)
        self.tick = 0

    def begin(self, il: int, t: int) -> None:
        pass

    def touch(self, key, s: int) -> None:
        self.tick += 1
        self.last[s] = self.tick

    def victim(self, protected: set, t: int) -> int:
        mask = self.evict_candidates(protected)
        score = np.where(mask, self.last, np.iinfo(np.int64).max)
        return int(np.argmin(score))


class NextUse(SlotPool):
    """Mirror of dee::NextUseEstimator (include/dee/next_use.h)."""
    name = "next_use"

    def __init__(self, n_slots: int, layers: Sequence[int], alpha: float = 0.2,
                 p0: float = 0.1, p_floor: float = 1e-3, **_) -> None:
        super().__init__(n_slots)
        self.pos = {il: i for i, il in enumerate(layers)}  # cycle position of a layer
        self.cycle = len(layers)
        self.alpha, self.p0, self.p_floor = alpha, p0, p_floor
        self.visits = np.zeros(self.cycle, dtype=np.int64)
        self.cur = 0
        self.ewma: Dict[Tuple[int, int], float] = {}
        self.last_visit: Dict[Tuple[int, int], int] = {}
        # per-slot copies for vectorised ranking
        self.s_pos = np.zeros(n_slots, dtype=np.int64)
        self.s_ewma = np.zeros(n_slots)
        self.s_last = np.zeros(n_slots, dtype=np.int64)

    def begin(self, il: int, t: int) -> None:
        self.cur = self.pos[il]
        self.visits[self.cur] += 1

    def touch(self, key, s: int) -> None:
        v = int(self.visits[self.pos[key[0]]])
        if key in self.ewma:
            gap = v - self.last_visit[key]
            if gap > 0:
                self.ewma[key] = self.ewma[key] * (1.0 - self.alpha) ** gap + self.alpha
                self.last_visit[key] = v
        else:
            self.ewma[key] = self.p0 * (1.0 - self.alpha) + self.alpha
            self.last_visit[key] = v
        self.s_pos[s] = self.pos[key[0]]
        self.s_ewma[s] = self.ewma[key]
        self.s_last[s] = self.last_visit[key]

    def victim(self, protected: set, t: int) -> int:
        mask = self.evict_candidates(protected)
        elapsed = self.visits[self.s_pos] - self.s_last
        elapsed = elapsed - (self.s_pos == self.cur)  # visit in progress is not a known miss
        p = self.s_ewma * np.power(1.0 - self.alpha, np.maximum(elapsed, 0))
        p = np.clip(p, self.p_floor, 1.0)
        d = (self.s_pos - self.cur) % self.cycle
        rank = d + self.cycle * (1.0 / p - 1.0)
        rank = np.where(self.s_pos == self.cur, -1.0, rank)  # running layer leaves last
        rank = np.where(mask, rank, -np.inf)
        return int(np.argmax(rank))


class Belady(SlotPool):
    """Offline optimum; needs the next-use time of each lookup (set by the driver)."""
    name = "belady"

    def __init__(self, n_slots: int, **_) -> None:
        super().__init__(n_slots)
        self.nxt = np.zeros(n_slots, dtype=np.int64)
        self.next_of: Dict[Tuple[int, int], int] = {}

    def begin(self, il: int, t: int) -> None:
        pass

    def touch(self, key, s: int) -> None:
        self.nxt[s] = self.next_of[key]

    def victim(self, protected: set, t: int) -> int:
        mask = self.evict_candidates(protected)
        return int(np.argmax(np.where(mask, self.nxt, -1)))


def run_pool(policy, records: List[Record], skip_tokens: int, first_layer: int,
             next_use_times: List[Dict] | None = None) -> Tuple[int, int]:
    hits = lookups = 0
    seen = 0
    for t, (il, n_used, ids) in enumerate(records):
        n_tokens = len(ids) // n_used
        if il == first_layer:
            seen += n_tokens
        uniq = list(dict.fromkeys(ids))
        if n_tokens > MAX_BATCH or len(uniq) > policy.n_slots:
            continue
        count = seen > skip_tokens
        policy.begin(il, t)
        keys = [(il, e) for e in uniq]
        protected = set(keys)
        if next_use_times is not None:
            policy.next_of.update(next_use_times[t])
        misses = []
        for k in keys:
            s = policy.slot_of.get(k)
            if s is not None:
                policy.touch(k, s)
                if count:
                    hits += 1
            else:
                misses.append(k)
        for k in sorted(misses):
            if policy.free:
                s = policy.free.pop()
            else:
                s = policy.victim(protected, t)
                del policy.slot_of[policy.key_of[s]]
            policy.slot_of[k] = s
            policy.key_of[s] = k
            policy.touch(k, s)
        if count:
            lookups += len(keys)
    return hits, lookups


def belady_times(records: List[Record]) -> List[Dict]:
    """for each record, the index of the next record that selects each of its experts"""
    out: List[Dict] = [dict() for _ in records]
    nxt: Dict[Tuple[int, int], int] = {}
    for t in range(len(records) - 1, -1, -1):
        il, n_used, ids = records[t]
        d = {}
        for e in set(ids):
            k = (il, e)
            d[k] = nxt.get(k, 1 << 60)
            nxt[k] = t
        out[t] = d
    return out


def run_classlru(tune, layers, n_expert, n_slots, records, skip_tokens, first_layer, policy="classlru"):
    c = tune.ClassLRU(layers, n_expert, n_slots, policy) if policy != "classlru" else tune.ClassLRU(layers, n_expert, n_slots)
    hits = lookups = seen = 0
    for il, n_used, ids in records:
        n_tokens = len(ids) // n_used
        if il == first_layer:
            seen += n_tokens
        n_uniq = len(set(ids))
        if n_tokens > MAX_BATCH or min(n_tokens * n_used, n_expert) > n_slots:
            continue
        n_fill, _ = c.plan(il, list(ids), n_used, 0, lambda n: 0)
        if seen > skip_tokens:
            hits += n_uniq - n_fill
            lookups += n_uniq
    return hits, lookups


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", required=True, help="path to llama.cpp scripts/moe-cache-tune.py")
    ap.add_argument("--dee-journal", nargs="*", default=[])
    ap.add_argument("--mct", nargs="*", default=[])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
    ap.add_argument("--skip-tokens", type=int, default=0)
    ap.add_argument("--policies", nargs="+", default=["lru", "classlru", "next_use", "belady"])
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("moe_cache_tune", args.tune)
    tune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune)

    if args.dee_journal:
        layers, n_expert, n_used, records = read_dee_journals(args.dee_journal)
    else:
        layers, n_expert, n_used, records = read_mct(args.mct, tune)
    first = layers[0]
    total = len(layers) * n_expert
    n_tok = sum(len(ids) // nu for il, nu, ids in records if il == first)
    print(f"{len(records)} lookups, {len(layers)} cached layers, {n_expert} experts, top-{n_used}, "
          f"{n_tok} tokens, {total} experts in all")

    bt = belady_times(records) if "belady" in args.policies else None
    results = []
    hdr = "cache %   slots  " + "  ".join(f"{p:>9}" for p in args.policies)
    print(hdr)
    for f in args.fractions:
        n_slots = max(n_used, int(round(f * total)))
        row = {"fraction": f, "slots": n_slots}
        for p in args.policies:
            if p == "classlru":
                h, n = run_classlru(tune, layers, n_expert, n_slots, records, args.skip_tokens, first)
            elif p == "lru":
                h, n = run_pool(LRU(n_slots), records, args.skip_tokens, first)
            elif p == "upstream_next_use":  # the mirror in the patched moe-cache-tune.py
                h, n = run_classlru(tune, layers, n_expert, n_slots, records, args.skip_tokens, first, "next_use")
            elif p == "next_use":
                h, n = run_pool(NextUse(n_slots, layers), records, args.skip_tokens, first)
            elif p == "belady":
                h, n = run_pool(Belady(n_slots), records, args.skip_tokens, first, bt)
            else:
                sys.exit(f"unknown policy {p}")
            row[p] = h / max(n, 1)
        results.append(row)
        print(f"{100*f:6.1f}  {n_slots:6d}  " + "  ".join(f"{row[p]:9.4f}" for p in args.policies), flush=True)
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
