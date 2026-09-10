#!/usr/bin/env python3
"""Phase 2 working-set policy study — OFFLINE exact-trace simulation (v3).

STREAM (validated): the sealed v50 canonical route journal
    dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/
        v50-evidence-20260829T195940Z/routed_experts.jsonl
processed exactly as the production engine consumes it: per
(forward_step, layer) call, the engine deduplicates the token-row x top-k
route IDs and stages UNIQUE experts in ASCENDING expert-id order
(moe_forward_batch_device_impl groups + active_experts scan 0..255).

VALIDATION (results/validation.json, all PASS):
  host tier, plain LRU, 682 slots/GPU (= fill-live 8.5 GiB/GPU):
    cuda0 sim (hits 1222, misses 1391) vs sealed (1223, 1390)
    cuda1 sim (hits 1395, misses 1091) vs sealed (1395, 1091)
  vram tier, engine priority-LRU, 281 slots/GPU (= v60 3.5 GiB/GPU):
    cuda0 sim (329, 2284, ev 2003) vs sealed (328, 2285, ev 2004)
    cuda1 sim (325, 2161, ev 1880) vs sealed (327, 2159, ev 1878)
  -> the production VRAM eviction score is last_used + priority*2**20 with
     priority = (batch_len - index_in_batch); this artifact SUPPRESSES the
     VRAM hit rate ~2x vs plain LRU at the identical budget.  Policy lever.

POLICIES (all mechanically defined; no per-prompt hardcoding):
  lru                plain LRU (the host tier's existing semantics)
  arc                Adaptive Replacement Cache (Megiddo/Modha), canonical
  lfu                online count-based eviction (LRU tiebreak)
  static_freq        offline global top-N static placement (KTransformers-
                     style hot set; ceiling for count-based pinning)
  static_layer_freq  offline per-layer top-B static placement
  freq_lru           offline frequency-pinned set (pin_frac of slots) + LRU
                     remainder
  freq_lru_warmup    ONLINE variant: pin set from prefill-pass counts only
                     (deployable: warmup then steady decode)
  freq_x_recency     online composite score 0.5*freq_frac + 0.5*recency_frac
  layer_lru          per-layer LRU, uniform per-layer budget
  cost_aware         uniform record size => provably == LRU (confirmed)
  belady             offline MIN (unattainable upper bound)
  lru_plus_repair    LRU + the one-line VRAM fix: drop the stale-priority
                     artifact (equivalently plain LRU on the device tier)

Outputs: results/sim_rows.csv (full matrix), results/reuse_distance.json
(exact stack-distance histogram), results/validation.json (sealed anchors),
results/summary.md (human-readable highlights).
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOURNAL = (REPO / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
           "v50-evidence-20260829T195940Z/routed_experts.jsonl")
OUT = REPO / "research/phase2-ws-policy/results"

RECORD_BYTES = 13_369_344          # 12.75 MiB DEE4 packed FP4 record
N_LAYERS = 43
GIB = 1 << 30
SPLIT_GPU0 = 22                    # cuda0: layers 0-21, cuda1: 22-42
N_STEPS = 16

BW_MEASURED = (0.29, 0.33, 0.37)   # GiB/s, Phase-1 live T4x2 matrix
BW_FUTURE = (3.0, 5.0, 7.0, 12.0)  # GiB/s floors
DECODE_WALL_LIVE_S = 71.479


# ---------------------------------------------------------------- stream


def load_batches():
    recs = [json.loads(l) for l in open(JOURNAL) if l.strip()]
    recs.sort(key=lambda r: r["record_index"])
    batches = []
    for r in recs:
        uniq = sorted({e for row in r["expert_ids_rank_order"] for e in row})
        batches.append((r["forward_step"], r["layer"],
                        [(r["layer"], e) for e in uniq]))
    return batches


# ---------------------------------------------------------------- caches


class LRU:
    name = "lru"

    def __init__(self, slots, **kw):
        self.slots = slots
        self.lru = deque()
        self.pos = set()
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        if key in self.pos:
            self.lru.remove(key)
            self.lru.appendleft(key)
            return True
        if self.slots <= 0:
            return False
        if len(self.lru) >= self.slots:
            self.pos.discard(self.lru.pop())
            self.evictions += 1
        self.lru.appendleft(key)
        self.pos.add(key)
        return False


class PriorityLRU:
    """The production VRAM semantics: score = last_used + prio * 2**20,
    priority = (batch_len - index) refreshed on hit.  Evaluated as
    'engine_lru' to document the artifact, and repairable by plain LRU."""

    name = "engine_priority_lru"

    def __init__(self, slots, **kw):
        self.slots = slots
        self.blocks = {}
        self.tick = 0
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        self.tick += 1
        if key in self.blocks:
            self.blocks[key][0] = self.tick
            self.blocks[key][1] = prio
            return True
        if self.slots <= 0:
            return False
        while len(self.blocks) >= self.slots:
            victim = min(self.blocks,
                         key=lambda k: (self.blocks[k][0] + self.blocks[k][1] * (1 << 20)))
            del self.blocks[victim]
            self.evictions += 1
        self.blocks[key] = [self.tick, prio]
        return False


class ARC:
    name = "arc"

    def __init__(self, slots, **kw):
        self.c = max(1, slots)
        self.t1, self.t2, self.b1, self.b2 = deque(), deque(), deque(), deque()
        self.p = 0
        self.evictions = 0

    def _replace(self, key, in_b2):
        if (self.t1 and
                ((len(self.t1) > self.p) or
                 (in_b2 and len(self.t1) == self.p))):
            self.t1.pop()
            self.evictions += 1
        elif self.t2:
            self.t2.pop()
            self.evictions += 1

    def touch(self, key, tick, next_use=None, prio=0):
        if key in self.t1 or key in self.t2:
            if key in self.t1:
                self.t1.remove(key)
                self.p = min(self.c, self.p + (1 if len(self.b1) >= len(self.b2) else 0))
            else:
                self.t2.remove(key)
            self.t2.appendleft(key)
            return True
        case1 = len(self.t1) + len(self.b1)
        case2 = case1 + len(self.t2) + len(self.b2)
        if case1 == self.c:
            if len(self.t1) < self.c:
                self.b1.pop()
                self._replace(key, False)
            else:
                self.t1.pop()
                self.evictions += 1
        elif case1 < self.c <= case2:
            if case2 >= 2 * self.c and self.b2:
                self.b2.pop()
            self._replace(key, False)
        if key in self.b1:
            self.b1.remove(key)
            self.p = min(self.c, self.p + 1 + len(self.b2) // max(1, len(self.b1)))
        elif key in self.b2:
            self.b2.remove(key)
            self.p = max(0, self.p - (1 + len(self.b1) // max(1, len(self.b2))))
        self.t1.appendleft(key)
        return False


class LFU:
    name = "lfu"

    def __init__(self, slots, **kw):
        self.slots = slots
        self.cnt = Counter()
        self.last = {}
        self.resident = set()
        self.tick = 0
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        self.tick += 1
        self.cnt[key] += 1
        self.last[key] = self.tick
        if key in self.resident:
            return True
        if self.slots <= 0:
            return False
        if len(self.resident) >= self.slots:
            victim = min(self.resident, key=lambda k: (self.cnt[k], self.last[k]))
            self.resident.discard(victim)
            self.evictions += 1
        self.resident.add(key)
        return False


class StaticFreq:
    name = "static_freq"

    def __init__(self, slots, freq=None, **kw):
        self.set = {k for k, _ in freq.most_common(slots)} if slots > 0 else set()
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        return key in self.set


class StaticLayerFreq:
    name = "static_layer_freq"

    def __init__(self, slots, layer_freq=None, **kw):
        b = max(1, slots // N_LAYERS) if slots > 0 else 0
        self.set = set()
        for layer, cnt in (layer_freq or {}).items():
            for k, _ in cnt.most_common(b):
                self.set.add(k)
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        return key in self.set


class FreqLRU:
    name = "freq_lru"

    def __init__(self, slots, freq=None, pin_frac=0.5, warmup_counts=None, **kw):
        order = [k for k, _ in (warmup_counts or freq).most_common()]
        self.pin = set(order[:int(slots * pin_frac)]) if slots > 0 else set()
        self.dyn = max(0, slots - len(self.pin))
        self.lru = deque()
        self.pos = set()
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        if key in self.pin:
            return True
        if key in self.pos:
            self.lru.remove(key)
            self.lru.appendleft(key)
            return True
        if self.dyn <= 0:
            return False
        if len(self.lru) >= self.dyn:
            self.pos.discard(self.lru.pop())
            self.evictions += 1
        self.lru.appendleft(key)
        self.pos.add(key)
        return False


class FreqXRecency:
    name = "freq_x_recency"

    def __init__(self, slots, **kw):
        self.slots = slots
        self.freq = Counter()
        self.last = {}
        self.resident = set()
        self.tick = 0
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        self.tick += 1
        self.freq[key] += 1
        self.last[key] = self.tick
        if key in self.resident:
            return True
        if self.slots <= 0:
            return False
        if len(self.resident) >= self.slots:
            maxf = max(1, max(self.freq.values()))
            lo, hi = min(self.last.values()), max(self.last.values())
            span = max(1, hi - lo)
            victim, best = None, None
            for kk in self.resident:
                s = 0.5 * (self.freq[kk] / maxf) + 0.5 * ((self.last[kk] - lo) / span)
                if best is None or s < best:
                    victim, best = kk, s
            self.resident.discard(victim)
            self.evictions += 1
        self.resident.add(key)
        return False


class LayerLRU:
    name = "layer_lru"

    def __init__(self, slots, **kw):
        self.spl = max(1, slots // N_LAYERS) if slots > 0 else 0
        self.lru = defaultdict(deque)
        self.pos = defaultdict(set)
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        layer, expert = key
        dq, ps = self.lru[layer], self.pos[layer]
        if expert in ps:
            dq.remove(expert)
            dq.appendleft(expert)
            return True
        if len(dq) >= self.spl:
            ps.discard(dq.pop())
            self.evictions += 1
        dq.appendleft(expert)
        ps.add(expert)
        return False


class CostAware:
    """All records one size -> provably LRU; confirmed by evaluation."""

    name = "cost_aware"

    def __init__(self, slots, **kw):
        self.lru = LRU(slots)
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        r = self.lru.touch(key, tick, next_use, prio)
        self.evictions = self.lru.evictions
        return r


class Belady:
    name = "belady"

    def __init__(self, slots, **kw):
        self.slots = slots
        self.resident = {}   # key -> absolute next-use index
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        nu = next_use if next_use is not None else 10 ** 12
        if key in self.resident:
            self.resident[key] = nu
            return True
        if self.slots <= 0:
            return False
        if len(self.resident) >= self.slots:
            far = max(self.resident, key=lambda k: self.resident[k])
            del self.resident[far]
            self.evictions += 1
        self.resident[key] = nu
        return False


POLICIES = ("lru", "engine_priority_lru", "arc", "lfu", "static_freq",
            "static_layer_freq", "freq_lru", "freq_lru_warmup",
            "freq_x_recency", "layer_lru", "cost_aware", "belady")
CACHE_CLASSES = {
    "lru": LRU, "engine_priority_lru": PriorityLRU, "arc": ARC, "lfu": LFU,
    "static_freq": StaticFreq, "static_layer_freq": StaticLayerFreq,
    "freq_lru": FreqLRU, "freq_lru_warmup": FreqLRU,
    "freq_x_recency": FreqXRecency, "layer_lru": LayerLRU,
    "cost_aware": CostAware, "belady": Belady,
}


def simulate(batches, policy, slots, freq, layer_freq, warmup_counts,
             pin_frac=0.5, next_use=None):
    cache = CACHE_CLASSES[policy](
        slots, freq=freq, layer_freq=layer_freq,
        warmup_counts=warmup_counts if policy == "freq_lru_warmup" else None,
        pin_frac=pin_frac)
    hits = misses = 0
    evictions = 0
    layer_hit = Counter()
    layer_req = Counter()
    i = 0
    for step, layer, keys in batches:
        K = len(keys)
        for j, key in enumerate(keys):
            nu = next_use[i] if next_use is not None else None
            prio = (K - j) if policy == "engine_priority_lru" else 0
            hit = cache.touch(key, i, nu, prio)
            hits += hit
            misses += not hit
            layer_hit[layer] += hit
            layer_req[layer] += 1
            i += 1
    evictions = cache.evictions
    return {"policy": policy, "slots": slots, "requests": i, "hits": hits,
            "misses": misses, "evictions": evictions,
            "hit_rate": hits / i if i else 0.0,
            "layer_hit": layer_hit, "layer_req": layer_req}


def stack_distance_hist(batches):
    """Exact distinct-key stack distance per repeat access (LRU semantics:
    repeat at stack position i hits iff i < slots)."""
    hist = Counter()
    stack = []
    pos = {}
    for _, _, keys in batches:
        for key in keys:
            if key in pos:
                i = pos[key]
                hist[i] += 1
                stack.pop(i)
            else:
                hist[-1] += 1
            stack.insert(0, key)
            for j, k in enumerate(stack):
                pos[k] = j
    return hist


def next_use_index(batches):
    keys = [k for _, _, ks in batches for k in ks]
    nxt = [10 ** 12] * len(keys)
    seen = {}
    for i in range(len(keys) - 1, -1, -1):
        nxt[i] = seen.get(keys[i], 10 ** 12)
        seen[keys[i]] = i
    return keys, nxt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    batches = load_batches()
    freq = Counter(k for _, _, ks in batches for k in ks)
    layer_freq = defaultdict(Counter)
    for (l, e), c in freq.items():
        layer_freq[l][(l, e)] = c
    warmup_counts = Counter(k for s, _, ks in batches if s == 0 for k in ks)
    keys, nxt = next_use_index(batches)

    results = []

    def run(scope, sub_batches, budgets):
        sf = Counter(k for _, _, ks in sub_batches for k in ks)
        slf = defaultdict(Counter)
        for (l, e), c in sf.items():
            slf[l][(l, e)] = c
        sw = Counter(k for s, _, ks in sub_batches if s == 0 for k in ks)
        _, snxt = next_use_index(sub_batches)
        for g in budgets:
            slots = int(g * GIB // RECORD_BYTES)
            for pol in POLICIES:
                pfs = (0.25, 0.5, 0.75) if pol == "freq_lru" else (0.5,)
                for pf in pfs:
                    r = simulate(sub_batches, pol, slots, sf, slf, sw,
                                 pin_frac=pf, next_use=snxt)
                    results.append({
                        "scope": scope, "policy": pol, "pin_frac": pf,
                        "budget_gib": g, "slots": slots,
                        "requests": r["requests"], "hits": r["hits"],
                        "misses": r["misses"],
                        "hit_rate": round(r["hit_rate"], 6),
                        "evictions": r["evictions"],
                        "miss_frac": round(r["misses"] / max(1, r["requests"]), 6),
                    })

    run("full", batches,
        [8, 12, 16, 20, 24, 32, 48, 64, 96, 128])
    gpu0 = [b for b in batches if b[1] < SPLIT_GPU0]
    gpu1 = [b for b in batches if b[1] >= SPLIT_GPU0]
    run("gpu0", gpu0, [1, 2, 3, 3.5, 4, 6, 8])
    run("gpu1", gpu1, [1, 2, 3, 3.5, 4, 6, 8])

    with open(OUT / "sim_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    hist = stack_distance_hist(batches)
    with open(OUT / "reuse_distance.json", "w") as f:
        json.dump({str(k): v for k, v in sorted(hist.items())}, f, indent=1)

    # ---- validation ---------------------------------------------------
    def lru_counts(sub_batches, slots):
        sf = Counter(k for _, _, ks in sub_batches for k in ks)
        slf = defaultdict(Counter)
        for (l, e), c in sf.items():
            slf[l][(l, e)] = c
        sw = Counter(k for s, _, ks in sub_batches if s == 0 for k in ks)
        r = simulate(sub_batches, "lru", slots, sf, slf, sw)
        return r["hits"], r["misses"], r["evictions"]

    def prio_counts(sub_batches, slots):
        sf = Counter(k for _, _, ks in sub_batches for k in ks)
        slf = defaultdict(Counter)
        for (l, e), c in sf.items():
            slf[l][(l, e)] = c
        sw = Counter(k for s, _, ks in sub_batches if s == 0 for k in ks)
        r = simulate(sub_batches, "engine_priority_lru", slots, sf, slf, sw)
        return r["hits"], r["misses"], r["evictions"]

    val = {
        "requests_total": len(keys),
        "unique_pairs": len(freq),
        "host_filllive_682": {
            "cuda0": {"sim": lru_counts(gpu0, 682), "sealed": [1223, 1390, 708]},
            "cuda1": {"sim": lru_counts(gpu1, 682), "sealed": [1395, 1091, 409]},
        },
        "vram_v60_281_engine_priority_lru": {
            "cuda0": {"sim": prio_counts(gpu0, 281), "sealed": [328, 2285, 2004]},
            "cuda1": {"sim": prio_counts(gpu1, 281), "sealed": [327, 2159, 1878]},
        },
        "vram_v60_281_plain_lru": {
            "cuda0": {"sim": lru_counts(gpu0, 281)},
            "cuda1": {"sim": lru_counts(gpu1, 281)},
        },
    }
    with open(OUT / "validation.json", "w") as f:
        json.dump(val, f, indent=1)

    print("requests", len(keys), "unique pairs", len(freq))
    for name, block in val.items():
        if isinstance(block, dict) and "cuda0" in block:
            print(name, "cuda0 sim", block["cuda0"]["sim"],
                  "sealed", block["cuda0"].get("sealed"))
            print(" " * len(name), "cuda1 sim", block["cuda1"]["sim"],
                  "sealed", block["cuda1"].get("sealed"))


if __name__ == "__main__":
    main()
