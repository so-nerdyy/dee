#!/usr/bin/env python3
"""Phase 2 working-set policy study — CORRECTED offline simulation (v4).

WHY v4 EXISTS (the b7b9c7f causality bug, fixed mechanically):

  v3 allowed count-based policies (static_freq, freq_lru, freq_lru_warmup) to
  return HIT on the very first access of a pinned record — including the
  prefill pass whose own statistics defined the pin set. That credited
  residency that never existed (an uncharged PREWARM) inside what was
  reported as a cold-start comparison, producing the impossible result that a
  realizable cold-start policy (static pin 91.4 %) beats offline Belady/MIN
  (53.6 %). No online policy beats MIN from an equal initial state; the
  comparison was invalid, not the theory.

v4 enforces the three initial-state regimes as separate contracts:

  REGIME A (COLD)      empty cache; every first physical materialization is
                       charged. All online policies + Belady/MIN (bound).
  REGIME B (WARMUP)    starts cold; statistics may be learned from already-
                       executed forwards only; a policy change happens at a
                       step boundary AFTER the evidence exists; every pinned
                       record no longer resident at the boundary is charged a
                       repopulation miss. (freq_lru_warmup only.)
  REGIME C (PREWARMED) initial residency exists and is EXPLICITLY labeled:
                       initial_records / initial_state_gib columns. Its cost
                       is never mixed into cold-start comparisons. The bound
                       is Belady/MIN GIVEN THE SAME INITIAL STATE.

Every row carries `regime`, `initial_records`, `initial_state_gib` and
`repopulation_charged`. Comparisons are only valid within a regime, or
against `belady_same_state_prewarm` for regime C.

Streams/anchors unchanged from v3 (all validated):
  journal: sealed v50 canonical route journal (engine-dedup stream, 5,099
  requests); host anchor fill-live 682 slots; VRAM anchor v60 281 slots.

Outputs (research/phase2-ws-policy/results/):
  sim_rows_v4.csv      full regime-labeled matrix
  sim_rows_v4_derived.csv  + SSD bytes/token + wall columns (0.29..12 GiB/s)
  ram_slope_v4.csv     marginal slow-bytes saved per added GiB (RAM knee)
  vram_audit.json      per-decode-step lru vs engine_priority_lru @281 slots
  validation_v4.json   sealed anchors + regime-B repopulation sanity + hashes
  reuse_distance.json  exact stack-distance histogram (unchanged content)
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOURNAL_REL = (REPO / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
               "v50-evidence-20260829T195940Z/routed_experts.jsonl")
# the phase2 worktree ships without benchmark_reports; fall back to the main
# checkout (byte-identical journal: v60 embeds the v50 journal, sha256-checked)
JOURNAL_FALLBACKS = [
    Path("C:/Users/carth/Downloads/dynamic_expert_eviction/dee.cpp/"
         "benchmark_reports/deepseek-v4-flash-0731-t4/"
         "v50-evidence-20260829T195940Z/routed_experts.jsonl"),
]
OUT = REPO / "research/phase2-ws-policy/results"

RECORD_BYTES = 13_369_344          # 12.75 MiB DEE4 packed FP4 record
N_LAYERS = 43
GIB = 1 << 30
SPLIT_GPU0 = 22                    # cuda0: layers 0-21, cuda1: 22-42

BW_MEASURED = (0.29, 0.33, 0.37)   # GiB/s, Phase-1 live T4x2 matrix
BW_FUTURE = (3.0, 5.0, 7.0, 12.0)  # GiB/s floors
HOST_BUDGETS = [8, 12, 16, 20, 24, 32, 48, 64, 96, 128]
VRAM_BUDGETS = [1, 2, 3, 3.5, 4, 6, 8]
FINE_BUDGETS = [float(x) for x in range(4, 41)]  # RAM-knee slope scan


def journal_path():
    for p in (JOURNAL_REL, *JOURNAL_FALLBACKS):
        if p.exists():
            return p
    raise FileNotFoundError("v50 routed_experts.jsonl not found")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- stream

def load_batches(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
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
    regime = "A"

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
    """Production VRAM semantics: score = last_used + prio*2**20, refreshed
    on hit (vram_cache.h PRIORITY_WEIGHT=1<<20; engine passes K-k). Regime A;
    evaluated to document the artifact."""
    name = "engine_priority_lru"
    regime = "A"

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
                         key=lambda k: (self.blocks[k][0]
                                        + self.blocks[k][1] * (1 << 20)))
            del self.blocks[victim]
            self.evictions += 1
        self.blocks[key] = [self.tick, prio]
        return False


class ARC:
    name = "arc"
    regime = "A"

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
    regime = "A"

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


class FreqXRecency:
    name = "freq_x_recency"
    regime = "A"

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
    regime = "A"

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
    """All records one size -> provably LRU; kept as the required
    equivalence check."""
    name = "cost_aware"
    regime = "A"

    def __init__(self, slots, **kw):
        self.lru = LRU(slots)
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        r = self.lru.touch(key, tick, next_use, prio)
        self.evictions = self.lru.evictions
        return r


class Belady:
    """Offline MIN. regime A: cold start (the honest bound for A and B)."""
    name = "belady_cold"
    regime = "A"

    def __init__(self, slots, **kw):
        self.slots = slots
        self.resident = {}
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


class BeladyPrewarm:
    """MIN GIVEN A PREWARMED INITIAL STATE (regime C bound). The initial
    resident set is the top-`slots` frequency-ranked set — exactly the state
    static_freq_prewarm is credited with. Never comparable to regime A."""
    name = "belady_same_state_prewarm"
    regime = "C"

    def __init__(self, slots, freq=None, first_use=None, **kw):
        self.slots = slots
        init = {k for k, _ in freq.most_common(slots)} if slots > 0 else set()
        self.initial = init
        # exact first-future-use distance per resident entry; 10**12 only
        # for records that never occur in the stream. (The earlier
        # 10**12-for-all init evicted soon-needed prewarmed records on the
        # first faults, understating this bound — see
        # research/phase2-ws-policy/SIM_INIT_FIX.md.)
        fu = first_use or {}
        self.resident = {k: fu.get(k, 10 ** 12) for k in init}
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


class StaticFreqPrewarm:
    """Offline top-N static pin, regime C ONLY: the top-N set is prewarmed
    (labeled), everything else misses forever. No dynamics, no evictions."""
    name = "static_freq_prewarm"
    regime = "C"

    def __init__(self, slots, freq=None, **kw):
        self.set = {k for k, _ in freq.most_common(slots)} if slots > 0 else set()
        self.initial = set(self.set)
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        return key in self.set


class StaticLayerFreqPrewarm:
    """Per-layer top-B static pin, regime C (layer-budgeted placement)."""
    name = "static_layer_freq_prewarm"
    regime = "C"

    def __init__(self, slots, layer_freq=None, **kw):
        b = max(1, slots // N_LAYERS) if slots > 0 else 0
        self.set = set()
        for layer, cnt in (layer_freq or {}).items():
            for k, _ in cnt.most_common(b):
                self.set.add(k)
        self.initial = set(self.set)
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        return key in self.set


class FreqLRUPrewarm:
    """Pin top pin_frac*slots (prewarmed, labeled) + LRU remainder (starts
    empty). Regime C."""
    name = "freq_lru_prewarm"
    regime = "C"

    def __init__(self, slots, freq=None, pin_frac=0.5, **kw):
        order = [k for k, _ in freq.most_common()]
        self.pin = set(order[:int(slots * pin_frac)]) if slots > 0 else set()
        self.initial = set(self.pin)
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


class BeladyPin:
    """REGIME C bound for the pin+remainder family, CONTRACT-MATCHED:
    initial resident = the SAME top-(pin_frac*slots) frequency set the pin
    policy is credited with, PROTECTED (never evicted, never charged);
    the remaining (1-pf)*slots dynamic slots run offline MIN (fetch-on-fault)
    over the non-protected substream. Dominates freq_lru_prewarm(pf)
    pointwise by construction (same protection, MIN >= LRU on the rest).
    Never comparable to regime A or to the top-slots prewarm policies."""
    name = "belady_pin_same_state"
    regime = "C"

    def __init__(self, slots, freq=None, pin_frac=0.5, **kw):
        order = [k for k, _ in freq.most_common()]
        self.protected = set(order[:int(slots * pin_frac)]) if slots > 0 else set()
        self.initial = set(self.protected)
        self.dyn_cap = max(0, slots - len(self.protected))
        self.resident = {}          # dynamic key -> next-use
        self.evictions = 0

    def touch(self, key, tick, next_use=None, prio=0):
        if key in self.protected:
            return True
        nu = next_use if next_use is not None else 10 ** 12
        if key in self.resident:
            self.resident[key] = nu
            return True
        if self.dyn_cap <= 0:
            return False
        if len(self.resident) >= self.dyn_cap:
            far = max(self.resident, key=lambda k: self.resident[k])
            del self.resident[far]
            self.evictions += 1
        self.resident[key] = nu
        return False


class LRUPrewarm:
    """Control row for regime C: give plain LRU the SAME prewarmed top-N set
    as the pin policies (N = slots) and then run LRU. Isolates policy value
    from prewarm value. (With N = slots it degenerates to the static set
    plus LRU churn of evicted pins; informative at the margins.)"""
    name = "lru_prewarm_topN"
    regime = "C"

    def __init__(self, slots, freq=None, **kw):
        self.slots = slots
        self.init_order = [k for k, _ in freq.most_common(slots)] if slots > 0 else []
        self.initial = set(self.init_order)
        self.lru = deque(self.init_order)   # prewarm order = rank order
        self.pos = set(self.init_order)
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


class WarmupPinLRU:
    """REGIME B: the online 'warmup-count pin' proposal, mechanically
    corrected. Runs plain LRU from cold; counts requests from already-
    executed forwards ONLY; at the step boundary it pins the top
    pin_frac*slots records by those counts, CHARGING a repopulation miss
    for every pinned record no longer resident, removing pins from the
    dynamic LRU pool (no phantom hits), and reserving the dynamic pool at
    slots - |pin|. No future knowledge, no uncharged residency.

    boundary_step=1: pin set from the step-0 prefill pass (every within-pass
    count is 1 on this architecture -- one layer visit per step -- so the
    pin set is tie-broken by first-encounter order).
    boundary_step=4: counts from the first four executed forwards (prefill
    + 3 decode steps) -- the first point where a real frequency signal
    exists. Still causal: evidence precedes the policy change."""
    name = "freq_lru_warmup"
    regime = "B"

    def __init__(self, slots, pin_frac=0.5, boundary_step=1, **kw):
        self.slots = slots
        self.pin_frac = pin_frac
        self.boundary_step = boundary_step
        self.counts = Counter()
        self.boundary_done = False
        self.pin = set()
        self.lru = deque()
        self.pos = set()
        self.lru_cap = slots          # dynamic pool cap (shrinks at boundary)
        self.evictions = 0
        self.repopulation = 0

    def _lru_touch(self, key):
        """Plain LRU step; returns True on hit."""
        if key in self.pos:
            self.lru.remove(key)
            self.lru.appendleft(key)
            return True
        if len(self.lru) >= self.lru_cap:
            self.pos.discard(self.lru.pop())
            self.evictions += 1
        self.lru.appendleft(key)
        self.pos.add(key)
        return False

    def close_boundary(self):
        """Apply the pin using counts of requests already executed
        (forward steps strictly before the boundary step)."""
        n_pin = int(self.slots * self.pin_frac)
        self.pin = {k for k, _ in self.counts.most_common(n_pin)}
        resident_pins = {k for k in self.pin if k in self.pos}
        # charge repopulation for pinned records evicted before the boundary
        self.repopulation += len(self.pin) - len(resident_pins)
        # remove ALL pins from the dynamic LRU pool (policy-resident now)
        for k in self.pin:
            if k in self.pos:
                self.pos.discard(k)
                self.lru.remove(k)
        # reserve the dynamic pool at slots - |pin|; drop oldest overflow
        self.lru_cap = self.slots - len(self.pin)
        while len(self.lru) > self.lru_cap:
            self.pos.discard(self.lru.pop())
            self.evictions += 1
        # repopulated pins materialize into the reserved pin region
        self.boundary_done = True

    def touch(self, key, tick, next_use=None, prio=0):
        if self.boundary_done and key in self.pin:
            return True
        hit = self._lru_touch(key)
        if not self.boundary_done:
            self.counts[key] += 1
        return hit


POLICIES_A = ("lru", "engine_priority_lru", "arc", "lfu", "freq_x_recency",
              "layer_lru", "cost_aware", "belady_cold")
POLICIES_B = ("freq_lru_warmup", "freq_lru_warmup_s4")
POLICIES_C = ("static_freq_prewarm", "static_layer_freq_prewarm",
              "freq_lru_prewarm", "lru_prewarm_topN",
              "belady_same_state_prewarm", "belady_pin_same_state")
CACHE_CLASSES = {
    "lru": LRU, "engine_priority_lru": PriorityLRU, "arc": ARC, "lfu": LFU,
    "freq_x_recency": FreqXRecency, "layer_lru": LayerLRU,
    "cost_aware": CostAware, "belady_cold": Belady,
    "belady_same_state_prewarm": BeladyPrewarm,
    "belady_pin_same_state": BeladyPin,
    "static_freq_prewarm": StaticFreqPrewarm,
    "static_layer_freq_prewarm": StaticLayerFreqPrewarm,
    "freq_lru_prewarm": FreqLRUPrewarm, "lru_prewarm_topN": LRUPrewarm,
    "freq_lru_warmup": WarmupPinLRU, "freq_lru_warmup_s4": WarmupPinLRU,
}


def simulate(batches, policy, slots, freq, layer_freq, pin_frac=0.5,
             next_use=None, boundary_step=1):
    """Run one (policy, slots) cell. Returns counters dict.

    boundary_step: for regime B, the forward step after which the pin set is
    applied (default 1 = after the step-0 prefill pass, the evidence source).
    """
    cls = CACHE_CLASSES[policy]
    boundary_step = 4 if policy == "freq_lru_warmup_s4" else 1
    # first occurrence index per record in this scope's stream — the exact
    # initial next-use distance BeladyPrewarm needs for the regime-C bound.
    first_use = {}
    for i0, key in enumerate(k for _, _, ks in batches for k in ks):
        if key not in first_use:
            first_use[key] = i0
    cache = cls(slots, freq=freq, layer_freq=layer_freq, pin_frac=pin_frac,
                boundary_step=boundary_step, first_use=first_use)
    hits = misses = 0
    prefill_misses = decode_misses = 0
    prefill_hits = decode_hits = 0
    layer_hit, layer_req = Counter(), Counter()
    decode_step_hits = defaultdict(int)     # step -> hits (VRAM audit)
    i = 0
    cur_step = None
    for step, layer, keys in batches:
        if policy in ("freq_lru_warmup", "freq_lru_warmup_s4") \
                and not cache.boundary_done and step >= boundary_step:
            cache.close_boundary()
        K = len(keys)
        for j, key in enumerate(keys):
            nu = next_use[i] if next_use is not None else None
            prio = (K - j) if policy == "engine_priority_lru" else 0
            hit = cache.touch(key, i, nu, prio)
            hits += hit
            misses += not hit
            if step == 0:
                prefill_hits += hit
                prefill_misses += not hit
            else:
                decode_hits += hit
                decode_misses += not hit
                decode_step_hits[step] += hit
            layer_hit[layer] += hit
            layer_req[layer] += 1
            i += 1
    repop = getattr(cache, "repopulation", 0)
    initial = getattr(cache, "initial", set())
    # regime B: repopulation is REAL charged work at the boundary (decode
    # phase for every boundary_step >= 1); misses_charged = stream misses
    # + repopulation. hit_rate stays over executed requests.
    return {
        "policy": policy, "regime": cls.regime, "slots": slots,
        "pin_frac": pin_frac if policy in ("freq_lru_prewarm",
                                           "belady_pin_same_state",
                                           "freq_lru_warmup",
                                           "freq_lru_warmup_s4") else "",
        "requests": i, "hits": hits, "misses": misses + repop,
        "stream_misses": misses,
        "repopulation_charged": repop,
        "evictions": cache.evictions,
        "hit_rate": hits / i if i else 0.0,
        "prefill_misses": prefill_misses,
        "decode_misses": decode_misses + repop,
        "prefill_hits": prefill_hits, "decode_hits": decode_hits,
        "initial_records": len(initial),
        "initial_state_gib": round(len(initial) * RECORD_BYTES / GIB, 3),
        "decode_step_hits": dict(decode_step_hits),
        "layer_hit": layer_hit, "layer_req": layer_req,
    }


def stack_distance_hist(batches):
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
    jp = journal_path()
    jsha = sha256_of(jp)
    batches = load_batches(jp)
    freq = Counter(k for _, _, ks in batches for k in ks)
    layer_freq = defaultdict(Counter)
    for (l, e), c in freq.items():
        layer_freq[l][(l, e)] = c
    keys, nxt = next_use_index(batches)
    n_steps = max(s for s, _, _ in batches) + 1
    print(f"journal {jp}")
    print(f"journal sha256 {jsha}")
    print(f"requests {len(keys)} unique pairs {len(freq)} steps {n_steps}")

    results = []

    def run(scope, sub_batches, budgets, fine=False):
        sf = Counter(k for _, _, ks in sub_batches for k in ks)
        slf = defaultdict(Counter)
        for (l, e), c in sf.items():
            slf[l][(l, e)] = c
        _, snxt = next_use_index(sub_batches)
        for g in budgets:
            slots = int(g * GIB // RECORD_BYTES)
            pols = FINE_POLICIES if fine \
                else (POLICIES_A + POLICIES_B + POLICIES_C)
            for pol in pols:
                pfs = (0.25, 0.5, 0.75, 0.9, 0.99) \
                    if pol in ("freq_lru_prewarm", "belady_pin_same_state") \
                    else (0.5,) if pol in ("freq_lru_warmup",
                                           "freq_lru_warmup_s4") else ("",)
                for pf in pfs:
                    r = simulate(sub_batches, pol, slots, sf, slf,
                                 pin_frac=pf, next_use=snxt)
                    results.append({
                        "scope": scope, "regime": r["regime"], "policy": pol,
                        "pin_frac": pf, "budget_gib": g, "slots": slots,
                        "initial_records": r["initial_records"],
                        "initial_state_gib": r["initial_state_gib"],
                        "repopulation_charged": r["repopulation_charged"],
                        "requests": r["requests"], "hits": r["hits"],
                        "misses": r["misses"], "evictions": r["evictions"],
                        "hit_rate": round(r["hit_rate"], 6),
                        "prefill_misses": r["prefill_misses"],
                        "decode_misses": r["decode_misses"],
                        "prefill_hits": r["prefill_hits"],
                        "decode_hits": r["decode_hits"],
                        "miss_frac": round(r["misses"] / max(1, r["requests"]), 6),
                    })
                    if fine:
                        results.pop()   # slope scan rows live in fine_rows
                        fine_rows.append({
                            "scope": scope, "regime": r["regime"],
                            "policy": pol, "budget_gib": g, "slots": slots,
                            "hits": r["hits"],
                        })

    fine_rows = []
    FINE_POLICIES = ("lru", "static_freq_prewarm", "freq_lru_warmup",
                     "freq_lru_warmup_s4", "freq_lru_prewarm")
    run("full", batches, HOST_BUDGETS)
    run("full", batches, FINE_BUDGETS, fine=True)
    gpu0 = [b for b in batches if b[1] < SPLIT_GPU0]
    gpu1 = [b for b in batches if b[1] >= SPLIT_GPU0]
    run("gpu0", gpu0, VRAM_BUDGETS)
    run("gpu1", gpu1, VRAM_BUDGETS)

    with open(OUT / "sim_rows_v4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # derived: bytes + walls
    derived = []
    for r in results:
        miss_bytes = r["misses"] * RECORD_BYTES
        row = dict(r)
        row["ssd_MiB_per_token"] = round(miss_bytes / GIB * 1024 / 16, 1)
        for bw in BW_MEASURED + BW_FUTURE:
            row[f"wall_s_at_{bw}GiBps"] = round(
                (miss_bytes / GIB) / bw, 3) if bw else 0.0
        derived.append(row)
    with open(OUT / "sim_rows_v4_derived.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(derived[0].keys()))
        w.writeheader()
        w.writerows(derived)

    # RAM-knee slopes: slow bytes saved per added GiB (regime A lru and
    # regime C static pin), central-difference on the fine scan
    slopes = []
    for scope in ("full",):
        for pol in ("lru", "static_freq_prewarm", "freq_lru_warmup",
                    "freq_lru_warmup_s4", "freq_lru_prewarm"):
            pts = sorted((r["budget_gib"], r["hits"]) for r in fine_rows
                         if r["scope"] == scope and r["policy"] == pol)
            for i in range(1, len(pts)):
                b0, h0 = pts[i - 1]
                b1, h1 = pts[i]
                if b1 - b0 != 1.0:
                    continue
                dh = h1 - h0
                slopes.append({
                    "scope": scope, "policy": pol, "regime":
                        CACHE_CLASSES[pol].regime,
                    "budget_gib_lo": b0, "budget_gib_hi": b1,
                    "extra_hits_per_GiB": dh,
                    "saved_slow_MiB_per_added_GiB": round(
                        dh * RECORD_BYTES / (1 << 20), 1),
                })
    with open(OUT / "ram_slope_v4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(slopes[0].keys()))
        w.writeheader()
        w.writerows(slopes)

    hist = stack_distance_hist(batches)
    with open(OUT / "reuse_distance.json", "w") as f:
        json.dump({str(k): v for k, v in sorted(hist.items())}, f, indent=1)

    # ---- VRAM audit: per-decode-step hits, lru vs production score @281 ----
    vram_audit = {"journal_sha256": jsha, "budget_slots": 281}
    for scope, sub in (("cuda0", gpu0), ("cuda1", gpu1)):
        sf = Counter(k for _, _, ks in sub for k in ks)
        slf = defaultdict(Counter)
        for (l, e), c in sf.items():
            slf[l][(l, e)] = c
        _, snxt = next_use_index(sub)
        entry = {}
        for pol in ("engine_priority_lru", "lru"):
            r = simulate(sub, pol, 281, sf, slf, next_use=snxt)
            entry[pol] = {
                "hits": r["hits"], "misses": r["misses"],
                "evictions": r["evictions"],
                "prefill_hits": r["prefill_hits"],
                "decode_hits": r["decode_hits"],
                "per_decode_step_hits": [r["decode_step_hits"].get(s, 0)
                                         for s in range(1, n_steps)],
            }
        loads_saved = (entry["engine_priority_lru"]["misses"]
                       - entry["lru"]["misses"])
        entry["loads_saved"] = loads_saved
        entry["h2d_bytes_saved"] = loads_saved * RECORD_BYTES
        entry["h2d_MiB_per_decode_token"] = round(
            loads_saved * RECORD_BYTES / (1 << 20) / (n_steps - 1), 1)
        vram_audit[scope] = entry
    with open(OUT / "vram_audit.json", "w") as f:
        json.dump(vram_audit, f, indent=1)

    # ---- validation -----------------------------------------------------
    def counts_for(sub, pol, slots):
        sf = Counter(k for _, _, ks in sub for k in ks)
        slf = defaultdict(Counter)
        for (l, e), c in sf.items():
            slf[l][(l, e)] = c
        _, snxt = next_use_index(sub)
        r = simulate(sub, pol, slots, sf, slf, next_use=snxt)
        return r["hits"], r["misses"], r["evictions"]

    # regime-B sanity: repopulation at 16/24 GiB must be within [0, pin size]
    warm = {g: {} for g in (16, 24)}
    for g in warm:
        slots = int(g * GIB // RECORD_BYTES)
        for scope, sub in (("full", batches), ("gpu0", gpu0), ("gpu1", gpu1)):
            sf = Counter(k for _, _, ks in sub for k in ks)
            slf = defaultdict(Counter)
            for (l, e), c in sf.items():
                slf[l][(l, e)] = c
            r = simulate(sub, "freq_lru_warmup", slots, sf, slf,
                         pin_frac=0.5, next_use=None)
            warm[g][scope] = {
                "hits": r["hits"], "misses": r["misses"],
                "repopulation": r["repopulation_charged"],
                "pin_size": int(slots * 0.5),
            }

    val = {
        "journal_path": str(jp),
        "journal_sha256": jsha,
        "requests_total": len(keys),
        "unique_pairs": len(freq),
        "host_filllive_682_lru": {
            "cuda0": {"sim": counts_for(gpu0, "lru", 682),
                      "sealed": [1223, 1390, 708]},
            "cuda1": {"sim": counts_for(gpu1, "lru", 682),
                      "sealed": [1395, 1091, 409]},
        },
        "vram_v60_281_engine_priority_lru": {
            "cuda0": {"sim": counts_for(gpu0, "engine_priority_lru", 281),
                      "sealed": [328, 2285, 2004]},
            "cuda1": {"sim": counts_for(gpu1, "engine_priority_lru", 281),
                      "sealed": [327, 2159, 1878]},
        },
        "vram_v60_281_plain_lru": {
            "cuda0": {"sim": counts_for(gpu0, "lru", 281)},
            "cuda1": {"sim": counts_for(gpu1, "lru", 281)},
        },
        "regimeB_warmup_repopulation": warm,
    }
    with open(OUT / "validation_v4.json", "w") as f:
        json.dump(val, f, indent=1)

    print("\nvalidation:")
    for name in ("host_filllive_682_lru", "vram_v60_281_engine_priority_lru"):
        for gpu in ("cuda0", "cuda1"):
            print(" ", name, gpu, "sim", val[name][gpu]["sim"],
                  "sealed", val[name][gpu].get("sealed"))
    print("  regimeB repop:", json.dumps(warm))
    print("\nvram audit:")
    for gpu in ("cuda0", "cuda1"):
        e = vram_audit[gpu]
        print(f"  {gpu}: priority hits {e['engine_priority_lru']['hits']}"
              f" vs lru {e['lru']['hits']}  loads saved {e['loads_saved']}"
              f"  H2D saved {e['h2d_bytes_saved']/1e9:.2f} GB"
              f"  ({e['h2d_MiB_per_decode_token']} MiB/decode-token)")


if __name__ == "__main__":
    main()
