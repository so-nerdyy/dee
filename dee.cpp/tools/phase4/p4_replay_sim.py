#!/usr/bin/env python3
"""p4_replay_sim.py — Phase-4 two-level cache replay simulator (workstream B1).

Replays authoritative route journals (one JSONL record per (forward_step,
layer)) through a source-faithful model of the engine's VRAM + host-pack
cache hierarchy, and emits per-policy / per-capacity counter tables for the
Phase-4 cache-hierarchy campaign.

Event model (verified against dee.cpp @ research/phase4-cache-hierarchy):

  per record (engine.cpp:549-552, moe_forward_batch_device_impl):
    active_experts = sorted(set(flatten(expert_ids_rank_order)))
    K              = len(active_experts)            # unique count of the CALL
    cache_batch    = floor(vram_bytes / blob_bytes) # 74 fp16 / 281 fp4 @3.5GiB
    for each chunk of `cache_batch` experts (ascending order):
      (a) HOST consult (prepare_fp4_experts, engine.cpp:2790): the whole
          chunk is pushed through HostPackCache::get_batch in sub-batches of
          queue_depth=6 *before* any VRAM staging (eager consult — the live
          config consults the pack even for would-be VRAM hits).
      (b) VRAM stage (stage_expert -> prefetch_impl, async_prefetcher.cpp:330):
          for key at global call index i: priority = K - i.
            resident -> hit : tick++, last_used=tick, priority=prio, pin
            miss     -> cold: evict_until_free(min score over pins==0)
                              then insert+pin
          and, first, an uncounted HostPackCache::get_if_present refresh
          (engine.cpp:2915) that moves the key to the pack MRU front.
      (c) consume: every chunk member's pin is released
          (wait_on_stream drops the prefetch pin; the compute pin is
          unpinned at engine.cpp:1054).  Later chunks of the SAME call may
          therefore evict earlier chunks' members; members staged so far in
          the CURRENT chunk are pin-protected.

  VRAM eviction score (vram_cache.h:196-199, bug-compat):
      score = last_used + priority * 2**20      # min evicted first
  ensure-hit refreshes BOTH last_used and priority (vram_cache.cpp:177-186).

  HostPackCache::get_batch (host_pack_cache.cpp:170-238):
      scan requests in order: in-batch duplicate -> counted hit (no LRU move);
      present -> counted hit + move-to-MRU; else counted miss and
      additional_bytes += rec.  Then evict LRU-back entries skipping every
      in-batch key until used + additional <= budget; then admit each unique
      miss at MRU-front in request order, BEFORE fills run.

  Host consult modes: eager = whole chunk consulted (live config; host
  hits+misses == requests).  lazy = only VRAM misses consult the pack
  (counterfactual; each miss is a 1-key consult at stage time).
  host=0 GiB = honest bypass arm: the pack is never consulted and every
  VRAM cold load is charged a store read (a 1-record bounce buffer adds
  no hits either way; entries/evictions are reported as 0).

  Multi-prompt: --persist-across-prompts reproduces the GPU-2 process
  semantics — cache contents (incl. per-block priority/last_used and the
  monotonic tick) persist across prompts while reported counters reset per
  prompt.  Verified vs result-a1-q{0,1,2}.json: with a warm 74-slot arena
  every cold load evicts (q1/q2 evictions == cold_loads exactly), and the
  host_pack counters are process-cumulative.

Sizes (DeepSeek-V4: hidden=4096, inter=2048, 256 experts, 43 MoE layers):
  fp16 cache blob  = 3*inter*hidden*2          = 50,331,648 B (48 MiB)
  fp4 packed blob  = (blob_elems*17+31)//32    = 13,369,344 B (12.75 MiB)
                     == host pack record size == H2D transfer size
  VRAM 3.5 GiB     -> 74 fp16 slots / 281 fp4 slots
  host 8.5 GiB     -> 682 records

Correctness gates are fail-closed (raise, never warn): accounting closure,
capacity, pin protection, cold-regime first-touch, Belady dominance
tripwire, and journal schema validation.

Usage:
  python3 p4_replay_sim.py --journals routed_experts-a1-q0.jsonl \
      --vram-policy priority_lru --persist-across-prompts \
      --anchors result-a1-q0.json
See README.md for the full CLI.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import OrderedDict, defaultdict

# ---------------------------------------------------------------------------
# Model constants (engine.cpp / vram_cache.h / host_pack_cache.cpp)
# ---------------------------------------------------------------------------

FP16_BLOB_BYTES = 50_331_648      # 3*2048*4096*2
FP4_RECORD_BYTES = 13_369_344     # (3*2048*4096*17+31)//32; pack record & H2D
PRIORITY_WEIGHT = 1 << 20         # vram_cache.h PRIORITY_WEIGHT
NUM_LAYERS = 43                   # MoE layers executed (0..42)
NUM_EXPERTS = 256
TOPK = 6
DEFAULT_LAYER_SPLIT = 22          # cuda:0 = layers 0-21, cuda:1 = 22-42
DEFAULT_QUEUE_DEPTH = 6
GIB = 1 << 30
DEVICES = ("cuda:0", "cuda:1")

POLICIES = (
    "lru",
    "priority_lru",       # bug-compat engine score (REQUIRED for anchor gate)
    "lfu",
    "freq_x_recency",
    "layer_lru",
    "segmented",
    "static_hotset+dyn",
    "belady",             # offline next-use bound (a.k.a. MIN/OPT)
    "min",                # alias for belady, normalized at simulate() entry
)

REQUIRED_FIELDS = (
    "forward_step", "layer", "phase", "device", "start_pos", "token_rows",
    "topk", "expert_ids_rank_order", "record_index", "run_id",
)

REUSE_CDF_CAPS = (64, 74, 128, 192, 281, 512, 682, 1024)


class SimError(RuntimeError):
    """Fail-closed violation of the replay contract."""


# ---------------------------------------------------------------------------
# Journal loading + validation (fail-closed)
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_journal(path, layer_split=DEFAULT_LAYER_SPLIT,
                 num_layers=NUM_LAYERS, topk=TOPK, num_experts=NUM_EXPERTS):
    """Load + validate one route journal. Returns (records, sha256)."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SimError(f"{path}:{lineno+1}: bad JSON: {exc}")
            records.append(rec)
    if not records:
        raise SimError(f"{path}: empty journal")

    for i, rec in enumerate(records):
        where = f"{path} record_index={rec.get('record_index')} line={i+1}"
        for f in REQUIRED_FIELDS:
            if f not in rec:
                raise SimError(f"{where}: missing field {f!r}")
        if rec["topk"] != topk:
            raise SimError(f"{where}: topk={rec['topk']} != {topk}")
        rows = rec["expert_ids_rank_order"]
        if not isinstance(rows, list) or len(rows) != rec["token_rows"]:
            raise SimError(
                f"{where}: expert_ids_rank_order rows "
                f"{len(rows) if isinstance(rows, list) else 'n/a'} "
                f"!= token_rows {rec['token_rows']}")
        for row in rows:
            if not isinstance(row, list) or len(row) != topk:
                raise SimError(f"{where}: rank row length != topk")
            for e in row:
                if not isinstance(e, int) or e < 0 or e >= num_experts:
                    raise SimError(f"{where}: expert id {e!r} out of range")
        layer = rec["layer"]
        if not isinstance(layer, int) or layer < 0 or layer >= num_layers:
            raise SimError(f"{where}: layer {layer!r} out of range")
        expected_dev = "cuda:0" if layer < layer_split else "cuda:1"
        if rec["device"] != expected_dev:
            raise SimError(
                f"{where}: device {rec['device']!r} inconsistent with "
                f"layer {layer} (split at {layer_split})")
        expected_phase = "prefill" if rec["forward_step"] == 0 else "decode"
        if rec["phase"] != expected_phase:
            raise SimError(
                f"{where}: phase {rec['phase']!r} inconsistent with "
                f"forward_step {rec['forward_step']}")

    # record_index must be contiguous 0..N-1 and file-ordered.
    for i, rec in enumerate(records):
        if rec["record_index"] != i:
            raise SimError(
                f"{path}: record_index {rec['record_index']} at file line "
                f"{i+1} breaks contiguity/order")
    # no duplicate (forward_step, layer); complete layer coverage per forward
    seen = set()
    per_fwd = defaultdict(set)
    for rec in records:
        key = (rec["forward_step"], rec["layer"])
        if key in seen:
            raise SimError(f"{path}: duplicate (forward_step,layer) {key}")
        seen.add(key)
        per_fwd[rec["forward_step"]].add(rec["layer"])
    for fwd, layers in sorted(per_fwd.items()):
        if len(layers) != num_layers or layers != set(range(num_layers)):
            raise SimError(
                f"{path}: forward {fwd} covers {len(layers)} layers, "
                f"expected {num_layers} complete")
    return records, sha256_file(path)


def calls_from_journal(records):
    """One staging call per record: unique experts in ASCENDING id order
    (engine.cpp:549-552 iterates expert ids 0..num_experts-1)."""
    calls = []
    for rec in records:
        flat = [e for row in rec["expert_ids_rank_order"] for e in row]
        experts = sorted(set(flat))
        calls.append({
            "fwd": rec["forward_step"],
            "layer": rec["layer"],
            "device": rec["device"],
            "phase": rec["phase"],
            "token_rows": rec["token_rows"],
            "raw_picks": len(flat),
            "dup_eliminated": len(flat) - len(experts),
            "experts": experts,
            "K": len(experts),
            "record_index": rec["record_index"],
        })
    return calls


def merge_cohort_records(journals):
    """Merge K per-sequence journals into one cohort record stream.

    dee-serve v0 runs an equal-length cohort in lockstep: at each
    (forward_step, layer) the engine sees ONE call whose request set is the
    flattened union of every active sequence's expert rows (the FFN
    flattens b*s rows into moe_forward_experts and engine.cpp:549-556
    dedupes).  Merging = concatenating the member records' rank rows for
    the shared key, exactly what a b=K forward emits.

    Fail-closed: every member must cover the identical (step, layer) key
    set (equal forwards x complete layers), matching the cohort contract
    (equal prompt length + fixed-length decode)."""
    if not journals:
        raise SimError("cohort merge requires >=1 journal")
    per_key = []
    keysets = []
    for j in journals:
        by_key = {}
        for rec in j["records"]:
            by_key[(rec["forward_step"], rec["layer"])] = rec
        per_key.append(by_key)
        keysets.append(set(by_key))
    base_keys = keysets[0]
    for j, ks in zip(journals[1:], keysets[1:]):
        if ks != base_keys:
            missing = sorted(base_keys - ks)[:5]
            extra = sorted(ks - base_keys)[:5]
            raise SimError(
                f"cohort member {j['label']}: (step,layer) key set "
                f"mismatch — missing {missing} extra {extra} "
                f"(cohort requires identical forward coverage)")
    merged = []
    for i, key in enumerate(
            sorted(base_keys, key=lambda k: (k[0], k[1]))):
        rows, n_rows, raw = [], 0, 0
        ref = per_key[0][key]
        for by_key in per_key:
            rec = by_key[key]
            rows.extend(rec["expert_ids_rank_order"])
            n_rows += rec["token_rows"]
            raw += sum(len(r) for r in rec["expert_ids_rank_order"])
        merged.append({
            "forward_step": key[0],
            "layer": key[1],
            "phase": ref["phase"],
            "device": ref["device"],
            "start_pos": ref["start_pos"],
            "token_rows": n_rows,
            "topk": ref["topk"],
            "expert_ids_rank_order": rows,
            "record_index": i,
            "run_id": f"cohort({'+'.join(j['label'] for j in journals)})",
        })
    return merged


# ---------------------------------------------------------------------------
# Host pack cache — faithful HostPackCache::get_batch replay
# ---------------------------------------------------------------------------

class HostLru:
    """Byte-bounded LRU of fixed-size records replicating get_batch().
    OrderedDict od: position 0 = LRU victim end, last = MRU
    (C++ lru_.push_front(k) == od[k]=None append at end; lru_.back() ==
    first item)."""

    def __init__(self, capacity_records):
        self.cap = capacity_records
        self.od = OrderedDict()          # key -> None ; [0]=LRU .. [-1]=MRU
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.consults = 0
        self.in_batch_dup_hits = 0

    def get_batch(self, keys, next_uses=None):
        """Counted batched consult (host_pack_cache.cpp:170-238).
        Returns list[(key, 'hit'|'miss')] in request order.
        next_uses is accepted for interface parity with HostBelady and
        ignored by LRU."""
        batch_set = set(keys)
        outcomes = [None] * len(keys)
        miss_keys = []
        # Phase 1: scan in request order.
        for idx, k in enumerate(keys):
            dup = False
            for j in range(idx):
                if keys[j] == k:
                    dup = True
                    break
            if dup:
                self.hits += 1
                self.in_batch_dup_hits += 1
                self.consults += 1
                outcomes[idx] = (k, "hit")
                continue
            if k in self.od:
                self.od.move_to_end(k)      # hit -> MRU front
                self.hits += 1
                outcomes[idx] = (k, "hit")
            else:
                miss_keys.append(k)
                self.misses += 1
                outcomes[idx] = (k, "miss")
            self.consults += 1
        # Phase 2: evict LRU-back skipping in-batch keys until misses fit.
        additional = len(miss_keys)
        if additional > self.cap:
            raise SimError(
                f"host get_batch infeasible: {additional} records > "
                f"capacity {self.cap} (engine get_batch would fail the "
                f"forward; pick a budget >= one sub-batch)")
        while len(self.od) + additional > self.cap:
            victim = None
            for k in self.od:               # front -> back = LRU -> MRU
                if k not in batch_set:
                    victim = k
                    break
            if victim is None:
                raise SimError("host evict: no non-batch victim")
            del self.od[victim]
            self.evictions += 1
        # Phase 3: admit unique misses at MRU-front in request order,
        # BEFORE fills run (host_pack_cache.cpp:240-266).
        for k in miss_keys:
            self.od[k] = None               # appended = MRU
        return outcomes

    def refresh_if_present(self, key):
        """Uncounted get_if_present (stage-time LRU refresh)."""
        if key in self.od:
            self.od.move_to_end(key)
            return True
        return False

    def resident(self, key):
        return key in self.od


class HostBelady:
    """Offline-optimal (MIN) host tier on a known consult stream.  Same
    interface as HostLru; each get_batch consults the keys with their
    precomputed next-use ordinals.  Eviction chooses the resident whose next
    consult is farthest out (+inf = never)."""

    def __init__(self, capacity_records):
        self.cap = capacity_records
        self.res = {}                  # key -> next_use
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.consults = 0
        self.in_batch_dup_hits = 0

    def get_batch(self, keys, next_uses):
        outcomes = [None] * len(keys)
        for idx, k in enumerate(keys):
            self.consults += 1
            nu = next_uses[idx]
            if k in self.res:
                self.hits += 1
                self.res[k] = nu
                outcomes[idx] = (k, "hit")
                continue
            self.misses += 1
            outcomes[idx] = (k, "miss")
            if len(self.res) >= self.cap:
                victim = max(self.res, key=lambda kk: (self.res[kk], kk))
                if nu > self.res[victim]:
                    continue      # incoming farthest: MIN declines to cache
                del self.res[victim]
                self.evictions += 1
            self.res[k] = nu
        return outcomes

    def refresh_if_present(self, key):
        # bound ignores ordering; presence is what the stage path needs
        return key in self.res

    def resident(self, key):
        return key in self.res

    @property
    def od(self):                      # for entries_end reporting
        return self.res


def next_use_array(stream):
    """next_use[i] = index of next occurrence of stream[i], else len+1."""
    INF = len(stream) + 1
    nxt = [INF] * len(stream)
    last = {}
    for i in range(len(stream) - 1, -1, -1):
        k = stream[i]
        nxt[i] = last.get(k, INF)
        last[k] = i
    return nxt


def belady_run(stream, cap, initial=()):
    """Offline bound over a request/consult stream. `initial` = declared
    pre-resident keys (prewarm); they start resident with next_use=-1 so
    belady can still evict them first if it chooses — same initial state as
    the policy under test."""
    if cap <= 0:
        return 0
    nxt = next_use_array(stream)
    res = {k: -1 for k in initial}
    if len(res) > cap:
        raise SimError("belady bound: prewarm set exceeds capacity")
    hits = 0
    for t, k in enumerate(stream):
        if k in res:
            hits += 1
            res[k] = nxt[t]
            continue
        if len(res) >= cap:
            victim = max(res, key=lambda kk: (res[kk], kk))
            if nxt[t] > res[victim]:
                continue          # incoming is farthest: don't cache it
            del res[victim]
        res[k] = nxt[t]
    return hits


# ---------------------------------------------------------------------------
# VRAM cache — chunked, pin-aware, multi-policy replay
# ---------------------------------------------------------------------------

class Block:
    __slots__ = ("key", "last_used", "priority", "pins", "freq", "seg")

    def __init__(self, key):
        self.key = key
        self.last_used = 0
        self.priority = 0
        self.pins = 0
        self.freq = 0
        self.seg = 0          # segmented: 0=probation 1=protected


class VramSim:
    """One device's VramCacheManager replay. Uniform-size blocks; capacity is
    floor(budget/blob) slots. Pins model in-flight chunk transfers: each
    staged member holds pins=1 until its chunk's consume phase."""

    FLAT = ("lru", "priority_lru", "lfu", "freq_x_recency")

    def __init__(self, cap_slots, policy, layer_caps=None, static_keys=None,
                 prob_frac=0.2, freq_weight=PRIORITY_WEIGHT):
        self.cap = cap_slots
        self.policy = policy
        self.freq_weight = freq_weight
        self.blocks = {}                 # key -> Block  (flat policies)
        self.tick = 0
        self.ensures = 0
        self.hits = 0
        self.cold = 0
        self.evictions = 0
        self.pinned_skipped = 0
        self.inflight = 0                # always 0 on this stream (checked)
        # layer_lru
        self.layer_caps = layer_caps     # {layer: cap} or None
        self.layer_od = defaultdict(OrderedDict)
        self.layer_of_key = {}
        # segmented SLRU
        self.prob_cap = (max(1, int(round(cap_slots * prob_frac)))
                         if cap_slots else 0)
        self.prot_cap = (cap_slots - self.prob_cap) if cap_slots else 0
        self.prob_od = OrderedDict()
        self.prot_od = OrderedDict()
        self.seg_of_key = {}
        # static_hotset+dyn
        self.static_keys = static_keys or set()
        self.static_resident = set()
        self.dyn_od = OrderedDict()
        self.dyn_cap = max(0, cap_slots - len(self.static_keys))
        # belady
        self.belady_res = {}             # key -> next use stream index
        # initial-state bookkeeping for prewarm / dominance comparisons
        self.prewarmed = set()

    def resident_count(self):
        if self.policy == "layer_lru":
            return len(self.layer_of_key)
        if self.policy == "segmented":
            return len(self.seg_of_key)
        if self.policy == "static_hotset+dyn":
            return len(self.static_resident) + len(self.dyn_od)
        if self.policy == "belady":
            return len(self.belady_res)
        return len(self.blocks)

    def pin(self, key, n=1):
        if self.policy == "belady":
            return                     # the bound ignores pins
        if self.policy in self.FLAT:
            self.blocks[key].pins += n
        elif self.policy == "layer_lru":
            self.layer_of_key[key].pins += n
        elif self.policy == "segmented":
            self.seg_of_key[key].pins += n
        elif self.policy == "static_hotset+dyn":
            if key in self.dyn_od:
                self.dyn_od[key].pins += n
            # static keys are never evicted; pins irrelevant
        else:
            raise SimError(f"pin: unknown policy {self.policy}")

    def unpin(self, key):
        self.pin(key, -1)

    # ---- flat policies ------------------------------------------------------
    def _score(self, b):
        if self.policy == "lru":
            return b.last_used
        if self.policy == "priority_lru":
            return b.last_used + b.priority * PRIORITY_WEIGHT
        if self.policy == "lfu":
            return (b.freq, b.last_used)      # freq, then recency tiebreak
        if self.policy == "freq_x_recency":
            return b.last_used + b.freq * self.freq_weight
        raise SimError(f"score: bad policy {self.policy}")

    def _evict_one_flat(self):
        victim = None
        vscore = None
        for k, b in self.blocks.items():
            if b.pins != 0:
                self.pinned_skipped += 1
                continue
            s = self._score(b)
            if victim is None or s < vscore:
                victim, vscore = k, s
        if victim is None:
            raise SimError(
                f"{self.policy}: evict_until_free found no evictable victim "
                f"(all {len(self.blocks)} resident blocks pinned)")
        del self.blocks[victim]
        self.evictions += 1
        return victim

    def _ensure_flat(self, key, prio):
        b = self.blocks.get(key)
        if b is not None:
            self.tick += 1
            b.last_used = self.tick
            b.priority = prio
            b.freq += 1
            b.pins += 1
            self.hits += 1
            return "hit"
        while len(self.blocks) >= self.cap:
            self._evict_one_flat()
        self.tick += 1
        nb = Block(key)
        nb.last_used = self.tick
        nb.priority = prio
        nb.freq = 1
        nb.pins = 1
        self.blocks[key] = nb
        self.cold += 1
        return "cold"

    # ---- layer_lru -----------------------------------------------------------
    # Per-layer LRU partitions (cap_slots split across the device's layers,
    # earlier layers get the remainder).  Partition budgets are eviction
    # *preferences*: when the calling layer's partition is fully pinned by
    # in-flight staging, the victim is stolen from the globally-least-recent
    # unpinned block of other partitions ("spill").  A layer may therefore
    # transiently exceed its share while a chunk is in flight — the global
    # slot cap is always enforced, and pins are never evicted.
    def _ensure_layer(self, key, prio):
        layer = key[0]
        od = self.layer_od[layer]
        b = od.get(key)
        if b is not None:
            self.tick += 1
            b.last_used = self.tick
            b.priority = prio
            b.freq += 1
            b.pins += 1
            od.move_to_end(key)
            self.hits += 1
            return "hit"
        # Evict only under GLOBAL capacity pressure.  Victim preference:
        # (1) the calling layer's partition if it is over its share,
        # (2) any other over-share partition's LRU,
        # (3) the globally-least-recent unpinned block.  Partitions may
        # transiently exceed their share while their members are pinned by
        # an in-flight chunk; the global slot cap is always enforced.
        while len(self.layer_of_key) >= self.cap:
            victim = None
            od_self = self.layer_od[layer]
            if len(od_self) > self.layer_caps[layer]:
                for k, bb in od_self.items():
                    if bb.pins != 0:
                        self.pinned_skipped += 1
                        continue
                    victim = k
                    break
            if victim is None:
                for l2 in sorted(self.layer_od):
                    od2 = self.layer_od[l2]
                    if l2 == layer or len(od2) <= self.layer_caps[l2]:
                        continue
                    for k, bb in od2.items():
                        if bb.pins == 0:
                            victim = k
                            break
                    if victim is not None:
                        break
            if victim is None:
                vscore = None
                for k2, bb in self.layer_of_key.items():
                    if bb.pins != 0:
                        self.pinned_skipped += 1
                        continue
                    if victim is None or bb.last_used < vscore:
                        victim, vscore = k2, bb.last_used
            if victim is None:
                raise SimError(
                    "layer_lru: no evictable victim anywhere "
                    f"(all {len(self.layer_of_key)} resident blocks pinned)")
            self.layer_od[victim[0]].pop(victim)
            del self.layer_of_key[victim]
            self.evictions += 1
        self.tick += 1
        nb = Block(key)
        nb.last_used = self.tick
        nb.priority = prio
        nb.freq = 1
        nb.pins = 1
        od[key] = nb
        self.layer_of_key[key] = nb
        self.cold += 1
        return "cold"

    # ---- segmented (SLRU: probationary + protected) ----------------------------
    # Segment sizes are eviction preferences under the pin contract: a whole
    # in-flight chunk can pin more members than the probationary segment, so
    # probation may transiently exceed its target until pins release; the
    # global slot cap is always enforced and pinned blocks are never evicted.
    def _ensure_segmented(self, key, prio):
        self.tick += 1
        b = self.seg_of_key.get(key)
        if b is not None:
            b.last_used = self.tick
            b.priority = prio
            b.freq += 1
            b.pins += 1
            self.hits += 1
            if b.seg == 0:
                del self.prob_od[key]
                b.seg = 1
                self.prot_od[key] = b
                self._seg_rebalance()
            else:
                self.prot_od.move_to_end(key)
            return "hit"
        while len(self.seg_of_key) >= self.cap:
            self._seg_evict_one()
        nb = Block(key)
        nb.last_used = self.tick
        nb.priority = prio
        nb.freq = 1
        nb.pins = 1
        nb.seg = 0
        self.prob_od[key] = nb
        self.seg_of_key[key] = nb
        self._seg_rebalance()
        self.cold += 1
        return "cold"

    def _seg_evict_one(self):
        for od in (self.prob_od, self.prot_od):
            victim = None
            for k, bb in od.items():
                if bb.pins != 0:
                    self.pinned_skipped += 1
                    continue
                victim = k
                break
            if victim is not None:
                del od[victim]
                del self.seg_of_key[victim]
                self.evictions += 1
                return
        raise SimError("segmented: no evictable victim (all pinned)")

    def _seg_rebalance(self):
        while len(self.prot_od) > self.prot_cap:
            k, b = self.prot_od.popitem(last=False)   # protected LRU demotes
            b.seg = 0
            self.prob_od[k] = b
        while len(self.prob_od) > self.prob_cap:
            # prefer evicting probationary LRU; if every probationary member
            # is pinned by the in-flight chunk, let probation overflow
            # transiently instead of evicting a pinned block
            for k, bb in self.prob_od.items():
                if bb.pins == 0:
                    del self.prob_od[k]
                    del self.seg_of_key[k]
                    self.evictions += 1
                    break
            else:
                break

    # ---- static_hotset+dyn -----------------------------------------------------
    def _ensure_static(self, key, prio):
        self.tick += 1
        if key in self.static_keys:
            if key in self.static_resident:
                self.hits += 1
                return "hit"
            if len(self.static_resident) >= len(self.static_keys):
                raise SimError("static partition overflow")
            self.static_resident.add(key)
            self.cold += 1
            return "cold"
        b = self.dyn_od.get(key)
        if b is not None:
            b.last_used = self.tick
            b.priority = prio
            b.freq += 1
            b.pins += 1
            self.dyn_od.move_to_end(key)
            self.hits += 1
            return "hit"
        # dynamic evicts its own LRU; the hard bound is the GLOBAL slot cap
        # (dyn may borrow slots the static partition has not filled yet).
        # If every dynamic member is pinned by the in-flight chunk the
        # configuration is genuinely infeasible — fail closed.
        while len(self.static_resident) + len(self.dyn_od) >= self.cap:
            victim = None
            for k, bb in self.dyn_od.items():
                if bb.pins != 0:
                    self.pinned_skipped += 1
                    continue
                victim = k
                break
            if victim is None:
                raise SimError(
                    "static_hotset+dyn: no evictable dyn victim (static "
                    f"partition {len(self.static_resident)}/"
                    f"{len(self.static_keys)} + all {len(self.dyn_od)} dyn "
                    "pinned) — shrink the static set")
            del self.dyn_od[victim]
            self.evictions += 1
        nb = Block(key)
        nb.last_used = self.tick
        nb.priority = prio
        nb.freq = 1
        nb.pins = 1
        self.dyn_od[key] = nb
        self.cold += 1
        return "cold"

    # ---- belady (offline bound) -----------------------------------------------
    def _ensure_belady(self, key, next_use):
        if key in self.belady_res:
            self.hits += 1
            self.belady_res[key] = next_use
            return "hit"
        if len(self.belady_res) >= self.cap:
            victim = max(self.belady_res,
                         key=lambda kk: (self.belady_res[kk], kk))
            if next_use > self.belady_res[victim]:
                # incoming key's next use is farthest of all: MIN declines
                # to cache it (miss, no eviction, no insertion)
                self.cold += 1
                return "cold"
            del self.belady_res[victim]
            self.evictions += 1
        self.belady_res[key] = next_use
        self.cold += 1
        return "cold"

    # ---- generic ensure ---------------------------------------------------------
    def ensure(self, key, prio, next_use=None):
        """One ensure+pin (prefetch_impl). Returns 'hit'|'cold'."""
        self.ensures += 1
        if self.policy == "belady":
            return self._ensure_belady(key, next_use)
        if self.policy == "layer_lru":
            return self._ensure_layer(key, prio)
        if self.policy == "segmented":
            return self._ensure_segmented(key, prio)
        if self.policy == "static_hotset+dyn":
            return self._ensure_static(key, prio)
        return self._ensure_flat(key, prio)

    # ---- prewarm ------------------------------------------------------------------
    def install_prewarm(self, keys):
        """Declared initial state (regime=prewarm). Blocks sit at
        last_used=0 / pins=0; cost is reported by the caller."""
        for key in keys:
            if self.policy == "belady":
                self.belady_res.setdefault(key, -1)
            elif self.policy == "layer_lru":
                nb = Block(key)
                self.layer_od[key[0]][key] = nb
                self.layer_of_key[key] = nb
            elif self.policy == "segmented":
                nb = Block(key)
                self.prot_od[key] = nb
                self.seg_of_key[key] = nb
            elif self.policy == "static_hotset+dyn":
                if key in self.static_keys:
                    self.static_resident.add(key)
                else:
                    self.dyn_od[key] = Block(key)
            else:
                self.blocks[key] = Block(key)
            self.prewarmed.add(key)
            if self.resident_count() > self.cap:
                raise SimError("prewarm set exceeds VRAM capacity")


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def make_vram(cfg, cap_slots, dev_layer_list, static_keys):
    layer_caps = None
    if cfg["vram_policy"] == "layer_lru":
        n = len(dev_layer_list)
        base = cap_slots // n if n else cap_slots
        rem = cap_slots - base * n if n else 0
        layer_caps = {}
        for idx, layer in enumerate(sorted(dev_layer_list)):
            layer_caps[layer] = base + (1 if idx < rem else 0)
    return VramSim(cap_slots, cfg["vram_policy"], layer_caps=layer_caps,
                   static_keys=set(static_keys),
                   prob_frac=cfg.get("segmented_probation", 0.2),
                   freq_weight=cfg.get("freq_weight", PRIORITY_WEIGHT))


def resolve_key_set(spec, journals, devices, cfg):
    """file path ('layer,expert' lines) or auto:FRAC (top floor(cap*FRAC)
    keys per device by run-stream frequency — a declared offline config)."""
    out = {d: set() for d in devices}
    if spec.startswith("auto:"):
        frac = float(spec.split(":", 1)[1])
        blob = (FP16_BLOB_BYTES if cfg["vram_dtype"] == "fp16"
                else FP4_RECORD_BYTES)
        cap = int(cfg["vram_bytes"] // blob)
        want = max(1, int(cap * frac))
        freq = {d: defaultdict(int) for d in devices}
        for j in journals:
            for call in j["calls"]:
                for e in call["experts"]:
                    freq[call["device"]][(call["layer"], e)] += 1
        for d in devices:
            top = sorted(freq[d].items(),
                         key=lambda kv: (-kv[1], kv[0]))[:want]
            out[d] = {k for k, _ in top}
        return out
    path = spec.split(":", 1)[1] if spec.startswith("file:") else spec
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.replace(":", ",").split(",")[:2]
            layer, expert = int(a), int(b)
            d = "cuda:0" if layer < cfg["layer_split"] else "cuda:1"
            out[d].add((layer, expert))
    return out


def simulate(cfg, journals):
    """Run one config over the journals.
    Returns (rows, fwd_rows, meta)."""
    cfg = dict(cfg)
    if cfg.get("vram_policy") == "min":
        cfg["vram_policy"] = "belady"      # MIN == Belady's algorithm
    devices = DEVICES
    dev_layers = {
        "cuda:0": [l for l in range(cfg["num_layers"])
                   if l < cfg["layer_split"]],
        "cuda:1": [l for l in range(cfg["num_layers"])
                   if l >= cfg["layer_split"]],
    }

    blob = FP16_BLOB_BYTES if cfg["vram_dtype"] == "fp16" else FP4_RECORD_BYTES
    cap_slots = int(cfg["vram_bytes"] // blob)
    host_caps = int(cfg["host_bytes"] // FP4_RECORD_BYTES)
    cache_batch = max(1, cap_slots)
    queue_depth = max(1, cfg["queue_depth"])
    bypass = cfg["host_bytes"] <= 0

    # ---- declared sets -------------------------------------------------------
    static_set = {d: set() for d in devices}
    if cfg["static_hotset"] is not None:
        static_set = resolve_key_set(cfg["static_hotset"], journals,
                                     devices, cfg)
    prewarm_set = {d: set() for d in devices}
    if cfg["regime"] == "prewarm":
        if cfg["prewarm"] is not None:
            prewarm_set = resolve_key_set(cfg["prewarm"], journals,
                                          devices, cfg)
        elif cfg["vram_policy"] == "static_hotset+dyn":
            prewarm_set = static_set
        else:
            raise SimError(
                "regime=prewarm requires a declared initial state: "
                "--prewarm <file|auto:FRAC> or --vram-policy "
                "static_hotset+dyn --static-hotset ...")

    # ---- per-device request streams for belady next-use -----------------------
    concat = {d: [] for d in devices}
    part_bounds = {d: [] for d in devices}
    for p_idx, j in enumerate(journals):
        for d in devices:
            start = len(concat[d])
            for c in j["calls"]:
                if c["device"] == d:
                    concat[d].extend((c["layer"], e) for e in c["experts"])
            part_bounds[d].append((start, len(concat[d])))
    run_next = {d: next_use_array(concat[d]) for d in devices}
    part_next = {d: [next_use_array(concat[d][s:e])
                     for (s, e) in part_bounds[d]] for d in devices}

    def vram_next_use(d, p_idx, run_ord, prompt_ord):
        if cfg["persist"]:
            return run_next[d][run_ord]
        return part_next[d][p_idx][prompt_ord]

    # ---- host belady consult streams (eager consult order) ---------------------
    # eager consults every requested key in chunk order; the per-scope consult
    # stream is therefore known a priori.  For per-device scope it equals the
    # device request stream; for pooled it is the record-order interleave.
    if cfg["host_policy"] == "belady" and cfg["host_consult"] == "lazy":
        raise SimError(
            "host_policy=belady requires eager consult (the lazy consult "
            "stream depends on the vram policy and is not known a priori); "
            "the lazy-mode host bound is still reported in metadata")
    hstream_parts = {"pooled": []}
    pooled_bounds = []
    for p_idx, j in enumerate(journals):
        s = len(hstream_parts["pooled"])
        for c in j["calls"]:
            hstream_parts["pooled"].extend(
                (c["layer"], e) for e in c["experts"])
        pooled_bounds.append((s, len(hstream_parts["pooled"])))
    hscope_stream = {
        "cuda:0": concat["cuda:0"], "cuda:1": concat["cuda:1"],
        "pooled": hstream_parts["pooled"],
    }
    hscope_bounds = {
        "cuda:0": part_bounds["cuda:0"], "cuda:1": part_bounds["cuda:1"],
        "pooled": pooled_bounds,
    }
    hrun_next = {s: next_use_array(hscope_stream[s]) for s in hscope_stream}
    hpart_next = {
        s: [next_use_array(hscope_stream[s][a:b])
            for (a, b) in hscope_bounds[s]] for s in hscope_stream}

    def host_next_use(scope, p_idx, run_ord, prompt_ord):
        if cfg["persist"]:
            return hrun_next[scope][run_ord]
        return hpart_next[scope][p_idx][prompt_ord]

    # ---- mutable cache state ---------------------------------------------------
    host_cls = HostBelady if cfg["host_policy"] == "belady" else HostLru

    def fresh_state():
        v, h, vs = {}, {}, {}
        for d in devices:
            v[d] = make_vram(cfg, cap_slots, dev_layers[d], static_set[d])
            h[d] = host_cls(host_caps)
            vs[d] = VramSim(cap_slots, "belady")  # dominance shadow
            if prewarm_set[d]:
                v[d].install_prewarm(prewarm_set[d])
                vs[d].install_prewarm(prewarm_set[d])
                for k in sorted(prewarm_set[d]):
                    h[d].od.setdefault(k, None)
        pooled = (host_cls(host_caps)
                  if cfg["host_scope"] == "pooled" else None)
        if pooled is not None:
            for d in devices:
                for k in sorted(prewarm_set[d]):
                    pooled.od.setdefault(k, None)
        return v, h, vs, pooled

    vram, host, vram_shadow, pooled_host = fresh_state()

    def host_for(d):
        return pooled_host if pooled_host is not None else host[d]

    # ---- accumulators -------------------------------------------------------------
    totals = {}
    per_fwd = defaultdict(lambda: defaultdict(int))
    per_layer_req = defaultdict(int)
    per_layer_hit = defaultdict(int)
    # consult streams per cache scope, tagged with prompt idx for bounds
    consult_stream = {"pooled": [], "cuda:0": [], "cuda:1": []}
    consult_prompt = {"pooled": [], "cuda:0": [], "cuda:1": []}
    shadow_prompt_hits = defaultdict(int)
    # first-touch sets: scoped to cache lifetime
    vram_seen_run = {d: set() for d in devices}
    host_seen_run = {"pooled": set(), "cuda:0": set(), "cuda:1": set()}

    def get_tot(p_idx, d):
        return totals.setdefault((p_idx, d), defaultdict(int))

    for p_idx, j in enumerate(journals):
        label = j["label"]
        if not cfg["persist"] and p_idx > 0:
            vram, host, vram_shadow, pooled_host = fresh_state()
            vram_seen_run = {d: set() for d in devices}
            host_seen_run = {"pooled": set(), "cuda:0": set(), "cuda:1": set()}
        n_forwards = len({c["fwd"] for c in j["calls"]})
        run_ord = {d: part_bounds[d][p_idx][0] for d in devices}
        prompt_ord = {d: 0 for d in devices}
        hrun_ord = {s: hscope_bounds[s][p_idx][0]
                    for s in ("pooled", "cuda:0", "cuda:1")}
        hprompt_ord = {s: 0 for s in hrun_ord}
        seen_in_prompt = {d: set() for d in devices}
        last_seen_idx = {d: {} for d in devices}
        reuse_dists = {d: [] for d in devices}

        for call in j["calls"]:
            d = call["device"]
            vc = vram[d]
            sc = vram_shadow[d]
            hc = host_for(d)
            scope_key = "pooled" if pooled_host is not None else d
            layer = call["layer"]
            active = call["experts"]
            K = call["K"]
            phase = call["phase"]
            fwd = call["fwd"]
            c = get_tot(p_idx, d)
            cp = get_tot(p_idx, "pooled")
            fc = per_fwd[(p_idx, fwd, d)]
            fcp = per_fwd[(p_idx, fwd, "pooled")]
            ev0 = vc.evictions
            ps0 = vc.pinned_skipped
            for cc in (c, cp, fc, fcp):
                cc["requests"] += K
                cc[f"requests_{phase}"] += K
                cc["raw_picks"] += call["raw_picks"]
                cc["dup_eliminated"] += call["dup_eliminated"]
            per_layer_req[(p_idx, d, layer)] += K

            for cstart in range(0, K, cache_batch):
                chunk = active[cstart:cstart + cache_batch]
                chunk_keys = [(layer, e) for e in chunk]

                # (a) HOST phase — eager consult in queue_depth sub-batches
                host_outcome = {}
                if not bypass and cfg["host_consult"] == "eager":
                    for sub in batched(chunk_keys, queue_depth):
                        nsub = len(sub)
                        if cfg["host_policy"] == "belady":
                            if cfg["persist"]:
                                o = hrun_ord[scope_key]
                                nu = hrun_next[scope_key][o:o + nsub]
                                hrun_ord[scope_key] += nsub
                            else:
                                o = hprompt_ord[scope_key]
                                nu = hpart_next[scope_key][p_idx][o:o + nsub]
                                hprompt_ord[scope_key] += nsub
                        else:
                            nu = None
                            hrun_ord[scope_key] += nsub
                            hprompt_ord[scope_key] += nsub
                        before = (hc.hits, hc.misses, hc.evictions)
                        outcomes = hc.get_batch(sub, nu)
                        dh = hc.hits - before[0]
                        dm = hc.misses - before[1]
                        de = hc.evictions - before[2]
                        for k, oc in outcomes:
                            host_outcome[k] = oc
                            if (oc == "hit"
                                    and k not in host_seen_run[scope_key]
                                    and k not in
                                    (prewarm_set.get(scope_key) or
                                     prewarm_set[d]) ):
                                raise SimError(
                                    f"{label} {d}: cold-regime first-touch "
                                    f"host hit on {k}")
                            host_seen_run[scope_key].add(k)
                            consult_stream[scope_key].append(k)
                            consult_prompt[scope_key].append(p_idx)
                        for cc in (c, cp):
                            cc["host_hits"] += dh
                            cc["host_misses"] += dm
                            cc["host_evictions"] += de
                            cc[f"host_hits_{phase}"] += dh
                            cc[f"host_misses_{phase}"] += dm
                        fc["host_hits"] += dh
                        fc["host_misses"] += dm
                        fc["host_evictions"] += de
                        fcp["host_hits"] += dh
                        fcp["host_misses"] += dm
                        fcp["host_evictions"] += de

                # (b) VRAM stage phase
                staged = []
                for jx, key in enumerate(chunk_keys):
                    i = cstart + jx
                    prio = K - i
                    if not bypass and cfg["host_consult"] == "eager":
                        present = hc.refresh_if_present(key)
                        if not present and cfg["host_policy"] == "lru":
                            raise SimError(
                                f"{label} {d}: stage consult for {key} "
                                f"found no pack entry (engine would fail "
                                f"the forward — host budget too small for "
                                f"the consult window)")
                    nu = vram_next_use(d, p_idx, run_ord[d], prompt_ord[d])
                    outcome = vc.ensure(key, prio, next_use=nu)
                    sb = sc.hits
                    sc.ensure(key, 0, next_use=nu)
                    if sc.hits > sb:
                        shadow_prompt_hits[(p_idx, d)] += 1
                    run_ord[d] += 1
                    prompt_ord[d] += 1
                    staged.append(key)
                    # stream-property counters
                    first_in_prompt = key not in seen_in_prompt[d]
                    if first_in_prompt:
                        seen_in_prompt[d].add(key)
                        c["first_touch_requests"] += 1
                        cp["first_touch_requests"] += 1
                    else:
                        c["repeat_requests"] += 1
                        cp["repeat_requests"] += 1
                    li = last_seen_idx[d].get(key)
                    if li is not None:
                        reuse_dists[d].append(prompt_ord[d] - 1 - li)
                    last_seen_idx[d][key] = prompt_ord[d] - 1
                    # cold-regime first-touch check (vram)
                    if (outcome == "hit"
                            and key not in vram_seen_run[d]
                            and key not in vc.prewarmed):
                        raise SimError(
                            f"{label} {d}: cold-regime first-touch VRAM "
                            f"hit on {key}")
                    vram_seen_run[d].add(key)
                    if outcome == "hit":
                        for cc in (c, cp, fc, fcp):
                            cc["vram_hits"] += 1
                            cc[f"vram_hits_{phase}"] += 1
                        per_layer_hit[(p_idx, d, layer)] += 1
                        if key in vc.prewarmed:
                            c["hits_on_prewarmed"] += 1
                            cp["hits_on_prewarmed"] += 1
                    else:
                        for cc in (c, cp, fc, fcp):
                            cc["vram_cold"] += 1
                            cc[f"vram_cold_{phase}"] += 1
                    # joint outcome pairing (ensure() returns hit|cold)
                    if cfg["host_consult"] == "eager" and not bypass:
                        ho = host_outcome.get(key)
                        jkey = ("joint_vram_" +
                                ("hit" if outcome == "hit" else "miss") +
                                "_host_" + (ho or "none"))
                        c[jkey] += 1
                        cp[jkey] += 1
                        if outcome == "cold":
                            if ho == "hit":
                                c["vram_miss_host_hit"] += 1
                                cp["vram_miss_host_hit"] += 1
                            else:
                                c["vram_miss_host_miss"] += 1
                                cp["vram_miss_host_miss"] += 1
                    elif not bypass and cfg["host_consult"] == "lazy":
                        if outcome == "cold":
                            before = (hc.hits, hc.misses, hc.evictions)
                            outcomes = hc.get_batch([key])
                            dh = hc.hits - before[0]
                            dm = hc.misses - before[1]
                            de = hc.evictions - before[2]
                            ho = outcomes[0][1]
                            if (ho == "hit"
                                    and key not in host_seen_run[scope_key]
                                    and key not in prewarm_set[d]):
                                raise SimError(
                                    f"{label} {d}: cold-regime first-touch "
                                    f"host hit on {key}")
                            host_seen_run[scope_key].add(key)
                            consult_stream[scope_key].append(key)
                            consult_prompt[scope_key].append(p_idx)
                            for cc in (c, cp):
                                cc["host_hits"] += dh
                                cc["host_misses"] += dm
                                cc["host_evictions"] += de
                                cc[f"host_hits_{phase}"] += dh
                                cc[f"host_misses_{phase}"] += dm
                            fc["host_hits"] += dh
                            fc["host_misses"] += dm
                            fc["host_evictions"] += de
                            fcp["host_hits"] += dh
                            fcp["host_misses"] += dm
                            fcp["host_evictions"] += de
                            if dm:
                                c["vram_miss_host_miss"] += 1
                                cp["vram_miss_host_miss"] += 1
                            else:
                                c["vram_miss_host_hit"] += 1
                                cp["vram_miss_host_hit"] += 1
                # (c) consume — release this chunk's pins
                for key in staged:
                    vc.unpin(key)
            # per-call deltas into per-forward counters
            fc["vram_evictions"] += vc.evictions - ev0
            fc["vram_pinned_skipped"] += vc.pinned_skipped - ps0
            fcp["vram_evictions"] += vc.evictions - ev0

        # ---- end-of-prompt snapshots ------------------------------------------
        for d in devices:
            c = totals[(p_idx, d)]
            c["vram_evictions"] = vram[d].evictions      # cumulative; fixed below
            c["vram_pinned_skipped"] = vram[d].pinned_skipped
            c["vram_resident_end"] = vram[d].resident_count()
            c["vram_inflight"] = vram[d].inflight
            c["host_entries_end"] = len(host[d].od)
            c["shadow_vram_hits"] = shadow_prompt_hits[(p_idx, d)]
            c["n_forwards"] = n_forwards
            dists = sorted(reuse_dists[d])
            c["reuse_n"] = len(dists)
            if dists:
                n = len(dists)
                c["reuse_p50"] = dists[n // 2]
                c["reuse_p90"] = dists[int(n * 0.9)]
                c["reuse_p99"] = dists[int(n * 0.99)]
                c["reuse_max"] = dists[-1]
                c["reuse_cdf"] = {cap_: sum(1 for x in dists if x <= cap_) / n
                                  for cap_ in REUSE_CDF_CAPS}
        cp = totals[(p_idx, "pooled")]
        cp["vram_evictions"] = sum(vram[d].evictions for d in devices)
        cp["vram_resident_end"] = sum(vram[d].resident_count()
                                      for d in devices)
        cp["vram_inflight"] = 0
        cp["n_forwards"] = n_forwards
        cp["shadow_vram_hits"] = sum(shadow_prompt_hits[(p_idx, d)]
                                     for d in devices)
        if pooled_host is not None:
            cp["host_entries_end"] = len(pooled_host.od)
        else:
            cp["host_entries_end"] = sum(len(host[d].od) for d in devices)

    # ---- convert cumulative snapshot counters to per-prompt deltas ------------
    if cfg["persist"]:
        for d in devices:
            prev_e = prev_s = 0
            for p_idx in range(len(journals)):
                c = totals[(p_idx, d)]
                ce, cs = c["vram_evictions"], c["vram_pinned_skipped"]
                c["vram_evictions"] = ce - prev_e
                c["vram_pinned_skipped"] = cs - prev_s
                prev_e, prev_s = ce, cs
        prev_e = 0
        for p_idx in range(len(journals)):
            cp = totals[(p_idx, "pooled")]
            ce = cp["vram_evictions"]
            cp["vram_evictions"] = ce - prev_e
            prev_e = ce

    # ---- host-tier belady bound on the recorded consult streams -----------------
    bounds = {}
    for scope in ("pooled", "cuda:0", "cuda:1"):
        stream = consult_stream[scope]
        if not stream:
            continue
        if cfg["persist"]:
            seg = [stream]
            seg_keys = [(scope, None)]
        else:
            seg = []
            seg_keys = []
            for p_idx in range(len(journals)):
                part = [k for k, p in zip(stream, consult_prompt[scope])
                        if p == p_idx]
                seg.append(part)
                seg_keys.append((scope, p_idx))
        for (sk, pi), part in zip(seg_keys, seg):
            if sk == "pooled":
                initial = prewarm_set["cuda:0"] | prewarm_set["cuda:1"]
            else:
                initial = prewarm_set[sk]
            hits_bound = belady_run(part, host_caps, initial=initial)
            bounds[f"host:{sk}:p{pi if pi is not None else 'all'}"] = hits_bound
            if cfg["persist"]:
                pol = sum(totals[(p, sk)]["host_hits"]
                          for p in range(len(journals)))
            else:
                pol = totals[(pi, sk)]["host_hits"]
            if pol > hits_bound:
                raise SimError(
                    f"DOMINANCE VIOLATION host {sk} p{pi}: policy hits "
                    f"{pol} > belady bound {hits_bound}")

    # ---- vram dominance tripwire ---------------------------------------------------
    if cfg["persist"]:
        for d in devices:
            pol = sum(totals[(p, d)]["vram_hits"]
                      for p in range(len(journals)))
            bnd = sum(totals[(p, d)]["shadow_vram_hits"]
                      for p in range(len(journals)))
            bounds[f"vram:{d}:all"] = bnd
            if pol > bnd:
                raise SimError(
                    f"DOMINANCE VIOLATION vram {d} (run total): policy "
                    f"hits {pol} > belady bound {bnd}")
    else:
        for p_idx, j in enumerate(journals):
            for d in devices:
                c = totals[(p_idx, d)]
                bound = c["shadow_vram_hits"]
                bounds[f"vram:{d}:p{p_idx}"] = bound
                if c["vram_hits"] > bound:
                    raise SimError(
                        f"DOMINANCE VIOLATION: {j['label']} {d} policy hits "
                        f"{c['vram_hits']} > belady bound {bound}")

    # ---- accounting-closure checks (fail-closed) ------------------------------------
    for p_idx, j in enumerate(journals):
        for d in devices:
            c = totals[(p_idx, d)]
            req = c["requests"]
            if req != c["vram_hits"] + c["vram_inflight"] + c["vram_cold"]:
                raise SimError(
                    f"{j['label']} {d}: requests {req} != hits "
                    f"{c['vram_hits']} + inflight {c['vram_inflight']} + "
                    f"cold {c['vram_cold']}")
            expected = 0 if bypass else (
                req if cfg["host_consult"] == "eager" else c["vram_cold"])
            got = c["host_hits"] + c["host_misses"]
            if got != expected:
                raise SimError(
                    f"{j['label']} {d}: host consults {got} != expected "
                    f"{expected} (consult={cfg['host_consult']})")
            ls_req = sum(v for (p, dd, l), v in per_layer_req.items()
                         if p == p_idx and dd == d)
            ls_hit = sum(v for (p, dd, l), v in per_layer_hit.items()
                         if p == p_idx and dd == d)
            if ls_req != req or ls_hit != c["vram_hits"]:
                raise SimError(
                    f"{j['label']} {d}: per-layer sums diverge "
                    f"(req {ls_req}/{req}, hits {ls_hit}/{c['vram_hits']})")
            fs_req = sum(per_fwd[(p_idx, f, d)]["requests"]
                         for f in range(c["n_forwards"]))
            fs_hit = sum(per_fwd[(p_idx, f, d)]["vram_hits"]
                         for f in range(c["n_forwards"]))
            if fs_req != req or fs_hit != c["vram_hits"]:
                raise SimError(
                    f"{j['label']} {d}: per-forward sums diverge "
                    f"(req {fs_req}/{req}, hits {fs_hit}/{c['vram_hits']})")
            if not bypass:
                joint = (c["vram_miss_host_hit"] + c["vram_miss_host_miss"])
                if joint != c["vram_cold"]:
                    raise SimError(
                        f"{j['label']} {d}: vram-miss joint pairing "
                        f"{joint} != vram_cold {c['vram_cold']}")
            if c["vram_resident_end"] > cap_slots:
                raise SimError(f"{j['label']} {d}: VRAM capacity exceeded")
            if len(host[d].od) > host_caps:
                raise SimError(f"{j['label']} {d}: host capacity exceeded")
        cp = totals[(p_idx, "pooled")]
        req = cp["requests"]
        if req != cp["vram_hits"] + cp["vram_cold"]:
            raise SimError(f"{j['label']} pooled: request closure failed")
        expected = 0 if bypass else (
            req if cfg["host_consult"] == "eager" else cp["vram_cold"])
        if cp["host_hits"] + cp["host_misses"] != expected:
            raise SimError(
                f"{j['label']} pooled: host consult closure failed")

    # ---- emit rows --------------------------------------------------------------------
    rows = []
    fwd_rows = []
    for p_idx, j in enumerate(journals):
        n_forwards = totals[(p_idx, "cuda:0")]["n_forwards"]
        for d in list(devices) + ["pooled"]:
            rows.append(make_row(cfg, j, d, totals[(p_idx, d)],
                                 cap_slots, host_caps, blob, n_forwards,
                                 bypass, p_idx))
        for f in range(n_forwards):
            for d in devices:
                fc = per_fwd[(p_idx, f, d)]
                fwd_rows.append(make_fwd_row(cfg, j["label"], p_idx, f, d,
                                             fc))
        for f in range(n_forwards):
            fcd = dict(per_fwd[(p_idx, f, "pooled")])
            fcd["vram_evictions"] = sum(
                per_fwd[(p_idx, f, d)]["vram_evictions"] for d in devices)
            fwd_rows.append(make_fwd_row(cfg, j["label"], p_idx, f,
                                         "pooled", fcd))

    meta = {
        "journals": [{"label": j["label"], "path": j["path"],
                      "sha256": j["sha256"], "records": len(j["records"])}
                     for j in journals],
        "config": dict(cfg),
        "cap_slots": cap_slots,
        "host_capacity_records": host_caps,
        "blob_bytes": blob,
        "record_bytes": FP4_RECORD_BYTES,
        "cache_batch": cache_batch,
        "queue_depth": queue_depth,
        "belady_bounds": bounds,
    }
    return rows, fwd_rows, meta


def make_row(cfg, journal, device, c, cap_slots, host_caps, blob,
             n_forwards, bypass, p_idx):
    req = c["requests"]
    vram_hits = c["vram_hits"]
    vram_cold = c["vram_cold"]
    host_hits = c["host_hits"]
    host_misses = c["host_misses"]
    consults = host_hits + host_misses
    store_reads = vram_cold if bypass else host_misses
    gen_tokens = n_forwards      # every forward emits one token
    h2d_bytes = vram_cold * FP4_RECORD_BYTES
    store_bytes = store_reads * FP4_RECORD_BYTES
    return {
        "prompt": journal["label"],
        "prompt_idx": p_idx,
        "device": device,
        "requests": req,
        "requests_prefill": c["requests_prefill"],
        "requests_decode": c["requests_decode"],
        "raw_picks": c["raw_picks"],
        "dup_eliminated": c["dup_eliminated"],
        "first_touch_requests": c["first_touch_requests"],
        "repeat_requests": c["repeat_requests"],
        "vram_hits": vram_hits,
        "vram_hits_prefill": c["vram_hits_prefill"],
        "vram_hits_decode": c["vram_hits_decode"],
        "vram_inflight": c["vram_inflight"],
        "vram_cold_loads": vram_cold,
        "vram_cold_prefill": c["vram_cold_prefill"],
        "vram_cold_decode": c["vram_cold_decode"],
        "vram_evictions": c["vram_evictions"],
        "vram_resident_end": c["vram_resident_end"],
        "vram_pinned_skipped": c["vram_pinned_skipped"],
        "belady_vram_hits_bound": c["shadow_vram_hits"],
        "host_consults": consults,
        "host_hits": host_hits,
        "host_hits_prefill": c["host_hits_prefill"],
        "host_hits_decode": c["host_hits_decode"],
        "host_misses": host_misses,
        "host_misses_prefill": c["host_misses_prefill"],
        "host_misses_decode": c["host_misses_decode"],
        "host_evictions": c["host_evictions"],
        "host_entries_end": c["host_entries_end"],
        "vram_miss_host_hit": c["vram_miss_host_hit"],
        "vram_miss_host_miss": c["vram_miss_host_miss"],
        "hits_on_prewarmed": c["hits_on_prewarmed"],
        "store_reads": store_reads,
        "h2d_bytes": h2d_bytes,
        "store_bytes": store_bytes,
        "h2d_mib": round(h2d_bytes / (1 << 20), 2),
        "store_mib": round(store_bytes / (1 << 20), 2),
        "generated_tokens": gen_tokens,
        "h2d_mib_per_gen_token": (round(h2d_bytes / (1 << 20) / gen_tokens, 2)
                                  if gen_tokens else ""),
        "store_mib_per_gen_token": (
            round(store_bytes / (1 << 20) / gen_tokens, 2)
            if gen_tokens else ""),
        "hit_rate_vram": vram_hits / req if req else 0.0,
        "hit_rate_host_of_consults": host_hits / consults if consults else 0.0,
        "hit_rate_host_of_vram_misses": (
            c["vram_miss_host_hit"] / vram_cold if vram_cold else 0.0),
        "hit_rate_e2e_store_avoid": (1.0 - store_reads / req) if req else 0.0,
        "vram_hits_per_raw_pick": (vram_hits / c["raw_picks"]
                                   if c["raw_picks"] else 0.0),
        "requests_per_raw_pick": (req / c["raw_picks"]
                                  if c["raw_picks"] else 0.0),
        "reuse_n": c.get("reuse_n", 0),
        "reuse_p50": c.get("reuse_p50", ""),
        "reuse_p90": c.get("reuse_p90", ""),
        "reuse_p99": c.get("reuse_p99", ""),
        "reuse_cdf": (json.dumps(c["reuse_cdf"], sort_keys=True)
                      if c.get("reuse_cdf") else ""),
        "vram_gib": cfg["vram_bytes"] / GIB,
        "vram_dtype": cfg["vram_dtype"],
        "vram_slots": cap_slots,
        "vram_policy": cfg["vram_policy"],
        "host_gib": cfg["host_bytes"] / GIB,
        "host_records": host_caps,
        "host_scope": cfg["host_scope"],
        "host_consult": cfg["host_consult"],
        "queue_depth": cfg["queue_depth"],
        "regime": cfg["regime"],
        "persist": int(cfg["persist"]),
        "journal_sha256": journal["sha256"],
    }


def make_fwd_row(cfg, label, p_idx, fwd, device, fc):
    return {
        "prompt": label,
        "prompt_idx": p_idx,
        "forward_step": fwd,
        "phase": "prefill" if fwd == 0 else "decode",
        "device": device,
        "requests": fc.get("requests", 0),
        "vram_hits": fc.get("vram_hits", 0),
        "vram_cold": fc.get("vram_cold", 0),
        "vram_evictions": fc.get("vram_evictions", 0),
        "host_hits": fc.get("host_hits", 0),
        "host_misses": fc.get("host_misses", 0),
        "host_evictions": fc.get("host_evictions", 0),
        "vram_policy": cfg["vram_policy"],
        "vram_dtype": cfg["vram_dtype"],
        "persist": int(cfg["persist"]),
    }


# ---------------------------------------------------------------------------
# Anchor comparison
# ---------------------------------------------------------------------------

ANCHOR_MAP = [
    ("prefetch_issued", "requests"),
    ("resident_hits", "vram_hits"),
    ("cold_loads", "vram_cold_loads"),
    ("evictions", "vram_evictions"),
]
ANCHOR_HOST_MAP = [
    ("hits", "host_hits"),
    ("misses", "host_misses"),
    ("evictions", "host_evictions"),
    ("entries", "host_entries_end"),
]


def compare_anchors(anchor_paths, rows, journals, tol, fwd_rows):
    """Match each anchor file to a prompt via route_journal.file_sha256
    (fall back to list order).

    Anchor semantics (result-*.json, verified against q0/q1/q2):
      engine_stats            — per-prompt counters
      host_pack               — PROCESS-CUMULATIVE counters across prompts
      per_token_accounting[0] — host/store fields cumulative since process
                                start; engine fields per-step
      per_token_accounting[s>0] — all fields are per-step deltas
    The sim rows are per-prompt, so host comparisons use cumulative sums."""
    sha_to_label = {j["sha256"]: j["label"] for j in journals}
    label_to_idx = {j["label"]: i for i, j in enumerate(journals)}
    ok = True

    def cum_host(p_idx, dev):
        """Cumulative host counters through END of prompt p_idx."""
        h = m = e = 0
        for p in range(p_idx + 1):
            r = next((x for x in rows
                      if x["prompt_idx"] == p and x["device"] == dev), None)
            if r is None:
                continue
            h += r["host_hits"]
            m += r["host_misses"]
            e += r["host_evictions"]
        return h, m, e

    def cum_pooled_fwd_host(p_idx, fwd):
        """Cumulative pooled host hits/misses through (prompt p_idx, fwd)."""
        h = m = 0
        for p in range(p_idx):
            r = next((x for x in rows
                      if x["prompt_idx"] == p and x["device"] == "pooled"),
                     None)
            if r is not None:
                h += r["host_hits"]
                m += r["host_misses"]
        for fr in fwd_rows:
            if (fr["prompt_idx"] == p_idx and fr["device"] == "pooled"
                    and fr["forward_step"] <= fwd):
                h += fr["host_hits"]
                m += fr["host_misses"]
        return h, m

    for a_idx, apath in enumerate(anchor_paths):
        with open(apath, encoding="utf-8") as fh:
            anchor = json.load(fh)
        jsha = anchor.get("route_journal", {}).get("file_sha256")
        label = sha_to_label.get(jsha)
        if label is None and a_idx < len(journals):
            label = journals[a_idx]["label"]
        p_idx = label_to_idx.get(label, a_idx)
        print(f"\n=== anchor gate: {os.path.basename(apath)} "
              f"(journal sha {str(jsha)[:12]} -> prompt {label}) ===")
        cand = [r for r in rows if r["prompt"] == label]
        for dev, akey in (("cuda:0", "cuda0"), ("cuda:1", "cuda1")):
            row = next((r for r in cand if r["device"] == dev), None)
            if row is None:
                print(f"  {dev}: NO SIM ROW")
                ok = False
                continue
            es = anchor["engine_stats"][akey]
            hp = anchor["host_pack"][akey]
            print(f"  {dev}:")
            for a_f, s_f in ANCHOR_MAP:
                av, sv = es[a_f], row[s_f]
                d = sv - av
                flag = "OK" if abs(d) <= tol else "DELTA"
                if abs(d) > tol:
                    ok = False
                print(f"    {a_f:16s} anchor={av:8d} sim={sv:8d} "
                      f"delta={d:+d} {flag}")
            ch, cm, ce = cum_host(p_idx, dev)
            for a_f, sv in (("hits", ch), ("misses", cm),
                            ("evictions", ce)):
                av = hp[a_f]
                d = sv - av
                flag = "OK" if abs(d) <= tol else "DELTA"
                if abs(d) > tol:
                    ok = False
                print(f"    host.{a_f:11s} anchor={av:8d} sim={sv:8d} "
                      f"(cum) delta={d:+d} {flag}")
            av, sv = hp["entries"], row["host_entries_end"]
            d = sv - av
            flag = "OK" if abs(d) <= tol else "DELTA"
            if abs(d) > tol:
                ok = False
            print(f"    host.entries   anchor={av:8d} sim={sv:8d} "
                  f"delta={d:+d} {flag}")
        # per-forward gates (per_token_accounting)
        pt = anchor.get("per_token_accounting") or []
        if pt:
            pool = next((r for r in cand if r["device"] == "pooled"), None)
            if pool is not None:
                fwd0 = pt[0]
                # step0 host fields are cumulative-since-process-start
                ch0, cm0 = cum_pooled_fwd_host(p_idx, 0)
                for name, av, sv in (
                        ("fwd0 host_misses", fwd0["host_pack_misses"], cm0),
                        ("fwd0 host_hits", fwd0["host_pack_hits"], ch0),
                        ("fwd0 cold_loads", fwd0["cold_loads"],
                         pool["vram_cold_prefill"]),
                        ("fwd0 requests",
                         fwd0["cold_loads"] + fwd0["resident_hits"],
                         pool["requests_prefill"])):
                    d = sv - av
                    flag = "OK" if abs(d) <= tol else "DELTA"
                    if abs(d) > tol:
                        ok = False
                    print(f"    pooled {name:16s} anchor={av:8d} "
                          f"sim={sv:8d} delta={d:+d} {flag}")
            for step in range(0, min(4, len(pt))):
                arow = pt[step]
                fr = next((r for r in fwd_rows
                           if r["prompt"] == label
                           and r["forward_step"] == step
                           and r["device"] == "pooled"), None)
                if fr is None:
                    continue
                if step == 0:
                    hh, hm = cum_pooled_fwd_host(p_idx, 0)
                else:
                    hh, hm = fr["host_hits"], fr["host_misses"]
                pairs = (("requests",
                          arow["cold_loads"] + arow["resident_hits"],
                          fr["requests"]),
                         ("cold_loads", arow["cold_loads"],
                          fr["vram_cold"]),
                         ("host_hits", arow["host_pack_hits"], hh),
                         ("host_misses", arow["host_pack_misses"], hm))
                line = []
                for name, av, sv in pairs:
                    d = sv - av
                    flag = "" if abs(d) <= tol else "  <-- DELTA"
                    if abs(d) > tol:
                        ok = False
                    line.append(f"{name} {sv}vs{av} (d={d:+d}){flag}")
                print(f"    fwd{step:02d} pooled: " + " | ".join(line))
    return ok


# ---------------------------------------------------------------------------
# Sweep machinery
# ---------------------------------------------------------------------------

SWEEP_KEYS = {
    "vram_gib": ("vram_bytes", lambda v: float(v) * GIB),
    "vram_dtype": ("vram_dtype", str),
    "vram_policy": ("vram_policy", str),
    "host_gib": ("host_bytes", lambda v: float(v) * GIB),
    "host_scope": ("host_scope", str),
    "host_consult": ("host_consult", str),
    "queue_depth": ("queue_depth", int),
    "regime": ("regime", str),
}


def expand_sweeps(base_cfg, sweep_specs):
    cfgs = [dict(base_cfg)]
    for spec in sweep_specs:
        if "=" not in spec:
            raise SimError(f"--sweep {spec!r}: expected key=v1,v2,...")
        key, vals = spec.split("=", 1)
        if key not in SWEEP_KEYS:
            raise SimError(f"--sweep key {key!r} not in {sorted(SWEEP_KEYS)}")
        cfg_key, conv = SWEEP_KEYS[key]
        new = []
        for c in cfgs:
            for v in vals.split(","):
                c2 = dict(c)
                c2[cfg_key] = conv(v)
                new.append(c2)
        cfgs = new
    return cfgs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Phase-4 two-level cache replay simulator over route "
                    "journals (stdlib-only).")
    p.add_argument("--journals", nargs="+", required=True,
                   help="route journal JSONL paths, in prompt order")
    p.add_argument("--cohort", action="store_true",
                   help="merge all --journals into ONE lockstep cohort "
                        "stream: records sharing (forward_step,layer) are "
                        "merged by concatenating rank rows (what a b=K "
                        "forward emits; engine dedupes the union)")
    p.add_argument("--prompt-order", default=None,
                   help="comma list of prompt labels applied to --journals "
                        "in order (default labels come from filenames)")
    p.add_argument("--persist-across-prompts", action="store_true",
                   help="cache contents persist across prompts, counters "
                        "reset per prompt (GPU-2 faithful)")
    p.add_argument("--vram", type=float, default=3.5,
                   help="VRAM cache budget in GiB (default 3.5)")
    p.add_argument("--vram-dtype", choices=("fp16", "fp4"), default="fp16",
                   help="resident block dtype: fp16 48MiB / fp4 12.75MiB")
    p.add_argument("--vram-policy", default="priority_lru",
                   choices=POLICIES,
                   help="eviction policy (priority_lru = bug-compat engine)")
    p.add_argument("--host", type=float, default=8.5,
                   help="host pack budget in GiB per scope unit "
                        "(0 = bypass: no consults, vram misses read store)")
    p.add_argument("--host-scope", choices=("per-device", "pooled"),
                   default="per-device")
    p.add_argument("--host-policy", choices=("lru", "belady"), default="lru",
                   help="host tier policy (belady = offline bound on the "
                        "consult stream)")
    p.add_argument("--host-consult", choices=("eager", "lazy"),
                   default="eager",
                   help="eager = whole chunk consulted before staging (live "
                        "config); lazy = only VRAM misses consult")
    p.add_argument("--queue-depth", type=int, default=DEFAULT_QUEUE_DEPTH,
                   help="host get_batch sub-batch size (engine default 6)")
    p.add_argument("--regime", choices=("cold", "prewarm"), default="cold")
    p.add_argument("--prewarm", default=None,
                   help="declared initial state for regime=prewarm: "
                        "file path ('layer,expert' lines) or auto:FRAC")
    p.add_argument("--static-hotset", default=None,
                   help="static_hotset+dyn set: file path or auto:FRAC "
                        "(top floor(cap*frac) keys by stream frequency)")
    p.add_argument("--freq-weight", type=int, default=PRIORITY_WEIGHT,
                   help="freq_x_recency: score = last_used + freq*W")
    p.add_argument("--segmented-probation", type=float, default=0.2,
                   help="segmented probationary fraction (default 0.2)")
    p.add_argument("--layer-split", type=int, default=DEFAULT_LAYER_SPLIT,
                   help="first cuda:1 layer (default 22)")
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--sweep", action="append", default=[],
                   help="axis spec key=v1,v2 — repeatable; keys: " +
                        ",".join(sorted(SWEEP_KEYS)))
    p.add_argument("--anchors", nargs="*", default=[],
                   help="sealed result-*.json files for the counter gate")
    p.add_argument("--anchor-tol", type=int, default=0,
                   help="allowed absolute counter delta (default 0 = exact)")
    p.add_argument("--out", default=None, help="summary CSV path")
    p.add_argument("--per-forward", dest="per_forward", default=None,
                   help="per-forward CSV path")
    p.add_argument("--json-out", default=None, help="run-metadata JSON path")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def label_for_path(path, idx):
    base = os.path.basename(path)
    stem = base.rsplit(".", 1)[0]
    for token in ("rep0", "q0", "q1", "q2", "q3", "r0", "r1", "r2"):
        if f"-{token}" in stem or f"_{token}" in stem:
            return token
    return stem or f"p{idx}"


def main(argv):
    args = parse_args(argv)

    journals = []
    for idx, path in enumerate(args.journals):
        records, sha = load_journal(path, layer_split=args.layer_split,
                                    num_layers=args.num_layers)
        journals.append({"path": path, "records": records, "sha256": sha,
                         "calls": calls_from_journal(records),
                         "label": label_for_path(path, idx)})
    if args.prompt_order:
        wanted = args.prompt_order.split(",")
        if len(wanted) != len(journals):
            raise SimError("--prompt-order length != --journals count")
        for j, lab in zip(journals, wanted):
            j["label"] = lab
    if args.cohort:
        k = len(journals)
        merged_records = merge_cohort_records(journals)
        journals = [{
            "path": ",".join(j["path"] for j in journals),
            "records": merged_records,
            "sha256": hashlib.sha256(
                "".join(j["sha256"] for j in journals).encode()
            ).hexdigest(),
            "calls": calls_from_journal(merged_records),
            "label": f"K{k}-cohort",
            "cohort_members": [j["label"] for j in journals],
            "cohort_k": k,
        }]

    base_cfg = {
        "vram_bytes": int(args.vram * GIB),
        "vram_dtype": args.vram_dtype,
        "vram_policy": args.vram_policy,
        "host_bytes": int(args.host * GIB),
        "host_scope": args.host_scope,
        "host_policy": args.host_policy,
        "host_consult": args.host_consult,
        "queue_depth": args.queue_depth,
        "regime": args.regime,
        "prewarm": args.prewarm,
        "static_hotset": args.static_hotset,
        "freq_weight": args.freq_weight,
        "segmented_probation": args.segmented_probation,
        "persist": bool(args.persist_across_prompts),
        "layer_split": args.layer_split,
        "num_layers": args.num_layers,
        "num_experts": NUM_EXPERTS,
        "topk": TOPK,
    }

    cfgs = expand_sweeps(base_cfg, args.sweep)
    all_rows, all_fwd_rows, metas = [], [], []
    for ci, cfg in enumerate(cfgs):
        if args.verbose:
            print(f"[sim] config {ci+1}/{len(cfgs)}: "
                  f"vram={cfg['vram_bytes']/GIB:.2f}GiB {cfg['vram_dtype']} "
                  f"policy={cfg['vram_policy']} "
                  f"host={cfg['host_bytes']/GIB:.2f}GiB "
                  f"scope={cfg['host_scope']} consult={cfg['host_consult']} "
                  f"qd={cfg['queue_depth']} regime={cfg['regime']} "
                  f"persist={cfg['persist']}", file=sys.stderr)
        rows, fwd_rows, meta = simulate(cfg, journals)
        all_rows.extend(rows)
        all_fwd_rows.extend(fwd_rows)
        metas.append(meta)

    if args.out:
        write_csv(args.out, all_rows)
        if args.verbose:
            print(f"[out] {len(all_rows)} rows -> {args.out}",
                  file=sys.stderr)
    if args.per_forward:
        write_csv(args.per_forward, all_fwd_rows)
        if args.verbose:
            print(f"[out] {len(all_fwd_rows)} fwd rows -> "
                  f"{args.per_forward}", file=sys.stderr)
    if args.json_out:
        _ensure_parent(args.json_out)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"runs": metas}, fh, indent=1)

    print_summary(all_rows)

    gate_ok = True
    if args.anchors:
        gate_ok = compare_anchors(args.anchors, all_rows, journals,
                                  args.anchor_tol, all_fwd_rows)
        print(f"\nANCHOR GATE: {'PASS' if gate_ok else 'FAIL'} "
              f"(tol={args.anchor_tol})")
        if not gate_ok:
            sys.exit(2)


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    _ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows):
    seen = set()
    hdr = (f"{'prompt':6s} {'dev':7s} {'req':>5s} {'vHit':>5s} {'vCold':>5s} "
           f"{'vEv':>5s} {'hHit':>5s} {'hMiss':>5s} {'hEv':>5s} "
           f"{'vHit%':>6s} {'hHit%':>6s} {'e2e%':>6s} {'belady':>6s}")
    for r in rows:
        key = (r["vram_policy"], r["vram_dtype"], r["vram_gib"],
               r["host_gib"], r["host_scope"], r["host_consult"],
               r["queue_depth"], r["regime"], r["persist"])
        if key not in seen:
            seen.add(key)
            print(f"\n--- vram={r['vram_gib']:.2f}GiB {r['vram_dtype']} "
                  f"({r['vram_slots']} slots) policy={r['vram_policy']} | "
                  f"host={r['host_gib']:.2f}GiB {r['host_scope']} "
                  f"consult={r['host_consult']} qd={r['queue_depth']} "
                  f"regime={r['regime']} persist={r['persist']} ---")
            print(hdr)
        print(f"{r['prompt']:6s} {r['device']:7s} {r['requests']:5d} "
              f"{r['vram_hits']:5d} {r['vram_cold_loads']:5d} "
              f"{r['vram_evictions']:5d} {r['host_hits']:5d} "
              f"{r['host_misses']:5d} {r['host_evictions']:5d} "
              f"{100*r['hit_rate_vram']:5.1f}% "
              f"{100*r['hit_rate_host_of_consults']:5.1f}% "
              f"{100*r['hit_rate_e2e_store_avoid']:5.1f}% "
              f"{r['belady_vram_hits_bound']:6d}")


if __name__ == "__main__":
    main(sys.argv[1:])
