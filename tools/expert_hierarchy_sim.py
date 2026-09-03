#!/usr/bin/env python3
"""expert_hierarchy_sim.py — route-trace expert-cache hierarchy simulator (Phase B/C/D).

Consumes dee's *sealed* route journals (routed_experts.jsonl, dee4 schema v1,
chain-hashed). READ-ONLY: never writes to the evidence directories.

Representations:
    MXFP4_CURRENT   4.25 bpw effective (4.0 payload + 0.25 e8m0 scale, group 32)
    STQ1_0_IDEALIZED configurable payload bits/weight (default 1.25) + scale overhead
    IQ2_XXS_APPROX  2.0625 bpw
    MIXED           per-(layer, expert) codec assignment from an explicit JSON map
                    or a value-ranked assignment (MEMORY-ONLY SIMULATION label)

Policies: LRU, LFU, BELADY (offline ceiling), ADMISSION_LRU (2-hit admit),
          reuse-distance oracle accounting. Two tiers: VRAM cache over RAM cache
          over SSD.

Metrics: VRAM/RAM hit rates, SSD miss rate, SSD bytes/token, H2D bytes/token,
         resident experts, evictions, reuse-distance distribution, required
         SSD GB/s per target TPS.

Geometry sources (all measured, sealed):
    - v62 terminal evidence (2x T4): per-expert FP4 record = 13,369,344 B.
    - dee4-metadata.json component table -> 25,165,824 logical weights/expert.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE = os.path.join(
    REPO_ROOT,
    "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
    "v62-terminal-evidence-20260902T003709Z/routed_experts.jsonl",
)
DEFAULT_METADATA = os.path.join(
    REPO_ROOT,
    "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
    "v62-terminal-evidence-20260902T003709Z/dee4-metadata.json",
)

GIB = 1024**3
MIB = 1024**2

# ---------------------------------------------------------------------------
# Sealed geometry (dee4-v3-trace, v62 terminal evidence)
# ---------------------------------------------------------------------------
NUM_LAYERS = 43
EXPERTS_PER_LAYER = 256
TOTAL_EXPERTS = NUM_LAYERS * EXPERTS_PER_LAYER  # 11008
TOPK = 6
LOGICAL_WEIGHTS_PER_EXPERT = 25_165_824  # [2048,4096]x2 + [4096,2048]
MXFP4_RECORD_BYTES = 13_369_344          # sealed per-expert record size
MXFP4_EFFECTIVE_BPW = MXFP4_RECORD_BYTES * 8 / LOGICAL_WEIGHTS_PER_EXPERT  # 4.25

# ---------------------------------------------------------------------------
# Codec models
# ---------------------------------------------------------------------------
CODECS = {
    "MXFP4_CURRENT": {"bpw": MXFP4_EFFECTIVE_BPW},
    "STQ1_0_IDEALIZED": {"bpw": 1.3125},
    "IQ2_XXS_APPROX": {"bpw": 2.0625},
}

BLOCK_WEIGHTS = 256   # ternary/iq2 scale block
SCALE_BYTES_PER_BLOCK = 2  # fp16 per-block scale


def expert_bytes(codec: str, bits_per_weight: float | None = None) -> int:
    """Bytes for one expert under the codec model.

    STQ/IQ2 block overhead: one fp16 scale per 256-weight block.
    Payload bits are rounded up to whole bytes per expert (real formats pad
    per-row, but per-expert granularity is what the cache simulates).
    """
    if codec == "MXFP4_CURRENT":
        return MXFP4_RECORD_BYTES
    if codec == "STQ1_0_IDEALIZED":
        # bits_per_weight = PAYLOAD bits (5 bits per 4 weights = 1.25 default);
        # block scale overhead (fp16 per 256 weights) adds 0.0625 bpw, so the
        # llama.cpp STQ1_0 total is 1.3125 bpw.
        bpw = 1.25 if bits_per_weight is None else bits_per_weight
        payload_bits = LOGICAL_WEIGHTS_PER_EXPERT * bpw
        blocks = math.ceil(LOGICAL_WEIGHTS_PER_EXPERT / BLOCK_WEIGHTS)
        overhead = blocks * SCALE_BYTES_PER_BLOCK
        return int(math.ceil(payload_bits / 8)) + overhead
    if codec == "IQ2_XXS_APPROX":
        payload_bits = LOGICAL_WEIGHTS_PER_EXPERT * 2.0
        blocks = math.ceil(LOGICAL_WEIGHTS_PER_EXPERT / BLOCK_WEIGHTS)
        overhead = blocks * SCALE_BYTES_PER_BLOCK
        return int(math.ceil(payload_bits / 8)) + overhead
    raise ValueError(codec)


def parse_codec_map(path: str) -> dict[tuple[int, int], str]:
    """Explicit per-expert codec map: {"layer,expert": "CODEC_NAME"}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for key, name in raw.items():
        layer_s, expert_s = key.split(",")
        out[(int(layer_s), int(expert_s))] = name
    return out


# ---------------------------------------------------------------------------
# Trace consumption
# ---------------------------------------------------------------------------
@dataclass
class Step:
    """One scheduled forward step = one simulated token row."""
    idx: int
    phase: str
    experts: list          # (layer, expert) touched once per layer (grouped)
    weight: int            # tokens in this step (prefill rows > 1)


def load_trace(path: str, dedupe_prefill: bool = True) -> list[Step]:
    """Parse the sealed routed_experts.jsonl into scheduled steps.

    Geometry notes (from the live scheduler, P2.3 analysis): duplicate expert
    IDs within a layer are grouped once; per-GPU caches are split at layer 22.
    """
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: (r["forward_step"], r["layer"]))
    steps = []
    by_fs: dict[int, list] = {}
    for r in recs:
        by_fs.setdefault(r["forward_step"], []).append(r)
    for fs in sorted(by_fs):
        recs_fs = by_fs[fs]
        phase = recs_fs[0]["phase"]
        token_rows = max(r.get("token_rows", 1) for r in recs_fs)
        experts = []
        seen = set()
        for r in recs_fs:
            layer = r["layer"]
            for row in r["expert_ids_rank_order"]:
                for eid in row:
                    key = (layer, eid)
                    if key not in seen:
                        seen.add(key)
                        experts.append(key)
        if phase == "prefill" and dedupe_prefill:
            weight = token_rows  # prefill: one group-load serves token_rows tokens
        else:
            weight = 1
        steps.append(Step(idx=len(steps), phase=phase, experts=experts, weight=weight))
    return steps


# ---------------------------------------------------------------------------
# Two-tier cache (VRAM over RAM over SSD)
# ---------------------------------------------------------------------------
class TwoTierCache:
    """VRAM tier (hit -> no traffic), RAM tier (hit -> H2D bytes), SSD tier
    (miss -> SSD bytes + H2D bytes). Separate LRU/LFU/Belady per tier.

    Accounting note: an SSD miss streams its record from SSD into the RAM tier
    and H2D into the VRAM tier (both counters += eb). A RAM hit costs H2D only.
    With --warm-start, every expert touched during prefill is pre-seeded into
    the RAM tier (perfect-warmup steady-state model), so SSD traffic counts
    decode-phase misses only; compulsory cold loads are reported separately
    as warmup_misses (one-time cost, excluded from per-token steady state)."""

    def __init__(self, vram_bytes: int, ram_bytes: int, per_expert_bytes: int,
                 vram_policy: str = "LRU", ram_policy: str = "LRU",
                 admission_min_reuse: int = 1):
        self.vram_bytes = vram_bytes
        self.ram_bytes = ram_bytes
        self.eb = per_expert_bytes
        self.vram_policy = vram_policy
        self.ram_policy = ram_policy
        self.admission_min_reuse = admission_min_reuse

        self.vram: OrderedDict = OrderedDict()   # key -> last-use tick
        self.ram: OrderedDict = OrderedDict()
        self.freq: Counter = Counter()           # global request frequency
        self.reqs_seen: Counter = Counter()      # admissions criterion
        self.future: list[set] = []              # for Belady (per-step key sets)
        self.step_of: dict = {}
        self.steps_list: list = []

        # metrics
        self.vram_hits = 0
        self.ram_hits = 0
        self.ssd_misses = 0
        self.h2d_bytes = 0
        self.ssd_bytes = 0
        self.vram_evictions = 0
        self.ram_evictions = 0
        self.reuse_distances: list[int] = []
        self.admitted = 0
        self.bypassed = 0
        self.warm_start = False
        self.warmup_misses = 0
        # decode-phase-only counters (steady-state view)
        self.dec_vram_hits = 0
        self.dec_ram_hits = 0
        self.dec_ssd_misses = 0
        self.dec_h2d_bytes = 0
        self.dec_ssd_bytes = 0

    # -- main ---------------------------------------------------------------
    def run(self, steps: list[Step]) -> None:
        # Per-key ordered request sequence for true Belady (next use after tick).
        seq: dict = {}
        for i, step in enumerate(steps):
            for k in step.experts:
                seq.setdefault(k, []).append(i)
        self._seq = seq
        self._ptr = {k: 0 for k in seq}
        self.step_of = {k: s[-1] for k, s in seq.items()}  # last use (info only)

        vram_cap = max(1, self.vram_bytes // self.eb)
        ram_cap = max(0, self.ram_bytes // self.eb)

        # Optional perfect warmup: prefill-touched experts pre-staged in RAM.
        if self.warm_start:
            for step in steps:
                if step.phase != "prefill":
                    continue
                for key in step.experts:
                    if key in self.vram or key in self.ram:
                        continue
                    if ram_cap > 0 and len(self.ram) < ram_cap:
                        self.ram[key] = -1
                        self.warmup_misses += 1  # one-time staging cost (SSD->RAM)
                    elif ram_cap > 0:
                        self.warmup_misses += 1  # staged then evicted: still one-time
                        # stage-and-displace: admit then evict LRU
                        self.ram[key] = -1
                        while len(self.ram) >= ram_cap:
                            victim, _ = self.ram.popitem(last=False)
                            if victim == key:
                                break

        for tick, step in enumerate(steps):
            is_decode = step.phase == "decode"
            for key in step.experts:
                self.freq[key] += 1
                # advance Belady pointer past the current use
                p = self._ptr.get(key, 0)
                s = seq.get(key, [])
                while p < len(s) and s[p] <= tick:
                    p += 1
                self._ptr[key] = p
                self._next_use_of = p < len(s) and s[p] or math.inf

                if self.vram_bytes and key in self.vram:
                    self.vram_hits += 1
                    self.vram.move_to_end(key)
                    if is_decode:
                        self.dec_vram_hits += 1
                elif self.ram_bytes and key in self.ram:
                    self.ram_hits += 1
                    self.h2d_bytes += self.eb
                    if is_decode:
                        self.dec_ram_hits += 1
                        self.dec_h2d_bytes += self.eb
                    # admit to VRAM
                    if vram_cap > 0:
                        while len(self.vram) >= vram_cap:
                            self._evict_from_vram(tick)
                        self.vram[key] = tick
                else:
                    self.ssd_misses += 1
                    self.ssd_bytes += self.eb
                    self.h2d_bytes += self.eb  # miss streams through to VRAM
                    if is_decode:
                        self.dec_ssd_misses += 1
                        self.dec_ssd_bytes += self.eb
                        self.dec_h2d_bytes += self.eb
                    # admit to RAM (SSD->RAM staging), then VRAM
                    if ram_cap > 0:
                        while len(self.ram) >= ram_cap:
                            self._evict_from_ram(tick)
                        self.ram[key] = tick
                    if vram_cap > 0:
                        while len(self.vram) >= vram_cap:
                            self._evict_from_vram(tick)
                        self.vram[key] = tick

    def _evict_from_vram(self, tick):
        if not self.vram:
            return
        if self.vram_policy == "LFU":
            victim = min(self.vram, key=lambda k: (self.freq[k], self.vram[k]))
            self.vram.pop(victim)
        elif self.vram_policy == "BELADY":
            # true offline Belady: evict key with farthest NEXT use after tick
            def next_use_after(k):
                s = self._seq.get(k, [])
                p = self._ptr.get(k, 0)
                # pointer already advanced past current tick for touched keys;
                # for victims, find first use strictly after tick
                lo = p
                while lo < len(s) and s[lo] <= tick:
                    lo += 1
                return s[lo] if lo < len(s) else math.inf
            victim = min(self.vram, key=lambda k: (next_use_after(k), self.vram[k]))
            self.vram.pop(victim)
        else:
            victim, _ = self.vram.popitem(last=False)
        self.vram_evictions += 1

    def _evict_from_ram(self, tick):
        if not self.ram:
            return
        if self.ram_policy == "LFU":
            victim = min(self.ram, key=lambda k: (self.freq[k], self.ram[k]))
            self.ram.pop(victim)
        elif self.ram_policy == "BELADY":
            def next_use_after(k):
                s = self._seq.get(k, [])
                lo = self._ptr.get(k, 0)
                while lo < len(s) and s[lo] <= tick:
                    lo += 1
                return s[lo] if lo < len(s) else math.inf
            victim = min(self.ram, key=lambda k: (next_use_after(k), self.ram[k]))
            self.ram.pop(victim)
        else:
            victim, _ = self.ram.popitem(last=False)
        self.ram_evictions += 1


class AdmissionLRUTwoTier(TwoTierCache):
    """Admission-aware: a key is admitted to RAM only after min_reuse requests
    (2-hit admission filters one-touch prefill outliers). VRAM admission is
    always-on for RAM hits (stream-through)."""

    def __init__(self, *a, admission_min_reuse: int = 2, **kw):
        super().__init__(*a, **kw)
        self.min_reuse = admission_min_reuse

    def run(self, steps):
        next_use = {}
        for i in range(len(steps) - 1, -1, -1):
            for k in steps[i].experts:
                next_use[k] = i
        self.step_of = next_use
        vram_cap = max(1, self.vram_bytes // self.eb)
        ram_cap = max(0, self.ram_bytes // self.eb)
        for tick, step in enumerate(steps):
            for key in step.experts:
                self.freq[key] += 1
                self.reqs_seen[key] += 1
                if self.vram_bytes and key in self.vram:
                    self.vram_hits += 1
                    self.vram.move_to_end(key)
                elif self.ram_bytes and key in self.ram:
                    self.ram_hits += 1
                    self.h2d_bytes += self.eb
                    if vram_cap > 0:
                        while len(self.vram) >= vram_cap:
                            self._evict_from_vram(tick)
                        self.vram[key] = tick
                else:
                    self.ssd_misses += 1
                    self.ssd_bytes += self.eb
                    self.h2d_bytes += self.eb
                    if ram_cap > 0 and self.reqs_seen[key] >= self.min_reuse:
                        while len(self.ram) >= ram_cap:
                            self._evict_from_ram(tick)
                        self.ram[key] = tick
                        self.admitted += 1
                    elif ram_cap > 0:
                        self.bypassed += 1
                    if vram_cap > 0:
                        while len(self.vram) >= vram_cap:
                            self._evict_from_vram(tick)
                        self.vram[key] = tick


# ---------------------------------------------------------------------------
# Reuse-distance oracle (stack distance over grouped steps)
# ---------------------------------------------------------------------------
def reuse_distance_distribution(steps: list[Step]) -> Counter:
    """LRU stack-distance histogram over (layer, expert) requests.
    Distance 0 = repeat within the same step. inf = first touch."""
    dist: Counter = Counter()
    stack: list = []
    pos: dict = {}
    for step in steps:
        for key in step.experts:
            if key in pos:
                d = len(stack) - 1 - pos[key]
                dist[d] += 1
                stack.pop(pos[key])
            else:
                dist[math.inf] += 1
            stack.append(key)
            pos[key] = len(stack) - 1
    return dist


def belady_offline_hit_rate(steps: list[Step], cache_experts: int) -> tuple[float, int]:
    """Exact offline Belady (MIN) hit rate for a single-tier cache of N experts.
    Victim = cached key whose next request is farthest in the future."""
    cap = max(1, cache_experts)
    cache: set = set()
    hits = 0
    total = 0
    # per-key ordered request index list + pointer per key
    seq: dict = {}
    for i, step in enumerate(steps):
        for k in step.experts:
            seq.setdefault(k, []).append(i)
    ptr = {k: 0 for k in seq}
    next_use: dict = {}
    for k, s in seq.items():
        next_use[k] = s[0]

    for i, step in enumerate(steps):
        for key in step.experts:
            total += 1
            if key in cache:
                hits += 1
            else:
                if len(cache) >= cap:
                    victim = max(cache, key=lambda k: next_use.get(k, math.inf))
                    cache.remove(victim)
                cache.add(key)
            # advance pointer: next use of key strictly after i
            p = ptr[key]
            s = seq[key]
            while p < len(s) and s[p] <= i:
                p += 1
            ptr[key] = p
            next_use[key] = s[p] if p < len(s) else math.inf
    return (hits / total if total else 0.0), total


def belady_decode_only_hit_rate(steps: list[Step], cache_experts: int,
                                prefill_warm: bool = True) -> tuple[float, int, int]:
    """Belady over decode-phase requests only, optionally seeding the cache
    with every expert touched during prefill (perfect warmup ceiling)."""
    cap = max(1, cache_experts)
    decode_steps = [s for s in steps if s.phase == "decode"]
    seq: dict = {}
    for i, step in enumerate(decode_steps):
        for k in step.experts:
            seq.setdefault(k, []).append(i)
    ptr = {k: 0 for k in seq}
    next_use = {k: s[0] for k, s in seq.items()}
    cache: set = set()
    if prefill_warm:
        for step in steps:
            if step.phase == "prefill":
                cache.update(step.experts)
        # trim to capacity with Belady against decode future
        while len(cache) > cap:
            victim = max(cache, key=lambda k: next_use.get(k, math.inf))
            cache.remove(victim)
    hits = total = 0
    for i, step in enumerate(decode_steps):
        for key in step.experts:
            total += 1
            if key in cache:
                hits += 1
            else:
                if len(cache) >= cap:
                    victim = max(cache, key=lambda k: next_use.get(k, math.inf))
                    cache.remove(victim)
                cache.add(key)
            p = ptr[key]
            s = seq[key]
            while p < len(s) and s[p] <= i:
                p += 1
            ptr[key] = p
            next_use[key] = s[p] if p < len(s) else math.inf
    return (hits / total if total else 0.0), total, hits


# ---------------------------------------------------------------------------
# Bandwidth / TPS roofline
# ---------------------------------------------------------------------------
def required_bandwidth(ssd_bytes_per_token: float, h2d_bytes_per_token: float,
                       tps: float) -> dict:
    return {
        "ssd_GBps": ssd_bytes_per_token * tps / 1e9,
        "pcie_GBps": h2d_bytes_per_token * tps / 1e9,
    }


# ---------------------------------------------------------------------------
# Mixed-precision assignment (Phase E hook)
# ---------------------------------------------------------------------------
def assign_mixed(steps: list[Step], frac_stq: float, frac_iq2: float,
                 value_rank: dict | None = None) -> dict:
    """MEMORY-ONLY SIMULATION: rank experts by observed request count (or an
    explicit value_rank from a sensitivity file) and assign codecs:
    most-requested frac_stq stay... assignment = sensitivity-ranked."""
    freq = Counter()
    for step in steps:
        for key in step.experts:
            freq[key] += 1
    keys = sorted(freq, key=lambda k: -(value_rank or {}).get(k, freq[k]))
    n = len(freq) or 1
    n_stq = int(round(frac_stq * n))
    n_iq2 = int(round(frac_iq2 * n))
    assignment = {}
    for i, k in enumerate(keys):
        if i < n_stq:
            assignment[k] = "MXFP4_CURRENT"
        elif i < n_stq + n_iq2:
            assignment[k] = "IQ2_XXS_APPROX"
        else:
            assignment[k] = "STQ1_0_IDEALIZED"
    return assignment


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
def simulate(steps: list[Step], per_expert_bytes_fn, vram_bytes: int, ram_bytes: int,
             policy: str = "LRU", admission: bool = False,
             trace_label: str = "", warm_start: bool = False) -> dict:
    """Run two-tier sim. per_expert_bytes_fn(key) -> bytes (codec-aware)."""
    eb_for = per_expert_bytes_fn
    # uniform-size fast path
    sizes = {eb_for(k) for step in steps for k in step.experts}
    if len(sizes) == 1:
        eb = sizes.pop()
        cls = AdmissionLRUTwoTier if admission else TwoTierCache
        cache = cls(vram_bytes, ram_bytes, eb, vram_policy=policy, ram_policy=policy)
        cache.warm_start = bool(warm_start)
        cache.run(steps)
    else:
        # variable-size model: pack tiers greedily by recency (approximate for MIXED)
        return simulate_variable(steps, eb_for, vram_bytes, ram_bytes, policy)

    total_reqs = cache.vram_hits + cache.ram_hits + cache.ssd_misses
    total_tokens = sum(s.weight for s in steps)
    decode_steps = [s for s in steps if s.phase == "decode"]
    decode_tokens = sum(s.weight for s in decode_steps)
    decode_reqs = sum(len(s.experts) for s in decode_steps)
    # decode-phase-only traffic accounting (steady-state view)
    if warm_start:
        dec_frac = decode_reqs / total_reqs if total_reqs else 0
        # recompute per-phase splits from counters (hits/misses recorded globally)
        out = {
            "label": trace_label,
            "per_expert_bytes": eb,
            "vram_budget_bytes": vram_bytes,
            "ram_budget_bytes": ram_bytes,
            "policy": "ADMISSION_LRU" if admission else policy,
            "warm_start": True,
            "requests": total_reqs,
            "tokens_total": total_tokens,
            "tokens_decode": decode_tokens,
            "vram_hits": cache.vram_hits,
            "ram_hits": cache.ram_hits,
            "ssd_misses": cache.ssd_misses,
            "vram_hit_rate": cache.vram_hits / total_reqs if total_reqs else 0,
            "ram_hit_rate": cache.ram_hits / total_reqs if total_reqs else 0,
            "ssd_miss_rate": cache.ssd_misses / total_reqs if total_reqs else 0,
            "combined_ram_vram_hit_rate": (cache.vram_hits + cache.ram_hits) / total_reqs if total_reqs else 0,
            "h2d_bytes_total": cache.h2d_bytes,
            "ssd_bytes_total": cache.ssd_bytes,
            "h2d_bytes_per_token": cache.h2d_bytes / decode_tokens if decode_tokens else 0,
            "ssd_bytes_per_token": cache.ssd_bytes / decode_tokens if decode_tokens else 0,
            "vram_evictions": cache.vram_evictions,
            "ram_evictions": cache.ram_evictions,
            "vram_resident_experts_max": vram_bytes // eb if eb else 0,
            "ram_resident_experts_max": ram_bytes // eb if eb else 0,
            "warmup_misses_one_time": cache.warmup_misses,
            "warmup_staging_bytes": cache.warmup_misses * eb,
            "decode_requests": sum(len(s.experts) for s in steps if s.phase == "decode"),
            "decode_vram_hit_rate": _safe_div(cache.dec_vram_hits,
                                              cache.dec_vram_hits + cache.dec_ram_hits + cache.dec_ssd_misses),
            "decode_ram_hit_rate": _safe_div(cache.dec_ram_hits,
                                             cache.dec_vram_hits + cache.dec_ram_hits + cache.dec_ssd_misses),
            "decode_combined_hit_rate": _safe_div(cache.dec_vram_hits + cache.dec_ram_hits,
                                                  cache.dec_vram_hits + cache.dec_ram_hits + cache.dec_ssd_misses),
            "decode_ssd_miss_rate": _safe_div(cache.dec_ssd_misses,
                                              cache.dec_vram_hits + cache.dec_ram_hits + cache.dec_ssd_misses),
            "decode_ssd_bytes_per_token": _safe_div(cache.dec_ssd_bytes, decode_tokens),
            "decode_h2d_bytes_per_token": _safe_div(cache.dec_h2d_bytes, decode_tokens),
        }
        return out
    out = {
        "label": trace_label,
        "per_expert_bytes": eb,
        "vram_budget_bytes": vram_bytes,
        "ram_budget_bytes": ram_bytes,
        "policy": "ADMISSION_LRU" if admission else policy,
        "warm_start": False,
        "requests": total_reqs,
        "tokens_total": total_tokens,
        "tokens_decode": decode_tokens,
        "vram_hits": cache.vram_hits,
        "ram_hits": cache.ram_hits,
        "ssd_misses": cache.ssd_misses,
        "vram_hit_rate": cache.vram_hits / total_reqs if total_reqs else 0,
        "ram_hit_rate": cache.ram_hits / total_reqs if total_reqs else 0,
        "ssd_miss_rate": cache.ssd_misses / total_reqs if total_reqs else 0,
        "combined_ram_vram_hit_rate": (cache.vram_hits + cache.ram_hits) / total_reqs if total_reqs else 0,
        "h2d_bytes_total": cache.h2d_bytes,
        "ssd_bytes_total": cache.ssd_bytes,
        "h2d_bytes_per_token": cache.h2d_bytes / total_tokens if total_tokens else 0,
        "ssd_bytes_per_token": cache.ssd_bytes / total_tokens if total_tokens else 0,
        "vram_evictions": cache.vram_evictions,
        "ram_evictions": cache.ram_evictions,
        "vram_resident_experts_max": vram_bytes // eb if eb else 0,
        "ram_resident_experts_max": ram_bytes // eb if eb else 0,
    }
    return out


def simulate_variable(steps: list[Step], eb_for, vram_bytes: int, ram_bytes: int,
                      policy: str = "LRU") -> dict:
    """Variable-size expert store (MIXED): per-tier LRU with byte budgets."""
    vram: OrderedDict = OrderedDict()   # key -> tick
    ram: OrderedDict = OrderedDict()
    vram_used = ram_used = 0
    vram_hits = ram_hits = ssd_misses = 0
    h2d = ssd = vram_ev = ram_ev = 0
    dec_vram_hits = dec_ram_hits = dec_ssd_misses = 0
    dec_h2d = dec_ssd = 0
    next_use = {}
    for i in range(len(steps) - 1, -1, -1):
        for k in steps[i].experts:
            next_use[k] = i

    # warm-start seed (prefill-touched -> RAM, byte-budget greedy)
    for step in steps:
        if step.phase != "prefill":
            continue
        for key in step.experts:
            if key in vram or key in ram:
                continue
            eb = eb_for(key)
            if eb <= ram_bytes and ram_used + eb <= ram_bytes:
                ram[key] = -1
                ram_used += eb

    for tick, step in enumerate(steps):
        is_decode = step.phase == "decode"
        for key in step.experts:
            eb = eb_for(key)
            if key in vram:
                vram_hits += 1
                vram.move_to_end(key)
                if is_decode:
                    dec_vram_hits += 1
            elif key in ram:
                ram_hits += 1
                h2d += eb
                if is_decode:
                    dec_ram_hits += 1
                    dec_h2d += eb
                # admit to VRAM
                if vram_used + eb <= vram_bytes:
                    vram[key] = tick
                    vram_used += eb
                else:
                    while vram and vram_used + eb > vram_bytes:
                        vk, vt = vram.popitem(last=False)
                        vram_used -= eb_for(vk)
                        vram_ev += 1
                    if eb <= vram_bytes:
                        vram[key] = tick
                        vram_used += eb
                # demote victim to RAM if it fits (write-back demotion)
            else:
                ssd_misses += 1
                ssd += eb
                h2d += eb
                if is_decode:
                    dec_ssd_misses += 1
                    dec_ssd += eb
                    dec_h2d += eb
                if ram_used + eb <= ram_bytes:
                    ram[key] = tick
                    ram_used += eb
                else:
                    while ram and ram_used + eb > ram_bytes:
                        rk, rt = ram.popitem(last=False)
                        ram_used -= eb_for(rk)
                        ram_ev += 1
                    if eb <= ram_bytes:
                        ram[key] = tick
                        ram_used += eb
                if eb <= vram_bytes:
                    while vram and vram_used + eb > vram_bytes:
                        vk, vt = vram.popitem(last=False)
                        vram_used -= eb_for(vk)
                        vram_ev += 1
                    vram[key] = tick
                    vram_used += eb

    total_reqs = vram_hits + ram_hits + ssd_misses
    total_tokens = sum(s.weight for s in steps)
    return {
        "per_expert_bytes": "variable(MIXED)",
        "vram_budget_bytes": vram_bytes,
        "ram_budget_bytes": ram_bytes,
        "policy": policy + "+byte-budget",
        "requests": total_reqs,
        "tokens_total": total_tokens,
        "tokens_decode": sum(s.weight for s in steps if s.phase == "decode"),
        "vram_hits": vram_hits,
        "ram_hits": ram_hits,
        "ssd_misses": ssd_misses,
        "vram_hit_rate": vram_hits / total_reqs if total_reqs else 0,
        "ram_hit_rate": ram_hits / total_reqs if total_reqs else 0,
        "ssd_miss_rate": ssd_misses / total_reqs if total_reqs else 0,
        "combined_ram_vram_hit_rate": (vram_hits + ram_hits) / total_reqs if total_reqs else 0,
        "h2d_bytes_total": h2d,
        "ssd_bytes_total": ssd,
        "h2d_bytes_per_token": h2d / total_tokens if total_tokens else 0,
        "ssd_bytes_per_token": ssd / total_tokens if total_tokens else 0,
        "vram_evictions": vram_ev,
        "ram_evictions": ram_ev,
        "decode_requests": dec_vram_hits + dec_ram_hits + dec_ssd_misses,
        "decode_vram_hit_rate": _safe_div(dec_vram_hits,
                                          dec_vram_hits + dec_ram_hits + dec_ssd_misses),
        "decode_ram_hit_rate": _safe_div(dec_ram_hits,
                                         dec_vram_hits + dec_ram_hits + dec_ssd_misses),
        "decode_combined_hit_rate": _safe_div(dec_vram_hits + dec_ram_hits,
                                              dec_vram_hits + dec_ram_hits + dec_ssd_misses),
        "decode_ssd_miss_rate": _safe_div(dec_ssd_misses,
                                          dec_vram_hits + dec_ram_hits + dec_ssd_misses),
        "decode_ssd_bytes_per_token": _safe_div(dec_ssd, sum(s.weight for s in steps if s.phase == "decode")),
        "decode_h2d_bytes_per_token": _safe_div(dec_h2d, sum(s.weight for s in steps if s.phase == "decode")),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _safe_div(num, den):
    return num / den if den else 0.0


TARGET_TPS = [1, 2, 5, 10, 20]


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase B/C/D expert hierarchy simulator")
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    ap.add_argument("--metadata", default=DEFAULT_METADATA)
    ap.add_argument("--codec", default="MXFP4_CURRENT",
                    choices=list(CODECS) + ["MIXED"])
    ap.add_argument("--stq-bits", type=float, default=1.25,
                    help="STQ1_0_IDEALIZED payload bpw (default 1.25; fp16/256 scale adds 0.0625 -> 1.3125 total)")
    ap.add_argument("--vram-gib", type=float, required=True)
    ap.add_argument("--ram-gib", type=float, required=True)
    ap.add_argument("--policy", default="LRU", choices=["LRU", "LFU", "BELADY", "ADMISSION_LRU"])
    ap.add_argument("--mixed-map", default=None,
                    help="JSON file {\"layer,expert\": CODEC} for MIXED codec")
    ap.add_argument("--mixed-frac-stq", type=float, default=0.0,
                    help="MEMORY-ONLY: fraction of experts kept at MXFP4 (value-ranked)")
    ap.add_argument("--mixed-frac-iq2", type=float, default=0.0,
                    help="MEMORY-ONLY: fraction assigned IQ2_XXS")
    ap.add_argument("--sensitivity-file", default=None,
                    help="explicit sensitivity/value file for MIXED ranking (no fabrication)")
    ap.add_argument("--warm-start", action="store_true",
                    help="seed RAM tier with prefill-touched experts (perfect warmup; steady-state decode metrics)")
    ap.add_argument("--belady", action="store_true",
                    help="also compute offline Belady ceiling at each tier capacity")
    ap.add_argument("--reuse-distance", action="store_true",
                    help="include LRU stack-distance histogram (bucketed) in output")
    ap.add_argument("--min-h2d-gbps-note", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    steps = load_trace(args.trace)
    n_stq = sum(1 for s in steps for _ in s.experts)

    if args.codec == "MIXED":
        if args.mixed_map:
            cmap = parse_codec_map(args.mixed_map)
            assignment = None
            def eb_for(k, _c=cmap):
                return expert_bytes(_c.get(k, "STQ1_0_IDEALIZED"))
        else:
            vr = None
            if args.sensitivity_file:
                with open(args.sensitivity_file, "r", encoding="utf-8") as f:
                    sens = json.load(f)
                vr = {tuple(map(int, k.split(","))): v for k, v in sens.items()}
            assignment = assign_mixed(steps, args.mixed_frac_stq, args.mixed_frac_iq2, vr)
            amap = assignment
            def eb_for(k, _c=amap):
                return expert_bytes(_c.get(k, "STQ1_0_IDEALIZED"))
    else:
        bpw = args.stq_bits if args.codec == "STQ1_0_IDEALIZED" else None
        eb = expert_bytes(args.codec, bpw)
        def eb_for(k, _e=eb):
            return _e

    res = simulate(steps, eb_for, int(args.vram_gib * GIB), int(args.ram_gib * GIB),
                   policy=args.policy,
                   admission=(args.policy == "ADMISSION_LRU"),
                   trace_label=args.label or args.codec,
                   warm_start=args.warm_start)

    if args.belady:
        eb_probe = eb_for((0, 0))
        vram_experts = int(args.vram_gib * GIB) // eb_probe
        total_experts = int(args.ram_gib * GIB) // eb_probe + vram_experts
        b_vram, n = belady_offline_hit_rate(steps, vram_experts)
        b_all, _ = belady_offline_hit_rate(steps, max(total_experts, 1))
        b_dec, n_dec, h_dec = belady_decode_only_hit_rate(steps, max(total_experts, 1))
        res["belady_ceiling_vram_only"] = round(b_vram, 4)
        res["belady_ceiling_vram_plus_ram"] = round(b_all, 4)
        res["belady_decode_only_warm_ceiling"] = round(b_dec, 4)
        res["belady_decode_requests"] = n_dec
        res["belady_request_count"] = n

    if args.reuse_distance:
        rd = reuse_distance_distribution(steps)
        total = sum(rd.values())
        buckets = {"0": 0, "1-63": 0, "64-255": 0, "256-1023": 0, "1024-4095": 0,
                   "4096+": 0, "inf": 0}
        for d, c in rd.items():
            if d is math.inf or d == float("inf"):
                buckets["inf"] += c
            elif d == 0:
                buckets["0"] += c
            elif d < 64:
                buckets["1-63"] += c
            elif d < 256:
                buckets["64-255"] += c
            elif d < 1024:
                buckets["256-1023"] += c
            elif d < 4096:
                buckets["1024-4095"] += c
            else:
                buckets["4096+"] += c
        res["reuse_distance_buckets"] = buckets
        res["reuse_distance_first_touch_fraction"] = round(buckets["inf"] / total, 4) if total else None

    # Phase D: required bandwidth at each target TPS
    for tps in TARGET_TPS:
        bw = required_bandwidth(res["ssd_bytes_per_token"], res["h2d_bytes_per_token"], tps)
        res[f"required_ssd_GBps_at_{tps}tps"] = round(bw["ssd_GBps"], 3)
        res[f"required_pcie_GBps_at_{tps}tps"] = round(bw["pcie_GBps"], 3)

    res["max_storage_feasible_tps_ssd_10GBps"] = round(
        10e9 / res["ssd_bytes_per_token"], 3) if res["ssd_bytes_per_token"] > 0 else math.inf

    print(json.dumps(res, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
