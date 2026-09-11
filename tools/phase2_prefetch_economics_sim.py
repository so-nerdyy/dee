#!/usr/bin/env python3
"""Phase-2 prefetch economics — the (recall r, precision p, lead k) surface
scored against the sealed idle-gap bound.

Sibling of tools/phase2_legal_prefetch_eval.py (same sealed v50 journal, same
two-level tier model, same idle-gap admission discipline), but the predictor
is a PARAMETERIZED synthetic source instead of an enumerated legal source:

    predictor(r, p, k): for each decode target row j, emitted once at row
    e(j) = max(j - k, first_decode_row); names a Bernoulli-r fraction of the
    row's true demand set (correct) plus enough never-demanded-at-j experts of
    the same layer that emit precision = correct / (correct + wrong) = p.
    Wrong candidates are drawn from the layer's ever-demanded record set
    first (a trained predictor names records it has seen; every wrong name is
    a real 12.75 MiB record on the full 11,776-record universe), then topped
    up from the layer's remaining 0..255 ids. Emission order within a target
    row is a seeded shuffle (predictor confidence is not assumed to rank
    correctness).

ADMISSION (strictly-idle model, tightened vs the published eval):
  * Prefetch reads run only inside the per-row disk-idle gap (gap_ms x BW),
    never delay a demand read, and each completion is a real host-LRU
    insertion (tail eviction => pollution priced, not assumed away).
  * A candidate is admitted only if the gaps remaining before its deadline
    can physically cover the bytes left to read
    ((deadline - i) * gap_ms >= need_ms - progress). Hopeless pendings are
    dropped as soon as they become hopeless — a real prefetch engine knows
    deadlines, progress and the gap rate, so queuing the impossible is not
    a mechanism anyone would ship. (The published eval only refused to spend
    on the last possible gap; run oracle with --no-prune for the bridge.)
  * Pending records are served earliest-deadline-first each gap; finishing
    in-flight work precedes new candidates.

POLLUTION COST MODEL (per the R3 card): every completed prefetch is a host
LRU insert that may evict a record later demanded. The realized charge is
computed by diffing per-demand outcomes vs the no-prefetch baseline:
    base-miss -> arm-prefetch-hit  = conversion (a pread truly avoided)
    base-hit  -> arm-miss          = pollution fill (an EXTRA demand pread)
    base-miss -> arm-natural-hit   = survival gain (arm happened to keep it)
Device-tier demand order is identical in every arm (prefetch targets host
only), so device misses — hence H2D bytes — are invariant by construction;
the sim asserts this per cell instead of assuming it.

ADMISSION GATE (must reproduce before any cell is scored): single-level
host-LRU @682 slots/GPU on the full per-scope stream = 1391/1091 misses
(cuda0/cuda1); device priority-LRU @281 = 2284/2161; two-level total 2453 —
matching TIER_REPLAY_VALIDATION.md and research/phase2-legal-prefetch
results (sealed live anchors are 1390/1091 and 2285/2159; the sim numbers
are the realized tier counters, ±1 convention documented there).

Outputs -> research/prior-art/r03-results/
    prefetch_rpk_grid.csv      one row per (r,p,k,bw,gap) cell
    prefetch_rpk_summary.json  validation + constants + per-cell table +
                               bound analysis
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import random
import statistics
from collections import defaultdict
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOURNAL = (REPO / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
           "v50-evidence-20260829T195940Z/routed_experts.jsonl")
OUT = REPO / "research/prior-art/r03-results"

RECORD_BYTES = 13_369_344          # 12.75 MiB sealed DEE4 record
GIB = 1 << 30
SPLIT_GPU0 = 22                    # cuda0: layers 0-21, cuda1: 22-42
DEVICE_SLOTS = 281                 # sealed 3.5 GiB arena / 12.75 MiB
HOST_SLOTS = 682                   # sealed 8.5 GiB pack budget / 12.75 MiB
N_LAYERS = 43
FIRST_DECODE_ROW = 43              # record_index of (step 1, layer 0)

# Phase-1 measured anchors (same as phase2_legal_prefetch_eval.py)
W1 = {0: 0.0, 1: 37.2, 2: 66.1, 3: 105.8, 4: 139.6, 5: 177.2, 6: 221.6}
W1_SLOPE = 35.4                    # ms per extra miss (linear fit of W1)
ROW_WALL_MS = 102.7                # sealed decode row mean (66.233 s / 645)
GAP_CONSERVATIVE_MS = 23.0         # derived no-legal-work gap
GAP_MAX_MS = 37.6                  # all non-fill time
BWS = (0.29, 0.33, 0.37)           # measured /tmp bank ceiling range

# Sealed admission-gate counters (TIER_REPLAY_VALIDATION.md)
GATE_HOST_MISS = {0: 1391, 1: 1091}     # sim single-level; sealed live 1390/1091
GATE_DEV_MISS = {0: 2284, 1: 2161}      # sim; sealed live 2285/2159
GATE_TWOLEVEL_PREADS = 2453             # == p2c sim total


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_batches(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs.sort(key=lambda r: r["record_index"])
    out = []
    for r in recs:
        uniq = sorted({e for row in r["expert_ids_rank_order"] for e in row})
        gpu = 0 if r["layer"] < SPLIT_GPU0 else 1
        out.append({"step": r["forward_step"], "layer": r["layer"],
                    "gpu": gpu, "phase": r["phase"], "experts": uniq})
    return out


class LRU:
    """Plain LRU; touch returns (hit, evicted_key_or_None)."""

    def __init__(self, slots):
        self.slots, self.dq, self.pos = slots, deque(), set()
        self.evictions = 0

    def resident(self, key):
        return key in self.pos

    def touch(self, key):
        if key in self.pos:
            self.dq.remove(key)
            self.dq.appendleft(key)
            return True, None
        ev = None
        if self.slots > 0:
            if len(self.dq) >= self.slots:
                ev = self.dq.pop()
                self.pos.discard(ev)
                self.evictions += 1
            self.dq.appendleft(key)
            self.pos.add(key)
        return False, ev


class PriorityLRU:
    """Sealed device semantics: score = last_used + prio*2**20."""

    def __init__(self, slots):
        self.slots, self.blocks, self.tick, self.evictions = slots, {}, 0, 0

    def resident(self, key):
        return key in self.blocks

    def touch(self, key, prio):
        self.tick += 1
        if key in self.blocks:
            self.blocks[key] = [self.tick, prio]
            return True
        if self.slots > 0:
            while len(self.blocks) >= self.slots:
                victim = min(self.blocks, key=lambda k: (
                    self.blocks[k][0] + self.blocks[k][1] * (1 << 20)))
                del self.blocks[victim]
                self.evictions += 1
            self.blocks[key] = [self.tick, prio]
        return False


def future_use_map(batches):
    """future_use[i][key] = earliest demand row > i (per flattened row)."""
    n = len(batches)
    future_use = []
    for i in range(n):
        nxt = {}
        for k in range(i + 1, n):
            nb = batches[k]
            for e in nb["experts"]:
                key = (nb["layer"], e)
                if key not in nxt:
                    nxt[key] = k
        future_use.append(nxt)
    return future_use


def replay(batches, emit_map=None, gap_ms=0.0, bw_gibps=0.33,
           prune_hopeless=True, future_use=None):
    """Two-level replay with idle-gap prefetch.

    emit_map: row_index i -> list of candidate dicts
        {"key": (layer,expert), "gpu": g, "deadline": j, "correct": bool}
    emitted during the idle gap after row i. correct marks membership in the
    target row's demand set (accounting only; the engine can't see it).

    Returns (rows, stats). rows[i] carries per-demand outcomes:
        outcomes[key] = 'D' device hit | 'H' host hit | 'M' host miss
                        | 'P' host hit on a prefetch-inserted record.
    """
    dev = {0: PriorityLRU(DEVICE_SLOTS), 1: PriorityLRU(DEVICE_SLOTS)}
    host = {0: LRU(HOST_SLOTS), 1: LRU(HOST_SLOTS)}
    need_ms = RECORD_BYTES / (bw_gibps * GIB) * 1000.0 if bw_gibps else 0.0
    rows = []
    pf_done = pf_hit_events = 0
    pf_keys = set()                # records currently marked pf-inserted
    pf_done_keys = set()
    pf_used_keys = set()
    pending = {}                   # key -> [gpu, progress_ms, deadline, correct]
    fate = {}                      # (key, deadline) -> terminal state
    emitted = admitted = dropped_infeasible_at_emit = 0
    emitted_correct = emitted_wrong = 0
    starved = partial = hopeless = completed = 0
    partial_ms_wasted = 0.0
    hopeless_skips = 0             # last-gap refuse-to-spend events
    pending_depth_samples = []
    gap_budget_total = gap_budget_spent = 0.0
    pf_evictions = {0: 0, 1: 0}
    evicted_live_by_prefetch = {0: 0, 1: 0}
    completion_leads = []          # deadline - completion_row per completion
    heap = []                      # persistent (deadline, key) min-heap;
                                   # stale entries lazily skipped

    for i, b in enumerate(batches):
        layer, gpu, K = b["layer"], b["gpu"], len(b["experts"])
        # pending records whose deadline row just arrived unfinished
        for key in [k for k, v in pending.items() if v[2] <= i]:
            if pending[key][1] > 1e-9:
                partial += 1
                partial_ms_wasted += pending[key][1]
                fate[(key, pending[key][2])] = "partial"
            else:
                starved += 1
                fate[(key, pending[key][2])] = "starved"
            del pending[key]
        d_hit = h_hit = h_miss = pf_hit = 0
        outcomes = {}
        for j, e in enumerate(b["experts"]):
            key = (layer, e)
            if key in pending:
                # demanded while partially fetched: partial work wasted
                v = pending[key]
                if v[1] > 1e-9:
                    partial += 1
                    partial_ms_wasted += v[1]
                    fate[(key, v[2])] = "partial_preempted"
                else:
                    starved += 1
                    fate[(key, v[2])] = "starved"
                del pending[key]
            if dev[gpu].touch(key, K - j):
                d_hit += 1
                outcomes[key] = 'D'
            elif host[gpu].resident(key):
                host[gpu].touch(key)
                h_hit += 1
                if key in pf_keys:
                    pf_hit += 1
                    pf_used_keys.add(key)
                    outcomes[key] = 'P'
                else:
                    outcomes[key] = 'H'
            else:
                host[gpu].touch(key)
                h_miss += 1
                outcomes[key] = 'M'
        pf_hit_events += pf_hit
        rows.append({"row": i, "step": b["step"], "layer": layer, "gpu": gpu,
                     "requests": K, "device_hits": d_hit,
                     "host_hits": h_hit, "host_misses": h_miss,
                     "prefetch_hits_at_demand": pf_hit,
                     "phase": b["phase"], "outcomes": outcomes})

        # ---- idle-gap prefetch admission (never delays demand reads) ----
        if emit_map is None or gap_ms <= 0 or b["phase"] != "decode":
            continue
        budget_ms = gap_ms
        gap_budget_total += gap_ms
        # new emissions for this gap's row
        for cand in emit_map.get(i, ()):
            emitted += 1
            if cand["correct"]:
                emitted_correct += 1
            else:
                emitted_wrong += 1
            key = cand["key"]
            if key in pending:
                # already in-flight for an earlier/equal deadline: merged
                fate[(key, cand["deadline"])] = "merged"
                continue
            if (prune_hopeless
                    and (cand["deadline"] - i) * gap_ms < need_ms - 1e-9):
                dropped_infeasible_at_emit += 1
                fate[(key, cand["deadline"])] = "infeasible"
                continue
            pending[key] = [cand["gpu"], 0.0, cand["deadline"],
                            cand["correct"]]
            heapq.heappush(heap, (cand["deadline"], key))
            admitted += 1
        pending_depth_samples.append(len(pending))
        # earliest-deadline-first service; entries that remain pending are
        # re-pushed (deadline order is stable across rows)
        while budget_ms > 0 and heap:
            _dl_top, key = heapq.heappop(heap)
            if key not in pending:      # completed/demanded/dropped/stale
                continue
            g, prog, dl, _corr = pending[key]
            if dev[g].resident(key) or host[g].resident(key):
                del pending[key]         # became resident anyway: free skip
                fate[(key, dl)] = "skipped_resident"
                continue
            # feasible iff remaining budget this gap plus every full future
            # gap before the deadline can cover the bytes still needed
            if prune_hopeless and (dl - i - 1) * gap_ms + budget_ms \
                    < need_ms - prog - 1e-9:
                # cannot finish even using every remaining gap fully
                partial_ms_wasted += prog
                if prog > 1e-9:
                    partial += 1
                    fate[(key, dl)] = "hopeless_partial"
                else:
                    fate[(key, dl)] = "hopeless"
                hopeless += 1
                del pending[key]
                continue
            if not prune_hopeless and dl <= i + 1 \
                    and prog + budget_ms < need_ms - 1e-9:
                hopeless_skips += 1
                heapq.heappush(heap, (dl, key))
                continue                 # last gap, can't complete: pass
            spend = min(need_ms - prog, budget_ms)
            prog += spend
            budget_ms -= spend
            gap_budget_spent += spend
            if prog >= need_ms - 1e-9:
                _hit, ev = host[g].touch(key)   # real LRU insert: pollution
                if ev is not None:
                    pf_evictions[g] += 1
                    if future_use is not None and ev in future_use[i]:
                        evicted_live_by_prefetch[g] += 1
                pf_keys.add(key)
                pf_done_keys.add(key)
                pf_done += 1
                completed += 1
                fate[(key, dl)] = "completed"
                completion_leads.append(dl - i)
                del pending[key]
            else:
                pending[key][1] = prog
                heapq.heappush(heap, (dl, key))

    stats = {
        "emitted_candidates": emitted,
        "emitted_correct": emitted_correct,
        "emitted_wrong": emitted_wrong,
        "admitted": admitted,
        "dropped_infeasible_at_emit": dropped_infeasible_at_emit,
        "prefetches_completed": pf_done,
        "prefetch_hit_events": pf_hit_events,
        "prefetches_used_keys": len(pf_used_keys),
        "prefetches_completed_never_used": len(pf_done_keys - pf_used_keys),
        "pending_starved_zero_progress": starved,
        "pending_partial_at_deadline": partial,
        "pending_partial_ms_wasted": round(partial_ms_wasted, 1),
        "pending_dropped_hopeless": hopeless,
        "hopeless_lastgap_skips": hopeless_skips,
        "pending_depth_mean": round(statistics.mean(pending_depth_samples), 2)
                              if pending_depth_samples else 0.0,
        "pending_depth_max": max(pending_depth_samples)
                             if pending_depth_samples else 0,
        "gap_budget_ms_total": round(gap_budget_total, 1),
        "gap_budget_ms_spent": round(gap_budget_spent, 1),
        "gap_budget_utilization": round(
            gap_budget_spent / gap_budget_total, 4)
            if gap_budget_total else 0.0,
        "host_evictions": {g: host[g].evictions for g in (0, 1)},
        "host_evictions_by_prefetch": pf_evictions,
        "evicted_live_by_prefetch": evicted_live_by_prefetch,
        "device_misses": {g: sum(r["requests"] - r["device_hits"]
                                 for r in rows if r["gpu"] == g)
                          for g in (0, 1)},
        "completion_lead_rows_mean": round(
            statistics.mean(completion_leads), 2) if completion_leads else 0.0,
        "need_ms_per_record": round(need_ms, 2),
        "fate_counts": dict(_count_values(fate)),
    }
    return rows, stats


def _count_values(d):
    from collections import Counter
    return Counter(d.values())


def fill_wall_ms(m):
    return W1[m] if m in W1 else W1_SLOPE * m


def build_rpk_emit_map(batches, r, p, k, layer_pool):
    """emit_map for the synthetic (r,p,k) predictor.

    Per-target-row draws are seeded by (journal,row) only, so the chosen
    correct/wrong sets are identical across k/bw/gap cells and nested in r
    (correct_j(r') superset of correct_j(r) for r' > r) and in p.
    """
    n = len(batches)
    emit = defaultdict(list)
    for j in range(FIRST_DECODE_ROW, n):
        b = batches[j]
        if b["phase"] != "decode":
            continue
        e_row = max(j - k, FIRST_DECODE_ROW)
        if e_row >= j:
            continue
        D = b["experts"]
        rng = random.Random(f"rpk:{j}:members")
        u = {e: rng.random() for e in D}
        correct = sorted(e for e in D if u[e] < r)
        n_corr = len(correct)
        n_wrong = int(round(n_corr * (1.0 / p - 1.0))) if p < 1.0 else 0
        wrong = []
        if n_wrong > 0:
            dset = set(D)
            pool = [e for e in layer_pool[b["layer"]] if e not in dset]
            pset = set(pool)
            rest = [e for e in range(256)
                    if e not in dset and e not in pset]
            wrng = random.Random(f"rpk:{j}:wrong")
            wrng.shuffle(pool)
            wrng.shuffle(rest)
            wrong = (pool + rest)[:n_wrong]
        cands = ([{"key": (b["layer"], e), "gpu": b["gpu"], "deadline": j,
                   "correct": True} for e in correct] +
                 [{"key": (b["layer"], e), "gpu": b["gpu"], "deadline": j,
                   "correct": False} for e in wrong])
        crng = random.Random(f"rpk:{j}:order")
        crng.shuffle(cands)
        emit[e_row].extend(cands)
    return emit


def build_oracle_emit_map(batches, base_miss_keys, future_use):
    """Bound-only oracle: every future baseline host-miss, earliest-deadline
    first, re-emitted from every decode gap (the published eval's oracle)."""
    n = len(batches)
    emit = defaultdict(list)
    for i in range(FIRST_DECODE_ROW, n):
        if batches[i]["phase"] != "decode":
            continue
        nxt = future_use[i]
        for key in sorted(nxt, key=lambda kk: nxt[kk]):
            use_i = nxt[key]
            if key not in base_miss_keys[use_i]:
                continue
            emit[i].append({"key": key, "gpu": batches[use_i]["gpu"],
                            "deadline": use_i, "correct": True})
    return emit


def diff_outcomes(base_rows, arm_rows):
    """Per-demand outcome diff, decode rows only."""
    conv = poll = surv = dev_mismatch = 0
    for br, ar in zip(base_rows, arm_rows):
        if br["phase"] != "decode":
            continue
        for key, bo in br["outcomes"].items():
            ao = ar["outcomes"][key]
            if bo == 'D':
                if ao != 'D':
                    dev_mismatch += 1
            elif bo == 'M' and ao == 'P':
                conv += 1
            elif bo == 'M' and ao == 'H':
                surv += 1
            elif bo == 'H' and ao == 'M':
                poll += 1
    return {"conversions": conv, "pollution_extra_fills": poll,
            "survival_gains": surv, "device_outcome_mismatches": dev_mismatch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quick", action="store_true",
                    help="small grid for smoke tests")
    ap.add_argument("--no-prune-oracle", action="store_true",
                    help="also run oracle under published no-prune admission")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jsha = sha256_of(JOURNAL)
    batches = load_batches(JOURNAL)
    n = len(batches)
    fut = future_use_map(batches)

    # ---------------- baseline + sealed admission gate --------------------
    base_rows, _ = replay(batches)
    per_gpu_dev_miss = {g: sum(r["requests"] - r["device_hits"]
                               for r in base_rows if r["gpu"] == g)
                        for g in (0, 1)}
    single_level_miss = {}
    for g in (0, 1):
        c = LRU(HOST_SLOTS)
        m = 0
        for b in batches:
            if b["gpu"] != g:
                continue
            for e in b["experts"]:
                hit, _ = c.touch((b["layer"], e))
                if not hit:
                    m += 1
        single_level_miss[g] = m
    base_decode_preads = sum(r["host_misses"] for r in base_rows
                             if r["phase"] == "decode")
    base_preads_total = sum(r["host_misses"] for r in base_rows)
    base_decode_wall_s = sum(fill_wall_ms(r["host_misses"])
                             for r in base_rows
                             if r["phase"] == "decode") / 1000.0
    base_miss_keys = []
    for r_ in base_rows:
        base_miss_keys.append({k for k, o in r_["outcomes"].items()
                               if o == 'M'})
    gate = {
        "journal_sha256": jsha,
        "sim_singlelevel_host_miss": single_level_miss,
        "expected_host_miss": GATE_HOST_MISS,
        "sim_device_miss": per_gpu_dev_miss,
        "expected_device_miss": GATE_DEV_MISS,
        "sim_twolevel_preads_total": base_preads_total,
        "expected_twolevel_preads": GATE_TWOLEVEL_PREADS,
        "sealed_live_anchors": {"host_pack_miss": [1390, 1091],
                                "device_miss": [2285, 2159]},
        "pass": (single_level_miss == GATE_HOST_MISS
                 and per_gpu_dev_miss == GATE_DEV_MISS
                 and base_preads_total == GATE_TWOLEVEL_PREADS),
    }
    print("ADMISSION GATE:", json.dumps(gate, indent=1))
    if not gate["pass"]:
        raise SystemExit("sealed-counter admission gate FAILED; "
                         "refusing to score variants")

    layer_pool = defaultdict(set)
    for b in batches:
        for e in b["experts"]:
            layer_pool[b["layer"]].add(e)

    # ---------------- grid -------------------------------------------------
    if args.quick:
        RS, PS, KS = (0.5, 1.0), (0.5, 1.0), (2, 6)
        GAPS, BWS_ = (GAP_CONSERVATIVE_MS,), (0.33,)
    else:
        RS = (0.1, 0.25, 0.5, 0.75, 1.0)
        PS = (0.1, 0.25, 0.5, 0.75, 1.0)
        KS = (1, 2, 3, 6, 12, 43)
        GAPS = (GAP_CONSERVATIVE_MS, GAP_MAX_MS)
        BWS_ = BWS

    grid_rows = []
    for gap_ms in GAPS:
        for bw in BWS_:
            need_ms = RECORD_BYTES / (bw * GIB) * 1000.0
            gap_rec = gap_ms / need_ms
            for k in KS:
                for r in RS:
                    for p in PS:
                        emit = build_rpk_emit_map(
                            batches, r, p, k, layer_pool)
                        rows, st = replay(batches, emit_map=emit,
                                          gap_ms=gap_ms, bw_gibps=bw,
                                          future_use=fut)
                        d = diff_outcomes(base_rows, rows)
                        arm_dec = sum(rr["host_misses"] for rr in rows
                                      if rr["phase"] == "decode")
                        arm_dev = {g: sum(rr["requests"] - rr["device_hits"]
                                          for rr in rows if rr["gpu"] == g)
                                   for g in (0, 1)}
                        wall_saved = sum(
                            fill_wall_ms(b_["host_misses"])
                            - fill_wall_ms(ar["host_misses"])
                            for b_, ar in zip(base_rows, rows)
                            if ar["phase"] == "decode")
                        # demand-set recall/precision realized at emit time
                        tgt = [j for j in range(FIRST_DECODE_ROW, n)
                               if max(j - k, FIRST_DECODE_ROW) < j]
                        dem = sum(len(batches[j]["experts"]) for j in tgt)
                        # emitted-correct candidates whose target demand was
                        # a baseline host miss — the only predictions that
                        # can convert a pread (hit-bound ones free-skip)
                        pred_on_miss = sum(
                            1 for cands in emit.values() for c in cands
                            if c["correct"]
                            and c["key"] in base_miss_keys[c["deadline"]])
                        fate = st["fate_counts"]
                        net_saved = base_decode_preads - arm_dec
                        grid_rows.append({
                            "gap_ms": gap_ms, "bw_gibps": bw,
                            "need_ms_per_record": round(need_ms, 2),
                            "gap_records_per_row": round(gap_rec, 3),
                            "r": r, "p": p, "k": k,
                            "target_rows": len(tgt),
                            "emitted": st["emitted_candidates"],
                            "emitted_correct": st["emitted_correct"],
                            "emitted_wrong": st["emitted_wrong"],
                            "emit_recall_realized": round(
                                st["emitted_correct"] / dem, 4),
                            "emit_precision_realized": round(
                                st["emitted_correct"] /
                                max(1, st["emitted_candidates"]), 4),
                            "admitted": st["admitted"],
                            "dropped_infeasible_at_emit":
                                st["dropped_infeasible_at_emit"],
                            "pf_completed": st["prefetches_completed"],
                            "pf_hit_events": st["prefetch_hit_events"],
                            "pf_used_keys": st["prefetches_used_keys"],
                            "pf_completed_never_used":
                                st["prefetches_completed_never_used"],
                            "pred_correct_on_miss": pred_on_miss,
                            "ready_before_use_frac": round(
                                d["conversions"] / pred_on_miss, 4)
                                if pred_on_miss else 0.0,
                            "conversions": d["conversions"],
                            "survival_gains": d["survival_gains"],
                            "pollution_extra_fills":
                                d["pollution_extra_fills"],
                            "fate_completed": fate.get("completed", 0),
                            "fate_skipped_resident":
                                fate.get("skipped_resident", 0),
                            "fate_merged": fate.get("merged", 0),
                            "fate_starved": fate.get("starved", 0),
                            "fate_partial": (fate.get("partial", 0)
                                             + fate.get("partial_preempted", 0)),
                            "fate_hopeless": (fate.get("hopeless", 0)
                                              + fate.get("hopeless_partial", 0)),
                            "fate_infeasible": fate.get("infeasible", 0),
                            "device_outcome_mismatches":
                                d["device_outcome_mismatches"],
                            "starved": st["pending_starved_zero_progress"],
                            "partial_at_deadline":
                                st["pending_partial_at_deadline"],
                            "partial_ms_wasted":
                                st["pending_partial_ms_wasted"],
                            "dropped_hopeless":
                                st["pending_dropped_hopeless"],
                            "hopeless_lastgap_skips":
                                st["hopeless_lastgap_skips"],
                            "pending_depth_mean": st["pending_depth_mean"],
                            "pending_depth_max": st["pending_depth_max"],
                            "gap_budget_utilization":
                                st["gap_budget_utilization"],
                            "host_evictions0": st["host_evictions"][0],
                            "host_evictions1": st["host_evictions"][1],
                            "pf_evictions0":
                                st["host_evictions_by_prefetch"][0],
                            "pf_evictions1":
                                st["host_evictions_by_prefetch"][1],
                            "evicted_live_pf0":
                                st["evicted_live_by_prefetch"][0],
                            "evicted_live_pf1":
                                st["evicted_live_by_prefetch"][1],
                            "completion_lead_rows_mean":
                                st["completion_lead_rows_mean"],
                            "lead_ms_nominal": round(k * ROW_WALL_MS, 1),
                            "decode_preads": arm_dec,
                            "preads_saved_net": net_saved,
                            "ssd_gib_avoided_net": round(
                                net_saved * RECORD_BYTES / GIB, 3),
                            "ssd_gib_converted_gross": round(
                                d["conversions"] * RECORD_BYTES / GIB, 3),
                            "ssd_gib_pollution_cost": round(
                                d["pollution_extra_fills"] *
                                RECORD_BYTES / GIB, 3),
                            "device_misses0": arm_dev[0],
                            "device_misses1": arm_dev[1],
                            "extra_h2d_gib": round(
                                ((arm_dev[0] + arm_dev[1])
                                 - (per_gpu_dev_miss[0]
                                    + per_gpu_dev_miss[1]))
                                * RECORD_BYTES / GIB, 4),
                            "decode_fill_wall_s": round(
                                base_decode_wall_s - wall_saved / 1000.0, 3),
                            "wall_saved_s": round(wall_saved / 1000.0, 3),
                        })

    # ---------------- oracle bound arms ------------------------------------
    oracle_rows = []
    for gap_ms in GAPS:
        for bw in BWS_:
            for prune in ([True, False] if args.no_prune_oracle else [True]):
                emit = build_oracle_emit_map(batches, base_miss_keys, fut)
                rows, st = replay(batches, emit_map=emit, gap_ms=gap_ms,
                                  bw_gibps=bw, prune_hopeless=prune,
                                  future_use=fut)
                d = diff_outcomes(base_rows, rows)
                arm_dec = sum(rr["host_misses"] for rr in rows
                              if rr["phase"] == "decode")
                wall_saved = sum(
                    fill_wall_ms(b_["host_misses"])
                    - fill_wall_ms(ar["host_misses"])
                    for b_, ar in zip(base_rows, rows)
                    if ar["phase"] == "decode")
                oracle_rows.append({
                    "gap_ms": gap_ms, "bw_gibps": bw, "prune_hopeless": prune,
                    "decode_preads": arm_dec,
                    "preads_saved_net": base_decode_preads - arm_dec,
                    "conversions": d["conversions"],
                    "pollution_extra_fills": d["pollution_extra_fills"],
                    "wall_saved_s": round(wall_saved / 1000.0, 3),
                    "pf_completed": st["prefetches_completed"],
                    "gap_budget_utilization": st["gap_budget_utilization"],
                })

    with open(out / "prefetch_rpk_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)
    with open(out / "prefetch_oracle_bound.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(oracle_rows[0].keys()))
        w.writeheader()
        w.writerows(oracle_rows)

    # bound analysis: best cell per (gap,bw) + fraction of oracle
    bound = {(o["gap_ms"], o["bw_gibps"]): o["wall_saved_s"]
             for o in oracle_rows if o["prune_hopeless"]}
    for gr in grid_rows:
        bnd = bound.get((gr["gap_ms"], gr["bw_gibps"]))
        gr["frac_of_oracle_bound"] = (round(gr["wall_saved_s"] / bnd, 4)
                                      if bnd else None)

    # rewrite CSV with the bound fraction appended
    with open(out / "prefetch_rpk_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)

    best = max(grid_rows, key=lambda g: g["wall_saved_s"])
    summary = {
        "journal_sha256": jsha,
        "admission_gate": gate,
        "baseline": {"decode_preads": base_decode_preads,
                     "total_preads": base_preads_total,
                     "decode_fill_wall_s": round(base_decode_wall_s, 3)},
        "constants": {"record_bytes": RECORD_BYTES,
                      "device_slots_per_gpu": DEVICE_SLOTS,
                      "host_slots_per_gpu": HOST_SLOTS,
                      "row_wall_ms": ROW_WALL_MS,
                      "w1_table_ms": W1, "w1_slope_ms_per_miss": W1_SLOPE},
        "oracle_bound_cells": oracle_rows,
        "best_cell": best,
        "n_cells": len(grid_rows),
    }
    with open(out / "prefetch_rpk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {len(grid_rows)} grid cells + "
          f"{len(oracle_rows)} oracle cells -> {out}")
    print("best cell:", json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
