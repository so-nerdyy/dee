#!/usr/bin/env python3
"""Phase-2 concurrent-fill quantification — serial vs bounded-lane cold-fill
wall on the sealed v50/v60 route journal. ANALYSIS ONLY; no engine code.

Question (W1-T2): on branch research/phase2-concurrent-fill (Luna's
phase2-integration-lru-fix head), prepare_fp4_experts() early-returns when
phase2.enabled && phase2.host_enabled (engine.cpp:2794), so every
stage_expert -> DeviceExpertTier::stage -> HostExpertTier::acquire ->
IdentityCodec::materialize -> ColdExpertStore::read -> ExpertStore::
materialize is ONE synchronous 12.75 MiB pread on the calling thread.
Would a bounded pre-acquire worker pool (the T9 repair) cut fill wall?

Conventions/anchors (all Phase-1 measured, research/route-pipeline/
results/fill_matrix_ingest.json + fill_verdict.json on 2x T4):

  RECORD_BYTES      12.75 MiB DEE4 packed FP4 record (sealed dee4 metadata)
  SVC_TASK          18.91 ms — task-prescribed "single-flight" anchor
                    (lanes1 replay per-REQUEST mean; hit-weighted: ~47% of
                    requests are host hits with ~0 service)
  SVC_MISS_LANES1   35.51 ms — measured per-MISS single-flight mean (lanes1)
  SVC_MISS_QD1      33.76 ms — measured per-MISS single-flight mean (qd1)
  SVC_MISS_PROD3    107.76 ms — measured per-MISS under lanes3+QD6 sharing
  W1(m)/W3(m)       measured per-batch wall vs misses-in-batch tables
  B1 / B_AGG3       single-stream 0.35 GiB/s / 3-lane aggregate 0.29 GiB/s
                    (device ceiling 0.29-0.37 GiB/s on /tmp; rider flat
                    lanes1-8 => one stream already saturates the bank)
  RESERVE_MS        7.08 ms/batch — Phase-1 fill-pool reservation bucket
  ENQUEUE_MS        14.63 ms/batch — sealed StageEnqueueWait decode mean
                    (engine-side transfer-submit cost; stays on caller)

Journal: sealed v50 canonical route journal (routed_experts.jsonl, 688
records = 16 forwards x 43 layers, topk 6; v60 journal is content-identical
on expert_ids/layer/forward_step). GPU split: layer<22 cuda0 (per
phase2_ws_policy_sim_v4). Per record the engine stages the sorted unique
expert set once (batch dedup; validated against sealed counters below).

Tier model (phase2 path, Luna branch):
  stage() -> device hit? (VramCacheManager arena, 281 slots sealed) : no
             host consult
          -> host.acquire on device miss (HostExpertTier, plain LRU, 682
             slots = sealed 8.5 GiB budget) -> Ready hit or Filling pread.
  host_miss == a real pread. Serial wall ~= sum of per-miss service.

Outputs -> research/phase2-concurrent-fill/results/:
  per_batch_miss_counts.csv   every batch's host-miss count (the workload)
  wall_by_miss_count.csv      measured W1/W3(m) tables + fair-share model
  fill_wall_models.csv        per-model response/token wall totals
  bandwidth_sweep.csv         lanes x device-ceiling sensitivity
  validation.json             sealed-vs-sim counters, anchors, gate verdict
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOURNAL = (REPO / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
           "v50-evidence-20260829T195940Z/routed_experts.jsonl")
JOURNAL_V60 = (REPO / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
               "v60-evidence-20260901T040935Z/routed_experts.jsonl")
OUT = REPO / "research/phase2-concurrent-fill/results"

RECORD_BYTES = 13_369_344          # 12.75 MiB sealed DEE4 record
GIB = 1 << 30
SPLIT_GPU0 = 22                    # cuda0: layers 0-21, cuda1: 22-42
DEVICE_SLOTS = 281                 # sealed 3.5 GiB arena / 12.75 MiB
HOST_SLOTS = 682                   # sealed 8.5 GiB pack budget / 12.75 MiB
N_TOKENS = 16                      # sealed forwards (1 prefill + 15 decode)
N_DECODE = 15

# ---- Phase-1 measured anchors (fill_matrix_ingest.json, 2xT4 /tmp bank) ----
SVC_TASK = 18.91                   # ms, task anchor (lanes1 per-request mean)
SVC_MISS_LANES1 = 35.51            # ms, measured per-miss single flight
SVC_MISS_QD1 = 33.76               # ms, measured per-miss single flight
SVC_MISS_PROD3 = 107.76            # ms, measured per-miss lanes3 QD6
B1_GIBPS = 0.35                    # single-stream achieved (lanes1 replay)
B_AGG3_GIBPS = 0.29                # 3-lane aggregate achieved (production)
B_CEILING_HI = 0.37                # best observed bank ceiling (qd1/rider)
RESERVE_MS = 7.08                  # Phase-1 replay reserve mean per batch
ENQUEUE_MS_DECODE = 14.63          # sealed StageEnqueueWait decode mean/batch
ENQUEUE_MS_ALL = 19.58             # sealed StageEnqueueWait all-rows mean

# measured per-batch wall tables by misses-in-batch (fill-live replay):
W1 = {0: 0.0, 1: 37.2, 2: 66.1, 3: 105.8, 4: 139.6, 5: 177.2, 6: 221.6}
W3 = {0: 0.0, 1: 45.1, 2: 85.9, 3: 127.1, 4: 173.3, 5: 216.3, 6: 261.9}
W1_SLOPE = 35.4                    # ms per extra miss (linear fit of W1)
W3_SLOPE = 43.4                    # ms per extra miss (linear fit of W3)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- stream

def load_batches(path):
    """(forward_step, layer, gpu, [sorted unique experts]) — engine-dedup
    stream convention from phase2_ws_policy_sim_v4 (validated vs sealed)."""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs.sort(key=lambda r: r["record_index"])
    batches = []
    for r in recs:
        uniq = sorted({e for row in r["expert_ids_rank_order"] for e in row})
        gpu = 0 if r["layer"] < SPLIT_GPU0 else 1
        batches.append((r["forward_step"], r["layer"], gpu, uniq))
    return batches


# ---------------------------------------------------------------- caches

class LRU:
    """Plain LRU — sealed host-pack / phase2 PlainLruHostPlacementPolicy."""
    def __init__(self, slots):
        self.slots, self.dq, self.pos, self.evictions = slots, deque(), set(), 0

    def touch(self, key):
        if key in self.pos:
            self.dq.remove(key)
            self.dq.appendleft(key)
            return True
        if self.slots > 0:
            if len(self.dq) >= self.slots:
                self.pos.discard(self.dq.pop())
                self.evictions += 1
            self.dq.appendleft(key)
            self.pos.add(key)
        return False


class PriorityLRU:
    """Sealed device semantics: score = last_used + prio*2**20 refreshed on
    hit (vram_cache PRIORITY_WEIGHT; engine passes K-k in dedup order)."""
    def __init__(self, slots):
        self.slots, self.blocks, self.tick, self.evictions = slots, {}, 0, 0

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


def replay_tiers(batches, device_policy="engine_priority_lru",
                 device_slots=DEVICE_SLOTS, host_slots=HOST_SLOTS):
    """Two-level phase2 replay per GPU. Returns per-batch counters."""
    dev = {0: (PriorityLRU(device_slots) if device_policy ==
               "engine_priority_lru" else LRU(device_slots)),
           1: (PriorityLRU(device_slots) if device_policy ==
               "engine_priority_lru" else LRU(device_slots))}
    host = {0: LRU(host_slots), 1: LRU(host_slots)}
    rows = []
    for step, layer, gpu, experts in batches:
        K = len(experts)
        d_hit = h_hit = h_miss = 0
        for j, e in enumerate(experts):
            key = (layer, e)
            prio = K - j
            if dev[gpu].touch(key, prio) if device_policy == \
                    "engine_priority_lru" else dev[gpu].touch(key):
                d_hit += 1
            elif host[gpu].touch(key):
                h_hit += 1
            else:
                h_miss += 1
        rows.append({"step": step, "layer": layer, "gpu": gpu,
                     "requests": K, "device_hits": d_hit,
                     "host_hits": h_hit, "host_misses": h_miss,
                     "phase": "prefill" if step == 0 else "decode"})
    return rows


def single_level_host_rows(batches, host_slots=HOST_SLOTS):
    """Legacy-equivalent stream: every request consults the host pack
    (prepare_fp4_experts/get_if_present on all requests, not just device
    misses). Produces per-batch pack-miss counts for the serial-wall
    comparison against the sealed 4-lane fill_batch_wall_ms."""
    cache = {0: LRU(host_slots), 1: LRU(host_slots)}
    rows = []
    for step, layer, gpu, experts in batches:
        m = 0
        for e in experts:
            if not cache[gpu].touch((layer, e)):
                m += 1
        rows.append({"step": step, "layer": layer, "gpu": gpu,
                     "requests": len(experts), "host_misses": m,
                     "phase": "prefill" if step == 0 else "decode"})
    return rows


# ---------------------------------------------------------------- walls

def makespan_fairshare(n_jobs, lanes, b1_gibps, b_ceiling_gibps):
    """Event-driven fair-share makespan (ms) for n equal jobs on `lanes`
    lanes: k active jobs share min(k*b1, ceiling) aggregate bandwidth."""
    if n_jobs <= 0 or lanes <= 0:
        return 0.0
    work = RECORD_BYTES / GIB          # GiB per record
    remaining = [work] * n_jobs
    t = 0.0
    active = min(lanes, n_jobs)
    idx = 0                            # jobs identical: track count only
    running = [work] * active
    queued = n_jobs - active
    while running:
        k = len(running)
        agg = min(k * b1_gibps, b_ceiling_gibps)
        per = agg / k                  # GiB/s per active job
        dt = min(running) / per * 1000.0   # ms to next completion
        t += dt
        done = min(running)
        running = [r - done for r in running]
        finished = sum(1 for r in running if r <= 0)
        running = [r for r in running if r > 0]
        for _ in range(min(finished, queued)):
            running.append(work)
            queued -= 1
    return t


def wall_serial_table(m):
    """Measured serial wall: W1(m) table, linear extrapolation past 6."""
    return W1[m] if m in W1 else W1_SLOPE * m


def wall_bounded3_table(m):
    return W3[m] if m in W3 else W3_SLOPE * m


MODELS = [
    # name, wall(ms) per batch given host-miss count m
    ("serial_task_anchor_18.9ms", lambda m: m * SVC_TASK),
    ("serial_measured_miss_35.5ms", lambda m: m * SVC_MISS_LANES1),
    ("serial_measured_qd1_33.8ms", lambda m: m * SVC_MISS_QD1),
    ("serial_measured_W1_table", wall_serial_table),
    ("bounded3_naive_57.4_flat", lambda m: 57.4 if m > 0 else 0.0),
    ("bounded3_wave_3x18.9",
     lambda m: ((m + 2) // 3) * 3 * SVC_TASK if m > 0 else 0.0),
    ("bounded3_measured_W3_table", wall_bounded3_table),
    ("bounded3_fairshare_Bagg0.29",
     lambda m: makespan_fairshare(m, 3, B1_GIBPS, B_AGG3_GIBPS)),
    ("bounded3_fairshare_Bceil0.37",
     lambda m: makespan_fairshare(m, 3, B1_GIBPS, B_CEILING_HI)),
    ("bounded4_fairshare_Bceil0.37",
     lambda m: makespan_fairshare(m, 4, B1_GIBPS, B_CEILING_HI)),
    ("bounded3_fairshare_B0.675",
     lambda m: makespan_fairshare(m, 3, 0.675, 0.675)),
    ("bounded3_fairshare_B2.9",
     lambda m: makespan_fairshare(m, 3, 0.675, 2.9)),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jsha = sha256_of(JOURNAL)
    batches = load_batches(JOURNAL)
    # v60 journal content check (expert routing identical; file sha differs)
    v60_same = None
    if JOURNAL_V60.exists():
        b60 = load_batches(JOURNAL_V60)
        v60_same = ([(s, l, u) for s, l, g, u in batches] ==
                    [(s, l, u) for s, l, g, u in b60])

    # ---- sealed-anchor validation --------------------------------------
    # single-level host LRU@682 on the full request stream must reproduce
    # the sealed host_pack counters (legacy prepare-path semantics).
    per_gpu_single = {}
    for g in (0, 1):
        c = LRU(HOST_SLOTS)
        h = m = 0
        for step, layer, gpu, experts in batches:
            if gpu != g:
                continue
            for e in experts:
                if c.touch((layer, e)):
                    h += 1
                else:
                    m += 1
        per_gpu_single[g] = {"sim_hits": h, "sim_misses": m,
                             "sim_evictions": c.evictions}

    rows = replay_tiers(batches)                       # sealed device policy
    rows_lru = replay_tiers(batches, device_policy="lru")  # Luna plain-LRU

    with open(OUT / "per_batch_miss_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- wall models ----------------------------------------------------
    def totals(rs, fn):
        dec = sum(fn(r["host_misses"]) for r in rs if r["phase"] == "decode")
        pre = sum(fn(r["host_misses"]) for r in rs if r["phase"] == "prefill")
        return dec, pre

    model_rows = []
    for name, fn in MODELS:
        for label, rs in (("device=sealed_priority_lru", rows),
                          ("device=plain_lru", rows_lru)):
            dec, pre = totals(rs, fn)
            model_rows.append({
                "model": name, "device_policy": label,
                "decode_fill_wall_s": round(dec / 1000, 3),
                "prefill_fill_wall_s": round(pre / 1000, 3),
                "response_fill_wall_s": round((dec + pre) / 1000, 3),
                "per_decode_token_ms": round(dec / N_DECODE, 1),
                "per_response_token_ms": round((dec + pre) / N_TOKENS, 1),
            })
    with open(OUT / "fill_wall_models.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
        w.writeheader()
        w.writerows(model_rows)

    # ---- legacy-equivalent single-level stream (sealed pool baseline) ---
    # The sealed v60 run's fill_batch_wall_ms is the bounded 4-lane pool on
    # the FULL request stream (every request consults the pack). Comparing
    # a serial fill on the same miss stream isolates the concurrency effect.
    legacy_rows = single_level_host_rows(batches)
    legacy_misses = sum(r["host_misses"] for r in legacy_rows)
    legacy_serial_ms = sum(wall_serial_table(r["host_misses"])
                           for r in legacy_rows)
    legacy_serial_task_ms = sum(r["host_misses"] * SVC_TASK
                                for r in legacy_rows)
    legacy_dec_serial_ms = sum(wall_serial_table(r["host_misses"])
                               for r in legacy_rows if r["phase"] == "decode")
    legacy_bounded4_ms = sum(
        makespan_fairshare(r["host_misses"], 4, B1_GIBPS, B_AGG3_GIBPS)
        for r in legacy_rows)

    # ---- per-miss-count wall table --------------------------------------
    wall_table = []
    for m in range(0, 9):
        wall_table.append({
            "misses_in_batch": m,
            "serial_W1_measured_ms": round(wall_serial_table(m), 1),
            "bounded3_W3_measured_ms": round(wall_bounded3_table(m), 1),
            "serial_task_anchor_ms": round(m * SVC_TASK, 1),
            "bounded3_wave_ms":
                round(((m + 2) // 3) * 3 * SVC_TASK if m else 0.0, 1),
            "fairshare_L3_Bagg0.29_ms":
                round(makespan_fairshare(m, 3, B1_GIBPS, B_AGG3_GIBPS), 1),
            "fairshare_L3_Bceil0.37_ms":
                round(makespan_fairshare(m, 3, B1_GIBPS, B_CEILING_HI), 1),
        })
    with open(OUT / "wall_by_miss_count.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wall_table[0].keys()))
        w.writeheader()
        w.writerows(wall_table)

    # ---- bandwidth / lane sensitivity (decode batches, sealed policy) ---
    dec_rows = [r for r in rows if r["phase"] == "decode"]
    sweep = []
    for b_ceil in (0.29, 0.33, 0.35, 0.37, 0.675, 1.0, 2.9):
        for lanes in (1, 2, 3, 4, 6):
            wall = sum(makespan_fairshare(r["host_misses"], lanes,
                                          B1_GIBPS, b_ceil)
                       for r in dec_rows)
            sweep.append({"device_ceiling_gibps": b_ceil, "lanes": lanes,
                          "decode_fill_wall_s": round(wall / 1000, 3),
                          "per_decode_token_ms": round(wall / N_DECODE, 1)})
    with open(OUT / "bandwidth_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)

    # ---- falsifiability gate --------------------------------------------
    # Removable caller-side CPU bucket (the only thing a worker pool can
    # legally take off the critical path; enqueue submission stays on the
    # caller because prefetch_host_lease owns transfer-stream submission):
    n_batches = len(rows)
    reserve_bucket_s = RESERVE_MS * n_batches / 1000.0
    enqueue_bucket_s = ENQUEUE_MS_ALL * n_batches / 1000.0
    # repair wall deltas vs the serial model at matched anchors:
    serial_meas = sum(wall_serial_table(r["host_misses"]) for r in rows)
    bnd_meas = sum(wall_bounded3_table(r["host_misses"]) for r in rows)
    serial_task = sum(r["host_misses"] * SVC_TASK for r in rows)
    bnd_task = sum(((r["host_misses"] + 2) // 3) * 3 * SVC_TASK
                   for r in rows)
    fs_ceil = sum(makespan_fairshare(r["host_misses"], 3, B1_GIBPS,
                                     B_CEILING_HI) for r in rows)
    gate = {
        "repair_wall_delta_vs_serial_measured_s":
            round((bnd_meas - serial_meas) / 1000, 3),
        "repair_wall_delta_vs_serial_task_anchor_s":
            round((bnd_task - serial_task) / 1000, 3),
        "repair_wall_delta_fairshare_at_ceiling_s":
            round((fs_ceil - serial_meas) / 1000, 3),
        "removable_reserve_bucket_s": round(reserve_bucket_s, 3),
        "enqueue_bucket_stays_on_caller_s": round(enqueue_bucket_s, 3),
        "gate_rule": ("repair wins iff (serial_wall - bounded_wall) >= "
                      "reservation+enqueue caller buckets it removes"),
        "gate_result": None,
    }
    win_meas = serial_meas - bnd_meas        # negative = repair loses wall
    win_best = serial_meas - fs_ceil         # best-case fair-share delta
    gate["gate_result"] = (
        "FAIL: bounded-3-lane fill wall is %.1f%% WORSE than serial on the "
        "measured /tmp bank (%.1fs vs %.1fs per response); even at the best "
        "observed ceiling 0.37 GiB/s the delta is %+.1fs. The only "
        "recoverable caller bucket is reservation <=%.1fs (%.1fms/batch); "
        "enqueue ~%.1fs stays on the caller (prefetch_host_lease owns "
        "transfer submission). The repair does not meet the gate." % (
            100.0 * (bnd_meas - serial_meas) / serial_meas,
            bnd_meas / 1000, serial_meas / 1000,
            (serial_meas - fs_ceil) / 1000,
            reserve_bucket_s, RESERVE_MS, enqueue_bucket_s))
    if win_meas >= (reserve_bucket_s + enqueue_bucket_s) * 1000:
        gate["gate_result"] = "PASS"

    # ---- validation ------------------------------------------------------
    def agg(rs):
        out = {}
        for g in (0, 1):
            sub = [r for r in rs if r["gpu"] == g]
            out[g] = {
                "requests": sum(r["requests"] for r in sub),
                "device_hits": sum(r["device_hits"] for r in sub),
                "host_hits": sum(r["host_hits"] for r in sub),
                "host_misses": sum(r["host_misses"] for r in sub),
                "miss_dist": dict(sorted(Counter(
                    r["host_misses"] for r in sub).items())),
            }
        return out

    val = {
        "journal": str(JOURNAL), "journal_sha256": jsha,
        "v60_journal_content_identical": v60_same,
        "record_bytes": RECORD_BYTES,
        "device_slots": DEVICE_SLOTS, "host_slots": HOST_SLOTS,
        "sealed_counters": {
            "host_pack": {"cuda0": {"hits": 1223, "misses": 1390,
                                    "evictions": 708},
                          "cuda1": {"hits": 1395, "misses": 1091,
                                    "evictions": 409}},
            "device": {"cuda0": {"hits": 328, "cold_loads": 2285,
                                 "evictions": 2004},
                       "cuda1": {"hits": 327, "cold_loads": 2159,
                                 "evictions": 1878}},
        },
        "sim_single_level_host_lru682_full_stream": per_gpu_single,
        "sim_two_level_sealed_device_policy": agg(rows),
        "sim_two_level_plain_lru_device": agg(rows_lru),
        "phase1_anchors": {
            "svc_task_anchor_ms": SVC_TASK,
            "svc_miss_single_flight_lanes1_ms": SVC_MISS_LANES1,
            "svc_miss_single_flight_qd1_ms": SVC_MISS_QD1,
            "svc_miss_lanes3_qd6_ms": SVC_MISS_PROD3,
            "single_stream_gibps": B1_GIBPS,
            "lanes3_aggregate_gibps": B_AGG3_GIBPS,
            "ceiling_hi_gibps": B_CEILING_HI,
            "within_batch_busy_frac_prod": 0.96,
            "reserve_ms_per_batch": RESERVE_MS,
            "enqueue_ms_per_batch_decode": ENQUEUE_MS_DECODE,
        },
        "legacy_equivalent_stream": {
            "host_misses_single_level": legacy_misses,
            "serial_fill_wall_s_W1": round(legacy_serial_ms / 1000, 3),
            "serial_fill_wall_s_task_anchor":
                round(legacy_serial_task_ms / 1000, 3),
            "serial_decode_fill_wall_s_W1":
                round(legacy_dec_serial_ms / 1000, 3),
            "bounded4_fill_wall_s_fairshare_Bagg0.29":
                round(legacy_bounded4_ms / 1000, 3),
            "sealed_v60_fill_batch_wall_s": {
                "cuda0": 55.114, "cuda1": 42.097, "total": 97.211,
                "lanes": 4, "queue_depth": 8},
            "sealed_filllive_decode_FillWait_s": 41.99,
            "note": ("serial model reproduces the sealed live FillWait "
                     "decode bucket within ~2%; the sealed 4-lane pool "
                     "paid MORE fill wall than serial single-flight"),
        },
        "falsifiability_gate": gate,
    }
    with open(OUT / "validation.json", "w") as f:
        json.dump(val, f, indent=1)

    # ---- console summary -------------------------------------------------
    print(f"journal sha256 {jsha}")
    print(f"batches {len(rows)}  requests {sum(r['requests'] for r in rows)}")
    for g in (0, 1):
        s = per_gpu_single[g]
        print(f"  single-level gpu{g}: hits {s['sim_hits']} "
              f"misses {s['sim_misses']} ev {s['sim_evictions']} "
              f"(sealed {val['sealed_counters']['host_pack'][f'cuda{g}']})")
    print("\nper-response fill wall (two-level, sealed device policy):")
    for r in model_rows:
        if r["device_policy"] == "device=sealed_priority_lru":
            print(f"  {r['model']:32s} resp {r['response_fill_wall_s']:7.2f}s "
                  f"decode {r['decode_fill_wall_s']:7.2f}s "
                  f"({r['per_decode_token_ms']:7.1f} ms/tok)")
    print("\ngate:", gate["gate_result"])


if __name__ == "__main__":
    main()
