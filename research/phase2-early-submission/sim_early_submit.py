"""Cheap offline early-submission model for Phase-2D decision (stdlib only).

Reads the sealed fill-live route journal (extracted via `git show`, read-only)
and combines it with sealed scalar counters to quantify mechanisms A-E.

Every number printed is labeled MEASURED (sealed live T4) / DERIVED (exact
arithmetic on sealed inputs) / SIMULATED (this script's model) / THEORETICAL
(closed-form bound). Nothing here runs a GPU or touches production code.

Usage:
    python3 sim_early_submit.py --journal routed_experts.jsonl --out results.json
"""
import argparse
import json
import math
import sys
from collections import Counter

RECORD_BYTES = 13369344          # sealed: dee4 record (13,369,344 B = 12.75 MiB)
GATE_UP_BYTES = 4194304 * 2 + 262144 * 2    # w1+w3 weights + w1+w3 scales
DOWN_BYTES = 4194304 + 262144               # w2 weight + w2 scale
assert GATE_UP_BYTES + DOWN_BYTES == RECORD_BYTES

# Sealed scalars (Phase-1 fill-live-t4x2-20260909 + route-pipeline live profile)
SEALED = {
    "decode_rows": 645,               # MEASURED: tokens 7..21 x 43 layers
    "decode_wall_s": 66.233,          # MEASURED: profiled arm ON
    "fill_wait_s": 41.990,            # MEASURED: critical host-blocked storage
    "stage_enqueue_wait_s": 9.436,    # MEASURED: H2D submit + cache ops, host serial
    "native_output_sync_s": 4.894,    # MEASURED: required drain, uniform
    "route_d2h_total_s": 0.0154,      # MEASURED: 15.4 ms / 645 rows
    "readiness_wait_s": 0.027,        # MEASURED: transfers always ready
    "combine_s": 0.139,               # MEASURED
    "expert_compute_dispatch_s": 0.256,  # MEASURED host span
    "expert_compute_device_s": 1.5,   # MEASURED whole-run device GEMMs (approx)
    "shared_device_decode_s": 0.308,  # MEASURED torch shared path, decode
    "unknown_s": 9.196,               # MEASURED gap: dense attn + orchestration
    "bank_gibps_prod": 0.29,          # MEASURED production qd6/l3
    "bank_gibps_qd1": 0.37,           # MEASURED single-flight ceiling
    "h2d_gbps": 5.54,                 # FITTED (exact-critical-path, x-checked vs 5.38 idle4)
    "req_service_ms_prod": 57.39,     # MEASURED per-request mean, 3-wide
    "req_service_ms_1lane": 18.91,    # MEASURED single-flight mean
    "reserve_ms_mean": 7.08,          # MEASURED per-batch reservation mean
    "reserve_ms_p95": 52.87,          # MEASURED per-batch reservation p95
    "within_batch_busy": 0.959,       # MEASURED production
    "qdepth1_busy": 0.533,            # MEASURED: disk idle ~half when serialized
    "hit_rate_prod": 0.4674,          # MEASURED fill-live cache hit rate
    "memset_s": 0.8,                  # DERIVED est (16.7 GB @ ~20 GB/s)
    "h2d_gather_memcpy_s": 1.5,       # DERIVED est midpoint of 1-2 s range
    "persist_recall_at12": 0.503,     # MEASURED cache-predictor persist_layer
    "persist_precision_at12": 0.251,  # MEASURED -> useful-byte rate 25.1%
    "persist_full_bundle_at12": 0.047,# MEASURED
}


def load_journal(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_journal(args.journal)
    decode = [r for r in rows if r.get("phase") == "decode"]
    prefill = [r for r in rows if r.get("phase") != "decode"]

    # --- Mechanism A inputs: per-layer batch structure (dedup per layer call) ---
    batch_sizes = []
    for r in decode:
        for tok_row in r["expert_ids_rank_order"]:
            batch_sizes.append(len(set(tok_row)))
    bs = Counter(batch_sizes)
    max_b = max(batch_sizes)
    mean_b = sum(batch_sizes) / len(batch_sizes)

    # hash-layer early-known ID volume (layers 0..2, decode)
    hash_ids = sum(1 for r in decode if r.get("layer", 99) < 3
                   for tok_row in r["expert_ids_rank_order"]
                   for _ in set(tok_row))
    hash_bytes = hash_ids * RECORD_BYTES

    # same-step adjacent-layer reuse (Fate hypothesis check on THIS journal)
    adj_reuse = 0
    adj_total = 0
    by_step_layer = {(r["forward_step"], r["layer"]): r for r in decode}
    for r in decode:
        key = (r["forward_step"], r["layer"])
        prev = by_step_layer.get((key[0], key[1] - 1))
        if prev is None:
            continue
        cur_sets = [set(t) for t in r["expert_ids_rank_order"]]
        # decode rows have token_rows==1
        prev_sets = [set(t) for t in prev["expert_ids_rank_order"]]
        for cs in cur_sets:
            for ps in prev_sets:
                adj_total += 1
                adj_reuse += len(cs & ps)

    n_decode_calls = len(decode)  # device-split? journal has per-device rows
    devices = sorted({r.get("device") for r in rows})

    S = SEALED
    per_row_wall_ms = S["decode_wall_s"] / S["decode_rows"] * 1000
    per_row_fill_ms = S["fill_wait_s"] / S["decode_rows"] * 1000
    per_row_enqueue_ms = S["stage_enqueue_wait_s"] / S["decode_rows"] * 1000
    per_row_sync_ms = S["native_output_sync_s"] / S["decode_rows"] * 1000

    # --- A. avoidable submit delay (DERIVED) ---
    # reservation serial per batch; batches ~= decode rows per GPU (one native
    # call per layer). Use rows/2 per GPU? Conservative: count rows as batches.
    a_reserve_total_s = S["reserve_ms_mean"] / 1000 * S["decode_rows"]
    a_h2d_headofline_ms = per_row_fill_ms  # hits wait ~full batch fill wall
    a_h2d_self_ms = RECORD_BYTES / (S["h2d_gbps"] * 1e9) * 1000

    # --- B. fixed-slot prize (DERIVED sum of host-serial buckets) ---
    b_host_serial_s = S["stage_enqueue_wait_s"] + S["memset_s"] + S["h2d_gather_memcpy_s"]
    # reservation partly overlaps worker span; count half as critical (conservative)
    b_prize_upper_s = b_host_serial_s + a_reserve_total_s / 2

    # --- C. component split prize (THEORETICAL bound) ---
    down_h2d_ms = DOWN_BYTES / (S["h2d_gbps"] * 1e9) * 1000
    gateup_h2d_ms = GATE_UP_BYTES / (S["h2d_gbps"] * 1e9) * 1000
    compute_per_row_ms = S["expert_compute_device_s"] / S["decode_rows"] * 1000
    # prize <= min(down H2D, gate/up H2D+compute) per row (overlap only)
    c_prize_per_row_ms = min(down_h2d_ms, gateup_h2d_ms + compute_per_row_ms)
    c_prize_total_s = c_prize_per_row_ms / 1000 * S["decode_rows"]
    # SSD-side variant bound: down SSD time overlapped with gate/up H2D+compute
    down_ssd_ms = DOWN_BYTES / (S["bank_gibps_prod"] * 1024**3) * 1000
    c_ssd_prize_per_row_ms = min(down_ssd_ms, gateup_h2d_ms + compute_per_row_ms)
    c_ssd_prize_total_s = c_ssd_prize_per_row_ms / 1000 * S["decode_rows"]

    # --- D. shared overlap prize (bound = shared wall itself) ---
    d_prize_upper_s = S["shared_device_decode_s"]  # host part UNKNOWN

    # --- E. prefetch break-even (THEORETICAL, disk-busy parity) ---
    # disk is 96% busy within batch: every wasted byte displaces useful bytes.
    # parity: useful_frac >= 0.50; with displacement margin: >= 0.60.
    e_required_parity = 0.50
    e_required_margin = 0.60
    e_measured_best = S["persist_precision_at12"]
    e_gap_pp = (e_required_margin - e_measured_best) * 100

    out = {
        "provenance": "SIMULATED/DERIVED on sealed inputs; see script header",
        "journal": {
            "rows_total": len(rows),
            "decode_rows": len(decode),
            "prefill_rows": len(prefill),
            "devices": devices,
            "decode_batch_size_hist": {str(k): v for k, v in sorted(bs.items())},
            "decode_batch_max": max_b,
            "decode_batch_mean": round(mean_b, 3),
            "hash_layer_early_ids_decode": hash_ids,
            "hash_layer_early_bytes": hash_bytes,
            "hash_layer_early_mib_per_token": round(
                hash_bytes / 15 / 1024**2, 1),  # 15 decode steps in journal
            "adjacent_layer_shared_experts": adj_reuse,
            "adjacent_layer_comparisons": adj_total,
        },
        "per_row_ms": {
            "wall": round(per_row_wall_ms, 2),
            "fill_critical": round(per_row_fill_ms, 2),
            "stage_enqueue": round(per_row_enqueue_ms, 2),
            "output_sync": round(per_row_sync_ms, 2),
        },
        "A_immediate_submit": {
            "reserve_serial_total_s_derived": round(a_reserve_total_s, 2),
            "h2d_headofline_for_hits_ms_per_row": round(a_h2d_headofline_ms, 1),
            "h2d_self_ms_per_record": round(a_h2d_self_ms, 2),
            "verdict": "within-layer submit already immediate (one batch at "
                       "route_known); residual is host serial (reservation + "
                       "enqueue), folded into B",
        },
        "B_fixed_slot": {
            "host_serial_buckets_s": round(b_host_serial_s, 2),
            "prize_upper_s_derived": round(b_prize_upper_s, 2),
            "prize_upper_pct_of_decode": round(
                b_prize_upper_s / S["decode_wall_s"] * 100, 1),
            "external_anchor": "exact-critical-path buffer_reuse SIMULATED "
                               "-12.3% on-demand (~-8.8 s), ~0 post-staging-lead",
        },
        "C_component_split": {
            "gate_up_first_deadline_mib": round(GATE_UP_BYTES / 1024**2, 2),
            "down_second_deadline_mib": round(DOWN_BYTES / 1024**2, 2),
            "down_h2d_ms_per_row": round(down_h2d_ms, 3),
            "prize_h2d_overlap_s_theoretical": round(c_prize_total_s, 2),
            "prize_ssd_overlap_s_theoretical": round(c_ssd_prize_total_s, 2),
            "prize_pct_of_decode": round(
                c_ssd_prize_total_s / S["decode_wall_s"] * 100, 2),
            "verdict": "requires splitting one contiguous pread; readiness~0 "
                       "proves H2D never exposed; compute 1.5 s of 66 s",
        },
        "D_shared_overlap": {
            "prize_upper_s_bound": round(d_prize_upper_s, 3),
            "prize_pct_of_decode": round(
                d_prize_upper_s / S["decode_wall_s"] * 100, 2),
            "caveats": "shared HOST wall UNKNOWN (instrumentation gap); "
                       "GEMM-under-H2D contention c=0.021 measured severe",
        },
        "E_prefetch_only": {
            "measured_useful_byte_rate": e_measured_best,
            "required_parity": e_required_parity,
            "required_with_margin": e_required_margin,
            "gap_pp_to_margin": round(e_gap_pp, 1),
            "measured_full_bundle_recall_at12": S["persist_full_bundle_at12"],
            "verdict": "in-situ prefetch MEASURED neutral-to-harmful at all "
                       "caps; need ~2.4x accuracy gain + cross-token scheduler",
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
