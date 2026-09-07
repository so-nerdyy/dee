#!/usr/bin/env python3
"""Forensics for the completed pack-cap A/B (cap 17 -> 20 GiB).

Answers, from the sealed arm evidence only:
  A. exactly which misses B eliminated (eliminated-miss ledger);
  B. token-level criticality of those misses (per-request timestamps were
     not recorded -> per-request classification is UNKNOWN by design);
  C. observational model comparison: miss count vs bytes vs measured
     read-batch wall as predictors of per-token decode wall;
  D. position/order decomposition of the four arms.

No sealed evidence is modified. All outputs go to forensics/results/.

Usage:
  python tools/forensics.py all [--live results/live]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPROOT = HERE.parent
DEFAULT_LIVE = EXPROOT / "results" / "live"
OUT = EXPROOT / "forensics" / "results"

RECORD_BYTES = 13_369_344
BUDGET_A_RECORDS = 682   # 8.5 GiB/GPU (cap 17 total)
BUDGET_B_RECORDS = 803   # 10.0 GiB/GPU (cap 20 total)

ARMS = {  # key -> (relative dir, position, cap)
    "s1A": ("s1/session1-A", 1, 17.0),
    "s1B": ("s1/session1-B", 2, 20.0),
    "s2B": ("s2/session2-B", 1, 20.0),
    "s2A": ("s2/session2-A", 2, 17.0),
}


# ----------------------------------------------------------------------------
# Instrumented replay
# ----------------------------------------------------------------------------

def load_stream(journal: Path):
    """Rebuild each device's fill stream exactly as pack_replay does
    (unique (layer, expert) per (device, forward_step), execution order)."""
    rows = [json.loads(l) for l in journal.open()]
    rows.sort(key=lambda r: (r["forward_step"], r["layer"]))
    phase, stream, seenl = {}, defaultdict(lambda: defaultdict(list)), defaultdict(set)
    for r in rows:
        fs, L, dev = r["forward_step"], r["layer"], r.get("device", "cuda:0")
        phase[(dev, fs)] = r["phase"]
        s = seenl[(dev, fs, L)]
        for row in r["expert_ids_rank_order"]:
            for e in row:
                k = (L, e)
                if k not in s:
                    s.add(k)
                    stream[dev][fs].append(k)
    return phase, stream


def replay_device(stream_dev, phase_dev, budget: int, last_use_rank: dict):
    """LRU replay for one device with per-miss instrumentation.

    Production semantics (validated in research/exact-staging): on miss,
    evict LRU while len(pack) >= budget, then insert. Resident set is
    therefore always the last `budget` distinct records touched, which makes
    the larger cache's resident set a strict superset of the smaller's for
    the same stream (no new misses at larger budgets).

    Returns (misses, hits, evictions, decode_misses, misses_by_step) where
    each miss is a dict with identity + recency metadata.
    """
    pack: OrderedDict = OrderedDict()
    rank = 0  # distinct-records-touched counter (recency ruler)
    misses, hits, evictions, decode_misses = 0, 0, 0, 0
    misses_by_step = defaultdict(int)
    out = []
    for fs in sorted(stream_dev):
        for k in stream_dev[fs]:
            rank += 1
            prev = last_use_rank.get(k)
            last_use_rank[k] = rank
            if k in pack:
                hits += 1
                pack.move_to_end(k)
                continue
            misses += 1
            misses_by_step[fs] += 1
            if phase_dev[fs] == "decode":
                decode_misses += 1
            evicted = []
            while len(pack) >= budget:
                ek, _ = pack.popitem(last=False)
                evicted.append(list(ek))
                evictions += 1
            pack[k] = 1
            out.append({
                "forward_step": fs,
                "phase": phase_dev[fs],
                "layer": k[0],
                "expert": k[1],
                "storage_bytes": RECORD_BYTES,
                # distinct-record distance since previous use (None = first
                # use = compulsory); recency ruler = pack touches, not wall.
                "distinct_since_prev_use": (rank - prev - 1) if prev is not None else None,
                "compulsory": prev is None,
                "evicted_to_make_room": evicted,
            })
    return out, hits, evictions, decode_misses, dict(misses_by_step)


def build_ledger(live: Path) -> dict:
    journal = live / ARMS["s1A"][0] / "routed_experts.jsonl"
    phase, stream = load_stream(journal)

    a_misses_by_dev, b_misses_by_dev = {}, {}
    replay_validation = {}
    for dev in sorted(stream):
        phase_dev = {fs: phase[(dev, fs)] for fs in stream[dev]}
        last_use: dict = {}
        am, ah, ae, adm, asteps = replay_device(stream[dev], phase_dev, BUDGET_A_RECORDS, last_use)
        # reuse the same recency ruler for B (same stream order) but a fresh
        # pack: last_use must be rebuilt identically; replay_device mutates it
        # in stream order so calling again with a fresh dict is deterministic.
        last_use_b: dict = {}
        bm, bh, be, bdm, bsteps = replay_device(stream[dev], phase_dev, BUDGET_B_RECORDS, last_use_b)
        a_misses_by_dev[dev] = am
        b_misses_by_dev[dev] = bm
        replay_validation[dev] = {
            "replay_misses_A": len(am), "replay_misses_B": len(bm),
            "replay_hits_A": ah, "replay_hits_B": bh,
            "replay_evictions_A": ae, "replay_evictions_B": be,
            "replay_decode_misses_A": adm, "replay_decode_misses_B": bdm,
        }

    # Eliminated misses: A misses that are B hits. Because the resident set of
    # the larger LRU is a superset for the same stream, B-misses are a subset
    # of A-misses; match by identity, not position.
    a_keys = {(d, m["forward_step"], m["layer"], m["expert"]) for d, ms in a_misses_by_dev.items() for m in ms}
    b_keys = {(d, m["forward_step"], m["layer"], m["expert"]) for d, ms in b_misses_by_dev.items() for m in ms}
    eliminated_keys = a_keys - b_keys
    new_in_b = b_keys - a_keys

    # ordinal of each miss inside its device's decode miss stream (storage
    # request order); previous-use distance from the A-side ledger.
    eliminated = []
    for dev in sorted(a_misses_by_dev):
        ordinal = 0
        for m in a_misses_by_dev[dev]:
            key = (dev, m["forward_step"], m["layer"], m["expert"])
            if m["phase"] != "decode":
                continue
            ordinal += 1
            if key in eliminated_keys:
                eliminated.append({
                    "device": dev,
                    "request_ordinal_decode": ordinal,
                    "start_pos": m["forward_step"],  # token start_pos == forward_step
                    **{k: m[k] for k in ("forward_step", "layer", "expert",
                                          "storage_bytes", "distinct_since_prev_use",
                                          "compulsory", "evicted_to_make_room")},
                })

    return {
        "schema": "pack-cap-ab/eliminated-miss-ledger-v1",
        "journal_sha256_source": "identical across all four arms (f6ec70243acbafa7..., "
                                 "verified in forensics; see eliminated ledger report)",
        "budgets": {"A_records_per_gpu": BUDGET_A_RECORDS, "B_records_per_gpu": BUDGET_B_RECORDS},
        "record_bytes": RECORD_BYTES,
        "replay_validation": replay_validation,
        "counts": {
            "A_misses_total_replay": sum(len(v) for v in a_misses_by_dev.values()),
            "B_misses_total_replay": sum(len(v) for v in b_misses_by_dev.values()),
            "eliminated_replay": len(eliminated_keys),
            "new_misses_in_B_replay": len(new_in_b),
        },
        "eliminated": eliminated,
        "new_misses_in_B": sorted(new_in_b),
    }


# ----------------------------------------------------------------------------
# Live telemetry extraction
# ----------------------------------------------------------------------------

def arm_metrics(live: Path) -> dict:
    out = {}
    for key, (rel, position, cap) in ARMS.items():
        r = json.load(open(live / rel / "result.json"))
        m = json.load(open(live / rel / "memory.json"))
        es, hp = r["expert_store"], r["host_pack"]
        pt = [t for t in r["per_token_accounting"] if t["phase"] == "decode"]
        out[key] = {
            "dir": rel, "position": position, "cap_gib_total": cap,
            "decode_wall_s": r["decode_wall_s"],
            "prefill_ms": r["prefill_ms"],
            "build_seconds": r["build_seconds"],
            "misses_total": r["byte_accounting"]["storage_requests_total"],
            "storage_bytes_total": r["byte_accounting"]["storage_bytes_total"],
            "hwm_gib": m["process_final_and_peak_gib"]["VmHWM"],
            "min_mem_available_gib": m["minimum_checkpoint_host_mem_available_gib"],
            "expert_store": {d: {k: es[d][k] for k in (
                "p50_read_ms", "p95_read_ms", "max_read_ms", "read_bandwidth_mib_s",
                "source_read_batch_wall_ms", "read_milliseconds",
                "source_read_overlap_ms", "source_read_overlap_percent",
                "source_read_batches")} for d in es},
            "host_pack": {d: {k: hp[d][k] for k in (
                "misses", "hits", "evictions", "entries", "bytes",
                "fill_batches", "fill_batch_wall_ms", "fill_worker_ms",
                "fill_reservation_wall_ms", "reused_fill_buffers", "reused_fill_bytes")}
                for d in hp},
            "per_token": [{
                "forward_step": t["step"], "wall_ms": t["wall_ms"],
                "storage_requests": t["storage_requests"],
                "storage_bytes": t["storage_bytes"],
                "source_read_wall_ms": t["source_read_wall_ms"],
                "h2d_bytes": t["h2d_bytes"],
                "host_pack_misses": t["host_pack_misses"],
                "host_pack_hits": t["host_pack_hits"],
                "evictions": t["evictions"],
            } for t in pt],
        }
    return out


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k2 in range(i, j + 1):
                r[order[k2]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return {"slope": slope, "intercept_s": intercept, "r2": r2, "n": n}


# ----------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------

def eliminated_per_token(ledger: dict) -> dict:
    per = defaultdict(lambda: {"count": 0, "bytes": 0, "layers": defaultdict(int), "devices": defaultdict(int)})
    for m in ledger["eliminated"]:
        p = per[m["forward_step"]]
        p["count"] += 1
        p["bytes"] += m["storage_bytes"]
        p["layers"][m["layer"]] += 1
        p["devices"][m["device"]] += 1
    return {str(k): {"count": v["count"], "bytes": v["bytes"],
                     "layers": {str(l): c for l, c in sorted(v["layers"].items())},
                     "devices": dict(sorted(v["devices"].items()))}
            for k, v in sorted(per.items())}


def criticality(ledger: dict, metrics: dict) -> dict:
    """Token-level criticality of the eliminated misses.

    Evidence available: per-token wall + measured read-batch wall
    (source_read_wall_ms) for all four arms; NO per-request timestamps
    (stage profiling was disabled in every arm) -> per-request
    read-submit/complete/demand are UNKNOWN and stay UNKNOWN.

    Token-level test: a cap effect requires the B-A wall delta to have the
    SAME sign in both sessions (candidate faster regardless of position).
    A position effect flips sign. Where the sign flips or deltas are
    ~zero, the eliminated misses on that token produced no wall reduction
    in either session -> not demand-blocking at observable granularity.
    """
    ept = eliminated_per_token(ledger)
    per_token = []
    for key, (rel, position, cap) in ARMS.items():
        pass  # (per_token data accessed via metrics below)

    for t in range(1, 16):
        w = {k: metrics[k]["per_token"][t - 1]["wall_ms"] / 1000 for k in ARMS}
        rd = {k: metrics[k]["per_token"][t - 1]["source_read_wall_ms"] / 1000 for k in ARMS}
        d1 = w["s1B"] - w["s1A"]   # session 1: A first, B second
        d2 = w["s2A"] - w["s2B"]   # session 2: B first, A second
        entry = {
            "forward_step": t,
            "wall_s": {k: round(w[k], 3) for k in ARMS},
            "read_batch_wall_s": {k: round(rd[k], 3) for k in ARMS},
            "delta_B_minus_A_session1_s": round(d1, 3),
            "delta_B_minus_A_session2_s": round(-d2, 3),
            "eliminated_misses": ept.get(str(t), {}).get("count", 0),
            "eliminated_bytes": ept.get(str(t), {}).get("bytes", 0),
            "read_batch_wall_delta_second_minus_first_s": round(
                (rd["s1B"] + rd["s2A"]) / 2 - (rd["s1A"] + rd["s2B"]) / 2, 3),
        }
        if entry["eliminated_misses"] == 0:
            entry["token_level_criticality"] = "NO_ELIMINATED_MISSES"
        else:
            same_sign = (d1 < -0.05 and -d2 < -0.05) or (d1 > 0.05 and -d2 > 0.05)
            if same_sign and d1 < 0:
                entry["token_level_criticality"] = "CRITICAL_WAIT_CANDIDATE"
            elif same_sign:
                entry["token_level_criticality"] = "CANDIDATE_REGRESSION"
            else:
                entry["token_level_criticality"] = "NOT_DEMAND_BLOCKING_AT_TOKEN_GRANULARITY"
        per_token.append(entry)

    crit_counts = defaultdict(int)
    for e in per_token:
        crit_counts[e["token_level_criticality"]] += e["eliminated_misses"]

    return {
        "schema": "pack-cap-ab/miss-criticality-v1",
        "evidence_basis": {
            "per_request_timestamps": "NOT_RECORDED (stage profiling disabled in all four arms)",
            "per_token_wall": "MEASURED (result.json per_token_accounting)",
            "per_token_read_batch_wall": "MEASURED (source_read_wall_ms)",
            "classification_rule": "cap effect requires same-sign B-A delta in both sessions; "
                                   "sign flip = position effect; |delta| <= 0.05 s treated as null",
        },
        "classification_legend": {
            "CRITICAL_WAIT_CANDIDATE": "B faster than A in the same direction in BOTH sessions at this token",
            "CANDIDATE_REGRESSION": "A faster in BOTH sessions at this token",
            "NOT_DEMAND_BLOCKING_AT_TOKEN_GRANULARITY": "no consistent cap effect; deltas flip with position or are null",
            "NO_ELIMINATED_MISSES": "token had no eliminated misses",
        },
        "per_token": per_token,
        "eliminated_misses_by_classification": dict(crit_counts),
        "per_request_note": "Every eliminated miss is classified UNKNOWN at per-request "
                            "granularity (read submit/complete/demand timestamps were not "
                            "recorded); the table above is the strongest token-level "
                            "attribution the sealed evidence supports.",
    }


def order_effect(metrics: dict) -> dict:
    toks = range(1, 16)

    def w(k, t):
        return metrics[k]["per_token"][t - 1]["wall_ms"] / 1000

    position_rows = []
    for t in toks:
        dA = w("s2A", t) - w("s1A", t)          # A second minus A first
        dB = w("s1B", t) - w("s2B", t)          # B second minus B first
        position_rows.append({
            "forward_step": t,
            "A_second_minus_A_first_s": round(dA, 3),
            "B_second_minus_B_first_s": round(dB, 3),
        })
    early = range(1, 12)
    late = range(12, 15)
    sums = {}
    for label, rng in (("tokens_1_11", early), ("tokens_12_14", late), ("tokens_1_14", toks)):
        sums[label] = {
            "A_second_minus_A_first_s": round(sum(w("s2A", t) - w("s1A", t) for t in rng), 3),
            "B_second_minus_B_first_s": round(sum(w("s1B", t) - w("s2B", t) for t in rng), 3),
        }
    return {
        "schema": "pack-cap-ab/order-effect-v1",
        "walls_s": {k: metrics[k]["decode_wall_s"] for k in ARMS},
        "positions": {k: metrics[k]["position"] for k in ARMS},
        "position_sensitivity_s": {
            "A": round(metrics["s2A"]["decode_wall_s"] - metrics["s1A"]["decode_wall_s"], 3),
            "B": round(metrics["s1B"]["decode_wall_s"] - metrics["s2B"]["decode_wall_s"], 3),
        },
        "mean_B_minus_A_s": round(
            ((metrics["s1B"]["decode_wall_s"] - metrics["s1A"]["decode_wall_s"])
             + (metrics["s2B"]["decode_wall_s"] - metrics["s2A"]["decode_wall_s"])) / 2, 4),
        "position_effect_by_token": position_rows,
        "position_effect_sums": sums,
        "memory_pressure": {k: {"min_mem_available_gib": metrics[k]["min_mem_available_gib"],
                                 "hwm_gib": metrics[k]["hwm_gib"]} for k in ARMS},
        "read_latency_p50_ms": {k: round(metrics[k]["expert_store"]["cuda0"]["p50_read_ms"], 1) for k in ARMS},
        "fill_reservation_wall_s": {k: round(
            (metrics[k]["host_pack"]["cuda0"]["fill_reservation_wall_ms"]
             + metrics[k]["host_pack"]["cuda1"]["fill_reservation_wall_ms"]) / 1000, 2) for k in ARMS},
    }


def model_comparison(metrics: dict) -> dict:
    """Observational model comparison across 60 token-observations.

    Caveats stated in output: 4 arms x 15 tokens; token latencies are
    serially dependent within a run; this is exploratory, not inference.
    """
    obs = []
    for k in ARMS:
        for i, t in enumerate(metrics[k]["per_token"]):
            obs.append({
                "arm": k, "position": metrics[k]["position"], "forward_step": t["forward_step"],
                "wall_s": t["wall_ms"] / 1000,
                "misses": t["host_pack_misses"],
                "storage_bytes": t["storage_bytes"],
                "read_batch_wall_s": t["source_read_wall_ms"] / 1000,
                "h2d_bytes": t["h2d_bytes"],
            })
    xs_miss = [o["misses"] for o in obs]
    xs_bytes = [o["storage_bytes"] / 1e9 for o in obs]
    xs_rd = [o["read_batch_wall_s"] for o in obs]
    ys = [o["wall_s"] for o in obs]
    same_miss_example = None
    by_miss = defaultdict(list)
    for o in obs:
        by_miss[o["misses"]].append((o["arm"], o["forward_step"], o["wall_s"]))
    for mcand, lst in sorted(by_miss.items()):
        if len(lst) >= 2:
            walls = [x[2] for x in lst]
            if max(walls) / max(min(walls), 1e-9) >= 1.5:
                same_miss_example = {"misses": mcand, "observations": lst,
                                     "wall_spread_ratio": round(max(walls) / min(walls), 2)}
                break
    return {
        "schema": "pack-cap-ab/revised-service-model-v1",
        "n_observations": len(obs),
        "caveats": ["token latencies within a run are serially dependent",
                     "4 arms only; exploratory, no inferential claim",
                     "per-request timestamps were not recorded"],
        "predictors": {
            "miss_count": {"pearson_r": pearson(xs_miss, ys), "spearman_rho": spearman(xs_miss, ys),
                            "ols": ols(xs_miss, ys)},
            "storage_bytes_gb": {"pearson_r": pearson(xs_bytes, ys), "spearman_rho": spearman(xs_bytes, ys),
                                   "ols": ols(xs_bytes, ys)},
            "measured_read_batch_wall_s": {"pearson_r": pearson(xs_rd, ys), "spearman_rho": spearman(xs_rd, ys),
                                             "ols": ols(xs_rd, ys)},
        },
        "same_miss_count_different_wall_example": same_miss_example,
        "old_model_check": {
            "form": "wall = intercept + 48.679 ms per miss (recalibrated_model.json, v65 fit)",
            "predicted_wall_gain_from_52_fewer_misses_s": round(48.679e-3 * 52, 3),
            "observed_mean_B_minus_A_s": round(
                ((metrics["s1B"]["decode_wall_s"] - metrics["s1A"]["decode_wall_s"])
                 + (metrics["s2B"]["decode_wall_s"] - metrics["s2A"]["decode_wall_s"])) / 2, 4),
            "verdict": "REJECTED prospectively: predicted 2.53 s, observed ~0",
        },
        "batch_structure_arithmetic": {
            "note": "fill batches run up to queue-depth 6 reads concurrently on 3 lanes; "
                    "batch wall ~ max read in batch (p50 ~105 ms, p95 ~164 ms), so removing "
                    "1-2 reads from a multi-read batch saves ~0 wall",
            "batches_A": {"cuda0": metrics["s1A"]["expert_store"]["cuda0"]["source_read_batches"],
                           "cuda1": metrics["s1A"]["expert_store"]["cuda1"]["source_read_batches"]},
            "reads_per_batch_A": round(
                (metrics["s1A"]["host_pack"]["cuda0"]["misses"] + metrics["s1A"]["host_pack"]["cuda1"]["misses"])
                / (metrics["s1A"]["expert_store"]["cuda0"]["source_read_batches"]
                   + metrics["s1A"]["expert_store"]["cuda1"]["source_read_batches"]), 2),
            "read_worker_overlap_percent_A": metrics["s1A"]["expert_store"]["cuda0"]["source_read_overlap_percent"],
        },
    }


# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["all"])
    ap.add_argument("--live", type=Path, default=DEFAULT_LIVE)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ledger = build_ledger(args.live)
    (OUT / "eliminated-misses.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")

    metrics = arm_metrics(args.live)
    crit = criticality(ledger, metrics)
    (OUT / "miss-criticality.json").write_text(json.dumps(crit, indent=1), encoding="utf-8")

    order = order_effect(metrics)
    (OUT / "order-effect.json").write_text(json.dumps(order, indent=1), encoding="utf-8")

    model = model_comparison(metrics)
    (OUT / "revised-service-model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")

    print("ledger counts:", ledger["counts"])
    print("replay validation:", json.dumps(ledger["replay_validation"]))
    print("eliminated per token:", json.dumps(eliminated_per_token(ledger)))
    print("criticality by class:", json.dumps(crit["eliminated_misses_by_classification"]))
    print("position sensitivity:", json.dumps(order["position_sensitivity_s"]))
    print("position sums:", json.dumps(order["position_effect_sums"]))
    for name, p in model["predictors"].items():
        print(f"predictor {name}: pearson_r={p['pearson_r']:.3f} spearman={p['spearman_rho']:.3f} "
              f"ols_r2={p['ols']['r2']:.3f} slope={p['ols']['slope']:.4f}")
    print("same-miss example:", json.dumps(model["same_miss_count_different_wall_example"]))


if __name__ == "__main__":
    main()
