#!/usr/bin/env python3
"""Ingest fill_replay matrix outputs into a machine JSON verdict input.

Reads dee.cpp/experiments/route_pipeline/fill-live-latest/{session-summary.json,
fill/replay_*.timeline.json, fill/rider_*.json} and emits per-case:
request service distribution, batch wall/worker-sum accounting (worker sums
are NEVER added as wall), within-batch SSD-busy union fraction, reserve/wake
waits, cache-hit rate, and cold-vs-warm half split. No CUDA needed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

EXP = Path(__file__).resolve().parent
LIVE = EXP / "fill-live-t4x2-20260909"


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, int(q * len(s)))
    return s[i]


def union_len(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    cur_s = cur_e = -1.0
    for s, e in sorted(intervals):
        if s > cur_e:
            total += max(0.0, cur_e - cur_s) if cur_e >= 0 else 0.0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e >= 0:
        total += max(0.0, cur_e - cur_s)
    return total


def ingest_replay(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    batches = d["batches"]
    svc_all: list[float] = []
    wall_all: list[float] = []
    wsum_all: list[float] = []
    busy_frac: list[float] = []
    reserve: list[float] = []
    wake: list[float] = []
    n_req = n_hit = n_miss = n_ev = n_bytes = 0
    half = len(batches) // 2
    svc_first: list[float] = []
    svc_second: list[float] = []
    for i, b in enumerate(batches):
        wall = float(b.get("batch_wall_ms", 0.0))
        wall_all.append(wall)
        wsum_all.append(float(b.get("worker_sum_ms", 0.0)))
        reserve.append(float(b.get("reserve_ms", 0.0)))
        wake.append(float(b.get("wake_ms", 0.0)))
        n_miss += int(b.get("misses", 0))
        n_ev += int(b.get("evictions", 0))
        n_bytes += int(b.get("bytes", 0))
        ivs = []
        for r in b.get("requests", []):
            s = float(r.get("service_ms", 0.0))
            svc_all.append(s)
            (svc_first if i < half else svc_second).append(s)
            n_req += 1
            n_hit += 1 if r.get("cache_hit") else 0
            o = float(r.get("start_offset_ms", 0.0))
            ivs.append((o, o + s))
        busy_frac.append(union_len(ivs) / wall if wall > 0 else 0.0)
    return {
        "batches": len(batches),
        "requests": n_req,
        "bytes": n_bytes,
        "misses": n_miss,
        "evictions": n_ev,
        "cache_hit_rate": n_hit / n_req if n_req else 0.0,
        "req_service_ms": {
            "mean": statistics.fmean(svc_all) if svc_all else 0.0,
            "p50": pct(svc_all, 0.50),
            "p95": pct(svc_all, 0.95),
            "max": max(svc_all) if svc_all else 0.0,
        },
        "batch_wall_ms": {
            "sum": sum(wall_all),
            "mean": statistics.fmean(wall_all) if wall_all else 0.0,
        },
        "worker_sum_ms": {"sum": sum(wsum_all)},
        "within_batch_busy_frac": {
            "mean": statistics.fmean(busy_frac) if busy_frac else 0.0,
            "p50": pct(busy_frac, 0.50),
        },
        "reserve_ms": {"mean": statistics.fmean(reserve) if reserve else 0.0,
                       "p95": pct(reserve, 0.95)},
        "wake_ms": {"mean": statistics.fmean(wake) if wake else 0.0,
                    "p95": pct(wake, 0.95)},
        "cold_half_req_mean_ms": (statistics.fmean(svc_first)
                                  if svc_first else 0.0),
        "warm_half_req_mean_ms": (statistics.fmean(svc_second)
                                  if svc_second else 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default=str(LIVE))
    ap.add_argument("--out", default=str(
        EXP.parents[2] / "research" / "route-pipeline" / "results"
        / "fill_matrix_ingest.json"))
    args = ap.parse_args()
    live = Path(args.live)
    summary = json.loads((live / "session-summary.json").read_text())
    out: dict = {
        "schema": "dee.route_pipeline.fill_matrix_ingest/v1",
        "source": "fill-live-t4x2-20260909/session-summary.json",
        "gate": summary.get("dry_run_gate", {}).get("status"),
        "gpus": summary.get("dry_run_gate", {}).get("nvidia_smi"),
        "fill_arm": {k: summary.get("arms", {}).get("FILL", {}).get(k)
                     for k in ("exit_code", "classification", "decode_wall_s",
                               "cap_check_ok", "run_id")},
        "variant_rules": summary.get("variant_proof", {}).get("rules_applied"),
        "fill_patch_sha256": summary.get("fill_patch_sha256"),
        "journal": summary.get("journal"),
        "replay_stdout": {k: v.get("output")
                          for k, v in summary.get("fill_matrix", {})
                          .get("cases", {}).items()},
        "cases": {},
    }
    for tag in ("production", "lanes1", "qdepth1"):
        p = live / "fill" / f"replay_{tag}.timeline.json"
        if p.is_file():
            out["cases"][f"replay_{tag}"] = ingest_replay(p)
    for tag in ("seq", "rand"):
        p = live / "fill" / f"rider_{tag}.json"
        if p.is_file():
            out["cases"][f"rider_{tag}"] = json.loads(p.read_text())
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for k, v in out["cases"].items():
        if k.startswith("replay"):
            print(f"{k}: batches={v['batches']} req={v['requests']} "
                  f"svc_mean={v['req_service_ms']['mean']:.1f}ms "
                  f"p95={v['req_service_ms']['p95']:.1f}ms "
                  f"busy={v['within_batch_busy_frac']['mean']:.2f} "
                  f"cold={v['cold_half_req_mean_ms']:.1f}ms "
                  f"warm={v['warm_half_req_mean_ms']:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
