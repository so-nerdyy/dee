#!/usr/bin/env python3
"""Discrete-event critical-path simulator for dee's exact DSV4 decode pipeline.

Pipeline modeled (per layer request):
    SSD read (shared saturated disk: aggregate BW, latency grows with
    concurrency) -> host pack copy (per-record host CPU) -> H2D (per-GPU
    PCIe aggregate; per-record or grouped) -> packed expert residency (VRAM)
    -> dequant + routed expert compute (per-GPU engine, overlaps copies)
    -> dense work + host orchestration (bridge syncs, dispatch) per layer.

Structural facts baked in (measured; see CALIBRATION.md):
  - v63 (3 lanes) vs v65 (6 lanes): identical batch wall, identical aggregate
    read throughput -> the disk saturates; lane count does not change drain.
  - v65 per-step regression: wall_ms ~= 891 + 3.58 * storage_MB -> a
    saturated-disk drain term plus a ~0.9 s/step non-disk serial tail.
  - The official router is sequential: step k+1 routes are unknown until
    step k completes -> no cross-step read lookahead. Within a step the
    runtime stages on demand per layer today; `staging_lead_layers` models
    earlier exact staging.

Simulated speedup != measured speedup. This tool ranks hypotheses.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

RECORD_BYTES = 13_369_344
RECORD_MIB = RECORD_BYTES / (1024 * 1024)   # 12.75 MiB packed fp4
LAYERS = 43
TOPK = 6
EXPERTS_PER_LAYER = 256
LAYER_SPLIT_GPU0 = 22                        # cuda0: layers 0..21, cuda1: 22..42

# --- sealed v65 (measured, 2x T4, 15 decode tokens) -------------------------
V65 = {
    "decode_wall_s": 72.267,
    "tokens": 15,
    "step_walls_ms": [5751.08, 4237.45, 4423.46, 6122.33, 6876.35, 6162.25,
                      5227.35, 4721.48, 3573.43, 3714.79, 4587.91, 3654.13,
                      4276.98, 5114.55, 3823.66],
    "requests": {"cuda0": 2613, "cuda1": 2488},
    "resident_hits": {"cuda0": 328, "cuda1": 327},
    "pack_hits": {"cuda0": 1223, "cuda1": 1395},
    "ssd_reads": {"cuda0": 1390, "cuda1": 1091},
    "h2d_bytes": {"cuda0": 30548951040, "cuda1": 28864413696},
    "bridge_host_syncs": 1376,
    # per-step (decode) measured accounting, summed over both GPUs, from
    # profile.json per_token_accounting:
    #   requests      = 43 layers x top-6 = 258 every step
    #   resident_hits = engine resident hits (no transfer)
    #   h2d_copies    = records transferred H2D (13.369 MB each; matches
    #                   h2d_bytes per step exactly)
    #   ssd_reads     = storage_requests (records read from the disk store;
    #                   x 13.369 MB = storage_bytes per step exactly)
    #   pack_sourced  = h2d_copies - ssd_reads (host-pack served transfers)
    "per_step": {
        "requests": [258] * 15,
        "resident_hits": [67, 74, 71, 49, 38, 41, 28, 31, 31, 43, 40, 35, 34, 37, 36],
        "h2d_copies": [191, 184, 187, 209, 220, 217, 230, 227, 227, 215, 218, 223, 224, 221, 222],
        "ssd_reads": [96, 69, 80, 111, 125, 114, 96, 80, 56, 64, 79, 61, 69, 91, 61],
    },
}

# --- sealed v52 stage profile (same route journal, profiler instrumented) ---
V52 = {
    "compute_per_record_ms": (467.8 + 207.7 + 16.7 + 302.7 + 41.8) / 2285.0,
    "host_pack_ms_per_record": 6523.7 / 2285.0,
    "decode_wall_s": 138.715,
}

# --- microbench (campaign-idle4 prepack end-to-end, T4, fp16 synthetic) -----
IDLE4_H2D_GBPS = 4.008706048 / 0.745045730        # ~5.38 GB/s pinned H2D


def default_params() -> dict:
    """`src:` measured, `cal:` calibrated, `model:` modeling choice,
    `unk:` unknown (marked UNKNOWN in reports)."""
    return {
        # Aggregate concurrent-read throughput of the shared disk (B/ms via
        # mb_s). cal: fitted to v65 per-step walls (marginal slope 3.58 s/GB);
        # independent p50-based estimate ~354 MB/s (CALIBRATION.md).
        "ssd_aggregate_mb_s": 279.0,
        "read_lanes": 6,                       # src: v65 engine_config
        "read_queue_depth": 6,                 # src: v65 engine_config
        "host_pack_ms_per_record": V52["host_pack_ms_per_record"],   # src: v52
        # H2D: idle4 microbench measures 5.38 GB/s (small fp16 copies, quiet
        # single-GPU). The effective rate in the real dual-GPU run is fit
        # (cal) and reported alongside the microbench ceiling.
        "h2d_gbps": 1.1,                       # cal: fit to v65 steps
        "h2d_gbps_microbench": round(IDLE4_H2D_GBPS, 3),  # src: idle4 ceiling
        "h2d_per_copy_overhead_us": 20.0,      # unk: not measured
        "gpu_compute_per_record_ms": round(V52["compute_per_record_ms"], 4),  # src: v52
        "compute_overlaps_copies": True,       # src: idle4 overlap_fraction 0.22
        "host_orchestration_per_layer_ms": 1.5,  # cal: fits v65 intercept
        "host_syncs_per_step": 86.0,           # src: v65 bridge 1376/16
        "host_sync_overhead_ms": 0.02,         # unk: not measured
        "staging_lead_layers": 8,              # cal: fit to v65 steps
        "grouped_h2d": False,
        "grouped_dispatch": False,             # model: halves host slices
        "host_orch_reduction": 0.0,
        "host_buffer_reuse": False,            # Codex A/B candidate
        "seed": 1234,
    }


# ---------------------------------------------------------------------------
# Route journal (sealed v65; read-only)
# ---------------------------------------------------------------------------

def load_route_bundles(trace_path: Path) -> list:
    """routed_experts.jsonl -> [forward][layer] = flat list of expert ids.

    Journal schema (sealed v65): one record per (forward_step, layer, device)
    with `expert_ids_rank_order` = [token_rows][topk]; layer split is implicit
    via the `device` field (cuda:0 first 22 layers, cuda:1 the rest)."""
    bundles = [[] for _ in range(16)]
    seen_fwds = set()
    with trace_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            fs = rec["forward_step"]
            L = rec["layer"]
            ids = [e for row in rec["expert_ids_rank_order"] for e in row]
            if len(ids) != TOPK * rec.get("token_rows", 1):
                raise ValueError(f"bad topk in forward {fs} layer {L}")
            bundles[fs].append((L, ids))
            seen_fwds.add(fs)
    if seen_fwds != set(range(16)):
        raise ValueError(f"expected 16 forwards, got {sorted(seen_fwds)}")
    out = []
    for fwd in bundles:
        fwd.sort(key=lambda t: t[0])
        if [L for L, _ in fwd] != list(range(LAYERS)):
            raise ValueError("non-contiguous layers in a forward")
        out.append([ids for _, ids in fwd])
    return out


def gpu_layer_uniques(bundles) -> list:
    """->[forward][gpu][layer] = sorted unique expert ids for that GPU."""
    out = []
    for fwd in bundles:
        per_gpu = ([], [])
        for L, ids in enumerate(fwd):
            g = 0 if L < LAYER_SPLIT_GPU0 else 1
            per_gpu[g].append(sorted(set(ids)))
        out.append(per_gpu)
    return out


# ---------------------------------------------------------------------------
# Demand plan: sealed per-step hit/miss structure distributed over GPUs/layers
# ---------------------------------------------------------------------------

def demand_plan(uniques: list) -> list:
    """->[step][gpu][layer] = {uniques, hit, pack, read} record counts.

    Uses the SEALED per-step accounting: requests = 258; resident hits and
    H2D copies measured per step; ssd_reads = storage_requests (which times
    13.369 MB reproduces storage_bytes exactly). pack-sourced transfers =
    h2d_copies - ssd_reads. Split across GPUs by the sealed per-GPU request
    totals (cuda0 2613 : cuda1 2488) and across layers proportional to the
    journal's per-layer unique-expert demand (remainder on the largest)."""
    ps = V65["per_step"]
    decode = uniques[1:]
    share = {g: V65["requests"][f"cuda{g}"] /
             (V65["requests"]["cuda0"] + V65["requests"]["cuda1"]) for g in (0, 1)}
    plan = []
    for si, per_gpu in enumerate(decode):
        step = {}
        for gpu in (0, 1):
            layers = per_gpu[gpu]
            total_uniq = sum(len(u) for u in layers)
            n_req = ps["requests"][si] * share[gpu]
            n_hit = ps["resident_hits"][si] * share[gpu]
            n_xfer = ps["h2d_copies"][si] * share[gpu]
            n_read = ps["ssd_reads"][si] * share[gpu]
            n_pack = n_xfer - n_read
            lay_plan = []
            rem = {"hit": n_hit, "pack": n_pack, "read": n_read}
            for u in layers:
                w = len(u) / max(total_uniq, 1)
                h = min(rem["hit"], n_hit * w)
                p = min(rem["pack"], n_pack * w)
                r = min(rem["read"], n_read * w)
                rem["hit"] -= h; rem["pack"] -= p; rem["read"] -= r
                lay_plan.append({"uniques": len(u), "hit": h, "pack": p, "read": r})
            big = max(range(len(layers)), key=lambda i: len(layers[i]))
            lay_plan[big]["hit"] += rem["hit"]
            lay_plan[big]["pack"] += rem["pack"]
            lay_plan[big]["read"] += rem["read"]
            step[gpu] = lay_plan
        plan.append(step)
    return plan


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SharedBandwidthServer:
    """Saturated shared medium (the SSD): jobs drain at the aggregate rate;
    a job submitted into a backlog finishes after all earlier work
    (work-conserving FIFO, per-read latency grows with concurrency — matches
    the v63/v65 lane-A/B signature). Times in ms, bytes in bytes."""
    bytes_per_ms: float
    next_free: float = 0.0

    def submit(self, nbytes: float, t: float) -> float:
        start = max(t, self.next_free)
        self.next_free = start + nbytes / self.bytes_per_ms
        return self.next_free


def simulate(params: dict, uniques: list) -> dict:
    """Event-driven decode simulation over the 15 sealed decode steps.

    Per step (GPUs in lockstep; the bridge syncs them once per step):
      - Layer L's transfers are submitted at the consumer's completion of
        layer (L - staging_lead). lead=0 => on-demand (submit after previous
        layer computes); lead>=42 => full-step-ahead staging.
      - ssd_read: SSD drain -> pack -> H2D. pack_hit: pack -> H2D.
      - GPU compute chain: start = max(prev layer compute end, this layer's
        H2D end, host slice); each layer's compute is sequential (43-layer
        transformer).
      - Step wall = max over GPUs of compute-chain end + bridge sync tail.
    """
    plan = demand_plan(uniques)
    h2d_bytes_per_ms = params["h2d_gbps"] * 1e9 / 1e3
    h2d_per_record_ms = RECORD_BYTES / h2d_bytes_per_ms \
        + params["h2d_per_copy_overhead_us"] / 1e3
    if params["host_buffer_reuse"]:
        pack_ms = 0.05                       # model: pointer swap, not copy
    else:
        pack_ms = params["host_pack_ms_per_record"]
    orch = params["host_orchestration_per_layer_ms"]
    if params["grouped_dispatch"]:
        orch *= 0.5
    orch *= (1.0 - params["host_orch_reduction"])
    sync_tail_ms = params["host_syncs_per_step"] * params["host_sync_overhead_ms"]

    steps_out = []
    for step in plan:
        ssd = SharedBandwidthServer(params["ssd_aggregate_mb_s"] * 1e6 / 1e3)
        pack_busy = {0: 0.0, 1: 0.0}
        h2d_busy = {0: 0.0, 1: 0.0}
        gpus = {}
        for gpu in (0, 1):
            layers = step[gpu]
            n_layers = len(layers)
            lead = min(params["staging_lead_layers"], n_layers - 1)
            compute_end = 0.0
            consumer_slot = [0.0] * n_layers       # completion time of layer L
            last_h2d_end = 0.0
            h2d_total_ms = 0.0
            compute_only_ms = 0.0
            for L in range(n_layers):
                lp = layers[L]
                n = lp["uniques"]
                n_xfer = lp["pack"] + lp["read"]
                # submission time for this layer's records: the consumer
                # reaches layer (L - lead) when layer (L - lead - 1) has
                # finished; lead=0 => submit right after the previous layer
                # computes (current on-demand staging).
                src = L - 1 - lead
                submit_t = consumer_slot[src] if src >= 0 else 0.0
                dep = submit_t
                if n_xfer > 0:
                    nbytes = n_xfer * RECORD_BYTES
                    if lp["read"] > 0:
                        dep = max(dep, ssd.submit(lp["read"] * RECORD_BYTES, submit_t))
                    # pack copies (pipelined batch after the read drain)
                    pack_end = max(dep, pack_busy[gpu]) + pack_ms * n_xfer
                    pack_busy[gpu] = pack_end
                    dep = max(dep, pack_end)
                    # H2D: grouped = one copy (one overhead); else per record
                    if params["grouped_h2d"]:
                        h2d_end = max(dep, h2d_busy[gpu]) + nbytes / h2d_bytes_per_ms \
                            + params["h2d_per_copy_overhead_us"] / 1e3
                    else:
                        h2d_end = max(dep, h2d_busy[gpu]) + n_xfer * h2d_per_record_ms
                    h2d_busy[gpu] = h2d_end
                    h2d_total_ms += n_xfer * h2d_per_record_ms
                    dep = max(dep, h2d_end)
                    last_h2d_end = h2d_end
                # compute chain
                cstart = max(compute_end, dep) + orch
                compute_end = cstart + n * params["gpu_compute_per_record_ms"]
                consumer_slot[L] = compute_end
                compute_only_ms += n * params["gpu_compute_per_record_ms"]
            gpus[gpu] = {
                "wall_ms": compute_end,
                "last_h2d_end_ms": last_h2d_end,
                "compute_end_ms": compute_end,
                "h2d_busy_ms": h2d_busy[gpu],
                "h2d_total_ms": h2d_total_ms,
                "pack_busy_ms": pack_busy[gpu],
                "compute_only_ms": compute_only_ms,
                "disk_drain_ms": ssd.next_free,
            }
        wall = max(gpus[0]["wall_ms"], gpus[1]["wall_ms"]) + sync_tail_ms
        if not params["compute_overlaps_copies"]:
            # non-overlap scenario: per GPU, transfers finish before compute
            # starts (conservative serialization of the two engines).
            serial = max(gpus[g]["compute_only_ms"] + gpus[g]["h2d_busy_ms"]
                         for g in (0, 1))
            wall = serial + sync_tail_ms
        steps_out.append({
            "wall_ms": wall,
            "disk_drain_ms": max(gpus[0]["disk_drain_ms"], gpus[1]["disk_drain_ms"]),
            "h2d_busy_ms": max(gpus[0]["h2d_busy_ms"], gpus[1]["h2d_busy_ms"]),
            "h2d_total_ms": max(gpus[0]["h2d_total_ms"], gpus[1]["h2d_total_ms"]),
            "compute_only_ms": max(gpus[0]["compute_only_ms"], gpus[1]["compute_only_ms"]),
            "compute_end_ms": max(gpus[0]["compute_end_ms"], gpus[1]["compute_end_ms"]),
            "reads": sum(step[g][L]["read"] for g in (0, 1) for L in range(len(step[g]))),
            "pack_hits": sum(step[g][L]["pack"] for g in (0, 1) for L in range(len(step[g]))),
            "resident_hits": sum(step[g][L]["hit"] for g in (0, 1) for L in range(len(step[g]))),
        })

    wall_s = sum(s["wall_ms"] for s in steps_out) / 1000.0
    return {
        "predicted_wall_s": wall_s,
        "observed_wall_s": V65["decode_wall_s"],
        "abs_error_s": wall_s - V65["decode_wall_s"],
        "rel_error": (wall_s - V65["decode_wall_s"]) / V65["decode_wall_s"],
        "steps": steps_out,
        "observed_steps_ms": V65["step_walls_ms"],
    }


# ---------------------------------------------------------------------------
# Calibration: fit disk rate + host orchestration to v65 per-step walls
# ---------------------------------------------------------------------------

def calibrate(uniques: list, params: dict, verbose: bool = False) -> tuple[dict, dict]:
    """Four-parameter fit (disk rate, effective H2D rate, host orch,
    staging lead) to the sealed v65 per-step walls. Coarse grid + refinement."""
    def sse_for(ssd_r, h2d_r, host, lead):
        p = dict(params)
        p["ssd_aggregate_mb_s"] = float(ssd_r)
        p["h2d_gbps"] = float(h2d_r)
        p["host_orchestration_per_layer_ms"] = float(host)
        p["staging_lead_layers"] = int(lead)
        res = simulate(p, uniques)
        pred = [s["wall_ms"] for s in res["steps"]]
        obs = V65["step_walls_ms"]
        return sum((a - b) ** 2 / b ** 2 for a, b in zip(pred, obs)), res

    best_sse, best = None, None
    for ssd_r in [230, 270, 290, 320, 360, 420]:
        for h2d_r in [0.7, 0.9, 1.2, 1.8, 2.5, 5.38]:
            for host in [0.1, 0.5, 1.0, 2.0]:
                for lead in [0, 4, 8, 16, 30, 42]:
                    sse, _ = sse_for(ssd_r, h2d_r, host, lead)
                    if best_sse is None or sse < best_sse:
                        best_sse, best = sse, (ssd_r, h2d_r, host, lead)
    for _ in range(2):
        cands = []
        for d_ssd in (-20, -10, 0, 10, 20):
            for d_h in (-0.08, -0.04, 0, 0.04, 0.08):
                for d_ho in (-0.15, -0.05, 0, 0.05, 0.15):
                    for d_l in (-2, -1, 0, 1, 2):
                        c = (best[0] + d_ssd, round(best[1] + d_h, 3),
                             round(best[2] + d_ho, 2), max(0, min(42, best[3] + d_l)))
                        if c[0] > 50 and c[1] > 0.1 and c[2] >= 0:
                            cands.append(c)
        for c in cands:
            sse, _ = sse_for(*c)
            if sse < best_sse:
                best_sse, best = sse, c
    if verbose:
        print("best fit:", best, "sse", best_sse)
    p = dict(params)
    p["ssd_aggregate_mb_s"] = float(best[0])
    p["h2d_gbps"] = float(best[1])
    p["host_orchestration_per_layer_ms"] = float(best[2])
    p["staging_lead_layers"] = int(best[3])
    return p, simulate(p, uniques)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_TRACE_CANDIDATES = [
    Path("dee.cpp/tmp/v65-terminal-fetch-20260903T0202Z/routed_experts.jsonl"),
    Path("../../dynamic_expert_eviction/dee.cpp/tmp/v65-terminal-fetch-20260903T0202Z/routed_experts.jsonl"),
    Path("../dynamic_expert_eviction/dee.cpp/tmp/v65-terminal-fetch-20260903T0202Z/routed_experts.jsonl"),
]


def find_trace() -> Path | None:
    for cand in DEFAULT_TRACE_CANDIDATES:
        if cand.exists():
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, default=None,
                    help="Path to sealed routed_experts.jsonl.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Fit disk rate + host orchestration to sealed v65 steps.")
    ap.add_argument("--baseline", action="store_true",
                    help="Run with default params; print predicted vs observed.")
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path.")
    args = ap.parse_args()

    trace = args.trace or find_trace()
    if trace is None:
        raise SystemExit("sealed routed_experts.jsonl not found; pass --trace")
    uniques = gpu_layer_uniques(load_route_bundles(trace))
    params = default_params()
    if args.calibrate:
        params, res = calibrate(uniques, params)
    else:
        res = simulate(params, uniques)
    summary = {k: res[k] for k in
               ("predicted_wall_s", "observed_wall_s", "abs_error_s", "rel_error")}
    summary["params"] = params
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"params": params, "result": res},
                                       indent=2, default=str))


if __name__ == "__main__":
    main()
