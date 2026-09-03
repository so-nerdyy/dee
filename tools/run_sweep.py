#!/usr/bin/env python3
"""run_sweep.py — Phases C + D batch driver for expert_hierarchy_sim.py.

Sweeps the consumer geometry (RTX 5070 Ti 16 GB / ~32 GB RAM / consumer NVMe)
across the bounded budget grids and codec hypotheses, using dee's sealed v62
route trace. Emits JSON results + CSV tables + heatmap PNGs under
research/ultralowbit/results/. READ-ONLY with respect to all evidence.

All TPS outputs are STORAGE-ONLY CEILING values (SSD+PCIE roofline). They are
not end-to-end performance predictions (compute, attention, scheduling and
kernel overheads are excluded by construction).
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_hierarchy_sim import (  # noqa: E402
    DEFAULT_TRACE, GIB, belady_decode_only_hit_rate, expert_bytes, load_trace,
    simulate, TARGET_TPS,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research/ultralowbit/results",
)

VRAM_GIB_GRID = [2, 4, 6, 8]
RAM_GIB_GRID = [4, 8, 12, 16, 20]
SSD_GBPS_GRID = [1, 2, 3.5, 5, 7, 10]

CODEC_SETS = {
    "MXFP4": ("MXFP4_CURRENT", None),
    "STQ1_0": ("STQ1_0_IDEALIZED", None),
    "IQ2_XXS": ("IQ2_XXS_APPROX", None),
}


def max_feasible_tps(ssd_bytes_per_token: float, h2d_bytes_per_token: float,
                     ssd_gbps: float, pcie_gbps: float) -> float:
    """STORAGE-ONLY CEILING from decode-phase steady-state bytes/token."""
    """STORAGE-ONLY CEILING: min over SSD and PCIe constraints."""
    if ssd_bytes_per_token <= 0 or h2d_bytes_per_token <= 0:
        return math.inf
    return min(ssd_gbps * 1e9 / ssd_bytes_per_token,
               pcie_gbps * 1e9 / h2d_bytes_per_token)


def required_hit_rate(tps: float, ssd_gbps: float, eb: float,
                      requests_per_decode_token: float) -> float:
    """Phase D: combined RAM+VRAM avoidance rate needed so that
    ssd_bytes/token = (1-hit)*eb*requests_per_decode_token <= ssd_gbps*1e9/tps."""
    budget_bytes_per_token = ssd_gbps * 1e9 / tps
    miss_bytes = budget_bytes_per_token / (eb * requests_per_decode_token)
    return max(0.0, min(1.0, 1.0 - miss_bytes))


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    trace = load_trace(DEFAULT_TRACE)
    decode_reqs = sum(len(s.experts) for s in trace if s.phase == "decode")
    decode_tokens = sum(s.weight for s in trace if s.phase == "decode")
    reqs_per_decode_token = decode_reqs / decode_tokens
    print(f"trace: {len(trace)} steps, decode reqs/token = {reqs_per_decode_token:.2f}")

    rows = []
    for codec_name, (codec, bits) in CODEC_SETS.items():
        eb = expert_bytes(codec, bits)
        for vram_gib, ram_gib in itertools.product(VRAM_GIB_GRID, RAM_GIB_GRID):
            warm = codec_name != "IQ2_XXS"  # run both modes for MXFP4/STQ
            res = simulate(trace, lambda k, _e=eb: _e,
                           int(vram_gib * GIB), int(ram_gib * GIB),
                           policy="LRU", trace_label=codec_name,
                           warm_start=True)
            # cold mode for comparison (prefill counted as normal traffic)
            cold = simulate(trace, lambda k, _e=eb: _e,
                            int(vram_gib * GIB), int(ram_gib * GIB),
                            policy="LRU", trace_label=codec_name + "-cold",
                            warm_start=False)
            for ssd_gbps, pcie_gbps in itertools.product(SSD_GBPS_GRID, [16.0]):
                tps = max_feasible_tps(res["decode_ssd_bytes_per_token"],
                                       res["decode_h2d_bytes_per_token"],
                                       ssd_gbps, pcie_gbps)
                rows.append({
                    "codec": codec_name,
                    "vram_gib": vram_gib,
                    "ram_gib": ram_gib,
                    "ssd_gbps": ssd_gbps,
                    "pcie_gbps": pcie_gbps,
                    "per_expert_bytes": eb,
                    "vram_hit_rate": round(res["vram_hit_rate"], 4),
                    "ram_hit_rate": round(res["ram_hit_rate"], 4),
                    "combined_hit_rate": round(res["combined_ram_vram_hit_rate"], 4),
                    "ssd_miss_rate": round(res["ssd_miss_rate"], 4),
                    "ssd_bytes_per_token": res["ssd_bytes_per_token"],
                    "h2d_bytes_per_token": res["h2d_bytes_per_token"],
                    "decode_combined_hit_rate": round(res["decode_combined_hit_rate"], 4),
                    "decode_ssd_bytes_per_token": res["decode_ssd_bytes_per_token"],
                    "decode_h2d_bytes_per_token": res["decode_h2d_bytes_per_token"],
                    "vram_evictions": res["vram_evictions"],
                    "ram_evictions": res["ram_evictions"],
                    "storage_only_ceiling_tps": round(tps, 3),
                    "warm_start": True,
                })
            # one cold row per budget for reference (not swept over SSD)
            rows.append({
                "codec": codec_name + "_cold",
                "vram_gib": vram_gib,
                "ram_gib": ram_gib,
                "ssd_gbps": None,
                "pcie_gbps": None,
                "per_expert_bytes": eb,
                "vram_hit_rate": round(cold["vram_hit_rate"], 4),
                "ram_hit_rate": round(cold["ram_hit_rate"], 4),
                "combined_hit_rate": round(cold["combined_ram_vram_hit_rate"], 4),
                "ssd_miss_rate": round(cold["ssd_miss_rate"], 4),
                "ssd_bytes_per_token": cold["ssd_bytes_per_token"],
                "h2d_bytes_per_token": cold["h2d_bytes_per_token"],
                "vram_evictions": cold["vram_evictions"],
                "ram_evictions": cold["ram_evictions"],
                "storage_only_ceiling_tps": None,
                "warm_start": False,
            })

    # MIXED 60/30/10 (memory-only, frequency-ranked)
    mixed_res = run_mixed(trace)
    rows.append(mixed_summary_row(mixed_res, "MIXED_60_30_10"))
    print('MIXED decode: ssd/tok=%.0f h2d/tok=%.0f dec_hit=%.3f' % (
        mixed_res['decode_ssd_bytes_per_token'], mixed_res['decode_h2d_bytes_per_token'],
        mixed_res['decode_combined_hit_rate']))

    # ------------------------------------------------------------------ JSON
    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "trace": DEFAULT_TRACE,
            "decode_requests_per_decode_token": reqs_per_decode_token,
            "decode_tokens_in_trace": decode_tokens,
            "rows": rows,
            "mixed_detail": mixed_res,
            "labels": {
                "storage_only_ceiling_tps":
                    "STORAGE-ONLY CEILING (SSD+PCIe roofline; NOT end-to-end TPS)",
                "warm_start": "RAM tier pre-seeded with prefill-touched experts "
                              "(perfect warmup); SSD traffic = decode-phase misses only",
                "mixed": "MEMORY-ONLY SIMULATION: frequency-ranked codecs, no quality data",
            },
        }, f, indent=2)

    # ------------------------------------------------------------------- CSV
    csv_path = os.path.join(OUT_DIR, "sweep_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --------------------------------------------------------------- heatmaps
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping heatmaps")
        return 0

    for codec_name in list(CODEC_SETS) + ["MIXED"]:
        grid = build_tps_grid(codec_name, rows)
        if grid is None:
            continue
        plot_heatmap(grid, codec_name, f"{OUT_DIR}/heatmap_{codec_name}.png")

    # Phase D tables: required hit rate + feasibility per target TPS
    phase_d = {}
    for codec_name, (codec, bits) in CODEC_SETS.items():
        eb = expert_bytes(codec, bits)
        phase_d[codec_name] = phase_d_table(eb, reqs_per_decode_token)
    phase_d["MIXED_60_30_10"] = phase_d_table(
        mixed_avg_bytes(mixed_res), reqs_per_decode_token)
    with open(os.path.join(OUT_DIR, "phase_d_required_hit_rates.json"), "w",
              encoding="utf-8") as f:
        json.dump(phase_d, f, indent=2)

    print(f"wrote {len(rows)} rows -> {csv_path}")
    print(f"wrote phase_d_required_hit_rates.json")
    return 0


def run_mixed(trace):
    """60% STQ / 30% IQ2 / 10% MXFP4 by observed request frequency."""
    from collections import Counter
    freq = Counter()
    for step in trace:
        for key in step.experts:
            freq[key] += 1
    keys = sorted(freq, key=lambda k: -freq[k])
    n = len(keys)
    n_mxfp4 = max(1, int(round(0.10 * n)))
    n_iq2 = int(round(0.30 * n))
    assign = {}
    for i, k in enumerate(keys):
        if i < n_mxfp4:
            assign[k] = "MXFP4_CURRENT"
        elif i < n_mxfp4 + n_iq2:
            assign[k] = "IQ2_XXS_APPROX"
        else:
            assign[k] = "STQ1_0_IDEALIZED"

    def eb_for(k, _a=assign):
        return expert_bytes(_a.get(k, "STQ1_0_IDEALIZED"))

    res = simulate(trace, eb_for, int(8 * GIB), int(16 * GIB),
                   policy="LRU", trace_label="MIXED_60_30_10", warm_start=True)
    res["assignment_counts"] = dict(Counter(assign.values()))
    return res


def mixed_summary_row(res, label):
    import statistics
    return {
        "codec": label,
        "vram_gib": 8, "ram_gib": 16, "ssd_gbps": "swept", "pcie_gbps": 16.0,
        "per_expert_bytes": "variable",
        "vram_hit_rate": round(res["vram_hit_rate"], 4),
        "ram_hit_rate": round(res["ram_hit_rate"], 4),
        "combined_hit_rate": round(res["combined_ram_vram_hit_rate"], 4),
        "ssd_miss_rate": round(res["ssd_miss_rate"], 4),
        "ssd_bytes_per_token": res["ssd_bytes_per_token"],
        "h2d_bytes_per_token": res["h2d_bytes_per_token"],
        "vram_evictions": res["vram_evictions"],
        "ram_evictions": res["ram_evictions"],
        "storage_only_ceiling_tps": None,
        "warm_start": True,
    }


def mixed_avg_bytes(res):
    """Average SSD bytes per miss for the mixed configuration."""
    if res["ssd_misses"]:
        return res["ssd_bytes_total"] / res["ssd_misses"]
    return expert_bytes("STQ1_0_IDEALIZED")


def build_tps_grid(codec_name, rows):
    """(vram_gib x ssd_gbps) grid of max STORAGE-ONLY CEILING TPS at the best
    RAM budget (i.e., max over ram_gib)."""
    grid = [[0.0] * len(SSD_GBPS_GRID) for _ in VRAM_GIB_GRID]
    found = False
    for i, vram in enumerate(VRAM_GIB_GRID):
        for j, ssd in enumerate(SSD_GBPS_GRID):
            best = 0.0
            for r in rows:
                if (r["codec"] == codec_name and r["vram_gib"] == vram
                        and r["ssd_gbps"] == ssd and r["warm_start"]
                        and r["storage_only_ceiling_tps"] is not None):
                    best = max(best, r["storage_only_ceiling_tps"])
            grid[i][j] = best
            if best > 0:
                found = True
    return grid if found else None


def plot_heatmap(grid, codec_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [[min(v, 50) for v in row] for row in grid]
    im = ax.imshow(data, cmap="viridis", aspect="auto",
                   extent=[0, len(SSD_GBPS_GRID), 0, len(VRAM_GIB_GRID)])
    ax.set_xticks([i + 0.5 for i in range(len(SSD_GBPS_GRID))])
    ax.set_xticklabels([str(s) for s in SSD_GBPS_GRID])
    ax.set_yticks([i + 0.5 for i in range(len(VRAM_GIB_GRID))])
    ax.set_yticklabels([str(v) for v in reversed(VRAM_GIB_GRID)])
    ax.set_xlabel("SSD bandwidth (GB/s)")
    ax.set_ylabel("VRAM expert cache (GiB)")
    for i in range(len(VRAM_GIB_GRID)):
        for j in range(len(SSD_GBPS_GRID)):
            val = grid[len(VRAM_GIB_GRID) - 1 - i][j]
            txt = f"{val:.1f}" if val < 50 else "50+"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                    color="white" if data[len(VRAM_GIB_GRID) - 1 - i][j] < 25 else "black",
                    fontsize=9)
    ax.set_title(f"STORAGE-ONLY CEILING (TPS) — {codec_name}\n"
                 f"best RAM budget; PCIe 16 GB/s; NOT end-to-end TPS")
    fig.colorbar(im, ax=ax, label="storage-only ceiling TPS (clipped at 50)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def phase_d_table(eb: float, reqs_per_decode_token: float) -> dict:
    """Required combined RAM+VRAM avoidance rate per target TPS per SSD BW."""
    table = {}
    for tps in TARGET_TPS:
        per_tps = {}
        for ssd in SSD_GBPS_GRID:
            hr = required_hit_rate(tps, ssd, eb, reqs_per_decode_token)
            per_tps[str(ssd)] = round(hr, 4)
        table[f"{tps}TPS"] = per_tps
    table["requests_per_decode_token"] = round(reqs_per_decode_token, 3)
    table["per_expert_bytes"] = eb
    return table


if __name__ == "__main__":
    sys.exit(main())
