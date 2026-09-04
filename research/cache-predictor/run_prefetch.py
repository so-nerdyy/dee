#!/usr/bin/env python3
"""Phase E CLI: prefetch / proactive-replacement simulation.

Reproducible:
  python research/cache-predictor/run_prefetch.py

Writes results/prefetch_<trace>.json and a theoretical stall sweep.
No future routing information is used for prefetch decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sim.trace import load_trace, load_default_trace  # noqa: E402
from sim.predictor import to_layer_bundles  # noqa: E402
from sim.keepvalue import build_ranked_fn  # noqa: E402
from sim.prefetch import (  # noqa: E402
    PrefetchEngine,
    PrefetchPolicy,
    theoretical_stall_sweep,
)

RECORD_BYTES = 13_369_344


def run_layer(layer, ds, by_pair, cap, mode, max_pf, min_dwell):
    ranked_fn = build_ranked_fn(ds, by_pair, layer)
    pol = PrefetchPolicy(cap * RECORD_BYTES, RECORD_BYTES,
                         ranked_fn=ranked_fn)
    eng = PrefetchEngine(pol, record_bytes=RECORD_BYTES,
                         ranked_fn=ranked_fn, prefetch_mode=mode,
                         max_prefetch_per_bundle=max_pf,
                         min_dwell=min_dwell)
    eng.run(ds)
    return eng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--trace", default=None)
    ap.add_argument("--caps", default="8,16,32")
    args = ap.parse_args()

    if args.trace:
        tpath = Path(args.trace)
        demands = load_trace(tpath)
    else:
        demands, tpath = load_default_trace()
    label = tpath.parent.name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = to_layer_bundles(demands)
    by_pair = {(b.step, b.layer): b.keys for b in bundles}
    layers = sorted({d.layer for d in demands})
    slices: dict[int, list] = {L: [] for L in layers}
    for d in demands:
        slices[d.layer].append(d)

    caps = [int(c) for c in args.caps.split(",")]
    configs = []
    for cap in caps:
        for mode in ("off", "safe", "replace"):
            for max_pf in (4, 6, 8, 12):
                if mode == "off" and max_pf != 4:
                    continue  # baseline only once
                configs.append((cap, mode, max_pf))

    rows = []
    for cap, mode, max_pf in configs:
        agg = {
            "accesses": 0, "hits": 0, "misses": 0,
            "prefetch_issued": 0, "prefetch_useful": 0,
            "prefetch_wasted": 0, "prefetch_bytes": 0,
            "wasted_prefetch_bytes": 0, "demand_misses_converted": 0,
            "evictions": 0,
        }
        for L in layers:
            eng = run_layer(L, slices[L], by_pair, cap, mode, max_pf,
                            min_dwell=6)
            agg["accesses"] += eng.m.accesses
            agg["hits"] += eng.m.hits
            agg["misses"] += eng.m.misses
            agg["prefetch_issued"] += eng.prefetch_issued
            agg["prefetch_useful"] += eng.prefetch_useful
            agg["prefetch_wasted"] += eng.prefetch_wasted
            agg["prefetch_bytes"] += eng.prefetch_bytes
            agg["wasted_prefetch_bytes"] += eng.wasted_prefetch_bytes
            agg["demand_misses_converted"] += eng.demand_misses_converted
            agg["evictions"] += eng.m.evictions
        rows.append({
            "capacity_experts_per_layer": cap,
            "mode": mode,
            "max_prefetch_per_bundle": max_pf,
            "min_dwell": 6,
            **agg,
            "hit_rate": round(agg["hits"] / agg["accesses"], 6),
            "useful_prefetch_pct": round(
                agg["prefetch_useful"]
                / max(1, agg["prefetch_useful"] + agg["prefetch_wasted"]), 6),
            "prefetch_bytes_gib": round(agg["prefetch_bytes"] / 2**30, 3),
            "wasted_prefetch_bytes_gib": round(
                agg["wasted_prefetch_bytes"] / 2**30, 3),
        })
        print(f"cap={cap:2d} mode={mode:7s} max_pf={max_pf:2d} "
              f"misses={agg['misses']:5d} pf={agg['prefetch_issued']:5d} "
              f"useful%={rows[-1]['useful_prefetch_pct']:.3f} "
              f"converted={agg['demand_misses_converted']:5d}")

    doc = {
        "trace": str(tpath),
        "semantics": "per-layer engines; causal cross-token predictor "
                     "(prev-token bundle + popularity); min_dwell=6",
        "rows": rows,
    }

    # theoretical stall sweep (LABELED theoretical, not measured)
    sel = []
    for r in rows:
        if r["capacity_experts_per_layer"] == 8:
            sel.append((f"cap8-{r['mode']}-pf{r['max_prefetch_per_bundle']}",
                        r["misses"]))
    doc["theoretical_stall_sweep_cap8"] = theoretical_stall_sweep(sel,
                                                                  RECORD_BYTES)
    doc["stall_sweep_note"] = (
        "THEORETICAL: exposed_stall = misses * read_p50 * (1-overlap), "
        "read_p50 scaled from the sealed v65 measured 256.99 ms at "
        "54.2 MiB/s; overlap 82.3% trace-measured. NOT a wall-time "
        "benchmark; no TPS claim is made.")

    p = out_dir / f"prefetch_{label}.json"
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
