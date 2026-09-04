#!/usr/bin/env python3
"""Phase B/D CLI: layered cache-policy tournament (dee per-layer semantics).

Reproducible:
  python research/cache-predictor/run_tournament.py
  python research/cache-predictor/run_tournament.py --predictor-only
  python research/cache-predictor/run_tournament.py --tune

Writes results/tournament_<trace>.csv/.json,
results/predictor_<trace>.json, results/keepvalue_tuning_<trace>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sim.trace import load_trace, load_default_trace  # noqa: E402
from sim.policies import make_policy  # noqa: E402
from sim.engine import LayeredEngine  # noqa: E402
from sim.predictor import (  # noqa: E402
    to_layer_bundles,
    evaluate_ranked_predictor,
)
from sim.keepvalue import (  # noqa: E402
    KeepValuePolicy,
    build_ranked_fn,
    evaluate_keepvalue,
    tune_coefficients,
    split_steps,
)

RECORD_BYTES = 13_369_344
N_TOKENS = 16

POLICIES = ["lru", "lfu", "least_stale", "arc", "recency_freq", "belady",
            "keepvalue"]
CAPACITIES = [8, 16, 24, 32, 48, 64, 96]
TIER2_EXPERTS = [0, 64, 128, 256]
DEFAULT_KV_WEIGHTS = (1.0, 1.0, 1.0, 0.5)


def trace_label(path: Path) -> str:
    return path.parent.name


def build_factory(pol_name, cap_bytes, fut, layer_slices, by_pair,
                  kv_weights):
    """Return policy_factory(layer) for the LayeredEngine."""
    if pol_name == "keepvalue":
        def factory(layer):
            ranked_fn = build_ranked_fn(layer_slices[layer], by_pair, layer)
            return KeepValuePolicy(cap_bytes, RECORD_BYTES,
                                   ranked_fn=ranked_fn,
                                   weights=kv_weights)
        return factory

    def factory(layer):
        return make_policy(pol_name, cap_bytes, RECORD_BYTES, future=fut)
    return factory


def run_all(demands, capacities, tier2_experts, policies, n_layers,
            layer_slices, by_pair, kv_weights=DEFAULT_KV_WEIGHTS):
    rows = []
    # per-layer futures for the offline oracle: next-use chains must be
    # indexed over each layer's own demand slice, not the global trace
    # (LayeredEngine drives Belady with slice-local positions).
    belady_futures = {L: [d.scope for d in ds]
                      for L, ds in layer_slices.items()}
    for cap in capacities:
        cap_bytes = cap * RECORD_BYTES
        for t2e in tier2_experts:
            t2_bytes = t2e * RECORD_BYTES
            for pol_name in policies:
                if pol_name == "belady":
                    def factory(layer, cb=cap_bytes, fu=None):
                        return make_policy("belady", cb, RECORD_BYTES,
                                           future=belady_futures[layer])
                else:
                    factory = build_factory(pol_name, cap_bytes, None,
                                            layer_slices, by_pair,
                                            kv_weights)
                eng = LayeredEngine(factory, n_layers=n_layers,
                                    record_bytes=RECORD_BYTES,
                                    tier2_bytes_per_layer=t2_bytes)
                m = eng.run(demands)
                n = m.accesses
                rows.append({
                    "trace": "",
                    "policy": pol_name,
                    "capacity_experts_per_layer": cap,
                    "tier2_experts_per_layer": t2e,
                    "accesses": n,
                    "hits": m.hits,
                    "misses": m.misses,
                    "hit_rate": round(m.hits / n, 6),
                    "miss_rate": round(m.misses / n, 6),
                    "compulsory_misses": m.compulsory_misses,
                    "capacity_misses": m.capacity_misses,
                    "evictions": m.evictions,
                    "miss_bytes_per_token": round(m.miss_bytes / N_TOKENS, 1),
                    "ssd_bytes_per_token": round(m.ssd_bytes / N_TOKENS, 1),
                    "h2d_bytes_per_token": round(m.tier2_h2d_bytes / N_TOKENS, 1),
                    "evictions_per_token": round(m.evictions / N_TOKENS, 3),
                    "cache_churn": round((m.evictions + m.insertions) / n, 6),
                    "useful_residency_ticks": m.useful_residency_ticks,
                    "wasted_residency_ticks": m.wasted_residency_ticks,
                    "regret_vs_belady": None,
                })
    return rows


def attach_regret(rows):
    belady = {}
    for r in rows:
        if r["policy"] == "belady":
            belady[(r["capacity_experts_per_layer"],
                    r["tier2_experts_per_layer"])] = r["misses"]
    for r in rows:
        b = belady.get((r["capacity_experts_per_layer"],
                        r["tier2_experts_per_layer"]))
        r["regret_vs_belady"] = (r["misses"] - b) if b is not None else None


def tune_keepvalue(demands, n_layers, out_dir, label, by_pair):
    """Phase D: even-step train / odd-step validation.

    Feature inputs may reference any EARLIER step's routing (causally
    available at decode time); coefficient selection never sees
    validation-step outcomes (miss counts).
    """
    trials = [
        (0.0, 1.0, 0.0, 0.0),   # frequency only
        (1.0, 0.0, 0.0, 0.0),   # predictor only
        (0.0, 0.0, 1.0, 0.0),   # staleness only
        (0.0, 0.0, 0.0, 1.0),   # pollution only
        (1.0, 1.0, 1.0, 0.5),   # default
        (1.0, 2.0, 1.0, 0.5),
        (2.0, 1.0, 0.5, 0.1),
        (1.0, 1.0, 2.0, 1.0),
        (3.0, 1.0, 1.0, 0.2),
    ]
    train, valid = split_steps(demands, range(0, N_TOKENS, 2),
                               range(1, N_TOKENS, 2))

    def slices(ds):
        out: dict[int, list] = {}
        for d in ds:
            out.setdefault(d.layer, []).append(d)
        return out

    best, results = tune_coefficients(slices(train), n_layers, trials,
                                      RECORD_BYTES, by_pair,
                                      capacity_experts=8)
    valid_m = evaluate_keepvalue(slices(valid), n_layers, best,
                                 RECORD_BYTES, by_pair, capacity_experts=8)
    doc = {
        "trace": label,
        "method": "even-step train / odd-step validation; features use "
                  "only earlier-step routing (causal); coefficient "
                  "selection never sees validation outcomes",
        "best_coefficients": {"w_pred": best[0], "w_freq": best[1],
                              "w_stale": best[2], "w_pollute": best[3]},
        "train_misses": results[str(best)]["misses"],
        "valid_misses": valid_m.misses,
        "valid_accesses": valid_m.accesses,
        "valid_hit_rate": round(valid_m.hits / valid_m.accesses, 6),
        "all_trials": results,
        "overfit_risk_note": "single 16-step trace; even/odd steps share "
                             "the same prefill token, prompt and session; "
                             "treat coefficients as trace-local only.",
    }
    p = out_dir / f"keepvalue_tuning_{label}.json"
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=None)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--predictor-only", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--policies", default=",".join(POLICIES))
    args = ap.parse_args()

    if args.trace:
        tpath = Path(args.trace)
        demands = load_trace(tpath)
    else:
        demands, tpath = load_default_trace()
    label = trace_label(tpath)
    layers = sorted({d.layer for d in demands})
    n_layers = len(layers)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------- predictor (Phase C)
    bundles = to_layer_bundles(demands)
    all_pred = {}
    for name in ("freq_expert", "persist_layer", "cross_union", "combo"):
        all_pred[name] = evaluate_ranked_predictor(name, bundles)
    pred_path = out_dir / f"predictor_{label}.json"
    pred_path.write_text(json.dumps({
        "trace": str(tpath),
        "note": "causal predictions; official router authoritative; "
                "no future routing used for prediction",
        "results": all_pred,
    }, indent=2), encoding="utf-8")
    print(f"wrote {pred_path}")

    if args.predictor_only:
        return

    layer_slices: dict[int, list] = {L: [] for L in layers}
    for d in demands:
        layer_slices[d.layer].append(d)
    by_pair = {(b.step, b.layer): b.keys for b in bundles}

    # ---------------------------------------------- Phase D tuning
    kv_weights = DEFAULT_KV_WEIGHTS
    if args.tune:
        kv_weights = tune_keepvalue(demands, n_layers, out_dir, label,
                                    by_pair)

    # ---------------------------------------------- tournament (Phase B)
    policies = [p.strip() for p in args.policies.split(",")]
    rows = run_all(demands, CAPACITIES, TIER2_EXPERTS, policies, n_layers,
                   layer_slices, by_pair, kv_weights)
    attach_regret(rows)
    for r in rows:
        r["trace"] = label

    csv_path = out_dir / f"tournament_{label}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = out_dir / f"tournament_{label}.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")

    print("\n=== summary at capacity=32/layer, tier2=0 ===")
    print(f"{'policy':>13s} {'misses':>7s} {'hit_rate':>9s} {'regret':>7s}")
    for r in rows:
        if (r["capacity_experts_per_layer"] == 32
                and r["tier2_experts_per_layer"] == 0):
            print(f"{r['policy']:>13s} {r['misses']:7d} "
                  f"{r['hit_rate']:9.3f} "
                  f"{str(r['regret_vs_belady']):>7s}")


if __name__ == "__main__":
    main()
