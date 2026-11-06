"""C. Temporal structure: the prefetch-relevant statistics.

Emitted statistics, all with sample counts:

  * empirical reaccess survival  P(expert reaccessed | delta-t) over
    (layer-call) and (decode-token) distance lags, per trace;
  * within-token cross-layer conditional  P(j in S_{l+1} | i in S_l),
    reported only for pairs with sufficient support (n >= MIN_PAIR_SUPPORT);
  * decode-step self-correlation  P(expert at token t+1 | at token t) per layer;
  * the effective IRM-vs-correlated hit-rate delta (real stream vs shuffled).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .cache import lru_replay, shuffled_stream
from .util import bootstrap_ci, dlog, figure, rng, savefig, write_csv, write_json

MIN_PAIR_SUPPORT = 5         # minimum observed source-expert uses to report a pair
MIN_SELF_SUPPORT = 5


def reaccess_survival(stream: Sequence[Tuple[int, int]], max_lag: int = 200):
    """P(record reaccessed within lag L cache-requests | it was just accessed).

    For each access, the lag to the record's next access (right-censored for
    the last occurrence) is the survival sample.  Returns lag-binned
    Kaplan-Meier-style survival 1 - F(lag) with counts at risk.
    """
    nxt: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, key in enumerate(stream):
        nxt[key].append(i)
    lags = []
    censored = 0
    n = len(stream)
    for key, occ in nxt.items():
        for a, b in zip(occ[:-1], occ[1:]):
            lags.append(b - a)
        censored += 1                  # last occurrence right-censored
    lags = np.asarray(lags, dtype=float)
    rows = []
    at_risk = lags.size
    for L in sorted(set([1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100, 128, 200, max_lag])):
        if L > max_lag:
            continue
        surv = float((lags > L).sum() / at_risk) if at_risk else float("nan")
        rows.append({"lag_requests": L, "survival": surv,
                     "n_intervals": int(at_risk), "n_censored_last_use": int(censored),
                     "mean_lag": float(lags.mean()) if lags.size else float("nan")})
    return rows, lags


def cross_layer_conditional(sets: Dict[Tuple[int, int], List[int]], keys,
                            n_layers: int) -> List[dict]:
    """P(j in S_{l+1} | i in S_l) within the same token/step, adjacent layers.

    Only pairs (i at layer l, j at layer l+1) with n(i at l) >= MIN_PAIR_SUPPORT
    are reported; the row carries both the observed P(j|i) and the baseline
    marginal P(j at l+1).
    """
    by_token: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
    for (tok, layer) in keys:
        by_token[tok][layer] = sets[(tok, layer)]
    src_cnt: Counter = Counter()
    pair_cnt: Counter = Counter()
    dst_cnt: Counter = Counter()
    n_tok = 0
    for tok, layers in by_token.items():
        n_tok += 1
        for layer in layers:
            if layer + 1 in layers:
                for i in sets[(tok, layer)]:
                    src_cnt[(layer, i)] += 1
                    for j in sets[(tok, layer + 1)]:
                        pair_cnt[((layer, i), j)] += 1
                for j in sets[(tok, layer + 1)]:
                    dst_cnt[(layer + 1, j)] += 1
    rows = []
    for ((layer, i), j), c in pair_cnt.items():
        n_src = src_cnt[(layer, i)]
        if n_src < MIN_PAIR_SUPPORT:
            continue
        p_ji = c / n_src
        n_dst_total = sum(v for (l, e), v in dst_cnt.items() if l == layer + 1)
        p_j = dst_cnt[(layer + 1, j)] / max(1, n_dst_total)
        rows.append({
            "src_layer": layer, "src_expert": i, "dst_layer": layer + 1, "dst_expert": j,
            "n_src": n_src, "n_pair": c, "p_j_given_i": p_ji, "p_j_base": p_j,
            "lift": p_ji / p_j if p_j > 0 else float("nan"),
        })
    rows.sort(key=lambda r: -r["n_pair"])
    return rows


def decode_self_correlation(stream_by_token: Dict[int, List[Tuple[int, int]]],
                            max_lag: int = 12) -> List[dict]:
    """P(expert at token t+d | expert at token t), same layer, decode steps.

    For each layer and lag d the estimator is

        sum_i |S_{t,i} cap S_{t+d,i}| / sum_i |S_{t,i}|

    over all decode tokens t where both t and t+d exist.  Under an IRM with
    within-layer popularity p_i the baseline is sum_i p_i^2 (Simpson).
    """
    toks = sorted(stream_by_token)
    rows = []
    layers = sorted({l for v in stream_by_token.values() for (l, _e) in v})
    for layer in layers:
        sets = {}
        for t in toks:
            sets[t] = {e for (l, e) in stream_by_token[t] if l == layer}
        all_e = sorted({e for s in sets.values() for e in s})
        cnt = Counter(e for t in toks for e in sets[t])
        tot = sum(cnt.values())
        p = {e: cnt[e] / tot for e in all_e} if tot else {}
        irm_base = sum(v * v for v in p.values())
        for d in range(1, max_lag + 1):
            pairs = [(t, t + d) for t in toks if (t + d) in sets]
            if not pairs:
                continue
            inter = 0
            union_ref = 0
            for t, t2 in pairs:
                inter += len(sets[t] & sets[t2])
                union_ref += len(sets[t])
            if union_ref == 0:
                continue
            est = inter / union_ref
            rows.append({
                "layer": layer, "lag_tokens": d, "n_pairs": len(pairs),
                "n_ref_experts": union_ref, "n_overlap": inter,
                "p_self": est, "irm_baseline": irm_base,
                "lift_vs_irm": est / irm_base if irm_base > 0 else float("nan"),
            })
    return rows


def reaccess_by_phase(rows: List[dict]) -> List[dict]:
    return rows


def run() -> Dict[str, object]:
    from .sources import load_ornith_streams, load_sealed_stream

    dlog("C. temporal structure")
    out: Dict[str, object] = {}

    surv_rows: List[dict] = []
    pair_rows: List[dict] = []
    self_rows: List[dict] = []
    delta_rows: List[dict] = []
    pair_all_rows: List[dict] = []      # all pairs incl. under-supported (for n reporting)

    sealed = load_sealed_stream()
    traces = [("sealed_dsv4", sealed)] + [(f"ornith_{k}", v)
                                          for k, v in load_ornith_streams().items()]

    for name, stream in traces:
        seq = stream.stream()
        rows, lags = reaccess_survival(seq, max_lag=min(400, max(20, len(seq) // 4)))
        ci = bootstrap_ci(lags.tolist() if lags.size else [0.0])
        for r in rows:
            r.update({"trace": name, "mean_lag_lo": ci["lo"], "mean_lag_hi": ci["hi"]})
            surv_rows.append(r)
        pairs = cross_layer_conditional(stream.sets, stream.keys,
                                        stream.n_layers)
        for r in pairs:
            r["trace"] = name
            pair_rows.append(r)
        by_tok: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for (tok, layer) in stream.keys:
            by_tok[tok].extend((layer, e) for e in stream.sets[(tok, layer)])
        for r in decode_self_correlation(by_tok):
            r["trace"] = name
            self_rows.append(r)

        # IRM-vs-correlated hit-rate delta
        from .cache import popularity_vector
        for gib in (8, 16, 32):
            cap = max(1, int(gib * (1 << 30) // 13369344))
            if cap > len(seq) * 2:
                continue
            real = lru_replay(seq, cap)
            shuf = lru_replay(shuffled_stream(seq), cap)
            delta_rows.append({
                "trace": name, "budget_gib": gib, "capacity_records": cap,
                "real_hit": real["hit_rate"], "shuffled_hit": shuf["hit_rate"],
                "delta_pp": 100.0 * (real["hit_rate"] - shuf["hit_rate"]),
                "n_requests": len(seq),
            })

    write_csv("temporal_survival.csv", surv_rows)
    write_csv("temporal_crosslayer.csv", pair_rows)
    write_csv("temporal_selfcorr.csv", self_rows)
    write_csv("temporal_irm_delta.csv", delta_rows)

    # Aggregated cross-layer bounds (correlation matrix summary): total
    # pair opportunities, supported pairs, and lift distribution.
    pair_summary = []
    for name, stream in traces:
        rows_n = [r for r in pair_rows if r["trace"] == name]
        lifts = [r["lift"] for r in rows_n if np.isfinite(r["lift"])]
        pair_summary.append({
            "trace": name,
            "pairs_reported": len(rows_n),
            "min_pair_support": MIN_PAIR_SUPPORT,
            "n_src_experts": len({(r["src_layer"], r["src_expert"]) for r in rows_n}),
            "mean_lift": float(np.mean(lifts)) if lifts else float("nan"),
            "median_lift": float(np.median(lifts)) if lifts else float("nan"),
            "max_lift": float(np.max(lifts)) if lifts else float("nan"),
            "sum_p_j_given_i": float(np.sum([r["p_j_given_i"] for r in rows_n])) if rows_n else 0.0,
        })

    top_pairs = pair_rows[:25]
    write_json("temporal_summary.json", {
        "cross_layer_reported_pairs": len(pair_rows),
        "cross_layer_pair_summary": pair_summary,
        "cross_layer_top": top_pairs,
        "min_pair_support": MIN_PAIR_SUPPORT,
        "irm_delta": delta_rows,
    })

    _plots(surv_rows, self_rows, delta_rows)
    out = {"survival": surv_rows, "pairs": pair_rows, "selfcorr": self_rows,
           "irm_delta": delta_rows, "pair_summary": pair_summary}
    dlog("   survival + correlations ->", len(pair_rows), "supported cross-layer pairs")
    return out


def _plots(surv_rows, self_rows, delta_rows) -> None:
    fig, ax = figure("temporal_survival", (7.2, 4.4))
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in surv_rows:
        by[r["trace"]].append(r)
    for name, rs in by.items():
        rs = sorted(rs, key=lambda r: r["lag_requests"])
        ax.plot([r["lag_requests"] for r in rs], [r["survival"] for r in rs],
                "o-", ms=3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lag since access (cache requests)")
    ax.set_ylabel("P(reaccess after lag)")
    ax.set_title("Reaccess survival (reuse-lag tail) by trace")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "temporal_survival.png")

    fig, ax = figure("temporal_selfcorr", (7.2, 4.4))
    by = defaultdict(list)
    for r in self_rows:
        by[r["trace"]].append(r)
    for name, rs in by.items():
        agg = defaultdict(list)
        for r in rs:
            agg[r["lag_tokens"]].append(r["p_self"])
        xs = sorted(agg)
        ys = [float(np.mean(agg[x])) for x in xs]
        base = float(np.mean([r["irm_baseline"] for r in rs]))
        ax.plot(xs, ys, "o-", ms=3, label=f"{name} P(same expert | lag d)")
        ax.axhline(base, ls=":", lw=0.8, alpha=0.6)
        ax.annotate(f"{name} IRM baseline {base:.3f}", (xs[0], base), fontsize=7)
    ax.set_xlabel("decode-token lag d")
    ax.set_ylabel("P(expert at t+d | expert at t)")
    ax.set_title("Decode-step self-correlation vs IRM baseline (sum p_i^2)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    savefig(fig, "temporal_selfcorr.png")

    fig, ax = figure("temporal_crosslayer_top", (7.2, 4.4))
    import csv as _csv
    top = sorted(self_rows, key=lambda r: -r["n_overlap"])[:20]
    ax.bar(range(len(top)), [r["p_self"] for r in top])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"L{r['layer']} d{r['lag_tokens']}" for r in top],
                       rotation=90, fontsize=6)
    ax.set_ylabel("P(same expert)")
    ax.set_title("Top-20 layer/lag self-correlations by sample count")
    ax.grid(True, alpha=0.3, axis="y")
    savefig(fig, "temporal_crosslayer_top.png")
