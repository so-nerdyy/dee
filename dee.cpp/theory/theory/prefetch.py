"""F. Prefetch bound: budget-constrained speculative selection.

The only legal use of prediction under the exactness contract is a prefetch
HINT: a predicted record may be staged early but the native router remains
authoritative and a prediction error may only waste bandwidth (AGENTS.md,
"Prediction may drive prefetch HINTS only").

Formulation (derived in THEORY.md section 7).  Given the observed set S_l of
layer l at the current token, the legal lookahead is layer l+1 of the same
token (lead = 1 layer-call).  Each candidate record j has a conditional
probability

    p_j(S_l) = 1 - prod_{i in S_l} (1 - P(j in S_{l+1} | i in S_l))

under a conditional-independence approximation over the sources i.  With
record cost P_rec equal for all candidates and a budget of B records per
hint window, the one-shot expected miss reduction is linear:

    f(S) = sum_{j in S} p_j(S_l)          (independent one-shot hints)

a *budgeted maximum coverage with equal item costs*, which greedy top-m by
p_j solves exactly.  If instead a prefetched record also covers correlated
records (staging i makes j free), f becomes probabilistic coverage --
monotone submodular -- and greedy is only a (1 - 1/e) Nemhauser-Wolsey-Fisher
approximation.  Both regimes are computed.

The provable ceiling is the *announceable* mass

    Delta H <= Sigma_corr = recall(S_{l+1} | candidates from S_l, budget = inf)

because prefetch can only convert accesses that were predictable from
already-observed information into earlier hits; compulsory first touches are
immovable and a wrong hint wastes bandwidth without changing execution.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .util import dlog, figure, savefig, write_csv, write_json


def conditional_next_layer(sets: Dict[Tuple[int, int], List[int]],
                           keys: Sequence[Tuple[int, int]],
                           min_support: int = 5):
    """Empirical P(j in S_{l+1} | i in S_l) over adjacent layers, same token.

    Returns (cond, support, n_src) where cond[(i, j)] = P(j | i) over the
    source expert ids within a layer, support[(i, j)] is the joint count and
    n_src[i] the marginal.  Only source experts with >= min_support
    observations are kept (per-layer: the statistic is within-layer pairs).
    """
    n_pair = defaultdict(int)
    n_src = defaultdict(int)
    keys_set = set(keys)
    for (tok, layer) in keys:
        k2 = (tok, layer + 1)
        if k2 not in keys_set:
            continue
        for i in sets[(tok, layer)]:
            n_src[i] += 1
            for j in sets[k2]:
                n_pair[(i, j)] += 1
    cond = {}
    support = {}
    for (i, j), c in n_pair.items():
        if n_src[i] >= min_support:
            cond[(i, j)] = c / n_src[i]
            support[(i, j)] = c
    return cond, support, n_src


def predict_candidates(cond, S_l: Sequence[int], m: int) -> List[Tuple[int, float]]:
    """Candidate (record, probability) for S_{l+1} from S_l, top-m per source.

    p_j(S_l) = 1 - prod_{i in S_l} (1 - P(j | i))  (conditional independence
    across sources; stated approximation).
    """
    per_src: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (i, j), p in cond.items():
        per_src[i].append((j, p))
    agg: Dict[int, float] = {}
    for i in S_l:
        cands = sorted(per_src.get(i, []), key=lambda t: -t[1])[:m]
        for j, p in cands:
            agg[j] = 1.0 - (1.0 - agg.get(j, 0.0)) * (1.0 - p)
    return sorted(agg.items(), key=lambda t: -t[1])


def run(temporal_result=None) -> Dict[str, object]:
    from .constants import D4_RECORD
    from .sources import load_sealed_stream

    dlog("F. prefetch bound")
    sealed = load_sealed_stream()
    rec = int(D4_RECORD.value)
    cond, support, n_src = conditional_next_layer(sealed.sets, sealed.keys,
                                                  min_support=5)

    keys_set = set(sealed.keys)
    # ---- predictor evaluation on every adjacent layer-call pair ---------
    eval_rows: List[dict] = []
    for m in (1, 2, 4, 8):
        tot_target = tot_hit = tot_cand = 0
        n_eval = 0
        for (tok, layer) in sealed.keys:
            k2 = (tok, layer + 1)
            if k2 not in keys_set:
                continue
            S_l = sealed.sets[(tok, layer)]
            S_next = set(sealed.sets[k2])
            cands = predict_candidates(cond, S_l, m)
            cand_set = {j for j, _p in cands}
            hit = len(cand_set & S_next)
            tot_target += len(S_next)
            tot_hit += hit
            tot_cand += len(cand_set)
            n_eval += 1
        recall = tot_hit / max(1, tot_target)
        precision = tot_hit / max(1, tot_cand)
        eval_rows.append({
            "m_per_source": m, "n_eval_pairs": n_eval,
            "n_candidates_total": tot_cand, "n_target_total": tot_target,
            "recall": recall, "precision": precision,
            "announced_records_per_call": tot_cand / max(1, n_eval),
            "bytes_per_call": tot_cand / max(1, n_eval) * rec,
            "bytes_per_call_gib": tot_cand / max(1, n_eval) * rec / (1 << 30),
        })
    write_csv("prefetch_predictor.csv", eval_rows)

    # ---- budget sweep (equal-cost linear objective + submodular variant) -
    n_max = (max(j for (_i, j) in cond) + 1) if cond else 1
    counts = np.zeros(n_max)
    for (i, j), p in cond.items():
        counts[j] += p
    rows = []
    for budget_records in (1, 2, 4, 8, 16, 32, 64, 128):
        S = np.argsort(-counts)[:budget_records]
        gain = float(counts[S].sum()) if counts.size else 0.0
        rows.append({
            "budget_records": budget_records,
            "budget_gib": budget_records * rec / (1 << 30),
            "gain_conditional_mass": gain,
            "gain_per_byte": gain / max(1e-12, budget_records * rec),
            "model": "linear_equal_cost (greedy top-m is optimal)",
        })
    write_csv("prefetch_budget_sweep.csv", rows)

    # ---- submodular coverage regime (with overlap) ----------------------
    top = np.argsort(-counts)[:64] if counts.size else np.zeros(0, dtype=int)
    cov = np.zeros((top.size, counts.size))
    for a_i, a in enumerate(top):
        for (i, j), p in cond.items():
            if i == a:
                cov[a_i, j] = max(cov[a_i, j], p)
    sub_rows = []
    for budget_records in (1, 2, 4, 8, 16, 32):
        S = _greedy_submodular(counts, cov, float(budget_records))
        cover = (1.0 - np.prod(1.0 - np.clip(cov[S, :], 0.0, 1.0), axis=0)
                 if S else np.zeros(counts.size))
        gain = float(np.sum(cover * np.clip(counts, 0, 1)))
        sub_rows.append({
            "budget_records": budget_records,
            "gain_coverage_mass": gain,
            "n_selected": len(S),
            "approximation": "greedy (1-1/e) NWF bound, monotone submodular "
                             "probabilistic coverage",
        })
    write_csv("prefetch_submodular.csv", sub_rows)

    # ---- ceiling -------------------------------------------------------
    sigma_corr = max((r["recall"] for r in eval_rows), default=0.0)
    ceiling = {
        "sigma_corr_max_recall": sigma_corr,
        "delta_H_ceiling": sigma_corr,
        "predictor": {
            "family": "empirical P(j in S_{l+1} | i in S_l), min_support=5",
            "lead_calls": 1,
            "recall_at_m": {str(r["m_per_source"]): r["recall"] for r in eval_rows},
            "precision_at_m": {str(r["m_per_source"]): r["precision"] for r in eval_rows},
        },
        "prior_art_gate": {
            "required": "lead >= 2 AND precision >= 0.75 to clear the idle-gap "
                        "bound (AGENTS.md research assimilation R1-R11)",
            "measured_lead": 1,
            "measured_precision_max": max((r["precision"] for r in eval_rows),
                                          default=0.0),
            "passes_gate": False,
            "verdict": "k=1 next-layer prediction: measurable reuse exists but "
                       "does not clear the prior-art gate; consistent with the "
                       "in-repo verdict that k=1 mechanisms are worth 0 s on the "
                       "sealed bank.",
        },
        "statement": "Delta H <= Sigma_corr: prefetch can only convert accesses "
                     "predictable from already-observed information into earlier "
                     "hits. Compulsory first touches (2,364 records on the sealed "
                     "window) are immovable, and a wrong hint wastes bandwidth "
                     "without changing execution (exactness contract).",
    }
    write_json("prefetch_ceiling.json", ceiling)
    _plots(rows, sub_rows, eval_rows)
    out = {"linear": rows, "submodular": sub_rows, "ceiling": ceiling,
           "predictor": eval_rows}
    dlog("   prefetch ceiling Delta H <= %.3f (recall at m=4: %.3f, precision: %.3f)"
         % (sigma_corr, eval_rows[2]["recall"] if len(eval_rows) > 2 else 0.0,
            eval_rows[2]["precision"] if len(eval_rows) > 2 else 0.0))
    return out


def _greedy_submodular(p: np.ndarray, cov: np.ndarray,
                       budget: float) -> List[int]:
    """Greedy by marginal gain per byte for a monotone submodular coverage."""
    S: List[int] = []
    spend = 0.0
    base = np.zeros(cov.shape[1])
    remaining = set(range(cov.shape[0]))
    while remaining and spend + 1.0 <= budget + 1e-12:
        best, best_g = None, -1.0
        for i in list(remaining):
            new_cover = np.maximum(base, cov[i, :])
            g = float(np.sum((new_cover - base) * np.clip(p, 0, 1)))
            if g > best_g:
                best, best_g = i, g
        if best is None or best_g <= 0.0:
            break
        base = np.maximum(base, cov[best, :])
        spend += 1.0
        S.append(best)
        remaining.discard(best)
    return S


def _plots(rows, sub_rows, eval_rows) -> None:
    fig, ax = figure("prefetch_budget", (7.2, 4.4))
    ax.plot([r["budget_gib"] for r in rows],
            [100 * r["gain_conditional_mass"] for r in rows], "o-",
            label="linear equal-cost (top-m by p)")
    ax.set_xscale("log")
    ax.set_xlabel("prefetch budget (GiB per hint window)")
    ax.set_ylabel("expected hit-rate gain (conditional mass, pp)")
    ax.set_title("Prefetch: miss-rate reduction vs bandwidth budget (lead = 1 layer-call)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "prefetch_budget.png")

    fig, ax = figure("prefetch_predictor", (7.2, 4.4))
    ax.plot([r["m_per_source"] for r in eval_rows],
            [100 * r["recall"] for r in eval_rows], "o-", label="recall")
    ax.plot([r["m_per_source"] for r in eval_rows],
            [100 * r["precision"] for r in eval_rows], "s--", label="precision")
    ax.axhline(75, color="r", ls=":", lw=1.0)
    ax.annotate("prior-art precision gate 0.75", (1.05, 77), color="r", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m candidates per source expert")
    ax.set_ylabel("%")
    ax.set_title("Next-layer predictor quality (empirical P(j|i), sealed trace)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "prefetch_predictor.png")

    peak = max((r["gain_conditional_mass"] for r in rows), default=1.0)
    fig, ax = figure("prefetch_ceiling", (7.2, 4.4))
    ax.plot([r["budget_records"] for r in rows],
            [100 * r["gain_conditional_mass"] / max(1e-12, peak) for r in rows],
            "o-")
    ax.axhline(100.0, color="r", ls="--", lw=1.0)
    ax.annotate("Sigma_corr ceiling (Delta H <= Sigma_corr)", (1, 96),
                color="r", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("budget (records per hint window)")
    ax.set_ylabel("fraction of the reuse ceiling captured (%)")
    ax.set_title("Budget-limited prefetch vs the provable ceiling")
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "prefetch_ceiling.png")
