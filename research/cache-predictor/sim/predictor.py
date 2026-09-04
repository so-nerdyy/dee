"""Fate-style cross-layer predictor (offline, strictly causal).

Goal: for each upcoming layer demand, RANK candidate experts before the
router runs, using only routing information available before the target
demand. The OFFICIAL router remains authoritative; predictions are used
only for prefetch/keep simulation.

Prediction targets are LAYER BUNDLES (all experts routed at one layer),
because dee schedules layer-by-layer.

Ranked candidate design: each predictor scores experts; the ranked list
is truncated at k for recall@k (6/8/12/16). Padding is intrinsic
(popularity tail), so recall@k is monotone and deterministic.

Feature families (Phase C):
  freq_expert      per-layer popularity from prior steps only
  persist_layer    previous token's bundle at the same layer (X1)
  cross_union      union of (step-1, L) and (step, L-1)          (X2)
  combo            X2 + per-layer popularity                      (X4)

Causality: (step-1, L) and (step, L-1) always precede (step, L) in
execution order; popularity counters only aggregate bundles with
step < current step at prediction time.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trace import Demand  # noqa: E402


@dataclass(frozen=True)
class LayerBundle:
    step: int
    layer: int
    keys: frozenset[int]


def to_layer_bundles(demands: list[Demand]) -> list[LayerBundle]:
    bundles: dict[tuple[int, int], set[int]] = defaultdict(set)
    for d in demands:
        bundles[(d.step, d.layer)].add(d.scope)
    return [LayerBundle(s, l, frozenset(k))
            for (s, l), k in sorted(bundles.items())]


class CausalContext:
    """Incremental causal state shared by predictors.

    After observe(idx) for all j < idx, exposes:
      prev_token[(layer)]  -> bundle keys of (step-1, layer)
      cur_prev_layer       -> bundle keys of (step, layer-1)
      pop[layer]           -> Counter of expert demand at that layer
                              over steps < current step
    """

    def __init__(self, bundles: list[LayerBundle]):
        self.bundles = bundles
        self.by_pair = {(b.step, b.layer): b.keys for b in bundles}
        self.pop: dict[int, Counter] = defaultdict(Counter)
        self._observed_step = -1
        self._observed_layer = -1

    def advance_to(self, idx: int) -> None:
        """Observe every bundle strictly before bundles[idx]."""
        b = self.bundles[idx]
        # observe all bundles with (step, layer) < (b.step, b.layer)
        for j in range(max(0, self._progress), idx):
            bj = self.bundles[j]
            if (bj.step, bj.layer) < (b.step, b.layer):
                self.pop[bj.layer].update(bj.keys)
        self._progress = idx

    _progress = 0

    def features(self, idx: int) -> dict:
        b = self.bundles[idx]
        prev_tok = self.by_pair.get((b.step - 1, b.layer), frozenset())
        cur_prev = self.by_pair.get((b.step, b.layer - 1), frozenset())
        if (b.step, b.layer - 1) not in self.by_pair:
            cur_prev = frozenset()
        pop = self.pop[b.layer]  # counts from steps < b.step only
        return {
            "step": b.step, "layer": b.layer,
            "prev_token": prev_tok, "cur_prev_layer": cur_prev,
            "pop": pop, "true": b.keys,
        }


PREDICTORS: dict[str, object] = {}


def register(name):
    def deco(fn):
        PREDICTORS[name] = fn
        return fn
    return deco


@register("freq_expert")
def _freq(f) -> list[tuple[int, float]]:
    scored = [(k, float(c)) for k, c in f["pop"].most_common(24)]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored


@register("persist_layer")
def _persist(f) -> list[tuple[int, float]]:
    scored: dict[int, float] = {}
    for k in f["prev_token"]:
        scored[k] = scored.get(k, 0.0) + 2.0
    for k, c in f["pop"].most_common(24):
        scored[k] = scored.get(k, 0.0) + float(c) * 1e-3  # tie-break
    out = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return out


@register("cross_union")
def _cross(f) -> list[tuple[int, float]]:
    scored: dict[int, float] = {}
    for k in f["prev_token"]:
        scored[k] = scored.get(k, 0.0) + 2.0
    for k in f["cur_prev_layer"]:
        scored[k] = scored.get(k, 0.0) + 1.0
    for k, c in f["pop"].most_common(24):
        scored[k] = scored.get(k, 0.0) + float(c) * 1e-3
    out = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return out


@register("combo")
def _combo(f) -> list[tuple[int, float]]:
    scored: dict[int, float] = {}
    for k in f["prev_token"]:
        scored[k] = scored.get(k, 0.0) + 2.0
    for k in f["cur_prev_layer"]:
        scored[k] = scored.get(k, 0.0) + 1.0
    for k, c in f["pop"].most_common(24):
        scored[k] = scored.get(k, 0.0) + float(c) * 1e-2
    out = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return out


def make_ranked_predictor(name: str, bundles: list[LayerBundle]):
    """Return predict(idx) -> ranked [(key, score)] for bundles[idx]."""
    fn = PREDICTORS[name]
    ctx = CausalContext(bundles)

    def predict(idx: int) -> list[tuple[int, float]]:
        ctx.advance_to(idx)
        return fn(ctx.features(idx))

    return predict


def evaluate_ranked_predictor(
    name: str,
    bundles: list[LayerBundle],
    candidate_sizes: tuple[int, ...] = (6, 8, 12, 16),
    record_bytes: int = 13_369_344,
) -> dict:
    """Recall / precision / useful-vs-wasted prefetch statistics.

    All numbers are per decode-layer-bundle (step >= 1). step 0 is the
    prefill token: nothing precedes it, so no prediction is possible
    without leakage.
    """
    predict = make_ranked_predictor(name, bundles)
    by_pair = {(b.step, b.layer): b.keys for b in bundles}
    ks = candidate_sizes
    recall_sum = {k: 0.0 for k in ks}
    full_bundle_hits = {k: 0 for k in ks}
    precision_sum = {k: 0.0 for k in ks}
    useful = {k: 0 for k in ks}   # predicted ∩ true (prefetched and used)
    wasted = {k: 0 for k in ks}   # predicted ∖ true (prefetched, never used)
    total = 0
    total_true = 0
    lead_hist: Counter = Counter()

    for idx, b in enumerate(bundles):
        if b.step == 0:
            continue
        ranked = predict(idx)
        total += 1
        total_true += len(b.keys)
        true = b.keys
        for k in ks:
            top = [key for key, _ in ranked[:k]]
            inter = true.intersection(top)
            recall_sum[k] += len(inter) / len(true)
            precision_sum[k] += len(inter) / k
            if inter == true:
                full_bundle_hits[k] += 1
            useful[k] += len(inter)
            wasted[k] += len(top) - len(inter)

        # lead distance: for each true expert, how many layers earlier
        # in the SAME step was it first ranked as a candidate?
        # (cross-token predictions have lead = 1 token = 43 layers.)
        for key in true:
            lead = None
            for l2 in range(b.layer):
                cand = by_pair.get((b.step, l2), frozenset())
                # that earlier bundle's predictor input isn't stored;
                # approximate same-step lead by membership in any
                # earlier same-step predictor source set:
                if key in cand:
                    lead = b.layer - l2
                    break
            if lead is None:
                lead = 43  # cross-token lead (1 step ≈ 43 layers)
            lead_hist[lead] += 1

    def pct(x, d):
        return x / d if d else 0.0

    out: dict = {
        "predictor": name,
        "bundles_evaluated": total,
        "true_experts_total": total_true,
        "mean_true_per_bundle": total_true / total if total else 0.0,
        "lead_layer_hist": {str(k): v for k, v in sorted(lead_hist.items())},
        "mean_lead_layers": (sum(k * v for k, v in lead_hist.items())
                             / sum(lead_hist.values())) if lead_hist else 0.0,
    }
    for k in ks:
        out[f"recall@{k}"] = recall_sum[k] / total
        out[f"full_bundle_recall@{k}"] = full_bundle_hits[k] / total
        out[f"precision@{k}"] = precision_sum[k] / total
        out[f"useful_prefetch_bytes@{k}"] = useful[k] * record_bytes
        out[f"wasted_prefetch_bytes@{k}"] = wasted[k] * record_bytes
        out[f"useful_prefetch_pct@{k}"] = pct(useful[k], useful[k] + wasted[k])
        out[f"wasted_prefetch_pct@{k}"] = pct(wasted[k], useful[k] + wasted[k])
    return out
