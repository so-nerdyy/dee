"""Phase D: cost-aware keep-value policy and even/odd-step tuning.

KeepValue(e) = P(reuse soon | history) * miss_cost(e) / resident_bytes(e)
               - pollution_cost(e)

Features: predictor rank (causal), access frequency, staleness
(1/(1+age)), pollution (resident but never reused). miss_cost and
resident_bytes are uniform across experts today, so they scale out of
the victim ranking but stay in the formula for future variable-size
codecs (byte-aware via set_bytes).

Coefficients are tuned on even steps and validated on odd steps of the
SAME trace; the report treats them as trace-local (no cross-trace
generalization claim).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policies import OnlinePolicy  # noqa: E402
from engine import _reindex  # noqa: E402


class KeepValuePolicy(OnlinePolicy):
    """Evict the resident expert with the LOWEST keep value.

    keep(k) = w_pred * p_reuse(k) + w_freq * f(k) + w_stale * 1/(1+age)
              - w_pollute * never_reused(k)

    p_reuse(k): 1/(rank+1) from the causal ranked candidate list for
    the upcoming bundle at this layer; 0 when absent.
    """

    name = "keepvalue"

    def __init__(self, capacity_bytes: int, default_nbytes: int = 1,
                 ranked_fn=None, weights=(1.0, 1.0, 1.0, 0.5)):
        super().__init__(capacity_bytes)
        self._default_nbytes = default_nbytes
        self.ranked_fn = ranked_fn  # global_tick -> {key: p_reuse}
        self.w_pred, self.w_freq, self.w_stale, self.w_pollute = weights
        self.freq: defaultdict[int, int] = defaultdict(int)
        self.last_used: dict[int, int] = {}
        self._tick = 0
        self._p: dict[int, float] = {}
        self._never_reused: set[int] = set()

    def note_tick(self, tick: int) -> None:
        self._tick = tick
        if self.ranked_fn is not None:
            ranked = self.ranked_fn(tick)
            self._p = {k: 1.0 / (r + 1) for r, (k, _) in enumerate(ranked)}
        else:
            self._p = {}

    def on_hit(self, key, tick):
        self.freq[key] += 1
        self.last_used[key] = tick
        self._never_reused.discard(key)

    def _insert(self, key, tick):
        self.freq[key] = 0
        self.last_used[key] = tick
        self._never_reused.add(key)

    def _remove_meta(self, key):
        self.freq.pop(key, None)
        self.last_used.pop(key, None)
        self._never_reused.discard(key)

    def pick_victim(self):
        freq = self.freq
        last_used = self.last_used
        tick = self._tick
        p = self._p

        def keep(k):
            age = tick - last_used[k]
            pollute = 1.0 if k in self._never_reused else 0.0
            return (self.w_pred * p.get(k, 0.0)
                    + self.w_freq * freq[k]
                    + self.w_stale * (1.0 / (1.0 + age))
                    - self.w_pollute * pollute)

        return min(self.resident, key=keep)


def build_ranked_fn(layer_demands: list, all_demands_by_pair: dict,
                    layer: int):
    """Causal ranked-list provider for one layer's Engine.

    layer_demands: the layer's demand slice (ordered, with .step/.seq).
    all_demands_by_pair: (step, layer) -> frozenset keys, from the FULL
        trace (used read-only for (step-1, L) lookups; popularity is
        aggregated only over bundles with step < target step).

    At ranked_fn(global_tick) the provider returns the ranked list for
    the bundle CONTAINING global_tick (the last bundle whose first
    position <= tick). Candidates are scored ONLY from strictly
    earlier steps: the (target-1, layer) bundle (cross-token
    recurrence, executed in the previous forward step) plus per-layer
    popularity aggregated over steps < target. No within-bundle or
    future information is used.
    """
    # bundle start positions within this layer's slice (0-based,
    # slice-local indices - the engine calls ranked_fn with these)
    starts: list[tuple[int, int]] = []  # (step, first_position)
    seen: set[int] = set()
    for pos, d in enumerate(layer_demands):
        if d.step not in seen:
            seen.add(d.step)
            starts.append((d.step, pos))
    starts.sort()

    def target_step_at(global_tick: int) -> int | None:
        """The step of the bundle CONTAINING global_tick (last start
        <= tick). Candidates for this bundle are scored strictly from
        EARLIER steps, so no within-bundle future leaks in."""
        target = None
        for step, first_pos in starts:
            if first_pos <= global_tick:
                target = step
            else:
                break
        return target

    def ranked(global_tick: int) -> list[tuple[int, float]]:
        target_step = target_step_at(global_tick)
        if target_step is None:
            return []
        scored: dict[int, float] = {}
        # cross-token recurrence: strictly EARLIER token's bundle only
        prev = all_demands_by_pair.get((target_step - 1, layer),
                                       frozenset())
        for k in prev:
            scored[k] = scored.get(k, 0.0) + 2.0
        pop: Counter = Counter()
        for (s, l), keys in all_demands_by_pair.items():
            if l == layer and s < target_step:
                pop.update(keys)
        for k, c in pop.most_common(12):
            scored[k] = scored.get(k, 0.0) + c * 1e-2
        return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))

    return ranked


def evaluate_keepvalue(layer_slices, n_layers, weights, record_bytes,
                       by_pair, capacity_experts: int = 8):
    """Run the layered engine with KeepValue policies.

    layer_slices: dict layer -> ordered demand list for that layer.
    capacity_experts: tier-1 capacity PER LAYER (policy matters most
    when capacity is scarce, so tuning defaults to cap-8).
    """
    from engine import Engine

    engines: dict[int, Engine] = {}
    for layer, ds in layer_slices.items():
        # CRITICAL: reindex to slice-local positions BEFORE building
        # the ranked provider, so ranked_fn(tick) resolves to the
        # bundle containing `tick`. Feeding global trace seqs here
        # made the provider return the LAST step's candidates for all
        # late accesses (a future-info leak) in an earlier revision.
        ds = _reindex(ds)
        ranked_fn = build_ranked_fn(ds, by_pair, layer)
        pol = KeepValuePolicy(capacity_experts * record_bytes,
                              record_bytes,
                              ranked_fn=ranked_fn, weights=weights)
        eng = Engine(pol, record_bytes=record_bytes)
        eng.run(ds)
        engines[layer] = eng

    from engine import Metrics, TwoTierStats
    m = Metrics()
    for eng in engines.values():
        em = eng.m
        m.accesses += em.accesses
        m.hits += em.hits
        m.misses += em.misses
        m.compulsory_misses += em.compulsory_misses
        m.capacity_misses += em.capacity_misses
        m.evictions += em.evictions
        m.insertions += em.insertions
        m.miss_bytes += em.miss_bytes
        m.ssd_bytes += em.ssd_bytes
        m.tier2_h2d_bytes += em.tier2_h2d_bytes
    return m


def tune_coefficients(layer_slices, n_layers, trials, record_bytes,
                      by_pair, capacity_experts: int = 8):
    """Grid over weights on the TRAIN slice only."""
    results = {}
    best, best_misses = None, None
    for w in trials:
        m = evaluate_keepvalue(layer_slices, n_layers, w, record_bytes,
                               by_pair, capacity_experts=capacity_experts)
        results[str(w)] = {"weights": list(w), "misses": m.misses,
                           "hits": m.hits, "accesses": m.accesses}
        if best_misses is None or m.misses < best_misses:
            best_misses, best = m.misses, w
    return best, results


def split_steps(demands, train_steps, valid_steps):
    ts, vs = set(train_steps), set(valid_steps)
    train = [d for d in demands if d.step in ts]
    valid = [d for d in demands if d.step in vs]
    return train, valid
