"""Unit tests for the offline cache/predictor research simulator.

Run:  python -m pytest research/cache-predictor/tests -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from sim.trace import load_default_trace, load_trace, trace_root  # noqa: E402
from sim.policies import make_policy, LRU, LFU, ARCLike, Belady  # noqa: E402
from sim.engine import Engine, LayeredEngine  # noqa: E402
from sim.predictor import (  # noqa: E402
    to_layer_bundles,
    make_ranked_predictor,
    evaluate_ranked_predictor,
)
from sim.keepvalue import split_steps, tune_coefficients  # noqa: E402
from sim.prefetch import PrefetchEngine, PrefetchPolicy  # noqa: E402

RECORD_BYTES = 13_369_344
TRACE_AVAILABLE = (trace_root()
                   / "v65-terminal-fetch-20260903T0202Z"
                   / "routed_experts.jsonl").is_file()


def synth_engine(name, seq, cap, rb=100):
    pol = make_policy(name, cap * rb, rb,
                      future=seq if name == "belady" else None)
    eng = Engine(pol, record_bytes=rb)
    for i, k in enumerate(seq):
        eng.access_demand(k, i, rb)
        assert len(pol.resident) <= cap, f"{name} over capacity at {i}"
    return eng


# --------------------------------------------------------------------------
# policies on synthetic sequences
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seq", [
    [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1, 1, 2, 7, 1, 2, 8, 1, 2, 9, 1, 2],
    [i % 7 for i in range(50)],
    [1] * 20 + [2, 3, 4, 5, 6, 7, 8] * 3,
    [1, 2, 3] * 10,
])
def test_belady_is_optimal_on_synthetics(seq):
    hits = {p: synth_engine(p, seq, 3).m.hits
            for p in ("lru", "lfu", "arc", "recency_freq", "least_stale",
                      "belady")}
    for p, h in hits.items():
        assert h <= hits["belady"], f"{p} beat OPT: {h} > {hits['belady']}"


def test_lru_evicts_lru_order():
    seq = [1, 2, 3, 1, 4]
    # cap 3: after 1,2,3 + hit 1 -> LRU order [2,3,1]; 4 evicts 2.
    eng = synth_engine("lru", seq, 3)
    assert eng.m.hits == 1  # the re-access of 1
    pol = eng.policy
    assert 4 in pol.resident and 2 not in pol.resident
    assert 1 in pol.resident and 3 in pol.resident


def test_eviction_count_matches_insertions_minus_resident():
    seq = [i % 10 for i in range(100)]
    eng = synth_engine("lru", seq, 4)
    m = eng.m
    assert m.insertions - m.evictions == len(eng.policy.resident)
    assert m.compulsory_misses == 10
    assert m.capacity_misses == m.misses - 10


def test_byte_capacity_respected_variable_sizes():
    pol = LRU(100, default_nbytes=1)
    pol.set_bytes({1: 60, 2: 60})  # oversize records
    eng = Engine(pol, record_bytes=1)
    eng.access_demand(1, 0, 60)
    eng.access_demand(2, 1, 60)
    # 60+60 > 100: exactly one must be evicted
    assert len(pol.resident) == 1
    assert pol.resident_bytes <= 100


# --------------------------------------------------------------------------
# engine metrics
# --------------------------------------------------------------------------

def test_useful_and_wasted_residency_hand_case():
    # Semantics: useful = residency between two consecutive DEMANDS of
    # the key; wasted = insert->first-demand and last-demand->evict
    # tails (residency that served no demand).
    # cap 2, LRU: t0:1(ins) t1:2(ins) t2:3(evict 1) t3:1(evict 2, ins)
    seq = [1, 2, 3, 1]
    eng = synth_engine("lru", seq, 2)
    m = eng.m
    # key1: insert t0, evicted t2, never re-demanded -> wasted 2
    # key2: insert t1, evicted t3, never re-demanded -> wasted 2
    assert m.useful_residency_ticks == 0
    assert m.wasted_residency_ticks == 4
    assert m.hits == 0 and m.misses == 4
    assert m.compulsory_misses == 3 and m.capacity_misses == 1


def test_useful_residency_counts_between_repeated_demands():
    # cap 2, LRU trace:
    #  t0:1 ins | t1:2 ins | t2:3 evict1(wasted 2) | t3:1 evict2(wasted 2)
    #  t4:1 HIT (useful 1, t3->t4) | t5:2 evict3(wasted 3, t2->t5)
    seq = [1, 2, 3, 1, 1, 2]
    eng = synth_engine("lru", seq, 2)
    m = eng.m
    assert m.hits == 1
    assert m.useful_residency_ticks == 1
    assert m.wasted_residency_ticks == 7
    assert m.compulsory_misses == 3
    assert m.capacity_misses == 2  # t3 re-miss of 1, t5 re-miss of 2


def test_belady_per_layer_future_matches_opt():
    """Per-layer Belady with slice-local next-use chains must still
    dominate online policies on the same layered sequence."""
    class D:
        def __init__(self, step, layer, expert):
            self.step, self.layer, self.expert = step, layer, expert
            self.scope = layer * 100_000 + expert
            self.nbytes = 100
            self.seq = 0
    seq = [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 2),
           (2, 0, 1), (2, 1, 4), (3, 0, 1), (3, 1, 4)]
    demands = [D(*t) for t in seq]
    for i, d in enumerate(demands):
        d.seq = i
    futures = {L: [d.scope for d in demands if d.layer == L]
               for L in (0, 1)}
    le = LayeredEngine(
        lambda L: Belady(2 * 100, 100, future=futures[L]),
        n_layers=2, record_bytes=100)
    m = le.run(demands)
    # layer 0: 1,3,1,1 -> hits 2 (t2, t3); layer 1: 2,2,4,4 -> hits 2
    assert m.hits == 4


def test_layered_engine_equals_manual_per_layer_sum():
    seq = [(0, 0, 1), (0, 1, 2), (1, 0, 1), (1, 1, 3), (2, 0, 4), (2, 1, 2)]

    class D:
        def __init__(self, step, layer, expert):
            self.step, self.layer, self.expert = step, layer, expert
            self.scope = layer * 100_000 + expert
            self.seq = 0
            self.nbytes = 100

    demands = [D(*t) for t in seq]
    for i, d in enumerate(demands):
        d.seq = i
    le = LayeredEngine(lambda layer: LRU(2 * 100, 100),
                       n_layers=2, record_bytes=100)
    m = le.run(demands)
    assert m.accesses == 6
    # per layer: L0 sees 1,1,4 (1 hit); L1 sees 2,3,2 (1 hit)
    assert m.hits == 2


# --------------------------------------------------------------------------
# predictor
# --------------------------------------------------------------------------

def _toy_bundles():
    """3 steps x 2 layers; step t routes the same experts as t-1."""
    from sim.predictor import LayerBundle
    keys = {(0, 0): frozenset({0, 1}), (0, 1): frozenset({2, 3}),
            (1, 0): frozenset({0, 1}), (1, 1): frozenset({2, 3}),
            (2, 0): frozenset({0, 1}), (2, 1): frozenset({2, 3})}
    return [LayerBundle(s, l, keys[(s, l)]) for s, l in sorted(keys)]


def test_predictor_causality_toy():
    bundles = _toy_bundles()
    predict = make_ranked_predictor("persist_layer", bundles)
    # step-0 bundles are unpredictable from routing history
    assert predict(0) == []
    # step-1 layer-0 must rank {0,1} (from step-0 same layer)
    ranked = predict(2)  # (1, 0)
    top = [k for k, _ in ranked[:2]]
    assert set(top) == {0, 1}


@pytest.mark.skipif(not TRACE_AVAILABLE, reason="sealed trace not present")
def test_predictor_causality_on_trace():
    demands, _ = load_default_trace()
    bundles = to_layer_bundles(demands)
    predict = make_ranked_predictor("persist_layer", bundles)
    # verify no future leakage: predict(idx) uses only (step-1, L) etc.
    for idx in (0, 1, 43, 100, len(bundles) - 1):
        b = bundles[idx]
        if b.step == 0:
            assert predict(idx) == []
        else:
            ranked = predict(idx)
            for k, _ in ranked:
                layer, expert = divmod(k, 100_000)
                assert layer == b.layer


@pytest.mark.skipif(not TRACE_AVAILABLE, reason="sealed trace not present")
def test_persist_beats_cross_union_on_dsv4():
    demands, _ = load_default_trace()
    bundles = to_layer_bundles(demands)
    r1 = evaluate_ranked_predictor("persist_layer", bundles)
    r2 = evaluate_ranked_predictor("cross_union", bundles)
    assert r1["recall@12"] > r2["recall@12"]
    assert 0.0 <= r1["recall@6"] <= r1["recall@8"] <= r1["recall@12"] \
        <= r1["recall@16"]


# --------------------------------------------------------------------------
# keep-value tuning
# --------------------------------------------------------------------------

@pytest.mark.skipif(not TRACE_AVAILABLE, reason="sealed trace not present")
def test_split_steps_disjoint_and_complete():
    demands, _ = load_default_trace()
    train, valid = split_steps(demands, range(0, 16, 2), range(1, 16, 2))
    assert {d.step for d in train}.isdisjoint({d.step for d in valid})
    assert {d.step for d in train} == set(range(0, 16, 2))
    assert len(train) + len(valid) == len(demands)


@pytest.mark.skipif(not TRACE_AVAILABLE, reason="sealed trace not present")
def test_tune_returns_train_best():
    demands, _ = load_default_trace()
    train, _ = split_steps(demands, range(0, 16, 2), range(1, 16, 2))
    slices: dict[int, list] = {}
    for d in train:
        slices.setdefault(d.layer, []).append(d)
    bundles = to_layer_bundles(demands)
    by_pair = {(b.step, b.layer): b.keys for b in bundles}
    trials = [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
    best, results = tune_coefficients(slices, 43, trials, RECORD_BYTES,
                                      by_pair, capacity_experts=8)
    best_miss = min(v["misses"] for v in results.values())
    assert results[str(best)]["misses"] == best_miss


# --------------------------------------------------------------------------
# prefetch engine
# --------------------------------------------------------------------------

def test_prefetch_capacity_never_exceeded():
    seq = []
    step_pat = [1, 2, 3, 4, 5]
    for s in range(4):
        for layer in range(2):
            for k in step_pat:
                seq.append((s, layer, k + 10 * layer))

    class D:
        def __init__(self, step, layer, expert):
            self.step, self.layer, self.expert = step, layer, expert
            self.scope = layer * 100_000 + expert
            self.nbytes = 100
            self.seq = 0

    demands = [D(*t) for t in seq]
    for i, d in enumerate(demands):
        d.seq = i
    bundles = to_layer_bundles(demands)
    by_pair = {(b.step, b.layer): b.keys for b in bundles}
    slices: dict[int, list] = {}
    for d in demands:
        slices.setdefault(d.layer, []).append(d)
    for layer, ds in slices.items():
        ranked_fn = (lambda ds=ds, layer=layer: None)  # placeholder
        from sim.keepvalue import build_ranked_fn
        rf = build_ranked_fn(ds, by_pair, layer)
        pol = PrefetchPolicy(3 * 100, 100, ranked_fn=rf)
        eng = PrefetchEngine(pol, record_bytes=100, ranked_fn=rf,
                             prefetch_mode="replace", max_prefetch_per_bundle=4,
                             min_dwell=1)
        eng.run(ds)
        assert len(pol.resident) <= 3
        issued = eng.prefetch_issued
        useful = eng.prefetch_useful
        wasted = eng.prefetch_wasted
        live = len(eng._prefetched_live)
        assert issued == useful + wasted + live


def test_prefetch_off_equals_demand_baseline():
    class D:
        def __init__(self, step, layer, expert):
            self.step, self.layer, self.expert = step, layer, expert
            self.scope = layer * 100_000 + expert
            self.nbytes = 100
            self.seq = 0

    seq = [(s, l, e) for s in range(3) for l in range(2)
           for e in (1, 2, 3, 4)]
    demands = [D(*t) for t in seq]
    for i, d in enumerate(demands):
        d.seq = i
    for mode in ("off", "safe"):
        slices: dict[int, list] = {}
        for d in demands:
            slices.setdefault(d.layer, []).append(d)
        pol = PrefetchPolicy(4 * 100, 100, ranked_fn=None)
        eng = PrefetchEngine(pol, record_bytes=100, ranked_fn=None,
                             prefetch_mode=mode)
        eng.run(demands)
        # no predictor -> no prefetching in either mode
        assert eng.prefetch_issued == 0


# --------------------------------------------------------------------------
# trace loader (integration, only when sealed evidence is present)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not TRACE_AVAILABLE, reason="sealed trace not present")
def test_default_trace_shape():
    demands, path = load_default_trace()
    assert path.name == "routed_experts.jsonl"
    assert len(demands) == 5676
    assert len({d.layer for d in demands}) == 43
    assert len({d.scope for d in demands}) == 2364
    assert all(d.nbytes == RECORD_BYTES for d in demands)
    step0 = [d for d in demands if d.step == 0]
    assert len(step0) == 1806  # 43 layers x 7 rows x top-6
