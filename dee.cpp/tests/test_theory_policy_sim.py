"""Tests for theory.policy_sim (trace-driven eviction / prefetch simulator)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

from theory import policy_sim as ps  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
JOURNALS = (REPO / "benchmark_reports/deepseek-v4-flash-0731-t4/"
            "gpu2-phase4-cache-hierarchy/evidence/a2_fp4")


def _cycle(n_layers=3, per_layer=2, steps=12):
    """A decode-like stream: every step sweeps the layers and asks for the same experts."""
    calls = []
    for step in range(1, steps + 1):
        for layer in range(n_layers):
            calls.append(ps.LayerCall(step, layer, tuple(range(per_layer))))
    return calls


def _window(calls, policy, host=4, vram=2):
    return ps.simulate(calls, host_records=host, vram_records=vram,
                       host_policy=policy, vram_policy=policy).window(3)


def test_lru_thrashes_on_an_over_budget_cycle():
    # 6 records per step, 4 fit: LRU always evicts the record needed next.
    assert _window(_cycle(), "lru")["host_hit_rate"] == 0.0


def test_next_use_keeps_part_of_the_cycle():
    w = _window(_cycle(), "next_use")
    assert w["host_hit_rate"] >= 0.25


def test_belady_bounds_every_online_policy():
    calls = _cycle(n_layers=5, per_layer=3, steps=10)
    hits = {p: _window(calls, p, host=9, vram=3)["host_hits"]
            for p in ("lru", "layer_distance", "next_use", "belady")}
    assert hits["belady"] >= max(hits["lru"], hits["layer_distance"], hits["next_use"])
    assert hits["next_use"] > hits["lru"]


def test_everything_hits_when_the_tier_holds_a_step():
    for policy in ("lru", "next_use", "belady"):
        assert _window(_cycle(), policy, host=6)["host_hit_rate"] == 1.0


def test_policies_never_change_the_request_stream():
    calls = _cycle(n_layers=4, per_layer=3)
    lookups = {p: _window(calls, p, host=5)["lookups"] for p in ("lru", "next_use", "belady")}
    assert len(set(lookups.values())) == 1


def test_next_use_mirrors_the_engine_estimator():
    # Same numbers as tests/test_host_pack_cache.cpp::test_next_use_estimator_model.
    pol = ps.NextUse(4, alpha=0.5, p0=0.25)
    pol.begin_layer(1, 0, 0)
    pol.on_use((0, 7))
    once = pol.probability((0, 7))
    pol.on_use((0, 7))
    assert pol.probability((0, 7)) == once == 0.25 * 0.5 + 0.5
    pol.begin_layer(1, 1, 1)
    assert pol.probability((0, 7)) == once
    assert pol.rank((2, 9)) > pol.rank((0, 7))
    for _ in range(2):
        pol.begin_layer(2, 0, 2)
        pol.begin_layer(2, 1, 3)
    assert pol.probability((0, 7)) == pytest.approx(once * 0.25)
    assert pol.rank((0, 7)) == pytest.approx(3 + 4 * (1 / (once * 0.25) - 1))


def test_hint_set_recall_and_precision():
    import random
    truth = [(5, e) for e in range(40)]
    exact = ps.hint_set(truth, recall=1.0, precision=1.0, n_experts=256,
                        rng=random.Random(0))
    assert sorted(exact) == truth
    noisy = ps.hint_set(truth, recall=1.0, precision=0.5, n_experts=256,
                        rng=random.Random(0))
    assert len(noisy) == 80 and set(truth) <= set(noisy)


def test_prev_step_hints_remove_demand_waits_on_a_stable_cycle():
    calls = _cycle(n_layers=6, per_layer=4, steps=12)
    tm = ps.TimeModel(read_ms=1.0, lanes=2, stage_ms=0.0, layer_ms=10.0)
    kw = dict(host_records=8, vram_records=4, time_model=tm,
              host_policy="next_use", vram_policy="next_use")
    plain = ps.simulate_timed(calls, **kw)
    hinted = ps.simulate_timed(calls, lookahead="prev_step", **kw)
    assert hinted.tokens_per_second(1, 4) > plain.tokens_per_second(1, 4)
    assert hinted.counts.window(4)["host_hit_rate"] > plain.counts.window(4)["host_hit_rate"]
    assert hinted.counts.window(4)["lookups"] == plain.counts.window(4)["lookups"]


@pytest.mark.skipif(not (JOURNALS / "routed_experts-q0.jsonl").is_file(),
                    reason="committed route journals not present")
def test_reproduces_the_r5_host_tier_measurements():
    journals = [ps.load_journal(JOURNALS / f"routed_experts-q{q}.jsonl") for q in range(8)]
    calls = [c for c in ps.merge_cohort(journals) if c.step <= 64]
    lru16 = ps.simulate(calls, host_records=1285, vram_records=281).window(33, 64)
    lru64 = ps.simulate(calls, host_records=5140, vram_records=281).window(33, 64)
    # p6m-l4-r5 measured 0/49,463 host hits at 16 GiB and 0.8351 at 64 GiB.
    assert lru16["host_hit_rate"] == 0.0
    assert lru64["host_hit_rate"] == pytest.approx(0.8351, abs=0.03)
    next16 = ps.simulate(calls, host_records=1285, vram_records=281,
                         host_policy="next_use", vram_policy="next_use").window(33, 64)
    assert next16["host_hit_rate"] > 0.35
    assert next16["source_reads_per_step"] < 0.65 * lru16["source_reads_per_step"]
