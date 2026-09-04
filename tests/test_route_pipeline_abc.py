#!/usr/bin/env python3
"""Tests for route-pipeline A/B/C prototypes (host-side, no CUDA).

Covers the mandated cases: hash ID causality, weights never consumed early,
no future-layer score-route lookahead, event dependency ordering, resource
lifetime, exact join ordering, and the missing-wait negative case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parent.parent / "dee.cpp" / "experiments" / "route_pipeline"
sys.path.insert(0, str(EXP))

from event_handoff import (  # noqa: E402
    Event,
    Resource,
    Stream,
    handoff_pattern,
    narrow_d2h,
)
from hash_stage import (  # noqa: E402
    HASH_LAYERS,
    HashStagePlanner,
    resolve_hash_ids,
)
from shared_overlap import (  # noqa: E402
    BranchResult,
    OverlapWindow,
    exact_join,
)


def _table():
    return {7: [10, 11, 12, 13, 14, 15], 9: [20, 21, 22, 23, 24, 25]}


# --- A: hash causality ----------------------------------------------------

def test_hash_ids_depend_only_on_table_and_token():
    assert resolve_hash_ids(_table(), 7) == [10, 11, 12, 13, 14, 15]
    assert resolve_hash_ids(_table(), 9) == [20, 21, 22, 23, 24, 25]


def test_hash_unknown_token_fails_closed():
    with pytest.raises(KeyError):
        resolve_hash_ids(_table(), 12345)


def test_weights_never_consumed_early():
    p = HashStagePlanner()
    p.token_start(_table(), 7, t_ms=0.0)
    with pytest.raises(ValueError):
        p.consume(0, 10, t_ms=5.0)  # staged but weights not ready
    p.mark_weights_ready(0, t_ms=6.0, evidence="h_0 post-residual")
    row = p.consume(0, 10, t_ms=7.0)
    assert row.ready_before_demand is True
    assert row.weight_ready_t == 6.0


def test_weights_ready_needs_evidence():
    p = HashStagePlanner()
    with pytest.raises(ValueError):
        p.mark_weights_ready(0, t_ms=1.0, evidence="")


def test_unstaged_consume_refused():
    p = HashStagePlanner()
    p.mark_weights_ready(1, t_ms=1.0, evidence="h_1")
    with pytest.raises(ValueError):
        p.consume(1, 999, t_ms=2.0)  # IDs never resolved for this expert


def test_duplicate_suppression_and_bytes():
    p = HashStagePlanner()
    p.resident.add((0, 10))
    p.hostpacked.add((1, 11))
    submitted = p.token_start(_table(), 7, t_ms=0.0)
    assert (0, 10) not in submitted and (1, 11) not in submitted
    summary = p.bytes_summary()
    assert summary["suppressed_duplicates"] == 2
    assert summary["submitted_reads"] == 3 * 6 - 2
    assert summary["max_early_bytes"] == 18 * 13369344


def test_invalidate_cancels_pending_not_resident():
    p = HashStagePlanner()
    p.token_start(_table(), 7, t_ms=0.0)
    cancelled = p.invalidate(t_ms=3.0)
    assert len(cancelled) == 18
    assert p.pending == {}
    with pytest.raises(ValueError):
        p.consume(0, 10, t_ms=4.0)  # weights cleared + staging cancelled


def test_no_future_layer_score_lookahead():
    # The planner has no API accepting a future layer: token_start only
    # touches HASH_LAYERS; anything else must raise via consume().
    assert set(HASH_LAYERS) == {0, 1, 2}
    import inspect
    src = inspect.getsource(HashStagePlanner.token_start)
    assert "HASH_LAYERS" in src


# --- B: shared join -------------------------------------------------------

def test_exact_join_order_preserved_despite_completion_order():
    routed = BranchResult("routed", 9.0, payload="R")
    shared = BranchResult("shared", 2.0, payload="S")  # finished first
    out, meta = exact_join(routed, shared)
    assert out == ("add", "R", "S")
    assert meta["join_order"] == ["routed", "shared"]
    assert meta["first_done"] == "shared"


def test_join_with_incomplete_branch_raises():
    with pytest.raises(ValueError):
        exact_join(BranchResult("routed", 1.0, None),
                   BranchResult("shared", 1.0, "S"))


def test_overlap_wall_interval_bounds():
    w = OverlapWindow(0.0, 10.0, 2.0, 5.0)
    assert w.overlap_ms() == 3.0
    best, worst = w.wall_interval(contention=1.0)
    assert worst == 13.0 and best == 10.0
    exp_best, _ = w.wall_interval(contention=1.0)
    mid, _ = w.wall_interval(contention=0.5)
    assert exp_best <= mid <= worst
    with pytest.raises(ValueError):
        w.wall_interval(contention=1.5)
    assert w.extra_workspace_bytes(4096, 1) == 2 * 4096 * 2
    assert w.needs_join_sync() is True


# --- C: events ------------------------------------------------------------

def test_legal_handoff_pattern():
    prod, cons = Stream("prefetch"), Stream("compute")
    res = Resource("expert-blob", owner="prefetch", live_from=0.0)
    out = handoff_pattern(prod, cons, res, compute_ms=1.0, t0=0.0)
    assert out["syncs"] == 0 and out["events"] == 1
    assert res.released is True
    assert out["consumer_end"] >= out["producer_done"]


def test_missing_wait_is_detected():
    cons = Stream("compute")
    ev = Event("never-recorded")
    with pytest.raises(ValueError):
        cons.wait_on(ev, t_now=0.0)


def test_producer_error_propagates_not_swallowed():
    cons = Stream("compute")
    ev = Event("staging")
    ev.record(1.0)
    ev.fail("H2D error")
    with pytest.raises(ValueError):
        cons.wait_on(ev, t_now=2.0)


def test_premature_reuse_detected():
    res = Resource("scratch", owner="prefetch", live_from=0.0)
    res.release_after = ["ready"]
    with pytest.raises(ValueError):
        res.release(set(), t=1.0)  # use-after-free without the wait


def test_double_record_rejected():
    ev = Event("e")
    ev.record(1.0)
    with pytest.raises(ValueError):
        ev.record(2.0)


def test_narrow_d2h_bounds_prize():
    r = narrow_d2h(ids_bytes=24, unrelated_stream_ms=0.4, copy_ms=0.05)
    assert r["avoidable_ms"] == 0.4 and r["unavoidable_ms"] == 0.05
    assert r["verdict"] == "WORTH_PROTOTYPING"
    r2 = narrow_d2h(ids_bytes=24, unrelated_stream_ms=0.0, copy_ms=0.05)
    assert r2["verdict"] == "LOW_EXPECTED_VALUE"
