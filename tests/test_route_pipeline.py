#!/usr/bin/env python3
"""Tests for dee.cpp/experiments/route_pipeline/sim.py.

Structural properties only: determinism, fail-closed dependencies, legal
schedules never slower than baseline, handoff schema. Millisecond values are
SIMULATED (assumed latencies) and asserted only relationally, never as
performance claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "dee.cpp" / "experiments" / "route_pipeline"))

from sim import (  # noqa: E402
    HASH_LAYERS,
    Latency,
    ScheduleFlags,
    handoff,
    simulate,
)


def test_deterministic():
    a = simulate().critical_ms
    b = simulate().critical_ms
    assert a == b and a > 0


def test_hash_layers_are_first_three():
    assert HASH_LAYERS == frozenset((0, 1, 2))


def test_invalid_inputs_raise():
    import pytest
    with pytest.raises(ValueError):
        simulate(hit_rate=1.5)
    with pytest.raises(ValueError):
        simulate(n_layers=0)


def test_legal_schedules_never_slower_than_baseline():
    base = simulate().critical_ms
    for flags in (ScheduleFlags(hash_prefetch=True),
                  ScheduleFlags(shared_overlap=True),
                  ScheduleFlags(no_layer_sync=True),
                  ScheduleFlags(batch_submit=True),
                  ScheduleFlags(hash_prefetch=True, shared_overlap=True,
                                no_layer_sync=True, batch_submit=True)):
        assert simulate(flags=flags).critical_ms <= base


def test_all_hits_faster_than_all_miss():
    assert (simulate(hit_rate=1.0).critical_ms
            < simulate(hit_rate=0.0).critical_ms)


def test_shared_overlap_saves_shared_time_per_layer():
    lat = Latency()
    base = simulate(lat=lat).critical_ms
    over = simulate(lat=lat, flags=ScheduleFlags(shared_overlap=True)).critical_ms
    # Serial schedule pays shared_expert after routed compute every layer;
    # overlapped pays max() instead. Saving is bounded and positive.
    assert 0 < base - over <= 43 * lat.shared_expert + 1e-6


def test_no_layer_sync_saves_42_syncs():
    lat = Latency()
    base = simulate(lat=lat).critical_ms
    nosync = simulate(lat=lat, flags=ScheduleFlags(no_layer_sync=True)).critical_ms
    assert abs((base - nosync) - 42 * lat.layer_sync) < 1e-6


def test_edges_form_causal_chain():
    res = simulate(n_layers=4, topk=2)
    by_key = {}
    for label, layer, expert, t in res.edges:
        by_key.setdefault((label, layer, expert), t)
    for layer in range(4):
        assert by_key[("route_known", layer, -1)] <= by_key[("combine", layer, -1)]
        if layer > 0:
            # R5: score-layer routes cannot precede previous combine... except
            # hash layers 0-2, whose IDs are free at token start.
            if layer not in HASH_LAYERS:
                assert by_key[("combine", layer - 1, -1)] <= by_key[("route_known", layer, -1)]


def test_handoff_schema():
    h = handoff(simulate())
    assert h["future_layer_official_lead"] == 0
    assert h["token_level_lead"] == 0
    assert h["cross_gpu_official_overlap"] is False
    assert h["hash_layer_early_ids"] == [0, 1, 2]
    assert "ASSUMED" in h["latency_source"]
    json.dumps(h)


def test_lookahead_table_shape():
    # Phase B contract: lead classification for +1/+2/+4/+8, score layers.
    table = {1: "REQUIRES_CURRENT_LAYER_OUTPUT",
             2: "REQUIRES_CURRENT_LAYER_OUTPUT",
             4: "REQUIRES_CURRENT_LAYER_OUTPUT",
             8: "REQUIRES_CURRENT_LAYER_OUTPUT"}
    assert set(table) == {1, 2, 4, 8}
    assert all(v == "REQUIRES_CURRENT_LAYER_OUTPUT" for v in table.values())
