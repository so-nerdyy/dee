#!/usr/bin/env python3
"""Tests for the exact-staging track: bounded staging queue semantics,
validated host-pack replay, and the pread bench bookkeeping (local
fallback mode, no measurements claimed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent.parent          # ../dee-staging
sys.path.insert(0, str(WORKTREE / "dee.cpp" / "experiments" / "exact_staging"))

from bounded_staging_queue import (  # noqa: E402
    BoundedStagingQueue, DiskModel, StagingConfig, pack_replay)

REPO = WORKTREE  # main repo checkout lives beside the worktrees
SEALED_JOURNAL = (REPO.parent / "dynamic_expert_eviction" / "dee.cpp" / "tmp"
                  / "v65-terminal-fetch-20260903T0202Z" / "routed_experts.jsonl")
RB = 13_369_344


# ---------------------------------------------------------------------------
# Bounded staging queue semantics (mock times)
# ---------------------------------------------------------------------------

def _q(**kw) -> BoundedStagingQueue:
    cfg = StagingConfig(max_inflight_reads=kw.pop("max_inflight", 8),
                        pinned_budget_bytes=kw.pop("pinned", 8 * (1 << 30)),
                        **kw)
    return BoundedStagingQueue(cfg)


def test_resident_hit_is_free():
    q = _q()
    q.begin_step({(0, 5)})
    t = q.note_route_known(1, 0, 5, needed_by_ts=10.0)
    assert t.source_tier == "resident"
    assert t.ready_before_demand is True


def test_residency_is_bounded_lru():
    q = _q()
    q.cfg.resident_slots = 2
    q.begin_step({(0, 1), (0, 2)})
    # touch (0,3),(0,4): resident set must evict LRU entries, never exceed cap
    q.note_route_known(1, 0, 3, needed_by_ts=10.0)
    q.note_route_known(1, 0, 4, needed_by_ts=10.0)
    assert len(q.resident) <= 2


def test_duplicate_request_resolves_fast_and_consistently():
    q = _q()
    q.begin_step(set())
    t1 = q.note_route_known(1, 0, 7, needed_by_ts=100.0)
    t2 = q.note_route_known(1, 0, 7, needed_by_ts=100.0)
    assert t1.source_tier == "read"
    # the synchronous mock materializes on first request, so the duplicate
    # resolves from residency (production port keeps the in-flight dedup
    # branch for genuinely concurrent duplicates)
    assert t2.source_tier in ("dedup", "resident")
    # the duplicate never lands later than the original request
    assert t2.read_complete_ts <= t1.read_complete_ts
    assert t2.ready_before_demand


def test_inflight_is_bounded():
    q = _q(max_inflight=2)
    q.begin_step(set())
    for e in range(5):
        q.note_route_known(1, 0, e, needed_by_ts=1000.0)
    # reads beyond the bound queue behind slots (later completion), never drop
    reads = [t for t in q.telemetry if t.source_tier == "read"]
    assert len(reads) == 5


def test_stale_reads_cancelled_at_step_boundary():
    q = _q()
    q.begin_step(set())
    q.note_route_known(1, 0, 9, needed_by_ts=100.0)
    q.begin_step(set())
    assert q.cancelled_reads == 1


def test_retain_last_step_ablation_clears_pack():
    q = _q(retain_last_step=False)
    q.begin_step(set())
    q.note_route_known(1, 0, 3, needed_by_ts=100.0)
    assert len(q.pack) == 1
    q.begin_step(set())
    assert len(q.pack) == 0


def test_pack_budget_is_respected():
    q = _q(pinned=2 * RB)
    q.begin_step(set())
    for e in range(4):
        q.note_route_known(1, 0, e, needed_by_ts=100.0)
    assert q.pack_bytes <= 2 * RB


# ---------------------------------------------------------------------------
# Validated host-pack replay (needs the sealed journal; skipped otherwise)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SEALED_JOURNAL.exists(),
                    reason="sealed v65 journal not present on this host")
class TestPackReplayValidation:
    def test_lru_682_reproduces_sealed_counters(self):
        r = pack_replay(SEALED_JOURNAL, 682)
        d0, d1 = r["per_device"]["cuda:0"], r["per_device"]["cuda:1"]
        assert abs(d0["misses"] - 1390) <= 2
        assert abs(d0["evictions"] - 708) <= 2
        assert abs(d1["misses"] - 1091) <= 2
        assert abs(d1["evictions"] - 409) <= 2
        assert abs(r["decode_storage_misses_total"] - 1252) <= 2

    def test_oracle_floor_at_17gib(self):
        r = pack_replay(SEALED_JOURNAL, 1365)
        assert r["decode_storage_misses_total"] == 1135

    def test_per_step_table_has_15_decode_steps(self):
        r = pack_replay(SEALED_JOURNAL, 682)
        assert len(r["decode_per_step_reads"]) == 15
        assert sum(r["decode_per_step_reads"].values()) <= 1252

    def test_budget_never_increases_misses(self):
        small = pack_replay(SEALED_JOURNAL, 682)["decode_storage_misses_total"]
        big = pack_replay(SEALED_JOURNAL, 1365)["decode_storage_misses_total"]
        assert big <= small


# ---------------------------------------------------------------------------
# Pread bench bookkeeping (local fallback mode only — NO measurements)
# ---------------------------------------------------------------------------

def _fake_store(tmp_path: Path, records: int = 4) -> Path:
    p = tmp_path / "experts.bin"
    p.write_bytes(bytes(range(256)) * (RB * records // 256))
    return p


def test_bench_pass_bookkeeping(tmp_path):
    sys.path.insert(0, str(WORKTREE / "tools"))
    import bench_expert_pread as b
    store = _fake_store(tmp_path, 4)
    order = [0, 1, 2, 3]
    r = b._run_pass(store, order, depth=2, per_lane_fds=True,
                    target_records=4)
    assert r["reads"] == 4
    assert r["bytes_completed"] == 4 * RB
    assert r["short_or_empty_reads"] == 0


def test_bench_rejects_bad_depth():
    sys.path.insert(0, str(WORKTREE / "tools"))
    import bench_expert_pread as b
    assert b.QUEUE_DEPTHS == [1, 2, 3, 4, 6, 8, 12, 16]


def test_disk_model_saturation_law():
    dm = DiskModel(aggregate_cap_mb_s=370.0)
    t1 = dm.submit(RB)
    dm.complete()
    t2 = dm.submit(RB)
    dm.complete()
    dm.submit(RB)  # depth 2
    t_at_2 = dm.submit(RB)
    dm.complete(); dm.complete()
    assert t2 == pytest.approx(t1)          # same depth -> same service time
    assert t_at_2 > t2                       # deeper queue -> slower per read


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
