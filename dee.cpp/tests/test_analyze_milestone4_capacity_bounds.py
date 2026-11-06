import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_milestone4_capacity_bounds import analyze


def request(phase_token, expert, result, *, source_bytes=100):
    return {
        "event_type": "expert_request",
        "token_step": phase_token,
        "resolved_shard_layer": 0,
        "gpu_destination": "cuda:0",
        "expert_id": expert,
        "source_bytes": source_bytes,
        "cache_result": result,
        "transfer_launched": result == "miss",
    }


def test_warmup_seeds_capacity_replay_and_bounds_policy():
    warmup = [
        request(0, 1, "miss"),
        request(1, 2, "miss"),
    ]
    measured = [
        request(2, 1, "hit"),
        request(3, 3, "miss"),
        request(4, 2, "miss"),
    ]

    report = analyze(warmup, measured, capacities=[2, 3])

    assert report["actual"]["hits"] == 1
    cap2, cap3 = report["capacities"]
    assert cap2["policies"]["lru"]["hits"] == 1
    assert cap2["policies"]["belady_offline"]["hits"] == 2
    assert cap2["policy_headroom_hits"] == 1
    assert cap3["policies"]["lru"]["hits"] == 2
    assert report["unlimited_causal_cache"]["hits"] == 2
    assert report["preloaded_perfect_cache"]["hits"] == 3


def test_missing_source_bytes_fails_closed():
    row = request(0, 1, "miss")
    row.pop("source_bytes")

    with pytest.raises(ValueError, match="source_bytes"):
        analyze([], [row])
