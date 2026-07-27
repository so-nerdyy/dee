import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_milestone4_transfer_ledger import analyze


def request(
    token,
    expert,
    transfer_id,
    *,
    kind="cold",
    victim=-1,
    layer=0,
    gpu="cuda:0",
    consumed=True,
    evicted_before_use=False,
    reason="demand_miss",
    before_entries=0,
    after_entries=1,
):
    launched = kind == "cold"
    return {
        "event_type": "expert_request",
        "run_id": "fixture",
        "token_step": token,
        "logical_layer": layer,
        "resolved_shard_layer": layer,
        "expert_id": expert,
        "gpu_destination": gpu,
        "source_checkpoint_shard": "model-00001.safetensors",
        "expert_bytes": 100,
        "source_bytes": 100,
        "cache_state_before": {
            "bytes": before_entries * 100,
            "resident_entries": before_entries,
        },
        "cache_state_after": {
            "bytes": after_entries * 100,
            "resident_entries": after_entries,
        },
        "cache_result": "miss" if launched else "hit",
        "cache_kind": kind,
        "transfer_id": transfer_id,
        "residency_generation": max(1, transfer_id + 1),
        "pin_count_after": 1,
        "transfer_launched": launched,
        "transfer_consumed": consumed,
        "evicted_before_use": evicted_before_use,
        "transfer_reason": reason,
        "evicted_expert_id": victim,
        "request_time_ms": float(token),
        "_source_file": "fixture.jsonl",
        "_source_line": token + 1,
    }


def transfer(row):
    return {
        "event_type": "expert_transfer",
        "run_id": row["run_id"],
        "token_step": row["token_step"],
        "logical_layer": row["logical_layer"],
        "resolved_shard_layer": row["resolved_shard_layer"],
        "expert_id": row["expert_id"],
        "gpu_destination": row["gpu_destination"],
        "transfer_id": row["transfer_id"],
        "component": "expert_weight",
        "direction": "h2d",
        "bytes": row["source_bytes"],
        "_source_file": "fixture.jsonl",
        "_source_line": row["_source_line"] + 100,
    }


def events(*rows):
    result = []
    for row in rows:
        result.append(row)
        if row["transfer_launched"]:
            result.append(transfer(row))
    return result


def test_exact_categories_reconcile_and_belady_separates_policy():
    warmup = events(
        request(0, 1, 0, victim=-1),
        request(1, 2, 1, victim=-1, before_entries=1, after_entries=2),
        request(
            2, 1, 0, kind="resident", before_entries=2, after_entries=2
        ),
    )
    measured = events(
        request(3, 3, 2, victim=2, before_entries=2, after_entries=2),
        request(4, 2, 3, victim=1, before_entries=2, after_entries=2),
    )

    report = analyze(warmup, measured, capacity=2)

    assert report["totals"]["transfers"] == 4
    assert report["totals"]["bytes"] == 400
    assert report["categories"]["compulsory cold miss"]["transfers"] == 3
    assert report["categories"]["policy miss"]["transfers"] == 1
    assert report["totals"]["classified_transfers"] == 4
    assert report["totals"]["classified_bytes"] == 400
    assert report["by_phase"]["warmup"]["totals"]["transfers"] == 2
    assert report["by_phase"]["measured"]["totals"]["transfers"] == 2
    assert report["totals"]["safe_cache_reducible_transfers"] == 1


def test_precedence_classifies_wrong_gpu_and_never_consumed_once():
    wrong = request(0, 1, 0, layer=0, gpu="cuda:1")
    unused = request(1, 2, 1, consumed=False)

    report = analyze([], events(wrong, unused), capacity=2)

    assert report["categories"]["wrong-GPU transfer"]["transfers"] == 1
    assert report["categories"]["transfer that was never consumed"]["transfers"] == 1
    assert sum(item["transfers"] for item in report["categories"].values()) == 2


def test_all_nine_categories_are_reachable_and_exclusive():
    scenarios = {
        "compulsory cold miss": (
            [],
            events(request(0, 1, 0)),
            2,
        ),
        "capacity miss": (
            [],
            events(
                request(0, 1, 0),
                request(1, 2, 1, victim=1, before_entries=1, after_entries=1),
                request(2, 1, 2, victim=2, before_entries=1, after_entries=1),
            ),
            1,
        ),
        "policy miss": (
            events(
                request(0, 1, 0),
                request(1, 2, 1, before_entries=1, after_entries=2),
                request(2, 1, 0, kind="resident", before_entries=2, after_entries=2),
            ),
            events(
                request(3, 3, 0, victim=2, before_entries=2, after_entries=2),
                request(4, 2, 1, victim=1, before_entries=2, after_entries=2),
            ),
            2,
        ),
        "duplicate transfer": (
            [],
            events(
                request(0, 1, 0),
                request(1, 1, 1, before_entries=1, after_entries=1),
            ),
            2,
        ),
        "stale-prefetch transfer": (
            [],
            events(request(0, 1, 0, consumed=False, reason="prefetch_lookahead")),
            2,
        ),
        "wrong-GPU transfer": (
            [],
            events(request(0, 1, 0, layer=0, gpu="cuda:1")),
            2,
        ),
        "reload caused by premature eviction": (
            [],
            events(
                request(0, 1, 0, consumed=False, evicted_before_use=True),
                request(1, 2, 1, victim=1, before_entries=1, after_entries=1),
                request(2, 1, 2, victim=2, before_entries=1, after_entries=1),
            ),
            1,
        ),
        "transfer caused by cache metadata inconsistency": (
            [],
            events(request(0, 1, 0, before_entries=1, after_entries=1)),
            2,
        ),
        "transfer that was never consumed": (
            [],
            events(request(0, 1, 0, consumed=False)),
            2,
        ),
    }

    for expected, (warmup, measured, capacity) in scenarios.items():
        report = analyze(warmup, measured, capacity=capacity)
        assert report["categories"][expected]["transfers"] >= 1, expected
        assert sum(
            item["transfers"] for item in report["categories"].values()
        ) == report["totals"]["transfers"]


def test_transfer_ids_are_scoped_by_profiler_phase():
    warm = request(0, 1, 0)
    measured = request(1, 2, 0, before_entries=1, after_entries=2)

    report = analyze(events(warm), events(measured), capacity=2)

    assert report["totals"]["transfers"] == 2
    assert all(row["metadata_consistent"] for row in report["ledger"])
    assert len({row["stable_key"] for row in report["ledger"]}) == 2


def test_missing_required_transfer_field_fails_closed():
    row = request(0, 1, 0)
    row.pop("residency_generation")

    with pytest.raises(ValueError, match="residency_generation"):
        analyze([], events(row), capacity=2)
