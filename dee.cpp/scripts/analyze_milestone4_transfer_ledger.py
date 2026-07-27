#!/usr/bin/env python3
"""Build and classify the complete Milestone 4 expert-transfer ledger."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable


CATEGORIES = (
    "compulsory cold miss",
    "capacity miss",
    "policy miss",
    "duplicate transfer",
    "stale-prefetch transfer",
    "wrong-GPU transfer",
    "reload caused by premature eviction",
    "transfer caused by cache metadata inconsistency",
    "transfer that was never consumed",
)
FIXED_CAPACITY_AVOIDABLE_CATEGORIES = set(CATEGORIES[2:])
LARGER_SAFE_CACHE_REDUCIBLE_CATEGORIES = set(CATEGORIES[1:])
REQUIRED_TRANSFER_FIELDS = (
    "token_step",
    "logical_layer",
    "resolved_shard_layer",
    "expert_id",
    "gpu_destination",
    "source_checkpoint_shard",
    "expert_bytes",
    "source_bytes",
    "cache_state_before",
    "cache_state_after",
    "transfer_id",
    "transfer_reason",
    "residency_generation",
    "pin_count_after",
    "transfer_launched",
    "transfer_consumed",
    "evicted_before_use",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            value["_source_file"] = str(path)
            value["_source_line"] = line_number
            rows.append(value)
    return rows


def expected_gpu(layer: int, split_layer: int) -> str:
    return f"cuda:{0 if layer < split_layer else 1}"


def cache_domain(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["resolved_shard_layer"]), str(row["gpu_destination"])


def transfer_identity(row: dict[str, Any]) -> tuple[str, int, str, int]:
    layer, gpu = cache_domain(row)
    return str(row.get("_ledger_phase", "")), layer, gpu, int(row["transfer_id"])


def belady_hits(requests: list[dict[str, Any]], capacity: int) -> list[bool]:
    experts = [int(row["expert_id"]) for row in requests]
    future_positions: dict[int, collections.deque[int]] = collections.defaultdict(
        collections.deque
    )
    for position, expert in enumerate(experts):
        future_positions[expert].append(position)
    resident: set[int] = set()
    hits: list[bool] = []
    for position, expert in enumerate(experts):
        queue = future_positions[expert]
        if queue and queue[0] == position:
            queue.popleft()
        hit = expert in resident
        hits.append(hit)
        if hit:
            continue
        if len(resident) >= capacity:
            victim = max(
                resident,
                key=lambda candidate: (
                    future_positions[candidate][0]
                    if future_positions[candidate]
                    else len(experts) + 1
                ),
            )
            resident.remove(victim)
        resident.add(expert)
    return hits


def _required_coverage(transfers: list[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, Any] = {"transfers": len(transfers), "fields": {}}
    missing_rows = []
    for field in REQUIRED_TRANSFER_FIELDS:
        present = sum(row.get(field) is not None for row in transfers)
        coverage["fields"][field] = {
            "present": present,
            "fraction": present / len(transfers) if transfers else None,
        }
        if present != len(transfers):
            missing_rows.append(field)
    coverage["complete"] = not missing_rows
    coverage["incomplete_fields"] = missing_rows
    return coverage


def _classify(
    row: dict[str, Any],
    *,
    metadata_consistent: bool,
    duplicate: bool,
    wrong_gpu: bool,
    seen_before: bool,
    belady_hit: bool,
    reload_after_premature_eviction: bool,
) -> str:
    reason = str(row.get("transfer_reason") or "demand_miss")
    consumed = bool(row["transfer_consumed"])
    if not metadata_consistent:
        return "transfer caused by cache metadata inconsistency"
    if duplicate:
        return "duplicate transfer"
    if wrong_gpu:
        return "wrong-GPU transfer"
    if reason.startswith("prefetch") and not consumed:
        return "stale-prefetch transfer"
    if reload_after_premature_eviction:
        return "reload caused by premature eviction"
    if not consumed:
        return "transfer that was never consumed"
    if seen_before and belady_hit:
        return "policy miss"
    if seen_before:
        return "capacity miss"
    return "compulsory cold miss"


def analyze(
    warmup_events: Iterable[dict[str, Any]],
    measured_events: Iterable[dict[str, Any]],
    *,
    capacity: int,
    split_layer: int = 20,
    allow_missing_warmup_timing: bool = False,
) -> dict[str, Any]:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    phase_rows = [
        ("warmup", row) for row in warmup_events
        if row.get("event_type") in {"expert_request", "expert_transfer"}
    ]
    phase_rows += [
        ("measured", row) for row in measured_events
        if row.get("event_type") in {"expert_request", "expert_transfer"}
    ]
    requests = [
        {**row, "_ledger_phase": phase}
        for phase, row in phase_rows if row.get("event_type") == "expert_request"
    ]
    transfer_events = [
        {**row, "_ledger_phase": phase}
        for phase, row in phase_rows
        if row.get("event_type") == "expert_transfer"
        and row.get("component") == "expert_weight"
        and row.get("direction") == "h2d"
    ]
    launched = [
        row for row in requests
        if bool(row.get("transfer_launched", row.get("cache_kind") == "cold"))
    ]
    coverage = _required_coverage(launched)
    if not coverage["complete"]:
        raise ValueError(
            "transfer ledger coverage is incomplete: "
            + ", ".join(coverage["incomplete_fields"])
        )

    transfer_by_id: dict[tuple[str, int, str, int], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for event in transfer_events:
        transfer_by_id[transfer_identity(event)].append(event)

    requests_by_domain: dict[tuple[int, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for request in requests:
        requests_by_domain[cache_domain(request)].append(request)
    belady_by_object: dict[int, bool] = {}
    for domain_requests in requests_by_domain.values():
        for request, hit in zip(
            domain_requests, belady_hits(domain_requests, capacity), strict=True
        ):
            belady_by_object[id(request)] = hit

    actual_resident: dict[tuple[int, str], set[int]] = collections.defaultdict(set)
    seen_experts: dict[tuple[int, str], set[int]] = collections.defaultdict(set)
    prior_transfer_ordinal: dict[tuple[int, str, int], int] = {}
    prior_transfer_time: dict[tuple[int, str, int], float] = {}
    prematurely_evicted: set[tuple[int, str, int]] = set()
    ledger = []
    transfer_ordinal = 0
    launched_ids = {id(row) for row in launched}
    for request in requests:
        domain = cache_domain(request)
        expert = int(request["expert_id"])
        resident = actual_resident[domain]
        is_transfer = id(request) in launched_ids
        victim_value = request.get("evicted_expert_id", -1)
        victim = int(victim_value) if victim_value is not None else -1
        was_resident = expert in resident

        if is_transfer:
            identity = transfer_identity(request)
            matching = transfer_by_id.get(identity, [])
            before = request["cache_state_before"]
            after = request["cache_state_after"]
            timing_observed = len(matching) == 1
            timing_optional = (
                allow_missing_warmup_timing
                and request["_ledger_phase"] == "warmup"
            )
            projected_after = set(resident)
            if victim >= 0:
                projected_after.discard(victim)
            projected_after.add(expert)
            metadata_checks = {
                "cold_request": request.get("cache_kind") == "cold",
                "cache_miss": request.get("cache_result") == "miss",
                "transfer_timing_unique_or_optional": (
                    timing_observed or (timing_optional and not matching)
                ),
                "transfer_bytes_match": (
                    not timing_observed
                    or int(matching[0].get("bytes", -1))
                    == int(request["source_bytes"])
                ),
                "expert_bytes_positive": int(request["expert_bytes"]) > 0,
                "source_bytes_positive": int(request["source_bytes"]) > 0,
                "residency_generation_positive": (
                    int(request["residency_generation"]) > 0
                ),
                "after_entries_within_capacity": (
                    1 <= int(after.get("resident_entries", -1)) <= capacity
                ),
                "after_bytes_cover_expert": (
                    int(after.get("bytes", -1)) >= int(request["expert_bytes"])
                ),
                "pin_count_positive": int(request["pin_count_after"]) >= 1,
                "before_entries_match_replay": (
                    int(before.get("resident_entries", -1)) == len(resident)
                ),
                "after_entries_match_replay": (
                    int(after.get("resident_entries", -1)) == len(projected_after)
                ),
            }
            metadata_failures = sorted(
                name for name, passed in metadata_checks.items() if not passed
            )
            metadata_consistent = not metadata_failures
            duplicate = was_resident or len(matching) > 1
            wrong_gpu = request["gpu_destination"] != expected_gpu(
                int(request["logical_layer"]), split_layer
            )
            reuse_key = (domain[0], domain[1], expert)
            another_copy_gpus = sorted(
                gpu
                for (other_layer, gpu), other_resident in actual_resident.items()
                if other_layer == domain[0]
                and gpu != domain[1]
                and expert in other_resident
            )
            previous_ordinal = prior_transfer_ordinal.get(reuse_key)
            current_time = float(request.get("request_time_ms", 0.0))
            previous_time = prior_transfer_time.get(reuse_key)
            category = _classify(
                request,
                metadata_consistent=metadata_consistent,
                duplicate=duplicate,
                wrong_gpu=wrong_gpu,
                seen_before=expert in seen_experts[domain],
                belady_hit=belady_by_object[id(request)],
                reload_after_premature_eviction=reuse_key in prematurely_evicted,
            )
            event = matching[0] if len(matching) == 1 else {}
            row = {
                "stable_key": (
                    f"{request['_ledger_phase']}:{domain[0]}:{domain[1]}:"
                    f"{int(request['transfer_id'])}"
                ),
                "phase": request["_ledger_phase"],
                "token": int(request["token_step"]),
                "layer": int(request["logical_layer"]),
                "resolved_layer": domain[0],
                "expert_id": expert,
                "destination_gpu": domain[1],
                "source_location": request["source_checkpoint_shard"],
                "source_kind": "mmap_bf16_checkpoint",
                "byte_count": int(event.get("bytes", request["source_bytes"])),
                "destination_byte_count": int(request["expert_bytes"]),
                "reason_for_transfer": request.get("transfer_reason") or "demand_miss",
                "cache_state_before": before,
                "cache_state_after": after,
                "residency_generation": int(request["residency_generation"]),
                "pin_count_after": int(request["pin_count_after"]),
                "transfer_timing_observed": timing_observed,
                "another_copy_already_resident": bool(another_copy_gpus),
                "another_copy_resident_gpus": another_copy_gpus,
                "another_copy_basis": "replayed per-layer per-GPU cache state",
                "same_expert_transferred_recently": previous_ordinal is not None,
                "transfer_distance": (
                    transfer_ordinal - previous_ordinal
                    if previous_ordinal is not None else None
                ),
                "milliseconds_since_previous_transfer": (
                    current_time - previous_time if previous_time is not None else None
                ),
                "later_used": bool(request["transfer_consumed"]),
                "evicted_before_use": bool(request["evicted_before_use"]),
                "transfer_start_ms": event.get("transfer_start_ms"),
                "transfer_completion_ms": event.get("transfer_completion_ms"),
                "overlap_with_compute_ms": event.get("overlap_with_compute_ms"),
                "serialized_transfer_ms": event.get("serialized_transfer_ms"),
                "belady_same_capacity_hit": belady_by_object[id(request)],
                "metadata_consistent": metadata_consistent,
                "metadata_failures": metadata_failures,
                "category": category,
                "source_trace": {
                    "file": request["_source_file"],
                    "line": request["_source_line"],
                },
            }
            ledger.append(row)
            prior_transfer_ordinal[reuse_key] = transfer_ordinal
            prior_transfer_time[reuse_key] = current_time
            if bool(request["evicted_before_use"]):
                prematurely_evicted.add(reuse_key)
            elif bool(request["transfer_consumed"]):
                prematurely_evicted.discard(reuse_key)
            transfer_ordinal += 1

        if victim >= 0:
            resident.discard(victim)
        if request.get("cache_kind") == "cold":
            resident.add(expert)
        elif request.get("cache_kind") in {"resident", "inflight"} and not was_resident:
            # The classifier assigns the associated transfer to metadata
            # inconsistency; maintaining the claimed state avoids cascades.
            resident.add(expert)
        seen_experts[domain].add(expert)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        category_totals = {
            category: {"transfers": 0, "bytes": 0}
            for category in CATEGORIES
        }
        for row in rows:
            total = category_totals[row["category"]]
            total["transfers"] += 1
            total["bytes"] += row["byte_count"]
        total_transfers = len(rows)
        total_bytes = sum(row["byte_count"] for row in rows)
        for total in category_totals.values():
            total["transfer_fraction"] = (
                total["transfers"] / total_transfers if total_transfers else 0.0
            )
            total["byte_fraction"] = (
                total["bytes"] / total_bytes if total_bytes else 0.0
            )
        fixed_avoidable_transfers = sum(
            category_totals[name]["transfers"]
            for name in FIXED_CAPACITY_AVOIDABLE_CATEGORIES
        )
        fixed_avoidable_bytes = sum(
            category_totals[name]["bytes"]
            for name in FIXED_CAPACITY_AVOIDABLE_CATEGORIES
        )
        safe_cache_reducible_transfers = sum(
            category_totals[name]["transfers"]
            for name in LARGER_SAFE_CACHE_REDUCIBLE_CATEGORIES
        )
        safe_cache_reducible_bytes = sum(
            category_totals[name]["bytes"]
            for name in LARGER_SAFE_CACHE_REDUCIBLE_CATEGORIES
        )
        reconciled_transfers = sum(
            item["transfers"] for item in category_totals.values()
        )
        reconciled_bytes = sum(
            item["bytes"] for item in category_totals.values()
        )
        if (
            reconciled_transfers != total_transfers
            or reconciled_bytes != total_bytes
        ):
            raise AssertionError("category totals do not reconcile")
        return {
            "totals": {
                "transfers": total_transfers,
                "bytes": total_bytes,
                "classified_transfers": reconciled_transfers,
                "classified_bytes": reconciled_bytes,
                "avoidable_transfers": fixed_avoidable_transfers,
                "avoidable_bytes": fixed_avoidable_bytes,
                "avoidable_transfer_fraction": (
                    fixed_avoidable_transfers / total_transfers
                    if total_transfers else 0.0
                ),
                "avoidable_byte_fraction": (
                    fixed_avoidable_bytes / total_bytes if total_bytes else 0.0
                ),
                "safe_cache_reducible_transfers": safe_cache_reducible_transfers,
                "safe_cache_reducible_bytes": safe_cache_reducible_bytes,
                "safe_cache_reducible_transfer_fraction": (
                    safe_cache_reducible_transfers / total_transfers
                    if total_transfers else 0.0
                ),
                "safe_cache_reducible_byte_fraction": (
                    safe_cache_reducible_bytes / total_bytes
                    if total_bytes else 0.0
                ),
            },
            "categories": category_totals,
        }

    aggregate = summarize(ledger)
    phase_summaries = {
        phase: summarize([row for row in ledger if row["phase"] == phase])
        for phase in ("warmup", "measured")
    }
    total_transfers = aggregate["totals"]["transfers"]
    for phase in ("warmup", "measured"):
        phase_transfer_events = [
            row for row in transfer_events if row["_ledger_phase"] == phase
        ]
        phase_launched = [
            row for row in launched if row["_ledger_phase"] == phase
        ]
        if len(phase_transfer_events) != len(phase_launched) and not (
            phase == "warmup" and allow_missing_warmup_timing
        ):
            raise ValueError(
                f"{phase} expert transfer timing mismatch: "
                f"{len(phase_transfer_events)} events, "
                f"{len(phase_launched)} launched requests"
            )
    matched_timing = sum(row["transfer_timing_observed"] for row in ledger)
    coverage["required_fields_complete"] = coverage["complete"]
    coverage["timing_complete"] = matched_timing == total_transfers
    coverage["complete"] = (
        coverage["required_fields_complete"] and coverage["timing_complete"]
    )
    coverage["transfer_timing"] = {
        "observed": matched_timing,
        "launched": total_transfers,
        "fraction": matched_timing / total_transfers if total_transfers else None,
        "by_phase": {
            phase: {
                "observed": sum(
                    row["transfer_timing_observed"]
                    for row in ledger if row["phase"] == phase
                ),
                "launched": sum(
                    row["phase"] == phase for row in ledger
                ),
            }
            for phase in ("warmup", "measured")
        },
        "note": (
            "request records are authoritative for count/bytes/lifecycle; "
            + (
                "legacy warmup timing gaps were explicitly allowed"
                if allow_missing_warmup_timing
                else "all transfer timing is required"
            )
        ),
    }

    return {
        "schema_version": 1,
        "artifact": "milestone-4-transfer-ledger",
        "capacity_experts_per_layer": capacity,
        "split_layer": split_layer,
        "coverage": coverage,
        "totals": aggregate["totals"],
        "categories": aggregate["categories"],
        "by_phase": phase_summaries,
        "avoidability_semantics": {
            "avoidable": (
                "policy and implementation waste at the observed fixed capacity"
            ),
            "safe_cache_reducible": (
                "all non-compulsory traffic if a larger measured-safe cache fits"
            ),
        },
        "ledger": ledger,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-trace", required=True, type=Path)
    parser.add_argument("--measured-trace", required=True, type=Path)
    parser.add_argument("--capacity", required=True, type=int)
    parser.add_argument("--split-layer", type=int, default=20)
    parser.add_argument(
        "--allow-missing-warmup-timing",
        action="store_true",
        help="Permit legacy traces truncated by the pre-M4 CUDA timing pool.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        read_jsonl(args.warmup_trace),
        read_jsonl(args.measured_trace),
        capacity=args.capacity,
        split_layer=args.split_layer,
        allow_missing_warmup_timing=args.allow_missing_warmup_timing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "result": "PASS",
        "output": str(args.output),
        "transfers": report["totals"]["transfers"],
        "bytes": report["totals"]["bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
