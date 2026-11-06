#!/usr/bin/env python3
"""Replay observed expert requests to bound cache-capacity and policy headroom."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable

from analyze_milestone4_transfer_ledger import belady_hits, read_jsonl


def _requests(events: Iterable[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    return [
        {**row, "_bound_phase": phase}
        for row in events
        if row.get("event_type") == "expert_request"
    ]


def _domain(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["resolved_shard_layer"]), str(row["gpu_destination"])


def _lru_hits(requests: list[dict[str, Any]], capacity: int) -> list[bool]:
    resident: collections.OrderedDict[int, None] = collections.OrderedDict()
    result = []
    for row in requests:
        expert = int(row["expert_id"])
        hit = expert in resident
        result.append(hit)
        if hit:
            resident.move_to_end(expert)
            continue
        if len(resident) >= capacity:
            resident.popitem(last=False)
        resident[expert] = None
    return result


def _unlimited_hits(requests: list[dict[str, Any]]) -> list[bool]:
    resident: set[int] = set()
    result = []
    for row in requests:
        expert = int(row["expert_id"])
        result.append(expert in resident)
        resident.add(expert)
    return result


def _measured_summary(
    requests: list[dict[str, Any]],
    hits: list[bool],
) -> dict[str, Any]:
    measured = [
        (row, hit)
        for row, hit in zip(requests, hits, strict=True)
        if row["_bound_phase"] == "measured"
    ]
    hit_count = sum(hit for _, hit in measured)
    miss_rows = [row for row, hit in measured if not hit]
    source_bytes = sum(int(row.get("source_bytes", 0) or 0) for row in miss_rows)
    return {
        "requests": len(measured),
        "hits": hit_count,
        "misses": len(miss_rows),
        "hit_rate": hit_count / len(measured) if measured else None,
        "estimated_h2d_bytes": source_bytes,
    }


def analyze(
    warmup_events: Iterable[dict[str, Any]],
    measured_events: Iterable[dict[str, Any]],
    *,
    capacities: Iterable[int] = (8, 16, 32),
) -> dict[str, Any]:
    warmup = _requests(warmup_events, "warmup")
    measured = _requests(measured_events, "measured")
    requests = [*warmup, *measured]
    if not measured:
        raise ValueError("measured trace contains no expert requests")
    required = (
        "resolved_shard_layer",
        "gpu_destination",
        "expert_id",
        "source_bytes",
    )
    incomplete = [
        field for field in required
        if any(row.get(field) is None for row in requests)
    ]
    if incomplete:
        raise ValueError(
            "capacity-bound coverage is incomplete: " + ", ".join(incomplete)
        )

    by_domain: dict[tuple[int, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in requests:
        by_domain[_domain(row)].append(row)

    actual_measured_hits = [
        row.get("cache_result") == "hit" for row in measured
    ]
    actual = _measured_summary(measured, actual_measured_hits)
    actual["launched_transfers"] = sum(
        bool(row.get("transfer_launched", row.get("cache_result") == "miss"))
        for row in measured
    )
    actual["observed_h2d_bytes"] = sum(
        int(row.get("source_bytes", 0) or 0)
        for row in measured
        if bool(row.get(
            "transfer_launched", row.get("cache_result") == "miss"
        ))
    )

    capacity_rows = []
    for capacity in capacities:
        if capacity <= 0:
            raise ValueError("capacities must be positive")
        policies: dict[str, dict[str, Any]] = {}
        for name, simulator in (
            ("lru", _lru_hits),
            ("belady_offline", belady_hits),
        ):
            combined = {
                id(row): hit
                for domain_rows in by_domain.values()
                for row, hit in zip(
                    domain_rows,
                    simulator(domain_rows, capacity),
                    strict=True,
                )
            }
            summary = _measured_summary(
                requests, [combined[id(row)] for row in requests]
            )
            summary["additional_hits_vs_actual"] = (
                summary["hits"] - actual["hits"]
            )
            summary["avoided_h2d_bytes_vs_actual"] = (
                actual["observed_h2d_bytes"] - summary["estimated_h2d_bytes"]
            )
            policies[name] = summary
        capacity_rows.append({
            "capacity_experts_per_layer": capacity,
            "policies": policies,
            "policy_headroom_hits": (
                policies["belady_offline"]["hits"] - policies["lru"]["hits"]
            ),
            "policy_headroom_bytes": (
                policies["lru"]["estimated_h2d_bytes"]
                - policies["belady_offline"]["estimated_h2d_bytes"]
            ),
        })

    unlimited_by_id = {
        id(row): hit
        for domain_rows in by_domain.values()
        for row, hit in zip(
            domain_rows, _unlimited_hits(domain_rows), strict=True
        )
    }
    causal_unlimited = _measured_summary(
        requests, [unlimited_by_id[id(row)] for row in requests]
    )
    measured_bytes = sum(int(row["source_bytes"]) for row in measured)
    preloaded_perfect = {
        "requests": len(measured),
        "hits": len(measured),
        "misses": 0,
        "hit_rate": 1.0,
        "estimated_h2d_bytes": 0,
        "assumption": "all observed measured experts resident before timing",
    }
    return {
        "schema_version": 1,
        "artifact": "milestone-4-capacity-bounds",
        "order_basis": "native per-layer request order; warmup precedes measured",
        "domains": len(by_domain),
        "warmup_requests": len(warmup),
        "measured_requests": len(measured),
        "measured_requested_source_bytes": measured_bytes,
        "actual": actual,
        "capacities": capacity_rows,
        "unlimited_causal_cache": causal_unlimited,
        "preloaded_perfect_cache": preloaded_perfect,
        "limitations": [
            "Belady is an offline upper bound and is not deployable.",
            "Byte estimates assume each simulated miss transfers source_bytes.",
            "This artifact bounds traffic, not wall time or decode throughput.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-trace", required=True, type=Path)
    parser.add_argument("--measured-trace", required=True, type=Path)
    parser.add_argument("--capacities", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        read_jsonl(args.warmup_trace),
        read_jsonl(args.measured_trace),
        capacities=args.capacities,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "result": "PASS",
        "output": str(args.output),
        "measured_requests": report["measured_requests"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
