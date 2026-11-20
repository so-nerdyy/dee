#!/usr/bin/env python3
"""Summarize a dee_cli --trace-requests JSON report deterministically."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def distribution(values):
    if not values:
        return {"count": 0, "average": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    return {
        "count": len(values),
        "average": statistics.fmean(values),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    document = json.loads(args.trace.read_text(encoding="utf-8"))
    benchmark = document["benchmark"]
    profile = document["profile"]
    records = profile["trace"]

    by_token = defaultdict(set)
    physical_requested = set()
    physical_loaded = set()
    kinds = Counter()
    raw_reuse = []
    distinct_reuse = []
    min_cache_bytes = []
    distinct_histogram = Counter()

    for record in records:
        key = (record["resolved_shard_layer"], record["expert"])
        by_token[record["token"]].add(key)
        physical_requested.add(key)
        kinds[record["kind"]] += 1
        if record["kind"] == "cold":
            physical_loaded.add(key)
        if record["reuse_distance"] >= 0:
            raw_reuse.append(record["reuse_distance"])
        if record["distinct_reuse_distance"] >= 0:
            distance = record["distinct_reuse_distance"]
            distinct_reuse.append(distance)
            distinct_histogram[str(distance)] += 1
            min_cache_bytes.append(record["theoretical_min_cache_bytes"])

    working_sets = [len(by_token[token]) for token in sorted(by_token)]
    repeated = len(distinct_reuse)
    resident_reuses = kinds["resident"]
    budget = benchmark["cache_budget_bytes"]
    theoretically_cacheable = sum(value <= budget for value in min_cache_bytes)

    observed_overlap = profile["derived"]["oracle_adjacent_topk_overlap"]
    random_overlap = profile["derived"]["oracle_random_overlap_expectation"]
    if observed_overlap >= random_overlap * 2.0:
        oracle_characterization = "stable"
    elif observed_overlap >= random_overlap:
        oracle_characterization = "weakly_stable"
    else:
        oracle_characterization = "less_overlap_than_random_expectation"

    report = {
        "source": str(args.trace),
        "requests": len(records),
        "classification": dict(sorted(kinds.items())),
        "unique_physical_experts_requested": len(physical_requested),
        "unique_physical_experts_loaded": len(physical_loaded),
        "working_set_per_token": {"values": working_sets, **distribution(working_sets)},
        "reuse_distance_requests": distribution(raw_reuse),
        "reuse_distance_distinct_experts": {
            **distribution(distinct_reuse),
            "histogram": dict(sorted(distinct_histogram.items(), key=lambda item: int(item[0]))),
        },
        "theoretical_minimum_cache_bytes_for_reuse": distribution(min_cache_bytes),
        "repeated_requests": repeated,
        "resident_reuses": resident_reuses,
        "reused_before_eviction_fraction": resident_reuses / repeated if repeated else 0.0,
        "reuses_theoretically_fitting_current_budget": theoretically_cacheable,
        "reuses_theoretically_fitting_current_budget_fraction": (
            theoretically_cacheable / repeated if repeated else 0.0
        ),
        "oracle": {
            "adjacent_topk_overlap": observed_overlap,
            "random_overlap_expectation": random_overlap,
            "characterization": oracle_characterization,
        },
    }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
