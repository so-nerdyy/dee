#!/usr/bin/env python3
"""Compare hidden output and expert routes from two --trace-requests reports."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if "final_hidden" not in report or "profile" not in report:
        raise ValueError(f"{path}: not a --trace-requests benchmark report")
    return report


def routes(report: dict) -> dict[tuple[int, int], list[int]]:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for request in report["profile"].get("trace", []):
        grouped[(request["token"], request["logical_layer"])].append(
            request["expert"]
        )
    return dict(grouped)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-abs", type=float, default=0.0)
    parser.add_argument("--relative-rmse", type=float, default=0.0)
    parser.add_argument("--require-exact-routes", action="store_true")
    args = parser.parse_args()

    reference = load_report(args.reference)
    candidate = load_report(args.candidate)
    expected = [float(value) for value in reference["final_hidden"]]
    actual = [float(value) for value in candidate["final_hidden"]]
    if len(expected) != len(actual) or not expected:
        raise ValueError(
            f"hidden vector length mismatch: {len(expected)} != {len(actual)}"
        )

    squared_error = sum((lhs - rhs) ** 2 for lhs, rhs in zip(expected, actual))
    max_absolute_error = max(abs(lhs - rhs) for lhs, rhs in zip(expected, actual))
    rmse = math.sqrt(squared_error / len(expected))
    reference_rms = math.sqrt(sum(value * value for value in expected) / len(expected))
    relative_rmse = rmse / reference_rms if reference_rms else (0.0 if rmse == 0 else math.inf)
    dot = sum(lhs * rhs for lhs, rhs in zip(expected, actual))
    expected_norm = math.sqrt(sum(value * value for value in expected))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    if expected_norm and actual_norm:
        cosine = dot / (expected_norm * actual_norm)
    else:
        cosine = 1.0 if expected == actual else 0.0

    reference_routes = routes(reference)
    candidate_routes = routes(candidate)
    route_keys = sorted(set(reference_routes) | set(candidate_routes))
    ordered_matches = 0
    ordered_total = 0
    exact_sets = 0
    for key in route_keys:
        expected_route = reference_routes.get(key, [])
        actual_route = candidate_routes.get(key, [])
        ordered_total += max(len(expected_route), len(actual_route))
        ordered_matches += sum(
            lhs == rhs for lhs, rhs in zip(expected_route, actual_route)
        )
        exact_sets += set(expected_route) == set(actual_route)

    print(f"max absolute error : {max_absolute_error:.9g}")
    print(f"RMSE               : {rmse:.9g}")
    print(f"relative RMSE      : {relative_rmse:.9g}")
    print(f"cosine similarity  : {cosine:.12g}")
    print(f"ordered routes     : {ordered_matches} / {ordered_total}")
    print(f"exact top-K sets   : {exact_sets} / {len(route_keys)}")

    routes_exact = (
        ordered_matches == ordered_total
        and exact_sets == len(route_keys)
        and reference_routes.keys() == candidate_routes.keys()
    )
    valid = (
        max_absolute_error <= args.max_abs
        and relative_rmse <= args.relative_rmse
        and (not args.require_exact_routes or routes_exact)
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
