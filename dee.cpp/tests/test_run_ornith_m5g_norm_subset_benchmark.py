from __future__ import annotations

from scripts import run_ornith_m5g_norm_subset_benchmark as benchmark


def test_balanced_order_builds_three_candidate_control_pairs() -> None:
    for candidate in benchmark.CANDIDATES:
        order = benchmark.balanced_order(candidate, 3)
        assert order == [
            benchmark.CONTROL,
            candidate,
            candidate,
            benchmark.CONTROL,
            benchmark.CONTROL,
            candidate,
        ]
        assert all(
            set(order[index:index + 2])
            == {benchmark.CONTROL, candidate}
            for index in range(0, len(order), 2)
        )


def test_balanced_order_rejects_unknown_candidate() -> None:
    try:
        benchmark.balanced_order("unknown", 3)
    except ValueError as exc:
        assert "unknown norm subset candidate" in str(exc)
    else:
        raise AssertionError("unknown candidate did not fail closed")


def test_expected_norm_calls_keep_subsets_disjoint() -> None:
    regular = benchmark.expected_norm_calls(benchmark.REGULAR)
    gated = benchmark.expected_norm_calls(benchmark.GATED)
    assert regular == {
        "regular": (
            benchmark.EXPECTED_REGULAR_WRAPPERS
            * len(benchmark.EXPECTED_TOKENS)
        ),
        "gated": 0,
        "allocations": (
            benchmark.EXPECTED_REGULAR_WRAPPERS
            * len(benchmark.EXPECTED_TOKENS)
        ),
    }
    assert gated == {
        "regular": 0,
        "gated": (
            benchmark.EXPECTED_GATED_WRAPPERS
            * len(benchmark.EXPECTED_TOKENS)
        ),
        "allocations": (
            benchmark.EXPECTED_GATED_WRAPPERS
            * len(benchmark.EXPECTED_TOKENS)
        ),
    }


def test_paired_analysis_uses_candidate_over_control() -> None:
    sequence = [
        {
            "execution_mode": benchmark.CONTROL,
            "tokens_per_second": 8.0,
        },
        {
            "execution_mode": benchmark.GATED,
            "tokens_per_second": 10.0,
        },
        {
            "execution_mode": benchmark.GATED,
            "tokens_per_second": 9.0,
        },
        {
            "execution_mode": benchmark.CONTROL,
            "tokens_per_second": 8.0,
        },
    ]
    report = benchmark.paired_trial_analysis(benchmark.GATED, sequence)
    assert report["pair_count"] == 2
    assert report["candidate_wins"] == 2
    assert report["pairs"][0]["speedup_ratio"] == 1.25
    assert report["pairs"][1]["speedup_ratio"] == 1.125
