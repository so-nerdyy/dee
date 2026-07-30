from types import SimpleNamespace

import pytest

from scripts.run_ornith_m5h_fla_recurrent_benchmark import (
    CANDIDATE,
    CONTROL,
    CountedBackend,
    balanced_order,
    expected_recurrent_calls,
    paired_trial_analysis,
    select_backend,
)


def test_balanced_order_has_three_balanced_pairs():
    order = balanced_order(3)
    assert order == [
        CONTROL,
        CANDIDATE,
        CANDIDATE,
        CONTROL,
        CONTROL,
        CANDIDATE,
    ]
    assert order.count(CONTROL) == 3
    assert order.count(CANDIDATE) == 3


def test_paired_trial_analysis_preserves_order_and_ratios():
    sequence = [
        {"execution_mode": CONTROL, "tokens_per_second": 10.0},
        {"execution_mode": CANDIDATE, "tokens_per_second": 11.0},
        {"execution_mode": CANDIDATE, "tokens_per_second": 12.0},
        {"execution_mode": CONTROL, "tokens_per_second": 10.0},
    ]
    result = paired_trial_analysis(sequence)
    assert result["pair_count"] == 2
    assert result["candidate_wins"] == 2
    assert result["minimum_speedup_ratio"] == pytest.approx(1.1)
    assert result["maximum_speedup_ratio"] == pytest.approx(1.2)
    assert result["pairs"][1]["first_mode"] == CANDIDATE


def test_paired_trial_analysis_rejects_malformed_pair():
    with pytest.raises(ValueError, match="does not contain"):
        paired_trial_analysis([
            {"execution_mode": CONTROL, "tokens_per_second": 10.0},
            {"execution_mode": CONTROL, "tokens_per_second": 10.0},
        ])


def test_expected_recurrent_calls_excludes_prefill():
    assert expected_recurrent_calls(4) == 90
    with pytest.raises(ValueError, match="positive"):
        expected_recurrent_calls(0)


def test_counted_backend_and_layer_selection():
    def fallback(value):
        return value + 1

    fallback.__module__ = "transformers.fake"
    backend = CountedBackend(CONTROL, fallback)
    modules = [
        SimpleNamespace(
            recurrent_gated_delta_rule=None,
            chunk_gated_delta_rule=fallback,
            causal_conv1d_update=fallback,
        )
        for _ in range(30)
    ]
    selection = select_backend(modules, backend)
    assert selection["layer_count"] == 30
    assert selection["all_layers_selected"]
    assert selection["backend"]["module"] == "transformers.fake"
    assert backend(4) == 5
    assert backend.calls == 1

