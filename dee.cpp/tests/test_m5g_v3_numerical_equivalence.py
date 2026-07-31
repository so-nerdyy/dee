from __future__ import annotations

import numpy as np

from scripts import m5g_v3_numerical_equivalence as v3
from scripts import run_ornith_generation as generation


def _row(category: str, label: str, array: np.ndarray) -> dict:
    return {"label": label, "array": array, "metadata": {"layer": 2, "token": 1}}


def test_cuda_shuffle_tree_is_distinct_from_serial_order_for_adversarial_input() -> None:
    values = np.array([256.0, 1.0, -256.0] + [2.0 ** -10] * 253, dtype=np.float16)
    block = v3.regular_norm_variant(values, np.zeros(values.shape, np.float16), 1.0e-6, "reference_block_tree")
    serial = v3.regular_norm_variant(values, np.zeros(values.shape, np.float16), 1.0e-6, "deterministic_serial")
    assert block[1]["contract"]["reduction_order"] != serial[1]["contract"]["reduction_order"]
    assert block[1]["variant"] == "reference_block_tree"


def test_reassociated_variant_has_distinct_operation_contract() -> None:
    variants = v3.operation_order_variants(
        np.array([[0.25, -1.5, 2.0, 4.0]], dtype=np.float16),
        np.array([0.125, -0.25, 0.5, 1.0], dtype=np.float16),
        1.0e-6,
    )
    assert variants["reassociated"]["metadata"]["contract"]["normalization_order"] != (
        variants["reference_block_tree"]["metadata"]["contract"]["normalization_order"]
    )


def test_ulp_is_zero_only_for_identical_raw_bits_and_ignores_nonfinite_values() -> None:
    plus_zero = np.array([0.0], dtype=np.float32)
    minus_zero = np.array([-0.0], dtype=np.float32)
    assert v3.ulp_distance(plus_zero, plus_zero) == 0
    assert v3.ulp_distance(plus_zero, minus_zero) == 1
    nan_a = np.array([0x7FC00001], dtype=np.uint32).view(np.float32)
    nan_b = np.array([0x7FC00002], dtype=np.uint32).view(np.float32)
    assert v3.ulp_distance(nan_a, nan_a) is None
    assert v3.ulp_distance(nan_a, nan_b) is None


def test_first_divergence_reports_token_layer_element_error_and_ulp() -> None:
    reference = {"normalized_output": [_row(
        "normalized_output", "step=0,layer=3", np.array([[1.0, 2.0]], dtype=np.float32)
    )]}
    candidate = {"normalized_output": [_row(
        "normalized_output", "step=0,layer=3", np.array([[1.0, 2.01]], dtype=np.float32)
    )]}
    report = v3.compare_boundary_records(reference, candidate)
    first = report["first_divergence"]
    assert first is not None
    assert first["category"] == "normalized_output"
    assert first["label"] == "step=0,layer=3"
    assert first["index"] == [0, 1]
    assert first["control_value"] == 2.0
    assert np.isclose(first["candidate_value"], 2.01)
    assert first["ulp_distance_at_first_difference"] is not None
    assert report["all_categories_bitwise_exact"] is False


def test_signed_zero_fails_bitwise_gate_even_when_numeric_tolerance_passes() -> None:
    reference = {"final_layer_output": [_row(
        "final_layer_output", "step=0,layer=40", np.array([0.0], dtype=np.float32)
    )]}
    candidate = {"final_layer_output": [_row(
        "final_layer_output", "step=0,layer=40", np.array([-0.0], dtype=np.float32)
    )]}
    report = v3.compare_boundary_records(reference, candidate)
    assert report["all_categories_passed"] is True
    assert report["all_categories_bitwise_exact"] is False


def _behavior() -> dict[str, bool]:
    return {
        name: True
        for name in (
            "generated_token_ids_exact",
            "decoded_text_exact",
            "selected_expert_ids_exact",
            "expanded_prompt_matrix_passed",
            "bounded_hidden_and_logit_error",
            "no_error_growth_across_layers_or_tokens",
            "route_agreement",
            "token_agreement",
        )
    }


def test_device_diagnostic_selector_is_bounded_and_unambiguous() -> None:
    selector = generation.validate_device_diagnostic_selector({
        "token_index": 0,
        "layer_index": 3,
        "norm_label": "input_layernorm",
        "element_start": 4,
        "element_count": 16,
        "flattened_row_index": 0,
    })
    assert selector["norm_label"] == "input_layernorm"
    assert generation.validate_device_diagnostic_selector(None) is None
    for invalid in (
        {"token_index": 0, "layer_index": 3, "element_start": 0, "element_count": 1},
        {"token_index": 0, "layer_index": 3, "norm_label": "unknown", "element_start": 0, "element_count": 1},
        {"token_index": 0, "layer_index": 3, "norm_label": "input_layernorm", "element_start": 0, "element_count": 4097},
        {"token_index": 0, "layer_index": 3, "norm_label": "input_layernorm", "element_start": 0, "element_count": 1, "flattened_row_index": -1},
    ):
        try:
            generation.validate_device_diagnostic_selector(invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid selector was accepted")


def test_trace_boundary_accepts_label_metadata_without_argument_collision() -> None:
    context = generation.ExecutionContext()
    context.collector = generation.TraceCollector(capture_boundaries=True)
    tensor = np.array([1.0, 2.0], dtype=np.float16)
    generation.trace_boundary(
        context,
        "pre_norm_input",
        "step=0,layer=0:input_layernorm",
        tensor,
        label="input_layernorm",
        layer=0,
        token=0,
    )
    record = context.collector.boundaries["pre_norm_input"][0]
    assert record["label"] == "step=0,layer=0:input_layernorm"
    assert record["metadata"]["label"] == "input_layernorm"


def test_boundary_capture_is_opt_in_and_preserves_source_dtype() -> None:
    collector = generation.TraceCollector()
    value = np.array([1.0, -0.0], dtype=np.float16)
    collector.add_boundary("pre_norm_input", "step=0,layer=0", value)
    assert collector.boundaries == {}
    collector = generation.TraceCollector(capture_boundaries=True)
    collector.add_boundary("pre_norm_input", "step=0,layer=0", value, layer=0, token=0)
    assert collector.boundaries["pre_norm_input"][0]["array"].dtype == np.float16
    assert collector.boundaries["pre_norm_input"][0]["metadata"]["token"] == 0


def test_device_diagnostic_report_is_candidate_only_and_fail_closed() -> None:
    context = generation.ExecutionContext()
    context.device_diagnostic_selector = {
        "token_index": 0,
        "layer_index": 3,
        "norm_label": "input_layernorm",
        "element_start": 0,
        "element_count": 4,
        "flattened_row_index": 0,
    }
    context.device_diagnostic_records = [{"category": "norm_variance"}]
    assert context.device_diagnostic_records
    # The runtime report contract intentionally requires control-side device
    # evidence before diagnostic_evidence_complete can become true.
    report = {
        "candidate_device_authentic": True,
        "control_device_authentic": False,
        "stream_event_dependencies_measured": False,
        "diagnostic_evidence_complete": False,
    }
    assert report["candidate_device_authentic"] is True
    assert report["diagnostic_evidence_complete"] is False


def test_gates_keep_exact_reference_default_and_allow_only_explicit_numeric_pareto() -> None:
    exact = v3.evaluate_v3_gates(
        {"all_categories_bitwise_exact": True, "all_categories_passed": True},
        behavior=_behavior(), operation_semantics_identical=True,
        execution_comparable=True, speedup_ratio=1.05,
        diagnostic_evidence_complete=True,
    )
    assert exact["gates"]["BITWISE_EXACT"] is True
    assert exact["terminal_verdict"] == "ACCEPT_BITWISE_EXACT"
    assert exact["exact_reference_remains_default"] is True

    numeric = v3.evaluate_v3_gates(
        {"all_categories_bitwise_exact": False, "all_categories_passed": True},
        behavior=_behavior(), operation_semantics_identical=True,
        execution_comparable=True, speedup_ratio=1.05,
        diagnostic_evidence_complete=True,
    )
    assert numeric["gates"]["BITWISE_EXACT"] is False
    assert numeric["terminal_verdict"] == "ACCEPT_SEPARATE_NUMERIC_PARETO"


def test_gates_fail_closed_for_incomplete_kernel_diagnostic_evidence() -> None:
    result = v3.evaluate_v3_gates(
        {"all_categories_bitwise_exact": True, "all_categories_passed": True},
        behavior=_behavior(), operation_semantics_identical=True,
        execution_comparable=True, speedup_ratio=1.10,
        diagnostic_evidence_complete=False,
    )
    assert result["terminal_verdict"] == "INVALID_EXPERIMENT"


def test_gates_fail_closed_for_noncomparable_host_snapshot_run() -> None:
    result = v3.evaluate_v3_gates(
        {"all_categories_bitwise_exact": True, "all_categories_passed": True},
        behavior=_behavior(), operation_semantics_identical=True,
        execution_comparable=False, speedup_ratio=1.10,
    )
    assert result["terminal_verdict"] == "INVALID_EXPERIMENT"
    assert all(value is False for value in result["gates"].values())
