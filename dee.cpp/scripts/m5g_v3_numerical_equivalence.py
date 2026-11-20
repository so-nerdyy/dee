"""M5G-v3 numerical-equivalence primitives.

This module is deliberately independent of the sealed M5G-v1/v2 harnesses.  It
contains only deterministic analysis and reference-order emulation; callers
must provide separately captured control/candidate boundaries and behavioral
validation.  No broad post-hoc tolerance is allowed: every tensor category has
an explicit, reviewable tolerance below.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import math

import numpy as np


SCHEMA_VERSION = 1

# These are predeclared per-boundary contracts, not a global tolerance.  Inputs
# and learned parameters must remain bitwise identical.  Derived reduction
# values permit only the error expected from a float32 reduction/rsqrt boundary;
# output and routing categories are intentionally stricter than the existing
# broad end-to-end trace tolerances.
V3_TENSOR_TOLERANCES: dict[str, tuple[float, float]] = {
    "pre_norm_input": (0.0, 0.0),
    "residual_input": (0.0, 0.0),
    "norm_weight": (0.0, 0.0),
    "norm_variance": (2.0e-6, 2.0e-5),
    "reciprocal_rms": (2.0e-6, 2.0e-5),
    "normalized_output": (2.0e-4, 2.0e-4),
    "weighted_normalized": (2.0e-4, 2.0e-4),
    "gated_input": (0.0, 0.0),
    "gated_output": (2.0e-4, 2.0e-4),
    "post_residual_output": (2.0e-4, 2.0e-4),
    "router_input": (0.0, 0.0),
    "router_logits": (2.0e-4, 2.0e-4),
    "routing_weights": (2.0e-4, 2.0e-4),
    "selected_expert_ids": (0.0, 0.0),
    "expert_input": (0.0, 0.0),
    "expert_output": (5.0e-4, 5.0e-4),
    "final_layer_output": (5.0e-4, 5.0e-4),
}


_OPERATION_CONTRACTS: dict[str, dict[str, Any]] = {
    "reference_block_tree": {
        "accumulation_dtype": "float32",
        "reduction_order": "256-thread partials, warp shuffle tree, warp-0 tree",
        "epsilon_placement": "square_sum / dim + epsilon",
        "normalization_order": "input * reciprocal_rms, then (1 + weight), then multiply",
        "cast_points": ["half input -> float32", "final float32 -> half"],
    },
    "deterministic_serial": {
        "accumulation_dtype": "float32",
        "reduction_order": "strict increasing element index",
        "epsilon_placement": "square_sum / dim + epsilon",
        "normalization_order": "input * reciprocal_rms, then (1 + weight), then multiply",
        "cast_points": ["half input -> float32", "final float32 -> half"],
    },
    "reassociated": {
        "accumulation_dtype": "float32",
        "reduction_order": "vectorized/reassociated reduction",
        "epsilon_placement": "square_sum / dim + epsilon",
        "normalization_order": "input * (reciprocal_rms * (1 + weight))",
        "cast_points": ["half input -> float32", "final float32 -> half"],
    },
}


def _as_float32(values: Any) -> np.ndarray:
    return np.asarray(values).astype(np.float32, copy=False)


def _shuffle_down_tree(values: np.ndarray) -> np.float32:
    """Emulate __shfl_down_sync offsets 16,8,4,2,1 for one warp."""
    state = np.asarray(values, dtype=np.float32).copy()
    for offset in (16, 8, 4, 2, 1):
        previous = state.copy()
        state[:32 - offset] = np.float32(
            previous[:32 - offset] + previous[offset:32]
        )
    return np.float32(state[0])


def _block_tree_sum(row: np.ndarray, threads: int = 256) -> np.float32:
    """Emulate the CUDA kernel's per-thread/warp/block sum order."""
    if threads <= 0 or threads % 32:
        raise ValueError("threads must be a positive multiple of 32")
    values = _as_float32(row).reshape(-1)
    partials = np.zeros(threads, dtype=np.float32)
    for thread in range(threads):
        accumulator = np.float32(0.0)
        for index in range(thread, values.size, threads):
            value = values[index]
            accumulator = np.float32(accumulator + np.float32(value * value))
        partials[thread] = accumulator
    warp_sums = np.asarray([
        _shuffle_down_tree(partials[warp:warp + 32])
        for warp in range(0, threads, 32)
    ], dtype=np.float32)
    return _shuffle_down_tree(
        np.pad(warp_sums, (0, 32 - warp_sums.size), constant_values=0.0)
    )


def _serial_sum(row: np.ndarray) -> np.float32:
    value = np.float32(0.0)
    for item in _as_float32(row).reshape(-1):
        value = np.float32(value + np.float32(item * item))
    return value


def regular_norm_variant(
    input_values: Any,
    weight: Any,
    epsilon: float,
    variant: str = "reference_block_tree",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute an FP16-boundary regular norm using an explicit operation order.

    This is a diagnostic oracle for comparing candidate instruction order.  It
    is not silently wired into the production path.  The returned contract is
    emitted into v3 evidence so a later CUDA implementation can be audited
    against it.
    """
    if variant not in _OPERATION_CONTRACTS:
        raise ValueError(f"unknown regular norm variant: {variant}")
    x = np.asarray(input_values)
    w = np.asarray(weight)
    if x.shape[-1:] != w.shape[-1:]:
        raise ValueError(f"input/weight shape mismatch: {x.shape} vs {w.shape}")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    rows = _as_float32(x).reshape(-1, x.shape[-1])
    weight_f32 = _as_float32(w).reshape(-1)
    output = np.empty(rows.shape, dtype=np.float16)
    variance = np.empty(rows.shape[0], dtype=np.float32)
    reciprocal = np.empty(rows.shape[0], dtype=np.float32)
    for row_index, row in enumerate(rows):
        if variant == "reference_block_tree":
            square_sum = _block_tree_sum(row)
        elif variant == "deterministic_serial":
            square_sum = _serial_sum(row)
        else:
            # np.sum is intentionally kept as a visibly different diagnostic
            # order; it must never be called the exact reference.
            square_sum = np.asarray(np.sum(row * row, dtype=np.float32), dtype=np.float32)
        variance[row_index] = np.float32(square_sum / np.float32(row.size))
        reciprocal[row_index] = np.float32(
            1.0 / np.sqrt(np.float32(variance[row_index] + np.float32(epsilon)))
        )
        if variant == "reassociated":
            # Deliberately changes the multiplication grouping to model a
            # compiler-reassociated candidate: x * (rms_reciprocal * scale).
            scale = np.float32(1.0 + weight_f32)
            scaled = np.float32(row * np.float32(reciprocal[row_index] * scale))
        else:
            normalized = np.float32(row * reciprocal[row_index])
            scaled = np.float32(normalized * np.float32(1.0 + weight_f32))
        output[row_index] = scaled.astype(np.float16)
    metadata = {
        "variant": variant,
        "rows": int(rows.shape[0]),
        "dim": int(rows.shape[1]),
        "input_dtype": str(x.dtype),
        "weight_dtype": str(w.dtype),
        "output_dtype": str(output.dtype),
        "epsilon": float(epsilon),
        "contract": _OPERATION_CONTRACTS[variant],
        "variance": variance,
        "reciprocal_rms": reciprocal,
    }
    return output.reshape(x.shape), metadata


def operation_order_variants(input_values: Any, weight: Any, epsilon: float) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for name in _OPERATION_CONTRACTS:
        output, metadata = regular_norm_variant(input_values, weight, epsilon, name)
        variants[name] = {"output": output, "metadata": metadata}
    return variants


def _ordered_bits(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.float16:
        unsigned = values.view(np.uint16)
        sign = np.uint16(0x8000)
        return np.where((unsigned & sign) != 0, ~unsigned, unsigned | sign).astype(np.int64)
    if values.dtype == np.float32:
        unsigned = values.view(np.uint32)
        sign = np.uint32(0x80000000)
        return np.where((unsigned & sign) != 0, ~unsigned, unsigned | sign).astype(np.int64)
    if values.dtype == np.float64:
        unsigned = values.view(np.uint64)
        sign = np.uint64(0x8000000000000000)
        # int64 cannot represent the unsigned positive range, so use object
        # integers only for the rare float64 diagnostic.
        return np.asarray(
            [~int(bits) if bits & int(sign) else int(bits | sign) for bits in unsigned.reshape(-1)],
            dtype=object,
        ).reshape(values.shape)
    raise TypeError(f"ULP distance requires a floating dtype, got {values.dtype}")


def _bitwise_equal(reference: np.ndarray, candidate: np.ndarray) -> bool:
    return (
        reference.shape == candidate.shape
        and reference.dtype == candidate.dtype
        and reference.tobytes(order="C") == candidate.tobytes(order="C")
    )


def ulp_distance(reference: Any, candidate: Any) -> int | None:
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    if ref.shape != got.shape or ref.dtype != got.dtype:
        return None
    if not np.issubdtype(ref.dtype, np.floating):
        return None
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(got)):
        return None
    if _bitwise_equal(ref, got):
        return 0
    ref_bits = _ordered_bits(ref)
    got_bits = _ordered_bits(got)
    distances = np.asarray(
        [abs(a - b) for a, b in zip(ref_bits.reshape(-1), got_bits.reshape(-1))],
        dtype=object,
    )
    return int(max(distances, default=0))


def _json_scalar(value: Any, bits: Any) -> Any:
    numeric = float(value)
    if math.isfinite(numeric):
        return numeric
    if math.isnan(numeric):
        return {"value": "nan", "raw_bits": hex(int(bits))}
    return {"value": "-inf" if numeric < 0 else "inf", "raw_bits": hex(int(bits))}


def _first_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any] | None:
    if reference.shape != candidate.shape:
        return {
            "index": "shape",
            "control_value": None,
            "candidate_value": None,
            "absolute_error": None,
            "relative_error": None,
            "ulp_distance": None,
        }
    ref = reference.reshape(-1)
    got = candidate.reshape(-1)
    if _bitwise_equal(reference, candidate):
        return None
    if reference.dtype != candidate.dtype:
        differing = np.flatnonzero(ref != got)
    else:
        ref_bits = reference.view(np.uint8).reshape(reference.shape + (reference.dtype.itemsize,))
        got_bits = candidate.view(np.uint8).reshape(candidate.shape + (candidate.dtype.itemsize,))
        differing = np.flatnonzero(np.any(ref_bits != got_bits, axis=-1).reshape(-1))
    if differing.size == 0:
        return {"index": "dtype_or_layout", "control_value": None, "candidate_value": None}
    flat_index = int(differing[0])
    control = ref[flat_index]
    value = got[flat_index]
    absolute = float(abs(np.float64(control) - np.float64(value))) if np.isfinite(control) and np.isfinite(value) else None
    relative = (
        absolute / max(abs(float(np.float64(control))), 1.0e-12)
        if absolute is not None else None
    )
    local_ulp = ulp_distance(reference, candidate)
    if reference.dtype == candidate.dtype:
        single_ulp = ulp_distance(ref[flat_index:flat_index + 1], got[flat_index:flat_index + 1])
        ref_bits_scalar = reference.reshape(-1).view(np.uint8).reshape(-1, reference.dtype.itemsize)[flat_index]
        got_bits_scalar = candidate.reshape(-1).view(np.uint8).reshape(-1, candidate.dtype.itemsize)[flat_index]
        ref_bits_value = int.from_bytes(bytes(ref_bits_scalar), "little")
        got_bits_value = int.from_bytes(bytes(got_bits_scalar), "little")
    else:
        single_ulp = None
        ref_bits_value = got_bits_value = None
    return {
        "index": list(np.unravel_index(flat_index, reference.shape)),
        "flat_index": flat_index,
        "control_value": _json_scalar(control, ref_bits_value or 0),
        "candidate_value": _json_scalar(value, got_bits_value or 0),
        "absolute_error": absolute,
        "relative_error": relative,
        "ulp_distance_at_first_difference": single_ulp,
        "maximum_ulp_distance": local_ulp,
    }


def compare_boundary_records(
    reference: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    tolerances: Mapping[str, tuple[float, float]] = V3_TENSOR_TOLERANCES,
) -> dict[str, Any]:
    """Compare boundary records and report the first token/layer/tensor mismatch."""
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "categories": {}}
    for category in sorted(set(reference) | set(candidate)):
        ref_rows = list(reference.get(category, ()))
        got_rows = list(candidate.get(category, ()))
        atol, rtol = tolerances.get(category, (0.0, 0.0))
        category_report: dict[str, Any] = {
            "absolute_tolerance": atol,
            "relative_tolerance": rtol,
            "reference_record_count": len(ref_rows),
            "candidate_record_count": len(got_rows),
            "passed": True,
            "bitwise_exact": True,
            "first_divergence": None,
            "operation_summaries": [],
        }
        if len(ref_rows) != len(got_rows):
            category_report.update({"passed": False, "bitwise_exact": False, "first_divergence": {"index": "record_sequence"}})
            report["categories"][category] = category_report
            continue
        for index, (ref_row, got_row) in enumerate(zip(ref_rows, got_rows)):
            label = ref_row.get("label")
            if label != got_row.get("label"):
                category_report.update({
                    "passed": False,
                    "bitwise_exact": False,
                    "first_divergence": {
                        "index": "label",
                        "control_label": label,
                        "candidate_label": got_row.get("label"),
                    },
                })
                break
            ref_array = np.asarray(ref_row["array"])
            got_array = np.asarray(got_row["array"])
            local_diff = _first_difference(ref_array, got_array)
            exact = local_diff is None
            if not exact:
                category_report["bitwise_exact"] = False
                # A bitwise mismatch is a first divergence even when the
                # predeclared numeric tolerance still passes (for example
                # +0.0 versus -0.0).  Exactness and numeric acceptance are
                # intentionally reported independently.
                if category_report["first_divergence"] is None:
                    category_report["first_divergence"] = {
                        "record_index": index,
                        "label": label,
                        **(local_diff or {"index": "shape"}),
                        "reference_dtype": str(ref_array.dtype),
                        "candidate_dtype": str(got_array.dtype),
                        "control_metadata": ref_row.get("metadata", {}),
                        "candidate_metadata": got_row.get("metadata", {}),
                    }
            if ref_array.shape != got_array.shape:
                local_passed = False
                comparison = {"shape": [list(ref_array.shape), list(got_array.shape)]}
            else:
                absolute = np.abs(ref_array.astype(np.float64) - got_array.astype(np.float64))
                allowed = atol + rtol * np.abs(ref_array.astype(np.float64))
                local_passed = bool(np.all(absolute <= allowed))
                comparison = {
                    "maximum_absolute_error": float(absolute.max(initial=0.0)),
                    "maximum_relative_error": float((absolute / np.maximum(np.abs(ref_array.astype(np.float64)), 1.0e-12)).max(initial=0.0)),
                    "maximum_ulp_distance": ulp_distance(ref_array, got_array),
                }
            category_report["operation_summaries"].append({
                "index": index,
                "label": label,
                "passed": local_passed,
                "bitwise_exact": exact,
                **comparison,
            })
            if not local_passed:
                category_report["passed"] = False
        report["categories"][category] = category_report
    report["all_categories_passed"] = all(row["passed"] for row in report["categories"].values()) and bool(report["categories"])
    report["all_categories_bitwise_exact"] = all(row["bitwise_exact"] for row in report["categories"].values()) and bool(report["categories"])
    first = []
    for category, row in report["categories"].items():
        if row["first_divergence"] is not None:
            first.append((row["first_divergence"].get("record_index", 0), category, row["first_divergence"]))
    report["first_divergence"] = min(first, default=None, key=lambda item: item[0])
    if report["first_divergence"] is not None:
        _, category, detail = report["first_divergence"]
        report["first_divergence"] = {"category": category, **detail}
    return report


def evaluate_v3_gates(
    boundary_comparison: Mapping[str, Any],
    *,
    behavior: Mapping[str, Any],
    operation_semantics_identical: bool,
    execution_comparable: bool,
    speedup_ratio: float | None,
    diagnostic_evidence_complete: bool = False,
) -> dict[str, Any]:
    """Evaluate the three named gates and map them to a terminal verdict."""
    bitwise = bool(boundary_comparison.get("all_categories_bitwise_exact"))
    numeric = bool(boundary_comparison.get("all_categories_passed"))
    behavior_required = (
        "generated_token_ids_exact",
        "decoded_text_exact",
        "selected_expert_ids_exact",
        "expanded_prompt_matrix_passed",
        "bounded_hidden_and_logit_error",
        "no_error_growth_across_layers_or_tokens",
        "route_agreement",
        "token_agreement",
    )
    behavior_gate = all(behavior.get(name) is True for name in behavior_required)
    diagnostic_ok = bool(diagnostic_evidence_complete)
    gates = {
        "BITWISE_EXACT": bitwise and operation_semantics_identical and execution_comparable and diagnostic_ok,
        "NUMERICALLY_EQUIVALENT": numeric and operation_semantics_identical and execution_comparable and diagnostic_ok,
        "MODEL_BEHAVIOR_EQUIVALENT": behavior_gate and execution_comparable and diagnostic_ok,
    }
    if not execution_comparable or not diagnostic_evidence_complete:
        verdict = "INVALID_EXPERIMENT"
    elif not operation_semantics_identical:
        verdict = "REJECT_SEMANTIC_MISMATCH"
    elif gates["BITWISE_EXACT"]:
        verdict = "ACCEPT_BITWISE_EXACT" if speedup_ratio is not None and speedup_ratio > 1.0 else "REJECT_PERFORMANCE"
    elif gates["NUMERICALLY_EQUIVALENT"] and gates["MODEL_BEHAVIOR_EQUIVALENT"]:
        verdict = "ACCEPT_SEPARATE_NUMERIC_PARETO"
    elif not numeric:
        verdict = "REJECT_NUMERICAL_INSTABILITY"
    else:
        verdict = "REJECT_SEMANTIC_MISMATCH"
    return {
        "schema_version": SCHEMA_VERSION,
        "gates": gates,
        "terminal_verdict": verdict,
        "speedup_ratio": speedup_ratio,
        "diagnostic_evidence_complete": diagnostic_evidence_complete,
        "exact_reference_remains_default": True,
        "required_behavior_fields": list(behavior_required),
    }
