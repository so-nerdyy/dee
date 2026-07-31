#!/usr/bin/env python3
"""Minimal authoritative M5G-v3 regular-norm CUDA diagnostic smoke run.

This driver is intentionally separate from the sealed M5G-v2 matrix. It runs
one untouched HF control generation and one regular-norm candidate generation
in the same loaded runtime, with one bounded diagnostic selector. Timing is
reported only as non-comparable diagnostic metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ornith_generation import (  # noqa: E402
    FUSED_REGULAR_NORM_EXECUTION_MODE,
    git_revision,
    load_runtime,
    run_generation,
)

RUN_ID = "20260730T235000Z-m5g-v3-regular-norm-smoke"
PROMPT = "Hello"
MAX_NEW_TOKENS = 1
NORM_LABEL = "input_layernorm"
LAYER_INDEX = 0
TOKEN_INDEX = 0
ELEMENT_START = 0
ELEMENT_COUNT = 8
EXPECTED_CATEGORIES = {
    "pre_norm_input",
    "norm_variance",
    "norm_denominator",
    "reciprocal_rms",
    "norm_weight",
    "normalized_output",
    "post_norm_output",
}
EXPECTED_KERNELS = {
    "control": "qwen_rms_norm_fp16_reference_diagnostic",
    "candidate": "qwen_rms_norm_fp16_diagnostic",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def selector() -> dict[str, Any]:
    return {
        "token_index": TOKEN_INDEX,
        "layer_index": LAYER_INDEX,
        "norm_label": NORM_LABEL,
        "element_start": ELEMENT_START,
        "element_count": ELEMENT_COUNT,
        "flattened_row_index": 0,
    }


def json_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep only JSON-safe generation fields; collector objects are excluded."""
    return {key: value for key, value in result.items() if key != "collector"}


def selected_records(result: dict[str, Any], side: str) -> list[dict[str, Any]]:
    records = result.get("device_diagnostic", {}).get("records", [])
    return [
        record for record in records
        if record.get("metadata", {}).get("side") == side
    ]


def record_bits(record: dict[str, Any]) -> list[list[str]]:
    values = np.asarray(record.get("array", []), dtype=np.float32)
    if values.size == 0:
        return []
    bits = values.view(np.uint32).reshape(-1)
    return [[hex(int(item)) for item in bits]]


def record_summary(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    return {
        "category": record.get("category"),
        "label": record.get("label"),
        "array": record.get("array"),
        "bits_float32": record_bits(record),
        "metadata": metadata,
    }


def validate_side(
    records: list[dict[str, Any]],
    side: str,
    expected: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    categories = [record.get("category") for record in records]
    if set(categories) != EXPECTED_CATEGORIES or len(records) != len(EXPECTED_CATEGORIES):
        failures.append({
            "name": f"{side}_diagnostic_categories_complete",
            "details": categories,
        })
    if len(categories) != len(set(categories)):
        failures.append({
            "name": f"{side}_diagnostic_records_unique",
            "details": categories,
        })
    for record in records:
        metadata = record.get("metadata", {})
        checks = {
            "side": metadata.get("side") == side,
            "layer": metadata.get("layer") == expected["layer_index"],
            "token": metadata.get("token") == expected["token_index"],
            "label": metadata.get("label") == expected["norm_label"],
            "selector": metadata.get("selector") == expected,
            "element_start": metadata.get("element_start") == expected["element_start"],
            "element_count": metadata.get("element_count") == expected["element_count"],
            "completion_sequence": int(metadata.get("completion_sequence", 0)) > 0,
            "kernel_identity": metadata.get("kernel_identity") == EXPECTED_KERNELS[side],
            "stream_id": (
                "stream_id" in metadata
                and isinstance(metadata.get("stream_id"), int)
                and not isinstance(metadata.get("stream_id"), bool)
                and metadata["stream_id"] >= 0
            ),
            "epsilon": math.isfinite(float(metadata.get("epsilon", float("nan")))),
            "source_dtype": metadata.get("source_dtype") == "torch.float16",
            "destination_dtype": metadata.get("destination_dtype") == "torch.float16",
        }
        for name, passed in checks.items():
            if not passed:
                failures.append({"name": f"{side}_{name}", "details": metadata})
    return failures


def compare_records(
    control_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    control_by_category = {record["category"]: record for record in control_records}
    candidate_by_category = {record["category"]: record for record in candidate_records}
    comparison: dict[str, Any] = {}
    for category in sorted(EXPECTED_CATEGORIES):
        control_values = np.asarray(
            control_by_category.get(category, {}).get("array", []), dtype=np.float32
        )
        candidate_values = np.asarray(
            candidate_by_category.get(category, {}).get("array", []), dtype=np.float32
        )
        same_shape = control_values.shape == candidate_values.shape
        comparison[category] = {
            "shape_equal": same_shape,
            "bitwise_equal": bool(
                same_shape and control_values.tobytes() == candidate_values.tobytes()
            ),
            "control": (
                record_summary(control_by_category[category])
                if category in control_by_category else None
            ),
            "candidate": (
                record_summary(candidate_by_category[category])
                if category in candidate_by_category else None
            ),
        }
    return comparison


def build_report(
    args: argparse.Namespace,
    *,
    actual_runtime_sha: str,
    failures: list[dict[str, Any]],
    control: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    exception: dict[str, Any] | None = None,
) -> dict[str, Any]:
    control = control or {}
    candidate = candidate or {}
    control_records = selected_records(control, "control")
    candidate_records = selected_records(candidate, "candidate")
    control_diag = control.get("device_diagnostic", {})
    candidate_diag = candidate.get("device_diagnostic", {})
    comparison = compare_records(control_records, candidate_records)
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "smoke_gates_passed": not failures and exception is None,
        "result": "PASS" if not failures and exception is None else "FAIL",
        "terminal_verdict": "INVALID_EXPERIMENT",
        "validation_failures": failures,
        "exception": exception,
        "runtime_sha": args.runtime_sha,
        "harness_sha": args.harness_sha,
        "actual_runtime_sha": actual_runtime_sha,
        "model_dir": str(args.model_dir),
        "configuration": {
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "selector": selector(),
            "norm_label": NORM_LABEL,
            "layer_index": LAYER_INDEX,
            "token_index": TOKEN_INDEX,
            "flattened_row_index": 0,
            "element_start": ELEMENT_START,
            "element_count": ELEMENT_COUNT,
            "performance_comparable": False,
        },
        "untouched_hf_control": json_result(control),
        "control_probe_output": next(
            (record_summary(record) for record in control_records
             if record.get("category") == "post_norm_output"),
            None,
        ),
        "regular_norm_candidate": json_result(candidate),
        "candidate_device_records": [record_summary(record) for record in candidate_records],
        "control_device_records": [record_summary(record) for record in control_records],
        "bounded_intermediate_comparison": comparison,
        "residual_provenance": candidate_diag.get("residual_provenance_records", []),
        "dependency_records": candidate_diag.get("event_graph_records", []),
        "control_probe_output_validated": control_diag.get(
            "control_probe_output_validated", False
        ),
        "candidate_device_authentic": candidate_diag.get(
            "candidate_device_authentic", False
        ),
        "classification_reason": (
            "Smoke-only diagnostic; internal HF control reduction and "
            "producer/event provenance remain unproven."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-sha", required=True)
    parser.add_argument("--harness-sha", required=True)
    parser.add_argument("--split-layer", type=int, default=20)
    parser.add_argument("--cache-experts", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, Any]] = []
    actual_runtime_sha = git_revision()
    control: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    runtime = None
    exception: dict[str, Any] | None = None
    expected = selector()

    try:
        import torch
        from transformers import AutoTokenizer

        if not args.runtime_sha or len(args.runtime_sha) != 40:
            failures.append({"name": "runtime_sha_format", "details": args.runtime_sha})
        if not args.harness_sha:
            failures.append({"name": "harness_sha_present", "details": args.harness_sha})
        if actual_runtime_sha != args.runtime_sha:
            failures.append({
                "name": "runtime_sha_exact",
                "details": [actual_runtime_sha, args.runtime_sha],
            })
        if not args.model_dir.is_dir():
            failures.append({"name": "model_directory_present", "details": str(args.model_dir)})
        if torch.cuda.device_count() != 2:
            failures.append({"name": "dual_gpu_topology", "details": torch.cuda.device_count()})
        else:
            for device in range(2):
                name = torch.cuda.get_device_name(device)
                if "T4" not in name:
                    failures.append({"name": "dual_t4_topology", "details": name})
        if failures:
            return 1

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        runtime = load_runtime(
            args.model_dir,
            2,
            args.split_layer,
            args.cache_experts,
            router_backend="torch-device",
        )
        control = run_generation(
            runtime,
            tokenizer,
            PROMPT,
            MAX_NEW_TOKENS,
            False,
            "reference",
            trace=False,
            execution_mode="production",
            device_diagnostic=expected,
        )
        candidate = run_generation(
            runtime,
            tokenizer,
            PROMPT,
            MAX_NEW_TOKENS,
            False,
            "dee",
            trace=False,
            execution_mode=FUSED_REGULAR_NORM_EXECUTION_MODE,
            device_diagnostic=expected,
        )

        control_records = selected_records(control, "control")
        candidate_records = selected_records(candidate, "candidate")
        failures.extend(validate_side(control_records, "control", expected))
        failures.extend(validate_side(candidate_records, "candidate", expected))
        control_diag = control.get("device_diagnostic", {})
        candidate_diag = candidate.get("device_diagnostic", {})
        if control_diag.get("control_probe_output_validated") is not True:
            failures.append({"name": "control_probe_bitwise_match", "details": control_diag})
        if candidate_diag.get("candidate_device_authentic") is not True:
            failures.append({"name": "candidate_device_diagnostics_present", "details": candidate_diag})
        if candidate_diag.get("candidate_stream_completion_observed") is not True:
            failures.append({"name": "candidate_completion_observed", "details": candidate_diag})
        if not candidate_diag.get("residual_provenance_records"):
            failures.append({"name": "candidate_residual_provenance_record", "details": candidate_diag})
        if not candidate_diag.get("event_graph_records"):
            failures.append({"name": "candidate_dependency_record", "details": candidate_diag})
        if control.get("generated_token_ids") != candidate.get("generated_token_ids"):
            failures.append({"name": "generated_token_ids_exact", "details": [control.get("generated_token_ids"), candidate.get("generated_token_ids")]})
        if control.get("generated_text") != candidate.get("generated_text"):
            failures.append({"name": "generated_text_exact", "details": [control.get("generated_text"), candidate.get("generated_text")]})
    except Exception as exc:
        exception = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        failures.append({"name": "smoke_execution_completed", "details": exception})
    finally:
        if runtime is not None:
            try:
                runtime["pool"].close()
            except Exception as exc:
                failures.append({"name": "runtime_cleanup", "details": str(exc)})
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        report = build_report(
            args,
            actual_runtime_sha=actual_runtime_sha,
            failures=failures,
            control=control,
            candidate=candidate,
            exception=exception,
        )
        write_json(args.output_dir / "smoke-report.json", report)
        write_json(args.output_dir / "smoke-status.json", {
            "run_id": RUN_ID,
            "result": report["result"],
            "terminal_verdict": report["terminal_verdict"],
            "smoke_gates_passed": report["smoke_gates_passed"],
            "validation_failure_count": len(failures),
        })
    return 0 if not failures and exception is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
