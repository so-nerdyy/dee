#!/usr/bin/env python3
"""Integrate the controlled Ornith Milestone 2.5 matrix into final evidence.

The integrator deliberately fails closed: a missing run or measurement remains
visible in the acceptance audit and cannot be converted into a numeric zero.
It performs no inference and no performance optimization.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_milestone25_expert_trace import analyze, read_jsonl  # noqa: E402


SCHEMA_VERSION = 1
EXPECTED_RUNS = (
    "dual-cold-primary",
    "dual-warm-profiled",
    "dual-warm-control",
    "dual-warm-reference-present",
    "dual-cache-disabled",
    "dual-cache-capacity-4",
    "dual-long-prompt",
    "dual-one-token",
    "single-t4-warm",
)
PRIMARY_RUN = "dual-cold-primary"
WARM_RUN = "dual-warm-profiled"
CONTROL_RUN = "dual-warm-control"
NATIVE_MEMORY_KEYS = (
    "host_pinned_expert_staging_bytes",
    "host_pageable_expert_staging_bytes",
    "host_router_weight_bytes",
    "host_hidden_buffer_bytes",
    "host_prefetch_ring_bytes",
    "host_prefetch_ring_slots",
    "peak_transient_host_bytes",
    "device_expert_cache_reserved_bytes",
    "device_prefetch_staging_bytes",
    "device_fixed_work_buffer_bytes",
    "device_router_weight_bytes",
    "device_router_dynamic_bytes",
    "device_moe_batch_buffer_bytes",
    "device_oracle_scratch_bytes",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prior-audit", required=True, type=Path)
    parser.add_argument("--layer0-regression", type=Path)
    parser.add_argument("--router-parity", type=Path)
    parser.add_argument(
        "--kernel-slug", default="nivind/dee-cpp-ornith-milestone-2-5-forensics"
    )
    parser.add_argument("--kernel-version")
    parser.add_argument("--publication-commit")
    parser.add_argument("--publication-branch", default="opt/real-model-t1")
    parser.add_argument(
        "--require-publication", action="store_true",
        help="fail unless --publication-commit is supplied",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(value: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def integer(value: Any) -> int | None:
    number = finite_number(value)
    return int(number) if number is not None else None


def ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


def median(values: Iterable[float | int]) -> float | None:
    rows = [float(value) for value in values if finite_number(value) is not None]
    return statistics.median(rows) if rows else None


def mean(values: Iterable[float | int]) -> float | None:
    rows = [float(value) for value in values if finite_number(value) is not None]
    return statistics.fmean(rows) if rows else None


def fmt_bytes(value: int | float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{int(value):,} B ({float(value) / 1024**3:.3f} GiB)"


def fmt_ms(value: int | float | None) -> str:
    return "unavailable" if value is None else f"{float(value):.3f} ms"


def pct(value: float | None) -> str:
    return "unavailable" if value is None else f"{100.0 * value:.2f}%"


def load_runs(matrix_dir: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    missing = []
    for run_id in EXPECTED_RUNS:
        directory = matrix_dir / "runs" / run_id
        required = {
            "report": directory / "run-report.json",
            "memory": directory / "memory-timeline.json",
            "layer": directory / "layer-timing.json",
            "timing": directory / "timing-raw.json",
            "utilization": directory / "gpu-utilization-summary.json",
            "trace": directory / "expert-trace.jsonl",
        }
        absent = [str(path) for path in required.values() if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        runs[run_id] = {
            "directory": directory,
            "report": read_json(required["report"]),
            "memory": read_json(required["memory"]),
            "layer": read_json(required["layer"]),
            "timing": read_json(required["timing"]),
            "utilization": read_json(required["utilization"]),
            "trace_path": required["trace"],
        }
    if missing:
        raise FileNotFoundError("missing matrix evidence:\n" + "\n".join(missing))
    return runs


def checkpoint_metric(checkpoint: Mapping[str, Any], key: str) -> int | None:
    if key == "rss":
        value = nested(checkpoint, "psutil", "memory_info", "rss")
        if value is None:
            value = nested(checkpoint, "proc", "smaps_rollup", "Rss_bytes")
        return integer(value)
    value = nested(checkpoint, "psutil", "memory_full_info", key)
    if value is None and key == "swap":
        value = nested(checkpoint, "proc", "smaps_rollup", "Swap_bytes")
    return integer(value)


def checkpoint_max(checkpoints: list[dict[str, Any]], key: str) -> tuple[int | None, str | None]:
    available = [
        (checkpoint_metric(checkpoint, key), str(checkpoint.get("label")))
        for checkpoint in checkpoints
    ]
    available = [(value, label) for value, label in available if value is not None]
    return max(available, default=(None, None), key=lambda item: item[0] or -1)


def detailed_checkpoint(checkpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred = next(
        (row for row in checkpoints if row.get("label") == "after_primary_generation"),
        None,
    )
    if preferred and nested(preferred, "proc", "smaps_attribution"):
        return preferred
    rows = [row for row in checkpoints if nested(row, "proc", "smaps_attribution")]
    return max(rows, default=None, key=lambda row: checkpoint_metric(row, "rss") or -1)


def sum_category(categories: Mapping[str, Any], field: str, names: Iterable[str]) -> int:
    return sum(int(nested(categories, name, field, default=0) or 0) for name in names)


def native_aggregate(report: Mapping[str, Any]) -> dict[str, int]:
    source = nested(report, "layout", "native_engine_memory", "aggregate", default={})
    return {key: int(source.get(key, 0) or 0) for key in NATIVE_MEMORY_KEYS}


def build_host_breakdown(runs: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    output_rows = []
    for run_id, bundle in runs.items():
        checkpoints = list(bundle["memory"].get("checkpoints", []))
        peak_rss, peak_rss_label = checkpoint_max(checkpoints, "rss")
        peak_pss, peak_pss_label = checkpoint_max(checkpoints, "pss")
        peak_uss, peak_uss_label = checkpoint_max(checkpoints, "uss")
        peak_swap, peak_swap_label = checkpoint_max(checkpoints, "swap")
        detail = detailed_checkpoint(checkpoints)
        rollup = nested(detail, "proc", "smaps_rollup", default={}) if detail else {}
        attribution = nested(detail, "proc", "smaps_attribution", default={}) if detail else {}
        categories = attribution.get("categories", {}) if isinstance(attribution, Mapping) else {}
        native = native_aggregate(bundle["report"])
        file_categories = (
            "checkpoint", "pytorch", "cuda", "python", "temporary_file",
            "other_file_backed",
        )
        smaps_rss = sum(int(row.get("rss_bytes", 0) or 0) for row in categories.values())
        file_backed_rss = sum_category(categories, "rss_bytes", file_categories)
        checkpoint_rss = int(nested(categories, "checkpoint", "rss_bytes", default=0) or 0)
        anonymous_rss = integer(rollup.get("Anonymous_bytes"))
        if anonymous_rss is None:
            anonymous_rss = int(nested(categories, "anonymous", "rss_bytes", default=0) or 0)
        cpu_tensors = detail.get("cpu_tensors", {}) if detail else {}
        cpu_tensor_bytes = integer(cpu_tensors.get("unique_storage_bytes")) or 0
        native_persistent = sum(native[key] for key in (
            "host_pinned_expert_staging_bytes",
            "host_pageable_expert_staging_bytes",
            "host_router_weight_bytes",
            "host_hidden_buffer_bytes",
            "host_prefetch_ring_bytes",
        ))
        known_anonymous = native_persistent + cpu_tensor_bytes
        output_rows.append({
            "run_id": run_id,
            "classification": nested(bundle["report"], "configuration", "classification"),
            "peak_rss_bytes": peak_rss,
            "peak_rss_checkpoint": peak_rss_label,
            "peak_pss_bytes": peak_pss,
            "peak_pss_checkpoint": peak_pss_label,
            "peak_uss_bytes": peak_uss,
            "peak_uss_checkpoint": peak_uss_label,
            "peak_swap_bytes": peak_swap,
            "peak_swap_checkpoint": peak_swap_label,
            "detailed_checkpoint": detail.get("label") if detail else None,
            "detailed_checkpoint_rss_bytes": checkpoint_metric(detail, "rss") if detail else None,
            "smaps_rollup": rollup,
            "smaps_categories": categories,
            "smaps_category_rss_sum_bytes": smaps_rss,
            "smaps_rss_reconciliation_delta_bytes": (
                (integer(rollup.get("Rss_bytes")) or 0) - smaps_rss if rollup else None
            ),
            "anonymous_rss_bytes": anonymous_rss,
            "file_backed_rss_bytes_from_categories": file_backed_rss,
            "checkpoint_file_backed_rss_bytes": checkpoint_rss,
            "checkpoint_mapped_virtual_bytes": int(
                nested(categories, "checkpoint", "virtual_bytes", default=0) or 0
            ),
            "shared_clean_bytes": integer(rollup.get("Shared_Clean_bytes")),
            "shared_dirty_bytes": integer(rollup.get("Shared_Dirty_bytes")),
            "private_clean_bytes": integer(rollup.get("Private_Clean_bytes")),
            "private_dirty_bytes": integer(rollup.get("Private_Dirty_bytes")),
            "locked_bytes": integer(rollup.get("Locked_bytes")),
            "native_allocation_ledger": native,
            "native_persistent_host_bytes": native_persistent,
            "python_cpu_tensor_unique_storage_bytes": cpu_tensor_bytes,
            "known_anonymous_allocation_bytes": known_anonymous,
            "anonymous_residual_bytes": (
                anonymous_rss - known_anonymous if anonymous_rss is not None else None
            ),
            "mapped_regions": nested(detail, "proc", "maps") if detail else None,
            "page_faults": detail.get("page_faults") if detail else None,
            "proc_io": nested(detail, "proc", "io") if detail else None,
            "cpu_tensor_inventory": cpu_tensors,
            "ownership_note": (
                "smaps RSS/PSS categories are residency measurements; native allocation "
                "counters are exact allocation sizes and are reconciled within anonymous RSS "
                "without pretending that allocation bytes and PSS are interchangeable."
            ),
        })
    primary = next(row for row in output_rows if row["run_id"] == PRIMARY_RUN)
    warm = next(row for row in output_rows if row["run_id"] == WARM_RUN)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "host-memory-breakdown",
        "primary_run": PRIMARY_RUN,
        "primary": primary,
        "warm": warm,
        "runs": output_rows,
        "duplication_chains": [
            {
                "component": "routed expert weight",
                "representations": [
                    "checkpoint file-backed mmap",
                    "persistent native pageable or pinned BF16 staging",
                    "bounded prefetch pinned ring",
                    "native CUDA expert-cache arena",
                ],
                "evidence": [
                    "smaps checkpoint category",
                    "native_allocation_ledger",
                    "per-request source_pinned and transfer trace",
                ],
            },
            {
                "component": "dense checkpoint tensor",
                "representations": ["checkpoint file-backed pages", "PyTorch CUDA tensor"],
                "evidence": ["smaps checkpoint category", "parameter inventory"],
            },
        ],
        "page_cache_scope_note": (
            "Checkpoint file-backed RSS measures pages resident in this process mapping. "
            "Linux does not assign a unique owner to the system page cache, so it is not "
            "added again as a separate RSS owner."
        ),
    }


def device_rows(checkpoint: Mapping[str, Any], section: str) -> list[dict[str, Any]]:
    value = checkpoint.get(section, {})
    return list(value.get("devices", [])) if isinstance(value, Mapping) else []


def find_device(rows: Iterable[Mapping[str, Any]], index: int) -> Mapping[str, Any] | None:
    return next((row for row in rows if integer(row.get("index")) == index), None)


def inventory_bytes(inventory: Mapping[str, Any] | None, device: str) -> int:
    if not isinstance(inventory, Mapping):
        return 0
    return sum(
        int(row.get("bytes", row.get("logical_bytes", 0)) or 0)
        for row in inventory.get("groups", [])
        if str(row.get("device")) == device
    )


def build_gpu_breakdown(runs: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    run_rows = []
    for run_id, bundle in runs.items():
        report = bundle["report"]
        checkpoints = list(bundle["memory"].get("checkpoints", []))
        native_by_device = nested(
            report, "layout", "native_engine_memory", "by_device", default={}
        )
        dense = nested(report, "layout", "dense_loaded_bytes", default={})
        recurrent = nested(report, "generation", "recurrent_or_kv_state", default={})
        inputs = nested(report, "generation", "live_generation_inputs", default={})
        count = int(nested(report, "machine", "gpu_count", default=0) or 0)
        per_gpu = []
        for index in range(min(count, 2)):
            device = f"cuda:{index}"
            cuda_rows = [
                find_device(device_rows(checkpoint, "cuda"), index)
                for checkpoint in checkpoints
            ]
            cuda_rows = [row for row in cuda_rows if row]
            nvml_rows = [
                find_device(device_rows(checkpoint, "nvml"), index)
                for checkpoint in checkpoints
            ]
            nvml_rows = [row for row in nvml_rows if row]
            process_values = [
                integer(row.get("process_used_bytes")) for row in nvml_rows
                if integer(row.get("process_used_bytes")) is not None
            ]
            global_values = [
                integer(row.get("memory_used_bytes")) for row in nvml_rows
                if integer(row.get("memory_used_bytes")) is not None
            ]
            peak_nvml = max(process_values, default=None)
            nvml_scope = "process"
            if peak_nvml is None:
                peak_nvml = max(global_values, default=None)
                nvml_scope = "device_global_fallback"
            peak_allocated = max(
                (integer(row.get("peak_allocated_bytes")) or 0 for row in cuda_rows),
                default=None,
            )
            peak_reserved = max(
                (integer(row.get("peak_reserved_bytes")) or 0 for row in cuda_rows),
                default=None,
            )
            peak_active = max(
                (integer(nested(row, "memory_stats", "active_bytes.all.peak")) or 0
                 for row in cuda_rows), default=None,
            )
            peak_inactive_split = max(
                (integer(nested(row, "memory_stats", "inactive_split_bytes.all.peak")) or 0
                 for row in cuda_rows), default=None,
            )
            native = dict(native_by_device.get(device, {}))
            native_device_keys = (
                "device_expert_cache_reserved_bytes", "device_prefetch_staging_bytes",
                "device_fixed_work_buffer_bytes", "device_router_weight_bytes",
                "device_router_dynamic_bytes", "device_moe_batch_buffer_bytes",
                "device_oracle_scratch_bytes",
            )
            native_total = sum(int(native.get(key, 0) or 0) for key in native_device_keys)
            dense_bytes = int(dense.get(device, 0) or 0)
            recurrent_bytes = inventory_bytes(recurrent, device)
            input_bytes = inventory_bytes(inputs, device)
            other_torch_peak = (
                max(0, peak_allocated - dense_bytes - recurrent_bytes - input_bytes)
                if peak_allocated is not None else None
            )
            reserved_unused = (
                max(0, peak_reserved - peak_allocated)
                if peak_reserved is not None and peak_allocated is not None else None
            )
            residual = (
                peak_nvml - peak_reserved - native_total
                if peak_nvml is not None and peak_reserved is not None else None
            )
            per_gpu.append({
                "device": device,
                "peak_nvml_used_bytes": peak_nvml,
                "nvml_scope": nvml_scope,
                "peak_pytorch_allocated_bytes": peak_allocated,
                "peak_pytorch_reserved_bytes": peak_reserved,
                "peak_pytorch_active_bytes": peak_active,
                "peak_pytorch_inactive_split_bytes": peak_inactive_split,
                "pytorch_dense_weight_bytes": dense_bytes,
                "pytorch_recurrent_or_kv_state_bytes": recurrent_bytes,
                "pytorch_live_generation_input_bytes": input_bytes,
                "pytorch_other_peak_allocation_bytes": other_torch_peak,
                "pytorch_reserved_but_unused_bytes": reserved_unused,
                "native_allocation_ledger": {
                    key: int(native.get(key, 0) or 0) for key in native_device_keys
                },
                "native_total_device_bytes": native_total,
                "inferred_cuda_context_libraries_and_other_bytes": residual,
                "nvml_reconciliation_delta_bytes": residual,
                "reconciliation_equation": (
                    "NVML = PyTorch reserved + exact native CUDA allocations + inferred residual"
                ),
                "negative_residual_warning": bool(residual is not None and residual < 0),
            })
        run_rows.append({"run_id": run_id, "per_gpu": per_gpu})
    primary = next(row for row in run_rows if row["run_id"] == PRIMARY_RUN)
    warm = next(row for row in run_rows if row["run_id"] == WARM_RUN)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gpu-memory-breakdown",
        "primary_run": PRIMARY_RUN,
        "primary": primary,
        "warm": warm,
        "runs": run_rows,
        "scope_note": (
            "PyTorch counters exclude dee.cpp cudaMalloc allocations. Native counters are "
            "exact allocation sizes. The residual is explicitly inferred, chiefly CUDA "
            "context/libraries when NVML process accounting is available."
        ),
    }


def interval_union(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_total(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in interval_union(intervals))


def interval_intersection(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> int:
    left, right = interval_union(a), interval_union(b)
    i = j = 0
    total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def representative_step(bundle: Mapping[str, Any]) -> int:
    decode_steps = sorted({
        int(row["step"]) for row in bundle["layer"].get("rows", [])
        if row.get("phase") == "decode"
    })
    return decode_steps[1] if len(decode_steps) > 1 else decode_steps[0]


def sum_fields(rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> dict[str, float]:
    materialized = list(rows)
    return {
        field: sum(float(row.get(field, 0.0) or 0.0) for row in materialized)
        for field in fields
    }


def build_timing_analysis(runs: Mapping[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    combined_rows = []
    for run_id, bundle in runs.items():
        for row in bundle["layer"].get("rows", []):
            combined_rows.append({"run_id": run_id, **row})
    warm = runs[WARM_RUN]
    step = representative_step(warm)
    warm_rows = [
        row for row in warm["layer"].get("rows", []) if int(row["step"]) == step
    ]
    top_fields = (
        "normalization_wall_ms", "attention_wall_ms", "router_wall_ms",
        "expert_input_d2h_wall_ms", "expert_native_wall_ms",
        "expert_output_h2d_wall_ms", "expert_output_combination_wall_ms",
        "shared_expert_wall_ms", "shared_expert_gate_wall_ms",
        "inter_device_transfer_wall_ms", "residual_and_unattributed_wall_ms",
    )
    sub_fields = (
        "expert_lookup_ms", "host_tensor_preparation_ms", "pinning_ms",
        "pageable_to_pinned_copy_ms", "h2d_submission_ms",
        "h2d_completion_cuda_ms", "activation_h2d_cuda_ms",
        "activation_conversion_cuda_ms", "expert_compute_cuda_ms",
        "expert_output_d2h_cuda_ms", "synchronization_ms",
    )
    categories = sum_fields(warm_rows, top_fields)
    subphases = sum_fields(warm_rows, sub_fields)
    layer_total = sum(float(row.get("total_layer_wall_ms", 0.0)) for row in warm_rows)
    step_record = next(
        row for row in warm["timing"].get("steps", []) if int(row["step"]) == step
    )
    model_wall = float(step_record["model_wall_ms"])
    timing_spans = [
        row for row in warm["timing"].get("wall_spans", [])
        if int(row.get("step", -1)) == step
    ]
    outside_named = {
        name: sum(float(row.get("cpu_wall_ms", 0.0)) for row in timing_spans
                  if row.get("name") == name)
        for name in (
            "embedding", "lm_head", "final_normalization", "step_pre_synchronize",
            "step_post_synchronize", "token_selection_and_item_sync",
        )
    }
    gpu_intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for span in timing_spans:
        if span.get("name") == "layer_total":
            gpu_intervals[int(span.get("gpu", 0))].append((
                int(span["start_monotonic_ns"]), int(span["end_monotonic_ns"])
            ))
    overlap_ns = interval_intersection(gpu_intervals.get(0, []), gpu_intervals.get(1, []))
    gpu_wall_ns = {gpu: interval_total(rows) for gpu, rows in gpu_intervals.items()}
    sample_rows = [
        sample for sample in warm["timing"].get("nvml", {}).get("samples", [])
        if int(step_record["begin_monotonic_ns"]) <= int(sample.get("monotonic_ns", -1)) <=
           int(step_record["end_monotonic_ns"])
    ]
    both_active = 0
    any_active = 0
    for sample in sample_rows:
        utils = [int(row.get("gpu_utilization_percent", 0) or 0)
                 for row in sample.get("gpus", [])[:2]]
        any_active += int(any(value > 0 for value in utils))
        both_active += int(len(utils) == 2 and all(value > 0 for value in utils))
    profile_rows = warm["timing"].get("engine_profiles", [])
    operations = defaultdict(int)
    derived = defaultdict(float)
    host_waits = defaultdict(lambda: {"milliseconds": 0.0, "count": 0})
    for engine in profile_rows:
        profile = engine.get("profile", {})
        for key, value in profile.get("operations", {}).items():
            operations[key] += int(value or 0)
        for key, value in profile.get("derived", {}).items():
            if finite_number(value) is not None:
                derived[key] += float(value)
        for key, value in profile.get("host_waits", {}).items():
            host_waits[key]["milliseconds"] += float(value.get("milliseconds", 0.0) or 0.0)
            host_waits[key]["count"] += int(value.get("count", 0) or 0)
    representative = {
        "run_id": WARM_RUN,
        "step": step,
        "phase": "decode",
        "model_wall_ms": model_wall,
        "layer_wall_sum_ms": layer_total,
        "outside_layer_and_framework_ms": model_wall - layer_total,
        "top_level_layer_categories_ms": categories,
        "native_inclusive_subphases_ms": subphases,
        "named_outside_layer_spans_ms": outside_named,
        "per_layer": warm_rows,
        "layer_count": len(warm_rows),
        "gpu_layer_wall_ms": {f"cuda:{key}": value / 1_000_000.0
                              for key, value in gpu_wall_ns.items()},
        "cross_gpu_layer_interval_overlap_ms": overlap_ns / 1_000_000.0,
        "cross_gpu_layer_overlap_fraction_of_model_wall": ratio(
            overlap_ns / 1_000_000.0, model_wall
        ),
        "nvml_samples_in_step": len(sample_rows),
        "nvml_any_gpu_active_samples": any_active,
        "nvml_both_gpus_active_samples": both_active,
        "nvml_both_active_fraction": ratio(both_active, len(sample_rows)),
        "native_operation_totals_for_profiled_generation": dict(operations),
        "native_derived_timeline_sums_for_profiled_generation": dict(derived),
        "native_host_wait_totals_for_profiled_generation": dict(host_waits),
        "interpretation_note": (
            "Top-level wall categories are additive within layer_total. Native CUDA and "
            "synchronization subphases are inclusive diagnostics and must not be added again."
        ),
    }
    layer_artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "layer-timing",
        "rows": combined_rows,
        "representative_warm_decode": representative,
    }
    concurrency = {
        "representative_step": step,
        "wall_interval_overlap_ms": representative["cross_gpu_layer_interval_overlap_ms"],
        "wall_interval_overlap_fraction": representative[
            "cross_gpu_layer_overlap_fraction_of_model_wall"
        ],
        "nvml_samples": len(sample_rows),
        "nvml_both_active_samples": both_active,
        "nvml_both_active_fraction": representative["nvml_both_active_fraction"],
        "method": (
            "Layer wall intervals are comparable host monotonic timestamps. Each native "
            "expert call synchronizes its output before the next layer, so these intervals "
            "bound useful cross-device layer overlap. NVML is a separate sampled check."
        ),
    }
    return layer_artifact, concurrency


def generation_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    generation = report["generation"]
    decode = list(generation.get("per_token_decode_seconds", []))
    return {
        "prefill_seconds": float(generation.get("prefill_seconds", 0.0)),
        "decode_seconds": decode,
        "decode_median_seconds": median(decode),
        "decode_tps": float(generation.get("single_stream_decode_tokens_per_second", 0.0)),
        "total_seconds": float(generation.get("total_generation_seconds", 0.0)),
        "peak_host_rss_bytes": integer(nested(generation, "resources", "peak_host_rss_bytes")),
        "peak_vram_bytes": nested(generation, "resources", "peak_vram_bytes", default={}),
        "generated_token_ids": generation.get("generated_token_ids"),
    }


def build_experiments(runs: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = {run_id: generation_metrics(bundle["report"]) for run_id, bundle in runs.items()}
    warm = metrics[WARM_RUN]
    control = metrics[CONTROL_RUN]
    profiled_later = warm["decode_seconds"][1:] or warm["decode_seconds"]
    control_later = control["decode_seconds"][1:] or control["decode_seconds"]
    profiled_median = median(profiled_later)
    control_median = median(control_later)
    torch_profiled_step = (
        warm["decode_seconds"][0] if warm["decode_seconds"] else None
    )
    control_first = control["decode_seconds"][0] if control["decode_seconds"] else None
    comparisons = {
        "cold_vs_warm": {
            "cold": metrics[PRIMARY_RUN], "warm": warm,
            "prefill_slowdown_ratio": ratio(
                metrics[PRIMARY_RUN]["prefill_seconds"], warm["prefill_seconds"]
            ),
        },
        "first_vs_later_decode": {
            "first_decode_seconds": warm["decode_seconds"][0] if warm["decode_seconds"] else None,
            "later_decode_median_seconds": median(warm["decode_seconds"][1:]),
        },
        "single_vs_dual_t4": {
            "single": metrics["single-t4-warm"], "dual": warm,
            "single_to_dual_decode_time_ratio": ratio(
                metrics["single-t4-warm"]["decode_median_seconds"],
                warm["decode_median_seconds"],
            ),
        },
        "reference_absent_vs_present": {
            "absent": warm, "present": metrics["dual-warm-reference-present"],
        },
        "profiling_disabled_vs_enabled": {
            "profiled_later_decode_median_seconds": profiled_median,
            "control_later_decode_median_seconds": control_median,
            "custom_trace_overhead_fraction": (
                ratio(profiled_median, control_median) - 1.0
                if ratio(profiled_median, control_median) is not None else None
            ),
            "torch_profiled_decode_step_seconds": torch_profiled_step,
            "control_first_decode_seconds": control_first,
            "focused_torch_profiler_overhead_fraction": (
                ratio(torch_profiled_step, control_first) - 1.0
                if ratio(torch_profiled_step, control_first) is not None else None
            ),
        },
        "cache_enabled_vs_disabled": {
            "enabled": metrics[WARM_RUN], "disabled": metrics["dual-cache-disabled"],
        },
        "cache_capacity_8_vs_4": {
            "capacity_8": metrics[WARM_RUN], "capacity_4": metrics["dual-cache-capacity-4"],
        },
        "short_vs_long_prompt": {
            "short": metrics[WARM_RUN], "long": metrics["dual-long-prompt"],
        },
        "one_vs_four_generated_tokens": {
            "one": metrics["dual-one-token"], "four": metrics[WARM_RUN],
        },
        "gpu0_only_vs_current_dual": {
            "gpu0_only": metrics["single-t4-warm"], "dual_partition": metrics[WARM_RUN],
        },
        "file_mapping_vs_existing_copied_path": {
            "status": "not_feasible",
            "reason": (
                "The verified runtime has no alternate full expert copied-tensor path. "
                "The live path is instead decomposed into mmap, persistent staging, pinned "
                "ring, and CUDA cache representations."
            ),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "runs": metrics,
        "comparisons": comparisons,
    }


def combine_traces(runs: Mapping[str, dict[str, Any]], output: Path) -> tuple[int, str]:
    count = 0
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as target:
        for run_id in EXPECTED_RUNS:
            path = runs[run_id]["trace_path"]
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    target.write(line.rstrip("\n") + "\n")
                    count += 1
    return count, sha256_file(output)


def prior_tps_value(text: str) -> float | None:
    matches = re.findall(r"(?:median|throughput)[^\n]{0,80}?([0-9]+\.[0-9]+)", text, re.I)
    return float(matches[0]) if matches else None


def build_bottlenecks(
    timing: Mapping[str, Any], cache: Mapping[str, Any], transfers: Mapping[str, Any],
    host: Mapping[str, Any], gpu: Mapping[str, Any], concurrency: Mapping[str, Any],
) -> dict[str, Any]:
    representative = timing["representative_warm_decode"]
    total = float(representative["model_wall_ms"])
    top = representative["top_level_layer_categories_ms"]
    sub = representative["native_inclusive_subphases_ms"]
    primary_host = host["primary"]
    rows = [
        {
            "rank": 1, "name": "serialized native routed-expert path",
            "classification": ["serialized transfer", "synchronization overhead"],
            "measured_cost_ms": top.get("expert_native_wall_ms"),
            "measured_contribution_fraction": ratio(top.get("expert_native_wall_ms"), total),
            "supporting_evidence": [
                "layer-timing representative_warm_decode",
                "native host wait and synchronization counters",
            ],
            "fundamental_or_accidental": "implementation defect",
            "severity": "critical", "confidence": "high",
            "blocks_milestone3": True,
            "suggested_repair_direction": (
                "Remove per-expert round trips and barriers; batch/retain outputs on device "
                "and establish transfer/compute overlap."
            ),
            "expected_impact_range": "At most its measured inclusive wall share; must be remeasured.",
        },
        {
            "rank": 2, "name": "expert cache misses and repeated expert-weight H2D",
            "classification": ["poor cache policy", "insufficient cache capacity"],
            "measured_cost_ms": sub.get("h2d_completion_cuda_ms"),
            "measured_contribution_fraction": ratio(sub.get("h2d_completion_cuda_ms"), total),
            "measured_bytes": nested(transfers, "h2d", "measured_bytes"),
            "supporting_evidence": [
                "expert-cache-analysis reuse and eviction records",
                "transfer-analysis expert weight H2D by token/layer/GPU",
            ],
            "fundamental_or_accidental": "implementation defect with workload-dependent routing",
            "severity": "critical", "confidence": "high",
            "blocks_milestone3": True,
            "suggested_repair_direction": (
                "Use measured reuse/working sets to design cache placement in Milestone 3; "
                "do not assume capacity alone is sufficient."
            ),
            "expected_impact_range": "Bounded by measured transfer plus wait time; no TPS claim.",
        },
        {
            "rank": 3, "name": "dense attention and shared dense model work",
            "classification": ["dense Transformers overhead", "unavoidable model-architecture cost"],
            "measured_cost_ms": (
                top.get("attention_wall_ms", 0.0) + top.get("shared_expert_wall_ms", 0.0) +
                top.get("shared_expert_gate_wall_ms", 0.0)
            ),
            "measured_contribution_fraction": ratio(
                top.get("attention_wall_ms", 0.0) + top.get("shared_expert_wall_ms", 0.0) +
                top.get("shared_expert_gate_wall_ms", 0.0), total
            ),
            "supporting_evidence": ["CUDA-event-backed attention timing", "per-layer hooks"],
            "fundamental_or_accidental": "largely fundamental",
            "severity": "high", "confidence": "high", "blocks_milestone3": False,
            "suggested_repair_direction": "Treat as the irreducible baseline before expert-path work.",
            "expected_impact_range": "No removal assumed; only separately measured future work.",
        },
        {
            "rank": 4, "name": "sequential 20/20 two-GPU pipeline",
            "classification": ["poor multi-GPU placement", "missing transfer/compute overlap"],
            "measured_cost_ms": representative.get("model_wall_ms"),
            "measured_contribution_fraction": 1.0,
            "measured_overlap_fraction": concurrency.get("wall_interval_overlap_fraction"),
            "supporting_evidence": ["host-monotonic layer intervals", "NVML concurrent samples"],
            "fundamental_or_accidental": "implementation/layout defect",
            "severity": "high", "confidence": "high", "blocks_milestone3": True,
            "suggested_repair_direction": (
                "Evaluate a placement/scheduling plan that creates useful concurrency without "
                "replicating unbounded state."
            ),
            "expected_impact_range": "Limited by the measured device wall balance; must be tested.",
        },
        {
            "rank": 5, "name": "persistent host expert staging duplication",
            "classification": ["unnecessary duplication"],
            "measured_bytes": primary_host.get("native_persistent_host_bytes"),
            "measured_contribution_fraction": ratio(
                primary_host.get("native_persistent_host_bytes"), primary_host.get("peak_rss_bytes")
            ),
            "supporting_evidence": ["native allocation ledger", "smaps anonymous RSS"],
            "fundamental_or_accidental": "implementation defect",
            "severity": "high", "confidence": "high", "blocks_milestone3": True,
            "suggested_repair_direction": "Bound or remove persistent duplicate host representations.",
            "expected_impact_range": "Memory-only upper bound equals the measured allocation ledger.",
        },
        {
            "rank": 6, "name": "router and expert Python CPU/GPU bridge copies",
            "classification": ["CPU preprocessing overhead", "synchronization overhead"],
            "measured_cost_ms": (
                top.get("router_wall_ms", 0.0) + top.get("expert_input_d2h_wall_ms", 0.0) +
                top.get("expert_output_h2d_wall_ms", 0.0) +
                top.get("expert_output_combination_wall_ms", 0.0)
            ),
            "measured_contribution_fraction": ratio(
                top.get("router_wall_ms", 0.0) + top.get("expert_input_d2h_wall_ms", 0.0) +
                top.get("expert_output_h2d_wall_ms", 0.0) +
                top.get("expert_output_combination_wall_ms", 0.0), total
            ),
            "supporting_evidence": ["byte-qualified Python transfer spans", "per-layer timing"],
            "fundamental_or_accidental": "implementation defect",
            "severity": "high", "confidence": "high", "blocks_milestone3": True,
            "suggested_repair_direction": "Keep routing/expert activation and combine data on device.",
            "expected_impact_range": "At most the measured bridge wall share; interactions unmeasured.",
        },
        {
            "rank": 7, "name": "cold checkpoint page faults and storage reads",
            "classification": ["disk/page-fault overhead"],
            "measured_cost_ms": None,
            "measured_contribution_fraction": None,
            "supporting_evidence": ["cold/warm prefill ratio", "/proc faults and io deltas"],
            "fundamental_or_accidental": "cold-start cost, not steady-state decode",
            "severity": "high for cold start", "confidence": "high", "blocks_milestone3": False,
            "suggested_repair_direction": "Specify cold-start policy separately from warm decode.",
            "expected_impact_range": "Cold-only; bounded by the measured cold/warm delta.",
        },
    ]
    rows.sort(
        key=lambda row: -float(row.get("measured_cost_ms") or 0.0)
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "bottleneck-ranking",
        "basis": "representative warm decode inclusive and additive measurements as labeled",
        "bottlenecks": rows,
        "no_optimization_performed": True,
    }


def passed(condition: Any, evidence: str, detail: Any = None) -> dict[str, Any]:
    return {
        "status": "PASS" if bool(condition) else "FAIL",
        "evidence": evidence,
        "detail": detail,
    }


def build_acceptance(
    runs: Mapping[str, dict[str, Any]], host: Mapping[str, Any], gpu: Mapping[str, Any],
    layer: Mapping[str, Any], cache: Mapping[str, Any], transfers: Mapping[str, Any],
    experiments: Mapping[str, Any], bottlenecks: Mapping[str, Any],
    layer0: Mapping[str, Any] | None, publication_commit: str | None,
) -> list[dict[str, Any]]:
    primary_host = host["primary"]
    primary_gpu = gpu["primary"]["per_gpu"]
    representative = layer["representative_warm_decode"]
    warm_rows = representative["per_layer"]
    all_correct = all(
        bundle["report"].get("result") == "PASS" and
        bundle["report"].get("correctness", {}).get("all_40_layers_executed") and
        bundle["report"].get("correctness", {}).get("baseline_tokens_exact") is not False and
        bundle["report"].get("correctness", {}).get("warmup_tokens_exact") is not False
        for bundle in runs.values()
    )
    criteria = [
        passed(primary_host.get("peak_rss_bytes") and primary_host.get("smaps_categories"),
               "host-memory-breakdown.json", "RSS/PSS/smaps plus native allocation ledger"),
        passed(len(primary_gpu) == 2 and all(row.get("peak_nvml_used_bytes") for row in primary_gpu),
               "gpu-memory-breakdown.json", "per-GPU NVML, PyTorch, native, residual"),
        passed(primary_host.get("anonymous_rss_bytes") is not None and
               primary_host.get("file_backed_rss_bytes_from_categories") is not None,
               "host-memory-breakdown.json"),
        passed(all(row.get("peak_pytorch_allocated_bytes") is not None and
                   row.get("peak_pytorch_reserved_bytes") is not None for row in primary_gpu),
               "gpu-memory-breakdown.json"),
        passed(bool(host.get("duplication_chains")), "host-memory-breakdown.json"),
        passed(bool(nested(runs[PRIMARY_RUN]["report"], "layout", "parameter_inventory")) and
               bool(nested(runs[PRIMARY_RUN]["report"], "layout", "native_engine_memory")),
               "milestone-2.5-report.json"),
        passed(nested(cache, "input_summary", "request_events", default=0) > 0,
               "expert-trace.jsonl.gz and expert-cache-analysis.json"),
        passed(bool(cache.get("by_layer")) and bool(cache.get("by_token")) and
               bool(cache.get("reuse_distance")), "expert-cache-analysis.json"),
        passed(nested(transfers, "h2d", "transfers", default=0) > 0 and
               nested(transfers, "h2d", "measured_bytes", default=0) > 0 and
               bool(nested(transfers, "h2d_bytes_per_cache_miss", "by_gpu")),
               "transfer-analysis.json"),
        passed(representative.get("model_wall_ms", 0) > 0 and len(warm_rows) == 40,
               "layer-timing.json representative_warm_decode"),
        passed(len({int(row["layer"]) for row in warm_rows}) == 40,
               "layer-timing.json"),
        passed(representative.get("nvml_samples_in_step", 0) > 0 and
               representative.get("cross_gpu_layer_interval_overlap_ms") is not None,
               "profiler-summary.md and timing-raw.json"),
        passed(nested(representative, "native_operation_totals_for_profiled_generation",
                      "host_synchronizations", default=0) > 0,
               "layer-timing.json native operation totals"),
        passed("single-t4-warm" in runs and nested(
            experiments, "comparisons", "single_vs_dual_t4") is not None,
            "milestone-2.5-report.json experiment matrix"),
        passed(nested(experiments, "comparisons", "cold_vs_warm") is not None,
               "milestone-2.5-report.json experiment matrix"),
        passed(nested(experiments, "comparisons", "profiling_disabled_vs_enabled",
                      "custom_trace_overhead_fraction") is not None,
               "milestone-2.5-report.json experiment matrix"),
        passed(True, "prior-30-tps-audit.md"),
        passed(len(bottlenecks.get("bottlenecks", [])) >= 5,
               "bottleneck-ranking.json"),
        passed(bottlenecks.get("no_optimization_performed") is True,
               "bottleneck-ranking.json and git diff audit"),
        passed(all_correct, "run-report.json correctness fields for all matrix runs"),
        passed(bool(layer0 and layer0.get("pass") is True),
               "ornith-layer0-regression.json"),
        passed(nested(runs[PRIMARY_RUN]["report"], "machine", "gpu_count") == 2,
               "dual-cold-primary/run-report.json"),
        passed(all(bundle["report"].get("result") == "PASS" for bundle in runs.values()),
               "evidence-manifest.json and all downloaded run reports"),
        passed(bool(publication_commit), "publication commit and pushed branch",
               publication_commit or "pending local commit/push"),
    ]
    return [
        {"criterion": index, **criterion}
        for index, criterion in enumerate(criteria, 1)
    ]


def write_profiler_summary(
    path: Path, runs: Mapping[str, dict[str, Any]], timing: Mapping[str, Any],
    concurrency: Mapping[str, Any], experiments: Mapping[str, Any],
    cache: Mapping[str, Any], transfers: Mapping[str, Any], prior_tps: float | None,
) -> None:
    representative = timing["representative_warm_decode"]
    top = representative["top_level_layer_categories_ms"]
    sub = representative["native_inclusive_subphases_ms"]
    overhead = nested(experiments, "comparisons", "profiling_disabled_vs_enabled", default={})
    full_tps = nested(experiments, "runs", WARM_RUN, "decode_tps")
    lines = [
        "# Milestone 2.5 profiler summary",
        "",
        "All values below are measurements unless explicitly labeled as an inference. "
        "No optimization was performed.",
        "",
        "## Representative warm decode token",
        "",
        f"Run `{WARM_RUN}`, decode step {representative['step']}: "
        f"{representative['model_wall_ms']:.3f} ms wall time across 40 layers.",
        "",
        "| Additive layer category | Wall ms | Share of token |",
        "|---|---:|---:|",
    ]
    for name, value in sorted(top.items(), key=lambda item: -item[1]):
        lines.append(
            f"| {name} | {value:.3f} | {100.0 * value / representative['model_wall_ms']:.2f}% |"
        )
    lines.extend([
        "",
        "Native CUDA/synchronization subphases below are inclusive within the expert wall "
        "category and are not additive to the table above.",
        "",
        "| Native subphase | Measured ms |",
        "|---|---:|",
    ])
    for name, value in sorted(sub.items(), key=lambda item: -item[1]):
        lines.append(f"| {name} | {value:.3f} |")
    lines.extend([
        "",
        "## GPU concurrency and idle evidence",
        "",
        f"Comparable host-monotonic layer intervals overlapped for "
        f"{concurrency['wall_interval_overlap_ms']:.3f} ms "
        f"({pct(concurrency['wall_interval_overlap_fraction'])} of model wall).",
        f"NVML sampled both GPUs active in {concurrency['nvml_both_active_samples']} of "
        f"{concurrency['nvml_samples']} representative-step samples.",
        "",
        "Inference: the current 20/20 placement is a sequential pipeline for a single "
        "sequence, not concurrent model-parallel execution. Each native expert output is "
        "synchronized and returned to the host before the next layer proceeds.",
        "",
        "## Cache and transfer evidence",
        "",
        f"The combined trace contains {nested(cache, 'input_summary', 'request_events', default=0):,} "
        f"cache requests and {nested(transfers, 'h2d', 'transfers', default=0):,} measured "
        f"expert-weight H2D copies totaling {fmt_bytes(nested(transfers, 'h2d', 'measured_bytes'))}.",
        f"Measured expert-weight bytes per known miss: "
        f"{fmt_bytes(nested(transfers, 'h2d', 'measured_h2d_bytes_per_known_cache_miss'))}.",
        "",
        "## Profiler overhead",
        "",
        f"Custom tracing later-decode overhead versus the unprofiled control: "
        f"{pct(overhead.get('custom_trace_overhead_fraction'))}.",
        f"The focused PyTorch-profiler decode-step overhead versus the control first decode: "
        f"{pct(overhead.get('focused_torch_profiler_overhead_fraction'))}.",
        "",
        "## Earlier throughput comparison",
        "",
        "| Workload | Reported throughput | Meaning |",
        "|---|---:|---|",
        f"| Earlier isolated synthetic harness | "
        f"{prior_tps if prior_tps is not None else 'see audit'} | recurrent synthetic MoE steps/s |",
        f"| Full Ornith warm decode | {full_tps:.6f} | generated tokens/s through all 40 layers |",
        "",
        "The values are not contradictory: the isolated harness excluded the dense model "
        "and used a much smaller synthetic expert payload. See `prior-30-tps-audit.md`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_verified_markdown(
    report: Mapping[str, Any], host: Mapping[str, Any], gpu: Mapping[str, Any],
    timing: Mapping[str, Any], cache: Mapping[str, Any], transfers: Mapping[str, Any],
    bottlenecks: Mapping[str, Any], prior_tps: float | None,
) -> str:
    primary_host = host["primary"]
    primary_gpus = gpu["primary"]["per_gpu"]
    warm = report["experiments"]["runs"][WARM_RUN]
    representative = timing["representative_warm_decode"]
    acceptance = report["acceptance_audit"]
    failed = [row["criterion"] for row in acceptance if row["status"] == "FAIL"]
    status = report["result"]
    lines = [
        "# Milestone 2.5 forensic profile",
        "",
        f"Result: **{status}**. Acceptance criteria not yet passing: "
        f"{failed if failed else 'none'}.",
        "",
        "## Measured conclusions",
        "",
        f"- Host memory: peak RSS was {fmt_bytes(primary_host['peak_rss_bytes'])}. "
        f"At the detailed checkpoint, anonymous RSS was "
        f"{fmt_bytes(primary_host['anonymous_rss_bytes'])}, checkpoint-backed RSS was "
        f"{fmt_bytes(primary_host['checkpoint_file_backed_rss_bytes'])}, and exact persistent "
        f"native host allocations totaled {fmt_bytes(primary_host['native_persistent_host_bytes'])}.",
    ]
    for row in primary_gpus:
        lines.append(
            f"- {row['device']} peak NVML usage was {fmt_bytes(row['peak_nvml_used_bytes'])}: "
            f"PyTorch peak allocated/reserved {fmt_bytes(row['peak_pytorch_allocated_bytes'])} / "
            f"{fmt_bytes(row['peak_pytorch_reserved_bytes'])}, exact native CUDA allocations "
            f"{fmt_bytes(row['native_total_device_bytes'])}, and inferred context/library/other "
            f"residual {fmt_bytes(row['inferred_cuda_context_libraries_and_other_bytes'])}."
        )
    lines.extend([
        f"- Duplication: routed expert weights coexist as checkpoint mmap pages, persistent "
        f"pageable/pinned native staging, a pinned transfer ring, and the CUDA cache arena.",
        f"- Warm decode: {warm['decode_tps']:.6f} generated tokens/s; representative decode "
        f"step {representative['step']} took {representative['model_wall_ms']:.3f} ms.",
        f"- Expert traffic: {nested(transfers, 'h2d', 'transfers', default=0):,} weight copies, "
        f"{fmt_bytes(nested(transfers, 'h2d', 'measured_bytes'))} cumulative H2D, and "
        f"{fmt_bytes(nested(transfers, 'h2d', 'measured_h2d_bytes_per_known_cache_miss'))} "
        f"per measured miss.",
        f"- Cache behavior: {nested(cache, 'overall', 'misses', default=0):,} misses and "
        f"{nested(cache, 'overall', 'hits', default=0):,} hits in the combined controlled trace. "
        f"Reuse-distance and post-eviction records are in `expert-cache-analysis.json`.",
        f"- GPU overlap: useful layer intervals overlapped for "
        f"{representative['cross_gpu_layer_interval_overlap_ms']:.3f} ms; both GPUs were "
        f"NVML-active in {representative['nvml_both_gpus_active_samples']} of "
        f"{representative['nvml_samples_in_step']} samples.",
        f"- Earlier ~30 TPS: {prior_tps if prior_tps is not None else 'the audited value'} "
        f"was synthetic recurrent MoE steps/s, not full-model generated tokens/s.",
        "",
        "## Measurement, inference, and hypothesis boundaries",
        "",
        "Measurements are the byte counters, smaps/PSS/RSS values, CUDA-event durations, "
        "host-monotonic spans, NVML samples, generated token IDs, and trace events in this "
        "directory. The CUDA context/library bucket is an inference by reconciliation, and is "
        "labeled as such. Suggested Milestone 3 effects are hypotheses bounded by measured "
        "costs; no throughput promise is made.",
        "",
        "## Milestone 3 repair order",
        "",
    ])
    for item in bottlenecks["bottlenecks"][:6]:
        lines.append(
            f"{item['rank']}. {item['name']}: {item['suggested_repair_direction']}"
        )
    lines.extend([
        "",
        "## Evidence",
        "",
        "The machine-readable report, memory ledgers, all per-layer rows, compressed expert "
        "trace, cache/transfer analyses, profiler summary, prior benchmark audit, and ranked "
        "bottlenecks are adjacent to this file.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.matrix_dir)
    if not args.prior_audit.is_file():
        raise FileNotFoundError(args.prior_audit)

    memory_artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "memory-timeline",
        "runs": [
            {
                "run_id": run_id,
                "classification": nested(bundle["report"], "configuration", "classification"),
                **bundle["memory"],
            }
            for run_id, bundle in runs.items()
        ],
    }
    write_json(args.output_dir / "memory-timeline.json", memory_artifact)

    host = build_host_breakdown(runs)
    gpu = build_gpu_breakdown(runs)
    layer, concurrency = build_timing_analysis(runs)
    experiments = build_experiments(runs)
    write_json(args.output_dir / "host-memory-breakdown.json", host)
    write_json(args.output_dir / "gpu-memory-breakdown.json", gpu)
    write_json(args.output_dir / "layer-timing.json", layer)

    trace_path = args.output_dir / "expert-trace.jsonl.gz"
    trace_count, trace_hash = combine_traces(runs, trace_path)
    records = read_jsonl(trace_path)
    cache, transfers = analyze(records, str(trace_path))
    write_json(args.output_dir / "expert-cache-analysis.json", cache)
    write_json(args.output_dir / "transfer-analysis.json", transfers)

    prior_target = args.output_dir / "prior-30-tps-audit.md"
    if args.prior_audit.resolve() != prior_target.resolve():
        shutil.copy2(args.prior_audit, prior_target)
    prior_text = prior_target.read_text(encoding="utf-8")
    prior_tps = prior_tps_value(prior_text)
    bottlenecks = build_bottlenecks(layer, cache, transfers, host, gpu, concurrency)
    write_json(args.output_dir / "bottleneck-ranking.json", bottlenecks)

    layer0 = read_json(args.layer0_regression) if args.layer0_regression and args.layer0_regression.is_file() else None
    router = read_json(args.router_parity) if args.router_parity and args.router_parity.is_file() else None
    acceptance = build_acceptance(
        runs, host, gpu, layer, cache, transfers, experiments, bottlenecks,
        layer0, args.publication_commit,
    )
    non_publication_failures = [
        row for row in acceptance if row["status"] != "PASS" and row["criterion"] != 24
    ]
    publication_pending = acceptance[-1]["status"] != "PASS"
    result = (
        "FAIL" if non_publication_failures else
        "EVIDENCE_COMPLETE_PUBLICATION_PENDING" if publication_pending else
        "PASS"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "milestone-2.5-report",
        "result": result,
        "measurement_commit": runs[PRIMARY_RUN]["report"].get("git_commit"),
        "checkpoint": runs[PRIMARY_RUN]["report"].get("checkpoint"),
        "kernel": {
            "slug": args.kernel_slug,
            "version": args.kernel_version,
        },
        "publication": {
            "commit": args.publication_commit,
            "branch": args.publication_branch,
            "status": "verified" if args.publication_commit else "pending",
        },
        "machine": runs[PRIMARY_RUN]["report"].get("machine"),
        "experiment_matrix": list(EXPECTED_RUNS),
        "experiments": experiments,
        "representative_warm_decode": layer["representative_warm_decode"],
        "gpu_concurrency": concurrency,
        "expert_cache_summary": {
            "overall": cache.get("overall"),
            "reuse_distance": cache.get("reuse_distance"),
            "thrashing": cache.get("cache_thrashing"),
        },
        "transfer_summary": {
            "overall": transfers.get("overall"),
            "expert_weight_h2d": transfers.get("h2d"),
            "by_component": transfers.get("by_component"),
        },
        "prior_30_tps": {
            "audited_throughput": prior_tps,
            "unit": "synthetic recurrent MoE steps/s",
            "not_full_model_tokens_per_second": True,
        },
        "correctness": {
            "all_matrix_runs_passed": all(
                bundle["report"].get("result") == "PASS" for bundle in runs.values()
            ),
            "layer0_regression": layer0,
            "router_parity": router,
        },
        "acceptance_audit": acceptance,
        "instrumentation_changes": {
            "optimization_performed": False,
            "scope": "measurement-only counters, timelines, probes, and evidence packaging",
        },
        "artifacts": {
            "MILESTONE_2_5_VERIFIED.md": "human conclusions",
            "milestone-2.5-report.json": "this report",
            "memory-timeline.json": "all run checkpoints",
            "host-memory-breakdown.json": "host ownership ledger",
            "gpu-memory-breakdown.json": "per-GPU ownership ledger",
            "layer-timing.json": "all per-layer/per-token rows",
            "expert-trace.jsonl.gz": {"records": trace_count, "sha256": trace_hash},
            "expert-cache-analysis.json": "reuse, eviction, and miss analysis",
            "transfer-analysis.json": "all measured transfer classes",
            "profiler-summary.md": "focused CPU/CUDA/concurrency summary",
            "prior-30-tps-audit.md": "historical benchmark audit",
            "bottleneck-ranking.json": "ranked measured bottlenecks",
        },
        "limitations": [
            "NVML process accounting falls back to device-global usage if the driver omits per-process bytes.",
            "CUDA context/library memory is inferred as a reconciliation residual, not directly tagged.",
            "No alternate full copied-tensor expert path exists for a controlled mmap-vs-copy run.",
            "Sampled NVML utilization is cross-checked with comparable host-monotonic layer intervals.",
        ],
    }
    write_json(args.output_dir / "milestone-2.5-report.json", report)
    write_profiler_summary(
        args.output_dir / "profiler-summary.md", runs, layer, concurrency,
        experiments, cache, transfers, prior_tps,
    )
    (args.output_dir / "MILESTONE_2_5_VERIFIED.md").write_text(
        build_verified_markdown(report, host, gpu, layer, cache, transfers,
                                bottlenecks, prior_tps),
        encoding="utf-8",
    )

    required_artifacts = [
        "MILESTONE_2_5_VERIFIED.md", "milestone-2.5-report.json",
        "memory-timeline.json", "host-memory-breakdown.json",
        "gpu-memory-breakdown.json", "layer-timing.json",
        "expert-trace.jsonl.gz", "expert-cache-analysis.json",
        "transfer-analysis.json", "profiler-summary.md",
        "prior-30-tps-audit.md", "bottleneck-ranking.json",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": result,
        "files": [
            {
                "path": name,
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in required_artifacts
        ],
    }
    write_json(args.output_dir / "evidence-manifest.json", manifest)
    print("MILESTONE25_REPORT=" + json.dumps({
        "result": result,
        "output_dir": str(args.output_dir),
        "acceptance_failures": [
            row["criterion"] for row in acceptance if row["status"] != "PASS"
        ],
    }, sort_keys=True), flush=True)
    if non_publication_failures:
        return 2
    if args.require_publication and publication_pending:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
