#!/usr/bin/env python3
"""Build a measured Ornith full-token critical-path and bandwidth roofline.

This analyzer intentionally keeps three evidence classes separate:

* measured: values read from run reports, layer timers, and the CUDA trace;
* calculated: exact arithmetic over the real tensor map and measured values;
* inferred: explicitly labelled differential estimates where current timers
  combine a component with residual/unattributed time.

It does not claim that a static weight-byte floor is a hardware-counter DRAM
measurement.  Nsight Compute counters remain the gate for that stronger claim.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
EXPERT_NAME_RE = re.compile(r"(?:^|\.)layers\.(\d+)\..*?experts\.(\d+)\.")
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "FLOAT16": 2,
    "FP16": 2,
    "F32": 4,
    "FLOAT32": 4,
    "FP32": 4,
    "I64": 8,
    "INT64": 8,
}
ACTIVE_COMPONENTS = {
    "full_attention",
    "linear_attention",
    "lm_head",
    "normalization",
    "router",
    "shared_expert",
    "token_embedding",
}
LAYER_FIELDS = {
    "full_attention": "attention_wall_ms",
    "router": "router_wall_ms",
    "expert_lookup": "expert_lookup_ms",
    "expert_compute": "expert_compute_cuda_ms",
    "expert_output_combination": "expert_output_combination_wall_ms",
    "shared_expert_gate": "shared_expert_gate_wall_ms",
    "shared_expert": "shared_expert_wall_ms",
    "normalization": "normalization_wall_ms",
    "inter_device_transfer": "inter_device_transfer_wall_ms",
    "explicit_synchronization": "synchronization_ms",
    "residual_and_unattributed": "residual_and_unattributed_wall_ms",
}


def read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_source(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def interval_union_us(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end >= start)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def positive_gaps_us(intervals: Iterable[tuple[float, float]]) -> list[float]:
    ordered = sorted((start, end) for start, end in intervals if end >= start)
    if not ordered:
        return []
    gaps: list[float] = []
    end = ordered[0][1]
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            gaps.append(next_start - end)
        end = max(end, next_end)
    return gaps


def dtype_nbytes(dtype: str) -> int:
    key = dtype.upper().replace("TORCH.", "")
    if key not in DTYPE_BYTES:
        raise ValueError(f"unsupported tensor-map dtype: {dtype}")
    return DTYPE_BYTES[key]


def analyze_tensor_map(tensor_map: dict[str, Any], topk: int) -> dict[str, Any]:
    tensors = tensor_map.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("tensor map has no tensors")

    component_bytes: Counter[str] = Counter()
    device_component_bytes: dict[str, Counter[str]] = defaultdict(Counter)
    resident_device_bytes: Counter[str] = Counter()
    expert_groups: Counter[tuple[int, int, str]] = Counter()
    embedding_row_by_device: Counter[str] = Counter()

    for tensor in tensors:
        component = str(tensor["target_runtime_component"])
        device = str(tensor["target_device"])
        byte_size = int(tensor["byte_size"])
        component_bytes[component] += byte_size
        if device.startswith("cuda:"):
            device_component_bytes[device][component] += byte_size
            if tensor.get("expected_lifetime") == "generation_resident":
                resident_device_bytes[device] += byte_size

        if component == "routed_expert":
            match = EXPERT_NAME_RE.search(str(tensor["tensor_name"]))
            if not match:
                raise ValueError(
                    "cannot identify layer/expert in routed tensor "
                    f"{tensor['tensor_name']}"
                )
            expert_groups[(int(match.group(1)), int(match.group(2)), device)] += byte_size
        elif component == "token_embedding":
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) < 2:
                raise ValueError(f"invalid embedding shape: {shape!r}")
            embedding_row_by_device[device] += (
                math.prod(int(dim) for dim in shape[1:])
                * dtype_nbytes(str(tensor["dtype"]))
            )

    layers: dict[int, dict[int, tuple[int, str]]] = defaultdict(dict)
    for (layer, expert, device), byte_size in expert_groups.items():
        if expert in layers[layer]:
            raise ValueError(f"expert {layer}:{expert} spans multiple devices")
        layers[layer][expert] = (byte_size, device)
    if not layers:
        raise ValueError("no routed expert groups found")

    active_routed_by_device: Counter[str] = Counter()
    expert_sizes: set[int] = set()
    for layer, experts in sorted(layers.items()):
        if len(experts) < topk:
            raise ValueError(f"layer {layer} has only {len(experts)} experts")
        sizes = {size for size, _device in experts.values()}
        if len(sizes) != 1:
            raise ValueError(f"layer {layer} expert byte sizes differ: {sorted(sizes)}")
        expert_size = next(iter(sizes))
        expert_sizes.add(expert_size)
        devices = {device for _size, device in experts.values()}
        if len(devices) != 1:
            raise ValueError(f"layer {layer} experts span devices: {sorted(devices)}")
        active_routed_by_device[next(iter(devices))] += expert_size * topk
    if len(expert_sizes) != 1:
        raise ValueError(f"expert byte sizes differ by layer: {sorted(expert_sizes)}")

    active_static_by_device: Counter[str] = Counter()
    excluded_by_device: dict[str, Counter[str]] = defaultdict(Counter)
    for device, components in device_component_bytes.items():
        for component, byte_size in components.items():
            if component == "token_embedding":
                active_static_by_device[device] += embedding_row_by_device[device]
                excluded_by_device[device]["unused_embedding_rows"] += (
                    byte_size - embedding_row_by_device[device]
                )
            elif component in ACTIVE_COMPONENTS:
                active_static_by_device[device] += byte_size
            elif component != "routed_expert":
                excluded_by_device[device][component] += byte_size

    mandatory_by_device = {
        device: active_static_by_device[device] + active_routed_by_device[device]
        for device in sorted(set(active_static_by_device) | set(active_routed_by_device))
    }
    return {
        "evidence_class": "calculated_from_real_tensor_map",
        "tensor_count": len(tensors),
        "tensor_map_total_bytes": sum(component_bytes.values()),
        "component_bytes": dict(sorted(component_bytes.items())),
        "generation_resident_bytes_by_device": dict(sorted(resident_device_bytes.items())),
        "active_static_weight_bytes_by_device": dict(sorted(active_static_by_device.items())),
        "active_routed_weight_bytes_by_device": dict(sorted(active_routed_by_device.items())),
        "mandatory_weight_bytes_by_device": mandatory_by_device,
        "mandatory_weight_bytes_per_token": sum(mandatory_by_device.values()),
        "max_device_mandatory_weight_bytes": max(mandatory_by_device.values()),
        "routed_layers": len(layers),
        "experts_per_layer": sorted({len(experts) for experts in layers.values()}),
        "topk": topk,
        "bytes_per_expert": next(iter(expert_sizes)),
        "embedding_row_bytes_by_device": dict(sorted(embedding_row_by_device.items())),
        "excluded_bytes_by_device": {
            device: dict(sorted(values.items()))
            for device, values in sorted(excluded_by_device.items())
        },
        "limitation": (
            "Mandatory weight bytes are an algorithmic lower bound, not measured "
            "HBM/DRAM traffic. They exclude activation traffic, allocator traffic, "
            "cache-line effects, and rereads caused by the implementation."
        ),
    }


def generation_summary(run_report: dict[str, Any]) -> dict[str, Any]:
    generation = run_report["generation"]
    state_groups = generation["recurrent_or_kv_state"]["groups"]
    state_by_device: Counter[str] = Counter()
    state_by_dtype: Counter[str] = Counter()
    for group in state_groups:
        byte_size = int(group["bytes"])
        state_by_device[str(group["device"])] += byte_size
        state_by_dtype[str(group["dtype"])] += byte_size
    state_bytes = sum(state_by_device.values())
    return {
        "evidence_class": "measured_run_report",
        "single_stream_decode_tokens_per_second": float(
            generation["single_stream_decode_tokens_per_second"]
        ),
        "per_token_decode_seconds": [
            float(value) for value in generation["per_token_decode_seconds"]
        ],
        "prefill_seconds": float(generation["prefill_seconds"]),
        "peak_host_rss_bytes": int(generation["resources"]["peak_host_rss_bytes"]),
        "peak_vram_bytes": {
            str(device): int(value)
            for device, value in generation["resources"]["peak_vram_bytes"].items()
        },
        "recurrent_or_kv_state_bytes_by_device": dict(sorted(state_by_device.items())),
        "recurrent_or_kv_state_bytes_by_dtype": dict(sorted(state_by_dtype.items())),
        "recurrent_or_kv_state_total_bytes": state_bytes,
        "state_read_plus_write_floor_bytes_per_token": state_bytes * 2,
        "state_traffic_limitation": (
            "The read-plus-write value is an algorithmic lower bound; hardware "
            "counters are required to establish actual state traffic."
        ),
    }


def analyze_layer_timing(layer_timing: dict[str, Any]) -> dict[str, Any]:
    rows = layer_timing.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("layer timing has no rows")
    decode_steps = sorted(
        {int(row["step"]) for row in rows if row.get("phase") == "decode"}
    )
    if not decode_steps:
        raise ValueError("layer timing has no decode rows")
    step = decode_steps[-1]
    selected = [
        row
        for row in rows
        if row.get("phase") == "decode" and int(row["step"]) == step
    ]
    if not selected:
        raise AssertionError("latest decode step selection is empty")

    sums = {
        name: sum(float(row.get(field, 0.0)) for row in selected)
        for name, field in LAYER_FIELDS.items()
    }
    total_ms = sum(float(row["total_layer_wall_ms"]) for row in selected)
    counts = Counter(str(row["layer_type"]) for row in selected)

    full_rows = [row for row in selected if row["layer_type"] == "full_attention"]
    linear_rows = [row for row in selected if row["layer_type"] == "linear_attention"]
    if not full_rows or not linear_rows:
        raise ValueError("need both full_attention and linear_attention rows")
    full_residual_avg = statistics.mean(
        float(row["residual_and_unattributed_wall_ms"]) for row in full_rows
    )
    linear_residual_total = sum(
        float(row["residual_and_unattributed_wall_ms"]) for row in linear_rows
    )
    inferred_linear_attention_ms = max(
        0.0, linear_residual_total - full_residual_avg * len(linear_rows)
    )
    remaining_residual_ms = sums["residual_and_unattributed"] - inferred_linear_attention_ms

    derived = {
        key: value
        for key, value in sums.items()
        if key != "residual_and_unattributed"
    }
    derived["linear_attention_inferred"] = inferred_linear_attention_ms
    derived["remaining_residual_and_unattributed"] = remaining_residual_ms
    ranked = [
        {
            "component": name,
            "milliseconds": round_float(value),
            "percent_of_layer_wall": round_float(value / total_ms * 100.0),
            "evidence_class": (
                "inferred_differential"
                if name == "linear_attention_inferred"
                else (
                    "derived_residual"
                    if name == "remaining_residual_and_unattributed"
                    else "measured_layer_timer"
                )
            ),
        }
        for name, value in sorted(derived.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "evidence_class": "measured_layer_timer_with_labelled_differential",
        "selected_phase": "decode",
        "selected_step": step,
        "layer_count": len(selected),
        "layer_type_counts": dict(sorted(counts.items())),
        "total_layer_wall_ms": round_float(total_ms),
        "measured_field_sums_ms": {
            key: round_float(value) for key, value in sums.items()
        },
        "expert_h2d_bytes": sum(
            int(row.get("expert_h2d_bytes", 0)) for row in selected
        ),
        "derived_ranked_critical_path": ranked,
        "linear_attention_differential": {
            "evidence_class": "inferred_differential",
            "full_attention_layer_residual_average_ms": round_float(full_residual_avg),
            "linear_attention_layer_residual_total_ms": round_float(
                linear_residual_total
            ),
            "inferred_linear_attention_ms": round_float(inferred_linear_attention_ms),
            "method": (
                "Subtract the average full-attention-layer residual from each "
                "linear-attention-layer residual. This is a ranking estimate, not "
                "a direct timer."
            ),
        },
    }


def kernel_family(name: str) -> str:
    lowered = name.lower()
    if "gemvx::kernel" in lowered:
        return "cublas_gemvx"
    if "gemv2t_kernel" in lowered:
        return "cublas_gemv2t"
    if "gemmk1_kernel" in lowered:
        return "cublas_gemmk1"
    if "swiglu_activation" in lowered:
        return "dee_swiglu_activation"
    if "direct_copy" in lowered:
        return "aten_direct_copy"
    if "reduce" in lowered:
        return "reduction"
    if "dot_kernel" in lowered:
        return "cublas_dot"
    if "bitonicsort" in lowered:
        return "aten_bitonic_sort"
    if "catarraybatchedcopy" in lowered:
        return "aten_cat_copy"
    if "elementwise" in lowered:
        return "elementwise"
    if "index" in lowered or "gather" in lowered or "topk" in lowered:
        return "index_gather_topk"
    return name[:160]


def analyze_trace(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("trace has no traceEvents")
    gpu_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") in {"kernel", "gpu_memcpy"}
        and float(event.get("dur", 0.0)) >= 0.0
    ]
    kernels = [event for event in gpu_events if event.get("cat") == "kernel"]
    memcpys = [event for event in gpu_events if event.get("cat") == "gpu_memcpy"]
    runtime = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "cuda_runtime"
    ]
    intervals = [
        (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
        for event in gpu_events
    ]
    span_us = (
        max(end for _start, end in intervals) - min(start for start, _end in intervals)
        if intervals
        else 0.0
    )
    busy_us = interval_union_us(intervals)
    gaps = positive_gaps_us(intervals)

    by_device: dict[str, dict[str, Any]] = {}
    devices = sorted(
        {
            int(event.get("args", {}).get("device"))
            for event in gpu_events
            if event.get("args", {}).get("device") is not None
        }
    )
    for device in devices:
        selected = [
            event
            for event in gpu_events
            if int(event.get("args", {}).get("device", -1)) == device
        ]
        device_intervals = [
            (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
            for event in selected
        ]
        device_span = (
            max(end for _start, end in device_intervals)
            - min(start for start, _end in device_intervals)
            if device_intervals
            else 0.0
        )
        device_busy = interval_union_us(device_intervals)
        by_device[f"cuda:{device}"] = {
            "event_count": len(selected),
            "kernel_count": sum(event.get("cat") == "kernel" for event in selected),
            "memcpy_count": sum(
                event.get("cat") == "gpu_memcpy" for event in selected
            ),
            "span_ms": round_float(device_span / 1000.0),
            "busy_ms": round_float(device_busy / 1000.0),
            "busy_percent": round_float(
                device_busy / device_span * 100.0 if device_span else 0.0
            ),
        }

    memcpy_groups: dict[str, dict[str, float | int]] = {}
    for name in sorted({str(event["name"]) for event in memcpys}):
        selected = [event for event in memcpys if str(event["name"]) == name]
        memcpy_groups[name] = {
            "count": len(selected),
            "bytes": sum(int(event.get("args", {}).get("bytes", 0)) for event in selected),
            "duration_ms": round_float(
                sum(float(event["dur"]) for event in selected) / 1000.0
            ),
        }

    families: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "duration_us": 0.0}
    )
    for event in kernels:
        family = kernel_family(str(event["name"]))
        families[family]["count"] = int(families[family]["count"]) + 1
        families[family]["duration_us"] = float(
            families[family]["duration_us"]
        ) + float(event["dur"])
    top_families = [
        {
            "family": family,
            "count": int(values["count"]),
            "duration_ms": round_float(float(values["duration_us"]) / 1000.0),
        }
        for family, values in sorted(
            families.items(),
            key=lambda item: float(item[1]["duration_us"]),
            reverse=True,
        )[:20]
    ]
    runtime_counts = Counter(str(event["name"]) for event in runtime)
    return {
        "evidence_class": "measured_torch_cuda_trace",
        "trace_event_count": len(events),
        "kernel_count": len(kernels),
        "gpu_memcpy_count": len(memcpys),
        "cuda_runtime_call_counts": {
            name: runtime_counts.get(name, 0)
            for name in (
                "cudaLaunchKernel",
                "cudaStreamSynchronize",
                "cudaDeviceSynchronize",
                "cudaStreamWaitEvent",
                "cudaEventRecord",
                "cudaEventRecordWithFlags",
            )
        },
        "gpu_timeline": {
            "span_ms": round_float(span_us / 1000.0),
            "busy_ms": round_float(busy_us / 1000.0),
            "idle_ms": round_float((span_us - busy_us) / 1000.0),
            "busy_percent": round_float(
                busy_us / span_us * 100.0 if span_us else 0.0
            ),
            "positive_gap_count": len(gaps),
            "gaps_over_10us": sum(gap > 10.0 for gap in gaps),
            "median_positive_gap_us": round_float(
                statistics.median(gaps) if gaps else 0.0
            ),
            "max_positive_gap_us": round_float(max(gaps) if gaps else 0.0),
        },
        "by_device": by_device,
        "memcpy_groups": memcpy_groups,
        "top_kernel_families": top_families,
        "interpretation": (
            "The trace is launch/GEMV-fragmentation dominated when thousands of "
            "launches coexist with low union busy time and negligible expert H2D. "
            "Trace overhead means its wall time is not the unprofiled throughput."
        ),
    }


def build_roofline(
    tensor_analysis: dict[str, Any],
    control: dict[str, Any],
    realistic_gbps: float,
    theoretical_gbps: float,
) -> dict[str, Any]:
    mandatory = int(tensor_analysis["mandatory_weight_bytes_per_token"])
    max_device = int(tensor_analysis["max_device_mandatory_weight_bytes"])
    achieved_tps = float(control["single_stream_decode_tokens_per_second"])
    effective_gbps = mandatory * achieved_tps / 1e9

    ceilings: dict[str, Any] = {}
    for label, bandwidth in (
        ("realistic_sustained", realistic_gbps),
        ("theoretical_peak", theoretical_gbps),
    ):
        ceilings[label] = {
            "bandwidth_gb_per_second": bandwidth,
            "single_sequence_sequential_device_tps": round_float(
                bandwidth * 1e9 / mandatory
            ),
            "ideal_multi_sequence_pipeline_tps": round_float(
                bandwidth * 1e9 / max_device
            ),
        }
    targets = {
        str(target): {
            "required_effective_weight_bandwidth_gb_per_second": round_float(
                mandatory * target / 1e9
            ),
            "percent_of_realistic_sustained_bandwidth": round_float(
                mandatory * target / 1e9 / realistic_gbps * 100.0
            ),
        }
        for target in (20, 30)
    }
    storage_scenarios = {}
    for name, factor, qualification in (
        ("bf16_or_fp16", 1.0, "Current two-byte weight storage floor."),
        (
            "int8_or_fp8_storage",
            0.5,
            "Ideal all-weight storage floor only. T4 has no native FP8 tensor-core path.",
        ),
        (
            "int4_storage",
            0.25,
            "Ideal all-weight storage floor only; excludes scales and dequantization.",
        ),
    ):
        scaled = mandatory * factor
        storage_scenarios[name] = {
            "weight_byte_factor": factor,
            "mandatory_weight_bytes_per_token": round(scaled),
            "single_sequence_tps_at_realistic_bandwidth": round_float(
                realistic_gbps * 1e9 / scaled
            ),
            "qualification": qualification,
        }
    return {
        "evidence_class": "calculated_roofline_not_hardware_counter",
        "mandatory_weight_bytes_per_token": mandatory,
        "max_device_mandatory_weight_bytes": max_device,
        "achieved_unprofiled_tps": achieved_tps,
        "achieved_effective_mandatory_weight_bandwidth_gb_per_second": round_float(
            effective_gbps
        ),
        "achieved_percent_of_realistic_sustained_bandwidth": round_float(
            effective_gbps / realistic_gbps * 100.0
        ),
        "ceilings": ceilings,
        "targets": targets,
        "storage_precision_scenarios": storage_scenarios,
        "mtp_or_speculative_decoding": {
            "projection": None,
            "reason": (
                "Effective TPS depends on measured draft acceptance, verification "
                "cost, and faithful MTP-head execution. No multiplier is claimed "
                "without those measurements."
            ),
        },
        "verdict_20_to_30_tps": (
            "Physically plausible under the static-weight bandwidth floor: both "
            "targets are below the realistic sequential-device ceiling. Not yet "
            "demonstrated: measured trace utilization shows launch/orchestration "
            "latency must be removed before the bandwidth ceiling is approached."
        ),
        "required_next_measurement": (
            "Collect Nsight Compute dram__bytes_read.sum and achieved DRAM "
            "throughput on a representative steady-state decode token."
        ),
    }


def historical_boundary(audit_path: Path) -> dict[str, Any]:
    text = audit_path.read_text(encoding="utf-8")
    required = (
        "30.571",
        "synthetic recurrent MoE steps per second",
        "75d5218ee7385985119f41e86375ba102755e274",
        "inter=64",
    )
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise ValueError(f"historical audit missing anchors: {missing}")
    return {
        "evidence_class": "tracked_forensic_audit",
        "headline_rate": 30.571,
        "truthful_unit": "synthetic recurrent MoE steps per second",
        "implementation_commit": "75d5218ee7385985119f41e86375ba102755e274",
        "hardware": "one Tesla T4",
        "physical_expert_layers": 1,
        "logical_moe_calls_per_step": 40,
        "topk": 8,
        "expert_intermediate_width": 64,
        "full_token_intermediate_width": 512,
        "full_token_costs_absent": [
            "tokenizer and vocabulary embedding/logits/sampling",
            "full and linear attention",
            "real recurrent state",
            "learned norms and residual stack",
            "real 40-layer checkpoint experts and routers",
            "dual-GPU placement and communication",
        ],
        "conversion_to_full_token_tps": None,
        "conclusion": (
            "The historical number cannot be compared numerically with genuine "
            "Ornith text-generation TPS."
        ),
    }


def make_markdown(report: dict[str, Any]) -> str:
    tensor = report["tensor_map_analysis"]
    control = report["unprofiled_control"]
    layers = report["critical_path"]
    trace = report["cuda_trace"]
    roof = report["roofline"]
    lines = [
        "# Ornith full-token T4 critical path and roofline",
        "",
        "Status: reproducible analysis complete. Values are labelled as measured, "
        "calculated, or inferred; no static byte floor is represented as a hardware "
        "counter.",
        "",
        "## Bottom line",
        "",
        f"- Accepted cap-32 unprofiled throughput: "
        f"**{control['single_stream_decode_tokens_per_second']:.6f} TPS**.",
        f"- Peak process VRAM: "
        + ", ".join(
            f"{device} {value / 2**30:.3f} GiB"
            for device, value in control["peak_vram_bytes"].items()
        )
        + ".",
        f"- Calculated mandatory active-weight floor: "
        f"**{tensor['mandatory_weight_bytes_per_token']:,} B/token** "
        f"({tensor['mandatory_weight_bytes_per_token'] / 2**30:.3f} GiB/token).",
        f"- Effective bandwidth at the measured rate: "
        f"**{roof['achieved_effective_mandatory_weight_bandwidth_gb_per_second']:.3f} "
        f"GB/s**, or "
        f"**{roof['achieved_percent_of_realistic_sustained_bandwidth']:.2f}%** "
        "of the 270 GB/s sustained assumption.",
        f"- The trace launches **{trace['kernel_count']:,} kernels** and its combined "
        f"GPU kernel/memcpy timeline is busy only "
        f"**{trace['gpu_timeline']['busy_percent']:.2f}%**. This is a "
        "launch/GEMV-fragmentation problem, not an expert-H2D problem.",
        f"- Verdict: **20-30 TPS is physically plausible but not demonstrated.** "
        f"The 270 GB/s single-sequence static floor is "
        f"{roof['ceilings']['realistic_sustained']['single_sequence_sequential_device_tps']:.2f} "
        "TPS; actual DRAM counters and substantially lower launch latency are still "
        "required.",
        "",
        "## Real tensor-map byte floor",
        "",
        "| Device | Active static B/token | Routed top-8 B/token | Total B/token |",
        "|---|---:|---:|---:|",
    ]
    for device, total in tensor["mandatory_weight_bytes_by_device"].items():
        lines.append(
            f"| {device} | "
            f"{tensor['active_static_weight_bytes_by_device'].get(device, 0):,} | "
            f"{tensor['active_routed_weight_bytes_by_device'].get(device, 0):,} | "
            f"{total:,} |"
        )
    lines += [
        "",
        "Only one token-embedding row is counted; the full LM head is counted. "
        "The routed term is eight complete experts per layer across all 40 layers. "
        "The floor excludes activation traffic, allocator effects, cache-line "
        "effects, and implementation rereads.",
        "",
        "## Measured latest decode-step critical path",
        "",
        f"The selected layer-timer step covers {layers['layer_count']} layers and "
        f"{layers['total_layer_wall_ms']:.3f} ms. The profiler perturbs wall time, "
        "so use this ranking, not its TPS, as the baseline.",
        "",
        "| Rank | Component | ms | Layer-wall share | Evidence |",
        "|---:|---|---:|---:|---|",
    ]
    for rank, row in enumerate(layers["derived_ranked_critical_path"], 1):
        lines.append(
            f"| {rank} | {row['component']} | {row['milliseconds']:.3f} | "
            f"{row['percent_of_layer_wall']:.2f}% | {row['evidence_class']} |"
        )
    lines += [
        "",
        "The linear-attention row is a labelled differential estimate because the "
        "current timer records linear attention inside residual/unattributed time. "
        "It subtracts the mean full-attention residual from each linear-attention "
        "layer; it is not a direct CUDA counter.",
        "",
        "## CUDA trace",
        "",
        f"- Timeline: {trace['gpu_timeline']['span_ms']:.3f} ms span, "
        f"{trace['gpu_timeline']['busy_ms']:.3f} ms busy, "
        f"{trace['gpu_timeline']['idle_ms']:.3f} ms idle.",
        f"- Runtime calls: "
        + ", ".join(
            f"{name}={count:,}"
            for name, count in trace["cuda_runtime_call_counts"].items()
        )
        + ".",
        f"- Expert H2D recorded by the layer timer: "
        f"{layers['expert_h2d_bytes']:,} B "
        "(the cap-32 experts were resident).",
        f"- Positive GPU gaps: {trace['gpu_timeline']['positive_gap_count']:,}; "
        f"{trace['gpu_timeline']['gaps_over_10us']:,} exceed 10 us; median "
        f"{trace['gpu_timeline']['median_positive_gap_us']:.3f} us.",
        "",
        "| Kernel family | Calls | CUDA duration ms |",
        "|---|---:|---:|",
    ]
    for row in trace["top_kernel_families"][:10]:
        lines.append(
            f"| {row['family']} | {row['count']:,} | {row['duration_ms']:.3f} |"
        )
    lines += [
        "",
        "## Roofline and target bandwidth",
        "",
        "| Assumption | Bandwidth GB/s | Sequential-device TPS | Ideal pipelined TPS |",
        "|---|---:|---:|---:|",
    ]
    for label, row in roof["ceilings"].items():
        lines.append(
            f"| {label} | {row['bandwidth_gb_per_second']:.1f} | "
            f"{row['single_sequence_sequential_device_tps']:.2f} | "
            f"{row['ideal_multi_sequence_pipeline_tps']:.2f} |"
        )
    lines += [
        "",
        "| Target | Required effective GB/s | Share of 270 GB/s |",
        "|---:|---:|---:|",
    ]
    for target, row in roof["targets"].items():
        lines.append(
            f"| {target} TPS | "
            f"{row['required_effective_weight_bandwidth_gb_per_second']:.3f} | "
            f"{row['percent_of_realistic_sustained_bandwidth']:.2f}% |"
        )
    lines += [
        "",
        "FP16 and BF16 have the same two-byte storage floor. Ideal INT8/FP8 and "
        "INT4 storage floors are included in JSON only as qualified arithmetic, "
        "not throughput promises: T4 has no native FP8 tensor-core path, and "
        "scales/dequantization are not free. No MTP/speculative multiplier is "
        "claimed without measured acceptance and verification cost.",
        "",
        "## Historical approximately-30 result",
        "",
        "The historical 30.571 figure is a single-T4 rate in **synthetic recurrent "
        "MoE steps/s**, at commit `75d5218`, with one physical expert layer "
        "replayed as 40 logical calls, intermediate width 64, generated weights, "
        "and no attention, recurrent state, vocabulary head, or dual-GPU path. It "
        "has no valid conversion to full-token Ornith TPS.",
        "",
        "## Next optimization gate",
        "",
        "The trace and layer timing nominate the linear-attention/recurrent path and "
        "Python/PyTorch launch fragmentation as the first target. A faithful next "
        "Pareto point must preserve exact token IDs/text, all 40 routed layers, "
        "zero host fallback, no trace abort, and <=8 GiB process VRAM per GPU. "
        "Before claiming a bandwidth-bound result, collect Nsight Compute "
        "`dram__bytes_read.sum` and achieved-DRAM-throughput counters on a steady "
        "decode token.",
        "",
        "## Source integrity",
        "",
    ]
    for name, source in report["sources"].items():
        lines.append(
            f"- `{name}`: `{source['sha256']}` ({source['bytes']:,} bytes), "
            f"`{source['path']}`"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--control-report", required=True, type=Path)
    parser.add_argument("--profiled-report", required=True, type=Path)
    parser.add_argument("--layer-timing", required=True, type=Path)
    parser.add_argument("--torch-trace", required=True, type=Path)
    parser.add_argument("--historical-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--realistic-bandwidth-gbps", type=float, default=270.0)
    parser.add_argument("--theoretical-bandwidth-gbps", type=float, default=320.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "tensor_map": args.tensor_map,
        "control_report": args.control_report,
        "profiled_report": args.profiled_report,
        "layer_timing": args.layer_timing,
        "torch_trace": args.torch_trace,
        "historical_audit": args.historical_audit,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    if args.topk <= 0:
        raise ValueError("--topk must be positive")

    tensor = analyze_tensor_map(read_json(args.tensor_map), args.topk)
    control_report = read_json(args.control_report)
    profiled_report = read_json(args.profiled_report)
    control = generation_summary(control_report)
    profiled = generation_summary(profiled_report)
    critical_path = analyze_layer_timing(read_json(args.layer_timing))
    trace = analyze_trace(read_json(args.torch_trace))
    roofline = build_roofline(
        tensor,
        control,
        args.realistic_bandwidth_gbps,
        args.theoretical_bandwidth_gbps,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "result": "PASS",
        "analysis_scope": "genuine_ornith_full_token_dual_t4_cap32",
        "sources": {name: canonical_source(path) for name, path in paths.items()},
        "tensor_map_analysis": tensor,
        "unprofiled_control": control,
        "profiled_run": profiled,
        "profiler_slowdown": {
            "profiled_tps": profiled["single_stream_decode_tokens_per_second"],
            "unprofiled_tps": control["single_stream_decode_tokens_per_second"],
            "profiled_over_unprofiled_ratio": round_float(
                profiled["single_stream_decode_tokens_per_second"]
                / control["single_stream_decode_tokens_per_second"]
            ),
            "qualification": (
                "Use the profiled run for ranking only; use the unprofiled control "
                "for throughput."
            ),
        },
        "critical_path": critical_path,
        "cuda_trace": trace,
        "roofline": roofline,
        "historical_30_tps_boundary": historical_boundary(args.historical_audit),
        "hard_requirements": {
            "hardware_dram_counters_present": False,
            "measured_full_token_timing_present": True,
            "real_tensor_map_present": True,
            "measured_cuda_trace_present": True,
            "static_floor_mislabeled_as_measured_dram": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "full-token-roofline.json"
    critical_path_file = args.output_dir / "critical-path-profile.json"
    markdown_path = args.output_dir / "ORNITH_FULL_TOKEN_ROOFLINE.md"
    full_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    critical_path_file.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "result": "PASS",
                "sources": {
                    name: report["sources"][name]
                    for name in ("profiled_report", "layer_timing", "torch_trace")
                },
                "profiler_slowdown": report["profiler_slowdown"],
                "critical_path": critical_path,
                "cuda_trace": trace,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(make_markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": "PASS",
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (full_path, critical_path_file, markdown_path)
        },
    }
    manifest_path = args.output_dir / "analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": "PASS",
                "output_dir": str(args.output_dir),
                "unprofiled_tps": control["single_stream_decode_tokens_per_second"],
                "mandatory_weight_bytes_per_token": tensor[
                    "mandatory_weight_bytes_per_token"
                ],
                "trace_busy_percent": trace["gpu_timeline"]["busy_percent"],
                "sequential_tps_at_realistic_bandwidth": roofline["ceilings"][
                    "realistic_sustained"
                ]["single_sequence_sequential_device_tps"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
