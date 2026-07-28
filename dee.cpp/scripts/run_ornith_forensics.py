#!/usr/bin/env python3
"""Run one controlled Ornith Milestone 2.5 forensic experiment.

The driver intentionally performs no optimization.  It layers bounded probes
over the verified Milestone 2 execution path and emits raw, attributable
evidence for later cross-experiment analysis.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_ENTRY_NS = time.time_ns()
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.milestone25_memory import MemoryProbe, inventory_tensors  # noqa: E402


BASELINE_TOKENS = {
    "Hello": [11, 271, 40, 1044],
    "2+2=": [19, 271, 248068, 198],
    "Paris": [11, 279, 4170, 314],
    # Exact M3 v4 full-runtime output prefix. This is a regression oracle
    # for the seventh matrix variant; router/layer-0 reference parity is
    # independently checked against Transformers later in the notebook.
    "The quick brown fox jumps over the lazy dog.": [198, 760],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--split-layer", type=int, default=20)
    parser.add_argument("--cache-experts", type=int, default=8)
    parser.add_argument(
        "--router-backend",
        choices=("native-host", "torch-device"),
        default="native-host",
        help=(
            "native-host reproduces the sealed baseline; torch-device keeps the "
            "official checkpoint router on the layer GPU"
        ),
    )
    parser.add_argument("--cache-disabled", action="store_true")
    parser.add_argument("--allow-sub-topk-cache", action="store_true")
    parser.add_argument("--classification", choices=("cold", "warm", "diagnostic"),
                        default="diagnostic")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-requests", action="store_true")
    parser.add_argument("--profile-timeline", action="store_true")
    parser.add_argument(
        "--debug-validate-cache", action="store_true",
        help="Enable expensive native cache invariant checks for validation runs.",
    )
    parser.add_argument("--torch-profiler-step", type=int)
    parser.add_argument("--reference-parity", action="store_true")
    parser.add_argument("--warmup-generation", action="store_true")
    parser.add_argument("--require-dual-gpu", action="store_true")
    parser.add_argument("--tensor-map", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.cache_experts < 1:
        parser.error("--cache-experts must be positive")
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def nvidia_smi_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,pci.bus_id,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=20)
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            rows.append({
                "index": int(fields[0]), "name": fields[1],
                "pci_bus_id": fields[2], "driver_version": fields[3],
                "memory_total_mib": int(fields[4]),
            })
    return rows


def cpu_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {
        "logical_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        result["total_host_ram_bytes"] = int(memory.total)
        result["physical_cpu_count"] = psutil.cpu_count(logical=False)
    except Exception as exc:
        result["psutil_error"] = f"{type(exc).__name__}: {exc}"
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                result["model_name"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return result


class PhaseBridge:
    """Forward load phase timings and take the required memory checkpoints."""

    CHECKPOINT_PHASES = {
        "model_structure_construction",
        "tensor_metadata_creation",
        "dense_tensor_acquisition",
        "initial_mmap_and_engine_setup",
        "model_setup_finalization",
    }

    def __init__(self, recorder: Any, memory: MemoryProbe):
        self.recorder = recorder
        self.memory = memory

    def record_phase(self, name: str, seconds: float,
                     metadata: dict[str, Any] | None = None) -> None:
        self.recorder.record_phase(name, seconds, metadata)
        if name in self.CHECKPOINT_PHASES:
            self.memory.checkpoint(
                f"after_{name}", include_smaps=name == "initial_mmap_and_engine_setup",
                include_cuda=True, include_nvml=True,
                metadata={"phase_wall_ms": seconds * 1000.0, **(metadata or {})},
            )


def parameter_inventory(model: Any) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    storage_locations: dict[int, list[str]] = collections.defaultdict(list)
    for name, tensor in model.named_parameters():
        if "embed_tokens" in name:
            owner = "embedding"
        elif name.startswith("lm_head"):
            owner = "lm_head"
        elif ".self_attn." in name:
            owner = "attention"
        elif "layernorm" in name or name.startswith("model.norm"):
            owner = "normalization"
        elif ".mlp.gate." in name:
            owner = "router"
        elif ".mlp.shared_expert" in name:
            owner = "shared_expert"
        else:
            owner = "other_dense"
        key = (owner, str(tensor.device), str(tensor.dtype))
        row = groups.setdefault(key, {
            "owner": owner, "device": str(tensor.device), "dtype": str(tensor.dtype),
            "tensor_count": 0, "bytes": 0,
        })
        row["tensor_count"] += 1
        row["bytes"] += int(tensor.numel() * tensor.element_size())
        try:
            storage_locations[int(tensor.untyped_storage().data_ptr())].append(name)
        except Exception:
            pass
    aliases = [names for names in storage_locations.values() if len(names) > 1]
    return {
        "groups": sorted(groups.values(), key=lambda item: (
            item["device"], item["owner"], item["dtype"]
        )),
        "shared_storage_groups": aliases,
    }


def _profile_delta(current: dict[str, Any], previous: dict[str, Any] | None,
                   section: str, key: str) -> float:
    value = float(current.get(section, {}).get(key, 0.0))
    if previous is None:
        return value
    return value - float(previous.get(section, {}).get(key, 0.0))


def build_layer_timing(timing: dict[str, Any]) -> dict[str, Any]:
    spans = timing.get("wall_spans", [])
    span_groups: dict[tuple[int, int], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    cuda_groups: dict[tuple[int, int], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for span in spans:
        step = int(span.get("step", -1))
        layer = int(span.get("layer", -1))
        if step < 0 or layer < 0:
            continue
        span_groups[(step, layer)][span["name"]] += float(span.get("cpu_wall_ms", 0.0))
        if span.get("cuda_event_ms") is not None:
            cuda_groups[(step, layer)][span["name"]] += float(span["cuda_event_ms"])

    previous_by_layer: dict[int, dict[str, Any]] = {}
    rows = []
    for step_snapshot in timing.get("profile_snapshots", []):
        step = int(step_snapshot["step"])
        phase = step_snapshot["phase"]
        for layer_item in step_snapshot["layers"]:
            layer = int(layer_item["layer"])
            profile = layer_item["profile"]
            previous = previous_by_layer.get(layer)
            wall = span_groups[(step, layer)]
            cuda = cuda_groups[(step, layer)]
            requests = profile.get("requests", {})
            previous_requests = previous.get("requests", {}) if previous else {}
            transfers = profile.get("transfers", {})
            previous_transfers = previous.get("transfers", {}) if previous else {}
            normalization_ms = (
                wall["normalization_input"] + wall["normalization_post_attention"]
            )
            router_ms = sum(value for name, value in wall.items()
                            if name.startswith("router_"))
            expert_wall_ms = sum(value for name, value in wall.items()
                                 if name.startswith("expert_"))
            known_top_level = (
                normalization_ms + wall["attention_or_linear_attention"] +
                router_ms + expert_wall_ms + wall["shared_expert"] +
                wall["shared_expert_gate"]
            )
            total_layer_ms = wall["layer_total"]
            rows.append({
                "step": step,
                "phase": phase,
                "layer": layer,
                "layer_type": "linear_attention" if layer % 4 != 3 else "full_attention",
                "gpu": int(timing["engine_profiles"][layer]["gpu"]),
                "normalization_wall_ms": normalization_ms,
                "attention_wall_ms": wall["attention_or_linear_attention"],
                "attention_cuda_event_ms": cuda["attention_or_linear_attention"],
                "router_wall_ms": router_ms,
                "router_hidden_d2h_wall_ms": wall["router_hidden_gpu_to_cpu"],
                "router_native_wall_ms": wall["router_native"],
                "router_output_h2d_wall_ms": wall["router_outputs_cpu_to_gpu"],
                "router_canonicalization_wall_ms": wall["router_canonicalization"],
                "expert_lookup_ms": _profile_delta(
                    profile, previous, "cpu_ms", "cache_lookup"
                ),
                "cache_hits": int(requests.get("resident_hits", 0) +
                                  requests.get("inflight_hits", 0) -
                                  previous_requests.get("resident_hits", 0) -
                                  previous_requests.get("inflight_hits", 0)),
                "cache_misses": int(requests.get("cold_loads", 0) -
                                    previous_requests.get("cold_loads", 0)),
                "host_tensor_preparation_ms": _profile_delta(
                    profile, previous, "cpu_ms", "host_tensor_preparation"
                ),
                "pinning_ms": _profile_delta(profile, previous, "cpu_ms", "pinning"),
                "pageable_to_pinned_copy_ms": _profile_delta(
                    profile, previous, "cpu_ms", "mmap_to_pinned"
                ),
                "pageable_to_pinned_bytes": int(
                    transfers.get("mmap_to_pinned_bytes", 0) -
                    previous_transfers.get("mmap_to_pinned_bytes", 0)
                ),
                "h2d_submission_ms": _profile_delta(
                    profile, previous, "cpu_ms", "transfer_submission"
                ),
                "h2d_completion_cuda_ms": _profile_delta(
                    profile, previous, "gpu_ms", "h2d"
                ),
                "activation_h2d_cuda_ms": _profile_delta(
                    profile, previous, "gpu_ms", "activation_h2d"
                ),
                "activation_conversion_cuda_ms": _profile_delta(
                    profile, previous, "gpu_ms", "activation_conversion"
                ),
                "expert_output_d2h_cuda_ms": _profile_delta(
                    profile, previous, "gpu_ms", "d2h"
                ),
                "expert_compute_cuda_ms": sum(
                    _profile_delta(profile, previous, "gpu_ms", key)
                    for key in ("gate_projection", "up_projection", "silu_multiply",
                                "down_projection")
                ),
                "expert_native_wall_ms": wall["expert_native"],
                "expert_input_d2h_wall_ms": wall["expert_inputs_gpu_to_cpu"],
                "expert_output_h2d_wall_ms": wall["expert_outputs_cpu_to_gpu"],
                "expert_output_combination_wall_ms": wall["expert_output_combination"],
                "shared_expert_wall_ms": wall["shared_expert"],
                "shared_expert_gate_wall_ms": wall["shared_expert_gate"],
                "residual_and_unattributed_wall_ms": max(0.0, total_layer_ms - known_top_level),
                "inter_device_transfer_wall_ms": wall["inter_device_transfer"],
                "synchronization_ms": _profile_delta(
                    profile, previous, "cpu_ms", "synchronization"
                ),
                "total_layer_wall_ms": total_layer_ms,
                "expert_h2d_bytes": int(
                    transfers.get("h2d_bytes", 0) -
                    previous_transfers.get("h2d_bytes", 0)
                ),
            })
            previous_by_layer[layer] = profile
    return {"schema_version": 1, "rows": rows}


def _source_shards(index: dict[str, Any], layer: int, expert: int) -> list[str]:
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
    return sorted({
        index["weight_map"][f"{prefix}.{projection}.weight"]
        for projection in ("gate_proj", "up_proj", "down_proj")
    })


# ---------------------------------------------------------------------------
# Milestone 3 derived artifacts.  These consume the M2.5 forensic recorder
# output verbatim (no additional per-call instrumentation).  They exist so
# the analyzer can compare M3 vs M2.5 without re-parsing the M2.5 schema.
# ---------------------------------------------------------------------------


def build_synchronization_analysis(timing: dict[str, Any]) -> dict[str, Any]:
    """Aggregate host/device synchronization counters from timing wall_spans
    plus the per-layer ExternalProfile already augmented each step.

    Reads two distinct host-sync sources to cover the full M2.5 vocabulary:
      * cpu_ms.synchronization (StageProfile::add_cpu_ms(CpuStage::Synchronization))
      * profile.host_synchronizations (StageProfile::note_host_synchronization)
    only one of which is the "host_synchronizations: 2488" M2.5 baseline.
    Without reading both the M3 vs M2.5 comparison breaks silently.
    """
    spans = timing.get("wall_spans", [])
    by_name: dict[str, int] = collections.Counter()
    by_name_wall_ms: dict[str, float] = collections.Counter()
    for span in spans:
        name = span.get("name", "")
        by_name[name] += 1
        by_name_wall_ms[name] += float(span.get("cpu_wall_ms", 0.0))
    snapshots = timing.get("profile_snapshots", [])

    def snapshot_totals(snapshot: dict[str, Any]) -> tuple[int, int, float]:
        stream_waits = 0
        host_synchronizations = 0
        synchronization_ms = 0.0
        for layer_item in snapshot.get("layers", []):
            profile = layer_item.get("profile", {})
            operations = profile.get("operations", {})
            stream_waits += int(operations.get("stream_waits", 0) or 0)
            host_synchronizations += int(
                operations.get("host_synchronizations", 0) or 0
            )
            synchronization_ms += float(
                profile.get("cpu_ms", {}).get("synchronization", 0) or 0
            )
        return stream_waits, host_synchronizations, synchronization_ms

    # Native profile snapshots are cumulative. Compare the second decode
    # snapshot with its predecessor, matching the representative warm decode
    # used by the timing analyzer, instead of summing cumulative snapshots.
    decode_indices = [
        index for index, snapshot in enumerate(snapshots)
        if snapshot.get("phase") == "decode"
    ]
    selected_index = decode_indices[1] if len(decode_indices) > 1 else (
        decode_indices[0] if decode_indices else None
    )
    stream_waits_total = 0
    host_synchronizations_total_dedicated = 0
    host_synchronization_ms = 0.0
    selected_step = None
    if selected_index is not None:
        current = snapshot_totals(snapshots[selected_index])
        previous = snapshot_totals(snapshots[selected_index - 1]) if selected_index else (0, 0, 0.0)
        stream_waits_total = max(0, current[0] - previous[0])
        host_synchronizations_total_dedicated = max(0, current[1] - previous[1])
        host_synchronization_ms = max(0.0, current[2] - previous[2])
        selected_step = snapshots[selected_index].get("step")
    by_step: dict[int, dict[str, int]] = collections.defaultdict(
        lambda: collections.Counter()
    )
    model_level_excluded_spans_count = 0
    for span in spans:
        step = int(span.get("step", -1))
        if step < 0:
            model_level_excluded_spans_count += 1
            continue
        by_step[step][span.get("name", "")] += 1
    host_synchronizations_total = (
        host_synchronizations_total_dedicated
    )
    return {
        "schema_version": 2,
        "wall_spans_by_name": dict(by_name),
        "wall_ms_by_name": {name: round(value, 6) for name, value in by_name_wall_ms.items()},
        "stream_wait_events_total": stream_waits_total,
        "host_synchronization_events_total": host_synchronizations_total,
        "host_synchronization_events_breakdown": {
            "cpu_section_synchronization_ms_total": host_synchronization_ms,
            "dedicated_host_synchronization_counter_total": host_synchronizations_total_dedicated,
        },
        "representative_step": selected_step,
        "snapshot_scope": "representative_decode_delta",
        "step_breakdown": {f"step_{step}": dict(counts)
                            for step, counts in sorted(by_step.items())},
        "model_level_excluded_spans_count": model_level_excluded_spans_count,
    }


def representative_decode_nvml_samples(
    timing: dict[str, Any],
) -> tuple[list[dict[str, Any]], int | None]:
    decode_steps = [
        row for row in timing.get("steps", [])
        if row.get("phase") == "decode"
    ]
    selected = (
        decode_steps[1] if len(decode_steps) > 1
        else (decode_steps[0] if decode_steps else None)
    )
    if selected is None:
        return [], None
    begin = int(selected["begin_monotonic_ns"])
    end = int(selected["end_monotonic_ns"])
    samples = [
        row for row in timing.get("nvml", {}).get("samples", [])
        if begin <= int(row.get("monotonic_ns", -1)) <= end
    ]
    return samples, int(selected["step"])


def build_overlap_analysis(nvml_samples: list[dict[str, Any]],
                            gpu_count: int) -> dict[str, Any]:
    """Compute TRUE simultaneous dual-GPU activity from NVML samples.

    both_busy_us  : time BOTH GPUs report non-zero utilization
    one_busy_us   : time exactly one GPU is busy / the other idle
    neither_busy_us: time both GPUs are idle by NVML
    """
    both_busy_us = 0
    one_busy_us = 0
    neither_us = 0
    samples_total = 0
    samples_with_both_busy = 0
    samples_both_above_50 = 0
    samples_both_above_25 = 0
    nvml_both_active_samples = 0
    if len(nvml_samples) < 2 or gpu_count < 2:
        return {
            "schema_version": 2,
            "note": "insufficient samples or fewer than 2 GPUs",
            "samples_total": samples_total,
            "both_busy_us": both_busy_us,
            "one_busy_us": one_busy_us,
            "neither_busy_us": neither_us,
            "samples_with_both_busy": samples_with_both_busy,
            "nvml_both_gpus_active_samples": nvml_both_active_samples,
            "samples_both_above_50_percent": samples_both_above_50,
            "samples_both_above_25_percent": samples_both_above_25,
            "intersample_interval_us_estimated": 0,
        }
    first_ns = int(nvml_samples[0]["monotonic_ns"])
    last_ns = int(nvml_samples[-1]["monotonic_ns"])
    total_elapsed_ns = max(0, last_ns - first_ns)
    avg_interval_us = (
        (total_elapsed_ns // 1_000) // (len(nvml_samples) - 1)
        if len(nvml_samples) > 1 else 50_000
    )
    for sample in nvml_samples:
        samples_total += 1
        gpus = sample.get("gpus", [])
        if len(gpus) < 2:
            continue
        u0 = int(gpus[0].get("gpu_utilization_percent", 0) or 0)
        u1 = int(gpus[1].get("gpu_utilization_percent", 0) or 0)
        if u0 > 0 and u1 > 0:
            both_busy_us += int(avg_interval_us)
            samples_with_both_busy += 1
            nvml_both_active_samples += 1
            if u0 >= 50 and u1 >= 50:
                samples_both_above_50 += 1
            if u0 >= 25 and u1 >= 25:
                samples_both_above_25 += 1
        elif u0 > 0 or u1 > 0:
            one_busy_us += int(avg_interval_us)
        else:
            neither_us += int(avg_interval_us)
    total_us = both_busy_us + one_busy_us + neither_us
    active_us = both_busy_us + one_busy_us
    return {
        "schema_version": 2,
        "samples_total": samples_total,
        "both_busy_us": both_busy_us,
        "one_busy_us": one_busy_us,
        "neither_busy_us": neither_us,
        "intersample_interval_us_estimated": int(avg_interval_us),
        "samples_with_both_busy": samples_with_both_busy,
        "nvml_both_gpus_active_samples": nvml_both_active_samples,
        "samples_both_above_50_percent": samples_both_above_50,
        "samples_both_above_25_percent": samples_both_above_25,
        # "active" excludes idle time, mirrors the M2.5 semantic that only
        # counts covered at least one working GPU.
        "both_busy_fraction_of_active": (
            both_busy_us / active_us if active_us else 0.0
        ),
        "both_busy_fraction_of_total": (
            both_busy_us / total_us if total_us else 0.0
        ),
    }


def build_multi_gpu_timeline(timing: dict[str, Any],
                              expert_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a per-GPU per-step summary from engine_timelines already emitted
    by the M2.5 forensic recorder.  Each layer's timeline is in absolute
    ms relative to model_wall_ms; we project them onto wall-clock model time
    so the analyzer can directly plot both-GPU activity per-step.

    Filters out current_token<0 rollup entries so end-of-gen snapshots
    (where StageProfile.current_token_ == last step) don't pollute the
    per-step view.
    """
    step_rollup: dict[tuple[int, int], dict[str, Any]] = collections.defaultdict(
        lambda: {
            "h2d_event_count": 0,
            "compute_event_count": 0,
            "h2d_active_ms_total": 0.0,
            "compute_active_ms_total": 0.0,
            "weight_conversion_ms_total": 0.0,
            "layers_touched": set(),
        }
    )
    for engine_item, timeline_item in zip(
            timing.get("engine_profiles", []),
            timing.get("engine_timelines", [])):
        layer = int(engine_item["layer"])
        gpu = int(engine_item["gpu"])
        events = timeline_item["timeline"].get("traceEvents", [])
        if not events:
            continue
        token_value = int(engine_item["profile"].get("current_token",
                                                     engine_item["profile"].get("token", -1)))
        if token_value < 0:
            continue
        h2d_count = sum(1 for event in events if event.get("name") == "h2d")
        h2d_ms = sum(event["dur"] for event in events
                     if event.get("name") == "h2d") / 1000.0
        compute_keys = {"gate_projection", "up_projection", "silu_multiply",
                        "down_projection"}
        compute_count = sum(1 for event in events if event.get("name") in compute_keys)
        compute_ms = sum(event["dur"] for event in events
                         if event.get("name") in compute_keys) / 1000.0
        conv_ms = sum(event["dur"] for event in events
                      if event.get("name") == "weight_conversion") / 1000.0
        step_rollup[(token_value, gpu)]["h2d_event_count"] += h2d_count
        step_rollup[(token_value, gpu)]["compute_event_count"] += compute_count
        step_rollup[(token_value, gpu)]["h2d_active_ms_total"] += h2d_ms
        step_rollup[(token_value, gpu)]["compute_active_ms_total"] += compute_ms
        step_rollup[(token_value, gpu)]["weight_conversion_ms_total"] += conv_ms
        step_rollup[(token_value, gpu)]["layers_touched"].add(layer)
    out = []
    for (step, gpu), row in sorted(step_rollup.items()):
        out.append({
            "step": step,
            "gpu": gpu,
            "h2d_event_count": row["h2d_event_count"],
            "compute_event_count": row["compute_event_count"],
            "h2d_active_ms_total": round(row["h2d_active_ms_total"], 6),
            "compute_active_ms_total": round(row["compute_active_ms_total"], 6),
            "weight_conversion_ms_total": round(row["weight_conversion_ms_total"], 6),
            "layers_touched": sorted(row["layers_touched"]),
        })
    transfer_only_drops: list[dict[str, Any]] = []
    transfer_events = [event for event in expert_events
                       if event.get("event_type") == "expert_transfer"]
    if transfer_events:
        by_layer = collections.defaultdict(list)
        for event_ in transfer_events:
            by_layer[int(event_["logical_layer"])].append(event_)
        for layer, rows in by_layer.items():
            drops = sum(1 for row in rows
                        if not row.get("overlapped_compute", False))
            total = len(rows)
            if total:
                transfer_only_drops.append({
                    "logical_layer": layer,
                    "transfer_drop_count": drops,
                    "transfer_total": total,
                    "fraction_drop": (drops / total) if total else 0.0,
                })
    return {
        "schema_version": 2,
        "per_step_per_gpu": out,
        "transfer_only_layer_summary": transfer_only_drops,
    }


def build_expert_events(run_id: str, timing: dict[str, Any],
                        index: dict[str, Any]) -> list[dict[str, Any]]:
    route_rows = timing.get("route_selections", [])
    routes: dict[tuple[int, int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in route_rows:
        routes[(int(row["step"]), int(row["layer"]), int(row["expert"]))].append(row)

    requests: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    evictions: list[dict[str, Any]] = []
    loaded_gpus: dict[tuple[int, int], set[str]] = collections.defaultdict(set)
    request_event_links: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for engine_item, timeline_item in zip(
            timing.get("engine_profiles", []), timing.get("engine_timelines", [])):
        layer = int(engine_item["layer"])
        gpu = f"cuda:{int(engine_item['gpu'])}"
        trace = engine_item["profile"].get("trace", [])
        timeline = timeline_item["timeline"].get("traceEvents", [])
        by_transfer: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        by_expert: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
        all_compute = [event for event in timeline if event.get("name") in {
            "gate_projection", "up_projection", "silu_multiply", "down_projection"
        }]
        for event in timeline:
            args = event.get("args", {})
            # Transfer IDs are Engine-local and start at zero. Do not treat
            # the first valid transfer as a false/no-ID sentinel.
            if (
                event.get("name") in {"h2d", "weight_conversion"}
                and args.get("transfer_id") is not None
            ):
                transfer_id = int(args["transfer_id"])
                by_transfer[transfer_id].append(event)
            token = int(args.get("token", -1))
            expert = int(args.get("expert", -1))
            if token >= 0 and expert >= 0:
                by_expert[(token, expert)].append(event)
        for raw in trace:
            step = int(raw.get("token", -1))
            expert = int(raw["expert"])
            transfer_id = int(raw.get("transfer_id", 0) or 0)
            occurrence = routes.get((step, layer, expert), [])
            related_transfer = by_transfer.get(transfer_id, [])
            related_compute = by_expert.get((step, expert), [])
            h2d = [event for event in related_transfer if event.get("name") == "h2d"]
            conversion = [event for event in related_transfer
                          if event.get("name") == "weight_conversion"]
            compute = [event for event in related_compute if event.get("name") in {
                "gate_projection", "up_projection", "silu_multiply", "down_projection"
            }]
            request = {
                "event_type": "expert_request",
                "run_id": run_id,
                "event_index": int(raw["index"]),
                "request_time_ms": float(raw.get("request_time_ms", 0.0)),
                "token_step": step,
                "token_phase": "prefill" if step == 0 else "decode",
                "logical_layer": layer,
                "resolved_shard_layer": int(raw["resolved_shard_layer"]),
                "expert_id": expert,
                "routing_occurrences": [{
                    "sequence_token": int(item["sequence_token"]),
                    "routing_rank": int(item["routing_rank"]),
                    "routing_weight": float(item["routing_weight"]),
                } for item in occurrence],
                "routing_rank": min((int(item["routing_rank"]) for item in occurrence),
                                    default=None),
                "routing_weight": max((float(item["routing_weight"]) for item in occurrence),
                                      default=None),
                "source_checkpoint_shards": _source_shards(index, layer, expert),
                "source_checkpoint_shard": _source_shards(index, layer, expert)[0],
                "expert_bytes": int(raw.get("destination_bytes", 0)),
                "source_bytes": int(raw.get("source_bytes", 0)),
                "cache_state_before": {
                    "bytes": int(raw.get("cache_bytes_before", 0)),
                    "resident_entries": int(raw.get("cache_entries_before", 0)),
                },
                "cache_state_after": {
                    "bytes": int(raw.get(
                        "cache_bytes_after", raw.get("cache_bytes_used", 0)
                    )),
                    "resident_entries": int(raw.get(
                        "cache_entries_after", raw.get("cache_entries_before", 0)
                    )),
                },
                "cache_result": "miss" if raw.get("kind") == "cold" else "hit",
                "cache_kind": raw.get("kind"),
                "gpu_destination": gpu,
                "transfer_id": transfer_id,
                "transfer_reason": (
                    "demand_miss" if raw.get("transfer_launched", False)
                    else "resident_or_inflight_reuse"
                ),
                "transfer_launched": bool(raw.get("transfer_launched", False)),
                "residency_generation": (
                    int(raw["generation"]) if raw.get("generation") is not None else None
                ),
                "pin_count_after": (
                    int(raw["pin_count"]) if raw.get("pin_count") is not None else None
                ),
                "transfer_consumed": (
                    bool(raw["consumed"]) if raw.get("consumed") is not None else None
                ),
                "evicted_before_use": (
                    bool(raw["evicted_before_use"])
                    if raw.get("evicted_before_use") is not None else None
                ),
                "source_pinned": bool(raw.get("source_pinned", False)),
                "evicted_layer": int(raw.get("evicted_layer", -1)),
                "evicted_expert_id": int(raw.get("evicted_expert", -1)),
                "evicted_generation": int(raw.get("evicted_generation", 0)),
                "eviction_reason": raw.get("eviction_reason"),
                "prior_reuse_distance": int(raw.get("reuse_distance", -1)),
                "prior_distinct_reuse_distance": int(raw.get("distinct_reuse_distance", -1)),
                "theoretical_min_cache_bytes": int(
                    raw.get("theoretical_min_cache_bytes", 0)
                ),
                "load_start_ms": min((event["ts"] / 1000.0 for event in h2d),
                                     default=None),
                "load_completion_ms": max((
                    (event["ts"] + event["dur"]) / 1000.0
                    for event in (conversion or h2d)
                ), default=None),
                "compute_start_ms": min((event["ts"] / 1000.0 for event in compute),
                                        default=None),
                "compute_completion_ms": max((
                    (event["ts"] + event["dur"]) / 1000.0 for event in compute
                ), default=None),
            }
            requests.append(request)
            request_event_links.append((request, raw))
            if raw.get("kind") == "cold":
                loaded_gpus[(layer, expert)].add(gpu)
                for event in h2d:
                    args = event.get("args", {})
                    overlap_us = sum(max(
                        0.0,
                        min(event["ts"] + event["dur"], candidate["ts"] + candidate["dur"]) -
                        max(event["ts"], candidate["ts"]),
                    ) for candidate in all_compute)
                    overlap_us = min(float(event["dur"]), overlap_us)
                    transfers.append({
                        "event_type": "expert_transfer",
                        "run_id": run_id,
                        "transfer_id": transfer_id,
                        "token_step": step,
                        "token_phase": "prefill" if step == 0 else "decode",
                        "logical_layer": layer,
                        "resolved_shard_layer": int(raw["resolved_shard_layer"]),
                        "expert_id": expert,
                        "gpu_destination": gpu,
                        "direction": "h2d",
                        "component": "expert_weight",
                        "bytes": int(args.get("bytes", 0)),
                        "transfer_start_ms": event["ts"] / 1000.0,
                        "transfer_completion_ms": (
                            event["ts"] + event["dur"]
                        ) / 1000.0,
                        "overlapped_compute": overlap_us > 0,
                        "overlap_with_compute_ms": overlap_us / 1000.0,
                        "serialized_transfer_ms": (
                            float(event["dur"]) - overlap_us
                        ) / 1000.0,
                    })
            if int(raw.get("evicted_expert", -1)) >= 0:
                evictions.append({
                    "event_type": "expert_eviction", "run_id": run_id,
                    "token_step": step, "logical_layer": layer,
                    "resolved_shard_layer": int(raw.get("evicted_layer", layer)),
                    "expert_id": int(raw["evicted_expert"]),
                    "gpu_destination": gpu,
                    "eviction_reason": raw.get("eviction_reason") or "capacity_lru",
                    "triggering_expert_id": expert,
                    "request_time_ms": float(raw.get("request_time_ms", 0.0)),
                })

        for event in timeline:
            name = event.get("name")
            if name not in {"activation_h2d", "d2h"}:
                continue
            event_args = event.get("args", {})
            token_step = int(event_args.get("token", -1))
            transfers.append({
                "event_type": "tensor_transfer",
                "run_id": run_id,
                "token_step": token_step,
                "token_phase": "prefill" if token_step == 0 else "decode",
                "logical_layer": layer,
                "resolved_shard_layer": layer,
                "expert_id": int(event_args.get("expert", -1)),
                "gpu_destination": gpu,
                "direction": "h2d" if name == "activation_h2d" else "d2h",
                "component": (
                    "native_expert_activation" if name == "activation_h2d"
                    else "native_expert_output"
                ),
                "bytes": int(event_args.get("bytes", 0)),
                "transfer_start_ms": event["ts"] / 1000.0,
                "transfer_completion_ms": (event["ts"] + event["dur"]) / 1000.0,
            })

    for span in timing.get("wall_spans", []):
        metadata = span.get("metadata", {})
        direction = metadata.get("direction")
        byte_count = int(metadata.get("bytes", 0) or 0)
        if direction not in {"h2d", "d2h", "d2d"} or byte_count <= 0:
            continue
        transfers.append({
            "event_type": "tensor_transfer",
            "run_id": run_id,
            "token_step": int(span.get("step", -1)),
            "token_phase": span.get("phase"),
            "logical_layer": int(span.get("layer", -1)),
            "resolved_shard_layer": int(span.get("layer", -1)),
            "expert_id": None,
            "gpu_destination": metadata.get(
                "destination", f"cuda:{int(span.get('gpu', 0))}"
            ),
            "direction": direction,
            "component": metadata.get("component", span.get("name")),
            "bytes": byte_count,
            "transfer_duration_ms": float(span.get("cpu_wall_ms", 0.0)),
            "transfer_start_monotonic_ns": span.get("start_monotonic_ns"),
            "transfer_completion_monotonic_ns": span.get("end_monotonic_ns"),
        })

    for row in build_layer_timing(timing).get("rows", []):
        byte_count = int(row.get("pageable_to_pinned_bytes", 0))
        if byte_count <= 0:
            continue
        transfers.append({
            "event_type": "tensor_transfer",
            "run_id": run_id,
            "token_step": int(row["step"]),
            "token_phase": row["phase"],
            "logical_layer": int(row["layer"]),
            "resolved_shard_layer": int(row["layer"]),
            "expert_id": None,
            "gpu_destination": f"cuda:{int(row['gpu'])}",
            "direction": "pageable_to_pinned",
            "component": "expert_weight_staging",
            "bytes": byte_count,
            "transfer_duration_ms": float(row.get("pageable_to_pinned_copy_ms", 0.0)),
        })

    request_groups: dict[tuple[int, int, int, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for request in requests:
        request_groups[(request["token_step"], request["logical_layer"],
                        request["expert_id"], request["gpu_destination"])].append(request)
    previous_time: dict[tuple[int, int], float] = {}
    for request in requests:
        key = (request["logical_layer"], request["expert_id"])
        when = request["request_time_ms"]
        request["time_since_previous_use_ms"] = (
            when - previous_time[key] if key in previous_time else None
        )
        previous_time[key] = when
        request["duplicated_on_both_gpus"] = len(loaded_gpus[key]) > 1
        request["same_expert_transferred_more_than_once_this_token"] = (
            sum(1 for item in request_groups[
                (request["token_step"], request["logical_layer"],
                 request["expert_id"], request["gpu_destination"])
            ] if item["cache_result"] == "miss") > 1
        )
    for row in route_rows:
        row["event_type"] = "route_selection"
        row["run_id"] = run_id
    return [*route_rows, *requests, *transfers, *evictions]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    memory = MemoryProbe(max_checkpoints=256)
    memory.checkpoint("process_start", include_maps=True, include_smaps=True,
                      include_cuda=False, include_nvml=False)

    dependency_begin = time.perf_counter()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer
    from scripts.milestone25_timing import ForensicTimingRecorder, summarize_nvml
    from scripts.ornith_support import build_complete_tensor_map, read_checkpoint_index
    from scripts.run_ornith_generation import (
        compare_trace, engine_stats, fresh_engine_path_proof,
        load_runtime, run_generation, serializable_generation,
    )
    dependency_seconds = time.perf_counter() - dependency_begin

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("Milestone 2.5 requires at least one CUDA GPU")
    if args.require_dual_gpu and gpu_count != 2:
        raise RuntimeError(f"expected exactly two visible GPUs, found {gpu_count}")
    effective_gpus = min(gpu_count, 2)

    recorder = ForensicTimingRecorder(
        torch, effective_gpus, args.split_layer, sample_nvml=True,
        memory_probe=memory,
        profile_decode_step=args.torch_profiler_step,
        profiler_trace_path=args.output_dir / "torch-profiler-trace.json.gz",
    )
    recorder.record_phase("dependency_import", dependency_seconds)
    process_create_seconds = None
    try:
        import psutil

        process_create_seconds = max(0.0, time.time() - psutil.Process().create_time())
    except Exception:
        pass
    recorder.record_phase("python_startup_to_dependency_ready",
                          process_create_seconds or 0.0,
                          {"clock": "psutil.Process.create_time"})
    memory.checkpoint("after_imports", include_maps=True, include_smaps=False,
                      include_cuda=False, include_nvml=True)

    cuda_begin = time.perf_counter()
    torch.cuda.init()
    for device in range(effective_gpus):
        with torch.cuda.device(device):
            torch.empty(1, device=f"cuda:{device}")
        torch.cuda.synchronize(device)
    recorder.record_phase("cuda_context_initialization", time.perf_counter() - cuda_begin)
    memory.checkpoint("after_cuda_initialization", include_maps=True,
                      include_cuda=True, include_nvml=True)

    index = read_checkpoint_index(args.model_dir)
    tensor_summary = None
    if args.tensor_map:
        with recorder.phase("complete_tensor_metadata_inventory"):
            tensor_rows, tensor_summary = build_complete_tensor_map(
                args.model_dir, effective_gpus, args.split_layer
            )
            write_json(args.output_dir / "tensor-map.json", {
                "summary": tensor_summary, "tensors": tensor_rows,
            })

    with recorder.phase("tokenizer_loading"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    memory.checkpoint("after_tokenizer_loading", include_cuda=True, include_nvml=True)

    bridge = PhaseBridge(recorder, memory)
    runtime = load_runtime(
        args.model_dir, effective_gpus, args.split_layer, args.cache_experts,
        profile_stages=args.profile,
        trace_requests=args.trace_requests,
        profile_timeline=args.profile_timeline,
        debug_validate_cache=args.debug_validate_cache,
        allow_diagnostic_sub_topk_cache=args.allow_sub_topk_cache,
        router_backend=args.router_backend,
        phase_recorder=bridge,
    )
    runtime["context"].cache_disabled = args.cache_disabled
    recorder.set_engines(runtime["engines"])
    warmup_generation = None
    warmup_engine_profiles = []
    warmup_engine_timelines = []
    if args.warmup_generation:
        # M4 transfer accounting needs the complete residency history that
        # seeds the measured warm-cache run. Start a separate profiler epoch,
        # capture it below, then retain the existing reset before measurement.
        if args.profile and args.trace_requests:
            recorder.reset_engine_profiles()
        with recorder.phase("warmup_generation"):
            warmup_generation = run_generation(
                runtime, tokenizer, args.prompt, args.max_new_tokens,
                False, "dee", trace=False,
            )
        if args.profile and args.trace_requests:
            warmup_wall_ms = warmup_generation["total_generation_seconds"] * 1000.0
            for layer, engine in enumerate(runtime["engines"]):
                warmup_engine_profiles.append({
                    "layer": layer,
                    "gpu": recorder.device_for_layer(layer),
                    "profile": json.loads(engine.external_profile_json(warmup_wall_ms)),
                })
                warmup_engine_timelines.append({
                    "layer": layer,
                    "gpu": recorder.device_for_layer(layer),
                    "timeline": json.loads(engine.external_timeline_json(warmup_wall_ms)),
                })
        memory.checkpoint("after_warmup_generation", include_maps=True,
                          include_smaps=True, include_cuda=True, include_nvml=True)
    if args.profile:
        recorder.attach_model_hooks(runtime["model"], runtime["context"])
        recorder.reset_engine_profiles()
        runtime["context"].forensics = recorder

    tensor_owners = {
        "transformers_model": runtime["model"],
        "reference_expert_caches": runtime["reference_expert_caches"],
    }
    memory.checkpoint(
        "before_prefill", tensor_owners=tensor_owners,
        tensor_owner_metadata={
            "transformers_model": {"owner": "Transformers", "purpose": "dense model"},
            "reference_expert_caches": {
                "owner": "Transformers", "purpose": "lazy reference expert cache"
            },
        }, include_maps=True, include_smaps=True, include_cuda=True,
        include_cuda_snapshot=True, include_nvml=True,
    )
    for device in range(effective_gpus):
        torch.cuda.reset_peak_memory_stats(device)

    # Milestone 3: reset the M3 path-proof counters so the optional warmup
    # generation above does NOT pollute the profiled run's evidence.  The
    # warmup runs HybridExperts.forward over the same context/engine, so
    # device_path_calls / host_path_fallback_calls / fp32_to_fp16_conversion_ms
    # would otherwise include the warmup's counts in addition to the
    # profiled run's.  This reset is critical for the warm vs cold control.
    runtime["context"].engine_path_proof = fresh_engine_path_proof()

    generation = run_generation(
        runtime, tokenizer, args.prompt, args.max_new_tokens,
        False, "dee", trace=False,
    )
    if args.profile:
        recorder.capture_final_engine_evidence(
            generation["total_generation_seconds"] * 1000.0
        )
    runtime["context"].forensics = None
    timing = recorder.stop()
    primary_engine_stats = engine_stats(runtime)
    cache_invariant_rows = []
    for layer, engine in enumerate(runtime["engines"]):
        if not hasattr(engine, "validate_cache_invariants"):
            cache_invariant_rows.append({
                "layer": layer,
                "available": False,
                "valid": None,
                "error": "binding unavailable",
            })
            continue
        valid, error = engine.validate_cache_invariants()
        cache_invariant_rows.append({
            "layer": layer,
            "available": True,
            "valid": bool(valid),
            "error": error or None,
        })
    cache_invariants = {
        "schema_version": 1,
        "debug_validation_enabled": args.debug_validate_cache,
        "all_layers_valid": all(
            row.get("valid") is True for row in cache_invariant_rows
        ),
        "layers": cache_invariant_rows,
    }
    write_json(args.output_dir / "cache-invariants.json", cache_invariants)
    if args.debug_validate_cache and not cache_invariants["all_layers_valid"]:
        raise RuntimeError("debug cache invariant validation failed")
    primary_tensor_inventory = inventory_tensors(
        {"transformers_model": runtime["model"]},
        owner_metadata={"transformers_model": {
            "owner": "Transformers", "purpose": "dense model parameters and buffers",
            "lifetime": "runtime", "copy_status": "checkpoint tensors copied to CUDA",
        }},
        max_tensor_details=256,
    )
    memory.checkpoint(
        "after_primary_generation", tensor_owners=tensor_owners,
        tensor_owner_metadata={
            "transformers_model": {"owner": "Transformers", "purpose": "dense model"},
            "reference_expert_caches": {
                "owner": "Transformers", "purpose": "lazy reference expert cache"
            },
        }, include_maps=True, include_smaps=True, include_cuda=True,
        include_cuda_snapshot=True, include_nvml=True,
    )

    expected = BASELINE_TOKENS.get(args.prompt)
    expected_prefix = expected[:args.max_new_tokens] if expected is not None else None
    exact_baseline = (
        generation["generated_token_ids"] == expected_prefix
        if expected_prefix is not None else None
    )
    routed_layers = generation["executed_router_layers"]
    all_40_layers = routed_layers == list(range(40))
    if exact_baseline is False:
        raise RuntimeError(
            f"instrumented tokens changed: got {generation['generated_token_ids']} "
            f"expected {expected_prefix}"
        )
    if not all_40_layers:
        raise RuntimeError(f"instrumented run did not trace all 40 layers: {routed_layers}")

    warmup_tokens_exact = (
        warmup_generation["generated_token_ids"] == generation["generated_token_ids"]
        if warmup_generation is not None else None
    )
    if warmup_tokens_exact is False:
        raise RuntimeError("profiled generation changed tokens relative to its warmup control")

    parity = None
    reference_tensor_inventory = None
    if args.reference_parity:
        memory.checkpoint("before_reference_validation", include_maps=True,
                          include_smaps=True, include_cuda=True, include_nvml=True)
        reference = run_generation(
            runtime, tokenizer, args.prompt, args.max_new_tokens,
            False, "reference", trace=True,
        )
        reference_tensor_inventory = inventory_tensors(
            {"reference_expert_caches": runtime["reference_expert_caches"]},
            owner_metadata={"reference_expert_caches": {
                "owner": "Transformers", "purpose": "reference expert weights",
                "lifetime": "reference validation", "copy_status": "checkpoint to CUDA copy",
            }},
            max_tensor_details=256,
        )
        memory.checkpoint(
            "reference_peak_before_cache_clear",
            tensor_owners={"reference_expert_caches": runtime["reference_expert_caches"]},
            tensor_owner_metadata={"reference_expert_caches": {
                "owner": "Transformers", "purpose": "live reference expert weights"
            }},
            include_maps=True, include_smaps=True, include_cuda=True,
            include_cuda_snapshot=True, include_nvml=True,
        )
        for expert in runtime["reference_experts"]:
            expert.clear_cache()
        torch.cuda.empty_cache()
        candidate = run_generation(
            runtime, tokenizer, args.prompt, args.max_new_tokens,
            False, "dee", trace=True,
        )
        comparisons = compare_trace(reference["collector"], candidate["collector"])
        parity = {
            "generated_token_ids_exact": (
                reference["generated_token_ids"] == candidate["generated_token_ids"]
            ),
            "decoded_text_exact": reference["generated_text"] == candidate["generated_text"],
            "all_trace_categories_passed": comparisons["all_categories_passed"],
            "comparisons": comparisons,
            "reference": serializable_generation(reference),
            "candidate": serializable_generation(candidate),
        }
        if not all((parity["generated_token_ids_exact"], parity["decoded_text_exact"],
                    parity["all_trace_categories_passed"])):
            raise RuntimeError("forensic instrumentation changed reference parity")
        memory.checkpoint(
            "after_reference_validation", tensor_owners=tensor_owners,
            include_maps=True, include_smaps=True, include_cuda=True,
            include_cuda_snapshot=True, include_nvml=True,
        )

    layer_timing = build_layer_timing(timing) if args.profile else {
        "schema_version": 1, "rows": [], "reason": "profiling disabled control",
    }
    expert_events = build_expert_events(args.run_id, timing, index) if args.profile else []
    with (args.output_dir / "expert-trace.jsonl").open("w", encoding="utf-8") as stream:
        for event in expert_events:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    warmup_expert_events = []
    if warmup_engine_profiles:
        warmup_expert_events = build_expert_events(
            f"{args.run_id}-warmup",
            {
                "engine_profiles": warmup_engine_profiles,
                "engine_timelines": warmup_engine_timelines,
                "route_selections": [],
                "wall_spans": [],
                "profile_snapshots": [],
            },
            index,
        )
    with (args.output_dir / "warmup-expert-trace.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for event in warmup_expert_events:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    # Milestone 3: flush the path-proof counters accumulated by
    # run_ornith_generation.HybridExperts.forward.  device_path_calls /
    # host_path_fallback_calls prove that the device-resident MoE bound
    # method actually ran (and how often it had to fall back).  The
    # fp32_to_fp16_conversion_ms_total measures the residual .to() cost
    # the device path still incurs because the C++ tensor contract is
    # FP32 even when the surrounding model is FP16.
    path_proof = dict(runtime["context"].engine_path_proof)
    path_proof["schema_version"] = 2
    path_proof["run_id"] = args.run_id
    path_proof["git_commit"] = git_revision()
    path_proof["router_backend"] = args.router_backend
    # Aggregates that let the analyzer prove the device path was the
    # dominant execution route.  device_share=1.0 means every hybrid
    # forward routed through moe_forward_batch_device; 0.0 means every
    # call fell back to the host path silently (a regression).
    total_calls = (path_proof["device_path_calls"] +
                   path_proof["host_path_fallback_calls"])
    if total_calls > 0:
        path_proof["device_path_share"] = (
            path_proof["device_path_calls"] / total_calls
        )
        path_proof["host_fallback_share"] = (
            path_proof["host_path_fallback_calls"] / total_calls
        )
    else:
        path_proof["device_path_share"] = 0.0
        path_proof["host_fallback_share"] = 0.0
    write_json(args.output_dir / "path-proof.json", path_proof)

    memory.write_json(args.output_dir / "memory-timeline.json")
    write_json(args.output_dir / "layer-timing.json", layer_timing)
    write_json(args.output_dir / "timing-raw.json", timing)
    write_json(args.output_dir / "gpu-utilization-summary.json", {
        "per_gpu": summarize_nvml(timing["nvml"].get("samples", []), effective_gpus),
        "sampler_error": timing["nvml"].get("error"),
    })

    # Milestone 3: derived artifacts consumed by analyze_milestone3_matrix.
    # Synchronization, true GPU overlap (both-active samples), and per-GPU
    # event timelines are all derivable from timing["nvml"] and
    # timing["engine_timelines"], already produced by the M2.5 forensic
    # recorder.  No additional per-call instrumentation is needed; these
    # JSONs are pure re-aggregation.
    nvml_samples, overlap_step = representative_decode_nvml_samples(timing)
    overlap_analysis = build_overlap_analysis(nvml_samples, effective_gpus)
    overlap_analysis["representative_step"] = overlap_step
    overlap_analysis["sample_scope"] = "representative_decode"
    write_json(args.output_dir / "synchronization-analysis.json",
               build_synchronization_analysis(timing))
    write_json(args.output_dir / "overlap-analysis.json",
               overlap_analysis)
    write_json(args.output_dir / "multi-gpu-timeline.json",
               build_multi_gpu_timeline(timing, expert_events))

    machine = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_inventory(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_allocator_configuration": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "framework default"
        ),
        "gpu_count": gpu_count,
        "gpus": nvidia_smi_inventory(),
        "pid": os.getpid(),
    }
    report = {
        "schema_version": 1,
        "result": "PASS",
        "run_id": args.run_id,
        "git_commit": git_revision(),
        "checkpoint": {
            "path": str(args.model_dir),
            "revision": "kaggle-dataset:nivind/ornith-1-0-35b",
            "tensor_summary": tensor_summary,
        },
        "machine": machine,
        "configuration": {
            "prompt": args.prompt,
            "prompt_token_ids": generation["prompt_token_ids"],
            "max_new_tokens": args.max_new_tokens,
            "execution_mode": "hybrid_transformers_dense_deecpp_experts",
            "router_backend": args.router_backend,
            "classification": args.classification,
            "split_layer": args.split_layer,
            "cache_experts_per_layer": args.cache_experts,
            "cache_disabled": args.cache_disabled,
            "profile_enabled": args.profile,
            "trace_requests": args.trace_requests,
            "profile_timeline": args.profile_timeline,
            "debug_validate_cache": args.debug_validate_cache,
            "reference_parity": args.reference_parity,
            "warmup_generation": args.warmup_generation,
        },
        "warmup_generation": (
            serializable_generation(warmup_generation)
            if warmup_generation is not None else None
        ),
        "generation": serializable_generation(generation),
        "correctness": {
            "milestone2_baseline_tokens": expected_prefix,
            "baseline_tokens_exact": exact_baseline,
            "all_40_layers_executed": all_40_layers,
            "routed_layers": routed_layers,
            "reference_parity": parity,
            "warmup_tokens_exact": warmup_tokens_exact,
        },
        "layout": {
            "dense_loaded_bytes": runtime["dense_loaded_bytes"],
            "expert_cache_budget_per_layer": runtime["expert_cache_budget_per_layer"],
            "parameter_inventory": parameter_inventory(runtime["model"]),
            "primary_tensor_inventory": primary_tensor_inventory,
            "reference_tensor_inventory": reference_tensor_inventory,
            "native_engine_memory": primary_engine_stats,
            "partition": "layers 0..split-1 on cuda:0; remaining layers on cuda:1",
        },
        "phase_timings": timing["phase_timings"],
        "instrumentation": {
            "bookkeeping_ms": timing["instrumentation_bookkeeping_ms"],
            "torch_profiler": timing["torch_profiler"],
            "expert_event_count": len(expert_events),
            "warmup_expert_event_count": len(warmup_expert_events),
        },
        "artifacts": {
            "memory_timeline": "memory-timeline.json",
            "layer_timing": "layer-timing.json",
            "expert_trace": "expert-trace.jsonl",
            "warmup_expert_trace": "warmup-expert-trace.jsonl",
            "cache_invariants": "cache-invariants.json",
            "timing_raw": "timing-raw.json",
            "gpu_utilization_summary": "gpu-utilization-summary.json",
        },
    }
    write_json(args.output_dir / "run-report.json", report)

    cleanup_begin = time.perf_counter()
    runtime["pool"].close()
    del tensor_owners
    del runtime
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    for device in range(effective_gpus):
        torch.cuda.synchronize(device)
    cleanup_seconds = time.perf_counter() - cleanup_begin
    memory.checkpoint("after_cleanup", include_maps=True, include_smaps=True,
                      include_cuda=True, include_nvml=True,
                      metadata={"cleanup_wall_ms": cleanup_seconds * 1000.0})
    memory.write_json(args.output_dir / "memory-timeline.json")
    report["phase_timings"].append({
        "name": "cleanup", "wall_ms": cleanup_seconds * 1000.0,
        "metadata": {},
    })
    write_json(args.output_dir / "run-report.json", report)
    print("MILESTONE25_RUN=" + json.dumps({
        "run_id": args.run_id,
        "result": report["result"],
        "tokens": generation["generated_token_ids"],
        "all_40_layers": all_40_layers,
        "output_dir": str(args.output_dir),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
