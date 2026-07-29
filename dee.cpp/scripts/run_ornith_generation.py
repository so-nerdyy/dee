#!/usr/bin/env python3
"""Real Ornith text generation with official dense/cache code and dee.cpp MoE.

The checkpoint is never loaded through a full ``from_pretrained`` call. Dense
text tensors are materialized directly onto an explicit 20/20 GPU layer split;
routed experts stay mmap-backed and are streamed by one bounded dee.cpp engine
per layer. Official Transformers owns tokenization, attention/linear-attention,
residuals, normalization, shared experts, DynamicCache, and the LM head.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # Windows CLI parsing/tests; real CUDA proof runs on Linux.
    resource = None

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.ornith_support import (  # noqa: E402
    build_complete_tensor_map,
    layer_device,
    numeric_comparison,
    read_checkpoint_index,
    shard_paths_for_layer,
    validate_expert_cache_budget,
)


TRACE_LAYERS = set(range(40))
FUSED_NORM_EXECUTION_MODE = "native-combined-direct-fused-norm"
EXECUTION_MODES = (
    "production",
    "parity",
    "profiler",
    "debug-full-logit",
    "native-combined",
    "native-combined-direct",
    FUSED_NORM_EXECUTION_MODE,
)
TOLERANCES = {
    "embedding_output": (0.0, 0.0),
    "router_logits": (2.0e-2, 2.0e-2),
    "routing_weights": (5.0e-3, 2.0e-2),
    "selected_expert_outputs": (5.0e-2, 5.0e-2),
    "intermediate_hidden_states": (1.0e-1, 5.0e-2),
    "final_hidden_state": (1.0e-1, 5.0e-2),
    "lm_head_logits": (2.5e-1, 5.0e-2),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def record_elapsed(recorder, name: str, started: float, **metadata):
    if recorder is not None:
        recorder.record_phase(name, time.perf_counter() - started, metadata)


class ResourceMonitor:
    def __init__(self, torch_module, gpu_count: int, interval: float = 0.05):
        self.torch = torch_module
        self.gpu_count = gpu_count
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_host_rss = 0
        self.peak_vram = [0] * gpu_count
        self.nvml = None
        self.nvml_handles = []
        self.nvml_error = None
        self.thermal_samples: list[list[dict[str, Any]]] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(device)
                for device in range(gpu_count)
            ]
        except Exception as exc:
            self.nvml_error = f"{type(exc).__name__}: {exc}"

    def _sample(self):
        try:
            import psutil

            self.peak_host_rss = max(self.peak_host_rss, psutil.Process().memory_info().rss)
        except Exception:
            pass
        for device in range(self.gpu_count):
            try:
                free_bytes, total_bytes = self.torch.cuda.mem_get_info(device)
                self.peak_vram[device] = max(self.peak_vram[device], total_bytes - free_bytes)
            except Exception:
                pass
        if self.nvml is not None:
            sample = []
            for device, handle in enumerate(self.nvml_handles):
                row: dict[str, Any] = {"device": device}
                for key, query in (
                    (
                        "temperature_c",
                        lambda: self.nvml.nvmlDeviceGetTemperature(
                            handle, self.nvml.NVML_TEMPERATURE_GPU
                        ),
                    ),
                    (
                        "sm_clock_mhz",
                        lambda: self.nvml.nvmlDeviceGetClockInfo(
                            handle, self.nvml.NVML_CLOCK_SM
                        ),
                    ),
                    (
                        "memory_clock_mhz",
                        lambda: self.nvml.nvmlDeviceGetClockInfo(
                            handle, self.nvml.NVML_CLOCK_MEM
                        ),
                    ),
                    (
                        "power_mw",
                        lambda: self.nvml.nvmlDeviceGetPowerUsage(handle),
                    ),
                ):
                    try:
                        row[key] = int(query())
                    except Exception:
                        pass
                sample.append(row)
            self.thermal_samples.append(sample)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self.thread = threading.Thread(target=self._run, name="ornith-resource-monitor", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self._sample()
        thermal_by_device = {}
        for device in range(self.gpu_count):
            rows = [
                sample[device]
                for sample in self.thermal_samples
                if device < len(sample)
            ]
            thermal_by_device[f"cuda:{device}"] = {
                "sample_count": len(rows),
                "minimum_temperature_c": min(
                    (row["temperature_c"] for row in rows
                     if "temperature_c" in row),
                    default=None,
                ),
                "maximum_temperature_c": max(
                    (row["temperature_c"] for row in rows
                     if "temperature_c" in row),
                    default=None,
                ),
                "median_temperature_c": (
                    statistics.median([
                        row["temperature_c"] for row in rows
                        if "temperature_c" in row
                    ])
                    if any("temperature_c" in row for row in rows) else None
                ),
                "minimum_sm_clock_mhz": min(
                    (row["sm_clock_mhz"] for row in rows
                     if "sm_clock_mhz" in row),
                    default=None,
                ),
                "maximum_sm_clock_mhz": max(
                    (row["sm_clock_mhz"] for row in rows
                     if "sm_clock_mhz" in row),
                    default=None,
                ),
                "median_sm_clock_mhz": (
                    statistics.median([
                        row["sm_clock_mhz"] for row in rows
                        if "sm_clock_mhz" in row
                    ])
                    if any("sm_clock_mhz" in row for row in rows) else None
                ),
                "minimum_memory_clock_mhz": min(
                    (row["memory_clock_mhz"] for row in rows
                     if "memory_clock_mhz" in row),
                    default=None,
                ),
                "maximum_power_mw": max(
                    (row["power_mw"] for row in rows if "power_mw" in row),
                    default=None,
                ),
            }
        result = {
            "peak_host_rss_bytes": self.peak_host_rss,
            "peak_vram_bytes": {f"cuda:{i}": value for i, value in enumerate(self.peak_vram)},
            "thermal_clock": {
                "nvml_error": self.nvml_error,
                "sample_interval_seconds": self.interval,
                "by_device": thermal_by_device,
            },
        }
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml = None
        return result


class CheckpointPool:
    def __init__(self, model_dir: Path, phase_recorder=None):
        from safetensors import safe_open

        self.model_dir = model_dir
        self.safe_open = safe_open
        self.contexts: dict[str, Any] = {}
        self.handles: dict[str, Any] = {}
        self.phase_recorder = phase_recorder

    def handle(self, shard_name: str):
        if shard_name not in self.handles:
            started = time.perf_counter()
            context = self.safe_open(
                str(self.model_dir / shard_name), framework="pt", device="cpu"
            )
            self.contexts[shard_name] = context
            self.handles[shard_name] = context.__enter__()
            record_elapsed(self.phase_recorder, "checkpoint_shard_open", started,
                           shard=shard_name)
        return self.handles[shard_name]

    def tensor(self, shard_name: str, tensor_name: str):
        return self.handle(shard_name).get_tensor(tensor_name)

    def close(self):
        for name, context in list(self.contexts.items()):
            context.__exit__(None, None, None)
            self.handles.pop(name, None)
        self.contexts.clear()


class TraceCollector:
    def __init__(self):
        self.records: dict[str, list[tuple[str, np.ndarray]]] = collections.defaultdict(list)
        self.router_layers: set[int] = set()

    def add(self, category: str, label: str, tensor):
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().float().cpu().numpy()
        self.records[category].append((label, np.asarray(tensor).copy()))

    def router(self, layer: int, logits, weights, experts):
        call = sum(1 for label, _ in self.records["router_logits"]
                   if label.startswith(f"layer={layer},"))
        prefix = f"layer={layer},call={call}"
        self.router_layers.add(layer)
        self.add("router_logits", prefix, logits)
        self.add("routing_weights", prefix, weights)
        self.add("expert_ids", prefix, experts)


class ExecutionContext:
    def __init__(self):
        self.mode = "dee"
        self.execution_mode = "production"
        self.collector: TraceCollector | None = None
        self.router_tie_fallback_rows = 0
        self.forensics = None
        self.current_step = -1
        self.current_phase = "unclassified"
        self.cache_disabled = False
        self.executed_router_layers: set[int] = set()
        # Keep the accepted native-host router as the default for sealed
        # baseline reproduction.  The opt-in torch-device backend executes
        # the checkpoint's already-resident official router module directly
        # on the layer GPU, eliminating the measured hidden D2H, logits H2D,
        # duplicate softmax/top-k, and scalar synchronization.  dee.cpp still
        # owns every routed expert forward.
        self.router_backend = "native-host"
        # Milestone 3 audit counters.  These are the only M3-deliverable
        # counters that the existing M2.5 forensic schema cannot infer, and
        # they prove that the device-resident MoE forward path actually ran
        # (vs. silently falling back to the host path on every call).  They
        # accumulate over the lifetime of a single generation and are flushed
        # to path-proof.json by run_ornith_forensics.py after recorder.stop().
        self.engine_path_proof = fresh_engine_path_proof()


def fresh_engine_path_proof() -> dict:
    """Single source of truth for Milestone 3 path-proof counter layout.
    Used both by ExecutionContext.__init__ and by the explicit warmup/profiled
    reset in run_ornith_forensics.main(); keep these two sites in lock-step
    to avoid silent field-trap drift in the analyzer."""
    return {
        "device_path_calls": 0,
        "host_path_fallback_calls": 0,
        "fp32_to_fp16_conversion_ms_total": 0.0,
        "expert_ids_d2h_total_bytes": 0,
        "expert_native_device_calls_total_ms": 0.0,
        "expert_native_host_calls_total_ms": 0.0,
        "pybind_device_calls": 0,
        "pybind_host_fallback_calls": 0,
        "python_combine_calls": 0,
        "raw_output_allocations": 0,
        "native_combined_calls": 0,
        "native_combined_fallback_calls": 0,
        "native_combined_ids_d2h_total_bytes": 0,
        "native_combined_raw_trace_allocations": 0,
        "native_combined_stream_handoffs": 0,
        "native_combined_input_materializations": 0,
        "native_combined_input_materialization_bytes": 0,
        "last_native_error_combined_attempt": "",
        "native_direct_calls": 0,
        "native_direct_fallback_calls": 0,
        "fused_rms_norm_calls": 0,
        "fused_rms_norm_gated_calls": 0,
        "fused_norm_output_allocations": 0,
        "fused_norm_failures": 0,
        "router_native_host_calls": 0,
        "router_torch_device_calls": 0,
        "router_hidden_d2h_total_bytes": 0,
        "router_outputs_h2d_total_bytes": 0,
        "router_scalar_sync_calls": 0,
        # Milestone 3 forensic: precise native diagnostic captured from
        # pydee.Engine.last_error_message() at each device attempt and at
        # any host-fallback RuntimeError.  Empty or "<none>/<empty: ...>"
        # placeholders are valid; analyzer should treat absence as the
        # default state and presence as an actual native-rootcause signal.
        "last_native_error_device_attempt": "",
        "last_native_error_host_fallback_attempt": "",
    }


def forensic_span(context: ExecutionContext, name: str, layer: int,
                  metadata: dict[str, Any] | None = None):
    if context.forensics is None:
        return contextlib.nullcontext()
    return context.forensics.span(name, layer, context.current_step,
                                  context.current_phase, metadata)


def tensor_transfer_bytes(value, destination_device) -> int:
    """Count tensor bytes whose current device differs from destination."""
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return (
            int(value.numel()) * int(value.element_size())
            if getattr(value, "device", None) != destination_device else 0
        )
    if isinstance(value, dict):
        return sum(tensor_transfer_bytes(item, destination_device)
                   for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_transfer_bytes(item, destination_device) for item in value)
    return 0


class LazyReferenceExperts:
    """Official eager expert equation with layer-local, bounded checkpoint reads."""

    def __init__(self, torch_module, nn_module, functional, layer: int,
                 pool: CheckpointPool, index: dict[str, Any], context: ExecutionContext,
                 hidden: int, intermediate: int, num_experts: int, cache_capacity: int = 8):
        class Module(nn_module.Module):
            pass

        self.module = Module()
        self.torch = torch_module
        self.F = functional
        self.layer = layer
        self.pool = pool
        self.index = index
        self.context = context
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.cache_capacity = cache_capacity
        self.cache: collections.OrderedDict[int, tuple[Any, Any, Any]] = collections.OrderedDict()
        self.module.forward = self.forward
        self.module.clear_cache = self.clear_cache

    def _load(self, expert: int, device, dtype):
        cached = self.cache.get(expert)
        if cached is not None:
            self.cache.move_to_end(expert)
            return cached
        prefix = f"model.language_model.layers.{self.layer}.mlp.experts.{expert}"
        tensors = []
        for kind, shape in (
            ("gate_proj", (self.intermediate, self.hidden)),
            ("up_proj", (self.intermediate, self.hidden)),
            ("down_proj", (self.hidden, self.intermediate)),
        ):
            name = f"{prefix}.{kind}.weight"
            shard = self.index["weight_map"].get(name)
            if shard is None:
                raise KeyError(f"missing genuine expert tensor {name}")
            tensor = self.pool.tensor(shard, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
            tensors.append(tensor.to(device=device, dtype=dtype))
        self.cache[expert] = tuple(tensors)
        self.cache.move_to_end(expert)
        while len(self.cache) > self.cache_capacity:
            self.cache.popitem(last=False)
        return self.cache[expert]

    def clear_cache(self):
        self.cache.clear()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        torch = self.torch
        final = torch.zeros_like(hidden_states)
        raw = None
        if self.context.collector is not None and self.layer in TRACE_LAYERS:
            raw = torch.zeros(
                hidden_states.shape[0], top_k_index.shape[1], hidden_states.shape[1],
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            hits = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        for expert in hits:
            top_k_pos, token_idx = torch.where(expert_mask[expert])
            gate_weight, up_weight, down_weight = self._load(
                expert, hidden_states.device, hidden_states.dtype
            )
            current = hidden_states[token_idx]
            gate = self.F.linear(current, gate_weight)
            up = self.F.linear(current, up_weight)
            current = self.F.silu(gate) * up
            current = self.F.linear(current, down_weight)
            if raw is not None:
                raw[token_idx, top_k_pos] = current
            weighted = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, weighted.to(final.dtype))
        if raw is not None:
            self.context.collector.add(
                "selected_expert_outputs", f"layer={self.layer}", raw
            )
        return final


class HybridRouter:
    def __init__(self, nn_module, reference, engine, layer: int,
                 context: ExecutionContext, top_k: int):
        class Module(nn_module.Module):
            pass

        self.module = Module()
        self.module.reference = reference
        self.engine = engine
        self.layer = layer
        self.context = context
        self.top_k = top_k
        self.module.forward = self.forward

    def forward(self, hidden_states):
        torch = __import__("torch")
        self.context.executed_router_layers.add(self.layer)
        flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
        if (
            self.context.mode != "reference"
            and hasattr(self.engine, "set_external_token")
        ):
            self.engine.set_external_token(self.context.current_step)
        if self.context.mode == "reference":
            with forensic_span(self.context, "reference_router", self.layer):
                logits, weights, experts = self.module.reference(flattened)
        elif self.context.router_backend == "torch-device":
            # This is the exact official checkpoint router module which was
            # retained when HybridRouter replaced block.mlp.gate.  Its
            # parameters are already assigned from the real checkpoint on the
            # layer GPU.  Keeping the entire equation on-device removes four
            # host orchestration points per layer while preserving the
            # Transformers routing contract.
            with forensic_span(self.context, "router_torch_device", self.layer):
                logits, weights, experts = self.module.reference(flattened)
            self.context.engine_path_proof["router_torch_device_calls"] += 1
        else:
            if self.context.router_backend != "native-host":
                raise RuntimeError(
                    f"unsupported router backend: {self.context.router_backend}"
                )
            self.context.engine_path_proof["router_native_host_calls"] += 1
            router_input_transfer = {
                "direction": "d2h",
                "bytes": int(flattened.numel()) * np.dtype(np.float32).itemsize,
                "component": "router_input",
            }
            with forensic_span(self.context, "router_hidden_gpu_to_cpu", self.layer,
                               router_input_transfer):
                flattened_cpu = flattened.detach().float().cpu().numpy()
            self.context.engine_path_proof[
                "router_hidden_d2h_total_bytes"
            ] += router_input_transfer["bytes"]
            with forensic_span(self.context, "router_native", self.layer):
                logits_np, weights_np, experts_np = self.engine.route_topk_batch(
                    self.layer, flattened_cpu
                )
            router_output_transfer = {"direction": "h2d", "component": "router_output"}
            with forensic_span(self.context, "router_outputs_cpu_to_gpu", self.layer,
                               router_output_transfer):
                router_output_transfer["bytes"] = int(
                    np.asarray(logits_np).nbytes + np.asarray(weights_np).nbytes +
                    np.asarray(experts_np).astype(np.int64, copy=False).nbytes
                )
                logits = torch.from_numpy(np.asarray(logits_np)).to(
                    flattened.device, dtype=flattened.dtype
                )
                native_weights = torch.from_numpy(np.asarray(weights_np)).to(
                    flattened.device, dtype=flattened.dtype
                )
                native_experts = torch.from_numpy(np.asarray(experts_np).astype(np.int64)).to(
                    flattened.device, dtype=torch.long
                )
            self.context.engine_path_proof[
                "router_outputs_h2d_total_bytes"
            ] += router_output_transfer["bytes"]
            # torch.topk does not promise a stable order for exact ties. The
            # native route is authoritative except for rows where its expert
            # IDs differ from torch despite selecting the exact same logit
            # multiset. Canonicalize only those tie rows so trace ordering and
            # eager expert accumulation remain bit-for-bit reference-compatible.
            with forensic_span(self.context, "router_canonicalization", self.layer):
                probabilities = torch.softmax(logits, dim=-1, dtype=torch.float)
                canonical_weights, canonical_experts = torch.topk(
                    probabilities, self.top_k, dim=-1
                )
                canonical_weights = (
                    canonical_weights / canonical_weights.sum(dim=-1, keepdim=True)
                ).to(flattened.dtype)
                mismatch = torch.any(native_experts != canonical_experts, dim=-1)
                self.context.engine_path_proof["router_scalar_sync_calls"] += 1
                if bool(mismatch.any().item()):
                    native_selected = probabilities.gather(1, native_experts)
                    native_selected = (
                        native_selected / native_selected.sum(dim=-1, keepdim=True)
                    ).to(flattened.dtype)
                    if not torch.equal(native_selected[mismatch], canonical_weights[mismatch]):
                        raise RuntimeError(
                            f"dee.cpp router selected non-equivalent experts at layer {self.layer}"
                        )
                    self.context.router_tie_fallback_rows += int(mismatch.sum().item())
                    experts = torch.where(mismatch[:, None], canonical_experts, native_experts)
                    weights = torch.where(mismatch[:, None], canonical_weights, native_weights)
                else:
                    experts = native_experts
                    weights = native_weights
        if self.context.forensics is not None:
            self.context.forensics.record_routes(
                self.layer, self.context.current_step, self.context.current_phase,
                experts, weights,
            )
        if self.context.collector is not None:
            self.context.collector.router(self.layer, logits, weights, experts)
        return logits, weights, experts


class HybridRMSNorm:
    """Fail-closed Qwen RMSNorm CUDA launch for the M5E execution mode."""

    def __init__(self, module, engine, layer: int, context: ExecutionContext,
                 label: str):
        self.module = module
        self.reference_forward = module.forward
        self.engine = engine
        self.layer = layer
        self.context = context
        self.label = label
        module.forward = self.forward

    def forward(self, hidden_states):
        if (
            self.context.mode == "reference"
            or self.context.execution_mode != FUSED_NORM_EXECUTION_MODE
        ):
            return self.reference_forward(hidden_states)
        import torch

        weight = self.module.weight
        supported = (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.float16
            and hidden_states.is_contiguous()
            and weight.is_cuda
            and weight.dtype == torch.float16
            and weight.is_contiguous()
            and hidden_states.shape[-1] == weight.numel()
        )
        if (
            not supported
            or not hasattr(self.engine, "qwen_rms_norm_device")
        ):
            self.context.engine_path_proof["fused_norm_failures"] += 1
            raise RuntimeError(
                f"fused Qwen RMSNorm unsupported at {self.label}, "
                f"layer={self.layer}"
            )
        output = torch.empty_like(hidden_states)
        proof = self.context.engine_path_proof
        proof["fused_norm_output_allocations"] += 1
        dim = int(hidden_states.shape[-1])
        rows = int(hidden_states.numel() // dim)
        with forensic_span(
            self.context, "fused_qwen_rms_norm", self.layer,
            {"component": self.label, "rows": rows, "dim": dim},
        ):
            ok = self.engine.qwen_rms_norm_device(
                hidden_states.data_ptr(),
                weight.data_ptr(),
                output.data_ptr(),
                rows,
                dim,
                float(self.module.eps),
                torch.cuda.current_stream(
                    hidden_states.device
                ).cuda_stream,
            )
        if not ok:
            proof["fused_norm_failures"] += 1
            detail = (
                self.engine.last_error_message()
                if hasattr(self.engine, "last_error_message")
                else ""
            )
            raise RuntimeError(
                f"fused Qwen RMSNorm failed at {self.label}, "
                f"layer={self.layer}: {detail or '<no native detail>'}"
            )
        proof["fused_rms_norm_calls"] += 1
        return output


class HybridRMSNormGated:
    """Fail-closed Qwen linear-attention gated RMSNorm CUDA launch."""

    def __init__(self, module, engine, layer: int, context: ExecutionContext):
        self.module = module
        self.reference_forward = module.forward
        self.engine = engine
        self.layer = layer
        self.context = context
        module.forward = self.forward

    def forward(self, hidden_states, gate=None):
        if (
            self.context.mode == "reference"
            or self.context.execution_mode != FUSED_NORM_EXECUTION_MODE
        ):
            return self.reference_forward(hidden_states, gate)
        import torch

        weight = self.module.weight
        supported = (
            gate is not None
            and hidden_states.is_cuda
            and gate.is_cuda
            and hidden_states.dtype == torch.float16
            and gate.dtype == torch.float16
            and hidden_states.is_contiguous()
            and gate.is_contiguous()
            and hidden_states.shape == gate.shape
            and weight.is_cuda
            and weight.dtype == torch.float16
            and weight.is_contiguous()
            and hidden_states.shape[-1] == weight.numel()
        )
        if (
            not supported
            or not hasattr(self.engine, "qwen_rms_norm_gated_device")
        ):
            self.context.engine_path_proof["fused_norm_failures"] += 1
            raise RuntimeError(
                "fused Qwen gated RMSNorm unsupported at "
                f"layer={self.layer}"
            )
        output = torch.empty_like(hidden_states)
        proof = self.context.engine_path_proof
        proof["fused_norm_output_allocations"] += 1
        dim = int(hidden_states.shape[-1])
        rows = int(hidden_states.numel() // dim)
        with forensic_span(
            self.context, "fused_qwen_rms_norm_gated", self.layer,
            {"rows": rows, "dim": dim},
        ):
            ok = self.engine.qwen_rms_norm_gated_device(
                hidden_states.data_ptr(),
                weight.data_ptr(),
                gate.data_ptr(),
                output.data_ptr(),
                rows,
                dim,
                float(self.module.variance_epsilon),
                torch.cuda.current_stream(
                    hidden_states.device
                ).cuda_stream,
            )
        if not ok:
            proof["fused_norm_failures"] += 1
            detail = (
                self.engine.last_error_message()
                if hasattr(self.engine, "last_error_message")
                else ""
            )
            raise RuntimeError(
                "fused Qwen gated RMSNorm failed at "
                f"layer={self.layer}: {detail or '<no native detail>'}"
            )
        proof["fused_rms_norm_gated_calls"] += 1
        return output


def stable_combine_selected_experts(
        output, raw, top_k_weights, expert_ids_np: np.ndarray,
        *, legacy_accumulator: bool):
    """Combine raw expert rows in exact eager ascending-expert order."""
    import torch

    for token in range(raw.shape[0]):
        positions = np.argsort(
            expert_ids_np[token], kind="stable"
        ).tolist()
        if legacy_accumulator:
            accumulator = torch.zeros_like(output[token])
            for position in positions:
                accumulator.add_(
                    raw[token, position] * top_k_weights[token, position]
                )
            output[token] = accumulator
        else:
            for position in positions:
                output[token].add_(
                    raw[token, position] * top_k_weights[token, position]
                )
    return output


class HybridExperts:
    def __init__(self, nn_module, reference, engine, layer: int,
                 context: ExecutionContext):
        class Module(nn_module.Module):
            pass

        self.module = Module()
        self.module.reference = reference
        self.engine = engine
        self.layer = layer
        self.context = context
        self.engine_stream = None
        self.module.forward = self.forward
        self.module.clear_cache = reference.clear_cache

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.context.mode == "reference":
            return self.module.reference(hidden_states, top_k_index, top_k_weights)
        import torch

        if self.context.cache_disabled and not self.engine.reset_runtime_cache():
            raise RuntimeError(f"failed to reset diagnostic cache at layer {self.layer}")
        if hasattr(self.engine, "set_external_token"):
            self.engine.set_external_token(self.context.current_step)

        # Milestone 3 fix (defect #6): try device-resident MoE path first.
        # Keeps hidden_states and expert outputs on-device; only expert_ids
        # (tokens * topk * sizeof(int32)) cross the host boundary.
        # Falls back to the host path (d2h->native->h2d) if unavailable.
        # The proof counters in context.engine_path_proof increment on
        # every call so path-proof.json can attribute either:
        #   (a) the device-resident path was actually exercised;
        #   (b) the device path returned false and the silent fallback took
        #       over (which is observable as host_path_fallback_calls > 0).
        proof = self.context.engine_path_proof
        if self.context.execution_mode in (
            "native-combined",
            "native-combined-direct",
            FUSED_NORM_EXECUTION_MODE,
        ):
            direct_mode = (
                self.context.execution_mode in (
                    "native-combined-direct",
                    FUSED_NORM_EXECUTION_MODE,
                )
            )
            method_name = (
                "moe_forward_combined_direct_device"
                if direct_mode
                else "moe_forward_combined_device"
            )
            if not hasattr(self.engine, method_name):
                proof["native_combined_fallback_calls"] += 1
                if direct_mode:
                    proof["native_direct_fallback_calls"] += 1
                proof["last_native_error_combined_attempt"] = (
                    f"<{method_name} binding unavailable>"
                )
                raise RuntimeError(
                    f"{self.context.execution_mode} execution requested but "
                    f"the loaded pydee binary has no {method_name} binding"
                )
            combined_method = getattr(self.engine, method_name)
            combined_base_supported = (
                hidden_states.is_cuda
                and hidden_states.dtype == torch.float16
                and top_k_index.is_cuda
                and top_k_weights.is_cuda
            )
            if not combined_base_supported:
                proof["native_combined_fallback_calls"] += 1
                if direct_mode:
                    proof["native_direct_fallback_calls"] += 1
                proof["last_native_error_combined_attempt"] = (
                    "<combined path unsupported tensor contract>"
                )
                raise RuntimeError(
                    "native-combined execution requires CUDA FP16 hidden "
                    "plus CUDA expert IDs and routing weights; got "
                    f"hidden(device={hidden_states.device}, "
                    f"dtype={hidden_states.dtype}, "
                    f"contiguous={hidden_states.is_contiguous()}), "
                    f"ids(device={top_k_index.device}, "
                    f"dtype={top_k_index.dtype}, "
                    f"contiguous={top_k_index.is_contiguous()}), "
                    f"weights(device={top_k_weights.device}, "
                    f"dtype={top_k_weights.dtype}, "
                    f"contiguous={top_k_weights.is_contiguous()})"
                )
            # The HF router is allowed to return strided views and a routing
            # dtype other than the native ABI's int64/FP32. Canonicalize only
            # a violating tensor, and make every allocation/byte observable.
            combined_hidden = hidden_states.contiguous()
            combined_ids = top_k_index.to(dtype=torch.int64).contiguous()
            combined_weights = top_k_weights.to(
                dtype=torch.float32
            ).contiguous()
            for original, canonical in (
                (hidden_states, combined_hidden),
                (top_k_index, combined_ids),
                (top_k_weights, combined_weights),
            ):
                if canonical.data_ptr() != original.data_ptr():
                    proof["native_combined_input_materializations"] += 1
                    proof[
                        "native_combined_input_materialization_bytes"
                    ] += int(canonical.numel()) * int(
                        canonical.element_size()
                    )
            combined_output = torch.empty_like(hidden_states)
            raw_trace = None
            if self.context.collector is not None and self.layer in TRACE_LAYERS:
                raw_trace = torch.empty(
                    hidden_states.shape[0],
                    top_k_index.shape[1],
                    hidden_states.shape[1],
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                proof["native_combined_raw_trace_allocations"] += 1
            combined_ok = False
            if self.engine_stream is None:
                stream_handle = int(self.engine.compute_stream_handle())
                if stream_handle == 0:
                    raise RuntimeError(
                        "native-combined engine has no CUDA compute stream"
                    )
                self.engine_stream = torch.cuda.ExternalStream(
                    stream_handle, device=hidden_states.device
                )
            combined_start = time.perf_counter()
            with forensic_span(
                self.context, "expert_native_combined", self.layer
            ):
                proof["pybind_device_calls"] += 1
                combined_ok = combined_method(
                    self.layer,
                    combined_hidden.data_ptr(),
                    combined_hidden.shape[0],
                    combined_ids.data_ptr(),
                    combined_ids.shape[1],
                    combined_weights.data_ptr(),
                    combined_output.data_ptr(),
                    raw_trace.data_ptr() if raw_trace is not None else 0,
                    torch.cuda.current_stream(
                        hidden_states.device
                    ).cuda_stream,
                )
            # The engine uses these PyTorch-owned allocations from its
            # non-default compute stream. Recording that stream prevents
            # the caching allocator from recycling storage before native
            # work reaches the completion event handed back above.
            for tensor in (
                hidden_states,
                combined_hidden,
                combined_ids,
                combined_weights,
                combined_output,
                raw_trace,
            ):
                if tensor is not None:
                    tensor.record_stream(self.engine_stream)
            proof["expert_native_device_calls_total_ms"] += (
                (time.perf_counter() - combined_start) * 1000.0
            )
            if combined_ok:
                expert_ids_bytes = (
                    int(top_k_index.numel())
                    * int(top_k_index.element_size())
                )
                proof["native_combined_calls"] += 1
                if direct_mode:
                    proof["native_direct_calls"] += 1
                proof["native_combined_ids_d2h_total_bytes"] += (
                    expert_ids_bytes
                )
                proof["expert_ids_d2h_total_bytes"] += expert_ids_bytes
                proof["native_combined_stream_handoffs"] += 1
                proof["device_path_calls"] += 1
                if raw_trace is not None:
                    self.context.collector.add(
                        "selected_expert_outputs",
                        f"layer={self.layer}",
                        raw_trace.to(dtype=hidden_states.dtype),
                    )
                return combined_output
            proof["native_combined_fallback_calls"] += 1
            if direct_mode:
                proof["native_direct_fallback_calls"] += 1
            proof["last_native_error_combined_attempt"] = (
                self.engine.last_error_message()
                if hasattr(self.engine, "last_error_message")
                else ""
            ) or "<combined path returned false>"
            raise RuntimeError(
                "native-combined execution failed at "
                f"layer {self.layer}: "
                f"{proof['last_native_error_combined_attempt']}"
            )

        output = torch.zeros_like(hidden_states)
        expert_ids_bytes = int(top_k_index.numel()) * int(top_k_index.element_size())
        with forensic_span(self.context, "expert_ids_gpu_to_cpu", self.layer,
                           {"direction": "d2h", "component": "expert_ids_only",
                            "bytes": expert_ids_bytes}):
            expert_ids_np = top_k_index.detach().cpu().numpy().astype(np.int32)
        proof["expert_ids_d2h_total_bytes"] += expert_ids_bytes

        # Allocate FP32 device output (matches moe_forward_batch contract);
        # Python converts to FP16 via .to(dtype=...) below.
        raw_f32 = torch.empty(
            hidden_states.shape[0], top_k_index.shape[1], hidden_states.shape[1],
            dtype=torch.float32, device=hidden_states.device
        )
        proof["raw_output_allocations"] += 1
        native_device_start = time.perf_counter()
        with forensic_span(self.context, "expert_native_device", self.layer):
            proof["pybind_device_calls"] += 1
            device_ok = self.engine.moe_forward_batch_device(
                self.layer,
                hidden_states.data_ptr(), hidden_states.shape[0],
                expert_ids_np, top_k_index.shape[1],
                raw_f32.data_ptr(),
            )
        proof["expert_native_device_calls_total_ms"] += (
            (time.perf_counter() - native_device_start) * 1000.0
        )
        # Capture native diagnostic so the analyzer can attribute any failure
        # to a precise file/line/state, not the generic "<none>".  Updated on
        # every call, irrespective of device_ok, so first-failure forensics
        # is preserved when the host fallback path also fails.
        if hasattr(self.engine, "last_error_message"):
            proof["last_native_error_device_attempt"] = (
                self.engine.last_error_message()
                or (f"<empty: moe_forward_batch_device returned {device_ok} "
                    f"at layer {self.layer}, tokens={int(expert_ids_np.shape[0])}, "
                    f"topk={int(expert_ids_np.shape[1])}>")
            )
        if device_ok:
            proof["device_path_calls"] += 1
            # M3 deliverable: explicitly time the FP32->FP16 device conversion
            # performed after the device MoE forward returns raw_f32.  This
            # is the conversion the deliverable list calls "FP32 output
            # followed by Python .to(FP16)" and is the residual cost of the
            # device path that the previous host path absorbed inside its
            # own H2D->FP16 staging.
            conv_start = time.perf_counter()
            raw = raw_f32.to(dtype=hidden_states.dtype)
            proof["fp32_to_fp16_conversion_ms_total"] += (
                (time.perf_counter() - conv_start) * 1000.0
            )
        else:
            proof["host_path_fallback_calls"] += 1
            # Fall back to host path (original code path).  The host path
            # also incurs an FP32->FP16 device conversion: torch.from_numpy
            # materializes the FP32 expert_outputs numpy on-device, then
            # .to(dtype=hidden_states.dtype) casts FP32 to FP16.  Time it
            # with the same timer as the device path so the before/after
            # comparison does not unfairly hide the host path's conversion
            # cost.
            expert_input_transfer = {
                "direction": "d2h",
                "component": "expert_input",
                "bytes": int(hidden_states.numel()) * np.dtype(np.float32).itemsize,
            }
            native_host_start = time.perf_counter()
            with forensic_span(self.context, "expert_inputs_gpu_to_cpu", self.layer,
                               expert_input_transfer):
                hidden_cpu = hidden_states.detach().float().cpu().numpy()
            with forensic_span(self.context, "expert_native", self.layer):
                try:
                    proof["pybind_host_fallback_calls"] += 1
                    expert_outputs = self.engine.moe_forward_batch(
                        self.layer, hidden_cpu, expert_ids_np,
                    )
                except RuntimeError as e:
                    # Preserve the engine's last_error_message() detail --
                    # already appended to the RuntimeError -- in path-proof for
                    # the analyzer to attribute to a precise native line.
                    if hasattr(self.engine, "last_error_message"):
                        native = self.engine.last_error_message()
                        proof["last_native_error_host_fallback_attempt"] = (
                            f"[host-fallback-path] layer={self.layer} "
                            f"tokens={int(expert_ids_np.shape[0])} "
                            f"topk={int(expert_ids_np.shape[1])}: "
                            f"native={native or '<none>'}"
                        )
                    raise
            proof["expert_native_host_calls_total_ms"] += (
                (time.perf_counter() - native_host_start) * 1000.0
            )
            # The FP32 device output buffer was allocated for the device
            # path; on the host-fallback it is unused.  Release it before
            # the pageable H2D copy so the host-path pinned allocation does
            # not co-exist with an idle FP32 device buffer.  This is purely
            # VRAM hygiene (no architectural change) but also bound the
            # window in which an unrelated Python exception could let the
            # CUDA driver observe a dangling device pointer.
            del raw_f32
            host_conv_start = time.perf_counter()
            expert_output_transfer = {
                "direction": "h2d", "component": "expert_output",
                "bytes": int(np.asarray(expert_outputs).nbytes),
            }
            with forensic_span(self.context, "expert_outputs_cpu_to_gpu", self.layer,
                               expert_output_transfer):
                raw = torch.from_numpy(np.asarray(expert_outputs)).to(
                    hidden_states.device, dtype=hidden_states.dtype
                )
            proof["fp32_to_fp16_conversion_ms_total"] += (
                (time.perf_counter() - host_conv_start) * 1000.0
            )
        if tuple(raw.shape) != (
            hidden_states.shape[0], top_k_index.shape[1], hidden_states.shape[1]
        ):
            raise RuntimeError(
                f"dee.cpp batched expert output shape mismatch at layer {self.layer}: "
                f"{tuple(raw.shape)}"
            )
        with forensic_span(self.context, "expert_output_combination", self.layer):
            proof["python_combine_calls"] += 1
            stable_combine_selected_experts(
                output,
                raw,
                top_k_weights,
                expert_ids_np,
                legacy_accumulator=(
                    self.context.execution_mode == "debug-full-logit"
                ),
            )
        if self.context.collector is not None and self.layer in TRACE_LAYERS:
            self.context.collector.add(
                "selected_expert_outputs", f"layer={self.layer}", raw
            )
        return output


def assign_parameter(model, name: str, value, nn_module):
    parts = name.split(".")
    owner = model
    for part in parts[:-1]:
        owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
    old = getattr(owner, parts[-1])
    if tuple(old.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch for {name}: checkpoint={tuple(value.shape)} model={tuple(old.shape)}")
    setattr(owner, parts[-1], nn_module.Parameter(value, requires_grad=False))


def checkpoint_to_text_name(name: str) -> str | None:
    prefix = "model.language_model."
    if name.startswith(prefix):
        return "model." + name[len(prefix):]
    if name == "lm_head.weight":
        return name
    return None


def synchronize_all(torch, gpu_count: int):
    for device in range(gpu_count):
        torch.cuda.synchronize(device)


def tensor_tree_inventory(root, max_objects: int = 100_000):
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[int] = set()
    stack = [root]
    visited = 0
    while stack and visited < max_objects:
        value = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        visited += 1
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            key = (str(value.device), str(value.dtype))
            row = groups.setdefault(key, {
                "device": key[0], "dtype": key[1], "tensor_count": 0,
                "bytes": 0,
            })
            row["tensor_count"] += 1
            row["bytes"] += int(value.numel() * value.element_size())
            continue
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set)):
            stack.extend(value)
        elif hasattr(value, "__dict__"):
            stack.extend(vars(value).values())
    return {
        "groups": sorted(groups.values(), key=lambda item: (item["device"], item["dtype"])),
        "objects_visited": visited,
        "truncated": bool(stack),
    }


def load_runtime(model_dir: Path, gpu_count: int, split_layer: int, cache_experts: int,
                 *, profile_stages: bool = False, trace_requests: bool = False,
                 profile_timeline: bool = False,
                 debug_validate_cache: bool = False,
                 allow_diagnostic_sub_topk_cache: bool = False,
                 router_backend: str = "native-host",
                 phase_recorder=None):
    import_started = time.perf_counter()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoConfig, Qwen3_5MoeForCausalLM

    import pydee
    record_elapsed(phase_recorder, "native_and_runtime_imports", import_started)

    index_started = time.perf_counter()
    index = read_checkpoint_index(model_dir)
    record_elapsed(phase_recorder, "checkpoint_index_parsing", index_started)
    config_started = time.perf_counter()
    outer_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    text_config = outer_config.text_config
    text_config.use_cache = True
    text_config._attn_implementation = "eager"
    record_elapsed(phase_recorder, "model_configuration", config_started)
    structure_started = time.perf_counter()
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(text_config)
    record_elapsed(phase_recorder, "model_structure_construction", structure_started)

    context = ExecutionContext()
    if router_backend not in {"native-host", "torch-device"}:
        raise ValueError(f"unsupported router backend: {router_backend}")
    context.router_backend = router_backend
    pool = CheckpointPool(model_dir, phase_recorder)
    reference_experts = []
    reference_expert_caches = []
    for layer, block in enumerate(model.model.layers):
        lazy = LazyReferenceExperts(
            torch, nn, F, layer, pool, index, context,
            text_config.hidden_size, text_config.moe_intermediate_size,
            text_config.num_experts, cache_capacity=cache_experts,
        )
        block.mlp.experts = lazy.module
        reference_experts.append(lazy.module)
        reference_expert_caches.append(lazy.cache)

    metadata_started = time.perf_counter()
    expected = dict(model.named_parameters())
    load_plan: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for checkpoint_name, shard in index["weight_map"].items():
        target = checkpoint_to_text_name(checkpoint_name)
        if target in expected:
            load_plan[shard].append((checkpoint_name, target))
    record_elapsed(phase_recorder, "tensor_metadata_creation", metadata_started,
                   parameter_count=len(expected), shard_count=len(load_plan))

    loaded: set[str] = set()
    loaded_bytes = [0] * gpu_count
    started = time.perf_counter()
    dense_started = time.perf_counter()
    for shard in sorted(load_plan):
        handle = pool.handle(shard)
        for checkpoint_name, target in sorted(load_plan[shard]):
            layer = None
            parts = target.split(".")
            if "layers" in parts:
                layer = int(parts[parts.index("layers") + 1])
            if target.startswith("model.embed_tokens"):
                device_id = 0
            elif target.startswith("model.norm") or target.startswith("lm_head"):
                device_id = min(gpu_count - 1, 1)
            else:
                device_id = layer_device(layer if layer is not None else 0, gpu_count, split_layer)
            tensor = handle.get_tensor(checkpoint_name).to(
                device=f"cuda:{device_id}", dtype=torch.float16
            )
            assign_parameter(model, target, tensor, nn)
            loaded.add(target)
            loaded_bytes[device_id] += tensor.numel() * tensor.element_size()
    record_elapsed(phase_recorder, "dense_tensor_acquisition", dense_started,
                   tensor_count=len(loaded), bytes=sum(loaded_bytes))
    missing = sorted(set(expected) - loaded)
    if missing:
        raise RuntimeError(f"dense checkpoint load left meta parameter uninitialized: {missing[0]}")

    kernel_started = time.perf_counter()
    rotary = model.model.rotary_emb
    inv_freq, attention_scaling = rotary.compute_default_rope_parameters(
        text_config, torch.device("cuda:0")
    )
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq.clone()
    rotary.attention_scaling = attention_scaling
    record_elapsed(phase_recorder, "kernel_and_rotary_initialization", kernel_started)

    engines = []
    fused_norm_wrappers = []
    fused_norm_inventory = {
        "wrapped_regular": [],
        "wrapped_gated": [],
        "intentionally_unwrapped_full_attention_head_norms": [],
    }
    expert_bytes = 3 * text_config.hidden_size * text_config.moe_intermediate_size * 2
    budget_bytes = cache_experts * expert_bytes
    if allow_diagnostic_sub_topk_cache:
        if cache_experts < 1:
            raise ValueError("diagnostic expert cache must hold at least one expert")
    else:
        validate_expert_cache_budget(
            budget_bytes, text_config.hidden_size, text_config.moe_intermediate_size,
            text_config.num_experts_per_tok,
        )
    mmap_started = time.perf_counter()
    for layer, block in enumerate(model.model.layers):
        device_id = layer_device(layer, gpu_count, split_layer)
        shard_names = shard_paths_for_layer(index, layer)
        cfg = pydee.EngineConfig()
        cfg.shard_path = str(model_dir / shard_names[0])
        cfg.shard_paths = [str(model_dir / name) for name in shard_names]
        cfg.oracle_path = ""
        cfg.num_tokens = 1
        cfg.topk = text_config.num_experts_per_tok
        cfg.num_layers = text_config.num_hidden_layers
        cfg.num_experts = text_config.num_experts
        cfg.hidden = text_config.hidden_size
        cfg.inter = text_config.moe_intermediate_size
        cfg.base_layer = layer
        cfg.device_id = device_id
        cfg.budget_bytes = budget_bytes
        cfg.prefetch_depth = 1
        cfg.use_cuda = True
        cfg.cache_dtype = pydee.DeviceCacheDType.Fp16
        cfg.transfer_dtype = pydee.WeightTransferDType.Bf16
        cfg.verbose = False
        cfg.profile_stages = profile_stages
        cfg.trace_requests = trace_requests
        cfg.profile_timeline = profile_timeline
        cfg.debug_validate_cache = debug_validate_cache
        engine = pydee.Engine()
        if not engine.init(cfg):
            raise RuntimeError(f"dee.cpp Engine::init failed for layer {layer}")
        engines.append(engine)

        reference_router = block.mlp.gate
        reference_expert = block.mlp.experts
        block.mlp.gate = HybridRouter(
            nn, reference_router, engine, layer, context,
            text_config.num_experts_per_tok,
        ).module
        block.mlp.experts = HybridExperts(
            nn, reference_expert, engine, layer, context
        ).module

        input_norm_wrapper = HybridRMSNorm(
            block.input_layernorm,
            engine,
            layer,
            context,
            "input_layernorm",
        )
        post_norm_wrapper = HybridRMSNorm(
            block.post_attention_layernorm,
            engine,
            layer,
            context,
            "post_attention_layernorm",
        )
        fused_norm_wrappers.extend([
            input_norm_wrapper,
            post_norm_wrapper,
        ])
        fused_norm_inventory["wrapped_regular"].extend([
            f"layer={layer}:input_layernorm",
            f"layer={layer}:post_attention_layernorm",
        ])
        if hasattr(block, "linear_attn"):
            fused_norm_wrappers.append(
                HybridRMSNormGated(
                    block.linear_attn.norm, engine, layer, context
                )
            )
            fused_norm_inventory["wrapped_gated"].append(
                f"layer={layer}:linear_attn.norm"
            )
        if hasattr(block, "self_attn"):
            fused_norm_inventory[
                "intentionally_unwrapped_full_attention_head_norms"
            ].extend([
                f"layer={layer}:self_attn.q_norm",
                f"layer={layer}:self_attn.k_norm",
            ])
    record_elapsed(phase_recorder, "initial_mmap_and_engine_setup", mmap_started,
                   engines=len(engines), cache_experts=cache_experts)

    fused_norm_wrappers.append(
        HybridRMSNorm(
            model.model.norm,
            engines[-1],
            text_config.num_hidden_layers,
            context,
            "final_norm",
        )
    )
    fused_norm_inventory["wrapped_regular"].append("final_norm")

    # Move layer inputs and kwargs at the exact 19->20 pipeline boundary.
    for layer, block in enumerate(model.model.layers):
        device = torch.device(f"cuda:{layer_device(layer, gpu_count, split_layer)}")

        def move_to_layer(_module, args, kwargs, device=device, layer=layer):
            moved_bytes = tensor_transfer_bytes(args, device) + tensor_transfer_bytes(
                {key: kwargs.get(key) for key in (
                    "attention_mask", "position_ids", "position_embeddings"
                )}, device)
            transfer_metadata = {
                "direction": "d2d" if moved_bytes else "none",
                "bytes": moved_bytes,
                "destination": str(device),
                "component": "pipeline_boundary",
            }
            with forensic_span(context, "inter_device_transfer", layer,
                               transfer_metadata):
                args = list(args)
                args[0] = args[0].to(device)
                for key in ("attention_mask", "position_ids"):
                    if kwargs.get(key) is not None:
                        kwargs[key] = kwargs[key].to(device)
                if kwargs.get("position_embeddings") is not None:
                    kwargs["position_embeddings"] = tuple(
                        tensor.to(device) for tensor in kwargs["position_embeddings"]
                    )
                return tuple(args), kwargs

        block.register_forward_pre_hook(move_to_layer, with_kwargs=True)

        if layer in TRACE_LAYERS:
            def capture_layer(_module, _args, output, layer=layer):
                if context.collector is not None:
                    context.collector.add(
                        "intermediate_hidden_states", f"layer={layer}", output
                    )
            block.register_forward_hook(capture_layer)

    def capture_embedding(_module, _args, output):
        if context.collector is not None:
            context.collector.add("embedding_output", "embedding", output)

    def capture_final(_module, _args, output):
        if context.collector is not None:
            context.collector.add("final_hidden_state", "final_norm", output)

    model.model.embed_tokens.register_forward_hook(capture_embedding)
    model.model.norm.register_forward_hook(capture_final)
    finalize_started = time.perf_counter()
    model.eval()
    synchronize_all(torch, gpu_count)
    record_elapsed(phase_recorder, "model_setup_finalization", finalize_started)
    return {
        "model": model,
        "config": text_config,
        "context": context,
        "pool": pool,
        "engines": engines,
        "reference_experts": reference_experts,
        "reference_expert_caches": reference_expert_caches,
        "fused_norm_wrappers": fused_norm_wrappers,
        "fused_norm_inventory": fused_norm_inventory,
        "load_seconds": time.perf_counter() - started,
        "dense_loaded_bytes": {f"cuda:{i}": value for i, value in enumerate(loaded_bytes)},
        "expert_cache_budget_per_layer": budget_bytes,
    }


def tokenize_prompt(tokenizer, prompt: str, chat_template: bool):
    if chat_template:
        if tokenizer.chat_template is None:
            raise ValueError("--chat-template requested but tokenizer has no chat template")
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        )
    else:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    return ids


def run_generation(runtime, tokenizer, prompt: str, max_new_tokens: int,
                   chat_template: bool, mode: str, trace: bool,
                   execution_mode: str = "production"):
    import torch

    if execution_mode not in EXECUTION_MODES:
        raise ValueError(
            f"unsupported execution mode {execution_mode!r}; "
            f"expected one of {EXECUTION_MODES}"
        )
    model = runtime["model"]
    config = runtime["config"]
    context = runtime["context"]
    if execution_mode == "parity" and not trace:
        raise ValueError("parity execution mode requires trace=True")
    if execution_mode == "profiler" and context.forensics is None:
        raise ValueError(
            "profiler execution mode requires an attached forensic recorder"
        )
    gpu_count = torch.cuda.device_count()
    context.mode = mode
    context.execution_mode = execution_mode
    context.executed_router_layers = set()
    tie_fallback_start = context.router_tie_fallback_rows
    collector = TraceCollector() if trace else None
    context.collector = collector
    tokenization_phase = (
        context.forensics.phase("prompt_tokenization", {"mode": mode, "prompt": prompt})
        if context.forensics is not None else contextlib.nullcontext()
    )
    with tokenization_phase:
        prompt_ids_cpu = tokenize_prompt(tokenizer, prompt, chat_template)
    prompt_transfer = {
        "direction": "h2d", "component": "prompt_ids",
        "bytes": int(prompt_ids_cpu.numel()) * int(prompt_ids_cpu.element_size()),
    }
    with forensic_span(context, "prompt_input_cpu_to_gpu", -1, prompt_transfer):
        prompt_ids = prompt_ids_cpu.to("cuda:0")
    prompt_tokens = int(prompt_ids.shape[1])
    capture_full_logits = execution_mode == "debug-full-logit"
    if capture_full_logits:
        # Evidence/debug mode intentionally retains the historical allocation
        # behavior, making it a same-session A/B control for the exact M5B
        # production-path changes.
        attention_mask_storage = None
        next_token_buffer = None
        attention_mask = torch.ones_like(prompt_ids, device="cuda:0")
    else:
        # M5B production path: allocate token/mask storage once. The model
        # still receives the exact growing attention-mask shape through a
        # view, but no per-token tensor construction or torch.cat allocation
        # occurs.
        attention_mask_storage = torch.ones(
            (prompt_ids.shape[0], prompt_tokens + max_new_tokens),
            dtype=prompt_ids.dtype,
            device="cuda:0",
        )
        attention_mask = attention_mask_storage[:, :prompt_tokens]
        next_token_buffer = torch.empty(
            (prompt_ids.shape[0], 1), dtype=torch.long, device="cuda:0"
        )
    current_input = prompt_ids
    past_key_values = None
    generated: list[int] = []
    # Full vocabulary copies are evidence/debug-only. Parity already captures
    # lm_head_logits through TraceCollector; production and profiler modes
    # must not unconditionally copy every logits row to host.
    logits_records = []
    step_seconds = []
    cache_lengths = []
    eos = config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()

    monitor = ResourceMonitor(torch, gpu_count).start()
    synchronize_all(torch, gpu_count)
    total_start = time.perf_counter()
    with torch.inference_mode():
        for step in range(max_new_tokens):
            context.current_step = step
            context.current_phase = "prefill" if step == 0 else "decode"
            if context.forensics is not None:
                context.forensics.begin_step(step, context.current_phase)
            # Milestone 2.5 fix (defects #1/#7): the per-step synchronize_all
            # across all visible GPUs was pure serialization. HF's sequential
            # layer walk already serializes the inter-layer residual chain
            # (layer N+1 reads layer N's hidden), and the next step's argmax
            # .item() forces its own CPU sync of the logits. Removing these two
            # barriers lets layer 20's expert H2D prefetch on cuda:1 overlap
            # with layer 19's compute on cuda:0, eliminating the measured
            # sequential 20/20 two-GPU pipeline (2249 ms model wall, 1/18
            # dual-active NVML samples). step_start still measures the warm
            # decode pass accurately; the argmax sync anchors the boundary.
            with forensic_span(context, "step_pre_synchronize", -1,
                               {"barrier": "removed_for_overlap"}):
                pass
            step_start = time.perf_counter()
            model_profile = (
                context.forensics.profile_model_call(step)
                if context.forensics is not None else contextlib.nullcontext()
            )
            with model_profile:
                outputs = model(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
            # Milestone 2.5 fix (defects #1/#7): no per-step post barrier. The
            # logits argmax below performs the necessary CPU sync; any further
            # GPU work drains naturally before the next step. Keeping this
            # barrier would re-introduce the measured 4.84 ms step_post gap
            # and re-serialize the two GPUs after each token.
            with forensic_span(context, "step_post_synchronize", -1,
                               {"barrier": "removed_for_overlap"}):
                pass
            model_call_elapsed = time.perf_counter() - step_start
            logits = outputs.logits[:, -1, :]
            if collector is not None:
                collector.add("lm_head_logits", f"step={step}", logits)
            if capture_full_logits:
                logits_records.append(logits.detach().float().cpu().numpy())
            with forensic_span(context, "token_selection_and_item_sync", -1,
                               {"barrier": "cuda_scalar_to_host"}):
                next_token = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_token)
            past_key_values = outputs.past_key_values
            try:
                cache_lengths.append(int(past_key_values.get_seq_length()))
            except Exception:
                cache_lengths.append(prompt_ids.shape[1] + step)
            stopped = next_token in eos_ids
            if not stopped:
                if capture_full_logits:
                    current_input = torch.tensor(
                        [[next_token]], dtype=torch.long, device="cuda:0"
                    )
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (1, 1),
                                dtype=attention_mask.dtype,
                                device="cuda:0",
                            ),
                        ],
                        dim=1,
                    )
                else:
                    next_token_buffer.fill_(next_token)
                    current_input = next_token_buffer
                    attention_mask = attention_mask_storage[
                        :, :prompt_tokens + step + 1
                    ]
            # Full-token latency includes model execution, any evidence-only
            # logits D2H, greedy selection/item synchronization, and token/mask
            # maintenance. The forensic snapshot below is deliberately outside
            # the measured interval.
            full_token_elapsed = time.perf_counter() - step_start
            step_seconds.append(full_token_elapsed)
            if context.forensics is not None:
                context.forensics.end_step(
                    step,
                    context.current_phase,
                    model_call_elapsed,
                    full_token_seconds=full_token_elapsed,
                )
            if stopped:
                break
    synchronize_all(torch, gpu_count)
    total_seconds = time.perf_counter() - total_start
    resources = monitor.stop()
    context.collector = None
    prefill_seconds = step_seconds[0] if step_seconds else 0.0
    decode_seconds = step_seconds[1:]
    detokenization_phase = (
        context.forensics.phase("detokenization", {"mode": mode, "prompt": prompt})
        if context.forensics is not None else contextlib.nullcontext()
    )
    with detokenization_phase:
        generated_text = tokenizer.decode(generated, skip_special_tokens=False)
        full_text = tokenizer.decode(
            prompt_ids_cpu[0].tolist() + generated, skip_special_tokens=False
        )
    return {
        "mode": mode,
        "execution_mode": execution_mode,
        "prompt": prompt,
        "prompt_token_ids": prompt_ids_cpu[0].tolist(),
        "prompt_token_count": prompt_tokens,
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "full_text": full_text,
        "stop_reason": "eos" if generated and generated[-1] in eos_ids else "max_new_tokens",
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": prompt_tokens / prefill_seconds if prefill_seconds else 0.0,
        "time_to_first_token_seconds": prefill_seconds,
        "per_token_decode_seconds": decode_seconds,
        "single_stream_decode_tokens_per_second": (
            len(decode_seconds) / sum(decode_seconds) if decode_seconds else 0.0
        ),
        "total_generation_seconds": total_seconds,
        "cache_sequence_lengths": cache_lengths,
        "recurrent_or_kv_state": tensor_tree_inventory(past_key_values),
        "live_generation_inputs": tensor_tree_inventory({
            "current_input": current_input,
            "attention_mask": attention_mask,
            "attention_mask_storage": attention_mask_storage,
            "next_token_buffer": next_token_buffer,
        }),
        "resources": resources,
        "router_tie_fallback_rows": context.router_tie_fallback_rows - tie_fallback_start,
        "executed_router_layers": sorted(context.executed_router_layers),
        "collector": collector,
        "logits_records": logits_records,
    }


def flatten_records(records: list[tuple[str, np.ndarray]]):
    if not records:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate([array.reshape(-1) for _, array in records])


def compare_trace(reference: TraceCollector, candidate: TraceCollector):
    report = {}
    categories = set(TOLERANCES) | {"expert_ids"}
    for category in sorted(categories):
        ref_records = reference.records.get(category, [])
        got_records = candidate.records.get(category, [])
        labels_match = [label for label, _ in ref_records] == [label for label, _ in got_records]
        if not labels_match or len(ref_records) != len(got_records):
            report[category] = {
                "passed": False,
                "first_failing_operation": "record_sequence",
                "reference_record_count": len(ref_records),
                "candidate_record_count": len(got_records),
            }
            continue
        atol, rtol = (0.0, 0.0) if category == "expert_ids" else TOLERANCES[category]
        comparison = numeric_comparison(
            flatten_records(ref_records), flatten_records(got_records), atol, rtol
        )
        comparison["operation_count"] = len(ref_records)
        comparison["first_failing_operation"] = None
        comparison["operation_summaries"] = []
        for (label, ref), (_, got) in zip(ref_records, got_records):
            local = numeric_comparison(ref, got, atol, rtol)
            comparison["operation_summaries"].append({
                "operation": label,
                "maximum_absolute_error": local.get("maximum_absolute_error"),
                "mean_absolute_error": local.get("mean_absolute_error"),
                "passed": local["passed"],
            })
            if comparison["first_failing_operation"] is None and not local["passed"]:
                comparison["first_failing_operation"] = label
                comparison["first_failing_index"] = local.get("first_failing_index")
        report[category] = comparison
    report["all_categories_passed"] = all(item.get("passed", False) for item in report.values())
    return report


def engine_stats(runtime):
    aggregate = collections.Counter()
    by_device = collections.defaultdict(collections.Counter)
    visible_gpus = min(__import__("torch").cuda.device_count(), 2)
    by_layer = []
    numeric_keys = (
        "cache_hits", "cache_loads", "cold_loads", "resident_hits", "inflight_hits",
        "evictions", "fallbacks", "prefetch_issued", "prefetch_fallbacks",
        "duplicate_requests", "h2d_bytes", "h2d_copies", "peak_vram", "current_vram",
        "resident_experts", "host_pinned_expert_staging_bytes",
        "host_pageable_expert_staging_bytes", "host_router_weight_bytes",
        "host_hidden_buffer_bytes", "host_moe_dispatch_bytes",
        "host_prefetch_ring_bytes",
        "host_prefetch_ring_slots", "peak_transient_host_bytes",
        "device_expert_cache_reserved_bytes", "device_prefetch_staging_bytes",
        "device_fixed_work_buffer_bytes", "device_router_weight_bytes",
        "device_router_dynamic_bytes", "device_moe_batch_buffer_bytes",
        "device_moe_raw_workspace_bytes",
        "d2d_gather_copies", "d2d_gather_bytes",
        "d2d_scatter_copies", "d2d_scatter_bytes",
        "direct_row_gather_bypasses", "direct_row_scatter_bypasses",
        "device_oracle_scratch_bytes",
    )
    for layer, engine in enumerate(runtime["engines"]):
        stats = json.loads(engine.last_stats_json())
        device = f"cuda:{layer_device(layer, visible_gpus)}"
        by_layer.append({"layer": layer, "device": device, **stats})
        for key in numeric_keys:
            aggregate[key] += stats.get(key, 0)
            by_device[device][key] += stats.get(key, 0)
    misses = aggregate["cold_loads"]
    hits = aggregate["resident_hits"] + aggregate["inflight_hits"]
    aggregate["cache_misses"] = misses
    aggregate["cache_hit_rate"] = hits / (hits + misses) if hits + misses else 0.0
    return {"aggregate": dict(aggregate),
            "by_device": {device: dict(values) for device, values in by_device.items()},
            "by_layer": by_layer}


def stats_delta(before: dict[str, Any], after: dict[str, Any]):
    keys = set(before["aggregate"]) | set(after["aggregate"])
    values = {key: after["aggregate"].get(key, 0) - before["aggregate"].get(key, 0)
              for key in keys if key != "cache_hit_rate"}
    hits = values.get("resident_hits", 0) + values.get("inflight_hits", 0)
    misses = values.get("cold_loads", 0)
    values["cache_hit_rate"] = hits / (hits + misses) if hits + misses else 0.0
    return values


def serializable_generation(result):
    return {key: value for key, value in result.items()
            if key not in {"collector", "logits_records"}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompt", action="append", required=True,
                        help="Prompt to generate; repeat for parity validation")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--greedy", action="store_true", help="Required deterministic mode")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="production",
        help=(
            "production removes evidence-only host copies; parity enables trace "
            "comparison; profiler enables measurement hooks; debug-full-logit "
            "also retains full vocabulary logits on host; native-combined "
            "uses the exact M5C C++ weighted-combine and stream handoff; "
            "native-combined-direct additionally bypasses one-row D2D copies; "
            "native-combined-direct-fused-norm also replaces layer input/post, "
            "final, and linear-attention gated Qwen RMSNorm kernels with the "
            "explicit M5E CUDA contract (full-attention q/k head norms remain "
            "eager and are separately inventoried)"
        ),
    )
    parser.add_argument("--reference-parity", action="store_true")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--require-dual-gpu", action="store_true")
    parser.add_argument("--split-layer", type=int, default=20)
    parser.add_argument("--cache-experts", type=int, default=8)
    parser.add_argument(
        "--router-backend",
        choices=("native-host", "torch-device"),
        default="native-host",
        help=(
            "native-host reproduces the sealed baseline; torch-device runs the "
            "official checkpoint router on the layer GPU while dee.cpp retains "
            "all expert execution"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ornith_generation_output"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.greedy:
        raise ValueError("only deterministic --greedy generation is supported")
    if args.max_new_tokens < 2:
        raise ValueError("Milestone 2 requires at least two consecutive generated tokens")
    if args.reference_parity and len(args.prompt) < 3:
        raise ValueError("--reference-parity requires at least three --prompt values")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from transformers import AutoTokenizer

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("real Ornith generation requires at least one CUDA GPU")
    if args.require_dual_gpu and gpu_count != 2:
        raise RuntimeError(f"dual-T4 proof requires exactly 2 visible GPUs, found {gpu_count}")
    effective_gpu_count = min(gpu_count, 2)
    machine = {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": gpu_count,
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                  "total_memory": torch.cuda.get_device_properties(i).total_memory}
                 for i in range(gpu_count)],
    }
    print("MACHINE=" + json.dumps(machine, sort_keys=True), flush=True)

    global_monitor = ResourceMonitor(torch, effective_gpu_count).start()
    tensor_rows, tensor_summary = build_complete_tensor_map(
        args.model_dir, effective_gpu_count, args.split_layer
    )
    (args.output_dir / "tensor-map.json").write_text(
        json.dumps({"summary": tensor_summary, "tensors": tensor_rows}, indent=2),
        encoding="utf-8",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    runtime = load_runtime(
        args.model_dir, effective_gpu_count, args.split_layer, args.cache_experts,
        router_backend=args.router_backend,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "result": "running",
        "machine": machine,
        "checkpoint": {
            "model_dir": str(args.model_dir),
            "config_sha256": sha256_file(args.model_dir / "config.json"),
            "index_sha256": sha256_file(args.model_dir / "model.safetensors.index.json"),
            "tensor_summary": tensor_summary,
            "revision": "kaggle-dataset:nivind/ornith-1-0-35b",
        },
        "git_commit": git_revision(),
        "layout": {
            "cuda:0": "embedding, rotary, transformer layers 0-19, layer-local expert caches",
            "cuda:1": "transformer layers 20-39, final norm, LM head, layer-local expert caches",
            "pipeline_transfer": "hidden, position embeddings, attention mask, and position ids at layer 20",
            "expert_cache_capacity_per_layer": args.cache_experts,
            "expert_cache_budget_per_layer_bytes": runtime["expert_cache_budget_per_layer"],
            "router_backend": args.router_backend,
            "dense_loaded_bytes": runtime["dense_loaded_bytes"],
        },
        "runtime_load_seconds": runtime["load_seconds"],
        "validation": [],
    }

    if args.benchmark:
        for engine in runtime["engines"]:
            if not engine.reset_runtime_cache():
                raise RuntimeError("failed to reset dee.cpp expert cache")
        baseline = engine_stats(runtime)
        cold = run_generation(
            runtime, tokenizer, args.prompt[0], args.max_new_tokens,
            args.chat_template, "dee", trace=False,
            execution_mode="production",
        )
        after_cold = engine_stats(runtime)
        warm = run_generation(
            runtime, tokenizer, args.prompt[0], args.max_new_tokens,
            args.chat_template, "dee", trace=False,
            execution_mode="production",
        )
        after_warm = engine_stats(runtime)
        report["benchmark"] = {
            "cold": serializable_generation(cold),
            "warm": serializable_generation(warm),
            "cold_expert_cache": stats_delta(baseline, after_cold),
            "warm_expert_cache": stats_delta(after_cold, after_warm),
            "note": "decode TPS excludes prefill and the first token; H2D bytes cover dee.cpp expert-cache transfers",
        }

    candidate_outputs = []
    for prompt in args.prompt:
        reference = None
        if args.reference_parity:
            reference = run_generation(
                runtime, tokenizer, prompt, args.max_new_tokens,
                args.chat_template, "reference", trace=True,
                execution_mode="parity",
            )
            for expert in runtime["reference_experts"]:
                expert.clear_cache()
            torch.cuda.empty_cache()
        candidate = run_generation(
            runtime, tokenizer, prompt, args.max_new_tokens,
            args.chat_template, "dee", trace=args.reference_parity,
            execution_mode=("parity" if args.reference_parity else args.execution_mode),
        )
        candidate_outputs.append(serializable_generation(candidate))
        item = {
            "prompt": prompt,
            "candidate": serializable_generation(candidate),
            "all_40_layers_executed": (
                candidate["executed_router_layers"] == list(range(40))
            ),
        }
        if reference is not None:
            item["reference"] = serializable_generation(reference)
            item["prompt_token_ids_exact"] = (
                reference["prompt_token_ids"] == candidate["prompt_token_ids"]
            )
            item["generated_token_ids_exact"] = (
                reference["generated_token_ids"] == candidate["generated_token_ids"]
            )
            item["decoded_text_exact"] = (
                reference["generated_text"] == candidate["generated_text"]
            )
            item["comparisons"] = compare_trace(
                reference["collector"], candidate["collector"]
            )
            item["passed"] = (
                item["prompt_token_ids_exact"] and
                item["generated_token_ids_exact"] and
                item["decoded_text_exact"] and
                item["all_40_layers_executed"] and
                item["comparisons"]["all_categories_passed"]
            )
        else:
            item["passed"] = item["all_40_layers_executed"] or not args.reference_parity
        report["validation"].append(item)
        print("GENERATION=" + json.dumps({
            "prompt": prompt,
            "tokens": candidate["generated_token_ids"],
            "text": candidate["generated_text"],
            "passed": item["passed"],
        }, ensure_ascii=False), flush=True)

    report["expert_cache_final"] = engine_stats(runtime)
    report["overall_resources"] = global_monitor.stop()
    report["process_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if resource is not None else report["overall_resources"]["peak_host_rss_bytes"]
    )
    report["result"] = "PASS" if all(item["passed"] for item in report["validation"]) else "FAIL"
    report_path = args.output_dir / "ornith-milestone2-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = args.output_dir / "ornith-milestone2-summary.txt"
    summary_path.write_text(
        "\n".join([
            f"result={report['result']}",
            f"git_commit={report['git_commit']}",
            f"prompts={len(report['validation'])}",
            f"all_40_layers={all(item['all_40_layers_executed'] for item in report['validation'])}",
            f"report={report_path}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"REPORT={report_path}", flush=True)
    print(f"RESULT={report['result']}", flush=True)
    runtime["pool"].close()
    if report["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
