#!/usr/bin/env python3
"""Bounded timing, route, and utilization probes for Ornith Milestone 2.5.

This module is deliberately measurement-only.  It adds hooks and CUDA events,
but it does not change placement, cache policy, kernels, or model math.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


class NvmlSampler:
    """Sample both GPUs without failing the run when NVML is unavailable."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handles: list[Any] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception as exc:  # pragma: no cover - depends on CUDA host
            self.error = f"{type(exc).__name__}: {exc}"

    def _sample_one(self, index: int, handle: Any) -> dict[str, Any]:
        nvml = self._nvml
        row: dict[str, Any] = {"index": index}
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            row["gpu_utilization_percent"] = int(util.gpu)
            row["memory_controller_utilization_percent"] = int(util.memory)
        except Exception:
            pass
        try:
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            row.update({
                "memory_used_bytes": int(memory.used),
                "memory_free_bytes": int(memory.free),
                "memory_total_bytes": int(memory.total),
            })
        except Exception:
            pass
        for label, counter in (
            ("pcie_tx_bytes_per_second", getattr(nvml, "NVML_PCIE_UTIL_TX_BYTES", None)),
            ("pcie_rx_bytes_per_second", getattr(nvml, "NVML_PCIE_UTIL_RX_BYTES", None)),
        ):
            if counter is None:
                continue
            try:
                # NVML reports KiB/s for this API.
                row[label] = int(nvml.nvmlDeviceGetPcieThroughput(handle, counter)) * 1024
            except Exception:
                pass
        return row

    def _run(self) -> None:
        origin = time.perf_counter_ns()
        while not self._stop.wait(self.interval_seconds):
            stamp = time.perf_counter_ns()
            self.samples.append({
                "monotonic_ns": stamp,
                "elapsed_ms": (stamp - origin) / 1_000_000.0,
                "gpus": [
                    self._sample_one(index, handle)
                    for index, handle in enumerate(self._handles)
                ],
            })

    def start(self) -> "NvmlSampler":
        if self._nvml is not None:
            self._thread = threading.Thread(
                target=self._run, name="milestone25-nvml", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return {"interval_seconds": self.interval_seconds,
                "error": self.error, "samples": self.samples}


class ForensicTimingRecorder:
    """Collect per-step/layer wall spans and non-blocking CUDA-event timings."""

    def __init__(self, torch_module, gpu_count: int, split_layer: int = 20,
                 sample_nvml: bool = True, memory_probe: Any | None = None,
                 profile_decode_step: int | None = None,
                 profiler_trace_path: Path | None = None):
        self.torch = torch_module
        self.gpu_count = gpu_count
        self.split_layer = split_layer
        self.wall_spans: list[dict[str, Any]] = []
        self.phase_timings: list[dict[str, Any]] = []
        self.route_selections: list[dict[str, Any]] = []
        self.step_records: list[dict[str, Any]] = []
        self.profile_snapshots: list[dict[str, Any]] = []
        self.engine_profiles: list[dict[str, Any]] = []
        self.engine_timelines: list[dict[str, Any]] = []
        self._engines: list[Any] = []
        self._hook_handles: list[Any] = []
        self._hook_starts: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._pending_cuda: list[dict[str, Any]] = []
        self._step_begin_ns: int | None = None
        self._overhead_ns = 0
        self.memory_probe = memory_probe
        self.profile_decode_step = profile_decode_step
        self.profiler_trace_path = profiler_trace_path
        self.torch_profiler_summary: dict[str, Any] = {
            "enabled": profile_decode_step is not None,
            "captured_step": None,
            "operators": [],
            "error": None,
        }
        self.nvml = NvmlSampler().start() if sample_nvml else None

    @contextlib.contextmanager
    def phase(self, name: str, metadata: dict[str, Any] | None = None):
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            ended = time.perf_counter_ns()
            self.phase_timings.append({
                "name": name,
                "start_monotonic_ns": started,
                "end_monotonic_ns": ended,
                "wall_ms": (ended - started) / 1_000_000.0,
                "metadata": metadata or {},
            })

    def record_phase(self, name: str, wall_seconds: float,
                     metadata: dict[str, Any] | None = None) -> None:
        self.phase_timings.append({
            "name": name,
            "wall_ms": float(wall_seconds) * 1000.0,
            "metadata": metadata or {},
        })

    def device_for_layer(self, layer: int) -> int:
        return 0 if self.gpu_count == 1 or layer < self.split_layer else 1

    @contextlib.contextmanager
    def span(self, name: str, layer: int, step: int, phase: str,
             metadata: dict[str, Any] | None = None):
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            ended = time.perf_counter_ns()
            row = {
                "name": name,
                "layer": int(layer),
                "step": int(step),
                "phase": phase,
                "gpu": self.device_for_layer(layer),
                "start_monotonic_ns": started,
                "end_monotonic_ns": ended,
                "cpu_wall_ms": (ended - started) / 1_000_000.0,
            }
            if metadata:
                row["metadata"] = dict(metadata)
            self.wall_spans.append(row)

    def set_engines(self, engines: list[Any]) -> None:
        self._engines = list(engines)

    def reset_engine_profiles(self) -> None:
        for layer, engine in enumerate(self._engines):
            if not hasattr(engine, "reset_external_profile"):
                raise RuntimeError(
                    f"pydee engine {layer} lacks Milestone 2.5 profiling bindings"
                )
            if not engine.reset_external_profile():
                raise RuntimeError(f"failed to reset external profile for layer {layer}")

    def begin_step(self, step: int, phase: str) -> None:
        self._step_begin_ns = time.perf_counter_ns()
        self.step_records.append({
            "step": int(step), "phase": phase,
            "begin_monotonic_ns": self._step_begin_ns,
        })
        if self.memory_probe is not None:
            self.memory_probe.checkpoint(
                f"before_{phase}_step_{step}", include_smaps=False,
                include_cuda=True, include_nvml=True,
                metadata={"step": step, "phase": phase},
            )

    @contextlib.contextmanager
    def profile_model_call(self, step: int):
        if step != self.profile_decode_step:
            yield
            return
        try:
            profiler_module = self.torch.profiler
            activities = [profiler_module.ProfilerActivity.CPU]
            if self.torch.cuda.is_available():
                activities.append(profiler_module.ProfilerActivity.CUDA)
            with profiler_module.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            ) as profile:
                yield
            operators = []
            for row in profile.key_averages():
                operators.append({
                    "key": row.key,
                    "count": int(row.count),
                    "cpu_time_total_us": float(row.cpu_time_total),
                    "self_cpu_time_total_us": float(row.self_cpu_time_total),
                    "cuda_time_total_us": float(getattr(row, "cuda_time_total", 0.0)),
                    "self_cuda_time_total_us": float(
                        getattr(row, "self_cuda_time_total", 0.0)
                    ),
                    "cpu_memory_usage_bytes": int(getattr(row, "cpu_memory_usage", 0)),
                    "cuda_memory_usage_bytes": int(getattr(row, "cuda_memory_usage", 0)),
                })
            operators.sort(
                key=lambda item: item["self_cuda_time_total_us"] +
                                 item["self_cpu_time_total_us"],
                reverse=True,
            )
            self.torch_profiler_summary = {
                "enabled": True, "captured_step": step,
                # key_averages already collapses individual events. Retain all
                # rows so CUDA API/kernel counts cannot disappear behind an
                # arbitrary top-N cutoff.
                "operators": operators, "error": None,
            }
            if self.profiler_trace_path is not None:
                raw_path = self.profiler_trace_path.with_suffix("")
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                profile.export_chrome_trace(str(raw_path))
                with raw_path.open("rb") as source, gzip.open(
                        self.profiler_trace_path, "wb", compresslevel=6) as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                raw_path.unlink()
        except Exception as exc:
            self.torch_profiler_summary = {
                "enabled": True, "captured_step": step,
                "operators": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            raise

    def end_step(
            self, step: int, phase: str, model_seconds: float,
            full_token_seconds: float | None = None) -> None:
        ended = time.perf_counter_ns()
        row = self.step_records[-1]
        row.update({
            "end_monotonic_ns": ended,
            "model_wall_ms": model_seconds * 1000.0,
            "full_token_wall_ms": (
                (full_token_seconds if full_token_seconds is not None
                 else model_seconds) * 1000.0
            ),
        })
        overhead_begin = time.perf_counter_ns()
        snapshots = []
        for layer, engine in enumerate(self._engines):
            profile = json.loads(engine.external_profile_json(model_seconds * 1000.0))
            snapshots.append({"layer": layer, "profile": profile})
        self.profile_snapshots.append({
            "step": int(step), "phase": phase, "layers": snapshots,
        })
        self._overhead_ns += time.perf_counter_ns() - overhead_begin
        if self.memory_probe is not None:
            self.memory_probe.checkpoint(
                f"after_{phase}_step_{step}", include_smaps=False,
                include_cuda=True, include_nvml=True,
                metadata={"step": step, "phase": phase,
                          "model_wall_ms": model_seconds * 1000.0,
                          "full_token_wall_ms": (
                              (full_token_seconds if full_token_seconds is not None
                               else model_seconds) * 1000.0
                          )},
            )

    def record_routes(self, layer: int, step: int, phase: str,
                      experts: Any, weights: Any) -> None:
        began = time.perf_counter_ns()
        expert_rows = experts.detach().cpu().tolist()
        weight_rows = weights.detach().float().cpu().tolist()
        for sequence_token, (expert_row, weight_row) in enumerate(
                zip(expert_rows, weight_rows)):
            for rank, (expert, weight) in enumerate(zip(expert_row, weight_row)):
                self.route_selections.append({
                    "event": "route_selection",
                    "step": int(step),
                    "phase": phase,
                    "sequence_token": int(sequence_token),
                    "layer": int(layer),
                    "gpu": self.device_for_layer(layer),
                    "expert": int(expert),
                    "routing_rank": int(rank),
                    "routing_weight": float(weight),
                })
        self._overhead_ns += time.perf_counter_ns() - began

    def _tensor_device(self, args: tuple[Any, ...], output: Any, layer: int) -> int:
        for value in list(args) + [output]:
            if isinstance(value, (tuple, list)):
                candidates = value
            else:
                candidates = (value,)
            for candidate in candidates:
                device = getattr(candidate, "device", None)
                if device is not None and getattr(device, "type", None) == "cuda":
                    return int(device.index or 0)
        return self.device_for_layer(layer)

    def _hook_begin(self, module: Any, args: tuple[Any, ...], name: str,
                    layer: int, context: Any) -> None:
        item: dict[str, Any] = {
            "name": name,
            "layer": layer,
            "step": int(context.current_step),
            "phase": context.current_phase,
            "cpu_begin_ns": time.perf_counter_ns(),
            "gpu": self.device_for_layer(layer),
        }
        if self.torch.cuda.is_available():
            try:
                device = item["gpu"]
                with self.torch.cuda.device(device):
                    event = self.torch.cuda.Event(enable_timing=True)
                    event.record(self.torch.cuda.current_stream(device))
                item["cuda_begin"] = event
            except Exception as exc:
                item["cuda_error"] = f"{type(exc).__name__}: {exc}"
        self._hook_starts[id(module)].append(item)

    def _hook_end(self, module: Any, args: tuple[Any, ...], output: Any,
                  context: Any) -> None:
        if not self._hook_starts[id(module)]:
            return
        item = self._hook_starts[id(module)].pop()
        item["cpu_end_ns"] = time.perf_counter_ns()
        item["cpu_wall_ms"] = (
            item["cpu_end_ns"] - item["cpu_begin_ns"]
        ) / 1_000_000.0
        if "cuda_begin" in item:
            try:
                device = item["gpu"]
                with self.torch.cuda.device(device):
                    end = self.torch.cuda.Event(enable_timing=True)
                    end.record(self.torch.cuda.current_stream(device))
                item["cuda_end"] = end
                self._pending_cuda.append(item)
            except Exception as exc:
                item["cuda_error"] = f"{type(exc).__name__}: {exc}"
                self.wall_spans.append(self._clean_hook_item(item))
        else:
            self.wall_spans.append(self._clean_hook_item(item))

    @staticmethod
    def _clean_hook_item(item: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in item.items()
                if key not in {"cuda_begin", "cuda_end"}}

    def _register(self, module: Any, name: str, layer: int, context: Any) -> None:
        if module is None:
            return
        self._hook_handles.append(module.register_forward_pre_hook(
            lambda mod, args, name=name, layer=layer:
                self._hook_begin(mod, args, name, layer, context)
        ))
        self._hook_handles.append(module.register_forward_hook(
            lambda mod, args, output:
                self._hook_end(mod, args, output, context)
        ))

    def attach_model_hooks(self, model: Any, context: Any) -> None:
        for layer, block in enumerate(model.model.layers):
            self._register(block, "layer_total", layer, context)
            self._register(getattr(block, "input_layernorm", None),
                           "normalization_input", layer, context)
            # Qwen3.5-MoE exposes the 30 recurrent mixers as ``linear_attn``
            # and only the 10 quadratic mixers as ``self_attn``. The original
            # M2.5 hook only checked self_attn, so most of the reported
            # "residual and unattributed" time included the entire recurrent
            # mixer. Keep the compatibility aggregate name while attaching it
            # to the module that actually exists.
            linear_attn = getattr(block, "linear_attn", None)
            self_attn = getattr(block, "self_attn", None)
            mixer = linear_attn if linear_attn is not None else self_attn
            self._register(mixer, "attention_or_linear_attention", layer, context)
            if linear_attn is not None:
                for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
                    self._register(
                        getattr(linear_attn, name, None),
                        f"linear_attention_{name}",
                        layer,
                        context,
                    )
                self._register(getattr(linear_attn, "conv1d", None),
                               "linear_attention_convolution", layer, context)
                self._register(getattr(linear_attn, "norm", None),
                               "linear_attention_gated_norm", layer, context)
                self._register(getattr(linear_attn, "out_proj", None),
                               "linear_attention_out_proj", layer, context)
            if self_attn is not None:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    self._register(
                        getattr(self_attn, name, None),
                        f"full_attention_{name}",
                        layer,
                        context,
                    )
                self._register(getattr(self_attn, "q_norm", None),
                               "full_attention_q_norm", layer, context)
                self._register(getattr(self_attn, "k_norm", None),
                               "full_attention_k_norm", layer, context)
            self._register(getattr(block, "post_attention_layernorm", None),
                           "normalization_post_attention", layer, context)
            mlp = getattr(block, "mlp", None)
            self._register(mlp, "moe_block_total", layer, context)
            self._register(getattr(mlp, "gate", None),
                           "router_module_total", layer, context)
            self._register(getattr(mlp, "experts", None),
                           "routed_experts_total", layer, context)
            self._register(getattr(mlp, "shared_expert", None),
                           "shared_expert", layer, context)
            self._register(getattr(mlp, "shared_expert_gate", None),
                           "shared_expert_gate", layer, context)
            shared = getattr(mlp, "shared_expert", None)
            for name in ("gate_proj", "up_proj", "down_proj"):
                self._register(
                    getattr(shared, name, None),
                    f"shared_expert_{name}",
                    layer,
                    context,
                )
        self._register(getattr(model.model, "embed_tokens", None),
                       "embedding", 0, context)
        self._register(getattr(model.model, "norm", None),
                       "final_normalization", len(model.model.layers) - 1, context)
        self._register(getattr(model, "lm_head", None),
                       "lm_head", len(model.model.layers) - 1, context)

    def resolve_cuda_events(self) -> None:
        for device in range(self.gpu_count):
            self.torch.cuda.synchronize(device)
        for item in self._pending_cuda:
            try:
                item["cuda_event_ms"] = float(
                    item["cuda_begin"].elapsed_time(item["cuda_end"])
                )
            except Exception as exc:
                item["cuda_error"] = f"{type(exc).__name__}: {exc}"
            self.wall_spans.append(self._clean_hook_item(item))
        self._pending_cuda.clear()

    def capture_final_engine_evidence(self, total_wall_ms: float) -> None:
        self.engine_profiles.clear()
        self.engine_timelines.clear()
        for layer, engine in enumerate(self._engines):
            self.engine_profiles.append({
                "layer": layer,
                "gpu": self.device_for_layer(layer),
                "profile": json.loads(engine.external_profile_json(total_wall_ms)),
            })
            self.engine_timelines.append({
                "layer": layer,
                "gpu": self.device_for_layer(layer),
                "timeline": json.loads(engine.external_timeline_json(total_wall_ms)),
            })

    def stop(self) -> dict[str, Any]:
        self.resolve_cuda_events()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._engines.clear()
        nvml = self.nvml.stop() if self.nvml is not None else {
            "interval_seconds": None, "error": "disabled", "samples": []
        }
        return {
            "phase_timings": self.phase_timings,
            "wall_spans": self.wall_spans,
            "route_selections": self.route_selections,
            "steps": self.step_records,
            "profile_snapshots": self.profile_snapshots,
            "engine_profiles": self.engine_profiles,
            "engine_timelines": self.engine_timelines,
            "nvml": nvml,
            "torch_profiler": self.torch_profiler_summary,
            "instrumentation_bookkeeping_ms": self._overhead_ns / 1_000_000.0,
        }


def summarize_nvml(samples: list[dict[str, Any]], gpu_count: int) -> dict[str, Any]:
    by_gpu: dict[str, Any] = {}
    for gpu in range(gpu_count):
        rows = [sample["gpus"][gpu] for sample in samples
                if gpu < len(sample.get("gpus", []))]
        util = [float(row["gpu_utilization_percent"]) for row in rows
                if "gpu_utilization_percent" in row]
        memory_util = [float(row["memory_controller_utilization_percent"])
                       for row in rows
                       if "memory_controller_utilization_percent" in row]
        used = [int(row["memory_used_bytes"]) for row in rows
                if "memory_used_bytes" in row]
        pcie_tx = [int(row["pcie_tx_bytes_per_second"]) for row in rows
                   if "pcie_tx_bytes_per_second" in row]
        pcie_rx = [int(row["pcie_rx_bytes_per_second"]) for row in rows
                   if "pcie_rx_bytes_per_second" in row]
        by_gpu[f"cuda:{gpu}"] = {
            "sample_count": len(rows),
            "gpu_utilization_mean_percent": statistics.fmean(util) if util else None,
            "gpu_utilization_p95_percent": _percentile(util, 0.95) if util else None,
            "gpu_utilization_max_percent": max(util) if util else None,
            "memory_controller_mean_percent": (
                statistics.fmean(memory_util) if memory_util else None
            ),
            "peak_nvml_used_bytes": max(used) if used else None,
            "pcie_tx_peak_bytes_per_second": max(pcie_tx) if pcie_tx else None,
            "pcie_rx_peak_bytes_per_second": max(pcie_rx) if pcie_rx else None,
        }
    return by_gpu


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
