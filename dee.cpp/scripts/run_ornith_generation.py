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
        return {
            "peak_host_rss_bytes": self.peak_host_rss,
            "peak_vram_bytes": {f"cuda:{i}": value for i, value in enumerate(self.peak_vram)},
        }


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
        self.collector: TraceCollector | None = None
        self.router_tie_fallback_rows = 0
        self.forensics = None
        self.current_step = -1
        self.current_phase = "unclassified"
        self.cache_disabled = False
        self.executed_router_layers: set[int] = set()


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
        if self.context.mode == "reference":
            with forensic_span(self.context, "reference_router", self.layer):
                logits, weights, experts = self.module.reference(flattened)
        else:
            if hasattr(self.engine, "set_external_token"):
                self.engine.set_external_token(self.context.current_step)
            router_input_transfer = {
                "direction": "d2h",
                "bytes": int(flattened.numel()) * np.dtype(np.float32).itemsize,
                "component": "router_input",
            }
            with forensic_span(self.context, "router_hidden_gpu_to_cpu", self.layer,
                               router_input_transfer):
                flattened_cpu = flattened.detach().float().cpu().numpy()
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
        self.module.forward = self.forward
        self.module.clear_cache = reference.clear_cache

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.context.mode == "reference":
            return self.module.reference(hidden_states, top_k_index, top_k_weights)
        import torch

        output = torch.zeros_like(hidden_states)
        if self.context.cache_disabled and not self.engine.reset_runtime_cache():
            raise RuntimeError(f"failed to reset diagnostic cache at layer {self.layer}")
        if hasattr(self.engine, "set_external_token"):
            self.engine.set_external_token(self.context.current_step)
        expert_input_transfer = {
            "direction": "d2h",
            "component": "expert_input",
            "bytes": int(top_k_index.numel()) * int(top_k_index.element_size()) +
                     int(hidden_states.numel()) * np.dtype(np.float32).itemsize,
        }
        with forensic_span(self.context, "expert_inputs_gpu_to_cpu", self.layer,
                           expert_input_transfer):
            expert_ids = top_k_index.detach().cpu().numpy().astype(np.int32)
            hidden_cpu = hidden_states.detach().float().cpu().numpy()
        with forensic_span(self.context, "expert_native", self.layer):
            expert_outputs = self.engine.moe_forward_batch(
                self.layer, hidden_cpu, expert_ids,
            )
        expert_output_transfer = {
            "direction": "h2d", "component": "expert_output",
            "bytes": int(np.asarray(expert_outputs).nbytes),
        }
        with forensic_span(self.context, "expert_outputs_cpu_to_gpu", self.layer,
                           expert_output_transfer):
            raw = torch.from_numpy(np.asarray(expert_outputs)).to(
                hidden_states.device, dtype=hidden_states.dtype
            )
        if tuple(raw.shape) != (
            hidden_states.shape[0], top_k_index.shape[1], hidden_states.shape[1]
        ):
            raise RuntimeError(
                f"dee.cpp batched expert output shape mismatch at layer {self.layer}: "
                f"{tuple(raw.shape)}"
            )
        with forensic_span(self.context, "expert_output_combination", self.layer):
            for token in range(hidden_states.shape[0]):
                # Official eager implementation accumulates contributions in
                # ascending expert id order, not router top-k order.
                accumulator = torch.zeros_like(hidden_states[token])
                for position in np.argsort(expert_ids[token], kind="stable").tolist():
                    accumulator.add_(raw[token, position] * top_k_weights[token, position])
                output[token] = accumulator
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
                 allow_diagnostic_sub_topk_cache: bool = False,
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
    record_elapsed(phase_recorder, "initial_mmap_and_engine_setup", mmap_started,
                   engines=len(engines), cache_experts=cache_experts)

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
                   chat_template: bool, mode: str, trace: bool):
    import torch

    model = runtime["model"]
    config = runtime["config"]
    context = runtime["context"]
    gpu_count = torch.cuda.device_count()
    context.mode = mode
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
    attention_mask = torch.ones_like(prompt_ids, device="cuda:0")
    current_input = prompt_ids
    past_key_values = None
    generated: list[int] = []
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
            with forensic_span(context, "step_pre_synchronize", -1,
                               {"barrier": "all_visible_gpus"}):
                synchronize_all(torch, gpu_count)
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
            with forensic_span(context, "step_post_synchronize", -1,
                               {"barrier": "all_visible_gpus"}):
                synchronize_all(torch, gpu_count)
            elapsed = time.perf_counter() - step_start
            step_seconds.append(elapsed)
            if context.forensics is not None:
                context.forensics.end_step(step, context.current_phase, elapsed)
            logits = outputs.logits[:, -1, :]
            if collector is not None:
                collector.add("lm_head_logits", f"step={step}", logits)
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
            if next_token in eos_ids:
                break
            current_input = torch.tensor([[next_token]], dtype=torch.long, device="cuda:0")
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device="cuda:0")],
                dim=1,
            )
    synchronize_all(torch, gpu_count)
    total_seconds = time.perf_counter() - total_start
    resources = monitor.stop()
    context.collector = None
    prompt_tokens = int(prompt_ids.shape[1])
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
        "evictions", "h2d_bytes", "h2d_copies", "peak_vram", "current_vram",
        "resident_experts", "host_pinned_expert_staging_bytes",
        "host_pageable_expert_staging_bytes", "host_router_weight_bytes",
        "host_hidden_buffer_bytes", "host_prefetch_ring_bytes",
        "host_prefetch_ring_slots", "peak_transient_host_bytes",
        "device_expert_cache_reserved_bytes", "device_prefetch_staging_bytes",
        "device_fixed_work_buffer_bytes", "device_router_weight_bytes",
        "device_router_dynamic_bytes", "device_moe_batch_buffer_bytes",
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
    parser.add_argument("--reference-parity", action="store_true")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--require-dual-gpu", action="store_true")
    parser.add_argument("--split-layer", type=int, default=20)
    parser.add_argument("--cache-experts", type=int, default=8)
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
        args.model_dir, effective_gpu_count, args.split_layer, args.cache_experts
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
        )
        after_cold = engine_stats(runtime)
        warm = run_generation(
            runtime, tokenizer, args.prompt[0], args.max_new_tokens,
            args.chat_template, "dee", trace=False,
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
            )
            for expert in runtime["reference_experts"]:
                expert.clear_cache()
            torch.cuda.empty_cache()
        candidate = run_generation(
            runtime, tokenizer, prompt, args.max_new_tokens,
            args.chat_template, "dee", trace=args.reference_parity,
        )
        candidate_outputs.append(serializable_generation(candidate))
        item = {
            "prompt": prompt,
            "candidate": serializable_generation(candidate),
            "all_40_layers_executed": (
                candidate["collector"] is not None and
                candidate["collector"].router_layers == set(range(40))
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
