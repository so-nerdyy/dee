"""Pure-Python checkpoint, placement, decoding, and comparison helpers."""

from __future__ import annotations

import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any, Callable, Iterable


DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def read_checkpoint_index(model_dir: os.PathLike[str] | str) -> dict[str, Any]:
    path = Path(model_dir) / "model.safetensors.index.json"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint index is missing: {path}")
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed checkpoint index {path}: {exc}") from exc
    if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        raise ValueError(f"checkpoint index has no non-empty weight_map: {path}")
    return index


def layer_from_tensor_name(name: str) -> int | None:
    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def component_for_tensor(name: str) -> str:
    if ".visual." in name:
        return "vision_not_used_for_text"
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp_not_used_for_greedy_text"
    if ".mlp.experts." in name:
        return "routed_expert"
    if name.endswith(".mlp.gate.weight"):
        return "router"
    if ".mlp.shared_expert" in name:
        return "shared_expert"
    if ".self_attn." in name:
        return "full_attention"
    if ".linear_attn." in name:
        return "linear_attention"
    if "layernorm" in name or name.endswith(".norm.weight"):
        return "normalization"
    if name.endswith("embed_tokens.weight"):
        return "token_embedding"
    if name == "lm_head.weight":
        return "lm_head"
    return "text_dense_or_metadata"


def target_device_for_tensor(name: str, gpu_count: int = 2, split_layer: int = 20) -> str:
    component = component_for_tensor(name)
    if component.endswith("not_used_for_text") or component.startswith("mtp_not_used"):
        return "not_loaded"
    if gpu_count <= 0:
        return "cpu"
    if name.endswith("embed_tokens.weight"):
        return "cuda:0"
    if name == "lm_head.weight" or name == "model.language_model.norm.weight":
        return f"cuda:{min(gpu_count - 1, 1)}"
    layer = layer_from_tensor_name(name)
    if layer is None or gpu_count == 1:
        return "cuda:0"
    return "cuda:0" if layer < split_layer else "cuda:1"


def expected_lifetime(name: str) -> str:
    component = component_for_tensor(name)
    if component == "routed_expert":
        return "mmap_checkpoint_plus_bounded_gpu_lru"
    if component in {"vision_not_used_for_text", "mtp_not_used_for_greedy_text"}:
        return "checkpoint_only_not_loaded"
    return "generation_resident"


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint shard is missing: {path}")
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 0 or header_size > file_size - 8:
            raise ValueError(f"invalid safetensors header length in {path}: {header_size}")
        encoded = stream.read(header_size)
    try:
        header = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed safetensors header {path}: {exc}") from exc
    return header, 8 + header_size, file_size


def build_complete_tensor_map(
    model_dir: os.PathLike[str] | str,
    gpu_count: int = 2,
    split_layer: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every indexed tensor against shard headers and describe runtime placement."""
    root = Path(model_dir)
    index = read_checkpoint_index(root)
    weight_map: dict[str, str] = index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    rows: list[dict[str, Any]] = []
    total_tensor_bytes = 0
    shard_summaries: list[dict[str, Any]] = []
    for shard_name in sorted(by_shard):
        path = root / shard_name
        header, data_start, file_size = _read_safetensors_header(path)
        actual_names = {name for name in header if name != "__metadata__"}
        expected_names = set(by_shard[shard_name])
        missing = sorted(expected_names - actual_names)
        if missing:
            raise ValueError(f"{shard_name} is missing indexed tensor {missing[0]}")
        shard_bytes = 0
        for name in sorted(expected_names):
            metadata = header[name]
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if dtype not in DTYPE_BYTES or not isinstance(shape, list) or len(offsets or []) != 2:
                raise ValueError(f"invalid tensor metadata for {name} in {shard_name}")
            start, end = map(int, offsets)
            if start < 0 or end < start or data_start + end > file_size:
                raise ValueError(f"out-of-bounds tensor {name} in {shard_name}")
            elements = math.prod(int(dim) for dim in shape)
            expected_bytes = elements * DTYPE_BYTES[dtype]
            if end - start != expected_bytes:
                raise ValueError(
                    f"byte-size mismatch for {name}: offsets={end-start}, shape/dtype={expected_bytes}"
                )
            shard_bytes += expected_bytes
            total_tensor_bytes += expected_bytes
            rows.append({
                "tensor_name": name,
                "source_shard": shard_name,
                "shape": shape,
                "dtype": dtype,
                "byte_size": expected_bytes,
                "target_runtime_component": component_for_tensor(name),
                "target_device": target_device_for_tensor(name, gpu_count, split_layer),
                "expected_lifetime": expected_lifetime(name),
            })
        shard_summaries.append({
            "shard": shard_name,
            "file_size": file_size,
            "tensor_count": len(expected_names),
            "tensor_bytes": shard_bytes,
        })

    declared_size = index.get("metadata", {}).get("total_size")
    if declared_size is not None and int(declared_size) != total_tensor_bytes:
        raise ValueError(
            f"index total_size={declared_size} but validated tensors total {total_tensor_bytes}"
        )
    summary = {
        "tensor_count": len(rows),
        "shard_count": len(by_shard),
        "validated_tensor_bytes": total_tensor_bytes,
        "declared_total_size": declared_size,
        "shards": shard_summaries,
    }
    return rows, summary


def shard_paths_for_layer(index: dict[str, Any], layer: int) -> list[str]:
    prefix = f"model.language_model.layers.{layer}.mlp."
    paths = {
        shard for name, shard in index["weight_map"].items()
        if name.startswith(prefix) and (".experts." in name or name.endswith("gate.weight"))
    }
    if not paths:
        raise ValueError(f"checkpoint index has no routed MoE tensors for layer {layer}")
    return sorted(paths)


def layer_device(layer: int, gpu_count: int = 2, split_layer: int = 20) -> int:
    if gpu_count <= 0:
        raise ValueError("at least one GPU is required for the real runtime")
    return 0 if gpu_count == 1 or layer < split_layer else 1


def validate_expert_cache_budget(
    budget_bytes: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    bytes_per_weight: int = 2,
) -> int:
    expert_bytes = 3 * hidden_size * intermediate_size * bytes_per_weight
    if budget_bytes < expert_bytes:
        raise ValueError(
            f"expert cache budget {budget_bytes} cannot hold one {expert_bytes}-byte expert"
        )
    capacity = budget_bytes // expert_bytes
    if capacity < top_k:
        raise ValueError(
            f"expert cache capacity {capacity} is smaller than top_k={top_k}; "
            "real-token execution requires one activation set"
        )
    return capacity


def greedy_token(logits: Iterable[float]) -> int:
    values = list(logits)
    if not values:
        raise ValueError("cannot select a token from empty logits")
    return max(range(len(values)), key=values.__getitem__)


def causal_position_ids(past_seen_tokens: int, input_length: int) -> list[int]:
    if past_seen_tokens < 0 or input_length < 0:
        raise ValueError("causal position arguments must be non-negative")
    return list(range(past_seen_tokens, past_seen_tokens + input_length))


def advance_decode_state(state: dict[str, Any], token: int) -> dict[str, Any]:
    """Pure state transition mirrored by the real DynamicCache decode loop."""
    updated = {
        "all_ids": list(state.get("all_ids", [])),
        "attention_length": int(state.get("attention_length", 0)),
        "cache_seen_tokens": int(state.get("cache_seen_tokens", 0)),
    }
    if updated["attention_length"] != len(updated["all_ids"]):
        raise ValueError("attention length does not match token history")
    updated["all_ids"].append(int(token))
    updated["attention_length"] += 1
    updated["cache_seen_tokens"] += 1
    return updated


def tiny_greedy_decode(
    prompt_ids: list[int],
    step: Callable[[list[int]], Iterable[float]],
    max_new_tokens: int,
    eos_token_ids: set[int] | None = None,
) -> dict[str, Any]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    eos = eos_token_ids or set()
    sequence = list(prompt_ids)
    generated: list[int] = []
    stopped = "max_new_tokens"
    for _ in range(max_new_tokens):
        token = greedy_token(step(sequence))
        generated.append(token)
        sequence.append(token)
        if token in eos:
            stopped = "eos"
            break
    return {"prompt_ids": prompt_ids, "generated_ids": generated,
            "all_ids": sequence, "stop_reason": stopped}


def numeric_comparison(reference: Any, candidate: Any, atol: float, rtol: float) -> dict[str, Any]:
    import numpy as np

    ref = np.asarray(reference)
    got = np.asarray(candidate)
    result: dict[str, Any] = {
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(got.dtype),
        "reference_shape": list(ref.shape),
        "candidate_shape": list(got.shape),
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
    }
    if ref.shape != got.shape:
        result.update({"passed": False, "first_failing_index": "shape"})
        return result
    ref64 = ref.astype(np.float64, copy=False)
    got64 = got.astype(np.float64, copy=False)
    absolute = np.abs(ref64 - got64)
    relative = absolute / np.maximum(np.abs(ref64), 1e-12)
    allowed = atol + rtol * np.abs(ref64)
    failing = np.argwhere(absolute > allowed)
    result.update({
        "maximum_absolute_error": float(absolute.max(initial=0.0)),
        "maximum_relative_error": float(relative.max(initial=0.0)),
        "mean_absolute_error": float(absolute.mean()) if absolute.size else 0.0,
        "passed": failing.size == 0,
        "first_failing_index": failing[0].tolist() if failing.size else None,
    })
    return result
