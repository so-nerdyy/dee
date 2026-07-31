"""Pure-Python DeepSeek-V4-Flash-0731 checkpoint ledger and resolver helpers.

Mirrors the conventions of scripts/ornith_support.py but for the official
deepseek-ai/DeepSeek-V4-Flash-0731 checkpoint layout:

- experts are stored packed as I8 (two FP4 e2m1fn values per byte) with one
  F8_E8M0 scale per 32 input elements;
- dense weights are F8_E4M3 (with F8_E8M0 scales), BF16, or F32;
- hash-layers carry I64 tid2eid routing tables and F32 gates;
- DSpark lives under the ``mtp.*`` namespace.

Nothing here assumes Ornith semantics; every classification is derived from
the official tensor names, shapes, and dtypes observed in the pinned shards.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import urllib.request
from pathlib import Path
from typing import Any

OFFICIAL_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
EXPECTED_SHARD_COUNT = 48
EXPECTED_TENSOR_COUNT = 72317

# safetensors dtype -> bytes per stored element.
DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
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

LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|mtp)\.(\d+)(?:\.|$)")
HASH_TENSOR_MARKERS = ("hc_", "indexer", "compress", "attn_sink", "ape", "weights_proj")
DSPARK_MARKERS = ("dspark", "markov", "confidence", "main_proj", "main_norm")


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


def fetch_shard_header(revision: str, shard_name: str, *,
                       repository: str = OFFICIAL_REPOSITORY,
                       max_attempt_bytes: tuple[int, ...] = (2 << 20, 8 << 20, 32 << 20)) -> dict[str, Any]:
    """Fetch only the safetensors header of one shard via HTTP range request.

    Returns the header JSON (tensor metadata) without downloading weight bytes.
    """
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{shard_name}"
    last_content_length = None
    for size in max_attempt_bytes:
        request = urllib.request.Request(url, headers={
            "Range": f"bytes=0-{size - 1}",
            "User-Agent": "freebuff-deepseek-ledger/1.0",
        })
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                last_content_length = int(content_range.split("/")[1])
        if len(data) < 8:
            raise ValueError(f"truncated safetensors prefix for {shard_name}")
        header_size = struct.unpack("<Q", data[:8])[0]
        if header_size <= 0:
            raise ValueError(f"invalid header length for {shard_name}: {header_size}")
        if header_size <= len(data) - 8:
            try:
                return json.loads(data[8:8 + header_size].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"malformed header for {shard_name}: {exc}") from exc
    raise ValueError(
        f"header for {shard_name} exceeds {max_attempt_bytes[-1]} bytes "
        f"(file size {last_content_length})"
    )


def layer_from_tensor_name(name: str) -> int | None:
    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def module_from_tensor_name(name: str) -> str:
    if name.startswith("embed.") or name == "embed.weight":
        return "embedding"
    if name.startswith("norm."):
        return "final_norm"
    if name.startswith("head.") or name.startswith("lm_head."):
        return "lm_head"
    if ".attn." in name:
        return "attention"
    if ".ffn." in name or ".mlp." in name:
        return "ffn"
    if name.startswith("mtp."):
        return "dspark"
    return "misc"


def component_for_tensor(name: str) -> str:
    if any(marker in name for marker in DSPARK_MARKERS) or name.startswith("mtp."):
        return "dspark"
    if ".ffn.experts." in name or ".mlp.experts." in name:
        return "routed_expert"
    if ".ffn.shared_experts." in name or ".mlp.shared_experts." in name:
        return "shared_expert"
    if ".ffn.gate." in name or ".mlp.gate." in name:
        return "router"
    if any(marker in name for marker in HASH_TENSOR_MARKERS):
        return "hash_compress"
    if name.startswith("embed.") or name == "embed.weight":
        return "embedding"
    if name.startswith("head.") or name.startswith("lm_head."):
        return "lm_head"
    if ".attn." in name or name.endswith(".attn_norm.weight") or name.endswith(".ffn_norm.weight") or name.startswith("norm."):
        return "attention_dense"
    return "other"


def is_scale_tensor(name: str) -> bool:
    return name.endswith(".scale")


def scale_for_weight(weight_name: str) -> str:
    """Official convention: weight ``x.y.weight`` pairs with ``x.y.scale``."""
    return weight_name[:-len(".weight")] + ".scale"


def packed_fp4_scale_shape(stored_shape: list[int]) -> list[int] | None:
    """For a packed FP4 expert weight [out, in//2], scale is [out, in//32]."""
    if len(stored_shape) != 2:
        return None
    out, packed_in = stored_shape
    return [out, packed_in // 16]  # in//32 where in = 2*packed_in


def expanded_bytes(shape: list[int], elements_per_byte: int, bytes_per_element: int) -> int:
    stored = math.prod(int(dim) for dim in shape)
    return stored * elements_per_byte * bytes_per_element


def storage_plan_for_tensor(name: str, dtype: str, shape: list[int]) -> dict[str, Any]:
    """Classify one tensor: compressed bytes, expanded FP16/INT8 bytes, scale linkage."""
    element_count = math.prod(int(dim) for dim in shape)
    stored = element_count * DTYPE_BYTES[dtype]
    fp16 = element_count * 2
    int8 = element_count * 1
    is_expert = ".ffn.experts." in name
    if is_expert and name.endswith(".weight") and dtype == "I8":
        # Packed FP4: two e2m1fn values per stored byte.
        return {
            "kind": "packed_fp4_expert",
            "stored_dtype": dtype,
            "packed_values_per_byte": 2,
            "compressed_bytes": stored,
            "expanded_fp16_bytes": expanded_bytes(shape, 2, 2),
            "expanded_int8_bytes": expanded_bytes(shape, 2, 1),
            "scale_tensor": scale_for_weight(name),
            "scale_block_elements": 32,
            "block_shape": packed_fp4_scale_shape(shape),
        }
    if is_expert and name.endswith(".scale"):
        return {
            "kind": "expert_scale_fp8",
            "stored_dtype": dtype,
            "packed_values_per_byte": 1,
            "compressed_bytes": stored,
            "expanded_fp16_bytes": fp16,
            "expanded_int8_bytes": int8,
            "scale_tensor": None,
            "scale_block_elements": 1,
            "block_shape": None,
        }
    if dtype in ("F8_E4M3", "F8_E5M2", "F8_E8M0"):
        # Dense FP8 weights pair with F8_E8M0 scales over [128, 128] blocks,
        # i.e. scale shape is [(out+127)//128, (in+127)//128]. Proven against
        # the pinned shard headers (e.g. wkv.weight [512,4096] -> scale [4,32]).
        block_shape = None
        if name.endswith(".weight") and len(shape) == 2:
            out, in_ = (int(dim) for dim in shape)
            block_shape = [(out + 127) // 128, (in_ + 127) // 128]
        return {
            "kind": "fp8_tensor",
            "stored_dtype": dtype,
            "packed_values_per_byte": 1,
            "compressed_bytes": stored,
            "expanded_fp16_bytes": fp16,
            "expanded_int8_bytes": int8,
            "scale_tensor": scale_for_weight(name) if name.endswith(".weight") else None,
            # Nominal 128x128 block size; edge blocks cover fewer elements
            # when out/in are not exact multiples of 128.
            "scale_block_elements": 16384,
            "block_shape": block_shape,
        }
    return {
        "kind": "plain_tensor",
        "stored_dtype": dtype,
        "packed_values_per_byte": 1,
        "compressed_bytes": stored,
        "expanded_fp16_bytes": fp16,
        "expanded_int8_bytes": int8,
        "scale_tensor": None,
        "scale_block_elements": 1,
        "block_shape": None,
    }


def build_complete_tensor_ledger(
    index: dict[str, Any],
    headers: dict[str, dict[str, Any]],
    *,
    expected_shard_count: int = EXPECTED_SHARD_COUNT,
    expected_tensor_count: int = EXPECTED_TENSOR_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the index against every shard header and describe each tensor.

    headers maps shard name -> full safetensors header JSON.
    """
    weight_map: dict[str, str] = index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    if len(by_shard) != expected_shard_count:
        raise ValueError(
            f"index references {len(by_shard)} shards, expected {expected_shard_count}"
        )
    missing_shards = sorted(set(by_shard) - set(headers))
    if missing_shards:
        raise ValueError(f"missing shard headers: {missing_shards}")

    known_names = set(weight_map)
    rows: list[dict[str, Any]] = []
    total_compressed = 0
    total_fp16 = 0
    total_int8 = 0
    shard_summaries: list[dict[str, Any]] = []
    for shard_name in sorted(by_shard):
        header = headers[shard_name]
        actual_names = {name for name in header if name != "__metadata__"}
        expected_names = set(by_shard[shard_name])
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing:
            raise ValueError(f"{shard_name} missing indexed tensor {missing[0]}")
        if unexpected:
            raise ValueError(f"{shard_name} has unexpected tensor {unexpected[0]}")
        shard_compressed = 0
        for name in sorted(expected_names):
            metadata = header[name]
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if dtype not in DTYPE_BYTES or not isinstance(shape, list) or len(offsets or []) != 2:
                raise ValueError(f"invalid tensor metadata for {name} in {shard_name}")
            start, end = map(int, offsets)
            if start < 0 or end < start:
                raise ValueError(f"invalid offsets for {name} in {shard_name}")
            plan = storage_plan_for_tensor(name, dtype, shape)
            layer = layer_from_tensor_name(name)
            component = component_for_tensor(name)
            scale = plan["scale_tensor"]
            scale_shard = None
            scale_shape = None
            scale_dtype = None
            if scale is not None:
                if scale not in known_names:
                    raise ValueError(f"{name} references missing scale tensor {scale}")
                scale_shard = weight_map[scale]
                scale_header = headers[scale_shard]
                if scale in scale_header:
                    scale_shape = scale_header[scale].get("shape")
                    scale_dtype = scale_header[scale].get("dtype")
                if plan["block_shape"] is not None and scale_shape != plan["block_shape"]:
                    raise ValueError(
                        f"{name} scale shape {scale_shape} != declared block shape "
                        f"{plan['block_shape']} for {scale}"
                    )
            rows.append({
                "tensor_name": name,
                "layer": layer,
                "module": module_from_tensor_name(name),
                "component": component,
                "source_shard": shard_name,
                "shape": shape,
                "stored_dtype": dtype,
                "byte_offset": start,
                "byte_length": end - start,
                "compressed_bytes": plan["compressed_bytes"],
                "expanded_fp16_bytes": plan["expanded_fp16_bytes"],
                "expanded_int8_bytes": plan["expanded_int8_bytes"],
                "kind": plan["kind"],
                "scale_tensor": scale,
                "scale_tensor_shard": scale_shard,
                "scale_tensor_shape": scale_shape,
                "scale_tensor_dtype": scale_dtype,
                "scale_block_elements": plan["scale_block_elements"],
                "block_shape": plan["block_shape"],
                "persistent": component not in {"routed_expert"},
                "routed": component == "routed_expert",
                "shared": component == "shared_expert",
                "cache_class": _cache_class(component),
                "active_every_token": _active_every_token(component),
            })
            if plan["compressed_bytes"] != end - start:
                raise ValueError(
                    f"byte-size mismatch for {name}: offsets={end-start}, dtype/shape={plan['compressed_bytes']}"
                )
            shard_compressed += plan["compressed_bytes"]
            total_compressed += plan["compressed_bytes"]
            total_fp16 += plan["expanded_fp16_bytes"]
            total_int8 += plan["expanded_int8_bytes"]
        shard_summaries.append({
            "shard": shard_name,
            "tensor_count": len(expected_names),
            "compressed_bytes": shard_compressed,
        })

    if len(rows) != expected_tensor_count:
        raise ValueError(f"expected {expected_tensor_count} tensors, found {len(rows)}")
    declared = index.get("metadata", {}).get("total_size")
    if declared is not None and int(declared) != total_compressed:
        raise ValueError(
            f"index total_size={declared} but validated tensors total {total_compressed}"
        )

    summary = {
        "tensor_count": len(rows),
        "shard_count": len(by_shard),
        "compressed_bytes": total_compressed,
        "expanded_fp16_bytes": total_fp16,
        "expanded_int8_bytes": total_int8,
        "declared_total_size": declared,
        "shards": shard_summaries,
    }
    return rows, summary


def _cache_class(component: str) -> str:
    if component == "routed_expert":
        return "mmap_checkpoint_plus_bounded_gpu_lru"
    if component == "dspark":
        return "speculation_only_optional"
    return "generation_resident"


def _active_every_token(component: str) -> bool:
    return component not in {"routed_expert", "dspark"}


def aggregate_by_component(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        component = row["component"]
        bucket = result.setdefault(component, {
            "tensor_count": 0,
            "compressed_bytes": 0,
            "expanded_fp16_bytes": 0,
            "expanded_int8_bytes": 0,
        })
        bucket["tensor_count"] += 1
        bucket["compressed_bytes"] += row["compressed_bytes"]
        bucket["expanded_fp16_bytes"] += row["expanded_fp16_bytes"]
        bucket["expanded_int8_bytes"] += row["expanded_int8_bytes"]
    return result


# ---------------------------------------------------------------------------
# DS8: generalized official expert resolution.
#
# Every MoE layer's expert tensors live entirely inside ONE shard
# (layer N -> model-0000(N+2)-of-00048, verified against the ledger), and
# every weight/scale pair is co-located in the same shard (0 cross-shard
# mismatches).  These helpers resolve the exact tensor names, shard, byte
# offsets, dtypes, shapes and scale linkage for ARBITRARY (layer, expert)
# routed experts and for the per-layer shared expert, straight from the
# validated ledger rows -- no hard-coded layer-6 / expert-0 assumptions.
# ---------------------------------------------------------------------------

ROUTED_PROJECTION_NAMES = ("w1", "w2", "w3")  # gate, down, up (official order)


def routed_expert_tensor_names(layer: int, expert_id: int) -> list[str]:
    """All 6 routed-expert tensor names (w1/w2/w3 weights + scales)."""
    return [
        f"layers.{layer}.ffn.experts.{expert_id}.{proj}.{kind}"
        for proj in ROUTED_PROJECTION_NAMES
        for kind in ("weight", "scale")
    ]


def shared_expert_tensor_names(layer: int) -> list[str]:
    """All 6 shared-expert tensor names (w1/w2/w3 weights + scales)."""
    return [
        f"layers.{layer}.ffn.shared_experts.{proj}.{kind}"
        for proj in ROUTED_PROJECTION_NAMES
        for kind in ("weight", "scale")
    ]


def router_tensor_names(layer: int, *, hash_layer: bool = False) -> list[str]:
    """Score-based layers use gate.weight + gate.bias; hash layers use tid2eid."""
    if hash_layer:
        return [f"layers.{layer}.ffn.gate.tid2eid"]
    return [f"layers.{layer}.ffn.gate.weight", f"layers.{layer}.ffn.gate.bias"]


def index_ledger_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build name -> row lookup from a validated ledger rows list."""
    return {row["tensor_name"]: row for row in rows}


def resolve_expert_tensors(
    by_name: dict[str, dict[str, Any]],
    layer: int,
    expert_id: int,
) -> dict[str, dict[str, Any]]:
    """Resolve all 6 routed-expert tensor rows for arbitrary (layer, expert).

    Fails closed on any missing tensor or cross-shard weight/scale split.
    Returns {tensor_name: ledger_row}.
    """
    names = routed_expert_tensor_names(layer, expert_id)
    resolved: dict[str, dict[str, Any]] = {}
    for name in names:
        row = by_name.get(name)
        if row is None:
            raise ValueError(f"routed expert tensor missing from ledger: {name}")
        resolved[name] = row
    _assert_same_shard(resolved)
    return resolved


def resolve_shared_expert_tensors(
    by_name: dict[str, dict[str, Any]],
    layer: int,
) -> dict[str, dict[str, Any]]:
    """Resolve all 6 shared-expert tensor rows for a layer."""
    names = shared_expert_tensor_names(layer)
    resolved: dict[str, dict[str, Any]] = {}
    for name in names:
        row = by_name.get(name)
        if row is None:
            raise ValueError(f"shared expert tensor missing from ledger: {name}")
        resolved[name] = row
    _assert_same_shard(resolved)
    return resolved


def resolve_router_tensors(
    by_name: dict[str, dict[str, Any]],
    layer: int,
    *,
    hash_layers: int = 3,
) -> dict[str, dict[str, Any]]:
    """Resolve router tensors for a layer (score-based vs hash)."""
    hash_layer = layer < hash_layers
    names = router_tensor_names(layer, hash_layer=hash_layer)
    resolved: dict[str, dict[str, Any]] = {}
    for name in names:
        row = by_name.get(name)
        if row is None:
            raise ValueError(f"router tensor missing from ledger: {name}")
        resolved[name] = row
    _assert_same_shard(resolved)
    return resolved


def _assert_same_shard(resolved: dict[str, dict[str, Any]]) -> None:
    shards = {row["source_shard"] for row in resolved.values()}
    if len(shards) != 1:
        raise ValueError(
            f"tensors span {len(shards)} shards; expected exactly one: "
            f"{sorted(shards)}"
        )


def expert_load_plan(
    resolved: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compact per-expert byte/offset plan for a bounded host staging loader.

    Records, per tensor: name, shard, byte offset/length, storage dtype, shape.
    Also aggregates the packed weight bytes + scale bytes for cache budgeting.
    """
    tensors: list[dict[str, Any]] = []
    weight_bytes = 0
    scale_bytes = 0
    for name in sorted(resolved):
        row = resolved[name]
        is_scale = name.endswith(".scale")
        tensors.append({
            "tensor_name": name,
            "shard": row["source_shard"],
            "byte_offset": row["byte_offset"],
            "byte_length": row["byte_length"],
            "dtype": row["stored_dtype"],
            "shape": row["shape"],
        })
        if is_scale:
            scale_bytes += row["byte_length"]
        else:
            weight_bytes += row["byte_length"]
    shards = {row["source_shard"] for row in resolved.values()}
    return {
        "tensor_count": len(tensors),
        "shard": sorted(shards)[0],
        "tensors": tensors,
        "packed_weight_bytes": weight_bytes,
        "scale_bytes": scale_bytes,
        "compressed_bytes": weight_bytes + scale_bytes,
    }
