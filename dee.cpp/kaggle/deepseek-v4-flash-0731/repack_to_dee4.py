#!/usr/bin/env python3
"""DEE4 v2 expert-major repack and integrity validation.

DEE4 v1 wrote six projection-major files. That improved sequential benchmark
bandwidth but still required six independent source regions for every routed
expert. DEE4 v2 is the live-serving layout: each ``(layer, expert)`` is one
fixed-stride contiguous record containing the checkpoint bytes verbatim:

    w1.weight | w3.weight | w2.weight | w1.scale | w3.scale | w2.scale

No numerical conversion occurs. The official FP4 E2M1 nibbles and E8M0 scale
bytes are copied exactly from the pinned safetensors checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import struct
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO

OFFICIAL_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
NUM_LAYERS = 43
EXPERTS_PER_LAYER = 256
GROUP_SIZE = 32
DATA_FILE = "experts.dee4"
INTEGRITY_FILE = "integrity.jsonl"

# Engine projection order: gate/w1, up/w3, down/w2. All weights precede all
# scales so the record is already in the native FP4 staging/cache layout.
PROJECTIONS = ("w1", "w3", "w2")
COMPONENTS = tuple((projection, "weight") for projection in PROJECTIONS) + tuple(
    (projection, "scale") for projection in PROJECTIONS
)


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read and validate the JSON header from one safetensors shard."""
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise IOError(f"short safetensors prefix: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length in {path}")
        raw_header = handle.read(header_length)
    return json.loads(raw_header)


def read_tensor_bytes(
    path: Path, header: dict[str, Any], tensor_name: str
) -> bytes:
    """Read one tensor's exact storage bytes without dtype conversion."""
    if tensor_name not in header:
        raise KeyError(f"tensor {tensor_name!r} not in {path.name}")
    meta = header[tensor_name]
    start, end = (int(value) for value in meta["data_offsets"])
    if start < 0 or end < start:
        raise ValueError(f"invalid data offsets for {tensor_name}")
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_length + start)
        data = handle.read(end - start)
    if len(data) != end - start:
        raise IOError(
            f"short read for {tensor_name}: {len(data)} != {end - start}"
        )
    return data


def _tensor_name(layer: int, expert: int, projection: str, kind: str) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{projection}.{kind}"


def _load_checkpoint_index(
    safetensors_dir: Path, index_path: Path | None
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Path]]:
    index_path = index_path or safetensors_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"index missing: {index_path}")
    index = json.loads(index_path.read_text("utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid weight_map in {index_path}")

    headers: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = safetensors_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"shard missing: {shard_path}")
        paths[shard_name] = shard_path
        headers[shard_name] = read_safetensors_header(shard_path)
    return weight_map, headers, paths


def _component_metadata(
    weight_map: dict[str, str],
    headers: dict[str, dict[str, Any]],
    *,
    layer: int,
    expert: int,
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    offset = 0
    for projection, kind in COMPONENTS:
        name = _tensor_name(layer, expert, projection, kind)
        if name not in weight_map:
            raise KeyError(f"missing tensor in index: {name}")
        shard = weight_map[name]
        meta = headers[shard].get(name)
        if not isinstance(meta, dict):
            raise KeyError(f"missing tensor in shard header: {name}")
        dtype = str(meta["dtype"])
        expected_dtype = "I8" if kind == "weight" else "F8_E8M0"
        if dtype != expected_dtype:
            raise ValueError(f"{name}: dtype {dtype}, expected {expected_dtype}")
        shape = [int(value) for value in meta["shape"]]
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError(f"{name}: invalid shape {shape}")
        start, end = (int(value) for value in meta["data_offsets"])
        nbytes = end - start
        if nbytes != shape[0] * shape[1]:
            raise ValueError(
                f"{name}: byte length {nbytes} does not match shape {shape}"
            )
        components.append(
            {
                "projection": projection,
                "kind": kind,
                "dtype": dtype,
                "shape": shape,
                "offset": offset,
                "nbytes": nbytes,
            }
        )
        offset += nbytes
    return components


def _layout_metadata(components: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (component["projection"], component["kind"]): component
        for component in components
    }
    weights = [by_key[(projection, "weight")] for projection in PROJECTIONS]
    scales = [by_key[(projection, "scale")] for projection in PROJECTIONS]
    return {
        "record_bytes": sum(component["nbytes"] for component in components),
        "weight_offsets": [component["offset"] for component in weights],
        "weight_nbytes": [component["nbytes"] for component in weights],
        "weight_out": [component["shape"][0] for component in weights],
        "weight_stored_in": [component["shape"][1] for component in weights],
        "scale_offsets": [component["offset"] for component in scales],
        "scale_nbytes": [component["nbytes"] for component in scales],
        "scale_out": [component["shape"][0] for component in scales],
        "scale_in": [component["shape"][1] for component in scales],
    }


def _validate_component_layout(
    name: str, meta: dict[str, Any], expected: dict[str, Any]
) -> None:
    dtype = str(meta.get("dtype"))
    shape = [int(value) for value in meta.get("shape", [])]
    start, end = (int(value) for value in meta.get("data_offsets", (0, -1)))
    if dtype != expected["dtype"] or shape != expected["shape"]:
        raise ValueError(
            f"{name}: layout {(dtype, shape)} != "
            f"{(expected['dtype'], expected['shape'])}"
        )
    if end - start != expected["nbytes"]:
        raise ValueError(
            f"{name}: byte length {end - start} != {expected['nbytes']}"
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), "utf-8")
    temporary.replace(path)


def repack(
    safetensors_dir: Path,
    output_dir: Path,
    *,
    index_path: Path | None = None,
    start_layer: int = 0,
    end_layer: int = NUM_LAYERS,
    experts_per_layer: int = EXPERTS_PER_LAYER,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repack routed expert tensors into fixed-stride DEE4 v2 records."""
    safetensors_dir = Path(safetensors_dir)
    output_dir = Path(output_dir)
    if start_layer < 0 or end_layer <= start_layer:
        raise ValueError(f"invalid layer range [{start_layer}, {end_layer})")
    if experts_per_layer <= 0:
        raise ValueError("experts_per_layer must be positive")

    weight_map, headers, shard_paths = _load_checkpoint_index(
        safetensors_dir, index_path
    )
    components = _component_metadata(
        weight_map, headers, layer=start_layer, expert=0
    )
    layout = _layout_metadata(components)
    record_bytes = int(layout["record_bytes"])
    total_experts = (end_layer - start_layer) * experts_per_layer
    report: dict[str, Any] = {
        "schema": "dee4-v2-repack-report",
        "source_repository": OFFICIAL_REPOSITORY,
        "source_revision": OFFICIAL_REVISION,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "num_layers": end_layer - start_layer,
        "experts_per_layer": experts_per_layer,
        "total_experts": total_experts,
        "record_bytes": record_bytes,
        "estimated_total_gib": total_experts * record_bytes / (1 << 30),
        "shard_count": len(shard_paths),
        "layers": [],
        "errors": [],
        "success": False,
    }
    if dry_run:
        for layer in range(start_layer, end_layer):
            for expert in range(experts_per_layer):
                for expected in components:
                    name = _tensor_name(
                        layer, expert, expected["projection"], expected["kind"]
                    )
                    if name not in weight_map:
                        raise KeyError(f"missing tensor in index: {name}")
                    shard = weight_map[name]
                    _validate_component_layout(name, headers[shard][name], expected)
        report.update(
            {
                "success": True,
                "total_experts_repacked": 0,
                "total_bytes_repacked": 0,
                "total_elapsed_s": 0.0,
                "throughput_mibps": 0.0,
                "dry_run": True,
            }
        )
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    occupied = [path for path in output_dir.iterdir() if path.name != ".gitkeep"]
    if occupied:
        raise FileExistsError(
            f"refusing to overwrite non-empty DEE4 directory: {output_dir}"
        )
    data_partial = output_dir / f"{DATA_FILE}.partial"
    integrity_partial = output_dir / f"{INTEGRITY_FILE}.partial"
    data_hash = hashlib.sha256()
    total_bytes = 0
    started = time.monotonic()
    data_handle: BinaryIO | None = None
    integrity_handle = None

    try:
        data_handle = data_partial.open("wb", buffering=8 << 20)
        integrity_handle = integrity_partial.open("w", encoding="utf-8")
        record_index = 0
        for layer in range(start_layer, end_layer):
            layer_started = time.monotonic()
            layer_bytes = 0
            for expert in range(experts_per_layer):
                record_offset = total_bytes
                record_hash = hashlib.sha256()
                component_hashes: dict[str, str] = {}
                source_shards: set[str] = set()
                for expected in components:
                    projection = expected["projection"]
                    kind = expected["kind"]
                    name = _tensor_name(layer, expert, projection, kind)
                    if name not in weight_map:
                        raise KeyError(f"missing tensor in index: {name}")
                    shard = weight_map[name]
                    source_shards.add(shard)
                    meta = headers[shard].get(name)
                    if not isinstance(meta, dict):
                        raise KeyError(f"missing tensor in shard header: {name}")
                    _validate_component_layout(name, meta, expected)
                    raw = read_tensor_bytes(shard_paths[shard], headers[shard], name)
                    data_handle.write(raw)
                    data_hash.update(raw)
                    record_hash.update(raw)
                    component_hashes[f"{projection}.{kind}"] = hashlib.sha256(
                        raw
                    ).hexdigest()
                    total_bytes += len(raw)
                    layer_bytes += len(raw)
                if total_bytes - record_offset != record_bytes:
                    raise RuntimeError(
                        f"record ({layer},{expert}) has "
                        f"{total_bytes - record_offset} bytes, expected {record_bytes}"
                    )
                integrity_handle.write(
                    json.dumps(
                        {
                            "layer": layer,
                            "expert": expert,
                            "record_index": record_index,
                            "record_offset": record_offset,
                            "record_bytes": record_bytes,
                            "record_sha256": record_hash.hexdigest(),
                            "component_sha256": component_hashes,
                            "source_shards": sorted(source_shards),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                record_index += 1
                if expert % 32 == 31:
                    elapsed = max(time.monotonic() - started, 0.001)
                    rate = total_bytes / elapsed / (1 << 20)
                    print(
                        f"  layer {layer}/{end_layer} expert "
                        f"{expert + 1}/{experts_per_layer} "
                        f"({record_index} records, {rate:.1f} MiB/s)",
                        flush=True,
                    )
            layer_elapsed = time.monotonic() - layer_started
            report["layers"].append(
                {
                    "layer": layer,
                    "experts_repacked": experts_per_layer,
                    "bytes": layer_bytes,
                    "elapsed_s": round(layer_elapsed, 3),
                }
            )
            print(
                f"  layer {layer}: {experts_per_layer} records, "
                f"{layer_bytes / (1 << 30):.2f} GiB in {layer_elapsed:.1f}s",
                flush=True,
            )
        data_handle.flush()
        os.fsync(data_handle.fileno())
        integrity_handle.flush()
        os.fsync(integrity_handle.fileno())
    except Exception as exc:
        report["errors"].append(repr(exc))
        report["total_bytes_repacked"] = total_bytes
        report["total_elapsed_s"] = round(time.monotonic() - started, 3)
        _write_json_atomic(output_dir / "repack_error.json", report)
        raise
    finally:
        if data_handle is not None:
            data_handle.close()
        if integrity_handle is not None:
            integrity_handle.close()

    elapsed = time.monotonic() - started
    expected_total = total_experts * record_bytes
    if total_bytes != expected_total:
        raise RuntimeError(f"DEE4 size {total_bytes} != expected {expected_total}")
    data_path = output_dir / DATA_FILE
    integrity_path = output_dir / INTEGRITY_FILE
    data_partial.replace(data_path)
    integrity_partial.replace(integrity_path)

    metadata: dict[str, Any] = {
        "format": "dee4-v2",
        "codec": "deepseek-fp4-e2m1-e8m0",
        "source_repository": OFFICIAL_REPOSITORY,
        "source_revision": OFFICIAL_REVISION,
        "data_file": DATA_FILE,
        "integrity_file": INTEGRITY_FILE,
        "data_sha256": data_hash.hexdigest(),
        "start_layer": start_layer,
        "num_layers": end_layer - start_layer,
        "experts_per_layer": experts_per_layer,
        "total_experts": total_experts,
        "total_bytes": total_bytes,
        "group_size": GROUP_SIZE,
        "components": components,
        **layout,
    }
    _write_json_atomic(output_dir / "metadata.json", metadata)
    report.update(
        {
            "success": True,
            "total_experts_repacked": total_experts,
            "total_bytes_repacked": total_bytes,
            "total_elapsed_s": round(elapsed, 3),
            "throughput_mibps": round(
                total_bytes / max(elapsed, 0.001) / (1 << 20), 3
            ),
            "data_sha256": data_hash.hexdigest(),
        }
    )
    _write_json_atomic(output_dir / "repack_report.json", report)
    return report


def _sample_experts(
    *,
    start_layer: int,
    num_layers: int,
    experts_per_layer: int,
    sample_count: int,
    seed: int,
) -> list[tuple[int, int]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    last_layer = start_layer + num_layers - 1
    middle_layer = start_layer + num_layers // 2
    anchors = [
        (start_layer, 0),
        (start_layer, experts_per_layer - 1),
        (middle_layer, experts_per_layer // 2),
        (last_layer, 0),
        (last_layer, experts_per_layer - 1),
    ]
    candidates = list(dict.fromkeys(anchors))
    rng = random.Random(seed)
    population = num_layers * experts_per_layer
    target = min(sample_count, population)
    while len(candidates) < target:
        candidate = (
            rng.randrange(start_layer, start_layer + num_layers),
            rng.randrange(experts_per_layer),
        )
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates[:target]


def validate_dee4_against_safetensors(
    safetensors_dir: Path,
    dee4_dir: Path,
    *,
    index_path: Path | None = None,
    samples: list[tuple[int, int]] | None = None,
    sample_count: int = 12,
    seed: int = 731,
) -> dict[str, Any]:
    """Prove selected DEE4 components are byte-exact canonical tensors."""
    safetensors_dir = Path(safetensors_dir)
    dee4_dir = Path(dee4_dir)
    metadata = json.loads((dee4_dir / "metadata.json").read_text("utf-8"))
    if metadata.get("format") != "dee4-v2":
        raise ValueError("unsupported DEE4 format")
    if metadata.get("source_revision") != OFFICIAL_REVISION:
        raise ValueError("DEE4 source revision is not the pinned official revision")
    weight_map, headers, shard_paths = _load_checkpoint_index(
        safetensors_dir, index_path
    )
    components = metadata["components"]
    selected = samples or _sample_experts(
        start_layer=int(metadata["start_layer"]),
        num_layers=int(metadata["num_layers"]),
        experts_per_layer=int(metadata["experts_per_layer"]),
        sample_count=sample_count,
        seed=seed,
    )
    record_bytes = int(metadata["record_bytes"])
    records: list[dict[str, Any]] = []
    all_shards: set[str] = set()
    success = True
    with (dee4_dir / metadata["data_file"]).open("rb") as dee4_handle:
        for layer, expert in selected:
            record_index = (
                (layer - int(metadata["start_layer"]))
                * int(metadata["experts_per_layer"])
                + expert
            )
            if record_index < 0 or record_index >= int(metadata["total_experts"]):
                raise IndexError(f"sample outside DEE4 range: {(layer, expert)}")
            dee4_handle.seek(record_index * record_bytes)
            record = dee4_handle.read(record_bytes)
            if len(record) != record_bytes:
                raise IOError(f"short DEE4 record: {(layer, expert)}")
            component_results = []
            record_ok = True
            for component in components:
                projection = component["projection"]
                kind = component["kind"]
                name = _tensor_name(layer, expert, projection, kind)
                shard = weight_map[name]
                all_shards.add(shard)
                canonical = read_tensor_bytes(
                    shard_paths[shard], headers[shard], name
                )
                start = int(component["offset"])
                end = start + int(component["nbytes"])
                candidate = record[start:end]
                matches = candidate == canonical
                record_ok = record_ok and matches
                component_results.append(
                    {
                        "tensor": name,
                        "source_shard": shard,
                        "nbytes": len(canonical),
                        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
                        "dee4_sha256": hashlib.sha256(candidate).hexdigest(),
                        "exact_match": matches,
                    }
                )
            success = success and record_ok
            records.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "record_index": record_index,
                    "record_sha256": hashlib.sha256(record).hexdigest(),
                    "exact_match": record_ok,
                    "components": component_results,
                }
            )
    return {
        "schema": "dee4-v2-import-validation",
        "source_revision": OFFICIAL_REVISION,
        "dee4_data_sha256": metadata["data_sha256"],
        "sample_seed": seed,
        "sample_count": len(selected),
        "source_shards_covered": sorted(all_shards),
        "records": records,
        "success": success,
    }


def benchmark_dee4_read(dee4_dir: Path, n_experts: int = 64) -> dict[str, Any]:
    """Benchmark the real serving access unit: contiguous expert records."""
    dee4_dir = Path(dee4_dir)
    metadata = json.loads((dee4_dir / "metadata.json").read_text("utf-8"))
    record_bytes = int(metadata["record_bytes"])
    total_experts = int(metadata["total_experts"])
    count = min(max(1, n_experts), total_experts)
    read_bytes = count * record_bytes
    started = time.monotonic()
    with (dee4_dir / metadata["data_file"]).open("rb") as handle:
        data = handle.read(read_bytes)
    elapsed = time.monotonic() - started
    if len(data) != read_bytes:
        raise IOError(f"short DEE4 benchmark read {len(data)} != {read_bytes}")
    bandwidth = read_bytes / max(elapsed, 0.001) / (1 << 20)
    return {
        "format": "dee4-v2",
        "access_pattern": "sequential fixed-stride expert records",
        "n_experts": count,
        "record_bytes": record_bytes,
        "read_mib": read_bytes / (1 << 20),
        "elapsed_s": round(elapsed, 6),
        "aggregate_mbps": round(bandwidth, 3),
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "tests": [
            {
                "file": metadata["data_file"],
                "n_experts": count,
                "read_mib": read_bytes / (1 << 20),
                "elapsed_s": round(elapsed, 6),
                "mbps": round(bandwidth, 3),
            }
        ],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DEE4 v2 expert-major repack")
    parser.add_argument("--safetensors-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-path")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=NUM_LAYERS)
    parser.add_argument("--experts-per-layer", type=int, default=EXPERTS_PER_LAYER)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--validate-samples", type=int, default=12)
    args = parser.parse_args()

    source = Path(args.safetensors_dir)
    output = Path(args.output_dir)
    report = repack(
        source,
        output,
        index_path=Path(args.index_path) if args.index_path else None,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        experts_per_layer=args.experts_per_layer,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["success"]:
        return 1
    if args.dry_run:
        return 0

    validation = validate_dee4_against_safetensors(
        source,
        output,
        index_path=Path(args.index_path) if args.index_path else None,
        sample_count=args.validate_samples,
    )
    _write_json_atomic(output / "import_validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    if not validation["success"]:
        return 1
    if args.benchmark:
        benchmark = benchmark_dee4_read(output)
        _write_json_atomic(output / "read_benchmark.json", benchmark)
        print(json.dumps(benchmark, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
