#!/usr/bin/env python3
"""P2.2: DEE4 expert-major contiguous repack.

Reads official DeepSeek-V4-Flash safetensors shards and writes DEE4 expert-major
banks with fixed-stride arithmetic addressing: offset = (L * E + e) * stride.

DEE4 format:
  dee4/
    metadata.json          -- dimensions, strides, checksums, mapping ledger
    expert_w1.fp4          -- gate weights  [L*E entries, stride=w1_stride]
    expert_w1.scale        -- gate E8M0 scales
    expert_w3.fp4          -- up weights
    expert_w3.scale        -- up E8M0 scales
    expert_w2.fp4          -- down weights
    expert_w2.scale        -- down E8M0 scales

No numerical transformation.  Exact bytes from the canonical safetensors.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

# ── Expert geometry (from the pinned checkpoint) ──────────────────────────
HIDDEN = 2048
INTERMEDIATE = 4096
NUM_LAYERS = 43          # routed MoE layers
EXPERTS_PER_LAYER = 256
FIRST_ROUTED_LAYER = 0   # layer 0 has routed experts
LAYER_STEP = 1           # every layer is a routed MoE layer

# FP4 packed shapes as stored in safetensors (I8 dtype, 2 el/byte):
#  w1 (gate): [HIDDEN, INTERMEDIATE] -> stored [2048, 4096//2] = [2048, 2048]
#  w3 (up):   same
#  w2 (down): [INTERMEDIATE, HIDDEN] -> stored [4096, 2048//2] = [4096, 1024]
# E8M0 scales: one per group of 32 elements in the inner FP4 dimension
#  w1 scale: [2048, 4096//32] = [2048, 128]
#  w2 scale: [4096, 2048//32] = [4096, 64]
GROUP_SIZE = 32

W1_STORED_SHAPE = [HIDDEN, INTERMEDIATE // 2]   # [2048, 2048] I8 bytes
W2_STORED_SHAPE = [INTERMEDIATE, HIDDEN // 2]   # [4096, 1024] I8 bytes
W1_SCALE_SHAPE  = [HIDDEN, INTERMEDIATE // GROUP_SIZE]  # [2048, 128]
W2_SCALE_SHAPE  = [INTERMEDIATE, HIDDEN // GROUP_SIZE]  # [4096, 64]

W1_BYTES      = HIDDEN * (INTERMEDIATE // 2)          # 4_194_304
W1_SCALE_BYTES = HIDDEN * (INTERMEDIATE // GROUP_SIZE) #   262_144
W2_BYTES      = INTERMEDIATE * (HIDDEN // 2)           # 4_194_304
W2_SCALE_BYTES = INTERMEDIATE * (HIDDEN // GROUP_SIZE) #   262_144
# w3 identical to w1
EXPERT_TOTAL_BYTES = (2 * W1_BYTES + 2 * W1_SCALE_BYTES
                       + W2_BYTES + W2_SCALE_BYTES)    # 13_369_344

TOTAL_EXPERTS = NUM_LAYERS * EXPERTS_PER_LAYER  # 11_008
TOTAL_GIB     = TOTAL_EXPERTS * EXPERT_TOTAL_BYTES / (1 << 30)  # ≈137 GiB

# ── Tensor name patterns ──────────────────────────────────────────────────
PROJECTIONS = ["w1", "w2", "w3"]
KINDS = ["weight", "scale"]

# Map (proj, kind) -> stride, stored_shape for verification
PROJ_META: dict[str, dict[str, Any]] = {
    ("w1", "weight"): {"stride": W1_BYTES,      "shape": W1_STORED_SHAPE, "dtype": "I8"},
    ("w1", "scale"):  {"stride": W1_SCALE_BYTES, "shape": W1_SCALE_SHAPE,  "dtype": "F8_E8M0"},
    ("w2", "weight"): {"stride": W2_BYTES,       "shape": W2_STORED_SHAPE, "dtype": "I8"},
    ("w2", "scale"):  {"stride": W2_SCALE_BYTES, "shape": W2_SCALE_SHAPE,  "dtype": "F8_E8M0"},
    ("w3", "weight"): {"stride": W1_BYTES,       "shape": W1_STORED_SHAPE, "dtype": "I8"},
    ("w3", "scale"):  {"stride": W1_SCALE_BYTES, "shape": W1_SCALE_SHAPE,  "dtype": "F8_E8M0"},
}

# Output file names
OUTPUT_FILES = {
    ("w1", "weight"): "expert_w1.fp4",
    ("w1", "scale"):  "expert_w1.scale",
    ("w3", "weight"): "expert_w3.fp4",
    ("w3", "scale"):  "expert_w3.scale",
    ("w2", "weight"): "expert_w2.fp4",
    ("w2", "scale"):  "expert_w2.scale",
}


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read the JSON header from a safetensors file."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def read_tensor_bytes(path: Path, header: dict[str, Any],
                      tensor_name: str) -> bytes:
    """Read raw tensor bytes from a safetensors file."""
    if tensor_name not in header:
        raise KeyError(f"tensor '{tensor_name}' not in {path.name}")
    meta = header[tensor_name]
    offsets = meta["data_offsets"]
    length = offsets[1] - offsets[0]
    with open(path, "rb") as f:
        f.seek(8 + offsets[0])  # skip header
        data = f.read(length)
    if len(data) != length:
        raise IOError(f"short read for {tensor_name}: {len(data)} < {length}")
    return data


def repack(
    safetensors_dir: Path,
    output_dir: Path,
    *,
    start_layer: int = 0,
    end_layer: int = NUM_LAYERS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repack routed expert tensors into DEE4 contiguous banks.

    Args:
        safetensors_dir: Directory containing model-*.safetensors + index
        output_dir: Where to write DEE4 files
        start_layer, end_layer: Range of layers to repack (default: all)
        dry_run: If True, only validate and measure, don't write output

    Returns a report dict.
    """
    report: dict[str, Any] = {
        "schema": "dee4-v1",
        "start_layer": start_layer,
        "end_layer": end_layer,
        "num_layers": end_layer - start_layer,
        "experts_per_layer": EXPERTS_PER_LAYER,
        "total_experts": (end_layer - start_layer) * EXPERTS_PER_LAYER,
        "expert_total_bytes": EXPERT_TOTAL_BYTES,
        "estimated_total_gib": ((end_layer - start_layer) * EXPERTS_PER_LAYER
                                 * EXPERT_TOTAL_BYTES / (1 << 30)),
        "layers": [],
    }

    # ── 1. Load safetensors index ──────────────────────────────────────
    index_path = safetensors_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"index missing: {index_path}")
    index = json.loads(index_path.read_text("utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    # Build shard -> header cache
    shard_headers: dict[str, dict[str, Any]] = {}
    shard_paths: dict[str, Path] = {}

    for shard_name in sorted(set(weight_map.values())):
        p = safetensors_dir / shard_name
        if not p.is_file():
            raise FileNotFoundError(f"shard missing: {p}")
        shard_paths[shard_name] = p
        shard_headers[shard_name] = read_safetensors_header(p)

    report["shard_count"] = len(shard_paths)

    # ── 2. Open DEE4 output files (appending, random-access) ──────────
    output_files: dict[tuple[str, str], Any] = {}
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for (proj, kind), fname in OUTPUT_FILES.items():
            fpath = output_dir / fname
            output_files[(proj, kind)] = open(fpath, "wb")

    # ── 3. Iterate layers, then experts, extract and write ────────────
    total_experts_repacked = 0
    total_bytes_repacked = 0
    t0 = time.monotonic()
    errors: list[str] = []

    for layer in range(start_layer, end_layer):
        layer_start = time.monotonic()
        layer_experts = 0
        layer_checksums: dict[int, dict[str, str]] = {}

        for expert_id in range(EXPERTS_PER_LAYER):
            for proj in PROJECTIONS:
                for kind in KINDS:
                    tensor_name = (
                        f"layers.{layer}.ffn.experts.{expert_id}"
                        f".{proj}.{kind}"
                    )

                    if tensor_name not in weight_map:
                        errors.append(f"missing: {tensor_name}")
                        continue

                    shard_name = weight_map[tensor_name]
                    header = shard_headers[shard_name]
                    shard_path = shard_paths[shard_name]

                    try:
                        raw = read_tensor_bytes(shard_path, header, tensor_name)
                    except Exception as exc:
                        errors.append(f"read error {tensor_name}: {exc}")
                        continue

                    # Verify expected size
                    meta = PROJ_META.get((proj, kind))
                    if meta is None:
                        errors.append(f"unknown (proj,kind): {tensor_name}")
                        continue

                    expected = meta["stride"]
                    if len(raw) != expected:
                        errors.append(
                            f"size mismatch {tensor_name}: "
                            f"got {len(raw)}, expected {expected}"
                        )
                        continue

                    # Compute checksum
                    sha = hashlib.sha256(raw).hexdigest()
                    if expert_id not in layer_checksums:
                        layer_checksums[expert_id] = {}
                    layer_checksums[expert_id][f"{proj}.{kind}"] = sha

                    # Write at arithmetic offset
                    if not dry_run:
                        expert_offset = (layer - start_layer) * EXPERTS_PER_LAYER + expert_id
                        byte_offset = expert_offset * meta["stride"]
                        f = output_files[(proj, kind)]
                        f.seek(byte_offset)
                        f.write(raw)
                        f.flush()  # ensure durable writes

                    total_bytes_repacked += len(raw)

            total_experts_repacked += 1
            layer_experts += 1

            # Progress every 32 experts
            if expert_id % 32 == 31:
                elapsed = time.monotonic() - t0
                rate = total_bytes_repacked / max(elapsed, 0.001) / (1 << 20)
                print(f"  layer {layer}/{end_layer} expert {expert_id+1}/{EXPERTS_PER_LAYER} "
                      f"({total_experts_repacked} total, {rate:.1f} MiB/s)", flush=True)

        layer_elapsed = time.monotonic() - layer_start
        report["layers"].append({
            "layer": layer,
            "experts_repacked": layer_experts,
            "bytes": total_bytes_repacked,
            "elapsed_s": round(layer_elapsed, 2),
            "checksums": layer_checksums,
        })
        print(f"  ✓ layer {layer}: {layer_experts} experts in {layer_elapsed:.1f}s",
              flush=True)

    # ── 4. Close output files ──────────────────────────────────────────
    for f in output_files.values():
        f.close()

    total_elapsed = time.monotonic() - t0
    report["total_elapsed_s"] = round(total_elapsed, 1)
    report["total_experts_repacked"] = total_experts_repacked
    report["total_bytes_repacked"] = total_bytes_repacked
    report["throughput_mibps"] = round(
        total_bytes_repacked / max(total_elapsed, 0.001) / (1 << 20), 1
    )
    report["errors"] = errors
    report["success"] = len(errors) == 0

    # ── 5. Write metadata ──────────────────────────────────────────────
    if not dry_run:
        metadata = {
            "format": "dee4-v1",
            "source_repository": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "hidden_dim": HIDDEN,
            "intermediate_dim": INTERMEDIATE,
            "group_size": GROUP_SIZE,
            "num_layers": end_layer - start_layer,
            "experts_per_layer": EXPERTS_PER_LAYER,
            "strides": {
                "w1_weight": W1_BYTES,
                "w1_scale": W1_SCALE_BYTES,
                "w2_weight": W2_BYTES,
                "w2_scale": W2_SCALE_BYTES,
                "w3_weight": W1_BYTES,      # same as w1
                "w3_scale": W1_SCALE_BYTES,
            },
            "total_experts": total_experts_repacked,
            "total_bytes": total_bytes_repacked,
            "output_files": {str(k): v for k, v in OUTPUT_FILES.items()},
            "checksums_per_expert": report["layers"],
            "repack_elapsed_s": total_elapsed,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), "utf-8"
        )

    return report


def benchmark_dee4_read(dee4_dir: Path, n_experts: int = 64) -> dict[str, Any]:
    """Benchmark contiguous expert reads from DEE4 format."""
    meta = json.loads((dee4_dir / "metadata.json").read_text("utf-8"))

    results: dict[str, Any] = {
        "n_experts": n_experts,
        "tests": [],
    }

    for fname_key, fname in OUTPUT_FILES.items():
        proj, kind = fname_key
        fpath = dee4_dir / fname
        if not fpath.is_file():
            continue

        stride = meta["strides"][f"{proj}_{kind}"]
        file_size = fpath.stat().st_size
        total_experts = file_size // stride

        # Sequential read of n_experts contiguous entries
        offset = 0  # start of file
        read_size = min(n_experts * stride, file_size)

        t0 = time.monotonic()
        with open(fpath, "rb") as f:
            f.seek(offset)
            data = f.read(read_size)
        elapsed = time.monotonic() - t0

        results["tests"].append({
            "file": fname,
            "proj_kind": f"{proj}.{kind}",
            "stride": stride,
            "n_experts": min(n_experts, total_experts),
            "read_mib": read_size / (1 << 20),
            "elapsed_s": round(elapsed, 4),
            "mbps": round(read_size / max(elapsed, 0.001) / (1 << 20), 1),
            "data_sha256": hashlib.sha256(data).hexdigest(),
        })

    overall_bytes = sum(t["read_mib"] for t in results["tests"]) * (1 << 20)
    overall_time = sum(t["elapsed_s"] for t in results["tests"])
    results["aggregate_mbps"] = round(
        overall_bytes / max(overall_time, 0.001) / (1 << 20), 1
    )

    return results


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="DEE4 expert-major repack")
    ap.add_argument("--safetensors-dir", required=True,
                    help="Directory with model-*.safetensors + index")
    ap.add_argument("--output-dir", required=True,
                    help="DEE4 output directory")
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=NUM_LAYERS)
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate only, don't write output")
    ap.add_argument("--benchmark", action="store_true",
                    help="Benchmark DEE4 reads after repack")
    args = ap.parse_args()

    safetensors_dir = Path(args.safetensors_dir)
    output_dir = Path(args.output_dir)

    print(f"DEE4 repack: {args.start_layer}-{args.end_layer} layers "
          f"({(args.end_layer-args.start_layer)*EXPERTS_PER_LAYER} experts, "
          f"~{(args.end_layer-args.start_layer)*EXPERTS_PER_LAYER*EXPERT_TOTAL_BYTES/(1<<30):.1f} GiB)",
          flush=True)
    print(f"Source: {safetensors_dir}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    report = repack(
        safetensors_dir,
        output_dir,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        dry_run=args.dry_run,
    )

    print(json.dumps({
        "success": report["success"],
        "total_elapsed_s": report["total_elapsed_s"],
        "total_experts": report["total_experts_repacked"],
        "total_gib": report["total_bytes_repacked"] / (1 << 30),
        "throughput_mibps": report["throughput_mibps"],
        "errors": report["errors"][:10],
    }, indent=2), flush=True)

    # Save full report
    (output_dir / "repack_report.json").write_text(
        json.dumps(report, indent=2), "utf-8"
    )

    if not report["success"]:
        print(f"\nFAILED: {len(report['errors'])} errors", flush=True)
        return 1

    # ── Benchmark ───────────────────────────────────────────────────
    if args.benchmark:
        print("\n=== DEE4 read benchmark ===", flush=True)
        bench = benchmark_dee4_read(output_dir)
        print(json.dumps(bench, indent=2), flush=True)
        (output_dir / "read_benchmark.json").write_text(
            json.dumps(bench, indent=2), "utf-8"
        )

    print("\n✓ DEE4 repack complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())