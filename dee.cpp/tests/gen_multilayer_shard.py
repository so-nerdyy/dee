#!/usr/bin/env python3
"""Generate a streaming multi-layer MoE shard for benchmarking.

Generates `num_layers` distinct physical layers, each with `num_experts`
distinct experts. Each projection (gate/up/down) uses BF16 values derived
from a seeded RNG so that distinct layers have distinct weights but the
shard is reproducible from the same seed.

The generator streams projections to disk: each expert's gate/up/down is
written as a single packed row-major BF16 blob before moving on. This
avoids holding all weights in RAM at once.
"""

import argparse
import io
import os
import struct
import sys
from typing import Tuple

try:
    import numpy as np
except ImportError:
    sys.stderr.write("[gen] numpy is required (pip install numpy).\n")
    sys.exit(1)


def bf16_bytes(values):
    """Return (packed_bytes, length_per_projection) BF16 row-major blob."""
    arr = values.astype(np.float32).reshape(-1)
    packed = arr.astype(np.bfloat16).tobytes()
    return packed, arr.shape[0]


def write_safetensors_header(tensors, total_bytes):
    """Write a minimal safetensors header JSON + 8-byte little-endian length.

    tensors: list of (name, (shape0, shape1), offset_bytes)
    """
    header = {}
    for name, shape, offset in tensors:
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + shape[0] * shape[1] * 2],
        }
    import json
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(raw) % 8)) % 8
    raw += b" " * pad
    return struct.pack("<Q", len(raw)) + raw, 8 + len(raw)


def gen(args):
    if args.num_layers <= 0 or args.num_experts <= 0:
        sys.stderr.write("[gen] --layers and --experts must be positive\n")
        sys.exit(2)

    inter = args.inter
    hidden = args.hidden

    outputs = {
        f"layers.{L}.mlp.experts.{E}.w1.weight": (inter, hidden),
        f"layers.{L}.mlp.experts.{E}.w2.weight": (hidden, inter),
        f"layers.{L}.mlp.experts.{E}.w3.weight": (inter, hidden),
    } if False else None  # not used

    tensors = []
    running_offset = 0
    chunks = bytearray()

    rng = np.random.default_rng(args.seed)
    total_tensors = args.num_layers * args.num_experts * 3

    written = 0
    for layer in range(args.num_layers):
        # Layer-specific random projection magnitudes so layers are distinct.
        layer_seed = args.seed + 1009 * (layer + 1)
        rng_l = np.random.default_rng(layer_seed)
        layer_scale = 0.05 + 0.02 * (layer % 8)
        for expert in range(args.num_experts):
            for pname, shape in (
                ("w1", (inter, hidden)),
                ("w2", (hidden, inter)),
                ("w3", (inter, hidden)),
            ):
                size = shape[0] * shape[1]
                values = (rng_l.standard_normal(size) * layer_scale).astype(np.float32)
                packed, _ = bf16_bytes(values)
                name = f"layers.{layer}.mlp.experts.{expert}.{pname}.weight"
                offset = running_offset
                tensors.append((name, shape, offset))
                chunks.extend(packed)
                running_offset += shape[0] * shape[1] * 2
                written += 1
                if written % 256 == 0:
                    sys.stderr.write(
                        f"[gen] wrote {written}/{total_tensors} tensors "
                        f"({running_offset/1024/1024:.1f} MB)\n"
                    )

    header_bytes, header_total = write_safetensors_header(tensors, running_offset)

    sys.stderr.write(
        f"[gen] header {header_total} bytes, payload {running_offset} bytes, "
        f"total {header_total + running_offset/1024/1024:.1f} MB\n"
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        f.write(header_bytes)
        f.write(bytes(chunks))

    # Emit a checksum for reproducibility
    import hashlib
    h = hashlib.sha256()
    h.update(header_bytes)
    h.update(bytes(chunks))
    sys.stderr.write(
        f"[gen] wrote {args.output} -- layers={args.num_layers} "
        f"experts={args.num_experts} inter={inter} hidden={hidden} "
        f"sha256={h.hexdigest()[:16]}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--inter", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    gen(args)


if __name__ == "__main__":
    main()
