#!/usr/bin/env python3
"""Generate a multi-layer MoE shard (W8).

Naming EXACTLY matches dee::TensorResolver::expert_tensor_name:
  model.language_model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.weight

dtype label is "BF16" (canonical for dee.cpp's json_min parser).

Per-layer random seeds (`args.seed + 1009*(layer+1)`) produce
statistically distinct per-layer weight statistics, so the synthetic
benchmark exercises real cache-eviction pressure across model layers
instead of aliasing all layers to the same physical expert set.

Memory discipline:
- numpy.random.standard_normal produces temporary Float32 arrays that are
  freed by `make_bf16_bytes`.
- BF16 payloads are packed once and added to a list of (key, bytes)
  tuples; total memory bound is the payload size plus a thin metadata
  index (~150 bytes per tensor).
- The shard file is written exactly once after the payload list is
  complete (header + padding + payloads appended in a single file.write).
"""

import argparse
import hashlib
import json
import os
import struct
import sys


def f32_to_bf16(x: float) -> int:
    """Round FP32 to BF16 (truncate to top 16 bits)."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return (u >> 16) & 0xFFFF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--inter", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--layer-scale-base", type=float, default=0.10)
    args = parser.parse_args()

    if args.layers <= 0 or args.experts <= 0 or args.inter <= 0 or args.hidden <= 0:
        sys.stderr.write("[gen] --layers --experts --inter --hidden must be positive\n")
        sys.exit(2)

    hidden = args.hidden
    inter = args.inter
    inter_hidden = inter * hidden

    try:
        import numpy as np
    except ImportError:
        sys.exit("[gen] numpy is required (pip install numpy)")

    name_for = lambda kind, layer, expert: (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight")

    # Pass 1: generate all per-tensor BF16 payloads and accumulate offsets.
    # Each entry: (key, shape, bf16_bytes)
    tensors = []
    cursor = 0
    for layer in range(args.layers):
        rng = np.random.default_rng(args.seed + 1009 * (layer + 1))
        layer_scale = args.layer_scale_base + 0.01 * (layer % 17)
        for expert in range(args.experts):
            # gate: [inter x hidden]
            w_gate = rng.standard_normal(inter_hidden).astype(np.float32) * layer_scale
            gate_bytes = b"".join(struct.pack("<H", f32_to_bf16(v)) for v in w_gate)
            tensors.append((("gate_proj", layer, expert), (inter, hidden), gate_bytes, cursor))
            cursor += len(gate_bytes)
            # up:   [inter x hidden] (same shape as gate)
            w_up = rng.standard_normal(inter_hidden).astype(np.float32) * layer_scale
            up_bytes = b"".join(struct.pack("<H", f32_to_bf16(v)) for v in w_up)
            tensors.append((("up_proj", layer, expert), (inter, hidden), up_bytes, cursor))
            cursor += len(up_bytes)
            # down: [hidden x inter]
            w_down = rng.standard_normal(inter_hidden).astype(np.float32) * layer_scale
            down_bytes = b"".join(struct.pack("<H", f32_to_bf16(v)) for v in w_down)
            tensors.append((("down_proj", layer, expert), (hidden, inter), down_bytes, cursor))
            cursor += len(down_bytes)
        if (layer + 1) % max(1, args.layers // 8) == 0:
            sys.stderr.write(f"[gen] prepared layer {layer+1}/{args.layers}\n")

    payload_bytes = cursor
    sys.stderr.write(f"[gen] prepared {len(tensors)} tensors, "
                     f"payload {payload_bytes/1024/1024:.2f} MB\n")

    # Build header once with real offsets.
    header = {
        name_for(kind, layer, expert): {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        for (kind, layer, expert), shape, payload, offset in tensors
    }
    header_json = json.dumps(header, separators=(",", ":"))
    hlen = len(header_json)
    pad = (8 - (hlen % 8)) % 8

    data_section_start = 8 + hlen + pad
    total_file_bytes = data_section_start + payload_bytes

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    sha = hashlib.sha256()
    with open(args.output, "wb") as f:
        header_prefix = struct.pack("<Q", hlen) + header_json.encode("utf-8") + (b" " * pad)
        f.write(header_prefix)
        sha.update(header_prefix)
        # Append all payloads; the offset values inside the header point into
        # this data section starting at offset 0.
        for key, shape, payload, offset in tensors:
            f.write(payload)
            sha.update(payload)

    sys.stderr.write(
        f"[gen] wrote {args.output} -- layers={args.layers} experts={args.experts} "
        f"inter={inter} hidden={hidden} total={total_file_bytes/1024/1024:.2f} MB "
        f"sha256={sha.hexdigest()[:16]}\n"
    )


if __name__ == "__main__":
    main()
