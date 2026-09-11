#!/usr/bin/env python3
"""Generate a multi-layer MoE shard (W8) -- numpy-free.

Naming EXACTLY matches dee::TensorResolver::expert_tensor_name:
  model.language_model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.weight

dtype token "BF16" matches dee.cpp weight_mmap.cpp's parser.

Per-layer random seeds (`args.seed + 1009*(layer+1)`) yield statistically
distinct per-layer weight magnitudes. random.gauss() from stdlib produces
zero-mean unit-variance Gaussian samples, then *layer_scale matches numpy.

Memory profile:
- One expert's three random projections are generated, packed to BF16
  bytes, and appended to a `tensors` list before moving to the next
  expert. Each Float32 work array is freed after packing.
- Final file.write is one shot (header_prefix + all payloads).
"""

import argparse
import hashlib
import json
import os
import random
import struct
import sys


def f32_to_bf16_packed(v: float) -> bytes:
    """Pack one FP32 -> BF16 -> little-endian 16-bit bytes (truncate top)."""
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    return struct.pack("<H", (u >> 16) & 0xFFFF)


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

    inter, hidden = args.inter, args.hidden
    inter_hidden = inter * hidden

    name_for = lambda kind, layer, expert: (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight")

    # Build the tensor list. Each entry: (key, shape, bytes, start_offset)
    tensors = []
    cursor = 0
    for layer in range(args.layers):
        rng = random.Random(args.seed + 1009 * (layer + 1))
        layer_scale = args.layer_scale_base + 0.01 * (layer % 17)
        for expert in range(args.experts):
            # gate [inter x hidden]
            buf = bytearray(inter_hidden * 2)
            for j in range(inter_hidden):
                v = rng.gauss(0.0, 1.0) * layer_scale
                packed = f32_to_bf16_packed(v)
                buf[j * 2:j * 2 + 2] = packed
            tensors.append((("gate_proj", layer, expert),
                            (inter, hidden), bytes(buf), cursor))
            cursor += len(buf)
            # up [inter x hidden]
            buf = bytearray(inter_hidden * 2)
            for j in range(inter_hidden):
                v = rng.gauss(0.0, 1.0) * layer_scale
                packed = f32_to_bf16_packed(v)
                buf[j * 2:j * 2 + 2] = packed
            tensors.append((("up_proj", layer, expert),
                            (inter, hidden), bytes(buf), cursor))
            cursor += len(buf)
            # down [hidden x inter] (same total element count, different shape)
            buf = bytearray(inter_hidden * 2)
            for j in range(inter_hidden):
                v = rng.gauss(0.0, 1.0) * layer_scale
                packed = f32_to_bf16_packed(v)
                buf[j * 2:j * 2 + 2] = packed
            tensors.append((("down_proj", layer, expert),
                            (hidden, inter), bytes(buf), cursor))
            cursor += len(buf)
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
        header_prefix = struct.pack("<Q", hlen) + header_json.encode("utf-8")
        if pad:
            header_prefix += b" " * pad
        f.write(header_prefix)
        sha.update(header_prefix)
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
