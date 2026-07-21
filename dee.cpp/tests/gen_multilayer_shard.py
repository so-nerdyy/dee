#!/usr/bin/env python3
"""Generate a streaming multi-layer MoE shard for benchmarking (W8).

Naming EXACTLY matches dee::TensorResolver::expert_tensor_name:
  model.language_model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.weight

Per-layer random seeds make adjacent layers have statistically distinct
weights so that cache eviction is exercised across layers rather than
reusing the same physical expert for each model layer.

Generator writes directly to the output file (no full payload buffered
in RAM), one tensor at a time. Only the tensor name index is kept in
memory.
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


def make_bf16_bytes(values):
    """Pack a float sequence into BF16 little-endian bytes."""
    return b"".join(struct.pack("<H", f32_to_bf16(v)) for v in values)


GEN_VERSION = 1


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

    if args.layers <= 0 or args.experts <= 0:
        sys.stderr.write("[gen] --layers and --experts must be positive\n")
        sys.exit(2)

    hidden = args.hidden
    inter = args.inter
    gate_shape = (inter, hidden)
    down_shape = (hidden, inter)

    try:
        import numpy as np
    except ImportError:
        sys.exit("[gen] numpy is required (pip install numpy)")

    # Build the tensor index incrementally. Each entry will be filled in
    # with the absolute file offset AFTER we know how much header padding to
    # use. Strategy: write a placeholder header first, accumulate offsets
    # per tensor, then re-write the placeholder with real offsets.
    tensor_index = []  # list of (name, kind, layer, expert, shape, payload_bytes)

    total_tensors = args.layers * args.experts * 3
    for layer in range(args.layers):
        rng = np.random.default_rng(args.seed + 1009 * (layer + 1))
        layer_scale = args.layer_scale_base + 0.01 * (layer % 17)
        for expert in range(args.experts):
            # gate: [inter x hidden]
            w_gate = rng.standard_normal(inter * hidden).astype(np.float32) * layer_scale
            tensor_index.append(("gate_proj", layer, expert, gate_shape,
                                 make_bf16_bytes(w_gate)))
            # up:   [inter x hidden]
            w_up = rng.standard_normal(inter * hidden).astype(np.float32) * layer_scale
            tensor_index.append(("up_proj", layer, expert, gate_shape,
                                 make_bf16_bytes(w_up)))
            # down: [hidden x inter]
            w_down = rng.standard_normal(hidden * inter).astype(np.float32) * layer_scale
            tensor_index.append(("down_proj", layer, expert, down_shape,
                                 make_bf16_bytes(w_down)))
        if (layer + 1) % 4 == 0:
            sys.stderr.write(f"[gen] prepared layer {layer+1}/{args.layers}\n")

    sys.stderr.write(f"[gen] prepared {total_tensors} tensors, writing file...\\n")

    name_for = lambda kind, layer, expert: (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight")

    # Build the header with provisional offsets, then re-emit with real ones.
    # Replace stdlib json with deterministic serialisation (no whitespace).
    def build_header(offset_map):
        return json.dumps(
            {name_for(kind, L, E): {
                "dtype": "BF16",
                "shape": list(shape),
                "data_offsets": [offset_map[(kind, L, E)], offset_map[(kind, L, E)] + len(payload)]
             } for kind, L, E, shape, payload in tensor_index},
            separators=(",", ":"))

    # Pass 1: emit a probe header to measure its length, compute padding, then
    # commit to the real offsets. The header JSON length is <= 64 bytes per
    # tensor name token, well bounded by total_tensors * ~96 bytes.
    probe_offsets = {key: 0 for key in [(k, L, E) for k, L, E, _, _ in tensor_index]}
    probe_header = build_header(probe_offsets)
    header_len_nopad = len(probe_header)
    pad_bytes = (8 - (header_len_nopad % 8)) % 8 if header_len_nopad % 8 else 0

    # The data section starts after the 8-byte LE length prefix and the
    # padded JSON header.
    data_start = 8 + header_len_nopad + pad_bytes
    real_offsets = {}
    cursor = 0
    for kind, L, E, shape, payload in tensor_index:
        real_offsets[(kind, L, E)] = cursor
        cursor += len(payload)

    real_header = build_header(real_offsets)
    assert len(real_header) == header_len_nopad, "header length must be stable across passes"

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    sha = hashlib.sha256()
    with open(args.output, "wb") as f:
        f.write(struct.pack("<Q", header_len_nopad))
        f.write(real_header.encode("utf-8"))
        f.write(b" " * pad_bytes)
        sha.update(f.getvalue() if hasattr(f, "getvalue") else b"")
        for kind, L, E, shape, payload in tensor_index:
            f.write(payload)
            sha.update(payload)

    total_bytes = data_start + cursor
    sys.stderr.write(
        f"[gen] wrote {args.output} -- layers={args.layers} experts={args.experts} "
        f"inter={inter} hidden={hidden} payload={cursor/1024/1024:.2f} MB "
        f"total={total_bytes/1024/1024:.2f} MB sha256={sha.hexdigest()[:16]}\\n"
    )


if __name__ == "__main__":
    main()
