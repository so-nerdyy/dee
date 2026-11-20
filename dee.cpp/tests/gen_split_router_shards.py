#!/usr/bin/env python3
"""Create two tiny BF16 shards that jointly contain one routed-MoE layer."""

import json
import os
import struct
import sys


OUT_DIR = os.path.abspath(sys.argv[1])
LAYER, HIDDEN, INTER, EXPERTS = 5, 4, 2, 3


def bf16(values):
    payload = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        payload += struct.pack("<H", bits >> 16)
    return bytes(payload)


def write_shard(path, tensors):
    data = bytearray()
    header = {}
    for name, shape, values in tensors:
        payload = bf16(values)
        start = len(data)
        data += payload
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [start, len(data)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(data)


def projection(expert, kind):
    scale = 0.05 * (expert + 1)
    if kind == "gate_proj":
        return [scale, 0, 0, scale, scale, scale, 0, 0]
    if kind == "up_proj":
        return [scale, scale, scale, scale, -scale, scale, -scale, scale]
    return [scale, 0, 0, scale, scale, scale, -scale, scale]


def expert_tensors(expert):
    prefix = f"model.language_model.layers.{LAYER}.mlp.experts.{expert}"
    return [
        (f"{prefix}.gate_proj.weight", (INTER, HIDDEN), projection(expert, "gate_proj")),
        (f"{prefix}.up_proj.weight", (INTER, HIDDEN), projection(expert, "up_proj")),
        (f"{prefix}.down_proj.weight", (HIDDEN, INTER), projection(expert, "down_proj")),
    ]


os.makedirs(OUT_DIR, exist_ok=True)
router = (
    f"model.language_model.layers.{LAYER}.mlp.gate.weight",
    (EXPERTS, HIDDEN),
    [1, 0, 0, 0,
     0, 1, 0, 0,
     0, 0, 1, 0],
)
write_shard(os.path.join(OUT_DIR, "split-a.safetensors"),
            [router] + expert_tensors(0) + expert_tensors(1))
write_shard(os.path.join(OUT_DIR, "split-b.safetensors"), expert_tensors(2))

