#!/usr/bin/env python3
"""Generate a faithful synthetic safetensors shard for the dee.cpp Step 4 smoke
test. Uses the EXACT Ornith/Qwen3.5-MoE tensor naming and BF16 dtype so the C++
resolver exercises the real offset math + bf16->f32 conversion.

We write a single shard with a few expert tensors for layer 0, plus an embed
tensor, with KNOWN float values so the C++ test can assert correctness.

Format (safetensors):
  [uint64 LE header_len][header JSON][data bytes]
  header: {"tensor_name": {"dtype","shape","data_offsets":[start,end]}, ...}
"""
import struct, json, os, sys

OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "data", "layer0_shard.safetensors")

def f32_to_bf16(x: float) -> int:
    import struct as _s
    u = _s.unpack("<I", _s.pack("<f", x))[0]
    return (u >> 16) & 0xFFFF

# Build tensors. Each expert gate_proj/up_proj is [inter, hidden]; down_proj is
# [hidden, inter]. We'll use small dims so the file is tiny but realistic.
HIDDEN, INTER = 16, 8

def name(layer, expert, kind):
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight"

# All experts share IDENTICAL known values so the C++ test can assert exact
# equality for any (layer, expert) without per-expert offset bookkeeping.
GATE = [0.5, -1.25, 2.0, -3.75, 4.25]
UP   = [1.0, -2.0, 3.0, -4.0, 5.0]
DOWN = [0.125, -0.25, 0.375, -0.5, 0.625]

tensors = {}
# layer 0, experts 0..2 (3 experts so the cache eviction test has pressure)
for e in (0, 1, 2):
    g = [float(GATE[i % len(GATE)]) for i in range(INTER*HIDDEN)]
    u = [float(UP[i % len(UP)]) for i in range(INTER*HIDDEN)]
    d = [float(DOWN[i % len(DOWN)]) for i in range(HIDDEN*INTER)]
    tensors[name(0, e, "gate_proj")] = ("BF16", [INTER, HIDDEN], g)
    tensors[name(0, e, "up_proj")]   = ("BF16", [INTER, HIDDEN], u)
    tensors[name(0, e, "down_proj")] = ("BF16", [HIDDEN, INTER], d)

# a non-expert tensor to make the header realistic
tensors["model.language_model.embed_tokens.weight"] = (
    "BF16", [4, HIDDEN], [0.1*i for i in range(4*HIDDEN)])

# Serialize data + header
buf = bytearray()
offsets = {}
for tname, (dtype, shape, vals) in tensors.items():
    start = len(buf)
    for v in vals:
        buf += struct.pack("<H", f32_to_bf16(v))
    end = len(buf)
    offsets[tname] = (start, end)

header = {tname: {"dtype": dt, "shape": sh, "data_offsets": list(offsets[tname])}
          for tname, (dt, sh, _) in tensors.items()}
header_json = json.dumps(header, separators=(",", ":"))
hlen = len(header_json)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    f.write(struct.pack("<Q", hlen))
    f.write(header_json.encode("utf-8"))
    f.write(buf)

# emit the known values for the C++ test to assert (identical for all experts)
manifest = {
    "gate": GATE,
    "up":   UP,
    "down": DOWN,
}
with open(OUT + ".expected.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"wrote {OUT}")
print(f"  size={os.path.getsize(OUT)} bytes, tensors={len(tensors)}")
print(f"  expert gate/up/down share known first5 (no per-expert offset)")
