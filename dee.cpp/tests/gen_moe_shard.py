#!/usr/bin/env python3
"""Generate a 256-expert Ornith-style MoE shard for the dee.cpp Step 8 mock run.

Single layer (layer 0) with 256 experts so top-8 Oracle prediction exercises
real prefetch + cache-eviction pressure on the CPU-mock backend. Dims match the
real Ornith MoE: hidden=2048, inter=256, BF16 weights.

Naming matches dee::TensorResolver::expert_tensor_name exactly:
  model.language_model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.weight

Experts 0/1/2 carry the canonical KNOWN float values (so a C++ test can assert
exact equality); experts 3..255 carry a deterministic filler pattern. Generation
is instant: we build one expert's BF16 bytes and repeat it.

safetensors format: [uint64 LE header_len][header JSON][data bytes].
"""
import struct
import json
import os

OUT = "/mnt/c/Users/carth/Downloads/dee.cpp/tests/data/ornith_moe256.safetensors"
HIDDEN, INTER, NUM_EXPERTS, LAYER = 2048, 64, 256, 0


def f32_to_bf16(x: float) -> int:
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return (u >> 16) & 0xFFFF


def name(layer, expert, kind):
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight"


# canonical known values (gate/up/down first5) — matches Step 4 smoke test
GATE = [0.5, -1.25, 2.0, -3.75, 4.25]
UP = [1.0, -2.0, 3.0, -4.0, 5.0]
DOWN = [1.5, -2.5, 3.5, -4.5, 5.5]


def weight_bytes(pattern, n_elems):
    """Repeat a float pattern into `n_elems` BF16 bytes (fast, no per-value loop)."""
    pb = b"".join(struct.pack("<H", f32_to_bf16(v)) for v in pattern)
    q, r = divmod(n_elems, len(pattern))
    return pb * q + pb[: r * 2]


N = HIDDEN * INTER  # elements per weight (gate/up/down are all INTER*HIDDEN)

canon_gate = weight_bytes(GATE, N)
canon_up = weight_bytes(UP, N)
canon_down = weight_bytes(DOWN, N)
canon_expert = canon_gate + canon_up + canon_down  # one expert's full blob

filler_pattern = [(i % 17) * 0.01 - 0.08 for i in range(11)]
filler_up = weight_bytes([(i % 13) * 0.02 - 0.12 for i in range(11)], N)
# one expert blob = [gate N][up N][down N] in BF16 bytes
filler_expert = (
    weight_bytes(filler_pattern, N)
    + weight_bytes([(i % 13) * 0.02 - 0.12 for i in range(11)], N)
    + weight_bytes([(i % 19) * 0.015 - 0.14 for i in range(11)], N)
)

tensors = {}
data = bytearray()
for e in range(NUM_EXPERTS):
    blob = canon_expert if e < 3 else filler_expert
    gate_b = blob[0 : N * 2]
    up_b = blob[N * 2 : 2 * N * 2]
    down_b = blob[2 * N * 2 : 3 * N * 2]
    for kind, sl in zip(("gate_proj", "up_proj", "down_proj"), (gate_b, up_b, down_b)):
        tname = name(LAYER, e, kind)
        start = len(data)
        data += sl
        end = len(data)
        shape = [INTER, HIDDEN] if kind != "down_proj" else [HIDDEN, INTER]
        tensors[tname] = (start, end, shape)

header = {tname: {"dtype": "BF16", "shape": sh, "data_offsets": [st, en]}
          for tname, (st, en, sh) in tensors.items()}
header_json = json.dumps(header, separators=(",", ":"))
hlen = len(header_json)

with open(OUT, "wb") as f:
    f.write(struct.pack("<Q", hlen))
    f.write(header_json.encode("utf-8"))
    f.write(data)

manifest = {"gate": GATE, "up": UP, "down": DOWN,
            "hidden": HIDDEN, "inter": INTER, "num_experts": NUM_EXPERTS}
with open(OUT + ".expected.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"wrote {OUT}")
print(f"  size={os.path.getsize(OUT)} bytes ({os.path.getsize(OUT)/1024/1024:.1f} MB)")
print(f"  tensors={len(tensors)} (layer {LAYER}, {NUM_EXPERTS} experts x 3)")
print(f"  hidden={HIDDEN} inter={INTER} dtype=BF16")
