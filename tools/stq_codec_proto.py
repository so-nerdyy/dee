#!/usr/bin/env python3
"""stq_codec_proto.py — Phase F: standalone STQ-like 3-of-4 ternary codec reference.

NOT for production. Must NOT enter dee Engine. Isolated reference implementation
for format verification and CPU decode-throughput measurement.

Format (derived from Sherry, arXiv:2601.07892 §3.1, and the public llama.cpp
STQ1_0 PR #22836 block layout, MIT):
    - weights constrained to {-d, 0, +d}; exactly one zero per contiguous
      group of 4 weights (3:4 structured sparsity)
    - per group of 4: 2 bits zero-position + 3 bits signs = 5 bits
      (C(4,3) * 2^3 = 32 permutations, saturating the 5-bit code)
    - per 256-weight block: 64 groups * 5 bits = 320 bits = 40 bytes payload
      + 2 bytes fp16 scale -> 42 bytes = 1.3125 bits/weight
    - encoder: smallest-|x| lane per group becomes the zero; d = max|x| of block

This file re-implements the format from the paper + public block layout; it
does not copy llama.cpp kernel code.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time

try:
    import numpy as np
except ImportError:
    np = None

QK_BLOCK = 256
GROUP = 4
GROUPS_PER_BLOCK = QK_BLOCK // GROUP          # 64
PAYLOAD_BYTES = GROUPS_PER_BLOCK * 5 // 8     # 40
BLOCK_BYTES = PAYLOAD_BYTES + 2               # 42
BPW = BLOCK_BYTES * 8 / QK_BLOCK              # 1.3125


def _pack_bits(bits: list[int]) -> bytes:
    """LSB-first bit packer."""
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def _unpack_bits(data: bytes, n: int) -> list[int]:
    return [(data[i >> 3] >> (i & 7)) & 1 for i in range(n)]


def encode_block(x: list[float]) -> bytes:
    """Quantize+pack 256 floats -> 42-byte STQ-like block."""
    assert len(x) == QK_BLOCK
    d = max(abs(v) for v in x)  # 0 for an all-zero block: decode returns all zeros
    bits: list[int] = []
    for g in range(GROUPS_PER_BLOCK):
        quad = x[g * GROUP:(g + 1) * GROUP]
        absq = [abs(v) for v in quad]
        z = absq.index(min(absq))            # smallest |x| -> zero lane
        zbits = [(z >> i) & 1 for i in range(2)]
        sbits = []
        for j in range(GROUP):
            if j == z:
                continue
            sbits.append(1 if quad[j] >= 0 else 0)
        bits.extend(zbits)
        bits.extend(sbits)
    payload = _pack_bits(bits)
    assert len(payload) == PAYLOAD_BYTES
    return payload + struct.pack("<e", d)


def decode_block(block: bytes) -> list[float]:
    """Unpack 42-byte block -> 256 ternary floats."""
    d = struct.unpack("<e", block[PAYLOAD_BYTES:])[0]
    bits = _unpack_bits(block[:PAYLOAD_BYTES], GROUPS_PER_BLOCK * 5)
    y = [0.0] * QK_BLOCK
    for g in range(GROUPS_PER_BLOCK):
        o = g * 5
        z = bits[o] | (bits[o + 1] << 1)
        signs = bits[o + 2:o + 5]
        si = 0
        for j in range(GROUP):
            if j == z:
                y[g * GROUP + j] = 0.0
            else:
                y[g * GROUP + j] = d if signs[si] else -d
                si += 1
    return y


# ---------------------------------------------------------------------------
# numpy vectorized decode (throughput path)
# ---------------------------------------------------------------------------
def decode_many_np(blocks: bytes) -> "np.ndarray":
    """Vectorized decode of n contiguous 42-byte blocks -> int8 ternary codes.
    Returns array of shape (n*256,) with values {-1, 0, +1}."""
    if np is None:
        raise RuntimeError("numpy not available")
    n = len(blocks) // BLOCK_BYTES
    raw = np.frombuffer(blocks, dtype=np.uint8).reshape(n, BLOCK_BYTES)
    payload = raw[:, :PAYLOAD_BYTES]
    scales = raw[:, PAYLOAD_BYTES:].copy().view(np.float16).astype(np.float32)

    # bit i of payload stream, LSB-first across 40 bytes
    bit_idx = np.arange(GROUPS_PER_BLOCK * 5)
    byte = payload[:, bit_idx >> 3]                       # (n, 320)
    bitv = (byte >> (bit_idx & 7).astype(np.uint8)) & 1   # (n, 320)

    bits = bitv.reshape(n, GROUPS_PER_BLOCK, 5)
    z = bits[:, :, 0] | (bits[:, :, 1].astype(np.int32) << 1)   # (n, 64) zero pos
    signs = bits[:, :, 2:5]                                     # (n, 64, 3)

    # lane j of group g gets: 0 if j==z else sign
    lanes = np.arange(GROUP)
    is_zero = lanes[None, None, :] == z[:, :, None]             # (n,64,4)
    # sign for non-zero lanes: lane j uses signs[min(j - (j>z), 3)]
    compact = lanes[None, None, :] - (lanes[None, None, :] > z[:, :, None])
    sign_per_lane = np.take_along_axis(
        signs, np.clip(compact, 0, 2), axis=2)                  # (n,64,4)
    vals = np.where(sign_per_lane == 1, 1, -1).astype(np.int8)
    out = np.where(is_zero, np.int8(0), vals)
    out = (out.reshape(n, QK_BLOCK) * scales).astype(np.float32)  # scales (n,1) broadcasts
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def run_tests() -> list[dict]:
    random.seed(1234)
    results = []

    # 1. round-trip correctness on random data (expected uses the fp16-rounded
    # scale exactly as stored in the block)
    x = [random.gauss(0, 1) for _ in range(QK_BLOCK)]
    block = encode_block(x)
    y = decode_block(block)
    d = struct.unpack("<e", struct.pack("<e", max(abs(v) for v in x)))[0]
    expected = []
    for g in range(GROUPS_PER_BLOCK):
        quad = x[g * GROUP:(g + 1) * GROUP]
        absq = [abs(v) for v in quad]
        z = absq.index(min(absq))
        for j in range(GROUP):
            expected.append(0.0 if j == z else (d if quad[j] >= 0 else -d))
    ok = all(abs(a - b) < 1e-6 for a, b in zip(y, expected))
    results.append({"test": "roundtrip_random_block", "pass": ok,
                    "block_bytes": len(block), "bpw": BPW})

    # 2. exact 3:4 sparsity invariant
    zeros = sum(1 for v in y if v == 0.0)
    results.append({"test": "sparsity_3of4", "pass": zeros == QK_BLOCK // 4,
                    "zeros_per_256": zeros})

    # 3. bit count / block overhead
    results.append({"test": "format_constants",
                    "pass": BLOCK_BYTES == 42 and BPW == 1.3125,
                    "payload_bytes": PAYLOAD_BYTES, "scale_bytes": 2,
                    "block_bytes": BLOCK_BYTES, "bpw": BPW})

    # 4. determinism: encode(decode(encode(x))) == encode(x)
    x2 = decode_block(encode_block(x))
    results.append({"test": "reencode_idempotent",
                    "pass": encode_block(x2) == encode_block(x)})

    # 5. all-zero and all-same-sign blocks
    b0 = encode_block([0.0] * QK_BLOCK)
    y0 = decode_block(b0)
    bn = encode_block([-2.5] * QK_BLOCK)
    yn = decode_block(bn)
    results.append({"test": "edge_blocks",
                    "pass": all(v == 0.0 for v in y0) and all(v < 0 for v in yn if v != 0)})

    # 6. numpy fast path equivalence (if available)
    if np is not None:
        nb = 1000
        blob = b"".join(encode_block([random.gauss(0, 1) for _ in range(QK_BLOCK)])
                        for _ in range(nb))
        fast = decode_many_np(blob)
        slow = []
        for i in range(nb):
            slow.extend(decode_block(blob[i * BLOCK_BYTES:(i + 1) * BLOCK_BYTES]))
        ok = np.allclose(np.array(slow), fast, atol=1e-3)
        results.append({"test": "numpy_decode_equivalence", "pass": bool(ok),
                        "blocks_checked": nb})
    return results


# ---------------------------------------------------------------------------
# CPU decode throughput
# ---------------------------------------------------------------------------
def bench_python_decode(n_blocks: int) -> dict:
    random.seed(99)
    blob = b"".join(encode_block([random.gauss(0, 1) for _ in range(QK_BLOCK)])
                    for _ in range(n_blocks))
    t0 = time.perf_counter()
    w = 0
    for i in range(n_blocks):
        w += len(decode_block(blob[i * BLOCK_BYTES:(i + 1) * BLOCK_BYTES]))
    dt = time.perf_counter() - t0
    return {"impl": "python_ref", "blocks": n_blocks, "weights": w,
            "seconds": round(dt, 4), "Mweights_per_s": round(w / dt / 1e6, 3)}


def bench_numpy_decode(n_blocks: int, reps: int = 5) -> dict:
    if np is None:
        return {"impl": "numpy", "error": "numpy unavailable"}
    random.seed(99)
    blob = b"".join(encode_block([random.gauss(0, 1) for _ in range(QK_BLOCK)])
                    for _ in range(n_blocks))
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = decode_many_np(blob)
        dt = time.perf_counter() - t0
        w = out.size
        mps = w / dt / 1e6
        if best is None or mps > best["Mweights_per_s"]:
            best = {"impl": "numpy_vectorized", "blocks": n_blocks, "weights": w,
                    "seconds": round(dt, 4), "Mweights_per_s": round(mps, 3)}
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase F STQ-like codec prototype")
    ap.add_argument("--bench-blocks", type=int, default=40_000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tests = run_tests()
    bench_py = bench_python_decode(min(args.bench_blocks, 2_000))
    bench_np = bench_numpy_decode(args.bench_blocks) if np else {"error": "no numpy"}

    # expert-level projection: 25,165,824 logical weights per expert
    expert_weights = 25_165_824
    expert_blocks = expert_weights // QK_BLOCK
    expert_bytes = expert_blocks * BLOCK_BYTES
    out = {
        "format": {
            "block_weights": QK_BLOCK, "group": GROUP,
            "payload_bytes": PAYLOAD_BYTES, "scale_bytes": 2,
            "block_bytes": BLOCK_BYTES, "bpw": BPW,
        },
        "per_expert_projection": {
            "logical_weights": expert_weights,
            "blocks": expert_blocks,
            "stq_bytes": expert_bytes,
            "mxfp4_bytes": 13_369_344,
            "ratio": round(expert_bytes / 13_369_344, 4),
        },
        "tests": tests,
        "all_tests_pass": all(t["pass"] for t in tests),
        "decode_throughput": {"python_reference": bench_py, "numpy": bench_np},
        "note": "reference prototype only; NOT for production; not benchmarked on CUDA",
    }
    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0 if out["all_tests_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
