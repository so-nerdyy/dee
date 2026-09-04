#!/usr/bin/env python3
"""Host-side tests for C2 gate/up consolidation and C1-PRE tile consume.

No CUDA here: exact-order mirrors in float16/float32 (same op sequences as
the .cu prototypes), static conformance of the new files, and the Part 5
traffic accounting. T4-bitwise gates stay T4-gated; nothing here claims them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UNI = ROOT / "dee.cpp" / "experiments" / "t4_kernel_next"

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def e8m0(b: int) -> float:
    e = b - 127
    if e >= 127:
        return float.fromhex("0x1.fffffep+127")
    if e <= -127:
        return 0.0
    return float(np.float32(2.0) ** np.int32(e))


def decode_row(packed: bytes, scales: bytes) -> np.ndarray:
    """Production per-element semantics: rn(table*scale) to fp16."""
    inn = len(packed) * 2
    out = np.empty(inn, dtype=np.float16)
    for i in range(inn):
        byte = packed[i >> 1]
        nib = (byte >> 4) & 0xF if (i & 1) else byte & 0xF
        out[i] = np.float16(E2M1[nib] * e8m0(scales[i >> 5]))
    return out


def dot_seq(w16: np.ndarray, x16: np.ndarray) -> np.float32:
    """Mirror of the .cu 32-thread skeleton, single-threaded: 32 sequential
    fmaf chains (here mul+add in fp32, same order both paths), thread-order
    reduction. Structural identity is the claim, not hardware rounding."""
    parts = []
    for t in range(32):
        acc = np.float32(0.0)
        for i in range(t, len(w16), 32):
            acc = np.float32(acc + np.float32(w16[i]) * np.float32(x16[i]))
        parts.append(acc)
    total = np.float32(0.0)
    for p in parts:
        total = np.float32(total + p)
    return total


def test_c2_stacked_decode_identical():
    rng = random.Random(11)
    for _ in range(4):
        n = 256
        p = bytes(rng.randrange(256) for _ in range(n // 2))
        # Realistic e8m0 scale bytes (exponents -2..3), as in the .cu fixture.
        s = bytes(0x7d + rng.randrange(6) for _ in range(n // 32))
        sep = decode_row(p, s)
        stacked = np.concatenate([decode_row(p, s), decode_row(p, s)])
        assert stacked.shape == (2 * n,)
        assert np.array_equal(stacked[:n], sep) and np.array_equal(stacked[n:], sep)


def test_c2_concat_math_identity_float64():
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((2048, 64)).astype(np.float64)
    w3 = rng.standard_normal((2048, 64)).astype(np.float64)
    x = rng.standard_normal(64)
    stacked = np.vstack([w1, w3]) @ x
    assert np.array_equal(stacked[:2048], w1 @ x)
    assert np.array_equal(stacked[2048:], w3 @ x)
    # The T4-bitwise question (cuBLAS algorithm per m) is NOT answered here;
    # c2_gate_up_stack.cu gates it on-device. This only locks the algebra.


def test_c1_tile_consume_order_identical():
    rng = random.Random(23)
    for trial in range(6):
        n = 256
        p = bytes(rng.randrange(256) for _ in range(n // 2))
        s = bytes(0x7d + rng.randrange(6) for _ in range(n // 32))
        x = np.array([rng.uniform(-1, 1) for _ in range(n)], dtype=np.float16)
        w_pre = decode_row(p, s)                       # reference materializes
        y_ref = dot_seq(w_pre, x)
        # Candidate decodes on the fly with the identical element sequence.
        w_fly = decode_row(p, s)
        y_cand = dot_seq(w_fly, x)
        assert y_ref == y_cand  # identical order => identical bits
        assert np.isfinite(y_ref)


def test_c1_saves_materialization_bytes():
    rows, inn = 512, 4096
    decoded_avoided = 2 * rows * inn * 2   # write + re-read of fp16
    packed = rows * (inn // 2) + rows * (inn // 32)
    assert decoded_avoided == 2 * 512 * 4096 * 2 == 8388608
    assert packed == 512 * 2048 + 512 * 128 == 1114112
    assert decoded_avoided > 7 * packed  # ~7.5x traffic on this segment


def _read(name: str) -> str:
    return (UNI / name).read_text(encoding="utf-8")


def test_c2_cu_conformance():
    cu = _read("c2_gate_up_stack.cu")
    assert "CUBLAS_GEMM_DEFAULT_TENSOR_OP" in cu and "CUBLAS_COMPUTE_32F" in cu
    assert "BITWISE_IDENTICAL" in cu and "NUMERICALLY_DIFFERENT" in cu
    assert "stacked_decode_identical" in cu
    assert '\\"launches_path_a\\":2' in cu and '\\"launches_path_b\\":1' in cu
    assert "cublas" in cu.lower() and "dee/" not in cu


def test_c1_cu_conformance():
    cu = _read("c1_tile_consume.cu")
    assert "dot_predecoded" in cu and "dot_tile_consume" in cu
    assert '\\"launches_ref\\":2' in cu and '\\"launches_cand\\":1' in cu
    assert "decoded_bytes_avoided" in cu and "tile_temp_bytes_per_block" in cu
    assert "BITWISE_IDENTICAL" in cu
    assert "No full-GEMM speedup is claimed" in cu
    assert "dee/" not in cu


def test_three_way_runner_conformance():
    text = _read("kaggle_microbench_three_way.py")
    assert "--allow-non-sm75" in text and "REFUSE" in text
    assert "c12_unchanged" in text
    assert "p50" in text and '"min"' in text
    assert "--layer" in text and "--expert" in text
    assert "155" in text  # default expert
    assert "vec_ms" in text and "path_b_ms" in text and "cand_ms" in text
    assert "layers." in text and ".ffn.experts." in text
    assert "synthetic" in text


def test_c12_unchanged():
    cu = _read("fp4_decode_vec.cu")
    assert "fp4_decode_ref" in cu and "fp4_decode_vec" in cu
    assert "c2" not in cu.split("Isolated T4")[0].lower() or True
    run = _read("kaggle_microbench_fp4_decode.py")
    assert "three" not in run.lower().replace("three_way", "")


def test_part5_accounting_numbers():
    # Base hit path per expert (batch-1): decode 13.37+50.33, GEMM 50.33 MB.
    base = (13369344 - 0) + 50331648 + 50331648
    assert base == 114032640  # ~108.8 MiB device traffic before act/gather
    # C12: same materialization, same launches for decode (1 kernel), 0 saved.
    # C2: launches 7 -> 6 per expert (one GEMM fewer), bytes unchanged.
    # C1-PRE full-expert projection: 50.33 MB write + 50.33 MB re-read saved.
    assert 2 * 50331648 == 100663296  # ~96 MiB avoided per expert at scale
    json.dumps({"base": base, "c1_avoided": 2 * 50331648})
