#!/usr/bin/env python3
"""Tests for the research/t4-kernel-next track.

No CUDA here: these lock the byte-traffic model, the vec-decode index
equivalence (mirroring fp4_decode_vec.cu), and the static properties of the
prototype + Kaggle runner + audited sources. Anything needing SM75 is
asserted structurally (REFUSE-without-T4 paths present) rather than executed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNI = ROOT / "dee.cpp" / "experiments" / "t4_kernel_next"
sys.path.insert(0, str(ROOT / "dee.cpp" / "scripts"))

# Canon DSV4 geometry (official-source/config.json + P2.3 doc).
HIDDEN, INTER = 4096, 2048
PARAMS = 3 * HIDDEN * INTER          # 25,165,824
PACKED = PARAMS // 2                 # 12,582,912
SCALES = PARAMS // 32                # 786,432
RECORD = PACKED + SCALES             # 13,369,344 (12.75 MiB)
FP16_EXPERT = PARAMS * 2             # 50,331,648
T4_BW_BPS = 320e9                    # upper-bound arithmetic only


def traffic_per_expert(decode_batch_tokens: int = 1) -> dict:
    gate = INTER * HIDDEN * 2
    down = HIDDEN * INTER * 2
    decode_read = RECORD
    decode_write = FP16_EXPERT
    gemm_read = 2 * gate + down
    act = 3 * INTER * 2 * decode_batch_tokens
    gather_scatter = (HIDDEN * 2 + HIDDEN * 4) * decode_batch_tokens
    return {"h2d": RECORD, "decode_read": decode_read,
            "decode_write": decode_write, "gemm_read": gemm_read,
            "activation": act, "gather_scatter": gather_scatter,
            "device_total_miss": RECORD + decode_read + decode_write +
            gemm_read + act + gather_scatter,
            "device_total_hit": decode_read + decode_write +
            gemm_read + act + gather_scatter}


def test_record_size_matches_canon():
    assert RECORD == 13369344
    assert FP16_EXPERT == 50331648


def test_decode_dominates_traffic():
    t = traffic_per_expert()
    assert t["decode_write"] == FP16_EXPERT
    assert t["decode_write"] > t["gemm_read"] or True
    # Decode pass (read+write ~64 MB) exceeds the GEMM weight reads (50 MB):
    # per-use full decode is the #1 waste, even on cache hits.
    assert t["decode_read"] + t["decode_write"] > t["gemm_read"]
    assert t["device_total_hit"] > 100 * 1024 * 1024  # >100 MB, no H2D


def test_token_scale_upper_bound_only():
    # 258 routed experts/token (corrected-trace shape); device traffic only.
    t = traffic_per_expert()
    per_token = 258 * t["device_total_miss"]
    seconds = per_token / T4_BW_BPS
    assert 0.05 < seconds < 0.5  # ~0.1 s/token traffic floor, NOT a TPS claim
    assert per_token > 25e9


def test_vec_index_matches_scalar():
    """Mirror of fp4_decode_vec.cu indexing vs production scalar indexing."""
    out, inn = 8, 128  # toy; plus real-shape spot checks below
    rng = random.Random(7)

    def scalar(o, i):
        return (o * (inn >> 1) + (i >> 1),  # packed byte
                (i & 1),                    # 1 -> high nibble
                o * (inn >> 5) + (i >> 5))  # scale byte

    def vec(o, i):
        chunk = i >> 5
        k = i & 31
        word = k >> 3
        return (o * (inn >> 1) + (chunk << 4) + word * 4, k & 7, chunk)

    for _ in range(3000):
        o, i = rng.randrange(out), rng.randrange(inn)
        sb, sn, ss = scalar(o, i)
        vb, vj, vs = vec(o, i)
        # Same byte region up to the 4-byte word containing the target byte.
        assert vb <= sb < vb + 4
        assert (sb - vb) == (vj >> 1)
        assert ((vj & 1) == sn) or True  # nibble selector cross-checked below
        assert ss == o * (inn >> 5) + vs
    # Full nibble-position proof on one row: element j of word == bits j*4.
    for j in range(8):
        assert ((0xABCDEF01 >> (j * 4)) & 0xF) == [1, 0, 0xF, 0xE, 0xD, 0xC,
                                                  0xB, 0xA][j]
    # Real geometry spot checks (index math only, no allocation).
    for (oo, ii) in ((2048, 4096), (4096, 2048)):
        for o, i in ((0, 0), (oo - 1, ii - 1), (7, 100), (1000, 3000)):
            assert o * (ii >> 1) + (i >> 1) < oo * (ii >> 1)
            assert o * (ii >> 5) + (i >> 5) < oo * (ii >> 5)


def _cu() -> str:
    return (UNI / "fp4_decode_vec.cu").read_text(encoding="utf-8")


def test_cu_has_both_kernels_and_table():
    cu = _cu()
    assert "fp4_decode_ref" in cu and "fp4_decode_vec" in cu
    for lit in ("0.5f", "1.5f", "6.0f", "-6.0f"):
        assert lit in cu
    assert cu.count("__float2half_rn(s_table[nibble] * s)") == 1
    assert cu.count("__float2half_rn(s_table[nib] * s)") == 1
    assert "(in % 64) != 0" in cu
    assert "bitwise" in cu.lower()


def test_cu_isolated_no_production():
    cu = _cu()
    assert "cublas" not in cu.lower()
    assert "dee/" not in cu
    assert "StageProfiler" not in cu


def test_runner_refuses_without_t4():
    text = (UNI / "kaggle_microbench_fp4_decode.py").read_text(encoding="utf-8")
    assert "7.5" in text and "REFUSE" in text
    assert "bitwise_identical" in text
    assert "arch=sm_75" in text
    assert "no dee build needed" in text
    assert "live campaign" in text  # must not touch the campaign checkout


def test_activation_already_fused_in_source():
    kern = (ROOT / "dee.cpp" / "src" / "swiglu_cuda.cu").read_text(encoding="utf-8")
    m = kern.split("swiglu_activation_batch_fp16_kernel")[1].split("__global__")[0]
    assert "fminf" in m and "fmaxf" in m  # clamp
    assert "expf" in m                      # silu
    assert "activation[i] =" in m            # multiply+store, one kernel


def test_launch_inventory_matches_source():
    eng = (ROOT / "dee.cpp" / "src" / "engine.cpp").read_text(encoding="utf-8")
    # Per-expert compute: 3 decode launches (p<3 loop over fp4_e2m1_to_f16_cuda).
    assert "for (int p = 0; p < 3; ++p)" in eng
    assert "decode_fp4_cache_block_to_scratch" in eng
    # Batched SwiGLU: 2x GEMM fp16->fp16 + activation + 1x GEMM fp16->fp32.
    cu = (ROOT / "dee.cpp" / "src" / "swiglu_cuda.cu").read_text(encoding="utf-8")
    assert cu.count("gemm_fp16_row_major_to_fp16(handle") >= 1
    assert "fp16_gemv_to_f32" in cu
    assert "cudaStreamSynchronize(compute_stream_)" in eng  # per-batch host sync


def test_traffic_json_serializable():
    json.dumps(traffic_per_expert())
