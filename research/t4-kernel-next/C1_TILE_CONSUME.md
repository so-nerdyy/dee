# C1-PRE tile-local decode/consume (prototype built, T4 gate pending)

Prototype: `dee.cpp/experiments/t4_kernel_next/c1_tile_consume.cu`
(+ `kaggle_microbench_three_way.py` case `c1pre`). Answers only the premise:
"can a packed MXFP4 tile be decoded immediately before use without
materializing the full FP16 projection?" Full-GEMM speedup is NOT claimed
from this tile test.

## Experiment

One output row (default 512 rows × 4096 for timing stability; real-fixture
mode uses the full 2048-row w1) is processed two ways with the IDENTICAL
accumulation skeleton (32 threads, sequential fmaf chains, thread-order
reduction): reference dots a pre-decoded FP16 row; the candidate decodes each
weight on the fly with production per-element semantics (table lookup +
e8m0 scale + `__float2half_rn` BEFORE the multiply, then `fmaf` in FP32).
Identical op order + identical rounding ⇒ BITWISE gate, not tolerance gate.

## What it measures/validates

1. **Elimination**: candidate never writes the FP16 row — `decoded_bytes_avoided`
   = 2×rows×in×2 (write + re-read). At full-expert scale: 100.66 MB/expert.
2. **Behavior**: `bitwise_identical` memcmp over FP32 outputs.
3. **Pressure**: `tile_temp_bytes_per_block` = 32-partial shared array
   (128 B) + 16-entry table (64 B); zero extra registers beyond the
   accumulation the dot already needs; packed streaming is coalesced
   (16 B contiguous per 32-elem chunk).
4. **Bytes**: packed consumed = rows×(in/2 + in/32); nothing else.
5. **SM75 plausibility**: plain FMA + shared + `__half2float`; no tensor
   cores, no `cp.async`, occupancy = rows blocks × 32 threads (2048-row
   real case fills 40 SMs; 512-row synthetic underfills — timing only).

## Admission

`BITWISE_IDENTICAL` (memcmp clean) or `NUMERICALLY_DIFFERENT` (any diff;
design then needs the F2 oracle gate at unchanged tolerances before any
further work). A pass proves the premise (materialization eliminable,
order-preserving, T4-plausible) — NOT that a full fused GEMM beats cuBLAS.

## Production touch if the premise passes AND a full kernel follows

New kernel file + `cuda_convert.cu` (decode entry) + `engine.cpp` decode
seam. The tile kernel itself is not production-shaped (row-dots, not GEMM);
its job is to retire premise risk before anyone writes the fused GEMM.
