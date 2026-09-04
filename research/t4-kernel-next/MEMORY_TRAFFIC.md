# Memory traffic per expert (source-derived arithmetic)

Geometry: one routed expert = 25,165,824 params (3×4096×2048). Packed record
13,369,344 B (12.75 MiB). Decoded FP16 50,331,648 B (48 MiB). Batch-1 decode.
"DRAM" = device memory traffic unless marked H2D.

## B1. Full FP16 decode scratch — 63.7 MB, the largest materialization

`decode_fp4_cache_block_to_scratch` writes all 50.33 MB FP16 (3 kernels) and
reads 13.37 MB packed+scales: **63.70 MB per expert use, hit or miss.**
Unnecessary in the strict sense: the values are consumed exactly once, in
order, by the three GEMMs — full materialization exists only because decode
and GEMM are separate launches. Tile-granular decode-at-consumption (fused
dequant-in-GEMV) removes the 50.33 MB write and most of the re-read
(CANDIDATE_DESIGNS C1/C5).

## B2. GEMM weight re-reads — 50.3 MB

Gate 16.78 MB + up 16.78 MB + down 16.78 MB FP16 reads, plus ~24 KB
input/activation traffic. Unavoidable for memory-bound GEMV at batch-1
(weights must stream once); the waste is that they stream a *decoded copy*
rather than the 13.37 MB packed original — same fused fix as B1.

## B3. Gate/up/down intermediates — ~12 KB at batch-1

4 KiB gate + 4 KiB up + 4 KiB activation (+16 KiB FP32 out). Already minimal;
scales with group tokens but stays orders below weights. No action.

## B4. Gather/scatter + H2D — 12.8 MB on miss, ~24 KB + 12.75 MB otherwise

D2D gather (8 KiB) + scatter (16 KiB) are noise; `direct_single_row_io`
already bypasses them for singleton groups. H2D 12.75 MB is the
information-theoretic floor per miss (packed record must arrive once).

## Total per expert per token (batch-1)

| | Miss | Hit (no H2D) |
|---|---|---|
| H2D | 12.75 MB | 0 |
| Decode read+write | 63.70 MB | 63.70 MB |
| GEMM reads | 50.33 MB | 50.33 MB |
| Rest | ~0.04 MB | ~0.04 MB |
| **Total** | **~126.8 MB** | **~114.1 MB** |

At ~258 routed experts/token (corrected-trace shape): ~30–33 GB device
traffic per token. Locked by `tests/test_t4_kernel_next.py`
(`test_decode_dominates_traffic`, `test_token_scale_upper_bound_only`).

## Roofline UPPER BOUNDS (T4 320 GB/s; bounds only, never TPS)

- Removing the 50.33 MB FP16 write + 50.33 MB re-read (fused
  decode-consume) saves ~100 MB of ~114 MB on hits: **up to ~7× less
  traffic on the decode+GEMM portion**, i.e. traffic floor drops from
  ~114 MB to ~14 MB per expert (packed read once + accumulator traffic).
- Gate+up single-GEMM fusion saves 0 bytes, ~1 launch + 1 epilogue per
  expert (launch-boundary win only).
- Vectorized nibble loads save ~0 traffic bytes (same data, fewer
  instructions); win is instruction/latency-bound only.
- None of these convert to tok/s: launch overhead, occupancy, wave
  quantization, H2D overlap, and layer pipelining dominate the translation.
  The only admissible next number is a T4 measurement (MICROBENCH_PROTOCOL).
