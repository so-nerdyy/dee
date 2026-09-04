# Candidate kernel designs (exactness-first)

Strict rule applied throughout: a mathematically equivalent transformation is
NOT automatically exact — any change in floating-point operation order,
rounding boundary, or library algorithm choice drops a design to
EXACT_NEEDS_TEST at best. Rankings: MICROBENCH_NOW / PROMISING_BUT_COMPLEX /
LOW_VALUE / REJECT_NUMERICS / REJECT_T4.

## C1. Fused packed-decode + GEMV (dequant-in-dot) — PROMISING_BUT_COMPLEX

Decode each FP4 tile into registers/shared at consumption inside a custom
GEMV (one per projection, or one persistent kernel over gate+up+down).
Removes B1+B2 waste: ~100 MB/expert saved (§MEMORY_TRAFFIC roofline: up to
~7× traffic cut on the decode+GEMM portion). **EXACT_NEEDS_TEST**: dot-product
summation order will differ from cuBLAS GEMM (tree-reduce vs
split-K/tensor-core order), so bitwise equality vs the FP16-cuBLAS reference
is unlikely; admission needs the FP32-oracle gate (MICROBENCH_PROTOCOL F2),
not bitwise. Complexity high (needs a fast SM75 FP16 kernel to beat cuBLAS
memory-bound streaming — plausible but unproven). T4-compatible (plain
FMA/shared memory; optional `mma.sync.m8n8k4` later).

## C2. Gate/up dual-projection single GEMM — MICROBENCH_NOW

One `cublasGemmEx` with m=4096 over the contiguous decoded gate|up region
instead of two m=2048 calls. Saves 1 launch + 1 epilogue per expert, 0 bytes.
**EXACT_NEEDS_TEST**: cuBLAS may select a different algorithm for m=4096 vs
m=2048 (different summation order); bitwise must be tested, and on failure
the design is REJECT_NUMERICS (no tolerance change allowed). Trivial to
implement and benchmark (host-side only). Production touch: `swiglu_cuda.cu`
`swiglu_expert_batch_fp16_cuda` only.

## C3. Fused clamp + SiLU + multiply — ALREADY DONE (LOW_VALUE)

`swiglu_activation_batch_fp16_kernel` (`swiglu_cuda.cu:46-59`) already fuses
clamp, SiLU-with-FP16-rounding, and multiply in one kernel, preserving the
torch FP16-eager rounding boundaries. No further fusion exists at this site;
any algebraic merge (e.g. folding the multiply into the down-GEMM alpha)
would delete a rounding boundary → REJECT_NUMERICS. Locked by test
`test_activation_already_fused_in_source`.

## C4. Output-stationary down projection — LOW_VALUE standalone

The down GEMM is already a memory-bound GEMV reading 16.78 MB once; an
output-stationary custom kernel cannot move fewer bytes than one streaming
read + accumulator write. Value exists only as part of C1 (fuse down-decode
with the dot loop). Alone: no traffic win, launch win nil.

## C5. Tile-local decode (decode into shared, consume in-block) — part of C1

Same traffic win as C1, framed as the mechanism: 32-element scale blocks fit
naturally in shared memory (one scale byte per 32 values). No standalone
value; merged into C1.

## C6. Persistent packed residency — ALREADY DONE (no action)

P2.3 packed-FP4 VRAM cache (`DeviceCacheDType::Fp4E2m1`) already keeps
12.75 MiB blocks verbatim; decode happens at compute into one bounded
scratch. Do not re-propose; protect it (any candidate must keep residency
packed).

## C7. Persistent warps / fewer launches — PROMISING_BUT_COMPLEX

One persistent grid decoding+computing all experts of a layer (or token)
would cut ~1,800 launches/token toward a handful. **EXACT_NEEDS_TEST**
(scheduling changes summation/rounding order nowhere if each expert keeps
its dot order — actually order-preserving per expert is achievable, but the
engineering risk is high: worklists, occupancy tuning for 2048/4096-row
shapes on 40 SMs, T4 shared-memory limits). Defer until C1 proves the
per-expert fused kernel; C7 is its launcher, not its replacement.

## C8. Grouped expert dispatch — PROMISING (host-side)

Batch experts sharing residency into fewer `cudaMemcpyAsync`/event pairs and
larger GEMM batches. Numerically neutral if per-expert math is untouched
(EXACT_PROVABLE modulo cuBLAS batching-algorithm choice → test). Medium
complexity, host-side first. Good second microbench after C2.

## C9. Device-side worklists — LOW_VALUE today

Worklists pay off when launch overhead dominates *and* the work items are
tiny; here each item is a 16.78 MB GEMV (microseconds of traffic each), so
host-side batching (C8) captures most of the win. Revisit only if profiling
shows launch-bound behavior after C2/C8.

## C10. Routing-weight fusion — REJECT_NUMERICS (in-engine), valid in HF

Folding routing weights into GEMM alpha/beta or the down-projection would
delete the documented FP16 rounding boundaries
(`weighted_combine_fp16_kernel`, `swiglu_cuda.cu:106-113`: "one fused float
expression is observably different"). The boundaries exist because they were
measured to matter. Do not fuse; the Python-side combine already runs once
per token, not per expert.

## C11. Shared-memory / register reuse — mechanism, not a candidate

Applies inside C1/C7 kernels (table in shared — already done in the decode
kernel; scale broadcast per 32-block; double-buffered packed streaming).
No standalone claim.

## C12. Vectorized packed-nibble loads — MICROBENCH_NOW (prototype built)

`fp4_decode_vec` (`dee.cpp/experiments/t4_kernel_next/fp4_decode_vec.cu`):
4× uint32 loads per 32-element block, one scale fetch, identical
`__float2half_rn(table[nib]*s)` per element → **EXACT_PROVABLE** (bitwise
verified by the binary's memcmp gate). Saves 0 traffic bytes; wins only via
fewer instructions/better coalescing on the decode pass. Small, safe, and
the single best first T4 measurement (FINAL_SUMMARY Q9). Production touch on
success: `cuda_convert.cu` decode kernel only.

## Ranked order for Codex

1. **C12** (MICROBENCH_NOW) — run `kaggle_microbench_fp4_decode.py` next.
2. **C2** (MICROBENCH_NOW) — bitwise A/B of m=4096 vs 2×m=2048 on T4.
3. **C1** (PROMISING_BUT_COMPLEX) — needs F2 oracle gate + a fast SM75 kernel.
4. **C8** (PROMISING) — host-side batching after C2.
5. **C7** (PROMISING_BUT_COMPLEX) — after C1 exists.
6. C4/C9/C11 LOW_VALUE-or-mechanism. C3/C6 already done. C10 REJECT_NUMERICS.
   Native FP4 GEMM / `cp.async` / FP8 paths: REJECT_T4 (no SM75 hardware).
