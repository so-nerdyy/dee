# T4 kernel-next — final summary

Track: isolated CUDA kernel research, SM75 only. No production file touched
(only `research/t4-kernel-next/`, `dee.cpp/experiments/t4_kernel_next/`,
`tests/` added); no semantics/tolerance change; no approximate quantization;
nothing wired in; KT assumptions not reused; Codex's host-buffer A/B
untouched; no merge.

## Continuation: C2 + C1-PRE prototypes (built, T4-gated, ungated locally)

C12 preserved unchanged as the control. New: `c2_gate_up_stack.cu`
(stacked-decode check + m=4096 vs 2×m=2048 bitwise A/B, `-lcublas`) and
`c1_tile_consume.cu` (tile-local decode→dot with identical accumulation
order, BITWISE-gated, bytes-avoided accounting), plus
`kaggle_microbench_three_way.py` (SM75-refusing, `--allow-non-sm75`
override, real L0/E155 W1+W3 fixture extraction with synthetic fallback,
per-test p50/mean/min over reps). New docs: `C2_GATE_UP_FUSION.md`,
`C1_TILE_CONSUME.md`. Host tests: 19 passed
(`test_t4_kernel_next*.py`), including order-identity mirrors and the Part 5
accounting (base hit path 114,032,640 B; C1-PRE avoids 100,663,296 B/expert
at scale).

## Ranking: C12 vs C2 vs C1-PRE

| Axis | C12 vec decode | C2 stacked GEMM | C1-PRE tile consume |
|---|---|---|---|
| Expected impact | tiny (instructions only) | small (1 launch/expert) | large (up to ~7× decode+GEMM traffic cut at full scale) |
| Implementation risk | nil (drop-in kernel) | low (host-side) | high (full fused GEMM still to be written) |
| Numerical risk | none (bitwise by construction) | medium (cuBLAS shape-dependent algorithm; gated) | low for premise (identical order); high for full C1 (order will change) |
| T4 suitability | perfect (coalesced, plain ALU) | perfect (same cuBLAS) | good (FMA/shared, no exotic ISA; occupancy needs rows) |
| Bytes avoided | 0 | 0 | ~100.7 MB/expert at scale |
| Launches avoided | 0 | 1/expert (7→6) | 1 per row-segment (decode launch absorbed) |
| Next live measurement | control timings from three-way run | `admission` field decides: BITWISE→integrate, else F2-or-reject | `bitwise_identical` retires premise risk; then scope full C1 kernel |

Strategic read: C12 is the control, not the prize — no further tuning
regardless of its timing. C2 is a cheap gated win. C1-PRE is the decision
point for the only change that removes the 50.3 MB/expert repeated
materialization. The single most informative live number remains the
three-way runner's report (kernel times + three admission fields), never a
TPS translation.

## The 10 required answers (original audit; still current)

1. **Biggest bandwidth waste: the per-use full FP16 decode.** 50.33 MB
   written + 13.37 MB re-read per expert *use* (hit or miss), larger than
   the 50.33 MB of GEMM weight reads it feeds (`MEMORY_TRAFFIC.md` B1/B2).
2. **Decoded/intermediate materialization: ~100.7 MB per expert** (50.33 MB
   FP16 scratch + 50.33 MB GEMM re-reads of that scratch); gate/up/act
   intermediates are ~12 KB at batch-1 — noise.
3. **Traffic-removing kernel: fused decode-consume (C1)** — up to ~7× less
   traffic on the decode+GEMM portion (floor ~14 MB vs ~114 MB per hit).
4. **W1/W3 fusion: yes, as one m=4096 GEMM — but EXACT_NEEDS_TEST** (cuBLAS
   algorithm choice may differ by shape; bitwise A/B decides; host-only
   change, microbench now).
5. **Activation fusion: already fused** — clamp+SiLU+multiply with torch FP16
   rounding boundaries in one kernel; nothing left to fuse (C10-class merges
   would delete measured rounding boundaries → REJECT_NUMERICS).
6. **Output-stationary W2: no standalone value** — the down GEMV already
   streams once; it only wins inside C1.
7. **Persistent execution: worthwhile only after C1 exists** (C7 is a
   launcher for a proven fused kernel, not a first step); ~1,800
   launches/token is real (~10–20 ms overhead ceiling) but each item is
   traffic-heavy, so host batching (C8) comes first.
8. **Impossible/poor on T4:** native FP4/FP8 GEMM, `cp.async` pipelines,
   >48 KB/block shared designs, INT8-tensor e2m1 without a numerics review.
9. **Single best next microbench:** `kaggle_microbench_three_way.py`
   (C12 control + C2 + C1-PRE with real L0/E155 bytes, gates first, JSON
   with p50/mean/min) — one Codex run answers all three admission fields.
10. **Production touch list on success:** C12 → `cuda_convert.cu` (decode
    kernel only); C2 → `swiglu_cuda.cu` (`swiglu_expert_batch_fp16_cuda`) +
    stacked decode-target layout in `engine.cpp`; C1-PRE → premise only
    (no touch); full C1 → new kernel file + `cuda_convert.cu` + `engine.cpp`
    decode seam; C8 → host-side `engine.cpp` batching + `async_prefetcher`;
    none until gates pass.

## Verification performed here (no T4, no working nvcc)

- `pytest tests/test_t4_kernel_next*.py` — 19 passed: traffic arithmetic,
  vec-vs-scalar index equivalence, C2 stacked-decode + concat-identity
  mirrors, C1 order-identity mirror, prototype/runner/source static checks.
- Honesty notes: T4 specs are established values with runtime re-query in
  the runner; roofline numbers are upper bounds, never TPS; the .cu
  correctness claim is "bitwise by construction, memcmp-gated on T4" — the
  gate has not run yet, which is exactly what Codex's run will do.
