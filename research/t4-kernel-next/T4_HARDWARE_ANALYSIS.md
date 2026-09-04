# T4 / SM75 hardware grounding (TU104, compute capability 7.5)

Values are established TU104/T4 specifications; the Kaggle runner re-queries
name, compute capability, and memory at runtime (`check_sm75`) and refuses
non-7.5 hardware rather than trusting this document. Do not assume any
Ampere/Blackwell instruction; do not reuse KT CUDA assumptions that need
sm_80+.

- **SMs/cores**: 40 SMs, 2560 FP32 CUDA cores (64/SM), 1024 threads/SM max,
  32 blocks/SM max, 64K 32-bit registers/SM, warp size 32 (unchanged).
- **Tensor cores**: 2nd generation, 320 total. FP16 (and INT8/INT4) HMMA
  only — `mma.sync.aligned.m8n8k4` FP16 exists, `wmma` FP16 works. **No FP8,
  no FP4, no structured sparsity, no `cp.async`, no distributed shared
  memory, no TMA, no wgmma.** Consequence: packed FP4 storage can never
  execute natively; every design must decode to FP16 (or INT8) first, and
  the decode itself is integer/bitwise + table lookup, not tensor work.
- **Throughputs (peak, bounds only)**: FP32 ~8.1 TFLOPS; FP16-tensor
  ~65 TFLOPS; INT8 ~130 TOPS; INT4 ~260 TOPS. Decode path is DRAM-bound, so
  these matter only for C1 kernel sizing (tensor-core GEMV is NOT the
  bottleneck at batch-1 — 16.78 MB streams dominate).
- **Memory**: 16 GB GDDR6, 256-bit, **~320 GB/s** peak; 4 MB L2. At batch-1
  the expert path streams with almost no reuse, so effective bandwidth is
  the roofline input (use measured, not 320, once the microbench reports).
- **Shared memory**: 64 KB per SM (48 KB per block max), 96 KB L1/shared
  carve-out class. A 32-element FP32 tile + scales fits trivially; full
  2048-element rows (8 KB FP32 / 4 KB FP16) also fit — C1 tiling is
  unconstrained by shared capacity. Register pressure is the real C1 bound
  (accumulators per thread × unroll).
- **PCIe**: Gen3 x16 (~12–13 GB/s achievable). A 12.75 MiB packed miss costs
  ~1 ms floor on the wire — consistent with the measured ~2.6–4.0 GiB/token
  H2D budgets in the DSV4 campaign. H2D/prefetch overlap (already in the
  runtime) matters more than any kernel for misses.
- **Launch overhead**: ~5–10 µs per launch on this class of host; ~1,800
  launches/token ≈ 10–20 ms/token of pure overhead ceiling — this is what
  C2/C7/C8 attack. Small-batch cuBLAS calls additionally pay algorithm-
  selection cost on first shapes (warmup in the protocol absorbs it).
- **Occupancy notes for C1**: gate/up rows (2048) and down rows (4096) with
  256-thread blocks give 8/16 blocks per GEMV-row-equivalent — far below
  the 40-SM fill; a fused kernel must parallelize over (row × tile), not
  rows alone, or most SMs idle. The existing `fp16_gemv_fp32_kernel`
  (one block per row) has the same property and survives only because it is
  DRAM-bound; a fused kernel must preserve that property while adding
  decode ALUs per thread.
- **What is impossible/poor on T4 specifically**: native FP4/FP8 GEMM
  (no units); `cp.async`-pipelined decode (no instruction); large-shared
  persistent kernels beyond 48 KB/block; INT8-tensor execution of e2m1
  values without a numerics review (different rounding domain → needs the
  F2 gate, not assumed).
