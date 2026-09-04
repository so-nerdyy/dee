# Microbench protocol + correctness contract (Phases F + H)

## H. Runner (built)

`dee.cpp/experiments/t4_kernel_next/kaggle_microbench_fp4_decode.py` +
`fp4_decode_vec.cu`. Codex flow: fresh notebook (never the campaign
checkout) → copy the two files → `python3
kaggle_microbench_fp4_decode.py --workdir /tmp/t4kn --iters 200 [--shard
<official-shard>]`. The script verifies SM75, compiles with
`nvcc -O3 -arch=sm_75` (no dee build), extracts one real w1 fixture
(header-only safetensors parse, sizes verified) when a shard is mounted,
runs three cases (synthetic w1-shape, real w1 bytes if present, synthetic
down-shape), requires the binary's memcmp gate, warms up (20), benchmarks
(200 default), and writes JSON. Bounded device memory (~120 MiB). No
full-model generation. Exit nonzero on any gate failure.

## F. Correctness contract (admission gates — defined BEFORE performance)

Uses existing dee reference semantics only. No tolerance is changed by this
track. Do NOT reuse the broken candidate-independent sample-validity logic
for numerical attribution (it measured harness validity, not kernel error).

- **F1 bitwise gate (decode-class kernels: C12).** Candidate output must be
  `memcmp`-identical to the production-algorithm scalar port over: all-zeros
  packed, all-`0xFF` packed, seeded-random packed × all 256 scale bytes
  (exhaustive scale sweep is only 256 values — run it), plus one real expert
  fixture when mounted. Any diff → REJECT, no timing admitted. Rationale:
  identical op order + identical rounding per element makes bitwise equality
  achievable and therefore required.
- **F2 oracle gate (order-changing kernels: C1/C2/C7/C8).** Compare against
  the trusted FP32 reference (`scripts/deepseek_v4_expert_reference.py`
  semantics): report max_abs, relative-error percentiles (p50/p99/p100),
  cosine similarity, finite-rate (must be 1.0), and determinism across 5
  repeats (bitwise repeat required). Admission thresholds are the EXISTING
  dee contract values — this track sets none. A design that cannot meet the
  existing contract is REJECT_NUMERICS, however fast.
- **F3 fusion-invariance gate (C2/C10-class).** cuBLAS-shape changes
  (m=4096 vs 2×m=2048) and any alpha/beta folding start at F1; on mismatch
  they fall to F2, and F2 failure is terminal for that design.
- **F4 harness-validity separation.** Fixture generation, warmup adequacy,
  clock stability, and first-launch algorithm selection are reported
  alongside numbers but never subtracted from error metrics.

## Benchmark discipline

Warmup ≥20 (absorb cuBLAS selection + clocks), timed reps ≥200, cudaEvent
timing on-device, report median + p99 (not just mean), device name + SM +
clocks (locked if possible) in every JSON record. Compare kernel time only;
never convert a kernel-time delta into Tok/s. One variable per experiment
(shape, then occupancy config, then fusion scope).
