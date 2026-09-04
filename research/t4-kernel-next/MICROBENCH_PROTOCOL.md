# Microbench protocol + correctness contract (Phases F + H)

## H. Runners (built)

Single-test runner (C12 control, UNCHANGED):
`dee.cpp/experiments/t4_kernel_next/kaggle_microbench_fp4_decode.py` — see
previous revision; preserved as-is.

Three-way runner (new):
`dee.cpp/experiments/t4_kernel_next/kaggle_microbench_three_way.py` builds
`fp4_decode_vec.cu` (untouched), `c2_gate_up_stack.cu` (`-lcublas`), and
`c1_tile_consume.cu`, then runs the `c12` / `c2` / `c1pre` cases
independently. Each case: `--verify-only` correctness gate first (timing
inadmissible on gate failure), then `--reps` timed repetitions (default 5 ×
200 iters). Every record carries GPU name, SM version, fixture identity
(tensor names + sizes, or `synthetic`), dimensions, packed bytes,
decoded/intermediate bytes, launch counts, correctness result, warmup (20,
binary default, reported not tweaked), repetitions, and p50/mean/min timing
across reps. Refuses non-SM75 unless `--allow-non-sm75` is passed, in which
case results are labeled development-only, never T4 evidence.

## F. Correctness contract (admission gates — defined BEFORE performance)

Admission levels used by all three prototypes (printed in every JSON
`admission` field):

- `BITWISE_IDENTICAL`: memcmp-clean vs the production-algorithm reference.
  Required before the word "exact" is used.
- `ORACLE_PASS_ONLY`: bitwise failed but the F2 oracle gate (below) passes
  at unchanged tolerances. Admissible for performance discussion, never
  called exact, never integrated without a follow-up review.
- `NUMERICALLY_DIFFERENT`: any mismatch without an oracle pass. Timing
  inadmissible for claims; design is REJECT_NUMERICS pending F2.

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

## Real fixture support (Part 4)

The three-way runner extracts W1+W3 packed+scales for one `(layer, expert)`
(default 0/155) via header-only safetensors parse with per-tensor size
verification (4,194,304 B packed + 262,144 B scales each); missing tensors
(e.g. dense/hash layers) fall back to labeled synthetic rather than failing.
C12/C1-PRE consume the W1 pair (C1-PRE rows=2048 in real mode); C2 consumes
the W1+W3 pair. Synthetic fixtures remain valid software tests and are
labeled `"fixture":"synthetic"` in every record; the full checkpoint is never
required.

## Benchmark discipline

Warmup ≥20 (absorb cuBLAS selection + clocks), timed reps ≥200, cudaEvent
timing on-device, report median + p99 (not just mean), device name + SM +
clocks (locked if possible) in every JSON record. Compare kernel time only;
never convert a kernel-time delta into Tok/s. One variable per experiment
(shape, then occupancy config, then fusion scope).
