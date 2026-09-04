# Live microbench protocol (A/B/C mechanics)

Runner: `dee.cpp/experiments/route_pipeline/kaggle_runner_abc.py` — runs
independent of the campaign and checkpoint-free (all bytes synthetic and
labeled; shared shape ASSUMED 4096×2048 fp16 and labeled).

## Cases and what each proves

- **A (hash staging mechanics)**: 18×12.75 MiB file reads + pinned H2D,
  serial vs batched-submit. Proves the submit path can actually hide reads;
  pairs with `hash_stage.py` telemetry (`ready_before_demand`) for the
  exactness half. Reports `route_table_realism:
  ROUTE_TABLE_REALISM_NOT_MEASURED` — the runner measures transfer
  mechanics with synthetic bytes only; it never sees real per-layer
  `tid2eid` rows and its IDs are not route evidence.
- **B (contention)**: shared-shaped GEMM alone vs under H2D saturation.
  Produces efficiency `c` consumed by `shared_overlap.OverlapWindow`.
  Decides whether shared hides under staging (expected) or contends.
- **C (barrier cost)**: `cudaStreamSynchronize` vs event record/wait.
  Bounds the prize for narrowing the per-layer sync and route-D2H sync from
  both sides (if events ≈ syncs, narrowing buys nothing — reject).

## Emission (every case)

GPU name, SM (+override flag), fixture identity (synthetic/assumed labels),
correctness status, baseline + candidate timings, p50/mean/min, sync count,
event count, bytes moved, overlap duration, status. Refuses no-CUDA hosts;
refuses non-SM75 without `--allow-non-sm75` (dev-labeled, never T4 evidence).

## Admission (before benchmarking; campaign tolerances unchanged)

- A: exact IDs (table-gated) + evidence-gated weights + identical bytes +
  unchanged consume order (host tests must pass; excluded_fraction-style
  harness-validity metrics are never used as candidate-error evidence).
- B: identical branch outputs + documented join order + exact final output.
- C: identical output + no premature reuse (negative tests) + deterministic
  completion + no lifecycle leak.
