# RECALIBRATION.md — the critical-path model after the host-reuse evidence

Branch: `research/exact-staging` (worktree `../dee-staging`) · Date: 2026-09-04
Replaces: the aggregate draft `recalibrate_frontier.py` (removed; superseded).
Ingests: seal `14864034a7354e0e29e11c1c09f18b0863afe6a0`
(`codex/host-reuse-evidence`): original matched A/B, independent replication,
valid profiled candidate, plus sealed v65, the exact-staging causality audit,
and Muse's `research/route-pipeline` lead=0 result (read-only).
Every seal input is sha256-gated before the pipeline runs.

Reproducible:

```bash
python research/exact-staging/recalibrate_model.py
# -> results/recalibrated_model.json, memory_frontier.json,
#    host_pack_frontier.csv, next_ab.json, pread_kaggle_package.json
```

---

## 1. Why the old model predicted −12.3% when measurement says −0.94%/−2.63%

The old event simulator (`research/exact-critical-path` @ `2db0fde`) modeled
buffer reuse as `pack_ms 2.855 -> 0.05 ms/record` on every pack-sourced
transfer, applied **serially inside the per-layer dependency chain**. That
structural choice alone produces −12.3% (measured: −0.94% original pair,
−2.63% replication). Four specific defects, all checkable against the new
seal (no scalar correction factor was used — the mechanism was removed and
the model refit):

1. **Pack copy charged as a serial batch-blocking stage.** 3,215 decode
   transfers × 2.855 ms ≈ 9.2 s of modeled chain. Measured fill telemetry
   (identical counters in all five seal runs) shows the fill worker pool
   **pipelines** those copies: `fill_worker_ms` ≈ 144 s (worker-time) vs
   `fill_batch_wall_ms` ≈ 54.6 s (wall) on cuda0, with `fill_overlap_ms`
   ≈ 89.5 s already overlapped. Most of the charged time was never on the
   critical path.
2. **Reuse was modeled as copy elimination; measured: it is not.** The
   profiled candidate ran WITH reuse enabled and still shows
   `mmap_to_pinned` = 12.38 s (full run, 4,444 copies = 2.786 ms/copy —
   the same per-record cost the old model thought reuse would remove) and
   `fill_batch_wall_ms` unchanged within run spread (54.63 → 55.76 s cuda0;
   42.64 → 42.97 s cuda1). What reuse removes is buffer
   **allocation/reservation** (`reused_fill_buffers` 1,117 = exactly total
   evictions 708+409; `reused_fill_bytes` = 14.93 GB = evictions × record
   size), and the measured reservation wall (~6.0 s/GPU) sits inside an
   already-overlapped worker pipeline.
3. **v52-derived pack rate leaked profiler inflation.** The 2.855 ms/record
   constant came from the v52 stage-profiled run whose decode wall is
   inflated 138.7 s vs 72.3 s unprofiled. The old simulator then double-
   counted: the calibrated disk-drain term already carried the bottleneck,
   so deleting the (overlapped) pack busy-time surfaced as a large
   counterfactual "win".
4. **Async completion treated as blocking.** The event model advanced the
   consumer to `pack_end`/`h2d_end` per batch; the engine's consumer waits
   per layer while the fill pool runs ahead asynchronously (the pack exists
   precisely to decouple them).

The decisive natural experiment inside the seal: **all five runs have
byte-identical cache counters (hits/misses/evictions) and identical
per-step read/copy tables — reuse changes no cache behavior — yet the
candidate wins only 0.67–1.93 s, i.e. inside the 1.0–1.9 s baseline
run-to-run spread.** Buffer allocation is simply not a wall-time term of
consequence in this pipeline.

## 2. The recalibrated model

One observational term, deliberately. Aggregate per-step timing cannot
identify which overlapped interval is wall-exposed, so the model does not
pretend to:

    wall_ms(step) = serial_residual_ms + miss_service_ms × ssd_misses(step)

- `miss_service_ms` = **effective wall service per storage miss** (marginal,
  observational; includes correlated allocation/orchestration). NOT raw disk
  bandwidth, NOT decomposed latency.
- Fit **orthogonally (total least squares)** on v65's 15 decode steps — both
  regressor and response carry noise — with physicality gates enforced
  (slope>0, intercept>0; the run fails otherwise).
- **No future-layer lead term** (`ordinary_future_layer_lead = 0`, enforced
  at predict time). **No host-buffer-reuse lever** (measured: none).
- Hash layers 0–2 early staging (≤18 records/token, Muse candidate A) is the
  only legal early-staging special case and is NOT modeled here.

### Calibration (fit) error — sealed v65

| Quantity | Value |
|---|---|
| miss_service_ms | 48.679 ms/miss (CI95 ±5.6% of slope) |
| serial_residual_ms_per_step | 754.74 ms |
| R² (15 steps) | 0.9785 |
| predicted decode wall | 72.267 s |
| observed decode wall | 72.267 s |
| fit total error | 0.0% (by construction) |
| per-step MAPE | 2.52% |

### Validation (held out) — the five host-reuse seal runs

These runs were never fit. They share v65's exact per-step reads/copies
tables (verified in `results/seal_host_reuse/derived_per_step_decode.json`),
so they validate the model's **level**, with only run-to-run noise varying:

| Run | Predicted | Observed | Total error | Role |
|---|---|---|---|---|
| ab_baseline | 72.267 | 71.362 | +1.27% | held out |
| ab_candidate | 72.267 | 70.691 | +2.23% | held out |
| rep_baseline | 72.267 | 73.213 | −1.29% | held out |
| rep_candidate | 72.267 | 71.284 | +1.38% | held out |
| profiled_candidate | 72.267 | 71.179 | +1.53% | held out |

Held-out total error ≤2.3% across all five. Baseline run-to-run spread
(same config, three baselines): 71.362–73.213 s → **1.85 s**; the model's
level sits inside that spread. Lane-count neutrality (v63/v64/v65: 71.3/71.8/
72.3 s across 3-vs-6 lanes) is consistent by construction (wall has no lane
term at saturation).

### What the recalibrated model now says about buffer reuse

Predicted reuse delta: **0.0 s** (no mechanistic lever). Measured deltas
(−0.671 s, −1.929 s) sit inside baseline run noise. The A/B
direction was correctly guessed by intuition, but the effect is not
distinguishable from noise at n=2 pairs — the model says the same thing
for once.

## 3. Relation to other tracks (no duplication)

- `research/exact-critical-path` @ `2db0fde`: its event simulator remains
  useful for ranking *large* structural scenarios, but its reuse/lead
  counterfactuals are retracted here (RECALIBRATION §1; CAUSALITY_AUDIT.md).
  This branch does not modify it.
- Muse `research/route-pipeline`: independently proves the lead=0 spine and
  identifies hash-layer (L0–2) early staging as the only legal in-step
  early-staging case. Ingested as a causality constraint; not duplicated.
- The pack-replay instrument (`pack_replay`) is unchanged and revalidated
  below.

## 4. Limitations

- `miss_service_ms` is observational; until the pread package runs, it
  conflates device service with correlated orchestration.
- Trace-local: 15 decode tokens, one prompt, one host class (2×T4).
- Held-out runs validate level only (identical reads tables).
- Continuous peak system memory is UNMEASURED (final + per-token checkpoints
  only).
- SIMULATED ≠ measured. No TPS claims anywhere in this track.
