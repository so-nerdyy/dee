# NEXT_EXPERIMENTS.md — ranked live A/B queue for Codex

Basis: calibrated model (`results/calibration.json`), simulated sweeps
(`results/sweeps.json`), sealed evidence (EVIDENCE_INVENTORY.md).
"Upper bound" = simulator's idealized case; "modeled gain" = realistic case.
Every gain is a SIMULATED value to be confirmed by a matched A/B.

---

## RUN_NEXT

### N1. Intra-step earlier expert staging (staging lead ≥ 8 layers)

- **Code area**: expert staging/fill pipeline (dee.cpp engine: host fill
  workers + prefetch ring submission; `source_read_lanes`/queue are already
  sufficient). Submit layer L's expert reads when the consumer reaches
  layer max(0, L−8), instead of layer L−1.
- **Hypothesized bottleneck**: saturated SSD drained too late — 459 MB/s
  capability vs 232 MB/s realized during decode (v65 byte accounting).
- **Measured evidence**: v65 source_read_overlap 77.9 % (reads mostly
  serialized after demand); v52 timeline shape (on-demand staging);
  per-step storage_bytes (0.75–1.67 GB) vs measured wall.
- **Simulator upper bound**: −26.4 % decode wall (52.6 s floor = disk roof).
- **Realistic modeled gain**: −26.1 % at lead 8; saturates by lead 8–16;
  robust across the disk-rate uncertainty band.
- **Implementation risk**: medium — fill-worker scheduling change; ring
  slots 41/33 today vs ~16 in flight at lead 8 (13.4 MB × 2 GPUs × 8 layers
  ≈ 214 MB) fits.
- **Correctness risk**: none to outputs — staging order is not observable in
  tokens (same records, same kernel, same combine); route journal unchanged.
- **Profiler counters required**: `source_read_overlap_percent`,
  `source_read_batch_wall_ms`, per-step `storage_bytes` vs `wall_ms`,
  `waiting_cache_entry_readiness` (should → 0), host_pack evictions.
- **Matched A/B design**: same commit except lead constant; 3+3 paired runs
  alternating baseline/candidate on the sealed prompt; gate = exact token
  equality + decode wall median + sub-5 % spread; expect candidate median
  ≈ 52–58 s. (52.6 s is the disk roof; landing there validates the model.)

### N2. Re-seal the v63/v64-style run-to-run noise band

- **Purpose**: N1's effect (−26 %) is 30× the observed 0.7 % noise between
  v63/v64/v65, but the claim rule requires a clean gate. One extra sealed
  control run (v65 config, unmodified) establishes the current session's
  spread for the paired comparison.
- **Cost**: one run. **Risk**: none. **Counters**: decode_wall_s only.

## WORTH_PROTOTYPING (after N1 lands)

### N3. Host staging-buffer reuse (Codex's active A/B), re-run post-N1

- **Status**: Codex's live A/B is running against the **on-demand** pipeline;
  the model predicts −12.3 % there. After N1, the modeled effect shrinks to
  ≈ −0.3 % (52.79 → 52.58) because staged-early reads no longer wait on the
  pack path. Ingest via `tools/ingest_ab.py`; re-run the A/B after N1.
- **Risk**: low. **Counters**: reuse telemetry, `mmap_to_pinned` ms,
  per-step wall.

### N4. Fill-worker → H2D handoff reduction (persistent pinned staging slots)

- The model folds this into buffer_reuse; prototype only if N1's measured
  gain lands below 52.6–58 s and `waiting_cache_entry_readiness` remains > 0.
- **Upper bound**: 0.3 % post-N1. **Not recommended pre-N1** (subsumed).

## LOW_EXPECTED_VALUE (model says skip; kept for the record)

- **Grouped H2D per layer** (−0.07 %): records are 13.4 MB; per-copy overhead
  is invisible. Modeled and measured evidence both say no.
- **More storage lanes** (0.0 %): disk saturates; sealed v63/v65 A/B already
  measured neutrality at 3→6 lanes; 12 lanes modeled identical.
- **Grouped expert dispatch / device worklists** (−0.5 %): host slices total
  ~1 % of wall. Correctness-sensitive (batched-path gate) for negligible gain.
- **H2D stream count** (−0.07 % at grouped, ≤ −5 % even at 2× rate): PCIe is
  not the constraint.
- **Pinned-ring depth** (0 %): idle4 sweep 8/12/16/24 measured flat.

## BLOCKED_BY_MISSING_MEASUREMENT

- **B1. True concurrent store capability (MB/s) at depth ≥ 8**: the
  −26 % floor's location depends on the fitted 320 vs derived 459 MB/s band.
  A 5-minute microbench on the Kaggle host (pread 13.4 MB records, depth 1/2/
  4/8/16, page cache cold) would pin the roof and shrink N1's prediction
  interval to ±2 %. **Highest-value measurement if the model stays ambiguous.**
- **B2. GPU dense/non-MoE time per token**: UNKNOWN in all sealed profiles;
  bounds the post-N1 floor from below. Requires a profiled run with
  `profile_stages: true` (v52-style) — costs ~2× wall, run once.
- **B3. Effective H2D under dual-GPU + compute contention**: fitted 5.54
  GB/s ≈ microbench 5.38 GB/s; if N1 lands far below prediction, measure
  this next (cudaEvent around staged copies at depth).
