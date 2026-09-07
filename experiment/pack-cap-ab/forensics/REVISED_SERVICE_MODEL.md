# REVISED_SERVICE_MODEL.md — what predicts decode wall if miss count doesn't

Branch: `experiment/pack-cap-ab` · Date: 2026-09-07
Machine-readable: `results/revised-service-model.json`

## 1. The old model, falsified prospectively

`research/exact-critical-path` (recalibrated) fit, on sealed v65 variants:

    wall = intercept + 48.679 ms × miss_count          (R² 0.978 in fit)

The pack-cap A/B is a **natural experiment** that holds everything else
fixed and removes exactly 51–52 misses:

- Predicted wall gain: 48.679 ms × 52 = **2.53 s**
- Observed mean B−A: **+0.03 s** (session deltas +2.27/−2.21, sign set by
  arm order, not cap)

**The global constant-per-miss model is rejected as a causal model.** It
survives only as a retrospective correlation: in the v65 calibration set,
miss count co-varied with lane/queue-depth structure, and the coefficient
absorbed that confound.

## 2. Observational comparison across 60 token-observations
(4 arms × 15 decode tokens; exploratory — token latencies are serially
dependent within a run; no inferential claim is made)

| Predictor | Pearson r | Spearman ρ | OLS R² | slope |
|---|---|---|---|---|
| host-pack miss count | 0.933 | 0.907 | 0.870 | 53.6 ms/miss |
| storage bytes (GB) | 0.933 | 0.907 | 0.870 | 4.01 s/GB |
| **measured read-batch wall (s)** | **0.971** | **0.971** | **0.942** | **0.384 s/s** |

Miss count and bytes are the same signal (records are uniform-size) and are
highly correlated with wall across tokens — but the pack-cap experiment
shows that correlation is **not causal**: −52 misses with ~0 wall change.

The measured read-batch wall (`source_read_wall_ms`, i.e., wall time with ≥1
read batch in flight during the token) predicts per-token wall best, with
slope ≈ 0.38: only ~38% of an added second of read-batch exposure reaches
the wall; the rest overlaps compute/H2D on the other GPU or within-token.

Demonstrator (same miss count, wildly different wall): 61-miss tokens span
3.54–5.08 s across arms (ratio 1.61×); token 3 vs token 8 in s1A: 80 vs 80
misses, 6.10 vs 3.43 s.

## 3. Why −52 misses ≈ 0 wall: the batch-structure arithmetic

- Reads execute in batches of up to qdepth 6 on 3 lanes. Batch wall ≈ the
  **max** of its concurrent reads (p50 read 105–108 ms, p95 ~164 ms).
  Removing 1–2 reads from a ~3.4-read batch almost never shortens the
  batch's critical read.
- 52 reads over ~764 batches (413 cuda0 + 351 cuda1 in arm A) ≈ touches
  ~6% of batches, most of which keep their max read.
- 62% of read-worker time was already overlapped; the exposed portion is
  what reaches the wall, and the eliminated reads were disproportionately
  inside overlapped windows (sign-flip tokens).
- The rescued records are long-recency re-reads (873–1900 distinct-records
  distance); their re-demands were routinely served by the prefetch ring
  (41 slots, 2613 prefetches issued vs 2285 cold loads) ahead of demand.

## 4. The revised observational model

    wall_token ≈ floor_token + κ · max(0, read_batch_wall_token),  κ ≈ 0.38

with `floor_token` = compute + H2D + host sync (token-specific, position-
and session-stable to ±0.1–0.2 s on tokens 1–11). This model explains:

- why −52 misses yield ~0 wall (they barely reduce read-batch wall);
- why the tail-token position effect moves the wall (it moves read-batch
  wall by −1.1…−3.6 s/token → wall by ~0.4× that);
- why the old ms/miss coefficient over-predicted (it charged serial read
  time for reads that were batched/overlapped).

## 5. What would sharpen this model

1. One profiler-enabled arm (stage profiling ON): `host_waits.cache_readiness`
   and `cache_readiness_attribution` give per-request blocking reasons —
   converts UNKNOWN per-request criticality into measured critical wait.
2. Per-request read submit/complete timestamps (the staging-track telemetry
   design already specifies them).
3. Page-cache/fault counters at decode granularity for the order effect.

Until then: **SIMULATED/theoretical claims about ms-per-miss are retired;
only read-batch-wall exposure is a measured predictor.**
