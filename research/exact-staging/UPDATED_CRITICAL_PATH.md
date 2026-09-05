# UPDATED_CRITICAL_PATH.md — what the critical path looks like after recalibration

Branch: `research/exact-staging` · Date: 2026-09-04
Prior model: `research/exact-critical-path` @ `2db0fde` (untouched on its branch).

## 1. Current critical path (measured anchors)

| Stage | Measured | Share of 72.3 s | Label |
|---|---|---|---|
| Storage-miss wall service | 48.68 ms/miss × 1,252 | ~61 s effective | measured regression slope (observational) |
| Non-storage serial residual | 754.7 ms/step × 15 | ~11.3 s | measured intercept |
| — of which H2D device time | 5.07 s (profiled, full run) | ≤7% | measured CUDA events |
| — of which expert compute | ~0.92 s (profiled) | ~1.3% | measured CUDA events |
| — of which mmap→pinned copies | 12.38 s (profiled, full run) | overlapped, NOT additive | measured but pipelined |
| Host orchestration/syncs | inside residual (sync profile 6.72 s full-run) | partially overlapped | measured, exposure UNKNOWN |

Consistency checks: the slope's implied per-miss service (48.7 ms over
~250 MB/s aggregate ≈ 12.75 MiB/record) matches the v65 byte-accounting
band (232–459 MB/s effective vs 320–459 capability) without any new free
parameters. Muse's profile cross-check (device-busy ≈ 3.5 s of 71 s wall)
independently confirms GPU compute is nowhere near dominant.

**Storage remains dominant. H2D and compute are not. Host serialization is
second-order but real (the 754.7 ms/step residual), and its composition is
the largest remaining UNKNOWN — attackable only with Muse's A/B/C
microbenches, not by this model.**

## 2. What changed vs the old model

| Claim | Old model (2db0fde) | Recalibrated (evidence) |
|---|---|---|
| host-buffer reuse | −12.3% (modeled) | **0 modeled**; measured −0.94%/−2.63% ≈ run noise; reuse removes overlapped allocation, not copy |
| future-layer staging lead ≥8 | −26.1% (modeled) | **RETRACTED as implementable** — IMPOSSIBLE_WITHOUT_FUTURE_ROUTE (CAUSALITY_AUDIT.md; Muse lead=0). Kept only as a labeled theoretical ceiling. |
| hash layers 0–2 early staging | not modeled | legal special case (Muse A), ≤18 records/token (~240 MB); prize UNKNOWN until Muse runner case A measures hidden ms |
| pack capacity | not a lever (fixed 682) | **the only large legal lever** — see HOST_PACK_FRONTIER.md |

## 3. Updated lever ranking (all SIMULATED unless marked measured)

1. **Host-pack budget 8.5 → 9.5–10.0 GiB/GPU (config-only)** — replay
   −42…−51 misses; simulated −2.1…−2.5 s (central), CI95 ±5.6% of the
   miss term; SAFE in the 32 GB envelope (10.0) / comfortable (9.5).
   `RUN_NEXT` — the one A/B in `next_ab.json`.
2. **Hash-layer (L0–2) early staging** — legal, exact, host-side only
   (Muse candidate A); ≤18 records/token ⇒ upper bound ~18×48.7 ms ≈
   0.9 s/step-class IF every record would otherwise stall (unlikely;
   overlaps with fill pool). `WORTH_PROTOTYPING` via Muse's live
   microbench, not via this model.
3. **Host residual decomposition (sync cost, launch serialization)** —
   ~0.75 s/step residual whose split is UNKNOWN; Muse candidates B/C
   measure it. `BLOCKED_BY_MISSING_MEASUREMENT` (runner cases B/C).
4. **Buffer-reuse-style allocation work** — measured ≈ noise.
   `LOW_EXPECTED_VALUE`.
5. **More storage lanes / H2D streams / grouped transfers** — measured
   neutral (v63/v64/v65) or ≤5% even at 2× H2D. `LOW_EXPECTED_VALUE`
   under on-demand staging.
6. Theoretical ceiling (NOT implementable): future-layer lead ≥8 →
   −26% class. Preserved as a ceiling only; falsified as a plan.

## 4. Open measurement that would move this model most

True concurrent pread capability of the trace store (queue depths 1–16):
package ready in `results/pread_kaggle_package.json` for Astra. It pins the
position of the storage floor (±2% class) and separates device service from
the orchestration currently folded into `miss_service_ms`.
