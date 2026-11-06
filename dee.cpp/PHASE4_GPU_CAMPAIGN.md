# Phase 4 — GPU Campaign Preflight (Wave D)

Branch `research/phase4-cache-hierarchy` @ `60943c8` (+ local runner/driver at `8229e08`).
All numbers marked SIM = replay-sim on sealed Phase-3 journals (anchors reproduced
tol=0); MEASURED = GPU-2 evidence; ESTIMATE = projection.

## Hypotheses

- **H1 (capacity lever):** fp4-packed VRAM residency (281 slots vs 74) + pure-recency
  eviction lifts the device hit rate from ~3% (MEASURED) to ~30-35% (SIM 33.3% on the
  sealed window), cutting cold-load H2D bytes/token ~25-35%. On the 128-token
  workload the gain should be larger — repeat-from-prior runs 51-64% and the
  working set keeps growing, so residency pays more over longer streams.
- **H2 (policy lever):** inside fp4 capacity, `lru` beats `rank_priority` by ~23pp
  of hit rate (SIM 33.3% vs 10.0%) — the legacy priority term is actively harmful
  at this operating point (it is descending expert-ID order in the live path).
- **H3 (host-tier value):** the 8.5 GiB/GPU host pack already serves ~46-49% of
  requests (MEASURED 48.9%) — the bypass arm prices its total contribution and the
  serial-fill cost of not having it.
- **H4 (exactness under repair):** all repaired machinery preserves identical
  token streams + route journals vs the reference arm.

## Arm matrix (as implemented in phase4_session_driver.py)

| arm | store | VRAM dtype | evict policy | host mode | host budget | lanes | reset | trace |
|-----|-------|-----------|--------------|-----------|-------------|-------|-------|-------|
| a0_bypass | dee4 | fp16 (74) | rank_priority | **bypass** (1-record bounce) | 13.4 MB | 1 | cold | on |
| a1_asis   | dee4 | fp16 (74) | rank_priority | lru | 8.5 GiB/GPU | 4 | warm/persist | on |
| a2_fp4    | dee4 | fp4 (281) | **lru** | lru | 8.5 GiB/GPU | 4 | warm/persist | on |
| a3_fp4p   | dee4 | fp4 (281) | rank_priority | lru | 8.5 GiB/GPU | 4 | warm/persist | on |
| (stretch) a4_fp16lru | dee4 | fp16 (74) | lru | lru | 8.5 GiB/GPU | 4 | warm | on |

- a1_asis = reference (Phase-3-equivalent config), runs first (fastest fail signal).
- a0_bypass = honest low-cache baseline: every expert access fills through a
  one-record bounce buffer (avoids both caching AND the FUSE mmap-fallback path).
- a2 vs a3 = the policy repair's live value at fp4 capacity (isolates H2).
- a4 (stretch, only if session budget remains): sim predicts fp16+lru < fp16+prio
  at 74 slots (0 vs 162 hits — sticky cohort helps under extreme thrash); a live
  inversion check that makes the policy story capacity-conditional and honest.

Workload: 8 prompts x 128 tokens (factual/code/math/long-form + 3 sealed Phase-3
regression prompts + q0 repeat). NATIVE_RUN_ID constant across arms; same device
split => journals hash-compare across ALL arms including bypass.

## Metrics (per arm, per prompt, per phase)

- Event classes (new): GPU_HIT / HOST_HIT / STORAGE_MISS / MmapFallback per access,
  with evicted identity, occupancy, generation, reuse distance.
- Counters: VRAM hits/cold/evictions, host hits/misses/evictions, store reads,
  H2D copies/bytes, per-forward + per-prompt + cumulative scopes (contamination
  fixed runner-side).
- Perf: wall, prefill/decode split, ITL p50/p95, tokens/s (cold vs warm vs
  steady-state reported separately), fetch p50/p95, fill service vs batch wall.
- Memory: configured budgets + new device ledger (fp4 scratch now counted) +
  host RSS evidence.
- Exactness: generated token sha256 + route-journal sha256 per prompt per arm,
  cross-arm equality vs a1_asis; lookup_failures==0; backend==dee4_segmented.

## Predictions (SIM on sealed 16-token window; 128-token arms will differ)

| arm | VRAM hit | host hit | cold loads | H2D/token |
|-----|----------|----------|-----------|-----------|
| a1_asis (fp16+prio) | ~3% | ~46-49% | ~5,350 | ~4.3 GiB |
| a2_fp4 (fp4+lru) | ~33% | similar | ~3,679 (-31%) | ~2.9 GiB |
| a3_fp4p (fp4+prio) | ~10% | similar | ~4,960 | ~4.0 GiB |
| a0_bypass | ~3% | ~0-3% (bounce) | ~5,350 | ~4.3 GiB + slower fills |

fp4 decode-on-compute overhead ~65 ms/token (ESTIMATE ~1.4% of wall at Phase-3
pace) — net win expected if H1's hit-rate gain materializes.

## Accept / reject criteria

- PASS-minimum: all 8 prompts x all arms ACCEPT_CORRECTNESS + journal/token sha
  cross-arm equality; a2_fp4 VRAM hit >= 20% and H2D/token < a1_asis by >=15%;
  bypass arm completes without mmap-fallback stall.
- STRONG: a2 >= 30% VRAM hit + wall improvement vs a1_asis; a2 vs a3 delta
  confirms the policy repair direction (a2 > a3 by >10pp).
- REJECT-EXACTNESS: any token/journal mismatch, any zeroed/poisoned block
  evidence, any lookup failure => arm FAIL, results quarantined.
- REJECT-PERF: a2 within noise of a1 on wall AND H2D/token (i.e. fp4 capacity
  bought nothing live) => report honestly, pivot to host-tier analysis.

## Why GPU is required (local can't answer)

Sim is counter-validated but cannot measure: real H2D/fill latencies under the
repaired staging-lifetime discipline, the deferred-unpin path's live behavior
(compile-gated on this box), fp4 decode-on-compute cost on T4, page-cache/OS
interactions on the FUSE mount, wall time, or TPS. The 128-token journals also
don't exist yet — a1_asis's trace capture IS the product.

## Budget + mechanics

- T4x2, one Kaggle session. Per-arm ~75-90 min ESTIMATE (8 prompts x 128 tok;
  a0 slower — serial fills) + build/seal/mount ~20-30 min => ~6-7 h total.
  If the session budget is tighter: drop a3 (policy contrast can ride on a2 vs
  a1 cross-dtype + sim) and/or a4 stretch.
- Driver: phase4_session_driver.py (P0 gate -> build -> mount+seal -> microbench
  -> arms -> verdict). Artifacts per arm: result/checkpoint/journal/cache_events/
  integrity/memory/profile/run_config/environment per prompt into p4-out/{arm}/.
- Abort rules: any arm's runner exits nonzero or writes REJECT_* classification
  -> driver logs, harvests artifacts, proceeds to next arm, records FAIL in
  p4_report.json; subprocess timeout 3 h/arm.
- Watcher: a dedicated agent watches the session log for hangs/OOM/disk/
  classification regressions (heartbeat lines every ~60 s are in the driver).
- Commit under test: 60943c8 (engine) + 8229e08 (harness) — recorded in every
  result's `commit` field by the runner.

## Known GPU-verification debts (compile-gated locally)

- `retire_completed`/`cudaEventQuery` slot-retirement path (B0c).
- Deferred-unpin drain path in moe_forward_batch_device_impl (B0d).
- fp4 decode-scratch bounds guard (B0d).
- These get exercised on the FIRST arm (a1_asis fp16) — its result validates the
  repaired machinery before fp4 arms run. If a1 shows anomalies, stop the
  session and fix locally.
