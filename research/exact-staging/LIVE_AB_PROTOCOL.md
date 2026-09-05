# LIVE_AB_PROTOCOL.md — matched A/B: host-pack budget 8.5 → 12.75 GiB

Branch: `research/exact-staging` · Date: 2026-09-04
> **SUPERSEDED (2026-09-04, v2 recalibration):** 12.75 GiB/GPU is classified
> NOT_SAFE_FOR_32GB (MEMORY_BUDGET.md §4) after the seal's measured memory
> anchors. The live A/B is now **LIVE_AB_V2_PROTOCOL.md** (17→20 GiB cap,
> 10.0 GiB/GPU; 19 GiB/9.5 conservative variant). Kept for provenance of the
> original simulation-based suggestion.
Prepared for Codex. **One candidate, one knob, matched everything else.**
All expected deltas below are SIMULATED estimates from the validated replay +
calibrated model (see STAGING_DESIGN.md); the A/B is the only arbiter.

---

## 1. Hypothesis

The production host-pack LRU at 8.5 GiB/GPU (682 records) evicts records that
re-decode demands within 2–3 steps (the 3-step decode working set is 7.92 GiB
combined > 2×8.5 GiB per-GPU accounting already shows thrash). Raising the
pack to 12.75 GiB/GPU retains the working set through its reuse distance and
removes ~99 of 1252 decode storage misses, shortening the storage-bound
decode by a simulated ~6.9%.

## 2. Exact matched design

| Element | Baseline arm | Candidate arm |
|---|---|---|
| Commit | latest accepted exact branch (Codex's current base) | **same commit** |
| Config diff | (v65 config) | `host_pack_cache_bytes` per GPU: **8.5 → 12.75 GiB** (harness arg only; engine unchanged) |
| Prompt / tokens | sealed v65 prompt | identical |
| Route authority | official router | identical (journal must hash-match) |
| Lanes / queue depth | 6 / 6 | identical |
| VRAM expert cache | unchanged | unchanged |
| Correctness gates | sealed suite | identical, must pass |

Arms run interleaved (A, B, A, B) ≥3 pairs on a quiet host to cancel
drift; discard the first pair as cache-warmup if page cache state differs.

## 3. Required telemetry (sealed with the run)

- Existing counters (both arms): `per_token_accounting` (decode rows:
  `storage_requests`, `storage_bytes`, `host_pack_hits/misses`,
  `resident_hits`, `wall_ms`), `host_pack` block (hits, misses, evictions,
  entries, budget_bytes), engine `cold_loads/evictions`.
- Route journal (both arms): must hash-match v65's chain — routes identical.
- Optional but recommended: per-request staging telemetry
  (`exact-staging/staging-telemetry-v1`, see STAGING_DESIGN.md §2) behind the
  profiler flag — proves *which* reads were hidden.

## 4. Pre-registered success / falsification

Success (direction): candidate decode wall < baseline decode wall in ≥2 of 3
pairs, with candidate `decode storage misses` ≈ 1153 (±10) and pack evictions
materially below baseline's 708/409 per GPU.

Falsification triggers:
1. Misses do NOT drop to ≈1153 → the replay's budget→miss mapping does not
   transfer live (page-cache interference, allocator behavior, or a pack
   implementation nuance beyond LRU@682). Update the replay before any
   further budget claims.
2. Misses drop but wall does not → storage is not the binding stage on the
   live host at this scale (contradicts the calibrated model) — re-calibrate
   with the pread bench data before believing any storage-side estimate.
3. Correctness gate fails → abort; the knob must be behavior-neutral, so a
   failure indicates a config plumbing error, not a staging effect.

Report wall deltas as measured decode seconds only. **No TPS claims, no
ACCEPT_PERFORMANCE from a single candidate run** — acceptance follows the
standard sealed-campaign process.

## 5. What to run after this A/B (regardless of outcome)

1. `tools/bench_expert_pread.py` on the Kaggle host (PREAD_BENCH_PROTOCOL.md)
   — pins the storage floor that every remaining storage-side decision keys on.
2. If the budget A/B succeeds and the floor measurement confirms ≥370 MB/s
   headroom: evaluate 17 GiB/GPU (oracle point) against actual free RAM
   (29.64 GiB available at v65 makes 34 GiB unsafe; a future host with more
   RAM takes the full floor).
3. Buffer-reuse A/B (Codex's active track) interacts additively today but its
   modeled gain shrinks toward zero once storage misses drop (previous
   track's finding); re-run that interaction after this A/B seals.
