# FINAL_SUMMARY.md — exact-staging track

Branch: `research/exact-staging` (worktree `../dee-staging`, base `2db0fde`) · Date: 2026-09-04
Track: OFFLINE exact storage/staging research. Official router authoritative.
No production runtime modified (prototype isolated under
`dee.cpp/experiments/exact_staging/`), no sealed evidence touched, nothing
merged, Codex's active host-buffer A/B untouched.

---

## The headline

**The critical-path model's simulated "lead ≥ 8 layers → −26%" is not
causally implementable in exact mode** — layer L+1's official route does not
exist until layer L's MoE output computes (CAUSALITY_AUDIT.md §2). The legal
cross-step mechanism production already has is the **host-pack LRU at
682 records (8.5 GiB/GPU)**, and a **config-only budget increase captures a
simulated −6.9% decode wall** — smaller than the impossible lead scenario,
but real, legal, and available without engine changes.

## Answers to the required questions

1. **What is true measured concurrent pread capability?**
   UNKNOWN — no expert store exists on this host, so the bench
   (`tools/bench_expert_pread.py`, validated for bookkeeping by tests) has
   produced no measurements. Sealed evidence pins two points: v63 3-lane and
   v65 6-lane both ≈370 MB/s aggregate (p50 × lanes invariant); the calibrated
   model uses 320 MB/s within a 320–459 MB/s byte-accounting band.
2. **At what queue depth does it saturate?**
   UNKNOWN directly; the sealed 3→6-lane neutrality (identical batch wall,
   per-read latency doubling) implies saturation is reached **by depth 3**.
   The bench measures depths 1–16 to replace this inference.
3. **Is simulator lead ≥ 8 causally implementable?**
   **No.** `IMPOSSIBLE_WITHOUT_FUTURE_ROUTE` for any in-step lead ≥ 1.
4. **What is the largest legal exact lead?**
   In-step: **0 layers**. Cross-step: one full step (step k−1's official
   routes → step k's demands) — already implemented as the host pack; its
   binding constraint is **capacity**, not mechanism.
5. **What code restructuring would expose more legal lead?**
   None identified that preserves exactness: official routes are produced
   sequentially by the router from each layer's output. Executing multiple
   layers of routing per pass would change execution architecture (and is
   therefore out of contract for this track). The exploitable axis is the
   pack budget, not the staging schedule.
6. **How much bounded host memory does the candidate require?**
   +4.25 GiB/GPU over sealed (8.5 → 12.75 GiB; +8.5 GiB total across both
   GPUs → 25.5 GiB committed). The sealed host reported 29.64 GiB host RAM
   available during v65 with both 8.5 GiB packs resident, so the candidate
   leaves ≥4 GiB margin. Labeled: derived from sealed counters + harness
   probe, not a fresh measurement.
7. **What exact candidate should Codex A/B next?**
   `host_pack_cache_bytes` 8.5 → **12.75 GiB per GPU**, same commit/config
   otherwise, per LIVE_AB_PROTOCOL.md (matched design, interleaved arms,
   pre-registered success/falsification).
8. **What telemetry proves reads were actually hidden?**
   Per-request `exact-staging/staging-telemetry-v1` (ready_before_demand +
   useful_lead_ms), plus aggregate proof: decode `storage_requests`
   1252 → ≈1153 and pack evictions 706/410 → 291/43 per GPU (replay at
   1024 rec; 0/0 at the 17 GiB oracle point — see
   results/pack_replay_sweep.json).
9. **What condition would falsify the staging hypothesis?**
   Pre-registered in LIVE_AB_PROTOCOL.md §4: (a) misses don't fall to ≈1153
   (±10) → the validated replay does not transfer to the live host; (b) misses
   fall but wall doesn't → storage is not the binding stage live; recalibrate
   with the pread bench before further storage-side claims.
10. **What should Codex test after staging if it succeeds?**
    (1) the pread bench to pin the true storage floor; (2) if RAM allows on
    the host fleet, the 17 GiB/GPU oracle point (simulated −7.9%, floor of
    all offline policies); (3) re-check the host-buffer-reuse interaction
    (its modeled gain shrinks toward zero once pack misses drop).

## Evidence chain (all reproducible)

- `results/pack_replay_sweep.json` — validated pack replay: LRU@682
  reproduces sealed v65 per-GPU misses/evictions within 2 records
  (1388/706 vs 1390/708; 1092/410 vs 1091/409; decode misses 1251 vs 1252;
  per-step reads within ±2 every step).
- `results/pack_budget_wall_estimates.json` — simulated walls via the
  calibrated critical-path model (baseline −1.19% error), per-step read
  tables substituted from the replay.
- Belady on the fill stream @ sealed budget: 1135 decode misses — identical
  to LRU@17 GiB. At fixed budget, policy headroom ≤ 9.3%; capacity is the
  whole game on this trace.
- `CAUSALITY_AUDIT.md` — source-verified route-known timing; lead
  classification table.
- `STAGING_DESIGN.md` — bounded queue semantics + candidate rationale.
- `LIVE_AB_PROTOCOL.md` — the matched A/B package.
- `PREAD_BENCH_PROTOCOL.md` — how to produce the missing measurement.
- Tests: 14 passed (`research/exact-staging/tests/`).

## Claim discipline

- Replay pack counters: **validated against sealed evidence** (±2 records).
- Budget → misses mapping: **trace-local replay** (15 decode tokens, one
  prompt; do not generalize across prompts/workloads without re-validation).
- Wall deltas: **simulated estimates only** (calibrated model, −1.19%
  baseline error). No TPS claims, no ACCEPT_PERFORMANCE.
- Pread capability: **UNKNOWN** until the bench runs on the store host.
