# AB_EXPERIMENT_DESIGN.md — pre-registered plan for the pack-cap A/B

Branch: `research/exact-staging` · Date: 2026-09-04
Machine-readable (source of truth): `results/next_ab.json` (experiment),
`results/ab_power_plan.json` (power math), `results/ab_noise.json` (noise).
Tool: `tools/plan_ab_power.py` (advisory; prices evidence, does not change
formal acceptance rules).

## 1. The experiment (one knob)

`LRU_TOTAL_CAP_GIB` (host-pack runtime cap): **17.0 → 20.0** total
(8.5 → 10.0 GiB/GPU; 682 → 803 records/GPU). Chosen as the largest budget
SAFE under BOTH safety systems (MEMORY_UNIT_CONTRACT.md): headroom 4.12 GiB
to measured MemTotal / 2.77 decimal GB to the strict contract, 7.68 GiB
margin to the nearest measured clean OOM. Full matched-run spec, metrics,
and abort criteria: `LIVE_AB_V2_PROTOCOL.md` and `results/next_ab.json`.

## 2. Is one pair informative? (pack-cap specific)

- Predicted effect ≈ 2.48 s (SIMULATED, 51 misses × 48.679 ms marginal
  service). SNR ≈ 2.7 vs measured baseline SD (0.926 s) — a single pair
  would LIKELY show the direction if the simulated size is real.
- The risk is not statistical power; it is **effect reality**: the misses
  removed first at a larger budget are the cheapest ones, so realized
  service per avoided miss may be lower than the global 48.7 ms marginal.
- Therefore: **2 matched pairs budgeted upfront** (direction consistency
  ≥ 97% for a true 2.5 s effect; mean-rule satisfied), both run regardless
  of pair 1 unless futility fires.

## 3. Pre-registered decision rule (fixed before the first run)

Run **exactly 2 matched pairs** (matched design = the host-reuse shape:
baseline arm + candidate arm per notebook session, two sessions).

1. **Futility (abort after pair 1):** candidate wall > baseline wall +
   1 baseline SD (0.926 s). A slower candidate is an obvious null for an
   effect whose hypothesis predicts an improvement.
2. **No early accept:** pair 2 is mandatory even if pair 1 looks great.
   No post-hoc stopping on favorable results, ever.
3. **Informative-positive** (proceeds to formal-acceptance consideration):
   both deltas negative AND mean delta ≥ baseline MAD (0.905 s).
4. **Null/UNRESOLVED** otherwise — reported as unresolved, NOT as evidence
   of absence; the plan then escalates to the interleaved-session strategy
   (below) rather than to more unpaired runs.
5. Formal performance acceptance is granted only by the campaign's own
   rules; this design does not loosen them.

## 4. Power table (from measured noise; DERIVED rules on EXPLORATORY sigma)

| True effect | σ_d = 0.89–0.93 s | pairs for direction ≥90% | pairs for mean rule (z≈2) |
|---|---|---|---|
| 1% (0.72 s) | SNR 0.78 | **UNATTAINABLE** (single-pair P(improve)=0.79; pⁿ falls with n) | 7 |
| 2% (1.45 s) | SNR 1.6 | 1–2 | 2 |
| 3% (2.17 s) | SNR 2.3 | 1 | 1 |
| 5% (3.61 s) | SNR 3.9 | 1 | 1 |

Smallest detectable effect with the current design: ≈ 0.93 s (1.3%) single
pair, direction-only; ≈ 1.5 s (2.1%) under the 2-pair pre-registered rule.
**1% effects are not detectable with this design** — see §5.

## 5. Strategy assessment (what would make 1% detectable)

- **Interleaved baseline/candidate in ONE notebook session** (A/B or
  A/B/A/B): removes the between-notebook offset from the paired delta.
  Highest-value structural change; blocked only by UNKNOWN within-session
  variance — measure it once as a cheap rider (one repeat arm) on the next
  session.
- **More matched pairs:** the only lever that scales evidence without
  changing the workload; 7 pairs for a 1% effect under the mean rule is
  expensive but finite.
- **Longer decode:** mechanical 1/√n_eff scaling suggests 64 tokens would
  cut relative SD ~2× (SIMULATED, unvalidated; lag-1 autocorrelation
  ≈ 0.49 → n_eff ≈ 5 of 15 tokens), **but** longer generation changes the
  recurrence/miss mix and the effect itself, invalidating direct comparison
  to the 15-token replay-validated counters. Verdict: only as a separate,
  pre-registered experiment — **not** for the pack-cap hypothesis.
- **15 tokens for THIS experiment:** yes — the predicted effect (~2.5 s)
  exceeds even the conservative SD by ~2.7×, and 15 tokens is what the
  replay counters validate; changing length desyncs the comparison.

## 6. What would falsify the host-pack hypothesis

1. Pack telemetry shows hits rising / misses falling as replayed
   (~51 fewer SSD reads), but decode wall does not improve beyond run
   noise (mean delta < MAD) in two matched pairs → the storage-miss model
   of decode time is wrong for these misses.
2. Process memory exceeds the projected envelope or approaches OOM →
   the memory model is wrong (audit `unknown_components`).
3. Any correctness gate fails → stop immediately, hypothesis irrelevant.
