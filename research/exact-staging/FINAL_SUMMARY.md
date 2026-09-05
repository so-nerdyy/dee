# FINAL_SUMMARY.md — recalibration + host-pack frontier + memory contract + A/B statistics

Branch: `research/exact-staging` (worktree `../dee-staging`) · Date: 2026-09-04
Inputs: seal `14864034a7354e0e29e11c1c09f18b0863afe6a0` (sha256-gated), sealed
v65, causality audit (this branch), Muse `research/route-pipeline` (read-only).
Pipeline: `python research/exact-staging/recalibrate_model.py` (regenerates all
results JSON/CSV **and** the sentinel-marked markdown tables — JSON is the
single source of truth per MEMORY_UNIT_CONTRACT.md).
Labels: **[M]** measured · **[S]** simulated · **[T]** theoretical ceiling ·
**[E]** exploratory statistic.

## Memory-contract revision (this installment)

The previous revision's MEMORY_BUDGET.md table was a hand-maintained artifact
that disagreed with `results/memory_frontier.json` (10.0 GiB/GPU shown as
31.36 GB / 0.64 GB headroom vs the JSON's 29.23 GB / 2.77 GB): the table had
silently folded the 2.0-unit safety headroom into projected totals with a
stale conversion. Fixed structurally, not by re-typing numbers:

- `MEMORY_UNIT_CONTRACT.md` defines units (GiB vs decimal GB, never an
  unqualified "GB"), the detected-vs-nominal host distinction, and TWO
  authoritative safety systems: `MEASURED_HOST_MEMTOTAL` (31.35 GiB, for
  host-execution decisions) and `STRICT_32_DECIMAL_GB` (the project thesis,
  as a portable envelope). Row classification = conservative intersection.
- Markdown tables are now **generated from the JSON** between sentinel
  markers; tests render-and-compare every row, so any future unit or
  formula mismatch fails CI (a regression test reproduces the fcc8ca2 bug).
- The finer grid exposed one nuance the old table hid: 10.5–11.0 GiB/GPU
  are SAFE on measured MemTotal but BORDERLINE under the strict 32-decimal-
  GB contract. **10.0 GiB/GPU remains the largest budget SAFE under BOTH**,
  so the recommendation is unchanged after correction.

## The 10 answers

1. **Why did the old model predict ~−12.3% reuse when measurement says
   −0.94%/−2.63%?** Structural artifact, quantified in RECALIBRATION.md §1:
   it charged the host-pack copy (2.855 ms/rec × 3,215 transfers ≈ 9.2 s) as
   a *serial chain stage* and modeled reuse as copy elimination. Seal
   evidence shows the copies are worker-pipelined (`fill_worker_ms` 144 s vs
   `fill_batch_wall_ms` 54.6 s; `fill_overlap_ms` 89.5 s), still present with
   reuse ON (`mmap_to_pinned` 12.38 s in the profiled candidate), and that
   reuse actually removes buffer allocation (= evictions: 1,117 reused =
   708+409 evicted), which was already hidden. Secondary: v52 profiler
   inflation leaked into the pack constant; async completion was treated as
   blocking. No scalar fudge was applied — the mechanism was deleted and the
   model refit. **[M]**

2. **Recalibrated v65 error?** Fit on v65: R²=0.9785, per-step MAPE 2.52%.
   `wall_ms = 754.74 + 48.679 × misses` (ODR, physicality-gated). **[M fit]**

3. **Reproduce the host-buffer A/B?** As a *level* prediction, yes: held-out
   error +1.27/+2.23/−1.29/+1.38/+1.53% on the five seal runs (never fit).
   As a *reuse effect*, the model now predicts 0.0 s — matching the truth
   that both measured deltas sit inside the 1.85 s baseline run spread.
   Direction of the tiny measured win is acknowledged as unresolvable from
   two pairs at n=15 tokens. **[M]**

4. **Safest pack budget under a real 32 GB TOTAL limit?** Projected envelope
   (measured non-pack 5.82 GiB + system 0.91 + 0.5 unknown allowance):
   **9.5 GiB/GPU** is the risk-balanced SAFE default (26.227 GiB = 28.16
   decimal GB projected; headroom 5.12 GiB to measured MemTotal / 3.84
   decimal GB to the strict contract; 8.68 GiB OOM margin); **10.0 GiB/GPU**
   is the largest SAFE point under BOTH systems (27.227 GiB = 29.23 decimal
   GB; headroom 4.12 GiB / 2.77 decimal GB; all gates pass). 10.5–11.0 are
   SAFE on measured MemTotal but BORDERLINE on the strict contract. **[M
   anchors, S projection]**

5. **SSD reads removed by that budget?** Replay (validated ≤2/step vs
   sealed): 9.5 → **−42**; 10.0 → **−51** (1,252 → 1,201/1,211-class tables;
   per-GPU split ≈ 27/24). **[M replay of sealed trace]**

6. **Marginal miss reduction per extra GiB?** 8.5→9.0: 44/GiB; →9.5: 42/GiB;
   →10.0: 34/GiB; →10.5: 31/GiB; →12.75: ~23/GiB. First 1.5–2 GiB buy 3×
   the last 2. **[M replay]**

7. **SIMULATED wall improvement after recalibration?** 9.5: −2.09 s;
   10.0: −2.53 s (central; statistical CI ±5.6% on the miss term; run noise
   ~1.0–1.9 s). These are hypotheses for the A/B, not results. **[S]**

8. **Is 12.75 GiB/GPU safe for the 32 GB thesis?** **NO.** Projected
   32.727 GiB = 35.14 decimal GB (+3.14 over the strict contract), projected
   min-available **negative** (−1.38 GiB), and within 2.18 GiB of the
   nearest clean measured OOM (27.68 GiB pack total). REJECTED for the main
   thesis; acceptable only on 48/64 GB hosts. (The previously quoted 37.71
   GB came from the buggy table; the corrected projection is 35.14 GB —
   still decisively over.) **[M ledger + S projection]**

9. **The one A/B Astra should run next?** `next_ab.json` /
   LIVE_AB_V2_PROTOCOL.md: **`LRU_TOTAL_CAP_GIB` 17.0 → 20.0** (10.0 GiB/GPU;
   drop to the 19.0-cap variant if memory telemetry looks tight),
   **exactly 2 matched pairs, pre-registered** (futility: abort after pair 1
   if the candidate is >0.93 s slower; no early accept; informative-positive
   = both deltas negative AND mean ≥ 0.905 s MAD). Measured noise context
   **[M]**: same-config baseline SD 0.926 s, spread 1.851 s; the reuse A/B's
   measured −1.30 s mean delta had SNR 1.4 — an unpaired read would likely
   have misjudged it. Smallest detectable effect with this design: ~1.3%
   (single pair, direction-only) / ~2.1% (2-pair rule); 1% effects are
   UNATTAINABLE without the interleaved-session strategy (within-session SD
   currently UNKNOWN). 15 tokens: yes — predicted 2.48 s effect ≈ 2.7× SD;
   longer decode is a separate experiment (changes the route workload).
   **[S expectation, M noise, E sigma, M design]**

10. **What would falsify the host-pack hypothesis?** Pack misses fall
    exactly as replayed while decode wall stays flat beyond run noise in
    two matched pairs (⇒ miss service isn't wall service on this path);
    or the candidate OOMs / breaches the projected envelope; or any
    correctness divergence. **[M criteria]**

## Bottom line

The recalibrated model keeps what survived evidence (disk-drain-dominated
decode, ~48.7 ms effective service per miss), deletes what did not (reuse
mechanism, future-layer lead), and points the next live experiment at the
one lever that is simultaneously large, legal, config-only, and memory-safe:
**a bigger bounded host pack, 8.5 → 9.5–10.0 GiB/GPU** — while the pread
package (Phase G) and Muse's hash-layer microbench remain the two
measurements that would most change the picture.

## Verification & provenance

- Existing suite + new coverage (see `tests/`): replay accounting, budget
  arithmetic, UNKNOWN-safe memory accounting, unit-contract linting (no
  unqualified-`GB` fields in any emitted JSON), markdown-vs-JSON row-by-row
  consistency (regression test reproduces the fcc8ca2 bug), no-future-
  lookahead, seal ingestion, A/B statistics on synthetic data, planner
  monotonicity, predicted-vs-observed labeling — all green.
- Seal inputs hash-verified at run time (`recalibrate_model.py` fails
  closed on any mismatch); A/B evidence traceable to the same seal.
- Nothing merged; no sealed evidence modified; no production runtime
  changed; Muse's branch untouched; performance acceptance stays FALSE.
