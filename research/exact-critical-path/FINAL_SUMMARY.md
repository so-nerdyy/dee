# FINAL_SUMMARY.md — exact critical-path model of dee's DSV4 decode

Track: `research/exact-critical-path` · offline only · no production code
touched · no sealed evidence modified · nothing merged.
Artifacts: `tools/exact_critical_path_sim.py`, `tools/critical_path_sweeps.py`,
`tools/ingest_ab.py`, `results/*.json|csv`, 16 passing tests.

## Answers to the mandated questions

**1. Can the simulator reproduce current decode wall time?**
Yes at baseline level: **71.41 s predicted vs 72.267 s observed (−1.19 % total,
3.28 % per-step MAPE)** on the sealed v65 run, with fitted parameters that
independently match measured anchors (disk ≈ 320–459 MB/s band, H2D 5.54 vs
5.38 GB/s microbench). It also reproduces the one sealed scheduling A/B it was
not fit to (3 vs 6 storage lanes → 0 effect). This validates reconstruction,
**not** counterfactuals.

**2. What is the current critical path?**
The saturated expert store: modeled disk drain is 70.9 s of the 71.4 s wall.
Records are read **on demand** (fitted staging lead = 0), so the realized
decode read rate is 232 MB/s against a 320–459 MB/s concurrent capability.

**3. Is storage still dominant?**
Yes — ~99 % of modeled decode wall. Decode moves 16.74 GB per emitted token
(1,252 records × 13.369 MB) through the store.

**4. Is H2D dominant?**
No. Total H2D ≈ 4.0 s (5.5 %), already overlapped; doubling its rate models
−5.3 %.

**5. Is GPU expert compute dominant?**
No. Routed compute ≈ 0.9 s (1.3 %) at the v52-measured 0.4537 ms/record.
Doubling it models +2.4 %.

**6. Is host orchestration/serialization dominant?**
No. ~1 % (1.2 ms/layer × 43 layers, calibrated). The v52-era
tensor-resolution and bridge concerns are gone from the fp4 trace-store
pipeline (v51 133.0 s vs v52 profiled 138.7 s; v65 72.3 s).

**7. What is the largest exact-preserving modeled opportunity?**
**Intra-step earlier expert staging** (fill pipeline requests layer L's
records at consumer layer L−8): modeled **−26.1 %** decode wall (71.4 → 52.8 s),
saturating at the disk roof (~52.6 s). Robust across the disk-rate
uncertainty band. Second: host staging-buffer reuse (Codex's current A/B),
modeled **−12.3 %** in today's on-demand pipeline.

**8. Which optimization should Codex test immediately after the active A/B?**
N1 (earlier staging, lead ≥ 8) — NEXT_EXPERIMENTS.md has the full matched
A/B design, counters (`source_read_overlap_percent`,
`waiting_cache_entry_readiness`, per-step storage_bytes vs wall), and gates
(exact tokens, paired medians, sub-5 % spread). Then re-run the buffer-reuse
A/B **after** N1: its effect is subsumed by staging (modeled −12.3 % → ~0.3 %).

**9. Which previously proposed ideas are now low-value?**
Grouped per-layer H2D (−0.07 %), more storage lanes (0 % — already sealed as
neutral in v63/v65), grouped dispatch / device worklists (−0.5 %), extra H2D
streams (≤ −5 % even at 2× rate), pinned-ring depth (measured flat in idle4).

**10. What measurement is most important if the model remains ambiguous?**
The store's true concurrent read capability on the Kaggle host
(depth 1→16 pread microbench of 13.4 MB records): it pins the −26 % floor's
exact position (modeled band 47.5–52.8 s). Second: one profiled run
(`profile_stages: true`) to resolve GPU dense/non-MoE time, the largest
remaining UNKNOWN.

## Claim discipline

- All gains above are **simulated results** on a calibrated model — not
  measured speedups. No ACCEPT_PERFORMANCE is claimed or implied.
- Not claimed: end-to-end TPS improvement (no runtime benchmark was run by
  this track); that the staging-lead gain survives unchanged on longer
  generations or different prompts.
- The measured facts this track leans on: sealed v63/v64/v65 walls and
  counters, v52 stage rates, idle4 microbench rates, v65 per-step accounting
  (internally consistent to the byte).
- Calibration update path for Codex's A/B is live (`tools/ingest_ab.py`,
  `AB_SCHEMA.json`); direction-agreement and pack-cost re-fit are automatic.

## Confidence and limitations

- Baseline reconstruction: **high confidence** (−1.19 %, cross-checked
  parameters, byte-exact per-step accounting).
- Staging-lead direction and rough magnitude: **medium-high** (structure is
  measured; the floor's position depends on the fitted disk rate).
- Buffer-reuse magnitude: **medium** (modeled as a pack-path cost elimination;
  exact effect depends on Codex's implementation — hence the ingestion path
  and the post-N1 re-run requirement).
- Trace limitation: 15 decode tokens, one prompt, one hardware host. The fit
  uses per-step variation, but no cross-trace generalization is claimed.
  Prefill (95.7 s of the 169 s run) is out of scope.
