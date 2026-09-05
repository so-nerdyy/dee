# AB_NOISE_ANALYSIS.md — how loud is an exact run?

Branch: `research/exact-staging` · Date: 2026-09-04
Machine-readable (source of truth): `results/ab_noise.json`
Tool: `tools/analyze_matched_ab.py` (offline; inputs are machine-readable run
JSON with sha256 provenance; every field carries a label:
MEASURED / DERIVED / EXPLORATORY_STATISTIC / SIMULATED / UNKNOWN).

Reproduce (from the worktree root; v63/v64 profiles are read-only sealed
campaign evidence in the main checkout):

```bash
python tools/analyze_matched_ab.py \
  --profile v63_3lane=<MAIN>/dee.cpp/tmp/v63-terminal-fetch-20260903T0048Z/profile.json \
  --profile v64_3lane=<MAIN>/dee.cpp/tmp/v64-terminal-fetch-20260903T0120Z/profile.json
```

## 1. Evidence inventory (configs not mixed without labels)

| Run | Config group | Wall (s) | Role |
|---|---|---|---|
| v65 sealed | exact-on-demand-6lane, no reuse | 72.267 | baseline anchor |
| host-reuse ab_baseline | identical | 71.362 | baseline |
| host-reuse rep_baseline | identical | 73.213 | baseline |
| host-reuse ab_candidate | + reuse flag | 70.691 | pair 1 candidate |
| host-reuse rep_candidate | + reuse flag | 71.284 | pair 2 candidate |
| profiled_candidate | + reuse flag + stage profiling | 71.179 | profiling bound |
| v63 / v64 | 3-lane era (different config) | 71.315 / 71.804 | context only, excluded from variance |

All six v63/v64/v65-era runs share byte-identical per-step reads/copies
tables; only walls vary — pure run-to-run noise, which is exactly what makes
them a valid same-config baseline set. v63/v64 are labeled context (different
lane config) and excluded from the baseline variance estimate.

## 2. Baseline variance (n=3 same-config runs — MEASURED walls, DERIVED stats)

- walls: 71.362 / 72.267 / 73.213 s → mean 72.281, median 72.267
- **SD 0.926 s · MAD 0.905 s · full spread 1.851 s (~2.6% of wall)**
- within-notebook SD: UNKNOWN (no same-config repeats inside one notebook;
  `BLOCKED_BY_MISSING_MEASUREMENT`)

Small-sample honesty: n=3. No normality claim is made or needed — the
decision rules in AB_EXPERIMENT_DESIGN.md use direction consistency and MAD
comparisons, which are valid without distributional assumptions.

## 3. Paired A/B effect of host-buffer reuse (the calibration case study)

| Pair | Baseline | Candidate | Δ (s) | Δ% |
|---|---|---|---|---|
| original | 71.362 | 70.691 | −0.671 | −0.94% |
| replication | 73.213 | 71.284 | −1.929 | −2.63% |

- mean Δ −1.300 s, median Δ −1.300 s, direction 2/2 negative
- bootstrap CI95 of mean Δ: **[−1.93, −0.67] s — EXPLORATORY only** (n=2
  resampling is degenerate: it can only reproduce the two observed deltas;
  stated for direction, not confidence)
- SNR vs baseline SD: **1.4 — "suggestive"** on the pre-registered verdict
  scale: an unpaired comparison of two single runs would likely have
  misread this effect as noise (or luck). The measured effect also sits
  entirely inside the 1.85 s baseline spread.
- per-token paired counts (12/15 and 15/15 tokens improved) are DESCRIPTIVE
  only: per-token walls are serially dependent (lag-1 autocorrelation
  ≈ 0.49–0.52), so 15 tokens are 15 dependent samples, not 15 replicates.
  They are reported with that caveat and are never used as a test.

## 4. What this means for the pack-cap candidate

The recalibrated model predicts ≈ 2.48 s (SIMULATED) for cap 17→20 GiB.
Against the measured noise: SNR ≈ 2.7–2.8 — above the "detectable with a
small matched-pair budget" bar IF the effect is real at the predicted size.
The reuse effect (SNR 1.4) is the cautionary tale: it was real but needed
paired design + replication. See AB_EXPERIMENT_DESIGN.md for the
pre-registered plan.

## 5. Known unknowns

- within-notebook variance: UNKNOWN (single measurement would resolve it;
  recommended as a cheap rider on the next A/B session)
- profiling perturbation: |Δ| = 0.105 s between unprofiled and profiled
  reuse candidates (EXPLORATORY bound; sessions also differ) — profiling
  cost is small at decode level but not zero-proven
- normality: not assumed, not proven; no t-test is reported at n≤3
