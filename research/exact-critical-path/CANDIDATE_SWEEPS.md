# CANDIDATE_SWEEPS.md — simulated what-if, sensitivity, and interactions

All numbers below are **SIMULATED RESULTS** from the calibrated model
(baseline 71.41 s ≈ observed 72.267 s). They are hypotheses to test live,
not measured speedups. Raw data: `results/sweeps.json`,
`results/phase_c_scenarios.csv`, `results/phase_e_interactions.csv`.

## Phase C — exact-preserving changes (single-lever)

| Scenario | Predicted wall | Δ vs baseline | Reading |
|---|---:|---:|---|
| staging_lead_8 / 16 / 30 / full | 52.79 s | **−26.1 %** | Saturates at the disk-drain floor 52.6 s; lead ≥ 8 layers exhausts it |
| buffer_reuse (host staging copies → pointer swap) | 62.60 s | **−12.3 %** | Only lever that attacks the wall while staging stays on-demand |
| h2d_2x_rate | 67.62 s | −5.3 % | Even doubling PCIe helps little — H2D is not the chain |
| host_orch −100 % | 70.65 s | −1.1 % | Host slices are small |
| grouped_dispatch (−50 % slices) | 71.03 s | −0.5 % | Same mechanism, half the effect |
| grouped_h2d | 71.36 s | −0.07 % | Negligible at 13.4 MB records |
| lanes_3 / lanes_12 | 71.41 s | 0.0 % | Lane count is inert by design (disk saturated) — reproduces the sealed v63/v65 A/B |
| no_overlap (H2D ⊥ compute serialized) | 72.21 s | +1.1 % | Current overlap already hides most H2D |

Ceiling stack: staging-lead + buffer-reuse + grouped-H2D together reach
52.58 s (−26.4 %) — i.e., after earlier staging, nothing else matters; the
residual is the saturated disk itself.

## Phase D — sensitivity around fitted values

| Parameter ×multiplier | 0.5× | 0.75× | 1.25× | 1.5× | 2× |
|---|---:|---:|---:|---:|---:|
| SSD aggregate BW | — | +24.4 % | −14.7 % | −24.4 % | −36.6 % |
| H2D rate | — | +3.5 % | −2.1 % | −3.5 % | −5.3 % |
| GPU compute/record | −1.2 % | −0.6 % | +0.6 % | +1.2 % | +2.4 % |
| Host orchestration/layer | −0.5 % | — | +0.6 % | — | +1.1 % |
| staging lead (0→4 layers) | **−26.1 %** by lead 4 | | | | |

Survives uncertainty:
- **Earlier staging (lead ≥ 8)**: −26.1 % across the whole disk-rate band
  320–459 MB/s (floor 47.5–52.8 s). Robust.
- **Buffer reuse**: −12.3 % when staging is on-demand; shrinks toward ~0 if
  staging lead lands first (interaction below). Robust in the current
  pipeline, fragile after staging changes — schedule accordingly.
- **H2D rate, GPU compute, host orchestration**: sub-6 % each even at ±50 %.
  Not worth live A/Bs at current traffic.

Roofline note (decode): 1,252 records × 13.369 MB = 16.74 GB must move per
generated token from a store whose concurrent capability is ≈ 320–459 MB/s
→ pure storage floor ≈ 37–52 s/token-set regardless of policy. The model's
−26 % floor (52.6 s) coincides with the 459 MB/s roof; the gap to the 320 MB/s
band is uncertainty in the store's true concurrent rate.

## Phase E — interactions

| Combination | Predicted wall | Δ |
|---|---:|---:|
| buffer_reuse + lead16 | 52.58 s | −26.4 % |
| buffer_reuse + grouped_h2d | 62.55 s | −12.4 % |
| orch−100 % + grouped_dispatch | 70.65 s | −1.1 % |
| grouped_h2d + grouped_dispatch | 70.98 s | −0.6 % |

Key interaction: **buffer reuse is subsumed by earlier staging**
(52.79 → 52.58, ~0.2 s). If the staging pipeline is fixed first, the Codex A/B
should be re-run after: its measured −12 % under on-demand staging will not
transfer to the staged pipeline. Grouped H2D never interacts meaningfully
with anything (records are already large).
