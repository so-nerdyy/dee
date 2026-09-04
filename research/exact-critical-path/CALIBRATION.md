# CALIBRATION.md — fitting and validating the baseline model

Command (reproducible):

```bash
python tools/exact_critical_path_sim.py --calibrate \
    --out research/exact-critical-path/results/calibration.json
python tools/critical_path_sweeps.py        # Phase C/D/E results
python -m pytest research/exact-critical-path/tests/ -q
```

Fit: grid + local refinement minimizing the sum of squared per-step relative
error against the 15 sealed v65 step walls. Free parameters: disk aggregate
rate, effective H2D rate, host orchestration per layer, staging lead.

## Result

| Quantity | Value | Note |
|---|---|---|
| Predicted decode wall | **71.41 s** | simulated |
| Observed decode wall | **72.267 s** | sealed v65 |
| Absolute error | −0.86 s | |
| Relative error | **−1.19 %** | total |
| Per-step MAPE | 3.28 % | worst step +11 % (step 9) |

## Fitted parameters and their measured cross-checks

| Parameter | Fitted | Independent evidence |
|---|---|---|
| `ssd_aggregate_mb_s` | 320 | v65 byte-accounting derivation 459 MB/s (all reads incl. prefill / decode wall, lower bound on capability); lanes×p50 invariant ≈ 368–372 MB/s (v63/v65); decode-only average realized rate 231.6 MB/s (on-demand staging, idle gaps) → capability between 320 and 459, realized 232 |
| `h2d_gbps` | 5.54 | idle4 microbench 5.38 GB/s (pinned, single GPU, quiet) — fitted value ≈ ceiling under dual-GPU load is physically sensible |
| `host_orchestration_per_layer_ms` | 1.2 | UNKNOWN directly; 86 host syncs/step + bridge dispatch; bounded by v52 CPU stage sums (inflated) |
| `staging_lead_layers` | 0 | v52 timeline shows on-demand staging; v65 read overlap 77.9 % is *intra-batch* overlap, not consumer lead |

The disk-rate fit is the load-bearing parameter: it sets −26 % of the wall.
Its uncertainty band (320–459 MB/s from three independent measurements) is
carried through the sensitivity analysis in CANDIDATE_SWEEPS.md.

## Validation outside the fit

1. **Lane A/B reproduction (measured)**: the model with 3 vs 6 lanes predicts
   identical wall — matching the sealed v63 (71.315 s) / v64 (71.804 s) /
   v65 (72.267 s) outcome (all within run-to-run noise). The model reproduces
   the *direction and magnitude* (≈0) of the only recorded scheduling A/B
   without having been fit to it.
2. **Per-step structure (measured)**: per-step walls correlate with per-step
   SSD reads through the disk-drain term; steps with zero reads (step 1: 4.24 s
   observed, all misses pack-sourced) are reproduced by the H2D+pack+host
   terms (4.00 s predicted). MAPE 3.28 % over 15 steps.
3. **Stage-rate sanity (measured anchors)**: GPU compute totals 0.9 s (1.3 % of
   wall) vs v52's measured rates; H2D totals 4.0 s — neither dominates, so the
   model cannot be hiding a dominant stage in a fitted term.

## Interpretation guardrails

- The −1.19 % total error does **not** validate the model's counterfactuals;
  it validates the baseline reconstruction only.
- Parameters fitted to one 15-token trace are trace-local. The fit uses
  per-step *variation* (reads 56–125/step) rather than only the total, which
  limits—but does not eliminate—overfitting risk.
- All sweep outputs are SIMULATED RESULTS. No ACCEPT_PERFORMANCE claim is
  made or implied anywhere in this track.
