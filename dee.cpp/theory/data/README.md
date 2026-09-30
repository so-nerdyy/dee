# theory.data — README

Every CSV/JSON here is regenerated deterministically by
`python -m theory.run_all` (from `dee.cpp/theory/`).  Nothing in this
directory is hand-edited; provenance for every constant is in
`provenance.csv` (MEASURED / DERIVED / CALIBRATED / ASSUMPTION tags).

## Files

| file | contents |
|---|---|
| `provenance.csv` | every constant with value, unit, tag, source path |
| `popularity_fits.csv` | family fits per profile + per-layer + per-run (log-lik, AIC/BIC, KS-D) |
| `popularity_bootstrap.csv` | parametric bootstrap CIs for Zipf-Mandelbrot (s,q) and lognormal-rank (mu,sigma) |
| `popularity_entropy.csv` | Shannon entropy vs log2(N) per profile + per layer |
| `popularity_coverage.csv` | coverage curve C(m) = sum_{i<=m} p_(i) |
| `popularity_tail.csv` | per-family mass error by rank region (head / mid / tail) |
| `cache_sim_check.csv` | LRU / Belady-MIN / shuffled-IRM / Che-steady / Che-finite vs budget on the sealed trace |
| `cache_sim_landmarks.json` | knee + repeat-count landmarks vs the phase-2 ws-policy measurements |
| `cache_che_curves.csv` | closed-form H(M) curves for all four store specs |
| `cache_store_knees.csv` | capacity at H=50/80/95% per store |
| `cache_summary.json` | landmarks + IRM-vs-correlated error decomposition |
| `temporal_survival.csv` | reaccess survival P(reaccess after lag) with sample counts |
| `temporal_crosslayer.csv` | P(j in S_{l+1} | i in S_l) pairs with support >= 5 |
| `temporal_selfcorr.csv` | decode-step self-correlation vs IRM baseline |
| `temporal_irm_delta.csv` | real vs shuffled LRU hit-rate delta (temporal correlation value) |
| `anchor_check.csv` | anchor prediction rows (miss model x B_ssd case) |
| `anchor_check.json` | anchor verdict + calibrated eta_storage + prefill-inclusive check |
| `roofline_curves.csv` | TPS bound + prediction grid (cell x model x batch x host budget x horizon) |
| `roofline_anchor.json` | anchor check summary (same as anchor_check.json) |
| `serve_hitrate.csv` | H_serve(M, lambda) and bytes/token vs concurrency |
| `serve_cost_frontier.csv` | $/1k tok per cell with hardware rate breakdown |
| `serve_breakeven.csv` | dee vs dense-residency baseline ratios |
| `serve_dense_baseline.json` | the baseline's explicit (conservative) assumptions |
| `serve_volume_cost.json` | volume-storage monthly cost at the published rate |
| `prefetch_predictor.csv` | next-layer predictor recall/precision/bandwidth at m = 1..8 |
| `prefetch_budget_sweep.csv` | linear equal-cost objective vs budget |
| `prefetch_submodular.csv` | submodular greedy (1-1/e) variant vs budget |
| `prefetch_ceiling.json` | Delta H <= Sigma_corr ceiling + prior-art gate verdict |
| `sensitivity.csv` | log-log elasticities of TPS in both horizon regimes |
| `regime_map.csv` | limiter + TPS over (B_SSD, host GiB) per (model, cell) |
| `regime_phase1_check.json` | Phase-1 storage-verdict consistency check |
