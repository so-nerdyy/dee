# FALSIFICATION.md — numeric predictions and the measurements that kill them

Each prediction is range-bounded and names the measurement that would
**kill** it. Ranges are 90%-style working ranges, not p-values; the model
behind each is in [`THEORY.md`](THEORY.md) at the cited section, and every
constant's provenance is in [`data/provenance.csv`](data/provenance.csv).
A prediction killed by its named measurement should be marked
`KILLED (date, evidence path)` in this file and the theory corrected — not
defended.

Hardware runs referenced below are the Phase-6 Modal campaign (L4-class) and
any GPU/CPU batch listed in the AGENTS.md ledger. No GPU job is required by
this document's own reproduction (all local).

---

## Headline predictions (the three to read first)

| id | prediction | range | kills it |
|---|---|---|---|
| **P1** | Decode TPS on 1xL4, DSv4-Flash, `b=1`, host 64 GiB, VRAM 22 GiB, cold-ish bank | **1.1-1.7 tok/s** (nominal 1.35, storage-limited; `data/roofline_curves.csv` cell 1xL4 row) | a measured decode rate outside 0.9-2.1 tok/s with storage reads accounted, at B_SSD 2.5 GiB/s |
| **P2** | Host-tier LRU hit rate at 16 GiB pooled on any >= 2k-record DSv4-Flash generation window of 4-20 tokens | **48 - 54%**, and **within 3pp** of offline MIN at that budget | an LRU replay (or live run) landing outside 45-57%, or more than 6pp from MIN at 16 GiB |
| **P3** | Cold SSD record reads per decode token at 16-token horizon, 281-slot VRAM + 682-slot host | **100 - 220 records/token** (finite-window bracket; measured anchor 83-155) | a measured count outside 70-260 with both cache levels sized as stated |

---

## P1-P3: hierarchy predictions

| id | prediction | numeric range | measurement that kills it | model |
|---|---|---|---|---|
| P1 | 1xL4 decode TPS (b=1, host 64 GiB) | 1.1-1.7 tok/s (kill: outside 0.9-2.1) | Phase-6 1xL4 run, 16+ decode tokens, `decode_tok_s` | THEORY.md 5 |
| P1b | 2xL4 at `b=8`, host 256 GiB | 12-18 tok/s steady-state regime (kill: outside 8-28) | `decode_tok_s` at batch 8, 256 GiB host | THEORY.md 5-6 |
| P2 | host LRU hit @ 16 GiB pooled | 0.48-0.54 (kill: outside 0.45-0.57) | LRU replay on a fresh >= 2k-record trace window | THEORY.md 3.2 |
| P2b | LRU within 3pp of MIN @ 16 GiB | -2.8pp nominal, kill: gap < -6pp | Belady-MIN replay on the same window | THEORY.md 3.2 |
| P2c | records ever repeating in a 16-token / 43-layer / top-6 window | 850-1000 of 2,364-2,500 unique | rerun of the sealed generation shape; count outside range | THEORY.md 3.2 |
| P3 | cold SSD record reads/token (16-tok, 281+682 slots) | 100-220 (kill: outside 70-260) | `per_token_accounting.storage_requests` on a matched run | THEORY.md 5.2 |
| P3b | device fills/token (VRAM misses) | 190-260 (kill: outside 150-320) | `cold_loads` per decode token | THEORY.md 5.1 |
| P3c | whole-generation TPS incl. prefill at 0.29-0.37 GiB/s storage | 0.062-0.13 tok/s (storage bound x1.25-2.0 slack) | 2xT4 rerun landing outside 0.05-0.16 tok/s over 16 tokens | THEORY.md 5.2 |
| P4 | finite-window Che bracket for cold records/token | measured always in [steady-Che, 2x finite-Che] = [46, 324] | any matched run whose cold count falls outside the bracket | THEORY.md 5.2 |
| P5 | eta_storage (fill hiding fraction) on any T4-class cell | 0.15-0.45 (nominal 0.292) | stage-profile-derived hiding fraction outside 0.10-0.55 on a second platform | THEORY.md 5.1 |

## P6-P8: popularity / cache-shape predictions

| id | prediction | numeric range | measurement that kills it | model |
|---|---|---|---|---|
| P6 | Zipf-Mandelbrot MLE on any DSv4-Flash router window >= 2k requests is **geometric-degenerate** (`degenerate_to_geometric=True`): fitted `q > 0.5 x n_seen` and decay `s/q` in | 0.05-0.5 %/rank | a window where the MLE yields `q < 0.1 x n_seen` (a genuine power law) with AIC better than the geometric fit | THEORY.md 2 |
| P6b | lognormal-rank sigma (pooled) | 1.7-2.2 (kill: outside 1.5-2.5) | refit on a fresh window | THEORY.md 2.2 |
| P7 | entropy of pooled record popularity | 9.5-11.5 bits (N_eff 700-2,900) | plug-in entropy outside 8.8-12.2 bits on a fresh window | THEORY.md 2.3 |
| P8 | top-10% of records capture >= 60% of accesses | 0.60-0.95 | coverage C(0.1N) < 0.55 on a fresh window | THEORY.md 2.3 |
| P8b | per-touch implementation cost `t_0` on the anchor dispatch path | 300-420 us (kill: outside 250-450 us) | per-expert dispatch timing from a stage profile on any T4-class run | THEORY.md 5.1 |
| P8c | `t_0` with a batched expert-kernel path (dee-serve style) | 50-150 us (kill: > 250 us) | the same measurement on a batched-kernel build | THEORY.md 5.1 |

## P9-P12: prefetch predictions (the only legal prediction surface)

| id | prediction | numeric range | measurement that kills it | model |
|---|---|---|---|---|
| P9 | next-layer predictor recall (empirical P(j\|i), m=8/source, lead=1) | 0.65-0.80 | replay on a fresh trace with recall outside 0.55-0.88 | THEORY.md 7 |
| P10 | same predictor **precision** | 0.10-0.18 (kill: > 0.30) | precision > 0.30 at m=8 would kill the "fails prior-art gate" claim and reopen prefetch as a wall-time lever | THEORY.md 7 |
| P11 | `Delta H <= Sigma_corr <= 0.80` at any hint budget | ceiling 0.65-0.80 | any hint scheme achieving > 0.85 hit-rate gain on a matched trace | THEORY.md 7 |
| P12 | prefetch at lead=1 is worth **~0 s** of wall on the sealed bank shape (per prior-art idle-gap bound) | 0-0.3 s/token | an A/B with lead-1 prefetch saving > 0.5 s/token at B_pf <= 0.3 GiB/hint | THEORY.md 7 |

## P13-P15: serving / economics predictions

| id | prediction | numeric range | measurement that kills it | model |
|---|---|---|---|---|
| P13 | H_serve at 16 GiB host drops ~6-12pp from K=1 to K=16 concurrency | -9.5pp nominal, kill: drop < 3pp or > 18pp | multi-request replay at K=1 vs K=16 | THEORY.md 6 |
| P14 | dee on 1xL40S at b=8, host 256 GiB: $/1k tok | $0.07-0.13 (kill: outside $0.05-0.20) | billed Modal run with measured TPS outside 8-28 tok/s at that config | THEORY.md 6 |
| P15 | dense baseline (8xH100, 568 GiB) $/1k tok <= $0.13 **if** it achieves >= 200 tok/s | — | a measured dense DSv4-Flash serving stack achieving > 400 tok/s aggregate at <= $0.26/1k tok would kill the break-even claims | THEORY.md 6 |
| P15b | at `b=8`, host 256 GiB, 2xL4 / 1xL40S / 1xRTX PRO 6000 beat the dense baseline by >= 25% | ratio 0.49-0.93 across `b=8-16` (kill: any >= 0.95 at `b=8`) | same billing + TPS measurement as P14 on those cells | THEORY.md 6 |
| P15c | at `b=1` no cell beats the dense baseline under the anchor-calibrated `t_0` | ratio 1.08-2.29 on GPU cells (kill: any < 0.85 at `b=1`, 256 GiB host) | single-request cost measurement per cell | THEORY.md 6 |

## P16-P17: regime-map predictions

| id | prediction | numeric range | measurement that kills it | model |
|---|---|---|---|---|
| P16 | at B_SSD <= 0.4 GiB/s the limiter is **storage** for every host budget <= 512 GiB (b=1) | 100% of grid points | any matched config at <= 0.4 GiB/s where compute/H2D/dense is the measured limiter (stage profile > storage share) | THEORY.md 8 |
| P17 | at B_SSD >= 5 GiB/s, host 256 GiB, the limiter moves to H2D or compute and TPS is within 2x of the dense-path bound | 0.5-1.0 of dense bound | a >= 5 GiB/s cell still storage-limited at 256 GiB host | THEORY.md 8 |

---

## Data needed (missing artifacts — no substitute numbers used)

| gap | what it blocks | what would fix it |
|---|---|---|
| >= 1k-token multi-request DSv4-Flash trace | steady-state Che validation (both traces here are 16-token / 4-token windows); P2/P7 ranges widen accordingly | one long generation + a small request stream with router journals |
| GPU GEMM microbenchmark (FP4->FP16 dequant, batch sweep) | `COMPUTE_EFF = 0.6` (ASSUMPTION) and all cross-cell TPS predictions (P1, P14) | a 1-line GEMM bench on each target GPU; range 0.3-0.8 is used meanwhile |
| router traces for MiMo-V2.6-Flash/Pro and MiniMax-M3 | H(M) curves for those stores use **transferred** DSv4-Flash popularity (`S_TRANSFER`, ASSUMPTION) | one 2k-request router trace per model |
| measured dense-residency baseline (DSv4-Flash on H100-class) | break-even claims P15 | a published or self-run serving measurement at stated prices |
| tuned CPU (AVX2) expert kernel | CPU-only cell uses portable-torch 2,750 ms/expert (MEASURED) and cannot represent an optimized kernel | the kernel + one bench |
| Phase-2 working-set sim's own CSVs (branch `research/phase2-ws-policy`, not in this worktree) | direct diff against its rows; landmarks are cross-checked via AGENTS.md recorded values + independent replay (exact match on 4/5 landmarks) | import its `data/` into the worktree |

## Known model failures (reported, not hidden)

1. **Steady-state Che is invalid on short windows**: over-predicts hits by
   +25pp at 16 GiB on the sealed trace (`data/cache_sim_check.csv`); the
   finite-window correction is used for all anchor/prediction work and
   *under-predicts* by up to -7pp at small budgets. The two bracket the
   measured curve (P4 encodes this).
2. **Zipf-Mandelbrot degenerates**: the MLE is not a power law on either
   trace (P6). Quoting a "Zipf exponent s" from these traces is unsupported.
3. **Every family over-predicts the extreme head** (ranks 1-5 hold 1.55% of
   sealed traffic vs 3-10% predicted) **and the tail** (empirical mass is 0
   below the seen set): the true rank curve is flatter-at-top and shorter
   than any fitted family (`data/popularity_tail.csv`).
4. **Anchor pipeline bound is 26x optimistic** (5.6 tok/s vs 0.21 measured):
   perfect overlap is not realized (eta = 0.038). The overlap-corrected
   prediction (0.19-0.23) is the one that fits; the bound is labeled a bound.
5. **Poisson-lognormal is not on the multinomial scale** (its `loglik` is NaN
   by construction); it is reported on its own scale and never compared
   directly to Zipf/uniform.
