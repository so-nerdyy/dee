# DEE-Theory — an analytical performance model for exact sparse-MoE inference

**Status:** research report, generated artifacts reproducible via `python -m theory.run_all`.
**Scope:** the dee hierarchy `NVMe -> host RAM -> VRAM -> accelerator` under the
exactness contract (authoritative routing, expert identity, checkpoint
representation, expert ordering and outputs preserved; prediction may only drive
prefetch hints). Everything below is derived from in-repo artifacts; every
constant is tagged MEASURED / DERIVED / CALIBRATED / ASSUMPTION in
[`data/provenance.csv`](data/provenance.csv) with its source path. ASSUMPTION
entries are named in §9 and individually ranged in
[`FALSIFICATION.md`](FALSIFICATION.md).

**Thesis this document supports:** hardware requirements scale with *active
parameters, working set, throughput and concurrency* — not with total
checkpoint size. The regime map (§8, `figs/regime_map.png`) and the cost
frontier (§6, `figs/serve_cost_frontier.png`) are the primary figures.

---

## 1. Definitions and store geometry

A **record** is one routed expert's packed weight block at one MoE layer — the
unit of movement through the hierarchy. A **layer-call** is one MoE layer
evaluated at one forward step (for decode, one token; for prefill, a row of
prompt positions). A **request** is the cache lookup for one record induced by
a layer-call; the engine stages each *distinct* record once per layer-call, so
the request stream is the per-layer-call deduplicated rank stream.

| symbol | meaning | DSv4-Flash value (source) |
|---|---|---|
| `L` | cached MoE buckets | 46 (43 main + 3 MTP) — `result.json` `layer_count_executed=43`; AGENTS.md CPU-4/10 |
| `k` | authoritative top-k routed experts | 6 — `routed_experts.jsonl` `topk=6` |
| `N` | record universe `L x E` | 11,776 — `tools/phase3/p3_bench.py` |
| `P_rec` | packed record bytes | 13,369,344 B (12.75 MiB) — `result.json` `average_request_bytes` |
| `R` | rank slots per token | `L x k` = 276 |
| `D(b)` | distinct records requested per token at batch `b` | 258 at `b=1` on the sealed trace |
| `M_v`, `M_h` | VRAM / host resident record capacity | anchor: 281 / 682 slots per GPU |

Three other stores are carried through the same algebra with their own
geometry (`tools/phase3/specs/*.json`): MiMo-V2.6-Flash (48x256 records,
13,369,344 B), MiMo-V2.6-Pro (70x384, 20,054,016 B), MiniMax-M3 (60x128,
113,246,208 B bf16).

**Two reference traces.** (T1) the sealed DSv4-Flash 16-token generation
(`experiments/route_pipeline/fill-live-t4x2-20260909/routed_experts.jsonl`):
43 layers, 946 token-rows, 5,676 rank slots, 2,364 unique records, 5,099
deduplicated cache requests. (T2) the milestone-2.5 router trace
(`benchmark_reports/milestone-2.5/.../expert-trace.jsonl.gz`): 8 runs of a
40-layer x 256-expert, top-8 model, 9,600 authoritative `route_selection`
events plus 8,011 `expert_request` cache events with reuse distances.

---

## 2. Expert popularity law (§A of the brief)

Let `c_i` be the access count of record `i` over a window of `R` requests and
`p_(1) >= p_(2) >= ...` the sorted empirical masses. Candidate families, all
written as rank curves over the *full* category space (never-seen records
carry their own tail mass):

1. **Uniform** `p_(i) = 1/N`.
2. **Zipf (power law)** `p_(i) = i^(-s)/Z`.
3. **Zipf-Mandelbrot** `p_(i) = (i+q)^(-s)/Z(s,q,N)`.
4. **Lognormal rank curve** `p_(i) ∝ exp(mu + sigma z_i)`, `z_i = Phi^-1(1 - (i-0.5)/N)` —
   the lognormal-on-frequency family written as a ranked rate profile.
5. **Poisson-lognormal abundance** `c_i ~ Poisson(L f_i)`, `f_i ~ LogNormal` —
   a *different sampling model* (species abundance), reported on its own
   likelihood scale (`loglik_poissonmix`), not comparable to 1-4.

**Likelihood convention.** Category labels are exchangeable under every family,
so the maximized multinomial likelihood assigns the largest `p_(i)` to the
largest observed count (the *sorted assignment*). The multinomial coefficient
cancels across families: the reported log-likelihoods compare families but are
not absolute generative scores. The KS-style statistic is
`D = max_m |C_emp(m) - C_model(m)|` on the coverage curves (§2.3).

### 2.1 Goodness of fit — pooled (T1 sealed, T2 Ornith), `data/popularity_fits.csv`

| profile | family | log-lik | AIC | KS-D | parameters |
|---|---|---:|---:|---:|---|
| sealed_dsv4 (R=5,099, N=11,008) | uniform | -47,453.2 | 94,906 | 0.785 | — |
| | zipf_pure | -40,440.1 | 80,882 | 0.291 | s = 0.819 |
| | **zipf_mandelbrot** | **-38,379.4** | **76,763** | **0.088** | s = 5.3e8, q = 3.6e11 (degenerate) |
| | lognormal_rank | -38,995.7 | 77,995 | 0.148 | mu = 1.46, sigma = 1.84 |
| | poisson_lognormal | (own scale) | — | 0.304 | mu = -10.7, sigma = 2.44 |
| ornith pooled (R=9,600, N=10,240) | uniform | -88,646.9 | 177,294 | 0.795 | — |
| | zipf_pure | -75,478.6 | 150,959 | 0.298 | s = 0.821 |
| | **zipf_mandelbrot** | **-70,031.0** | **140,066** | **0.028** | s = 1.6e10, q = 8.7e12 (degenerate) |
| | lognormal_rank | -72,428.7 | 144,861 | 0.152 | mu = 1.46, sigma = 1.86 |
| | poisson_lognormal | (own scale) | — | 0.722 | mu = -11.9, sigma = 3.79 |

**Which family fits.** Zipf-Mandelbrot wins on both traces by AIC and by the
coverage statistic. But the fitted `(s, q)` is *not* a power law: the optimizer
runs to `q -> 3.6e11` with `s/q` = a decay rate of **0.146% per rank** (T1) and
**0.185% per rank** (T2), with the power-law bend at rank `q^2/s ~ 2e14` — far
beyond the 11k-record universe. The MLE has collapsed to the **geometric
(exponential-in-rank) limit** of the Zipf-Mandelbrot family. Both traces are
flagged `degenerate_to_geometric=True` in the CSV. So the honest statement is:

> **Expert popularity is not a power law on either real trace. A
> two-parameter geometric decay over ranks — equivalently a light-tailed
> "lognormal-ish" bump with a finite working set — fits 2-3x better in
> likelihood than pure Zipf and ~10x better than uniform, and the Zipf-
> Mandelbrot MLE degenerates to it.** The lognormal rank curve is the best
> *non-degenerate* named family (ll within 3.4% of the degenerate optimum at
> KS-D 0.15), and the Poisson-lognormal abundance fit confirms the shape on a
> separate sampling model.

This matters: a genuine power law would make the cache story hopeless (heavy
tail = unbounded working set). A geometric decay over ranks is what makes a
~16 GiB knee (§3) the *right* order of magnitude for a 147 GiB universe.

**Where the fits fail** (`data/popularity_tail.csv`). All families over-
predict the extreme head (ranks 1-5 hold 1.55% of T1 traffic — 3-6x less than
any smooth family predicts) and every family assigns mass to the lower half of
the rank range where the empirical mass is exactly 0 (those records were never
routed in the window). The empirical rank curve is therefore *flatter than any
family at the very top and shorter than any family at the bottom*: a finite
working set with mild, bounded skew.

### 2.2 Per-layer fits and uncertainty

Per-layer Zipf-Mandelbrot fits (`data/popularity_fits.csv`, 129 layer rows for
T1) sit at decay rates of the same order with wide scatter; layers with < 50
observations are flagged `well_supported=False` and their parameters should
not be quoted. Parametric bootstrap (60 access-level resamples, warm-started
refits; `data/popularity_bootstrap.csv`) gives tight intervals on the
*non-degenerate* parameters — lognormal rank: T1 mu = 1.373 [1.296, 1.446],
sigma = 1.962 [1.960, 1.964] — and enormous ones on the degenerate `(s, q)`,
which is itself evidence of degeneracy (the ridge `s/q = const` is flat).

### 2.3 Entropy and coverage

| profile | H (bits) | log2 N | effective experts `2^H` |
|---|---:|---:|---:|
| sealed_dsv4 pooled | 10.68 | 13.43 | 1,635 |
| ornith pooled | 10.47 | 13.32 | 1,418 |
| sealed_dsv4 per-layer | 4.35 - 6.54 (mean 5.24) | 8.00 | 20 - 93 |

Coverage `C(m) = sum_{i<=m} p_(i)` (`figs/popularity_coverage.png`): the top-5%
of records capture 58.5% of T2 traffic and the top-10% capture 83.5% (T1's
window saturates even faster: 2,364 of 11,008 records carry 100% of its
traffic by construction). Uniformity is decisively rejected (KS-D 0.79).

---

## 3. The cache model: closed form vs the sealed simulation (§B)

### 3.1 Che's common-TTL residence model (closed form)

Under an **Independent Reference Model** (IRM: requests i.i.d. over records with
probabilities `w_j`), give every resident record the same lifetime `tau`
measured in request steps. Record `j` is requested at rate `w_j`, so the
probability its inter-reuse gap exceeds `tau` is `exp(-w_j tau)`. A record is
resident exactly when its backward recurrence time is under `tau`; by the
exponential gap law

```
h_j(tau) = 1 - e^(-w_j tau)            per-record hit probability
M(tau)   = sum_j (1 - e^(-w_j tau))    expected resident records
H(M)     = sum_j w_j (1 - e^(-w_j tau(M)))
```

with `tau(M)` the unique positive root of `M(tau) = M` (monotone; solved by
bisection, or by a tabulated monotone inversion — `theory/cache.py CheTable`).
This is **Che's TTL approximation** to LRU; nothing here is novel machinery.

**Finite-window correction.** Che's steady state assumes a long stationary
window. A generation of `R` requests touching `U` distinct records can hit at
most `(R - U)/R` of its accesses no matter the capacity. Counting each record's
first touch as a compulsory miss gives the per-record finite form

```
hits_j(tau)   = (n_j - 1)(1 - e^(-w_j tau))        n_j = observed count of j
misses_j      = n_j - hits_j
```

which converges to Che as `n_j -> inf`. On T1 (`R=5,099`, `U=2,364`) the
ceiling is **53.6%** and this correction dominates.

**Two levels.** The host level sees only the VRAM miss stream. With `u_i` the
per-token request probabilities and `h_v,i = 1 - exp(-w_i tau_v)`,

```
host stream:  u_i (1 - h_v,i),  tau_h solved on the renormalized weights
cold_i        = u_i (1 - h_v,i)(1 - h_h,i)         (SSD record reads/token)
```

### 3.2 Reproducing the sealed working-set simulation

The phase-2 working-set sim's measured landmarks (AGENTS.md: causal pooled LRU
knee **~16 GiB**, LRU **~2.8pp** below offline MIN there, reaches MIN by
**~32 GiB**, **935/2364** records ever repeat) were produced on the sealed
trace. `data/cache_sim_check.csv` re-derives them independently by exact LRU /
Belady-MIN replay on the per-layer-call deduplicated stream (5,099 requests):

| landmark | measured (phase-2 ws-policy) | recomputed here | verdict |
|---|---|---|---|
| records ever repeating | 935 / 2,364 | **935 / 2,364** | exact |
| LRU knee | ~16 GiB pooled | **16.0 GiB** | exact |
| LRU - MIN at knee | ~2.8 pp | **-2.82 pp** | exact |
| LRU reaches MIN | ~32 GiB | 24-28 GiB on our budget grid (0.5pp at 24 GiB, 0.04pp at 28) | consistent (grid resolution) |
| LRU hit at 16 GiB | — | 50.8% (MIN 53.6%) | — |

**Where the knee lands, analytically.** The knee is where the *reusable* mass
is captured: only 935 of 2,364 records can ever hit (each contributes at most
one compulsory miss plus one hit per reuse), so the LRU hit rate saturates at
`(5,099 - 2,364)/5,099 = 53.6%` — exactly the measured MIN plateau. The top
1,285 records (16 GiB) cover the dense half of the reuse-distance distribution
(`figs/cache_hitrate.png`, stack-distance coverage curve), and the remaining
reuse is spread thinly enough that 24-32 GiB only buys the last 2.8pp. The Che
closed form agrees with the *shape*: its H=50% point for the full
11,776-record universe is **11.2 GiB**, H=80% at 19.0 GiB, H=95% at 30.6 GiB
(`data/cache_store_knees.csv`) — the same 16-32 GiB band.

### 3.3 Where the closed form fails (honestly)

| budget | LRU (real) | MIN | shuffled-LRU (IRM control) | Che steady | Che finite |
|---|---:|---:|---:|---:|---:|
| 8 GiB | 0.404 | 0.536 | 0.340 | 0.458 | 0.333 |
| 16 GiB | 0.508 | 0.536 | 0.478 | 0.727 | 0.475 |
| 24 GiB | 0.531 | 0.536 | 0.529 | 0.906 | 0.528 |
| 32 GiB | 0.536 | 0.536 | 0.536 | 1.000 | 0.536 |

Three separate errors are visible and all are reported
(`data/cache_summary.json`):

1. **IRM error (temporal correlation)** = real LRU - shuffled LRU =
   **+3.1pp at 16 GiB**, +6.4pp at 8 GiB, 0 at 32 GiB (`temporal_irm_delta.csv`).
   Correlation *helps* LRU slightly and vanishes once the working set fits.
2. **Che steady-state error** vs the IRM control = +25pp at 16 GiB. This is
   *not* an IRM failure: it is the compulsory-miss artifact of a 5,099-request
   window, and the finite-window form removes it (Che finite - shuffled =
   -0.3pp at 16 GiB).
3. **Policy gap** LRU - MIN = -2.8pp at the knee, matching the phase-2 number.

**Verdict:** the closed form reproduces the sim's landmarks within the stated
tolerance when the finite-window correction is applied; the steady-state form
is only valid for long stationary streams and is *not* used for the anchor
prediction (§5).

---

## 4. Temporal structure — prefetch-relevant statistics (§C)

**Reaccess survival** `P(record reaccessed after lag d)` over cache-request
lags (`data/temporal_survival.csv`, `figs/temporal_survival.png`): a steep
drop over the first ~10 requests followed by a thin tail — consistent with the
geometric-decay popularity finding and with 935/2,364 records ever repeating.

**Within-token cross-layer conditional** `P(j in S_{l+1} | i in S_l)`
(`data/temporal_crosslayer.csv`, support >= 5 on the source expert):
**11,858** supported pairs on T1 (228 source experts) and 4,213 on the T2
long-prompt run. Mean lift over the marginal is **12.2x (T1)** and 10.7x (T2),
median 11.2x / 9.8x, max 24.6x. Cross-layer routing is strongly correlated
within a token — this is real, exploitable structure, and it is exactly the
structure a prefetch hint may legally use.

**Decode-step self-correlation** `P(expert at t+d | expert at t)` per layer
(`data/temporal_selfcorr.csv`, `figs/temporal_selfcorr.png`): at lag 1 the
estimate sits ~1.4-5.3x above the IRM baseline `sum_i p_i^2` per layer
(e.g. layer 0: 0.016 observed vs 0.012 IRM baseline at d=1, 0.063 vs 0.012 at
d=3), decaying with lag. The consecutive-token reuse is the ~37.5% overlap
already measured in `CACHE1_ANALYSIS.md` §2.

**IRM-vs-correlated hit-rate delta** (`data/temporal_irm_delta.csv`): the
effective value of temporal correlation is +3.1pp (T1, 16 GiB) up to +10.7pp
(T2 long-prompt, 8 GiB), zero by 32 GiB. Correlation is worth more the tighter
the cache — which is exactly where prefetch can act.

---

## 5. Throughput roofline (§D)

### 5.1 The bound

For one generated token at batch `b`, with `u_i(b) = 1 - (1 - pi_i)^b`
(`pi_i` = inclusion probability of record `i` per layer-call; top-k picks are
*distinct*, so this is the without-replacement form, not `k(1-H)`):

```
D(b)      = sum_i u_i(b)                        distinct records requested
M_v       = D(b) (1 - H_v)                      device fills  (host lookups)
M_h       = D(b) (1 - H_v)(1 - H_h)             cold SSD reads
cold_bytes  = P_rec * M_h
spill_bytes = P_rec * M_v
```

One miss serves the whole batch — the batch-overlap correction is `u_i(b)`,
never `k (1-H)`. Then

```
T_tok(b, M_h, M_v) = max( t_storage, t_h2d, t_compute, t_dense )

t_storage = max( M_h * P_rec/B_SSD , M_h * t_service )   (t_service = P_rec/B_SSD:
             the demand-gated dependency path, storage verdict §4: <= k legal
             reads per layer call, pread ~= service, 96% device busy)
t_h2d     = spill_bytes / B_H2D
t_compute = D(b) t_0 / b + L k W_touch / F_eff            (t_0 = per-touch
             launch+gather overhead measured at b=1; W_touch = 50,331,648 FLOP)
t_dense   = t_dense,measured (batch shape)
```

`T_tok` is a *pipeline-max bound*: it assumes perfect overlap and is therefore
a lower bound on latency. Because fills are demand-gated by the layer
dependency chain, the realistic prediction adds the unhidden fill service:

```
T_pred = t_dense + t_compute + (1 - eta_s) M_h t_service + (1 - eta_h) t_h2d
```

with `eta_s` the fill-over-compute hiding fraction, **CALIBRATED once** on the
anchor run at 0.33 GiB/s (`eta_s = 0.292`, `eta_h = 0.5` ASSUMPTION) and then
held fixed as a prediction across the other bandwidth cases.

### 5.2 The empirical anchor: 0.21 tok/s at 0.29-0.37 GiB/s on 2xT4

The measured anchor (`result.json`: `decode_tok_s=0.21`, 15 decode tokens in
71.479 s, storage 0.29-0.37 GiB/s per `research/route-pipeline/STORAGE_VERDICT.md`).

**Miss counts first** (`data/anchor_check.json`, `miss_model_comparison`). The
steady-state Che model *under*-predicts the cold reads of a 16-token
generation: **67.84** cold SSD records per decode token against a measured
83.47 (rel err **-18.7%**) — the "46 cold records (-45%)" this section
previously asserted is a **text/artifact mismatch**: no 46 appears in the
cited artifact, whose steady-state row is 67.84/-18.7% (Phase-7 correction,
recorded with the old claim visible here). The finite-window model
*over*-predicts at **161.96** (**+94.0%**). The measured value sits between
the two bracketing models — consistent with the fact that the run is neither
stationary (cold start, 2,364-record universe touched once) nor purely
compulsory (host tier absorbs **51.3%** of host lookups: `host_pack` hits
2,618/5,099 — note 2,618/5,099 **is** 51.3%, not the 23-25% this document
previously asserted; only the cuda0-GPU share of the pooled stream
(1,223/5,099 = 24.0%) lands in that band; all eight share definitions are
computed in `data/anchor_gate.json` `host_share.definitions`).

**Token rate** (decode accounting, finite-window model, `data/anchor_check.csv`):

| B_SSD case | T_pred (s) | TPS predicted | measured | ratio meas/pred |
|---|---:|---:|---:|---:|
| 0.29 GiB/s | 5.36 | **0.187** | 0.21 | 1.13 |
| 0.33 GiB/s (calibration) | 4.77 | **0.210** | 0.21 | 1.00 |
| 0.37 GiB/s | 4.29 | **0.233** | 0.21 | 0.90 |

**The model predicts the anchor within 2x at every point of the measured
bandwidth band (ratios 0.90-1.13)** — and the whole bandwidth sensitivity
across 0.29-0.37 GiB/s is reproduced without refitting. The pipeline-max
*bound* alone is 5.6 tok/s (ratio 0.038): the anchor run is **not** within 2x
of the bound because the bound assumes perfect overlap and the prototype
realizes `eta = 0.038` of it (stage profile `everything_else` bucket: 153 s of
unattributed orchestration on cuda0). The prefill-inclusive view is the
cleanest storage statement: 16 tokens over 163.4 s at 2.07 GB storage bytes
per emitted token gives a storage-only bound of 0.062 tok/s vs 0.098 measured
— **ratio 1.57x, the run is ~80% storage-service bound** once prefill is
included. Both facts are reported; neither is hidden.

---

## 6. Serving extension and the cost frontier (§E)

**Cross-request mixture.** With Poisson arrivals at rate `lambda`, mean
response time `W`, and Little's law `K = lambda W` concurrent requests, the
pooled reference stream mixes `K` per-request popularity vectors. Under an IRM
with homogeneous requests the mixture is `u_i(K b) = 1 - (1-pi_i)^(K b)`, so
Che applies to the mixture with the *same* capacity: `H_serve(M, lambda) =
H(w_bar(K), M)` (`data/serve_hitrate.csv`, `figs/serve_hitrate_vs_concurrency.png`).
At fixed host budget, per-request working set grows with `K` and the hit rate
falls: 63.9% at `K=1` down to 54.4% at `K=16` for 16 GiB — the price of
concurrency is ~9.5pp of hit rate at that budget, while at 256 GiB the pooled
universe is fully covered and the loss vanishes. Slow-tier bytes/token follow
(`figs/serve_bytes_per_token.png`).

**Cost frontier** (`data/serve_cost_frontier.csv`, `figs/serve_cost_frontier.png`).
`$/1k tok = hardware_rate_s / (TPS_pred * 1000)`, with the Modal price
schedule from the brief as ASSUMPTION and `COMPUTE_EFF = 0.6` of vendor dense-
tensor peak (ASSUMPTION, ranged 0.3-0.8 in FALSIFICATION):

| cell | TPS (b=8, host 256 GiB) | limiter | $/1k tok |
|---|---:|---|---:|
| 2xT4 (anchor) | 1.31 | H2D | 0.527 |
| 1xL4 | 6.75 | H2D | 0.140 |
| 2xL4 | 14.6 | compute | 0.090 |
| 1xA10 | 6.76 | H2D | 0.152 |
| 1xL40S | 15.0 | compute | 0.095 |
| 1xRTX PRO 6000 | 15.3 | compute | 0.120 |
| CPU-only | 0.011 | compute | 93.8 |

Compute-bound cells carry the anchor-calibrated per-touch dispatch cost
(387 us, assumption 7); with a batched-kernel path (50-150 us) their TPS
rises 2-6x and their $/1k tok falls proportionally.

**Dense-residency baseline** (`data/serve_dense_baseline.json` — all
ASSUMPTION, deliberately conservative *toward the competitor*): DSv4-Flash
bf16 resident ~568 GiB on 8xH100-80GB at $0.003223/GPU/s, achieved 200 tok/s
aggregate decode -> **$0.129/1k tok**. We do not measure this baseline in-repo
and we do not claim to. The 200 tok/s figure is a *high* achieved-throughput
assumption for the competitor (a weaker competitor would flatter dee), and the
H100 price is list-rate cloud pricing, not reserved/spot — both choices push
against dee's advantage. A baseline achieving less than 200 tok/s would only
widen dee's margin; a baseline achieving more is the live falsification risk
(FALSIFICATION.md, P7).

**Break-even** (`data/serve_breakeven.csv`, `figs/serve_breakeven.png`): at
host 256 GiB the honest picture under the conservative per-touch cost is that
dee does **not** win at `b=1` anywhere (ratio 1.1-1.8 on GPU cells; the cold
record traffic is fully exposed there), wins marginally-to-clearly at `b>=8`
on the compute-bound multi-GPU cells (2xL4 0.70 at b=8 and 0.49 at b=16;
1xL40S 0.74/0.51; 1xRTX PRO 6000 0.93/0.64), and loses everywhere on 2xT4
(2.3-5.2x — the anchor platform is a correctness instrument, not a cost
performer) and CPU-only (360-5,800x worse, matching the measured 2,750
ms/expert portable-torch CPU execution). The single-GPU H2D-bound cells
(1xL4, 1xA10: ratio 1.08-1.34) are the knife's edge: their fate is decided by
whether a batched dispatch path delivers the 50-150 us touch cost of
assumption 7, which would push them to ~0.5-0.7x of the baseline. Volume
storage for the full 146.6 GiB universe costs $3.8/mo above the 1 TiB free
tier (`data/serve_volume_cost.json`) — negligible against compute.

---

## 7. The prefetch bound (§F)

Prediction is legal only as a **hint**. Given the observed set `S_l`, the legal
lookahead is `S_{l+1}` (lead = 1 layer-call). With empirical conditionals
`P(j in S_{l+1} | i in S_l)` (§4), the candidate probability is

```
p_j(S_l) = 1 - prod_{i in S_l} (1 - P(j | i))    (conditional independence)
```

**Optimal prefetch set.** With equal record costs `P_rec` and a budget of `B`
records, the one-shot expected miss reduction `f(S) = sum_{j in S} p_j` is a
*linear objective under a uniform-cost budget constraint* — greedy top-`B` by
`p_j` is **optimal**. With heterogeneous costs it becomes 0-1 knapsack (NP-hard;
density-greedy is a 1/2-approximation). If a prefetched record also covers
correlated records, `f(S) = sum_j q_j [1 - prod_{i in S}(1 - c_ij)]` is
**monotone submodular** (probabilistic coverage) and greedy by marginal gain per
byte carries the **(1 - 1/e) Nemhauser-Wolsey-Fisher guarantee**. Both regimes
are computed (`data/prefetch_budget_sweep.csv`, `data/prefetch_submodular.csv`).

**Measured predictor quality** (`data/prefetch_predictor.csv`,
`figs/prefetch_predictor.png`) on all adjacent layer-call pairs of T1:

| m cands/source | recall | precision | announced | bandwidth |
|---:|---:|---:|---:|---:|
| 1 | 0.227 | 0.275 | 6.1 records/call | 0.08 GiB/call |
| 2 | 0.360 | 0.229 | 11.6 | 0.14 GiB |
| 4 | 0.545 | 0.183 | 22.0 | 0.27 GiB |
| 8 | 0.729 | 0.131 | 41.2 | 0.51 GiB |

**The provable ceiling** `Delta H <= Sigma_corr` = 0.729 at `m=8`: prefetch can
only convert accesses predictable from already-observed information into
earlier hits. Compulsory first touches (2,364 records on T1) are immovable, and
a wrong hint wastes bandwidth **without changing execution** (exactness
contract).

**What prefetch can never do under the exactness contract:** it cannot change
which expert executes, cannot substitute a predicted route for the router,
cannot prune or merge, and cannot buy correctness with probability — a wrong
hint is only wasted `B_pf` bytes. The in-repo prior-art gate (AGENTS.md
research assimilation) requires lead >= 2 **and** precision >= 0.75 to clear
the idle-gap bound; this measured next-layer predictor reaches precision 0.28
at lead 1 and **fails the gate**, consistent with the in-repo verdict that
k=1 mechanisms are worth 0 s on the sealed bank.

---

## 8. Sensitivity and the regime map (§G)

**Elasticities** `d log TPS / d log x` (`data/sensitivity.csv`,
`figs/sensitivity.png`), anchor cell, both horizon regimes:

| input | steady state (512 tok) | finite window (16 tok) |
|---|---:|---:|
| record size `P_rec` | -0.49 | -0.49 |
| dense-path cost | -0.93 | -0.50 |
| host capacity | +0.45 | 0.00 |
| locality exponent `s` | +0.04 | 0.00 |
| VRAM slots | +0.002 | 0.00 |
| B_SSD, B_H2D | 0.00 | 0.00 |

Reading: at the anchor configuration the finite-window (compulsory-dominated)
regime is insensitive to *capacity and bandwidth* and only responds to record
size and the dense path — because a 16-token generation cannot reuse anything
the caches cannot already hold, and its cost is fixed record traffic. In the
steady-state regime host capacity becomes the lever (elasticity +0.45) and the
dense path dominates. **The single most effective lever across both regimes is
record size** (elasticity -0.49 in both): fewer physical bytes per expert beats
every bandwidth or capacity purchase. This is the quantitative form of Phase-1's
"fewer physical cold bytes + earlier legally-possible work" verdict.

**Regime map** (`figs/regime_map.png`, `data/regime_map.csv`): limiter over
`(B_SSD, M_host)` per (model, cell) at `b=1` in the finite-window regime.

**Phase-1 consistency check** (`data/regime_phase1_check.json`): at the
measured bank bandwidth 0.29-0.37 GiB/s on the 2xT4 cell, the model classifies
**30/30 grid points as storage-limited** — the Phase-1 verdict
("storage-bandwidth constrained, ~0.3 GiB/s, feed-side") lands *inside* the
storage-bound region. **CONSISTENT; not flagged.**

The map's other cells show the dee thesis directly: at >= 5 GiB/s storage the
limiter moves to H2D or compute and TPS becomes capacity-insensitive — the
hierarchy stops being a storage problem and becomes an ordinary serving
problem, while the record universe stays 147 GiB.

---

## 9. Constants, assumptions, limitations

**Provenance.** [`data/provenance.csv`](data/provenance.csv) lists every
constant with `MEASURED` (in-repo artifact path), `DERIVED` (arithmetic on
MEASURED values, shown), `CALIBRATED` (fit with residual stated) or
`ASSUMPTION` (with range). No number in this document lacks a row there.

**ASSUMPTIONS carried into predictions** (each falsifiable in FALSIFICATION.md):

1. `COMPUTE_EFF = 0.6` of vendor dense-tensor peak for FP4->FP16 dequant GEMMs
   (range 0.3-0.8); no in-repo GPU microbenchmark exists (**data needed**).
2. Vendor bandwidths/peaks for L4 / A10 / L40S / RTX PRO 6000 and the Modal
   price schedule (as the brief stipulates).
3. `eta_h2d = 0.5` H2D-over-compute hiding; `t_dense` batch shape (35% fixed).
4. Popularity-shape transfer from DSv4-Flash to MiMo/MiniMax stores
   (`S_TRANSFER`) — no router traces for those models exist in-repo
   (**data needed**).
5. Dense-residency baseline: 568 GiB bf16, 8xH100, $0.003223/GPU/s, 200 tok/s
   achieved (deliberately conservative toward the competitor).
6. Serving: Poisson arrivals, `W = 4 s`, 512-token requests; cross-request
   mixture under homogeneous IRM.
7. The per-touch implementation cost `t_0` = 387 us is **CALIBRATED** from the
   anchor run's unbatched dispatch path (`use_batched_experts=false`) and
   carried unchanged to all cells; on a cell with batched expert kernels the
   true `t_0` could be 50-100 us, which moves compute-bound cells (2xL4,
   L40S, RTX PRO 6000 at `b=8`) to 2-6x higher TPS (FALSIFICATION P8b).
   The anchor cell and the storage/H2D-bound cells are insensitive to it.

**Limitations.** (a) Both reference traces are short windows (16 tokens and
~4-token runs): the finite-window correction is mandatory and the steady-state
closed form is not validated against a long stationary stream (**data needed**:
a >= 1k-token multi-request trace). (b) The anchor's `eta_storage = 0.292` is
calibrated on one platform; it is applied unchanged to other cells and is the
weakest link in the cross-cell predictions. (c) The two-level independence
approximation (host sees exactly the VRAM miss stream) is exact in the finite
form and approximate in the steady state. (d) The degenerate Zipf-Mandelbrot
fit means *no* power-law exponent should be quoted from these traces. (e) The
CPU cell uses one measured point (2,750 ms/expert portable-torch) and cannot
represent a tuned AVX2 kernel (**data needed**).

---

## 10. Reproduction

```
cd dee.cpp/theory
python -m theory.run_all            # all stages, ~4-6 min
python -m theory.run_all cache D    # subsets by name or letter (A-G)
```

Deterministic (fixed RNG seed 20260915); no network access; writes only
`theory/data/` and `theory/figs/`. Package layout: `theory/sources.py`
(loaders), `constants.py` (provenance), `popularity.py` (A), `cache.py` (B),
`temporal.py` (C), `roofline.py` (D), `serving.py` (E), `prefetch.py` (F),
`sensitivity.py` (G).
