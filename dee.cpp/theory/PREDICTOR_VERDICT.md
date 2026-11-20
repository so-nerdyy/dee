# PREDICTOR_VERDICT.md — real next-layer / next-step predictors, real verdict on P9/P10

Phase-7 component B (predictor lab). Everything below is a **prefetch-hint**
measurement: a prediction may only stage records early as a hint; the native
router stays authoritative and a wrong hint wastes bandwidth, never changes
execution (AGENTS.md exactness contract). All numbers are out-of-sample on
time-ordered splits, produced by `python -m theory.run_all pred` (fixed seeds,
torch CPU, 1 thread) from `dee.cpp/theory/` (~170-195 s wall).

---

## 1. Setup

### Data

| trace | content | windows | window shape | record size |
|---|---|---|---|---|
| **T2 Ornith** (`benchmark_reports/milestone-2.5/.../expert-trace.jsonl.gz`) | 9,600 authoritative `route_selection` rows, 40 layers x 256 experts, top-8 | 3 distinct complete routing streams + 1 dropped prefix (the 8 runs carry only **4** distinct streams!) | long-prompt run: 10 prefill + 1 decode step; others 1-4 steps | **6,291,456 B** (`expert_request.expert_bytes`, all 8,011 rows) |
| **T1 sealed DSv4** (`experiments/route_pipeline/fill-live-t4x2-20260909/routed_experts.jsonl`) | 688 layer-calls, 43 layers x 256 experts, top-6 (16-token journal) | 1 stream | 1 prefill + 15 decode steps | 13,369,344 B |

**Disagreement (recorded, not suppressed):** the task brief carried
"13,369,344 B on Ornith-class geometry", but the Ornith trace itself records
**6,291,456 B** per expert on every `expert_request` row. Both are reported in
`data/pred_byte_value.csv`; the saved/consumed **ratio is invariant to record
size**, only the absolute byte columns change.

### Leak prevention and the split (fixed BEFORE any test number)

The first structural analysis found that runs 1-3 (`dual-cold-primary`,
`dual-warm-profiled`, `dual-warm-reference-present`) carry **byte-identical**
routing streams, and runs 4/5/8 carry one other stream (run 8's stream is a
prefix of runs 4/5's). Any run-level "train on runs 1-5 / test on 6-8" split
would therefore have leaked test routing into training. The split was fixed
by **stream identity** (exact definition in `data/pred_splits.json`, declared
in `theory/pred/__init__.py` before evaluation):

- **T2**: TRAIN = stream groups 1-2 (240 layer-calls after dedup); TEST =
  `dual-long-prompt` (440 layer-calls, disjoint run id, strictly later in
  trace time, max Jaccard 0.081 vs train); `dual-one-token` (prefix) dropped.
- **T1**: strict time cut. TRAIN = forward steps 0-10 (462 `next_layer` /
  430 `next_step` examples); TEST = steps 11-15 (210 / 172). Cross-boundary
  `next_step` training examples whose label sits in the test window are
  dropped (step 10 -> 11 never used in fitting).
- Vocabulary is fixed by geometry (256 experts/layer), never learned from
  data. Hyperparameters fixed a priori in `theory/pred/__init__.py`
  (`pred_logreg`, `pred_mlp`, `pred_xlayer`); no test-driven selection.

Two prediction tasks (both lead = 1): **next_layer** (predict `S_{t,l+1}` from
`S_{t,l}` + history, the P9/P10 surface) and **next_step** (predict
`S_{t+1,l}` from `S_{t,l}` + history).

### Arms (6; every arm emits a ranked candidate list per source context)

| arm | model | notes |
|---|---|---|
| `popularity` | per-target-layer empirical prior | cheap baseline |
| `cond_pooled` | empirical-conditional baseline, exactly THEORY.md §7: `p_j(S_l) = 1 - prod_{i in S_l}(1 - P(j|i))`, expert-id keyed, min support 5 | the arm P9/P10 were derived from |
| `cond_perlayer` | same estimator keyed per layer-pair, backoff per-layer -> pooled -> prior | |
| `logreg_pairwise` | hashed-feature logistic regression (scikit-learn SGD, 1024 hash dim, 256 one-vs-rest classes per context) | shared across layers |
| `mlp_set` | CPU torch MLP on multi-hot history features (current/prev/prev2 same-token sets, prev/prev2 same-step sets, layer, mean rank/weight) -> 256-way BCE | |
| `xlayer_bag` | within-token cross-layer arm: learned set embeddings of layers l, l-1, l-2 in the same token -> 256-way BCE | the "(3) within-token cross-layer predictor" of the brief |

**Budget semantics (both reported; P9/P10 wording is ambiguous between them):**

- **set** = `m` announced records per source CALL (budgeted max-coverage, the
  THEORY.md §7 knapsack semantics; the literal home of "precision at m=8").
- **src** = `m` candidates per source EXPERT, unioned (the semantics
  `theory/prefetch.py::predict_candidates` and the P9/P10 **ranges** were
  derived in). At m=8 this announces 35-51 records/call on T2 (|S_l| = 8),
  34-38 on T1 (|S_l| = 6).

---

## 2. Results

### 2.1 T2 Ornith, next_layer (= the P9/P10 surface), test = long-prompt run

429 test calls, 11 token clusters, 3,432 target records. Precision =
hits/announced; recall = hits/targets. Brackets = 95% token-cluster bootstrap
(400 resamples, seed 20260915).

**Budget semantics "set" (m records per call):**

| arm | P@4 | P@8 | P@16 | R@4 | R@8 | R@16 |
|---|---|---|---|---|---|---|
| popularity | 0.189 [0.143,0.242] | 0.150 [0.104,0.204] | 0.118 [0.079,0.171] | 0.095 | 0.150 | 0.236 |
| cond_pooled | 0.121 [0.049,0.222] | 0.118 [0.046,0.219] | 0.091 [0.047,0.147] | 0.060 | 0.118 | 0.181 |
| cond_perlayer | 0.159 [0.076,0.274] | **0.154 [0.076,0.266]** | 0.108 [0.062,0.167] | 0.079 | 0.154 | 0.215 |
| logreg_pairwise | 0.030 [0.024,0.037] | 0.027 [0.021,0.033] | 0.028 [0.024,0.032] | 0.015 | 0.027 | 0.057 |
| mlp_set | 0.169 [0.075,0.290] | 0.151 [0.062,0.267] | 0.101 [0.057,0.160] | 0.084 | 0.151 | 0.202 |
| xlayer_bag | 0.150 [0.070,0.266] | 0.131 [0.062,0.238] | 0.091 [0.052,0.148] | 0.075 | 0.131 | 0.183 |

**Budget semantics "src" (m per source expert):**

| arm | P@8 | R@8 | ann/call | P@16 | R@16 | ann/call |
|---|---|---|---|---|---|---|
| popularity | 0.150 [0.104,0.204] | 0.150 | 8.0 | 0.118 | 0.236 | 16.0 |
| cond_pooled | 0.054 [0.041,0.071] | 0.344 [0.272,0.434] | 51.0 | 0.043 | 0.481 | 88.8 |
| cond_perlayer | 0.061 [0.045,0.085] | 0.355 [0.286,0.446] | 46.7 | 0.047 | 0.494 | 84.1 |
| logreg_pairwise | 0.034 [0.028,0.040] | 0.053 | 12.6 | 0.034 | 0.103 | 24.3 |
| mlp_set | 0.103 [0.052,0.180] | 0.183 | 14.2 | 0.073 | 0.244 | 26.7 |
| xlayer_bag | 0.055 [0.041,0.076] | 0.283 [0.210,0.378] | 41.1 | 0.045 | 0.404 | 71.3 |

### 2.2 T1 sealed DSv4 (secondary surface; 210 test calls, 5 step clusters,
1,260 target records)

**next_layer, "set":** `popularity` P@8 = **0.333 [0.302,0.357]**, R@8 = 0.444;
`cond_perlayer` 0.306 [0.276,0.331] / 0.408; `mlp_set` 0.286 [0.240,0.318] /
0.381; `cond_pooled` 0.249 [0.221,0.268] / 0.332; `xlayer_bag` 0.227
[0.202,0.245] / 0.303; `logreg_pairwise` 0.026 / 0.034.

**next_layer, "src":** best recall `cond_perlayer` R@8 = 0.524 [0.483,0.557]
at P = 0.090 (35.0 ann/call); `cond_pooled` 0.491 / 0.078; `mlp_set` (best
precision) P@8 = 0.189 [0.164,0.207], R = 0.420.

**next_step (m=8), T2:** best P = `popularity` 0.129 [0.097,0.169] (set);
best R = `cond_perlayer` 0.333 (src, 44 ann/call). **T1:** best P =
`popularity` 0.323 [0.287,0.346] (set), `cond_perlayer` 0.310 [0.283,0.333];
best R = `cond_perlayer` 0.580 [0.553,0.607] (src, 34.3 ann/call, P = 0.102).
The next_step surface is the same story at ~+0.02-0.16 precision; it does not
change any verdict below.

Full 288-row table incl. m in {1,2,32}: `data/pred_metrics.csv`.

### 2.3 Byte value: saved cold bytes vs consumed bytes

Definition: a hint is **valuable** only if its record lands in the target set
AND the baseline hierarchy (causal LRU demand replay, no hints) would have
**cold-loaded** that record at that call. Landed-but-resident hints are
*stale*, never-landed hints are *wasted*; both consume full record bytes.
Three baseline residency scenarios (0 / 963 / 5,120 slots) were replayed; the
scenario moves saved counts by <10% (the traces are mostly compulsory
touches), so the m=8 table shows the middle scenario. m=8, per test TOKEN
(T2 = 6 MiB records, T1 = 12.75 MiB records):

| cell | arm (set sem) | saved MB/tok | consumed MB/tok | net MB/tok | saved/consumed | wasted bytes per saved byte |
|---|---|---|---|---|---|---|
| T2 next_layer | popularity | 136.1 | 1,962.9 | **-1,826.8** | 0.069 | 13.5 |
| T2 next_layer | cond_pooled | 163.6 | 1,962.9 | -1,799.4 | 0.083 | 11.0 |
| T2 next_layer | cond_perlayer | 184.7 | 1,962.9 | -1,778.2 | 0.094 | 9.6 |
| T2 next_layer | logreg_pairwise | 24.0 | 1,962.9 | -1,938.9 | 0.012 | 80.7 |
| T2 next_layer | mlp_set | 201.3 | 1,962.9 | -1,761.6 | 0.103 | 8.7 |
| T2 next_layer | xlayer_bag | 175.0 | 1,962.9 | -1,787.9 | 0.089 | 10.2 |
| T1 next_layer | popularity | 516.1 | 4,492.1 | -3,976.0 | 0.115 | 7.7 |
| T1 next_layer | mlp_set | 465.3 | 4,492.1 | -4,026.8 | 0.104 | 8.7 |
| T1 next_layer | cond_perlayer | 486.6 | 4,492.1 | -4,005.5 | 0.108 | 8.2 |

Under "src" semantics it is worse (T2 `cond_pooled` m=8: saved 366.6 vs
consumed 12,505.1 MB/tok, ratio 0.029). The **best ratio anywhere in the
grid** is 0.182 at m=4 (T1 next_step popularity, always-cold baseline); at
m=8 the best cell is 0.134 (T1 next_step popularity) and the best next_layer
cell is 0.115 (T1 popularity); at m=16 the best is 0.090. Full grid (432 rows,
3 residency scenarios, saved/stale/wasted record counts):
`data/pred_byte_value.csv`.

**Wall translation (P12-consistent):** at the sealed bank's measured
0.29-0.37 GiB/s storage ceiling, the marginal hint traffic of the best T2
arm (1,761 MB/tok of pure waste) adds **~4.5-5.9 s/token** of storage time to
save ~201 MB/tok of cold loads (~0.5-0.7 s/token). Hints at these precisions
make the wall **worse by roughly 7-10x their benefit** before any cache
pollution from staging 3,432 extra records into LRU.

### 2.4 Calibration (reliability at m=8, per-candidate landing frequency;
equal-count bins in `data/pred_calibration.csv`, figure
[`figs/pred_calibration.png`](figs/pred_calibration.png))

| arm | ECE T2 next_layer | ECE T1 next_layer | reading |
|---|---|---|---|
| popularity | 0.122 | 0.316 | mildly under-confident (obs > pred) |
| mlp_set | **0.073** | 0.251 | best-calibrated on T2; under-confident low bins, over-confident top bin (pred 0.88 vs obs 0.61) |
| xlayer_bag | 0.410 | 0.668 | badly over-confident |
| cond_pooled | 0.798 | 0.419 | **catastrophically over-confident** (aggregated p_j reaches 1.000 while obs = 0.03-0.24): the conditional-independence product is not a probability |
| cond_perlayer | 0.814 | 0.491 | same failure mode |
| logreg_pairwise | 0.773 | 0.621 | predicts 0.74-0.89 uniformly; observed 0.02-0.04 |

Only `mlp_set` scores are usable as probabilities, and even it over-confirms
its top bin. The empirical-conditional arms' "p_j" values are ranking keys,
not probabilities — never threshold them for hint admission.

---

## 3. The four verdict answers

### (1) Does any legal predictor exceed precision 0.30 at m=8? (P10 kill threshold)

**YES on T1, NO on T2 — the threshold is crossed only on the tiny sealed
surface, by the popularity prior, not by a learned predictor.**

- T2 (the 50k-event trace, 429 test calls): best precision@8 is
  **0.154 [0.076,0.266]** (`cond_perlayer`, set semantics). Nothing crosses
  0.30; even the CI upper bound stays below (0.266). Under src semantics the
  maximum is 0.103 (`mlp_set`).
- T1 (sealed, 210 test calls, 5 step clusters): `popularity` reaches
  **0.333 [0.302,0.357]** at m=8 (set) — the entire CI is above 0.30,
  exceeding the kill threshold by **+0.033 (CI low) to +0.053 (point)**.
  `cond_perlayer` 0.306 [0.276,0.331] and `mlp_set` 0.286 [0.240,0.318]
  straddle it.

Caveats that keep this from being a clean kill: the T1 test window is 5
forward steps of **one prompt** (clusters are heavy-tailed; the CI is a
400-resample cluster bootstrap and should be read as optimistic at n=5), the
crossing arm is a context-free popularity prior (the effect is expert
concentration in one prompt, not predictability learned from history), and
the same arm on T2 sits at 0.150 [0.104,0.204] — **-0.18 point estimate
below** the threshold. **Verdict: P10's "precision > 0.30" kill criterion is
CROSSED (CONFIRMED kill) on the single-prompt sealed surface at m=8, and NOT
crossed on the larger multi-run surface.** Prefetch can be reopened as a
wall-time lever only for workloads that look like T1 (hot expert
concentration within one generation), and §3(2) says even there the bytes do
not pay.

### (2) Does expected saved cold-bytes/token beat the bandwidth the hints consume?

**No, in every arm, every m, every scenario — by at least 5.5x and typically
~10x (worst cell 83x).** Best ratio in the whole grid is 0.182 (m=4, T1
next_step popularity); at m=8 the best cell is 0.134 and the best next_layer
cell is 0.115. Typical m=8: consumed/saved ≈ 9.7/1 (T2 `mlp_set`: 1,962.9
consumed vs 201.3 saved MB/tok, net **-1,761.6 MB/tok**). In wall terms at
the measured bank ceiling the hints add ~4.5-5.9 s/token of marginal storage
time to save ~0.5-0.7 s/token (see §2.3). The byte accounting in
`data/pred_byte_value.csv` is optimistic for the hint side (staging 8-51
extra records per call into LRU would pollute the cache; not modeled), so
the true net is no better.

### (3) Each arm's m=8 numbers vs P9 (recall 0.65-0.80, kill outside 0.55-0.88)
and P10 (precision 0.10-0.18, kill > 0.30) — next_layer surface

| arm | R@8 T2 / T1 (best sem) | P9 verdict | P@8 T2 / T1 (set sem) | P10 verdict |
|---|---|---|---|---|
| popularity | 0.150 / 0.444 | **KILLED** (-40pp / -11pp vs 0.55 bound) | 0.150 / 0.333 | T2 CONFIRMED (in band); T1 **kill threshold crossed** (+0.033..0.053 > 0.30) |
| cond_pooled (the P9/P10 derivation arm) | 0.344 (src) / 0.491 | **KILLED** (-20.6pp / -5.9pp vs 0.55) | 0.118 / 0.249 | T2 CONFIRMED (in band); T1 outside band above by +0.07 (CI +0.04..+0.09 over the 0.18 edge) but < 0.30 |
| cond_perlayer | 0.355 (src) / 0.524 | **KILLED** (-19.5pp / -2.6pp) | 0.154 / 0.306 | T2 CONFIRMED; T1 point > 0.30, CI [0.276,0.331] straddles — **borderline** |
| logreg_pairwise | 0.053 / 0.083 | **KILLED** (-49.7pp / -46.7pp) | 0.027 / 0.026 | outside band below by -0.07 vs the 0.10 floor (both traces); not killed |
| mlp_set | 0.183 / 0.420 | **KILLED** (-36.7pp / -13.0pp) | 0.151 / 0.286 | T2 CONFIRMED; T1 CI straddles 0.30 — **borderline** |
| xlayer_bag | 0.283 / 0.474 | **KILLED** (-26.7pp / -7.6pp) | 0.131 / 0.227 | T2 CONFIRMED; T1 outside band above by +0.05 (CI +0.02..+0.07 over the 0.18 edge) but < 0.30 |

**P9 is KILLED by every arm on both traces at m=8** (the kill bound is
"outside 0.55-0.88"; all recalls land 0.05-0.52). Even at m=32 (up to 138
announced records/call) T2 recall only reaches 0.638 and T1 0.751 — the
0.65-0.80 band needs ~32-64 candidates/call, i.e. 4-8x the m=8 hint budget.

**Where a measured number lands outside a P9/P10 range (all disagreements,
with magnitude):** P9 range violated downward by every arm
(-2.6pp to -49.7pp vs the 0.55 kill bound at m=8). P10 range violated
**downward** by `logreg_pairwise` everywhere (-0.073/-0.074 below the 0.10
floor, T2/T1) and by the src-sem conditional arms (-0.01 to -0.05 below the
floor: T2 `cond_pooled` 0.054, `cond_perlayer` 0.061; T1 0.078 / 0.090), and
**upward** on T1 by `popularity` (+0.153 over the 0.18 edge, crossing 0.30),
`cond_perlayer` (+0.126), `cond_pooled` (+0.069), `xlayer_bag` (+0.047).
P10's kill direction (> 0.30) is the one that matters and it fires on T1 only.

### (4) Would I ship this predictor in dee-serve as a hint source, and at what m?

**No — do not ship.** The honest byte ledger says every configuration spends
at best 7.5x and typically ~10x more slow-tier bandwidth than it saves (worst
cell 83x; best net -1,762 MB/token at m=8), which at the storage-limited
regime dee lives in converts a hint source into a wall-time regression of
several seconds per token; precision never exceeds 0.154 on the multi-run
surface at any m, so the prior-art gate (precision >= 0.75, lead >= 2) is not
even approached; and the conditional arms' probabilities are so miscalibrated
(ECE up to 0.81) that no admission threshold can be trusted to cut the waste.
The single exception worth carrying forward as a *research* note, not a
product: on a single-prompt generation with hot expert concentration (the T1
shape), a 1-2 record popularity-prior hint reaches precision 0.56-0.64
(P@1 = 0.643, P@2 = 0.562) and could pay for itself *if* it is deduplicated
against live host-residency state (our stale-hit counts show 277-366 of the
landed hints at m=8 would hit already-resident records). If dee-serve ever
ships a hint source it should be (a) m <= 2, (b) popularity/recency based,
not learned, (c) admission-controlled by observed host-residency and (d)
gated by a live A/B on wall time — all four conditions unmet today.

---

## 4. Conflicts with the previous run (disagreement is data)

1. **The P9/P10 ranges were in-sample artifacts.** `theory/prefetch.py` fit
   `P(j|i)` on the whole trace and scored on the same adjacent pairs. Under
   that protocol this stage measures **recall 0.765 at m=8** on T2 — exactly
   inside P9's 0.65-0.80 band. Out-of-sample the same estimator scores
   **0.344 [0.272,0.434]**: the in-sample protocol overstates recall by
   **+42pp (T2)** and **+23pp (T1: 0.717 -> 0.491)**. The leave-one-cluster-out
   jackknife shows the in-sample numbers are stable (T2 0.757-0.775) — they
   are not noise, they are overfit. Evidence:
   `data/pred_leak_diagnostic.csv` (the in-sample rows are diagnostic only).
   Corollary for the campaign: THEORY.md §7's "measured predictor quality"
   table and the `Sigma_corr = 0.729` ceiling quoted from it are in-sample
   numbers and overstate the realizable hint ceiling by ~2x.
2. **Ornith record size**: brief says 13,369,344 B; the trace says 6,291,456 B
   on all 8,011 `expert_request` rows (reported both ways; ratios invariant).
3. **Geometry transfer is not free**: the same `popularity` arm scores
   0.333 precision@8 on T1 vs 0.150 on T2 (+0.18) — a single-prompt DSv4
   window is far more predictable than multi-prompt Ornith. Any P9/P10-style
   range must be per-surface.
4. **Learned models did not beat the baselines on the primary trace.**
   `logreg_pairwise` collapsed to precision 0.027 (its hashed pair features
   cannot generalize across the stream shift); `mlp_set` ties the
  per-layer conditional (0.151 vs 0.154). The "Edge0 shows a trained
   per-layer predictor could be much stronger" hypothesis is NOT supported at
   these training-set sizes (702 / 462 examples).

## 5. Threats to validity

- **Sample size.** T2 test = 429 calls / 11 tokens / 1 decode step; T1 test =
  210 calls / 5 tokens. Cluster CIs at n=5-11 are wide and
  tail-sensitive — every T1 number here should be read as provisional
  (jackknife bounds in `data/pred_leak_diagnostic.csv` for the conditional
  arm).
- **Stream multiplicity.** T2's 8 runs are 4 routing streams; the effective
  independent sample is 3 train + 1 test prompts. A run-level split would
  have been leaky (§1).
- **Model/geometry transfer.** Primary results are on Ornith (40 layers,
  top-8, 6 MiB records), not DSv4-Flash; T1 is DSv4 but 16 tokens of one
  prompt. P9/P10 were derived on T1 in-sample; both surfaces disagree on
  precision at m=8 by up to 0.18.
- **No live wall A/B.** The byte ledger bounds wall impact through the
  measured storage ceiling (P12-consistent); cache pollution from staging
  hints is unmodeled (favors the hint side).
- **Multiplicity.** 6 arms x 2 tasks x 2 semantics x 2 traces; the T1
  threshold crossing is exactly the kind of cell multiple comparisons
  produce. It survives at CI level but not at jackknife level.

## 6. What measurement would change the verdict

1. A >= 1k-token multi-request DSv4-Flash trace (the FALSIFICATION
   "data needed" row) with a train/test cut across requests — would move P9
   from KILLED to testable with useful power and settle the T1/T2 precision
   disagreement.
2. A lead >= 2 surface (predict `S_{t,l+2}` or step t+2) measured the same
   way — the prior-art gate needs lead >= 2 AND precision >= 0.75; lead-1
   data says the second condition is unreachable, but lead-2 was never
   measured here.
3. A live hint A/B on the sealed bank with an admission-controlled m <= 2
   popularity hint, measuring wall time and storage traffic (kills or
   confirms the byte ledger's -1.7 GB/tok net).
4. Per-record host-residency telemetry at hint time — converts stale hints
   (10-20% of landed hints here) into avoided announcements and would raise
   the saved/consumed ratio toward its ceiling of ~1/(1 - P(hit)).

---

## 7. Artifacts

| file | content |
|---|---|
| `data/pred_metrics.csv` | 288 rows: arm x trace x task x semantics x m (1..32) x {recall, precision, announced/call} + cluster-bootstrap CIs |
| `data/pred_byte_value.csv` | 432 rows: saved / stale / wasted records and bytes per token per config, 3 baseline residency scenarios |
| `data/pred_calibration.csv` | reliability bins + ECE per arm (m=8) |
| `data/pred_splits.json` | exact split definition, leak evidence, counts |
| `data/pred_leak_diagnostic.csv` | in-sample (old protocol) vs out-of-sample + leave-one-cluster-out jackknife |
| `figs/pred_precision_recall.png`, `figs/pred_recall.png` | P/R vs m with P9/P10 bands and kill thresholds |
| `figs/pred_calibration.png` | reliability curves |
| `figs/pred_byte_value.png` | saved vs consumed bytes per token |
| `theory/pred/` | the package (`__init__.run(fast)`, `data`, `arms`, `metrics`, `lab`) |

Reproduce: `cd dee.cpp/theory && python -m theory.run_all pred`
(deterministic; `--fast` drops the bootstrap CIs only).
