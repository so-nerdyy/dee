# PREDICTOR_RESULTS.md — Fate-style cross-layer prediction for DSV4 (offline)

Branch: `research/cache-predictor` · Date: 2026-09-03/04
Status: **offline trace study** · The OFFICIAL router is authoritative ·
Predictions used only for scheduling/prefetch simulation. No runtime
benchmark was run; no TPS claim is made.

---

## 1. Causality protocol (strict)

- Every prediction for the bundle at (step t, layer L) uses **only**
  bundles from steps **< t**: the previous token's bundle at the same
  layer (executed one full forward step earlier) and per-layer
  popularity aggregated over steps < t.
- No future routing information is used for prediction. Future routing
  is used **only for evaluation** (recall denominators).
- Unit tests enforce this (`test_predictor_causality_*`): predictions
  for step-0 bundles are empty; every ranked key belongs to the target
  layer.
- Leaks were found and fixed during this study (documented in §6): two
  earlier drafts leaked the target bundle's own routing into its
  prediction or mis-indexed slice positions. All numbers below are
  post-fix; pre-fix numbers were too good (useful-prefetch ≈ 100%,
  recall gains that do not exist) and are **retracted**.

Prediction target = **layer bundle** (all ≤6 experts routed for the
token at that layer), because dee schedules layer-by-layer.

---

## 2. Predictors compared (feature families, Phase C)

| name | features |
|---|---|
| `freq_expert` | per-layer popularity from prior steps only |
| `persist_layer` | previous **token's** bundle at the same layer (cross-token recurrence) |
| `cross_union` | `persist_layer` ∪ current token's previous-layer bundle (same-token adjacent-layer) |
| `combo` | `cross_union` ∪ per-layer popularity |

Note: same-token adjacent-layer information was **measured to be
near-zero** on DSV4 (adjacent-layer overlap 0.13/6; **0 of 3,870**
decode selections share an expert with an earlier same-step layer), so
`cross_union`/`combo` are included to *test the Fate hypothesis*, not
because prior work suggested they would win here.

---

## 3. Recall / precision (measured on the sealed v65 trace, decode steps)

645 decode layer-bundles evaluated (15 steps × 43 layers); each true
bundle has 6 experts.

### recall@k (fraction of the true bundle found in the top-k candidates)

| predictor | @6 | @8 | @12 | @16 |
|---|---:|---:|---:|---:|
| freq_expert | 0.336 | 0.389 | 0.458 | 0.501 |
| **persist_layer** | **0.346** | **0.418** | **0.503** | **0.544** |
| cross_union | 0.346 | 0.349 | 0.354 | 0.475 |
| combo | 0.346 | 0.349 | 0.354 | 0.475 |

### precision@k, full-bundle recall, useful/wasted prefetch bytes

| predictor | precision@12 | full-bundle@12 | useful-prefetch %@12 | wasted-prefetch %@12 |
|---|---:|---:|---:|---:|
| freq_expert | 0.229 | 0.025 | 22.9% | 77.1% |
| **persist_layer** | **0.251** | 0.047 | 25.1% | 74.9% |
| cross_union | 0.177 | 0.012 | 17.7% | 82.3% |
| combo | 0.177 | 0.012 | 17.7% | 82.3% |

(Useful = prefetched and then actually demanded while still resident;
wasted = prefetched but not demanded before eviction. Bytes are useful/
wasted counts × 13,369,344.)

**Findings:**

1. **Cross-token recurrence is the only useful signal.** The best
   predictor (`persist_layer`) reaches recall@12 = 0.503 — half of each
   next token's expert set is predictable one full forward step in
   advance.
2. **Adding same-token adjacent-layer features actively hurts**
   (recall@12 0.503 → 0.354): they inject ~zero-recall noise ahead of
   the popularity tail. **The Fate mechanism does not transfer to
   DSV4** in its cross-layer form; DSV4's router produces
   layer-uncorrelated expert sets.
3. Full-bundle prediction (all 6 before the layer runs) is rare even
   for the best predictor (4.7% @12, 6.2% @16 — see JSON).
4. **Lead distance:** the useful signal is cross-token, i.e. it
   arrives **43 layers ahead** (one full forward step ≈ 43 layer
   crossings on this 43-layer model; measured mean lead = 43.0 layers).
   Within-step lead distance is not usable because within-step reuse is
   zero. Practically: predictions for token t+1 are fully
   materializable while token t is still decoding (ideal prefetch
   window = inter-step latency), **provided** the router is run ahead
   of the expert-fetch — which dee's current layer-by-layer schedule
   does not do (see §5).

---

## 4. Prediction ↔ prefetch conversion (Phase C/E join, post-fix)

Simulated prefetch (mode `replace`, pure-LRU demand path with
prediction-aware hysteresis victims, `min_dwell=6`, ≤4 prefetches per
bundle; predictor = `persist_layer` features via `build_ranked_fn`).
Prefetch accounting is closed and unit-tested:
issued = useful + wasted + still-resident, per layer.

| cap/layer | prefetches issued | useful | wasted | useful % | demand misses converted |
|---:|---:|---:|---:|---:|---:|
| 8  | 460 | 139 | 321 | **30.2%** | 139 |
| 16 | 1,208 | 242 | 966 | **20.0%** | 242 |
| 32 | 148 | 18 | 130 | **12.2%** | 18 |

(Per-bundle cap 4–12 changed nothing material; larger budgets were
exhausted by hysteresis, not by candidates.)

Demand-miss totals vs the **same-policy, no-prefetch** baseline
(LRU demand path — the `off` mode reproduces tournament LRU exactly):

| cap | baseline (LRU) | prefetch+replace | Δ misses | Δ SSD bytes/token |
|---:|---:|---:|---:|---:|
| 8  | 3,839 | 3,822 | **−17 (−0.4%)** | −44 MiB (−1.4%) |
| 16 | 2,955 | 3,007 | **+52 (+1.8%, harmful)** | +156 MiB (+5.3%) |
| 32 | 2,490 | 2,497 | **+7 (harmful)** | +20 MiB |

**Interpretation:**

- **In-situ same-step prefetch is neutral-to-harmful on DSV4 at every
  capacity tested.** The 12–30% useful share never pays for the 70–88%
  waste plus the displacement of demand-hot residents.
- This is **not** a contradiction of the 0.503 recall@12: conversion is
  destroyed by (a) zero within-step lead (the prediction for a bundle
  is only acted on when that bundle's layer is reached), (b) capacity
  pressure evicting prefetched experts before their demand (at cap-16,
  966 of 1,208 prefetched records — 12.9 GiB — were evicted unused),
  and (c) hysteresis (min_dwell=6), which cannot distinguish
  soon-useful prefetched experts from pollution.
- **Pollution is real and measurable**: every prefetch that survives to
  eviction unused both wasted its own H2D/SSD bytes and displaced a
  resident that LRU would have kept.
- **Compulsory misses recoverable via prediction:** 1,232 prefill
  compulsory misses are unrecoverable (nothing precedes the first
  token — information-theoretic). Of the 1,132 decode compulsory
  misses, realized conversion peaks at 242 (cap-16) ≈ **21%**, bounded
  by slot pressure; in-principle coverage at k=12 is 50%.

---

## 5. Consequences for scheduling (hypothesis, not measured)

- A **cross-token prefetch scheduler** (predict token t+1's bundles
  from token t's routing at each layer while token t is still in
  flight) is the only variant with real lead (43 layers). It requires
  running the router for the next token before the current token's
  MoE fetches complete — a runtime design change **out of scope here**
  (and out of bounds for this branch: no production scheduling was
  modified). Whether it pays depends on inter-step slack, which this
  trace cannot measure.
- In-situ same-step prefetch has **zero lead** (no within-step reuse)
  and cannot hide any read latency; its only levers are capacity
  effects, and those measured out to net-negative (§4).

## 6. Honest-error log (required reading before trusting numbers)

| date | error | effect | disposition |
|---|---|---|---|
| 2026-09-03 | Belady used a global next-use chain against per-layer slice positions | invalid regret column in first tournament run | fixed (per-layer futures + slice-local reindex); regression test added |
| 2026-09-03 | KeepValue tuner passed record-size as cache size | all tuning trials identical | fixed; re-tuned |
| 2026-09-03 | **Prefetch candidate provider scored the target bundle itself** (off-by-one in "next bundle" logic) | prefetched experts using the router's own current knowledge; useful-prefetch ≈ 100%, misses ↓ 32% — fabricated-quality predictions | root-caused via Phase A cross-check (0 within-step reuse makes near-perfect same-step prediction impossible); fixed to strictly-earlier-step scoring; all Phase E numbers regenerated; causality tests added |
| 2026-09-03 | KeepValue tuning/eval ran un-reindexed slices through the ranked provider | provider returned late-step (future) candidates for early accesses; tuning contaminated | fixed; re-tuned — winner degenerated to staleness-only ≡ LRU; "keep-value beats LRU by −11.2%" claim **retracted** |
| 2026-09-04 | Phase E baseline policy scored victims by (predicted, freq, age) instead of pure LRU | prefetch deltas were measured against a weaker LFU-ish baseline, overstating prefetch benefit | fixed: demand path is pure LRU; `off` mode now reproduces tournament LRU exactly (3839/2955/2490) |
| 2026-09-04 | Prefetch accounting holes: re-prefetched keys evicted after a prior demand not counted wasted; internal evictions by `admit_new` on the prefetch path uncounted | 199/1208 issued prefetches unaccounted at cap-16 | fixed; balance test added (issued = useful + wasted + live ≡ 0 residue) |

## 7. Reproduction

```bash
python research/cache-predictor/run_tournament.py --predictor-only
python research/cache-predictor/run_prefetch.py
python -m pytest research/cache-predictor/tests -q
```

Raw: `results/predictor_*.json`, `results/prefetch_*.json`.
Plots: `results/predictor_recall.png`, `results/prefetch_effect.png`.
