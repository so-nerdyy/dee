# REPORT_PHASE7 — the dee digital twin, predictor verdict, provisioning solver, and pre-registered Phase-6 scoreboard

**Status:** integrated Phase-7 report (all four components landed;
this file is written by the orchestrator from the component artifacts and
reports).
**Branch:** `research/phase7-theory`. **Campaign brief:**
[`AGENT_BRIEF_PHASE7.md`](AGENT_BRIEF_PHASE7.md).
**Reproduction:** `cd dee.cpp/theory && python -m theory.run_all` — now
regenerates stages `provenance, anchor, popularity, cache, temporal,
roofline, serving, prefetch, sensitivity, sim, pred, solve, predecl`
(fixed seeds; deterministic; verified bit-identical for the classical
stages at commit `70dab10`).

Phase 7 builds **instruments**, not more bounds: a digital twin that must
earn the right to predict by first reproducing the sealed anchor run; a
predictor lab that *trains and evaluates* real prefetch predictors instead
of bounding them; a capacity planner that turns the thesis into "for X
traffic, provision Y"; and a timestamped prediction matrix that turns the
Phase-6 hardware campaign into an experiment. This report integrates the
four, and — honestly — records where the instruments disagree with the
theory and which side to trust.

---

## 0. What Phase 7 corrected in the inherited theory layer (orchestrator
independent review)

Before spawning the four components, an independent review compared the
Phase-6-era theory claims against their own cited artifacts. Three
correction sets landed (each keeps the old claim visible in place — a
correction, never a silent edit):

| # | Claim as inherited | Artifact says | Impact | commit |
|---|---|---|---|---|
| 1 | THEORY.md 5.2: steady Che predicts "46 cold records/tok (-45%)" | `data/anchor_check.json`: **67.84**, −18.7% | the steady model *under*-predicts, not catastrophically; P4 bracket lower end was wrong | `6747aaa` |
| 2 | THEORY.md 5.2: host tier "absorbs 23-25% of fills: hits 2,618/5,099" | 2,618/5,099 = **51.3%** pooled; 23-25% matches only cuda0-hits/pooled-requests (24.0%) | host absorption is 2x the claimed share; all 8 share definitions now computed in [`data/anchor_gate.json`](data/anchor_gate.json) | `6747aaa` |
| 3 | dense baseline "8xH100 at $3.223/GPU/h… conservative toward the competitor" | computed `price_gpu_s=0.003223 $/GPU/s` = **$11.60/GPU/h** (note said $3.223/h — 3.223/1000 vs 3.223/3600) | **3.57x ambiguity that flips the break-even**: cheap reading makes dense $0.0358/1k tok and *no* cell beats it at b=8; the "conservative toward competitor" posture is retracted on price | `72ebd5e` |

Also corrected: the `$/1k tok` formula printed in THEORY.md 6 /
serving.py docstring (dimensionally 1e6 off; code was always right), and a
P7→P15 cross-reference. Full detail in the commit messages and in the
marked corrections inside [THEORY.md](THEORY.md) §5.2/§6 and
[FALSIFICATION.md](FALSIFICATION.md) P4/P15/P15b.

**Consequence carried through Phase 7:** every economics number below is
conditional on the dense-price reading; P15/P15b are now explicitly
conditional predictions.

## 1. The anchor gate (shared target, pre-registered)

`theory/anchor_gate.py` extracts the sealed run's ground truth
(`experiments/route_pipeline/fill-live-t4x2-20260909/result.json` +
`routed_experts.jsonl`) into [`data/anchor_gate.json`](data/anchor_gate.json)
and **declares the gate tolerances in the scaffold commit — before any
simulator existed** (pre-registration against gate-tuning):

| dimension | measured anchor | registered tolerance |
|---|---|---|
| decode `storage_requests` | mean 83.5, range 56–125 (prefill 1,229) | mean ±10%, ≥12/15 tokens ±15% |
| host_pack hit share (pooled_dedup) | 51.3% (2,618/5,099) | ±3 pp |
| pooled counters | cold_loads 4,444 / resident_hits 655 / h2d_copies 4,444 | ±5% / ±20% / ±5% |
| decode wall shape | mean 4,765 ms, 15-token profile | mean ±30%, ≥12/15 tokens ±50%, Spearman ≥ 0.5 |
| calibration policy | — | any fitted parameter must be CALIBRATED with residual; fit/holdout split stated |

## 2. The digital twin (`theory/sim/` + [`sim/SIM_VS_THEORY.md`](sim/SIM_VS_THEORY.md))

PENDING — component A report. (Orchestrator verification so far: the
sim's transcribed pipeline mechanics match the source exactly —
`engine.cpp:565-567` staging order, `:592` priority `K - i`,
`vram_cache.h:244-247` `last_used + priority*(1<<20)` — and its mechanism
claims reproduce from [`data/anchor_tokens.csv`](data/anchor_tokens.csv):
`storage_requests ≡ host_pack_misses` on all 16 forward steps,
decode `r(wall, storage_requests) = 0.987` vs
`r(wall, cold_loads) = -0.07`. The anchor decode wall is driven by the
*host-pack fill pipeline* (682-slot LRU churn), not by cold SSD loads
alone — a mechanism insight that predicts where prefetch could matter and
where it cannot.)

## 3. The predictor lab (`theory/pred/` + [`PREDICTOR_VERDICT.md`](PREDICTOR_VERDICT.md))

**VERDICT (component B, complete): do not ship a lead-1 hint source.**
Six real predictors (empirical-conditional, per-layer conditional, hashed
logistic, MLP, within-token cross-layer, popularity prior) on the real
50k-event router trace + the sealed journal, time-ordered splits, 288
metric rows with cluster-bootstrap CIs:

- **P9 KILLED**: every arm lands recall 0.05–0.52 at m=8 vs P9's 0.65–0.80
  (kill band 0.55–0.88) on both traces. The prior run's 0.729 was
  in-sample resubstitution — measured inflation **+42pp (T2) / +23pp (T1)**
  in [`data/pred_leak_diagnostic.csv`](data/pred_leak_diagnostic.csv);
  jackknife shows the in-sample numbers are overfit, not noise.
- **P10 partially killed**: its kill threshold (> 0.30 at m=8) is crossed
  only on the tiny single-prompt sealed surface (popularity prior 0.333
  [0.302, 0.357], 5 clusters — borderline under jackknife), NOT on the
  50k-event surface (max 0.154 [0.076, 0.266]). "Fails the prior-art gate"
  survives everywhere.
- **Byte ledger decides it anyway**: net bytes/token is negative in every
  cell (consumed 5–16x saved; best ratio anywhere 0.182 at m=4) — hints at
  these precisions are a wall regression of ~7–10x their benefit at the
  sealed bank's storage ceiling.

**Orchestrator audit + calibration** (leak audit: features use only
hint-time information; run-grouped splits keep byte-identical streams on
one side; test-window labels dropped from training; a-priori
hyperparameters; metrics arithmetic verified on 288 rows). Calibration is
bad in both directions (corrected reading — an earlier orchestrator note
said "under-confident" from one arm's bins alone): conditional-combination
arms are badly OVER-confident (predicted 0.79–1.00 vs observed 0.04–0.34 in
the top bins — the `1-∏(1-P(j|i))` independence product saturates;
ECE ~0.4–0.84), while the popularity arm is UNDER-confident (predicted
0.018–0.033 vs observed 0.08–0.25; ECE ~0.10–0.32). The MLP arm is the
best-calibrated learned arm (ECE 0.07–0.25) but does not beat the
conditional baselines on precision. Practical consequence (B §2.4): the
empirical-conditional "probabilities" are ranking keys, not probabilities —
never threshold them for hint admission.

## 4. The provisioning solver (`theory/solve/` + [`solve/SOLVER.md`](solve/SOLVER.md))

Component C enumerates 19,040 allocation rows (7 cells × 17 host budgets ×
4 VRAM fractions × 10 batch policies × 4 model stores) and solves the
constrained minimum-cost deployment per (model, arrival rate, SLO).

**Pareto knees** (`data/solve_knees.json`, hints off):

| model | knee cell | host GiB | b | tok/s | $/1k tok | ratio vs dense (computed) | limiter |
|---|---|---:|---:|---:|---:|---:|---|
| dsv4_flash | 1xL40S | 4 | 32 | 34.2 | 0.0169 | 0.131 | compute |
| mimo_v2_flash | 1xRTXPRO6000 | 128 | 2 | 204 | 0.0057 | 0.094 | compute |
| mimo_v2_pro | 1xRTXPRO6000 | 64 | 1 | 83.6 | 0.0121 | 0.054 | storage |
| minimax_m3 | 1xRTXPRO6000 | 768 | 1 | 98.4 | 0.0262 | 0.067 | h2d |

Knees sit at deep-coverage operating points; under the *computed* H100
price reading dee beats dense residency by 7–19× at the knee — under the
*cheap* reading the margin compresses 3.57× but does not disappear on the
knee cells.

**The decisive negative result:** all 120 (model × λ × SLO) provisioning
queries are **infeasible** (`data/solve_provisioning_infeasible.csv`). The
per-stream decode SLO (≥5 tok/s/stream) is unreachable — feasibility needs
`X/b ≥ slo/(1−ρ)`, ~16.7 tok/s per stream at SLO=5, while b=1 per-replica
rates top out at ~3–8 tok/s on bounded cells. Replication scales aggregate
throughput; it cannot speed a single stream. **dee-serve is a
cost/throughput play at batch concurrency; it cannot sell per-stream
latency.** Consistent with FALSIFICATION P15c (no cell beats dense at
b=1).

The solver also re-derived the prefetch verdict independently: at the knee
cells the measured m=1 hint arm multiplies cost by ×1.0 (dsv4, zero benefit
region), ×1.9 (mimo-flash), ×5.1 (minimax-m3) and ×16.3 (mimo-pro) —
hints-off dominates everywhere, agreeing with component B.

## 5. The pre-registered Phase-6 matrix ([`PREDICTIONS_PHASE6.md`](PREDICTIONS_PHASE6.md))

Frozen at commit `6107aab` (registration of record; 0 hardware runs existed
at freeze). **60 scored cells** = 5 Modal cells × 2 models × b ∈ {1,8} ×
host {16,64,256} GiB, each predicting decode tok/s, host hit %, cold
records/tok, and $/1k tok — every row carrying nominal + working range +
kill band + governing FALSIFICATION IDs. Plus 24 non-scored reference rows
(2xT4 anchor, CPU floor). Machine-readable twin: `data/predecl_matrix.csv`
(336 rows) regenerable via `python -m theory.run_all predecl`.

Scoring rules are pre-declared: PASS (all metrics in range) / KILLED
(any metric outside its kill band or its referenced FALSIFICATION id's) /
WEAKENED (the deliberate gap band) / UNTESTABLE (missing prerequisite — a
run without stage profile or measured B_SSD cannot score TPS rows).

**All rows are tagged `CLOSED-FORM-ONLY (sim pending at freeze)`** — the
freeze happened before component A's sim outputs existed. The sim-vs-
closed-form disagreement (§6) means the closed-form TPS rows should be
read as the storage/H2D-bound side of the prediction band; an ADDENDUM
may annotate sim-predicted values but must not edit the frozen ranges.

## 6. Where theory, sim and predictor disagree — and what to trust

| disagreement | magnitude | trust | resolution |
|---|---|---|---|
| sim anchor replay vs measured run | 12/12 dims PASS, counters bitwise | **sim** — structural replay, not fitted | settled |
| cache hit-rate: sim vs Che two-level | 5–10 pp host-hit, 5–15% cold | sim for policy mechanics; Che for trends | enough for both uses |
| token timing: sim vs closed-form T_pred | up to ~10× where storage limits (12.5% grid pass vs 80% criterion) | **sim** — blocking demand-gated fill matches anchor's wall-vs-storage r=0.987 | Phase-6 stage profile |
| dense-path constant | 180 vs 570.7 ms/tok (3.2×) | unresolved — two decompositions of one measurement | Phase-6 stage profile |
| solver vs theory $/1k tok | −14% to −60% | solver — it adds volume amortisation + SOLVE_CORES | accounting convention, documented |
| predictor calibration | ECE 0.07–0.84 by arm | none — conditional arms' "probabilities" are ranking keys | never threshold for admission |
| prefetch verdict | pred lab −5..10× net bytes; solver ×1.9–16.3 cost | **both agree: hints off** | two independent instruments concurring |
| knee stability on long streams | sealed 16 GiB → long-stream knee drifts to 4 GiB | synth (negative result) — P2's band is window-conditional | ≥1k-token real trace (data needed) |

## 7. What the Phase-6 measurements will tell us (mapping to prediction IDs)

1. **P1** (1xL4 b=1 decode 1.1–1.7 tok/s, kill <0.9|>2.1): the headline —
   tests the whole roofline chain end-to-end.
2. **P2/P2b/P2c** (LRU 48–54% at 16 GiB, within 3pp of MIN, 850–1000
   repeats): tests the cache model on a fresh window — and now carries the
   synth caveat that the knee is window-conditional.
3. **P3/P3b/P4** (cold records/token brackets): tests the finite-window
   correction directly.
4. **P5** (η_storage 0.15–0.45): tests fill-hiding — the sim's blocking
   makespan vs theory's (1−η) disagreement lives here; a stage profile
   adjudicates.
5. **P8b/P8c** (t₀ 300–420µs; batched 50–150µs): a stage profile on any
   T4-class run plus a batched-kernel build would replace the weakest
   calibrated constant with a measurement.
6. **P15c** (no cell beats dense at b=1) + provisioning infeasibility:
   the serving-side reality check — dee-serve's value case is batch
   throughput economics, not latency.
7. **P9/P10 post-kill state**: prefetch is off the table at current
   trace-derived accuracies — the only resurrection path is a predictor
   beating precision 0.30 at m=8 on a fresh window, which the lab measured
   at ≤0.154 on the big trace.

The campaign's real deliverable is therefore: one L4-class run (b=1 and
b=8 if budget allows, ≥16 tokens, stage profile + per-token accounting
captured, B_SSD measured) scores P1–P5 in a single shot; a MiMo-Flash
run adds the cross-model transfer check (S_TRANSFER, the solver's
assumption-4 risk).
