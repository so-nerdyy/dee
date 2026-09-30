# REPORT_PHASE7 — the dee digital twin, predictor verdict, provisioning solver, and pre-registered Phase-6 scoreboard

**Status:** integrated Phase-7 report (work in progress — sections marked
PENDING are filled as the four component reports land; this file is written
by the orchestrator from the component artifacts and reports).
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

PENDING — component B report. (Orchestrator audit of the evaluation
pipeline: no future leakage — time-ordered/run-grouped splits with
byte-identical streams kept on one side, test-window labels dropped from
training, a-priori hyperparameters; metrics arithmetic verified on 288
rows.)

## 4. The provisioning solver (`theory/solve/` + [`solve/SOLVER.md`](solve/SOLVER.md))

PENDING — component C report.

## 5. The pre-registered Phase-6 matrix ([`PREDICTIONS_PHASE6.md`](PREDICTIONS_PHASE6.md))

PENDING — component D report.

## 6. Where theory, sim and predictor disagree — and what to trust

PENDING — integrated disagreement table.

## 7. What the Phase-6 measurements will tell us (mapping to prediction IDs)

PENDING — final mapping table.
