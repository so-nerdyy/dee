# AGENT BRIEF — PHASE 7: the dee digital twin, predictor lab, and pre-registered Phase-6 predictions

You are working in `dynamic_expert_eviction` (dee.cpp), an exact-inference
runtime for very large sparse MoE models served through `NVMe -> host RAM ->
VRAM -> accelerator` with bitwise-exact model semantics. A prior agent built
`dee.cpp/theory/` — a closed-form performance model with 25 falsifiable
predictions. Your job is the next, bigger layer: build the *instruments* that
the theory only gestures at, and pre-register the predictions that the
Phase-6 hardware campaign will score.

You may spawn subagents freely; components A–D below are designed to run in
parallel after a shared scaffold lands. Budget ~1-2 hours wall clock. All
work is local computation — no GPU jobs, no network access at analysis time,
no engine modifications, no runtime semantics changes.

## Read first (shared context — all agents)

- `dee.cpp/theory/THEORY.md`, `FALSIFICATION.md`, `data/provenance.csv`
- `dee.cpp/theory/theory/` — the existing package (constants, sources,
  popularity, cache, temporal, roofline, serving, prefetch, sensitivity)
- `dee.cpp/experiments/route_pipeline/fill-live-t4x2-20260909/` — sealed
  anchor run (routed_experts.jsonl, per-layer host profiles, result.json,
  stage profile)
- `dee.cpp/benchmark_reports/milestone-2.5/.../expert-trace.jsonl.gz` —
  50,715-event router trace
- `dee.cpp/tools/phase3/specs/*.json` — four model geometries
- `dee.cpp/src/engine.cpp` + `include/dee/engine.h` — for the simulator's
  mechanics to match the real pipeline (host tier arming, fill coalescing,
  device slots, pread service path); read for structure, do not modify.
- `AGENTS.md` — exactness contract and phase history

## Performance review of the previous run — learn from it

What it did WELL (keep doing): honest calibration discipline (fitted on a
calibration point, predicted endpoints); reported model failures instead of
hiding them; provenance tags on every constant; conservative competitor
baseline direction; killable predictions with named killing measurements.

What it did NOT do (your opportunity):
1. It BOUNDED the prefetch predictor instead of BUILDING it. P9/P10 give
   ranges for recall/precision derived from conditional statistics — nobody
   trained or evaluated an actual predictor. That question (can a learned
   predictor beat precision 0.30 at m=8?) is the single highest-information
   open item — a real eval answers it decisively instead of with bounds.
2. The traces are short (16-token anchor, 50k events). It never tested its
   steady-state claims against long synthesized streams generated from its
   own fitted models — a free check it skipped.
3. `t_0 = 387 µs` and `COMPUTE_EFF = 0.6` stayed calibrated constants. A
   micro-simulation of the dispatch path (or a real torch microbench of the
   dequant+GEMM shapes) could replace them with measured structure.
4. Its theory curves were never checked against an INDEPENDENT
   implementation — one codebase validated itself. An event-driven
   simulator that must agree with the closed forms is a mutual
   falsification instrument, not a repeat.
5. Minor: it left `data/provenance.csv` with an uncommitted regen diff —
   generated artifacts must be regenerated-and-committed atomically.

## Required deliverables — `dee.cpp/theory/` extension

### A. `theory/sim/` — discrete-event digital twin

An event-driven simulator of the real hierarchy, NOT a reimplementation of
the closed forms: entities = request arrivals -> token decode steps -> layer
router decisions -> expert demands -> VRAM slots (recency policy) -> host
tier (LRU, bounded slots) -> SSD service queue (bandwidth + per-request
service time) -> H2D transfers -> compute stages. Configured per hardware
cell and per model spec.

Hard gates:
- REPLAY MODE: feed the sealed routed_experts.jsonl stream through it and
  reproduce the anchor run's per-token storage_requests (83-155), host_pack
  hit share (23-25%), and decode wall shape within a stated tolerance. If
  it can't reproduce the anchor, say so and stop — do not generate
  predictions from an unanchored sim.
- CROSS-CHECK MODE: on the same synthetic streams, sim vs closed-form must
  agree within stated tolerance at >= 80% of grid points; disagreements are
  reported in SIM_VS_THEORY.md with the disagreement quantified, not
  silently reconciled.
- SYNTH MODE: generate >= 100k-request streams from the fitted popularity/
  temporal models to test steady-state claims the 5k trace can't reach
  (does the +25pp Che error shrink as theory predicts? Does the knee
  stabilize?).

### B. `theory/pred/` — the predictor lab

Train and evaluate actual next-layer / next-step expert predictors on the
real traces — the only legal prediction surface (prefetch hints). Minimum
arms: (1) empirical-conditional baseline P(j|S_l); (2) per-layer learned
model (e.g., hashed-feature logistic / small MLP on history features) with
time-ordered train/test split; (3) within-token cross-layer predictor.
Metrics: recall AND precision at m in {4,8,16}, calibration, and —
critically — the byte-value-weighted benefit (a prefetch hit on a 13.4MB
record that would have missed is what counts). Verdict against P9/P10 with
real numbers: does any legal predictor exceed precision 0.30 at m=8, and
does expected saved cold-bytes/token beat the bandwidth it consumes?
Answer in PREDICTOR_VERDICT.md.

### C. `theory/solve/` — the provisioning solver

The dee-serve capacity planner: given (model spec, target tok/s or SLO,
arrival rate, price table), solve for minimum-cost allocation over
{GPU cell, host GiB, VRAM GiB, prefetch budget}. State it as a constrained
optimization (enumerate the discrete cell grid, continuous memory sizing),
produce the Pareto frontier $/1k-tok vs throughput for each model, and the
break-even surface vs the dense-residency baseline carried from the theory
layer. This is the artifact that turns the thesis into a product decision:
"for X traffic, provision Y."

### D. `PREDICTIONS_PHASE6.md` — pre-registered experiment matrix

For every actually-runnable Phase-6 Modal config (1xL4, 2xL4, 1xA10,
1xL40S, RTX PRO 6000; DSv4-Flash and MiMo-Flash; b in {1,8}; host budgets
{16, 64, 256 GiB}): predicted decode TPS, host hit rate, cold records/tok,
$/1k tok — each as a range with a kill criterion. Commit this file FIRST,
timestamped, before any hardware run exists — that timestamp is the
pre-registration. Mark which cells are sim-predicted, which are
closed-form-only, and where the two disagree.

### E. `REPORT_PHASE7.md` — the integrated report

Sim validation summary, predictor verdict, solver frontier, pre-registered
matrix, and — honestly — where theory and sim disagreed and which you'd
trust. Ends with "what the Phase-6 measurements will tell us" mapped to
prediction IDs.

## Hard rules

- Every constant cited to an artifact or tagged ASSUMPTION/CALIBRATED —
  extend provenance.csv, don't fork it.
- Determinism: `python -m theory.run_all` (extended with your stages) must
  regenerate everything; fixed seeds; outputs committed with the code that
  made them.
- Disagreement is data: any place sim/theory/predictor conflict, report
  it with magnitude. A clean-looking result produced by suppressing a
  conflict is a failed result.
- The anchor reproduction gate in (A) is a gate, not a guideline.
- Predictor eval must use time-ordered splits — no future leakage.
- Commit on `research/phase7-theory`, coherent commits per component.

## Framing

The project thesis: hardware scales with active parameters and working
set, not checkpoint size. Phase 7 makes that thesis an *instrumented*
claim — a digital twin anyone can run, a predictor verdict that settles
the only legal speculation question, a solver that says what hardware a
given traffic load actually needs, and a pre-registered scoreboard for
the hardware campaign. The Phase-6 runs then stop being measurements and
become experiments.
