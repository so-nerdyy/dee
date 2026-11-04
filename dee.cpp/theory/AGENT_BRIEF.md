# AGENT BRIEF — DEE-Theory: an analytical performance model for exact sparse-MoE inference

You are working in the repository `dynamic_expert_eviction` (dee.cpp), a C++/Python
exact-inference runtime for very large sparse Mixture-of-Experts models whose full
expert universe cannot fit in accelerator or host memory. The system serves experts
through the hierarchy `NVMe -> host RAM -> VRAM -> accelerator`, preserving EXACT
model behavior: authoritative routing, expert identity, checkpoint representation,
expert ordering, outputs. Prediction is legal only as a prefetch hint — never as
routing.

Your task is purely mathematical/analytical: build the theory layer that turns the
prototype into a research-grade result. You must NOT run GPU jobs, modify engine
code, or change runtime semantics. All work is derivation + empirical analysis of
in-repo artifacts + reproducible Python computation.

## Ground truth you must use (do not fabricate constants)

Read these before writing anything:

- `AGENTS.md` (repo root) — project state, exactness contract, phase history.
- `dee.cpp/tools/phase3/specs/*.json` — record geometry for 4 models:
  DeepSeek-V4-Flash (11,776 records, 13,369,344 B), MiMo-V2.6-Flash (12,032,
  13,369,344 B), MiMo-V2.6-Pro (26,496, 20,054,016 B), MiniMax-M3 (7,296,
  ~118.6 MB bf16 records).
- `dee.cpp/benchmark_reports/milestone-2.5/kaggle-forensics-latest-output/
  ornith-milestone25-evidence/expert-trace.jsonl.gz` — real per-token router
  decisions (layer, expert, routing_rank, routing_weight, phase prefill/decode).
- `dee.cpp/experiments/route_pipeline/fill-live-t4x2-20260909/` — routed_experts.
  jsonl, per-layer host-profile rows, dee4 integrity journal, run_config.
- `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/p2_3_trace_replay_simulator.py`
  and the phase2 working-set results referenced in AGENTS.md — the sealed-trace
  sim whose measured numbers your cache model must reproduce: plain-LRU causal RAM
  knee ~16 GiB pooled, LRU ~2.8pp below offline MIN at that budget, reaches MIN by
  ~32 GiB, 935/2364 records ever repeat, candidate VRAM policy `pure last_used`
  ~14.48 GB/response H2D reduction at ~281-slot cap.
- `dee.cpp/kaggle/deepseek-v4-flash-0731/` — run configs + measured decode stats
  (~0.21 tok/s on 2xT4 SM75, storage ~0.29–0.37 GiB/s, service time ~= pread).
- Modal price schedule (document as ASSUMPTION where not in-repo): L4 $0.000222/s,
  A10 $0.000306/s, L40S $0.000542/s, RTX PRO 6000 $0.000842/s, CPU
  $0.0000131/core/s, RAM $0.00000222/GiB/s, volume $0.09/GiB/mo, first 1 TiB free.

## Deliverable

Create `dee.cpp/theory/` containing:

1. `theory/` — a Python package, stdlib + numpy/scipy/matplotlib only, where
   `python -m theory.run_all` regenerates every figure, table and CSV
   deterministically. No network access at analysis time.
2. `THEORY.md` — the derivation document (technical-report style: definitions,
   model, derivations, empirical fits with goodness-of-fit, predictions,
   limitations).
3. `FALSIFICATION.md` — a table of numeric predictions with ranges that future
   hardware runs can confirm or kill, each tagged with which measurement kills it.
4. `figs/` + `data/` — generated curve CSVs and plots.

## Required components — each must be derived, not asserted

### A. Expert popularity law

From the real traces, estimate the per-layer and pooled expert access
distributions. Fit at least: Zipf–Mandelbrot `p_i ∝ (i+q)^{-s}`, log-normal on
frequency, and a uniform-baseline. Report log-likelihoods and a KS-style
statistic; state honestly which family fits and where it fails (tail behavior
matters most). Key outputs: estimated (s, q) per layer and pooled; Shannon
entropy vs log(N_experts); the coverage curve `C(m) = Σ_{i≤m} p_(i)` — what
fraction of traffic the top-m experts capture.

### B. Cache model — closed form vs simulated

Under an Independent Reference Model, derive aggregate hit-rate `H(M)` vs
resident capacity M (records and bytes) using Che's TTL approximation or
equivalent. Overlay the empirical LRU and offline-MIN curves where in-repo sim
artifacts permit, and quantify the IRM error introduced by temporal correlation
(the sim's real numbers must be reproduced within a stated tolerance — if they
can't, document exactly why). Deliver `H(M)` curves for all four store specs.

### C. Temporal structure — the prefetch-relevant statistics

From the traces, estimate: empirical survival `P(expert reaccessed | Δt)`,
within-token cross-layer conditional `P(j ∈ S_{l+1} | i ∈ S_l)` (sufficiently
sampled pairs only — report counts), and decode-step self-correlation
`P(expert at token t+1 | at token t)` per layer. Output: correlation matrices
or aggregated bounds, and the effective IRM-vs-correlated hit-rate delta.

### D. Throughput bound — the roofline of the hierarchy

Derive decode token-latency bound as the pipeline max:

    T_tok(b, M_host, M_vram) = max( cold_bytes/B_SSD,
                                    spill_bytes/B_H2D,
                                    expert_compute(b)/F_gpu,
                                    dense_path(b) )

where `cold_bytes = P_rec · Σ_l E[misses_l(b, M)]` with `E[misses]` from (B)
corrected for batch overlap (the same expert missed once serves the batch — do
this correctly, not naively k·(1-H)). Produce TPS-bound curves vs host budget
for each (model, hardware) cell: 1×L4, 2×L4, 1×A10, 1×L40S, RTX PRO 6000, and a
CPU-bound/no-GPU reference cell. Mark the measured anchor (~0.21 tok/s at
0.29–0.37 GiB/s on the T4 bank) and check the model predicts it within ~2× —
report the ratio, don't hide it.

### E. Serving extension — cross-request economics

Model a request stream (Poisson arrivals, batch-of-b decode). Steady-state
cache popularity becomes a mixture over concurrent requests: derive H_serve(M, λ)
as a function of concurrency, and `bytes/token vs concurrent requests`. Then the
cost frontier: `$/1k tok = hardware_rate / (TPS_bound · 1000)` per cell, and the
break-even request volume against an honestly-parameterized dense-residency
baseline (e.g., serving DSv4-Flash bf16 ~568 GiB resident on H100-class hardware
at published prices — state the baseline's assumptions explicitly and
conservatively; a weak baseline makes the result worthless).

### F. Prefetch bound — the only legal use of prediction

Formulate speculative prefetch as budget-constrained selection: given
conditional probabilities from (C) and prefetch bandwidth `B_pf`, derive the
optimal prefetch set rule (top-m by marginal miss-rate reduction per byte —
identify when this is a submodular/max-coverage instance and when it isn't),
expected miss-rate reduction vs B_pf, and the provable ceiling `ΔH ≤ Σ_corr`.
State plainly what prefetch can never do under the exactness contract.

### G. Sensitivity + regime map

Global sensitivity (partial derivatives or Sobol-style sweep) of TPS_bound to:
record size, top-k, SSD bandwidth, locality exponent s, host/VRAM capacity.
Output the regime map: regions of (B_SSD, M_host) where dee is
storage-bound / H2D-bound / compute-bound, per model. This is the figure that
makes the work citable — the prototype's phase conclusions must appear on it
where derivable (Phase-1's "storage-bandwidth constrained, ~0.3 GiB/s" verdict
must land inside the storage-bound region or be flagged as contradicting).

## Rigor requirements (hard rules)

- Every constant cites its source file, or is explicitly tagged ASSUMPTION.
- Named methods only (Che TTL approximation, IRM, Little's law, renewal
  reward, submodular greedy) — no invented machinery where standard results
  suffice, and no standard results dressed as novel contributions.
- Goodness-of-fit reported including failures. A bad Zipf fit reported
  honestly is worth more than a claimed fit that isn't there.
- Uncertainty: empirical estimates get bootstrap CIs where sample sizes allow.
- FALSIFICATION.md predictions must be numeric, range-bounded, and each must
  name the measurement that would kill it. Vague predictions are worthless.
- Zero network calls in analysis; zero modifications outside `dee.cpp/theory/`.
- If a needed artifact is missing, document the gap in FALSIFICATION.md as
  "data needed" — never substitute a made-up number.

## Context for framing (do not repeat verbatim)

The intended product split: `dee-local` open source (bounded-hardware exact
inference) and `dee-serve` (shared expert caches, continuous batching). The
thesis your math must serve: hardware requirements scale with active
parameters, working set, throughput and concurrency — not total checkpoint
size. THE THEORY.md should read like the technical core of a workshop paper,
with the regime map and cost frontier as its primary figures. Phase-6 hardware
measurements (Modal, L4-class GPUs) will test your predictions in days —
calibrate so they're falsifiable, not so they're safe.
