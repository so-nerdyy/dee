# DS9 v13 — expert-integration audit: causal proof

Run: `nivind/dee-cpp-deepseek-v4-flash-0731-ds9-one-layer` v13 (COMPLETE pipeline,
`fatal_error: null`, `performance_comparable: false`)
Repository pin: `672c2f14` — Ornith seal `9ff967e` untouched.
Evidence: this directory (archive `9a2b8b0d89e9762363e60bd8e71c8888a066af9bcc67cf1bf22825179c9e1a95`).

## The question

`moe_out` / `shared_out` fail the sealed 0.05 p99 gate at both tested steps
(step 0 p99 0.0739 / 0.0677; step 16 p99 0.0995 / 0.0817). The audit
(`scripts/deepseek_v4_expert_audit.py` + `expert_integration_classify`) was
built to separate: integrated input drift, routing-weight drift, FP4/FP8
storage, FP16 execution precision, accumulation order, shared-expert
semantics, combination order, and harness/reference mismatch.

## Proved-clean components (both steps, device-authentic)

| gate | step 0 | step 16 | verdict |
|---|---|---|---|
| capture fidelity (replays bitwise reproduce captured outputs) | True | True | harness valid |
| FP16 storage representability (FP4 grid x e8m0, E4M3 -> FP16 lossless) | True | True | storage exact |
| accumulation-order sensitivity (group vs sorted-eid fp32) | 1.1e-7 rel | 1.1e-7 rel | irrelevant |
| routing-weight delta (ref vs cand weights) | 0.00165 max | 5.5e-5 max | irrelevant |
| candidate FP16 kernels on the REFERENCE input (`kernel_ref_input`) | p99 0.0228 | p99 0.0211 | **within the 0.05 gate** |

The last row is the decisive one: the sealed DS8 FP16 expert kernels are
WITHIN the 0.05 gate when fed the reference input. FP16 execution precision
is NOT the cause of the tail.

## The proven cause: bounded BF16 storage-rounded input boundary, amplified

The candidate's FFN input (`ffn_norm_out`) differs from the reference's by
BF16 storage rounding only:

- step 0: 3595/65536 elements differ, `max_abs 0.015625` = **2.0 bf16 steps**
  at max magnitude 1.98, fraction-within-one-bf16-step 0.99998
  (`within_bf16_storage_bound: true`)
- step 16: 353/4096 elements differ, `max_abs 0.0078125` = **1.0 bf16 step**,
  fraction 1.0 (bounded)

That bounded input delta alone moves the PURE-FP32 reference math over the
gate:

- `ref_input_sensitivity` (CPU-FP32 reference on the candidate input vs its
  own input): **p99 0.0707 (step 0) / 0.1094 (step 16)** — no fp16 anywhere.
- `fp32exec_cand_input` (dequantized-FP32 CUDA execution on the candidate
  input): p99 0.0707 / 0.1094 — identical, confirming the input is the
  driver, not the backend.

## Why so much amplification (tail locator)

The over-gate elements are near-cancellation outputs of the weighted 6-expert
+ shared combination:

- step 0 worst: tok 5, dim 4081 — ref −0.00140 vs cand +0.00017, rel error
  1.12; **cancellation ratio 0.0021** (the final value is 0.2 % of the sum of
  the |expert contributions|).
- step 16 worst: tok 0, dim 1967 — ref −0.00189 vs cand −0.00287, rel error
  0.52; cancellation ratio 0.0023.

A ~0.4 % (1 bf16 step) input perturbation shifts a near-cancelled sum by more
than its own magnitude; the relative error of those elements dominates p99.
The step-16 leader-expert stage profile confirms the divergence enters at the
projection stages (gate/up p99 0.09/0.058 for the full comparison) while the
compute-only fp16 profile (same input) stays at p99 0.0004–0.021.

## Terminal classification

The audit's primary cause per its predeclared taxonomy is
`INTEGRATED_INPUT_DISTRIBUTION`: the layer-produced FFN input differs at the
bf16-storage level, and the FP32 weighted-expert + shared combination
amplifies it over the sealed gate — including in pure FP32.

Per the DS9 terminal-verdict set, the correct classification is
**REJECT_NUMERICAL**: semantics and integration are proven correct (every
integration/storage/routing/order/capture defect is excluded by evidence),
but the sealed 0.05 p99 contract fails. The run's pre-audit gate chain
reported `REJECT_EXPERT_INTEGRATION`; this audit re-attributes it.

## What was NOT found

No race, aliasing, buffer-lifetime, residual-source, routing, storage, or
semantic defect. No unsupported fallback. Cache cold==warm bitwise, 77 loads,
zero warm reloads, zero evictions. Candidate CUDA-resident, peak VRAM
4.65 GB. The failure is a numerical floor of the comparison methodology:
cross-backend BF16-storage-rounded input boundaries amplified by
near-cancellation in the MoE combination.

## Note on the corpus

The DS9 corpus is synthetic (official hidden-state traces are a DS5
dependency, not yet available). The p99 magnitudes are therefore
corpus-provisional; the causal mechanism (bounded input-boundary drift
amplified by the weighted combination) is proven independent of the corpus.
