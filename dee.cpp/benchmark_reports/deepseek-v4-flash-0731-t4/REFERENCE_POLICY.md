# REFERENCE_POLICY — sealed DS8/DS9 discipline as the campaign reference

**Date:** 2026-08-04
**Status:** Product-policy decision, accepted.
**Decision:** The trusted reference for DeepSeek-V4-Flash-0731 parity is the
**sealed DS8/DS9 discipline**: official checkpoint tensors + FP32 CPU
reference implementations of the official semantics + the validated DS8/DS9
CUDA kernels. The official tilelang inference stack is **not required** for
campaign parity.

## Rationale

- DS5 v6 proved a hardware-format blocker: the pinned official tilelang stack
  cannot launch on the campaign's only GPU (Kaggle dual T4, SM75).
  `fp8_gemm` requests 82,048 B dynamic shared memory vs the 65,536 B ceiling;
  `sparse_attn` (h=64, d=512) needs ~280 KiB; the indexer variant ~74 KiB.
  No pipelining tweak can fit the official attention path on SM75. Evidence:
  `ds5-v6-hardware-blocker/` (archive SHA256 `923b65fb…`-independent;
  full kernel.log + verdict + module copies preserved).
- The DS5 contract forbade substituting a re-tiled kernel as the trusted
  reference *without a product decision*. This document is that decision:
  the substitution is the **sealed DS8/DS9 discipline**, which is already
  validated end-to-end on T4:
  - DS7 `MATCH_WITHIN_TOLERANCE` — one official routed expert on T4,
    FP4-packed storage + FP16 SM75 GEMV vs full-FP32 reference.
  - DS8 `ACCEPT_EXPERT_RUNTIME` — 3 complete official layers (3/20/41),
    top-6 routed + shared expert through the bounded expert cache,
    cold==warm bitwise, eviction-pressure bitwise-reproducible.
  - DS9 `REJECT_NUMERICAL` (accepted) — complete official layer 20 on T4:
    state proven correct (v9), router proven exact (v10/v11), set-based
    expert-ID gate adopted (v12), expert-integration audit (v13) proves the
    sole MoE/shared p99 tail is bounded BF16-storage-rounded input drift
    amplified at near-cancellation elements — not a semantic, integration,
    storage, routing, ordering, or harness defect.

## What this changes

- **DS5** is no longer a campaign blocker. Its trusted-reference role is
  fulfilled by the sealed DS8/DS9 discipline. The official-stack trace
  remains documented evidence of the hardware limit and stays optional —
  it may be re-attempted only on non-T4 (SM80+) hardware or after an
  explicit fork/re-tile product decision.
- **DS9 p99 magnitudes** (corpus-provisional on the synthetic corpus) will be
  re-measured on **real hidden states produced by the campaign's own full
  forward path** (DS10), per the DS9 v13 policy decision. The causal
  mechanism is already proven corpus-independent.
- **DS10** (dual-T4 full-model decode) becomes the next milestone, using the
  sealed kernels as both the runtime and the reference for hidden-state
  generation.

## What this does NOT change

- The sealed 0.05 p99 contract is unchanged and non-negotiable.
- No expert-ID, router, state, or gate policy changed by this decision.
- No TPS is claimed. `performance_comparable` remains `false` for all
  correctness milestones.
- Ornith seal `9ff967e` and M5G-v1/v2/v3 evidence remain immutable.
- DS7/DS8/DS9 evidence remain byte-for-byte immutable.

## Campaign status after this decision

| ID | Status |
|---|---|
| DS0–DS4 | COMPLETE (DS4 encoding parity 17/17 golden tests) |
| DS5 | BLOCKED hardware-format → **reference role superseded by this policy** |
| DS6–DS8 | COMPLETE |
| DS9 | TERMINAL `REJECT_NUMERICAL` (accepted) |
| DS10 | NEXT — dual-T4 full-model decode |
| DS11 | One-T4 path (after DS10) |
| DS12 | DSpark (after DS10/DS11) |
