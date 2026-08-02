# DS9 v13 — Product-policy decision: ACCEPT `REJECT_NUMERICAL`

**Date:** 2026-08-02
**Status:** Terminal verdict accepted by product policy.
**Verdict:** `REJECT_NUMERICAL`

## Decision

The DS9 v13 expert-integration audit (`672c2f14…`, archive `9a2b8b0d…`) proves
the sole remaining `moe_out`/`shared_out` p99 failure (0.068–0.099 vs the
sealed 0.05 gate) is a **numerical floor of the comparison methodology**, not a
semantic, integration, storage, routing, ordering, or harness defect:

- FP16 execution on the *reference* input: p99 0.021–0.023 — **inside** the gate.
- Pure-FP32 reference on the *candidate* input: p99 0.0707–0.1094 — **over** the gate.
- The trigger is bounded BF16-storage-rounded input-boundary drift (max 2.0
  bf16 grid steps) amplified by the FP32 weighted 6-expert + shared
  combination at near-cancellation output elements (cancellation ratio ~0.002).
- All integration candidates are excluded: harness replay bitwise faithful,
  FP4/FP8 storage exactly lossless, accumulation order irrelevant (~1e-7),
  routing weights ≤ 0.0016 delta, shared-expert decomposition consistent.

## Accepted implications

- `REJECT_NUMERICAL` is the **terminal DS9 verdict**. No runtime correction
  exists under the Phase-11 priority list (all entries excluded by evidence).
- The sealed 0.05 p99 gate is **unchanged** and non-negotiable.
- No expert-ID, router, state, or gate policy was changed by this decision.
- The v13 corpus is synthetic; the p99 *magnitudes* are corpus-provisional.
  The causal mechanism is proven independent of the corpus.
- Re-measuring on **official hidden-state traces (DS5 dependency)** is the
  designated next check for the p99 magnitudes, not a gate change.

## Campaign status after this decision

- DS7 (one expert on T4) — sealed, `MATCH_WITHIN_TOLERANCE`.
- DS8 (generalized expert runtime + cache) — sealed, `ACCEPT_EXPERT_RUNTIME`.
- DS9 (one complete layer) — **terminated** with accepted `REJECT_NUMERICAL`
  after proving the only failing component is bounded input-boundary
  numerical sensitivity.
- DS4 (tokenizer/encoding parity) — in progress (next milestone).
- DS5 (official reference traces) — required to re-measure the p99 tail on
  real hidden-state distributions.

Signed: product-policy acceptance 2026-08-02 (user decision).
