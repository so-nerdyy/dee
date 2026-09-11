# DS9 v10 router diagnosis — v11 reclassification

Date: 2026-08-02
Run: `nivind/dee-cpp-deepseek-v4-flash-0731-ds9-one-layer` v10 (COMPLETE)
Evidence: `ds9-v10-router-diag/` (this directory), archive SHA256 `9596a44b…`,
manifest validated.

## Why this document exists

The v10 harness self-classified the step-0 top-6 expert-ID mismatch as

```
REJECT_UPSTREAM_LAYOUT_OR_STATE
```

via the raw heuristic `max_ulp > 64`. That heuristic is **invalid for
BF16-stored activations**: a single bf16 rounding step at magnitude 2**e
spans 2**16 fp32 ULPs (bf16 keeps 8 mantissa bits), so any bf16 storage
rounding trivially exceeds a 64-ULP threshold. The v10 evidence therefore
carried a mislabeled diagnostic verdict while the underlying data was
complete and conclusive.

The v11 classifier refinement (`scripts/deepseek_v4_contract.py`:
`bf16_storage_bound` + `router_diagnosis_classify`) replaces the raw-ULP
heuristic with an absolute-error discriminator scaled by the bf16 ulp of the
tensor's largest magnitude, adds an isolation-fidelity guard
(`captured_ref_matches_cpu_recompute` / `captured_cand_matches_cuda_recompute`)
as the structural signal, and re-orders the outcome rules.

The v10 evidence was then reclassified with the corrected classifier
(`router-reclassification.json`, machine-readable).

## Reclassified verdict (computed from the recorded v10 evidence)

```
VERDICT:    ROUTER_IMPLEMENTATION_EXACT_INPUT_DRIVEN_FLIP
FLIP_SCOPE: ORDERING_WITHIN_SET
```

### Evidence used

| fact | value |
| --- | --- |
| isolation `ref_in_ids_cpu_vs_cuda_equal` | True (all 16 rows) |
| isolation `cand_in_ids_cpu_vs_cuda_equal` | True (all 16 rows) |
| `topk_same_scores_cpu_vs_cuda_ids_equal` | True |
| captures faithful | True (both recompute flags) |
| flip token | 4 (of 16 checked) |
| top-6 SETS identical at flip token | True (`symmetric_difference: []`) |
| boundary scope | intra-set rank swap at positions 4–5 (102 ↔ 198) |
| selection (rank-6/7) margin | ref 0.0153 / cand 0.0146 (robust) |
| router-input delta | 3595/65536 elements, `max_abs 0.015625` = 2.0 bf16 steps at max magnitude 1.84375 (ulp 0.0078125), fraction within one step ≥ 0.945 (lower bound from bitwise-equal elements) |
| flip reproduced by input delta alone | True (CPU router on ref vs cand inputs) |
| first upstream divergence | `attn_o` (CUDA fp32 accumulation vs CPU fp32, ~1 ULP) surfaced as 1-bf16-ulp storage rounding; `attn_norm_in`/`attn_norm_out` bitwise exact |

### Conclusion

The router implementation and top-k semantics are **exact** for identical
input (CPU and CUDA agree on every row, for both the reference and candidate
inputs). The captured inputs themselves are faithful reproductions of the
device paths. The expert-ID gate failure is an **ordering artifact of bounded
BF16 storage rounding**: the candidate's router input differs from the
reference's by at most ~2 bf16 grid steps (absolute), two experts inside the
top-6 swap ranks at one token, and the per-token selected expert **sets are
identical at every checked token**.

No layout, lifetime, transfer, or stale-data defect exists. No semantic or
state defect remains (state masks, boundary structural keys, and attention
selection are all proven exact in v9/v10).

## Full-layer verdict status

The run-level verdict remains `REJECT_ROUTER` because the sealed DS9 exact
expert-ID gate compares the ordered top-6 tuple (`torch.equal`), and the
intra-set rank order differs at token 4. Changing that gate (e.g. set-based
route agreement) is a product-policy decision and is **not** done here.

## Files

- `router-reclassification.json` — machine-readable reclassified verdict.
- Evidence JSON/log preserved unmodified from the v10 run.

## Historical note

The v6/v8 `REJECT_STATE` attribution was already invalidated by the v9
harness fix (state snapshot aliasing). This document corrects the v10
diagnostic label in the same spirit: instrumentation/metrics defects, not
model behavior, produced those verdicts.
