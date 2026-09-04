# Correctness-contract audit of the sealed real KT replay

**Classification: `CONTRACT_REVIEW_REQUIRED_KT_STILL_FAILS`.**

The exclusion cap is a sample-coverage condition currently folded into an
implementation verdict. It rejects the trusted FP32 oracle against itself on
two of three real rows. That structural issue does not explain the KT
emulator's relative-error failures. Independently, even an ideal BF16 output
cannot satisfy the unchanged strict FP32 allclose on any captured row.
Preserve `KT_NUMERICALLY_INCOMPATIBLE_WITH_STRICT_FP32_CONTRACT` for the
audited BF16 output surface; do not infer that all possible future KT
precision variants, or native KT under DS8 alone, have been disproved.

No policy, tolerance, executor, production source, CPU-enable setting,
scheduler, or notebook was modified. No merge, cherry-pick, or Kaggle launch
occurred. All new files are audit evidence under `research/kt-cpu-bridge`.

## Scope, provenance, and proof level

The [machine-readable audit](evidence/captured-real-expert155-contract-audit-20260904.json)
revalidates the guarded capture, all listed artifact hashes, terminal
43-layer/16-token/688-route generation gates, real source record, six weight
component hashes, complete router vectors, and unchanged contract source.
It reruns the existing portable C++ FP32 probe twice per row and recomputes
trusted dee and KT emulator outputs. All nine output hashes, original DS8
metrics, and the previously sealed boundary-ablation metrics reproduce
exactly. The trusted oracle is a **CPU FP32 replay**, not a captured GPU
expert-output dump. No native KT code was executed.

- Isolated branch: `codex/kt-real-expert-review`; audit base `f44967f`.
- Model scope: layer 0, expert 155; gate/up 2048x4096, down 4096x2048.
- Capture SHA-256: `444ed65883ece90811b3fffb17da81192741e6abb936b56d7026b04e3d805496`.
- Record SHA-256: `b8763344da8fcb3ed511c61c53079ac9bc3fa7727b4c85b03ba8729e7cf3e61d`.
- Unmodified contract SHA-256: `52361fbaeaefd6b03b64fb05888668a7c60513f4746ee067e788883a7174e895`.
- Rows `(forward step, token row, rank)`: `(0,4,3)`, `(3,0,3)`, `(10,0,0)`.
- Selected router weights: `0.29187867045402527`, `0.3334415555000305`,
  `0.4717525541782379`.

This is three weighted contributions from one expert, not coverage of all
experts, six-expert merging, multiple TP partitions, or full-model KT parity.
The public KT BF16 result boundary supports the representation proof for a
BF16 vector compared to these reference rows; it does not turn these rows
into a captured full-MoE reference.

## 1. Every DS8 gate, separately

`P`/`F` indicate the unmodified gate result. Relative metrics use only
reference coordinates with `abs(ref) >= 0.001`, with denominator
`abs(ref)+1e-8`. Absolute/global metrics include every finite coordinate.
Full-precision values and booleans for every check are retained in JSON.

### Trusted dee versus portable C++ FP32 reference

| Check | Limit | Step 0 | Step 3 | Step 10 |
|---|---:|---:|---:|---:|
| max_abs_error | <=2 | 1.34110451e-7 P | 1.71363354e-7 P | 4.09781933e-7 P |
| mean_abs_error | <=0.5 | 1.55975766e-8 P | 1.86415381e-8 P | 4.43985009e-8 P |
| mean_rel_error | <=0.01 | 1.30050410e-6 P | 1.36196149e-6 P | 1.47670892e-6 P |
| p95_rel_error | <=0.03 | 4.47300681e-6 P | 4.68578264e-6 P | 4.94336882e-6 P |
| p99_rel_error | <=0.05 | 1.10040373e-5 P | 1.26151899e-5 P | 1.63261375e-5 P |
| cosine_similarity | >=0.999 | 0.999999940 P | 1.000000238 P | 1.000000119 P |
| normalized_rmse | <=0.01 | 8.32200726e-7 P | 8.66785399e-7 P | 8.31642296e-7 P |
| output_norm_rel_error | <=0.02 | 7.26037460e-8 P | 0 P | 0 P |
| excluded_fraction | <=0.02 | 0.03076171875 F | 0.02685546875 F | 0.01123046875 P |
| exact finite/sentinel mask | required | P | P | P |
| finite overlap | required | 4096 P | 4096 P | 4096 P |
| NaN / +inf / -inf counts, either side | no forbidden values | 0 / 0 / 0 P | 0 / 0 / 0 P | 0 / 0 / 0 P |
| fixed FP32 allclose | atol=1e-5, rtol=1e-4 | P; 0 failures | P; 0 failures | P; 0 failures |
| aggregate DS8 | all DS8 checks | F: coverage only | F: coverage only | P |

Cosines very slightly above one are the existing FP32 computation, preserved
verbatim rather than clamped for this audit. The allclose check is separate
from `ds8_gate_passed`: its preserved call is `torch.allclose(ref,cand)`, so
its relative term uses **candidate** magnitude. No argument order is silently
changed.

### Trusted dee versus existing KT emulator

| Check | Limit | Step 0 | Step 3 | Step 10 |
|---|---:|---:|---:|---:|
| max_abs_error | <=2 | 0.000367850065 P | 0.000354126096 P | 0.001411452889 P |
| mean_abs_error | <=0.5 | 0.0000641321094 P | 0.0000701775862 P | 0.000242170921 P |
| mean_rel_error | <=0.01 | 0.00597759522 P | 0.00593725219 P | 0.01043707784 F |
| p95_rel_error | <=0.03 | 0.02091914043 P | 0.02144110575 P | 0.03953742981 F |
| p99_rel_error | <=0.05 | 0.05617049709 F | 0.05554705858 F | 0.12115249783 F |
| cosine_similarity | >=0.999 | 0.999995589 P | 0.999996185 P | 0.999992192 P |
| normalized_rmse | <=0.01 | 0.00318513764 P | 0.00299137691 P | 0.00407654978 P |
| output_norm_rel_error | <=0.02 | 0.000347045906 P | 0.000227314037 P | 0.000260788666 P |
| excluded_fraction | <=0.02 | 0.03076171875 F | 0.02685546875 F | 0.01123046875 P |
| exact finite/sentinel mask | required | P | P | P |
| finite overlap | required | 4096 P | 4096 P | 4096 P |
| NaN / +inf / -inf counts, either side | no forbidden values | 0 / 0 / 0 P | 0 / 0 / 0 P | 0 / 0 / 0 P |
| fixed FP32 allclose | atol=1e-5, rtol=1e-4 | F; 3602 failures | F; 3650 failures | F; 3925 failures |
| aggregate DS8 | all DS8 checks | F: p99 + coverage | F: p99 + coverage | F: mean + p95 + p99 |

Thus excluding only the candidate-independent coverage predicate from the
**diagnostic decomposition**, without adopting a changed acceptance policy,
leaves C++ numerically passing all rows and the emulator failing all rows.
Step 10 already satisfies coverage and demonstrates the independent KT issue.

## 2. Contract well-formedness and historical rationale

Source: [`deepseek_v4_contract.py`](../../dee.cpp/scripts/deepseek_v4_contract.py),
especially the module rationale, `compute_ds8_metrics`, and `ds8_gate_passed`.
The initial contract is commit `3ff61ff075975a2e74f418129a91268d6abeac07`
(2026-07-31), declared before DS8 execution. Its rationale cites DS7
max/mean absolute and mean/p99 relative errors with headroom for the weighted
six-expert combination. The
[DS7 state record](../../dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/PROJECT_STATE.md)
describes the near-zero cancellation observation; the
[DS8 dashboard](../../dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/CAMPAIGN_DASHBOARD.md)
records successful combined-output synthetic-corpus exclusion fraction
0.0005. That evidence is not a calibration of weighted single-expert real
contributions. No reviewed source derives the precise 0.001 cutoff or 0.02
cap quantitatively, nor declares a separate sample-validity status.

Answers to the requested questions:

- **Reference-only?** On every admissible finite comparison, yes. The mask
  is `abs(reference)<0.001`; candidate values never enter that expression.
  Identity and all 17 finite boundary variants have identical coverage per
  real row. The reference-only excluded counts are 126, 110, and 46 / 4096.
- **Can a candidate alter it?** A finite candidate cannot. Technically the
  implementation first selects the **finite intersection** of both vectors,
  so a candidate can alter the reported fraction by injecting nonfinites.
  Against these all-finite references, that immediately fails the exact-mask
  semantic check. The audit reproduces this explicitly; it is not a bypass.
- **Why pass/fail today?** It is an explicit conjunction in
  `ds8_gate_passed`, apparently limiting how much relative-error coverage may
  be excluded. That coverage-control interpretation is an inference from
  code and tests, not a separately documented author decision.
- **Was it intended as sample validity?** Its semantics fit sample validity,
  but history does not establish that intended separation. The shipped API
  instead reports a single Boolean. It is logically consistent as a joint
  sample-and-candidate acceptance predicate, but not a valid attribution of
  implementation error: even an identical candidate is rejected.
- **Multiple real rows?** Yes: steps 0 and 3, including FP32 identity.
  Step 10 passes coverage. The earlier lattice JSON's phrase
  `all_row_ds8_rejected...` means the all-rows acceptance conjunction fails;
  it does not mean every individual row fails coverage.

The documentation's statement that relative error is undefined for a
near-zero reference is imprecise: at a nonzero value it is defined but
ill-conditioned; at zero it is undefined without regularization. The code
uses a fixed cutoff and epsilon, which must be declared as contract choices.

### Additional nonfinite enforcement defect

The `ds8_gate_passed` docstring says NaN/+inf anywhere fails. The function
checks only `sentinel_mask_exact` (actually equality of **finite** masks),
`finite_overlap`, and numeric metrics; it does not inspect NaN/+inf counts.
The audit reproduces accepted `[1,NaN]` versus itself, `[1,+inf]` versus
itself, `[1,-inf]` versus `[1,NaN]`, and `[1,-inf]` versus `[1,+inf]`.
Each violates the documented no-NaN/+inf rule while actual DS8 returns true.
The tests record this existing defect rather than patching it. None of the
real replay comparisons has any nonfinite or sentinel value, so this defect
does not affect the six observed comparisons or their numerical diagnosis.

### Proposed design, not implemented

1. Seal sample identity, reference/output surface, dtype, unit/scale, routing
   semantics, finite-reference mask, expected permitted sentinel type, and
   coverage before evaluating any candidate. Keep cutoff **0.001** and cap
   **0.02** unchanged pending explicit review. Above-cap samples become
   `SAMPLE_INVALID_RELATIVE_COVERAGE`, not candidate failures or passes.
2. Evaluate candidate shape, exact permitted sentinel classes/positions,
   explicit forbidden NaN/+inf counts, and all unchanged numerical gates.
   Use the sealed reference mask, not a candidate-selected intersection.
   Require strict FP32 allclose separately with its existing orientation.
3. Report sample validity, candidate fidelity, and overall acceptance
   independently. Invalid samples cannot contribute a pass or simply be
   dropped to improve an aggregate. Preserve their observed diagnostics;
   require predeclared valid replacement coverage before acceptance. This
   audit proposes no new samples or relaxed thresholds.

That structure preserves strictness and explains why a reference can be
numerically faithful even when its sample is invalid for a coverage rule.
It does not certify KT: its candidate-dependent failures remain measured,
and the strict BF16 representability obstruction is independent of coverage.

## 3. Actual error counts and the unavoidable BF16 output floor

All counts are out of 4096; comparisons are against trusted FP32. Threshold
counts use the exact FP32 values widened to FP64 for subtraction, and strict
`>` rather than `>=`. The ideal row is nearest-even BF16(reference), an
absolute-error lower bound, **not** a proposed candidate implementation.

| Output | Step | Different | >1e-6 | >1e-5 | >1e-4 | >1e-3 |
|---|---:|---:|---:|---:|---:|---:|
| C++ FP32 | 0 | 3973 | 0 | 0 | 0 | 0 |
| C++ FP32 | 3 | 3967 | 0 | 0 | 0 | 0 |
| C++ FP32 | 10 | 3950 | 0 | 0 | 0 | 0 |
| Existing KT emulator | 0 | 4096 | 4053 | 3679 | 862 | 0 |
| Existing KT emulator | 3 | 4096 | 4062 | 3741 | 1019 | 0 |
| Existing KT emulator | 10 | 4095 | 4081 | 3983 | 3068 | 4 |
| KT + missing down BF16 boundary | 0 | 4096 | 4056 | 3705 | 1042 | 0 |
| KT + missing down BF16 boundary | 3 | 4096 | 4065 | 3768 | 1212 | 0 |
| KT + missing down BF16 boundary | 10 | 4095 | 4083 | 4002 | 3097 | 22 |
| Ideal BF16(reference) | 0 | 4095 | 3848 | 2716 | 199 | 0 |
| Ideal BF16(reference) | 3 | 4096 | 3874 | 2865 | 255 | 0 |
| Ideal BF16(reference) | 10 | 4095 | 4001 | 3423 | 1283 | 0 |

Ideal-BF16 output metrics:

| Step | max_abs | mean_abs | mean_rel | p95_rel | p99_rel | cosine | NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000240519643 | 0.0000291013512 | 0.00140853529 | 0.00297062914 | 0.00343586528 | 0.999999285 | 0.00166682142 |
| 3 | 0.000242501497 | 0.0000332550408 | 0.00141189445 | 0.00295524905 | 0.00347151142 | 0.999999344 | 0.00163399812 |
| 10 | 0.000486686826 | 0.0000842053487 | 0.00141590356 | 0.00302794063 | 0.00348224537 | 0.999998987 | 0.00166519894 |

Ideal BF16 passes all candidate-dependent DS8 numerical gates. Therefore
**final output rounding alone does not explain the emulator's DS8 failures**.
Nevertheless, the bracketing-neighbor proof finds **2578 / 2719 / 3276**
coordinates where *no BF16 value* satisfies fixed FP32 allclose. For
`rtol<1`, moving farther from the reference on either side only increases
the strict residual, so testing both adjacent BF16 values suffices. This
proves the strict-FP32 obstruction without claiming native KT fidelity.

KT versus the portable FP32 executor remains essentially the same diagnosis:
p99 is `0.0561736934 / 0.0555450618 / 0.1211496517`; fixed-allclose failures
are `3602 / 3649 / 3925`. Full metrics/counts are in `cpp_vs_kt_emulator`.
KT differs from ideal BF16 at `2208 / 2123 / 2570` coordinates; p99 relative
errors against that ideal are `0.0557487980 / 0.0567033663 / 0.1202210188`.
These comparisons isolate substantial internal arithmetic error beyond the
unavoidable final BF16 lattice error.

### ULP interpretation

FP32 ULP distance counts representable steps, with signed zeros collapsed.
Below are same-sign coordinates with `abs(trusted)>=0.001`, avoiding
misleading sign crossings/near-zero cancellation. JSON additionally records
all-finite ULP distributions and BF16 ULP distances for BF16-to-BF16 pairs.

| Comparison | Step | ULP mean | ULP p95 | ULP p99 | ULP max |
|---|---:|---:|---:|---:|---:|
| C++ | 0 | 15.65 | 55 | 126.62 | 337 |
| C++ | 3 | 16.35 | 57 | 141 | 414 |
| C++ | 10 | 17.76 | 63 | 189.02 | 773 |
| KT emulator | 0 | 71779.26 | 265718 | 660973.86 | 2093476 |
| KT emulator | 3 | 71274.08 | 262219.5 | 617973.4 | 1961652 |
| KT emulator | 10 | 125345.48 | 475470 | 1427464.72 | 7513216 |
| Ideal BF16 | 0 | 16348.23 | 31181.05 | 32524.62 | 32767 |
| Ideal BF16 | 3 | 16456.42 | 31086.5 | 32384.2 | 32762 |
| Ideal BF16 | 10 | 16403.57 | 31157.2 | 32457.02 | 32754 |

KT sign disagreements over all finite coordinates are 6 / 2 / 5, all below
the reference non-near-zero cutoff. ULP is descriptive, not another gate;
large FP32 ULP counts are expected when converting to a seven-bit BF16
fraction and do not on their own establish DS8 failure.

## 4. Attribution using the same real samples

The audit recomputes all previously sealed independent/cumulative boundary
interventions against unchanged references and thresholds. Each intervention
has full metrics, counts, ULPs, allclose, and gate decomposition in JSON.

| Intervention | p99 step 0 | p99 step 3 | p99 step 10 |
|---|---:|---:|---:|
| BF16 input only | 0 | 0 | 0 |
| BF16 gate only | 0.034053 | 0.033267 | 0.081916 |
| BF16 up only | 0.030990 | 0.032470 | 0.051475 |
| BF16 gate + up only | 0.046655 | 0.043684 | 0.087976 |
| BF16 intermediate only | 0.033040 | 0.033302 | 0.049785 |
| Gate/up + intermediate, no final rounding | 0.056767 | 0.054971 | 0.121267 |
| Existing emulator | 0.056170 | 0.055547 | 0.121152 |
| Existing + preweight-down BF16 | 0.055774 | 0.056001 | 0.121152 |

- **Input/codec:** every captured original FP32 activation is exactly BF16
  and exactly its captured FP16-widened value. Input conversion changes zero
  coordinates. Weight decode and E8M0 adaptation are exact for all six real
  tensors; all scale bytes are 119-122 and all weights exactly BF16.
- **Gate/up boundaries:** gate rounding changes 2048 coordinates in every
  row; up changes 2048 / 2047 / 2048. These isolated interventions cause
  substantial relative tails; gate alone fails step 10. No clamp fires.
- **Intermediate/down input:** full intermediate changes all 2048 entries
  per row. Maximum errors are `0.002289295 / 0.002352953 / 0.013246536`.
  Combined gate/up/intermediate rounding crosses p99 in all rows before
  final output rounding. Errors are not additive causal percentages: the
  nonlinear activation and subsequent rounding interact.
- **Down output boundary:** the existing emulator omits pinned KT's BF16
  down materialization before weighting. Restoring it diagnostically changes
  1109 / 1025 / 1005 outputs, but fails the same relative gates; max absolute
  error becomes `0.000472054 / 0.000463128 / 0.001891911`.
- **Late routing weight:** moving only the FP32 router multiplication has
  p99 `1.0284e-5 / 1.1993e-5 / 1.3524e-5`, max absolute error <=2.0862e-7.
  Thus it is not the observed tail's dominant cause. Captured full router
  vectors and rank selection are exact; no renormalization is substituted.
- **Cancellation witness:** step 10 coordinate 1022 is reference
  `-0.001279011369`, emulator `-0.000690460205`. Absolute down terms sum to
  `1.077744484`, about 843 times the result. Changed intermediate terms
  project to `+0.000587672170`; final rounding adds `+0.000000910426`.
  This directly links internal rounding to the tail.

### Reduction order and AVX-512 dpbf16 sensitivity

Pinned KT uses zero-initialized BF16 pair dots, 32-coordinate groups, FP32
group scaling/accumulation, and horizontal reduction; natural decode pairs
evens-then-odds while the logical path pairs adjacent columns. These are
AVX-512 BF16 operations, not AMX tile arithmetic.
[Pinned kernel](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/fp4-moe.hpp).
BF16 multiplication is exact in FP32 before accumulation; denormal handling
and rounding remain relevant to a complete ISA model.
[Intel hardware numerics](https://www.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf).

New controlled real-data experiments retain PyTorch SiLU, all BF16 boundaries
including the missing down boundary, one contribution, and TP=1 assumptions:

| Dot/reduction intervention | Changed final coordinates vs restored-boundary BLAS, steps 0 / 3 / 10 |
|---|---:|
| FP64 dot, rounded FP32 at each projection | 0 / 0 / 0 |
| Left-to-right separate FP32 multiply/add | 0 / 0 / 0 |
| Logical pairs, adjacent horizontal tree | 0 / 0 / 0 |
| Logical pairs, halves horizontal tree | 0 / 0 / 0 |
| Natural pairs, adjacent horizontal tree | 0 / 0 / 1 |
| Natural pairs, halves horizontal tree | 0 / 0 / 0 |

The single differing output is one BF16 step (`0.00048828125`); p99 remains
unchanged. Some step-3 grouped variants change one rounded up-projection
coordinate but no final output. Every variant still fails p99 on all rows,
plus mean/p95 on step 10. The all-FP32 scalar and FP64-dot paths separately
pass fixed allclose; scalar max absolute errors are <=4.09782e-7 and FP64-dot
max absolute errors <=1.49012e-7.

The grouped diagnostic explicitly asserts exact real BF16/FP4 products,
exact power-of-two scaling, and no scaled-pair subnormals. It keeps pair
addition, ascending-group accumulation, and two explicit horizontal trees
separate. This quantifies reduction sensitivity for these inputs; it is
**not** proof of a particular compiler's `_mm512_reduce_add_ps` expansion,
native dispatch, TP partitioning, or exact native KT output.

Native KT also uses a polynomial exponential in its activation. The existing
division/exp ablation does not reproduce that polynomial, and this audit
does not claim to isolate its native contribution.
[Pinned activation](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/amx.hpp).
KT down outputs are BF16 before rank-ordered FP32 weighted accumulation;
TP results are merged before final BF16 conversion. Independently rounding
each expert contribution is not a full-MoE equivalent.
[Pinned weighting and merge](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/moe_base.hpp).

Consequently native KT's DS8-only status remains unresolved. The final
classification is nevertheless **STILL_FAILS**, not STATUS_UNRESOLVED,
because the unchanged strict FP32 requirement is impossible for any BF16
vector on these rows regardless of native polynomial, reduction, or ISA.

## 5. Commit visibility and verification

At audit start, live origin ended at `be46276`; `5650045` and `f44967f` were
local-only. Both existing research commits were pushed without rewriting
history. A fresh `git ls-remote` observation and `git merge-base --is-ancestor`
checks verify all three are reachable from the intended remote branch:

| Commit | Full SHA | Reachable |
|---|---|---|
| be46276 | be462765772cccc2a9825366a62c23da746bbc73 | yes |
| 5650045 | 5650045aac6481615b8260ab27f3fbbffd8f6215 | yes |
| f44967f | f44967ffbe29d62e89c2f005d95c5d2b6613a568 | yes |

[Remote receipt](evidence/contract-audit-remote-visibility-20260904.json):
`https://github.com/so-nerdyy/dee.git`,
`refs/heads/codex/kt-real-expert-review`, observed 2026-09-04 04:30:13 UTC.
The audit commit is pushed as a descendant and its final live remote tip is
verified separately after commit; this receipt intentionally records the
non-circular pre-audit-commit observation.

Verification: sealed replay completed with expected rejection classification;
all archived hashes and metrics reproduced. **53 tests passed in 8.09s**:
7 new audit tests, 4 output-lattice proofs, 7 captured-diagnostic tests, and
35 existing DS8/contract tests. This is evidence-test success, not KT parity
or a performance acceptance. No production-enable test was invoked.

Reproduce in the isolated checkout (PyTorch 2.7.1+cu118, one CPU thread;
the script sets its own thread count; source paths are read-only):

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python research/kt-cpu-bridge/evidence/audit_captured_contract.py --out tmp/contract-audit-rerun.json
python -m pytest research/kt-cpu-bridge/evidence/test_captured_contract_audit.py research/kt-cpu-bridge/evidence/test_bf16_output_lattice.py dee.cpp/experiments/kt_cpu_bridge/tests/test_captured_real_diagnostics.py dee.cpp/tests/test_deepseek_v4_contract.py -q
python research/kt-cpu-bridge/evidence/verify_contract_audit_remote.py --out tmp/contract-audit-remote-rerun.json
```

No acceptance or policy change is authorized by this audit. Production CPU
execution remains disabled. Stop after committing and pushing this evidence.
