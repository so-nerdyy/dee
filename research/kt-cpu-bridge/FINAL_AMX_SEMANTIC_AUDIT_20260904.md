# Final pinned-KT semantic audit: no passing faithful fix

Audit base: isolated branch `codex/kt-real-expert-review`, commit `5650045`.
Upstream pin: `31985f40bcc40da08107efdb1f81bf88cb38c6b2`.

There is no faithful fix that can satisfy the current all-row acceptance
requirements. This is stronger than merely lacking native KT outputs, but
does **not** mean implementing a more faithful software model is impossible:

- Exact equality with the FP32 oracle is impossible for the public KT BF16
  output surface on these samples. The unchanged strict allclose check is
  also impossible for thousands of coordinates, even with optimal BF16
  outputs and otherwise perfect arithmetic (proof below).
- DS8 is a different, looser numerical contract. Its unchanged reference-only
  coverage gate rejects steps 0 and 3 even for identity. No candidate can
  repair that gate on the current three-row acceptance set.
- Source-faithful grouped arithmetic can be implemented as a diagnostic
  without native dumps, but existing emulation is not that implementation.
  No native KT execution, binary-specific equivalence, or passing native DS8
  result follows from this audit.

Only research evidence, tests, and this document were added. No executor,
shared dee source, tolerance, integration, scheduler, notebook, or CPU-enable
setting was changed; no branch was merged and no Kaggle run was launched.

## 1. Kernel and layout corrections

Despite its AMX class/directory name, this pinned MXFP4 operator uses
AVX-512 BF16 ZMM dot products, not AMX tile dot-product instructions. Its
4-by-4 prefill block is register blocking, not a hardware AMX tile. Each
32-value group produces 16 pairwise sums; FP32 group scaling feeds lane
accumulators in ascending group order, followed by horizontal reduction.
`reduce4` reduces four independent output rows, not four terms of one dot.
[Pinned kernel, lines 165-173, 305-367, 439-478, 528-590](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/fp4-moe.hpp#L305).

Logical prefill pairs adjacent columns. Natural single-token decode first
permutes each 32-value activation group to evens then odds and decodes weights
in matching order. This preserves real arithmetic but changes finite-precision
pair grouping. A one-row replay must not silently stand in for the original
prefill dispatch. The sealed step-0 route has seven token rows; steps 3 and
10 are decode. The emulator has no such dispatch distinction.
[Pinned dispatch and permutation](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/fp4-moe.hpp#L755).

Packed weights use row-major byte offsets `row*(K/2)+column/2`, with
low-nibble-first logical columns. Gate/up dimensions are 2048x4096; down
is 4096x2048. No weight transpose is missing in the existing `x @ W.T`
reference. The `col majored` comment on the generic buffer is not its actual
address calculation. `BufferABF16Impl` can block K generally; here K is below
7168 and M_STEP is 1, so logical input rows are contiguous before the explicit
decode permutation. BufferC uses blocked N offsets internally and restores
logical output order in `to_mat`.
[Weight addresses](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/amx_buffers.hpp#L1114),
[activation addresses](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/amx_raw_buffers.hpp#L69).

The earlier down-buffer conversion links to `amx_buffers.hpp:1660-1674`
were incorrect for this pin: the actual conversion is **1776-1793**, not
the unrelated packed-weight transpose at 1660. Also, scale compaction changes
logical payload bytes but does not shrink the original buffer allocation:
`required_size` still reserves four bytes per scale.
[Correct conversion and allocation](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/amx_buffers.hpp#L1776).

## 2. Conversion, activation, weighting, and merge

The public KT path has BF16 input, FP32 dot accumulators, BF16 gate/up
outputs, FP32 activation evaluation, BF16 activation output, FP32 down
accumulators, **BF16 down outputs**, FP32 weighted expert accumulation, and
final BF16 TP output. The existing emulator omits the bold boundary.
The shared conversion helper uses native nearest-even BF16 conversion when
available, otherwise an explicit rounding/packing implementation.
[Gate/up and down materialization](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/moe_base.hpp#L549),
[conversion helper](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/utils.hpp#L14).

For alpha=0, KT uses the same asymmetric limit-10 clamp and SiLU-times-up
formula, but its exponential is a range-reduced fifth-degree FP32 polynomial
with fused Horner steps, followed by FP32 division. PyTorch SiLU or ordinary
`exp` does not reproduce those instructions. The existing division/exp
ablation does not measure KT's polynomial. Native integer conversion rounding,
FMA semantics, exceptional-value behavior, and compiler expansion of horizontal
reduction must be fixed before calling an emulator instruction-exact.
[Pinned activation](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/amx.hpp#L22).

Router weights are applied after widening BF16 down outputs, in supplied
expert-rank order with FP32 FMA; skipped experts are not renormalized. TP
partials are then added by increasing TP index before final BF16 conversion;
incremental mode first adds the previous BF16 output. This is not equivalent
to rounding every expert's final contribution separately. The captured single
expert cannot prove a six-expert or multi-TP merge.
[Weighting and merge](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/moe_base.hpp#L620).

TP divides intermediate channels: gate/up split rows, down splits columns.
Each partition's down output is rounded before weighted/TP accumulation.
Therefore changing TP count changes arithmetic boundaries, not merely stride.
No native KT TP count/build/dispatch is present in this dee-only capture.
[TP division](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/moe-tp.hpp#L118),
[partition copies](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/fp4-moe.hpp#L1114).

## 3. New representation proof on the sealed samples

[The proof script](evidence/check_bf16_output_lattice.py) revalidates capture
provenance and the six source tensors, recomputes all three FP32 oracle rows,
and requires their hashes and existing final-only/identity diagnostics to
match the archived evidence exactly. It changes no reference computation.

For every reference coordinate it finds the immediately bracketing BF16
values. With `atol=1e-5`, `rtol=1e-4`, the strict inequality is
`abs(reference-candidate) <= atol + rtol*abs(candidate)`. Because rtol is
below one, its residual increases as a candidate moves away from the reference
on either side. If both neighboring BF16 values fail, every BF16 value fails.
The neighbor argument is also tested against an exhaustive enumeration of
all finite BF16 values on independent positive and negative fixtures.

| Captured step | Non-BF16 oracle coordinates / 4096 | Coordinates no BF16 value can satisfy under strict allclose | Minimum possible maximum absolute error |
|---|---:|---:|---:|
| 0 | 4095 | 2578 | 0.000240519643 |
| 3 | 4096 | 2719 | 0.000242501497 |
| 10 | 4095 | 3276 | 0.000486686826 |

Example: step 10 coordinate 0 is `0.1059316098690033`. Adjacent BF16 values
are `0.10546875` and `0.10595703125`. Both fail the fixed inequality; even
the closer one exceeds its allowance by `4.825677872e-6`.

This is a lower bound, not a fabricated KT candidate. It applies to the
current public BF16 result and emulator return surface. Exposing FP32
internals or changing conversion boundaries would define a different output
contract; this proof does not rule out evaluating such a future variant.

[Durable proof results](evidence/captured-real-expert155-bf16-output-lattice-20260904.json)
retain the original oracle and capture hashes. Nearest BF16 alone is within
DS8 numerical limits; it does **not** explain the emulator's DS8 error.
Existing real boundary ablations still show p99 failures after adding the
missing preweight down rounding, with unchanged gates. Steps 0/3 additionally
exclude 126/4096 and 110/4096 outputs, above DS8's reference-only 2% cap.

## 4. Verdict and verification

No adjustment to input layout, nibble order, scales, clamp, routing weights,
or the known omitted down boundary is evidenced as a passing faithful fix.
All captured input values and decoded weights are exactly BF16-representable;
the internal-rounding/cancellation findings remain valid for the emulator,
but are not a native KT numerical measurement.

A faithful source-level implementation is possible in principle without
native outputs. **Verified native parity is not established**, and no such
implementation can overcome the output-representation and reference-coverage
obstructions under the present contracts. A native intermediate/output trace
with exact build, ISA, dispatch, TP, and merge configuration would establish
which native arithmetic actually occurs; it would not waive those contracts.

Verification in the isolated worktree, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`:

- Sealed lattice proof: completed; all three oracle SHA-256 values exactly
  reproduced, source/capture gates revalidated, expected blocking verdicts.
- Four new proof tests plus seven existing captured-diagnostic tests:
  **11 passed** in 10.16 seconds. This is evidence-test success, not parity.
- Initial proof attempt rejected a Python tuple versus serialized JSON array
  comparison of route occurrences. JSON normalization preserves every field,
  ordered triple, and hash; no provenance check was removed. The initial test
  run lacked the not-yet-generated result JSON; the completed rerun is above.

Reproduce from this isolated worktree:

```text
python research/kt-cpu-bridge/evidence/check_bf16_output_lattice.py --out <new-report.json>
python -m pytest research/kt-cpu-bridge/evidence/test_bf16_output_lattice.py dee.cpp/experiments/kt_cpu_bridge/tests/test_captured_real_diagnostics.py -q
```

The default archived diagnostics contain the exact local read-only source
paths. The new script never calls the C++ executor or native KT. CPU fallback
remains disabled, and this result does not justify a merge or cherry-pick of
the runtime interface.
