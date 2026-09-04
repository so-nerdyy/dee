# Captured real input review: internal BF16 rounding remains a blocker

Result: **no faithful passing numerical fix was found**. On the three sealed
real layer-0/expert-155 samples, the current emulator's gate/up BF16 conversion
and SwiGLU-intermediate BF16 conversion produce down-projection cancellation
error above the unchanged DS8 relative-error gates. Input conversion, the six
source tensors, FP4 decoding, E8M0 scale adaptation, routing weights, and clamp
selection do not explain this failure. No executor, runtime, tolerance,
scheduler, or notebook was changed; CPU execution remains disabled.

This review is confined to `codex/kt-real-expert-review`, starting at `be46276`.
The new [diagnostics](evidence/captured-real-expert155-diagnostics-20260904.json)
contain all measurements, source/input hashes, and local reproduction paths.
The [original evidence](evidence/captured-real-expert155-replay-20260904.json)
is unchanged. Every original row field, output hash, numerical metric, and
repeat check reproduced exactly after removing the added diagnostic fields.

## Scope and verified inputs

- Capture manifest SHA-256:
  `444ed65883ece90811b3fffb17da81192741e6abb936b56d7026b04e3d805496`.
- Real record SHA-256:
  `b8763344da8fcb3ed511c61c53079ac9bc3fa7727b4c85b03ba8729e7cf3e61d`.
  The replay revalidates all six source component hashes against both the
  capture validation and sealed official-checkpoint record, not a synthetic
  reconstruction. Shapes are gate/up 2048x4096 and down 4096x2048.
- Row identities `(step, token row, top-k rank)` are `(0,4,3)`, `(3,0,3)`,
  `(10,0,0)`. Selected FP32 weights are `0.29187867045402527`,
  `0.3334415555000305`, `0.4717525541782379`. The replay now additionally
  checks all six captured numerical router weights against their manifest
  vector. No router computation or renormalization is substituted.
- Each original FP32 input is exactly BF16-representable. Each native FP16
  input widened to FP32 is exactly the original input, including its complete
  FP32 SHA-256. All 4096 values agree in every row. Input BF16 conversion
  changes zero values. Thus this is also an exact-original-FP32 replay, not
  an input-dtype approximation. The previous synthetic-input review's
  input-rounding explanation does **not** apply to these samples.
- Every real scale byte is 119-122 (2^-8 through 2^-5). Source-to-BF16 scale
  adaptation, E8 byte compaction, and expansion exactly match the trusted
  E8M0 decoder. All dequantized weights are exactly BF16-representable.
  Bridge low-nibble-first FP4 dequantization exactly matches the trusted dee
  decoder for all three matrices. Neither E8M0 0 nor 255 occurs.
- Asymmetric clamp limit stays 10. No captured gate or up value reaches a
  clamp boundary: the observed gate range is within [-1.740, 2.155] and up
  within [-3.738, 1.969]. A clamp-placement change cannot explain these rows.

The oracle is the trusted dee **FP32 CPU replay** over the captured inputs,
not a captured GPU expert-output dump. There is no native KT execution,
supported-ISA performance result, or full six-expert merge parity claim.

## Boundary attribution on the same reference and gates

The table reports non-near-zero p99 relative error. DS8 remains `p99<=0.05`,
`mean_rel<=0.01`, `p95<=0.03`, and all other original gates remain enforced.
Each row uses the exact same tensors, input, and router weight for every
intervention. Removing boundaries is diagnostic only, not an accepted fix.

| Intervention | Step 0 | Step 3 | Step 10 |
|---|---:|---:|---:|
| BF16 input only | 0 | 0 | 0 |
| BF16 gate only | 0.034053 | 0.033267 | 0.081916 |
| BF16 up only | 0.030990 | 0.032470 | 0.051475 |
| BF16 gate and up only | 0.046655 | 0.043684 | 0.087976 |
| BF16 SwiGLU intermediate only | 0.033040 | 0.033302 | 0.049785 |
| Gate/up + intermediate BF16, before final rounding | 0.056767 | 0.054971 | 0.121267 |
| Final BF16 output only | 0.003436 | 0.003472 | 0.003482 |
| Existing emulator | 0.056170 | 0.055547 | 0.121152 |
| Existing + preweight down-output BF16 | 0.055774 | 0.056001 | 0.121152 |

The first two samples cross p99 when internal rounding is combined. On step
10, gate rounding alone already fails. The full step-10 emulator also fails
mean relative error (`0.010437`) and p95 (`0.039537`). Removing only final
rounding still fails all three samples; removing gate/up rounding still
leaves step 10 p99 at `0.050293`. These are not acceptable replacement paths.

FP32 late router-weight placement alone has p99
`1.0284e-5 / 1.1993e-5 / 1.3524e-5`, and maximum absolute errors at most
`2.0862e-7`. Replacing PyTorch SiLU with FP32 division/exp alone has p99 at
most `1.5950e-6`. Neither explains the observed error scale.

### Concrete cancellation witness

Step 10, coordinate 1022, has reference `-0.001279011369` and candidate
`-0.000690460205`, a relative error of about 46%. The absolute values of the
reference down-dot terms sum to `1.077744484`, approximately 843 times the
result's magnitude. Projecting the changed intermediate through the same
down weights accounts for `+0.000587672170`; the final BF16 rounding adds only
`+0.000000910426`. Corresponding worst-coordinate cancellation ratios are
about 528 and 601 for steps 0 and 3. This directly locates the error in the
internal rounded representation and cancellation, not in final output
rounding or a routing/codec mismatch.

## Known emulator fidelity defect is not a passing fix

At the pinned upstream KT commit, both decode and prefill store the down
projection into a BF16 buffer before FP32 weighted accumulation. The current
portable Python and scalar C++ emulators omit that boundary. Sources:
[down buffer and weighting](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/moe_base.hpp#L542-L576),
[FP32-to-BF16 buffer conversion](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/amx_buffers.hpp#L1660-L1674).

The diagnostic adds exactly `B(weight * B(down))` in place of
`B(weight * down)`, where B is the BF16 roundtrip. It changes 1109, 1025, and
1005 final coordinates, but all three samples still fail p99 and step 10
still fails mean and p95. Maximum absolute errors increase to
`0.000472054 / 0.000463128 / 0.001891911`. Consequently this fidelity defect
is documented and tested, but is not shipped as a purported numerical fix.

Native KT also has grouped SIMD reductions and
[a polynomial SiLU exponential](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/amx.hpp#L20-L66).
Neither the existing emulator nor these PyTorch boundary ablations is an
exact substitute for that implementation. A future precision-preserving KT
variant must be implemented and measured as a new candidate on supported
hardware; it cannot be accepted by removing BF16 boundaries only in this
emulator or by changing DS8 exclusions.

## Independent acceptance blocker: coverage

Steps 0 and 3 exclude 126/4096 (`3.076171875%`) and 110/4096 (`2.685546875%`)
outputs below the unchanged `1e-3` reference threshold. Both exceed the
unchanged `2%` cap. Even comparison of the FP32 oracle with **itself** fails
that coverage gate. This is independent of KT error and cannot be repaired
by changing a candidate. The portable C++ FP32 reference passes its fixed
allclose, finite, deterministic, and numerical checks on all three rows,
but strict DS8 is false for the first two because of coverage. No strict
all-row acceptance is claimed, even for a hypothetical exact candidate.

## Verification and reproduction

- CMake Release builds in both `tmp/kt-reference-build` and the isolated
  bridge's `build` directory: passed. Reference self-test and C++ smoke:
  passed. Reference executable hash stayed
  `a92f3eaa3d9abdf3642a53aa018370c874d2ca95bd4557f64fe342f2a80eb584`.
- Sealed replay with `--diagnostics`: completed, expected rejection exit 1.
  All original replay rows are structurally identical to the baseline;
  original output hashes and both repeat checks remain intact.
- Full isolated bridge tests plus trusted expert/MoE/DS8 suites, with
  `DEE_REAL_EXPERT_DIR` configured: **82 passed, 1 failed, 0 skipped** in
  67.02 seconds, including all seven new tests. The failure is the unchanged
  `test_real_expert_if_available` strict allclose gate on the previous
  **synthetic activation** probe; it is not this captured-input replay.
- New diagnostics test file covers path fidelity, missing down boundary,
  asymmetric clamp, finite masks, unmodified coverage rejection, archived
  replay failures, and exact preservation of the original evidence.

Run from the isolated worktree, with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`, using the exact `diagnostics.sources` and `executor.path`
fields in the new JSON:

```text
python research/kt-cpu-bridge/evidence/replay_captured_real_expert.py
  --shard <sources.shard> --capture_bundle <sources.capture_bundle>
  --validation <sources.validation> --sealed_bundle <sources.sealed_bundle>
  --terminal_seal <sources.terminal_seal> --executor <executor.path>
  --diagnostics --out <new report path>
```

The diagnostic report includes the replay, diagnostic-script, and unchanged
DS8 contract source SHA-256 values; environment was PyTorch 2.7.1+cu118, one
CPU thread. This offline measurement does not enable runtime CPU fallback.
No merge or cherry-pick is justified by this result.
