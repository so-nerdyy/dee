# Real-expert numerical review: blocked, no runtime change

Reviewed on `codex/kt-real-expert-review` at base `95db693`. All investigation
and builds were confined to the isolated review worktree; the existing local
checkpoint shard and sealed terminal bundle were read-only inputs. No merge,
Kaggle run, notebook change, scheduler change, tolerance change, or runtime
source change was made.

**Result:** the supplied failure reproduces exactly. The portable FP32
reference passes the existing dee DS8 contract but fails the additional strict
allclose check. The BF16 emulator fails DS8. An omitted native-KT rounding
boundary is source-proven, but inserting that boundary in a diagnostic-only
calculation does not resolve either reported failure. No faithful,
strict-correctness fix has been demonstrated; real activation/routing replay
and native KT execution remain missing.

## Evidence and scope

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`, revision
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`, layer 0, expert 155.
- Canonical record: 13,369,344 bytes, SHA-256
  `b8763344da8fcb3ed511c61c53079ac9bc3fa7727b4c85b03ba8729e7cf3e61d`.
- The six component hashes, canonical record hash, model identity, seal, and
  expert occurrences were rechecked by `load_real_expert` before execution.
- Inputs remain `randn(2,4096, seed=20260903)`, second row multiplied by 3;
  routing weights remain `1.0, 0.37`. Input SHA-256:
  `8fcdb0491511496f4edd3221c1b8257dc265444bc9eef1fa23cbe67ffa5d87e5`.
- All 688 sealed route-journal records contain expert IDs and route metadata,
  not activation tensors or numerical routing weights. Observed expert
  occurrences `(forward_step, token_row, rank)` are `(0,4,3)`, `(3,0,3)`,
  `(10,0,0)`; the diagnostic rows are **not** those captured model rows.
- Python: PyTorch `2.7.1+cu118`, one CPU thread, MKL 2025.1, AVX2 runtime.
  Build: MSVC 19.44.35228.0, Visual Studio 17 2022, Release.
- The unchanged reference executable SHA-256 is
  `a92f3eaa3d9abdf3642a53aa018370c874d2ca95bd4557f64fe342f2a80eb584`.
- `tmp/kt-real-expert-report-rerun.json` is structurally identical to
  `tmp/kt-real-expert-report-final.json`; both failure statuses are unchanged.
- Durable measurements, exact failing coordinates, codec checks, input paths,
  environment, tolerances, and probe identity are in
  [the diagnostic evidence](evidence/real-expert-numeric-ablation-20260903.json).
  No tensors or native-KT outputs are represented by this evidence.

## 1. C++ strict failure: reduction sensitivity, not FP4 layout

Strict tolerance remains `atol=1e-5, rtol=1e-4`; DS8's non-near-zero p99
relative-error limit remains `0.05`. The reference follows the same clamp and
before-down routing-weight placement as dee's trusted Python bridge.
As called by the verifier, PyTorch allclose uses
`abs(reference-candidate) <= atol + rtol*abs(candidate)`.

| Calculation | Row 0 max abs / strict failures | Row 1 max abs / strict failures |
|---|---:|---:|
| Actual unchanged C++ reference | 1.740456e-5 / 0 | 4.959106e-5 / 9 |
| Explicit left-to-right FP32 dots, division/exp SiLU | 1.740456e-5 / 0 | 4.959106e-5 / 9 |
| Scalar gate/up dots only, PyTorch down dot | 1.287460e-5 / 0 | 4.434586e-5 / 6 |
| PyTorch gate/up dots, scalar down dot only | 9.775162e-6 / 0 | 3.623962e-5 / 0 |
| FP64 dot accumulations cast to FP32 at each output | 6.675720e-6 / 0 | 2.008677e-5 / 1 |

The explicitly scalar Python calculation differs from the actual C++ output
by at most `4.768372e-7` in either row, with row-1 mean absolute difference
`4.656613e-10`. It reproduces the same nine failing coordinates. This isolates
the dominant discrepancy to the FP32 reduction path in
`reference_cpu_executor.cpp:123-138`, compared with PyTorch matrix
multiplication; nonlinear implementation differences are much smaller here.

The maximum-absolute-error coordinate 2909 is not itself a strict failure:
its reference magnitude is about 6.61, so the relative allowance covers it.
Failures occur at smaller outputs. For example, coordinate 3507 is
`-0.00824546814` versus `-0.00826625340`: error `2.078526e-5` exceeds its
`1.082663e-5` allowance. Coordinate 932 still fails even with FP64 dot
accumulation: error `1.198146e-5` exceeds `1.127005e-5`.

Actual C++ DS8 p99 values remain `9.317662e-5` and `9.798646e-5`, both passing.
The trusted reference explicitly does not promise bitwise identity to a
particular reduction order. A higher-precision accumulation is not an exact
match to the trusted FP32 execution, and did not satisfy the unchanged strict
gate on these probes. No accumulation change was justified as a verified fix.

## 2. BF16 failure and omitted down-output boundary

The pinned KT implementation has a BF16 down-output buffer before weighting:
`down_bc_->to_mat` converts its FP32 result to BF16, then that buffer is widened
for FP32 weighted expert accumulation. Final TP merge converts the combined
result to BF16. This applies to both decode and prefill. The prototype's
`reference.py:59-61` and `kt_cpu_executor.cpp:159-166` omit the preweight
down-output conversion. Sources: [AMX MoE base, pinned lines 362-391,
549-576, 707-710](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/moe_base.hpp#L549),
[BufferCReduceImpl conversion, lines 1660-1674](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/amx_buffers.hpp#L1660).

Let `B` be the prototype's FP32-to-BF16-to-FP32 round trip, and `y` its
unweighted down dot. The existing result is `B(weight*y)`; the tested
additional-boundary result is `B(weight*B(y))`. This is a boundary ablation,
**not** an exact emulator of KT's grouped SIMD reductions or activation
approximation, and **not** native KT execution.

| BF16 calculation vs original FP32 reference | Row 0 p99 | Row 1 p99 | DS8 |
|---|---:|---:|---|
| Input conversion only | 0.151330 | 0.151617 | fail / fail |
| Gate/up conversion only | 0.131109 | 0.128397 | fail / fail |
| SwiGLU intermediate conversion only | 0.137190 | 0.116749 | fail / fail |
| Final output conversion only | 0.003474 | 0.003480 | pass / pass |
| Existing emulator | 0.226189 | 0.271500 | fail / fail |
| Existing emulator plus preweight down conversion | 0.226189 | 0.271853 | fail / fail |

Row 0 has routing weight 1, so the additional boundary changes zero elements.
Row 1 changes 1,084/4,096 outputs and increases maximum absolute error from
`0.117431640625` to `0.13319015502929688`. Therefore the omission is a real
fidelity defect, but it is **not the cause of the two reported DS8 failures**.
On these particular probes, input and internal BF16 boundaries already
produce p99 errors above the gate; this is not evidence about unseen model
activations.

The original emulator also exceeds DS8's mean-relative-error and p95 limits:
mean `0.020291/0.022643` against `0.01`, and p95 `0.044226/0.047566` against
`0.03`. These failures remain failures; p99 is not the only failing metric.

Even full-FP32 late routing-weight placement is only algebraically equivalent:
it produces one strict row-1 failure (DS8 still passes). Exact numerical
identity must not be inferred from the distributive law.

Further fidelity limits: the pinned [MXFP4 kernel uses paired BF16 products,
per-group scaling, and vector reduction](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/fp4-moe.hpp#L148),
and its [SiLU uses a polynomial exponential approximation](https://github.com/kvcache-ai/ktransformers/blob/31985f40bcc40da08107efdb1f81bf88cb38c6b2/kt-kernel/operators/amx/la/amx.hpp#L20).
The portable PyTorch/scalar code does not reproduce those details. Rounding
each expert separately also cannot be assumed to model a multi-expert final
merge. The existing label "KT-faithful" is therefore too strong for exact
native parity; this report supersedes that interpretation.

## 3. Codec and test-construction findings

For all three real projections, the bridge's unpack/dequant result equals
the trusted dee result exactly. The source scales are 119-122 (`2^-8` through
`2^-5`). Direct `(u8<<7).view(bf16)->fp32` scale adaptation equals trusted
E8M0 decoding, compaction preserves every byte, and expansion recovers every
value. Every resulting real dequantized weight is exactly BF16-representable.
Thus nibble order, projection order, scale adaptation, and weight rounding do
not explain these probes' failures. This is measured evidence for this
expert, not an assertion about every possible E8M0 codepoint or checkpoint.

The `0x00` loader edge is still different: raw exponent-bit shifting yields
zero, whereas the bridge's canonical decoder returns `2^-127`. The bridge
rejects `0xFF`. Neither value occurs in the verified record, so neither was
used to explain or repair the failures.

The small synthetic tests use scale bytes 125-130 and `hidden=64, inter=32`,
not this real expert's scales and dimensions. The emulator test gates on
`p95<0.15`, mean relative error, and cosine; it does not gate on DS8 p99.
Passing that test does not imply a real-weight DS8 pass. The real-weight
probe does exercise clamping: row 1 has 44 gate and 90 up clamp events;
row 0 has none. Changing seeds, activation dtypes, routing weights, exclusion
thresholds, or tolerances to obtain a pass was not attempted.

## Verification and reproduction

Executed in
`C:\Users\carth\Downloads\dynamic_expert_eviction_kt_real_expert`:

1. `cmake --build tmp\kt-reference-build --config Release`: PASS.
2. `tmp\kt-reference-build\Release\kt_bridge_reference_probe.exe --self-test`:
   PASS (`reference guards OK`).
3. `tmp\kt-reference-build\Release\kt_bridge_smoke.exe`: PASS
   (`ref=851.986450`, `kt=856.000000`; synthetic smoke only).
4. Unchanged `bench/verify_real_expert.py`: exit 1, identical supplied report.
5. Diagnostic-only `python tmp\diagnose_kt_numeric.py`: completed; source
   SHA-256 `48bd26fd0ebfe706119da36ec3c8e622adfac61d65ce0a87c97287000501b105`.
   It uses sealed real tensors, sequential FP32 dot ablations, FP64 dot
   ablations, and independently enabled BF16 boundaries. It modifies no
   executable source or acceptance gate. The local script is not committed.
6. Emulator/reference unit tests (`test_kt_bridge_correctness.py`, excluding
   the separately executed real integration): 4 passed, 1 deselected.
7. Full isolated bridge suite with `DEE_REAL_EXPERT_DIR` set and a fresh
   CMake Release smoke build in the test-discovered directory:
   **26 passed, 1 failed, 0 skipped** in 41.86 seconds. The failure is
   `test_real_expert_if_available` at line 125, the unchanged strict gate.
   The earlier unconfigured suite was 25 passed, 2 skipped; that result is
   not used as real-expert acceptance.
8. Existing dee expert reference, MoE reference, and DS8 contract tests:
   **49 passed** in 8.07 seconds (`test_deepseek_v4_expert_reference.py`,
   `test_deepseek_v4_moe_reference.py`, `test_deepseek_v4_contract.py`).

The exact verifier arguments are the `shard`, `bundle`, `seal`, and executable
paths recorded in the evidence (executor path ends in
`tmp\kt-reference-build\Release\kt_bridge_reference_probe.exe`), plus
`--out tmp\kt-real-expert-report-rerun.json`. Full-suite command:

```powershell
$env:DEE_REAL_EXPERT_DIR = 'C:\Users\carth\Downloads\dynamic_expert_eviction_kt_real_expert\tmp'
python -m pytest dee.cpp\experiments\kt_cpu_bridge\tests -q -ra --tb=short
```

An initial attempt at the similarly named `ds5-v3c-output` shard failed
closed because that file is empty. The actual validated input is the
`ds5-v3b-output` partial shard identified in the evidence; no download or
checkpoint mutation occurred.

## Required evidence before a numerical change can be accepted

Obtain sealed expert-input activation rows, exact numerical routing weights,
and dtype/configuration from the same real run; preserve row identity and
source hashes. Then compare the unchanged trusted dee path with native KT
on its supported ISA, recording intermediate tensors and complete merge
semantics. Reproduce any proposed correction under the original strict/DS8
gates. Until that evidence exists, this is a diagnosed blocker, not a
successful repair or model-parity result.
