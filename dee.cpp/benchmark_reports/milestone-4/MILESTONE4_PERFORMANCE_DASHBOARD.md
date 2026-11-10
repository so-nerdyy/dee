# Milestone 4 Ornith performance dashboard

Status: Phase 1 sealed; exact seven-variant Phase 2 matrix sealed; first
measured post-baseline Pareto point passed on two Tesla T4 GPUs.

## Outcome

- The production cache point remains 32 experts/layer, with a measured peak
  process allocation of 7,036,862,464 B (6.554 GiB) on each GPU.
- The accepted pre-optimization cap-32 control is 6.487701 generated tokens/s.
  The same-session native-host control in the optimization A/B measured
  6.963004 TPS.
- Keeping the official checkpoint router on its resident layer GPU measured
  **7.948850 TPS**, a **14.1583%** median gain over the paired native-host
  control with no measured VRAM increase.
- Exact tokens, decoded text, all 40 layers, selected expert IDs, routing
  weights, router logits, expert outputs, intermediate hidden states, final
  hidden state, and LM-head logits were preserved.
- **20-30 full-model TPS remains physically plausible, but is not achieved or
  demonstrated.** The new point is 39.74% of 20 TPS and 26.50% of 30 TPS.

## First real Pareto point

Run identity:

- Kaggle kernel:
  `nivind/dee-cpp-ornith-m4-device-router-pareto`, version 1
- Kaggle terminal state: `COMPLETE`
- Run ID: `20260728T235600Z-router-pareto`
- Runtime commit: `9aaf727873cd63198c03f76492ade98f32b5d75c`
- Model: Ornith-1.0-35B, 16 real checkpoint shards, 70,218,758,304 B
- Runtime: PyTorch 2.10.0+cu128, Transformers 5.14.1, two Tesla T4 GPUs
- Workload: prompt `Hello`, greedy generation, IDs `[11, 271, 40, 1044]`,
  all 40 layers, top-8, cap 32, 20/20 layer split, one loaded runtime
- Measurement: one excluded warmup/backend, then three pair-balanced
  alternating trials/backend

| Backend | Trial TPS | Median TPS | Median decode/token | Peak process VRAM/GPU |
|---|---|---:|---:|---:|
| `native-host` | 6.973971, 6.562279, 6.963004 | 6.963004 | 144.358 ms | 7,036,862,464 B |
| `torch-device` | 7.951658, 7.948850, 7.943851 | **7.948850** | **125.868 ms** | 7,036,862,464 B |

The paired median improvement is 0.985846 TPS, 14.1583%, and 17.812 ms per
decode token. The optimized run's peak host RSS was 18,898,866,176 B
(17.601 GiB).

The changed boundary is deliberately narrow. `native-host` copies the hidden
state to the host, calls the dee.cpp router, copies router outputs back, and
performs a scalar tie-canonicalization synchronization. `torch-device` calls
the already-loaded official Transformers checkpoint router module on its
layer GPU. dee.cpp still owns every selected expert cache lookup, expert
forward, SwiGLU, and weighted combination.

| Three-trial proof counter | `native-host` | `torch-device` |
|---|---:|---:|
| Router calls | 480 native | 480 device |
| Router hidden D2H | 3,932,160 B | **0 B** |
| Router output H2D | 537,600 B | **0 B** |
| Router scalar synchronizations | 480 | **0** |
| Expert host fallback calls | 0 | 0 |
| Expert device-path share | 100% | 100% |

The traced A/B pair had zero maximum absolute and relative error in all eight
categories: embedding output, expert IDs, router logits, routing weights,
selected expert outputs, intermediate hidden states, final hidden state, and
LM-head logits. Remote gates also passed 11/11 CUDA CTests and 9/9 focused
Python tests.

## Exact Phase 2 matrix

Phase 2 ran at
`17e32c9f462ace4d416d330181a6efad716f00bd`, a descendant of the sealed
Phase 1 integration `b98696395b51006f71a85c328acacffbf65d29d2`.
Every row passed its variant-specific exact-token gate, all 40 layers, 100%
device expert execution, zero host fallback, no trace-abort marker, and the
8 GiB/GPU process-allocation ceiling.

| Variant | Policy | Output IDs | Full-model decode TPS | Peak process VRAM/GPU |
|---|---|---|---:|---:|
| `dual-cold-primary` | cap 32, cold, profiler + parity | 11, 271, 40, 1044 | 0.002794 | 7,053,639,680 B |
| `dual-warm-profiled` | cap 32, warm, profiled | 11, 271, 40, 1044 | 0.550327 | 7,053,639,680 B |
| `dual-warm-control` | cap 32, warm, unprofiled | 11, 271, 40, 1044 | **5.984249** | 7,036,862,464 B |
| `dual-warm-reference-present` | cap 32, warm, parity | 11, 271, 40, 1044 | 4.816952 | 7,036,862,464 B |
| `dual-cache-disabled` | cache disabled diagnostic | 11, 271 | 2.420891 | 7,032,668,160 B |
| `dual-cache-capacity-4` | explicit sub-top-k diagnostic | 11, 271 | 2.502366 | 3,509,452,800 B |
| `dual-long-prompt` | cap 32 long-prompt diagnostic | 198, 760 | 1.323567 | 7,049,445,376 B |

The cold and profiled rates include intentional first-touch/profiler costs and
are not steady performance baselines. The warm control is the Phase 2
throughput row.

### Phase 2 terminal-state qualification

The seven experiments and inner Phase 2 report passed with terminal reason
`ALL_SEVEN_UNIQUE_CAP32_MATRIX_ROWS_PASS`. Kaggle nevertheless marked the
outer notebook `ERROR` after the matrix because the packaging cell rejected
the expected zero-byte
`runs/dual-cold-primary/warmup-expert-trace.jsonl`; a cold run has no warmup.

The host recovery seal does not relabel that notebook state. It records
`matrix_result=PASS`, `notebook_result=ERROR`, and
`manifest_provenance=HOST_RECOVERED_AFTER_NOTEBOOK_PACKAGING_ERROR`. It
verified 128 downloaded artifacts, all 104 intended required paths, the 32-row
inner artifact manifest, exact semantic fingerprints, token IDs, VRAM, and the
31,384,431-byte raw allocation trace. The notebook allowlist is fixed for
future reruns.

## Critical path and physical bound

The real tensor map contains 70,214,363,872 B of model tensors, of which
64,424,509,440 B are routed experts. A single token has a calculated mandatory
active-weight floor of 5,892,863,232 B across the two GPUs.

| Point | TPS | Effective mandatory-weight bandwidth | Share of 270 GB/s |
|---|---:|---:|---:|
| Accepted cap-32 baseline | 6.487701 | 38.231 GB/s | 14.16% |
| Paired native-host control | 6.963004 | 41.032 GB/s | 15.20% |
| Device-router Pareto | **7.948850** | **46.841 GB/s** | **17.35%** |
| 20 TPS target | 20.000000 | 117.857 GB/s | 43.65% |
| 30 TPS target | 30.000000 | 176.786 GB/s | 65.48% |

At a realistic 270 GB/s sustained assumption, the static single-sequence
floor is 45.82 TPS. That is a permissive physical ceiling, not a forecast: it
excludes activation traffic, rereads, cache-line effects, and orchestration.
The measured trace contains 6,166 kernels and is GPU-busy for only 18.80% of
its combined 330.222 ms span, so the current implementation is dominated by
fragmented GEMV/elementwise launch orchestration rather than cap-32 expert H2D.

The device-router point still needs a 2.516x gain to reach 20 TPS and a 3.774x
gain to reach 30 TPS. The earlier 30.571 number is not evidence for either
target: it measured single-T4 synthetic recurrent MoE steps/s using generated
one-layer 2048x64 experts and omitted attention, recurrent state, norms,
residuals, vocabulary head, tokenization, and genuine multi-GPU execution.

## Next measured gates

1. Add direct CUDA timing around the linear-attention/recurrent path, then fuse
   or capture that path without changing its recurrence or precision. Its
   current 47.074 ms attribution is a differential inference, so direct
   counters must precede a causal speedup claim.
2. Fuse expert weighted combination with adjacent normalization/residual work.
   The current directly measured rows are 15.728 ms for combination and
   21.159 ms for normalization per profiled token.
3. Reduce the 6,166-launch decode graph through CUDA Graphs or faithful fused
   GEMV/elementwise kernels. Preserve the exact eight-category trace gate.
4. Collect Nsight Compute `dram__bytes_read.sum` and achieved-bandwidth
   counters before calling a later point bandwidth-bound.

Quantization, batching, speculative/MTP decoding, and approximate routing are
separate Pareto families. None may be multiplied into this single-sequence
exact-FP16 result without its own quality, acceptance, memory, and end-to-end
verification.

## Reproduction and integrity

```powershell
python -m pytest -q tests --import-mode=importlib

python scripts/seal_milestone4_phase2.py `
  --download-root <downloaded-phase2-evidence> `
  --expected-commit 17e32c9f462ace4d416d330181a6efad716f00bd `
  --expected-run-id 20260728T232556Z-phase2-cap32 `
  --output-dir <phase2-seal-output> `
  --recover-post-matrix-packaging-error `
  --notebook-terminal-state ERROR

python scripts/seal_ornith_router_pareto.py `
  --download-root <downloaded-router-pareto-evidence> `
  --expected-commit 9aaf727873cd63198c03f76492ade98f32b5d75c `
  --expected-run-id 20260728T235600Z-router-pareto `
  --output-dir <router-seal-output>
```

Tracked seal SHA-256 values:

- Phase 1 compact seal:
  `9038b0648feefd91342868b5e176bc1485ce91a7617cbac8e175aec1e898b2a7`
- Phase 2 host-revalidated seal:
  `971fc7eb6f62cc2307e4a7a9154daba8842515346c8cbb5e66c3ad0ad980e5a1`
- Phase 2 recovered 128-artifact manifest:
  `259e104df68493aa42a38be3011da005ae16c46580aec1b77af6a812e059897b`
- Router Pareto benchmark report:
  `dd5f3256d083ab72a42e9c60570b0984366d4e2e140281e4c623494583a01b24`
- Router Pareto downloaded root manifest:
  `4e44de287588ab46b19d0589604c5cf79e3ce74ff7bbdffa81e25facf6b090ae`

The router seal hash is generated from the current validator and should be
read from `router-device-pareto/seal-manifest.json`; changing the validator's
compact summary intentionally changes that seal hash while the immutable
downloaded report hash above remains fixed.
