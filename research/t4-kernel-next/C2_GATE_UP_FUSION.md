# C2 gate/up fusion (prototype built, T4 gate pending)

Prototype: `dee.cpp/experiments/t4_kernel_next/c2_gate_up_stack.cu`
(+ `kaggle_microbench_three_way.py` case `c2`). C12 untouched.

## What it tests

Whether the two production GEMMs — W1 gate and W3 up, each
`cublasGemmEx` m=2048/n=1/k=4096, FP16→FP16, FP32 accumulate, TENSOR_OP
(`swiglu_cuda.cu:214`, via `swiglu_expert_batch_fp16_cuda`) — can be replaced
by ONE m=4096 GEMM over a stacked `[W1;W3]` buffer. Concatenation is NOT
assumed exact: the binary first proves stacked-decode halves equal separate
decodes bitwise (same per-element algorithm, structural), then compares
2×(m=2048) vs 1×(m=4096) outputs.

## Preserved exactly

Official packed bytes, scalar-port E2M1/E8M0 decode, `__float2half_rn`
per-element rounding, row-major orientation via `CUBLAS_OP_T`, FP32
accumulation, TENSOR_OP algorithm selection, FP16 output. Clamp/SiLU/multiply
are downstream and untouched (existing kernel keeps its semantics).

## Admission

- `BITWISE_IDENTICAL`: stacked halves match AND GEMM outputs match bitwise.
- `NUMERICALLY_DIFFERENT`: any mismatch — timings reported but inadmissible
  for an exactness claim; design becomes REJECT_NUMERICS unless the F2
  oracle gate (unchanged tolerances) passes, which then labels it
  `ORACLE_PASS_ONLY` (still not "exact").
- The word "exact" is used only if `BITWISE_IDENTICAL` prints.

## Traffic / launch accounting (batch-1 per expert)

- Bytes read: unchanged (2×16.78 MB either way). Bytes written: unchanged
  (8 KB either way). Temporary: +33.55 MB stacked buffer if separate buffers
  are also kept; zero if the stacked layout becomes the decode target.
- Launches: 7 → 6 per expert (one GEMM fewer). Arithmetic: unchanged by
  construction; library algorithm choice is the only variable.
- Bottleneck affected: launch overhead (~5–10 µs saved per expert), not
  bandwidth. Expected value is small but nearly free if the gate passes.

## Production touch on success

`swiglu_cuda.cu` (`swiglu_expert_batch_fp16_cuda` GEMM pair) + decode-target
layout in `engine.cpp` (`decode_fp4_cache_block_to_scratch` writes stacked).
No kernel rewrite, no residency change, no API change.
