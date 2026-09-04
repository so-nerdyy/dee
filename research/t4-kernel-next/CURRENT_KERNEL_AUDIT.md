# Current expert CUDA path audit (DeepSeek-V4, packed-FP4 residency)

Traces one routed expert on the live device path
(`Engine::moe_forward_batch_device_impl`, `engine.cpp:756-1067`,
`cache_dtype == Fp4E2m1`). Storage is packed official MXFP4; T4 executes
FP16 cuBLAS + custom kernels after decode — there is no native T4 FP4 GEMM
anywhere in this path. All line numbers at base `9c16529`.

## Pipeline per expert per token (batch-1 decode, cache MISS)

1. **H2D upload** — `stage_expert` (`:3104`) → `prefetch_fp4_regions_packed`
   gathers 6 mmap regions into pinned staging, `cudaMemcpyAsync` **12.75 MiB**
   packed (12,582,912 B weights + 786,432 B e8m0) to the VRAM cache block on
   the prefetch stream + event record (`async_prefetcher.cpp:713-722`).
2. **Stream wait** — `prefetcher_.wait_on_stream(source_layer, expert,
   compute_stream_)` (`:950`); `cache_.pin` (`:951`).
3. **Full FP16 decode** — `decode_fp4_cache_block_to_scratch` (`:3206-3229`)
   launches **3× `fp4_e2m1_to_f16_kernel`** (one per projection p<3 loop,
   `:3213-3221`), each reading packed+scales and writing FP16 into the single
   bounded `d_fp4_decode_scratch_`. **This runs on EVERY use, not just on
   miss** — packed residency saves capacity, not decode.
4. **Gate + up GEMMs** — `swiglu_expert_batch_fp16_cuda`
   (`swiglu_cuda.cu:387-431`): 2× `cublasGemmEx` FP16→FP16
   (`gemm_fp16_row_major_batch_to_fp16`), m=2048, n=tokens, k=4096.
5. **Activation** — 1× `swiglu_activation_batch_fp16_kernel` (`:46-59`):
   clamp (limit 10.0) + SiLU with FP16-round-then-multiply (already fused;
   see CANDIDATE_DESIGNS C3).
6. **Down GEMM** — 1× `cublasGemmEx` FP16→FP32
   (`gemm_fp16_row_major_batch_to_fp32`).
7. **D2D gather/scatter** — per-row `cudaMemcpyAsync` DeviceToDevice of the
   hidden row in (`:918-939`) and FP32 row out (`:1003-1021`); skipped when
   `direct_single_row_io` hits (`:903-906`, `:988-989`).
8. **Host sync** — `cudaStreamSynchronize(compute_stream_)` once per layer
   batch when `synchronize_output` (`:1037-1054`).

Cache HIT skips step 1 only; steps 2–7 (including the full 50.3 MB decode)
still run.

## Kernel / launch inventory per expert (batch-1)

| # | Launch | Source |
|---|---|---|
| 1–3 | 3× `fp4_e2m1_to_f16_kernel` (gate/up/down) | `cuda_convert.cu:231`, via `engine.cpp:3219` |
| 4–5 | 2× `cublasGemmEx` FP16→FP16 (gate, up) | `swiglu_cuda.cu:214`, via `:405-410` |
| 6 | 1× `swiglu_activation_batch_fp16_kernel` | `swiglu_cuda.cu:46`, via `:415-417` |
| 7 | 1× `cublasGemmEx` FP16→FP32 (down) | `swiglu_cuda.cu:239`, via `:422-424` |
| + | 2×group_tokens D2D `cudaMemcpyAsync` + 1 event wait | `engine.cpp:918,1003,950` |

7 compute launches + copies + waits per expert; ~258 experts/token →
~1,800 launches/token plus one host sync per layer. (`forward_layer_cuda`
SGEMV path `:3872+` is the synthetic-harness path, not the DSV4 path.)

## Buffers (device, batch-1)

- Cache block: 12.75 MiB packed (resident until evicted).
- `d_fp4_decode_scratch_`: **50.33 MiB FP16**, rewritten per expert use —
  the largest single materialization (MEMORY_TRAFFIC.md B1).
- `d_moe_batch_{input,gate,up,activation}_half_`: 8/4/4/4 KiB; `d_moe_batch_output_`
  16 KiB FP32 — negligible at batch-1, scale with group tokens.
- No persistent per-expert FP16 anywhere (by design, P2.3).

## Projection ordering / routing-weight placement

- Projections decode gate, up, down in that order into contiguous scratch;
  cuBLAS reads them as row-major `[inter,hidden]`, `[inter,hidden]`,
  `[hidden,inter]` via `CUBLAS_OP_T` with FP32 accumulation
  (`CUBLAS_COMPUTE_32F, GEMM_DEFAULT_TENSOR_OP` — T4 2nd-gen tensor cores).
- Routing weights are applied **outside** this path (Python/HF owns
  router+combine on `d_experts_out`); in-engine `weighted_combine_fp16_kernel`
  (`:86-116`) preserves two explicit FP16 rounding boundaries that any fusion
  must keep (CANDIDATE_DESIGNS C10).

## Classification of statements above

- *Measured*: none on T4 by this track (no T4 locally; nothing fabricated).
- *Source-derived*: launch counts, buffer sizes, ordering, sync points (all
  cited). Byte totals in MEMORY_TRAFFIC.md are arithmetic on canon geometry
  (12.75 MiB record, 25,165,824 params).
- *Theoretical*: roofline bounds in MEMORY_TRAFFIC.md §4, explicitly upper
  bounds, never converted to TPS.
