# Format Compatibility — dee official MXFP4 vs KTransformers MXFP4

2026-09-03 correction: real tensor decoding/scale adaptation was verified,
but the portable emulator omits KT's preweight BF16 down-output boundary and
does not reproduce native SIMD reductions/activation approximation. Its
real-weight DS8 probe fails. The mathematical-equivalence claims below do
not establish exact numerical parity; see
[the numerical review](NUMERICAL_REVIEW_20260903.md).

Branch: `research/kt-cpu-bridge`. Upstream commit `31985f40bcc40da08107efdb1f81bf88cb38c6b2`.
References: dee `dee.cpp/include/dee/weight_mmap.h`, `dee.cpp/src/weight_mmap.cpp`,
`dee.cpp/src/cuda_convert.cu`, `dee.cpp/scripts/deepseek_v4_expert_reference.py`,
`dee.cpp/scripts/deepseek_v4_moe_reference.py`; KT
`kt-kernel/operators/amx/fp4-moe.hpp`, `kt-kernel/operators/avx2/mxfp4-moe.hpp`,
`kt-kernel/operators/amx/la/amx_buffers.hpp`, `kt-kernel/python/utils/loader.py`,
`kt-kernel/python/utils/amx.py`.

Question: can KTransformers directly execute dee's official DeepSeek-V4-Flash
routed-expert representation (checkpoint MXFP4: E2M1 packed + E8M0 scales)?
"MXFP4" is NOT assumed byte-identical — proven field by field below.

## 1. Verdict up front

- **Weight bytes: byte-identical, zero-copy.** `memcpy` dee's mmap `I8` bytes
  straight into KT `BufferB.b` / `tpc.gate/up/down_proj`. No transformation.
- **Scales: value/bit-exact but NOT byte-identical on disk.** Checkpoint `1 B
  ue8m0` -> loader `2 B bf16` -> C++ `4 B fp32` -> compacted `1 B e8` are all
  lossless round-trips for every scale byte found in real checkpoints. The
  adapter must perform this re-encode per expert on demand (costed in §9).
- **Math: equivalent up to KT's documented bf16-boundary rounding.** No
  semantic mismatch. Only `e=0/255` edge encodings differ; absent from real
  checkpoints; adapter fails closed on `0xFF`.
- **A full-model second RAM copy is NOT required and NOT used.** Per-expert
  on-demand conversion into a bounded host-cache entry is viable (§9-§10).

## 2. Tensor layout

| Item | dee (official) | KT MXFP4 | Match |
|---|---|---|---|
| Matrix order | Row-major `[out, in]`. `fp4_e2m1_dequantize(packed, scale, out, in, dst)` rows `o*packed_in`, `o*scale_in`, `o*in` (`weight_mmap.h:190-194`, `weight_mmap.cpp:101-117`) | Row-major. `BufferBInt4KGroupImpl::from_raw_mat/get_submat/get_scale`: `row_bytes=k/2`, `d + n_begin*(k/gs) + g` (`amx_buffers.hpp:1114-1138`); `loader.py:1202-1203` documents `[N,K/2]`, `[N,K/32]` | YES |
| Projection order | `GATE=w1, UP=w3, DOWN=w2` (`weight_mmap.h:107,112,129`, `weight_mmap.cpp:319-322`) | `PROJ_NAMES=("w1","w3","w2")=(gate,up,down)` (`loader.py:1214`, `amx.py:877-878`) | YES |
| Shapes | `w1/w3:[inter,hidden]`, `w2:[hidden,inter]`; packed `[out,in//2]`, scale `[out,in//32]` (`tests/test_deepseek_v4_moe_reference.py:69-74`, `scripts/deepseek_v4_expert_reference.py:85-110`) | Same (`gate:N=inter,K=hidden`; `down:N=hidden,K=inter`, `fp4-moe.hpp:728-753,1062-1071`) | YES |
| Row ordering / transpose | None, out-major contiguous | None on MXFP4 path: `from_raw_mat` is plain `memcpy` (`amx_buffers.hpp:1114-1124`). Blocked-tiled variant is RAWINT4-only, unused by `GemmKernel224MXFP4SmallKGroup` (`N_STEP=32,K_STEP=32`, `fp4-moe.hpp:31-33`) | YES |
| Endianness | x86-64 LE; safetensors header `u64 LE`; weights/scales are byte arrays | Same; only bf16/fp32 reinterpret is LE-native | YES |

## 3. Packing order

Both **low nibble first**:

- dee: `low = byte & 0x0F -> col 2i`, `high = (byte >> 4) -> col 2i+1`
  (`weight_mmap.h:168-173`, `weight_mmap.cpp:110-111`,
  `cuda_convert.cu:241-244`, `scripts/deepseek_v4_expert_reference.py:65-80`).
- KT: `lo = packed & 0xF`, `hi = (packed >> 4) & 0xF`, output
  `[lo0,hi0,lo1,hi1,...]` (`fp4-moe.hpp:69-98`; AVX2 comment
  `packed[j]={lo=col2j,hi=col2j+1}`, `avx2/mxfp4-moe.hpp:60-83`).

Verified by dee test `test_unpack_fp4_official_nibble_order` (`0x12 ->
[1.0, 0.5, ...]`) and KT `examples/test_fp4_moe_v4.py` V4-layout validation.

## 4. E2M1 codepoints

Identical including the `-0.0` quirk:

```
index: 0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
value: 0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0  -0.0 -0.5 -1.0 -1.5 -2.0 -3.0 -4.0 -6.0
```

- dee: `FP4_TABLE` (`scripts/deepseek_v4_expert_reference.py:35-41`),
  `weight_mmap.cpp:76-82`, `cuda_convert.cu:208-212`.
- KT BF16 LUT `{0x0000,0x3F00,0x3F80,0x3FC0,0x4000,0x4040,0x4080,0x40C0,
  0x8000,0xBF00,...}` = same values (`fp4-moe.hpp:49-64`, AVX2 `53-56`).

KT decodes via PSHUFB byte-LUT (`mxfp4_to_bf16_32`) or word-permute natural
LUT (`mxfp4_to_bf16_32_natural`, `__AVX512BF16__` only); AVX2 fallback uses
`_mm512_permutexvar_ps` over the same fp32 table. All three are exact.

## 5. Block size

**32 on the K (in) axis, both sides.** dee `FP4_BLOCK_SIZE=32`,
`w[o,i] = fp4 * scale[o, i//32]` (`expert_reference.py:39,85-110`); KT
`K_STEP=32`, `group_size` inferred as `hidden // scale.shape[1]` and required
to be 32 for the AMX natural path (`amx.py:879-881`, `fp4-moe.hpp:658,756`).
`NVFP4/group-16` is explicitly rejected on AMX (`amx.py:911-917`).

## 6. Scale representation

Both `value = 2^(bits - 127)`:

- dee: `e8m0_to_f32: (exp+127)<<23` (`weight_mmap.cpp:88-99`), CUDA identical
  (`cuda_convert.cu:218-224`). Python: `torch.pow(2.0, bits-127)` with
  bit-reinterpret across `uint8/int8/float8_e8m0fnu` dtypes
  (`expert_reference.py:49-62`).
- KT: loader `_ue8m0_to_bf16 = (u8 << 7).view(bf16)` (`loader.py:1222-1231`)
  — lossless (`exp=e, mant=0`); C++ `convert_or_copy(bf16->fp32)` exact
  (`common.hpp:442-470`); `finalize_scale_e8` validates positive-pow2
  (`sign=0, mant=0, exp not in {0,0xFF}`) and compacts `fp32 -> 1 B exp` in
  place; `get_scale / expand_e8_scales` rebuild via `e << 23`,
  `copy_scale_to_bf16` via `e << 7` (`fp4-moe.hpp:243-303,390-401`).

Edge bytes (never in real checkpoints, `REALISTIC_SCALE_BYTES=(0x7D,0x83)`
-> scales `0.25..8.0`):

| byte | dee host | dee CUDA | KT |
|---|---|---|---|
| `0x00` (`e=0`) | `2^-127` (subnormal-adjacent; doc notes below-BF16) | same formula | `+0.0` (validation path treats as non-E8M0 or compacts exactly; bf16 round-trip is `+0.0`) |
| `0xFF` (`e=255`) | clamps to `2^127` | `FLT_MAX` | `+inf` |

Realistic bytes are bit-exact end to end. Adapter fails closed on `0xFF`
(see `cpu_executor.hpp`).

## 7. Output precision

- KT: `BF16 in -> E2M1->BF16 LUT -> _mm512_dpbf16_ps -> FP32 dot ->
  FP32 x scale -> FP32 acc -> reduce -> BufferC FP32 -> fp32->bf16`
  (`fp4-moe.hpp:165-173,350-355`, `amx_buffers.hpp:1776-1793`; AVX2
  `fmadd_ps`, `mxfp4-moe.hpp:189-217`). Gate/up outputs are bf16; SwiGLU runs
  in fp32 and rounds back to bf16 (`moe_base.hpp:705-737`, natural
  `fp4-moe.hpp:771-799`).
- dee reference: full fp32 (`expert_reference.py`); dee CUDA: fp16/bf16
  SwiGLU with fp32 accumulation (`swiglu_cuda.cu`).

Expected delta vs dee fp32 reference is therefore KT's bf16-boundary rounding
(gate/up outputs, final merge), NOT a layout mismatch. Phase D gates quantify
it (`max_abs/mean_abs/mean_rel/p95/p99/cosine/finite`) without weakening
existing dee gates.

## 8. Routing-weight semantics

- dee official: scalar applied to the intermediate BEFORE `w2`
  (`h = weights * h; h @ W2`, `scripts/deepseek_v4_moe_reference.py:108-110`).
- KT: applied AFTER `w2` in fp32 (`x += w*down_out`, `moe_base.hpp:620-638`).

Mathematically identical (`(w·h)W^T = w·(hW^T)` — dee's own MoE test asserts
the combine identity). Only numeric delta is KT's intermediate bf16
round-trips. Adapter implements the dee placement (weight before `w2`) in the
portable reference and documents the KT placement as algebraically equivalent;
`test_kt_bridge_correctness.py::test_routing_weight_placement_equivalence`
proves it.

Router itself (scores `sqrt(softplus(x@W^T))`, bias shifts selection only,
weights normalized non-softmax × `route_scale=1.5`) is OUTSIDE both the KT CPU
kernel and the dee adapter — owned by dee `route_topk` / HF model. The adapter
takes a scalar `routing_weight` per call and never renormalizes (skipped
experts contribute 0, matching KT `should_skip`).

## 9. Required conversion (exact pipeline)

```
SSD packed expert (official safetensors)
  layers.<L>.ffn.experts.<E>.{w1,w2,w3}.weight  I8  [out, in//2]
  layers.<L>.ffn.experts.<E>.{w1,w2,w3}.scale   F8_E8M0 [out, in//32]
        |
        v  (HostPackCache fill: verbatim gather, no transform)
bounded RAM cache entry  (~12.6 MB/expert at H=4096/I=2048 scale;
                          ~23.4 MB at H=7168/I=2048; 6 regions contiguous)
        |
        +---> weights: ZERO-COPY memcpy -> adapter/KT BufferB.b
        |
        +---> scales:  LOSSLESS re-encode (per expert, on demand):
                       u8 mmap --(u8<<7).view(bf16)--> bf16
                            --(exact bf16->fp32)--> fp32
                            --(finalize_scale_e8 validate+compact)--> 1 B e8
                       (torch vectorized part <1 ms; C++ validate+compact 1-2 ms)
        |
        v
KT-resident flat blob:  required_size = n*k/2 + n*k/gs  (64 B aligned)
  [packed weights][compacted e8 scales], no pointers -> memcpy-evictable
        |
        v
CPUInfer (BufferB)  -- fp4_mat_vec/mat_mat_kgroup --> fp32 -> bf16 out
```

What is NOT needed: GGUF conversion, offline `.kt` pre-quant, full-model
second copy, AMX-only instructions (AVX2 fallback uses the same layout with
fp32 scales).

## 10. Cost quantification

Shapes: per projection `(out, in)` with packed `out*in/2` B + scales
`out*in/32` B. Per expert total = `3*(I*H/2 + I*H/32)`.

| example shape | per-expert source | KT compacted resident | pre-compact transient | python transient |
|---|---|---|---|---|
| `H=4096, I=2048` (V4-Flash-0731 family scale) | ~12.6 MB (~11.9 MB weights + ~0.7 MB scales) | ~12.6 MB | ~14.2 MB (weights + fp32 scales) | +~1.4 MB bf16 scale copy |
| `H=7168, I=2048` (V3-family ref shape) | ~23.4 MB (~21 MB + ~1.3 MB) | ~22.3 MB | ~26.5 MB | +~2.6 MB |

Time/expert (single core @ ~10-20 GB/s memcpy + vectorized convert):

- weights `memcpy` (~12-21 MB): ~1-2 ms.
- scales convert+validate+compact (~0.4-1.4 M elems): ~1-2 ms; torch
  vectorized slice `<1 ms`.
- Parallelized over experts (`do_work_stealing_job`); on-demand top-6/8 adds
  ~20-35 ms single-threaded decode latency — feasible with prefetch/overlap,
  but a bounded host LRU is strictly better (below).

RAM rule: conversion transient is per-expert and freed in place
(`scale_e8` bytes OVERWRITE the fp32 contents — forward compression is
overlap-safe by construction, `fp4-moe.hpp` `BufferB` comment). Steady-state
residency per cached expert EQUALS the source size (no inflation).

Cacheability: post-`finalize_scale_e8` the blob is flat with no pointers;
evict/reload is `memcpy`. Cache the COMPACTED `e8` bytes in dee's bounded host
cache to skip torch conversion on hit. `WeightMmap::discard_source_pages`
composes: mmap source pages droppable after the private copy.

## 11. dee-side proof obligations (Phase D)

`dee.cpp/experiments/kt_cpu_bridge/tests/test_kt_bridge_correctness.py`:

- `unpack_fp4` nibble order vs `scripts/deepseek_v4_expert_reference.py`.
- `decode_e8m0` formula vs dee (incl. cross-dtype reinterpret).
- `dequantize` shape/scaling identity.
- `expert_forward` clamp asymmetry (`gate=max-only`, `up=±`, `limit=10.0`,
  `limit=0` disables).
- routing-weight placement equivalence (before-`w2` vs after-`w2`).
- determinism (repeated execution bitwise identical).
- error metrics `max_abs/mean_abs/mean_rel/p95_rel/p99_rel/cosine/finite`
  of the KT-emulated (bf16-roundtrip) path vs the fp32 reference — gates
  recorded but existing dee gates NOT weakened.

## 12. Blocker list (format)

1. NONE blocking for weights (byte-identical).
2. Scale re-encode is REQUIRED but lossless, per-expert, cacheable — NOT a
   blocker, but the adapter MUST implement it (done in `reference.py` +
   `cpu_executor.hpp`) and MUST fail closed on `0xFF`.
3. Do NOT pass raw `F8_E8M0 u8*` where KT expects `bf16*` scales (that ABI is
   the MXFP8 path) — wrong-kernel silent corruption risk. Guarded by
   `config_.gate_scale != nullptr` check + adapter static asserts.
4. `group_size != 32` (e.g. NVFP4 group-16) is OUT OF SCOPE for this bridge —
   AMX rejects it; adapter fails closed.
