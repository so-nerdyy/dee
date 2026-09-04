# dee universality audit (source-grounded, no rewrite)

Scope: `dee.cpp/src`, `dee.cpp/include/dee`, `dee.cpp/main.cpp`,
`dee.cpp/scripts/deepseek_v4_*.py`, `dee.cpp/scripts/ornith_support.py`,
`dee.cpp/pydee`. Every claim below names exact files; line numbers are
accurate at base `9c16529` and may drift by a few lines as the tree moves.
Companion: `DSV4_ASSUMPTION_INVENTORY.md` + `dsv4_assumption_inventory.json`
(machine-readable list of hard-coded assumptions).

Classes: GENERIC (reusable verbatim) / MOSTLY_GENERIC (generic mechanism,
shaped defaults/names) / DSV4_SHAPED (works, but structure mirrors DSV4) /
DSV4_SPECIFIC (only correct for DeepSeek-V4-Flash-0731).

Headline: the systems core (mmap, byte caches, eviction, prefetch control
plane, telemetry) is already model-independent. The MoE execution stack is
SwiGLU-tri-projection generic with a DSV4 codec branch. Full-model frontend
(attention, residual, router math, tokenizer, MTP) is DSV4-specific or absent
by design — dee.cpp is today a routed-expert server + norm helper behind a
Python/HF dense frontend, not a full-model runtime.

## 1. ExpertStore / disk-backed expert source — MOSTLY_GENERIC

- `include/dee/expert_store.h`: `ExpertStore` ABC (`get(layer,expert)`,
  `materialize()`, `record_source_read_batch()`, `stats()`), `ExpertView`
  (`weights[3]` + `scales[3]` + `codec` + `record_index`), `SafetensorsExpertStore`,
  `Dee4ExpertStore` (fixed-stride expert-major file, `dee4-v2` dense /
  `dee4-v3-trace` sparse). Shaping: `ExpertCodec{Unknown, DeepSeekFp4E2m1E8m0}`
  single real value (`expert_store.h:21`), `[3]` projection arrays and the
  `gate/w1, up/w3, down/w2` + `[packed×3][scales×3]` comment (`:26-31`).
- `src/expert_store.cpp`: geometry (offsets, counts, `record_bytes`) is
  data-driven from `metadata.json`; hard gates are `codec !=
  "deepseek-fp4-e2m1-e8m0"` (`:325-328`), forced `view.codec =
  ExpertCodec::DeepSeekFp4E2m1E8m0` (`:187`, `:502`), `I8/F8` dtypes with a
  1-byte/element size check (`:367-375`, `:507-521`), 3-Kind resolve
  (`:191-198`), dense `record_index=(layer-start)*experts_per_layer+expert`
  (`:490-493`) vs sorted-trace `lower_bound` (`:416`, `:475-488`).
- Embedded assumptions: codec DSV4-specific; 3-projection/6-region record
  MoE-family (SwiGLU) as DSV4 instantiation; `mmap_memcpy` vs `pread`
  (`:555-578`) hardware/OS-specific.
- Generalize: easy–moderate. Add codec enum values + dtype/bytes-per-element
  table, `N`-projection weight/scale arrays in metadata, generic Kind→suffix
  map. The `pread` path already takes `record_bytes` and is generic.

## 2. Host cache — GENERIC

- `include/dee/host_pack_cache.h`, `src/host_pack_cache.cpp`: bounded LRU
  `uint64→bytes` with persistent fill-lane pool (`kMaxFillLanes=8`,
  `kMaxBatchRequests=256`), opaque `BatchFill{key,nbytes,fill,source_order}`
  seam, fail-closed batch commit/rollback. Zero tensor names, dtypes, shapes,
  codecs. DSV4 numbers (`~12.6 MB/entry`, six regions, `152.8 GiB`, `8 GiB`
  default budget) live in comments/tuning only. `source_order` sort is a
  generic positional-read optimization (hardware-specific) currently fed by
  DEE4 `record_index`.
- Reuse verbatim for any model; re-tune budget/lanes per box. Trivial.

## 3. VRAM cache — GENERIC

- `include/dee/vram_cache.h`, `src/vram_cache.cpp`: `Arena` first-fit +
  coalesce over opaque bytes, `VramCacheManager` LRU+Oracle
  `eviction_score()=last_used+priority*PRIORITY_WEIGHT` (`:196-199`),
  pin/generation forensics, identical `host`/`cuda` backends. Key
  `(layer,expert)` is MoE-family-trivial, not DSV4. Zero codec/geometry
  branches. Reuse verbatim; re-tune budget/weight. Trivial.

## 4. Admission / eviction — GENERIC

- Mechanism delegated to `VramCacheManager` (pin/unpin/touch/ensure) in
  `src/engine.cpp` (`stage_expert:3104-3204`, priority `=topk-k`,
  `staging_key=(layer<<32|expert)` at `:54-55`). Policy is LRU + pin +
  byte budget; geometry flows via `blob_elems_`/`cache_blob_bytes_`. No
  per-model fork needed.

## 5. Prefetch — DSV4_SHAPED (generic control plane, shaped data plane)

- `include/dee/async_prefetcher.h`, `src/async_prefetcher.cpp`:
  batch-dedup, inflight-coalesce, resident-hit pin, `ensure()+pin` cold path,
  `begin_batch/wait/wait_on_stream/mark_consumed`, staging ring (default 64),
  pinned accounting are generic. Shaping: `Transfer` hard-wires
  `quant_scales[3]`, `fp4_out/in/packed_offsets/scale_offsets[3]`,
  `fp4_region_src/nbytes[6]` in `[gate,up,down,scales]` order (`:36-53`,
  `:112-132`); every quantized path asserts `elements==3*projection_elements`
  or loops `p<3` (`async_prefetcher.cpp:228-230`, `:245-248`, `:266-295`,
  `:408-419`, `:713-722`); decode dispatches `fp4_e2m1_to_f16_cuda(packed,
  scale, out[p], in[p])` (`:757-789`).
- Generalize: moderate. Replace `[3]/[6]` with `N/2N` vectors +
  `ProjectionDesc{out,in,packed_off,scale_off,block}` + codec-enum dispatch.
  BF16/INT8/INT4 scalar-scale paths are easy; FP4 needs a generic
  `(N, block_N, scale_dtype)` descriptor and new `cuda_convert` kernels.

## 6. Async I/O — GENERIC

- `WeightMmap` (`weight_mmap.h/.cpp`): OS mmap, safetensors header parse,
  `lookup(name)`, `posix_fadvise(DONTNEED)`/`madvise` page drop — fully
  generic safetensors. `trace_alloc.h/.cpp`: CUDA lifetime forensics, no
  weight/codec knowledge. Reuse verbatim. Trivial.

## 7. Route journal — GENERIC (trace exists, journal emittable)

- No persistent route journal in C++ (grep `journal` → 0 hits outside
  `tmp/` scripts). Closest: `StageProfiler::note_prediction(token, logical,
  resolved, experts)` (`profiling.h:306-307`), `RequestTraceRecord{token,
  logical,resolved,expert,kind,bytes,transfer_id,...}` (`:120-148`). A
  journal can be emitted from existing hooks without touching compute. Easy.

## 8. Telemetry / profiling — GENERIC

- `include/dee/profiling.h` (`CpuStage/GpuStage/RequestKind/HostWait/...`),
  `src/profiling.cpp`: stage counters (`Gate/Up/Silu/Down/Combine/H2D/
  WeightConversion`) are MoE-generic; zero `shared/mtp/mla/fp4` keys. One
  DSV4-shaped comment only (`profiling.h:427-433`: `258 transfers/layer x 43
  layers` sizing note for `kMaxTimingEvents`). Reuse verbatim. Trivial.

## 9. Scheduling — GENERIC

- `src/engine.cpp` scheduling keys are `(token, logical_layer, resolved_layer,
  expert, bytes, transfer_id)`; `avail_layer` (`:2432-2437`) falls back to
  `base_layer` for single-layer synthetic shards (MOSTLY_GENERIC convenience).
  No model fork needed.

## 10. CPU/GPU placement — MOSTLY_GENERIC

- Buffers sized by runtime `hidden_/inter_/tokens/selections`
  (`engine.h:398-450`); `kPinnedStagingLimit=192 MiB` (`:556`) and
  `kThreads=256` are model-neutral tuning. Enums
  `DeviceCacheDType{Fp32,Fp16,Fp4E2m1}` (`:65-71`) and
  `WeightTransferDType{Bf16,Int8,Int4,Fp4E2m1}` (`:75-80`) gate placement;
  only the packed-verbatim-residency branch is DSV4-specific. Next codec =
  add enum + `stage_expert` branch + decode kernel. Medium.

## 11. Expert descriptor / layout — DSV4_SHAPED

- `ExpertView` + `Fp4Proj fp4[3]` (`engine.h:393-394`, `:452-474`) assume
  gate/up `[inter,hidden]`, down `[hidden,inter]`, block 32
  (`engine.cpp:3351-3354`, `:3355-3356` `(blob*17+31)/32`). The `[3]` is
  MoE-family (SwiGLU); the packed FP4 + e8m0 + `*2` width probe
  (`engine.cpp:3480-3504`, `inter=shape[0]`, `hidden=shape[1]*2` for DSV4) is
  DSV4-specific. See ABSTRACTION_DESIGN.md for the replacement descriptor.

## 12. Quant codec handling — DSV4_SPECIFIC (one real codec + test paths)

- Authoritative decode: e2m1 16-entry table
  (`weight_mmap.cpp:76-80`, `cuda_convert.cu:212-245`), `e8m0=2^(bits-127)`
  (`:88-95`), `packed I8[out,in//2]` low-nibble→col `2i`
  (`:103-113`), scale `[out,in//32]`, `in%64==0` guard
  (`cuda_convert.cu:252-288`). `DType::F8` conflates
  `F8_E4M3/E5M2/E8M0` (`weight_mmap.cpp:21-24`) — codec-specific bug-shaped
  for generalization. INT8/INT4 host-quantize paths
  (`cuda_convert.cu:36-59`, `engine.cpp:2554-2697`) are synthetic-workload
  test harnesses, not checkpoint codecs (MOSTLY_GENERIC, ignorable).
- See ABSTRACTION_DESIGN.md §Codec for the `decode_tile/storage_bytes/
  validate/metadata_layout/supported_backend` boundary. New codec = new
  kernel + descriptor entry; hard, but additive and isolated.

## 13. MoE expert execution — MOSTLY_GENERIC + DSV4 clamp knob

- CPU `Engine::swiglu` (`engine.cpp:240-254`) plain SiLU/FP32 — GENERIC.
  CUDA `swiglu_expert_cuda` family (`swiglu_cuda.cu:120-498`) takes
  `(inter,hidden,tokens,experts)` at runtime — GENERIC. `swiglu_limit` clamp
  (`swiglu_cuda.cu:16-59`, `engine.h:120-122`, default 0 = Ornith, 10.0 =
  DSV4) is already a clean DSV4_SPECIFIC-but-parameterized knob. Trivial.
- Combine is mean + unit-RMS stand-in (`engine.cpp:3307-3346`,
  `swiglu_cuda.cu:61-67` combine_kernel) — a mock-harness formula, wrong for
  any real checkpoint (needs `y=x+Σw·E(x)` + norm); the external
  `moe_forward_*` seam (`engine.h:241-298`, caller owns router+combine) is
  the correct GENERIC boundary. Keep the seam, replace the mock.

## 14. Router — MOSTLY_GENERIC shell, ORNITH-shaped built-in

- Built-in `route_topk_batch` (`engine.cpp:2231-2427`): hardcoded
  `model.language_model.layers.<L>.mlp.gate.weight` (`:2232`, `:2277`,
  ORNITH/Qwen3.5 naming), pure FP16-GEMM → FP32-softmax → ordered top-K →
  renorm. Shape-generic, name-shaped; math is family-standard but missing
  DSV4's `sqrtsoftplus/sigmoid/bias-split/hash-tid2eid/route_scale=1.5`
  (all in `scripts/deepseek_v4_layer_common.py:349-398`, never in C++).
  Real-model mode bypasses it (`is_no_op` + `moe_forward_experts`;
  `pydee` adapter owns router). Generalize with scoring-enum + name table.
  Easy–medium.
- Oracle predictor (`oracle.h/cpp`, `pt_loader.*`): per-layer
  `2048→256→256→256` MLP (`oracle.h:41,109`) for Ornith `E=256` —
  synthetic-control-path only, DSV4-irrelevant, leave alone.

## 15. Attention / state — absent by design (GENERIC by omission)

- Zero attention in C++: grep `attn|qkv|mla|kv_cache` in `src/engine.cpp` →
  0 hits (the single `attn` grep hit is `cudaGetDeviceProperties`);
  `Engine::generate` (`:3690-3740`) loops MoE-only `forward_layer`s.
  Attention/KV/RoPE/MLA live in Python HF (`pydee/adapter.py:8-10,163-169`:
  router/shared/attention/norm/residual/head stay HF). Nothing to
  un-hardcode; bringing MLA in-engine is a hard future project, not a
  refactor. Qwen-Next linear-attention state would likewise stay outside.

## 16. Residual topology — mock (MOSTLY_GENERIC, must be replaced per model)

- See §13 combine. The `+1`-convention RMSNorm offload
  (`rmsnorm_cuda.h:59-77`, `rmsnorm_cuda.cu:44-316`, `engine.h:300-350`) is
  Qwen/Ornith-family-shaped; DeepSeek plain `y=x_norm*weight` is not
  implemented — add a `norm_formula` flag (easy–medium).

## 17. Tokenizer / model frontend — Python-side, DSV4_SPECIFIC

- C++/CLI has no tokenizer (`--tokens` is a count). `scripts/
  deepseek_v4_encoding.py`: pinned tokenizer SHAs, vocab 128000 vs config
  129280, chat template, thinking-mode flags — all DSV4_SPECIFIC, correctly
  isolated in one script. Qwen needs a sibling script, not a refactor.

## 18. MTP / NextN — excluded by documented assumption (DSV4_SPECIFIC gap)

- `mtp.*→dspark`, `cache=speculation_only/active=false`
  (`deepseek_v4_support.py:121-128,365-374`); `weight_mmap.h:106` notes
  `mtp.<I>.* (layers 40-42)`; greedy-text path excludes MTP. Qwen-Next MTP
  layout differs; needs a new namespace + target-layer map when speculative
  decode is wanted. Medium, and not on the critical path (base decode first).

## 19. Checkpoint loader — GENERIC core, dialect namers

- `WeightMmap::open/parse_header/lookup/register_shard` GENERIC;
  `TensorResolver{ORNITH,DEEPSEEK_V4}` (`weight_mmap.h:95-151`,
  `weight_mmap.cpp:303-370`) is already an adapter seam (MOSTLY_GENERIC):
  `model.language_model...{gate,up,down}_proj` vs
  `layers.%d.ffn.experts.%d.{w1|w3|w2}.{weight|scale}` + shared +
  `GATE→w1,UP→w3,DOWN→w2`. `engine.cpp:3452` conflates codec with model
  (`transfer==Fp4E2m1 → DEEPSEEK_V4`) — the one leak to fix with an explicit
  `model` knob (easy). `scripts/ornith_support.py` is a second hardcoded
  frontend (ORNITH_SPECIFIC), not a generic base — Qwen needs a third
  dialect, not Ornith reuse.

## 20. Model configuration parsing — missing in C++, hardcoded in Python

- C++ parses no `config.json`: `EngineConfig` (`engine.h:84-123`) knobs only
  (`topk=8/layers=40/hidden=2048/inter=256/num_experts→256` defaults are
  Ornith-shaped; DSV4 needs 6/43/4096/2048/256). Shard discovery overrides
  (`engine.cpp:3407-3408`, `:3503-3512` fail-closed) make the mechanism
  MOSTLY_GENERIC. Missing knobs: vocab, norm eps/formula, rope, scoring,
  route_scale, hash/compress, MTP, tokenizer. Python `deepseek_v4_model.py:
  288-367 ModelConfig` copies DSV4 defaults (DSV4_SPECIFIC, easy to add a
  Qwen dataclass); Ornith `AutoConfig→EngineConfig` (`run_ornith_generation.
  py:1753-1858`) is the reusable pattern.

## Scorecard basis

Systems core (items 2,3,4,6,7,8,9): estimated in
`research/universality/SCORECARD.md` (Phase G), which cites the item numbers
above as evidence. Overall numbers are restated with the 10 required answers
in `FINAL_SUMMARY.md`. No percentages are asserted without the per-item
source evidence in §§1–20.
