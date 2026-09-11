# KT CPU Audit — KTransformers CPU Expert Execution for dee

Branch: `research/kt-cpu-bridge` (parallel track, does not touch `freebuff/deepseek-v4-flash-0731-t4`).
Upstream: https://github.com/kvcache-ai/ktransformers
Pinned upstream commit audited: `31985f40bcc40da08107efdb1f81bf88cb38c6b2` (2026-09-01).
Upstream license: Apache-2.0 (`LICENSE` at repo root). See `research/kt-cpu-bridge/THIRD_PARTY_KTRANSFORMERS.md`.

Scope: map ONLY the CPU-expert execution machinery dee might reuse. dee keeps
ownership of SSD source of truth, bounded RAM/VRAM caches, admission, eviction,
prediction, scheduling, and exact model semantics. KTransformers must supply only
CPU execution kernels + placement/async primitives.

Target dee planner (unchanged by this doc):

```
requested expert
  VRAM resident -> GPU execution
  RAM resident  -> GPU transfer + exec  OR  CPU exec
  SSD only      -> async read -> admission -> RAM / GPU / transient exec
```

Method: sparse checkout (`kt-kernel/operators`, `kt-kernel/python`, `doc`) +
`rg` over `ext_bindings.cpp`, `operators/{common,moe-tp}.hpp`,
`operators/amx/*moe*.hpp`, `operators/avx2/*moe*.hpp`,
`operators/llamafile/moe.hpp`, `operators/moe_kernel/moe.hpp`,
`kt-kernel/python/{experts,experts_base,_cpu_detect}.py`,
`kt-kernel/python/utils/{amx,loader}.py`, `doc/en/DeepSeek-V4-Flash.md`,
PR #1970 (V4-Flash MXFP4), release v0.6.2 notes. Where line numbers are cited
they refer to the pinned commit above.

`kt-kernel/cpu_backend/{cpuinfer.h,worker_pool.h}` is NOT in the sparse
checkout; CPUInfer lifecycle/sync below is inferred from the pybind surface
(`ext_bindings.cpp`) and the Python call sites (`experts_base.py`), plus the
C++ consumer (`operators/*`). This gap is flagged per component.

---

## 1. CPUInfer lifecycle

| Field | Value |
|---|---|
| Upstream file | `kt-kernel/ext_bindings.cpp:641-652` (`class CPUInfer` pybind), `kt-kernel/python/experts_base.py:296-343` (`_MoEBase._get_cpu_infer`), `:458` (per-layer attach), `kt-kernel/python/experts.py:123-162` (factory) |
| Class/function | `CPUInfer{submit, sync(allow_n_pending=0), submit_with_cuda_stream, sync_with_cuda_stream, backend_}`; singleton `_MoEBase._cpu_infer_instance`; `WorkerPoolConfig{subpool_count, subpool_numa_map, subpool_thread_count}` |
| Required inputs | `cpuinfer_threads` (total CPU threads), `threadpool_count` (NUMA shards / subpools), `numa_nodes` (len must equal `threadpool_count`, else `ValueError`, default `list(range(tp))`). Per-forward tasks are `(fn-ptr, args-ptr)` pairs from `moe.{warm_up,load_weights,forward}_task` (`ext_bindings.cpp:183-256,459-475`). Forward args: `qlen_ptr:int32*`, `k:int`, `expert_ids:int64*`, `weights:fp32*`, `input:bf16*`, `output:bf16*`, `incremental:bool` (`operators/moe-tp.hpp:195-199`, `ext_bindings.cpp:239-242`). |
| Output format | No tensor returned. Side effect: enqueued C++ `TP_MOE::forward_binding(...)` on the worker pool. |
| Precision | N/A (orchestration only; compute precision is backend-defined, §5). |
| Memory ownership | `CPUInfer` is process-global singleton; first wrapper's `threads/tp_count` wins, later values ignored. `GeneralMOEConfig.pool = cpu_infer.backend_` (`python/utils/amx.py:377,492`, `python/llamafile.py:212`) — C++ holds raw `WorkerPool*`, Python holds `CPUInfer`. `atexit shutdown_ascend_callback_worker` only on NPU (`experts_base.py:107-110`); no CPU shutdown/refcount path observed. |
| Synchronization | `submit(task)` async enqueue; `sync(allow_n_pending)` host drain; `sync_with_cuda_stream(stream, allow)` CUDA-ordered drain; NPU: `bypass -> sync`, else `subscribe_ascend_stream + _wait_device + sync` (`experts_base.py:835-879`). During CUDA graph capture (`torch.{cuda,npu}.is_current_stream_capturing()` / sglang capture-mode guard, `:113-159`) sync is skipped. `KTRANSFORMERS_CPU_ONLY` build removes stream methods (`ext_bindings.cpp:647-651`). `KT_FORCE_SYNC_SUBMIT=1` or degraded ACL worker forces `submit+sync` (`experts_base.py:73-82`). |
| Licensing | `python/experts.py:3`, `experts_base.py:2`, `python/__init__.py:2` carry SPDX Apache-2.0. `ext_bindings.cpp:1-9` is `Copyright KVCache.AI, All Rights Reserved` with no Apache header — preserve notice, do not strip. |
| Portability | CPU side ISA-independent; CUDA/NPU overlap needs CUDA or ACL build. Needs a live CUDA stream int for `submit_with_cuda_stream`; CPU-only build cannot use it. |

dee conflict: singleton + first-wins thread/NUMA config means dee cannot
create per-request or per-layer pools. `buffer_depth=2, slot=layer_idx%2`
(§9) further assumes strictly sequential layer visits — concurrent forwards
would serialize or corrupt slots.

## 2. MOEConfig (`GeneralMOEConfig`)

| Field | Value |
|---|---|
| Upstream file | `kt-kernel/operators/common.hpp:230-336` (struct), `kt-kernel/ext_bindings.cpp:833-918` (pybind) |
| Class/function | `GeneralMOEConfig{expert_num, routed_expert_num(num_experts_per_tok), hidden_size, intermediate_size, layer_idx, pool, max_len, gate/up/down[_proj(s),_scale(s)] ptrs, quant_config{bits,group_size,zero_point,per_channel}, swiglu_{limit,alpha}, num_gpu_experts / gpu_experts_mask_ptr, physical_to_logical_map, ...}`; helpers `compute_num_gpu_experts():246-253`, `should_skip_expert(id):256-258`, `max_possible_qlen():335` |
| Required inputs | Ctor `(expert_num, routed_expert_num, hidden_size, intermediate_size)` + optional `num_gpu_experts:839` or `gpu_experts_mask_ptr:843-849`. Python always fills `layer_idx, pool=backend_, max_len=chunked_prefill_size, {gate,up,down} ptrs, quant_config, swiglu_limit/alpha`, e.g. `python/utils/amx.py:369-390,484-526,821-856`. Llamafile additionally sets `m_block=32, group_min_len=10, group_max_len=max_len, gate/up/down/hidden_type` (`python/llamafile.py:215-247`). |
| Output format | Plain struct (POD + borrowed pointers). No compute. |
| Precision | Carries only pointers + `ggml_type` enums; compute precision is backend-defined. |
| Memory ownership | ALL `void*` weight/scale/mask/map pointers are borrowed — the Python `torch`/`numpy`/mmap owner must outlive `load_weights + forward`. `gate_projs:[tp][expert]` 2-D vectors via `DEF_PTR_2D_PROPERTY` (`ext_bindings.cpp:176-181,881-898`). `physical_to_logical_map:void*` set from `load_weights_task(ptr)` (`:211-224`, `common.hpp:42-46`). |
| Synchronization | None (POD). |
| Licensing | `common.hpp` has no license header; root `LICENSE` (Apache-2.0) still applies. |
| Portability | ISA-independent. `swiglu_limit/alpha:float=0.0` consumed only by vector `act_fn` (§7). |

dee conflict: none structurally — dee can construct an equivalent descriptor
per expert from its own bounded record. Must NOT adopt the borrowed-pointer
lifetime model; dee's adapter copies or pins per-expert bytes explicitly (§5,
`CPU_EXECUTOR_DESIGN.md`).

## 3. DeepSeek-V4-Flash MXFP4 expert loading

| Field | Value |
|---|---|
| Upstream file | `kt-kernel/python/utils/loader.py` V4 layout branch (`layers.{L}.ffn.experts.{i}.{w1,w3,w2}.{weight,scale}`), `kt-kernel/python/utils/amx.py` `MXFP4` dispatch (`NativeMoEWrapper`), `kt-kernel/operators/amx/fp4-moe.hpp:808-856,1114+` (`AMX_FP4_MOE_TP::load_weights`), `kt-kernel/examples/test_fp4_moe_v4.py`, `kt-kernel/bench/bench_fp4_moe.py` (from PR #1970) |
| Class/function | `loader` V4 branch; `convert_or_copy(bf16 scales -> fp32)` + `finalize_scale_e8()`; `from_raw_mat` weight memcpy; `write_weights_to_buffer(...)` (CPU->GPU expert move-back) |
| Required inputs | Per expert: packed `I8 [N, K/2]` weights + `bf16 [N, K/32]` scales (loader produces bf16 scales from checkpoint `ue8m0` via lossless `(u8<<7).view(bf16)` bit-cast — no `uint16` lshift path, which corrupted one group on some torch builds per PR #1970 commit `25eebb4`). `PROJ_NAMES=("w1","w3","w2")=(gate,up,down)` (`loader.py:1214`, `amx.py:877-878`). `group_size = hidden // scale.shape[1]` must be 32 for the AMX natural path (`amx.py:879-881`). `config_.gate_scale != nullptr` else `throw "only support load native weight"` (`fp4-moe.hpp:808`). |
| Output format | C++ NUMA-sharded `BufferB.b` (packed weights, row-major) + `BufferB.d` (fp32 scales, compacted in place to 1-byte `scale_e8` after validation). |
| Precision | Bit-exact for real checkpoints (proof in `FORMAT_COMPATIBILITY.md`). |
| Memory ownership | Native loader freed per-layer after `sync` (`amx.py:986-1006`) but C++ NUMA copies stay resident. Full-pool residency assumed (see conflicts). |
| Synchronization | `load_weights` fans out over `pool->do_work_stealing_job(nth * expert_num, ...)` for weights then `expert_num` jobs for scales; caller must `sync` before forward. |
| Licensing | Apache-2.0 (Python files carry SPDX headers; `fp4-moe.hpp` carries KVCache.AI copyright header — preserve). |
| Portability | Needs `__AVX512BF16__` for the fast natural path; AVX2 fallback exists (`operators/avx2/mxfp4-moe.hpp`) with same layout, `fp32` scales. |

dee conflict: loader assumes it can walk ALL experts' safetensors and keep the
C++ copies resident. dee must replace this with per-expert on-demand fill from
its bounded `HostPackCache` (packed bytes + `e8m0` bytes) + explicit
`ue8m0 -> bf16 -> fp32 -> compact-e8` step inside the adapter (costed in
`FORMAT_COMPATIBILITY.md`). Weight bytes themselves are zero-copy `memcpy`.

## 4. Expert weight representation (resident)

| Field | Value |
|---|---|
| Upstream file | `kt-kernel/operators/amx/fp4-moe.hpp:243-303` (`BufferB`), `kt-kernel/operators/amx/la/amx_buffers.hpp:1114-1138` (`BufferBInt4KGroupImpl::from_raw_mat/get_submat/get_scale`), `kt-kernel/operators/avx2/mxfp4-moe.hpp:60-158` |
| Class/function | `BufferB{n, k, k_group_size=32, b:uint8*, d:float*, scale_e8:uint8*, scale_e8_valid:bool}`; `required_size(n,k,gs) = n*k/2 + n*k/gs` bytes (64 B aligned) |
| Required inputs | `(n, k, group_size)` + backing allocation from `shared_mem_buffer_numa`. |
| Output format | Row-major `[N, K/2]` nibble-packed weights (`low nibble = col 2i`, `high = col 2i+1`, x86-LE) + per-32-group scales (fp32, or compacted 1-byte exponents after `finalize_scale_e8`). No transpose/tiling on the MXFP4 path (the blocked-tiled variant is RAWINT4-only; MXFP4 uses `N_STEP=32, K_STEP=32`, `fp4-moe.hpp:31-33`). |
| Precision | E2M1 codepoints `{0,±0.5,±1,±1.5,±2,±3,±4,±6}` incl. `-0.0` quirk, identical to dee's `FP4_TABLE` (proof in `FORMAT_COMPATIBILITY.md`). |
| Memory ownership | Owned by C++ NUMA buffer, keyed on `this`; flat (`b` + `scale_e8`, no pointers) so evict/reload is `memcpy`. |
| Synchronization | None after `finalize_scale_e8`; safe for concurrent forward reads within one pool. |
| Licensing / portability | As §3. `finalize_scale_e8` validates positive-pow2 (`sign=0, mant=0, exp not in {0,0xFF}`) with an exact fp32 fallback for non-E8M0 input. |

dee mapping: `dee::ktbridge::PackedExpertView` (bounded record: 3× packed +
3× `e8m0` views + shapes) -> adapter-owned `KtExpertBlob` (flat `b` +
`scale_e8`). Post-`finalize` bytes are cacheable in dee's bounded host cache
to skip torch conversion on hit.

## 5. CPU expert execution (GEMM)

| Field | Value |
|---|---|
| Upstream file | `operators/moe-tp.hpp:185-246` (`TP_MOE_Common::forward`), `operators/amx/moe_base.hpp:179-453,455-653` (`AMX_MOE_BASE::{forward,forward_prefill,forward_decode}`), `operators/amx/fp4-moe.hpp:350-700` (`fp4_mat_vec/mat_mat_kgroup`, natural fast path), `operators/avx2/moe_base.hpp:545-593`, `operators/llamafile/moe.hpp:412-537,625-929`, `operators/moe_kernel/moe.hpp:414-669` |
| Class/function | `AMX_FP4_MOE_TP : AMX_MOE_BASE` (CRTP); backends `AMX_*_MOE_TP`, `AVX2_*`, `Llamafile`, `moe_kernel`. Dispatch `do_gate_up_gemm(do_up, expert, ith, nth, qlen)` / `do_down_gemm(...)`: `qlen > 4*expert_num/num_experts_per_tok` -> `mat_mul_kgroup` (4×4 register tile, `M_TILE=4,N_TILE=4`), else `vec_mul_kgroup` (N-dim 4-rows-at-once + horizontal reduce). `m==1 && group==32` + natural-order activation -> `fp4_mat_vec_kgroup_natural` (`__AVX512BF16__` only). |
| Required inputs | Per token: `qlen:int, k=topk:int, expert_ids:int64[qlen*k], weights:fp32[qlen*k]` (router softmax done OUTSIDE, on GPU/SGLang). Scratch sized by `max_len × k × H/I` (`amx/moe_base.hpp:106-114`): `m_local_{input,gate,up,down}` via `shared_mem_buffer_numa.alloc(tp_part_idx)` (`:161`), freed (`:164`); TP outputs `local_output_numa[tp]: fp32[max_possible_qlen*H]` (`moe-tp.hpp:156-166`). |
| Output format | Per-part `fp32`, merged to `bf16` output (`merge_results`, §6). |
| Precision | `BF16 in -> E2M1->BF16 LUT (PSHUFB or word-permute natural) -> _mm512_dpbf16_ps -> fp32 dot -> fp32 × scale -> fp32 acc -> reduce -> BufferC fp32 -> fp32->bf16` (`fp4-moe.hpp:165-173,350-355`, `amx_buffers.hpp:1776-1793`). AVX2 same with `fmadd_ps` (`mxfp4-moe.hpp:189-217`). Gate/up outputs are bf16; SwiGLU runs in fp32 and rounds back to bf16 (`moe_base.hpp:705-737`, natural variant `fp4-moe.hpp:771-799`). |
| Memory ownership | C++ NUMA-local scratch owned per `TP_MOE` instance; user I/O buffers borrowed (pinned CPU, §9). |
| Synchronization | `TP::forward` fans out `do_numa_job` then inline `merge_results` on caller thread (`moe-tp.hpp:212-216`). Inner GEMM/act steps use `get_subpool(tp)->do_work_stealing_job`; `qlen<10` runs serially (`moe_base.hpp:730-736,791-798`). |
| Licensing | Headers carry KVCache.AI copyright; root Apache-2.0 applies. |
| Portability | AMX path needs `__AVX512BF16__` (+ `F/BW/VBMI/VNNI` per `CMakeLists.txt:281-350`); AVX2 fallback covers `BF16/FP8/GPTQ/RawInt4/MXFP4/MXFP8` (`ext_bindings.cpp:986-997`) but NOT `FP8_PERCHANNEL` / K-group AMX. See §12. |

dee reuse verdict: the `fp4_mat_vec_kgroup[_natural]` + `fp4_mat_mat_kgroup`
kernels are the reusable core. dee must NOT import `TP_MOE` wholesale (it
assumes full-pool residency + borrowed-pointer lifetimes); instead the adapter
instantiates ONE expert's `BufferB` on demand and calls the vec/mat entry
points (design in `CPU_EXECUTOR_DESIGN.md`).

## 6. Routing-weight application

| Field | Value |
|---|---|
| Upstream file | `operators/amx/moe_base.hpp:413-436` (prefill), `:620-638` (decode); mirrors in `avx2/moe_base.hpp`, `moe_kernel/moe.hpp:629-660`, `llamafile/moe.hpp:588,881-887` |
| Class/function | Inline in forward: `x=0; for j in k: if !should_skip(id): w=broadcast(weights[i*k+j]); x += fmadd(down_out[expert][pos], w, x)` — **late weighting after `down_proj`, in fp32, before TP-merge**. Skipped/GPU experts contribute 0 (NOT renormalized in C++). |
| Required inputs | `weights:fp32[qlen*k]` from the external router. |
| Output format | fp32 partial per (token) row. |
| Precision | fp32 multiply-add into the accumulator. |
| Memory ownership / sync | Accumulator is the per-TP `local_output_numa` scratch; no extra allocation. |
| Licensing / portability | ISA-independent logic; identical across backends. |

dee note: dee's official reference applies the scalar to the intermediate
BEFORE `w2` (`h = w*h; h@W2`, `scripts/deepseek_v4_moe_reference.py:108-110`);
KT applies AFTER `w2` in fp32. Mathematically identical
(`(w·h)Wᵀ = w·(hWᵀ)` — dee's own test asserts this) up to KT's intermediate
bf16 round-trips. The adapter MUST document which placement it implements and
prove equivalence in `test_kt_bridge_correctness.py` (§8, Phase D).

`merge_results(incremental)` (`amx/moe_base.hpp:760-804`, `avx2:615-`,
`llamafile:956-1011`): `incremental==false` -> sum TP shards + `fp32->bf16`
to `output`; `==true` -> first add existing `output (bf16->fp32)` into
`local_output_numa[0]` (deferred-expert accumulation across the two submits in
`experts_base.py:653-671,800-817`). Base `TP_MOE_Common:252-258` throws if
unimplemented. dee's single-expert adapter does not need `incremental`; the
caller combines per-expert outputs with routing weights exactly as today.

## 7. DeepSeek clamp / SwiGLU semantics

| Field | Value |
|---|---|
| Upstream file | `operators/amx/la/amx.hpp:47-101` (`act_fn`), `operators/avx2/avx2_bf16_utils.hpp:118-169`, `operators/avx2/moe_base.hpp:551-581`, `operators/amx/moe_base.hpp:705-722`, `operators/amx/fp4-moe.hpp:788-789`, `operators/llamafile/moe.hpp:404-410,521,811`; plumbing `python/experts.py:153-161,378-390`, `python/utils/amx.py` (`NativeMoEWrapper`), `operators/common.hpp:311-320,325`, `ext_bindings.cpp:915-916` |
| Class/function | `act_fn(gate_fp32, up_fp32, limit, alpha)` |
| Required inputs | `gate/up fp32 vectors`, `swiglu_limit:float` (default 0.0 = disabled, single `cmp+jmp` per 32-lane tile), `swiglu_alpha:float` (default 0.0). |
| Output format | fp32 activation vector, rounded back to bf16 for the down-projection input. |
| Precision | fp32 clamp + `silu(g)*u`; scalar tails mirror the vector path. |
| Memory ownership / sync | Register-only; no allocation, no sync. |
| Licensing / portability | ISA-independent semantics; vectorized per ISA. |

Semantics (must match dee exactly):

- `alpha == 0` (standard SiLU, DeepSeek-V4-Flash): `gate = min(gate, limit)` **one-sided**, `up = clamp(up, ±limit)` symmetric; then `silu(gate)*up`. Matches `trtllm gemm1_clamp_limit` / `deep_gemm _apply_swiglu_limit` comments (`common.hpp:319-325`, `amx.hpp:47-56`). dee's CUDA (`src/swiglu_cuda.cu:22-24`) and Python reference (`scripts/deepseek_v4_expert_reference.py:124-126`, `:80-82` moe ref) implement the SAME asymmetric clamp with `limit=10.0`. No divergence.
- `alpha > 0` (MiniMax M3 `swigluoai`): symmetric clamp on both, then `gate*sigmoid(gate*alpha)*(up+1)` (`amx.hpp:78-99`). Irrelevant for V4-Flash; adapter rejects `alpha != 0`.
- Plumbing guard: `experts.py:153-161,378-390` only forwards `swiglu_limit/alpha` for `method in (FP8,MXFP4,MXFP8,LLAMAFILE)` else raises; `NativeMoEWrapper` re-guards (`:581`, `:838-848`); SFT rejects non-zero (`:236-243`). Llamafile scalar `act_fn(g,u,limit)` has NO alpha (`moe.hpp:404`) — `swiglu_alpha` plumbed but unused there. dee adapter takes `swiglu_limit` explicitly per call (default 10.0 for V4).

## 8. Asynchronous `submit_with_cuda_stream`

| Field | Value |
|---|---|
| Upstream file | `kt-kernel/python/experts_base.py:730-817` (`submit_forward`), `:600-671` (`forward_on_pinned_buffers`), `:673-728` (`run_pinned_forward_sync`), `kt-kernel/ext_bindings.cpp:479-539` (`forward_task` 6-arg and 7-arg `incremental` overloads `:467-472`) |
| Class/function | `cpu_infer.submit_with_cuda_stream(stream:int, moe.forward_task(bsz,k,ids,weights,input,output,incremental))` — args are raw `data_ptr()` ints (`:624-632`). |
| Required inputs | Live CUDA stream handle (int), pinned I/O buffers (§9), `allow_pending` flag. |
| Output format | None (async); completion observed via `sync_with_cuda_stream`. |
| Precision | N/A. |
| Memory ownership | Caller-owned pinned buffers must outlive the drain. |
| Synchronization | CUDA path is stream-ordered. NPU path deliberately AVOIDS it: `sync_submit = bypass or device==npu` (`:768`), then `_wait_device + submit + subscribe_ascend_stream` (`:769-798`) because `aclrtLaunchCallback` firing is host-unobservable (`NO_BLOCK`) and a later drain can read stale `output_cpu` (`:636-644,761-767`). |
| Licensing / portability | Needs CUDA (or ACL) build; `KTRANSFORMERS_CPU_ONLY` removes it. |

dee reuse: this is the pattern dee wants for `RAM resident -> CPU exec`
overlapping GPU work — submit CPU experts on the pool while the GPU stream
runs VRAM-resident experts, then `sync_with_cuda_stream` before combine. The
adapter's Phase-1 implementation is synchronous (correctness first); the async
submit/sync pair is exposed as an interface stub with ownership rules so a
later phase can bind the real `CPUInfer` without changing callers
(`CPU_EXECUTOR_DESIGN.md` §6).

## 9. CPU output delivery back to CUDA

| Field | Value |
|---|---|
| Upstream file | `kt-kernel/python/experts_base.py:216-283` (`KExpertsCPUBuffer.get_buffer`), `:542-589` (`_prepare_forward_cpu_buffers`), `:819-879` (`copy_forward_output_to_device/sync_forward`) |
| Class/function | Double-buffered pinned staging + explicit H2D of the result. |
| Required inputs | `B = qlen` tokens, `H` hidden, `k` topk, `layer_idx` (slot = `layer_idx % 2`, `buffer_depth=2`), `hidden.device` for `output_gpu`. |
| Output format | `output_gpu:[B,H]@hidden.device` (bf16), returned to the SGLang graph. |
| Precision | bf16 end-to-end on the wire; fp32 internally (§5). |
| Memory ownership | ALL `pin_memory=True`: `input_cpu:bf16[B,H]`, `imm_ids:int64[B,k]`, `def_ids:int64[B,k](-1)`, `weights:fp32[B,k]`, `output_cpu:bf16[B,H]`, `bsz:int32[1]`, `output_gpu:[B,H]`. `capture_bs/capture_buffers` pin per-B buffers forever until `clear_buffer_cache` (`:223-283,903-939`). D2H `copy_(non_blocking=True)` (`:574-578`); NPU-sync-submit drains `_wait_device` first. `sync_forward`: CUDA -> `sync_with_cuda_stream(stream, allow_pending)`; NPU -> `_wait_device+sync`; bypass -> `sync` (`:859-877`). Then `output_gpu[slot].copy_(output_cpu[slot], non_blocking=True)` (`:832`). `allow_pending=1` iff this layer has deferred experts (`:858,727`). |
| Synchronization | Stream-ordered drain then async H2D; deferred experts use the `next_slot` + `incremental=true` second submit. |
| Licensing / portability | Needs CUDA for the device side; CPU-only build keeps `output_cpu` only. |

dee conflict: buffer cache assumes bounded `B <= chunked_prefill_size`
(`qlen > max -> ValueError`, `:522-540`, mirrors C++ `max_possible_qlen`
sizing) and sequential layer visits. dee's adapter copies ONE expert output
per call into caller-owned memory and never retains buffers across calls.

## 10. Thread-pool ownership

Singleton `_MoEBase._cpu_infer_instance` (`experts_base.py:297`);
`WorkerPoolConfig` built once (`:323-341`); `CPUInfer(worker_config)` (`:341`).
No shutdown/refcount; layers share. `GeneralMOEConfig.pool` is a raw
`WorkerPool*` (`common.hpp:238`). C++ splits via
`pool->dispense_backend()->do_numa_job` (construction `moe-tp.hpp:136-142`,
forward `:212`, `load_weights` e.g. `amx/moe.hpp:432-443`) and
`pool->get_subpool(tp)->do_work_stealing_job` (GEMM/act/merge). A bare
`WorkerPool(int)` ctor is also bound (`ext_bindings.cpp:634`) but the Python
wrapper never uses it.

dee must NOT share this singleton in production: one global pool + global
`sync(allow_pending)` + class-global `_layer_has_pending_deferred[layer_idx]`
(`experts_base.py:378,653,800`) means per-request pools or cross-layer
concurrent forwards would serialize/corrupt slots. The adapter owns NO threads
in Phase 1 (caller-thread synchronous); a future `KtThreadPool` binding must
be per-engine, not global.

## 11. NUMA assumptions

`subpool_count == tp_count == NUMA shards` (`moe-tp.hpp:67`);
`intermediate_size % tp_count == 0` else throw (`:68-73,120-125`); llamafile
instead requires `intermediate % 256 == 0` and QK_K-block split (`:81-117`,
`python/llamafile.py:91-129`). Default map sequential (`experts_base.py:332`);
explicit `numa_nodes` allowed. Weights: AMX pre-quant sharded
`[numa][expert]` (`SafeTensorLoader.load_experts:215-326`, keys
`ffn_{up,gate,down}_exps.{e}.numa.{n}.{weight,scale}`); native
`[[ptr per expert]]` single-NUMA dim (`amx.py:806-818`); GGUF mmap zero-copy
viewed via `np.frombuffer -> torch.from_numpy` (`loader.py:1067-1078`), kept
alive by `weights_to_keep` until `sync` (`llamafile.py:199,256`).
`shared_mem_buffer[_numa].alloc(tp_part_idx, this, ...)` pins per-TP scratch
on its node.

dee conflict: NUMA topology is fixed at first init; dynamic thread/NUMA
reassignment or non-divisible sharding throws at construction. dee's T4/Kaggle
targets are typically single-NUMA or `numactl --interleave=all` (per the
V4-Flash tutorial launch line); the adapter therefore defaults to
single-shard (`tp_count=1`) and documents NUMA as a bring-up knob, not a
correctness dependency.

## 12. AVX2 fallback

| Field | Value |
|---|---|
| Upstream file | `kt-kernel/operators/avx2/moe_base.hpp`, `kt-kernel/operators/avx2/mxfp4-moe.hpp:60-217`, `kt-kernel/operators/avx2/bf16-moe.hpp`, `kt-kernel/ext_bindings.cpp:986-997`, `kt-kernel/python/utils/amx.py:67-246` (`KT_MXFP4/BACKEND` env), `doc/en/kt-kernel/AVX2-Tutorial.md` |
| Class/function | `AVX2_*_MOE` family; `AVX2MXFP4` group-16 forced for NVFP4 (`amx.py:911-917`); AVX-VNNI-256 auto-selected if `avx_vnni` + `group%32==0`, `<=256` (`:96-177`). |
| Required inputs | Same `GeneralMOEConfig` + packed weights / fp32 scales; no BF16-dot instruction needed (`fmadd_ps` path, `mxfp4-moe.hpp:189-217`). Same packing/LUT/row-major `memcpy` conclusion as AMX (§4). |
| Output format / precision | Same `bf16 -> fp32 -> bf16` contract; more `fmadd` traffic, no `_mm512_dpbf16_ps`. |
| Memory ownership / sync | Same pool/NUMA model as AMX. |
| Licensing / portability | Builds with `-mavx2 -mfma` (`CMakeLists.txt:281-350`, MSVC `/arch:AVX2`); runs on consumer CPUs without AVX-512/AMX (v0.6.1+ AVX2-only backend, v0.6.2 AVX2/AVX-VNNI RAWINT4). |

dee relevance: HIGH. The 2×T4 campaign host CPU is not guaranteed to have
AVX512-BF16/AMX; the adapter MUST compile and pass correctness on AVX2 (the
reference implementation in `dee.cpp/experiments/kt_cpu_bridge` is
ISA-independent C++/Python and therefore the AVX2-equivalent baseline). Perf
porting to AMX (`_mm512_dpbf16_ps` + natural-order permute + `expand_e8_scales`
hoist + 4×4 tile) is a later optimization, not a correctness gate.

## 13. AVX512 / AMX paths

Requirements asserted in Python: AMX INT4/INT8 needs `AVX512F+BW`
(`amx.py:295-306`); FP8 needs `AVX512F+BW+BF16+VBMI` or AVX2+FMA (`:598-604`);
BF16 AMX needs `AVX512+BF16` else AVX2 (`:611-617`); MXFP4 AMX needs
`AVX512+BF16` else AVX2 (`:633-639`); SYCL needs `CPUINFER_USE_SYCL+icpx` +
render-node perms (`:82-93,624-628`). ISA dispatch hierarchy in
`python/_cpu_detect.py`: `amx > avx512_bf16 > avx512_vbmi > avx512_vnni >
avx512_base > avx2` via `/proc/cpuinfo` or `cpufeature`, override
`KT_KERNEL_CPU_VARIANT`, fallback chain in `load_extension`
(`__cpu_variant__/__int8_kernel__/__fp8_kernel__`, `ext_bindings.cpp:543-568`).
Build flags `kt-kernel/CMakeLists.txt:281-350` (`-mavx512f/bw/vbmi/vnni/bf16`,
`-mavx2 -mfma`, MSVC `/arch:AVX512/AVX2`); presets `CMakePresets.json:14-52`.
All `AMX*_MOE` classes gated `__x86_64__ && USE_AMX_AVX_KERNEL` +
`__AVX512F/BF16/VBMI` (`ext_bindings.cpp:47-59,945-984`); without AMX the
classes silently vanish (`getattr(..., None)` in `amx.py:26-62`).

dee rule: never `assert(AMX)` on the hot path. Detect at init, log the variant,
run the portable path. The audit's perf-critical AMX details to port LATER
(only): `BufferA::from_mat_natural` + `permute_activation_group` (hoist the
16-bit interleave out of every N-row worker), `expand_e8_scales` (vector
`e<<23` before the weight loop), `fast_memcpy` (64 B chunks,
`fp4-moe.hpp:873+`), `fp4_mat_mat_kgroup` 4×4 tile sharing one 4-row decode
across 4 tokens.

## 14. Memory ownership (summary)

| Region | Owner | Lifetime | dee rule |
|---|---|---|---|
| Checkpoint safetensors / GGUF mmap | Python `torch`/`numpy`/`np.frombuffer` view | Must outlive `load_weights + forward`; `weights_to_keep` until `sync` (llamafile) / freed per-layer after `sync` (AMX native) | dee's `WeightMmap` + `HostPackCache` own it; adapter only borrows `PackedExpertView` for the call duration |
| C++ NUMA weight copies (`BufferB.b/d`) | `shared_mem_buffer[_numa]` keyed on `this` | Resident until `TP_MOE` destruction; no eviction | Adapter owns ONE expert's flat blob per call (or per bounded cache entry); no global residency |
| Per-forward scratch (`m_local_*`, `local_output_numa`) | `shared_mem_buffer[_numa].alloc(tp, this, ...)` | Per-`forward`, sized by `max_possible_qlen`; freed at `:164` | Adapter uses caller-provided or bounded scratch; never sizes by full-model `max_len` |
| Pinned I/O (`input_cpu/weights/output_cpu/...`) | `KExpertsCPUBuffer` (pinned forever until `clear_buffer_cache`) | Per-`(B, layer_slot)`; oversize `qlen` fails loud | Adapter takes caller-owned `hidden`/`out` spans; no retained buffers |
| Mask/map (`gpu_experts_mask`, `physical_to_logical_map`) | Python pinned tensor / `load_weights_task` ptr | Shared R/W between Python and C++ | Adapter takes plain `(layer, expert_id)` + `routing_weight` scalars; no shared mask |

## 15. GPU expert mask / dynamic expert placement

| Field | Value |
|---|---|
| Upstream file | `operators/common.hpp:241-258`, `ext_bindings.cpp:858-862,477-539`, `python/experts_base.py:427-441,162-213`, `python/utils/amx.py:1008-1081`, `fp8_layerwise_transport.{hpp,cpp}` |
| Class/function | `gpu_experts_mask:bool[expert_num]` (pinned CPU tensor, `:431-435`, `data_ptr()` shared as `uint8_t*` R/W); `num_gpu_experts` computed C++-side (`compute_num_gpu_experts`); `should_skip_expert(id)` (skip GPU-resident + `-1` deferred padding); `generate_gpu_experts_masks(freq[L,E], K)` top-K across layers (`:162-213`); `write_weight_scale_to_buffer_task(gpu_tp_count, expert_id, ...)` (SFINAE-bound only if backend implements, `:479-519`) + `run_layerwise_fp8_batch(transport, epoch, layer_id, expert_count)` (`:521-539`, `amx.py:1050-1081`, drains `cpu_infer.sync()` first) |
| Required inputs | Frequency table or explicit mask; per-backend `write_weights_to_buffer` (e.g. `amx/bf16-moe.hpp:307,531`, `fp8-moe.hpp:392,692`, `fp4-moe.hpp:873,1168`, `mxfp8-moe.hpp:683,934`, `avx2/bf16-moe.hpp:131,321`). |
| Output format | Updated GPU-side `w13/u` + `w2/u` pointers + C++ mask flip. |
| Precision | Bit-copy (weights) + `copy_scale_to_bf16` (scales). |
| Memory ownership | GPU pointer lists owned by SGLang; C++ mask points into Python pinned memory. |
| Synchronization | `cpu_infer.sync()` drained before any move-back to avoid racing the non-reentrant distributor. |
| Licensing / portability | Apache-2.0 Python + KVCache.AI-header C++; needs CUDA for the GPU side. |

dee mapping: this is the closest upstream analog to dee's future
admission/eviction — but dee must NOT adopt it. KT's mask is a static-ish
top-K promotion with a full-pool CPU backing store; dee needs per-token
bounded admission (`RAM resident -> CPU vs GPU-transfer` choice, `q* =
argmin max(T_gpu(q), T_cpu(m-q))`) with a bounded host cache. The cost-model
simulator (`cost_model.py`) prepares that interface now without changing live
scheduling (`CPU_EXECUTOR_DESIGN.md` §7, Phase E).

---

## Conflicting assumptions (do-not-import list)

1. **Full expert pool in RAM, no eviction.** Scratch + `local_output_numa` sized by `chunked_prefill_size` (`moe-tp.hpp:156-166`); `capture_bs/capture_buffers` pinned per-B forever; GGUF mmap held per path; native loader freed per-layer after `sync` but C++ NUMA copies stay resident. Oversize `qlen` = fail-loud, not paging. dee is bounded by design (`HostPackCache` budget, VRAM arena) — reuse kernels, not residency.
2. **GGUF universal.** Only `LLAMAFILE` consumes GGUF (`loader.py:816-1078`); AMX wants pre-quant `.kt`/NUMA-sharded safetensors; native wants method-specific safetensors. `Llamafile.load_weights_from_tensors` and `Native.load_weights_from_tensors` both raise for the other's format. dee stays on official safetensors MXFP4.
3. **Converted weights required (for AMX INT4/INT8 / pre-quant paths).** `moe_config.save/load/path`, `load_merged_weight` glob, online BF16→AMX quant only via `AMXMoEWrapper.load_weights_from_tensors(save=True)`. NO JIT GGUF→AMX. For MXFP4 specifically: weights are zero-copy, scales need the lossless `ue8m0->bf16->fp32->e8` re-encode (costed in `FORMAT_COMPATIBILITY.md`) — NOT a full-model conversion.
4. **AVX512 assumed for fast paths.** All `AMX*_MOE` gated on AVX512/BF16/VBMI; AVX2 covers only a subset; `FP8_PERCHANNEL` / K-group AMX have no AVX2 equivalent. dee adapter defaults to portable + AVX2-verified baseline.
5. **SGLang coupling.** Capture-mode guards import `sglang.srt...`; `swiglu_*` comments cite `kt-sglang` origin; doc launch is SGLang-only (`--kt-method MXFP4 --kt-num-gpu-experts 10 --kt-cpuinfer 60 --kt-threadpool-count 2`). Standalone use must replicate router/topk/placement + `physical_to_logical_map:int32[expert_num]`. dee keeps its own router (`route_topk`) and calls the adapter per expert.
6. **CUDA >= SM80 matrix is NOT covered here.** Validated V4-Flash matrix is consumer Blackwell/Ada/Ampere `SM_120/89/86` via `triton_kernels` + Triton NSA fallback (`DeepSeek-V4-Flash.md:21-36`); needs `CUDA>=12.8, flashinfer>=0.6.9, transformers==4.57.1`. T4 is `SM_75` — OUTSIDE the validated matrix. CPU-expert execution itself is CUDA-arch-independent (the point of this track), but any GPU-side validation must use dee's own T4 kernels, not KT's Triton path.
7. **NUMA topology fixed at first init** (`intermediate % tp == 0`, sequential map, frozen singleton). dee targets single-shard default + `numactl --interleave=all` bring-up knob.
8. **Threadpool shared, not owned** (global `sync`, class-global deferred bookkeeping, 2-slot layer ring). dee adapter owns no threads in Phase 1.

---

## What dee CAN reuse directly (no adaptation)

- MXFP4 nibble packing order + E2M1 codepoints + block-32 + `2^(e-127)` scale law (byte-identical weights; value-exact scales — proof in `FORMAT_COMPATIBILITY.md`).
- Asymmetric SwiGLU clamp (`gate=min(g,10)`, `up=clamp(u,±10)`, then `silu(g)*u`) — already matches `src/swiglu_cuda.cu` and `scripts/deepseek_v4_expert_reference.py`.
- `fp4_mat_vec_kgroup[_natural]` / `fp4_mat_mat_kgroup` kernel structure (decode-via-LUT → `dpbf16` → `fmadd(scale)` → 4-wide reduce → 4×4 tile) as a porting template.
- `submit_with_cuda_stream` + `sync_with_cuda_stream` overlap pattern (interface only in Phase 1).
- `finalize_scale_e8` validation + `expand_e8_scales` hoist + `fast_memcpy` + `from_mat_natural` permute as named optimizations for the later AMX port.

## What MUST be adapted (not imported)

- Loader: full-pool walk -> per-expert bounded fill from `HostPackCache`.
- Lifetimes: borrowed global pointers -> per-call `PackedExpertView` borrows + adapter-owned flat blob.
- Threading: global singleton pool -> caller-thread sync now, per-engine pool later.
- Placement: static top-K mask + full CPU backing -> bounded admission/eviction + `q*` cost model (Phase E simulator, no live-scheduling change).
- Router: SGLang-side topk/weights -> dee `route_topk` + per-expert `routing_weight` scalar; adapter never renormalizes (skipped experts contribute 0, matching KT).

## Open verification gaps

- `cpu_backend/` headers not in sparse checkout: `WorkerPool::do_work_stealing_job` / `do_numa_job` exact work-stealing + exception semantics not verified. No production binding until full-tree audit of that directory.
- No T4-adjacent perf numbers claimed here: CPU `ms/expert` must be measured on the campaign host via the `cost_model.py` microbench API, not extrapolated from KT's 8×5090 `~29 tok/s` system number.
- `e=0/255` scale bytes: KT `+inf` vs dee clamp — absent from real checkpoints, documented in `FORMAT_COMPATIBILITY.md`; adapter fails closed on `e=0xFF`.
