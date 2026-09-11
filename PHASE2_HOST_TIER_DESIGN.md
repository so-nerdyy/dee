# Phase 2 host tier skeleton

Experimental, default OFF. Base: Phase-1 closure
`ca8abd075e116a3924c3316cf50d5447de143ed8`. No canonical merge, new model
benchmark, accepted-evidence edits, route changes, or numerical changes.

## Scope and architecture

```
ExpertStore (accepted DEE4 / canonical safetensors)
  -> ExpertStoreColdAdapter : ColdExpertStore
  -> IdentityCodec : StorageCodec
  -> HostExpertTier (persistent aligned slots + leases)
  -> DeviceExpertTier (existing VramCacheManager + AsyncPrefetcher)
  -> existing Engine exact decode / expert kernels / ordered combine
```

Host payloads and policy interfaces are model neutral. The first Engine adapter
explicitly supports packed FP4 CUDA residency only. Unsupported experimental
configurations fail initialization; they never silently substitute another
expert, format, route, or executor. Other exact representations can use the
opaque-byte tiers, but require a separately validated Engine adapter.

The accepted findings remain fixed: `/tmp` is the storage ceiling, pread does
not need optimization, and more lanes/depth do not solve exposed cold fills.
This skeleton creates residency and codec seams for reducing those fills.
It makes no speedup claim and chooses neither Flash's policy nor Muse's codec.
The research verdict is `NO_CODEC_WORTH_BUILDING`; `IdentityCodec` is the only
codec implemented or wired in this phase.

## Identity and route authority

`TierExpertKey` contains an immutable checkpoint/model identity, physical layer,
expert ID, and versioned exact representation/layout identity. Strings are
bounded to 1024 bytes each. The Engine cold adapter appends the store integrity
identity to the caller's checkpoint ID. FP4 representation identity includes
projection dimensions and scale offsets. Device scope is fixed for the lifetime
of an exclusively owned arena; the adapter rejects a different model or layout
even if layer/expert numbers match. The host tier compares the entire key.

The caller remains the routing authority. The Engine uses the same resolved
`source_layer` and expert IDs, in the same stage order. The policy receives IDs
already authorized by that caller; it may choose residency/eviction, not alter
the route. Sparse DEE4 missing records still fail through `ExpertStore::get`.
Store identity, layout, record pointer/index, and exact byte-count checks remain.
The new contiguous adapter additionally rejects a noncanonical region layout.

DEE4's accepted loader validates metadata and record identity; this work does
not add a per-read cryptographic rehash or change the existing seal/import
contract. Synthetic fixture hashes are fixtures, not real-model evidence.

## Persistent allocation and ownership

`HostTierConfig` explicitly sets slot size, alignment, static/policy slot count,
dynamic slot count, and a byte budget. At construction, every payload slot is
allocated once with `_aligned_malloc` or `posix_memalign`. Its allocation size
is rounded to the requested alignment (4096 by default). CUDA slots use
`cudaHostRegister(..., cudaHostRegisterPortable)` once, on the configured device.
The CUDA backend restores the prior device after registration/unregistration.

The budget test is `slots <= budget / aligned_slot_bytes`, with checked addition
and alignment rounding. There is no implicit 8-GiB Phase-2 allocation. Pool
geometry cannot grow or shrink while leases exist. Idle slots are reused
without payload malloc/free, register/unregister, or full-record zeroing.
Key string capacity and the candidate array are reserved at construction, so
the tier's lookup/reservation path also reuses its metadata. Existing store
lookup tensor metadata, CUDA events, and caller policies retain their existing
allocation mechanisms; the no-per-request allocation guarantee concerns the
host-slot payload allocator, not arbitrary external callbacks.

Payload accounting includes alignment padding. C++ metadata, CUDA event objects,
page tables/driver overhead, OS page cache, and other Engine buffers are separate
from that budget. Slot metadata and candidate count are bounded by slot count;
this is not a whole-process RSS limit.

Slot lifecycle:

```
Empty -> Filling (exclusive writer, generation increment)
      -> Ready (exact successful fill only)
      -> Empty/next generation (idle eviction only)
```

`HostExpertLease` is copyable RAII. It holds shared pool ownership, slot index,
and generation; reference counts prevent reuse while any lease exists. The
pointer and key are immutable during that lease. Leases can safely outlive the
`HostExpertTier` facade. Final pool destruction unregisters successful
registrations and frees all payloads exactly once. Partial construction unwinds
previous slots. A caller may not concurrently destroy a facade while another
thread is calling its methods, or destroy a store while a read is active.

Fills run outside the pool mutex. Concurrent duplicate requests wait on the
reserved generation and share its result, retaining references even during
failure publication. A failed/throwing/short Identity fill publishes no lease,
wakes waiters, and becomes reusable after references drain. No stale bytes are
exposed. Capacity exhaustion returns an explicit status; it does not allocate
an unbudgeted bypass buffer or select a different expert.

## Residency and placement seams

`PolicyResident` and `Dynamic` use separate configured slot partitions.
Policy-resident slots are protected even when idle, independent of whether OS
pinning succeeded. Dynamic eviction candidates contain only ready, unleased
dynamic slots. `HostPlacementPolicy::residency` chooses the partition and
`victim` chooses an offered slot or declines. Offered keys are borrowed only for
the callback. An invalid victim index cannot bypass protection. Explicit
`evict(key)` can withdraw either class, but only when idle.

There is no production HostPlacementPolicy implementation. Tests use a local
first-eligible policy solely to create deterministic eviction scenarios.
Callbacks run under the host mutex, must not reenter the tier, and should avoid
blocking. No route prediction, activation collection, or placement training is
implemented. Flash can capture its independently computed statistics/mask in
its policy object.

`DevicePlacementPolicy` supplies request priority and eviction score to the
existing arena. Omission retains the existing priority/LRU behavior. Even a
custom score cannot evict pinned DMA/compute blocks or expand the byte budget.
Device access retains Engine's single-caller ownership contract.

## Exact copies and codecs

On Linux, canonical DEE4 `materialize(view, final_slot, bytes)` runs the existing
pread loop directly into the final registered cache slot. There is no preceding
temporary full-record allocation and no subsequent host gather for H2D.
On Windows, the accepted materializer copies its mapped record once into that
same final slot. Safetensors requires six region copies to assemble the final
packed record; each region goes directly to its final position. Packed weights
and scale bytes remain unchanged in all cases.

The adapter rejects contiguous metadata whose weight/scale region ordering
does not match the executor layout, rather than silently reinterpreting bytes.
Supporting such a layout later would require explicit normalization or a
matching executor representation; current accepted DEE4 layouts need neither.

`StorageCodec::accepts` validates even cache-hit descriptors. `IdentityCodec`
requires matching stored/exact sizes and delegates the read directly into the
destination. `materialize` is the future lossless decode seam. No codec library,
CPU FP16 expansion, or storage rewrite is included. Future codec scratch and
integrity verification must be explicit and bounded before enabling it.

## CUDA ordering and fallback

The new `prefetch_host_lease` branch submits packed bytes on the existing
AsyncPrefetcher stream and records its existing style of per-transfer completion
event. It does not allocate the legacy gather ring or a device decode staging
buffer. The existing exact executor decodes packed device bytes into its
existing bounded FP16 scratch at compute time.

DMA retains both a host lease and the arena's transfer pin. A compute-stream
`cudaStreamWaitEvent` orders device work but does **not** prove host-source
completion. The experimental branch keeps the lease until event query/wait
confirms completion, even after a compute dependency is armed. This ownership
is independent of Engine's compute pin, which protects the device block until
the existing compute completion boundary. A synchronous wait after a stream
wait still checks actual DMA completion.

Before acquiring another host slot, the device facade polls completions. Under
host capacity pressure it may wait for one specific pending transfer event and
retry. That releases host ownership only, retaining the device reservation for
its consumer. No new `cudaDeviceSynchronize` is used. Existing reset/teardown
stream drains remain.

If host registration fails, the same aligned pageable slot stays usable.
CUDA may stage pageable bytes internally; the adapter submits the same packed
bytes and waits for that copy's event. This fallback may block and may include
an unavoidable driver staging copy. It neither expands weights on the CPU nor
allocates an application-owned full-record staging buffer.

On submission failure after a DMA might have launched, the experimental path
drains that transfer stream before releasing either allocation, then discards
only its own unpinned failed device generation. If a fatal CUDA error prevents
proving a safe drain/unregistration, it terminates rather than free potentially
in-use memory. This fail-stop rule affects the explicit experimental path only.

## Architectural references consulted

- [MoE-Infinity host store](https://github.com/EfficientMoE/MoE-Infinity/blob/main/moe_infinity/models/deepseek_v4/official_offload_adapter.py):
  persistent pinned packed expert tensors, byte-preserving FP4 transfer, and a
  dedicated asynchronous copy stream. We borrow the separation of responsibilities,
  not its LRU policy, prefetch schedule, executor, or throughput claims.
- [MoE-Infinity DeepSeek-V4 architecture](https://github.com/EfficientMoE/MoE-Infinity/blob/main/moe_infinity/models/deepseek_v4/README.md):
  packed host/GPU representation and on-device dequantization boundary.
- [KTransformers placement interface](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md):
  activation-stat/mask input is separate from placement execution. Only the
  injection seam is adopted; no activation-based policy is selected.
- [moe-l2 architecture](https://github.com/yalun753/moe-l2#architecture):
  host-resident activated experts and a bounded GPU cache. Its predictors,
  mapped-host direct access path, and numerical choices are not imported.
- [CUDA memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html):
  registration and asynchronous transfer contracts inform source lifetimes.

References were inspected during this implementation; they are architectural
references, not dee.cpp acceptance evidence.
