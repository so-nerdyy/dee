# Phase 2 integration map

Branch: `research/phase2-integration-lru-fix`, rooted directly at
`ca8abd075e116a3924c3316cf50d5447de143ed8`. No unrelated research commits.

## Integration boundaries

| Boundary | Implementation | Contract |
|---|---|---|
| Cold source | `ExpertStoreColdAdapter` in `expert_tiers.cpp` | Accepted DEE4 positional materialization, or exact six-region safetensors gather, into final slot. Serialized legacy lookup/stat mutation; independent reads can overlap. |
| Codec | `StorageCodec` / `IdentityCodec` | Descriptor validation even on hits; only identity implemented/wired. |
| Host residency | `HostExpertTier` | Two configured partitions, reusable aligned slots, registration fallback, concurrent fill coalescing, RAII leases. |
| Host placement | `PlainLruHostPlacementPolicy` via `HostPlacementPolicy` | Selected plain LRU; all entries dynamic, least-recently-used idle victim. |
| Device residency | `DeviceExpertTier` over existing arena | Exclusive model/layout/device scope, same device budget, existing consumer/DMA pins. |
| Device placement | `VramCacheManager::set_experimental_plain_lru` | Explicit actual-recency repair; independently switchable from host tier. |
| H2D | `AsyncPrefetcher::prefetch_host_lease` | Packed bytes on existing stream, retained source through actual completion event, no gather ring. |
| Executor | Existing `Engine` FP4 decode and expert calls | Same compute scratch, kernels, routes, order, clamp, and tolerances. |
| Entry | `EngineConfig::phase2` | Master and host/VRAM sub-switches default OFF; unsupported combinations fail closed. |
| Metrics | `Engine::phase2_metrics` | Separate tier-lifetime snapshot with explicit token denominator. |

The initial experimental Engine path fills at the authoritative per-expert
stage boundary. It does not reuse the legacy host-cache batch worker queue;
parallel host acquisition is supported/tested at the tier interface, but a
Phase-2 Engine lane scheduler is deliberately not selected here. Existing
`source_read_lanes`/queue-depth settings still govern the disabled legacy path.
No pread changes, lane/depth increases, generic route prediction, GDS, io_uring,
DSpark, tile streaming, or DualDeadline are implemented.

## Explicit C++ enablement

Start from an existing valid packed-FP4 CUDA Engine configuration:

```cpp
cfg.phase2.enabled = true;
cfg.phase2.host_enabled = true;              // host arm; false for VRAM-only arm
cfg.phase2.vram_priority_fix_enabled = false; // independently set true for VRAM arm
cfg.phase2.model_identity = immutable_checkpoint_revision;
cfg.phase2.host.dynamic_slots = reviewed_dynamic_slots;
cfg.phase2.host.policy_slots = 0; // plain LRU has no policy-resident/frequency slots
cfg.phase2.host.budget_bytes = reviewed_total_aligned_slot_budget;
cfg.phase2.host_policy = {}; // null selects PlainLruHostPlacementPolicy
cfg.phase2.device_policy = {}; // no custom score in this candidate
// slot_bytes==0: Engine derives the exact packed record size from its layout.
// No implicit total host budget or chosen slot split.
if (!engine.init(cfg)) { /* fail experiment; inspect the initialization error */ }
```

The experiment is C++ opt-in in this commit. Python and CLI launchers do not
expose a hidden default or silently supply a placeholder policy. Rebuild all
Engine consumers/bindings when integrating: EngineConfig/Engine and Transfer
layout changed, so old binary objects must not be mixed with these headers.

## Smallest next steps for Flash and Muse

Flash's corrected result is integrated here: host policy is plain LRU and no
frequency pin policy is implemented. The selected VRAM repair is independently
enabled with `vram_priority_fix_enabled`; it uses actual recency only. No
source-fill, transfer, lease, route, or executor change is needed for either
policy selection.

Muse, **only if a lossless codec is selected**: implement
`StorageCodec::accepts/materialize` plus a codec-aware `ColdExpertStore` record
descriptor/reader. Decode into the final destination, explicitly account for
bounded scratch, verify exact bytes and corruption/truncation failures, then
inject that selected codec into the Engine seam currently held by
`phase2_codec_`. Keep the executor representation identity unchanged only when
the resulting packed bytes/layout are identical. Storage codec identity and
stored size are separate from executor representation and exact size. No codec
is selected, linked, or used in this commit.

Before enabling real-model experiments, run the provided small CUDA synthetic
test and existing FP4 CUDA tests on an available GPU. This is independent of
the deferred expensive full-model A/B, which remains deferred until Flash and
Muse return.

## Build scope

The integration branch does not modify `host_pack_cache.cpp` or accepted
benchmark/evidence artifacts. The legacy Phase-1 path remains the default when
the Phase-2 master and sub-switches are off.

## Tests and verification

Commands (run from the repository root; choose local build-directory paths):

```sh
cmake -S dee.cpp -B build/phase2 -DDEE_CUDA=OFF -DDEE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/phase2 --parallel 4
ctest --test-dir build/phase2 --output-on-failure --parallel 4
```

`test_phase2_host_tier` exercises hit/miss, both residency classes, eviction,
duplicate concurrent requests and shared failure, allocation/pin failure,
persistent slot reuse, exact byte identity, host/device budgets, surviving
leases, pending-DMA source protection, device compute reservation protection,
wrong model/layout/layer IDs, codec rejection, and default/explicit-OFF
bit-identical synthetic Engine outputs. It requires no real Oracle/model.

`test_expert_store` additionally routes actual synthetic DEE4 v2 and sparse-v3
files through the adapter, codec, and final host slot. It checks byte equality,
cache-hit read avoidance, missing trace expert failure, and recovery after an
invalid fill. Existing DEE4 repack/import/route contract tests are retained.

`test_phase2_cuda` is built only with `DEE_CUDA=ON`. It uses a stream callback
gate to keep DMA deliberately pending, verifies a compute-stream wait cannot
release the host source early, then verifies actual copied bytes, slot reuse,
and forced pageable fallback. It exits 77 (CTest SKIP) without a GPU and has a
60-second timeout. It is a synthetic lifecycle check, not model parity proof.

### Validation status (2026-09-10)

- Clean MinGW CPU build passed after the minimal ca8 host-cache compile fix.
- Focused local gate passed 10/10, including the Phase-2 host/device tests,
  exact DEE4 store integration, plain-LRU VRAM test, async lifetime test,
  disabled Engine equivalence, and existing resolver/router tests.
- CUDA was not compiled or run, so direct H2D/runtime lifetime and model parity
  remain open. This is mechanism/correctness evidence, not T4 performance
  acceptance.

## Exact changed-file manifest

New:

- `PHASE2_HOST_TIER_DESIGN.md`
- `PHASE2_INTEGRATION_MAP.md`
- `PHASE2_METRICS.md`
- `PHASE2_AB_RUNBOOK.md`
- `dee.cpp/include/dee/host_expert_tier.h`
- `dee.cpp/include/dee/expert_tiers.h`
- `dee.cpp/src/host_expert_tier.cpp`
- `dee.cpp/src/expert_tiers.cpp`
- `dee.cpp/tests/test_phase2_host_tier.cpp`
- `dee.cpp/tests/test_phase2_cuda.cpp`

Modified:

- `dee.cpp/CMakeLists.txt`
- `dee.cpp/include/dee/async_prefetcher.h`
- `dee.cpp/include/dee/engine.h`
- `dee.cpp/include/dee/vram_cache.h`
- `dee.cpp/src/async_prefetcher.cpp`
- `dee.cpp/src/engine.cpp`
- `dee.cpp/src/vram_cache.cpp`
- `dee.cpp/src/host_pack_cache.cpp` (build prerequisite only)
- `dee.cpp/tests/test_expert_store.cpp`

No paths under accepted benchmark/evidence, `research/route-pipeline`, the
repacker, numerical kernels, router selection, or canonical model artifacts
are changed. Build outputs and generated assets are not part of the commit.
