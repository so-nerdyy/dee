# Phase-2 Host/Device Tier Verification Audit

**Audit target:** `08d3d51490f6d4f9eea2ae1c7432220b82eeb99e` ("feat(phase2): add generic host and device tier seams")
**Branch:** `research/phase2-host-tier-verification`
**Method:** adversarial static review of every new seam + a new CPU-buildable adversarial test suite (`dee.cpp/tests/test_phase2_tier_audit.cpp`, 130 checks) exercising the invariant surface on MinGW/GCC 15.2 (LLP64, DEE_CUDA=OFF). CUDA-only paths were reviewed line-by-line but could not be executed (no nvcc/GPU in environment).
**Verdict: ACCEPT** — no Critical or High defects found. The seams are defensively implemented: every failure mode I could construct either fails closed (Invalid/Capacity/FillFailed/false) or is a documented contract consequence. Two Medium findings are both latent/API-hardening issues, not reachable in the sanctioned Engine flow.

---

## 1. Findings

### MEDIUM — LLP64 `map_key` drops the layer entirely (pre-existing, deliberately retained)
- **File:** `dee.cpp/src/async_prefetcher.cpp:56-60` (`map_key`), parent form at `08d3d51~1:find_inflight` (`static_cast<long>(key_id(...))`), map type `unordered_map<long,int>` → now `unordered_map<uint64_t,int>`.
- **Defect:** `static_cast<uint64_t>(static_cast<long>(key))` narrows the 64-bit `(layer<<32)|expert` key to 32-bit `long` on Windows/LLP64 — the high 32 bits (layer) are discarded, so `map_key(L,E)` collides for **all** layers sharing an expert id.
- **Violating input:** `prefetch(0,5,src)` then `prefetch(1,5,src)` on MinGW — the second call hits the first's in-flight map entry, returns `prior.id`, and never loads layer-1 bytes. `wait(1,5)` fails closed (`is_resident(1,5)==false` after `release_transfer`); no wrong bytes are served, but a multi-layer CUDA run on LLP64 cannot stage layer>0 experts through the legacy path.
- **Reproduction:** `test_phase2_tier_audit.cpp::llp64_layer_collision` — PASSES, confirming `b == a`, `wait(1,5)==false`, `!is_resident(1,5)` under LLP64. Guarded by `#if sizeof(long)==4` so it also validates correct behavior on LP64.
- **Disposition:** pre-existing defect that this commit *intentionally preserves* for default-OFF equivalence ("Keep legacy behavior when disabled"). Under the experimental flag, `map_key` returns the full 64-bit `key_id`, fixing it. Verified by `experimental_mapkey_distinct` + `experimental_flag_persistence` (LLP64 branch asserts the post-enablement fix permanently applies to that prefetcher).
- **Transfers to Luna's rewrite:** YES — if the rewrite keeps a keyed in-flight map on any control path, it must use the full 64-bit key unconditionally. Also note `batch_keys_` (duplicate-request accounting) inherits the same collision → misattributes duplicates on LLP64 legacy runs (cosmetic).

### MEDIUM — `DeviceExpertTier` scope exclusivity enforced only at construction, not for lifetime
- **File:** `dee.cpp/src/expert_tiers.cpp:87-100` (ctor guards), `expert_tiers.cpp:113-125` (resident-hit path checks only `is_resident`+`size_of`), `dee.cpp/src/vram_cache.cpp` (blocks keyed by `(layer,expert)` only — no model dimension).
- **Defect:** The ctor requires an empty cache and drained prefetcher, but nothing prevents a *second* `DeviceExpertTier` from being constructed on the same `cache_`+`prefetcher_` later — e.g., after `prefetcher_.reset()` + `cache_.clear()` while the first tier is still alive (`reset()`/`clear()` are public and reachable via `Engine::reset_runtime_cache`, engine.cpp:2144-2145).
- **Violating interleaving:** tier1 (scope model-A) alive → all transfers drained, blocks evicted → `prefetcher.reset()` → `in_flight()==0` → tier2 ctor (scope model-B) **succeeds** → `enable_experimental_host_tier` overwrites `experimental_scope_` → tier1's subsequent `stage()` calls: `accepts` passes (checks record vs tier1's own scope), `is_resident(0,5)` may now see a block tier2 loaded with **model-B bytes** — `size_of` matches if layouts agree → resident-hit path returns success → tier1's consumer reads model-B bytes as model-A. Silent cross-model byte confusion.
- **Reproduction:** reasoning trace — requires constructing two tiers on shared arena outside Engine (Engine creates exactly one per init and replaces it on re-init, destructing the old first). Unreachable through sanctioned Engine flow; reachable through the public C++ API.
- **Transfers to Luna's rewrite:** YES — pin the scope in the shared arena/prefetcher for the tier's lifetime (e.g., an ownership token/generation the tier validates on every `stage`, or a scope epoch bumped by `reset()` that invalidates stale tiers). Ctor-time emptiness is not lifetime exclusivity.

### LOW — Armed experimental state outlives `DeviceExpertTier`
- **File:** `dee.cpp/include/dee/async_prefetcher.h:95-100` (`enable_experimental_host_tier` sets flag+scope, no disable), `dee.cpp/src/expert_tiers.cpp:100` (`~DeviceExpertTier` clears only the eviction-score callback).
- **Defect:** after tier destruction, `experimental_host_tier_`/`experimental_scope_` persist: `prefetch_host_lease` still accepts leases matching the stale scope, and `map_key` permanently uses 64-bit keys for this prefetcher instance.
- **Reproduction:** `experimental_flag_persistence` — PASSES: `prefetch_host_lease` returns ≥0 and `prefetch(0,6)`/`prefetch(1,6)` no longer collide on LLP64 after the tier is destroyed.
- **Assessment:** mostly inert under the single-caller contract (only `DeviceExpertTier::stage` calls `prefetch_host_lease`), and re-enabling a second tier overwrites the scope — but the stale flag has one behavioral consequence: `synchronize_all` at `async_prefetcher.cpp:624` calls `std::terminate` on a prefetch-stream sync failure whenever `experimental_host_tier_` is set — a stale armed flag upgrades that legacy failure path from `return` to process abort even when zero managed transfers exist. Narrow (needs tier teardown + prefetcher reuse + a sticky CUDA error) but the fail-stop is keyed on flag state, not on actual in-flight managed work. Deliberately *not* auto-fixed here: clearing the flag would revert `map_key` to the LLP64-truncating legacy form, re-introducing finding #1 for post-teardown prefetches.
- **Transfers:** YES — decide whether the armed seam should be scoped (guard object) or explicitly documented as one-shot-per-prefetcher; if kept, key the terminate on "managed transfers actually in flight", not the flag.

### LOW — `cuda_submit_host` failure path leaks the CUDA event on the abandoned transfer
- **File:** `dee.cpp/src/async_prefetcher.cpp:732-738` (event created + assigned to `transfer.event`), `749-754` (failure path drains stream, sets `dma_complete`, returns false), `499-504` (caller abandons transfer, keeps it in `inflight_`), `657-663` (events destroyed only in `reset()`).
- **Defect:** on submit/record failure the just-created event is not destroyed; it lingers on the abandoned transfer until `reset()`. Bounded by in-flight table size per reset cycle — an object leak, not memory.
- **Violating input:** repeated `cuda_submit_host` failures (e.g., error-injected `cudaMemcpyAsync`) accumulate events on abandoned transfers.
- **Reproduction:** static — CUDA-only path, not executable in this environment. Legacy `cuda_submit` has the same pattern (event assigned at 875, no destroy on its failure paths at 887-926), so this is a consistent-but-incomplete cleanup convention.
- **Transfers:** YES — in the `!ok` branch after the mandatory stream drain, `cudaEventDestroy` the event and null `transfer.event` (safe: the drain already proved no in-flight work references it).

### LOW — Pageable (unregistered) host slots force a host block inside `cuda_submit_host`
- **File:** `dee.cpp/src/async_prefetcher.cpp:756-766` — `cudaEventSynchronize(event)` on the just-submitted copy when `!transfer.source_pinned`.
- **Defect:** a host-side block inside the supposedly-async submission path. Documented honestly ("This fallback may block and may include an unavoidable driver staging copy") and metric'd via `experimental_pageable_wait_ms_`, but note it triggers whenever `HostExpertTier` silently failed `cudaHostRegister` at init (surfaced only as the `pin_failures` stat). On a pin-starved host this turns every H2D into a synchronous wait — a throughput cliff, not a correctness bug.
- **Transfers:** keep the fallback; consider surfacing pin-failure rate in engine-level init logs (currently only `stats().pin_failures`).

### LOW — `HostExpertLease::key()` returns a `const&` into pooled slot memory
- **File:** `dee.cpp/src/host_expert_tier.cpp:145-148`.
- **Defect:** safe while the lease is held (`references>0` forbids slot reuse — verified), but a caller storing the reference past `reset()`/destruction dangles. `TierExpertKey` is ≤~2 KiB of strings, so the choice is deliberate (cheap lease ops); the contract isn't documented on the accessor.
- **Transfers:** document "borrowed for lease lifetime" on `key()` (mirroring `HostVictim::key`'s comment) or return by value if the rewrite makes key access cold.

### LOW — Same-key record with different `exact_bytes` is permanently `Invalid` while resident
- **File:** `dee.cpp/src/host_expert_tier.cpp:190-191` — hit path returns `Invalid` on `slot.bytes != record.exact_bytes`, and every subsequent acquire of that key+size fails until `evict(key)` withdraws the resident copy.
- **Reproduction:** `guard: same-key different-size record rejected` + `did not poison the resident slot` — PASSES. Unreachable via `ExpertStoreColdAdapter` (fixed `bytes_` per adapter); reachable only by a caller hand-building inconsistent records.
- **Assessment:** fail-closed and recoverable via explicit `evict()` — a semantic consequence to preserve, not a bug. Verified so rewrites don't "fix" it into silent re-fill.

### LOW — mmap-based materializers crash (AV/SIGBUS) on externally truncated files; `pread` path cannot
- **File:** `dee.cpp/src/expert_store.cpp:111-119` (base `materialize` = `memcpy` from mapped view), `555-557` (Windows DEE4 = `memcpy` from mapped view), `563-577` (Linux DEE4 = `pread` loop, `count<=0` fails cleanly at EOF).
- **Defect:** on Windows (and on the base/Safetensors path on all platforms), materializing from a file truncated after mapping raises an access violation instead of a clean failure. `materialization_mode()` is correctly `#ifdef _WIN32`-reported as `mmap_memcpy`/`pread`, and the design doc already qualifies "On Windows, the accepted materializer copies its mapped record once" — so the claim is accurate; the crash surface is a pre-existing platform caveat, not introduced here.
- **Transfers:** unchanged — POSIX `pread` path already handles this correctly; Windows has no `pread`-into-mapping equivalent (would need `ReadFile` on the file handle instead of view memcpy if hardening is wanted).

### INFO — `std::terminate` on unprovable-drain CUDA failures (deliberate fail-stop)
- **File:** `dee.cpp/src/async_prefetcher.cpp:624` (`synchronize_all` stream-sync failure under experimental mode), `752` (post-submit drain failure), `760-761` (pageable event-sync failure); `dee.cpp/src/host_expert_tier.cpp:74` (failed `cudaHostUnregister`).
- **Assessment:** defensible and documented ("If a fatal CUDA error prevents proving a safe drain/unregistration, it terminates rather than free potentially in-use memory"). Asymmetry worth noting: the **legacy** path at 624-625 `return`s instead, and `reset()` then destroys events/frees pinned staging while copies are unprovably pending — pre-existing teardown hazard on sticky-error streams; sticky errors doom the process anyway.
- **Transfers:** keep fail-stop; optionally extend it to the legacy branch for symmetry.

### INFO — `Engine::init` ordering and cleanup quirks
- `phase2_device_` is constructed at `engine.cpp:3673` **before** `prefetcher_.init` at `3749`. `enable_experimental_host_tier` only needs `inflight_.empty()` (h:95-96) so this is legal, but on Engine re-init without `reset_runtime_cache`, stale `inflight_` entries make the ctor throw even though `prefetcher_.init` would have reset them one statement later — fails closed, just brittle.
- `host_config.slot_bytes < cache_blob_bytes_ → return false` at `engine.cpp:3670` sits *inside* the `try` but bypasses the catch's `phase2_*_.reset()` cleanup — `phase2_cold_` remains constructed-but-inert (unique_ptr member, no leak; next init overwrites). Cosmetic.
- Phase-2 + `BenchmarkScenario::{ComputeOnly,ResidentBypass}`: `bypass_cache` skips staging entirely, `cache_.data` returns nullptr → `swiglu_ok` false → `return false` — fails closed (engine.cpp:4093-4118). Not a supported combination; no guard needed.

### INFO — Accounting caveats (cosmetic, verified consistent)
- `HostTierStats::SSD_bytes` adds store-reported `bytes_read` even on failed/lying reads (`host_expert_tier.cpp:245` runs before the success check) — a store reporting `bytes_read > n` inflates the counter; no functional impact (the fill still fails).
- `stats().leased_slots` counts Filling slots and held references — intended; verified mid-fill.
- `metrics_.H2D_bytes` counts at submission, not DMA completion — standard for enqueue metrics.
- `resident_bytes` add/sub is balanced under one mutex (fill-success vs evict paths) — no underflow reachable.
- Fill waits (`cv.wait`, `host_expert_tier.cpp:196`) have no timeout — a store that hangs (e.g., dead filesystem) blocks coalesced waiters indefinitely. Inherent to the no-cancellation design; callers are threads, not signals.

### INFO — Verified-correct mechanisms (no defect found; evidence attached)
- **Lease refcounting:** copy ctor/copy-assign `++references`, move transfers without double-counting, self-assign guarded, `reset()` idempotent, `--references` under mutex. Tested: `lease_refcount_lifecycle`, `empty_lease_contract`.
- **Facade teardown:** `acquire` copies `state_` once at entry (`host_expert_tier.cpp:184`) and never touches `this` again — pool survives facade death while in-flight fills/waiters hold the shared state. Tested: `facade_teardown_mid_fill` (this exceeds the documented contract — "a caller may not concurrently destroy a facade while another thread is calling its methods" — and the implementation survives it anyway; that's hardening evidence, not license to rely on it).
- **Duplicate fills:** single `Filling` phase per key; waiters hold references so the slot can't be recycled mid-wait; failure publishes `Empty` + `notify_all`; waiters report their own generation's failure and must retry. Tested: `fill_failure_publication`, `concurrent_stress` (8 threads → `reads==1`, `fills==1`, `host_hit==7`).
- **Generation/ABA:** slot `generation` monotonic; eviction/recycle requires `references==0`; stale `Empty`-slot keys invisible to the hit path (loop skips `Empty` before key compare). `discard_unpinned` double-guards on generation AND `pins==0` — stale-generation discards cannot erase a replacement block. Tested: `discard_unpinned_guards`.
- **Eviction-while-leased:** `evict()` requires `Ready && references==0`; Filling slots and any lease block withdrawal. Tested: `evict_semantics`, `lease_refcount_lifecycle`.
- **Residency partitions:** policy slots are never offered as victims (candidate gate `residency==Dynamic` at `host_expert_tier.cpp:215`), can't host Dynamic keys, and a reclassified key is not relocated on hit — it stays stranded in its original partition, reclaimable only by Dynamic-residency victims or explicit `evict()`. Tested: `residency_partitions`, `residency_flip_stays_in_slot` (the flip test documents that a runtime policy flip can strand partition capacity — a policy-contract consequence, verified not a crash).
- **Policy callback safety:** `victim()` return validated against offered candidates (`any_of`, `host_expert_tier.cpp:220-221`) — an out-of-range answer degrades to `Capacity`. Callbacks run under the tier mutex — documented at `host_expert_tier.h:75` ("must not reenter the tier"); a reentrant/blocking policy would deadlock all acquires — contract-documented hazard. `DevicePlacementPolicy::eviction_score` has no equivalent header note; it runs inside `ensure → evict_until_free` and a policy that reenters `cache_` would recurse into eviction — same contract class, undocumented on this side.
- **Store-side guards:** `IdentityCodec::materialize` requires `bytes_read == exact_bytes` — short reads, lied counts, throws, and `{false,*}` all publish `FillFailed`. Tested: `guard: fail/throw/short/lied fill all fail closed` (4 modes).
- **Adapter identity/layout:** per-read re-verification of `store_.integrity_identity()`, `same_layout` (shapes/dtypes/nbytes per region), contiguous tiling order (weights-then-scales exact-position check), and `contiguous_nbytes == n` — a mis-ordered or shrunk store is rejected rather than reinterpreted. Tested: `adapter_layout_guards` (identity flip, nbytes mutation, mis-ordered tiling all detected).
- **DMA source lifetime (mock-verified, CUDA-static):** `host_lease` is reset only when `dma_complete` is proven — `cudaEventQuery`/`cudaEventSynchronize` (`release_staging` 63-69, `collect_host_sources` 695-721), stream sync (`synchronize_all` 637-646), post-submit drain (`cuda_submit_host` 752), or mock memcpy (`drain_until` 522-526). `wait_on_stream` (554-605) arms the compute stream but does **not** free the source — `done=true` while `dma_complete` may stay false; the lease is released later by `collect_host_sources`/`reset`. Tested: `mock_collect_and_reset` (non-blocking collect can't reclaim a pending source; blocking collect copies+frees; `reset()` releases outstanding leases; drained blocks stay resident).
- **Scope guards:** `prefetch_host_lease` rejects foreign model/representation and empty leases (`async_prefetcher.cpp:686-688`); `DeviceExpertTier` ctor rejects non-empty cache / non-empty inflight (`expert_tiers.cpp:90-92`). Tested: `prefetch_host_lease_guards`, `device_tier_ctor_guards`.
- **Budget/stride accounting:** ctor rejects `count > budget/stride` with checked arithmetic; exact-fit accepted; alignment padding included in `allocated_bytes`. Tested: `budget_stride_boundary`.
- **Registration fallback:** `try_pin=false` → zero pin calls; `pin` returning false → per-slot `pin_failures` counted once at init, slot stays usable pageable. Tested: `pin_fallback_accounting`.
- **Packed layout consistency:** `configure_fp4_quantized` (`engine.cpp:2699-2733`) produces `[w0,w1,w2,s0,s1,s2]` offsets identical to the adapter's gather order (`expert_tiers.cpp:69-79`); init cross-checks `layout.fp4_total_nbytes == cache_blob_bytes_` (`3653`) and `slot_bytes >= cache_blob_bytes_` (`3670`); `decode_fp4_cache_block_to_scratch` (`3236-3259`) consumes the same offsets. Byte-exact verified on the mock path (`mock: collect wrote exact bytes to the arena block`).
- **Default-OFF equivalence:** `prefetch`/`prepare_fp4_experts`/`stage_expert`/`cuda_submit` legacy paths unchanged when `phase2.enabled==false`; `map_key` deliberately preserves the LLP64 cast; `prepare_fp4_experts` early-returns only under the flag (`2758`). Existing suite `test_phase2_host_tier` covers explicit-OFF output equivalence.
- **No new `cudaDeviceSynchronize`:** confirmed — all waits are `cudaEventSynchronize`/`cudaStreamSynchronize(prefetch stream)`/`cudaStreamWaitEvent`. `collect_host_sources(true)` blocks on one event under host-capacity pressure inside `DeviceExpertTier::stage` — bounded by in-flight count, metric'd as `host_capacity_wait_ms`, documented.
- **Member destruction order (Engine):** `phase2_device_`→`phase2_host_`→`phase2_cold_`→`prefetcher_`→`cache_`→`expert_store_` (reverse of declaration, engine.h:392-399). Device tier clears its eviction-score callback while `cache_` is still alive; host pool outlives facade via shared state; `prefetcher_` dtor drains the stream before destroying it. Order-correct.

---

## 2. Doc-claim accuracy review (no overclaims found)

`PHASE2_HOST_TIER_DESIGN.md` claims were checked line-by-line against the implementation:
- "pread loop directly into the final registered cache slot" — qualified correctly as **Linux/DEE4-only**; Windows and Safetensors copies are explicitly acknowledged (doc:122-130). Accurate.
- "no application-owned full-record staging copy" — accurate: the six-region gather writes directly into the final slot; the driver-internal pageable staging is explicitly disclaimed (doc:163-168).
- "No new cudaDeviceSynchronize" — verified in code.
- "Callbacks run under the host mutex, must not reenter the tier" — contract documented (doc + `host_expert_tier.h:75`).
- "Leases can safely outlive the HostExpertTier facade" — verified destructively.
- "A caller may not concurrently destroy a facade while another thread is calling its methods" — the implementation actually survives this beyond contract (my teardown test relies on it), but the contract remains the right rule.
- Bounded budget/alignment accounting claims — verified at the stride boundary.

`PHASE2_INTEGRATION_MAP.md` / `PHASE2_METRICS.md` / `PHASE2_AB_RUNBOOK.md` — consistent with observed semantics; no contradictions found.

## 3. Test status

| Suite | Build | Run | Result |
|---|---|---|---|
| `test_phase2_tier_audit` (new, 130 checks) | MinGW GCC 15.2, DEE_CUDA=OFF | ctest #10 | **PASS** (0 failures) |
| `test_phase2_host_tier` (existing) | same | ctest | PASS |
| `test_expert_store`, `test_vram_cache`, `test_async_prefetcher` | same | ctest | PASS |
| Full C++ suite (12 tests) | same | ctest | 12/12 PASS |
| Python tests (`test_deepseek_v4_*`) | — | ctest | 4× FAIL — `requires pytest` (environmental, pre-existing, unrelated to this audit) |
| `test_phase2_cuda` | CUDA-only | — | not built (no DEE_CUDA/GPU in env); static review only |

Build: `cmake -S dee.cpp -B build -G "MinGW Makefiles" -DDEE_CUDA=OFF -DDEE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug` + `cmake --build build --parallel 4` — clean.

## 4. Findings by severity

| Severity | Count | Items |
|---|---|---|
| Critical | 0 | — |
| High | 0 | — |
| Medium | 2 | LLP64 `map_key` layer truncation (pre-existing, retained, fixed-on-enable); `DeviceExpertTier` scope exclusivity is ctor-time only |
| Low | 6 | armed-state persistence past `~DeviceExpertTier`; event-object leak on failed `cuda_submit_host`; pageable fallback host-block; `lease.key()` reference lifetime; same-key size-mismatch `Invalid` semantic; mmap-truncation crash surface (pre-existing platform caveat) |
| Info | ~10 | fail-stop `std::terminate` policy; Engine init ordering/cleanup quirks; accounting caveats; verified-correct mechanism list |

## 5. Go/No-Go for live mechanism testing

**GO.** No defect found that would corrupt data or crash under the sanctioned Engine flow; the two Mediums are latent/API-hardening issues worth fixing in the rewrite, and the Lows are mostly documentation/contract notes. The CUDA-side paths (event ownership, `cuda_submit_host` failure drain, `wait_on_stream` transient-unpin window) were verified statically only — the first live T4 run should specifically exercise: (a) `cuda_submit_host` failure injection → confirm event cleanup, (b) pageable-slot path (`pin_failures>0`) → confirm per-copy `cudaEventSynchronize` cost is visible in `experimental_pageable_wait_ms_`, and (c) `synchronize_all`/`reset` under in-flight managed transfers.
