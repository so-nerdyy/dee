# Phase-2 Tier Hardening — finding → disposition

**Branch:** `audit/phase2-tier-hardening` (worktree `.freebuff/wt/p2h`), base
`research/phase2-integration-lru-fix` @ `56dad3c1` (Luna's rewrite).
**Source audit:** `research/phase2-host-tier-verification` @
`e63a3bc7c1001ce9343da1e41c09e09f4489d296`, report `PHASE2_VERIFICATION_AUDIT.md`
(worktree `.freebuff/wt/v2a`): 0 Critical / 0 High / 2 Medium / 6 Low, verdict
ACCEPT. This document maps every audit finding to its disposition in the
rewrite and records where the audit's suggested mechanism did **not** transfer
cleanly.

**Verification:** MinGW GCC 15.2, `DEE_CUDA=OFF`, `cmake --build` clean;
`ctest`: 15/19 pass — all 13 C++ binaries (including the ported+extended
adversarial suite `test_phase2_tier_audit`, 149 checks, 0 failures) plus the
two pytest-free Python tests. 4 Python tests fail with `requires pytest` —
pre-existing environmental issue, identical to the audit's own run.

---

## Findings applied (code changes)

### M1 — LLP64 `map_key` layer truncation → **FIXED unconditionally**
- **Audit:** `async_prefetcher.cpp:56-60` computed
  `experimental ? key : (uint64_t)(long)key`; on Windows/LLP64 the legacy OFF
  path discarded the layer bits, so `prefetch(0,5)` then `prefetch(1,5)`
  silently coalesced (fail-closed — `wait(1,5)` returned false — but layer>0
  experts could never stage through the legacy path on Windows).
- **Fix:** `map_key` now returns the full 64-bit `key_id` on **every** path
  (`src/async_prefetcher.cpp:56-68`). The conditional is gone; there is no
  truncating form left.
- **Legacy-OFF decision:** the truncating form was *not* retained. Rationale:
  the "default-OFF equivalence" the audit preserved only ever existed on
  LP64, where `static_cast<long>(key)` is the identity — all sealed Phase-1
  evidence is Linux/LP64. Keeping the conditional preserved a Windows-only
  divergence, not equivalence. `key_to_idx_`/`batch_keys_` are internal state
  (no wire/persistent format), so widening the key is safe for the legacy
  path, and it removes the coupling where experimental-flag persistence was
  load-bearing for key width (audit L3's caution). Bonus: `batch_keys_`
  duplicate accounting is now correct on LLP64 too.
- **Test:** `mapkey_full_width_unconditional` (formerly
  `llp64_layer_collision`) now asserts `b != a`, `wait(1,5)`, and residency
  on **all** platforms — the pre-hardening MinGW collision is gone.

### M2 — `DeviceExpertTier` scope exclusivity was ctor-time only → **FIXED**
- **Audit:** the ctor required an empty cache + drained prefetcher, but a
  second tier could be constructed later on the same `cache_`+`prefetcher_`
  (e.g., after `prefetcher_.reset()` + `cache_.clear()`, both public and
  reachable via `Engine::reset_runtime_cache`). `enable_experimental_host_tier`
  then overwrote `experimental_scope_`, and tier-1's resident-hit `stage()`
  path could observe blocks staged under the foreign scope — silent
  cross-model byte confusion.
- **Fix (adapted to this codebase — see note below):**
  - `AsyncPrefetcher::experimental_tier_live_` — one-shot arming token taken
    by `enable_experimental_host_tier`, released **only** by
    `~DeviceExpertTier` via `release_experimental_host_tier(scope)`
    (scope-matched). A second live tier on the same prefetcher fails at
    construction under *every* interleaving — clean prefetcher, post-drain,
    and post-`reset()`+`clear()` alike.
    `include/dee/async_prefetcher.h:117-130`.
  - `AsyncPrefetcher::scope_epoch_` — armed-scope generation bumped on every
    successful arm; `DeviceExpertTier` captures it at construction and
    `stage()`/`wait()` fail closed on mismatch
    (`src/expert_tiers.cpp:96,126,169`). Defense in depth: while the token
    works, no live tier can observe a moved epoch.
  - `~DeviceExpertTier` releases the token; a `try/catch` around the
    eviction-score install prevents the token stranding if the ctor throws
    after arming (`src/expert_tiers.cpp:97-109`).
  - `enable_experimental_host_tier`/`release_experimental_host_tier` are now
    private with `friend class DeviceExpertTier` — the tier ctor is the only
    arming path, so the token can never be held without a live tier to
    release it.
- **Audit-mechanism adaptation (verified against Luna's code):** the audit
  suggested "a scope epoch bumped by `reset()`". That does **not** transfer:
  `AsyncPrefetcher::reset()` is a routine drain invoked while the tier is
  legitimately armed — `Engine::reset_runtime_cache` (engine.cpp:2170),
  `preload_all_experts` every 64 experts (engine.cpp:3321/3324), and
  `generate()` teardown (engine.cpp:3848). Bumping the epoch there would
  brick the armed tier in all three flows. Instead the epoch counts
  *armed-scope generations* — ownership moves only when a new scope is armed,
  which the token already serializes. Consequently the stale-scope rejection
  is unreachable via the public API while the token holds (by construction —
  the confused state cannot be built rather than merely detected); the epoch
  check guards against future arming paths.
- **Also fixed as a side effect:** on Engine re-init,
  `phase2_device_ = std::make_unique<DeviceExpertTier>(...)` evaluates the
  new ctor **before** destroying the old tier. Previously the new ctor
  succeeded and the old dtor then cleared the *new* tier's eviction-score
  callback — silent policy loss. Now the second ctor throws, the catch block
  destroys the old tier, and init fails closed and cleanly.
- **Test:** `device_tier_scope_exclusivity` (15 checks): second-live-tier
  rejection with and without the reset+clear interleaving, owner tier
  survives routine `reset()` (engine-flow preservation), observable epoch
  bump on re-arm, stale-scope lease rejected post-re-arm.
  `device_tier_ctor_guards` retained.

### L3 — armed state outlives the tier; `synchronize_all` fail-stop keyed on flag → **FIXED (fail-stop re-keyed) + documented**
- **Audit:** after `~DeviceExpertTier`, `experimental_host_tier_` persisted;
  `synchronize_all`'s stream-sync-failure path called `std::terminate()` on
  flag state alone — a stale armed flag upgraded a legacy drain failure to a
  process abort with zero managed transfers in flight.
- **Fix:** the fail-stop now scans `inflight_` and terminates only if a
  transfer still holds a host lease (`managed_source && host_lease` — the
  exact set whose DMA completion is unprovable and whose memory cannot be
  freed safely). No leased sources in flight → legacy `return` path
  (`src/async_prefetcher.cpp:631-642`). `DEE_CUDA`-only code; verified
  statically, not executable in this environment.
- **Documented contract:** the armed flag + last scope intentionally persist
  after `~DeviceExpertTier` — one-shot-per-prefetcher semantics. Re-arming
  under a new scope is legal only after the token is released (header comment,
  `include/dee/async_prefetcher.h:107-116`).
- **Test:** `experimental_flag_persistence` retained and updated — asserts
  post-teardown lease acceptance, epoch stability, and (now universal)
  full-width keys.

### L4 — `cuda_submit_host` failure path leaked the CUDA event → **FIXED**
- **Audit:** event created at submit, assigned to `transfer.event`; on
  submit/record failure the drain ran but the event lingered on the abandoned
  transfer until `reset()` — a bounded event-object leak per failed submit.
- **Fix:** in the `!ok` branch, after the mandatory stream drain proves no
  in-flight work can signal it, `DEE_TA_EVENT_DESTROY(event, "phase2_event")`
  destroys the event and `transfer.event` is nulled
  (`src/async_prefetcher.cpp:769-776`). Trace-alloc bookkeeping stays paired
  (the event was recorded at creation).
- **Note:** legacy `cuda_submit` has the same pattern (event at :875, no
  destroy on the :887-926 failure returns). Left unchanged deliberately —
  legacy callers never drain on failure, the convention is
  consistent-but-incomplete, and touching six return sites is outside this
  finding's scope. Recorded as known-gap below.
- **Test:** not CPU-mockable — the mock stream has no submit-failure
  injection point (`use_cuda_` short-circuits before any submit call). Static
  review only; the first live CUDA run should exercise failure injection.

## Findings documented as contract (no behavior change)

### L6 — `HostExpertLease::key()` returns `const&` into pooled slot memory
- Documented "borrowed for lease lifetime" on the accessor
  (`include/dee/host_expert_tier.h:142-146`): valid while the lease is held;
  do not store past `reset()`/destruction/move — same class as
  `HostVictim::key`. Verified safe: `references > 0` forbids slot reuse.

### L7 — same-key record with different `exact_bytes` → permanent `Invalid` while resident
- Documented at the check (`src/host_expert_tier.cpp:201-205`): deliberate
  fail-closed semantic; `evict(key)` withdraws the resident copy, then the
  new size may fill. **Do not "fix" into silent re-fill.**
- **Test:** `store_guard_matrix` extended — after `evict()`, the same key
  with the new size fills successfully (the `Invalid` is recoverable, not
  permanent).

### L8 — mmap materializers crash (AV/SIGBUS) on externally truncated files; `pread` path fails cleanly
- Pre-existing platform caveat, unchanged: Windows/base/Safetensors
  `materialize` copies from the mapped view — a file truncated after mapping
  raises an access violation; the Linux DEE4 `pread` loop fails cleanly at
  EOF. `materialization_mode()` reports `mmap_memcpy`/`pread` accurately.
  `expert_store.*` is outside this task's scope; the design doc already
  qualifies the Windows path ("copies its mapped record once"). Hardening
  option if ever wanted: `ReadFile` on the file handle instead of view
  memcpy.

### L5 — pageable (unregistered) host slots block inside `cuda_submit_host`
- Kept: honest fallback, metric'd via `experimental_pageable_wait_ms_`.
  Throughput cliff on pin-starved hosts, not a correctness bug. Carried
  suggestion: surface `stats().pin_failures` in engine-level init logs.

## Findings deliberately not changed

| Audit item | Disposition |
|---|---|
| `synchronize_all`/`cuda_submit_host`/`host_expert_tier` `std::terminate` on unprovable drains | Keep — deliberate fail-stop; only re-keyed per L3. Legacy branch asymmetry (returns instead of aborting) left as-is. |
| Engine init ordering quirks (tier ctor before `prefetcher_.init`; budget check bypassing catch cleanup) | Unchanged — fails closed, brittle only on re-init which already fails closed today. |
| `HostTierStats::SSD_bytes` counts lying stores' reported `bytes_read`; `leased_slots` counts Filling; `H2D_bytes` at submission | Accounting caveats, verified consistent — cosmetic, unchanged. |
| Fill waits without timeout | Inherent to the no-cancellation design — unchanged. |
| Legacy `cuda_submit` event-leak parity | Known gap, unchanged (see L4 note). |
| `DevicePlacementPolicy::eviction_score` callback-reentrancy contract undocumented on the device side | Recorded here: like `HostPlacementPolicy`, the callback runs inside `ensure → evict_until_free`; a policy that reenters `cache_` recurses into eviction. Contract class noted, no code change. |

## Test counts

| Suite | Checks | Result |
|---|---|---|
| `test_phase2_tier_audit` (ported + extended) | 149 `check()` assertions | **PASS** (was 130 on the audit branch; +19 from `mapkey_full_width_unconditional`, `device_tier_scope_exclusivity`, post-evict refill, epoch persistence) |
| Full C++ ctest set | 13 binaries | 13/13 PASS |
| Python tests | 4 | FAIL — `requires pytest` (environmental, pre-existing) |
| `test_phase2_cuda` | — | not built (DEE_CUDA=OFF; no GPU in env) |
