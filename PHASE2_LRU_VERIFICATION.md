# PHASE2_LRU_VERIFICATION.md — second adversarial pass on Luna's rewrite

**Audit target:** `research/phase2-integration-lru-fix` @
`56dad3c1dccb0d7cee774df20d8f0047059d0bd7` ("feat(phase2): integrate plain LRU
tiers and VRAM repair"), base `ca8abd0`.
**Auditor:** T4 (independent verification), worktree `.freebuff/wt/v2b`,
branch `audit/phase2-lru-verification`.
**Prior audit:** `research/phase2-host-tier-verification` @ `e63a3bc`
(`.freebuff/wt/v2a`, `PHASE2_VERIFICATION_AUDIT.md`) — 0C/0H/2M/6L, ACCEPT,
against the original seam impl at `08d3d51`.
**Hardening candidate:** T1's `688bd98` ("fix(phase2): fold tier-verification
audit hardening into rewrite") — reviewed but NOT part of the audit base.

**Verdict: ACCEPT** — 0 Critical, 0 High, 1 Medium (extension of a known
prior-audit Medium), 5 Low, plus confirmed-still-present findings that T1's
`688bd98` already fixes. Recommended campaign candidate is **`688bd98`, not
`56dad3c`** — the hardening closes both live Medium-class holes against this
exact code and its own 149-check suite passes the same build.

## 0. What the rewrite actually changed vs the audited original

The seam implementation files are **byte-identical** to what the first audit
covered: `git diff 08d3d51 56dad3c` on `src/async_prefetcher.cpp`,
`src/expert_tiers.cpp`, `src/vram_cache.cpp`, `include/dee/async_prefetcher.h`
shows **zero** delta. The rewrite's real new surface is:

| New code | Where |
|---|---|
| `PlainLruHostPlacementPolicy` (production host policy) | `host_expert_tier.h:89-96`, `host_expert_tier.cpp:28-37` |
| `Phase2TierConfig.host_enabled` / `vram_priority_fix_enabled` sub-switches | `expert_tiers.h:76-83`, `engine.cpp:3424-3440` |
| `experimental_plain_lru_` flag + precedence over `experimental_score_` and legacy `priority*2^20` | `vram_cache.h:167-216` |
| Engine init wiring (cold adapter, host tier, device tier, null-policy→LRU default, `policy_slots` rejection) | `engine.cpp:3679-3719` |
| `phase2_metrics` arm-dependent reporting | `engine.h:208-219` |
| `prepare_fp4_experts`/`stage_expert` predicate narrowed `enabled` → `enabled && host_enabled` | `engine.cpp:2794, 3153` |

Consequence: every prior-audit finding on the seam mechanics **transfers
unchanged** (the code is literally the same), and this audit's novel ground is
the policy, the switch matrix, the metrics, and the wiring.

## 1. Findings

### MEDIUM — second-live-tier scope replacement is wider than ctor-time exclusivity; also fires on *sequential* replacement (extends prior M2, still live at 56dad3c)

- **Files:** `dee.cpp/src/expert_tiers.cpp:87-100` (ctor guards + unconditional
  dtor clear), `dee.cpp/include/dee/async_prefetcher.h:95-100`
  (`enable_experimental_host_tier` overwrites `experimental_scope_` whenever
  `inflight_` is empty).
- **Defect:** the first audit flagged that a second `DeviceExpertTier` can be
  constructed on a drained `cache_`+`prefetcher_` while the first lives. This
  pass demonstrates the failure is **three distinct holes**, all live:
  1. `enable_experimental_host_tier` *replaces* the armed scope under the live
     tier — afterwards tier-A `stage()` fails closed at `prefetch_host_lease`
     (`-1`), but only **after** `host.acquire` already filled a host slot that
     can never be staged (record stays resident-unservable until evicted).
  2. `~DeviceExpertTier` unconditionally runs
     `cache_.set_experimental_eviction_score({})` — destroying tier-B strips
     tier-A's live eviction policy; legacy `last_used + priority*2^20` scoring
     silently resumes while A still runs.
  3. **New, sharper:** `phase2_device_ = std::make_unique<DeviceExpertTier>(...)`
     — the Engine re-init pattern — constructs the replacement *first*
     (installing its callback), then destroys the old tier, whose dtor clears
     the **new** tier's just-installed callback. The replacement tier is fully
     valid, in-scope, and alive — with its configured `device_policy` silently
     dead. No re-init guard exists (`Engine::init` has none).
- **Reproduction:** `test_phase2_tier_audit_v2b.cpp::second_tier_scope_replacement`
  and `::tier_replace_strips_new_callback` — both PASS, observing the defect:
  stderr shows `score=-2` (callback, evicts newest) before `~B` and `score=1`
  (legacy, evicts oldest) after; the unique_ptr-replaced tier evicts E1
  (legacy) instead of E2 (callback).
- **Reachability:** public C++ API and Engine re-init only. The sanctioned
  single-init Engine flow creates exactly one tier and never re-enters the
  ctor — unreachable in the planned A/B campaign. API-misuse class, same as
  prior M2.
- **688bd98 fixes?** YES for all three: the arming token makes
  `enable_experimental_host_tier` fail while any tier lives, so (1) the second
  ctor throws and (3) the re-init throws (fails closed rather than silently
  stripping); the scope-matched `release_experimental_host_tier` plus
  one-shot token means ~B cannot disturb A. The epoch re-check in
  `stage()`/`wait()` is defense-in-depth on top.

### LOW — the VRAM-repair-only arm inherits the LLP64 `map_key` layer truncation (new arm interaction)

- **Files:** `dee.cpp/src/async_prefetcher.cpp:56-61` (`map_key` conditional on
  `experimental_host_tier_`), `dee.cpp/src/engine.cpp:3679` (vram arm applies
  only `set_experimental_plain_lru`).
- **Defect:** `experimental_host_tier_` is armed only by `DeviceExpertTier`'s
  ctor, which is created only under `host_enabled`. A VRAM-only arm
  (`enabled+vram`, no host) therefore runs the *legacy* `prefetch` path with
  the legacy truncated `map_key` — on LLP64 (Windows/MinGW, `long`==4) every
  layer>0 prefetch coalesces onto layer 0's transfer. Fails closed
  (`wait()` → false; no wrong bytes served) but a multi-layer VRAM-repair A/B
  on LLP64 cannot stage anything above layer 0 — the arm is silently inert.
- **Reproduction:** `vram_arm_inherits_llp64_mapkey` — PASSES, observing
  `b == a`, `!wait(1,5)`, `!is_resident(1,5)` on this LLP64 build.
- **Reachability:** Windows builds only; live campaign targets are Kaggle
  Linux LP64 where the cast is the identity. No effect on the sealed trace
  platform. Flagged because the rewrite *advertises* the arms as
  "independently switchable" — on LLP64 the vram arm is switchable but broken.
- **688bd98 fixes?** YES — `map_key` is unconditional 64-bit there.

### LOW — `Invalid` acquires are invisible in `HostTierStats`; `failures` counts fill events, not `FillFailed` responses

- **File:** `dee.cpp/src/host_expert_tier.cpp:195-196, 234-235, 257-261`;
  `dee.cpp/include/dee/host_expert_tier.h:117-125`.
- **Defect:** (a) an `Invalid` return touches no counter — rejected
  descriptors leave no trace in stats. (b) a coalesced waiter whose joined
  fill fails returns `FillFailed` but only `coalesced` was incremented at join
  — `failures` counts fill *events* (1 failure, N FillFailed responses).
  `host_miss == fills + failures` does hold (verified).
- **Reproduction:** `metrics_invariants_and_gaps` — PASSES.
- **Assessment:** cosmetic; the metrics doc (`PHASE2_METRICS.md:13,22`) already
  hedges `host_miss`/`failures` toward reservations/events. Worth one counter
  or one doc line, not a code fix.
- **688bd98 fixes?** No (untouched there); doc-level only.

### LOW — prefetcher-level experimental accumulators outlive the tier (cross-arm counter carryover)

- **File:** `dee.cpp/include/dee/async_prefetcher.h:264-268`
  (`experimental_host_tier_`, `experimental_scope_`,
  `experimental_readiness_wait_ms_`, `experimental_pageable_wait_ms_` — never
  reset, not even by `reset()`).
- **Defect:** `DeviceExpertTier::metrics` reports
  `prefetcher_.experimental_*_wait_ms_` — if a second tier is ever armed on
  the same prefetcher (post-`reset()`/`clear()` interleaving or Engine
  re-init), its reported wait-ms include the **first** tier's traffic. The
  armed scope/flag persistence itself was prior-audit L3 (deliberate);
  the *metrics* carryover across tier lifetimes is the new edge —
  `PHASE2_METRICS.md:58` claims "a lifetime of the tier/transfer facade" but
  the prefetcher facade outlives tiers.
- **Reproduction:** static (wait-ms accumulators are CUDA-only paths; the
  flag/scope persistence half is exercised by `experimental_flag_persistence`
  and `second_tier_scope_replacement`).
- **688bd98 fixes?** Partially — second-tier arming becomes impossible while
  one lives, but after release a re-arm still inherits the accumulators.
  Per-tier metrics hygiene for multi-arm campaigns only.

### LOW — `Engine::phase2_metrics` reports attempted-config arms, and stale tier pointers survive a disabled re-init

- **File:** `dee.cpp/include/dee/engine.h:208-219`, `engine.cpp:3409`
  (`cfg_ = cfg` before validation), `engine.cpp:3679-3719`.
- **Defect:** (a) the VRAM-arm branch keys on `cfg_.phase2.*`, which is stored
  *before* the gate — a config rejected at the phase2 gate or dead at the
  shard check still reports `device_bytes`/`budget`/tokens as if armed (flags
  were never applied; reads are safe zeros). (b) `phase2_device_`/
  `phase2_host_`/`phase2_cold_` are cleared only inside the host-arm `catch`;
  a subsequent `init()` with phase2 disabled leaves the stale tier objects
  alive, and `phase2_metrics` then delegates to the **dead arm's** counters —
  a counter leak across arm switches if re-init is ever used. Also noted:
  `host_config.slot_bytes < cache_blob_bytes_ → return false`
  (engine.cpp:~3703) sits inside the `try` but bypasses the catch's cleanup —
  `phase2_cold_` stays constructed-but-inert (carried over from prior audit
  INFO, still present).
- **Reproduction:** `phase2_metrics_vram_arm` — PASSES (documents the
  attempted-config semantic); the stale-pointer half is static-only (tier
  creation needs a full init: real shards + CUDA).
- **688bd98 fixes?** No; cosmetic/diagnostic-surface only.

### LOW — explicit `PlainLruHostPlacementPolicy` + `policy_slots > 0` bypasses the engine's stranding guard

- **File:** `dee.cpp/src/engine.cpp:3701-3708` — the `policy_slots != 0`
  rejection runs only under `if (!host_policy)` (the null → default-LRU path).
  A caller explicitly passing `make_shared<PlainLruHostPlacementPolicy>()`
  with `policy_slots > 0` skips the guard and gets a permanently unfilled
  policy partition (plain LRU classifies everything `Dynamic` → policy slots
  can never fill → silent capacity loss).
- **Reproduction:** `plain_lru_policy_slots_stranded` demonstrates the
  tier-level stranding; the gate-bypass is static (engine path needs full
  init). Config-misuse class — the guard exists precisely for the default.
- **688bd98 fixes?** No.

### LOW — `device_hit` counted before the prefetch attempt returns; hit-path failures skip `device_failures`

- **File:** `dee.cpp/src/expert_tiers.cpp:113-125` — `++metrics_.device_hit`
  precedes `prefetcher_.prefetch(...)`; if it returns `<0`, `stage()` returns
  false **without** `++metrics_.device_failures` (the miss path at 136/145
  does increment it). Asymmetric accounting; in practice unreachable — a
  resident-block `prefetch` cannot fail (`ensure` on resident + `pin` are
  infallible for a present block). Doc-level consistency note for
  `PHASE2_METRICS.md:15,23`.
- **688bd98 fixes?** No.

## 2. Prior-audit findings confirmed still present at 56dad3c

The seam files are unchanged, so all prior findings transfer; status vs T1's
`688bd98`:

| Prior finding | Present at 56dad3c | 688bd98 |
|---|---|---|
| M1 LLP64 `map_key` truncation | YES (legacy + vram-only arms) | Fixed (unconditional 64-bit) |
| M2 ctor-time-only scope exclusivity | YES — extended above | Fixed (arming token + epoch) |
| L3 stale armed flag + flag-keyed `terminate` in `synchronize_all` (`async_prefetcher.cpp:624`) | YES | Fixed (re-keyed on managed+leased in flight) |
| L4 `cuda_submit_host` `!ok` event leak (`async_prefetcher.cpp:749-755`) | YES | Fixed (destroy+null after drain) |
| L5 pageable-fallback host block | YES | Kept by design |
| L6 `lease.key()` borrowed reference | YES | Documented |
| L7 same-key/different-size `Invalid`-until-evict | YES | Documented |
| mmap-truncation crash surface | YES | Platform caveat |

One extra ctor-hardening gap T1 also closed: at 56dad3c, if
`set_experimental_eviction_score` throws inside `DeviceExpertTier`'s ctor
(bad_alloc on the `std::function`), the prefetcher is left armed with no tier
— T1 wraps it in try/catch and releases the token.

## 3. Verified-correct in the new code (adversarial pass results)

All verified by `test_phase2_tier_audit_v2b.cpp` (260 checks, 0 failures,
MinGW GCC DEE_CUDA=OFF):

- **Plain LRU policy correctness.** `victim` = min-`last_use` over the offered
  set with slot-index tiebreak (`host_expert_tier.cpp:28-37`); every key is
  `Dynamic`. Tick semantics: `last_use = ++tick` stamped at fill-start and per
  hit (incl. successful coalesced waiters); recycled slots carry their new
  record's fresh stamp — observed eviction order exactly LRU under hit/fill/
  evict/recycle interleavings (`plain_lru_victim_unit`,
  `plain_lru_tier_ordering`, `plain_lru_recycle_and_evict_ticks`).
- **Eviction-under-lease and under-fill.** Offered candidates are gathered as
  `{Ready, references==0, Dynamic}` only (`host_expert_tier.cpp:221-227`):
  leased records survive arbitrary eviction pressure without a re-read, and a
  `Filling` slot is never offered mid-fill
  (`plain_lru_never_offers_filling_or_leased`, `plain_lru_leased_never_offered`,
  `metrics_invariants_and_gaps` — the Capacity-under-lease case).
- **Hostile policy fail-closed.** A `victim()` answer outside the offered set
  — including naming a *real but unoffered* (leased) slot — degrades to
  `Capacity`, no eviction (`hostile_victim_out_of_set`; `any_of` revalidation
  at `host_expert_tier.cpp:230-231`).
- **8-combo switch matrix.** `enabled × host_enabled ×
  vram_priority_fix_enabled` enumerated at `Engine::init`: 000 dies past the
  gate at the shard check (default-OFF); 001/010/011 rejected
  ("subfeatures require phase2.enabled"); 100 rejected ("requires host_enabled
  or vram_priority_fix_enabled"); 101/110/111 pass the phase2 gate. Host-arm
  sub-requirements (model_identity, `use_cuda`, FP4 cache+transfer dtype) each
  fail closed individually (`engine_subswitch_matrix`, engine.cpp:3424-3440).
- **Arm interaction verified at the seams:** `prepare_fp4_experts` and
  `stage_expert` now predicate on `enabled && host_enabled` (engine.cpp:2794,
  3153) — a VRAM-only arm correctly falls through to legacy staging rather
  than the tier path. `set_experimental_plain_lru` is applied outside the host
  block (engine.cpp:3679) so 101 and 111 both get the repair.
- **`experimental_plain_lru_` precedence.** Flag outranks both an injected
  `experimental_score_` callback and the legacy priority term; toggling off
  restores callback authority mid-run (`vram_plain_lru_precedence`,
  `vram_cache.h:212-215`). Undocumented-but-correct interaction noted: under
  `vram_priority_fix_enabled`, a configured `device_policy`'s
  `eviction_score` is shadowed (its `priority()` still feeds `block.priority`,
  which the flag then ignores) — consistent with "the repair is the policy".
- **Metrics arm correctness.** `phase2_metrics`: never-initialized → empty;
  disabled config → empty; VRAM arm → device arena readout + token
  denominators + `bytes_per_token_valid` honoring zero; rejected host arm →
  no spoof. Tier arm → delegates to `DeviceExpertTier::metrics` incl. live
  `host.stats()`. Per-token rates divide by caller-supplied `tokens` only
  (`phase2_metrics_vram_arm`, engine.h:208-219, expert_tiers.cpp:155-169).
- **Counter consistency.** `host_miss == fills + failures` holds under mixed
  fill/fail/capacity traffic; duplicate `stage()` for an in-flight key is a
  `device_hit` with zero extra `H2D_bytes`/host fill (one miss + one hit +
  one fill + one store read total — `stage_dedup_metrics`);
  `device_failures`/`budget_rejections`/`evictions`/`resident_bytes`/
  `peak_resident_bytes` all reconcile under stress.
- **Device-side pin protection.** A pending managed transfer's block holds
  `pins==1` and survives `ensure()` eviction pressure — second stage on a
  1-block cache fails closed (stderr: `evict_until_free ABORT ... pinned`)
  rather than evicting in-flight DMA destination; retry post-`wait()` evicts
  the released block and serves from the retained host record with zero
  re-read (`managed_pin_blocks_eviction`).
- **Exception/failure cleanup in new code.** Engine init `try/catch` resets
  all three phase2 unique_ptrs on any throw (adapter/host/device ctor or the
  policy_slots guard) and returns false (engine.cpp:3694-3718);
  `HostExpertTier` ctor partial-failure unwinds allocated slots via
  `~HostTierState`; store fail/throw/short/lied all publish `FillFailed` with
  balanced references; codec `materialize` exceptions are caught and become
  `FillFailed` (`host_expert_tier.cpp:250-251`). `IdentityCodec` forces
  `bytes_read == exact_bytes` (lying store → FillFailed).
- **Claimed guarantees vs code.** "No per-request payload
  allocation/copy": slots preallocated once; `acquire` allocates nothing on
  the data path (candidate vector and key capacity reserved at ctor); fill
  writes directly into the final slot (adapter non-contiguous gather or
  `materialize` into slot), verified exact. "Leases protect DMA lifetime":
  `transfer.host_lease` released only after `dma_complete` is *proven* (event
  query/sync, stream drain, or mock memcpy). Generation checks:
  `discard_unpinned` double-guards generation AND `pins==0`; host slot
  `generation++` only at fill-start with `references==0`. Scope checks:
  `prefetch_host_lease` rejects foreign model/representation/empty leases;
  `DeviceExpertTier::accepts` is model+representation-scoped.
- **Concurrency.** 8-way same-key race → exactly one store read/fill; 4-thread
  LRU churn → no torn leases, no unexpected statuses; coalesced joiners
  observe their own generation's failure and must retry
  (`concurrent_stress`, `plain_lru_concurrent`,
  `fill_failure_publication`, `metrics_invariants_and_gaps`).

## 4. Doc-claim accuracy (PHASE2_HOST_TIER_DESIGN.md, PHASE2_METRICS.md)

Checked line-by-line: "fail closed, never silently substitute" (verified at
every gate), "least-recently-used idle slot" (exact), "Dynamic eviction
candidates contain only ready, unleased dynamic slots" (exact — the gather
predates the policy call and revalidates its answer), "no implicit 8-GiB
allocation" / "budget = slots ≤ budget/stride incl. padding" (exact),
"callback under host mutex, must not reenter" (documented; a reentrant policy
would deadlock — contract-class), "no new cudaDeviceSynchronize" (none
added), "device_gpu_wait_ms empty means UNKNOWN" (`std::optional` never set —
accurate). Gaps found: `Invalid` invisibility and the tier-lifetime wording
for prefetcher accumulators (Lows above); no overclaims.

## 5. Test status

- `test_phase2_tier_audit_v2b` (this suite): **260 checks, 0 failures** —
  all first-audit checks ported (130) + 130 new: LRU policy/tick/offered-set
  suite, hostile-victim set, stranded policy partition, LRU concurrency, plain-
  LRU precedence incl. flag toggle, 8-combo + fail-closed matrix,
  phase2_metrics arm suite, second-live-tier scope replacement, unique_ptr
  callback strip, vram-arm LLP64 inheritance, pin-vs-evict, dedup metrics,
  accounting invariants, adapter identity bound.
- Full `ctest` (MinGW GCC 15.2, `DEE_CUDA=OFF`): **15/15 C++ suites PASS**;
  4 Python tests fail on missing `pytest` (pre-existing, environmental — same
  as the first audit and T1's run).
- CUDA paths (`cuda_submit_host`, event/lease reclamation, pageable fallback)
  reviewed statically only — unchanged from the audited original except the
  known L3/L4 items; first live run should still exercise them per the prior
  audit's go/no-go list.

## 6. Findings by severity

| Severity | Count | Items |
|---|---|---|
| Critical | 0 | — |
| High | 0 | — |
| Medium | 1 | Second-live-tier scope replacement incl. sequential-replacement callback strip (extends prior M2; API/re-init only) |
| Low | 5 | vram-only arm inherits LLP64 map_key; Invalid acquires invisible + `failures` counts events not responses; prefetcher accumulators carry across tier lifetimes; attempted-config metrics + stale tier pointers on disabled re-init; explicit-LRU+`policy_slots` gate bypass; asymmetric `device_hit`/`device_failures` |

## 7. Verdict rationale

**ACCEPT** the rewrite as a campaign candidate — with the explicit
recommendation that the shipped candidate be **`688bd98`** (hardened head),
not `56dad3c`: every Medium/Low found live at 56dad3c is either closed there
(scope token, unconditional map_key, event destroy, terminate re-key) or is a
doc/metrics nit. Under the sanctioned single-init Engine flow, nothing found
here corrupts bytes, misroutes, or breaks exactness; the plain-LRU host policy
and the three-switch arm matrix are implemented and verified as designed.
The residual risks are the same ones the first audit carried: ctor-time-only
exclusivity reachable only outside Engine, LLP64 divergence on Windows-only
arms, and CUDA-path items that need the live T4 run to close out.
