# Phase 4 Kickoff — Audit Synthesis & Campaign Baseline

Branch `research/phase4-cache-hierarchy` @ `a817755` (worktree `.freebuff/wt/p4`).
Synthesis of the Wave-A audit swarm (9 read-only agents) + supervisor route-journal
reuse study + local MinGW build/test gate (19/19 ctests PASS).

Evidence labels follow AGENTS.md: VERIFIED-MEASURED / VERIFIED-INSPECTED /
INFERRED / HYPOTHESIS / ESTIMATE.

---

## 1. What Phase 4 actually starts from (VERIFIED-INSPECTED)

The bounded hierarchy already exists and ran during the Phase-3 GPU-2 PASS:

```
Dee4ExpertStore (146.6 GiB, 46 segments, pread materialize)
  -> HostPackCache (bounded LRU, byte budget, multi-lane batch fill)
  -> AsyncPrefetcher (pinned ring, event-tracked H2D, dedicated stream)
  -> VramCacheManager (fixed cudaMalloc arena, pin/generation-safe)
  -> FP16 SwiGLU compute
```

GPU-2 committed config: `cache_dtype=fp16` (74 slots/GPU), host pack 8.5 GiB/GPU
(682 records), lanes=4, queue_depth=6. Measured: VRAM hits 2.9-5.2%, host-pack
hits **48.9% process-wide** (8,017/16,398 requests), store reads 8,381 rec /
104.35 GiB total.

**Correction to Phase-3 headline (A1):** the RESULTS.md 699.9s-vs-472.8s
comparison includes ~198 s of a0-only P2.2 repack work. Honest generate-only
delta: a0 373.26 s vs a1 341.24 s = **+9.4%**, and per-record read p50
75.7-83.6 ms vs 63.9-67.2 ms. Phase-3's real claim is exactness + a modest
read-path win, not 48%.

## 2. Correctness defects found by the audits (must repair before any arm)

Ranked. All VERIFIED-INSPECTED by >=1 agent; none observed firing on the sealed
evidence (they are latent), all are silent-exactness hazards under Phase-4's
higher-hit-rate regimes.

### R1 — staging-slot lifetime race (A2, CRITICAL)
`AsyncPrefetcher::wait_on_stream` marks a pinned staging slot free when the
compute-stream wait is *armed* (`cudaStreamWaitEvent`), not when the H2D has
run (`async_prefetcher.cpp:559-579`). A later `cuda_submit` reuses the slot and
the host-side 6-region gather overwrites the pinned buffer while the still-
queued device memcpy reads it -> the pending H2D lands the *new* expert's bytes
in the *old* expert's block; the armed event then correctly orders compute over
corrupted data. Applies to every fp4 transfer. Thrash regime deepens the
prefetch queue -> larger window. Fix: hold `busy` until event completion is
observed (retire pass), or make slot reuse stream-ordered.

### R2 — HostPackCache::get() fill-poisoning (A3/A10/A6, triple-confirmed)
`get()` inserts the entry unconditionally after a `void` fill
(`host_pack_cache.cpp:146-157`). Engine signals failure out-of-band
(`fill_ok` + memset 0) for the *current* call, but the zeroed ready entry
stays cached and is served as a hit forever after -> silent zeroed-weight
compute. `get_batch` is correctly fail-closed; the lanes=1 path is not.
Fix: `bool` fill channel + rollback-on-false + same-key/different-size guard
on the hit path.

### R3 — pre-sync unpin in the live path (A2/A6, double-confirmed)
`moe_forward_batch_device_impl` unpins at `engine.cpp:1054` immediately after
enqueue, before the call-end `cudaStreamSynchronize`. When unique experts in
one call exceed `cache_batch` (74 fp16 / 281 fp4 — reachable at prefill), a
later chunk's `ensure` can evict a block whose GEMM is still queued ->
prefetch-stream H2D overwrites a live compute source. Sibling paths all unpin
after sync or via deferred event (`drain_pointer_batch_pending`). Fix: defer
chunk unpins behind a per-call completion event (mirror the pointer-batch
pattern).

### R4 — multi-victim eviction drops all but the last victim (A2)
`evict_until_free` overwrites `last_ensure_info_.evicted_key` per iteration;
`prefetch_impl` erases only that one key from `key_to_idx_`. Stale entries for
earlier victims -> `validate_invariants` false-failures under
`debug_validate_cache` + wrong generation attribution in the trace. Fix:
`EnsureInfo` carries a victim *list*; prefetcher erases all.

### R5 — supporting defects (smaller, same workstream)
- `cuda_submit` failure leaves a resident uninitialized block that later
  hits serve (A2 M7) -> invalidate the block on submit failure.
- LLP64: `key_to_idx_`/`batch_keys_` are `long` (32-bit on Windows/MinGW) ->
  `(layer<<32)|expert` truncates; port Phase-2 fix to `int64_t`/`size_t`
  (A10 P2, A12 #1).
- `ensure(nbytes > budget)` drains the entire cache before failing
  (A12 #2) -> early-reject before evicting.
- `ensure`/`get` hit paths ignore `nbytes` (A12 #3, A3 L1) -> fail closed on
  same-key/different-size.
- `avail_layer` silently aliases unresolvable layers (incl. store buckets
  43-45 = MTP) to `base_layer` -> wrong-expert service (A3 M1, A4 D4, A6 D6).
  Latent for greedy (layers 0-42 only); must fail closed or be gated before
  any MTP/speculative path.
- `evict_until_free` counts `pinned_blocks_skipped` once per scan iteration,
  not per block (A2 M5) -> counter semantics fix while in there.
- `inflight_`/events are never reclaimed until `reset()` (A2 M6) -> leak
  per cold request over long decodes; add retire-with-completion.

## 3. Eviction-policy verdict (A8 supervisor study + A2/A10)

- `priority` arriving at `ensure()` is *request position* — in the live
  batched path it is **descending expert-ID order** (A2 Q1). One rank step =
  1,048,576 ticks ~= 4,000 tokens of recency. The effective Phase-3 policy was
  "keep low expert IDs; LRU tiebreak" — an arbitrary sticky cohort, not an
  oracle signal.
- The 3.4% fp16 hit rate is primarily capacity, not policy: LRU replay at 74
  slots predicts ~0% (reuse-distance p50 is ~130-270 events). At the fp4
  capacity (281 slots > ~132 touches/token/GPU) the priority artifact *will*
  decide which blocks persist — the repair matters there.
- Repair (agreed with Phase-2 audit): pure-recency LRU behind a runtime flag
  `gpu_evict = lru | rank_priority` (default `lru` once validated; flag keeps
  the legacy arm available for A/B). Lockstep updates required:
  `test_vram_cache.cpp:75-86`, `scripts/deepseek_v4_cache.py` mirror,
  `tests/test_deepseek_v4_cache.py` (mirror==engine invariant).
- Validation band from the sealed Phase-2 trace at 281 slots (A10 P1):
  repaired policy should land `resident_hits ~= 680 (cuda0) / 1057 (cuda1)`,
  `cold_loads ~= 1933/1429`.
- Supervisor reuse study on the 3 Phase-3 journals: LFU/hybrid/layer-aware
  gain <=4pp over LRU; admission-filtered LRU is *worse* (-2 to -13pp);
  offline MIN sits ~14-19pp above LRU at 281 slots but the gap is mostly
  future-knowledge (single-use experts). **No exotic policy is justified;
  capacity + pure recency are the levers.** Re-confirm on the 128-token G1
  trace via the sim.

## 4. Reuse economics (VERIFIED-COMPUTED from Phase-3 journals)

- Per (forward,layer) the engine dedups across token rows: ~126-132 unique
  experts/forward/device; repeat-from-any-prior 51-64%; novelty keeps
  arriving (~33-49 new/forward late in decode) — the working set grows
  through a 128-token workload; a steady-state miss floor exists.
- Cross-prompt: 44-56% of each prompt's unique experts overlap another
  prompt's; union over 3 prompts ~= 4,763 experts ~= 59 GiB total
  (~28-31 GiB/device) — a persistent host cache earns real inter-prompt
  hits (confirmed live: q1/q2 prefills scored 105/200 host hits).
- Sim fidelity vs measured GPU-2: host 40-52% predicted vs 48.9% measured;
  VRAM ~0% LRU predicted vs 3.4% measured (excess = the priority artifact's
  accidental sticky cohort). Sim is trustworthy within a few pp.
- Capacity knee: ~128-281 device slots is where reuse distance concentrates
  — exactly the fp4-packed range (281 slots @ 3.5 GiB vs fp16's 74).
- Host budget knee: ~17-19 GiB/GPU covers one prompt's working set;
  ~28-31 GiB/GPU approaches the 3-prompt union ceiling.

## 5. FP4 packed residency (A9)

- Path is **complete and fail-closed end-to-end**; v60 evidence already shows
  `cache_dtype: fp4-packed`, `ACCEPT_CORRECTNESS`, 281 resident experts on a
  sealed 16-token gate (VERIFIED-MEASURED, Phase-2 line).
- Bit-identical vs FP16 residency: same `fp4_e2m1_to_f16_cuda` decode kernel
  on the same gathered bytes; only the stream differs (prefetch-stream
  decode-per-miss vs compute-stream decode-per-use).
- Cost: ~65 ms/token decode-on-compute on T4 (~0.25 ms x 43 x 6) — ~1.4% of
  wall at Phase-3 pace; pays only on hits the capacity actually creates.
- Missing gates (Wave B): device-block byte-identity readback test, A/B
  bitwise gate (Fp16 vs Fp4E2m1 arms on the same expert), eviction-churn
  determinism, batched-group-under-fp4, decode-scratch bounds check
  (`engine.cpp:3266-3281` has none), scratch added to the device ledger.
- `dee_cli` cannot arm fp4 residency (`main.cpp:93-103`) — pydee-only today;
  fine for the campaign (all arms run through the runner).

## 6. Telemetry — what exists vs the Phase-4 event contract

Already plumbed (A6/A7): `RequestTraceRecord` carries
token/layer/expert/kind/occupancy/evicted/generation/pin/reuse_distance/
consumed/evicted_before_use; serialization exists and lands in
`external_profile_json["trace"]`. `FillBatchRecord`/`FillRequestSample`
decompose batch fills. `SpanProvenance` marks measured-vs-derived.

Missing links (the keystone build — now smaller than feared):
1. `RequestKind` cannot express HOST_HIT vs STORAGE_MISS — `ColdLoad`
   conflates both. Minimal fix (A6 recipe): capture the pack outcome in
   `get_staging_fp4` (hit / fill-executed / mmap-fallback), thread it into
   `prefetch_impl`, emit `HostHit` when VRAM missed but pack held.
2. `trace_requests` is never enabled by the runner — add
   `NATIVE_TRACE_REQUESTS=1` -> `cfg.trace_requests` (requires
   `NATIVE_PROFILE=1`), plus a durable `cache_events{suffix}.jsonl` sink,
   integrity-hash it, and have the driver retrieve it.
3. Token attribution: `current_token_` stays -1 on the whole native path —
   the runner must call `eng.set_external_token(step)` per forward (binding
   already exists, `pydee.cpp:192`).
4. Victim lists per eviction event (R4 makes this required anyway).
5. Host-side per-access fields: pack occupancy snapshot + evicted identity +
   fill `resident_bytes`/cold flag (store-side `mincore` is aggregate-only;
   extend `FillRequestSample` or return residency from `materialize`).
6. The `get()` single-lane path emits no fill records at all — instrument
   parity with `get_batch` (R2 repair touches this anyway).
7. Per-prompt counter contamination (A1 A1 / A7 #3): `host_pack` and
   `expert_store` stats are process-cumulative while `previous_totals`
   re-zero per prompt — snapshot cumulative counters at prompt entry and
   seed `previous_totals` (or add reset APIs).
8. Counter additions from Phase-2 port list (A10 P7): `budget_rejections`,
   `host_capacity_wait_ms`, `device_enqueue_ms`/`device_host_wait_ms`,
   `bytes_per_token` with a `bytes_per_token_valid` flag, `optional<double>`
   for unmeasured GPU-side waits, `waited`/`fallbacks` parity for
   `wait_on_stream`, accounting-closure invariants
   (`hits+misses==requests`, `requests==resident+inflight+cold`,
   `SSD_bytes==fills*record`).

## 7. Budget semantics (A5) — the "bounded" claim is currently soft

- **No total-RSS ceiling exists anywhere.** The runner's "17 GiB" caps only
  pack payload; pinned ring (~816 MiB/engine), pinned staging (<=192 MiB),
  staging maps, temp vectors, page cache, and ~6 GiB Python/Torch baseline
  are all outside it -> explains the historical 90-95% RSS kills.
- **No total-VRAM ceiling.** `budget_bytes` caps only the arena; ~3 GiB of
  non-arena device use (fp4 scratch ~50 MiB, MoE workspaces, router, torch
  dense ~2.5 GiB, context) rides on top with no admission check.
- Zero-budget footguns: `host_pack_cache_bytes=0` -> silently 8 GiB;
  `budget_bytes=0` -> silently 4 experts. An honest "cache-off" arm does not
  exist.
- `DEE_RELEASE_MMAP_PAGES` is a no-op for dee4 (range-check only covers
  safetensors mmaps) *and* the runner force-disables it. Page-cache
  double-booking is real on local-disk profiles; on the Kaggle FUSE mount
  mincore shows ~0.2% residency, so it is moot there.
- `reset_runtime_cache` frees almost nothing; `HostPackCache::clear()` has
  no pydee binding; `ExpertStore` stats have no reset.

Phase-4 additions (Wave B):
- Configure-time admission check: `pack + ring_bytes + pinned_cap + baseline
  <= host_total_budget` fails at configure, not at OOM-kill.
- `NATIVE_CACHE_RESET=cold|warm` per prompt boundary; pydee `clear_host_pack`
  binding + store-stats reset.
- Honest bypass arm: `host_cache_mode = lru|bypass` where bypass = pack
  budget clamped to exactly 1 record with lanes forced to 1 (a bounce
  buffer — avoids both caching and the FUSE mmap-fallback death path); GPU
  "off" = 1-block arena (every ensure evicts+reloads).
- `ExpertStore::release_source_pages(view)` ->
  `posix_fadvise(DONTNEED)` on the segment range (fixes the dee4 page-cache
  gap; local-disk profiles only).
- tiny/small/medium profile presets (A5 suggested starting points recorded
  in the campaign driver).

## 8. Harness/driver deltas for the campaign (A7)

- Keep `NATIVE_RUN_ID` constant across arms and keep the same device split —
  `run_id` and `device` are inside the journal hash payload; single-GPU arms
  can never hash-compare to dual-GPU journals.
- Add `NATIVE_ARM_ID` (metadata only) + per-arm `arm_config{suffix}.json`
  resolved-manifest artifact.
- Driver matrix = repeated `run_arm` with per-arm env; retrieve
  `integrity/memory/profile/run_config/environment/cache_events` per arm.
- Mind: runner always exits 0; classification/verdict fields are the truth.
- Mind: `run_config.json` committed values silently override env defaults.
- Per-arm wall estimate for the 8-prompt x 128-token workload: ~75-85 min
  cold -> ~4-5 h per 3-4-arm session including build+seal; split sessions if
  needed.

## 9. Contract decisions (supervisor-pinned for Wave B)

1. LLP64 keys -> `int64_t`/`size_t` unconditionally. YES.
2. Oversized `ensure` -> early-reject before evicting (preserve residents).
3. Same-key different-size on any hit path -> fail closed.
4. Zero budgets: keep "0 = default" semantics BUT add explicit
   `host_cache_mode`/`gpu` knobs so "off" is a named, honest arm; add
   init-time validation `pack_budget >= record_bytes` when lanes>1
   (or document the lanes=1 fail-soft contract).
5. Eviction flag `gpu_evict=lru|rank_priority`, default `lru` after the
   local gate; both arms ship in one build.
6. Store lifecycle: forbid `close()/open()` while views/fills outstanding —
   document + assert, no locking machinery (A4 D3).
7. Read-time integrity: add optional per-record sha256 verify on cold fill
   behind `DEE4_VERIFY_READS=1` (off-lane, ~10-40 ms/record); default off —
   seal-at-open + trusted transport remains the shipped contract, the flag
   is a debug net.

## 10. Wave-B workstream decomposition (write-capable, file-disjoint)

- **B0a host tier** (`host_pack_cache.{h,cpp}`, `test_host_pack_cache.cpp`):
  bool fill channel + rollback; size guard; get() instrumentation parity;
  evicted-identity surface; `budget_rejections`; stats reconciliation.
- **B0b VRAM tier** (`vram_cache.{h,cpp}`, `test_vram_cache.cpp`,
  `scripts/deepseek_v4_cache.py`, `tests/test_deepseek_v4_cache.py`):
  victim list in `EnsureInfo`; early-reject; nbytes check; `gpu_evict` flag;
  mirror lockstep; pinned-skip counting fix.
- **B0c transfer tier** (`async_prefetcher.{h,cpp}`,
  `test_async_prefetcher.cpp`): staging-slot lifetime fix (R1); LLP64 keys;
  erase all victims; submit-failure block invalidation; event/slot retire
  pass; `wait_on_stream` counter parity.
- **B0d engine** (`engine.cpp`, `engine.h`): deferred unpin in
  `moe_forward_batch_device_impl` (R3); `avail_layer` fail-closed/gated;
  host-resolution plumbing into `prefetch_*` (RequestKind::HostHit);
  `set_external_token` consumed correctly; pack/store stats snapshot API;
  `clear_host_pack` + reset coverage; fp4 scratch bounds check + ledger.
- **B1 sim** (`tools/phase4/`): two-level journal-replay simulator promoted
  from `tmp/reuse_analysis.py` — validated vs GPU-2 counters, MIN column +
  dominance tripwire, per-policy arms, regime columns.
- **B2 workload+driver** (`tools/phase4/`, kaggle driver): the 8-prompt x
  128-token workload spec (drafted), `NATIVE_TRACE_REQUESTS`,
  `NATIVE_ARM_ID`, `NATIVE_CACHE_RESET`, per-prompt stats snapshot fix,
  cache_events retrieval, campaign driver + watcher protocol.
- **B3 budgets** (runner + engine config): admission check, bypass arm,
  release_source_pages, profile presets.

Sequencing: B0* first (they change semantics that tests pin); B1-B3 in
parallel with B0; Wave-C gate = 19/19 ctests + new phase-4 tests + sim
validation vs sealed counters; only then Wave-D GPU.

## 11. Risks / open items

- The R1/R3 races mean Phase-3's exact PASS plausibly benefited from
  storage-latency masking (a fill is slower than an H2D, so queues drained
  before reuse). Phase-4 raises hit rates -> the masking shrinks -> the
  repairs are prerequisites, not hardening.
- `prepare_fp4_experts` consults the pack for VRAM-resident demands too —
  pack "misses" overstate compute-needed bytes by <=~3% (A1 A3). The event
  stream must record both resolutions to keep attribution honest.
- fp4 decode-per-use adds ~65 ms/token critical path; if the GPU campaign
  shows fp4 winning big on hits but losing on decode overhead, the
  prewarm/batched-decode variant is a Phase-5-shaped lever — record, don't
  implement.
- Two audit streams (repo map, roadmap/docs) still outstanding; their
  findings fold into this doc when they land — nothing above is expected to
  change.
