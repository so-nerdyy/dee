# research/phase2-t4 — evidence sources consulted for PHASE2_T4_CAMPAIGN.md

Track: `research/phase2-t4-campaign`, worktree `.freebuff/wt/p2r`, base
`56dad3c1dccb0d7cee774df20d8f0047059d0bd7` (Luna's
`research/phase2-integration-lru-fix` head). Every number cited in the
campaign doc traces to one of these sources.

## Budget + pre-flight rules

- `AGENTS.md` (main checkout root, canonical): CPU 10 / GPU 2 hard caps,
  ledger 0/10 + 0/2 at write time; GPU-run exclusions list; 10-item
  pre-flight gate; per-run header fields (RUN TYPE / RUN NUMBER / QUESTION
  ANSWERED / WHY LOCAL INSUFFICIENT / ARMS INCLUDED / SUCCESS CRITERIA /
  FAILURE CRITERIA / ARTIFACTS EXPECTED); "Kaggle weekly GPU quota 30h"
  precedent noted in RUN_REGISTRY `cache1-pending` entry.

## Arm definitions + matched-run rules

- `PHASE2_AB_RUNBOOK.md` (this worktree, 34 lines): arm switch table
  (`phase2.enabled` / `host_enabled` / `vram_priority_fix_enabled`),
  matched-pair rules, no-retry-into-score, "one favorable run" ban.
- `.freebuff/wt/v2a/PHASE2_AB_RUNBOOK.md` (71 lines, audit-track version):
  the "cron guard" sentence this task references lives at its tail:
  *"Before any future remote launch, pause the active goal and create the
  requested cron guard before starting the next job."*

## Phase-2 switch surface (what the arms actually flip)

- `dee.cpp/include/dee/expert_tiers.h:75-84` — `Phase2TierConfig`:
  `enabled`, `host_enabled`, `vram_priority_fix_enabled`,
  `model_identity`, `host` (HostTierConfig), `host_policy` (null => plain
  LRU), `device_policy`.
- `dee.cpp/include/dee/host_expert_tier.h:98-105` — `HostTierConfig`:
  `slot_bytes` (0 => Engine derives packed record size), `alignment=4096`,
  `policy_slots`, `dynamic_slots`, `budget_bytes` (no implicit budget),
  `try_pin` (the pageable-fallback knob).
- `dee.cpp/src/engine.cpp:3424-3435` — `phase2_host = enabled &&
  host_enabled`, `phase2_vram = enabled && vram_priority_fix_enabled`;
  `enabled` with neither subfeature fails closed; `phase2_host` requires
  packed-FP4 cache + CUDA.
- `dee.cpp/src/engine.cpp:3154-3156` — experimental stage path is invoked
  from `stage_expert` only when `phase2.enabled && phase2.host_enabled`;
  sequential per-expert call (the T9 concurrency gap).
- `dee.cpp/src/engine.cpp:3679` — `cache_.set_experimental_plain_lru(
  phase2_vram)`; `dee.cpp/include/dee/vram_cache.h:210-214` —
  `PRIORITY_WEIGHT = 1<<20`, plain-LRU branch returns `b.last_used` only.
- `dee.cpp/src/engine.cpp:3696-3714` — init wiring: ExpertStoreColdAdapter
  (appends store integrity identity to `model_identity`), HostExpertTier,
  DeviceExpertTier; failure unwinds all three and fails init.

## Sealed baseline the arms are matched against (v60)

`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/v60-seal-20260901T041158Z.json`
+ `v60-evidence-20260901T040935Z/native-generate-result.json`:

- pinned commit `011a30341a397ea4af6939b519acc4d05e36c754`, 2xT4 SM75,
  `budget_bytes=3,758,096,384` (3.5 GiB = **281 packed-FP4 slots**),
  `host_pack_cache_bytes=9,126,805,504` per engine (8.5 GiB x2 = ~17 GiB
  pooled), `source_read_lanes=4`, `source_read_queue_depth=8`,
  `cache_dtype=fp4`, `expert_store=dee4_trace`, n_tokens=16.
- engine_stats cuda0: `resident_hits=328, cold_loads=2285, evictions=2004,
  h2d_bytes=30,548,951,040, h2d_copies=2285, resident_experts=281`.
- engine_stats cuda1: `resident_hits=327, cold_loads=2159, evictions=1878,
  h2d_bytes=28,864,413,696, h2d_copies=2159`.
- host_pack: cuda0 hits=1223/misses=1390; cuda1 hits=1395/misses=1091
  (pooled 2,618/2,481 ~= sim 2,621/2,478, 51.40% hits).
- expert_store: pread materialization, ~96-98 MiB/s per GPU, ~70%
  source-read overlap at q8/l4.
- metrics: build 104.71 s, prefill 94.76 s, decode_wall 72.607 s
  (15 decode tokens, 0.207 tok/s, ITL p50 4,603 ms, p95 6,844 ms),
  total generation wall 168.53 s. Whole kernel session ~20 min
  (03:48:50 build start -> 04:08:51 verdict), incl. 250 s dee4_trace
  repack of 29.4 GiB into /tmp.
- Sealed token IDs + decoded text in `deepseek_v4_native_generate.py`
  (`SEALED_TOKEN_IDS`, `SEALED_DECODED_TEXT`); route journal final chain
  `d8539b6e7c61d18820ccdcc17e492168a4132186f021ba30ca182581c5fcc75e`
  (v50 journal sha `665aac3e...`).

## VRAM-fix validation contract (predeclared numbers)

`C:/Users/carth/Downloads/dee-phase2-ws-policy/VRAM_PRIORITY_AUDIT.md`
(track `research/phase2-ws-policy` @ `95dfe0d032c2b035cbb0b8b31fa589a68a329bd9`):

- §7: at 281 slots, repaired engine should report `resident_hits`
  ~= 680 (cuda0) / 1,057 (cuda1) and `cold_loads` ~= 1,933 / 1,429;
  counters outside those bands mean the semantics were not simplified.
- §3-4: sim exact vs sealed counters (+/-1..2); LRU captures 2.07x/3.25x
  resident hits; saves 1,083 device loads = 14.48 GB H2D per 16-token
  response (298.4/622.2 MiB per decode token per GPU); host-fill upside
  bounded <=6.6 GB NVMe/response (~419 MiB/decode-token pooled, strict
  upper bound).
- §6: wall link is inferred, not measured — the campaign's host-fill and
  ITL evidence is what confirms/refutes it.
- `PHASE2_RECOMMENDATION.md` (same dir): HOST POLICY NO_CHANGE (plain
  LRU), RAM knee ~16 GiB pooled causal, Belady gap 2.8pp @16 GiB / 0 @
  >=32 GiB, regime-C prewarm numbers labeled NOT eligible for this A/B.

## Audit go-checklist items (Batch #1 must exercise live)

`.freebuff/wt/v2a/PHASE2_VERIFICATION_AUDIT.md` (branch
`research/phase2-host-tier-verification` @ `e63a3bc7c1001ce9343da1e41c09e09f4489d296`),
verdict GO, §5:

- (a) `cuda_submit_host` failure injection -> confirm event cleanup
  (defect: `async_prefetcher.cpp:732-738` creates event, failure path
  `749-754` drains+`dma_complete` but never `cudaEventDestroy`; events
  only destroyed in `reset()` `657-663`).
- (b) pageable-slot path with `pin_failures>0` -> per-copy
  `cudaEventSynchronize` cost must be visible in
  `experimental_pageable_wait_ms_` (`async_prefetcher.cpp:756-766`;
  `pin_failures` counted once at init, `host_expert_tier.cpp:187`).
- (c) `synchronize_all`/`reset` under in-flight managed transfers
  (`async_prefetcher.cpp:618-646`; armed-flag `std::terminate` hazard at
  :624 keyed on flag state, not in-flight work — LOW finding).
- Also the two Mediums transfer to the rewrite: LLP64 `map_key`
  truncation (deliberately retained for legacy equivalence), tier scope
  exclusivity ctor-only.
- Audit CPU suite `test_phase2_tier_audit.cpp` (130 checks) lives on that
  branch, NOT in this worktree — porting it into the campaign commit is a
  listed dependency.

## CUDA-gated ctests (enumerated from `dee.cpp/CMakeLists.txt:122-131`)

`if(DEE_CUDA)` appends: `test_swiglu_cuda`, `test_rmsnorm_cuda`,
`test_router_cuda`, `test_expert_batch_cuda`, `test_profiling_cuda`,
`test_deepseek_v4_fp4_cuda`, `test_deepseek_v4_fp4_expert`,
`test_phase2_cuda` (the last additionally gets `SKIP_RETURN_CODE 77`,
`TIMEOUT 60` at line 142). Eight CUDA-gated ctest targets total.
`test_phase2_cuda` (`dee.cpp/tests/test_phase2_cuda.cpp`): stream-callback
gate keeps DMA pending; verifies compute-stream wait cannot release the
host source early, actual copied bytes, slot reuse, forced pageable
fallback, `pinned_staging_bytes()==0` (no gather ring on direct path).

Non-CUDA but Phase-2 relevant (run locally AND in the CPU pre-flight):
`test_phase2_host_tier` (TIMEOUT 30), `test_expert_store`,
`test_vram_cache` (covers `set_experimental_plain_lru` recency order,
lines 88-99), `test_engine`, `test_real_router`, `test_host_pack_cache`.

## Fixed-slot staging HOLD + the microbench gate

`research/phase2-fixedslot-proto` @ `fbc8b205d4ab6a6bbba56c7660b804332277b8f5`,
`research/phase2-fixedslot-proto/FIXEDSLOT_VERDICT.md`:

- FIXED_SLOT_STAGING_HOLD. Reservation prize retracted (us not ms:
  ~11 us baseline vs ~12 us proto at 12.75 MiB records).
- Measured real win: HIT-behind-MISS H2D submission delay 10-21 ms ->
  ~0.02 ms (500-800x), but hidden by storage fills today.
- Revised prize ~1 s firm / ~2-5 s plausible; biggest unmeasured piece is
  gather-memcpy elimination (~2-4 s derived) needing real CUDA.
- Narrow live gate: pinned-slot-direct A/B on dual-T4 decode measuring
  the 9.4 s enqueue split (gather-memcpy vs submit vs cache-ops) +
  pinned-alloc cost + ACCEPT_CORRECTNESS. PROMOTE iff enqueue drops >=6 s
  with no pinned-alloc regression; REJECT iff <3 s or any
  correctness/lifetime defect.

## Incremental-evidence lesson (cache1b/cache1c/v9-v11)

`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/RUN_REGISTRY.json`
entries `cache1b-modal-20260806` (8h timeout, harness wrote evidence only
in `finally` -> EMPTY evidence dir) and `cache1c-modal-20260807` (heartbeat
thread died on unbounded `evidence_volume.commit()`; silent run to 16h
SIGTERM -> INVALID_EXPERIMENT stub, zero gates). v12 harness comment
(`deepseek_v4_native_generate.py:~1453`): "checkpoint every generated
token to /kaggle/working so an OOM kill (v9/v11 lost ALL tokens) still
leaves the exact token stream + timing" — `generated_checkpoint.jsonl`,
one JSONL record per token carrying route-journal link, proc/sys memory,
host_pack + engine_stats snapshots. CACHE1c lesson additionally: heartbeat
must never do a blocking upload/commit.

## Harness mechanics (what T3 must extend)

`dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_native_generate.py`:

- Clones `REPO` @ `BRANCH`/`NATIVE_COMMIT`; builds dee_core +
  `test_deepseek_v4_fp4_cuda` + `test_deepseek_v4_fp4_expert`
  (sm_60;sm_75, -j2) as admission gate; builds pydee via
  `pydee/setup.py build_ext --inplace`; repacks dee4_trace bank into /tmp
  if absent (>=45 GiB free check); builds eng0/eng1 + model; greedy 16
  tokens; `generated_checkpoint.jsonl` + `routed_experts.jsonl` per step.
- Env vars unreliable on Kaggle -> committed `run_config.json` next to the
  script is the config surface (v47 mojibake lesson: ASCII-only source).
- `dee.cpp/pydee/pydee.cpp:73-102` — EngineConfig bindings exist for
  everything EXCEPT `phase2`; no `phase2_metrics` binding. That gap is
  the T3 blocker.
- `dee.cpp/scripts/deepseek_v4_model.py:1257` `build_native_engine()` —
  the pydee.Engine factory the harness uses; phase2 kwargs must reach
  `cfg.phase2` here.
- Seal tooling: `dee.cpp/scripts/seal_deepseek_v4_ds10.py`.
- Kernel identity: `nivind/dee-cpp-dsv4-native-generate`,
  `machine_spec` dual T4; dataset `nivind/deepseek-v4-flash-0731-shards`
  mounted read-only at `/kaggle/input` (avoids 167 GB download).
- Hardware lottery precedent: P2.4 note in harness — dual-T4 pool
  exhausted ~12 consecutive launches -> 1xP100; `SINGLE_GPU` fallback
  exists; P100 correctness valid, perf not comparable.

## Sibling tracks (dependency section)

- T3 `feat/phase2-pydee-arming` — worktree `.freebuff/wt/p2b`, still at
  base `56dad3c`. Exposes `EngineConfig::phase2` + `phase2_metrics` to
  pydee and the run_config surface = the arm-switch env surface.
- T1 `audit/phase2-tier-hardening` — worktree `.freebuff/wt/p2h`, at base.
  Preferred landing of audit Medium/Low fixes before Batch #1.
- T9 `research/phase2-concurrent-fill` — worktree `.freebuff/wt/p2c`, at
  base. Fixes the sequential per-expert `stage_expert` loop that
  serializes cold physical reads (the "Open Phase-2 perf issue" in
  AGENTS.md) — required for host arms to count as representative perf
  evidence.
- Phase-3: `research/phase3-full-expert-store` @
  `8a3a2cda7e265ad57d130229ffecc7d1920bb0c9` (`.freebuff/wt/p3`),
  `PHASE3_FULL_EXPERT_STORE.md`: 46x256=11,776 records ~146.6 GiB,
  dee4-v4 segmented (3.1875 GiB/bucket, resumable journal), dee4-v5 lazy;
  cloud build ~35 min at prior repack rate; Kaggle dataset cap 20 GB /
  account ~100 GB (CACHE1 §10) — full store does NOT fit datasets, must
  build in-session or on volume.
- `research/phase2-regime-c` (rgc) and `research/phase2-v41-check` (v41)
  also in flight — neither is a Batch #1 blocker.

## Other facts used

- `dee.cpp/tools/fill_replay.cpp` — host-only replay/rider tool
  (`-DDEE_BUILD_FILL_REPLAY=ON`), replays production access through real
  `Dee4ExpertStore::materialize` + `HostPackCache::get_batch`; emits
  `fill_timeline.json`. CPU-run candidate.
- Kaggle container facts: ~1 TiB free on root overlay (no `/kaggle/temp`
  on medium dual-T4); host RAM 31.35 GiB total / ~29.6 GiB available at
  engine time; 29.4 GiB bank repack 250 s measured; dataset cap ~20 GB,
  account cap ~100 GB (CACHE1 §10), weekly GPU quota ~30 h (registry
  `cache1-pending` blob).
- `PHASE2_METRICS.md` — TierMetrics field semantics; nested/service-time
  waits must not be summed into critical path.
- `PHASE2_INTEGRATION_MAP.md` — integration boundaries; experimental
  Engine path fills at per-expert stage boundary, deliberately no lane
  scheduler in this commit.
