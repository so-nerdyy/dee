# PHASE2_T4_CAMPAIGN.md — mechanical Kaggle campaign plan (W1-T5)

Status: **plan only**. Nothing here launches anything. Every remote run
requires the AGENTS.md pre-flight gate, the cron-guard step, and explicit
user authorization.

Track: `research/phase2-t4-campaign`, worktree `.freebuff/wt/p2r`.
Evidence base: `research/phase2-t4/EVIDENCE_SOURCES.md` (every number in
this doc is anchored there).

Hard ledger (user-set caps, AGENTS.md 2026-09-10): **CPU 0/10, GPU 0/2**.
This plan spends at most **GPU 2/2** and **CPU 4/10** in the primary path,
holding CPU 5–10 in reserve for the Phase-3 branch of the decision tree.

---

## 0. Design rules inherited (non-negotiable)

From `AGENTS.md` and `PHASE2_AB_RUNBOOK.md` (both versions):

1. GPU runs are never spent on syntax debugging, ordinary unit-test
   failures, dependency debugging, speculation without a decision gate,
   anything reproducible locally, or questions simulation can answer.
2. All experiment arms packed into the SAME batch when practical.
3. Matched arms share: immutable commit, checkpoint/store identity,
   container/driver, GPU placement, route inputs, seed, prompt/token
   workload, budgets, prefetch depth. Only the Phase-2 switches differ.
4. Predeclared: hypothesis, baseline, candidates, metrics, accept/reject
   criteria, warmups, reps, timeout, failure handling, artifact schema.
   A failed rep is *unresolved* — never retried into a score.
5. No claim from one favorable rep, a mock, or a changed workload.
   Seal only after every measured rep passes exactness.
6. Exactness is the routing/output contract: 16/16 sealed token IDs, the
   sealed route-journal chain, packed-byte identity. Tier choices may
   never change which expert executes.
7. Host policy is plain LRU only. No frequency pinning, compression,
   regime-C prewarming, codec, or unvalidated device score is eligible.
8. Before any remote launch: pause the active goal and create the
   requested cron guard (v2a runbook tail). See §8.

---

## 1. What the campaign must decide

Phase-2 closure question (AGENTS.md roadmap): does the integrated
SSD->RAM->VRAM hierarchy preserve exactness and reduce exposed
slow-tier traffic on live 2xT4 hardware, and what is the next binding
bottleneck? GPU Batch #1 answers five sub-questions in one job:

- **Q1 mechanism:** do all CUDA-gated mechanisms pass on real CUDA
  (8 ctests + audit §5 probes)?
- **Q2 exactness:** does `phase2.enabled` keep 16/16 token IDs and the
  sealed route chain under every arm and every rep?
- **Q3 VRAM causal:** does `vram_priority_fix_enabled` alone reproduce the
  predeclared counter contract (§4.6) on the sealed trace?
- **Q4 host causal:** does the host tier cut logical SSD bytes/token at a
  matched budget, and is the current sequential-fill shape representative
  (§4.7 honesty rule)?
- **Q5 next bottleneck:** post-fix, where does the wall go — cold bytes,
  H2D/staging, expert compute, or orchestration? (feeds the reproup
  decision and Batch #2's branch)

---

## 2. GPU BATCH #1 — Phase-2 mechanism/causal campaign (GPU 1/2)

| Field | Value |
|---|---|
| RUN TYPE | GPU batch (Kaggle `nivind/dee-cpp-dsv4-native-generate` class kernel, `machine_spec` 2xT4 SM75) |
| RUN NUMBER | **GPU 1/2** |
| QUESTION ANSWERED | Q1–Q5 above: Phase-2 mechanism sound on live CUDA? VRAM repair reproduces its contract? Host hierarchy reduces slow-tier traffic at matched budgets? What remains binding? |
| WHY LOCAL INSUFFICIENT | No GPU, no nvcc, no CUDA driver on this machine; every CUDA path (event lifetime, direct H2D, pageable fallback, `synchronize_all` behavior) has only been reviewed statically. The wall-clock causal questions exist only on 2xT4 + Kaggle /tmp storage regime. |
| ARMS INCLUDED | 4 scored arms — Baseline / VRAM-only / Host-only / Combined — plus an unscored mechanism segment, audit-probe segment, microbench segment, and a bracket drift-check rep. |
| SUCCESS CRITERIA | §4.8 accept table: all mechanism gates pass; every measured rep exact; VRAM arms inside the predeclared counter bands; baseline reproduces sealed v60 counters; artifacts complete and sealable. |
| FAILURE CRITERIA | §4.9: any mechanism ctest fail or audit-probe defect => `MECHANISM_FAIL` (hard abort, no arms); baseline counter drift => `INVALID_EXPERIMENT`; any arm exactness break => `REJECT_EXACTNESS` for that arm; hardware lottery => `INVALID_HARDWARE_LOTTERY` after mechanism-only completion. |
| ARTIFACTS EXPECTED | §4.10: per-rep `generated_checkpoint.<arm>.<rep>.jsonl`, `routed_experts.<arm>.<rep>.jsonl`, `result.<arm>.<rep>.json` with `engine_stats` + `phase2_metrics` + `host_pack` + `expert_store` + `stage_profile`, segment reports, kernel log, environment.json, manifest with sha256 of every artifact — registry-schema conforming. |

### 2.1 Hypothesis (written before launch)

- H1 (mechanism): the Phase-2 tier machinery is exact and leak-free under
  real CUDA: all 8 CUDA-gated ctests pass, and the three audit §5 probes
  produce their specified behavior (§3.2).
- H2 (VRAM causal, simulated -> live): at the sealed 281-slot budget,
  `vram_priority_fix_enabled` moves `resident_hits` from ~328/327 to
  ~680/1,057 and `cold_loads` from 2,285/2,159 to ~1,933/1,429 on
  cuda0/cuda1, removing ~1,083 H2D copies (~14.48 GB) per 16-token
  response without touching route or output bytes.
- H3 (host causal): at the matched ~8.5 GiB-per-engine host budget, the
  Phase-2 host tier reproduces the sealed LRU hit surface (within sim
  tolerance) while converting `host_pack` fill behavior into
  lease/fill-coalesced tier traffic — logical `SSD_bytes` per token near
  the sim's prediction, not above it.
- H4 (wall, directional only): the VRAM fix's *certain* component is H2D
  reduction; its wall value rides on host-fill savings bounded at
  <=6.6 GB NVMe/response — expected decode-wall effect 0.3–1.4 s/token at
  the 0.29–0.37 GiB/s bank, partially overlapped. Wall is reported, never
  gated on, per the audit's honest-inference rule.

### 2.2 Arm definitions

Every arm: same pinned campaign commit (§7), same `dee4_trace` bank
(`data_sha256 c83462ba…`, 2,364 records x 13,369,344 B = 29.4 GiB), same
canonical prompt + 16 tokens, same `source_read_lanes=4`,
`source_read_queue_depth=8`, same VRAM `budget_bytes=3,758,096,384`
(281 slots/GPU), same host envelope (~8.5 GiB per engine), same seed-free
greedy decode, `cache_dtype=fp4`, `transfer_dtype=fp4`.

| Arm | `phase2.enabled` | `host_enabled` | `vram_priority_fix_enabled` | Data path actually exercised |
|---|---:|---:|---:|---|
| **A0 baseline** | false | false | false | legacy `host_pack` + `prefetch` + priority-LRU VRAM (sealed v60 semantics) |
| **A1 VRAM-only** | true | false | true | identical data path to A0; ONLY the eviction score changes (`experimental_plain_lru`) |
| **A2 host-only** | true | true | false | tier path: `stage_expert -> DeviceExpertTier::stage -> HostExpertTier acquire -> prefetch_host_lease -> direct H2D` |
| **A3 combined** | true | true | true | tier path + plain-LRU VRAM |

Engine-config deltas for A2/A3 (matched envelope, derived from v60):

```text
phase2.model_identity = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
                        + dee4_trace integrity identity (appended by adapter)
phase2.host.slot_bytes      = 0            # engine derives 13,369,344
phase2.host.alignment       = 4096
phase2.host.policy_slots    = 0            # plain LRU: no policy residents
phase2.host.dynamic_slots   = 682          # floor(9,126,805,504 / 13,369,344)
phase2.host.budget_bytes    = 9,126,805,504  # = v60 host_pack_cache_bytes
phase2.host.try_pin         = true         # pageable probe is a separate segment
phase2.host_policy          = null         # -> PlainLruHostPlacementPolicy
phase2.device_policy        = null
```

(`budget_bytes` includes slot padding per `HostTierConfig`; the exact
dynamic_slots is recomputed by the config loader and logged — the
*predeclared* value is the 8.5 GiB byte envelope, not the slot count.)

Fail-closed config rules already in `engine.cpp:3424-3445` (verified):
subfeatures without `phase2.enabled` → init fails; `enabled` with
neither subfeature → init fails; `host_enabled` without packed-FP4 +
CUDA + `Fp4E2m1` transfer + non-empty `model_identity` → init fails.
An arm that misconfigures dies at engine init, in seconds, never in a
scored rep.

### 2.3 Reps, warmups, ordering

Per AGENTS.md + runbook: predeclared, never retried into a score.

- **Smoke rep (unscored):** each arm runs ONE `n_tokens=2` generation
  first. Purpose: prove the arm boots, tokens 1–2 match the sealed
  prefix, metrics snapshot writes. Its numbers are *never* scored.
- **Measured reps:** 3 per arm, `n_tokens=16`, each preceded by a full
  engine teardown + rebuild + cache reset so every rep is a **regime-A
  cold start** (contract A from the ws-policy work: no cross-request
  warmth, no prewarm). Regime-B/C contamination (stale host/VRAM content
  from a prior rep) is checked per rep: `phase2_metrics`/engine counters
  must start at zero for the rep's lifetime window.
- **Order:** A0 -> A1 -> A2 -> A3, then **A0-rep4 drift bracket** at the
  end (a 4th baseline rep) so a slow environmental drift inside the
  session is detectable instead of silently attributing it to an arm.
- Total generations: 4 arms x (1 smoke + 3 reps) + 1 bracket = **17**.

### 2.4 Job internal ordering — hard-gated phases

The job is a single Kaggle kernel run whose harness executes these phases
in order. **Mechanism gates come first; the A/B arms are unreachable if
any mechanism gate fails.** The 29.4 GiB bank repack (~250 s) and the
167 GB shard availability are needed only for phases P6+; a mechanism
abort never pays for them.

```text
P0  env + hardware probe
      - record GPU UUIDs/CCs, driver, CUDA toolkit, CPU count,
        host RAM, /tmp free (>=45 GiB), dataset mount presence
      - HARDWARE GATE: if accelerator != 2xT4 SM75 -> run ONLY
        mechanism+bench segments (they are hardware-tolerant), then
        exit INVALID_HARDWARE_LOTTERY. Never run arms on P100.
P1  clone pinned commit + build
      - dee_core + ALL 8 CUDA ctest targets + pydee + bench tools,
        sm_75 (+sm_60 fallback cubins as in current harness), -j2
P2  MECHANISM SEGMENT (hard abort on any failure)         [~10-20 min]
      - run the 8 CUDA-gated ctests, section 3.1
      - run the audit-probe binary/tests, section 3.2
      - on any FAIL: write MECHANISM_FAIL verdict + evidence, STOP
P3  MICROBENCH SEGMENT (informational, never aborts)      [~5-10 min]
      - pinned_direct_bench, section 3.3; records enqueue split
P4  bank prepare                                          [~4-6 min]
      - mount segment dataset (CPU-2 product) OR repack_trace in /tmp;
        validate data_sha256 + metadata integrity before proceeding
P5  ARM LOOP                                              [~90-110 min]
      - per arm: smoke(2 tok) -> 3x measured(16 tok), cold reset each rep
      - per rep: checkpoint every token; on rep end write result JSON
        + engine_stats + phase2_metrics immediately (cache1c lesson)
      - A0 drift bracket rep last
P6  evidence pack + seal
      - manifest.sha256 over every artifact; progress.log closed;
        RUN_REGISTRY-shaped result record emitted
```

### 2.5 Cost math (Batch #1)

| Component | Wall est. |
|---|---|
| P0 probe | ~1 min |
| P1 clone+build (dee_core + 8 tests + pydee + bench) | ~8–15 min (-j2, Kaggle CPU side of a GPU session) |
| P2 mechanism ctests + audit probes | ~10–20 min |
| P3 microbench | ~5–10 min |
| P4 bank repack/validate (once, shared by all arms) | ~5 min |
| P5 arms: 17 generations x ~3.5–5.5 min (2-tok smoke ~1.5 min; 16-tok rep ~4.5–5.5 min at v60's 168.5 s gen wall + ~105 s build/reset) | ~80–110 min |
| P6 seal/pack | ~3 min |
| **Total** | **~2.0–2.8 h wall** |

Quota: dual T4 bills **2 GPU-h per wall-h** against the ~30 h/week quota
→ worst case ~5.6 GPU-h, ~19% of a week. Even a hardware-lottery abort
after P3 costs <=1 GPU-h. This is comfortably inside budget; the binding
scarcity is *run count* (2), not hours.

---

## 3. Batch #1 non-arm segments

### 3.1 Mechanism-validation segment (P2, hard-abort)

All 8 CUDA-gated ctest targets (enumerated from
`dee.cpp/CMakeLists.txt:122-131`), each run under `ctest` with output
captured:

| Test | What it proves on live CUDA |
|---|---|
| `test_swiglu_cuda` | SwiGLU kernel numerics on SM75 |
| `test_rmsnorm_cuda` | RMSNorm kernel numerics |
| `test_router_cuda` | router/top-k CUDA path |
| `test_expert_batch_cuda` | batched expert execution path |
| `test_profiling_cuda` | CUDA profiling/timing plumbing |
| `test_deepseek_v4_fp4_cuda` | packed-FP4 device decode (numerical admission gate; already in harness) |
| `test_deepseek_v4_fp4_expert` | exact packed-cache device API used by the full model (admission gate; already in harness) |
| `test_phase2_cuda` | the Phase-2 lifecycle itself: gated-DMA ordering, `wait_on_stream` cannot release host source early, copied-byte identity, pinned + pageable (`run(false)`) fallback, slot reuse, zero gather-ring (`pinned_staging_bytes()==0`). Exits 77 (SKIP) without a GPU — on 2xT4 it must PASS, and it is the single most campaign-relevant test. |

CPU-side (non-CUDA) tests run *locally* pre-flight (they are part of the
gate, not of the paid run): `test_phase2_host_tier`, `test_vram_cache`
(incl. the `experimental_plain_lru` recency checks), `test_engine`,
`test_expert_store`, `test_real_router`, `test_host_pack_cache`, plus the
audit suite `test_phase2_tier_audit` **if T1 ports it into the campaign
commit** (§6).

Hard abort rule: any non-PASS (FAIL/TIMEOUT; SKIP=77 for
`test_phase2_cuda` is a *fail* here because a GPU is present) →
`MECHANISM_FAIL`, write evidence, STOP. No arm may run on a broken
mechanism; the whole point of ordering is that a broken tier never gets
scored against the baseline.

### 3.2 Audit §5 checklist segment (P2, hard-abort)

The verification audit (`.freebuff/wt/v2a/PHASE2_VERIFICATION_AUDIT.md`,
branch `research/phase2-host-tier-verification` @ `e63a3bc7`) returned
**GO** conditioned on three live exercises that static review could not
settle. Each is spec'd as a small test binary (or a mode of
`test_phase2_cuda`) that runs in P2:

**(a) `cuda_submit_host` failure injection → event cleanup.**
- *What must be proven:* on injected `cudaMemcpyAsync` failure, the
  failure path (drain stream, set `dma_complete`, return false) destroys
  the just-created `cudaEvent_t` on the abandoned transfer instead of
  leaking it until `reset()`. Success = N injected failures leave
  event-handle count flat (query via `cudaEventQuery` sweep + a
  test-visible `events_live` counter) and the prefetcher still serves a
  subsequent good transfer.
- *Hook needed (spec):* a fault-injection seam in `AsyncPrefetcher`
  compiled **only** under a test flag, e.g.
  `dee/tests/fault_inject.h` + `AsyncPrefetcher::inject_submit_failure(n)`
  setting a counter that forces the *n*-th `cuda_submit_host`'s
  `cudaMemcpyAsync` to return `cudaErrorUnknown` without enqueuing. This
  is a T1-adjacent code change; if it is not landed by launch, this probe
  degrades to "observational": drive enough transfers that a natural
  failure would surface, and record that the injected variant was not
  run (verdict note, not a gate). **Do not** build an ad-hoc injector
  inside the production build.

**(b) Forced pageable path via `pin_failures>0`.**
- *What must be proven:* when `cudaHostRegister` fails at pool init
  (surfaced only as `stats().pin_failures`), every H2D falls into the
  pageable fallback and the per-copy `cudaEventSynchronize` cost is
  visible in `experimental_pageable_wait_ms_` / TierMetrics
  `pageable_fallback_wait_ms`.
- *How to force it:* two knobs, both already in-tree —
  (i) `HostTierConfig.try_pin=false` (cleanest; no registration
  attempted, slots stay usable pageable, `pin_failures` stays 0 — note
  this does NOT produce `pin_failures>0`, it produces the pageable path
  with zero failures), and
  (ii) a backend whose `pin` returns `false` (exactly what
  `test_phase2_cuda`'s `run(false)` does) — **this** is the
  `pin_failures>0` case. The segment runs BOTH: an engine-level probe
  with `try_pin=false` *and* the `run(false)` half of `test_phase2_cuda`,
  asserting `pin_failures>=1` (case ii) and `pageable_fallback_wait_ms>0`
  when traffic flows.
- *Success:* `experimental_pageable_wait_ms_`/TierMetrics field is
  nonzero and monotonically growing under load; no crash; byte-exact
  outputs on the probe.

**(c) `synchronize_all`/`reset` under in-flight managed transfers.**
- *What must be proven:* with a deliberately held DMA (the
  `cudaLaunchHostFunc` gate already used by `test_phase2_cuda`),
  `synchronize_all()` blocks until the gate releases and completes all
  managed transfers without `std::terminate`, and `reset()` while
  transfers are in flight drains + destroys events + frees leases
  cleanly. The audit's LOW finding is that the armed-flag `terminate` at
  `async_prefetcher.cpp:624` is keyed on flag state, not on actual
  in-flight work — the probe must show *no* terminate when managed
  transfers are genuinely in flight and syncing succeeds, and must log
  the path taken if a sticky CUDA error ever triggers it.
- *Success:* no terminate/abort, `in_flight()==0` after sync, host
  `leased_slots` returns to 0, subsequent stage still works.

### 3.3 Pinned-direct vs gather-copy microbench (P3, informational)

Purpose: produce the *measured* enqueue decomposition that
`research/phase2-fixedslot-proto` @ `fbc8b205` could not (its
FIXED_SLOT_STAGING_HOLD verdict: reservation prize retracted to ~µs,
HIT-behind-MISS submit delay cut 10–21 ms -> ~0.02 ms but hidden behind
storage fills; ~1 s firm / ~2–5 s plausible prize pending one real-CUDA
measurement).

**Spec — `dee.cpp/tools/pinned_direct_bench.cpp`** (new file, to be
authored under the campaign commit or T1; host-buildable CUDA tool like
`fill_replay`):

- Allocates: one `VramCacheManager` arena (281-slot, 3.5 GiB) + one
  `AsyncPrefetcher` + one `HostExpertTier` on `host_memory_backend(
  cuda=true)` with `try_pin` toggled by flag.
- Drives the *production record size* (13,369,344 B) through the tier
  staging path, timing three components per stage:
  1. `t_gather` — memcpy/assembly into staging (the legacy gather-copy
     component the tier removes),
  2. `t_submit` — `prefetch_host_lease` enqueue (direct H2D submission
     incl. event record),
  3. `t_ops` — surrounding cache bookkeeping (ensure/pin/unpin/lease
     acquire-release).
- Measures the *pinned-alloc cost* the HOLD gate requires:
  `cudaHostAlloc`/registration time for the ~8.5 GiB slot pool at init,
  and per-slot `cudaHostRegister` marginal cost.
- Sweeps: pinned vs forced-pageable source; hit-path submit latency with
  a resident source vs miss-path; record counts 64/256/512/1024.
- Emits `pinned_direct_bench.json`: per-op p50/p95/max µs, modeled
  enqueue total for the sealed 16-token decode (1,391 + 1,092 host
  requests at v60 counts), and the derived "enqueue split" comparable to
  the proto's 9.4 s claim.
- **Decision bound (predeclared from the proto's gate):** PROMOTE case =
  modeled enqueue drop **>=6 s** vs the gather-copy baseline with **no
  pinned-alloc regression** (pool registration stays a small fraction of
  its prize); REJECT case = **<3 s** or any correctness/lifetime defect.
  3–6 s = stays HOLD. This number feeds only the Phase-2D fixed-slot
  decision; it does not gate the arms.

---

## 4. Batch #1 scored content

### 4.4 Metrics captured per rep

Existing (unchanged) engine surface: `inter_token_latency_ms` (p50/p95/max
+ per-token vector), `decode_wall_s`, `prefill_s`, `total_wall_s`,
`build_s`, `decode_tok_s`; `engine_stats` per GPU (requests,
`resident_hits`, `inflight_hits`, `cold_loads`, `evictions`,
`duplicate_requests`, `h2d_bytes`, `h2d_copies`, `peak_vram`,
`current_vram`, `resident_experts`, `host_prefetch_ring_*`,
`d2d_gather/scatter_*`); `host_pack` stats (hits/misses/evictions/
fill_batches/lanes/queue depth/wall/overlap); `expert_store` stats
(lookups, `source_reads`, `bytes_requested`, read ms p50/p95,
`read_bandwidth_mib_s`, `materialization_mode`, batch overlap %);
`stage_profile` CPU/GPU ms + `host_waits` + `idle_gap_attribution`;
`bridge_counters`; `gpu_memory` per device; `host_mem_available_gib`;
`proc`/`sys` memory per checkpoint step.

Phase-2 additions (`TierMetrics` via `Engine::phase2_metrics`, same
lifetime interval as the rep): `host.host_hit`, `host_miss`, `coalesced`,
`SSD_bytes`, `fills`, `evictions`, `failures`, `budget_rejections`,
`pin_failures`, `allocated_bytes`, `pinned_bytes`, `resident_bytes`,
`peak_resident_bytes`, `leased_slots`, `host_wait_ms`,
`storage_service_ms`, `host_capacity_wait_ms`; `device_hit`,
`device_miss`, `H2D_bytes`, `device_evictions`, `device_failures`,
`device_bytes`, `device_peak_bytes`, `device_budget`,
`device_enqueue_ms`, `device_host_wait_ms`, `pageable_fallback_wait_ms`,
`device_gpu_wait_ms` (optional/unmeasured = UNKNOWN, never 0); `tokens`
denominator + `SSD_bytes_per_token`, `H2D_bytes_per_token`,
`bytes_per_token` with `bytes_per_token_valid`.

Accounting rule (from PHASE2_METRICS.md): never sum service-time,
nested-wait, or overlapping-CUDA totals into a critical-path speedup.

### 4.5 Exactness gates per rep (all must hold, every arm, every rep)

1. `generated_token_ids` == `SEALED_TOKEN_IDS` (16/16).
2. `decoded_text` == `SEALED_DECODED_TEXT`.
3. `routed_experts.jsonl` final `chain_sha256` ==
   `d8539b6e7c61d18820ccdcc17e492168a4132186f021ba30ca182581c5fcc75e`
   (the sealed v50/v60 chain); record indices contiguous 0..687,
   steps 0..15, layers 0..42 exactly once per step.
4. `finite_outputs`, `official_router_authoritative`,
   `cpu_expert_fallback=false`, all `bridge_zeros` counters 0.
5. `dee4_trace_validation.success`, `data_sha256` match, lookup
   failures 0 both engines.
6. Cold-start check: rep-lifetime counters begin at zero (regime A).

### 4.6 Predeclared VRAM-fix validation contract (the causal gate)

From `VRAM_PRIORITY_AUDIT.md` §7 (sealed-counter-validated sim, ±1–2
counts), at 281 slots on this exact trace:

| Counter | cuda0 expected | cuda1 expected | Reject band |
|---|---|---|---|
| `resident_hits` | **~680** | **~1,057** | outside ±12 of expected, or ~= sealed 328/327 => repair not active |
| `cold_loads` | **~1,933** | **~1,429** | same band logic; ~= sealed 2,285/2,159 => repair not active |
| `h2d_copies` | ~1,933 | ~1,429 | should track cold_loads exactly (1 copy/load) |
| `h2d_bytes` | ~25.9 GB | ~19.1 GB | derived: sealed minus ~4.69/9.79 GB |
| `evictions` | ~1,652 | ~1,148 | informational, not gated |

A VRAM arm whose counters sit at the *sealed* values means the switch did
not take (config bug) — that is `REJECT_MECHANISM`, not "no effect".
A VRAM arm landing *between* sealed and predicted = the live stream
diverges from the sealed journal — first check the route-journal chain;
if routes match and counters still drift, mark `PROVISIONAL` and do not
score the host arms' causal claim either.

### 4.7 Host-arm expectations (sim-validated, honesty-labeled)

At the matched ~8.5 GiB/engine envelope the sealed-validated sim
predicts the pooled LRU hit surface ~51.4% (2,621 hits / 2,478 misses on
the sealed journal). Live `host_pack` at v60 measured 2,618/2,481 pooled
— the host tier should land within a few counts of that on A2/A3 if the
LRU is equivalent. **The decisive host-arm read is the byte/traffic
side**: `SSD_bytes` + `H2D_bytes` per token vs A0, and whether
`host_miss`/`fills` drop relative to the legacy path's cold load count.

Honesty rule (AGENTS.md open issue): the integrated Engine path fills at
the *sequential* per-expert `stage_expert` boundary and does **not**
reuse the legacy parallel batch-prep queue. Until T9's bounded-concurrent
acquisition lands, host-arm *wall* numbers are labeled
**mechanism-representative, perf-lower-bound** — they prove the tier
works exactly and measure per-request costs, but do not yet claim the
concurrent-fill wall improvement. If T9 lands in the campaign commit,
this label lifts (§6).

### 4.8 Accept / reject

- **ACCEPT_MECHANISM:** P2 all-PASS (8/8 ctests + 3 audit probes).
- **ACCEPT_EXACTNESS:** every measured rep passes all §4.5 gates.
- **ACCEPT_VRAM_REPAIR:** A1 and A3 counters inside §4.6 bands on every
  rep, with 16/16 exact.
- **ACCEPT_HOST_TIER (mechanism):** A2/A3 exact, host counters sane
  (`pin_failures=0` at `try_pin=true`, `failures=0`, `leased_slots`
  returns to 0 between layer batches, `resident_bytes` <= budget).
- **ACCEPT_HOST_TIER (perf, provisional):** SSD+H2D bytes/token reduced
  vs A0 **and** wall not regressed beyond noise — only claimed if T9
  landed; otherwise reported as lower-bound.
- **REJECT_EXACTNESS:** any token/route/byte divergence -> that arm is
  rejected; if the *baseline* arm diverges, the whole run is
  `INVALID_EXPERIMENT`.
- **REJECT_VRAM_REPAIR:** counters at sealed values (switch inert) or
  wildly outside bands.
- **MECHANISM_FAIL / INVALID_HARDWARE_LOTTERY:** §2.2 gates.

### 4.9 Abort behavior

- Any P2 failure → immediate `MECHANISM_FAIL` verdict + STOP (no bank
  work, no arms).
- Hardware != 2xT4 at P0 → mechanism+bench segments still run (they are
  hardware-tolerant and yield evidence), then
  `INVALID_HARDWARE_LOTTERY` exit before P4. Arms never run on P100.
- Exactness break mid-arm → finish the current rep's checkpoint record
  marked `terminal=exactness_break`, skip remaining reps of that arm,
  continue to next arm (a broken arm must not starve the others).
- Baseline (A0) exactness or counter drift at rep 1 →
  `INVALID_EXPERIMENT` — the trace/environment is not the sealed one;
  continuing wastes the run.
- Wall watchdog: if cumulative wall exceeds 5 h or per-token ITL
  exceeds 3x sealed p95 for 3 consecutive tokens, freeze remaining reps,
  seal what exists, exit `TIMEOUT_PARTIAL`.
- All aborts write whatever checkpoint/segment evidence exists
  *immediately* — no evidence only in a `finally` block (cache1b lesson).

### 4.10 Artifact collection spec (RUN_REGISTRY conforming)

Layout under kernel output → preserved to
`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/p2t4-b1-evidence-<UTC>/`:

```text
progress.log                            # heartbeat, append-only, never blocks
environment.json                        # P0 probe
segment_mechanism.json                  # P2 per-test results + probe outputs
pinned_direct_bench.json                # P3
dee4-trace-validation.json              # P4 bank identity
generated_checkpoint.<arm>.<rep>.jsonl  # per-token, every rep incl. smoke
routed_experts.<arm>.<rep>.jsonl        # route journal per measured rep
result.<arm>.<rep>.json                 # RESULT block + engine_stats +
                                        #   phase2_metrics + host_pack +
                                        #   expert_store + stage_profile
result.<arm>.summary.json               # per-arm rollup
native-generate-result.json             # top-level verdict + registry fields
manifest.sha256                         # sha256 of every artifact
kernel.log                              # raw stdout/stderr
```

**Incremental-evidence rule (the cache1c lesson):**
`generated_checkpoint.jsonl` already exists — one JSONL record per
generated token carrying elapsed_s, host/proc/sys memory, route-journal
link, host_pack + engine_stats snapshots (harness ~line 1455). The
campaign extends it: (i) one checkpoint file *per arm+rep* (never a
single shared file that a late arm could truncate), (ii) `phase2_metrics`
snapshot appended to each per-token record when the arm has it,
(iii) `result.<arm>.<rep>.json` written *at rep end* — not deferred to
job end, so a crash on rep 3 of arm 4 still yields 11/12 scored reps,
(iv) the heartbeat thread only appends `progress.log` — it never performs
a blocking volume commit/upload (the cache1c 16h-SIGTERM failure mode).

**Registry entry:** one `RUN_REGISTRY.json`-schema record per *job* (this
is one batch run), keyed `run_id`, with `campaign:"PHASE2-T4"`,
`platform:"kaggle"`, `gpu:"T4:2"`, `kernel`,
`kernel_version`, `repository_commit` (campaign pin), `branch_head_at_
launch`, `hardware`, `launched_utc`, `policy` block listing the four
arm configs, `expected_verdicts`, and post-run `verdict`,
`gates_summary` (per-arm exactness + counter bands), `metrics` (wall,
ITL, hit/byte counters per arm), `evidence_dir`, `local_archive`,
`performance_comparable`, `notes`. Mirrors the `cache1*`/`v60` entries.
A nested `arms[]` sub-record per arm holds `arm_id`, `phase2.*` flags,
`reps[]` (each rep's verdict + counters + artifact hashes).

---

## 5. GPU BATCH #2 — reserved decision tree (GPU 2/2)

**Not pre-consumed.** Exactly one of two shapes, selected by Batch #1's
evidence:

| Batch #1 outcome | Batch #2 becomes |
|---|---|
| **B1a.** Mechanism FAIL or audit-probe defect that is *fixable* (T1-class patch) | **Corrected Phase-2 rerun** — same 4-arm design, new pinned commit with the fix. The job spec is literally §2 rerun on the fixed tree. |
| **B1b.** VRAM contract missed by a *diagnosable* drift (route journal matches but counters land between sealed and predicted) | **Corrected Phase-2 rerun** with instrumented counters, OR folded diagnosis into the fix first — decided by the B1 postmortem. |
| **B1c.** All segments pass, arms accepted | **Phase-3 arbitrary-prompt/full-store real inference** (next section). |
| **B1d.** Exactness break that is architectural (not a one-line fix) | **No automatic rerun.** Evidence goes to review; Batch #2 held pending user decision. |
| **B1e.** `INVALID_HARDWARE_LOTTERY` (P100 draw, arms never ran) | **Rerun Batch #1's arm portion** — mechanism+bench evidence already banked, so the rerun skips P2/P3 and goes straight to P4 arms. |
| **B1f.** `INVALID_EXPERIMENT` (baseline drift) | **Diagnose locally first**; rerun only once the platform/trace cause is identified. |

**Batch #2 header (shared fields regardless of branch):**

| Field | Value |
|---|---|
| RUN TYPE | GPU batch |
| RUN NUMBER | **GPU 2/2** — final; there is no GPU 3 |
| QUESTION ANSWERED | branch B1a/B1b: "does the fix restore the contract?" / branch B1c: "does dee execute arbitrary prompts over the full 11,776-expert universe exactly?" |
| WHY LOCAL INSUFFICIENT | same as Batch #1 (no GPU locally); B1c additionally needs the ~147 GiB store + Kaggle /tmp regime |
| ARMS INCLUDED | B1a/B1b: the §2.2 arm set reduced to the arms the defect touched. B1c: single-arm Phase-3 run (full store + arbitrary prompts; no Phase-2 arms — that question is closed by B1c) |
| SUCCESS CRITERIA | B1a/B1b: previously-failing gates pass + §4.8. B1c: exact arbitrary-prompt generation over the full store, sealed evidence, arbitrary route resolution with 0 lookup failures |
| FAILURE CRITERIA | B1a/B1b: §4.9 unchanged. B1c: any unresolved expert, byte mismatch, or store-completeness failure |
| ARTIFACTS EXPECTED | same schema; B1c adds store manifest/completeness evidence + Phase-3 journal |

### 5.1 Batch #2 branch B1c — Phase-3 shape (if selected)

Per `PHASE3_FULL_EXPERT_STORE.md` @ `8a3a2cda`: the corrected universe is
**46x256 = 11,776 records** (43 MoE layers + mtp.{0,1,2} buckets),
~146.6 GiB packed. The GPU run mounts a `dee4-v4-segmented` store (46
bucket segments, 3.1875 GiB each) — built by CPU-side prep, NOT inside
the GPU session (35-min build + /tmp capacity + resume semantics all
argue for prep, see CPU-4 below). Arbitrary-prompt set + token count +
completeness gates are predeclared at launch time against whatever
prompt corpus the reviewers choose; this doc deliberately does not
fix the prompt set (that is a review decision, not a mechanical one).

---

## 6. Dependencies — what must land before Batch #1

| Dep | Track / worktree | Status at write time | Blocker? | If absent |
|---|---|---|---|---|
| **T3 — pydee arming** | `feat/phase2-pydee-arming` (`.freebuff/wt/p2b`) | at base `56dad3c`, nothing landed | **BLOCKER** | There is *no* way to flip `phase2.*` from the Kaggle harness: `pydee.cpp` binds every EngineConfig field except `phase2`, and `build_native_engine()` has no phase2 kwargs. Without T3, Batch #1 cannot exist. |
| Campaign harness loop | (this doc's spec; lands on campaign commit) | — | **BLOCKER** | The current `deepseek_v4_native_generate.py` runs exactly one config once. The multi-arm loop (P5), per-arm checkpoint files, `phase2_metrics` capture, mechanism segment, and abort logic all need to be *in the payload*. This is mechanical harness work — same repo surface, not a budget item. |
| Audit §5 probe binaries + `pinned_direct_bench.cpp` | spec'd §3.2/§3.3; natural owner = T1 or the campaign commit | — | **BLOCKER for the segments** (not for arms) | If unlanded at launch, P2 reduces to the 8 ctests and the fixedslot question stays HOLD — arms still run. Degrades evidence, does not invalidate it. |
| **T1 — tier hardening** | `audit/phase2-tier-hardening` (`.freebuff/wt/p2h`) | at base | nice-to-have | Audit was already GO; the two Mediums are latent/unreachable via Engine. Preferred so the probes test the *hardened* code. If T1 lands, campaign commit rebases onto it. |
| **T9 — concurrent host fill** | `research/phase2-concurrent-fill` (`.freebuff/wt/p2c`) | at base | **BLOCKER for host-arm perf claims only** | See degraded design below. |
| `test_phase2_tier_audit` port | `research/phase2-host-tier-verification` | exists on that branch only | nice-to-have | 130-check CPU suite; port into campaign commit for the local pre-flight. Free (local CPU). |
| Bank-segment dataset | CPU-2 below | — | nice-to-have | Saves ~250 s repack + one failure mode; job works without it (repack in-session). |

### 6.1 Degraded Batch #1 if T9 is not ready

T9 fixes the sequential per-expert `stage_expert` loop that serializes
cold reads (the known perf issue). Without it the *host arms still run*
— they answer Q2 (exactness) and Q4-mechanism (does the tier function,
hit surface sane) — but their wall/byte numbers are labeled
**perf-lower-bound** and no host-vs-baseline speedup claim is made.

If the reviewers want Batch #1 to remain *fully decisive* without T9, the
degraded design is: **drop A2/A3 scored perf to mechanism-only arms**
(they still run reps for exactness + counters, but the campaign's causal
conclusion is carried entirely by **A0 vs A1** — the baseline-vs-VRAM-only
cell). Rationale: A1's data path is byte-identical to A0 except the
eviction score, so A0-vs-A1 is the cleanest causal pair in the whole
design and needs neither T9 nor the tier path. The host arms then become
informational for the "does it work / next bottleneck" questions. The
decisive deliverable of Batch #1 survives: mechanism + VRAM contract +
exactness.

---

## 7. Pinned campaign commit

Batch #1 pins one immutable commit = merge/rebase of
`56dad3c` (integration-lru-fix) + **T3 pydee arming** + the campaign
harness loop + probe/bench tools + (T1 + T9 if ready). Recorded in
`branch_head_at_launch` + `repository_commit`; the kernel clones that
SHA via `NATIVE_COMMIT`. No run ever launches off a moving branch head.

---

## 8. Pre-launch checklist (the cron guard + quota gate)

Before `kaggle kernels push` of the Batch #1 payload, **in order**:

1. **Local gate:** MinGW `DEE_CUDA=OFF` build PASS; focused ctest PASS
   (`test_phase2_host_tier`, `test_expert_store`, `test_vram_cache`,
   `test_engine`, `test_real_router`, `test_host_pack_cache`, +
   `test_phase2_tier_audit` if ported); MSVC syntax PASS. `py_compile`
   on the harness.
2. **CPU-1 gate:** remote CPU-session build+test on the pinned commit
   PASS (this *is* the "CUDA build / smallest CUDA mechanism test where
   possible" item — nvcc compile on a CPU session, ctest run for the
   host-side suite).
3. **Hypothesis + contract written** (§2.1, §4.6) — done by this doc.
4. **Ledger check:** GPU ledger reads 0/2 before push; write the
   `cache1-pending`-style registry stub marking GPU 1/2 *committed*.
5. **Quota check:** `kaggle` GPU quota headroom >= ~8 h confirmed
   (worst-case ~5.6 h).
6. **Cron guard (v2a runbook rule):** pause the active goal and create
   the requested cron guard *before* starting the job — the guard is a
   scheduled job-status/kill-switch check so a hung kernel is caught by
   the watchdog cadence, not discovered at the session timeout. The
   kernel itself also carries a wall watchdog (§4.9) as defense-in-depth.
7. Push + record `kernel_version` back into the registry stub.

---

## 9. CPU run plan (ledger CPU x/10)

Numbered in intended spend order. **5–10 held in reserve** — they exist
for the B1c (Phase-3) branch or a diagnosed defect rerun; they are not
allocated to speculation.

### CPU 1/10 — remote build + host-test gate

| Field | Value |
|---|---|
| RUN TYPE | CPU batch |
| RUN NUMBER | CPU 1/10 |
| QUESTION ANSWERED | Does the pinned campaign commit compile under the Kaggle toolchain (gcc + nvcc, `DEE_CUDA=ON`) and pass every host-runnable test — including the new probe/bench binaries — before GPU time is risked? |
| WHY LOCAL INSUFFICIENT | Local toolchain is MinGW/Windows with no nvcc; the audit explicitly notes CUDA paths were reviewed statically only. A Kaggle CPU session has the real toolchain and headers (build-only; ctest CUDA targets will SKIP/77 — that is expected and recorded, the *compile* is the gate). |
| ARMS INCLUDED | none (build/test only) |
| SUCCESS CRITERIA | clean `cmake -DDEE_CUDA=ON` configure + `dee_core` + all test targets + `pydee` + tools compile; `ctest` host suite all-PASS; CUDA targets SKIP-by-absence is allowed and recorded |
| FAILURE CRITERIA | any compile error, any host-test FAIL, or a CUDA target that *fails* rather than skips |
| ARTIFACTS EXPECTED | build log, ctest log, `compile_commands.json`, binary set listing |

### CPU 2/10 — dee4-v4 trace-bank segmentation + dataset publish

| Field | Value |
|---|---|
| RUN TYPE | CPU batch |
| RUN NUMBER | CPU 2/10 |
| QUESTION ANSWERED | Can the sealed 2,364-record trace bank be packaged as `dee4-v4`-style segments <= the ~20 GB dataset cap, published as a private Kaggle dataset, and mounted read-only so the GPU job skips the ~250 s repack *and* one of its failure modes? Also smoke-tests the segment/publish machinery Phase-3 needs. |
| WHY LOCAL INSUFFICIENT | Needs the mounted 167 GB shard dataset + Kaggle dataset-write API; local disk (~24 GB free) cannot hold even the 29.4 GiB bank plus margin, and the *point* is a remote artifact. |
| ARMS INCLUDED | none |
| SUCCESS CRITERIA | segment set whose metadata integrity identity == `c83462ba…`; dataset mounts read-only; GPU-side `dee4_trace` validation passes on the mounted segments (verified in Batch #1 P4) |
| FAILURE CRITERIA | segment hash/identity mismatch, dataset >20 GB cap breach, mount unreadable |
| ARTIFACTS EXPECTED | dataset slug + version, segment table `metadata.json`, integrity report |

### CPU 3/10 — `fill_replay` remote replay validation

| Field | Value |
|---|---|
| RUN TYPE | CPU batch |
| RUN NUMBER | CPU 3/10 |
| QUESTION ANSWERED | On the actual Kaggle /tmp storage regime, does the production fill path still show ~0.29–0.37 GiB/s + ~70% overlap, and does the sealed route journal replay to the sealed host hit/miss counters? This is the *interpretation key* for Batch #1's host arms. |
| WHY LOCAL INSUFFICIENT | Local storage is not the Kaggle /tmp bank regime (Phase-1's whole finding is regime-specific); `fill_replay` is host-only so it runs fine on a CPU session — it just needs the remote bank + journal. |
| ARMS INCLUDED | none |
| SUCCESS CRITERIA | `fill_timeline.json` produced; per-request service p50/p95 and lane-overlap within ~±20% of sealed v60 telemetry; replayed hit/miss counts within sim tolerance |
| FAILURE CRITERIA | replay diverges from sealed counters beyond tolerance (implies trace/bank mismatch — would invalidate Batch #1's A0 arm anyway) |
| ARTIFACTS EXPECTED | `fill_timeline.json`, summary JSON, service-time distributions |

### CPU 4/10 — Phase-3 full-store build prep (dee4-v4 segments)

| Field | Value |
|---|---|
| RUN TYPE | CPU batch |
| RUN NUMBER | CPU 4/10 |
| QUESTION ANSWERED | Can the 46x256 (~146.6 GiB) full expert store be built as resumable `dee4-v4-segmented` bucket segments on Kaggle CPU, with the manifest + completeness proof, so a B1c GPU run never pays the ~35-min+ build in-session? |
| WHY LOCAL INSUFFICIENT | ~147 GiB cannot be materialized locally (24 GB free); residential HF fetch is too slow; needs the mounted shard dataset + Kaggle disk/egress. |
| ARMS INCLUDED | none |
| SUCCESS CRITERIA | all 46 segments built + `manifest_sha256`/`universe_sha256` sealed; `p3_completeness` audit clean; resumable journal verified by one deliberate interrupt+resume |
| FAILURE CRITERIA | any segment fails integrity, completeness audit reports out-of-scope/missing routed tensors, resume corrupts a segment tail |
| ARTIFACTS EXPECTED | segment set + journal + manifest + completeness report (note: >100 GB account cap means segments live on /tmp or a volume, *not* datasets — transport plan recorded in the report) |

### CPU 5–10 — reserve

Held for: B1c Phase-3 extra build arms if the store exceeds one session;
a re-packaging rerun if CPU-2's dataset flakes; a second replay
configuration if reviewers want a different lane/depth probe; or simply
**unspent** — a reserve is a valid ledger state, not a waste.

### What must stay LOCAL (never a paid run)

- All code authoring, review, and the MinGW `DEE_CUDA=OFF` build+ctest.
- `test_phase2_tier_audit` port + all host-side test iteration.
- Payload authoring, `run_config.json` arm matrix, registry stub writes.
- Seal/verification of downloaded evidence; RUN_REGISTRY bookkeeping.
- The ws-policy sim runs that generate per-arm expected counters.
- Kernel *push* is a launch action (goes through §8), not a CPU batch.

---

## 10. Per-run cost table

| Run | Type | Est. wall | GPU-h cost | Counts | Paid only if |
|---|---|---|---|---|---|
| CPU-1 | CPU | ~1.5–2.5 h | 0 | CPU 1/10 | T3+campaign commit exist |
| CPU-2 | CPU | ~1–2 h | 0 | CPU 2/10 | bank packaging wanted |
| CPU-3 | CPU | ~1–2 h | 0 | CPU 3/10 | host-arm interpretation wanted |
| CPU-4 | CPU | ~1–3 h (+resume) | 0 | CPU 4/10 | B1c branch likely |
| **GPU-1** | GPU 2xT4 | ~2.0–2.8 h | ~4.0–5.6 | GPU 1/2 | §8 checklist all green |
| GPU-2 | GPU 2xT4 | B1a/b ~2–2.8 h; B1c ~3–5 h (full store + arbitrary prompts) | ~4–10 | GPU 2/2 | B1 evidence selects branch |
| CPU 5–10 | CPU | — | 0 | reserve | only on a listed question |

Quota sanity: worst-case GPU spend = ~5.6 h (B#1) + ~10 h (B1c) =
~15.6 GPU-h ≈ 52% of the ~30 h/week quota — fits one week; if split
across two weeks there is no pressure at all.

---

## 11. Proposed runbook deltas (for PHASE2_AB_RUNBOOK.md — I do not edit it)

1. Add the 4-arm table (baseline/VRAM-only/host-only/combined) as the
   *single-job* arm set; the current doc's "first matched pair" language
   predates the multi-arm packing rule.
2. Add the P0 hardware-lottery gate (2xT4 or mechanism-only-then-abort).
3. Add mechanism-first ordering (P2 hard abort before bank/arms).
4. Add per-arm+per-rep `generated_checkpoint.*.jsonl` incremental
   evidence rule + no-blocking-heartbeat rule.
5. Add the audit §5 segment and the pinned-direct microbench segment as
   non-arm content of Batch #1.
6. Carry the cron-guard sentence forward verbatim.
7. Add the regime-A cold-start requirement per rep (fresh teardown +
   zeroed counters) so reps are contract-A, not accidental regime-C.
8. Add "perf-lower-bound" labeling for host arms when T9 has not landed.

---

## 12. What this plan deliberately does NOT do

- No codec/compression/frequency-pin/regime-C/predictor arm — all
  rejected or out-of-contract per AGENTS.md + the ws-policy guardrails.
- No second hardware target, no extra GPU run, no "just one more arm" —
  the two-batch cap is treated as final.
- No kernel payload edits on someone else's branch; the campaign commit
  is the only vehicle.
- No spend of CPU 5–10 without a listed question.
- No claim that Batch #1 alone proves a speedup; it proves mechanism +
  one causal VRAM cell + bounded host evidence, and selects Batch #2's
  branch.
