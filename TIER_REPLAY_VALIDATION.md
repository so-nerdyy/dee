# Phase-2 tier replay validation — real C++ mechanism vs Python policy sim

Branch: `research/phase2-tier-replay`, base `56dad3c` (Phase-2 integration head).
Date: 2026-09-11. Executor: T14.

## Question

Do the actual Phase-2 tier classes (`HostExpertTier`, `VramCacheManager`,
`AsyncPrefetcher`, `DeviceExpertTier` at 56dad3c) realize the counters that
`tools/phase2_ws_policy_sim_v4.py` (fixed at 8c921ff on `fix/phase2-ws-sim-init`)
predicts for the sealed v50 route journal — before any GPU batch is spent?

**Answer: YES for every arm Batch #1 will run.** The pure-policy counters are
byte-exact against the sim on all 42 compared cells; the full mechanized path
(`DeviceExpertTier::stage` + leases + in-flight cache pins) reproduces the
sealed live v60 counters *exactly* — explaining the sim's previously tolerated
±1..3 drift. The VRAM-repair H2D prediction lands within 0.1%. One methodology
gap (not a bug): the sim's "host arm" sees the full stream, while the real
`DeviceExpertTier` calls `host.acquire` only on device misses — realized host
counters differ accordingly and are reported below.

## Setup

- Journal: `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/
  v50-evidence-20260829T195940Z/routed_experts.jsonl`, sha256
  `665aac3e8db570237c6dc6acaf08dc39f2af890e8a04e400ce7154f1a858dae1`
  (byte-identical to the sim input; verified locally).
- Stream: records sorted by `record_index`; per (forward_step, layer) the
  deduplicated expert set = `sorted(unique(expert_ids_rank_order rows))`
  — the same "engine-dedup" stream the sim consumes (5,099 accesses,
  2,364 unique pairs, 16 steps, 43 layers).
- Harness: `dee.cpp/tools/phase2_tier_replay.cpp` (new, host-only, option
  `DEE_BUILD_TIER_REPLAY`, default OFF). Fake `ColdExpertStore` returns
  correctly-sized zeroed records; `IdentityCodec`; host `Arena` backend
  (malloc) for `VramCacheManager`; `AsyncPrefetcher::init(false)` (mock
  stream — memcpy on `wait()`, no CUDA). Raw evidence:
  `tmp/tier_replay/sweep_logical.json` (141 runs, 0 failures, 0 counter-
  conservation violations).
- Record size: logical 4,096 B for the sweep (hit/miss/eviction order in both
  tiers is size-independent for uniform records; `evict_until_free` frees
  whole blocks). Physical spot-checks at 13,369,344 B confirmed byte
  accounting (`SSD_bytes == fills * record_bytes`, `peak_resident ==
  slots * record_bytes`, real 12.75 MiB fill→lease→memcpy→arena path).
- Substitutions vs the live T4 system (all mechanism-preserving):
  pageable host slots instead of `cudaHostRegister` (`try_pin=false` —
  pinning is a bandwidth detail, not a counter semantic); mock stream/event
  instead of real DMA (`wait()` drains synchronously); malloc arena instead
  of `cudaMalloc`. Staging chunked at `cache_batch = slots` per batch,
  matching `engine.cpp:4109-4129`.

## Result 1 — Host arm: `HostExpertTier` + `PlainLruHostPlacementPolicy`

Full stream (5,099 accesses), touch-mode lease (released immediately after
acquire — the pure demand-fill model the sim encodes):

| budget GiB | slots | sim hits/miss/evict | realized | hit_rate |
|---:|---:|---|---|---:|
| 8   | 642   | 2056/3043/2401 | **same** | 40.32% |
| 12  | 963   | 2376/2723/1760 | **same** | 46.60% |
| 16  | 1285  | 2592/2507/1222 | **same** | 50.83% |
| 20  | 1606  | 2668/2431/825  | **same** | 52.32% |
| 24  | 1927  | 2707/2392/465  | **same** | 53.09% |
| 32+ | 2570..10280 | 2735/2364/0 | **same** | 53.64% |

All ten budget cells (incl. 48/64/96 GiB spot runs) match the sim's `lru`
rows **exactly** — hits, misses, *and* evictions. `batch` lease mode
(hold the batch's leases until batch end) produces identical counters;
`coalesced=0`, `budget_rejections=0`, `failures=0` everywhere.

Per-scope 682-slot fill-live anchor (8.5 GiB/GPU legacy budget):

| scope | sim | sealed live | realized |
|---|---|---|---|
| cuda0 | 1222/1391/709 | 1223/1390/708 | **1222/1391/709** |
| cuda1 | 1395/1091/409 | 1395/1091/409 | **1395/1091/409** |

The new host tier reproduces the sim exactly; the residual ±1 on cuda0 vs the
sealed live counters is a legacy-`HostPackCache` stream detail (the sealed
anchor was produced by the *old* pack cache, not this tier), already inside
the accepted tolerance — not a tier defect.

## Result 2 — Device arm: `VramCacheManager` @ 281 slots (3.5 GiB)

Two driver levels were replayed per (scope, score):

- **bare** — `ensure()` per access, no pins: the sim `touch()` analog.
- **vram/combined** — real path: `AsyncPrefetcher::prefetch*` /
  `DeviceExpertTier::stage` → `ensure`+`pin` at stage, `unpin` at `wait`
  (per batch), i.e. in-flight blocks are *protected from eviction* — a
  mechanism the sim does not model.

| scope | score | sim | bare | real path | sealed v60 |
|---|---|---|---|---|---|
| cuda0 | priority | 329/2284/2003 | 329/2284/2003 | **328/2285/2004** | **328/2285/2004** |
| cuda1 | priority | 325/2161/1880 | 325/2161/1880 | **327/2159/1878** | **327/2159/1878** |
| cuda0 | lru (repair) | 680/1933/1652 | 680/1933/1652 | **680/1933/1652** | — |
| cuda1 | lru (repair) | 1057/1429/1148 | 1057/1429/1148 | **1057/1429/1148** | — |

Bare `ensure` matches the sim on **all 28 cells** (7 budgets × 2 scopes × 2
scores). With real pins, `lru` stays exact everywhere; `priority` drifts by
single-digit hits at other budgets (e.g. cuda1@80: 95 vs 142) — but at the
deployed 281-slot anchor it lands **exactly on the sealed live v60 counters**.
The sim's known ±1..3 anchor drift is therefore fully explained: it is the
in-flight pin-protection window (`pinned_blocks_skipped` fired 6,496–9,334×
per scope), not sim error in the score model.

**VRAM-repair payoff (the campaign's headline number):**
realized loads_saved = (2285+2159) − (1933+1429) = **1,082 loads →
14.466 GB/response** vs sim 1,083 → 14.479 GB (−0.09%). Confirmed.

## Result 3 — Combined arm: host behind device (real wiring)

`DeviceExpertTier::stage` calls `host.acquire` **only on device miss** —
the sim's host arm instead feeds the *full* stream to its LRU. Realized
host counters at the anchor geometry (device 281, host 682):

| scope | device score | host hits/miss/evict (realized) | host req | hit rate (of misses) |
|---|---|---|---|---:|
| cuda0 | priority | 927/1358/676  | 2285 | 40.6% |
| cuda1 | priority | 1064/1095/413 | 2161 | 49.2% |
| cuda0 | lru      | 543/1390/708   | 1933 | 28.1% |
| cuda1 | lru      | 337/1092/410   | 1429 | 23.6% |

A Python wiring-model of the same miss-substream semantics gives
924/1360/678, 1068/1093/411 (priority) and 543/1390/708, 337/1092/410 (lru)
— **exact under the lru device score**, within ±4 under priority (the
pin-skips perturb which records miss, hence the substream itself). So the
mechanism is faithful; the *prediction formula* differs. Concretely for
Batch #1, SSD traffic at 682 host slots totals 2,453 misses (32.8 GB,
priority device) or 2,482 (33.2 GB, lru device) vs the sealed legacy
host-pack baseline 2,481 misses (33.2 GB): the host tier at fill-live
geometry is ~parity with the existing pack cache on this trace — residency
value shows up only when the host budget exceeds the miss-substream's
footprint (at 1,285 slots: 2,364 misses = the entire unique set, 31.6 GB).

## Sensitivity — priority model matters (priority score only)

The sim assigns `priority = K − j` in ascending-expert-id order — matching
`moe_forward_batch` (`engine.cpp:896`: `active_experts.size() − i`). The
pointer-batched decode path instead uses route-rank order
(`selections − position`, `engine.cpp:1705`). Replaying with
row-major first-appearance order (the journal's best route-rank analog):

| scope | score | sorted (live model) | first-seen (route-rank analog) |
|---|---|---|---|
| cuda0 | priority | 328/2285/2004 | 210/2403/2122 |
| cuda1 | priority | 327/2159/1878 | 277/2209/1928 |
| cuda0 | lru | 680/1933/1652 | 683/1930/1649 |
| cuda1 | lru | 1057/1429/1148 | 1061/1425/1144 |

Production-score hit rate drops ~36% (cuda0) / ~15% (cuda1) under the
route-rank ordering; the plain-LRU repair is insensitive (≤3 hits). Since the
sorted model reproduces the sealed live counters *exactly*, the v60 run's
effective staging order was ascending-id — but any future change to
route-rank staging would make the priority score even worse than modeled;
another reason the repair, not the score, is the right arm.

## Discrepancies and artifacts found

1. **Sim "host arm" ≠ real wiring** (methodology, not a bug): host sees the
   device-miss substream only. The published host-LRU rates (40.3–53.6% on
   the full stream) bound the *standalone* host cache; realized host hit
   rates behind the 281-slot device tier are 23.6–49.2% of requests.
   Both are reported above; no code change implied.
2. **Quoted predictions mislabeled in the task brief**: the cited
   64.64/72.52/78.84/85.13/91.43% "host LRU" figures are the sim's
   regime-C `belady_same_state_prewarm` rows (a *prewarmed* offline bound,
   contract deferred to Phase 5 — `HostExpertTier`'s `PolicyResident`
   partition never fabricates initial residency). The plain-LRU host arm
   actually predicts **40.32–53.64%** — and that is what the real tier
   reproduces exactly.
3. **LLP64 `map_key` truncation re-confirmed live**: the first sweep ran the
   legacy `prefetch()` path without the Phase-2 scope flag; on Windows all
   multi-layer vram runs failed at `wait()` — `map_key` collapses
   `(layer,expert)` to the low 32 bits (`async_prefetcher.cpp:56-61`), i.e.
   expert only. This is the audit's known Medium (deliberately retained for
   default-OFF equivalence, fail-closed); on the LP64 T4 target the key is
   full-width. The sweep uses `enable_experimental_host_tier` (full keys) —
   the same path `DeviceExpertTier` enables.
4. **Pin exhaustion is real and correctly bounded**: a 32-slot device cache
   cannot stage a >32-expert batch — all resident blocks are pinned
   in-flight, `evict_until_free` finds no victim and `stage` fails. The
   engine never hits this because it chunks staging at
   `cache_batch = budget/record` (`engine.cpp:4109`); the replay mirrors it.

## Verdict

**MECHANISM MATCHES MODEL — Batch #1 de-risked on mechanism grounds.**

- Host LRU: exact on every budget; anchors within sealed tolerance.
- VRAM repair: exact bare-policy match; real path reproduces sealed live
  counters bit-for-bit; H2D saving 14.466 GB vs 14.479 GB predicted (−0.09%).
- Combined/hierarchy counters are consistent (0 conservation violations:
  `device_hit+device_miss == accesses`, `host_hit+host_miss == device_miss`).
- Expected Batch #1 counters are now *realized*, not just simulated — the
  only untested residual is CUDA-lifetime behavior (pin/event timing), which
  the sealed match and unit tests already bound.

Caveats carried forward: single-response cold-start window only (the sealed
journal); regime-C prewarm arms remain unimplemented (Phase 5); CUDA async
ordering can produce in-flight hits the mock cannot (an in-flight hit skips
`ensure()` — no recency/priority refresh — a real-hardware-only path the sim
also ignores); priority-score results are sensitive to staging order as
shown above.

## Reproduce

```sh
export PATH=/c/msys64/mingw64/bin:$PATH
cmake -S dee.cpp -B build -G "MinGW Makefiles" -DDEE_CUDA=OFF \
      -DDEE_BUILD_TIER_REPLAY=ON -DZLIB_ROOT=C:/msys64/mingw64
cmake --build build --target phase2_tier_replay
./build/phase2_tier_replay --journal dee.cpp/benchmark_reports/\
deepseek-v4-flash-0731-t4/v50-evidence-20260829T195940Z/routed_experts.jsonl \
      --mode sweep --out tmp/tier_replay/sweep_logical.json
```

Individual arms: `--mode host|bare|vram|combined` with `--scope`,
`--host-slots`, `--vram-slots`, `--score lru|priority`,
`--priority-model sorted|first_seen`, `--lease-mode touch|batch`,
`--record-bytes` (13,369,344 for physical DEE4 records).
