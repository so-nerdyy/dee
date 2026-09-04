# STAGING_DESIGN.md — exact staging: bounded queue, pack budget, telemetry

Branch: `research/exact-staging` · Date: 2026-09-04
Companion docs: CAUSALITY_AUDIT.md (what is legal), LIVE_AB_PROTOCOL.md (the A/B),
PREAD_BENCH_PROTOCOL.md (the missing measurement).

---

## 1. Executive finding (changes the candidate)

The audit found that **in-step staging lead is causally impossible** in exact
mode (layer L+1's official route does not exist until layer L computes), and
that production **already implements cross-step retention** as a bounded
host-pack LRU of 682 records (8.5 GiB/GPU), configured via
`host_pack_cache_bytes` (kaggle harness: `deepseek_v4_native_generate.py:1380`).

The validated replay (pack_replay in
`dee.cpp/experiments/exact_staging/bounded_staging_queue.py`) shows:

- production pack = LRU @ 682 records (reproduces sealed v65 counters ±2);
- raising the budget to **1024 records (12.75 GiB/GPU)** removes 99 of 1252
  decode storage misses (−7.9%); at **1365 records (17 GiB/GPU)** the pack
  reaches the **offline oracle floor** (1135 = Belady @ sealed budget), i.e.
  LRU loses nothing to policy once capacity covers the 3-step working set;
- ablations that discard prefill-warmed entries are strictly worse.

**Therefore the recommended exact-staging candidate is a pack-budget increase
(config-only), not new staging machinery.** The bounded staging queue below
remains the reference for (a) the telemetry the A/B must emit, and (b) any
future in-step staging if the causality box is ever lifted (it cannot be in
exact mode).

## 2. Bounded staging queue (reference prototype)

`dee.cpp/experiments/exact_staging/bounded_staging_queue.py` — isolated,
imported by no production path. Semantics (all enforced in code + tests):

- **Official ids only.** `note_route_known(step, layer, expert)` is called at
  exactly the moment `prepare_fp4_experts` runs in production. No predicted
  expert is ever staged.
- **Bounded in-flight reads** (`max_inflight_reads`): excess requests queue
  behind slots; nothing is dropped.
- **Bounded pinned buffers** (`pinned_budget_bytes`): retained pack entries
  evict LRU; total host bytes never exceed the budget.
- **Bounded residency** (`resident_slots`, sealed v65: 281/GPU): the mock
  VRAM set is itself LRU-bounded so the prototype's accounting can never
  assume unbounded residency.
- **Cancellation / stale work**: in-flight reads owned by an older step are
  cancelled at `begin_step` (counted in telemetry).
- **No duplicate reads**: a repeated id resolves via resident / pack /
  in-flight dedup tiers, never a second disk read.
- **Deterministic ownership**: first requester owns the read; duplicates
  reference the original telemetry record.
- **Per-request telemetry** (schema `exact-staging/staging-telemetry-v1`):
  step, layer, expert, record_bytes, route_known_ts, read_submit_ts,
  read_complete_ts, h2d_submit_ts, h2d_complete_ts, compute_needed_ts,
  ready_before_demand, source_tier ∈ {resident, pack, read, dedup},
  useful_lead_ms, cancelled.

## 3. Candidate configuration for the live A/B

| Parameter | Baseline (sealed v65) | Candidate |
|---|---|---|
| `host_pack_cache_bytes` / GPU | 8.5 GiB (682 rec) | **12.75 GiB (1024 rec)** |
| route authority / router | official | unchanged (identical journal) |
| lanes / queue depth | 6 / 6 (v65) | unchanged |
| VRAM expert cache | 3.5 GiB budget (281 rec/GPU observed) | unchanged |
| correctness gates | sealed suite | unchanged |

Why 1024 and not 1365 (the oracle point):
- Host RAM at the sealed host showed **29.64 GiB available** during v65; two
  packs at 17 GiB = 34 GiB does not fit safely. Two packs at 12.75 GiB
  (+8.5 GiB total) does, with margin for the 8.5 GiB already in use.
- 1024 rec captures 99/116 of the oracle-gap miss reduction (85%) at 60% of
  the extra memory of the full floor.

Simulated wall estimate (calibrated critical-path model @2db0fde, baseline
error −1.19%; per-step read tables from the validated replay):

| Budget | Decode misses | Simulated decode wall | vs observed 72.267 s |
|---|---|---|---|
| 682 rec (sealed) | 1251 | 71.37 s | −1.24% (baseline fit) |
| 851 rec | 1187 | 68.69 s | −4.9% |
| **1024 rec** | **1153** | **67.27 s** | **−6.9%** |
| 1365 rec | 1135 | 66.52 s | −7.9% |

**These are SIMULATED estimates, not measured speedups.** The A/B decides.

Falsification condition (pre-registered): if the candidate run's sealed
decode storage misses do not drop to ≈1153 (±10), the replay's transfer to
the live host is wrong (e.g. page-cache interference or allocator effects) —
see LIVE_AB_PROTOCOL.md §4.

## 4. Mapping to production (for Codex, if adopted)

One config value: `host_pack_cache_bytes` is already plumbed
(engine.cpp:3582 `pack_cache_.set_budget(cfg_.host_pack_cache_bytes)`). The
A/B needs **no engine change** — only the harness argument. If telemetry beyond
the existing `host_pack` stats is wanted, add the per-request fields of §2
behind the existing profiler flag; do not alter scheduling.

## 5. Pread bench interface

The `DiskModel` in the prototype implements the two-point saturation law
calibrated from the sealed lane A/B (v63 3-lane p50 109.1 ms ×3 ≈ v65 6-lane
p50 215.5 ms ×6 ≈ 370 MB/s aggregate; N=1,2 unmeasured). The Phase A bench
(`tools/bench_expert_pread.py`) replaces this law with measured depth→rate
data; its JSON drops into the critical-path simulator unchanged
(PREAD_BENCH_PROTOCOL.md).
