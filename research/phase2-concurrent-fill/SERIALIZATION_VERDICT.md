# Serialization verdict: phase2 host-tier cold fill IS per-expert serial — and that is NOT the regression

Status: PROVEN FROM SOURCE on `research/phase2-concurrent-fill` @ 56dad3c1
(Luna's phase2-integration-lru-fix head). Quantified by sealed-journal replay
(`tools/phase2_fill_concurrency_sim.py`, results in `results/`). Analysis
only — no engine code changed. Phase-1 anchors are measured live 2x T4
(`research/route-pipeline/results/fill_matrix_ingest.json`,
`fill_verdict.json`; STORAGE_VERDICT.md).

## TL;DR

- **Serialized: YES, structurally certain.** Under
  `phase2.enabled && phase2.host_enabled` every cold fill is one synchronous
  12.75 MiB `pread` on the calling thread, inside the per-expert `stage()`
  call. The legacy bounded fill pool is bypassed by an early return.
- **Wall per response: 86.3 s of serial cold-fill (measured-miss model),
  42.8 s of it in decode (~2.86 s/token of the sealed ~4.8 s/token decode);
  46.4 s at the task's 18.9 ms anchor.** This reproduces the sealed live
  decode FillWait bucket (42.0 s) within ~2%.
- **But parallelizing it does NOT win on this bank.** The /tmp device is
  saturated by a single stream (0.35 of the 0.29–0.37 GiB/s ceiling). The
  measured bounded-3-lane fill wall is **23% WORSE** than serial at every
  batch miss count. Falsifiability gate: **FAIL**.
- The "possible serialized cold-fill" suspicion is confirmed as a fact and
  refuted as a loss: serial single-flight is already at (above, in
  aggregate-BW terms) what the measured multi-lane pool achieved.

## (a) Proof from source — the synchronous per-expert chain

`prepare_fp4_experts` early-returns under the phase2 host path, so no batch
fill ever runs:

```cpp
// dee.cpp/src/engine.cpp:2790
bool Engine::prepare_fp4_experts(
        int source_layer, const int* experts, size_t count) {
    // Phase 2 fills the final leased host slot at the authoritative stage
    // boundary. The legacy batch cache/pool remains unchanged when disabled.
    if (cfg_.phase2.enabled && cfg_.phase2.host_enabled) return true;   // :2794
```

`stage_expert` then takes the phase2 branch on every expert:

```cpp
// dee.cpp/src/engine.cpp:3153
    if (cfg_.phase2.enabled && cfg_.phase2.host_enabled) {
        if (!phase2_device_ || !phase2_host_ || !phase2_cold_) return false;
        return phase2_device_->stage(phase2_cold_->record(source_layer, expert),
            *phase2_cold_, *phase2_host_, phase2_codec_, priority,
            current_token_, logical_layer);
    }
```

On a device miss, `DeviceExpertTier::stage` calls `host.acquire` on the
calling thread and loops only on Capacity (caller-side reclaim):

```cpp
// dee.cpp/src/expert_tiers.cpp:126
    ++metrics_.device_miss;
    prefetcher_.collect_host_sources();
    auto acquired = host.acquire(record, store, codec);
    while (acquired.status == HostAcquireStatus::Capacity) {
        const auto begin = Clock::now();
        const bool reclaimed = prefetcher_.collect_host_sources(true);
        metrics_.host_capacity_wait_ms += elapsed(begin);
        if (!reclaimed) break;
        acquired = host.acquire(record, store, codec);
    }
    if (!acquired.lease) { ++metrics_.device_failures; return false; }
    ...
    const bool ok = prefetcher_.prefetch_host_lease(acquired.lease, ...) >= 0;
```

Inside `HostExpertTier::acquire` the tier mutex is released and the
materialization runs **on the calling thread**:

```cpp
// dee.cpp/src/host_expert_tier.cpp:245
    slot.phase = HostTierState::Phase::Filling;
    ++state->stats.host_miss;
    lock.unlock();                                        // :247
    const auto begin = std::chrono::steady_clock::now();
    ColdReadResult result;
    try { result = codec.materialize(store, record, slot.data,
                                    state->config.slot_bytes); }   // :250
```

`IdentityCodec::materialize` is a pass-through to the cold store
(host_expert_tier.cpp:38–44: `store.read(record.key, dst,
record.exact_bytes)`); `ExpertStoreColdAdapter::read`
(expert_tiers.cpp:45–84) calls `store_.materialize(view, dst, n)` at :68
outside its mutex; and `Dee4ExpertStore::materialize`
(expert_store.cpp:584–645) is a single positional `::pread` loop of the
12.75 MiB record (:631). Synchronous, caller-thread, one record per call —
serial per expert. No worker, no queue, no overlap between consecutive
`stage()` calls.

### Call sites (all in engine.cpp)

| site | context | serial fill? |
|---|---|---|
| ~295–300 `moe_forward_cuda` external path | `prepare` + `stage` loop | yes |
| ~571–577 grouped host-batch MoE | `prepare` + `stage` loop | yes |
| ~889–899 grouped device-batch MoE | `FillWait`/`StageEnqueueWait` spans; fills land inside StageEnqueueWait | yes |
| ~1699–1711 pointer-batched API | `prepare` + `stage` loop | yes |
| ~4117–4122 decode inner loop | `prepare` + `stage` loop | yes |
| ~3315 `preload_all_experts` | scenario setup | serial (stays) |
| ~3369 `forward_layer` | CPU/mock fallback + oracle | serial (stays) |

PHASE2_METRICS.md already documents the accounting consequence:
"Phase-2 host filling occurs inside stage, so existing `StageEnqueueWait`
includes host acquisition, while the legacy prepare `FillWait` does no
Phase-2 fill work."

## (b) Quantification on the sealed journals

Workload: sealed v50 route journal (sha256 `665aac3e…ae1`; v60 journal is
content-identical on layer/forward_step/expert sets). 688 batches =
16 forwards × 43 layers; per record the engine stages the sorted unique
expert set (topk 6 in decode → ≤6-expert batches; prefill rows dedup to
22–41). GPU split at layer 22 (cuda0: 22 layers, cuda1: 21). Per-GPU tier
model, both sealed-validated: device `engine_priority_lru` @ 281 slots
(3.5 GiB arena) → host `PlainLruHostPlacementPolicy` LRU @ 682 slots
(8.5 GiB budget). A host miss == one real pread.

Validation vs sealed counters (sim == the corrected v4 policy sim's
numbers; ±1 boundary delta is pre-existing convention noise):

| counter | sim cuda0 | sealed cuda0 | sim cuda1 | sealed cuda1 |
|---|---|---|---|---|
| host pack hits (single-level check) | 1222 | 1223 | 1395 | 1395 |
| host pack misses | 1391 | 1390 | 1091 | 1091 |
| pack evictions | 709 | 708 | 409 | 409 |
| device hits (two-level) | 329 | 328 | 325 | 327 |
| device misses (cold_loads) | 2284 | 2285 | 2161 | 2159 |

Two-level phase2 stream: **2 453 preads/response** (1 224 decode,
1 229 prefill; host hit rate on the device-miss substream 40.5%/49.4%).
Decode batch miss distribution: m ∈ {0..6}, mean ≈1.9 (cuda0 mostly 1–3,
cuda1 mostly 0–2).

### Fill-wall models (per response = 16 forwards, both GPUs sequential in time)

| model | decode | response | ms/decode-tok |
|---|---|---|---|
| serial, task anchor 18.9 ms/miss | 23.15 s | 46.39 s | 1 543 |
| serial, measured miss service 35.5 ms (lanes1) | 43.46 s | 87.11 s | 2 898 |
| serial, measured miss service 33.8 ms (qd1) | 41.32 s | 82.81 s | 2 755 |
| serial, measured per-batch W1(m) | 42.84 s | 86.34 s | 2 856 |
| bounded-3, naive "57.4 ms/batch" flat | 30.65 s | 33.12 s | 2 043 |
| bounded-3, wave model 3×18.9 ms/wave | 35.74 s | 59.74 s | 2 383 |
| **bounded-3, measured W3(m)** | **52.96 s** | **106.30 s** | **3 531** |
| bounded-3, fair-share @ measured agg 0.29 GiB/s | 52.55 s | 105.32 s | 3 504 |
| bounded-3, fair-share @ best ceiling 0.37 | 41.65 s | 83.04 s | 2 777 |
| bounded-4, fair-share @ best ceiling 0.37 | 41.58 s | 82.95 s | 2 772 |

Cross-check: single-level legacy stream serial estimate 88.1 s vs the
sealed v60 4-lane/QD8 pool's actual fill_batch_wall **97.2 s** — the sealed
concurrent pool paid *more* fill wall than serial single-flight. And the
serial model's decode estimate (42.8 s) reproduces the sealed live decode
FillWait bucket (41.99 s / 645 rows) within ~2%.

### Why the naive "~57.4 ms/batch wins" reading is wrong

57.4 ms is the measured **per-request** service under 3-way device sharing
(production replay), i.e. ≈3× the 18.9 ms single-flight figure. A 6-miss
batch filled "in 57.4 ms" implies 76.5 MiB in 57.4 ms ≈ 1.30 GiB/s — 3.5×
above the measured bank ceiling; physically impossible on this store. The
correct per-batch model is a fair-share makespan: k concurrent reads each
get B/k, so wall ≈ m×R/B regardless of lane count — equal to serial when a
single stream already saturates. Measured, the 3-lane pool is strictly
worse at every miss count (W3(m)/W1(m) ≈ 1.21–1.24; e.g. m=3: 127.1 vs
105.8 ms; m=6: 261.9 vs 221.6 ms), because interleaved streams degrade the
bank aggregate (0.29 vs 0.35 GiB/s) and stragglers extend the batch wall.

Serial fill on this device is therefore already at (slightly better than)
the achievable concurrency bound: **the recoverable wall from within-batch
parallelism is ≤0 ms on the measured bank, ≤~3.3 s/response (~210 ms/tok)
at the best observed ceiling.**

## Sensitivity — when the repair WOULD win

`results/bandwidth_sweep.csv`: lanes only help once the device ceiling
exceeds the single-stream rate (0.35 GiB/s). At a 0.675 GiB/s-class bank
(the rate implied by "18.9 ms/miss"), L=3 saves ~39% of decode fill wall
(43.5→26.7 s); at the input-mount capability (2.9 GiB/s), ~73%
(43.5→11.6 s). The lever is bank placement, not lane count — Phase-1's
feed-side verdict.

## Limitations

- Miss-service anchors come from the fill-live replay (same device, same
  record size, 15.7–32.7% page-cache residency); the phase2 host tier has
  no sealed live run yet — its counters exist only in synthetic tests.
- Per-stage caller CPU (drain/enqueue ~ms) is excluded from fill wall; it
  is the reservation+enqueue bucket accounted in the gate, not a pread.
- Prefill batches exceed 6 unique experts (22–41); W(m) tables extrapolate
  linearly past 6 — conservative, matches the measured slopes.
