# T9 repair contract: pre-acquire scheduler for the phase2 host path

Status: SIGNED-OFF DESIGN, with a measured falsifiability gate that currently
reads **FAIL** on the sealed bank (see §6). Implement only if the go
conditions in §7 hold; otherwise this document is the record of why the
repair was deferred. No engine code changed by this task (W1-T2).

## 0. What is being repaired

Under `phase2.enabled && phase2.host_enabled`, `prepare_fp4_experts`
early-returns (engine.cpp:2794) and each `stage_expert` performs
`DeviceExpertTier::stage → HostExpertTier::acquire → IdentityCodec::
materialize → ColdExpertStore::read → Dee4ExpertStore::materialize` — one
synchronous 12.75 MiB `pread` on the calling thread per host miss
(engine.cpp:3153–3157, expert_tiers.cpp:126–146, host_expert_tier.cpp:
245–270, expert_store.cpp:625–638). The repair moves those acquires off the
critical path per batch without changing routing, byte-exactness, or the
metrics contract.

## 1. Shape: pre-acquire scheduler

A bounded worker pool, owned by `Engine` (constructed iff `phase2_host`),
of `N = min(cfg_.source_read_lanes, 4)` workers — Phase-1 measured the bank
saturated at 3 lanes / QD6 with ~96% within-batch device busy, so more than
3–4 lanes only adds straggler surface (`HostPackCache::kMaxFillLanes = 8`
remains the absolute cap and the existing lanes/queue validation at
engine.cpp:3460–3468 stays authoritative).

```
prepare_fp4_experts(...)            // batch boundary, CALLER thread
  -> if (phase2.enabled && phase2.host_enabled)
       return phase2_preacquire(source_layer, experts, count);
       // replaces the bare `return true` at engine.cpp:2794 — one hook,
       // no call-site churn

phase2_preacquire(layer, experts, count):        // caller thread
  for each expert e in batch order:
      record = phase2_cold_->record(layer, e)    // bounded key, no IO
      if (cache_.is_resident(record.key.layer, record.key.expert))
          continue;                              // device hit: no consult
      map[record.key] = pool.submit(worker, record)

worker(record):                                  // pool thread
  r = phase2_host_->acquire(record, *phase2_cold_, phase2_codec_);
  map[record.key] = {r.status, std::move(r.lease)}
  // NEVER loops on Capacity; NEVER calls prefetcher_.*

stage_expert(...) phase2 branch (caller thread, authoritative order):
  entry = map.take(record.key)
  if (entry && entry.status == Ready)
      -> prefetch_host_lease(entry.lease, ...)   // unchanged enqueue path
  else
      -> exact serial path: host.acquire + Capacity reclaim loop +
         prefetch_host_lease (all on caller)
```

The pool has **no queue persistence across batches**: `prefetcher_.
begin_batch()` boundaries delimit the map; the next batch's dispatch
replaces it. Outstanding worker futures whose fills are still running when
the map is dropped simply complete — their lease is released on destruction,
leaving a Ready unreferenced slot that is immediately LRU-victim-eligible.
No join is required for correctness; a drain may be logged for telemetry.

## 2. Why the tier can take concurrency (pre-existing guarantees)

- `HostExpertTier::acquire` is concurrency-safe by design: slot selection
  under `state->mutex` (host_expert_tier.cpp:197), the materialization runs
  with the lock released (:247 `lock.unlock()` → :254 `lock.lock()`), and
  the `Filling` phase + `cv` implements the coalesced-join for a second
  acquirer (:203–216; exercised by `concurrent_duplicate`,
  tests/test_phase2_host_tier.cpp:173–193).
- `ExpertStoreColdAdapter::read` holds its mutex only for identity/lookup/
  stat mutation (expert_tiers.cpp:49–53, 80–83); `store_.materialize` runs
  lock-free (:68).
- `Dee4ExpertStore::materialize` is a positional `::pread` on a shared fd
  (expert_store.cpp:631) with atomic telemetry (`note_pread_service`,
  :140–150). Concurrent positional reads are safe.
- `HostExpertLease` is a copyable RAII slot reference; a held lease pins
  (slot, generation) — `references > 0` excludes the slot from victim
  candidacy (host_expert_tier.cpp:223) so bytes cannot be recycled under a
  queued lease.

Caller-only, never in workers: `cache_.*` (VramCacheManager is
single-caller), `prefetcher_.collect_host_sources`,
`prefetcher_.prefetch_host_lease`, `drain_pointer_batch_pending`,
metric boundaries `host_capacity_wait_ms` / `device_enqueue_ms`.

## 3. Call sites

Hook the five batched sites via the existing `prepare_fp4_experts` call —
engine.cpp ~295 (external MoE), ~571 (grouped host-batch), ~889 (grouped
device-batch), ~1699 (pointer-batched), ~4117 (decode inner loop). The
`stage_expert` loops at ~299/575/896/1705/4121 consume the lease map in
authoritative route order. **Stay serial:** ~3315 `preload_all_experts`
(scenario setup, not measured work) and ~3369 `forward_layer` (CPU/mock
fallback/oracle path — per-expert `stage+wait` semantics preserved).

## 4. Per-expert failure-semantics mapping (T9 must reproduce exactly)

Serial `DeviceExpertTier::stage` accounting (expert_tiers.cpp:104–147) and
the required pre-acquire behavior:

| serial outcome | serial counters | pre-acquire contract |
|---|---|---|
| record invalid / oversize / codec reject (:107–110) | `device_failures++`, return false; host untouched | caller-side pre-check at dispatch is OPTIONAL; if dispatched, worker's acquire returns `Invalid` (host stats untouched, host_expert_tier.cpp:195–196) → stage() → `device_failures++`, false |
| device-resident size mismatch (:113–116) | `device_failures++`, false | unreachable: resident keys are never dispatched (caller filter) |
| device hit → `prefetch()` fails (:117–124) | `device_hit++`, return false — **no `device_failures++`** (existing asymmetry; preserve it, do not "fix") | unchanged — device hits bypass the pool entirely |
| device miss → `acquire` → `Invalid`/`FillFailed` (:128,136) | `device_miss++`; FillFailed also `host_miss++`,`host_failures++` inside acquire (:246,258–261); then `device_failures++`, false | worker surfaces status verbatim; **no retry, no re-acquire** (a second acquire would double-count `host_miss`/`failures`); stage() → `device_failures++`, false |
| `Capacity` (:129–135) | `host.budget_rejections++` per attempt; caller reclaim loop `collect_host_sources(true)` + `host_capacity_wait_ms`; terminal `device_failures++`, false | worker returns Capacity marker **without retrying**; stage() runs the exact serial reclaim loop on the caller (metric boundary preserved), then serial `host.acquire` — counts identical |
| `Ready` → `prefetch_host_lease` < 0 (:139–145) | `device_failures++`, false; lease released | unchanged: enqueue stays in stage() on the caller |
| worker threw / never ran | n/a | map entry absent → stage() runs the unmodified serial acquire (zero tier stats were touched — exact) |

Net rule: **each logical request produces exactly the counter deltas the
serial path would produce**; a failed pre-acquire is delivered to stage()
as the same `HostAcquireStatus`, and only the caller may convert it to
`device_failures` + `return false`.

## 5. Nondeterminism budget (perf-only, documented)

Concurrent acquires make `slot.last_use = ++tick` ordering nondeterministic
→ different LRU victim choices across runs → different downstream hit/miss
sequences. This is a **performance-only** nondeterminism: every consumed
lease is generation-pinned to exact bytes, and `acquire` revalidates
key+`exact_bytes` before reusing a Ready slot. Byte-exactness, token ids,
and the route journal are unaffected. Same-key duplicates inside a batch
are impossible (engine dedups the batch set); a cross-batch Filling join
can only occur via a dropped-map straggler and is handled by the existing
coalesced path.

## 6. Falsifiability gate (required before merge)

Rule: `serial_wall − bounded_wall ≥ reservation+enqueue caller buckets the
repair removes`. Measured/simulated on the sealed journal
(`tools/phase2_fill_concurrency_sim.py`, `results/validation.json`):

- bounded-3-lane fill wall vs serial: **+23.1% WORSE** (106.3 s vs 86.3 s
  per response; measured W3 vs W1 batch walls, same miss stream).
- best theoretical at the observed 0.37 GiB/s ceiling: **−3.3 s/response**
  (~4%); at the measured 3-lane aggregate 0.29 GiB/s: +19 s worse.
- removable caller bucket: reservation ≤ 4.9 s/response (7.08 ms/batch,
  Phase-1 replay mean — itself an upper bound since phase2's acquire
  reservation is mutex+scan, not the fill pool's); enqueue ~13.5 s stays
  on the caller by construction (lease submission owns the transfer stream).
- **Gate: FAIL.** The repair does not beat serial on the measured /tmp
  bank; the headline win in naive arithmetic (57.4 ms "per batch") was a
  per-request service figure mislabeled — a flat 57.4 ms batch wall would
  require ≥1.3 GiB/s on a 0.37 GiB/s-ceiling device.

## 7. Go / no-go conditions for the implementer

- **NO-GO** on the current bank as a wall optimization: measured result is
  a wash-to-regression, consistent with Phase-1's feed-side verdict
  (limiter = device ceiling + ≤6-known dependency structure, not software
  serialization).
- **GO is justified only if** (any): (a) the bank moves to a device where
  one stream does not saturate — sweep shows lanes pay off once the ceiling
  exceeds ~0.6–0.7 GiB/s (e.g. the input-mount-class 2.9 GiB/s: decode fill
  43.5→11.6 s); (b) a legal ahead-of-router candidate source exists
  (speculative/oracle prefetch is a different, separately-reviewed design —
  this contract does not authorize it); (c) the goal is CPU-decoupling for
  another reason — then cap expectations at the §6 removable bucket.
- If implemented, merge-gate: replay sim must reproduce §6 within ±10%
  against a live A/B on the target bank, and the §4 counter table must hold
  in a deterministic seeded test (mock store + gated fills).
