# Fill-path anatomy: the 16 requested stages mapped to code

All references base 217a333 (execution base). Every stage below has a
measurement hook after this track's instrumentation (marked ★); stages
without ★ use pre-existing counters.

1. **request known** — router IDs → `prepare_fp4_experts` call
   (`engine.cpp`, per layer, batches of `queue_depth`=6).
2. **enqueue** — `HostPackCache::get_batch` entry.
3. **queue wait (dedup scan)** — duplicate + hit scan (`host_pack_cache.cpp`
   get_batch head). ★ phase clock (`phase_t0`).
4. **worker wake/start** — epoch bump + `fill_start_cv_` notify; workers +
   caller lane share `next_fill_index_`. ★ `wake_begin/end`.
5. **source read submit** — `request.fill()` invocation = `fill_fp4_record`
   → `Dee4ExpertStore::materialize`. (Single call; submit≡service start.)
6. **source read service** — ONE `::pread(fd, dst, 12.75 MiB)` loop.
   ★ `pread_service_ns/calls/short_reads/bytes` atomics + per-request
   `fill_milliseconds` (pre-existing) and `fill_start_offset_ms` (new).
7. **source read complete** — pread returns full count (short reads counted).
8. **host destination acquisition** — `entry.bytes.resize(nbytes)` in the
   reservation loop (malloc + memset zero; SKIPPED for reused victim
   payloads, which move instead — 217a333 reuse path).
9. **mmap/page access** — dst pages fault on first touch (during the
   memset in step 8 for fresh buffers). ★ mincore probe on the SOURCE
   range in materialize (residency at read time).
10. **memcpy / mmap_to_pinned** — N/A on the dee4_trace path (pread writes
    dst directly; no separate copy). Zero by construction here.
11. **validation / bookkeeping** — map emplace, `ready` flags, stats,
    duplicate fan-out. ★ commit phase (inside batch wall).
12. **cache insertion / eviction** — LRU victim scan (`is_evictable`) +
    erase; evictions counted per batch. ★ `evictions` in FillBatchRecord.
13. **H2D submit** — `stage_expert` → prefetch pinned gather + async copy
    (existing `stage_enqueue_wait` span, 9.4 s decode).
14. **H2D complete** — device event (existing H2D CUDA-event stages, 5.3 s
    whole-run, hidden: readiness ≈ 0).
15. **consumer demand** — `wait_on_stream` per expert (existing
    ReadinessWait, 27 ms total: transfers always ready).
16. **consumer release** — `mark_consumed`/unpin (no wait; counter-only).

## What the existing counters already proved (whole run, both GPUs)

- 868 batches, 2481 requests (2.9/batch avg), lanes ≤3, QD ≤6.
- batch_wall sum 91.2 s vs worker sum 239.1 s (2.6× concurrency).
- per-request mean service 96 ms → **0.13 GB/s per-lane service rate**.
- reservation wall 13.0 s; reused buffers 1117 (14.9 GB, no zero-fill).
- `fill_mutex_` is NEVER held during fills (epoch/pending only) — no
  mutex serialization by code audit. `record_source_read*` run on the
  calling thread post-batch — no worker locking.
- Fills complete fully before staging starts (prepare→stage order), so
  H2D backpressure INTO fills is impossible by construction.
