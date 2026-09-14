# Storage verdict: bank reads run at the /tmp device ceiling (measured live)

Status: MEASURED live on 2x T4 20260909
(`dee.cpp/experiments/route_pipeline/fill-live-t4x2-20260909/`,
ingest: `research/route-pipeline/results/fill_matrix_ingest.json`).
Phase 1 closed — every open item below now has a live number. No number
here is simulated.

## 1. The 42.0 s critical fill bucket (decode, 645 rows)

| Component | Value | Tier |
|---|---|---|
| Calling-thread batch waits | 42.0 s | MEASURED (FillWait spans) |
| └─ per-request worker service | ≈96 ms mean (239 s / 2481, whole-run sealed) | MEASURED |
| └─ reservation (dedup/LRU/alloc) | ≤13.0 s whole-run (existing counter) | MEASURED |
| └─ memset zero-fill (fresh allocs only) | ≈0.8 s est. (16.7 GB @ ~20 GB/s; 14.9 GB reused, no zero) | DERIVED est. |
| └─ LRU victim scans | ≈10–50 ms total by audit (24k ops/batch) | DERIVED est. |
| └─ wake/commit/mutex | µs–ms scale (new spans quantify live) | UNKNOWN (bounded small) |
| short-read chunking | 0 short reads / 6046 preads (1.065 preads per call) | MEASURED live |
| mincore residency at read | 15.7% (production, coldest replay); 29.7%/32.7% later cases = page-cache warming by run order | MEASURED live |
| wake/commit/mutex | wake p95 320 ms prod / 54 ms qd1 (batch join wait, part of wall, not service) | MEASURED live |

## 2. SSD busy fraction: MEASURED — production keeps the disk 96% busy

Within-batch union of request windows over batch wall (never worker-sum
division): production (lanes 3, QD 6) busy mean 0.96; QD 1 busy 0.53
(disk idle ~half — underlap, not device speed). Worker/wall 2.49 of 3
lanes: overlap is already good; the ceiling is the device, not software.

## 3. Why idle, when idle (proven structurally)

At most ~6 experts are legally known per layer call (R5; journal shows
exactly 6/6/6), so at most 6 reads submit per batch and lanes (3) drain
them in ~2 waves — then the SSD has NO legal work until the next layer's
router runs (attention + combine + sync in between). Dependency starvation
between layer batches is CERTAIN by construction; its wall share is what
the timeline quantifies. Consume-side starvation is ~0 (readiness 27 ms:
transfers always ready).

## 4. Limiter verdict: COMBINATION, ranked

1. **Cold device ceiling of the bank store (dominant, measured)**: all
   patterns and concurrencies converge on 0.29–0.37 GB/s on the /tmp
   backing store (production 0.29, lanes1 0.35, qdepth1 0.37, rider seq
   0.33–0.38 flat over lanes 1–8, rider rand 0.37–0.51). No lane scaling:
   the disk, not the software, is the cap. The 2.9 GB/s rider datum is
   the INPUT mount — live fills run at the /tmp ceiling.
   Per-request: 57.4 ms mean / 162 ms p95 production (3-wide batches
   serialize on the device: reserve p95 53 ms ≈ one service time),
   18–19 ms single-flight. pread ≈ 100% of service (650 s ≈ 651 s
   worker sum): zero software overhead, zero short reads.
2. **Legal dependency structure (bounding)**: ≤6/batch, layer-serial
   submission guarantees intermittent SSD idleness regardless of device
   speed. Even infinite bandwidth leaves the submit gaps.
3. **Software serialization: RULED OUT** (lock-free fill path by audit;
   only epoch/pending atomics + join wait).
4. **Host memory path (minor, bounded)**: memset ≈ 0.8 s + H2D-gather
   memcpy ≈ 1–2 s; reuse already avoids ~half the zeroing.
5. **H2D backpressure: RULED OUT** (fills complete before staging starts
   by construction; readiness ≈ 0 independently confirms).
6. NUMA/CPU contention: unmeasured, likely minor; lane-sweep replay bounds it.

## 5. Perfect-utilization ceiling (theoretical, labeled)

Decode storage ≈ 31 GB at 2.9 GB/s ≈ 11 s vs 42 s measured → fill could
at most shed ~31 s IF every byte streamed at capability with zero gaps,
i.e. decode wall 66 s → ~35 s, all else equal. This is an UPPER bound, not
a plan: it requires both fixing per-request service AND eliminating
submit gaps. Never converted to tok/s.

## 6. Cause decided: FEED-side. Phase 1 closed, Phase 2 = engineering

mincore shows cold (15.7%) + replay shows NO pattern/concurrency
sensitivity (all ≈0.3 GB/s) + pread ≈ 100% of service with zero short
reads → the fix is FEEDING (fewer/smaller fills, earlier submission,
bank placement), not the read path, which needs nothing. The narrowed
Phase 2 plan applies unchanged. No event-handoff experiment.
No 20 GiB cache.
