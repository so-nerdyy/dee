# Storage verdict: why production sees ~0.74 GB/s (measured + bounded)

Status: structural case closed from sealed evidence; per-request service
split + residency + timeline shapes require the instrumented live run
(packaged, staged; see end). No number below is simulated — each is
measured, derived with stated arithmetic, or explicitly UNKNOWN.

## 1. The 42.0 s critical fill bucket (decode, 645 rows)

| Component | Value | Tier |
|---|---|---|
| Calling-thread batch waits | 42.0 s | MEASURED (FillWait spans) |
| └─ per-request worker service | ≈96 ms mean (239 s / 2481, whole-run sealed) | MEASURED |
| └─ reservation (dedup/LRU/alloc) | ≤13.0 s whole-run (existing counter) | MEASURED |
| └─ memset zero-fill (fresh allocs only) | ≈0.8 s est. (16.7 GB @ ~20 GB/s; 14.9 GB reused, no zero) | DERIVED est. |
| └─ LRU victim scans | ≈10–50 ms total by audit (24k ops/batch) | DERIVED est. |
| └─ wake/commit/mutex | µs–ms scale (new spans quantify live) | UNKNOWN (bounded small) |
| short-read chunking | UNKNOWN (new counter) | UNKNOWN |
| mincore residency at read | UNKNOWN (new probe) | UNKNOWN |

## 2. SSD busy fraction: UNKNOWN precisely; bounded structurally

Worker-service sums (239 s whole-run) across ≤3 lanes with 868 batches
beckon timeline analysis, not division. The live timeline measures it;
until then: NOT computed (refusing the Flash-bug class).

## 3. Why idle, when idle (proven structurally)

At most ~6 experts are legally known per layer call (R5; journal shows
exactly 6/6/6), so at most 6 reads submit per batch and lanes (3) drain
them in ~2 waves — then the SSD has NO legal work until the next layer's
router runs (attention + combine + sync in between). Dependency starvation
between layer batches is CERTAIN by construction; its wall share is what
the timeline quantifies. Consume-side starvation is ~0 (readiness 27 ms:
transfers always ready).

## 4. Limiter verdict: COMBINATION, ranked

1. **Per-request service slowness (dominant, measured)**: 96 ms mean per
   12.75 MiB read = 0.13 GB/s vs 2.9 GB/s rider — a ~22× gap PER REQUEST
   that no queue depth can hide at QD ≤ 6. Cause split (cold/random faults
   vs chunking vs overhead) is the open item for mincore + replay.
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

## 6. ONE implementation change with the best mechanical case

NONE yet — deliberately. The data isolates the anomaly to per-request
service (96 ms vs ~4 ms at capability) but NOT its cause:
- if mincore shows cold + replay shows pattern-sensitivity → the fix is
  FEEDING (earlier submission / deeper outstanding queue / bank relayout),
  not the fill path;
- if warm-but-slow or short-read-heavy → the fix is the READ path
  (chunking/vectorization, fd strategy).
The fill-measurement session (rider QD sweep + journal/monotonic/random
replay + mincore, packaged as kernel_fill_measure) decides between these
with one live run. Advancing any code change before that would be guessing
with a 22× uncertainty. No event-handoff experiment. No 20 GiB cache.
