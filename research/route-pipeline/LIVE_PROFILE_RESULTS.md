# Live T4 host/sync profile results (measured, dual-T4 SM75)

Source identity: engine 217a333 + profiler patch aae0f41a (6 files,
additive except one const relaxation; per-arm identity proof recorded).
Baseline 17 GiB host-pack (8.5/GPU, cap verified from logs), FP4 packed,
dee4_trace, 16 tokens, lanes=3, qdepth=6, model rev 9e165c30. All arms
ACCEPT_CORRECTNESS with identical IDs/text. Kernel v8, ABA order
OFF1/ON/OFF2. Evidence: `research/route-pipeline/evidence-live/`.

## Walls (decode_wall_s, ACCEPT each)

| Arm | Wall | Position | Note |
|---|---|---|---|
| OFF1 | 71.447 | 1st | cold |
| ON | 66.233 | 2nd | profiled |
| OFF2 | 68.669 | 3rd | warm |
| (prior session) OFF1 70.5/71.759, OFF2 65.103/65.305 | — | — | same cold-first pattern |

Perturbation (ON − mean(OFFs)) = −5.2 s (−7.3%). ORDER-CONFOUNDED, not a
speedup: OFF2−OFF1 = −2.8 s same session; prior session −5.4 s. Profiler
effect unresolvable below ±3–5 s run noise. PROFILER_PERTURBING: NO —
construction adds only clock reads + preallocated event pools, zero added
syncs/calls; no mechanism exists for a multi-second shift either way.

## Decode attribution (645 rows, tokens 7–21, wall 66.233 s)

| Component | Total | Share | p50/row | Note |
|---|---|---|---|---|
| fill_wait (storage reads) | 41.990 s | 63.4% | — | host-blocked in prepare; critical |
| stage_enqueue_wait | 9.436 s | 14.2% | — | H2D submit + cache ops, host serial |
| native_output_sync | 4.894 s | 7.4% | 7.99 ms / p95 8.08 / max 8.12 | 645 rows; uniform => required drain |
| combine | 0.139 s | 0.2% | 0.21 ms | host loop |
| expert_compute dispatch | 0.256 s | 0.4% | host span | device GEMMs 1.5 s whole-run |
| decode dispatch | 0.044 s | 0.07% | host span | |
| gather_scatter | 0.114 s | 0.2% | | |
| readiness_wait | 0.027 s | 0.04% | ≈0 | transfers always ready at consume |
| route_d2h_host_wait | 0.015 s | 0.02% | 0.022 ms / max 0.16 | copy floor dominates |
| shared (device, serial) | 0.308 s | 0.5% | shared host UNKNOWN (gap, fix committed) | |
| unknown | 9.196 s | 13.9% | dense attention decode part + orchestration + journal/checkpoint | |
| **closure** | **0.861** | — | accounted 57.0 s / 66.2 s | target met, margin thin |

Prefill (43 rows, wall 93.9 s): native 68.3 s (fill 62.2 s), shared device
24.1 s → ≈98% closed. Prefill is fill-dominated like decode.

## Storage: service vs critical vs overlapped

- Worker volume (whole run): 33.17 GB, 2481 requests, 868 batches,
  batch-wall sum 91.2 s (multi-lane overlapped — NOT wall).
- Critical storage wait (decode): fill spans 42.0 s (host thread blocked).
- Overlapped: H2D device ≤5.3 s concurrent with subsequent fills;
  lane parallelism netted inside fill spans.
- Effective decode read rate ≈ 31 GB / 42 s ≈ 0.74 GB/s.
- Pack-cap paradox resolved: −52 misses ≈ 663 MB ≈ 0.9 s at 0.74 GB/s,
  below ±2–5 s order noise → observed +0.03 consistent. The deterministic
  per-miss law is falsified as a predictive model; the causal mechanism
  (blocking fills) is confirmed. Never convert bytes→wall below resolution.

## Sync analysis (Candidate C verdict input)

645 decode sync waits, p50 7.99 ms / p95 8.08 / max 8.12 ms, top layers
≈120 ms each — uniform, i.e. fixed-quantum required drain, no pathology.
Whole-run device-busy 6.89 s ≈ whole-run sync 6.53 s: the host waits on
completion that must happen before combine. No legal host work exists for
these windows (next routes unknown by R5; verified against the DAG).
Expected recoverable wall from event handoff ≈ 0. Gate (7.4% ≥ 5%) still
fires mechanically → PROMOTE with this expectation attached.

## ABC mechanics (real T4, synthetic bytes)

- A: serial p50 111 ms vs batched 209 ms (hidden NEGATIVE) — submit
  batching does not hide time; the hash question is lead-window hiding,
  still unmeasured (needs profile spans, not submit patterns).
- B: GEMM alone 0.075 ms vs under H2D saturation 3.56 ms (c=0.021,
  worst-case bound) — shared-vs-transfer contention is severe.
- C: empty sync 8.5 µs vs event 5.4 µs — barrier mechanics are µs-scale;
  loaded syncs are required completion (see above).

## Ranking (predeclared gates on measured data)

EVENT_HANDOFF PROMOTE; SHARED/ROUTE-D2H/HASH/C1-C2 HOLD (shared host
unknown + device 0.5%; D2H 0.02%; HASH hidden negative; compute 0.5%).
Next: EVENT_HANDOFF_PROMOTED — narrow event-handoff-only A/B with abort
criteria (abort if first-pair delta < resolution); predicted outcome ≈ 0
by the required-completion analysis above. No candidate implemented.
