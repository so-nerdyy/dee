# PREAD_RIDER_RESULT.md — corrected concurrent-pread measurement

Branch: `experiment/pack-cap-ab` · Date: 2026-09-07
Machine-readable: `results/pread-rider.json` (raw bench output),
`results/live/rider/` (kernel log, bank validation, stdout).
Kernel: `nivind/dsv4-pack-cap-pread-rider-storage-only-post-a-b` (v1),
dispatched AFTER both A/B sessions were terminal (contract-true rider).

## 1. What was fixed

The post-experiment rider inside session 2 failed harmlessly: `--store` was
given `metadata.json` (188 KB) where the bench requires the multi-record
file (≥ 13,369,344 B). One-line fix in `tools/session_driver.py`
(+ regression test `test_session_driver_rider_targets_experts_dee4`):

    --store experts.dee4   --journal-meta metadata.json

The standalone rider kernel additionally: embeds the repair bundle
(sha256 `76e1b437…`) → clone → checkout `217a3335` verified; rebuilds the
v50 trace bank via the repo's own `repack_to_dee4` CLI hash-gated to the
sealed v50 journal; validates bank identity (`data_sha256 =
c83462ba…` = the value observed in every A/B arm → PASS); then runs the
sealed bench at depths 1/2/3/4/6/8/12/16, patterns seq+dispersed,
repeat=2 (coldish pass + immediate warm pass). Bank build 492.6 s;
bench 21.0 s. No model, no decode, no A/B rerun.

## 2. Cache-state honesty

`coldish` = `posix_fadvise(DONTNEED)` over the store before the pass; the
advisory is best-effort and **page-cache ground truth is UNKNOWN**
(recorded per pass by the bench). `warm` = same record sequence repeated
immediately (hot page cache). The two states are never mixed. The rider VM
(T4 class, MemTotal 31.35 GiB, 30.42 GiB available at start) is the same
storage class as the A/B arms but a different instance/day — these are
storage-class measurements, not per-run reconstructions.

## 3. Results (96 records/pass, 13,369,344 B/record)

| depth | coldish seq MB/s | coldish disp MB/s | warm seq MB/s | coldish p50/p99 ms | warm p50 ms |
|---|---|---|---|---|---|
| 1 | 258 | 425 | 5107 | 53.1 / 58.3 | 2.5 |
| 2 | 2850 | 2942 | 7820 | 9.0 / 13.7 | 3.3 |
| 3 | 2901 | 2785 | 8455 | 13.3 / 19.0 | 5.1 |
| 4 | 2946 | 2895 | 9434 | 18.1 / 21.8 | 5.6 |
| 6 | 2935 | 2900 | 9156 | 27.1 / 31.4 | 5.7 |
| 8 | 2970 | 2894 | 8973 | 35.8 / 41.1 | 11.7 |
| 12 | 2869 | 2845 | 8963 | 54.6 / 65.1 | 16.9 |
| 16 | 2836 | 2798 | 8787 | 74.5 / 84.3 | 23.9 |

Reading it:

- **Saturation at ~2.9 GB/s cold, reached by depth 2–4.** Deeper queues do
  NOT add throughput; per-read latency simply scales ~linearly with depth
  (p50 ≈ 4.7 ms × depth) — the device is a shared back-end, not per-lane
  bandwidth. Actual concurrency achieved ≈ requested (0.99–15.8), so the
  concurrency is real; the back-end is the limit.
- **Warm (page-cache) reads: ~8.8–9.4 GB/s, 3.0–5.7× cold** — the measured
  basis for the tail-token position effect (a few GiB of hot file pages
  halving read-batch wall; see ORDER_EFFECT_FORENSICS.md).
- Pattern barely matters at saturation (seq ≈ dispersed within 3%); at
  depth 1 dispersed is ~1.6× seq (fewer sequential-readahead benefits).
- Zero short/empty reads in every pass; `io_mechanism = threads + os.pread,
  one fd per lane` (the position-safe variant; the shared-fd contrast was
  not requested).

## 4. What this says about the decode runs

- During decode the arms demanded ~33.17 GB in 70.6 s ≈ **470 MB/s
  aggregate — ~6× below the 2.9 GB/s cold saturation**. The backing store
  had large headroom; per-byte bandwidth arithmetic could not have governed
  the wall. This independently corroborates REVISED_SERVICE_MODEL.md:
  removing 695 MB of reads cannot buy wall time proportional to bytes.
- Decode-era per-read p50 (105–108 ms) is ~2–4× the rider's coldish p50 at
  comparable depth (27 ms @ depth 6) — decode reads ran under conditions
  the rider does not reproduce (two GPUs' streams interleaved on one
  device, 4.8–7.9 GiB MemAvailable with active reclaim, H2D/CPU load).
  Treat decode-era latency as the ground truth for the model; treat the
  rider as the device-capability envelope.
- Queue-depth 6 (production `source_read_queue_depth`) is already beyond
  the saturation knee; adding depth or lanes cannot increase cold
  throughput — consistent with the sealed lane-count neutrality result and
  with the recalibrated model's "more storage lanes ≈ 0" prediction.
