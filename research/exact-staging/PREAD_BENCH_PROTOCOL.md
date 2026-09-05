# PREAD_BENCH_PROTOCOL.md — measuring true concurrent pread capability

Branch: `research/exact-staging` · Date: 2026-09-04
Tool: `tools/bench_expert_pread.py` (standalone; runs on the Kaggle host
against the `/tmp` trace store). Status: **written and locally validated for
bookkeeping only** — no local store exists on this Windows host, so **no
measurements are reported here**. This protocol defines how to produce them.

> **v2 audit (2026-09-04, recalibration installment):** the bench was audited
> against the Phase G spec and passes unchanged — required depths
> {1,2,3,4,6,8,12,16} (default `QUEUE_DEPTHS`), representative 13,369,344 B
> reads, per-lane fds (shared-fd contrast opt-in), seq/dispersed/journal
> patterns, coldish/warm separated with `page_cache_ground_truth: unknown`,
> and the required outputs (aggregate MB/s, p50/p90/p99, wall, cpu,
> requested/completed bytes, `concurrency_achieved_est`). No local
> measurement is substituted for Kaggle. A ready-to-run command block is
> packaged in `results/pread_kaggle_package.json` for Astra.

---

## 1. Why this measurement gates the model

The calibrated critical-path model's dominant term is the saturated expert
store; its single largest uncertainty is the store's **true concurrent pread
capability** as a function of queue depth. Sealed evidence only pins two
points (v63 3-lane p50 109.1 ms, v65 6-lane p50 215.5 ms — both ≈370 MB/s
aggregate) and the model's fitted 320 MB/s sits inside a 320–459 MB/s band
derived from byte accounting. The pack-budget candidate's simulated −6.9% and
every future staging scenario scale with this number. Measuring depth
1→16 pins the storage floor to ±2% (previous track's stated blocker).

## 2. Command (Kaggle host)

```bash
# coldish + warm, seq + dispersed, all depths, plus sealed demand order:
python tools/bench_expert_pread.py \
    --store /tmp/dee4/experts.dee4 \
    --journal routed_experts.jsonl \
    --journal-meta experts.dee4.metadata.json \
    --records 96 \
    --repeat 2 \
    --shared-fd-variant \
    --label "kaggle-t4-$(date -u +%Y%m%dT%H%MZ)" \
    --out pread_capability.json
```

- `--records 96` = one decode step's reads (sealed v65 max per-step SSD reads).
- `pattern=journal` requires the dee4 `metadata.json` to map
  (layer, expert) → record index; the bench errors with the observed keys if
  the schema differs.
- If no store is mounted, run seq/dispersed only (omit `--journal*`); the
  journal pattern is a nice-to-have, not required.

## 3. What it measures (per pass)

Aggregate MB/s, per-read p50/p90/p99/max, wall s, CPU s, bytes requested vs
completed, short/empty reads, concurrency achieved (`busy/wall`), at queue
depths 1, 2, 3, 4, 6, 8, 12, 16, for patterns `seq` (sequential-ish) and
`dispersed` (shuffled record order) and optionally `journal/fwdK` (sealed
demand order), each in a **coldish** attempt (best-effort
`posix_fadvise(DONTNEED)`; page-cache ground truth is reported as UNKNOWN —
the bench does not pretend otherwise) and a **warm** repeat.

Concurrency honesty (mechanism, from the tool's own output header):
- threads + `os.pread`, **one fd per lane** (independent file positions);
- a `--shared-fd-variant` pass measures the contrast (shared fd serializes at
  the file position on some platforms);
- **kernel-level async (aio/io_uring) is NOT used and NOT claimed**; if
  threads under-report kernel concurrency, that is itself the finding —
  production `source_read_lanes` uses the same mechanism class.
- Windows fallback (`lseek+read`, no `os.pread`) exists **only** so the test
  suite can validate bookkeeping locally; output on Windows must never be
  reported as a measurement (the tool prints this warning itself).

## 4. Output contract (`exact-staging/pread-bench-v1`)

Top-level: `schema, label, store, store_bytes, record_bytes, record_count,
io_mechanism, concurrency_caveat, results[]`. Each result: pattern,
cache_state, queue_depth_requested, fds, reads, bytes_requested,
bytes_completed, wall_s, cpu_s, aggregate_mb_s, p50/p90/p99/max ms,
concurrency_achieved_est, fadvise_available, page_cache_ground_truth="unknown".

Ingestion into `research/exact-critical-path` (Phase G): map
`aggregate_mb_s` at each depth to `ssd_aggregate_mb_s`; rerun the calibration
grid restricted around the measured value; report how the −26.1%-style
staging floors and the −6.9% pack-budget estimate move. **Do not modify that
branch** — run the ingestion in a scratch checkout and hand the resulting
calibration delta to Codex.

## 5. Acceptance / interpretation rules

- The curve `aggregate_mb_s(depth)` is the deliverable; saturation depth =
  smallest depth within 2% of the plateau.
- If coldish ≈ warm at large depths, the page cache defeated `fadvise`:
  report the run as warm-cache-only and mark the cold claim UNKNOWN (retry
  with a store larger than RAM if available).
- Sealed cross-check: depth 6 dispersed should land near the sealed
  ~370 MB/s aggregate; a large deviation flags a changed storage backend and
  invalidates comparability (report, do not average).
