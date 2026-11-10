# Fill timeline schema (occupancy, not wall accounting)

`StageProfiler::fill_timeline_json()` emits raw batch + request windows:

```json
{"batches": [{"batch_id": 1, "token": 7, "layer": 3, "device_id": 0,
  "misses": 6, "bytes": 80216064, "evictions": 2, "lanes": 3,
  "reserve_ms": 1.2, "wake_ms": 0.05, "batch_wall_ms": 210.5,
  "worker_sum_ms": 540.2,
  "requests": [{"key": 123, "start_offset_ms": 0.4, "service_ms": 88.1,
                "nbytes": 13369344, "cache_hit": false, "success": true}]}]}
```

`fill_timeline.py` derives (never measures twice):

- `reads_outstanding(t)`, `bytes_in_flight(t)` by sweep line;
- `ssd_busy_ms` = UNION of service windows (worker sums are never added);
- `starvation_gaps` = busy→idle transitions (SSD has no legal work);
- per-bucket series + self-contained HTML/SVG rendering.

## Reading the timeline (decision rules)

- **HARDWARE SATURATED**: busy_fraction ≈ 1 with max_outstanding ≥ lanes
  and per-request service ≈ rider sequential rate → device-bound.
- **PIPELINE CANNOT KEEP UP**: large starvation gaps aligned to layer
  boundaries with max_outstanding < lanes → dependency starvation
  (only ~6 known/layer, R5-confirmed); fix = earlier submission, not device.
- **PER-REQUEST PATH SLOW**: busy_fraction ≈ 1 but per-request rate ≪
  rider sequential (0.13 vs 2.9 GB/s) with low outstanding → each read is
  slow (cold faults / chunking / overhead); fix = path, not queue.
- Current evidence (sealed counters): per-request 96 ms (0.13 GB/s) with
  QD ≤ 6 and 3 lanes 2.6× utilized → points at per-request service, with
  starvation as the open second question the timeline answers.
