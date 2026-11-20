# Accounting closure method

## Rule

```
accounted = route_d2h + native_call_wall + shared + combine
          + orchestration + handoff          (top-level siblings only)
unknown   = max(0, wall - accounted)
closure   = accounted / wall                 (never forced to 1.0)
```

Nested children (`source/fill/h2d/decode/compute/sync` inside the native
call) are reported in `by_field_nested_ms` and excluded from the sum.
`validate_record` fails closed on malformed input (missing meta, negative
time, non-numeric values, bad provenance, null-with-non-unknown
provenance). Implementation: `host_profiler.compute_closure()`.

## Flash-bug avoidance (worker time vs wall)

The previous bug summed overlapping worker durations as wall components.
This schema makes that structurally impossible: workers (fill lanes,
transfer stream, device kernels) carry NO wall timers — only byte counters
and CUDA-event device intervals. The sole wall timers sit on the calling
thread's waits (`FillWait`, `ReadinessWait`, `NativeOutputSync`,
`route_d2h_host_wait_ms`). Overlapping waits on one thread still serialize
in the sum (correct: the thread did wait that long); parallel device work
covered by a wait is counted once via the wait.

## Worked shape (illustrative, not data)

10 layers × {native_call 80, sync(nested) 8, shared 8, d2h 3, combine 1} ms
against a 1000 ms wall → accounted 920, unknown 80, closure 0.92, nested
detail preserved separately. A useful profiler run targets ≥0.85 honest
closure; below that, promote nothing (decision rules hold by default).

## What closure cannot do

It cannot attribute genuinely concurrent host work twice, split a wait into
causes (e.g. how much of a sync is "unrelated work" — that needs the event
timeline, reported alongside, not subtracted), or rescue missing spans.
`unmeasured_top_level_fields` counts the gaps explicitly.
