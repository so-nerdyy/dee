# Host/sync profiler — what it measures and how it stays exact

Instrumentation-only track: nothing is removed, reordered, or optimized.
Every addition is a timer read or record append, disabled by default.

## C++ (`profiling.h` / `profiling.cpp`, additive)

- `HostSpan` (14 spans: LayerWall, NativeCallWall, SourceLookupWait,
  FillWait, StageEnqueueWait, ReadinessWait, DecodeWait, ExpertComputeWait,
  GatherScatterWait, NativeOutputSync, SharedExpert, Combine, Orchestration,
  Handoff) + `SpanProvenance` (HOST_WALL/CUDA_EVENT/COUNTER/DERIVED/UNKNOWN/
  NESTED) + `HostLayerRecord{token,layer,device,ms[14],counters[4]}`.
- `HostSpanGuard` / `HostLayerScope` RAII (early-return safe); all methods
  no-op unless `enabled_` (member default `false`; `EngineConfig`
  `profile_stages=false` default; no call-site default changed).
- `host_layer_records_json/csv()` emitters; unmeasured spans serialize as
  `{"ms":null,"provenance":"UNKNOWN"}` (CSV: empty cell).

## engine.cpp markers (all in `moe_forward_batch_device_impl`)

Layer scope + NativeCallWall at entry; FillWait around
`prepare_fp4_experts`; StageEnqueueWait around the stage loop; ReadinessWait
around `wait_on_stream`+pin; DecodeWait around
`decode_fp4_cache_block_to_scratch`; ExpertComputeWait around the batched
SwiGLU call; GatherScatterWait around gather/scatter dispatch;
NativeOutputSync around the EXISTING final `cudaStreamSynchronize` (times
it; adds none). One `const bool`→`bool` relaxation (guarded assignment);
verified by count that no other statement changed shape
(`test_no_arithmetic_or_routing_change`).

## Python (`deepseek_v4_layer_candidate.py`, `DEE_HOST_PROFILE=1` gated)

Split route-D2H timers (copy submit / wait start / wait end), native-call
wall, combine-loop wall, deferred CUDA-event pair around shared work
(recorded, never synchronized; elapsed resolved at dump, incomplete events
→ UNKNOWN). Default off: one flag check per forward. `dump_host_profile()`
writes the rows.

## Phase mapping

- A (route-D2H split): Python timers + `ids_bytes`/copy counters.
- B (native decomposition): C++ spans; inclusive parents
  (LayerWall/NativeCallWall) + exclusive children; unprovable splits stay
  UNKNOWN/NESTED, never derived by subtraction.
- C (output sync): NativeOutputSync host span per layer/token.
- D (shared): host wall always; CUDA-event interval where completable;
  any sync-added measurement labeled PROFILE_PERTURBING (this design adds
  none — events only).
- E (combine/orchestration/handoff): Python combine wall; orchestration and
  handoff spans reserved in schema for the model-runtime caller (C++
  Handoff span wired when the handoff crosses the engine).
- F (storage attribution): SourceWait/FillWait host spans + existing
  request-kind counters (resident/inflight/cold) + H2D CUDA events. CRITICAL
  HOST WAIT = span walls on the calling thread, never summed worker
  durations (the Flash bug class is structurally impossible: workers have no
  timers, only the waiter does).

## Overhead (Phase H)

PROFILE_OFF: one `enabled()` branch per marker + two RAII constructions per
native call; Python: one flag check per forward. No extra syscalls, CUDA
calls, allocations, or syncs in either mode. Matched off/on comparison is
part of the Kaggle protocol; instrumented timings are not acceptance
evidence unless campaign rules permit.
