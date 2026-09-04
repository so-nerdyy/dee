# Pipeline prototype (Phase F)

`dee.cpp/experiments/route_pipeline/sim.py` — discrete-event simulator for
one decode token over 43 layers. A prototype was built (rather than the
"minimum-change" fallback) because three pieces of useful legal overlap DO
exist: hash-prefetch, shared-expert overlap, and batch submission.

## What it models

Events `route_known → stage_submit → stage_done → consume → shared_done →
combine → (handoff) → next layer`, with rules R1–R6 enforced fail-closed
(violations raise; predictor routes have no representation). Schedules:
`baseline`, `+hash_prefetch`, `+shared_overlap`, `+no_layer_sync`,
`+batch_submit`, and combined. `sim.py` runs standalone and prints the
Phase G handoff JSON.

## Calibration (required before any number is cited)

All `Latency` defaults are ASSUMED placeholders. Calibrate from: per-expert
H2D from `host.json` pinned medians (consumer-harness track), decode/GEMM
from the T4 stage profile (`--profile-stages`), sync costs from the
timeline (`--profile-timeline`), hit rate from cache telemetry. Until then,
millisecond outputs are SIMULATED; the shippable outputs are the legal edge
lists and the bound-by chains.

## What it proves

- The R5 serial spine (every score-layer route bound by the previous
  combine) under assumed latencies — i.e., staging lead is structurally 0.
- Hash-prefetch and shared-overlap are legal and never pessimizing
  (locked by `test_legal_schedules_never_slower_than_baseline`).
- The per-layer sync removal hypothesis is quantified as 42×sync IF the
  event-handoff proof is done (it is not — see DUAL_GPU_PIPELINE.md).

## Minimum change if the prototype were productized (not done here)

1. Resolve hash ids at token start; submit L0–2 reads before layer-0 dense
   work (Python caller + engine submit path).
2. Move `shared_forward` before/overlapping routed staging on a second
   stream with event join at combine.
3. Batch-submit a layer's reads at `route_known` (host loop reorder only).
4. Event-handoff sync reduction with a reader-proof audit. In that order;
   stop where measured value stops.
