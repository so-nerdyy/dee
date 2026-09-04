# C. Event-based layer handoff

## What the sync protects (audit; nothing removed)

The per-layer `cudaStreamSynchronize` (`engine.cpp:1037-1054`, forced at
`:748-754`) covers: output readiness, routing-buffer lifetime, packed expert
lifetime, decoded-scratch lifetime, host-pack fill ownership, error
delivery, cache-eviction safety, D2D gather/scatter completion, torch
consumer readiness. Each maps to an event wait in the prototype; eviction
and error paths are the highest-risk mappings (a waited eviction must prove
no stream still references the block; an error must surface at the waiter,
never as a silent consume).

## Dependency DAG (narrow handoff)

```text
produce(resource) → record(event) → consumer stream waits → consume
       → release(resource) after waited event
```

Prototype `event_handoff.handoff_pattern()` implements exactly this with
counts (1 event, 0 syncs). Error case: producer failure flags the event;
the waiter raises (`test_producer_error_propagates_not_swallowed`).

## Exactness/lifecycle risk

- Numerical: none (ordering only; zero arithmetic change).
- Lifecycle (the real risk): premature resource reuse (use-after-free),
  premature eviction under an unwaited stream, double-recorded events,
  swallowed producer errors, cleanup reordering. Each has a negative test:
  `test_missing_wait_is_detected`, `test_premature_reuse_detected`,
  `test_double_record_rejected`, `test_producer_error_propagates_not_swallowed`.
- Reader proof still required per consumer (torch combine stream, eviction
  path, error path) before any production change — listed, not done.

## Implementation shape (if ever productized)

Export one event per layer-batch from the engine; torch consumer waits via
`torch.cuda.Event` interop instead of host sync; release paths keyed off the
waited event; keep exactly one end-of-token host sync until the reader audit
is sealed. Estimated touch: `engine.cpp` handoff seam + `pydee.cpp`
binding + `layer_candidate.py` consume path. No kernel, codec, or scheduler
change.
