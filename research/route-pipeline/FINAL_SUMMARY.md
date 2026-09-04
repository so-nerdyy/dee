# Route-pipeline track — final summary

No production modified (only `research/route-pipeline/`,
`dee.cpp/experiments/route_pipeline/`, `tests/` added). No router-output,
tolerance, campaign, sealed-evidence, or `research/t4-kernel-next` change.
No merge.

## The 10 required answers

1. **One layer ahead? No** for score layers (R5: router needs post-residual
   hidden). Hash-layer IDs for layers 0–2: yes, at token start.
2. **Two layers? No** (transitive chain through expert execution).
3. **Eight layers? No** (same chain, longer).
4. **Preventing dependency:** the residual stream — `route(L+1) = f(h_{L+1})`,
   `h_{L+1}` requires `combine(L)` requires `compute(L)`. Centimeters of code
   cannot cut it; only different model math could.
5. **Same-layer legal overlap:** shared-expert forward across all routed
   staging/compute; batch-submit all reads at route-known; transfer/combine
   prep. Simulated saving ≈ one shared-expert time per layer when routed
   work dominates (locked relationally in tests; ms pending calibration).
6. **Dual-GPU additional overlap: effectively none for official decode routes**
   (serial layer order + handoff on the critical path; streams independent
   but starved of early routes). The wins are intra-side and barrier cuts.
7. **Biggest avoidable barrier (candidate, NOT removed):** per-layer
   `cudaStreamSynchronize` (`engine.cpp:1037-1054`, forced at `:748-754`) —
   an event handoff suffices if every reader is proven to wait; the combined
   path already runs sync-free (`:1927-1935`). Runner-up: route-ID D2H full
   stream sync per layer (24 B of data, sync-dominated cost).
8. **Source changes for legal earlier staging:** (a) token-start hash-id
   resolve + L0–2 read submission; (b) shared-forward reorder onto an
   overlapped stream with event join; (c) batch-submit host loop reorder;
   (d) event-handoff audit + sync reduction. (a)–(c) are reorder-only;
   (d) needs the reader proof.
9. **Purely simulated:** every millisecond in `sim.py` outputs (assumed
   latencies until calibrated); the no-layer-sync saving; any "lead-N"
   schedule for score layers. Structural claims (lead 0, legal edge lists,
   barrier inventory) are source-grounded, not simulated.
10. **Return to C1/C2 when:** the campaign profiler shows SSD-staging no
    longer dominating the measured critical path — i.e. staging fraction
    below decode+compute on the profiled token timeline for consecutive
    runs. No fixed X is set here; X must come from profiler evidence
    (`--profile-timeline`), per Phase H. C1-PRE stays the queued candidate;
    C2 stays a cheap gated win; `eddb752` is preserved untouched on its
    branch.

## Phase G handoff (for Flash's research/exact-staging track)

```json
{
  "future_layer_official_lead": 0,
  "hash_layer_early_ids": [0, 1, 2],
  "hash_ids_basis": "tid2eid[input_ids]; weights still need hidden",
  "same_layer_overlap": ["shared_expert", "batch_submit_all_reads",
    "transfer_setup", "combine_prep"],
  "cross_gpu_official_overlap": false,
  "token_level_lead": 0,
  "barrier_candidates": ["per_layer_cudaStreamSynchronize",
    "route_id_d2h_stream_sync", "per_rank_addcmul_loop",
    "sequential_prepare_stage_loop"],
  "removed_barriers": [],
  "latency_source": "ASSUMED (calibrate from profiler/host.json)",
  "kernel_track": "research/t4-kernel-next @ eddb752 (untouched; queued)"
}
```

(`sim.py:handoff()` emits the live version of this record.)

## Verification performed

- `pytest tests/test_route_pipeline.py` — 10 passed, warning-free:
  determinism, fail-closed inputs, legal schedules never pessimizing,
  hit/miss ordering, shared-overlap bound, 42-sync accounting, causal edge
  order incl. the hash exception, handoff schema, +1/+2/+4/+8 table.
- The simulator caught and fixed one real modeling bug during development
  (hash-layer compute incorrectly unchained); regression locked in
  `test_edges_form_causal_chain`.
