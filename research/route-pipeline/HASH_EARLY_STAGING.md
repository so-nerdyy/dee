# A. Hash L0–L2 early expert staging

Source re-audit (accepted finding re-proven): `router_select` hash branch
(`deepseek_v4_layer_common.py:383-387`) sets `indices = tid2eid[input_ids]`
with no use of `x`; `weights` still derive from `scores = x @ W^T`
(`:375-394`). Hash layers are 0–2 (`deepseek_v4_model.py:360,596`).

## Proven

- **ID known at token start**: `resolve_hash_ids()` depends on (table,
  token id) only; unknown ids raise (fail-closed). Test:
  `test_hash_ids_depend_only_on_table_and_token`,
  `test_hash_unknown_token_fails_closed`.
- **Weights not yet known**: this prototype has no weight path at all;
  `consume()` raises until `mark_weights_ready(layer, evidence)` with a
  named hidden-state source. Tests: `test_weights_never_consumed_early`,
  `test_weights_ready_needs_evidence`, `test_unstaged_consume_refused`.

## Legal early operations (IDs alone)

Host-pack lookup → SSD read → packed-buffer prep → H2D packed → VRAM
residency insert. Forbidden and absent: routed compute before true weights.

## Measured/modeled (prototype `hash_stage.py`)

- Bytes staged: ≤18 reads × 12.75 MiB = **240,648,192 B max**/token
  (`bytes_summary`, `test_duplicate_suppression_and_bytes`).
- Reads hidden: up to 18 minus resident/host-packed duplicates (suppression
  locked in tests; redundant staging never re-submits).
- Lead time: L0 reads lead by (embedding + dense_0 + router_0); L1/L2 reads
  lead by their layer predecessors' full compute — exact ms pending live
  measurement (runner case A); the sim models it structurally.
- Extra host memory: ≤240.6 MB pinned in flight, bounded and released per
  consume; `bytes_summary` reports both submitted and suppressed.
- Duplicates: suppressed against resident + host-packed + pending sets.
- Stale/cancel: `invalidate()` at token boundaries cancels pending submits
  for accounting; resident bytes stay valid (records immutable — a staged
  unused expert wastes bandwidth, never correctness).
  Test: `test_invalidate_cancels_pending_not_resident`.

## Telemetry (every row)

`layer, expert id, id_known_t, stage_submit_t, stage_complete_t,
weight_ready_t, compute_needed_t, ready_before_demand,
suppressed_as_duplicate, cancelled` (`TelemetryRow`). The
`ready_before_demand` flag is the exact metric the live run must report.

## Admission (before benchmarking)

Exact IDs (table-gated) + exact eventual weights (evidence-gated) +
identical expert bytes (immutable records, dup-suppressed) + identical final
output (consume order unchanged). Existing campaign tolerances unchanged.
