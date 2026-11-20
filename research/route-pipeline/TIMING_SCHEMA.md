# Timing schema v1 (per token/layer records)

Each record carries `token/start_pos`, `layer`, `device`, the twelve span
fields below, `layer_wall_ms` (independent wall, may be null), and a
`provenance` label per field.

| Field | Meaning | Primary provenance |
|---|---|---|
| `route_d2h_host_wait_ms` | host wait inside route-ID D2H section | HOST_WALL |
| `native_call_wall_ms` | `moe_forward_batch_device` wall (inclusive parent) | HOST_WALL |
| `source_wait_ms` | host-pack lookup + source-read wait | HOST_WALL |
| `fill_wait_ms` | fill/reservation wait | HOST_WALL |
| `h2d_ms` | H2D completion wait (host side) | HOST_WALL / CUDA_EVENT |
| `decode_ms` | packed→FP16 decode dispatch span | HOST_WALL / CUDA_EVENT |
| `expert_compute_ms` | GEMM/activation dispatch span | HOST_WALL / CUDA_EVENT |
| `native_output_sync_wait_ms` | final sync wait (times existing sync) | HOST_WALL |
| `shared_expert_ms` | torch shared path (host wall; device interval where completable) | HOST_WALL / CUDA_EVENT / UNKNOWN |
| `combine_ms` | host weighted reduction | HOST_WALL |
| `orchestration_ms` | Python between native return and next call | HOST_WALL |
| `handoff_ms` | cross-GPU handoff at 21/22 | HOST_WALL |
| `accounted_wall_ms` | DERIVED closure sum (never a measurement) | DERIVED |
| `unknown_wall_ms` | wall minus accounted, clamped ≥ 0 | DERIVED |

Provenance: `HOST_WALL` (clock delta), `CUDA_EVENT` (existing device
intervals), `COUNTER` (bytes/counts), `DERIVED` (closure only),
`UNKNOWN` (unmeasured — carried, never filled), `NESTED` (sub-span of an
inclusive parent — excluded from closure sums).

## Nesting rules (normative)

- `NESTED_IN_NATIVE_CALL` = source/fill/h2d/decode/compute/sync spans:
  reported for attribution, never summed into closure (the parent
  `native_call_wall_ms` already contains them).
- `route_d2h`, `shared`, `combine`, `orchestration`, `handoff` are top-level
  siblings of the native call: summed.
- Device-only work overlapped by a host wait is not added separately.
- `layer_wall_ms`, when present, is the independent reconciliation anchor;
  when absent, closure is reported against the accounted sum with
  `closure_fraction` marked accordingly (see ACCOUNTING_CLOSURE.md).
