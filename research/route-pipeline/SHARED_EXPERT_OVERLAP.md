# B. Shared-expert overlap

## Exact layer math (DSV4)

`routed = Σ_rank weights[rank] * expert_{ids[rank]}(h)` (rank order kept;
`layer_candidate.py:483-486` `addcmul_` loop), then `out = routed +
shared(h)` (`adapter.py:169`; `moe_layer_forward` reference agrees).
`shared(h)` input is the layer input hidden state — available before the
router runs, let alone before routed experts finish. Current serial
placement (`_shared_forward` after routed combine, `:488-492`) is code
order, not dependence.

## Earliest start

Immediately after `h_L` is ready (pre-router). No route, weight, staging, or
routed-compute dependency exists on the shared path.

## Prototype (`shared_overlap.py`)

Two-branch stream model + `exact_join`: either branch may finish first (the
other is buffered); the join always computes `routed + shared` in documented
order, and the rank loop inside `routed` is untouched. Tests:
`test_exact_join_order_preserved_despite_completion_order`,
`test_join_with_incomplete_branch_raises`.

## Sourced quantities

- Shared runtime: UNKNOWN in the sealed profile (torch path, unprofiled) —
  must be measured (runner case B); the sim default (1.5 ms) is ASSUMED.
- Overlap window: full routed staging + decode + compute per layer
  (structural; ms pending calibration).
- Extra workspace: `2 * hidden * tokens * dtype` for the buffered branch
  (16 KiB at batch-1 FP16; `extra_workspace_bytes`).
- Join sync: exactly one event (counted, never removed).
- Launches: unchanged in count (shared launches move earlier; none added or
  removed).

## Contention — the point that matters

Do NOT assume two concurrent GPU workloads speed up. Routed GEMMs and shared
GEMMs share T4 DRAM bandwidth and SMs. BUT the dominant overlapped region is
routed STAGING (host/SSD + H2D — zero SMs, near-zero DRAM), not routed GEMM:
shared compute hiding under staging is close to free, while shared under
routed GEMM splits bandwidth. `OverlapWindow.wall_interval(c)` reports the
[best, worst] wall given measured contention `c`; `c` comes from runner case
B (GEMM alone vs under H2D saturation). Until `c` is measured, expect value
to come from the staging region, not the GEMM region. Test:
`test_overlap_wall_interval_bounds`.
