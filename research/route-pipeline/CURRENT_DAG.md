# Current layer dependency DAG (one decode token, DSV4 path)

Sources: `pydee/adapter.py` (Ornith-tier order, same skeleton),
`scripts/deepseek_v4_layer_candidate.py::_run_experts` (`:397-493`),
`scripts/deepseek_v4_layer_common.py::router_select` (`:349-398`),
`src/engine.cpp` (`moe_forward_batch_device_impl` `:756-1067`, `stage_expert`
`:3104`, public overload `:748-754` hardcodes `synchronize_output=true`),
`pydee.cpp` (`moe_forward_batch_device` binding).

## Per-layer op sequence (layer L, batch-1 decode)

1. `dense_attn_norm(L)`: HF attention + norms → hidden `h_L`. (No dee involvement.)
2. `router(L)`: gate GEMM → scores → top-6 ids + weights (`router_select`;
   hash layers 0–2 take ids from `tid2eid[input_ids]`, weights still from scores).
3. `route_d2h(L)`: compact int32 ids to pinned host + **full sync of the
   torch current stream** (`layer_candidate.py:448` + `host_synchronizations`).
4. `stage(L)`: per expert — host-pack fill then H2D 12.75 MiB packed
   (`engine.cpp:880-886` sequential host loop; device transfers async).
5. `decode(L)`: 3 kernels → 50.33 MB FP16 scratch (every use, not just miss).
6. `compute(L)`: 2× GEMM FP16→FP16, activation kernel, 1× GEMM FP16→FP32.
7. `layer_sync(L)`: **`cudaStreamSynchronize(compute_stream_)` before return**
   (`engine.cpp:1037-1054`; the public overload forces it on).
8. `combine(L)`: host-driven per-rank `addcmul_` loop (6 launches) + shared
   expert forward (currently AFTER routed combine, `:488-492`) + residual/norm.
9. `h_{L+1}` → next layer. After layer 21: hidden-state handoff A→host→B
   (no P2P assumed; split at 22 per P2.3 replay geometry).

## DAG edges (what blocks what)

```text
h_L → router(L) → route_d2h(L) → stage(L) → decode(L) → compute(L)
                                                        → layer_sync(L)
h_L → shared(L) ──────────────────────────────────────→ combine(L) ─→ h_{L+1}
compute(L) ──────────────────────────────────────────→ combine(L)
```

- `route(L+1)` needs `h_{L+1}` (router input is post-residual hidden) for all
  score layers → **the edge `combine(L) → route(L+1)` is the serial spine**.
  Nothing official about layer L+1 exists before layer L's experts finish.
- `shared(L)` needs only `h_L`: its only true predecessors are `dense` and
  `router`-independent work — it is ordered after `combine` today by code
  order, not by dependence.
- `stage(L)` needs `route(L)` ids (official only) — except hash layers 0–2
  (OFFICIAL_LOOKAHEAD.md).
- Cross-token edge: `sample(t) → input_ids(t+1) → everything(t+1)`.
