# Official route lookahead limit (Phase B)

Rule: a route is "official" only if computed from the model's own specified
inputs. Predictor outputs are never official (no knob, no exception).

## Lead classification (score layers, ids AND weights)

| Lead | Class | Reason |
|---|---|---|
| +1 | REQUIRES_CURRENT_LAYER_OUTPUT | `router(L+1)` input `h_{L+1}` needs `combine(L)` (CURRENT_DAG.md spine) |
| +2 | REQUIRES_CURRENT_LAYER_OUTPUT | transitive: needs `combine(L+1)`, which needs `compute(L+1)` |
| +4 | REQUIRES_CURRENT_LAYER_OUTPUT | same chain, longer |
| +8 | REQUIRES_CURRENT_LAYER_OUTPUT | same chain, longer |

"Lead N layers" of official routes is therefore 0 for all score layers: the
dependency is the residual stream itself, not an implementation detail. No
reordering, threading, or speculation changes this without changing what the
model computes.

## Hash-layer exception (exact, narrow)

- Layers 0–2 (`layer_id < n_hash_layers`, `deepseek_v4_model.py:360,596`):
  expert IDs = `tid2eid[input_ids]`, needing only the token id
  (`layer_common.py:383-387`). **IDs: EXACTLY_AVAILABLE at token start.**
- Hash routing WEIGHTS still need `scores = x @ W^T` at that layer:
  **REQUIRES_CURRENT_LAYER_OUTPUT**. Staging can start early; consumption
  cannot.
- Effect: up to 18 experts' (3 layers × top-6) SSD reads submit while early
  dense work runs. Weights/compute stay chained.

## Predictor and token rules

- Predictor-selected experts: REQUIRES_APPROXIMATION — excluded from every
  schedule in this track by construction (`sim.py` has no predictor input).
- Token t+1 (any layer, any kind): IMPOSSIBLE before `sample(t)` — the token
  id is a data dependency of the final logits (LEGAL_OVERLAP.md §E).
- Prefill note: all prompt token ids are known, so hash-layer IDs for every
  position are early-known; score-layer routes still chain layer by layer.
  Prefill staging policy can use this; decode (the target) cannot go further.
