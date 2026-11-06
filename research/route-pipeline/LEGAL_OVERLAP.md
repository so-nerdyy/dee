# Legal overlap within one layer + token-level proof (Phases C + E)

## C. Source-grounded within-layer schedule

After `route_known(L)`, these are dependency-legal concurrently with the
routed experts' SSD reads (each justified, none reordered against a true
edge):

1. **`shared_expert(h_L)`** — needs only the layer input hidden state, known
   before the router runs (`layer_candidate.py:496-...`; currently executed
   after routed combine at `:488-492` by code order, not dependence).
   Biggest legal window: entire routed staging + decode + compute.
2. **Batch-submit all of layer L's expert reads at `route_known`** instead of
   the sequential prepare→stage host loop (`engine.cpp:880-886`). Device
   transfers are already async; submitting all fills first uses the
   `HostPackCache` fill lanes, then streams H2Ds back-to-back. Removes
   host-serial gaps (small with defaults; grows with miss bursts).
3. **Transfer setup + combine preparation**: pin next expert, build pointer
   tables, convert `weights_f32` (`layer_candidate.py:483`) — all
   route-known-local, none need expert outputs.
4. **Other GPU's already-known work**: only hash-layer prefetches qualify
   (DUAL_GPU_PIPELINE.md); score-layer work for any other layer is unknown
   by R5.

Resulting legal order per layer: `route → {submit ALL reads; run shared;
prep combine} → consume as each staging completes → combine → next layer`.
No official route is used before it exists; router outputs bit-identical.

## E. Token-level pipelining: proven "no" for decode

Claim: no part of token t+1's expert staging can legally begin before token t
completes. Proof by dependency chain (all edges source-grounded):

1. Staging needs official ids (R2). Ids need router inputs: hidden states
   (score layers) or `input_ids[t+1]` (hash layers).
2. `input_ids[t+1] = sample(logits(t))`; `logits(t)` needs `h_42(t)`;
   `h_42(t)` needs the full 43-layer chain (CURRENT_DAG.md spine).
3. Therefore every staging input for t+1 is downstream of t's final layer.
   No cut exists; prefetching t+1 under any guessed id would be
   predictor-style speculation, which is excluded (and explicitly NOT
   speculative decoding — nothing is decoded, so nothing is gained either).

The minimum architectural change that could alter this answer is a
speculative mechanism (draft ids + verification + rollback), which is out of
scope for an exact-staging track and is NOT proposed here.
