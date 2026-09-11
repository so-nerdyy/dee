# PHASE2_CAUSAL_TIMELINE.md — exact causal timeline, one representative decode layer/token

Branch: `research/phase2-early-submission` (independent systems/causal research;
no production change, no merge, Luna's branch untouched, Flash's tournament untouched).

Provenance labels: **MEASURED** = sealed live dual-T4 evidence (Phase-1
`fill-live-t4x2-20260909`, route-pipeline profiled arm ON, 688-row sealed journal:
43 prefill + 645 decode rows, run `host-sync-profile-17g-20260905-v1`);
**DERIVED** = exact arithmetic on sealed inputs; **SIMULATED** = this track's
stdlib model (`sim_early_submit.py`); **THEORETICAL** = closed-form bound.
Nothing here ran a GPU.

---

## 1. Representative row (decode, batch-1, top-6, means over 645 sealed rows)

Per-row means: wall **102.7 ms** (MEASURED: 66.233 s / 645).

```
t=0.000 ms   hidden_L ready (post-residual; input to router AND shared path)
  |
  | router device GEMM + top-k  (torch stream; device time inside UNKNOWN bucket)
  v
t=R          earliest_route_known  (router outputs exist on device)
  |
  | route D2H: compact int32 copy + torch current-stream sync
  |            MEASURED 0.022 ms mean / 0.16 ms max  (copy floor dominates)
  v
t=R+0.02ms   route result on host (ids_host); PYTHON/ENGINE BOUNDARY
  |
  | native entry: ID validation + 256-expert group build + batch-buffer check
  | DERIVED ~0.1 ms (µs-scale loops; BatchConstruction span, never on top-5)
  v
t=R+~0.1ms   earliest_storage_submit (all 6 IDs legally submittable HERE)
  |
  | prepare_fp4_experts: per-miss metadata resolve (expert_store->get +
  |   configure_fp4_quantized, calling-thread serial) + get_batch reservation
  |   (dedup scan + LRU victim scan + vector resize/memset-or-reuse)
  |   MEASURED reservation 7.08 ms mean / 52.9 ms p95 per batch
  v
t=R+~7ms     actual_storage_submit (fill workers woken; preads in flight)
  |             lanes=3, qdepth=6, <=6 reads/batch (journal: exactly 6/6/6)
  |
  | source read service: ONE pread per 12.75 MiB record
  |   MEASURED 57.4 ms mean prod (3-wide) / 18.9 ms single-flight
  |   MEASURED pread ~= 100% of service, 0 short reads / 6046 preads
  |   MEASURED within-batch disk busy 0.96 (overlap already good)
  v
t=R+~65ms    storage_complete  (MEASURED critical fill 65.1 ms/row mean)
  |              (42.0 s / 645; the Flash domain: FEWER/SMALLER fills)
  |
  | stage_expert loop (SEQUENTIAL host loop, engine.cpp:883-886):
  |   consume_if_present + pinned-slot gather memcpy + async H2D submit
  |   MEASURED stage_enqueue_wait 14.6 ms/row  (9.436 s / 645)
  |   H2D self cost DERIVED 2.4 ms/record @5.54 GB/s (fitted, x-checked)
  v
t=R+~80ms    expert_ready (all 6; MEASURED readiness_wait 0.04 ms/row:
  |            transfers ALWAYS ready at consume — H2D never exposed)
  |
  | decode (packed->FP16, bounded scratch) + compute (2xGEMM + act + 1xGEMM,
  |   sequential per expert)  MEASURED device GEMMs ~1.5 s whole-run
  |   (~2.3 ms/row); dispatch host span 0.4 ms/row
  v
t=R+~83ms    layer_sync: FORCED cudaStreamSynchronize(compute_stream_)
  |            MEASURED 7.6 ms/row, p50 7.99 / p95 8.08 (uniform required drain)
  v
t=R+~91ms    combine: host rank-order addcmul_ loop + shared forward
  |            MEASURED 0.21 ms/row (shared DEVICE 0.5 ms/row; shared HOST unknown)
  v
t=R+~91ms    hidden_L+1  (+ orchestration/dense-attention gap ~11 ms/row mean
                 inside the 9.2 s UNKNOWN bucket) -> next layer's router
```

Closure check (DERIVED): 65.1 + 14.6 + 7.6 + 0.2 + 0.4 + 0.02 + ~14.3 ≈ 102.7 ms. Closes.

## 2. Required timeline definitions

| Symbol | Value (representative decode row) | Provenance |
|---|---|---|
| `earliest_route_known` | router device completion (t=R) | structural |
| `earliest_storage_submit` | native entry, t=R+~0.1 ms | DERIVED |
| `actual_storage_submit` | fill workers woken, t=R+~7 ms mean (p95 ~53 ms) | MEASURED reservation |
| `storage_complete` | t=R+65.1 ms mean | MEASURED fill_wait |
| `earliest_h2d_submit` | per-miss fill completion; for HIT experts, t=earliest_storage_submit | structural |
| `actual_h2d_submit` | after batch join + sequential stage loop, spread over +65..+80 ms | MEASURED enqueue |
| `expert_ready` | t=R+~80 ms (H2D hidden; readiness ~0) | MEASURED |
| `compute_needed` | after ALL staged (sequential per-expert compute loop) | structural |
| `compute_begin` | t=R+~80 ms; device ~2.3 ms/row | MEASURED |
| `avoidable_submit_delay` | **~7 ms mean / ~53 ms p95 per batch** (reservation + metadata resolve between earliest and actual storage submit) | MEASURED+DERIVED |
| `exposed_wait` | **fill 65.1 + enqueue 14.6 + sync 7.6 + unknown/dense ~14.3 ms/row**; of this, only the enqueue share (~14.6) and part of reservation (~7) are host-serial avoidable without changing routing | MEASURED |

## 3. Where submission is later than causally necessary (ranked)

1. **HIT experts wait behind MISS fills (head-of-line, tens of ms).**
   `prepare` (blocking `get_batch` join over ALL misses) runs before ANY
   `stage_expert` H2D submit (engine.cpp:880-886 order). A HIT expert's H2D
   could legally submit at `earliest_storage_submit`; it actually submits
   after the slowest MISS fill of the batch (~batch wall; MEASURED batch-wall
   mean 138 ms prod across the matrix, 65 ms/row decode-critical). At the
   sealed 46.7% hit rate roughly every second expert pays this. Hidden today
   (readiness ~0) — it is latency that matters only once fills shrink.
2. **Reservation + metadata resolve serial (~7 ms mean, 53 ms p95).**
   Dedup scan, LRU victim scan, `expert_store_->get` + `configure_fp4_quantized`
   per miss, vector resize/memset-or-reuse — all on the calling thread between
   earliest and actual submit. Small vs 65 ms but pure overhead, every batch.
3. **Sequential prepare→stage→compute staging (14.6 ms/row enqueue).**
   H2D submits serialize behind the fill join; pinned-gather memcpys
   (DERIVED 1–2 s whole-run) run on the calling thread. Structural, exact-safe
   to pipeline per-completion instead of per-batch-join.
4. **Cross-layer disk starvation (~23 ms/row of no legal work).**
   After a batch's ≤6 reads drain (~2 waves on 3 lanes, disk 96% busy
   WITHIN batch), the SSD has NO legal work until the next layer's router
   runs (attention + combine + sync + router + D2H in between). DERIVED gap
   ≈ wall − fill − enqueue ≈ 23 ms/row (~15 s over decode). Causally
   unfillable for score layers (lead-0 spine, §4); the ONLY exact exception is
   hash layers 0–2 (§4).

## 4. What is NOT later than necessary (do not "fix")

- **Route D2H (0.022 ms/row).** Copy floor; nothing to win.
- **Final layer sync (7.6 ms/row).** Uniform required drain; verified no legal
  host work exists for these windows (next routes unknown by R5). Event-handoff
  PROMOTEd mechanically but predicted ~0 by the required-completion analysis.
- **Cross-layer score routing (lead 0).** `router(L+1)` needs `h_{L+1}` needs
  `combine(L)`: the residual stream itself, not an implementation detail. No
  threading, reordering, or buffering changes this without changing the model.
  The exact-critical-path SIMULATED staging-lead≥8 (−26%) counterfactual
  assumes routes known 8 layers early and is therefore UNATTAINABLE in exact
  mode; it must not be cited as a plan.
- **Token t+1 staging.** `input_ids[t+1] = sample(logits(t))` — data dependency
  through the full 43-layer chain. No cut exists (speculative decoding out of
  scope for this track).
- **Hash-layer IDs (layers 0–2): the narrow exact exception.** IDs =
  `tid2eid[input_ids]`, EXACTLY_AVAILABLE at token start; SIMULATED journal
  count: 270 decode IDs = 229.5 MiB/token early-submittable (matches the
  route-pipeline 18-record/240.6 MB structural bound). WEIGHTS still need
  scores at that layer, so consumption stays chained; ABC mechanics measured
  batching hidden NEGATIVE. Value ≈ 0 wall; keep as a free side-effect of
  fixed slots, never a standalone project.

## 5. Same-step cross-layer prediction check on THIS journal (SIMULATED)

Mean shared experts per adjacent decode layer pair: **0.129** (81 shared
occurrences / 630 pairs) — reproduces the cache-predictor 0.13/6 datum on an
independent journal cut. The Fate-style same-token mechanism does not transfer
to DSV4; only cross-token recurrence carries signal (recall@12 = 0.503,
measured by the cache-predictor track).
