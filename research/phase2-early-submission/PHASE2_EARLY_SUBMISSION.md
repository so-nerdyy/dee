# PHASE2_EARLY_SUBMISSION.md — how much expert work can be submitted earlier (exact routing unchanged)

Independent causal-research verdict on class-2 (EARLIER LEGAL SUBMISSION).
Flash owns class-1 (fewer fills). No implementation, no T4 A/B, no production
touch. Provenance labels as in PHASE2_CAUSAL_TIMELINE.md.

## 0. Bottom line

- Within-layer submission is already immediate in structure (one blocking
  `get_batch` over ≤6 misses at `route_known`); the residual within-layer
  avoidable delay is host-serial: **~7 ms mean / ~53 ms p95 reservation per
  batch + 14.6 ms/row stage-enqueue**, all folded into the FIXED_SLOT_STAGING
  mechanism below.
- Cross-layer early submission of score-layer experts is **causally
  impossible** (lead-0 residual spine); the only exact lead is hash layers
  0–2 (229.5 MiB/token, ~0 wall value).
- No impossible lookahead was invented at any point; every schedule below
  consumes only official router outputs.

## A. Immediate demand submission — audit (source-inspected + MEASURED)

After the native router result exists (`h_expert_ids` in hand at native entry):

| Suspect | Finding | Size |
|---|---|---|
| Batch construction (ID validation, 256-expert group build, buffer check, `BatchConstruction` span) | real, tiny | DERIVED ~0.1 ms |
| Per-miss metadata resolve (`expert_store_->get` + `configure_fp4_quantized`, calling-thread serial, before `get_batch`) | real, serial | inside reservation span |
| `get_batch` reservation (dedup scan + LRU victim scan + vector resize/memset-or-reuse) | real, serial, the whole `avoidable_submit_delay` | MEASURED 7.08 ms mean / 52.9 ms p95 per batch → DERIVED ~4.6 s over decode |
| Fill join (calling thread + 2 workers share `next_fill_index_`, `fill_done_cv_` wait) | required for MISS data, but HEAD-OF-LINE for HIT H2Ds | hits (~46.7%) wait ~batch wall, DERIVED tens of ms |
| `stage_expert` sequential host loop (consume callback + pinned gather + H2D submit per expert) | real, serial | MEASURED 9.436 s decode (14.6 ms/row) |
| Staging-ring serialization (`AsyncPrefetcher` bounded ring, 64 slots) | RULED OUT at this shape (≤6/batch7595; ring-full path never taken) | structural |
| Stream ownership (fill workers → prefetch stream → compute stream, per-transfer events, `wait_on_stream`) | sound; device already async | readiness MEASURED ~0 |
| `fill_mutex_` / epoch atomics | RULED OUT (lock-free fill path by prior audit; epoch/pending only) | prior audit |
| Route-D2H full-stream sync | floor-bounded, 0.022 ms/row | MEASURED — do not touch |

**Quantified avoidable delay: ~7 ms mean / ~53 ms p95 per batch before first
byte is requested, plus HIT-expert H2D head-of-line blocking (~batch wall).
Neither is the wall; both are the fixed-slot prize (B).**

## B. Fixed staged slots (Edge0-style, exact-safe) — STUDY, recommended

Design studied (not implemented): at router-result time, reserve a fixed
destination slot per demanded expert (double-buffered host pinned slot +
VRAM cache block); submit the fill immediately into the reserved slot;
meanwhile execute already-ready work (HIT H2Ds, combine prep, shared path);
consume each arrival with the existing exact compute + rank-order combine
when its event fires. The prediction layer is NOT required: reservations are
keyed strictly on official IDs; a slot reserved but never filled (cannot
happen — reservations derive from demand) needs no invalidation path beyond
the existing fail-closed `get_batch` unit semantics.

What it removes (all host-serial, all MEASURED/DERIVED buckets):
1. LRU victim scans + dedup scans per batch (→ slot-indexed reserve, O(1));
2. `memset` zero-fill on fresh allocs (DERIVED ~0.8 s; reuse already avoids
   ~half — slots make it total);
3. H2D-gather memcpy on the calling thread (DERIVED 1–2 s → single copy
   straight into the reserved pinned slot);
4. Repeated graph/assembly work per batch (pointer tables / region
   re-pointing `point_fp4_regions` per call → built once per slot fill);
5. HIT H2D head-of-line blocking (per-completion submit instead of
   per-batch-join; hits submit at `earliest_storage_submit`).

Predicted exposed-wall reduction: **DERIVED upper ~14 s (21%) by bucket sum;
expected ~5–9 s (8–13%)**, anchored on the independent exact-critical-path
SIMULATED `buffer_reuse −12.3% on-demand (~−8.8 s)` datum, which attacks the
same host-serial surface. Note the documented interaction: buffer-reuse gains
shrink toward ~0 AFTER staging-lead lands — but staging-lead≥8 is causally
unavailable (§4 of timeline), so no subsumption occurs in any attainable
schedule; fixed slots ARE the attainable form of that prize. Residual risk:
implementation complexity in eviction/error paths (the same lifecycle proof
the event-handoff track lists, not done here).

## C. Component deadline splitting (DualDeadline-style SwiGLU) — STUDY, rejected

Record geometry (MEASURED layout): gate 4.19 + up 4.19 + down 4.19 MiB weights
+ 3×0.25 MiB scales = 12.75 MiB contiguous, no padding. Split candidate:
first deadline gate/up (+scales) = **8.5 MiB (66.7%)**, second deadline down
(+scale) = **4.25 MiB (33.3%)**. Exactness preserved in principle: gate/up
GEMMs + activation run first with identical numerics; down GEMM + rank-order
accumulation unchanged; only arrival order changes.

Why rejected (measured, not aesthetic):
- H2D-overlap prize THEORETICAL **0.52 s (0.8% of decode)**: down H2D is
  0.80 ms/row @5.54 GB/s; gate/up H2D+compute ≈ 4 ms/row; readiness≈0 proves
  H2D is never exposed, and device compute is 1.5 s of 66 s.
- SSD-overlap variant THEORETICAL **2.54 s (3.8%)** requires splitting the ONE
  contiguous pread into two positional reads per record — same bytes plus an
  extra seek per miss on a device where pread is already ~100% of service
  with zero short reads and flat pattern/concurrency throughput. Expected net
  after seek regression: **~0–0.5 s**, for a choreography that touches the
  fill path Phase-1 closed as needs-nothing.
- Revisit ONLY if a future profile shows expert compute (not staging) on the
  critical path AFTER fills shrink — the same C1/C2 gate the route-pipeline
  track pre-registered. Not Phase-2D.

## D. Shared expert / other independent work — STUDY, rejected for 2D

Shared math is exact-independent: `shared(h_L)` needs only the layer input,
available pre-router; current placement after routed combine is code order,
not dependence (source-grounded in `layer_candidate.py:488-492` vs
`adapter.py:169`). Overlap window is structural: entire routed staging +
decode + compute per layer. Prize bound: **MEASURED shared device 0.308 s
decode (0.47%)**; shared HOST wall is UNKNOWN (live instrumentation gap —
see Luna list item 5). Contention measured severe: GEMM-under-H2D 3.56 ms vs
0.075 ms alone (c=0.021). Expected net **≤0.3–0.6 s with lifecycle churn and
bandwidth-splitting risk**. Not a Phase-2D mechanism; keep the exact-join
prototype on the shelf for the post-fill era. No other exact independent work
exists in the window (route-D2H 0.02%, combine 0.2%, orchestration is dense
attention, syncs are required drains — all MEASURED).

## E. Trained prerouter as PREFETCH ONLY — STUDY, research track (not 2D)

Exact-mode contract respected throughout: prediction may ONLY submit
low-priority speculative transfers; native router authoritative; wrong
predictions waste bandwidth / pollute cache but never change execution,
routing, or output. No predictor built or trained here.

Required accuracy to beat wasted /tmp bandwidth (THEORETICAL disk-busy
parity + MEASURED predictor numbers from the cache-predictor track):
- The disk is 96% busy WITHIN batch: every wasted byte displaces useful
  bytes. Parity needs **useful-byte rate ≥ 50%**; with displacement margin
  **≥ 60%**.
- Best MEASURED causal predictor (`persist_layer`, strictly past-step
  features): recall@12 = 0.503 but **useful-byte rate = 25.1%**
  (74.9% wasted); full-bundle recall 4.7% @12.
- In-situ prefetch SIMULATED/MEASURED outcome: **neutral-to-harmful at every
  capacity** (cap-8 −0.4% misses; cap-16 +1.8% harmful; cap-32 harmful) —
  the 43-layer cross-token lead exists but the in-situ scheduler has zero
  within-step lead (within-step reuse ≈ 0: 0.129 experts/pair, §5 timeline)
  and hysteresis cannot distinguish soon-useful prefetches from pollution.
- **Gap: +35 pp (≈2.4×) accuracy improvement needed**, PLUS a cross-token
  router-ahead scheduler redesign (running next-token routers while current
  fetches fly — a runtime design change, out of scope for Phase-2D).

Verdict: TRAINED_PREFETCH_RESEARCH stays a research track. Do not schedule
implementation against Phase-2D. Revisit when (a) a causal predictor
demonstrates ≥60% useful-byte rate on sealed journals AND (b) the Phase-2
host tier has headroom for speculative residency without displacing demand.

## Decision (exactly one)

**BEST NEXT MECHANISM: FIXED_SLOT_STAGING** — predicted exposed-wall
reduction ~5–9 s of 66 s decode (8–13%; SIMULATED anchor −12.3%, DERIVED
upper 21%). Basis: it is the only candidate whose prize (i) is on the
measured critical path (host-serial staging, 9.4 s + 4.6 s + 0.8–2.8 s
buckets), (ii) requires no prediction, no reorder, no semantic change, and
(iii) survives the causality audit (all other "earlier" schedules are either
impossible or ~0). Alternatives rejected on measured grounds: C (~0–0.5 s),
D (≤0.6 s + contention), E (needs 2.4× accuracy + redesign),
NO_EARLY_SUBMISSION_WIN_YET (falsified: host-serial prize ≫ 3–5 s run noise).
