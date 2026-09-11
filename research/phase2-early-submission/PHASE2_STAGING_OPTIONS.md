# PHASE2_STAGING_OPTIONS.md — fixed slots, component split, shared overlap, prefetch guardrails

All options exact-safe (routing, execution, outputs bit-identical) or rejected.
No implementation in this track. Provenance labels as in PHASE2_CAUSAL_TIMELINE.md.

## Option 1 — FIXED_SLOT_STAGING (recommended for Phase-2D)

**Shape (Edge0-style, adapted to dee's seams, staged_replace semantics EXCLUDED):**
- At `route_known(L)`: for each of the ≤6 demanded experts, reserve a fixed
  destination tuple `{pinned host slot, VRAM cache block, transfer event}`
  indexed by `(layer, expert)` — no LRU scan on the demand path, no
  malloc/memset in the fill path (double-buffered reuse).
- Submit the fill immediately into the reserved slot (misses), or submit H2D
  immediately from the resident slot (hits) — per-completion, not
  per-batch-join. Hits never wait behind misses.
- Meanwhile the calling thread executes already-ready work: remaining H2D
  submits, combine preparation (`weights_f32` conversion, pointer tables),
  shared-expert launch (still exact-joined, see Option 3).
- Consume each arrival when its event fires, with the existing exact compute
  (sequential per-expert SwiGLU, bounded scratch decode) and the untouched
  rank-order `addcmul_` combine.

**What disappears:** reservation scans (~4.6 s), memset (~0.8 s),
gather-memcpy (1–2 s), repeated region re-pointing per call, HIT head-of-line
blocking, repeated graph/assembly per batch. **What stays:** the H2D bytes
themselves, the fill bytes themselves (Flash's domain), the required output
sync, and every correctness gate (cold==warm, token identity, byte-exact
materialization).

**Explicit non-goals (do NOT smuggle in):** Edge0 `staged_replace` eviction
semantics (dee keeps its own LRU/priority policy; slots are a submission
mechanism, not a replacement policy); any predictor input; any compute
reorder; any sync removal (event-handoff lifecycle proof still required
separately and is NOT assumed).

**Acceptance for Luna's build (suggested):** matched A/B on dual-T4 decode,
ACCEPT_CORRECTNESS (identical IDs/text) + stage-enqueue + reservation spans
down with no fill-byte change; abort if first-pair delta < run noise
(3–5 s). Expected ~5–9 s (SIMULATED/DERIVED, not promised).

## Option 2 — COMPONENT_DEADLINE_SPLIT (studied, do NOT build for 2D)

Two-deadline SwiGLU: transfer+execute gate/up first (8.5 MiB), down arrives
while gate/up executes (4.25 MiB), exact accumulation/order preserved by the
existing rank loop. Prize THEORETICAL 0.5 s (H2D overlap) to 2.5 s (SSD
overlap requiring split preads); expected net ~0–0.5 s after seek regression
on a pread-bound device; readiness≈0 and 1.5 s compute-of-66 s confirm neither
deadline is exposed. Costs: splits the ONE contiguous record read Phase-1
proved optimal; adds a second deadline choreography through the fill path
Phase-1 closed. **Gate for revisit:** expert compute on the critical path
AFTER fills shrink (profiler evidence, no fixed threshold).

## Option 3 — EXACT_SHARED_WORK_OVERLAP (studied, shelf for post-fill era)

Shared needs only `h_L`; run it across routed staging with a buffered exact
join (`routed + shared` in documented order; rank loop untouched). Prize bound
MEASURED 0.3 s decode (+UNKNOWN host part); contention measured severe
(c=0.021). The value sits in the staging region (zero-SM overlap ≈ free),
never under GEMM (bandwidth split). Keep the prototype; do not productize for
2D: prize ≤0.6 s cannot clear run noise against lifecycle risk.

## Option 4 — TRAINED_PREFETCH_RESEARCH (guardrails, no build)

If/when researched, hard contract: prefetch path submits through a
LOW-PRIORITY lane only (never head-of-line ahead of demand fills); residency
for speculative bytes capped so demand-hot residents are never displaced
(prediction-aware hysteresis victims with dwell floor, as the cache-predictor
sim tested); every speculative byte accounted `issued = useful + wasted +
live` with zero residue; promotion requires ≥60% useful-byte rate on a sealed
journal PLUS a cross-token router-ahead scheduler design. Until then, no
capacity, no bandwidth, no branch.

## What was deliberately NOT turned into an option

- More lanes / qdepth (rider sweep flat 0.33–0.38 GiB/s lanes 1–8; QD6
  already 96% busy; QD1 idles — the ceiling is the device).
- GDS / read-path surgery (pread ≈ 100% of service, 0 short reads).
- Compression (NO_CODEC_WORTH_BUILDING: best real zstd-10 saves ~3.5 s of
  42 s at most, ~5% of decode, for a second on-disk format + CPU).
- Quantization change, router-semantic change, sync removal without the
  per-consumer reader/eviction/error proof.
