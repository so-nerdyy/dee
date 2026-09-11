# FIXEDSLOT_VERDICT.md — prototype evidence, revised prize, final verdict

Track: `research/phase2-fixedslot-proto` (this branch). Prototype + bench are
research-only; the REAL `dee::HostPackCache::get_batch` was compiled
unmodified from `dee.cpp/src/host_pack_cache.cpp` as the baseline.
No GPU, no production change, no merge.

## 1. What was built and run

- `fixed_slot_staging.{h,cpp}`: O(1) slot-indexed reserve (never waits for
  fills), lane-pool miss submit, per-completion publish with finish-order log,
  rank-order consume, host-slot-busy-until-DMA + device-block-until-consume
  lifetimes (generational DMA matching), policy-resident protection, LRU-tail
  victim choice, fail-closed on every violation path.
- `bench_fixedslot.cpp`: identical demand streams + identical key-derived
  fill bytes through BOTH paths; byte-compare + FNV rank-order combine check.
- 8 evidence runs (`results_M*.json`): entry sweeps 64/128/256/512/1024,
  record sweeps 256 KiB/4 MiB/**12.75 MiB production size**, scenarios
  cold/mixed/churn(1.5×)/lifetime(2× + policy + violation probe)/stagger.
  All: `exact_combine_match: true`, violation probe fails closed, policy
  residents intact under 2× churn (1613 victim probes, 0 policy takes).

## 2. Measured results (host mechanism; no GPU)

| Metric (measured batches) | Baseline (real get_batch) | Prototype | Ratio |
|---|---|---|---|
| Reserve/batch, 64–1024 entries, 256 KiB–4 MiB | 7–12 µs | 25–34 µs | ~0.3–0.4× (PARITY, proto slower) |
| Reserve/batch, **12.75 MiB cold** (M5) | 11 µs (same-size reuse, 36/36) | 12 µs | parity |
| First fill invoke | 23–55 µs | 47–76 µs | parity (both immediate post-reserve) |
| HIT H2D submit (mixed) | 10–21 ms (waits full join) | 0.02 ms | **500–800× eliminated** |
| First H2D submit/batch | 12–20 ms | 0.006–0.6 ms | 20–2000× earlier |
| Per-completion publish (stagger) | n/a (join) | 17/20 batches out-of-order published, rank-order consumed exact | proven |
| Zero-fill bytes | fresh-resize path exists | 0 (slots amortized: 64–1024 allocs, rest reuses) | eliminated |
| Victim probes/take | ~1 (LRU tail) | 1 (LRU tail; 25 under 2× policy pressure) | parity |
| Combine exactness (incl. churn) | — | bit-exact vs baseline | holds |

## 3. Falsification outcome: the 5–9 s estimate was OVERSTATED — retracted

The prior 5–9 s rested on two components this prototype falsifies as
decode-wall prizes:

1. **Reservation (~4.6 s of the claim): RETRACTED.** Sealed mean 7.08 ms/batch
   does not reproduce as steady-state reservation at ANY geometry (µs at
   64–1024 entries, incl. production 12.75 MiB records with evictions).
   Decomposition: same-size victim reuse makes steady-state reservation
   scan + bookkeeping only. The sealed mean is dominated by prefill
   cold-start fresh 12.75 MB allocs (up to 42-wide prefill batches × ~1 ms
   memset/page-fault each) plus unmodeled prepare-side metadata resolve and
   live allocator pressure — none of which a decode steady-state reserve
   rework removes. Decode reservation prize ≈ 645 × ~0.02 ms ≈ **0.01 s**.
2. **HIT head-of-line (remainder): real but HIDDEN.** 500–800× submission
   improvement MEASURED, yet readiness≈0 (sealed) proves H2D already always
   completes before consume — converting 0 ms of today's wall. It becomes
   wall only after fills shrink (post-hierarchy/fast bank), where serial
   H2D submission would otherwise surface.

## 4. Revised prize (what survives)

| Component | Size | Status |
|---|---|---|
| Steady-state reservation removal | ~0.01 s | MEASURED (dead) |
| Cold-start fresh-alloc amortization | ~0.7–1.8 s whole-run one-time (1364 fresh × 0.5–1.3 ms @ 10–28 GB/s MEASURED memset) | MEASURED rate, DERIVED total; prefill-side |
| Calling-thread gather-memcpy elimination (fill direct into pinned slot; 52 GB decode @ ~15 GB/s) | DERIVED **~2–4 s** | **Unproven**: needs 17 GB pinned footprint + real `cudaHostAlloc` behavior + real H2D-submit costs — all device-side, all unmodeled here |
| Repeated assembly/metadata resolve | small, unmeasured | open |
| **Revised total** | **~2–5 s plausible, ~1 s firm** | below 2× run noise as a firm claim |

Note: production already reuses same-size victim buffers (`reused_fill_buffers`
path) — the remaining delta is pinned-direct + per-completion + amortized
allocs, not "reuse vs no-reuse". The external `buffer_reuse −12.3%` sim datum
is not reproduced by this bench and should be treated as an upper-bound model
output, not a measurement.

## 5. Criterion scoring (pre-registered in PROTOTYPE.md)

- (a) reserve ≥5× lower: **FAIL** (parity; the ms-scale mean was prefill/env).
- (b) head-of-line eliminated: **PASS** (mechanism), zero wall today.
- (c) per-completion + rank-order exact: **PASS** (incl. churn, stagger).
- (d) lifetimes + policy protection: **PASS** (probe fails closed; 2× churn).
- (e) revised prize ≥ ~6 s: **FAIL** (~1 s firm, ~2–5 s plausible).

## 6. Final verdict

**FIXED_SLOT_STAGING_HOLD**

The original 5–9 s justification is rejected as stated. The mechanism itself
is validated (exact, lifetimes sound, head-of-line structurally removed) but
its firm wall prize (~1 s) does not clear run noise, and its largest
remaining component (gather-memcpy ~2–4 s) turns on pinned-memory and H2D-path
behavior this track cannot measure. HOLD for exactly one scoped live A/B:

**Gate for Luna (narrow, abortable):** pinned-slot-direct staging A/B on
dual-T4 decode measuring the 9.4 s enqueue split (gather-memcpy vs submit vs
cache-ops) + pinned-alloc cost + ACCEPT_CORRECTNESS. PROMOTE iff enqueue
drops ≥6 s with no pinned-alloc regression; REJECT iff <3 s or any
correctness/lifetime defect. No predictor, no reorder, no sync removal in
that A/B. Until then: no Phase-2D implementation commitment; Flash's
class-1 (fewer fills) work remains the binding lever.
