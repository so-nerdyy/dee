# Fixed-slot staging prototype — design, pre-registered verdict criteria, and evidence

`FIXED_SLOT_STAGING` was selected (Phase-2D) on DERIVED/SIMULATED grounds:
~7 ms/batch reservation + 14.6 ms/row enqueue + HIT head-of-line blocking.
This track attempts to **falsify or validate** that selection with a real
mechanism prototype. No production code touched; no merge; Luna's branch
untouched (her `host_expert_tier.h` vocabulary — PolicyResident/Dynamic —
was read-only referenced for the placement hint).

## 1. What the prototype is

`fixed_slot_staging.{h,cpp}`: exact-safe per-completion staging where
`reserve()` (O(1) slot-indexed, never waits for fills) replaces
`HostPackCache::get_batch`'s reserve-then-join, and completions publish
independently while consumption stays rank-ordered.

`bench_fixedslot.cpp`: drives BOTH paths with identical demand streams and
identical fill byte-patterns:
- **Baseline**: the REAL `dee::HostPackCache::get_batch` (compiled from
  `dee.cpp/src/host_pack_cache.cpp`, unmodified) + a stage loop mirroring
  `engine.cpp:880-886` (blocking prepare → sequential per-expert consume).
- **Prototype**: `fixedslot::FixedSlotStaging` + simulated async DMA thread.
- Fills write deterministic key-derived bytes (+ configurable sleep standing
  in for storage service); H2D is a simulated transfer thread with its own
  latency/counters — host-side mechanism and ordering are MEASURED here,
  wall-clock translation to T4 rows is DERIVED (stated, not hidden).

## 2. Pre-registered verdict criteria (written before the first run)

- **PROMOTE** iff ALL hold: (a) reserve latency ≥5× lower than `get_batch`
  reservation at production-like map occupancy with zero-fill cost
  eliminated (slot reuse, `zero_fill_bytes == 0`); (b) HIT H2D head-of-line
  eliminated (hit submits land before the slowest miss completes, MEASURED
  per-expert timestamps); (c) per-completion publish in finish order with
  rank-order combine bit-exact vs baseline across churn (memcmp + FNV);
  (d) lifetime rules hold under violation injection (consume-before-DMA
  fails closed; policy-resident never victimized); (e) revised wall prize
  still clears 2× run noise (≥ ~6 s of 66 s decode).
- **REJECT** iff: reserve win <2× (i.e. the sealed 7 ms was environmental,
  not structural), or prototype synchronization/copy overhead meets or
  exceeds the removed serial work, or any exactness/lifetime violation
  requires reintroducing a serial barrier.
- **HOLD** iff: mechanism wins are real but the honest wall translation
  falls below run noise (estimate overstated), or device-side unknowns
  (real H2D-submit syscall costs, CUDA ring pressure) dominate the residual
  and only Luna's live A/B can decide.

## 3. What the bench CANNOT prove (read before citing numbers)

- No GPU: real `cudaMemcpyAsync` submit costs, stream/event behavior, and
  ring pressure are simulated. The 9.4 s enqueue bucket contains real H2D
  submit work that fixed slots move earlier but do not delete.
- Synthetic record sizes (256 KiB–4 MiB) with DERIVED projection to the
  12.75 MiB production record via measured per-byte rates — the projection
  is explicit in `results_*.json` (`projection` block), never mixed into
  measured rows.
- Sleep-stand-in storage service (2–20 ms) models ORDERING and overlap, not
  the T4 /tmp device.

## 4. Files

- `fixed_slot_staging.h/.cpp` — prototype (research only).
- `bench_fixedslot.cpp` — baseline-vs-prototype harness, JSON evidence out.
- `build_bench.sh` — one-line g++ build (MSYS2 mingw, stdlib only).
- `results_bench_*.json` — sealed-by-commit evidence (reproduced by rerun).
- `FIXEDSLOT_VERDICT.md` — final PROMOTE/REJECT/HOLD with the revised prize.
