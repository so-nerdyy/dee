# Rider / replay / generation comparison method

Three measurements of the same 13,369,344-byte records, each answering a
different question. Tool: `dee.cpp/tools/fill_replay` (host-only,
`DEE_BUILD_FILL_REPLAY=ON`); runner: fill-measurement kernel session.

## A. Isolated rider (capability)

Sequential vs random record reads, lanes 1–8, pass 1 (cold-ish) vs pass 2
(warm), direct `materialize` (no cache). Answers: raw path capability per
pattern and QD scaling. Reference points: independent rider reported cold
~2.9 GB/s with saturation knee at qdepth 2–4. Re-run here to confirm on the
same host/state as B/C (numbers drift with page-cache and SSD conditions).

## B. Journal replay (production pattern, isolated service)

Route-journal order (`layer expert [token]` lines) through the REAL
`Dee4ExpertStore::materialize` + REAL `HostPackCache::get_batch` with
production lanes=3 / qdepth=6, plus (1,6) and (3,1) ablations; cold-ish
pass 1 vs warm pass 2; mincore deltas per pass. Emits fill_timeline.json
(request windows with starts/durations) + summary (throughputs, p50/p95,
short reads, mincore fractions).

Distinguishes:
- QD starvation: lanes1 ≈ lanes3 throughput ⇒ starvation/containment;
  linear scaling ⇒ independent requests.
- Cold vs overhead: pass1 ≫ pass2 ⇒ page-cache faults dominate;
  pass1 ≈ pass2-but-slow ⇒ per-call overhead (chunking/mutex/memset).
- Pattern: journal vs monotonic vs random ⇒ fragmentation sensitivity.

## C. Actual generation (reference)

The profiled decode itself (66.233 s wall, 42.0 s fill spans). B must
reproduce C's per-request service distribution for the replay to be a
valid stand-in; divergence localizes the missing factor (GPU-memory
pressure on page cache, torch allocator activity, Python orchestration).

## Label discipline

Rider/replay bytes are real bank bytes but the CONTEXT (cache warmth,
concurrent load) differs from generation; label every number with its
mode. Never convert replay throughput into tok/s.
