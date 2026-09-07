# ELIMINATED_MISS_LEDGER.md — exactly which misses cap 17→20 removed

Branch: `experiment/pack-cap-ab` · Date: 2026-09-07 · Base: `45b2a16`
Machine-readable: `results/eliminated-misses.json` (produced by
`tools/forensics.py all`, deterministic).

## 1. Construction

The route journal is **byte-identical across all four arms**
(`routed_experts.jsonl`, sha256 `f6ec70243acbafa7…`, chain-linked, 688
records). One instrumented dual-budget LRU replay over that journal
therefore reconstructs every arm's host-pack miss stream:

- budget A = 682 records/GPU (8.5 GiB), budget B = 803 records/GPU (10.0 GiB);
- production semantics validated in `research/exact-staging` (LRU, evict-on-fill).

For pure LRU on one stream, the larger cache's resident set is a superset of
the smaller's at every point, so B can have **no new misses** — the replay
confirms: `new_misses_in_B_replay = 0`.

## 2. Validation against live counters

| Quantity | Live | Replay | Diff |
|---|---|---|---|
| A total misses | 2481 (1390 cuda0 + 1091 cuda1) | 2480 (1388 + 1092) | −1 (+1/−2 per GPU) |
| B total misses | 2429 (1348 + 1081) | 2429 (1348 + 1081) | 0 |
| Eliminated | 52 | 51 | −1 |
| New misses in B | 0 | 0 | 0 |

The ±1 is the same per-GPU tolerance the validated replay showed against
sealed v65 (−2/+1); the eliminated set is reported at replay granularity
(51 identities) and the live delta (52) is consistent within that tolerance.

## 3. The eliminated misses (51 replay identities ≈ 52 measured)

- **All 51 are capacity re-reads, 0 compulsory.** Every eliminated miss is a
  record evicted from the 682-record pack and later re-demanded. The extra
  2 GiB/GPU recovers **no** compulsory traffic — it only lengthens the
  re-read horizon.
- Previous-use distance: **873–1900 distinct records** (median 1298) —
  long-recency re-reads, exactly the population the +121-record budget
  rescues.
- Bytes: **681.8 MB** (51 × 13,369,344 B) ≈ the ~695 MB measured SSD
  reduction (52 × 13.37 MB = 695.2 MB live).
- Device skew: **cuda:0 40, cuda:1 11**.
- Layers 0–38 (23 distinct layers, skewed to low layers on cuda:0).
- Token spread: tokens 3–15; **38 in tokens 1–11, 13 in tokens 12–15**
  (6,7,4,2,5,1,5,3,5 | 1,4,6,2). NOT concentrated in the tail tokens where
  the wall deltas live (see MISS_CRITICALITY.md / ORDER_EFFECT_FORENSICS.md).

Each ledger entry records: device, token (forward_step/start_pos), layer,
expert, decode request ordinal, storage bytes, `distinct_since_prev_use`,
compulsory flag, and the record A evicted to make room
(`evicted_to_make_room`). Per-request live telemetry (read submit/complete
timestamps) does not exist in the arms — see MISS_CRITICALITY.md for what
that bounds.

## 4. Live-vs-replay difference

One miss (≈ 2% of the delta) differs between the live count and the replay
set, within the previously documented replay tolerance. No systematic
discrepancy: hit/eviction counters match within the same tolerance.
