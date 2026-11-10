### Addendum 4 — Phase-6 results, 1xL4 slice (DRAFT, checked 2026-11-10, NOT yet appended)

Status: every number below was re-read from the r5 evidence (volume `dee-p6-evidence`,
`runs/p6m-l4-r5-*`) and the scorecards generated from it by `aws/phase6/score.py`
(committed under `dee.cpp/runs/<run_id>/scorecard.{md,json}`). It is not appended to
`PREDICTIONS_PHASE6.md` until the open decision (L4 as own reference / 2xT4 / pause) is made,
because that choice changes what the b=1 rows are called.

**Runs.** `p6m-l4-r5` at 311d9d0bf821: one 1xL4 container (96 GiB), four rows, torch 2.10.0+cu128.
Container wall to the last row: 2,450 s (b8-h16 `wall_s`). Billed cost: $1.58 as reported from
the Modal dashboard (not in the evidence; the scorecards carry only the estimates $0.107 / $0.128 /
$0.307 / $0.544, which are GPU-only). `p6m-t4-r1` at cc31cc3: 1xT4, OOM at load (~16.2 GiB needed,
14.57 GiB available), no tokens, $0.28 (reported). Spend to date ~$3.31 of $30.
Measured B_SSD on this L4: 3.955 GiB/s (fio randread, QD6, O_DIRECT) = 1.70x the cell spec
(2.5 GB/s), so every frozen-row decode tok/s and $/1k is UNTESTABLE (gate 0.9-1.1x).

**b=1 correctness (measured).** Gate `REJECT_NUMERICAL`: `exact_16_token_ids` and
`exact_decoded_text` False; every other non-hardware gate True. The L4 output differs from the
sealed T4 answer (`benchmark_reports/.../v57-evidence-20260831T044713Z`) from **token 0**
(T4 666 / L4 95140): L4 reads "Alan Turing was a brilliant British ...", T4 "**Alan Turing
(1912-1954)** was an English ...". First route divergence: **step 0 (prefill), layer 6**, 688 of 688
records compared; layers 0-5 identical. At layer 6 two of the 7 prompt rows differ: row 0's 6th
expert is 190 (L4) vs 216 (T4); row 3 selects the same six experts but ranks 4-6 come in a different
order (L4 99,76,236 vs T4 76,236,99). The "near-tie" is NOT independently verified: the captured
`router_top8_values/ids` for that layer do not contain every selected expert (L4 selected 241 for row 0;
it is not in the captured top-8 ids), so the raw top-8 scores do not show the margin that decided
the selection. The earlier report called it a near-tie; the evidence in hand supports only "a rank
flip at layer 6".
The L4 is self-consistent: b1-h64 and b1-h16 give identical token ids, identical routed-expert
journals and identical route-weight hashes (run_id and chain hashes aside).
*Hypothesis, not a result:* a different GPU's float rounding flips a close selection margin. Not separated
from a one-GPU-mode bug, because the 1xT4 check could not load. Torch build was reported ruled out (same
2.10.0+cu128 as the sealed build; not re-checked here).

| row | metric | measured | prediction | grade | evidence |
|---|---|---|---|---|---|
| L4 b=1 h64 | decode tok/s | 0.522 (15 decode tokens) | not scored | UNTESTABLE (correctness gate) | runs/p6m-l4-r5-b1-h64 |
| L4 b=1 h16 | decode tok/s | 0.654 (15 decode tokens) | not scored | UNTESTABLE (correctness gate) | runs/p6m-l4-r5-b1-h16 |
| L4 b=8 h64 | decode tok/s, aggregate | 0.838 | Add.3 nominal 9.006, kill <5.310 | KILLED (Add.3); frozen UNTESTABLE (B_SSD gate) | runs/p6m-l4-r5-b8-h64 |
| L4 b=8 h64 | $/1k | 0.644 | Add.3 nominal 0.0600, kill >0.1189 | KILLED (Add.3); frozen UNTESTABLE | same |
| L4 b=8 h64 | host hit | 0.835 (41307/49463) | 0.54 [0.47, 1.0] | PASS | same |
| L4 b=8 h64 | cold rec/tok | 31.86 | [0, 295.5] | PASS | same |
| L4 b=8 h16 | host hit | **0.000 (0/49463)** | 0.51 [0.44, 0.75], kill <0.45 | KILLED (frozen and Add.3) | runs/p6m-l4-r5-b8-h16 |
| L4 b=8 h16 | decode tok/s, aggregate | 0.591 | Add.3 nominal 4.074, kill <2.250 | KILLED (Add.3); frozen UNTESTABLE | same |
| L4 b=8 h16 | cold rec/tok | 193.2 | Add.3 [60.3, 370.0] | PASS | same |

b=8 steady-state gate: OPEN on both rows. The two b=1 decode rates are below 1 tok/s, but are
ungraded.

**What the kills say (measured, then hypothesis).**
- b=8 at 16 GiB, host hit 0. Measured: the 16 GiB host tier holds 1,285 experts (budget
  17,179,869,184 B / 13,369,344 B; `host_pack.entries` = 1,285). Decode steps request 915-1,644
  distinct (layer, expert) pairs, mean 1,521; 62 of 64 decode steps exceed 1,285. Prefill requests
  6,147. Over the whole row `host_pack` logged 617 hits and 102,857 misses. *Hypothesis:* LRU
  evicts each expert before its next use when the per-step set exceeds capacity (the earlier
  "up to 2,064" was the 43 x 48 upper bound, not what was observed). The model's host-hit term assumes
  the per-step working set fits. Theory fix goes on `research/phase7-theory`.
- b=8 at 64 GiB, 0.838 tok/s. Measured (cuda0 stage profile, whole row of 640 s): `mmap_to_pinned`
  CPU time 262 s, `everything_else` 367 s, store read worker time 187 s, GPU `h2d` time **5.2 s**,
  GPU compute ~1.0 s, layer wall 587 s. Engine read rate 1,522 MiB/s per read (fio on the same
  disk: 3.96 GiB/s). Decode step time rose from 4.4 s (first) to 11.2 s (last) with a constant
  per-step request count. The frozen/Add.3 predicted limiter was `h2d`; the profile does not
  show h2d as the limiter. *Hypothesis:* a host-side copy/`mmap_to_pinned` path, not the SSD and not
  PCIe. Unverified. The bucket times overlap, so the sum exceeds wall. P16/P17 are read by hand.
- b=1 at 0.52-0.65 tok/s: below 1 tok/s; ungraded until the token question is settled.

**Harness changes found by these runs** (merged): PR #15 (gate names in scorecards, warm-cache
warning for rows after the first in a container, router dump, verdict fix), PR #16 (VRAM-fit
check in preflight and bench; L40S default cache lowered).

**Corrections to the earlier draft:** hit rate at b8-h64 is 0.835 (PASS), not omitted; per-step
need is max 1,644, not 2,064; engine read rate is 1,522 MiB/s, not ~1.3 GiB/s; the divergence is
at prefill step 0 and starts at token 0; b1-h16 decode is 0.654 tok/s.
