### Addendum 4 — Phase-6 results, 1xL4 slice (DRAFT, not committed)

Status: draft built from the Nov 6 r5 report and the Nov 8 T4 attempt. Every number marked
[check] must be replaced with the value in `runs/p6m-l4-r5-<tag>/scorecard.json` before commit,
and each row needs its evidence path. The scorecards are on the `dee-p6-evidence` volume and the
laptop checkout, not in the repo yet.

**Runs.** `p6m-l4-r5` at 311d9d0: one 1xL4 container, four rows (b1-h64, b1-h16, b8-h64, b8-h16),
torch 2.10.0+cu128 / Python 3.12 (sealed build). Billed $1.58 [check]. `p6m-t4-r1` at cc31cc3:
1xT4, OOM at model load (needs ~16.2 GiB before the expert cache, T4 has 14.57 GiB), no tokens,
$0.28. Campaign spend to date ~$3.31 of $30.

**b=1 correctness.** The 16 tokens differ from the sealed T4 answer. The first 6 layers match;
layer 6 picks a different 6th-ranked expert (near-tie). Torch is ruled out (sealed build used).
The L4 is self-consistent: b1-h64 and b1-h16 match bit-for-bit. Remaining suspects: GPU
architecture/driver rounding, or one-GPU mode. Not separated: the 1xT4 check could not load.
Decision pending: [treat L4 as its own reference | 2xT4 check | pause].

| row | metric | measured | Addendum-3 prediction | grade | evidence |
|---|---|---|---|---|---|
| L4 b=1 h64 | decode tok/s | 0.52 [check] | see predecl_modal at measured B_SSD | UNGRADED (token gate) | runs/p6m-l4-r5-b1-h64 |
| L4 b=1 h16 | decode tok/s | [check] | [check] | UNGRADED (token gate) | runs/p6m-l4-r5-b1-h16 |
| L4 b=8 h64 | decode tok/s (aggregate) | 0.84 [check] | ~6x higher [check exact] | KILLED | runs/p6m-l4-r5-b8-h64 |
| L4 b=8 h16 | host hit % | 0 [check] | 51 [check] | KILLED | runs/p6m-l4-r5-b8-h16 |

**What the kills say.**
- b=8 at 16 GiB: the host tier holds about 1,285 experts, but one b=8 step asks for up to 2,064
  distinct experts, so LRU evicts before reuse and hit rate collapses to 0. The model's host-hit
  term assumes the per-step working set fits; at b=8 it doesn't. Theory fix goes on
  `research/phase7-theory`.
- b=8 at 64 GiB: aggregate throughput about 6x below prediction. Cause not yet isolated
  (storage at ~1.3 GiB/s engine vs 7.5 fio in r4 is the main suspect). [check r5 stage profile]
- b=1 at 0.52 tok/s is below P1's kill line (0.9) but stays ungraded until the token question is
  settled.

**Harness changes found by these runs** (all merged): PR #15 (gate names in scorecards,
warm-cache warning for rows after the first in a container, router dump of both score sets,
verdict fix), PR #16 (VRAM-fit check in preflight and bench; L40S default cache lowered).
