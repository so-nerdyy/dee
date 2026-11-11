# Correction: host-tier hit rate under batched decode (2026-11-10)

**What was killed.** `p6m-l4-r5-b8-h16` measured a host hit rate of 0 out of 49,463 lookups. The
frozen row and the Addendum-3 row both predicted 0.51 (`PREDICTIONS_PHASE6.md` Addendum 4). The
registered term depends on host capacity only, and batch size never enters it.

**Why the term was wrong.** The term rests on the Independent Reference Model behind Che's
approximation, and under that model batching only rescales time. The engine's stream isn't like
that: each decode step sweeps the layers in order and looks up its de-duplicated (layer, expert)
set once. A record can only be reused at a later step, and between the two lookups LRU sees
roughly every record of the steps in between. If one step's set is bigger than the host tier,
every record gets evicted before its next use. That is the loop-larger-than-cache case, and the
hit rate drops to about 0 however skewed popularity is.

**The fix.** `theory/theory/batch_cliff.py` adds a step-granular LRU model:

- `u_i(b) = 1 - (1 - pi_i)^b` is the chance that record i is in one step's set.
- `U_k(b)` is the expected number of distinct records over k steps.
- `K` is the largest step gap with `U_k <= C`.
- The steady-state hit is `H_step = sum u_i (1 - (1 - u_i)^K) / sum u_i`.

`K = 0` exactly when `U_1(b) > C`, and that is the cliff. The corrected nominal is
`min(registered prior, H_step)`. It is a cap, so it only binds near or below the cliff.

**Check against r5** (`python -m theory.batch_cliff`, which writes `data/batch_cliff_summary.json`
and `data/batch_cliff_grid.csv`):

| row | capacity (records) | one step's set U_1 | registered | corrected | measured |
|---|---|---|---|---|---|
| b=1, 64 GiB | 5,140 | 319 | 0.54 | 0.54 | 0.5745 |
| b=1, 16 GiB | 1,285 | 319 | 0.51 | 0.51 | 0.5578 |
| b=8, 64 GiB | 5,140 | 1,317 | 0.54 | 0.54 | 0.8351 |
| b=8, 16 GiB | 1,285 | 1,317 | 0.51 | **0.00** | **0.00** |

The three rows that passed keep their registered value, and the killed row now matches.

**Where the cliff sits.** b=1: 4.0 GiB. b=2: 6.8. b=4: 10.9. b=8: 16.4 in the model. b=16: 22.4.
These come from the sealed popularity.

**Limits, stated before any new run.**
- This is a correction made after seeing the result. It explains the kill, but it is not a
  prediction until it is registered in a new addendum before the next run.
- The popularity comes from the sealed b=1 trace: one prompt, with 2,364 of 11,776 records ever
  seen. It has no tail, so it undercounts one step's set at b=8. The model gives 1,317, while r5
  measured a mean of 1,521 (915 to 1,644). Using r5's measured sets, the b=8 cliff is about
  19 GiB, not 16.4. The edge is real, but its position is uncertain by about 15%. At b=16 the
  model saturates at the 2,364 records it knows about.
- `H_step` is steady-state and ignores compulsory misses in short windows. That's why it is only
  used as a cap.
- The VRAM tier in front of the host tier (3.5 GiB on L4) is ignored. It shifts the cliff by at
  most about 270 records.

**Test this cheaply next.** A b=8 row at 24 GiB should land near the registered 0.52 nominal,
because it is above both cliff estimates. A row at 16 GiB should read about 0 again. A row at
18 GiB sits between the two estimates, so it would show which one is right. Any of these is a
billed Modal run and needs Nivin's yes first.

**Update 2026-11-11.** The 16.4 vs 19 GiB gap is closed in `theory/batch_cliff_replay.py`. The r5
cohort is the Phase-4 eight-prompt workload, whose full route journals are in the repo. Rebuilt
from them, the model's b=8 cliff moves to 18.6 GiB, and an exact LRU replay of the rebuilt stream
reproduces both r5 b=8 rows (0.000 at 16 GiB, 0.823 vs 0.835 at 64 GiB) and puts the first hit at
19.1 GiB. The replay also shows a plateau at 0.46 from 21 to 28 GiB that the independent-draw model
puts at 0.30. Registered as Addendum 5 for run `p6m-l4-r6` (b=8 at 18, 24 and 32 GiB).
