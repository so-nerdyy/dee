# Next-use eviction and prefetch hints, scored on committed route journals

2026-11-17. No GPU was used for anything in this file. Predictions for the hardware run are in
`PREDICTIONS_PHASE6.md` Addendum 5.

> **Correction, 2026-11-18 (after run `p6m-l4-r6`).** The hint rows in the timed table below are
> wrong. The replay kept a finished hint read available forever, so a demand many steps later
> still counted as a hint hit after the record had left the host tier. It also admitted every
> hint, with no reservation cap and no rank test. Both are fixed: `simulate_timed` now follows
> `HostPackCache::submit_hints` and `absorb_hints`, and `hint_rank` chooses how a hinted record
> is ranked. With the fix, previous-step hints give host hit 0.522 at 16 GiB, not 0.869, which
> matches r6's measured 0.503. The eviction tables (no hints) are unaffected; r6 confirmed them.
> The corrected hint results are in `PREDICTIONS_PHASE6.md` Addendum 6.

## The problem

Phase 6 run `p6m-l4-r5` measured a host-tier hit rate of 0/49,463 at b=8 with a 16 GiB host tier
(Addendum 4). The tier holds 1,285 records. One decode step asks for about 1,500 distinct
(layer, expert) records, and it sweeps the 43 layers in the same order every step. LRU evicts the
record used longest ago, which is exactly the one whose layer comes up next, so every record
leaves just before it is needed again.

## The three changes

All three only decide where a record sits and when it is read. The experts that run are always
the ones the real router selected, so tokens and route journals cannot change.

1. **Next-use eviction** (`include/dee/next_use.h`, both tiers). Evict the record whose expected
   next use is furthest away: `E = d + N * (1/p - 1)`, where `d` is the number of layer calls
   until the sweep reaches the record's layer again, `N` is the number of layers, and `p` is a
   running estimate of how often the record is used when its layer runs. Records of the layer
   running now leave last.
2. **Prefetch hints** (`HostPackCache::submit_hints`). Background lanes read a record into host
   RAM before the model asks for it. Two hint sources: `prev_step` (the experts the next layer
   used on the previous decode step) and `router` (the next layer's own router evaluated on the
   current layer's input; exact table lookups for the hash-routed layers).
3. **Deadline-ordered hint queue.** Hints for the nearest layer are read first, hints whose layer
   has started are dropped, and a demand read never waits behind a hint that has not started.

## The simulator

`theory/policy_sim.py` replays a route journal through a two-tier cache model that follows the
engine's lookup order, sub-batching and pinning. A b=8 cohort is built by merging the eight
sealed single-prompt journals in
`benchmark_reports/deepseek-v4-flash-0731-t4/gpu2-phase4-cache-hierarchy/evidence/a2_fp4/`.
`belady` evicts by true next use with full knowledge of the future: no online policy can beat it.

Reproduce every table below:

```
J=benchmark_reports/deepseek-v4-flash-0731-t4/gpu2-phase4-cache-hierarchy/evidence/a2_fp4
python theory/theory/policy_sim.py $J/routed_experts-q{0..7}.jsonl --host-gib 16 24 32 64 --timed
python theory/theory/policy_sim.py $J/routed_experts-q0.jsonl --max-step 15 --window 1 15   # b=1
python -m pytest -q tests/test_theory_policy_sim.py
```

### It reproduces what r5 measured

| | simulator (LRU) | r5 measured |
|---|---|---|
| b=8, host 16 GiB, host hit, last 32 steps | 0.000 | 0.000 (0/49,463) |
| b=8, host 64 GiB, host hit, last 32 steps | 0.824 | 0.835 (41,307/49,463) |
| distinct records per decode step | 1,585 | 1,521 |

The journals are the sealed T4 routes. The L4 routes differ from layer 6 on (Addendum 4), which
is why the step size is 4% off.

### Eviction policy, b=8, last 32 of 64 decode steps, VRAM 281 records

| host tier | policy | host hit | VRAM hit | source reads per step |
|---|---|---|---|---|
| 16 GiB (1,285 records) | lru | 0.000 | 0.000 | 1,585 |
| | layer_distance | 0.320 | 0.064 | 1,078 |
| | **next_use** | **0.418** | **0.121** | **923** |
| | belady (bound) | 0.607 | 0.156 | 623 |
| 24 GiB (1,927) | lru | 0.460 | 0.000 | 855 |
| | **next_use** | **0.529** | **0.121** | **747** |
| | belady (bound) | 0.716 | 0.156 | 451 |
| 32 GiB (2,570) | lru | 0.578 | 0.000 | 668 |
| | **next_use** | **0.614** | **0.121** | **611** |
| | belady (bound) | 0.786 | 0.156 | 339 |
| 64 GiB (5,140) | lru | 0.824 | 0.000 | 280 |
| | **next_use** | **0.833** | **0.121** | **264** |
| | belady (bound) | 0.931 | 0.156 | 109 |

- Next-use removes the 0% failure: 42% of the lookups hit at 16 GiB and source reads fall 42%.
- It is never worse than LRU at any size tried, and the gain shrinks as the tier grows, as it
  should: with room for three steps, LRU already keeps most of what recurs.
- `layer_distance` (next-use without the reuse estimate) is worse than LRU above 16 GiB. The
  reuse estimate is what makes the policy safe to leave on.
- About 0.19 of host hit separates next-use from the offline bound at 16 GiB. That gap is what
  better prediction of which experts recur could still buy.
- The VRAM tier (281 records, far below one step) goes from 0 to 12% hits.
- The estimator's constants matter little: alpha 0.5 / p0 0.25 gives 0.423 at 16 GiB against
  0.418 for the shipped alpha 0.2 / p0 0.1.
- b=1 (sealed prompt 0, 15 decode steps, host 16 GiB): host hit 0.653 with LRU, 0.674 with
  next-use; VRAM hit 0.381 and 0.383. No regression where LRU already works.

### Hints and queue order (timed replay)

The timed replay adds a clock: 4 read lanes at 12.3 ms per record, 5 ms to stage a record that is
not in VRAM, 20 ms per layer for everything else. Those constants were fitted so the LRU row gives
r5's 0.59 tok/s. Read the tok/s column as a model, not a measurement.

| host tier | policy | hints | queue | tok/s (model) | host hit | reads per step, hints included |
|---|---|---|---|---|---|---|
| 16 GiB | lru | off | | 0.58 | 0.000 | 1,576 |
| | next_use | off | | 0.74 | 0.418 | 944 |
| | next_use | prev_step | deadline | 0.92 | 0.869 | 946 |
| | next_use | prev_step | fifo | 0.93 | 0.874 | 949 |
| 24 GiB | lru | off | | 0.69 | 0.460 | 884 |
| | next_use | off | | 0.77 | 0.529 | 776 |
| | next_use | prev_step | deadline | 0.88 | 0.785 | 791 |
| 32 GiB | lru | off | | 0.72 | 0.580 | 703 |
| | next_use | off | | 0.81 | 0.614 | 644 |
| | next_use | prev_step | deadline | 0.86 | 0.728 | 654 |
| 64 GiB | lru | off | | 0.81 | 0.824 | 338 |
| | next_use | off | | 0.91 | 0.833 | 324 |
| | next_use | prev_step | deadline | 0.91 | 0.834 | 323 |

- **Previous-step hints do not add reads, they move them.** Reads per step stay within 1% while
  the model stops waiting for most of them: host hit at 16 GiB goes from 0.42 to 0.87.
- **The deadline queue makes no measurable difference here.** Hints that look one layer ahead
  already arrive in deadline order, so FIFO serves them the same way. Deeper lookahead (3 or 6
  layers) did not help either. The queue's real job in the engine is the guarantee that demand
  never waits behind a hint, which this replay cannot fail. It is kept, and `fifo` stays available
  for an A/B, but no hardware row is spent on it.
- **Router-ahead hints cannot be scored offline.** The journals hold routes, not hidden states.
  A synthetic sweep (hint drawn from the true next set at a chosen recall and precision) gives,
  at 16 GiB: recall 0.5 / precision 0.5, 0.90 tok/s; recall 0.7 / 0.5, 0.94; recall 0.9 / 0.5,
  0.98; a perfect hint, 1.02. So a router hint has to reach about 0.7 recall to match
  previous-step hints. The hardware run measures its recall and precision.
- At 64 GiB neither change matters much for tok/s. r5 showed that cell is limited on the host
  copy path (`mmap_to_pinned`), not by source reads.

## What this does not show

- No tok/s number here is measured. The counts (hits, reads) come from an exact replay of real
  routes. The times come from a three-constant model.
- The engine caps the bytes reserved for hints at a quarter of the host budget, and a hint may
  only displace a record expected later than itself. The simulator has neither limit.
- Hint lanes compete with the pinned-copy path for CPU. The simulator does not model CPU.
- The routes are from the T4 seal, not from the L4.

## Prior work

- **SeqMoE** (arXiv 2609.12978): next-use ("probabilistic Belady") eviction and a
  deadline-ordered prefetch queue for sequential expert offload. Changes 1 and 3 follow it.
- **Pre-gated MoE** (Hwang et al., ISCA 2024) and **ProMoE**: selecting or predicting the next
  layer's experts one layer early. Change 2's `router` source uses that idea as a hint only. DEE
  never executes a predicted route.
- **Belady (1966)**: the optimal offline replacement policy, used here as the bound.
- Earlier DEE results that shaped this: `PREDICTOR_VERDICT.md` (set predictors are weak,
  popularity is the best single signal, which is why `p` is a per-record reuse estimate) and
  `CORRECTION_HOST_HIT_BATCH.md` (the capacity cliff at one step set).
