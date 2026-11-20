# Milestone 3 cache-policy research

Measured 2026-07-27 from Kaggle dual-T4 kernel version 17, run
`20260727T003223Z-4ec6d17d`, commit
`f8e6f79e43cbff419e86052d91358828e85d4d40`.

The source evidence is the immutable downloaded bundle at
`tmp/m3_v6_output/ornith-milestone3-evidence-20260727T003223Z-4ec6d17d`.
All four rows below passed correctness, all 40 layers, and the device-path
gate. H2D is the measured `expert_weight` component, not a prediction.

## Observed configurations

| Configuration | Retained experts/layer | Reserved cache/GPU | Prompt / generated tokens | Distinct reuse distance, median / p95 | Cache hit rate | Expert H2D | Evictions | Generation wall | Decode TPS | Peak VRAM/GPU |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cache disabled | 0 between forwards; 8-slot arena is cleared each forward | 960 MiB allocated | `Hello` / 2 | 7 / 9 | 0.000% | 96 MiB | 0 | 2.159 s | 2.1918 | 3,930.9 MiB |
| capacity 4 | 4 | 480 MiB | `Hello` / 2 | 7 / 9 | 1.5625% | 102 MiB | 1,260 | 2.191 s | 2.0436 | 3,450.9 MiB |
| capacity 8, profiled | 8 | 960 MiB | `Hello` / 4 | 7 / 17.8 | 21.25% | 252 MiB | 2,016 | 9.176 s | 0.5169 | 3,950.9 MiB |
| capacity 8, long prompt | 8 | 960 MiB | 10-token prompt / 2 | 23.5 / 40 | 2.6929% | 564 MiB | 3,758 | 242.087 s | 0.3795 | 3,946.9 MiB |

Evidence fields:

- `runs/*/run-report.json`: configuration, generation wall, decode TPS, and
  peak VRAM.
- `runs/*/expert-cache-analysis.json`: hit rate, reuse-distance distribution,
  and `input_summary.eviction_events`.
- `runs/*/transfer-analysis.json`:
  `by_component[component=expert_weight,direction=h2d]`.

The capacity-0 and capacity-4 rows are controlled: prompt, token count,
profiling, tracing, and warmup are identical. Capacity 8 uses the canonical
four-token profiled run, so its wall time and TPS must not be read as a
controlled speedup/regression against the two-token rows. The strict
performance verdict instead uses unprofiled `dual-warm-control`.

## Reuse-distance coverage

The request traces give a workload-only capacity curve without pretending
that unexecuted capacities have measured wall time. The values below are the
fraction of repeated requests whose prior distinct-expert reuse distance is
strictly smaller than the candidate capacity. They are opportunity coverage,
not predicted cache hit rates.

| Candidate experts/layer | Short prompt: repeated requests covered | Long prompt: repeated requests covered |
|---:|---:|---:|
| 4 | 2 / 405 (0.49%) | 0 / 196 (0.00%) |
| 8 | 252 / 405 (62.22%) | 6 / 196 (3.06%) |
| 16 | 369 / 405 (91.11%) | 46 / 196 (23.47%) |
| 32 | 405 / 405 (100.00%) | 151 / 196 (77.04%) |
| 48 | 405 / 405 (100.00%) | 192 / 196 (97.96%) |

The observed capacity-8 short run had 405 repeated requests, 179 repeated
misses, and 453 traced short post-eviction reuses. The long prompt had 196
repeated requests, 182 repeated misses, and 336 short post-eviction reuses.
This is direct evidence that the remaining misses are not all first-touch
loads.

## Attribution

Implementation defect:

- The cache uses one fixed arena per layer and chooses the lowest
  `last_used + priority * PRIORITY_WEIGHT` score
  (`include/dee/vram_cache.h:177-184`,
  `src/vram_cache.cpp:97-150`).
- A miss immediately evicts until one expert fits and then loads the new
  block (`src/vram_cache.cpp:154-249`). The trace shows hundreds of experts
  evicted shortly before reuse. Admission has no measured reuse-distance or
  frequency term, and the large priority multiplier can dominate recency.

Workload limitation:

- Short-prompt p95 distinct reuse distance is 17.8, already above capacity 8.
- Long-prompt median is 23.5 and p95 is 40, so an eight-expert arena cannot
  retain most recurring experts even under an ideal recency policy.

Policy headroom:

- Moving from disabled retention to capacity 4 produced only 1.56% hits and
  slightly more measured H2D; capacity 4 is below the short workload's median
  reuse distance.
- Capacity 8 reached 21.25% on the short canonical trace, but 179 repeated
  misses and 453 short post-eviction reuses remain.
- The trace-distance curve says capacity 16 covers 91.11% of short repeated
  requests, while capacity 32 covers 77.04% of long-prompt repeats. These are
  the next capacities worth measuring.

Hardware-capacity limit:

- Forty independent layer engines are split 20/20 across the T4s
  (`scripts/run_ornith_generation.py:737-776`). A BF16 expert is 6 MiB, so
  capacity 8 reserves 960 MiB/GPU, capacity 16 would reserve 1,920 MiB/GPU,
  and capacity 32 would reserve 3,840 MiB/GPU.
- The measured capacity-8 peak was about 3.95 GiB/GPU on 15 GiB T4s.
  Therefore 16 and 32 are plausible test points, but their projected memory
  use is arithmetic, not a measured success claim.

## Decision

Do not label the remaining residency cost fundamental and do not change the
default from this mixed-token matrix alone. The next isolated experiment
should hold prompt, generated-token count, profiling, and warmup constant at
capacities 8, 16, and 32. If capacity 16 materially raises hit rate and lowers
H2D without a wall/TPS regression, it is the conservative default candidate.
If capacity growth does not convert the measured reuse-distance opportunity,
the next implementation should add reuse/frequency-aware admission (for
example a probationary/protected segmented LRU) before increasing the arena
again.
