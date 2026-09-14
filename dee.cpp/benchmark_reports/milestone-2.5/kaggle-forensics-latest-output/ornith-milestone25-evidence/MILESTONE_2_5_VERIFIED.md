# Milestone 2.5 forensic profile

Result: **EVIDENCE_COMPLETE_PUBLICATION_PENDING**. Acceptance criteria not yet passing: [24].

## Measured conclusions

- Host memory: peak RSS was 20,472,827,904 B (19.067 GiB). At the detailed checkpoint, anonymous RSS was 2,382,303,232 B (2.219 GiB), checkpoint-backed RSS was 10,466,742,272 B (9.748 GiB), and exact persistent native host allocations totaled 252,313,600 B (0.235 GiB).
- cuda:0 peak NVML usage was 5,026,873,344 B (4.682 GiB): PyTorch peak allocated/reserved 3,502,453,760 B (3.262 GiB) / 3,539,992,576 B (3.297 GiB), exact native CUDA allocations 1,155,983,360 B (1.077 GiB), and inferred context/library/other residual 330,897,408 B (0.308 GiB).
- cuda:1 peak NVML usage was 5,026,873,344 B (4.682 GiB): PyTorch peak allocated/reserved 3,502,947,328 B (3.262 GiB) / 3,539,992,576 B (3.297 GiB), exact native CUDA allocations 1,155,983,360 B (1.077 GiB), and inferred context/library/other residual 330,897,408 B (0.308 GiB).
- Duplication: routed expert weights coexist as checkpoint mmap pages, persistent pageable/pinned native staging, a pinned transfer ring, and the CUDA cache arena.
- Warm decode: 0.240229 generated tokens/s; representative decode step 2 took 2249.124 ms.
- Expert traffic: 276 weight copies, 1,736,441,856 B (1.617 GiB) cumulative H2D, and 257,365 B (0.000 GiB) per measured miss.
- Cache behavior: 6,747 misses and 1,264 hits in the combined controlled trace. Reuse-distance and post-eviction records are in `expert-cache-analysis.json`.
- GPU overlap: useful layer intervals overlapped for 0.000 ms; both GPUs were NVML-active in 1 of 18 samples.
- Earlier ~30 TPS: 1.046735 was synthetic recurrent MoE steps/s, not full-model generated tokens/s.

## Measurement, inference, and hypothesis boundaries

Measurements are the byte counters, smaps/PSS/RSS values, CUDA-event durations, host-monotonic spans, NVML samples, generated token IDs, and trace events in this directory. The CUDA context/library bucket is an inference by reconciliation, and is labeled as such. Suggested Milestone 3 effects are hypotheses bounded by measured costs; no throughput promise is made.

## Milestone 3 repair order

1. sequential 20/20 two-GPU pipeline: Evaluate a placement/scheduling plan that creates useful concurrency without replicating unbounded state.
2. serialized native routed-expert path: Remove per-expert round trips and barriers; batch/retain outputs on device and establish transfer/compute overlap.
3. expert cache misses and repeated expert-weight H2D: Use measured reuse/working sets to design cache placement in Milestone 3; do not assume capacity alone is sufficient.
4. router and expert Python CPU/GPU bridge copies: Keep routing/expert activation and combine data on device.
5. dense attention and shared dense model work: Treat as the irreducible baseline before expert-path work.
6. persistent host expert staging duplication: Bound or remove persistent duplicate host representations.

## Evidence

The machine-readable report, memory ledgers, all per-layer rows, compressed expert trace, cache/transfer analyses, profiler summary, prior benchmark audit, and ranked bottlenecks are adjacent to this file.
