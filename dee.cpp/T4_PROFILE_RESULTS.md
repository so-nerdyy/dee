# T4 profiling campaign

Environment: Lightning AI Studio, NVIDIA Tesla T4 (`sm_75`), CUDA runtime
13.0, driver 580.159.03. Workload for every regression comparison:

```bash
cd ~/dee/dee/dee.cpp && ./build/dee_cli \
  --shard tests/data/ornith_moe256.safetensors \
  --oracle oracle.pt \
  --tokens 32 --warmup 2 --topk 8 --layers 40 --cuda
```

## Profiling infrastructure

Commit `a480558109289c527bd714fd3cbf3aa217080952` adds bounded stage timing,
JSON request traces, and controlled A-G scenarios. CPU CTest passed 6/6 and
CUDA CTest passed 7/7. Normal-mode runs were 6.591, 6.560, and 6.575 tok/s
(median 6.575), so the opt-in profiler added no normal-mode regression versus
the prior 6.568 tok/s median.

The initial 6 MiB decomposition was:

| Scenario | Wall ms | tok/s | Interpretation |
|---|---:|---:|---|
| End-to-end | 5131.641 | 6.236 | Forced streaming |
| Full resident | 1754.292 | 18.241 | Same Oracle-driven requests, no weight transfers |
| Resident bypass | 1773.517 | 18.043 | Resident weights, cache lookup bypassed |
| Transfer only | 4455.514 | 7.182 | Oracle plus cache/staging/H2D; kernels disabled |
| Compute only | 406.333 | 78.753 | Deterministic experts; Oracle/transfers disabled |
| Oracle only | 1105.216 | 28.954 | Kernels/transfers/cache disabled |
| Cache metadata only | 1139.275 | 28.088 | Oracle plus cache simulation |

Cache metadata was negligible. Removing streaming reduced end-to-end wall time
by about 66%, while transfer-only remained 4.456 s. The representative detailed
profile measured 2.631 s H2D, 1.893 s pageable-to-pinned copy, 0.777 s host
waiting, 1.346 s Oracle, and 0.498 s GPU expert compute. CPU and GPU categories
overlap and therefore must not be summed as percentages of wall time.

## Accepted optimization: packed BF16 transfer

Commit `1ce6754f789111e2ee5f3d00aff2c09e3ec421a3` keeps expert weights in BF16
through pageable and bounded pinned staging, transfers BF16, then expands the
raw bit patterns to FP32 on the prefetch stream before recording completion.
The FP32 expert cache and cuBLAS layout are unchanged.

Normal regression runs were 9.714, 9.750, and 9.640 tok/s (median 9.714), a
47.74% improvement over 6.575 tok/s. CUDA CTest passed 7/7, including exact
BF16 expansion and CPU-vs-CUDA SwiGLU comparison; output remained finite.
Cache behavior was unchanged: 10,240 requests, 3 resident hits, 0 in-flight
hits, 10,237 cold loads, 10,233 evictions, and 0 prefetch fallbacks.

Median detailed timings across three optimized runs:

| Stage | Time ms | Wall share using 3485.049 ms median wall |
|---|---:|---:|
| Oracle | 1345.517 | 38.61% |
| Tensor resolution | 143.351 | 4.11% |
| Pageable-to-pinned copy | 938.519 | 26.93% |
| Transfer submission | 168.582 | 4.84% |
| Host waiting | 341.136 | 9.79% |
| Synchronization | 17.514 | 0.50% |
| GPU H2D | 1401.560 | 40.22% |
| GPU BF16 expansion | 119.681 | 3.43% |
| GPU expert compute | 506.334 | 14.53% |

Weight traffic fell from 16.10 GB to 8.05 GB pageable-to-pinned and from
16.11 GB to 8.06 GB H2D. H2D copy count remains 11,517 because this change
reduces bytes rather than coalescing requests. Detailed profile runs were
9.182, 9.263, and 8.930 tok/s (median 9.182); opt-in event timing therefore
costs about 5.48% relative to the optimized normal median.

The optimized controlled run still measures 3.470 s end-to-end, 1.811 s full
resident, 2.954 s transfer-only, and 1.130 s Oracle-only. H2D is now the largest
individual device stage (1.394 s in that run), closely followed by Oracle CPU
time (1.345 s). Streaming remains the largest removable aggregate cost under
the 6 MiB budget, but Oracle is now a co-primary bottleneck.

## Rejected experiment

Commit `80fe4d40e3af649fad8ecda18cfe54fdef2f2f57` changed pinned slots to
write-combined allocation. It passed CUDA CTest 7/7 and retained finite output,
but produced 6.304, 6.517, and 6.574 tok/s (median 6.517), 0.88% below the
6.575 baseline. It was reverted and was never pushed.

Earlier rejected experiments retained for context were fused gate/up SGEMM
(+0.18%), stream-wait/event pooling (-0.96%), and an unsafe source-layer cache
prototype (-0.23%). None met the 2% acceptance threshold.
