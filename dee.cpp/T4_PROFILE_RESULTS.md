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

## Oracle and host-staging campaign

The follow-up campaign began at `6519ea99a4b27cf14e857e332eefcd7092df41c3`.
Commit `35948c3390400e19daf99813897aa0f0b3b6118b` added opt-in Oracle-internal
timing without changing the primary benchmark path. Its normal-mode runs were
9.604, 9.527, and 9.689 tok/s (median 9.604), 1.13% below the preceding 9.714
median. Detailed timing identified the first Oracle matrix product as the
dominant Oracle cost:

| Oracle stage | Baseline ms | Share of 1346 ms Oracle total |
|---|---:|---:|
| Linear 0 | 1121.011 | 83.26% |
| Linear 1 | 101.928 | 7.57% |
| Linear 2 | 101.031 | 7.51% |
| Top-K sort | 15.505 | 1.15% |
| All other classified work | 6.526 | 0.48% |

Model lookup was only 0.079 ms total; tensor conversion and synchronization
were zero. The Oracle made 1,280 calls and 10,240 small output allocations
(5,283,840 bytes), so repeated model loading, map lookup, and conversion were
not the cause. A next-layer prediction cannot be started correctly before the
current layer's D2H result and normalization produce its input. The benchmark
also showed only 0.0137 adjacent top-K overlap versus a 0.0312 random
expectation. Consequently no speculative prediction queue was accepted, and
Oracle overlap on the critical path remains 0%.

### Accepted: runtime AVX2/FMA Oracle products

Commit `ac1351ca50ca814c7cce6fb675dde7a5ece5f7ba` adds a runtime-dispatched
AVX2/FMA dot-product implementation with a scalar fallback. It reuses the
already loaded tensor pointers and preserves the exact Oracle call sequence;
there is no synthetic-route cache or Oracle bypass.

Normal runs were 13.120, 13.185, and 12.821 tok/s (median 13.120), a 36.61%
gain over the instrumented 9.604 median and 35.06% over 9.714. CPU CTest passed
6/6, CUDA CTest passed 7/7, the independent Python top-8 comparison passed,
and output remained finite. Median detailed Oracle time fell from about
1346 ms to 455.115 ms: Linear 0 fell to 369.526 ms, Linear 1 to 31.774 ms,
Linear 2 to 31.172 ms, and Top-K sort measured 17.687 ms.

### Accepted: bounded persistent pinned BF16 sources

Commit `73a94f0d4823328768cdf943e338e246444ac09b` materializes each repeatedly
requested BF16 expert blob once in pinned host memory and submits subsequent
fine-grained H2D copies directly from it. The cache is capped at 192 MiB;
experts beyond the cap use the existing staging ring. The synthetic route used
230 physical experts and 180,879,360 bytes (172.5 MiB) of this host cache.
Device traffic, FP32 cache size, request ordering, and H2D call count are
unchanged.

Normal runs were 14.825, 14.617, and 14.778 tok/s (median 14.778), a 12.64%
gain over 13.120. This is 52.14% above the campaign's starting 9.714 median
and 186.90% above the original 5.151 tok/s result. CPU CTest passed 6/6, CUDA
CTest passed 7/7, output remained finite, and cache accounting remained 10,240
requests, 3 resident hits, 10,237 cold loads, and 10,233 evictions.

Median detailed results were:

| Stage | Before pinned cache ms | After ms |
|---|---:|---:|
| Oracle | 461.101 | 455.742 |
| Pageable/mmap to pinned | 943.408 | 32.316 |
| GPU H2D | 1399.249 | 1406.215 |
| Transfer submission | not isolated in control summary | 133.594 |
| Host waiting | not isolated in control summary | 1081.187 |
| BF16 expansion | about 120 | 114.995 |
| GPU expert compute | 506.700 | 677.246 |

Detailed runs were 13.733, 13.617, and 13.253 tok/s (median 13.617). The
opt-in profiler therefore costs 7.86% relative to the 14.778 normal median.
H2D remained 8,061,190,144 bytes in 11,517 copies: 699,938 bytes average,
786,432 bytes p95, and approximately 5.73 GB/s effective event-time bandwidth.
The profiler does not establish a common cross-stream origin, so an exact
transfer/compute overlap percentage is intentionally not claimed. As a
non-overlap proxy only, 1,081.187 ms of host waiting versus 1,406.215 ms summed
H2D duration leaves 23.11% of H2D duration not exposed as host waiting; this is
not equivalent to useful compute overlap.

The final controlled A-G run was:

| Scenario | Wall ms | tok/s |
|---|---:|---:|
| End-to-end | 2339.450 | 13.678 |
| Full resident | 917.372 | 34.882 |
| Resident bypass | 913.275 | 35.039 |
| Transfer only | 1975.593 | 16.198 |
| Compute only | 393.565 | 81.308 |
| Oracle only | 295.546 | 108.274 |
| Cache metadata only | 306.379 | 104.446 |

At the normal median wall time (about 2.165 s), the full-resident control gives
an empirical ceiling of roughly 34.9 tok/s, or 2.36x current throughput, if
streaming were removed without changing Oracle/compute. H2D is now the largest
measured device category (about 1.406 s summed event duration), while host
waiting is the largest CPU critical-path category (about 1.081 s). These are
the next measured targets; speculative GEMM tuning is not justified by this
profile.

### Rejected follow-up experiments

The following candidates passed correctness but were reverted and never
pushed:

| Experiment | Candidate median | Change vs 13.120 | Reason rejected |
|---|---:|---:|---|
| One coalesced BF16 transfer per four-expert batch (`61f66c2`) | 9.542 | -27.27% | Removed useful fine-grained copy/DMA overlap |
| Four-worker OpenMP Oracle rows (`31901fc`) | 8.092 | -38.32% | Contended with the pipeline on the 2-core/4-thread Studio CPU |

An earlier fused gate/up SGEMM (+0.18%) and event-pool/stream-wait path
(-0.96%) also remain rejected. The campaign therefore retained only changes
supported by measured bottlenecks and the 2% median gate.

## Cross-stream transfer campaign

The next campaign began from `07a239d08dc70d96f2335831e1d246426f640e1c`.
Its fresh FP32 normal-mode baseline was 14.703 tok/s (14.703, 14.469, 14.727).

Commit `6e74d40d6398adf721a17a91e26a29e8296831d7` added an opt-in,
common-origin CUDA timeline and Chrome trace. A representative FP32 run
measured a 2365.098 ms GPU span: copy active for 1404.933 ms (59.40%), compute
active for 786.513 ms (33.25%), copy/compute overlap for 574.627 ms (24.30%),
and neither active for 748.279 ms (31.64%). Cache-readiness waits accounted for
1099.724 ms; staging-slot waits were zero. This identified fine-grained H2D
readiness, rather than ring exhaustion, as the exposed critical path.

Commit `ca346c2ae19af0bea9338dcfd97b9222e6ab00e1` made ring depth configurable
and fixed cache-pin lifetime when a shallow ring completes DMA before compute.
Depths 2, 4, 8, and 16 produced medians of 14.534, 14.734, 14.499, and 14.724
tok/s. None improved performance by 2%; the default remains 64. The queue still
peaked at four because the 6 MiB FP32 cache, not the staging ring, bounded the
active batch. A persistent completion-event-pool experiment (`02cc8b7`) passed
all tests but measured 14.657 tok/s (-0.31%) and was rejected.

The existing pinned source cache already packs gate, up, and down into one
contiguous BF16 expert blob and submits one weight H2D copy per cold expert.
The 11,517 total copies are 10,237 expert copies plus 1,280 hidden-input copies,
so a second per-expert packing format was not pursued. The previously rejected
four-expert coarse transfer remains invalid because it destroys useful overlap.

### Accepted: FP16 device cache and direct mixed-precision compute

Commit `bce363501a9661f0560d1059244ce0ec10047a87` adds a configurable FP16
device cache. BF16 source weights convert directly to FP16 on the prefetch
stream; they are never expanded to FP32 before cache readiness. Gate, up, and
down projections use FP16 cuBLAS GEMM with FP32 accumulation, and the fused
SiLU/multiply kernel writes the FP16 down-projection activation. The exact FP32
cache/SGEMV path remains available with `--cache-dtype fp32`. The 6 MiB byte
budget is unchanged and now holds eight instead of four expert entries.

Final explicit-FP16 normal runs were 16.073, 14.909, and 16.161 tok/s (median
16.073), 9.32% above the fresh 14.703 baseline. Cache behavior improved to
10,240 requests, 72 resident hits, 10,168 cold loads, 10,160 evictions, and
zero prefetch fallbacks. CPU CTest passed 6/6 and CUDA CTest passed 7/7.

Numerical validation used both a deterministic CPU-vs-CUDA SwiGLU test and an
end-to-end trace comparison. A non-degenerate one-layer output had max absolute
error 1.04e-4, relative RMSE 6.62e-5, and cosine similarity 0.9999999978.
Across the primary 32-token workload, 10,237/10,240 ordered expert requests and
1,279/1,280 complete layer top-K sets matched FP32; mean top-K Jaccard was
0.999826. Output remained finite.

The detailed FP16 timeline measured a 2217.347 ms GPU span, 1396.546 ms copy
active (62.98%), 719.002 ms compute active (32.43%), 550.511 ms overlap
(24.83%), and 652.310 ms with neither engine active (29.42%). Average/max
transfer queue depth increased from 2.5/4 to 4.475/8. Cache-readiness waiting
fell from 1068.747 to 957.718 ms, while compute-batch synchronization fell from
15.753 to 7.444 ms. H2D traffic fell slightly to 8,006,926,336 bytes in 11,448
copies because 69 additional resident hits avoided transfers; weight-copy size
remains 786,432 bytes.

The updated profiled controls are:

| Scenario | tok/s | Key observation |
|---|---:|---|
| End-to-end | 14.414 | Detailed event/timeline mode |
| Full resident | 35.009 | Streaming removed |
| Resident bypass | 38.344 | Cache metadata bypassed |
| Transfer only | 16.285 | H2D/cache path remains close to end-to-end |
| Compute only | 79.612 | Compute is not the primary ceiling |
| Oracle only | 107.158 | Oracle no longer dominates |
| Cache metadata only | 105.259 | Metadata remains negligible |

The remaining largest bottleneck is still expert H2D/cache readiness. Against
the 35.009 tok/s full-resident control, the 16.073 normal median has about a
2.18x empirical streaming-removal ceiling. The exact timeline shows only
24.83% copy/compute overlap and 29.42% idle GPU time, leaving overlap scheduling
and transfer latency as better-supported next targets than GEMM tuning.
