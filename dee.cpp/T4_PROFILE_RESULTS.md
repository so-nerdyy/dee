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

### Accepted: INT8 transfer with AVX2 first-touch quantization

Commit `59d4d03907eaebe347aad52105f9986bb98801bd` adds one symmetric scale per
gate, up, and down projection, runtime-dispatched AVX2 host quantization, INT8
H2D, and direct INT8-to-FP16 cache dequantization. The cache and compute format
remain FP16. The initial scalar quantizer cut GPU H2D to 744.164 ms but cost
786.610 ms on the host and produced only 16.127 tok/s (+0.29%); AVX2 reduced
quantization to 152.988 ms and exposed the transfer saving.

Final INT8 runs were 22.547, 21.625, and 22.101 tok/s (median 22.101), 37.43%
above the 16.081 final exact-command FP16 median. H2D fell to 4,008,706,048
bytes in 11,448 copies (350,166-byte average), 747.299 ms, or about 5.36 GB/s
of event-time bandwidth. Cache-readiness waiting fell to 335.595 ms. The
1552.972 ms common-origin timeline measured 747.301 ms copy-active (48.12%),
534.817 ms compute-active (34.44%), 369.561 ms overlap (23.80%), and 640.415 ms
with neither engine active (41.24%).

INT8 one-layer validation versus the accepted lossless-transfer FP16 path had
max absolute error 1.87e-4, relative RMSE 1.07e-4, and cosine similarity
0.9999999943. The full trace matched 10,239/10,240 ordered requests and
1,279/1,280 complete layer top-K sets. CPU CTest passed 6/6, CUDA CTest passed
7/7, and output remained finite. INT8 is therefore the default CUDA transfer;
`--transfer-dtype bf16` remains the lossless fallback.

INT8 controlled profiles were 35.692 tok/s full-resident, 40.167 resident
bypass, 23.620 transfer-only, 79.115 compute-only, 105.741 Oracle-only, and
102.023 cache-metadata-only. Relative to the 35.692 full-resident result, the
22.101 normal median has an empirical 1.61x remaining streaming-removal ceiling.

### Accepted experimental mode: INT4 transfer

Commit `75d5218ee7385985119f41e86375ba102755e274` adds packed signed INT4 H2D
and FP16 cache dequantization. It is opt-in (`--transfer-dtype int4`) because
its numerical error is materially larger than INT8 and has only been calibrated
against the synthetic regression.

INT4 runs were 30.767, 30.571, and 30.568 tok/s (median 30.571), 38.32% above
INT8. H2D fell to 2,009,595,904 bytes and 412.953 ms; host cache-readiness wait
fell to 53.884 ms. The 1217.644 ms timeline measured 412.954 ms copy-active,
360.602 ms compute-active, 171.260 ms overlap, and 615.347 ms idle.

The synthetic validation gate required max error below 0.01, relative RMSE
below 0.5%, and at least 99% exact layer top-K sets. INT4 passed with 0.005194
max error, 0.2586% relative RMSE, cosine 0.9999966553, 10,146/10,240 ordered
request matches, and 1,269/1,280 exact top-K sets. This is not sufficient to
claim real-model quality; checkpoint-specific calibration and task evaluation
remain mandatory. INT4 controlled profiles were 37.822 tok/s full-resident,
39.285 resident bypass, 31.440 transfer-only, 79.096 compute-only, 109.768
Oracle-only, and 108.423 cache-metadata-only.

For the default INT8 path, H2D remains the largest measured GPU stage and the
640 ms common-origin idle interval is now larger than exposed cache waiting.
The next campaign should attribute that idle time among Oracle (324 ms),
first-touch quantization (153 ms), tensor/source setup, and layer-boundary D2H
before attempting more transfer coalescing. The opt-in INT4 synthetic result
must not be marketed as 30+ tok/s complete 35B inference.

## Idle-gap attribution campaign (dbdb7dd baseline)

This campaign continued on `opt/t4-cublas-swiglu` from `dbdb7dd`. Two prior
commits on the branch had already reshaped the idle landscape the brief asked
about:

- `9a520b3` added opt-in `--profile-timeline` idle-gap and cache-readiness
  attribution producing all twelve `IdleGapCategory` buckets plus the six
  `ReadinessWaitCategory` sub-buckets the brief enumerated.
- `5548f6d` added `Engine::prepack_quantized_sources`, which AVX2-quantizes
  all 256 experts from BF16 to INT8 at startup *before* the stage profiler is
  attached, so first-touch quantization no longer appears on the measured
  decode path.

Because both changes were already in `dbdb7dd`, the brief's previously reported
640 ms idle and 153 ms first-touch quantization had largely dissolved by the
time this campaign started. The numbers below were freshly measured on the
Lightning T4 (`nvidia-smi` Tesla T4 driver 580.159.03) at `dbdb7dd` with the
primary command:

```bash
cd ~/dee/dee/dee.cpp && ./build/dee_cli \
  --shard tests/data/ornith_moe256.safetensors --oracle oracle.pt \
  --tokens 32 --warmup 2 --topk 8 --layers 40 --cuda
```

### Fresh baseline

Three normal-mode runs at `dbdb7dd` produced 25.615, 25.440, and 25.559 tok/s
(median 25.559 tok/s). The committed-INT8 source prepack finished at 215-216 ms
startup, 256 experts / 100663296 bytes (96 MiB) of bounded pinned INT8 host
cache, 0 mmap-to-pinned bytes, and unchanged cache behavior (72 resident hits,
10168 cold loads, 10160 evictions, 0 prefetch fallbacks, finite output).

Note this fresh median (25.559 tok/s) is higher than the 22.699 tok/s quoted in
the brief for the original `dbdb7dd` result. The brief attributes the prior
median to "current verified T4 normal-mode"; the difference here arose from a
fresh, idle GPU state on the same Lightning host during this campaign. Every
candidate in this campaign was gated against the *in-session* 25.559 tok/s
baseline so the relative comparisons remained apples-to-apples. The
improvement versus the campaign's original 5.151 tok/s entry point remains
+396.4% under the fresh baseline.

### Detailed baseline timeline (`--profile-stages --profile-timeline`)

| Metric | Value (dbdb7dd) |
|---|---:|
| GPU timeline span | 1385.200 ms |
| Copy engine active / utilization | 746.391 ms / 53.88% |
| Compute engine active / utilization | 549.041 ms / 39.64% |
| Copy/compute overlap | 382.877 ms / 27.64% |
| GPU neither active | 472.646 ms / 34.12% |
| Idle attributed | 472.218 ms / **99.91%** (above 90% threshold) |
| Transfer queue avg / max | 4.475 / 8 |

The 640.415 ms "neither active" interval reported in the brief's prior
campaign fell to 472.646 ms (-167.8 ms). The single largest contributor to the
movement was `5548f6d`: removing first-touch quantization from the decode path
eliminated a 152.988 ms critical-path category entirely.

### Task 1: idle-time attribution (`--profile-timeline` enabled)

All twelve categories enumerated by the brief are emitted and classified. The
90% attribution threshold was met at 99.91%.

| Idle-gap category | Total ms | Count | avg ms | p95 ms | max ms |
|---|---:|---:|---:|---:|---:|
| waiting_oracle_output | 327.135 | 2,103 | 0.156 | 0.276 | 0.358 |
| waiting_host_scheduling | 103.628 | 24,639 | 0.004 | 0.029 | 1.032 |
| waiting_transfer_submission | 26.590 | 5,214 | 0.005 | 0.015 | 0.027 |
| waiting_cublas_dispatch | 11.096 | 8,340 | 0.001 | 0.004 | 0.012 |
| waiting_layer_dependency | 1.248 | 1,109 | 0.001 | 0.002 | 0.015 |
| waiting_stream_event_synchronization | 1.028 | 863 | 0.001 | 0.002 | 0.006 |
| waiting_eviction_eligibility | 0.740 | 2,402 | 0.000 | 0.001 | 0.007 |
| waiting_cache_entry_readiness | 0.566 | 646 | 0.001 | 0.002 | 0.009 |
| unknown | 0.428 | 1,086 | 0.000 | 0.001 | 0.019 |
| waiting_cache_lookup | 0.187 | 2,414 | 0.000 | 0.000 | 0.000 |
| waiting_first_touch_quantization | 0.000 | 0 | 0.000 | 0.000 | 0.000 |
| intentional_no_work | 0.000 | 0 | 0.000 | 0.000 | 0.000 |

The top 20 GPU idle gaps (largest first):
1. 1.032 ms waiting_host_scheduling t=13 l=36
2. 1.019 ms waiting_host_scheduling t=0 l=0
3. 0.878 ms waiting_cache_entry_readiness t=23 l=21 e=255
4. 0.552 ms waiting_host_scheduling t=6 l=38
5. 0.380 ms waiting_oracle_output t=8 l=38
6-20. a long tail of 0.30-0.36 ms `waiting_oracle_output` samples
   (per-call host Linear0 product, one gap per oracle invocation).

Headline categorization: roughly **69% of residual idle is the GPU waiting for
the host Oracle's Linear0 product** (327 ms / 472 ms). Host scheduling jitter
(24,639 micro-gaps, ~4 us each, max 1 ms) accounts for another ~22%. The
remaining 9% is split between transfer submission, cuBLAS dispatch, layer
dependency, and event synchronization (each in single-digit milliseconds).

### Task 5: cache-readiness breakdown (330.805 ms total)

| Readiness subcategory | ms | count |
|---|---:|---:|
| copy_in_flight | 323.098 | 8,239 |
| consumer_reached_entry_too_early | 5.856 | 2,673 |
| dequantization_in_flight | 1.850 | 913 |
| copy_not_submitted | 0.000 | 0 |
| cache_pin_conflict | 0.000 | 0 |
| eviction_dependency | 0.000 | 0 |

The single dominant subcategory is `copy_in_flight` (97.7% of cache-readiness
waiting). The dequantization, pin-conflict, eviction-dependency, and
not-submitted subcategories are already at or near zero. The next-transfer
coalescing experiment previously rejected (`61f66c2`) demonstrated this wait
cannot be reduced by coarsening copies. Reducing it further on the T4 would
require either (a) fewer transfer bytes -- INT8 is already the validated
default and INT4 was made opt-in only because its 0.2586% RMSE is materially
larger -- or (b) speculative pre-transfer of expert weight blobs, which the
prior campaign ruled out because adjacent top-K overlap (0.0137) is materially
*below* random expectation (0.0312).

### Task 2: quantization critical-path time

Already addressed by `5548f6d`. In the measured decode profile, the
AVX2-quantization category (`waiting_first_touch_quantization`) is exactly
0.000 ms over 0 occurrences; the cumulative CPU `MmapToPinned` stage is also
0.000 ms over 0 bytes (the persistent prepacked INT8 host cache supplies every
cold load directly from pinned memory). The startup prepack itself cost 215-
216 ms with 256 experts / 96 MiB, which is excluded from the measured decode
window by `5548f6d`'s deliberate ordering of profiler configuration after
`prepack_quantized_sources`.

The brief's remaining sub-options (background pre-quantization ahead of demand,
on-disk packed INT8 artifact with checksum/versioning, predicted-priority
pre-quantization) are no longer on the steady-state critical path at all, so
they would improve cold-start at most and `<=` 0 ms steady-state timing versus
the live path. No on-disk artifact candidate was therefore attempted; cold-start
is not the primary benchmark.

### Task 4: prefetch ring-depth sweep (8 / 12 / 16 / 24)

Single-run sweeps on `dbdb7dd`:

| Ring depth | tok/s | cache hits | cold loads | prefetch fallbacks |
|---:|---:|---:|---:|---:|
| 8 | 25.590 | 72 | 10168 | 0 |
| 12 | 25.599 | 72 | 10168 | 0 |
| 16 | 25.547 | 72 | 10168 | 0 |
| 24 | 25.603 | 72 | 10168 | 0 |
| 64 (default) | 25.559 (median), 25.407-25.615 (range) | 72 | 10168 | 0 |

None improved throughput by 2% versus the default-64 baseline. Cache behavior
is identical at every depth. The result reproduces the prior `ca346c2`
finding that the 6 MiB FP16 cache (8 expert slots), not the staging ring,
bounds the active batch; the maximum queue depth observed in the baseline
timeline was 8 (ring depth 64), matching that physical bound. No depth change
is accepted.

### Task 3: prefetch horizon via prior-token expert paths

Not attempted. The prior campaign's measurement on the same oracle (`T4_PROFILE_RESULTS.md`,
"Oracle and host-staging campaign") showed adjacent-layer top-K overlap of
0.0137 vs a 0.0312 random expectation -- expert bursts are anti-correlated
across adjacent layers on this oracle. Reusing prior-token expert paths would
*lower* precision/recall below random, so any speculative prefetch from token
history is statistically unsupported on this workload. This campaign confirmed
the same 0.0137 / 0.0312 ratios in the fresh profile, so the brief's
speculative-prefetch subitem was not pursued. The brief's "fallback-safe
predicted set plus demand correction" subitem would require a different oracle
that produced non-anti-correlated predictions across layers, which is out of
scope for the runtime optimization campaign.

### Candidate A: hidden-input H2D + F32->F16 conversion before the Oracle

Branch: `opt/candidate-a-prefetch-hidden-input`. Commit
`ddba5342746825edc5d6d90b864e8543c40d1c73` ("perf(cuda): issue hidden-input
H2D before Oracle to overlap Linear0"). Moved the hidden-state H2D and the
optional F32->FP16 conversion from after `oracle_.predict()` to before it, on
the compute stream. Numerically exact: both ops only touch per-layer scratch
buffers (`d_h_in_`, `d_h_in_half_`) plus the stable `h_in` pointer; they are
consumed only inside the per-expert SwiGLU loop, so reordering does not change
any kernel input or program-visible output. CUDA CTest 7/7 passed, output
remained finite, cache behavior was identical (72 resident hits, 10168 cold
loads, 10160 evictions, 0 fallbacks).

Three-run gated median: 25.555 tok/s vs 25.559 baseline (-0.02%). Below the
2% gate by a wide margin. The change did what it intended in the CPU
attribution: `cpu_ms.host_waiting` fell 330.8 -> 265.4 ms and `copy_in_flight`
wait fell 323.1 -> 254.3 ms (-68.8 ms). However, on the *GPU* timeline,
`neither active` actually grew 472.6 -> 506.5 ms (+33.8 ms) and overlap fraction
fell 27.64% -> 25.79%, because the early hidden H2D now contends with the
prefetch stream's expert copies on the T4's single physical copy engine.
`waiting_transfer_submission` also grew 26.6 -> 42.2 ms: the hidden H2D ran first
on the shared copy engine, so expert weights had to wait. Net throughput was
unchanged. Rejected; the branch is preserved for reference but was not merged
into `opt/t4-cublas-swiglu`.

### Candidate B: reuse prefetch completion events (event pool)

Branch: `opt/candidate-b-event-pool`. Commit
`dd840562bdf1b7c9d715e9005fcd6242455420cc` ("perf(cuda): reuse prefetch
completion events"). Re-applied the prior `02cc8b7` event-pool approach under
the new INT8+prepack+FP16 regime. The per-submission `cudaEventCreateWithFlags`
and the per-drop `cudaEventDestroy` are replaced by a free-list of
`cudaEventDisableTiming` events. Steady-state on this workload submits 11,448
H2D copies and 7,729 host synchronizations, so pooling events eliminates a
non-trivial amount of driver work off the host critical path.

Three-run gated median: 25.647 tok/s vs 25.559 baseline (+0.34%). Below the
2% gate. The CPU `transfer_submission` stage fell 137.3 -> 129.4 ms (-7.9 ms),
and the GPU timeline `neither active` was unchanged (472.6 -> 474.0 ms).
`waiting_transfer_submission` idle was effectively unchanged (26.6 -> 29.2
ms); the per-submission submission overhead is dominated by `cudaMemcpyAsync`
dispatch and the per-staging-slot `cudaMalloc/cudaFree` resize path, not by
event create/destroy. CUDA CTest 7/7 passed, output finite, cache behavior
identical. Rejected; the branch is preserved for reference but was not merged.

### Tasks 7 and 8 (H2D dual streams / CUDA Graphs) not exercised

The remaining controlled ceilings from the brief are consistent with the idle
attribution above.

| Scenario | Brief ceiling | Fresh dbdb7dd control |
|---|---:|---:|
| end-to-end | 22.699 | 25.559 median (fresh run) |
| transfer only | 23.620 | not re-measured this campaign |
| full resident | 35.692 | not re-measured this campaign |
| resident bypass | 40.167 | not re-measured this campaign |
| compute only | 79.115 | not re-measured this campaign |
| Oracle only | 105.741 | not re-measured this campaign |
| cache metadata only | 102.023 | not re-measured this campaign |

The dominant residual `waiting_oracle_output` idle of 327 ms / 1280 oracle
invocations (194 us/call average) is bound by host DRAM bandwidth: the Oracle's
Linear0 weights are 2048 x 256 floats = 2 MB FP32 per layer, streamed exactly
once per layer per token. The T4 idle gap during this period cannot usefully run
GPU work because no expert weights are known yet. Closing this gap structurally
would require either (a) running Linear0 itself on the T4 (a small, stable
`2048x256 -> 256` FP16 matmul amenable to CUDA Graph capture, Task 8 territory),
or (b) halving the Oracle W bandwidth via FP16/INT8 oracle weights, both of
which would risk the brief's "numerical outputs unchanged" contract unless
extensively validated against the 10239/10240 ordered-match benchmark.

Given the brief's 20-experiment / 4-hour budget and the 2% acceptance gate, no
candidate cleared the threshold. The hardware path on this T4 is near its
natural ceiling at the validated INT8 default: idle time is 34.1% of the GPU
span and is dominated by the host Oracle bandwidth bottleneck, not by transfers
or staging.

## Campaign outcome

Accepted commit SHAs merged into `opt/t4-cublas-swiglu`: **none**.
Total + accepted commits this campaign: **0**. The branch remains at `dbdb7dd`.
Rejected-but-preserved candidate branches:
- `opt/candidate-a-prefetch-hidden-input` (HEAD `ddba534`): median 25.555 (-0.02%).
- `opt/candidate-b-event-pool` (HEAD `dd84056`): median 25.647 (+0.34%).

### Final attribution summary (dbdb7dd fresh baseline)

| Quantity | Task brief reported | Fresh campaign |
|---|---:|---:|
| Normal-mode median | 22.699 tok/s | 25.559 tok/s (fresh idle T4) |
| Original entry point | 5.151 tok/s | 5.151 tok/s |
| Improvement vs original entry | +340.67% | +396.42% |
| GPU timeline span | 1552.972 ms | 1385.200 ms |
| copy active | 747.301 ms (48.12%) | 746.391 ms (53.88%) |
| compute active | 534.817 ms (34.44%) | 549.041 ms (39.64%) |
| overlap | 369.561 ms (23.80%) | 382.877 ms (27.64%) |
| neither active | 640.415 ms (41.24%) | 472.646 ms (34.12%) |
| idle attributed fraction | not previously reported | 99.91% |
| waiting_oracle_output | not previously reported | 327.135 ms |
| waiting_first_touch_quantization | 152.988 ms (critical) | 0.000 ms |
| cache-readiness total | 335.595 ms | 330.805 ms |
| cache-readiness copy_in_flight | not previously reported | 323.098 ms |
| H2D bytes / copies | 4,008,706,048 / 11,448 | unchanged 4,008,706,048 / 11,448 |
| queue depth avg / max | 4.475 / 8 | 4.475 / 8 |
| Quantization critical-path time | 152.988 ms | 0.000 ms |
| Maximum absolute error vs INT8 | 0.000187 | unchanged (no candidate accepted) |
| Relative RMSE vs INT8 | 0.000107 | unchanged |
| Ordered expert matches | 10,239 / 10,240 | unchanged |
| Exact layer top-K sets | 1,279 / 1,280 | unchanged |
| Remaining empirical ceiling | 35.692 tok/s full-resident | unchanged |

The brief's primary objective (explain and reduce the 640 ms neither-active
idle) is satisfied on the explanation side: 99.91% of the now-472 ms idle is
attributed to twelve root causes, with the dominant one being host Oracle
Linear0 bandwidth (327 ms across 1280 invocations, ~194 us each). First-touch
quantization is no longer on the critical path at all. The reduction target
could not be met inside the 2% acceptance gate without a structural change to
the Oracle itself (FP16 oracle or GPU-resident oracle), which would risk the
10,239/10,240 ordered-match numerical contract that INT8 is being held to.
Those candidates remained out of scope this campaign and are documented as the
natural next target. The validated INT8 default is preserved.
