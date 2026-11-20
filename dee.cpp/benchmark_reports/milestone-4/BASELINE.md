# Milestone 4 transfer-accounting baseline

## Preserved source of truth

- Branch: `opt/real-model-t1`
- HEAD and pushed origin: `eef727c1e5155b1d56be8d416a18b7a2a31ea70c`
- V17 run: `20260727T003223Z-4ec6d17d`
- Immutable local evidence:
  `tmp/m3_v6_output/ornith-milestone3-evidence-20260727T003223Z-4ec6d17d`
- Live preservation check: 137 files, 209,305,144 bytes; root
  `artifact-manifest.json` SHA-256
  `6CC8632459418746C223A8BBD9A679DF02D747AEC7CF2F3545F589B2CCD2C1C2`.
- The pre-M4 worktree contained only preserved untracked forensic/generated
  artifacts. No M3 artifact was modified, moved, or deleted.

## Exact accepted workload

M3 used two otherwise matched rows; M4 must not conflate their boundaries:

| Measurement | Run | Exact workload | Value |
|---|---|---|---:|
| Decode + TTFT | `dual-warm-control` | `Hello` / token 9419, four generated tokens, warmup, cap 8, 20/20 dual GPU, no profiler | 2.813886761 TPS; 609.278467 ms |
| Layer-hook wall sum | `dual-warm-profiled` | Same workload plus request/timeline/Torch profiling | 403.608263 ms |
| True profiled step model span | `dual-warm-profiled` | Same representative decode step | 418.113847 ms |

The matrix declares the profiled and control rows at
`scripts/run_milestone25_matrix.py:164-172`. The timing analyzer deliberately
retains both the end-to-end model span and layer sum at
`scripts/analyze_milestone25_matrix.py:511-557,608-613`. M3's accepted
performance gate consumes the layer sum as `model_wall_ms_total` at
`scripts/analyze_milestone3_matrix.py:669-674,1128-1141`; this historical gate
is preserved, while M4 reports both measurements by their actual meaning.

## Runtime policy

| Policy | Exact baseline | Source |
|---|---|---|
| Expert size | `3 * 2048 * 512 * 2 = 6,291,456` bytes (6 MiB) | `scripts/run_ornith_generation.py:737-739`; shard-derived cache blob at `src/engine.cpp:1746-1761` |
| Arena topology | One independent Engine and arena per layer; 40 engines | `scripts/run_ornith_generation.py:749-776`; `src/engine.cpp:1814-1848` |
| Arena capacity | 8 full experts = 50,331,648 bytes per layer | `scripts/run_ornith_generation.py:738-764` |
| GPU placement | Layers 0-19 on CUDA 0; 20-39 on CUDA 1 | `scripts/ornith_support.py:207-210`; `scripts/run_ornith_generation.py:749-750` |
| Resident ceiling | 8/layer; 160/GPU; 320 total | Capacity and 20 layers/GPU above; independently present in v17 `run-report.json` |
| Routing | Top-k 8 from the model configuration | `scripts/run_ornith_generation.py:756-759` |
| Cache/transfer precision | FP16 resident cache; BF16 source transfer | `scripts/run_ornith_generation.py:764-768` |
| Transfer granularity | Whole 6 MiB expert (gate/up/down), then BF16-to-FP16 conversion | `src/engine.cpp:1538-1553`; `src/async_prefetcher.cpp:511-565` |
| Prefetch lookahead/ring | One staging slot per layer Engine | `scripts/run_ornith_generation.py:764-768`; `src/engine.cpp:1699-1701` |
| Duplicate request handling | Engine-local in-flight lookup reuses the resident/in-flight request | `src/async_prefetcher.cpp:168-254` |
| Eviction | Lowest `last_used + priority * 2^20`; pinned blocks excluded | `include/dee/vram_cache.h:149-152,174-182`; `src/vram_cache.cpp:97-150` |
| Pinning | Pin through DMA; add compute pin; synchronize compute batch before unpin | `src/async_prefetcher.cpp:251-270,323-373`; `src/engine.cpp:2188-2236` |
| Fragmentation | Equal-size blocks in one fixed arena per Engine; adjacent holes coalesce | `src/vram_cache.cpp:17-79` |

## Capacity-8 memory budget and residency ceiling

Each GPU owns 20 layer Engines:

- Expert arenas: `20 * 8 * 6 MiB = 960 MiB`
  (`1,006,632,960` bytes/GPU).
- Dense model partition: `2,448,353,920` bytes on CUDA 0 and
  `2,448,358,016` bytes on CUDA 1.
- Native prefetch staging: `125,829,120` bytes/GPU.
- Native router weights: `20,971,520` bytes/GPU.
- Fixed work, MoE batch, and router dynamic buffers:
  `1,822,720 + 471,040 + 256,000 = 2,549,760` bytes/GPU.
- V17 peak NVML process use: `4,020,240,384` bytes/GPU; minimum CUDA free:
  `11,493,244,928` bytes/GPU; allocator OOMs/retries: zero.

Cap 16 adds exactly 960 MiB/GPU, projecting 5,026,873,344 bytes peak and
10,486,611,968 bytes free. Cap 32 adds 2,880 MiB/GPU over cap 8, projecting
7,040,139,264 bytes peak and 8,473,346,048 bytes free. Both fit a 15,360 MiB
T4 with substantial VRAM headroom. The sweep still records native arena totals,
NVML/process usage, allocator retries/OOMs, and minimum free memory because
PyTorch allocator counters exclude dee.cpp's native `cudaMalloc` arenas.

Host memory is the tighter secondary risk: v17 long-prompt evidence reached
7.5 GiB of persistent pinned expert staging plus 240 MiB of host prefetch rings
and 25.199 GB peak RSS. The short controlled sweep retains host RSS and pinned
staging evidence.

## Corrected v17 transfer baseline

V17's raw `dual-warm-profiled` expert trace and native engine totals agree:

- 1,280 expert requests.
- 272 resident hits and 1,008 misses/transfers/evictions.
- 6,341,787,648 bytes of expert H2D traffic.
- 252 loads and evictions per generated token.
- 1,585,446,912 H2D bytes per generated token.
- 573.284 ms summed copy-active duration.
- 59.277 ms useful copy/compute overlap across 160 transfers.
- Zero same-token duplicate `(layer, expert, GPU)` loads.

The old projected `transfer-analysis.json` reported only 42 transfers and
264,241,152 bytes because `AsyncPrefetcher` transfer IDs are Engine-local
(`include/dee/async_prefetcher.h:194`) while the analyzer deduplicated only
`(run_id, transfer_id)`. It also counted each eviction twice because the
projection emits both request-embedded and standalone eviction records. The M4
analysis fix scopes transfer identity by layer/GPU and deduplicates those paired
eviction records; it does not change runtime, teardown, lifetime, or correctness.

## First controlled experiment

`scripts/run_milestone4_capacity_sweep.py` runs paired control/profiled rows for
capacities 8, 16, and 32. Every pair holds the model, `Hello` prompt, token IDs,
four-token output, top-k, precision, 20/20 placement, and warmup fixed. Controls
measure throughput without profiling overhead; matched profiled rows provide
wall, transfers, reuse, overlap, synchronization, utilization, and memory
diagnostics. Exact tokens, all 40 layers, 100% device path, and zero host
fallback are hard gates for every row.
