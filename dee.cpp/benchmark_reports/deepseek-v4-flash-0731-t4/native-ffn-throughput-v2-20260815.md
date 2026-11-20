# Native FP4 FFN throughput — v2 (region gather)

Date: 2026-08-15
Branch: `freebuff/deepseek-v4-flash-0731-t4`
Commit: `0ea4dbf` (region gather; prior `ff66a72` GEMV kernel, `41101bc` revert pinned staging)
Benchmark: `scripts/deepseek_v4_native_throughput.py --shard model-00002-of-00048.safetensors --tokens 3 --profile`
Hardware: single Kaggle T4 (sm_75, CUDA 12.8)

## Result

| Metric | baseline (`c84bcbf`) | v2 (`0ea4dbf`) |
|---|---|---|
| native median (6 experts, layer 0) | 47.44 ms | **26.22 ms** |
| projected 43-layer decode | 2.04 s/token | **1.13 s/token** |
| projected tok/s | 0.49 | **0.887** |
| speedup vs host-dequant | 29.9x | **62.0x** |

native_ms_per_token = [201.1, 26.2, 25.3] (token 0 = cold: cuBLAS warmup + first mmap page faults).

## What changed (two commits)

1. **Coalesced GEMV kernel** (`ff66a72`): replaced cuBLAS's slow n==1 fallback for
   the down projection [4096,1,2048] with a one-block-per-row kernel (coalesced
   reads + tree reduce). `down_projection` 15.0 -> 0.86 ms/token — but NOT on the
   critical path, so the wall was unchanged (47.7 ms).

2. **Region gather** (`3b96c9a`, fixed in `0ea4dbf`): `get_staging_fp4` was
   gathering the six mmap regions (3 packed-I8 weights + 3 e8m0 scales) into a
   fresh heap vector (`resize` zeroing + first-touch page faults ~2 us/page) and
   the prefetcher then copied heap -> pinned. Two 80 MiB passes + page faults.
   Now the six regions are recorded verbatim and `cuda_submit` gathers them into
   its persistent pinned slot in ONE memcpy pass. `tensor_resolution` 48 -> 0.1
   ms/token.

## Stage profile (accumulated over tokens 0..1)

| Stage | ms (token 1, /2 for warm) |
|---|---|
| cpu tensor_resolution | 0.16 |
| cpu mmap_to_pinned (gather) | 39.15 (~19.6/token) |
| gpu h2d | 16.01 (~8/token) |
| gpu weight_conversion (dequant) | 12.73 (~6.4/token) |
| gpu gate_projection | 133.28 (token-0 cuBLAS warmup ~120 + warm) |
| gpu down_projection | 1.63 (~0.8/token) |

## Remaining bottleneck

The wall (26 ms/token) is now dominated by the **mmap->pinned gather** (~19.6
ms/token, ~160 MB DRAM traffic at ~8 GB/s) overlapping the H2D (~8 ms) + dequant
(~6.4 ms). The gather is memory-bandwidth bound. Candidate next levers:

- `cudaHostRegister` the mmap regions in place (skip the gather entirely) —
  H2D direct from pinned mmap; bounded pinning needed for a 37 GiB model.
- Dequant kernel speedup (6.4 -> ~1.3 ms memory-bound floor).
- cuBLAS warmup at init (removes the ~120 ms cold-start gate projection).

## Status

Still **not** full generated-text inference: this measures the routed-expert FFN
on one real layer shard; the dense path (embed/attention/RMSNorm/LM-head) is not
run. tok/s is a projection from the FFN-dominated DS10 decode trace.
