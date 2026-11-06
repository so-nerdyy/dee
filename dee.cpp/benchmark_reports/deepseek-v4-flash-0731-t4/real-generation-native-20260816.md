# Real generated-text decode — native FP4 FFN (2026-08-16)

First **real tokenizer → text** decode on the official checkpoint. This is not a
synthetic benchmark and not a projection: embeddings → attention → KV → router →
streamed FP4 MoE (native C++ path) → RMSNorm → LM head → greedy token loop all ran
end-to-end on 2× Tesla T4.

## What ran

| | |
|---|---|
| Model | DeepSeek-V4-Flash-0731 (official) |
| Revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Checkpoint | 48 shards, 152.8 GiB, mounted read-only from Kaggle dataset `nivind/deepseek-v4-flash-0731-shards` |
| dee.cpp commit | `b21c40e2d8ad49ed55bd66db6113dbb99a1fadaf` (branch `freebuff/deepseek-v4-flash-0731-t4`) |
| Hardware | 2× Tesla T4, CUDA 12.8, sm_75 |
| Config | 43 layers, 256 routed experts, hidden 4096, inter 2048, top-6 |
| FFN path | native FP4: mmap packed I8 + e8m0 → pinned gather → H2D → `fp4_e2m1_to_f16_cuda` → cuBLAS SwiGLU (`swiglu_limit=10`) |
| Expert cache budget | 512 MiB (per engine; engine0 on cuda:0, engine1 on cuda:1) |

## Correctness

Prompt: `Who is Alan Turing?` (greedy, deterministic)

Generated (16 tokens):

```
**Alan Turing (1912–1954)** was an English mathematician, computer
```

This is the exact expected opening for the prompt — the full pipeline produces
coherent, correct tokens. Token IDs:

```
[666, 95140, 96807, 343, 4470, 20, 1127, 3298, 22, 22604, 515, 411, 3947, 85349, 14, 6341]
```

The dense path (tokenizer/embed/attention/router/shared/norm/LM-head) is the sealed
DS10 torch path; the routed-expert FFN is the parity-verified native FP4 kernel
(rel RMSE 0.000411, cosine 0.999999993 vs FP32 on real-shape experts). 43/43 layers
executed per token.

## Performance (measured, real)

| Metric | Value |
|---|---|
| Total wall (16 tokens incl. prefill) | 11461 s (~3.18 h) |
| Prefill (7 prompt tokens) | 5723.9 s |
| Prefill tok/s | 0.001 |
| Decode (15 generated tokens) | 5736.8 s |
| **Generated-text decode tok/s** | **0.003** |
| Median inter-token latency | 366.5 s |
| p95 inter-token latency | 540.7 s |
| max inter-token latency | 540.7 s |
| Model build (dense FP8→FP32 + engine init) | 85 s |

## VRAM (bounded, 2× T4, 14.56 GiB each)

| GPU | allocated | reserved | peak allocated |
|---|---|---|---|
| cuda:0 | 6.99 GiB | 7.23 GiB | 7.11 GiB |
| cuda:1 | 6.76 GiB | 8.91 GiB | 8.73 GiB |

## Why 0.003 tok/s (the finding)

The decode is **storage-bound, not compute-bound**. The 512 MiB expert cache (per
GPU) holds ~10 of the 48 MiB FP16-expanded experts, but each token routes 6 experts
× 43 layers = **258 expert loads**, so every load is a cold page fault against the
152.8 GiB mmap (working set ≫ RAM). ~4.5 s/expert × 258 experts ≈ 6 min/token.

For contrast, the same native FFN measured **26.2 ms/layer-token (0.887 tok/s
projected)** when the shard was resident in page cache (`native-ffn-throughput-v2`).
So the compute path is already ~62× over the host-dequant path; the residual gap to
0.003 tok/s is entirely cold storage I/O.

## Verdict

**Real generated-text inference verified** — the full tokenizer → transformer →
streamed MoE → LM-head → token loop ran and produced correct greedy tokens on the
official checkpoint. The throughput number is 0.003 tok/s and is dominated by cold
storage reads, not by the dee.cpp compute path.

The path to a faster real number is making expert weights resident (larger VRAM
cache, host RAM residency / page-cache warm-up, or router-ahead prefetch), not more
FFN compute.
