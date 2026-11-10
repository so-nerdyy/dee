# Native FP4 FFN — real-weight throughput (2026-08-14)

## What this measures

First real-TPS-relevant measurement on OFFICIAL DeepSeek-V4-Flash-0731 expert
weights. Shard `model-00002-of-00048.safetensors` (layer 0, 256 routed
experts, 3.566 GB) is streamed through the two FFN candidate paths at the real
4096x2048 dims, top-6 experts, cold (no-reuse) stream — the realistic decode
scenario where routing is near-uniform.

Kernel: `nivind/dee-cpp-dsv4-native-throughput` v1 (`f7088dc`), single T4.

## Measured per-layer-token time (6 experts, median of 5 cold tokens)

| Path | Median ms | Breakdown |
|---|---|---|
| **Native** (`moe_forward_experts`: mmap → H2D compressed FP4 → on-device dequant → cuBLAS SwiGLU) | **50.2 ms** | — |
| **Host-dequant** (DS8/DS10: torch FP4→FP16 on CPU → pin → H2D FP16 → matmul) | **1513 ms** | dequant ~1135–1819 ms, H2D ~54 ms, matmul ~3 ms |

- **Speedup: 30.1×**
- H2D per token: native ~80 MiB (compressed FP4) vs host ~302 MiB (FP16) —
  3.8× less transfer, and the host dequant (~1.4 s/token) is eliminated
  entirely (dequant moves to the GPU transfer stream).
- Host path reproduces the sealed DS10 decode number: 1513 ms × 43 layers ≈
  65 s/token ≈ DS10's measured 66.5 s/token (0.015 tok/s).

## Projected full-model decode

Native FFN is ~97% of the DS10 decode wall-clock, so:

- 50.2 ms/layer × 43 layers ≈ **2.16 s/token → ~0.46 tok/s** (FFN-dominated
  projection, dense path unchanged).

This is a real-weight measurement, not synthetic. It is **not** full
generated-text inference: the dense path (embed / sparse-attention / router /
RMSNorm / LM head) is not run here, so it is not a tokenizer-to-text tok/s.

## Remaining bottleneck

50 ms/token for 6 experts ≈ 8 ms/expert is still serialized: the prefetcher
streams experts one-at-a-time with per-expert stream sync. Overlapping the 6
expert H2D+dequant transfers (the engine already has an async prefetcher ring)
is the next lever, followed by the dense path (attention/router) which is the
remaining ~1.5 s/token in the DS10 trace.
