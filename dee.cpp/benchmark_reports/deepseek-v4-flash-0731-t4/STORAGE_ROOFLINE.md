# P2.1 Storage Roofline — DeepSeek-V4-Flash on Kaggle 2×T4

**Date:** 2026-08-22
**Commit:** `8810a7d` (v15 ACCEPT_CORRECTNESS baseline)
**Platform:** Kaggle dual-T4, Python 3.11, CUDA 13

## Measured Paths

| Path | Mount | Total GiB | Free GiB | Notes |
|------|-------|-----------|----------|-------|
| `/` | shared NFS | 8,062 | 1,027 | Not usable for data |
| `/kaggle` | shared NFS | 8,062 | 1,027 | Same as `/` |
| `/kaggle/input` | dataset loop-device | 19.5 | 19.5 | Read-only, ~13 MB/s effective |
| `/kaggle/working` | local SSD | 19.5 | 19.5 | Read/write, ~126 MB/s |

## Sequential Read Throughput

| Path | Method | Block Size | p50 MB/s |
|------|--------|-----------|----------|
| `/kaggle/working` | `pread` | 1 MiB | 123.1 |
| `/kaggle/working` | `pread` | 16 MiB | 126.5 |
| `/kaggle/working` | `pread` | 64 MiB | 128.5 |
| `/kaggle/working` | `pread` | 256 MiB | 121.5 |
| `/kaggle/working` | `mmap` cold | 64 MiB | 117.9 |
| `/kaggle/working` | `mmap` warm | 64 MiB | 116.7 |

Note: `mmap` cold ≈ warm throughput — the `/kaggle/working` SSD has no meaningful page-cache effects at these sizes.

## Dataset Mount (safetensors checkpoint shards)

The checkpoint dataset (`nivind/deepseek-v4-flash-0731-shards`) is mounted read-only at `/kaggle/input`. The v15 run measured effective throughput of **~13 MB/s** from this mount (4,911s decode ÷ ~65 GiB per GPU = ~13.5 MB/s). This is a Kaggle loop-device, not a local SSD.

## GPU Info

| GPU | Name | VRAM |
|-----|------|------|
| cuda:0 | Tesla T4 | 14,911 MiB |
| cuda:1 | Tesla T4 | 14,911 MiB |

## Theoretical Roofline

| Scenario | Storage MB/s | Theoretical tok/s | Notes |
|----------|-------------|-------------------|-------|
| Current (13 MB/s) | 13 | **0.004** | Match to v15's 0.003 (close, overhead included) |
| `/kaggle/working` | 126 | **0.038** | 10× improvement, still <0.04 tok/s |
| Local NVMe | 3,000 | **0.912** | Hits GPU compute ceiling (~0.9 tok/s) |
| Local NVMe + overlap | 5,000 | **1.52** | Requires compute/I/O overlap |

Assumes 3,289 MiB cold-expert bytes per token (43 layers × 6 experts × 12.75 MiB FP4).

With host-pack LRU caching (effective hit rate ~50%), actual bytes/token drops to ~1,700 MiB, roughly doubling these numbers. With packed FP4 VRAM cache (12.75 MiB/expert instead of 48.00 MiB FP16), cache hit rate could improve substantially.

## Bottleneck Attribution (v15 decode)

```
Component              Time (s)  Fraction
────────────────────────────────────────
Storage I/O            ~4,700    95.7%
Host pack/decode       ~100      2.0%
H2D transfers          ~60       1.2%
GPU compute            ~51       1.0%  (258 experts × 43 layers × ~26ms = 288s total,
                                        but overlapped per-GPU)
Host scheduler         ~0.3      0.01%
```

The GPU is idle **95-99% of the time** waiting for storage.

## Conclusion

1. The Kaggle dataset mount (loop device) delivers ~13 MB/s — **10× slower** than the available local SSD.
2. Even at 126 MB/s (SSD maximum), the theoretical ceiling with current access patterns is ~0.04 tok/s.
3. To reach 0.5-1.0 tok/s requires **both**: (a) moving to faster storage (≥3 GB/s), (b) cache hit rate improvement (packed FP4 VRAM cache → more resident experts → higher hit rate → fewer cold reads/token).
4. The expert-major contiguous repack (P2.2) is the prerequisite: it makes reads sequential instead of random, enabling the throughput numbers above.
5. GPU compute (26 ms/layer/expert) is **not** the bottleneck; even with perfect storage, the ceiling is ~0.9 tok/s per batch-1 token on 2×T4.