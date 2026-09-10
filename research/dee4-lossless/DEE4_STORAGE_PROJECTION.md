# DEE4 storage projection (measured route/fill statistics x codec ratios)

## Baseline: where fill bytes come from (live 2xT4 run)

- Journal: 5676 expert uses / 16 decode tokens = **355 uses/token**.
- Bank working set touched: 2364 unique records x 12.75 MiB =
  **29.4 GiB miss bytes per 16-token run = 1.84 GiB/token**.
- Device: 0.30 GiB/s sustained cold (replay production datum).
- Lane overlap: worker/wall 2.49 of 3.
- Model check: 1.84 / 0.30 / 2.49 = **2.47 s/token fill** vs measured
  critical fill 42.0 s / 16 = 2.63 s/token. Closes within 6% — the
  accounting is sound.

## Projected bytes with the best real codec (zstd-10 whole, 0.9178)

| Quantity | Raw | x 0.9178 | Delta |
|---|---|---|---|
| Bytes / record | 12.75 MiB | 11.70 MiB | -1.05 MiB |
| Miss GiB / token | 1.84 | 1.69 | -0.15 |
| Fill s / token (model) | 2.47 | 2.26 | -0.21 |
| Fill wall / 16 tokens | 42.0 s | ~38.5 s | **~-3.5 s (~5% of 66 s decode)** |

Theoretical ceiling (order-1 bound 0.8940, no real codec attains it):
~-4.5 s off the 42 s bucket, ~7% of decode. That is the most any
lossless byte codec can ever buy on this bank.

## Why the saving is small even though "8%"

- Decode keeps up (zstd ~460 MB/s single-thread > device 300 MB/s),
  so the cost is CPU + complexity + a second on-disk format, not latency.
- The saving applies to MISS bytes only; reuse (p50 260 in production)
  already avoids most bytes for free.
- Worst-case records compress worse (whole zstd-10 worst 0.9196,
  order-1 worst 0.9004): provisioning must assume ~0.92, not the mean.
