# DEE4 lossless scan (method + results)

Mission: decide whether real Dee4 DeepSeek-V4-Flash expert records can be
stored LOSSLESSLY in fewer physical bytes. Every cold byte from the
`/tmp` bank costs device time (Phase 1: ~0.3 GB/s ceiling), so FEWER
bytes per fill is the feed-side lever under test here.

## Method

- **Record source**: exact bank bytes reconstructed from source shards
  (`research/dee4-lossless/records.py`), NOT dequantized/requantized
  approximations. Each record verified sha256 against the bank integrity
  sidecar before any codec touches it.
- **Selection**: all 2364 trace-bank records = sorted journal union
  (reproduced exactly from `inputs/journal.txt`).
- **Regions**: w1w/w3w/w2w packed FP4 (4 MiB each), w1s/w3s/w2s scales
  (256 KiB each), whole record (12.75 MiB). See `DEE4_RECORD_LAYOUT.md`.
- **Battery** (`dq_codecs.py`, `scan.py`):
  - `lz4_fast` (lz4-frame default), `lz4_hc` (level 16),
    `zstd_1/3/10`, `nib_split_zstd3` (lo/hi nibble streams, zstd-3 each),
    `order0`/`order1` Shannon bounds, real rANS (`constriction`,
    order-0, model cost excluded), `zeros` (zero/RLE stats).
  - Every codec roundtrip verified byte-for-byte (sha256); any mismatch
    fails the row, never the run.
  - Tiers: `sweep` (all records), `full` (every 25th: +lz4_hc, zstd_10,
    rANS, nibble/bit-plane structure).
- **Global-state studies** (labeled, not per-record): zstd dictionary
  (train 256 → test 64), exact region dedup via region sha256.
- **Venuess**: 8-record pilot over HF range fetches (layers 0–42);
  full 2364 sweep on a Kaggle CPU worker with the dataset mounted
  (`dee.cpp/experiments/dee4_lossless/kernel_scan/`).

## Pilot results (8 records, all EXACT-verified)

Tight across layers (min/max within 0.005): the bank is homogeneous;
expect the full sweep to confirm, not surprise.

| Region | lz4 | zstd-10 | order0 | order1 | rANS |
|---|---|---|---|---|---|
| w1w/w3w/w2w packed | 1.00 | 0.964 | 0.959 | 0.939 | 0.959 |
| w1s/w3s/w2s scales | 0.18–0.48 | 0.16 | 0.12 | 0.12 | 0.12 |
| whole record | 0.95–0.97 | 0.917 | 0.949 | 0.893 | n/a |

- Nibble-split + zstd: 1.04 (WORSE — destroys byte alignment, gains nothing).
- Nibble streams: 3.86 bits/nibble each; conditionals 3.81;
  bit-planes 0.95–1.00 bit (sign bit 0.99999). Maximum entropy at every
  sub-byte decomposition — no nibble/bit-plane/context coder can help.
- Zeros: 0.4% of packed bytes, longest run 3 → RLE useless.
- Scales use 4 distinct byte values (hence order-0 0.12) but are only
  5.9% of the record.
- zstd-dict (train 7 → test 1): 0.9203 vs independent 0.9203 — zero gain.
- Exact region dedup 8/8 unique everywhere.

## Full-sweep results (2364/2364 records, 0 errors, 117,736 rows)

Every record reconstructed byte-exact (0 mismatches vs integrity sidecar).
Distributions are razor-tight (worst ~= mean): the bank is homogeneous.

| Region | lz4 | zstd-1/3/10 | order0 | order1 | rANS (95) |
|---|---|---|---|---|---|
| w1w/w3w/w2w packed | 1.0000 | 0.965/0.967/0.965 | 0.9597–0.9599 | 0.940–0.941 | 0.9597–0.9599 |
| w1s/w3s/w2s scales | 0.48/0.19 | 0.196/0.188/0.164 | 0.120–0.123 | 0.120–0.122 | 0.120–0.123 |
| whole record | 0.9697/0.9519 | 0.9194/0.9210/0.9178 | 0.9499 | 0.8940 (worst 0.9004) | n/a |

- Cross-expert zstd-dict (train 256 -> test 64): mean 0.9201
  (min 0.9192, max 0.9231) vs independent 0.9210 — **zero gain**.
- Exact region dedup implied by tightness + pilot 8/8-unique: no sharing.
- Whole-record decode throughput (median): lz4 640, zstd-1 460,
  zstd-3 435, zstd-10 481 MB/s — all above the 300 MB/s device rate.
- Raw evidence: `dee.cpp/experiments/dee4_lossless/scan-live-full/`
  (`scan_out.csv` 117,736 rows, `scan_summary.json`).
- Aggregate table: `DEE4_CODEC_RESULTS.csv` (this directory).

## Classification (per-record independent storage)

- `>= 0.90`: SIDE_OPTIMIZATION
- `0.75–0.90`: USEFUL
- `0.60–0.75`: STRONG
- `< 0.60`: BREAKTHROUGH_CANDIDATE
