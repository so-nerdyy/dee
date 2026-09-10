# DEE4 trace-bank record layout (measured + code-traced)

Source model: `deepseek-ai/DeepSeek-V4-Flash-0731`
revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.
Bank: `dee4-v3-trace`, codec `deepseek-fp4-e2m1-e8m0`, produced by
`dee.cpp/kaggle/deepseek-v4-flash-0731/repack_to_dee4.py::repack_trace`
(verbatim byte copy, no numerical conversion).

## Model geometry

| Param | Value | Provenance |
|---|---|---|
| Hidden dim | 4096 | `deepseek_v4_model.py:295`, `w2.out=4096` |
| MoE intermediate dim | 2048 | `deepseek_v4_model.py:296` |
| Layers | 43 | `deepseek_v4_model.py:297` |
| Routed experts / layer | 256 | `deepseek_v4_model.py:300` |
| Top-k | 6 | `deepseek_v4_model.py:302` |
| FP4 format | E2M1 (fn), 16-entry table | `deepseek_v4_expert_reference.py:34-41` |
| Scale format | F8 E8M0, 1 byte = `2^(bits-127)` | `deepseek_v4_expert_reference.py:49-62` |
| FP4 block (group) size | 32 cols | `repack_to_dee4.py:44`, `metadata group_size=32` |
| Nibble order in byte | low nibble = even index, high = odd | `cuda_convert.cu:241-242` |

## Per-record region map (record = 13,369,344 bytes = 12.75 MiB)

| Region | Offset | Size (bytes) | Stored dtype/shape | Logical shape |
|---|---|---|---|---|
| `w1.weight` (gate) | 0 | 4,194,304 | I8 `[2048,2048]` | FP4 `[2048,4096]` |
| `w3.weight` (up) | 4,194,304 | 4,194,304 | I8 `[2048,2048]` | FP4 `[2048,4096]` |
| `w2.weight` (down) | 8,388,608 | 4,194,304 | I8 `[4096,1024]` | FP4 `[4096,2048]` |
| `w1.scale` | 12,582,912 | 262,144 | E8M0 `[2048,128]` | 1 byte / 32 cols |
| `w3.scale` | 12,845,056 | 262,144 | E8M0 `[2048,128]` | ditto |
| `w2.scale` | 13,107,200 | 262,144 | E8M0 `[4096,64]` | ditto |

Arithmetic: packed `3 x 4,194,304 = 12,582,912` (94.1%) +
scales `3 x 262,144 = 786,432` (5.9%) = `13,369,344`.
Validated by `expert_store.cpp:408-417` range checks.

## Format invariants (all verified)

- **No padding, no header, no alignment gaps.** Offsets are contiguous
  (`repack_to_dee4.py:128-160`, validator `:876-883`).
- **File offset = `record_index x 13,369,344`** (`expert_store.cpp:536-541`,
  `repack_to_dee4.py:902-904`).
- **Sparse selection, not dense.** 2364 records = sorted union of the
  trace journal (`layers,expert` pairs), mapped via sorted `records[]`
  + `lower_bound` (`expert_store.cpp:516-534`). Dense universe would be
  `43 x 256 = 11,008`. Journal union reproduced locally: exactly 2364.
- **Component order is the format**: w1,w3,w2 weights then w1,w3,w2
  scales (`repack_to_dee4.py:52-57`, `expert_store.h:28-30`,
  `engine.cpp:2907-2911`).
- **Row-major C order**, implicit (`weight_mmap.cpp:106-108`,
  `cuda_convert.cu:241-244`); repack never transposes.
- **Per-record/component sha256 live only in sidecars**
  (`integrity.jsonl`, `dee4-integrity.jsonl`); the bank file itself has
  no checksums. Whole-bank `data_sha256=c83462ba...` (`metadata.json:8`).
- **Reconstruction proven byte-exact**: independent Python
  re-implementation (safetensors header parse + range concat in component
  order) reproduces records with sha256 identical to the integrity
  sidecar — 8/8 pilot records across layers 0–42, then all 2364 in the
  full scan (see `DEE4_LOSSLESS_SCAN.md`).

## What this means for compression

- The record is two sharply different populations: 94.1% near-maximum-
  entropy packed FP4 nibbles + 5.9% tiny-alphabet E8M0 scales
  (4 distinct byte values observed). Any codec is decided by the packed
  94.1%.
- No padding to strip, no metadata to factor out: the stored bytes ARE
  the payload. Savings must come from entropy coding, not relayout.
