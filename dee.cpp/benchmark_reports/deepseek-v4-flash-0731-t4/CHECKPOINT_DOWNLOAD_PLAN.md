# CHECKPOINT_DOWNLOAD_PLAN — DeepSeek-V4-Flash-0731 (DS3)

Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
Pinned revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
Total: 166,878,536,440 bytes (48 safetensor shards, 72,317 tensors).
The repository is **public (not gated)** — no auth token required.

## Search for an existing Kaggle mirror

Checked 2026-07-31 with `kaggle datasets list -s deepseek-v4`:
no verified mirror of this checkpoint exists on Kaggle. We must create one.

## Download strategy

Never store two canonical copies. One canonical download lives in
`dee.cpp/tmp/dsv4-checkpoint/` (or a Kaggle dataset once created); all other
paths use hard links / symlinks / direct mapped access.

### Step 1 — local/resumable download (all 48 shards)

Use `tools/download_deepseek_v4_shard.py`:

```bash
python tools/download_deepseek_v4_shard.py \
  --shard model-00001-of-00048.safetensors \
  --out tmp/dsv4-checkpoint \
  --manifest benchmark_reports/deepseek-v4-flash-0731-t4/CHECKPOINT_MANIFEST.json
```

- Resumable HTTP range downloads (0-`size-1`, `Range` header, append mode).
- Every shard is verified against the manifest's `compressed_bytes` size, and
  the downloaded file's ACTUAL safetensors header is re-serialized and
  hashed to match the manifest `header_sha256` of the pinned revision (a
  same-size tampered file fails). Full-file SHA256 is a post-download step
  recorded in the dataset manifest.
- `--all` downloads all 48 shards sequentially.

Download volume: 166.88 GB. Kaggle local disk (20 GB) cannot hold the full
checkpoint in one dataset version. **Kaggle plan below partitions it.**

### Step 2 — Kaggle dataset mirror (chosen plan)

Because the checkpoint exceeds one Kaggle dataset slot, partition into
immutable revision-pinned datasets, each with its own manifest:

- `nivind/deepseek-v4-flash-0731-part1` — shards 00001–00016 (~55.6 GB)
- `nivind/deepseek-v4-flash-0731-part2` — shards 00017–00032 (~55.6 GB)
- `nivind/deepseek-v4-flash-0731-part3` — shards 00033–00048 (~55.6 GB)
- plus config/index/tokenizer in `part1`.

Creation path (run once, on a Kaggle notebook with internet):
1. Download shards in the notebook with the same tool.
2. `kaggle datasets create -p <dir>` per part.
3. Record dataset version IDs + every shard size in `RUN_REGISTRY.json`.

Alternative (lower risk for the first smoke runs): do not mirror; download
only the one shard needed for DS7 (model-00008-of-00048.safetensors, which
holds `layers.6.ffn.experts.0.*` plus the index/config) directly inside the
smoke notebook. That is ~3.5 GB and fits Kaggle's 20 GB local disk.

## Why shard 00008 for the DS7 smoke

From the validated ledger (`MODEL_LEDGER.json`):

- `layers.6.ffn.experts.0.w1.weight` → `model-00008-of-00048.safetensors`
- `layers.6.ffn.experts.0.w1.scale`  → `model-00008-of-00048.safetensors`
- `layers.6.ffn.experts.0.w2.weight` → `model-00008-of-00048.safetensors`
- `layers.6.ffn.experts.0.w2.scale`  → `model-00008-of-00048.safetensors`
- `layers.6.ffn.experts.0.w3.weight` → `model-00008-of-00048.safetensors`
- `layers.6.ffn.experts.0.w3.scale`  → `model-00008-of-00048.safetensors`

So one shard gives us a complete routed expert.

The DS7 smoke kernel (`kernel-metadata.json`) intentionally requests ONE
GPU (`gpu_count: 1`): a single-expert validation needs no model split. The
dual-T4 configuration is only introduced at DS10.

## Validation on mount

After mounting any Kaggle dataset:

1. Verify every shard file size against the manifest.
2. Verify the shard header JSON hash against the committed cached headers.
3. Re-run `python tools/build_deepseek_v4_ledger.py --model-dir <mounted index>
   --report-dir <fresh output>` and require tensor_count 72,317 /
   compressed_bytes 166,878,536,440.
4. Never silently mix revisions — every artifact pins
   `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.

## Rules

- Do not duplicate the checkpoint; one canonical copy only.
- Never present an unvalidated shard as the official checkpoint.
- Record archive SHA256, manifest status, and dataset versions in
  `RUN_REGISTRY.json`.
- **Mandatory before use**: the size + header-pin checks cannot detect
  corruption in the body of a same-size file. For the bulk `--all` path,
  run the post-download full-file SHA256 (streamed) against HF's
  X-Linked-Size/hash and record it before any shard is used for execution.
  The DS7 smoke already streams a full-file SHA256 into its evidence.
