# CAMPAIGN — DeepSeek-V4-Flash-0731 on Tesla T4 via Dynamic Expert Eviction

Status: **DS0–DS2 COMPLETE** (freeze, audit, ledger). DS6 (resolver) in progress.

## Campaign identity

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Official revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Branch | `freebuff/deepseek-v4-flash-0731-t4` |
| Hardware target | Kaggle dual T4 (SM75, 2×16 GB) → one T4 stretch |
| Storage | 166.88 GB (48 shards, 72,317 tensors) |
| Prior objective | GLM-5.2-on-L40S — **CANCELLED** |

## Sealed Ornith state (immutable)

| Item | Value |
|---|---|
| Branch | `codex/phase2-cap32-matrix` @ `9ff967ef4429fb08a433d6ef0a4495468d89b4ba` |
| M5G-v3 smoke | v6 COMPLETE, gates PASS, archive SHA256 `923b65fbd00e680ac64036d4738acced65bad57a181910a21aa73424fe427c19` |
| M5G-v3 verdict | `INVALID_EXPERIMENT` (bitwise-exact at selected boundary; residual/event provenance unproven) |
| +9.21% M5G claim | quarantined, not accepted, not published |

M5G-v1/v2/v3 evidence is immutable. No M5H work until the DeepSeek campaign reaches a valid verdict.

## Milestone ladder

| ID | Milestone | Status |
|---|---|---|
| DS0 | Freeze Ornith, create branch, campaign scaffold | ✅ |
| DS1 | Official source/config audit + revision pin | ✅ |
| DS2 | Byte-accurate tensor ledger | ✅ |
| DS3 | Checkpoint download / Kaggle dataset plan | 🔲 |
| DS4 | Tokenizer + encoding parity golden tests | 🔲 |
| DS5 | Trusted reference traces | 🔲 |
| DS6 | Freebuff tensor resolver for V4 | 🔲 (Python resolver done; C++ `TensorResolver` extension pending) |
| DS7 | One routed expert on T4 | 🔲 |
| DS8 | Expert cache + Dynamic Expert Eviction | 🔲 |
| DS9 | Architecture bring-up → first token | 🔲 |
| DS10 | Dual-T4 full-model decode | 🔲 |
| DS11 | One-T4 path | 🔲 |
| DS12 | DSpark speculative decoding | 🔲 |

## Ledger headline numbers (from real shard headers, not estimates)

| Component | Tensors | Compressed | FP16-expanded |
|---|---|---|---|
| Routed experts | 66,048 | 147.17 GB | 571.37 GB |
| DSpark | 4,705 | 10.86 GB | 40.90 GB |
| Attention dense | 603 | 4.60 GB | 9.20 GB |
| Shared experts | 258 | 1.08 GB | 2.16 GB |
| Embedding | 1 | 1.06 GB | 1.06 GB |
| LM head | 1 | 1.06 GB | 1.06 GB |
| Hash/compress | 615 | 0.94 GB | 1.04 GB |
| Router | 86 | 0.11 GB | 0.10 GB |
| **Total** | **72,317** | **166.88 GB** | **626.88 GB** |

## Precision families (measured, not assumed)

- **A — official reference**: BF16/FP8 semantics on supported hardware, source of truth.
- **B — FP4 storage, FP16 accumulation**: on-demand unpack of `I8` expert pairs (e2m1fn) + `F8_E8M0` scales → SM75 FP16 GEMV.
- **C — offline INT8 expert conversion**: block-scale INT8 on T4 INT8 tensor cores; never labeled bitwise exact.
- **D — FP16 active subset**: correctness bring-up only; no full expansion to host RAM.

## Reproducibility

Rebuild the ledger from the committed pinned assets (no network, no weight bytes):

```bash
python tools/build_deepseek_v4_ledger.py \
  --model-dir benchmark_reports/deepseek-v4-flash-0731-t4/official-source \
  --report-dir benchmark_reports/deepseek-v4-flash-0731-t4
```

`MODEL_LEDGER.json` (68 MB) is regenerable from `official-source/` + `shard-headers/`;
`CHECKPOINT_MANIFEST.json` records the real header SHA256s.

## Acceptance rules

- Never present Ornith TPS as DeepSeek TPS.
- Never present expert microbenchmarks as model TPS.
- Never present proposal TPS as accepted TPS.
- Never present dual-T4 as one-T4.
- Never present quantized results as bitwise exact.
- Never present preview-model traces as official-0731 traces.
