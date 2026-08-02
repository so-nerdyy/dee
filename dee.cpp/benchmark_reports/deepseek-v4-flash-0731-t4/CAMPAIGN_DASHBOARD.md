# CAMPAIGN — DeepSeek-V4-Flash-0731 on Tesla T4 via Dynamic Expert Eviction

Status: **DS0–DS3, DS6, DS7, DS8 COMPLETE** (freeze, audit, ledger,
download plan/tool, resolver, one routed expert on T4, and the generalized
expert runtime + bounded cache executing routed+shared expert portions of
complete official layers on T4 with evidence). DS4/DS5 in progress.

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
| DS3 | Checkpoint download / Kaggle dataset plan | ✅ (plan + resumable shard tool + header pin) |
| DS4 | Tokenizer + encoding parity golden tests | 🔲 |
| DS5 | Trusted reference traces | 🔲 |
| DS6 | Freebuff tensor resolver for V4 | ✅ (Python ledger + C++ `TensorResolver` DEEPSEEK_V4 dialect, w1/w3/w2 + scale names) |
| DS7 | One routed expert on T4 | ✅ (kernel v5 COMPLETE, verdict `MATCH_WITHIN_TOLERANCE`, evidence `ds7-smoke-v5`) |
| DS8 | Expert cache + Dynamic Expert Eviction | ✅ (kernel v3 COMPLETE, verdict `ACCEPT_EXPERT_RUNTIME`, evidence `ds8-runtime-v3`) |
| DS9 | Architecture bring-up → first token | 🔶 (kernel v6 COMPLETE, verdict `REJECT_STATE` — full layer 20 executes on T4, compressor/indexer state diverges; evidence `ds9-v6-reject-state`) |
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

## DS7 milestone — one routed expert on T4 (COMPLETE)

Kaggle kernel `nivind/dee-cpp-deepseek-v4-flash-0731-ds7-expert-smoke` v5
terminated **COMPLETE** with result **PASS**.

- Pinned harness commit: `cc8910e8518f80947ab0fff711dc56e8a00279b1`
- Pinned harness SHA256: `d8c97005282e07b3c8a1be9ae4577b4579365a42de8a97536e7ea0fb811df9a7`
- Pinned reference SHA256: `9c28375b17898a6908d61ca3a769f4ba2eab4104e4049439cfa76784eaa86ef5`
- Evidence: `benchmark_reports/deepseek-v4-flash-0731-t4/ds7-smoke-v5/`
  (20/20 local validation checks PASS, incl. external cross-checks of the
  evidence copies of the harness and reference against the pinned SHAs).
- Archive SHA256: `f6992fbe1dcc9cf3504f8545802ab5294432279e7c8fa91570063b9b85687e6f`;
  manifest SHA256: `6e0b02f98d4f18cf2cee47b4e5e244613438bfdb6dd028c6bc6bbda40546953a`
  (from `archive-metadata.json`, cross-checked against the downloaded tar.gz).
- Verdict: `MATCH_WITHIN_TOLERANCE` — one official routed expert
  (`layers.6.ffn.experts.0`, shard `model-00008-of-00048`) executed on T4
  (CUDA, `candidate_executed_on_cuda: true`) and matched the trusted full-FP32
  reference within the predeclared tolerance contract:
  - `max_abs_error` 0.0046 (gate 2.0) ✓
  - `mean_abs_error` 0.0009 (gate 0.5) ✓
  - `mean_rel_error` 0.0059 (gate 0.01) ✓
  - `p99_rel_error` 0.038 (gate 0.05) ✓
  - `max_rel_error` 36.7 **excluded from the gate** — mathematically undefined
    for a near-zero reference element (~1e-4, catastrophic cancellation;
    absolute error stays tiny). Recorded as a diagnostic only.
- Integrity gate: header pin of the actual downloaded shard vs the committed
  cached header (canonical, EOL-immune) PASS; X-Linked-Etag opportunistic.
- Shard download used bounded retry (6 attempts, 2–32 s backoff) on transient
  CDN 5xx; header pin + reference self-test passed.
- `performance_comparable: false` — this is a correctness milestone, not a
  throughput number. No TPS is claimed.

Per the campaign contract, this validates **Family B** bring-up at the expert
level (FP4-packed storage + on-demand FP16 dequant → SM75 GEMV). It does not
imply full-model decode, an expert cache, or one-T4 residency.

## DS8 milestone — generalized expert runtime + bounded cache (COMPLETE)

Kaggle kernel `nivind/dee-cpp-deepseek-v4-flash-0731-ds8-expert-runtime` v3
terminated **COMPLETE** with result **PASS** and verdict
`ACCEPT_EXPERT_RUNTIME`.

- Pinned harness commit: `a2e1d214080af523655b3dcb8ad627274158052d`
- Pinned harness SHA256: `24a82819bbe7bbb76651c2b2470ec7887d3dfe8babc8006906a6c4cb7c373022`
- Evidence: `benchmark_reports/deepseek-v4-flash-0731-t4/ds8-runtime-v3/`
  (12/12 artifact hashes PASS + manifest + tar.gz cross-checks).
- Archive SHA256: `69b8f712b12a46ec90c7cafe083bdbfd8de284684b541f97c941035d8671867f`;
  manifest SHA256: `80ae02df3a0b3371ca5b52927a7eeeb32ee20428392b95c3695ffe33f19e37c1`.
- Scope: 3 complete official MoE layers — **3, 20, 41** — shards
  `model-00005/00022/00043-of-00048` — routed top-6 (union of selected
  experts across all 7 corpus cases loaded) + the official shared expert,
  executed on T4 CUDA through `DeepSeekExpertCache` (FP16-expanded active
  experts, bounded LRU+priority eviction, async staging stream/event).
- Numerical gates (warm combined MoE output vs trusted FP32 reference,
  predeclared DS8_TOLERANCE):
  - `max_abs_error` 0.0054 (gate 2.0) ✓
  - `mean_abs_error` 0.0008 (gate 0.5) ✓
  - `mean_rel_error` 0.0026 (gate 0.01) ✓
  - `p95_rel_error` 0.0067 (gate 0.03) ✓
  - `p99_rel_error` 0.034 (gate 0.05) ✓
  - cosine 0.9999998 (gate 0.999) ✓, normalized RMSE 0.0005 (gate 0.01) ✓
  - output-norm rel 1.8e-05 (gate 0.02) ✓, excluded fraction 0.0005 (gate 0.02) ✓
  - Shared-expert gate PASS on all layers (max_abs 0.0047).
- Cache correctness: cold==warm output bitwise, warm H2D bytes unchanged,
  **zero warm-reloaded experts** (all-hits replay); a 120 MiB
  eviction-pressure cache (41–44 evictions/layer) reproduced the full-budget
  output **bitwise** (`output_identical_to_full_budget: true`).
- Route agreement: reference re-routing matched the harness up-front routing
  on **all 7 corpus cases × 3 layers** (`reference_route_agreement: true`).
- Input: deterministic synthetic corpus (normal, low/high magnitude, sparse,
  adversarial, repeated, near-zero); official hidden-state traces remain a
  DS5 dependency.
- Iteration record: v1 crashed (`'str' object has no attribute 'float'` —
  corpus tuple unpacked name-into-x_route), v2 crashed (`KeyError: 203` —
  per-case reference needed the full selected-expert union, not only the
  first case's), v3 fixed both and reached COMPLETE.
- `performance_comparable: false` — correctness milestone, not a throughput
  number. No model TPS is claimed.

## DS9 milestone — one complete official layer on T4 (first full-layer run)

Kaggle kernel `nivind/dee-cpp-deepseek-v4-flash-0731-ds9-one-layer` v6
terminated **COMPLETE** with a full evidence set (no fatal error) and verdict
`REJECT_STATE` — a valid terminal rejection with the first failing component
proven. This is the first run where the COMPLETE official layer 20 executes
on T4 CUDA end-to-end.

- Pinned harness commit: `a3a90c2cb9ad18c75d7a39c475655585ad7fb41d`
- Pinned harness SHA256: `7d825a16fdc328259710d4508b03d5ea116d519bfcf17705b085652db61b6d04`
- Evidence: `benchmark_reports/deepseek-v4-flash-0731-t4/ds9-v6-reject-state/`
  (18 files: evidence JSON, manifest, environment, harness + 9 module copies,
  archive metadata, kernel log).
- Candidate: `candidate_cuda_resident: true`, peak VRAM 4.47 GB, sequence
  1.96 s, `performance_comparable: false`.
- **Attention path PASSES** (first time): attn_norm_in/out, qr, q, kv,
  kv_compressed, attn_o, attn_out, attn_hc_out all cosine ≈ 1.0 within
  predeclared DS9_TOLERANCES.
- **Router scores PASS** (cosine 0.9999999, max_abs 0.0016); route
  agreement true; exact gates `attn_window_idxs_exact` +
  `attn_compress_idxs_exact` true.
- **Cache correctness PASS**: 77 loads, 0 warm reloads, warm outputs bitwise
  identical, 3.88 GB H2D, 0 evictions.
- **First failing component (REJECT_STATE)** — compressor/indexer state
  buffers diverge structurally:
  - `compressor_kv_state` max_rel 0.387, `indexer_compressor_kv_state` 0.548
    (both vs 0.001 bound);
  - `compressor_score_state` + `indexer_compressor_score_state`
    `finite_agreement: false` (reference and candidate disagree on which
    positions are the `-inf` sentinel vs written) — a state-semantics
    mismatch, not ULP drift;
  - `attn_kv_cache` and `indexer_kv_cache` match exactly (0.0).
- Secondary gate failures (recorded, not the attribution):
  - `expert_ids_exact: false` at step 0 (top-6 boundary flip between the
    CPU fp32 reference and CUDA bf16 candidate);
  - `moe_out`/`shared_out` p99_rel 0.07–0.12 vs the predeclared 0.05 gate
    (fp16-payload vs fp32-reference gap at near-zero elements);
  - `indexer_scores` step-0 metrics NaN (harness artifact: the causal mask
    leaves `-inf` in the captured scores, which poison cosine/error
    metrics).
- Iteration record (all preserved, none overwritten): v1 `KeyError: 214`
  (chained-sequence union), v2 `KeyError: 'w1.weight'` (lazy dict keyed by
  full tensor names), v3 CUDA×CPU device mismatch (x_step stayed on CPU),
  v4 `freqs_cis`/window-idx/indexer-mask device boundaries, v5 candidate
  FFN returned CPU tensors (`.cpu()` leftover from the DS8 harness), v6
  first complete full-layer execution → `REJECT_STATE`.
- Next: diagnose the compressor/indexer state divergence (write position,
  sentinel, or aliasing) with targeted device diagnostics; fix the
  `indexer_scores` `-inf` metrics artifact; re-run v7.

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
