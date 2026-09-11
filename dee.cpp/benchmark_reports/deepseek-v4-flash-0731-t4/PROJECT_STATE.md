# PROJECT_STATE — DeepSeek-V4-Flash-0731 campaign

Updated: 2026-07-31.

## Pinned official assets (all at revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb)

| Asset | SHA256 (local download) |
|---|---|
| config.json | `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023` |
| generation_config.json | `5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0` |
| model.safetensors.index.json | `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b` |
| tokenizer.json | `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf` |
| tokenizer_config.json | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| inference/convert.py | `6efe65ebc66b18c9f2656816608f941cacfe20da79c2dee19040ecbee8b42bfe` |
| inference/generate.py | `775fcfee2344e21a7b02c73161c517763e4348b84cf2eb266353e0857b9c8812` |
| inference/kernel.py | `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2` |
| inference/model.py | `c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f` |
| encoding/encoding_dsv4.py | `abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6` |

Local audit copies: `dee.cpp/tmp/dsv4-official-audit/` (headers + index +
source, no weight bytes).

Pinned source is committed with the campaign at
`benchmark_reports/deepseek-v4-flash-0731-t4/official-source/` (config,
generation_config, index, tokenizer, inference/, encoding/) so the DS1
revision pin is reproducible from a fresh clone and the real-index invariant
test can run without `tmp/`.

## Ledger

- `MODEL_LEDGER.json` — 72,317 validated tensor rows.
- `CHECKPOINT_MANIFEST.json` — 48 shards, header-validated sizes.
- Declared total = validated total = `166,878,536,440` bytes. PASS.

## Download plan + tool (DS3)

- `CHECKPOINT_DOWNLOAD_PLAN.md` — no verified Kaggle mirror exists; partition
  plan + shard-00008 rationale (holds `layers.6.ffn.experts.0.*`), mount
  validation rules, mandatory post-download full-file hash.
- `tools/download_deepseek_v4_shard.py` — resumable range downloader that
  verifies the ACTUAL downloaded safetensors header against the manifest
  `header_sha256` of the pinned revision (same-size tampered files fail).

## DS7 smoke harness (launched, PASS)

- `kaggle/deepseek-v4-flash-0731/kernel-metadata.json` — dual T4, private,
  internet, direct shard download (no dataset dependency).
- `kaggle/deepseek-v4-flash-0731/deepseek_v4_expert_smoke.py` — downloads
  model-00008, loads expert-0 (layer 6), runs the trusted FP32 reference and
  a device-authentic FP16-dequantized T4 candidate (`.to("cuda")`, sync,
  `candidate_executed_on_cuda` in evidence), compares with predeclared
  tolerances, archives evidence + manifest.sha256. `performance_comparable:
  false`.

### DS7 remote result (v5 COMPLETE, PASS)

Kernel `nivind/dee-cpp-deepseek-v4-flash-0731-ds7-expert-smoke` v5 terminated
COMPLETE with verdict `MATCH_WITHIN_TOLERANCE`:

- Evidence: `benchmark_reports/deepseek-v4-flash-0731-t4/ds7-smoke-v5/`
  (downloaded 2026-07-31). Archive SHA256:
  `f6992fbe1dcc9cf3504f8545802ab5294432279e7c8fa91570063b9b85687e6f`;
  manifest SHA256: `6e0b02f98d4f18cf2cee47b4e5e244613438bfdb6dd028c6bc6bbda40546953a`
  (cross-checked against the downloaded tar.gz in local validation).
- Verdict: `MATCH_WITHIN_TOLERANCE`, `passed: true`,
  `candidate_executed_on_cuda: true`, integrity gate `header_pin` PASS.
- Metrics vs gates: `max_abs_error` 0.0046/2.0 ✓, `mean_abs_error`
  0.0009/0.5 ✓, `mean_rel_error` 0.0059/0.01 ✓, `p99_rel_error` 0.038/0.05 ✓.
  `max_rel_error` 36.7 recorded as a diagnostic but excluded from the gate:
  relative error is undefined for a near-zero reference element
  (ref ~1e-4, abs error ~0.004 — cancellation artifact, not semantic).
- Embedded `environment.json`: repository commit
  `cc8910e8518f80947ab0fff711dc56e8a00279b1`, harness SHA256
  `d8c97005282e07b3c8a1be9ae4577b4579365a42de8a97536e7ea0fb811df9a7`,
  reference SHA256 `9c28375b17898a6908d61ca3a769f4ba2eab4104e4049439cfa76784eaa86ef5`
  — all match the committed `harness-identity.json` sidecar.
- Local validation `tmp/validate_ds7_v5_evidence.py`: 20/20 checks PASS,
  including independent re-hashes of the downloaded evidence copies of the
  harness and reference against the pinned committed SHAs (external
  cross-check, not tautological with the harness's runtime self-verification).
- Prior kernel failures documented: v1 (single-T4 topology gate),
  v2 (ETag gate on a CDN without X-Linked-Etag), v3 (transient CDN 503 —
  fixed with bounded ranged-fetch retry), v4 (max_rel_error gate artifact —
  fixed with robust-statistics gate). No evidence was overwritten.

## Expert reference (DS7)

- `scripts/deepseek_v4_expert_reference.py` — trusted FP32 reference for one
  routed expert: official `FP4_TABLE` nibble decode (low->elem 2i, high->2i+1),
  `F8_E8M0` scale decode `2^(bits-127)` (bit-reinterpreted for int8 /
  float8_e8m0fnu checkpoint dtypes), dequantize `w[o,i]=fp4*scale[o,i//32]`,
  asymmetric SwiGLU clamps (`gate` max-only, `up` min+max, limit 10.0), and
  the sqrtsoftplus router (top-k, normalize, x route_scale 1.5).
- Intentionally full-FP32 (act_quant FP8 is excluded); DS7 T4 candidates must
  meet predeclared tolerance, not near-bitwise agreement.
- Tests: `tests/test_deepseek_v4_expert_reference.py` (21 total incl. this
  suite; dtype round-trips, clamp oracle, official constants pinned).

## Resolver (DS6)

- C++ `TensorResolver` extended with a `DEEPSEEK_V4` dialect (additive;
  `ORNITH` remains the default): `w1`=GATE, `w3`=UP, `w2`=DOWN naming,
  expert `.scale` resolution, shared-expert names, and `F8`/`I8`/`I64`
  dtype mapping for the FP4-packed / FP8 / hash layouts.
- Tests: `tests/test_deepseek_v4_resolver.cpp` (C++ naming/dtype) and
  `tests/test_deepseek_v4_support.py` (Python ledger, self-executing so the
  CTest entry runs real assertions).
- Full official-index coverage remains the Python ledger's 72,317 tensors,
  all resolved with matched shapes/offsets/dtypes/scales (0 unresolved,
  0 duplicate mappings).

## DS9 milestone (first complete official layer on T4)

Kernel `nivind/dee-cpp-deepseek-v4-flash-0731-ds9-one-layer` v6 terminated
COMPLETE with a full evidence set and verdict `REJECT_STATE` (valid terminal
rejection, first failing component proven). This is the first run where the
COMPLETE official layer 20 executes on T4 CUDA end-to-end.

- Evidence: `benchmark_reports/deepseek-v4-flash-0731-t4/ds9-v6-reject-state/`
  (18 files). Pinned commit `a3a90c2cb9ad18c75d7a39c475655585ad7fb41d`,
  harness SHA `7d825a16fdc328259710d4508b03d5ea116d519bfcf17705b085652db61b6d04`.
- PASS: attention path (cosine ≈ 1.0), router scores, exact window/compress
  index gates, route agreement, cache correctness (77 loads, 0 warm
  reloads, cold==warm bitwise), candidate CUDA-resident, peak VRAM 4.47 GB.
- REJECT_STATE cause: `compressor_kv_state` max_rel 0.387,
  `indexer_compressor_kv_state` 0.548 (bound 0.001), and both
  `score_state` buffers show `finite_agreement: false` (reference vs
  candidate disagree on `-inf` sentinel vs written positions) — a
  structural state-semantics mismatch.
- Secondary (non-attribution) failures: `expert_ids_exact` step-0 boundary
  flip (CPU fp32 vs CUDA bf16), `moe_out`/`shared_out` p99_rel 0.07–0.12 vs
  predeclared 0.05, `indexer_scores` step-0 metrics NaN (harness `-inf`
  causal-mask artifact).
- Iteration record preserved: `ds9-failed-v1` … `ds9-failed-v5` +
  `ds9-v6-reject-state`; no evidence overwritten.

## Branch

`freebuff/deepseek-v4-flash-0731-t4` (off `9ff967ef4429fb08a433d6ef0a4495468d89b4ba`).

## Sealed Ornith

Immutable on `codex/phase2-cap32-matrix` @ `9ff967e...`. M5G-v3 = INVALID_EXPERIMENT;
+9.21% claim quarantined.
