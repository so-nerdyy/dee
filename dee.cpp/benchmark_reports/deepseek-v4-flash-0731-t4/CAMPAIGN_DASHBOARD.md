# CAMPAIGN — DeepSeek-V4-Flash-0731 on Tesla T4 via Dynamic Expert Eviction

Status: **DS0–DS4, DS6, DS7, DS8 COMPLETE; DS9 terminated (REJECT_NUMERICAL
accepted); DS5 BLOCKED (hardware-format)** — official tilelang kernels need
dynamic shared memory above the T4 SM75 64 KiB ceiling (fp8_gemm 82,048 B;
sparse_attn ~280 KiB), so the pinned official reference cannot execute on the
campaign's only GPU (see `ds5-v6-hardware-blocker/DS5_BLOCKER_REPORT.md`).

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
| DS4 | Tokenizer + encoding parity golden tests | ✅ (official `encoding_dsv4.py` + `tokenizer.json` pinned w/ SHA-256; wrapper `scripts/deepseek_v4_encoding.py`; 15 golden tests — exact IDs for chat/thinking/low/high/max reasoning/tool/multi-turn; parse roundtrips; pinned 2026-08-02) |
| DS5 | Trusted reference traces | ⛔ **BLOCKED — hardware-format** (kernel v6, `ds5-v6-hardware-blocker/`): scaffold + 16 tests committed; identity/config/tokenizer/shard/convert/reference-load all PASS on Kaggle T4; the pinned official tilelang kernels cannot launch on SM75 (fp8_gemm requests 82,048 B dynamic shared > 64 KiB ceiling; sparse_attn h=64,d=512 needs ~280 KiB). Reference = official inference stack is not SM75-runnable; unblocking requires a non-T4 reference host, a documented kernel re-tile (product decision), or reusing the sealed DS8/DS9 trusted-reference discipline |
| DS6 | Freebuff tensor resolver for V4 | ✅ (Python ledger + C++ `TensorResolver` DEEPSEEK_V4 dialect, w1/w3/w2 + scale names) |
| DS7 | One routed expert on T4 | ✅ (kernel v5 COMPLETE, verdict `MATCH_WITHIN_TOLERANCE`, evidence `ds7-smoke-v5`) |
| DS8 | Expert cache + Dynamic Expert Eviction | ✅ (kernel v3 COMPLETE, verdict `ACCEPT_EXPERT_RUNTIME`, evidence `ds8-runtime-v3`) |
| DS9 | Architecture bring-up → first token | ✅ **terminated — `REJECT_NUMERICAL` ACCEPTED (product decision 2026-08-02)** (kernel v13 COMPLETE: state fixed (v9); router cause proven (v10/v11); set-based expert-ID gate adopted (v12); expert-integration audit (v13) proves the sole `moe_out`/`shared_out` p99 failure is bounded BF16-storage-rounded input drift amplified at near-cancellation elements — FP16 execution within gate (p99 0.021–0.023), storage/routing/order/capture clean; no runtime correction exists; sealed 0.05 gate unchanged; p99 magnitudes corpus-provisional until DS5 official traces; evidence `ds9-v13-reject-numerical/` + `DS9_V13_POLICY_DECISION.md`) |
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

## DS9 v9 — state semantics proven correct (focused state-parity run)

Kaggle kernel `nivind/dee-cpp-deepseek-v4-flash-0731-ds9-one-layer` v9
terminated **COMPLETE** with verdict `REJECT_ROUTER`. This run proves the
v6/v8 `REJECT_STATE` was a **harness instrumentation defect, not a model
state defect**: `state_buffers()` aliased the already-fp32 CPU buffers
(`detach().float().cpu()` is identity for fp32), so the warm-replay
`reset_state()` zeroed/`-inf`-filled the aliased snapshots before the gates
compared them — manufacturing the phantom 0.387/0.548 divergence.

- Pinned runtime commit: `aad6c02ab4158ed4af238c36a8344b56d1aff80b`
- Pinned harness SHA256: `0b45e60c02772d1d85ca72d3b2d3addcc0087a9a4d6afaccedd978dc7b389321`
- Evidence: `benchmark_reports/deepseek-v4-flash-0731-t4/ds9-v9-reject-router/`
  (manifest validated; archive SHA256 `5e6047c1…`, manifest SHA256
  `ae69c19a…`).
- **State gates PASS at both steps (0 prefill + 16 decode):** all six buffers
  (`attn_kv_cache`, `compressor_kv_state`, `compressor_score_state`,
  `indexer_kv_cache`, `indexer_compressor_kv_state`,
  `indexer_compressor_score_state`) have exact sentinel / written / untouched
  masks; `boundary_captures_ok` true (structural keys bitwise).
- **Locator ULP evidence:** raw compressor projections drift 1–10 ULP
  (CPU-fp32 vs CUDA-fp32 reduction order — expected), state carry 1–2 ULP;
  inputs/APE/scalars bitwise. `attn_kv_cache` bitwise identical.
- Fixes landed: snapshot clone in `state_buffers()`; `state_mask_analysis`
  `ok` structural-only (value bounds stay in the 0.001 rel `state_agreement`
  gates); `boundary_captures_ok` gates structural keys only.
- **Remaining failures (valid `REJECT_ROUTER` via the exact-gates gate):**
  - step-0 top-6 expert-ID exact gate: CPU-fp32 vs CUDA-bf16 boundary flip;
  - `moe_out` / `shared_out` p99_rel 0.068–0.099 vs the predeclared 0.05
    gate (cosine 0.999997–0.9999999 — small overall error with a heavier
    tail in the integrated FP16 expert path).
- Attention (15 categories), router scores, exact window/compress indices,
  route agreement, and cache correctness all PASS. Candidate CUDA-resident,
  peak VRAM 4.47 GB, `performance_comparable: false`, no model TPS.

## DS9 v10/v11 — router cause proven: exact router, input-driven flip (ordering within set)

Kernels `nivind/dee-cpp-deepseek-v4-flash-0731-ds9-one-layer` v10 (focused
router diagnostic, COMPLETE) and v11 (full-layer rerun, COMPLETE), verdict
`REJECT_ROUTER` with the step-0 expert-ID failure now **fully diagnosed and
proven**.

**v10 — isolation matrix proves the router is exact.**

- Full 256-score 5-stage capture (raw/softplus/sqrt/biased) with SHA256 per
  side, per-stage error stats + fp32 hex, top-10 boundary ranks with
  rank-6/7 margin + IEEE bits, symmetric difference, `torch.topk` tie audit,
  and linearized sensitivity.
- 4-way isolation matrix: `ref_in_ids_cpu_vs_cuda_equal` = True (all 16
  rows), `cand_in_ids_cpu_vs_cuda_equal` = True, `topk_same_scores_...` =
  True, captures faithful — **identical input → identical IDs on CPU and
  CUDA**. The router implementation and top-k semantics are exact.
- Per-token top-6 **SETS are identical at all 16 tokens**; only token 4 has
  an intra-set rank swap (102 ↔ 198, `other_ranks_changed [4,5]`); the
  selection (rank-6/7) margin is robust (0.015).
- ULP trace: `attn_norm_in`/`attn_norm_out` **bitwise**; first divergence at
  `attn_o` (CUDA fp32 accumulation vs CPU fp32, ~1 ULP) surfaced as 1-bf16-ulp
  storage rounding; router input delta `max_abs 0.015625` = 2.0 bf16 steps at
  max magnitude — bounded, non-compounding.
- The v10 run self-labeled `REJECT_UPSTREAM_LAYOUT_OR_STATE` via the invalid
  `max_ulp > 64` heuristic (one bf16 grid step spans 2¹⁶ fp32 ULPs). The v11
  classifier (`bf16_storage_bound` absolute-error discriminator + isolation
  fidelity guard) **reclassifies it** to
  `ROUTER_IMPLEMENTATION_EXACT_INPUT_DRIVEN_FLIP` (`flip_scope
  ORDERING_WITHIN_SET`) — see `ds9-v10-router-diag/ROUTER_RECLASSIFICATION.md`.

**v11 — full-layer rerun with the corrected classifier (runtime unchanged).**

- Pinned runtime commit `02673aa…` (harness `b6bd31d0…`); evidence
  `benchmark_reports/deepseek-v4-flash-0731-t4/ds9-v11-reject-router/`
  (manifest validated `96a99d7f…`, archive SHA256 `c3fd449b…`).
- Step 0 diagnostic: `ROUTER_IMPLEMENTATION_EXACT_INPUT_DRIVEN_FLIP`
  (`ORDERING_WITHIN_SET`, flip token 4, flip reproduced by the input delta
  alone, bf16-bound 2.0 steps, fraction-within-one 0.99998).
- Step 16 (decode): **`NO_FLIP_OBSERVED` — expert IDs exactly matched**;
  state masks, boundary structural keys, window/compress indices exact at
  both steps; 15/15 attention categories pass; router scores p99 ≈ 1e-4.
- **MoE drift is now proven flip-independent**: `moe_out`/`shared_out`
  p99_rel 0.068–0.099 vs the sealed 0.05 gate fail at **step 16 as well**,
  where there is no flip at all. Per the DS9 procedure this opens a separate
  expert-integration audit (FP16-expanded payloads vs CPU-FP32 reference).

Remaining blockers for `ACCEPT_ONE_LAYER`, in order:

1. step-0 exact expert-ID tuple gate — proven an **ordering artifact** of
   bounded bf16 storage rounding (router exact, sets identical); **product
   policy decision 2026-08-02: the exact-ID gate is now SET-based
   (order-insensitive)** — validated in v12;
2. `moe_out`/`shared_out` p99 tail — flip-independent, separate
   expert-integration audit (approved as next focus).

`performance_comparable: false`, no model TPS.

## DS9 v12 — set-based expert-ID gate validated, verdict REJECT_EXPERT_INTEGRATION

Kernel `nivind/dee-cpp-deepseek-v4-flash-0731-ds9-one-layer` v12 (full-layer
rerun, COMPLETE) applies the product-policy decision of 2026-08-02: the exact
expert-ID gate is redefined as the **selected expert SET, order-insensitive**
(`expert_ids_exact` set-based; the ordered tuple is preserved as
`expert_ids_tuple_exact` for evidence). Runtime unchanged.

- **Expert-ID gate now PASSES at both steps**: step 0 `expert_ids_exact`
  True (tuple False — the proven intra-set rank swap at token 4, recorded as
  evidence only); step 16 True (tuple also True, `NO_FLIP_OBSERVED`).
- State masks, boundary structural keys, window/compression indices exact at
  both steps; 15/15 attention categories pass; router diagnosis unchanged
  (`ROUTER_IMPLEMENTATION_EXACT_INPUT_DRIVEN_FLIP` / `ORDERING_WITHIN_SET`
  at step 0).
- **Verdict re-attributed**: `REJECT_EXPERT_INTEGRATION` — exact routing
  passes; the only remaining failures are `moe_out`/`shared_out` p99_rel
  0.068–0.099 vs the sealed 0.05 gate at **both** steps (flip-independent).
- Pinned runtime commit `7cca1086…` (harness `bdb54faa…`); evidence
  `benchmark_reports/deepseek-v4-flash-0731-t4/ds9-v12-reject-expert-integration/`
  (manifest validated `f1269a64…`, archive SHA256 `8a8264e0…`).

Next (user-approved): **expert-integration audit** — DS8 isolated-expert
inputs vs DS9 integrated inputs, route weights, input dtype, accumulation
order, FP4 unpack, FP16 execution, shared/routed combination.

## DS5 — trusted reference traces (BLOCKED — hardware-format, v6 evidence)

The trusted reference is the OFFICIAL inference stack pinned in `official-source/`
(`model.py` + tilelang `kernel.py` + `convert.py` + `generate.py`; transformers 5.x
ships no `deepseek_v4` module, so the official inference code is the reference).
`scripts/deepseek_v4_trace_spec.py` pins identities (config/generation/inference-
config SHA-256, revision) and defines the boundary contract + bounded-capture
policy (bitwise tensor SHA-256, head-of-dim slices, NaN/inf counts).
`kaggle/deepseek-v4-flash-0731/ds5_trace_runtime.py` runs the reference on the
layer-0 subset (embedding + layer 0 + final norm/head; shards 00001/00002/00045
≈ 5.7 GB, size+header verified, resumable download), converts with the official
`convert.py` (fp4, mp=1), builds `Transformer(n_layers=1)` with a post-load
dtype contract (FP4 experts / FP8-BF16 dense), captures bounded traces at every
boundary over prefill + greedy decode, and fails closed with gates. Payload
assembler: `tools/build_ds5_kernel.py`. 16 local tests.

**Remote iteration record (Kaggle `…-ds5-trace`, all preserved):**

- v1: payload modules did not reach the kernel import path (flat imports);
  harness rewritten to the sealed DS9 repo-clone identity pattern.
- v2: `BLOCKED source_identity` — the harness had 11 non-ASCII em-dashes;
  Kaggle's script wrap re-encoded them so `/kaggle/src/script.py` hashed
  differently from the pin. Fixed by making the harness pure ASCII.
- v3: `INVALID_EXPERIMENT` — shard size pins omitted the safetensors header
  overhead (8 + header_size), truncating downloads by 96/172240/400 B;
  official convert failed "incomplete metadata, file not fully covered".
  Fixed sizes + added a fail-closed physical-coverage check.
- v4: download/convert PASS; reference load failed with the tilelang 0.1.8 ×
  apache-tvm-ffi >= 0.1.10 Python 3.12 incompatibility (`setattr(type_cls,
  '__dict__', …)`). v5 added full-traceback capture; v6 pinned
  `apache-tvm-ffi<=0.1.9` (installed first) and cleared the load.
- **v6 terminal state: ERROR on the first dense projection** — official
  `fp8_gemm` requests 82,048 B dynamic shared memory; T4 SM75 caps at
  65,536 B. Full traceback in `kernel.log`; source audit shows `sparse_attn`
  (h=64, d=512) needs ~280 KiB and `sparse_attn` indexer ~74 KiB, so no
  pipelining tweak can make the official attention path SM75-runnable.

**Verdict: hardware-format blocker.** The pinned official reference cannot
execute on the campaign's only GPU (Kaggle dual T4). DS5's contract forbids
substituting a re-tiled kernel as the trusted reference. Unblock options:
(1) a non-T4 (SM80+) reference host, (2) an explicit product decision to fork
and re-tile the official kernels for SM75, or (3) reuse the sealed DS8/DS9
trusted-reference discipline (official tensors + FP32 CPU reference) as the
DS6+ parity basis, recording DS5 as blocked. See
`ds5-v6-hardware-blocker/DS5_BLOCKER_REPORT.md` and `evidence/`.

## DS4 — tokenizer + encoding parity (official implementation pinned)

Official `encoding_dsv4.py` + `tokenizer.json`/`tokenizer_config.json` pinned
(asset SHA-256 verified fail-closed). Freebuff wrapper
`scripts/deepseek_v4_encoding.py` is a pure passthrough to the official
encoder/parser and the official tokenizer (TokenizersBackend, vocab 128000,
BOS=0, EOS=1). 15 golden tests freeze exact token IDs for plain completion,
chat, thinking low/high/max reasoning effort, tool/agent (DSML) messages,
multi-turn, plus parse roundtrips and wrapper-parity checks. No generic Jinja
chat template is used. Tokenizer assets are the clean two-file set
(`tokenizer-assets/`, git-deduped against `official-source/tokenizer.json`).

## DS9 v13 — expert-integration audit: cause proven → REJECT_NUMERICAL

Kernel `nivind/dee-cpp-deepseek-v4-flash-0731-ds9-one-layer` v13 (full pipeline
COMPLETE, `fatal_error: null`, performance non-comparable). Repository pin
`672c2f14`; evidence `ds9-v13-reject-numerical/`; archive
`9a2b8b0d…`; 268 local tests (15 new audit tests).

New device-authentic audit (`scripts/deepseek_v4_expert_audit.py` +
`expert_integration_classify`) decomposes the ONLY remaining failures
(`moe_out` p99 0.074/0.099, `shared_out` p99 0.068/0.082 vs the sealed 0.05
gate) across steps 0 and 16:

**Proven clean (both steps):** capture fidelity (replays bitwise reproduce the
captured outputs on both sides); FP16 storage representability (FP4 grid ×
e8m0 scales and E4M3 → FP16 lossless); accumulation-order sensitivity
(~1e-7); routing-weight delta (≤0.0016); and — decisively — **FP16 kernel
execution on the reference input: p99 0.0228 (step 0) / 0.0211 (step 16) =
WITHIN the 0.05 gate**. FP16 execution precision is NOT the cause.

**Proven cause — INTEGRATED_INPUT_DISTRIBUTION (re-attributed to
REJECT_NUMERICAL):** the candidate's FFN input (`ffn_norm_out`) differs from
the reference's by bounded BF16 storage rounding only (step 0: 3595/65536
elements, 2.0 bf16 steps at max magnitude; step 16: 1.0 bf16 step;
`within_bf16_storage_bound: true`). That bounded delta alone moves the PURE-
FP32 reference over the gate: `ref_input_sensitivity` p99 = 0.0707 (step 0) /
0.1094 (step 16); the dequantized-FP32 CUDA execution matches it exactly.
The tail locator shows the over-gate elements are near-cancellation outputs
of the weighted 6-expert + shared combination (cancellation ratio ~0.002; a
~0.4% input perturbation shifts a near-cancelled sum by more than its own
magnitude).

**Terminal classification:** `REJECT_NUMERICAL` — semantics and integration
are proven correct (every integration/storage/routing/order/capture defect is
excluded by evidence); the sealed 0.05 p99 contract fails at the input-boundary
sensitivity floor. The pre-audit gate chain reported `REJECT_EXPERT_INTEGRATION`;
the audit re-attributes it. No runtime correction in the Phase-11 priority
list applies (all excluded by evidence). Corpus is synthetic (official
hidden-state traces are a DS5 dependency); the causal mechanism is proven
independent of the corpus.

**Product-policy decision requested** — options discussed with the user:
(1) keep the sealed 0.05 gate and accept REJECT_NUMERICAL (no runtime fix
can reduce the cross-backend BF16-storage-rounded input boundary below ~1
bf16 step, which the FP32 combination amplifies past the gate);
(2) re-baseline the expert contract to an input-invariant comparison (candidate
kernels on the reference input pass at p99 ≤ 0.023 — this is what DS8-style
isolated validation measures);
(3) replace the synthetic DS9 corpus with official hidden-state traces (DS5)
and re-measure before deciding.

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
