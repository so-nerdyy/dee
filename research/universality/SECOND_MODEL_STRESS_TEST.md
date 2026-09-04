# Second-model stress test (paper design — not a compatibility claim)

Subject: a Qwen-MoE/Next-style architecture (Qwen3-Next-class: hybrid
linear/full attention, MoE with its own geometry, non-MXFP4 checkpoint).
Status: **no Qwen-Next code, config, or docs exist in this repo** (glob
`*qwen*/*mtp*/*nextn*` under `dee.cpp/` → 0 hits; only `qwen_*rms_norm*`
kernel names and a ledger one-liner). What IS local: the Ornith
(Qwen3.5-MoE-derived) dialect already in the C++ resolver
(`model.language_model.layers.%d.mlp.experts.%d.{gate,up,down}_proj.weight`,
`weight_mmap.cpp:304-315`) and `scripts/ornith_support.py`. This stress test
therefore grounds every "Qwen-side" row in either (a) repo-observed Ornith
evidence, or (b) an explicit **VERIFY** item to check against the candidate
checkpoint's `config.json` on the target host. Nothing here claims support;
code does not exist yet.

Question asked throughout: "if we tried to add this model tomorrow, which dee
interfaces survive and which break?"

## 1. Checkpoint format — SURVIVES

- `WeightMmap` (mmap + safetensors header + `lookup`) is format-generic; Qwen
  checkpoints are standard safetensors with an index JSON like DSV4's.
- VERIFY: shard count, tensor count, total bytes, revision pin — the same
  `CHECKPOINT_DOWNLOAD_PLAN.md` procedure re-runs with new constants (H1-class
  work, one script).

## 2. Expert geometry — SURVIVES via descriptor, BREAKS today

- Contrast (VERIFY all four against `config.json`): different `hidden_size`,
  `moe_intermediate_size`, expert count, layer count than 4096/2048/256/43;
  projections named `gate_proj/up_proj/down_proj` (repo-observed Ornith form)
  instead of `w1/w3/w2`; scales absent or differently suffixed/blocked.
- Today breaks: `ExpertView[3]` + forced MXFP4 codec (L1/L2), `GATE→w1…` namer
  (L3), `hidden=shape[1]*2` probe (`engine.cpp:3480-3504`), `record_bytes`
  math (`(blob*17+31)/32`).
- With `ExpertDescriptor` + `ModelAdapter`: the Qwen adapter emits N
  `ProjectionDesc` rows with its own names/shapes/blocks; core code unchanged.
  The 13 MiB-record assumption evaporates because `record_bytes` is per-model
  data. **This is the highest-leverage abstraction in the design.**

## 3. Router — SURVIVES via RouterDesc, BREAKS today (bypass available)

- Contrast: Qwen MoE routers are typically `softmax → top-k → renorm` over
  `model.layers.*.mlp.gate` (Ornith form already in `route_topk_batch`);
  Next-style models may add bias or grouped variants (VERIFY scoring fields).
  Either way there is no `sqrtsoftplus/bias-split/tid2eid/route_scale=1.5`
  outside DSV4's Python reference.
- Today: built-in router speaks only the Ornith name + softmax math (L7) —
  adequate for a Qwen-style router name, wrong math for DSV4's in-engine, and
  no scoring switch for anything else.
- With `RouterDesc{scoring,top_k,scale,renorm,…}`: one struct covers
  softmax (Qwen/Ornith), sqrtsoftplus+1.5 (DSV4), and sigmoid variants.
- De-risk: real-model mode bypasses the built-in router entirely
  (`moe_forward_*` + external router), so base decode for model #2 does not
  wait on router generalization.

## 4. Top-k — SURVIVES (already runtime)

- `topk` threads through `EngineConfig`, Oracle bypass, prefetch priority,
  and combine as a value (`engine.cpp` 117 `topk` hits are threading, per
  grep). VERIFY the model's `num_experts_per_tok`; set the knob. No break.

## 5. Codec — BREAKS today, isolated by design

- Contrast: Qwen checkpoints are typically BF16/FP8, not MXFP4
  (VERIFY `quantization_config` + expert dtypes). The current store fails
  closed on anything but `deepseek-fp4-e2m1-e8m0` (L1) and `DType::F8`
  conflates scale formats (L5).
- With the codec registry: Qwen experts arrive as `bf16` (native, no decode
  kernel) or `block-fp8-e4m3` (new plugin). The MXFP4 plugin keeps DSV4
  bit-exact; no shared kernel is touched. **No STQ/IQ2 work is implied or
  required for model #2.**

## 6. Attention / state — OUT OF SCOPE on both sides (seam holds)

- DSV4 side: MLA + indexer + YaRN + sliding window live in Python/HF; C++
  holds zero attention (§15 of audit).
- Qwen-Next side: hybrid linear (Gated DeltaNet-class) + full attention with
  its own state layout (VERIFY per-layer pattern and state dims) — likewise
  stays in the dense frontend behind `AttentionBackend::kind()="external-hf"`.
- The seam survives precisely because neither model puts attention in the
  MoE server. A future in-engine attention project is independent of model
  #2 and explicitly out of this plan.

## 7. Residual structure — NEEDS the per-model flag (small break)

- DSV4: `y = x + shared(x) + Σ w·E(x)`, plain RMSNorm. Ornith/Qwen: `+1`
  RMSNorm convention (already the only CUDA kernel, L8), different shared
  handling (DSV4 shared is F8-always-on; Ornith singular BF16 via HF).
- With `CombineDesc` + `norm_formula`: adapter-selected, no core fork. The
  mock mean-combine (L9) must be replaced by the real formula for ANY real
  checkpoint — model #2 does not add new work here, it shares DSV4's.

## 8. Special mechanisms — DEFERRED on both sides

- DSV4: MTP/DSpark excluded from greedy text (H8); hash/tid2eid layers,
  compress ratios, sinkhorn routing stay in Python.
- Qwen-Next: its own speculative/MTP-class heads and hybrid-attention state
  (VERIFY) — same treatment: exclude from base decode, add behind new
  namespaces only when speculative decode is wanted.
- Neither blocks the other; the plan sequences them after base decode works.

## Verdict of the stress test

Survive: mmap/lookup, both byte caches, eviction, prefetch control plane,
telemetry, scheduling, SwiGLU CUDA kernels, top-k threading, `swiglu_limit`
knob pattern, `moe_forward_*` bypass seam, `TensorResolver` dialect-switch
pattern. Break (all isolated, none structural): codec gate + `[3]`/`[6]`
descriptor shape, name tables, `*2` width probe, codec↔model conflation,
`F8` conflation, router scoring switch, norm formula flag, mock combine,
config defaults. The proposed abstractions express both the current DSV4
expert and the Qwen-style expert (proven by the Phase F prototypes, which
instantiate both); attention never enters the picture. Estimated shape of the
work: adapter + codec-registry + descriptor plumbing, with zero changes to
caches, scheduler, or telemetry — detailed in MIGRATION_PLAN.md.
