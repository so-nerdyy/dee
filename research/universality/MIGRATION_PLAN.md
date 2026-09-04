# Migration plan: second MoE without jeopardizing DSV4 (Phase H)

Principles: extract interfaces only when touched by real work; the DSV4
adapter stays the bit-exact reference; the second model arrives behind the
same interfaces; abstractions that fail the second-model test get fixed
then — never a "rewrite everything generic first" plan. Performance rule:
no step may change a hot path for DSV4 (kernels, residency layout, prefetch
order) until the second model proves the new seam is neutral; every step
keeps `exact_match` green on the DSV4 contract.

## M0 — Decouple model select from codec (do first; trivial)

- Fix L4: add explicit `--model dsv4|ornith` knob selecting
  `TensorResolver::Model` in `engine.cpp:3452` instead of inferring it from
  `transfer_dtype==Fp4E2m1`. Fix L5 alongside: split `DType::F8` into
  `F8_E4M3/E5M2/E8M0` (`weight_mmap.cpp:21-24`).
- Files: `engine.cpp`, `weight_mmap.{h,cpp}`, `main.cpp` (+1 CLI flag),
  `pydee` bindings passthrough. Risk: low (init-path only). Benefit:
  unblocks every later step; kills the codec↔model conflation. Required
  before model #2: yes. Perf impact: none (init only).

## M1 — Codec registry behind the existing enum (next real codec touch)

- Fix L1/L6: introduce `ExpertCodec` registry (ABSTRACTION_DESIGN §6) with
  `mxfp4-e2m1-e8m0` as the first plugin, migrating the existing
  `weight_mmap`/`cuda_convert` decode verbatim behind
  `decode_tile/storage_bytes/validate/metadata_layout/supported_backend`.
  Reserve `stq`/`iq2` ids without implementing (owned by
  research/expert-precision).
- Files: new `include/dee/expert_codec.h` (+ maybe `src/expert_codec.cpp`),
  edits in `expert_store.{h,cpp}`, `weight_mmap.{h,cpp}`,
  `cuda_convert.{h,cu}`, `async_prefetcher.{h,cpp}` call sites. Risk: medium
  (touches staging math; mitigate by byte-identical migration + parity
  tests). Benefit: new codecs become additive. Required: yes for any
  non-MXFP4 model. Perf: neutral if migration is verbatim (verify with
  existing parity tests).

## M2 — N-projection ExpertDescriptor (with M1 or when prefetch is touched)

- Fix L2/L3: replace `weights[3]/scales[3]`, `p<3`/`[6]` loops, and the
  `GATE→w1…` + `*2` probe with `ExpertDescriptor`/`ProjectionDesc` +
  per-model name tables (ABSTRACTION_DESIGN §§2–3). `record_bytes` becomes
  per-model data; metadata gains projection arrays.
- Files: `expert_store.{h,cpp}`, `async_prefetcher.{h,cpp}`, `engine.h`
  (`Fp4Proj[3]`), `engine.cpp` staging/configure paths, DEE4 metadata
  writer/reader. Risk: medium-high (widest touch). Benefit: the single
  highest-leverage change (stress-test §2). Required: yes. Perf: keep the
  `[gate,up,down]` fast path layout for DSV4 bit-identical; measure
  prefetch micro-benchmarks before/after.

## M3 — Config-file loader + RouterDesc + norm flag (frontend pass)

- Fix L7/L8/L10: parse the checkpoint `config.json` into an extended
  `EngineConfig` (hidden/inter/layers/experts/topk/norm_formula/router
  name/scoring); add `RouterDesc` scoring switch to `route_topk_batch`;
  add `rms-plain` norm formula next to `rms-plus1`. Replace mock combine
  (L9) with `y=x+Σw·E(x)` + selected norm on the paths that serve real
  checkpoints (keep `moe_forward_*` seam untouched).
- Files: new config loader, `engine.h` config, `engine.cpp` router/combine,
  `rmsnorm_cuda.*`, Python `ModelConfig` sibling dataclass. Risk: medium
  (combine change must stay exact — gate on contract tests). Benefit: C++
  stops needing Ornith-shaped defaults. Required: yes for in-engine
  correctness on model #2; no for bypass-mode bring-up. Perf: combine change
  is formula, not throughput architecture; re-profile scenarios A–G after.

## M4 — Second-model adapter + bypass-mode bring-up (the payoff)

- Add `QwenNextAdapter` (or chosen model) implementing `ModelAdapter`;
  bring up base decode in bypass mode (`moe_forward_*`, external router,
  dense in HF) reusing the Ornith `AutoConfig→EngineConfig` pattern. Shared
  expert (L11) wired only if the model needs it for parity; MTP stays
  excluded (H8 pattern).
- Files: new adapter (Python + small C++ namer), ledger/contract scripts
  mirroring `deepseek_v4_*` minimally, no core changes expected — and any
  core change forced here is precisely "fixing abstractions that fail the
  second-model test" (do it, then re-run DSV4 contract).
- Risk: low to DSV4 (additive; reference untouched). Benefit: first real
  plug-and-play datapoint. Required: this IS model #2. Perf: none to DSV4
  (separate code path selection at init).

## Explicitly NOT in this plan

- STQ/IQ2 implementation (research/expert-precision owns it).
- In-engine attention/MLA (independent hard project; both models stay
  external-HF per `AttentionBackend` seam).
- Oracle generalization (synthetic-path only; leave alone).
- Tokenizer unification (one script per model).
- Cache/scheduler/telemetry rewrites (already generic; re-tuning is ops,
  not engineering).
