# Ornith M5 External Model Review Audit

Date: 2026-07-30

This note records which outside-model suggestions entered the measured
campaign. It is not benchmark evidence. Every performance or exactness claim
still requires the repository's same-session paired A/B, trace, path, VRAM,
thermal, manifest, and archive gates.

## Model identity and access

- **GLM-5.2:** reviewed in the official Z.ai web application. The signed-in
  model selector explicitly displayed `GLM-5.2`.
- **Kimi K2.5:** the official API documentation still lists the model, but the
  public chat surface used by the signed-in account exposed the current
  successor, K2.6 Instant. The official developer console required a separate
  login and no `MOONSHOT_API_KEY` or `KIMI_API_KEY` was present. The collected
  Kimi review is therefore labeled **K2.6 successor review**, not K2.5.

No result from a substitute model is represented as a K2.5 result.

## Suggestions adopted as bounded experiments

| Review idea | Campaign action | Acceptance rule |
|---|---|---|
| Isolate regular and gated normalization paths | M5G runs disjoint regular-only and gated-only arms | Each arm independently needs bitwise traces, at least 1% paired median gain, bounded VRAM, and clean path evidence |
| Test the fused recurrent implementation before assuming a win | M5H switches only the 30 cached single-token recurrent calls to FLA 0.5.2 | Same-runtime eager fallback control, 3 balanced pairs, all trace categories bitwise exact |
| Test graph replay on a shape-stable subset first | M5I captures only the 30 batch-one cached-decode linear-attention modules | Eager prefill and accepted pointer expert path remain fixed; 30 graphs, 90 replays/generation, exact cache copy-back, full traces, and less than 8 GiB/GPU |
| Treat graph staging overhead as measurable | M5I counts hidden and live-cache input/output copies per run | Candidate must still deliver at least 1% paired and independent median gain |

## Suggestions rejected or corrected before implementation

1. **Published TPS forecasts are not evidence.** The reviews proposed values
   ranging from roughly 12 TPS to 30 TPS. None is used as a result or
   denominator.

2. **The GLM ceiling arithmetic double-counted nested timing categories.**
   `attention/linear-attention` and `MoE` are parent categories; router,
   routed experts, and shared experts are MoE subcomponents. Adding every
   number as disjoint work cannot establish a 15.8-TPS hardware ceiling.

3. **Dynamic device data is not inherently ungraphable.** A graph may consume
   a static-address device pointer table whose contents are produced by prior
   device work. The current expert implementation still has host-side pointer
   construction, so whole-step capture is not yet justified, but dynamic
   top-8 routing alone does not make it impossible.

4. **Cross-token double buffering is invalid here.** Greedy token `t+1`
   depends on token `t` logits. A boundary activation also cannot be copied
   before its producer layer finishes. Only dependency-preserving overlap may
   be tested.

5. **Top-k is eight, not four.** Persistent-expert estimates based on four
   experts do not describe this run.

6. **FLA's recurrent state is not smaller because the prompt is short.** The
   tested recurrent call has sequence length one, but the fixed state matrix
   shape is model-defined. M5H measures the actual path.

7. **Three balanced pairs are not a claim of `p < 0.05`.** They are a bounded
   stability gate supplemented by exact traces, thermal/clock checks, and
   independent medians.

8. **`cudaGraphExecKernelNodeSetParams` is not the first probe.** The PyTorch
   graph wrapper does not expose stable internal cuBLAS node handles. Static
   staging is the simpler contract-preserving experiment; only measured copy
   erosion can justify a lower-level rewrite.

9. **Graph capture is not assumed bitwise safe.** Warm-up handles lazy cuBLAS
   allocation, but eager and captured execution still pass only if the
   repository's intermediate and final bitwise comparisons pass.

## Ranked evidence-driven order

1. Seal M5G.
2. Seal M5H.
3. Seal M5I.
4. Stack only independently accepted, compatible candidates and remeasure.
5. Reprofile the newest accepted stack before selecting any router/shared
   expert or lower-level whole-step graph follow-up.

Official references consulted:

- <https://www.kimi.com/help/getting-started/agentic-chat>
- <https://www.kimi.com/ai-models/kimi-k2-5>
- <https://platform.kimi.com/docs/models>
- <https://github.com/fla-org/flash-linear-attention>
- <https://github.com/Dao-AILab/causal-conv1d>
- <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py>
