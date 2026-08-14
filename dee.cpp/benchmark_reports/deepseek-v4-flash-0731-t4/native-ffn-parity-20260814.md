# Native FP4 FFN — parity + compile-verify (2026-08-14)

## Scope

Wire `pydee.Engine.moe_forward_experts` (native FP4 e2m1fn decode on the
transfer stream + cuBLAS SwiGLU) into the DeepSeek-V4-Flash-0731 harness FFN,
replacing the host-side torch dequant + matmul for the top-K **routed**
experts. Router + shared expert stay on the DS8 torch path.

## Code

- `scripts/deepseek_v4_layer_candidate.py` — `DeepseekV4NativeFfn`
  (`moe_forward_experts` per token, routing weight applied after w2 — exact
  because weight-before-w2 commutes with the linear down projection) and
  `make_native_candidate_layer`.
- `pydee/pydee.cpp` + `pydee/__init__.py` — expose `EngineConfig.swiglu_limit`
  (DeepSeek-V4 clamp = 10.0) and `prepack_quantized_source`;
  `configure(..., swiglu_limit=...)`.
- `scripts/deepseek_v4_native_ffn_parity.py` — self-contained parity test
  (mini real-shaped FP4 shard, `layers.{l}.ffn.experts.{e}.w*` naming).

Commits: `bc1b4fa` → `86179d5` → `75e9b20` on
`freebuff/deepseek-v4-flash-0731-t4`.

## Kaggle T4 verification (sm_75, CUDA 12.8)

Kernel `nivind/dee-cpp-dsv4-native-ffn-parity` v3 (`75e9b20`), single T4:

| Check | Result |
|---|---|
| `dee_cli` + `libdee_core.a` build (DEE_CUDA=ON) | PASS |
| FP4 host decode test | 17/17 |
| FP4 on-device parity | bit-exact |
| FP4 end-to-end expert (native vs FP32 ref) | rel RMSE 0.000411 |
| pydee pybind11 build + import | PASS |
| **Native FFN vs DS8 FP32 MoE reference** | **PASS** |
| — max_abs_err | 19.18 (abs; large-magnitude synthetic weights) |
| — relative RMSE | **0.000651** |
| — cosine similarity | **0.999999762** |
| — finite | True |
| — routed expert IDs | match reference exactly |

## Status

Native FP4 routed-expert path is compile-verified and numerically validated
end-to-end through the pydee binding + harness FFN class on the T4.

**Not yet done:** running the native FFN against the real 43-layer / 256-expert
checkpoint to produce a generated-text tok/s. That requires the 37 GiB of
staged shards resident on a Kaggle run (~1 h) and the dense path (embed /
sparse-attention / RMSNorm / LM head) still on torch. This remains synthetic
until the full tokenizer → transformer → streamed MoE → LM-head loop runs on
the official checkpoint.
