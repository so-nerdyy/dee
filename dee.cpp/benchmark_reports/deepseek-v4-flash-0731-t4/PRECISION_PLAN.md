# PRECISION_PLAN — DeepSeek-V4-Flash-0731 on SM75 (T4)

T4 (SM75) has no native FP8 or FP4 tensor cores. The official checkpoint
stores experts in packed FP4 and dense weights in FP8/BF16/FP32. This plan
keeps **source / cache storage / execution / accumulation / output** precision
separate and measured per family.

## Observed official storage (from shard headers, pinned revision)

| Family | Stored dtype | Element layout | Scale |
|---|---|---|---|
| Routed expert `w1/w2/w3.weight` | `I8` | 2 × e2m1fn per byte (`[out, in//2]`) | `F8_E8M0` `[out, in//32]` |
| Expert scales | `F8_E8M0` | 1 byte | — |
| Dense `wq_b/wo_a/...` | `F8_E4M3` | 1 byte | `F8_E8M0` (block 128×128) |
| `wo_b`, norms, gates | `BF16`/`F32` | — | — |
| Hash `tid2eid` | `I64` | — | — |

## Family A — official reference (trusted outputs)

- Run the official `inference/model.py` on hardware that executes the
  checkpoint natively (H100/B200 class or CPU FP32 reference).
- Dtype policy: `dtype=fp8`, `scale_fmt=ue8m0`, `expert_dtype=fp4`,
  matching `inference/config.json`.
- Outputs (embed, layer boundaries, index scores, expert IDs/weights, hidden
  states, logits, tokens) become the pinned reference traces for DS5.

## Family B — FP4 storage, FP16 accumulation (primary T4 path)

- Keep experts packed `I8` in the checkpoint and the bounded cache.
- On demand: unpack e2m1fn pairs → FP16 values; multiply by `F8_E8M0` scale.
- Execute `w1/w3/w2` GEMV in FP16 with FP32 accumulation on SM75.
- Fuse unpack into the GEMV epilogue where possible; never materialize a
  permanent expanded expert copy.
- Validated against Family A with predeclared per-tensor tolerances.

## Family C — offline INT8 expert conversion

- Convert active experts to INT8 with per-block scales (FP8-style block 128×128
  or the official 32-element K-block).
- Use T4 INT8 tensor cores when shapes/batching permit.
- Validate quality and route/token behavior against Family A.
- **Never labeled bitwise exact.**

## Family D — FP16 reference subset

- Expand only the currently active expert working set to FP16 for correctness
  bring-up and early layers.
- Full checkpoint expansion would be **626.88 GB FP16** — prohibited.

## Explicit separation contract

Every benchmark record must state, per tensor family:

- source checkpoint precision (I8-packed-FP4 / F8_E4M3 / BF16 / F32 / I64);
- cache storage precision (raw packed / INT8 / FP16 / prepacked);
- execution precision (FP16 / INT8 / FP32);
- accumulation precision (FP32 / FP16);
- output precision (BF16 / FP16).

## T4 feasibility summary (from ledger, exact bytes)

| Metric | Value | Source |
|---|---|---|
| One packed expert (w1+w2+w3 + scales) | 12.75 MiB | ledger rows for `layers.6.ffn.experts.0.*` |
| One FP16-expanded expert | 49.5 MiB | ledger `expanded_fp16_bytes` |
| One INT8-expanded expert | 24.75 MiB | ledger `expanded_int8_bytes` |
| Top-6 active experts / token / layer | 297.0 MiB FP16, 148.5 MiB INT8 | 6 × ledger expert rows |
| Persistent dense (excl. experts/DSpark) | 8.85 GiB compressed | ledger components |
