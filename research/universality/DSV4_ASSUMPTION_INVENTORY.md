# DSV4 assumption inventory

Machine-readable companion: `dsv4_assumption_inventory.json` (same entries,
`leak` field distinguishes the two classes below). Verified by direct grep at
base `9c16529`; the headline counts: `e2m1|e8m0` 58 hits in `src/`,
`topk|top_k` 117 (threading, not hardcode), `GATE/UP/DOWN_PROJ` 32,
`record_bytes|record_index` 29, `swiglu_limit` 29, `w1|w2|w3` 18 in
`src/` (oracle `.net.N.weight` + v4 namers), `ffn.experts|mlp.gate` 8,
`shared_expert|mtp.|dspark` 5, `sqrtsoftplus|noaux_tc|tid2eid|route_scale` 0
in C++ (router math is Python-only), `attn|qkv|mla|kv_cache` 0 in
`src/engine.cpp` (the one repo-wide `attn` hit is
`cudaGetDeviceProperties`).

## Harmless constants (DSV4 adapter only — correctly isolated, do not touch)

| # | Assumption | Where | Why harmless |
|---|---|---|---|
| H1 | `OFFICIAL_REPOSITORY/REVISION/48 shards/72317 tensors` | `scripts/deepseek_v4_support.py:27-30` | Ledger validation input, never imported by C++ |
| H2 | `LAYER_PATTERN (layers\|mtp)`, `module/component_for_tensor` maps | `deepseek_v4_support.py:52,110-143` | Python classifier; C++ never sees tensor names except via resolver |
| H3 | `.weight→.scale` pairing, `packed [out,in//2]→scale [out,in//32]`, FP8 block `[128,128]` | `deepseek_v4_support.py:146-231` | Storage-planning math in one script |
| H4 | `ROUTED_PROJECTION_NAMES=(w1,w2,w3)`, 6-tensor expert names, shared/router/dense name tables, `n_layers=43/n_hash=3/compress_ratios/dspark` | `deepseek_v4_support.py:406-424,554-618,673-765` | Name tables; the C++ seam takes resolved views |
| H5 | `sqrtsoftplus/bias-split/route_scale=1.5/tid2eid` router, `clamp10/silu*up/weight-before-w2/shared-F8` MoE, FP4/FP8/RMS/YaRN/sinkhorn math | `deepseek_v4_layer_common.py:58-435`, `deepseek_v4_moe_reference.py:32-110`, `deepseek_v4_expert_reference.py:34-163` | Reference numerics; engine calls them, never reimplements |
| H6 | Tokenizer SHAs, vocab 128000≠129280, chat template, thinking flags | `scripts/deepseek_v4_encoding.py` | One script; C++ has no tokenizer |
| H7 | `ModelConfig` DSV4 defaults, SHA-pinned configs | `deepseek_v4_model.py:288-367`, `deepseek_v4_trace_spec.py:212-247` | Python config dataclass |
| H8 | `mtp.* (layers 40-42)`, `MTP excluded from greedy text` | `weight_mmap.h:106`, `deepseek_v4_support.py:121-128` | Documented exclusion, not an execution branch |
| H9 | `258 transfers/layer x 43 layers` telemetry-pool comment | `profiling.h:427-433` | Comment sizing `kMaxTimingEvents`; code generic |
| H10 | `~12.6 MB/entry`, six regions, `152.8 GiB`, `8 GiB` cache default | `host_pack_cache.h:3-14,127` | Comments/tuning; bytes are opaque at runtime |

## Generic-runtime leaks (DSV4 structure inside shared code — fix per plan)

| # | Assumption | Where | Scope | Fix |
|---|---|---|---|---|
| L1 | `ExpertCodec` has one real value; `get()` forces `DeepSeekFp4E2m1E8m0`; `open()` fails closed on other codec strings | `expert_store.h:21`, `expert_store.cpp:187,325-328,502` | codec→DSV4 | Codec table + enum values (ABSTRACTION_DESIGN §Codec) |
| L2 | `ExpertView weights[3]/scales[3]`; 3-Kind resolve; 6-region record; `p<3` / `[6]` loops in every quantized prefetch path | `expert_store.h:30-31`, `expert_store.cpp:191-198,507-521`, `async_prefetcher.h:36-53`, `async_prefetcher.cpp:228-295,408-419,713-722` | MoE-family→DSV4 | `N`-projection descriptor vectors |
| L3 | `GATE→w1,UP→w3,DOWN→w2` + `I8` packed/`F8` scale dtypes + 1 B/elem check + `hidden=shape[1]*2` probe | `weight_mmap.cpp:320-339,359-361`, `expert_store.cpp:367-375`, `engine.cpp:3480-3504` | codec→DSV4 | Per-model name table + codec width fn |
| L4 | `transfer_dtype==Fp4E2m1` implies `DEEPSEEK_V4` resolver dialect | `engine.cpp:3452` | DSV4 (conflation) | Explicit `model` knob; one-line + plumbing |
| L5 | `DType::F8` conflates `F8_E4M3/E5M2/E8M0` | `weight_mmap.cpp:21-24` | codec | Split enum; mechanical |
| L6 | e2m1 table, `2^(bits-127)`, nibble order, block 32, `in%64==0` in shared decode (host + CUDA) | `weight_mmap.cpp:76-113`, `cuda_convert.cu:208-288`, `cuda_convert.h:32-42` | codec→DSV4 | Codec ops table; new codec = new entry |
| L7 | Built-in router: `...mlp.gate.weight` name + softmax-only math | `engine.cpp:2231-2427` | ORNITH-shaped name; family math | Scoring enum + name table (bypassed in real-model mode today) |
| L8 | `+1`-convention RMSNorm only; no DeepSeek plain norm | `rmsnorm_cuda.h:59-77` | Qwen-family formula | `norm_formula` flag |
| L9 | Mean-combine + unit-RMS mock in `forward_layer`; no residual/LN | `engine.cpp:3307-3346,4086-4112` | mock harness | Replace with `y=x+Σw·E(x); norm` per model (keep `moe_forward_*` seam) |
| L10 | `EngineConfig` defaults (`topk=8/layers=40/hidden=2048/inter=256`) + Oracle `256/256` hardcode; no `config.json` parsing | `engine.h:84-123`, `engine.cpp:3527` | ORNITH-shaped defaults | Config-file loader (see MIGRATION_PLAN M4) |
| L11 | Shared expert + MTP silently absent from execution (helper exists, unwired) | `weight_mmap.cpp:342-344` vs `engine.cpp` (0 shared/mtp exec hits) | DSV4 omission | Dense-expert branch + residency policy |

## Reading guide

- H-entries: changing them generalizes nothing; they are the DSV4 adapter
  and must be preserved as the reference during migration.
- L-entries: ordered roughly by leverage-per-risk; L4/L5 are one-line-class
  fixes, L1–L3/L6 are the codec-boundary work, L7–L11 are frontend work that
  can wait for a real second model.
