# Minimal abstraction design (interfaces only — not wired in)

Goal: the generic storage/cache/scheduler layer operates on `ExpertId, byte
size, codec id, location, execution cost, reuse/prediction metadata` and never
on DeepSeek tensor semantics. Names follow repository conventions (`Expert*`,
`*Cache*`, `TensorResolver`, `EngineConfig`); nothing here is implemented in
production by this task. Isolated compilable sketches live in
`dee.cpp/experiments/universality/` (`expert_descriptor.hpp`,
`expert_codec.hpp`, `model_adapter.hpp` + standalone `test_universality.cpp`
built directly with g++, outside CMake).

## 1. Layering

```text
ModelAdapter (per model: dsv4, qwen-next, ornith)
  │  owns: tensor names, config parse, router math, combine/residual/norm
  │  emits: ExpertDescriptor[], RouterPlan hooks, LayerPlan
  ▼
Generic core (model-blind): WeightMmap, ExpertStore, HostPackCache,
  VramCacheManager, AsyncPrefetcher, StageProfiler, Engine scheduler
  │  speaks: ExpertId{layer,expert}, bytes, CodecId, cost, reuse hints
  ▼
Codec plugins (per storage codec: mxfp4-e2m1-e8m0, block-fp8, int8, …)
    decode_tile / storage_bytes / validate / metadata_layout /
    supported_backend
```

Rule: model knowledge flows down at init (adapter → descriptors/codec ids);
runtime calls carry ids and byte ranges up. No `w1/w2/w3`, `ffn/mlp`, layer
counts, or scale formats cross the boundary after init.

## 2. ExpertDescriptor (replaces ExpertView's `[3]` + enum)

```cpp
struct ProjectionDesc {
  std::string role;        // "gate","up","down" — informational only
  uint64_t out_elems, in_elems;   // logical (unpacked) shape
  uint64_t weight_offset, weight_nbytes;  // into the expert record
  uint64_t scale_offset, scale_nbytes;    // 0/0 if codec has no scales
  uint32_t scale_block;                   // e.g. 32 (MXFP4), 128 (block FP8)
};

struct ExpertDescriptor {
  ExpertId id;                 // {layer, expert}
  CodecId codec;               // "mxfp4-e2m1-e8m0", "block-fp8-e4m3", …
  std::vector<ProjectionDesc> projections;  // N, any order — NOT fixed [3]
  uint64_t record_bytes;       // authoritative stride (replaces 13369344)
  std::string activation;      // "silu", "swiglu-clamp10", "gelu", …
  CombineDesc combine;         // {weighted_sum, residual_add, norm}
  std::string norm_formula;    // "rms-plus1" | "rms-plain" | …
  double exec_cost_hint;       // FLOPs or measured µs, for the planner
  std::map<std::string,std::string> model_meta;  // escape hatch, never read by core
};
```

Expressiveness check: current DSV4 expert = 3 `ProjectionDesc`s
(`gate [2048,4096]`, `up [2048,4096]`, `down [4096,2048]`, block 32,
activation `silu-clamp10`, record `13369344`). A hypothetical Qwen-Next expert
with different hidden/inter, fused or reordered projections, or no scales is
the same struct with different rows — no `W1/W3/W2` type is forced anywhere.
Cache/scheduler APIs take `const ExpertDescriptor&` and use `id`,
`record_bytes`, `codec`, `exec_cost_hint` only (proven by the prototype's
`static_assert`-style compile test: a `consume(const CacheKey&)` overload that
cannot name a projection).

## 3. ModelAdapter (replaces the TensorResolver dialect switch + L4 conflation)

```cpp
class ModelAdapter {
 public:
  virtual ~ModelAdapter() = default;
  virtual std::string name() const = 0;              // "dsv4-flash-0731"
  virtual bool parse_config(const uint8_t* json, size_t n,
                            ModelGeometry& out) = 0; // replaces EngineConfig defaults
  virtual std::string expert_weight_name(int l,int e,int proj) const = 0;
  virtual std::string expert_scale_name(int l,int e,int proj) const = 0; // "" if none
  virtual std::string router_weight_name(int l) const = 0;
  virtual RouterDesc router() const = 0;  // {scoring: softmax|sqrtsoftplus|sigmoid(+bias-split,hash),
                                          //  top_k, scale, renorm, clamp}
  virtual std::vector<ExpertDescriptor> describe_layer(int l) const = 0;
  virtual CombineDesc combine() const = 0;  // residual? shared-expert branch? MTP?
};
```

`Dsv4Adapter` preserves every H-entry behavior byte-for-byte (it is the
reference); `QwenNextAdapter` is a second implementation, not a flag on the
first. `engine.cpp:3452`'s `transfer==Fp4 → V4` conflation becomes an explicit
`--model dsv4|qwen-next|ornith` knob selecting the adapter; codec stays an
independent choice (L4/L5 fixes).

## 4. RouterAdapter

Folded into `ModelAdapter::router()` + a free function, since routing is
model math, not systems code:

```cpp
struct RouterDesc {
  std::string scoring;   // "softmax" | "sqrtsoftplus" | "sigmoid"
  int top_k; bool renorm; double scale;       // 1.5 for DSV4, 1.0 Ornith
  bool bias_select_only;                       // DSV4 bias-split
  bool hash_routing;                           // tid2eid path
  double swiglu_limit;                         // 10.0 DSV4, 0.0 Ornith (already a knob)
};
std::vector<ScoredExpert> route_topk(const RouterDesc&, const float* logits, int E);
```

The existing `route_topk_batch` becomes `route_topk(RouterDesc{softmax,…})`;
DSV4's Python router is ported only when the second model needs an in-engine
router (it does not block base decode — real-model mode bypasses via
`moe_forward_*`).

## 5. AttentionBackend (seam only — no implementation in this task)

```cpp
class AttentionBackend {
 public:
  virtual ~AttentionBackend() = default;
  // The engine never calls this today; the seam exists so a future MLA /
  // Gated-DeltaNet / sliding-window backend can be selected per adapter
  // without touching MoE code.
  virtual std::string kind() const = 0;  // "external-hf" (today, always)
};
```

Today's only implementation is `ExternalHfAttention` ("dense stays in
Python"). This documents the omission (§15 of the audit) as architecture
instead of accident, and is the explicit answer to "where would Qwen-Next
linear attention live": outside the MoE server, behind this seam.

## 6. Codec boundary (Phase D)

One codec = one plugin; STQ/IQ2 are NOT implemented here (that is
`research/expert-precision`'s work; this task only reserves their slot):

```cpp
struct TileReq { const uint8_t* packed; const uint8_t* scales;
                 uint64_t out_elems, in_elems; uint32_t scale_block; float* dst; };
class ExpertCodec {
 public:
  virtual ~ExpertCodec() = default;
  virtual CodecId id() const = 0;
  // Pure decode of one (out,in) tile. Returns false on unsupported shape
  // instead of silently mis-decoding (fail-closed, cf. engine.cpp:3510-3512).
  virtual bool decode_tile(const TileReq& req) const = 0;
  virtual uint64_t storage_bytes(uint64_t out, uint64_t in) const = 0;
  virtual bool validate(const ProjectionDesc& p, std::string& err) const = 0;
  virtual std::string metadata_layout() const = 0;  // documents record order
  virtual bool supported_backend(const std::string& backend) const = 0;
  // "cuda-fp16" | "cuda-fp32" | "cpu-f32" — 4-bit weights NEVER imply
  // native FP4 execution; each codec declares what it can actually run.
};
```

Codec table (additive, no behavior change until selected):

| Codec id | `storage_bytes` | `supported_backend` today | Owner |
|---|---|---|---|
| `mxfp4-e2m1-e8m0` | `out*in/2 + out*in/32` | `cuda-fp16` (unpack), `cpu-f32` | migrate existing kernels behind it (L6) |
| `block-fp8-e4m3` | `out*in + blocks` | `cuda-fp16` | new (dense + shared experts already F8) |
| `int8` | `out*in + scales` | `cuda-fp16` | promote test path or drop |
| `stq` / `iq2` | TBD | TBD | `research/expert-precision` — slot reserved, nothing claimed |

Fixes consumed: L1 (enum→registry), L5 (split `F8`; each codec names its own
scale dtype), L6 (table/nibble/block/`in%64` move into the MXFP4 plugin).

## 7. What is NOT abstracted (deliberately)

- Oracle predictor (`oracle.h`, `pt_loader.*`): synthetic-control-path only;
  abstracting it generalizes nothing. Leave alone.
- `trace_alloc`, `StageProfiler`, `HostPackCache`, `VramCacheManager`:
  already generic; wrapping them in interfaces adds seams with no second
  implementation. Do not abstract.
- Tokenizer/chat template: one script per model (`deepseek_v4_encoding.py`
  pattern); a common tokenizer API buys nothing while dense stays in HF.
- MTP/speculative heads: no base-decode caller; abstract only when a second
  model forces the shape (see SECOND_MODEL_STRESS_TEST §6).
