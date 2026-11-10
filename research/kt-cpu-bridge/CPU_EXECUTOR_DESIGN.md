# CPU Executor Design — thin adapter between dee and KTransformers-style CPU MoE

Location: `dee.cpp/experiments/kt_cpu_bridge/`
Status: isolated prototype. NOT production integration. No KTransformers types
leak past the adapter boundary. No live-scheduling change.

## 1. Goals / non-goals

Goals:

- Execute ONE routed expert on CPU from a BOUNDED record supplied by dee
  (`HostPackCache` entry / `WeightMmap` views), never requiring ownership of
  the whole model.
- Keep dee model-neutral: the interface is `(layer, expert_id,
  packed_weight_view, hidden, routing_weight)` conceptually; DeepSeek-V4
  specifics (E2M1/E8M0/block-32/clamp-10) live in the V4 codec, not the
  interface.
- Preserve exact dee semantics (clamp asymmetry, nibble order, scale law,
  routing-weight algebra) while allowing a KT-faithful execution path
  (bf16-roundtrip GEMM + late weighting) for parity measurement.
- Prepare the async + cost-model hooks the future planner needs, without
  binding them yet.

Non-goals (explicitly OUT):

- No `TP_MOE` import, no `CPUInfer` singleton binding, no SGLang dependency,
  no GGUF path, no full-pool residency, no thread-pool ownership, no NUMA
  sharding, no AMX intrinsics in Phase 1.
- No change to `Engine::moe_forward_*`, `route_topk`, `HostPackCache`,
  `VramCacheManager`, or any `freebuff/*` benchmark path.
- No full-model TPS claim.

## 2. Layout

```
dee.cpp/experiments/kt_cpu_bridge/
  include/kt_bridge/
    packed_expert_view.hpp   # borrowed views into dee's bounded cache (no own)
    cpu_executor.hpp         # CpuExpertExecutor interface + config + errors
    kt_cpu_executor.hpp      # KtCpuExecutor (KT-faithful: bf16-roundtrip path)
    reference_cpu_executor.hpp # ReferenceCpuExecutor (exact fp32, ISA-neutral)
  src/
    kt_cpu_executor.cpp      # e8m0 codec + dequant + swiglu + GEMM (portable)
    reference_cpu_executor.cpp
  python/kt_cpu_bridge/
    __init__.py
    codec.py                 # E2M1/E8M0 codec (mirrors weight_mmap.h exactly)
    reference.py             # fp32 expert_forward + KT-emulated bf16 path
    cost_model.py            # microbench API + q* simulator (Phase E)
  tests/
    test_kt_bridge_codec.py       # nibble order / e8m0 / dequant / clamp
    test_kt_bridge_correctness.py # ref vs KT-emulated: metrics + determinism
    test_kt_bridge_cost_model.py  # q* enumeration, edge cases, no hardcode
    test_kt_bridge_executor_cpp.py# compiles/runs C++ reference (if toolchain)
  bench/
    bench_cpu_expert.py      # microbench driver -> JSON for cost_model
  CMakeLists.txt             # standalone, NOT added to dee.cpp root build
  README.md
```

Root `dee.cpp/CMakeLists.txt` is UNCHANGED — the bridge builds standalone
(`cmake -S dee.cpp/experiments/kt_cpu_bridge -B build-kt-bridge`) so the
campaign build can never pick it up by accident.

## 3. Interface (conceptual, language-neutral)

```
CpuExpertExecutor::execute(
    layer: int,               # logical model layer (opaque to kernel)
    expert_id: int,           # routed expert id (opaque to kernel)
    packed: PackedExpertView, # borrowed: 3x packed I8 + 3x e8m0 scales + shapes
    hidden: float[hidden_dim],# borrowed input row (fp32; adapter converts)
    routing_weight: float,    # scalar; adapter never renormalizes
    config: ExecuteConfig,    # {swiglu_limit=10.0, swiglu_alpha=0(reject!=0)}
) -> float[hidden_dim]        # caller-owned output (routing weight INCLUDED)
```

C++ spelling (`include/kt_bridge/cpu_executor.hpp`):

```cpp
struct PackedProjection { const uint8_t* packed; const uint8_t* scale;
                           size_t out, in; size_t packed_nbytes, scale_nbytes; };
struct PackedExpertView { PackedProjection gate, up, down; /*w1,w3,w2*/ };
struct ExecuteConfig { float swiglu_limit = 10.0f; float swiglu_alpha = 0.0f; };
struct ExecuteStats { double dequant_ms=0, gemm_ms=0, total_ms=0; };
class CpuExpertExecutor {
 public:
  virtual ~CpuExpertExecutor() = default;
  virtual bool execute(int layer, int expert_id, const PackedExpertView&,
                       const float* hidden, size_t hidden_dim, float routing_weight,
                       const ExecuteConfig&, float* out, size_t out_dim,
                       ExecuteStats* = nullptr) = 0;
  virtual const char* name() const = 0;
};
```

Rules (enforced, fail-closed):

- Borrowed pointers valid for the call only; executor copies what it keeps.
- `gate/up: [inter, hidden]`, `down: [hidden, inter]`; packed
  `[out, in//2]`, scale `[out, in//32]`; `in % 32 == 0`, `packed_nbytes ==
  out*in/2`, `scale_nbytes == out*in/32`, else `kShape`.
- `scale byte 0xFF -> kScale` (KT `+inf` vs dee clamp divergence — never in
  real ckpts, fail rather than silently diverge).
- `swiglu_alpha != 0 -> kConfig` (MiniMax path out of scope).
- `hidden_dim/out_dim` mismatch -> `kShape`; null/empty -> `kNull`; non-finite
  input -> `kNonFinite` (checked BEFORE compute).
- Output INCLUDES the routing weight (dee placement: weight on the
  intermediate before `w2`; KT late-weighting proven algebraically equivalent
  in tests). Skipped experts are the CALLER's concern (contribute 0, no
  renormalization — matching KT `should_skip`).
- Deterministic: same inputs -> bitwise identical outputs (no threading in
  Phase 1; proven by test).

## 4. Implementations

### ReferenceCpuExecutor (exact, ISA-neutral, default)

Portable C++ (and mirrored Python `reference.py::expert_forward`):

1. Validate view + config + dims.
2. Dequantize each projection on the fly per output row:
   `w[o,i] = FP4_TABLE[nibble] * 2^(scale[o,i//32]-127)` (dee `weight_mmap`
   semantics exactly; LUT shared with `fp4_e2m1_table()`).
3. `gate = x@W1^T`, `up = x@W3^T` (fp32 row-major dot).
4. Clamp: `gate = min(gate, limit)`, `up = clamp(up, ±limit)`; `limit<=0`
   disables (Ornith compat).
5. `h = silu(gate)*up; h *= routing_weight; y = h@W2^T` (fp32).

No bf16 anywhere. This is the correctness arbiter and the AVX2-equivalent
baseline (AVX2 KT path has the same contract with `fmadd_ps`).

### KTransformersCpuExecutor (KT-faithful)

Same validation + codec, then emulates KT's numeric path so Phase D can
quantify the bf16-boundary delta WITHOUT needing AVX512 hardware:

1. Weights decoded through the same E2M1 LUT (bit-exact).
2. Activations rounded through bf16 at KT's boundaries: input row -> bf16,
   gate/up outputs -> bf16, SwiGLU in fp32, down-input -> bf16, final merge
   fp32 -> bf16 -> fp32-out (Python `reference.py::kt_emulated_forward`
   implements this with explicit `float32->bf16->float32` round-trips;
   C++ `kt_cpu_executor.cpp` uses the same order with portable arithmetic).
3. Routing weight applied AFTER `w2` in fp32 (KT placement), proven equivalent
   to dee placement by test (up to the bf16 rounding above).
4. Scale path mirrors KT: `ue8m0 -> (u8<<7).view(bf16) -> fp32 -> validate
   positive-pow2 -> compact-e8 -> expand (e<<23)`; C++ keeps the compacted
   `e8` form in its per-call blob to prove cacheability.

Future AMX port (NOT in Phase 1, named hooks only): replace step 2's portable
dots with `mxfp4_to_bf16_32[_natural]` + `_mm512_dpbf16_ps` + 4-wide reduce +
4×4 tile (`fp4_mat_mat_kgroup`), hoist `permute_activation_group` +
`expand_e8_scales`, use `fast_memcpy`. Interface unchanged.

## 5. Memory ownership

```
SSD (safetensors, mmap)  -- owned by dee WeightMmap (source of truth)
  -> HostPackCache entry -- owned by dee engine (bounded LRU, ~12.6 MB/entry)
    -> PackedExpertView  -- BORROWED by adapter for one execute() call
      -> KtExpertBlob    -- OWNED by KtCpuExecutor per call (flat b+e8,
                            required_size = n*k/2+n*k/gs, 64 B aligned,
                            memcpy-evictable; may be cached in HostPackCache
                            as compacted-e8 to skip torch conversion)
        -> out[hidden]   -- OWNED by caller
```

Adapter retains NOTHING across calls in Phase 1. No threads, no NUMA
allocation, no pinned buffers, no global state. `discard_source_pages`
composes (mmap source droppable after the private copy).

## 6. Async (prepared, not bound)

Phase-1 `execute()` is synchronous on the caller thread. The interface
reserves the KT overlap pattern for later without changing callers:

```cpp
// future (stub only, NOT implemented in Phase 1):
// submit(layer, expert, view, hidden, w, config, out, token) -> ticket
// sync_with_stream(stream, allow_pending)  // CUDA-ordered drain
```

Rules for the future binding (from audit §8-§10): caller-owned buffers must
outlive the drain; `allow_pending=1` iff deferred experts exist; NPU path must
NOT use stream-submit (host-unobservable callback hazard); CPU-only build has
no stream methods; per-engine pool, never the KT global singleton; 2-slot
layer ring assumption must be removed (dee needs arbitrary interleaving).

## 7. Scheduler cost model (prepared, not wired)

`python/kt_cpu_bridge/cost_model.py` (Phase E) exposes the microbench API the
future planner needs WITHOUT changing live scheduling and WITHOUT hardcoding
FreeToken equations:

```python
measure_cpu_ms_per_expert(...) -> CpuMeasurement
measure_h2d_ms_per_expert(...) -> H2DMeasurement
measure_gpu_ms_per_expert(...) -> GpuMeasurement
plan_split(m, t_cpu, t_h2d, t_gpu, transfer_overlaps=True) -> SplitPlan(q_star, ...)
```

`q* = argmin_q max(T_gpu(q), T_cpu(m-q))` by ENUMERATION (dee top-k tiny:
top-6/8, so all assignments enumerable). See Phase E doc + simulator tests.
No `Engine` dependency; pure data in, plan out.

## 8. Correctness contract (Phase D)

`tests/test_kt_bridge_correctness.py` proves against dee's trusted references
(`scripts/deepseek_v4_expert_reference.py`, `scripts/deepseek_v4_moe_reference.py`):

- codec identity (nibble order, e8m0 law, dequant shape/scaling),
- clamp asymmetry + `limit=0` disable,
- routing-weight placement equivalence,
- determinism (repeat execute bitwise identical),
- metrics `max_abs/mean_abs/mean_rel/p95_rel/p99_rel/cosine/finite` of
  KT-emulated vs fp32 reference on synthetic fixtures (small multiples of 32)
  + one real-expert path if fixtures/artifacts provide it (falls back to
  synthetic with a logged skip — never fabricates "real" weights).

Existing dee gates are NOT weakened; bridge gates are additive.

## 9. Portability

- Phase-1 C++/Python: ISA-neutral, no intrinsics, no CUDA, builds anywhere
  (MSVC `/arch:AVX2` not even needed; plain scalar fp32).
- AVX2 machines: reference executor IS the verified baseline (same contract
  as KT AVX2 `fmadd_ps` path).
- AMX/AVX512-BF16 machines: KT-faithful executor quantifies the
  bf16-boundary delta; native AMX kernel port is a later optimization behind
  the same interface (dispatch by `_cpu_detect`-style probe at init, log
  variant, never assert).
- T4 (`SM_75`) note: outside KT's validated GPU matrix (`SM_86/89/120`);
  irrelevant here — CPU execution is CUDA-arch-independent by construction.

## 10. What Codex should cherry-pick later

1. `packed_expert_view.hpp` + `cpu_executor.hpp` (interface — stable).
2. `reference_cpu_executor.{hpp,cpp}` (arbiter — keep).
3. `kt_cpu_executor.{hpp,cpp}` scale-compaction + bf16-boundary order (port
   inner dots to AMX when ready).
4. `cost_model.py` `plan_split` enumeration (wire to planner with measured
   `t_cpu/t_h2d/t_gpu`, not constants).
5. Do NOT cherry-pick: singleton pool, global mask, SGLang router coupling,
   full-pool loader, GGUF path.
