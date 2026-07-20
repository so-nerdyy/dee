# dee.cpp — Dynamic Expert Eviction Inference Engine

> A high-performance C++ inference engine optimized for **Dynamic Expert Eviction (DEE)**:
> run large Mixture-of-Experts models (e.g. Ornith-1.0-35B, 256 experts / top-8) on
> consumer hardware (NVMe SSD + RTX 3090, 24 GB VRAM) by streaming only the
> Oracle-predicted experts into VRAM.

## Why this exists

The Python prototype proved the concept end-to-end:

- Runs a ~70 GB / 35B MoE in **~1 GB** resident memory.
- **100% Oracle recall** of the truly-activated top-8 experts.
- Throughput was the blocker: **0.05 tok/s**, bottlenecked by Python + PyTorch
  hooks + Kaggle network storage.

`dee.cpp` replaces that with a native engine built on **ggml** (llama.cpp's
tensor backend) targeting **30+ tok/s** on a single RTX 3090.

## Architecture (high level)

```
            NVMe SSD (.safetensors / .gguf)
                  │  mmap + async prefetch
                  ▼
        ┌─────────────────────────────┐
        │  WeightMmap  (read-only)    │   zero-copy file mapping
        └─────────────────────────────┘
                  │  cudaMemcpyAsync (prefetch stream)
                  ▼
        ┌─────────────────────────────┐
        │  VramCacheManager (LRU)      │   predicted experts live in VRAM
        │  - priority queue           │
        │  - sync fallback on miss    │
        └─────────────────────────────┘
                  │  compute stream
                  ▼
        ┌─────────────────────────────┐
        │  MoE forward (ggml graph)    │   only resident experts
        └─────────────────────────────┘
                  ▲
        ┌─────────────────────────────┐
        │  Oracle (3-layer MLP)        │   predicts next-layer experts
        │  loads oracle.pt weights     │
        └─────────────────────────────┘
```

## Build

See `CMakeLists.txt`. Requires CMake ≥ 3.18 and (for GPU) CUDA Toolkit.

```bash
cmake -B build -DDEE_CUDA=ON
cmake --build build -j
```

## Status

- [x] Step 1 — cloned & analyzed llama.cpp (mmap / MoE / offload)
- [x] Step 2 — CMake + directory skeleton
- [x] Step 3 — C++ architecture (this doc)
- [x] Step 4 — WeightMmap + TensorResolver (data layer) + smoke test PASS
- [x] Step 5 — VramCacheManager (LRU + Oracle-priority, arena + free-list) + test PASS
- [x] Step 6 — AsyncPrefetcher (decoupled stream + per-transfer events, mock + CUDA-guarded) + test PASS
- [x] Step 7 — Oracle loader (dependency-free PyTorch .pt ZIP+pickle reader) + OracleScheduler (3-layer MLP → top-K) + test PASS + torch-free Python cross-check EXACT match
- [x] Step 8 — MoE forward (SwiGLU, raw C++) + autoregressive DEE loop wired Oracle→Prefetcher→Cache + CLI driver (dee_cli) + test PASS + CPU-mock run verified
- [x] Step 9 — CUDA GPU port: real SwiGLU `__global__` kernels + async prefetcher (cudaMemcpyAsync H2D + per-expert cudaEvent_t sync) + cudaMalloc VRAM + cudaMemGetInfo reporting; guarded by `#ifdef DEE_CUDA`, builds clean on CPU, DEE_CUDA=ON configured for the RTX 3090
- [ ] Step 10 — final benchmark on RTX 3090 (real hardware)

> Note: Steps 5 and 6 were implemented in this session (VramCacheManager then
> AsyncPrefetcher). The broad CUDA arch list ("60;70;75;80;86;89;90") is set in
> CMakeLists.txt. The Step 8 CLI driver is deferred until Step 7 (Oracle) and
> the forward pass exist.

## Build

```bash
# CPU-only (data layer, no CUDA needed) — verified green:
cmake -B build -DDEE_BUILD_GGML=ON -DDEE_CUDA=OFF
cmake --build build -j
./build/test_weight_mmap

# RTX 3090 (sm_86) — requires CUDA Toolkit 12.x:
cmake -B build -DDEE_BUILD_GGML=ON -DDEE_CUDA=ON
cmake --build build -j
```

Step 4 status: the data layer (WeightMmap + TensorResolver) is pure POSIX/C++ and
is fully unit-tested without ggml. `DEE_BUILD_GGML=ON` additionally compiles the
vendored ggml backend (CPU) and links it into `dee_core` for the forward pass.

## Step 8 — MoE forward + DEE loop + CLI driver

The runtime (`Engine` in `src/engine.cpp`) wires the full Dynamic Expert
Eviction pipeline for one autoregressive token:

```
for token in 0..N:
  hidden = embed(token)                 # mock: deterministic init
  for layer L in 0..L-1:
    experts = oracle.predict(L, hidden, topk)   # which experts activate
    for e in experts: prefetcher.prefetch(L, e) # stream gate|up|down into VRAM
    for e in experts: prefetcher.wait(L, e)     # sync fallback if needed
    hidden = moe_swiglu(L, hidden, experts)     # SwiGLU combine (RMS-norm)
```

- **SwiGLU** is plain C++ matmul (row-major gate/up/down slices carved out of
  one F32 expert blob). The CUDA/ggml kernel slots in later behind the same API.
- Each expert's `gate|up|down` weights are bundled into a single F32 staging blob,
  prefetched as one VRAM block, and sliced in the forward pass.
- `OracleScheduler.predict` (Step 7) drives which experts get staged; the
  `VramCacheManager` evicts by `last_used + priority`, so top-ranked experts
  survive under a tight VRAM budget.

### Synthetic shard

`tests/gen_moe_shard.py` writes `tests/data/ornith_moe256.safetensors`: 256
experts at layer 0 (Ornith naming, BF16, hidden=2048, inter=64) so top-8 Oracle
prediction exercises real prefetch + cache-eviction pressure. Experts 0/1/2 carry
canonical KNOWN values; 3..255 a deterministic filler. (The real multi-layer
shard maps every layer to its own experts; the mock maps all layers to layer 0.)

### CLI driver

```bash
./build/dee_cli --shard tests/data/ornith_moe256.safetensors \
                --oracle /path/to/oracle.pt --tokens 16 --topk 8 --layers 40
# optional: --budget BYTES (default 4 experts), --cuda (real CUDA path)
```

### Verified results (CPU-mock, honest local numbers — WSL, no GPU)

| scenario            | throughput | peak VRAM | evictions | cache hits | finite |
|---------------------|-----------:|----------:|----------:|-----------:|:------:|
| budget = 4 experts  | 0.84 tok/s | 6.0 MB    | 5116      | 0          | yes    |
| budget = 16 experts | 0.85 tok/s | 24.0 MB   | 5104      | 0          | yes    |

The 0.8 tok/s is single-threaded naive matmul on a CPU-only box — the point of
`dee.cpp` is the RTX 3090 path, where the same control flow runs the SwiGLU on
the GPU and the prefetcher overlaps NVMe→VRAM copies with compute. Evictions are
high because the per-layer Oracle scatters predictions across the 256-expert
space (realistic DEE pressure); the cache-hit path is covered by
`test_vram_cache`. All 6 ctest suites pass (incl. `test_engine`: SwiGLU kernel
matches a hand-computed 2×2 case, and the e2e loop produces finite output with
real eviction activity).

## Step 9 — CUDA GPU port

The CPU forward path is replaced by a real CUDA path behind `#ifdef DEE_CUDA`:

- **SwiGLU kernels** (`src/swiglu_cuda.cu`): two `__global__` phases per expert —
  `swiglu_h_kernel` computes `h[i] = silu(Wg[i,:]·x)·(Wu[i,:]·x)` (one thread per
  `inter` row), `swiglu_y_kernel` computes `y[o] = Σ_i Wd[o·inter+i]·h[i]` (one
  thread per `hidden` output), then `combine_kernel` mean-pools the K expert
  outputs. All launched on a dedicated **compute stream**.
- **Async prefetcher (CUDA)** (`src/async_prefetcher.cpp`): `prefetch()` issues
  `cudaMemcpyAsync(host_staging → cudaMalloc arena, stream=secondary)` straight
  from the mmap-derived host blob, then `cudaEventRecord`s a **per-expert
  `cudaEvent_t`**. `wait()` calls `cudaEventSynchronize(expert_event)` — the
  compute stream only blocks on the one expert it needs *now* (sync fallback), so
  independent transfers never stall it.
- **VRAM arena**: the `VramCacheManager` backend switches to `cudaMalloc`/
  `cudaFree` under `use_cuda`, so `cache_.data(L,e)` returns a device pointer the
  kernel reads directly. `Engine::init` also `cudaMemGetInfo`s total/free GPU
  memory and reports it; `dee_cli --cuda` switches the backend + kernel path.

### Build (RTX 3090 / sm_86 — requires CUDA Toolkit 12.x)

```bash
cmake -B build -DDEE_CUDA=ON -DDEE_BUILD_TESTS=ON -DDEE_BUILD_GGML=OFF
cmake --build build -j
./build/dee_cli --cuda --shard tests/data/ornith_moe256.safetensors \
                --oracle /path/to/oracle.pt --tokens 32 --topk 8 --layers 40
# typename           throughput  peak VRAM  evictions  finite
# CPU-mock (WSL)     0.84 tok/s   6.0 MB     5116       yes
# RTX 3090 (target)  (run on HW)  (run on HW) (run on HW)  yes
```

### Honest status of this step

This WSL dev box has **no nvcc/CUDA toolkit and no GPU**, so the `DEE_CUDA=ON`
binary could not be compiled or run here. What WAS verified:
- The `DEE_CUDA=OFF` build (all the new code behind `#ifdef DEE_CUDA`) compiles
  clean and **all 6 ctest suites still pass**; the CPU-mock CLI runs unchanged.
- The CUDA source was reviewed line-by-line for correctness (kernel launch
  syntax, `cudaMemcpyAsync` HostToDevice from the mmap host pointer, per-expert
  `cudaEvent_t` record/sync, `cudaMemGetInfo`, device-buffer lifetimes, and
  `cudaMalloc`/`cudaFree` teardown via the cache destructor). One definite
  compile bug (`float**` passed to `cudaMalloc` which needs `void**`) was found
  and fixed.

The CUDA path is therefore **compile-reviewed but not runtime-verified** on this
box. The CMake configuration (`find_package(CUDAToolkit)`, `enable_language(CUDA)`,
broad arch list, `DEE_CUDA_SRC` glob + `CUDA::cudart` link) is in place so it
should compile on the 3090 machine with the command above. Final throughput /
VRAM numbers come from the Step 10 run on real hardware.
