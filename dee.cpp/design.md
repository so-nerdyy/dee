# dee.cpp — Design

This document maps the proven Python prototype to the C++ engine, and records the
**findings from reading llama.cpp source** that inform the design.

---

## A. Findings from llama.cpp analysis (Step 1)

### A.1 mmap weight loading — `src/llama-mmap.cpp`
- `llama_mmap::impl` calls `mmap(NULL, file->size(), PROT_READ, MAP_SHARED, fd, 0)`
  on POSIX, `CreateFileMapping`/`MapViewOfFile` on Windows. The whole file is
  mapped read-only; no eager copy into RAM.
- After mapping it issues **`posix_madvise(addr, prefetch, POSIX_MADV_WILLNEED)`**
  (warm a prefix) and **`POSIX_MADV_RANDOM`** over the whole region — tells the
  kernel this is random-access (our exact access pattern: jumping to expert
  slices). This is the key primitive we reuse: the OS page-cache becomes our
  "free" staging buffer, and `cudaMemcpyAsync` from the mmap'd address pulls
  exactly the pages we touch.
- `use_mmap` is toggled off when a weight must live on a non-CPU device; in that
  path llama.cpp allocates a device buffer instead. For DEE we keep mmap as the
  **always-on source of truth** and copy slices into VRAM on demand.

### A.2 MoE forward pass — `src/llama-graph.cpp` `build_moe_ffn`
- The MoE block is built as a **ggml compute graph** (`ctx0`), not imperative
  loops. `build_moe_ffn(...)` takes the per-expert weight tensors
  `up_exps / gate_exps / down_exps` (shape `[n_expert, ...]`), computes router
  `logits = matmul(gate_inp, cur)`, selects `n_expert_used` experts, and runs
  the grouped FFN.
- Critical detail: expert weights are stored **densely as `[n_expert, ...]`**
  and the graph indexes/slices them. For DEE we invert this: instead of holding
  all 256 experts resident, we hold only the Oracle-predicted `K` experts in
  VRAM (a dense `[K, ...]` sub-tensor) and rewrite the graph to index that.
- `src/models/qwen2moe.cpp` shows the per-model FFN wiring (gate/up/down +
  router) — our Ornith mapping target.

### A.3 GPU offloading / async I/O — `src/llama-model-loader.cpp`
- Device selection is capability-driven: `ggml_backend_dev_by_type`,
  `weight_buft_supported(...)`, and a per-tensor buffer-type list decide where
  each weight lands (CPU host buffer vs CUDA device buffer).
- **Async upload path** (lines ~1442–1661) is exactly our prefetch blueprint:
  - Allocates **64 MB host (staging) buffers** ("64MB works well for NVMe
    drives") + a ring of `ggml_backend_event_t` + `host_ptrs`.
  - Uses `ggml_backend_dev_cpy_async` (or equivalent) so the copy runs on a
    **secondary stream**, decoupled from compute.
  - `ggml_backend_event_synchronize(event)` only at the join point.
- This is the structure we replicate with **`cudaMemcpyAsync` on a dedicated
  CUDA stream** + `cudaEvent_t` per in-flight transfer.

### A.4 Summary of llama.cpp's async/I/O structure
- Weights live in an mmap'd file (page cache = free staging).
- A load-balancer assigns each tensor to a device buffer type.
- Async copies use host-staging buffers + events so disk→VRAM never stalls the
  compute stream.
- The forward is a static ggml graph re-evaluated per token.

---

## B. Mapping Python prototype → C++

### B.1 The Oracle (predicts next-layer experts)
**Python:** `oracle.pt` = `{config, layers:{0..39: state_dict}}`, each a 3-layer
MLP `Linear(2048)→ReLU→Linear(256)→ReLU→Linear(256)` with BCE logits; sigmoid →
`topk(PRED_K=48)`.

**C++ plan:**
- Add a tiny **PyTorch-free** weight format, OR load `oracle.pt` by vendoring
  `torch::jit` / `torch::load`. Decision: **vendor libtorch `torch::load`** for
  v1 (fastest path, matches the proven weights); later add a custom
  `safetensors`-style flat binary for zero-dependency deployment.
- `class Oracle { load(path); std::vector<int> predict(const float* hidden, int k); }`
  runs the 3 matmuls via ggml (CPU) — it is tiny (2048→256→256→256) and runs in
  microseconds, well within the inter-layer budget.
- Oracle output = list of expert indices to pre-stage for the *next* layer.

### B.2 VRAM Cache Manager (LRU / priority)
**Python:** hooks materialized a layer, ran it, then `_free_layer` reset params
to meta — i.e. implicit per-layer LRU.

**C++ plan:**
- `class VramCacheManager`
  - Fixed VRAM budget (e.g. 20 GB on the 3090, leaving headroom).
  - `std::unordered_map<int, ExpertBlock>` keyed by `(layer, expert)`; each
    block owns a `cudaMalloc`'d buffer + metadata (last-used tick, priority).
  - Eviction policy = **LRU with Oracle priority**: on insert over budget, evict
    lowest `(tick, oracle_priority)`; Oracle-predicted experts get a priority
    boost so they are never the first evicted.
  - Custom allocator: a **bump/arena allocator** over a single large
    `cudaMalloc` region (avoids per-expert malloc fragmentation); free-list
    reuses slots. `std::vector` holds the free slots; indices are handles.
  - `ensure(layer, expert)` → returns pointer; triggers `AsyncPrefetcher` if
    not resident; `sync_fallback()` blocks only on a cache miss that compute
    already reached.

### B.3 Async Prefetcher (NVMe → VRAM, decoupled stream)
**Python:** none (hooks were synchronous, hence the 0.05 tok/s floor).

**C++ plan:**
- `class AsyncPrefetcher`
  - Owns a **secondary `cudaStream_t`** + a pool of pinned host staging
    buffers (mirroring llama.cpp's 64 MB host buffers).
  - `prefetch(layer, expert, dst_ptr)`: `cudaMemcpyAsync(staging, mmap_addr,
    size, HostToHost)` is unnecessary — mmap is already in host memory, so go
    straight `cudaMemcpyAsync(dst_ptr, mmap_addr + offset, size,
    HostToDevice, prefetch_stream)`.
  - Each in-flight copy tagged with a `cudaEvent_t`; the compute stream calls
    `cudaStreamWaitEvent(compute_stream, ev)` only for experts it needs *now*.
  - Driven by the **Oracle look-ahead**: while layer `L` computes, the
    prefetcher streams layer `L+1`'s predicted experts on the side stream.
  - **Sync fallback:** if compute reaches an expert not yet resident (Oracle
    miss), block on that one `cudaEvent_t` — same semantics as the Python
    prototype's `fallbacks` counter. We will measure fallback rate; target
    <5% (prototype showed 100% recall at K=48, so fallback should be ~0).

### B.4 MoE forward with eviction
- Build a ggml graph like `build_moe_ffn` but with the **resident `[K,...]`
  expert tensor** instead of all 256. Router still scores all experts on CPU
  (cheap) to pick top-8; the top-8 set is guaranteed ⊆ predicted-K (recall
  100% at K=48), so every chosen expert is already in VRAM.
- Per-layer loop mirrors the Python hook pattern but in C++: materialize
  (ensure in cache) → compute → advance LRU tick.

---

## C. Dependencies
- **ggml** (vendored from llama.cpp) — tensor ops + CUDA kernels.
- **CUDA Toolkit** — `cudaMemcpyAsync`, streams, events (RTX 3090 = sm_86).
- **libtorch** (v1, optional) — `torch::load` for `oracle.pt`; swappable for a
  custom flat format later.
- Std: `std::thread`, `std::vector`, `std::unordered_map`, `<filesystem>`,
  (optional `liburing` for io_uring on Linux — deferred; mmap+cudaMemcpyAsync
  already removes the disk-syscall bottleneck).

## D. Performance targets
| Metric            | Python prototype | dee.cpp target |
|-------------------|------------------|----------------|
| Resident VRAM     | ~1 GB (Kaggle)   | < 22 GB (3090) |
| Oracle recall     | 100% @ K=48      | 100% @ K=48    |
| Tok/s (35B MoE)   | 0.05             | **30+**        |
