# dee.cpp: Dynamic Expert Eviction

`dee.cpp` is a correctness-first C++17/CUDA implementation of the Dynamic
Expert Eviction control path: a safetensors shard is mapped read-only, Oracle
predictions select experts, a fixed cache manages residency, and CUDA runs a
small mixed-precision SwiGLU reference kernel on resident expert weights.

On the default CUDA streaming path, BF16 weights are quantized once into
per-projection INT8 pinned sources, transferred as INT8, dequantized to FP16 on
the prefetch stream, and stay FP16 in the expert cache. cuBLAS reads FP16
weights and activations with FP32 accumulation. Pass `--transfer-dtype bf16`
for lossless BF16 transfer, or `--cache-dtype fp32` for the exact
BF16-to-FP32 cache and SGEMV fallback.
Frequently reused packed source blobs are materialized once in a bounded 192 MiB
pinned-host cache; shards that exceed that bound fall back to the bounded
staging ring. The configured cache remains 6 MiB; FP16 doubles its physical
expert capacity from four to eight entries.

Oracle matrix products use a runtime-dispatched AVX2/FMA dot product on
supported x86 CPUs and retain a portable scalar fallback.

The supported implementation is deliberately singular:

```text
WeightMmap -> TensorResolver -> OracleScheduler (oracle.h)
           -> VramCacheManager (vram_cache.h) -> AsyncPrefetcher -> Engine
```

The older duplicate cache/scheduler/generation prototype has been removed; see
[`legacy/README.md`](legacy/README.md).

## Lightning AI / NVIDIA T4 benchmark

From a fresh Ubuntu Lightning Studio with an NVIDIA T4 (compute capability
`sm_75`), CUDA 13, CMake, Python 3, and Git LFS:

```bash
git clone https://github.com/so-nerdyy/dee.git
cd dee/dee.cpp
git lfs pull
./scripts/setup_lightning_t4.sh
./scripts/setup_lightning_t4.sh --benchmark
```

The setup script verifies `nvidia-smi`, `nvcc`, and Git LFS; rejects an
unresolved `oracle.pt` LFS pointer; deterministically generates the synthetic
benchmark shard; configures `sm_75`; builds; and runs a CUDA smoke test.  The
benchmark form additionally uses two warmup tokens before measuring 32 tokens.

Equivalent manual build:

```bash
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEE_CUDA=ON \
  -DDEE_BUILD_TESTS=OFF \
  -DDEE_BUILD_GGML=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --parallel "$(nproc)"
./build/dee_cli \
  --shard tests/data/ornith_moe256.safetensors \
  --oracle oracle.pt \
  --tokens 32 --warmup 2 --topk 8 --layers 40 --cuda
```

`CMAKE_CUDA_ARCHITECTURES` is respected when supplied. If omitted for a CUDA
build, dee.cpp defaults to `75` rather than compiling every historical GPU
architecture.

### Assets

`oracle.pt` is stored in Git LFS. Run `git lfs pull` after clone. The CLI and
Engine detect the text LFS pointer and print that exact recovery command.

`tests/data/ornith_moe256.safetensors` is intentionally generated rather than
committed as a large normal Git object:

```bash
python3 tests/gen_moe_shard.py tests/data/ornith_moe256.safetensors
```

The generator has no third-party Python dependency and produces the same
single-layer, 256-expert BF16 shard every time. It is a synthetic benchmark
asset, not an Ornith model checkpoint.

### CLI

```text
dee_cli --help
dee_cli --shard PATH --oracle PATH --tokens N --warmup N --topk N --layers N
        --budget BYTES --prefetch-depth N --cache-dtype fp16|fp32
        --transfer-dtype int8|bf16|int4 --cuda
        --profile-stages --profile-scenario MODE --profile-json PATH
        --profile-timeline PATH --trace-requests PATH --verbose
```

Defaults are `tests/data/ornith_moe256.safetensors` and `oracle.pt`, relative to
the project root. The CLI validates malformed numeric arguments, missing files,
unresolved LFS pointers, CPU-only `--cuda`, and CUDA runtime failures before
starting a benchmark. It reports device/runtime metadata, warmup and measured
tokens, throughput, cache budget/peak, cache activity, prefetch activity, and
whether the output is finite.

### Stage profiling and request traces

Detailed profiling is opt-in so CUDA timing events do not affect the normal
regression benchmark. Warmup uses a separate unprofiled Engine instance, so
all reported stage timings and counters cover measured tokens only:

```bash
./build/dee_cli \
  --shard tests/data/ornith_moe256.safetensors \
  --oracle oracle.pt \
  --tokens 32 --warmup 2 --topk 8 --layers 40 --cuda \
  --profile-stages \
  --profile-json benchmark_reports/t4-stage-profile.json
```

Add the request trace only when working-set analysis is needed:

```bash
./build/dee_cli \
  --shard tests/data/ornith_moe256.safetensors \
  --oracle oracle.pt \
  --tokens 32 --warmup 2 --topk 8 --layers 40 --cuda \
  --profile-stages \
  --profile-json benchmark_reports/t4-stage-profile.json \
  --trace-requests benchmark_reports/t4-request-trace.json
```

The stage profile reports CPU wall-clock spans, CUDA-event device durations,
operation counts, token latency percentiles, request classification, transfer
volume, and working-set/reuse statistics. CUDA timing uses a bounded reusable
event pool and collects samples only at synchronization points already present
in the runtime. BF16 cache conversion is reported separately from H2D and
projection time. The request trace also includes the final hidden vector for
FP32/FP16 validation and is not enabled by the normal benchmark command.

Add `--profile-timeline benchmark_reports/t4-timeline.json` for a Chrome trace
and exact common-origin copy/compute interval union. The summary reports copy
and compute utilization, overlap, idle bubbles, transfer queue depth, and
host-wait reasons. This detailed timeline is intentionally opt-in.

Quantized transfer modes pre-pack a bounded persistent host source cache before
timed decoding. The CLI reports startup cost, packed expert count, and packed
bytes separately. Use `--dynamic-quantization` to retain first-touch
quantization for cold-start/control comparisons.

Compare numerical output and expert routes between two request-trace reports:

```bash
python3 scripts/compare_benchmark_outputs.py reference.json candidate.json \
  --max-abs 0 --relative-rmse 0 --require-exact-routes
```

Summarize working-set and reuse-distance behavior with:

```bash
python3 scripts/analyze_request_trace.py \
  benchmark_reports/t4-request-trace.json \
  --output benchmark_reports/t4-working-set.json
```

For controlled A-G decomposition, run:

```bash
./scripts/profile_scenarios.sh
```

This writes one JSON report for each of `end-to-end`, `full-resident`,
`resident-bypass`, `transfer-only`, `compute-only`, `oracle-only`, and
`cache-metadata-only`. Full-resident and resident-bypass preload every physical
expert before the measured clock and retain the Oracle-driven request sequence.
The other modes are explicit ablations: disabled compute means their recurrent
hidden state and later Oracle choices cannot be identical to end-to-end. The
compute-only mode uses a deterministic expert schedule and does not include
Oracle time. These modes are diagnostic controls, not alternative throughput
claims; the default 6 MiB end-to-end regression remains unchanged.

### What the benchmark means

This is a synthetic-kernel/control-path benchmark. It validates cache
residency, bounded pinned BF16 source/staging memory, H2D transfers, GPU cache
conversion, and FP32/FP16 cuBLAS SwiGLU correctness. It is **not complete 35B
end-to-end model inference**, and the
reference kernels must not be interpreted as a claim of 30+ tok/s on a
real 35B model. In particular, the first copy from file-backed/pageable data
into pinned memory is host work; only the pinned-host-to-device leg is
asynchronous.

INT8 is validated for this deterministic synthetic workload but is not a
model-quality claim. `--transfer-dtype int4` is opt-in and experimental: it is
faster here but has a larger measured output error and requires calibration on
real checkpoints before use in complete inference.

The measured T4 profiling campaign, controlled A-G decomposition, accepted
optimization, and rejected experiments are recorded in
[`T4_PROFILE_RESULTS.md`](T4_PROFILE_RESULTS.md).

### Debugging CUDA

```bash
nvidia-smi
CUDA_LAUNCH_BLOCKING=1 ./build/dee_cli --cuda --tokens 1 --topk 1 --layers 1
compute-sanitizer --tool memcheck ./build/dee_cli --cuda --tokens 1 --topk 1 --layers 1
compute-sanitizer --tool racecheck ./build/dee_cli --cuda --tokens 1 --topk 1 --layers 1
```

For close launch-site error checking, configure with
`-DDEE_CUDA_VALIDATE=ON`. This synchronizes after individual kernel launches
and is intended for validation, not performance measurement.

## CPU test suite

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEE_CUDA=OFF \
  -DDEE_BUILD_TESTS=ON
cmake --build build-cpu --parallel "$(nproc)"
ctest --test-dir build-cpu --output-on-failure
```

The test build generates deterministic safetensors test assets automatically.
