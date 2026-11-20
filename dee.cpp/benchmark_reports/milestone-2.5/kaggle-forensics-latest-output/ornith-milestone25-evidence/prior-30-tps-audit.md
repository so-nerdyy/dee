# Prior approximately-30-TPS benchmark audit

Status: forensic inspection complete; historical T4 result identified and
correctly labeled. The exact historical run was not re-executed in this
workstream because the current Windows host has neither `nvidia-smi` nor
`nvcc`. No runtime code was changed.

## Bottom line

The earlier approximately-30-TPS number was **not Ornith/Qwen3.5-MoE text
generation throughput**. It was the median of three unprofiled runs of the
single-GPU, opt-in INT4 path added at
`75d5218ee7385985119f41e86375ba102755e274`:

| Run | Measured seconds | Printed `tokens per second` |
|---:|---:|---:|
| 1 | 1.040072 | 30.767 |
| 2 | 1.046735 | 30.571 |
| 3 | 1.046834 | 30.568 |
| Median | 1.046735 | **30.571** |

The truthful unit is **synthetic recurrent MoE steps per second**. Each
reported step executes 40 logical MoE calls and requests eight experts per
call, but it does not consume or emit a vocabulary token. At the median, the
same result can be normalized to 1,222.84 logical MoE calls/s or 9,782.72
selected-expert requests/s. Those normalizations are count-rate conversions,
not independent kernel measurements.

The result proves that the historical dee.cpp cache/transfer/reference-SwiGLU
control path could execute this small synthetic workload on one Tesla T4. It
does not prove 30 generated tokens/s, 30 complete transformer passes/s, real
checkpoint quality under INT4, or dual-GPU scaling. The repository already
contains the same warning in `README.md:181-195` and
`T4_PROFILE_RESULTS.md:306-332`.

## Exact identification and commit provenance

| Commit | Role in the result |
|---|---|
| `59d4d03907eaebe347aad52105f9986bb98801bd` | Parent baseline: INT8 transfer/dequantization. |
| `75d5218ee7385985119f41e86375ba102755e274` | Adds the opt-in packed INT4 transfer path. This is the source commit used by the three 30.767/30.571/30.568 runs. |
| `ff0b538237386d439488c38158b20d117a8a71ee` | Immediate child that records the three-run result and makes validated INT8, not INT4, the default. |
| `5548f6d` | Later adds startup prepacking outside the timed region. This behavior was **not** present in the historical 30.571 run. |
| `dbdb7dd3563e65701d80158c2cadf7b004ef2806` | Later descendant that only fixes the output comparator. It contains the prior INT4 history but is not the 30.571 implementation commit. Its default CUDA mode is prepacked INT8. |

`dbdb7dd` therefore must not be cited as a 30-TPS full-model result. Its fresh,
unprofiled default-INT8 runs were 25.615, 25.440, and 25.559 steps/s (median
25.559), with a separately reported 215-216 ms startup prepack; see
`T4_PROFILE_RESULTS.md:334-375`.

## Exact historical invocation

The execution record pins branch `candidate/int4-transfer`, commit
`75d5218ee7385985119f41e86375ba102755e274`, a Tesla T4 with compute capability
7.5, and this command repeated three times:

```bash
./build/dee_cli \
  --shard tests/data/ornith_moe256.safetensors \
  --oracle oracle.pt \
  --tokens 32 \
  --warmup 2 \
  --topk 8 \
  --layers 40 \
  --cuda \
  --transfer-dtype int4
```

No `--profile-stages`, `--profile-json`, `--trace-requests`, or
`--profile-timeline` flag was present in the headline runs. Defaults supplied
the 64-entry prefetch ring, FP16 device cache under `--cuda`, and a 6,291,456 B
expert-cache budget.

The tracked campaign header records the common base invocation and environment
(Lightning AI Studio, one Tesla T4, `sm_75`, CUDA runtime 13.0, driver
580.159.03) at `T4_PROFILE_RESULTS.md:1-11`. The explicit INT4 selector and
result are recorded at `T4_PROFILE_RESULTS.md:306-325`.

## Exact metric semantics

At `75d5218`, `Engine::generate()` starts a `steady_clock` timer, creates a
fixed sinusoidal 2,048-element hidden vector, executes the configured number of
synthetic steps and logical layers, drains the prefetcher, and stops the timer.
It then assigns:

```text
tokens = cfg.num_tokens
tokens_per_second = cfg.num_tokens / elapsed_seconds
```

Historical source: `75d5218:dee.cpp/src/engine.cpp:745-823`.

For this invocation:

```text
measured steps              = 32
logical MoE calls           = 32 * 40 = 1,280
selected-expert requests    = 32 * 40 * 8 = 10,240
median wall time per step   = 32.711 ms
median logical-call rate    = 1,222.84/s
median request rate         = 9,782.72/s
```

The word `token` is misleading for two independent reasons:

1. There is no tokenizer, embedding lookup from token IDs, vocabulary logits,
   sampling, or decoded text.
2. Within one outer step, the 40 logical calls all read the same `h_in` buffer
   and overwrite the same `h_out` buffer; the buffers are swapped only after
   the 40-call loop. Thus layer 0's output is not fed to layer 1. Only the final
   logical call's output becomes the next outer step's input. See
   `75d5218:dee.cpp/src/engine.cpp:754-777`.

The benchmark is consequently a repeated MoE control-path loop, not a
40-layer transformer forward pass.

## Exact workload

| Property | Historical approximately-30 result |
|---|---|
| Hardware | One NVIDIA Tesla T4, compute capability 7.5; no second GPU is addressed by this runtime. |
| CUDA | Runtime 13.0 (`cuda_runtime_version=13000`). |
| Scenario | `end-to-end` **inside the synthetic MoE harness**. |
| Batch size | One hidden vector. The expert projection uses a one-column cuBLAS GEMM (`m=rows, n=1, k=cols`). |
| Sequence/context | No attention context or KV/recurrent-attention state. Thirty-two sequential synthetic hidden-state iterations. |
| Logical layers | 40 Oracle/MoE calls per measured step. |
| Physical expert layers | One: generated shard layer 0. Missing logical layers are explicitly remapped to physical layer 0 (`75d5218:dee.cpp/src/engine.cpp:231-235`). |
| Experts | 256 physical expert IDs; top-8 selected per logical call. |
| Expert dimensions | `hidden=2048`, **`inter=64`**, three matrices per expert. The generator's prose says 256, but its executable constant at `75d5218:dee.cpp/tests/gen_moe_shard.py:24` is authoritative. |
| Expert data | One deterministic BF16 layer. Experts 3-255 repeat one filler byte pattern; this is not a genuine checkpoint (`gen_moe_shard.py:42-78`). |
| Routing | CPU `oracle.pt` MLP, invoked once per logical call; IDs only. It is not the real checkpoint router. |
| Combination | Equal mean of eight expert outputs, not learned router weights (`75d5218:dee.cpp/src/swiglu_cuda.cu:30-35`). |
| Source precision | Generated BF16 weights. |
| Transfer precision | Signed packed INT4, with one symmetric scale per gate/up/down projection. |
| Cache/compute precision | INT4 is dequantized into an FP16 cache; FP16 weights/activations use FP32 cuBLAS accumulation; expert outputs and equal-mean combine are FP32. |
| Cache | One global 6,291,456 B cache. A synthetic FP16 expert is 786,432 B, so exactly eight experts fit. |
| Warmup | Two excluded steps in a **separate Engine instance** (`75d5218:dee.cpp/main.cpp:404-415`). Its expert cache is destroyed before the measured Engine is created. |

The generated shard is 201,326,592 B of BF16 expert data plus its header. The
real Milestone 2 expert transfer is 6,291,456 B per miss; the historical INT4
transfer is 196,608 B per miss. The old benchmark therefore transfers **32x
less expert data per miss**, due to an 8x smaller intermediate width (64 versus
512) and 4x packing (FP16/BF16-sized data versus INT4).

## What the headline includes and excludes

| Cost/category | Headline 30.571 run | Evidence/qualification |
|---|---|---|
| CPU Oracle routing | Included | 1,280 calls. The profiled sibling records 306.975 ms. |
| Tensor lookup and cache metadata | Included | Normal `end-to-end` scenario uses the cache path. |
| First-touch BF16-to-INT4 quantization | Included | At `75d5218`, `get_staging_int4()` is called lazily from `stage_expert()` inside the timed loop (`engine.cpp:383-445`). Startup prepack did not exist yet. |
| Pinned-source allocation/quantization | Included on first unique expert use | Profile `mmap_to_pinned_bytes` counts 180,879,360 BF16 source bytes read/quantized over 230 experts (159.898 ms); the packed pinned destinations total 45,219,840 B. This combined category is not a literal byte-for-byte pinned copy. |
| Expert H2D | Included | Profile records 1,999,110,144 B from 10,168 196,608-B expert copies. |
| Hidden-vector H2D | Included | 1,280 copies * 8,192 B = 10,485,760 B. |
| INT4-to-FP16 device conversion | Included | Profile records 98.758 CUDA-event ms. |
| Expert gate/up/down, SiLU, combine | Included | Profile records 316.827 CUDA-event ms excluding weight conversion. |
| Cache misses/evictions | Included | 72 resident hits, 10,168 cold loads, 10,160 evictions: 99.296875% miss rate. |
| Transfer/compute overlap | Included | Separate prefetch and compute streams overlap partially. |
| Synchronization | Included | Each expert batch and each logical-layer output calls `cudaStreamSynchronize`; see `engine.cpp:979-1019`. |
| Simple RMS stabilization | Included | CPU stand-in at `engine.cpp:1022-1029`; it is not the model's learned norms/residual path. |
| File open, mmap setup, Oracle load | Excluded | Performed by `Engine::init()` before `Engine::generate()` starts its timer. |
| CUDA context/stream/cuBLAS/cache/work-buffer setup | Excluded | Performed by `Engine::init()` before the timer. |
| Two warmup steps | Excluded | Separate Engine run before measured Engine construction. |
| Cold disk state | Not established | The separate warmup maps and reads the same assets, so OS page cache is plausibly warm; no cold-page control was recorded. Demand faults during measured first-touch are not separately reported. |
| Profiler overhead | Excluded from headline | The same profiled path reached only 26.222 steps/s, 14.23% below the unprofiled median. |
| Full-resident preload | Not used | The headline uses the 6 MiB streaming cache; it is not the 37.822-step/s full-resident control. |
| Attention/linear attention, real recurrent state, residuals, learned norms | Absent | No corresponding code executes in this harness. |
| Tokenizer, embeddings, RoPE, mask construction, LM head, sampling, detokenization | Absent | The input is a synthetic vector and the only validation is finite hidden output. |
| Real checkpoint expert weights and router weights | Absent | Generated one-layer shard plus `oracle.pt`. |
| Multi-GPU placement/communication | Absent | Single-device runtime and single T4 result. |

Thus H2D, cache behavior, and synchronization were **not** excluded from the
headline. The major exclusions were model completeness and setup/startup, not
the synthetic streaming path itself.

## Retained profiled evidence for the same INT4 path

The unprofiled normal logs that printed 30.767/30.571/30.568 are not retained
under `benchmark_reports/`. The tracked narrative preserves those values. A
retained, self-describing profiled sibling is available at
`benchmark_reports/campaign-transfer3/int4-profile.json`:

| Field | Value |
|---|---:|
| Device/runtime | Tesla T4 / CUDA 13.0 |
| Steps/warmup/logical calls | 32 / 2 / 1,280 |
| Profiled throughput | 26.222 steps/s |
| Cache budget/peak | 6,291,456 / 6,291,456 B |
| Requests/hits/misses/evictions | 10,240 / 72 / 10,168 / 10,160 |
| Unique requested/loaded | 230 / 230 |
| Total H2D | 2,009,595,904 B in 11,448 copies |
| Expert H2D | 1,999,110,144 B in 10,168 copies |
| Hidden H2D | 10,485,760 B in 1,280 copies |
| H2D CUDA-event time | 412.953 ms (4.866 GB/s over event time) |
| INT4-to-FP16 conversion | 98.758 ms |
| Expert compute plus combine | 316.827 ms |
| Copy/compute overlap | 171.260 ms |
| Neither copy nor compute active | 615.347 ms |

The byte identity is exact:

```text
10,168 * 196,608 B expert copies = 1,999,110,144 B
 1,280 *   8,192 B hidden copies =    10,485,760 B
                                      -------------
                                      2,009,595,904 B
```

Artifact SHA-256:

```text
d562080b3bf12840ba25a468bd2a05335a5d725f65514d3eaee863613155b139  int4-profile.json
```

The JSON does not contain a git commit or full command. Its association with
`75d5218` is supported by the commit history, the tracked campaign narrative,
the historical execution record, and the absence of later prepack fields in
its schema. This is strong corroborating evidence, but not a hermetic benchmark
manifest.

## Controlled modes: what was and was not resident

These are profiled diagnostic controls, not the unprofiled headline:

| Control | Steps/s | What it removes | Interpretation |
|---|---:|---|---|
| INT4 end-to-end profile | 26.222 | Nothing inside the synthetic timed loop | Instrumented counterpart of the headline path. |
| Full resident | 37.822 | Expert streaming; all 256 synthetic experts are preloaded before the clock | Still includes Oracle, expert compute, hidden transfers, combine, and synchronization. |
| Resident bypass | 39.285 | Streaming plus cache lookup/pin bookkeeping | Preloaded weights and deterministic direct access. |
| Transfer only | 31.440 | Expert GEMMs/combine and hidden H2D | Includes Oracle plus cache/staging/expert H2D. Later hidden recurrence is therefore not equivalent to end-to-end. |
| Compute only | 79.096 | Oracle, expert transfers, and normal cache path | Closest retained “isolated expert compute” control, but it still performs 1,280 layer wrappers, hidden H2D/D2H, equal-mean combine, and synchronization. |
| Oracle only | 109.768 | Cache, H2D, and expert compute | CPU Oracle plus hidden copies in the harness. |

Relevant artifact SHA-256 values:

```text
27046c917d8a1a1a36f76f415b90177aa4969414387b08e60c471216669be844  int4-compute-only.json
a95086a9645a1655347c12fc159ccee5723ab5e1ea6966601021559bc464593a  int4-full-resident.json
7886e42646ea542b565585857b355f3db6cc73950c2edbdfc83280a3ae3a0555  int4-resident-bypass.json
286ce96d3d1f9da119dc48a77f660ddc7d9be2858640076b56dbf812eaf1f591  int4-transfer-only.json
```

There is no retained pure single-GEMM or single-expert throughput artifact.
Calling 79.096 “isolated expert-kernel throughput” without the wrapper caveat
would overstate the evidence.

## Direct comparison with verified full-model warm decode

| Level | Measured rate | Normalized work rate | Included scope |
|---|---:|---:|---|
| Closest expert-compute control | 79.096 synthetic steps/s | 3,163.84 logical calls/s; 25,310.74 expert evaluations/s | Preloaded small synthetic experts; deterministic IDs; no Oracle/cache/expert H2D; still includes wrapper, combine, hidden copies, and sync. |
| Historical approximately-30 headline | **30.571 synthetic steps/s** | 1,222.84 logical calls/s; 9,782.72 expert requests/s | Synthetic Oracle + one physical expert layer + 6 MiB cache + INT4 H2D/dequant + expert compute. |
| Verified Milestone 2 warm decode | **0.465670 generated tokens/s** | 18.6268 real full-layer passes/s; 149.014 routed expert selections/s | Genuine 40-layer checkpoint path, actual prompt/token IDs, dense Transformers-owned attention/state/norm/head plus dee.cpp routing/expert cache/execution, split over two T4s. |

The Milestone 2 value is measured from three post-first-token decode durations
(2.139442, 2.148716, and 2.154179 s). It explicitly excludes prefill and the
first token but includes the complete warm decode pass. See
`benchmark_reports/kaggle-milestone2/v14-output/ornith-milestone2-evidence/VERIFIED.md:37-53`
and `ornith-milestone2-report.json#/benchmark/warm`.

The real warm run reports 1,008 expert misses and 6,341,787,648 expert H2D
bytes, exactly 6,291,456 B per miss. In contrast, the old INT4 harness moved
196,608 B per expert miss. This 32x transfer-size difference alone makes the two
rates non-comparable; it is not a claim that transfer size explains the entire
65.65x headline-rate ratio.

## Why 30.571 does not appear in full-model decode

Measured facts support the following explanation:

1. **Different output unit.** The numerator is 32 synthetic loop iterations,
   not generated token IDs.
2. **One physical expert layer, not 40.** Every absent logical layer resolves
   back to the same layer-0 shard. A real model has distinct tensors for all 40
   layers.
3. **Experts are much smaller.** Historical `inter=64`; verified Ornith expert
   transfers correspond to `inter=512`. INT4 then reduces the old transfer by a
   further 4x.
4. **The dense model is absent.** No attention/linear-attention state, learned
   norms/residuals, embedding, LM head, sampling, or tokenizer executes.
5. **Layer dependencies are synthetic.** The 40 calls in one step do not form a
   sequential hidden-state stack.
6. **Routing and combination differ.** A CPU surrogate Oracle supplies IDs;
   selected experts are averaged equally rather than weighted by the genuine
   router.
7. **Startup is outside the metric.** File/model/CUDA construction is excluded.
   The later `dbdb7dd` prepack also moves quantization outside the timer, which
   is why `dbdb7dd` must not be used as an exact reproduction without disabling
   prepack.
8. **The old path is approximate.** INT4 validation was only synthetic: max
   absolute error 0.005194, relative RMSE 0.2586%, cosine 0.9999966553,
   10,146/10,240 ordered route matches, and 1,269/1,280 exact top-K sets. It was
   never validated as a real-checkpoint quality mode.

No defensible conversion factor maps 30.571 synthetic steps/s to real decode
TPS. The observed 0.465670 token/s is not a regression against the earlier
number; it is the first measurement of a materially broader workload.

## Practical reproduction command and status

For a faithful rerun, use a clean Linux/T4 checkout pinned to the historical
source commit. Do not use the current branch's defaults:

```bash
git switch --detach 75d5218ee7385985119f41e86375ba102755e274
git lfs pull
./scripts/setup_lightning_t4.sh

for run in 1 2 3; do
  ./build/dee_cli \
    --shard tests/data/ornith_moe256.safetensors \
    --oracle oracle.pt \
    --tokens 32 --warmup 2 --topk 8 --layers 40 \
    --cuda --transfer-dtype int4
done
```

Record the git commit, driver, CUDA runtime, GPU clocks/utilization, raw stdout,
and SHA-256 of the generated shard and `oracle.pt`. Do not enable profiling for
the headline rerun. Run the profiled command separately if stage attribution is
needed.

If reproduction must use `dbdb7dd`, add both `--transfer-dtype int4` and
`--dynamic-quantization`; otherwise its startup prepack changes the timing
boundary. Even with those flags, `dbdb7dd` is a later code state and should be
reported as a compatibility rerun, not an exact historical reproduction.

Current status: **inspected, not newly re-measured**. The historical source
commit and generated shard are available locally, but this Windows machine has
no NVIDIA runtime/toolkit. A current T4 rerun would also use a different host
state than the original Lightning measurement. Milestone 2.5 acceptance only
requires the prior result to be reproduced **or inspected**; the inspection is
sufficient to label it correctly. A Kaggle single-T4 rerun remains practical if
the primary investigation wants a contemporary variance estimate.

## Evidence inventory and uncertainties

Authoritative or tracked evidence:

- `dee.cpp/T4_PROFILE_RESULTS.md:1-11,306-375` — environment, base workload,
  INT4 commit/result/caveat, and later `dbdb7dd` distinction.
- `dee.cpp/README.md:76-85,105-195` — generated single-layer asset, profiling
  behavior, controlled scenarios, and explicit synthetic-only scope.
- Git object `75d5218`:
  - `dee.cpp/main.cpp:15-35,309-415` — CLI arguments/defaults and separate
    warmup Engine;
  - `dee.cpp/src/engine.cpp:231-235,383-445,584-670,745-823,861-1029` —
    physical-layer fallback, lazy INT4 staging, cache sizing, timing loop,
    transfer/compute/synchronization path;
  - `dee.cpp/tests/gen_moe_shard.py:1-24,42-99` — actual synthetic dimensions
    and repeated deterministic expert data;
  - `dee.cpp/src/swiglu_cuda.cu:30-56,110-165` — equal-mean combine and
    single-column FP16/FP32-accumulate expert projections.
- `benchmark_reports/kaggle-milestone2/v14-output/ornith-milestone2-evidence/ornith-milestone2-report.json`
  (SHA-256 `eaf29c0b13cc84b508d3d2727e2374dd91a568f31afad0e390b11045455a8609`)
  — verified full-model comparison.

Evidence limitations that must remain visible:

- The three unprofiled `int4-normal-*.log` files are not present in the current
  repository or retained campaign directory; the three values survive in the
  tracked campaign report and historical execution record.
- The local `campaign-transfer3/*.json` files are currently untracked and their
  schema omits git commit, driver version, command line, asset hashes, CPU
  identity, and wall-clock timestamp.
- The profiled 26.222-step/s artifact corroborates workload and counters but is
  not the raw source of the 30.571 median.
- No pure expert-kernel microbenchmark artifact exists, so the 79.096
  compute-only control is the closest supported proxy and must keep its wrapper
  caveat.
- No current rerun has quantified result drift across CUDA 13 Lightning and the
  present Kaggle CUDA 12.8 environment.

## Required label for future reports

Use:

> Historical single-T4 synthetic MoE harness: 30.571 recurrent steps/s median
> (32 steps, 40 logical calls/step, top-8, one physical 2048x64 expert layer,
> opt-in INT4 H2D, 6 MiB cache; not full-model token generation).

Do not shorten this to “dee.cpp runs Ornith at 30 TPS.”
