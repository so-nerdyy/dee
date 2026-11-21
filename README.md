# DEE: Dynamic Expert Eviction

DEE runs Mixture-of-Experts models that are far larger than GPU memory. It keeps
a few experts in VRAM and streams the rest from host RAM and NVMe on demand. The
test model is **DeepSeek-V4-Flash-0731**: 43 MoE layers, 256 routed experts per
layer, top-6 routing, and about 146.6 GiB of packed expert weights. That is six
times the memory of the 24 GB GPU it runs on.

This README is a summary of the experiments. Every number below links to the
evidence it came from, and the numbers that turned out wrong are reported next
to the ones that held. The full write-up is the paper,
[`paper/dee.pdf`](paper/dee.pdf): *Exact by Construction, Graded in Advance: A
Pre-Registered Hardware Evaluation of SSD-to-GPU Expert Offloading for
DeepSeek-V4-Flash*.

## The two rules every experiment follows

1. **Exactness.** Caching, prefetching, eviction and copy timing may move expert
   weights between tiers. They may never change which expert runs, in what
   order, or what it computes. A run passes only if its generated token ids
   **and** its full per-layer route journal are byte-identical to the reference
   path on the same GPU.
2. **Predictions frozen before the hardware.** Before any cloud GPU run, the
   expected speed, cache hit rate, cold reads per token and cost of every
   hardware cell were committed to
   [`PREDICTIONS_PHASE6.md`](dee.cpp/theory/PREDICTIONS_PHASE6.md)
   (registered 2026-09-30, commit `6107aab`). Each run is graded against that
   file by a script. A prediction that fails is marked `KILLED` in an
   append-only addendum and is never edited.

## Headline results

- **Exactness held on every run, on every GPU.** Six Modal GPU setups each
  reproduced their own reference tokens and route journals. The Modal 2x T4 run
  in November 2026 reproduced, token for token, the answer sealed on a Kaggle
  2x T4 in August 2026 (sealed by commit `e19c7e2`).
- **Fastest setup: 6.27 tokens/s** on one RTX PRO 6000 (batch 8, 64 GiB host
  cache), about 3.4x one L4.
- **Next-use eviction beat LRU by 1.30x**, against a prediction of 1.27x
  registered before the run. It removed a cliff where LRU got **0 hits out of
  49,463** host-cache lookups.
- **Most speed predictions were wrong, and the record says so.** 9 of the 12
  testable decode-speed rows were `KILLED`, measuring 2 to 6.5x below the
  registered nominal. Learned and heuristic prefetch hints were also killed.
- **A post-hoc speed model** fitted to r7 predicts each GPU from the other three
  with a **14% median error** (worst 56%).
- **First port to llama.cpp:** identical outputs, 2 to 3x the cache hit rate of
  llama.cpp's current policy, 4 to 30% faster, at very small cache sizes. This
  is early and has caveats (see below).
- **Total GPU spend for Phase 6: under $14** on Modal, across seven runs.

## Speed across GPUs (run r7, November 2026)

DeepSeek-V4-Flash, exact decode, next-use eviction. Aggregate decode tokens/s.
`b` is batch size and `host` is the host-RAM expert cache.

| GPU | b=1, 16 GiB host | b=1, 64 GiB host | b=8, 16 GiB host | b=8, 64 GiB host |
|---|---|---|---|---|
| 1x L4 (24 GB) | 0.77 | 0.73 | 1.20 | 1.87 |
| 2x L4 | 0.70 | 0.67 | 1.45 | 1.83 |
| 1x A10 (24 GB) | 0.41 | 0.76 | 2.47 | 3.23 |
| 1x L40S (48 GB) | 1.55 | 1.35 | 2.48 | 3.87 |
| 1x RTX PRO 6000 (96 GB) | 2.22 | 1.95 | 4.86 | **6.27** |
| 2x T4 (16 GB each) | 0.70 | 0.63 | not run | not run |

- 24 rows, 6 GPU setups, **$8.80** on the real Modal bill.
- Every row passed exactness on its own GPU. Different GPU generations can
  produce different (still exact on that GPU) text, because the dense-layer
  floating-point kernels differ between SM75, SM86 and SM89/Blackwell. That is
  floating-point drift, not an expert-placement effect.
- Going from 16 to 64 GiB of host RAM roughly doubles the host cache hit rate
  at b=8 (about 0.44 to 0.85).
- A second GPU buys nothing: 2x L4 is no faster than 1x L4 and gives
  byte-identical tokens.
- On the same L4 cell (b=8, 64 GiB), throughput went from 0.84 tokens/s in r5
  (LRU) to 1.87 in r7 (next-use plus harness fixes).

Per-row evidence (scorecards, summaries, decoded answers) is in
[`dee.cpp/runs/`](dee.cpp/runs/). The environment of each container (GPU,
driver, VRAM budget, RAM, measured disk bandwidth, price, commit) is in
[`r7_environment.json`](dee.cpp/runs/r7_environment.json). A replay page built
from a real r7 run is in [`dee.cpp/demo/`](dee.cpp/demo/).

## What the frozen predictions got right and wrong

| claim | predicted | measured | grade |
|---|---|---|---|
| Decode speed, testable rows | registered nominal per cell | 2 to 6.5x slower on 9 of 12 rows | **KILLED** (9 of 12) |
| L4, b=8, 64 GiB host, r5 | 9.0 tok/s (Addendum 3 nominal) | 0.84 tok/s | **KILLED** |
| LRU host hit, b=8, 16 GiB, r5/r6 | 0.51 | 0 of 49,463 | **KILLED** |
| Next-use vs LRU throughput, r6 | 1.27x | 1.30x | **PASS** |
| Next-use host hit, b=8, 16 GiB, r6 | simulator: cliff removed | 0.44 (21,703 of 49,463) | cliff removed |
| Prefetch hints on top of next-use, r6 | 0.87 total hit rate | hints added only 0.06 to 0.11, with 27% more reads | **KILLED** |
| Exactness on every GPU | identical tokens and journals | identical | **PASS** |

Why the misses matter: the LRU cliff was a real mechanism, not noise. At b=8
one decode step touches more distinct experts than a 16 GiB LRU cache holds, so
LRU evicts every record before it is used again. The correction is written up
in [`CORRECTION_HOST_HIT_BATCH.md`](dee.cpp/theory/CORRECTION_HOST_HIT_BATCH.md),
and the next-use simulator that predicted the fix is in
[`NEXT_USE_SIM.md`](dee.cpp/theory/NEXT_USE_SIM.md). The prefetch-hint
prediction was wrong because of a bug in the free trace simulator, which was
then fixed. On hardware the hints did not pay for their extra reads, so that
line of work was stopped. All addenda and per-row grades are in
`PREDICTIONS_PHASE6.md` and each run's `scorecard.md`.

## A speed model for new hardware

After r7, a two-term model was fitted to the batch-8 rows
([`TPS_MODEL.md`](dee.cpp/theory/r7/TPS_MODEL.md)):

    T_step = t_fixed(GPU) + cold_reads_per_step x c_read(container)

- The fixed part follows GPU memory bandwidth:
  `t_fixed ≈ 114 x BW^-0.62` seconds, with BW in GB/s.
- The read cost depends on the container, not the disk spec. The A10 had the
  slowest measured disk and still read records about 3x faster than the L4,
  because part of the "SSD" tier was served from the host page cache.
- Leave-one-GPU-out, the model predicts each single GPU with a **14% median
  error**. The 64 GiB rows land within 23%. The 16 GiB rows carry most of the
  error, because `c_read` varies 6x between containers.

This was fitted after the data, so it is a description of r7, not a tested law.
The next paid run is meant to test it as a frozen prediction.

## First run inside llama.cpp

llama.cpp recently merged a VRAM cache for MoE experts kept in host RAM
(`--moe-cache-mib`, using a ClassLRU policy). DEE's next-use rule was ported
into it as an opt-in policy, with a test that outputs do not change with cache
size. The patch and harness are on branch
[`claude/project-thread-atma5q`](https://github.com/so-nerdyy/dee/tree/claude/project-thread-atma5q/dee.cpp/modal/llamacpp).

First run (Qwen3-30B-A3B Q4_K_M with experts on the CPU, one L4, about $0.80):

| cache | requests | ClassLRU tok/s | next-use tok/s | change |
|---|---|---|---|---|
| 5% | 1 / 4 / 8 | 5.12 / 4.50 / 5.05 | 5.89 / 4.70 / 5.35 | +15% / +4% / +6% |
| 10% | 1 / 4 / 8 | 7.42 / 4.71 / 5.23 | 7.75 / 5.20 / 6.78 | +4% / +10% / +30% |

- Greedy text was identical across all four policy and size combinations, and
  the architecture tests passed.
- Trace-replay hit rate: 0.056 vs 0.116 at 5% cache, 0.084 vs 0.236 at 10%
  (2.1x and 2.8x).

Caveats, stated plainly: 5 to 10% caches are smaller than one token's expert
lookups, which is a thrashing regime real users rarely run in (they run 40 to
100%). The server logs did not print which policy was live, so the live
numbers are not yet confirmed. The model also fit in the L4's VRAM if loaded
normally, so this is not the case DEE is built for. A free replay on
DeepSeek-V4-Flash route journals
([`LLAMACPP_POLICY_SIM.md`](https://github.com/so-nerdyy/dee/blob/claude/project-thread-atma5q/dee.cpp/theory/LLAMACPP_POLICY_SIM.md))
found next-use **ties** ClassLRU at batch 1 and wins by 4.6 to 8.5 points only
with 4 or more parallel requests and a 5 to 10% cache. The next run uses
DeepSeek-V4-Flash at ~85 GiB, which does not fit in any single GPU tested, at
realistic cache sizes.

## What DEE is, and what it is not

- **It is not a cheaper way to serve a model.** On rented GPUs, exact
  offloading costs orders of magnitude more per token than hosted APIs. The
  paper says this directly.
- **It is a way to run a model that does not fit in your GPU's memory**, with a
  guarantee that the offloading changes nothing about the output.
- **The intended user is someone running big MoE models locally**, especially
  on unified-memory machines, who would rather run a higher-quality
  quantization slowly than a heavily degraded one quickly.
- **Its limit is single-user speed.** Batch 1 runs at 0.4 to 2.2 tokens/s on the
  GPUs tested. Other recent systems that also run DeepSeek-V4-Flash (for example
  SeqMoE) report higher speed. DEE's contribution is exactness and a
  preregistered evaluation, not being the fastest.
- **The evaluation is narrow.** One model, short generations (48 or 65
  tokens), an 8-prompt cohort, and an exactness reference that is DEE's own
  sealed path, not DeepSeek's official kernels (which cannot run on SM75).

## What I would run next with SSH access to hardware

Every run so far was in Modal containers. They are cheap, but under gVisor I
can't drop the page cache, can't do true O_DIRECT reads, and can't run full
GPU profilers, and every run pays for a cold container. That is the main
reason `c_read` varied 6x and why the disk bandwidth checks were often flagged.
With an SSH-able machine I would:

1. **Test the speed model as a frozen prediction.** Freeze predicted tokens/s
   for the machine from `TPS_MODEL.md` plus one cheap `c_read` measurement, then
   measure with a cold page cache and real NVMe reads.
2. **Run llama.cpp next-use on a model that does not fit in VRAM**
   (DeepSeek-V4-Flash, ~85 GiB quant) at 25 to 100% cache sizes, with the
   policy confirmed in the logs, then send the result upstream.
3. **Run the unified-memory case**: a big MoE model at a higher quantization on
   a UMA machine, measuring speed and output quality against a low quant that
   fits.
4. **Run long generations** (thousands of tokens and more prompts) to see
   whether cache behaviour and exactness hold past the 65-token runs.
5. **Settle the cross-GPU drift question**: run a large card with the L4's
   3,584 MiB VRAM budget. If its tokens then match the L4, the batch-8
   differences come from the budget, not the GPU.
6. **Profile the host-to-GPU copy path**, which the paper names as the main
   remaining cost, with Nsight Systems.

## How it works

```text
DeepSeek-V4 model (Python, pydee)                        dee_core (C++17 / CUDA)
  dense layers, attention, router  ──route──▶  segmented DEE4 expert store (NVMe, mmap)
                                                       │
                                                       ▼
                                              host-RAM pack cache (pinned)
                                                       │  async prefetcher
                                                       ▼
                                              VRAM expert cache ──▶ expert GEMMs
```

- **Expert store** (`dee4-v4-segmented`): 46 segments, 11,776 records, read
  with mmap and sha256-sealed against a metadata table.
- **Two cache tiers**: a pinned host-RAM pack cache and a VRAM expert cache.
  Eviction is LRU or next-use (a probabilistic Belady policy, after SeqMoE).
- **Async prefetcher** with a deadline queue and multi-lane pinned copies.
- **Exactness gate**: the native router decides expert ids. Prediction may only
  drive prefetch, never routing.

Engine details: [`dee.cpp/README.md`](dee.cpp/README.md). Theory and simulator:
[`dee.cpp/theory/`](dee.cpp/theory/).

## Build

Requirements: CMake 3.18+, a C++17 compiler, and Python 3.12. GPU builds need
CUDA 12 (the cloud runs used CUDA 12.8 with torch 2.10.0+cu128). The `ggml`
backend is vendored in `dee.cpp/third_party/ggml`.

CPU build and tests (no GPU needed):

```bash
cd dee.cpp
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DDEE_CUDA=OFF -DDEE_BUILD_TESTS=ON
cmake --build build-cpu --parallel "$(nproc)"
ctest --test-dir build-cpu --output-on-failure
```

`test_oracle` needs `oracle.pt` from Git LFS (`git lfs pull`), and the
DeepSeek-V4 Python reference tests need `torch`.

CUDA build, as used for the cloud runs (set the architecture for your GPU, for
example 89 for L4/L40S):

```bash
cd dee.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target dee_core --parallel "$(nproc)"
DEE_BUILD_DIR=build python pydee/setup.py build_ext --inplace
```

## Reproduce the cloud results

The campaign runbook is
[`dee.cpp/modal/phase6/CAMPAIGN.md`](dee.cpp/modal/phase6/CAMPAIGN.md), and the
exact r7 plan (configs, cost caps, kill rules) is
[`dee.cpp/modal/phase6/R7_PLAN.md`](dee.cpp/modal/phase6/R7_PLAN.md). In short:

1. Stage the checkpoint and build the segmented expert store on Modal volumes
   (`modal_p6_ops.py::stage_ckpt`, `::build_store`).
2. Run the CPU `preflight` for the GPU you want (about $0.10).
3. Launch one container per GPU type with `modal_p6.py::bench_<GPU>`, passing
   the row configs from the plan and a `--max-usd` cap.
4. Score each row with `aws/phase6/score.py` against the frozen matrix.

The containers clone this repository at a pinned commit, so each run is tied
to an exact source revision (r7 used `7fc878a`).

## Repository layout

| path | what it is |
|---|---|
| `dee.cpp/` | the engine: `src/`, `include/`, `pydee/` bindings, `tests/`, model scripts in `scripts/` |
| `dee.cpp/modal/phase6/` | Modal harness and campaign runbooks |
| `dee.cpp/aws/phase6/` | the shared scorer (`score.py`) and the paused AWS port |
| `dee.cpp/theory/` | performance model, simulator, and the frozen prediction matrix |
| `dee.cpp/runs/` | Phase 6 per-run evidence and scorecards |
| `dee.cpp/benchmark_reports/` | earlier Kaggle/T4 evidence bundles, kept byte-for-byte |
| `dee.cpp/kaggle/`, `dee.cpp/experiments/`, `research/` | earlier campaign drivers and research notes |
| `paper/` | the paper (LaTeX source, figures, build script, PDF) |
| `dee.cpp/demo/` | a replay page of a real r7 run, built from its evidence |
| `docs/phase3/` | Phase 3 design notes and results |
| `archive/` | superseded prototype and supervisor scripts, kept for provenance |

Evidence directories are never edited after a run. Some older evidence files
refer to paths such as `tmp/` or `RESULTS.md` at the repository root; see
[`archive/README.md`](archive/README.md) for where those files live now.

## License

MIT, see [`LICENSE`](LICENSE). Third-party components and their licenses are
listed in [`NOTICE.md`](NOTICE.md).
