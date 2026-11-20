# DEE: Dynamic Expert Eviction

DEE runs Mixture-of-Experts models that are far larger than GPU memory by
keeping only a few experts in VRAM and streaming the rest from host RAM and
NVMe on demand. Its main target is **DeepSeek-V4-Flash-0731** (43 MoE layers,
256 routed experts per layer, top-6 routing, about 146.6 GiB of packed expert
weights) on a single 24 GB-class GPU.

Two things set DEE apart from other offloading engines:

1. **Exactness.** Caching, prefetching, eviction and transfer timing may move
   expert weights between tiers. They may never change which expert runs, the
   order experts run in, or the numbers they compute. A run passes only if its
   generated token ids **and** its complete per-layer route journal are
   byte-identical to the reference path on the same GPU.
2. **Preregistered evaluation.** Before any cloud GPU run existed, the expected
   speed, cache hit rate, cold reads per token and cost of every hardware cell
   were frozen in [`theory/PREDICTIONS_PHASE6.md`](dee.cpp/theory/PREDICTIONS_PHASE6.md)
   (registered 2026-09-30). Every run is graded against that file. Wrong
   predictions are marked `KILLED` in append-only addenda, never edited.

DEE is a research engine. It is correct, and it is slow next to engines that
approximate or skip experts. The point of this repository is a careful,
reproducible measurement of what exact expert offloading costs.

## Results (run r7, November 2026)

DeepSeek-V4-Flash, exact decode on Modal cloud GPUs, next-use eviction, aggregate
decode tokens/s. `b` is batch size; `host` is the host-RAM expert cache.

| GPU | b=1, 16 GiB host | b=1, 64 GiB host | b=8, 16 GiB host | b=8, 64 GiB host |
|---|---|---|---|---|
| 1x L4 (24 GB) | 0.77 | 0.73 | 1.20 | 1.87 |
| 2x L4 | 0.70 | 0.67 | 1.45 | 1.83 |
| 1x A10 (24 GB) | 0.41 | 0.76 | 2.47 | 3.23 |
| 1x L40S (48 GB) | 1.55 | 1.35 | 2.48 | 3.87 |
| 1x RTX PRO 6000 (96 GB) | 2.22 | 1.95 | 4.86 | **6.27** |
| 2x T4 (16 GB each) | 0.70 | 0.63 | not run | not run |

- **Exactness held everywhere it is defined.** Every GPU reproduced its own
  reference tokens and route journals. The 2x T4 run reproduced the 16 sealed
  tokens of the original Kaggle reference run exactly.
- Different GPU architectures can produce different (still exact on that GPU)
  text, because floating-point kernels differ across SM generations. That is
  floating-point drift in the dense layers, not an expert-placement effect.
- Raising host RAM from 16 to 64 GiB roughly doubles the host cache hit rate at
  b=8 (about 0.44 to 0.85).
- Against the frozen predictions, the b=8 / 16 GiB cells and most b=1 / 16 GiB
  cells were **KILLED** (too slow). The 64 GiB cells graded PARTIAL. Per-row
  grades are in each run's `scorecard.md`.
- The whole r7 campaign (6 GPU setups, 24 rows) cost $8.80 on Modal.

Per-run evidence (scorecards, summaries, decoded answers):
[`dee.cpp/runs/`](dee.cpp/runs/). Environment table (GPU, driver, VRAM budget,
container RAM, measured SSD bandwidth, price, commit):
[`dee.cpp/runs/r7_environment.json`](dee.cpp/runs/r7_environment.json).
Earlier results: Phase 3 full-expert-store inference
([`docs/phase3/RESULTS_PHASE3.md`](docs/phase3/RESULTS_PHASE3.md)), Phase 4
cache hierarchy ([`dee.cpp/PHASE4_RESULTS.md`](dee.cpp/PHASE4_RESULTS.md)),
and the graded addenda in `PREDICTIONS_PHASE6.md`.

## How it works

```text
DeepSeek-V4 / MiMo model (Python, pydee)                 dee_core (C++17 / CUDA)
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
- **Exactness gate**: the native router decides expert ids; prediction may only
  drive prefetch hints, never routing.

Engine details: [`dee.cpp/README.md`](dee.cpp/README.md). Theory and simulator:
[`dee.cpp/theory/`](dee.cpp/theory/).

## Build

Requirements: CMake 3.18+, a C++17 compiler, and Python 3.12. GPU builds need
CUDA 12 (the cloud runs used CUDA 12.8 with torch 2.10.0+cu128). The `ggml` backend is vendored in `dee.cpp/third_party/ggml`.

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

The hardware campaign runbook is
[`dee.cpp/modal/phase6/CAMPAIGN.md`](dee.cpp/modal/phase6/CAMPAIGN.md), and the
exact r7 plan (configs, cost caps, kill rules) is
[`dee.cpp/modal/phase6/R7_PLAN.md`](dee.cpp/modal/phase6/R7_PLAN.md). In short:

1. Stage the checkpoint and build the segmented expert store on Modal volumes
   (`modal_p6_ops.py::stage_ckpt`, `::build_store`).
2. Run the CPU `preflight` for the GPU you want (about $0.10).
3. Launch one container per GPU type with `modal_p6.py::bench_<GPU>`, passing
   the row configs from the plan and a `--max-usd` cap.
4. Score each row with `aws/phase6/score.py` against the frozen matrix.

The containers clone this repository at a pinned commit, so a run is tied to an
exact source revision (r7 used `7fc878a`).

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
| `paper/` | the arXiv paper (LaTeX source, figures, build script) |
| `dee.cpp/demo/` | a replay page of a real r7 run, built from its evidence |
| `docs/phase3/` | Phase 3 design notes and results |
| `archive/` | superseded prototype and supervisor scripts, kept for provenance |

Evidence directories are never edited after a run. Some older evidence files
refer to paths such as `tmp/` or `RESULTS.md` at the repository root; see
[`archive/README.md`](archive/README.md) for where those files live now.

## License

MIT, see [`LICENSE`](LICENSE). Third-party components and their licenses are
listed in [`NOTICE.md`](NOTICE.md).
