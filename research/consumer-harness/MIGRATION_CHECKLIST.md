# Migration checklist: Kaggle dual-T4 → consumer GPU host

Target envelope: ~16 GB consumer NVIDIA GPU, ~32 GB system RAM, fast local
NVMe SSD. Candidate GPUs include 5070 Ti / 4090 / 5090 / similar — support is
"measure here, then decide", never assumed. Do NOT assume the consumer host
automatically outperforms dual T4: fewer GPUs, different PCIe, different CPU
and driver can all move the bottleneck.

Complete in order. Each step gates the next.

## 1. Clone / build dee

- [ ] `git clone https://github.com/so-nerdyy/dee.git` on the new host.
- [ ] `git worktree list` shows no stale checkouts; use a fresh worktree per track.
- [ ] CPU build first: `cmake -S dee.cpp -B dee.cpp/build-cpu -DDEE_CUDA=OFF -DDEE_BUILD_TESTS=ON`,
      `cmake --build dee.cpp/build-cpu --parallel`, `ctest --test-dir dee.cpp/build-cpu`.
- [ ] CUDA build per `dee.cpp/README.md` with the host's `CMAKE_CUDA_ARCHITECTURES`
      (do NOT reuse `sm_75`; query with `nvidia-smi` / `nvcc --list-gpu-arch`).
- [ ] Record CUDA runtime, driver, compiler versions in the run notes.

## 2. CUDA / compiler requirements

- [ ] `nvidia-smi` present; note GPU name, VRAM, driver version.
- [ ] `nvcc --version`; toolkit compatible with the GPU's compute capability
      (Blackwell-class cards need a recent toolkit — do not assume T4-era CUDA works).
- [ ] CMake ≥ project minimum; host compiler builds the CPU suite green.

## 3. Model storage layout

- [ ] NVMe path selected, e.g. `/mnt/nvme/dsv4` (Linux) or `D:\dsv4` (Windows).
- [ ] Free space ≥ 200 GB (166.88 GB checkpoint + headroom + temp).
- [ ] Layout: one canonical copy only; everything else symlinks/hardlinks.
- [ ] Filesystem noted (ext4/ntfs/…); record in `host.json`.

## 4. Checkpoint integrity

- [ ] Model `deepseek-ai/DeepSeek-V4-Flash-0731`, revision
      `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`, 48 shards, 72,317 tensors.
- [ ] Every shard: size check + header-pin check per
      `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/CHECKPOINT_DOWNLOAD_PLAN.md`.
- [ ] Full-file SHA256 streamed and recorded before any execution use.
- [ ] `model.safetensors.index.json` present; never mix revisions.

## 5. Host qualification

- [ ] `python tools/qualify_host.py --storage-path <path> --output host.json`.
- [ ] Inspect MEASURED vs DERIVED vs UNKNOWN; no UNKNOWN waved through silently.
- [ ] Confirm PCIe gen/width, VRAM free, RAM available, SSD throughput, H2D cost.
- [ ] Attach `host.json` to every later benchmark record (`BENCHMARK_SCHEMA.md`).

## 6. Bounded RAM / VRAM budgets

- [ ] `python tools/memory_budget.py --vram-total <measured> --codec mxfp4`
      (then `--codec fp16` / `--expert-bytes` for comparison).
- [ ] Set explicit `--budget` / `--ram-cache` bounds for every run; unbounded
      runs are not evidence.
- [ ] Worst-case working set (topk × layers) computed; full-model residency
      NOT assumed (623-ish MXFP4 slots per 16 GiB-style budget vs 11,008 routed
      experts — eviction is mandatory).

## 7. Correctness baseline

- [ ] Run the exactness contract (sealed DS10-style token/trace checks where
      available) on the new host BEFORE any timing.
- [ ] `exact_match: true` vs the pinned reference; record output text, token
      IDs, reference hash, evidence path.
- [ ] Any mismatch stops the line: no throughput numbers from a wrong run.

## 8. Current exact throughput

- [ ] Measure TTFT / prefill tok/s / decode tok/s / wall / per-token latency
      with the exact path only; 2+ warmup tokens, ≥32 measured tokens.
- [ ] Fill `research/consumer-harness/BENCHMARK_SCHEMA.md` v1 record.
- [ ] State prominently: this is the new host's measured baseline, not a
      comparison claim vs dual-T4 until both records exist side by side.

## 9. Profiler evidence

- [ ] `--profile-stages --profile-json` + optional `--profile-timeline`;
      capture H2D bytes/token, cache hits, stage breakdown.
- [ ] Export `cost_model` inputs (`t_h2d_ms`, SSD latency/bandwidth, CPU meta)
      via `host.json`; EXPORT ONLY — do not wire into scheduling yet.

## 10. Only then: new optimizations

- [ ] New kernels/caches/schedulers gated on steps 1–9 staying green.
- [ ] Each optimization: correctness re-baselined, then measured, then
      profiler-evidenced. No bandwidth-to-TPS leaps, no 20 TPS claims, no
      "4-bit weights ⇒ native FP4 exec" claims, no named-GPU claims without
      that host's measurement.
