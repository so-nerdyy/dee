# R10 — Benchmark evidence schema + external baseline plan

- Track: R10 — common benchmark evidence format + matched-baseline strategy
- Branch: `research/prior-art-r10` @ `dc78dc4` (worktree `.freebuff/wt/r10`)
- Scope: (a) standardize the record every dee benchmark/baseline must emit;
  (b) decide which external systems can run the canonical model as a matched
  comparison on the campaign hardware (2x Tesla T4 SM75 + Kaggle `/tmp`
  storage); (c) pin the derived-metrics recipe for SSD / H2D /
  activation-transfer bytes per generated token when a system does not
  publish them.
- Rules observed: read-only research; no code changed; no remote spend; no
  baseline was executed — every "runnable" claim below is a feasibility
  assessment with named blockers, not a measured result. Paper and vendor
  numbers are labeled `paper-reported` and are metadata, never dee evidence.

## Verdict

**One runnable matched baseline exists: llama.cpp static placement**
(`-cmoe`/`-ncmoe` + mmap GGUF) — it loads the canonical checkpoint's expert
weights on the *same E2M1+E8M0/32 grid* and streams them through the OS page
cache from the same `/tmp` device, making it the honest "unmanaged storage +
static split" control. It is **not** an exactness-matched run (dense path is
requantized to Q8_0, kernels differ, MTP is dropped) and it needs one
multi-hour CPU batch for GGUF conversion before any GPU cell.

**One conditional candidate: MoE-Infinity** — the only external system that
natively consumes the canonical FP4 checkpoint *and* offloads experts to an
SSD path. Its CUDA build targets `sm_80/90/120` (T4 = SM75 is outside), its
`transformers>=DeepseekV4ForCausalLM` dependency needs verification, and its
host-RAM budget on a 31.35 GiB host is unproven. Runnable-in-principle after
bring-up work; treat as a GPU-batch-2-class decision, not a free cell.

**Blocked for matched runs:** Mixtral-Offloading (Mixtral-8x7B + HQQ only),
Fiddler (Mixtral-8x7B only, research code), KTransformers (validated matrix
is SM_86/89/120 and its V4 path assumes the full ~147 GiB expert pool
resident in host RAM), FreeToken (Ampere+ GPU requirement plus a
host-RAM floor the campaign host cannot meet). All four remain *conceptual*
prior art; their published numbers are cross-tier metadata only.

The primary matched comparison for Phase-2 closure remains dee's own
internal arms (the four-arm A0/A1/A2/A3 GPU batch design); external
baselines are secondary controls for the bytes/token economics.

---

## 1. Canonical reference points (all tier-labeled)

Every number a baseline is compared against lives in
`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/` and
`RUN_REGISTRY.json`. This report extends that format; nothing here replaces it.

### 1.1 Model identity (checkpoint-pinned)

| Field | Value | Source tier |
|---|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` | pinned |
| Official revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` | pinned |
| Architecture | `DeepseekV4ForCausalLM` (`model_type: deepseek_v4`, `transformers_version: 4.57.1`) | `official-source/config.json` |
| Layers / routed experts / top-k | 43 / 256 / 6 (+1 shared) | config |
| Hash-routed layers | `num_hash_layers = 3` (`tid2eid` table on layers 0-2) | config + `model.safetensors.index.json` |
| Hidden / expert intermediate | 4096 / 2048 | config |
| Other arch facts | `hc_mult=4` hyper-connections, `compress_ratios` ∈ {0,4,128}, sliding_window 128, YARN 16x to 1M ctx, `num_nextn_predict_layers=1` (DSpark on layers 40-42) | config |
| Precision contract | dense FP8 `e4m3`/`ue8m0` block-128; `expert_dtype: fp4` (E2M1 + e8m0 block-32) | config |
| Checkpoint size | 48 shards, 72,317 tensors, **166.88 GB** compressed | `CAMPAIGN_DASHBOARD.md` ledger |
| Routed-expert pool | 66,048 tensors, **~147.17 GB** compressed (~571 GB FP16-expanded) | ledger |
| DEE4 record | 25,165,824 params / **13,369,344 B = 12.75 MiB** per expert | sealed dee4 metadata |
| Full expert universe | 46x256 = 11,776 records ≈ **146.6 GiB packed** (incl. 3 MTP buckets) | Phase-3 corrected count |
| Sealed trace bank | 2,364 records, **31,605,129,216 B (~29.43 GiB)**, data SHA256 `c83462ba…` | `v60-seal-20260901T041158Z.json` |

### 1.2 Sealed dee run — the numbers any baseline must sit next to

`v60-seal-20260901T041158Z` (pinned commit `011a3034…`), 2x Tesla T4 SM75,
Kaggle `/tmp` dee4-trace bank, 7-token prompt, 16 generated tokens:

| Metric | Value | Tier |
|---|---|---|
| Prefill | 94.755 s | sealed dee dual-T4 run |
| Decode wall | 72.607 s (15 decode steps) | sealed |
| Decode TPS | 0.207 | sealed |
| ITL median / p95 | 4,603.05 / 6,843.79 ms | sealed |
| `storage_bytes` (host.SSD_bytes) | 33,169,342,464 | sealed |
| `expert_h2d_bytes` | 59,413,364,736 | sealed |
| Per-GPU pread bandwidth | 96.2 / 98.1 MiB/s (cuda0/cuda1) | sealed |
| Source-read overlap | ~70% of read service time overlapped | sealed |
| Engine per-GPU (loads/hits/evictions) | cuda0 2,285/328/2,004; cuda1 2,159/327/1,878 | sealed |
| Expert-cache VRAM buffer | 3,756,785,664 B (~3.5 GiB) per GPU; 281 resident experts | sealed |
| Host prefetch ring | 548 / 441 MiB | sealed |
| Exactness | 16/16 exact token IDs; text `**Alan Turing (1912–1954)** was an English mathematician, computer`; 43 layers; route journal SHA256 `f20f63ff…`, final chain `d8539b6e…` | sealed |

A/B replication (`v63-v64-terminal-ab-seal-20260903T013055Z`, commits
`236bdb29`/`c3182fc4`): decode wall 71.315 / 71.804 s, TPS 0.210 / 0.209,
median ITL 4,426.9 / 4,543.6 ms — same storage/H2D totals; the seal itself
warns the n=1 pair is mechanistic, not a speedup claim. Treat ~0.21 tok/s,
~2.07 GB/tok SSD, ~3.71 GB/tok H2D (16-token window) as the dee reference row.

### 1.3 Storage and memory envelope (Kaggle)

| Quantity | Value | Tier |
|---|---|---|
| `/tmp` expert-bank throughput | 0.29-0.37 GiB/s, 3 lanes QD6, ~96% device busy | Phase-1 measured |
| `/kaggle/working` pread | 121-128 MB/s (1-256 MiB) | `STORAGE_ROOFLINE.md` measured |
| `/kaggle/input` loop device | ~13 MB/s effective | `STORAGE_ROOFLINE.md` measured |
| Host RAM | 31.35 GiB MemTotal; v60 envelope 17 GiB packs -> 22.5-22.9 GiB peak RSS | sealed env |
| VRAM | 2x 15.6 GiB cuda_total | sealed env |
| Cold expert bytes/token (dee) | ~3,289 MiB/token at 0% host hit; ~1,700 MiB/token at ~50% hit | `STORAGE_ROOFLINE.md` derived |
| Host LRU knee | ~16 GiB pooled (~2.8 pp below offline MIN; reaches MIN ~32 GiB); 935/2,364 records ever repeat | ws-policy sim, sealed-trace |

Cross-system consistency note: FreeToken issue #151 reports ~3.07 GB of
expert weights moved per token on the same checkpoint family (2x RTX 3090)
— independently corroborating dee's ~3.3-3.4 GB/tok compulsory-miss
physics. That is the property a matched baseline must reproduce, not the
absolute TPS.

---

## 2. Common benchmark record schema (`r10-bench-record-v1`)

Extends `RUN_REGISTRY.json`'s required-field list. One JSON object per run;
every field carries a `status` of `measured` | `derived` | `unavailable` |
`paper_metadata` and every byte/latency field carries a `tier` label.

```json
{
  "schema": "r10-bench-record-v1",
  "identity": {
    "system": "dee | llamacpp | moe-infinity | ktransformers | fiddler | mixtral-offloading | freetoken | other",
    "system_version": "release tag or 'git:<sha>' (always a pinned commit)",
    "runtime_sha256": null, "harness_sha256": null,
    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "model_revision": "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
    "total_params": "~284B", "active_params_per_token": "~13B",
    "expert_record_bytes": 13369344,
    "weights_representation": "e.g. official-fp4/fp8 | gguf-mxfp4+q8_0 | hqq-2bit | ftfp4"
  },
  "hardware": {
    "gpus": "2x Tesla T4 (SM75, 2x15.6GiB)",
    "gpu_vram_total_gib": 31.2,
    "cpu_model": "REQUIRED — currently unrecorded by dee environment.json (schema gap, fix forward)",
    "cpu_threads": null, "host_ram_gib": 31.35,
    "storage": {"medium": "kaggle-/tmp|kaggle-working|kaggle-input-loop|nvme|other",
                "measured_bandwidth_mib_s": null, "measurement_method": "pread probe|iostat|cgroup io.stat"}
  },
  "cache_state": {
    "regime": "A|B|C", "contract": "for C: C1|C2|C3",
    "initial_records": 0, "initial_state_gib": 0.0,
    "cache_budget_gib": {"host": null, "vram": null},
    "prewarm_cost": {"bytes": 0, "seconds": 0.0, "charged_outside_window": true}
  },
  "workload": {
    "prompt": "exact prompt string + SHA256", "prompt_tokens": 7,
    "generated_tokens": 16, "concurrency": 1, "batch": 1,
    "reasoning_effort": null, "context_length": null,
    "sampling": "greedy|sampling-params"
  },
  "timing": {
    "prefill_s": null, "decode_wall_s": null,
    "ttft_ms": null, "tpot_ms": null,
    "itl_ms": {"p50": null, "p95": null, "max": null},
    "decode_tps": null, "total_wall_s": null, "conversion_or_load_s": null
  },
  "bytes": {
    "ssd_bytes": null, "ssd_bytes_per_token": null,
    "h2d_bytes": null, "h2d_bytes_per_token": null,
    "activation_transfer_bytes": {"d2h": null, "h2d": null, "gpu_gpu": null,
                                  "control_metadata": null, "per_token": null},
    "logical_vs_physical": "which bytes are counted (see §5)",
    "per_gpu": true
  },
  "memory": {"peak_vram_bytes_per_gpu": null, "peak_rss_bytes": null,
             "mapped_bytes": null, "kernel_count": null, "gpu_busy_pct": null},
  "power": {"gpu_w_avg": null, "gpu_w_peak": null, "method": "nvidia-smi power.draw sampling|null"},
  "exactness": {
    "verdict": "exact-16-token|tolerance-pass|non-exact|unverified",
    "token_ids_sha256": null, "route_journal_sha256": null,
    "artifact_hashes": {}
  },
  "comparability": {
    "matched_to_dee_v60": false,
    "notes": "why comparable or why not"
  }
}
```

Field-by-field rules:

- `regime` is mandatory and follows `research/phase2-regime-c/CONTRACT.md`:
  **A** cold, **B** causal warmup, **C** prewarmed. Every C row must carry
  `initial_records`, `initial_state_gib`, `contract` (C1/C2/C3), and a
  separately-stated `prewarm_cost` paid outside the timed window. C rows are
  never scored against A/B rows without explicit labels.
- `weights_representation` names the *stored* precision, not the compute
  precision (dee: packed FP4 store + FP16-expanded compute; llama.cpp:
  MXFP4 store + f32/bf16 kernel path). A run whose stored bytes differ from
  the checkpoint grid is `non-exact` by definition.
- `ssd_bytes` is *logical cold-tier payload bytes* per
  `PHASE2_METRICS.md`: successfully-materialized record bytes, not physical
  device sectors; partial failed preads are unattributed; host counters are
  lifetime values — use snapshot deltas with matching token deltas.
- `activation_transfer_bytes` separates hidden-state/expert-IO movement from
  expert-weight movement (dee v60 measured **0** — `bridge_counters_zero`
  gate: no numpy bridge, no hidden D2H/H2D, no raw expert-output D2H — that
  is a measured zero, not a missing field).
- `conversion_or_load_s` records format conversion or model-load wall time
  separately from inference; for llama.cpp this is where the GGUF build cost
  lands (paid once, outside the window).
- Never emit `0` for a metric that was not instrumented — emit
  `"unavailable"` with a note. A missing metric is not zero bytes.

---

## 3. Baseline feasibility matrix (matched run = canonical checkpoint, 2xT4
SM75, Kaggle `/tmp` storage, ~31 GiB host RAM, single request, greedy
decode, comparable workload)

| Baseline | Matched on 2xT4+/tmp? | Model support for V4-Flash-0731 | Memory floor vs 31.35 GiB host | Code state | License | Exactness status |
|---|---|---|---|---|---|---|
| **llama.cpp** static (`-cmoe`/`-ncmoe`) | **YES — conditional** (needs GGUF build + pin) | Native `deepseek4` arch + `DeepseekV4ForCausalLM` converter → MOSTLY_MXFP4_MOE | Experts mmap'd (~147 GiB file, page-cache streamed); non-expert ~10-14 GiB on GPUs; host RSS can stay small | Active dev tree, `deepseek4` unreleased — pin commit | MIT | Non-exact (dense →Q8_0 requant, different kernels, MTP dropped); expert weights same E2M1 grid |
| **MoE-Infinity** | **Conditional — bring-up risk** | Native V4-Flash FP4 offload path (2026-06-23 merge); needs `transformers` shipping `DeepseekV4ForCausalLM` | Expert pool can live in `--offload-dir` (SSD); pinned buffers + cache must fit ~25 GiB — unverified | Active (349★, Apache-2.0); from-source builds target sm_80/90 (+sm_120 flag) — SM75 not a target | Apache-2.0 | Potentially exact-capable (byte-preserving FP4, native router); kernel numerics differ → verify tokens |
| **KTransformers** | **No** | V4-Flash MXFP4 path exists (PR #1970, doc `DeepSeek-V4-Flash.md`) | **Full expert pool resident in host RAM, no eviction** → ~147+ GiB required >> 31.35 GiB | Mature; SGLang-coupled launch; validated GPU matrix SM_86/89/120 only | Apache-2.0 | Exact-grid FP4 but SGLang-attention path unverified vs dee |
| **Mixtral-Offloading** | **No** | Mixtral-8x7B only, HQQ-only; DeepSeek-V2 requested (issue #36, open) | ~16 GB VRAM + ~11 GB RAM for Mixtral — irrelevant, no V4 | Research repo (2K★), notebook entry point | MIT (verify at pin) | Would be non-exact anyway (HQQ requant changes weights) |
| **Fiddler-style hybrid** | **No** | Mixtral-8x7B only (>90 GB unquantized, >3 tok/s on 24 GB GPU) | Would need ~147 GiB host + activation path | ICLR'25 artifact code | MIT | Concept only; its *insight* (move activations, not weights) maps to dee's CPU-sink R6 track |
| **FreeToken** | **No** | Native `DeepSeek-V4-Flash-0731` (fp8 dense + fp4 experts), FTW format | Demonstrated on 503 GB host; expert-pool residency assumption >> 31.35 GiB | Active serving engine (1.1K★); `freetoken[accel]` PyPI | Apache-2.0 | Native-format, potentially exact-capable; but Ampere+ GPU required — T4 SM75 unsupported (only an unofficial sm75 patch lab exists, unresolved provenance) |

### 3.1 What "runnable" means here

- **Runnable matched** — same checkpoint revision, same 2xT4 + `/tmp`
  regime, same single-request greedy 16-token workload; only llama.cpp
  qualifies, after a one-time GGUF conversion.
- **Runnable after adapter/port** — MoE-Infinity: needs an SM75-capable
  build of its CUDA ops (or a pure fallback path), a `transformers` build
  that ships `DeepseekV4ForCausalLM`, and proof its host-side budget fits
  31 GiB. Its SSD `offload_path` means the *physics* can match even if the
  plumbing is heavy.
- **Conceptually comparable, not executable as-is** — Fiddler (activation
  movement vs weight movement), Mixtral-Offloading (LRU expert cache +
  speculative prefetch), MoE-Infinity's activation-aware tracing (when its
  model constraint blocks the run).
- **Not fair — changes the model or its representation** — Mixtral-Offloading's
  HQQ requant on any port; any system substituting predicted routing for the
  authoritative router (dee contract forbids this in exact mode).
- **Blocked by floor** — KTransformers (host-RAM residency + SM86+),
  FreeToken (Ampere+ + RAM floor). Their numbers are `paper_metadata` /
  different-tier only.

---

## 4. Baseline notes

### 4.1 llama.cpp static placement — the matched control

Source basis (local clone `C:/Users/carth/Downloads/llama.cpp`, dev tree,
unreleased — pin the exact commit at run time):

- `src/llama-arch.cpp:80` registers `deepseek4`; `src/models/deepseek4.cpp`
  implements the full V4 graph: hash-layer `ffn_gate_tid2eid` I32 routing
  table (lines 131-136), compressor/indexer CSA+HCA attention,
  hyper-connection (`hc_*`) tensors, dedicated `llama_kv_cache_dsv4`
  (`src/llama-model.cpp:2193-2209`).
- `conversion/deepseek.py::DeepseekV4Model` (line 475+) produces
  `MOSTLY_MXFP4_MOE` GGUFs: routed experts repacked to GGUF MXFP4
  (`_pack_mxfp4_blocks`, lines 599-622 — safetensors' adjacent low/high
  nibbles → ggml's 0..15-low/16..31-high block layout; **a nibble-order
  repack on the same E2M1+E8M0/32 grid** — value-identical to dee's packed
  records); FP8 dense dequantized then forced `Q8_0` (lines 569-597,
  761-771); `tid2eid` → I32, never quantized (`src/llama-quant.cpp:309`);
  MTP tensors skipped in conversion v0.
- Placement controls: `-cmoe`/`--cpu-moe` pins every
  `ffn_(up|down|gate|gate_up)_(ch|)exps` tensor to the CPU buffer type
  (`common/common.h:1073-1081`, `common/arg.cpp:2510-2530`); `-ncmoe N` pins
  the first N layers' experts; `-ot` arbitrary tensor overrides;
  `llama-bench` carries `n_cpu_moe` (`tools/llama-bench/llama-bench.cpp:1221`).
  Shared-expert `*_shexp` tensors do **not** match the exps regex — they stay
  on GPU (~1.08 GB), which is the right static choice anyway.
- CPU MXFP4 kernels exist (`ggml/src/ggml-cpu/repack.cpp` generic
  gemv/gemm + repack); CUDA MXFP4 exists (`mmq-instance-mxfp4.cu`) with
  mmvq/dequant fallback — T4 SM75 is a supported ggml target.

Matched-run shape on 2xT4 + Kaggle `/tmp`:

- Build the GGUF once (CPU batch): reads the 166.88 GB checkpoint — at the
  ~13 MB/s loop-mount rate this is the dominant one-time cost (~3.5+ h of
  reads plus repack compute and a ~150+ GiB write to `/tmp`; `/tmp` has the
  headroom). Record under `conversion_or_load_s`.
- Run `llama-cli`/`llama-bench` with `-cmoe` (all routed experts on CPU),
  `-ngl` for the non-expert layers across the two T4s (`-ts`), mmap enabled.
  Expert tensors then stream through the OS page cache at the same
  ~0.29-0.37 GiB/s ceiling dee measured — this is the *unmanaged* version of
  dee's hierarchy and the honest control: page-fault granularity is 4 KiB
  pages (finer than dee's 12.75 MiB records, but with kernel readahead
  overshoot and no application-level dedup/lease/pinning).
- Known risks: cgroup page-cache accounting on Kaggle can count mmap pages
  against the memory limit; `llama-bench`'s direct-IO/`--no-mmap` knobs are
  the fallback arms (direct-IO arm = pure streaming, no page cache =
  regime-A-pure). CPU speed (Kaggle vCPU count) bounds the MXFP4 gemv — with
  43x6 experts/token this is secondary to storage but must be recorded.
- Metrics it cannot self-report: per-token SSD bytes (page faults are
  invisible to the app) — derive from cgroup `io.stat`/`/proc/diskstats`
  deltas over the decode window (§5). `h2d_bytes` for CPU experts is 0 *by
  construction* — report `measured:0` with the note "static CPU placement;
  no expert H2D"; activation traffic is the D2H/H2D of hidden rows at the
  MoE subgraph boundary (~8 KiB each way per layer-token ≈ ~0.7 MB/token —
  negligible vs weights, but instrumented via ggml backend-copy accounting
  if needed).
- Exactness verdict: **non-exact systems baseline.** Same expert weight
  values (MXFP4 grid), but the dense path is requantized (FP8→Q8_0),
  reduction order differs, MTP/DSpark is absent, and hash-layer routing is
  table-identical but implemented differently. A 16-token output match
  against the sealed text is a bonus observation, never a gate.

### 4.2 MoE-Infinity — conditional, most-capable external candidate

- Repo: `EfficientMoE/MoE-Infinity`, Apache-2.0, active. Paper arXiv
  2401.14361 (`paper-reported`: 4-20x latency reduction vs baselines on
  Switch/NLLB/Mixtral/DSv2-Lite, cluster hardware — different tier).
- Mechanics closest to dee's: expert offload to host memory **and SSD**
  (`offload_path`/`--offload-dir` — a filesystem path, so `/tmp` residency
  is expressible), activation-aware cache + sequence-level tracing +
  prefetch, pinned packed-FP4 expert tensors with byte-preserving H2D on a
  dedicated async copy stream (`models/deepseek_v4/official_offload_adapter.py`
  per the dee host-tier design doc).
- V4-Flash support merged 2026-06-23 (`6285c09`), gated on the installed
  `transformers` shipping `DeepseekV4ForCausalLM` — the canonical config
  declares `transformers_version: 4.57.1`, and dee's campaign notes that
  transformers 5.x shipped no `deepseek_v4` module as of Aug 2026, so the
  concrete dependency version must be pinned and verified.
- Blockers for a matched run: from-source build targets `sm_80`/`sm_90`
  (sm_120 via `MOE_ENABLE_SM120=1`) — SM75 needs a custom arch build or a
  fallback path (FlashAttention/FlashInfer are optional with graceful
  fallback; FP4 has a Triton fallback); host-side pinned buffers + cache
  budget vs 31.35 GiB is unverified; PyTorch-eager non-expert path on T4 is
  viable but slow.
- Verdict: schedule as a bring-up experiment only if a matched external
  number is wanted beyond llama.cpp; it is the one system whose *concepts*
  (SSD-tier offload + activation-aware prefetch + pinned FP4 transfer) map
  1:1 onto dee's tiers.

### 4.3 KTransformers — blocked by floors, already mined for components

- Audited in-repo at `research/kt-cpu-bridge/` (`KT_CPU_AUDIT.md`,
  `FORMAT_COMPATIBILITY.md`, `SUMMARY.md`, `CPU_EXECUTOR_DESIGN.md`) against
  upstream pin `31985f40…`.
- V4-Flash MXFP4 expert path exists (PR #1970, `doc/en/DeepSeek-V4-Flash.md`)
  — but the validated GPU matrix is SM_86/89/120 with `triton_kernels` +
  flashinfer ≥0.6.9 + CUDA ≥12.8 + `transformers==4.57.1`; T4 is outside.
- Structural blocker independent of GPU: the loader walks the full expert
  pool into C++ NUMA-resident copies with **no eviction** — ~147+ GiB host
  RAM floor vs 31.35 GiB. Its launch path is SGLang-coupled.
- Role for dee: component donor, not baseline — the kt_cpu_bridge already
  reuses its MXFP4 packing/kernel semantics per-expert. Any future
  "KTransformers-as-baseline" claim needs a ≥256 GiB host + SM86+ GPU — a
  different tier; vendor numbers (~29 tok/s class on 8x5090) are
  `paper_metadata`.

### 4.4 Fiddler-style hybrid — concept source, not a runnable baseline

- `efeslab/fiddler`, ICLR'25 (arXiv 2402.07033): Mixtral-8x7B only; moves
  *activations* to the CPU (batch×hidden, ~KB) instead of expert weights
  (~MB) and executes experts on AVX512_BF16 CPU; offline expert-popularity
  pinning; `paper-reported` >3 tok/s on a 24 GB GPU.
- In V4 units its economic argument is ~43x(2x4096 B bf16) ≈ 0.7 MB/token of
  activation traffic vs ~3.4 GB/token of cold expert weights — a ~4,700x
  ratio. That asymmetry is exactly dee's R6 CPU-sink track
  (`research/prior-art/r06-heterogeneous-sink-audit.md`: host tier admits a
  CPU sink today; the device-tier `stage()` seam is the gap, G1-G5).
- Matched run: impossible without porting the entire V4 arch (attention,
  hash layers, hyper-connections, FP4) — classify as concept baseline.

### 4.5 Mixtral-Offloading — LRU-cache ancestor, wrong model family

- `dvmazur/mixtral-offloading` (arXiv 2312.17238, ~2K★): Mixtral-8x7B via
  HQQ (2-bit experts / 4-bit attention), per-layer LRU expert cache, async
  swap-in, `offload_per_layer` knob; demo floor ~16 GB VRAM + ~11 GB RAM.
- DeepSeek-V2 support was requested (issue #36, still open); no DeepSeek
  arch of any generation is in the shipped path, and HQQ cannot represent
  the checkpoint's E2M1 grid (any port is requantized → non-exact anyway).
- Role: the LRU-expert-cache prior art dee's host LRU is compared against
  conceptually; its Colab-era numbers are `paper_metadata`.

### 4.6 FreeToken — strongest external system, wrong hardware envelope

- `FlashML-org/FreeToken`, Apache-2.0 (arXiv 2608.16157): edge-native
  serving; bandwidth-adaptive `q*` CPU/GPU miss split, global LRU expert
  cache, FTW fast-weight format, double-buffered prefill streaming, elastic
  VRAM reallocation; **native `DeepSeek-V4-Flash-0731` support** (fp8 dense +
  fp4 experts — the canonical checkpoint).
- Blockers on the campaign host: documented GPU support is Ampere-and-up
  (RTX 30/40/50, driver r580+/CUDA 13); T4 SM75 is unsupported (a
  third-party `sm75` patch lab exists with unresolved binary provenance —
  not usable for evidence). Demonstrated V4-Flash config used a 503 GB RAM
  host; the ~147 GiB expert pool plus runtime exceeds the 31.35 GiB
  campaign host unless an SSD-resident pool mode is proven.
- `paper_metadata` worth quoting (with tier labels): issue #151 measures
  5.58 tok/s (`offload`) vs 0.67 tok (`hybrid`) on 2x RTX 3090 + Xeon 6252
  + 503 GB RAM, ~3.07 GB expert bytes/token — same physics as dee's
  ~3.3-3.4 GB/tok. Useful as a *cross-system consistency check* on
  bytes/token, not a TPS comparison.

---

## 5. Derived-metrics recipe (when a system publishes none of them)

General rule: instrument the **logical payload** first, the physical device
second, and never conflate the two. All ratios divide by *generated decode
tokens* unless stated; prefill is always reported separately.

### 5.1 SSD bytes/token

1. Preferred: instrument the cold-tier read path directly (dee does:
   `expert_store.source_reads` × 13,369,344 B; `PHASE2_METRICS.md` defines
   `host.SSD_bytes` as logical materialized bytes).
2. If only OS visibility exists (llama.cpp page faults, mmap'd stores):
   snapshot cgroup `io.stat` / `/proc/diskstats` read-bytes deltas across the
   decode window; mark `derived` and note that page-cache hits do not appear
   as device reads (that *is* the intended signal for an mmap baseline).
3. If only expert counts exist:
   `SSD_bytes = cold_expert_loads × expert_record_bytes`, adjusted for
   cache hits, dedup/coalescing, and partial reads where evidence exists;
   label `derived`, name the record size and store layout.
4. Separate logical-requested bytes, physical device traffic (iostat),
   page-cache hits, prefetch/readahead reads, and duplicate/coalesced reads
   when the platform exposes them; record which are included.
5. Dee reference: 33,169,342,464 B / 16 tok = **2.073 GB/tok** (sealed;
   whole-run counter including prefill touches — the decode-only split needs
   snapshot deltas, recorded as a known limitation).

### 5.2 H2D bytes/token

1. Count submitted host→device transfer payload bytes (dee:
   `engine_stats.h2d_bytes` per GPU; pooled 59,413,364,736 B →
   **3.713 GB/tok** sealed).
2. State whether bytes are packed or expanded (dee moves packed 12.75 MiB
   records; a system expanding on host moves more bytes for the same expert).
3. Never infer H2D from model size or VRAM; exclude device-resident hits;
   include or separately report router/control-metadata transfers.
4. Multi-GPU: report per-GPU and pooled (dee: cuda0 30.55 GB, cuda1 28.86 GB).
5. For CPU-execution systems (llama.cpp `-cmoe`, Fiddler): expert-weight H2D
   is a *measured zero*, not a missing field — say so; their traffic is
   activation-side (5.3).

### 5.3 Activation-transfer bytes/token

1. Instrument D2H/H2D of hidden states, expert inputs/outputs, routing
   metadata, and inter-GPU handoffs separately. Dee's gate is
   `bridge_counters_zero` (v60: all zeros — experts execute where they land;
   route IDs are the only cross-tier control traffic).
2. State dtype, shape, batching, dedup: e.g. a Fiddler-shape V4 run moves
   ~4096 B bf16 each way per token-layer → ~0.35 MB/token one-way; GPU-GPU
   expert handoffs (dee d2d gather/scatter: ~23.8/47.6 MB per GPU per run)
   are reported separately from host-device traffic.
3. If only shapes are known: `bytes = element_count × bytes_per_element ×
   transfer_count`, marked `derived`; route IDs (6x43x8 B ≈ 2 KiB/token) are
   control-plane, not activation payload.
4. If instrumentation cannot distinguish host-device vs GPU-GPU vs control
   traffic, report `unavailable` with a note — never collapse to zero.

### 5.4 Comparability rules (binding)

- Same model revision, prompt/workload, concurrency, precision family,
  cache regime, and hardware class wherever possible; every deviation is a
  `comparability.notes` entry.
- Prefill and decode are separate rows; never blend them into one TPS.
- Regime A/B rows never sit next to regime-C rows without both labels and a
  caveat (CONTRACT.md §5).
- Service-time sums (per-read ms, nested waits) are not critical-path wall
  time — dee's own profile keeps `read_milliseconds` (sum) distinct from
  `decode_wall_s`.
- Paper/vendor speedups are `paper_metadata`, never measured dee evidence.
- Missing metric → `unavailable`, never zero.
- Different representation, routing, precision, or expert identity →
  `non-exact` or `non-comparable`; that label propagates to every derived
  ratio.
- Power: sample `nvidia-smi --query-gpu=power.draw` (T4 reports draw;
  70 W cap) at ≥1 Hz over the window; report avg/peak or `unavailable`.

---

## 6. Recommended plan

1. **Now (no new compute):** adopt `r10-bench-record-v1` as the schema for
   all future benchmark rows; backfill the dee v60/v63/v64 rows into it
   (all fields are already available in the sealed bundles).
2. **CPU batch candidate:** llama.cpp GGUF conversion of the canonical
   checkpoint on Kaggle CPU (est. multi-hour; `/tmp` fits ~150+ GiB output).
   Deliverable: `dsv4-flash.mxfp4.gguf` + manifest + conversion metrics.
3. **GPU-batch arm (if a matched external control is wanted):**
   `llama-cli -m dsv4-flash.mxfp4.gguf -p "<canonical 7-token prompt>" -n 16
   -cmoe -ngl <fit> -ts 1,1` with cgroup `io.stat` deltas + wall/ITL capture.
   Produces the first external row in the matrix — the "OS page cache +
   static placement" control against dee's managed hierarchy at identical
   storage physics.
4. **MoE-Infinity:** feasibility spike only (build for sm_75 or confirm
   fallback; verify `transformers` V4 class; measure host-RAM floor under
   `--offload-dir` on `/tmp`). Do not spend a GPU cell until the spike clears.
5. **KTransformers/FreeToken:** keep as component donors and different-tier
   metadata; revisit only on ≥256 GiB + SM86+ hardware.
6. **Fiddler/Mixtral-Offloading:** cite as design ancestors (activation
   movement, LRU expert cache); no port is planned — their role is closed by
   this report.

## 7. Runnable-baseline conclusion

| Status | Systems |
|---|---|
| Runnable matched (after one CPU conversion batch) | **llama.cpp** `-cmoe`/`-ncmoe` |
| Runnable after bring-up (conditional) | **MoE-Infinity** |
| Concept only / not executable on canonical model | **Mixtral-Offloading**, **Fiddler** |
| Blocked by hardware matrix + memory floor | **KTransformers**, **FreeToken** |

No external system produces a dee-style route-journal/exactness artifact;
all external comparisons are systems-level (bytes/token, TPS, latency,
memory), and llama.cpp/MoE-Infinity token-parity observations are recorded
as bonuses, never as gates.

## 8. Evidence inventory

- Sealed dee runs: `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/`
  `v60-seal-20260901T041158Z.json`, `v63-v64-terminal-ab-seal-20260903T013055Z.json`,
  `v60-evidence-20260901T040935Z/` (integrity/profile/memory/run_config),
  `RUN_REGISTRY.json`, `STORAGE_ROOFLINE.md`, `CAMPAIGN_DASHBOARD.md`,
  `official-source/config.json`.
- Metrics/regime contracts: `dee.cpp/benchmark_reports/PHASE2_METRICS.md`,
  `research/phase2-regime-c/CONTRACT.md`.
- Component audits: `research/kt-cpu-bridge/{KT_CPU_AUDIT,FORMAT_COMPATIBILITY,SUMMARY,CPU_EXECUTOR_DESIGN,THIRD_PARTY_KTRANSFORMERS}.md`,
  `research/prior-art/r06-heterogeneous-sink-audit.md`,
  `dee.cpp/third_party/README.md`.
- llama.cpp source (local clone, pin at use): `src/models/deepseek4.cpp`,
  `src/models/models.h:1088`, `src/llama-arch.cpp:80`,
  `conversion/deepseek.py:475-776`, `common/common.h:1073-1081`,
  `common/arg.cpp:2510-2530`, `tools/llama-bench/llama-bench.cpp:1221`,
  `ggml/src/ggml-cpu/repack.cpp`, `ggml/src/ggml-cuda/mmq-instance-mxfp4.cu`.
- External (`paper-reported`/metadata): arXiv 2401.14361 (MoE-Infinity),
  arXiv 2402.07033 (Fiddler), arXiv 2312.17238 (Mixtral-Offloading),
  arXiv 2608.16157 + `FlashML-org/FreeToken` issue #151 (FreeToken),
  `EfficientMoE/MoE-Infinity` commit `6285c09` (V4-Flash FP4 offload).
