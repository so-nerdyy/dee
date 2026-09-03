# Sherry / STQ Ultra-Low-Bit Expert Storage — Audit

Branch: `research/ultralowbit-storage` (parallel research track; **no changes to the
MXFP4 runtime on `freebuff/deepseek-v4-flash-0731-t4`**).
Date: 2026-09-02.
Evidence discipline: every claim below is tagged **[VERIFIED]** (source read
directly in this audit), **[INFERENCE]** (derived from verified facts, reasoning
shown), or **[HYPOTHESIS]** (plausible, unverified, testable).

---

## 1. What Sherry / STQ1_0 is

**[VERIFIED]** Sherry is a ternary weight-quantization framework from Tencent
(paper: "Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via
Fine-grained Sparsification", arXiv:2601.07892, accepted ACL 2026; authors
affiliated with City University of Hong Kong and **Tencent**). It constrains
weights to `{-1, 0, +1} · alpha` with a **3:4 structured sparsity**: in every
contiguous group of 4 weights, exactly 3 are non-zero (±1) and exactly 1 is 0.

**[VERIFIED]** Bit-state math (also stated in the paper §3.1):
`C(4,3) × 2^3 = 4 × 8 = 32` unique sign/position permutations per 4-weight
group, which exactly saturates a 5-bit index (2^5 = 32) → **zero wasted
codewords**, power-of-two-aligned 4-way grouping (SIMD friendly).

**[VERIFIED]** The paper's headline results (LLaMA-3.2-1B/3B, 10B-token QAT,
group size 128):
- Sherry at **1.25 bpw** matches Tequila at 1.67 bpw average accuracy
  (1B: 0.519 vs 0.519; 3B: 0.567 vs 0.576) across ARC-e/ARC-c/HellaSwag/PIQA/
  WinoGrande.
- Sherry is **QAT-based**: quantization-aware training with Straight-Through
  Estimator plus an "Arenas" annealing residual bypass during training
  (annealed to zero by end of training). Not a pure post-training method.
- On Intel i7-14700HX CPU (BitNet.cpp framework): 3B model at 1.25 bpw
  reaches 45.55 tok/s vs 38.80 tok/s for the 1.67-bit TL2 baseline
  (+18%) and 205.5 MB vs 846 MB size in that harness.
- Per-group (g=128) granularity is best but per-channel loses little
  (Table 3: 0.519 vs 0.513 avg on 1B).

## 2. On-disk format actually used downstream ("STQ1_0")

**[VERIFIED]** The concrete on-disk/block format is named **STQ1_0**
(GGML type 42), implemented in the (still-open) llama.cpp PR
ggml-org/llama.cpp#22836 ("ggml-cpu : add STQ1_0 ternary quantization with ARM
NEON vec_dot kernel", opened 2026-05-08 by jinlongsong@tencent.com, last
commit 2026-05-22; **state: OPEN, not merged as of 2026-09-02** — the
ONNX Runtime issue #28549 that says "just landed in mainline" is stale).
I fetched the PR branch (`pull/22836/head`, head commit `1e411d8`) and read
the patch directly.

**[VERIFIED]** Per-256-element block (`QK_K = 256`), 42 bytes = **1.3125 bpw**:

```c
// 1.3125 bpw
typedef struct {
    uint8_t qs[QK_K/8];    // 4-bit code per group of 4   (32 B)
    uint8_t sign[QK_K/32]; // 1-bit table select per group (8 B)
    ggml_half d;           // fp16 scale per block        (2 B)
} block_stq1_0;
static_assert(sizeof(block_stq1_0) == 42, ...);
```

- Payload bits: 256 weights × (5 bits / 4 weights) = 320 bits = 40 B.
- Overhead: 2 B fp16 scale per 256 weights = +0.0625 bpw
  → 1.25 payload + 0.0625 scale = 1.3125 bpw total.
- The 5-bit code is split into a 4-bit slot (which of 4 positions is the
  zero lane — actually its complement: which 3 lanes are non-zero) plus a
  1-bit sign-table select that flips all three non-zero lanes.
- **[VERIFIED]** Group layout detail from the PR: groups are **stride-16
  within each 64-weight chunk** (group g, chunk-local 0..15, covers weights
  `{c*64 + g + p*16 : p in 0..3}`), not plain contiguous quads. Decode is a
  32-entry byte codebook lookup:
  `qpack = stq1_0_codebook[(sign << 4) | slot]`, where each qpack byte packs
  4 ternary values as 4 × 2-bit lanes. Encoder rule (reference impl,
  `quantize_row_stq1_0_ref`): per group, the **smallest-|x| weight becomes the
  zero lane**, the other 3 take their signs; scale `d = max|x|` of the block
  (fp16). This is a plain PTQ-style projection — the PR itself is a *format +
  kernel* contribution, independent of how the ternary values were produced
  (QAT-trained or naive projection).

**[VERIFIED]** Kernels in the PR (as of head `1e411d8`):
- ARM NEON `ggml_vec_dot_stq1_0_q8_K` (`vqtbl2q` codebook lookup +
  `vdotq_s32`), with an ARMv8.0 no-dotprod fallback, SVE path, and a generic
  C reference.
- **No CUDA kernel in this PR** (0 `stq` references under `ggml/src/ggml-cuda/`
  in the diff; the ~33 CUDA files touched by the branch are pre-existing drift
  vs the merge base, not STQ additions).
- License: llama.cpp is MIT; the PR is part of that repository, authored on a
  Tencent email but contributed under the project's MIT license.

**[VERIFIED]** Quality context for the *format family*:
- The shipped production example is **AngelSlim/Hy-MT1.5-1.8B-1.25bit**
  (Hunyuan-MT translation model, 3.3 GB FP16 → 440 MB, Android demo via a
  Tencent llama.cpp fork with a custom "STQ kernel" for mobile CPUs). This is
  a QAT'd dense translation model, not an MoE expert store.
- **[VERIFIED]** The MoE-scale example is **AngelSlim/Hy4-preview-GGUF**
  ("Hy4 Preview"), total repo size ≈ 467 GB with three variants:
  `Hy4-preview-STQ1_0.gguf`, `Hy4-preview-UD-IQ1_M.gguf`,
  `Hy4-preview-Q4_K_M.gguf`. The published deployment guide
  (AngelSlim `docs/source/_extra/hy4_preview_gguf_guideline.md`) runs
  **`Hy4-preview-STQ1_0.gguf` across two machines (4× RTX A4000 16 GB +
  1× RTX 4090 Laptop 16 GB)** with a tiered "local-cold-hot" MoE provider,
  mmap load, route-page prefetch, and pinned profile cache. Reported
  steady-state decode on that 5-GPU rig: **≈ 1.02 s/token** for an (unnamed
  parameter count but large, ~460+ GB on-disk) Hunyuan MoE preview model.
- **[INFERENCE]** The same guide's server flags reference an
  `--moe-model-adapter hyv4-middle-iq1-m-r8-v1` and `UD-IQ1_M` sibling, i.e.
  the STQ1_0 GGUF is used **together with** IQ1_M-style parts — mixed
  precision in practice (attention/dense at higher precision, expert FFN at
  STQ/IQ1). This is the closest public precedent to what this research track
  simulates for DeepSeek-V4-Flash.
- **[HYPOTHESIS]** The "~1 s/token at 5 GPUs" figure suggests the Hy4 STQ
  deployment is still storage/host-bandwidth dominated, not compute bound —
  consistent with dee.cpp's own measurements (95.7% storage wait in v15,
  `STORAGE_ROOFLINE.md`).

**[VERIFIED]** llama.cpp's older ternary formats for reference:
- `TQ1_0`: 1.6875 bpw (3 weights in 5 bits, 3-way SIMD-unfriendly packing;
  block 110 B per 256 w), exists in mainline since mid-2025.
- `TQ2_0`: 2.0625 bpw, plain 2-bit ternary.
- `IQ2_XXS`: 2.0625 bpw block (`d` + `uint16 qs[QK_K/8]` = 2 B + 64 B per
  256 w), 2-bit codebook quants with 32-entry sub-tables.
- `IQ1_M`: 1.75 bpw (verified block size 56 B per 256 w:
  `qs[QK_K/16]` 2-bit + `qh[QK_K/32]` 2-bit + `scales[QK_K/32]` 4-bit + `d`).

## 3. Application to DeepSeek-V4-Flash routed experts (this repo)

**[VERIFIED] Current expert geometry (sealed, dee4-v3-trace):**
- `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/v62-terminal-evidence-20260902T003709Z/`
  (run `p2.3-dee4-trace-fillpool-ab2-q8l4-v62`, ACCEPT_CORRECTNESS):
  `dee4-metadata.json` — codec `deepseek-fp4-e2m1-e8m0`, 43 layers × 256
  experts = 11,008 total... (metadata declares `total_experts: 2364` — the
  trace store's *selected* subset; the full model has 256/layer),
  `total_bytes: 31,605,129,216` (29.44 GiB), per-expert record
  13,369,344 bytes (12.75 MiB), group size 32.
- Per-expert composition: w1 [2048×2048] + w3 [2048×2048] + w2 [4096×1024]
  FP4-packed payload (3×4 MiB = 12 MiB) + three e8m0 scale tensors
  [2048×128]+[2048×128]+[4096×64] (3×0.25 MiB = 0.75 MiB).
- Sealed trace journal: `routed_experts.jsonl`, 688 records (43 prefill +
  645 decode), SHA-chained (`chain_sha256`), run id `p2.3-route-journal-v50`.
- v62 engine counters: 3.5 GiB/GPU device cache (packed FP4), host-pack cache
  8.5 GiB/GPU, cuda0 328 resident hits / 2285 cold loads / 2004 evictions,
  h2d 30.5 GB + 28.9 GB per GPU over 15 decode steps.

**[VERIFIED] Ternary math vs current FP4 (per routed expert).**
Derived exactly from the sealed `dee4-metadata.json` component table:
logical weights per expert = gate w1 [2048,4096] + up w3 [2048,4096] + down
w2 [4096,2048] = **25,165,824 elements** (metadata lists packed-byte dims:
w1/w3 [2048,2048] I8 = 2 fp4/byte, w2 [4096,1024] I8). Effective current
bpw = 13,369,344 × 8 / 25,165,824 = **4.25 bits/weight** (4.0 FP4 payload
+ 0.25 e8m0 scale overhead at group 32).

| Quantity | MXFP4 current | STQ1_0 (3:4 ternary) | IQ2_XXS |
|---|---|---|---|
| weight elements/expert | 25,165,824 | 25,165,824 | 25,165,824 |
| bits/weight total | 4.25 | 1.3125 (1.25 payload + 0.0625 scale) | 2.0625 (2.0 payload + 0.0625 scale) |
| scale storage | 0.75 MiB (e8m0, group 32) | 98,304 blocks × 2 B fp16 = 196,608 B (block 256) | 98,304 × 2 B = 196,608 B |
| total bytes/expert | **13,369,344 (12.75 MiB)** | **4,128,768 (3.9375 MiB)** | **6,488,064 (6.1875 MiB)** |
| vs FP4 | 1× | **0.309× (3.238× smaller)** | 0.485× (2.061× smaller) |

Working-set totals (43 layers × 256 experts = 11,008 experts):
MXFP4 137.02 GB (127.6 GiB) ↔ STQ1_0 42.32 GB (39.4 GiB) ↔ IQ2_XXS 66.55 GB
(62.0 GiB). Sealed v50-trace selected subset (2,364 experts,
`total_bytes` 31,605,129,216 B = 29.44 GiB = 31.61 GB):
STQ1_0 → **9.77 GB (9.09 GiB)**, IQ2_XXS → 15.34 GB (14.28 GiB).

**[INFERENCE]** With 3:4 sparsity the *compute* semantics change (a zero lane
contributes nothing; an STQ GEMV is add/sub-select, not FP4 dequant-multiply).
An STQ expert cannot be "dropped into" the current FP4 dequant path without a
new kernel — this is a future optional mode, not a runtime patch. The stored
bytes, not the arithmetic, are what this track evaluates.

**[VERIFIED — the critical negative result]** **Naive per-group 3:4 ternary
projection of MXFP4 expert weights is NOT lossless and is NOT claimed to work.**
- Sherry's accuracy results are QAT-trained models (10B tokens, STE + Arenas);
  the paper's own ablation (Fig. 6) shows naive 3:4 sparse ternary training
  *without* Arenas collapses (weight trapping).
- No public PTQ (calibration-only) variant of Sherry/STQ1_0 exists in
  AngelSlim docs or the llama.cpp PR: the encoder in the PR is a magnitude
  projection with no error compensation, and the paper's Table 1 shows PTQ-style
  baselines (AbsMean 1.67-bit: 0.489 avg on 1B vs BF16 0.558) already lose
  several points at *1.67* bits, before even attempting 1.25.
- The Hy4-preview STQ1_0 GGUF was produced by Tencent with internal training/
  calibration pipelines; its quality is not independently benchmarked here.
- MXFP4 checkpoint weights (e2m1 + e8m0, group 32) cannot be reconstructed to
  a distribution that makes a 3:4 zero-forcing projection harmless; e2m1 has
  4 mantissa magnitudes (0, 0.5, 1, 1.5, 2 × 2^e), ternary ±d collapses all of
  them to one magnitude per block.

**[HYPOTHESIS]** A viable path exists that does **not** require full QAT of
DeepSeek-V4-Flash: per-layer *output-distance-aware* selection of which
(expert, projection) gets STQ1_0 vs IQ2_XXS vs MXFP4, with calibration-only
error feedback (e.g.,Activation-aware per-block scale fitting, GPTQ-style
Hessian corrections on a small calibration set) applied *per expert*, keeping
a small MXFP4 "premium tier" for the most reuse-heavy/sensitive experts.
This is unproven for this model; it is exactly what Phase E's optimizer
scaffolds, and what a future calibration experiment must measure.

**[HYPOTHESIS]** Because routed experts in DeepSeek-class routers exhibit
persistent per-domain hot experts, a small hot set at high precision + a large
cold pool at ultra-low-bit could capture most of the storage win at a small
fraction of the quality cost. No dee route-trace evidence contradicts this,
but no quality measurement exists in this repo either — tagged as hypothesis.

## 4. Kernel / licensing status

| Item | Status | License | Action for dee |
|---|---|---|---|
| Sherry paper method | published (ACL 2026) | arXiv paper | method only |
| AngelSlim toolkit | public repo | Apache-2.0 (repo LICENSE) | reference impl OK |
| llama.cpp STQ1_0 PR #22836 | **OPEN, unmerged** | MIT (llama.cpp project) | may port kernel; must attribute; watch PR for changes since format could still shift before merge |
| Hy4-preview GGUF | public HF repo | model license on HF | deployment precedent only |
| ONNX Runtime request #28549 | closed/stale | — | none |

**[VERIFIED]** AngelSlim repo LICENSE is Apache-2.0 (read in local clone
`/tmp/angelslim-audit`). llama.cpp is MIT (LICENSE in local clone). Both are
compatible with permissive reuse *with attribution*; the Phase F prototype in
this branch re-implements the 3-of-4 ternary packing from the paper's
description and the PR's public format spec, and does not copy kernel code.

## 5. Bottom line for the storage question

1. STQ1_0's **1.3125 bpw** format shrinks a routed expert from
   12.75 MiB → **3.9375 MiB (3.238×)** — exact numbers in §3. The full
   working set (11,008 experts) drops from 137.02 GB → 42.32 GB; the
   *served* trace subset (2,364 experts, 31.61 GB packed FP4) drops to
   **9.77 GB — which fits in 32 GB RAM entirely**, eliminating SSD misses
   at steady state (simulated in Phases B–D).
2. This is the largest single storage lever available without changing the
   runtime: it converts "SSD-streaming expert server" into "RAM-resident
   expert server" on a 32 GB consumer machine.
3. It is **not** a drop-in: needs a ternary GEMV kernel (Phase F prototype),
   needs quality validation (QAT or strong PTQ — unresolved), and needs the
   same route-cache machinery dee already has.
4. IQ2_XXS (2.0625 bpw, 6.1875 MiB/expert, 2.061× smaller) is the
   conservative fallback with far better understood PTQ behavior
   (llama.cpp mainline, widely used), but 15.34 GB trace-subset / 66.55 GB
   full-model — *not* RAM-resident on a 32 GB machine for the full model.

---

*Sources read in this audit:* arXiv:2601.07892 (HTML, full text);
llama.cpp PR #22836 branch (local fetch, head `1e411d8`, files
`ggml-common.h`, `ggml-quants.c`, `ggml-cpu/arch/arm/quants.c`);
llama.cpp main `ggml-common.h` (TQ/IQ block formats);
AngelSlim repo sparse clone (`docs/source/models/Hy-MT1.5/hy-mt1.5.md`,
`docs/source/_extra/hy4_preview_gguf_guideline.md`, `LICENSE`);
ONNX Runtime issue #28549; AngelSlim/Hy4-preview-GGUF HF metadata;
sealed dee evidence v62 (see §3).
