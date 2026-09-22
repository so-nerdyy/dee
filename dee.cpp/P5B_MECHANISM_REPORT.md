# P5b Mechanism Report — kernel v2 (dee-cpp-dsv4-p5b-mechanism)

Watcher/analysis of Kaggle kernel `nivind/dee-cpp-dsv4-p5b-mechanism`
**version 2** — the 3-arm warm-process divergence bisect. Artifacts:
`kaggle/deepseek-v4-flash-0731/p5b-v2-out/` (this dir is fresh; v1
artifacts untouched at `p5b-kaggle-out/`). Companion docs:
`P5_EXACTNESS_FORENSICS.md`, `P5_KAGGLE_WATCHER_REPORT.md`.

## 1. Kernel status / wall time / arm completion

- **Status: COMPLETE.** Kernel initiated ~00:13:08 UTC 2026-09-22
  (lastRunTime), driver P0 gate at 00:26:14 UTC (Kaggle VM boot/setup
  ~13 min), last arm exited 00:54:52 UTC, terminal ~00:55–00:57 UTC.
  **Total wall ~44 min** — inside the expected 40–60 min window.
- Code under test: `research/phase4-cache-hierarchy` @
  **88ed8af8eadcee0d48529c6365b3ea261f2edaf1** (unpinned branch head;
  v1 tested `4de5a47055cf` — the defect persists across both commits).
- **All 11 driver checks PASS** (2×T4 gate, clone, cmake, dee_core
  build, test_dee4_segmented build+run, pydee build, index mount,
  46-segment assembly).
- **All 3 arms completed, all 9 units ACCEPT_CORRECTNESS** (rc=0,
  16 tokens each, N_TOKENS=16, prompt = the mRNA prompt Q0 ×3 per arm):

| Arm | Subprocess wall | Units | Arm env delta vs mA |
|---|---|---|---|
| mA (plain repro) | 565.3 s | q0/q1/q2 all ACCEPT | — |
| mB (full det) | 540.2 s | q0/q1/q2 all ACCEPT | `NATIVE_TORCH_DETERMINISTIC=1` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| mC (cuBLAS ws only) | 540.2 s | q0/q1/q2 all ACCEPT | `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

Arm config verified: only `arm_id`/`extra_env`/`torch_deterministic`
differ across arms (all store/cache/budget/lanes knobs identical);
`cohort=None` everywhere — the v1 cohort-bug (duplicate prompt index,
INVALID_EXPERIMENT) is fixed. mB log line: `[det] torch deterministic
algorithms armed (warn_only=True, cudnn deterministic, tf32 off)`.

## 2. Per-arm results

Driver `verdict` field = `PASS` (execution). Mechanism analysis per
`p5b_report.json` + direct journal diffs (688 (step,layer) records/unit,
steps 0–15 × 43 layers):

| Arm | Accepted | Token shas (q0/q1/q2) | bit-identical | first div. weight | first div. ids |
|---|---|---|---|---|---|
| mA | 3/3 | `df99f177` / `4316daf6` / `4cd33ce4` | **false** | (step 0, layer 1), unit 1 (`e1232235`→`18e36a2b`) | (step 0, layer 3), unit 1 |
| mB | 3/3 | `df99f177` / `5b722cb8` / `31d329f9` | **false** | (step 0, layer 1), unit 1 (`e1232235`→`6c806966`) | (step 0, layer 3), unit 1 |
| mC | 3/3 | `df99f177` / `7d1246b1` / `08fd104b` | **false** | (step 0, layer 1), unit 1 (`e1232235`→`6c806966`) | (step 0, layer 3), unit 1 |

Journal divergence counts vs unit 0 (direct diff):

| Arm | q1 weights/ids div | q2 weights/ids div |
|---|---|---|
| mA | 687/685 | 687/685 |
| mB | 687/685 | 687/685 |
| mC | 687/685 | 686/682 |

Structural signature (all arms, step 0): **only layer-0 weight records
match** between unit 0 and warm units; layers 0–2 ids match (hash-routed,
input-determined); ids diverge wholesale from layer 3 (first learned
router). Same signature as v1 and the original P5 campaign.

Unit-0 output (all arms, identical): tokens
`[79, 14644, 33325, 3293, 260, 61457, 4090, 304, 90588, 14, 46010, 270, 3197, 734, 1956, 19786]`
— "mRNA vaccines represent a groundbreaking approach to immunization,
leveraging the body's own cellular" — matches v1 mC-q0 (`df99f177…`)
and the P4-seal anchor prefix. Warm units all produce degenerate text
(e.g. mB-q2: "vacc, for vaccine vaccines, vaccine vaccines …").

## 3. Verdict mapping

From `p5b_report.json → interpretation`:

| Flag | Value | Meaning |
|---|---|---|
| `repro_confirmed` | **true** | mA sequential units diverge — campaign signature reproduced on the plain `generate()` path |
| `cublas_convicted` | **false** | mC dirty: `CUBLAS_WORKSPACE_CONFIG=:4096:8` alone does NOT fix it |
| `torch_component` | **false** | not "mB clean + mC dirty" — no torch-only component isolated |
| `unresolved` | **true** | **mB dirty: divergence survives the full determinism config** |

## 4. Interpretation — both library-config fixes FAILED

This is a conviction-by-elimination, and it is stronger than a bare
"unresolved":

1. **cuBLAS-workspace nondeterminism is exonerated as the controllable
   cause.** `CUBLAS_WORKSPACE_CONFIG=:4096:8` is process-global — it
   covered torch's handles AND the engine's own `cublasGemmEx` handle —
   and warm units still diverged with an identical (0,1) signature.
2. **torch-op nondeterminism is exonerated** to the reach of
   `use_deterministic_algorithms(warn_only=True)` + cudnn-deterministic +
   TF32-off: no "does not have a deterministic implementation" warnings
   fired in mB, yet units still diverged. Torch forward ops
   (attention/router/shared-expert/dense GEMMs) are deterministic given
   identical inputs — and mB proves they were not the perturbation source.
3. **Fresh-state determinism is absolute — proven at file level.** All
   three v2 unit-0 route-journal files are **byte-identical to v1's
   mC-q0 journal** (full sha256
   `f4bd09925457f5fa9462270dda4bd969772a1af9f6ce22035eb0271ecb269609`,
   raw bytes verified equal) — across two kernel runs, two VMs, and two
   commits. Unit-0 token sha `df99f177…` identical in all arms, and the
   unit-0 cache-event stream is identical across arms (5721 events,
   cold=3139/host_hit=738/resident=1844). The defect is specifically
   **warm-process, second-and-later inference**.
4. **Perturbation has low entropy.** At (step 0, layer 1) there are 6
   distinct weight-shas across 9 units; mB-q1 and mC-q1 (the two
   CUBLAS-env arms) collide on `6c806966`. This looks like a small set of
   discrete numerical outcomes — consistent with algorithm/kernel-path
   selection or a bounded stale-state variable, not arbitrary corruption.
5. **Localization unchanged:** injection sits between layer-0's router
   input and layer-1's router input — layer-0's routed-expert FFN output
   (the dee_core native engine path: gather → FP4 dequant/GEMM → swiglu →
   scatter), layer-0 shared expert, or layer-1 pre-router attention. With
   torch-side ops deterministic under mB, the **dee_core native expert
   path and its glue are the prime site**.

### Surviving hypotheses (ranked)

- **H1 — dee_core custom-kernel state/address dependence.** The engine's
  FP4 dequant/GEMM/swiglu/gather kernels are not governed by
  `CUBLAS_WORKSPACE_CONFIG` or torch flags. Per-call deterministic but
  warm-state-correlated behavior (stale scratch reads, launch config or
  vectorization chosen on warm allocator/arena state) fits every
  observation, including within-forward pad-row agreement.
- **H2 — engine-state carryover.** Between units the runner calls
  `reset_runtime_cache`/`clear_host_cache`/`reset_store_stats`/
  `reset_external_profile` — but persistent engine allocations, streams,
  generations, pinned registrations, and the torch caching-allocator
  layout survive. A stale-read or generation-skewed path is
  deterministic-per-call and unit-correlated.
- **H3 — residual cuBLAS heuristic drift.** The env var bounds workspace
  but does not pin algorithm choice; cuBLAS/cuBLASLt heuristics are
  documented to depend on pointer alignment, which drifts in a warm
  allocator. Would still be "library-class" but is NOT fixed by the env
  var alone — needs explicit `cublasSetWorkspace` + algo pinning
  (engine.cpp:4091–4092 creates the handle with stream binding only).
- **H4 — driver-level.** Weakest: unit-0 reproduces across VMs/kernels.

### Recommended next fix / experiment (no new GPU batch needed yet)

1. **Run the never-executed forensics ARM C:** same 3-unit repro but with
   routed-expert FFN computed by the CPU/torch reference path instead of
   `moe_forward_batch_device`. Clean ⇒ dee_core device path convicted;
   dirty ⇒ torch/dense side. This is the single most discriminating
   remaining bisect and should be the next GPU batch (one arm suffices).
2. **Per-layer hidden-state SHA hook** (original ARM A instrument): hash
   `h` at block boundaries — post-attn-0, post-MoE-0, post-block-0,
   post-attn-1 — to pinpoint the exact producer of the first divergence
   inside layer 0→1.
3. **Engine-internal determinism sweep:** explicit `cublasSetWorkspace` +
   pinned `cublasGemmEx` algo on the engine handle; audit dee_core custom
   kernels (FP4 dequant, swiglu_cuda, gather/scatter) for alignment- or
   state-dependent code paths and stale-scratch reads.
4. **Allocator hypothesis probe:** `torch.cuda.empty_cache()` + full
   engine teardown/rebuild between units in one process. If warm
   divergence vanishes under allocator-state reset, it is
   address/layout-dependent kernel behavior.
5. **Contract decision still open** (forensics §contract): if the root
   cause proves to be unpinnable library numerics, choose between a
   hardened bitwise gate vs a bounded-tolerance exactness bar.

## 5. Anomalies

- **None blocking.** Zero timeouts, OOM, CUDA errors, tracebacks, or
  torch-deterministic warnings across all 9 units; all arms exited rc=0.
- mB-q1 ≡ mC-q1 at (0,1) (`6c806966`) — the two CUBLAS-env arms drew the
  same perturbed state at that one record; see §4.4 (low-entropy draw).
- mC-q2 diverged in slightly fewer records (686/682 vs 687/685) — two
  weight records and six id records re-converged downstream; cosmetic.
- `integrity*.json` carries a stale `"branch": "freebuff/deepseek-v4-flash-0731-t4"`
  field; actual tested code was `research/phase4-cache-hierarchy` @
  `88ed8af8` per the driver's clone check and RESULT records.
- v1→v2 commit drift (4de5a470 → 88ed8af8): the defect reproduces on both.
- Harmless warnings only: `host_pack_cache_bytes=0` defaulting notice;
  non-writable-buffer UserWarning (suppressed after first hit); cosmetic
  mistune/nbconvert SyntaxWarnings at log teardown.

## v3 (kernel v3, commit 518f4a6)

Third kernel version — the causal-isolation arms. Artifacts:
`kaggle/deepseek-v4-flash-0731/p5b-v3-out/`. Code under test:
`research/phase4-cache-hierarchy` @ `518f4a61a4d6` (clone check +
RESULT records). Three sequential arms × 3 identical prompts (Q0 mRNA),
N_TOKENS=8 (v2 used 16), route-weight + **capture** journals on every
arm. The capture journal sha256s `moe_out` (combined routed+shared FP32),
`shared_out`, `router_scores`, `expert_ids`, `routing_weights` per
(step, layer) — the instrument that finally splits layer-0 internals.

### 1. Run status / timeline

- **Status: COMPLETE, verdict PASS** (execution completeness, not a
  mechanism verdict). Driver 01:25:03 → 01:51:26 UTC (~26 min; shorter
  than v2's ~44 min because N_TOKENS=8). All 11 driver checks PASS
  (2×T4, clone, cmake, dee_core, test_dee4_segmented, pydee, index
  mount, 46-segment assembly).
- Arm subprocess walls: mD 495.0 s, mE 490.3 s, mF 515.2 s. All runner
  processes exited rc=0.

### 2. Per-arm outcomes

| Arm | Intent | Result | Verdict |
|---|---|---|---|
| mD `NATIVE_FFN_BACKEND=cache_fp16` | torch reference FFN bypasses `moe_forward_batch_device` | **all 3 units ERROR** — `DeepSeekExpertCache::reserve failed: no evictable victim` | **INVALID — harness bug, zero evidence** |
| mE `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | disable torch caching allocator | 3/3 executed, units **diverge** | **dirty → torch allocator layout exonerated** |
| mF `NATIVE_BATCHED=1` | pointer-batched `cublasGemmBatchedEx` | 3/3 executed, units **diverge** | **dirty, but NULL ARM — batched path never engaged** |

Token shas (unit 0/1/2): mE `89c93f0a`/`a2d68594`/`9ce15d19`,
mF `89c93f0a`/`94859897`/`15e38de6`. Unit-0 outputs are the coherent
"[79, 14644, 33325, 3293, 260, 61457, 4090, 304]" = "mRNA vaccines
represent a groundbreaking approach to" — same prefix as v1/v2 unit 0.
All warm units degenerate (" mRNA vaccines work, mRNA vaccines work,",
"vaccine vaccines vaccines …", " : mRNA mRNA mRNA …").

#### mD failure — analysis, not a result

`DeepseekV4CacheFfn` stages each layer's shared expert under `skey=-1`
and **pins it permanently — the pin survives `reset_state` by design**
(`deepseek_v4_layer_candidate.py:197-227`). FP16 shared payloads are
50,331,448 B each; the reference cache budget `NATIVE_REF_CACHE_BYTES`
defaults to 1 GiB → exactly 21 pinned shared experts fit
(`used=1056964608` = 21×50331648, `free=16 MiB`, 21 pinned `((L,-1),1)`
entries in the error). q0 died reserving a routed expert at **layer 21**;
q1/q2 died at **layer 0** because unit-0's pins persist across units.
The arm was un-runnable as configured: shared pins alone need ~2.02 GiB
for all 43 layers, plus routed working set. `NATIVE_REF_CACHE_BYTES ≥ 4
GiB` (env knob already exists) would fix it; per-unit pin release or
unpinned shared payloads would fix it structurally.

Driver-side artifact worth noting: mD-q1/q2 wrote **empty** journals
(sha256 `e3b0c442…` = empty file) yet count as "accepted" — the
`produced` criterion accepts any present journal. All `first_divergent_*`
fields for mD are `"missing": true` artifacts of the empty files, and
`interpretation.engine_exonerated`/`unresolved` inherit the crash — see
corrected mapping in §4.

#### mF null-arm — flag never reaches the exercised path

`NATIVE_BATCHED=1` set `engine_config.use_batched_experts=true`
(confirmed in RESULT records), but the runner calls
`moe_forward_batch_device` → `moe_forward_batch_device_impl`
(engine.cpp:754-762), which **never consults the flag** — it always runs
per-expert `swiglu_expert_batch_fp16_cuda`. The flag only gates the
legacy host-side `moe_forward` path at engine.cpp:309-311, and that
branch additionally requires `cache_dtype == Fp16` while all v3 arms run
Fp4E2m1. Evidence: `pointer_batched_expert_calls=0` on every mF unit,
cublas_calls ≈ mE's (mF 5619 vs mE 5532 on cuda0 cumulative at q2;
identical 5775/5196 at q0), identical `kernel_launches`. mF is a
baseline-path repeat — its divergence re-confirms the repro but says
nothing about batched GEMM. **The per-expert-dispatch hypothesis is
still untested.**

#### mE engagement evidence

`gpu_memory.reserved_gib = 0.0` on both GPUs in all mE units (vs
7.31/9.02 GiB reserved in mF) — torch's caching allocator genuinely off;
every tensor alloc went through cudaMalloc. Divergence unchanged → the
warm-state variable is not torch allocator layout/alignment. Note this
exonerates only *torch-side* allocations: all dee_core workspaces/cache
blocks are the engine's own `cudaMalloc`s, unaffected by the flag.

### 3. Capture-journal localization — the decisive datum

Per-field first divergence vs unit 0 (344 records/unit = 8 steps × 43
layers; identical for q1 and q2 in both arms):

| Field | mE first div | mF first div | ndiff (q1/q2) |
|---|---|---|---|
| `moe_out_sha256` | **(step 0, layer 0)** | **(step 0, layer 0)** | 344/344 both |
| `shared_out_sha256` | (0, 1) | (0, 1) | 343/343 |
| `router_scores_sha256` | (0, 1) | (0, 1) | 343/343 |
| `routing_weights_sha256` | (0, 1) | (0, 1) | 343/343 |
| `expert_ids_sha256` | (0, 3) | (0, 3) | 341/338, 341/341 |

At (step 0, layer 0) — the unit's first forward — **every capture
matches unit 0 except `moe_out`**:

- `router_scores` match ⇒ the FFN input `xf` is bit-identical
  (router scores are a torch GEMM on `xf`).
- `expert_ids` + `routing_weights` match ⇒ same experts, same gate
  weights.
- `shared_out` match ⇒ the shared expert — a same-shape **torch** GEMM
  path — reproduces bit-exactly in the warm process. In-arm control:
  the warm process is not globally perturbed; torch compute on
  identical input is stable.
- `moe_out` = `Σ_rank raw·weights + shared_out` (native subclass,
  `deepseek_v4_layer_candidate.py:482-534`) differs ⇒ the **routed-expert
  term `raw` written by `moe_forward_batch_device` differs on
  bit-identical (layer, hidden_fp16, ids) inputs.**

Cache events close the loop on inputs: at token 0 / layer 0 all three
units cold-load the *identical expert set* (`cold`/`storage_miss` for
every one of ~81 experts — same store reads, same request stream;
`evicted_before_use=0` everywhere).

moe_out(0,0) draw structure: the fresh value `9942585c…` is identical
for mE-q0 and mF-q0 (cross-process unit-0 reproducibility, again), and
the four warm units take **four distinct** values (`9df27edf`,
`9f76d0ff`, `7d69ffe3`, `56b0bb6f`) — no collision this time; each warm
unit draws its own perturbation.

Cascade: everything downstream of layer-0's FFN output diverges —
shared_out/router_scores/weights from layer 1, learned-router ids from
layer 3. Identical signature to v1/v2, now with the injection site
resolved one level deeper: **not "between layer-0 FFN input and layer-1
router input" but specifically inside layer-0's routed-expert
execution.**

### 4. Corrected verdict mapping

The driver's `interpretation` block must be read against the two arm
defects:

| Flag | Report value | Corrected reading |
|---|---|---|
| `engine_convicted` | false (mD dirty) | **true, by capture localization** — mD crashed, but `moe_out`-only divergence at (0,0) with all other layer-0 tensors matching convicts the routed-expert path on identical inputs |
| `allocator_convicted` | false (mE dirty) | stands — **torch caching allocator exonerated** (defect persists with direct cudaMalloc; engine-side arena state remains live) |
| `batched_stable` | false (mF dirty) | **vacuous** — batched path never engaged; per-expert-vs-batched dispatch UNTESTED |
| `engine_exonerated` | true | **artifact of the mD crash** — invalid |
| `unresolved` | true | **resolved further than the driver knows**: injection is inside `moe_forward_batch_device_impl`'s serve+compute chain |

### 5. Fresh determinism reconfirmed (file level)

v3 mE-q0 route_weights are **344/344 record-identical to v2 mA-q0's
first-8-step prefix** — across kernel versions, VMs, and commits
(88ed8af → 518f4a6). mE-q0 and mF-q0 journals are byte-identical to
each other (weights `d97c566e…`, captures `eea12aab…`). The defect
remains specifically warm-process, second-and-later inference.

### 6. Hypothesis state after v3

**Convicted region:** inside `moe_forward_batch_device_impl` for the
Fp4 path — the serve+compute chain per expert group:
`prepare_fp4_experts`/`stage_expert` (host pack → H2D → slot),
`prefetcher_.wait_on_stream` ordering, `cache_.pin`/`data`,
`decode_fp4_cache_block_to_scratch` (packed → shared FP16 scratch),
`swiglu_expert_batch_fp16_cuda` (3 `cublasGemmEx` + silu-mul per expert
on `cublas_handle_`), D2D gather/scatter, deferred-unpin retirement.

**Two surviving sub-hypotheses** (sha256 can't separate them — needs a
magnitude measure):
- **S1 — wrong bytes served (correctness bug, not numerics):** a warm
  unit consumes a slot holding stale/wrong-generation/wrong-expert
  packed bytes, or a torn/misordered H2D fill. Fits the categorical
  output collapse and the statefulness (device cache slots, host-pack
  generations, prefetch ring all differ cold vs warm). Host-pack hits
  exist in warm units (mE-q1: 521 host_hit events) — a fill-coalescing
  or generation bug there would inject different payload bytes with
  identical request streams.
- **S2 — same bytes, different compute (numerics):** state-dependent
  kernel behavior inside decode/swiglu — cuBLAS heuristic drift on the
  engine handle (workspace fixed at handle creation, but algo choice
  may still depend on per-slot blob pointer addresses, which do change
  across units as eviction order shifts), or stale-scratch reads in the
  shared decode/batch buffers.

**Exonerated at this boundary:** torch-side compute (shared expert +
router GEMMs bit-reproduce warm), embed/attention upstream of the
layer-0 FFN input, torch caching allocator, per-expert-vs-batched
question (untested, not exonerated), `CUBLAS_WORKSPACE_CONFIG`/
`use_deterministic_algorithms` (v2).

### 7. Recommended next steps

1. **In-engine consumption-point hashing (single discriminating arm):**
   sha256 the packed `d_blob` bytes (or a fixed prefix) at pin time,
   the decoded FP16 scratch, and each expert's raw output for a fixed
   (layer, expert) probe across units. Blob sha differs → S1 (serve
   path: slot/generation/host-pack/prefetch ordering). Blob identical
   but raw differs → S2 (kernel). This is the minimal remaining bisect.
2. **Add a magnitude measure** to the capture journal — max |Δ| and
   divergent-element count, not just sha — to separate ULP drift from
   categorical byte-swap (changes remediation: tolerance bar vs
   correctness fix).
3. **If mF's question is still wanted:** plumb `use_batched_experts`
   into `moe_forward_batch_device_impl` (or run an Fp16-cache arm) —
   the current flag is inert on the exercised path.
4. **mD is no longer load-bearing** — the capture journal answered the
   engine-vs-upstream question. If rerun for completeness:
   `NATIVE_REF_CACHE_BYTES ≥ 4 GiB` (shared pins alone ≈ 2.02 GiB) or
   drop the permanent shared pinning.
5. **Driver hardening for future runs:** treat empty journals as
   no-evidence (sha `e3b0c442…` counted as "accepted" today); include
   `PYTORCH_*`/`extra_env` keys in the report env dump (mE's flag is
   absent from `arms.mE.env` — only `resolved.extra_env` proves it).

### 8. Anomalies (v3)

- **mD arm structurally unrunnable** — ~8 min of GPU time produced no
  mechanism evidence; see §2.
- **mF arm inert** — `NATIVE_BATCHED=1` set a config flag no exercised
  code path reads; see §2.
- `integrity*.json` still carries stale `"branch":
  "freebuff/deepseek-v4-flash-0731-t4"`; actual code was
  `research/phase4-cache-hierarchy` @ `518f4a6` (git_commit field +
  driver clone check agree).
- mE `gpu_memory` reads 0.0 GiB allocated/reserved — the expected
  signature of `PYTORCH_NO_CUDA_MEMORY_CACHING` (no caching-allocator
  pool to report), corroborating arm engagement.
- Zero timeouts/CUDA errors/tracebacks in mE/mF; all units rc=0,
  ACCEPT_CORRECTNESS.
