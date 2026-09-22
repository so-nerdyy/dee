# P5 Exactness Forensics — warm-process divergence

Companion to `P5_KAGGLE_WATCHER_REPORT.md` (campaign execution + verdict).
This document is the root-cause triage of the exactness-gate FAIL and the
proposed minimal mechanism test. No repository code was changed by this
analysis.

## Verdict recap

- Campaign executed cleanly: 16/16 units `ACCEPT_CORRECTNESS`, c0 anchored
  to the sealed P4 manifest, zero infra failures.
- Exactness gate FAIL: cohort rows do not reproduce c1 singletons, and the
  c1 singleton path itself is unstable across sequential units in one
  process.

## Established facts (from artifacts, not theory)

1. **Fresh process = clean.** Every arm's first unit is coherent:
   c1-c0, c2-c0, c4-c0, c8h (single cohort) all produce clean output.
   Fresh processes reproduce each other bit-exactly: c2-c0 row p1 ==
   c8h row p1 (`2d25792e1197`), c4 row p2 == c8h row p2 (`6814a55107bf`),
   and c0_anchor passed the P4 seal.
2. **Warm process = divergent.** Units ≥1 in the same process degrade
   progressively: c1-c1/c2 coherent-but-different, c1-c5 all-`?` stream
   (`[33]×128`), c1-c7 `The.??` loop. Severity grows with unit index.
3. **Inputs are byte-identical.** c1-c0 vs c1-c7: same prompt text (the
   mRNA prompt, index 0 and 7), same L*=32, same pad=18, same prompt_sha.
   Cache-event streams at layer 0 are byte-identical (same experts, same
   request order).
4. **Divergence injects during layers 0–2 compute.** Layers 0–2 are
   hash-routed (`n_hash_layers=3`): expert IDs are a deterministic
   `tid2eid` function of input token IDs, so matching route IDs there is
   guaranteed regardless of hidden state. The first learned-routing layer
   (layer 3) diverges 0/18 pad rows + 0/14 real rows at step 0 — the
   *prefill*, the unit's very first forward. The hidden state entering
   layer 3 already differs.
5. **Within-forward determinism holds.** Unit 7's 18 pad rows — byte-
   identical inputs (same pad token, same mask) — route identically to
   each other at layer 3 (`unique_pad=1`). Identical inputs in the same
   call produce identical results. This rules out races, atomics, and
   torn reads: a racy kernel would scatter pad rows against each other.
6. **Not a member-index bug.** Cohort-internal determinism holds
   (identical prompts in the same cohort produce identical rows:
   c8h p0==p7 bitwise). The "member>0 diverges" pattern is an artifact of
   comparing warm-unit cohort rows against c1-c{r} references that are
   themselves warm-unit-corrupted for r≥1.

## Eliminated mechanisms (statically verified clean)

| Suspect | Result |
|---|---|
| Corrupt expert bytes | Packed FP4 immutable; host pack serves byte-identical data on hit vs miss (synchronous gather); store integrity sha matches sealed manifest |
| Stale Python/model state | `reset_state()` coverage complete; CPU repros stable — 6 sequential units bit-identical for BOTH the reference path AND the real `DeepseekV4CacheFfn` candidate-class path (stubbed cache/loader) |
| Unwritten `_native_raw_output` slots | `torch.empty` buffer fully covered: every valid `(token, rank)` position lands in a per-expert group and is scattered; invalid IDs fail-closed |
| KV-cache / indexer staleness | `kv_cache.zero_()` on reset; `-1` index masking verified in `sparse_attn`; decode writes bounded by `start_pos % win` |
| Pin/lease/eviction races | Deferred-unpin retirement is event-gated and drained on every reset; `release_source_pages` bounds/page-exact checked; `staging_int8_` re-pointing generation-gated |
| Rotary-cache mutation | `precompute_freqs_cis` lru_cached tensor is read-only in `apply_rotary_emb` (conj/view non-mutating) |
| Handoff race (GPU0→GPU1) | Event-chained D2H/H2D, full-tensor copies, retention until completion |
| Host-pack persistence | `clear()` empties map+lru; unit-7's low miss count (2166 vs 7649) is a *symptom* — degenerate routing touches fewer unique experts, not a cause |
| Pinned-memory exhaustion | VmLck stays 0.0; RSS grows only ~1.5 GiB across 8 units (pack re-faults) |
| RNG leakage | No RNG consumers in the runtime path |
| Allocator/OOM pressure | `gpu_memory` stable at 6.991 GiB allocated across all units |

## Remaining hypothesis (high confidence)

**GPU-runtime numerical nondeterminism correlated with warm-process
state** — most plausibly cuBLAS algorithm/workspace drift. No determinism
controls exist anywhere in the runtime: no `CUBLAS_WORKSPACE_CONFIG`, no
`cublasSetWorkspace` on `cublas_handle_` (engine.cpp:4091–4092 creates the
handle + binds the nonblocking stream only), no
`torch.use_deterministic_algorithms`, no TF32 flags.

Mechanism: cuBLAS heuristic algorithm selection is workspace- and
heap-state-dependent. Unit 0 runs on a clean allocator; by unit 1+ the
torch caching allocator + engine workspace state has drifted, the
heuristic selects a different kernel/tiling for an identical GEMM shape,
and the result differs in the low bits. During hash-routed layers 0–2 the
expert *IDs* can't move (input-determined) but the routing *weights* and
all dense compute (attention, router GEMMs, shared expert, head) still
perturb the hidden state; at layer 3 the learned router's top-6 boundary
flips wholesale; 128 autoregressive steps amplify it to collapse.

Consistent with: fresh-process reproducibility (same initial heap → same
algo choices), within-forward determinism (algo choice fixed per call),
CPU stability (no cuBLAS), progressive severity (each unit gets a
different perturbation draw), and every structural mechanism auditing
clean.

Secondary (not fully excludable without instrumentation): an
allocator-address-dependent kernel-path difference, or driver-level
nondeterminism. Both are the same experiment class.

## Proposed mechanism test — ONE GPU batch, four arms (needs authorization)

Preflight gate (AGENTS.md): hypothesis above; baseline = unit 0;
candidate = units 1..3 in-process; metric = per-(step,layer) hidden-state
SHA + route journal equality; accept = bit-exact across all units under
the fixed config; reject = any divergence; abort = first unit failure.

- **ARM A (reproduce, minimal):** one process, 4× identical K=1 units
  (same prompt, same padding, existing resets) with a `post_layer_hook`
  that SHA-256s `h` per layer per step. Output: first divergent
  (step, layer) pair — pinpoints whether injection is in attention,
  router GEMM, engine MoE, or shared expert. Cost: ~4× one decode.
- **ARM B (convict/exonerate cuBLAS):** same as A plus
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `torch.use_deterministic_algorithms(True, warn_only=True)`, and
  `cublasSetWorkspace` on the engine handle (small code change, behind an
  env flag). PASS = bit-exact across units → cuBLAS convicted and the
  fix is identified; FAIL = deeper kernel/driver issue.
- **ARM C (torch-vs-engine isolation):** same prompt but routed experts
  computed by the CPU/torch reference FFN path instead of
  `moe_forward_batch_device`. If units reproduce → engine-side;
  if they diverge → torch-side (attention/router/shared).
- **ARM D (sequential control):** `generate()` instead of
  `generate_cohort`, 4 sequential units — tells whether the bug is
  cohort-specific or any-second-inference (P4 ran multi-arm sessions but
  never two inference units per process — this surface is untested).

All four arms fit one batch: total GPU time ≈ 8–10 K=1 equivalents at
reduced `N_TOKENS` (32 tokens suffices — divergence appears at prefill).

## Contract question for the user

The current gate requires bit-exact token streams between units. If the
divergence is confirmed as cuBLAS-class library nondeterminism, decide:

- (a) enforce deterministic GEMM config and keep the bitwise gate, or
- (b) accept bounded numerical equivalence (formal tolerance on
  hidden-state drift + route-agreement threshold) as the exactness bar.

Option (a) is the stricter and cheaper path; (b) needs a defensible
tolerance derivation.

## P5b mechanism test — v1 result (kernel dee-cpp-dsv4-p5b-mechanism v1)

v1 carried a driver bug (cohort groups reused prompt index 0 across
cohorts — the runner correctly rejects duplicate indices) so only arm mC
executed, but it produced the decisive datum anyway:

- **mC: two plain sequential `generate()` units on byte-identical
  unpadded input DIVERGE.** The defect is not cohort-specific — it is
  any-second-inference in a warm process.
- **Route-weight journal localization (new instrument):** 687/688
  (step, layer) records diverge. The ONLY matching record is
  (step 0, layer 0): layer-0's routing weights are identical → embed +
  attention-0 produce identical hidden state. Layer-1 weights differ at
  step 0 → the injection sits between layer-0's FFN input and layer-1's
  router input: layer-0's MoE output (native engine `cublasGemmEx`),
  shared expert, or layer-1's attention. Expert-ID flips begin at
  layer 3 (first learned router) — same signature as the campaign.
- **Cross-kernel fresh determinism:** v1's first sequential unit
  reproduces the P4-seal anchor prefix `[79, 14644, …]` — fresh-process
  output is reproducible across kernels/VMs; warm-process second runs
  diverge from token 0.

v2 (in flight): all-sequential 3-arm bisect — mA plain repro, mB
`NATIVE_TORCH_DETERMINISTIC` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`, mC
`CUBLAS_WORKSPACE_CONFIG` only. Clean mB+mC ⇒ the env var alone is the
fix; clean mB + dirty mC ⇒ torch-op nondeterminism beyond cuBLAS; dirty
mB ⇒ deeper than library config.

## What this means for Phase 5

The serving machinery itself is proven: cohort execution, dedup, evidence
chain, cold resets, and cross-process reproducibility all work — 16/16
units executed exactly as designed. The failed gate is a numerics-layer
issue in the accelerator path, not a hierarchy or scheduling defect. The
fix is likely a determinism configuration, not an architectural change.
