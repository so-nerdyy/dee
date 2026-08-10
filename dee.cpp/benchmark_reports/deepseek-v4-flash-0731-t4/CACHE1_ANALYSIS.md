# CACHE1.1 — Expert Access-Pattern Analysis (sealed DS10 v12 trace)

Zero Kaggle time: pure local analysis of the sealed
`ds10-v12-accept-dual-t4-decode` evidence (`token_trace`, 16 tokens, 43
layers, 6 routed + 1 shared expert per layer call).

## 1. Volume

| Metric | Value |
|---|---|
| Tokens | 16 (1 prefill of 7 positions + 15 decode) |
| Layers | 43 (split 22/21 across GPU0/GPU1) |
| Expert accesses (incl. shared) | 6,364 |
| Unique (layer, expert) pairs | 2,365 |
| Accesses per token (decode) | 43 × 7 = 301 |
| Shared-expert accesses | 688 (10.8% of all) |

## 2. Reuse structure — the deciding question

- **Consecutive-token reuse (same layer, t→t+1): 37.5%** of the 6 expert
  slots overlap between adjacent tokens. So reuse *exists*.
- **Within-layer revisit: 155%** of unique slots are used by ≥2 tokens.
- **Popularity: top-5% of (layer,expert) pairs cover 23.6% of accesses**
  (vs 5% under uniformity) — a mild power-law tail, not flat, not extreme.

**Conclusion: this is NOT a uniform-routing dead end.** Reuse is
structural (37.5% consecutive) but spread over a huge working set
(2,365 × 48 MiB ≈ 111 GiB if held resident). Caching can capture only the
*temporal* reuse; the rest must be prefetch/overlap.

## 3. Why DS10 measured 0% hits

Simulation (`cache1_policy_sim.py`) now reproduces the sealed numbers
exactly: current policy → **0.00% GPU hits, 100% of accesses fetched over
HTTP**. Mechanism:

1. Budget 2 GiB/GPU = 42 experts (48 MiB each).
2. Prefill (token_0) alone touches ~40 distinct experts per layer —
   ~1,700 unique experts — completely churning the 42-slot cache before
   decode starts.
3. During decode, the reuse distance (same layer revisited ~300 accesses
   later) far exceeds 42 slots, so LRU retains nothing.
4. Provider host LRU disabled (`raw_experts_per_layer=0`): every miss =
   full HTTP range fetch (w1+w2+w3 + scales). Measured: 10,202 fetches /
   298 GiB H2D in the primary run.

## 4. Policy simulation results (per-GPU split, 2-level cache)

| Policy | GPU hit | Provider hit | HTTP fetch % |
|---|---|---|---|
| **DS10 as-run** (2 GiB, raw=0) | 0.00% | 0.00% | **100.0%** |
| + provider raw=8 (2 GiB) | 0.00% | 32.6% | 67.4% |
| + provider raw=16 (2 GiB) | 0.00% | 46.7% | 53.3% |
| + pin shared (2 GiB, raw=8) | 11.1% | 23.8% | 65.1% |
| + pin shared (8 GiB, raw=16) | 33.6% | 13.5% | 52.9% |
| + pin shared + LFU (8 GiB, raw=16) | 27.5% | 19.8% | 52.8% |
| + pin shared + pop2 (4 GiB, raw=8) | 24.6% | 16.3% | 59.0% |

Belady (oracle) GPU-cache ceilings per GPU:
- 2 GiB: 20–22% · 4 GiB: 33–41% · 8 GiB: 46–56% · 12 GiB: 52–60%

## 5. Honest achievable target

**≥70% GPU hit rate is NOT reachable on this 16-token trace** — even
oracle eviction tops out at 56–60% at 12 GiB/GPU (which itself busts the
14 GiB memory ceiling). The campaign's ≥70% assumes a longer trajectory
with more reuse; on the sealed canonical trace the *reachable* outcome is:

- **Combined GPU+provider (no HTTP) ≥ 47%** at 8 GiB with pinned shared +
  raw=16 — a 2× reduction in HTTP fetches vs 100% today.
- GPU hit rate **~34%** at 8 GiB (vs 0% measured).

Primary recommendations (justified by these numbers):
1. **Re-enable provider raw LRU (raw=16)** — cheapest, −47% HTTP fetches.
2. **Pin shared experts** — 43 guaranteed residents, +11% GPU hits, removes
   688 fetches of the most frequent expert.
3. **Raise GPU budget to 8 GiB/GPU** where the memory plan permits.
4. **Router-ahead prefetch** of the next token's likely experts (the 37.5%
   consecutive-reuse set) — overlaps HTTP latency with layer compute.

## 6. Verdict path

`ACCEPT_CACHE_HITRATE_TARGET` (≥70%) is not achievable on the sealed
trace; the campaign should terminate with **`ACCEPT_CACHE_PARTIAL`** once
the implemented stack shows a *measured* multi-fold reduction in HTTP
fetches and a GPU hit rate well above 0%, with all CACHE1.4 correctness
checks passing (deterministic rerun, cold==warm, budget invariance,
identical token IDs vs sealed DS10).

Artifacts: `cache1_analysis.py`, `cache1_policy_sim.py` (this directory).

## 7. v13 remote run (Modal dual-T4, 2026-08-06): REJECT_MEMORY — diagnosis

First implementation of the policy stack (4 GiB/GPU, provider raw=16,
shared pinned) ran on Modal and produced **REJECT_MEMORY
(cache1_memory_primary)**: host RSS after the warm decode was **28.1 GB**
vs the 12 GB sealed ceiling.  Critically, **correctness held**:
`tokens_match_sealed_ds10: true`, `cold_warm_equal: true`, all logits
finite, GPU peaks 10.1/11.8 GiB < 13.67 GiB ceiling.

### What pushed host RSS to 28.1 GB

| component | bytes | notes |
|---|---|---|
| post-build base | 6.4 GB | incl. 43 shared FP16 payloads (2.16 GB) |
| provider raw LRU | 9.2 GB | 688 entries × 13.4 MB (16/layer × 43) |
| transient churn | ~12 GB | ~10,245 FP16 payload builds (50 MB each) held by glibc arenas + torch pinned/payload allocators; `ru_maxrss` is a high-water mark so post-hoc trim cannot undo it |

DS10 (2 GiB budget, raw LRU disabled) measured only 6.97 GB at the same
snapshot point — the 9.2 GB raw LRU plus missing per-step RSS hygiene are
both new in cache1.

### Measured cache behavior (v13, primary model, cold+warm)

- GPU cache: `hits: 0`, `loads: 5,252+4,993` — every access cold-stages; the
  per-token routed working set (~258 × 50 MB ≈ 12.9 GB) exceeds even the
  4 GiB budget, so the GPU level still turns over fully each token.
  (FFN-level `get()` hits for pinned shared experts are counted in per-layer
  `ffn_cache_counters`, not the cache-level reserve counter.)
- Provider raw LRU: **4,330 raw hits** vs DS10's 0 — HTTP fetches dropped
  10,202 → **5,872** (measured; sim predicted 5,308 at 16/layer).
- Decode was HTTP-bound: ~288 s/token (5,872 range fetches over ~15 tokens).

### CACHE1b fix (commit 710e82d, run 20260806T224102Z)

1. **Global byte cap** `raw_max_bytes=3 GiB` on the provider raw LRU
   (evicts the globally-oldest entry across ALL layers).  Sim: 16/layer
   uncapped ≈ 53% fetches; **cap 3 GiB ≈ 60%** — nearly the same hit rate
   at one third the host footprint (9.2 → 3 GB).
2. **Per-step RSS hygiene**: `post_step_hook` in `generate()` runs
   `gc.collect() + torch.cuda.empty_cache() + malloc_trim` after every
   forward (prefill + each decode step) so the process never crosses the
   12 GB ceiling mid-run (`ru_maxrss` cannot be trimmed afterwards).
3. **Release primary model before the alternate-budget build** (mirrors
   DS10 `stage_generate`; v13 skipped this and kept two full candidates
   resident during the rerun phase).
4. `cache1_policy_sim.py` gained a byte-cap mode; a unit test pins the
   global-byte-cap eviction semantics.

Expected v14 outcome: host RSS ≈ 6.4 + 3.0 + bounded transient < 12 GB,
HTTP fetches ≈ 3,474 (60%), provider hits ≈ 28.9% → **ACCEPT_CACHE_PARTIAL**.

## 8. v14 remote run (CACHE1c, 2026-08-07T181447Z): REJECT_MEMORY — diagnosis

Second implementation (commit `710e82d`, raw cap 3 GiB + per-step hygiene
+ release-before-alternate) ran on Modal dual-T4 with a **16 h timeout and
a 10-min heartbeat** (liveness + GPU util/mem + harness RSS).  It failed
the primary memory gate again: **steady decode RSS ~13.3 GB, peak 15.67 GB
vs the 12 GB ceiling**, so `cache1_memory_primary` → REJECT_MEMORY even
though the workload stayed healthy (harness_alive=True throughout; build
finished ~1.3 h, then HTTP-stalled decode at 0% GPU util).

### Why 3 GiB cap was still too fat

The 3 GiB cap fixed only ONE RSS driver.  The measured steady state:

| component | bytes | notes |
|---|---|---|
| DS10 sealed baseline | 6.49 GiB | current, post-primary (peak 6.93) |
| raw LRU (cap 3 GiB) | ~3 GiB | fills during first-token fetch storm |
| **eager shared-FP16 host copies** | **2.06 GiB** | 43 layers × 48 MiB, built in `build_candidate`, retained for the whole run |
| torch pinned/staging churn | ~1.7 GiB | `pin_memory()` per staged payload + glibc arenas; per-step hygiene only trims to ~13.3 GB |

Peak 15.67 GB occurred at 19:34–19:55Z (first-token storm: raw LRU fills
+ eager shared copies already resident + staging).  `ru_maxrss` is a
process-wide high-water mark, so the peak is locked in regardless of
post-decode trimming.

### CACHE1d fix (local, ready to commit)

1. **Lazy shared-FP16 payloads**: `build_candidate` no longer builds all
   43 shared FP16 payloads eagerly.  The layer builds the payload on first
   forward (`provider.get_shared_fp16_payload`), stages + **pins** it, then
   frees the host copy (`shared_payload = None` and pops the provider's
   copy).  Pins survive `reset_state` (cache is not cleared), so the cold==
   warm rerun hits pinned entries without re-building host payloads.
   Saves ~2.06 GiB steady AND cuts the first-token build peak.
2. **Raw cap 3 → 2 GiB**: sim shows 1.5 GiB is a hard cliff (0% provider
   hits — global LRU thrashes); 2 GiB keeps 22.6% provider hits / 66.2%
   HTTP vs 29.0% / 59.9% at 3 GiB.  One GiB saved.

Expected v15: steady RSS ≈ 6.49 + 2.0 + ~1.7 churn ≈ 10.2 GiB, peak
≈ 11.5 GiB → **under the 12 GB ceiling with margin**; HTTP ≈ 3,830
(66.2%), provider hits ≈ 22.6% → `ACCEPT_CACHE_PARTIAL`.

## 9. v15 remote run (Kaggle dual-T4, 20260808T183416Z): REJECT_CACHE_CORRECTNESS — false positive

Third implementation (commit `a756dd1`, lazy shared FP16 + raw cap 2 GiB +
per-step hygiene + runner v3) ran on **Kaggle (free weekly quota)** with a
20 h timeout and hardened heartbeat.  All decode passes completed
(primary 16 tokens + warm 16 tokens), then the evidence stage crashed in
`runtime_snapshot()` — `layer.ffn_fn.shared_payload.values()` on `None` —
because CACHE1d **frees the shared host FP16 copy after pinning**
(`shared_payload = None`).  The crash tripped the fail-closed exception
handler → verdict **REJECT_CACHE_CORRECTNESS** with zero remaining gates
computed, **even though every correctness gate that ran passed**.

### Measured (from the sealed evidence)

| gate | value |
|---|---|
| tokens_match_sealed_ds10 | **true (16/16 exact)** |
| cold_warm_equal | **true** |
| token_ids_in_vocab | true |
| peak host RSS | 11.24 GiB < 12 GiB ceiling ✓ |
| GPU allocated | 5.95 / 5.77 GiB ✓ |
| GPU hit rate (total) | **11.1%** (645/5789) — exactly the sim's pin-shared prediction |
| routed GPU hit rate | **0.0%** — every routed access cold-staged |
| shared pins | 43/43 hit every token (the only hits) |
| decode wall | 10,158 s primary / 13,032 s warm (HTTP-bound) |
| build | 587.5 s, 1,351 reqs, 7.23 GiB |

Per-token delta is constant at **+301 requests, +43 hits** — the 43 hits
are precisely the pinned shared experts; all 258 routed accesses miss at
GPU level every token.

### Verdict correction

This is **not** a genuine correctness rejection: `runtime_snapshot()`
never guarded for the CACHE1d-freed host payload.  Fix committed:
`deepseek_v4_model.py` `runtime_snapshot()` now tolerates `None`
`shared_payload` (counts 0 shared-host bytes).  The re-run (v16) should
produce the designed **ACCEPT_CACHE_PARTIAL** verdict with the full
CACHE1.5 metrics block computed.

### Why routed hits stay 0% (measured evidence, matches sim)

- Budget 2 GiB/GPU = 42 routed slots (48 MiB FP16 each), minus shared pins.
- Prefill (token_0) touches ~1,231 unique routed experts — the 42-slot
  cache churns completely before decode starts.
- Per-token routed working set ≈ 258 × 48 MiB ≈ 12.1 GiB ≫ 2 GiB budget,
  so LRU retains nothing across tokens (reuse distance ≫ capacity).
- Consecutive-token reuse of 37.5% is real but unreachable by retention
  at this budget; it must be captured by **provider raw LRU** (host) or
  **router-ahead prefetch** — the next campaign phase.

Artifacts: `ds10-cache1d-kaggle-20260808T183416Z/` (evidence archive).

## 10. CACHE1f (2026-08-10): tensor-level LOCAL weight staging -- design & expected numbers

User proposal: pre-load checkpoint shards as Kaggle datasets / kernel-local
files so decode reads are local instead of HTTP.  Measured reality:

- Full checkpoint: 166.9 GB / 48 shards (committed headers).  Kaggle dataset
  cap 20 GB -> ~9 datasets; account storage cap ~100 GB -> does not fit.
- Minimal SHARD set for the sealed 16-token trace: 45/48 shards = 145.3 GiB
  (the canonical trace touches 2,365 unique (layer, expert) = 14,190 unique
  routed-expert tensors; prefill alone is 7 positions x 6 experts x 43 layers).
  The kernel's /kaggle/working + /kaggle/input share ~19.5 GiB -> shard-level
  staging is IMPOSSIBLE (11.6% coverage at 19.5 GB).
- TENSOR-level staging (partial safetensors per shard, staged tensors only):
  the full needed set is 15,754 tensors = 37.69 GiB.  Greedy by access
  frequency within the kernel disk:
      budget 12 GiB -> 42.3% | 16 GiB -> 60.9% | 18 GiB -> 66.6%
      19.5 GiB -> ~69.5% | 24 GiB -> 80.7% | 38 GiB -> 100%
  Must-stage build-time set (dense 5.26 + shared 1.01 + top-level 1.97 =
  8.24 GiB) is staged FIRST, so ALL build-time HTTP is eliminated; only the
  decode-time routed long tail remains remote (HybridTensorSource fallback).

Implementation (commit a156a9e): scripts/deepseek_v4_staging.py (manifest
builder, partial-shard writer, HybridTensorSource local-first/HTTP-fallback),
committed cache1-staging-manifest.json (15,754 tensors, 37.69 GiB),
runtime _build_source_with_staging() in stage_cache1, staging gates.  v18
queued after v17.  Correctness is invariant: staging is a pure performance
change; any unstaged tensor falls back to the sealed remote path, and the
CACHE1.4 gates (tokens==SEALED_DS10_TOKENS, cold==warm, deterministic
rerun, memory ceilings) all still apply.  Local tests 5/5 staging + 40 pass
1 skip (model/runtime/support/cache).
