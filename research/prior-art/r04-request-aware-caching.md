# R4 — Request-aware caching: MoE-Infinity-style trace requirements, CPU router-replay feasibility, and the scan-resistant completeness replay

**Track:** R4 (prior-art sweep, dee-serve request-level caching question)
**Branch:** `research/prior-art-r04` — base `dc78dc4`
**Status:** analysis + sealed-gated offline replay only. No remote spend, no
integration edits, no GPU. Every number carries its tier label; "derived"
means arithmetic on sealed inputs.

## TL;DR

1. **A "router-only" CPU replay of arbitrary prompts is exact only for the
   three hash-routed layers (0–2)** — selection there is `tid2eid[input_ids]`,
   a pure token-id lookup needing no hidden state. For the 40 score layers
   (3–42) the router input is the layer's true `ffn_norm_out` hidden state,
   which exists only after executing every prior layer's attention + MoE.
   No hidden states exist in any sealed artifact; the v50 journal records
   expert ids only. So: **multi-request *exact* traces require executing the
   model — there is no checkpoint-free router replay.** (§3)
2. **The cheapest lawful multi-request trace source is one Kaggle CPU batch
   running the existing Python reference model on CPU** (all components
   already exist and are device-agnostic): estimated ~2–4 min/request on the
   mounted checkpoint dataset → ~150–350 requests per 12 h session, versus
   ~10–30 requests if they are piggybacked on GPU Batch #2's wall clock.
   Hash-layer-only replay (exact, 3/43 layers ≈ 7.6 % of accesses) is nearly
   free and is included as an arm. Pre-flight proposal in §6 — PROPOSAL ONLY,
   no spend authorized or implied.
3. **Completeness replay (new tool, sealed-counter-gated):** on the sealed
   single-request window, canonical ARC *ties* LRU almost exactly
   (2,057 vs 2,056 hits @642 slots); LIRS, 2Q, SLRU and W-TinyLFU all *lose*
   to LRU on the cold stream. The v4 sim's `arc` row understated ARC by up
   to 9 pp — a real (non-verdict-changing) implementation gap worth noting.
   **But** under two multi-request probes — an identical-request repeat and a
   full-universe scan flush — LIRS / W-TinyLFU / canonical ARC beat LRU by up
   to +27 pp (77.8 % vs 50.8 % @16 GiB repeated; 77.8 % vs 50.8 % post-scan
   survival @16 GiB). Scan/frequency-awareness is real, and its payoff is a
   *cross-request* property — which is exactly the property the missing
   multi-request corpus would measure. (§5)
4. **Verdict: the Phase-2 host-LRU selection stands for the cold
   single-request regime** (nothing here reopens it — no causal policy beats
   LRU on the sealed window). Request-aware/MoE-Infinity-style caching remains
   correctly a Phase-5 question, and this card now prices the trace collection
   it requires.

## 1. Prior art: what MoE-Infinity actually needs (trace spec)

MoE-Infinity (arXiv 2401.14361; source: TorchMoE/EfficientMoE repos) builds
on **sequence-level expert activation tracing**:

- Per request (sequence), record the **Expert Activation Matrix (EAM)**:
  which experts fire, per layer, across that request's forwards. Separate
  EAM per sequence — their key insight is that activation locality is a
  *per-sequence* property that disappears when traces are pooled.
- Offline, an **EAM-selection algorithm** picks a small library of
  representative EAMs per task class.
- Online, a new request is matched to a representative EAM, which then
  drives (a) activation-aware **prefetch** (fetch experts the EAM expects
  before demand) and (b) activation-aware **eviction** (drop what the EAM
  says the request won't reuse). Reported 3.1–16.7× per-token latency
  gains over vLLM/DeepSpeed-class offload baselines (their hardware, their
  models — not dee evidence).

Translated to dee: an EAM is exactly one v50-schema route journal
(`dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_native_generate.py:342-450`,
`RoutedExpertJournal`: per `(forward_step, layer)` the per-token-row top-6
expert id matrix). So the **trace requirement is N diverse route journals** —
the schema already exists and needs no changes.

Derived statistics a request-aware study would then compute:

| Statistic | How from journals | Question answered |
|---|---|---|
| Cross-request overlap (Jaccard / shared-set size) | per-request unique (layer,eid) sets | does a persistent host tier earn anything across requests? |
| Request-level popularity | per-request activation count per record | how stable is the top-N (prewarm rank stability — the open regime-C caveat) |
| EAM predictability | best-match EAM coverage of a held-out request | can a *prior* request's trace predict a new request's expert set (MoE-Infinity's mechanism) |
| Miss-stream novelty per request | per-request cold records under a shared cache of size X | cross-request miss rate vs budget — the actual serving curve |
| Decode-step conditional reuse | route(t) vs route(t−1) overlap per request | whether prev-token hints ever work in serving (cold single-request answer was 0.54 % miss-stream recall — NO) |

Minimum viable corpus: order 10² requests × (prompt + ~16 decode tokens),
spread over ≥3–5 task classes (Q&A, code, summarization, long-document,
multilingual), because every one of the above is a statement about a request
*distribution*, not a single request.

## 2. What the sealed single-request analysis already settled (inputs used)

Read before this analysis (all verified at the cited commits):

- `research/phase2-ws-policy` @ `95dfe0d` + `8c921ff`/`6904c31` (docs in the
  wsm/wsf worktrees): LRU within 2.8 pp of Belady @16 GiB; recency ceiling
  53.64 % (every repeat at stack distance ≥ 213); host knee ≈16 GiB;
  regime-C prewarm deferred to Phase 5.
- `TIER_REPLAY_VALIDATION.md` @ dc78dc4: real C++ tiers reproduce the sim and
  the sealed live counters exactly; pooled LRU counters per budget; the
  `cache_batch = slots` staging convention.
- `research/phase2-legal-prefetch/LEGAL_PREFETCH.md` @ 4d7fddb: legal
  ahead-of-router sources bounded (oracle ≤ ~29 % of decode fill wall; hash
  layers 0–2 exact but ≤1.6 s); miss-stream prev-token recall 0.54 % →
  speculative hints NO-GO *on this workload*.
- `research/route-pipeline/OFFICIAL_LOOKAHEAD.md`: official lookahead = 0
  layers — route(L+1) chains on combine(L) through the residual stream.

Sealed journal anatomy (recomputed here, `routed_experts.jsonl` sha256
`665aac3e…ae1`, 688 records = 16 forwards × 43 layers, topk 6):
engine-dedup stream 5,099 accesses / 2,364 unique (layer,expert) records /
2,735 repeats; **1,429 of 2,364 records (60.4 %) never repeat** — the window
is majority-one-shot, i.e. it already contains intrinsic scan structure.
Hash layers 0–2 contribute 295 unique records / 386 accesses (7.6 % of the
stream).

## 3. Router-only CPU replay: feasibility verdict

### 3.1 The two router classes (source-verified)

From `dee.cpp/scripts/deepseek_v4_layer_common.py:349-398` (`router_select`,
the official `Gate.forward`) and `deepseek_v4_support.py:427-431`:

- **Hash layers 0–2** (`n_hash_layers = 3`, `deepseek_v4_model.py:298`):
  selection = `tid2eid[input_ids]` — an I64 `[vocab, 6]` table
  (`layers.{0,1,2}.ffn.gate.tid2eid`). Needs **only token ids**. Scores/weights
  still come from `sqrt(softplus(x·W))`, but weights do not enter the journal
  or the cache demand stream.
- **Score layers 3–42**: `scores = sqrt(softplus(x @ gate_wᵀ))`;
  selection = `topk(scores + gate_b)` — needs `x`, the layer's FFN-norm
  input hidden state `[4096]`, plus `gate.bias` (a real F32[256] tensor per
  layer — confirmed present in shard headers, e.g. layer 20
  `ffn.gate.bias [256] F32`; DeepSeek-V3-family aux-loss-free bias is trained
  and generally nonzero at inference).

### 3.2 The hidden-state dependency is total

`x` at layer L is a function of the residual stream after layers 0..L−1 —
i.e., of every prior layer's attention **and routed-expert outputs**
(`deepseek_v4_layer_reference.py:595-649`: `_hc_pre → rms_norm → attn →
_hc_post → _hc_pre → rms_norm → ffn → _hc_post`). There is no shortcut: the
sealed evidence contains **no hidden-state dumps** (the journal stores only
`expert_ids_rank_order`; `generated_checkpoint.jsonl` stores token ids and
engine counters). Feeding an approximate hidden state produces approximate
routes — top-6-of-256 selection is sensitive to small input perturbations at
the rank boundary — and no bound on that divergence exists without running
the model anyway. **Only hash-routed layers are replayable without compute.**

### 3.3 Existing CPU-callable surfaces (checked, all insufficient alone)

| Surface | CPU? | Faithful to authoritative DSv4 routing? |
|---|---|---|
| `Engine::route_topk{,_batch}` (engine.cpp:2295-2466; CPU path :2419-2432) | yes (when `use_cuda=false`) | **NO for DSv4**: computes bias-free softmax top-k. Selection is monotonic-equivalent to `sqrt(softplus)` *only if* `gate.bias` were absent/zero — it is present per-layer. No hash path at all. Dev/test-only consumer (`tests/test_real_router.cpp`, `test_router_cuda.cpp`); production routing is the torch `router_scores` call in the FFN (`deepseek_v4_layer_candidate.py:100-102`). |
| `Engine::moe_forward_experts` host path (engine.cpp:433-448) | yes (`swiglu` on a host-staged blob) | executes given ids — needs ids first |
| `router_select` / `router_scores` (torch, layer_common.py / expert_reference.py) | yes (pure torch) | **YES — this is the authoritative implementation** (sqrtsoftplus + bias + tid2eid) |
| `DeepseekV4Model` reference harness (`deepseek_v4_model.py`) | yes in principle — pure torch; `_handoff` degrades to `.to(dst)` on CPU (`model.py:787-797`); CUDA events/profiling are all `is_cuda`-gated | yes — it *is* the model; produces hidden states by real forward |

So a CPU trace collector is buildable from verified parts
(`LocalDirTensorSource` over the dataset mount → `layer_dense_tensor_names`
→ `build_layer_weights_from_tensors` → `DeepseekV4Layer` with a small
route-recording FFN wrapper calling `moe.moe_layer_forward` →
`DeepseekV4Model.forward/generate` with `diagnostics=False` + a
journal writer emitting the v50 record schema). Estimated new code
~150–250 lines of glue; no engine changes, no CUDA.

### 3.4 Feasibility verdict

| Question | Answer |
|---|---|
| Can we replay *routes* for arbitrary prompts on CPU without executing the model? | **NO** for layers 3–42 (hidden-state dependency, structural). **YES** for layers 0–2 (hash table lookup). |
| Can the *sealed journal or harness* produce router inputs for arbitrary prompts? | No — no hidden states stored anywhere; harness produces them only by running. |
| Can we run the *whole model* on Kaggle CPU? | Yes in principle: pure-torch reference path + dataset-mounted shards; bounded by expert-tensor fetch bandwidth and host RAM, not compute. Costed in §4/§6. |
| Is an approximate-hidden-state replay worth anything? | Only as a labeled sensitivity experiment (measure top-6 divergence vs the sealed prompt's journal). Never as exact trace evidence. |

## 4. Cheapest multi-request trace paths, costed

Cost model constants (all sealed or header-verified): expert record =
13,369,344 B packed (12.75 MiB); official compact tensor set per expert ≈
12.6 MiB (3 packed-FP4 I8 mats + F8 scales, `deepseek_v4_support.py:409-416`);
dense+shared weights ≈ 143–153 M params/layer measured from committed shard
headers (~6.3 B total ≈ 12.6 GiB at FP16, ~25 GiB at FP32 — the reference
`dense()` dequantizes to FP32, so the harness must cast to FP16/BF16 to fit
the ~29 GiB Kaggle CPU host); per-token compute ≈ 2 × 12.8 B active params ≈
26 GFLOP plus small attention terms.

**Path P0 — hash-layer replay (exact, partial, ~free).**
Tokenizer assets are in-repo and SHA-pinned
(`benchmark_reports/deepseek-v4-flash-0731-t4/tokenizer-assets/`;
`deepseek_v4_encoding.py` — "Nothing here requires CUDA, a GPU, or the model
checkpoint"). The only missing bytes are the three `tid2eid` tensors
(~6 MB each ≈ 19 MB total), fetchable via `RemoteTensorSource` range GETs
or read from the dataset mount inside a CPU run. Output: exact expert-id
sets for layers 0–2 over an *arbitrarily large* prompt corpus. It directly
answers "how much do hash-layer expert sets overlap across requests" — a
lower bound on cross-request reuse — but covers only 3/43 layers and is
structurally *token-local* (context-independent), so it must not be
extrapolated to score layers. **Cost: zero compute spend; optionally folded
into any CPU run.**

**Path P1 — Kaggle CPU full-forward tracer (exact, complete; the proposal).**
Reference-model forward on CPU with journal writer (§3.3). Per request
(prompt ≤ ~64 tokens + 16 decode tokens):
- compute ≈ 26 GFLOP/token at an assumed 30–80 GFLOP/s (4-vCPU Kaggle CPU,
  FP32/MKL-class) → ~0.4–0.9 s/token → ~25–70 s per 23-forward request;
- expert-tensor fetches ≈ unique records/request × 12.6 MiB; a sealed-like
  request touches ~2,364 records ≈ 29.8 GiB cold → at 0.15–0.5 GiB/s
  dataset-mount/host-disk reads ≈ 60–200 s; a shared host cache of
  ~8–13 GiB (~650–1,000 records) recovers whatever cross-request overlap
  exists — which is itself the measurement;
- net ≈ **2–4 min/request → ~150–350 requests per 12 h Kaggle CPU session**,
  the right scale for an EAM library (MoE-Infinity itself uses trace sets of
  tens of sequences per class).
- RAM: FP16 dense ~13 GiB + ~8–13 GiB expert cache + transient FP16
  payloads — inside 29 GiB, tight but precedented (the sealed run lived
  under a 12 GiB host budget on this code path).
- Fidelity note: the tracer *is* the producer of record for each new
  prompt's routes (authoritative router, real weights, deterministic greedy
  decode); matching the sealed GPU run bit-for-bit is not required — each
  new prompt has no other truth. Recommended: mirror the sealed pipeline's
  dtype recipe anyway (dense FP16, embed/head BF16) to keep the numeric
  regime identical.

**Path P2 — GPU Batch #2 piggyback (exact, small, conditional).**
If the reserved GPU batch runs Phase-3 arbitrary-prompt inference, each
prompt's journal is produced free by the existing `RoutedExpertJournal`
wiring — zero marginal spend, but bounded by T4 wall time (~86 s+ fill per
response) → realistically ~10–30 requests, and gated on a GPU decision that
this card does not spend.

**Path P3 — synthetic/bootstrap streams (free; methodology only, never
evidence).** The repeat3 and scan probes shipped here are the honest version
of this: they measure *policy mechanics* under request boundaries, not real
cross-request statistics. Also possible: parametric stream generators fit to
sealed anatomy (popularity, per-layer unique span, repeat structure) — useful
to develop EAM-selection/sim tooling before the corpus exists, clearly
labeled synthetic.

**Rejected shortcut — approximate-hidden-state replay** (e.g., zeroed or
stale MoE outputs): produces wrong routes by construction; its only use is
as a labeled divergence-measurement arm, and it saves no fetch bandwidth
(the dense backbone is cheap; experts dominate the cost).

## 5. Completeness replay: ARC/LIRS/2Q/SLRU/W-TinyLFU + scan and repeat probes

Tool: `tools/phase2_scan_resistance_replay.py` (new, stdlib-only).
Output: `research/prior-art/results/r04_scan_replay.json`
(+ `r04_scan_replay_hir10.json` sensitivity). **Admission gate: 14/14 PASS**
— pooled LRU reproduced the published sim counters exactly at all six slot
counts (642→2056/3043/2401 … 2570→2735/2364/0), per-scope 682-slot LRU
reproduced sim exactly and sealed live within ±1 (cuda0 1222/1391/709 vs
sealed 1223/1390/708; cuda1 1395/1091/409 exact), Belady = 2,735 hits at
every budget ≥ 642. All rows below are post-gate.

### 5.1 Cold single-request window (probe `sealed`; hit %, pooled host tier)

| Policy | 4 GiB (321) | 8 (642) | 12 (963) | 16 (1285) | 20 (1606) | 24 (1927) | 32 GiB (2570) |
|---|---|---|---|---|---|---|---|
| lru (control) | 25.75 | 40.32 | 46.60 | 50.83 | 52.32 | 53.09 | 53.64 |
| **arc (canonical)** | 25.79 | 40.34 | 46.97 | 50.83 | 52.32 | 53.09 | 53.64 |
| arc_v4 (v4-sim variant) | 20.75 | 31.30 | 42.91 | 50.44 | 52.23 | 53.09 | 53.64 |
| lirs (hir 1 %) | 28.34 | 37.18 | 40.97 | 44.22 | 48.13 | 51.72 | 53.64 |
| lirs (hir 10 %) | 27.32 | 36.50 | 41.18 | 45.75 | 49.54 | 52.66 | 53.64 |
| 2q | 0.00 | 31.52 | 35.05 | 38.75 | 40.93 | 41.58 | 42.99 |
| slru (20/80) | 0.00 | 0.00 | 0.00 | 32.44 | 38.62 | 41.95 | 44.13 |
| wtinylfu_exactfreq | 27.16 | 36.26 | 41.09 | 44.30 | 48.17 | 51.74 | 53.64 |
| belady (bound) | 47.17 | 53.64 | 53.64 | 53.64 | 53.64 | 53.64 | 53.64 |

Findings:

- **Canonical ARC ≈ LRU pointwise** (Δ ≤ +1 hit everywhere ≥ 8 GiB) — it is
  the only scan-resistant policy that fully tracks LRU here. The v4 sim's
  `arc` row (31.30 % @8 GiB) understated ARC by ~9 pp: its `p` update fires
  on T1 hits and its B-ghost deltas differ from the canonical rule. The
  published verdict is unaffected (canonical ARC still does not *beat* LRU),
  but the matrix should be read with this correction. Recorded for R12
  synthesis.
- **LIRS loses to LRU at every sub-saturation budget** (−3.1 to −6.6 pp) and
  is robust to the HIR fraction (1 % vs 10 % rows). Mechanical reason: the
  stream's minimum reuse distance is 213 while a LIRS HIR resident window is
  ~1–10 % of slots — re-referenced records have already drained from Q, so
  the recency that LIRS protects is the wrong signal here, same as ARC's.
- **2Q and SLRU cliff at the reuse-distance floor.** SLRU scores literally
  0 % until its 20 % probation segment exceeds ~213 records (≥1285 slots);
  2Q recovers only via its ghost list. A segmented policy whose "prove
  yourself" window is smaller than the workload's minimum reuse distance can
  never earn a hit — structural, not a tuning issue.
- **W-TinyLFU (exact-frequency variant)** loses ≤ LRU everywhere cold
  (−4 to −6.5 pp at 8–24 GiB): most records have frequency 1–2 (5,676
  activations / 2,364 records ≈ 2.4 mean), so the admission filter is noise
  on a cold window.
- Net: **on the sealed single-request window, no scan-resistant policy
  beats LRU; canonical ARC ties it.** The Phase-2 selection is unchanged and
  now survives the completeness check it was missing.

### 5.2 Identical-request repeat (probe `repeat3`: sealed stream ×3)

Per-pass hit % — the dee-serve ceiling when the next request is literally
this one:

| Policy | pass2 @8 GiB | pass2 @16 GiB | pass2 @32 GiB |
|---|---|---|---|
| lru | 41.5 | 55.6 | 100.0 |
| arc | 46.0 | 72.0 | 100.0 |
| lirs | 45.6 | **77.8** | 100.0 |
| 2q | 49.5 | 64.2 | 76.9 |
| slru | 0.0 | 53.5 | 65.1 |
| wtinylfu_exactfreq | 54.8 | 72.0 | 100.0 |
| belady | 63.9 | 78.8 | 100.0 |

**Once frequency history exists (pass ≥2), LIRS ≈ Belady (77.8 vs 78.8 %
@16 GiB) and beats LRU by +22.2 pp.** LRU cannot converge below full
working-set capacity because the request's access order thrashes it; LIRS
locks the low-IRR core after one observed pass. This is the honest
*mechanics* version of the MoE-Infinity bet: cross-request reuse is worth
~+20 pp at mid budgets **if** consecutive requests share a stable expert
set. The size of that "if" is exactly what a real corpus measures — the
sealed window cannot say whether two different requests overlap like two
identical ones.

### 5.3 Scan pollution (probe `scan`: sealed → one pass over the 8,644
untouched universe records → sealed again)

Post-scan second-pass hit % (scan hits = 0 everywhere — all one-shot):

| Policy | p2 @8 GiB | p2 @16 GiB | p2 @32 GiB |
|---|---|---|---|
| lru | 40.32 | 50.83 | 53.64 |
| arc | 47.77 | 71.58 | 71.97 |
| lirs | 45.60 | **77.82** | 100.00 |
| 2q | 49.77 | 62.29 | 64.29 |
| slru | 0.00 | 53.19 | 63.48 |
| wtinylfu_exactfreq | 49.48 | 71.97 | 100.00 |
| belady | 63.91 | 78.82 | 100.00 |

**LRU retains nothing through the scan** — its post-scan pass is a byte-
exact cold replay (p2 hits = p1 hits at every budget: 2,056 / 2,592 / 2,735).
An 8,644-record foreign burst costs an LRU cache the entire rebuilt working
set: at 16 GiB, post-scan re-fetch = 2,507 misses ≈ 31.2 GiB of bank traffic
(derived: misses × 12.75 MiB), i.e. ~107 s @0.29 GiB/s. LIRS and W-TinyLFU
keep the protected set (~+27 pp post-scan at 16 GiB); canonical ARC sits
between (+20.8 pp). dee-serve relevance: continuous batching interleaves
requests constantly — every novel request IS this scan.

### 5.4 What this completeness check does and does not say

- DOES: the host-LRU choice is now checked against the canonical
  scan-resistant family, and the sealed window's verdict survives (LRU is
  optimal-or-tied on the cold stream).
- DOES: it quantifies the *mechanism* request-aware caching would exploit —
  frequency/protected-residency earns +20–27 pp over LRU exactly when the
  workload has request-to-request reuse.
- DOES NOT: say anything about whether real DSv4 requests share expert sets
  — that needs the §6 corpus. The repeat3/scan probes are bounds, not
  measurements of cross-request overlap.

## 6. Pre-flight proposal — multi-request trace collection (PROPOSAL ONLY)

Per AGENTS.md accounting format. Nothing here is authorized; ledger
recommendation is CPU 6/10 or later.

```text
RUN TYPE:   CPU batch (Kaggle), trace collection — NO GPU
RUN NUMBER: CPU 6/10 (next free reserve slot)
QUESTION:   What does the DSv4 multi-request expert-activation distribution
            look like — i.e., the input MoE-Infinity-style request-aware
            caching (EAM matching, activation-aware prefetch/eviction,
            regime-C prewarm rank stability) requires before it can be
            evaluated at all?
WHY LOCAL EXECUTION IS INSUFFICIENT:
            Requires the ~284 GB official checkpoint (48 safetensors
            shards). Local disk has ~24 GB free; residential HF pull is
            too slow (AGENTS.md). On Kaggle the checkpoint is an existing
            mounted dataset (/kaggle/input/deepseek-v4-flash-0731-shards)
            readable by the committed LocalDirTensorSource. Local execution
            was used for everything else in this card (the sealed-journal
            replay) — only checkpoint access forces Kaggle.
ARMS INCLUDED (same batch):
  A0 smoke    1 request, the sealed canonical prompt, 4 decode tokens.
              Gate: journal ids for forwards 0-4 must equal the sealed
              v50 journal prefix for those forwards (prefill rows must
              match exactly; decode rows must match for the produced
              tokens) — this is a REAL exactness check, since the CPU
              reference path was validated against the same pipeline that
              produced v50. Also verifies generate() on CPU end-to-end.
  A1 corpus   ~100-200 diverse prompts (mixed task classes: short Q&A,
              code, summarization, multilingual; prompt ≤ ~64 tokens,
              16 decode tokens, greedy). Per-request v50-schema journal +
              generated token ids + per-request unique-record count and
              fetch counters (provider.stats()).
  A2 hash     tid2eid-exact replay of the SAME corpus plus a much larger
              prompt-id corpus (thousands): layers 0-2 request-level
              activation matrices for free. Zero model execution —
              tokenizer + table lookup. Validates the P0 lower-bound path.
  A3 audit    cheap side-artifacts: per-layer max|gate.bias| (settles
              whether engine route_topk could ever serve as a CPU router),
              per-request wall split (fetch vs compute), host RSS envelope.
  (deferred, optional) A4 approximate-hidden-state divergence probe:
              replay the sealed prompt with routed-MoE outputs zeroed and
              report per-layer top-6 overlap decay vs the sealed journal —
              prices the "how wrong is skipping experts" question. Only if
              A0/A1 land early; clearly labeled approximate.
SUCCESS CRITERIA:
  - A0: CPU-produced journal == sealed journal on the overlapping prefix
    (record-for-record id equality). This is the exactness gate; if it
    fails the corpus is inadmissible.
  - A1: ≥ 100 request journals written, each passing the v50 record schema
    (chain-hashed), plus aggregate EAM stats: pairwise request overlap
    distribution, per-layer unique-expert span distribution, per-request
    cold-record counts.
  - A2: hash-layer request stats over ≥ 1000 prompts.
FAILURE CRITERIA / ABORT:
  - A0 mismatch → abort, artifact = the divergent journal + first-diff
    record (the corpus would be built on an unfaithful forward).
  - Fetch-bound throughput < ~15 requests/hour sustained → abort after
    smoke+10 requests and keep the partial corpus (still analyzable).
  - Host RSS > 26 GiB sustained → halve the expert cache and continue.
ARTIFACTS EXPECTED:
  - journals/<prompt_id>.routed_experts.jsonl (v50 schema, chain-sha256)
  - requests.jsonl: per-request {prompt hash, prompt_len, n_decode,
    unique_records, fetch_count, wall_s, rss_peak}
  - eam_summary.json: pairwise-overlap matrix stats, popularity ranks,
    per-layer span, per-class aggregates
  - hash_replay.jsonl (A2), gate_bias_audit.json (A3)
  - The corpus is the input for a FOLLOW-ON offline pass (free, local):
    re-run this card's replay tool per-request + cross-request LIRS/ARC/
    W-TinyLFU vs LRU on the real interleaved stream — i.e., the actual
    request-aware caching verdict.
ESTIMATED COST: 1 CPU batch slot; ~12 h wall; zero GPU; zero new code
  beyond a ~150-250-line collector harness (all components verified present
  and device-agnostic: LocalDirTensorSource, build_layer_weights_from_tensors,
  DeepseekV4Layer, moe_layer_forward, generate(), RoutedExpertJournal
  schema).
```

Non-goals for that run: no engine code, no CUDA, no dee.cpp tier changes,
no model-modifying shortcuts. The journal writer is observation-only.

## 7. Falsification & open checks

- **The repeat3/scan advantage is a bound, not a prediction.** If the A1
  corpus shows pairwise request overlap ≈ small (e.g., mean Jaccard < 0.1 on
  (layer,expert) sets), the +20–27 pp mechanisms collapse and plain LRU is
  also the right *serving* host policy — a clean falsification path.
- **Canonical-ARC vs v4-`arc` divergence** is recorded here; if R12/the
  ws-policy owners want, re-running `phase2_ws_policy_sim_v4.py` with the
  canonical p-update would produce a corrected `arc` row (expected: tie LRU
  within ~1 hit at every budget ≥ 8 GiB, −5 pp at 4 GiB). No verdict changes.
- **LIRS robustness:** checked at hir_ratio 1 % and 10 % (±1–1.5 pp); the
  sealed-window loss is structural (HIR window ≪ reuse distance 213).
- **SLRU's 0 % rows are the mechanism, not a bug**: probation capacity
  (slots/5) < min reuse distance (213) ⇒ no record can be promoted before it
  is evicted. Verified by the cliff landing exactly at ≥1285 slots.
- **Belady bound sanity:** belady hits (2,735 ≥642 slots; 2,405 @321) and
  post-scan/repeat behavior (100 % where the working set fits) are
  internally consistent — no realizable policy exceeds them anywhere.
- **Open check A (gate.bias):** the engine's CPU `route_topk` omits it; if a
  future CPU router-replay tool is ever built on pydee, A3's audit settles
  whether that matters (if max|bias| ≈ 0, softmax and sqrtsoftplus+bias give
  identical top-6 modulo FP ties; else it diverges).
- **Open check B:** hash-layer replay assumes `tid2eid` is constant per
  model — it is a frozen checkpoint tensor (I64), so yes; the A2 arm reads
  it directly rather than assuming.

## 8. Reproduce

```sh
# completeness replay (stdlib only, ~50 s)
python tools/phase2_scan_resistance_replay.py \
    --out research/prior-art/results/r04_scan_replay.json
# LIRS sensitivity
python tools/phase2_scan_resistance_replay.py --policies lru,lirs \
    --hir-ratio 0.10 --out research/prior-art/results/r04_scan_replay_hir10.json
```

Gate cells are compiled into the tool (exit 2, no output file, on any
mismatch). Journal: `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/
v50-evidence-20260829T195940Z/routed_experts.jsonl` sha256 `665aac3e…ae1`.

## 9. Sources

- MoE-Infinity: arXiv 2401.14361 (sequence-level EAM tracing, EAM-selection,
  activation-aware prefetch+caching; github.com/EfficientMoE/MoE-Infinity).
- ARC: Megiddo & Modha, FAST'03. LIRS: Jiang & Zhang, SIGMETRICS'02.
  2Q: Johnson & Shasha, VLDB'94. SLRU: Karedla et al. W-TinyLFU: Caffeine
  (Einziger et al.).
- In-repo evidence: `TIER_REPLAY_VALIDATION.md` @dc78dc4;
  `research/phase2-ws-policy` docs @95dfe0d/8c921ff/6904c31;
  `research/phase2-legal-prefetch/LEGAL_PREFETCH.md` @4d7fddb;
  `research/route-pipeline/OFFICIAL_LOOKAHEAD.md`; `dee.cpp/src/engine.cpp`
  (route_topk :2295-2466, CPU expert path :433-448);
  `dee.cpp/scripts/deepseek_v4_{model,layer_reference,layer_candidate,
  layer_common,moe_reference,expert_reference,support,encoding}.py`;
  `dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_native_generate.py`
  (RoutedExpertJournal :342-450, harness wiring :1600-1770).
