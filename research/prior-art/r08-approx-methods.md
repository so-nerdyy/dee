# R8 — Approximate-only MoE methods classified against the dee EXACT contract

Track: R8 (prior-art classification, classify-do-not-integrate).
Branch: `research/prior-art-r08`, worktree `.freebuff/wt/r08`.
Scope: methods that change WHAT executes (routing, expert set, precision,
stored representation) rather than WHERE/WHEN bytes move. All performance
figures are PAPER-REPORTED unless marked otherwise; none are dee evidence.

## 1. The contract being tested (canonical clauses)

dee-exact preserves (AGENTS.md §Exactness philosophy): authoritative routing,
expert identity, checkpoint representation semantics, expert execution,
expert ordering, outputs. Anything else may change freely (placement,
caching, scheduling, prefetch). For matrix use, violations are tagged:

- ROUTING — executed expert set or routing weights differ from the
  checkpoint router's top-k for the same hidden state (skip, adaptive-k,
  biased logits, predicted routing, router retraining).
- IDENTITY — the thing that runs is not the checkpoint expert E_i
  (substitute/buddy/virtual/merged/pruned expert).
- PRECISION — E_i runs at lower/different numeric precision than its
  checkpoint representation (re-quantized copies, sliced bits, surrogates).
- BYTES — the stored/served record is not the checkpoint record
  (re-quantized bank, decomposed factors, pruned manifest).

dee-exact legal surface (verified in tree): `ColdExpertStore` /
`ExpertStoreColdAdapter` (dee.cpp/include/dee/expert_tiers.h:13),
`StorageRecord{key,stored_bytes,exact_bytes,codec}` + `StorageCodec` /
`IdentityCodec` (dee.cpp/include/dee/host_expert_tier.h:37-64),
`HostExpertTier` + `HostPlacementPolicy`/`PlainLruHostPlacementPolicy`
(host_expert_tier.h:78-96,163-177), `DeviceExpertTier` +
`DevicePlacementPolicy` + `VramCacheManager` (expert_tiers.h:28-81),
`AsyncPrefetcher::prefetch_host_lease` (H2D), `Engine::route_topk` /
`route_topk_batch` (dee.cpp/include/dee/engine.h:373-377 — the
authoritative-router seam), `ExactExpertExecutor` = existing Engine FP4
kernels (PHASE2_INTEGRATION_MAP.md boundary table).

## 2. Classified methods

### 2.1 HOBBIT — mixed-precision expert offloading (APPROX-ONLY: PRECISION, BYTES)
- Ref: Tang et al., arXiv:2411.01433 (2024), on llama.cpp.
- What: token-level dynamic loading — on cache miss, serves a LOW-PRECISION
  replica of the missed expert instead of the real record (precision
  cascading: int4 for fp16, int2 for int8, chosen by gate-score threshold);
  layer-level adaptive prefetch driven by gating-input similarity across
  adjacent layers; sequence-level multidimensional cache policy.
- PAPER-REPORTED: up to 9.93× decode vs MoE-Infinity (Phi-MoE, Jetson AGX
  Orin); 13.0× decode vs llama.cpp + 79% prefill cut (Mixtral-8x7B, Jetson);
  3.21–3.92× vs MoE-Offloading/MoE-Infinity on RTX 4090; <1% accuracy drop
  (GSM8K, TruthfulQA). HW/model: Jetson AGX Orin (32GB unified), RTX 4090
  24GB + NVMe 980 PRO; Mixtral-8x7B, Phi-MoE.
- Violation: an executed expert's VALUES differ from the checkpoint
  (PRECISION), and the system stores extra derived copies (BYTES). dee
  already proved packed FP4 entropy-dense and byte-exact — serving an int2
  replica of a 12.75 MiB DEE4 record is a different model.
- Exact-safe salvage: YES — the layer-level similarity predictor is a
  prefetch HINT source (dee allows prediction→hint only); the multidim
  cache policy is a legal Host/DevicePlacementPolicy candidate (recency +
  sequence-level reuse). Neither requires executing predicted/substituted
  experts.

### 2.2 EdgeMoE — expert-wise bitwidth + preloading (APPROX-ONLY: PRECISION, BYTES)
- Ref: Yi et al., arXiv:2308.14352; IEEE TMC 2025. Code: mllm.
- What: non-expert weights resident in device memory, experts on external
  storage fetched on activation; OFFLINE per-expert bitwidth assignment
  (each expert stored at a profiled precision); statistical expert
  preloading pipelined with compute.
- PAPER-REPORTED: 1.19–2.77× speedup vs dynamic-loading/STI baselines;
  memory footprint −1.05–1.18× vs full-resident; enables >10B MoE on COTS;
  ≤2% accuracy loss. HW/model: Jetson TX2, Raspberry Pi 4B; Switch
  Transformer family (ST-base/large), 7 MoE LLMs.
- Violation: per-expert bitwidths change stored bytes and executed values
  (BYTES+PRECISION). The storage hierarchy itself (hot non-expert in RAM,
  cold experts on flash) is dee's own design — not a violation.
- Exact-safe salvage: preloading predictor → prefetch hint only; the
  "experts are bulky but cold" characterization (86.5% memory / 26.4%
  compute on ST-base-16) is already consistent with dee Phase-1.

### 2.3 AdapMoE — adaptive gating + prefetch + DP cache (APPROX-ONLY: ROUTING)
- Ref: Zhong et al., arXiv:2408.10284; ASPLOS'25. Code: PKU-SEC-Lab/AdapMoE.
- What: sensitivity-based adaptive gating reduces the number of activated
  experts per token/layer (Hessian/sensitivity threshold, ~25% fewer);
  activation-similarity prefetch; DP-based per-layer cache allocation;
  tile-wise scheduling.
- PAPER-REPORTED: −25% activated experts vs top-2, 1.35× over prior expert
  management, no measured accuracy degradation (MMLU/ARC-C, MT-Bench
  prompts). HW/model: Mixtral-8x7B on RTX 4090 + A6000; Mixtral-8x22B on
  A6000; quantized configs.
- Violation: ROUTING — fewer experts execute than the authoritative top-k
  named; a dropped expert contributes 0 to the output.
- Exact-safe salvage: the similarity prefetcher and DP cache-budget
  formulation are legal (hint + placement). The gating table itself is
  NOT salvageable — a per-layer "how many experts" decision is a router
  output change.

### 2.4 QMoE — sub-1-bit MoE compression (APPROX-ONLY: PRECISION, BYTES)
- Ref: Frantar & Alistarh, arXiv:2310.16795; MLSys'24. Code: IST-DASLab/qmoe.
- What: calibration-driven compression of expert weights to <1 bit/param
  in a custom co-designed format + bespoke GPU decode kernels.
- PAPER-REPORTED: SwitchTransformer-c2048 1.6T params → <160GB (20×, ~0.8
  bit/param) at minor accuracy loss, <1 day on one GPU; runs on 4×A6000 /
  8×3090 with <5% runtime overhead vs ideal uncompressed. HW/model: Switch
  family incl. c2048; A6000/3090 servers.
- Violation: PRECISION+BYTES at the checkpoint level — produces a NEW
  model artifact. dee-exact's source of truth is the checkpoint; QMoE's
  format could only ever be a dee-fast input bank.
- Exact-safe salvage: none at runtime. Design lesson only: format/kernel
  co-design is what dee's IdentityCodec boundary already anticipates — a
  lossy codec is structurally possible behind `StorageCodec`, and its
  output would carry a different `representation` string (fail-closed in
  exact mode, since IdentityCodec requires stored_bytes == exact_bytes).

### 2.5 Cache-aware routing family — the briefing's "CacheMoE" (APPROX-ONLY: ROUTING, +IDENTITY for substitution variants)
Naming note: the only paper literally titled "CacheMoE" (Sinthia et al.,
IEEE IoT J. 2025) is task-aware expert caching on edge nodes via
DWFL/MADRL demand predictors — placement-only, arguably hint-adjacent,
NOT a contract violation by itself. The briefing's clause "cache-aware
logit modification" matches the cache-aware ROUTING family; classified
together:
- (a) Mixture of Cache-Conditional Experts / "Cache-Prior" (Skliar et
  al., arXiv:2412.00099, TMLR'25, Qualcomm): boosts gating logits of
  DRAM-resident experts so the router prefers cached experts; training-free.
  PAPER-REPORTED: >50% cache-miss reduction, perplexity Δ0.1–3%, ≤2×
  on-device speedup vs LRU (Snapdragon phones 12/16GB, Qwen1.5-MoE-A2.7B
  4/8-bit; analysis on DSv2-Lite, Phi-3.5-MoE, Mixtral). Violation:
  ROUTING (logit bias changes which experts execute).
- (b) BuddyMoE (arXiv:2511.10054): offline co-activation table finds
  "buddy" experts; on prefetch miss, substitutes a cached buddy.
  PAPER-REPORTED: ≤+10% tps, negligible accuracy loss. Violation:
  ROUTING+IDENTITY (a different expert executes for the missed id).
- (c) ReMoE (arXiv:2605.27081): router fine-tuned toward temporally stable
  reuse. PAPER-REPORTED: +26% expert reuse; TPOT −43.6–49.8% (1.77–1.99×
  decode, llama.cpp Jetson Orin NX); +8.4% throughput under vLLM offload;
  DeepSeek/Qwen. Violation: ROUTING (checkpoint router replaced by
  fine-tuned router).
- (d) SMoE / importance-driven scheduling (arXiv:2508.18983): expert-cache
  router replaces low-score activated experts with similar cached ones.
  PAPER-REPORTED: −48% decode latency, >60% hit rate, ~lossless. Same
  ROUTING+IDENTITY violations.
- Exact-safe salvage: cache-residency awareness may steer PLACEMENT (what
  to keep hot / pin in policy_slots) and prefetch ORDER — never logits.
  The co-activation/redundancy tables (BuddyMoE) are a legal predictor
  input for hints. Falsification note: SliceMoE (below) reports
  Cache-Prior accuracy collapses under <5% miss-rate regimes — the family
  trades correctness for locality exactly where dee cares most.

### 2.6 Pre-gated MoE — selection/execution decoupling (APPROX-ONLY: ROUTING)
- Ref: Hwang et al., arXiv:2308.12066, ISCA'24 (Microsoft). Code:
  ranggihwang/Pregated_MoE.
- What: redefines the gate in MoE block N to select experts for block
  N+1 (pre-gating function), trained via fine-tuning; system overlaps
  CPU→GPU expert migration with execution using the one-layer lookahead;
  expert-aware buffer management.
- PAPER-REPORTED: only ~23% overhead vs oracular all-GPU execution while
  offloading experts to CPU; large block-latency advantages over
  MoE-OnDemand / MoE-Prefetch baselines; comparable-or-better task scores
  after fine-tune. HW/model: Switch-Base/Large (8–128 experts), single
  GPU + CPU offload; XSum/CB-WebQA/SQuAD.
- Violation: ROUTING — the executed set is produced by a REPLACED,
  retrained gate function, not the checkpoint router applied to the
  current layer's hidden state. Even where pre-gate ≈ router output,
  identity of the selection function changes.
- Exact-safe salvage: YES, the strongest salvage in this survey — a
  pre-gating-style per-layer predictor as a pure PREFETCH HINT keeps
  `route_topk` authoritative (dee already validated this pattern:
  "Edge0 shows a trained per-layer predictor could be much stronger —
  future speculative-prefetch only, never execution"). Its
  selection-decoupled-from-execution insight is already how dee's
  AsyncPrefetcher/staging works, derived independently.

### 2.7 MoE-I² — inter-expert pruning + intra-expert low-rank (APPROX-ONLY: IDENTITY, PRECISION, BYTES)
- Ref: Yang et al., arXiv:2411.01016, EMNLP Findings'24.
- What: layer-wise genetic search with non-uniform pruning ratios removes
  whole experts; remaining experts low-rank decomposed with non-uniform
  rank allocation.
- PAPER-REPORTED: >50% parameter reduction with maintained zero-shot
  performance. Model: Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite, Mixtral-8x7B
  (GPU eval).
- Violation: removes experts (IDENTITY+BYTES: pruned model, fewer
  records) and replaces weights with factorized approximations
  (PRECISION/IDENTITY).
- Exact-safe salvage: none (offline model surgery). Its per-layer
  importance scores could rank PREFETCH order in an exact system —
  marginal.

### 2.8 NAEE / Expert_Sparsity — pruning + dynamic skipping (APPROX-ONLY: IDENTITY, ROUTING)
- Ref: Lu et al., arXiv:2402.14800, ACL'24. Code: Lucky-Lance/Expert_Sparsity.
- What: post-training expert pruning (keep r of 8 per layer; ~30–90 min
  calibration) + dynamic expert skipping (drop e1 when w_e1 < β·w_e0,
  per-layer β from calibration medians).
- PAPER-REPORTED: Mixtral-8x7B r=6 → single A100-80G (was 2×), 1.2×
  speedup, −2.9pts task-agnostic / −6.2pts (−1.6 w/ fine-tune)
  task-specific; r=4 → 1.27×, ~50% param cut, beats Wanda 2:4 on same
  budget; dynamic skipping 1.2–1.3× alone.
- Violation: pruning = IDENTITY+BYTES (records gone); skipping = ROUTING
  (a router-selected expert is dropped).
- Exact-safe salvage: the w-ratio statistics could prioritize prefetch
  order (hint). The skip rule itself can never run in exact mode —
  dee executes every expert the router names.

### 2.9 MC-SMoE — merge then compress (APPROX-ONLY: IDENTITY, PRECISION, BYTES)
- Ref: Li et al., arXiv:2310.01334, ICLR'24 Spotlight.
- What: routing-statistics-guided expert merging (permutation alignment,
  dominant-expert groups, activation-frequency-weighted merge) then
  low-rank + structured-sparse decomposition.
- PAPER-REPORTED: ≤80% memory and ≤20% FLOPs reduction, 8 benchmarks
  (smaller SMoE backbones).
- Violation: merged expert is not any checkpoint expert (IDENTITY);
  decomposition changes values (PRECISION); model is rewritten (BYTES).
- Exact-safe salvage: none. Routing-frequency data = prefetch-priority
  hint input only.

### 2.10 SiDA-MoE — hash-predicted activation (APPROX-ONLY: ROUTING)
- Ref: Du et al., arXiv:2310.18859, MLSys'24. Code: timlee0212/SiDA-MoE.
- What: parallel hash-building thread maintains a table mapping input
  batches → per-layer activated-expert sets (offline-trained LSTM with
  classification heads); GPU hosts only predicted-active experts, rest in
  main memory.
- PAPER-REPORTED: ≤3.93× throughput, ≤72% latency cut, ≤80% GPU memory
  saved, ≥1% performance drop floor; up to 80% experts idle on
  Switch-base-256. HW/model: Switch/NLLB-MoE family, server CPU+GPU.
- Violation: ROUTING — a learned hash function decides WHICH experts
  activate (predicted routing replaces the router). On hash miss the
  wrong set runs.
- Exact-safe salvage: YES — the hash predictor is a prefetch-hint source;
  a wrong hint wastes bandwidth but cannot change the executed set. This
  is the canonical "their predictor, our hint" pattern.

### 2.11 AdaMoE — token-adaptive routing w/ null experts (APPROX-ONLY: ROUTING)
- Ref: Zeng et al., arXiv:2406.13233, EMNLP Findings'24.
- What: adds FLOPs-free "null experts" to the expert set, increases k,
  load-balancing loss on nulls → per-token adaptive expert count;
  requires fine-tuning of the checkpoint.
- PAPER-REPORTED: −14.5% FLOPs with +1.69pts ARC-C on fine-tuned
  Mixtral-8x7B.
- Violation: ROUTING (executed count/set differs from checkpoint top-k;
  expert set itself modified = IDENTITY too). Also a training-time method
  — dee-exact consumes frozen checkpoints.
- Exact-safe salvage: none (model modification).

### 2.12 SliceMoE — bit-sliced expert caching (APPROX-ONLY: PRECISION, BYTES)
- Ref: arXiv:2512.12990 (2025).
- What: Dynamic Bit-Sliced Caching — experts cached at slice granularity,
  precision assigned on demand; Calibration-Free Asymmetric Matryoshka
  Quantization keeps low/high-bit slices compatible (a low-bit prefix
  EXECUTES when the high-bit rest isn't resident); Predictive Cache
  Warmup reshapes cache during prefill.
- PAPER-REPORTED: −2.37×/−2.85× decode energy and 1.81×/1.64× latency
  (DSv2-Lite / Qwen1.5-MoE-A2.7B edge), near-high-bit accuracy.
- Violation: an expert may execute at truncated precision — same
  correctness class as HOBBIT's low-precision substitute (PRECISION),
  with derived stored copies (BYTES).
- Exact-safe salvage: PCW (prefill-time cache warmup) is a legal hint
  mechanism — warm the LRU with likely experts, still execute checkpoint
  bytes only.

### 2.13 DynaExq — runtime mixed-precision residency (APPROX-ONLY: PRECISION)
- Ref: arXiv:2511.15015 (2025).
- What: online budget-constrained precision allocation — hot experts
  resident at high precision, low-precision fallback for the rest;
  async promote/demote; forward always runs a "fully materialized" expert
  (i.e., at whatever precision is currently resident).
- PAPER-REPORTED: ≤2.73× throughput vs offload/prefetch baselines
  @batch32; +4.5pts acc vs static PTQ (Qwen3-MoE-30B/80B), single-GPU
  HBM envelope.
- Violation: PRECISION — the same expert id executes at different
  precision depending on residency; output varies with cache state.
- Exact-safe salvage: router-trace hotness estimation is legal for
  placement/hints (dee already uses route-driven residency).

### 2.14 DyMoE — dynamic per-expert quantization (APPROX-ONLY: PRECISION)
- Ref: arXiv:2603.19172 (2026).
- What: importance-aware runtime quantization of experts, depth-adaptive
  scheduling, look-ahead prefetch.
- PAPER-REPORTED: TTFT −3.44–22.7×, TPOT ≤14.58× vs offloading baselines
  on commercial edge hardware.
- Violation: PRECISION (runtime-chosen bit-width per expert execution).
- Exact-safe salvage: look-ahead prefetch = hint; importance maps =
  placement.

### 2.15 MoMP — mixture of precisions as QoS knob (APPROX-ONLY: PRECISION)
- Ref: Imani et al., arXiv:2407.14417, ICRC'24.
- What: partial expert quantization with dynamically chosen count/
  placement across CPU+GPU to trade throughput vs quality on a Pareto
  frontier.
- PAPER-REPORTED: 0.63→13.00 tok/s tunable on A100 Mixtral-8x7B;
  perplexity +~0.2 under max quantization (WikiText2/PTB/C4).
- Violation: PRECISION (quality is the knob — exact mode has no such
  knob).
- Exact-safe salvage: none beyond the general placement idea.

### 2.16 SwapMoE — virtual experts (APPROX-ONLY: IDENTITY, ROUTING)
- Ref: Kong et al., arXiv:2308.15030, ACL'24.
- What: keeps a small dynamic set of "Virtual Experts" in memory; masked
  gating redirects all requests onto them; expert weights seamlessly
  swapped into the virtual slots by importance scores.
- PAPER-REPORTED: Switch Transformer summarization 14.2→4.7 GiB, −50%
  latency, ROUGE-2 −0.041 (HF Transformers implementation).
- Violation: IDENTITY — tokens routed to expert E_i can execute on a
  virtual slot holding different content (masked gating remaps
  selection). ROUTING too (the executed set ≠ router's set).
- Exact-safe salvage: importance-based slot population is a legal
  placement policy ONLY if every virtual slot carries the true bytes of
  the expert currently mapped — i.e., it degenerates to dee's own
  bounded cache, losing the memory win.

### 2.17 SPICE — speculative prefetch + LoRE surrogates (APPROX-ONLY: PRECISION, IDENTITY)
- Ref: arXiv:2608.21240 (2026).
- What: lightweight draft model predicts expert sequence w/ confidence-
  aware lookahead; on low-confidence miss, approximates via resident
  shared expert + low-rank (LoRE) surrogates; exact residual work to CPU
  in parallel.
- PAPER-REPORTED: ≤3.12× TPOT, minimal quality loss (DSv2-Lite,
  Qwen2-57B-A14B, multiple GPUs).
- Violation: miss path executes a surrogate, not the routed expert
  (PRECISION+IDENTITY). Prefetch side is hint-legal.
- Exact-safe salvage: draft-model predictor + confidence gating is the
  same shape as a dee prefetch-hint plugin; "which misses deserve exact
  recovery" is always "all of them" in exact mode.

### 2.18 Read-ME — router-decoupled pre-gating refactor (APPROX-ONLY: ROUTING, IDENTITY — different model)
- Ref: Cai et al., arXiv:2410.19123, NeurIPS'24.
- What: MoE-fies a DENSE pretrained LLM into experts via activation
  sparsity; decoupled pre-gating router enables lookahead scheduling,
  expert-aware batching/caching.
- PAPER-REPORTED: ≤+10.1% MMLU vs similar dense models; −6.1% mean e2e
  latency; provably optimal caching given the pre-gate.
- Violation: not a modification of an existing MoE — a different model
  family. As a method pattern (decoupled router) it is the same ROUTING
  violation class as Pre-gated MoE.
- Exact-safe salvage: lookahead-decoupled scheduling is hint-legal.

### 2.19 Prediction-for-prefetch family — EXACT-SAFE AS HINT (not approx when router stays authoritative)
- SpecMoE/Eliseev & Mazur (arXiv:2312.17238): LRU + hidden-state expert
  guessing + prefetch overlap; Mixtral-8x7B on T4/3060/3080M at 2–3
  tok/s. Their shipped recipe uses mixed quantization (PRECISION choice,
  but as deployment config, not runtime substitution).
- ProMoE (arXiv:2410.22134): learned predictor + GOODPRED metric +
  sliding-window prefetch. PAPER-REPORTED: 2.20×/2.07× avg prefill/decode
  (≤3.21×/5.02×) vs reactive offload.
- SP-MoE (arXiv:2510.10302): SD-aware speculative expert prefetch,
  cutoff-layer bound; 1.07–3.5× TPOT.
- MoE-SpeQ (arXiv:2511.14102): draft-model expert-sequence prediction +
  Amortization Roofline governor; ≤2.34× (Phi-MoE).
- ExpertFlow (arXiv:2510.26730): adaptive-horizon prefetch + token-aware
  scheduling + cache-aware routing fallback to top-k; stall <0.1%.
  NOTE: its cached-prediction fast path that bypasses live top-k would be
  a ROUTING violation; the paper says it falls back to direct top-k —
  must verify which path computes the OUTPUT before adopting any piece.
- Verdict: EXACT-COMPATIBLE — prediction used for prefetch/placement
  only; native router authoritative. This is the legal Phase-5
  speculative-prefetch plugin shape. dee caution already on record:
  generic predictor recall@12 ≈0.503 hurt (wrong prefetch + pollution);
  trained per-layer predictors (Edge0) remain the credible direction.

### 2.20 Exact-adjacent systems (out of R8 scope; recorded for R12 matrix completeness)
MoE-Infinity (trace-aware prefetch/caching, no model change), Fiddler
(CPU-expert execution of unchanged weights), MoE-Lightning (CPU-GPU
pipeline + placement policy), EIO-MoE (expert-granularity I/O pipeline),
Mixtral-Offloading, PreMoE, CoMoE, BigMoeOnEdge (llama.cpp-class local
tool; its `--drop-cold-experts` flag IS an approx mode = ROUTING
violation; `--overlap`/`--predict-prefetch` are hint-legal). These move
bytes/placement only → not approximate; other tracks own them.
Related falsification-worthy datapoint: "Cacheable by Design?"
(arXiv:2608.18261) trained routers for locality — pre-registered NEGATIVE
result (miss −60% but every config failed the ≤1% perplexity gate at
137M scale): supports dee's stance that routing changes are never free.

## 3. Where dee-fast would branch from dee-exact

dee-fast = a separately-evidenced mode that may break the four clauses.
Branch seams, in contract order:

1. STORE/CODEC seam (`StorageCodec`, `StorageRecord.codec`,
   host_expert_tier.h:37-64): the cleanest approx entry point.
   dee-fast adds e.g. `Fp4LiteCodec`/`SlicedCodec` whose materialize()
   produces a DIFFERENT executor representation. The tier key already
   carries `representation` (TierExpertKey, host_expert_tier.h:15-23) —
   approx payloads MUST use distinct representation strings
   (`"dee4-fp4-int2-v1"`, never `"identity-v1"`/`dee4-fp4`), so exact
   tiers fail closed on them (IdentityCodec::accepts requires
   stored_bytes == exact_bytes). QMoE/HOBBIT/SliceMoE/DynaExq/MoMP
   techniques all enter here.
2. ROUTER seam (`Engine::route_topk`, engine.h:373-377): dee-fast
   installs an alternate selection function — pre-gated lookahead router,
   adaptive-k gating (AdapMoE/AdaMoE), cache-biased logits (Cache-Prior/
   ReMoE), substitution map (BuddyMoE/SMoE), or predictor-decided set
   (SiDA). Must be a separate code path; dee-exact's route_topk output is
   the exactness baseline for parity evidence.
3. EXECUTOR seam (ExactExpertExecutor / Engine FP4 calls): dee-fast may
   run `ApproxExpertExecutor` variants — int2/int4-lite kernels, merged
   experts, virtual-expert slots (SwapMoE), surrogate+residual (SPICE),
   dynamic skipping (NAEE/AdapMoE: execute k' < k).
4. TIER seams stay shared in shape but not in evidence: placement
   policies, hints, and residency are exact-legal already; dee-fast adds
   nothing new there except that approx payloads flow through the same
   bounded-slot machinery under different representation keys.

EVIDENCE RULE (hard): dee-fast runs write to a separate ledger with
`runtime_mode="dee-fast"` + an explicit `approximations[]` field
(codec name, router variant, skip policy, per-expert precision map).
dee-fast output can NEVER enter sealed exact evidence — no
router-parity.json, PARITY_MATRIX rows, REAL_GENERATION_LEDGER entries,
or "exactness preserved" claims may cite a dee-fast artifact. Hint: add
`runtime_mode` to evidence manifests so a fast-mode record is
mechanically rejectable from exact bundles. A dee-fast cell may report
speed and quality-delta, never parity.

## 4. Matrix rows for R12 (column set per briefing §10; flag for reconciliation)

| # | Method | Ref | Core mechanism | PAPER-REPORTED gain | Model/HW | Clause(s) | Verdict | Exact-safe salvage | dee-fast seam |
|---|--------|-----|----------------|--------------------|----------|-----------|---------|--------------------|---------------|
| 1 | HOBBIT | 2411.01433 | low-precision replica on cache miss + similarity prefetch + multidim cache | ≤9.93× decode vs MoE-Infinity; 13.0× vs llama.cpp; <1% acc | Mixtral-8x7B, Phi-MoE; Jetson AGX Orin, RTX4090 | PRECISION, BYTES | APPROX-ONLY | predictor→prefetch hint; cache policy→placement | codec (int2/int4-lite record) |
| 2 | EdgeMoE | 2308.14352 | per-expert offline bitwidth + statistical preload | 1.19–2.77× vs load baselines; −1.05–1.18× mem; ≤2% acc | Switch family; Jetson TX2, RPi4B | PRECISION, BYTES | APPROX-ONLY | preloader→hint | codec |
| 3 | AdapMoE | 2408.10284 | sensitivity adaptive-k + similarity prefetch + DP cache | −25% experts, 1.35×, ~0 acc loss | Mixtral-8x7B/8x22B; RTX4090/A6000 | ROUTING | APPROX-ONLY | prefetcher→hint; DP→placement | router (adaptive-k) |
| 4 | QMoE | 2310.16795 | sub-1-bit compression + custom format/kernels | 20× compress (0.8b/p); <5% vs ideal | Switch-c2048 1.6T; 4×A6000/8×3090 | PRECISION, BYTES | APPROX-ONLY | none (design lesson: codec seam) | codec + store bank |
| 5 | Cache-Prior/MCCE | 2412.00099 | boost logits of cached experts | >50% miss↓; ≤2× on-device; ppl Δ≤3% | Qwen1.5-MoE +3 more; Snapdragon phones | ROUTING | APPROX-ONLY | cache-awareness→placement only | router (logit bias) |
| 5b | BuddyMoE | 2511.10054 | co-activation buddy substitution on miss | ≤+10% tps, ~lossless | SOTA MoE; GPU offload | ROUTING, IDENTITY | APPROX-ONLY | buddy table→prefetch order | router+executor (substitute) |
| 5c | ReMoE | 2605.27081 | router fine-tune for temporal reuse | +26% reuse; TPOT −43.6–49.8% | DeepSeek/Qwen; Orin NX, vLLM | ROUTING | APPROX-ONLY | none (training) | router (retrained) |
| 5d | SMoE | 2508.18983 | expert-cache router: low-score→cached similar | −48% decode lat; >60% hit | Qwen/DeepSeek-class; edge GPU | ROUTING, IDENTITY | APPROX-ONLY | none | router (substitution) |
| 5e | CacheMoE (namesake) | IEEE IoT'25 | task-aware edge expert caching, DWFL+MADRL | improved hit/cost/latency | multi-task ViT MoE; edge nodes | none (placement+prediction) | EXACT-ADJACENT | demand predictors→hint/policy | n/a |
| 6 | Pre-gated MoE | 2308.12066 | gate_N selects experts_{N+1}; overlap prefetch | ~23% over oracle GPU-only; >> on-demand | Switch-Base/Large; single GPU+CPU | ROUTING | APPROX-ONLY | pre-gate predictor→hint (strongest salvage) | router (pre-gate) |
| 7 | MoE-I² | 2411.01016 | genetic-search expert pruning + low-rank | >50% params cut, zero-shot held | Qwen1.5-MoE, DSv2-Lite, Mixtral | IDENTITY, PRECISION, BYTES | APPROX-ONLY | none (offline surgery) | store (pruned manifest) |
| 8 | NAEE | 2402.14800 | post-training prune r/8 + w-ratio skip | 1.2–1.27×; 1 GPU vs 2; −2.9pts | Mixtral-8x7B; A100-80G | IDENTITY, ROUTING, (BYTES) | APPROX-ONLY | w-ratio stats→prefetch priority | router (skip) + store (prune) |
| 9 | MC-SMoE | 2310.01334 | routing-stats merge + low-rank/sparse | ≤80% mem, ≤20% FLOPs ↓ | SMoE backbones; 8 benchmarks | IDENTITY, PRECISION, BYTES | APPROX-ONLY | frequency→hint | store (merged records) |
| 10 | SiDA-MoE | 2310.18859 | LSTM hash predicts activated set; GPU holds predicted | ≤3.93× tps; −80% VRAM; ≤1% drop | Switch/NLLB; CPU+GPU server | ROUTING | APPROX-ONLY | hash predictor→hint (canonical) | router (predicted set) |
| 11 | AdaMoE | 2406.13233 | null experts + larger k, fine-tuned | −14.5% FLOPs, +1.69 ARC-C | Mixtral-8x7B fine-tune | ROUTING (IDENTITY) | APPROX-ONLY | none | router (adaptive-k) |
| 12 | SliceMoE | 2512.12990 | bit-sliced caching, low-bit prefix executes + warmup | 1.81×/1.64× lat; 2.37×/2.85× energy | DSv2-Lite, Qwen1.5-MoE; edge | PRECISION, BYTES | APPROX-ONLY | PCW→warmup hint | codec (sliced record) |
| 13 | DynaExq | 2511.15015 | runtime precision allocation, low-precision fallback executes | ≤2.73× tps @b32; +4.5 vs PTQ | Qwen3-MoE-30B/80B; 1 GPU | PRECISION | APPROX-ONLY | hotness→placement | codec (promote/demote) |
| 14 | DyMoE | 2603.19172 | runtime importance-driven quantization | TTFT 3.44–22.7×; TPOT ≤14.58× | edge HW | PRECISION | APPROX-ONLY | lookahead→hint | codec |
| 15 | MoMP | 2407.14417 | partial expert quant as QoS knob | 0.63→13.00 tok/s tunable | Mixtral-8x7B; A100 | PRECISION | APPROX-ONLY | none | codec |
| 16 | SwapMoE | 2308.15030 | virtual experts + masked gating redirect | 14.2→4.7 GiB; −50% lat; ROUGE-2 −0.041 | Switch; HF/CPU-GPU | IDENTITY, ROUTING | APPROX-ONLY | none as-designed | executor (virtual slots) |
| 17 | SPICE | 2608.21240 | draft predictor + LoRE surrogate on low-conf miss | ≤3.12× TPOT | DSv2-Lite, Qwen2-57B; multi-GPU | PRECISION, IDENTITY | APPROX-ONLY | draft predictor→hint | executor (surrogate) |
| 18 | Read-ME | 2410.19123 | dense→MoE refactor + decoupled pre-gate | +10.1% MMLU; −6.1% e2e lat | refactorized LMs | ROUTING, IDENTITY | APPROX-ONLY | lookahead scheduling→hint | router (pre-gate) |
| 19 | SpecMoE/ProMoE/SP-MoE/MoE-SpeQ/ExpertFlow | 2312.17238, 2410.22134, 2510.10302, 2511.14102, 2510.26730 | predict experts → prefetch only | 2.2×/2.07× avg (ProMoE); 1.07–3.5× TPOT (SP-MoE); ≤2.34× (SpeQ) | Mixtral/Phi-MoE/DSv2; consumer GPUs | none IF router stays authoritative (ExpertFlow's cached-prediction fast path = ROUTING if it bypasses top-k) | EXACT-SAFE-AS-HINT | the predictors ARE the salvage | n/a (exact mode) |

## 5. Falsification notes / open items for R12

- "CacheMoE" ambiguity resolved two ways above; if the briefing intended
  yet another paper, the cache-aware-routing rows (5a-5d) still cover the
  named mechanism class.
- dee already falsified the GENERIC version of this whole space: route
  prediction recall@12 ≈0.503 REJECTED (AGENTS.md); lossless codec scan
  REJECTED (packed FP4 entropy-dense). Every APPROX-ONLY row above gains
  by spending the thing dee refuses to spend (model fidelity).
- Cheapest exact-safe imports (Phase-5 candidates, prefetch-hint only):
  Pre-gated-style per-layer predictor; SiDA hash table; ProMoE
  GOODPRED-style accuracy+lead-time metric for hint quality;
  SliceMoE-style prefill warmup of the LRU.
