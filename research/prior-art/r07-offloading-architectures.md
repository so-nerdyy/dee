# R7 — Published offloading system architectures (hierarchy/cache/prefetch audit)

- Track: R7 — published offloading system architectures, surveyed for the
  consolidated prior-art matrix.
- Branch: `research/prior-art-r07` @ `dc78dc4` (worktree `.freebuff/wt/r07`).
- Scope: Mixtral-Offloading, MoE-Infinity, ExpertFlow (two namesake papers),
  SiDA-MoE, DAOP, ProMoE, fMoE/FineMoE, Read-ME. Per system: storage
  hierarchy shape; cache policy; prefetch source; H2D/SSD reduction
  mechanism; measurement context (baseline, hardware, model); reported
  gains; applicability to dee.
- Rules observed: no code changed; no remote spend. All external numbers
  are PAPER-REPORTED; dee-side numbers are MEASURED/SIMULATED per the
  cited sealed evidence. Paper speedups are metadata — the upside column
  scores each mechanism against dee's MEASURED wall components, not the
  paper's headline.
- Naming note: TWO unrelated systems are named "ExpertFlow":
  (a) arXiv:2410.17954 (He et al., NTU/HKUST-GZ) — predictive caching +
  token scheduling; (b) arXiv:2510.26730 (Shen et al.) — adaptive-horizon
  prefetch + cache-aware routing. Both covered; r08's ExpertFlow row is (b).
- Overlap note: r08 already classified SiDA and Read-ME as APPROX-ONLY
  (ROUTING) and ProMoE/ExpertFlow-(b) as prediction-for-prefetch. This
  track re-covers them at ARCHITECTURE level; verdicts agree.

## 1. The baseline every "expected upside" is scored against (dee MEASURED)

Sealed evidence (2×T4 SM75, Kaggle /tmp bank 0.29–0.37 GiB/s; 16-token
decode; sources `research/route-pipeline/LIVE_PROFILE_RESULTS.md`,
`STORAGE_VERDICT.md`, `research/phase2-concurrent-fill/
SERIALIZATION_VERDICT.md`, `research/phase2-legal-prefetch/
LEGAL_PREFETCH.md`):

| dee wall component | MEASURED value | Share of decode wall |
|---|---|---|
| Decode wall (16 tok) | 66.2–71.4 s (briefing: 71.2 s) | 100% |
| fill_wait — cold storage reads, critical | 41.99 s | 63.4% |
| stage_enqueue_wait — H2D submit + cache ops | 9.44 s | 14.2% |
| native_output_sync | 4.89 s | 7.4% |
| dense attention/orchestration/journal (unattributed) | 9.20 s | 13.9% |
| compute dispatch + combine + gather + readiness + D2H + shared | ~1.2 s | ~1.9% |
| Storage bytes / H2D bytes | 2.07 GB/tok / 3.71 GB/tok | — |
| Serial fill wall (response, prefill+decode) | 86.3 s | — |
| Prefill wall / prefill fill | 93.9 s / 62.2 s | — |

Prefetch legality bounds (LEGAL_PREFETCH.md, MEASURED+SIMULATED): bank is
single-stream saturated → ahead-of-router work only pays inside idle gaps,
~0.54–1.12 records/decode-row. Oracle-with-future-knowledge bound: ≤24.8 s
of the 42 s decode fill (≤29% of fill; ≤35% of wall). All *legal* sources
combined: ≤1.6 s (≤2.3% of decode wall). Host LRU is already within ~2.8pp
of offline MIN at the 16 GiB knee; 935/2364 sealed records ever repeat.

Consequence used throughout: any mechanism whose lever is "prefetch
earlier" or "cache smarter than LRU" is bounded to low-single-digit % of
dee's current decode wall on this bank. The dominant fill_wait component
can only shrink via (a) fewer cold bytes (residency/reuse — Phase-2 host
tier, already built), (b) faster bank (hardware, Phase-6), or (c) not
fetching at all (execute-in-place — R6/R9 CPU-sink track).

## 2. System-by-system

### 2.1 Mixtral-Offloading (SpecMoE) — Eliseev & Mazur, arXiv:2312.17238 (Dec 2023)

- Hierarchy: checkpoint quantized offline (HQQ mixed: attention 4-bit,
  experts 2–3 bit) → resident in host RAM → per-layer GPU expert cache.
  No SSD tier in the serving loop (model must fit RAM+VRAM).
- Cache policy: per-MoE-layer LRU over experts (k most-recently used kept
  on GPU; same k each layer).
- Prefetch source: speculative loading — apply layer L+1's gating function
  to the CURRENT layer's hidden state h_L (`predicted = gate_{L+1}(h_L)`);
  residual-stream smoothness makes h_{L+1} ≈ h_L; prefetch top-1–2 experts
  async during layer L's compute. Prediction input: current hidden state ×
  next layer's router weights; lookahead: 1 layer (also tested 2, 10 —
  recall decays). Training-free.
- H2D reduction: LRU hits between adjacent tokens + overlap of the one
  prefetch with current-layer compute. SSD reduction: none (no SSD tier).
- Measured against: HF Accelerate naive offloading. Mixtral-8x7B(-Instruct)
  on T4-16GB Colab (PCIe3), RTX3060-12GB, RTX3080M-16GB, A100-80GB;
  OpenAssistant conversations, batch 1.
- PAPER-REPORTED gains: full algorithm T4 2.092 vs naive 0.661 tok/s
  (~3.2×); A100 3.061 vs 1.392 (~2.2×); ablations: w/o preloading 1.567,
  w/o LRU+preload 1.168 on T4 — LRU alone ~1.34×, speculation adds ~1.33×
  more. Interactive 2–3 tok/s on consumer GPUs.
- Exactness: router untouched (predicted experts only prefetched; a miss
  falls back to demand load). The HQQ quantization is a separate
  deployment-config choice — the checkpoint itself is altered (PRECISION),
  but the caching/prefetch machinery is exact-safe for an already-packed
  checkpoint like dee's.
- dee mapping: dee already IS this architecture at two tiers (host LRU +
  bounded device residency), plus the SSD tier Mixtral-Offloading lacks.
  Its speculation = dee's deferred idle-gap hint channel. Its expert
  record: ~90 MiB at 3-bit (Mixtral ~176 MiB/expert FP16) vs dee 12.75
  MiB packed — dee's records are 7–14× smaller AND dee's bank is ~100×
  slower than PCIe.
- Code: github.com/dvmazur/mixtral-offloading (public; README lists
  speculative prefetching as "upcoming" — repo lags the tech report).
- Disposition: REJECT as new work (subsumed by built Phase-2 tiers);
  speculation arm = DEFER to the Phase-5 idle-gap engine, where its legal
  ceiling is already priced (≤1.6 s/response exact-legal).

### 2.2 MoE-Infinity — Xue, Fu, Lu, Mai, Marina, arXiv:2401.14361 (Jan 2024; v1 "serving", v3 "personal machines")

- Hierarchy: the only surveyed system with dee's full shape —
  GPU expert cache → host RAM → SSD (`--offload_dir` in OSS; paper v1
  stores model on NVMe, traces to disk). OSS multi-GPU: experts
  round-robin across GPUs, per-GPU caches, dedicated I/O threads.
- Cache policy: activation-aware replacement. Traces per-request expert
  activation as a sparse Expert Activation Matrix (EAM); online predictor
  produces pEAM (reuse/activation likelihood per expert) triggered after
  each layer's routing decision; eviction + prefetch priority follow the
  pEAM rather than raw recency.
- Prefetch source: the request's own activation trace — temporal locality
  of expert reuse across decode iterations within a sequence (batch-1
  personal-machine regime; v1: small-batch serving). Lookahead:
  next-iteration (temporal), not layer-index lookahead. Training-free
  (statistical tracing).
- H2D reduction: trace-guided hits + prefetch overlap during decode;
  worst-case analysis bounds cache size. SSD: OSS adds the SSD→RAM leg
  with dedicated I/O threads (dee's ColdExpertStore analog).
- Measured against: DeepSpeed-Inference/FastGen, llama.cpp/Ollama, vLLM,
  BrainStorm, Mixtral-Offloading. Single A5000-24GB over PCIe4 (~24 GB/s
  H2D); models DeepSeek-V2-Lite (31 GB), Switch-128x0.2B,
  NLLB-128x0.4B (220 GB), Mixtral-8x7B (120 GB), Snowflake Arctic (900 GB).
- PAPER-REPORTED gains: DeepSeek-V2-Lite TPOT 155 ms avg (169 tail) vs
  DeepSpeed 737/803, Mixtral-Offloading 1250/1530, Ollama 2590/2599, vLLM
  485/493 → 3.1–16.7× per-token; GPU idle 51 ms vs 207–2073 ms. v1
  serving: 4–20× latency, >8× deployment-cost reduction.
- Exactness: prefetch/eviction only; router authoritative; offloading is
  "lossless" by design claim. Exact-safe for existing checkpoints: YES.
- dee mapping: this is dee's Phase-2 architecture with a different cache
  policy and a serving engine on top. Two dee-relevant deltas: (1) the
  pEAM policy is a `DevicePlacementPolicy`-shaped candidate — worth a sim
  arm, but the upside is bounded by the measured LRU→MIN gap (~2.8pp ≈
  ~1–2 s/response); (2) the OSS repo ALREADY offloads DeepSeek-V4-Flash
  FP4-quantized experts (dee's canonical model) and GLM-5.2 — a runnable
  cross-validation/reference implementation at zero porting cost.
- Code: github.com/EfficientMoE/MoE-Infinity (ex-TorchMoE; active;
  OpenAI-compatible server, continuous batching, paged KV, CUDA graphs,
  FP4/MXFP4 expert paths).
- Disposition: SIMULATE (pEAM-style policy arm in the ws sim, sealed-journal
  gated); DEFER serving-engine features to Phase 5; flag repo as the
  cheapest independent exactness cross-check available for DSv4-Flash.

### 2.3 ExpertFlow (a) — He et al., arXiv:2410.17954 (Oct 2024; title v1: "Optimized Expert Activation and Token Allocation")

- Hierarchy: GPU expert cache → host RAM (SSD only as checkpoint store).
- Cache policy: Predictive Locality-aware Expert Caching (ECE) — loads
  only predictor-named experts, corrects mispredictions at runtime
  (demand-fetch on mispredict → router stays authoritative). Hit rate
  +15.35–35.67% over LRU (paper).
- Prefetch source: Routing Path Predictor (RPP) — T5-style encoder-decoder
  taking the raw input token sequence, emitting a (B,S,L,E) activation-
  probability matrix for ALL layers in one pass; trained as multi-label
  classification (BCE) on offline traces. Lookahead: full model depth.
- Token Scheduler: regroups tokens across consecutive batches (K-means on
  predicted routes) to minimize unique experts per batch — +9.94% XSUM /
  +16.19% WMT16 speed. A serving/batching lever.
- H2D reduction: early all-layer prefetch list + fewer unique experts per
  scheduled batch. SSD reduction: none.
- Measured against: Cache-MoE (LRU + CPU fallback, Mixtral-Offloading
  class), SE-MoE, Pregated-MoE. Switch-32/64/128, Mixtral-8×7B,
  Qwen1.5-MoE, Deepseek-MoE; Alpaca/WMT16/XSUM/AIME2024; single GPU.
- PAPER-REPORTED gains: 2.01×/3.19×/5.86× vs SE-MoE on Switch-32/64/128
  (CS=16, BS=32); up to 10× overall; −93.72% GPU memory.
- dee mapping: all-layer route hints are a prefetch-feedstock source for
  the deferred idle-gap engine (gap-bound ≤2% on sealed bank); the token
  scheduler is a dee-serve (Phase-5) idea — cross-request route-aware
  batching. Training cost is offline per-model.
- Code: no official repo located (a Zenodo DOI exists but is an
  auto-generated artifact, not a release).
- Disposition: DEFER (Phase-5 hint + batching machinery).

### 2.4 ExpertFlow (b) — Shen, Chu, Zhang, Xiang, Wu, Zhang, arXiv:2510.26730 (Oct 2025)

- Hierarchy: GPU two-level LRU expert cache → host RAM → SSD checkpoint.
- Cache policy: two-level LRU; cache-aware routing consults CACHED
  PREDICTIONS first — if a prediction for (token-seq, layer, step) exists
  it is reused; else falls back to live top-k router logits (§3.4).
- Prefetch source: hybrid cross-layer prediction — pregating outputs +
  intermediate states fused via a pre-trained RandomForest regressor that
  learns the deviation Δ between pre-gate scores and actual activations;
  predicted set = pre-gate + Δ. ADAPTIVE HORIZON: step size
  S = N_e·E_s/(C_s·T_l) — estimated experts × expert size over (link
  bandwidth × per-layer compute time) — recomputed from runtime stats;
  feedback loop: stall counter over threshold → S+1; overfetch counter →
  S−1. numSteps=2 optimal on A6000.
- H2D reduction: adaptive-distance prefetch tuned to measured bandwidth;
  stall <0.1% of baseline; ~99.9% waiting-latency reduction, +30%
  prediction accuracy (PAPER-REPORTED).
- Measured against: (unnamed in abstract) baseline offloading;
  DeepSeek-V2-Lite, Qwen1.5-MoE, Qwen2-MoE (4-bit); ShareGPT workload;
  GPU capped 20 GB; platforms A6000, Ascend 910B, + bandwidth table for
  H20/A100/RTX4090/Arc B580/RX6500XT.
- Exactness: the §3.4 cached-prediction path bypasses live top-k → on
  that path the EXECUTED set is the prediction (ROUTING violation);
  fallback mode is exact. Flagged pending source-level verification —
  r08 noted the same.
- dee mapping: the adaptive-S formula is directly portable — dee has all
  its constants (E_s = 12.75 MiB, C_s = 0.29–0.37 GiB/s bank / 5.4–5.7
  GB/s H2D, T_l ≈ 102.7 ms decode row). Plugging in: S ≈ 6×12.75 MiB /
  (0.33 GiB/s × 0.103 s) ≈ 2.2 layers — i.e., the formula itself says the
  sealed bank is too slow for lookahead to pay (records ≈ row time).
  Cheap and worth keeping as the Phase-5 prefetch scheduler's horizon rule.
- Code: no public repo found.
- Disposition: DEFER (horizon formula → Phase-5 hint engine);
  APPROX-ONLY flag on cache-aware-routing path.

### 2.5 SiDA-MoE — Du et al., arXiv:2310.18859, MLSys'24

- Hierarchy: GPU (only hash-predicted-activated experts resident) → host
  RAM (everything else). No SSD tier — repo TODO: "Add Disk Offload".
- Cache policy: none in the classic sense — per-batch RESIDENCY SET
  decided by the hash table, not an evict-on-miss cache.
- Prefetch source: offline-trained hash function — LSTM + sparse attention
  + truncated knowledge distillation, trained on (sample, activation-
  pattern) pairs; predicts per-token per-layer activated-expert set +
  scaling factor α for an entire incoming BATCH; hash-building thread runs
  one batch AHEAD of the inference thread (batch-level pipeline).
  Lookahead: whole forward pass, one batch ahead.
- H2D reduction: predicted-set preload hides transfers behind the previous
  batch's compute; up to 80% GPU memory saved.
- Measured against: Standard, DeepSpeed, Tutel. Switch-base-8/64/128/256
  (≤55 GB), NLLB-MoE; A100-80GB + 64-core Xeon Platinum 8358; GLUE/
  SuperGLUE (SST2/MRPC/MultiRC); batch 1 for isolation experiments.
- PAPER-REPORTED: ≤3.93× throughput, ≤72% latency cut, ≤80% GPU memory,
  ≤1% performance drop; hash top-3 prediction accuracy ≤99%.
- Exactness: the predicted set bounds what executes — a router-selected
  expert absent from the resident set is not served its true bytes on the
  critical path (hence the ≤1% accuracy drop). ROUTING-class divergence on
  mispredict (matches r08's APPROX-ONLY). As a pure hint it degenerates to
  ProMoE-class prefetch.
- dee mapping: hint-only salvage exists but adds nothing over the generic
  predictor family already REJECTED on the sealed miss stream (recall
  0.54%); its batch-ahead structure presumes a serving stream (Phase 5).
- Code: github.com/timlee0212/SiDA-MoE (public).
- Disposition: APPROX-ONLY as designed; hint salvage DEFER.

### 2.6 DAOP — Zhang, Aggarwal, Mitra (NUS), arXiv:2501.10375, DATE'25

- Hierarchy: GPU (per-sequence-allocated expert subset) ↔ host RAM, where
  CPU-resident experts are EXECUTED on CPU rather than transferred;
  SSD = checkpoint only.
- Cache policy: sequence-specific static allocation — the PREFILL
  activation pattern informs the decode-phase CPU/GPU expert split
  (prefill→decode expert-pattern cosine similarity ~90.7% measured on
  Mixtral-8x7B across C4/MATH/GSM8K, 512 samples each).
- Prefetch source: training-free 1-layer lookahead — apply layer L+1's
  gating function to the same hidden states the current layer's gate
  consumed (the Mixtral-Offloading trick); ~84% avg per-layer prediction
  accuracy one layer ahead (paper Fig. 5).
- H2D reduction: NOT reduced but BYPASSED — predicted CPU-resident experts
  are pre-calculated on CPU during the GPU's current-layer compute; only
  activations cross the link. BUT the pre-calc consumes a PROXY input (the
  not-yet-final next-layer hidden state) — "approximate methods" is the
  paper's own wording; graceful degradation bounds the accuracy loss.
- Measured against: expert caching/prefetching methods (≤8.20×) and
  offloading methods (1.35×; Fiddler-class CPU-execution baseline cited).
  Mixtral-8x7B (>90 GB, UNQUANTIZED) >4.5 tok/s, Phi-3.5-MoE (>80 GB)
  >8.2 tok/s, single A6000; C4/MATH/GSM8K/Alpaca.
- Exactness: pre-calculation is approximate twice over — proxy hidden
  input AND predicted expert set; graceful degradation is an accuracy
  management mechanism, not an exactness guarantee. APPROX-ONLY as
  published. The exact-safe kernel: per-sequence placement policy +
  activation-movement shape (at dee geometry ~32 KiB activation vs
  12.75 MiB record per call, ~408:1 — R9's inversion).
- dee mapping: THIS IS THE ONLY SURVEYED SYSTEM WHOSE MECHANISM TARGETS
  dee's dominant wall component — it converts fetches into in-RAM
  execution. dee's exact variant is already scoped: R6 gap list
  (HostExpertLease is CPU-readable today; needs stage_ex sink split,
  CpuExpertExecutor interface, region views, submit/join). At sealed-bank
  service (~35–96 ms/record) an exact CPU expert at ~5–10 ms (R9 derived
  estimate) beats the fetch at m=1 → the CPU-sink arm could in principle
  retire a large share of the 42 s fill_wait — the largest single upside
  in this survey, owned by R6/R9's rows, not importable from DAOP directly.
- Code: github.com/ecolab-nus/DAOP (public, 2025-02).
- Disposition: APPROX-ONLY as published; reinforces (does not replace)
  the R6 exact CPU-sink → MEASURE via that track.

### 2.7 ProMoE — Song, Zhong, Chen, Chen (SJTU/ZJU), arXiv:2410.22134 (Oct 2024)

- Hierarchy: GPU proactive expert cache → host RAM → SSD checkpoint.
- Cache policy: proactive cache driven by predictions; coordination
  machinery: chunked prefetch (granular transfers), early preemption
  (demand misses preempt speculative fills), reordered inference (cached
  experts first). Router untouched — demand-miss fallback preserves the
  executed set.
- Prefetch source: learned per-layer MLP predictor (~2M params/layer)
  trained OFFLINE on domain traces mapping layer inputs → gate outputs;
  ~84.7% avg accuracy vs 58.3% token-based / 66.9% skip-based heuristics;
  1–2 h training per model. STRIDE prefetch: predict layer i+1 while at
  layer i−1 (~2-layer lead) at ~5% accuracy cost. Quality metric:
  GoodPred = Accuracy × FetchRate (lead-time-aware).
- H2D reduction: removes fetch from the critical path via lookahead +
  interference-aware scheduling. SSD reduction: none.
- Measured against: framework offloading in HF transformers and llama.cpp
  plus hand-crafted caching baselines; RTX 4090-24GB; Mixtral-class MoEs.
- PAPER-REPORTED: 2.20× avg (≤3.21×) prefill, 2.07× avg (≤5.02×) decode vs
  offloading; 1.78×/1.34× vs hand-crafted caching.
- Exactness: EXACT-SAFE-AS-HINT — prediction never executes; mispredicts
  cost bandwidth only. Canonical dee-legal shape.
- dee mapping: the trained per-layer predictor is the Edge0-pattern dee
  already flagged as the credible direction — BUT the gate is miss-stream
  recall on the sealed journal, where prev-token recall is 0.54% and the
  idle-gap budget is ~1 record/row. A trained predictor must beat the miss
  stream's intrinsic novelty, then still fit the ≤1.6 s legal window.
  GoodPred = the right acceptance metric for a future dee hint engine.
- Code: no public release located.
- Disposition: SIMULATE first (train-on-trace recall on the sealed miss
  stream — CPU-only, no Kaggle GPU needed); then DEFER to Phase 5.

### 2.8 fMoE/FineMoE — Yu, Cui, Zhang, Wang, Wang (IntelliSys-Lab), arXiv:2502.05370, EuroSys'26

- Hierarchy: GPU expert cache → host RAM (480 GB testbed) → SSD
  (checkpoint + expert-map persistence). Built ON TOP of the MoE-Infinity
  codebase.
- Cache policy: fine-grained per-expert priority from a searched "expert
  map"; low-priority evicted/offloaded to CPU.
- Prefetch source: EXPERT MAP STORE — cross-request state. Expert map =
  iteration-level gate-probability distributions per request ("trajectory").
  Two similarity searches: (a) SEMANTIC — new prompt's embedding vs stored
  prompts → seed map before inference starts; (b) TRAJECTORY — partial
  observed trajectory {P_1..P_k} vs historical maps → predict layers
  k+1..N; cosine similarity, negligible overhead (§6.8). Prefetch distance
  (layers-ahead) configurable.
- H2D reduction: higher hit rate (+39% vs SOTA) from map-guided prefetch +
  eviction. SSD reduction: none (maps on disk are metadata, not weights).
- Measured against: No-offload, MoE-Infinity, ProMoE, Mixtral-Offloading,
  DeepSpeed-Inference. 6× RTX3090-24GB (pairwise NVLink; PCIe4 32 GB/s to
  CPU), Threadripper PRO 3955WX, 480 GB RAM; Mixtral-8×7B, Qwen1.5-MoE,
  Phi-3.5-MoE; LMSYS-Chat-1M, ShareGPT.
- PAPER-REPORTED: −47% inference latency, +39% expert hit rate vs SOTA.
- Exactness: maps guide prefetch/eviction only; router authoritative →
  EXACT-SAFE-AS-HINT.
- dee mapping: this is the published shape of dee's CONTRACT-DEFERRED
  regime-C prewarm (cross-request working-set reuse via a persistent
  store). Its semantic-hint seed is the one idea dee's regime-C contract
  lacks: prompt-similarity → policy_slots population. Bounded on the
  sealed trace by thin cross-request reuse (935/2364 repeat) — a
  multi-request journal is prerequisite evidence anyway (CONTRACT.md §5.5).
- Code: github.com/IntelliSys-Lab/FineMoE-EuroSys26 (demo on MoE-Infinity).
- Disposition: DEFER (Phase-5 / regime-C; semantic-map seeding is the
  salvageable novelty).

### 2.9 Read-ME — Cai et al., arXiv:2410.19123, NeurIPS'24

- Hierarchy: GPU expert cache → host RAM; SSD checkpoint. Serving-side
  expert-aware batching + caching.
- Cache policy: Belady-style eviction — PROVABLY OPTIMAL because the
  pre-gating router makes all future routes known before execution (an
  oracle-realizable MIN, matching dee's ws-sim methodology of scoring
  policies against Belady/MIN).
- Prefetch source: decoupled pre-gating router (~18M params) computes ALL
  layers' routes up front → lookahead = full model depth; enables
  lookahead scheduling + expert-aware batching (tokens grouped by
  assignment).
- H2D reduction: perfect prefetch ordering + optimal eviction; −6.1% mean
  e2e latency, −10% tail vs SOTA.
- Measured against: similar-scale dense OSS models + serving baselines;
  refactored models (dense→MoE via activation sparsity); 8× A100-80GB;
  +10.1% MMLU vs dense peers.
- Exactness: NOT a system for an existing checkpoint — it produces a NEW
  model family (refactored dense LLM + trained decoupled router). For
  dee's canonical checkpoint this is a different model: REJECT.
- dee mapping: the one portable idea — a route known ahead is a Belady
  solvable cache — dee already exploits in simulation (MIN upper bounds);
  making real routers pre-gated is a model-architecture change dee does
  not control. dee-serve could adopt expert-aware batching (Phase 5) on
  hints without the architectural change.
- Code: github.com/VITA-Group/READ-ME (public; HF Transformers fork).
- Disposition: REJECT (out of contract — different model); methodological
  salvage only (oracle-cache-given-known-routes validates dee's MIN sims).

## 3. Cross-cutting findings

1. **RAM is the literature's floor; SSD is dee's.** Every surveyed system
   except MoE-Infinity's OSS build assumes experts fit in host RAM — their
   bottleneck is PCIe H2D (~24–64 GB/s), dee's is a 0.29–0.37 GiB/s bank
   (~70–200× slower). Their headline gains (2–20×) measure H2D-link fixes
   dee mostly doesn't have on its critical path: dee's H2D waits are
   already 14% of wall and largely hidden behind fills. Transferring their
   prefetch machinery to a bank-bound regime changes the lever from
   "hide the transfer" to "there is no idle bandwidth to hide it in" —
   the sealed-bank prefetch ceiling is the ≤1.6 s legal / ≤24.8 s oracle
   bound, i.e. ≤2% / ≤35% of decode wall respectively.
2. **Two-layer "same-hidden-state" gating is a rediscovered trick** —
   Mixtral-Offloading (2023) and DAOP (2025) both use gate_{L+1}(h_L);
   ~84% one-layer accuracy (DAOP). dee's exact analog is the hash-layer
   early-staging channel (exact ids for L0–2, ≤1.6 s): the gate-reuse
   variant is an additional hint source for the deferred Phase-5 engine,
   not a new mechanism.
3. **The one wall-sized mechanism is execution-side, not transfer-side.**
   DAOP's shape (CPU executes cold experts; ~32 KiB activations cross the
   link instead of a 12.75 MiB record) is the only surveyed architecture
   that attacks dee's 63% fill_wait component. Its published form is
   approximate; the exact form is already in-tree as R6's CPU-sink gap
   list + R9's activation-movement economics.
4. **Cache-policy novelty is nearly exhausted against dee's baseline.**
   dee's plain host LRU sits ~2.8pp below offline MIN at the causal knee;
   pEAM/expert-map/two-level-LRU policies are fighting for ≤ that gap on
   the sealed trace (order ~1–2 s/response). Their real value is
   cross-request (regime C, Phase 5), where fMoE's semantic-seeded expert
   maps are the most developed published mechanism.
5. **Trained predictors are the unproven bet** — ProMoE/ExpertFlow(a)/
   SiDA all require offline per-model training; dee's generic predictor
   already failed (recall@12 0.503; miss-stream prev-token recall 0.54%).
   ProMoE's 84.7% is demand-set accuracy, not miss-stream accuracy — the
   sim gate remains "beat the miss stream's novelty", and even success is
   gap-bounded on this bank.
6. **Independent cross-validation available for free** — EfficientMoE's
   MoE-Infinity repo runs DeepSeek-V4-Flash FP4 experts on commodity
   GPUs with an SSD offload dir: a runnable reference for dee's exact
   architecture class on the SAME canonical model (also GLM-5.2).
   Useful for Phase-5 serving design and as an A/B sanity implementation;
   any remote run needs the normal spend gate.

## 4. Matrix rows (ready-to-merge; column set per R7 briefing)

| paper/system | year | exact-safe for existing checkpoint? | requires training? | changes router? | prediction input | lookahead distance | GPU cache policy | RAM/SSD use | CPU compute? | H2D reduction mechanism | SSD reduction mechanism | cross-request assumptions | model/hardware evaluated | reported gain | baseline | code availability | applicability to dee | dee phase | expected upside as fraction of dee's MEASURED wall components | implementation complexity | evidence confidence | disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mixtral-Offloading (SpecMoE) | 2023 | YES (machinery); HQQ quant is separate config choice | no | no | gate_{L+1} applied to current-layer hidden state h_L (training-free) | 1 layer (tested 1/2/10) | per-layer LRU (k experts) | experts all in host RAM; no SSD tier | no | LRU hits + 1-layer speculative prefetch overlap | none | within-request temporal locality (adjacent tokens) | Mixtral-8x7B(-Instruct) HQQ 2–3bit; T4-16GB, RTX3060-12GB, RTX3080M-16GB, A100-80GB | ~3.2× vs naive offload on T4 (2.09 vs 0.66 tok/s); ~2.2× A100; 2–3 tok/s consumer HW | HF Accelerate naive offloading | public: dvmazur/mixtral-offloading (spec-prefetch listed upcoming) | architecture already subsumed by Phase-2 tiers; speculation = deferred hint channel | P2 done / P5 hints | ≤2% of decode wall (legal hint bound ≤1.6 s of 71.2 s; oracle ≤35%) | LOW (already built) | HIGH (peer-cited tech report + repo) | REJECT as new build (subsumed); DEFER hint arm to P5 |
| MoE-Infinity | 2024 (v1/v3 rev.) | YES (lossless offload claim; router authoritative) | no (online tracing) | no | per-request EAM activation trace → pEAM reuse/activation likelihood | next-iteration temporal + within-trace | activation-aware (pEAM-guided) eviction + prefetch priority | RAM primary + SSD offload dir (OSS); traces on disk | no | trace-guided prefetch overlap + higher hit rate | SSD→RAM leg exists in OSS (dedicated I/O threads) | within-request (batch-1/batch<32 locality); serving version keeps per-request traces | DeepSeek-V2-Lite, Switch-128, NLLB-MoE-128, Mixtral-8x7B, Arctic; single A5000-24GB PCIe4 | 3.1–16.7× TPOT (155 vs 485–2590 ms); v1: 4–20× latency, >8× cost | DeepSpeed-FastGen, vLLM, Ollama, BrainStorm, Mixtral-Offloading | public, active: EfficientMoE/MoE-Infinity — incl. DSv4-Flash FP4 + GLM-5.2 offload | closest published analog of dee's architecture; pEAM = candidate DevicePlacementPolicy; repo = zero-port cross-check | P2 (policy sim) / P5 (serving engine) | ≤2–3% decode wall on sealed bank (LRU→MIN gap ~2.8pp ≈ 1–2 s); larger under regime-C/dee-serve | MEDIUM (policy arm in existing ws sim) | HIGH (paper + active repo + r08 corroboration) | SIMULATE policy; DEFER serving features; flag repo as cross-validation target |
| ExpertFlow (a) — predictive caching + token scheduling | 2024 | YES (runtime mispredict correction = demand fetch; router authoritative) | YES (offline T5-style RPP, BCE) | no | raw input token seq → all-layer (B,S,L,E) activation matrix in one pass | full model depth | Predictive Locality-aware Expert Caching (ECE); +15–36% hit vs LRU | host RAM; SSD checkpoint only | no | early all-layer prefetch + token re-batching cuts unique experts/batch | none | token scheduler regroups ACROSS batches (serving stream) | Switch-32/64/128, Mixtral-8x7B, Qwen1.5-MoE, DeepSeek-MoE; Alpaca/WMT16/XSUM/AIME2024; single GPU | 2.01–5.86× vs SE-MoE (Switch); ≤10× overall; −93.7% GPU mem | Cache-MoE (LRU+CPU fallback), SE-MoE, Pregated-MoE | no official repo located | all-layer hint feedstock for idle-gap engine; route-aware re-batching is dee-serve machinery | P5 | ≤2% decode wall sealed (gap-bound); scheduler upside unmeasurable at batch-1 | MEDIUM-HIGH (needs trained RPP + serving stream) | MEDIUM (arXiv-only, no code) | DEFER (Phase-5) |
| ExpertFlow (b) — adaptive scheduling + memory coordination | 2025 | PARTIAL — §3.4 cached-prediction path bypasses live top-k (flagged, verify); fallback is exact | YES (pre-trained RandomForest Δ-corrector) | YES on cached-prediction path; no on top-k fallback | pregating outputs + intermediate states + RF-learned Δ deviation | ADAPTIVE S = N_e·E_s/(C_s·T_l), feedback-tuned (≈2 on A6000) | two-level LRU | host RAM staging; SSD checkpoint | no | hardware-aware adaptive-horizon prefetch; stall <0.1% baseline | none | cached predictions keyed (seq, layer, step) — cross-iteration/request reuse | DeepSeek-V2-Lite, Qwen1.5/2.0-MoE (4-bit); ShareGPT; 20GB-cap GPUs incl. A6000, 910B | ~99.9% waiting-latency reduction; +30% prediction accuracy | baseline offloading (paper §4) | no public repo found | adaptive-S formula portable to dee's hint scheduler (constants in hand → S≈2 layers on sealed bank, i.e. formula itself says prefetch can't pay here) | P5 (horizon rule) | ~0 incremental on sealed bank; formula is free validation of the defer verdict | LOW-MEDIUM (formula only) | MEDIUM (arXiv; router-bypass semantics need source check) | DEFER (formula to P5 engine); APPROX-ONLY flag on routing path |
| SiDA-MoE | 2024 (MLSys'24) | NO — predicted resident set bounds the executed set (≤1% acc drop proves divergence) | YES (offline LSTM + sparse-attn + truncated KD) | YES (on mispredict; hash constrains execution) | raw input batch → per-token per-layer activated set + scaling α | whole forward, one batch ahead | none — per-batch residency set, not evict-on-miss | host RAM only (disk offload = repo TODO) | YES — parallel hash-building thread (CPU) | batch-ahead preload of predicted set | none | streaming-batch assumption (batch j hashed while i runs) | Switch-base-8/64/128/256, NLLB-MoE; A100-80GB + Xeon 8358; GLUE/SuperGLUE; batch 1 | ≤3.93× tps; ≤72% latency; ≤80% GPU mem; ≤1% acc drop; ≤99% hash acc | Standard, DeepSpeed, Tutel | public: timlee0212/SiDA-MoE | hint-only salvage = degenerate prefetch; batch-ahead shape presumes serving stream | P5 at earliest | ≤2% decode wall as hint (gap-bound); 0 as designed (APPROX) | HIGH (training + hash plumbing) | HIGH (peer-reviewed + repo) | APPROX-ONLY as designed; hint salvage DEFER |
| DAOP | 2025 (DATE) | NO — CPU pre-calc uses proxy hidden state + predicted set ("approximate methods", graceful degradation) | no | partial (predicted set drives CPU pre-calc path) | gate_{L+1} on current-layer's gate input (training-free); prefill pattern → decode placement | 1 layer | per-sequence static CPU/GPU split (prefill-informed) + cache-ratio sweep | host RAM as EXECUTION tier; SSD checkpoint | YES — CPU executes predicted experts (approx) | BYPASS: activations cross link, not weights (~KB vs MB) | none | per-sequence (prefill→decode pattern transfer, ~90.7% cosine) | Mixtral-8x7B >90GB unquantized >4.5 tok/s; Phi-3.5-MoE >8.2 tok/s; single A6000; C4/MATH/GSM8K/Alpaca | ≤8.20× vs caching/prefetch; 1.35× vs offloading | caching/prefetch + Fiddler-class CPU offload | public: ecolab-nus/DAOP | shape = the only mechanism attacking dee's 63% fill component; exact variant already scoped as R6 CPU-sink | P5/P6 via R6-R9 (exact sink) | upper bound = fill_wait share (~59–63% of decode wall) IF exact CPU exec beats 35–96 ms fetch at m=1 — owned by R6/R9 rows | HIGH (needs R6 gaps G1–G4) | HIGH (peer-reviewed + repo) | APPROX-ONLY as published; MEASURE the exact analog under R6/R9 |
| ProMoE | 2024 | YES (hint-only; demand-miss fallback keeps router authoritative) | YES (offline per-layer MLP ~2M params, 1–2 h/model on domain traces) | no | layer input vector → predicted gate outputs (per-layer MLPs) | stride prefetch: predict i+1 during i−1 (~2 layers) | proactive cache: chunked prefetch + early preemption + cached-first reorder | host RAM; SSD checkpoint | no | lookahead prefetch + interference-aware scheduling removes fetch from critical path | none | within-request; predictor trained on cross-request domain traces | Mixtral-class MoEs; RTX4090-24GB; HF transformers + llama.cpp integrations | 2.20×/2.07× avg prefill/decode (≤3.21×/5.02×) vs offload; 1.78×/1.34× vs hand-crafted cache; 84.7% pred acc | framework offloading; hand-crafted caching | canonical exact-safe trained-predictor shape (Edge0 direction); GoodPred = right acceptance metric | P5 (after miss-stream recall gate) | ≤2% decode wall sealed-bank legal bound; ≤35% oracle; gated on miss-stream recall (prev-token 0.54%) | MEDIUM (train pipeline + hint plumbing + gap scheduler) | MEDIUM-HIGH (no code; consistent multi-source detail) | SIMULATE (sealed miss-stream recall, CPU-only) then DEFER P5 |
| fMoE / FineMoE | 2026 (EuroSys; arXiv Feb 2025) | YES (maps steer prefetch/eviction only) | no (similarity search over accumulated maps; needs history) | no | prompt semantic embedding + partial gate-probability trajectory vs historical expert maps | configurable prefetch distance (layers ahead) | map-guided fine-grained per-expert priority | host RAM (480GB testbed); SSD for checkpoint + map store | no | +39% hit rate → fewer H2D transfers | none (maps = metadata) | EXPLICIT cross-request: Expert Map Store persists across requests; semantic seed per new prompt | Mixtral-8x7B, Qwen1.5-MoE, Phi-3.5-MoE; 6×RTX3090-24GB NVLink + 480GB host; LMSYS-1M/ShareGPT | −47% latency; +39% hit vs SOTA | No-offload, MoE-Infinity, ProMoE, Mixtral-Offloading, DeepSpeed | public demo: IntelliSys-Lab/FineMoE-EuroSys26 (built on MoE-Infinity) | published shape of dee's deferred regime-C prewarm; semantic-seeded policy_slots is the novel salvage | P5 (regime-C contract) | ~0 on sealed single-request trace (contract-deferred; 935/2364 repeat bounds reuse); Phase-5 upside | MEDIUM (map store + embedding index + policy) | HIGH (peer-reviewed + demo repo) | DEFER (Phase-5/regime-C) |
| Read-ME | 2024 (NeurIPS) | NO — produces a different model (dense→MoE refactor + trained pre-gate router) | YES (refactorization + router training) | YES (decoupled pre-gate IS the router) | pre-gating router computes all routes before backbone runs | full depth (all routes pre-computed) | Belady-optimal eviction given known routes (realizable MIN) | host RAM experts; SSD checkpoint | no | perfect prefetch ordering + optimal cache | none | expert-aware batching across queued requests | refactored dense models (Llama-class); 8×A100-80GB tuning | −6.1% mean / −10% tail e2e latency; +10.1% MMLU vs dense peers | SOTA serving systems on similar-scale models | public: VITA-Group/READ-ME | none for canonical checkpoint; Belady-given-known-routes validates dee's MIN sim methodology | N/A (P5 batching idea only) | 0 — out of contract for dee-exact | N/A (model change) | HIGH (peer-reviewed + repo) | REJECT (different model); methodological salvage only |

## 5. Falsification notes / open items for R12

- The survey's collective blind spot is the cold tier: all H2D-side gains
  are measured on banks ≥100× faster than dee's sealed /tmp. Re-simulation
  against a fast bank (>0.6–0.7 GiB/s) reopens every prefetch row
  simultaneously (LEGAL_PREFETCH sensitivity: gap capacity ∝ BW) — the
  defer verdicts are bank-regime-conditioned, not absolute.
- ExpertFlow(b)'s router-bypass claim rests on §3.4 text ("reuses cached
  prediction… falls back to direct top-k"); if the consolidated matrix
  needs certainty, a source-level check of which path emits executed
  logits is the one unresolved verification — flagged MEDIUM confidence.
- ProMoE's 84.7% accuracy is on the DEMAND set; nothing in any surveyed
  paper measures miss-stream recall (the metric that killed dee's generic
  predictor). Any SIMULATE arm must score against dee's miss stream, not
  demand.
- DAOP's upside column deliberately points at R6/R9's exact CPU-sink, not
  DAOP's approx precalc; double-counting the fill_wait prize across R7 and
  R6/R9 rows should be reconciled at merge time.
- SiDA/fMoE cross-request claims presume multi-request workloads dee
  hasn't measured yet (single sealed journal); regime-C's standing caveat
  applies — skew stability is a workload assumption, not a measured
  property.
