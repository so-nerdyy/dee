# R11 — Bibliography gap search: what the named set does NOT cover

- Track: R11 — gap analysis of `MoE-Inf/awesome-moe-inference` (maintained
  bibliography behind arXiv 2412.14219, ACM TALLIP doi 10.1145/3794845)
  plus 2025–26 arXiv, against the assigned named set.
- Branch: `research/prior-art-r11` @ `dc78dc4` (worktree `.freebuff/wt/r11`).
- Rules observed: no remote spend; every figure below is PAPER/REPO-REPORTED
  (external claims, not dee evidence) and carries its tier/context label;
  falsification entries are first-class. Nothing here is acceptance
  evidence.
- Named set (already assigned to other tracks — intentionally NOT re-rowed;
  duplicates/aliases against it flagged in §4): Cross-Layer Gate/Fate,
  ProMoE, DAOP, fMoE, SiDA, MoE-Infinity, Mixtral-Offloading, ExpertFlow,
  EdgeMoE, Pre-gated MoE, Read-ME, Fiddler, KTransformers, FreeToken,
  HybriMoE, MoE-Lightning, HOBBIT, AdapMoE, QMoE, CacheMoE, MoNDE,
  llama.cpp CPU-MoE.

## TL;DR

The named set covers ~all of the awesome list's Expert-Offloading section
but leaves three real gaps: (1) **2025–26 router-preserving prefetch systems
that are exact-safe by construction** (SpecPrefetch, APEX exact-mode,
ST-MoE, PreScope, DuoServe-MoE — the named set skews toward
router-replacing predictors); (2) **storage-hierarchy work below host RAM**
— SSD/flash/page-cache residency (FlashMoE, FluxMoE, the kernel-managed
tiering study 2608.12103, HBF dual-route, the SSD-energy counterpaper);
(3) **the DeepSeek serving stack itself** (EPLB/DeepEP/DeepGEMM + the V3/R1
inference overview, metro, Orders-in-Chaos trace study) — directly
load-bearing for dee's canonical model. One more gap axis: **practitioner
systems** (moe-l2, kimi-k3-in-c, Edge0, sabrewing, memra) that already
implement dee-shaped SSD→RAM→VRAM pipelines and have independently measured
the same cold-first-touch wall.

dee question tags used below:
- **D-RES** — residency/eviction policy (host LRU knee ~16 GiB, ~2.8pp
  below MIN, VRAM `last_used` repair, regime-C prewarm contract)
- **D-HINT** — legal ahead-of-router candidate sources
  (`research/phase2-legal-prefetch`: oracle caps ~29% of fill wall; hash
  layers 0–2 only exact early IDs; generic predictor recall@12≈0.503
  REJECTED — trained per-layer predictors remain open)
- **D-SINK** — second execution sink on device miss (R6 gap list G1–G10)
- **D-FILL** — serial cold fill, single-stream bank ceiling
  0.29–0.37 GiB/s, conditional-GO >~0.6–0.7 GiB/s
- **D-SERVE** — dee-serve direction (global caches, batching, scheduling)
- **D-DSV** — DeepSeek V3/V4-family routing specifics
- **D-ECON** — $/J per byte per token (roofline economics)
- **D-EXACT** — where the exactness boundary sits

## 1. Coverage map: what the bibliography actually contains

awesome-moe-inference sections and their disposition vs the R-swarm:

- **System-Level / Expert Offloading** (survey's list): HOBBIT, ExpertFlow
  (see §4 collision), SiDA, MoE-Infinity, Pre-gated MoE,
  Mixtral-Offloading, EdgeMoE, DyNN-Offload, Read-ME, ProMoE, AdapMoE,
  MoE-Fiddler (=Fiddler), EIO-MoE, MoE-Deploy, Swap-MoE (=SwapMoE),
  MoE-Lightning, CacheMoE. → Named set + R8 cover ~all; leftovers rowed
  below (EIO-MoE unresolved-ID; DyNN-Offload, MoE-Deploy as ancestors).
- **System-Level / Expert Parallel** (≈35 entries: GShard, FastMoE, Tutel,
  MoESys, Alpa, BaGuaLu, SmartMoE, Switch-Transformers, HashLayer, Prophet,
  MoE-Prediction, Lazarus, FlexMoE, MoE-Deploy, Brainstorm, Lynx,
  Base-Layers, MoE-ECR, Janus(SoCC'23), Hetu-MoE, DeepSpeed-MoE,
  DeepSpeed-TED, Lina, ExFlow, TA-MoE, Aurora, LocMoE, Parm, ScMoE, HiDup,
  ScheMoE, PipeMoE, EPS-MoE, MoE-SLC, MPMoE). → Mostly
  distributed/training-side comm and placement; individually weak for dee's
  single-node hierarchy. The handful worth a row are below; the rest are
  condensed in Appendix A for matrix completeness.
- **Model-Level** (architecture/compression/routing/merging): R8's domain;
  skipped here.
- **Hardware-Level** (MoNDE, FLAME, Duplex, M³ViT, Edge-MoE, Space-Mate,
  …): R9's domain; skipped except one storage-delivery row (B3) that sits
  between the two tracks.
- **Gap in the bibliography itself**: KTransformers, FreeToken, HybriMoE,
  llama.cpp CPU-MoE, MoE-SpeQ/SP-MoE-class SD work, and essentially all
  2026 systems (FlashMoE/FluxMoE/PreScope/SpecPrefetch/APEX/Janus-inf/
  MoEless/metro/ESS/EaaS) are absent or only partially tracked — the list
  lags ~6–12 months, so arXiv sweep was the load-bearing source for P1–P3.

## 2. Candidate rows — Priority 1: exact-safe prefetch / residency

Prediction drives transfer/placement ONLY; the frozen router decides the
executed set. This is the class dee's contract admits without amendment
(hints legal, prediction errors waste bandwidth never correctness).

| # | Work | Ref | One-line mechanism | Exact-safe? | Claimed gain + HW | Code | dee Q |
|---|------|-----|--------------------|-------------|-------------------|------|-------|
| A1 | **SpecPrefetch** | arXiv 2607.24787 | lightweight layer adapters predict next-layer expert priorities for async transfer; frozen native router executes; window-aware prefetch budget | **Y — router-preserving by construction** ("prediction errors affect transfer efficiency rather than model outputs") | best expert recall 9/10 settings; +20% decode on Snapdragon 8 Elite; Qwen3-VL-30B-A3B, DeepSeek-VL2-Tiny | promised on acceptance (v1 claims URL) | D-HINT, D-DSV (DeepSeek-family VLM) |
| A2 | **APEX** | arXiv 2608.11688 | prefetch router before attention block + learned confidence model sizes the fetched set per token | **Y in its "correctness-preserving mode"** (explicit dual-mode; "stall-free" mode executes available-experts-only = approx) | ≤26% latency, ≤41% EDP reduction (exact mode); >99% overlap accuracy; edge MoEs | not seen | D-HINT, D-RES |
| A3 | **ST-MoE** | arXiv 2606.15453 | spatio-temporal (adjacent-layer + consecutive-token) predictor stages experts ahead; paired reconfigurable-HW design | **Y** — paper states the predictor "preserves the original routing behavior" | 85% prediction accuracy; 2.5×/2.2×/1.5× vs GPU / Adap-Gating / Pre-gated; energy 2.5×/1.8×/2.0× | not seen | D-HINT, D-RES |
| A4 | **PreScope** | arXiv 2509.23638 | LLaPor layer-group-aware predictor + PreSched global cross-layer prefetch/on-demand scheduler + AsyncIO overlap; hybrid CPU-GPU co-exec | **Y-probably** — predictor drives prefetch/scheduling; verify the CPU-exec path consumes router output not prediction | +141% e2e throughput vs SOTA; 10B-class MoE on single commodity GPU | not seen | D-HINT, D-SINK, D-RES |
| A5 | **"Who Should Own the Expert Cache?" (kernel-managed tiering)** | arXiv 2608.12103 | controlled study: OS page cache + MGLRU + readahead as the expert tier vs user-space oracle pinning; router-trace replay on real bank | **Y — measurement study, token-identical outputs claimed** | at equal enforced memory kernel recency ≈ oracle static-frequency; pread replay oracle-pinned arena only 1.09–1.11× faster; lookahead at 64.7% recall moves median 0.3%, perfect 1-layer advice +5.0% via readahead; balloon+MGLRU can **overstate low-capacity device traffic ~2×** (methodology warning); page-cache admission +9–10% steady decode | GH200, 1.45 TB expert pool, 3 models' traces (128–896 experts/layer) | **D-RES (directly), D-FILL, D-HINT** — most dee-shaped external study found |
| A6 | **MoE-SpAc** | arXiv 2603.09983 | repurposes speculative decoding as a lookahead *sensor* for expert demand (Speculative Utility Estimator) + heterogeneous workload balancer + unified prefetch/evict | **Y-probably** — SD lookahead informs residency; verify draft never bypasses target verify | +42% TPS vs SOTA SD baseline; 4.04× avg; 7 benchmarks | not seen | D-HINT, D-RES, D-SINK |
| A7 | **SpecMoEOff** | arXiv 2508.21706 | SD enlarges per-expert workload to hide offload latency; CPU chunked-attention verify kernel; roofline auto-tuner | **Y** — standard SD verify keeps outputs exact; offloading unchanged | ≤2.5× decode throughput vs SOTA MoE offloading | not seen | D-SERVE, D-HINT (V4-Flash has in-checkpoint DSpark — direct relevance) |
| A8 | **Speculating Experts (YALIS)** | arXiv 2603.19289 | internal representations predict near-future experts; prefetched; **executes speculated experts to avoid re-fetch** | **N as shipped — ROUTING** (predicted set can replace router's set); salvageable as hint-only | −14% TPOT vs on-demand CPU load; multi-MoE | github.com/axonn-ai/yalis `offload_prefetch` | D-HINT (predictor is the salvage), D-EXACT |
| A9 | **DuoServe-MoE** | arXiv 2509.07379 | split-phase: 2-stream CUDA prefetch during prefill; offline-trained layer predictor prefetches in decode; no model change | **Y** — prefetch-only predictor, router untouched | 1.42–7.54× e2e latency; peak mem 15% of model; 4-bit Mixtral-8x7B/8x22B single GPU | not seen | D-HINT, D-RES |
| A10 | **Predictive prefetch + expert replication** | arXiv 2605.11537 | predicts soon-overloaded experts, deep-replicates them for batch parallelism | **partial/N** — "90–95% of baseline accuracy" implies executed-set drift; verify | ~100% GPU util; ≤3× inference speed; Switch-base-128/256 | not seen | D-SERVE |
| A11 | **Cache-Aware Joint Router Adaptation** (Temporal/Spatio cache routers) | arXiv 2609.04895 | post-training adds auxiliary cache routers steering residency; **native top-k preserved at inference** | **mixed** — inference-time mechanism exact-safe, but the adapted backbone is a new artifact (BYTES for existing checkpoints) | hit rate +1.15–18.03pp, expert traffic −4.6–53.3%; Qwen3, GPT-OSS | not seen | D-RES, D-EXACT (train-for-cacheability family — cf. R8's "Cacheable by Design?" negative) |

## 3. Candidate rows — Priority 2: storage-hierarchy MoE serving

dee's cold tier (NVMe/page-cache) is the measured wall; the named set
mostly stops at host RAM. These go below RAM or rethink the tier owner.

| # | Work | Ref | One-line mechanism | Exact-safe? | Claimed gain + HW | Code | dee Q |
|---|------|-----|--------------------|-------------|-------------------|------|-------|
| B1 | **FlashMoE** | arXiv 2601.17063 | experts offloaded to **SSD** (not RAM); ML cache-replacement mixing recency+frequency | **Y** — placement/eviction only | +51% hit rate vs LRU/LFU; ≤2.6× vs existing systems; real user-grade desktop | not seen | **D-RES, D-FILL** (only SSD-native baseline found) |
| B2 | **FluxMoE** | arXiv 2604.02715 | PagedTensor virtualizes expert tensors (stable vaddrs, dynamic physical bind) + bandwidth-proportional storage hierarchy (compressed GPU mem + DRAM) + closed-loop budget-aware residency planner vs KV pressure | **Y** — materialization/eviction only, kernels unmodified | dynamic expert residency under KV pressure; framework-compatible (PyTorch/Triton) | not seen | **D-RES** (a real "paged expert" design to compare against dee's slot model), D-SERVE |
| B3 | **HBF dual-route expert delivery** | arXiv 2608.14333 | die-stacked NAND (HBF) feeds GPU via TWO concurrent paths — direct GPU↔HBF + HBF→HBM→GPU; early expert determination overlaps read latency; immutable weights vs mutable KV separated | **Y** (architecture sim, event-driven) | 1.94× throughput / 1.90× e2e vs HBM-relay-only route | simulator | D-FILL, D-ECON (R9-adjacent hardware row; storage-delivery focus) |
| B4 | **"SSD Offloading for MoE Weights Considered Harmful" (energy)** | arXiv 2508.06978 | quantitative energy analysis of SSD vs DDR vs HBM expert reads during decode | **Y (analysis)** | NEGATIVE: SSD expert reads raise decode energy up to **~12×** vs HBM baseline (DeepSeek-R1, experts=96.1% of params); prefetch hides latency not energy; SSD viable only if flash read energy improves ~10× | n/a | **D-ECON, D-FILL — falsification-grade for the dee-local SSD-heavy product class** |
| B5 | **ESS (Extended Sparse Server)** | arXiv 2512.10576 | offload-centric latent-cache mgmt for **DeepSeek-V3.2-Exp** sparse attention: Top-2K latent entries' temporal locality → GPU sparse pool, rest to CPU | **Y** — KV/latent-cache tiering (attention side, not expert weights) | decouples decode batch from VRAM cap; DSv3.2 PD-disagg decode | not seen | D-DSV (DS-specific storage tiering; adjacent — KV not experts) |
| B6 | **Klotski** | arXiv 2502.06888, ASPLOS'25 | expert-aware multi-batch pipeline: hot experts' compute overlaps cold experts' I/O; constraint-sensitive I/O-compute planner + correlation-aware prefetcher | **Y** — scheduling/prefetch only | ≤85.12× throughput vs SOTA (batch/offload regime) | not seen | D-SERVE, D-FILL |

## 4. Candidate rows — Priority 3: CPU-hybrid execution

| # | Work | Ref | One-line mechanism | Exact-safe? | Claimed gain + HW | Code | dee Q |
|---|------|-----|--------------------|-------------|-------------------|------|-------|
| C1 | **OSDI'26 local CPU–GPU hybrid (SLP/DSLP system)** | arXiv 2606.10493 | stream-loading prefill + SmallEP multi-GPU prefill + intra-node P/D disagg w/ zero-copy weights + AVX-512 FP8 GEMV + fine-grained CPU parallelism; **explicitly intact-precision** | **Y** — anti-quantization stance matches dee's contract | 21.5 tok/s intact FP8 DeepSeek-V3 on CPU-parallel path; 28 tok/s INT4; 1200→1800 tok/s prefill on ≤2× RTX 5090; 32K–45K prompts <30 s TTFT | not seen (Tsinghua CRAFT + Xingyun) | **D-SINK, D-DSV** — the closest peer system to dee's exact-CPU-sink goal |
| C2 | **moe-l2** | PyPI `moe-l2` (v0.3→0.8.x), llama.cpp-fork proxy | host-buffer/pinned experts, per-token H2D of activated experts only, shared-mem LRU + selective router-map pin, VRAM expert cache | **Y** — executes router's set after activation (post-route copies); quantized GGUF artifacts aside | DS-V2-Lite 12.5→37.9 t/s; Qwen3.6-A3B 10→50.2 t/s; **DSv4-Flash 4–5 t/s, RSS 84→17–24 GB** (RTX 4090) | github + prebuilt CUDA bins (sm_61–sm_120a) | D-RES, D-DSV — already an AGENTS.md idea source; this verifies its mechanism |
| C3 | **kimi-k3-in-c** | github FareedKhan-dev | C99 engine; O_DIRECT expert streaming from 1.56 TB checkpoint; routed-expert LRU + batch prefetch; dense trunk pinned+ring | **Y — claims byte-identical output across memory budgets** (verifiable property, exactly dee's contract language) | 2.78T-param Kimi K3 in 8.24 GB RSS; 26.5→5.6 s/tok across 8→128+ GB ladder | full C99 source | **D-RES, D-FILL** — strongest independent existence proof of dee's thesis; AGENTS.md idea source |
| C4 | **Edge0** | edge0.ai + HF Edge0-35B-A3B-preview | MLX/Apple-Silicon SSD expert streaming + trained "Prerouter" head (dual layer+token shift) + Recover-LoRA | **partial/N** — `staged_replace`/`pred_inds` decode path routes through *predicted* set (ROUTING on that path); base SSD-offload+prefetch pipeline is exact-safe; int4+LoRA = PRECISION | +59% decode from prerouter overlap; 35B in ~2.9 GiB active mem, 14.9–17.7 tok/s | code mirrors (cephalochromoscope.net), HF checkpoints | D-HINT (its trained per-layer predictor = AGENTS.md's "credible direction"), D-EXACT |
| C5 | **eMoE** | arXiv 2503.06823 | task-aware memory-efficient serving: periodic expert-predictor invocation, reuse prior prompts' expert sets, task-sensitivity-aware scheduling | **N — ROUTING** (executed set can be a stale/previous-prompt set) | −80% memory, −17% latency | not seen | D-EXACT (boundary case: "reuse previous routing" is a violation) |
| C6 | **CoMoE** | arXiv 2508.09208 | joint expert-aggregation granularity + offloading adaptation for mobile edge; multi-tier storage + prediction | **partial — mixed**: aggregation = IDENTITY/BYTES; the offloading/prefetch submechanisms are hint-legal | −70% memory, −10.5% latency; Switch-Base-128 15.6→4.7 GB | not seen | D-EXACT; (also name-dropped in R8 §2.20) |
| C7 | **EC2MoE** | arXiv 2508.06024 | end-cloud pipeline collaboration + hardware-aware group gate (local filtering merged with global gating) | **N — ROUTING** (gate modified by hardware-awareness) | scalable end-cloud inference (edge testbed) | not seen | D-EXACT |
| C8 | **HetRoute** | arXiv 2608.00577 | unified per-assignment cost model (xmit+offload+queue+quant-loss); routes the router's top-k set *as a whole* across servers via exact enumeration/beam search | **Y at token→expert level** (top-k set preserved); optional replica-precision knob = PRECISION if used | −59% avg / −58% P99 latency; −72% cross-server traffic; 10-server edge testbed, 3 MoEs | not seen | D-SERVE (dispatch-level cost model dee-serve could borrow), D-EXACT |

## 5. Candidate rows — Priority 4: DeepSeek-V3/V4-family routing & serving stack

| # | Work | Ref | One-line mechanism | Exact-safe? | Claimed gain + HW | Code | dee Q |
|---|------|-----|--------------------|-------------|-------------------|------|-------|
| D1 | **DeepSeek official serving stack** — EPLB + DeepEP + DeepGEMM + FlashMLA + profile-data + V3/R1 Inference System Overview | deepseek-ai/* repos + open-infra-index Day-6 | redundant-expert replication + hierarchical/global expert-parallel load balancing exploiting V3's group-limited routing; dispatch/combine all-to-all kernels (0-SM, FP8); P/D disagg + cross-node EP | **Y** — replication/placement/comm only; router untouched | production: 73.7k in / 14.8k out tok/s per H800 node, 545% cost margin; SGLang replication ≈52.3k/22.3k per 8×H100 node | all open source | **D-DSV, D-SERVE** — the authoritative reference for DSv-family EP deployment |
| D2 | **metro** ("Balance Activated Experts, Not Tokens") | arXiv 2512.09277 | EP token-routing balancing *activated experts per GPU* instead of token counts; allGather preserves global top-k | **Y-probably — verify**: claims top-k quality guaranteed; if it ever narrows a token's reachable experts it becomes ROUTING | −11–22% decode latency, +3–21% throughput vs EPLB; ≤4.11× decode tput at fixed SLO; Qwen3 + **DeepSeek-V3** on 8×A100 vLLM + B200 sim | not seen | **D-DSV** (memory-bound-regime insight maps to dee: expert count, not tokens, sets the wall) |
| D3 | **"Orders in Chaos" data-movement study** | arXiv 2510.05497 | 24k-request spatio-temporal profiling of 4 large MoEs (200B–1000B, incl **DeepSeek-V3**, Qwen3); 6 insights; wafer-scale case study | **Y** — measurement + architecture study | 5.3×/3.1× avg speedup on DSv3/Qwen3 (wafer-sim); MoE data movement = 60–90% of latency | **traces open-sourced** (HF dataset, >1k downloads) | **D-DSV, D-HINT** (independent predictability evidence for real DSv3 routing), D-RES |
| D4 | **EPS-MoE** | arXiv 2410.12247 | DP/TP attention + EP MoE + pipeline scheduler + dynamic GroupGemm↔DenseGemm selection under load | **Y** — kernel/scheduling only | +52.4% prefill; **DeepSeek-V2** 100k→120k+ tok/s claimed | not seen | D-DSV, D-SERVE |
| D5 | **GRACE-MoE** | arXiv 2509.25041 | expert grouping + replication + locality-aware routing for distributed inference | **verify** — "locality-aware routing" may bias which experts a token reaches (ROUTING) vs dispatch-only | joint comm+load-balance gains; distributed SMoE | not seen | D-DSV |
| D6 | **DeepSeek-V3/V4 model-level routing facts** (context row, not a system) | DSv3 tech report; AGENTS.md canonical model | sigmoid gating + **aux-loss-free bias LB** + **group/node-limited routing** + shared expert; V4-Flash adds CSA attention + in-checkpoint **DSpark/MTP** buckets (dee's 46×256 universe) | **Y — semantics, not a system** | group-limited routing ⇒ a token's experts co-cluster in ≤M groups — a structural co-activation constraint exploitable for fetch bursts/placement; aux-loss-free bias *drifts* expert popularity over time (residency stats are non-stationary) | HF checkpoints | **D-DSV** — the facts every DSv-family candidate above depends on |
| D7 | **vLLM/SGLang DeepSeek-V4-Flash support + llama.cpp CPU-MoE path** | vLLM recipes; ai-infrastructure cookbook | production EP/PD recipes; llama.cpp target-only GGUF executes routed experts on CPU (3.45 GB stored expert bytes/token across 43 layers) | **Y** (deployment facts) | V4-Flash ~167–180 GB VRAM full; cookbook: ~23 tok/s ceiling at 80 GB/s host-byte-rate CPU-MoE | vLLM 0.25+, llama.cpp | D-DSV — confirms dee's canonical geometry is the field's torture case too |
| D8 | **FlashMLA / DeepGEMM** | deepseek-ai/* | MLA decode kernel / FP8 MoE-grouped-GEMM for DSv3-class models | **Y** | production kernels | open source | D-DSV (kernel-level reference for DSv numerics — relevant to any dee-fast FP8 comparisons) |

## 6. Candidate rows — dee-serve / distributed (deprioritized but recorded)

All exact-safe at the token→expert level (scheduling/placement/comm only);
relevance is Phase-5 dee-serve, not current phases.

| # | Work | Ref | One-line mechanism | Claimed gain + HW | Code | dee Q |
|---|------|-----|--------------------|-------------------|------|-------|
| E1 | **MegaScale-Infer** | arXiv 2504.02263 | attention/FFN disaggregation ("disaggregated EP") + ping-pong micro-batch pipeline + M2N comm lib | 1.9× per-GPU throughput | not seen | D-SERVE |
| E2 | **Janus (inference)** | arXiv 2512.13525 | attention-pool vs MoE-pool disaggregation + μs-scale scheduler balancing activated experts across MoE instances + SLO-aware scaling | ≤4.7×/3.9× per-GPU throughput (numbers differ across versions) | not seen | D-SERVE |
| E3 | **EaaS (Expert-as-a-Service)** | arXiv 2509.17863 | MoE modules as stateless disaggregated services + CPU-free P2P comm; fault-tolerant | <2% throughput loss under failures; −37.5% resources via fine-grained scaling | not seen | D-SERVE |
| E4 | **MoEless** | arXiv 2603.06350 | serverless MoE serving; layer-aware *load* predictors drive expert scaling/placement | −43% latency, −84% cost; Megatron, 8-GPU | not seen | D-SERVE, D-HINT (predictor→scaling, legal) |
| E5 | **Aurora** | arXiv 2410.17043 | joint model-deployment + all-to-all transmission-order optimization (3/4 cases provably optimal) | ≤2.38× hom / 3.54× het clusters | not seen | D-SERVE |
| E6 | **Comet** | arXiv 2502.19811 (MLSys'25), bytedance/flux | fine-grained compute↔comm overlap inside the MoE layer via dependency-resolved shared-tensor pipelines | 1.96× layer / 1.71× e2e; deployed at 10k-GPU scale | github.com/bytedance/flux | D-SERVE |
| E7 | **MoE-Gen** | arXiv 2503.09716 | module-based batching: tokens accumulate in host RAM, large batches launched to GPU; per-module batch sizing for overlap | 8–31× vs FlexGen/MoE-Lightning/DeepSpeed; DSv2-236B on A5000 24GB + 512 GB RAM | github.com/EfficientMoE/MoE-Gen | D-SERVE (throughput regime dee says matters most) |
| E8 | **MoE-SLC** | liu2025optimizing (survey [107]) | Bayesian-opt + δ-greedy deployment config for billed-cost serverless MoE | cost-optimized deployment | not seen | D-SERVE |
| E9 | **CoServe** | arXiv 2503.02354, ASPLOS'25 | dependency-aware request scheduling + expert management for **CoE** (collaboration-of-experts / compositional experts) on CPU+GPU | reduced expert switching vs LRU-class mgmt | ASPLOS artifact | D-EXACT boundary note: CoE ≠ sparse-gated MoE — different model class; keep for matrix disambiguation |
| E10 | **ExFlow** | arXiv 2401.08383 | inter-layer expert-affinity mining → ILP expert placement; context-coherent EP halves all-to-alls | −67% cross-GPU routing latency; ≤2.2× vs DeepSpeed-MoE | github.com/YJHMITWEB/ExFlow | D-SERVE, D-RES (affinity tables = legal placement/hint input) |

## 7. Ancestors, practitioner systems, unresolved IDs

| # | Work | Ref | One-line mechanism | Exact-safe? | Claimed gain + HW | Code | dee Q |
|---|------|-----|--------------------|-------------|-------------------|------|-------|
| F1 | **MoE-Deploy / "Towards MoE Deployment" (Expert Buffering)** | arXiv 2303.06182, NeurIPS'24 | hot-experts-in-GPU + rest-in-CPU buffering (the ancestor of every expert cache) + dynamic gating + load balancing | **mixed** — Expert Buffering is exact-safe; dynamic gating drops experts = ROUTING | 6.21–11.55× throughput; −1.47× memory | NeurIPS artifact | D-RES (lineage), D-EXACT |
| F2 | **Brainstorm** | OSDI'23 (Cui et al.) | dynamic-NN framework; runtime MoE dispatch optimization (expert reordering, batched dispatch) — the standard MoE-Infinity baseline | **mostly Y** (scheduling); contains routing-shape optimizations → verify piece-wise | baseline in MoE-Infinity evals | microsoft | D-SERVE |
| F3 | **sabrewing MoE runtime plan** | github Schneewolf-Labs/sabrewing `docs/moe-runtime-plan.md` | practitioner rebuild: explicit VRAM↔RAM↔NVMe unified pager (experts+KV same pages), io_uring deep-QD cold reads, mmap warmth, heat-driven eviction, cross-layer lookahead bet | **Y by design** ("explicit — not OS page cache — because we know the access pattern: the router tells us") | **independent measurement: decode is ~94% cold-first-touch** on a too-big model → matches dee's sealed finding (935/2364 records repeat) | full repo | **D-FILL, D-RES — direct corroboration of dee's core observation** |
| F4 | **memra** | github avifenesh/memra `ARCHITECTURE.md` | persistent fixed-address GPU expert-slot cache + SLRU + second-miss admission + pinned host experts + per-op vs dispatch critique | **Y by design** | targets beating vLLM/SGLang/llama.cpp on MoE-that-doesn't-fit | full repo | D-RES (second-miss admission = concrete alternative to plain LRU worth simulating on sealed trace) |
| F5 | **PowerInfer / PowerInfer-2 / DejaVu lineage** | PowerInfer (SOSP'24), PowerInfer-2, DejaVu (ICML'23) | activation-sparsity predictors + CPU-GPU hybrid execution for **dense** LLMs — the generic ancestor of "predict→fetch" | predictor-as-fetch is hint-legal pattern; PowerInfer-2 splits neuron exec CPU/GPU | open source | D-HINT (predictor lineage), D-SINK |
| F6 | **EIO-MoE** | survey ref [212], Expert-Offloading section | per R8's §2.20: expert-granularity I/O pipeline | likely Y (I/O only) | — | **unresolved**: title search does not surface it; R12 should pin the citation (possibly a venue-version-only name) | D-FILL |

## 8. Appendix A — condensed EP/training family (not gap candidates; recorded so R12 does not re-surface them)

GShard, FastMoE, Tutel, MoESys, Alpa, BaGuaLu, SmartMoE,
Switch-Transformers, HashLayer, Prophet (placement model), MoE-Prediction
(load forecasting), Lazarus (elastic training), FlexMoE (NSDI'23 dynamic
placement), Lynx, Base-Layers, MoE-ECR, Janus-SoCC'23 (training), Hetu-MoE,
DeepSpeed-MoE, DeepSpeed-TED, Lina, TA-MoE (2302.09915 topology-aware
training), LocMoE, Parm (2407.00599 training comm schedules), HiDup,
ScheMoE, PipeMoE (INFOCOM'23), MPipeMoE (2506.22175), MPMoE, ScMoE
(2404.05019 — **note: architecture change, top-1 shortcut + shared MLP,
different model → not exact-adjacent**), DyNN-Offload (HPCA'24 DyNN
training memory mgmt, pilot-model prefetch ancestor), Hardware-level
leftovers for R9 (FLAME-accelerator, Space-Mate, M³ViT, Edge-MoE-hw).

## 9. Duplicates / aliases of the named set — flag for R12 dedup

1. **fMoE = FineMoE**: arXiv 2502.05370's system name is "FineMoE"; the
   named set's "fMoE" is the same paper (EuroSys'26). One entry.
2. **Cross-Layer Gate / Fate**: ONE paper — "Fate: Fast Edge Inference of
   MoE Models via Cross-Layer Gate" (arXiv 2502.12224, WWW'26; code
   FFFzy/Fate_open). The slash in the named set is name+mechanism, not two
   systems. Note it also bundles INT2/INT4 hybrid transfer (PRECISION knob).
3. **MoE-Fiddler = Fiddler** (survey's name; arXiv 2402.07033). One entry.
4. **Swap-MoE = SwapMoE** (survey name vs R8's SwapMoE 2308.15030).
5. **ExpertFlow collision — TWO papers**: 2410.17954 (predictive caching +
   token scheduling; survey's [60], He et al.) vs 2510.26730
   (adaptive-horizon prefetch + cache-aware routing fallback; R8's cite,
   Shen et al.). Different mechanisms; matrix needs both IDs disambiguated.
6. **Janus collision**: SoCC'23 Janus (training framework, survey [105]) vs
   arXiv 2512.13525 Janus (attention/expert disaggregation serving). My E2
   is the latter only.
7. **SpecMoE collision cluster — FOUR distinct items**: (a) "SpecMoE"
   self-assisted SD+offloading (arXiv 2604.10152); (b) R8's
   "SpecMoE/Eliseev & Mazur" note actually points at 2312.17238 =
   **Mixtral-Offloading itself** (likely a conflation — the named set's
   Mixtral-Offloading already covers it); (c) SP-MoE 2510.10302;
   (d) SpecMoEOff 2508.21706 (row A7); plus MoE-SpeQ 2511.14102 and
   MoE-SpAc 2603.09983 in the same neighborhood. R12 must not merge (a),
   (c), (d).
8. **"MoE-Offloading"/"MoE-OnDemand"** in several papers' baselines are
   generic labels for Mixtral-Offloading/reactive-load baselines, not
   separate systems.
9. **PreMoE ambiguity**: R8 §2.20 listed "PreMoE" as exact-adjacent;
   arXiv 2505.17639 "PreMoE" is actually expert *pruning+retrieval*
   (TCESS/TAER — IDENTITY/BYTES, not bytes-movement). Either two PreMoEs
   exist or R8's classification needs correcting — flag.
10. **EdgeMoE vs Edge-MoE**: survey lists EdgeMoE [208] (offloading, Yi et
    al. 2308.14352) AND Edge-MoE [152] (hardware-level) — two distinct
    papers sharing a near-identical name.
11. **FLAME collision**: hardware-level FLAME [103] (accelerator) vs
    FLAME-MoE (arXiv 2505.20225, open MoE model/training platform — a
    usable small-scale experiment substrate, not a system; not rowed as a
    candidate).
12. **MoE-Infinity repo lineage**: paper repo TorchMoE/MoE-Infinity;
    current maintained repo EfficientMoE/MoE-Infinity (same team; the new
    version already lists DeepSeek-V4-Flash FP4 expert offload support —
    worth a footnote on the named-set row: the maintained implementation
    targets dee's canonical model).

## 10. Overlap with sibling tracks (do-not-double-count list)

Already catalogued by R8 (approx-methods): HOBBIT, EdgeMoE, AdapMoE, QMoE,
Cache-Prior/MCCE, BuddyMoE, ReMoE, SMoE, CacheMoE, Pre-gated, MoE-I², NAEE,
MC-SMoE, AdaMoE, SiDA, SliceMoE, DynaExq, DyMoE, MoMP, SwapMoE, SPICE,
Read-ME, SpecMoE(2312.17238), ProMoE, SP-MoE, MoE-SpeQ, ExpertFlow
(2510.26730), "Cacheable by Design?" + name-drops of EIO-MoE, PreMoE,
CoMoE, BigMoeOnEdge. → My rows A6–A11/C5–C7 are NEW (different papers);
CoMoE (C6) and EIO-MoE (F6) got full rows here only because they were
one-line mentions in R8's out-of-scope list.
Already catalogued by R9 (hardware/activation): MoNDE, PIMoE, DynaNDE,
context-aware CXL-NDP, Sieve, Duplex, HDA-MoE, HCRMap, M2NDP, CXL-BW-amp,
Aquabolt-XL/AXDIMM/CXL-PNM/LPDDR5X-PIM, KVNAND, NVLLM, InstInfer,
"HBF Sucks!" → not re-rowed; B3 (HBF dual-route) is a different paper.
Already named via AGENTS.md external-refs but NOT in the named set →
upgraded to full rows here: moe-l2 (C2), kimi-k3-in-c (C3), Edge0 (C4).
BigMoeOnEdge (R8 mention): could not verify a public repo/paper in this
pass — left to R12.

## 11. Falsification notes (deliverable-grade)

- **B4 (2508.06978)** is a direct attack on dee-local's premise: SSD expert
  reads ~12× the decode energy of HBM residency on DS-R1 geometry. dee's
  counter is residency amortization (the sealed bank's 935/2364 repeat
  rate means GB/token ≪ all-miss), but the paper's framing — "prefetch
  hides latency, never energy" — must be answered in the economics phase.
- **A5 (2608.12103)** cuts both ways for dee: (i) recency ≈ oracle
  frequency at equal memory — supports dee's plain-LRU choice; (ii) oracle
  pin still 1.09–1.11× on pread replay — bounds the headroom dee is
  leaving; (iii) lookahead-at-64.7%-recall ≈ +0.3% median — consistent
  with dee's own "generic predictor REJECTED" and the legal-prefetch
  ~29%-of-wall oracle bound; (iv) the balloon+MGLRU ~2× traffic
  overstatement is a *methodology* warning for any future dee page-cache
  arm: enforce capacity via cgroup/mem=, never balloon+mlock.
- **F3 (sabrewing)** independently measured ~94% cold-first-touch decode —
  the same finding as dee's sealed journal (935/2364 records ever repeat).
  Two independent systems ⇒ the cold-miss wall is structural, not
  dee-artifact.
- **FreeToken issue #151** (github): its auto-selected `hybrid` backend ran
  8.3× SLOWER than `offload` on DSv4-Flash/2×3090 — a live falsification
  datapoint for naive CPU/PCIe-split heuristics (relevant to D-SINK arm
  design: the q* split policy can mis-pick).
- **C5 eMoE / A8 YALIS / C6 CoMoE / C7 EC2MoE** form a family that saves
  bandwidth by changing which experts run — each is an exact-mode
  counterexample that keeps the contract non-negotiable.

## 12. What this means for R12 synthesis

1. **dee's differentiator narrows but holds**: the field's 2026 exact-safe
   frontier is router-preserving prefetch (A1–A4) + kernel-vs-user tier
   ownership (A5) + intact-precision CPU-hybrid (C1). Nobody else combines
   *sealed-counter evidence + byte-exact packed format + SSD-native
   three-tier + fail-closed contract*. Closest peer for the CPU-sink arm:
   C1 (OSDI'26) — recommend reading before finalizing the sink design.
2. **Top-5 candidate rows for matrix promotion**: A5 (kernel-tiering
   study), A1 SpecPrefetch, B1 FlashMoE, C1 OSDI'26 system, D1 DeepSeek
   stack — each answers a dee question the named set leaves open.
3. **D-DSV needs its own matrix column**: D1–D8 show DeepSeek-family
   routing (aux-loss-free bias drift, group-limited co-activation,
   in-checkpoint DSpark) is its own systems problem — dee's canonical
   model choice is well-aligned with where the literature is moving.
4. **Alias debt is real**: §9's twelve flags must be resolved before the
   matrix dedups correctly (ExpertFlow×2, Janus×2, SpecMoE×4, fMoE/FineMoE,
   Fate=Cross-Layer-Gate).
5. **Re-verify before scoring**: exact-safe verdicts marked "probably/
   verify" (A4, A6, D2, D5, E2, F2) rest on abstract-level claims of
   router preservation; R12 should read full texts before any of them
   anchors an "exact-safe exists in prior art" claim.

### Sources mined
- github.com/MoE-Inf/awesome-moe-inference (full section enumeration via
  the survey's citation graph, arXiv 2412.14219).
- arXiv 2025–26 sweep on: expert offloading/prefetch, SSD/flash/HBF/page-
  cache residency, CPU-GPU hybrid, DeepSeek V3/V3.2/V4 serving+EP, MoE
  serving disaggregation, SD×MoE interaction, edge/collaborative routing.
- Practitioner layer: PyPI/github (moe-l2, kimi-k3-in-c, sabrewing, memra,
  FreeToken docs+issues, edge0), vLLM/SGLang recipes, DeepSeek
  open-infra-index.
- dee-internal cross-checks: AGENTS.md, LEGAL_PREFETCH.md,
  OFFICIAL_LOOKAHEAD.md, SERIALIZATION_VERDICT.md, regime-C CONTRACT.md,
  R6/R8/R9 prior-art deliverables (sibling worktrees) for dedup flags.
