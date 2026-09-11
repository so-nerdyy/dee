# DEE_MOE_PRIOR_ART_MATRIX — consolidated R-series synthesis

- Source corpus: `research/prior-art/r01`–`r11-*.md` (branch
  `research/prior-art-rNN`, all committed at worktree base `dc78dc4`;
  R13's resident-garbage fix lives on the campaign line at `79eac7e`).
- Author: orchestrator (R12 deliverable), 2026-09-11.
- **Tier labels** — MEASURED = sealed dee evidence; SIMULATED = sealed-
  journal replay (gate-validated); DERIVED = arithmetic on labeled inputs;
  PAPER-REPORTED = authors' claims (metadata, never dee evidence);
  UNKNOWN = no evidence. Paper speedups below are **contextual metadata
  only** — the "dee upside" column scores each mechanism against dee's
  MEASURED wall components, never the paper's headline.
- **dee wall anatomy (MEASURED, sealed 2×T4, /tmp bank 0.29–0.37 GiB/s)**:
  decode wall 66.2–71.4 s / 16 tok. `fill_wait` 41.99 s (**63.4%**) —
  cold storage reads. `stage_enqueue` 9.44 s (**14.2%**) — H2D submit +
  gather + cache ops. `native_output_sync` 4.89 s (**7.4%**). Unattributed
  dense-attention/orchestration 9.20 s (**13.9%**). Compute dispatch +
  combine + readiness ~1.9%. SSD 2.07 GB/tok, H2D 3.71 GB/tok.
- **Exactness contract**: prediction may drive prefetch *hints* only;
  native router authoritative; never change which expert executes, its
  bytes, precision, or ordering. A hint error wastes bandwidth, never
  correctness.
- **Alias resolutions applied** (R11 §9 + R7 naming notes): fMoE = FineMoE
  (2502.05370); Fate = "Cross-Layer Gate" (2502.12224); MoE-Fiddler =
  Fiddler (2402.07033); Swap-MoE = SwapMoE (2308.15030); ExpertFlow is TWO
  papers (2410.17954 "EF-a" + 2510.26730 "EF-b"); SpecMoE name resolves
  FOUR ways (mixtral-offloading's internal name 2312.17238 / SP-MoE
  2510.10302 / SpecMoEOff 2508.21706 / self-assisted SD 2604.10152);
  Janus: SoCC'23 (training) ≠ 2512.13525 (serving); EdgeMoE ≠ Edge-MoE;
  FLAME accelerator ≠ FLAME-MoE model; PreMoE 2505.17639 is
  pruning+retrieval (corrects an earlier exact-adjacent listing); EIO-MoE
  citation unresolved.

## Matrix

Column shorthand: XS = exact-safe for existing checkpoint; Train = training
required; ΔRouter = changes router; PredIn = prediction input; Look =
lookahead; GPUcache = device cache policy; Tier = RAM/SSD usage; CPU =
CPU computation; H2D↓ / SSD↓ = reduction mechanism; XReq = cross-request
assumption; EvalHW = model/hardware evaluated; Gain = PAPER-REPORTED
headline; Base = baseline; Code = reproduction availability; deePhase;
Upside = expected upside vs dee's measured wall components; Cmplx;
Conf; Dispo (BUILD / SIMULATE / MEASURE / DEFER / APPROX-ONLY / REJECT).

### Exact-safe movement/placement systems

| System | Yr | XS | Train | ΔRouter | PredIn | Look | GPUcache | Tier | CPU | H2D↓ | SSD↓ | XReq | EvalHW | Gain | Base | Code | deePhase | Upside (dee-measured) | Cmplx | Conf | Dispo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Mixtral-Offloading** (2312.17238; "SpecMoE" name (a)) | 23 | Y (HQQ quant = sep config) | no | no | gate_{L+1}(x_L) | ~1 L | per-layer LRU | RAM-only | no | LRU + 1L spec-prefetch | none | in-request temporal | Mixtral-8x7B HQQ; T4/RTX30/A100 | 3.2× vs naive | HF offload | public | P2 done / P5 | ≤2% decode wall — subsumed by built tiers; hint arm bank-bound (R3: k=1 ⇒ 0 s) | LOW | HIGH | REJECT (subsumed); DEFER hint |
| **MoE-Infinity** (2401.14361) | 24 | Y | no (online trace) | no | per-request EAM → pEAM | temporal n+1 | activation-aware evict | RAM + SSD offload dir | no | trace-guided prefetch | SSD→RAM leg in OSS | in-request | DSv2-Lite/Switch/NLLB/Mixtral/Arctic; A5000 | 3.1–16.7× TPOT | DeepSpeed/vLLM/Ollama | **public, runs DSv4-Flash FP4** | P2 sim / P5 | ≤2–3% on sealed bank (LRU→MIN gap ~2.8pp); cross-request larger | MED | HIGH | SIMULATE pEAM arm; DEFER serving; **cross-validation target** |
| **ExpertFlow-a** (2410.17954) | 24 | Y (mispredict→demand fetch) | YES (T5 RPP, BCE) | no | full input seq → (B,S,L,E) | all layers | ECE predictive cache | RAM; SSD ckpt | no | all-layer prefetch + token re-batch | none | batch regrouping | Switch/Mixtral/Qwen/DS-MoE; 1 GPU | ≤10×; −93.7% mem | SE-MoE/CacheMoE | none found | P5 | ≤2% sealed; scheduler unmeasurable at batch-1 | MED-HI | MED | DEFER (P5) |
| **ExpertFlow-b** (2510.26730) | 25 | PARTIAL (cached-pred path bypasses top-k — flagged) | YES (RF Δ) | yes on cached path | pregate + intermediate + RF | adaptive S=N_e·E_s/(C_s·T_l)≈2 | two-level LRU | RAM staging; SSD | no | adaptive-horizon prefetch | none | cached preds keyed (seq,layer,step) | DSv2-Lite/Qwen-MoE 4bit; A6000/910B | ~99.9% wait cut | baseline offload | none | P5 (horizon rule) | ~0 sealed — the formula itself says S≈2 layers ⇒ prefetch can't pay at 0.33 GiB/s | LOW-MED | MED | DEFER (formula to P5); APPROX-ONLY flag |
| **fMoE / FineMoE** (2502.05370) | 25/26 | Y | no (map similarity) | no | prompt embedding + partial trajectory vs stored maps | config | map-guided per-expert priority | RAM 480GB; SSD maps | no | +39% hit rate | maps only | EXPLICIT cross-request store | Mixtral/Qwen/Phi-MoE; 6×3090 | −47% lat | MoE-Inf/ProMoE et al | public demo | P5 (regime-C) | ~0 sealed (935/2364 repeat); semantic-seeded policy_slots = the salvage | MED | HIGH | DEFER (P5/regime-C) |
| **ProMoE** (2410.22134) | 24 | Y (hint-only) | YES (per-layer MLP ~2M) | no | layer input → gate outputs | ~2 L (stride) | proactive cache + preempt | RAM; SSD | no | lookahead prefetch + scheduling | none | in-request | Mixtral-class; RTX4090 | 2.07–2.20× | framework offload | none | P5 | R3: needs k≥2 AND p≥0.75 ⇒ ~8–13 s region; stride is the ONLY ~2L published shape | MED | MED-HIGH | SIMULATE (miss-stream recall, CPU-only) → DEFER P5 |
| **Fate / Cross-Layer Gate** (2502.12224) | 25 | Y (hint arm; INT2/4 side = PRECISION) | no | no | gate_{L+1}(x_L) | ~1 L | n/a | RAM | no | 1-layer gate-hint prefetch | none | none | DeepseekMoE-16B (64e top-6), Qwen | ~97% pred acc | — | public | P5 candidate | **R3: k=1 ⇒ 0.00 s on sealed bank** — dead regardless of accuracy; reopens ≥0.7 GiB/s | LOW | HIGH | DEFER (bank-bound); evaluate only after gate_trace capture |
| **SpecPrefetch** (2607.24787) | 26 | Y ("errors affect transfer not outputs") | YES (layer adapters) | no | per-layer adapters | 1 L | window-aware budget | edge | no | async prefetch | none | none | Qwen3-VL/DSv-VL2-Tiny; Snapdragon | +20% decode | — | promised | P5 | same k=1 ceiling (0 s sealed) | MED | MED | DEFER |
| **APEX** (2608.11688) | 26 | Y (correctness-preserving mode) | YES | no | pre-attention hidden | intra-layer | confidence-sized fetch | RAM | no | intra-layer prefetch | none | none | DSv2-Lite/Granite/Phi-MoE | −26% TPOT exact-mode | — | none | P5 | intra-layer lead ≪ k=1 ⇒ ~0 sealed | MED | MED | DEFER |
| **ST-MoE** (2606.15453) | 26 | Y (claims routing preserved) | YES (CCT) | no | ids→ids cross-layer table | 1 L | — | HW codesign | no | prefetch | none | none | Qwen/DeepSeek claims | 2.5× vs GPU | — | none | P5 | R2 F4: same-index overlap at chance; CCT is the *only* ids-channel worth a journal-only test | LOW (journal test) | MED | MEASURE (journal CCT test, local, free) |
| **PreScope** (2509.23638) | 25 | Y-probably (verify CPU-exec consumes router output) | YES (LLaPor) | no | layer-group features | cross-layer | PreSched global | commodity | partial (CPU co-exec) | prefetch+sched | none | none | 10B MoE; commodity GPU | +141% tput | SOTA | none | P5 | hint + scheduler; CPU-exec side = R6 seam | MED | MED | DEFER; verify exactness at source |
| **DuoServe-MoE** (2509.07379) | 25 | Y | YES (layer predictor) | no | prefill 2-stream + decode predictor | phase-split | — | RAM | no | prefetch | none | none | 4-bit Mixtral; 1 GPU | 1.42–7.54× | SOTA | none | P5 | phase-split prefetch model | MED | MED | DEFER |
| **SpecMoEOff** (2508.21706) | 25 | Y (SD verify exact) | — | no | SD enlarges workload | draft | — | — | CPU verify kernel | hide offload lat under SD | none | none | SD MoE | ≤2.5× | SOTA | none | P5 | SD-out-of-scope; V4-Flash has in-checkpoint DSpark (R1 F8) | HIGH | MED | DEFER |
| **FlashMoE** (2601.17063) | 26 | Y | no | no | ML recency+frequency | — | — | **SSD-native** | no | — | SSD-hit policy | none | user desktop | +51% hit; ≤2.6× | LRU/LFU | none | P5/D-RES | only SSD-native cache-policy baseline; ~1–2 s band on sealed | MED | MED | SIMULATE (ws sim arm) |
| **FluxMoE** (2604.02715) | 26 | Y | no | no | budget-aware residency planner | — | paged expert tensors | GPU+DRAM+storage | no | residency planner | — | none | framework-compat | dynamic under KV pressure | — | none | P5 | paged-expert comparison point vs dee slot model | MED | MED | DEFER |
| **Kernel-tiering study "Who Should Own the Expert Cache?"** (2608.12103) | 26 | Y (token-identical claimed) | no | no | router-trace replay | — | page cache + MGLRU + readahead | OS-managed | no | — | kernel recency ≈ oracle | none | GH200, 1.45 TB pool, 3 models' traces | oracle pin only 1.09–1.11× | user-space oracle | methodology | **D-RES direct** | most dee-shaped external result: validates LRU≈MIN + bounds pin headroom; methodology warning (balloon ~2× traffic overstatement) | LOW (read) | HIGH | MEASURE (cite + reuse methodology) |
| **Klotski** (2502.06888) | 25 | Y | no | no | activation paths | batch | — | — | no | I/O-compute planner | — | batch | serving | ≤85× (batch regime) | SOTA | none | P5 | batch-only | HIGH | MED | DEFER (P5) |
| **DeepSeek official stack** (EPLB/DeepEP/DeepGEMM/FlashMLA) | 25–26 | Y | no | no | EP placement/dispatch | — | replication | EP nodes | — | dispatch/combine comm | — | EP serving | H800 prod | 73.7k tok/s/node prod | — | open | D-DSV | authoritative DSv-family reference; group-limited routing + aux-free-bias facts every DSv row depends on | — | HIGH | REFERENCE (not a dee mechanism) |
| **metro** (2512.09277) | 25 | Y-probably | no | verify | balance activated experts/GPU | — | — | EP | — | — | — | — | Qwen3 + DSv3, 8×A100 | −11–22% decode lat | EPLB | none | D-DSV | expert-count-sets-wall insight transfers | — | MED | REFERENCE |
| **Orders in Chaos** (2510.05497) | 25 | Y (study) | no | no | 24k-req traces | — | — | — | — | — | — | real traces | DSv3/Qwen3/4 MoEs | data movement = 60–90% latency | — | **traces open (HF)** | D-DSV | independent predictability evidence for real DSv3 routing — free corpus | LOW | HIGH | MEASURE (mine traces for dee predictor eval) |
| **sabrewing MoE plan** (practitioner) | 26 | Y by design | no | no | explicit pager; router = access pattern | — | unified VRAM↔RAM↔NVMe pager | SSD-native | — | io_uring deep-QD | — | — | too-big model | **~94% cold-first-touch decode (independent)** | — | full repo | D-FILL | direct corroboration of dee's sealed finding | LOW | HIGH | REFERENCE (corroboration) |
| **moe-l2** | 26 | Y (post-route copies) | no | no | — | — | shm LRU + selective pin | RAM→VRAM | CPU path | per-token activated-only H2D | — | — | DSv2-Lite/Qwen3.6/DSv4-Flash; RTX4090 | DSv4-Flash 4–5 t/s, RSS 84→17–24 GB | — | github+bins | D-RES | verified external analog of dee's per-token H2D shape | LOW | HIGH | REFERENCE |
| **kimi-k3-in-c** | 26 | Y (byte-identical claim) | no | no | routed LRU + batch prefetch | — | — | O_DIRECT SSD | — | — | streaming | — | 2.78T Kimi K3; 8→128 GB | 26.5→5.6 s/tok | — | full C99 | D-RES | strongest independent existence proof of dee thesis | LOW | HIGH | REFERENCE |
| **llama.cpp CPU-MoE (`-cmoe`/`-ncmoe`)** | 26 | Y (GGUF MXFP4 same grid; dense→Q8_0 = non-exact) | no | no | static placement | — | — | page-cache streamed | YES (CPU experts) | 0 expert H2D by construction | page cache | none | V4-Flash supported | ~23 t/s ceiling @80 GB/s host | — | active | **matched baseline** | the ONE runnable matched control on 2×T4 (R10) | MED (GGUF build = CPU batch) | HIGH | **MEASURE (CPU batch)** |
| **HBF dual-route** (2608.14333) | 26 | Y (arch sim) | no | no | early expert det. | — | — | die-stacked NAND | — | dual-path delivery | HBF direct | — | simulator | 1.94× tput | HBM-relay | sim | P6 | hardware row (R9) | — | MED | DEFER (P6) |
| **SSD-energy counterpaper** (2508.06978) | 25 | Y (analysis) | no | no | — | — | — | — | — | — | — | — | DS-R1 | **NEGATIVE: SSD reads ~12× decode energy vs HBM** | — | n/a | D-ECON | falsification-grade: "prefetch hides latency, never energy" — dee's answer = residency amortization | LOW | HIGH | MEASURE (answer in econ phase) |
| **HetRoute** (2608.00577) | 26 | Y (top-k preserved) | no | no | per-assignment cost model | — | — | 10-server edge | — | −72% cross-server traffic | — | — | 3 MoEs edge | −59% lat | — | none | D-SERVE | dispatch-level cost model | MED | MED | DEFER (P5) |
| **MoE-Gen** (2503.09716) | 25 | Y | no | no | host-RAM accumulate → large-batch launch | — | — | 512 GB host | — | module batching | — | batch | DSv2-236B; A5000 | 8–31× | FlexGen et al | public | D-SERVE | throughput regime | MED | HIGH | DEFER (P5) |
| **ESS** (2512.10576) | 25 | Y | no | no | KV/latent temporal locality | — | sparse latent pool | SSD KV | — | — | latent cache tier | — | DSv3.2 decode | — | — | none | D-DSV | adjacent (KV not experts) | — | MED | REFERENCE |
| **Comet** (2502.19811) | 25 | Y | no | no | compute↔comm overlap | — | — | — | — | — | — | — | 10k-GPU prod | 1.96× layer | — | public | D-SERVE | prod-scale comm overlap | — | HIGH | REFERENCE |
| **MegaScale-Infer / Janus-inf / EaaS / MoEless / Aurora / ExFlow / EPS-MoE / GRACE-MoE / MoE-SLC / CoServe / Brainstorm** | 23–26 | mostly Y | mixed | verify piecewise | scheduling/placement/comm | — | — | EP/serverless | — | — | — | serving | various | various | mixed | D-SERVE | disaggregation/EP/batching prior art; none touch dee's single-node bank wall | — | MED | DEFER (P5) |

### Second execution sink (the wall-sized class)

| System | Yr | XS | Train | ΔRouter | PredIn | Look | GPUcache | Tier | CPU | H2D↓ | SSD↓ | XReq | EvalHW | Gain | Base | Code | deePhase | Upside | Cmplx | Conf | Dispo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Fiddler** (2402.07033) | 25 | Y as mechanism | no | no | latency model per-layer | — | — | host exec | **YES** | activation round-trip ≪ record | — | none | Mixtral; 24GB GPU | 1.26×/11.57× beam | SOTA | public | **P2.x/P5 via R6** | R5: overlapped t_cpu≤25–30 ms ⇒ ~6–9 s stage-enqueue prize; fill invariant | HIGH (R6 G1–G10) | HIGH | **MEASURE (t_cpu at real geometry → then sink seam)** |
| **KTransformers** (kernel donor) | 25–26 | Y (byte-identical MXFP4 input proven) | no | no | — | — | — | full-pool host residency (no evict) | YES | hybrid | — | — | DSv3/V4; Xeon4+4090 | 21 TFLOPS BF16 claim | — | public | component donor | kernel structure import; do-not-import list stands (pool/mask/SGLang); AVX2 path is Kaggle-realistic; no AMX/BF16 | MED | HIGH | **BUILD components (kt_cpu_bridge); no baseline run** |
| **FreeToken** (2608.16157) | 26 | Y native V4-Flash | no | no | q* bandwidth-adaptive split | — | global LRU | RAM floor high | YES | activation | — | — | DSv4-Flash; 2×3090+503GB | 1.5–2.3× decode | — | public | D-SINK | q* split = plan_split analog; **issue #151: hybrid 8.3× SLOWER than offload** = live falsification datapoint | MED | HIGH | REFERENCE + cautionary |
| **DAOP** (2501.10375) | 25 | **N** — proxy hidden state + predicted set (approx by design) | no | partial | gate_{L+1}(x_L) + prefill→decode placement | 1 L | per-seq static split | RAM exec; SSD ckpt | YES (approx) | activations cross link | none | per-seq | Mixtral/Phi-3.5-MoE; A6000 | ≤8.2× vs caching | Fiddler-class | public | P5 via R6/R9 | **only surveyed mechanism attacking the 63% fill wall** — exact variant = R6 CPU-sink, NOT importable | HIGH | HIGH | APPROX-ONLY as published; MEASURE exact analog |
| **OSDI'26 CPU-GPU hybrid (SLP/DSLP)** (2606.10493) | 26 | **Y — explicitly intact-precision** | no | no | AVX-512 FP8 GEMV + P/D disagg | — | — | zero-copy | YES | hybrid exec | — | — | DSv3 intact FP8; ≤2×5090 | 21.5 tok/s intact FP8 | — | none | **D-SINK closest peer** | the nearest exact-CPU-sink peer — read before finalizing sink design | HIGH | MED-HIGH | MEASURE (read + compare design) |
| **HybriMoE** (2504.05897) | 25 | Y (scheduling) | no | no | intra-layer CPU/GPU dyn sched | — | score cache | KT-based | YES | intra-layer split | — | — | KT-based | 1.33×/1.70× | SOTA hybrid | none | P5 | evidence intra-layer split > static under unstable patterns → per-miss policy | MED | MED | REFERENCE |
| **MoE-Lightning** (2411.11217) | 25 | Y | no | no | HRM placement | — | — | — | YES | CGOPipe overlap | — | batch | Mixtral; T4 | ≤10.3× | offloading | public | P5 | overlap theorem behind R5 §3.2; batch regime ≠ dee batch-1 | HIGH | HIGH | REFERENCE |
| **PowerInfer/-2 / DejaVu** (ancestor) | 23–24 | predictor-as-fetch pattern | YES | no | activation sparsity | — | — | — | partial | — | — | — | dense LLMs | — | — | public | lineage | ancestor; dense-model sparsity ≠ expert routing | — | HIGH | REFERENCE |

### Approximate / contract-violating as published (dee-fast branch candidates)

All rows below are APPROX-ONLY or REJECT under exact mode; salvage noted
where a hint arm survives. Full per-clause analysis in
`research/prior-art/r08-approx-methods.md`.

| System | Ref | Violation class | Salvage | Dispo |
|---|---|---|---|---|
| **SiDA-MoE** | 2310.18859 | ROUTING (predicted set bounds execution; ≤1% acc drop proves divergence) | hint-only degenerate | APPROX-ONLY |
| **Pre-gated MoE** | 2308.12066 | ROUTING (retrained gate IS the router) | hint salvage only | APPROX-ONLY |
| **Speculating-Experts (YALIS)** | 2603.19289 | ROUTING (exec arm runs predicted set); qHS prefetch arm hint-legal | qHS = Fate refinement | APPROX-ONLY as shipped |
| **eMoE** | 2503.06823 | ROUTING (stale/prev-prompt set executes) | none | APPROX-ONLY |
| **EC2MoE** | 2508.06024 | ROUTING (hardware-aware gate filter) | none | APPROX-ONLY |
| **CoMoE** | 2508.09208 | IDENTITY/BYTES (aggregation) + ROUTING | prefetch submechanism hint-legal | APPROX-ONLY |
| **Cache-aware routing family** (Cache-Prior 2412.00099, ReMoE, BuddyMoE, SMoE, MCCE, CacheMoE, EF-b cached path) | various | ROUTING (bias executed set toward resident) | residency insight as contract-deferred regime-C | APPROX-ONLY |
| **Cache-Aware Joint Router Adaptation** | 2609.04895 | BYTES for existing ckpt (adapted backbone); inference mechanism hint-legal | train-for-cacheability — cf. "Cacheable by Design?" 2608.18261 failed perplexity gate (honest negative) | APPROX-ONLY |
| **HOBBIT** | 2411.01433 | PRECISION arm (adaptive bit-width) + ROUTING (approx gating); prefetch arm hint-legal | inter-layer similarity evidence | APPROX-ONLY |
| **AdapMoE** | 2408.10284 | adaptive-k (top-k varies) = ROUTING | prefetch arm | APPROX-ONLY |
| **QMoE** | — | PRECISION (compression) | none | REJECT |
| **EdgeMoE** | 2308.14352 | PRECISION bitwidth arm | path-statistics preload ≈ freq-backfill (measured weak) | APPROX-ONLY |
| **MoE-I² / NAEE / MC-SMoE / AdaMoE / SliceMoE / DynaExq / DyMoE / MoMP / SPICE / MoBiLE** | various | PRECISION/IDENTITY/ROUTING cluster (R8 catalog) | per R8 | APPROX-ONLY/REJECT |
| **Read-ME** | 2410.19123 | different model (refactor + trained pre-gate) | Belady-given-known-routes validates dee's MIN sims | REJECT (out of contract) |
| **PreMoE** | 2505.17639 | IDENTITY/BYTES (pruning) — retrieval side = legal prewarm | evidence FOR regime-C rank stability on DS-R1 | APPROX-ONLY (pruning); retrieval = DEFER-P5 |
| **Predictive prefetch + replication** | 2605.11537 | ~90–95% baseline accuracy ⇒ executed-set drift | — | APPROX-ONLY |
| **Swap-MoE / DyNN-Offload / MoE-Deploy** | 2308.15030 / HPCA'24 / 2303.06182 | ancestors (dynamic gating in MoE-Deploy = ROUTING) | lineage | REFERENCE |

## Per-disposition summary

- **BUILD**: nothing new clears the bar for current phases. The only
  BUILD-flagged item is continuation of already-scoped work — the R6
  CPU-sink seam + kt_cpu_bridge components.
- **MEASURE (cheap, local, unblocks a decision)**:
  1. `t_cpu(1)` at real geometry (R5 — the single most valuable missing
     number; Kaggle-CPU-batch eligible).
  2. `gate_trace` capture + cross-gate recall eval (R2 F12/F13 — inert
     hook, piggybacks on any real-forward evidence rep).
  3. Journal-only ids→ids CCT test (R2 — free, today).
  4. `phase2_prefetch_economics_sim.py` wrong-name variants (R3 — free).
  5. Orders-in-Chaos open traces mined for DSv3 predictability priors.
- **SIMULATE**: MoE-Infinity pEAM arm + FlashMoE policy arm in the ws sim;
  ProMoE-shape trained predictor on the miss stream (needs captures).
- **DEFER (Phase-5 / bank-move-gated)**: the entire ~1-layer-lead
  predictor family (R3 F1 — physically dead at 0.33 GiB/s), all
  cross-request mechanisms (regime-C contract), SD/draft hints, serving
  engines.
- **APPROX-ONLY**: the R8 catalog — never touches sealed exact evidence;
  lives behind the `dee-fast` branch seam.
- **REJECT**: mechanisms subsumed by built tiers (Mixtral-Offloading
  machinery), different-model work (Read-ME), precision-only families.
- **REFERENCE (corroboration, not mechanism)**: sabrewing (independent
  ~94% cold-first-touch = dee's 935/2364), kimi-k3-in-c (existence
  proof), moe-l2, DeepSeek serving stack, llama.cpp CPU-MoE ceiling.

## Cross-cutting conclusions

1. **The literature's floor is host RAM; dee's is the bank.** Every H2D-
  side mechanism is bounded by the idle-gap surface R3 priced — k≥2 +
  p≥0.75 is the entry ticket, worth ~8–13 s max on this bank.
2. **The only wall-sized mechanism class is execution-side** (R5/R6):
   ~6–9 s stage-enqueue prize at `t_cpu ≤ 25–30 ms` under overlap — and
   the fill itself (63.4%) is sink-invariant, so the CPU sink is the
   largest non-hardware lever that survives exactness review.
3. **dee's differentiator narrowed but holds**: the 2026 frontier is
   router-preserving prefetch + kernel-vs-user tier ownership + intact-
   precision CPU-hybrid — nobody else combines sealed-counter evidence +
   byte-exact packed format + SSD-native three-tier + fail-closed
   contract.
4. **Falsification is documented**: SSD-energy counterpaper (~12× energy
   vs HBM — answered by residency amortization, must be argued in econ
   phase); FreeToken issue #151 (hybrid 8.3× slower than offload — the
   q* policy can mis-pick); kernel-tiering study's balloon+MGLRU ~2×
   traffic-overstatement warning for any future page-cache arm.
