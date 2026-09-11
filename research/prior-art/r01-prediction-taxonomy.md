# R1 — Exact-safe prediction taxonomy: which published predictors are pure prefetch hints, and what input/lead they need

- Track: R1 — prior-art classification (classify-do-not-integrate)
- Branch: `research/prior-art-r01` @ `dc78dc4` (worktree `.freebuff/wt/r01`)
- Scope: predictors for expert activation under an **unmodified checkpoint**.
  The exactness contract (AGENTS.md §Exactness philosophy): a predictor may
  initiate IO early; it may NEVER decide which expert executes. Prediction may
  drive prefetch *hints* only; the native router (`router_select`,
  `dee.cpp/scripts/deepseek_v4_layer_common.py:349-398`; engine seam
  `Engine::route_topk`, `dee.cpp/include/dee/engine.h:373-377`) stays
  authoritative. A wrong hint costs bandwidth + ≤1 LRU eviction — never
  correctness (`research/phase2-legal-prefetch/LEGAL_PREFETCH.md` §Legality).
- Tier labels: MEASURED = sealed dee evidence; SIMULATED = journal-replay sim
  (validated vs sealed counters ±1); DERIVED = arithmetic from labeled inputs;
  PAPER-REPORTED = authors' claims, not dee evidence; UNKNOWN = no evidence.

## QUESTION

Which published predictors are pure prefetch hints for an unmodified
checkpoint, what input state and lead does each need, how much lead is claimed
on DeepSeek-style routing, and can any beat dee's idle-gap bound by making
misses REPEAT (residency) rather than by moving reads earlier?

## WHAT WAS ALREADY KNOWN IN-REPO (with paths)

- **Official lookahead = 0 for all score layers** — `route(L+1)` needs
  `h_{L+1}` = `combine(L)` output; the residual stream is the dependency
  (`research/route-pipeline/OFFICIAL_LOOKAHEAD.md`; DAG spine in
  `research/route-pipeline/CURRENT_DAG.md`). Predictor outputs are never
  official by construction.
- **Hash layers 0–2 give EXACT ids at token start** — `ids = tid2eid[input_ids]`
  (`deepseek_v4_layer_common.py:384-387`; hash branch confirmed
  `deepseek_v4_model.py:360,596`; `research/route-pipeline/HASH_EARLY_STAGING.md`).
  ≤18 records = ≤240,648,192 B/token early-submittable; weights still chain on
  `x @ W^T` (staging early, consumption gated).
- **Idle-gap capacity is the binding resource** — decode row wall 102.7 ms,
  fill 65.1 ms; idle-bank 23.09–37.6 ms/row ⇒ 0.54–1.12 records/row ⇒
  4.3–8.9 GiB early-movable vs 15.24 GiB decode demand; ORACLE cap −11.3 to
  −24.8 s (26–58% of decode fill) (`LEGAL_PREFETCH.md`; sim
  `tools/phase2_legal_prefetch_eval.py`; results
  `research/phase2-legal-prefetch/results/gap_prefetch_sweep.csv`,
  `legal_prefetch_summary.json`). Sealed-counter validation: host 1391/1091 vs
  sealed 1390/1091; device 2284/2161 vs 2285/2159.
- **Prev-token hint measured dead on the miss stream** — same-layer prev-token
  recall 0.360 on demand but **0.0054 on the miss stream**: misses are
  precisely the records that did not just fire (`legal_prefetch_summary.json`
  `demand_structure`; `LEGAL_PREFETCH.md` §3). Consecutive-token same-layer
  reuse 37.5% on the earlier sealed trace
  (`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/CACHE1_ANALYSIS.md` §2).
- **Generic predictor rejected on DSv4** — recall@12 ≈ 0.503; wrong prefetches
  + cache pollution worsened behavior (AGENTS.md §"Route prediction — generic
  family REJECTED"). Ornith-era in-repo predictor design: per-layer MLP on the
  *previous token's* hidden state predicting the N+1/N+2 union
  (`dee.cpp/design.md` §B.1; `modal_step2_train_oracle.py` — cross-TOKEN lead,
  the largest-lead hint shape).
- **Residency already bounded in-request** — plain host LRU within ~2.8pp of
  offline MIN at the ~16 GiB pooled knee; 935/2364 records ever repeat on the
  sealed window; fetch-on-fault MIN ≥ static pin at every budget
  (`research/phase2-regime-c/CONTRACT.md`; AGENTS.md §Working-set research).
  Regime-C cross-request prewarm is a deployment contract DEFERRED to Phase 5.
- **DSv4 router semantics (for hint-fidelity)** — score layers compute
  `topk(sqrt(softplus(x @ W^T)) + gate_bias)`; bias is a static checkpoint
  tensor (`layers.{L}.ffn.gate.bias`; `deepseek_v4_support.py:428-430,579-586`);
  weights come from un-biased scores × route_scale 1.5
  (`deepseek_v4_layer_common.py:374-398`). Gate weights are dense mmap/resident
  tensors, not bank records (`LEGAL_PREFETCH.md` §5) — any hint that applies a
  checkpoint gate to an earlier hidden state is computable for free.
- **Bank ceiling** — 0.29–0.37 GiB/s, ~96% device busy, per-request service
  ~35.4 ms/miss slope / 57.4 ms mean in-batch (`STORAGE_VERDICT.md`;
  `LEGAL_PREFETCH.md` §"binding constraint").
- **Sealed-evidence anchors** — 16/16 tokens + sealed text; route journal
  final chain `d8539b6e7c61d18820ccdcc17e492168a4132186f021ba30ca182581c5fcc75e`;
  v50 journal sha `665aac3e…ae1` (`research/phase2-t4/EVIDENCE_SOURCES.md`).

## SOURCES READ

In-repo: `AGENTS.md`; `research/route-pipeline/{OFFICIAL_LOOKAHEAD.md,
LEGAL_OVERLAP.md, HASH_EARLY_STAGING.md, CURRENT_DAG.md, STORAGE_VERDICT.md,
FINAL_SUMMARY.md}`; `research/phase2-legal-prefetch/{LEGAL_PREFETCH.md,
results/legal_prefetch_summary.json, results/gap_prefetch_sweep.csv}`;
`tools/phase2_legal_prefetch_eval.py`; `research/phase2-regime-c/CONTRACT.md`;
`research/phase2-t4/EVIDENCE_SOURCES.md`; `TIER_REPLAY_VALIDATION.md`;
`dee.cpp/design.md`; `modal_step2_train_oracle.py`;
`dee.cpp/scripts/deepseek_v4_layer_common.py`;
`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/CACHE1_ANALYSIS.md`;
`dee.cpp/benchmark_reports/milestone-3/CACHE_POLICY_RESEARCH.md`;
sibling tracks `research/prior-art/r06`, `r08`, `r09` (format + clause set).

External (all PAPER-REPORTED): the briefing survey — "A Survey on Inference
Optimization Techniques for MoE Models", arXiv:2412.14219 / ACM TALLIP
10.1145/3794845, repo github.com/MoE-Inf/awesome-moe-inference (its
prefetch paragraph: DyNN-Offload pilot model [their ref 144]; MoE-Infinity
request-level frequency prefetch "across multiple layers" [201]).
Per-mechanism: 2312.17238 (Eliseev & Mazur / "mixtral-offloading");
2401.14361 (MoE-Infinity); 2502.12224 (Fate); 2603.19289 ("Speculating
Experts", axonn-ai/yalis); 2608.11688 (APEX); 2511.10676 (pre-attention
linear predictors, ETH); 2410.22134 (ProMoE); 2410.17954 (ExpertFlow-A,
RPP); 2510.26730 (ExpertFlow-B, adaptive horizon); 2509.07379
(DuoServe-MoE); 2508.17137 (MoE-Beyond); 2310.18859 (SiDA-MoE);
2308.14352 (EdgeMoE); 2308.12066 (Pre-gated MoE); 2408.10284 (AdapMoE);
2411.01433 (HOBBIT); 2504.05897 (HybriMoE); 2509.23638 (PreScope);
2502.05370 (FineMoE); 2510.12357 (MoBiLE); 2503.04398 (s-MoE/Speculative
MoE); 2510.10302 (SP-MoE); 2511.14102 (MoE-SpeQ); 2603.09983 (MoE-SpAc);
2604.10152 (SpecMoE, self-assisted SD); 2606.15453 (ST-MoE); 2210.17223
(Lina); 2511.05814 (caching/prefetching in-depth analysis); 2402.07033
(Fiddler); 2505.17639 (PreMoE); 10.1109/hpca57654.2024.00066 (DyNN-Offload);
10.1145/3779212.3790187 (MoE-APEX); github.com/andreyivan4enkov/
moe-orbit-prefetch (alpha-grade prototype, no arXiv).
Naming collision flagged for R12: R8's "SpecMoE" = Eliseev & Mazur's
speculative loading inside 2312.17238; a DISTINCT newer paper "SpecMoE:
Self-Assisted Speculative Decoding" = 2604.10152 (Ranggi Hwang co-author).

## FINDINGS (numbered, tier-labeled)

**F1. The hint/router line is one sentence and the literature splits cleanly
on it.** A predictor is exact-safe iff its output only schedules data
movement; it becomes a ROUTING violation the moment a predicted set executes,
a retrained gate replaces the checkpoint gate, or logits are biased
(repo: `LEGAL_PREFETCH.md` §Legality; clause set per
`r08-approx-methods.md` §1). Every mechanism below is tagged on *that* axis;
performance numbers are PAPER-REPORTED unless labeled otherwise.

**F2. Input-state taxonomy — six classes, all committed-state legal in dee.**

| Class | Input needed | When it exists in dee | Members |
|---|---|---|---|
| C1 token id / sequence | `input_ids` (+ maybe embeddings) | token start (decode); prompt arrival (prefill) | hash layers (exact, in-checkpoint), SiDA LSTM hash, ExpertFlow-A RPP, FineMoE semantic hints, moe-orbit |
| C2 pre-attention hidden | `h_L` before attention | layer L row start | APEX prefetch router, 2511.10676 2-linear predictors |
| C3 gate input (post-attn hidden) | `x_L` (= router input of layer L) | at `route(L)` | mixtral-offloading, Fate, AdapMoE, HOBBIT, Speculating-Experts qHS, ProMoE, dee's Ornith oracle |
| C4 committed routing history | expert ids of already-routed layers/tokens | continuously, zero extra compute | MoE-Infinity pEAM/EAM, DuoServe (popularity+affinity+path), Lina, dee's prevtok baseline |
| C5 draft-model state | draft activations / drafted token ids | only if dee runs a draft head | SP-MoE, MoE-SpeQ, MoE-SpAc, SpecMoE-2604.10152 |
| C6 offline statistics | calibration-time activation data | before deployment | EdgeMoE path statistics, PreMoE PEU task patterns, Speculating-Experts default vectors, ProMoE predictor training |

C1–C4 are computable from state dee already has or could capture; C5 needs a
draft mechanism dee does not run (mtp.{0,1,2} buckets exist but unused — the
sealed journal has zero mtp rows, `LEGAL_PREFETCH.md` §6); C6 needs a
calibration pass (legal: predictors are auxiliary, never authoritative).

**F3. The zero-training "cross-layer gate" family — strongest exact-safe
candidate.** Fate (2502.12224): apply layer L+1's *own checkpoint gate* to
layer L's gate input — residual connections keep adjacent gate inputs
similar, so `gate_{L+1}(x_L)` approximates `route(L+1)`; no trained
parameters at all. Same mechanism: mixtral-offloading (2312.17238 §3.2,
"apply the next-layer's gating function to the current layer's hidden
states"); AdapMoE's prefetch arm (2408.10284 §4.3, activation-similarity);
HOBBIT's prefetch arm (2411.01433); Speculating-Experts (2603.19289)
formalizes it as the quasi-hidden state `q_l = LN_{l+1}(d_l + r_l)` adding a
calibrated "default vector" `d_l` (mean MoE-block output, Panda et al. 2025
per that paper). PAPER-REPORTED accuracy: Fate ~97.15% prefetch accuracy at
75th-percentile confidence gating, evaluated on **DeepseekMoE-16B (64 routed
experts, top-6) and Qwen1.5-MoE**; Speculating-Experts shows qLS > baseline
cosine similarity and recall@k' gains on GPT-OSS + Qwen3-30B-A3B, −14% TPOT
vs on-demand loading in prefetch mode (its *speculative-execution* arm is
ROUTING-violating — execute only the hinted set). Lead: exactly ~1 layer.
dee mapping: on DSv4 the hint = `topk'(sqrt(softplus(x_L @ W_{L+1}^T)) +
bias_{L+1})` with k' > 6 for coverage — checkpoint weights, monotone
nonlinearity + post-nonlinearity bias (the bias is applied AFTER
sqrtsoftplus, so raw-logit topk is NOT equivalent —
`deepseek_v4_layer_common.py:374-393`). Zero training, zero new params.

**F4. Same-layer pre-attention predictors — intra-layer lead.** APEX
(2608.11688): a learned "prefetch router" reads the **pre-attention hidden
state** and ranks candidates for the *current* layer, with a learned
confidence model deciding how many to fetch; >99% "overlap accuracy",
correctness-preserving mode is exact by construction (stall-free mode is
approx); evaluated on **DeepSeek-V2-Lite-16B** + Granite + Phi-mini-MoE;
−26% per-token latency correctness-preserving. 2511.10676: two linear
functions + ranking-aware loss on pre-attention activations (softmax/LN are
ranking-preserving → linear ranking match); 93.03% DSv2-Lite, 94.69%
Qwen3-30B, 97.62% Phi-mini-MoE, ~+15pp over Fate. Lead = the layer's
attention window — shorter than 1-layer but usable on layer 0 where
cross-layer tricks have no predecessor.

**F5. Learned per-layer predictors — ~1–2 layer lead, training required.**
ProMoE (2410.22134): per-layer ~2M-param MLP on the layer input vector;
**stride prefetching predicts layer i+1 while processing layer i−1** (~2-layer
effective lead) at ~5% accuracy cost; 84.7% average prediction accuracy vs
token-based 58.3% / skip-based 66.9%; evaluated on **DeepseekMoE-16B and
DeepSeek-V2-Lite (both 64 experts, top-6)**, Qwen1.5/2-MoE, Mixtral-8x7B;
GoodPred = accuracy × fetchable-fraction — the composite dee's sim already
implements operationally (completed/abandoned/used columns in
`gap_prefetch_sweep.csv`). Precedents at neuron granularity: PowerInfer
(2312.12456) / DejaVu (2310.17157) per-layer MLPs predict MLP sparsity from
hidden state. dee's own prior art: the Ornith oracle (`dee.cpp/design.md`
§B.1; `modal_step2_train_oracle.py`) is a per-layer MLP on **token N's**
hidden state predicting the **union of tokens N+1,N+2's** experts —
cross-TOKEN lead (~4.4 s at 0.21 tok/s decode), the largest-lead hint shape
anyone has tried in-repo; its DSv4 instantiation is the rejected generic
predictor (recall@12 0.503, AGENTS.md).

**F6. Routing-history predictors — cheapest legal input, measured weak on
dee's miss stream.** MoE-Infinity (2401.14361): per-request Expert Activation
Matrix (L×E); after each layer's routing it emits a predicted EAM —
activation likelihood for future layers + reuse likelihood — driving both
prefetch AND cache replacement; 3.1–16.7× vs vLLM/Ollama/DeepSpeed on
DeepSeek-MoE + Mixtral. DuoServe-MoE (2509.07379): popularity +
inter-layer affinity + activation path → next-layer classifier, "without
changing model architecture or accuracy". Lina (2210.17223): adjacent-layer
selection patterns → expert-popularity estimate for device scheduling.
EdgeMoE (2308.14352): offline activation-path statistics (power-law paths)
→ statistical preload. dee's direct measurement of this family's floor:
prev-token same-layer recall = 0.360 demand / **0.0054 miss-stream**
(`legal_prefetch_summary.json` — MEASURED on sealed journal); on Mixtral the
adjacent-token same-expert rate is ~30% vs 12.5% random (2511.05814 §3 —
consistent with dee's 37.5% consecutive reuse, CACHE1_ANALYSIS §2). The
family's information ceiling: a miss is by definition a record NOT recently
used, so history-only predictors aim at exactly the wrong targets on a
cold-heavy trace (dee) but work on temporal-locality-heavy workloads.

**F7. Whole-sequence predictors — the only class matching dee's token-start
hash lead.** SiDA-MoE (2310.18859): offline-trained LSTM + sparsemax
attention over the input batch predicts per-token per-layer activated sets
(up to 99% top-3, Switch/NLLB — not DeepSeek); as published it *replaces the
router* and offloads predicted-inactive experts = ROUTING violation; as a
hint it predicts ALL layers at batch arrival. ExpertFlow-A (2410.17954):
T5-style encoder-decoder RPP maps the full input sequence → (B,S,L,E)
routing matrix before the first MoE layer runs; 91.96% cache hit, −93.72%
memory, ≤10× throughput (Mixtral-class). MoE-Beyond (2508.17137):
transformer over 66M recorded activation traces of **DeepSeek-V2-Lite**,
97.5% accuracy/86.6% F1; simulated cache hit 17%→72% at 10% cache.
FineMoE (2502.05370): prompt semantic hints + selection patterns, serving.
moe-orbit-prefetch (GitHub alpha): embedding/residual `h` → per-layer expert
law on DSv2-Lite + GigaChat; honest negatives reported (can fail vs
freq/LRU/SGD baselines). All need training data dee does not have; all are
hint-legal if the router stays authoritative.

**F8. Draft-lookahead predictors — multi-token lead, needs SD machinery.**
SP-MoE (2510.10302): during SD *drafting*, draft-model attention outputs ×
*target-model* gates predict verification-stage experts; cutoff-layer bounds
depth; 1.07–3.5× TPOT. MoE-SpeQ (2511.14102): a quantized draft model's
expert sequence predicts the target's future-token experts; 2.34× Phi-MoE.
MoE-SpAc (2603.09983): SD explicitly repurposed as a "lookahead sensor" for
memory; +42% TPS vs SD baselines. SpecMoE (2604.10152): self-assisted SD
(target model drafts itself), 4.30× throughput. dee relevance: the ONLY
published class with >1-token lead; all are hint-legal in principle (the
predicted token/expert stream only prefetches; SD verification keeps the
target router authoritative). dee has mtp.{0,1,2} draft buckets (768 bank
records) — an in-checkpoint draft source whose predicted token ids would
yield *next-token hash-layer ids early*, the one legal crack in the
"nothing of token t+1 exists before sample(t)" wall
(`LEGAL_OVERLAP.md` §E) — as a HINT, never official. Untested; MTP records
live in the same bank they would prefetch (self-defeating wrinkle); SD is
out of current scope. Phase-5 note.

**F9. Nobody claims >2-layer lead on fine-grained DeepSeek-style routing
without a draft token or a whole-sequence trained model.** Published
DeepSeek-adjacent evidence concentrates on 64-expert top-6 models
(DeepseekMoE-16B, DSv2-Lite): Fate ~97% (1-layer), APEX >99% overlap
(intra-layer), pre-attention 93.03% (intra-layer), ProMoE 84.7% (~2-layer),
MoE-Beyond 97.5% (whole-sequence, trained on 66M DSv2-Lite traces).
**Zero published results on 256-expert top-6 sqrtsoftplus+bias routing**
(DSv3/V4 geometry). Two warnings align: MoBiLE (2510.12357 intro) —
predictor approaches "show diminished effectiveness on recent MoE models
with fine-grained expert segmentation"; dee's own generic-predictor
recall@12 = 0.503 on DSv4 (MEASURED, rejected). Finer segmentation (256 vs
64 choices, top-6) is exactly where predictor recall should degrade.

**F10. Residency: the only mechanism class that can beat the idle-gap bound —
and predictors don't create reuse, they only steer placement.** Moving reads
earlier is capped by gap capacity (0.54–1.12 records/row, DERIVED from
measured gaps × bank BW). The literature's gap-bound-beating wins all come
from converting future misses into hits — i.e., *which records are resident*,
not *when reads issue*: MoE-Infinity's pEAM-guided replacement; SiDA's
predicted resident set; ExpertFlow-A's predictive cache + token scheduling
(groups same-route tokens so one fetch serves many — an amortization, not a
prefetch); MoE-Beyond's predicted-residency 17%→72% sim hit rate; PreMoE
(2505.17639) task-adaptive retrieval — load a task-keyed expert manifest
(pruning side = IDENTITY/BYTES violation; the *retrieval* side is
regime-C-shaped prewarm, and its finding of strong task-level expert
specialization on DeepSeek-R1 is PAPER-REPORTED evidence FOR the cross-request
rank stability dee's regime-C needs). Cache-aware-routing papers create
reuse by changing routing (Cache-Prior 2412.00099, ReMoE, BuddyMoE, SMoE —
all ROUTING, R8-covered; "Cacheable by Design" 2608.18261 trained-for-
locality failed its perplexity gate — the honest negative). dee's sealed
window: 935/2364 records ever repeat; host LRU already within 2.8pp of
offline MIN at the 16 GiB knee → in-request placement headroom ~exhausted;
residency prize is cross-request (regime-C, DEFERRED Phase 5).

**F11. dee holds an exact source the literature lacks.** tid2eid hash layers
0–2 give ≤18 exact ids at token start with zero prediction error — the only
*exact* early source anywhere in this survey; measured conversion is
0.54–1.60 s/response because the window, not the knowledge, binds
(`LEGAL_PREFETCH.md` §2; `HASH_EARLY_STAGING.md`). Every learned/statistical
predictor above is a strictly worse-input substitute for this trick applied
to score layers — and none is exact.

**F12. Timeline translation kills most published speedup claims at dee's
bank.** Papers assume host→device or fast-SSD fills of ~1–3 ms/expert
(PCIe/NVMe); a 1-layer lead then hides several experts. On the sealed bank a
12.75 MiB record needs 34–44 ms (DERIVED: 12.75 MiB / 0.29–0.37 GiB/s;
MEASURED-anchored W1 slope 35.4 ms/miss). A hint for row L+1 issued at
`route(L)` can use only row L's idle gap (~23–37.6 ms ⇒ ≤~1.1 records) —
issuing during fill is wall-neutral at best on a saturated stream
(`LEGAL_PREFETCH.md`; the 4-lane pool measured +23% WORSE,
`SERIALIZATION_VERDICT.md`). Consequence: per-row miss mean 1.898 cannot be
fully covered by ANY ≤2-layer-lead hint on this bank even at 100%
precision; the oracle bound (−11.3..−24.8 s decode fill, SIMULATED) already
caps all of them; at realistic ~0.85–0.97 precision the conversion is
low-single-digit seconds. On a >0.6–0.7 GiB/s bank the ordering changes
(gap capacity scales linearly; at 2.9 GiB/s ~9 records/gap — `LEGAL_PREFETCH`
§Sensitivity) and F3/F4-class hints become real.

## PRIZE MODEL vs dee timelines

dee decode row (MEASURED live, sealed profile): wall 102.7 ms, fill 65.1 ms,
idle-bank 23.09–37.6 ms; miss distribution per row mean 1.898 (0:111, 1:179,
2:161, 3:98, 4:62, 5:23, 6:11 of 645 rows — `legal_prefetch_summary.json`);
record service 34–44 ms; response decode fill 42.84 s; response fill 86.3 s.

| Hint class | Earliest legal issue point | Usable window before demand | Records coverable @0.29–0.37 GiB/s | Best-case response win | Verdict ceiling |
|---|---|---|---|---|---|
| C4 routing-history (prevtok/EAM) | `route(L)` of current row | row L idle gap 23–37.6 ms | ≤1.1/row | SIMULATED 0.25–0.74 s (measured recall 0.0054 on misses) | NO-GO on this bank |
| C2 pre-attention same-layer | layer L row start | layer L attention portion of non-fill window | ≤~1/row | DERIVED ≲ hash-scale (~1 s) | weak |
| C3 cross-layer gate (Fate/qHS) | `x_L` known ≈ route(L) | row L gap (+ row L+1's gap if it fires early) | ≤~1.1/row | DERIVED ≲ few s at ≤97% precision | DEFER (bank-bound) |
| C3 stride (ProMoE ~2 layers) | during row L−1 | rows L−1,L gaps | ≤~2 records ahead | DERIVED ≲ few s; capacity-bound not lead-bound | DEFER |
| C1 exact hash ids (in-model) | token start / token boundary | gaps before rows (t,0..2) + post-(t,42) | ≤18 records but window ≤3 rows | SIMULATED 0.54–1.60 s | DEFER-until-bank-move (existing verdict) |
| C1/C7 whole-sequence learned | token/sequence start | all 43 rows' gaps | gap-capacity-bound ≤1.1/row | DERIVED ≤ oracle 11.3–24.8 s, ×precision | needs training + DSv4 eval first |
| C5 draft/MTP-hinted | draft completes before verify | whole verify forward's gaps | multi-token set | UNKNOWN (no dee draft path; mtp buckets unused) | Phase-5 idea only |
| ORACLE bound (not legal) | — | every idle gap | 321–709 preads | SIMULATED −11.3..−24.8 s (26–58% decode fill) | the cap for ALL of the above |
| Residency (placement-steering) | before demand, any time | converts misses→hits; no gap needed | bounded by capacity + true reuse | in-request ≈ exhausted (LRU ≈ MIN−2.8pp @16 GiB; 935/2364 repeat) | cross-request = regime-C, DEFERRED |

## DISPOSITION per mechanism

| Mechanism | Ref | Input class | Lead | DSv-style evidence | Exact-safe? | dee disposition |
|---|---|---|---|---|---|---|
| Hash-layer exact ids | in-model (`tid2eid`) | C1 | token-start, exact | DSv4 itself | EXACT (not a predictor) | DEFER — 0.54–1.60 s; reopen with bank move |
| mixtral-offloading spec-load | 2312.17238 | C3 | ~1 layer | Mixtral only | HINT-legal | Candidate hint; bank-bound |
| Fate cross-layer gate | 2502.12224 | C3 | ~1 layer | DeepseekMoE-16B ~97% | HINT-legal (separate its INT2/4 side) | **Cheapest new candidate** — zero training; needs DSv4 recall eval |
| Speculating-Experts qHS | 2603.19289 | C3+C6 | ~1 layer | GPT-OSS, Qwen3-30B | prefetch arm HINT-legal; exec arm ROUTING | qHS refinement of Fate; same gate |
| APEX prefetch router | 2608.11688 | C2 | intra-layer | DSv2-Lite >99% overlap | HINT-legal (correctness-preserving mode) | Needs trained router per layer; DSv4 recall unknown |
| Pre-attention linear predictors | 2511.10676 | C2 | intra-layer | DSv2-Lite 93.03% | HINT-legal | Cheap to train; same eval gate |
| ProMoE stride MLP | 2410.22134 | C3 | ~2 layers | DSv2-Lite + DeepseekMoE 84.7% | HINT-legal | GoodPred framework reusable; stride adds little under capacity bound |
| MoE-Infinity pEAM/EAM | 2401.14361 | C4 | multi-layer (weak) | DeepSeek-MoE, Mixtral | HINT-legal | EAM replayable on sealed journal today; expect low on miss stream |
| DuoServe-MoE | 2509.07379 | C4 | 1 layer | serving QoS focus | HINT-legal | affinity/path tables reusable for placement hints |
| EdgeMoE statistical preload | 2308.14352 | C6 | static | Switch-family | HINT-legal (its bitwidth side = PRECISION) | ≈dee freq-backfill, measured weak |
| MoE-Beyond | 2508.17137 | C1+C4 traces | whole-sequence | DSv2-Lite 97.5% | HINT-legal | needs 66M-trace-scale data; residency-shaped win |
| ExpertFlow-A RPP | 2410.17954 | C1 | all layers pre-first-MoE | Mixtral-class | HINT-legal | heaviest training; residency+token-scheduling shape |
| ExpertFlow-B adaptive horizon | 2510.26730 | C3+pregating fusion | adaptive depth | A6000/H20/910B | HINT mostly — its cache-aware token *scheduling* is legal scheduling, but R8 flags a cached-prediction fast path: must verify which path computes output | verify-before-use |
| SiDA-MoE hash net | 2310.18859 | C1 | all layers | Switch/NLLB 99% top-3 | as-published ROUTING (replaces router); hint-legal as hint | salvage = hint only; all-layer-at-once shape matches hash trick |
| ST-MoE spatio-temporal | 2606.15453 | C3/C4 | adjacent-layer + consecutive-token | Qwen/DeepSeek claims | HINT-legal (claims routing preserved) | correlation claim supports C3/C4 family |
| PreScope LLaPor + PreSched | 2509.23638 | layer-group features | cross-layer | commodity-GPU offload | HINT-legal; its CPU-GPU co-execution is R6 territory | layer-group insight (input/middle/output differ) useful |
| FineMoE | 2502.05370 | C1 semantics | prompt-time | serving | HINT-legal | semantic-hint = regime-C adjacent |
| HybriMoE | 2504.05897 | scores + activation stats | inter-layer | DSv2-Lite + 2 more | HINT-legal; CPU-exec side = R6/numerics | impact-driven scheduling = prefetch-priority metric |
| SP-MoE / MoE-SpeQ / MoE-SpAc / SpecMoE-2604 | 2510.10302 / 2511.14102 / 2603.09983 / 2604.10152 | C5 draft | multi-token | various incl. MoE SD | HINT-legal only as hints; SD itself out of scope | Phase-5 idea: MTP buckets as in-checkpoint draft |
| s-MoE (Speculative MoE) | 2503.04398 | partial activations | within-EP | distributed EP/TP | claims lossless comm-only | different problem (all-to-all), note only |
| Klotski | ASPLOS'25 | activation paths | batch-level | serving | legal (scheduling) | dee-serve note |
| PreMoE TAER | 2505.17639 | C6 task manifests | request-level | DS-R1 task specialization | retrieval side = legal prewarm; pruning side = IDENTITY/BYTES | evidence FOR regime-C rank stability; defer with it |
| Pre-gated MoE | 2308.12066 | retrained gate_N→N+1 | 1 layer | Switch | NOT exact (ROUTING) | R8-covered; predictor-as-hint salvage only |
| Cache-aware routing family | 2412.00099 etc. | router change | — | various | NOT exact (ROUTING/IDENTITY) | R8-covered; residency-by-contract-violation |
| MoE-APEX / HOBBIT-precision / SliceMoE / DynaExq / MoBiLE | 10.1145/3779212.3790187, 2411.01433, 2512.12990, 2511.15015, 2510.12357 | mixed | ~1 layer | edge models | NOT exact (PRECISION/BYTES + adaptive-k for MoBiLE) | R8-covered; prefetch arms salvageable as hints |
| DyNN-Offload pilot | HPCA'24 | input sample | op-level | AlphaFold (DyNN) | concept only | historical precedent |
| moe-orbit-prefetch | GitHub alpha | C1 embedding/residual | all layers (claimed) | DSv2-Lite + GigaChat | HINT-legal | alpha evidence; watch only |
| dee prevtok + backfill | in-repo sim | C4 | 1–2 rows | DSv4 sealed | HINT-legal, MEASURED | NO-GO (0.0054 miss recall) — closed |
| dee generic predictor | AGENTS.md | C3 (per-layer MLP) | token-level | DSv4 sealed | HINT-legal, MEASURED | REJECTED (recall@12 0.503) — closed |
| dee Ornith oracle design | design.md §B.1 | C3 cross-TOKEN | ~1 token | Ornith-era only | HINT-legal | ancestor of the rejected family |

## WHAT WOULD FALSIFY THIS

- **Any hint source whose sim-replay beats the hash arm materially.** Extend
  `tools/phase2_legal_prefetch_eval.py` with a candidate predictor and require
  `decode_preads` below the hash arm's 1182–1210 range on the sealed journal
  with sealed-counter validation intact. A negative result at ~97% synthetic
  precision would show the bound is capacity-, not knowledge-limited — already
  strongly indicated by oracle ≈ 4× the best legal source.
- **Cross-layer-gate recall collapse on DSv4.** If `topk'(sqrtsoftplus(x_L @
  W_{L+1}^T) + b_{L+1})` recall@k' lands near the generic predictor's 0.503
  rather than Fate's ~0.97, the zero-training family dies on DSv4's 256-expert
  fine-grained routing — needs hidden-state captures (below).
- **Residency claims.** A multi-request journal showing rank-stable hot sets
  would reopen regime-C; conversely a journal where reuse stays ~935/2364-class
  per response closes predictor-steered residency in-request for good.
- **A published mechanism that moves >~1.1 records/row on a ~0.3 GiB/s link
  without prefill/cross-request idle time** would break the capacity model —
  none observed; all reviewed systems assume ≥PCIe-class links.
- **Contract side:** any "prefetch" mechanism found to let predicted sets
  execute (SiDA, Speculating-Experts exec arm, ExpertFlow fast path, all
  cache-aware routing) is not a counterexample — it reclassifies into R8's
  approx family, which is exactly where they already sit.

## OPEN UNKNOWNS + cheapest experiment to close each

1. **Cross-layer-gate (Fate/qHS) recall on DSv4 score layers** — UNKNOWN.
   Cheapest: capture gate inputs `x_L` (16 KiB/layer/token) in the next
   profile/capture run or via the host reference path for a few tokens; then
   local CPU computes `sqrtsoftplus(x_L W_{L+1}^T)+b_{L+1}` topk' vs the
   sealed journal's true ids. Zero remote spend if piggybacked; no engine
   change — a telemetry hook.
2. **Predictor-precision breakeven on the miss stream** — UNKNOWN at what p
   a hint beats hash's 42-pread best. Cheapest: add a parameterized-precision
   synthetic predictor + the EAM/pEAM (routing-history) predictor to
   `phase2_legal_prefetch_eval.py`; pure local CPU on the sealed journal;
   must keep sealed-counter validation (1391/1091, 2284/2161 within ±1).
   EAM needs routes only — runnable TODAY.
3. **Learned per-layer MLP on DSv4 (ProMoE/APEX shape)** — UNKNOWN; the
   sealed rejection was a *generic* predictor; trained per-layer may differ
   (Edge0 precedent, AGENTS.md). Cheapest: needs (1) gate-input/hidden-state
   captures (same hook as item 1) + (2) offline MLP training on CPU — local,
   hours-scale; gate on item-1 recall first (if cross-gate ≈0.5, don't
   bother training).
4. **MTP-as-hint channel** — UNKNOWN entirely: mtp.{0,1,2} draft quality
   (DSv3-class MTP acceptance is PAPER-REPORTED ~85–90%; DSv4 UNKNOWN),
   whether predicted t+1 ids are early enough to matter, and whether the mtp
   buckets' own bank-read cost eats the win. Cheapest: host-side reference
   run of the MTP head on the sealed prompt (CPU, no GPU needed) measuring
   predicted-id acceptance vs sealed tokens; then sim the hint.
5. **Residency at response scale** — UNKNOWN beyond 16 tokens: the sealed
   window's 935/2364 repeat structure may be prompt-specific. Cheapest:
   multi-prompt journal from an existing harness run (CPU-side replay of
   recorded routes; the Phase-5 multi-request artifact regime-C already
   flags) — then recompute MIN/LRU gap.
6. **DSv4 bias/router detail for hint fidelity** — partially UNKNOWN: whether
   `gate.bias` is static across inference in dee's engine (evidence: it is a
   checkpoint tensor loaded via `deepseek_v4_support.py:428-430,579-586`;
   no update path found in `dee.cpp/src` — grep-clean). If the official
   stack ever updates bias online, hint math must snapshot it. Cheapest:
   confirm in `official-source/inference/model.py` (vendored reference in
   `benchmark_reports/deepseek-v4-flash-0731-t4/official-source/`).
7. **Pre-attention (C2) input availability in the engine** — UNKNOWN whether
   the engine's current seam can hand a predictor `h_L` before attention
   without a new hook (gate input `x_L` is the natural C3 tap —
   `route_d2h`/`router_select` boundary). Cheapest: read
   `dee.cpp/src/engine.cpp` layer loop + `layer_candidate.py:397-493` to
   name the insertion seam — a read-only audit.

## Bottom line

The prediction literature is rich in hint shapes and poor in dee-applicable
wins: every in-request prefetch source is capped by the measured
idle-gap×bandwidth product (≤~1.1 records/row) that no predictor can change,
and the only class that escapes it — predictor-steered residency — has its
in-request prize already ~exhausted by measured LRU≈MIN and its cross-request
prize contractually deferred to Phase 5. The single most attractive new
candidate is the zero-training cross-layer-gate hint (Fate/qHS): free to
compute from checkpoint weights, ~1-layer lead, ~97% claimed on 64-expert
DeepSeek-style routing — and falsifiable locally the moment gate-input
captures exist. On this bank it cannot beat ~2 s/response; on a >0.7 GiB/s
bank it and the exact hash channel reopen together.
