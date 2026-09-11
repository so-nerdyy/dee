# R2 — Cross-Layer Gate feasibility for DeepSeek-V4-Flash

Track: R2 (prior-art + mechanism audit; analysis only — no code changed).
Branch: `research/prior-art-r02` @ `dc78dc4`, worktree `.freebuff/wt/r02`.
Question: can an earlier committed hidden state feed a *future* gate —
evaluate gate(L+k)'s real checkpoint weights on layer L's state — and is
that legal, useful, and measurable on the sealed T4 evidence base?

**Tier labels used throughout:** `T0` official DeepSeek source (authoritative
semantics) · `T1` dee repo source-grounded fact · `T2` sealed journal/replay
evidence · `T3` derived/modeled arithmetic · `T4` live T4 measurement ·
`EXT` external paper-reported (never dee acceptance evidence).

## QUESTION

1. What state exists before routing layer N+1 commits?
2. Can any earlier hidden state serve as the official input to gate(N+1)
   without changing semantics? (If not: can it serve as a prefetch HINT?)
3. What is the legal evaluation distance per gate?
4. What does a router evaluation cost on T4 (FLOPs + launch/copy overhead)?
5. Are router weights device-resident? Do they consume expert-bank bytes?
6. Would cross-layer predictions overlap the sealed route journal's demand
   and, specifically, its *miss* stream? Can gate-input correlation be
   computed from existing artifacts?
7. Hash layers 0–2 vs score layers 3–42 — how do the answers differ?
8. If hidden-state traces are missing, what is the smallest evidence-only,
   provably inert instrumentation?

## ALREADY-KNOWN (with paths)

- `research/route-pipeline/OFFICIAL_LOOKAHEAD.md` — score-layer official
  lookahead = 0 at leads +1/+2/+4/+8 (all `REQUIRES_CURRENT_LAYER_OUTPUT`);
  hash ids `tid2eid[input_ids]` exact at token start; token t+1 impossible
  before `sample(t)`; prefill knows all prompt ids.
- `research/route-pipeline/CURRENT_DAG.md` — serial spine
  `combine(L) → route(L+1)`; per-layer dependency enumeration.
- `research/route-pipeline/LEGAL_OVERLAP.md` — same-layer legal work
  (shared expert, batch submission, transfer setup); t+1 rule in §E.
- `research/route-pipeline/HASH_EARLY_STAGING.md` — ≤18 exact records/token
  (3 layers × top-6) submittable early; weights still chain on x@Wᵀ.
- `research/route-pipeline/STORAGE_VERDICT.md` — bank ceiling 0.29–0.37
  GiB/s (measured, 96% in-batch busy); ≤6-expert-per-batch dependency
  starvation is structural.
- `research/route-pipeline/LIVE_PROFILE_RESULTS.md` — decode 66.233 s /
  645 rows; fill_wait 41.99 s; combine 0.139 s (0.21 ms/row, host loop);
  route_d2h 15 ms total (p50 0.022 ms); native_output_sync p50 7.99 ms/row;
  whole-run device-busy 6.89 s; "no legal host work exists" for sync
  windows because next routes unknown.
- `research/phase2-legal-prefetch/LEGAL_PREFETCH.md` — idle-bank pool
  0.54–1.12 records/row (346–720 slots/response); oracle bound
  −11.3 to −24.8 s decode fill; hash source 0.54–1.60 s; prev-token hint
  miss-stream recall 0.0054 (NO-GO); **item 7 already verdicts
  "cross-layer score-layer pipelining — ILLEGAL — permanent"** (for
  *official* routes); idle-gap admission engine DEFERRED, reopen iff bank
  >~0.6–0.7 GiB/s or a trained predictor with real miss-stream precision
  arrives (§10).
- `tools/phase2_legal_prefetch_eval.py` + `research/phase2-legal-prefetch/
  results/` (journal sha256 `665aac3e…ae1`; `miss_stream.csv`;
  `gap_prefetch_sweep.csv`; `legal_prefetch_summary.json`).
- `AGENTS.md` — exactness contract: prediction may drive prefetch HINTS
  only; native router authoritative; generic route-prediction family
  REJECTED (recall@12 ≈ 0.503); "Edge0 shows a trained per-layer predictor
  could be much stronger — future speculative-prefetch only."

## SOURCES

Official (T0):
`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/official-source/inference/model.py`
— `Gate` :551-589 (weight [256,4096] :562; tid2eid int32 [vocab,6] :564;
bias :567; forward :569-589); `MoE.forward` :634-649 (gate call :637);
`Block.forward` :695-707 (hc_pre/attn_norm/attn/hc_post :697-700,
hc_pre/ffn_norm :703-704, `self.ffn(x, input_ids)` :705, post :706);
`Transformer.forward` :913-926; `sample` :939-946; `config.json`
(n_hash_layers=3, n_routed_experts=256, n_activated_experts=6,
score_func=sqrtsoftplus, route_scale=1.5, compress_ratios alternating
4/128 with 0,0 at {0,1,40,41,42}).

dee implementation (T1):
`dee.cpp/scripts/deepseek_v4_layer_reference.py` — `DeepseekV4Layer.forward`
:595-649 (capture taps: `layer_input` :608, `attn_norm_in` :612,
`attn_norm_out` :614, `attn_hc_out` :619, `ffn_norm_in` :626,
`ffn_norm_out` :628, `self.ffn_fn(x, input_ids, c)` :637, `output` :641;
`c = capture if capture is not None else {}` :600 — taps write
unconditionally into a throwaway dict when capture is off).
`dee.cpp/scripts/deepseek_v4_layer_common.py` — `router_select` :349-398
(GEMM `x.float() @ W.float().T` :374; hash branch :384-388; score topk
:392; weights gather/normalize/scale :394-397).
`dee.cpp/scripts/deepseek_v4_expert_reference.py` — `router_scores`
:132-163.
`dee.cpp/scripts/deepseek_v4_layer_candidate.py` — FFN `__call__` :78-128
(`xf = x.reshape(-1,d).float()` :83; hash `router_select` :88-91; score
`ds7.router_scores` :100-102; `last_route` D2H branch when
`diagnostics or capture` :109-113; `_run_experts` :117/:139; pinned route
D2H copy :460; `current_stream().synchronize()` :467; native
`batch_device` :482-484; `attach_gate`/`attach_hash` :131-136).
`dee.cpp/scripts/deepseek_v4_model.py` — `_move_to_device(w, device)`
:605/:1093-1097 (whole dense tree, incl. gate weights, to the layer's
device at build); device split at layer 22 (`device_for_layer` :720);
`generate()` + `per_step_captures` plumbing :897-974 (`cap_map` :954);
`state_signatures` :979.
`dee.cpp/scripts/deepseek_v4_support.py` — dense tensor enumeration
incl. `ffn.gate.weight` + (`gate.bias`|`gate.tid2eid`) :578-586;
`tid2eid` widened to I64 at load (`deepseek_v4_layer_reference.py:776`).

Engine (T1):
`dee.cpp/src/engine.cpp` — `get_router_weights` :2264-2293 (dense gate
resolve/cache, expects `model.language_model.layers.N.mlp.gate.*` names —
incompatible with DSv4's `layers.N.ffn.gate.*`); `route_topk_batch`
:2295+; real-model integration mode "routes come from Python"
:3597-3606.
`dee.cpp/include/dee/oracle.h` — legacy synthetic-model predictor:
`predict_next(current_layer, hidden, topk, out)` :61-67 = "predict for the
NEXT layer given the hidden state at the current layer's input"; no-op
stub in real-model mode :11-16.
`dee.cpp/include/dee/profiling.h` :316-317 + `src/profiling.cpp` :678-693 —
`oracle_adjacent_topk_overlap` (mean Jaccard of consecutive routed sets)
and `oracle_random_overlap_expectation` exist as metrics.
`modal_step2_train_oracle.py` :131-183 — per-layer 3-layer-MLP oracle
trainer with union_recall@top_k_pred evaluation (dee's own prior
hidden-state→next-layer predictor harness).

Sealed evidence (T2/T4):
`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/v50-evidence-20260829T195940Z/routed_experts.jsonl`
(688 records = 16 forwards × 43 layers × top-6; prefill step 0 has
token_rows=7, decode steps 1–15 have token_rows=1; sha `665aac3e…`),
`generated_checkpoint.jsonl` (engine stats incl.
`host_router_weight_bytes: 0`, `device_router_weight_bytes: 0`),
`research/route-pipeline/evidence-live/per-layer.csv`,
`evidence-live/profile-run-on/stage-profile.json`.

External (EXT):
Fate arXiv:2502.12224 ("Accurate Expert Predictions in MoE Inference via
Cross-Layer Gate" — the namesake mechanism); HOBBIT arXiv:2411.01433
(inter-layer gating-input similarity, 96% top-1 next-layer prediction on
Mixtral-8x7B); Mixtral-Offloading/SpecMoE (Eliseev & Mazur 2023 — apply
next-layer gating to current hidden states); Speculating Experts
arXiv:2603.19289 (parameter-free "quasi-hidden state" from the residual
stream); ST-MoE arXiv:2606.15453 (cross-layer correlation table CCT,
ids→ids); PreScope arXiv:2509.23638 (layer-group correlation structure);
PROBE arXiv:2602.00509 (gate-initialized lookahead/distilled router);
Pre-gated MoE arXiv:2308.12066 (trained pre-gate — model modification).

## FINDINGS

**F1. [T0] The official gate input is strictly the current layer's
post-attention state — official lookahead is 0, permanently.**
`Block.forward` (model.py:695-707) computes, in order:
hc_pre(attn) → attn_norm → attn (KV-cache-dependent) → hc_post →
hc_pre(ffn) → ffn_norm → `self.ffn(x, input_ids)` → hc_post.
The gate therefore consumes
`x_L = ffn_norm(hc_pre_ffn(hc_post(attn(attn_norm(hc_pre_attn(h_L))))))`.
`x_{L+1}` additionally requires `h_{L+1}` = the hc_post output of layer
L's FFN (:706) — i.e., *all six routed experts of layer L must finish*,
plus layer L+1's own attention over the KV cache. Confirms
OFFICIAL_LOOKAHEAD.md + LEGAL_PREFETCH.md item 7 at the source level:
**no reordering or dual-GPU staging crosses this; it is the residual
stream itself, not an implementation detail.** Any earlier state fed to
gate(L+1) computes a *different function* than the official gate — legal
only as a hint whose output can never execute.

**F2. [T0+T1] Exact inventory of committed state at the instant route(N)
commits** (post-sync, layer_candidate.py:467):
- Device, dynamic: `x_N` (ffn_norm_out, [1,1,4096] bf16 ≈ 8 KiB — the
  freshest single-vector state); `h_N` (x_hc [1,1,4,4096] ≈ 32 KiB);
  attention intermediates of layer N; KV caches through layer N;
  scores/ids/weights(N) still on device.
- Host: ids(N) in a 24 B pinned buffer (:447-476); `input_ids(t)`.
- Static, resident: every dense weight including all 43 gate weights.
- **Not existing:** h_{N+1}, x_{N+1}, ids(N+1), weights(N+1), and any
  token-t+1 state.
→ The only lawful inputs to a future-gate hint are committed states of
layers ≤ N (freshest: x_N), input_ids, and static weights — exactly the
input class Fate/MoE-Offloading/Speculating-Experts use.

**F3. [T2] Hash-layer ids verified deterministic in the sealed journal.**
Decode step 3 generated token_id **343** (generated_checkpoint.jsonl
step 3). Its L0/L1/L2 id sets — L0 `[20,13,96,155,113,238]`, L1
`[232,116,213,164,197,105]`, L2 `[96,210,149,231,70,32]` (journal lines
130-132, record_index 129-131) — equal prefill token_row 4's sets at all
three layers **including rank order** (journal line 1-3), i.e., prompt
position 4 is token 343 and ids are a pure function of token_id.
Hash layers need no predictor: ids are exact at token start; but their
routing *weights* still require `x @ Wᵀ` on the current layer input
(model.py:570-588; common.py:384-397), so even hash-layer *consumption*
chains normally.

**F4. [T2] Adjacent-layer same-index id overlap is at chance level**
(spot check of decode steps 1/3/5: L0∩L1 = 0 ids; L1∩L2 = 0; L0∩L2 = 1
of 6; L1∩L2 = 1 of 6 — expectation under independence = 6·6/256 ≈ 0.14
ids ≈ 2.3%). A numeric-id→id "index-aligned" hint carries no signal;
cross-layer information, if any, must come through the hidden state (or a
trained ids→ids co-activation table à la ST-MoE). Full-pass analysis
spec'd in OPEN UNKNOWNS (journal-only, zero spend).

**F5. [T0+T1+T2] Router weights are dense, device-resident, and consume
zero expert-bank bytes.** gate.weight [256,4096] + gate.bias [256]
(score layers) + gate.tid2eid int32 [129280,6] (hash layers) are ordinary
dense checkpoint tensors (`deepseek_v4_support.py:578-586`). The whole
dense tree is moved to each layer's device at build
(`deepseek_v4_model.py:591-605`, `_move_to_device` :1093-1097); the
candidate FFN holds them via `attach_gate`/`attach_hash`
(layer_candidate.py:131-136). Bytes [T3 derived]: 43 × 2 MiB FP16 gate_w
+ 40 × 512 B bias + 3 × 5.91 MiB tid2eid (I64-widened; 2.96 MiB as stored
int32) ≈ **~104 MiB device-resident, 0 bank records**. Sealed run
counters confirm `host_router_weight_bytes=0`,
`device_router_weight_bytes=0` — the legacy engine router
(engine.cpp:2264-2293) is unused in DSv4 (name scheme + real-model mode
:3597-3606). A hint evaluator needs no weight staging of any kind; every
gate is already on-device on the correct side of the 21/22 split
(device_for_layer :720; DUAL_GPU_PIPELINE.md).

**F6. [T3 anchored on T4] Router evaluation cost on T4 is negligible.**
- FLOPs: 2·4096·256 = **2.097 MFLOP** per (token, gate) as an FP32 GEMV,
  plus ~256-element softplus/sqrt/bias-add/topk/gather/normalize.
  Device math alone ≈ 0.26 µs at T4's ~8.1 TFLOP/s FP32.
- Host-visible cost ≈ ~8 small CUDA ops ≈ **~0.1–0.3 ms serial**, anchored
  on the *measured* 6-op combine loop = 0.215 ms/row
  (LIVE_PROFILE_RESULTS.md: 0.139 s / 645 rows). Worst-case sanity bound:
  the live ABC microbench put a much larger GEMM at 0.075 ms alone /
  3.56 ms under H2D saturation — still ≪ row scale.
- One +1-layer hint per row: ≤0.3 ms vs the 102.7 ms row wall ≈ ≤0.3%
  *even if issued on the critical stream*; the device is ~90% idle
  (6.89 s busy / 66.2 s) so it can also ride a side stream during the
  native call.
- All-40-score-gates burst per token: 84 MFLOP ≈ ≤4 ms/token ≈ ~0.1% of
  the ~4.4 s decode-token wall.
- Hint ids to host: pad the existing pinned route buffer (24 B → ~1 KiB)
  — the same single D2H + one existing sync; zero added syncs.
- Honest caveat: no *isolated* router-kernel timing exists in evidence;
  the router lives inside the 13.9% "unknown" bucket (≤14 ms/row shared
  with attention + orchestration). The overhead claim is T3-modeled on
  T4-anchored per-op costs, not a clean T4 measurement.

**F7. [T1+T3] Legal evaluation distance per gate — the complete table.**
Hint legality does not degrade with distance (inputs are committed
either way); only precision does. Lead = how much idle-bank time exists
before demand:

| Hint | Source state | Earliest legal issue | Lead to demand(L′) | Idle-bank budget before demand |
|---|---|---|---|---|
| `ids(L∈0-2) = tid2eid[t]` | input_ids(t) | token start (post sample) | L rows + token boundary | exact, not a hint; ≤L+1 gaps |
| `gate_{L+1}(x_L)` | ffn_norm_out(L) | mid-row L (post-attn, pre-stage) | ~1 row (65–103 ms) | gap(L) = 0.54–1.12 records |
| `gate_{L+k}(x_L)`, k≥2 | ffn_norm_out(L) | mid-row L | ~k rows | k consecutive gaps |
| `gate_{L′}(h_L)` / `(attn_hc_out_L)` | layer input / post-attn hc state | up to ~half-row earlier than x_L | ≥ same | staler input, same budget |
| `gate_{L}(x_L(t−1))` | previous token's gate input | previous row | whole token (~4.4 s) | adds a token of drift — strictly staler than same-token sources; still measurable |
| `gate_{22}(x_21)` boundary | ffn_norm_out(21) on cuda:0 | mid-row 21 | ~1 row | gate_22's weight lives on cuda:1 → keep a 2 MiB mirror on cuda:0 OR evaluate on cuda:1 right after the handoff (8 KiB h already crosses). Only pair affected. |

Note the asymmetry: a +1 hint issued at route(L) faces a 34–44 ms record
service vs a 23.0–37.6 ms gap — a single record may not even complete in
one gap (the sim already prices 199–250 abandoned partial fills for the
hash arm). Deeper lookahead (k≥2) lets progress accumulate across
consecutive gaps; the *total* per-token gap pool (~23–48 record-slots) is
fixed regardless.

**F8. [EXT] The mechanism is well-trodden externally — but DSv4 is a
different regime.** Fate (arXiv:2502.12224) is literally this mechanism:
adjacent-layer gate input → next-layer gate → prefetch, "high prediction
accuracy without additional GPU overhead," ≤4.1×/2.2× decode vs on-demand
loading. HOBBIT reports 96% top-1 next-layer prediction on Mixtral-8x7B
from inter-layer gating-input similarity (residual-stream-driven).
Mixtral-Offloading applies next-layer gating to current hidden states.
Speculating Experts (2603.19289) removes even the gate eval: a
parameter-free quasi-hidden state (normalized residual + default vector)
predicts next-layer experts — and explicitly warns that small-expert-pool
results "may not extend to modern MoEs with larger expert pools." ST-MoE
predicts via an ids→ids cross-layer correlation table (no hidden states).
PreScope documents that correlation strength varies by layer group
(input/output strong, middle weaker). PROBE distills the router into a
lookahead predictor. Pre-gated MoE replaces the gate with a *trained*
function — a ROUTING violation if adopted as the route; hint-only is the
dee-legal form.
DSv4 deltas that break external numbers: top-6-of-256 vs papers'
top-2-of-8 (Mixtral); sqrtsoftplus + learned *bias* on the selection
path (bias can dominate or dampen input sensitivity — unmeasured either
direction); Hyper-Connections 4-channel residual (a learned mix, not a
plain residual) plus attention + two norms between adjacent gate inputs;
compress_ratios alternate 4/128 (attention-output magnitude varies by
layer parity). **No external accuracy figure may be imported.**

**F9. [T1] dee has already built this seam once.** The legacy
OracleScheduler.predict_next (oracle.h:61-67) is verbatim this idea for
the Ornith-era synthetic model — "predict for the NEXT layer given the
hidden state at the current layer's input" — with a per-layer MLP trainer
(modal_step2_train_oracle.py:131-183, union_recall@k eval) and profiling
metrics for adjacent-overlap vs random expectation (profiling.h:316-317).
In real-model mode it is a deliberate no-op stub (oracle.h:11-16). The
C++ prefetch engine therefore has a *dormant* hint seam; what's missing
is a DSv4-faithful hint function plus the deferred idle-gap admission
engine (LEGAL_PREFETCH §10).

**F10. [T2+T3] Value on the sealed timeline is capacity-bound, not
knowledge-bound.** Decode misses = 1224 preads / 15.24 GiB over 645 rows
(miss mean 1.898/row; hash misses 196/270 requests). The idle pool admits
0.54–1.12 records/row → 346–720 slots/response vs ~81.6 misses/token.
Even *perfect* miss-stream precision approaches only the oracle bound
(−11.3 to −24.8 s decode fill = 26–58% of decode fill, 13–29% of the
86.3 s response fill wall); realistic = precision × min(covered misses,
capacity). The mechanism's ceiling on this bank equals the ceiling every
legal source already shares — but unlike prev-token id-copying (miss
recall 0.0054), the gate-input channel is structurally different:
it predicts from *state*, not from *history*, so its miss-stream
precision is an independent open quantity.

**F11. [T2] No hidden-state trace exists anywhere in the corpus.**
Captures are transient in-memory dict references — taps at
layer_reference.py:608-641 write into a throwaway dict when capture is
off (:600). trace_spec.py:92 hard-rules "Never captures a full
hidden-state or logits tensor" (boundary JSONs carry sha256 + bounded
slices only). The route journal is ids-only. No `.pt`/tensor dump of
gate inputs exists in `benchmark_reports/`, `experiments/`, or
`evidence-live/`. → **Gate-input correlation for DSv4 is UNMEASURED and
cannot be computed from existing artifacts.** The journal supports only
id-space analyses (F4, and the CCT spec below).

**F12. [T1-spec] Smallest provably-inert instrumentation ("gate trace").**
- Hook: on `DeepseekV4Layer` (shared by reference and candidate
  backends — model.py:616-622 builds the same class), a default-`None`
  attribute `self.gate_trace`; in `forward()` after the existing
  `c["ffn_norm_out"] = x` (:628) add
  `if self.gate_trace is not None: self.gate_trace.append(x)` — and
  optionally the same for `x_hc` (:608), `attn_hc_out` (:619),
  `ffn_norm_in` (:626). These are already-computed, freshly-allocated
  tensors; appending stores a *reference* — no detach, clone, cast,
  device op, or sync.
- Inertness: OFF = one `is not None` per layer-forward (688 checks/run,
  ~ns total). ON = one list-append per tap — zero CUDA calls, zero
  synchronization, zero allocation (the tensor already exists); retained
  footprint ≈ 8 KiB (ffn_norm_out) to ~88 KiB (4-tap set) per layer per
  step → ~60 MB over the 16-forward run, released after a post-`generate()`
  `torch.save`. Dump gate_w/gate_b once (+tid2eid): ~110–172 MB one-time.
- **Hard rule:** do NOT route this through `per_step_captures` on the
  native path — `capture is not None` forces the `last_route`
  `.cpu().tolist()` branch (layer_candidate.py:109-113) = an added D2H +
  host sync per layer per token → *not* inert. `gate_trace` bypasses the
  capture dict entirely.
- Enable only on non-scored evidence reps; **do not add it to GPU Batch
  #1 at all** (a scored campaign takes zero new code). Collection needs a
  real forward (mounted checkpoint): piggyback on the next real-forward
  evidence run, or a Kaggle CPU batch (the fp16-candidate/fp32-direct
  backends are device-agnostic).

**F13. [T3-spec] The offline eval that closes the question — pure local
CPU, zero remote spend.** Inputs: gate-trace artifact + sealed journal +
`miss_stream.csv`. For each target score layer L′∈[3..42], each forward
step/position, each source (x_{L′−1}, x_{L′−2}, h_{L′}, attn_hc_out_{L′−1},
ffn_norm_in_{L′−1}, x_{L′−1}(t−1)): hint =
`top6(sqrt(softplus(x_src @ W_{L′}ᵀ)) + b_{L′})` via
`common.router_select` verbatim. Metrics: recall@6 on *demand* AND on the
*baseline host-miss set* per row (the only recall that prices pollution
honestly), per-layer/per-distance curves, and cosine similarity between
each source and the true x_{L′}. Self-validation anchor: source = x_{L′}
must reproduce the journal's ids exactly (recall 1.0 incl. bias — proves
the harness). Sample mass: prefill contributes 7 positions/row
(≈294 adjacent pairs) + decode ≈630 pairs per trace run.

## PRIZE MODEL vs dee timelines

T4 compute is constant across both columns; only bank speed moves.
Gap capacity = gap_ms × BW / 12.75 MiB.

| | dee sealed timeline (2×T4, /tmp bank 0.29–0.37 GiB/s) [T2/T4] | prize-model timeline (bank ≥0.7–7 GB/s; AGENTS.md economics) [T3 derived] |
|---|---|---|
| Row wall / non-fill window | 102.7 ms; gap 23.0–37.6 ms | ~same ms (compute-side unchanged) |
| Records admissible per gap | 0.54–1.12 | @0.7 GiB/s ≈ 1.3–2.1; @2.9 ≈ 5.4–8.9; @7 ≈ 13–21 |
| Record-slots per token (43 rows) | ~23–48 | ~56–900 — exceeds worst-case demand (258) |
| Decode miss demand | 81.6 misses/token; 15.24 GiB/run | up to all-miss on cold banks |
| Binding constraint | idle-bank capacity | **hint precision on the miss stream** |
| Mechanism ceiling | oracle −24.8 s (all sources combined) | most of the miss stream hidable behind attention/combine windows |
| Verdict | hint channel adds ≤ its share of a ~29%-of-fill bound — mechanism DEFERRED with the engine | exactly the LEGAL_PREFETCH §10 reopening condition — precision becomes the whole question |

On the sealed bank the cross-layer gate cannot beat capacity; on a
roofline-target bank it becomes the candidate that determines whether the
bank disappears behind compute. That asymmetry is the reason to spend the
cheap instrumentation now rather than the mechanism.

## DISPOSITION

- **As an official route / gate input: REJECTED, permanently** (T0; F1 —
  matches LEGAL_PREFETCH item 7 and OFFICIAL_LOOKAHEAD).
- **As a prefetch-hint source: LEGAL, cheap, and currently unproven.**
  Legal under the contract (hint-only; a wrong hint wastes idle bytes,
  never changes execution). Mechanically cheap (F5/F6 — resident weights,
  ≤0.3 ms/hint, piggyback D2H). Unproven where it matters: recall on the
  *miss stream* for a top-6-of-256 sqrtsoftplus+bias router over an HC
  backbone is an open quantity no prior-art number covers (F8/F11).
- **Action: instrument, don't build.** Deliverable from this track: the
  `gate_trace` spec (F12) + the journal-only analyses (OPEN UNKNOWNS).
  Reopen as a Phase-5 speculative-prefetch artifact iff (i) measured
  miss-stream recall clears the pre-registered bar (FALSIFIERS) AND (ii)
  the idle-gap admission engine exists (itself gated on bank >~0.6–0.7
  GiB/s or a Phase-5 decision). **Not for GPU Batch #1.** No speculative
  prediction enters any scored arm.
- Net: the idea survives exactness review; its fate is decided by two
  cheap measurements, not by building anything.

## FALSIFIERS

1. Any T0 path where gate(L+1) consumes an earlier-than-`combine(L)`
  state would invalidate F1 (requires a different model.py — pinned at
  rev 9e165c30).
2. Trace-measured recall@6 on the *miss stream* < ~0.10–0.20 → the
   channel is dead in kind (parity with prevtok's 0.0054 outcome); close
   permanently. Demand-side recall alone is NOT sufficient evidence —
   misses are the priced stream.
3. Journal full-pass showing same-index and CCT structure at literal
   chance level AND trace cosine-sim(x_L, x_{L+1}) ≈ 0 → no cross-layer
   signal exists in any channel.
4. Measured hint-eval cost >~10 ms/row forced onto the critical stream
   would falsify "negligible overhead" (current bound: ~0.1–0.3 ms T3,
   3.56 ms worst-case saturated-GEMM T4).
5. Hash-determinism falsifier: any two journal rows with equal token_id
   and differing L0–2 id sets (spot check held for token 343; full pass
   spec'd).
6. A hint that ever reaches execution = contract violation regardless of
   accuracy (standing rule; not a finding).
7. Bank-speed falsifier-in-reverse: on a ≥0.7 GiB/s bank, capacity stops
   binding and this memo's "capacity-bound" verdict flips to
   precision-bound — already priced in the timelines table.

## OPEN UNKNOWNS + cheapest closure

1. **Gate-input correlation + hint recall for DSv4** — UNKNOWN; the
   decision quantity. Closure: F12 `gate_trace` on the next real-forward
   evidence rep (non-scored; or a Kaggle CPU batch) + F13 local eval.
   Cost: ~20 lines of inert hook + one ≤250 MB artifact + local CPU hours.
2. **ids→ids cross-layer structure (CCT + same-index)** — UNKNOWN but
   computable TODAY from the sealed journal, pure local Python, zero
   remote spend: ~60-line tool over `phase2_legal_prefetch_eval.load_batches`
   reporting P(e′∈ids(L+1)|e∈ids(L)) marginals, top-r CCT-hint recall vs
   actual, and the F4 same-index metric full-pass. If a strong ids→ids
   channel exists, it is a *cheaper* hint than hidden-state gating and
   changes the instrumentation's priority.
3. **Isolated router-kernel timing on T4** — UNKNOWN precisely (bounded
   ~0.1–3.56 ms/GEMM-class by the ABC suite; modeled 0.1–0.3 ms).
   Cheapest: one microbench inside an already-planned profiled run; not
   decision-blocking.
4. **Best hint-input variant** (x vs h vs attn_hc_out vs ffn_norm_in vs
   cross-token) — UNKNOWN; F13's eval ranks them for free once traces
   exist. Include per-layer curves (PreScope warns correlation is
   layer-grouped; compress_ratios 4/128 alternation may add parity
   structure).
5. **Gate-22 boundary mirror** — 2 MiB detail, matters only at build time.
6. **Hint weights** — moot by construction: routing weights are
   re-derived by the official gate at demand; hints carry ids only.
