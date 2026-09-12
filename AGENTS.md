# AGENTS.md — dee.cpp onboarding (canonical handoff, stored 2026-09-10)

Single-agent context file. If a fact conflicts with older docs (e.g.
PROJECT_STATE.md, which describes the pre-DSv4 Ornith/GLM campaign era),
this file wins for current direction. Phase facts should be cross-checked
against the branches/commits cited — they are the evidence, not this file.

## Project overview

dee.cpp (Dynamic Expert Eviction / dee): inference systems project for
running extremely large sparse MoE models without accelerator memory
proportional to total parameter count.

Hierarchy: complete sparse model -> NVMe -> host RAM -> VRAM -> exact
accelerator compute. Hardware should scale with active params, working
set, throughput, concurrency — not checkpoint size.

Product classes (eventual): dee-local (minimum-hardware, SSD-heavy) and
dee-serve (global expert caches, continuous batching, cross-request reuse,
expert-aware scheduling, hardware-aware provisioning). Currently in
research-to-POC transition, not product hardening.

## Exactness philosophy

Primary runtime mode is EXACT: may change where data lives, when it is
transferred, how cached/scheduled — never silently change the model.
Preserves: authoritative routing, expert identity, checkpoint
representation semantics, expert execution, expert ordering, outputs.

Out of contract: pruning, substitution, merging, altered routing, lossy
re-quantization, approximate state transfer, predicted routing replacing
the router. Prediction may drive prefetch HINTS only; native router stays
authoritative. A prediction error may waste bandwidth — never change which
expert executes.

## Canonical model

deepseek-ai/DeepSeek-V4-Flash-0731 (rev 9e165c30e2704aec5d9d593cce3eebd58bbef1cb):
~284B backbone, ~13B active/token, 43 MoE layers, 256 routed
experts/layer, top-6 + 1 shared expert, hidden 4096, expert inter 2048.
DEE4 packed expert record: ~25,165,824 params / 13,369,344 bytes
(12.75 MiB) per expert. Canonical torture-test model until current
campaign completes. V4.1/Kimi K3/Qwen are future portability targets.

## Phase 0 — DONE. Exact real-model execution proven

Branch freebuff/deepseek-v4-flash-0731-t4, commit
7b137846893c46b3aed2c7e322f0dbbd8d3ce0ec. 2xT4 SM75, real
checkpoint-derived tensors, all 43 layers, packed FP4 storage, ~0.21
tok/s decode, 16-token generation beginning "Alan Turing (1912-1954)
was an English mathematician, computer". Caveat: selected trace-backed
expert bank only — not the full arbitrary-prompt universe (Phase 3's job).

## Phase 1 — DONE. Bottleneck characterized

Closure: ca8abd075e116a3924c3316cf50d5447de143ed8. Kaggle /tmp expert
bank is storage-bandwidth constrained at 0.29-0.37 GiB/s (3 reader lanes,
QD6, ~96% device busy; service time ~= pread). Not primarily: pread API,
io_uring, more workers/qdepth/lanes, GDS alone. Established levers: fewer
physical cold bytes + earlier legally-possible work. Treat as settled
unless hardware changes the storage regime.

## Lossless compression — REJECTED (do not reopen)

research/dee4-lossless-scan @ a00d2b03e9736ddc01c8ce2e455a34f0a3527fd0:
NO_CODEC_WORTH_BUILDING. All 2,364 records byte-exact; packed FP4 is
entropy-dense (LZ4 ~1.000, zstd modest, rANS near Shannon limit); best
~8% whole-record reduction, projected decode-wall savings too small for a
second format. Reopen only if the physical representation changes.

## Route prediction — generic family REJECTED

Generic predictor recall@12 ~0.503; wrong prefetches + cache pollution
worsened behavior. Edge0 shows a trained per-layer predictor could be
much stronger — future speculative-prefetch only, never execution.

## Phase 2 — ACTIVE. Universal expert hierarchy

ColdExpertStore -> HostExpertTier -> DeviceExpertTier ->
ExactExpertExecutor. Host tier: bounded reusable slots, optional pinned
registration, exact identity, duplicate-fill coalescing, safe eviction,
leases protecting DMA lifetime, direct materialization to final
destination. Device tier: bounded packed residency, async H2D,
generation safety, transfer+compute pins, safe eviction, exact packed
execution. Tier code must not depend on DeepSeek geometry (V4.1 is the
abstraction sanity check: 40 layers, 384 routed, top-6, hidden 5120,
expert inter 2304, ~17.93 MiB/expert, ~269 GiB routed pool).

### Working-set research (corrected)

research/phase2-ws-policy @ 95dfe0d032c2b035cbb0b8b31fa589a68a329bd9.
Earlier b7b9c7f had a causality bug (pin policies counted pinned records
as first-access hits during the same prefill = free prewarming → a
"realizable" policy appeared to beat Belady/MIN — impossible, retracted).
Contracts: A cold start, B causal within-request warmup, C explicitly
prewarmed cross-request.

Trusted conclusions: HOST POLICY = plain LRU (causal RAM knee ~16 GiB
pooled on sealed trace; ~2.8pp below offline MIN there, reaches MIN by
~32 GiB). VRAM_PRIORITY_FIX_RECOMMENDED: current score
`last_used + priority*2^20` persistently protects wrong experts;
candidate repair = pure `last_used` recency; simulated ~14.48
GB/response H2D reduction at ~281-slot VRAM cap (simulated, not live T4).
Regime-C/prewarm: RESOLVED 2026-09-10 — research/phase2-regime-c @
643d3559 (CONTRACT.md). Prewarm is a deployment contract
(PolicyResident partition + policy_slots + residency()); the flagged
never-used-distance figures traced to the sim's init (fixed on
fix/phase2-ws-sim-init @ 8c921ff — CSVs regenerated, only the 19 stale
rows moved; prose docs corrected on docs/phase2-ws-figure-fix @ 6904c31);
fetch-on-fault MIN >= static pin at every budget on the sealed
window (935/2364 records ever repeat). Prewarm arm DEFERRED to Phase 5;
regime-C rows never score against cold-start rows. True C1
reject-admission needs a non-caching bypass path (not Engine-expressible
today).

### Current clean implementation

research/phase2-integration-lru-fix, base ca8abd0, head
56dad3c1dccb0d7cee774df20d8f0047059d0bd7 (Luna's rebuild). Prereq build
fix d68cf4ad68245f9136548f3acd6e6a81f7f8e3fb (dup decls + stale var in
host-pack). Contains: host/device tier abstractions, IdentityCodec, plain
host LRU, independently switchable host-tier and VRAM-repair controls,
aligned reusable slots, optional pinned registration, leases,
generations, fill coalescing, failure handling, metrics, tests. Validated:
MinGW CPU build PASS, focused tests PASS, MSVC syntax PASS. CUDA runtime
validation PENDING. Phase 2 remains default-OFF.

### Cold-fill serialization — VERDICTED 2026-09-10 (repair NO-GO)

research/phase2-concurrent-fill @
5adda90d70e1fbca0cd2a08537c75c0f54b492b6 (SERIALIZATION_VERDICT.md,
DESIGN.md, fill_wall_models.csv). Proven serial from source: armed host
tier -> prepare_fp4_experts early-return (engine.cpp:2794) -> one
synchronous 12.75 MiB pread per stage_expert. BUT the concurrency repair
LOSES: serial fill wall 86.3 s/response vs bounded-3-lane +23% worse at
every miss count — the sealed bank is single-stream saturated
(0.29-0.37 GiB/s ceiling; 4-lane pool paid 97.2 s vs 87.5 s modeled
serial on the same miss stream). Falsifiability gate failed -> T9
implementation CANCELED as a wall optimization. Host tier's benefit is
residency (avoiding repeat reads), not read concurrency — host-arm A/B
cells remain exactness/residency evidence, perf-lower-bound only.
Conditional GO preserved in DESIGN.md for: bank on >~0.6-0.7 GiB/s
storage (at 2.9 GiB/s decode fill drops 43.5->11.6 s) or an approved
legal ahead-of-router candidate source. Consistent with Phase-1's
feed-side verdict.

### Prior audit (this machine)

research/phase2-host-tier-verification @
e63a3bc7c1001ce9343da1e41c09e09f4489d296 (worktree .freebuff/wt/v2a,
report PHASE2_VERIFICATION_AUDIT.md). 0 Critical/0 High/2 Medium/6 Low —
ACCEPT for live mechanism testing. Mediums: (1) pre-existing LLP64
map_key truncation in legacy path (deliberately retained for default-OFF
equivalence, fail-closed); (2) DeviceExpertTier scope exclusivity is
construction-time only — cross-model byte-serving possible via shared
drained cache/prefetcher (API-misuse, unreachable via Engine). Both
transfer to the rewrite. New suite test_phase2_tier_audit.cpp: 130
checks PASS; MinGW DEE_CUDA=OFF build 12/12 ctests.

### Fixed-slot staging — HOLD

research/phase2-fixedslot-proto @
fbc8b205d4ab6a6bbba56c7660b804332277b8f5: FIXED_SLOT_STAGING_HOLD.
Steady-state reservation is microseconds not milliseconds (~11us baseline
vs ~12us proto at 12.75 MiB records); the ~4.6s reservation prize was
retracted. Did cut HIT-behind-MISS submission delay ~10-21ms -> ~0.02ms
but H2D is hidden by storage fills → ~1s firm, ~2-5s plausible benefit,
remainder tied to pinned-direct/gather-copy needing real CUDA
measurement. Preserve, don't integrate.

## Phase 3 — research-prototyped. Full arbitrary expert universe

research/phase3-full-expert-store @
8a3a2cda7e265ad57d130229ffecc7d1920bb0c9 (worktree .freebuff/wt/p3, base
ca8abd0, doc PHASE3_FULL_EXPERT_STORE.md + dee.cpp/tools/phase3/).

Corrected universe: 43x256 = 11,008 main-layer experts PLUS three
MTP/DSpark buckets (mtp.{0,1,2} x256 = 768) = 46x256 = 11,776 records,
~146.625 GiB packed. Current real bank = 2,364 records — conclusively
insufficient for arbitrary routing. Prototype: manifest builder,
completeness checker, resumable builder (dee4-v2 batch / dee4-v4
segmented), dee4-v5 lazy demand-paged store, sparse-file support (fsutil
sparse — plain truncate allocates for real on NTFS), bench. Tests 9/9
incl. shipped Dee4ExpertStore resolving arbitrary (bucket,expert) over
true-geometry 157 GB sparse fixture. Cloud build est. ~35 min at prior
Kaggle repack rate; residential HF fetch too slow for eager local.

## External references (idea sources, not acceptance evidence)

MoE-Infinity (SSD/host/GPU hierarchy, pinned, async, caching, serving),
KTransformers (model-aware placement, CPU/GPU hybrid), moe-l2 (aggressive
host residency + bounded GPU cache), FreeToken (hybrid CPU/GPU + global
expert cache), Kimi-K3-in-C (RAM allocation policy > raw capacity),
Edge0 (staged streaming + trained prediction — not exact-equivalent).

## dee-serve direction (future)

Global expert residency, continuous batching, expert batching,
hardware-aware scheduling, speculative-prefetch plugins, CPU/GPU hybrid
miss execution, model-family adapters, multi-model cascades. Cross-model
KV transfer is approximate today — approximate serving mode only, never
core exact contract.

## Roadmap

P0 real giant-model exact execution DONE. P1 bottleneck DONE. P2
SSD->RAM->VRAM hierarchy ACTIVE/late-stage. P3 arbitrary universe
research-prototyped. P4 second MoE family not started. P5
serving+concurrency future. P6 modern hardware/economics future. Phases
answer existential questions — don't become endless optimization
campaigns.

Immediate Phase-2 closure path: CUDA mechanism validation -> VRAM-only
matched T4 A/B -> host-tier concurrency fix/verification -> host-only
matched A/B -> combined matched A/B -> reprofile. Reprofile decides next:
cold bytes dominate -> residency/reuse; H2D/staging exposed ->
fixed-slot/pinned-direct; expert compute dominates -> kernel work; dense
attention/orchestration dominates -> expert-I/O solved enough, move on.

## Performance economics

Roofline = slow-tier bytes per generated token. ~7 GB/s storage:
20 tok/s needs <=350 MB/tok exposed; 30 tok/s <=233 MB/tok. High serving
throughput is impossible while pulling GiB/token from SSD — cross-request
caching matters far more in dee-serve than in a cold torture benchmark.

## Dev structure at Devin handoff

This Devin session acts as ORCHESTRATOR: it maintains roadmap context
(this file), identifies parallelizable research/engineering work, spawns
isolated subagent swarms (own branch + own worktree each), collects their
reports, and presents evidence for user review. Swarm launches require
user approval. Subagent tasks must be disjoint (no shared write targets)
and each must deliver branch + commit SHA + written report.

Most engineering/research runs in Devin (~8-12 concurrent agents tested:
20 spawned / 12 concurrent OK on 2026-09-10; profiles: subagent_explore
read-only, subagent_general full tools; 1 foreground at a time). Astra =
planning + independent review. ChatGPT = external: phase tracking, commit
inspection, evidence review, accept/reject, roadmap, funding. Push
important branches and hand reviewers: branch + commit SHA + agent
report. Keep producers of work separate from deciders of evidence.

## Kaggle execution budget (HARD CAPS — user-set 2026-09-10)

    CPU batch runs: 10 total
    GPU batch runs:  2 total

No workaround-by-splitting: a "batch run" is any submitted job intended to
execute a meaningful experiment/validation. More requires explicit user
authorization.

CPU runs: build/store prep, artifact generation, large trace/replay
validation, packaging, env verification that genuinely requires Kaggle,
Phase-3 full-store prep where local execution is impractical.

GPU runs must NEVER be spent on: syntax debugging, ordinary unit-test
failures, dependency debugging, speculative experiments without a decision
gate, anything reproducible locally, or information source
inspection/simulation could answer.

Pre-flight gate before ANY GPU batch: (1) local build PASS, (2) relevant
CPU/unit tests PASS, (3) CUDA build or smallest CUDA mechanism test PASS
where possible, (4) exact hypothesis written, (5) baseline + candidate
defined, (6) metrics defined, (7) accept/reject criteria defined, (8) all
experiment arms packed into the SAME batch when practical, (9) artifact
collection verified, (10) abort behavior established.

Design both GPU batches as multi-arm campaigns:

- GPU BATCH #1 — Phase-2 mechanism/causal campaign, one controlled job:
  baseline + VRAM-only + host-only + combined arms plus warmups/reps.
  Answers: Phase 2 exactness preserved? VRAM repair works live? Host
  hierarchy reduces slow traffic? Wall time improves? Next bottleneck?
- GPU BATCH #2 — RESERVED, not pre-consumed. Candidates: corrected
  Phase-2 rerun if #1 exposes a fixable defect, OR Phase-3
  arbitrary-prompt/full-store real inference — decided by #1's outcome.

Accounting is explicit and per-run proposals must state: RUN TYPE, RUN
NUMBER (e.g. CPU 3/10, GPU 1/2), QUESTION ANSWERED, WHY LOCAL EXECUTION
IS INSUFFICIENT, ARMS INCLUDED, SUCCESS CRITERIA, FAILURE CRITERIA,
ARTIFACTS EXPECTED. If a run can't answer a roadmap-relevant question,
don't spend it.

Current ledger: CPU 0/10, GPU 0/2.

## Canonical heads + research-head state (2026-09-11)

ACCEPTED INTEGRATION LINES (research-head §1 decision (a) — verified:
zero post-3335b79 commits touch dee.cpp/src or dee.cpp/include):
- integration/phase2-campaign @ 79eac7e — all Phase-2 work merged:
  hardening + pydee + payload + event-leak + 7 research/audit lines +
  v2b regression conversion + R13 resident-garbage fix (failed legacy
  cold submit now discards the ensured block on both paths; stub-test
  negative control fails 5 checks pre-fix). 15/15 ctests.
- integration/phase3-store @ 0707fd4 — Phase-3 line + segmented reader +
  json_min null + host_pack fix + build-execution/kernel package.
- research/phase2-ws-policy remote @ 63edfea — sim init fix merged.
  Local branch ref still 95dfe0d; external worktree dee-phase2-ws-policy
  holds uncommitted third-party edits — do not touch.
- research/phase2-host-tier-verification @ e63a3bc — kept unmerged by
  design (old-seam audit evidence; findings folded via T1).
- PRs: #1 campaign, #2 phase3-store, #3 ws-policy, #4 verification audit.

RESOLVED LEDGER (campaign doc §9 numbering; PARALLEL lanes — user
clarified 2026-09-11 the caps are 10 parallel CPU + 2 parallel GPU):
    CPU 1/10  remote build+host-test gate   RUNNING (dee-cpp-cpu-build-gate v1)
    CPU 2/10  trace-bank segmentation       SUPERSEDED 2026-09-11 — dee4-v4
              segments are contiguous-per-bucket; sparse trace bank needs a
              new format for ~250s repack savings; CPU-4 already exercises
              segment+publish machinery at full-store scale
    CPU 3/10  fill_replay remote replay     RUNNING (dee-fill-replay-cpu3 v1)
    CPU 4/10  Phase-3 full-store build      RUNNING (dee4-p3-store-build-cpu1 v1)
    CPU 5/10  p3 resume contingency         held pending CPU-4 outcome
    CPU 6/10  t_cpu(1) real-geometry bench  RUNNING (dee-tcpu-real-geometry v1)
              (R5 gating measurement: portable executor, 4096/2048, thread sweep)
    CPU 7-10  reserve
    GPU 1/2   4-arm Phase-2 causal campaign RUNNING (dee-cpp-dsv4-phase2-campaign
              v66 @ 79eac7e — A0/A1/A2/A3 + drift bracket, ~2-3.2 h)
    GPU 2/2   decision tree — HELD pending GPU-1 outcome
    NOTE: PHASE3_BUILD_PLAN.md/kernel id "cpu1" is cosmetic; the store
    build is CPU 4/10 in the authoritative ledger.

RESEARCH ASSIMILATION PASS (COMPLETE 2026-09-11): R1-R11 prior-art swarm
landed on research/prior-art-rNN branches (all pushed) -> consolidated on
research/prior-art-r12 @ f22be19, which carries all 11 track notes under
research/prior-art/*.md plus research/DEE_MOE_PRIOR_ART_MATRIX.md +
research/DEE_ROADMAP_REVISED.md. R13 resident-garbage fix integrated into
integration/phase2-campaign @ 79eac7e (15/15 ctests). Headline synthesis:
(1) prefetch-hint surface needs lead>=2 AND precision>=0.75 to clear the
idle-gap bound — k=1 mechanisms worth 0 s on sealed bank; (2) CPU hybrid
miss execution: portable-torch t_cpu(1) MEASURED 2,750 ms/expert (CPU 6/10)
— ~90x over the ~25-30 ms overlapped bound; portable path dead, only a
tuned AVX2 kernel (unbuilt) could clear; demoted from near-term lever to
post-Phase-3 option; (3) llama.cpp is the only
matched-runnable baseline (one GGUF-conversion CPU batch); (4) MoE-Infinity
repo already offloads DSv4-Flash FP4 = free cross-validation; (5) request-
aware caching flips LRU +20-27pp only under multi-request traces. No GPU
Batch #1 changes — gate_trace instrumentation is non-scored-rep only.
Ledger: CPU 1,3,4,6 launched (parallel); GPU 1 running; see resolved ledger.

## Financial posture

No reason to buy hardware for Phase 2 — T4 env must first prove the
architecture works. Spend only where money resolves uncertainty cheap
experiments can't: full expert-store build, modern NVMe bandwidth, modern
consumer GPU scaling, large-RAM serving, second-model universality,
throughput/cost demos. POC story does NOT need magical 20-30 tok/s on
dual T4s: real giant model + exact execution + bounded accelerator memory
+ arbitrary prompts + predictable RAM/VRAM scaling + reduced slow-tier
dependence + second family + credible cost curve is stronger.

## Philosophy

Failed hypotheses, causal corrections, negative benchmarks, retracted
estimates are useful research output. Goal: a system whose performance
claims survive scrutiny, not impressive-looking numbers.

---

## Repo-local state (2026-09-10, this machine)

- Main checkout: research/phase2-early-submission @ 08d3d51
  (feat(phase2): generic host/device tier seams — the audit target).
- Disk was at 100%; cleanup freed ~20 GB (dee.cpp/tmp scratch, old
  Ornith-era venvs/kaggle dirs/oracle.pt/build dirs). ~24 GB free —
  do NOT materialize the full expert store locally (~147 GiB needed).
- Agent worktrees under .freebuff/wt/: v2a (audit branch), p3 (phase3
  branch). Other worktrees exist across C:/Users/carth/Downloads/dee-*
  and Temp/ — never `git worktree remove` or branch-delete them.
- Windows/MSYS shell: no nvcc/GPU locally; `sort`/`find`/`awk` flaky —
  prefer ls, grep tooling, python one-liners. MAX_PATH ~260: repo has
  ~175-char evidence paths, keep new worktree paths short (.freebuff/wt/
  not .freebuff/worktrees/long-name).
- Build: cmake -S dee.cpp -B build -G "MinGW Makefiles" -DDEE_CUDA=OFF.
  TOOLCHAIN GOTCHA (verified 2026-09-10, root cause found): a
  Tesseract-OCR dir in PATH shadows MinGW runtime DLLs
  (libgcc_s_seh-1/zlib1/libzstd/libwinpthread-1), killing cc1/cc1plus
  silently (exit 1). Fix: PATH=/c/msys64/mingw64/bin:$PATH before
  cmake/make, or use clang++/clang 22.1.8 (x86_64-w64-windows-gnu):
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++, or direct
  clang++ -I dee.cpp/include + needed src/*.cpp. zlib needs
  -DCMAKE_PREFIX_PATH=C:/msys64/mingw64 -DZLIB_ROOT=C:/msys64/mingw64.
  ctest for the C++ suites; python tests need pytest.
- PROJECT_STATE.md below documents the retired Ornith/GLM-5.2 campaign —
  historical reference only.
