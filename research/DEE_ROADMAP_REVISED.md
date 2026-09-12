# DEE_ROADMAP_REVISED — post-R-series ordering

- Author: orchestrator (R12 deliverable), 2026-09-11. Basis: the
  consolidated matrix (`research/DEE_MOE_PRIOR_ART_MATRIX.md`) + all 11
  track notes under `research/prior-art/`.
- Everything below is conditioned on the sealed evidence base; the ledger
  stays CPU 0/10, GPU 0/2 until the remote-run gate report + explicit
  authorization.

## 1. What still closes Phase 2

- **GPU Batch #1 stands** — four-arm causal campaign (A0 baseline /
  A1 VRAM-only / A2 host-only / A3 combined) on the re-pinned campaign
  head `79eac7e` (now carrying the R13 resident-garbage fix + its
  regression test; 15/15 ctests green). It must answer: exactness
  preserved? VRAM repair live? host hierarchy reduces slow-tier traffic?
  wall improves? what's next bottleneck?
- **The CPU build gate precedes it** — CPU 1/10: remote build +
  host-test gate on the pinned campaign commit.
- Remaining pre-launch work is *evidence packaging*, not mechanism work:
  remote-run gate report (R13 SHA + tests + negative control + integration
  SHA, this matrix, this roadmap, explicit impact statement, resolved
  ledger + launch SHAs, remaining-risk list with kernel detection
  mechanisms — P0 hardware gate / P2 mechanism abort / §4.9 wall watchdog
  / incremental evidence / cron guard).
- Nothing in the R-series adds a Phase-2 blocker. Every "interesting"
  finding is post-Phase-2 or bank-regime-gated.

## 2. Does the matrix change GPU Batch #1?

**No.** The only candidate is R2's `gate_trace` — a default-None
list-append hook on `DeepseekV4Layer` (zero CUDA calls, zero sync, ~60 MB
footprint when on, ~ns when off). Per the campaign rule it's
**evidence-only instrumentation and explicitly NOT added to scored arms**;
it piggybacks on a non-scored evidence rep if one happens, else waits.
Batch #1 launches as causally clean as designed: no predictor prefetch, no
CPU execution, no new cache policy, no model change.

## 3. Does the Phase-3 store remain the immediate capacity step?

**Yes, unchanged.** R-series sharpened the argument rather than revising
it: the sealed bank is trace-backed (2,364/11,776 records) — every
arbitrary-prompt claim, every predictor eval, every cross-request study
needs the full 146.6 GiB universe. Phase-3's resumable builder (dee4-v2/v4)
and lazy store (dee4-v5) are already prototyped; `integration/phase3-store
@ 0707fd4` is staged. Do not materialize locally — disk-constrained.

## 4. First arbitrary-prompt run: eager or lazy store?

**Lazy (dee4-v5 demand-paged), for evidence economics.** Eager build is
~35 min of one-time CPU-batch work but produces the full 146.6 GiB artifact
before any inference question is answered; lazy lets the *same* run answer
"does arbitrary routing work?" while building only the records the prompt
actually touches — the first arbitrary prompt doubles as a miss-stream
generator (the thing every predictor/residency question needs). Eager
becomes right only if lazy's per-record fill latency breaks the run
budget — mitigable by falling back to the resumable builder mid-run
(CPU 4→5 contingency). The ledger already reserves CPU 4/10 (+5) for the
store build either way.

## 5. After arbitrary-prompt execution: predictor-prefetch vs CPU hybrid?

**CPU hybrid first — argued from the R3 surface × R5 break-even:**

- **Prefetch side (R3)**: the 900-cell surface says a legal hint needs
  **lead ≥2 layers AND precision ≥0.75** to clear the capacity bound —
  and even then the ceiling is ~8–13 s of a 42.8 s decode fill. The only
  mechanisms that produce lead ≥2 are trained predictors (unproven on
  DSv4's 256-expert sqrtsoftplus+bias router — MoBiLE's warning + dee's
  own 0.503 rejection) or draft tokens (out of scope). Worse, the
  zero-training k=1 family is worth exactly 0 s on this bank. Building
  the hint engine first means building machinery whose best legal input
  may not exist.
- **CPU-sink side (R5)**: break-even `t_cpu(1) ≲ 25–30 ms` under worker
  overlap. **MEASURED 2026-09-11 (CPU 6/10, kernel dee-tcpu-real-geometry):
  the portable-torch reference runs ~2,750 ms/expert at real geometry —
  ~90x over the bound. The portable path is dead.** Only a tuned
  AVX2/AVX-512 dequant+GEMV kernel (~2–6 ms derived, does not exist yet)
  could clear it. Prize if a kernel is built: ~6–9 s stage-enqueue +
  VRAM-churn relief, PLUS the T9 pool revives under its §7(c)
  CPU-decoupling clause. The exactness contract is settled
  (gate-equivalence to fp32 reference + deterministic partition + re-seal).
- **Ordering (REVISED by the measurement)**: CPU hybrid is no longer a
  near-term lever — it requires authoring a tuned kernel first, which is
  its own work item. Revised sequence: Phase-2 gate → Phase-3
  arbitrary-prompt → THEN decide between (a) tuned-CPU-kernel authoring
  (est. days of kernel work + exactness validation, prize ~6–9 s) and
  (b) predictor-prefetch (needs lead≥2 + precision≥0.75, prize ~8–13 s).
  Predictor work stays gated on gate_trace recall evidence.

## 6. What requires multi-request serving traces

- **All cross-request residency work** — R4's flip is decisive: LIRS/
  WTinyLFU/canonical-ARC *lose* to LRU on the cold sealed stream but win
  **+20–27 pp at 16 GiB** on multi-request probes (identical-repeat +
  scan-flush). LRU retains nothing through a scan; request-aware policies
  do. The payoff is unmeasurable without a real corpus.
- **fMoE semantic-seeded expert maps** (regime-C contract); **EPLB/metro-
  class** balancing; **SiDA/ExpertFlow-a** batch-ahead shapes; **dee-serve**
  global caches generally.
- Cheapest corpus: R4's router-only CPU replay — exact only for hash
  layers 0–2; score layers need real hidden states → the pure-torch
  reference model on the mounted checkpoint dataset (~2–4 min/req, ~150–350
  req/session, ~150–250 lines glue). Proposal only.

## 7. Second model family for universality (least adapter effort)

**Qwen MoE (Qwen3-MoE class)** — strongest case:
- Score-routed (no hash layers — the tid2eid path is DSv-specific anyway),
  standard sigmoid/softmax gate, dense checkpoint on HF, large community,
  and *every* prior-art row evaluates on it (MoE-Infinity, APEX, Fate,
  DuoServe all report Qwen numbers — free comparability).
- Kimi K2/K3 is the *thesis-aligned* choice (kimi-k3-in-c already proves
  dee-shaped streaming on it) but its scale (~1T+) exceeds Kaggle /tmp
  headroom — a Phase-6 hardware-tier question, not a portability check.
- Llama-4-MoE is third: real but sparse community evidence.
- The dee tier code is already model-agnostic (V4.1 was the sanity
  check); a Qwen adapter = safetensors enumeration + a gate function +
  record format — days, not a phase.

## 8. Most informative modern-hardware test per dollar

**One A100/H100 (or GH200) run on a fast bank (NVMe ≥2 GiB/s), not a
multi-GPU rig.** Rationale: dee's sealed bank is ~100× slower than PCIe —
every deferred mechanism (R3's whole surface, R2's hint channel, the
idle-gap engine) is gated on `bank > ~0.6–0.7 GiB/s`. A single fast-bank
run re-tests the entire defer stack in one shot: gap capacity scales
linearly with BW, so k=1 predictors resurrect, oracle bound expands, and
the "capacity-bound vs knowledge-bound" question answers itself. The T4
campaign answers the residency/exactness questions; the fast-bank run
answers the *prefetch economy* questions — orthogonal information per
dollar. (Phase-6 framing; not a current ledger item.)

## 9. Where approximate mode branches from exact

R8's seam spec, unchanged by synthesis: an explicit `runtime_mode:
"dee-fast"` run-mode flag — separate branch, separate seal metadata
(`approximations[]` listing each contract clause relaxed), never feeds
sealed exact evidence. Candidate first arms: SiDA-style resident-set
bounding, ExpertFlow-b cached-prediction path, precision-modified
transfers. The branch exists to *contain* approx work, not to legitimize
it into exact results.

## 10. Candidate sequences if evidence stays ambiguous

| If… | Then sequence |
|---|---|
| `t_cpu(1)` ≤ ~30 ms (expected) | CPU 6 measure → R6 G1–G3 sync cell → G4 overlap → Phase-5 hint engine gated on gate_trace recall |
| `t_cpu(1)` > 30 ms (falsifier 1) | CPU sink degrades to host-hit fast-path only → predictor work reorders to top priority ONLY if gate_trace recall ≥0.75 @ k≥2; else Phase-3→Phase-6 (fast bank) |
| gate_trace recall < ~0.10–0.20 | hint channel closed in kind (parity with prevtok 0.0054) — prediction line permanently dead in-request; residency is the only remaining lever → regime-C corpus (R4 CPU replay) becomes the next capacity question |
| Both fail | Phase-2 closes on hierarchy alone; Phase-3 arbitrary-prompt; then economics/hardware (Phase-6 fast-bank test) is the only remaining wall-mover |
| GPU Batch #1 shows hierarchy working | miss stream shrinks → absolute prizes shrink with it → prioritize Phase-3 + dee-serve framing over mechanism work |

## Remote-run gate checklist (per briefing, for the permission ask)

1. §4 resident-garbage fix: branch `fix/legacy-submit-resident-garbage` @
   `76eb640`; test `failed_submit_leaves_no_resident_block()` (17 checks,
   `test_legacy_submit_event_leak.cpp`); negative control = 5 fails
   pre-fix; integration SHA `79eac7e`; campaign ctest 15/15.
2. This matrix + roadmap.
3. Impact statement: **no changes** to CPU build gate, Phase-3 build job,
   or GPU Batch #1 — the R-series produced no integration edits; only
   candidate instrumentation (gate_trace) is scoped to non-scored reps.
4. Ledger: CPU 0/10, GPU 0/2; launch SHAs pinned at request time
   (campaign `79eac7e`, phase3 `0707fd4`).
5. Remaining risks + kernel detection: P0 hardware gate, P2 mechanism
   abort, §4.9 wall watchdog, incremental evidence rule, cron guard —
   enumerated in the campaign runbook.
