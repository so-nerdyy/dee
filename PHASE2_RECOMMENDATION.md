# PHASE2_RECOMMENDATION.md — the one default host policy, one fallback, and the mechanical reasons

**Deliverable.** Exactly ONE default host-tier policy for the first Phase-2
implementation, at most ONE fallback, with mechanical reasons, plus the four
mandated verdicts. Everything is offline-derived from dee's sealed v50 route
journal and validated against the sealed `fill-live` (host tier) and `v60`
(VRAM tier) counters (±2 counts; see `research/phase2-ws-policy/results/
validation.json`). No GPU run, no implementation, no merge. Phase 1 remains
CLOSED at `ca8abd0`; nothing here reopens it.

---

## 1. DEFAULT: Frequency-pinned host set from a prefill-count warmup pass (freq_lru_warmup), 16 GiB pooled budget

**Mechanical definition (no future knowledge, no per-prompt knowledge):**

1. On the first forward pass (the 7-row prefill every response already
   executes), count routed requests per `(layer, expert)`. The engine's
   dedup stream makes this pass cost exactly 2,364 records ≈ 29.4 GiB of
   first-touch fills — which are compulsory anyway.
2. Rank records by count (ties: ascending expert id — deterministic).
3. Pin the top `⌊0.5 × slots⌋` records into the host tier's policy-resident
   slots (the `HostPlacementPolicy::residency → PolicyResident` path already
   exists in `host_expert_tier.h` — no new mechanism is required).
4. The remaining dynamic slots run plain LRU (today's semantics), unchanged.
5. Pin set is fixed for the response's lifetime; re-derivation happens on
   the next response's own prefill (per-response, never cross-prompt).

**Expected on the sealed trace (validated simulator):**

| Budget (pooled) | Slots | Hit rate | MiB cold/token | Wall @0.29 GiB/s | Wall @0.33 | @0.37 | @3 | @5 | @7 | @12 |
|---|---|---|---|---|---|---|---|---|---|---|
| **16 GiB** | 1,285 | **61.95 %** | **1,609.6** | **5.42 s** | 4.76 | 4.24 | 0.52 | 0.31 | 0.22 | 0.13 |
| 24 GiB | 1,927 | 71.96 % | 1,137.2 | 3.83 s | 3.36 | 3.00 | 0.37 | 0.22 | 0.16 | 0.09 |
| (today's LRU, 17 GiB live) | 1,364 | 51.2 % | ~1,960 | ~6.6 s | — | — | — | — | — | — |

At the deployed 17 GiB envelope (8.5+8.5), interpolating the validated
curves, warmup-pin yields ≈ 60 % hits vs ≈ 51 % for today's LRU —
**−22 % cold bytes** with zero additional RAM.

**Why this one (mechanical reasons, ranked):**

1. **The trace's reuse is frequency-shaped, not recency-shaped.** Exact
   stack distances: zero repeats below 213; median repeat sits 200–399
   records back. Every recency policy (LRU, ARC, LFU-tiebreak, F×R,
   layer-LRU, cost-aware) converges to the same capacity curve and
   saturates at **53.64 %** by 32 GiB — even offline Belady/MIN scores
   exactly 53.64 % at *every* budget, i.e. MIN is the recency bound and it
   is not the binding constraint. Only frequency-pinned placement beats
   that bound (static top-N reaches 91.4 % at 24 GiB), because pins catch
   far-distance repeats that no eviction policy can.
2. **Warmup counts are free and online.** The pin set derives from the
   prefill pass the runtime already performs; no predictor, no learned
   model, no cross-prompt state, no future knowledge. It realizes ~79–85 %
   of the offline pin ceiling (61.95 % vs 78.84 % at 16 GiB; 71.96 vs
   91.43 % at 24 GiB).
3. **It is implementable with existing seams.** The Phase-2
   `HostExpertTier` already separates policy-resident from dynamic slots
   and delegates residency + victim choice to a `HostPlacementPolicy`.
   Default = a policy object that answers `PolicyResident` for the
   prefill-derived top-half set and LRU-victim for the rest. No runtime
   change beyond wiring the counting pass.
4. **Zero eviction cost for the pinned half.** Phase 1 measured a 13.0 s
   whole-run reservation/LRU bucket; pinned slots do zero scan work, and
   the dynamic half is the same LRU as today.
5. **Correctness is structurally untouched.** Placement only changes which
   records are resident; every miss path is the existing exact
   materialization. Cold == warm and token identity are unaffected
   (same argument accepted for CACHE1g and the P2.x seals).

**Why 16 GiB and not more:** the knee of the online-pin curve is ~16 GiB
(61.95 %); 24 GiB adds +10 pp but pushes projected peak RSS to ~29–30 GiB
on the 31.35 GiB Kaggle host (today: 22.9 GiB at 17 GiB budget) — outside
the safety margin the campaign has historically required. 16 GiB keeps
today's envelope (≈ 26 GiB peak projected) and still cuts cold bytes ~19 %
vs today at equal RAM. If a future host has more RAM, the same policy
simply raises `policy_slots` — the policy is budget-parametric.

## 2. FALLBACK: Offline-style static frequency pin (static_freq shape), 16 GiB

**Mechanical definition:** pin the frequency-ranked top-⌊slots/2⌋…top-slots
set computed from *any completed prior run of the same model+trace class*
(e.g., the sealed v50 journal's own counts, distributed as a static
manifest), fill it once at engine build, keep zero dynamic slots (pure
static placement).

**Expected:** 78.84 % hits at 16 GiB (859.8 MiB/token; wall 2.90 s @0.29
GiB/s) — better than the default on this trace because the "offline" counts
are exactly this trace's counts.

**Why it is only the fallback (mechanical):** its guarantee is conditional
on the response's expert distribution matching the manifest. The sealed
16-token window is one workload; a different prompt shifts ranks (top-1 %
covers only 6.8 % of activations — the tail is wide). The default policy
self-calibrates per response and degrades to plain LRU behavior at worst;
the fallback degrades to zero hits if the manifest mismatches. Ship the
fallback only if the warmup pass proves operationally awkward (e.g., a
future no-prefill decode-only path), with the manifest regenerated per
trace-class and its coverage re-validated offline first.

**Relationship between them:** both are the same mechanism (policy-resident
host slots) differing only in where the counts come from (this response's
prefill vs a sealed manifest). The fallback is the default's upper bound
and its operational escape hatch.

## 3. Mandatory rider (not a policy choice): repair the VRAM priority artifact

The sealed v60 engine's VRAM tier runs
`eviction_score = last_used + priority·2²⁰` with
`priority = batch_len − batch_index` (ascending expert-id order), refreshed
on hit (`vram_cache.cpp`, `engine.cpp stage_expert`). At the sealed 281
slots this captured **12.9 %** of requests where plain LRU captures
**26.6 %** — the stale priority boost protects the first-staged expert of
every batch for the run's lifetime. One-line repair: evict by `last_used`
only (or drop the weight). Validated: plain LRU at v60's exact budget
reproduces the expected 26.58 % / 26.2 % per GPU and strictly dominates the
sealed semantics at every budget. This rider costs nothing, changes no
policy, and should ride along with whichever host policy ships.

## 4. The four mandated verdicts

1. **Does increasing RAM meaningfully reduce slow-tier traffic?**
   Yes up to ~24 GiB pooled, and *only* for frequency-pinned placement;
   beyond that, no for anyone:
   - LRU-family: 40.3 % (8 GiB) → 53.6 % (32 GiB) → **flat forever**; the
     traffic floor is 1,883.8 MiB/token regardless of RAM. More RAM is
     dead weight.
   - Frequency pin: 60.5 % → 78.8 % (16) → 91.4 % (24) → 100 % (29.43 GiB);
     traffic falls 1,735 → 860 → 348 → 0 MiB/token. Meaningful through
     24 GiB; the last 5 GiB to 100 % exceed the practical host envelope.
2. **Knee of the RAM-capacity curve:** **≈ 15.3 GiB** — the 80 %
   activation-coverage point and the empirical knee of the online-pin
   curve (61.95 % @16 GiB, +10 pp to 24 GiB, +5.8 pp to 32, flat after).
   For pure recency policies the knee is ~12 GiB (46.6 %) and the curve
   dies at 32.
3. **How close does the best realizable policy get to Belady?** It beats
   it. Belady/MIN scores exactly 53.64 % at every budget — identical to
   LRU at ≥ 32 GiB — because the trace's reuse horizon exceeds any
   realizable capacity and MIN exploits only recency structure, of which
   this window has little. The static pin (the true ceiling for exact
   caching) reaches 91.4 % at 24 GiB and the recommended online default
   reaches 71.96 % — i.e., the default is at **135 % of Belady's hit
   rate**, and the question "how close to Belady" is inverted: Belady is
   not the bound; the activation-coverage curve is.
4. **Is cache-policy work worth implementing at all?** Yes, for exactly
   two changes and nothing more:
   - the VRAM one-line priority repair (12.9 % → 26.6 % VRAM hits, free);
   - the host warmup-pin default (≈ 60 % vs 51 % at today's 17 GiB RAM;
     62–72 % at 16–24 GiB; −22 % to −40 % cold bytes → direct reduction of
     the 42 s critical fill bucket Phase 1 attributed to FEED-side cold
     fills).
   ARC, LFU, cost-aware, F×R, and layer-LRU are all LRU-shaped on this
   trace (within ±0.1–1.5 pp of LRU, mostly worse at small budgets) and
   should not be built. Router-ahead prefetch remains a scheduling lever
   (Phase-1's "earlier submission"), complementary to, not part of, the
   placement policy.

## 5. What Phase 2 implementation must NOT do (guardrails from this study)

- Do not count predicted/probable hits as resident hits; the policy uses
  only exact counts of already-issued requests.
- Do not key the pin set on the prompt or hardcode any expert set; the
  counts are mechanical byproducts of executed forwards.
- Do not import locality parameters from Qwen/moe-l2-style models; every
  number here is DSV4's own.
- Do not exceed the host envelope: budget changes must re-run the RSS
  projection (16 GiB pooled ≈ 26 GiB peak projected on 31.35 GiB; 24 GiB
  pooled ≈ 29–30 GiB — the latter only with an explicit memory-gate
  re-approval).
- Do not touch correctness gates: placement is a pure performance change;
  cold==warm, token identity, and byte-exact materialization remain the
  acceptance spine.

## 6. Reproducibility

- Simulator: `tools/phase2_ws_policy_sim.py` (single file, stdlib only).
- Streams and anchors: `research/phase2-ws-policy/results/validation.json`.
- Full matrix: `results/sim_rows.csv` (336 rows) +
  `results/sim_rows_derived.csv` (adds SSD bytes/token and all wall
  columns at 0.29/0.33/0.37/3/5/7/12 GiB/s).
- Curves: `PHASE2_CAPACITY_CURVES.csv`. Background: `PHASE2_WORKING_SET.md`,
  `PHASE2_BYTE_FLOOR.md`, `PHASE2_POLICY_MATRIX.md`.
- Source journal: v50 sealed route journal (sha256 `665aac3e…`), route
  arrays verified identical to the Phase-1 live T4×2 run.
