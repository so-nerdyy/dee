# R3 — Prefetch economics: the (recall, precision, lead) surface vs the idle-gap bound

- Track: R3 — parameterized prefetch-value simulation on the sealed journal.
- Branch: `research/prior-art-r03` @ `dc78dc4` (worktree `.freebuff/wt/r03`).
- Tool: `tools/phase2_prefetch_economics_sim.py` (stdlib-only sibling of
  `tools/phase2_legal_prefetch_eval.py`; same sealed v50 journal, same
  two-level tier model, tightened idle-gap admission).
- Results: `research/prior-art/r03-results/prefetch_rpk_grid.csv` (900 cells),
  `prefetch_rpk_summary.json`, `prefetch_oracle_bound.csv` (6 cells).
- Rules observed: local CPU only, no remote spend, no integration edits.
  SIMULATED throughout; admission gate reproduced sealed counters exactly
  before any variant was scored.

## QUESTION

Given the idle-gap bound and the LRU tiers, what does a predictor with
recall `r`, precision `p`, lead `k` layers actually save on the sealed
journal, and how much pollution does it cost? Where — if anywhere — is the
(r,p,k) region that clears the bound?

## ADMISSION GATE — PASSED before scoring

Same gate `TIER_REPLAY_VALIDATION.md` / `phase2_legal_prefetch` used:

| Quantity | Sim | Expected | Sealed live anchor |
|---|---:|---:|---:|
| Single-level host miss, cuda0 | 1391 | 1391 | 1390 |
| Single-level host miss, cuda1 | 1091 | 1091 | 1091 |
| Device miss, cuda0 | 2284 | 2284 | 2285 |
| Device miss, cuda1 | 2161 | 2161 | 2159 |
| Two-level total preads | 2453 | 2453 | — |

Journal sha256 `665aac3e…ae1`. Baseline: 1,224 decode preads,
42.838 s decode fill wall. Device outcomes asserted identical in every cell
(`device_outcome_mismatches = 0` everywhere — prefetch targets the host
tier only, so H2D bytes are invariant by construction, not by assumption).

## MODEL

`predictor(r,p,k)`: emitted once per decode target row j at row
`max(j−k, first_decode_row)`; names a Bernoulli-r fraction of the row's
true demand set plus enough never-demanded-at-j same-layer experts to make
precision = p. Wrong names drawn from the layer's ever-demanded set first
(a trained predictor names records it has seen), then 0..255. Emission
order seeded-shuffled (no confidence ranking assumed).

Admission is strictly-idle and tightened vs the published eval: reads run
only inside per-row disk-idle gaps (`gap_ms × BW`), never delay a demand
read, complete into the real host LRU (tail evictions priced), and a
candidate is dropped the moment remaining gaps before its deadline can no
longer cover its bytes — a real engine knows deadlines and progress.
Pending served earliest-deadline-first; finishing in-flight work precedes
new candidates.

Pollution cost is *realized*, not assumed: per-demand outcomes are diffed
vs the no-prefetch baseline — `base-miss → arm-hit` = conversion (a pread
truly avoided); `base-hit → arm-miss` = pollution fill (an EXTRA demand
pread); `base-miss → arm-natural-hit` = survival gain.

Grid: r,p ∈ {0.1,0.25,0.5,0.75,1.0} × k ∈ {1,2,3,6,12,43} × gap ∈
{23.0, 37.6 ms} × BW ∈ {0.29,0.33,0.37 GiB/s} = 900 cells + 6 oracle cells.

## FINDINGS

**F1. Lead k=1 is dead on this bank — the single most important cell.**
At gap=23.0 ms a 1-layer lead saves **0.00 s at every (r,p) and every
bandwidth** — a 12.75 MiB record needs 33.65–43.90 ms of bank time and the
conservative gap is 23 ms; nothing can complete. Only at the most
optimistic corner (gap 37.6 ms, 0.37 GiB/s → need 33.65 ms) does k=1 wake
up, and then fully (~10.85 s mean). **Every ~1-layer-lead predictor in the
literature — Fate/cross-layer gate, APEX, pre-attention linear, HOBBIT,
Speculating-Experts, Mixtral-Offloading spec-load — is worth exactly zero
on the sealed bank regardless of accuracy.** R1/R2's cheapest candidate is
bank-blocked by physics, not by recall. This is the simulation-level proof
of what LEGAL_PREFETCH.md's per-row arithmetic implied.

**F2. Minimum viable lead is k≥2, saturating by k≈6.** Mean wall_saved
over the r×p surface:

| k | gap 23.0 / BW .29–.37 | gap 37.6 / BW .29–.37 |
|---|---|---|
| 1 | 0.00 – 0.00 s | 0.00 – 10.85 s |
| 2 | 6.62 – 8.43 | 10.26 – 12.46 |
| 3 | 7.31 – 9.11 | 11.13 – 13.07 |
| 6 | 7.83 – 9.66 | 11.60 – 13.54 |
| 12 | 7.98 – 9.84 | 11.76 – 13.66 |
| 43 | 8.00 – 9.86 | 11.72 – 13.63 |

Deeper lead buys nothing past ~k=6 — pending depth grows (max 72 records
at k=12) but gap capacity is fixed (~0.54–1.12 records/row). Lead k≥2
means: **only ProMoE-stride, whole-sequence predictors (SiDA/RPP/
MoE-Beyond class), or draft-token lookahead can produce it** — the
zero-training cross-layer gate cannot (its natural lead is exactly 1;
a k≥2 variant needs x_{L} to predict gate_{L+2}, two layers of drift —
unmeasured, likely far below the ~97% claimed at k=1).

**F3. Precision is the real currency; recall is nearly free to waste.**
At k=6 / gap 23.0 / BW 0.33: r=0.5, p=1.0 → **12.26 s**; r=1.0, p=0.5 →
**10.44 s**. High recall at low precision wastes the scarce resource (gap
bytes) on wrong names — a predictor that names everything half-right loses
to one that names half the set perfectly. Precision p≥0.75 is where the
surface tops out; below p≈0.5 the gains collapse toward the noise even at
full recall (r=1.0, p=0.1 → 3.46 s).

**F4. Pollution is real but bounded — and always net-positive in-grid.**
Worst pollution in the whole 900-cell grid: ~30 extra demand fills
(r=1.0, p=0.1) — the host LRU absorbs wrong prefetches because a
completed prefetch only displaces the LRU tail, and `evicted_live_pf`
(prefetch evicting a record later demanded) stays ≤ 83 even at the deepest
queues. No cell went net-negative on wall; the failure mode is wasted gap
capacity (`pf_completed_never_used` up to 247), not evictions. Caveat:
this is LRU's forgiveness, not a free pass — the cost shows up as
lower conversions, i.e., inside the same wall_saved number.

**F5. The sim's own oracle reproduces the published bound.** Oracle arm
(perfect future knowledge, same admission discipline):
11.94 s @ gap23/BW.29 → 24.80 s @ gap37.6/BW.37 — matching
LEGAL_PREFETCH.md's 11.3–24.8 s (26–58% of decode fill) end-to-end. The
best parameterized cell (r=p=1.0, k=12, gap37.6, BW.37) lands at
**100.15% of oracle** — i.e., the synthetic predictor at perfection IS
the oracle, a clean self-consistency check on the harness.

**F6. Realistic-predictor cells price the actual prize.** Taking the
published-accuracy region (p≈0.75–1.0, r≈0.5–1.0) at a feasible lead
(k≥3) and conservative gap: **~8–13 s of decode fill** — vs the oracle's
11.9–24.8 s. The capacity-bound verdict from LEGAL_PREFETCH survives a
full parameter sweep: on this bank, prefetch value is
`min(knowledge, gap_capacity)` and capacity is the smaller term for every
lead ≥2.

**F7. H2D is structurally untouched.** `extra_h2d_gib = 0.0` in all 900
cells (asserted) — host-tier prefetch never changes device residency on
this workload. Any H2D claim for a predictor must come through a device-
tier arm, which this sim does not model (device misses are demand-driven).

## PRIZE MODEL vs dee timelines

| Regime | Value | Basis |
|---|---|---|
| Oracle ceiling (all sources) | −11.94 to −24.80 s decode fill | SIMULATED, reproduces LEGAL_PREFETCH |
| Best legal-shapable cell | ≈ −13.5 s (r=p=1, k=6, optimistic gap) | SIMULATED |
| Realistic published-accuracy cell | ≈ −8 to −13 s decode fill | SIMULATED @ p≥0.75, r≥0.5, k≥3 |
| Any k=1 mechanism | **0.00 s** at conservative gap | SIMULATED — falsifying |
| Pollution floor | ≥ −3.46 s even at p=0.1 (net still positive) | SIMULATED |
| On a ≥0.7 GiB/s bank | gap capacity ×~2–20 → lead-1 mechanisms reopen | DERIVED (gap_records_per_row scales with BW) |

## DISPOSITION

- The (r,p,k) surface is now priced; prefetch-hint value on the sealed
  bank is **SIMULATED, not speculative**.
- Any future dee hint engine must have lead ≥2 AND precision ≥~0.75 to
  matter here — that rules out every zero-training ~1-layer mechanism and
  puts the bar on trained predictors' *precision on the miss stream*.
- No change to any Phase-2/GPU-Batch-1 arm. The deferred idle-gap engine's
  reopening condition (bank >~0.6–0.7 GiB/s, or a predictor that clears
  this surface's bar) is unchanged — this sim refines, not revises, it.

## FALSIFIERS

1. A measured DSv4 hint source with lead ≥2 and miss-stream precision
   ≥0.75 would move from "priced region" to "build candidate" — the sim
   says the payoff exists; existence of the predictor is the open part.
2. If the `pf_completed_never_used` or `evicted_live_pf` counters in a
   real implementation exceed the sim's bounds at equal (r,p,k), the
   admission discipline is wrong — re-run with the real scheduler.
3. A faster bank (>0.6–0.7 GiB/s) changes need_ms below gap_ms → k=1
   resurrects and the whole surface lifts; the verdict is
   bank-regime-conditioned (oracle column is the same mechanism at
   different physics).
4. If gate-input captures later show cross-layer recall ≈ generic-
   predictor level (~0.5) at k=1, that channel stays dead at ALL banks —
   precision and lead both fail.

## OPEN UNKNOWNS + cheapest closure

1. **Whether any realizable predictor reaches lead ≥2 at usable precision
   on DSv4** — UNKNOWN. Cheapest: R2's `gate_trace` capture + local eval;
   test `gate_{L+2}(x_L)` directly — if two-layer drift keeps recall ≥0.7,
   the zero-training family reopens; if not, only trained/draft predictors
   remain.
2. **Pollution at device tier** — this sim models host prefetch only; a
   device-side arm needs a device-residency policy spec first.
3. **Wrong-name realism** — the synthetic predictor names wrong records
   from the same layer's demand pool; a real predictor's error structure
   (systematically wrong vs randomly wrong) could change eviction overlap.
   Cheap variant: bias wrong names toward recently-demanded records.
4. **Partial-fill accounting** — `partial_ms_wasted = 0` everywhere under
   the hopeless-drop rule; a real engine that can't predict gap sizes
   reintroduces abandoned-read waste (published eval priced 199–250
   abandoned partials for the hash arm).
