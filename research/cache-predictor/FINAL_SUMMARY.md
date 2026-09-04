# FINAL_SUMMARY.md — cache replacement & cross-layer prediction study for DeepSeek-V4 (dee)

Branch: `research/cache-predictor` (worktree `../dee-cache`) · 2026-09-04
**Track: OFFLINE RESEARCH.** Sealed evidence untouched; production runtime,
scheduling, and correctness gates untouched; no merges.

Evidence: the single sealed DSV4 route journal (16 forwards / 43 layers /
256 experts / top-6 / 5,676 selections; all 19 on-disk copies are
byte-identical). Provenance in `CACHE_POLICY_AUDIT.md §1`.

---

## 1. Headline answers (all trace-simulated, 16-forward window)

| Question | Answer | Class of claim |
|---|---|---|
| **Best practical cache policy** | **recency×freq hybrid** (only policy to beat LRU at every scarce-capacity point: −50 misses at cap-8, −1.3% miss bytes); at cap ≥ 32, **nothing beats plain LRU by >0.3%** | simulated |
| **Belady upper bound** | 2,364 misses = exactly the compulsory count at cap ≥ 24; 2,849 at cap-8 (−25.8% vs LRU) | simulated (offline oracle) |
| **Exact gain over LRU** | cap-8: best practical **−1.3%** (3,789 vs 3,839); oracle −25.8%. cap-32: **−0.3%** (2,483 vs 2,490). **The large cap-8 oracle gap is mostly NOT harvestable by the online policies tested** (best regret 940 of 990) | simulated |
| **Least-Stale materially helps DSV4?** | **No** — within ±1.4% of LRU at all 28 geometry points | simulated, negative result |
| **ARC materially helps DSV4?** | **No** — statistically indistinguishable from LRU/Least-Stale | simulated, negative result |
| **Cost-aware keep-value (Phase D)?** | **Degenerates to LRU** after the causality leak fix: tuning selects staleness-only weights (≡ LRU); predictor-rank-only is worse. Earlier −11.2% claim retracted | simulated, negative result |
| **Predictor recall** | best causal predictor (previous-token recurrence): **recall@6 .346 / @8 .418 / @12 .503 / @16 .544**; precision@12 .251 | measured on trace |
| **Useful vs wasted prefetch (in-situ)** | **12–30% useful / 70–88% wasted** across cap-8/16/32 | simulated |
| **Does in-situ prefetch help?** | **No.** Net misses: cap-8 −17 (−0.4%), cap-16 **+52 (+1.8%)**, cap-32 **+7** — neutral-to-harmful at every capacity | simulated |
| **Compulsory misses recoverable via prediction** | prefill: **0** (1,232 — nothing precedes token 0, information-theoretic). decode: **~21%** realized (242 of 1,132 at cap-16); 50% coverable in principle at k=12 | simulated |
| **Estimated stall reduction** | **THEORETICAL ONLY**: exposed-stall ∝ misses × read-p50 (256.99 ms measured) × (1−82.3% overlap); e.g. cap-8 LRU→Belady 3,839→2,849 misses would scale exposed stall ~26% down at equal bandwidth. **Not a wall-time measurement** | theoretical ceiling |
| **Confidence level** | **LOW-to-MODERATE** on magnitudes (single 16-forward trace), **HIGH** on direction of structure findings (0 within-step reuse, recurrence dominance) — the latter are exact trace facts, not samples | — |
| **Trace-length limitation** | 15 decode steps/layer; deltas are trace-local percentages; steady-state behavior over long generations is **not established**; one prompt, one session | — |

## 2. What was learned about DSV4 routing (measured trace facts)

1. **DSV4 has zero within-step expert reuse**: 0 of 3,870 decode
   selections appear in an earlier same-step layer; adjacent-layer
   overlap 0.13/6. **Fate-style same-token cross-layer prediction does
   not transfer to DSV4**; adding those features *degrades* recall
   (0.503 → 0.354 @12).
2. **Cross-token recurrence is the signal**: adjacent-step same-layer
   overlap 2.03/6; top-16/layer popularity covers 62.7% of demand.
   recall@12 = 0.503 for next-token bundles.
3. Because the useful signal leads by one full forward step (~43 layer
   crossings), only a **router-ahead scheduling change** could convert
   it into latency hiding; in-situ prefetch cannot (0 lead) and
   measured net-negative even on misses alone (§1).

## 3. Recommendation to Codex (exact)

1. **Do not** adopt ARC, LFU, Least-Stale, or the tuned keep-value
   policy for DSV4 expert caches — none beats plain LRU materially at
   any geometry on this trace. If any change is made at scarce
   capacity, recency×freq is the only candidate with a (small, trace-
   local) measured win: 3,789 vs 3,839 misses at cap-8 (−1.3%).
2. **Keep LRU at cap ≥ 32** (policy choice saves <0.3% there; sealed
   M4 already shows ~100% post-warmup hit rate at cap-32).
3. **Do not invest in same-token cross-layer prefetch (Fate-style)**
   for DSV4 — the reuse it exploits does not exist in this router.
4. **Do not ship in-situ prefetch**: measured net-neutral-to-harmful
   at every capacity (miss deltas −0.4% / +1.8% / +0.3% at cap-8/16/32;
   70–88% of prefetched bytes wasted). This held *after* removing two
   accounting bugs and one causality leak that had manufactured the
   earlier optimistic numbers.
5. The only untested upside is the **cross-token (router-ahead)**
   scheduler: prediction quality exists (recall@12 = 0.50, lead = 43
   layers) but requires running the router ahead of expert fetch — a
   runtime design change that must be evaluated with a real runtime
   benchmark, out of scope for this branch.
6. If expert capacity is ever scarcer than cap-32, remember the
   two-tier fact: at cap-8, tier-2 host-pack cuts SSD traffic 38% but a
   better policy delivers the same SSD relief at **3.1× less H2D
   bandwidth** — policy quality is PCIe bandwidth.
7. Re-run this study on a **longer trace (≥ 256 decode forwards)**
   before treating any number here as steady-state.

## 4. Claim discipline (per brief)

- No end-to-end TPS improvement is claimed anywhere; no runtime
  benchmark was executed.
- SpecMD's and Fate's published gains are **not** assumed to transfer;
  both were tested here and **did not** (Least-Stale ≈ LRU; keep-value
  ≡ LRU; same-token cross-layer features hurt recall).
- Theoretical stall-hidden time is reported only as a bandwidth-sweep
  ceiling, labeled THEORETICAL (`results/prefetch_*.json`
  `stall_sweep_note`).
- Every number in the tables above is tagged measured / simulated /
  theoretical; hypotheses are marked as such (§5 of
  PREDICTOR_RESULTS.md).
- Six honest errors (including one causality leak that inflated
  prefetch quality and two Phase E accounting bugs) are documented
  with their fixes in `PREDICTOR_RESULTS.md §6`; all reported numbers
  postdate the fixes, and the regression tests now cover each failure
  mode (20 tests passing).

## 5. Deliverables

| item | path |
|---|---|
| Phase A audit | `CACHE_POLICY_AUDIT.md` |
| Predictor results | `PREDICTOR_RESULTS.md` |
| Simulator code | `sim/{trace,policies,engine,predictor,keepvalue,prefetch}.py` |
| Reproducible CLIs | `run_tournament.py`, `run_prefetch.py`, `phase_a_profile.py`, `make_plots.py` |
| Raw results | `results/*.csv`, `results/*.json` |
| Plots | `results/misses_vs_capacity.png`, `results/predictor_recall.png`, `results/prefetch_effect.png` |
| Tests | `tests/test_sim.py` (20 tests, all passing) |

```bash
# full reproduction
python research/cache-predictor/phase_a_profile.py
python research/cache-predictor/run_tournament.py --tune
python research/cache-predictor/run_prefetch.py
python research/cache-predictor/make_plots.py
python -m pytest research/cache-predictor/tests -q
```
