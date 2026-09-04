# CACHE_POLICY_AUDIT.md — DSV4 expert cache policy tournament

Branch: `research/cache-predictor` · Worktree: `../dee-cache` · Date: 2026-09-03/04
Status: **offline research, trace-simulated only — no runtime/TPS claims**

---

## 1. Evidence provenance (Phase A)

| Field | Value |
|---|---|
| Source trace | `dee.cpp/tmp/v65-terminal-fetch-20260903T0202Z/routed_experts.jsonl` (sealed run directory; **read-only** in this project) |
| Run id | `p2.3-dee4-storage-readahead-ab1-candidate-q6l6-v65` |
| Journal schema | v1, hash-chained (`chain_sha256`/`previous_chain_sha256`), 688 links, **0 broken links, 0 malformed records** |
| Forwards (tokens) | 16 (1 prefill step of 7 token rows + 15 decode steps of 1 row) |
| Layers | 43 (0–42) |
| Experts/layer | 256 |
| top-k | 6 (rank order preserved) |
| Total routed selections | 5,676 (1,806 prefill + 3,870 decode) |
| Unique (layer, expert) pairs | 2,364 |
| Record size | **13,369,344 bytes (uniform)** per `dee4-metadata.json` (`record_bytes`), dee4-v3-trace, deepseek-fp4-e2m1-e8m0 codec |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Trace integrity sidecars | `result.json` / `integrity.json` report `route_journal_complete: true` |

**Trace dedup:** all 19 `routed_experts.jsonl` copies under `dee.cpp/tmp`
(v50 … v65, kt-input-capture-v3) were diffed record-by-record: **all carry
byte-identical routing content**. They are re-fetches/re-runs of the same
sealed journal (run ids differ in metadata only). The evidence set is
therefore **one trace**, and the newest copy (v65) is used. Inventory:
`results/phase_a_trace_inventory.json`.

No synthetic routes were used anywhere in this study.

### Known limitations of the evidence
- 16 forwards is a **very short window** (15 decode steps per layer).
  All results are trace-local; nothing here establishes steady-state
  behavior over hundreds of tokens.
- The 15 decode steps repeat one prompt context; working-set structure
  may be prompt-specific.
- Single request, single session, dual-T4 capture path.

---

## 2. Measured structure of DSV4 routing (trace statistics, not simulation)

These measurements drive everything below:

| Statistic | Value | Implication |
|---|---|---|
| Same-token adjacent-layer expert overlap (decode) | **mean 0.13 of 6** (630/630 pairs: 551×0, 77×1, 2×2) | Fate-style same-token cross-layer sharing is essentially absent |
| Experts appearing in an earlier same-step layer | **0 of 3,870** decode selections | DSV4 re-routes from scratch per layer; no within-step reuse at all |
| Adjacent-step same-layer overlap (decode) | **mean 2.03 of 6**, spread 0–6 | Token-recurrence is the dominant reuse signal |
| Top-16 most-demanded experts per layer | capture **62.7% of demand on average** (min 27.3%, max 83.3%) | Moderate popularity skew, not enough for a static hot set |
| Decode distinct experts per layer (15 steps) | mean **37.7**, max 77 | Per-layer working sets exceed dee's cap-8 control ~5× |
| Prefill(step0) vs step-1 expert overlap | 161/161 step-1 experts already seen in prefill | Warm-start from prefill is real but prefill itself demands all 256/layer |

---

## 3. Method

- Simulator: `research/cache-predictor/sim/` (pure Python, offline).
- **Semantics match dee:** capacity is **per layer** (`LayeredEngine`):
  one independent cache per transformer layer over that layer's demand
  slice — mirroring M4's "capacity-N = N experts resident per layer per
  GPU". A **global-pool** engine is also provided; it is *not* used for
  headline numbers because it does not match dee's runtime accounting.
- Cache keys are (layer, expert) pairs; record bytes are uniform today
  but the engine supports **byte-aware capacity** with variable-size
  records (`set_bytes`) for future codecs.
- Metrics per config: hits/misses, compulsory vs capacity misses,
  evictions, miss bytes/token, SSD bytes/token, H2D bytes/token,
  evictions/token, churn, useful/wasted residency ticks, regret vs
  Belady. Hit rate is reported but explicitly **not** the objective;
  miss bytes and evictions are the cost proxies.
- 16 forwards = 16 "tokens" for the per-token normalizations.
- Tier-2 (host RAM) is modeled as an equal per-layer LRU slice
  (dee's `host_pack` is global; this approximation is noted).

Policies: LRU, LFU, ARC (Megiddo–Modha, count-faithful for uniform
records), Least-Stale (SpecMD-inspired staleness ratio), recency×freq
hybrid, **Belady offline oracle** (per-layer next-use chains; validated
optimal on synthetic sequences and never beaten by any online policy),
keep-value (Phase D; see §5.1 and PREDICTOR_RESULTS §6 for why its
tuned form degenerates to LRU).

Correctness gates: `tests/test_sim.py` (20 tests) include OPT-domination
on 5 synthetic sequences, capacity invariants, byte-aware eviction,
residency accounting hand-cases, prefetch accounting closure
(issued = useful + wasted + still-live), and **predictor causality**
checks.

---

## 4. Tournament results (sealed v65 trace)

### 4.1 Demand misses by capacity (tier-2 off), 43 layers × 16 forwards

| cap/layer | LRU | LFU | ARC | Least-Stale | RecFreq | KeepValue (tuned) | Belady |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8  | 3839 | 4782 | 3856 | 3894 | **3789** | 3839 | 2849 |
| 16 | 2955 | 3832 | 2985 | 3035 | 2957 | 2955 | 2397 |
| 24 | 2594 | 3172 | 2646 | 2666 | 2620 | 2594 | 2364 |
| 32 | 2490 | 2741 | 2488 | 2488 | **2483** | 2490 | 2364 |
| 48 | 2398 | 2426 | 2398 | 2398 | 2398 | 2398 | 2364 |
| 64 | **2375** | 2376 | 2377 | 2377 | 2377 | **2375** | 2364 |
| 96 | 2364 | 2364 | 2364 | 2364 | 2364 | 2364 | 2364 |

Exact machine-readable grid: `results/tournament_v65-terminal-fetch-20260903T0202Z.{csv,json}`.

Key observations (all simulated, trace-local):

1. **Belady's ceiling = 2,364 misses at cap ≥ 24.** That equals the
   number of unique (layer, expert) pairs: with ≥24 slots per layer,
   every *capacity* miss is avoidable; the residual misses are exactly
   the compulsory ones. No policy can go below 2,364 without changing
   what is routed.
2. **Cap-8 (dee's measured M4 control point) is where policy matters —
   and where headroom is real but hard to capture.** Belady does
   3,839 → 2,849 (−25.8%) vs LRU, but **the best practical online
   policy found (recency×freq) captures only −1.3%** (3,789). The
   large LRU→Belady gap at cap-8 is mostly *not* harvestable by the
   online policies tested here.
3. **At cap ≥ 32 all practical policies converge** (spread ≤ 0.3%).
   The M4 sealed observation that cap-32 reaches ~100% post-warmup hit
   rate is consistent with this: cap-32 already covers the per-layer
   decode working set, so replacement policy is nearly irrelevant.
4. **LFU is the worst** practical policy, badly (＋943 misses vs LRU at
   cap-8, +24.6% miss bytes): pure frequency is anti-correlated with
   DSV4's token-recurrence structure at scarce capacity.
5. **Least-Stale ≈ ARC ≈ LRU** everywhere (differences ≤ 55 misses,
   sign varies). Neither SpecMD-inspired staleness nor ARC's adaptive
   pivot materially helps DSV4 beyond plain LRU. This is a *negative*
   result and it is trace-local, but it is consistent across all 28
   geometry points run.
6. **Regret vs Belady at cap-8:** LRU 990, RecFreq 940, ARC 1007,
   Least-Stale 1045, LFU 1933, KeepValue 990 (≡ LRU), Belady 0.
   Practical policies capture ≈ 0–5% of the available improvement.

### 4.2 Cost metrics at the M4 control point (cap-8, tier-2 off)

| policy | miss bytes/token | SSD bytes/token | evictions/token | churn | useful res. ticks | wasted res. ticks |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 3,059.2 MiB | 3,059.2 MiB | 218.4 | 1.292 | 11,116 | 32,730 |
| RecFreq | 3,019.4 MiB | 3,019.4 MiB | 215.3 | 1.274 | 13,196 | 30,650 |
| LFU (worst) | 3,810.7 MiB | 3,810.7 MiB | 277.4 | 1.624 | 11,307 | 32,539 |
| Belady | 2,270.3 MiB | 2,270.3 MiB | 156.6 | 0.943 | 29,070 | 14,776 |

(Per-token = per forward step across all 43 layers; uniform 13,369,344 B
records, so miss count, miss bytes and eviction count rank identically.
Under uniform size the cost ordering equals the miss ordering above.
Belady roughly **triples the useful share of residency** — the oracle
wins by keeping the right experts, not by churning less per se.)

### 4.3 Two-tier (RAM + VRAM) findings

Tier-2 host-pack **does not change miss counts** (correctly — same
demand sequence); it re-sources misses. At cap-8 with tier-2 = 64
experts/layer:

| policy | SSD MiB/token | H2D MiB/token |
|---|---:|---:|
| LRU t2=0 | 3,059.2 | 0 |
| LRU t2=64 | 1,893.4 (−38.1%) | 1,165.8 |
| Belady t2=0 | 2,270.3 | 0 |
| Belady t2=64 | 1,891.8 | **378.5** |

**Policy quality determines how much H2D you must spend to unload the
SSD**: at equal SSD offload, the oracle needs **3.1× less host→device
traffic than LRU**. A better policy is worth real PCIe bandwidth.

---

## 5. Answers to the Phase-B key questions

1. **How much improvement over LRU is actually available?**
   At cap-8: up to −25.8% misses (Belady oracle), but the best
   *practical online* policy found captures only −1.3% (recency×freq).
   At cap-32: ~0.3% total spread — essentially nothing is available.
   The headroom exists at scarce capacity; harvesting it needs
   information the tested online policies do not have (see Phase D/E).
2. **How close can practical policies get to Belady?**
   At cap-8: LRU regret 990, best practical regret 940 — ~5% of the
   gap closed. At cap ≥ 48 everything is within ~1.4% of Belady.
3. **What policy minimizes MISS COST rather than hit rate?**
   Under uniform record size and per-layer capacity, miss count,
   miss bytes, and eviction count rank identically, so the ordering
   above is also the cost ordering. This equality breaks under future
   variable-size codecs (supported by the simulator's byte-aware mode
   but not exercised by this trace — all records are exactly
   13,369,344 bytes).
4. **Does Least-Stale materially help DSV4? No.** Within ±1.4% of LRU
   at every point; worse at scarce capacity.
5. **Does ARC materially help DSV4? No.** Statistically
   indistinguishable from LRU across the sweep (its published
   advantages do not transfer to this workload shape: 43 tiny
   independent caches with token-recurrence skew, not one large cache
   with strong frequency skew).

### 5.1 Phase D (cost-aware keep-value) result: degenerates to LRU

The KeepValue policy (`sim/keepvalue.py`) was tuned on even steps and
validated on odd steps at cap-8. **After a causality leak in the
candidate provider was found and fixed** (see PREDICTOR_RESULTS §6),
tuning selects **staleness-only weights (0,0,1,0)** — which ranks
victims exactly like LRU (evict largest age) — and the tuned policy
equals LRU at *every* capacity (identical miss counts in §4.1). The
predictor-rank-only variant is worse on train (2,791 vs 2,719 misses).
Earlier claims that keep-value beat LRU by −11.2% at cap-8 were an
artifact of that leak and are **retracted**.

---

## 6. Threats to validity

- **Trace length (16 forwards)**: the single largest threat. All deltas
  are trace-local percentages, not steady-state rates.
- Prefill dominance: 1,806 of 5,676 selections are the single prefill
  step; per-layer demand sequences are short (128 selections each).
- The per-layer tier-2 slice approximation vs dee's global host pack.
- Residency "ticks" are demand indices, not milliseconds.
- No runtime benchmark was executed; nothing here measures TPS.

## 7. Reproduction

```bash
python research/cache-predictor/run_tournament.py --tune
python research/cache-predictor/phase_a_profile.py
python -m pytest research/cache-predictor/tests -q
```

Raw outputs: `results/tournament_*.csv|json`, `results/phase_a_trace_inventory.json`.
Plots: `results/misses_vs_capacity.png`.
