# CACHE1.1 — Expert Access-Pattern Analysis (sealed DS10 v12 trace)

Zero Kaggle time: pure local analysis of the sealed
`ds10-v12-accept-dual-t4-decode` evidence (`token_trace`, 16 tokens, 43
layers, 6 routed + 1 shared expert per layer call).

## 1. Volume

| Metric | Value |
|---|---|
| Tokens | 16 (1 prefill of 7 positions + 15 decode) |
| Layers | 43 (split 22/21 across GPU0/GPU1) |
| Expert accesses (incl. shared) | 6,364 |
| Unique (layer, expert) pairs | 2,365 |
| Accesses per token (decode) | 43 × 7 = 301 |
| Shared-expert accesses | 688 (10.8% of all) |

## 2. Reuse structure — the deciding question

- **Consecutive-token reuse (same layer, t→t+1): 37.5%** of the 6 expert
  slots overlap between adjacent tokens. So reuse *exists*.
- **Within-layer revisit: 155%** of unique slots are used by ≥2 tokens.
- **Popularity: top-5% of (layer,expert) pairs cover 23.6% of accesses**
  (vs 5% under uniformity) — a mild power-law tail, not flat, not extreme.

**Conclusion: this is NOT a uniform-routing dead end.** Reuse is
structural (37.5% consecutive) but spread over a huge working set
(2,365 × 48 MiB ≈ 111 GiB if held resident). Caching can capture only the
*temporal* reuse; the rest must be prefetch/overlap.

## 3. Why DS10 measured 0% hits

Simulation (`cache1_policy_sim.py`) now reproduces the sealed numbers
exactly: current policy → **0.00% GPU hits, 100% of accesses fetched over
HTTP**. Mechanism:

1. Budget 2 GiB/GPU = 42 experts (48 MiB each).
2. Prefill (token_0) alone touches ~40 distinct experts per layer —
   ~1,700 unique experts — completely churning the 42-slot cache before
   decode starts.
3. During decode, the reuse distance (same layer revisited ~300 accesses
   later) far exceeds 42 slots, so LRU retains nothing.
4. Provider host LRU disabled (`raw_experts_per_layer=0`): every miss =
   full HTTP range fetch (w1+w2+w3 + scales). Measured: 10,202 fetches /
   298 GiB H2D in the primary run.

## 4. Policy simulation results (per-GPU split, 2-level cache)

| Policy | GPU hit | Provider hit | HTTP fetch % |
|---|---|---|---|
| **DS10 as-run** (2 GiB, raw=0) | 0.00% | 0.00% | **100.0%** |
| + provider raw=8 (2 GiB) | 0.00% | 32.6% | 67.4% |
| + provider raw=16 (2 GiB) | 0.00% | 46.7% | 53.3% |
| + pin shared (2 GiB, raw=8) | 11.1% | 23.8% | 65.1% |
| + pin shared (8 GiB, raw=16) | 33.6% | 13.5% | 52.9% |
| + pin shared + LFU (8 GiB, raw=16) | 27.5% | 19.8% | 52.8% |
| + pin shared + pop2 (4 GiB, raw=8) | 24.6% | 16.3% | 59.0% |

Belady (oracle) GPU-cache ceilings per GPU:
- 2 GiB: 20–22% · 4 GiB: 33–41% · 8 GiB: 46–56% · 12 GiB: 52–60%

## 5. Honest achievable target

**≥70% GPU hit rate is NOT reachable on this 16-token trace** — even
oracle eviction tops out at 56–60% at 12 GiB/GPU (which itself busts the
14 GiB memory ceiling). The campaign's ≥70% assumes a longer trajectory
with more reuse; on the sealed canonical trace the *reachable* outcome is:

- **Combined GPU+provider (no HTTP) ≥ 47%** at 8 GiB with pinned shared +
  raw=16 — a 2× reduction in HTTP fetches vs 100% today.
- GPU hit rate **~34%** at 8 GiB (vs 0% measured).

Primary recommendations (justified by these numbers):
1. **Re-enable provider raw LRU (raw=16)** — cheapest, −47% HTTP fetches.
2. **Pin shared experts** — 43 guaranteed residents, +11% GPU hits, removes
   688 fetches of the most frequent expert.
3. **Raise GPU budget to 8 GiB/GPU** where the memory plan permits.
4. **Router-ahead prefetch** of the next token's likely experts (the 37.5%
   consecutive-reuse set) — overlaps HTTP latency with layer compute.

## 6. Verdict path

`ACCEPT_CACHE_HITRATE_TARGET` (≥70%) is not achievable on the sealed
trace; the campaign should terminate with **`ACCEPT_CACHE_PARTIAL`** once
the implemented stack shows a *measured* multi-fold reduction in HTTP
fetches and a GPU hit rate well above 0%, with all CACHE1.4 correctness
checks passing (deterministic rerun, cold==warm, budget invariance,
identical token IDs vs sealed DS10).

Artifacts: `cache1_analysis.py`, `cache1_policy_sim.py` (this directory).
