# Phase-4 cache-hierarchy replay simulator

`p4_replay_sim.py` replays the authoritative route journals written by
dee.cpp's instrumented engine and reproduces the two-level expert-cache
event model — VRAM residency + host pack cache — counter-for-counter
against the sealed GPU-2 evidence.  It exists so Phase-4 cache questions
(policy, budget, consultation mode, persistence) can be answered without
spending GPU runs, while staying bug-compatible with the engine it
models.

Standard library only.  Requires Python ≥ 3.9 (developed on 3.12).
No pip, no pytest, no third-party imports.

```
python3 tools/phase4/p4_replay_sim.py --help
```

## What it models

For every journal record (`(forward_step, layer)` = one MoE call):

1. `active = sorted(set(flatten(expert_ids_rank_order)))`, `K = len(active)`.
2. `active` is split into `cache_batch` chunks (`cache_batch` = VRAM slot
   capacity — the engine stages one chunk at a time).
3. For each chunk, in order:
   a. **Host consult** (`eager` mode, the live configuration): every key
      `(layer, expert)` in the chunk is consulted against the host pack
      cache in `queue_depth` sub-batches (`get_batch`).  Misses admit at
      MRU *after* the sub-batch scan; in-batch duplicates count as hits;
      eviction never removes a key requested in the current sub-batch.
   b. **Staging**: each expert in chunk order performs an *uncounted*
      host `get_if_present` LRU refresh, then `ensure()` on the VRAM
      cache — resident ⇒ hit (tick++, refresh `last_used`, **replace**
      priority, pin++), nonresident ⇒ evict unpinned blocks until free,
      insert, pin, count a cold load.
   c. **Consume**: all pins held by the chunk are released.
4. `priority` for request `i` of a call is `K - i` (rank 1 = highest).

The bug-compatible engine VRAM score is

```
score = last_used + priority * 2**20      # min score evicted first
```

(`PRIORITY_WEIGHT = 2**20`; priority dominates recency by ~1M ticks.)

### Host pack cache semantics (host_pack_cache.cpp)

- Capacity is in bytes → records (`budget // 13,369,344`); 8.5 GiB ⇒ 682.
- `get_batch`: scan order, in-batch dup = hit (no move), resident = hit
  (move to MRU), else miss; after the scan, evict LRU while over
  capacity skipping keys in the current batch; admit misses at MRU in
  request order.
- Per staging step the engine does an *uncounted* `get_if_present`
  refresh — it moves a resident entry to MRU without counting a consult.
- Host misses == store reads in the sealed runs (each miss = one source
  materialization request).
- `--host 0` models a bypass: no consults, every VRAM cold load reads
  the store directly.

## Journal format & validation

Input: route-journal JSONL, one JSON object per line.  Required fields:
`forward_step, layer, phase, device, start_pos, token_rows, topk,
expert_ids_rank_order, record_index, run_id`.

Validation is fail-closed (`SimError` on any violation):

- nonempty file, one JSON object per line, required fields present
- `topk == 6`; `len(expert_ids_rank_order) == token_rows`; every rank
  row has exactly 6 integer expert IDs in `[0, 255]`
- `layer` in `[0, num_layers)`; device consistent with
  `layer < layer_split` (`cuda:0`) vs `>=` (`cuda:1`)
- `phase` ∈ `{prefill, decode}`; `forward_step == 0` ⇒ prefill, else decode
- `record_index` contiguous and ordered; no duplicate
  `(forward_step, layer)`; every forward covers all `num_layers` layers
- each journal's SHA-256 is recorded in every output row and metadata

## Quickstart

```bash
cd dee.cpp
B=benchmark_reports/deepseek-v4-flash-0731-t4/gpu2-phase3-fullstore

# Anchor gate — must print "ANCHOR GATE: PASS"
python3 tools/phase4/p4_replay_sim.py \
  --journals $B/routed_experts-a1-q0.jsonl \
             $B/routed_experts-a1-q1.jsonl \
             $B/routed_experts-a1-q2.jsonl \
  --prompt-order q0,q1,q2 \
  --vram-policy priority_lru --persist-across-prompts \
  --anchors $B/result-a1-q0.json $B/result-a1-q1.json $B/result-a1-q2.json

# fp4 evidence anchor (queue depth 8)
python3 tools/phase4/p4_replay_sim.py \
  --journals benchmark_reports/deepseek-v4-flash-0731-t4/v58-evidence-20260901T025855Z/routed_experts.jsonl \
  --vram-dtype fp4 --queue-depth 8 --vram-policy priority_lru \
  --persist-across-prompts \
  --anchors benchmark_reports/deepseek-v4-flash-0731-t4/v58-evidence-20260901T025855Z/result.json

# Policy sweep at fp4 (281 slots)
python3 tools/phase4/p4_replay_sim.py --journals $B/routed_experts-a1-q0.jsonl \
  --persist-across-prompts --vram-dtype fp4 \
  --sweep vram_policy=lru,priority_lru,lfu,freq_x_recency,segmented,layer_lru,static_hotset+dyn,belady \
  --static-hotset auto:0.25 \
  --out out/fp4_policy.csv --per-forward out/fp4_fwd.csv --json-out out/meta.json

# Host budget sweep
python3 tools/phase4/p4_replay_sim.py --journals $B/routed_experts-a1-q0.jsonl \
  --persist-across-prompts --sweep host_gib=0,4,8.5,17,32
```

## CLI reference (key flags)

| flag | default | meaning |
|---|---|---|
| `--journals` | — | JSONL paths in prompt order (repeatable) |
| `--prompt-order` | filename | comma labels for `--journals` |
| `--persist-across-prompts` | off | caches carry across prompts; counters reset per prompt (GPU-2 faithful) |
| `--vram` | 3.5 | VRAM budget GiB → `floor(GiB/blob)` slots (fp16 48 MiB → 74; fp4 12.75 MiB → 281) |
| `--vram-dtype` | fp16 | `fp16` 48 MiB/block · `fp4` 12.75 MiB/block |
| `--vram-policy` | priority_lru | `lru` · `priority_lru` (bug-compat engine) · `lfu` · `freq_x_recency` · `layer_lru` · `segmented` · `static_hotset+dyn` · `belady`/`min` |
| `--host` | 8.5 | host budget GiB per scope unit; `0` = bypass |
| `--host-scope` | per-device | one cache per device, or one `pooled` |
| `--host-policy` | lru | `lru` (engine) or `belady` (offline bound on the consult stream; requires `eager`) |
| `--host-consult` | eager | `eager` = whole chunk before staging (live config) · `lazy` = only VRAM misses consult |
| `--queue-depth` | 6 | host `get_batch` sub-batch size (v58–v60 used 8) |
| `--regime` | cold | `cold` or `prewarm` (needs `--prewarm`) |
| `--prewarm` | — | `auto:FRAC` or file of `layer,expert` lines — declared initial state |
| `--static-hotset` | — | `auto:FRAC` or file — static partition for `static_hotset+dyn` |
| `--sweep` | — | `key=v1,v2` repeatable; keys: `vram_gib, vram_dtype, vram_policy, host_gib, host_scope, host_consult, queue_depth, regime` |
| `--anchors` | — | sealed `result-*.json` for the counter gate |
| `--anchor-tol` | 0 | allowed absolute counter delta |
| `--out` / `--per-forward` / `--json-out` | — | summary CSV / per-forward CSV / run metadata JSON |

## Policies

| policy | semantics |
|---|---|
| `lru` | pure `last_used` recency |
| `priority_lru` | engine score `last_used + priority·2^20` — **required for anchor gates** |
| `lfu` | min `(freq, last_used)` evicted |
| `freq_x_recency` | `last_used + freq·W` (`--freq-weight`, default 2^20) |
| `layer_lru` | per-layer soft partitions (capacity split across the device's layers); evicts only under global pressure, preferring over-share partitions, then the global LRU; never evicts pinned |
| `segmented` | SLRU probationary/protected (`--segmented-probation`, default 0.2); segment sizes are preferences — probation may transiently overflow while its members are pinned in flight |
| `static_hotset+dyn` | declared static set never evicted; dynamic LRU borrows unused static slots; needs `--static-hotset` |
| `belady` / `min` | offline MIN/OPT on the known request stream — runnable policy **and** the dominance bound |

Partition capacities are *soft eviction preferences*, not hard walls: a
whole in-flight chunk can pin more members than any sub-partition, so
hard partitions would deadlock against the engine's pin contract.  The
global slot cap is always enforced and pinned blocks are never evicted.

### Belady / dominance tripwire

Every run also simulates an unpinned Belady shadow cache per device and
raises `DOMINANCE VIOLATION` if any candidate policy ever exceeds the
bound — the tripwire that caught the earlier causality bug class.  The
host tier gets the same treatment: an offline MIN bound over the
recorded consult stream per scope (and a runnable `--host-policy belady`
for `eager` consult).  The `belady` output column reports the per-device
bound; `--vram-policy belady` reproduces it exactly.

## Persistence

`--persist-across-prompts` (the GPU-2 behavior): VRAM and host contents,
priority/last-used state, and the logical tick carry across prompts;
per-prompt engine counters reset; host-pack counters stay cumulative —
the anchor comparator sums simulator rows when comparing against the
sealed process-cumulative host fields.  Without the flag each prompt
replays against fresh caches.

## Correctness checks (all fail-closed)

- request closure `requests == vram_hits + vram_cold (+inflight)`
- host consult closure: eager ⇒ consults == requests; lazy ⇒ consults ==
  vram_cold; bypass ⇒ 0
- joint pairing `vram_miss_host_hit + vram_miss_host_miss == vram_cold`
- cold-regime first-touch invariants (a first-seen key cannot hit)
- per-layer and per-forward sums equal prompt totals
- VRAM/host capacity never exceeded; pinned blocks never evicted
- dominance: policy hits ≤ Belady bound (VRAM and host)

## Outputs

Console table plus, on request, `summary.csv` (per prompt × device +
pooled), `per_forward.csv` (per forward × device + pooled), and
`meta.json` (config, journal paths + SHA-256, capacities, Belady
bounds).  Rows include per-phase splits, joint counters, store/H2D byte
accounting, per-token bandwidth, reuse-distance p50/p90/p99 and a reuse
CDF at `REUSE_CDF_CAPS` slot counts.

Reference outputs from the sealed corpus live in `out/`:

- `out/fp16_anchor/` — 3-prompt fp16 anchor run + gate log
- `out/fp4_policy/`, `out/fp16_policy/` — 8-policy sweeps
- `out/host_budget/` — fp16 host capacity sweep
- `out/anchors/` — all six anchor-gate logs (q0–q2, v58–v60)

## Anchors validated (all exact, tol=0)

| evidence | config | result |
|---|---|---|
| `gpu2-phase3-fullstore` q0, q1, q2 | fp16, 74 slots, host 682, qd=6, eager, persist | PASS, every counter +0 |
| `v58`, `v59`, `v60` evidence | fp4, 281 slots, host 682, qd=8, eager, persist | PASS, every counter +0 |

Anchor semantics handled by the comparator: `engine_stats` is per-prompt;
`host_pack` and `per_token_accounting[0]` host/store fields are
process-cumulative; `per_token_accounting[s>0]` fields are per-step
deltas.

## Findings (q0, 5,512 pooled requests)

VRAM policy, fp4 / 281 slots, pooled hits (Belady bound 2750):

| policy | hits | hit rate |
|---|---|---|
| static_hotset+dyn (auto:0.25) | 1983 | 36.0% |
| layer_lru | 1889 | 34.3% |
| lfu / freq_x_recency | 1877 | 34.1% |
| lru | 1833 | 33.3% |
| **priority_lru (engine)** | **552** | **10.0%** |
| segmented | 0 | 0.0% |
| belady | 2750 | 49.9% |

fp16 / 74 slots, pooled hits (bound 1742): static_hotset+dyn 473 (8.6%),
priority_lru 162 (2.9%), layer_lru 18, everything else 0 — the small
arena thrashes under pure recency/frequency, and the engine's priority
term actually *helps* here by protecting high-rank experts.

Host budget, fp16 priority_lru, per-device eager (pooled e2e store-avoid):

| host GiB | host hits | e2e |
|---|---|---|
| 0 (bypass) | 0 | 2.9% |
| 4 | 1908 | 34.6% |
| 8.5 (live) | 2532 | 45.9% |
| 17 | 2805 | 50.9% |
| 32 | 2808 | 50.9% |

Host LRU saturates ≈17 GiB; the offline bound at 8.5 GiB (1347/1461
per-device hits) matches LRU at 32 GiB — i.e., LRU needs ~4× the budget
to reach MIN on this stream.  `lazy` consult costs ~156 pooled host hits
vs eager at 8.5 GiB (2376 vs 2532) because VRAM-hit experts never refresh
the host tier.

## Known deviations & counterfactuals

- `lazy` consult, `host belady`, `pooled` scope, non-default queue
  depths, non-`priority_lru` policies, and `prewarm`/`static_hotset`
  declared states are *counterfactuals* — they model configs the engine
  could run, not what the sealed runs did.  Only
  `priority_lru` + per-device + eager + qd 6/8 + persist is
  anchor-validated.
- Under `belady` host, a consulted key can be legitimately absent at
  staging (MIN declined to cache it); the engine-faithful stage-presence
  check therefore applies to `lru` only.
- Partition policies use soft shares (see above) — a deviation forced by
  the pin contract, not an approximation we chose.
- In-batch duplicate consult hits under `lru` follow the engine exactly
  (hit without move); under `belady` host they are sequential consults.
- Reuse distances are per-prompt (cross-prompt reuse under persistence
  is not folded into the distribution).
- a0 journals are byte-identical to a1 on the sealed corpus — a1 is the
  canonical input.
