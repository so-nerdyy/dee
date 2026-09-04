# CAUSALITY_AUDIT.md — what lookahead is legally available to exact staging

Branch: `research/exact-staging` (worktree `../dee-staging`, base `2db0fde`) · Date: 2026-09-04
Track: OFFLINE exact storage/staging research. The OFFICIAL router is authoritative.
No production code modified; prototype lives under `dee.cpp/experiments/exact_staging/`.

---

## 1. How the engine learns official routes (measured, from source)

`dee.cpp/src/engine.cpp`:

- `prepare_fp4_experts(...)` (line ~2754) is invoked **per layer, at the top of
  that layer's MoE forward**, with that layer's official expert ids.
- The fill path reads the expert store for ids not already resident and blocks
  the layer until records are packed and copied (`get_batch`, H2D in
  `get_staging_fp4`).
- Layer L+1's official route is produced by the router from layer L's output.
  **Layer L+1's route cannot exist before layer L's MoE output is computed.**
  Route authority is strictly sequential (verified: `avail_lay...` handoff and
  the per-layer `prepare_fp4_experts` call sites; the route journal's
  `chain_sha256` chaining confirms the same order).

## 2. Classification of candidate leads (Phase C)

| Lead | Classification | Reason |
|------|----------------|--------|
| 1 layer (in-step) | **IMPOSSIBLE_WITHOUT_FUTURE_ROUTE** | Layer L+1's official ids do not exist until layer L's MoE output is computed. No amount of engineering exposes them earlier; only a predictor would, and predictor-selected experts are forbidden in exact mode. |
| 2 / 4 / 8 / 16 layers (in-step) | **IMPOSSIBLE_WITHOUT_FUTURE_ROUTE** | Same dependency, strictly worse. The critical-path simulator's "lead ≥ 8 → −26.1%" is **not causally implementable** in exact mode; it requires future official routes (or a prediction, which is out of contract). |
| Cross-step retention (step k−1 routes → step k) | **IMPLEMENTABLE_WITH_CURRENT_INFORMATION** | Routes for step k−1 are fully known while step k runs. Production *already* implements this: the bounded host-pack LRU (8.5 GiB/GPU) retains records across steps (`host_pack_cache_bytes`, engine.cpp:3579). |

> **This corrects the previous track's simulated headline**: the −26.1% "lead ≥ 8"
> scenario cannot be built without violating exact-mode causality. The legal
> equivalent that production does not yet fully exploit is **pack capacity**,
> not staging lead (see STAGING_DESIGN.md).

`REQUIRES_PIPELINE_RESTRUCTURE` would apply only if dee executed multiple
layers' routing per pass (it does not); no such restructure is identified that
exposes earlier *official* routes.

## 3. The legal lever, quantified (validated replay, sealed v65 journal)

Replay of the production host-pack LRU over the sealed journal's per-device
fill stream (unique (layer, expert) per forward step, execution order, pack
state carried from prefill into decode) **reproduces the sealed v65 counters
within 2 records per GPU**:

| Quantity | Replay (LRU @ 682 rec = 8.5 GiB/GPU) | Sealed v65 |
|---|---|---|
| cuda:0 pack misses / evictions | 1388 / 706 | 1390 / 708 |
| cuda:1 pack misses / evictions | 1092 / 410 | 1091 / 409 |
| decode storage misses (both GPUs) | 1251 | 1252 |
| per-step decode reads | within ±2 of sealed every step | — |

Production pack semantics are therefore **confirmed LRU at 682 records**.

Budget sweep (decode storage misses, both GPUs):

| Pack budget | Decode misses | vs sealed (1252) |
|---|---|---|
| 682 rec / 8.5 GiB (sealed) | 1251 | — |
| 851 rec / 10.6 GiB | 1187 | −5.1% |
| 1024 rec / 12.75 GiB | 1153 | −7.9% |
| **1365 rec / 17.0 GiB** | **1135** | **−9.3% = oracle floor** |

- The **oracle floor** (offline Belady on the fill stream) at the *sealed*
  682-record budget is also 1135 decode misses: at fixed budget, a perfect
  offline eviction policy gains exactly what +8.5 GiB of budget gains. LRU at
  the sealed budget is within **9.3%** of the offline optimum; no exact
  eviction policy can beat that bound on this trace.
- Ablations that *drop* prefill records or clear the pack at decode start are
  strictly worse (1636 / 1791 misses): prefill-warmed pack entries are
  genuinely reused in decode.

## 4. Causal recency facts on this trace (labels)

- MEASURED (journal): per decode step, 258 official requests; sealed resident
  hits 655 total; host-pack hits 2618 total; storage misses 1252 total.
- MEASURED (journal): per-GPU distinct demanded records over the whole decode:
  cuda:0 1297 (16.15 GiB), cuda:1 1067 (13.29 GiB) — the 3-step decode working
  set exceeds the 8.5 GiB pack, which is why the budget helps.
- HYPOTHESIS (needs live A/B): the pack budget increase converts directly into
  fewer stall-bearing reads because the disk is the calibrated dominant stage
  (research/exact-critical-path CALIBRATION.md).

## 5. What this audit forbids

- Any exact-mode staging driven by predicted experts (SpecMD/Fate-style
  candidate sets) — out of contract for this track.
- Claiming the simulator's in-step lead scenarios as implementable.
- Claiming pack-budget wall gains as measured (they are simulated estimates
  until Codex's matched A/B seals them).
