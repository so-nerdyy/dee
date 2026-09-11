# CAUSALITY_AND_INITIAL_STATE_AUDIT.md — how b7b9c7f let an impossible result exist, and how it is now prevented

**Verdict up front.** The initial Phase-2 result (b7b9c7f) contained a
causality / initial-state accounting bug in the simulator's count-based
policies. It credited residency that never existed (an uncharged PREWARM)
inside comparisons that were labeled cold-start. The impossible headline —
"a realizable cold-start policy (static pin, 91.4 %) beats offline
Belady/MIN (53.6 %)" — is retracted. Corrected numbers, same sealed trace,
are below and in `PHASE2_POLICY_MATRIX.md` v4.

---

## 1. The bug, mechanically

In `tools/phase2_ws_policy_sim.py` (v3), the count-based caches decided
hit/miss by **set membership of the pin set**, which was built from the
*whole-trace* frequency counts before the simulation ran:

```python
class StaticFreq:                       # v3
    def __init__(self, slots, freq=None, **kw):
        self.set = {k for k, _ in freq.most_common(slots)}
    def touch(self, key, tick, ...):
        return key in self.set          # <-- True on the FIRST touch
```

`FreqLRU` (and `freq_lru_warmup`, whose counts came from step 0 and were
applied during that same step 0) had the same shape. Consequences:

1. **First-touch hits.** For every pinned record, the very first request —
   including during the prefill pass whose statistics defined the pin set —
   returned `True`. A cold cache cannot serve a record from DRAM before it
   has ever been read from NVMe. That is PREWARM residency, uncharged and
   unlabeled, inside a COLD-START comparison.
2. **Statistics leaked across causality.** The pin set used future
   information (whole-trace counts) while being reported as a realizable
   policy, and the warmup variant used step-0 counts against step-0 itself.
3. **The impossible result.** Under regime A (equal, empty initial state)
   Belady/MIN is an upper bound for every online policy. v3 reported
   static pin 91.4 % > Belady 53.6 % at 24 GiB. A realizable policy cannot
   beat MIN from the same initial state; therefore the comparison was
   invalid. (The mechanisms are genuinely different — frequency vs recency
   — but a pin from a cold start must still PAY for its first
   materializations; v3 skipped that payment.)

## 2. Magnitude of the accounting error (measured, corrected sim)

Accounting closes exactly. `freq_lru_warmup` @16 GiB:

| | hits | misses | hit rate |
|---|---|---|---|
| v3 (buggy) | 3,159 | 1,940 | 61.95 % |
| v4 (corrected) | 2,517 | 2,582 | 49.36 % |
| **delta** | **+642** | **−642** | −12.59 pp |

The delta 642 is **exactly** `int(1285 slots × 0.5)` = the pin size: every
pinned record had been credited on its first touch. The v3 `static_freq`
family carries the same error (its pinned first touches were all phantom):
at 24 GiB, static pin's true cold-start value is bounded by Belady
(53.6 %) — v3's 91.4 % was 642+ phantom hits plus reject-admission
dynamics, not a realizable cold-start outcome.

## 3. The corrected regime contracts (v4)

Every row in `results/sim_rows_v4.csv` carries `regime`,
`initial_records`, `initial_state_gib`, `repopulation_charged`:

- **A — COLD.** Empty cache. Every first physical materialization charged.
  Policies: lru, engine_priority_lru, arc, lfu, freq_x_recency, layer_lru,
  cost_aware, and `belady_cold` = **strict upper bound for A**.
- **B — WITHIN-REQUEST WARMUP.** Starts cold; counts only from
  already-executed forwards; the policy change happens at a step boundary
  *after* the evidence exists; every pinned record not resident at the
  boundary is **charged a repopulation miss**; pins are removed from the
  dynamic LRU pool (no phantom hits) and the dynamic cap shrinks to
  `slots − |pin|`. Variants: boundary after step 0 (prefill counts) and
  after step 3 (first 4 forwards — the first point with a real frequency
  signal, since within a single pass every count is 1 on this architecture:
  each layer is visited once per step).
- **C — PREWARMED (explicitly labeled).** Initial residency exists,
  `initial_state_gib` is stated, and its cost is never mixed into A/B
  comparisons. Bound: **Belady/MIN given the same initial state**.
  Two contract-matched bounds are reported:
  - `belady_same_state_prewarm`: initial resident = top-`slots`
    frequency set (matches `static_freq_prewarm`/`lru_prewarm_topN`).
  - `belady_pin_same_state(pf)`: initial resident = top-`pf·slots`
    set, protected, plus MIN dynamics on the remainder (matches
    `freq_lru_prewarm(pf)` pointwise).
  Additionally, `static_freq_prewarm` is labeled **not
  fetch-on-fault** (reject-admission): its comparison target for the
  same prewarm is `lru_prewarm_topN` (admission variant), which it beats
  at every budget (60.5 vs 43.4 % @8 GiB … 91.4 vs 88.9 % @24 GiB).

**Cross-regime comparisons are prohibited** unless the prewarm is
explicitly labeled and its cost accounted (e.g., "pin policies beat cold
Belady" is a statement about regime C vs regime A and is now only made
with the prewarm named, sized in GiB, and its build cost discussed).

## 4. Corrected headline results (sealed trace, 5,099 requests)

Regime A (cold) — Belady 53.64 % at every budget; nothing beats it:

| policy | 8 | 12 | 16 | 20 | 24 | 32 GiB |
|---|---|---|---|---|---|---|
| lru | 40.32 | 46.60 | 50.83 | 52.32 | 53.09 | 53.64 |
| freq_lru_warmup (B) | 30.93 | 40.91 | 49.36 | 52.07 | 53.07 | 53.64 |
| freq_lru_warmup_s4 (B) | 36.79 | 46.50 | 50.44 | 52.48 | 53.42 | 53.64 |
| belady_cold (bound) | 53.64 | 53.64 | 53.64 | 53.64 | 53.64 | 53.64 |

Regime C (prewarmed, labeled) — pin value is real *as a prewarm contract*:

| policy | 8 | 12 | 16 | 24 | 32 GiB |
|---|---|---|---|---|---|
| static_freq_prewarm | 60.48 | 72.52 | 78.84 | 91.43 | 100 |
| freq_lru_prewarm pf0.75 | 56.17 | 66.13 | 72.52 | 81.98 | 91.43 |
| belady_same_state_prewarm | 64.64 | 72.52 | 78.84 | 91.43 | 100 |

*(Figure corrected 2026-09-10: prior row used never-used-distance init;
exact contract init per `research/phase2-regime-c` @ 643d3559.)*

Note the corrected regime-C ordering: with the *same* top-slots prewarm,
MIN's dynamic management ties or beats the static pin from 8 through
24 GiB (91.43 vs 91.43 % at 24 GiB — an exact tie; strictly better at
8 GiB, 64.64 vs 60.48 %). v3's "pins beat MIN" claim came from
comparing v3's phantom static pin against MIN; with equal initial
states MIN is never beaten.

Regime B interpretation (the honest online statement): on THIS sealed
16-token window, a warmup pin **cannot pay for itself** — the window is
too short and the per-pass frequency signal too thin (each layer visited
once per pass; counts 1 everywhere at step 0; cnt≥2 records after 4
forwards: only 389 of 1,465 distinct). Its measured cost at 16 GiB is
−1.5 pp vs plain LRU (49.36 vs 50.83 %); the pin machinery neither gains
nor loses much, and it carries repopulation exposure whenever the pin set
exceeds what the warmup pass itself left resident (charged misses — s0:
321 @8 GiB, 266 @12, 0 at ≥ 16 GiB; s4: 0 @8, 92 @12, 180 @16, 0 @≥ 20).

## 5. Why the sealed anchors were never in danger

The v3 numbers that reproduced sealed evidence (host LRU @682 slots,
VRAM engine-priority @281 slots) are **online recency policies** — no
future counts, no pin sets. They were and remain exact (±1..2 counts;
`results/validation_v4.json`). The bug lived only in the count-based pin
policies, i.e., in the *new policy proposals*, not in the validated
baseline. That is precisely why the impossible result could slip through:
the anchors all passed while the speculative rows were wrong.

## 6. Mechanical checks that now prevent recurrence

Built into the v4 simulator and its outputs:

1. Every CSV row carries `regime` + `initial_records` +
   `initial_state_gib` + `repopulation_charged`.
2. In-regime dominance check (rerun in CI-style each sim pass):
   - no regime-A or regime-B policy exceeds `belady_cold` at any budget;
   - `freq_lru_prewarm(pf)` never exceeds `belady_pin_same_state(pf)`;
   - `static_freq_prewarm` never loses to `lru_prewarm_topN`.
   All PASS on the sealed trace at 8..128 GiB (host) and 1..8 GiB (GPU).
3. Warmup-pin repopulation is charged and reported per budget; the pin
   set can never hit on a request whose record was not either resident
   pre-boundary (paid by the warmup pass itself) or repopulated
   (charged at the boundary).
4. The regime-A Belady row exists at every budget and is printed in the
   same table as every candidate — an impossibility is visible by
   inspection.

## 7. Residual caveats (honest limits)

- Single 16-token window, one prompt family ("Who is Alan Turing?").
  Frequency-shaped conclusions (regime C ceilings, coverage curve) are
  properties of this window; the regime-B "cannot pay for itself"
  conclusion is *strengthened* for longer windows (more decode steps give
  frequency time to matter) but the pin-vs-LRU gap direction at short
  windows is trace-specific.
- The engine-dedup stream is the correct unit for tier traffic (validated
  against both sealed counters), but per-`forward_step` intra-batch
  scheduling (e.g., layer-to-layer overlap) is outside a cache simulator;
  wall projections remain feed-bandwidth bounds, not schedules.
- Regime C prewarm build cost (one-time read of the prewarm set from
  NVMe, ~29.43 GiB worst case) is real work that regime C does not charge
  inside the measured window; it is a per-process (or per-cache-reset)
  amortized cost, and is labeled rather than simulated.
