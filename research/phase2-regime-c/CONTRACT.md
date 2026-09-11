# CONTRACT.md — the regime-C / prewarmed cross-request residency contract

**Status.** Decision memo, analysis only — no engine changes. Closes the open
caveat carried since the corrected working-set study
(`research/phase2-ws-policy @ 95dfe0d`, worktree
`C:/Users/carth/Downloads/dee-phase2-ws-policy`, read-only): "the
prewarmed-Belady oracle initialized untouched resident entries with an
artificial never-used distance; static-pinned/reject-admission caching vs
ordinary fetch-on-fault MIN are different contracts that must not be presented
as identical bounds."

**Verdict up front.** Regime C is a *deployment contract*, not a cache policy:
it names the state in which a persistent host tier begins a measured window
with a chosen set of expert records already resident, paid for once outside
that window. The mechanism for it already exists in the integrated host tier
(`PolicyResident` partition + `policy_slots` + `residency()`), so the open
questions are contractual — which bound is legal against which policy — and
economic — when a live prewarm arm is worth a T4 cell. Both resolve cleanly:
the "never-used distance" issue was a draft-simulator artifact whose stale
figures persist only in three prose tables (the committed CSV is exact);
fetch-on-fault MIN and the static pin are different contracts that happen to
coincide on this window for a structural reason (§3); and the prewarm arm is
**deferred to Phase 5** (§4).

Evidence anchor for every number: sealed v50 route journal, sha256
`665aac3e8db570237c6dc6acaf08dc39f2af890e8a04e400ce7154f1a858dae1` (5,099
engine-dedup requests, 2,364 unique 12.75 MiB records, 16 forwards).
Re-derivation: `tools/phase2_regime_c_eval.py` (stdlib only) →
`results/regime_c_eval.json`; inputs `inputs/sim_rows_v4{,_derived}.csv` are
byte copies of the committed v4 sim output.

---

## 1. Regime labels (carried over unchanged)

| Regime | Initial state | Cost accounting | Bound |
|---|---|---|---|
| **A — cold** | empty cache | every first materialization charged in-window | `belady_cold` (strict, 53.64 %) |
| **B — causal warmup** | empty; statistics only from already-executed forwards | pin applied at a step boundary; non-resident pins charged repopulation misses | `belady_cold` |
| **C — prewarmed** | a named set is resident at t=0; `initial_records`/`initial_state_gib` stated on every row | prewarm read is a one-time cost amortized **outside** the window; never mixed into A/B comparisons | contract-matched MIN given the same initial state |

Regime C subdivides by *admission contract* (v4 sim `contract` column):

- **C1 — bypass / fixed set** (`static_freq_prewarm`): the resident set is
  pinned; requests to non-pinned records are never admitted to the cache.
  Optimum of this contract = top-N by frequency (exchange argument,
  brute-force-verified in `verify_contracts`).
- **C2 — fetch-on-fault** (`lru_prewarm_topN`, bound
  `belady_same_state_prewarm`): same initial state, but every miss is
  admitted and pays its compulsory fill; eviction is optimal (MIN) for the
  bound, LRU for the realizable control.
- **C3 — protected pin + dynamic remainder** (`freq_lru_prewarm(pf)`, bound
  `belady_pin_same_state(pf)`): `pf·slots` prewarmed and protected, the
  remaining slots run a fetch-on-fault policy over non-pins.

## 2. The contract in `HostExpertTier` terms (deliverable a)

The integrated host tier (`dee.cpp/include/dee/host_expert_tier.h`,
`dee.cpp/src/host_expert_tier.cpp`) already contains every mechanism regime C
needs. The mapping is exact:

| Contract element | Mechanism | Where |
|---|---|---|
| Two residency classes | `enum class HostResidency { PolicyResident, Dynamic }`; partition fixed at construction: slot `i < policy_slots` is `PolicyResident`, rest `Dynamic` | `host_expert_tier.h:66`; `host_expert_tier.cpp:181` |
| Pin-set membership | `HostPlacementPolicy::residency(key)` is consulted on every miss and selects the partition the fill lands in | `host_expert_tier.h:81`; `host_expert_tier.cpp:218` |
| Protection | `victim()` is only ever offered `Dynamic` + `Ready` + unleased slots — policy-resident slots are never eviction candidates | `host_expert_tier.cpp:223-226`; design doc "protected even when idle" |
| Withdrawal | explicit `evict(key)` only, and only when the slot is idle — works on either class | `host_expert_tier.cpp:273-283` |
| Pin-set size bound | `policy_slots`; the (N+1)-th distinct `PolicyResident` key gets `Capacity` — there is no intra-partition victim selection | `host_expert_tier.cpp:233-236` |
| Persistence across requests | `phase2_host_` is a `std::unique_ptr` member created once at engine init; `reset_runtime_cache()` clears the *device* arena only; `reset_external_profile()` explicitly does not reset host residency or counters | `engine.h:410`; `engine.cpp:3709`, `2158-2178`; `PHASE2_METRICS.md:59` |
| Prewarm | `HostExpertTier::acquire(record, store, codec)` is public: issue one acquire per pinned/top-N record before the measured window; each performs the normal `IdentityCodec`/`ExpertStoreColdAdapter` materialization into a policy slot | `host_expert_tier.h:165`; `expert_tiers.cpp:45-85` |
| Prewarm cost | one-time `N × 12.75 MiB` logical read through the same cold path; counted in `stats.SSD_bytes`/`fills` of the tier lifetime | §2.3 accounting note |
| Cross-request metric hygiene | host counters have no independent reset — exclude prewarm by `TierMetrics` snapshot-delta (post-prewarm snapshot minus end-of-window snapshot), not by reset | `PHASE2_METRICS.md:58-62` |

So **"static pin" in the simulator = `PolicyResident` partition +
`policy_slots` + a `residency()` that returns `PolicyResident` exactly for the
chosen set**, and **"prewarmed cross-request residency" = that partition
populated once, then left resident for the process lifetime** — which is
precisely the dee-serve shape (a persistent tier across requests), not a new
mechanism.

### 2.1 Contract → concrete configuration

| Sim contract | `HostTierConfig` | Policy object | Prewarm driver | Engine-expressible today? |
|---|---|---|---|---|
| C2 `lru_prewarm_topN` | `policy_slots=0`, `dynamic_slots=N` | `PlainLruHostPlacementPolicy` (default) | N acquire()s before window | Mechanism yes; needs a prewarm entry point (none exists — `acquire` is public but `Engine` exposes no prewarm pass) |
| C3 `freq_lru_prewarm(pf)` | `policy_slots=⌊pf·N⌋`, `dynamic_slots=N−⌊pf·N⌋` | custom policy, `residency()` = PolicyResident on the pin set | acquire each pinned record | Yes, modulo the same driver; `engine.cpp:3703-3707` requires `host_policy` non-null when `policy_slots>0` (fail-closed otherwise) |
| C1 `static_freq_prewarm` | `policy_slots=N`, `dynamic_slots=0` | custom policy as above | same | **No** — see §2.2 |

### 2.2 The one contract the current engine cannot express: true C1

C1's "non-pins are never admitted" means the *cache* never stores them, but
the *expert still must execute* — a real C1 deployment needs a non-caching
bypass (a transient read buffer per request). The integrated path has none:
with `dynamic_slots=0`, a non-pinned key's `acquire()` returns `Capacity`
immediately; `DeviceExpertTier::stage` retries once after reclaiming DMA-held
leases (`expert_tiers.cpp:129-135`), then counts `device_failures` and returns
false; on the real-model path `stage_expert()==false` aborts the run
(`engine.cpp:299, 4121`). The mechanism fails closed — correct behavior, but
it means C1 is not a configuration, it is a design task (bypass path) if ever
wanted. The within-contracts approximation runnable today is C3 with
`dynamic_slots` small (a streaming remainder that admits faults but retains
~nothing), or C2 — and on this window both coincide with C1 above the tie
boundary anyway (§3.2).

Two further mechanism caveats for any future arm:

- `policy_slots` are allocated (and `cudaHostRegister`'d when `try_pin`) at
  construction — the pin cost is in `allocated_bytes`/`pinned_bytes` whether
  or not slots are filled. `HostTierConfig.try_pin=false` keeps them
  pageable.
- On a 2-GPU deployment there are **two** engine instances, hence two host
  tiers; a "24 GiB pooled prewarm" is split by layer partition (trace:
  cuda0 layers 0-21 ≈ 16.7 GiB, cuda1 layers 22-42 ≈ 12.2 GiB of the 29.43
  GiB working set). The sim's per-layer-budgeted row
  (`static_layer_freq_prewarm`) is the closest model of a per-engine pin —
  78.03/90.51 % at 16/24 GiB, within 0.8-0.9 pp of the global pin.

### 2.3 Prewarm cost, stated not hidden

One-time reads at the Phase-1 bank ceiling (0.29-0.37 GiB/s), charged
**outside** the measured window — this is the sense in which regime C is
"free" inside a window and why its rows can never sit next to cold-start
rows:

| Prewarm | Records | One-time read @0.29 / @0.37 GiB/s |
|---|---:|---|
| 8 GiB | 642 | 27.6 s / 21.6 s |
| 16 GiB | 1,285 | 55.2 s / 43.2 s |
| 24 GiB | 1,927 | 82.7 s / 64.8 s |
| 29.43 GiB (100 %) | 2,364 | 101.5 s / 79.5 s |

Per-process (or per-cache-reset), amortized across every subsequent request.
In dee-serve terms this is cache-build cost at process start, re-paid only
when the pin set is rebuilt.

## 3. Reconciliation of the two flagged sub-questions (deliverable b)

### 3.1 The "never-used distance" artifact — resolved, located, bounded

The committed `BeladyPrewarm` initializes each resident entry with its **true
first future use** (`fu.get(k, 10**12)`; `phase2_ws_policy_sim_v4.py:366`) —
exactly what MIN requires before its first eviction decision. The flagged
artifact was real but lives in the *prose tables*, not the committed data:
all three docs quote a `belady_same_state_prewarm` row generated by a draft
that initialized **every** resident entry to `10**12` ("never used") until
its first real access. Under that init the first fault evicts a resident by
rank order regardless of its real next use, systematically understating the
C2 bound.

`tools/phase2_regime_c_eval.py` reproduces both inits mechanically:

| Budget | C1 static pin | C2 MIN — exact init (committed CSV) | C2 MIN — never-used init (= doc figures) |
|---|---|---|---|
| 8 GiB | 60.48 % | **64.64 %** | 56.83 % |
| 12 GiB | 72.52 % | **72.52 %** | 59.11 % |
| 16 GiB | 78.84 % | **78.84 %** | 70.68 % |
| 20 GiB | 85.13 % | **85.13 %** | 83.02 % |
| 24 GiB | 91.43 % | **91.43 %** | 91.14 % |
| ≥29.43 GiB | 100 % | 100 % | 100 % |

The never-used column reproduces the doc figures to <0.05 pp — including the
specifically quoted "91.14 vs 91.43 at 24 GiB" (`PHASE2_BYTE_FLOOR.md` §3,
`PHASE2_POLICY_MATRIX.md` §1.2, `PHASE2_WORKING_SET.md` §5,
`CAUSALITY_AND_INITIAL_STATE_AUDIT.md` §4). The committed
`results/sim_rows_v4.csv` instead matches the exact init at every budget.
**Resolution: the bound in the committed data is correct; the doc numbers are
stale.** Corrected reading: C2-MIN ≥ C1-pin at *every* budget on this window
(strictly better at 8 GiB, exact tie at 12-24 GiB) — the artifact had
inverted the comparison below saturation. The docs' qualitative conclusion
("the prewarm *size* is the lever, not static-vs-dynamic") survives and is
strengthened; only the figures were wrong. Nothing about regime A/B, host
LRU, or the VRAM finding is touched.

### 3.2 Why MIN "ties" the pin at equal prewarm — a structural coincidence, not a contract identity

On this window, **935 of 2,364 records ever repeat** (freq ≥ 2); the other
1,429 are single-use. Two consequences:

- Once capacity ≥ 935 records (≈ 11.64 GiB), any top-N set contains *every*
  repeating record. Then under **both** contracts all remaining misses are
  compulsory first touches of single-use records — the unbeatable floor.
  Hence the exact ties at 12/16/20/24 GiB (identical miss counts 1401/1079/
  758/437, not rounded). The tie is a property of the workload's tail, not
  evidence that the contracts coincide in general.
- Below that boundary the frozen pin cannot hold all repeaters: at 8 GiB
  (642 < 935), 293 repeat-requests land on non-pinned repeaters and miss
  under C1. C2-MIN does strictly better (1,803 vs 2,015 misses) because it
  may evict pins whose remaining uses are all far away and admit
  soon-repeating non-pins — i.e., the prewarmed set is a *starting state*,
  not a protected set, under C2.

The contractual difference that must be preserved in every writeup:

- **C1 (bypass)**: misses = requests outside the pin set — non-pin *repeats*
  also miss. Optimal set = top-N frequency; needs no oracle dynamics because
  it never admits.
- **C2 (fetch-on-fault)**: misses = compulsory fills of non-resident records
  + capacity misses; MIN is optimal *within the admit-all obligation*. The
  obligation is a real liability — admitting a never-repeating fault can
  evict a record that would have hit — yet on this window MIN's freedom to
  drop late-use pins outweighs it.

`C1_misses − C2_misses = (non-pin repeats) − (capacity misses)`: the sign is
workload-dependent. "MIN ties or beats the static pin at equal prewarm" is a
measured property of this sealed window — reportable, but **never to be
stated as the two contracts sharing a bound**. The correct sentence for
future docs is the corrected one: *at equal prewarm size and state, the
fetch-on-fault MIN bound and the static-pin optimum coincide once capacity
covers the repeating set (~11.6 GiB here); below that, contract choice
matters; neither bounds the other.*

## 4. Decision memo (deliverable c): prewarm arm — DEFER to Phase 5

**Question.** Is a prewarm arm worth a T4 cell now (Phase-2 closure), or
deferred to Phase 5 (serving/concurrency)?

**Decision: DEFER.** A prewarm arm is a dee-serve contract demonstration,
not a Phase-2 policy question, and a T4 cell for it buys ~no decision value
today. Reasons, in order of strength:

1. **The standing rules make it non-comparable by construction.** Regime-C
   rows are never scored against cold-start rows, and the Phase-2 A/B
   runbook already rules it out of eligibility ("no … regime-C prewarming …
   is eligible", `PHASE2_AB_RUNBOOK.md`). A prewarm arm cannot join the
   matched comparison the GPU batch is built around; it would be a separate
   one-arm demonstration inside a budget-capped batch.
2. **Its measurable outcome is already bounded.** A within-envelope arm
   (≤17 GiB pooled, the current RSS envelope) lands at the simulated
   ~78.8 % coverage / 859.8 MiB-per-token point; the run would confirm a
   feed-bound number, not discover one. The cells that would add
   information — ≥24 GiB — are memory-gated out (below).
3. **The memory gate caps exactly the interesting region.** v60 envelope:
   17 GiB host packs → 22.48-22.9 GiB peak RSS on the 31.35 GiB host. A
   24 GiB pooled prewarm projects ≈ 29-30 GiB peak RSS — inside the limit
   on paper but ~1.3-2.3 GiB of margin, outside the campaign's safety
   margin, and worse in practice because policy slots are
   `cudaHostRegister`'d (non-pageable) under `try_pin`. 100 % coverage
   (29.43 GiB) does not fit at all. So the arm is confined to the
   already-simulated region by hardware, not by choice.
4. **The open question it can't answer is the one that matters.** Regime C's
   real risk is cross-request *rank stability* (top-1 % covers 6.8 % of
   activations; the tail is wide). A T4 cell on the same sealed window is
   rank-stable by construction — it cannot measure the property that
   decides whether prewarming pays in deployment. That requires a
   multi-request/multi-prompt workload — a Phase-5 artifact.
5. **Mechanism assurance does not need a GPU.** Partition protection,
   persistence, and the prewarm-via-`acquire` path are host-side and
   unit-testable on the CPU build (the tier tests already exercise a
   `PolicyResident` scenario via a fixture policy). What is genuinely
   missing — an Engine-level prewarm entry point and, for true C1, a
   non-caching bypass — is small design work for whenever the arm is
   scheduled, not a reason to spend a T4 run.

**Honest counterweight.** If the Phase-2 closure reprofile still shows
feed-side dominance *and* a cheap demonstration is wanted inside the same
GPU batch, the least-bad cell is: same commit and workload; `phase2` host
arm on; pooled host budget **unchanged** at ≤17 GiB (no new RSS risk);
`policy_slots` + a pinned-top-N custom policy (C3 shape, `dynamic_slots ≥ 1`
so non-pins never hit the fail-closed `Capacity` path); prewarm driven
through `acquire()` after init, before the timed window; host counters
separated by snapshot-delta since they cannot be reset; the row labeled
regime C with `initial_state_gib` stated. Even then, treat it as a
mechanism demo, not evidence for/against a policy.

**Revisit triggers (any one suffices):** a multi-prompt or multi-request
route journal exists to prewarm from; a dee-serve prototype owns a
persistent tier and needs the economics; the host side gains ≥24 GiB of
headroom (bigger host or smaller dense footprint); or the closure reprofile
shows residency, not concurrency, is the binding lever *and* a decision
depends on the live number.

## 5. Standing rules for regime-C claims (binding on future artifacts)

1. Every regime-C row carries `regime`, `initial_records`,
   `initial_state_gib`, `contract` (C1/C2/C3), and the prewarm cost is
   stated alongside, never inside, the window.
2. No regime-C figure is scored against regime-A/B figures in the same
   table without both labels and an explicit cross-regime caveat.
3. Bounds are contract-matched: C1 is its own offline optimum; C2 is
   bounded by MIN *given the same initial state*; C3 by
   `belady_pin_same_state(pf)`. Cross-contract comparisons are empirical
   observations about this window — never "the" bound.
4. Quote `sim_rows_v4.csv` (or a rerun of the v4 sim / the eval tool), not
   the stale prose figures listed in §3.1.
5. Rank stability across requests is an assumption about a workload class,
   not a measured property of this window.

## 6. Artifacts

- `tools/phase2_regime_c_eval.py` — stdlib-only re-derivation; verifies the
  journal hash, re-computes C1/C2(+never-used-init)/C2-LRU bounds at all
  budgets, cross-checks the committed CSV, and emits
  `results/regime_c_eval.json`. Exit 0 = all checks pass.
- `inputs/sim_rows_v4.csv`, `inputs/sim_rows_v4_derived.csv` — byte copies
  of `research/phase2-ws-policy @ 95dfe0d` committed outputs.
- `results/regime_c_eval.json` — this branch's verified contract table.
- Mechanism citations: `dee.cpp/include/dee/host_expert_tier.h`,
  `dee.cpp/src/host_expert_tier.cpp`, `dee.cpp/src/expert_tiers.cpp`,
  `dee.cpp/src/engine.cpp`, `dee.cpp/include/dee/engine.h` at this branch's
  head; ws-policy docs at `95dfe0d` (`PHASE2_BYTE_FLOOR.md` §3,
  `CAUSALITY_AND_INITIAL_STATE_AUDIT.md`, `PHASE2_POLICY_MATRIX.md`,
  `PHASE2_WORKING_SET.md`, `PHASE2_RECOMMENDATION.md`).
