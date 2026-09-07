# ORDER_EFFECT_FORENSICS.md — why B swings 4.4 s by position while A is stable

Branch: `experiment/pack-cap-ab` · Date: 2026-09-07
Machine-readable: `results/order-effect.json`

## 1. The measured facts

| Arm | Position | Cap | decode wall | min MemAvailable | p50 read (cuda0) |
|---|---|---|---|---|---|
| s1A | 1st | 17 GiB | 70.646 s | 7.87 GiB | 107.8 ms |
| s1B | 2nd | 20 GiB | 68.483 s | 4.83 GiB | 102.5 ms |
| s2B | 1st | 20 GiB | 72.918 s | 4.94 GiB | 108.5 ms |
| s2A | 2nd | 17 GiB | 70.692 s | 7.73 GiB | 107.3 ms |

Position sensitivity: **A +0.046 s, B −4.435 s**. Mean B−A = +0.0315 s.

## 2. Decomposition by token region (the key finding)

Summing per-token walls (all from `per_token_accounting`):

| Region | s1A | s1B | s2B | s2A |
|---|---|---|---|---|
| tokens 1–11 | 57.75 | 58.32 | 60.14 | 60.87 |
| tokens 12–14 (+15) | 12.90 / (3.67 t15) | 10.16 / (3.06) | 12.78 / (3.70) | 9.82 / (2.27) |

- **Tokens 12–15: a position effect, arm-agnostic.** The second-position arm
  is faster in BOTH sessions (B: −2.08 s over 12–14; A: −1.56 s over 12–14,
  plus token 15: −2.41 s). Mechanistically visible: measured read-batch wall
  at tokens 13–15 roughly **halves for the second-position arm**
  (mean second−first: tok13 −1.10 s, tok14 −3.59 s, tok15 −2.41 s).
- **Tokens 1–11: a session effect, arm-agnostic.** Both session-2 arms are
  ~+2.3–2.5 s slower than both session-1 arms (57.75/58.32 vs 60.14/60.87).
  Coinciding measured difference: cuda1 read bandwidth drops
  123.8/124.9 → 109.1/107.2 MiB/s in session 2.

The apparent "B is order-sensitive, A is stable" is the **sum of two
arm-agnostic effects landing differently on the four (arm, position) cells**:

    A-first  = session-fast mid + position-cold tail = 70.65
    B-second = session-fast mid + position-warm tail = 68.48
    B-first  = session-slow mid + position-cold tail = 72.92
    A-second = session-slow mid + position-warm tail = 70.69

## 3. Mechanisms, with evidence grades

- **PROVEN (measured):** the tail-token read-batch-wall reduction for
  second-position arms; the session-2 mid-run slowdown and its cuda1
  bandwidth drop; B's memory footprint (VmHWM 25.54–25.64 vs 22.62–22.80
  GiB; MemAvailable 4.83–4.94 vs 7.73–7.88 GiB); B's larger
  fill-reservation wall (+1.2–1.6 s summed across GPUs: 14.8 vs 13.4 s for
  A... specifically 7.66+7.70 vs 6.33+6.41 s).
- **SUPPORTED (consistent, not directly counted):** page-cache warmth as the
  tail mechanism — arms run the identical journal, so the second arm's
  tokens-13–15 reads re-touch file regions the first arm read most recently
  (~2.4 GB), which plausibly survive the intervening clone/build/prefill;
  second-position p50 read latency improves for the candidate (102.5 vs
  108.5 ms) consistent with warm file pages.
- **PLAUSIBLE (unmeasured):** disk/SLC-cache state differences between
  sessions explaining the session effect; readahead behavior differences
  under differing MemAvailable.
- **UNKNOWN:** page-cache hit counts, major/minor faults, per-request
  timestamps — none recorded. The exact reason the candidate's *mid-run*
  position swing (+2.4 s first / −2 s second) exceeds A's (+0.0 s) cannot
  be isolated from these artifacts.

## 4. The goal's stated hypothesis, tested

> "larger pack B has a larger cold-start/page-fault/reclaim cost, but
> benefits more from a warmed backing store when run second."

- Larger-footprint cost: **SUPPORTED but small** — B pays ~+1.3 s of extra
  fill-reservation wall and runs with 3 GiB less MemAvailable; however the
  dominant B-first penalty (~+2.3 s mid-run vs A-first) is shared by
  A-second, so it is **session**, not candidate-specific.
- Warmed-store benefit when second: **SUPPORTED but NOT candidate-specific**
  — the tail read-batch-wall halving appears for the second arm whichever
  arm it is (A-second's tail is as warm as B-second's).
- Net: the hypothesis is **partially supported and incomplete**. The
  B-specific asymmetry arises because the session effect (±2.4 s on mid
  tokens) and position effect (−2.1…−3.0 s on tail tokens) compose
  differently per cell; it is not a property of the cap itself.

## 5. Rejection of the generic story

"This is a generic second-run-faster effect" is **rejected**: A's position
sensitivity is +0.046 s. The second-position advantage is real but
token-localized (tail) and is canceled for A by the session-2 mid-run
slowdown.

## 6. What would resolve the remainder

- Per-arm page-cache/fault telemetry (requires engine/telemetry change —
  out of scope here) or a profiler-enabled repeat;
- A same-session A/B/A triple with randomized middle arm would separate
  session from position effects (see FINAL_SUMMARY.md recommendation).
