# PREDICTIONS_PHASE6.md — pre-registered Phase-6 prediction matrix (FROZEN)

**This file is the scoreboard the Phase-6 Modal hardware campaign will be graded
against. Its git commit timestamp is the pre-registration of record.**

---

## 0. Header block

| field | value |
|---|---|
| registration timestamp | **2026-09-30 (UTC)** — registration instant `2026-09-30T03:45:16Z`, Phase-7 campaign clock |
| frozen at commit | the commit introducing this file on `research/phase7-theory` — registration of record = that commit's timestamp (`git log --follow -- dee.cpp/theory/PREDICTIONS_PHASE6.md`) |
| campaign | Phase-7 component D — pre-registered Phase-6 prediction matrix |
| hardware runs existing at freeze | **0** (no Phase-6 hardware run existed when these ranges were written) |
| freeze rule | Sections 0–5 and the four matrix tables are **FROZEN** at the freeze commit. Ranges and kill criteria must NOT be edited later. Later information may ONLY be appended as clearly-marked, timestamped **ADDENDA** (section 6+). |
| machine-readable twin | `theory/predecl.py` stage `predecl` → [`data/predecl_matrix.csv`](data/predecl_matrix.csv) (336 rows = 84 config cells × 4 metrics) + [`data/predecl_meta.json`](data/predecl_meta.json). `python -m theory.run_all predecl` regenerates both deterministically (no RNG). |
| source tags at freeze | **every row is `CLOSED-FORM-ONLY (sim pending at freeze)`** — Component A's simulator outputs (`data/sim_*.json`) and Component C's solver outputs (`data/solve_*.csv`) did not exist at freeze (checked 2026-09-30). SIM-PREDICTED tagging and the sim-vs-closed-form disagreement table go in the ADDENDUM only. |

### Scope

- **60 scored Phase-6 cells** = 5 Modal cells (**1xL4, 2xL4, 1xA10, 1xL40S, RTX PRO 6000**) × 2 models (**DSv4-Flash** `dsv4_flash`, **MiMo-Flash** `mimo_v2_flash`) × batch **b ∈ {1, 8}** × host budget **{16, 64, 256} GiB** — each predicting 4 metrics (decode tok/s, host-tier hit %, cold records/tok, $/1k tok), each with nominal + working range + kill criterion.
- **24 reference rows** (2xT4 anchor, CPU-only) marked **non-Phase-6** — correctness/calibration instruments and a CPU floor, not Modal campaign targets (section 2).
- Host budgets are **TOTAL GiB, split evenly over GPUs** (the engine host tier is per-GPU); the LRU-knee evidence is on a **pooled** basis (assumption A13). A run that provisions per-GPU instead must say so or the row is UNTESTABLE.

### Storage provisioning per cell (stated honestly)

- **DSv4-Flash** full packed store = **146.625 GiB** (11,776 records × 13,369,344 B = 157,437,394,944 B). **MiMo-Flash** full store = **153.0 GiB** (12,288 records × 13,369,344 B; AGENT_BRIEF.md claims 12,032 records — inconsistent, see §3b; provision 153 GiB to cover both).
- Every scored cell needs: the full packed store on **local NVMe** (≥ 153 GiB scratch free), the named host budget in RAM, and the per-GPU VRAM expert cache (L4/A10 **22 GiB = 1,766 records/GPU**; L40S **48 GiB = 3,855**; RTX PRO 6000 **96 GiB = 7,710**; 2xT4 3.5 GiB = 281/GPU).
- At host 256 GiB the DSv4 store (146.6 GiB) fits whole in host RAM and the MiMo store (153.0 GiB) fits **tightly** (98% of budget — provisioning note: no headroom for a second model copy). The first fill is still charged to the measurement window; steady-state rows assume the fill has been amortized and finite-window rows (b=1) assume a **cold-ish** start.
- Volume cost of the store: $0.09/GiB/mo above the 1 TiB free tier ≈ **$3.8/mo** for 146.6 GiB — negligible vs compute (THEORY.md 6).

### Scoring rules (how a cell is graded)

- **PASS** — every measured metric lands inside its registered `[lo, hi]` range.
- **KILLED** — any measured metric lands outside its kill band `<kill_lo | >kill_hi`, **or** outside the registered kill band of a FALSIFICATION.md id the row references (the effective kill band is the **intersection**; the registered id governs where they conflict). A killed row is marked `KILLED (date, evidence path)` here in an addendum and the theory is corrected — not defended.
- **WEAKENED** — a metric lands in the deliberate gap between a range edge and its kill edge (neither confirmed nor killed); logged in the addendum with magnitude.
- **UNTESTABLE** — missing prerequisite/artifact (see below) or a metric's measurement definition ambiguous (e.g. host provisioning basis unstated). Unscored; counts as no evidence.

### Hardware-run prerequisites per cell (a run missing any → UNTESTABLE)

1. billed Modal run on the **named config** at the **named host budget** (billing receipt captured);
2. full packed store on local NVMe as sized above;
3. **≥ 16 decode tokens** generated (b=1 rows are scored on the finite 16-token window, b=8 rows on the steady-state regime; a 4-token run may only be used for the host-hit and cold brackets, never for decode TPS);
4. **stage profile captured** (fills/compute/H2D/dense buckets — needed to adjudicate limiter claims and P5/P8b);
5. **measured B_SSD reported** (rows are registered against Cell spec × 0.9–1.1; a run outside that band is UNTESTABLE for decode TPS and $/1k tok);
6. host provisioning **basis stated** (total vs per-GPU).

---

## 1. The 60-cell matrix (4 tables: one per model × b)

Cell format: `nominal [lo-hi] kill <kill_lo\|>kill_hi`. `CF` = CLOSED-FORM-ONLY (sim pending at freeze). Decode tok/s horizon: **finite 16-token at b=1** (P1 convention), **steady-state at b=8** (P1b convention). $/1k tok is on the **steady-state TPS basis at both b** (THEORY.md 6 economics convention — the b=1 rows of `serve_cost_frontier.csv` embed steady TPS; this is the P15c identity's basis). Host hit % is a **FINITE/STEADY band**: nominal = finite-window LRU prior (short 4–20 token runs should land near it), upper edge = steady-Che (valid for ≥512-token serving windows; known **+25pp** over-prediction on short windows, FALSIFICATION known-failure 1). Host hit % is the host-tier hit share `hits/(hits+misses)`, pooled_dedup definition. Cold-rec lower edge `0` is the steady-Che floor (the model legitimately reads ~0 cold once host+VRAM cover the working set; a cold start's compulsory reads live in the nominal/horizon terms) — the falsifiable content of that metric is the **upper** edge and the P4 bracket.

#### DSv4-Flash, b=1 (primary horizon: finite 16-token)

| cell | host GiB | decode tok/s | host hit % | cold rec/tok | $/1k tok | src | IDs |
|---|---|---|---|---|---|---|---|
| 1xL4 | 16 | 1.3 [1-2.11] kill <0.702\|>3.02 | 51 [44-75.3] kill <37\|>80.3 | 156.1 [15.7-312.2] kill <11.8\|>390.3 | 0.0854 [0.032-0.0998] kill <0.0192\|>0.167 | CF | P2,P2b,P3,P4 |
| 1xL4 | 64 | 1.35 [1.05-2.22] kill <0.733\|>3.17 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0766 [0.0185-0.0844] kill <0.0111\|>0.141 | CF | P1,P2,P3,P4 |
| 1xL4 | 256 | 1.35 [1.05-2.22] kill <0.733\|>3.17 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.14 [0.0338-0.154] kill <0.0203\|>0.257 | CF | P15c,P2,P3,P4 |
| 2xL4 | 16 | 2.33 [1.85-4.31] kill <1.29\|>6.17 | 51 [44-75.3] kill <37\|>80.3 | 147.8 [0-295.5] kill <0\|>369.4 | 0.102 [0.0177-0.111] kill <0.0106\|>0.185 | CF | P2,P2b,P3,P4 |
| 2xL4 | 64 | 2.33 [1.85-4.31] kill <1.29\|>6.17 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.116 [0.0201-0.125] kill <0.012\|>0.21 | CF | P2,P3,P4 |
| 2xL4 | 256 | 2.33 [1.85-4.31] kill <1.29\|>6.17 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.172 [0.0296-0.185] kill <0.0178\|>0.31 | CF | P15c,P2,P3,P4 |
| 1xA10 | 16 | 1.49 [1.16-2.48] kill <0.809\|>3.55 | 51 [44-75.3] kill <37\|>80.3 | 156.1 [15.7-312.2] kill <11.8\|>390.3 | 0.0977 [0.0349-0.113] kill <0.0209\|>0.189 | CF | P2,P2b,P3,P4 |
| 1xA10 | 64 | 1.55 [1.2-2.6] kill <0.843\|>3.72 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0888 [0.0213-0.0979] kill <0.0128\|>0.163 | CF | P2,P3,P4 |
| 1xA10 | 256 | 1.55 [1.2-2.6] kill <0.843\|>3.72 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.152 [0.0364-0.167] kill <0.0218\|>0.279 | CF | P15c,P2,P3,P4 |
| 1xL40S | 16 | 2.63 [2.1-5.08] kill <1.47\|>7.27 | 51 [44-75.3] kill <37\|>80.3 | 147.8 [0-295.5] kill <0\|>369.4 | 0.114 [0.0181-0.124] kill <0.0109\|>0.206 | CF | P2,P2b,P3,P4 |
| 1xL40S | 64 | 2.63 [2.1-5.08] kill <1.47\|>7.27 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.128 [0.0203-0.138] kill <0.0122\|>0.231 | CF | P2,P3,P4 |
| 1xL40S | 256 | 2.63 [2.1-5.08] kill <1.47\|>7.27 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.182 [0.0289-0.197] kill <0.0174\|>0.329 | CF | P15c,P2,P3,P4 |
| 1xRTX PRO 6000 | 16 | 3.19 [2.58-6.69] kill <1.8\|>9.56 | 51 [44-75.3] kill <37\|>80.3 | 147.8 [0-295.5] kill <0\|>369.4 | 0.163 [0.0236-0.177] kill <0.0142\|>0.296 | CF | P2,P2b,P3,P4 |
| 1xRTX PRO 6000 | 64 | 3.19 [2.58-6.69] kill <1.8\|>9.56 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.177 [0.0256-0.192] kill <0.0154\|>0.32 | CF | P2,P3,P4 |
| 1xRTX PRO 6000 | 256 | 3.19 [2.58-6.69] kill <1.8\|>9.56 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.23 [0.0333-0.25] kill <0.02\|>0.417 | CF | P15c,P2,P3,P4 |
| 2xT4 *(ref)* | 16 | 0.178 [0.135-0.254] kill <0.0944\|>0.363 | 51 [44-75.3] kill <37\|>80.3 | 179.4 [74.1-358.8] kill <55.6\|>448.5 | 0.398 [0.277-0.518] kill <0.166\|>0.865 | CF | P3c,P5,P2,P2b,P3,P4,P3b |
| 2xT4 *(ref)* | 64 | 0.212 [0.161-0.303] kill <0.113\|>0.434 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.112 [0.0724-0.127] kill <0.0435\|>0.212 | CF | P3c,P5,P2,P3,P4,P3b |
| 2xT4 *(ref)* | 256 | 0.212 [0.161-0.303] kill <0.113\|>0.434 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.295 [0.19-0.334] kill <0.114\|>0.558 | CF | P3c,P5,P2,P3,P4,P3b |
| CPU-only *(ref)* | 16 | 0 [0-0.118] kill <0\|>0.263 | 51 [44-75.3] kill <37\|>80.3 | 190.3 [86.9-380.6] kill <65.1\|>475.8 | 345.3 [3.84-690.7] kill <2.3\|>1153 | CF | P2,P2b,P3,P4 |
| CPU-only *(ref)* | 64 | 0 [0-0.118] kill <0\|>0.263 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 426.1 [4.73-852.2] kill <2.84\|>1423 | CF | P2,P3,P4 |
| CPU-only *(ref)* | 256 | 0 [0-0.118] kill <0\|>0.263 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 749.7 [8.33-1499] kill <5\|>2504 | CF | P2,P3,P4 |

*(ref) = reference row, non-Phase-6.*

#### DSv4-Flash, b=8 (primary horizon: steady-state)

| cell | host GiB | decode tok/s | host hit % | cold rec/tok | $/1k tok | src | IDs |
|---|---|---|---|---|---|---|---|
| 1xL4 | 16 | 1.62 [1.23-2.54] kill <0.864\|>3.63 | 51 [44-75.3] kill <37\|>80.3 | 156.1 [124.0-312.2] kill <93\|>390.3 | 0.254 [0.162-0.333] kill <0.0972\|>0.556 | CF | P2,P2b,P3,P4 |
| 1xL4 | 64 | 6.75 [5.44-16] kill <3.81\|>22.9 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0767 [0.0323-0.0951] kill <0.0194\|>0.159 | CF | P2,P3,P4 |
| 1xL4 | 256 | 6.75 [5.44-16] kill <3.81\|>22.9 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.14 [0.0589-0.173] kill <0.0354\|>0.29 | CF | P2,P3,P4,P15 |
| 2xL4 | 16 | 14.6 [13.5-78.3] kill <9.48\|>112.0 | 51 [44-75.3] kill <37\|>80.3 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0538 [0.01-0.0581] kill <0\|>0.097 | CF | P2,P2b,P3,P4 |
| 2xL4 | 64 | 14.6 [13.5-78.3] kill <9.48\|>112.0 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0611 [0.0114-0.0659] kill <0\|>0.11 | CF | P2,P3,P4 |
| 2xL4 | 256 | 14.6 [13.5-78.3] kill <9.48\|>112.0 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0902 [0.0169-0.0974] kill <0.0101\|>0.163 | CF | P1b,P15b,P2,P3,P4,P15 |
| 1xA10 | 16 | 1.85 [1.42-2.95] kill <0.992\|>4.22 | 51 [44-75.3] kill <37\|>80.3 | 156.1 [124.0-312.2] kill <93\|>390.3 | 0.267 [0.168-0.349] kill <0.101\|>0.583 | CF | P2,P2b,P3,P4 |
| 1xA10 | 64 | 6.76 [5.45-16.1] kill <3.82\|>23 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0889 [0.0374-0.11] kill <0.0224\|>0.184 | CF | P2,P3,P4 |
| 1xA10 | 256 | 6.76 [5.45-16.1] kill <3.82\|>23 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.152 [0.0639-0.189] kill <0.0383\|>0.315 | CF | P2,P3,P4,P15 |
| 1xL40S | 16 | 15 [13.8-88.8] kill <9.69\|>126.9 | 51 [44-75.3] kill <37\|>80.3 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0597 [0.0101-0.0646] kill <0\|>0.108 | CF | P2,P2b,P3,P4 |
| 1xL40S | 64 | 15 [13.8-88.8] kill <9.69\|>126.9 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0669 [0.0113-0.0723] kill <0\|>0.121 | CF | P2,P3,P4 |
| 1xL40S | 256 | 15 [13.8-88.8] kill <9.69\|>126.9 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.0954 [0.0161-0.103] kill <0\|>0.172 | CF | P15b,P2,P3,P4,P14,P15 |
| 1xRTX PRO 6000 | 16 | 15.3 [14.1-102.0] kill <9.89\|>145.9 | 51 [44-75.3] kill <37\|>80.3 | 147.8 [0-295.5] kill <0\|>369.4 | 0.085 [0.0127-0.092] kill <0\|>0.154 | CF | P2,P2b,P3,P4 |
| 1xRTX PRO 6000 | 64 | 15.3 [14.1-102.0] kill <9.89\|>145.9 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.092 [0.0138-0.0995] kill <0\|>0.166 | CF | P2,P3,P4 |
| 1xRTX PRO 6000 | 256 | 15.3 [14.1-102.0] kill <9.89\|>145.9 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.12 [0.018-0.13] kill <0.0108\|>0.217 | CF | P15b,P2,P3,P4,P15 |
| 2xT4 *(ref)* | 16 | 0.0779 [0.0585-0.111] kill <0.041\|>0.159 | 51 [44-75.3] kill <37\|>80.3 | 179.4 [179.4-840.8] kill <134.6\|>1051 | 2 [1.4-2.66] kill <0.842\|>4.45 | CF | P3c,P5,P2,P2b,P3,P4,P3b |
| 2xT4 *(ref)* | 64 | 1.31 [1-2.07] kill <0.7\|>2.96 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.201 [0.127-0.262] kill <0.076\|>0.438 | CF | P3c,P5,P2,P3,P4,P3b |
| 2xT4 *(ref)* | 256 | 1.31 [1-2.07] kill <0.7\|>2.96 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 0.527 [0.333-0.688] kill <0.2\|>1.15 | CF | P3c,P5,P2,P3,P4,P3b,P15 |
| CPU-only *(ref)* | 16 | 0.0103 [0-0.931] kill <0\|>2.07 | 51 [44-75.3] kill <37\|>80.3 | 190.3 [190.3-1083] kill <142.7\|>1354 | 44 [0.489-88] kill <0.293\|>146.9 | CF | P2,P2b,P3,P4 |
| CPU-only *(ref)* | 64 | 0.0105 [0-0.947] kill <0\|>2.11 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 53.3 [0.593-106.7] kill <0.356\|>178.1 | CF | P2,P3,P4 |
| CPU-only *(ref)* | 256 | 0.0105 [0-0.947] kill <0\|>2.11 | 54 [47-100.0] kill <40\|>100.0 | 147.8 [0-295.5] kill <0\|>369.4 | 93.8 [1.04-187.6] kill <0.625\|>313.4 | CF | P2,P3,P4,P15 |

*(ref) = reference row, non-Phase-6.*

#### MiMo-Flash, b=1 (primary horizon: finite 16-token)

| cell | host GiB | decode tok/s | host hit % | cold rec/tok | $/1k tok | src | IDs |
|---|---|---|---|---|---|---|---|
| 1xL4 | 16 | 1.16 [0.898-1.89] kill <0.629\|>2.71 | 51 [39-75.9] kill <27\|>83.9 | 174.3 [17.6-348.5] kill <13.2\|>435.6 | 0.0953 [0.0358-0.111] kill <0.0215\|>0.186 | CF | P2,P2b,P3,P4 |
| 1xL4 | 64 | 1.21 [0.938-1.99] kill <0.656\|>2.84 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.0855 [0.0207-0.0942] kill <0.0124\|>0.157 | CF | P1,P2,P3,P4 |
| 1xL4 | 256 | 1.21 [0.938-1.99] kill <0.656\|>2.84 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.156 [0.0377-0.172] kill <0.0226\|>0.287 | CF | P15c,P2,P3,P4 |
| 2xL4 | 16 | 2.09 [1.65-3.86] kill <1.16\|>5.52 | 51 [39-75.9] kill <27\|>83.9 | 164.9 [0-329.9] kill <0\|>412.3 | 0.114 [0.0197-0.123] kill <0.0118\|>0.206 | CF | P2,P2b,P3,P4 |
| 2xL4 | 64 | 2.09 [1.65-3.86] kill <1.16\|>5.52 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.13 [0.0224-0.14] kill <0.0134\|>0.234 | CF | P2,P3,P4 |
| 2xL4 | 256 | 2.09 [1.65-3.86] kill <1.16\|>5.52 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.192 [0.0331-0.207] kill <0.0198\|>0.346 | CF | P15c,P2,P3,P4 |
| 1xA10 | 16 | 1.33 [1.03-2.22] kill <0.724\|>3.18 | 51 [39-75.9] kill <27\|>83.9 | 174.3 [17.6-348.5] kill <13.2\|>435.6 | 0.109 [0.039-0.127] kill <0.0234\|>0.211 | CF | P2,P2b,P3,P4 |
| 1xA10 | 64 | 1.39 [1.08-2.33] kill <0.755\|>3.33 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.0991 [0.0238-0.109] kill <0.0143\|>0.182 | CF | P2,P3,P4 |
| 1xA10 | 256 | 1.39 [1.08-2.33] kill <0.755\|>3.33 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.169 [0.0406-0.187] kill <0.0244\|>0.312 | CF | P15c,P2,P3,P4 |
| 1xL40S | 16 | 2.35 [1.88-4.55] kill <1.31\|>6.51 | 51 [39-75.9] kill <27\|>83.9 | 164.9 [0-329.9] kill <0\|>412.3 | 0.128 [0.0202-0.138] kill <0.0121\|>0.23 | CF | P2,P2b,P3,P4 |
| 1xL40S | 64 | 2.35 [1.88-4.55] kill <1.31\|>6.51 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.143 [0.0227-0.154] kill <0.0136\|>0.258 | CF | P2,P3,P4 |
| 1xL40S | 256 | 2.35 [1.88-4.55] kill <1.31\|>6.51 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.204 [0.0323-0.22] kill <0.0194\|>0.368 | CF | P15c,P2,P3,P4 |
| 1xRTX PRO 6000 | 16 | 2.85 [2.31-5.99] kill <1.62\|>8.56 | 51 [39-75.9] kill <27\|>83.9 | 164.9 [0-329.9] kill <0\|>412.3 | 0.183 [0.0264-0.198] kill <0.0158\|>0.33 | CF | P2,P2b,P3,P4 |
| 1xRTX PRO 6000 | 64 | 2.85 [2.31-5.99] kill <1.62\|>8.56 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.197 [0.0286-0.214] kill <0.0171\|>0.357 | CF | P2,P3,P4 |
| 1xRTX PRO 6000 | 256 | 2.85 [2.31-5.99] kill <1.62\|>8.56 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.257 [0.0372-0.279] kill <0.0223\|>0.465 | CF | P15c,P2,P3,P4 |
| 2xT4 *(ref)* | 16 | 0.159 [0.121-0.227] kill <0.0846\|>0.325 | 51 [39-75.9] kill <27\|>83.9 | 200.3 [82.7-400.5] kill <62\|>500.7 | 0.444 [0.309-0.578] kill <0.186\|>0.965 | CF | P3c,P5,P2,P2b,P3,P4,P3b |
| 2xT4 *(ref)* | 64 | 0.19 [0.144-0.272] kill <0.101\|>0.389 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.125 [0.0809-0.142] kill <0.0485\|>0.237 | CF | P3c,P5,P2,P3,P4,P3b |
| 2xT4 *(ref)* | 256 | 0.19 [0.144-0.272] kill <0.101\|>0.389 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.329 [0.212-0.373] kill <0.127\|>0.622 | CF | P3c,P5,P2,P3,P4,P3b |
| CPU-only *(ref)* | 16 | 0 [0-0.106] kill <0\|>0.236 | 51 [39-75.9] kill <27\|>83.9 | 212.4 [97-424.9] kill <72.7\|>531.1 | 385.5 [4.28-771.0] kill <2.57\|>1288 | CF | P2,P2b,P3,P4 |
| CPU-only *(ref)* | 64 | 0 [0-0.106] kill <0\|>0.236 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 475.7 [5.29-951.3] kill <3.17\|>1589 | CF | P2,P3,P4 |
| CPU-only *(ref)* | 256 | 0 [0-0.106] kill <0\|>0.236 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 836.9 [9.3-1674] kill <5.58\|>2795 | CF | P2,P3,P4 |

*(ref) = reference row, non-Phase-6.*

#### MiMo-Flash, b=8 (primary horizon: steady-state)

| cell | host GiB | decode tok/s | host hit % | cold rec/tok | $/1k tok | src | IDs |
|---|---|---|---|---|---|---|---|
| 1xL4 | 16 | 1.45 [1.11-2.27] kill <0.774\|>3.25 | 51 [39-75.9] kill <27\|>83.9 | 174.3 [138.4-348.5] kill <103.8\|>435.6 | 0.284 [0.181-0.372] kill <0.109\|>0.621 | CF | P2,P2b,P3,P4 |
| 1xL4 | 64 | 6.05 [4.88-14.4] kill <3.41\|>20.5 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.0856 [0.0361-0.106] kill <0.0216\|>0.177 | CF | P2,P3,P4 |
| 1xL4 | 256 | 6.05 [4.88-14.4] kill <3.41\|>20.5 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.156 [0.0658-0.194] kill <0.0395\|>0.323 | CF | P2,P3,P4,P15 |
| 2xL4 | 16 | 13.1 [12.1-70.1] kill <8.5\|>100.3 | 51 [39-75.9] kill <27\|>83.9 | 164.9 [0-329.9] kill <0\|>412.3 | 0.06 [0.0112-0.0648] kill <0\|>0.108 | CF | P2,P2b,P3,P4 |
| 2xL4 | 64 | 13.1 [12.1-70.1] kill <8.5\|>100.3 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.0682 [0.0127-0.0736] kill <0\|>0.123 | CF | P2,P3,P4 |
| 2xL4 | 256 | 13.1 [12.1-70.1] kill <8.5\|>100.3 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.101 [0.0188-0.109] kill <0.0113\|>0.182 | CF | P1b,P15b,P2,P3,P4,P15 |
| 1xA10 | 16 | 1.66 [1.27-2.65] kill <0.889\|>3.78 | 51 [39-75.9] kill <27\|>83.9 | 174.3 [138.4-348.5] kill <103.8\|>435.6 | 0.298 [0.187-0.39] kill <0.112\|>0.651 | CF | P2,P2b,P3,P4 |
| 1xA10 | 64 | 6.06 [4.88-14.4] kill <3.42\|>20.6 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.0993 [0.0417-0.123] kill <0.025\|>0.206 | CF | P2,P3,P4 |
| 1xA10 | 256 | 6.06 [4.88-14.4] kill <3.42\|>20.6 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.17 [0.0713-0.21] kill <0.0428\|>0.352 | CF | P2,P3,P4,P15 |
| 1xL40S | 16 | 13.4 [12.4-79.5] kill <8.68\|>113.7 | 51 [39-75.9] kill <27\|>83.9 | 164.9 [0-329.9] kill <0\|>412.3 | 0.0667 [0.0112-0.0721] kill <0\|>0.12 | CF | P2,P2b,P3,P4 |
| 1xL40S | 64 | 13.4 [12.4-79.5] kill <8.68\|>113.7 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.0746 [0.0126-0.0807] kill <0\|>0.135 | CF | P2,P3,P4 |
| 1xL40S | 256 | 13.4 [12.4-79.5] kill <8.68\|>113.7 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.106 [0.0179-0.115] kill <0.0108\|>0.192 | CF | P15b,P2,P3,P4,P14,P15 |
| 1xRTX PRO 6000 | 16 | 13.7 [12.7-91.4] kill <8.86\|>130.7 | 51 [39-75.9] kill <27\|>83.9 | 164.9 [0-329.9] kill <0\|>412.3 | 0.0949 [0.0142-0.103] kill <0\|>0.172 | CF | P2,P2b,P3,P4 |
| 1xRTX PRO 6000 | 64 | 13.7 [12.7-91.4] kill <8.86\|>130.7 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.103 [0.0154-0.111] kill <0\|>0.186 | CF | P2,P3,P4 |
| 1xRTX PRO 6000 | 256 | 13.7 [12.7-91.4] kill <8.86\|>130.7 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.134 [0.0201-0.145] kill <0.012\|>0.242 | CF | P15b,P2,P3,P4,P15 |
| 2xT4 *(ref)* | 16 | 0.0698 [0.0524-0.0995] kill <0.0367\|>0.142 | 51 [39-75.9] kill <27\|>83.9 | 200.3 [200.3-938.6] kill <150.2\|>1173 | 2.23 [1.57-2.97] kill <0.94\|>4.97 | CF | P3c,P5,P2,P2b,P3,P4,P3b |
| 2xT4 *(ref)* | 64 | 1.17 [0.896-1.86] kill <0.627\|>2.65 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.224 [0.141-0.293] kill <0.0849\|>0.489 | CF | P3c,P5,P2,P3,P4,P3b |
| 2xT4 *(ref)* | 256 | 1.17 [0.896-1.86] kill <0.627\|>2.65 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 0.588 [0.371-0.768] kill <0.223\|>1.28 | CF | P3c,P5,P2,P3,P4,P3b,P15 |
| CPU-only *(ref)* | 16 | 0 [0-0.834] kill <0\|>1.85 | 51 [39-75.9] kill <27\|>83.9 | 212.4 [212.4-1209] kill <159.3\|>1511 | 49.1 [0.545-98.2] kill <0.327\|>164.0 | CF | P2,P2b,P3,P4 |
| CPU-only *(ref)* | 64 | 0 [0-0.849] kill <0\|>1.89 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 59.5 [0.661-119.1] kill <0.397\|>198.8 | CF | P2,P3,P4 |
| CPU-only *(ref)* | 256 | 0 [0-0.849] kill <0\|>1.89 | 54 [42-100.0] kill <30\|>100.0 | 164.9 [0-329.9] kill <0\|>412.3 | 104.7 [1.16-209.5] kill <0.698\|>349.8 | CF | P2,P3,P4,P15 |

*(ref) = reference row, non-Phase-6.*

### Range derivations (cited per metric — what each band is computed from)

- **decode tok/s** — nominal = the substrate's own `tps_pred` column of [`data/roofline_curves.csv`](data/roofline_curves.csv) (the THEORY.md 5.1 closed form `T_pred = t_dense + t_compute + (1-eta_s)t_storage + (1-eta_h)t_h2d` instantiated on the empirical DSv4 popularity model), scaled ×1.1163 (= MIMO_TOUCH_RATIO) for MiMo-Flash. This reproduces the registered centers exactly: P1 1.35, P1b 14.6. **Range = T_pred at B_SSD 0.9x-1.1x spec, COMPUTE_EFF 0.3-0.8, t_0 50-420 µs, eta_s 0.15-0.45, eta_h 0.3-0.7** (32-corner envelope of the same closed form). Kill = [0.70×lo, 1.43×hi], tightened to FALSIFICATION's own bands where an id applies (P1: 0.9-2.1; P1b: 8-28).
- **host-tier hit %** — the **Che finite/steady band**. Nominal = finite-window LRU prior (`predecl_finite_hit_prior`: 51% at 16 GiB pooled = P2's registered 0.48-0.54 midpoint; saturating at 0.536-0.54 = the sealed-window repeat ceiling (5,099−2,364)/5,099, cf. P2c 850-1000 of 2,364-2,500 records ever repeat). Upper edge = steady-Che (HOST_HIT_CURVE over [`data/cache_store_knees.csv`](data/cache_store_knees.csv) H50/H80/H95 + 1.0 once pooled slots ≥ universe) — the model that is known to **over-predict short windows by +25pp** and governs ≥512-token serving windows. MiMo bands widened (−12pp/+8pp; kill −24pp/+16pp) for `S_TRANSFER`. At host 16 GiB rows the P2 kill band 0.45-0.57 governs short-window measurements (registered-ID rule). Independent of b.
- **cold records/token** — the **FALSIFICATION P4 bracket construction**, order-clamped: `[min(steady-Che, finite-Che), 2 × max(steady-Che, finite-Che)]`, nominal = finite-Che (anchor check: 162 predicted vs 155.06 measured `storage_requests/token`, +4.5%). The anchor-calibrated bracket is the corrected P4 value **[67.84, 323.92]** (Phase-7 correction 2026-09-30; the old "[46, 324]" text was a mismatch). The steady-Che floor of `0` at big budgets is honest (the model reads ~0 cold once host+VRAM cover the working set); the falsifiable content is the upper edge and the bracket. Kill = [0.75×lo, 1.25×hi]; P3's own kill 70-260 governs the anchor-sized rows (2xT4 reference at 16 GiB).
- **$/1k tok** — `$/1k tok = 1000 × rate_s / TPS` on the **steady-horizon TPS at the row's own b** (THEORY.md 6 economics convention). Nominal reproduces [`data/serve_cost_frontier.csv`](data/serve_cost_frontier.csv) exactly and with it the registered identities: P14 0.095 (0.07-0.13), P15b ratios 0.70 (2xL4) / 0.74 (1xL40S) / 0.93 (1xRTX PRO 6000), P15c ratios 1.08 (1xL4) … 2.29 (2xT4) against the dense baseline **0.12892 $/1k tok (expensive price reading)**, and THEORY.md 6's table (0.527 at 2xT4 b=8/256, 93.8 at CPU-only). Range = relative T_pred envelope. Kill = [0.60×lo, 1.67×hi]; P14's kill $0.05-0.20 (plus its TPS cross-check 8-28 tok/s) and P15b/P15c ratio kills govern their rows (see §4).

### Example rows verbatim (machine-readable column form, = `data/predecl_matrix.csv` columns)

| cell | model | b | host | metric | nominal | lo | hi | kill_lo | kill_hi | source_tag | prediction_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1xL4 | dsv4_flash | 1 | 64 | decode_tps | 1.35 | 1.05 | 2.22 | 0.733 | 3.17 | CLOSED-FORM-ONLY (sim pending at freeze) | P1 |
| 1xL4 | dsv4_flash | 1 | 64 | host_hit_rate | 0.54 | 0.47 | 1 | 0.4 | 1 | CLOSED-FORM-ONLY (sim pending at freeze) | P2 |
| 1xL4 | dsv4_flash | 1 | 64 | cold_records_per_tok | 147.8 | 0 | 295.5 | 0 | 369.4 | CLOSED-FORM-ONLY (sim pending at freeze) | P3,P4 |
| 1xL4 | dsv4_flash | 1 | 64 | usd_per_1k_tok | 0.0766 | 0.0185 | 0.0844 | 0.0111 | 0.141 | CLOSED-FORM-ONLY (sim pending at freeze) | - |
| 2xL4 | dsv4_flash | 8 | 256 | decode_tps | 14.6 | 13.5 | 78.3 | 9.48 | 112.0 | CLOSED-FORM-ONLY (sim pending at freeze) | P1b,P15b |
| 2xL4 | dsv4_flash | 8 | 256 | host_hit_rate | 0.54 | 0.47 | 1 | 0.4 | 1 | CLOSED-FORM-ONLY (sim pending at freeze) | P2 |
| 2xL4 | dsv4_flash | 8 | 256 | cold_records_per_tok | 147.8 | 0 | 295.5 | 0 | 369.4 | CLOSED-FORM-ONLY (sim pending at freeze) | P3,P4 |
| 2xL4 | dsv4_flash | 8 | 256 | usd_per_1k_tok | 0.0902 | 0.0169 | 0.0974 | 0.0101 | 0.163 | CLOSED-FORM-ONLY (sim pending at freeze) | P15b,P15 |
| 1xL40S | dsv4_flash | 8 | 256 | decode_tps | 15 | 13.8 | 88.8 | 9.69 | 126.9 | CLOSED-FORM-ONLY (sim pending at freeze) | P15b |
| 1xL40S | dsv4_flash | 8 | 256 | host_hit_rate | 0.54 | 0.47 | 1 | 0.4 | 1 | CLOSED-FORM-ONLY (sim pending at freeze) | P2 |
| 1xL40S | dsv4_flash | 8 | 256 | cold_records_per_tok | 147.8 | 0 | 295.5 | 0 | 369.4 | CLOSED-FORM-ONLY (sim pending at freeze) | P3,P4 |
| 1xL40S | dsv4_flash | 8 | 256 | usd_per_1k_tok | 0.0954 | 0.0161 | 0.103 | 0 | 0.172 | CLOSED-FORM-ONLY (sim pending at freeze) | P14,P15b,P15 |
| 1xL4 | mimo_v2_flash | 1 | 16 | decode_tps | 1.16 | 0.898 | 1.89 | 0.629 | 2.71 | CLOSED-FORM-ONLY (sim pending at freeze) | - |
| 1xL4 | mimo_v2_flash | 1 | 16 | host_hit_rate | 0.51 | 0.39 | 0.759 | 0.27 | 0.839 | CLOSED-FORM-ONLY (sim pending at freeze) | P2,P2b |
| 1xL4 | mimo_v2_flash | 1 | 16 | cold_records_per_tok | 174.3 | 17.6 | 348.5 | 13.2 | 435.6 | CLOSED-FORM-ONLY (sim pending at freeze) | P3,P4 |
| 1xL4 | mimo_v2_flash | 1 | 16 | usd_per_1k_tok | 0.0953 | 0.0358 | 0.111 | 0.0215 | 0.186 | CLOSED-FORM-ONLY (sim pending at freeze) | - |
| 2xT4 | dsv4_flash | 1 | 16 | decode_tps | 0.178 | 0.135 | 0.254 | 0.0944 | 0.363 | REFERENCE-NON-PHASE6; CLOSED-FORM-ONLY (sim pending at freeze) | P3c,P5 |
| 2xT4 | dsv4_flash | 1 | 16 | host_hit_rate | 0.51 | 0.44 | 0.753 | 0.37 | 0.803 | REFERENCE-NON-PHASE6; CLOSED-FORM-ONLY (sim pending at freeze) | P2,P2b |
| 2xT4 | dsv4_flash | 1 | 16 | cold_records_per_tok | 179.4 | 74.1 | 358.8 | 55.6 | 448.5 | REFERENCE-NON-PHASE6; CLOSED-FORM-ONLY (sim pending at freeze) | P3,P4,P3b |
| 2xT4 | dsv4_flash | 1 | 16 | usd_per_1k_tok | 0.398 | 0.277 | 0.518 | 0.166 | 0.865 | REFERENCE-NON-PHASE6; CLOSED-FORM-ONLY (sim pending at freeze) | - |

---

## 2. Reference rows (NON-Phase-6 — never counted in the 60)

The 24 `*(ref)*` rows above (2 cells × 2 models × 2 b × 3 host budgets = 24 config cells × 4 metrics) are **reference rows, non-Phase-6**. They are not Modal campaign targets and do not count toward the 60:

- **2xT4 (anchor)** — the calibration/correctness instrument (fill-live-t4x2-20260909). It carries P3c (whole-generation TPS incl. prefill, 0.062-0.13, kill outside 0.05-0.16), P5 (eta_s 0.15-0.45, kill 0.10-0.55 on a second platform — the Phase-6 cells ARE the second platform for P5), P3b (device fills/token 190-260, kill 150-320, read from `cold_loads` per decode token), and at 16 GiB the P3/P4 anchor-sized cold bracket (nominal 179.4 [74.1-358.8]; P3's registered kill 70-260 tightens it). Its measured anchor values are 0.21 tok/s, host share 51.3% pooled-dedup (46.8%/56.1% per-GPU), 155.06 storage_requests/token — the calibration points behind the priors.
- **CPU-only** — portable-torch MEASURED floor (2,750 ms/expert, AGENTS.md CPU 6/10). Nominal decode TPS ~0.01-0.013; range upper edge = the **unbuilt** tuned-kernel arithmetic bound (25-30 ms/expert, ~90x) — DATA NEEDED (tuned AVX2 expert kernel). $/1k tok rows ($44-750) are included to keep the cost picture honest: CPU execution is 360-5,800x worse than dense (THEORY.md 6).

---

## 3. Assumptions ledger

Each assumption with its range and its FALSIFICATION.md citation (all constants also registered in `theory/predecl.py` with provenance → [`data/provenance.csv`](data/provenance.csv) picks them up automatically).

| # | assumption | value | range used in bands | tag | source / citation |
|---|---|---|---|---|---|
| A1 | COMPUTE_EFF (fraction of vendor dense-tensor peak realized by batched FP4→FP16 dequant GEMMs) | 0.6 | 0.3-0.8 (TPS/cost corners) | ASSUMPTION | FALSIFICATION.md "data needed" (GPU GEMM microbenchmark missing); provenance `compute_eff_assumption` |
| A2 | t_0 per-touch dispatch cost | 387.47 µs | 50-420 µs = union of P8c (50-150, batched kernel) and P8b (300-420, anchor dispatch path) | CALIBRATED on anchor (`T4_TOUCH_OVERHEAD_US`), carried cross-GPU | FALSIFICATION P8b/P8c |
| A3 | eta_s fill-over-compute hiding | 0.292 | 0.15-0.45 | CALIBRATED on anchor | THEORY.md 5.1; FALSIFICATION **P5** (0.15-0.45; kill 0.10-0.55 second platform) |
| A4 | eta_h H2D hiding | 0.5 | 0.3-0.7 | ASSUMPTION | THEORY.md 5.1 (`eta_h = 0.5` ASSUMPTION); width chosen here |
| A5 | B_SSD per cell | Cell spec: L4 2.5 / 2xL4 5.0 / A10 3.0 / L40S 6.0 / RTX 8.0 GB/s | 0.9-1.1× spec | ASSUMPTION | `constants.Cell` provenance (price/`B_ssd` all ASSUMPTION); live run must report measured B_SSD or the row is UNTESTABLE |
| A6 | popularity transfer `S_TRANSFER` (MiMo-Flash) | DSv4 closed form × MIMO_TOUCH_RATIO **1.1163** (48 layers × topk 6 / (43 × 6) = 288/258; records byte-identical 13,369,344 B) | MiMo hit band widened −12pp/+8pp (kill −24pp/+16pp); MIMO_TOPK=6 is itself ASSUMPTION | ASSUMPTION | FALSIFICATION "data needed" (no MiMo router traces exist); `S_TRANSFER`; **DATA NEEDED**: confirm routed top-k from `tools/phase3/specs/mimo_v2_flash.json` |
| A7 | Modal price schedule | L4 $0.000222/s, A10 $0.000306/s, L40S $0.000542/s, RTX PRO 6000 $0.000842/s, CPU $0.0000131/core/s, RAM $0.00000222/GiB/s | as listed (no band) | ASSUMPTION | AGENT_BRIEF.md price schedule; `MODAL_PRICES` |
| A8 | volume storage | $0.09/GiB/mo, first 1 TiB free | — | ASSUMPTION | `VOLUME_PRICE_GIB_MO`; 146.6 GiB store = **$3.8/mo** above free tier — negligible vs compute (THEORY.md 6) |
| A9 | **dense-residency baseline** | 8×H100-80GB-class, 568 GiB resident, 200 tok/s achieved | **price AMBIGUOUS — carried both**: 0.003223 $/GPU/s = $11.60/GPU/h → **$0.12892/1k tok (expensive reading)**; the note's "$3.223/GPU/h" = 0.000895 $/GPU/s → **$0.03581/1k tok (cheap reading)**; 3.57× sensitivity | ASSUMPTION | `DENSE_BASELINE` (Phase-7 correction 2026-09-30). **P15b/P15c as registered were written against the expensive reading** (ratios 0.49-0.93 / 1.08-2.29 reproduce against 0.12892). Under the cheap reading P15b **flips** (ratios 2.5-4.2× against dee). Every break-even kill criterion states its reading; until a dense stack is actually measured (P15) the reading is unresolved and break-even outcomes are graded "vs expensive-reading baseline" with the cheap-reading ratio reported alongside |
| A10 | host-tier hit priors | finite-window LRU prior 51% @ 16 GiB pooled saturating 0.536-0.54 (sealed repeat ceiling (5,099−2,364)/5,099); steady-Che as upper edge | finite −7pp / steady +5pp (DSv4); −12pp/+8pp (MiMo); kills at 2× | DERIVED | FALSIFICATION **P2** (0.48-0.54, kill 0.45-0.57), **P2c** (850-1000 of 2,364-2,500 ever repeat), known-failure 1 (**+25pp steady-Che error on short windows**); `HOST_HIT_CURVE` from `cache_store_knees.csv` |
| A11 | $/1k tok formula & basis | `$/1k tok = 1000 × rate_s / TPS`, steady-horizon TPS at the row's own b | — | DERIVED (economics convention) | THEORY.md 6 — its printed formula was **dimensionally wrong** (corrected 2026-09-30); reproduces `serve_cost_frontier.csv` + P14/P15b/P15c centers exactly |
| A12 | CPU expert execution | 2,750 ms/expert portable-torch | 25-30 ms/expert tuned-kernel bound (~90×) as range ceiling | MEASURED (nominal) + DATA NEEDED (ceiling) | AGENTS.md CPU 6/10; FALSIFICATION "data needed" (tuned AVX2 kernel) |
| A13 | host budget basis | 16/64/256 GiB **TOTAL**, split evenly over GPUs (engine host tier is per-GPU) | if provisioned **per GPU**, pooled basis doubles and multi-GPU rows shift ~one prior point up (16/GPU → 32 pooled ≈ 53%) | ASSUMPTION (convention) | `ws_lru_knee_gib` = 16 GiB **pooled** (provenance.csv); run must state its basis or row is UNTESTABLE |
| A14 | measurement protocol shape | ≥ 16 decode tokens; b=1 scored on the finite 16-token window (cold-ish start), b=8 on the steady regime | — | ASSUMPTION (protocol) | P1/P3 conventions (16-token horizon); P13/P9-P12 are explicitly NOT scored here (see §5) |

### 3b. Internal inconsistencies found in THEORY.md / FALSIFICATION.md / brief (frozen observations, reported not hidden)

1. **host_pack hit share arithmetic** — THEORY.md 5.2 (L327) says "host tier absorbs 23-25% of fills: host_pack hits 2,618/5,099" — but 2,618/5,099 = **51.3%**, not 23-25%. The 23-25% band is the **cuda0-only share (24.0%)**; pooled-dedup ground truth is **51.3%** (per-GPU 46.8%/56.1%) per `data/anchor_gate.json` `host_share.definitions` (also pre-noted in provenance.csv `anchor_host_share_definitions`). This matrix uses the 51.3%/finite-prior treatment; the Phase-7 brief's 23-25% band is definition-mismatched.
2. **dense H100 price ambiguity (3.57×)** — `DENSE_BASELINE` `price_gpu_s=0.003223` ($11.60/GPU/h) vs its own note "$3.223/GPU/h" (= 0.000895 $/GPU/s). Both readings carried (A9); flips P15b under the cheap reading.
3. **MiMo-Flash record count** — AGENT_BRIEF.md L22-23 says 12,032 records; THEORY.md L40-41 and `cache_store_knees.csv` say 48×256 = **12,288** (153.0 GiB). This matrix uses 12,288/153.0 GiB (MIMO_LAYERS=48 registered) and provisions 153 GiB scratch to cover either.
4. **THEORY.md 6's printed cost formula** `hardware_rate_s / (TPS_pred * 1000)` is dimensionally wrong; corrected to `1000 × rate_s / TPS` (serving.py implements the correct form; Phase-7 correction 2026-09-30).
5. **Cross-reference error** — THEORY.md 6 (L393) cites "(FALSIFICATION.md, P7)" for the dense-baseline falsification risk, but the dense-baseline prediction is **P15** (P7 is the popularity-entropy prediction). This matrix scores P15/P15b/P15c against the dense baseline.
6. **P4 bracket text vs construction** — the old P4 text "[46, 324]" mismatched its own construction (steady 67.84 / 2×finite 323.92); corrected by Phase-7 orchestrator 2026-09-30 to **[67.8, 323.9]** — carried here.
7. **P4 semantics vs batch rows** — P4's "[steady-Che, 2× finite-Che]" assumes steady ≤ finite; the substrate's **b=8** steady rows violate it (2xT4 h16: steady 420.4 vs finite 179.4 rec/tok) because the batch mixture `u_i(8)` inflates steady-Che. Bracket order-clamped to `[min(s,f), 2×max(s,f)]`; semantic tension flagged, not silently absorbed.
8. **`roofline_curves.csv` MiMo transferred rows are structurally suspect** — the `zipf_mandelbrot_transferred (ASSUMPTION)` rows imply ~8 cold records/token at b=1 (`U_per_layer` 0.125) against the structural bound L×topk = **288 distinct picks per decode token** (top-k picks are distinct per layer-call), and their cold/token scales **up** with b (8.2 → 22-29) while DSv4's scales down (156 → 124). This matrix therefore derives MiMo rows as **DSv4 × MIMO_TOUCH_RATIO (1.1163)** — structurally consistent (288/258) — and does not use the transferred rows; recorded here so the disagreement is visible.
9. **Horizon mixing in `serve_cost_frontier.csv`** — its b=1 rows embed **steady-state** TPS (not the finite-horizon decode TPS). That is the economics convention this matrix adopts for $/1k tok (and the P15c identity's basis), but the decode column and cost column are therefore **not the same horizon at b=1** (decode = finite 16-tok per P1; cost = steady per THEORY.md 6). Stated so the tables are not misread.

---

## 4. Scoring + kill-criteria summary (which measurement artifact field each metric reads)

| metric | measurement artifact field | registered range construction | kill rule (row band) | registered-id tightening (intersection governs) |
|---|---|---|---|---|
| decode TPS | `result.json` → `decode_tok_s` (cross-check `decode_tokens / decode_wall_s`) | nominal = `roofline_curves.csv` `tps_pred` (finite 16-tok at b=1, steady at b=8); range = T_pred envelope (A1-A5 corners) | outside [0.70×lo, 1.43×hi] | **P1** rows → FALSIFICATION kill outside **0.9-2.1**; **P1b** rows → outside **8-28**; **P14** row → its TPS cross-check outside **8-28** |
| host-tier hit rate | `result.json` → `host_pack` hits/(hits+misses), **pooled_dedup** definition (`data/anchor_gate.json` `host_share.definitions`; 51.3% anchor figure) | FINITE/STEADY band (A10): nominal finite-LRU prior, hi = steady-Che | outside [nominal−2×w_lo, steady+2×w_hi] | **P2** rows (host 16 GiB) → outside **0.45-0.57** (short windows); **P2b** additionally: LRU-vs-MIN gap < −6pp (needs Belady-MIN replay of the run's router journal) |
| cold records/token | `per_token_accounting` → `storage_requests / decode tokens` (`byte_accounting.storage_requests_per_generated_token`) | P4 bracket `[min(steady,finite), 2×max(steady,finite)]` (nominal = finite-Che) | outside [0.75×lo, 1.25×hi] | **P3** rows at anchor sizing (2xT4/16 GiB) → outside **70-260**; **P4** → any matched run outside the bracket [67.84, 323.92]-style construction kills P4 |
| $/1k tok | Modal **billing receipt**: `1000 × (billed_s × rate_s) / emitted_tokens` (cross-check `1000 × rate_s / decode_tok_s`) | steady-horizon `1000 × rate_s / TPS` (A11); range = relative T_pred envelope | outside [0.60×lo, 1.67×hi] | **P14** row → outside **$0.05-0.20**; **P15b** rows (b=8, 256 GiB) → ratio vs dense **≥ 0.95 kills** (registered band 0.49-0.93; **expensive dense reading**); **P15c** rows (b=1, 256 GiB) → ratio **< 0.85 kills** (registered band 1.08-2.29; expensive reading); the dense-price reading used must be recorded in the addendum |
| device fills/token (2xT4 ref only) | `cold_loads` per decode token | P3b band 190-260 | outside 150-320 | **P3b** governs directly |
| per-touch dispatch µs (from stage profile) | stage profile per-touch dispatch timing | P8b 300-420 (anchor path) | outside 250-450 | **P8c** (batched-kernel build): 50-150, kill > 250 |
| limiter identity (from stage profile) | stage profile bucket shares | row's `limiter` column (storage / h2d / compute / dense) | measured limiter ≠ predicted limiter at the cell's B_SSD | **P16** (≤0.4 GiB/s: storage-limited at all host ≤ 512 GiB — 2xT4 ref probes this), **P17** (≥5 GiB/s at 256 GiB: limiter moves to H2D/compute and TPS ≥ 0.5× dense-path bound — L40S/RTX rows probe this; a ≥5 GiB/s cell still storage-limited at 256 GiB kills P17) |

Grading: **PASS** = all four metrics in range · **KILLED** = any metric outside the effective (intersected) kill band · **WEAKENED** = in a range-edge/kill-edge gap · **UNTESTABLE** = missing prerequisite (§0) or ambiguous definition.

---

## 5. What each outcome will tell us (mapping to prediction IDs)

| outcome observed on a Phase-6 run | ids scored | what it tells us |
|---|---|---|
| 1xL4, b=1, h64 decode within 1.05-2.22 (P1 kill 0.9-2.1) | **P1** | the storage-limited T_pred (eta_s 0.292 calibrated on T4) transfers to Modal NVMe at B_SSD 2.5 GiB/s. Outside 0.9-2.1 → P1 KILLED → either the anchor's eta_s is platform-specific or the limiter regime differs |
| 2xL4, b=8, h256 decode within 13.5-78.3 (P1b kill 8-28) | **P1b** | the compute-bound steady regime at batch 8 with 2 GPUs (dispatch-cost dominated per assumption A2). Below 8 → the compute roofline is wrong or t_0 is worse than 420 µs |
| 1xL40S, b=8, h256 $/1k within 0.0161-0.103 + TPS 8-28 | **P14** | the flagship cost claim ($0.07-0.13, kill $0.05-0.20). Outside → the serving-economics formula or the Modal price assumption is wrong |
| host hit ≈ 51% at 16 GiB pooled (P2 kill 0.45-0.57) | **P2** | the finite-window LRU prior is right for short runs (and the +25pp steady-Che error claim is confirmed if the measured value sits far below steady-Che). MIN-replay gap of the same run scores **P2b** |
| cold rec/tok inside [min(s,f), 2·max(s,f)] (anchor bracket [67.84, 323.92]) | **P3**, **P4** | the Che bracket is valid on Modal hardware; at anchor sizing additionally P3's 70-260. `cold_loads/token` 190-320 scores **P3b** |
| stage profile: eta_s of the run 0.15-0.45 | **P5** | fill-hiding fraction transfers cross-platform (the T4-calibrated 0.292 held); kill 0.10-0.55 |
| 2xT4 whole-generation TPS incl. prefill 0.062-0.13 | **P3c** | the prefill-inclusive storage bound ×1.25-2.0 slack story on the anchor |
| b=8, h256 ratios 2xL4 0.70 / 1xL40S 0.74 / 1xRTX 0.93 vs 0.12892 | **P15b** (expensive dense reading) | dee beats dense residency by ≥ 25% at batch 8 — the core break-even claim. Any ratio ≥ 0.95 kills it. Report the cheap-reading ratio (vs 0.03581) alongside until P15 resolves the price |
| b=1, h256 ratios 1.08 (1xL4) … 2.29 (2xT4) | **P15c** (expensive reading) | dee does NOT win at single-stream anywhere (cold traffic exposed). Any ratio < 0.85 kills it |
| measured dense stack ≥ 200 tok/s at ≤ $0.13/1k (or > 400 tok/s at ≤ $0.26/1k) | **P15** | the conservative competitor assumption holds / is killed — and adjudicates the 3.57× price ambiguity (A9) |
| per-touch dispatch 300-420 µs (anchor path) / 50-150 µs (batched build) | **P8b** / **P8c** | the cross-GPU t_0 carrying holds / a batched kernel exists (rows would move toward their hi edges — a WEAKENING, not a kill). > 250 µs on a batched build kills P8c |
| limiter = storage on 2xT4 (0.33 GiB/s) at every host budget | **P16** | the storage-limited regime claim at ≤ 0.4 GiB/s |
| limiter ∈ {h2d, compute} on 6.0-8.0 GiB/s cells at h256, TPS ≥ 0.5× dense-path bound | **P17** | the regime shift at ≥ 5 GiB/s; a ≥ 5 GiB/s cell still storage-limited at 256 GiB kills P17 |
| **NOT scored by this matrix** | P2c (trace replay: 850-1000 repeating records), P6/P6b/P7/P8 (popularity fits — need fresh router windows), P9-P12 (prefetch — no prefetch surface in these configs), P13 (K=1 vs K=16 **concurrency** — b∈{1,8} is batching, not cross-request mixing) | each needs its own artifact (multi-request trace, MIN replay, hint A/B); DATA NEEDED per FALSIFICATION.md |

---

## 6. ADDENDUM — post-registration (append-only; frozen ranges above are never edited)

**Status at freeze (2026-09-30): PENDING.** Checked at freeze time: `dee.cpp/theory/data/sim_*.json` — **ABSENT**; `dee.cpp/theory/data/solve_*.csv` — **ABSENT**. Component A's simulator (running in parallel) and Component B's solver had produced no outputs in this worktree at registration. Every row's sim column therefore reads **"pending sim"** (`source_tag = CLOSED-FORM-ONLY (sim pending at freeze)`), and there is nothing to reconcile yet.

**Procedure when sim/solve outputs land (does not touch frozen ranges):**
1. Re-run `python -m theory.run_all predecl` with `results_SIM` / `results_SOLVE` passed to `theory.predecl.run(...)` (accepted shapes documented in `theory/predecl.py`). Matching rows flip to `SIM-PREDICTED+CLOSED-FORM`; `predecl_meta.json → sim_reconciliation` records per-row `abs_disagreement` / `rel_disagreement`.
2. Fill the table below with **both numbers** and the disagreement magnitude, timestamped. Where sim and closed form disagree, BOTH are reported — disagreement is data.

### Addendum 1 — sim-vs-closed-form reconciliation (PENDING, 2026-09-30)

| cell | model | b | host GiB | metric | closed-form nominal | sim | abs diff | rel diff | agreement |
|---|---|---|---|---|---|---|---|---|---|
| *(all 84 config cells)* | | | | | *(see §1)* | **pending sim** | — | — | PENDING (2026-09-30; `data/sim_*.json` absent at freeze) |

*(later addenda append below, each with its own UTC timestamp and heading; section 0-5 stays byte-frozen)*
