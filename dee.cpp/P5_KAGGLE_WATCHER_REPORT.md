# P5 KAGGLE WATCHER REPORT — `nivind/dee-cpp-dsv4-phase5-cohort`

> **Follow-up:** post-run forensic triage is in **`P5_EXACTNESS_FORENSICS.md`** — root cause narrowed to warm-process GPU-numerics divergence (cuBLAS-class), with the minimal gated mechanism test defined there.

**MORNING SUMMARY: FAIL (verdict) / CLEAN RUN (execution).** The kernel ran to completion in ~4h52m of driver time with zero infrastructure failures — clone, build, 46-segment store assembly + seal verification, all 16 units (c0_anchor + 8×c1 + 4×c2 + 2×c4 + 1×c8h) exited 0 and classified `ACCEPT_CORRECTNESS`. The authoritative `p5_report.json` verdict is **FAIL**, driven entirely by the 24 "cohort row == c1 exact" token-SHA checks: no cohort row reproduces its singleton reference bit-exactly. Independent verification (`p5_verify.py`) confirms the same result: 91 PASS / 48 FAIL, exit 1. Important context for triage: the singleton reference itself is *not stable* — c1-c0 vs c1-c7 ran the byte-identical prompt text (prompts 0 and 7 are both the mRNA prompt) and produced different outputs (one coherent, one a degenerate ` The.??` loop), sharing only 6/5504 route-journal records. The cohort engine is internally deterministic (c8h members 0 and 7 produced identical prefill journals AND identical 128-token streams), so cohort≠c1 is a real, reproducible mode difference — but the exactness gate is calibrated against a reference that does not reproduce itself. See "Failure analysis" — the divergence originates at prefill layer 3 (first MoE layer), and shows a member-index asymmetry: member 0 ≈ clean vs singleton, members >0 diverge at essentially every real position.

---

## 1. Kernel status & wall time

- **Final status:** `KernelWorkerStatus.COMPLETE` (first observed COMPLETE at poll 05:11:49 CDT 2026-09-21; observed RUNNING continuously from first poll 23:58 CDT 2026-09-20, 32 polls at ~10-min cadence).
- **Driver wall (kernel log clock, UTC):** first log line `[05:08:59]` → verdict `[10:00:43]` = **17,505 s ≈ 4h52m**. (`P5 VERDICT: FAIL` printed at log t=17505.2 s; nbconvert finished ~t=17510 s.)
- **Session/driver setup:** clone→cmake→dee_core+pydee builds→test_dee4_segmented all PASS by 05:10:27; 46 buckets assembled 05:10:27; **46 segment seals verified** 05:15:58 (`bad=[]`).
- **Branch/commit under test:** `freebuff/deepseek-v4-flash-0731-t4` @ `9d6d403d134133f5c50146fd79a00a7409c8d99a` (branch head, unpinned); model_revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`; 2× Tesla T4 (sm_75).
- **Artifacts downloaded:** 345 files, ~3.85 GB → `kaggle/deepseek-v4-flash-0731/p5-kaggle-out/` (per-arm dirs under `p5-out/` plus a flat top-level dump of the last arm, c8h). Kernel log: `dee-cpp-dsv4-phase5-cohort.log` (187 KB).

## 2. `p5_report.json` verdict & check list

**`"verdict": "FAIL"`** (last line of report). 54 checks total: 30 ok, 24 not-ok — every not-ok is a `row pX == c1 exact` token-sha check.

| Check | Result | Detail |
|---|---|---|
| 2x T4 | PASS | two T4s, UUIDs recorded |
| clone / branch head | PASS | `9d6d403d1341` |
| cmake configure, dee_core, dee_test_assets, pydee builds | PASS | warnings only |
| test_dee4_segmented | PASS | ctest 1/1 |
| index dataset mounted / segment table 46 | PASS | `missing: []` |
| 46 segments assembled / seals verified | PASS | `bad=[]` |
| p4 token manifest readable | PASS | 8 prompts |
| **c0 anchor == phase4 a2 q0** | **PASS** | `got=7613595e9707 exp=7613595e9707` — rebuilt stack reproduces sealed P4 evidence |
| c1-c0..c1-c7 accepted | 8× PASS | all `ACCEPT_CORRECTNESS`, K=1 |
| c2-c0..c2-c3 accepted | 4× PASS | `ACCEPT_CORRECTNESS`, K=2 |
| c4-c0, c4-c1 accepted | 2× PASS | `ACCEPT_CORRECTNESS`, K=4 |
| c8h-c0 accepted | 1× PASS | `ACCEPT_CORRECTNESS`, K=8 |
| **cohort row == c1 exact (24 rows: c2×8, c4×8, c8h×8)** | **24× FAIL** | every cohort row sha ≠ c1 sha for same prompt_index |

No `host_pack_clamped` flags appear anywhere in the report. Engine counters clean throughout: `fill_failures=0`, `alloc_failures=0`, `budget_rejections=0`, `lookup_failures=0`, `pread_short_reads=0` on both GPUs for all units.

## 3. Independent verifier (`tools/phase5/p5_verify.py`)

Command: `python tools/phase5/p5_verify.py kaggle/deepseek-v4-flash-0731/p5-kaggle-out/p5-out` → **exit 1**.

- **91 PASS / 48 FAIL** (the 48 = 24 row-token-sha mismatches + 24 route-journal `routes == c1` divergences).
- Passed: P4 manifest present; c0 integrity + **anchor sha recomputable** (`7613595e9707`); all 8 c1 sha recompute+accept; all 16 journals canonical order + `chain_sha256` recompute (5504 records each = 128 steps × 43 layers); journal completeness (K×L\* prefill rows); all checkpoint fan-out / result-sha/stream/checkpoint agreements for c2/c4/c8h rows.
- Failed: for every cohort unit — `row pX == c1` and `row pX routes == c1`. All route divergences report the same locus: **`divergence at (0, 3)`** = forward_step 0 (prefill), layer 3 — the first MoE layer. Layers 0–2 prefill rows match c1 exactly.

## 4. Per-arm results

| Unit | K | wall_s | emitted | cls | notes |
|---|---|---|---|---|---|
| c0_anchor-q0 | 1 | 855.3 (total 687.3) | 128 | ACCEPT_CORRECTNESS | sha `7613595e97…` **== P4 a2 ref**; coherent mRNA answer |
| c1-c0..c7 | 1 | 3212.8 arm (unit totals 219–625 s) | 8×128 | all ACCEPT_CORRECTNESS | c0 coherent; **c7 degenerate loop** (`" to the.?? …"` ×128) despite same text as c0 |
| c2-c0..c3 | 2 | 3483.9 arm (unit 447.7–1229.0 s) | 4×256 | all ACCEPT_CORRECTNESS | rows coherent/on-topic; dedup_ratio ≈1.21–1.38 |
| c4-c0..c1 | 4 | 4067.9 arm (unit 1206.4, 2641.6 s) | 2×512 | all ACCEPT_CORRECTNESS | dedup_ratio ≈1.85 (c1) |
| c8h-c0 | 8 | 5443.3 (cohort 5216.0) | 1024 | ACCEPT_CORRECTNESS | dedup_ratio 1.61; rows p0==p7 byte-identical |

Throughput context: T4s are host-pack-bound — c0 decode ~0.22 tok/s (p50 ITL ≈ 4.3 s, `evict_until_free` churn every token; `theoretical_min_cache_bytes` ≈ 9 GB ≫ 3.5 GiB/GPU budget). c8h emits 1024 row-tokens in 5443 s ≈ 0.19 tok/s aggregate vs c1's 1024 tokens in 3213 s ≈ 0.32 tok/s — K=8 cohorting did **not** improve aggregate throughput on this run (reads still dominated; dedup cut unique staged experts ~27% vs 8 sequential runs).

## 5. Failure analysis — what diverges and where

Every cohort row fails the bit-exact gate vs its c1 reference, but the artifact evidence shows this is a *routing-numerics* divergence, not a serving/integrity bug:

1. **Divergence starts at prefill, layer 3** (first MoE layer; DSv4's layers 0–2 records match exactly between each cohort member's slice and the c1 reference, including pad rows). Decode then amplifies it.
2. **Prefill journal layout verified as [K, L\*] member-major**, pad-left (`pad_tokens`: p0=18, p1=7, p2=3, p3=0, p4=20, p5=19, p6=22, p7=18 of L\*=32). Pad rows carry a content-independent routing signature (`[153,180,48,251,216,30]`-family) and match across runs — they are not the defect.
3. **Member-index asymmetry** (c8h layer-3 prefill real rows vs c1 refs): member0 13/14 match (the 1 diff = same set, tie-order flip); members 1–7 = **0/N match** — every real position routes to a different top-6 *set*. Same pattern in c2-c0 (member0 12/14 w/ 2 tie-flips; member1 0/25). Members >0 are perturbed at essentially every position — consistent with member>0 hidden states being computed under different numerics than member0 (batched-kernel reduction order, workspace sharing, or cross-member contamination), rather than uniform per-row jitter.
4. **Cohort mode is internally deterministic:** c8h members 0 & 7 (byte-identical prompt text) → identical 32/32 prefill rows and identical 128-token output sha `c3e8eceb1a4a`. Cross-arm: p1 `2d25792e1197` identical in c2 and c8h; p2 `6814a55107bf` identical in c4 and c8h.
5. **BUT the reference itself is unstable:** c1-c0 vs c1-c7 (identical prompt, separate cold-reset K=1 runs) → **0/14** real-row match at layer-3 prefill, only **6/5504** journal records identical, and textually different outputs (coherent vs `The.??` loop). So even a "correct" cohort could never satisfy `== c1 exact` reliably; the W3 gate measures against a nondeterministic baseline. The member0≈clean / members>0≈fully-diverged split is still real evidence of a member-index-dependent effect worth investigating — it is *larger* than the singleton noise floor (member0 tracks c1, member1+ do not track their own references at all).
6. **`c4-c1` row p7 sha `b848af633114` == `c1-c5` sha — explained, benign:** both are degenerate streams of 128× token `33`; the collision is the degenerate fixed-point, not cross-row contamination (verified: `generated_token_ids` all `33` in both `c4/result-c1-r3.json` and `c1/result-c5-r0.json`).
7. Sample generations are coherent and on-topic despite sha mismatch: c2-c0-r1 → valid LCS dynamic-programming answer; c8h-r3 → sci-fi novel opening; c8h-r0/r7 → identical coherent mRNA answers. Classification `ACCEPT_CORRECTNESS` reflects structural validity, not gate equality.

Log tail at verdict (kernel log, UTC):
```
[10:00:43] [p5] PASS c8h-c0 accepted cls=ACCEPT_CORRECTNESS K=8
[10:00:43] [p5] FAIL c8h-c0 row p0 == c1 exact cohort=c3e8eceb1a4a c1=b28d95485c46
… (23 more identical-pattern row fails)
P5 VERDICT: FAIL
```

## 6. Artifact inventory (`p5-kaggle-out/`, 3.85 GB, 345 files)

- `p5-out/p5_report.json` (150 KB) — checks, per-unit counters (host_pack / expert_store / engine_stats per GPU), byte accounting, run envs, verdict.
- `p5-out/token-sha-manifest-p4ref.json` — 8-prompt P4 reference manifest.
- `p5-out/{c0_anchor,c1,c2,c4,c8h}/` — per-arm: `result*.json` (per-cohort + per-row, incl. `decoded_text`), `integrity*.json`, `routed_experts-*.jsonl` journals (5504 recs each), `generated_checkpoint-*.jsonl`, `cache_events-*.jsonl` (c8h: 153 MB / 203,781 records), `profile*.json` (up to 222 MB), `memory*.json`, `cohort*.json`, `arm_config`/`run_config`/`environment`, `log-*.txt`, `progress.log`, `dee4-segmented-store.json`, `native-generate-*.json`.
- Top level: flat re-dump of the c8h working dir + `dee-cpp-dsv4-phase5-cohort.log`.
- Verifier transcript: `/tmp/p5_verify_full.txt` on watcher host (190 lines; 91 PASS / 48 FAIL).

## 7. Suggested triage (non-authoritative)

- The exactness gate compares cohort output to a c1 reference that is itself non-reproducible (same prompt → different output between c1-c0 and c1-c7). Either the gate needs a tolerance/fixed-point definition, or the engine nondeterminism (router-score tie sensitivity under eviction-heavy execution) must be pinned first.
- Investigate the member-index asymmetry at prefill layer 3: member 0 tracks singleton ~93%, members >0 track ~0%. Candidates: batched router GEMM reduction order per row-block, cohort workspace/accumulator sharing across members, or per-member hidden-state offset bug in the packed prefill path.
- Budget note: this consumed the authorized Phase-5 GPU run (GPU lane 2/2 by the resolved ledger interpretation — the campaign doc calls it the authorized P5 run). No further GPU pushes were made by this watcher.

*Watcher run: 2026-09-20 23:58 CDT → 2026-09-21 ~05:30 CDT. 32 status polls; single output pull; local verify only. No repo files modified.*
