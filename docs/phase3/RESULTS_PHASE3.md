# RESULTS.md — Phase 3: Arbitrary-Prompt Inference over the Full Expert Universe

**Verdict: PASS** — Kaggle kernel `nivind/dee-p3-gpu2-fullstore` v13, 2026-09-15.

Authoritative evidence bundle:
`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/gpu2-phase3-fullstore/`
(report: `gpu2_report.json`; per-arm logs, per-prompt results, route
journals, and per-token checkpoints included in full).

---

## Hypothesis

The `dee4-v4-segmented` expert store — 46 segments, 11,776 records
(43 routed layers x 256 experts + 3 MTP buckets), ~146.6 GiB packed
`dee4` records — can serve **arbitrary** prompts end-to-end through the
exact runtime, resolving every `(layer, expert)` the native router emits,
and produce output **byte-identical** to the authoritative safetensors
checkpoint path.

## Exactness contract

Placement, caching, scheduling, and transfer timing may change; routing
decisions, expert identity, checkpoint representation semantics, expert
execution semantics, expert ordering, and outputs may not. Prediction may
drive prefetch hints only. Success therefore requires identical generated
token ids AND identical complete route journals, not just similar text.

## Experiment configuration

| Item | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` rev `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Store | `dee4-v4-segmented`, 46 segments x 256 records, `record_bytes=13369344`; `universe_sha256=2081ada5e37e…`, `manifest_sha256=70ee9269cd26…` |
| Store integrity identity | `4846d482b4f091bf0c3e6f74c72c0a1e72a243c75c31cacdfe64ee51c69a0e59` (reported live by both engines, all prompts) |
| Commit | `research/phase3-gpu2-inference @ 40e2050407dbf2c01a59a4cfbf937477118010ce` |
| Hardware | Kaggle 2x Tesla T4 (SM75) |
| Prompts | 3 arbitrary prompts (prose / code / history), `NATIVE_PROMPTS_JSON` — one process per arm, engines+store opened once |
| Generation | 16 tokens per prompt, greedy |
| a0 baseline | `expert_store=safetensors` over mounted checkpoint shards |
| a1 candidate | `expert_store=dee4_segmented` over the 46 mounted segment datasets (symlinked — **zero staging** into /tmp) |
| Read config | `source_read_lanes=4`, queue depth 6, both arms |
| Integrity | Driver re-sealed all 46 segments via sequential pread in-session: **46/46 sha256 match, 5.9 min** |

Driver + kernel metadata preserved alongside the report
(`session-driver.py`, `kernel-metadata.json`).

## v12 failure and root cause

v12 (commit `71320f0`) failed with a0 TIMEOUT at 14,400 s — not a crash.
Root cause, quantified from a0's checkpoint telemetry: the safetensors
single-lane fill memcpy'd expert bytes out of FUSE-mmap'd shard files —
`mmap_memcpy` at **2.3-2.7 MiB/s, ~4,700-5,500 ms per 12.75 MiB record**
(~60x slower than pread). a0 reached only 12/16 tokens on q0. The overlap
that completed was already exact (553/553 routing records, 12/12 tokens
identical), which made the defect clearly a storage-path problem, not a
correctness problem.

## v13 repair: `pread_gather`

`SafetensorsExpertStore::materialize()` (commit `40e2050`) now gathers the
six tensor regions (`w1|w3|w2|s1|s3|s2`) via `pread()` on the owning
shard's fd — file offset = `view.data - shard->base()`, whole-file mapping
so the pointer delta is exact. The bounded batch path
(`Engine::prepare_fp4_experts`) accepts non-contiguous views when
`can_gather_materialize()`; byte output is identical (asserted in
`test_expert_store.cpp` against per-region memcpy, plus fail-closed
cases). dee4 stores are unaffected: they still require the contiguous
record check.

## v13 results

| Prompt | tokens | a0==a1 token sha256 | a0==a1 journal sha256 | decoded (a0) |
|---|---|---|---|---|
| q0 | 16/16 | `4537a0525e77…` | `591072115feb…` | "The sky appears blue because air molecules scatter shorter (blue) wavelengths of sunlight much" |
| q1 | 16/16 | `15897d13b715…` | `0d4f1a12e469…` | "Here's a Python function that returns the nth Fibonacci number:…" |
| q2 | 16/16 | `bb9324463d67…` | `ee0808ddfc21…` | "The French Revolution (1789–1799) was a complex event with deep" |

- Route journals: **688 records per prompt** — all 43 routed layers x 16
  forwards, canonical order, chain-hashed; identical files across arms.
- Store reads: a1 `dee4_segmented` 0 lookup failures, all `pread`;
  a0 `safetensors` 0 lookup failures, all `pread_gather`.

### Measured performance

| Arm | Wall (3 prompts, incl. model+engine init) | Expert-read bandwidth | avg read latency |
|---|---|---|---|
| a0 safetensors | 699.9 s | 149-174 MiB/s | ~75-86 ms/record |
| a1 dee4_segmented | 472.8 s | 179-197 MiB/s | ~67-71 ms/record |

Read-path microbench on the mounted store: `mmap=7ms/rec`,
`pread=2ms/rec`. a0's gather (~6 preads/record) sustains ~85-90% of a1's
contiguous-read bandwidth.

## Limitations

- The three prompts exercised only the routed subset — **not** all 11,776
  experts. The claim is arbitrary-prompt capability: any `(layer,expert)`
  the router emits is resolvable (0 lookup failures); expert coverage is
  whatever routing selected.
- Bandwidth numbers are Kaggle dataset-mount (FUSE) specific; local NVMe
  is faster. They establish feasibility, not a production rate.
- 16 tokens/prompt at greedy decoding; longer generations were not
  measured in this campaign.

## Conclusion

Phase 3 demonstrates arbitrary-prompt inference using authoritative
checkpoint routing and exact expert retrieval from the complete published
expert universe, with identical complete routing journals and
generated-token sequences against the original safetensors reference
backend.
