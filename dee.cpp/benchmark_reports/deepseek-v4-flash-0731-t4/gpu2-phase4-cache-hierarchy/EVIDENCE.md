# Phase 4 Evidence Bundle — gpu2-phase4-cache-hierarchy

Immutable evidence for the Phase-4 full-store cache-hierarchy campaign
(2xT4 Kaggle run, kernel `nivind/dee-cpp-p4-cache-campaign`).

- Commit under test: `216ad6553ebd961c6688e335f79f6a7588dd3653`
  (recorded in every `integrity-q*.json` `git_commit` field)
- Results commit: `ee132b1` (`PHASE4_RESULTS.md`)
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` rev
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Workload: 8 prompts x 128 forced tokens (`NATIVE_IGNORE_EOS=1`)
- Arms: `a0_bypass` (uncached honest baseline, timed out by design),
  `a1_asis` (fp16 + rank_priority reference), `a2_fp4` (fp4 + LRU
  candidate), `a3_fp4p` (fp4 + rank_priority policy isolation)

## Verify

    python verify.py

Recomputes every headline number from this bundle alone:
58 checks — acceptance (8x ACCEPT_CORRECTNESS + 128 tok per arm),
token-sha and journal-sha equality across all three completed arms,
journal shas vs Kaggle-side integrity records (24/24), 278,879
requests/arm, residency/host/cold rates, policy delta, H2D + wall
totals, checkpoint token chains, manifest self-consistency.

## Layout

- `evidence/{arm}/` — per-prompt artifacts copied verbatim:
  `routed_experts-q*.jsonl` (route journals), `generated_checkpoint-q*.jsonl`
  (per-step token ids + rolling chain sha), `integrity-q*.json`
  (Kaggle-side artifact shas, commit, classification), `arm_config`,
  `environment`, `memory`, `run_config*`, plus per-arm `run_config.json`,
  `dee4-segmented-store.json`.
- `evidence/_extract/` — compact extracts derived locally from the raw
  download: `{arm}-result-q{i}.json` (metrics incl. token/decoded shas,
  H2D, wall, scoped counters), `{arm}-runner-result-q{i}.json` (runner
  originals cross-checked identical), `{arm}-profile-q{i}.json` (compact
  profile incl. per-token decode timings), `{arm}-ce-q{i}.json` +
  `{arm}-ce-summary.json` (cache-event kind counts), and
  `file-sha-manifest.json` (sha256 of every raw file too large to
  commit, computed before deletion).
- `evidence/kernel-metadata.json`, `phase4_session_driver.py` — exact
  campaign definitions (driver is the repo copy under test).
- `SHA256SUMS.json` — sha256 over every committed file.
- `token-sha-manifest.json`, `journal-sha-manifest.json` — regenerated
  by `verify.py`.

## Provenance and limitations (read before citing)

- Raw source: `kaggle kernels output nivind/dee-cpp-p4-cache-campaign`,
  two pulls. First pull truncated mid-harvest (a3 journals missing,
  ~594 MB of 6.8 GB); second pull complete. All three completed arms
  are fully present here — journal equality is verifiable for
  a1/a2/a3, not only a1<->a2.
- `p4_report.json` was never written: the Kaggle session ended during
  final harvest, after all arms completed. The driver's always-write
  final report does not exist; this bundle + `PHASE4_RESULTS.md` were
  reconstructed from per-prompt artifacts.
- `a0_bypass` timed out by design (~385 s/token through a 1-record
  bounce buffer, lanes=1); its partial harvest (9 files) is included
  as the honest-baseline datapoint.
- Not committed (too large for git; shas preserved): 40-55 MB
  `result-q*.json` / `profile-q*.json` per prompt (compact extracts
  committed instead; full-file shas in
  `_extract/file-sha-manifest.json` and Kaggle-side shas in each
  `integrity-q*.json`), ~25 MB `cache_events-q*.jsonl` streams
  (per-kind counts committed; integrity files pin their shas),
  ~300 MB/arm driver + progress logs (sha records only — destroyed in
  local condensation after hashing; not recoverable from this bundle).
- Top-level a3 `result-q1.json` synced as 0 bytes (Kaggle-side
  truncation); the harvested arm-dir copy is complete and verified.
- `kernel-metadata.json` committed here is the repo file whose sha256
  matches the `kernel_metadata_sha256` recorded by integrity files.

## Status

INDEPENDENTLY RECOMPUTED (verify.py, this bundle): all headline
metrics — 45.3%/7.4%/13.2% residency, 56.1%/18.0%/52.7% host-hit,
36.5%/36.7%/34.1% cold, +32.1pp policy delta, 3,454/2,038/3,237 GB
H2D, 5,848/5,345/5,717 s wall, 278,879 requests/arm, token+journal
equality a1<->a2<->a3.

CAMPAIGN-REPORTED ONLY (not in bundle): per-request cache-event
records, driver console logs, `p4_report.json` (never existed),
a0's partial-q0 internals beyond the 9 harvested files.
