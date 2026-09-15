# Handoff prompt — GPU2 Phase-3 full-store run (v12) evidence triage

Copy everything below the line into a fresh Arena.ai agent session that has the
evidence files attached / accessible.

---

## Context

I have the evidence output of a Kaggle dual-T4 notebook run and I need you to
triage it. The repo is `so-nerdyy/dee`. The run was pinned to branch
`research/phase3-gpu2-inference` at commit `71320f0c7bf410fb14f111cf36d13b93d104746f`.
Clone it and read these before touching the evidence, because the acceptance
logic lives in code, not in the log:

- `dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_native_generate.py`
  (the runner; see `classify_full_generation()` and the `_run_prompt` /
  `NATIVE_PROMPTS_JSON` multi-prompt loop)
- `PHASE3_FULL_EXPERT_STORE.md`, `PHASE3_SEGMENTED_READER.md` (design + reader
  semantics)
- `dee.cpp/src/expert_store.cpp` (`Dee4ExpertStore::open`, segment table
  validation, seal verification)
- `dee.cpp/src/engine.cpp` around the `DEE4_STORE_SKIP_SEAL` env check

## What the run did

Phase-3 question: can `dee.cpp` serve **arbitrary** prompts from the complete
46-bucket `dee4-v4-segmented` expert store (11,776 records, 146.6 GiB) with
byte-exact behavior versus the authoritative safetensors backend?

Two arms, same kernel, same 3 prompts, 16 tokens each:

- **A0** = `expert_store=safetensors` (reference), `NATIVE_SOURCE_READ_LANES=1`
- **A1** = `expert_store=dee4_segmented` (candidate), `NATIVE_SOURCE_READ_LANES=4`,
  `DEE4_STORE_SKIP_SEAL=1`

Both arms share `NATIVE_RUN_ID=gpu2-fullstore` deliberately: run_id is baked
into every route-journal record's canonical payload, so identical routes must
produce identical journal bytes across arms. Per-prompt journal *files* keep
the prompts separate.

Accept criteria per prompt: A1's `generated_token_ids`, `decoded_text`, and
routed-experts journal sha256 all identical to A0's; A1 reports
`backend=dee4_segmented`, zero `lookup_failures`, all source reads contiguous,
43 layers executed, route journal complete.

## What happened

```
a1 exit=0 wall=534.1s          <- 3 prompts, completed normally
a0 TIMEOUT wall=14402.6s       <- hit the 14400s subprocess cap
FAIL q0 a0==a1 exact tok=4537a0525e77 journal=591072115feb
PASS q0 a1 store clean {"cuda0": {"backend": "dee4_segmented",
  "integrity_identity": "4846d482b4f091bf0c3e6f74c72c0a1e72a243c75c31cacdfe64ee51c69a0e59",
  "lookups": 1695, "lookup_failu...
FAIL q1 a0==a1 exact tok=15897d13b715 journal=0d4f1a12e469
PASS q1 a1 store clean ... "lookups": 3196 ...
FAIL q2 a0==a1 exact tok=bb9324463d67 journal=ee0808ddfc21
PASS q2 a1 store clean ... "lookups": 4622 ...
GPU2 VERDICT: FAIL
```

Also from the log: all 46 segment seals verified against the build-time
segment table (`bad=[]`, 6.6 min wall, ~97 MiB/s per stream x 4 workers over
157,437,394,944 bytes); `test_dee4_segmented` ctest passed; 46 segments
symlinked into the store tree in 0s.

**My read, which I want you to verify or refute:** the `FAIL` is a missing
baseline, not a divergence. The three `a0==a1` checks fail on `accepted(a0)`
being false (A0 has `classification=TIMEOUT`, no result file was read), never
on a hash comparison. A1's evidence hashes are non-null in the same log lines,
so A1 produced three complete outputs.

## The suspected recoverable data — check this FIRST

The driver's `TimeoutExpired` handler returns `classification=TIMEOUT` for all
three prompts **without ever reading** `native-generate-result-q{0,1,2}.json`
or `routed_experts-q{0,1,2}.jsonl` from `/kaggle/working`. But the runner
writes and fsyncs those per prompt as it goes. A0 needed roughly 80 min/prompt
and ran 240 min, so it plausibly finished q0 and maybe q1 before the cap.

Critical naming rule for attribution:

- Files the **runner** writes into `/kaggle/working` are **untagged**:
  `native-generate-result-q0.json`, `routed_experts-q0.jsonl`,
  `generated_checkpoint-q0.jsonl`
- Files the **driver** collects into `gpu2-out/` are **arm-tagged**:
  `result-a1-q0.json`, `routed_experts-a1-q0.jsonl`, `checkpoint-a1-q0.jsonl`

And `run_arm()` deletes all untagged files at the **start of every arm**. Arm
order was **a1 first, then a0**. Therefore any untagged file surviving in the
final output was written by **A0** after that deletion. Confirm this reasoning
against `run_arm()` in the driver before relying on it.

I believe at least `routed_experts-q0` (untagged) exists in my output.

## What I want from you

1. **Inventory.** List every evidence file, classify each as A0 / A1 / stale,
   and justify each call. Note: `dee4-metadata.json`, `dee4-repack_report.json`,
   `dee4-import-validation.json`, `dee4-serving-access-benchmark.json`,
   `p2.2-dee4-evidence.json` are written by the *repack* branch, which the
   `dee4_segmented` path skips via `_SkipDee4Prepare` before reaching it —
   so those are likely stale from an earlier run. Verify.

2. **Recover A0.** For every prompt where untagged A0 evidence exists, extract
   `generated_token_ids`, `decoded_text`, `classification`, and the journal
   file, and compute the same three hashes the driver computes:
   - `token_ids_sha256` = sha256 of `json.dumps(generated_token_ids)`
   - `decoded_sha256` = sha256 of `decoded_text.encode()`
   - `journal_sha256` = sha256 of the journal file bytes
   Match the driver's exact serialization — copy it from the driver, don't
   reimplement from memory.

3. **Compare against A1.** A1's truncated hash prefixes are above
   (`q0 tok=4537a0525e77 journal=591072115feb`, `q1 tok=15897d13b715
   journal=0d4f1a12e469`, `q2 tok=bb9324463d67 journal=ee0808ddfc21`);
   full values are in `result-a1-q*.json` / `gpu2_report.json`. Report per
   prompt: MATCH / DIVERGE / NO A0 DATA. If anything diverges, find the first
   differing token index and the first differing journal record.

4. **Re-derive the verdict honestly.** Per prompt, apply the driver's real
   `accepted()` + store-clean gates. For arbitrary (non-sealed) prompts the
   seal gates are excluded via `SEAL_APPLICABLE`, so `ACCEPT_CORRECTNESS` is
   reachable — confirm that in `classify_full_generation()`. Give me a verdict
   table and say plainly what is proven, what is unproven, and what is still
   missing.

5. **Check A1's internal consistency** even without A0: 43 layers executed,
   `record_count == n_tokens * 43`, `topk == 6`, `lookup_failures == 0`,
   `contiguous_source_reads == source_reads`, finite outputs, and that
   `integrity_identity` is stable across all three prompts and equals the
   segment-table digest (sha256 over the ordered concatenation of the declared
   per-segment sha256 hex strings) from the store `metadata.json`.

## Caveats to carry into your analysis

- **`DEE4_STORE_SKIP_SEAL=1` means A1's engine never content-verified the
  store at open.** The `dee4_integrity` gate only checks that
  `integrity_identity` is 64 hex chars, and for segmented stores that identity
  is the *declared* segment-table digest, not a re-hash of bytes. The driver's
  own P2b seal pass is the entire content guarantee. Don't let the gate name
  imply more than it proves.
- The read-path microbench in the driver is **not** sound evidence: it runs on
  `bucket-45` moments after the seal pass read it (page-cache warm), and the
  pread loop reuses the same offsets the mmap loop just faulted in. Treat its
  `mmap=8ms / pread=2ms` as an artifact.
- `lookups` on cuda0 is cumulative across prompts (1695 -> 3196 -> 4622) and
  only increments on a host-pack-cache miss.

## Then recommend next steps

Rank by cost. Specifically evaluate this option, which I think is cheapest:
instead of re-running A0 end to end, use A1's route journals (which enumerate
every `(layer, expert)` actually served) to do a **record-level byte
equivalence check** — pull those same records from the safetensors shards via
`dee.cpp/tools/phase3/p3_builder.py`'s range-source code and compare bytes
against the segment store. That is O(~4,600 records x 12.75 MiB) instead of
three full inferences, and it isolates the store from the kernel (A0 and A1
also differ in `SOURCE_READ_LANES`, 1 vs 4, so end-to-end equality is
confounded anyway).

Also propose the patch to the driver's `TimeoutExpired` handler so partial
per-prompt evidence is harvested from `/kaggle/working` instead of discarded.

Do not assume my framing is right. If the evidence says A1 actually diverged
or that the untagged files are not A0's, say so directly.
