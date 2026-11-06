# Kaggle profile protocol (exact run the root should perform)

One command (T4 host, campaign checkout):

    python3 dee.cpp/experiments/route_pipeline/run_evidence.py \
        --out evidence/ --command "python3 <canonical_decode.py> ..." \
        --prompt-hash <sha256> --tokens 16 --reps 5

Optional pre-flight (no model execution):

    python3 dee.cpp/experiments/route_pipeline/run_evidence.py --dry-run-live \
        --out evidence/ --command "python3 <canonical_decode.py> ..."

Runner: `run_evidence.py` orchestrates `kaggle_profile_pack.py` (matched
off/on + ingestion) and `kaggle_runner_abc.py` (or `--mock-abc`).
Do NOT launch Kaggle from here; run where the campaign checkout + 2xT4 live.

## Canonical command contract (required)

The command MUST accept `--out <dir> [--profile-stages]` and write
`profile-run/{result.json, host-profile.jsonl, stage-profile.json,
correctness.json}` there (see MOCK_CAMPAIGN_PROTOCOL.md; the mock
implements exactly this). Required producer call sites (profiling-only):

1. `host_layer_records_json()` → `stage-profile.json` under
   `host_layer_records`.
2. `dump_host_profile()` rows → `host-profile.jsonl`.

## Preconditions (refuse otherwise)

1. `nvidia-smi` shows exactly 2× SM75 T4 (checked by the pack).
2. `git rev-parse HEAD` recorded as `source_sha` (no dirty-tree execution;
   record `git status --porcelain` alongside).
3. Canonical exact decode command, fixed prompt (`--prompt-hash` recorded)
   and token count (default 16, matching sealed runs).
4. Correctness gates present and passing (same sealed gates as the campaign).
5. Profiler enabled: `--profile-stages` (+ `DEE_HOST_PROFILE=1` for the
   Python rows).
6. Command scanned against the behavior-change denylist
   (`--transfer-dtype`, `--cache-dtype`, `--budget`, `--prefetch-depth`,
   `--dynamic-quantization`, `--topk`, `--layers`, int4/int8, …) — any hit
   refuses the run.

## Matched pair (perturbation control)

Run profile-OFF then profile-ON with identical command/prompt/tokens.
Compare walls and token counts: systematic deltas estimate profiler
perturbation. Per Phase H, instrumented timings are not acceptance evidence
unless campaign rules explicitly permit; the closure/decisions consume the
profile-ON records with the perturbation note attached.

## Artifacts (per run)

- `sync-profile-result.json` — pack report (GPU, source, gates, both runs,
  perturbation note, closure scaffold, decisions scaffold).
- `per-layer.csv` — C++ `host_layer_records_csv()` output.
- `per-token.json` — token rollups (summed top-level spans per token).
- `closure.json` — `compute_closure()` + `evaluate_decisions()` output.

## Decision consumption (provisional until real data)

`evaluate_decisions()` promotes only on measured fractions with closure ≥
0.85 (defaults: shared ≥0.05, sync ≥0.05, d2h ≥0.02; HASH via runner-case-A
mechanics; C1/C2 held until decode/compute is measured critical). No winner
is selected from unavailable timing — the pack outputs HOLD everywhere when
fields are UNKNOWN.
