# Mock campaign protocol

`dee.cpp/experiments/route_pipeline/mock_campaign.py` implements the
canonical command contract with deterministic fixture data (no CUDA, no
checkpoint). It exists so the ingestion/closure/ranking/artifact code that a
real T4 run will execute is itself executed and tested locally.

## Command contract (canonical commands MUST accept these)

```
<command> --out <dir> [--profile-stages] [--extra ...]
```

and write `profile-run/{result.json, host-profile.jsonl, stage-profile.json,
correctness.json}` under `--out`. Mode (off/on) is derived from
`DEE_HOST_PROFILE` env (1 → on), which the pack already sets per run.

## Canonical profile-run/ layout

- `result.json`: `{status, metrics:{decode_wall_s (SECONDS), tokens},
  generated_ids, decoded_text, source_sha, prompt_hash,
  config{... (profile_stages + DEE_HOST_PROFILE differ by mode)},
  hardware}`. Walls come from here ONLY.
- `host-profile.jsonl`: Python `dump_host_profile()` rows, one JSON/line
  (may include a duplicate `native_call_wall_ms`: dropped on merge with a
  note when C++ spans exist for the key).
- `stage-profile.json`: `{stage_profile: {...},
  host_layer_records: {records: [...]}}` with C++ `host_span_name()` keys
  (e.g. `native_call_wall`, `source_lookup_wait`, `native_output_sync`).
- `correctness.json`: `{classification, gates: {name: bool}}` (all must pass).

## Required campaign call sites (profiling-only, one line each)

1. After the profiled decode: write `host_layer_records_json()` into
   `stage-profile.json` under `host_layer_records`.
2. After the profiled decode with `DEE_HOST_PROFILE=1`: write
   `dump_host_profile()` rows into `host-profile.jsonl`.

## Scenarios (fixed values, geometry 2 tokens × 4 layers)

| # | Shape | Expected ranking |
|---|---|---|
| 1 | sync 6.0 ms/row, rest quiet | EVENT_HANDOFF PROMOTE |
| 2 | shared 3.0 ms/row | SHARED_OVERLAP PROMOTE |
| 3 | route-D2H 2.0 ms/row | ROUTE_D2H_NARROW PROMOTE |
| 4 | all quiet | NO_OVERLAP_OPTIMIZATION_JUSTIFIED |
| 5 | wall ×3.0 (`--wall-scale 3.0` both modes) | PROFILE_INCOMPLETE, all HOLD |
| 6 | on-run IDs differ | INVALID_PROFILE_PAIR |
| 7 | on-run jsonl corrupt | INVALID_PROFILE_EVIDENCE |

Mock ABC (`--emit-abc`): A hidden ms 5.0 for scenarios 1/3 else 0.2;
B `efficiency_c` 0.83; C sync/event p50s. Mock values exercise logic only
and are never T4 measurements; the mock `decode_wall_s` is derived from the
same span tables so closure is 1.0 except scenario 5.

## Mock-vs-live boundary

Identical: CLI contract, file layout, pair validation, merge precedence,
closure math, ranking, next-A/B derivation, manifest/resume, dry-run
checks. Different ONLY: span values (fixed vs measured), GPU presence,
checkpoint reads. A test that passes on mock and would fail live can only
differ in measured values, never in code path — enforced by sharing
`run_evidence.py` → `kaggle_profile_pack.py` → `evidence.py` end to end.
