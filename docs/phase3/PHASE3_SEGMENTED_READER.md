# Phase 3 — dee4-v4-segmented reader (Dee4ExpertStore)

Status: **implemented and locally proven** (W1-T10). This was the one unbuilt
Phase-3 code piece called out in `PHASE3_FULL_EXPERT_STORE.md` §7 ("a
`dee4-v4-segmented` reader extension if the segmented shape is adopted for
serving").
Branch: `feat/dee4-segmented-reader` · Base: `8a3a2cd`
Files: `dee.cpp/include/dee/expert_store.h`, `dee.cpp/src/expert_store.cpp`
(additive), `dee.cpp/tests/test_dee4_segmented.cpp`,
`dee.cpp/CMakeLists.txt` (one-line test registration).

## 1. Format (as written by `p3_builder.build_segmented`)

A segmented store is a directory:

```
<store>/
  metadata.json                 format = "dee4-v4-segmented"
  integrity.jsonl               per-record sha256 (tooling/verify_store only;
                                the reader does not consume it)
  segments/
    experts-bucket-00.dee4      record_count contiguous fixed-stride records
    experts-bucket-01.dee4
    ...
```

`metadata.json` carries the full shared `_store_metadata` schema (codec,
`start_layer`, `num_layers` = bucket count, `experts_per_layer`,
`total_experts`, `record_bytes`, the eight component tables,
`universe_sha256`, `manifest_sha256`, `mtp_bucket_offset`) plus:

```json
"data_file": null,
"integrity_file": "integrity.jsonl",
"segments": [
  {"file": "segments/experts-bucket-00.dee4",
   "bucket": 0, "domain": "main",
   "first_record": 0, "record_count": 256,
   "bytes": 3422552064,
   "sha256": "<sha256 of the segment file's bytes>"},
  ...
]
```

Segment `i` holds records `first_record .. first_record + record_count - 1`
at `(record_index - first_record) * record_bytes` — i.e. one contiguous
slice of the dee4-v2 byte space. Concatenating segments in table order
reproduces the dee4-v2 file byte-for-byte.

## 2. Reader semantics

`Dee4ExpertStore::open(dir_or_metadata[, options])` accepts
`format == "dee4-v4-segmented"` alongside the existing `dee4-v2` and
`dee4-v3-trace`. For a segmented store it:

1. Parses metadata (json_min parses the `null` literal natively — see §5).
2. Validates the shared schema exactly as v2/v3 (codec, geometry, component
   tables, shape×byte consistency).
3. Validates the segment table **strictly, matching the writer**:
   - `total_experts` required and must equal `num_layers × experts_per_layer`;
   - `segments` must be an array of exactly `num_layers` entries;
   - entry `i` must declare `bucket == i`,
     `first_record == i × experts_per_layer` (equivalently: ranges tile
     `[0, total_experts)` contiguously from 0 — same invariant
     `p3_completeness.audit_store_structure` checks),
     `record_count == experts_per_layer`,
     `bytes == record_count × record_bytes`,
     `sha256` a well-formed 64-hex string;
   - `file` must be a non-empty store-relative path (absolute paths and
     `..` escapes are rejected);
   - `domain` is informational only (validated as a string if present);
   - `data_file` must be absent, JSON `null`, or an empty string — a
     segmented store naming a monolithic file is contradictory and
     rejected.
4. mmaps every segment file (`MapViewOfFile`/`mmap` per segment, read-only)
   and requires `actual size == declared bytes` — fail-closed on missing,
   unreadable, or mis-sized segments.
5. **Seal verification (default on):** re-hashes each segment's mapped bytes
   and compares to the declared `sha256`; a mismatch fails open. Disable via
   `Dee4OpenOptions{verify_segment_hashes = false}` for trusted local
   mirrors — the pass is O(total store bytes) once at open (~157 GiB for the
   full universe), so the escape hatch exists for the giant store; the
   structural checks (existence, exact size, table validity) always apply.
6. Store identity: `integrity_identity()` returns a derived
   **segment-table digest** — sha256 over the ordered concatenation of the
   per-segment `sha256` hex strings. (The segmented format has no whole-file
   `data_sha256`; the table seals are its content binding. Rationale:
   deterministic, always available, changes iff any declared seal changes.)

Lookup (`get(layer, expert)`), `get_layout_reference`, and `materialize()`
are **identical in semantics to the monolithic reader**: dense
`record_index = (layer - start_layer) × experts_per_layer + expert`, then a
binary search of the segment table by `first_record` (correct under the
contiguity invariant; the index is not simply assumed). The returned
`ExpertView` exposes `contiguous_data` pointing into the owning segment's
mapping with the same component offsets/dtypes. `materialize()` re-derives
the owning segment from `view.record_index`, enforces the same
`contiguous_data == segment_base + offset` identity check, and on POSIX
`pread`s from the segment's fd (Windows: `memcpy` from the mapping), so
independent records — including records in *different* segments — can be
materialized concurrently.

Telemetry: `backend_name()`/`stats().backend` report `"dee4_segmented"`;
`trace_indexed()` is false; `segmented()` is true; `segment_count()`,
`stored_records()`, `start_layer()`, `num_layers()`,
`experts_per_layer()`, `record_bytes()` report the store geometry. The mtp
draft layers are reached as layer indices 43–45 when `num_layers == 46`,
exactly as in the v2 reader.

## 3. Failure modes (all fail-closed)

| Condition | Where | Result |
|---|---|---|
| `metadata.json` missing / unparseable | open | `last_error_` + false |
| Unknown `format` / wrong codec / bad component tables | open | false |
| `segments` absent, wrong count, gap/overlap in `first_record`, `record_count != experts_per_layer`, `bucket` out of order, `bytes` inconsistent | open | false |
| `data_file` non-null in segmented metadata | open | false |
| malformed `sha256`/`universe_sha256`/`manifest_sha256` | open | false |
| segment file missing/unreadable | open (map) | false |
| segment size ≠ `record_count × record_bytes` | open | false |
| segment bytes ≠ declared `sha256` | open (verify, default) | false |
| `layer`/`expert` outside `start_layer..+num_layers` / `0..epl` | get | false, counted |
| `record_index` unresolvable through the table | get/materialize | false |
| forged `ExpertView` (index/pointer mismatch) | materialize | false |

## 4. Test evidence (`tests/test_dee4_segmented.cpp`)

- sha256 known-answer vectors (empty / "abc" / 10⁶×'a').
- Small synthetic store (4 buckets × 4 experts × 70 B): every pair resolves
  with byte-exact content — including the last/first records straddling
  segment boundaries; component views and offsets match; `materialize()`
  round-trips the writer's bytes; forged views rejected; out-of-range
  `(4,0)/(-1,0)/(0,4)/(0,-1)` fail closed.
- Missing segment, mis-sized segment, mis-hashed segment (one flipped byte):
  open fails — mis-hash only when seal verification is on; the
  `verify_segment_hashes=false` escape hatch opens it (documented trade-off).
- Malformed tables rejected: first_record gap, short table, `record_count`
  mismatch, non-hex seal, non-null `data_file`, out-of-order `bucket`,
  `total_experts` inconsistent with geometry.
- True-geometry sparse fixture: 46 sparse segments × 3,422,552,064 B
  (~146.6 GiB logical, <1 MB real), real component tables; sampled
  (bucket, expert) across all three domains incl. mtp buckets 43–45
  resolve; seeded marker bytes read back; `materialize()` correct;
  `(46,0)/(-1,0)/(0,256)/(0,-1)` fail closed.
- Separate end-to-end interop check (scratch, not committed): a store built
  by the real `p3_builder.build_segmented` (mini universe) opens under
  default seal verification and all 12 records are byte-exact.

## 5. Schema ambiguities resolved (documented per task)

1. **`"data_file": null` vs json_min — RESOLVED on the integration
   line.** `json_min` originally had no `null` literal, so a conforming
   segmented metadata.json was *unparseable*; this branch carried a
   retry-only `scrub_json_nulls()` (value-position `null` → `""`) as the
   workaround. The `fix/json-min-null` merge (T17) made `null` parse
   natively to a Null-typed `Value` with `is_null()`, so the scrub and its
   retry were deleted as dead code. The segmented `data_file` gate was
   updated to accept `is_null()` (plus absent key / empty string for
   hand-edited metadata) — without that arm, native null parsing would
   have made `is_string()` false and `open()` would reject every
   conforming segmented store. v2/v3 still require a non-empty
   `data_file`. Regression check: `test_segment_table_validation` opens
   metadata carrying the literal `"data_file": null`.
2. **Store identity.** Segmented metadata has no `data_sha256`; identity is
   derived as the segment-table digest (sha256 of the concatenated declared
   seals). If a canonical whole-store seal is ever emitted, switching
   `identity_` to it is a one-line change.
3. **Table strictness.** The writer emits exactly one segment per bucket
   (`record_count == experts_per_layer`, `first_record == bucket × epl`,
   `bucket ==` ordinal). The reader requires that contract rather than
   general arbitrary tiling — simpler fail-closed surface; the lookup itself
   remains a general range search. A future writer emitting non-bucket
   tiling would need this relaxed (validation only; lookup already copes).
4. **Seal verification cost.** `mis-hashed → fail closed` is implemented as
   a real sha256 pass over segment bytes at open — the only sound meaning of
   the requirement. On the full universe this is ~157 GiB of hashing, so it
   is opt-out (`verify_segment_hashes=false`); structural checks are never
   optional. This mirrors the repo's split between structural audit
   (`audit_store_structure`, no hashing) and content verification
   (`verify_store`, full hashing) — the reader defaults to the strict side
   because a published segment's sha256 IS its transport seal.

## 6. Merge notes

- `dee.cpp/CMakeLists.txt` was edited on this branch (one-line
  `tests/test_dee4_segmented.cpp` registration in `DEE_TEST_SOURCES`); the
  branch's ancestry differs from the Luna-line swarm — this file is a
  future merge point.
- `dee.cpp/src/host_pack_cache.cpp` is broken at the base commit `8a3a2cd`
  (duplicate `additional_bytes`/`unique_misses` declarations + stale
  `phase_reserved_`; the fix `d68cf4ad` exists on
  `research/phase2-integration-lru-fix` and is cherry-picked onto
  `integration/phase3-store` as `78265b5`). It blocked the `dee_core`
  target and therefore the full `ctest` suite on this branch. On the
  integration line it is fixed; on this branch standalone, tests were
  verified by direct compilation of the
  store sources + test — the same path `test_p3_full_store.py`'s C++ proof
  uses — and by the phase3 pytest suite (9/9 PASS, which recompiles the
  modified `expert_store.cpp` and exercises the unchanged v2 path).
