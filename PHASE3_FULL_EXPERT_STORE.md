# Phase 3 — Full routed-expert universe store (design + prototype)

Status: **prototyped and locally proven**; full-store construction is a cloud/IO job, not yet executed.
Branch: `research/phase3-full-expert-store` · Base: `ca8abd0`
Model: `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`

> Goal: arbitrary prompt → native authoritative router → **any** routed expert identity →
> storage-backed exact packed record → dee tier hierarchy → exact execution.
> Today's bank cannot do this: it is a `dee4-v3-trace` store sealed to a 16-token route
> journal with **2,364 of 11,008** main-model identities (and none of the MTP draft
> universe). This document specifies the full-universe store, the resumable builder,
> the lazy alternative, and the proof that nothing in the routing universe is missing.

---

## 1. Verified universe — exact and complete

Verified against the committed safetensors headers
(`dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers/model-*.json`,
48 shards, 72,317 tensors, declared 166,878,536,440 B) and cross-checked against
`EXPERT_SEMANTICS.json`, `MODEL_LEDGER.json`, `CHECKPOINT_MANIFEST.json`, and
`scripts/deepseek_v4_*.py`.

### 1.1 Two routed domains — the previous "43×256" claim was incomplete

The checkpoint contains **two** routed-expert namespaces:

| Domain | Names | Modules | Experts/module | Pairs | Routed tensors |
|---|---|---|---|---|---|
| Main model | `layers.{0..42}.ffn.experts.{0..255}.{w1,w2,w3}.{weight,scale}` | 43 | 256 | **11,008** | 66,048 |
| MTP/DSpark draft head | `mtp.{0..2}.ffn.experts.{0..255}.{w1,w2,w3}.{weight,scale}` | 3 | 256 | **768** | 4,608 |
| **Total** | | **46** | | **11,776** | **70,656** |

- Layers 0–2 are hash-routed (`gate.tid2eid`, no bias); layers 3–42 and all three
  `mtp.*` layers use the score router (`gate.weight`/`gate.bias`). **Routing
  mechanism does not change expert storage**: every one of the 46 modules carries
  the identical 256-expert × 6-tensor payload with identical shapes/dtypes.
- The three `mtp.*` modules are the multi-token-prediction / draft (DSpark) head.
  The main forward pass never routes into them, but they *are* routed experts of
  the checkpoint, and any speculative-decode path needs them. The store is
  designed to hold all 46 modules so **no `*.ffn.experts.*` tensor in the
  checkpoint is unowned**; a main-only 43-bucket build remains a valid strict
  sub-scope (all numbers given for both).

### 1.2 Record geometry (identical for all 11,776 records)

DEE4 record order `w1.weight | w3.weight | w2.weight | w1.scale | w3.scale | w2.scale`
(matches `kaggle/deepseek-v4-flash-0731/repack_to_dee4.py` and the sealed bank):

| Component | dtype | shape | bytes | record offset |
|---|---|---|---|---|
| w1.weight (gate) | I8 packed FP4 | [2048, 2048] | 4,194,304 | 0 |
| w3.weight (up)   | I8 packed FP4 | [2048, 2048] | 4,194,304 | 4,194,304 |
| w2.weight (down) | I8 packed FP4 | [4096, 1024] | 4,194,304 | 8,388,608 |
| w1.scale | F8_E8M0 | [2048, 128] | 262,144 | 12,582,912 |
| w3.scale | F8_E8M0 | [2048, 128] | 262,144 | 12,845,056 |
| w2.scale | F8_E8M0 | [4096, 64]  | 262,144 | 13,107,200 |
| **record** | | | **13,369,344 B = 12.75 MiB** | |

### 1.3 Capacity

| Artifact | Records | Bytes | GiB |
|---|---|---|---|
| Main-model bank (43 buckets) | 11,008 | 147,169,738,752 | 137.0625 |
| **Full bank (46 buckets)** | **11,776** | **157,437,394,944** | **146.625** |
| Per-bucket segment (256 records) | 256 | 3,422,552,064 | 3.1875 |
| Sealed v60 bank (reference) | 2,364 | 31,605,129,216 | 29.43 |
| Source checkpoint | — | 166,878,536,440 | 155.42 |

`integrity.jsonl` ≈ 11,776 lines ≈ 5 MiB; `p3_records.jsonl` ≈ 14 MiB; residency
bitmap for the lazy store = 1,472 B.

### 1.4 Bucket → shard binding (proven, not assumed)

Every bucket's 1,536 tensors live **entirely inside one shard**:

- bucket `L` (0..42) → `model-{L+2:05d}-of-00048.safetensors` (shards 2–44)
- bucket `43+N` (mtp.N) → `model-{46+N:05d}-of-00048.safetensors` (shards 46–48)
- shard 1 = `embed.weight`; shard 45 = `norm.weight`, `head.weight`, `hc_head_*`
  (no routed experts)

This single-shard-per-bucket property is what makes per-bucket segments,
per-bucket resume, and remote range-fetch construction clean: a bucket build
needs exactly one shard's ranges, in name order.

---

## 2. The mapping (deterministic, bijective)

`p3_manifest.parse_expert_tensor_name` is THE mapping:

```
layers.<L>.ffn.experts.<E>.<w1|w2|w3>.<weight|scale> -> (bucket L,     E, proj, kind)
mtp.<N>.ffn.experts.<E>.<w1|w2|w3>.<weight|scale>    -> (bucket 43+N,  E, proj, kind)
```

`tensor_name(bucket, expert, proj, kind)` is its exact inverse.
`build_manifest()` proves bijectivity against all 48 committed headers at
construction time:

1. every `.ffn.experts.` name parses (fail-closed on unparsable routed names);
2. all 70,656 names map to distinct (bucket, expert, component) triples;
3. every bucket has exactly experts `0..255` (no id holes);
4. every record's six `data_offsets` ranges are copied verbatim from the headers
   and sum to 13,369,344;
5. `records[i]` has `record_index == i` and pair `(i//256, i%256)` — dense.

`p3_completeness.audit_headers_vs_manifest()` re-proves all of this
independently and reports `routed_tensors_out_of_scope` so a main-only manifest
can never silently drop the mtp tensors (it reports 4,608 out-of-scope).

---

## 3. Store formats

### 3.1 `dee4-v2` single file — recommended production shape, zero reader changes

`experts.dee4` = `total_experts` contiguous 13,369,344-byte records;
`record_index = bucket * 256 + expert`; `metadata.json` `format="dee4-v2"`,
`num_layers=46`, `start_layer=0`, `experts_per_layer=256`, plus the existing
component tables and new `universe_sha256`/`manifest_sha256`/`data_sha256`
bindings (extra keys are ignored by `json_min`'s by-name lookup).

The shipped `dee::Dee4ExpertStore` (`dee.cpp/src/expert_store.cpp:421-536`)
already implements dense `dee4-v2` lookup: `get(layer, expert)` computes
`record_index = layer*epl + expert`, mmaps the file, and returns one
`ExpertView` with `contiguous_data`/offsets — no index file, no trace journal.
**Proof:** `test_p3_dee4_full_geometry.cpp` opens a 157,437,394,944-byte sparse
store and resolves arbitrary pairs across buckets 0–45 through the real reader,
including fail-closed behavior at 46/−1/256 and exact component offsets
(`ALL PASS` under `pytest` on this branch).

Convention for runtime: mtp.N is requested as layer `43+N`. A main-only build
(`num_layers=43`) is identical minus buckets 43–45.

### 3.2 `dee4-v4-segmented` — distribution/queue shape

`segments/experts-bucket-{00..45}.dee4`, one 3.1875 GiB file per bucket +
segment table in `metadata.json`. Same record stride; a segment is the v2 file
restricted to `first_record..first_record+256`. Needed when the bank must be
uploaded/downloaded as a Kaggle-style dataset (per-file limits, per-file
retries, incremental publish) or repaired per bucket. Requires one small
reader extension (segment mmap); not needed for `dee4-v2`.

### 3.3 `dee4-v5-lazy` — demand-paged materialization

Full-size `experts.dee4` allocated sparse (`fsutil sparse`/`truncate`), a
`total_experts`-bit residency bitmap (1,472 B), `integrity.jsonl` +
`lazy.state.json`. First read of a pair → assemble from source ranges → write
at the final offset → fsync → commit integrity line → set bitmap bit. A read
with the bit clear never returns uninitialized bytes. When all 11,776 bits are
set, `finalize()` emits byte-identical dee4-v2 metadata — **lazy and batch are
the same artifact at different fill levels.** This is the only option that
serves arbitrary prompts *while* the bank is still filling.

### 3.4 Recommendation

- **Authoritative production path: batch `dee4-v2` (single file), built in the
  cloud from the pinned checkpoint, shipped as a dataset.**
- Use `dee4-v4-segmented` as the build/transport shape when the store must be
  published incrementally or repaired per bucket; it converts to v2 by plain
  concatenation (same bytes, same offsets).
- Use `dee4-v5-lazy` for dev machines and for bring-up before the full bank
  exists: it is the *only* shape that answers arbitrary prompts without a
  second full copy, at the cost of a synchronous cold-miss fill per first touch
  (~4.7 s/record over residential HF; sub-second over a datacenter link).

---

## 4. Builder: sources, resumability, failure recovery

`tools/phase3/p3_builder.py` — both modes share `assemble_record()` =
fetch six ranges → concat in record order → sha256.

**Sources** (the `RangeSource` protocol, `fetch(shard, data_offset, nbytes)`):

- `LocalShardSource` — local shards; header length cached per file
  (`8 + hlen + data_offset` absolute addressing).
- `RemoteRangeSource` — HF `Range:` requests on the pinned revision; one 8-byte
  probe discovers each header length; per-request retry (6 attempts, backoff);
  requires no local checkpoint copy. Verified end-to-end: real records fetched
  byte-exact and deterministic, scale bytes cross-checked against the committed
  256 MiB shard-2 prefix (`test_real_record_fetch_matches_committed_ranges`).

**Resume/recovery (single):** journal line `{committed_through: N}` per finished
bucket + per-record `integrity.jsonl` (fsync'd after the data fsync). On
restart: truncate `experts.dee4.partial` to `min(size//stride, committed)*stride`,
drop integrity lines beyond the journal, **re-hash the last committed record**
(torn tail can never persist), then continue appending.

**Resume/recovery (segmented):** journal line per finished bucket; a `.partial`
segment resumes at its last whole-record boundary after the same tail
re-verify; segment is published by `os.replace` only when complete and its
sha256 is known. Re-running a finished build is a pure no-op (all buckets
skipped). `repair_record()` rewrites a single record in place (fixed stride
makes it exact) and atomically rewrites its integrity line.

**Integrity identity** — mirrors the sealed bank's conventions:

| Level | Identity |
|---|---|
| component | `component_sha256` in `integrity.jsonl` (6 per record) |
| record | `record_sha256` in `integrity.jsonl` |
| segment | `sha256` in journal + `metadata.json.segments[]` |
| store | `data_sha256` (v2/v5) over the whole data file |
| universe | `universe_sha256` = sha256 of canonical `[[bucket,expert],…]` for all 11,776 pairs (same convention as `selection_sha256`, but covering the full universe instead of a trace subset) |
| manifest | `manifest_sha256` over the canonical manifest JSON; `records_sha256` over the JSONL |

Model/revision binding: `source_repository` + `source_revision` pinned in every
metadata file; verification is fail-closed (`verify_store` re-hashes stored
bytes against the integrity lines; `audit_store_structure` checks sizes, stride,
segment contiguity, and universe binding without hashing).

---

## 5. Disk and throughput — honest numbers

### 5.1 Measured (this box, this branch)

| Measurement | Result | Full-store extrapolation |
|---|---|---|
| Pipeline sans I/O (assemble+sha256+fsync, `--source synthetic`, 8 records) | 168.4 MiB/s, 75.8 ms/record | **~15 min** for 11,776 records — CPU/disk bound only |
| Real HF range build (`--source remote`, 2 records = 25.5 MiB, 13 requests) | 2.91 MiB/s residential | ~14.3 h — usable for lazy fill, not a batch build |
| Prior sealed-bank repack on Kaggle T4 (31,605,129,216 B in 428.0 s) | 70.42 MiB/s | **~35.5 min** for 46 buckets (~33.2 min for 43) |

### 5.2 Platform bounds

| Path | Bound | Time for 146.6 GiB |
|---|---|---|
| Kaggle input dataset read | ~13 MB/s (loop device) | ~3.4 h read-side floor |
| Kaggle working SSD write | ~126 MB/s | ~20.8 min write-side floor |
| Kaggle combined working budget | **19.5 GiB** | cannot stage source+dest; remote-build or segmented-publish only |
| This dev box free disk | ~24 GiB | cannot materialize; sparse/lazy tests only (done) |

### 5.3 No 147 GiB second copy

- **Streaming repack** (recommended): read source ranges → write dest records;
  the checkpoint is only ever *read* (locally or via HTTP ranges), never
  duplicated. Extra disk beyond source+dest = 0 (modulo one `.partial` tail).
- **In-place conversion**: impossible — source tensors are scattered inside
  shared shard files at non-record-stride offsets; the packed layout is a
  different file. Correctly rejected.
- **Hardlinks/symlinks**: cannot synthesize a contiguous 12.75 MiB record from
  six ranges inside other files. Rejected.
- **Reflinks**: filesystem-dependent; NTFS ReFS-block-clone needs matching
  extents — N/A for scattered ranges; Kaggle/ext4 loop devices don't expose
  them either. Rejected.
- **Sparse/lazy (dee4-v5)**: *does* avoid the second copy — the 157 GB logical
  file is allocated with `fsutil sparse setflag` + `fsutil file seteof` (or
  POSIX `truncate`) at ~zero cost and fills only on demand. **Measured:**
  `test_cpp_dee4_reader_covers_full_universe` creates the true-geometry sparse
  store with <1 MB of real data. Caveat discovered on this box: Python
  `truncate()` on an FSCTL-marked file *still allocates all clusters* — use
  `fsutil` on Windows (encoded in `p3_sparse_store._make_sparse`).

---

## 6. Phase-2 seam compatibility (commit `08d3d51`)

The full store plugs into the existing boundary unchanged:

- `ColdExpertStore::read(const TierExpertKey&, uint8_t*, size_t)` —
  `ExpertStoreColdAdapter` already wraps any `ExpertStore`; `Dee4ExpertStore`
  opens the v2 bank directly (`materialize()` path proven in the C++ test).
- `TierExpertKey{model, layer, expert, representation}` — `model` binds to
  `model@9e165c30…`; `representation` = `deepseek-fp4-e2m1-e8m0` (existing
  codec string); `layer` is the bucket index (43–45 = mtp draft layers, a new
  documented convention needing no struct change).
- `StorageRecord{stored_bytes=exact_bytes=13,369,344, codec="identity-v1"}` —
  records are already in executor format; `IdentityCodec` accepts and
  materializes byte-exact; `HostExpertTier` slots (`slot_bytes` ≥ record)
  pin/lease the same mmap'd bytes — zero decode path, zero policy change.
- The bank replaces the trace store behind the same `ExpertStore` interface;
  no tier code changes. Hash layers 0–2 resolve identically (their experts are
  just records; `tid2eid` routing happens above the store).

---

## 7. What is proven vs. what remains

**Proven on this branch** (`pytest dee.cpp/tools/phase3 -q`: 9 passed):

- 70,656 routed tensors ↔ 11,776 pairs, bijective, zero orphans/duplicates
  (main-only scope: 11,008 pairs, 4,608 mtp tensors correctly out-of-scope).
- Real byte fetches: 2 records assembled from live HF ranges are byte-exact,
  deterministic, and match the committed 256 MiB shard prefix.
- Single + segmented + lazy builders build, resume torn tails, verify, and
  repair corrupt records on a synthetic universe.
- The *shipped* C++ `Dee4ExpertStore` resolves arbitrary (bucket, expert)
  lookups over the true 157,437,394,944-byte geometry on a sparse file,
  fail-closed outside 0–45/0–255.
- The v60 bank is a strict subset (2,364 ⊂ 11,008 main pairs).

**Not yet done (needs real capacity/cloud):** an actual 147–157 GB build, a
Kaggle-scale dataset publish, and the `dee4-v4-segmented` reader extension
(small mmap-of-segments lookup) if the segmented shape is adopted for serving.

---

## 8. Recommended implementation sequence (≤10 steps)

1. Land `tools/phase3` manifest + completeness on main; publish
   `p3_manifest.json`/`p3_records.jsonl` (both scopes) as the pinned universe.
2. Batch-build in the cloud: `RemoteRangeSource`/`LocalShardSource` →
   `build_segmented` (per-bucket journals make the ~35 min job resumable and
   each 3.19 GiB segment independently publishable/repairable).
3. Concatenate segments → `experts.dee4`, run `verify_store` (full sha pass),
   emit `metadata.json` (dee4-v2, num_layers=46) + `integrity.jsonl`.
4. Publish the 46 segments (or the single file) as a versioned dataset
   artifact named by `universe_sha256`/`data_sha256`.
5. Runtime: point `Dee4ExpertStore` at the bank; route mtp draft layers as
   buckets 43–45; keep `ExpertStoreColdAdapter`/`HostExpertTier` unchanged.
6. For dev boxes: use `LazyFullStore` (sparse file + bitmap) so arbitrary
   prompts work before the bank lands; `finalize()` converts it to the same
   v2 artifact.
7. (Optional) Add the segmented reader if datasets ship segments directly.
8. Gate arbitrary-prompt tests on the full bank; the sealed 2,364-record bank
   stays only as a sealed-benchmark fixture.

---

## 9. Artifacts on this branch

```
dee.cpp/tools/phase3/
  p3_manifest.py       universe math, name<->(bucket,expert) mapping, manifest builder
  p3_completeness.py   fail-closed audit: headers <-> manifest <-> store structure
  p3_builder.py        RangeSource (local + HF ranges), build_single (dee4-v2),
                       build_segmented (dee4-v4), verify_store, repair_record
  p3_lazy_store.py     dee4-v5-lazy demand-paged store + audit + finalize->v2
  p3_sparse_store.py   fsutil-sparse full-geometry fixture generator
  p3_bench.py          honest throughput measurement + extrapolation
  test_p3_full_store.py        9 tests, all passing (incl. network-marked)
  test_p3_dee4_full_geometry.cpp  C++ proof over the real shipped reader
  conftest.py          pytest markers (network, slow)
```

Reproduce: `python -m pytest dee.cpp/tools/phase3/test_p3_full_store.py -q`
(network test self-skips if huggingface.co is unreachable).
