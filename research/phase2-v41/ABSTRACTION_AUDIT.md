# Phase-2 tier seams — DeepSeek-V4.1 geometry conformance audit

- Task: W1-T7 (Phase-4 groundwork)
- Branch: `research/phase2-v41-check` @ `56dad3c1` (Luna's `phase2-integration-lru-fix` head)
- Audit target: `dee.cpp/` Phase-2 tier seam — `ColdExpertStore -> HostExpertTier -> DeviceExpertTier -> StorageCodec` plus the `engine.cpp` phase2 wiring block and `pydee`.
- Sanity-check geometry (V4.1): 40 MoE layers, 384 routed experts/layer, top-6,
  hidden 5120, expert intermediate 2304 → record = 3·(2304·5120)/2 packed I8 +
  3·(out·in/32) e8m0 scales = **18,800,640 B (17.93 MiB)/expert**, routed pool
  40·384·18,800,640 = 288,777,830,400 B ≈ **269 GiB**.

## Verdict

**The tier seam is V4.1-clean for model geometry.** No DSv4 size/count literal
(43 layers, 256 experts, 13,369,344 B, 4096 hidden, 2048 inter) is embedded in
`host_expert_tier.*`, `expert_tiers.*`, `async_prefetcher.*`, `vram_cache.*`, or
`expert_store.*`. Every record size, layer/expert index, slot/arena byte count,
and scope string reaches the tiers parametrically through `HostTierConfig`,
`StorageRecord`, `TierExpertKey`, and the `ExpertView` layout.

What *is* bound is the **representation family**, intentionally: the FP4-e2m1
weights + per-block e8m0 scales, three-projection (gate/up/down) record layout.
V4.1 uses the same family, so a V4.1 deployment needs **zero tier-code
changes** — only a V4.1 `ExpertStore` (a DEE4 bank with V4.1 metadata, or a
V4.1 tensor resolver) and `EngineConfig` values.

The one hard coupling: `Engine::init` arms the Phase-2 host tier only under
`WeightTransferDType::Fp4E2m1` + `DeviceCacheDType::Fp4E2m1` + `--cuda` +
non-empty `phase2.model_identity` (engine.cpp:3435-3440). That binds Phase-2 to
the FP4 representation *family*, not to DSv4 geometry. A second family keeping
the format is clean; a family with a different executor format needs a new
`StorageCodec`/dtype extension (by design — codecs are the representation seam).

## Conformance table

`P` = parametric (derived from store/layout/config at runtime) — no change for
V4.1. `R` = representation-family binding (not geometry) — reusable for V4.1 as
-is; only a *different format* needs change. `X` = out-of-seam context constant.

| Site | Constant | P/R/X | Required change for V4.1 |
|---|---|---|---|
| `include/dee/host_expert_tier.h:16` `max_identity_bytes = 1024` | key bound | P | none (model+`\nstore:`+sha fits) |
| `include/dee/host_expert_tier.h:100` `alignment = 4096` | alloc alignment | P | none (default; 18,800,640 is 4K-aligned exactly) |
| `src/host_expert_tier.cpp` (all) | — | P | none; sizes from `StorageRecord.exact_bytes` vs `slot_bytes`/`budget_bytes` |
| `src/expert_tiers.cpp:33-39` adapter `bytes_` | sum of 6 `ExpertView` tensor nbytes | P | none |
| `src/expert_tiers.cpp:56-79` gather | `weights[3]` + `scales[3]` arrays | R | none for V4.1 (same 3-projection format); a non-3-projection family needs `ExpertView` generalization |
| `src/expert_tiers.cpp:82` `record_source_read(…, 6, …)` | region count 6 | R | none (derived from the 3+3 layout) |
| `src/expert_tiers.cpp:90-98,101-103` device scope check | model+representation only | P | none; layer/expert fields of scope are placeholders |
| `src/async_prefetcher.cpp:21-24,56-61` `key_id`/`map_key` | `layer<<32 \| expert` | P | none (V4.1 max layer 42, expert 383 ≪ 2³²). Legacy LLP64 `long` truncation retained only in non-tier mode (v2a Medium, deliberate); tier path keeps all 64 bits |
| `src/async_prefetcher.cpp:684-693` `prefetch_host_lease` | scope check | P | none |
| `include/dee/async_prefetcher.h:45-57` `Transfer.fp4_*[3]`/`[6]`, `quant_scales[3]` | 3 projections × weight+scale | R | none for V4.1 |
| `include/dee/vram_cache.h:36` `ExpertKeyHash` | `layer<<32 ^ expert` | P | none |
| `include/dee/vram_cache.h:210` `PRIORITY_WEIGHT = 1<<20` | policy weight | P | none (policy, not geometry; Phase-2 repair path already switches scoring) |
| `src/expert_store.cpp:340-505` `Dee4ExpertStore::open` | `dee4-v2`/`dee4-v3-trace` formats; geometry from `metadata.json` | P | none — V4.1 bank carries `num_layers=40(+M)`, `experts_per_layer=384`, `record_bytes=18800640`, component tables for [2304,2560]/[2304,2560]/[5120,1152] I8 + [2304,160]/[2304,160]/[5120,72] F8 |
| `src/expert_store.cpp:369` `codec != "deepseek-fp4-e2m1-e8m0"` | DEE4 metadata codec name | R | none if V4.1 bank reuses the format name; a renamed codec string needs an accepted-alias list here |
| `include/dee/expert_store.h:23-26` `ExpertCodec::DeepSeekFp4E2m1E8m0` | codec enum name | R | none for same format; add an enumerator for a different format |
| `include/dee/expert_store.h:32-33` `ExpertView` | fixed 3+3 tensors | R | none for V4.1 |
| `src/expert_store.cpp:229` `record_index` (safetensors) | `layer<<32 \| expert` | P | none |
| `src/expert_store.cpp:531-534` `record_index` (dee4-v2) | `layer*epl + expert` | P | none (metadata-driven `epl`) |
| `src/expert_store.cpp:610` `kPage = 4096` | OS page | P | none |
| `src/engine.cpp:3684-3693` representation string | `"fp4-e2m1-e8m0-gate-up-down-v1"` + `:<out>x<in>:<scale_off>` ×3 | P suffix / R prefix | none — suffix is built from `layout.fp4[p]` values (`configure_fp4_quantized`, tensor shapes), so V4.1 yields `…:2304x5120:17694720:2304x5120:18063360:5120x2304:18432000` automatically; family tag is the versioned format name |
| `src/engine.cpp:3696-3697` `phase2.model_identity` | caller-supplied | P | none — pass the V4.1 `repo@revision` string |
| `src/engine.cpp:3699` `host_config.slot_bytes` | defaults to `cache_blob_bytes_` | P | none — `blob_elems_ = 3·inter·hidden` → `packed_fp4_cache_blob_bytes = elems·17/32` = 18,800,640 for V4.1 |
| `src/engine.cpp:3711-3712` device scope | `record(0,0).key` | P | none (accepts() ignores scope layer/expert) |
| `src/engine.cpp:3435-3440` phase2_host gate | `Fp4E2m1` transfer+cache + cuda + identity | R | none for V4.1; a non-FP4 family needs a dtype/codec extension |
| `src/engine.cpp:3517` `deepseek_v4` flag | `transfer_dtype == Fp4E2m1` | R | none for V4.1; gates `expert_store_` creation + shard-derived shape discovery |
| `src/engine.cpp:3415-3416,3472` `inter=256`, `num_experts=256` | Ornith-era fallbacks | X | none — overwritten from shard (`inter_` at :3568) / oracle; never reach the tier seam |
| `src/engine.cpp:3592,3601,3612,2144` `256`/`256+256` | Oracle predictor arch (D2048→H256→E256) | X | none — Ornith `OracleScheduler`, unrelated to tiers |
| `include/dee/engine.h:109-110` `hidden=2048`, `inter=256` | config defaults | X | caller must pass V4.1 values anyway; `hidden` is cross-checked vs shard (:3575) |
| `include/dee/engine.h:578` `kPinnedStagingLimit = 192 MiB` | staging bound | X | none (byte bound, not geometry) |
| `include/dee/weight_mmap.h:116` `TensorResolver::Model{ORNITH,DEEPSEEK_V4}` | tensor-name dialect | R | V4.1 safetensors path needs a dialect entry **or** reuse if V4.1 names match `v4_expert_tensor_name`; DEE4-store path bypasses the resolver entirely |
| `src/engine.cpp:1361,1638`; `src/rmsnorm_cuda.cu:111,254,303` `dim > 4096` | Qwen RMSNorm helper bound | X | none — Qwen diagnostic APIs, not on the DS/V4.1 tier path (V4.1 hidden 5120 would exceed *if* reused there — noted only) |
| `include/dee/host_pack_cache.h:41-42` `kMaxFillLanes=8`, `kMaxBatchRequests=256` | queue caps | X | none (bounded-queue constants on the legacy path; the `256` is a batch cap, not expert count) |
| `src/profiling.cpp:20` `physical_key` | `layer<<32 \| expert` | P | none |
| `pydee/*` | no `phase2` surface at all | X | Phase-2 is `EngineConfig`-only today; exposing it to Python is optional bring-up work, not a conformance defect |
| `include/dee.h` | no phase2 exposure | X | same |

Sites swept and found to contain **no** target literals (43 / 256 / 13369344 /
4096 / 2048 in geometry roles): `host_expert_tier.*`, `expert_tiers.*`,
`vram_cache.*`, `async_prefetcher.*`, `expert_store.*` (geometry all from
metadata), `pydee/*` (only Ornith demo defaults), `dee.h`, `main.cpp`.

`13369344` appears only in test fixtures, experiment scripts, and profile JSON
(`tests/test_deepseek_v4_support.py`, `experiments/route_pipeline/*`) — never in
the seam.

## Adapter-boundary spec — what a second model family must supply

1. **`ExpertStore` implementation** (`get(layer, expert) -> ExpertView`):
   - `weights[0..2]` = gate(w1), up(w3), down(w2); `scales[0..2]` same order.
   - Each `TensorView` = `{data, nbytes, dtype, shape}`; `ok()` needs data+nbytes.
   - `codec = ExpertCodec::DeepSeekFp4E2m1E8m0` for the FP4-e2m1+e8m0 family
     (name is DS-branded; semantics are the format), or a new enumerator.
   - `contiguous_data`/`contiguous_nbytes` optional: set → single-region
     `materialize()`; unset → adapter gathers the six regions directly into the
     final host slot.
   - `integrity_identity()` — stable store-integrity string.
   - **Owns range admission**: out-of-universe `(layer, expert)` must fail
     closed (tiers treat keys as opaque).
2. **Record gather order** (fixed convention, `ExpertStoreColdAdapter::read`):
   `[gate_w][up_w][down_w][gate_s][up_s][down_s]` — matches DEE4 record order
   `w1.weight|w3.weight|w2.weight|w1.scale|w3.scale|w2.scale` and the
   prefetcher's `fp4_region_src[0..5]` ordering.
3. **Codec identity**: `StorageRecord.codec = "identity-v1"` +
   `IdentityCodec` whenever stored bytes *are* executor bytes
   (`exact_bytes == stored_bytes`). A family needing transcode registers a new
   `StorageCodec` + its own codec string; `stored_bytes`/`exact_bytes` may then
   differ.
4. **`TierExpertKey.model`**: caller convention `repo@revision`
   (checkpoint-integrity binding); `ExpertStoreColdAdapter` appends
   `"\nstore:" + store.integrity_identity()`. Total ≤ 1024 B.
5. **`TierExpertKey.representation`**: versioned format tag + parametric
   layout suffix. Engine convention:
   `fp4-e2m1-e8m0-gate-up-down-v1:<out>x<in>:<scale_off>` per projection.
   Same-format families get distinct strings automatically via dims; a
   different format must mint a new tag — never overload a tag across formats.
   - *Doc drift to note*: PHASE3_FULL_EXPERT_STORE.md §6 writes
     `representation = "deepseek-fp4-e2m1-e8m0"` — that is the **DEE4 metadata
     `codec` field**, not the tier representation string. Distinct fields;
     adapters must not conflate them.
6. **Layer convention** (`43+N` mtp buckets): `layer` = dense bucket index —
   main MoE layers `0..L-1`, draft/mtp modules appended as `L+N` (DSv4:
   mtp.{0,1,2} = 43,44,45; V4.1: 40,41,42). The tiers never interpret the
   index; the convention lives entirely in store metadata
   (`num_layers = L+M`) and router mapping above the store.
7. **Index packing**: map keys `(uint32 layer)<<32 | (uint32 expert)`;
   dee4-v2 `record_index = layer*experts_per_layer + expert`. Both parametric.
8. **Sizes**: `slot_bytes ≥ record_bytes`, `budget_bytes ≥ slots·stride`,
   device arena ≥ record. All caller-supplied; engine defaults derive from the
   resolved layout.

## Conformance test + build evidence

New source: `dee.cpp/tests/test_phase2_v41_geometry.cpp` — constructs the real
`ExpertStoreColdAdapter` + `HostExpertTier` + `DeviceExpertTier` +
`IdentityCodec` at V4.1 geometry over synthetic stores (six-region gather and
contiguous paths), 55 checks: byte-exact 18,800,640 B records, LRU eviction at
17.93 MiB stride, scope/model/representation isolation, draft-bucket layers
(40..42 = L+N), full 40×384+3 universe identity sweep, fail-closed surfaces.
**Result: 55/55 PASS, ~2.7 s** (run log additionally shows the expected
`evict_until_free` forensic line from deliberate arena pressure).

Registration (CMakeLists is T1-owned — snippet only):
```cmake
# in DEE_TEST_SOURCES, after tests/test_phase2_host_tier.cpp:
tests/test_phase2_v41_geometry.cpp
# optional, beside the existing TIMEOUT block:
set_tests_properties(test_phase2_v41_geometry PROPERTIES TIMEOUT 60)
```

Standalone build (verified on this box):
```
g++ -std=c++17 -O2 -Wall -Wextra -I include \
    tests/test_phase2_v41_geometry.cpp \
    src/host_expert_tier.cpp src/expert_tiers.cpp src/vram_cache.cpp \
    src/async_prefetcher.cpp src/expert_store.cpp src/json_min.cpp \
    src/weight_mmap.cpp src/profiling.cpp -pthread \
    -o test_phase2_v41_geometry
```
**Environment note**: this sandbox's MinGW `g++`/`gcc` 15.2 cannot spawn
`cc1plus`/`cc1` (silent exit 1; direct launch → 127), so the verified build used
`clang++` 22.1.8 (`x86_64-w64-windows-gnu`, same MinGW target, same flags).
The command above is the canonical g++ form for environments where the GCC
driver works; the source compiles clean under `-Wall -Wextra` (the only
warnings were pre-existing, in `vram_cache.cpp`/`async_prefetcher.h`).

## Carryover (previously documented, not V4.1-blocking)

- LLP64 `map_key` truncation in the legacy (non-tier) prefetch path — retained
  deliberately for default-OFF equivalence; bypassed under
  `experimental_host_tier_` (async_prefetcher.cpp:56-61).
- `DeviceExpertTier` scope exclusivity is construction-time only (v2a Medium):
  cross-model byte-serving is possible only via shared drained
  cache/prefetcher (API misuse, unreachable via `Engine`).
