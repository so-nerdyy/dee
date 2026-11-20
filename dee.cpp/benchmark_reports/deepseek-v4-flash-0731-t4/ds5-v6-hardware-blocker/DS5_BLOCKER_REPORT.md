# DS5 v6 — Hardware-Format Blocker Report

**Status:** BLOCKED (hardware-format)
**Kernel:** `nivind/dee-cpp-deepseek-v4-flash-0731-ds5-trace` v6
**Terminal state:** ERROR
**First failing gate:** `reference_loaded` (runtime kernel launch, not a load failure)
**Verdict recorded:** `INVALID_EXPERIMENT` with error
`InternalError: Failed to set the allowed dynamic shared memory size to 82048`

---

## 1. What the run proved (progress vs v1–v5)

| Gate | v3 | v4 | v5 | v6 |
|------|----|----|----|----|
| source_identity | PASS (after ASCII fix) | PASS | PASS | PASS |
| probe_ok | — | PASS | PASS | PASS |
| config_identity | — | PASS | PASS | PASS |
| tokenizer_identity | — | PASS | PASS | PASS |
| shard_identity (3 shards, size+header+count+coverage) | — | PASS (after size fix) | PASS | PASS |
| convert.py (fp4, mp=1) | — | PASS | PASS | PASS |
| reference load (Transformer n_layers=1, dtype contract) | — | — | FAIL (tvm_ffi) | PASS |
| forward trace (fp8_gemm launch) | — | — | — | **BLOCKED** |

Fixes landed on the way:

- **v3 → v4** (`8fa511a`): shard size pins now include the safetensors header
  overhead (`8 + header_size + max(data_offsets[1])`). The old pins truncated
  downloads by 96/172240/400 bytes and convert failed with "incomplete metadata".
- **v5 → v6** (`27149e9`): pinned `apache-tvm-ffi<=0.1.9` (installed before
  `tilelang==0.1.8`). tilelang 0.1.8 is incompatible with apache-tvm-ffi
  >= 0.1.10 on Python 3.12 (tvm_ffi registry `setattr(type_cls, '__dict__', …)`
  crash). Full traceback captured in `kernel.log`.
- v5 added full-traceback capture to evidence (`error_traceback`), which is what
  localized this blocker precisely.

## 2. The blocker

The official inference stack (`inference/kernel.py`, tilelang 0.1.8) requests
more dynamic shared memory per block than a Tesla T4 (SM75) can provide.

- SM75 opt-in max dynamic shared per block: **65536 bytes (64 KiB)**.
- `fp8_gemm` kernel (used by every dense linear): **82048 bytes requested**
  at runtime (`num_stages=4` pipelines 20 KiB of A/B tiles, plus C and scale
  buffers). 82048 > 65536 → kernel launch fails.

The failure surfaced on the very first dense projection of layer 0
(`self.wq_a(x)` → `fp8_gemm`), i.e. before any attention, routing, or expert
work. Full stack in `kernel.log`:

```
model.py:158 linear -> kernel.py:124 fp8_gemm ->
tilelang/jit/adapter/tvm_ffi.py:244 -> executable(...) ->
tvm.error.InternalError: Failed to set the allowed dynamic shared memory
size to 82048
```

### 2.1 Source-derived shared-memory audit of all official kernels

Computed from the pinned `inference/kernel.py` allocations (BF16=2 B, FP8=1 B,
FP32=4 B), with pipelined buffers multiplied by `num_stages`:

| kernel | buffers | bytes | vs 64 KiB |
|--------|---------|-------|-----------|
| fp8_gemm (block 32x128x128, stages=4) | A/B pipelined x4 + C + scale | 90240 | EXCEEDS |
| sparse_attn attention (h=64, d=512, block=64, stages=2) | q + kvx2 + o + accx2 + fragments | ~280064 | EXCEEDS |
| sparse_attn indexer (h=64, d=128) | q + kvx2 + o + accx2 | ~73728 | EXCEEDS |

**Conclusion:** the official kernels are not SM75-runnable. Even if `fp8_gemm`
were re-tiled to fit (e.g. `num_stages=2` -> ~49 KiB), `sparse_attn` with
h=64, d=512 alone needs `q_shared (64x512x2=64 KiB)` + `kv_shared` +
`o_shared` + `acc_s_cast`, which cannot fit in 64 KiB under any pipelining
depth. The official attention path cannot execute on T4 without rewriting the
kernel tiling, which would no longer be the pinned official implementation.

## 3. Why this is a blocker (not a harness bug)

- Every harness gate before the kernel launch passed (identity, config,
  tokenizer, all 3 shard bytes/headers/counts/coverage, convert, dtype
  contract).
- The failure is inside the **official tilelang kernel launch** — a physical
  shared-memory ceiling, not a download, parse, or orchestration defect.
- DS5's contract (per campaign spec) requires the reference to be the pinned
  official implementation; substituting a re-tiled kernel makes the reference
  non-official and would break the DS5/DS9 parity contract.
- The DS5 spec also states the reference "does not need to run on T4" — the
  intent is a trusted reference on capable hardware. The campaign's only
  remote GPU (Kaggle dual T4, SM75) cannot host it.

## 4. Evidence preserved

- `evidence/` — ds5-trace-evidence.json (error + error_traceback), verdict,
  manifest, harness identity, module copies.
- `kernel.log` — full kernel log incl. complete Python traceback.

## 5. Options to unblock DS5 (for product decision)

1. **Non-T4 reference host** (preferred): run the official stack on an
   SM80+ GPU (e.g. A100/H100, ≥164 KiB dynamic shared) to produce the pinned
   reference traces. No such host is currently attached to this campaign.
2. **Documented reference-kernel fork**: re-tile the official kernels for
   SM75 and validate equivalence on the shared-memory-compatible path. This
   changes the reference definition and must be an explicit product decision.
3. **Skip DS5 official-trace generation**: use the already-sealed DS8/DS9
   trusted-reference discipline (official tensors + FP32 CPU reference) as the
   DS6+ parity basis, with DS5 marked blocked and this report as the record.

No TPS, no speedup claims are associated with this milestone. Performance
remains non-comparable.
