"""Kaggle dual-T4: real tokenizer->text generation with the native FP4 FFN.

Routed-expert FFN runs through pydee.Engine.moe_forward_experts (mmap packed
FP4 -> on-GPU dequant -> cuBLAS SwiGLU); tokenizer/attention/KV/router/shared
expert/norm/LM head stay on the sealed DS10 torch path.

Progress + errors are written to /kaggle/working (captured in the output
tarball even when the run fails) so failures are diagnosable without the live
console log.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
COMMIT = os.environ.get("NATIVE_COMMIT", "")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
N_SHARDS = 48
# /kaggle/temp is NOT present on the dual-T4 "medium" container (verified via
# a disk probe on 2026-08-15); /tmp and / both sit on the ~8 TB root overlay
# with ~1 TiB free, so stage the 167 GB checkpoint there instead.
# NATIVE_SOURCE_TREE lets an outer driver hand in a pre-cloned/pre-built
# tree (GPU-2 session driver): the self-clone of BRANCH is skipped and the
# existing build-kaggle/pydee artifacts are reused incrementally.
ROOT = Path(os.environ.get("NATIVE_SOURCE_TREE", "/tmp/dsv4-native-src"))
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build-kaggle"
CKPT = Path("/tmp/dsv4-checkpoint")
# When the checkpoint is published as a Kaggle dataset it mounts read-only at
# /kaggle/input/<slug>/; prefer that (no download, no disk quota). The local
# /tmp fallback remains for the 155 GiB-free case.
# GPU sessions have mounted datasets flat; CPU kernels moved to the nested
# /kaggle/input/datasets/<owner>/<slug>/ layout (2026-09-12).  Probe both.
DATASET_DIR = next(
    (p for p in (
        Path("/kaggle/input/deepseek-v4-flash-0731-shards"),
        Path("/kaggle/input/datasets/nivind/deepseek-v4-flash-0731-shards"))
     if p.is_dir()),
    Path("/kaggle/input/deepseek-v4-flash-0731-shards"))
WORK = Path("/kaggle/working")
HEADERS_DIR = (DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers")
CONFIG = (DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/"
          "official-source/inference/config.json")
SEALED_PROMPT = (
    "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>Who is Alan Turing?"
    "<\uFF5CAssistant\uFF5C>")
SEALED_TOKEN_IDS = [
    666, 95140, 96807, 343, 4470, 20, 1127, 3298,
    22, 22604, 515, 411, 3947, 85349, 14, 6341,
]
SEALED_DECODED_TEXT = (
    "**Alan Turing (1912\u20131954)** was an English mathematician, computer")
# Fail closed in seconds (not after a 3-hour run) if the push pipeline
# re-transcodes this file: v47 executed a cp1252-mojibake copy of the en dash
# (E2 80 93 -> U+00E2 U+20AC U+201C), flipping the sealed-text gate to a false
# REJECT_NUMERICAL while all 16 token IDs were exact. ASCII-only source is
# immune; this guard fires at import if anything still mangles the value.
assert SEALED_DECODED_TEXT.encode("utf-8") == (
    b"**Alan Turing (1912\xe2\x80\x931954)**"
    b" was an English mathematician, computer"), (
    "SEALED_DECODED_TEXT corrupted in transit; refusing to judge exactness")
# GPU-2 arbitrary-prompt runs override the sealed prompt via env; the
# seal-token gates are only meaningful for the canonical prompt.
CANONICAL_PROMPT = os.environ.get("NATIVE_PROMPT", SEALED_PROMPT)
SEAL_APPLICABLE = CANONICAL_PROMPT == SEALED_PROMPT
N_TOKENS = int(os.environ.get("NATIVE_N_TOKENS", "16"))
RUN_ID = os.environ.get("NATIVE_RUN_ID", "unconfigured")
SOURCE_READ_LANES = int(os.environ.get("NATIVE_SOURCE_READ_LANES", "1"))
SOURCE_READ_QUEUE_DEPTH = int(
    os.environ.get("NATIVE_SOURCE_READ_QUEUE_DEPTH", "6"))
# Stage 1: raise the per-GPU VRAM expert cache from 512 MiB (~10 experts) to
# 3.5 GiB (~73 FP16 experts) — measured free VRAM after dense + engine is
# ~6.6/4.9 GiB on the two T4s, so 3.5 GiB/GPU keeps clear headroom.
BUDGET_BYTES = int(os.environ.get("NATIVE_BUDGET_BYTES", str(3584 << 20)))
# P2.3 packed FP4 residency: "fp16" (sealed exact path) or "fp4" (packed
# VRAM cache, decode-at-compute, experimental).  Env override for A/B runs.
CACHE_DTYPE = os.environ.get("NATIVE_CACHE_DTYPE", "fp16")
EXPERT_STORE_BACKEND = os.environ.get(
    "NATIVE_EXPERT_STORE", "safetensors").strip().lower()
DEE4_VALIDATE_SAMPLES = int(os.environ.get("NATIVE_DEE4_VALIDATE_SAMPLES", "12"))
DEE4_DIR = Path("/tmp/dsv4-dee4-v2")
DEE4_TRACE_DIR = Path("/tmp/dsv4-dee4-v3-trace")
DEE4_TRACE_PATH = Path(os.environ.get(
    "NATIVE_DEE4_TRACE_PATH", str(DEE4_TRACE_DIR)))
# V50 is the committed canonical route trace that defines this sparse store.
# Do not accept another well-formed journal here: the v51 bank must be bound
# to the exact sealed v50 bytes and terminal hash chain.
V50_TRACE_JOURNAL_RELATIVE = Path(
    "benchmark_reports/deepseek-v4-flash-0731-t4/"
    "v50-evidence-20260829T195940Z/routed_experts.jsonl")
V50_TRACE_JOURNAL_SHA256 = (
    "665aac3e8db570237c6dc6acaf08dc39f2af890e8a04e400ce7154f1a858dae1")
V50_TRACE_FINAL_CHAIN_SHA256 = (
    "086f8ca83b6a3c467cdf950096141fa9bc3e55a285d7d1fed8a0ad9913e3eb3d")
# Stage 1b (v8/v9/v10/v11/v12): host RAM LRU of packed FP4 expert bytes
# (12.6 MB/entry).  The 16-token working set is ~2,365 pairs ≈ 30 GiB;
# per-engine it is NOT symmetric: the v8 run measured 1,247 unique experts on
# cuda0 (layers 0-21, ≈ 16.7 GiB of packs) vs 916 on cuda1 (layers 22-42,
# ≈ 12.2 GiB).  Memory-safety history:
#   - v8: 12.87 GiB/GPU = 25.74 GiB total, NO madvise -> SURVIVED (correct
#     per-step memory, wrong tokens only because batched=True diverged).
#   - v9: 15.79/11.89 = 27.68 GiB -> OOM-killed.
#   - v11: 14.5/11.5 = 26.0 GiB + madvise ON -> OOM-killed; the DONTNEED
#     re-faulted evicted experts against the slow loop device (~4 MB/s,
#     ~3.4 s/miss) instead of hitting the page cache.
# v12 therefore returns to the v8-PROVEN ceiling (12.87 GiB/GPU symmetric)
# and leaves madvise OFF by default (env DEE_MADVISE_DONTNEED=1 opts in).
# The cuda0 shortfall (16.7 GiB set vs 12.87 budget) means cuda0 still
# evicts ~3.8 GiB of experts per pass; those re-faults now hit the page
# cache (v8 behavior) rather than the loop device.
HOST_PACK_CACHE_BYTES_GPU0 = int(os.environ.get(
    "NATIVE_HOST_PACK_GPU0_BYTES", str(int(12.87 * (1 << 30)))))
HOST_PACK_CACHE_BYTES_GPU1 = int(os.environ.get(
    "NATIVE_HOST_PACK_GPU1_BYTES", str(int(12.87 * (1 << 30)))))
# Stage 2 (v9): the pointer-batched SwiGLU path (cublasGemmBatchedEx) is a
# DIFFERENT numerical kernel than the per-expert path (cublasGemmEx per
# expert): a 1-ULP FP16 difference flips greedy tokens. v8 enabled it and
# DIVERGED from the v7 gate tokens. It stays OFF by default so the strict
# token-identity gate holds; it remains an experimental speed mode.
USE_BATCHED_EXPERTS = os.environ.get("NATIVE_BATCHED", "0") == "1"
# Stage 0 (v8): diagnostics/profiling are opt-in. Headline timing must not
# include per-layer finite checks, route serialization, checksums, or CUDA
# timing-event allocation. Run a separate diagnostic pass with both enabled.
PROFILE_STAGES = os.environ.get("NATIVE_PROFILE", "0") == "1"
DIAGNOSTICS = os.environ.get("NATIVE_DIAGNOSTICS", "0") == "1"
# ── Phase-4 arm-matrix knobs (kickoff §8, Wave-B B2) ─────────────────────
# TRACE_REQUESTS -> EngineConfig.trace_requests: emit one RequestTraceRecord
# per expert request (token/layer/kind/occupancy/evicted/reuse_distance).
# The records serialize into external_profile_json["trace"], which the
# runner also mirrors into a durable cache_events{suffix}.jsonl sink.
# Requires profile_stages; if NATIVE_PROFILE is not "1" the runner
# auto-enables it with a loud log (see main()).
TRACE_REQUESTS = os.environ.get("NATIVE_TRACE_REQUESTS", "0") == "1"
# EVICTION_POLICY -> EngineConfig.eviction_policy ("lru"|"rank_priority").
# The pydee binding is being added by a parallel workstream;
# build_native_engine applies it via hasattr-fallback so this harness is
# correct under both old and new binaries.  "rank_priority" is the legacy
# Phase-3 behavior (descending expert-ID priority), kept for A/B.
EVICTION_POLICY = os.environ.get(
    "NATIVE_EVICTION_POLICY", "rank_priority").strip().lower()
# HOST_CACHE_MODE -> EngineConfig.host_cache_mode ("lru"|"bypass").
# "bypass" = pack budget clamped to exactly one packed record (a bounce
# buffer that avoids both caching and the FUSE mmap-fallback death path)
# with source read lanes forced to 1 -- enforced below so the contract
# holds even when the engine binding is absent.
HOST_CACHE_MODE = os.environ.get(
    "NATIVE_HOST_CACHE_MODE", "lru").strip().lower()
# ARM_ID: run-metadata only.  NEVER folded into RUN_ID: run_id is inside
# the route-journal canonical hash payload, so it must stay constant
# across arms for journals to hash-compare.
ARM_ID = os.environ.get("NATIVE_ARM_ID", "").strip()
# CACHE_RESET: "warm" (default) = today's behavior: only the external
# profile (measurement counters + trace) resets at each prompt boundary;
# VRAM arena / host pack / store stats persist across prompts.  "cold" =
# additionally evict the VRAM arena + host pack + fp4 staging metadata and
# reset ExpertStore counters before EVERY prompt.  The OS page cache is
# never cleared either way (logged loudly at each cold reset).
CACHE_RESET = os.environ.get("NATIVE_CACHE_RESET", "warm").strip().lower()
# IGNORE_EOS: "1" disables generate()'s eos early-stop so every prompt
# produces exactly N_TOKENS decode steps -- the campaign's fixed-length
# workload semantics (uniform 128-forward streams make wall/ITL and the
# driver's n_tokens acceptance gate comparable across arms/prompts).
# Exactness is unaffected: tokens remain the deterministic greedy output;
# only the stopping rule changes.  Default "0" = natural stop (legacy).
IGNORE_EOS = os.environ.get("NATIVE_IGNORE_EOS", "0") == "1"
# P5b mechanism arms: NATIVE_TORCH_DETERMINISTIC=1 arms
# torch.use_deterministic_algorithms(warn_only) + deterministic cudnn —
# a torch-side determinism probe for the warm-process divergence.  The
# CUBLAS_WORKSPACE_CONFIG env itself must come from the spawning process
# (read at first cuBLAS handle creation); the session driver sets it.
TORCH_DETERMINISTIC = os.environ.get(
    "NATIVE_TORCH_DETERMINISTIC", "0") == "1"
# NATIVE_FFN_BACKEND: "native" (default; dee_core engine path via
# moe_forward_batch_device) or "cache_fp16" (reference torch expert path —
# DeepseekV4CacheFfn computes routed experts with torch GEMMs over FP16
# payloads from the provider, bypassing the native engine entirely).
# P5b mechanism arm: convicts/exonerates the native FFN path for the
# warm-process divergence.
FFN_BACKEND = os.environ.get(
    "NATIVE_FFN_BACKEND", "native").strip().lower()
# Python-side reference cache budget used only when FFN_BACKEND != native.
# FP16 expert payloads are ~50 MiB each — 1 GiB keeps ~20 resident.
REF_CACHE_BYTES = int(os.environ.get(
    "NATIVE_REF_CACHE_BYTES", str(1024 << 20)))
# NATIVE_CAPTURE_JOURNAL=1 writes captures{suffix}.jsonl: per
# (step, layer) sha256 of the FFN capture tensors (moe_out combined,
# shared_out, router_scores, expert_ids, routing_weights) — splits
# "routed-expert output" vs "shared-expert output" vs "attention-side
# input" inside the divergent layer boundary.  Sequential path only.
CAPTURE_JOURNAL = os.environ.get(
    "NATIVE_CAPTURE_JOURNAL", "0") == "1"
# NATIVE_ROUTE_WEIGHT_JOURNAL=1 writes route_weights{suffix}.jsonl:
# per (forward_step, layer) sha256 of the routing-weight + expert-id
# matrices already collected for diagnostics — a continuous-value
# fingerprint of the hidden state entering each layer's router, used to
# bisect the first divergent layer across sequential units.
ROUTE_WEIGHT_JOURNAL = os.environ.get(
    "NATIVE_ROUTE_WEIGHT_JOURNAL", "0") == "1"
# v15: return to v8-PROVEN storage behavior.  v13's discard_source_pages
# (posix_fadvise + MADV_DONTNEED on the shared mmap after every pack fill)
# re-introduced the v10 behavior that v12 measured as OOM + re-fault
# thrash against the loop device (~4 MB/s): v8 (no discard, page-cache
# hits on evicted re-reads) survived 16/16 tokens at 225 s/token, while
# v14 (discard ON) OOM'd at token 9 at 369 s/token.  The engine default is
# ON, so this runtime opts OUT explicitly.  The v15 LRU cap (17 GiB)
# bounds the anonymous side regardless.
os.environ.setdefault("DEE_RELEASE_MMAP_PAGES", "0")
# P2.4 storage decision (2026-08-23 probe): the dataset mount is a ~13 MB/s
# loop device (95.7% of v15/v16 decode wall), while the /tmp root overlay
# measured 1,550-1,830 MB/s pread / ~9-11 GB/s mmap on the same GPU worker
# class.  v15/v16 preferred the mount when present, which is exactly the
# bottleneck.  NATIVE_FORCE_TMP=1 stages all shards into /tmp (copy from the
# mount when present, else HF download) so the engine's expert mmap reads
# hit the fast overlay instead of the loop device.  Default OFF until a
# clean 2-GPU run proves the 16/16 token gate holds with the staged path.
FORCE_TMP = os.environ.get("NATIVE_FORCE_TMP", "1") == "1"

# P2.3 A/B: Kaggle kernel metadata env_vars are not reliably passed to the
# script, so commit-time knobs live in run_config.json next to this file.
# The kernel clones the branch and reads it from the working tree; the file
# only exists AFTER the clone, so this is applied lazily in main().  Env
# overrides still win when actually set.
def apply_run_config() -> None:
    global CACHE_DTYPE, N_TOKENS, EXPERT_STORE_BACKEND, DEE4_VALIDATE_SAMPLES
    global PROFILE_STAGES, RUN_ID, DEE4_TRACE_PATH
    global SOURCE_READ_LANES, SOURCE_READ_QUEUE_DEPTH
    global TRACE_REQUESTS, EVICTION_POLICY, HOST_CACHE_MODE
    global ARM_ID, CACHE_RESET, IGNORE_EOS
    cfg_path = DEE / "kaggle/deepseek-v4-flash-0731/run_config.json"
    if not cfg_path.is_file():
        log(f"[config] run_config.json not found at {cfg_path}; using defaults")
        return
    cfg = json.loads(cfg_path.read_text("utf-8"))
    if not os.environ.get("NATIVE_CACHE_DTYPE"):
        CACHE_DTYPE = str(cfg.get("cache_dtype", CACHE_DTYPE)).strip().lower()
    if not os.environ.get("NATIVE_N_TOKENS"):
        N_TOKENS = int(cfg.get("n_tokens", N_TOKENS))
    if not os.environ.get("NATIVE_EXPERT_STORE"):
        EXPERT_STORE_BACKEND = str(
            cfg.get("expert_store", EXPERT_STORE_BACKEND)).strip().lower()
    if not os.environ.get("NATIVE_DEE4_VALIDATE_SAMPLES"):
        DEE4_VALIDATE_SAMPLES = int(
            cfg.get("dee4_validate_samples", DEE4_VALIDATE_SAMPLES))
    if not os.environ.get("NATIVE_DEE4_TRACE_PATH"):
        DEE4_TRACE_PATH = Path(cfg.get("dee4_trace_path", DEE4_TRACE_PATH))
    if not os.environ.get("NATIVE_PROFILE"):
        PROFILE_STAGES = bool(cfg.get("profile_stages", PROFILE_STAGES))
    if not os.environ.get("NATIVE_RUN_ID"):
        RUN_ID = str(cfg.get("run_id", RUN_ID))
    if not os.environ.get("NATIVE_SOURCE_READ_LANES"):
        SOURCE_READ_LANES = int(
            cfg.get("source_read_lanes", SOURCE_READ_LANES))
    if not os.environ.get("NATIVE_SOURCE_READ_QUEUE_DEPTH"):
        SOURCE_READ_QUEUE_DEPTH = int(
            cfg.get("source_read_queue_depth", SOURCE_READ_QUEUE_DEPTH))
    # Phase-4 knobs follow the same contract: committed values apply only
    # when the env var is absent (env always wins when set).
    if not os.environ.get("NATIVE_TRACE_REQUESTS"):
        TRACE_REQUESTS = bool(cfg.get("trace_requests", TRACE_REQUESTS))
    if not os.environ.get("NATIVE_EVICTION_POLICY"):
        EVICTION_POLICY = str(
            cfg.get("eviction_policy", EVICTION_POLICY)).strip().lower()
    if not os.environ.get("NATIVE_HOST_CACHE_MODE"):
        HOST_CACHE_MODE = str(
            cfg.get("host_cache_mode", HOST_CACHE_MODE)).strip().lower()
    if not os.environ.get("NATIVE_ARM_ID"):
        ARM_ID = str(cfg.get("arm_id", ARM_ID)).strip()
    if not os.environ.get("NATIVE_CACHE_RESET"):
        CACHE_RESET = str(cfg.get("cache_reset", CACHE_RESET)).strip().lower()
    if not os.environ.get("NATIVE_IGNORE_EOS"):
        IGNORE_EOS = bool(cfg.get("ignore_eos", IGNORE_EOS))
    if CACHE_DTYPE not in {"fp16", "fp4"}:
        raise ValueError(f"unsupported cache_dtype: {CACHE_DTYPE!r}")
    if EXPERT_STORE_BACKEND not in {"safetensors", "dee4", "dee4_trace",
                                    "dee4_segmented"}:
        raise ValueError(
            f"unsupported expert_store: {EXPERT_STORE_BACKEND!r}")
    if DEE4_VALIDATE_SAMPLES <= 0:
        raise ValueError("dee4_validate_samples must be positive")
    if not 1 <= SOURCE_READ_LANES <= 8:
        raise ValueError("source_read_lanes must be in [1, 8]")
    if not 1 <= SOURCE_READ_QUEUE_DEPTH <= 256:
        raise ValueError("source_read_queue_depth must be in [1, 256]")
    if EVICTION_POLICY not in {"lru", "rank_priority"}:
        raise ValueError(f"unsupported eviction_policy: {EVICTION_POLICY!r}")
    if HOST_CACHE_MODE not in {"lru", "bypass"}:
        raise ValueError(f"unsupported host_cache_mode: {HOST_CACHE_MODE!r}")
    if CACHE_RESET not in {"warm", "cold"}:
        raise ValueError(f"unsupported cache_reset: {CACHE_RESET!r}")
    log(
        "[config] run_config.json: "
        f"run_id={RUN_ID} arm_id={ARM_ID or 'none'} "
        f"cache_dtype={CACHE_DTYPE} n_tokens={N_TOKENS} "
        f"expert_store={EXPERT_STORE_BACKEND} "
        f"dee4_trace_path={DEE4_TRACE_PATH} "
        f"dee4_validate_samples={DEE4_VALIDATE_SAMPLES} "
        f"source_read_lanes={SOURCE_READ_LANES} "
        f"source_read_queue_depth={SOURCE_READ_QUEUE_DEPTH} "
        f"profile_stages={PROFILE_STAGES} "
        f"trace_requests={TRACE_REQUESTS} "
        f"eviction_policy={EVICTION_POLICY} "
        f"host_cache_mode={HOST_CACHE_MODE} "
        f"cache_reset={CACHE_RESET} "
        f"ignore_eos={IGNORE_EOS}"
    )
# P2.4 (2026-08-23): the dual-T4 pool has been exhausted for ~12 consecutive
# launches (Kaggle hands out 1x P100 instead).  SINGLE_GPU runs the full
# 43-layer model on one CUDA device (split=n_layers, same-device handoff,
# one engine with a capped budget).  NATIVE_SINGLE_GPU=1 forces it;
# otherwise check_gpu_allocation() flips it on when the worker only has one
# GPU (Kaggle metadata env_vars are NOT reliably passed, so this must be
# auto-detected).  The 16/16-token gate is arch-independent (sm_60/sm_75
# cubins, same math), so a P100 run validates the identical correctness
# contract while the T4 pool recovers; the log labels hardware so
# performance numbers stay honest.
SINGLE_GPU = os.environ.get("NATIVE_SINGLE_GPU", "0") == "1"
PROGRESS = WORK / "progress.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _ntfy(msg)
    try:
        with open(PROGRESS, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_evidence(name: str, payload: dict) -> None:
    """Atomically publish one required remote evidence artifact."""
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), "utf-8")
    temporary.replace(path)


class _SkipDee4Prepare(Exception):
    """Raised after the dee4_segmented fast path sets dee4_store_path; the
    repack/validate evidence block is skipped entirely for a pre-built
    segmented store."""


class RoutedExpertJournal:
    """Durable, canonical route-ID journal for long full-model runs.

    The native FFN already copies its compact ``[rows, topk]`` route matrix
    into pinned host memory before dispatch.  The harness passes that existing
    buffer here after each layer, so journaling adds no CUDA event, device
    transfer, or synchronization.  Records are emitted strictly in
    ``forward_step, layer, token_row, topk_rank`` order.  Each record hashes
    its canonical JSON payload, including the preceding record hash, forming
    an incrementally verifiable chain.

    Every layer is flushed to the kernel immediately.  Layer ``n_layers - 1``
    also fsyncs the file before the generated-token checkpoint is allowed to
    link to it.  This keeps the per-layer failure boundary visible while
    limiting the stronger filesystem barrier to once per model forward.
    """

    SCHEMA_VERSION = 1
    GENESIS_SHA256 = hashlib.sha256(b"").hexdigest()
    CANONICAL_ORDER = "forward_step,layer,token_row,topk_rank"

    def __init__(self, path: Path, *, run_id: str, n_layers: int,
                 topk: int) -> None:
        if n_layers <= 0 or topk <= 0:
            raise ValueError("route journal n_layers/topk must be positive")
        self.path = Path(path)
        self.run_id = str(run_id)
        self.n_layers = int(n_layers)
        self.topk = int(topk)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "w", encoding="utf-8", newline="\n")
        self._chain_sha256 = self.GENESIS_SHA256
        self._record_count = 0
        self._completed_forwards = 0
        self._checkpoint_steps: list[int] = []
        self._last_step: int | None = None
        self._last_layer: int | None = None
        self._closed = False

    @staticmethod
    def _canonical_bytes(payload: dict) -> bytes:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("utf-8")

    def append_layer(self, *, step: int, start_pos: int, layer: int,
                     device: str, expert_ids) -> dict:
        """Append one layer's rank-ordered route matrix and flush it."""
        if self._closed:
            raise RuntimeError("route journal is closed")
        expected_step = self._completed_forwards
        expected_layer = self._record_count % self.n_layers
        if int(step) != expected_step or int(layer) != expected_layer:
            raise RuntimeError(
                "non-canonical route journal order: "
                f"got step={step} layer={layer}, expected "
                f"step={expected_step} layer={expected_layer}")
        raw_rows = (expert_ids.tolist()
                    if hasattr(expert_ids, "tolist") else expert_ids)
        if not isinstance(raw_rows, (list, tuple)) or not raw_rows:
            raise ValueError("route journal expert_ids must be a non-empty matrix")
        rows: list[list[int]] = []
        for row_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, (list, tuple)):
                raise ValueError(
                    f"route journal row {row_index} is not a sequence")
            row = [int(value) for value in raw_row]
            if len(row) != self.topk:
                raise ValueError(
                    f"route journal row {row_index} topk={len(row)}; "
                    f"expected {self.topk}")
            if any(value < 0 for value in row):
                raise ValueError(
                    f"route journal row {row_index} has negative expert id")
            rows.append(row)

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "record_index": self._record_count,
            "forward_step": int(step),
            "phase": "prefill" if int(step) == 0 else "decode",
            "start_pos": int(start_pos),
            "layer": int(layer),
            "device": str(device),
            "token_rows": len(rows),
            "topk": self.topk,
            "expert_ids_rank_order": rows,
            "canonical_order": self.CANONICAL_ORDER,
            "previous_chain_sha256": self._chain_sha256,
        }
        chain_sha256 = hashlib.sha256(
            self._canonical_bytes(payload)).hexdigest()
        record = {**payload, "chain_sha256": chain_sha256}
        line = self._canonical_bytes(record).decode("utf-8") + "\n"
        self._handle.write(line)
        self._handle.flush()

        self._chain_sha256 = chain_sha256
        self._record_count += 1
        self._last_step = int(step)
        self._last_layer = int(layer)
        if int(layer) == self.n_layers - 1:
            # This is the durable boundary referenced by the token journal.
            os.fsync(self._handle.fileno())
            self._completed_forwards += 1
        return record

    def checkpoint_link(self, step: int) -> dict:
        """Return a fail-closed link from one token checkpoint to layer N-1."""
        step = int(step)
        if step != len(self._checkpoint_steps):
            raise RuntimeError(
                f"route checkpoint link out of order: step={step}, "
                f"expected={len(self._checkpoint_steps)}")
        if (self._last_step != step
                or self._last_layer != self.n_layers - 1
                or self._completed_forwards != step + 1):
            raise RuntimeError(
                "token checkpoint cannot link an incomplete route forward: "
                f"step={step} last_step={self._last_step} "
                f"last_layer={self._last_layer} "
                f"completed={self._completed_forwards}")
        self._checkpoint_steps.append(step)
        return {
            "artifact": self.path.name,
            "schema_version": self.SCHEMA_VERSION,
            "forward_step": step,
            "terminal_layer": self.n_layers - 1,
            "record_count": self._record_count,
            "chain_sha256": self._chain_sha256,
            "file_bytes": self.path.stat().st_size,
            "layer_flush_complete": True,
            "terminal_layer_fsync_complete": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._closed = True

    def summary(self) -> dict:
        if not self._closed:
            self._handle.flush()
        file_bytes = self.path.stat().st_size
        return {
            "artifact": self.path.name,
            "schema_version": self.SCHEMA_VERSION,
            "canonical_order": self.CANONICAL_ORDER,
            "n_layers": self.n_layers,
            "topk": self.topk,
            "record_count": self._record_count,
            "completed_forwards": self._completed_forwards,
            "checkpoint_link_count": len(self._checkpoint_steps),
            "checkpoint_steps": list(self._checkpoint_steps),
            "last_forward_step": self._last_step,
            "last_layer": self._last_layer,
            "genesis_sha256": self.GENESIS_SHA256,
            "chain_sha256": self._chain_sha256,
            "file_sha256": sha256_file(self.path),
            "file_bytes": file_bytes,
            "flush_each_layer": True,
            "fsync_each_completed_forward": True,
            "source": "existing_native_pinned_route_id_buffer",
            "adds_cuda_events": False,
            "adds_device_transfers": False,
            "adds_host_synchronizations": False,
        }


def _weight_journal_rec(layer, *, step: int, start_pos: int) -> str:
    """One route_weights jsonl line: sha256 of the routing-weight and
    expert-id matrices in ``layer.ffn_fn.last_route`` (already collected
    when diagnostics are on — no extra device reads).  The weights are a
    continuous fingerprint of the hidden state entering this layer's
    router, so the first divergent (step, layer) localizes injection."""
    lr = getattr(layer.ffn_fn, "last_route", None) or {}
    w = lr.get("routing_weights")
    i = lr.get("expert_ids")
    rec = {"step": int(step), "start_pos": int(start_pos),
           "layer": int(getattr(layer, "layer_id", -1)),
           "weights_sha256": (hashlib.sha256(
               json.dumps(w).encode()).hexdigest()
               if w is not None else None),
           "ids_sha256": (hashlib.sha256(
               json.dumps(i).encode()).hexdigest()
               if i is not None else None)}
    return json.dumps(rec) + "\n"


def classify_full_generation(result: dict) -> tuple[str, dict, bool]:
    """Apply the sealed exactness contract and hardware gate fail-closed."""
    tokens = [int(token) for token in result.get("generated_token_ids", [])]
    bridge = result.get("bridge_counters", {})
    engine_stats = result.get("engine_stats", {})
    engine_config = result.get("engine_config", {})
    expert_store = result.get("expert_store", {})
    runtime = result.get("model_runtime_snapshot", {})
    gpu = result.get("gpu_environment", {})
    gpu_lines = gpu.get("nvidia_smi_lines", [])

    expected_cache = "fp4-packed" if CACHE_DTYPE == "fp4" else "fp16"
    required_bridge_zero = {
        "numpy_bridge_calls": int(bridge.get("numpy_bridge_calls", -1)) == 0,
        "full_hidden_d2h_copies":
            int(bridge.get("full_hidden_d2h_copies", -1)) == 0,
        "raw_expert_output_d2h_copies":
            int(bridge.get("raw_expert_output_d2h_copies", -1)) == 0,
    }
    engine_keys = ("cuda0",) if gpu.get("single_gpu_mode") else ("cuda0", "cuda1")
    finite_values = [
        engine_stats.get(key, {}).get("hidden_finite")
        for key in engine_keys]
    finite_outputs_observed = all(
        isinstance(value, bool) for value in finite_values)
    finite_outputs = finite_outputs_observed and all(finite_values)
    effective_cache_dtype = all(
        engine_config.get(key, {}).get("cache_dtype") == expected_cache
        for key in engine_keys)
    effective_cuda = all(
        engine_config.get(key, {}).get("use_cuda") is True
        for key in engine_keys)
    expected_store = all(
        expert_store.get(key, {}).get("backend") == EXPERT_STORE_BACKEND
        and int(expert_store.get(key, {}).get("lookup_failures", -1)) == 0
        for key in engine_keys)
    dee4_contiguous = True
    dee4_integrity = True
    if EXPERT_STORE_BACKEND in {"dee4", "dee4_trace", "dee4_segmented"}:
        dee4_contiguous = all(
            int(expert_store.get(key, {}).get("source_reads", 0)) > 0
            and int(expert_store.get(key, {}).get("contiguous_source_reads", -1))
            == int(expert_store.get(key, {}).get("source_reads", 0))
            for key in engine_keys)
        dee4_integrity = all(
            len(str(expert_store.get(key, {}).get("integrity_identity", ""))) == 64
            for key in engine_keys)
    trace_store = result.get("dee4_trace_validation", {})
    trace_store_linked = True
    if EXPERT_STORE_BACKEND == "dee4_trace":
        trace_store_linked = (
            trace_store.get("success") is True
            and trace_store.get("format") == "dee4-v3-trace"
            and trace_store.get("record_indices_contiguous") is True
            and trace_store.get("integrity_records_complete") is True
            and all(len(str(trace_store.get(key, ""))) == 64 for key in (
                "data_sha256", "trace_journal_sha256",
                "trace_final_chain_sha256", "selection_sha256"))
        )
    no_cpu_expert_fallback = (
        runtime.get("backends", {}).get("cpu_expert_execution") is False)
    route_journal = result.get("route_journal", {})
    route_journal_complete = (
        int(route_journal.get("schema_version", -1)) == 1
        and route_journal.get("canonical_order")
        == RoutedExpertJournal.CANONICAL_ORDER
        and int(route_journal.get("n_layers", -1)) == 43
        and int(route_journal.get("topk", -1)) == 6
        and int(route_journal.get("record_count", -1)) == len(tokens) * 43
        and int(route_journal.get("completed_forwards", -1)) == len(tokens)
        and int(route_journal.get("checkpoint_link_count", -1)) == len(tokens)
        and route_journal.get("checkpoint_steps") == list(range(len(tokens)))
        and int(route_journal.get("last_forward_step", -1)) == len(tokens) - 1
        and int(route_journal.get("last_layer", -1)) == 42
        and len(str(route_journal.get("chain_sha256", ""))) == 64
        and len(str(route_journal.get("file_sha256", ""))) == 64
        and route_journal.get("flush_each_layer") is True
        and route_journal.get("fsync_each_completed_forward") is True
    )
    t4_hardware = (
        int(gpu.get("gpu_count", 0)) == 2
        and len(gpu_lines) == 2
        and all("Tesla T4" in str(line) for line in gpu_lines))

    gates = {
        "exact_16_token_ids": (SEAL_APPLICABLE and N_TOKENS == 16
                              and tokens == SEALED_TOKEN_IDS),
        "exact_decoded_text": (SEAL_APPLICABLE
                              and result.get("decoded_text")
                              == SEALED_DECODED_TEXT),
        "all_43_layers": int(result.get("layer_count_executed", -1)) == 43,
        "finite_outputs_observed": finite_outputs_observed,
        "finite_outputs": finite_outputs,
        **required_bridge_zero,
        "official_router_authoritative": (
            runtime.get("backends", {}).get("router")
            == "torch_cuda_validated_ds9_path"),
        "no_cpu_expert_fallback": no_cpu_expert_fallback,
        "effective_cuda_execution": effective_cuda,
        "effective_cache_dtype": effective_cache_dtype,
        "effective_expert_store": expected_store,
        "dee4_contiguous_reads": dee4_contiguous,
        "dee4_integrity_identity": dee4_integrity,
        "dee4_trace_metadata_linkage": trace_store_linked,
        "route_journal_complete": route_journal_complete,
        "required_performance_hardware": t4_hardware,
    }
    token_or_text_failed = (SEAL_APPLICABLE and not (
        gates["exact_16_token_ids"] and gates["exact_decoded_text"]))
    observed_nonfinite = (
        gates["finite_outputs_observed"] and not gates["finite_outputs"])
    seal_keys = {"exact_16_token_ids", "exact_decoded_text"}
    contract_gates = [value for key, value in gates.items()
                      if key != "required_performance_hardware"
                      and (SEAL_APPLICABLE or key not in seal_keys)]
    if token_or_text_failed or observed_nonfinite:
        classification = "REJECT_NUMERICAL"
    elif not all(contract_gates):
        classification = "REJECT_INTEGRITY"
    else:
        classification = "ACCEPT_CORRECTNESS"
    performance_eligible = classification == "ACCEPT_CORRECTNESS" and t4_hardware
    return classification, gates, performance_eligible


def mem_report(tag: str) -> None:
    """Heartbeat: current host RAM so a silent OOM kill is attributable."""
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            for key in ("MemTotal", "MemAvailable"):
                if line.startswith(key + ":"):
                    mem[key] = round(int(line.split()[1]) / (1024 * 1024), 1)
        log(f"[mem:{tag}] {mem}")
    except OSError:
        pass


def run(cmd, **kw):
    log("+ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        log(f"FAILED (exit {r.returncode})")
        raise RuntimeError("command failed: "
                           + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    return r


def _download_one(shard: str) -> Path:
    CKPT.mkdir(parents=True, exist_ok=True)
    dest = CKPT / shard
    url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
           f"resolve/{REV}/{shard}")
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        cr = resp.headers.get("Content-Range", "")
    if not cr or "/" not in cr:
        raise RuntimeError(f"no Content-Range for {shard}")
    want = int(cr.split("/")[1])
    have = dest.stat().st_size if dest.is_file() else 0
    if have == want:
        return dest
    chunk = 16 << 20
    with open(dest, "ab") as fh:
        while have < want:
            end = min(have + chunk - 1, want - 1)
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={have}-{end}"})
            last = None
            for attempt in range(8):
                try:
                    with urllib.request.urlopen(req, timeout=600) as r:
                        data = r.read(chunk + 1)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError,
                        ConnectionError, TimeoutError) as e:
                    last = e
                    time.sleep(2.0 * (2 ** attempt))
            else:
                raise ConnectionError(f"{shard} download failed: {last!r}")
            fh.write(data)
            have += len(data)
    free = shutil.disk_usage(str(CKPT)).free / (1 << 30)
    log(f"[download] {shard} complete ({want / (1 << 30):.2f} GiB, "
        f"{free:.0f} GiB scratch free)")
    return dest


def _snapshot_download(shards: list[str]) -> bool:
    """Fast path: huggingface_hub + xet (~263 MB/s measured on a single T4)."""
    from huggingface_hub import snapshot_download
    t0 = time.monotonic()
    snapshot_download(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision=REV,
        local_dir=str(CKPT),
        allow_patterns="model-*.safetensors",
        max_workers=4,
    )
    total = sum((CKPT / s).stat().st_size for s in shards
                if (CKPT / s).is_file()) / (1 << 30)
    log(f"[download] snapshot_download {total:.1f} GiB in "
        f"{time.monotonic() - t0:.0f}s")
    return True


def _copy_mount_to_tmp(shards: list[str], dst_dir: Path) -> list[str]:
    """Sequential per-shard copy from the dataset mount to /tmp.

    The mount loop device does ~194 MB/s on large sequential reads (P2.2
    evidence) but ~13 MB/s on mmap scatter (v15/v16).  One sequential copy
    per shard amortizes to a few minutes total and makes all later expert
    reads hit the fast root overlay instead.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    total_bytes = 0
    for s in shards:
        src = DATASET_DIR / s
        dst = dst_dir / s
        if not src.is_file():
            raise FileNotFoundError(f"mount shard missing: {src}")
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            total_bytes += dst.stat().st_size
            continue
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=32 << 20)
        if dst.stat().st_size != src.stat().st_size:
            raise IOError(f"copy size mismatch for {s}")
        total_bytes += dst.stat().st_size
        mbps = total_bytes / (1 << 20) / max(time.monotonic() - t0, 1e-9)
        log(f"[download] copied {s} ({dst.stat().st_size/(1<<30):.2f} GiB) "
            f"agg {mbps:.0f} MB/s")
    dt = time.monotonic() - t0
    log(f"[download] mount->/tmp staged {total_bytes/(1<<30):.1f} GiB in "
        f"{dt:.0f}s ({total_bytes/(1<<20)/max(dt,1e-9):.0f} MB/s)")
    return [str(dst_dir / s) for s in shards]


def download_all_shards() -> list[str]:
    shards = [f"model-{i:05d}-of-00048.safetensors"
              for i in range(1, N_SHARDS + 1)]
    paths = [str(CKPT / s) for s in shards]
    # P2.4: NATIVE_FORCE_TMP=1 stages everything into /tmp (fast root
    # overlay, ~1.5 GB/s) instead of reading experts from the ~13 MB/s
    # dataset mount.  Copy from the mount when present (no re-download),
    # else snapshot_download from HF.
    if FORCE_TMP:
        CKPT.mkdir(parents=True, exist_ok=True)
        if DATASET_DIR.is_dir() and all((DATASET_DIR / s).is_file()
                                        for s in shards):
            # Complete mount: use it DIRECTLY.  Staging/copying 153 GiB
            # into /tmp exceeds single-GPU container disk quotas (v33-v37
            # were hard-killed silently mid-download); dual-GPU mounts are
            # the intended source.
            ds_paths = [str(DATASET_DIR / s) for s in shards]
            log(f"[download] complete dataset mount at {DATASET_DIR}; "
                f"using directly (no /tmp staging)")
            return ds_paths
        log(f"[download] FORCE_TMP: mount absent/incomplete; downloading")
        if not _snapshot_download(shards):
            raise RuntimeError("snapshot_download failed")
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            log(f"[download] {len(missing)} shards missing; range-fetch fallback")
            failures = []

            def work(p):
                try:
                    return _download_one(Path(p).name)
                except Exception as e:  # noqa: BLE001
                    failures.append((p, repr(e)))
                    return None

            with ThreadPoolExecutor(max_workers=3) as ex:
                list(ex.map(work, missing))
            if failures:
                raise RuntimeError(f"{len(failures)} downloads failed: {failures}")
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise RuntimeError(f"{len(missing)} shards still missing")
        log_host_resources("post-stage")
        return paths
    # Legacy path: dataset-mounted checkpoint, no download, no disk quota.
    if DATASET_DIR.is_dir():
        ds_paths = [str(DATASET_DIR / s) for s in shards]
        if all(Path(p).is_file() for p in ds_paths):
            log(f"[download] using dataset-mounted checkpoint at {DATASET_DIR}")
            return ds_paths
        log(f"[download] dataset dir present but incomplete; downloading")
    CKPT.mkdir(parents=True, exist_ok=True)
    if not _snapshot_download(shards):
        raise RuntimeError("snapshot_download failed")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        # Fall back to resume-capable range fetches for whatever is missing.
        log(f"[download] {len(missing)} shards missing; range-fetch fallback")
        failures = []

        def work(p):
            try:
                return _download_one(Path(p).name)
            except Exception as e:  # noqa: BLE001
                failures.append((p, repr(e)))
                return None

        with ThreadPoolExecutor(max_workers=3) as ex:
            list(ex.map(work, missing))
        if failures:
            raise RuntimeError(f"{len(failures)} downloads failed: {failures}")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} shards still missing after download")
    return paths


def host_mem_available_gib() -> float:
    """MemAvailable from /proc/meminfo, in GiB (0.0 on failure)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def host_mem_total_gib() -> float:
    """MemTotal from /proc/meminfo, in GiB (0.0 on failure)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def process_mem_gib() -> dict:
    """This process's VmRSS/VmData/VmLck/VmSwap from /proc/self/status.

    VmData = anonymous heap growth; VmLck = pinned (cudaMallocHost) growth;
    the gap between VmRSS and (VmData + VmLck) is file-backed page cache
    attributed to this process.  This is the definitive leak localizer for
    the v12/v14 decode-time host-RAM growth.
    """
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            for key in ("VmRSS", "VmHWM", "VmData", "VmPeak", "VmLck", "VmSwap"):
                if line.startswith(key + ":"):
                    out[key] = round(int(line.split()[1]) / (1024 * 1024), 2)
    except OSError:
        pass
    return out


def system_mem_gib() -> dict:
    """MemTotal/MemAvailable/Cached/SReclaimable, in GiB."""
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            for key in ("MemTotal", "MemAvailable", "Cached", "SReclaimable"):
                if line.startswith(key + ":"):
                    out[key] = round(int(line.split()[1]) / (1024 * 1024), 2)
    except OSError:
        pass
    return out


def gpu_memory_snapshot() -> dict:
    import torch
    out = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            out[f"cuda{i}"] = {
                "allocated_gib": round(torch.cuda.memory_allocated(i) / (1 << 30), 3),
                "reserved_gib": round(torch.cuda.memory_reserved(i) / (1 << 30), 3),
                "free_gib": round(free / (1 << 30), 3),
                "total_gib": round(total / (1 << 30), 3),
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated(i) / (1 << 30), 3),
            }
    return out


def check_gpu_allocation() -> dict:
    """Fail fast (before the ~40-min build) if the worker lacks 2 GPUs.

    Kaggle's Dual-GPU pool intermittently allocates 1 GPU (v17/v18/v19 all
    hit this AFTER a full build + P2.2 repack).  Exiting early turns a wasted
    45 minutes into a 5-second failure we can re-push immediately.

    The 16/16-token gate is arch-independent (same cubin math on sm_60/sm_75),
    so any 2-GPU worker validates correctness; the log records the actual
    hardware so performance numbers are labeled correctly (T4 vs P100).
    """
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True,
                                      stderr=subprocess.STDOUT)
    except Exception as e:
        log(f"GPU_ALLOC_FAIL nvidia-smi unavailable: {e}")
        raise RuntimeError(f"expected 2 GPUs, nvidia-smi failed: {e}")
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    n_gpus = len(lines)
    log(f"GPU_ALLOC n={n_gpus}: " + " | ".join(lines))
    # The Kaggle preinstalled torch wheel (2.10+cu128) has no sm_60 kernels:
    # any P100 allocation dies later in torch.zeros with
    # cudaErrorNoKernelImageForDevice.  Instead of rejecting the worker,
    # REPAIR it: install torch 2.3.1+cu118 (the last line with sm_60 support,
    # verified by the p100_torch_probe kernel: matmul PASS on P100).  This
    # makes every allocation usable and ends the T4-pool deadlock.
    try:
        cc_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader"], text=True)
        caps = [float(x.strip()) for x in cc_out.splitlines() if x.strip()]
    except Exception:
        caps = []
    if caps and min(caps) < 7.0:
        log(f"TORCH_REPAIR sub-sm_70 GPU ({caps}): installing torch "
            f"2.3.1+cu118 (last sm_60 line) before any torch import")
        t0 = time.time()
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "torch==2.3.1+cu118", "--index-url",
                 "https://download.pytorch.org/whl/cu118"],
                capture_output=True, text=True, timeout=900)
        except Exception as e:
            log(f"TORCH_REPAIR pip install raised: {e}")
            r = None
        if r is None or r.returncode != 0:
            log("TORCH_REPAIR FAILED: " +
                (r.stderr[-1500:] if r is not None else ""))
            raise RuntimeError(
                f"GPU {caps} needs torch repair but pip install failed")
        log(f"TORCH_REPAIR installed in {time.time()-t0:.0f}s")
        ver = subprocess.check_output(
            [sys.executable, "-c",
             "import torch; print(torch.__version__, torch.version.cuda)"],
            text=True, stderr=subprocess.STDOUT).strip()
        log(f"TORCH_REPAIR now: {ver}")
    else:
        log(f"GPU compute caps {caps} OK for preinstalled torch")
    global SINGLE_GPU
    if not os.environ.get("NATIVE_SINGLE_GPU"):
        # Auto-detect: a 1-GPU worker runs the full model on cuda:0.
        SINGLE_GPU = n_gpus == 1
    need = 1 if SINGLE_GPU else 2
    if n_gpus < need:
        raise RuntimeError(
            f"expected {need} GPUs, got {n_gpus}: {out.strip()}")
    return {
        "gpu_count": n_gpus,
        "nvidia_smi_lines": lines,
        "compute_capabilities": caps,
        "single_gpu_mode": SINGLE_GPU,
        "requested_gpu": "NvidiaTeslaT4",
        "requested_gpu_count": 2,
    }


def log_host_resources(stage: str) -> dict:
    """Log + return disk/RAM state so hard kills are diagnosable."""
    info = {}
    try:
        for mnt in ("/", "/tmp", "/kaggle/working"):
            try:
                t, u, f = shutil.disk_usage(mnt)
                info[mnt] = {"total_gb": round(t / 2**30, 1),
                             "free_gb": round(f / 2**30, 1)}
            except OSError:
                pass
        try:
            meminfo = {}
            for ln in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = ln.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    meminfo[k] = round(int(v.split()[0]) / 2**20, 1)
            info["ram_gb"] = meminfo
        except OSError:
            pass
    finally:
        log(f"RESOURCES[{stage}] {json.dumps(info)}")
    return info


# Out-of-band observability: fire-and-forget log lines to ntfy.sh so hard
# worker kills (which produce ZERO Kaggle output snapshot) are still
# diagnosable.  Best-effort; never blocks or raises.
NTFY_TOPIC = os.environ.get("NATIVE_NTFY", "dsv4-dee-debug-9k2f1")


def _ntfy(msg: str) -> None:
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=msg.encode("utf-8")[:3500],
            headers={"Title": "dee-gen"})
        urllib.request.urlopen(req, timeout=3).read(16)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    global FORCE_TMP, PROFILE_STAGES, SOURCE_READ_LANES
    gpu_environment = check_gpu_allocation()
    res = log_host_resources("startup")
    tmp_free = res.get("/tmp", {}).get("free_gb", 0)

    log("=== clone + checkout ===")
    if os.environ.get("NATIVE_SOURCE_TREE"):
        if not (DEE / "CMakeLists.txt").is_file():
            raise RuntimeError(f"NATIVE_SOURCE_TREE invalid: {ROOT}")
        log(f"reusing driver-provided source tree {ROOT}")
    else:
        if ROOT.exists():
            run(["rm", "-rf", str(ROOT)])
        run(["git", "clone", "--branch", BRANCH, "--single-branch",
             REPO, str(ROOT)])
    if COMMIT:
        run(["git", "-C", str(ROOT), "checkout", "--quiet", COMMIT])
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    log(f"pinned commit {head}")
    apply_run_config()
    # trace_requests serialization lives inside the stage profiler, so the
    # request trace requires profile_stages.  Per the Phase-4 contract we
    # auto-enable it (loudly) rather than fail: the arm wants the event
    # stream, and a silent no-trace run would be worse than measured
    # overhead.  The driver also sets NATIVE_PROFILE=1 explicitly.
    if TRACE_REQUESTS and not PROFILE_STAGES:
        PROFILE_STAGES = True
        log("[p4] NATIVE_TRACE_REQUESTS=1 requires profile_stages but "
            "NATIVE_PROFILE was not '1': auto-enabled PROFILE_STAGES "
            "(stage profiling is now inside the measured wall)")

    # Storage geometry is backend-specific. Safetensors execution may stage
    # the full 153-GiB checkpoint into /tmp only with very large headroom.
    # DEE4 execution instead reads canonical bytes once from the dataset mount
    # and writes a ~137-GiB expert bank to /tmp; duplicating both forms there
    # would waste quota without helping dense/state loads.
    if EXPERT_STORE_BACKEND == "dee4":
        if tmp_free and tmp_free < 160:
            raise RuntimeError(
                f"DEE4 requires at least 160 GiB free in /tmp, found {tmp_free}")
        if DATASET_DIR.is_dir():
            FORCE_TMP = False
            log("DEE4 storage mode: canonical shards stay on dataset mount; "
                "expert-major bank will be written to /tmp")
    elif EXPERT_STORE_BACKEND == "dee4_trace":
        if tmp_free and tmp_free < 45:
            raise RuntimeError(
                f"DEE4 trace requires at least 45 GiB free in /tmp, found {tmp_free}")
        if DATASET_DIR.is_dir():
            FORCE_TMP = False
            log("DEE4 trace storage mode: canonical shards stay on dataset mount; "
                f"sparse expert bank is {DEE4_TRACE_PATH}")
    elif FORCE_TMP and tmp_free and tmp_free < 400:
        log(f"RESOURCES-GATE: /tmp has {tmp_free} GiB (<400); "
            f"disabling FORCE_TMP staging, will use dataset mount/download "
            f"fallbacks as available")
        FORCE_TMP = False

    source_run_config = (
        DEE / "kaggle/deepseek-v4-flash-0731/run_config.json")
    cloned_harness = (
        DEE / "kaggle/deepseek-v4-flash-0731/"
        "deepseek_v4_native_generate.py")
    kernel_metadata = (
        DEE / "kaggle/deepseek-v4-flash-0731/kernel-metadata.json")
    launch_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        driver_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception:
        driver_rows = []
    environment_payload = {
        "recorded_at_utc": launch_utc,
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": gpu_environment,
        "gpu_driver_rows": driver_rows,
        "storage_mount": str(DATASET_DIR),
        "storage_filesystem": "Kaggle dataset mount plus /tmp root overlay",
        "startup_resources": res,
    }
    run_config_payload = {
        "recorded_at_utc": launch_utc,
        "run_id": RUN_ID,
        "arm_id": ARM_ID or None,
        "cache_dtype": CACHE_DTYPE,
        "expert_store": EXPERT_STORE_BACKEND,
        "dee4_trace_path": str(DEE4_TRACE_PATH),
        "dee4_validate_samples": DEE4_VALIDATE_SAMPLES,
        "profile_stages": PROFILE_STAGES,
        "n_tokens": N_TOKENS,
        "cache_budget_bytes_per_gpu": BUDGET_BYTES,
        "cache_budget_gib_per_gpu": BUDGET_BYTES / (1 << 30),
        "host_pack_requested_bytes": [
            HOST_PACK_CACHE_BYTES_GPU0, HOST_PACK_CACHE_BYTES_GPU1],
        "host_pack_runtime_cap_gib_total": float(
            os.environ.get("NATIVE_LRU_TOTAL_CAP_GIB", "17.0")),
        "eviction_policy": EVICTION_POLICY,
        "host_cache_mode": HOST_CACHE_MODE,
        "cache_reset": CACHE_RESET,
        "trace_requests": TRACE_REQUESTS,
        "ignore_eos": IGNORE_EOS,
        "source_read_lanes": SOURCE_READ_LANES,
        "source_read_queue_depth": SOURCE_READ_QUEUE_DEPTH,
        "force_tmp": FORCE_TMP,
        "source_path": str(source_run_config),
        "source_sha256": sha256_file(source_run_config),
    }
    integrity_payload = {
        "recorded_at_utc": launch_utc,
        "run_id": RUN_ID,
        "repository": REPO,
        "branch": BRANCH,
        "git_commit": head,
        "model_revision": REV,
        "executing_harness_sha256": sha256_file(Path(__file__).resolve()),
        "cloned_harness_sha256": sha256_file(cloned_harness),
        "run_config_sha256": sha256_file(source_run_config),
        "kernel_metadata_sha256": sha256_file(kernel_metadata),
        "expected_token_ids": SEALED_TOKEN_IDS,
        "expected_decoded_text": SEALED_DECODED_TEXT,
        "required_gpu": "2x Tesla T4 for performance acceptance",
    }
    write_evidence("environment.json", environment_payload)
    write_evidence("run_config.json", run_config_payload)
    write_evidence("integrity.json", integrity_payload)
    write_evidence("memory.json", {
        "status": "RUNNING",
        "startup_process": process_mem_gib(),
        "startup_system": system_mem_gib(),
        "startup_gpu": {},
    })
    write_evidence("profile.json", {"status": "RUNNING"})
    write_evidence("result.json", {
        "status": "RUNNING", "run_id": RUN_ID, "git_commit": head,
        "model_revision": REV, "started_at_utc": launch_utc})

    # Build with bounded parallelism: single-GPU "medium" workers have ~13 GB
    # RAM (dual-GPU has 32 GB), and nvcc+gcc at -j4 can OOM the worker mid-
    # build (v33 died silently 21 min in with no log = hard kill).  dee_cli
    # is NOT used by this harness (only pydee + the FP4 regression tests),
    # so it is skipped entirely to cut build memory and wall time.
    mem_report("prebuild")
    log("=== build dee_core + FP4 regression tests (sm_60;sm_75, -j2) ===")
    build_jobs = max(1, min(2, os.cpu_count() or 2))
    run(["cmake", "-S", str(DEE), "-B", str(BUILD),
         "-DCMAKE_CUDA_ARCHITECTURES=60;75", "-DDEE_CUDA=ON",
         "-DDEE_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(BUILD), "--target", "dee_core",
         "-j", str(build_jobs)])
    mem_report("post-dee_core")
    for target in ("test_deepseek_v4_fp4_cuda", "test_deepseek_v4_fp4_expert"):
        run(["cmake", "--build", str(BUILD), "--target", target,
             "-j", str(build_jobs)])
        # These tests are the numerical admission gate for the candidate.
        # In particular, test_deepseek_v4_fp4_expert now exercises the exact
        # packed-cache device API used by the full model. A failure must stop
        # before the multi-hour generation, never degrade into a warning.
        run([str(BUILD / target)], cwd=str(DEE))
    mem_report("post-tests")

    log("=== build pydee ===")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--user", "pybind11"])
    run([sys.executable, "pydee/setup.py", "build_ext", "--inplace"],
        env={**os.environ, "DEE_BUILD_DIR": str(BUILD)}, cwd=str(DEE))

    log("=== download all shards ===")
    mem_report("pre-download")
    shard_paths = download_all_shards()
    mem_report("post-download")

    # ── DEE4: component evidence or selected live serving bank ────────
    log(f"=== DEE4 prepare (backend={EXPERT_STORE_BACKEND}) ===")
    dee4_store_path = ""
    dee4_trace_validation = {}
    try:
        if EXPERT_STORE_BACKEND == "dee4_segmented":
            # GPU-2 path: serve from a pre-built dee4-v4-segmented store
            # (46-bucket full universe published as Kaggle datasets).  No
            # repack — point the native store at the assembled directory.
            _seg_root = Path(os.environ.get(
                "NATIVE_DEE4_SEGMENTED_STORE", ""))
            _meta_path = (_seg_root if _seg_root.name == "metadata.json"
                          else _seg_root / "metadata.json")
            if not _meta_path.is_file():
                raise RuntimeError(
                    f"dee4_segmented store metadata missing: {_meta_path}")
            _smeta = json.loads(_meta_path.read_text("utf-8"))
            if _smeta.get("format") != "dee4-v4-segmented":
                raise RuntimeError(
                    f"dee4_segmented store format={_smeta.get('format')!r}")
            _segs = _smeta.get("segments") or []
            _missing = [s["file"] for s in _segs
                        if not (_seg_root / s["file"]).is_file()]
            _wrong = [
                f"{s['file']}:{(_seg_root / s['file']).stat().st_size}"
                for s in _segs
                if (_seg_root / s["file"]).is_file()
                and (_seg_root / s["file"]).stat().st_size
                != int(s["bytes"])]
            if _missing or _wrong:
                raise RuntimeError(
                    f"dee4_segmented store incomplete: "
                    f"missing={_missing[:3]} wrong_size={_wrong[:3]}")
            dee4_store_path = str(_meta_path)
            log(f"DEE4 segmented store armed: {dee4_store_path} "
                f"segments={len(_segs)} "
                f"universe={str(_smeta.get('universe_sha256', ''))[:16]}")
            write_evidence("dee4-segmented-store.json", {
                "format": _smeta["format"],
                "metadata_path": dee4_store_path,
                "n_segments": len(_segs),
                "universe_sha256": _smeta.get("universe_sha256"),
                "verify_segment_hashes": True,
            })
            raise _SkipDee4Prepare()
        sys.path.insert(0, str(DEE / "kaggle" / "deepseek-v4-flash-0731"))
        from repack_to_dee4 import (
            _filesystem_identity as _storage_identity,
            benchmark_dee4_read as _b4r,
            benchmark_dee4_serving_access as _b4s,
            repack,
            repack_trace as _repack_trace,
            validate_dee4_against_safetensors as _validate_dee4,
            validate_dee4_trace_store as _validate_dee4_trace,
        )
        import struct as _struct
        _source_dir = Path(shard_paths[0]).parent
        _idx = _source_dir / "model.safetensors.index.json"
        if not _idx.is_file():
            _idx_url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
                        f"resolve/{REV}/model.safetensors.index.json")
            _idx = WORK / "model.safetensors.index.json"
            log(f"P2.2: downloading index from HF: {_idx_url}")
            _idx_data = urllib.request.urlopen(_idx_url, timeout=300).read()
            _idx.write_bytes(_idx_data)
            log(f"P2.2: index downloaded ({len(_idx_data)} bytes)")
        if EXPERT_STORE_BACKEND == "dee4_trace":
            _dee4_out = (DEE4_TRACE_PATH.parent
                         if DEE4_TRACE_PATH.name == "metadata.json"
                         else DEE4_TRACE_PATH)
            _trace_metadata_path = (
                DEE4_TRACE_PATH if DEE4_TRACE_PATH.name == "metadata.json"
                else _dee4_out / "metadata.json")
            _trace_journal = DEE / V50_TRACE_JOURNAL_RELATIVE
            # A previous bank is evidence, not a cache to regenerate. Validate
            # it in place; a missing metadata file in a non-empty directory is
            # invalid rather than permission to overwrite partial/prior output.
            _trace_created = False
            if _trace_metadata_path.is_file():
                log(f"DEE4 trace bank exists; validating {_trace_metadata_path}")
            else:
                if _dee4_out.exists() and any(_dee4_out.iterdir()):
                    raise RuntimeError(
                        "DEE4 trace metadata is absent from non-empty bank "
                        f"directory; refusing to overwrite {_dee4_out}")
                log("DEE4 trace bank absent; repacking the committed v50 "
                    f"journal into {_dee4_out}")
                _repack_trace(
                    _source_dir, _dee4_out, _trace_journal, index_path=_idx,
                    expected_journal_sha256=V50_TRACE_JOURNAL_SHA256,
                    expected_final_chain_sha256=V50_TRACE_FINAL_CHAIN_SHA256,
                )
                _trace_created = True
            _t0 = time.monotonic()
            dee4_trace_validation = _validate_dee4_trace(
                _trace_metadata_path, _trace_journal,
                expected_journal_sha256=V50_TRACE_JOURNAL_SHA256,
                expected_final_chain_sha256=V50_TRACE_FINAL_CHAIN_SHA256,
            )
            dee4_trace_validation.update({
                "configured_metadata_path": str(_trace_metadata_path),
                "created_this_run": _trace_created,
            })
            _dt = time.monotonic() - _t0
            write_evidence("dee4-trace-validation.json", dee4_trace_validation)
            _trace_metadata = json.loads(
                (_dee4_out / "metadata.json").read_text("utf-8"))
            _dee4_rpt = {
                "record_bytes": int(_trace_metadata["record_bytes"]),
                "total_bytes_repacked": int(_trace_metadata["total_bytes"]),
                "data_sha256": _trace_metadata["data_sha256"],
            }
            _dee4_bench = _b4r(_dee4_out, n_experts=64)
            _dee4_serving_bench = {
                "groups": 0, "topk": 0, "request_count": 0,
                "bytes_requested_per_sweep": 0,
                "record_order_sha256": _trace_metadata["selection_sha256"],
                "winner": None,
                "unavailable_modes": [{
                    "mode": "synthetic-serving-order",
                    "reason": "trace store uses its committed sparse record map",
                }],
            }
        else:
            _dee4_out = (
                DEE4_DIR if EXPERT_STORE_BACKEND == "dee4"
                else Path("/tmp/dsv4-dee4-v2-component")
            )
            _end_layer = 43 if EXPERT_STORE_BACKEND == "dee4" else 3
            _t0 = time.monotonic()
            _dee4_rpt = repack(
                _source_dir, _dee4_out, index_path=_idx,
                start_layer=0, end_layer=_end_layer, dry_run=False)
            _dt = time.monotonic() - _t0
            _dee4_bench = _b4r(_dee4_out, n_experts=64)
            _dee4_serving_bench = _b4s(_dee4_out, groups=8, topk=6)
            (WORK / "dee4-serving-access-benchmark.json").write_text(
                json.dumps(_dee4_serving_bench, indent=2), "utf-8")
        _dee4_validation = _validate_dee4(
            _source_dir, _dee4_out, index_path=_idx,
            sample_count=DEE4_VALIDATE_SAMPLES)
        (WORK / "dee4-import-validation.json").write_text(
            json.dumps(_dee4_validation, indent=2), "utf-8")
        if not _dee4_validation["success"]:
            raise RuntimeError("DEE4 canonical-byte import validation failed")
        for _evidence_name in ("metadata.json", "repack_report.json",
                               "integrity.jsonl"):
            shutil.copy2(_dee4_out / _evidence_name,
                         WORK / f"dee4-{_evidence_name}")

        # The live 137-GiB bank only fits on /tmp. Probe /kaggle/working with
        # the same byte-exact three-layer component bank when its quota permits,
        # then remove it before generation so Kaggle does not snapshot a giant
        # transient output. Failures are evidence, not a reason to discard an
        # otherwise valid live DEE4 run.
        _working_location_benchmark = {
            "status": "NOT_REQUESTED",
            "reason": "live backend is not DEE4",
        }
        if EXPERT_STORE_BACKEND == "dee4":
            _working_probe = WORK / "dsv4-dee4-working-location-probe"
            _working_required = int(_dee4_rpt["record_bytes"]) * 3 * 256
            _working_free = shutil.disk_usage(WORK).free
            _working_location_benchmark = {
                "status": "SKIPPED_CAPACITY",
                "required_data_bytes": _working_required,
                "free_bytes_before": _working_free,
            }
            if _working_free >= _working_required + (1 << 30):
                try:
                    _working_rpt = repack(
                        _source_dir, _working_probe, index_path=_idx,
                        start_layer=0, end_layer=3, dry_run=False)
                    _working_location_benchmark = _b4s(
                        _working_probe, groups=8, topk=6)
                    _working_location_benchmark["status"] = "COMPLETE"
                    _working_location_benchmark["repack_seconds"] = (
                        _working_rpt["total_elapsed_s"])
                except Exception as _working_error:
                    _working_location_benchmark = {
                        "status": "FAILED",
                        "error": repr(_working_error),
                        "required_data_bytes": _working_required,
                        "free_bytes_before": _working_free,
                    }
                finally:
                    shutil.rmtree(_working_probe, ignore_errors=True)
            (WORK / "dee4-working-serving-access-benchmark.json").write_text(
                json.dumps(_working_location_benchmark, indent=2), "utf-8")

        # Compare: safetensors random gather
        _idx_data = json.loads(_idx.read_text("utf-8"))
        _wm = _idx_data["weight_map"]; _hdr = {}; _hdr_len = {}; _sp = {}
        for _sn in sorted(set(_wm.values())):
            _p = _source_dir / _sn
            _sp[_sn] = _p
            with open(_p, "rb") as _f:
                _hl = _struct.unpack("<Q", _f.read(8))[0]
                _hdr_len[_sn] = _hl
                _hdr[_sn] = json.loads(_f.read(_hl))
        _st0 = time.monotonic(); _tb = 0; _rc = 0
        for _L in range(3):
            for _eid in range(min(21, 256)):  # 21 * 3 layers = 63 experts
                for _proj in ["w1","w2","w3"]:
                    for _kind in ["weight","scale"]:
                        _nm = f"layers.{_L}.ffn.experts.{_eid}.{_proj}.{_kind}"
                        if _nm not in _wm: continue
                        _sn = _wm[_nm]; _hh = _hdr[_sn]
                        _off = _hh[_nm]["data_offsets"]
                        _ln = _off[1] - _off[0]
                        with open(_sp[_sn], "rb") as _f:
                            _f.seek(8 + _hdr_len[_sn] + _off[0])
                            if len(_f.read(_ln)) != _ln:
                                raise IOError(f"short scatter read: {_nm}")
                        _tb += _ln; _rc += 1
                if _rc >= 64 * 6: break
            if _rc >= 64 * 6: break
        _ste = time.monotonic() - _st0
        _st_mbps = _tb / max(_ste, 0.001) / (1 << 20)
        _d4_mbps = _dee4_bench["aggregate_mbps"]
        _serving_winner = _dee4_serving_bench.get("winner") or {}
        log(f"P2.2: DEE4 contiguous {_d4_mbps:.0f} MB/s vs "
            f"safetensors scatter {_st_mbps:.0f} MB/s "
            f"({_d4_mbps/max(_st_mbps,0.01):.1f}x) "
            f"repack {_dt:.0f}s {_dee4_rpt['total_bytes_repacked']/(1<<30):.1f}GiB")
        log("P2.2: serving-access winner "
            f"mode={_serving_winner.get('mode', 'none')} "
            f"bandwidth={_serving_winner.get('bandwidth_mib_s', 0):.1f}MiB/s "
            f"p95={_serving_winner.get('p95_latency_ms', 0):.1f}ms "
            f"cold={_serving_winner.get('cold_cache_observed', False)}")
        _p22_evidence = {
            "format": ("dee4-v3-trace" if EXPERT_STORE_BACKEND == "dee4_trace"
                       else "dee4-v2"),
            "serving_backend": EXPERT_STORE_BACKEND,
            "dee4_mbps": _d4_mbps, "safetensors_mbps": _st_mbps,
            "speedup": _d4_mbps / max(_st_mbps, 0.01),
            "repack_s": _dt, "repack_gib": _dee4_rpt["total_bytes_repacked"]/(1<<30),
            "io_count_reduction": f"{_rc} random -> 1 contiguous record stream",
            "data_sha256": _dee4_rpt["data_sha256"],
            "validation_samples": _dee4_validation["sample_count"],
            "validation_source_shards": _dee4_validation["source_shards_covered"],
            "safetensors_storage": _storage_identity(_source_dir),
            "serving_access_benchmark": {
                "groups": _dee4_serving_bench["groups"],
                "topk": _dee4_serving_bench["topk"],
                "request_count": _dee4_serving_bench["request_count"],
                "bytes_requested_per_sweep": (
                    _dee4_serving_bench["bytes_requested_per_sweep"]),
                "record_order_sha256": (
                    _dee4_serving_bench["record_order_sha256"]),
                "winner": _dee4_serving_bench["winner"],
                "unavailable_modes": _dee4_serving_bench["unavailable_modes"],
            },
            "working_location_benchmark": _working_location_benchmark,
        }
        (WORK / "p2.2-dee4-evidence.json").write_text(
            json.dumps(_p22_evidence, indent=2), "utf-8")
        if EXPERT_STORE_BACKEND in {"dee4", "dee4_trace"}:
            # Preserve the configured metadata-file path in result evidence and
            # pass that same exact path to the native store. The native reader
            # supports either a bank directory or its metadata file.
            dee4_store_path = (
                str(_trace_metadata_path)
                if EXPERT_STORE_BACKEND == "dee4_trace" else str(_dee4_out))
            log(f"DEE4_LIVE backend={EXPERT_STORE_BACKEND} path={dee4_store_path} "
                f"identity={_dee4_rpt['data_sha256']}")
        else:
            # Component-only evidence must not occupy runtime disk or be
            # mistaken for the serving backend selected by this run.
            shutil.rmtree(_dee4_out)
    except _SkipDee4Prepare:
        pass
    except Exception as _e:
        log(f"DEE4 prepare failed: {_e}")
        import traceback as _tb
        _tb.print_exc()
        if EXPERT_STORE_BACKEND in {"dee4", "dee4_trace", "dee4_segmented"}:
            raise

    sys.path.insert(0, str(DEE))
    sys.path.insert(0, str(DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/"
                          "official-source/inference"))
    import torch
    if TORCH_DETERMINISTIC:
        # warn_only records (rather than raises on) any op lacking a
        # deterministic implementation — the warning list itself is
        # diagnostic: it names the nondeterministic ops in the hot path.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        log("[det] torch deterministic algorithms armed "
            "(warn_only=True, cudnn deterministic, tf32 off)")
    log(f"cuda devices: {torch.cuda.device_count()}")
    environment_payload.update({
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())],
    })
    write_evidence("environment.json", environment_payload)
    need = 1 if SINGLE_GPU else 2
    if torch.cuda.device_count() < need:
        raise RuntimeError(
            f"expected {need} GPUs, got {torch.cuda.device_count()}")

    from scripts import deepseek_v4_model as vm
    from scripts import deepseek_v4_encoding as enc

    cfg = vm.model_config_from_official(CONFIG)
    tokenizer = enc.load_tokenizer()
    log(f"cfg layers/exps/dim/topk: {cfg.n_layers} {cfg.n_routed} "
        f"{cfg.dim} {cfg.topk}")

    log("=== build native engines (device 0 + 1) ===")
    mem_avail = host_mem_available_gib()
    mem_total = host_mem_total_gib()
    # v15: the v12/v14 "leak" is the host-pack LRU materializing its full
    # 25.74 GiB budget on a box that physically cannot hold it.  Both runs
    # were OOM-killed with the LRU at ~21 GiB plus the ~5-8 GiB resident
    # baseline (torch + 2 CUDA contexts + python + dense host refs), and
    # the old clamp (mem_avail - 3.5 = 26.2 GiB > 25.74) never engaged.
    # Cap the TOTAL LRU budget hard at 17 GiB (8.5 GiB/GPU, below the
    # measured ~21 GiB death point with margin).  v15 also logs MemTotal +
    # per-token VmRSS/VmData/VmLck so v16 can tune the exact ceiling.
    # NATIVE_LRU_TOTAL_CAP_GIB overrides for arms that deliberately probe
    # above the safe default (p5 c8h); the mem_avail clamp still applies.
    LRU_TOTAL_CAP_GIB = float(
        os.environ.get("NATIVE_LRU_TOTAL_CAP_GIB", "17.0"))
    pack_budget0 = HOST_PACK_CACHE_BYTES_GPU0
    pack_budget1 = HOST_PACK_CACHE_BYTES_GPU1
    if mem_avail > 0 or mem_total > 0:
        total_cap = int(LRU_TOTAL_CAP_GIB * (1 << 30))
        if mem_avail > 0:
            total_cap = min(total_cap, int((mem_avail - 3.5) * (1 << 30)))
        # Keep the measured 16.2:12.2 ratio when clamping under total RAM.
        ratio = HOST_PACK_CACHE_BYTES_GPU0 + HOST_PACK_CACHE_BYTES_GPU1
        if total_cap < ratio:
            share0 = HOST_PACK_CACHE_BYTES_GPU0 / max(1, ratio)
            pack_budget0 = max(0, int(total_cap * share0))
            pack_budget1 = max(0, int(total_cap * (1.0 - share0)))
        pack_budget0 = min(pack_budget0, HOST_PACK_CACHE_BYTES_GPU0)
        pack_budget1 = min(pack_budget1, HOST_PACK_CACHE_BYTES_GPU1)
    # host_cache_mode=bypass (kickoff §7): the pack degenerates to a
    # one-record bounce buffer with source lanes forced to 1 -- an honest
    # "host cache off" arm that still avoids the FUSE mmap-fallback path.
    # Enforced harness-side (after the RAM clamp so it wins) because older
    # pydee binaries lack the EngineConfig.host_cache_mode binding;
    # build_native_engine re-enforces it at cfg assembly as well.
    P4_ONE_RECORD_BYTES = 13_369_344  # one packed dee4 expert record
    if HOST_CACHE_MODE == "bypass":
        if pack_budget0 > P4_ONE_RECORD_BYTES \
                or pack_budget1 > P4_ONE_RECORD_BYTES:
            log(f"[p4] host_cache_mode=bypass: clamping host pack budgets "
                f"{pack_budget0}/{pack_budget1} -> {P4_ONE_RECORD_BYTES} "
                f"(exactly one record; bounce buffer, no caching)")
            pack_budget0 = min(pack_budget0, P4_ONE_RECORD_BYTES)
            pack_budget1 = min(pack_budget1, P4_ONE_RECORD_BYTES)
        if SOURCE_READ_LANES != 1:
            log(f"[p4] host_cache_mode=bypass: forcing source_read_lanes "
                f"{SOURCE_READ_LANES} -> 1 (bounce-buffer contract)")
            SOURCE_READ_LANES = 1
    # Loud resolved-knob printout (one line per knob) so a watcher can
    # attribute any later behavior to the exact armed configuration.
    log("=== Phase-4 resolved knobs ===")
    for _knob, _value in (
        ("run_id", RUN_ID),
        ("arm_id", ARM_ID or "none"),
        ("cache_dtype", CACHE_DTYPE),
        ("expert_store", EXPERT_STORE_BACKEND),
        ("eviction_policy", EVICTION_POLICY),
        ("host_cache_mode", HOST_CACHE_MODE),
        ("cache_reset", CACHE_RESET),
        ("trace_requests", TRACE_REQUESTS),
        ("ignore_eos", IGNORE_EOS),
        ("profile_stages", PROFILE_STAGES),
        ("diagnostics", DIAGNOSTICS),
        ("use_batched_experts", USE_BATCHED_EXPERTS),
        ("n_tokens", N_TOKENS),
        ("budget_bytes_per_gpu", BUDGET_BYTES),
        ("host_pack_bytes_gpu0_requested", HOST_PACK_CACHE_BYTES_GPU0),
        ("host_pack_bytes_gpu1_requested", HOST_PACK_CACHE_BYTES_GPU1),
        ("host_pack_bytes_gpu0_effective", pack_budget0),
        ("host_pack_bytes_gpu1_effective", pack_budget1),
        ("source_read_lanes", SOURCE_READ_LANES),
        ("source_read_queue_depth", SOURCE_READ_QUEUE_DEPTH),
        ("single_gpu", SINGLE_GPU),
        ("force_tmp", FORCE_TMP),
        ("mem_avail_gib", round(mem_avail, 2)),
        ("mem_total_gib", round(mem_total, 2)),
    ):
        log(f"[p4cfg] {_knob} = {_value}")
    log(f"budget={BUDGET_BYTES/2**30:.2f}GiB/GPU host_pack="
        f"{pack_budget0/2**30:.2f}/{pack_budget1/2**30:.2f}GiB "
        f"batched={USE_BATCHED_EXPERTS} profile={PROFILE_STAGES} "
        f"diagnostics={DIAGNOSTICS} mem_avail={mem_avail:.1f}GiB "
        f"mem_total={mem_total:.1f}GiB lru_cap={LRU_TOTAL_CAP_GIB}GiB "
        f"cache_dtype={CACHE_DTYPE} source_read_lanes={SOURCE_READ_LANES} "
        f"source_read_queue_depth={SOURCE_READ_QUEUE_DEPTH}")
    # P2.4 single-GPU mode: one engine on cuda:0 carrying the FULL budget
    # (both halves merged), all 43 layers on device0 (split=n_layers), and
    # the same-device handoff path.  eng1 is not built.  Cache budget is
    # capped below the dense+torch baseline (~7 GiB dense + ~1.5 GiB
    # torch/CUDA on a 14.5 GiB-usable 16 GB card) so the run cannot OOM.
    if SINGLE_GPU:
        single_budget = min(int(BUDGET_BYTES * 2),
                            int(4.0 * (1 << 30)))
        single_pack = pack_budget0 + pack_budget1
        eng0 = vm.build_native_engine(
            shard_paths, device_id=0, budget_bytes=single_budget,
            host_pack_cache_bytes=single_pack,
            use_batched_experts=USE_BATCHED_EXPERTS,
            profile_stages=PROFILE_STAGES,
            cache_dtype=CACHE_DTYPE,
            source_read_lanes=SOURCE_READ_LANES,
            source_read_queue_depth=SOURCE_READ_QUEUE_DEPTH,
            expert_store_path=dee4_store_path,
            trace_requests=TRACE_REQUESTS,
            eviction_policy=EVICTION_POLICY,
            host_cache_mode=HOST_CACHE_MODE)
        eng1 = eng0
        log(f"engines built SINGLE_GPU budget={single_budget/2**30:.2f}GiB "
            f"host_pack={single_pack/2**30:.2f}GiB cache_dtype={CACHE_DTYPE}")
    else:
        eng0 = vm.build_native_engine(
            shard_paths, device_id=0, budget_bytes=BUDGET_BYTES,
            host_pack_cache_bytes=pack_budget0,
            use_batched_experts=USE_BATCHED_EXPERTS,
            profile_stages=PROFILE_STAGES,
            cache_dtype=CACHE_DTYPE,
            source_read_lanes=SOURCE_READ_LANES,
            source_read_queue_depth=SOURCE_READ_QUEUE_DEPTH,
            expert_store_path=dee4_store_path,
            trace_requests=TRACE_REQUESTS,
            eviction_policy=EVICTION_POLICY,
            host_cache_mode=HOST_CACHE_MODE)
        eng1 = vm.build_native_engine(
            shard_paths, device_id=1, budget_bytes=BUDGET_BYTES,
            host_pack_cache_bytes=pack_budget1,
            use_batched_experts=USE_BATCHED_EXPERTS,
            profile_stages=PROFILE_STAGES,
            cache_dtype=CACHE_DTYPE,
            source_read_lanes=SOURCE_READ_LANES,
            source_read_queue_depth=SOURCE_READ_QUEUE_DEPTH,
            expert_store_path=dee4_store_path,
            trace_requests=TRACE_REQUESTS,
            eviction_policy=EVICTION_POLICY,
            host_cache_mode=HOST_CACHE_MODE)
        log(f"engines built (cache_dtype={CACHE_DTYPE})")

    # P5b engine-only determinism probe: NATIVE_MICRO_PROBE=1 skips the
    # model build entirely and drives moe_forward_batch_device on eng0 in a
    # suite of configs over fixed inputs — isolating the dee_core expert
    # path from all torch-model state.  The suite runs cold-reset, warm
    # resident-hit, allocator-churn, and a post-churn reset in one process.
    if os.environ.get("NATIVE_MICRO_PROBE", "0") == "1":
        import numpy as _np
        _pn = int(os.environ.get("NATIVE_MICRO_N", "18"))
        _ptopk = int(os.environ.get("NATIVE_MICRO_TOPK", "6"))
        _phid = int(os.environ.get("NATIVE_MICRO_HIDDEN", "4096"))
        _player = int(os.environ.get("NATIVE_MICRO_LAYER", "0"))
        _piters = int(os.environ.get("NATIVE_MICRO_ITERS", "4"))
        # 108 selections over 64 experts: mirrors the real prefill mix —
        # some experts group 2-3 rows, ~44 run the single-row path.
        _pexperts = int(os.environ.get("NATIVE_MICRO_EXPERTS", "64"))
        _suite = json.loads(os.environ.get(
            "NATIVE_MICRO_SUITE",
            '[{"name":"coldreset","reset":true,"churn_mb":0},'
            ' {"name":"resident","reset":false,"churn_mb":0},'
            ' {"name":"churn","reset":true,"churn_mb":256},'
            ' {"name":"postchurn","reset":true,"churn_mb":0}]'))
        log(f"[micro] n={_pn} topk={_ptopk} hidden={_phid} "
            f"layer={_player} iters={_piters} experts={_pexperts} "
            f"suite={[s['name'] for s in _suite]}")
        _g = torch.Generator(device="cpu").manual_seed(1234)
        _h = torch.randn(_pn, _phid, generator=_g,
                         dtype=torch.float32).to("cuda:0").half()
        _h = _h.contiguous()
        _ids = (_np.arange(_pn * _ptopk, dtype=_np.int32)
                .reshape(_pn, _ptopk) % _pexperts).copy()
        _raw = torch.empty(_pn, _ptopk, _phid,
                           dtype=torch.float32, device="cuda:0")
        _results = {}
        for _spec in _suite:
            _name = _spec["name"]
            _shas = []
            for _it in range(_piters):
                if _spec.get("reset"):
                    eng0.reset_runtime_cache()
                    eng0.clear_host_cache()
                    eng0.reset_store_stats()
                if _spec.get("churn_mb", 0) > 0:
                    _junk = torch.randn(_spec["churn_mb"] * 256, _phid,
                                        device="cuda:0")
                    _ = (_junk @ _junk.t()).sum().item()
                    del _junk
                torch.cuda.current_stream(0).synchronize()
                _ok = bool(eng0.moe_forward_batch_device(
                    _player, _h.data_ptr(), _pn, _ids, _ptopk,
                    _raw.data_ptr()))
                _arr = _raw.detach().cpu().numpy()
                _sha = hashlib.sha256(_arr.tobytes()).hexdigest()
                _shas.append(_sha)
                _np.save(str(WORK / f"micro_raw-{_name}-i{_it}.npy"), _arr)
                log(f"[micro] {_name} iter={_it} ok={_ok} "
                    f"raw_sha={_sha[:16]}")
                if not _ok:
                    log(f"[micro] engine error: "
                        f"{eng0.last_error_message()}")
                    break
            _results[_name] = {"raw_shas": _shas,
                               "distinct": len(set(_shas))}
        (WORK / "micro_probe.json").write_text(json.dumps({
            "config": {"n": _pn, "topk": _ptopk, "hidden": _phid,
                       "layer": _player, "iters": _piters,
                       "experts": _pexperts,
                       "cuda_launch_blocking":
                           os.environ.get("CUDA_LAUNCH_BLOCKING", "0")},
            "results": _results}, indent=2))
        log(f"[micro] done: " + ", ".join(
            f"{k}={v['distinct']}sha" for k, v in _results.items()))
        return 0

    # Phase-5 cohort mode: NATIVE_COHORT_JSON carries {"groups": [[i,...]]}
    # — indices into NATIVE_PROMPTS_JSON forming lockstep cohorts.  Each
    # group runs one generate_cohort call; rows are left-padded to the
    # group max token length (scalar start_pos contract) and padding is
    # recorded per row.  With the env absent, max_batch=1 and behavior is
    # identical to the Phase-4 path.  max_batch must be resolved BEFORE
    # build_candidate sizes the kv_cache rows.
    # Multi-prompt mode: NATIVE_PROMPTS_JSON carries a JSON list of prompt
    # strings; engines + model are built ONCE and every prompt runs through
    # the same process (one store open+seal per engine, not per prompt).
    # With the env absent, behavior is identical: one prompt, suffix "".
    # Parsed BEFORE the model build so cohort-group validation can fail
    # fast instead of wasting the ~40-min build.
    _prompts_json = os.environ.get("NATIVE_PROMPTS_JSON", "")
    try:
        PROMPT_LIST = (json.loads(_prompts_json) if _prompts_json else None)
    except Exception:
        raise RuntimeError(
            f"NATIVE_PROMPTS_JSON unparsable: {_prompts_json[:200]!r}")
    if not PROMPT_LIST:
        PROMPT_LIST = [CANONICAL_PROMPT]
    _multi = len(PROMPT_LIST) > 1

    _cohorts_json = os.environ.get("NATIVE_COHORT_JSON", "")
    try:
        _cohorts_parsed = json.loads(_cohorts_json) if _cohorts_json else None
    except Exception as exc:
        raise RuntimeError(
            f"NATIVE_COHORT_JSON unparsable (fail-closed; a malformed "
            f"cohort spec must not silently fall back to sequential "
            f"mode): {_cohorts_json[:200]!r}") from exc
    COHORT_GROUPS = (_cohorts_parsed.get("groups")
                     if _cohorts_parsed else None)
    COHORT_PAD_TO = (_cohorts_parsed.get("pad_to")
                     if _cohorts_parsed else None)
    # PAD_TOKEN_ID (deepseek_v4_encoding.py:59); BOS=0 would prepend
    # extra BOS tokens instead of inert pads.
    COHORT_PAD_TOKEN = int(os.environ.get("NATIVE_COHORT_PAD_TOKEN", "1"))
    COHORT_MAX_BATCH = (max((len(g) for g in COHORT_GROUPS), default=1)
                        if COHORT_GROUPS else 1)
    if _cohorts_json and not COHORT_GROUPS:
        raise RuntimeError(
            "NATIVE_COHORT_JSON parsed but produced no groups "
            "(fail-closed; refusing silent downgrade to sequential)")
    if COHORT_GROUPS:
        _seen = set()
        for _g in COHORT_GROUPS:
            if not _g or max(_g) >= len(PROMPT_LIST) or min(_g) < 0:
                raise RuntimeError(
                    f"NATIVE_COHORT_JSON group {_g} indexes outside "
                    f"PROMPT_LIST (n={len(PROMPT_LIST)})")
            _dupes = set(_g) & _seen
            if len(set(_g)) != len(_g) or _dupes:
                raise RuntimeError(
                    f"NATIVE_COHORT_JSON duplicate prompt index in "
                    f"{_g} (seen={sorted(_dupes)})")
            _seen.update(_g)
        log(f"=== cohort mode: {len(COHORT_GROUPS)} groups, "
            f"max_batch={COHORT_MAX_BATCH}, pad_token={COHORT_PAD_TOKEN} ===")

    log("=== build full model (native FFN) ===")
    t0 = time.monotonic()
    # The tensor source must read dense tensors (embed/head/norm/attention/
    # router/shared) from whichever directory actually holds the shards:
    # the dataset mount when attached, or the local /tmp download otherwise.
    shards_dir = Path(shard_paths[0]).parent
    source = vm.LocalDirTensorSource(HEADERS_DIR, shards_dir)
    provider = vm.ExpertProvider(source)
    # Reference-FFN arm (P5b): FFN_BACKEND=cache_fp16 builds the
    # DeepseekV4CacheFfn path — routed experts computed by torch GEMMs
    # over FP16 payloads, bypassing moe_forward_batch_device entirely.
    # It needs a real python-side cache + loader per device.
    cache0 = loader0 = cache1 = loader1 = None
    if FFN_BACKEND != "native":
        from scripts import deepseek_v4_cache as v4cache
        cache0 = v4cache.DeepSeekExpertCache(
            REF_CACHE_BYTES, device="cuda:0", eviction_policy="lru")
        loader0 = v4cache.DeepSeekExpertLoader(cache0)
        dev1 = "cuda:0" if SINGLE_GPU else "cuda:1"
        if dev1 == "cuda:0":
            cache1, loader1 = cache0, loader0
        else:
            cache1 = v4cache.DeepSeekExpertCache(
                REF_CACHE_BYTES, device=dev1, eviction_policy="lru")
            loader1 = v4cache.DeepSeekExpertLoader(cache1)
        log(f"[ffn] reference cache_fp16 backend armed, "
            f"cache={REF_CACHE_BYTES / (1 << 20):.0f} MiB/device")
    if SINGLE_GPU:
        model = vm.DeepseekV4Model.build_candidate(
            cfg, source, device0="cuda:0", device1="cuda:0",
            cache0=cache0, loader0=loader0, cache1=cache1,
            loader1=loader1,
            provider=provider, ffn_backend=FFN_BACKEND,
            engine0=eng0, engine1=eng1, split=cfg.n_layers,
            diagnostics=DIAGNOSTICS, profile_stages=PROFILE_STAGES,
            max_batch=COHORT_MAX_BATCH)
    else:
        model = vm.DeepseekV4Model.build_candidate(
            cfg, source, device0="cuda:0", device1="cuda:1",
            cache0=cache0, loader0=loader0, cache1=cache1,
            loader1=loader1,
            provider=provider, ffn_backend=FFN_BACKEND,
            engine0=eng0, engine1=eng1, diagnostics=DIAGNOSTICS,
            profile_stages=PROFILE_STAGES,
            max_batch=COHORT_MAX_BATCH)
    model.reset_state()
    build_s = time.monotonic() - t0
    log(f"model build {build_s:.1f}s")

    def _run_prompt(prompt_text: str, qi: int):
        global CANONICAL_PROMPT, SEAL_APPLICABLE
        CANONICAL_PROMPT = prompt_text
        SEAL_APPLICABLE = prompt_text == SEALED_PROMPT
        # Driver-invoked runs (NATIVE_PROMPTS_JSON set) always suffix,
        # even for single-prompt arms like the p5 c0_anchor -- the
        # session driver harvests per-stem filenames.
        suffix = f"-q{qi}" if (_multi or _prompts_json) else ""
        # P5b probe tagging: layer-0 dump files are named by unit so the
        # post-hoc diff can align raw tensors across sequential units.
        os.environ["NATIVE_PROBE_TAG"] = f"q{qi}"
        os.environ["NATIVE_PROBE_DIR"] = str(WORK)
        log(f"=== prompt {qi}: {len(prompt_text)} chars, "
            f"seal_applicable={SEAL_APPLICABLE} ===")
        log("=== tokenize + greedy decode ===")
        ids = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([ids], device="cuda:0").long()
        decode_ms: list[float] = []
        _p4_engines = (
            (("cuda0", eng0),) if SINGLE_GPU
            else (("cuda0", eng0), ("cuda1", eng1))
        )
        # NATIVE_CACHE_RESET=cold: per-prompt cold start.  In addition to the
        # measurement reset below, evict the VRAM arena + host pack + fp4
        # staging metadata and re-zero the ExpertStore counters.  The pinned
        # bindings (clear_host_cache/reset_store_stats) are being added by a
        # parallel workstream -> getattr fallback + loud log under old bins.
        # Runs for EVERY prompt (including q0) so the semantics are uniform.
        # NOTE: the OS page cache is NOT cleared -- evicted segment pages may
        # still re-fault quickly on local-disk profiles (moot on the Kaggle
        # FUSE mount, which shows ~0.2% mincore residency).
        if CACHE_RESET == "cold":
            for _ck, _ce in _p4_engines:
                if not _ce.reset_runtime_cache():
                    raise RuntimeError(
                        f"{_ck} reset_runtime_cache failed (cold reset): "
                        f"{_ce.last_error_message() or 'no native diagnostic'}")
                _clear_host = getattr(_ce, "clear_host_cache", None)
                if _clear_host is None:
                    log(f"[p4] {_ck}: pydee binary lacks "
                        "clear_host_cache(); host pack + fp4 staging "
                        "metadata NOT cleared (old binary)")
                else:
                    _clear_host()
                _reset_store = getattr(_ce, "reset_store_stats", None)
                if _reset_store is None:
                    log(f"[p4] {_ck}: pydee binary lacks "
                        "reset_store_stats(); ExpertStore counters stay "
                        "process-cumulative (old binary)")
                else:
                    _reset_store()
            log(f"[p4] NATIVE_CACHE_RESET=cold: VRAM arena + host pack + "
                f"fp4 staging + store stats reset before prompt {qi}; "
                f"OS page cache NOT cleared")
        if not eng0.reset_external_profile():
            raise RuntimeError(
                "cuda0 external-profile reset failed before measured generation: "
                f"{eng0.last_error_message() or 'no native diagnostic'}"
            )
        if not SINGLE_GPU and not eng1.reset_external_profile():
            raise RuntimeError(
                "cuda1 external-profile reset failed before measured generation: "
                f"{eng1.last_error_message() or 'no native diagnostic'}"
            )
        # Prompt-scoped counter baseline.  host_pack_stats() /
        # expert_store_stats() return PROCESS-CUMULATIVE counters that no
        # reset touches, while previous_totals re-zeros per prompt -- so
        # without a baseline the first checkpoint row of every prompt
        # re-attributes all prior prompts' counters to step 0.  Snapshot
        # AFTER the resets: reset_external_profile() already zeroes the
        # prefetcher/cache stats feeding engine_stats (so those baselines
        # come out 0, matching the reset semantics), and the cold-reset
        # above has already cleared the pack/store counters when armed.
        prompt_start_counters = {"host_pack": {}, "expert_store": {},
                                 "engine_stats": {}}
        try:
            # Key set mirrors the result sections ("cuda0","cuda1" always;
            # eng1 aliases eng0 in single-GPU mode so its delta computes
            # identically rather than falling back to zero-baseline).
            for _ck, _ce in (("cuda0", eng0), ("cuda1", eng1)):
                prompt_start_counters["host_pack"][_ck] = dict(
                    _ce.host_pack_stats())
                prompt_start_counters["expert_store"][_ck] = dict(
                    _ce.expert_store_stats())
                prompt_start_counters["engine_stats"][_ck] = json.loads(
                    _ce.last_stats_json())
        except Exception as _snap_exc:
            log(f"[p4] prompt-start counter snapshot failed: {_snap_exc!r}")
        # Token attribution (kickoff §6.3): RequestTraceRecord.token is
        # stamped from Engine.current_token_ at request-emission time, so
        # the value must be armed BEFORE each forward's layer loop.  It is
        # sticky and reset_external_profile() sets it back to -1, so we
        # re-arm it here (prefill = journal forward_step 0) and advance it
        # inside _token_checkpoint (which runs after each forward) so the
        # NEXT forward's records carry its own step (decode = 1..N-1,
        # matching the route journal's forward_step).
        def _set_forward_token(step: int) -> None:
            for _ck, _ce in _p4_engines:
                try:
                    _ce.set_external_token(int(step))
                except Exception as _tok_exc:
                    log(f"[p4] {_ck} set_external_token({step}) failed: "
                        f"{_tok_exc!r}")
        _set_forward_token(0)
        # arm_config{suffix}.json: the fully RESOLVED effective config for
        # this arm+prompt (kickoff §8).  Written BEFORE generation so it
        # survives a mid-run kill, then integrity-hashed with the other
        # artifacts.  arm_id is metadata only -- run_id stays constant
        # across arms so route journals hash-compare.
        _engine_rc = {}
        try:
            _engine_rc = {
                _ck: _ce.runtime_config() for _ck, _ce in _p4_engines}
        except Exception as _rc_exc:
            log(f"[p4] engine runtime_config snapshot failed: {_rc_exc!r}")
        arm_config_payload = {
            "recorded_at_utc": launch_utc,
            "schema": "phase4-arm-config/v1",
            "arm_id": ARM_ID or None,
            "run_id": RUN_ID,
            "git_commit": head,
            "prompt_index": qi,
            "prompt_sha256": hashlib.sha256(
                prompt_text.encode("utf-8")).hexdigest(),
            "resolved": {
                "cache_dtype": CACHE_DTYPE,
                "expert_store": EXPERT_STORE_BACKEND,
                "expert_store_path": dee4_store_path,
                "eviction_policy": EVICTION_POLICY,
                "host_cache_mode": HOST_CACHE_MODE,
                "cache_reset": CACHE_RESET,
                "trace_requests": TRACE_REQUESTS,
                "ignore_eos": IGNORE_EOS,
                "profile_stages": PROFILE_STAGES,
                "diagnostics": DIAGNOSTICS,
                "use_batched_experts": USE_BATCHED_EXPERTS,
                "n_tokens": N_TOKENS,
                "budget_bytes_per_gpu": BUDGET_BYTES,
                "host_pack_cache_bytes_requested": [
                    HOST_PACK_CACHE_BYTES_GPU0,
                    HOST_PACK_CACHE_BYTES_GPU1],
                "host_pack_cache_bytes_effective": [
                    pack_budget0, pack_budget1],
                "source_read_lanes": SOURCE_READ_LANES,
                "source_read_queue_depth": SOURCE_READ_QUEUE_DEPTH,
                "single_gpu": SINGLE_GPU,
                "force_tmp": FORCE_TMP,
                "dee4_validate_samples": DEE4_VALIDATE_SAMPLES,
                "dee4_store_skip_seal": os.environ.get(
                    "DEE4_STORE_SKIP_SEAL", "0") == "1",
                "device_split": getattr(model, "split", None),
                "n_prompts": len(PROMPT_LIST),
            },
            "engine_runtime_config": _engine_rc,
            "native_env": {
                k: v for k, v in sorted(os.environ.items())
                if k.startswith(("NATIVE_", "DEE_", "DEE4_"))},
        }
        write_evidence(f"arm_config{suffix}.json", arm_config_payload)
        # v12: checkpoint every generated token to /kaggle/working so an OOM kill
        # (v9/v11 lost ALL tokens) still leaves the exact token stream + timing.
        # The checkpoint file format is a JSONL of per-token records; the final
        # RESULT block below mirrors the old single-JSON shape.
        CHECKPOINT = WORK / f"generated_checkpoint{suffix}.jsonl"
        cp_handle = open(CHECKPOINT, "w", encoding="utf-8")
        ROUTE_JOURNAL_PATH = WORK / f"routed_experts{suffix}.jsonl"
        route_journal = RoutedExpertJournal(
            ROUTE_JOURNAL_PATH, run_id=RUN_ID, n_layers=cfg.n_layers,
            topk=cfg.topk)
        weight_jf = (open(WORK / f"route_weights{suffix}.jsonl", "w",
                          encoding="utf-8")
                     if ROUTE_WEIGHT_JOURNAL else None)
        capture_jf = (open(WORK / f"captures{suffix}.jsonl", "w",
                           encoding="utf-8")
                      if CAPTURE_JOURNAL else None)
        _captures = {} if CAPTURE_JOURNAL else None
        _step_captures = ([{} for _ in range(N_TOKENS)]
                          if CAPTURE_JOURNAL else None)
        route_step = 0
        route_start_pos = 0

        def _route_checkpoint(layer_id: int) -> None:
            """Persist the exact CPU route buffer already consumed by native."""
            nonlocal route_step, route_start_pos
            layer = model.layer(int(layer_id))
            ids_host = getattr(
                layer.ffn_fn, "_native_route_ids_host", None)
            if ids_host is None:
                # Reference-FFN backend has no native pinned route buffer —
                # last_route already holds the exact ids the FFN consumed.
                ids_host = (layer.ffn_fn.last_route or {}).get("expert_ids")
            if ids_host is None:
                raise RuntimeError(
                    f"route buffer unavailable after layer {layer_id}")
            if bool(getattr(ids_host, "is_cuda", False)):
                raise RuntimeError(
                    f"route journal refuses a device read at layer {layer_id}")
            route_journal.append_layer(
                step=route_step, start_pos=route_start_pos,
                layer=int(layer_id), device=str(layer.device),
                expert_ids=ids_host)
            if weight_jf is not None:
                weight_jf.write(_weight_journal_rec(
                    layer, step=route_step, start_pos=route_start_pos))
                weight_jf.flush()
            if capture_jf is not None:
                cap_src = (_captures if route_step == 0
                           else _step_captures[min(route_step,
                                                   len(_step_captures) - 1)])
                cap = (cap_src or {}).get(int(layer_id)) or {}
                rec = {"step": int(route_step),
                       "start_pos": int(route_start_pos),
                       "layer": int(layer_id)}
                for key in ("moe_out", "shared_out", "router_scores",
                            "expert_ids", "routing_weights"):
                    t = cap.get(key)
                    rec[f"{key}_sha256"] = (
                        hashlib.sha256(
                            t.detach().cpu().numpy().tobytes()
                        ).hexdigest() if torch.is_tensor(t) else None)
                capture_jf.write(json.dumps(rec) + "\n")
                capture_jf.flush()
            if int(layer_id) == cfg.n_layers - 1:
                route_step += 1
                route_start_pos = len(ids) + route_step - 1

        def _token_checkpoint(step: int, tok: int) -> None:
            # This hook fires AFTER forward `step` completed; arm the token
            # index for the NEXT forward (journal forward_step = step+1)
            # before any of its requests can be emitted.
            _set_forward_token(step + 1)
            mem = host_mem_available_gib()
            rec = {"step": step, "token_id": int(tok),
                   "elapsed_s": round(time.monotonic() - t0, 2),
                   "host_mem_available_gib": round(mem, 2)}
            # This link is admitted only after the same forward's layer 42 row
            # has been flushed and fsynced.  A token checkpoint can therefore
            # never claim a partially journaled route.
            rec["route_journal"] = route_journal.checkpoint_link(step)
            # v15 diagnostics: process + system memory breakdown and engine
            # cache counters, so a v12-style OOM is attributable to a component
            # (heap vs pinned vs page cache) instead of a mystery "leak".
            rec["proc"] = process_mem_gib()
            rec["sys"] = system_mem_gib()
            try:
                rec["host_pack0"] = eng0.host_pack_stats()
                rec["host_pack1"] = eng1.host_pack_stats()
                _checkpoint_engines = (
                    (("cuda0", eng0),) if SINGLE_GPU
                    else (("cuda0", eng0), ("cuda1", eng1))
                )
                rec["engine_stats"] = {
                    key: json.loads(engine.last_stats_json())
                    for key, engine in _checkpoint_engines
                }
                rec["expert_store"] = {
                    key: engine.expert_store_stats()
                    for key, engine in _checkpoint_engines
                }
                rec["host_pack"] = {
                    key: engine.host_pack_stats()
                    for key, engine in _checkpoint_engines
                }
            except Exception:
                pass
            try:
                rec["bridge"] = model.bridge_counters()
            except Exception:
                pass
            cp_handle.write(json.dumps(rec) + "\n")
            cp_handle.flush()
            os.fsync(cp_handle.fileno())
            if step % 4 == 0 or mem < 4.0:
                log(f"[tok {step}] id={int(tok)} elapsed={rec['elapsed_s']}s "
                    f"mem_avail={rec['host_mem_available_gib']}GiB")
                pm = rec.get("proc", {})
                hp0 = rec.get("host_pack0", {})
                log(f"[tok {step}] proc={pm} hp0_bytes_gib="
                    f"{hp0.get('bytes', 0) / (1 << 30):.1f} "
                    f"hp0_entries={hp0.get('entries', 0)} "
                    f"hp0_evict={hp0.get('evictions', 0)}")

        t0 = time.monotonic()
        try:
            toks = model.generate(
                input_ids, max_new_tokens=N_TOKENS,
                eos_id=(-1 if IGNORE_EOS else 1),
                captures=_captures,
                per_step_captures=_step_captures,
                decode_timings_ms=decode_ms,
                post_step_hook=_token_checkpoint,
                post_layer_hook=_route_checkpoint)
            wall_s = time.monotonic() - t0
        finally:
            route_journal.close()
            if weight_jf is not None:
                weight_jf.close()
            if capture_jf is not None:
                capture_jf.close()
            cp_handle.close()
        log(f"decode done in {wall_s:.1f}s, {len(toks)} tokens")

        prefill_ms = decode_ms[0] if decode_ms else 0.0
        decode_only = decode_ms[1:]
        decode_sum_s = sum(decode_only) / 1000.0
        decode_tok_s = (len(decode_only) / decode_sum_s
                        if decode_sum_s > 0 else float("inf"))
        lat = sorted(decode_only)
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
        text = tokenizer.decode(toks)

        result = {
            "run_id": RUN_ID,
            "arm_id": ARM_ID or None,
            "commit": head,
            "host_mem_available_gib": round(mem_avail, 2),
            "host_pack_budget_gib": [round(pack_budget0 / (1 << 30), 2),
                                      round(pack_budget1 / (1 << 30), 2)],
            "model_revision": REV,
            "gpu_environment": gpu_environment,
            "cache_dtype": CACHE_DTYPE,
            "expert_store_backend": EXPERT_STORE_BACKEND,
            "expert_store_path": dee4_store_path,
            "prompt": prompt_text,
            "prompt_len": len(ids),
            "n_tokens": N_TOKENS,
            "generated_token_ids": toks,
            "decoded_text": text,
            "decoded_fragments": [tokenizer.decode([t]) for t in toks],
            "build_seconds": round(build_s, 2),
            "total_wall_seconds": round(wall_s, 2),
            "prefill_ms": round(prefill_ms, 2),
            "prefill_tokens": len(ids),
            "prefill_tok_s": round(len(ids) / (prefill_ms / 1000.0), 3)
            if prefill_ms > 0 else None,
            "decode_wall_s": round(decode_sum_s, 3),
            "decode_tokens": len(decode_only),
            "decode_tok_s": round(decode_tok_s, 3),
            "inter_token_latency_ms": {
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "max": round(max(decode_only), 2) if decode_only else None,
                "median": round(float(p50), 2),
            },
            "decode_timings_ms": [round(t, 2) for t in decode_only],
            "gpu_memory": gpu_memory_snapshot(),
            "diagnostics": DIAGNOSTICS,
            "bridge_counters": model.bridge_counters(),
            "layer_count_executed": int(
                model.last_execution.get("layers_executed", -1)),
            "execution_terminal": dict(model.last_execution),
            "route_journal": route_journal.summary(),
            "dee4_trace_validation": dee4_trace_validation,
            "eviction_policy": EVICTION_POLICY,
            "host_cache_mode": HOST_CACHE_MODE,
            "cache_reset": CACHE_RESET,
            "trace_requests": TRACE_REQUESTS,
            "ignore_eos": IGNORE_EOS,
        }

        # Stage 0 instrumentation: per-engine expert-cache + host-pack + stage
        # profile dumps so every run reports WHERE the wall went.
        try:
            result["engine_stats"] = {
                "cuda0": json.loads(eng0.last_stats_json()),
                "cuda1": json.loads(eng1.last_stats_json()),
            }
            result["engine_config"] = {
                "cuda0": eng0.runtime_config(),
                "cuda1": eng1.runtime_config(),
            }
            result["host_pack"] = {
                "cuda0": eng0.host_pack_stats(),
                "cuda1": eng1.host_pack_stats(),
            }
            result["expert_store"] = {
                "cuda0": eng0.expert_store_stats(),
                "cuda1": eng1.expert_store_stats(),
            }
            result["stage_profile"] = {
                "cuda0": json.loads(eng0.external_profile_json(wall_s * 1000.0)),
                "cuda1": json.loads(eng1.external_profile_json(wall_s * 1000.0)),
            }
            result["model_cuda_stage_profile"] = model.cuda_stage_profile()
        except Exception as exc:  # never fail the run over instrumentation
            log(f"instrumentation dump failed: {exc}")
            result["instrumentation_error"] = repr(exc)
        try:
            result["model_runtime_snapshot"] = model.runtime_snapshot()
        except Exception as exc:
            log(f"runtime snapshot failed: {exc}")
            result["runtime_snapshot_error"] = repr(exc)

        # cache_events{suffix}.jsonl: durable per-request event sink for the
        # Phase-4 telemetry contract.  When trace_requests is on, the stage
        # profiler's "trace" array (one RequestTraceRecord per expert
        # request) is mirrored here -- one JSON object per line:
        # {engine, device, seq, <record fields>}.  The file is integrity-
        # hashed below and retrieved per arm by the session driver.
        cache_events_name = f"cache_events{suffix}.jsonl"
        cache_events_meta = {
            "enabled": bool(TRACE_REQUESTS),
            "artifact": cache_events_name if TRACE_REQUESTS else None,
            "records": 0,
        }
        if TRACE_REQUESTS:
            try:
                _seq = 0
                _ce_path = WORK / cache_events_name
                with _ce_path.open("w", encoding="utf-8") as _cef:
                    for _ck, _ce in _p4_engines:
                        _dev = (result.get("engine_config", {})
                                .get(_ck, {}).get("device_id", _ck))
                        _trace = (result.get("stage_profile", {})
                                  .get(_ck, {}) or {}).get("trace") or []
                        for _rec in _trace:
                            _cef.write(json.dumps(
                                {"engine": _ck, "device": _dev,
                                 "seq": _seq, **_rec},
                                separators=(",", ":")) + "\n")
                            _seq += 1
                cache_events_meta.update({
                    "records": _seq,
                    "bytes": _ce_path.stat().st_size,
                    "sha256": sha256_file(_ce_path),
                })
                log(f"[p4] cache_events{suffix}.jsonl: {_seq} records "
                    f"({_ce_path.stat().st_size} bytes)")
            except Exception as _ce_exc:
                log(f"[p4] cache_events dump failed: {_ce_exc!r}")
                cache_events_meta["error"] = repr(_ce_exc)
        result["cache_events"] = cache_events_meta

        # Prompt-scoped counter deltas: the top-level host_pack /
        # expert_store / engine_stats sections stay PROCESS-CUMULATIVE
        # (classify_full_generation's store gates read cumulative
        # source_reads/lookup_failures), so per-prompt views live here
        # alongside the cumulative copies under a cumulative_ prefix.
        result["cumulative_counters_at_prompt_start"] = prompt_start_counters
        result["cumulative_host_pack"] = result.get("host_pack", {})
        result["cumulative_expert_store"] = result.get("expert_store", {})
        result["cumulative_engine_stats"] = result.get("engine_stats", {})
        _end_counters = {
            "host_pack": result.get("host_pack", {}),
            "expert_store": result.get("expert_store", {}),
            "engine_stats": result.get("engine_stats", {}),
        }
        _prompt_scoped = {}
        for _section in ("host_pack", "expert_store", "engine_stats"):
            _prompt_scoped[_section] = {}
            for _ck, _end_vals in _end_counters[_section].items():
                _start_vals = (
                    prompt_start_counters.get(_section, {}).get(_ck, {}))
                _delta = {}
                for _f, _v in (_end_vals or {}).items():
                    if (isinstance(_v, (int, float))
                            and isinstance(_start_vals.get(_f), (int, float))):
                        _delta[_f] = _v - _start_vals[_f]
                    else:
                        _delta[_f] = _v
                _prompt_scoped[_section][_ck] = _delta
        result["prompt_scoped_counters"] = _prompt_scoped

        classification, gates, performance_eligible = classify_full_generation(result)
        completed_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.update({
            "status": "COMPLETE",
            "completed_at_utc": completed_utc,
            "classification": classification,
            "performance_eligible": performance_eligible,
            "hardware_classification": (
                "ELIGIBLE_2X_TESLA_T4" if performance_eligible
                else "REJECT_HARDWARE_FOR_PERFORMANCE"),
            "correctness": {
                "sealed_contract_gates": gates,
                "all_non_hardware_gates_pass": all(
                    value for key, value in gates.items()
                    if key != "required_performance_hardware"),
            },
        })

        # Derive physical byte/token accounting directly from the live serving
        # backends. In single-GPU mode cuda1 aliases cuda0 and must not be counted
        # twice.  The per-token values are PROMPT-SCOPED deltas (process-
        # cumulative counters minus the prompt-start snapshot); the raw
        # cumulative values are kept under a cumulative_ prefix.
        store_keys = ("cuda0",) if SINGLE_GPU else ("cuda0", "cuda1")
        stores = result.get("expert_store", {})
        cum_storage_bytes = sum(
            int(stores.get(key, {}).get("bytes_requested", 0))
            for key in store_keys)
        cum_source_reads = sum(
            int(stores.get(key, {}).get("source_reads", 0))
            for key in store_keys)
        cum_h2d_bytes = sum(
            int(result.get("engine_stats", {}).get(key, {}).get("h2d_bytes", 0))
            for key in store_keys)
        _psc_store = _prompt_scoped.get("expert_store", {})
        _psc_eng = _prompt_scoped.get("engine_stats", {})
        storage_bytes = sum(
            int(_psc_store.get(key, {}).get("bytes_requested", 0))
            for key in store_keys)
        source_reads = sum(
            int(_psc_store.get(key, {}).get("source_reads", 0))
            for key in store_keys)
        result["byte_accounting"] = {
            "storage_bytes_total": storage_bytes,
            "storage_bytes_per_generated_token": (
                storage_bytes / len(toks) if toks else None),
            "storage_requests_total": source_reads,
            "storage_requests_per_generated_token": (
                source_reads / len(toks) if toks else None),
            "expert_h2d_bytes_total": sum(
                int(_psc_eng.get(key, {}).get("h2d_bytes", 0))
                for key in store_keys),
            "cumulative_storage_bytes_total": cum_storage_bytes,
            "cumulative_storage_requests_total": cum_source_reads,
            "cumulative_expert_h2d_bytes_total": cum_h2d_bytes,
            "scope": ("prompt_scoped deltas from cumulative engine "
                      "counters minus the prompt-start snapshot"),
        }

        min_host_available = None
        checkpoint_records = 0
        checkpoint_rows = []
        try:
            for line in CHECKPOINT.read_text("utf-8").splitlines():
                record = json.loads(line)
                checkpoint_rows.append(record)
                available = float(record["host_mem_available_gib"])
                min_host_available = (
                    available if min_host_available is None
                    else min(min_host_available, available))
                checkpoint_records += 1
        except Exception as exc:
            log(f"checkpoint memory summary failed: {exc}")

        def _checkpoint_total(row: dict, section: str, field: str) -> float:
            return sum(
                float(values.get(field, 0))
                for values in row.get(section, {}).values()
            )

        per_token_accounting = []
        # Seed previous_totals from the prompt-start cumulative snapshot so
        # the FIRST checkpoint row's delta is this prompt's own traffic, not
        # the process-wide sum of every earlier prompt (the old re-zero
        # silently re-attributed all prior counters to step 0).  Sum over
        # the same key set the checkpoint rows use (_p4_engines: cuda0 only
        # in single-GPU mode) so the baseline never double-counts the
        # cuda0/cuda1 alias.
        _p4_baseline_keys = tuple(key for key, _ in _p4_engines)
        def _baseline_total(section: str, field: str) -> float:
            return sum(
                float(prompt_start_counters.get(section, {})
                      .get(key, {}).get(field, 0))
                for key in _p4_baseline_keys
            )
        previous_totals = {
            "storage_bytes": _baseline_total(
                "expert_store", "bytes_requested"),
            "storage_requests": _baseline_total(
                "expert_store", "source_reads"),
            "source_read_wall_ms": _baseline_total(
                "expert_store", "read_milliseconds"),
            "h2d_bytes": _baseline_total("engine_stats", "h2d_bytes"),
            "h2d_copies": _baseline_total("engine_stats", "h2d_copies"),
            "resident_hits": _baseline_total("engine_stats", "resident_hits"),
            "cold_loads": _baseline_total("engine_stats", "cold_loads"),
            "evictions": _baseline_total("engine_stats", "evictions"),
            "host_pack_hits": _baseline_total("host_pack", "hits"),
            "host_pack_misses": _baseline_total("host_pack", "misses"),
        }
        for index, row in enumerate(checkpoint_rows):
            totals = {
                "storage_bytes": _checkpoint_total(
                    row, "expert_store", "bytes_requested"),
                "storage_requests": _checkpoint_total(
                    row, "expert_store", "source_reads"),
                "source_read_wall_ms": _checkpoint_total(
                    row, "expert_store", "read_milliseconds"),
                "h2d_bytes": _checkpoint_total(row, "engine_stats", "h2d_bytes"),
                "h2d_copies": _checkpoint_total(row, "engine_stats", "h2d_copies"),
                "resident_hits": _checkpoint_total(
                    row, "engine_stats", "resident_hits"),
                "cold_loads": _checkpoint_total(row, "engine_stats", "cold_loads"),
                "evictions": _checkpoint_total(row, "engine_stats", "evictions"),
                "host_pack_hits": _checkpoint_total(row, "host_pack", "hits"),
                "host_pack_misses": _checkpoint_total(row, "host_pack", "misses"),
            }
            deltas = {
                key: max(0.0, value - previous_totals[key])
                for key, value in totals.items()
            }
            previous_totals = totals
            timing_ms = (
                float(prefill_ms) if index == 0
                else float(decode_only[index - 1])
                if index - 1 < len(decode_only) else None
            )
            per_token_accounting.append({
                "step": int(row.get("step", index)),
                "phase": "prefill" if index == 0 else "decode",
                "token_id": int(row.get("token_id", -1)),
                "wall_ms": round(timing_ms, 3) if timing_ms is not None else None,
                **{
                    key: int(value) if key != "source_read_wall_ms"
                    else round(value, 3)
                    for key, value in deltas.items()
                },
                "resident_experts": int(_checkpoint_total(
                    row, "engine_stats", "resident_experts")),
            })
        result["per_token_accounting"] = per_token_accounting

        # Prompt-scoped read wall: cumulative read_milliseconds minus the
        # prompt-start snapshot, matching the per-prompt stage_profile
        # windows the h2d/compute terms already use.
        storage_read_ms = sum(
            float(_psc_store.get(key, {}).get("read_milliseconds", 0))
            for key in store_keys)
        h2d_gpu_ms = sum(
            float(result.get("stage_profile", {}).get(key, {})
                  .get("gpu_ms", {}).get("h2d", 0))
            for key in store_keys)
        compute_gpu_ms = sum(
            float(result.get("stage_profile", {}).get(key, {})
                  .get("derived", {}).get("total_gpu_compute_ms", 0))
            for key in store_keys)
        generated_count = len(toks)
        result["measured_roofline"] = {
            "scope": "whole generation amortized over emitted tokens",
            "storage": {
                "bytes_per_emitted_token": (
                    storage_bytes / generated_count if generated_count else None),
                "observed_source_read_bytes_per_second": (
                    storage_bytes / (storage_read_ms / 1000.0)
                    if storage_read_ms > 0 else None),
                "roof_tokens_per_second": (
                    generated_count / (storage_read_ms / 1000.0)
                    if storage_read_ms > 0 else None),
            },
            "pcie_h2d": {
                "bytes_per_emitted_token": (
                    result["byte_accounting"]["expert_h2d_bytes_total"]
                    / generated_count if generated_count else None),
                "observed_bytes_per_second": (
                    result["byte_accounting"]["expert_h2d_bytes_total"]
                    / (h2d_gpu_ms / 1000.0) if h2d_gpu_ms > 0 else None),
                "roof_tokens_per_second": (
                    generated_count / (h2d_gpu_ms / 1000.0)
                    if h2d_gpu_ms > 0 else None),
            },
            "routed_compute": {
                "measured_gpu_ms_per_emitted_token": (
                    compute_gpu_ms / generated_count if generated_count else None),
                "roof_tokens_per_second": (
                    generated_count / (compute_gpu_ms / 1000.0)
                    if compute_gpu_ms > 0 else None),
            },
            "vram_weight_reads": {
                "bytes_per_emitted_token": None,
                "reason": "kernel-level global traffic is not measured by this run",
            },
        }

        profile_payload = {
            "status": "COMPLETE",
            "classification": classification,
            "profile_stages_enabled": PROFILE_STAGES,
            "build_seconds": result["build_seconds"],
            "total_wall_seconds": result["total_wall_seconds"],
            "prefill_ms": result["prefill_ms"],
            "decode_wall_s": result["decode_wall_s"],
            "decode_timings_ms": result["decode_timings_ms"],
            "inter_token_latency_ms": result["inter_token_latency_ms"],
            "stage_profile": result.get("stage_profile", {}),
            "model_cuda_stage_profile": result.get("model_cuda_stage_profile", {}),
            "engine_stats": result.get("engine_stats", {}),
            "expert_store": result.get("expert_store", {}),
            "dee4_trace_validation": result.get("dee4_trace_validation", {}),
            "host_pack": result.get("host_pack", {}),
            "byte_accounting": result["byte_accounting"],
            "per_token_accounting": result["per_token_accounting"],
            "measured_roofline": result["measured_roofline"],
        }
        memory_payload = {
            "status": "COMPLETE",
            "classification": classification,
            "process_final_and_peak_gib": process_mem_gib(),
            "system_final_gib": system_mem_gib(),
            "gpu_final_and_peak_gib": result["gpu_memory"],
            "minimum_checkpoint_host_mem_available_gib": min_host_available,
            "checkpoint_records": checkpoint_records,
            "cache_budget_bytes_per_gpu": BUDGET_BYTES,
            "host_pack_budget_bytes": [pack_budget0, pack_budget1],
        }

        # Publish the five non-integrity artifacts first, then bind their exact
        # serialized bytes from integrity.json. This avoids a circular hash while
        # making the evidence package independently verifiable.
        write_evidence(f"environment{suffix}.json", environment_payload)
        write_evidence(f"run_config{suffix}.json", run_config_payload)
        write_evidence(f"profile{suffix}.json", profile_payload)
        write_evidence(f"memory{suffix}.json", memory_payload)
        write_evidence(f"result{suffix}.json", result)
        _artifact_names = [
            f"environment{suffix}.json", f"run_config{suffix}.json",
            f"result{suffix}.json", f"profile{suffix}.json",
            f"memory{suffix}.json", f"routed_experts{suffix}.jsonl",
            f"arm_config{suffix}.json",
        ]
        # cache_events is a trace-only artifact; hash it only when written.
        if (WORK / cache_events_name).is_file():
            _artifact_names.append(cache_events_name)
        if (WORK / f"route_weights{suffix}.jsonl").is_file():
            _artifact_names.append(f"route_weights{suffix}.jsonl")
        if (WORK / f"captures{suffix}.jsonl").is_file():
            _artifact_names.append(f"captures{suffix}.jsonl")
        integrity_payload.update({
            "completed_at_utc": completed_utc,
            "classification": classification,
            "performance_eligible": performance_eligible,
            "arm_id": ARM_ID or None,
            "actual_token_ids": [int(token) for token in toks],
            "actual_token_ids_sha256": hashlib.sha256(
                json.dumps([int(token) for token in toks])
                .encode("utf-8")).hexdigest(),
            "actual_decoded_text_sha256": hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            "sealed_contract_gates": gates,
            "expert_store": result.get("expert_store", {}),
            "dee4_trace_validation": result.get("dee4_trace_validation", {}),
            "cache_events": result.get("cache_events", {}),
            "artifact_sha256": {
                name: sha256_file(WORK / name)
                for name in _artifact_names
            },
        })
        if EXPERT_STORE_BACKEND == "dee4_trace":
            integrity_payload["artifact_sha256"]["dee4-trace-validation.json"] = (
                sha256_file(WORK / "dee4-trace-validation.json"))
        write_evidence(f"integrity{suffix}.json", integrity_payload)
        log("RESULT " + json.dumps(result))
        (WORK / f"native-generate-result{suffix}.json").write_text(
            json.dumps(result, indent=2))

        # Clean up the local download only; the dataset mount is read-only and
        # must not be touched (unlink would raise PermissionError there).
        if not _multi and not (DATASET_DIR.is_dir() and Path(shard_paths[0]).parent == DATASET_DIR):
            for p in shard_paths:
                Path(p).unlink(missing_ok=True)
        log(f"=== VERDICT: {classification}; performance_eligible="
            f"{performance_eligible} ===")
        return {"prompt_index": qi, "classification": classification, "result": result}

    def _run_cohort(group: list[int], ci: int):
        """Lockstep K-cohort generation (Phase-5 / dee-serve v0).

        Same evidence discipline as _run_prompt, cohort-scoped: one
        K-row route journal, one cohort checkpoint (journal link bound
        once per forward then fanned out), per-row checkpoint/result
        artifacts, cohort-level counters + dedup statistics.
        """
        global SEAL_APPLICABLE
        # Cohort inputs are padded groups, never the sealed canonical
        # prompt -- the seal gates are inapplicable here and would force
        # REJECT_NUMERICAL on every cohort if left set from module load.
        SEAL_APPLICABLE = False
        _kdir = str(DEE / "kaggle" / "deepseek-v4-flash-0731")
        if _kdir not in sys.path:
            sys.path.insert(0, _kdir)
        import phase5_serve_driver as p5drv
        prompt_texts = [PROMPT_LIST[i] for i in group]
        suffix = f"-c{ci}"
        k = len(group)
        log(f"=== cohort {ci}: K={k} prompt_indices={group} ===")
        _p4_engines = (
            (("cuda0", eng0),) if SINGLE_GPU
            else (("cuda0", eng0), ("cuda1", eng1))
        )
        if CACHE_RESET == "cold":
            for _ck, _ce in _p4_engines:
                if not _ce.reset_runtime_cache():
                    raise RuntimeError(
                        f"{_ck} reset_runtime_cache failed (cold reset): "
                        f"{_ce.last_error_message() or 'no native diagnostic'}")
                _clear_host = getattr(_ce, "clear_host_cache", None)
                if _clear_host is None:
                    log(f"[p5] {_ck}: pydee binary lacks "
                        "clear_host_cache(); host pack + fp4 staging "
                        "metadata NOT cleared (old binary)")
                else:
                    _clear_host()
                _reset_store = getattr(_ce, "reset_store_stats", None)
                if _reset_store is not None:
                    _reset_store()
            log(f"[p5] NATIVE_CACHE_RESET=cold: caches + store stats reset "
                f"before cohort {ci}")
        if not eng0.reset_external_profile():
            raise RuntimeError(
                "cuda0 external-profile reset failed before cohort: "
                f"{eng0.last_error_message() or 'no native diagnostic'}")
        if not SINGLE_GPU and not eng1.reset_external_profile():
            raise RuntimeError(
                "cuda1 external-profile reset failed before cohort: "
                f"{eng1.last_error_message() or 'no native diagnostic'}")
        mem_avail = host_mem_available_gib()
        prompt_start_counters: dict[str, dict[str, dict]] = {
            "host_pack": {}, "expert_store": {}, "engine_stats": {}}
        try:
            for _ck, _ce in _p4_engines:
                prompt_start_counters["host_pack"][_ck] = dict(
                    _ce.host_pack_stats())
                prompt_start_counters["expert_store"][_ck] = dict(
                    _ce.expert_store_stats())
                prompt_start_counters["engine_stats"][_ck] = json.loads(
                    _ce.last_stats_json())
        except Exception as _snap_exc:
            log(f"[p5] cohort-start counter snapshot failed: {_snap_exc!r}")

        # Prebuild padded ids so L* is known before the first forward (the
        # route hook derives start_pos from it).  pad_to="max" pads every
        # group to the workload-global token length so the K=1 reference
        # arm sees byte-identical padded inputs.
        if COHORT_PAD_TO == "max":
            _min_len = max(len(tokenizer.encode(t)) for t in PROMPT_LIST)
        elif COHORT_PAD_TO is not None:
            _min_len = int(COHORT_PAD_TO)
        else:
            _min_len = 0
        ids_list, pad_meta = p5drv.build_cohort_ids(
            tokenizer.encode, prompt_texts, COHORT_PAD_TOKEN,
            min_len=_min_len)
        lstar = len(ids_list[0])
        log(f"[p5] cohort {ci}: L*={lstar} pads={[m['pad_tokens'] for m in pad_meta]}")

        def _set_forward_token(step: int) -> None:
            for _ck, _ce in _p4_engines:
                try:
                    _ce.set_external_token(int(step))
                except Exception as _tok_exc:
                    log(f"[p5] {_ck} set_external_token({step}) failed: "
                        f"{_tok_exc!r}")
        _set_forward_token(0)

        launch_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _engine_rc = {}
        try:
            _engine_rc = {
                _ck: _ce.runtime_config() for _ck, _ce in _p4_engines}
        except Exception as _rc_exc:
            log(f"[p5] engine runtime_config snapshot failed: {_rc_exc!r}")
        arm_config_payload = {
            "recorded_at_utc": launch_utc,
            "schema": "phase5-cohort-config/v1",
            "arm_id": ARM_ID or None,
            "run_id": RUN_ID,
            "git_commit": head,
            "cohort_id": ci,
            "cohort_k": k,
            "prompt_indices": group,
            "cohort_pad_token": COHORT_PAD_TOKEN,
            "cohort_pad_to": COHORT_PAD_TO,
            "cohort_prompt_len": lstar,
            "prompt_sha256": [hashlib.sha256(t.encode("utf-8")).hexdigest()
                              for t in prompt_texts],
            "resolved": {
                "cache_dtype": CACHE_DTYPE,
                "expert_store": EXPERT_STORE_BACKEND,
                "expert_store_path": dee4_store_path,
                "eviction_policy": EVICTION_POLICY,
                "host_cache_mode": HOST_CACHE_MODE,
                "cache_reset": CACHE_RESET,
                "trace_requests": TRACE_REQUESTS,
                "ignore_eos": IGNORE_EOS,
                "profile_stages": PROFILE_STAGES,
                "diagnostics": DIAGNOSTICS,
                "use_batched_experts": USE_BATCHED_EXPERTS,
                "n_tokens": N_TOKENS,
                "budget_bytes_per_gpu": BUDGET_BYTES,
                "host_pack_cache_bytes_requested": [
                    HOST_PACK_CACHE_BYTES_GPU0,
                    HOST_PACK_CACHE_BYTES_GPU1],
                "host_pack_cache_bytes_effective": [
                    pack_budget0, pack_budget1],
                "source_read_lanes": SOURCE_READ_LANES,
                "source_read_queue_depth": SOURCE_READ_QUEUE_DEPTH,
                "single_gpu": SINGLE_GPU,
                "device_split": getattr(model, "split", None),
                "cohort_mode": True,
                "max_batch": COHORT_MAX_BATCH,
            },
            "engine_runtime_config": _engine_rc,
            "native_env": {
                k2: v for k2, v in sorted(os.environ.items())
                if k2.startswith(("NATIVE_", "DEE_", "DEE4_"))},
        }
        write_evidence(f"arm_config{suffix}.json", arm_config_payload)

        ROUTE_JOURNAL_PATH = WORK / f"routed_experts{suffix}.jsonl"
        route_journal = RoutedExpertJournal(
            ROUTE_JOURNAL_PATH, run_id=RUN_ID, n_layers=cfg.n_layers,
            topk=cfg.topk)
        weight_jf = (open(WORK / f"route_weights{suffix}.jsonl", "w",
                          encoding="utf-8")
                     if ROUTE_WEIGHT_JOURNAL else None)
        route_step = 0
        route_start_pos = 0

        def _route_checkpoint(layer_id: int) -> None:
            nonlocal route_step, route_start_pos
            layer = model.layer(int(layer_id))
            ids_host = getattr(
                layer.ffn_fn, "_native_route_ids_host", None)
            if ids_host is None:
                # Reference-FFN backend: last_route holds the exact ids
                # the FFN consumed (no native pinned buffer exists).
                ids_host = (layer.ffn_fn.last_route or {}).get("expert_ids")
            if ids_host is None:
                raise RuntimeError(
                    f"route buffer unavailable after layer {layer_id}")
            if bool(getattr(ids_host, "is_cuda", False)):
                raise RuntimeError(
                    f"route journal refuses a device read at layer {layer_id}")
            route_journal.append_layer(
                step=route_step, start_pos=route_start_pos,
                layer=int(layer_id), device=str(layer.device),
                expert_ids=ids_host)
            if weight_jf is not None:
                weight_jf.write(_weight_journal_rec(
                    layer, step=route_step, start_pos=route_start_pos))
                weight_jf.flush()
            if int(layer_id) == cfg.n_layers - 1:
                route_step += 1
                route_start_pos = lstar + route_step - 1

        cohort_cp_path = WORK / f"generated_checkpoint{suffix}.jsonl"
        cohort_cp = open(cohort_cp_path, "w", encoding="utf-8")
        row_handles = [
            open(WORK / f"generated_checkpoint{suffix}-r{r}.jsonl", "w",
                 encoding="utf-8")
            for r in range(k)]
        c_t0 = time.monotonic()

        def _cohort_step(cid: int, step: int, toks: list[int]) -> None:
            # Arm the NEXT forward's token attribution, then bind this
            # forward's journal link once and fan it into every row record.
            _set_forward_token(step + 1)
            link = route_journal.checkpoint_link(step)
            mem = host_mem_available_gib()
            rec = {"step": step, "token_ids": [int(t) for t in toks],
                   "elapsed_s": round(time.monotonic() - c_t0, 2),
                   "host_mem_available_gib": round(mem, 2),
                   "route_journal": link,
                   "proc": process_mem_gib(),
                   "sys": system_mem_gib()}
            try:
                rec["engine_stats"] = {
                    key: json.loads(engine.last_stats_json())
                    for key, engine in _p4_engines}
                rec["expert_store"] = {
                    key: engine.expert_store_stats()
                    for key, engine in _p4_engines}
                rec["host_pack"] = {
                    key: engine.host_pack_stats()
                    for key, engine in _p4_engines}
            except Exception:
                pass
            cohort_cp.write(json.dumps(rec) + "\n")
            cohort_cp.flush()
            os.fsync(cohort_cp.fileno())
            row_rec = {"step": step, "elapsed_s": rec["elapsed_s"],
                       "host_mem_available_gib": rec["host_mem_available_gib"],
                       "route_journal": link}
            for r, h in enumerate(row_handles):
                h.write(json.dumps({**row_rec, "row": r,
                                    "token_id": int(toks[r])}) + "\n")
                h.flush()
                os.fsync(h.fileno())
            if step % 4 == 0 or mem < 4.0:
                log(f"[c{ci} step {step}] ids={toks} "
                    f"elapsed={rec['elapsed_s']}s "
                    f"mem_avail={rec['host_mem_available_gib']}GiB")

        def _counters_snapshot(cid: int) -> dict:
            snap: dict[str, Any] = {}
            try:
                snap["host_pack"] = {
                    key: engine.host_pack_stats()
                    for key, engine in _p4_engines}
                snap["expert_store"] = {
                    key: engine.expert_store_stats()
                    for key, engine in _p4_engines}
                snap["engine_stats"] = {
                    key: json.loads(engine.last_stats_json())
                    for key, engine in _p4_engines}
            except Exception as exc:
                snap["error"] = repr(exc)
            return snap

        try:
            drv = p5drv.ServeDriver(
                model, tokenizer.encode, tokenizer.decode, WORK, RUN_ID,
                pad_token_id=COHORT_PAD_TOKEN,
                on_cohort_step=_cohort_step,
                post_layer_hook=_route_checkpoint,
                on_cohort_counters=_counters_snapshot)
        except Exception:
            route_journal.close()
            if weight_jf is not None:
                weight_jf.close()
            cohort_cp.close()
            for h in row_handles:
                h.close()
            raise
        try:
            res = drv.run_cohort(
                prompt_texts, ci, group, N_TOKENS,
                eos_id=(-1 if IGNORE_EOS else 1),
                prebuilt=(ids_list, pad_meta))
        finally:
            route_journal.close()
            if weight_jf is not None:
                weight_jf.close()
            cohort_cp.close()
            for h in row_handles:
                h.close()
        wall_s = res.wall_seconds
        log(f"[p5] cohort {ci} done in {wall_s:.1f}s, K={k} x "
            f"{len(res.rows[0].token_ids)} tokens")

        # Dedup: unique experts staged per forward vs the K*topk request
        # slots — the direct cross-request sharing measure.
        try:
            _jr = [json.loads(ln) for ln in
                   ROUTE_JOURNAL_PATH.read_text("utf-8").splitlines() if ln.strip()]
            _jrecs = [r for r in _jr if r.get("kind") == "layer_route"
                      or "expert_ids_rank_order" in r]
            res.dedup = p5drv.dedup_stats_from_journal(_jrecs)
        except Exception as _dd_exc:
            log(f"[p5] dedup stats failed: {_dd_exc!r}")
            res.dedup = {"error": repr(_dd_exc)}

        decode_ms = res.decode_timings_ms
        prefill_ms = decode_ms[0] if decode_ms else 0.0
        decode_only = decode_ms[1:]
        n_forwards = len(decode_ms)
        emitted = sum(len(r.token_ids) for r in res.rows)
        journal_summary = route_journal.summary()
        res.route_journal = journal_summary  # lands in cohort-c{ci}.json
        result = {
            "run_id": RUN_ID,
            "arm_id": ARM_ID or None,
            "commit": head,
            "mode": "cohort",
            "cohort_id": ci,
            "cohort_k": k,
            "cohort_prompt_len": lstar,
            "cohort_pad_token": COHORT_PAD_TOKEN,
            "prompt_indices": group,
            "prompts": prompt_texts,
            "host_mem_available_gib": round(mem_avail, 2),
            "host_pack_budget_gib": [round(pack_budget0 / (1 << 30), 2),
                                     round(pack_budget1 / (1 << 30), 2)],
            "model_revision": REV,
            "gpu_environment": gpu_environment,
            "cache_dtype": CACHE_DTYPE,
            "expert_store_backend": EXPERT_STORE_BACKEND,
            "expert_store_path": dee4_store_path,
            "n_tokens": N_TOKENS,
            "n_forward_steps": n_forwards,
            "emitted_tokens": emitted,
            "generated_token_ids": res.rows[0].token_ids,
            # classify_full_generation gates on these sequential-path
            # fields; row 0's stream is representative (all rows run the
            # same fixed length under ignore_eos).
            "decoded_text": res.rows[0].decoded_text,
            "layer_count_executed": int(
                model.last_execution.get("layers_executed", -1)),
            "execution_terminal": dict(model.last_execution),
            "trace_requests": TRACE_REQUESTS,
            "dee4_trace_validation": dee4_trace_validation,
            "rows": [{
                "row": r.row, "prompt_index": r.prompt_index,
                "pad_tokens": r.pad_tokens,
                "token_ids_sha256": r.token_ids_sha256,
                "n_tokens": len(r.token_ids),
                "decoded_text": r.decoded_text,
            } for r in res.rows],
            "build_seconds": round(build_s, 2),
            "total_wall_seconds": round(wall_s, 2),
            "prefill_ms": round(prefill_ms, 2),
            "decode_wall_s": round(sum(decode_only) / 1000.0, 3),
            "decode_timings_ms": [round(t, 2) for t in decode_only],
            "gpu_memory": gpu_memory_snapshot(),
            "diagnostics": DIAGNOSTICS,
            "bridge_counters": model.bridge_counters(),
            "route_journal": journal_summary,
            "dedup": res.dedup,
            "cohort_counters": res.counters,
            "eviction_policy": EVICTION_POLICY,
            "host_cache_mode": HOST_CACHE_MODE,
            "cache_reset": CACHE_RESET,
            "ignore_eos": IGNORE_EOS,
        }
        try:
            result["engine_stats"] = {
                "cuda0": json.loads(eng0.last_stats_json()),
                "cuda1": json.loads(eng1.last_stats_json()),
            }
            result["engine_config"] = {
                "cuda0": eng0.runtime_config(),
                "cuda1": eng1.runtime_config(),
            }
            result["host_pack"] = {
                "cuda0": eng0.host_pack_stats(),
                "cuda1": eng1.host_pack_stats(),
            }
            result["expert_store"] = {
                "cuda0": eng0.expert_store_stats(),
                "cuda1": eng1.expert_store_stats(),
            }
            result["stage_profile"] = {
                "cuda0": json.loads(eng0.external_profile_json(wall_s * 1000.0)),
                "cuda1": json.loads(eng1.external_profile_json(wall_s * 1000.0)),
            }
            result["model_cuda_stage_profile"] = model.cuda_stage_profile()
        except Exception as exc:
            log(f"instrumentation dump failed: {exc}")
            result["instrumentation_error"] = repr(exc)
        try:
            result["model_runtime_snapshot"] = model.runtime_snapshot()
        except Exception as exc:
            result["runtime_snapshot_error"] = repr(exc)

        # Scoped counter deltas (same shape as _run_prompt's
        # prompt_scoped_counters; cohort-scoped).
        result["cumulative_counters_at_cohort_start"] = prompt_start_counters
        _end_counters = {
            "host_pack": result.get("host_pack", {}),
            "expert_store": result.get("expert_store", {}),
            "engine_stats": result.get("engine_stats", {}),
        }
        _psc = {}
        for _section in ("host_pack", "expert_store", "engine_stats"):
            _psc[_section] = {}
            for _ck, _end_vals in _end_counters[_section].items():
                _start_vals = (
                    prompt_start_counters.get(_section, {}).get(_ck, {}))
                _delta = {}
                for _f, _v in (_end_vals or {}).items():
                    if (isinstance(_v, (int, float))
                            and isinstance(_start_vals.get(_f), (int, float))):
                        _delta[_f] = _v - _start_vals[_f]
                    else:
                        _delta[_f] = _v
                _psc[_section][_ck] = _delta
        result["cohort_scoped_counters"] = _psc

        # cache_events{suffix}.jsonl: durable per-request event sink --
        # same telemetry contract as _run_prompt (one RequestTraceRecord
        # per expert request, mirrored from the stage profiler's trace).
        cache_events_name = f"cache_events{suffix}.jsonl"
        cache_events_meta = {
            "enabled": bool(TRACE_REQUESTS),
            "artifact": cache_events_name if TRACE_REQUESTS else None,
            "records": 0,
        }
        if TRACE_REQUESTS:
            try:
                _seq = 0
                _ce_path = WORK / cache_events_name
                with _ce_path.open("w", encoding="utf-8") as _cef:
                    for _ck, _ce in _p4_engines:
                        _dev = (result.get("engine_config", {})
                                .get(_ck, {}).get("device_id", _ck))
                        _trace = (result.get("stage_profile", {})
                                  .get(_ck, {}) or {}).get("trace") or []
                        for _rec in _trace:
                            _cef.write(json.dumps(
                                {"engine": _ck, "device": _dev,
                                 "seq": _seq, **_rec},
                                separators=(",", ":")) + "\n")
                            _seq += 1
                cache_events_meta.update({
                    "records": _seq,
                    "bytes": _ce_path.stat().st_size,
                    "sha256": sha256_file(_ce_path),
                })
                log(f"[p5] cache_events{suffix}.jsonl: {_seq} records "
                    f"({_ce_path.stat().st_size} bytes)")
            except Exception as _ce_exc:
                log(f"[p5] cache_events dump failed: {_ce_exc!r}")
                cache_events_meta["error"] = repr(_ce_exc)
        result["cache_events"] = cache_events_meta

        classification, gates, performance_eligible = classify_full_generation(
            result)
        completed_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.update({
            "status": "COMPLETE",
            "completed_at_utc": completed_utc,
            "classification": classification,
            "performance_eligible": performance_eligible,
            "correctness": {
                "sealed_contract_gates": gates,
                "all_non_hardware_gates_pass": all(
                    value for key, value in gates.items()
                    if key != "required_performance_hardware"),
            },
        })

        store_keys = ("cuda0",) if SINGLE_GPU else ("cuda0", "cuda1")
        _psc_store = _psc.get("expert_store", {})
        _psc_eng = _psc.get("engine_stats", {})
        storage_bytes = sum(
            int(_psc_store.get(key, {}).get("bytes_requested", 0))
            for key in store_keys)
        source_reads = sum(
            int(_psc_store.get(key, {}).get("source_reads", 0))
            for key in store_keys)
        result["byte_accounting"] = {
            "storage_bytes_total": storage_bytes,
            "storage_bytes_per_emitted_token": (
                storage_bytes / emitted if emitted else None),
            "storage_requests_total": source_reads,
            "storage_requests_per_emitted_token": (
                source_reads / emitted if emitted else None),
            "expert_h2d_bytes_total": sum(
                int(_psc_eng.get(key, {}).get("h2d_bytes", 0))
                for key in store_keys),
            "scope": ("cohort-scoped deltas; per-emitted-token amortizes "
                      "over K*N tokens — cross-request sharing shows here"),
        }

        profile_payload = {
            "status": "COMPLETE",
            "classification": classification,
            "mode": "cohort",
            "cohort_id": ci, "cohort_k": k,
            "build_seconds": result["build_seconds"],
            "total_wall_seconds": result["total_wall_seconds"],
            "prefill_ms": result["prefill_ms"],
            "decode_wall_s": result["decode_wall_s"],
            "decode_timings_ms": result["decode_timings_ms"],
            "stage_profile": result.get("stage_profile", {}),
            "engine_stats": result.get("engine_stats", {}),
            "expert_store": result.get("expert_store", {}),
            "host_pack": result.get("host_pack", {}),
            "dedup": res.dedup,
            "byte_accounting": result["byte_accounting"],
        }
        memory_payload = {
            "status": "COMPLETE",
            "classification": classification,
            "mode": "cohort",
            "cohort_id": ci, "cohort_k": k,
            "process_final_and_peak_gib": process_mem_gib(),
            "system_final_gib": system_mem_gib(),
            "gpu_final_and_peak_gib": result["gpu_memory"],
            "cache_budget_bytes_per_gpu": BUDGET_BYTES,
            "host_pack_budget_bytes": [pack_budget0, pack_budget1],
        }
        write_evidence(f"environment{suffix}.json", environment_payload)
        write_evidence(f"run_config{suffix}.json", run_config_payload)
        write_evidence(f"profile{suffix}.json", profile_payload)
        write_evidence(f"memory{suffix}.json", memory_payload)
        write_evidence(f"result{suffix}.json", result)
        # Row artifacts + cohort summary land BEFORE the integrity seal so
        # artifact_sha256 covers every evidence file, not just the core set.
        drv.write_row_artifacts(res)
        drv.write_cohort_summary(res)
        _artifact_names = [
            f"environment{suffix}.json", f"run_config{suffix}.json",
            f"result{suffix}.json", f"profile{suffix}.json",
            f"memory{suffix}.json", f"routed_experts{suffix}.jsonl",
            f"arm_config{suffix}.json",
            f"generated_checkpoint{suffix}.jsonl",
            f"cohort{suffix}.json",
        ] + [f"generated_checkpoint{suffix}-r{r}.jsonl" for r in range(k)
             ] + [f"result{suffix}-r{r}.json" for r in range(k)]
        if (WORK / cache_events_name).is_file():
            _artifact_names.append(cache_events_name)
        if (WORK / f"route_weights{suffix}.jsonl").is_file():
            _artifact_names.append(f"route_weights{suffix}.jsonl")
        integrity_payload = {
            "schema": "phase5-cohort-integrity/v1",
            "recorded_at_utc": launch_utc,
            "completed_at_utc": completed_utc,
            "classification": classification,
            "performance_eligible": performance_eligible,
            "arm_id": ARM_ID or None,
            "run_id": RUN_ID,
            "repository": REPO,
            "branch": BRANCH,
            "git_commit": head,
            "model_revision": REV,
            "executing_harness_sha256": sha256_file(
                Path(__file__).resolve()),
            "cloned_harness_sha256": sha256_file(cloned_harness),
            "run_config_sha256": sha256_file(source_run_config),
            "kernel_metadata_sha256": sha256_file(kernel_metadata),
            "cohort_id": ci,
            "cohort_k": k,
            "cohort_prompt_len": lstar,
            "rows": [{
                "row": r.row, "prompt_index": r.prompt_index,
                "pad_tokens": r.pad_tokens,
                "actual_token_ids_sha256": p5drv.token_ids_sha(
                    r.token_ids),
                "n_tokens": len(r.token_ids),
            } for r in res.rows],
            "sealed_contract_gates": gates,
            "expert_store": result.get("expert_store", {}),
            "artifact_sha256": {
                name: sha256_file(WORK / name)
                for name in _artifact_names
            },
        }
        write_evidence(f"integrity{suffix}.json", integrity_payload)
        (WORK / f"native-generate-result{suffix}.json").write_text(
            json.dumps({"cohort_id": ci, "k": k,
                        "classification": classification,
                        "performance_eligible": performance_eligible},
                       indent=2))
        log("RESULT " + json.dumps(
            {"cohort_id": ci, "k": k, "classification": classification,
             "wall_s": wall_s, "dedup": res.dedup}))
        log(f"=== VERDICT cohort {ci}: {classification} ===")
        return {"cohort_id": ci, "k": k, "prompt_indices": group,
                "classification": classification, "result": result,
                "row_shas": {r.prompt_index: r.token_ids_sha256
                             for r in res.rows}}

    _all_results = []
    if COHORT_GROUPS:
        for _ci, _grp in enumerate(COHORT_GROUPS):
            try:
                _all_results.append(_run_cohort(_grp, _ci))
            except Exception as _exc:
                log(f"cohort {_ci} failed: {_exc!r}")
                _all_results.append({"cohort_id": _ci, "k": len(_grp),
                                     "prompt_indices": _grp,
                                     "classification": "ERROR",
                                     "error": repr(_exc)[:400],
                                     "result": {}, "row_shas": {}})
                (WORK / f"native-generate-result-c{_ci}.json").write_text(
                    json.dumps({"classification": "ERROR",
                                "error": repr(_exc)[:400]}, indent=2))
    else:
        for _qi, _ptext in enumerate(PROMPT_LIST):
            try:
                _all_results.append(_run_prompt(_ptext, _qi))
            except Exception as _exc:
                if not (_multi or _prompts_json):
                    # Standalone single-prompt run: let the fatal handler
                    # write terminal evidence (result.json, error.txt).
                    raise
                # One prompt's failure must not kill the remaining prompts.
                log(f"prompt {_qi} failed: {_exc!r}")
                _all_results.append({"prompt_index": _qi,
                                     "classification": "ERROR",
                                     "error": repr(_exc)[:400],
                                     "result": {}})
                (WORK / f"native-generate-result-q{_qi}.json").write_text(
                    json.dumps({"classification": "ERROR",
                                "error": repr(_exc)[:400],
                                "generated_token_ids": []}, indent=2))
    (WORK / "native-generate-all.json").write_text(
        json.dumps(
            [{"prompt_index": r.get("prompt_index"),
              "cohort_id": r.get("cohort_id"),
              "k": r.get("k"),
              "prompt_indices": r.get("prompt_indices"),
              "classification": r["classification"],
              "n_tokens": len(r["result"].get("generated_token_ids") or []),
              "row_shas": r.get("row_shas"),
              "token_ids_sha256": hashlib.sha256(json.dumps(
                  r["result"].get("generated_token_ids")).encode())
              .hexdigest()}
             for r in _all_results], indent=2))
    return 0


if __name__ == "__main__":
    try:
        # main() returns an exit code, but do not raise SystemExit inside this
        # catch boundary. The historical wrapper caught its own sys.exit(0)
        # and overwrote successful evidence with a false error artifact.
        main()
    except BaseException as exc:
        tb = traceback.format_exc()
        log("FATAL " + tb)
        message = str(exc).lower()
        if "non-finite" in message or "numerical" in message:
            classification = "REJECT_NUMERICAL"
        elif "out of memory" in message or "oom" in message or isinstance(exc, MemoryError):
            classification = "REJECT_MEMORY"
        elif "checksum" in message or "integrity" in message or "sha256" in message:
            classification = "REJECT_INTEGRITY"
        elif "storage" in message or "disk" in message or "no space" in message:
            classification = "REJECT_STORAGE"
        elif "gpu" in message or "nvidia-smi" in message:
            classification = "REJECT_HARDWARE"
        else:
            classification = "INVALID_EXPERIMENT"
        failed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        terminal = {
            "status": "ERROR",
            "classification": classification,
            "performance_eligible": False,
            "failed_at_utc": failed_at,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": tb,
        }
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "error.txt").write_text(tb)
        write_evidence("result.json", terminal)
        write_evidence("profile.json", {
            **terminal, "profile_stages_enabled": PROFILE_STAGES})
        try:
            failed_gpu = gpu_memory_snapshot()
        except Exception:
            failed_gpu = {}
        write_evidence("memory.json", {
            **terminal,
            "process_final_and_peak_gib": process_mem_gib(),
            "system_final_gib": system_mem_gib(),
            "gpu_final_and_peak_gib": failed_gpu,
        })
        if not (WORK / "environment.json").is_file():
            write_evidence("environment.json", {
                **terminal, "platform": platform.platform(),
                "python": sys.version})
        if not (WORK / "run_config.json").is_file():
            write_evidence("run_config.json", {
                **terminal, "run_id": RUN_ID, "cache_dtype": CACHE_DTYPE,
                "expert_store": EXPERT_STORE_BACKEND,
                "n_tokens": N_TOKENS,
                "profile_stages": PROFILE_STAGES})
        integrity_path = WORK / "integrity.json"
        try:
            failed_integrity = (json.loads(integrity_path.read_text("utf-8"))
                                if integrity_path.is_file() else {})
        except Exception:
            failed_integrity = {}
        failed_integrity.update({
            **terminal,
            "repository": failed_integrity.get("repository", REPO),
            "branch": failed_integrity.get("branch", BRANCH),
            "model_revision": failed_integrity.get("model_revision", REV),
            "executing_harness_sha256": sha256_file(Path(__file__).resolve()),
            "artifact_sha256": {
                name: sha256_file(WORK / name)
                for name in (
                    "environment.json", "run_config.json", "result.json",
                    "profile.json", "memory.json", "routed_experts.jsonl")
                if (WORK / name).is_file()
            },
        })
        write_evidence("integrity.json", failed_integrity)
        (WORK / "native-generate-result.json").write_text(
            json.dumps(terminal, indent=2))
        # Exit 0 so Kaggle snapshots /kaggle/working (error-exit kernels drop
        # their output/log, which is why the earlier failures were undiagnosable).
        # os._exit, not sys.exit: evidence is already durably written, and the
        # interpreter teardown (engine destructors, CUDA teardown, non-daemon
        # fill threads) can hang indefinitely after a mid-generation fault --
        # v11's a1 burned 4h that way.
        os._exit(0)
