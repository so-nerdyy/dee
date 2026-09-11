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
ROOT = Path("/tmp/dsv4-native-src")
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build-kaggle"
CKPT = Path("/tmp/dsv4-checkpoint")
# When the checkpoint is published as a Kaggle dataset it mounts read-only at
# /kaggle/input/<slug>/; prefer that (no download, no disk quota). The local
# /tmp fallback remains for the 155 GiB-free case.
DATASET_DIR = Path("/kaggle/input/deepseek-v4-flash-0731-shards")
WORK = Path("/kaggle/working")
HEADERS_DIR = (DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers")
CONFIG = (DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/"
          "official-source/inference/config.json")
CANONICAL_PROMPT = (
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
    if CACHE_DTYPE not in {"fp16", "fp4"}:
        raise ValueError(f"unsupported cache_dtype: {CACHE_DTYPE!r}")
    if EXPERT_STORE_BACKEND not in {"safetensors", "dee4", "dee4_trace"}:
        raise ValueError(
            f"unsupported expert_store: {EXPERT_STORE_BACKEND!r}")
    if DEE4_VALIDATE_SAMPLES <= 0:
        raise ValueError("dee4_validate_samples must be positive")
    if not 1 <= SOURCE_READ_LANES <= 8:
        raise ValueError("source_read_lanes must be in [1, 8]")
    if not 1 <= SOURCE_READ_QUEUE_DEPTH <= 256:
        raise ValueError("source_read_queue_depth must be in [1, 256]")
    log(
        "[config] run_config.json: "
        f"run_id={RUN_ID} cache_dtype={CACHE_DTYPE} n_tokens={N_TOKENS} "
        f"expert_store={EXPERT_STORE_BACKEND} "
        f"dee4_trace_path={DEE4_TRACE_PATH} "
        f"dee4_validate_samples={DEE4_VALIDATE_SAMPLES} "
        f"source_read_lanes={SOURCE_READ_LANES} "
        f"source_read_queue_depth={SOURCE_READ_QUEUE_DEPTH} "
        f"profile_stages={PROFILE_STAGES}"
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
    if EXPERT_STORE_BACKEND in {"dee4", "dee4_trace"}:
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
        "exact_16_token_ids": N_TOKENS == 16 and tokens == SEALED_TOKEN_IDS,
        "exact_decoded_text": result.get("decoded_text") == SEALED_DECODED_TEXT,
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
    token_or_text_failed = not (
        gates["exact_16_token_ids"] and gates["exact_decoded_text"])
    observed_nonfinite = (
        gates["finite_outputs_observed"] and not gates["finite_outputs"])
    contract_gates = [value for key, value in gates.items()
                      if key != "required_performance_hardware"]
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
    global FORCE_TMP
    gpu_environment = check_gpu_allocation()
    res = log_host_resources("startup")
    tmp_free = res.get("/tmp", {}).get("free_gb", 0)

    log("=== clone + checkout ===")
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
        "host_pack_runtime_cap_gib_total": 17.0,
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
    except Exception as _e:
        log(f"DEE4 prepare failed: {_e}")
        import traceback as _tb
        _tb.print_exc()
        if EXPERT_STORE_BACKEND in {"dee4", "dee4_trace"}:
            raise

    sys.path.insert(0, str(DEE))
    sys.path.insert(0, str(DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/"
                          "official-source/inference"))
    import torch
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
    LRU_TOTAL_CAP_GIB = 17.0
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
            expert_store_path=dee4_store_path)
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
            expert_store_path=dee4_store_path)
        eng1 = vm.build_native_engine(
            shard_paths, device_id=1, budget_bytes=BUDGET_BYTES,
            host_pack_cache_bytes=pack_budget1,
            use_batched_experts=USE_BATCHED_EXPERTS,
            profile_stages=PROFILE_STAGES,
            cache_dtype=CACHE_DTYPE,
            source_read_lanes=SOURCE_READ_LANES,
            source_read_queue_depth=SOURCE_READ_QUEUE_DEPTH,
            expert_store_path=dee4_store_path)
        log(f"engines built (cache_dtype={CACHE_DTYPE})")

    log("=== build full model (native FFN) ===")
    t0 = time.monotonic()
    # The tensor source must read dense tensors (embed/head/norm/attention/
    # router/shared) from whichever directory actually holds the shards:
    # the dataset mount when attached, or the local /tmp download otherwise.
    shards_dir = Path(shard_paths[0]).parent
    source = vm.LocalDirTensorSource(HEADERS_DIR, shards_dir)
    provider = vm.ExpertProvider(source)
    if SINGLE_GPU:
        model = vm.DeepseekV4Model.build_candidate(
            cfg, source, device0="cuda:0", device1="cuda:0",
            cache0=None, loader0=None, cache1=None, loader1=None,
            provider=provider, ffn_backend="native",
            engine0=eng0, engine1=eng1, split=cfg.n_layers,
            diagnostics=DIAGNOSTICS, profile_stages=PROFILE_STAGES)
    else:
        model = vm.DeepseekV4Model.build_candidate(
            cfg, source, device0="cuda:0", device1="cuda:1",
            cache0=None, loader0=None, cache1=None, loader1=None,
            provider=provider, ffn_backend="native",
            engine0=eng0, engine1=eng1, diagnostics=DIAGNOSTICS,
            profile_stages=PROFILE_STAGES)
    model.reset_state()
    build_s = time.monotonic() - t0
    log(f"model build {build_s:.1f}s")

    log("=== tokenize + greedy decode ===")
    ids = tokenizer.encode(CANONICAL_PROMPT)
    input_ids = torch.tensor([ids], device="cuda:0").long()
    decode_ms: list[float] = []
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
    # v12: checkpoint every generated token to /kaggle/working so an OOM kill
    # (v9/v11 lost ALL tokens) still leaves the exact token stream + timing.
    # The checkpoint file format is a JSONL of per-token records; the final
    # RESULT block below mirrors the old single-JSON shape.
    CHECKPOINT = WORK / "generated_checkpoint.jsonl"
    cp_handle = open(CHECKPOINT, "w", encoding="utf-8")
    ROUTE_JOURNAL_PATH = WORK / "routed_experts.jsonl"
    route_journal = RoutedExpertJournal(
        ROUTE_JOURNAL_PATH, run_id=RUN_ID, n_layers=cfg.n_layers,
        topk=cfg.topk)
    route_step = 0
    route_start_pos = 0

    def _route_checkpoint(layer_id: int) -> None:
        """Persist the exact CPU route buffer already consumed by native."""
        nonlocal route_step, route_start_pos
        layer = model.layer(int(layer_id))
        ids_host = getattr(
            layer.ffn_fn, "_native_route_ids_host", None)
        if ids_host is None:
            raise RuntimeError(
                f"native route buffer unavailable after layer {layer_id}")
        if bool(getattr(ids_host, "is_cuda", False)):
            raise RuntimeError(
                f"route journal refuses a device read at layer {layer_id}")
        route_journal.append_layer(
            step=route_step, start_pos=route_start_pos,
            layer=int(layer_id), device=str(layer.device),
            expert_ids=ids_host)
        if int(layer_id) == cfg.n_layers - 1:
            route_step += 1
            route_start_pos = len(ids) + route_step - 1

    def _token_checkpoint(step: int, tok: int) -> None:
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
            decode_timings_ms=decode_ms,
            post_step_hook=_token_checkpoint,
            post_layer_hook=_route_checkpoint)
        wall_s = time.monotonic() - t0
    finally:
        route_journal.close()
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
        "commit": head,
        "host_mem_available_gib": round(mem_avail, 2),
        "host_pack_budget_gib": [round(pack_budget0 / (1 << 30), 2),
                                  round(pack_budget1 / (1 << 30), 2)],
        "model_revision": REV,
        "gpu_environment": gpu_environment,
        "cache_dtype": CACHE_DTYPE,
        "expert_store_backend": EXPERT_STORE_BACKEND,
        "expert_store_path": dee4_store_path,
        "prompt": CANONICAL_PROMPT,
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
    # twice.
    store_keys = ("cuda0",) if SINGLE_GPU else ("cuda0", "cuda1")
    stores = result.get("expert_store", {})
    storage_bytes = sum(
        int(stores.get(key, {}).get("bytes_requested", 0))
        for key in store_keys)
    source_reads = sum(
        int(stores.get(key, {}).get("source_reads", 0))
        for key in store_keys)
    result["byte_accounting"] = {
        "storage_bytes_total": storage_bytes,
        "storage_bytes_per_generated_token": (
            storage_bytes / len(toks) if toks else None),
        "storage_requests_total": source_reads,
        "storage_requests_per_generated_token": (
            source_reads / len(toks) if toks else None),
        "expert_h2d_bytes_total": sum(
            int(result.get("engine_stats", {}).get(key, {}).get("h2d_bytes", 0))
            for key in store_keys),
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
    previous_totals = {
        "storage_bytes": 0.0,
        "storage_requests": 0.0,
        "source_read_wall_ms": 0.0,
        "h2d_bytes": 0.0,
        "h2d_copies": 0.0,
        "resident_hits": 0.0,
        "cold_loads": 0.0,
        "evictions": 0.0,
        "host_pack_hits": 0.0,
        "host_pack_misses": 0.0,
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

    storage_read_ms = sum(
        float(stores.get(key, {}).get("read_milliseconds", 0))
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
    write_evidence("environment.json", environment_payload)
    write_evidence("run_config.json", run_config_payload)
    write_evidence("profile.json", profile_payload)
    write_evidence("memory.json", memory_payload)
    write_evidence("result.json", result)
    integrity_payload.update({
        "completed_at_utc": completed_utc,
        "classification": classification,
        "performance_eligible": performance_eligible,
        "actual_token_ids": [int(token) for token in toks],
        "actual_token_ids_sha256": hashlib.sha256(
            json.dumps([int(token) for token in toks], separators=(",", ":"))
            .encode("utf-8")).hexdigest(),
        "actual_decoded_text_sha256": hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        "sealed_contract_gates": gates,
        "expert_store": result.get("expert_store", {}),
        "dee4_trace_validation": result.get("dee4_trace_validation", {}),
        "artifact_sha256": {
            name: sha256_file(WORK / name)
            for name in (
                "environment.json", "run_config.json", "result.json",
                "profile.json", "memory.json", "routed_experts.jsonl")
        },
    })
    if EXPERT_STORE_BACKEND == "dee4_trace":
        integrity_payload["artifact_sha256"]["dee4-trace-validation.json"] = (
            sha256_file(WORK / "dee4-trace-validation.json"))
    write_evidence("integrity.json", integrity_payload)
    log("RESULT " + json.dumps(result))
    (WORK / "native-generate-result.json").write_text(
        json.dumps(result, indent=2))

    # Clean up the local download only; the dataset mount is read-only and
    # must not be touched (unlink would raise PermissionError there).
    if not (DATASET_DIR.is_dir() and Path(shard_paths[0]).parent == DATASET_DIR):
        for p in shard_paths:
            Path(p).unlink(missing_ok=True)
    log(f"=== VERDICT: {classification}; performance_eligible="
        f"{performance_eligible} ===")
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
        sys.exit(0)
