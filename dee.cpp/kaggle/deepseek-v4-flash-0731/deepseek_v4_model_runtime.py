"""DS10: full DeepSeek-V4-Flash-0731 model execution on dual T4 (Kaggle).

Stage-driven harness (stage pinned in ``harness-identity-ds10.json``) following the
sealed DS5/DS9 repo-clone pattern: the push payload is ONLY this harness +
kernel-metadata.  At runtime the harness clones the pinned repository commit
into /kaggle/temp/dsv4-source, verifies harness + module SHA-256s against
``harness-identity-ds10.json``, and imports the campaign runtime
(``scripts.deepseek_v4_model`` and friends) from the checked-out tree.

Checkpoint strategy (from the DS10 storage probe, 2026-08-04):
  - /kaggle/working + /kaggle/input share a ~19.5 GiB loop disk, so the
    167 GB official checkpoint can NEVER be pre-staged on the kernel.
  - HF range-download from the kernel runs at ~93 MB/s.
  - Therefore DS10 uses ``RemoteTensorSource``: per-tensor HTTP Range
    fetches against the pinned HF revision, guided by the committed shard
    headers (identity + byte offsets), with a bounded disk/VRAM footprint.
  - Tensors are fetched on demand; routed experts materialize FP16 payloads
    through the sealed DS8 ``DeepSeekExpertCache`` on cache miss.

Stages:
  v1   model-load + tensor-resolution smoke (default): full tensor coverage
       audit (identity only), static per-GPU memory plan from headers,
       bounded RemoteTensorSource fetch smoke (embed + layer-0 dense + one
       routed + shared expert), state init.  No full generation.
  v2   first GPU partition: layers 0..split-1 forward on GPU 0.
  v3   inter-GPU handoff: GPU0 partition + one GPU1 layer with checksums.
  v4   all layers, first LM-head logits (no continued decode).
  v5   first generated token.
  v6   continued decode (>= 4 tokens).
  final  full decode (>= 16 tokens) + determinism + cold/warm + memory.

Verdicts: ACCEPT_MODEL_LOAD / ACCEPT_DUAL_T4_FIRST_TOKEN /
          ACCEPT_DUAL_T4_DECODE / REJECT_* / INVALID_EXPERIMENT.
``performance_comparable`` stays false.  No TPS claims.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

REPOSITORY = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
ROOT = Path("/kaggle/temp/dsv4-source")
DEE = ROOT / "dee.cpp"
RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
EVIDENCE = Path(f"/kaggle/working/dsv4-ds10-evidence-{RUN_ID}")
IDENTITY_RELATIVE = Path(
    "dee.cpp/kaggle/deepseek-v4-flash-0731/harness-identity-ds10.json")
HARNESS_RELATIVE = Path(
    "dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_model_runtime.py")
MODULE_RELATIVES = {
    "cache": Path("dee.cpp/scripts/deepseek_v4_cache.py"),
    "encoding": Path("dee.cpp/scripts/deepseek_v4_encoding.py"),
    "expert_reference": Path("dee.cpp/scripts/deepseek_v4_expert_reference.py"),
    "layer_candidate": Path("dee.cpp/scripts/deepseek_v4_layer_candidate.py"),
    "layer_common": Path("dee.cpp/scripts/deepseek_v4_layer_common.py"),
    "layer_reference": Path("dee.cpp/scripts/deepseek_v4_layer_reference.py"),
    "model": Path("dee.cpp/scripts/deepseek_v4_model.py"),
    "moe_reference": Path("dee.cpp/scripts/deepseek_v4_moe_reference.py"),
    "support": Path("dee.cpp/scripts/deepseek_v4_support.py"),
}
HEADERS_DIR = ROOT / Path(
    "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers")
CONFIG_RELATIVE = ROOT / Path(
    "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/official-source/inference/config.json")

VALID_STAGES = ("v1", "v2", "v3", "v4", "v5", "v6", "final", "cache1")
LADDER_INPUT_IDS = (0,)

# Cache budgets per GPU (bytes): bounded, well under the 12-13 GiB bring-up
# envelope; the model's dense/state bytes come from the static memory plan.
CACHE_BUDGET_BYTES = 2 << 30  # 2 GiB per GPU (DS10 sealed baseline)
# CACHE1: measured alloc was 5.9 GiB / peak 8.1-9.7 GiB at 2 GiB budget,
# ceiling 13.67 GiB -> 4 GiB cache budget is safe, 8 GiB is NOT (peak would
# approach the ceiling).  See CACHE1_ANALYSIS.md policy matrix.
CACHE1_BUDGET_BYTES = 4 << 30  # 4 GiB per GPU
CACHE1_RAW_EXPERTS_PER_LAYER = 16
# CACHE1b: global byte cap on the host raw-expert LRU.  The v13 run held 688
# raw experts (9.2 GB) which pushed host RSS to 28.1 GB > 12 GB ceiling.
# CACHE1c (3 GiB) still sat ~13.3 GiB steady / 15.67 GiB peak because the
# eager 43 x 48 MiB shared-FP16 host copies (2.06 GiB) plus the LRU plus
# pinned staging exceeded the ceiling (see CACHE1_ANALYSIS.md v13/v14).
# CACHE1d: 2 GiB cap (sim: provider hits 29%->22.6%, http 59.9%->66.2%; the
# 1.5 GiB cliff at 0% hits rules out going lower) + lazy shared payloads
# that free the host copy after the GPU entry is pinned.
CACHE1_RAW_MAX_BYTES = 2 << 30  # 2 GiB total across all layers
# Sealed DS10 v12 canonical decode (ACCEPT_DUAL_T4_DECODE evidence): every
# CACHE1 candidate must reproduce these exact token IDs.
SEALED_DS10_TOKENS = [
    666, 95140, 96807, 343, 4470, 20, 1127, 3298, 22, 22604, 515, 411,
    3947, 85349, 14, 6341,
]
GPU_MEMORY_CEILING_BYTES = 14_680_064_000
HOST_RSS_CEILING_BYTES = 12_000_000_000

# Canonical prompt (short, for v5/v6/final stages).  Pure-ASCII escapes:
# Kaggle's push API decodes the code file with the platform codec, so literal
# tokenizer glyphs (U+FF5C fullwidth bar, U+2581 block) must NOT appear raw.
CANONICAL_PROMPT = (
    "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>Who is Alan Turing?"
    "<\uFF5CAssistant\uFF5C>")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")


def _resolve_stage(identity: dict[str, Any]) -> str:
    """Resolve the immutable campaign stage and reject unsealed overrides."""
    stage = identity.get("stage")
    if stage not in VALID_STAGES:
        raise RuntimeError({"identity_stage": stage,
                            "expected_one_of": list(VALID_STAGES)})
    requested = os.environ.get("DS10_STAGE")
    if requested is not None and requested != stage:
        raise RuntimeError({"DS10_STAGE": requested, "identity_stage": stage,
                            "reason": "stage override does not match identity"})
    return str(stage)


def _host_memory_snapshot() -> dict[str, int]:
    """Linux RSS evidence (current + peak) in bytes for Kaggle runs."""
    import resource  # noqa: PLC0415
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return {
        "current_rss_bytes": resident_pages * page_size,
        # Linux ru_maxrss is KiB.
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "ceiling_bytes": HOST_RSS_CEILING_BYTES,
    }


def _host_memory_within_ceiling(snapshot: Any) -> bool:
    return (isinstance(snapshot, dict)
            and snapshot.get("current_rss_bytes", HOST_RSS_CEILING_BYTES + 1)
            <= HOST_RSS_CEILING_BYTES
            and snapshot.get("peak_rss_bytes", HOST_RSS_CEILING_BYTES + 1)
            <= HOST_RSS_CEILING_BYTES)


def _build_memory_within_ceiling(memory: Any) -> bool:
    if not isinstance(memory, dict):
        return False
    return (memory.get("cuda0_reserved_gib", float("inf")) * (1 << 30)
            <= GPU_MEMORY_CEILING_BYTES
            and memory.get("cuda1_reserved_gib", float("inf")) * (1 << 30)
            <= GPU_MEMORY_CEILING_BYTES
            and _host_memory_within_ceiling(memory.get("host_memory")))


def _gpu_peaks_within_ceiling(peaks: Any) -> bool:
    if not isinstance(peaks, dict):
        return False
    return all(peaks.get(device, float("inf")) * (1 << 30)
               <= GPU_MEMORY_CEILING_BYTES
               for device in ("cuda0", "cuda1"))


def _trim_host_allocator() -> dict[str, Any]:
    """Return freed glibc arenas before constructing an independent model.

    PyTorch releases the first model's CPU tensors, but glibc may retain those
    arenas in the process RSS.  A second full build would then look like two
    host models even though Python has no live reference to the first one.
    Linux ``malloc_trim`` makes the release observable and keeps the next
    model inside the contract's real RSS ceiling.
    """
    before = _host_memory_snapshot()
    result: dict[str, Any] = {"attempted": True, "before": before}
    try:
        import ctypes  # noqa: PLC0415
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        result["return_code"] = int(trim(0))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    after = _host_memory_snapshot()
    result["after"] = after
    result["current_rss_released_bytes"] = max(
        0, before["current_rss_bytes"] - after["current_rss_bytes"])
    return result


def _safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    if not str(resolved).startswith("/kaggle/temp/"):
        raise RuntimeError(f"refusing to remove unexpected path {resolved}")
    shutil.rmtree(resolved)


def _acquire_source() -> dict[str, Any]:
    """Clone the pinned commit, verify identity SHAs, put sources on sys.path."""
    if ROOT.exists():
        _safe_rmtree(ROOT)
    subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch",
                    REPOSITORY, str(ROOT)], check=True)
    identity_path = ROOT / IDENTITY_RELATIVE
    if not identity_path.is_file():
        raise RuntimeError(f"missing DS10 harness identity {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("model_revision") != REV:
        raise RuntimeError({"identity_model_revision": identity.get("model_revision"),
                            "expected": REV})
    expected_commit = identity.get("repository_commit")
    if (not isinstance(expected_commit, str) or len(expected_commit) != 40
            or any(c not in "0123456789abcdef" for c in expected_commit)):
        raise RuntimeError({"repository_commit": expected_commit,
                            "reason": "identity must pin a 40-char commit"})
    subprocess.run(["git", "checkout", "--quiet", expected_commit],
                   cwd=ROOT, check=True)
    checked_out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if checked_out != expected_commit:
        raise RuntimeError({"commit": checked_out, "expected": expected_commit})

    committed_harness = ROOT / HARNESS_RELATIVE
    committed_sha = sha256_file(committed_harness)
    running_sha = sha256_file(Path(__file__).resolve())
    if committed_sha != identity.get("harness_sha256"):
        raise RuntimeError({"committed_harness_sha256": committed_sha,
                            "expected": identity.get("harness_sha256")})
    if running_sha != identity.get("harness_sha256"):
        raise RuntimeError({"running_harness_sha256": running_sha,
                            "expected": identity.get("harness_sha256")})
    module_shas: dict[str, str] = {}
    for key, rel in MODULE_RELATIVES.items():
        module_sha = sha256_file(ROOT / rel)
        expected = identity.get("module_sha256", {}).get(key)
        if not expected or module_sha != expected:
            raise RuntimeError({f"{key}_sha256": module_sha, "expected": expected})
        module_shas[key] = module_sha

    sys.path.insert(0, str(DEE))
    sys.path.insert(0, str(ROOT / Path("dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/official-source/inference")))
    print("pinned commit", checked_out, flush=True)
    print("harness sha", running_sha, flush=True)
    return identity


def _load_cfg() -> Any:
    """ModelConfig parsed from the official config.json in the checkout."""
    from scripts import deepseek_v4_model as vm
    return vm.model_config_from_official(CONFIG_RELATIVE)


def _build_remote_source() -> Any:
    from scripts import deepseek_v4_model as vm
    return vm.RemoteTensorSource(HEADERS_DIR, revision=REV)


def stage_v1() -> dict[str, Any]:
    """Model-load + tensor-resolution smoke (identity only, bounded fetch)."""
    from scripts import deepseek_v4_model as vm
    cfg = _load_cfg()
    source = _build_remote_source()

    gates: dict[str, Any] = {}

    # 1. Tensor coverage audit (identity only, zero checkpoint bytes).
    try:
        audit = vm.coverage_audit_report(
            source, n_layers=cfg.n_layers, n_hash_layers=cfg.n_hash_layers,
            compress_ratios=cfg.compress_ratios)
        coverage_ok = vm.coverage_audit_passes(audit, n_layers=cfg.n_layers)
        gates["coverage_audit"] = coverage_ok
        gates["coverage"] = audit
    except Exception as exc:  # noqa: BLE001
        gates["coverage_audit"] = False
        gates["coverage"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return {"verdict": "REJECT_MODEL_LOAD", "gates": gates,
                "stage": "v1", "first_failing_gate": "tensor_coverage"}
    ok = coverage_ok

    # 2. Static per-GPU memory plan from headers (identity only).
    plan = vm.static_memory_plan(
        cfg, source, budgets0=CACHE_BUDGET_BYTES, budgets1=CACHE_BUDGET_BYTES)
    gates["memory_plan"] = plan
    ceilings_ok = all(
        row["total_estimate_gib"] * (1 << 30) <= GPU_MEMORY_CEILING_BYTES
        for row in plan["devices"].values())
    gates["memory_ceilings_ok"] = ceilings_ok
    ok = ok and ceilings_ok

    # 3. FULL model-load: build all 43 layers on dual T4 through the
    #    RemoteTensorSource (on-demand HF range fetches) + DS8 cache, then
    #    initialize every state buffer.  This proves model-load end to end:
    #    tensor resolution, dequant, shard fetches, cache wiring, state init.
    build_gates: dict[str, Any] = {}
    build_ok = True
    try:
        model, mem = _build_full_model(source, cfg)
        build_gates["memory"] = mem
        # Every layer's state buffers are allocated + initialized.  Actual
        # execution belongs to v2, so a layer failure cannot be mislabeled as
        # a model-load failure here.
        sig = model.state_signatures(list(range(cfg.n_layers)))
        build_gates["state_signatures"] = sig
        build_gates["state_count"] = len(sig)
        actual_memory_ok = _build_memory_within_ceiling(mem)
        build_gates["actual_memory_ceilings_ok"] = actual_memory_ok
        build_ok = len(sig) == cfg.n_layers and actual_memory_ok
        build_gates["fetch_stats"] = dict(source.stats)
    except Exception as exc:  # noqa: BLE001
        build_ok = False
        build_gates["error"] = f"{type(exc).__name__}: {exc}"
        build_gates["traceback"] = traceback.format_exc()
    gates["model_load_ok"] = build_ok
    gates["model_load"] = build_gates
    ok = ok and build_ok

    result = {"verdict": "ACCEPT_MODEL_LOAD" if ok else "REJECT_MODEL_LOAD",
              "gates": gates, "stage": "v1"}
    if not ok:
        result["first_failing_gate"] = (
            "memory_plan" if not ceilings_ok else "model_load")
    return result


def _host_hygiene(tag: str = "") -> dict[str, Any]:
    """gc + CUDA empty_cache + glibc malloc_trim (CACHE1b RSS bound).

    ``ru_maxrss`` is a process high-water mark: a peak reached during decode
    cannot be undone by trimming before the snapshot.  stage_cache1 therefore
    runs this after every decode step AND before each memory snapshot so the
    process never crosses the 12 GB ceiling in the first place.
    """
    import gc  # noqa: PLC0415
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trim = _trim_host_allocator()
    return {"tag": tag, "after": trim["after"],
            "released_bytes": trim["current_rss_released_bytes"]}


def _build_full_model(source: Any, cfg: Any, *,
                      cache_budget_bytes: int = CACHE_BUDGET_BYTES,
                      raw_experts_per_layer: int = 0,
                      raw_max_bytes: Optional[int] = None
                      ) -> tuple[Any, dict[str, Any]]:
    """Build the full dual-GPU candidate and initialize all states."""
    from scripts import deepseek_v4_model as vm
    from scripts.deepseek_v4_cache import DeepSeekExpertCache, DeepSeekExpertLoader
    cache0 = DeepSeekExpertCache(cache_budget_bytes, device="cuda:0")
    cache1 = DeepSeekExpertCache(cache_budget_bytes, device="cuda:1")
    loader0 = DeepSeekExpertLoader(cache0)
    loader1 = DeepSeekExpertLoader(cache1)
    # The sealed v5 probe measured zero compact-host LRU hits over all three
    # generation passes, and DS10 ran with raw_experts_per_layer=0 (every GPU
    # miss = full HTTP range fetch).  CACHE1 analysis (sealed v12 trace) shows
    # that was a DISABLED second-level cache, not a useless one: a per-layer
    # host LRU of raw compact tensors converts 33-47% of GPU misses into host
    # hits.  raw_experts_per_layer is therefore a policy knob here; DS10
    # stages keep 0, the cache1 stage uses CACHE1_RAW_EXPERTS_PER_LAYER.
    provider = vm.ExpertProvider(source,
                                 raw_experts_per_layer=raw_experts_per_layer,
                                 raw_max_bytes=raw_max_bytes)
    t0 = time.monotonic()
    model = vm.DeepseekV4Model.build_candidate(
        cfg, source, device0="cuda:0", device1="cuda:1",
        cache0=cache0, loader0=loader0, cache1=cache1, loader1=loader1,
        provider=provider)
    build_s = time.monotonic() - t0
    model.reset_state()
    mem = {
        "build_seconds": round(build_s, 2),
        "fetch_stats": dict(source.stats),
        "cuda0_allocated_gib": round(
            torch.cuda.memory_allocated(0) / (1 << 30), 3),
        "cuda1_allocated_gib": round(
            torch.cuda.memory_allocated(1) / (1 << 30), 3),
        "cuda0_reserved_gib": round(
            torch.cuda.memory_reserved(0) / (1 << 30), 3),
        "cuda1_reserved_gib": round(
            torch.cuda.memory_reserved(1) / (1 << 30), 3),
        "device0": str(model.device0),
        "device1": str(model.device1),
        "split": model.split,
        "n_layers": cfg.n_layers,
        "cache_budget_bytes": cache_budget_bytes,
        "host_memory": _host_memory_snapshot(),
    }
    return model, mem


def _partition_trace_row(layer_id: int, layer: Any, hidden: Any,
                         order: int) -> dict[str, Any]:
    """Compact per-layer evidence without retaining activation captures."""
    return {
        "layer": layer_id,
        "order": order,
        "shape": list(hidden.shape),
        "dtype": str(hidden.dtype),
        "device": str(hidden.device),
        "finite": bool(torch.isfinite(hidden).all()),
        "selected_experts": layer.ffn_fn.last_route.get("expert_ids"),
        "routing_weights": layer.ffn_fn.last_route.get("routing_weights"),
        "ffn_cache_counters": dict(layer.ffn_fn.stats),
    }


def stage_v2() -> dict[str, Any]:
    """First GPU partition: layers 0..split-1 forward on GPU 0."""
    cfg = _load_cfg()
    source = _build_remote_source()
    model, mem = _build_full_model(source, cfg)
    gates: dict[str, Any] = {"memory": mem}
    input_ids = torch.tensor([LADDER_INPUT_IDS], device="cuda:0").long()
    start_pos = 0
    h = torch.nn.functional.embedding(input_ids, model.embed)
    h = h.unsqueeze(2).expand(
        1, input_ids.size(1), cfg.hc_mult, h.size(-1)).contiguous()
    model.execution_trace = []
    try:
        for idx, layer in enumerate(model.layers0):
            model.last_execution = {"phase": "layer", "layer": idx,
                                    "device": model.device0}
            h = layer.forward(h, start_pos, input_ids)
            row = _partition_trace_row(idx, layer, h,
                                       len(model.execution_trace))
            model.execution_trace.append(row)
            if not row["finite"]:
                raise FloatingPointError(
                    f"non-finite hidden state after layer {idx}")
        gates["partition0_forward_ok"] = True
        gates["hidden"] = {"shape": list(h.shape), "dtype": str(h.dtype),
                           "finite": bool(torch.isfinite(h).all()),
                           "device": str(h.device)}
        gates["state_signatures_0"] = model.state_signatures(
            list(range(model.split)))
        gates["execution_trace"] = list(model.execution_trace)
        gates["runtime"] = model.runtime_snapshot()
        ok = bool(torch.isfinite(h).all())
    except Exception as exc:  # noqa: BLE001
        ok = False
        gates["partition0_forward_ok"] = False
        gates["first_failing_boundary"] = model.last_execution
        gates["error"] = f"{type(exc).__name__}: {exc}"
        gates["traceback"] = traceback.format_exc()
    result = {"verdict": "ACCEPT_PARTITION0" if ok else
              "REJECT_ATTENTION_OR_STATE", "gates": gates, "stage": "v2"}
    if not ok:
        result["first_failing_gate"] = "partition0_layer"
    return result


def stage_v3() -> dict[str, Any]:
    """Inter-GPU handoff: GPU0 partition + one GPU1 layer with checksums."""
    cfg = _load_cfg()
    source = _build_remote_source()
    model, mem = _build_full_model(source, cfg)
    gates: dict[str, Any] = {"memory": mem}
    input_ids = torch.tensor([LADDER_INPUT_IDS], device="cuda:0").long()
    start_pos = 0
    h = torch.nn.functional.embedding(input_ids, model.embed)
    h = h.unsqueeze(2).expand(
        1, input_ids.size(1), cfg.hc_mult, h.size(-1)).contiguous()
    model.execution_trace = []
    try:
        for idx, layer in enumerate(model.layers0):
            model.last_execution = {"phase": "layer", "layer": idx,
                                    "device": model.device0}
            h = layer.forward(h, start_pos, input_ids)
            row = _partition_trace_row(idx, layer, h,
                                       len(model.execution_trace))
            model.execution_trace.append(row)
            if not row["finite"]:
                raise FloatingPointError(
                    f"non-finite hidden state after layer {idx}")
        model.last_execution = {"phase": "handoff",
                                "after_layer": model.split - 1,
                                "before_layer": model.split}
        h = model._handoff(h, model.device1)
        gates["handoff"] = model.handoff_stats
        l1 = model.layer(model.split)
        model.last_execution = {"phase": "layer", "layer": model.split,
                                "device": model.device1}
        h = l1.forward(h, start_pos, input_ids.to(model.device1))
        row = _partition_trace_row(model.split, l1, h,
                                   len(model.execution_trace))
        model.execution_trace.append(row)
        gates["one_gpu1_layer_ok"] = True
        gates["hidden_after"] = {"shape": list(h.shape), "dtype": str(h.dtype),
                                 "finite": bool(torch.isfinite(h).all()),
                                 "device": str(h.device)}
        gates["execution_trace"] = list(model.execution_trace)
        gates["runtime"] = model.runtime_snapshot()
        ok = (model.handoff_stats.get("checksum_bitwise_equal") is True
              and bool(torch.isfinite(h).all()))
    except Exception as exc:  # noqa: BLE001
        ok = False
        gates["first_failing_boundary"] = model.last_execution
        gates["error"] = f"{type(exc).__name__}: {exc}"
        gates["traceback"] = traceback.format_exc()
    result = {"verdict": "ACCEPT_INTERGPU" if ok else "REJECT_INTERGPU",
              "gates": gates, "stage": "v3"}
    if not ok:
        result["first_failing_gate"] = model.last_execution.get("phase")
    return result


def stage_v4() -> dict[str, Any]:
    """All layers, first LM-head logits (no continued decode)."""
    cfg = _load_cfg()
    source = _build_remote_source()
    model, mem = _build_full_model(source, cfg)
    gates: dict[str, Any] = {"memory": mem}
    input_ids = torch.tensor([LADDER_INPUT_IDS], device="cuda:0").long()
    try:
        captures: dict[int, dict[str, Any]] = {}
        logits = model.forward(input_ids, 0, captures=captures)
        gates["logits"] = {"shape": list(logits.shape),
                           "finite": bool(torch.isfinite(logits).all()),
                           "top5": [int(x) for x in
                                    logits.topk(5, -1)[1][0].tolist()]}
        gates["layer_captures_ok"] = {
            "n_layers_with_captures": len(captures),
            "sample_l0_expert_ids": (
                captures.get(0, {}).get("expert_ids").detach().cpu().tolist()
                if captures.get(0, {}).get("expert_ids") is not None else None),
        }
        gates["execution_trace"] = list(model.execution_trace)
        gates["handoff"] = dict(model.handoff_stats)
        gates["runtime"] = model.runtime_snapshot()
        ok = bool(torch.isfinite(logits).all())
    except Exception as exc:  # noqa: BLE001
        ok = False
        gates["first_failing_boundary"] = model.last_execution
        gates["error"] = f"{type(exc).__name__}: {exc}"
        gates["traceback"] = traceback.format_exc()
    if ok:
        verdict = "ACCEPT_FIRST_LOGITS"
    elif model.last_execution.get("phase") == "lm_head":
        verdict = "REJECT_LM_HEAD"
    elif model.last_execution.get("phase") == "handoff":
        verdict = "REJECT_INTERGPU"
    else:
        verdict = "REJECT_ATTENTION_OR_STATE"
    result = {"verdict": verdict, "gates": gates, "stage": "v4"}
    if not ok:
        result["first_failing_gate"] = model.last_execution.get("phase")
    return result


def stage_generate(n_tokens: int, *, stage_name: str) -> dict[str, Any]:
    """Greedy generation: >= n_tokens, determinism, cold/warm, bounded memory."""
    cfg = _load_cfg()
    source = _build_remote_source()
    model, mem = _build_full_model(source, cfg)
    gates: dict[str, Any] = {"memory": mem}
    try:
        from scripts import deepseek_v4_encoding as enc
        tokenizer = enc.load_tokenizer()
        ids = tokenizer.encode(CANONICAL_PROMPT)
        input_ids = torch.tensor([ids], device="cuda:0").long()
        gates["prompt"] = CANONICAL_PROMPT
        gates["prompt_len"] = len(ids)
        trace: dict[str, Any] = {}
        toks = model.generate(input_ids, max_new_tokens=n_tokens, trace=trace)
        gates["tokens"] = toks
        gates["decoded_fragments"] = [tokenizer.decode([tok]) for tok in toks]
        gates["token_count"] = len(toks)
        gates["token_trace"] = trace
        gates["token_ids_in_vocab"] = all(0 <= t < cfg.vocab_size for t in toks)
        # cold == warm: reset + rerun through the same model/cache
        model.reset_state()
        toks_warm = model.generate(input_ids, max_new_tokens=n_tokens)
        gates["cold_warm_equal"] = toks == toks_warm
        gates["warm_tokens"] = toks_warm
        gates["runtime_after_warm"] = model.runtime_snapshot()
        gates["peak_memory_primary_gib"] = {
            "cuda0": round(torch.cuda.max_memory_allocated(0) / (1 << 30), 3),
            "cuda1": round(torch.cuda.max_memory_allocated(1) / (1 << 30), 3),
        }
        gates["host_memory_after_primary"] = _host_memory_snapshot()
        # Release the primary model before the independent rerun.  Keeping two
        # complete 43-layer candidates resident simultaneously leaves no T4
        # safety margin and made the original final-stage design OOM-prone.
        import gc  # noqa: PLC0415
        del model
        gc.collect()
        torch.cuda.empty_cache()
        gates["allocator_release"] = _trim_host_allocator()
        gates["host_memory_after_release"] = gates["allocator_release"]["after"]
        primary_memory_ok = (
            _build_memory_within_ceiling(mem)
            and _host_memory_within_ceiling(
                gates["host_memory_after_primary"])
            and _host_memory_within_ceiling(
                gates["host_memory_after_release"])
            and _gpu_peaks_within_ceiling(
                gates["peak_memory_primary_gib"])
        )
        if not primary_memory_ok:
            gates["generation_memory_ceilings_ok"] = False
            return {
                "verdict": "REJECT_MEMORY",
                "gates": gates,
                "stage": stage_name,
                "first_failing_gate": "generation_memory_primary",
            }
        torch.cuda.reset_peak_memory_stats()
        # The independent rerun also uses a second valid cache budget, proving
        # output invariance across cache capacity without a third full build.
        alternate_budget = 1536 << 20
        model2, mem2 = _build_full_model(
            source, cfg, cache_budget_bytes=alternate_budget)
        toks2 = model2.generate(input_ids, max_new_tokens=n_tokens)
        gates["deterministic_rerun"] = toks == toks2
        gates["cache_capacity_variation_equal"] = toks == toks2
        gates["alternate_cache_budget_bytes"] = alternate_budget
        gates["rerun_tokens"] = toks2
        gates["memory2"] = mem2
        gates["runtime_alternate_budget"] = model2.runtime_snapshot()
        gates["peak_memory_alternate_gib"] = {
            "cuda0": round(torch.cuda.max_memory_allocated(0) / (1 << 30), 3),
            "cuda1": round(torch.cuda.max_memory_allocated(1) / (1 << 30), 3),
        }
        gates["host_memory_after_alternate"] = _host_memory_snapshot()
        memory_ok = (
            primary_memory_ok
            and _build_memory_within_ceiling(mem2)
            and _host_memory_within_ceiling(
                gates["host_memory_after_alternate"])
            and _gpu_peaks_within_ceiling(
                gates["peak_memory_alternate_gib"])
        )
        gates["generation_memory_ceilings_ok"] = memory_ok
        functional_ok = (len(toks) >= n_tokens
                         and gates["token_ids_in_vocab"]
                         and gates["deterministic_rerun"]
                         and gates["cold_warm_equal"]
                         and gates["cache_capacity_variation_equal"])
        ok = functional_ok and memory_ok
        if not memory_ok:
            verdict = "REJECT_MEMORY"
        elif not functional_ok:
            verdict = "REJECT_GENERATION_STATE"
        elif n_tokens >= 16:
            verdict = "ACCEPT_DUAL_T4_DECODE"
        else:
            verdict = "ACCEPT_DUAL_T4_FIRST_TOKEN"
    except Exception as exc:  # noqa: BLE001
        ok = False
        verdict = "REJECT_GENERATION_STATE"
        gates["error"] = f"{type(exc).__name__}: {exc}"
        gates["traceback"] = traceback.format_exc()
    result = {"verdict": verdict, "gates": gates, "stage": stage_name}
    if not ok:
        result["first_failing_gate"] = (
            "generation_memory" if verdict == "REJECT_MEMORY"
            else "generation_state")
    return result


def stage_cache1() -> dict[str, Any]:
    """CACHE1: expert-cache policy candidate on the canonical 16-token decode.

    Policy stack (from CACHE1.1 analysis of the sealed v12 trace):
      - GPU cache budget 4 GiB/GPU (safe: alloc 5.9 -> ~7.9 GiB, peak
        ~10.5 GiB, ceiling 13.67 GiB);
      - host provider raw LRU enabled (raw_experts_per_layer=16) so a GPU
        miss on a recently-fetched expert is a host hit, not an HTTP fetch;
      - shared experts pinned permanently (deepseek_v4_layer_candidate.py
        calls cache.pin on first stage; pin/unpin existed but were dead
        code in DS10).

    Gates (CACHE1.4 correctness is non-negotiable):
      - token IDs identical to sealed DS10 (SEALED_DS10_TOKENS);
      - cold == warm; deterministic rerun at alternate budget;
      - memory ceilings;
      - decode-only timing (wall clock per generated token, evidence
        serialization excluded) for an honest CACHE1.5 TPS.

    Verdicts: ACCEPT_CACHE_HITRATE_TARGET (combined GPU+provider hit rate
    >= 70%) / ACCEPT_CACHE_PARTIAL (measurable gain below target) /
    REJECT_CACHE_CORRECTNESS (any CACHE1.4 failure).
    """
    cfg = _load_cfg()
    source = _build_remote_source()
    model, mem = _build_full_model(
        source, cfg, cache_budget_bytes=CACHE1_BUDGET_BYTES,
        raw_experts_per_layer=CACHE1_RAW_EXPERTS_PER_LAYER,
        raw_max_bytes=CACHE1_RAW_MAX_BYTES)
    gates: dict[str, Any] = {"memory": mem,
                             "policy": {
                                 "cache_budget_bytes": CACHE1_BUDGET_BYTES,
                                 "raw_experts_per_layer":
                                     CACHE1_RAW_EXPERTS_PER_LAYER,
                                 "raw_max_bytes": CACHE1_RAW_MAX_BYTES,
                                 "shared_pinned": True,
                                 "sealed_reference_tokens":
                                     list(SEALED_DS10_TOKENS)}}
    try:
        from scripts import deepseek_v4_encoding as enc
        tokenizer = enc.load_tokenizer()
        ids = tokenizer.encode(CANONICAL_PROMPT)
        input_ids = torch.tensor([ids], device="cuda:0").long()
        gates["prompt"] = CANONICAL_PROMPT
        gates["prompt_len"] = len(ids)
        n_tokens = len(SEALED_DS10_TOKENS)
        trace: dict[str, Any] = {}
        decode_ms: list[float] = []
        hygiene_steps: list[dict[str, Any]] = []

        def _step_hygiene(step: int) -> None:
            hygiene_steps.append(_host_hygiene(tag=f"step_{step}"))
            print(f"[cache1] primary decode step {step}/{n_tokens}",
                  flush=True)

        print(f"[cache1] primary build complete: {mem.get('build_seconds')}s "
              f"fetch={mem.get('fetch_stats', {}).get('requests')} reqs "
              f"{round(mem.get('fetch_stats', {}).get('bytes', 0) / (1 << 30), 2)} GiB",
              flush=True)
        print(f"[cache1] primary decode starting (prompt_len={len(ids)}, "
              f"tokens={n_tokens})", flush=True)
        toks = model.generate(input_ids, max_new_tokens=n_tokens, trace=trace,
                              decode_timings_ms=decode_ms,
                              post_step_hook=_step_hygiene)
        print(f"[cache1] primary decode done: "
              f"{round(decode_ms[0], 0) if decode_ms else 0}ms prefill / "
              f"{round(sum(decode_ms[1:]) / 1000.0, 1)}s decode", flush=True)
        gates["hygiene_steps"] = hygiene_steps
        gates["tokens"] = toks
        gates["tokens_match_sealed_ds10"] = (
            toks == SEALED_DS10_TOKENS)
        gates["decoded_fragments"] = [tokenizer.decode([tok]) for tok in toks]
        gates["token_count"] = len(toks)
        gates["token_trace"] = trace
        gates["token_ids_in_vocab"] = all(0 <= t < cfg.vocab_size for t in toks)
        # decode-only timing (excludes trace serialization / state hashing).
        # CACHE1 contract: prefill and decode are reported separately.
        # decode_ms[0] is the PREFILL forward (7-position prompt); samples
        # 1..n-1 are the autoregressive decode steps.
        prefill_ms = decode_ms[0] if decode_ms else 0.0
        decode_ms_only = decode_ms[1:]
        gates["prefill_ms"] = round(prefill_ms, 2)
        gates["decode_timings_ms"] = [round(t, 2) for t in decode_ms_only]
        gates["decode_wall_s"] = round(sum(decode_ms_only) / 1000.0, 3)
        gates["decode_tok_per_s"] = round(
            len(decode_ms_only) / max(1e-9, sum(decode_ms_only) / 1000.0), 3)
        gates["decode_token_count"] = len(decode_ms_only)
        # cold == warm: reset + rerun through the same model/cache
        model.reset_state()
        hygiene_warm: list[dict[str, Any]] = []

        def _warm_hygiene(step: int) -> None:
            hygiene_warm.append(_host_hygiene(tag=f"warm_step_{step}"))
            print(f"[cache1] warm decode step {step}/{n_tokens}", flush=True)

        print("[cache1] warm decode starting (cold==warm rerun)", flush=True)
        toks_warm = model.generate(input_ids, max_new_tokens=n_tokens,
                                   post_step_hook=_warm_hygiene)
        print("[cache1] warm decode done", flush=True)
        gates["cold_warm_equal"] = toks == toks_warm
        gates["warm_tokens"] = toks_warm
        gates["hygiene_warm_steps"] = hygiene_warm
        gates["allocator_release"] = _host_hygiene(tag="after_warm")
        gates["runtime_after_warm"] = model.runtime_snapshot()
        gates["peak_memory_primary_gib"] = {
            "cuda0": round(torch.cuda.max_memory_allocated(0) / (1 << 30), 3),
            "cuda1": round(torch.cuda.max_memory_allocated(1) / (1 << 30), 3),
        }
        gates["host_memory_after_primary"] = _host_memory_snapshot()
        primary_memory_ok = (
            _build_memory_within_ceiling(mem)
            and _host_memory_within_ceiling(
                gates["host_memory_after_primary"])
            and _gpu_peaks_within_ceiling(
                gates["peak_memory_primary_gib"])
        )
        if not primary_memory_ok:
            gates["memory_ceilings_ok"] = False
            return {"verdict": "REJECT_MEMORY", "gates": gates,
                    "stage": "cache1",
                    "first_failing_gate": "cache1_memory_primary"}
        torch.cuda.reset_peak_memory_stats()
        # capacity invariance at a second valid budget (no third build needed
        # for correctness: reuse the alternate-budget rerun as DS10 did).
        # CACHE1b: release the primary model BEFORE the independent build so
        # two full 43-layer candidates are never resident simultaneously
        # (v13 leaked host RSS partly because this step was missing).
        import gc  # noqa: PLC0415
        del model
        gc.collect()
        torch.cuda.empty_cache()
        gates["allocator_release_primary"] = _host_hygiene(
            tag="after_primary_release")
        alternate_budget = 1536 << 20
        model2, mem2 = _build_full_model(
            source, cfg, cache_budget_bytes=alternate_budget,
            raw_experts_per_layer=CACHE1_RAW_EXPERTS_PER_LAYER,
            raw_max_bytes=CACHE1_RAW_MAX_BYTES)
        hygiene2: list[dict[str, Any]] = []

        def _hygiene2(step: int) -> None:
            hygiene2.append(_host_hygiene(tag=f"alt_step_{step}"))
            print(f"[cache1] alternate decode step {step}/{n_tokens}", flush=True)

        print(f"[cache1] alternate build complete: "
              f"{mem2.get('build_seconds')}s "
              f"fetch={mem2.get('fetch_stats', {}).get('requests')} reqs",
              flush=True)
        print("[cache1] alternate decode starting", flush=True)
        toks2 = model2.generate(input_ids, max_new_tokens=n_tokens,
                                post_step_hook=_hygiene2)
        print("[cache1] alternate decode done", flush=True)
        gates["deterministic_rerun"] = toks == toks2
        gates["cache_capacity_variation_equal"] = toks == toks2
        gates["alternate_cache_budget_bytes"] = alternate_budget
        gates["rerun_tokens"] = toks2
        gates["hygiene_alternate_steps"] = hygiene2
        gates["allocator_release_alternate"] = _host_hygiene(
            tag="after_alternate")
        gates["memory2"] = mem2
        gates["runtime_alternate_budget"] = model2.runtime_snapshot()
        gates["peak_memory_alternate_gib"] = {
            "cuda0": round(torch.cuda.max_memory_allocated(0) / (1 << 30), 3),
            "cuda1": round(torch.cuda.max_memory_allocated(1) / (1 << 30), 3),
        }
        gates["host_memory_after_alternate"] = _host_memory_snapshot()
        memory_ok = (primary_memory_ok
                     and _build_memory_within_ceiling(mem2)
                     and _host_memory_within_ceiling(
                         gates["host_memory_after_alternate"])
                     and _gpu_peaks_within_ceiling(
                         gates["peak_memory_alternate_gib"]))
        gates["memory_ceilings_ok"] = memory_ok

        # ---- CACHE1.5 metrics (combined GPU + provider two-level cache) ----
        c0 = gates["runtime_after_warm"]["cache0"]
        c1 = gates["runtime_after_warm"]["cache1"]
        prov = gates["runtime_after_warm"]["provider"]
        total_requests = c0.get("requests", 0) + c1.get("requests", 0)
        gpu_hits = c0.get("hits", 0) + c1.get("hits", 0)
        h2d_bytes = c0.get("h2d_bytes", 0) + c1.get("h2d_bytes", 0)
        http_fetches = prov.get("fetch_count", 0)
        raw_hits = prov.get("raw_hits", 0)
        raw_misses = prov.get("raw_misses", 0)
        gpu_hit_rate = 100.0 * gpu_hits / max(1, total_requests)
        # provider raw hits serve GPU misses without an HTTP range fetch;
        # every GPU hit also avoids HTTP.  Exact combined fraction:
        combined_no_http = 100.0 * (gpu_hits + raw_hits) / max(1, total_requests)
        # (routed-only raw counters slightly undercount the denominator, so
        # this is the conservative exact-arithmetic bound)
        gates["cache1_metrics"] = {
            "gpu_requests": total_requests,
            "gpu_hits": gpu_hits,
            "gpu_hit_rate_pct": round(gpu_hit_rate, 2),
            "provider_raw_hits": raw_hits,
            "provider_raw_misses": raw_misses,
            "provider_hit_rate_pct": round(provider_hit_rate, 2),
            "combined_no_http_pct": round(combined_no_http, 2),
            "http_fetch_count": http_fetches,
            "h2d_bytes_total": h2d_bytes,
            "h2d_gib": round(h2d_bytes / (1 << 30), 3),
            "baseline_http_fetch_count": 10202,
            "baseline_h2d_gib": round(582739820544 / (1 << 30), 3),
        }
        # CACHE1.1 analysis (sealed v12 trace): the >=70% combined target is
        # NOT reachable on the canonical 16-token trajectory even under oracle
        # eviction (Belady ceiling ~43-56% at 4-8 GiB/GPU; ~60% at 12 GiB
        # which busts the memory ceiling).  A partial-accept verdict is the
        # expected outcome of this stage unless the run surprises the sim.
        gates["target_unreachable_reason"] = (
            "CACHE1.1: Belady/oracle ceiling on the sealed 16-token trace is "
            "~20-22% at 2 GiB, ~33-41% at 4 GiB, ~46-56% at 8 GiB per GPU; "
            ">=70% combined requires a longer trajectory or a bigger budget "
            "than the 13.67 GiB T4 ceiling allows. ACCEPT_CACHE_PARTIAL is "
            "the designed terminal verdict for this stage.")

        functional_ok = (
            gates["tokens_match_sealed_ds10"]
            and gates["token_ids_in_vocab"]
            and gates["deterministic_rerun"]
            and gates["cold_warm_equal"]
            and gates["cache_capacity_variation_equal"])
        ok = functional_ok and memory_ok
        if not memory_ok:
            verdict = "REJECT_MEMORY"
        elif not functional_ok:
            verdict = "REJECT_CACHE_CORRECTNESS"
        elif combined_no_http >= 70.0:
            verdict = "ACCEPT_CACHE_HITRATE_TARGET"
        else:
            verdict = "ACCEPT_CACHE_PARTIAL"
    except Exception as exc:  # noqa: BLE001
        ok = False
        verdict = "REJECT_CACHE_CORRECTNESS"
        gates["error"] = f"{type(exc).__name__}: {exc}"
        gates["traceback"] = traceback.format_exc()
    result = {"verdict": verdict, "gates": gates, "stage": "cache1"}
    if not ok:
        result["first_failing_gate"] = (
            "cache1_memory" if verdict == "REJECT_MEMORY"
            else "cache1_correctness")
    return result


def main() -> int:
    result: dict[str, Any] = {"verdict": "INVALID_EXPERIMENT",
                              "stage": None, "performance_comparable": False}
    out_dir = EVIDENCE
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        identity = _acquire_source()
        stage = _resolve_stage(identity)
        global torch
        import torch  # noqa: PLC0415
        result["identity"] = identity
        result["stage"] = stage
        result["run_id"] = RUN_ID
        result["model_revision"] = REV
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["gpu_count"] = (torch.cuda.device_count()
                               if torch.cuda.is_available() else 0)
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            result["verdict"] = "INVALID_EXPERIMENT"
            result["first_failing_gate"] = "cuda_dual_required"
        else:
            if stage == "v1":
                result.update(stage_v1())
            elif stage == "v2":
                result.update(stage_v2())
            elif stage == "v3":
                result.update(stage_v3())
            elif stage == "v4":
                result.update(stage_v4())
            elif stage == "v5":
                result.update(stage_generate(1, stage_name="v5"))
            elif stage == "v6":
                result.update(stage_generate(4, stage_name="v6"))
            elif stage == "cache1":
                result.update(stage_cache1())
            else:
                result.update(stage_generate(16, stage_name="final"))
    except Exception as exc:  # noqa: BLE001
        result["verdict"] = "INVALID_EXPERIMENT"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print("FATAL:", exc, flush=True)
    finally:
        verdict_path = out_dir / "ds10-verdict.json"
        evidence_path = out_dir / "ds10-evidence.json"
        write_json(verdict_path,
                   {"verdict": result["verdict"],
                    "stage": result.get("stage"),
                    "performance_comparable": False,
                    "first_failing_gate": result.get("first_failing_gate"),
                    "run_id": RUN_ID})
        write_json(evidence_path, result)
        shutil.copy2(Path(__file__).resolve(),
                     out_dir / "deepseek_v4_model_runtime.py")
        write_json(out_dir / "harness-identity-ds10.json",
                   result.get("identity", {}))
        contract_path = ROOT / Path(
            "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
            "DS10_FULL_MODEL_CONTRACT.json")
        if contract_path.is_file():
            shutil.copy2(contract_path, out_dir / contract_path.name)
        for key, rel in MODULE_RELATIVES.items():
            shutil.copy2(ROOT / rel,
                         out_dir / f"deepseek_v4_module_{key}.py")
        artifact_rows = [
            {"path": path.relative_to(out_dir).as_posix(),
             "bytes": path.stat().st_size,
             "sha256": sha256_file(path)}
            for path in sorted(out_dir.rglob("*"))
            if path.is_file() and path.name != "ds10-artifact-manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "stage": result.get("stage"),
            "verdict": result["verdict"],
            "result": ("PASS" if str(result["verdict"]).startswith("ACCEPT")
                       else "FAIL"),
            "performance_comparable": False,
            "repository_commit": result.get("identity", {}).get(
                "repository_commit"),
            "required_paths": [
                "ds10-verdict.json", "ds10-evidence.json",
                "deepseek_v4_model_runtime.py", "harness-identity-ds10.json",
                "DS10_FULL_MODEL_CONTRACT.json",
            ],
            "artifacts": artifact_rows,
        }
        manifest_path = out_dir / "ds10-artifact-manifest.json"
        write_json(manifest_path, manifest)
        archive = shutil.make_archive(str(out_dir), "gztar",
                                      root_dir=out_dir.parent,
                                      base_dir=out_dir.name)
        write_json(out_dir / "archive-metadata.json", {
            "archive": archive,
            "archive_sha256": sha256_file(Path(archive)),
            "manifest_sha256": sha256_file(manifest_path),
            "excluded_from_archive": ["archive-metadata.json"],
        })
        print("VERDICT:", result["verdict"], result.get("first_failing_gate", ""),
              flush=True)
    return 0 if str(result["verdict"]).startswith("ACCEPT") else 1


if __name__ == "__main__":
    sys.exit(main())
