"""DS5: trusted official-reference trace generation (layer-0 subset, Kaggle).

Runs the OFFICIAL DeepSeek-V4 inference stack
(``inference/model.py`` + tilelang ``inference/kernel.py``, converted with
``inference/convert.py``) for a bounded single-layer reference — embedding +
layer 0 (hash-routed, pure sliding-window: ``compress_ratios[0] == 0``) +
final norm/head — on the canonical prompt, capturing bounded traces at every
official boundary.

Checkpoint footprint (pinned subset, ~5.7 GB, from the 48-shard checkpoint):

- model-00001-of-00048.safetensors  embed.weight                    1.06 GB
- model-00002-of-00048.safetensors  layer 0 (dense + 256 routed + shared) 3.57 GB
- model-00045-of-00048.safetensors  norm.weight, head.weight, hc_head_* 1.06 GB

Stages (each importable and unit-testable; ``main`` runs them all):

0. probe      environment (torch/tilelang/transformers, CUDA, SM, disk, HF)
1. identity   pinned config/tokenizer/encoding SHA-256 verification
2. encode     canonical prompt -> official token IDs (DS4 encoder)
3. shards     find mounted subset or resumable-download + verify (size+header)
4. convert    official convert.py (expert_dtype=fp4, mp=1) -> model0-mp1
5. reference  official Transformer(n_layers=1) + load_state_dict(strict=False)
6. hooks      name-driven boundary hooks (embed, layers.0.*, norm, head)
7. trace      prefill forward + greedy decode steps; bounded captures
8. gates      fail-closed gate chain -> verdict + evidence + manifest

Every blocker (missing shards, tilelang SM75 failure, load failure) is
reported as a verdict with the FIRST failing gate; evidence is still written.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

# --- flat-copy imports (kernel payload layout) -----------------------------
import deepseek_v4_encoding as enc
import deepseek_v4_trace_spec as spec

OFFICIAL_REPOSITORY = spec.OFFICIAL_REPOSITORY
OFFICIAL_REVISION = spec.OFFICIAL_REVISION

# Pinned subset manifest: shard -> (size_bytes, header_sha256, tensor_count)
SUBSET_MANIFEST = {
    "model-00001-of-00048.safetensors": (
        1059061760, "ab23277543e81504fc621596b78b61db207fef4e9d1b522c09a543127220d261", 1,
    ),
    "model-00002-of-00048.safetensors": (
        3566148952, "8c5453415265f75d1b5afeeafbbb92fe20b3ccf4ce1ff5c55db932b28134ec38", 1565,
    ),
    "model-00045-of-00048.safetensors": (
        1059332116, "4e0d9e8684cf49832e97f565708d518012623a6881b2e392a21c0d16c23f1c65", 5,
    ),
}

MAX_DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_CHUNK = 8 << 20  # 8 MiB range requests


# ---------------------------------------------------------------------------
# Dependency bootstrap (Kaggle script kernels have no setup step)
# ---------------------------------------------------------------------------

def bootstrap_deps() -> dict[str, Any]:
    """Best-effort pip install of the non-torch reference dependencies.

    Kaggle's torch is a custom build; this harness NEVER upgrades torch in
    place. Missing torch features (e.g. ``float4_e2m1fn_x2`` on torch < 2.10)
    surface later as precise gate failures from convert/reference stages.
    """
    import importlib.metadata as _md
    import subprocess

    def _have(name: str) -> bool:
        try:
            _md.version(name)
            return True
        except Exception:
            return False

    wanted = ["tqdm", "fast_hadamard_transform", "tilelang==0.1.8"]
    missing = [pkg for pkg in wanted if not _have(pkg.split("==")[0])]
    results: dict[str, bool] = {}
    for pkg in missing:
        run = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            capture_output=True, text=True, timeout=900,
        )
        results[pkg] = run.returncode == 0
    return {"missing_before": missing, "install_ok": results,
            "all_present": not missing or all(results.values())}


# ---------------------------------------------------------------------------
# Stage 0 — environment probe
# ---------------------------------------------------------------------------

def run_probe() -> dict[str, Any]:
    import importlib.metadata as _md
    probe = spec.EnvProbe()
    try:
        import torch
        probe.torch_version = torch.__version__
        probe.cuda_available = torch.cuda.is_available()
        if probe.cuda_available:
            probe.cuda_device = torch.cuda.get_device_name(0)
            probe.sm_arch = spec.parse_sm_arch(torch.cuda.get_device_properties(0))
    except Exception as exc:  # pragma: no cover
        probe.first_failing_requirement = f"torch: {exc}"
    for pkg in ("transformers", "safetensors", "tilelang", "tqdm"):
        try:
            setattr(probe, f"{pkg.replace('-', '_')}_version", _md.version(pkg))
        except Exception:
            if pkg == "tilelang":
                setattr(probe, "tilelang_version", "")
            if pkg == "tqdm":
                setattr(probe, "tqdm_version", "")
    try:
        import shutil as _sh
        probe.disk_free_gib = _sh.disk_usage("/kaggle" if Path("/kaggle").exists() else "/tmp").free / (1 << 30)
    except Exception:
        pass
    probe.hf_reachable = _hf_reachable()
    probe.checkpoint_mounted = _find_mounted_checkpoint() is not None
    return probe.to_dict()


def _hf_reachable() -> bool:
    try:
        with urllib.request.urlopen(
            f"https://huggingface.co/{OFFICIAL_REPOSITORY}/resolve/{OFFICIAL_REVISION}/config.json",
            timeout=30,
        ) as response:
            return response.status == 200 and len(response.read()) > 500
    except Exception:
        return False


def _find_mounted_checkpoint() -> Path | None:
    """Look for a checkpoint dir: env DSV4_CKPT_DIR, then /kaggle/input/*."""
    env = os.environ.get("DSV4_CKPT_DIR")
    if env and (Path(env) / "model.safetensors.index.json").is_file():
        return Path(env)
    for candidate in Path("/kaggle/input").glob("*") if Path("/kaggle/input").exists() else []:
        if (candidate / "model.safetensors.index.json").is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Stage 1 — identity
# ---------------------------------------------------------------------------

def verify_identity(assets_dir: Path) -> dict[str, Any]:
    """Fail-closed verification of pinned config + tokenizer/encoding assets."""
    files = spec.verify_pinned_files(assets_dir)
    tokenizer_hashes = enc.verify_tokenizer_assets()
    return {"files": files, "tokenizer": tokenizer_hashes,
            "revision": OFFICIAL_REVISION}


# ---------------------------------------------------------------------------
# Stage 2 — prompt encode
# ---------------------------------------------------------------------------

def encode_canonical_prompt() -> dict[str, Any]:
    messages = [{"role": "user", "content": spec.CANONICAL_PROMPT}]
    text, ids = enc.encode_message_ids(messages, thinking_mode=spec.CANONICAL_THINKING_MODE)
    return {"prompt": spec.CANONICAL_PROMPT, "mode": spec.CANONICAL_THINKING_MODE,
            "text": text, "token_ids": ids, "n_tokens": len(ids),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


# ---------------------------------------------------------------------------
# Stage 3 — shard acquisition + verification
# ---------------------------------------------------------------------------

def _header_sha256(path: Path) -> str:
    """Re-serialize the safetensors header JSON and hash it (manifest contract)."""
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    return hashlib.sha256(json.dumps(header, sort_keys=True).encode("utf-8")).hexdigest()


def _download_shard_resumable(shard: str, out_dir: Path) -> None:
    url = (f"https://huggingface.co/{OFFICIAL_REPOSITORY}/resolve/"
           f"{OFFICIAL_REVISION}/{shard}")
    target = out_dir / shard
    expected_size = SUBSET_MANIFEST[shard][0]
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        current = target.stat().st_size if target.exists() else 0
        if current >= expected_size:
            return
        headers = {"Range": f"bytes={current}-{expected_size - 1}",
                   "User-Agent": "freebuff-ds5-trace/1.0"}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=600) as response:
                with target.open("ab") as handle:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
        except Exception as exc:
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(f"download failed for {shard}: {exc}") from exc
            time.sleep(5 * attempt)


def ensure_subset_shards(work_dir: Path) -> dict[str, Any]:
    """Resolve the 3 subset shards (mounted or downloaded) and verify them.

    Fail-closed: any size or header mismatch raises; the record lists what
    was verified.
    """
    hf_dir = work_dir / "hf-subset"
    hf_dir.mkdir(parents=True, exist_ok=True)
    mounted = _find_mounted_checkpoint()
    result: dict[str, Any] = {"mode": "download", "verified": {}}
    for shard, (expected_size, expected_header, expected_count) in SUBSET_MANIFEST.items():
        src: Path | None = None
        if mounted is not None and (mounted / shard).is_file():
            src = mounted / shard
            result["mode"] = "mount"
        if src is None:
            _download_shard_resumable(shard, hf_dir)
            src = hf_dir / shard
        actual_size = src.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"{shard}: size {actual_size} != expected {expected_size}"
            )
        actual_header = _header_sha256(src)
        if actual_header != expected_header:
            raise RuntimeError(
                f"{shard}: header sha256 {actual_header[:16]}... != expected "
                f"{expected_header[:16]}..."
            )
        if src.parent != hf_dir:
            shutil.copy2(src, hf_dir / shard)
        result["verified"][shard] = {"size": actual_size, "header_sha256": actual_header}
    return result


# ---------------------------------------------------------------------------
# Stage 4 — official convert (convert.py semantics, fp4, mp=1)
# ---------------------------------------------------------------------------

def convert_subset(hf_dir: Path, save_dir: Path, *, n_experts: int = 256) -> Path:
    try:
        import convert as official_convert
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"official convert.py import failed (pip install tqdm safetensors): {exc}"
        ) from exc
    save_dir.mkdir(parents=True, exist_ok=True)
    official_convert.main(str(hf_dir), str(save_dir), n_experts, 1, "fp4")
    converted = save_dir / "model0-mp1.safetensors"
    if not converted.is_file():
        raise RuntimeError("convert.py did not produce model0-mp1.safetensors")
    return converted


# ---------------------------------------------------------------------------
# Stage 5 — build the official reference (layer-0 subset)
# ---------------------------------------------------------------------------

_REFERENCE_MODEL: Any = None


def build_reference(assets_dir: Path, converted_path: Path) -> Any:
    """Build the official Transformer(n_layers=1) and load the converted subset.

    Memoized: the converted file is ~5 GB, so the model is built once.
    Fails closed if CUDA is unavailable or any required parameter is missing.
    """
    global _REFERENCE_MODEL
    if _REFERENCE_MODEL is not None:
        return _REFERENCE_MODEL
    import torch
    import model as official_model
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available for the official reference")
    kwargs = spec.build_subset_config(assets_dir / "inference" / "config.json")
    args = official_model.ModelArgs(**kwargs)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        transformer = official_model.Transformer(args)
    state = load_file(str(converted_path))
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    missing = sorted(missing)
    unexpected = sorted(unexpected)
    if missing:
        raise RuntimeError(
            f"reference load missing {len(missing)} params: {missing[:8]}"
        )
    # Trusted-reference dtype contract: the load must exercise the OFFICIAL
    # quantized kernel path (FP4 experts, FP8/BF16 dense), never an F.linear
    # fallback on accidentally-unquantized weights.
    import torch as _torch
    ffn0 = transformer.layers[0].ffn
    e0 = ffn0.experts[0]
    dtype_checks = {
        "expert_w1_fp4": e0.w1.weight.dtype == torch.float4_e2m1fn_x2,
        "expert_w2_fp4": e0.w2.weight.dtype == torch.float4_e2m1fn_x2,
        "expert_w3_fp4": e0.w3.weight.dtype == torch.float4_e2m1fn_x2,
        "gate_weight_bf16_or_f32": ffn0.gate.weight.dtype
        in (torch.bfloat16, torch.float32),
        "head_weight_f32": transformer.head.weight.dtype == torch.float32,
        "attn_dense_quantized": transformer.layers[0].attn.wq_b.weight.dtype
        in (torch.float8_e4m3fn, torch.bfloat16),
    }
    failed = [name for name, ok in dtype_checks.items() if not ok]
    if failed:
        raise RuntimeError(
            f"reference dtype contract failed: {failed} "
            f"(dtypes={ {k: str(getattr(v, 'dtype', None)) for k, v in [
                ('w1', e0.w1.weight), ('gate', ffn0.gate.weight),
                ('head', transformer.head.weight),
                ('wq_b', transformer.layers[0].attn.wq_b.weight)]} })"
        )
    _REFERENCE_MODEL = transformer
    return transformer


def describe_reference(model: Any, assets_dir: Path) -> dict[str, Any]:
    """JSON-safe metadata about the built reference (for evidence)."""
    import torch
    kwargs = spec.build_subset_config(assets_dir / "inference" / "config.json")
    state = model.state_dict()
    return {
        "model_args": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in kwargs.items()},
        "missing_params": [],
        "param_count": sum(p.numel() for p in model.parameters()),
        "state_dict_keys": len(state),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ---------------------------------------------------------------------------
# Stage 6 — name-driven boundary hooks
# ---------------------------------------------------------------------------

def _resolve_path(module: Any, dotted: str) -> Any:
    node = module
    for part in dotted.split("."):
        if part.isdigit():
            node = node[int(part)]
        else:
            node = getattr(node, part)
    return node


class _HookHandle:
    __slots__ = ("path", "kind", "handle")

    def __init__(self, path: str, kind: str, handle: Any):
        self.path, self.kind, self.handle = path, kind, handle


def register_boundary_hooks(model: Any, *, all_experts: bool = True) -> tuple[list[_HookHandle], dict[str, Any]]:
    """Register forward hooks for every spec boundary on an nn.Module tree.

    Returns (handles, registered) where registered maps key -> module path.
    Hooks append to ``_CAPTURE_STORE``: a dict key -> list of bounded records
    (one entry per forward invocation of that module).
    """
    import torch
    global _CAPTURE_STORE
    _CAPTURE_STORE = {}

    def _capture(path: str, kind: str):
        def hook(module, args, output):
            value = args[0] if kind == "input" else output
            _append_capture(path, value)
        return hook

    handles: list[_HookHandle] = []
    registered: dict[str, str] = {}
    for boundary in spec.BOUNDARIES:
        if boundary.module_path == "":
            continue
        try:
            node = _resolve_path(model, boundary.module_path)
        except Exception:
            if boundary.required:
                raise
            continue
        handle = node.register_forward_hook(_capture(boundary.key, boundary.kind))
        handles.append(_HookHandle(boundary.key, boundary.kind, handle))
        registered[boundary.key] = boundary.module_path
    if all_experts:
        for eid in range(256):
            path = f"layers.0.ffn.experts.{eid}"
            key = f"layer0_expert{eid}_out"
            try:
                node = _resolve_path(model, path)
            except Exception:
                continue
            handle = node.register_forward_hook(_capture(key, "output"))
            handles.append(_HookHandle(key, "output", handle))
            registered[key] = path
        shared = "layers.0.ffn.shared_experts"
        if _resolve_path(model, shared) is not None:
            handle = _resolve_path(model, shared).register_forward_hook(
                _capture("layer0_shared_expert_out", "output"))
            handles.append(_HookHandle("layer0_shared_expert_out", "output", handle))
            registered["layer0_shared_expert_out"] = shared
    return handles, registered


def _append_capture(key: str, value: Any) -> None:
    import torch
    entries: list[dict[str, Any]] = _CAPTURE_STORE.setdefault(key, [])
    if isinstance(value, tuple):
        for idx, element in enumerate(value):
            entries.append({"kind": f"tuple[{idx}]",
                            "record": spec.bounded_capture(element)})
    elif isinstance(value, torch.Tensor):
        entries.append({"kind": "tensor", "record": spec.bounded_capture(value)})
    else:
        entries.append({"kind": "other", "record": {"repr": repr(value)[:200]}})


def take_captures() -> dict[str, Any]:
    """Snapshot and clear the capture store into a JSON-safe dict."""
    out = {}
    for key, entries in _CAPTURE_STORE.items():
        out[key] = {"invocations": len(entries), "entries": entries}
    _CAPTURE_STORE.clear()
    return out


_CAPTURE_STORE: dict[str, list[dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Stage 7 — trace: prefill + greedy decode, bounded captures
# ---------------------------------------------------------------------------

def run_trace(model: Any, token_ids: list[int],
              *,
              max_decode_tokens: int = 2,
              max_seq_len: int = 4096) -> dict[str, Any]:
    import torch
    handles, registered = register_boundary_hooks(model)
    try:
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        torch.manual_seed(0)
        with torch.inference_mode():
            output_ids, logits, main_hidden = model.forward(token_tensor, 0)
        prefill = take_captures()
        prefill["output_ids"] = output_ids.tolist()
        prefill["logits_top10"] = _topk_logits(logits, 10)
        prefill["logits_hash"] = spec.tensor_sha256(logits)

        decode = {"steps": []}
        start_pos = len(token_ids)
        for _ in range(max_decode_tokens):
            if start_pos >= max_seq_len:
                break
            token_tensor = torch.tensor([[output_ids[0].item()]], dtype=torch.long,
                                        device="cuda")
            with torch.inference_mode():
                output_ids, logits, _ = model.forward(token_tensor, start_pos)
            step = take_captures()
            step["start_pos"] = start_pos
            step["input_id"] = int(token_tensor[0, 0].item())
            step["output_id"] = int(output_ids[0].item())
            step["logits_top5"] = _topk_logits(logits, 5)
            decode["steps"].append(step)
            start_pos += 1
        return {"prefill": prefill, "decode": decode,
                "total_decode_tokens": len(decode["steps"])}
    finally:
        for handle in handles:
            handle.handle.remove()


def _topk_logits(logits: Any, k: int) -> list[list[float]]:
    import torch
    topk = torch.topk(logits.float(), k=k, dim=-1)
    return [[int(i) for i in row] + [round(float(v), 6) for v in vs]
            for row, vs in zip(topk.indices.tolist(), topk.values.tolist())]


# ---------------------------------------------------------------------------
# Stage 8 — gates, verdict, evidence
# ---------------------------------------------------------------------------

def evaluate_gates(result: dict[str, Any]) -> dict[str, Any]:
    gates: dict[str, bool] = {}
    gates["shard_identity"] = bool(result["shards"]["verified"])
    gates["config_identity"] = bool(result["identity"]["files"])
    gates["tokenizer_identity"] = bool(result["identity"]["tokenizer"])
    gates["reference_loaded"] = result["reference"]["missing_params"] == []
    gates["boundary_coverage"] = _boundary_coverage_ok(result["trace"])
    gates["no_nan"] = _no_nan_ok(result["trace"])
    if result.get("artifact"):
        gates["artifact_complete"] = bool(result["artifact"]["artifact_complete"])
    failures = [name for name, ok in gates.items() if not ok]
    verdict = "ACCEPT_TRACE_GENERATED" if not failures else "INVALID_EXPERIMENT"
    return {"verdict": verdict, "gates": gates,
            "first_failing_gate": failures[0] if failures else None}


def _boundary_coverage_ok(trace: dict[str, Any]) -> bool:
    required = [b.key for b in spec.BOUNDARIES if b.required and b.module_path]
    captured = set(trace["prefill"].keys())
    return all(key in captured for key in required)


def _no_nan_ok(trace: dict[str, Any]) -> bool:
    phases: list[dict[str, Any]] = [trace.get("prefill") or {}]
    for step in (trace.get("decode") or {}).get("steps", []):
        phases.append(step)
    for phase in phases:
        for key, bucket in phase.items():
            if not isinstance(bucket, dict) or "entries" not in bucket:
                continue
            for entry in bucket["entries"]:
                record = entry.get("record") or {}
                if record.get("nan_count"):
                    return False
    return True


def write_evidence(result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema": "ds5-trace-evidence/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": OFFICIAL_REPOSITORY,
        "revision": OFFICIAL_REVISION,
        "probe": result["probe"],
        "identity": result["identity"],
        "prompt": result["prompt"],
        "reference": result["reference"],
        "trace": result["trace"],
    }
    evidence_path = out_dir / "ds5-trace-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    manifest = {
        "evidence_file": evidence_path.name,
        "evidence_sha256": evidence_sha,
        "repository_commit": os.environ.get("DSV4_RUNTIME_COMMIT", "unknown"),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_count": 1,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {"artifact_complete": True, "evidence_sha256": evidence_sha,
            "manifest_path": str(manifest_path)}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    work = Path(os.environ.get("DS5_WORK_DIR", "/kaggle/working/ds5"))
    assets = Path(os.environ.get("DS5_ASSETS_DIR",
                                 str(Path(__file__).resolve().parent)))
    out_dir = Path(os.environ.get("DS5_OUT_DIR", "/kaggle/working/ds5-evidence"))
    result: dict[str, Any] = {"verdict": "INVALID_EXPERIMENT"}
    gates: dict[str, bool] = {}
    result["deps"] = bootstrap_deps()
    try:
        result["probe"] = run_probe()
        gates["probe_ok"] = not result["probe"].get("first_failing_requirement")
        result["identity"] = verify_identity(assets)
        gates["config_identity"] = bool(result["identity"]["files"])
        gates["tokenizer_identity"] = bool(result["identity"]["tokenizer"])
        result["prompt"] = encode_canonical_prompt()
        result["shards"] = ensure_subset_shards(work)
        gates["shard_identity"] = bool(result["shards"]["verified"])
        converted = convert_subset(work / "hf-subset", work / "converted")
        model = build_reference(assets, converted)
        gates["reference_loaded"] = True
        result["reference"] = describe_reference(model, assets)
        result["trace"] = run_trace(model, result["prompt"]["token_ids"])
        gates["boundary_coverage"] = _boundary_coverage_ok(result["trace"])
        gates["no_nan"] = _no_nan_ok(result["trace"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    # Evidence is written BEFORE the verdict so an artifact failure can never
    # be masked by a pass verdict.
    result["artifact"] = write_evidence(result, out_dir)
    gates["artifact_complete"] = result["artifact"]["artifact_complete"]
    result["gates"] = gates
    failures = [name for name, ok in gates.items() if not ok]
    result["first_failing_gate"] = failures[0] if failures else None
    result["verdict"] = (
        "ACCEPT_TRACE_GENERATED"
        if not failures and not result.get("error")
        else "INVALID_EXPERIMENT"
    )
    (out_dir / "ds5-verdict.json").write_text(
        json.dumps({"verdict": result["verdict"], "gates": gates,
                    "first_failing_gate": result.get("first_failing_gate"),
                    "error": result.get("error")}, indent=1), encoding="utf-8")
    print("VERDICT:", result["verdict"], result.get("first_failing_gate"))
    return 0 if result["verdict"].startswith("ACCEPT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
