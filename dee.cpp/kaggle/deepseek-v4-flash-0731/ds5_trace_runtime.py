"""DS5: trusted official-reference trace generation (layer-0 subset, Kaggle).

Runs the OFFICIAL DeepSeek-V4 inference stack for a bounded single-layer
reference -- embedding + layer 0 (hash-routed, pure sliding-window:
``compress_ratios[0] == 0``) + final norm/head -- on the canonical prompt,
capturing bounded traces at every official boundary.

Source discipline (mirrors the sealed DS9 kernel): the kernel payload contains
ONLY this harness + kernel-metadata. At runtime the harness clones the pinned
repository commit into /kaggle/temp/dsv4-source, verifies the harness and
module SHA-256s against ``harness-identity-ds5.json``, and imports the official
inference stack (``scripts.deepseek_v4_encoding``, ``scripts.deepseek_v4_trace_spec``,
``inference/model.py``, ``inference/kernel.py``, ``inference/convert.py``)
from the checked-out tree.

Checkpoint footprint (pinned subset, ~5.7 GB, from the 48-shard checkpoint):

- model-00001-of-00048.safetensors  embed.weight                    1.06 GB
- model-00002-of-00048.safetensors  layer 0 (dense + 256 routed + shared) 3.57 GB
- model-00045-of-00048.safetensors  norm.weight, head.weight, hc_head_* 1.06 GB

Every blocker (missing shards, tilelang SM75 failure, load failure) is
reported with the FIRST failing gate; evidence is written before the verdict
so an artifact failure can never be masked by a pass verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

REPOSITORY = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
ROOT = Path("/kaggle/temp/dsv4-source")
DEE = ROOT / "dee.cpp"
RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
EVIDENCE = Path(f"/kaggle/working/dsv4-ds5-evidence-{RUN_ID}")
IDENTITY_RELATIVE = Path(
    "dee.cpp/kaggle/deepseek-v4-flash-0731/harness-identity-ds5.json")
HARNESS_RELATIVE = Path(
    "dee.cpp/kaggle/deepseek-v4-flash-0731/ds5_trace_runtime.py")
MODULE_RELATIVES = {
    "encoding": Path("dee.cpp/scripts/deepseek_v4_encoding.py"),
    "trace_spec": Path("dee.cpp/scripts/deepseek_v4_trace_spec.py"),
    "support": Path("dee.cpp/scripts/deepseek_v4_support.py"),
}
INFERENCE_DIR = Path(
    "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/official-source/inference")
OFFICIAL_SOURCE_DIR = Path(
    "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/official-source")
ASSETS_DIR = ROOT / OFFICIAL_SOURCE_DIR

OFFICIAL_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = REV

# Pinned subset manifest: shard -> (size_bytes, header_sha256, tensor_count)
#
# size_bytes is the FULL physical file size = 8 (length prefix) + header_size +
# max(data_offsets[1]). The v3 run pinned only max(data_offsets[1]) (the data
# extent), so the Range download stopped 96/172240/400 bytes short and the
# official convert.py failed with ``incomplete metadata, file not fully
# covered``. Sizes below were re-derived from the live official headers and
# cross-checked against the HF Content-Length of the pinned revision.
SUBSET_MANIFEST = {
    "model-00001-of-00048.safetensors": (
        1059061856, "ab23277543e81504fc621596b78b61db207fef4e9d1b522c09a543127220d261", 1,
    ),
    "model-00002-of-00048.safetensors": (
        3566321192, "8c5453415265f75d1b5afeeafbbb92fe20b3ccf4ce1ff5c55db932b28134ec38", 1565,
    ),
    "model-00045-of-00048.safetensors": (
        1059332516, "4e0d9e8684cf49832e97f565708d518012623a6881b2e392a21c0d16c23f1c65", 5,
    ),
}

MAX_DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_CHUNK = 8 << 20  # 8 MiB range requests

# Lazy module accessors: the scripts modules become importable only after
# _acquire_source() puts the cloned repo on sys.path.
_ENC: Any = None
_SPEC: Any = None


def _enc() -> Any:
    global _ENC
    if _ENC is None:
        from scripts import deepseek_v4_encoding as module  # type: ignore
        _ENC = module
    return _ENC


def _spec() -> Any:
    global _SPEC
    if _SPEC is None:
        from scripts import deepseek_v4_trace_spec as module  # type: ignore
        _SPEC = module
    return _SPEC


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Source acquisition + identity (repo-clone pattern, mirrors DS9)
# ---------------------------------------------------------------------------

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
        raise RuntimeError(f"missing DS5 harness identity {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("model_revision") != REV:
        raise RuntimeError({"identity_model_revision": identity.get("model_revision"),
                            "expected": REV})
    if sorted(identity.get("shards", [])) != sorted(SUBSET_MANIFEST):
        raise RuntimeError({"identity_shards": identity.get("shards"),
                            "expected": sorted(SUBSET_MANIFEST)})
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
    sys.path.insert(0, str(ROOT / INFERENCE_DIR))
    print("pinned commit", checked_out, flush=True)
    print("harness sha", running_sha, flush=True)
    return identity


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

    def _have(name: str) -> bool:
        try:
            _md.version(name)
            return True
        except Exception:
            return False

    def _have_ok(spec: str) -> bool:
        name = spec.split("==")[0].split("<=")[0]
        if not _have(name):
            return False
        version = _md.version(name)
        if "<=" in spec and version > spec.split("<=")[1]:
            return False  # installed, but too new for this pin
        if "==" in spec and version != spec.split("==")[1]:
            return False
        return True

    wanted = ["tqdm", "fast_hadamard_transform",
              "apache-tvm-ffi<=0.1.9", "tilelang==0.1.8"]
    missing = [pkg for pkg in wanted if not _have_ok(pkg)]
    results: dict[str, bool] = {}
    # Install apache-tvm-ffi pin FIRST so tilelang resolves against it.
    # tilelang 0.1.8 is incompatible with apache-tvm-ffi >= 0.1.10 on
    # Python 3.12 (tvm_ffi registry setattr '__dict__' crash at import);
    # the compatible ceiling is apache-tvm-ffi <= 0.1.9.
    ordered = [p for p in wanted if p.startswith("apache-tvm-ffi")] + \
              [p for p in wanted if not p.startswith("apache-tvm-ffi")]
    for pkg in ordered:
        if pkg not in missing:
            results[pkg] = True
            continue
        run = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            capture_output=True, text=True, timeout=900,
        )
        results[pkg] = run.returncode == 0
        if not results[pkg]:
            # surface the pip error tail for remote diagnosis
            tail = (run.stderr or "").strip().splitlines()[-4:]
            results[f"{pkg}_error"] = " ".join(tail)
    return {"missing_before": missing, "install_ok": results,
            "all_present": all(results.get(p, False) for p in wanted)}


# ---------------------------------------------------------------------------
# Stage 0 -- environment probe
# ---------------------------------------------------------------------------

def run_probe() -> dict[str, Any]:
    import importlib.metadata as _md
    probe = _spec().EnvProbe()
    try:
        import torch
        probe.torch_version = torch.__version__
        probe.cuda_available = torch.cuda.is_available()
        if probe.cuda_available:
            probe.cuda_device = torch.cuda.get_device_name(0)
            probe.sm_arch = _spec().parse_sm_arch(torch.cuda.get_device_properties(0))
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
        probe.disk_free_gib = shutil.disk_usage(
            "/kaggle" if Path("/kaggle").exists() else "/tmp").free / (1 << 30)
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
    env = os.environ.get("DSV4_CKPT_DIR")
    if env and (Path(env) / "model.safetensors.index.json").is_file():
        return Path(env)
    if Path("/kaggle/input").exists():
        for candidate in Path("/kaggle/input").glob("*"):
            if (candidate / "model.safetensors.index.json").is_file():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Stage 1 -- identity
# ---------------------------------------------------------------------------

def verify_identity(assets_dir: Path) -> dict[str, Any]:
    files = _spec().verify_pinned_files(assets_dir)
    tokenizer_hashes = _enc().verify_tokenizer_assets()
    return {"files": files, "tokenizer": tokenizer_hashes,
            "revision": OFFICIAL_REVISION}


# ---------------------------------------------------------------------------
# Stage 2 -- prompt encode
# ---------------------------------------------------------------------------

def encode_canonical_prompt() -> dict[str, Any]:
    messages = [{"role": "user", "content": _spec().CANONICAL_PROMPT}]
    text, ids = _enc().encode_message_ids(
        messages, thinking_mode=_spec().CANONICAL_THINKING_MODE)
    return {"prompt": _spec().CANONICAL_PROMPT,
            "mode": _spec().CANONICAL_THINKING_MODE,
            "text": text, "token_ids": ids, "n_tokens": len(ids),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


# ---------------------------------------------------------------------------
# Stage 3 -- shard acquisition + verification
# ---------------------------------------------------------------------------

def _header_sha256(path: Path) -> tuple[str, int, int]:
    """Return (header_sha256, declared_tensor_count, declared_data_end).

    ``declared_data_end`` is max(data_offsets[1]) over all tensors; the full
    physical size must be 8 (length prefix) + header_size + declared_data_end.
    A mismatch means the file is truncated or the pin is wrong, and is a
    fail-closed blocker (v3 pinned only the data extent and silently
    truncated the download).
    """
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    count = 0
    data_end = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        count += 1
        data_end = max(data_end, entry["data_offsets"][1])
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, count, data_end


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
    hf_dir = work_dir / "hf-subset"
    hf_dir.mkdir(parents=True, exist_ok=True)
    mounted = _find_mounted_checkpoint()
    result: dict[str, Any] = {"mode": "download", "verified": {}}
    for shard, (expected_size, expected_header, _count) in SUBSET_MANIFEST.items():
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
                f"{shard}: size {actual_size} != expected {expected_size}")
        actual_header, actual_count, data_end = _header_sha256(src)
        if actual_header != expected_header:
            raise RuntimeError(
                f"{shard}: header sha256 {actual_header[:16]}... != expected "
                f"{expected_header[:16]}...")
        if actual_count != _count:
            raise RuntimeError(
                f"{shard}: tensor count {actual_count} != expected {_count}")
        # Fail closed on physical coverage: 8 + header_size + data extent must
        # equal the exact file size (v3 truncated files by omitting header
        # overhead from the pinned size).
        with src.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
        covered = 8 + header_size + data_end
        if covered != actual_size:
            raise RuntimeError(
                f"{shard}: declared coverage {covered} != size {actual_size} "
                f"(truncated or pin error)")
        if src.parent != hf_dir:
            shutil.copy2(src, hf_dir / shard)
        result["verified"][shard] = {"size": actual_size,
                                     "header_sha256": actual_header,
                                     "tensor_count": actual_count,
                                     "coverage_bytes": covered}
    return result


# ---------------------------------------------------------------------------
# Stage 4 -- official convert (convert.py semantics, fp4, mp=1)
# ---------------------------------------------------------------------------

def convert_subset(hf_dir: Path, save_dir: Path, *, n_experts: int = 256) -> Path:
    try:
        import convert as official_convert  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"official convert.py import failed "
            f"(pip install tqdm safetensors): {exc}") from exc
    save_dir.mkdir(parents=True, exist_ok=True)
    official_convert.main(str(hf_dir), str(save_dir), n_experts, 1, "fp4")
    converted = save_dir / "model0-mp1.safetensors"
    if not converted.is_file():
        raise RuntimeError("convert.py did not produce model0-mp1.safetensors")
    return converted


# ---------------------------------------------------------------------------
# Stage 5 -- build the official reference (layer-0 subset)
# ---------------------------------------------------------------------------

_REFERENCE_MODEL: Any = None


def build_reference(assets_dir: Path, converted_path: Path) -> Any:
    """Build the official Transformer(n_layers=1) and load the converted subset.

    Memoized (the converted file is ~5 GB). Fails closed if CUDA is
    unavailable, any parameter is missing, or the loaded dtypes do not match
    the official quantized contract.
    """
    global _REFERENCE_MODEL
    if _REFERENCE_MODEL is not None:
        return _REFERENCE_MODEL
    import torch
    import model as official_model  # type: ignore
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available for the official reference")
    kwargs = _spec().build_subset_config(assets_dir / "inference" / "config.json")
    args = official_model.ModelArgs(**kwargs)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        transformer = official_model.Transformer(args)
    state = load_file(str(converted_path))
    missing, _ = transformer.load_state_dict(state, strict=False)
    missing = sorted(missing)
    if missing:
        raise RuntimeError(
            f"reference load missing {len(missing)} params: {missing[:8]}")
    # Trusted-reference dtype contract: the load must exercise the OFFICIAL
    # quantized kernel path (FP4 experts, FP8/BF16 dense), never an F.linear
    # fallback on accidentally-unquantized weights.
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
            f"reference dtype contract failed: {failed}")
    _REFERENCE_MODEL = transformer
    return transformer


def describe_reference(model: Any, assets_dir: Path) -> dict[str, Any]:
    import torch
    kwargs = _spec().build_subset_config(assets_dir / "inference" / "config.json")
    return {
        "model_args": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in kwargs.items()},
        "param_count": sum(p.numel() for p in model.parameters()),
        "state_dict_keys": len(model.state_dict()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ---------------------------------------------------------------------------
# Stage 6 -- name-driven boundary hooks
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


_CAPTURE_STORE: dict[str, list[dict[str, Any]]] = {}


def register_boundary_hooks(model: Any, *, all_experts: bool = True) -> tuple[list[_HookHandle], dict[str, Any]]:
    """Register forward hooks for every spec boundary on an nn.Module tree."""
    global _CAPTURE_STORE
    _CAPTURE_STORE = {}

    def _capture(path: str, kind: str):
        def hook(module, args, output):
            value = args[0] if kind == "input" else output
            _append_capture(path, value)
        return hook

    handles: list[_HookHandle] = []
    registered: dict[str, str] = {}
    for boundary in _spec().BOUNDARIES:
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
                            "record": _spec().bounded_capture(element)})
    elif isinstance(value, torch.Tensor):
        entries.append({"kind": "tensor", "record": _spec().bounded_capture(value)})
    else:
        entries.append({"kind": "other", "record": {"repr": repr(value)[:200]}})


def take_captures() -> dict[str, Any]:
    out = dict(_CAPTURE_STORE)
    _CAPTURE_STORE.clear()
    return out


# ---------------------------------------------------------------------------
# Stage 7 -- trace: prefill + greedy decode, bounded captures
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
            output_ids, logits, _ = model.forward(token_tensor, 0)
        prefill = take_captures()
        prefill["output_ids"] = output_ids.tolist()
        prefill["logits_top10"] = _topk_logits(logits, 10)
        prefill["logits_hash"] = _spec().tensor_sha256(logits)

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
# Stage 8 -- gates
# ---------------------------------------------------------------------------

def _boundary_coverage_ok(trace: dict[str, Any]) -> bool:
    required = [b.key for b in _spec().BOUNDARIES if b.required and b.module_path]
    captured = set(trace.get("prefill", {}).keys())
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
    return {"verdict": "ACCEPT_TRACE_GENERATED" if not failures else "INVALID_EXPERIMENT",
            "gates": gates,
            "first_failing_gate": failures[0] if failures else None}


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def write_evidence(result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema": "ds5-trace-evidence/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": OFFICIAL_REPOSITORY,
        "revision": OFFICIAL_REVISION,
        "identity": result.get("identity"),
        "deps": result.get("deps"),
        "probe": result.get("probe"),
        "prompt": result.get("prompt"),
        "reference": result.get("reference"),
        "trace": result.get("trace"),
        "gates": result.get("gates"),
        "verdict": result.get("verdict"),
        "first_failing_gate": result.get("first_failing_gate"),
        "error": result.get("error"),
        "error_traceback": result.get("error_traceback"),
    }
    evidence_path = out_dir / "ds5-trace-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    evidence_sha = sha256_file(evidence_path)
    manifest = {
        "evidence_file": evidence_path.name,
        "evidence_sha256": evidence_sha,
        "repository_commit": result.get("identity", {}).get("repository_commit"),
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "artifact_count": 1,
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {"artifact_complete": True, "evidence_sha256": evidence_sha,
            "manifest_path": str(manifest_path)}


def copy_evidence_modules(out_dir: Path) -> None:
    for key, rel in MODULE_RELATIVES.items():
        src = ROOT / rel
        if src.is_file():
            shutil.copy2(src, out_dir / f"deepseek_v5_module_{key}.py")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    work = Path(os.environ.get("DS5_WORK_DIR", "/kaggle/working/ds5"))
    out_dir = EVIDENCE
    result: dict[str, Any] = {"verdict": "INVALID_EXPERIMENT"}
    gates: dict[str, bool] = {}
    try:
        identity = _acquire_source()
        result["identity"] = identity
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["error_traceback"] = traceback.format_exc()
        result["identity"] = {}
        result["verdict"] = "BLOCKED"
        result["first_failing_gate"] = "source_identity"
        result["artifact"] = write_evidence(result, out_dir)
        write_json(out_dir / "ds5-verdict.json",
                   {"verdict": result["verdict"],
                    "first_failing_gate": result.get("first_failing_gate"),
                    "error": result.get("error")})
        print("VERDICT:", result["verdict"], result.get("first_failing_gate"))
        return 1

    result["deps"] = bootstrap_deps()
    try:
        result["probe"] = run_probe()
        gates["probe_ok"] = not result["probe"].get("first_failing_requirement")
        result["identity"] = verify_identity(ASSETS_DIR)
        gates["config_identity"] = bool(result["identity"]["files"])
        gates["tokenizer_identity"] = bool(result["identity"]["tokenizer"])
        result["prompt"] = encode_canonical_prompt()
        result["shards"] = ensure_subset_shards(work)
        gates["shard_identity"] = bool(result["shards"]["verified"])
        converted = convert_subset(work / "hf-subset", work / "converted")
        model = build_reference(ASSETS_DIR, converted)
        gates["reference_loaded"] = True
        result["reference"] = describe_reference(model, ASSETS_DIR)
        result["trace"] = run_trace(model, result["prompt"]["token_ids"])
        gates["boundary_coverage"] = _boundary_coverage_ok(result["trace"])
        gates["no_nan"] = _no_nan_ok(result["trace"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["error_traceback"] = traceback.format_exc()
    result["gates"] = gates
    failures = [name for name, ok in gates.items() if not ok]
    result["first_failing_gate"] = failures[0] if failures else None
    result["verdict"] = (
        "ACCEPT_TRACE_GENERATED"
        if not failures and not result.get("error")
        else "INVALID_EXPERIMENT")
    # Evidence is written BEFORE the final verdict line and always includes
    # the first failing gate; an artifact failure can never be masked.
    result["artifact"] = write_evidence(result, out_dir)
    copy_evidence_modules(out_dir)
    shutil.copy2(ROOT / IDENTITY_RELATIVE, out_dir / "harness-identity-ds5.json")
    write_json(out_dir / "ds5-verdict.json",
               {"verdict": result["verdict"], "gates": gates,
                "first_failing_gate": result.get("first_failing_gate"),
                "error": result.get("error")})
    print("VERDICT:", result["verdict"], result.get("first_failing_gate"))
    if result.get("error_traceback"):
        print("--- ERROR TRACEBACK ---", flush=True)
        print(result["error_traceback"][-6000:], flush=True)
        print("--- END TRACEBACK ---", flush=True)
    return 0 if result["verdict"].startswith("ACCEPT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
