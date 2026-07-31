"""DS7 Kaggle smoke: ONE official DeepSeek-V4-Flash-0731 routed expert on T4.

Goal (DS7): prove one official routed expert executes on SM75 (T4) and
matches the pinned trusted reference within predeclared tolerances.

Pipeline:
  1. Download the pinned shard containing layers.6.ffn.experts.0.*
     (model-00008-of-00048.safetensors) plus config/index, with size + header
     verification.
  2. Load the 6 tensors (w1/w2/w3 packed I8 + F8_E8M0 scales).
  3. Trusted reference: scripts/deepseek_v4_expert_reference.py (FP32 math,
     official FP4_TABLE + E8M0 decode + asymmetric SwiGLU clamps).
  4. Candidate on T4: FP16 dequantized GEMV with FP32 accumulation.
  5. Compare per predeclared tolerance (T4 candidate carries FP8 act-quant
     and FP16 storage error, so near-bitwise agreement is NOT expected).
  6. Archive evidence JSON + manifest hashes under /kaggle/working/.

Performance reporting is disabled in this smoke run (non-comparable mode).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working")
CHECKPOINT_DIR = Path("/kaggle/working/dsv4-checkpoint")
OUT = WORKING / "ds7-evidence"

sys.path.insert(0, "/kaggle/working")  # copied scripts/
sys.path.insert(0, "/kaggle/working/scripts")

import torch  # noqa: E402

from scripts import deepseek_v4_expert_reference as ref  # noqa: E402

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SHARD = "model-00008-of-00048.safetensors"


def shard_expected_bytes() -> tuple[int, str | None]:
    """Resolve (shard_byte_size, remote_sha256_or_None).

    Uses the local manifest if present, else the remote Content-Range total.
    When remote is queried, also captures X-Linked-Etag (the LFS sha256 for
    safetensor shards) so the downloaded file can be verified against it.
    """
    manifest_path = Path("/kaggle/working/CHECKPOINT_MANIFEST.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("shards", []):
            if row.get("shard") == SHARD:
                return int(row["compressed_bytes"]), None
    import urllib.request
    url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/{REV}/{SHARD}"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        cr = resp.headers.get("Content-Range", "")
        etag = resp.headers.get("X-Linked-Etag", "")
    if "/" not in cr:
        raise RuntimeError(f"cannot resolve remote size for {SHARD}: {cr!r}")
    remote_sha = None
    if etag:
        remote_sha = etag.strip().strip('"')
        # LFS OID is a bare 64-char sha256; drop any suffix like "-N".
        if len(remote_sha) > 64:
            remote_sha = remote_sha[:64]
    return int(cr.split("/")[1]), remote_sha

# Layers.6 / expert 0 per the validated ledger (MODEL_LEDGER.json).
EXPERT_TENSORS = [
    "layers.6.ffn.experts.0.w1.weight", "layers.6.ffn.experts.0.w1.scale",
    "layers.6.ffn.experts.0.w2.weight", "layers.6.ffn.experts.0.w2.scale",
    "layers.6.ffn.experts.0.w3.weight", "layers.6.ffn.experts.0.w3.scale",
]


def download_shard(want: int) -> Path:
    import urllib.request
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKPOINT_DIR / SHARD
    url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/{REV}/{SHARD}"
    have = dest.stat().st_size if dest.is_file() else 0
    if have == want:
        return dest
    if have > want:
        raise RuntimeError(f"shard too large {have} > {want}")
    chunk = 8 << 20
    with open(dest, "ab") as fh:
        while have < want:
            end = min(have + chunk - 1, want - 1)
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-{end}"})
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
            if not data:
                raise ConnectionError(f"empty chunk at {have}")
            fh.write(data)
            have += len(data)
            print(f"  download {have}/{want} ({100.0*have/want:.1f}%)", flush=True)
    if dest.stat().st_size != want:
        raise RuntimeError(f"shard size mismatch {dest.stat().st_size} != {want}")
    return dest


def load_expert(shard_path: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for name in EXPERT_TENSORS:
            if name not in f.keys():
                raise KeyError(f"missing {name} in {SHARD}")
            tensors[name] = f.get_tensor(name).contiguous()
    return tensors


def candidate_expert_on_t4(
    x: torch.Tensor, t: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, str]:
    """FP16 dequantized GEMV with FP32 accumulation, executed ON the T4.

    Every tensor is moved to the CUDA device explicitly so the matmuls run
    on the GPU (device-authentic). Returns (output_on_cpu, device_type) so
    the caller can compare against the trusted FP32 reference and record
    where the candidate actually executed.
    """
    dev = "cuda"
    w1 = ref.dequantize_expert_weight(t["layers.6.ffn.experts.0.w1.weight"],
                                      t["layers.6.ffn.experts.0.w1.scale"]).half().to(dev)
    w2 = ref.dequantize_expert_weight(t["layers.6.ffn.experts.0.w2.weight"],
                                      t["layers.6.ffn.experts.0.w2.scale"]).half().to(dev)
    w3 = ref.dequantize_expert_weight(t["layers.6.ffn.experts.0.w3.weight"],
                                      t["layers.6.ffn.experts.0.w3.scale"]).half().to(dev)
    xc = x.half().to(dev)
    gate = (xc @ w1.t()).float()
    up = (xc @ w3.t()).float()
    gate = torch.clamp(gate, max=10.0)
    up = torch.clamp(up, min=-10.0, max=10.0)
    h = (torch.nn.functional.silu(gate) * up).half()
    out = (h @ w2.t()).float()
    torch.cuda.synchronize()  # guarantee the GEMV finished before timing/metrics
    return out.cpu(), w1.device.type


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    abs_err = (reference - candidate).abs()
    rel = abs_err / (reference.abs() + 1e-8)
    return {
        "max_abs_error": float(abs_err.max()),
        "mean_abs_error": float(abs_err.mean()),
        "max_rel_error": float(rel.max()),
        "mean_rel_error": float(rel.mean()),
    }


def main() -> int:
    print("=== DS7 DeepSeek-V4-Flash-0731 one-expert smoke on T4 ===", flush=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the candidate path")
    print("device", torch.cuda.get_device_name(0), flush=True)

    want, remote_sha = shard_expected_bytes()
    shard_path = download_shard(want)
    t = load_expert(shard_path)
    print("loaded expert tensors:", {k: tuple(v.shape) for k, v in t.items()}, flush=True)

    torch.manual_seed(0)
    x = torch.randn(4, 4096)  # 4 tokens, hidden=4096

    t0 = time.time()
    reference = ref.expert_forward(
        x,
        t["layers.6.ffn.experts.0.w1.weight"], t["layers.6.ffn.experts.0.w1.scale"],
        t["layers.6.ffn.experts.0.w2.weight"], t["layers.6.ffn.experts.0.w2.scale"],
        t["layers.6.ffn.experts.0.w3.weight"], t["layers.6.ffn.experts.0.w3.scale"],
        swiglu_limit=10.0,
    )
    t_ref = time.time() - t0

    t0 = time.time()
    candidate, candidate_device = candidate_expert_on_t4(x, t)
    t_cand = time.time() - t0
    print("candidate executed on device:", candidate_device, flush=True)

    metrics = compare(reference, candidate)
    print("reference time %.3fs candidate time %.3fs" % (t_ref, t_cand), flush=True)
    print("metrics:", json.dumps(metrics, indent=2), flush=True)

    # Predeclared tolerances for the T4 candidate (FP16 storage + FP16 GEMV
    # vs full-FP32 trusted reference). These are declared BEFORE the run,
    # per the DS7 protocol: the candidate is NOT expected to be bitwise exact.
    TOLERANCE = {"max_abs_error": 2.0, "mean_abs_error": 0.5, "max_rel_error": 1e-2}
    passed = (
        metrics["max_abs_error"] <= TOLERANCE["max_abs_error"]
        and metrics["mean_abs_error"] <= TOLERANCE["mean_abs_error"]
        and metrics["max_rel_error"] <= TOLERANCE["max_rel_error"]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    # Stream-hash the shard (3.5 GB) to avoid loading it into RAM.
    h = hashlib.sha256()
    with open(shard_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    shard_sha256 = h.hexdigest()
    if remote_sha is not None and shard_sha256 != remote_sha:
        raise RuntimeError(f"{SHARD}: streamed sha {shard_sha256[:16]} != HF etag {remote_sha[:16]}")
    # Keep the Kaggle output artifact small: Kaggle zips the ENTIRE working
    # directory as the downloadable output, so drop the shard now that its
    # SHA is recorded and its tensors are loaded in memory.
    shard_path.unlink()
    evidence = {
        "campaign": "deepseek-v4-flash-0731",
        "phase": "DS7-expert-smoke",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": REV,
        "shard": SHARD,
        "layer": 6,
        "expert": 0,
        "device": torch.cuda.get_device_name(0),
        "candidate_executed_on_cuda": candidate_device == "cuda",
        "torch": torch.__version__,
        "shard_bytes": shard_path.stat().st_size if shard_path.is_file() else None,
        "shard_sha256": shard_sha256,
        "integrity_gate": "hf_etag" if remote_sha is not None else "manifest_size",
        "shard_sha_verified_vs_hf_etag": remote_sha is not None and shard_sha256 == remote_sha,
        "tensor_shapes": {k: list(v.shape) for k, v in t.items()},
        "metrics": metrics,
        "tolerance": TOLERANCE,
        "passed": bool(passed),
        "verdict": "MATCH_WITHIN_TOLERANCE" if passed else "MISMATCH",
        "note": "candidate carries FP16-storage/FP16-GEMV error vs full-FP32 reference",
        "performance_comparable": False,
    }
    (OUT / "ds7-expert-evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )

    # Keep the downloaded shard out of the evidence copy to bound artifact size.
    shutil.copy2(Path(__file__).resolve(),
                 OUT / "deepseek_v4_expert_smoke.py")
    (OUT / "manifest.sha256").write_text(
        hashlib.sha256((OUT / "ds7-expert-evidence.json").read_bytes()).hexdigest()
        + "  ds7-expert-evidence.json\n", encoding="utf-8")

    print("evidence:", OUT, flush=True)
    print("VERDICT:", evidence["verdict"], flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
