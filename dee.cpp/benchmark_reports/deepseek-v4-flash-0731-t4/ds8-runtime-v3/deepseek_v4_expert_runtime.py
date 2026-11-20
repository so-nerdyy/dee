#!/usr/bin/env python3
"""DS8 Kaggle harness: generalized official expert runtime + bounded cache.

Advances from the DS7 single-expert smoke to the DS8 milestone: execute the
routed + shared expert portions of COMPLETE official model layers on T4
through a bounded expert cache, validated against the trusted FP32 reference
with the PRE-DECLARED DS8 numerical contract.

Pipeline per selected layer (3, 20, 41 -> shards 00005, 00022, 00043):

  1. Bootstrap environment (Tesla T4 required).
  2. Clone the pinned branch; verify the DS8 harness identity sidecar
     (repository commit + SHA256 of the running/committed harness and of the
     6 trusted modules imported from the pinned tree).
  3. Download the layer's shard with size + header pin against the committed
     cached header (canonical, EOL-immune) + bounded CDN retry.
  4. Load the official router (gate.weight BF16 + gate.bias F32), the shared
     expert (F8_E4M3 + F8_E8M0 scales), and the 6 top-scoring routed experts
     selected by the official sqrtsoftplus router on the input corpus.
  5. Trusted reference (CPU FP32): full MoE FFN (weighted top-6 routed +
     shared) via scripts/deepseek_v4_moe_reference.py.
  6. Candidate on T4 (CUDA): FP16-expanded expert weights staged through
     DeepSeekExpertCache (cold then warm), FP16 GEMV with FP32 accumulation,
     official weight placement, plus the shared expert.
  7. Gate on the predeclared DS8 contract (max/mean abs, mean/p95/p99 rel,
     cosine, normalized RMSE, output-norm rel, near-zero exclusion).
  8. Cache behavior: cold (all misses) vs warm (all hits, no H2D), plus a
     bounded-budget eviction-pressure scenario (forced evictions + reloads
     stay numerically correct).
  9. Archive evidence + manifest hashes.

performance_comparable: false -- correctness milestone, not a throughput run.
No model TPS is claimed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

RUN_ID = "20260731T080000Z-dsv4-ds8-expert-runtime"
REPOSITORY = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
ROOT = Path("/kaggle/temp/dsv4-source")
EVIDENCE = Path(f"/kaggle/working/dsv4-ds8-evidence-{RUN_ID}")
ARCHIVE_BASE = EVIDENCE
DEE = ROOT / "dee.cpp"
IDENTITY_RELATIVE = Path(
    "dee.cpp/kaggle/deepseek-v4-flash-0731/harness-identity-ds8.json")
HARNESS_RELATIVE = Path(
    "dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_expert_runtime.py")
MODULE_RELATIVES = {
    "moe_reference": Path("dee.cpp/scripts/deepseek_v4_moe_reference.py"),
    "expert_reference": Path("dee.cpp/scripts/deepseek_v4_expert_reference.py"),
    "cache": Path("dee.cpp/scripts/deepseek_v4_cache.py"),
    "contract": Path("dee.cpp/scripts/deepseek_v4_contract.py"),
    "corpus": Path("dee.cpp/scripts/deepseek_v4_corpus.py"),
    "support": Path("dee.cpp/scripts/deepseek_v4_support.py"),
}

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
# early / middle / late MoE layers; each layer's tensors live in one shard
# (layer N -> model-000(N+2)).  All three are score-based (>= num_hash_layers).
LAYERS = (3, 20, 41)
SHARDS = {layer: f"model-000{layer + 2:02d}-of-00048.safetensors"
          for layer in LAYERS}
CACHED_HEADER_RELATIVE = Path(
    "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers")

HIDDEN = 4096
MOE_INTER = 2048
N_ROUTED = 256
TOPK = 6
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
N_TOKENS = 8

# Transient HTTP codes from HF's CDN that are safe to retry with backoff.
RETRYABLE_CODES = {429, 500, 502, 503, 504}
MAX_DOWNLOAD_ATTEMPTS = 6
RETRY_BACKOFF_SECONDS = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def run_logged(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        rc = process.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, command)
    return rc


def canonical_header_sha256(parsed: dict[str, Any]) -> str:
    canonical = json.dumps(parsed, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def downloaded_header_sha256(shard_path: Path) -> str:
    with open(shard_path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{shard_path}: truncated 8-byte prefix")
        hlen = struct.unpack("<Q", raw)[0]
        if hlen <= 0 or hlen > (1 << 31):
            raise ValueError(f"{shard_path}: implausible header length {hlen}")
        header_bytes = fh.read(hlen)
        if len(header_bytes) != hlen:
            raise ValueError(f"{shard_path}: truncated header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{shard_path}: malformed header JSON: {exc}") from exc
    return canonical_header_sha256(header)


def shard_expected_bytes(shard: str) -> int:
    import urllib.request
    url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
           f"resolve/{REV}/{shard}")
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        cr = resp.headers.get("Content-Range", "")
    if "/" not in cr:
        raise RuntimeError(f"cannot resolve remote size for {shard}: {cr!r}")
    return int(cr.split("/")[1])


def fetch_chunk_with_retries(req: Any) -> bytes:
    import urllib.error
    import urllib.request
    last_error: Exception | None = None
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                if r.status != 206:
                    raise RuntimeError(
                        f"server did not honor Range (status {r.status})")
                return r.read(8 << 20)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_CODES:
                raise
            last_error = exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
        print(f"  transient download error {type(last_error).__name__}: "
              f"{last_error}; retry {attempt + 1}/{MAX_DOWNLOAD_ATTEMPTS}",
              flush=True)
        time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
    raise ConnectionError(
        f"download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts: "
        f"{last_error!r}")


def download_shard(shard: str, checkpoint_dir: Path) -> Path:
    import urllib.request
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir / shard
    want = shard_expected_bytes(shard)
    have = dest.stat().st_size if dest.is_file() else 0
    if have == want:
        return dest
    if have > want:
        raise RuntimeError(f"shard too large {have} > {want}")
    url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
           f"resolve/{REV}/{shard}")
    chunk = 8 << 20
    with open(dest, "ab") as fh:
        while have < want:
            end = min(have + chunk - 1, want - 1)
            req = urllib.request.Request(url, headers={
                "Range": f"bytes={have}-{end}"})
            data = fetch_chunk_with_retries(req)
            if not data:
                raise ConnectionError(f"empty chunk at {have}")
            fh.write(data)
            have += len(data)
            print(f"  {shard} {have}/{want} ({100.0*have/want:.1f}%)",
                  flush=True)
    if dest.stat().st_size != want:
        raise RuntimeError(f"shard size mismatch {dest.stat().st_size} != "
                           f"{want}")
    return dest


def load_tensors_from_shard(shard_path: Path, names: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for name in names:
            if name not in f.keys():
                raise KeyError(f"missing {name} in {shard_path.name}")
            tensors[name] = f.get_tensor(name).contiguous()
    return tensors


def candidate_moe_on_t4(
    x_cpu: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    cache: Any,
    loader: Any,
    layer: int,
    fp16_payloads: dict[int, dict[str, torch.Tensor]],
    shared_payload: dict[str, torch.Tensor],
    shared_cache_key: int = -1,
) -> dict[str, torch.Tensor]:
    """FP16 cache-backed weighted MoE candidate, executed ON the T4.

    Matches the trusted reference math: routing weight applied to the
    intermediate before w2, asymmetric swiglu clamps, shared expert added
    unweighted, FP32 accumulation of the FP16 GEMVs.

    Warm hits use cache.get() (no re-stage, no H2D); misses stage through the
    bounded loader.  Returns {moe_output, shared_output, per_expert} on CPU.
    """
    dev = "cuda"
    n = x_cpu.shape[0]
    moe = torch.zeros(n, HIDDEN, dtype=torch.float32)
    per_expert: dict[int, torch.Tensor] = {}

    groups: dict[int, list[tuple[int, float]]] = {}
    for tok in range(n):
        for pos in range(TOPK):
            eid = int(expert_ids[tok, pos])
            w = float(routing_weights[tok, pos])
            if w == 0.0:
                continue
            groups.setdefault(eid, []).append((tok, w))

    for eid, pairs in groups.items():
        entry = cache.get(layer, eid)
        if entry is None:
            entry = loader.stage(layer, eid, fp16_payloads[eid],
                                 metadata={"expert_type": "routed"})
        loader.wait(entry)
        payload = entry.payload
        toks = [pair[0] for pair in pairs]
        ws = torch.tensor([[pair[1]] for pair in pairs],
                          dtype=torch.float32).reshape(-1, 1)
        xc = x_cpu[toks].half().to(dev)
        gate = (xc @ payload["w1.weight"].t()).float()
        up = (xc @ payload["w3.weight"].t()).float()
        gate = torch.clamp(gate, max=SWIGLU_LIMIT)
        up = torch.clamp(up, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        h = torch.nn.functional.silu(gate) * up
        h = ws.to(dev) * h  # routing weight applied before w2 (official)
        out = (h.half() @ payload["w2.weight"].t()).float().cpu()
        moe[toks] += out
        bucket = per_expert.setdefault(
            eid, torch.zeros(n, HIDDEN, dtype=torch.float32))
        bucket[toks] += out

    shared_entry = cache.get(layer, shared_cache_key)
    if shared_entry is None:
        shared_entry = loader.stage(layer, shared_cache_key, shared_payload,
                                    metadata={"expert_type": "shared"})
    loader.wait(shared_entry)
    sp = shared_entry.payload
    xc = x_cpu.half().to(dev)
    gate = torch.clamp((xc @ sp["w1.weight"].t()).float(), max=SWIGLU_LIMIT)
    up = torch.clamp((xc @ sp["w3.weight"].t()).float(),
                     min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    h = torch.nn.functional.silu(gate) * up
    shared_out = (h.half() @ sp["w2.weight"].t()).float().cpu()
    moe = moe + shared_out

    return {"moe_output": moe, "shared_output": shared_out,
            "per_expert": per_expert}


def main() -> int:
    print("=== DS8 DeepSeek-V4-Flash-0731 expert runtime on T4 ===", flush=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    fatal_error: dict[str, object] | None = None
    commit: str | None = None
    module_shas: dict[str, str] = {}
    layer_results: dict[int, dict[str, Any]] = {}
    running_sha = ""
    all_gates = False
    all_shared = False

    try:
        bootstrap = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [{
                "index": device,
                "name": torch.cuda.get_device_name(device),
                "memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            } for device in range(torch.cuda.device_count())],
        }
        write_json(EVIDENCE / "bootstrap-environment.json", bootstrap)
        if torch.cuda.device_count() < 1 or not all(
            "T4" in torch.cuda.get_device_name(device)
            for device in range(torch.cuda.device_count())
        ):
            raise RuntimeError(
                f"expected Tesla T4 topology, got {bootstrap['gpus']}")

        run_logged(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
             "safetensors==0.8.0"],
            EVIDENCE / "logs/pip-install.log", EVIDENCE,
        )

        if ROOT.exists():
            resolved_root = ROOT.resolve()
            if not str(resolved_root).startswith("/kaggle/temp/"):
                raise RuntimeError(
                    f"refusing to remove unexpected path {resolved_root}")
            shutil.rmtree(ROOT)
        subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch",
                        REPOSITORY, str(ROOT)], check=True)

        identity_path = ROOT / IDENTITY_RELATIVE
        if not identity_path.is_file():
            raise RuntimeError(f"missing DS8 harness identity "
                               f"{identity_path}")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("model_revision") != REV:
            raise RuntimeError({"identity_model_revision":
                                identity.get("model_revision"),
                                "expected": REV})
        if identity.get("shards") != sorted(SHARDS.values()):
            raise RuntimeError({"identity_shards": identity.get("shards"),
                                "expected": sorted(SHARDS.values())})
        expected_harness_commit = identity.get("repository_commit")
        if (not isinstance(expected_harness_commit, str)
                or len(expected_harness_commit) != 40
                or any(c not in "0123456789abcdef"
                       for c in expected_harness_commit)):
            raise RuntimeError({"repository_commit": expected_harness_commit,
                                "reason": "identity must pin a 40-char commit"})
        subprocess.run(["git", "checkout", "--quiet", expected_harness_commit],
                       cwd=ROOT, check=True)
        checked_out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if checked_out != expected_harness_commit:
            raise RuntimeError({"commit": checked_out,
                                "expected": expected_harness_commit})

        committed_harness = ROOT / HARNESS_RELATIVE
        committed_harness_sha = sha256_file(committed_harness)
        running_sha = sha256_file(Path(__file__).resolve())
        if committed_harness_sha != identity.get("harness_sha256"):
            raise RuntimeError({"committed_harness_sha256":
                                committed_harness_sha,
                                "expected": identity.get("harness_sha256")})
        if running_sha != identity.get("harness_sha256"):
            raise RuntimeError({"running_harness_sha256": running_sha,
                                "expected": identity.get("harness_sha256")})

        for key, rel in MODULE_RELATIVES.items():
            committed = ROOT / rel
            module_sha = sha256_file(committed)
            expected = identity.get("module_sha256", {}).get(key)
            if not expected or module_sha != expected:
                raise RuntimeError({f"{key}_sha256": module_sha,
                                    "expected": expected})
            module_shas[key] = module_sha
        commit = checked_out
        print("pinned commit", commit, flush=True)
        print("harness sha", running_sha, flush=True)
        print("module shas", json.dumps(module_shas), flush=True)

        sys.path.insert(0, str(ROOT / "dee.cpp"))
        from scripts import (deepseek_v4_cache as v4cache,  # noqa: E402
                             deepseek_v4_contract as v4contract,
                             deepseek_v4_corpus as v4corpus,
                             deepseek_v4_expert_reference as ds7,
                             deepseek_v4_moe_reference as moe,
                             deepseek_v4_support as v4support)
        # Run the pinned module self-tests from the checked-out tree.
        moe.main()
        ds7.main()

        # Build the corpus ONCE (deterministic) and route every layer on it.
        corpus_cases, corpus_meta = v4corpus.build_corpus(
            N_TOKENS, HIDDEN, base_seed=7,
            official_trace=Path("/kaggle/input/") / "missing.npz")
        print("corpus:", [name for name, _ in corpus_cases], flush=True)

        checkpoint_dir = Path("/kaggle/working/dsv4-checkpoint")
        for layer in LAYERS:
            shard = SHARDS[layer]
            print(f"\n=== layer {layer} ({shard}) ===", flush=True)
            layer_result: dict[str, Any] = {
                "layer": layer, "shard": shard, "expert_ids": [],
                "routing_weights": [], "per_case": [], "cache": {}, "shard_sha256": "",
                "integrity_gate": "header_pin",
            }
            shard_path = download_shard(shard, checkpoint_dir)
            cached_header_path = ROOT / CACHED_HEADER_RELATIVE / f"{shard}.json"
            if not cached_header_path.is_file():
                raise RuntimeError(
                    f"missing committed cached shard header "
                    f"{cached_header_path}")
            cached = json.loads(cached_header_path.read_text(encoding="utf-8"))
            expected_header_sha = canonical_header_sha256(cached)
            got_header_sha = downloaded_header_sha256(shard_path)
            if got_header_sha != expected_header_sha:
                raise RuntimeError({"downloaded_header_sha256": got_header_sha,
                                    "expected_committed_header_sha256":
                                    expected_header_sha})
            h = hashlib.sha256()
            with open(shard_path, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            layer_result["shard_sha256"] = h.hexdigest()

            # Router tensors (score layer): gate.weight BF16 + gate.bias F32.
            router_names = v4support.router_tensor_names(layer)
            router = load_tensors_from_shard(shard_path, router_names)
            gate_w = router[router_names[0]].float()
            gate_b = router[router_names[1]].float() \
                if len(router_names) > 1 else None
            # Shared expert tensors (F8_E4M3 + F8_E8M0 scales).
            shared_names = v4support.shared_expert_tensor_names(layer)
            shared_raw = load_tensors_from_shard(shard_path, shared_names)

            # Route EVERY corpus case up-front (cheap: just the gate matmul).
            # Different input distributions select different experts, so the
            # trusted per-case reference needs the packed weights of EVERY
            # selected expert -- v2 crashed with KeyError: <eid> because only
            # the first case's experts were loaded.  The candidate itself
            # runs on x_route (first case, "normal").  corpus_cases[i] is
            # (name, tensor), so the tensor is the SECOND element (v1 crashed
            # here by assigning the name string to x_route).
            _, x_route = corpus_cases[0]
            case_routes = []
            reference_union: set[int] = set()
            for case_name, x in corpus_cases:
                sc, ids_c, wts_c = ds7.router_scores(
                    x, gate_w, bias=gate_b, score_func="sqrtsoftplus",
                    topk=TOPK, route_scale=ROUTE_SCALE)
                case_routes.append({"case": case_name, "scores": sc,
                                    "expert_ids": ids_c,
                                    "routing_weights": wts_c})
                reference_union.update(int(e)
                                       for e in ids_c.flatten().tolist())
            x_scores = case_routes[0]["scores"]
            ids = case_routes[0]["expert_ids"]
            weights = case_routes[0]["routing_weights"]
            layer_result["expert_ids"] = ids.tolist()
            layer_result["routing_weights"] = weights.tolist()
            layer_result["router_max_abs_score"] = float(x_scores.abs().max())

            reference_selected = sorted(reference_union)
            candidate_selected = sorted(
                {int(e) for e in ids.flatten().tolist()})
            print("  reference expert union (%d):" % len(reference_selected),
                  reference_selected, flush=True)
            print("  candidate (x_route) experts:", candidate_selected,
                  flush=True)

            # Load packed FP4 weights for every reference-selected expert
            # (the trusted per-case reference dequantizes on the fly); build
            # FP16 candidate payloads only for the candidate's experts so the
            # bounded cache and staging stay proportional to the workload.
            routed_raw: dict[int, dict[str, torch.Tensor]] = {}
            fp16_payloads: dict[int, dict[str, torch.Tensor]] = {}
            for eid in reference_selected:
                names = v4support.routed_expert_tensor_names(layer, eid)
                t = load_tensors_from_shard(shard_path, names)
                routed_raw[eid] = {
                    "w1.weight": t[f"layers.{layer}.ffn.experts.{eid}.w1.weight"],
                    "w1.scale": t[f"layers.{layer}.ffn.experts.{eid}.w1.scale"],
                    "w2.weight": t[f"layers.{layer}.ffn.experts.{eid}.w2.weight"],
                    "w2.scale": t[f"layers.{layer}.ffn.experts.{eid}.w2.scale"],
                    "w3.weight": t[f"layers.{layer}.ffn.experts.{eid}.w3.weight"],
                    "w3.scale": t[f"layers.{layer}.ffn.experts.{eid}.w3.scale"],
                }
                if eid in candidate_selected:
                    fp16_payloads[eid] = {
                        "w1.weight": ds7.dequantize_expert_weight(
                            routed_raw[eid]["w1.weight"],
                            routed_raw[eid]["w1.scale"]).half(),
                        "w2.weight": ds7.dequantize_expert_weight(
                            routed_raw[eid]["w2.weight"],
                            routed_raw[eid]["w2.scale"]).half(),
                        "w3.weight": ds7.dequantize_expert_weight(
                            routed_raw[eid]["w3.weight"],
                            routed_raw[eid]["w3.scale"]).half(),
                    }
            shared_payload = {
                "w1.weight": moe.dequantize_fp8_e4m3(
                    shared_raw[f"layers.{layer}.ffn.shared_experts.w1.weight"],
                    shared_raw[f"layers.{layer}.ffn.shared_experts.w1.scale"]).half(),
                "w2.weight": moe.dequantize_fp8_e4m3(
                    shared_raw[f"layers.{layer}.ffn.shared_experts.w2.weight"],
                    shared_raw[f"layers.{layer}.ffn.shared_experts.w2.scale"]).half(),
                "w3.weight": moe.dequantize_fp8_e4m3(
                    shared_raw[f"layers.{layer}.ffn.shared_experts.w3.weight"],
                    shared_raw[f"layers.{layer}.ffn.shared_experts.w3.scale"]).half(),
            }
            # Free the shard from the working dir (keeps the Kaggle output
            # archive small) once all needed tensors are in memory.
            shard_path.unlink()

            shared_t = {
                "w1.weight": shared_raw[f"layers.{layer}.ffn.shared_experts.w1.weight"],
                "w1.scale": shared_raw[f"layers.{layer}.ffn.shared_experts.w1.scale"],
                "w2.weight": shared_raw[f"layers.{layer}.ffn.shared_experts.w2.weight"],
                "w2.scale": shared_raw[f"layers.{layer}.ffn.shared_experts.w2.scale"],
                "w3.weight": shared_raw[f"layers.{layer}.ffn.shared_experts.w3.weight"],
                "w3.scale": shared_raw[f"layers.{layer}.ffn.shared_experts.w3.scale"],
            }

            # ---- trusted reference per corpus case -----------------------
            # The candidate runs on x_route == corpus_cases[0]; its numerical
            # gate must compare against the reference computed on the SAME
            # input (a cross-input comparison would be meaningless).
            per_case = []
            ref_route: dict[str, torch.Tensor] | None = None
            for case_idx, (case_name, x) in enumerate(corpus_cases):
                case_ref = moe.moe_layer_forward(
                    x, gate_w, gate_b, routed_raw, shared_t, topk=TOPK,
                    route_scale=ROUTE_SCALE, swiglu_limit=SWIGLU_LIMIT,
                    keep_per_expert=True)
                if case_idx == 0:
                    ref_route = case_ref
                # The reference re-routes internally; confirm it selects the
                # same experts as the harness's up-front routing for the
                # same input distribution.
                pre_routed = case_routes[case_idx]["expert_ids"]
                route_agree = bool(torch.equal(case_ref["expert_ids"],
                                               pre_routed))
                per_case.append({
                    "case": case_name,
                    "reference_expert_ids_first_token":
                        [int(e) for e in case_ref["expert_ids"][0]],
                    "reference_route_agreement": route_agree,
                    "reference_moe_output_norm":
                        float(case_ref["moe_output"].norm()),
                    "reference_finite": bool(
                        torch.isfinite(case_ref["moe_output"]).all()),
                })
                if not route_agree:
                    failures.append({"name": f"layer_{layer}_route_agreement",
                                     "details": {"case": case_name}})
            layer_result["per_case"] = per_case
            if ref_route is None:
                raise RuntimeError("corpus produced no reference for x_route")

            # ---- cache-backed candidate (cold then warm) ------------------
            # Full-budget cache is sized to the ACTUAL distinct selected
            # expert union (+ shared expert + headroom) so the cold run
            # cannot evict: the warm replay is then a true all-hits,
            # zero-H2D, zero-reload replay (warm_reloaded == 0 gate).
            # 8 tokens x top-6 -> typically ~10-20 distinct experts; the
            # sized budget is airtight for any candidate union up to the
            # 48-slot max (candidate_selected, not the broader reference
            # union -- only the candidate's experts are staged).
            budget_bytes = int((len(candidate_selected) + 2) * 50 * (1 << 20))
            cache = v4cache.DeepSeekExpertCache(
                budget_bytes, device="cuda")
            loader = v4cache.DeepSeekExpertLoader(cache)
            t0 = time.time()
            cand_cold = candidate_moe_on_t4(
                x_route, ids, weights, cache, loader, layer,
                fp16_payloads, shared_payload)
            torch.cuda.synchronize()
            t_cold = time.time() - t0
            cold_stats = dict(cache.stats)
            cold_h2d = cold_stats["h2d_bytes"]
            cold_peak = cache.peak_resident_bytes

            t0 = time.time()
            cand_warm = candidate_moe_on_t4(
                x_route, ids, weights, cache, loader, layer,
                fp16_payloads, shared_payload)
            torch.cuda.synchronize()
            t_warm = time.time() - t0
            warm_stats = dict(cache.stats)
            warm_h2d = warm_stats["h2d_bytes"]
            warm_peak = cache.peak_resident_bytes

            # Numerical gate on the warm combined MoE output vs the trusted
            # FP32 reference computed on the SAME input (x_route).
            if ref_route is None:
                raise RuntimeError("corpus produced no reference for x_route")
            warm_metrics = v4contract.compute_ds8_metrics(
                ref_route["moe_output"], cand_warm["moe_output"])
            shared_metrics = v4contract.compute_ds8_metrics(
                ref_route["shared_output"], cand_warm["shared_output"])
            gate_passed = v4contract.ds8_gate_passed(warm_metrics)
            shared_gate_passed = v4contract.ds8_gate_passed(shared_metrics)
            # Cold vs warm candidate must be identical (cache correctness):
            # warm H2D must not grow and NO expert may be re-staged (the
            # candidate uses cache.get(), which never calls reserve, so warm
            # loads remaining equal to cold loads is the direct no-reload
            # proof), and outputs must match bit-for-bit.
            cold_warm_identical = bool(
                torch.equal(cand_cold["moe_output"], cand_warm["moe_output"]))
            warm_reloaded = warm_stats["loads"] - cold_stats["loads"]
            cache_ok = (cold_warm_identical and warm_h2d == cold_h2d
                        and warm_reloaded == 0)
            if not cache_ok:
                failures.append({"name": f"layer_{layer}_cache_behavior",
                                 "details": {"cold_warm_identical":
                                             cold_warm_identical,
                                             "cold_h2d": cold_h2d,
                                             "warm_h2d": warm_h2d,
                                             "warm_reloaded":
                                             warm_reloaded}})
            layer_result["warm_metrics"] = warm_metrics
            layer_result["shared_metrics"] = shared_metrics
            layer_result["gate_passed"] = bool(gate_passed)
            layer_result["shared_gate_passed"] = bool(shared_gate_passed)
            layer_result["cache"] = {
                "budget_bytes": cache.budget_bytes,
                "cold": {k: cold_stats[k] for k in
                         ("hits", "loads", "evictions", "h2d_bytes",
                          "wait_ms")} | {"peak_resident_bytes": cold_peak},
                "warm": {k: warm_stats[k] for k in
                         ("hits", "loads", "evictions", "h2d_bytes",
                          "wait_ms")} | {"peak_resident_bytes": warm_peak},
                "cold_warm_identical": cold_warm_identical,
                "warm_reloaded_experts": warm_reloaded,
                "cold_seconds": t_cold,
                "warm_seconds": t_warm,
            }
            print("  cold %.3fs warm %.3fs h2d cold=%d warm=%d"
                  % (t_cold, t_warm, cold_h2d, warm_h2d), flush=True)
            print("  warm moe gate:", gate_passed, "shared gate:",
                  shared_gate_passed, flush=True)

            # ---- bounded-budget eviction-pressure scenario ---------------
            small_cache = v4cache.DeepSeekExpertCache(
                int(120 << 20), device="cuda")  # ~2 FP16 experts -> evictions
            small_loader = v4cache.DeepSeekExpertLoader(small_cache)
            cand_small = candidate_moe_on_t4(
                x_route, ids, weights, small_cache, small_loader, layer,
                fp16_payloads, shared_payload)
            torch.cuda.synchronize()
            small_stats = dict(small_cache.stats)
            small_identical = bool(
                torch.equal(cand_small["moe_output"], cand_warm["moe_output"]))
            if not small_identical:
                failures.append({"name": f"layer_{layer}_eviction_pressure",
                                 "details": {"evictions":
                                             small_stats["evictions"],
                                             "loads": small_stats["loads"]}})
            layer_result["eviction_pressure"] = {
                "budget_bytes": small_cache.budget_bytes,
                "requests": small_stats["requests"],
                "hits": small_stats["hits"],
                "loads": small_stats["loads"],
                "evictions": small_stats["evictions"],
                "output_identical_to_full_budget": small_identical,
            }
            print("  eviction-pressure: loads=%d evictions=%d identical=%s"
                  % (small_stats["loads"], small_stats["evictions"],
                     small_identical), flush=True)

            layer_results[layer] = layer_result
            if not gate_passed or not shared_gate_passed or not cache_ok \
                    or not small_identical:
                failures.append({"name": f"layer_{layer}_gate",
                                 "details": {
                                     "gate_passed": bool(gate_passed),
                                     "shared_gate_passed":
                                     bool(shared_gate_passed),
                                     "cache_ok": bool(cache_ok),
                                     "eviction_identical": bool(
                                         small_identical)}})

        write_json(EVIDENCE / "environment.json", {
            "schema_version": 1,
            "run_id": RUN_ID,
            "repository": REPOSITORY,
            "branch": BRANCH,
            "repository_commit": commit,
            "harness_sha256": running_sha,
            "module_sha256": module_shas,
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REV,
            "shards": sorted(SHARDS.values()),
            "performance_comparable": False,
        })

        all_gates = all(layer_results[l].get("gate_passed", False)
                        for l in LAYERS)
        all_shared = all(layer_results[l].get("shared_gate_passed", False)
                         for l in LAYERS)
        verdict = ("ACCEPT_EXPERT_RUNTIME"
                   if all_gates and all_shared and not failures
                   else "REJECT_NUMERICAL")

        write_json(EVIDENCE / "ds8-expert-runtime-evidence.json", {
            "campaign": "deepseek-v4-flash-0731",
            "phase": "DS8-expert-runtime",
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REV,
            "layers": list(LAYERS),
            "topk": TOPK,
            "route_scale": ROUTE_SCALE,
            "swiglu_limit": SWIGLU_LIMIT,
            "n_tokens": N_TOKENS,
            "corpus_meta": corpus_meta,
            "corpus_cases": [name for name, _ in corpus_cases],
            "device": torch.cuda.get_device_name(0),
            "contract": v4contract.DS8_TOLERANCE,
            "near_zero_threshold": v4contract.NEAR_ZERO_THRESHOLD,
            "layers_results": layer_results,
            "verdict": verdict,
            "passed": bool(all_gates and all_shared and not failures),
            "performance_comparable": False,
            "note": (
                "DS8 correctness milestone: 3 complete official layers' "
                "routed+shared expert runtime on T4 through a bounded cache, "
                "validated against the trusted FP32 reference on the "
                "predeclared DS8 contract. Synthetic input corpus (official "
                "hidden-state traces are a DS5 dependency). No TPS is "
                "claimed. Note: warm-cache hits are proven by unchanged "
                "loads/h2d_bytes (the candidate uses cache.get(), which does "
                "not increment the hits counter; only reserve() does)."
            ),
        })

        shutil.copy2(Path(__file__).resolve(),
                     EVIDENCE / "deepseek_v4_expert_runtime.py")
        for key, rel in MODULE_RELATIVES.items():
            shutil.copy2(ROOT / rel,
                         EVIDENCE / f"deepseek_v4_module_{key}.py")
        (EVIDENCE / "manifest.sha256").write_text(
            hashlib.sha256(
                (EVIDENCE / "ds8-expert-runtime-evidence.json").read_bytes()
            ).hexdigest() + "  ds8-expert-runtime-evidence.json\n",
            encoding="utf-8")

    except Exception as exc:  # noqa: BLE001
        fatal_error = {"type": type(exc).__name__, "message": str(exc),
                       "traceback": traceback.format_exc()}
        write_json(EVIDENCE / "fatal-error.json", fatal_error)

    finally:
        required_paths = [
            EVIDENCE / "bootstrap-environment.json",
            EVIDENCE / "environment.json",
            EVIDENCE / "ds8-expert-runtime-evidence.json",
            EVIDENCE / "manifest.sha256",
            EVIDENCE / "logs/pip-install.log",
        ]
        required_status = []
        for path in required_paths:
            required_status.append({
                "path": path.relative_to(EVIDENCE).as_posix()
                if path.is_file() else str(path),
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            })
        missing = [row["path"] for row in required_status if not row["present"]]
        if missing:
            failures.append({"name": "required_artifacts_present",
                             "details": missing})
        result = "PASS" if fatal_error is None and not failures else "FAIL"
        artifacts = [
            {"path": path.relative_to(EVIDENCE).as_posix(),
             "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(EVIDENCE.rglob("*"))
            if path.is_file() and path.name != "artifact-manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "result": result,
            "terminal_verdict": ("ACCEPT_EXPERT_RUNTIME"
                                 if (result == "PASS" and all_gates is True)
                                 else "REJECT_NUMERICAL"),
            "repository_commit": commit,
            "module_sha256": module_shas,
            "fatal_error": fatal_error,
            "validation_failures": failures,
            "required_paths": required_status,
            "artifacts": artifacts,
            "performance_comparable": False,
        }
        write_json(EVIDENCE / "artifact-manifest.json", manifest)
        archive = shutil.make_archive(str(ARCHIVE_BASE), "gztar",
                                      root_dir=EVIDENCE.parent,
                                      base_dir=EVIDENCE.name)
        write_json(EVIDENCE / "archive-metadata.json", {
            "archive": archive,
            "archive_sha256": sha256_file(Path(archive)),
            "manifest_sha256": sha256_file(
                EVIDENCE / "artifact-manifest.json"),
            "excluded_from_archive": ["archive-metadata.json"],
        })
        print(json.dumps({"run_id": RUN_ID, "result": result,
                          "archive": archive, "fatal_error": fatal_error},
                         sort_keys=True), flush=True)
        if result != "PASS":
            raise SystemExit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
