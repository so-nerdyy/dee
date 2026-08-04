"""DS10: full DeepSeek-V4-Flash-0731 model execution on dual T4 (Kaggle).

Stage-driven harness (``DS10_STAGE`` env, default ``v1``) following the
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
from typing import Any

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

STAGE = os.environ.get("DS10_STAGE", "v1")
if STAGE not in ("v1", "v2", "v3", "v4", "v5", "v6", "final"):
    STAGE = "v1"

# Cache budgets per GPU (bytes): bounded, well under the 12-13 GiB bring-up
# envelope; the model's dense/state bytes come from the static memory plan.
CACHE_BUDGET_BYTES = 2 << 30  # 2 GiB per GPU

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
    from scripts import deepseek_v4_support as vs
    cfg = _load_cfg()
    source = _build_remote_source()

    gates: dict[str, Any] = {}

    # 1. Tensor coverage audit (identity only, zero checkpoint bytes).
    audit = vm.coverage_audit_report(
        source, n_layers=cfg.n_layers, n_hash_layers=cfg.n_hash_layers)
    gates["coverage_audit"] = audit.get("ok", False)
    gates["coverage"] = {
        "total_tensors": audit.get("total_tensors"),
        "unresolved": audit.get("unresolved", []),
        "duplicates": audit.get("duplicates", []),
        "by_component": audit.get("by_component"),
        "summary": audit.get("summary"),
    }
    ok = audit.get("ok", False) and not audit.get("unresolved") \
        and not audit.get("duplicates")

    # 2. Static per-GPU memory plan from headers (identity only).
    plan = vm.static_memory_plan(
        cfg, source, budgets0=CACHE_BUDGET_BYTES, budgets1=CACHE_BUDGET_BYTES)
    gates["memory_plan"] = plan
    ceilings_ok = all(
        row["total_estimate_gib"] <= 14.0
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
        # Every layer's state buffers are allocated + initialized
        sig = model.state_signatures(list(range(cfg.n_layers)))
        build_gates["state_signatures"] = sig
        build_gates["state_count"] = len(sig)
        # bounded forward sanity on GPU0 partition only (no full generation):
        # proves the built layers are callable with live weights
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]],
                                 device="cuda:0").long()
        h = torch.nn.functional.embedding(input_ids, model.embed)
        h = h.unsqueeze(2).expand(1, 8, cfg.hc_mult, h.size(-1)).contiguous()
        for idx, layer in enumerate(model.layers0):
            h = layer.forward(h, 0, input_ids)
        build_gates["partition0_smoke"] = {
            "shape": list(h.shape), "dtype": str(h.dtype),
            "finite": bool(torch.isfinite(h).all()),
            "device": str(h.device)}
        build_ok = bool(torch.isfinite(h).all())
        build_gates["fetch_stats"] = dict(source.stats)
    except Exception as exc:  # noqa: BLE001
        build_ok = False
        build_gates["error"] = f"{type(exc).__name__}: {exc}"
        build_gates["traceback"] = traceback.format_exc()
    gates["model_load_ok"] = build_ok
    gates["model_load"] = build_gates
    ok = ok and build_ok

    return {"verdict": "ACCEPT_MODEL_LOAD" if ok else "REJECT_MODEL_LOAD",
            "gates": gates, "stage": "v1"}


def _build_full_model(source: Any, cfg: Any) -> tuple[Any, dict[str, Any]]:
    """Build the full dual-GPU candidate and initialize all states."""
    from scripts import deepseek_v4_model as vm
    from scripts.deepseek_v4_cache import DeepSeekExpertCache, DeepSeekExpertLoader
    cache0 = DeepSeekExpertCache(CACHE_BUDGET_BYTES, device="cuda:0")
    cache1 = DeepSeekExpertCache(CACHE_BUDGET_BYTES, device="cuda:1")
    loader0 = DeepSeekExpertLoader(cache0)
    loader1 = DeepSeekExpertLoader(cache1)
    provider = vm.ExpertProvider(source)
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
    }
    return model, mem


def stage_v2() -> dict[str, Any]:
    """First GPU partition: layers 0..split-1 forward on GPU 0."""
    from scripts import deepseek_v4_model as vm
    cfg = _load_cfg()
    source = _build_remote_source()
    model, mem = _build_full_model(source, cfg)
    gates: dict[str, Any] = {"memory": mem}
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device="cuda:0").long()
    start_pos = 0
    h = torch.nn.functional.embedding(input_ids, model.embed)
    h = h.unsqueeze(2).expand(1, 8, cfg.hc_mult, h.size(-1)).contiguous()
    try:
        for idx, layer in enumerate(model.layers0):
            h = layer.forward(h, start_pos, input_ids)
        gates["partition0_forward_ok"] = True
        gates["hidden"] = {"shape": list(h.shape), "dtype": str(h.dtype),
                           "finite": bool(torch.isfinite(h).all()),
                           "device": str(h.device)}
        gates["state_signatures_0"] = model.state_signatures(
            list(range(model.split)))
        ok = bool(torch.isfinite(h).all())
    except Exception as exc:  # noqa: BLE001
        ok = False
        gates["partition0_forward_ok"] = False
        gates["error"] = f"{type(exc).__name__}: {exc}"
    return {"verdict": "ACCEPT_PARTITION0" if ok else "REJECT_ATTENTION_OR_STATE",
            "gates": gates, "stage": "v2"}


def stage_v3() -> dict[str, Any]:
    """Inter-GPU handoff: GPU0 partition + one GPU1 layer with checksums."""
    from scripts import deepseek_v4_model as vm
    cfg = _load_cfg()
    source = _build_remote_source()
    model, mem = _build_full_model(source, cfg)
    gates: dict[str, Any] = {"memory": mem}
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device="cuda:0").long()
    start_pos = 0
    h = torch.nn.functional.embedding(input_ids, model.embed)
    h = h.unsqueeze(2).expand(1, 8, cfg.hc_mult, h.size(-1)).contiguous()
    try:
        for idx, layer in enumerate(model.layers0):
            h = layer.forward(h, start_pos, input_ids)
        h = model._handoff(h, model.device1)
        gates["handoff"] = model.handoff_stats
        l1 = model.layer(model.split)
        h = l1.forward(h, start_pos, input_ids.to(model.device1))
        gates["one_gpu1_layer_ok"] = True
        gates["hidden_after"] = {"shape": list(h.shape), "dtype": str(h.dtype),
                                 "finite": bool(torch.isfinite(h).all()),
                                 "device": str(h.device)}
        ok = (model.handoff_stats.get("checksum_bitwise_equal") is True
              and bool(torch.isfinite(h).all()))
    except Exception as exc:  # noqa: BLE001
        ok = False
        gates["error"] = f"{type(exc).__name__}: {exc}"
    return {"verdict": "ACCEPT_INTERGPU" if ok else "REJECT_INTERGPU",
            "gates": gates, "stage": "v3"}


def stage_v4() -> dict[str, Any]:
    """All layers, first LM-head logits (no continued decode)."""
    from scripts import deepseek_v4_model as vm
    cfg = _load_cfg()
    source = _build_remote_source()
    model, mem = _build_full_model(source, cfg)
    gates: dict[str, Any] = {"memory": mem}
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device="cuda:0").long()
    try:
        captures: dict[int, dict[str, Any]] = {}
        logits = model.forward(input_ids, 0, captures=captures)
        gates["logits"] = {"shape": list(logits.shape),
                           "finite": bool(torch.isfinite(logits).all()),
                           "top5": [int(x) for x in
                                    logits.topk(5, -1)[1][0].tolist()]}
        gates["layer_captures_ok"] = {
            "n_layers_with_captures": len(captures),
            "sample_l0_expert_ids": captures.get(0, {}).get("expert_ids", None),
        }
        ok = bool(torch.isfinite(logits).all())
    except Exception as exc:  # noqa: BLE001
        ok = False
        gates["error"] = f"{type(exc).__name__}: {exc}"
    return {"verdict": "ACCEPT_FIRST_LOGITS" if ok else "REJECT_LM_HEAD",
            "gates": gates, "stage": "v4"}


def stage_generate(n_tokens: int, *, stage_name: str) -> dict[str, Any]:
    """Greedy generation: >= n_tokens, determinism, cold/warm, bounded memory."""
    from scripts import deepseek_v4_model as vm
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
        gates["token_count"] = len(toks)
        gates["token_trace"] = trace
        gates["token_ids_in_vocab"] = all(0 <= t < cfg.vocab_size for t in toks)
        # determinism: fresh model, same input -> same tokens
        model2, mem2 = _build_full_model(source, cfg)
        toks2 = model2.generate(input_ids, max_new_tokens=n_tokens)
        gates["deterministic_rerun"] = toks == toks2
        gates["memory2"] = mem2
        # cold == warm: reset + rerun through the same model/cache
        model.reset_state()
        toks_warm = model.generate(input_ids, max_new_tokens=n_tokens)
        gates["cold_warm_equal"] = toks == toks_warm
        ok = (len(toks) >= n_tokens
              and gates["token_ids_in_vocab"]
              and gates["deterministic_rerun"]
              and gates["cold_warm_equal"])
        verdict = "ACCEPT_DUAL_T4_DECODE" if ok else "REJECT_GENERATION_STATE"
    except Exception as exc:  # noqa: BLE001
        ok = False
        verdict = "REJECT_GENERATION_STATE"
        gates["error"] = f"{type(exc).__name__}: {exc}"
        gates["traceback"] = traceback.format_exc()
    return {"verdict": verdict, "gates": gates, "stage": stage_name}


def main() -> int:
    result: dict[str, Any] = {"verdict": "INVALID_EXPERIMENT",
                              "stage": STAGE, "performance_comparable": False}
    out_dir = EVIDENCE
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        identity = _acquire_source()
        global torch
        import torch  # noqa: PLC0415
        result["identity"] = identity
        result["run_id"] = RUN_ID
        result["model_revision"] = REV
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["gpu_count"] = (torch.cuda.device_count()
                               if torch.cuda.is_available() else 0)
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            result["verdict"] = "INVALID_EXPERIMENT"
            result["first_failing_gate"] = "cuda_dual_required"
        else:
            if STAGE == "v1":
                result.update(stage_v1())
            elif STAGE == "v2":
                result.update(stage_v2())
            elif STAGE == "v3":
                result.update(stage_v3())
            elif STAGE == "v4":
                result.update(stage_v4())
            elif STAGE == "v5":
                result.update(stage_generate(1, stage_name="v5"))
            elif STAGE == "v6":
                result.update(stage_generate(4, stage_name="v6"))
            else:
                result.update(stage_generate(16, stage_name="final"))
    except Exception as exc:  # noqa: BLE001
        result["verdict"] = "INVALID_EXPERIMENT"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print("FATAL:", exc, flush=True)
    finally:
        write_json(out_dir / "ds10-verdict.json",
                   {"verdict": result["verdict"],
                    "stage": result.get("stage"),
                    "performance_comparable": False,
                    "first_failing_gate": result.get("first_failing_gate"),
                    "run_id": RUN_ID})
        write_json(out_dir / "ds10-evidence.json", result)
        print("VERDICT:", result["verdict"], result.get("first_failing_gate", ""),
              flush=True)
    return 0 if str(result["verdict"]).startswith("ACCEPT") else 1


if __name__ == "__main__":
    sys.exit(main())
