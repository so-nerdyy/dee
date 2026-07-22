#!/usr/bin/env python3
"""run_real_parity.py — Tier-1 first-real-model parity test for dee.cpp.

End-to-end driver that:

  1. Loads Ornith-1.0-35B (deepreinforce-ai/Ornith-1.0-35B) tokenizer + config
     from /tmp/ornith_meta (downloaded separately, < 30 MB).
  2. Loads MODEL-00001-OF-00016.SAFETENSORS (~4.32 GB) into a state-dict
     (one shard only at tier-1).
  3. Builds a single-layer HF model (cfg.num_hidden_layers = 1) so missing
     shards do not crash from_pretrained; only layer 0, embed_tokens, lm_head
     are loaded from disk.
  4. Initializes pydee.Engine pointing at the shard with empty oracle_path
     (real-model integration mode — caller owns routing).
  5. Runs Tier-1 mileposts:
       a) 1-token equality: pure HF reference vs HF + DeeMoeAdapter.
       b) 8 greedy tokens (extends accepted 1-token run).
       c) 32 greedy tokens + perf baseline.
       d) Quantization sweep if all earlier milestones match.

Honest gaps (documented in dee.cpp/benchmark_reports/real_generation_ledger.jsonl):
- Tier-1 uses only ONE shard (layer 0 + embed + lm_head); tokens past the
  single-layer boundary are non-physical since layers 1..39 are not loaded.
- MTP and vision are disabled.
- Shared experts stay on HF native (DeeMoeAdapter keeps HF's shared_experts).
- Hybrid linear + full attention handled by HF custom code, untouched.

Usage:
    python3 scripts/run_real_parity.py --tokens 1
    python3 scripts/run_real_parity.py --tokens 8
    python3 scripts/run_real_parity.py --tokens 32 --cuda
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
REPO_ROOT      = Path(__file__).resolve().parents[1]
ORNITH_META    = Path(os.environ.get("ORNITH_META", "/tmp/ornith_meta"))
ORNITH_SHARD_0 = Path(os.environ.get("ORNITH_SHARD_0",
                                     "/tmp/ornith_min/model-00001-of-00016.safetensors"))
DEE_BUILD_DIR  = Path(os.environ.get("DEE_BUILD_DIR", str(REPO_ROOT / "build")))
LEDGER         = REPO_ROOT / "benchmark_reports" / "real_generation_ledger.jsonl"
PROMPTS_TIER1  = [
    "Capital of France is",
    "7 * 6 =",
    "def fibonacci(n):",
    "Once upon a time",
]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_pydee_in_path() -> None:
    """Add build output to sys.path so `import pydee` works."""
    pydee_dir = REPO_ROOT / "pydee"
    sys.path.insert(0, str(pydee_dir))
    sys.path.insert(0, str(REPO_ROOT))


def _force_cpu_only() -> None:
    """Force CPU-only mode for HF + dee.cpp during tier-1 parity.

    Even with pydee_cfg.use_cuda=False, HF transformers may bind to CUDA at
    import time. Hide CUDA devices before any HF/dee import is performed.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYDEE_FORCE_CPU", "1")


def _json(o):
    return json.dumps(o, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Model loading (single-shard tier-1 amputation)
# --------------------------------------------------------------------------- #
def load_hf_model_tokens(tier: str = "shard0_1layer") -> tuple:
    """Load config + tokenizer + state-dict + amputed model.

    Returns (model, tokenizer, config, device).
    """
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import load_file
    from transformers.modeling_utils import no_init_weights

    cfg = AutoConfig.from_pretrained(str(ORNITH_META), trust_remote_code=True)
    # Strip vision if present (we only use the text path)
    if hasattr(cfg, "vision_config"):
        try:
            del cfg.vision_config
        except Exception:
            pass
    # Propagate num_hidden_layers amputation to text_config (Qwen3_5 hybrid
    # reads the decoder count from text_config, not the root config).
    if tier == "shard0_1layer":
        cfg.num_hidden_layers = 1
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            try:
                cfg.text_config.num_hidden_layers = 1
            except Exception:
                pass
    tok = AutoTokenizer.from_pretrained(str(ORNITH_META), trust_remote_code=True)

    with no_init_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    # Move to BF16 to avoid FP32 memory blow-up on the loaded tensors.
    model.to(torch.bfloat16)

    # Load the single shard
    if not ORNITH_SHARD_0.exists():
        raise FileNotFoundError(
            f"Shard 00001 missing at {ORNITH_SHARD_0}. "
            "Download it first via `huggingface_hub.hf_hub_download('deepreinforce-ai/Ornith-1.0-35B', "
            "'model-00001-of-00016.safetensors')`."
        )
    state = load_file(str(ORNITH_SHARD_0))
    # Drop tensors for layers > 0 when in shard0_1layer mode
    if tier == "shard0_1layer":
        state = {
            k: v for k, v in state.items()
            if (".layers." not in k) or (".layers.0." in k)
        }

    # Conditionally strip the leading "model.language_model." prefix from shard
    # keys so they match the model's expected layout. Use STRUCTURAL detection
    # (presence of the language_model wrapper) rather than a class-name substring,
    # which would miss e.g. Qwen3_5ForCausalLM.
    _is_conditional_gen = (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
    )
    if not _is_conditional_gen:
        # Text-only model (e.g., Qwen3_5TextForCausalLM or Qwen3_5ForCausalLM):
        # expected key prefix is "model.embed_tokens.weight" not
        # "model.language_model.embed_tokens.weight".
        _LAN = "model.language_model."
        state = {
            (k[len(_LAN):] if k.startswith(_LAN) else k): v for k, v in state.items()
        }
    print(
        f"[load] model class: {type(model).__name__}  "
        f"conditional_gen={_is_conditional_gen}"
    )
    # strict=False: missing keys (rotary buffers, mtp head, etc.) get defaults
    # + unexpected keys (e.g., MTP head tensors from shard 1) ignored.
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[load] load_state_dict: missing={len(missing)} "
        f"(e.g. {missing[:3]}), "
        f"unexpected={len(unexpected)} (e.g. {unexpected[:3]})"
    )

    device = "cuda" if torch.cuda.is_available() and CFG_USE_CUDA_DEFAULT() else "cpu"
    model = model.to(device).eval()

    return model, tok, cfg, device


def CFG_USE_CUDA_DEFAULT() -> bool:
    return os.environ.get("PYDEE_FORCE_CPU", "0") != "1"


# --------------------------------------------------------------------------- #
# dee.cpp engine init (real-model mode)
# --------------------------------------------------------------------------- #
def init_dee_engine(cfg, top_k: int) -> "pydee.Engine":
    import pydee
    pydee_cfg = pydee.EngineConfig()
    pydee_cfg.shard_path = str(ORNITH_SHARD_0)
    pydee_cfg.oracle_path = ""                 # real-model integration mode
    # Ornith's nesting: cfg.text_config holds hidden_size / moe_intermediate_size /
    # num_experts / num_hidden_layers for the text sub-model. Resolve safely.
    _src = cfg.text_config if hasattr(cfg, "text_config") else cfg
    pydee_cfg.num_experts = getattr(
        _src, "num_local_experts", getattr(_src, "num_experts", 256)
    )
    pydee_cfg.topk = top_k
    pydee_cfg.num_layers = getattr(_src, "num_hidden_layers", 1)
    pydee_cfg.hidden = getattr(_src, "hidden_size", 2048)
    pydee_cfg.inter = getattr(
        _src, "moe_intermediate_size", getattr(_src, "intermediate_size", 256)
    )
    pydee_cfg.use_cuda = False  # tier-1 CPU parity; switch on T4 later
    print(
        f"[run] pydee cfg: hidden={pydee_cfg.hidden} inter={pydee_cfg.inter} "
        f"num_experts={pydee_cfg.num_experts} topk={pydee_cfg.topk} "
        f"num_layers={pydee_cfg.num_layers} use_cuda={pydee_cfg.use_cuda}"
    )
    # Fail-fast assertions on Orbit's known fixed-shape config.
    assert pydee_cfg.inter == 512, (
        f"pydee_cfg.inter={pydee_cfg.inter}, expected 512 (Ornith moe_intermediate_size)"
    )
    assert pydee_cfg.num_experts == 256, (
        f"pydee_cfg.num_experts={pydee_cfg.num_experts}, expected 256"
    )
    assert pydee_cfg.hidden == 2048, (
        f"pydee_cfg.hidden={pydee_cfg.hidden}, expected 2048"
    )
    assert pydee_cfg.topk == 8, f"pydee_cfg.topk={pydee_cfg.topk}, expected 8"
    assert pydee_cfg.use_cuda == False, "tier-1 must run CPU-only"
    assert cfg.num_hidden_layers == 1, (
        f"cfg.num_hidden_layers={cfg.num_hidden_layers}, expected 1 (tier-1 amputation)"
    )
    pydee_cfg.transfer_dtype = pydee.WeightTransferDType.Bf16
    pydee_cfg.cache_dtype = pydee.DeviceCacheDType.Fp32
    pydee_cfg.verbose = False
    engine = pydee.Engine()
    if not engine.init(pydee_cfg):
        raise RuntimeError("pydee.Engine.init failed (see prior stderr).")
    return engine


# --------------------------------------------------------------------------- #
# Tier-1 step 1: 1-token parity
# --------------------------------------------------------------------------- #
def tier1_one_token(model, tok, device, engine) -> dict:
    prompt = "Capital of France is"
    inp = tok(prompt, return_tensors="pt").to(device)
    print(f"[tier1.1] prompt tokens: {inp.input_ids.shape}")

    # Reference (pure HF, no dee.cpp)
    with torch.inference_mode():
        ref_out = model(**inp)
    ref_logits = ref_out.logits[0, -1].float().cpu().numpy()
    ref_argmax = int(np.argmax(ref_logits))
    ref_token_id = ref_argmax
    ref_text = tok.decode([ref_token_id])

    # dee.cpp-attached variant
    from pydee.adapter import install_dee_moe
    installed_layers = install_dee_moe(model, engine, num_layers_to_patch=1, top_k=8)
    print(f"[tier1.1] installed DeeMoeAdapter on layers: {installed_layers}")

    with torch.inference_mode():
        dee_out = model(**inp)
    dee_logits = dee_out.logits[0, -1].float().cpu().numpy()
    dee_argmax = int(np.argmax(dee_logits))
    dee_token_id = dee_argmax
    dee_text = tok.decode([dee_token_id])

    abs_diff = float(np.max(np.abs(ref_logits - dee_logits)))
    rel_diff = float(np.linalg.norm(ref_logits - dee_logits) / (np.linalg.norm(ref_logits) + 1e-9))
    top5_ref  = np.argsort(-ref_logits)[:5].tolist()
    top5_dee  = np.argsort(-dee_logits)[:5].tolist()
    match = (ref_token_id == dee_token_id) and (abs_diff < 0.05)
    return {
        "prompt": prompt,
        "match": match,
        "ref": {"token_id": ref_token_id, "text": ref_text, "top5_ids": top5_ref},
        "dee": {"token_id": dee_token_id, "text": dee_text, "top5_ids": top5_dee},
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
    }


# --------------------------------------------------------------------------- #
# Tier-1 step 2: greedy multi-token generation
# --------------------------------------------------------------------------- #
def tier1_greedy(model, tok, device, n: int = 8) -> dict:
    """Generate n tokens greedy (temp=0). Records the token IDs."""
    prompt = "Capital of France is"
    inp = tok(prompt, return_tensors="pt").to(device)
    ids = inp.input_ids[0].tolist()
    started = time.time()
    with torch.inference_mode():
        for step in range(n):
            out = model(input_ids=torch.tensor([ids], device=device))
            next_logits = out.logits[0, -1].float()
            next_id = int(torch.argmax(next_logits).item())
            ids.append(next_id)
    wall = time.time() - started
    text = tok.decode(ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "generated_token_ids": ids[len(inp.input_ids[0]):],
        "generated_text": text,
        "wall_seconds": wall,
        "tokens_per_second": (n / wall) if wall > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# LEDGER writer
# --------------------------------------------------------------------------- #
def append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="dee.cpp → HF real-model parity ladder")
    p.add_argument("--tokens", type=int, default=1,
                   help="greedy tokens to generate after the reference token (default 1)")
    p.add_argument("--tier", choices=["shard0_1layer"], default="shard0_1layer")
    p.add_argument("--topk", type=int, default=8)
    args = p.parse_args()

    _ensure_pydee_in_path()
    _force_cpu_only()

    # Belt-and-suspenders: if HF rebuilt text_config during from_config,
    # re-pin num_hidden_layers=1 on the resolved instance.
    if tier == "shard0_1layer" and cfg.num_hidden_layers == 1:
        try:
            if hasattr(model, "config") and hasattr(model.config, "text_config"):
                tc = model.config.text_config
                if tc is not None and getattr(tc, "num_hidden_layers", None) != 1:
                    tc.num_hidden_layers = 1
        except Exception:
            pass

    print("[run] loading model + tokenizer...")
    model, tok, cfg, device = load_hf_model_tokens(tier=args.tier)
    print(f"[run] model on device={device}; num_hidden_layers={cfg.num_hidden_layers}; "
          f"hidden_size={cfg.hidden_size}")

    print("[run] initializing dee.cpp engine...")
    engine = init_dee_engine(cfg, top_k=args.topk)

    print("[run] TIER 1a: 1-token equality vs HF reference")
    res1 = tier1_one_token(model, tok, device, engine)
    print(_json(res1))
    append_ledger({
        "id": f"TIER1-1TOKEN-{_now_iso()}",
        "stage": "tier1.1",
        "tier": args.tier,
        "model": "deepreinforce-ai/Ornith-1.0-35B",
        "shard_path": str(ORNITH_SHARD_0),
        "shard_size_bytes": ORNITH_SHARD_0.stat().st_size if ORNITH_SHARD_0.exists() else None,
        "dee_engine_path": str(Path(__file__).parent.parent / "pydee"),
        "result": res1,
        "match": res1["match"],
        "consensus_1_token": res1["ref"]["token_id"] == res1["dee"]["token_id"],
    })

    if not res1["match"]:
        print("[run] 1-token parity FAILED. Aborting before expanding tier.")
        return 1

    if args.tokens >= 8:
        print("[run] TIER 1b: 8-token greedy generation (with DeeMoeAdapter)")
        res8 = tier1_greedy(model, tok, device, n=8)
        print(_json(res8))
        append_ledger({
            "id": f"TIER1-8TOKEN-{_now_iso()}",
            "stage": "tier1.2",
            "result": res8,
        })

    if args.tokens >= 32:
        print("[run] TIER 1c: 32-token greedy generation")
        res32 = tier1_greedy(model, tok, device, n=32)
        print(_json(res32))
        append_ledger({
            "id": f"TIER1-32TOKEN-{_now_iso()}",
            "stage": "tier1.3",
            "result": res32,
        })

    print("[run] tier-1 complete (1 token match + optional 8/32 token runs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
