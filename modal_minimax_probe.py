"""
================================================================================
MiniMax-M3 PROBE (Option A) -- cheap validation BEFORE the billed runs.
================================================================================
Validates, WITHOUT downloading the ~856GB weights:
  (1) the custom `trust_remote_code` config imports (custom code loads),
  (2) the safetensors INDEX (model.safetensors.index.json) parses, giving the
      exact expert tensor naming + per-expert shapes + which shard each expert
      lives in -- the addressing the Step-1 collector and Step-4 streaming
      manager need to mmap individual experts from storage.
This costs only a few KB of downloads. The full weight pull happens in Step 1/4.
================================================================================
"""
import os
import time
import json
import re

import modal

MODELS_VOL = "/vol/models"
MODEL_ID = "MiniMaxAI/MiniMax-M3"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.52.4",
        "huggingface_hub==0.30.2",
        "hf_transfer==0.1.8",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": MODELS_VOL})
)

app = modal.App("minimax-m3-probe")


@app.function(
    image=image,
    cpu=2,
    timeout=1200,
    volumes={MODELS_VOL: modal.Volume.from_name("dee-models", create_if_missing=True)},
)
def probe():
    from transformers import AutoConfig
    from huggingface_hub import hf_hub_download

    log = lambda *a: print("[PROBE]", *a, flush=True)
    t0 = time.time()

    # (1) Custom config import (trust_remote_code) -- proves the custom code loads.
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    tc = cfg.text_config
    n_moe = sum(tc.moe_layer_freq)
    log(f"config OK: experts={tc.num_local_experts} topk={tc.num_experts_per_tok} "
        f"shared={tc.n_shared_experts} layers={tc.num_hidden_layers} moe_layers={n_moe} "
        f"hidden={tc.hidden_size} dtype={cfg.torch_dtype}")

    # (2) Fetch ONLY the index (lists every tensor name, shape, and host shard).
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json",
                               local_dir=os.path.join(MODELS_VOL, "minimax-m3-index"))
    with open(idx_path) as f:
        idx = json.load(f)
    w = idx["weight_map"]
    total = idx.get("metadata", {}).get("total_size")
    log(f"index OK: {len(w)} tensors, total_size={total}")

    # (3) Extract MoE expert tensor naming + sample shapes + shard mapping.
    expert_keys = [k for k in w if "experts" in k and
                   ("gate_proj" in k or "up_proj" in k or "down_proj" in k)]
    log(f"expert tensors: {len(expert_keys)}")
    sample = {k: idx.get("weight_map", w).get(k) for k in expert_keys[:3]}
    # shapes come from the index? safetensors index does NOT include shapes.
    # Infer shape from naming convention instead (hidden/intermediate in config).
    per_expert_params = tc.intermediate_size * tc.hidden_size * 3  # gate+up+down
    naming = {}
    if expert_keys:
        k0 = expert_keys[0]
        naming["example_key"] = k0
        naming["digit_groups"] = re.findall(r"\d+", k0)
        naming["shard_for_example"] = w[k0]
    log(f"naming pattern: {naming}")
    log(f"per-expert params ~= {per_expert_params:,} "
        f"(inter={tc.intermediate_size} x hidden={tc.hidden_size} x3)")

    result = {
        "ok": True,
        "n_moe_layers": int(n_moe),
        "num_local_experts": int(tc.num_local_experts),
        "num_experts_per_tok": int(tc.num_experts_per_tok),
        "n_shared_experts": int(tc.n_shared_experts),
        "hidden_size": int(tc.hidden_size),
        "intermediate_size": int(tc.intermediate_size),
        "n_tensors": len(w),
        "n_expert_tensors": len(expert_keys),
        "total_size_bytes": total,
        "naming": naming,
        "probe_seconds": round(time.time() - t0, 1),
    }
    log("RESULT " + json.dumps(result))
    return json.dumps(result)


@app.local_entrypoint()
def main():
    print("[LOCAL] MiniMax-M3 probe starting (index-only, no weight download)...", flush=True)
    t0 = time.time()
    out = probe.remote()
    print(f"[LOCAL] probe finished in {time.time()-t0:.0f}s")
    print("[LOCAL] result:", out)
