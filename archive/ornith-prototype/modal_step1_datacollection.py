"""
================================================================================
Step 1: Environment Setup & Data Collection  (Dynamic Expert Eviction MoE)
================================================================================

Modal app that:
  1. Spins up a GPU container (A10G by default, A100 configurable) with
     PyTorch + Triton installed.
  2. Loads a Qwen3 MoE model (real, downloadable stand-in for the target
     Qwen 3.5 397B-A17B). The TARGET architecture is documented below.
  3. Runs a small text dataset through the model and, at EVERY MoE layer,
     intercepts two things via a forward hook:
        (a) the token hidden states of Token N        ->  hidden_states
        (b) the ACTUAL activated expert indices of Token N+1 -> gate top-k
     This pairs (hidden_state[N]) -> (experts[N+1]) which is exactly the
     supervised signal the Oracle (Step 2) needs to learn.
  4. Persists collected tensors + metadata to a Modal Volume so we never
     re-download the model or re-collect data on subsequent runs.

Run with:
    modal run modal_step1_datacollection.py
    modal run modal_step1_datacollection.py --model-name Qwen/Qwen3-30B-A3B
    modal run modal_step1_datacollection.py --gpu A100 --n-sequences 16

--------------------------------------------------------------------------------
TARGET MODEL (the eventual deployment target) -- NOT downloaded by default:
    Qwen 3.5 397B-A17B
      total params   : ~397B
      active / token : ~17B
      experts/layer  :  (Qwen3-style, e.g. 128 experts, top-8 routed)
      router         : top-k gating (SwiGLU MLP experts)
    We architect the collector so swapping in the 397B config is a one-line
    change once weights are available; the hook logic is model-agnostic.
--------------------------------------------------------------------------------
"""

import os
import json
import time
import argparse

import modal
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Modal Image: PyTorch + Triton + transformers + safetensors
# --------------------------------------------------------------------------- #
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.51.3",   # 4.51+ adds qwen3_moe architecture support
        "safetensors==0.4.5",
        "triton==3.0.0",          # needed in later steps; installed now for parity
        "numpy==1.26.4",
        "huggingface_hub[cli]==0.30.2",   # 4.51.3 needs huggingface-hub>=0.30.0
        "accelerate==1.0.1",              # required for device_map="auto"
        "hf_transfer==0.1.8",     # required for HF_HUB_ENABLE_HF_TRANSFER fast path
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# --------------------------------------------------------------------------- #
# Volumes: persist model weights cache + collected oracle training data
# --------------------------------------------------------------------------- #
MODEL_VOL = modal.Volume.from_name("dee-models", create_if_missing=True)
DATA_VOL = modal.Volume.from_name("dee-oracle-data", create_if_missing=True)

MODEL_VOL_PATH = "/vol/models"   # HF cache lives here
DATA_VOL_PATH = "/vol/data"      # collected hidden states + expert indices

app = modal.App("dynamic-expert-eviction-step1")

# --------------------------------------------------------------------------- #
# Sample dataset (kept tiny for a fast prototype run)
# --------------------------------------------------------------------------- #
SAMPLE_PROMPTS = [
    "The quick brown fox jumps over the lazy dog and then",
    "In a distant galaxy, a small robot learned to dream about",
    "Quantum mechanics suggests that particles can exist in",
    "The history of Venice is closely tied to the trade routes of",
    "Large language models partition computation across many",
    "A chef carefully balanced the spices because the recipe required",
    "The mathematician proved the theorem by reductio ad absurdum and",
    "Climate change accelerates because the ocean absorbs excess",
]


# --------------------------------------------------------------------------- #
# The Modal function
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "A10G"),   # override with --gpu via env, or edit
    volumes={MODEL_VOL_PATH: MODEL_VOL, DATA_VOL_PATH: DATA_VOL},
    timeout=60 * 60 * 3,                        # 3h ceiling
    secrets=[],                                  # no HF token needed for public models
)
def collect_data(model_name: str, n_sequences: int, max_new_tokens: int,
                 out_dir: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collections import defaultdict

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODEL_VOL_PATH, exist_ok=True)

    # Route HF downloads into the persistent model volume
    os.environ["HF_HOME"] = MODEL_VOL_PATH
    os.environ["TRANSFORMERS_CACHE"] = MODEL_VOL_PATH

    log = lambda *a: print("[STEP1]", *a, flush=True)
    log(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    log(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"model={model_name}  n_seqs={n_sequences}  max_new_tokens={max_new_tokens}")

    # ---- Load model + tokenizer ------------------------------------------- #
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_VOL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_VOL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"model loaded in {time.time()-t0:.1f}s")

    # ---- Discover MoE layer structure ------------------------------------- #
    # Qwen3 MoE places SparseMoeBlock modules; the router is `block.gate`.
    moe_blocks = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "SparseMoeBlock" in cls or "MoE" in cls:
            moe_blocks.append((name, mod))
    log(f"found {len(moe_blocks)} MoE blocks: "
        + ", ".join(n for n, _ in moe_blocks[:6]) + (" ..." if len(moe_blocks) > 6 else ""))
    if not moe_blocks:
        raise RuntimeError("No MoE blocks found -- model_name is not an MoE model.")

    # Per-layer capture targets, defined up-front:
    #   raw_caps[layer_i] : most-recent forward's (B,T,H) hidden + (B,T,k) experts
    #   pairs[layer_i]    : accumulated (hidden[N], experts[N+1]) training pairs
    raw_caps = defaultdict(dict)   # layer_i -> {"hidden":(B,T,H), "experts":(B,T,k)}
    pairs = defaultdict(list)      # layer_i -> list of (hidden_vec, experts_vec)

    # We hook the MoE BLOCK (not the gate) and, on each forward, recompute the
    # router logits from block.gate(hidden_states) ourselves. This is robust to
    # whatever the router's `forward` returns (weights vs indices vs tuple) and
    # always gives us the true raw logits -> top-k expert indices for that exact
    # token position. block input #0 is the (B,T,H) hidden state -> Token N.
    router_top_k = getattr(model.config, "num_experts_per_tok", 8)

    def make_block_hook_raw(layer_i):
        def hook(block_module, inp, out):
            hidden = inp[0].detach().to(torch.float16).cpu()   # (B, T, H)
            gate = getattr(block_module, "gate", None)
            with torch.no_grad():
                if gate is not None:
                    routing_logits = gate(inp[0].detach()).float()  # (B, T, E)
                else:
                    routing_logits = out.detach().float()
                try:
                    topk_vals, topk_idx = torch.topk(routing_logits, router_top_k, dim=-1)
                except Exception:
                    topk_idx = torch.zeros((*routing_logits.shape[:-1], router_top_k),
                                           dtype=torch.long)
            raw_caps[layer_i] = {
                "hidden": hidden,
                "experts": topk_idx.detach().cpu(),
            }
        return hook

    hooks = []
    for i, (name, mod) in enumerate(moe_blocks):
        h = mod.register_forward_hook(make_block_hook_raw(i))
        hooks.append(h)
    log(f"registered {len(hooks)} forward hooks (router_top_k={router_top_k})")

    # ---- Run the dataset, shifting to get Token N -> Token N+1 ------------- #
    # raw_caps / pairs are defined above; hooks are already registered to write
    # into raw_caps on each forward. After each prompt's forward we slice
    # raw_caps into pairs and drop the raw buffers.
    total_pairs = 0
    seq_count = 0
    for prompt in SAMPLE_PROMPTS[:n_sequences]:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            _ = model(input_ids, output_hidden_states=False, use_cache=False)
        seq_len = input_ids.shape[1]
        # raw_caps now holds this prompt's (B,T) router data for every layer.
        # 3-token window: for Token N, capture hidden[N] + experts[N+1] + experts[N+2].
        # Requires a position with at least two successors (t+2 in range).
        for layer_i in sorted(raw_caps.keys()):
            entry = raw_caps[layer_i]
            H = entry["hidden"]      # (B, T, H_dim)
            E = entry["experts"]     # (B, T, k)
            B, T = H.shape[0], min(H.shape[1], E.shape[1])
            for b in range(B):
                for t in range(T - 2):   # need N+1 and N+2 positions
                    h_vec = H[b, t].contiguous()              # Token N hidden
                    e_n1 = E[b, t + 1].contiguous()           # Token N+1 experts
                    e_n2 = E[b, t + 2].contiguous()           # Token N+2 experts
                    pairs[layer_i].append((h_vec, e_n1, e_n2))
                    total_pairs += 1
        raw_caps.clear()   # drop raw buffers; keep only accumulated pairs
        seq_count += 1
        log(f"prompt #{seq_count} len={seq_len} pairs-so-far={total_pairs}")

    # ---- Save per-layer pair tensors -------------------------------------- #
    meta = {"model": model_name, "n_layers": len(moe_blocks),
            "n_pairs": 0, "top_k": None, "hidden_dim": None, "num_experts": None,
            "window": 2}
    saved_pairs = 0
    for layer_i in sorted(pairs.keys()):
        plist = pairs[layer_i]
        if not plist:
            continue
        hs = torch.stack([p[0] for p in plist])   # (P, H)
        e1 = torch.stack([p[1] for p in plist])   # (P, k)  experts N+1
        e2 = torch.stack([p[2] for p in plist])   # (P, k)  experts N+2
        # Infer num_experts from the data for the Oracle
        num_experts = int(max(e1.max().item(), e2.max().item())) + 1
        fname = os.path.join(out_dir, f"layer_{layer_i:03d}.pt")
        torch.save({"hidden_states": hs, "next_experts_n1": e1,
                    "next_experts_n2": e2}, fname)
        meta["n_pairs"] += hs.shape[0]
        meta["top_k"] = e1.shape[1]
        meta["hidden_dim"] = hs.shape[1]
        meta["num_experts"] = num_experts
        saved_pairs += hs.shape[0]
        log(f"layer {layer_i}: saved {hs.shape[0]} pairs (top_k={e1.shape[1]}, "
            f"hidden_dim={hs.shape[1]}, num_experts={num_experts}) -> {fname}")

    if saved_pairs == 0:
        raise RuntimeError("No (hidden, experts) pairs were collected -- check "
                           "that the gate hook captured router outputs.")

    for h in hooks:
        h.remove()

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    DATA_VOL.commit()
    log(f"DONE. total_pairs={total_pairs}  saved_pairs={saved_pairs}  meta={meta}")
    log(f"metadata at {meta_path}")
    return json.dumps(meta)


# --------------------------------------------------------------------------- #
# Local entrypoint
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3-30B-A3B",
    gpu: str = "A10G",
    n_sequences: int = 32,
    max_new_tokens: int = 32,
    out_dir: str = DATA_VOL_PATH + "/collected2",
):
    # Allow GPU override through env (handy for `modal run --gpu` parity)
    os.environ["MODAL_GPU"] = gpu
    print(f"[LOCAL] launching Step1 collection on gpu={gpu} model={model_name}", flush=True)
    t0 = time.time()
    result = collect_data.remote(
        model_name=model_name,
        n_sequences=n_sequences,
        max_new_tokens=max_new_tokens,
        out_dir=out_dir,
    )
    print(f"[LOCAL] finished in {time.time()-t0:.1f}s")
    print("[LOCAL] metadata:", result)
