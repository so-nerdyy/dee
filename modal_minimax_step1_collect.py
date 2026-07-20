"""
================================================================================
MiniMax-M3 Step 1: 10-token-window router data collection (Option A)
================================================================================
The model is ~809 GiB (BF16) and cannot fit on any single Modal GPU. We run the
OFFICIAL forward (correct router + MSA attention) but stream every module's
weights from mmap storage JIT: a forward_pre_hook loads a module's weights into
GPU, a forward_hook frees them after. The whole 428B graph runs on one GPU with
only one layer's worth of weights resident at a time -- the same streaming
principle the final engine uses. Step 1 is its first real use.

For each MoE layer L and token position N we save:
  hidden[N]  : layer-L input hidden state (Oracle input)
  union(N+1..N+10) of activated experts at layer L (Oracle target, 10-ahead)
================================================================================
"""
import os
import json
import time
import glob

import modal

MODELS_VOL = "/vol/models"
DATA_VOL = "/vol/data"
MODEL_ID = "MiniMaxAI/MiniMax-M3"
OUT_DIR = os.path.join(DATA_VOL, "minimax_collected")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.52.4",
        "huggingface_hub==0.30.2",
        "hf_transfer==0.1.8",
        "safetensors",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": MODELS_VOL})
)

app = modal.App("minimax-step1-collect")


def _make_streaming_hooks(get_tensor, key_prefix_root):
    """Return (pre_hook, post_hook) that JIT-load then free a module's weights
    from mmap. `get_tensor(full_key)` returns the mmap'd tensor (CPU)."""
    import torch

    def pre_hook(module, args):
        # map this module's params to safetensors keys
        with torch.no_grad():
            for name, p in module.named_parameters():
                full = f"{key_prefix_root}{name}"
                try:
                    t = get_tensor(full)
                except KeyError:
                    continue
                if p.is_meta or p.device.type != "cuda":
                    p.data = t.to("cuda", non_blocking=True)
                else:
                    p.data.copy_(t.to("cuda", non_blocking=True))

    def post_hook(module, args, output):
        # free CUDA weights to keep only one layer resident
        with torch.no_grad():
            for name, p in module.named_parameters():
                if p.is_meta:
                    continue
                p.data = torch.empty(0, device="meta", dtype=p.dtype)
        torch.cuda.empty_cache()

    return pre_hook, post_hook


@app.function(
    image=image,
    gpu="L4",                       # 24GB fits one streamed MoE layer (~14.5GB) at a time
    timeout=7200,
    volumes={MODELS_VOL: modal.Volume.from_name("dee-models", create_if_missing=True),
             DATA_VOL: modal.Volume.from_name("dee-oracle-data", create_if_missing=True)},
)
def collect(n_sequences: int = 2, lookahead: int = 10):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import safetensors.torch as st

    log = lambda *a: print("[STEP1]", *a, flush=True)
    t0 = time.time()

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    tc = cfg.text_config
    NUM_LAYERS = tc.num_hidden_layers
    NUM_EXPERTS = tc.num_local_experts
    TOPK = tc.num_experts_per_tok
    HID = tc.hidden_size
    N_MOE = sum(tc.moe_layer_freq)
    log(f"config: layers={NUM_LAYERS} moe={N_MOE} experts={NUM_EXPERTS} topk={TOPK} hid={HID}")

    # --- Ensure weights are in the Volume (download once, idempotent), then
    #     mmap the shards. snapshot_download writes into /vol/models/MiniMax-M3.
    from huggingface_hub import snapshot_download
    model_dir = os.path.join(MODELS_VOL, "MiniMax-M3")
    log(f"snapshot_download -> {model_dir} (may pull ~809GiB on first run)")
    snap = snapshot_download(MODEL_ID, local_dir=model_dir,
                             local_files_only=False, allow_patterns=["*.safetensors",
                             "*.json", "*.py"])
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    handles = [st.safe_open(sh, framework="pt") for sh in shards]
    log(f"shards ready: {len(shards)} in {snap}")

    def get_tensor(name: str) -> torch.Tensor:
        sh_file = wmap[name]
        for h in handles:
            if os.path.basename(h._path) == sh_file:       # type: ignore[attr-defined]
                return h.get_tensor(name)                   # mmap (no copy)
        raise KeyError(name)

    # --- Build the model on the meta device (no weight allocation), then attach
    #     JIT streaming hooks keyed by parameter FQN. The official forward runs
    #     unmodified, so router indices are correct.
    model = AutoModelForCausalLM.from_pretrained(
        snap, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="meta", local_files_only=True)
    model.eval()

    # Determine the safetensors key prefix that matches module FQNs.
    # Module FQN example: language_model.model.layers.3.block_sparse_moe...
    # safetensors key:     language_model.model.layers.3.block_sparse_moe...
    # So prefix_root = "" and we match FQN -> key directly (strip nothing).
    # We verify by checking a known param exists in wmap.
    sample_fqn = None
    for n, _ in model.named_parameters():
        sample_fqn = n
        break
    prefix_root = ""
    if sample_fqn not in wmap:
        # try stripping a leading 'language_model.' or 'model.' segment
        for cand in [sample_fqn, sample_fqn.replace("language_model.", ""),
                     sample_fqn.replace("model.", "", 1)]:
            if cand in wmap:
                prefix_root = sample_fqn[: len(sample_fqn) - len(cand)]
                break
    log(f"sample param FQN='{sample_fqn}' prefix_root='{prefix_root}' in_wmap={sample_fqn in wmap}")

    pre_hook, post_hook = _make_streaming_hooks(get_tensor, prefix_root)
    hook_handles = []
    for mod in model.modules():
        if list(mod.parameters(recurse=False)):   # only modules that own params
            hook_handles.append(mod.register_forward_pre_hook(pre_hook))
            hook_handles.append(mod.register_forward_hook(post_hook))

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    prompts = [
        "Explain dynamic expert eviction for mixture-of-experts models.",
        "Write a Python function that computes the Fibonacci sequence recursively.",
        "Summarize the causes of the French Revolution in two paragraphs.",
        "Translate to French: the quick brown fox jumps over the lazy dog.",
        "A robot must sort colored blocks by size. Describe its algorithm step by step.",
        "What are the tradeoffs between FlashAttention and standard attention?",
    ][:n_sequences]

    # --- Router capture: hook each MoE gate. Record per-layer top-k indices.
    router_caps = {}      # layer_idx -> list of [S, TOPK] tensors
    gate_modules = {}

    def attach_router_hooks():
        caps = {}
        handles = []
        for li in range(NUM_LAYERS):
            # find gate module for MoE layer li
            gate = None
            for n, m in model.named_modules():
                if f"layers.{li}.block_sparse_moe.gate" in n or \
                   (f"layers.{li}." in n and (n.endswith(".gate") or "router" in n)):
                    gate = m
                    break
            if gate is None:
                continue
            cap = {}
            def _hook(_m, _i, out, cap=cap):
                logits = out[0] if isinstance(out, tuple) else out
                if logits.dim() == 2:
                    _, idx = torch.topk(logits.float(), TOPK, dim=-1)
                    cap["idx"] = idx.detach().cpu()
            hh = gate.register_forward_hook(_hook)
            handles.append(hh)
            caps[li] = cap
        return caps, handles

    caps, router_handles = attach_router_hooks()

    all_hiddens = {li: [] for li in range(NUM_LAYERS)}
    all_targets = {li: [] for li in range(NUM_LAYERS)}
    total_pairs = 0

    for p_i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        S = ids.shape[1]
        if S < lookahead + 2:
            pad = torch.full((1, lookahead + 2 - S), ids[0, 0])
            ids = torch.cat([ids, pad], dim=1)
            S = ids.shape[1]
        log(f"prompt {p_i}: '{prompt[:40]}...' tokens={S}")
        # Run official forward; JIT streaming loads/frees weights per module.
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        # hidden_states[i] = input to layer i (transformers convention: index 0
        # is embedding output). We need hidden[N] = layer-L input.
        hs = out.hidden_states           # tuple length NUM_LAYERS+1
        for li in caps:
            if caps[li].get("idx") is None:
                continue
            ridx = caps[li]["idx"].long()        # [S, TOPK]
            # hidden input to layer li = hs[li] (embedding is hs[0])
            h_in = hs[li][0].detach().cpu().to(torch.float32)   # [S, H]
            tgt = torch.zeros(S, NUM_EXPERTS, dtype=torch.uint8)
            for off in range(1, lookahead + 1):
                sl = ridx[off:]
                for k in range(TOPK):
                    tgt[: (S - off), sl[:, k]] = 1
            all_hiddens[li].append(h_in)
            all_targets[li].append(tgt)
            total_pairs += S
            caps[li]["idx"] = None     # reset for next prompt

    # --- Save per layer.
    os.makedirs(OUT_DIR, exist_ok=True)
    saved = 0
    for li in range(NUM_LAYERS):
        if not all_hiddens[li]:
            continue
        H = torch.cat(all_hiddens[li], dim=0).to(torch.float32)
        Tg = torch.cat(all_targets[li], dim=0)
        torch.save({"hiddens": H, "targets": Tg,
                    "meta": {"layer": li, "topk": TOPK, "lookahead": lookahead,
                             "n_tokens": H.shape[0], "num_experts": NUM_EXPERTS,
                             "hidden_dim": HID}},
                   os.path.join(OUT_DIR, f"layer_{li:02d}.pt"))
        saved += 1
    for h in hook_handles + router_handles:
        h.remove()
    meta = {"model": MODEL_ID, "n_layers": NUM_LAYERS, "n_moe_layers": N_MOE,
            "num_experts": NUM_EXPERTS, "topk": TOPK, "lookahead": lookahead,
            "hidden_dim": HID, "n_sequences": len(prompts), "total_pairs": total_pairs,
            "saved_layers": saved}
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("=" * 60)
    log(f"SAVED {saved} layers, total_pairs={total_pairs} in {time.time()-t0:.0f}s")
    log(f"meta: {json.dumps(meta)}")
    log("=" * 60)
    return json.dumps(meta)


@app.local_entrypoint()
def main(n_sequences: int = 2, lookahead: int = 10):
    print(f"[LOCAL] MiniMax Step1 collect: n_sequences={n_sequences} lookahead={lookahead}",
          flush=True)
    t0 = time.time()
    out = collect.remote(n_sequences=n_sequences, lookahead=lookahead)
    print(f"[LOCAL] finished in {time.time()-t0:.0f}s")
    print("[LOCAL] meta:", out)
