"""
================================================================================
Step 2: Train the Oracle  (Dynamic Expert Eviction MoE)
================================================================================
A tiny 3-layer MLP that, given the hidden state of Token N, predicts which
experts will be activated for Token N+1 (the routing decision one step ahead).

It consumes the data collected in Step 1:
    /vol/data/collected/layer_{iii}.pt   -> {"hidden_states": (P,H), "next_experts": (P,k)}
    /vol/data/collected/metadata.json     -> {n_layers, n_pairs, top_k, hidden_dim, model}

Per MoE layer we train ONE independent Oracle (routing patterns differ per
layer). Each Oracle:
    input : hidden state of Token N         (dim = hidden_dim)
    head  : linear -> scores over num_experts
    target: multi-hot of the actual top-k experts of Token N+1
    loss  : BCEWithLogits (per-expert binary)  -> we take top-K at inference
This framing lets the Cache Manager (Step 3/5) request the *top 20 predicted*
experts regardless of the model's native top_k.

Run AFTER Step 1 confirms data is collected:
    modal run modal_step2_train_oracle.py
    modal run modal_step2_train_oracle.py --layer 0 --epochs 20
    modal run modal_step2_train_oracle.py --top-k-pred 20 --batch-size 1024

Outputs (saved to the dee-oracle-data volume):
    /vol/data/oracle/oracle_layer_{iii}.pt  -> {state_dict, num_experts, top_k,
                                               hidden_dim, cfg}
    /vol/data/oracle/oracle_meta.json
================================================================================
"""

import os
import json
import time
import argparse

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "numpy==1.26.4",
        "safetensors==0.4.5",
    )
)

DATA_VOL = modal.Volume.from_name("dee-oracle-data", create_if_missing=True)
DATA_VOL_PATH = "/vol/data"

app = modal.App("dynamic-expert-eviction-step2")


@app.function(
    image=image,
    # CPU-only is fine for a tiny MLP + a few thousand pairs; no GPU needed.
    cpu=4,
    volumes={DATA_VOL_PATH: DATA_VOL},
    timeout=60 * 60 * 2,
)
def train_oracle(
    collected_dir: str,
    oracle_dir: str,
    layer: int,            # -1 => train ALL layers
    epochs: int,
    batch_size: int,
    top_k_pred: int,       # how many experts the cache manager will request
    hidden_mlp: int,
    lr: float,
) -> str:
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    os.makedirs(oracle_dir, exist_ok=True)
    log = lambda *a: print("[STEP2]", *a, flush=True)

    # ---- Load metadata ---------------------------------------------------- #
    meta_path = os.path.join(collected_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    top_k = meta["top_k"]
    hidden_dim = meta["hidden_dim"]
    log(f"metadata: n_layers={n_layers} top_k={top_k} hidden_dim={hidden_dim} "
        f"n_pairs={meta['n_pairs']}")

    # Determine which layers to train
    if layer == -1:
        layers = list(range(n_layers))
    else:
        layers = [layer]
    # Check the requested layer files exist
    for L in layers:
        fp = os.path.join(collected_dir, f"layer_{L:03d}.pt")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"missing {fp} -- run Step 1 first")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    oracle_meta = {"layers": [], "top_k_native": top_k, "top_k_pred": top_k_pred,
                   "hidden_dim": hidden_dim, "hidden_mlp": hidden_mlp, "window": 2}

    for L in layers:
        fp = os.path.join(collected_dir, f"layer_{L:03d}.pt")
        d = torch.load(fp, map_location="cpu")
        H = d["hidden_states"].float()          # (P, H)
        e1 = d["next_experts_n1"].long()        # (P, k_native) experts N+1
        e2 = d["next_experts_n2"].long()        # (P, k_native) experts N+2
        P = H.shape[0]

        # 2-ahead UNION target: experts active at N+1 OR N+2.
        union = torch.cat([e1, e2], dim=1)       # (P, 2*k_native)  (may repeat)

        # Infer num_experts from the data, but never below the model's true
        # expert count (some layers may not activate all experts in the sample).
        model_num_experts = meta.get("num_experts", None)
        data_max = int(union.max().item()) + 1
        num_experts = max(data_max, model_num_experts or data_max)
        log(f"layer {L}: pairs={P} num_experts={num_experts} union_width={union.shape[1]}")

        # Build multi-hot UNION target (P, num_experts)
        Y = torch.zeros(P, num_experts, dtype=torch.float32)
        for i in range(union.shape[1]):
            Y.scatter_(1, union[:, i:i + 1], 1.0)
        # Ensure no double counting beyond 1.0
        Y.clamp_(max=1.0)

        # ---- Tiny 3-layer MLP Oracle ------------------------------------- #
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_mlp),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_mlp, hidden_mlp),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_mlp, num_experts),
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        ds = TensorDataset(H.to(device), Y.to(device))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        t0 = time.time()
        for ep in range(epochs):
            model.train()
            running = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.shape[0]
            # Eval: recall@top_k_pred of the UNION target (true supervised set)
            model.eval()
            with torch.no_grad():
                all_logits = model(H.to(device))
                pred_topk = torch.topk(all_logits, top_k_pred, dim=-1).indices
                hit = 0
                for i in range(P):
                    true_set = set(union[i].tolist())
                    pred_set = set(pred_topk[i].tolist())
                    if true_set:
                        hit += len(true_set & pred_set) / len(true_set)
                recall = hit / P
            avg_loss = running / P
            if ep % 5 == 0 or ep == epochs - 1:
                log(f"layer {L} epoch {ep:03d} loss={avg_loss:.4f} "
                    f"union_recall@{top_k_pred}={recall:.3f} ({time.time()-t0:.0f}s)")
        # Final union recall
        with torch.no_grad():
            all_logits = model(H.to(device))
            pred_topk = torch.topk(all_logits, top_k_pred, dim=-1).indices
            hit = 0
            for i in range(P):
                true_set = set(union[i].tolist())
                pred_set = set(pred_topk[i].tolist())
                if true_set:
                    hit += len(true_set & pred_set) / len(true_set)
            final_recall = hit / P
        log(f"layer {L} FINAL union_recall@{top_k_pred}={final_recall:.3f}")

        # ---- Save Oracle weights ----------------------------------------- #
        out_path = os.path.join(oracle_dir, f"oracle_layer_{L:03d}.pt")
        torch.save({
            "state_dict": model.cpu().state_dict(),
            "num_experts": num_experts,
            "top_k_native": top_k,
            "top_k_pred": top_k_pred,
            "window": 2,
            "hidden_dim": hidden_dim,
            "hidden_mlp": hidden_mlp,
            "final_recall": final_recall,
            "cfg": {
                "layers": [hidden_dim, hidden_mlp, hidden_mlp, num_experts],
                "activation": "relu",
            },
        }, out_path)
        log(f"layer {L}: saved oracle -> {out_path}")
        oracle_meta["layers"].append({
            "layer": L, "num_experts": num_experts,
            "final_recall": final_recall, "path": out_path,
        })

    with open(os.path.join(oracle_dir, "oracle_meta.json"), "w") as f:
        json.dump(oracle_meta, f, indent=2)
    DATA_VOL.commit()
    log(f"DONE. oracles saved to {oracle_dir}")
    return json.dumps(oracle_meta)


@app.local_entrypoint()
def main(
    collected_dir: str = DATA_VOL_PATH + "/collected2",
    oracle_dir: str = DATA_VOL_PATH + "/oracle2",
    layer: int = -1,
    epochs: int = 30,
    batch_size: int = 1024,
    top_k_pred: int = 24,
    hidden_mlp: int = 256,
    lr: float = 1e-3,
):
    print(f"[LOCAL] training Oracle on collected data in {collected_dir}", flush=True)
    t0 = time.time()
    result = train_oracle.remote(
        collected_dir=collected_dir,
        oracle_dir=oracle_dir,
        layer=layer,
        epochs=epochs,
        batch_size=batch_size,
        top_k_pred=top_k_pred,
        hidden_mlp=hidden_mlp,
        lr=lr,
    )
    print(f"[LOCAL] finished in {time.time()-t0:.1f}s")
    print("[LOCAL] oracle_meta:", result)
