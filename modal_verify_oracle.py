"""Read-only verification that Step 2 oracles landed on the dee-oracle-data volume."""
import os, json
import modal

DATA_VOL = modal.Volume.from_name("dee-oracle-data", create_if_missing=True)
DATA_VOL_PATH = "/vol/data"
app = modal.App("dee-verify-oracle")

@app.function(volumes={DATA_VOL_PATH: DATA_VOL}, timeout=300)
def verify():
    import os, json
    oracle_dir = os.path.join(DATA_VOL_PATH, "oracle")
    if not os.path.isdir(oracle_dir):
        return {"error": "oracle dir missing"}
    files = sorted(os.listdir(oracle_dir))
    pt_files = [f for f in files if f.endswith(".pt")]
    meta_path = os.path.join(oracle_dir, "oracle_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    recalls = [l["final_recall"] for l in meta["layers"]] if meta else []
    summary = {
        "n_oracle_pt": len(pt_files),
        "n_total_files": len(files),
        "first_5": files[:5],
        "last_5": files[-5:],
        "top_k_native": meta.get("top_k_native"),
        "top_k_pred": meta.get("top_k_pred"),
        "hidden_dim": meta.get("hidden_dim"),
        "hidden_mlp": meta.get("hidden_mlp"),
        "n_layers_trained": len(recalls),
        "recall_min": min(recalls) if recalls else None,
        "recall_max": max(recalls) if recalls else None,
        "recall_mean": sum(recalls)/len(recalls) if recalls else None,
    }
    print("ORACLE_VERIFY", json.dumps(summary, indent=2), flush=True)
    return summary

@app.local_entrypoint()
def main():
    print(verify.remote())
