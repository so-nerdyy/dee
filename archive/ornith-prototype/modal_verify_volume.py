"""Quick read-only verification that Step 1 data landed on the dee-oracle-data volume."""
import os, json
import modal

DATA_VOL = modal.Volume.from_name("dee-oracle-data", create_if_missing=True)
DATA_VOL_PATH = "/vol/data"
app = modal.App("dee-verify-volume")

@app.function(volumes={DATA_VOL_PATH: DATA_VOL}, timeout=300)
def verify():
    import os, json
    collected = os.path.join(DATA_VOL_PATH, "collected")
    files = sorted(os.listdir(collected)) if os.path.isdir(collected) else []
    pt_files = [f for f in files if f.endswith(".pt")]
    meta_path = os.path.join(collected, "metadata.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    total_bytes = 0
    for f in pt_files:
        total_bytes += os.path.getsize(os.path.join(collected, f))
    out = {
        "n_pt_files": len(pt_files),
        "n_total_files": len(files),
        "total_pt_bytes_mb": round(total_bytes / 1e6, 2),
        "first_5_files": files[:5],
        "last_5_files": files[-5:],
        "metadata": meta,
    }
    print("VERIFY", json.dumps(out, indent=2), flush=True)
    return out

@app.local_entrypoint()
def main():
    print(verify.remote())
