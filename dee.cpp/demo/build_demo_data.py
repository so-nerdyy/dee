"""Pack the r7 L4 b8-ask run into demo/demo_data.js for the replay page.

Inputs: runs/p6m-r7-l4-b8-ask (generated_checkpoint-c0.jsonl, result-c0-r*.json, summary.json)
plus cache_events-c0.jsonl (path as argv[1]; too big to keep in the repo) and
runs/r7_environment.json + the r7 scorecards for the results table.
"""
import base64, json, sys, glob, os
from tokenizers import Tokenizer

RUN = "runs/p6m-r7-l4-b8-ask"
EOS = 1  # <｜end▁of▁sentence｜>
tok = Tokenizer.from_file("benchmark_reports/deepseek-v4-flash-0731-t4/tokenizer-assets/tokenizer.json")
assert tok.token_to_id("<\uff5cend\u2581of\u2581sentence\uff5c>") == EOS

steps = [json.loads(l) for l in open(f"{RUN}/generated_checkpoint-c0.jsonl", encoding="utf-8")]
rows = [json.load(open(f"{RUN}/result-c0-r{i}.json", encoding="utf-8")) for i in range(8)]
for r in rows:
    assert r["generated_token_ids"] == [s["token_ids"][r["row"]] for s in steps], "checkpoint/result mismatch"

def clean_prompt(p):
    return p.split("<\uff5cbegin\u2581of\u2581sentence\uff5c>")[-1].split("<\uff5cAssistant\uff5c>")[0]

# per-row text fragments, incremental decode so multi-byte chars land whole; frozen after first EOS
frags, done_at = [], []
for r in rows:
    ids, out, prev, stop = r["generated_token_ids"], [], "", None
    for i, t in enumerate(ids):
        if stop is None and t == EOS:
            stop = i
        if stop is not None:
            out.append(""); continue
        cur = tok.decode(ids[: i + 1], skip_special_tokens=False)
        out.append(cur[len(prev):]); prev = cur
    frags.append(out); done_at.append(stop)

N = len(steps)
LAYERS, EXPERTS = 43, 256
TIER = {"resident": 1, "host_hit": 2, "cold": 3}
grid = [bytearray(LAYERS * EXPERTS) for _ in range(N)]
counts = [[0, 0, 0] for _ in range(N)]
with open(sys.argv[1], encoding="utf-8") as f:
    for ln in f:
        e = json.loads(ln)
        s, t = e["token"], TIER[e["kind"]]
        cell = e["logical_layer"] * EXPERTS + e["expert"]
        grid[s][cell] = max(grid[s][cell], t)
        counts[s][t - 1] += 1

def pack(g):  # 2 bits per cell, 4 cells per byte
    b = bytearray((len(g) + 3) // 4)
    for i, v in enumerate(g):
        b[i >> 2] |= v << ((i & 3) * 2)
    return base64.b64encode(bytes(b)).decode()

summ = json.load(open(f"{RUN}/summary.json", encoding="utf-8"))
card = json.load(open(f"{RUN}/scorecard.json", encoding="utf-8"))

results = []
for g, name in [("l4", "L4"), ("2xl4", "2× L4"), ("a10", "A10G"), ("l40s", "L40S"), ("rtx", "RTX PRO 6000"), ("2xt4", "2× T4")]:
    row = {"gpu": name}
    for t in ["b1-h16-nu", "b1-h64-nu", "b8-h16-nu", "b8-h64-nu"]:
        p = f"runs/p6m-r7-{g}-{t}/scorecard.json"
        if os.path.exists(p):
            c = json.load(open(p, encoding="utf-8"))
            m = c["metrics"].get("decode_tps", {}).get("measured")
            if m is None:  # unregistered cell (2xT4): fall back to runner decode tok/s
                m = json.load(open(f"runs/p6m-r7-{g}-{t}/result.json", encoding="utf-8")).get("decode_tok_s")
            row[t] = round(m, 3) if m else None
    results.append(row)

env = json.load(open("runs/r7_environment.json", encoding="utf-8"))["containers"]

data = {
    "run_id": summ["run_id"], "commit": summ["commit"], "gpu": summ["nvidia"],
    "host_gib": summ["host_gib_effective"], "vram_budget_mib": summ["budget_mib"],
    "store_gib": summ["store_gib"], "b_ssd_gib_s": summ["measured_b_ssd_gib_s"],
    "record_bytes": 13369344, "layers": LAYERS, "experts": EXPERTS,
    "prompts": [clean_prompt(r["prompt"]) for r in rows],
    "elapsed": [s["elapsed_s"] for s in steps],
    "frags": frags, "done_at": done_at,
    "counts": counts, "grid": [pack(g) for g in grid],
    "window_hit_rate": card["metrics"]["host_hit_rate"]["measured"],
    "decode_tps": card["metrics"]["decode_tps"]["measured"],
    "results": results, "env": env,
}
open("demo/demo_data.js", "w", encoding="utf-8").write("window.DEE_DEMO=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
tot = [sum(c[i] for c in counts) for i in range(3)]
print("steps", N, "done_at", done_at, "tiers vram/ram/ssd", tot, "bytes", os.path.getsize("demo/demo_data.js"))
assert sum(tot) == 278029

# standalone page: inline the data so one file opens anywhere (double-click, artifact, recording)
page = open("demo/index.html", encoding="utf-8").read()
tag = '<script src="demo_data.js"></script>'
assert tag in page
js = open("demo/demo_data.js", encoding="utf-8").read()
open("demo/dee-replay.html", "w", encoding="utf-8").write(page.replace(tag, "<script>" + js + "</script>"))
print("wrote demo/dee-replay.html", os.path.getsize("demo/dee-replay.html"))
