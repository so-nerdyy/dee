"""P5c isolation analyzer — direct evidence for the cohort
numerical-equivalence contract.

Usage: python p5c_analyze.py <p5c-out/mN dir>

Units (cohort groups): c0 = p0@K1, c1 = p5@K1, c2 = p0..7@K8,
c3 = p0..7@K8 (repeat).  Dumps: iso-c{u}_s{step}_l{layer}_{key}.npy.

Produces:
  1. input-identity check (padded ids byte-equal)
  2. first-divergence bisect: elementwise diff of every dumped tensor,
     member0 of c2 vs c0 — (step, layer, key) -> max_abs/ndiff/rel
  3. boundary-margin table: every (step,layer,row) where member0's
     expert_ids differ from the singleton -> rank-6/rank-7 router-score
     gap on both sides + the hidden-state perturbation at that layer
  4. member5-vs-c1 bitwise comparison (expected: identical)
  5. c2-vs-c3 same-shape bitwise comparison (expected: identical)
  6. token-sha summary from result-c*.json
"""
import json
import sys
from pathlib import Path

import numpy as np

SMALL = ("router_scores", "expert_ids", "routing_weights")
BIG = ("layer_input", "attn_norm_out", "attn_out", "attn_hc_out",
       "ffn_norm_in", "ffn_norm_out", "moe_out", "shared_out", "output")
TOPK = 6


def load(d, unit, step, layer, key):
    p = d / f"iso-c{unit}_s{step}_l{layer}_{key}.npy"
    return np.load(str(p)) if p.is_file() else None


def member_slice(t, member, k_members):
    """Slice member `member` out of a K-member tensor.

    Hidden tensors: [b, s, hc, d] -> member axis 0.
    Router tensors: [b*s, e] -> row-major, member block of s rows.
    expert_ids/routing_weights: [b*s, topk] same rule.
    """
    if t is None:
        return None
    if t.ndim >= 3 and t.shape[0] == k_members:
        return t[member:member + 1]
    if t.ndim == 2 and t.shape[0] % k_members == 0 and k_members > 1:
        s = t.shape[0] // k_members
        return t[member * s:(member + 1) * s]
    return t


def diff_stats(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    ndiff = int((d > 0).sum())
    denom = np.abs(a.astype(np.float64))
    rel = np.where(denom > 0, d / np.maximum(denom, 1e-30), 0.0)
    return {"max_abs": float(d.max()), "ndiff": ndiff,
            "max_rel": float(rel.max()),
            "bitwise": ndiff == 0}


def top6_margin(scores_row):
    """score[rank5] - score[rank6] for one row (descending top-k gap)."""
    s = np.sort(scores_row.astype(np.float64))[::-1]
    if len(s) <= TOPK:
        return None
    return float(s[TOPK - 1] - s[TOPK])


def main(d):
    d = Path(d)
    out = {"dir": str(d)}

    # ---- 0. inventory ----
    dumps = sorted(p.name for p in d.glob("iso-c*_s*_l*_*.npy"))
    units = sorted({n.split("_")[0] for n in dumps})
    out["units"] = units
    out["n_dumps"] = len(dumps)

    # ---- 1. input identity ----
    ids = {}
    for u in range(4):
        t = load(d, u, 0, 0, "ids")
        if t is not None:
            ids[u] = t
    id_checks = []
    if 0 in ids and 2 in ids:
        id_checks.append({"pair": "c0-vs-c2.member0",
                          "equal": bool(np.array_equal(
                              ids[0][0], ids[2][0]))})
    if 1 in ids and 2 in ids:
        id_checks.append({"pair": "c1-vs-c2.member5",
                          "equal": bool(np.array_equal(
                              ids[1][0], ids[2][5]))})
    out["input_identity"] = id_checks

    # ---- 2. first-divergence bisect (c0 singleton vs c2 member0) ----
    first = None
    chain = []
    for step in range(8):
        for layer in range(43):
            keys = list(SMALL)
            if layer <= 3 and step == 0:
                keys += list(BIG)
            for key in keys:
                a = load(d, 0, step, layer, key)
                b = member_slice(load(d, 2, step, layer, key), 0, 8)
                st = diff_stats(a, b)
                if st is None:
                    continue
                rec = {"step": step, "layer": layer, "key": key, **st}
                chain.append(rec)
                if first is None and not st["bitwise"]:
                    first = rec
    out["first_divergence"] = first
    # summary: per (step,layer) the max_abs of layer_input/output
    out["chain_excerpt"] = [
        r for r in chain if r["key"] in
        ("layer_input", "ffn_norm_out", "output", "router_scores")
        and (not r["bitwise"] or r["layer"] <= 3)][:80]

    # ---- 3. boundary margins at flip sites ----
    flips = []
    for step in range(8):
        for layer in range(43):
            e1 = load(d, 0, step, layer, "expert_ids")
            e8 = member_slice(load(d, 2, step, layer, "expert_ids"), 0, 8)
            if e1 is None or e8 is None or e1.shape != e8.shape:
                continue
            diff_rows = np.where((e1 != e8).any(axis=tuple(
                range(1, e1.ndim))))[0]
            if not len(diff_rows):
                continue
            s1 = load(d, 0, step, layer, "router_scores")
            s8 = member_slice(load(d, 2, step, layer, "router_scores"),
                              0, 8)
            h1 = load(d, 0, step, layer, "ffn_norm_out")
            h8 = member_slice(load(d, 2, step, layer, "ffn_norm_out"),
                              0, 8)
            for r in diff_rows[:8]:
                rec = {"step": step, "layer": layer, "row": int(r),
                       "k1_margin": None, "k8_margin": None,
                       "router_max_abs": None, "hidden_max_abs": None}
                if s1 is not None and s8 is not None:
                    rec["k1_margin"] = top6_margin(
                        s1.reshape(-1, s1.shape[-1])[r])
                    rec["k8_margin"] = top6_margin(
                        s8.reshape(-1, s8.shape[-1])[r])
                    rd = np.abs(s1.reshape(-1, s1.shape[-1])[r].astype(
                        np.float64)
                        - s8.reshape(-1, s8.shape[-1])[r].astype(
                        np.float64))
                    rec["router_max_abs"] = float(rd.max())
                if h1 is not None and h8 is not None:
                    # row r in expert_ids maps to seq position r in the
                    # [1, s, hc, d] hidden tensor (batch dim already
                    # sliced to the member).
                    st = diff_stats(
                        h1.reshape(h1.shape[1], -1)[r:r + 1],
                        h8.reshape(h8.shape[1], -1)[r:r + 1])
                    rec["hidden_max_abs"] = (st or {}).get("max_abs")
                flips.append(rec)
    out["flip_sites"] = flips[:200]
    out["n_flip_sites"] = len(flips)

    # ---- 4. member5 (p5) vs c1 singleton — expected bitwise ----
    m5 = []
    for step in range(8):
        for layer in range(43):
            keys = list(SMALL)
            if layer <= 3 and step == 0:
                keys += list(BIG)
            for key in keys:
                a = load(d, 1, step, layer, key)
                b = member_slice(load(d, 2, step, layer, key), 5, 8)
                st = diff_stats(a, b)
                if st is not None and not st["bitwise"]:
                    m5.append({"step": step, "layer": layer, "key": key,
                               **st})
    out["member5_vs_singleton_diffs"] = m5[:100]
    out["member5_bitwise_clean"] = not m5

    # ---- 5. c2 vs c3 same-shape bitwise ----
    det = {"compared": 0, "differing": 0, "first_diff": None}
    for p in d.glob("iso-c2_s*_l*_*.npy"):
        q = d / p.name.replace("iso-c2_", "iso-c3_", 1)
        if not q.is_file():
            continue
        a, b = np.load(str(p)), np.load(str(q))
        det["compared"] += 1
        if not np.array_equal(a, b):
            det["differing"] += 1
            if det["first_diff"] is None:
                det["first_diff"] = p.name
    out["same_shape_determinism"] = det

    # ---- 6. token shas ----
    shas = {}
    for u in range(4):
        rp = d / f"result-c{u}.json"
        if rp.is_file():
            r = json.loads(rp.read_text())
            shas[f"c{u}"] = {
                row.get("prompt_index"): row.get("token_ids_sha256", "")[:16]
                for row in (r.get("rows") or [])}
    out["token_shas"] = shas

    print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
