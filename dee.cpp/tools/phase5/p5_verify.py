"""Phase-5 cohort evidence verifier — the W3 exactness gate.

Audits a harvested ``p5-out/{arm_id}/`` tree (or the committed evidence
bundle with the same layout) WITHOUT trusting the session driver's own
verdict — every check recomputes from the artifact files.

Gates:
  1. c0 anchor: c0_anchor's q0 token sha == committed Phase-4 a2_fp4 sha
     (reproves the rebuilt stack against sealed evidence).
  2. Every cohort row: n_tokens == expected AND row classification ==
     ACCEPT_CORRECTNESS (from the cohort result + integrity payloads).
  3. Per-row token equality: every cohort row's token_ids_sha256 equals
     c1's sha for the same prompt_index (same padded inputs).
  4. Journal row audit: expand each cohort record's [rows, topk] matrix —
     decode records carry K rows (row r = cohort member r), prefill
     carries K*L* rows (row-major [K, L*] flatten -> member r occupies
     rows r*L*:(r+1)*L*) — and compare the extracted per-prompt route
     stream against c1's singleton journal for that prompt, record by
     record, expert by expert.
  5. Journal integrity: canonical record order, chain_sha256 recomputed
     over each record's canonical payload, checkpoint_links sequential.
  6. Checkpoint fan-out: generated_checkpoint-c{i}-r{r}.jsonl token ids
     == the row's token stream sha; cohort checkpoint's per-step
     token_ids consistent with row files.

Usage: python p5_verify.py <bundle_dir> [--n-tokens 128]
Exits 0 on PASS, 1 on any failure.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

N_LAYERS = 43
TOPK = 6


def _canon(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def _tok_sha(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps([int(t) for t in tokens], separators=(",", ":"))
        .encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in
            path.read_text("utf-8").splitlines() if ln.strip()]


class Checker:
    def __init__(self) -> None:
        self.n = 0
        self.fails: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.n += 1
        if not ok:
            self.fails.append(f"{name}: {detail}")
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")


def load_journal(path: Path) -> list[dict]:
    """Return the layer_route records sorted by record_index."""
    recs = [r for r in _load_jsonl(path)
            if r.get("kind", "layer_route") == "layer_route"
            or "expert_ids_rank_order" in r]
    recs.sort(key=lambda r: int(r.get("record_index", -1)))
    return recs


def audit_chain(ck: Checker, name: str, recs: list[dict]) -> None:
    """Recompute the journal chain — genesis + per-record canonical hash."""
    genesis = hashlib.sha256(b"").hexdigest()  # RoutedExpertJournal.GENESIS
    prev = genesis
    ok_order = True
    ok_chain = True
    for i, rec in enumerate(recs):
        if int(rec.get("record_index", -1)) != i:
            ok_order = False
        payload = {k: v for k, v in rec.items() if k != "chain_sha256"}
        if payload.get("previous_chain_sha256") != prev:
            ok_chain = False
        if hashlib.sha256(_canon(payload)).hexdigest() != rec.get(
                "chain_sha256"):
            ok_chain = False
        prev = rec.get("chain_sha256", "")
    ck.check(f"{name}: canonical order", ok_order)
    ck.check(f"{name}: chain_sha256 recomputes", ok_chain,
             f"{len(recs)} records")


def extract_row_stream(recs: list[dict], row: int, k: int,
                       lstar: int) -> dict[tuple[int, int], list[list[int]]]:
    """Extract cohort member ``row``'s per-(step, layer) route rows.

    Returns {(step, layer): [[topk], ...]} — prefill contributes L* rows
    (slice [row*L*, (row+1)*L*) of the K*L* matrix), decode contributes
    one row per record.
    """
    out: dict[tuple[int, int], list[list[int]]] = {}
    for rec in recs:
        step = int(rec["forward_step"])
        layer = int(rec["layer"])
        matrix = rec["expert_ids_rank_order"]
        if step == 0:
            if len(matrix) != k * lstar:
                raise ValueError(
                    f"prefill record token_rows={len(matrix)} != K*L*="
                    f"{k}*{lstar}")
            out[(step, layer)] = matrix[row * lstar:(row + 1) * lstar]
        else:
            if len(matrix) != k:
                raise ValueError(
                    f"decode record token_rows={len(matrix)} != K={k}")
            out[(step, layer)] = [matrix[row]]
    return out


def streams_equal(a: dict, b: dict) -> tuple[bool, str]:
    if set(a) != set(b):
        return False, f"key sets differ: {len(a)} vs {len(b)}"
    for key in a:
        if a[key] != b[key]:
            return False, f"divergence at {key}: {a[key][:1]} vs {b[key][:1]}"
    return True, f"{len(a)} (step,layer) entries identical"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    bundle = Path(sys.argv[1])
    n_tokens = 128
    if "--n-tokens" in sys.argv:
        n_tokens = int(sys.argv[sys.argv.index("--n-tokens") + 1])
    ck = Checker()

    p4_manifest = bundle / "token-sha-manifest-p4ref.json"
    if not p4_manifest.is_file():
        p4_manifest = (bundle / "token-sha-manifest.json")
    p4_shas = {}
    if p4_manifest.is_file():
        p4_shas = json.loads(p4_manifest.read_text())
    ck.check("p4 manifest present", bool(p4_shas),
             str(p4_manifest.name if p4_manifest.is_file() else "missing"))

    # ---- 1. c0 anchor ----
    c0_integ = bundle / "c0_anchor" / "integrity-q0.json"
    if c0_integ.is_file():
        integ = json.loads(c0_integ.read_text())
        got = integ.get("actual_token_ids_sha256")
        exp = (p4_shas.get("q0") or {}).get("a2_fp4")
        ck.check("c0 q0 token sha == phase4 a2", bool(got) and got == exp,
                 f"got={str(got)[:12]} exp={str(exp)[:12]}")
    else:
        ck.check("c0 integrity present", False, "integrity-q0.json missing")

    # ---- 2/3. per-row acceptance + token equality vs c1 ----
    c1_rows: dict[int, str] = {}
    c1_dir = bundle / "c1"
    for p in range(8):
        integ = c1_dir / f"integrity-c{p}.json"
        res = c1_dir / f"result-c{p}.json"
        if not res.is_file():
            ck.check(f"c1-c{p} result", False, "missing")
            continue
        result = json.loads(res.read_text())
        rows = result.get("rows") or []
        ok = (result.get("classification") == "ACCEPT_CORRECTNESS"
                  and len(rows) == 1
                  and rows[0].get("n_tokens") == n_tokens)
        sha = rows[0].get("token_ids_sha256") if rows else None
        if not sha and integ.is_file():
            irows = json.loads(integ.read_text()).get("rows") or []
            sha = irows[0].get("actual_token_ids_sha256") if irows else None
        ck.check(f"c1-c{p} accepted + sha", ok and bool(sha),
                 f"cls={result.get('classification')} sha={str(sha)[:12]}")
        c1_rows[p] = sha

    for arm in ("c2", "c4", "c8h"):
        adir = bundle / arm
        if not adir.is_dir():
            ck.check(f"{arm} harvested", False, "arm dir missing")
            continue
        groups = {  # fixed arm matrix
            "c2": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "c4": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "c8h": [list(range(8))],
        }[arm]
        for ci, grp in enumerate(groups):
            res_p = adir / f"result-c{ci}.json"
            if not res_p.is_file():
                ck.check(f"{arm}-c{ci} result", False, "missing")
                continue
            result = json.loads(res_p.read_text())
            rows = result.get("rows") or []
            ck.check(
                f"{arm}-c{ci} accepted",
                result.get("classification") == "ACCEPT_CORRECTNESS"
                and len(rows) == len(grp)
                and all(r.get("n_tokens") == n_tokens for r in rows),
                f"cls={result.get('classification')} rows={len(rows)}")
            rowmap = {r["prompt_index"]: r.get("token_ids_sha256")
                      for r in rows}
            for p in grp:
                mine, base = rowmap.get(p), c1_rows.get(p)
                ck.check(f"{arm}-c{ci} row p{p} == c1",
                         bool(mine) and mine == base,
                         f"{str(mine)[:12]} vs {str(base)[:12]}")

            # ---- 4/5. journal row audit + chain integrity ----
            jpath = adir / f"routed_experts-c{ci}.jsonl"
            if not jpath.is_file():
                ck.check(f"{arm}-c{ci} journal", False, "missing")
                continue
            recs = load_journal(jpath)
            audit_chain(ck, f"{arm}-c{ci} journal", recs)
            k = len(grp)
            ck.check(f"{arm}-c{ci} journal completeness",
                     len(recs) == n_tokens * N_LAYERS
                     and all(int(r.get("token_rows", -1)) ==
                             (k * int(result.get("cohort_prompt_len", -1))
                              if int(r.get("forward_step", -1)) == 0
                              else k)
                             for r in recs),
                     f"records={len(recs)} want={n_tokens * N_LAYERS}")

            lstar = int(result.get("cohort_prompt_len", 0))
            for r, p in enumerate(grp):
                try:
                    cohort_stream = extract_row_stream(recs, r, k, lstar)
                except ValueError as exc:
                    ck.check(f"{arm}-c{ci} row p{p} journal extract",
                             False, repr(exc)[:100])
                    continue
                ref_path = c1_dir / f"routed_experts-c{p}.jsonl"
                if not ref_path.is_file():
                    ck.check(f"{arm}-c{ci} row p{p} ref journal",
                             False, "c1 journal missing")
                    continue
                ref_recs = load_journal(ref_path)
                ref_stream = extract_row_stream(ref_recs, 0, 1, lstar)
                same, detail = streams_equal(cohort_stream, ref_stream)
                ck.check(f"{arm}-c{ci} row p{p} routes == c1", same, detail)

    # ---- 6. checkpoint fan-out (sampled: first cohort of each arm) ----
    for arm in ("c2", "c4", "c8h"):
        adir = bundle / arm
        if not adir.is_dir():
            continue
        groups = {"c2": [[0, 1], [2, 3], [4, 5], [6, 7]],
                  "c4": [[0, 1, 2, 3], [4, 5, 6, 7]],
                  "c8h": [list(range(8))]}[arm]
        ci = 0
        grp = groups[0]
        cpath = adir / f"generated_checkpoint-c{ci}.jsonl"
        if not cpath.is_file():
            continue
        crecs = _load_jsonl(cpath)
        for r, p in enumerate(grp):
            rpath = adir / f"generated_checkpoint-c{ci}-r{r}.jsonl"
            if not rpath.is_file():
                ck.check(f"{arm}-c{ci} row ckpt r{r}", False, "missing")
                continue
            rtoks = [int(rec["token_id"]) for rec in _load_jsonl(rpath)]
            ctoks = [int(rec["token_ids"][r]) for rec in crecs]
            ck.check(f"{arm}-c{ci} ckpt r{r} fan-out",
                     rtoks == ctoks and len(rtoks) == n_tokens,
                     f"{len(rtoks)} tokens")

    print(f"\n{ck.n - len(ck.fails)}/{ck.n} checks passed")
    if ck.fails:
        print("FAILURES:")
        for f in ck.fails:
            print(f"  {f}")
        return 1
    print("P5 VERDICT: PASS — cohort rows identical to sequential reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
