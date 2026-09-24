"""P5b v2 artifact analyzer.

Usage: python p5b_analyze.py <p5b-v2-out-dir>
Prints per-arm divergence analysis from p5b_report.json + route_weights
journals. Read-only.
"""
import json
import sys
from pathlib import Path

def load_journal(path):
    rows = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        rows[(r["step"], r["layer"])] = r
    return rows

def analyze_arm(arm_dir):
    """Diff route_weights-q{0,1,2}.jsonl pairwise vs unit 0."""
    units = []
    for ui in range(4):
        p = arm_dir / f"route_weights-q{ui}.jsonl"
        if p.is_file():
            units.append((ui, load_journal(p)))
    if len(units) < 2:
        return {"units_found": len(units)}
    ref_ui, ref = units[0]
    out = {"units_found": len(units), "unit_ids": [u for u, _ in units],
           "first_div_weight": None, "first_div_ids": None,
           "n_records": len(ref), "div_counts": {}, "identical": {}}
    for ui, rows in units[1:]:
        n_div_w = 0
        n_div_i = 0
        for key in sorted(ref):
            if key not in rows:
                continue
            if rows[key].get("weights_sha256") != ref[key].get("weights_sha256"):
                n_div_w += 1
            if rows[key].get("ids_sha256") != ref[key].get("ids_sha256"):
                n_div_i += 1
        out["div_counts"][ui] = {"weights": n_div_w, "ids": n_div_i}
        out["identical"][ui] = (n_div_w == 0 and n_div_i == 0)
    for key in sorted(ref):
        for ui, rows in units[1:]:
            got = rows.get(key)
            if got is None:
                continue
            if out["first_div_weight"] is None and \
               got.get("weights_sha256") != ref[key].get("weights_sha256"):
                out["first_div_weight"] = {"step": key[0], "layer": key[1],
                                           "unit": ui,
                                           "ref": ref[key]["weights_sha256"][:16],
                                           "got": got["weights_sha256"][:16]}
            if out["first_div_ids"] is None and \
               got.get("ids_sha256") != ref[key].get("ids_sha256"):
                out["first_div_ids"] = {"step": key[0], "layer": key[1],
                                        "unit": ui,
                                        "ref": ref[key]["ids_sha256"][:16],
                                        "got": got["ids_sha256"][:16]}
    return out

def main():
    root = Path(sys.argv[1])
    out = root / "p5b-out"
    rep_path = out / "p5b_report.json"
    if rep_path.is_file():
        rep = json.loads(rep_path.read_text())
        print("=== p5b_report.json ===")
        print("verdict:", rep.get("verdict"))
        print("commit:", rep.get("commit"))
        for c in rep.get("checks", []):
            print(f"  check {'PASS' if c['ok'] else 'FAIL'}: {c['check']} | {str(c.get('detail'))[:120]}")
        print("\n=== analysis (report) ===")
        print(json.dumps(rep.get("analysis"), indent=1))
        print("\n=== interpretation (report) ===")
        print(json.dumps(rep.get("interpretation"), indent=1))
        print("\n=== runs (report) ===")
        for k, v in (rep.get("runs") or {}).items():
            print(f"  {k}: rc={v.get('rc')} wall={v.get('wall_s')} "
                  f"class={v.get('classification')} "
                  f"row_shas={v.get('row_shas')}")
    else:
        print("NO p5b_report.json at", rep_path)
    print("\n=== direct journal diffs ===")
    for arm in ("mA", "mB", "mC"):
        arm_dir = out / arm
        if not arm_dir.is_dir():
            print(f"{arm}: dir missing")
            continue
        res = analyze_arm(arm_dir)
        print(f"{arm}: {json.dumps(res)}")

if __name__ == "__main__":
    main()
