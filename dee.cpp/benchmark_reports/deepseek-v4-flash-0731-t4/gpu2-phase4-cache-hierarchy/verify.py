#!/usr/bin/env python3
"""Phase-4 evidence-bundle verifier.

Recomputes every headline number from the committed bundle and
cross-checks against the Kaggle-side integrity records.  Run from
anywhere:

    python verify.py [bundle_dir]

Exits non-zero if any check fails.  Prints a per-check PASS/FAIL table
and a summary of which PHASE4_RESULTS.md numbers are reproduced.
"""
import hashlib
import json
import sys
from pathlib import Path

BUNDLE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
EV = BUNDLE / "evidence"
EXT = EV / "_extract"
ARMS_FULL = ["a1_asis", "a2_fp4", "a3_fp4p"]
NPROMPTS = 8
EXPECTED_TOKENS = 128
CAMPAIGN_COMMIT = "216ad6553ebd961c6688e335f79f6a7588dd3653"

# headline numbers from PHASE4_RESULTS.md to reproduce
REPORTED = {
    "requests_per_arm": 278879,
    "a1_asis": {"resident_pct": 7.4, "host_hit_pct": 56.1,
                "cold_pct": 36.5, "h2d_gb": 3454, "wall_s": 5848},
    "a2_fp4": {"resident_pct": 45.3, "host_hit_pct": 18.0,
               "cold_pct": 36.7, "h2d_gb": 2038, "wall_s": 5345},
    "a3_fp4p": {"resident_pct": 13.2, "host_hit_pct": 52.7,
                "cold_pct": 34.1, "h2d_gb": 3237, "wall_s": 5717},
    "policy_delta_pp": 32.1,
}

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(p):
    return json.loads(Path(p).read_text())


print("== 1. integrity: commit under test ==")
for arm in ARMS_FULL:
    q0 = jload(EV / arm / "integrity-q0.json")
    check(f"{arm} git_commit", q0.get("git_commit") == CAMPAIGN_COMMIT,
          q0.get("git_commit"))
    check(f"{arm} classification q0",
          q0.get("classification") == "ACCEPT_CORRECTNESS")

print("== 2. per-prompt acceptance + 128 tokens ==")
for arm in ARMS_FULL:
    allok = True
    for i in range(NPROMPTS):
        integ = jload(EV / arm / f"integrity-q{i}.json")
        res = jload(EXT / f"{arm}-result-q{i}.json")
        ok = (integ.get("classification") == "ACCEPT_CORRECTNESS"
              and res.get("n_tokens") == EXPECTED_TOKENS
              and res.get("classification") == "ACCEPT_CORRECTNESS")
        allok &= ok
        if not ok:
            print(f"    {arm} q{i}: integ={integ.get('classification')} "
                  f"res={res.get('classification')} "
                  f"n={res.get('n_tokens')}")
    check(f"{arm} 8x ACCEPT_CORRECTNESS + 128 tok", allok)

print("== 3. token-id sha equality across arms ==")
tok_manifest = {}
for i in range(NPROMPTS):
    shas = {}
    for arm in ARMS_FULL:
        shas[arm] = jload(EXT / f"{arm}-result-q{i}.json")[
            "token_ids_sha256"]
    tok_manifest[f"q{i}"] = shas
    check(f"q{i} token sha equal", len(set(shas.values())) == 1,
          list(set(shas.values()))[0][:16])
(BUNDLE / "token-sha-manifest.json").write_text(
    json.dumps(tok_manifest, indent=1, sort_keys=True))

print("== 4. route-journal sha: recompute + cross-arm equality ==")
journal_manifest = {}
for i in range(NPROMPTS):
    shas = {}
    for arm in ARMS_FULL:
        p = EV / arm / f"routed_experts-q{i}.jsonl"
        shas[arm] = sha_file(p) if p.exists() else None
    journal_manifest[f"q{i}"] = shas
    present = [s for s in shas.values() if s]
    check(f"q{i} journal present all arms", len(present) == 3)
    check(f"q{i} journal sha equal", len(set(present)) == 1,
          present[0][:16] if present else "")
(BUNDLE / "journal-sha-manifest.json").write_text(
    json.dumps(journal_manifest, indent=1, sort_keys=True))

print("== 5. journal sha vs Kaggle-recorded integrity artifact sha ==")
n_checked = 0
n_match = 0
for arm in ARMS_FULL:
    for i in range(NPROMPTS):
        integ = jload(EV / arm / f"integrity-q{i}.json")
        arts = integ.get("artifact_sha256") or {}
        key = next((k for k in arts if "routed_experts" in k), None)
        if key:
            n_checked += 1
            local = journal_manifest[f"q{i}"][arm]
            n_match += (arts[key] == local)
check("journal shas match Kaggle records",
      n_checked > 0 and n_match == n_checked,
      f"{n_match}/{n_checked}")

print("== 6. cache-event aggregates ==")
ce_totals = {}
for arm in ARMS_FULL:
    agg = jload(EXT / f"{arm}-ce-summary.json")
    total = sum(agg.values())
    ce_totals[arm] = agg
    check(f"{arm} total requests == {REPORTED['requests_per_arm']}",
          total == REPORTED["requests_per_arm"], f"{total:,}")

print("== 7. residency / host-hit / cold rates vs report ==")
for arm in ARMS_FULL:
    agg = ce_totals[arm]
    total = sum(agg.values())
    res_pct = 100.0 * agg.get("resident", 0) / total
    host_pct = 100.0 * agg.get("host_hit", 0) / total
    cold_pct = 100.0 * agg.get("cold", 0) / total
    rep = REPORTED[arm]
    check(f"{arm} resident% ~{rep['resident_pct']}",
          abs(res_pct - rep["resident_pct"]) < 0.15, f"{res_pct:.2f}")
    check(f"{arm} host_hit% ~{rep['host_hit_pct']}",
          abs(host_pct - rep["host_hit_pct"]) < 0.15, f"{host_pct:.2f}")
    check(f"{arm} cold% ~{rep['cold_pct']}",
          abs(cold_pct - rep["cold_pct"]) < 0.15, f"{cold_pct:.2f}")

delta = (100.0 * ce_totals["a2_fp4"]["resident"]
         / sum(ce_totals["a2_fp4"].values())
         - 100.0 * ce_totals["a3_fp4p"]["resident"]
         / sum(ce_totals["a3_fp4p"].values()))
check("policy delta (a2-a3) ~32.1pp",
      abs(delta - REPORTED["policy_delta_pp"]) < 0.3, f"{delta:.2f}pp")

print("== 8. H2D + wall totals vs report ==")
for arm in ARMS_FULL:
    h2d = 0
    wall = 0.0
    for i in range(NPROMPTS):
        m = jload(EXT / f"{arm}-result-q{i}.json")
        h2d += m.get("expert_h2d_bytes_total") or 0
        wall += m.get("total_wall_seconds") or 0.0
    rep = REPORTED[arm]
    h2d_gb = h2d / 1e9
    check(f"{arm} H2D ~{rep['h2d_gb']} GB",
          abs(h2d_gb - rep["h2d_gb"]) / rep["h2d_gb"] < 0.02,
          f"{h2d_gb:.0f} GB")
    check(f"{arm} wall ~{rep['wall_s']} s",
          abs(wall - rep["wall_s"]) / rep["wall_s"] < 0.02,
          f"{wall:.0f} s")

print("== 9. checkpoint token chains ==")
for arm in ARMS_FULL:
    ok = True
    for i in range(NPROMPTS):
        p = EV / arm / f"generated_checkpoint-q{i}.jsonl"
        toks = []
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            if "token_id" in rec:
                toks.append(rec["token_id"])
        ok &= (len(toks) == EXPECTED_TOKENS)
    check(f"{arm} checkpoints carry {EXPECTED_TOKENS} token ids", ok)

print("== 10. a0_bypass status ==")
a0_files = sorted(p.name for p in (EV / "a0_bypass").iterdir())
check("a0 partial harvest present (timeout by design)",
      any("integrity" in f or "arm_config" in f for f in a0_files),
      f"{len(a0_files)} files")

print("== 11. SHA256SUMS self-consistency ==")
sums = jload(BUNDLE / "SHA256SUMS.json")
bad = [k for k, v in sums.items()
       if sha_file(BUNDLE / k) != v]
check("manifest recomputes clean", not bad,
      f"{len(sums)} entries, {len(bad)} mismatches")

print()
fails = [r for r in results if not r[1]]
print(f"{'='*60}")
print(f"VERDICT: {len(results)-len(fails)}/{len(results)} checks pass"
      + (f" — {len(fails)} FAILURES" if fails else " — ALL PASS"))
sys.exit(1 if fails else 0)
