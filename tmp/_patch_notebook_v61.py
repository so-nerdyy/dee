import json, os, shutil, sys
from pathlib import Path

NB = Path("dee.cpp/kaggle/ornith-milestone3/ornith_milestone3.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]
print("[PATCH] starting cell count:", len(cells))

# --- modify git-checkout cell (currently index 3): evidence path uses RUN_ID ---
git_src = "".join(cells[3]["source"])
old_evidence_line = "EVIDENCE = Path('/kaggle/working/ornith-milestone3-evidence')"
new_evidence_line = "EVIDENCE = Path(f'/kaggle/working/ornith-milestone3-evidence-{RUN_ID}')"
assert old_evidence_line in git_src, "git checkout cell evidence line not found"
git_src_new = git_src.replace(old_evidence_line, new_evidence_line, 1)
cells[3]["source"] = git_src_new.splitlines(keepends=True)
print("[PATCH] modified cells[3] (git checkout) to use RUN_ID-conditioned EVIDENCE path")

# --- insert at index 2: RUN_ID + COMMIT_EXPECTED env var readout ---
run_id_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, json\n",
        "RUN_ID = os.environ.get('RUN_ID', 'LOCAL_RUN')\n",
        "COMMIT_EXPECTED = os.environ.get('COMMIT_EXPECTED', '4d8ccf2')\n",
        "print(json.dumps({'RUN_ID': RUN_ID, 'COMMIT_EXPECTED': COMMIT_EXPECTED}), flush=True)\n",
    ],
}
cells.insert(2, run_id_cell)
print("[PATCH] inserted cells[2] RUN_ID + COMMIT_EXPECTED env var readout")

# --- cells[5] is now the commit-hash assertion ---
commit_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "EXPECTED = COMMIT_EXPECTED\n",
        "assert commit == EXPECTED, f'kernel preflight rejected: commit mismatch -- got {commit}, expected {EXPECTED}. The instrumentation target must be re-pinned.'\n",
        "print(json.dumps({'preflight_1_commit_PASS': True, 'commit': commit}), flush=True)\n",
    ],
}
cells.insert(5, commit_cell)
print("[PATCH] inserted cells[5] commit-hash hard assertion")

# --- cells[8] is now the build-freshness assertion ---
fresh_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os as _os\n",
        "_so_candidates = list((DEE / 'pydee').glob('*.so'))\n",
        "_so_candidates.extend(BUILD.glob('**/*.so'))\n",
        "assert _so_candidates, 'Preflight #2 rejected: pydee or build-kaggle-cuda produced no .so'\n",
        "_so = _so_candidates[0]\n",
        "_git_index = ROOT / '.git' / 'index'\n",
        "_mtime_diff = _os.path.getmtime(_so) - _os.path.getmtime(_git_index)\n",
        "assert _mtime_diff > -1e-3, f'Preflight #2 rejected: built native extension is older than git checkout (mtime diff={_mtime_diff}s). The build did not run after the fresh checkout.'\n",
        "print(json.dumps({'preflight_2_build_freshness_PASS': True, 'so_path': str(_so), 'mtime_diff_s': _mtime_diff}), flush=True)\n",
    ],
}
cells.insert(8, fresh_cell)
print("[PATCH] inserted cells[8] build-freshness / so-vs-git-index assertion")

nb["cells"] = cells

# --- atomically write back ---
tmp_path = NB.with_suffix(".ipynb.tmp")
tmp_path.write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
)
os.replace(tmp_path, NB)
print("[PATCH] atomic write OK; final cell count:", len(cells))
