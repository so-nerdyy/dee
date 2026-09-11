#!/usr/bin/env python3
"""M3 v6.2 harness-only patcher.

Applies three narrow co-routines:
  1. Supervisor (tmp/m3_supervisor_v6.py) `run()` merges PYTHONUTF8 +
     PYTHONIOENCODING into every `env=` so child Kaggle CLI runs never
     decode via cp1252 charmap (the v6.1 supervisor charmap crash source).
  2. Analyzer (tmp/m3_analyze_trace.py) sets the same env vars at the
     module top BEFORE any open()/write_text() so the partial-write crash
     at line ~518 can't recur on non-ASCII forensic characters.
  3. Notebook cells[5] (commit-hash) and cells[8] (build-freshness) have
     their `assert ...` lines replaced with a try/except wrap that:
        - writes the failure to /kaggle/working/preflight_failure.txt
        - prints the failure with `flush=True`
        - sleeps 60s so the supervisor's live_tail heartbeat has time to
          scrape the traceback before the kernel exits
        - re-raises so the kernel still ends up in ERROR state (the
          RUN_ID/evidence-dir hygiene is preserved).

No edits under dee.cpp/src/, dee.cpp/include/, or dee.cpp/pydee/.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT.parent / "dee.cpp" / "kaggle" / "ornith-milestone3" / "ornith_milestone3.ipynb"
SUP = ROOT / "m3_supervisor_v6.py"
ANL = ROOT / "m3_analyze_trace.py"


def assert_in(text, needle, label):
    if needle not in text:
        raise SystemExit(f"FATAL: anchor for {label} not found")
    return text


print(f"[v6.2] supervisor={SUP}")
print(f"[v6.2] analyzer={ANL}")
print(f"[v6.2] notebook={NB}")

# ---------- 1. supervisor run(): PYTHONUTF8 env merge --------------
sup_src = SUP.read_text(encoding="utf-8")
old_run_anchor = (
    "def run(cmd, timeout=120, env=None):\n"
    "    \"\"\"Run shell command; return (rc, combined stdout+stderr text).\"\"\"\n"
    "    try:\n"
    "        cp = subprocess.run(\n"
    "            cmd, capture_output=True, text=True,\n"
    "            encoding=\"utf-8\", errors=\"replace\",\n"
    "            timeout=timeout, env=env,\n"
    "        )\n"
    "        return cp.returncode, (cp.stdout or \"\") + (cp.stderr or \"\")\n"
)
new_run_body = (
    "def run(cmd, timeout=120, env=None):\n"
    "    \"\"\"Run shell command; return (rc, combined stdout+stderr text).\n"
    "\n"
    "    v6.2 / FIX-12: merge PYTHONUTF8 + PYTHONIOENCODING into every\n"
    "    child env so subprocess Python (including the Kaggle CLI) never\n"
    "    defaults to cp1252 charmap and crashes mid-write on non-ASCII\n"
    "    bytes captured from the kernel log.\n"
    "    \"\"\"\n"
    "    merged_env = os.environ.copy()\n"
    "    if env:\n"
    "        merged_env.update(env)\n"
    "    merged_env.setdefault(\"PYTHONUTF8\", \"1\")\n"
    "    merged_env.setdefault(\"PYTHONIOENCODING\", \"utf-8\")\n"
    "    try:\n"
    "        cp = subprocess.run(\n"
    "            cmd, capture_output=True, text=True,\n"
    "            encoding=\"utf-8\", errors=\"replace\",\n"
    "            timeout=timeout, env=merged_env,\n"
    "        )\n"
    "        return cp.returncode, (cp.stdout or \"\") + (cp.stderr or \"\")\n"
)
assert_in(sup_src, old_run_anchor, "supervisor run() body")
sup_src = sup_src.replace(old_run_anchor, new_run_body, 1)
SUP.write_text(sup_src, encoding="utf-8")
print("[v6.2] supervisor: run() merges PYTHONUTF8 into env")

# ---------- 2. analyzer: PYTHONUTF8 setdefault at module-top -------
anl_src = ANL.read_text(encoding="utf-8")
old_anl_anchor = (
    "from __future__ import annotations\n\n"
    "import argparse\n"
    "import json\n"
    "import os\n"
    "import re\n"
    "import sys\n"
    "from collections import Counter, defaultdict\n"
)
new_anl_top = (
    "from __future__ import annotations\n\n"
    "#  v6.2 / FIX-12: analyzer stdio must be UTF-8 BEFORE the first\n"
    "#  open()/write_text(), otherwise the partial-write crash at line 518\n"
    "#  (\"'charmap' codec can't encode characters in position 10xxx-10yyy\")\n"
    "#  re-fires whenever the log contains non-ASCII forensic bytes.\n"
    "import os as _anl_os\n"
    "_anl_os.environ.setdefault(\"PYTHONUTF8\", \"1\")\n"
    "_anl_os.environ.setdefault(\"PYTHONIOENCODING\", \"utf-8\")\n\n"
    "import argparse\n"
    "import json\n"
    "import os\n"
    "import re\n"
    "import sys\n"
    "from collections import Counter, defaultdict\n"
)
assert_in(anl_src, old_anl_anchor, "analyzer module-top")
anl_src = anl_src.replace(old_anl_anchor, new_anl_top, 1)
ANL.write_text(anl_src, encoding="utf-8")
print("[v6.2] analyzer: PYTHONUTF8 setdefault at top")

# ---------- 3. notebook preflight asserts: log+sleep+raise wrap -----
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]
print(f"[v6.2] notebook: {len(cells)} cells before patch")

PREFLIGHT_CELL_INDICES = [5, 8]


def split_with_newlines(s):
    """Split `s` into a list of strings each ending with \\n."""
    parts = []
    rest = s
    while True:
        idx = rest.find("\n")
        if idx == -1:
            if rest:
                parts.append(rest)
            return parts
        parts.append(rest[: idx + 1])
        rest = rest[idx + 1 :]


for idx in PREFLIGHT_CELL_INDICES:
    cell = cells[idx]
    raw_src = cell.get("source", [])
    if isinstance(raw_src, str):
        src_lines = [raw_src]
    else:
        src_lines = list(raw_src)
    new_src: list[str] = []
    wrapped_here = 0
    for line in src_lines:
        if line.lstrip().startswith("assert ") and wrapped_here == 0:
            stripped = line.strip()
            indent = line[: len(line) - len(line.lstrip())]
            preflight_msg_var = (
                indent
                + "_preflight_msg_cell"
                + str(idx)
                + " = "
                + repr(stripped)
                + "\n"
            )
            safe_lines = [
                preflight_msg_var,
                indent + "_preflight_ok_cell" + str(idx) + " = False\n",
                indent + "try:\n",
                indent + "    " + stripped + "\n",
                indent + "    _preflight_ok_cell" + str(idx) + " = True\n",
                indent + "except BaseException as _preflight_exc:\n",
                indent + "    import time as _preflight_t, traceback as _preflight_tb\n",
                indent + "    try:\n",
                indent + "        with open('/kaggle/working/preflight_failure.txt', 'w', encoding='utf-8') as _pf:\n",
                indent + "            _pf.write('notebook cell[" + str(idx) + "] preflight failed\\n')\n",
                indent + "            _pf.write(repr(_preflight_exc) + '\\n')\n",
                indent + "            _pf.write('original assertion: ' + _preflight_msg_cell" + str(idx) + " + '\\n')\n",
                indent + "            _pf.write(_preflight_tb.format_exc() + '\\n')\n",
                indent + "    except Exception:\n",
                indent + "        pass\n",
                indent + "    print('PREFLIGHT cell[" + str(idx) + "] FAILED: ' + repr(_preflight_exc), flush=True)\n",
                indent + "    _preflight_t.sleep(60)\n",
                indent + "    raise\n",
            ]
            new_src.extend(safe_lines)
            wrapped_here += 1
        else:
            new_src.append(line)
    if wrapped_here == 0:
        print(f"[v6.2] WARN: cell[{idx}] had no assert line to wrap")
    else:
        print(f"[v6.2] notebook cell[{idx}]: "
              f"wrapped {wrapped_here} assert line(s) with log+sleep+raise")
    cell["source"] = new_src

NB.write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
)
print(f"[v6.2] notebook saved (still {len(cells)} cells)")
print("[v6.2] DONE")
