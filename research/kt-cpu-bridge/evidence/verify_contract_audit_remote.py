"""Read-only remote reachability check, writing only an audit receipt."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BRANCH = "refs/heads/codex/kt-real-expert-review"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run():
    url = git("remote", "get-url", "origin")
    assert url == "https://github.com/so-nerdyy/dee.git"
    remote_line = git("ls-remote", "--exit-code", "origin", BRANCH)
    tip, name = remote_line.split()
    assert name == BRANCH
    commits = {}
    for short in ("be46276", "5650045", "f44967f"):
        commit = git("rev-parse", short + "^{commit}")
        code = subprocess.run(["git", "merge-base", "--is-ancestor", commit, tip], cwd=ROOT).returncode
        assert code == 0, f"{commit} is not reachable from observed remote tip {tip}"
        commits[short] = {"full_sha": commit, "reachable_from_observed_remote_tip": True}
    return {"schema": "kt-contract-audit-remote-visibility-v1", "observed_utc": datetime.now(timezone.utc).isoformat(),
            "remote_url": url, "remote_ref": name, "remote_tip": tip, "ls_remote_output": remote_line,
            "commits": commits, "local_head": git("rev-parse", "HEAD"),
            "local_head_equals_remote_tip": git("rev-parse", "HEAD") == tip}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    opts = parser.parse_args()
    report = run()
    opts.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
