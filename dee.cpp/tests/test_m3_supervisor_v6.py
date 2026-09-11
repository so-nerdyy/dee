import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_PATH = REPO_ROOT / "tmp" / "m3_supervisor_v6.py"
SPEC = importlib.util.spec_from_file_location("m3_supervisor_v6", SUPERVISOR_PATH)
SUPERVISOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUPERVISOR)


class SupervisorRegressionTests(unittest.TestCase):
    def test_manifest_paths_are_normalized_across_platforms(self):
        self.assertEqual(
            SUPERVISOR.normalize_manifest_path(
                r"lifetime-analysis\lifetime_report.json"
            ),
            "lifetime-analysis/lifetime_report.json",
        )
        with self.assertRaises(ValueError):
            SUPERVISOR.normalize_manifest_path(r"..\outside.json")

    def test_prefixed_kaggle_error_is_terminal_error(self):
        raw = (
            'nivind/dee-cpp-ornith-milestone-3-forensics has status '
            '"KernelWorkerStatus.ERROR"\n'
        )
        self.assertEqual(SUPERVISOR.parse_status(raw), "ERROR")

    def test_kaggle_primary_path_receives_environment(self):
        seen = {}

        def fake_run(command, timeout=120, env=None):
            seen["command"] = command
            seen["env"] = env
            return 0, "ok"

        with mock.patch.object(SUPERVISOR, "run", side_effect=fake_run):
            rc, _ = SUPERVISOR.kaggle_invoke(
                ["kernels", "status", "owner/slug"],
                env={"KAGGLE_API_TOKEN": "sentinel"},
            )
        self.assertEqual(rc, 0)
        self.assertEqual(seen["env"]["KAGGLE_API_TOKEN"], "sentinel")

    def test_kernel_stage_injects_run_and_commit_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "kernel"
            stage_root = root / "stage-root"
            kernel.mkdir()
            stage_root.mkdir()
            notebook = {
                "cells": [{
                    "cell_type": "code",
                    "source": [
                        "RUN_ID = os.environ.get('RUN_ID', 'LOCAL_RUN')\n",
                        "COMMIT_EXPECTED = os.environ.get('COMMIT_EXPECTED', 'old')\n",
                        "HARNESS_NONCE = os.environ.get('HARNESS_NONCE', 'old')\n",
                    ],
                }]
            }
            (kernel / "kernel-metadata.json").write_text(
                json.dumps({"code_file": "job.ipynb"}), encoding="utf-8"
            )
            original = json.dumps(notebook)
            (kernel / "job.ipynb").write_text(original, encoding="utf-8")
            with mock.patch.object(SUPERVISOR, "ROOT", stage_root):
                stage, identity = SUPERVISOR._stage_kernel(
                    kernel, "run-123", "deadbeef"
                )
            staged = json.loads((stage / "job.ipynb").read_text(encoding="utf-8"))
            source = "".join(staged["cells"][0]["source"])
            self.assertIn("RUN_ID = 'run-123'", source)
            self.assertIn("COMMIT_EXPECTED = 'deadbeef'", source)
            self.assertRegex(source, r"HARNESS_NONCE = '[0-9a-f]{32}'")
            self.assertEqual(
                (kernel / "job.ipynb").read_text(encoding="utf-8"), original
            )
            self.assertEqual(identity["run_id"], "run-123")
            self.assertEqual(len(identity["harness_nonce"]), 32)
            self.assertEqual(len(identity["notebook_sha256"]), 64)

    def test_missing_manifest_fails_even_if_kaggle_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            with mock.patch.object(
                    SUPERVISOR, "log", return_value=None):
                self.assertFalse(
                    SUPERVISOR.validate_artifacts(evidence, "run-123")
                )
            report = json.loads(
                (evidence / "artifact-validation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["result"], "FAIL")


if __name__ == "__main__":
    unittest.main()
