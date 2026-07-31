from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from scripts import m5g_v3_cuda_smoke as smoke


ROOT = Path(__file__).parents[1]


def _metadata(side: str) -> dict:
    return {
        "side": side,
        "layer": 0,
        "token": 0,
        "label": "input_layernorm",
        "selector": smoke.selector(),
        "element_start": 0,
        "element_count": 8,
        "completion_sequence": 1,
        "kernel_identity": smoke.EXPECTED_KERNELS[side],
        "stream_id": 7,
        "epsilon": 1.0e-6,
        "source_dtype": "torch.float16",
        "destination_dtype": "torch.float16",
    }


def _records(side: str) -> list[dict]:
    return [
        {
            "category": category,
            "label": "step=0,layer=0:input_layernorm",
            "array": [[0.0] * 8] if category not in {"norm_variance", "norm_denominator", "reciprocal_rms"} else [0.0],
            "metadata": _metadata(side),
        }
        for category in sorted(smoke.EXPECTED_CATEGORIES)
    ]


def test_smoke_selector_is_bounded_and_exact() -> None:
    selected = smoke.selector()
    assert selected == {
        "token_index": 0,
        "layer_index": 0,
        "norm_label": "input_layernorm",
        "element_start": 0,
        "element_count": 8,
        "flattened_row_index": 0,
    }


def test_validate_side_rejects_duplicate_or_wrong_kernel_records() -> None:
    records = _records("candidate")
    assert smoke.validate_side(records, "candidate", smoke.selector()) == []
    records[0]["metadata"]["kernel_identity"] = "wrong"
    assert any(
        failure["name"] == "candidate_kernel_identity"
        for failure in smoke.validate_side(records, "candidate", smoke.selector())
    )
    records[1] = records[0]
    assert any(
        failure["name"] == "candidate_diagnostic_records_unique"
        for failure in smoke.validate_side(records, "candidate", smoke.selector())
    )


def test_compare_records_reports_bitwise_category_results() -> None:
    control = _records("control")
    candidate = _records("candidate")
    comparison = smoke.compare_records(control, candidate)
    assert set(comparison) == smoke.EXPECTED_CATEGORIES
    assert all(row["bitwise_equal"] for row in comparison.values())


def test_harness_identity_sidecar_matches_committed_source() -> None:
    harness_dir = ROOT / "kaggle/ornith-m5g-v3-smoke"
    identity = json.loads((harness_dir / "harness-identity.json").read_text())
    actual = hashlib.sha256((harness_dir / identity["harness_file"]).read_bytes()).hexdigest()
    assert identity["harness_sha256"] == actual


def test_harness_identity_is_loaded_from_harness_commit_before_runtime_checkout() -> None:
    harness = (ROOT / "kaggle/ornith-m5g-v3-smoke/ornith_m5g_v3_smoke.py").read_text()
    identity_lookup = 'ROOT / "dee.cpp/kaggle/ornith-m5g-v3-smoke/harness-identity.json"'
    assert identity_lookup in harness
    assert harness.index(identity_lookup) < harness.index('subprocess.run(["git", "checkout", EXPECTED_RUNTIME_COMMIT]')
    assert 'Path(__file__).with_name("harness-identity.json")' not in harness


def test_harness_identity_allows_only_kaggle_script_rename() -> None:
    source = (ROOT / "kaggle/ornith-m5g-v3-smoke/ornith_m5g_v3_smoke.py").read_text()
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "is_accepted_harness_filename"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<harness-helper>", "exec"), namespace)
    accepts = namespace["is_accepted_harness_filename"]
    assert accepts("ornith_m5g_v3_smoke.py", "ornith_m5g_v3_smoke.py") is True
    assert accepts("script.py", "ornith_m5g_v3_smoke.py") is True
    assert accepts("arbitrary.py", "ornith_m5g_v3_smoke.py") is False
    assert accepts("script.py", None) is True


def test_v3_kernel_isolated_from_sealed_v2_kernel() -> None:
    v3_metadata = json.loads(
        (ROOT / "kaggle/ornith-m5g-v3-smoke/kernel-metadata.json").read_text()
    )
    v2_metadata = json.loads(
        (ROOT / "kaggle/ornith-m5g-norm-subsets/kernel-metadata.json").read_text()
    )
    assert v3_metadata["id"] != v2_metadata["id"]
    assert v3_metadata["code_file"] == "ornith_m5g_v3_smoke.py"
    assert v2_metadata["code_file"] == "ornith_m5g_norm_subsets.py"
    harness = (ROOT / "kaggle/ornith-m5g-v3-smoke/ornith_m5g_v3_smoke.py").read_text()
    assert "m5g-v3-regular-norm-smoke" in harness
    assert "m5g-v2-execution-equivalent" not in harness
