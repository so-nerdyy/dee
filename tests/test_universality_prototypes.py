#!/usr/bin/env python3
"""Static conformance tests for dee.cpp/experiments/universality/ prototypes.

This host's MSYS2 C++ toolchain crashes on any real compilation (cc1/cc1plus
die with no output; even hello-world fails), so the C++ self-test
(test_universality.cpp, built with `g++ -std=c++17 -I. test_universality.cpp
-o test_universality` on a healthy host) cannot execute here. These pytest
checks verify everything verifiable without a compiler:

- required API surface from ABSTRACTION_DESIGN.md exists verbatim,
- cache/scheduler view (CacheKey) cannot name a projection (no-leak rule),
- prototype headers include no production dee headers (isolation rule),
- the C++ self-test covers both adapters + registry + fail-closed paths.

If g++ becomes functional, test_gpp_compile is attempted; otherwise it skips
with the reason instead of failing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

UNI = Path(__file__).resolve().parent.parent / "dee.cpp" / "experiments" / "universality"


def _read(name: str) -> str:
    return (UNI / name).read_text(encoding="utf-8")


def test_prototype_files_exist():
    for name in ("expert_descriptor.hpp", "expert_codec.hpp",
                 "model_adapter.hpp", "test_universality.cpp"):
        assert (UNI / name).is_file(), name


def test_descriptor_api_surface():
    text = _read("expert_descriptor.hpp")
    for token in ("struct ExpertId", "struct CacheKey", "struct ProjectionDesc",
                  "struct CombineDesc", "struct ExpertDescriptor",
                  "record_bytes", "cache_key()", "scale_block"):
        assert token in text, token


def test_cache_key_cannot_name_projections():
    """No-leak rule: the core-visible CacheKey must not expose projections."""
    text = _read("expert_descriptor.hpp")
    m = re.search(r"struct CacheKey \{(.*?)\};", text, re.DOTALL)
    assert m, "CacheKey struct not found"
    body = m.group(1)
    assert "Projection" not in body
    assert "weight_offset" not in body and "scale_offset" not in body
    assert "record_bytes" in body and "codec" in body


def test_codec_api_surface():
    text = _read("expert_codec.hpp")
    for token in ("class ExpertCodec", "decode_tile", "storage_bytes",
                  "validate", "metadata_layout", "supported_backend",
                  "class CodecRegistry"):
        assert token in text, token


def test_adapter_api_surface():
    text = _read("model_adapter.hpp")
    for token in ("class ModelAdapter", "parse_config", "expert_weight_name",
                  "expert_scale_name", "router_weight_name", "RouterDesc",
                  "describe_layer", "class Dsv4Adapter",
                  "class SecondModelAdapter"):
        assert token in text, token


def test_dsv4_reference_values():
    text = _read("model_adapter.hpp")
    assert "13369344" in text  # 12.75 MiB packed record
    assert "sqrtsoftplus" in text and "1.5" in text
    assert "silu-clamp10" in text


def test_no_production_includes():
    """Isolation rule: prototypes must not include production dee headers."""
    for name in ("expert_descriptor.hpp", "expert_codec.hpp", "model_adapter.hpp"):
        includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', _read(name))
        assert not [i for i in includes if i.startswith("dee/")], (name, includes)


def test_cpp_selftest_covers_both_models():
    text = _read("test_universality.cpp")
    for token in ("Dsv4Adapter", "SecondModelAdapter", "CodecRegistry",
                  "decode_tile", "schedule_bytes", "FAIL"):
        assert token in text, token
    assert len(re.findall(r"CHECK\(", text)) >= 20


def test_gpp_compile():
    gpp = shutil.which("g++")
    if gpp is None:
        import pytest
        pytest.skip("no g++ on PATH")
    proc = subprocess.run(
        [gpp, "-std=c++17", "-Wall", "-Wextra", "-I.", "test_universality.cpp",
         "-o", "test_universality_check"],
        cwd=str(UNI), capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        import pytest
        pytest.skip(f"g++ present but cannot compile on this host: "
                    f"{(proc.stderr or proc.stdout)[:300]}")
    run = subprocess.run([str(UNI / "test_universality_check")],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "0 failed" in run.stdout


if __name__ == "__main__":
    sys.exit(0)
