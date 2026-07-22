# dee.cpp/pydee/setup.py
#
# pybind11 Python build of pydee (the dee.cpp MoE expert engine binding).
# Builds the shared library against the existing dee_core static library
# produced by the parent CMake build.
#
# Build (from the parent dee.cpp/ directory):
#   cmake --build build --parallel 4
#   python3 pydee/setup.py build_ext --inplace
#
# Workflow:
#   - The CMake build produces build/libdee_core.a (or dee_core.lib on Windows).
#   - This script invokes g++ (or MSVC) via pybind11's compile helpers, links
#     against dee_core, ZLIB, and the standard library.
#   - Output: pydee.cpython-*.so alongside pydee/__init__.py.

import os
import sys

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "pybind11 is not installed. Run:\n"
        "  python3 -m pip install pybind11 --user --break-system-packages\n"
        "Original error: " + repr(exc) + "\n"
    )
    sys.exit(1)

DEE_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BUILD_DIR = os.environ.get("DEE_BUILD_DIR", os.path.join(DEE_ROOT, "build"))
SRC_PYDEE = os.path.join(DEE_ROOT, "pydee", "pydee.cpp")
INC_PYDEE = os.path.join(DEE_ROOT, "include")

# Locate the dee_core static library (.a on Linux/macOS, .lib on Windows).
if sys.platform.startswith("win"):
    STATIC_LIB = os.path.join(BUILD_DIR, "Release", "dee_core.lib")
else:
    STATIC_LIB = os.path.join(BUILD_DIR, "libdee_core.a")
if not os.path.exists(STATIC_LIB):
    candidates = []
    for root, _, files in os.walk(BUILD_DIR):
        for f in files:
            if f in ("libdee_core.a", "dee_core.lib"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        sys.stderr.write(
            "dee_core not found in " + BUILD_DIR + ".\n"
            "Run the CMake build first (cmake --build build).\n"
        )
        sys.exit(1)
    STATIC_LIB = candidates[0]

DEE_INCLUDES = [
    INC_PYDEE,
    os.path.join(DEE_ROOT, "third_party", "ggml", "include"),
]

DEE_LIBS = [STATIC_LIB]

ext_modules = [
    Pybind11Extension(
        "pydee",
        [SRC_PYDEE],
        include_dirs=DEE_INCLUDES,
        library_dirs=[os.path.dirname(STATIC_LIB)],
        libraries=["dee_core", "z", "stdc++"],
        extra_compile_args=["-std=c++17", "-O3"],
    )
]

setup(
    name="pydee",
    version="0.1.0",
    description="dee.cpp MoE expert engine - Python binding",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
