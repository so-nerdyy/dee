# Dynamic Expert Eviction

Research code for dynamic expert eviction and offload experiments, including
the `dee.cpp` native runtime and the Python/Modal experiment drivers.

## Included components

- `dee.cpp/`: native runtime, tests, and its vendored `ggml` dependency.
- `modal_*.py`, `inspect_header.py`, and `test_step3_cache.py`: experiment and
  validation scripts.

Large Kaggle outputs, downloaded model weights, virtual environments, Python
caches, and native build directories are intentionally ignored. They are local
artifacts rather than source needed to reproduce or review the project.

The separate `Downloads/llama.cpp` checkout is a clean upstream checkout and
contains no project-specific changes. `dee.cpp` already includes the `ggml`
sources required by its CMake build, so the unrelated full upstream checkout is
not duplicated here.

## Build

```powershell
cmake -S dee.cpp -B dee.cpp/build
cmake --build dee.cpp/build --config Release
```

See [`dee.cpp/README.md`](dee.cpp/README.md) for runtime details.
