# third_party / vendored dependencies

## ggml (from llama.cpp)

We vendor a subset of the [ggml](https://github.com/ggml-org/ggml) tree, as
shipped inside [llama.cpp](https://github.com/ggml-org/llama.cpp), as our
tensor and compute backend (CPU reference and CUDA kernels). Version 0.17.0
(see `ggml/CMakeLists.txt`). ggml is MIT licensed; its license is in
[`ggml/LICENSE`](ggml/LICENSE).

To refresh this directory from a llama.cpp checkout:

    SRC=/path/to/llama.cpp/ggml
    DST=dee.cpp/third_party/ggml
    mkdir -p "$DST/src" "$DST/include" "$DST/cmake"
    cp -r "$SRC/src/."       "$DST/src/"
    cp -r "$SRC/include/."   "$DST/include/"
    cp    "$SRC/CMakeLists.txt" "$DST/CMakeLists.txt"
    cp    "$SRC/../LICENSE"  "$DST/LICENSE"
    # optional: CUDA backend
    cp -r "$SRC/src/ggml-cuda" "$DST/src/" 2>/dev/null || true

Then the top-level `CMakeLists.txt` does `add_subdirectory(third_party/ggml)`.

Do NOT copy the whole llama.cpp `src/` (llama-*.cpp, models/*). Those are the
high-level model layers that the DEE engine replaces.
