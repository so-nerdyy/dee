# third_party / vendored dependencies

## ggml (from llama.cpp)

We vendor a subset of the [llama.cpp](https://github.com/ggerganov/llama.cpp)
`ggml/` tree as our tensor + compute backend (CPU reference + CUDA kernels).

The full llama.cpp clone lives at `/mnt/c/Users/carth/Downloads/llama.cpp`
(it was cloned for source analysis only — it is NOT compiled by dee.cpp).

To populate this directory, copy the needed ggml pieces:

    SRC=/mnt/c/Users/carth/Downloads/llama.cpp/ggml
    DST=/mnt/c/Users/carth/Downloads/dee.cpp/third_party/ggml
    mkdir -p "$DST/src" "$DST/include" "$DST/cmake"
    cp -r "$SRC/src/."       "$DST/src/"
    cp -r "$SRC/include/."   "$DST/include/"
    cp    "$SRC/CMakeLists.txt" "$DST/CMakeLists.txt"
    # optional: CUDA backend
    cp -r "$SRC/src/ggml-cuda" "$DST/src/" 2>/dev/null || true

Then the top-level `CMakeLists.txt` does `add_subdirectory(third_party/ggml)`.

Do NOT copy the whole llama.cpp `src/` (llama-*.cpp, models/*) — those are the
high-level model layers we are replacing with our own DEE engine.
