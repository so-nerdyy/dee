#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# cmake lives in ~/.local/bin on this WSL; make sure it is on PATH.
export PATH="$HOME/.local/bin:$PATH"
mkdir -p build
rm -rf build-cmake
cmake -S . -B build-cmake > build/cmake_cfg.log 2>&1
echo "CMAKE CONFIG EXIT=$?"
cmake --build build-cmake >> build/cmake_cfg.log 2>&1
echo "CMAKE BUILD EXIT=$?"
for t in dee_step5_tests dee_weight_mmap_tests dee_step6_tests dee_generation_tests; do
  echo "===== $t ====="
  ./build-cmake/$t
done
echo "===== dee_bench (Step 8 CLI driver) ====="
./build-cmake/dee_bench --max-new 64
echo "CMAKE ALL RUN EXIT=$?"
