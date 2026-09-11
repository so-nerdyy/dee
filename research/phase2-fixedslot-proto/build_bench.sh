#!/bin/sh
# build_bench.sh — bench build (stdlib only, no CUDA, no CMake).
# Toolchain note (measured 2026-09-11): MSYS2 g++ 15.2's cc1plus dies
# silently in this sandbox, so this uses the scoop mstorsjo clang++ which
# builds clean with -Wformat. Either compiler is fine elsewhere.
set -e
D="$(dirname "$0")"
"${CXX:-clang++}" -std=c++17 -O2 -Wformat \
    -I"$D" -I"$D/../../dee.cpp/include" \
    "$D/bench_fixedslot.cpp" \
    "$D/fixed_slot_staging.cpp" \
    "$D/../../dee.cpp/src/host_pack_cache.cpp" \
    -o "$D/bench_fixedslot"
echo "built: $D/bench_fixedslot"
