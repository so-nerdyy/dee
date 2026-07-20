#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p build
g++ -std=c++17 -Wall -Wextra -Iinclude \
  src/vram_cache_manager.cpp \
  src/async_prefetcher.cpp \
  src/cuda_backend_mock.cpp \
  src/cuda_backend_cuda.cpp \
  tests/test_vram_prefetch.cpp \
  -o build/dee_step5_tests
echo "BUILD OK"
./build/dee_step5_tests
echo "RUN EXIT=$?"
