#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p build
# Exercise the REAL DEE_CUDA backend path using the cuda_runtime.h stub.
SRC="src/vram_cache_manager.cpp src/async_prefetcher.cpp src/weight_mmap.cpp src/oracle_scheduler.cpp src/moe_forward.cpp src/generation.cpp src/cuda_backend_cuda.cpp"
g++ -std=c++17 -Wall -Wextra -DDEE_CUDA -Iinclude -Istub $SRC tests/test_vram_prefetch.cpp -o build/dee_step5_cuda_stub
g++ -std=c++17 -Wall -Wextra -DDEE_CUDA -Iinclude -Istub $SRC tests/test_weight_mmap.cpp -o build/dee_weight_mmap_cuda_stub
g++ -std=c++17 -Wall -Wextra -DDEE_CUDA -Iinclude -Istub $SRC tests/test_step6.cpp -o build/dee_step6_cuda_stub
g++ -std=c++17 -Wall -Wextra -DDEE_CUDA -Iinclude -Istub $SRC tests/test_generation.cpp -o build/dee_generation_cuda_stub
echo "DEE_CUDA BUILD OK (all sources compile against real backend headers)"
for t in dee_step5_cuda_stub dee_weight_mmap_cuda_stub dee_step6_cuda_stub dee_generation_cuda_stub; do
  echo "===== $t ====="
  ./build/$t
done
echo "===== dee_bench_cuda_stub (Step 8 CLI driver) ====="
g++ -std=c++17 -Wall -Wextra -DDEE_CUDA -Iinclude -Istub $SRC tools/dee_bench.cpp -o build/dee_bench_cuda_stub
./build/dee_bench_cuda_stub --max-new 64
echo "RUN EXIT=$?"
