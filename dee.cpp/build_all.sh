#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p build
SRC="src/vram_cache_manager.cpp src/async_prefetcher.cpp src/weight_mmap.cpp src/oracle_scheduler.cpp src/moe_forward.cpp src/generation.cpp src/cuda_backend_mock.cpp src/cuda_backend_cuda.cpp"
g++ -std=c++17 -Wall -Wextra -Iinclude $SRC tests/test_vram_prefetch.cpp -o build/dee_step5_tests
g++ -std=c++17 -Wall -Wextra -Iinclude $SRC tests/test_weight_mmap.cpp -o build/dee_weight_mmap_tests
g++ -std=c++17 -Wall -Wextra -Iinclude $SRC tests/test_step6.cpp -o build/dee_step6_tests
g++ -std=c++17 -Wall -Wextra -Iinclude $SRC tests/test_generation.cpp -o build/dee_generation_tests
echo "BUILD OK"
for t in dee_step5_tests dee_weight_mmap_tests dee_step6_tests dee_generation_tests; do
  echo "===== $t ====="
  ./build/$t
done
echo "===== dee_bench (Step 8 CLI driver) ====="
g++ -std=c++17 -Wall -Wextra -Iinclude $SRC tools/dee_bench.cpp -o build/dee_bench
./build/dee_bench --max-new 64
echo "ALL RUN EXIT=$?"
