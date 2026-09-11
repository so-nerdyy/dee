#!/usr/bin/env bash
# Build a reproducible CUDA 13 / sm_75 Lightning Studio checkout.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark=0
if [[ "${1:-}" == "--benchmark" ]]; then
  benchmark=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--benchmark]" >&2
  exit 2
fi

cd "$project_root"

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 1; }
}

require_command nvidia-smi
require_command nvcc
require_command git
require_command python3
require_command git-lfs

echo "== NVIDIA device =="
nvidia-smi
echo "== CUDA compiler =="
nvcc --version
echo "== Git LFS =="
git lfs version

git lfs pull

if [[ ! -s oracle.pt ]]; then
  echo "oracle.pt is missing. Run 'git lfs pull' from $project_root." >&2
  exit 1
fi
if head -c 80 oracle.pt | grep -q "version https://git-lfs.github.com/spec/v1"; then
  echo "oracle.pt is an unresolved Git LFS pointer. Run 'git lfs pull' from $project_root." >&2
  exit 1
fi

python3 tests/gen_synthetic_shard.py tests/data/layer0_shard.safetensors
python3 tests/gen_moe_shard.py tests/data/ornith_moe256.safetensors

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEE_CUDA=ON \
  -DDEE_BUILD_TESTS=OFF \
  -DDEE_BUILD_GGML=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --parallel "$(nproc)"

echo "== CUDA smoke test =="
./build/dee_cli \
  --shard tests/data/ornith_moe256.safetensors \
  --oracle oracle.pt \
  --tokens 1 --warmup 0 --topk 1 --layers 1 --cuda

if [[ "$benchmark" -eq 1 ]]; then
  echo "== Measured benchmark =="
  ./build/dee_cli \
    --shard tests/data/ornith_moe256.safetensors \
    --oracle oracle.pt \
    --tokens 32 --warmup 2 --topk 8 --layers 40 --cuda
fi
