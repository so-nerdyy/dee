#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

binary="${DEE_BINARY:-./build/dee_cli}"
output_dir="${DEE_PROFILE_OUTPUT:-benchmark_reports/controlled}"
mkdir -p "$output_dir"

if [[ ! -x "$binary" ]]; then
  echo "profile_scenarios: benchmark binary is missing or not executable: $binary" >&2
  echo "Run ./scripts/setup_lightning_t4.sh first." >&2
  exit 1
fi

common=(
  --shard tests/data/ornith_moe256.safetensors
  --oracle oracle.pt
  --tokens 32
  --warmup 2
  --topk 8
  --layers 40
  --cuda
  --profile-stages
)

scenarios=(
  end-to-end
  full-resident
  resident-bypass
  transfer-only
  compute-only
  oracle-only
  cache-metadata-only
)

for scenario in "${scenarios[@]}"; do
  echo "=== controlled profile: $scenario ==="
  "$binary" "${common[@]}" \
    --profile-scenario "$scenario" \
    --profile-json "$output_dir/$scenario.json"
done

echo "Controlled profile reports: $output_dir"
