# dee.cpp genuine Ornith execution status

**Verified**: 2026-07-22 on a Lightning Tesla T4 devbox

**Branch**: `opt/real-model-t1`

**Result**: PASS — genuine Ornith-1.0-35B layer-0 execution and dee.cpp parity

## What is proven

`scripts/run_ornith_layer0_parity.py` executes this real path for four prompts:

```text
real tokenizer
  -> real embedding lookup
  -> complete real layer-0 Gated DeltaNet token mixer
  -> real layer-0 router logits and ordered top-8 selection
  -> real Ornith routed expert tensors through dee.cpp
  -> real Transformers routing-weight combine and shared expert
  -> layer-0 residual hidden state
```

The original Transformers `Qwen3_5MoeDecoderLayer` owns normalization, the
Gated DeltaNet, residuals, routing, shared-expert execution, and combination.
Only its routed-expert module is swapped between an FP32 PyTorch reference and
`Engine::moe_forward_experts`. Both paths consume the same real BF16 expert
tensors and identical hidden states.

The committed report is
`benchmark_reports/ornith-layer0-parity-20260722T185639Z.json`.

## Shard map and download decision

The local `model.safetensors.index.json` was inspected rather than assuming a
full checkpoint was necessary.

| Shard | Bytes | Cumulative | Supplies | Decision |
|---|---:|---:|---|---|
| `model-00001-of-00016.safetensors` | 4,324,097,448 | 4,324,097,448 | embedding, all 784 layer-0 tensors, router, 256 routed experts, shared expert, layer-0 `linear_attn.out_proj`, LM head | already present; used |
| `model-00016-of-00016.safetensors` | 2,565,780,424 | 6,889,877,872 | final model RMSNorm after all 40 blocks | not required for layer-0 milestone; not downloaded |

Additional checkpoint bytes downloaded: **0**.

## Parity results

All prompts had exact router logits, exact routing weights, exact expert order,
and exact top-k membership.

| Prompt | Expert max abs | Combined MoE max abs | Hidden max abs | Hidden relative L2 |
|---|---:|---:|---:|---:|
| `Capital of France is` | 1.1920929e-7 | 1.7229468e-8 | 5.9604645e-8 | 1.6749104e-7 |
| `7 * 6 =` | 1.3411045e-7 | 3.3527613e-8 | 5.9604645e-8 | 2.3424895e-7 |
| `def fibonacci(n):` | 1.7881393e-7 | 3.7252903e-8 | 2.9802322e-8 | 1.8678370e-7 |
| `Once upon a time` | 8.3446503e-7 | 3.2782555e-7 | 3.2782555e-7 | 4.8974102e-7 |

The harness records the tensor name, layer, expert, maximum absolute error,
maximum relative error, and coordinate for the first failure. No divergence
crossed the gate in this run.

## Runtime measurements

- End-to-end proof: 7.875 s
- Selective model load: 2.095 s
- dee.cpp engine initialization: 0.014 s
- Lazy reference expert loading: 1.375 s
- dee.cpp expert forward total: 3.606 s
- Unique experts touched across all prompts: 100 of 256
- Process RSS: 626,737,152 bytes at start; 4,705,681,408 bytes observed peak
- Tesla T4 VRAM: 3 MiB before and after (CPU FP32 correctness backend; no GPU allocation)

## Build and verification commands

```bash
python3 -m pip install --user --break-system-packages --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
python3 -m pip install --user --break-system-packages --no-cache-dir \
  numpy==1.26.4 scipy==1.11.4 scikit-learn==1.3.2 \
  transformers==5.14.1 accelerate safetensors huggingface_hub psutil
sudo apt-get install -y python3.12-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEE_CUDA=OFF
cmake --build build --parallel 4
python3 pydee/setup.py build_ext --inplace --force
ctest --test-dir build --output-on-failure
python3 scripts/run_ornith_layer0_parity.py --max-prompt-tokens 16
```

Native result: 6/6 CTest tests passed. The Transformers optimized linear
attention kernels were unavailable, so the documented PyTorch correctness
implementation was used.

## Scope boundary

This proves the requested first complete transformer-block milestone, not full
40-layer generation. Shard 16 and the other layer shards remain intentionally
absent. The dee.cpp routed-expert proof uses its CPU FP32 path; GPU expert
execution and full-generation validation are separate future milestones.
