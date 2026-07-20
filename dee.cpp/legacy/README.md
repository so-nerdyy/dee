# Removed prototype generation

The pre-CMake prototype used `CudaDevice`, `WeightSource`, a second
`VramCacheManager`, and a second `OracleScheduler`.  Those declarations shared
names with the production Engine pipeline but exposed incompatible APIs.

They were removed in the Lightning/T4 build cleanup instead of being hidden by
source globs.  The only supported public pipeline is:

`WeightMmap -> TensorResolver -> OracleScheduler (oracle.h) -> VramCacheManager
(vram_cache.h) -> AsyncPrefetcher -> Engine/SwiGLU`.

`Engine::swiglu` and `src/swiglu_cuda.cu` are the canonical CPU and CUDA MoE
forward implementations respectively.

Old standalone build scripts and tests exercised the removed API and were also
removed.  Use CMake and `scripts/setup_lightning_t4.sh` for supported builds.
