// kt_bridge/reference_cpu_executor.hpp
//
// Exact-FP32 portable executor. Correctness arbiter + AVX2-equivalent baseline.
#pragma once

#include "kt_bridge/cpu_executor.hpp"

namespace dee {
namespace ktbridge {

// ISA-neutral, single-threaded, full-FP32. No BF16 anywhere.
class ReferenceCpuExecutor : public CpuExpertExecutor {
 public:
    ExecuteError execute(int layer, int expert_id,
                         const PackedExpertView& packed, const float* hidden,
                         size_t hidden_dim, float routing_weight,
                         const ExecuteConfig& config, float* out,
                         size_t out_dim, ExecuteStats* stats = nullptr) override;
    const char* name() const override { return "ReferenceCpuExecutor"; }
};

}  // namespace ktbridge
}  // namespace dee
