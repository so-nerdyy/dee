#pragma once

#include "dee/async_prefetcher.h"
#include "dee/expert_store.h"
#include "dee/host_expert_tier.h"

#include <mutex>

namespace dee {

// Adapter over the accepted exact store. It owns no checkpoint bytes. Store
// and mapped tensors must outlive read(); host leases are independent of them.
class ExpertStoreColdAdapter final : public ColdExpertStore {
public:
    ExpertStoreColdAdapter(ExpertStore&, std::string model_identity,
                           std::string representation, const ExpertView& layout);
    StorageRecord record(int layer, int expert) const;
    ColdReadResult read(const TierExpertKey&, uint8_t*, size_t) override;
private:
    ExpertStore& store_;
    std::string store_identity_;
    TierExpertKey scope_;
    ExpertView layout_;
    size_t bytes_ = 0;
    std::mutex mutex_; // legacy lookup/read-stat mutation is caller-serialized
};

class DevicePlacementPolicy {
public:
    virtual ~DevicePlacementPolicy() = default;
    virtual int priority(const TierExpertKey&, int authoritative_route_priority) const = 0;
    virtual int64_t eviction_score(const TierExpertKey&, int64_t last_use,
                                   int priority) const = 0;
};

struct TierMetrics {
    HostTierStats host;
    uint64_t device_hit = 0, device_miss = 0, H2D_bytes = 0;
    uint64_t device_evictions = 0, device_failures = 0;
    size_t device_bytes = 0, device_peak_bytes = 0, device_budget = 0;
    double host_capacity_wait_ms = 0, device_enqueue_ms = 0;
    double device_host_wait_ms = 0;
    double pageable_fallback_wait_ms = 0;
    std::optional<double> device_gpu_wait_ms; // unmeasured, never encoded as zero
    uint64_t tokens = 0;
    bool bytes_per_token_valid = false;
    double SSD_bytes_per_token = 0, H2D_bytes_per_token = 0, bytes_per_token = 0;
};

// Facade over the existing arena and transfer stream, not a second GPU cache.
// Exclusive to one model/representation/device for its whole lifetime. Same
// single-caller ownership as Engine/AsyncPrefetcher; host tier is concurrent.
// Existing Engine compute pins remain the ExactExpertExecutor lifetime fence.
class DeviceExpertTier {
public:
    DeviceExpertTier(VramCacheManager&, AsyncPrefetcher&, TierExpertKey scope,
                     std::shared_ptr<const DevicePlacementPolicy> = {});
    ~DeviceExpertTier();
    DeviceExpertTier(const DeviceExpertTier&) = delete;
    DeviceExpertTier& operator=(const DeviceExpertTier&) = delete;
    bool stage(const StorageRecord&, ColdExpertStore&, HostExpertTier&,
               const StorageCodec&, int route_priority, int token = -1,
               int logical_layer = -1);
    bool wait(const TierExpertKey&); // synchronous test/explicit consumer boundary
    TierMetrics metrics(const HostExpertTier&, uint64_t completed_tokens = 0) const;
private:
    bool accepts(const TierExpertKey&) const;
    VramCacheManager& cache_;
    AsyncPrefetcher& prefetcher_;
    TierExpertKey scope_;
    std::shared_ptr<const DevicePlacementPolicy> policy_;
    TierMetrics metrics_;
};

struct Phase2TierConfig {
    bool enabled = false;
    std::string model_identity; // immutable checkpoint ID/revision supplied by caller
    HostTierConfig host;
    std::shared_ptr<const HostPlacementPolicy> host_policy;
    std::shared_ptr<const DevicePlacementPolicy> device_policy;
    // Only IdentityCodec is wired into Engine in this phase.
};
} // namespace dee
