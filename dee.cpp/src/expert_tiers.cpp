#include "dee/expert_tiers.h"

#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace dee {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
bool same_layout(const ExpertView& a, const ExpertView& b) {
    if (!a.ok() || a.codec != b.codec) return false;
    for (size_t i = 0; i < 3; ++i) {
        if (a.weights[i].shape != b.weights[i].shape || a.scales[i].shape != b.scales[i].shape ||
            a.weights[i].dtype != b.weights[i].dtype || a.scales[i].dtype != b.scales[i].dtype ||
            a.weights[i].nbytes != b.weights[i].nbytes || a.scales[i].nbytes != b.scales[i].nbytes)
            return false;
    }
    return true;
}
}
ExpertStoreColdAdapter::ExpertStoreColdAdapter(ExpertStore& store,
        std::string model, std::string representation, const ExpertView& layout)
    : store_(store), store_identity_(store.integrity_identity()),
      scope_{std::move(model), 0, 0, std::move(representation)}, layout_(layout) {
    if (!scope_.valid() || !layout.ok()) throw std::invalid_argument("invalid cold store scope/layout");
    scope_.model += "\nstore:" + store_identity_;
    if (!scope_.valid()) throw std::invalid_argument("cold store identity exceeds bounded key storage");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t n : {layout.weights[i].nbytes, layout.scales[i].nbytes}) {
            if (n > std::numeric_limits<size_t>::max() - bytes_)
                throw std::invalid_argument("expert size overflow");
            bytes_ += n;
        }
    }
}
StorageRecord ExpertStoreColdAdapter::record(int layer, int expert) const {
    auto key = scope_; key.layer = layer; key.expert = expert;
    return {std::move(key), bytes_, bytes_, "identity-v1"};
}
ColdReadResult ExpertStoreColdAdapter::read(const TierExpertKey& key, uint8_t* dst, size_t n) {
    if (!key.valid() || key.model != scope_.model || key.representation != scope_.representation ||
        !dst || n != bytes_) return {};
    ExpertView view;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (store_.integrity_identity() != store_identity_ || !store_.get(key.layer, key.expert, &view) ||
            !same_layout(view, layout_)) return {};
    }
    const auto begin = Clock::now();
    const bool contiguous = view.contiguous_data != nullptr;
    if (contiguous) {
        size_t offset = 0;
        for (const auto& tensor : view.weights) {
            if (offset > n || tensor.nbytes > n - offset ||
                tensor.data != view.contiguous_data + offset) return {};
            offset += tensor.nbytes;
        }
        for (const auto& tensor : view.scales) {
            if (offset > n || tensor.nbytes > n - offset ||
                tensor.data != view.contiguous_data + offset) return {};
            offset += tensor.nbytes;
        }
        if (view.contiguous_nbytes != n || !store_.materialize(view, dst, n)) return {};
    } else {
        // Canonical safetensors: gather six original packed regions directly
        // into the FINAL host slot. This is the first/only host record copy.
        size_t offset = 0;
        for (const auto& tensor : view.weights) {
            std::memcpy(dst + offset, tensor.data, tensor.nbytes); offset += tensor.nbytes;
        }
        for (const auto& tensor : view.scales) {
            std::memcpy(dst + offset, tensor.data, tensor.nbytes); offset += tensor.nbytes;
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        store_.record_source_read(n, elapsed(begin), contiguous ? 1 : 6, contiguous);
    }
    return {true, n};
}

DeviceExpertTier::DeviceExpertTier(VramCacheManager& cache, AsyncPrefetcher& prefetcher,
        TierExpertKey scope, std::shared_ptr<const DevicePlacementPolicy> policy)
    : cache_(cache), prefetcher_(prefetcher), scope_(std::move(scope)), policy_(std::move(policy)) {
    if (!scope_.valid() || cache.resident_count() != 0 || prefetcher.in_flight() != 0 ||
        !prefetcher.enable_experimental_host_tier(scope_))
        throw std::invalid_argument("device tier requires an empty exclusive cache and valid scope");
    if (policy_) {
        cache_.set_experimental_eviction_score([scope = scope_, policy = policy_](const ExpertBlock& block) {
            auto key = scope; key.layer = block.key.layer; key.expert = block.key.expert;
            return policy->eviction_score(key, block.last_used, block.priority);
        });
    }
}
DeviceExpertTier::~DeviceExpertTier() { cache_.set_experimental_eviction_score({}); }
bool DeviceExpertTier::accepts(const TierExpertKey& key) const {
    return key.valid() && key.model == scope_.model && key.representation == scope_.representation;
}
bool DeviceExpertTier::stage(const StorageRecord& record, ColdExpertStore& store,
        HostExpertTier& host, const StorageCodec& codec, int route_priority,
        int token, int logical_layer) {
    if (!accepts(record.key) || record.exact_bytes == 0 ||
        record.exact_bytes > cache_.budget_bytes() || !codec.accepts(record)) {
        ++metrics_.device_failures; return false;
    }
    const auto& key = record.key;
    const int priority = policy_ ? policy_->priority(key, route_priority) : route_priority;
    if (cache_.is_resident(key.layer, key.expert)) {
        if (cache_.size_of(key.layer, key.expert) != record.exact_bytes) {
            ++metrics_.device_failures; return false;
        }
        ++metrics_.device_hit;
        // Existing prefetch hit path retains the DMA/compute reservation pin.
        const bool ok = prefetcher_.prefetch(key.layer, key.expert,
            cache_.data(key.layer, key.expert), record.exact_bytes,
            priority, token, logical_layer) >= 0;
        metrics_.device_peak_bytes = std::max(metrics_.device_peak_bytes,
                                              cache_.used_bytes());
        return ok;
    }
    ++metrics_.device_miss;
    prefetcher_.collect_host_sources();
    auto acquired = host.acquire(record, store, codec);
    while (acquired.status == HostAcquireStatus::Capacity) {
        const auto begin = Clock::now();
        const bool reclaimed = prefetcher_.collect_host_sources(true);
        metrics_.host_capacity_wait_ms += elapsed(begin);
        if (!reclaimed) break;
        acquired = host.acquire(record, store, codec);
    }
    if (!acquired.lease) { ++metrics_.device_failures; return false; }
    const auto begin = Clock::now();
    const auto evictions_before = cache_.stats().evictions;
    const bool ok = prefetcher_.prefetch_host_lease(acquired.lease, priority, token, logical_layer) >= 0;
    metrics_.device_peak_bytes = std::max(metrics_.device_peak_bytes,
                                          cache_.used_bytes());
    metrics_.device_evictions += cache_.stats().evictions - evictions_before;
    metrics_.device_enqueue_ms += elapsed(begin);
    if (ok) metrics_.H2D_bytes += record.exact_bytes;
    else ++metrics_.device_failures;
    return ok;
}
bool DeviceExpertTier::wait(const TierExpertKey& key) {
    if (!accepts(key)) return false;
    const auto begin = Clock::now();
    const bool ready = prefetcher_.wait(key.layer, key.expert);
    if (!prefetcher_.using_cuda()) metrics_.device_host_wait_ms += elapsed(begin);
    return ready;
}
TierMetrics DeviceExpertTier::metrics(const HostExpertTier& host, uint64_t tokens) const {
    auto result = metrics_;
    result.host = host.stats();
    if (prefetcher_.using_cuda()) result.device_host_wait_ms = prefetcher_.experimental_readiness_wait_ms();
    result.pageable_fallback_wait_ms = prefetcher_.experimental_pageable_wait_ms();
    result.device_bytes = cache_.used_bytes(); result.device_budget = cache_.budget_bytes();
    result.tokens = tokens;
    result.bytes_per_token_valid = tokens != 0;
    if (tokens) {
        result.SSD_bytes_per_token = static_cast<double>(result.host.SSD_bytes) / tokens;
        result.H2D_bytes_per_token = static_cast<double>(result.H2D_bytes) / tokens;
        result.bytes_per_token = result.SSD_bytes_per_token + result.H2D_bytes_per_token;
    }
    return result;
}
} // namespace dee
