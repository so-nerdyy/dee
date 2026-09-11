#include "dee/host_expert_tier.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#ifdef _WIN32
#include <malloc.h>
#endif
#ifdef DEE_CUDA
#include <cuda_runtime.h>
#endif

namespace dee {
bool TierExpertKey::valid() const {
    return !model.empty() && model.size() <= max_identity_bytes &&
           !representation.empty() && representation.size() <= max_identity_bytes &&
           layer >= 0 && expert >= 0;
}
bool TierExpertKey::operator==(const TierExpertKey& o) const {
    return model == o.model && layer == o.layer && expert == o.expert &&
           representation == o.representation;
}
std::optional<size_t> PlainLruHostPlacementPolicy::victim(
        const TierExpertKey&, const std::vector<HostVictim>& candidates) const {
    if (candidates.empty()) return std::nullopt;
    const auto it = std::min_element(candidates.begin(), candidates.end(),
        [](const HostVictim& a, const HostVictim& b) {
            if (a.last_use != b.last_use) return a.last_use < b.last_use;
            return a.slot < b.slot;
        });
    return it->slot;
}
ColdReadResult IdentityCodec::materialize(ColdExpertStore& store,
        const StorageRecord& record, uint8_t* dst, size_t capacity) const {
    if (!accepts(record) || !dst ||
        capacity < record.exact_bytes) return {};
    auto result = store.read(record.key, dst, record.exact_bytes);
    result.success = result.success && result.bytes_read == record.exact_bytes;
    return result;
}

HostMemoryBackend host_memory_backend(bool cuda, int device) {
    HostMemoryBackend backend;
    backend.allocate = [](size_t bytes, size_t alignment) -> void* {
#ifdef _WIN32
        return _aligned_malloc(bytes, alignment);
#else
        void* ptr = nullptr;
        return posix_memalign(&ptr, alignment, bytes) == 0 ? ptr : nullptr;
#endif
    };
    backend.free = [](void* ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    };
    backend.pin = [cuda, device](void* ptr, size_t bytes) {
#ifdef DEE_CUDA
        if (!cuda) return false;
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess ||
            cudaSetDevice(device) != cudaSuccess) return false;
        const auto status = cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
        if (previous != device) cudaSetDevice(previous);
        return status == cudaSuccess;
#else
        (void)cuda; (void)device; (void)ptr; (void)bytes;
        return false;
#endif
    };
    backend.unpin = [device](void* ptr) {
#ifdef DEE_CUDA
        int previous = -1;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        // Failed unregistration cannot safely be followed by free.
        if (cudaHostUnregister(ptr) != cudaSuccess) std::terminate();
        if (previous >= 0 && previous != device) cudaSetDevice(previous);
#else
        (void)device; (void)ptr;
#endif
    };
    return backend;
}

struct HostTierState {
    enum class Phase { Empty, Filling, Ready };
    struct Slot {
        uint8_t* data = nullptr;
        TierExpertKey key;
        size_t bytes = 0, references = 0;
        uint64_t generation = 0, last_use = 0;
        bool pinned = false;
        HostResidency residency = HostResidency::Dynamic;
        Phase phase = Phase::Empty;
    };
    std::mutex mutex;
    std::condition_variable cv;
    HostTierConfig config;
    HostMemoryBackend backend;
    std::shared_ptr<const HostPlacementPolicy> policy;
    std::vector<Slot> slots;
    std::vector<HostVictim> candidates;
    HostTierStats stats;
    uint64_t tick = 0;
    size_t stride = 0;
    ~HostTierState() {
        for (auto& slot : slots) {
            if (slot.pinned) backend.unpin(slot.data);
            if (slot.data) backend.free(slot.data);
        }
    }
};

HostExpertLease::HostExpertLease(std::shared_ptr<HostTierState> state,
        size_t slot, uint64_t generation)
    : state_(std::move(state)), slot_(slot), generation_(generation) {}
HostExpertLease::HostExpertLease(const HostExpertLease& other)
    : state_(other.state_), slot_(other.slot_), generation_(other.generation_) {
    if (state_) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        ++state_->slots[slot_].references;
    }
}
HostExpertLease& HostExpertLease::operator=(const HostExpertLease& other) {
    if (this != &other) { HostExpertLease copy(other); *this = std::move(copy); }
    return *this;
}
HostExpertLease::HostExpertLease(HostExpertLease&& other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_), generation_(other.generation_) {}
HostExpertLease& HostExpertLease::operator=(HostExpertLease&& other) noexcept {
    if (this != &other) {
        reset(); state_ = std::move(other.state_);
        slot_ = other.slot_; generation_ = other.generation_;
    }
    return *this;
}
HostExpertLease::~HostExpertLease() { reset(); }
void HostExpertLease::reset() {
    if (!state_) return;
    auto state = std::move(state_);
    std::lock_guard<std::mutex> lock(state->mutex);
    --state->slots[slot_].references;
}
const uint8_t* HostExpertLease::data() const { return state_ ? state_->slots[slot_].data : nullptr; }
size_t HostExpertLease::size() const { return state_ ? state_->slots[slot_].bytes : 0; }
bool HostExpertLease::pinned() const { return state_ && state_->slots[slot_].pinned; }
const TierExpertKey& HostExpertLease::key() const {
    if (!state_) throw std::logic_error("empty host expert lease");
    return state_->slots[slot_].key;
}

HostExpertTier::HostExpertTier(const HostTierConfig& config,
        HostMemoryBackend backend, std::shared_ptr<const HostPlacementPolicy> policy) {
    const size_t limit = std::numeric_limits<size_t>::max();
    if (!policy || !backend.allocate || !backend.free || !backend.pin || !backend.unpin ||
        config.slot_bytes == 0 || config.alignment < sizeof(void*) ||
        (config.alignment & (config.alignment - 1)) != 0 ||
        config.slot_bytes > limit - (config.alignment - 1) ||
        config.policy_slots > limit - config.dynamic_slots)
        throw std::invalid_argument("invalid host tier configuration");
    const size_t count = config.policy_slots + config.dynamic_slots;
    const size_t stride = (config.slot_bytes + config.alignment - 1) & ~(config.alignment - 1);
    if (count == 0 || count > config.budget_bytes / stride)
        throw std::invalid_argument("host slots exceed explicit budget");
    auto state = std::make_shared<HostTierState>();
    state->config = config; state->backend = std::move(backend);
    state->policy = std::move(policy); state->stride = stride;
    state->slots.resize(count); state->candidates.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto& slot = state->slots[i];
        slot.key.model.reserve(TierExpertKey::max_identity_bytes);
        slot.key.representation.reserve(TierExpertKey::max_identity_bytes);
        slot.residency = i < config.policy_slots ? HostResidency::PolicyResident : HostResidency::Dynamic;
        slot.data = static_cast<uint8_t*>(state->backend.allocate(stride, config.alignment));
        if (!slot.data) throw std::bad_alloc();
        state->stats.allocated_bytes += stride;
        slot.pinned = config.try_pin && state->backend.pin(slot.data, stride);
        if (slot.pinned) state->stats.pinned_bytes += stride;
        else if (config.try_pin) ++state->stats.pin_failures;
    }
    state_ = std::move(state);
}

HostAcquireResult HostExpertTier::acquire(const StorageRecord& record,
        ColdExpertStore& store, const StorageCodec& codec) {
    auto state = state_;
    if (!record.key.valid() || record.exact_bytes == 0 ||
        record.exact_bytes > state->config.slot_bytes || !codec.accepts(record)) return {};
    std::unique_lock<std::mutex> lock(state->mutex);
    for (size_t i = 0; i < state->slots.size(); ++i) {
        auto& slot = state->slots[i];
        if (slot.phase == HostTierState::Phase::Empty || !(slot.key == record.key)) continue;
        if (slot.bytes != record.exact_bytes) return {};
        ++slot.references;
        if (slot.phase == HostTierState::Phase::Filling) {
            ++state->stats.coalesced;
            const auto begin = std::chrono::steady_clock::now();
            state->cv.wait(lock, [&] { return slot.phase != HostTierState::Phase::Filling; });
            state->stats.host_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        }
        if (slot.phase != HostTierState::Phase::Ready) {
            --slot.references;
            return {HostAcquireStatus::FillFailed, {}};
        }
        ++state->stats.host_hit;
        slot.last_use = ++state->tick;
        return {HostAcquireStatus::Ready, HostExpertLease(state, i, slot.generation)};
    }
    const auto residency = state->policy->residency(record.key);
    size_t selected = state->slots.size();
    state->candidates.clear();
    for (size_t i = 0; i < state->slots.size(); ++i) {
        auto& slot = state->slots[i];
        if (slot.residency != residency || slot.references != 0) continue;
        if (slot.phase == HostTierState::Phase::Empty) { selected = i; break; }
        if (residency == HostResidency::Dynamic && slot.phase == HostTierState::Phase::Ready)
            state->candidates.push_back({i, &slot.key, slot.generation, slot.last_use, slot.bytes});
    }
    if (selected == state->slots.size() && !state->candidates.empty()) {
        const auto victim = state->policy->victim(record.key, state->candidates);
        if (victim && std::any_of(state->candidates.begin(), state->candidates.end(),
                [&](const HostVictim& v) { return v.slot == *victim; })) selected = *victim;
    }
    if (selected == state->slots.size()) {
        ++state->stats.budget_rejections;
        return {HostAcquireStatus::Capacity, {}};
    }
    auto& slot = state->slots[selected];
    if (slot.phase == HostTierState::Phase::Ready) {
        ++state->stats.evictions; state->stats.resident_bytes -= slot.bytes;
    }
    slot.phase = HostTierState::Phase::Empty;
    slot.key = record.key;
    slot.bytes = record.exact_bytes; ++slot.generation;
    slot.last_use = ++state->tick; slot.references = 1;
    slot.phase = HostTierState::Phase::Filling;
    ++state->stats.host_miss;
    lock.unlock();
    const auto begin = std::chrono::steady_clock::now();
    ColdReadResult result;
    try { result = codec.materialize(store, record, slot.data, state->config.slot_bytes); }
    catch (...) { result = {}; }
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    lock.lock();
    state->stats.SSD_bytes += result.bytes_read;
    state->stats.storage_service_ms += elapsed;
    if (!result.success) {
        slot.phase = HostTierState::Phase::Empty; --slot.references;
        ++state->stats.failures;
        lock.unlock(); state->cv.notify_all();
        return {HostAcquireStatus::FillFailed, {}};
    }
    slot.phase = HostTierState::Phase::Ready;
    ++state->stats.fills;
    state->stats.resident_bytes += slot.bytes;
    state->stats.peak_resident_bytes = std::max(
        state->stats.peak_resident_bytes, state->stats.resident_bytes);
    const auto generation = slot.generation;
    lock.unlock(); state->cv.notify_all();
    return {HostAcquireStatus::Ready, HostExpertLease(state, selected, generation)};
}

bool HostExpertTier::evict(const TierExpertKey& key) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (auto& slot : state_->slots) {
        if (slot.phase == HostTierState::Phase::Ready && slot.key == key && slot.references == 0) {
            slot.phase = HostTierState::Phase::Empty;
            state_->stats.resident_bytes -= slot.bytes; ++state_->stats.evictions;
            return true;
        }
    }
    return false;
}
HostTierStats HostExpertTier::stats() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto result = state_->stats;
    for (const auto& slot : state_->slots) result.leased_slots += slot.references != 0;
    return result;
}
} // namespace dee
