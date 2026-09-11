#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace dee {

// Model identity includes checkpoint revision/integrity, not just model family.
// Representation identifies the exact packed layout AND its version.
struct TierExpertKey {
    static constexpr size_t max_identity_bytes = 1024;
    std::string model;
    int layer = -1;
    int expert = -1;
    std::string representation;
    bool valid() const;
    bool operator==(const TierExpertKey& other) const;
};

struct ColdReadResult {
    bool success = false;
    size_t bytes_read = 0; // successfully materialized logical storage bytes
};

class ColdExpertStore {
public:
    virtual ~ColdExpertStore() = default;
    virtual ColdReadResult read(const TierExpertKey&, uint8_t* destination,
                                size_t bytes) = 0;
};

struct StorageRecord {
    TierExpertKey key;
    size_t stored_bytes = 0;
    size_t exact_bytes = 0;
    std::string codec = "identity-v1";
};

// Decode must write the exact executor representation into destination.
// Future codecs own bounded scratch explicitly; no implicit host expansion.
class StorageCodec {
public:
    virtual ~StorageCodec() = default;
    virtual const char* name() const = 0;
    virtual bool accepts(const StorageRecord&) const = 0;
    virtual ColdReadResult materialize(ColdExpertStore&, const StorageRecord&,
                                      uint8_t* destination, size_t capacity) const = 0;
};

class IdentityCodec final : public StorageCodec {
public:
    const char* name() const override { return "identity-v1"; }
    bool accepts(const StorageRecord& record) const override {
        return record.key.valid() && record.codec == name() && record.exact_bytes != 0 &&
            record.exact_bytes == record.stored_bytes;
    }
    ColdReadResult materialize(ColdExpertStore&, const StorageRecord&,
                              uint8_t*, size_t) const override;
};

enum class HostResidency { PolicyResident, Dynamic };
struct HostVictim {
    size_t slot = 0;
    const TierExpertKey* key = nullptr; // borrowed only during policy callback
    uint64_t generation = 0;
    uint64_t last_use = 0;
    size_t bytes = 0;
};

// Callbacks run under the tier mutex; must not reenter the tier.
// Only idle dynamic entries are offered as victims. A policy cannot override
// a lease or evict a policy-resident slot. No production policy is chosen here.
class HostPlacementPolicy {
public:
    virtual ~HostPlacementPolicy() = default;
    virtual HostResidency residency(const TierExpertKey&) const = 0;
    virtual std::optional<size_t> victim(const TierExpertKey&,
                                        const std::vector<HostVictim>&) const = 0;
};

struct HostTierConfig {
    size_t slot_bytes = 0;
    size_t alignment = 4096;
    size_t policy_slots = 0;
    size_t dynamic_slots = 0;
    size_t budget_bytes = 0; // includes slot padding; no implicit budget
    bool try_pin = true;
};

// Injection supports deterministic allocation/pinning-failure tests. Pin false
// must mean no registration was retained. Unpin/free must not throw.
struct HostMemoryBackend {
    std::function<void*(size_t, size_t)> allocate;
    std::function<void(void*)> free;
    std::function<bool(void*, size_t)> pin;
    std::function<void(void*)> unpin;
};
HostMemoryBackend host_memory_backend(bool cuda, int device = 0);

struct HostTierStats {
    uint64_t host_hit = 0, host_miss = 0, coalesced = 0;
    uint64_t SSD_bytes = 0, fills = 0, evictions = 0, failures = 0;
    uint64_t budget_rejections = 0, pin_failures = 0;
    size_t allocated_bytes = 0, pinned_bytes = 0, resident_bytes = 0;
    size_t peak_resident_bytes = 0;
    size_t leased_slots = 0;
    double host_wait_ms = 0, storage_service_ms = 0;
};

struct HostTierState;
// Copyable RAII lease; copying adds a reference to an existing slot, not a new
// payload allocation. Even destruction of HostExpertTier cannot free a lease.
class HostExpertLease {
public:
    HostExpertLease() = default;
    HostExpertLease(const HostExpertLease&);
    HostExpertLease& operator=(const HostExpertLease&);
    HostExpertLease(HostExpertLease&&) noexcept;
    HostExpertLease& operator=(HostExpertLease&&) noexcept;
    ~HostExpertLease();
    explicit operator bool() const { return state_ != nullptr; }
    const uint8_t* data() const;
    size_t size() const;
    bool pinned() const;
    const TierExpertKey& key() const;
    uint64_t generation() const { return generation_; }
    void reset();
private:
    friend class HostExpertTier;
    HostExpertLease(std::shared_ptr<HostTierState>, size_t, uint64_t);
    std::shared_ptr<HostTierState> state_;
    size_t slot_ = 0;
    uint64_t generation_ = 0;
};

enum class HostAcquireStatus { Ready, Capacity, Invalid, FillFailed };
struct HostAcquireResult {
    HostAcquireStatus status = HostAcquireStatus::Invalid;
    HostExpertLease lease;
};

class HostExpertTier {
public:
    HostExpertTier(const HostTierConfig&, HostMemoryBackend,
                   std::shared_ptr<const HostPlacementPolicy>);
    HostExpertTier(const HostExpertTier&) = delete;
    HostExpertTier& operator=(const HostExpertTier&) = delete;
    HostAcquireResult acquire(const StorageRecord&, ColdExpertStore&,
                              const StorageCodec&);
    // Explicit policy withdrawal, including static residency; never revokes
    // a filling slot or outstanding lease. Does not free the reusable payload.
    bool evict(const TierExpertKey&);
    HostTierStats stats() const;
private:
    std::shared_ptr<HostTierState> state_;
};
} // namespace dee
