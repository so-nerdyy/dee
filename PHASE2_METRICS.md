# Phase 2 metrics

`Engine::phase2_metrics(completed_tokens)` returns `TierMetrics`; lower-level
users call `DeviceExpertTier::metrics(host, completed_tokens)`. Disabled Engine
returns an empty, invalid-denominator snapshot. The C++ API is intentional:
no implicit environment variable or Python/CLI default enables the experiment.

## Definitions

| Field | Meaning |
|---|---|
| `host.host_hit` | Successful reuse of a ready record, including a successful coalesced waiter. Device hits do not request the host tier. |
| `host.host_miss` | New fill reservations, including failed fills. Capacity/invalid requests do not reserve and do not count here. |
| `host.coalesced` | Requests that joined an in-progress fill. Included in host hits only on successful completion. |
| `device_hit` | In-scope, size-valid resident or in-flight device requests; no host lookup or H2D. |
| `device_miss` | Device lookups requiring a host record, including attempts later rejected for host/device capacity or I/O failure. |
| `host.SSD_bytes` | Logical storage bytes reported by codec reads. Identity/ExpertStore adapter reports successfully materialized record bytes. It is **not** physical disk-sector traffic. |
| `H2D_bytes` | Packed bytes submitted successfully to device; on CPU mock, modeled submitted copy bytes. Host/device hits add zero. |
| `host.fills` | Successful committed cold-to-host fills; a duplicate group fills once. |
| `host.evictions` | Idle ready host records withdrawn/replaced, including explicit static withdrawal. Failed reservations do not count. |
| `device_evictions` | Existing arena evictions caused by Phase-2 stages. Not counter-reset subtraction at snapshot time. |
| `host.failures` | Failed/throwing/short codec fills. |
| `device_failures` | Invalid descriptors/scope and failed staging attempts. |
| `host.budget_rejections` | Capacity responses; may include intermediate responses successfully retried after DMA completion. |
| `host.pin_failures` | Slot registrations that failed at pool initialization. |
| `host.allocated_bytes` | All aligned slot allocations, including empty/failed/idle slots and padding. |
| `host.pinned_bytes` | Successfully registered allocation bytes, including padding. |
| `host.resident_bytes` | Logical bytes in ready entries, not allocated capacity. |
| `host.peak_resident_bytes` | High-water mark of logical ready-entry bytes for this tier lifetime. |
| `host.leased_slots` | Slots with at least one active reference; includes filling slots. |
| `device_bytes`, `device_budget` | Current arena use and its fixed allocation capacity. |
| `device_peak_bytes` | High-water mark of device arena use observed by Phase 2. |
| `tokens` | Explicit denominator supplied by caller for the same metric lifetime. |
| `bytes_per_token_valid` | False when denominator is zero; zero-valued ratio fields then mean unavailable. |
| `SSD_bytes_per_token` | Logical storage bytes / supplied tokens. |
| `H2D_bytes_per_token` | Submitted packed transfer bytes / supplied tokens. |
| `bytes_per_token` | `(SSD_bytes + H2D_bytes) / tokens`; counts traffic at two boundaries. |

For the accepted store's Boolean materialize interface, partial failed preads
cannot be attributed exactly per concurrent call. That adapter reports zero
successful record bytes for a failed read; lower-level pread telemetry remains
the source for partial read-service totals. A codec/store with exact partial
byte accounting can return it in `ColdReadResult`. OS page-cache hits are still
logical source reads. Never label this counter as measured SSD utilization.

## Wait time and provenance

| Field | Measurement boundary |
|---|---|
| `host.host_wait_ms` | Calling-thread condition-variable wait for a duplicate fill. |
| `host.storage_service_ms` | Sum of codec materialization wall times across filler threads; includes store bookkeeping and codec work. This is service sum, **not critical-path wall time**. |
| `host_capacity_wait_ms` | Calling-thread reclaim/wait boundary when final host slots are unavailable; may include a targeted DMA event wait. |
| `device_enqueue_ms` | Caller time issuing the owned packed transfer, inclusive of pageable fallback waiting. |
| `device_host_wait_ms` | Experimental per-transfer synchronous readiness waits, including calls made by existing Engine wait paths. CPU mock facade waits are measured separately. |
| `pageable_fallback_wait_ms` | Event wait after a pageable H2D submission. CUDA's pageable staging during submission is included in enqueue time instead. |
| `device_gpu_wait_ms` | Optional/unmeasured in this skeleton. Empty means UNKNOWN, not zero. |

These counters have a lifetime of the tier/transfer facade. Existing
`reset_external_profile` does not reset Phase-2 host residency or its counters.
Pass a matching lifetime token count or subtract two snapshots and use the
corresponding token delta. Do not sum nested waits, concurrent service times,
and enqueue intervals into a critical-path total.

The existing route profiler and authoritative request tracing remain. Direct
CUDA submission records the same H2D GPU stage, request, and transfer-submit
timeline categories and leaves `mmap_to_pinned_bytes` at zero for that path.
Phase-2 host filling occurs inside stage, so existing `StageEnqueueWait` includes
host acquisition, while the legacy prepare `FillWait` does no Phase-2 fill work.
Use the tier service/wait counters for this experiment; do not reinterpret
Phase-1 bucket names as a new additive decomposition or overwrite its evidence.

## Validation scope

The intended synthetic tests check single-fill accounting, byte identity, hit
traffic of zero, bounded capacities, evictions, and explicit token
denominators. The optional real-CUDA test checks zero application gather-copy
bytes and exact H2D bytes. The current local MinGW run passes the Phase-2
mechanism and affected C++ tests. CUDA remains uncompiled/unexecuted and there
is no model-parity run; a mock result is never GPU performance evidence.
