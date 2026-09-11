# R6 — Heterogeneous execution sink audit (MoE-Lightning shape)

- Track: R6 — heterogeneous execution scheduler / pipelining interface audit
- Branch: `research/prior-art-r06` @ `dc78dc4` (worktree `.freebuff/wt/r06`)
- Audit target: `dee.cpp` tier seam — `ColdExpertStore -> HostExpertTier ->
  DeviceExpertTier -> ExactExpertExecutor` — read against the question: *do the
  tier interfaces already admit a SECOND EXECUTION SINK (CPU expert execution
  on device-cache miss) without redesign?* MoE-Lightning reference shape
  (arXiv 2411.11217, ASPLOS'25): CGOPipe overlaps CPU expert compute with GPU
  compute + I/O; an HRM cost model picks per-expert CPU-vs-GPU placement under
  a memory budget. dee mapping: device hit -> GPU (existing); device miss ->
  CPU-execute from host-tier bytes instead of (or ahead of) H2D.
- Rules observed: no code changed; this is an interface-gap list.

## Verdict

**Host tier: YES, today.** `HostExpertTier::acquire()` +
`HostExpertLease` already provide exactly the primitive a CPU sink needs —
a refcounted, eviction-proof, generation-stamped, `Ready`-gated view of the
exact packed record, safe to hold on a worker thread, safe to read while a
prefetch DMA reads the same bytes. No tier-ownership redesign needed.

**Device tier / engine: NO without a seam.** `DeviceExpertTier::stage()`
returns `bool` and *always* terminates a device miss in
`prefetch_host_lease` (`src/expert_tiers.cpp:148-166`) — the caller cannot
learn hit-vs-miss, cannot receive the lease, and cannot divert the expert.
All five engine consume loops then require the expert to appear in
`AsyncPrefetcher::inflight_`; a CPU-routed expert fails both readiness calls
and hard-errors the layer. 9 named gaps below; 5 blocking, 4 additive.

## What exists (cited)

### Host tier — lease lifetimes, generation safety, scope
- `include/dee/host_expert_tier.h:128-155` — `HostExpertLease`: copyable RAII;
  copy adds a slot reference, not a payload copy; "Even destruction of
  HostExpertTier cannot free a lease" (:129). `data()/size()/pinned()` at
  :139-141; `key()` is a borrowed ref valid only while held (:142-146);
  `generation()` at :147.
- `src/host_expert_tier.cpp:93-120` — `HostTierState`: `Phase {Empty,
  Filling, Ready}` (:94); per-slot `references`, `generation`, `key`, `bytes`.
- Acquire semantics (`src/host_expert_tier.cpp:192-275`): resident-key hit
  bumps `references` (:206); a `Filling` slot coalesces waiters on a CV
  (:207-217); lease is minted only when `phase == Ready` (:214-220, :267-274).
  Refill/eviction candidates are restricted to `references == 0` slots
  (:227, and `evict()` :280). A live lease therefore *freezes* key, bytes,
  and generation — `generation_` can never advance under a held lease.
- Generation is **diagnostic, not load-bearing**: `data()/size()/key()` never
  re-check it; only tests read it (`tests/test_phase2_host_tier.cpp:81-93`).
  Safety is the refcount, not the generation. A non-DMA consumer inherits
  full protection for free.
- Scope: `HostExpertTier` itself is scope-agnostic (a keyed cache). Scoping
  lives in `ExpertStoreColdAdapter` (`src/expert_tiers.cpp:26-85`): `record()`
  mints scoped keys (:41-44); `read()` rejects out-of-scope keys and re-verifies
  store integrity (:46-53).
- Concurrency: "host tier is concurrent" (`include/dee/expert_tiers.h:56`);
  `acquire()` unlocks the tier mutex around the cold `codec.materialize` fill
  (`src/host_expert_tier.cpp:251-254`), so worker-thread lease holders and an
  engine-thread acquirer coexist by design.

### Device tier — staging, arming, scope checks
- `include/dee/expert_tiers.h:50-81` — `DeviceExpertTier`: facade over
  `VramCacheManager` + `AsyncPrefetcher`; single-model/scope/device; one-shot
  arming token; `scope_epoch_` re-check on every `stage()/wait()`.
- `src/expert_tiers.cpp:118-167` — `stage()`: stale-scope/fail-closed gate
  (:126-130); device-hit path retains DMA/compute pin via `prefetch()` on
  `cache_.data()` (:133-145); miss path: `collect_host_sources()` ->
  `host.acquire()` -> Capacity loop waiting on DMA-lease reclaim (:147-155) ->
  `prefetch_host_lease(acquired.lease, ...)` (:159). **The lease is consumed
  into the transfer and never escapes to the caller** — the single-sink
  hard-wire point.
- `wait()` (:168-174) delegates to `prefetcher_.wait` — device readiness only.

### AsyncPrefetcher — stream/event ownership, in-flight tracking
- `include/dee/async_prefetcher.h:30-69` — `Transfer`: `host_lease` "retained
  until DMA completion" (:31), `managed_source`, `dma_complete`, `done`,
  `event`, `cache_pin_held`, `staging_slot`.
- `prefetch_host_lease` (`src/async_prefetcher.cpp:700-709`): scope re-check
  (:702-704); passes `lease.pinned()` as `source_pinned`; `prefetch_impl`
  copies the lease into the transfer — refcount++ (:498-501). The stage()
  caller's lease copy dies at scope exit; the transfer's copy survives until
  `dma_complete` (`release_staging` :70-88; `collect_host_sources` :711-738).
- Readiness boundary is wait-based only: `wait()` (:538-559),
  `wait_on_stream()` (:561-617, arms `cudaStreamWaitEvent` on the compute
  stream, drops the transfer's cache pin :606-611). **No public "ready"
  query** — `is_resident` is allocation, `transfer.done`/`dma_complete` is
  readiness; only `wait*` resolves it.
- Pageable fallback: `cuda_submit_host` synchronizes this copy's event then
  releases the transfer's lease (:779-789) — other lease copies unaffected.
- `synchronize_all` fail-stop: unprovable drain with a managed lease in
  flight -> `std::terminate()` (:636-639). CPU-held leases are *not* managed
  sources — invisible here, which is safe but see Gap 9.
- `accounting_valid()` (:260-262): `requests == resident + inflight + cold`.
  A CPU-routed expert never calls `prefetch_*`, so the identity holds — but
  `RequestKind` has no sink alternative (profiling.h:48).

### Engine — staging loop and the per-layer sync
- `stage_expert` is THE central seam (`src/engine.cpp:3144-3250`): all five
  call sites route through it; Phase-2 dispatch at :3153-3158 delegates to
  `phase2_device_->stage(...)`.
- Stage sites: `moe_forward_experts` :298-299; `moe_forward_batch` :574-576;
  `moe_forward_batch_device_impl` :895-898; pointer-batched :1704-1711;
  `forward_layer` (generate loop) :4120-4121; mock `forward_layer` :3369.
- Consume pattern is uniform: `wait_on_stream || wait` -> `cache_.pin` ->
  compute -> `unpin` (:313-315, :355-357, :966-968, :1753-1760, :4128-4132).
  `wait_on_stream`/`wait` return false when the key is not in `inflight_`
  (`src/async_prefetcher.cpp:541-544, :563-564`) — a CPU-routed expert has no
  inflight entry → `return false`, layer aborts. This is the engine-side
  coupling a sink decision must unwind.
- The briefing's "~engine.cpp:748-754" is `moe_forward_batch_device` — a thin
  wrapper (:748-754). The actual per-layer `cudaStreamSynchronize` inside its
  impl is at :1066-1068 (gated by `synchronize_output`); sibling syncs:
  per-batch :417-418, output :426-427, per-expert-output :679-684, batch
  :4169, layer output :4198. **These are the natural join points for a CPU
  sink**: join pending CPU tasks, enqueue the small H2D of result rows on
  `compute_stream_`, and the existing sync covers it — ordering is trivially
  satisfied because the weighted combine happens downstream of the layer
  return (Python-side).
- `prepare_fp4_experts` early-returns under Phase-2 (:2794) — the serial
  cold fill already moved inside `host.acquire()` in `stage()`; a CPU sink
  inherits that serial fill (research/phase2-concurrent-fill
  SERIALIZATION_VERDICT applies unchanged: the sink dodges H2D and VRAM
  residency, not the 0.29-0.37 GiB/s storage ceiling).
- Phase-2 wiring: `Engine::init` :3680-3718 (cold adapter :3696, host tier
  :3709, device tier :3711); gate :3435-3440 requires packed-FP4 + CUDA +
  model_identity. Note the live Phase-2 config is Fp4E2m1-cache, which the
  combined-device API rejects (:1857 FP16-only) — the CPU sink's device-side
  entry point is `moe_forward_batch_device_impl`, plus the host-API paths
  `moe_forward_experts`/`moe_forward_batch` whose outputs are already host
  memory.
- `drain_pointer_batch_pending` (:1289-1322, called at :3149 inside
  `stage_expert`) is pin-retirement for the pointer-batch path — orthogonal
  to a CPU sink (no pins taken), no interaction.

## The four "what breaks" questions — answered

**Q1. Can a lease's bytes be read by a CPU executor while a DMA is in-flight?**
YES. Slot payload is immutable while `references > 0` (no refill, no evict:
`host_expert_tier.cpp:227,280`). DMA and CPU executor are both pure readers;
the transfer holds its own lease copy (`async_prefetcher.cpp:498-501`), so
the CPU copy is lifetime-independent. No write hazard exists anywhere —
`Phase::Ready` is certified *before* the lease is returned (:267-274) or
before a coalesced waiter wakes (:210-220).

**Q2. Is there a "ready" state distinct from "resident"?**
At host level, yes: `Phase{Empty,Filling,Ready}` — the lease *is* the ready
certificate. At device level: `resident` (block allocated) vs ready
(`transfer.done`/`dma_complete`/event) exist internally but there is **no
public ready query** — `wait()`/`wait_on_stream` are the only boundaries.
A CPU sink needs nothing new here: host `Ready` is exactly its precondition.

**Q3. Do generation checks survive a non-DMA consumer?**
Trivially — host `generation_` is never consulted on read (diagnostic-only;
only tests assert on it), and it *cannot* advance under a live lease because
refill requires `references == 0`. Device generation guards
transfer↔resident-block matching (`validate_invariants`
`async_prefetcher.cpp:140-144`) — irrelevant to a consumer that never takes a
device block. Scope checks: a CPU path acquiring via `host.acquire(record,
store, codec)` gets scope enforcement indirectly (record() mints scoped keys;
adapter `read()` re-checks them), but `HostExpertTier` itself is
scope-agnostic — the caller must carry the scope comparison. Minor, see Gap 8.

**Q4. What sync ordering would a CPU sink need?**
Only one new ordering primitive: **CPU completion**. (a) Input readiness is
the lease itself — no CUDA objects. (b) Output to host-API paths
(`moe_forward_experts`, `moe_forward_batch`) is a plain `memcpy` into
`experts_out` — zero ordering. (c) Output to the device-resident path must
land in `d_experts_out`: join the CPU task, then `cudaMemcpyAsync` the FP32
rows (tiny: group_tokens × hidden × 4 B) on `compute_stream_` before the
existing layer-end sync — stream ordering plus the downstream Python combine
make this sufficient; no new event machinery required. (d) Fail-closed: a
CPU executor failure must abort the layer, never skip the expert (exactness
contract; same shape as `!computed` at :650-653, :1001-1004).

## Gap list — smallest interface addition per missing capability

Blocking (the sink cannot function without):

- **G1 — `stage()` is single-sink and opaque.** Caller cannot learn
  hit-vs-miss nor receive the lease (`expert_tiers.cpp:118-167`,
  header `expert_tiers.h:65-67`). Smallest addition: a sibling method
  `DeviceExpertTier::stage_ex(record, store, host, codec, priority, token,
  layer, StageOutcome* out)` where
  `enum class ExecutionSink { DeviceResident, DeviceStaged, CpuHost }` and
  `struct StageOutcome { ExecutionSink sink; HostExpertLease lease; }` —
  on `CpuHost` the acquired lease is returned to the caller and no H2D is
  enqueued; `stage()` stays a wrapper that asserts non-CpuHost.
- **G2 — no production CPU executor interface.** The only packed-FP4 CPU
  executor is the isolated prototype `dee::ktbridge::CpuExpertExecutor`
  (`dee.cpp/experiments/kt_cpu_bridge/include/kt_bridge/cpu_executor.hpp:58-69`),
  standalone CMake, deliberately unwired. Smallest addition: promote a
  `dee::CpuExpertExecutor` abstract interface into `include/dee/` (the bridge
  contract — borrowed views, unweighted-capable output via routing_weight,
  fail-closed `ExecuteError` — already matches dee semantics).
- **G3 — no lease → region-view descriptor.** `HostExpertLease::data()`
  returns one flat blob (`host_expert_tier.h:139`); the CPU executor needs
  the six-region `PackedExpertView` (packed+scale per projection,
  `packed_expert_view.hpp:22-37`). The offsets exist only as Engine privates
  (`fp4_cache_packed_offsets_` etc., `engine.cpp:3686-3693`). Smallest
  addition: a codec-level descriptor, e.g.
  `StorageCodec::regions(const StorageRecord&) -> std::array<Region,6>` or a
  free `packed_expert_view(const HostExpertLease&, const ExpertView&)` —
  the IdentityCodec record layout ([gate_w][up_w][down_w][g_s][u_s][d_s],
  `expert_tiers.cpp:56-79`) already matches the bridge view order.
- **G4 — no async submit/join seam.** Overlap (the MoE-Lightning point)
  needs CPU tasks in flight while the host keeps staging GPU experts; join
  must precede the per-layer sync (:1067/:4198). Smallest addition:
  `CpuExpertExecutor::submit(...) -> Ticket` + `join(Ticket)` /
  `drain_layer()` (the bridge §6 already sketches `submit`/`sync_with_stream`
  unbound — promote it). Zero-overlap variant needs none of this: synchronous
  `execute()` on host-API paths works today once G1-G3 land.
- **G5 — consume loops hard-error on non-inflight experts** (five sites:
  `engine.cpp:313-314, 355-356, 966-967, 1753-1755, 4129-4130`). Not an
  interface addition so much as the mandatory integration point: the loops
  need the per-expert `ExecutionSink` tag from G1 to skip
  `wait_on_stream`/`pin` and dispatch to the sink. Smallest supporting
  addition: nothing beyond G1 — but flag `forward_layer`'s
  `stats_.fallbacks++` (:3370) so a CPU-routed expert is not counted as a
  fallback.

Additive (needed for policy quality / operability, not for first motion):

- **G6 — device-resident input invisible to CPU sink.**
  `moe_forward_batch_device_impl` takes `d_h_in` (device FP16,
  `engine.cpp:748-759`); a CPU executor needs host FP32 rows of the expert's
  token group. Smallest addition: an optional host-mirror parameter
  (`const float* h_in_host`) on the device batch API, or an engine helper
  `hidden_rows_to_host(...)` performing the small D2H during gather.
  Host-API paths already have `h_in` — no gap there.
- **G7 — no sink policy interface.** `DevicePlacementPolicy`
  (`expert_tiers.h:28-34`) covers priority/eviction only; MoE-Lightning needs
  a per-route CPU/GPU decision (HRM/q*). Smallest addition:
  `class ExecutionSinkPolicy { virtual ExecutionSink decide(
  const StorageRecord&, bool device_resident) const = 0; }` consulted inside
  `stage_ex`; the bridge's `cost_model.py::plan_split` enumeration is the
  offline input to wire later (top-k is tiny — full enumeration).
- **G8 — no non-blocking host-residency probe.** `acquire()` always waits
  on Filling or performs a synchronous cold fill (`host_expert_tier.cpp:
  207-217, 252-265`). A "CPU only if already in RAM" policy needs
  `HostExpertTier::probe(const TierExpertKey&) -> {Absent, Filling, Ready}`
  or `acquire(..., AcquireMode::NoFill)` returning a new
  `HostAcquireStatus::Absent`. Also: `HostExpertTier` carries no scope check;
  a probing/divert path should compare `key` against the device tier scope
  (accepts-equivalent) — name `HostExpertTier::scope`-agnostic note or expose
  `DeviceExpertTier::accepts` publicly.
- **G9 — metrics/forensics have no sink dimension.** `TierMetrics`
  (`expert_tiers.h:36-48`), `RequestKind` (`profiling.h:48`), `HostWaitReason`
  (:50-58), `HostSpan` (:179-190) cannot attribute miss→CPU vs miss→H2D;
  `device_miss` counts both while `H2D_bytes` silently excludes the diverted
  one. Smallest additions: `TierMetrics::cpu_sink_executions`,
  `cpu_sink_bytes` (H2D avoided), `RequestKind::CpuSink`,
  `HostWaitReason::CpuSinkJoin` / `HostSpan::CpuSinkWait`.
- **G10 — slot-capacity backpressure between DMA and CPU consumers.**
  `collect_host_sources` (`async_prefetcher.cpp:711-738`) only reclaims
  *transfer-owned* leases; CPU-held leases are invisible, and `stage()`'s
  Capacity loop (`expert_tiers.cpp:149-155`) can therefore wait forever /
  fail when all dynamic slots are CPU-leased. Smallest addition:
  `HostTierConfig::max_outstanding_leases` (or a `cpu_sink_lease_budget` on
  the sink policy) bounding concurrent non-DMA leases below `dynamic_slots`;
  `HostTierStats::leased_slots` (:123, computed at :291) already exposes the
  observable.

## Non-gaps (verified)

- Lease copyability/thread-safety: refcount under `state_->mutex`
  (`host_expert_tier.cpp:128-131, 146-151`) — a worker-held lease is
  already legal; `~HostExpertTier` cannot free it (header :129).
- `reset()`/`synchronize_all()` coexistence: CPU leases are not managed
  sources; the shared `HostTierState` keeps memory alive past prefetcher
  reset and even tier destruction.
- Cold fill: `ColdExpertStore::read` is synchronous — the sink inherits the
  serial fill verdict; no new gap (conditional-GO in DESIGN.md unchanged).
- Exactness: routing stays authoritative (sink choice is post-route, per
  authoritative `expert_ids`); unweighted per-expert outputs preserved by
  passing `routing_weight = 1.0f` to the bridge executor contract (dee places
  the weight in the caller's combine).

## Roadmap placement

Lands **after Phase-2 closure, before/at Phase-3 productization, with the
full CGOPipe overlap as an early Phase-5 capability**:

1. The sink rides the lease/DMA machinery — it cannot be validated before
   Phase-2's GPU BATCH #1 (mechanism A/B) proves host tier + prefetcher live.
2. Cheapest first motion: a synchronous CPU-sink arm on the **host-API**
   paths (`moe_forward_experts`/`moe_forward_batch`) — no G4/G6 needed —
   giving an exactness + miss-latency A/B cell as a Phase-2.x mechanism
   experiment (adds one arm, not a new batch).
3. The device-path overlap (G4/G6, CGOPipe shape) pairs naturally with
   Phase-3's full store — an arbitrary 11,776-record universe makes misses
   dominate, which is exactly when a second sink pays — and is explicitly
   listed under AGENTS.md dee-serve direction ("CPU/GPU hybrid miss
   execution"). Recommend: interface closure (G1-G3, G7-G9) in a post-GPU-#1
   cleanup pass; overlap feature deferred to Phase-5 serving work.
