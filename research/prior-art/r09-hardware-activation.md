# R9 — Hardware / activation movement: mapping dee's hierarchy onto near-data execution

**Track:** R9 (Phase-6+ architecture note, prior-art sweep)
**Branch:** `research/prior-art-r09` — base `dc78dc4`
**Status:** analysis + prior-art survey only. No engine changes, no remote spend,
no integration edits. Every number below carries its tier label (which storage/
memory tier it was measured or modeled on); unlabeled figures are derived
arithmetic and say so.

## TL;DR

dee today has exactly one cold-tier verb: `ColdExpertStore::read(key, dst, bytes)`
moves a 12.75 MiB expert record toward the compute (`dee.cpp/include/dee/
host_expert_tier.h:30-35`). The MoNDE line of work (DAC'24, arXiv 2405.18832)
inverts the direction: for cold experts, move the ~32 KiB activation round-trip
to where the record lives and compute in place. On dee's canonical geometry the
payload asymmetry is ~408:1 per expert call and ~400:1 per token. The codebase
already contains every hook this needs — a bounded-record CPU executor prototype
(`dee.cpp/experiments/kt_cpu_bridge/`), a per-expert split cost model
(`plan_split`), and a keyed-tier abstraction whose scope/exclusivity rules were
designed model-neutrally (V4.1 audit). What it does NOT yet contain: an
execute-in-place verb in the tier seam, and an exactness ruling on near-data
numeric paths — the two items a Phase-6 evaluation must pin down first.

## 1. The inversion, quantified at dee's geometry

Canonical model (AGENTS.md): 43 MoE layers, 256 routed experts/layer, top-6 +
1 shared, hidden 4096, expert inter 2048. DEE4 record R = 13,369,344 B
(12.75 MiB). Expert FLOPs per (token, expert) = 2·25,165,824 ≈ 50.3 MFLOP —
arithmetic intensity ≈ 3.9 FLOP/byte against the record: GEMV-class, exactly
the "low-Op/B" regime Duplex (MICRO'24, arXiv 2409.01141) built Logic-PIM for.

| Quantity | Value | Tier label |
|---|---|---|
| Parameter movement per expert call | 12.75 MiB | DEE4 record, any tier |
| Activation movement per expert call | 16 KiB in + 16 KiB out (fp32, hidden 4096) = 32 KiB | any link |
| Payload ratio | ~408:1 | derived |
| Cold parameter bytes per decode token (all-miss) | 43 × 6 × 12.75 MiB ≈ 3,289.5 MiB | sealed bank (`STORAGE_ROOFLINE.md:49`) |
| Activation bytes per decode token, worst case (per-expert round trips) | 43 × 6 × 32 KiB ≈ 8.25 MiB fp32 | derived |
| … with near-data weighted combine (one 16 KiB return per layer) | 43 × (96 + 16) KiB ≈ 4.7 MiB fp32 | derived; needs §5's combine note |
| Inversion ratio per token | ~400–700× | derived |

Batch-amortization boundary: parameter movement costs R once per fetch
regardless of how many of the layer's tokens hit that expert; activation
movement scales with m tokens. Break-even at equal link bandwidth is
m* ≈ R / 32 KiB ≈ 408 tokens/expert — i.e. activation movement wins whenever
an expert serves fewer than ~400 tokens before eviction. MoNDE's premise
("the majority of experts receive a significantly small number of tokens") is
therefore not an assumption dee needs to import; it is measurable, and on the
sealed v50 journal it is true in the extreme: 2,364 unique records / 5,099
engine-dedup requests / 16 forwards, only 935 records ever repeat, top-1% of
records cover 6.8% of activations (`research/phase2-regime-c/CONTRACT.md`
§3.2, sealed-journal stats). At batch-1 decode, m per expert call is 1.

The one regime where the inversion weakens: high-throughput serving with
large batch (dee-serve), where m per expert grows and parameter fetches
amortize. This is a scheduling input, not a blocker — the cost model must
carry m explicitly (§4).

## 2. What near-data execution means in dee's vocabulary

dee's hierarchy is a residency scheduler: `ColdExpertStore → HostExpertTier →
DeviceExpertTier → ExactExpertExecutor`. Placement is already a policy
decision keyed by `TierExpertKey{model, layer, expert, representation}`
(`host_expert_tier.h:15-23`); identity never changes with location. Near-data
execution adds a second verb — `execute(key, activation_in, routing_weight) →
activation_out` — implemented by an executor co-located with a tier. Per tier:

| dee tier (today) | Near-data analog | Prior art |
|---|---|---|
| `DeviceExpertTier` (VRAM, T4 HBM ~5.4–5.7 GB/s effective H2D — T4 profile) | hot experts stay on GPU — unchanged | MoNDE hot path; Sieve GPU side |
| `HostExpertTier` (DDR4/5 host RAM; sealed envelope ≤17–24 GiB) | **in-RAM expert execution**: `CpuExpertExecutor` exists today | kt-cpu-bridge; KTransformers AMX/AVX2; FreeToken; moe-l2 |
| `ColdExpertStore` behind CXL.mem (DDR-class expander) | CXL-NDP executes cold experts in the expander | MoNDE; context-aware CXL-NDP (2512.04476); DynaNDE; M2NDP (2404.19381); CXL-NDP BW amplification (2509.03377) |
| Device-resident HBM-PIM / LPDDR-PIM | hot-ish experts executed in PIM banks | Samsung Aquabolt-XL / LPDDR5X-PIM; Duplex; Sieve; PIMoE; HDA-MoE; HCRMap |
| `Dee4ExpertStore` on NAND (sealed bank: Kaggle /tmp @0.29–0.37 GiB/s; theoretical NVMe 3–7 GB/s) | **compute-enabled storage / in-flash computing** executes cold experts at the bank | KVNAND (2512.03608); NVLLM (2604.25699); InstInfer (2409.04992); Samsung SmartSSD class |

Note the two placements dee gets for free that the literature treats as
separate systems: in-RAM (the bridge already prototyped) and in-flash (the
dee4 record is read-mostly, immutable, page-aligned — `kPage=4096`,
`expert_store.cpp:610` — and 12.75 MiB contiguous, far friendlier to NAND
than KV cache, which is write-heavy; cf. the C1–C3 conditions in "HBF Sucks!",
arXiv 2608.11668 — expert records satisfy them, KV does not).

## 3. Tier by tier

### 3a. CPU-RAM compute — the bridge is the prototype

`dee.cpp/experiments/kt_cpu_bridge/` (`research/kt-cpu-bridge/` docs): a
bounded per-expert CPU executor whose input is exactly a host-tier record.
Proven there: weight bytes are byte-identical/zero-copy to KT's MXFP4 path;
scale re-encode is lossless and cacheable; a `ReferenceCpuExecutor` (fp32,
ISA-neutral, deterministic, bitwise-stable) exists as the arbiter, and a
KT-faithful executor quantifies the bf16-boundary delta rather than assuming
it (`CPU_EXECUTOR_DESIGN.md` §4, `SUMMARY.md`). `plan_split` in
`cost_model.py` already enumerates `q* = argmin max(T_gpu(q), T_cpu(m−q))`
over dee's tiny top-k — that *is* the hot/cold split decision, minus measured
constants. Steady-state residency equals source size (~12.75 MiB/expert), no
full-model RAM copy — the property upstream KT lacks and dee's contract
requires.

Sanity check on the economics, tier-labeled: one cold fetch on the sealed
bank costs ~18.9 ms (single-flight anchor) to ~96 ms (measured shared-lane
service) (`SERIALIZATION_VERDICT.md`); a 12.75 MiB fp32 GEMV-class expert on
a host DDR channel (~25–100 GB/s) is bandwidth-≈0.13–0.5 ms, compute-bound at
~50 MFLOP — even at a pessimistic 5–10 GFLOP/s effective, ~5–10 ms. In-RAM
execution beats the sealed-bank fetch at m=1 without heroic CPU hardware; it
loses to a modern NVMe bank (~1.9 ms at 7 GB/s) only if CPU exec exceeds that
— measure, don't assume. This is the first Phase-6 arm because it needs no
new hardware.

### 3b. CXL-attached memory + CXL-NDP

MoNDE's device is literally a CXL-attached LPDDR module with NDP MAC arrays
(64 × 4×4 MAC arrays per core, 3.0 mm² overhead; up to 7.5×/3.7× vs parameter
offloading on Switch-Transformer-class models, encoder/decoder). For dee:
a CXL.mem expander is a `HostMemoryBackend` (`allocate`/`pin` over CXL memory)
with no tier-code change — the V4.1 audit shows all geometry is parametric.
A CXL-NDP device adds the `execute` verb behind the same keying. Follow-on
work supplies the scheduling ideas dee-serve would need: context-aware
placement from prefill statistics (2512.04476 — caution: it pairs this with
per-expert 1–4-bit quantization, which is approximate-mode-only under dee's
contract), DynaNDE's per-layer NPU/NDP split with reuse-awareness
(2609.00407), M2NDP's general-purpose memory-mapped NDP offload ABI
(2404.19381), and transparent in-device bandwidth amplification
(2509.03377). CXL link reality check: load-to-use latency is hundreds of ns,
link BW ≈ PCIe-class (~50–64 GB/s gen5 x16) — plenty for 32 KiB activations,
irrelevant for records that stay put.

### 3c. PIM (HBM/LPDDR/AXDIMM)

Product anchors: Samsung Aquabolt-XL HBM2-PIM (8.9× GEMV microkernel; IEEE
Micro'22), AXDIMM (+80% vs RDIMM), CXL-PNM (4.4× / −53% energy; IEEE
Micro'24), LPDDR5X-PIM at Hot Chips 2026 (614 GB/s internal vs 76.8 GB/s
pin-side; +3.01× token throughput on Llama-3.1-8B INT8×INT4; standard
561-ball package). MoE-specific: PIMoE (DAC'25; throttle-aware NPU↔PIM
offload, 4.5× vs A100), Duplex (Op/B-tiered xPU/Logic-PIM co-processing —
dee's experts sit at Op/B ≈ 4, inside Logic-PIM's target band), Sieve
(runtime GPU↔HBM-PIM expert partitioning driven by the measured bimodal
token-per-expert distribution — the same skew dee's journal shows),
HDA-MoE / HCRMap (3D/3.5D NMP placement). Role in dee: PIM extends
*device-tier* compute, i.e. experts already hot enough to deserve HBM get
executed without SM involvement — useful for dee-serve batching more than
for the cold-fill problem.

### 3d. Compute-enabled storage / in-flash — the KVNAND tie

KVNAND (arXiv 2512.03608): DRAM-free in-flash computing — model weights AND
KV cache in compute-enabled 3D hybrid-bonding NAND; IFC for memory-bound ops,
head-group parallelism, page-level mapping; ~2× geomean vs DRAM-equipped IFC
baselines at ≤10K context. NVLLM (2604.25699) is the direct expert analog:
FFN weights live in flash, GEMV decomposed to dot-product primitives on
out-of-order PE lanes reading raw NAND pages with inline ECC — 16.7–37.9× vs
A800 out-of-core. InstInfer (2409.04992) shows the CSD pattern for the
attention/KV side (flash-aware in-storage attention on computational storage
drives). Mapping: `Dee4ExpertStore` records are immutable, contiguous, and
read-mostly — the ideal IFC workload class (no program/erase on the read
path). An in-flash expert executor is a `ColdExpertStore` variant that can
answer `execute()` without the bytes ever leaving the device — the extreme
end of the same inversion, on exactly the tier where dee's measured pain
lives (single-stream-saturated bank, `SERIALIZATION_VERDICT.md`).

## 4. Near-data execution cost model

Per (layer, expert) call, let R = record bytes (12.75 MiB), a = activation
round-trip bytes (32 KiB fp32 baseline), m = tokens routed to that expert
this window, B_tier = tier bandwidth, L_link = activation RTT+submit latency,
T_exec_* = per-expert compute on that engine.

- Parameter movement (today): T_param ≈ R/B_tier (+ T_h2d if device-bound) + T_exec_gpu
- Activation movement: T_act ≈ L_link + a/B_link + T_exec_ndp

Executed-in-place wins when T_exec_ndp < R/B_tier + T_h2d + T_exec_gpu −
(L_link + a/B_link). With per-token batching: the left side scales with m,
the right side amortizes as R/(B_tier·m) — the crossover m* ≈ 408 at equal
link speed (§1). Reference points, all tier-labeled:

- Sealed bank (Kaggle /tmp, 0.29–0.37 GiB/s): R/B ≈ 35–46 ms; measured
  single-flight service 18.9–35.5 ms, shared ~96 ms → near-data exec has a
  ~20–100 ms budget per expert. Trivially winnable by anything faster than a
  spreadsheet.
- Modern NVMe (3–7 GB/s): R/B ≈ 1.9–4.5 ms → NDP must deliver a full
  fp32-exact expert in ~<2 ms plus link RTT; only real PIM/IFC silicon
  clears this, and only at m small.
- H2D tier (T4 PCIe gen3 measured 5.4–5.7 GB/s effective): R/B_h2d ≈ 2.3 ms —
  the same arithmetic applies one tier up (host-RAM exec vs H2D+GPU exec).
- GPU exec anchor: DSv4 campaign attributes ~26 ms-class per-invocation GPU
  expert compute on T4 with total GPU compute ~1% of decode wall
  (`STORAGE_ROOFLINE.md:55-66`) — compute is not today's bottleneck;
  movement is. Under activation movement, the GPU's idle-95-99% problem
  becomes "idle waiting on a 32 KiB reply" only if L_link is small — CXL/PCIe
  RTTs (~sub-µs to ~µs) are noise next to all of the above.

`plan_split(m, t_cpu, t_h2d, t_gpu)` is the wired-in skeleton; Phase 6 =
feeding it measured constants per tier and adding an `execute` arm to the
enumeration.

## 5. What changes vs stays invariant under the exactness contract

Invariant (unchanged — the contract is the point):

- **Router authoritative.** `route_topk` runs where it runs today; the
  near-data device receives (expert key, activation, routing_weight) — never
  routing freedom. Prediction, if ever used, stays hint-only.
- **Expert identity.** `TierExpertKey` scoping is location-independent by
  construction; execution site is a scheduling attribute like residency.
- **Expert bytes immutable.** In-place execution is read-only — aligned with
  flash/NDP media and with dee's integrity model.
- **Expert semantics.** The executor must implement dee's exact math: FP4
  E2M1 dequant + E8M0 block-32 scale law, asymmetric SwiGLU clamp
  (gate max-only, up ±, limit 10.0), routing-weight placement, fp32
  accumulation, deterministic ordering — i.e. `ReferenceCpuExecutor`'s
  contract, verbatim, on whatever silicon.
- **Ordering/outputs.** Per-token outputs must be reproducible against the
  trusted reference, bit-exact, as today.

What changes (the Phase-6 design surface):

- **A second verb in the tier seam.** `execute(key, in, w) → out` beside
  `read(key, dst)`. Placement becomes hot-execute / move-params /
  execute-in-place, per expert per layer — a scheduler decision, not an
  identity change.
- **Activation transport + combine locality.** Sending routing weights with
  activations lets a remote node return ONE weighted-summed 16 KiB vector per
  layer (dee's executor already applies rw inside the expert call —
  `CPU_EXECUTOR_DESIGN.md` §3). Caveat for bit-exactness: cross-expert
  accumulation order must match dee's combine order, or the delta must be
  bounded and declared.
- **Numerics boundary must be pinned, not assumed.** Host-CPU fp32 is already
  arbiter-exact. PIM/IFC MAC arrays are typically bf16/fp8/int8 — that is
  the same class of deviation the bridge measured (KT bf16 boundaries:
  cosine 0.99999, p95_rel ~0.067 on synthetic fixtures). Under dee's rules
  that path is approximate-serving-mode material, never EXACT, unless the
  silicon reproduces fp32 semantics. The ds9-v13 reject-numerical precedent
  applies: an unmeasured numeric path is a rejected path.
- **Activation dtype.** Shipping activations in bf16 halves a (32→16 KiB) but
  adds a rounding the exact path lacks — approximate-mode or a contract
  amendment, decide explicitly.
- **Failure semantics.** Today `stage()==false` fails closed (abort;
  `expert_tiers.cpp:126-135`, `engine.cpp:3153`). A remote-execute failure
  must degrade to the parameter-fetch path, never skip an expert.
- **Metrics.** `TierMetrics` already splits `SSD_bytes`/`H2D_bytes`
  (`PHASE2_METRICS.md`); Phase 6 adds `ACT_bytes_out/in` + per-site
  `exec_ms` so bytes-per-token stays the roofline unit (AGENTS.md economics).

## 6. Evidence a Phase-6 evaluation would need

1. **Measured constants at real geometry** — t_cpu (Reference + KT-faithful),
   t_gpu, t_h2d, L_link/a per link, each labeled by tier and host; the
   bridge's `bench_cpu_expert.py` + `plan_split` is the harness.
2. **Exactness gate** — bitwise/delta metrics vs `ReferenceCpuExecutor` on
   real-checkpoint fixtures (`DEE_REAL_EXPERT_DIR`), Phase-D-style; any
   non-fp32 silicon path declared approximate before benchmarking, not after.
3. **Workload-truth for the premise** — tokens-per-expert distribution and
   repeat structure on the sealed journal (already: 935/2364 repeat, top-1%
   = 6.8% of activations) plus a multi-request journal for rank stability
   (the regime-C caveat: skew stability is a workload assumption, not yet a
   measured property — `CONTRACT.md` §5.5).
4. **End-to-end A/B on sealed replay** — fetch-path vs execute-in-place arm
   on identical miss streams; wall/token; the serial-fill floor (86.3 s/
   response sealed-bank) and the >~0.6–0.7 GiB/s conditional-GO bank are the
   two reference regimes.
5. **Economics** — $/GB and pJ/byte by tier (DRAM vs CXL-DDR vs NAND vs HBM),
   because Phase 6 is explicitly the economics phase; the inversion's real
   prize is cost-per-token, not a tok/s headline.
6. **Falsifiers, stated now** — near-data exec loses if: NDP fp32-exact
   throughput can't beat ~2 ms/expert on a modern bank; activation RTT
   dominates at high m (serving batch); no silicon exists that meets the
   exactness bar (→ the idea survives only as approximate mode); or bank
   placement (faster SSD) makes fetch cheap enough that the second verb buys
   nothing — the Phase-1 feed-side verdict in new clothes.

## 7. Prior-art index (external; idea sources, not acceptance evidence)

- **MoNDE** — Kim et al., DAC'24, arXiv 2405.18832. The seed: activation
  movement, hot→GPU / cold→CXL-LPDDR-NDP; 7.5×/3.7× encoder/decoder vs
  offloading; 3.0 mm² NDP overhead.
- **PIMoE** — DAC'25 (doi 10.1109/DAC63849.2025.11132528): throttle-aware
  NPU↔PIM offload + data condenser; 4.5× vs A100.
- **DynaNDE** — arXiv 2609.00407: per-layer NPU/NDP expert scheduling with
  reuse-aware runtime; 2.6×/2.2× prefill/decode.
- **Context-aware CXL-NDP MoE** — arXiv 2512.04476: prefill-statistics-guided
  hot-pinning + per-expert bitwidths (approximate-mode flag for dee).
- **Sieve** — arXiv 2605.11277: bimodal token-per-expert quantification;
  runtime GPU↔HBM-PIM partitioner.
- **Duplex** — MICRO'24, arXiv 2409.01141: Op/B-tiered xPU + Logic-PIM
  co-processing; dee's experts land at Op/B ≈ 4 (GEMV-class) at m=1.
- **HDA-MoE** (2609.08682), **HCRMap** (2607.11586): 3D/3.5D NMP mapping.
- **Survey** — "A Survey on Inference Optimization Techniques for MoE
  Models", ACM TALLIP (doi 10.1145/3794845; arXiv 2412.14219): hardware-level
  section = the taxonomy this note's §3 borrows; repo awesome-moe-inference.
- **CXL infra** — M2NDP (2404.19381), CXL-NDP bandwidth amplification
  (2509.03377), Panmnesia CXL-GPU (two-digit-ns controller claim).
- **PIM products** — Samsung HBM2-PIM Aquabolt-XL / AXDIMM (IEEE Micro'22),
  CXL-PNM + LPDDR-PIM (IEEE Micro'24, doi 10.1109/mm.2024.3375352),
  LPDDR5X-PIM (Hot Chips 2026: 614 GB/s internal, 3.01× tok/s edge demo).
- **Flash/CSD** — KVNAND (2512.03608), NVLLM (2604.25699), InstInfer
  (2409.04992), counterpoint "HBF Sucks!" (2608.11668).
- **In-repo CPU-hybrid prior art** — KTransformers (audited, bridge built),
  MoE-Infinity, moe-l2, FreeToken, Kimi-K3-in-C, Edge0 (per AGENTS.md
  external-references list).

## 8. Standing rules for this note's claims

1. Every performance figure carries a tier label (sealed bank / modern NVMe /
   H2D link / host RAM / NDP-internal); derived arithmetic is marked derived.
2. No number here is acceptance evidence: external figures are vendors' and
   authors' claims; internal figures are from the cited sealed evidence.
3. Anything executing experts on non-fp32-exact silicon is approximate-mode
   until a bounded-delta proof exists — same rule as the kt bridge's two
   executor classes.
4. This note proposes no engine edit; the `execute` verb is a Phase-6 design
   task gated on §6's evidence list.
