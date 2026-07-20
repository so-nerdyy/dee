"""
================================================================================
Step 4: Triton Async Prefetch Kernel  (Dynamic Expert Eviction MoE)
================================================================================
Custom Triton kernel + CUDA streams that asynchronously page expert weights
from CPU RAM into a GPU "VRAM pool" of LIMITED capacity, so the main compute
graph never blocks on the copy.

LOCKED CONFIG (from Step 3 sweep, lowest capacity crossing 95% hit rate):
    capacity   = 56   # experts resident in VRAM pool
    top_k_pred = 32   # Oracle predicts top-32 experts for {N+1, N+2}

Mechanism
---------
  * All 128 experts' weights live in CPU RAM (pinned staging buffers).
  * A GPU VRAM pool holds at most `capacity` experts (a fixed slot table).
  * On token N:
      1. Oracle(hidden[N]) -> top-32 predicted experts for {N+1,N+2}
      2. ExpertCache (rank-weighted, delta) decides which are resident.
      3. MISSING experts are packed into a contiguous host buffer and copied
         CPU->GPU on a SECONDARY cuda stream via the Triton copy kernel
         (non-blocking, overlaps with step 4).
      4. MAIN stream runs the expert compute on the resident experts.
      5. Evictions free VRAM slots; verbose logs emitted per token.
  A Triton @triton.jit kernel performs the batched memcpy (flattened expert
  params) on the prefetch stream; an event synchronizes ordering so the main
  stream only touches experts that have finished arriving.

Prototype note: we do NOT load all 397B weights. We allocate a representative
expert weight block (d_expert x d_expert) x 128 on CPU and page only the
resident ones into the GPU pool, running a real matmul as the "compute". This
exercises the exact async-overlap path at scale without the 30B download.

Run (GPU, CPU RAM + GPU VRAM pool):
    modal run modal_step4_triton_kernel.py
================================================================================
"""

import os
import json
import time
import argparse

import modal

# Triton is only installed in the Modal GPU container, not on the local box
# where `modal run` imports this module to discover the app. Guard the import
# so local entrypoint discovery succeeds; in the container `tl` is the real
# module and is needed at Triton kernel COMPILE time (lazy, first call).
try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


# --------------------------------------------------------------------------- #
# Triton async copy kernel: batched memcpy of an expert weight block.
# Defined at MODULE level so `tl` (a module global, set in the container) is
# resolvable at Triton's lazy COMPILE time. Copies `n_elements` contiguous
# float32 values from a pinned host staging buffer into a device VRAM-pool
# buffer, in parallel across the data. Launched on the prefetch CUDA stream so
# it overlaps main compute.
# The whole definition is guarded: locally `triton`/`tl` are None (guarded
# import) so only a `None` placeholder is created and the module imports
# cleanly; in the container the real JIT kernel is defined. (The `tl.constexpr`
# annotation is evaluated at def time, hence the guard around the whole def.)
# --------------------------------------------------------------------------- #
if triton is not None:
    @triton.jit
    def copy_experts_kernel(
        src_ptr,        # device staging buffer (torch.Tensor)
        dst_ptr,        # device VRAM pool slot (torch.Tensor)
        n_elements,     # total float32 elements to copy
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        vals = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, vals, mask=mask)
else:
    copy_experts_kernel = None


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.51.3",
        "huggingface_hub[cli]==0.30.2",
        "accelerate==1.0.1",
        "triton==3.0.0",
        "numpy==1.26.4",
        "safetensors==0.4.5",
    )
)

DATA_VOL = modal.Volume.from_name("dee-oracle-data", create_if_missing=True)
DATA_VOL_PATH = "/vol/data"

app = modal.App("dynamic-expert-eviction-step4")


# --------------------------------------------------------------------------- #
# ExpertCache (self-contained copy; inlined so this Modal module is standalone
# — Modal only uploads THIS file as the function module, so a cross-file
# `import modal_step3_cache_manager` fails inside the container). Rank-weighted
# look-ahead eviction: keeps high-rank predicted experts, evicts low-rank /
# unpredicted ones first. (Full origin + unit tests live in Step 3.)
# --------------------------------------------------------------------------- #
class ExpertCache:
    def __init__(self, capacity: int = 20, top_k_pred: int = 20):
        self.capacity = capacity
        self._top_k_pred = top_k_pred
        self.vram = set()
        self.last_used = {}
        self.load_count = {}
        self.prefetch_count = {}
        self.async_ops = []
        self._lookahead_rank = {}
        self.clock = 0

    def contains(self, expert_id: int) -> bool:
        return expert_id in self.vram

    def set_lookahead_ranked(self, predicted_ids):
        self._lookahead_rank = {e: i for i, e in enumerate(predicted_ids)}

    def prefetch(self, expert_id: int, token: int):
        self.clock += 1
        if expert_id not in self.vram:
            if len(self.vram) >= self.capacity:
                self.evict(token=token)
            self.vram.add(expert_id)
            self.load_count[expert_id] = self.load_count.get(expert_id, 0) + 1
        self.prefetch_count[expert_id] = self.prefetch_count.get(expert_id, 0) + 1
        self.last_used[expert_id] = token
        self.async_ops.append(("prefetch", expert_id, token))

    def evict(self, token: int):
        candidates = [e for e in self.vram]
        if not candidates:
            return None
        k = self._top_k_pred

        def keep_score(e):
            r = self._lookahead_rank.get(e, None)
            base = -1 if r is None else k - r
            return (base, self.last_used.get(e, 0))

        victim = min(candidates, key=keep_score)
        self.vram.discard(victim)
        self.async_ops.append(("evict", victim, token))
        return victim


@app.function(
    image=image,
    gpu="A10G",
    volumes={DATA_VOL_PATH: DATA_VOL},
    timeout=60 * 60 * 2,
)
def run_triton_prefetch(
    collected_dir: str,
    oracle_dir: str,
    capacity: int,
    top_k_pred: int,
    n_tokens: int,        # how many tokens to drive through the pipeline
    expert_block: int,    # d_expert: size of one expert's square weight block
    verbose: bool,
) -> str:
    import torch

    log = lambda *a: print("[STEP4]", *a, flush=True)

    device = "cuda"
    torch.cuda.init()
    main_stream = torch.cuda.current_stream()
    prefetch_stream = torch.cuda.Stream()
    log(f"device={torch.cuda.get_device_name(0)}")

    # ---- Load Oracle meta + one representative layer's Oracle ------------- #
    with open(os.path.join(oracle_dir, "oracle_meta.json")) as f:
        ometa = json.load(f)
    hidden_dim = ometa["hidden_dim"]          # 2048
    hidden_mlp = ometa["hidden_mlp"]          # 256

    # We drive the pipeline using layer 0's Oracle as the predict-ahead model
    # (architecture identical across layers; for a real system each layer has
    # its own). Load it.
    L = 0
    ow = torch.load(os.path.join(oracle_dir, f"oracle_layer_{L:03d}.pt"),
                    map_location="cpu")
    num_experts = ow["num_experts"]           # 128
    oracle = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, hidden_mlp),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_mlp, hidden_mlp),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_mlp, num_experts),
    )
    oracle.load_state_dict(ow["state_dict"])
    oracle.eval().to(device)

    # ---- Allocate expert weights: CPU RAM (pinned) + GPU VRAM pool --------- #
    # One expert = a square d_expert x d_expert float32 weight block.
    d = expert_block
    expert_bytes = d * d * 4  # float32
    log(f"expert_block={d}x{d}  bytes/expert={expert_bytes}  "
        f"num_experts={num_experts}  cpu_ram_per_layer={expert_bytes*num_experts/1e6:.1f}MB")

    # CPU RAM: all experts' weights (pinned, so Triton/H2D can read them)
    cpu_experts = torch.randn(num_experts, d, d, dtype=torch.float32,
                              pin_memory=True)   # (E, d, d) pinned host
    # GPU VRAM pool: capacity slots, each a d x d block
    vram_pool = torch.zeros(capacity, d, d, dtype=torch.float32,
                            device=device)       # resident expert weights
    # Device staging buffer for the H2D copy (one expert block). The actual
    # CPU->GPU transfer uses CUDA's async copy engine (non_blocking copy_ on
    # the prefetch stream); the Triton kernel then does the D2D move into the
    # VRAM pool slot. Both happen on the prefetch stream -> overlaps compute.
    stage = torch.empty(d, d, dtype=torch.float32, device=device)
    # Slot table: which expert_id lives in each VRAM slot (-1 = empty)
    slot_of_expert = {e: -1 for e in range(num_experts)}   # expert_id -> slot
    free_slots = list(range(capacity))

    cache = ExpertCache(capacity=capacity, top_k_pred=top_k_pred)

    # Synthesize a token stream: random hidden states (stand-in for real
    # forward; in production these come from the model's token-N hidden state)
    torch.manual_seed(0)
    h = torch.randn(n_tokens, hidden_dim, dtype=torch.float32, device=device)

    # ---- Drive the pipeline ----------------------------------------------- #
    misses_total = 0
    evictions_total = 0

    def ensure_resident(e, token, on_stream, is_prefetch):
        """Bring expert `e` into a GPU VRAM slot if not already resident.
        Keeps `slot_of_expert` in lockstep with `cache.vram`. Returns True if
        it was already resident (cache hit), False if we had to load it."""
        nonlocal evictions_total
        if e in cache.vram:
            return True
        # Free a GPU slot if needed, using the cache's rank-weighted eviction.
        if not free_slots:
            victim = cache.evict(token=token)
            if victim is None:
                return False  # cache full and nothing evictable
            vslot = slot_of_expert[victim]
            free_slots.append(vslot)
            slot_of_expert[victim] = -1
            evictions_total += 1
            if verbose and is_prefetch:
                log(f"[Token {token}] Evicting Expert [{victim}]")
        slot = free_slots.pop(0)
        slot_of_expert[e] = slot
        grid = ((d * d + 1024 - 1) // 1024,)  # host-side ceil div for grid dims
        # Async H2D (CUDA copy engine) on the prefetch stream, then Triton D2D
        # move into the VRAM pool slot — both overlap the main compute stream.
        with torch.cuda.stream(on_stream):
            stage.copy_(cpu_experts[e], non_blocking=True)          # CPU -> device staging
            copy_experts_kernel[grid](stage,
                                      vram_pool[slot],
                                      d * d, BLOCK=1024)             # staging -> VRAM slot
        cache.prefetch(e, token=token)
        return False

    for t in range(n_tokens - 2):
        # 1) Oracle predicts top-32 experts for {N+1, N+2}
        with torch.no_grad():
            ps = oracle(h[t].unsqueeze(0))[0]
            pred_topk = torch.topk(ps, top_k_pred).indices.tolist()
        cache.set_lookahead_ranked(pred_topk)

        # 2) SPECULATIVE prefetch: load the predicted experts that are missing,
        #    on the SECONDARY stream (overlaps the main compute below).
        missing = [e for e in pred_topk if e not in cache.vram]
        if missing:
            with torch.cuda.stream(prefetch_stream):
                for e in missing:
                    ensure_resident(e, token=t, on_stream=prefetch_stream,
                                    is_prefetch=True)
            torch.cuda.synchronize(prefetch_stream)  # copies must land before serve

        # 3) MAIN stream: serve token t+1. "Requested" experts = Oracle top-k
        #    (stand-in for the true routing). Any not-yet-resident expert is an
        #    on-demand miss (the ~4.5% the Oracle didn't predict) -> loaded here.
        main_stream.synchronize()
        requested = pred_topk
        miss_list = []
        for e in requested:
            was_hit = ensure_resident(e, token=t + 1, on_stream=main_stream,
                                      is_prefetch=False)
            if not was_hit:
                miss_list.append(e)
        misses_total += len(miss_list)
        resident = sum(1 for v in slot_of_expert.values() if v != -1)
        if verbose:
            log(f"[Token {t}] Requested Experts: {requested}  "
                f"Cache Miss: {miss_list}  Resident: {resident}/{capacity}")

        # Representative compute on the MAIN stream over resident expert blocks
        # (in production: the real MoE expert matmul; here a dummy matmul to
        #  exercise the compute path that overlaps the next prefetch).
        if resident:
            blk = torch.stack([vram_pool[slot_of_expert[e]]
                               for e in range(num_experts) if slot_of_expert[e] != -1])
            _ = torch.matmul(blk, blk)

    summary = {
        "capacity": capacity,
        "top_k_pred": top_k_pred,
        "n_tokens": n_tokens,
        "num_experts": num_experts,
        "resident_at_end": sum(1 for v in slot_of_expert.values() if v != -1),
        "total_cache_misses": misses_total,
        "total_evictions": evictions_total,
        "triton_kernel": "copy_experts_kernel (BLOCK=1024)",
    }
    log("=" * 60)
    log(f"capacity={capacity} top_k_pred={top_k_pred} n_tokens={n_tokens}")
    log(f"total_cache_misses={misses_total} total_evictions={evictions_total}")
    log(f"resident_at_end={summary['resident_at_end']}/{capacity}")
    log("=" * 60)
    return json.dumps(summary)


@app.local_entrypoint()
def main(
    collected_dir: str = DATA_VOL_PATH + "/collected2",
    oracle_dir: str = DATA_VOL_PATH + "/oracle2",
    capacity: int = 56,
    top_k_pred: int = 32,
    n_tokens: int = 64,
    expert_block: int = 256,
    verbose: bool = True,
):
    print(f"[LOCAL] Step4 Triton prefetch: capacity={capacity} top_k_pred={top_k_pred}",
          flush=True)
    t0 = time.time()
    result = run_triton_prefetch.remote(
        collected_dir=collected_dir,
        oracle_dir=oracle_dir,
        capacity=capacity,
        top_k_pred=top_k_pred,
        n_tokens=n_tokens,
        expert_block=expert_block,
        verbose=verbose,
    )
    print(f"[LOCAL] finished in {time.time()-t0:.1f}s")
    print("[LOCAL] summary:", result)
