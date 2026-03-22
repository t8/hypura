# Research Integration Plan for Hypura

> Temporary working document for implementing techniques from FlexGen, PIPO, PowerInfer/PowerInfer-2, ntransformer, CHEOPS, and FlashMoE into the Hypura codebase.

## Codebase Context

- **Target hardware:** M1 Max, 32GB unified memory, ~5.1 GB/s NVMe sequential read
- **Current perf:** Mixtral 8x7B Q5_K_M expert-streaming 2.19 tok/s; Llama 3.3 70B Q4_K_M dense-FFN-streaming 0.20 tok/s
- **Architecture:** Rust + llama.cpp FFI, Metal GPU, custom GGML buffer type for NVMe tensors
- **Unified memory:** No CPU<->GPU DMA hop. NVMe->unified memory is the only I/O path.

---

## Change 1: Fix LP Objective to Model I/O Overlap

**Source:** FlexGen (LP formulation uses `max(compute, io)` per stage, not `compute + io`)

**Problem:** The LP in `src/scheduler/placement.rs` minimizes `sum(size * access_freq / bandwidth)` across all tensors. This is additive — it treats NVMe I/O cost as always paid, even when prefetch can hide it behind the previous layer's compute. Meanwhile, the estimator in `src/scheduler/estimator.rs` already models prefetch hiding (`if prev_compute >= io_time, io is free`). The LP and estimator disagree on cost, so the LP makes suboptimal placement decisions.

**File:** `src/scheduler/placement.rs`

**What to change in `lp_assign()`:**

The current objective is:
```
objective += size_bytes * access_freq / tier_bandwidth * x_tier[i]
```

Replace with a per-layer `max(compute_time, io_time)` formulation:
- For each layer L, define:
  - `compute_L` = sum of `(size * freq / gpu_bw) * x_gpu[i]` + `(size * freq / ram_bw) * x_ram[i]` for tensors in layer L
  - `io_L` = sum of `(size * effective_freq / nvme_bw) * x_nvme[i]` for tensors in layer L (with MoE cache hit discount)
- Introduce auxiliary variable `layer_cost_L >= compute_L` and `layer_cost_L >= io_L` (standard LP linearization of max)
- Minimize `sum(layer_cost_L)` + sync overhead per layer

This requires adding per-layer auxiliary variables. The LP is already using `good_lp` MIP solver which supports this.

**Also update `greedy_assign()`:** The greedy fallback should use the same overlap-aware cost model when deciding whether an additional tensor on NVMe is "worth it" — if the layer's GPU compute time already exceeds the NVMe read time, more NVMe tensors in that layer are nearly free.

**Expected impact:** Better placement decisions for the "borderline" tensors at the GPU/NVMe boundary. Especially relevant for MoE layers where expert tensors are large but only 25% accessed (Mixtral top-2-of-8).

**Validation:** After change, run `estimate_performance()` on the same model and verify the LP plan's estimated tok/s matches more closely with the estimator's prediction. Currently they may diverge.

---

## Change 2: I/O Request Coalescing

**Source:** CHEOPS (found 128 KiB reads dominate; NVMe bandwidth saturates at 512 KiB+)

**Problem:** `load_layer_data()` in `src/compute/nvme_backend.rs` issues one `pread()` per tensor region. Individual tensors (norms, biases) can be very small. Many small reads underutilize NVMe bandwidth even at 5.1 GB/s.

**File:** `src/compute/nvme_backend.rs`

**What to change in `load_layer_data()` and IoPool task splitting:**

1. After sorting regions by `file_offset` (already done), scan for **adjacent or nearby regions** (gap < 64 KiB) and merge them into a single coalesced read. Read the entire span including the gap bytes (which are discarded). This trades ~64 KB wasted reads for far fewer I/O syscalls.

2. Enforce a **minimum read size of 512 KiB**. If a coalesced region is still smaller than 512 KiB, pad the read to 512 KiB (aligned to page boundaries). The extra bytes land in the already-allocated buffer and are ignored.

3. When splitting work across IoPool workers, split by **byte volume** not by region count. Currently regions may be distributed unevenly if some are much larger than others. Each worker should get roughly `total_bytes / num_workers` of contiguous I/O.

**Implementation sketch:**
```rust
fn coalesce_regions(regions: &[TensorLocation], max_gap: usize, min_read: usize) -> Vec<CoalescedRead> {
    // Sort by file_offset (already done upstream)
    // Merge regions with gap < max_gap into single reads
    // Pad reads smaller than min_read to min_read (page-aligned)
    // Return list of (file_offset, read_size, Vec<(buffer_offset, data_offset_within_read, data_size)>)
}
```

The `CoalescedRead` struct maps each original tensor's data within the larger read buffer:
```rust
struct CoalescedRead {
    file_offset: u64,        // aligned start of pread
    read_size: usize,        // total bytes to read (>= min_read)
    mappings: Vec<RegionMapping>,  // where each tensor's data lands
}
struct RegionMapping {
    buffer_dest: usize,     // destination offset in posix_memalign buffer
    read_offset: usize,     // offset within the coalesced read
    size: usize,            // actual tensor data size
}
```

After the coalesced `pread()`, copy each tensor's slice from the read buffer to its final buffer position (if they differ). For truly contiguous tensors that are already adjacent in the buffer, no copy is needed — the pread lands directly.

**Expected impact:** Reduce I/O syscall count by 3-10x per layer. More importantly, each syscall transfers enough data to approach peak NVMe bandwidth. CHEOPS data suggests this alone could improve effective bandwidth from ~2-3 GB/s to ~4.5+ GB/s.

---

## Change 3: Async I/O (dispatch_io or io_uring-style)

**Source:** CHEOPS (libaio is 2.9-5.5x over POSIX for tensor transfers)

**Problem:** IoPool workers use synchronous `pread()`. Even with multiple worker threads, synchronous I/O means each thread blocks during the kernel I/O path. On macOS, the async I/O options are:
- `dispatch_io` (GCD) — Apple's recommended async file I/O
- `aio_read` (POSIX AIO) — available but less optimized on macOS
- `kqueue` + non-blocking fd — lower level

**File:** `src/compute/nvme_backend.rs`

**Recommended approach: `dispatch_io` via raw FFI:**

Create a new module `src/io/dispatch_io.rs` that wraps macOS Grand Central Dispatch file I/O:

```rust
// FFI to libdispatch
extern "C" {
    fn dispatch_io_create(type_: u64, fd: i32, queue: *mut c_void, cleanup: extern "C" fn(i32)) -> *mut c_void;
    fn dispatch_io_read(channel: *mut c_void, offset: i64, length: usize, queue: *mut c_void, handler: extern "C" fn(*mut c_void, i32, *mut c_void));
    // ...
}
```

However, this is complex FFI. A **simpler first step** is to use `preadv()` (scatter-gather I/O) which is available on macOS and can read multiple non-contiguous regions in a single syscall:

```rust
use libc::{iovec, preadv};

fn read_coalesced(fd: RawFd, regions: &[CoalescedRead]) {
    let iovecs: Vec<iovec> = regions.iter().map(|r| iovec {
        iov_base: buffer_ptr.add(r.buffer_offset) as *mut c_void,
        iov_len: r.size,
    }).collect();
    preadv(fd, iovecs.as_ptr(), iovecs.len() as i32, file_offset);
}
```

`preadv` does **scatter-gather** in a single syscall, reducing kernel transitions. Combined with F_NOCACHE (already set), this should approach the async I/O performance gains CHEOPS found.

**Migration path:**
1. First: replace per-region `pread()` calls with `preadv()` using coalesced regions from Change 2
2. Later (optional): explore `dispatch_io` for true async submission if `preadv` isn't sufficient

**Expected impact:** 1.5-3x bandwidth improvement on the I/O path. Combined with coalescing, this addresses the primary bottleneck for streaming mode (0.03 tok/s on 70B model).

---

## Change 4: Double-Buffered Layer Streaming

**Source:** ntransformer SLEP pipeline

**Problem:** Current prefetch thread loads the next layer while the current layer computes, but `ensure_layer_loaded()` blocks until the layer is fully resident. If the NVMe read takes longer than the GPU compute, there's a stall. The prefetch thread also can't start loading layer N+2 until layer N+1's load completes.

**File:** `src/compute/nvme_backend.rs`

**What to change:**

Allocate **two staging regions** in the NVMe buffer instead of one:

```rust
struct DoubleBuffer {
    buffer_a: (*mut u8, usize),  // (ptr, capacity)
    buffer_b: (*mut u8, usize),
    active: AtomicBool,  // false = A is for compute, B is for loading; true = swapped
}
```

Pipeline:
```
Step 1: Load layer N into Buffer A (blocking first load)
Step 2: Start loading layer N+1 into Buffer B | GPU computes layer N from Buffer A
Step 3: Start loading layer N+2 into Buffer A | GPU computes layer N+1 from Buffer B
Step 4: (repeat, alternating buffers)
```

On unified memory (M1 Max), there's no DMA copy — the GPU reads directly from the buffer. So the pipeline is:
- **NVMe read** (one buffer) overlapped with **GPU compute** (other buffer)
- Buffer swap is just a pointer swap + status flag flip

**Changes to `eval_callback()`:**
- Instead of `ensure_layer_loaded(current_layer)` which blocks, check if `active_buffer` already has the layer (it should, from prefetch)
- After launching compute, immediately signal prefetch thread to start loading `current_layer + 2` into the inactive buffer
- The `release_layer()` call becomes a buffer swap instead of MADV_FREE

**Changes to `PrefetchState`:**
- Add `double_buffer: Option<DoubleBuffer>` field
- Modify `prefetch_worker` to always target the inactive buffer
- Add buffer-swap synchronization (Condvar or atomic flag)

**Memory cost:** 2x the per-layer NVMe buffer size. For Mixtral with ~2 MoE layers on NVMe, each layer is ~1.7 GB (all 8 experts). Two buffers = ~3.4 GB. This is tight on 32GB but feasible since keep-resident mode (where double-buffering isn't needed) handles the small-spill case.

**Alternative for memory-constrained cases:** Use single buffer but with **overlapped partial reads** — start reading the first half of the next layer while computing, then read the second half. This is the PIPO approach (8 MB blocks from disk, 32 MB blocks to GPU). On unified memory, the block sizes would differ — worth profiling.

**Expected impact:** Eliminates the NVMe stall gap between layers. For layers that take longer to read than compute, this effectively doubles throughput since I/O and compute now overlap continuously.

---

## Change 5: Wire Up Two-Phase MoE Expert Loading

**Source:** PowerInfer-2 (load gate first, compute routing, then load only selected experts)

**Problem:** The infrastructure exists but isn't connected:
- `intercept_router_output()` in `nvme_backend.rs` — reads `ffn_moe_argsort` tensor to extract selected experts
- `load_expert_slices()` — loads only selected expert strides (25% of data for top-2-of-8)
- `neuron_cache` — tracks which expert slices are already loaded
- `coactivation.rs` — predicts which experts will fire next

But in the `eval_callback()` hot path, these are not fully integrated. The callback currently does layer-level prefetch, not expert-level.

**File:** `src/compute/nvme_backend.rs` (eval_callback function)

**What to change:**

In `eval_callback()`, for MoE layers on NVMe:

**Before compute (ask=true):**
1. Check if this is an MoE layer (has `expert_layouts`)
2. If `selected_experts` are known (from previous layer's router output or co-activation prediction):
   - Call `load_expert_slices(layer, expert_ids)` instead of `load_layer_data(layer)`
   - This loads only the 2 selected experts (~25% I/O) + non-expert regions (norms, router)
3. If `selected_experts` are NOT known (first MoE layer or cold start):
   - Load non-expert regions (norms, router) immediately
   - Load all experts (fallback to full layer load) — or use co-activation prediction

**After compute (ask=false):**
1. After `ffn_moe_argsort` tensor is computed, call `intercept_router_output()` to capture selected experts
2. Store in `selected_experts` HashMap for the CURRENT layer (used if this layer is re-evaluated)
3. Use co-activation matrix to predict next MoE layer's experts:
   ```rust
   let predicted = coactivation.predict_next_layer(current_layer, &selected, top_k=4);
   ```
   Prefetch those predicted experts for the next MoE layer (speculative)
4. Record observation: `coactivation.record(layer, &selected)` and `record_cross_layer(layer, &prev_selected, &selected)`

**Prefetch thread changes:**
- Add `PrefetchRequest::ExpertSlices { layer_idx, expert_ids }` variant (already defined)
- Prefetch worker handles expert-level loads using `load_expert_slices()`

**Neuron cache integration:**
- Before `pread()` in `load_expert_slices()`, check `neuron_cache.is_loaded(layer, expert_id, tensor_type)`
- If hit, skip the pread (data is already in buffer from previous token)
- If miss, pread + `neuron_cache.mark_loaded(layer, expert_id, tensor_type)`
- On eviction return from `mark_loaded()`, call `MADV_FREE` on the evicted expert's buffer region

**Expected impact:** For Mixtral (top-2-of-8), reduces NVMe I/O per MoE layer by ~75%. Combined with neuron cache (expected 96.5% hit rate after warmup), the actual NVMe traffic after the first ~10 tokens drops to near-zero for MoE layers. This is the single biggest win for Mixtral specifically.

---

## Change 6: Recency+Frequency Cache Heuristic (Replace Pure LRU)

**Source:** FlashMoE (LRU evicts reused-within-5-steps experts 34.2% of the time; combined recency+frequency scores are complementary)

**Problem:** `src/cache/neuron_cache.rs` uses pure LRU eviction. FlashMoE found LRU makes correct eviction decisions only ~56% of the time vs optimal. A simple weighted recency+frequency score outperforms both LRU and LFU.

**File:** `src/cache/neuron_cache.rs`

**What to change:**

Replace the `VecDeque<CacheKey>` LRU list with a scored eviction policy:

```rust
struct CacheEntry {
    last_access: u64,       // token counter when last accessed
    access_count: u32,      // total accesses since loaded
    loaded_at: u64,         // token counter when first loaded
}

struct NeuronCache {
    entries: HashMap<CacheKey, CacheEntry>,
    capacity: usize,
    current_token: u64,     // monotonically increasing token counter
    hits: u64,
    misses: u64,
}
```

**Eviction score** (lower = more likely to evict):
```rust
fn eviction_score(entry: &CacheEntry, current_token: u64) -> f64 {
    let recency = 1.0 / (1 + current_token - entry.last_access) as f64;
    let frequency = entry.access_count as f64 / (1 + current_token - entry.loaded_at) as f64;
    0.5 * recency + 0.5 * frequency  // equal weight; tune empirically
}
```

On eviction, evict the entry with the **lowest score** (least recently and least frequently used).

**Add `advance_token()` method** called once per generated token from the inference loop:
```rust
fn advance_token(&mut self) {
    self.current_token += 1;
}
```

This is a minimal change that doesn't require ML training (unlike FlashMoE's full neural cache policy) but captures the complementary signals that FlashMoE identified.

**Expected impact:** +10-15% cache hit rate over pure LRU (FlashMoE's neural policy achieved +21%, this simpler heuristic should capture roughly half of that). More importantly, it eliminates the worst-case LRU pathology where a frequently-used expert is evicted just because it wasn't the most recent access.

---

## Change 7: Layer Skip via Cosine Similarity (Experimental/Lossy)

**Source:** ntransformer (skip 20/80 layers at >0.98 cosine similarity for 67% throughput gain)

**Problem:** For heavy-overflow models (70B), most layers are on NVMe. Each NVMe layer costs ~200-400ms of I/O. Skipping layers that contribute minimally to output quality eliminates their I/O cost entirely.

**This is lossy/experimental.** It should be behind a `--layer-skip` flag and NOT enabled by default.

**New file:** `src/scheduler/layer_skip.rs`

**Approach:**

1. **Calibration pass** (offline, run once per model):
   - Forward pass a reference prompt through all layers
   - After each layer, capture the hidden state tensor
   - Compute cosine similarity between layer N's output and layer N-1's output
   - Record per-layer similarity scores
   - Save to `~/.hypura/layer_skip/{model_name}.json`

2. **Runtime skip logic:**
   - Load similarity scores at startup
   - For layers with similarity > threshold (default 0.98), skip the layer entirely
   - "Skip" means: don't load from NVMe, don't compute, pass the previous layer's output forward
   - Only skip NVMe-tier layers (GPU-resident layers are cheap to compute anyway)

3. **Integration with eval_callback:**
   - In `eval_callback(ask=true)` for a skippable layer, return early without loading
   - Need to verify llama.cpp's compute graph allows skipping layers (may need to zero out the layer's contribution or use an identity pass)

**Limitations:**
- Requires llama.cpp compute graph modification (or a way to make a layer a no-op)
- Quality impact varies by model — needs per-model calibration
- Not compatible with all architectures (residual connections may propagate errors)

**Expected impact:** If 25% of NVMe layers can be skipped, that's a 25% reduction in NVMe I/O time per token. On the 70B model where NVMe I/O dominates, this could be the difference between 0.03 and 0.04+ tok/s. Combined with other I/O improvements, the cumulative effect is larger.

---

## Experimental Results (2026-03-17)

### I/O Coalescing (Change 2): DISPROVEN

Implemented coalescing of file-adjacent regions into fewer, larger pread calls. Reduced syscall count per layer from ~9 to ~1-3. **Result: no throughput improvement.** Added ~10% overhead to Mixtral from CoalescedRegion indirection in the hot path. Reverted.

**Conclusion:** Syscall count is not the bottleneck for NVMe streaming. The CHEOPS assumptions about small-read overhead do not apply to this hardware/access pattern.

### LP Overlap Objective (Change 1): IMPLEMENTED

Replaced additive cost model with per-layer `max(compute, io)` formulation including cross-layer prefetch overlap constraints. LP, greedy fallback, and `quick_estimate` all now use the same overlap model as the estimator.

**Result:** Placement unchanged for both Mixtral and Llama 70B — capacity constraints dominate on M1 Max 32GB. The change is architecturally correct and will affect placement on hardware where the GPU/NVMe boundary is less capacity-bound.

### I/O Microbenchmark (`hypura iobench`): KEY FINDINGS

Isolated microbenchmark testing raw pread throughput under each condition:

| Variant | Throughput | vs Baseline |
|---------|-----------|-------------|
| A. Raw sequential pread (baseline) | 6.80 GB/s | — |
| B. pread + F_NOCACHE | 5.07 GB/s | -25% |
| C. F_NOCACHE + MADV_FREE cycle | 4.78 GB/s | -30% |
| D. Multi-threaded F_NOCACHE (4 threads) | 6.80 GB/s | 0% |
| E. MT + MADV_FREE (4 threads) | 6.81 GB/s | 0% |
| F. Scattered per-tensor reads | 4.05 GB/s | -40% |

**Critical findings:**
- MADV_FREE page fault overhead is negligible (~6% vs F_NOCACHE alone)
- F_NOCACHE costs 25% on single thread, but **multi-threading fully recovers it**
- The full Hypura I/O path (MT + F_NOCACHE + MADV_FREE) achieves **6.8 GB/s** in isolation
- The 300 MB/s effective bandwidth during Llama 70B inference is NOT caused by the I/O syscall path
- The bottleneck must be in how inference interleaves I/O with compute

---

## Expert-Streaming Mode: IMPLEMENTED (2026-03-17)

### Results: Mixtral 8x7B Q5_K_M on M1 Max 32 GB

| Mode | tok/s | n_gpu_layers | NVMe I/O | Memory footprint |
|------|-------|-------------|----------|-----------------|
| Keep-resident (before) | 0.87 | 24 (8 on CPU) | 0 (all resident) | ~31 GB committed |
| Expert-streaming (after) | **2.19** | **33 (all Metal)** | 23 GB/s via pool | ~5 GB committed |

**2.5x improvement.** 99.5% neuron cache hit rate after warmup. All 33 layers on Metal GPU.

### Architecture

- Non-expert tensors (attention, norms, router): ~1.1 GB on GPU, permanently resident
- Expert tensors: 29.8 GB on NVMe, streamed through 2.3 GB pool buffer (6 slots × 385 MB)
- Pool buffer: small mmap'd region. Loading buffer (29.8 GB) munmap'd after tensor->data rewriting
- `use_mmap=false`: prevents Metal from creating a single 30.9 GB MTLBuffer for the model file
- `n_batch=1`: limits Metal compute buffer for MoE intermediates (prompt eval is token-by-token)

### Key Learnings

1. **The bottleneck was CPU backend layers, not I/O.** Keep-resident mode had 8 layers on CPU backend (slow matmuls). Expert-streaming puts all layers on Metal by only keeping 1.1 GB of non-expert tensors in GPU.
2. **Pool buffer avoids Metal OOM.** On unified memory, any virtual mapping counts against Metal's address space. The 29.8 GB loading buffer is released after pool activation; Metal only sees the 2.3 GB pool.
3. **Neuron cache is highly effective for MoE.** Mixtral top-2-of-8 with 32 layers: after ~2 tokens, the cache is warm and 99.5% of expert loads are hits. Steady-state NVMe I/O is near-zero.

---

## Dense FFN-Streaming Mode: IMPLEMENTED (2026-03-17)

### Change 13 Implementation

Extends the MoE expert-streaming pool buffer approach to dense models (Llama 70B).
FFN tensors (gate, up, down — ~57% of model) stream through the pool while
attention + norms stay GPU-resident.

#### Files modified:
- `src/scheduler/types.rs` — added `InferenceMode::DenseFfnStreaming`
- `src/scheduler/placement.rs` — added `try_dense_ffn_streaming_assign()`: routes FFN to NVMe, non-FFN to GPU/RAM, sizes pool at 6 slots (2 layers × 3 tensors)
- `src/compute/nvme_backend.rs`:
  - `DenseFfnLayout` struct for per-layer FFN tensor metadata
  - `ensure_dense_ffn_loaded()` — blocking load of FFN tensors into pool slots
  - `prefetch_dense_ffn()` — non-blocking prefetch of next layer's FFN
  - `rewrite_dense_ffn_ptrs()` — repoint tensor->data to pool slot offsets
  - `eval_callback_dense_ffn_streaming()` — load current layer FFN, release previous, prefetch next
  - `activate_dense_ffn_pool()` — allocate pool, rewrite ptrs, release loading buffer
  - `enable_dense_ffn_scratch()` — **OOM fix** (see below)
  - `on_tensor_init_cb` extended to capture FFN tensor pointers
  - `build_prefetch_state` populates `dense_ffn_layouts` from GGUF
- `src/compute/inference.rs`:
  - `gpu_layers_from_placement()` excludes FFN tensors from GPU budget (like MoE experts)
  - `generate_with_nvme_scheduling()` activates dense FFN pool, sets `dense_ffn_streaming` flag
  - `use_mmap=false` for DenseFfnStreaming (same reason as MoE: prevent Metal from wrapping entire file as one MTLBuffer)

#### OOM Crash Fix (loading scratch buffer)

**Problem:** With `use_mmap=false`, llama.cpp reads tensor data via `fread` directly into
`tensor->data`. For Llama 70B, this commits ~22.6 GB of anonymous mmap pages for FFN tensors
during model loading. Combined with ~17 GB non-FFN on GPU + Metal overhead + macOS, this
exceeds 32 GB and crashes the machine (OOM kill / system freeze).

MoE expert-streaming doesn't hit this because fused expert tensors have a different size ratio —
the loading buffer is smaller relative to available memory.

**Fix:** `enable_dense_ffn_scratch()` allocates a small scratch buffer (~280 MB = largest single
FFN tensor) via mmap before model loading. In `on_tensor_init_cb`, after capturing the tensor
pointer and file offset, FFN tensor `->data` is redirected to the scratch buffer. llama.cpp's
`fread` writes to the scratch (overwriting each tensor sequentially — data is discarded since
we pread from file later). The 22.6 GB anonymous mmap stays with near-zero committed pages.

After model loading, `activate_dense_ffn_pool()` allocates the real pool (~2.4 GB), rewrites
tensor pointers to pool slots, then releases both the loading buffer (uncommitted) and scratch.

**Memory timeline (Llama 70B Q4_K_M on M1 Max 32 GB):**
- Loading phase: ~17 GB non-FFN (GPU) + 0.3 GB scratch + ~0 GB loading buffer (uncommitted) ≈ 17.3 GB
- Inference phase: ~17 GB non-FFN (GPU) + 2.4 GB pool ≈ 19.4 GB
- Headroom: ~12 GB for Metal compute buffers, KV cache, macOS

#### Bug Fix: Missing override patterns for DenseFfnStreaming (2026-03-17)

**Root cause of OOM crashes:** `build_override_patterns()` only had a special branch for
`ExpertStreaming`. `DenseFfnStreaming` fell through to "standard mode" which uses
`n_gpu_layers - 1` as the first non-GPU layer. But `gpu_layers_from_placement()` excludes
FFN tensors from GPU budget, so ALL layers appeared GPU-resident. Result: zero override
patterns generated, ALL 39.6 GB of tensors went to Metal's default buffer type → OOM.

The scratch fix from the previous session was correct but never fired because the FFN
tensors never reached the custom buffer type's callbacks (`on_tensor_loaded_cb`,
`on_tensor_init_cb`).

**Fix:** Extended `build_override_patterns()` to handle both `ExpertStreaming` and
`DenseFfnStreaming` in the same branch — override only tensors explicitly assigned to
NVMe (FFN tensors) to use the custom buffer type. Non-FFN tensors stay on Metal.

#### Status: VALIDATED (2026-03-17)

### Results: Llama 3.3 70B Q4_K_M on M1 Max 32 GB

| Mode | tok/s | n_gpu_layers | NVMe I/O | Memory footprint |
|------|-------|-------------|----------|-----------------|
| FullStreaming (before) | 0.03 | 47 (34 on CPU) | ~300 MB/s effective | ~31 GB committed |
| DenseFfnStreaming (after) | **0.20** | **81 (all Metal)** | 6.6-9.9 GB/s per layer | ~10 GB committed |

**6.7x improvement.** All 80 layers on Metal GPU. No OOM.

#### Per-token trace

| Token | Decode time | Stalls | Stall time | Hits |
|-------|------------|--------|------------|------|
| 1 (cold) | 6696ms | 157 | 6643ms | 2883 |
| 2 | 6689ms | 79 | 4272ms | 1441 |
| 3 | 6500ms | 79 | 4065ms | 1441 |

#### Analysis

- **Bottleneck is I/O stalls, not compute.** Each layer stalls ~50ms waiting for FFN data
  (396-457 MB at 6.6-9.9 GB/s). 80 layers × 50ms = ~4s I/O per token.
- **Token 1 is 2x worse** because the pool is cold — all 80 layers stall (157 stalls).
  Tokens 2-3 benefit from +1 layer prefetch (79 stalls = 80 layers minus 1 prefetched).
- **No neuron cache benefit** for dense models — unlike MoE where only 2/8 experts fire,
  dense FFN loads all 3 tensors every layer, every token. Cache hit rate is meaningless here.
- **I/O bandwidth is near-peak** (6.6-9.9 GB/s per layer vs 6.8 GB/s raw benchmark).
  The bottleneck is structural: I/O cannot be hidden behind compute because each layer's
  I/O time (50ms) >> compute time (~30ms per layer on Metal).

#### Optimization opportunities

1. **Deeper prefetch / double-buffering (Change 4/15):** Prefetch 2+ layers ahead so I/O
   for layer N+2 overlaps with compute of layer N. Currently only layer N+1 is prefetched.
   With 80 layers of ~450 MB each, 2 layers of double-buffering (~900 MB) would hide most
   I/O behind compute. Expected improvement: ~1.5-2x (0.3-0.4 tok/s).
2. **Warm pool on startup:** Pre-load layer 0's FFN tensors before prompt eval to eliminate
   the cold-start penalty on token 1 (saves ~2.4s on first token).
3. **n_batch tuning (Change 11):** Prompt eval at 1.3 tok/s with n_batch=512 is reasonable
   but may be improvable.

---

## Deeper Prefetch for Dense FFN-Streaming: IMPLEMENTED (2026-03-17)

### Change 16: 3-layer prefetch lookahead + initial warm-up

**Problem:** With +1 layer prefetch, every layer stalls ~20ms because I/O time (50ms) >
compute time (30ms). The I/O pipe goes idle between prefetch submissions.

**Theoretical ceiling:** With fully pipelined I/O, throughput is limited by the slower
pipe: `max(total_compute, total_io)` = `max(2.4s, 4.1s)` = 4.1s → ~0.25 tok/s.

#### Changes made:

- `src/compute/nvme_backend.rs`:
  - `activate_dense_ffn_pool()`: pool increased from 6 to 12 slots (4 layers × 3 tensors)
  - `eval_callback_dense_ffn_streaming()`: after compute, prefetch +1/+2/+3 layers ahead
    instead of just +1. Checks status to avoid duplicate submissions.
  - `prefetch_dense_ffn()`: made `pub` for initial warm-up from inference.rs

- `src/compute/inference.rs`:
  - Before prompt eval, pre-load layers 0-3 FFN data into the pool so the first
    eval_callback doesn't stall on a cold pool.

#### Pool slot math:

With 12 slots (4 layers × 3 FFN tensors):
- Layer N computing: 3 slots
- Layer N+1 prefetched (ready): 3 slots
- Layer N+2 prefetching: 3 slots
- Layer N-1 just released: 3 free slots → used for N+3

Each step releases 1 layer (3 slots) and submits 1 new prefetch (3 slots). Balanced.

Pool memory: 12 × 192.7 MB = 2.3 GB (was 1.16 GB). Still within the ~12 GB headroom.

#### Status: VALIDATED (2026-03-17)

### Results: Deeper prefetch on Llama 3.3 70B Q4_K_M

| Metric | +1 prefetch (before) | +3 prefetch (after) | Change |
|--------|---------------------|---------------------|--------|
| **tok/s** | 0.20 | **0.30** | **+50%** |
| Token 1 stalls | 157 (6643ms) | 72 (2540ms) | -54% |
| Token 2 stalls | 79 (4272ms) | 29 (946ms) | -78% |
| Token 3 stalls | 79 (4065ms) | 60 (2328ms) | -43% |
| Token 2 decode | 6689ms | 2853ms | -57% |
| Pool size | 1.16 GB (6 slots) | 2.31 GB (12 slots) | +1.15 GB |

**Token 2 is the standout:** only 29 stalls (946ms I/O wait), decode dropped from 6.7s to
2.9s. The deeper prefetch successfully hides most I/O behind compute for this token.

**Token 3 regressed vs token 2:** 60 stalls (2328ms), 4.7s decode. The per-layer trace
shows a bimodal pattern: some layers stall 3-15ms (well-prefetched) while others stall
50-70ms (pipe fell behind). This suggests the I/O pipe is getting disrupted at the token
boundary — possibly because `prefetch_all_nvme()` or the token-boundary callback reset
interferes with the in-flight prefetch pipeline.

#### Root cause of token 3 regression: I/O pipe idle at token boundary

**Diagnosis:** At the end of each token's forward pass, layer 79's ask=false callback
tries to prefetch layers 80-82 which don't exist → the I/O pipe goes idle. During the
sampling/emission gap (~10-20ms), no I/O is submitted. When the next token starts, layer
0's ask=true blocks on a cold load, and the prefetch pipeline needs several layers to
refill. Token 2 was fast because the pipeline was warm from prompt eval → token 1 overlap.

**Fix applied:** Added between-token prefetch priming in the generation loop
(`src/compute/inference.rs`). Before each `ctx.decode()`, pre-submit prefetches for
layers 0-3 (same as initial warm-up). This keeps the I/O pipe busy through the token
boundary gap so early layers of the next token are already loaded or loading.

**Status: READY FOR TESTING**

Test command: `cargo run --release -- bench --max-tokens 3 --context 512 ./test-models/llama-3.3-70b-q4_k_m.gguf`

**Result:** Token 3 improved (53 stalls/4308ms vs 60/4731ms before) but didn't match
token 2 (29 stalls/2984ms). Overall tok/s stays at 0.3. The token boundary priming helps
but the gap is likely NVMe controller variance after sustained sequential I/O — the first
layers of each new token read from low file offsets after the controller was reading from
high offsets, causing a seek penalty even on NVMe.

**Memory headroom observation:** Dense FFN-streaming uses only ~10 GB committed (7.8 GB
non-FFN + 2.3 GB pool). The 22 GB loading buffer stays uncommitted (scratch fix). This
leaves ~20 GB headroom vs ~1 GB in the old full-streaming mode. This headroom could be
used for larger pool (more prefetch depth), partial layer residency, or larger KV cache.

---

## Hybrid Residency for Dense FFN-Streaming: IN PROGRESS (2026-03-17)

### Change 17: Keep first N layers' FFN permanently resident

**Problem:** With all 80 layers streaming from NVMe, the I/O pipe (50ms/layer) is slower
than compute (24ms/layer). Even with 3-layer prefetch, the pipe falls behind. With ~20 GB
of memory headroom, we can pin some layers' FFN data in RAM permanently.

**Approach:** Allocate a separate mmap buffer for the first N layers' FFN data (N determined
by available memory). These layers need zero I/O during inference, giving the I/O pipe a
~N×24ms head start before streaming layers begin. Transparent within `DenseFfnStreaming` —
the `resident_ffn_layers` set being non-empty is the runtime flag.

#### Changes:

- `src/compute/nvme_backend.rs`:
  - `PrefetchState`: add `resident_ffn_layers`, `resident_ffn_base`, `resident_ffn_offsets`,
    `resident_ffn_size` fields
  - `allocate_resident_ffn_buffer()`: compute which layers fit, mmap resident buffer,
    build tensor-name→offset map
  - `load_resident_ffn_data()`: pread resident layers' FFN data using I/O pool at startup
  - `rewrite_resident_ffn_ptrs()`: point resident layers' tensor->data to resident buffer
  - `eval_callback_dense_ffn_streaming()`: skip pool allocation and I/O for resident layers,
    skip release for resident layers, skip prefetch for resident layers
  - Drop impl: munmap resident buffer

- `src/compute/inference.rs`:
  - Compute resident budget: total_ram - committed - 4 GB headroom
  - Call allocate + load after pool activation
  - Adjust between-token prefetch to skip resident layers

#### Memory budget (Llama 70B Q4_K_M on M1 Max 32 GB):

- Non-FFN on GPU: ~7.8 GB
- Pool buffer: ~2.3 GB
- Runtime overhead: ~2.5 GB (KV cache + Metal compute)
- Safety headroom: 4 GB
- Available for resident: 32 - 7.8 - 2.3 - 2.5 - 4 = ~15.4 GB
- Per-layer FFN: ~427 MB → max ~36 resident layers
- Cap at half (40 layers) → 36 layers resident, 44 streaming
- Resident data: ~15.4 GB, loaded at startup in ~2.3s

#### Expected performance:

- Resident compute runway: 36 layers × 24ms = 864ms
- During that time, I/O pipe preloads: 864ms / 50ms ≈ 17 streaming layers
- Only 44 - 17 = 27 streaming layers could stall
- Expected: ~0.4-0.6 tok/s (vs 0.3 currently)

#### Status: IMPLEMENTED, MARGINAL IMPROVEMENT ON M1 MAX 32 GB

Test command: `cargo run --release -- bench --max-tokens 3 --context 512 ./test-models/llama-3.3-70b-q4_k_m.gguf`

#### Test Results

| Resident layers | Resident size | Token 2 stalls | Token 2 decode | tok/s | Problem |
|----------------|--------------|---------------|---------------|-------|---------|
| 0 (baseline) | 0 | 29 | 2984ms | 0.30 | — |
| 10 | 4.6 GB | 33 | 3275ms | 0.30 | Same perf, no improvement |
| 27 | 11.6 GB | 13 | 7940ms | 0.10 | Memory pressure, Metal 3x slower |
| 40 | 17.1 GB | 28 | 16395ms | 0.10 | Severe memory pressure |

**Conclusion:** On M1 Max 32 GB, hybrid residency is a wash. Each resident layer saves
~24ms of I/O overlap opportunity but consumes ~427 MB. More than ~10 layers causes
enough memory pressure to slow Metal compute beyond the I/O savings. The code is correct
and would help on machines with >48 GB (where 20+ layers could be resident without
pressure), but is not beneficial at 32 GB.

**Key learning:** The `gpu_committed_estimate = gpu_bytes * 60%` heuristic is wrong for
`use_mmap=false` (dense FFN streaming). GPU tensors are 100% committed. KV cache
(~2.6 GB at context 512) and Metal compute buffers are not accounted for in the simple
`runtime_overhead = 2.5 GB` estimate. Actual overhead: ~5+ GB. The resident budget
calculation now uses `gpu_bytes` directly and 8 GB safety headroom, capped at 25% of layers.

---

## Next Optimizations (post expert-streaming)

### Change 11: Increase n_batch for Expert-Streaming Prompt Eval

**Status: HIGHEST PRIORITY — quick win**

**Problem:** Expert-streaming uses `n_batch=1` to avoid Metal compute buffer OOM from MoE intermediates. This makes prompt eval very slow (4.5s for 9 tokens — each token processed individually). Generation is unaffected (already n_batch=1).

**Opportunity:** n_batch=1 was chosen to avoid OOM at n_batch=512. There's a large gap between 1 and 512. Testing n_batch=16, 32, 64 would find the sweet spot that speeds up prompt eval without OOM.

**File:** `src/compute/inference.rs`

**Expected impact:** Prompt eval speedup of 5-15x (from 4.5s to ~0.3-0.9s). No effect on generation tok/s. Directly improves time-to-first-token (TTFT).

### Change 12: Warm Neuron Cache from Co-Activation Data

**Problem:** Token 1 has 64 stalls (752ms) because the neuron cache is cold. The co-activation persistence file (`~/.hypura/coactivation/{model}.json`) already records which experts fire most frequently at each layer.

**What to change:** On startup in expert-streaming mode, pre-load the top-2 most frequently activated experts per layer from co-activation data. This fills the neuron cache before the first token.

**File:** `src/compute/nvme_backend.rs` (add `warm_cache_from_coactivation` method)

**Expected impact:** Eliminate most first-token stalls. Token 1 goes from 752ms stall to near-zero.

### Change 13: Expert-Streaming for Dense Models (Llama 70B)

**Problem:** Llama 70B on 32 GB: 0.03 tok/s with 34 layers on CPU backend. The I/O trace showed 30-40s decode times — same root cause as Mixtral before expert-streaming (CPU backend layers).

**Opportunity:** The same pool buffer approach can work for dense models. Treat FFN tensors (gate, up, down weights — ~60% of per-layer size) as "expert-like" and stream them through the pool, keeping attention tensors on GPU.

**What to change:** Extend `try_expert_streaming_assign` to detect dense models where non-FFN tensors fit in GPU but full layers don't. Route FFN weights to NVMe, attention/norms to GPU. The eval_callback loads FFN data into pool slots before each layer's FFN computation.

**Complexity:** Higher than MoE expert-streaming because dense FFN tensors are not fused — each layer has 3 separate weight tensors (gate, up, down) rather than 3 fused expert tensors. The pool slot management and tensor->data rewriting need to handle individual tensors.

**Expected impact:** Similar 2-3x improvement if all layers can move to Metal. May transform Llama 70B from 0.03 tok/s to 0.06-0.10 tok/s.

### Change 14: Tune Pool Slot Count and Neuron Cache Size

**Problem:** Current: 6 pool slots, default neuron cache capacity. The 99.5% hit rate is excellent, suggesting we might be over-provisioned on some dimension and under-provisioned on others.

**Tests to run:**
- 3 slots (minimum: current layer only) — does prefetch still help?
- 9 slots (3 layers) — does deeper prefetch improve hit rate further?
- Neuron cache capacity: current auto-sizing vs explicit 64/128/256 entries

### Change 15: Deeper Speculative Prefetch

**Problem:** The eval_callback currently prefetches 1 MoE layer ahead. With 6 pool slots, we have room for 2 layers of data. Prefetching 2-3 layers ahead would start I/O earlier, hiding latency behind compute.

**What to change:** In `eval_callback_expert_streaming`, increase the prefetch lookahead from `1..4` to `1..6` (or based on available pool slots).

---

## Changes 8-10: SUPERSEDED BY EXPERT-STREAMING

Changes 8 (GPU headroom), 9 (Q4 KV), and 10 (CPU backend measurement) were designed to squeeze more layers onto Metal under keep-resident mode. Expert-streaming achieved the same goal more effectively — all 33 layers on Metal with only 1.1 GB GPU footprint. These changes are no longer needed for Mixtral.

They may still be relevant for dense models that can't use expert-streaming (Change 13 addresses this).

---

## Revised Strategy: Llama 70B Streaming (CPU-backend-bound)

The iobench proves the raw I/O path achieves 6.8 GB/s. The 300 MB/s effective throughput during inference means **~95% of I/O bandwidth is lost to the inference-I/O interaction**, not to the I/O path itself.

### Theories to Test

#### Theory A: eval_callback Lock Contention

**Hypothesis:** The eval_callback acquires `layer_status` mutex on every tensor evaluation (`ensure_layer_loaded`). The I/O pool workers also acquire this mutex when completing tasks. In streaming mode with 80 layers and ~9 tensors per layer, the callback fires ~720 times per token. Each `ensure_layer_loaded` call takes the lock, checks status, and releases it. If the I/O workers are trying to update status simultaneously, the mutex ping-pongs between threads.

**Test:** Add `Instant::now()` timing around `ensure_layer_loaded` and log total time spent waiting on the lock per token. If it's >100ms, contention is significant.

**Fix if confirmed:** Use lock-free `AtomicU8` for layer status instead of `Mutex<HashMap>`. Each layer gets an atomic status that the callback reads without locking and I/O workers update with `store(Release)`.

#### Theory B: Synchronous Layer Loading in eval_callback

**Hypothesis:** In streaming mode, `ensure_layer_loaded` blocks until the layer is fully loaded. If prefetch hasn't finished the next layer by the time compute reaches it, the GPU stalls waiting for the CPU thread to complete I/O. The iobench shows 6.8 GB/s is possible, but only if I/O runs continuously. If I/O pauses while compute runs (because the I/O pool is idle waiting for a new layer request), the effective bandwidth drops to `layer_bytes / (io_time + idle_time)`.

**Test:** Log timestamps of `submit_layer_load` and `LayerStatus::Loaded` transitions. Compute the gap between "layer N loaded" and "layer N+1 load submitted". If there's dead time between loads, the prefetch lookahead is insufficient.

**Fix if confirmed:** Increase `adaptive_lookahead` minimum. Currently clamped to 2-8 layers. For Llama 70B with 15 NVMe layers, the I/O pool should always have 3-4 layers queued to maintain continuous NVMe utilization.

#### Theory C: Metal GPU Blocking the CPU

**Hypothesis:** On unified memory, `ctx.decode()` submits Metal commands and may block the CPU thread until the GPU finishes. While the CPU is blocked in the Metal driver, it cannot run the eval_callback. The I/O pool threads can still run (they're separate threads), but no new prefetch requests are submitted until the CPU returns from `decode()`. This creates a serial pattern: GPU compute → CPU resumes → eval_callback fires → I/O starts → I/O completes → GPU compute.

**Test:** Measure wall time of `ctx.decode()` vs actual GPU compute time (from `llama_perf_context`). If decode wall time >> GPU compute time, the CPU is stalled in the Metal driver.

**Fix if confirmed:** Move eval_callback processing off the main thread. Submit prefetch requests from a dedicated coordinator thread that doesn't block on Metal.

#### Theory D: MADV_FREE + Metal Memory Pressure

**Hypothesis:** Metal's working set and the posix_memalign buffer compete for physical memory. When `release_layer` calls MADV_FREE, the kernel may not reclaim those pages immediately. But when `pread` writes to new layer pages, the kernel must find free physical pages. If Metal has pinned a large working set, the kernel has to evict something — possibly Metal's own pages, causing GPU stalls on next compute. The iobench doesn't reproduce this because it runs I/O in isolation without Metal.

**Test:** Run iobench variant E while simultaneously running a Metal compute workload (e.g., a simple matmul kernel in a loop). If throughput drops significantly, memory pressure from Metal is the cause.

**Alternative test:** Replace `MADV_FREE` with `MADV_DONTNEED` (immediately discards pages rather than lazily marking them reclaimable). If this changes streaming throughput, the lazy reclaim is interacting with Metal's memory management.

#### Theory E: Prefetch Timing Mismatch

**Hypothesis:** The current prefetch logic submits layer N+2 through N+lookahead when layer N finishes computing. But with 4 I/O workers and large layers (~600 MB each), the workers take ~100ms per layer at 6 GB/s. If compute takes 50ms per layer but the next I/O takes 100ms, there's always a 50ms stall. The lookahead should be deep enough that the I/O pipeline never drains.

**Test:** Add tracing to log per-layer I/O start/complete timestamps alongside compute start/complete. Build a timeline to visualize whether I/O and compute are actually overlapping or serialized.

**Fix if confirmed:** Pre-submit ALL NVMe layers at token start (like `prefetch_all_nvme` does for the first token), not lazily via eval_callback. For streaming mode, the layer sequence is deterministic — there's no reason to wait for compute to trigger prefetch.

---

## Implementation Priority (Revised 2026-03-17)

### Immediate (high confidence, directly measurable):
1. **Deeper prefetch for dense FFN-streaming** — currently prefetches +1 layer. Increasing to +2-3 layers would overlap I/O with compute. Each layer is ~450 MB, so 2 extra layers = ~900 MB more pool buffer. The 80-layer × 50ms stall pattern shows I/O is never hidden. Expected: 0.3-0.4 tok/s.
2. **Change 11** (n_batch tuning) — find optimal batch size for expert-streaming prompt eval. Quick binary search, directly measurable.
3. **Change 12** (warm neuron cache) — pre-load frequently-activated experts on startup. Infrastructure exists.

### Medium-term (moderate effort):
4. **Change 15** (deeper prefetch for expert-streaming) — same idea, applied to MoE models.

### Low priority / deferred:
5. **Change 14** (pool/cache tuning) — measure once other changes stabilize
6. **Theories A-E** (Llama 70B streaming) — may be superseded by Change 13
7. **Changes 2-4** (I/O coalescing, async, double buffer) — disproven or superseded
8. **Changes 8-10** (GPU headroom) — superseded by expert-streaming for MoE models

### Status of all changes:
- **Change 1** (LP objective) — DONE
- **Change 2** (I/O coalescing) — DISPROVEN
- **Change 3** (async I/O) — SUPERSEDED
- **Change 4** (double buffering) — SUPERSEDED
- **Change 5** (MoE expert loading) — DONE (integrated into expert-streaming)
- **Change 6** (cache heuristic) — deferred (LRU is sufficient at 99.5% hit rate)
- **Change 7** (layer skip) — deferred
- **Changes 8-10** — SUPERSEDED by expert-streaming
- **Change 11** (n_batch tuning) — DONE (n_batch=1 cap removed in earlier session, Mixtral prompt eval 4.5s → 1.2s)

## Sparse MoE Mmap Mode: IMPLEMENTING (2026-03-22)

### Change 18: OS page cache for ultra-sparse MoE models

**Problem:** Expert-streaming adds overhead (eval callback, pool slots, tensor pointer
rewriting) on every layer. For ultra-sparse MoE models (e.g. Qwen3-Coder-Next, 512
experts, 10 active = 2%), the active working set (~900 MB) fits easily in RAM. The OS
mmap page cache handles sparsity natively — only active expert pages get loaded.
CPU-only llama.cpp (ngl=0) achieved 3.4 tok/s on this model vs Hypura's 1.2 tok/s
because the pool overhead dominated.

**Fix:** Detect ultra-sparse MoE models at placement time. When `activation_ratio < 15%`
AND `active_bytes < 30% of unified memory` AND the full model fits in GPU working set,
use `SparseMoeMmap` mode: all tensors on GPU via `use_mmap=true`, no custom buffer, no
eval callback. Falls through to `generate_blocking` — identical to the "fits in memory"
path.

**Decision threshold:**
- Mixtral (2/8 = 25%): too dense → expert-streaming (pool + callback)
- Qwen3-Coder-Next (10/512 = 2%): sparse → mmap (OS page cache)

#### Status: READY FOR TESTING

Test: `cargo run --release -- bench --max-tokens 30 --context 512 ./test-models/Qwen3-Coder-Next-Q4_K_M.gguf`

Expected: ~3-4 tok/s (matching llama.cpp CPU-only, but with GPU acceleration via Metal
mmap shared buffers).

**If crash:** The model is 45.2 GB. Metal `recommendedMaxWorkingSetSize` on M1 Max 32GB
is ~26.8 GB. If Metal OOMs on the full mmap, the `total_bytes <= caps.gpu_bytes` check
in `try_sparse_moe_mmap` should have prevented this mode. Check `caps.gpu_bytes`.

**Pending benchmarks (2026-03-21):**

1. **Qwen3-Coder-Next 80B-A3B Q4_K_M** (~45.2 GB) — MoE with 512 experts, 10 active
   (2% activation ratio). Should trigger expert-streaming. Extreme sparsity should give
   very high neuron cache hit rate. Key test of expert-streaming on high-expert-count models.
   `cargo run --release -- bench --max-tokens 10 --context 512 ./test-models/Qwen3-Coder-Next-Q4_K_M.gguf`

2. **Qwen 2.5 Coder 32B Q8_0** (~32.4 GB) — Dense, most popular coding model at max
   quality quant. Q8 exceeds 32 GB RAM. Should trigger dense FFN-streaming.
   `cargo run --release -- bench --max-tokens 10 --context 512 ./test-models/qwen2.5-coder-32b-instruct-q8_0.gguf`

   **If either crashes, start with `--max-tokens 3` and verify placement/mode selection first.**
- **Change 12** (warm cache) — planned
- **Change 13** (dense FFN-streaming) — DONE: 0.20 tok/s (6.7x over FullStreaming 0.03). Bottleneck is per-layer I/O stalls (~50ms × 80 layers)
- **Change 14** (pool/cache tuning) — DONE: unified MemoryBudget, dynamic pool slots, scaled prefetch lookahead
- **Change 15** (deeper prefetch) — planned

---

## Validation Plan

After each change, run:
```sh
# Mixtral (keep-resident mode)
cargo run --release -- bench --max-tokens 10 --context 2048 ./test-models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf

# Llama 70B (streaming mode)
cargo run --release -- bench --max-tokens 3 --context 512 ./test-models/llama-3.3-70b-q4_k_m.gguf

# I/O diagnostic (after any I/O path change)
cargo run --release -- iobench ./test-models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf
```

Current baselines (2026-03-17):
- Mixtral expert-streaming: **2.19 tok/s** (was 0.87 keep-resident)
- Mixtral prompt eval (n_batch=1): 4.5s for 9 tokens (target: <1s with n_batch tuning)
- Llama 70B dense-FFN-streaming: **0.30 tok/s** (was 0.03 FullStreaming — **10x improvement**)
- Llama 70B per-token decode: ~2.9s best (token 2), ~4.7s worst (token 3, pipeline disruption)
- Raw I/O (MT + F_NOCACHE + MADV_FREE): 6.8 GB/s
- Neuron cache hit rate: 99.5% (expert-streaming Mixtral)

Save results to `benchmarks/results/` for tracking.

---

## Key Constants to Tune

| Constant | Current | Suggested | Source |
|----------|---------|-----------|--------|
| GPU_RUNTIME_OVERHEAD | 2 GB | Test 1.5 GB, 1.0 GB | Empirical (Change 8) |
| KV cache GPU budget | 20% of GPU | Reduce if Q4 KV | Empirical (Change 9) |
| MoE cache hit rate | 0.965 | Measure empirically | PowerInfer-2 |
| Cache recency weight | N/A | 0.5 | FlashMoE |
| Cache frequency weight | N/A | 0.5 | FlashMoE |
| Layer skip threshold | N/A | 0.98 | ntransformer |
| Prefetch lookahead min | 2 | 4+ for heavy streaming | Theory E |
| adaptive_lookahead clamp | 2-8 | 4-12 for 70B class | Theory B |

---

## Reference Papers

- **FlexGen:** Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU" (ICML 2023). LP formulation, zig-zag scheduling, 4-bit KV cache.
- **PIPO:** "Pipelined Offloading for Efficient Inference on Consumer Devices" (arXiv 2504.03664, April 2025). Fine-grained pipeline, NVMe block sizing, 3.1x over FlexGen.
- **PowerInfer:** Xue et al., "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU" (SOSP 2024). Hot/cold neuron profiling, ILP placement, sparse operators.
- **PowerInfer-2:** "PowerInfer-2: Fast Large Language Model Inference on a Smartphone" (arXiv 2406.06282, June 2024). Flash I/O, neuron cluster prefetch, two-phase loading.
- **ntransformer:** github.com/xaskasdf/ntransformer. 3-tier adaptive caching, SLEP double-buffer, layer skip, 83x over mmap.
- **CHEOPS:** Ren et al., "An I/O Characterizing Study of Offloading LLM Models and KV Caches to NVMe SSD" (EuroSys Workshop 2025). 128 KiB read dominance, libaio 2.9-5.5x over POSIX.
- **FlashMoE:** "FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices" (arXiv 2601.17063). LRU deficiency, recency+frequency complementarity, +21% hit rate.
