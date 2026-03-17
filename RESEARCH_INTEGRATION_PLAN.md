# Research Integration Plan for Hypura

> Temporary working document for implementing techniques from FlexGen, PIPO, PowerInfer/PowerInfer-2, ntransformer, CHEOPS, and FlashMoE into the Hypura codebase.

## Codebase Context

- **Target hardware:** M1 Max, 32GB unified memory, ~5.1 GB/s NVMe sequential read
- **Current perf:** Mixtral 8x7B Q5_K_M keep-resident 1.3 tok/s; Llama 3.3 70B Q4_K_M streaming 0.03 tok/s
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

## Revised Strategy: Mixtral (keep-resident, compute-bound)

Mixtral runs in keep-resident mode: after the first forward pass, all layers are in physical memory and the eval callback is disabled. Generation is **purely GPU-compute-bound**. I/O optimizations have zero effect on generation tok/s.

Current bottleneck: only 24 of 32 layers run on the Metal GPU backend. The remaining 8 layers (6.3 GB RAM + 2.0 GB NVMe) use the CPU backend, which is dramatically slower for matmuls.

### Change 8: Tighten GPU Headroom Reservations

**Problem:** The placement solver reserves 2 GB for `GPU_RUNTIME_OVERHEAD` (compute buffers + Metal overhead) and KV cache headroom based on conservative estimates. If either is overestimated, the freed GPU space could fit 1-2 more layers on Metal — each one replacing a slow CPU-backend layer.

**File:** `src/scheduler/placement.rs`

**What to investigate:**
1. Run Mixtral and measure actual Metal compute buffer usage via `recommendedMaxWorkingSetSize` vs actual working set. If Metal uses only 1.2 GB of the 2 GB reservation, reclaim the difference.
2. Check KV cache sizing: with Q8 KV at context=2048, the KV cache is ~80 MB. The current 20%-of-GPU-budget allocation may over-reserve.
3. Test empirically: try `GPU_RUNTIME_OVERHEAD = 1.5 GB` and `1.0 GB`, run Mixtral, see if it OOMs or gains GPU layers.

**Expected impact:** Each additional GPU layer replaces a CPU-backend layer. Going from 24 → 26 GPU layers could improve tok/s by 10-20% (2 fewer slow layers in the critical path).

### Change 9: Q4 KV Cache

**Problem:** KV cache uses Q8_0 or FP16. Q4_0 uses 0.28x the memory of FP16 (vs 0.53x for Q8_0), freeing GPU space for model layers.

**File:** `src/scheduler/placement.rs` (KV cache plan), `src/compute/inference.rs` (KV quantization selection)

**What to change:** Add Q4_0 as an auto-selection option when GPU budget is very tight. The quality impact of Q4 KV is model-dependent but generally acceptable for short-context inference.

**Expected impact:** On Mixtral at context=2048, KV cache shrinks from ~80 MB (Q8) to ~42 MB (Q4). Marginal GPU savings, but combined with Change 8 could push one more layer onto GPU.

### Change 10: Measure CPU Backend Cost Directly

**Diagnostic, not a code change.** Run Mixtral with `n_gpu_layers=32` (all layers on Metal) to measure the tok/s ceiling if the CPU backend were eliminated entirely. This will OOM or swap on 32 GB, but even a single-token measurement shows the theoretical maximum. Compare against current 24-layer performance to quantify the CPU backend cost per layer.

```sh
# WARNING: may OOM. Use --max-tokens 1 and monitor memory.
cargo run --release -- bench --max-tokens 1 --context 512 --force ./test-models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf
# With manual n_gpu_layers override (if supported)
```

---

## Revised Strategy: Llama 70B Streaming (I/O-bound, 300 MB/s mystery)

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

## Implementation Priority & Dependencies (Revised)

### For Mixtral (compute-bound):
1. **Change 10** (diagnostic) — measure CPU backend cost, zero effort
2. **Change 8** (GPU headroom) — tune constants, low risk, potentially +10-20% tok/s
3. **Change 9** (Q4 KV) — small change, marginal gain on this hardware

### For Llama 70B (streaming, 300 MB/s mystery):
1. **Theory E test** (prefetch timing) — add tracing, identify I/O gaps, highest-probability fix
2. **Theory B test** (synchronous loading) — same tracing, look for dead time
3. **Theory A test** (lock contention) — instrument mutex, easy to measure
4. **Theory D test** (MADV_FREE vs MADV_DONTNEED) — one-line change, quick experiment
5. **Theory C test** (Metal blocking) — harder to measure, investigate last

### Previously planned changes — status:
- **Change 1** (LP objective) — DONE
- **Change 2** (I/O coalescing) — DISPROVEN, skip
- **Change 3** (async I/O) — preadv irrelevant (syscall count not the issue). dispatch_io still possible but speculative.
- **Change 4** (double buffering) — deferred until bandwidth mystery is solved
- **Change 5** (MoE expert loading) — already implemented
- **Change 6** (cache heuristic) — low priority, no impact on current hardware
- **Change 7** (layer skip) — experimental, deferred

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
- Mixtral keep-resident: 0.87 tok/s (target: 2.0+)
- Llama 70B streaming: 0.03 tok/s (target: 0.1+)
- Raw I/O (MT + F_NOCACHE + MADV_FREE): 6.8 GB/s

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
