```
 _   _
| | | |_   _ _ __  _   _ _ __ __ _
| |_| | | | | '_ \| | | | '__/ _` |
|  _  | |_| | |_) | |_| | | | (_| |
|_| |_|\__, | .__/ \__,_|_|  \__,_|
       |___/|_|
   Storage-Tier-Aware LLM Inference
```

# Hypura Benchmarks

## What is Hypura?

Hypura is a storage-tier-aware LLM inference scheduler for Apple Silicon.
It places model tensors across GPU, RAM, and NVMe tiers based on access
patterns, bandwidth costs, and hardware capabilities — enabling models
that exceed physical memory to run without crashing the system.

## Why does this matter?

Consumer hardware (MacBook Pro, Mac Studio) ships with fast unified memory
and NVMe storage, but limited capacity. A 32GB M1 Max cannot naively load
a 40GB model — the OS will swap-thrash until the OOM killer intervenes.

Hypura solves this by understanding the model architecture:

- **Norms and embeddings** are tiny but accessed every token — pinned to GPU
- **Attention/FFN weights** for early layers stay in GPU/RAM for low-latency compute
- **Overflow layers** stream from NVMe with lookahead prefetch (read layer N+1 while N computes)
- **MoE expert routing** exploits sparsity — only 2 of 8 experts fire per token.
  Router interception reads `ffn_moe_argsort` output in the eval callback to
  identify selected experts, then loads only 2/8 expert strides from NVMe (75%
  I/O reduction). A neuron cache (LRU) tracks loaded expert slices across tokens,
  achieving ~96% hit rate from temporal locality. Speculative prefetch loads the
  same experts for the next layer based on cross-layer correlation.

The result: models that would crash your machine under naive mmap become runnable.
Models that fit in memory run at full Metal GPU speed with zero overhead.

## Hardware

All benchmarks run on:

- **CPU/GPU:** Apple M1 Max (10-core, 32-core GPU)
- **Memory:** 32 GB unified (LPDDR5, ~400 GB/s bandwidth)
- **Storage:** Apple NVMe (~5.1 GB/s sequential read)

## Results

| Date | Model | Params | Quant | Size | Mode | GPU | NVMe | Hypura tok/s | Baseline | Notes |
|------|-------|--------|-------|------|------|-----|------|--------------|----------|-------|
| 2026-03-13 | TinyLlama 1.1B | 1.1B | Q4_K_M | 0.6 GB | full-resident | 0.6 GB | — | 71.8 | — | Fits in GPU |
| 2026-03-14 | Qwen 2.5 14B | 14.8B | Q4_K_M | 8.4 GB | full-resident | 8.4 GB | — | 24.0 | — | Fits in GPU |
| 2026-03-17 | Mixtral 8x7B | 46.7B | Q5_K_M | 30.9 GB | expert-streaming | 1.1 GB | 29.8 GB | **2.2** | OOM | 99.5% neuron cache hit rate |
| 2026-03-17 | Llama 3.3 70B | 70.6B | Q4_K_M | 39.6 GB | dense-FFN-streaming | 7.8 GB | 31.8 GB | **0.2** | OOM | All layers on Metal, I/O-bound |

### Key observations

- **Fits in GPU:** TinyLlama and Qwen 14B run entirely on Metal at full speed.
  Hypura adds no overhead when tiering isn't needed.
- **MoE overflow (Mixtral):** 31 GB model on 32 GB machine. Expert-streaming puts
  only non-expert tensors (~1.1 GB) on GPU and streams expert data through a 2.3 GB
  pool buffer. Neuron cache achieves 99.5% hit rate — steady-state NVMe I/O is
  near-zero. **2.2 tok/s** where vanilla llama.cpp OOMs.
- **Dense overflow (Llama 70B):** 40 GB model. Dense FFN-streaming keeps attention +
  norms (~7.8 GB) on GPU and streams FFN tensors (~31.8 GB) from NVMe. All 80 layers
  run on Metal. **0.2 tok/s** (6.7x over previous full-streaming at 0.03 tok/s).
  Bottleneck is per-layer I/O stalls (~50ms × 80 layers per token).

## Vanilla llama.cpp vs Hypura: Mixtral 8x7B on M1 Max 32GB

Mixtral 8x7B Q5_K_M (30.9 GB) cannot run on a 32 GB M1 Max under vanilla
llama.cpp — at all. Hypura runs it at 2.2 tok/s with all layers on Metal GPU.

### Why vanilla llama.cpp fails

Vanilla llama.cpp memory-maps the entire GGUF file (~31 GB) as a single
allocation. On Apple Silicon, Metal creates a shared buffer from this mmap'd
region and tracks the full size against the GPU working set — even if only a
fraction of layers are offloaded to the GPU.

The M1 Max 32GB reports `recommendedMaxWorkingSetSize` = 26.8 GB. With a 31 GB
mmap'd buffer, Metal immediately exceeds this limit the moment it tries to
execute any GPU compute, regardless of `n_gpu_layers`:

| Setting | Result |
|---------|--------|
| `ngl=24` (recommended for Mixtral) | `kIOGPUCommandBufferCallbackErrorOutOfMemory` |
| `ngl=16` | `kIOGPUCommandBufferCallbackErrorOutOfMemory` |
| `ngl=8` | `kIOGPUCommandBufferCallbackErrorOutOfMemory` |
| `ngl=0` (CPU only) | No OOM, but >10 min for a single token (swap thrash) |
| `ngl=0 --no-mmap` | Hangs — malloc(31 GB) on 32 GB causes swap death |

Tested with llama.cpp build 8329 (1d3da8b8a), `-c 512 -n 10`.

### How Hypura solves it: expert-streaming

Hypura exploits Mixtral's MoE sparsity. Only 2 of 8 experts fire per token —
there's no reason to keep all expert tensors in GPU memory.

- **Non-expert tensors (~1.1 GB):** Attention, norms, router weights stay on Metal GPU
- **Expert tensors (~29.8 GB):** Stored on NVMe, streamed through a 2.3 GB pool buffer
  (6 slots) via the eval callback
- **Neuron cache:** LRU cache tracks loaded expert slices. After ~2 tokens of warmup,
  achieves 99.5% hit rate — steady-state NVMe I/O is near-zero
- **`use_mmap=false`:** Prevents Metal from wrapping the entire model file as one
  MTLBuffer. Non-expert tensors get individual Metal shared buffers; expert tensors
  go to our custom GGML buffer type

### Results

| | Vanilla llama.cpp | Hypura |
|---|---|---|
| **GPU offload** | OOM (any ngl > 0) | All 33 layers on Metal |
| **Generation** | N/A (crash) | **2.2 tok/s** |
| **GPU memory** | 31 GB (full mmap) | ~1.1 GB (non-expert) + 2.3 GB pool |
| **Cache hit rate** | N/A | 99.5% |

## Llama 3.3 70B on M1 Max 32GB

40 GB dense model — every layer needs FFN tensors loaded, every token.

### Dense FFN-streaming

For dense (non-MoE) models, Hypura splits each layer's tensors by role:

- **Attention + norms (~7.8 GB):** Permanently on Metal GPU
- **FFN gate/up/down (~31.8 GB):** On NVMe, streamed through a ~2.4 GB pool buffer
  (6 slots = 2 layers × 3 FFN tensors)
- **All 80 layers on Metal:** `n_gpu_layers=81`. No CPU backend layers (which were
  the main bottleneck at 0.03 tok/s before)

The eval callback loads each layer's FFN tensors into pool slots before computation,
releases the previous layer's slots, and prefetches the next layer.

### Results

| | Full-streaming (before) | Dense FFN-streaming (after) |
|---|---|---|
| **n_gpu_layers** | 47 (34 on CPU) | 81 (all Metal) |
| **Generation** | 0.03 tok/s | **0.3 tok/s** |
| **Improvement** | — | **10x** |
| **Per-token decode** | ~30s | ~6.5s (4s I/O + 2.5s compute) |
| **Per-layer I/O** | ~300 MB/s effective | 6.6-9.9 GB/s (near peak) |

Bottleneck: each of 80 layers stalls ~50ms waiting for FFN data from NVMe.
Prefetch hides 1 layer but I/O time (50ms) exceeds compute time (~30ms per layer).
Deeper prefetch and double-buffering are planned to further reduce stalls.

## Running benchmarks

```sh
# Hypura only (safe for any model size)
hypura bench ./model.gguf

# With baseline comparison (only safe if model fits in RAM)
hypura bench --baseline ./model.gguf

# Force baseline even for oversized models (may OOM)
hypura bench --baseline --force ./model.gguf

# Custom settings
hypura bench --max-tokens 64 --context 4096 ./model.gguf
```

Results are saved as JSON in `benchmarks/results/`.

## Streaming Architecture

Both expert-streaming and dense FFN-streaming share the same core architecture:

1. **Custom GGML buffer type** (`HypuraBuftController`): Registers tensor overrides
   so llama.cpp routes selected tensors (experts or FFN) to our buffer instead of Metal.
2. **Pool buffer** (`ExpertPool`): Small mmap'd region (2-3 GB) divided into slots.
   Tensors are loaded from NVMe into slots via multi-threaded `pread` with `F_NOCACHE`.
3. **Eval callback**: Fires before/after each layer's computation. Before: loads data
   into pool slots, rewrites `tensor->data` pointers. After: releases previous layer's
   slots, prefetches next layer.
4. **Loading scratch** (dense FFN only): During model loading with `use_mmap=false`,
   redirects FFN tensor `fread` to a ~280 MB scratch buffer to avoid committing ~22 GB
   of anonymous mmap pages. Released after pool activation.

### MoE expert-streaming details

| Feature | Implementation |
|---------|---------------|
| Router interception | Reads `ffn_moe_argsort` tensor to extract selected expert IDs |
| Selective loading | Loads only 2/8 expert strides per layer (75% I/O reduction) |
| Neuron cache | LRU cache tracks loaded expert slices; 99.5% hit rate |
| Co-activation prefetch | Cross-layer correlation matrix predicts next layer's experts |
| Cache warming | Pre-loads top-2 experts per layer from co-activation data on startup |

## Key Technical Learnings

### mmap page management on Apple Silicon

- llama.cpp's `use_mmap=true` provides tensor data via file-backed mmap pages,
  even for tensors routed to custom buffer types (our `posix_memalign` buffer).
  The kernel overlays mmap pages onto the buffer's virtual address range.
- **Do not pread into mmap-backed buffers.** This replaces efficient file-backed
  pages (reclaimable by the kernel without I/O) with anonymous dirty pages
  (must be compressed/swapped). This was the root cause of the keep-resident
  regression: pre-loading via pread converted pages and caused 2.5x slowdown.
- For keep-resident mode, rely on mmap for data population. Only use pread for
  streaming mode where explicit lifecycle management (load/release) is needed.

### Eval callback tensor naming

- llama.cpp's `cb_eval` is called for compute graph **nodes** (intermediate result
  tensors), NOT for source weight tensors. Names use `{operation}-{layer}` format
  (e.g., `attn_norm-0`, `ffn_moe_gate-5`, `l_out-31`), not GGUF weight names
  (`blk.0.attn_norm.weight`).
- View/reshape operations append suffixes: `Qcur-0 (reshaped)`, `cache_k_l0 (view)`.
  The `-N` suffix parser must handle these gracefully (parse fails → skip).

### Committed memory estimation

- On Apple Silicon, GPU mmap'd layers commit ~60% of their size (demand paging).
- Keep-resident threshold: `gpu_committed_60% + buffer_bytes + 2.5GB overhead + nvme_bytes < RAM - 4GB`
- This correctly enables keep-resident for Mixtral (26.3 GB < 28 GB) while
  forcing streaming for Llama 70B (33 + 9.8 GB >> 28 GB).
| 2026-03-17 | llama-3.3-70b-q4_k_m Q4K | Apple M1 Max 32GB | 7.8 GB | 0.0 GB | 31.8 GB | — | 0.3 | — |
