# Hypura

Storage-tier-aware LLM inference scheduler for Apple Silicon. Places model
tensors across GPU (Metal), RAM, and NVMe based on access patterns and
hardware bandwidth to enable running models larger than available memory.

## Architecture

- **Cargo workspace:** root `hypura` crate + `hypura-sys` (FFI bindings)
- **llama.cpp:** vendored as git submodule at `vendor/llama.cpp/`, built via cmake
- **CLI binary:** `hypura` (not "reach" from the original spec)
- **Library-first:** all logic in `src/lib.rs` modules, thin CLI wrapper in `src/main.rs`

## Key modules

- `src/scheduler/placement.rs` — LP + greedy tensor placement across GPU/RAM/NVMe tiers
- `src/compute/inference.rs` — inference engine, `generate_blocking` (baseline) and `generate_with_nvme_scheduling` (tiered)
- `src/compute/nvme_backend.rs` — multi-threaded I/O pool, custom GGML buffer type, NVMe-tier tensor prefetch
- `src/cache/coactivation.rs` — expert co-activation tracking for MoE speculative prefetch
- `src/cache/kv_cache.rs` — windowed KV cache compaction via `llama_memory_seq_rm`
- `src/cache/neuron_cache.rs` — LRU cache tracking loaded expert slices
- `src/cli/optimize.rs` — GGUF expert layout rewriter (greedy TSP for contiguous co-activated experts)
- `src/profiler/` — hardware detection (CPU, GPU, memory bandwidth, NVMe throughput)
- `src/cli/bench.rs` — A/B benchmark harness (Hypura vs naive mmap baseline)
- `src/model/tensor_role.rs` — tensor classification for scoring (norms, attention, MoE experts, etc.)

## Build & run

```sh
cargo build --release
cargo run --release -- bench ./test-models/model.gguf
cargo run --release -- bench --baseline ./test-models/model.gguf  # A/B comparison
```

## Safety

- `bench --baseline` is hard-blocked when model exceeds RAM - 4GB headroom. Use `--force` to override.
- Always start with `--max-tokens 10` on untested models before scaling up.
- Test models live in `./test-models/` (not checked in).

## Dev hardware

M1 Max, 32GB unified memory, ~5.1 GB/s NVMe sequential read.
GPU budget after KV cache + Metal overhead: ~22-24 GB.
Baseline safety threshold: model must be < 28 GB.

## Commit style

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat(scope): description`, `fix(scope):`, `docs:`, `perf:`, `refactor:`, `test:`, etc.

## Current status (2026-03-17)

- Scaffold, FFI, hardware profiler, placement optimizer, benchmark harness: done
- Multi-threaded I/O pool (IoPool): per-worker F_NOCACHE fds, barrier-based completion, region splitting across workers
- Co-activation tracking: same-layer + cross-layer matrices, persistence to `~/.hypura/coactivation/`, integrated into speculative prefetch
- KV cache tiering: auto Q8 selection when GPU budget is tight, windowed compaction via `llama_memory_seq_rm`
- `hypura optimize` command: greedy TSP expert reordering with sidecar `.permutations.json`
- MoE expert-aware prefetch with neuron cache and router interception: done

### Benchmark results (M1 Max 32GB)

| Model | Mode | tok/s | Notes |
|-------|------|-------|-------|
| Mixtral 8x7B Q5_K_M (30.9 GB) | keep-resident | 1.3 | 60% improvement from IoPool parallel first-pass loading |
| Llama 3.3 70B Q4_K_M (39.6 GB) | streaming | 0.03 | NVMe controller bound — multi-threading doesn't help when I/O is sequential |

### Known limitations

- Heavy-overflow streaming (>10 GB NVMe spill) is bottlenecked by NVMe controller bandwidth, not fd contention or thread count
- Co-activation predictions need 100+ tokens of inference to accumulate useful data
- `hypura optimize` writes a full copy of the model file
