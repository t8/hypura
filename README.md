# Hypura

**Run models too big for your Mac's memory.**

Hypura is a storage-tier-aware LLM inference scheduler for Apple Silicon. It places model tensors across GPU (Metal), RAM, and NVMe based on access patterns and hardware bandwidth — enabling you to run models that exceed your available memory where vanilla llama.cpp would OOM or thrash to a halt.

Run a 31 GB Mixtral 8x7B on a 32 GB Mac Mini at 2.2 tok/s. A 40 GB Llama 70B at 0.3 tok/s. Vanilla llama.cpp crashes on both.

## How it works

On Apple Silicon, GPU memory is unified with system RAM. When a model doesn't fit, llama.cpp either OOMs with GPU offloading or falls back to CPU-only with swap thrashing (~10+ min/token). Hypura adds a third option: **NVMe-tiered scheduling**.

Hypura reads the GGUF file, profiles your hardware (GPU working set, RAM, NVMe bandwidth), and solves a placement optimization that assigns every tensor to a tier:

- **GPU (Metal)** — Attention layers, norms, embeddings. Fastest access, limited by `recommendedMaxWorkingSetSize`.
- **RAM** — Overflow layers that don't fit in the GPU working set. Accessed via mmap.
- **NVMe** — Remaining layers loaded on-demand via direct I/O (`F_NOCACHE` + `pread`), prefetched ahead of the forward pass.

Hypura selects the best inference mode automatically:

- **Keep-resident** — For small NVMe spill. Loads NVMe layers once, keeps them in memory. Zero disk I/O after the first pass.
- **Expert-streaming** — For MoE models (Mixtral). Only non-expert tensors (~1 GB) stay on GPU. Expert tensors stream from NVMe through a pool buffer on demand, with a neuron cache (99.5% hit rate) that eliminates most I/O after warmup.
- **Dense FFN-streaming** — For dense models too large for GPU (Llama 70B). Attention + norms stay on GPU (~8 GB). FFN tensors (~32 GB) stream from NVMe through a pool buffer, with layer-ahead prefetch.

## Performance

All benchmarks on **M1 Max, 32 GB unified memory, ~5.1 GB/s NVMe sequential read**.

| Model | Size | GPU | NVMe | Mode | Hypura | llama.cpp | Notes |
|---|---|---|---|---|---|---|---|
| Qwen 2.5 14B Q4_K_M | 8.4 GB | 8.4 GB | — | full-resident | **21 tok/s** | ~21 tok/s | Fits in GPU; no overhead |
| Mixtral 8x7B Q5_K_M | 30.9 GB | 1.1 GB | 29.8 GB | expert-streaming | **2.2 tok/s** | **OOM** | All layers on Metal; 99.5% cache hit rate |
| Llama 3.3 70B Q4_K_M | 39.6 GB | 7.8 GB | 31.8 GB | dense-FFN-streaming | **0.3 tok/s** | **OOM** | All layers on Metal; 3-layer prefetch lookahead |

**Key takeaway:** For models that fit in memory, Hypura adds zero overhead. For models that don't fit, Hypura is the difference between "runs" and "crashes." Expert-streaming on Mixtral achieves usable interactive speeds by keeping only non-expert tensors on GPU and exploiting MoE sparsity (only 2/8 experts fire per token). Dense FFN-streaming extends this to non-MoE models like Llama 70B.

## Install

Hypura builds from source with Cargo. You'll need Rust 1.75+ and CMake (for the vendored llama.cpp).

```sh
git clone --recurse-submodules https://github.com/hypura/hypura.git
cd hypura
cargo build --release
```

The binary is at `target/release/hypura`.

> Homebrew tap coming soon.

## Quick start

```sh
# Profile your hardware (runs once, cached)
hypura profile

# Run inference on a GGUF model
hypura run ./model.gguf --prompt "Hello, world"

# Interactive chat
hypura run ./model.gguf --interactive

# Benchmark: Hypura scheduling vs naive baseline
hypura bench ./model.gguf

# Inspect model placement plan without loading
hypura inspect ./model.gguf
```

Start with `--max-tokens 10` on untested models before scaling up.

## Ollama-compatible server

Hypura exposes an Ollama-compatible HTTP API, making it a drop-in replacement for any tool that talks to Ollama — including [OpenClaw](https://github.com/openclaw/openclaw).

```sh
hypura serve ./model.gguf
# Hypura serving Mixtral 8x7B Instruct v0.1
#   Endpoint: http://127.0.0.1:8080
#   Ollama-compatible API: /api/generate, /api/chat, /api/tags
```

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `GET /api/tags` | List loaded model |
| `GET /api/version` | Server version |
| `POST /api/show` | Model metadata |
| `POST /api/generate` | Text completion (streaming NDJSON or single response) |
| `POST /api/chat` | Chat completion (streaming NDJSON or single response) |

### Usage with OpenClaw

Point OpenClaw at Hypura by setting the Ollama base URL in `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "ollama": {
        "baseUrl": "http://127.0.0.1:8080",
        "api": "ollama"
      }
    }
  }
}
```

Or via the CLI:

```sh
openclaw config set models.providers.ollama.baseUrl "http://127.0.0.1:8080"
```

Hypura speaks native Ollama protocol (`/api/chat` with NDJSON streaming), so no compatibility shims are needed.

### Server options

```
hypura serve <MODEL> [OPTIONS]

Options:
  --host <HOST>        Host to bind to [default: 127.0.0.1]
  --port <PORT>        Port to bind to [default: 8080]
  --context <N>        Maximum context length [default: 4096]
```

## Architecture

Hypura is a Cargo workspace with two crates:

- **`hypura`** — Main binary and library. CLI in `src/main.rs`, all logic in `src/lib.rs` modules.
- **`hypura-sys`** — FFI bindings to llama.cpp (vendored at `vendor/llama.cpp/`, built via CMake).

### Key modules

| Module | Purpose |
|---|---|
| `scheduler/placement.rs` | LP + greedy tensor placement across GPU/RAM/NVMe tiers |
| `compute/inference.rs` | Inference engine: `generate_blocking`, `generate_with_nvme_scheduling`, server-oriented `load_model` / `generate_from_loaded` |
| `compute/nvme_backend.rs` | Custom GGML buffer type, pool-based expert/FFN streaming, neuron cache, eval callback |
| `server/routes.rs` | Axum HTTP handlers for Ollama-compatible API |
| `profiler/` | Hardware detection (CPU, GPU, memory bandwidth, NVMe throughput) |
| `cli/bench.rs` | A/B benchmark harness |
| `model/tensor_role.rs` | Tensor classification for placement scoring (norms, attention, MoE experts) |

## Safety notes

- `bench --baseline` is blocked when the model exceeds RAM minus 4 GB headroom. Use `--force` to override at your own risk.
- Always start with `--max-tokens 10` on untested models.
- Test models belong in `./test-models/` (not checked in).

## License

MIT
