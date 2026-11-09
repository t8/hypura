# Changelog

All notable changes to the **Hypura** project are documented in this file.

---

## [0.2.5] - 2026-10-06

### ✨ Features & Enhancements

#### 1. Dynamic Context Scaling for Agentic Coding (64k–128k tokens)
* **Context Parameter in CLI & API (`src/main.rs`, `src/cli/estimate.rs`):**
  * Added `-c, --context <N>` (with visible alias `--ctx-size`, default: 8192) to `hypura estimate <model>` to inspect layer offload trade-offs and KV memory consumption upfront.
  * Standardized visible alias `--ctx-size` across `hypura run` and `hypura serve` to match Ollama/OpenAI API flags.
  * Updated `hypura estimate` terminal output to display the auto-selected KV quantization format (`FP16`, `Q8_0`, `Q4_0`).

#### 2. Adaptive 4-Bit & 8-Bit KV Cache Quantization (`src/scheduler/placement.rs`)
* **Auto-Selected Quantization (`compute_kv_cache_plan`):**
  * Automatically switches to `Q4_0` for contexts $\ge 32\text{k}$ tokens or when FP16 KV cache exceeds 4 GB (saving ~72% memory).
  * Automatically switches to `Q8_0` for contexts $\ge 16\text{k}$ tokens or when memory headroom is constrained (saving ~47% memory).
* **Quantized Headroom Accounting (`compute_tier_capacities`):**
  * Scaled KV headroom reservation by the actual quantization factor ($0.28\times$ for Q4_0, $0.53\times$ for Q8_0) during placement calculation, preventing premature eviction of model layers to disk while keeping Metal working sets safe.

#### 3. KV Cache & Context Disk Serialization (`src/compute/ffi.rs`, `src/cache/kv_cache.rs`)
* Added `save_state_to_file` and `load_state_from_file` to `LlamaContext`, bridging to `llama_state_save_file` and `llama_state_load_file`.
* Added `save_to_disk` and `restore_from_disk` in `KvCacheManager` for checkpointing long agent prompts and restoring sessions across restarts.

#### 4. Startup Script Flexibility & CLI Parameter Forwarding (`start_hypura_funnel.sh`)
* Added dynamic option parsing supporting `-c / --context / --ctx-size`, `-p / --port`, and arbitrary trailing Hypura CLI flags.
* Default baseline context increased to **32k** (`32768`) for modern long-horizon agent coding workloads.
* Supports runtime environment variables `HYPURA_CONTEXT`, `HYPURA_PORT`, and `HYPURA_EXTRA_ARGS`.
* **Usage Examples:**
  ```bash
  # Launch with 64k context window (all models dynamic on-demand)
  ./start_hypura_funnel.sh -c 65536

  # Launch a specific model pre-warmed with 128k context on port 8080
  ./start_hypura_funnel.sh qwen2.5-coder-32b -c 131072 -p 8080

  # Pass custom scan directories or extra flags through to hypura serve
  ./start_hypura_funnel.sh -c 65536 --models-dir /Volumes/Models/gguf
  ```

---

## [0.2.4] - 2026-09-09

### ✨ Features & Enhancements

#### 1. Full OpenAI-Compatible API Support (`/v1/*` and direct route aliases)
* Added standard OpenAI endpoints to allow seamless drop-in integration with tools like **TealKit**, **Cursor**, **Cline**, **Continue.dev**, **LangChain**, and official OpenAI SDKs:
  * `GET /v1/models` and `GET /models` — lists all discovered local and Ollama models in OpenAI model object format.
  * `POST /v1/chat/completions` and `POST /chat/completions` — supports single responses, structured `tool_calls` output, thought/reasoning channel stripping, and Server-Sent Events (SSE) streaming (`stream: true`).
  * `POST /v1/completions` and `POST /completions` — text completion with SSE streaming.
* Defined standard request and response data structures in `src/server/openai_types.rs`.
* Implemented Server-Sent Events streaming handlers (`sse_openai_chat_stream`, `sse_openai_completion_stream`) in `src/server/streaming.rs`.

#### 2. Metal Memory & Context Overflow Protections
* **Strict Memory Capacity Check for Sparse MoE (`src/scheduler/placement.rs`):** Ensured `try_sparse_moe_mmap` verifies that the total mapped model size strictly fits within the unified memory ceiling, preventing Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory` command buffer failures.
* **Safety Prompt Windowing (`src/compute/inference.rs`):** Added automatic prompt truncation when massive tool outputs exceed the context capacity, preventing `failed to find a memory slot for batch` / unrecoverable KV cache exhaustion.
* **Effective Context Clamping (`src/compute/inference.rs`):** Capped `effective_ctx` to the server's configured context boundary (`config.n_ctx`) so multi-turn tool outputs do not trigger unbounded memory allocation.

#### 3. Dynamic KV Cache & Adaptive GPU Layer Offload
* **Dynamic Context Scaling (`src/server/manager.rs`):** Replaced static `default_context` clamp with dynamic scaling bounded by the model's native context limit (`metadata.context_length`), allowing client requests (`num_ctx`) to safely expand up to 16k, 32k, or native model context.
* **Adaptive Layer Placement on Context Expansion:** When an incoming request demands more context than currently loaded (`needed_ctx > active.context_size`), `ModelManager` dynamically recomputes the placement plan (`compute_placement_with_context`), automatically offloading fewer layers to GPU if needed to fit the expanded KV cache within the Metal working set without OOM.
* **OpenAI API Context Option Support (`src/server/openai_types.rs`, `src/server/routes.rs`):** Added `num_ctx` and `options` parsing to `OpenAIChatCompletionRequest` and `OpenAICompletionRequest`, passing client-requested context lengths directly to the dynamic model manager.
* **Startup Script Default Updated (`start_hypura_funnel.sh`):** Increased default context from 8k to 16k (`CONTEXT="${HYPURA_CONTEXT:-16384}"`) so server startup budgets GPU memory for 16k context upfront.

#### 4. Modern 14B–30B Model Optimization & Evaluation
* Authored comprehensive analysis and evaluation guide for modern 2026 models in **[docs/MODERN_MODELS_AND_OPENAI_INTERFACE.md](docs/MODERN_MODELS_AND_OPENAI_INTERFACE.md)**.
* Detailed performance and tier placement guidelines for 24GB & 32GB Apple Silicon Mac mini Pro models (M4 Pro / M5 / M6).

#### 5. CLI & Startup Banner Updates
* Updated `hypura serve` console output and `start_hypura_funnel.sh` startup script to display both Ollama and OpenAI endpoint availability.
* Added `.gitignore` rules for benchmark JSON results, test logs, and temporary documentation screenshots.

---

## [0.2.3] - 2026-08-29

### 🎥 Real-World Agent Demos & Verification
* 📺 **CumulusAI + Hypura: Gemma 4 26B Checks Datalogger Flash Space:** [https://youtu.be/6BRQYrkONqg](https://youtu.be/6BRQYrkONqg)
  *(Diagnostic check of remote meteorological station datalogger storage via cloud API).*
* 📺 **CumulusAI + Hypura: Run an Oversized Qwen 3.8 27B on Mac mini M4 Pro – Stations, Reasoning & Charts:** [https://youtu.be/brzBlL2LutQ](https://youtu.be/brzBlL2LutQ)
  *(Multi-station weather telemetry query, temperature filtering, reasoning, and chart formatting on 24GB Unified RAM).*
* Detailed benchmark summary available in **[docs/REAL_WORLD_TESTS.md](docs/REAL_WORLD_TESTS.md)**.

### ✨ Features & Enhancements

#### 1. Full CORS Middleware Support (`src/server/routes.rs`)
* Integrated `tower-http` CORS layer allowing browser-based web applications (such as web clients for CumulusAI, OpenWebUI, or custom agent frontends) to connect over local network or remote HTTPS tunnels (e.g. Tailscale Funnel).
* Automatically handles browser preflight `OPTIONS` requests and configures `Access-Control-Allow-Origin: *`, `Access-Control-Allow-Methods`, and `Access-Control-Allow-Headers`.
* Fully supports cross-origin chunked NDJSON streaming (`/api/chat`, `/api/generate`).

#### 2. Stream Chunk Deduplication (`src/server/streaming.rs`)
* Fixed the final streaming chunk (`done: true`) to emit an empty delta string (`message.content = ""`) in compliance with the Ollama streaming protocol.
* **Fixes UI Duplication:** Prevents web and desktop chat interfaces from duplicating the generated message or tables at the end of streaming.

#### 3. Multi-Model Native Tool Calling & Chat Formatting (`src/server/chat.rs`)
* **Qwen (2.5 & 3.8 27B):** Added XML and ChatML parameter extraction with reasoning/think tag isolation to prevent parameter cross-contamination and over-execution loops.
* **Mistral & Ministral (Small 24B, Ministral 3 14B):** Added native `[INST]`, `[AVAILABLE_TOOLS]`, and `[TOOL_RESULTS]` format support.
* **Gemma (4-26B & 4-31B):** Added `<start_of_turn>` / `<end_of_turn>` and `<|tool_call|>` support with automatic thought channel filtering (`<|channel>thought...<channel|>`).
* **IBM Granite (4.1 30B):** Structured turn boundaries and stop sequence detection.
* **GLM-4:** Added native GLM-4 prompt template and tool observation formatting.

#### 4. GLM-4 MoE Lite Architecture & Multi-Head Latent Attention (MLA) Support
* Implemented `LLM_ARCH_GLM4MOELITE` support across the model loader, compute graph, and RoPE.
* Optimized KV cache sizing for compressed MLA dimensions (576 dims per token), maximizing GPU layer offloading (up to 45/48 layers on 24GB Unified RAM).

#### 5. Metal Memory Headroom & SSM Hybrid Stability (`src/compute/inference.rs`, `src/scheduler/placement.rs`)
* **SSM / Gated Delta Net Headroom:** Added automatic architecture detection and scaled the Metal safety headroom (up to 5.8 GB) for hybrid SSM models (Gemma 4, Gated Delta Net, Mamba) on unified memory machines.
* **Eliminates Metal OOM:** Resolves `kIOGPUCommandBufferCallbackErrorOutOfMemory` command buffer failures on 24GB Apple Silicon Macs when evaluating models with 180+ compute graph splits.
* **Dynamic Context Headroom:** Automatically scales `effective_ctx` for large multi-turn tool payloads (tested with 25KB+ tool responses) to prevent KV cache saturation.

#### 6. Multi-Turn Resident Model Caching (`src/server/manager.rs`)
* Retains active models in memory between multi-turn tool execution steps without redundant reloads.
* Enforces server context limits to safeguard against client requests requesting oversized context windows.

---

## [0.2.2] - 2026-08-20

### ✨ Features & Bug Fixes

#### 1. Sampler State Synchronization (`llama_sampler_accept`)
* Integrated `llama_sampler_accept` into `LlamaSampler::sample()` to ensure newly sampled tokens are registered in the sampler chain.
* **Fixes Repetition Loops:** Resolves infinite word repeating loops (`type, type, type...`) during tool call generation on models like Qwen 2.5 / 3.8.

#### 2. Max 90% Unified Memory Limit Guard
* Added a hard 90% physical system memory limit (`(hw.memory.total_bytes * 0.90)`) for Metal GPU offloading (`compute_gpu_budget`) and RAM keep-resident mode (`load_model`).
* Prevents total memory exhaustion and system instability on Apple Silicon Macs (e.g. 24GB M-series).

#### 3. Expanded Native Tool Call Format Support (Mistral 24B & Qwen)
* Added support for `[TOOL_CALLS] [...]` array structures (standard Mistral v3 / Mistral 24B tool calling format) in `parse_tool_calls`.
* Reverted experimental `muse-glimmer` additions to maintain clean macOS compatibility.

---

## [0.2.1] - 2026-08-16

### 🚀 Major Highlights & Real-World Agent Demo
* **Demonstration & Verification:** Tested and validated with the **[Tealkit Agentic App](https://github.com/lschaffer/tealkit)** (Windows native application running on Windows 11) connected over local LAN to the **Hypura engine running on a Mac mini**.
* **Hardware & Engine:** Ran 100% locally on **Apple Silicon Mac mini M4 Pro (24 GB Unified RAM)** with **`devstral-small-2:24b`** (Mistral 3 architecture) with a **12k context window** under full Metal GPU acceleration.
* 🎥 **Video Demo:** Watch the cross-platform agentic workflow in action on YouTube: **[https://youtu.be/i28xrFum3KM](https://youtu.be/i28xrFum3KM)**

---

### ✨ Features & Enhancements

#### 1. Dynamic On-Demand Multi-Model Serving (`hypura serve`)
* `hypura serve` can now start as a dynamic background daemon without specifying a model upfront.
* Models are loaded on-demand when client requests arrive at `/api/generate` or `/api/chat`.
* Automatically hot-swaps models in GPU/Unified memory when the client selects a different model.
* Caches active models for instant response times on subsequent requests.

#### 2. Zero-Copy Ollama Model Sharing (`src/server/registry.rs`)
* Hypura automatically discovers all models installed in Ollama (`~/.ollama/models` or `$OLLAMA_MODELS`) by parsing manifests and mapping directly to `blobs/sha256-*` GGUF files.
* **Zero disk duplication:** Access all your downloaded Ollama models seamlessly without copying or manual symlinking.

#### 3. Native Ollama & OpenAI Tool Calling Support (`src/server/chat.rs`)
* Added full native tool calling parser and JSON schema prompt formatting.
* Supports Gemma 4 channel syntax (`<|tool_call>call:func{...}<tool_call|>`), standard ChatML JSON/XML tool calls, and OpenAI function calling structures.
* Automatically filters internal thought tokens (`<|channel>thought...<channel|>`) to ensure clean responses for client agents (Cline, Roo Code, Tealkit, Open WebUI).

#### 4. Dynamic Context Sizing & Auto-Expanding KV Cache
* Added `num_ctx` support in `GenerateOptions` and client request payloads.
* Context size is dynamically tokenized and allocated per-request (`effective_ctx = (prompt_len + max_tokens).max(config.n_ctx)`).
* Prevents `KV cache full / failed to find a memory slot` errors when prompts contain large tool schemas or long multi-turn conversations.

#### 5. New CLI Commands & Monitoring APIs
* **`hypura list` (`src/cli/list.rs`):** Displays all available local and Ollama models with parameter counts, quantization levels, architectures, and sizes.
* **`hypura ps` (`src/cli/ps.rs`):** CLI tool to inspect active models loaded in memory, context sizes, and GPU offload status.
* **`GET /api/ps` (`src/server/routes.rs`):** Standard Ollama-compatible process status endpoint.

#### 6. Multimodal GGUF LLM Loading Support
* Patched `llama.cpp` loader (`llama_model_loader::done_getting_tensors`) to tolerate unmapped multimodal vision/audio encoder tensors.
* Enables single-file loading of unified multimodal GGUF models (e.g. Mistral-Small 3.2 24B / Pixtral, Gemma 4) directly for text generation and tool calling.

#### 7. Tailscale Funnel & Public HTTPS Script (`start_hypura_funnel.sh`)
* Added automated startup script with background Tailscale Funnel configuration (`tailscale funnel --bg --yes 6000`).
* Displays public HTTPS endpoints (`https://<mac>.<tailnet>.ts.net`) for remote agent access over SSL.

---

### 📚 Documentation
* Added **[docs/MEMORY_SIZING.md](docs/MEMORY_SIZING.md)**: Detailed memory breakdown tables, KV cache calculation formulas, and context window sizing for Apple Silicon Unified RAM architectures.
