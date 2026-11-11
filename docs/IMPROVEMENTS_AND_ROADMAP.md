# Hypura Performance Optimizations & Architecture Roadmap

This document outlines prioritized architectural improvements, real-world diagnostic findings, and optimization strategies for **Hypura** serving modern 14B–32B models on Apple Silicon (M4 Pro / M5 / M6 with 24GB Unified Memory).

---

## 1. Executive Summary & Diagnostic Findings

Recent real-world testing with local coding agents (e.g. executing multi-turn project analysis on `qwen3.8:27b` over LAN/Tailscale) revealed two critical bottlenecks during multi-turn workflows:

1. **Turn 3 Client Socket Timeout (125.75s Prefill vs 120s Timeout)**:
   - **Turn 1 (4,634 tokens)**: Prefill completed in 64.8s (71.5 tok/s). Single tool call (`fs_list_dir`) generated in 1.46s.
   - **Turn 2 (4,845 tokens)**: Prefill completed in 73.5s (65.9 tok/s). File read tool call (`fs_read_file README.md`) generated in 2.76s.
   - **Turn 3 (7,709 tokens)**: Accumulated context included directory listings and the full README. Prefill took **125.75s** (61.3 tok/s avg), exceeding the agent's default 120s HTTP socket read timeout by 5.75 seconds.

2. **Memory Contention & Swap Throttling on 24 GB Unified Memory**:
   - Activity Monitor telemetry during Turn 3:
     - **Physical Memory**: 24.00 GB
     - **Memory Used**: 21.94 GB (Memory Pressure: **Yellow**)
     - **Wired / Metal Residency**: 17.09 GB (Hypura 27B model layers)
     - **Active Swap**: 3.11 GB
   - Paging contention with background developer processes (Xcode, SWBBuildService, Dart AOT runtime, browser tabs) dropped chunk prefill speed from **75.7 tok/s** down to **43.7 tok/s** at token 5,632, directly causing the timeout.

3. **Zero Prefix Cache Reuse on Hybrid SSM Models (`reused 0/7709 tokens`)**:
   - Because `qwen3.8` utilizes a Gated Delta Net (hybrid SSM/linear attention) architecture, tail sequence removal (`llama_memory_seq_rm`) is rejected by recurrent memory. Hypura cleared the entire context memory each turn, forcing all 7,709 tokens to be re-evaluated from scratch instead of only processing the newly appended ~2,864 tokens.

---

## 2. Prioritized Improvements Roadmap

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ P0: Multi-Turn State Checkpointing & Sequence Cloning                        │
│     Avoids 125s re-prefills by caching sequence state across turns           │
├──────────────────────────────────────────────────────────────────────────────┤
│ P1: Memory Pressure Mitigation & Quantized KV Cache (q8_0 / q4_0)             │
│     Halves KV cache footprint; prevents macOS swap paging on 24GB machines   │
├──────────────────────────────────────────────────────────────────────────────┤
│ P2: Client Timeout Configuration & Prefill Progress Signaling                 │
│     Document 300s+ agent timeouts; provide early progress heartbeat telemetry│
├──────────────────────────────────────────────────────────────────────────────┤
│ P3: Universal GGUF Jinja2 Chat Template Engine                                │
│     Embed minijinja to parse GGUF tokenizer.chat_template automatically      │
├──────────────────────────────────────────────────────────────────────────────┤
│ P4: Progressive Layer Splitting (Dynamic FFN Tiering)                         │
│     Offload middle FFN layers to NVMe/RAM to fit 30B+ models without swap    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. High Impact: Multi-Turn State Checkpointing & Sequence Cloning (P0)

### Problem Analysis
In [`src/compute/inference.rs`](file:///Volumes/RealtekData/projects/hypura/src/compute/inference.rs), Hypura attempts to trim existing KV memory to the longest matching token prefix:
```rust
let common_len = session.tokens.iter().zip(eval_tokens.iter()).take_while(|(a, b)| a == b).count();
let desired_reuse = common_len.min(eval_tokens.len().saturating_sub(1));
let kept = session.ctx.trim_kv_to_prefix(desired_reuse, n_swa);
```
In [`src/compute/ffi.rs`](file:///Volumes/RealtekData/projects/hypura/src/compute/ffi.rs), `trim_kv_to_prefix` invokes:
```c
llama_memory_seq_rm(mem, 0, keep_len, -1);
```
For hybrid recurrent architectures (**Qwen 3.5/3.8 Gated Delta Net**, **Gemma 4**), `llama_memory_recurrent::seq_rm` cannot roll back compressed recurrent states ($R, S$). Consequently, it returns `false`, triggering `self.clear_memory()` and resetting `reused_len` to 0.

### Proposed Solutions

#### Approach A: Sequence Cloning via `llama_memory_seq_cp`
- When an assistant turn completes at token position $N$:
  - Clone sequence 0 state to an inactive sequence id (e.g., `s0` $\rightarrow$ `s1`):
    ```c
    llama_memory_seq_cp(mem, 0, 1, 0, N);
    ```
- When the next prompt arrives with a matching prefix of length $N$:
  - Rather than attempting to delete tokens beyond $N$ in sequence 0, copy the clean checkpoint from sequence 1 back to sequence 0 (`s1` $\rightarrow$ `s0`).
  - Decode only the appended delta tokens ($N \rightarrow N + M$).

#### Approach B: Turn-Boundary State Snapshots via `llama_state_seq_get_data_ext`
- Use the official sequence state serialization API in `llama.cpp`:
  - `llama_state_seq_get_size_ext(ctx, seq_id, flags)`
  - `llama_state_seq_get_data_ext(ctx, buf, size, seq_id, flags)`
  - `llama_state_seq_set_data_ext(ctx, buf, size, seq_id, flags)`
- Store state snapshots in RAM keyed by prefix token hashes.
- Restore the snapshot before prefilling subsequent turns.

### Expected Performance Impact
- **Turn 1 (4,634 tokens)**: ~65s (initial prompt evaluation).
- **Turn 2 (4,845 tokens)**: Only 211 tokens evaluated $\rightarrow$ **~3.2 seconds** (was 73.5s).
- **Turn 3 (7,709 tokens)**: Only 2,864 tokens evaluated $\rightarrow$ **~38 seconds** (was 125.75s).
- Completely eliminates timeouts on subsequent conversation turns.

---

## 4. Memory Pressure & KV Cache Quantization (P1)

### Diagnostic Summary
On a 24 GB Mac mini:
- System + OS baseline: ~2.5 GB
- Model weights (`qwen3.8:27b` Q4_K_M): ~16.02 GB wired Metal allocation
- Remaining free memory for KV cache and desktop apps: **~5.5 GB**
- When context allocation is set to 32,768 tokens, KV memory and scratch buffers consume 2–4 GB. Running developer tooling (Xcode, compilers, IDEs) pushes the system into yellow memory pressure (21.94 GB used) with 3.11 GB swap, halving GPU memory bandwidth.

### Proposed Solutions

1. **Auto-Select Quantized KV Cache (`q8_0` / `q4_0`)**:
   - Enable `q8_0` KV cache:
     - Halves KV buffer memory with virtually zero measurable perplexity degradation.
     - Saves 1.5–2.0 GB on long context windows ($>16\text{k}$).
   - Enable `q4_0` KV cache for contexts $\ge 32\text{k}$:
     - Reduces KV buffer memory by 72%.
   - Add explicit CLI flags to `hypura serve`:
     ```bash
     hypura serve --model qwen3.8:27b --ctx 32768 --kv-quant q8_0
     ```

2. **Idle Resource Reclamation (`--idle-timeout`)**:
   - In [`src/server/manager.rs`](file:///Volumes/RealtekData/projects/hypura/src/server/manager.rs), add an idle TTL tracker.
   - If no requests are received for a specified duration (e.g., 10 minutes), unload model weights and free the 17.09 GB Metal residency set.
   - Reload transparently upon the arrival of the next request.

---

## 5. Client Socket Timeout & Streaming Guidance (P2)

### Diagnostic Summary
When coding agents trigger tool calls, their internal harnesses frequently force `stream: false` to parse completed JSON tool call definitions. In non-streaming mode:
- Standard HTTP clients maintain an idle TCP socket while awaiting HTTP response headers.
- If prefill exceeds the client socket read timeout (commonly 60s or 120s), the client forcibly terminates the connection just before completion.

### Proposed Solutions

1. **Client Configuration Recommendation**:
   - Configure coding agent client profiles (e.g. in Continue, Cline, Tealkit, Cursor) with a connection read timeout of **300s (5 minutes)** or **600s (10 minutes)** for local models $\ge 24\text{B}$.

2. **Hypura Server Prefill Warning & Logging**:
   - In [`src/server/routes.rs`](file:///Volumes/RealtekData/projects/hypura/src/server/routes.rs), detect non-streaming requests (`stream: false`) where prompt length exceeds 4,000 tokens.
   - Log an informative notice with estimated prefill duration:
     ```text
     [WARN] Non-streaming request with 7709 tokens detected; prefill estimated at ~110-130s. Ensure client HTTP read timeout is >= 300s.
     ```

3. **HTTP Keep-Alive Delimiter on Supported Clients**:
   - For clients supporting HTTP 1.1 Chunked Transfer Encoding on JSON endpoints, stream empty chunk delimiters or progress comments to prevent intermediate gateway timeouts.

---

## 6. Universal GGUF Jinja2 Chat Template Engine (P3)

### Current Architecture
[`src/server/chat.rs`](file:///Volumes/RealtekData/projects/hypura/src/server/chat.rs) utilizes explicit custom Rust formatters for each supported model architecture (`Qwen`, `Gemma`, `Mistral`, `Nemotron`). While reliable, each new model release requires dedicated formatting code.

### Proposed Implementation
1. **Embed `minijinja`**:
   - Add `minijinja` to `Cargo.toml`.
2. **Extract Embedded Template**:
   - Inspect GGUF metadata key `tokenizer.chat_template`.
3. **Template Rendering**:
   - Pass conversation messages (`system`, `user`, `assistant`, `tool`), tool definitions, and system prompts directly to the GGUF's native Jinja template.
   - Seamlessly support new model conventions (Harmony, ChatML, Turn tokens, Tekken) out of the box without code modifications.
4. **Fallback Mechanism**:
   - If a GGUF file lacks a valid chat template, fall back to Hypura's verified native formatters.

---

## 7. Progressive Layer Splitting / Dynamic FFN Tiering (P4)

### Context & Opportunity
For models in the 27B–32B range (e.g. `qwen2.5-coder:32b`, `granite-30b`), keeping all layers on GPU requires ~19.5–22 GB, which cannot run concurrently with macOS desktop apps on a 24 GB machine without heavy swapping.
Conversely, moving all FFN layers to NVMe reduces generation speed to ~3–4 tok/s.

### Proposed Strategy
- Implement progressive tiering:
  - **First $N$ layers (0 to $K$)**: Full GPU residency.
  - **Middle layers ($K$ to $L$)**: FFN weights streamed from RAM or high-speed NVMe via double-buffered prefetching.
  - **Final layers ($L$ to end)**: Full GPU residency.
- Frees 3–5 GB of unified memory while maintaining **10–14 tok/s** generation throughput.

---

## 8. Strata-Inspired 180B Ultra-Sparse MoE & Sparse N-Gram Embedding Tiering

### Architectural Breakdown (Qwen 3.8 Flash Next & Future 180B Models)
Modern massive open-weight models like **Qwen3.8-Flash-Next** scale parameter counts to **180 Billion**, but decompose them into distinct functional tiers:
1. **51 Billion N-Gram / PLE Embeddings**:
   - Pure sparse dictionary point-lookups (`embedding[token_id]`), requiring zero matrix multiplications.
   - Reading a few kilobytes per token takes microseconds.
   - Hypura assigns `TensorRole::NgramEmbedding` strictly to `StorageTier::Ram` / `StorageTier::Nvme` and overrides them to the CPU buffer type, preventing 25–35 GiB of embedding data from taking up Metal GPU residency.
2. **125 Billion MoE Backbone**:
   - Ultra-sparse routing activates **only ~6 Billion parameters** per token.
   - Shared Gated DeltaNet linear attention, QSA (Qwen Sparse Attention), and gating weights stay pinned on Metal GPU (`MTL0`), while routed experts stream through Hypura's neuron cache.
3. **4 Billion Multi-Token Prediction (MTP) Heads**:
   - Built-in speculative draft heads that predict 2–3 tokens in parallel, verified in one batched forward step to multiply generation throughput by $1.8\times - 2.5\times$.

### Apple Silicon Unified Memory Advantage over PC PCIe
- On PC GPUs (e.g. 12GB RTX 3060/4070), weights must stream over PCIe Gen 4 x16 ($\sim 25\text{--}30\text{ GB/s}$).
- On Apple Silicon Mac mini (M4 Pro), Unified Memory provides **$273\text{ GB/s}$** bandwidth ($10\times$ faster). Processing 6B active parameters natively achieves **$18\text{--}28\text{ tok/s}$**.

---

## 9. Summary Table of Action Items

| Item | Focus Area | Complexity | Expected Impact |
|---|---|---|---|
| **Turn State Checkpointing** | Prefill latency | Medium | Cuts multi-turn prefill from 125s to <40s |
| **Quantized KV Cache (`q8_0`)** | Memory pressure | Low | Frees 1.5–2.5 GB unified RAM, avoids swap |
| **Client Timeout Docs & Log ETA** | Connection stability | Low | Eliminates 120s premature client timeouts |
| **Sparse N-Gram Embedding Tiering** | 180B MoE Scaling | Low | Runs 180B Qwen3.8 Flash Next on 24GB Macs |
| **Idle Memory TTL (`--idle-timeout`)**| System responsiveness | Low | Automatically releases 17 GB VRAM when idle |
| **Universal Jinja2 Template** | Model compatibility | Medium | Automatic zero-code support for new models |
| **Progressive FFN Tiering** | Scaling to 32B models | High | Runs 32B models at 10–14 tok/s on 24GB Macs |

