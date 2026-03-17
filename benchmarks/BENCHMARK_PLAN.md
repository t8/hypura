# Benchmark Plan: Finding the 3-5x Multiplier

## Goal

Demonstrate Hypura's 3-5x speedup over naive mmap for models that exceed GPU
capacity on consumer Apple Silicon hardware (M1 Max, 32GB unified memory).

## Current State (Updated 2026-03-17)

| Model | Size | NVMe Spill | Mode | Hypura | Vanilla | Status |
|-------|------|------------|------|--------|---------|--------|
| TinyLlama 1.1B | 0.6 GB | — | GPU-only | 71.8 tok/s | N/A | No tiering needed |
| Qwen 2.5 14B | 8.4 GB | — | GPU-only | 24.0 tok/s | N/A | No tiering needed |
| Qwen 2.5 32B | 22.0 GB | — | GPU+RAM | TBD | TBD | Downloaded, not yet tested |
| Mixtral 8x7B | 30.9 GB | 2.0 GB | Keep-resident | **0.8 tok/s** | **OOM** | **Runs vs crashes** |
| Llama 3.3 70B | 39.6 GB | 9.8 GB | Streaming | **0.03 tok/s** | OOM | 1.7x improvement from eval fix |

### Progress on 2026-03-17

1. **MoE expert optimizations (Phases 0-4)** implemented: ExpertLayout, router
   interception, neuron cache, speculative prefetch. Not yet measurable because
   Mixtral uses keep-resident mode (2 GB NVMe fits in RAM).
2. **Eval callback fix:** Compute graph tensor names use `{name}-{layer}` format,
   not `blk.N.xxx`. Fixed parser → proper layer release/reload in streaming mode.
   Llama 70B improved 0.02 → 0.03 tok/s (~1.7x).
3. **Keep-resident mode:** Committed memory estimator gates keep-resident on actual
   memory pressure, not model file size. Mixtral stays at 0.8 tok/s with zero
   per-token NVMe I/O.
4. **Key learning:** pread into mmap-backed buffers converts file-backed pages to
   anonymous pages → 2.5x slowdown. Keep-resident mode now relies on mmap for
   data population.

There is still a gap between 8.4 GB (no tiering) and 30.9 GB (baseline crashes).
A ~20 GB model would enable a direct speed comparison (Strategy 1). A larger MoE
model (~48 GB) would test the expert-level optimizations in streaming mode.

## Hardware Constraints (M1 Max 32GB)

- GPU budget (after KV cache + Metal overhead): ~22-24 GB
- Usable RAM (after OS overhead): ~28-30 GB
- Baseline safety threshold: model must be < 28 GB (total_ram - 4 GB headroom)
- NVMe sequential read: ~5.1 GB/s

## Strategy 1: Sweet-Spot Models (15-28 GB)

Download models that overflow GPU but fit in RAM. Baseline can run safely,
giving a direct A/B comparison.

Candidates (all Q4_K_M unless noted):

| Model | Approx Size | Why interesting |
|-------|-------------|-----------------|
| Codestral 22B | ~13 GB | Overflows GPU by a few GB if context is large |
| Mistral Small 24B | ~14 GB | Similar, common model |
| Command-R 35B | ~20 GB | Solidly overflows GPU, fits in RAM |
| DeepSeek-Coder 33B | ~19 GB | Same tier |
| Yi 34B | ~19 GB | Same tier |
| Llama 3.1 70B Q2_K | ~25 GB | Aggressive quant, fits in RAM, many layers spill from GPU |

The ~20 GB models are the best candidates. They overflow GPU by ~half their
weight, forcing meaningful GPU/RAM tiering. Baseline will mmap everything
and let the OS page-fault randomly; Hypura will pin hot tensors on GPU.

Run with: `hypura bench --baseline --max-tokens 128 ./model.gguf`

## Strategy 2: Vanilla llama-cli Comparison on Mixtral — COMPLETE

Tested vanilla llama.cpp (build 8329, 1d3da8b8a) against Hypura on
Mixtral 8x7B Q5_K_M (30.9 GB) on M1 Max 32GB.

### Result: Vanilla llama.cpp cannot run this model

Vanilla llama.cpp mmap's the entire GGUF file (~31 GB) as a single Metal
shared buffer. Metal tracks the full buffer size against the GPU working
set (`recommendedMaxWorkingSetSize` = 26.8 GB), causing
`kIOGPUCommandBufferCallbackErrorOutOfMemory` at any ngl > 0.

| Setting | Result |
|---------|--------|
| `ngl=24` | GPU OOM |
| `ngl=16` | GPU OOM |
| `ngl=8` | GPU OOM |
| `ngl=0` (CPU only) | No OOM, but >10 min per token (swap thrash on 31 GB mmap) |
| `ngl=0 --no-mmap` | Hangs — malloc(31 GB) on 32 GB RAM |

### Hypura result

| Metric | Value |
|--------|-------|
| GPU offload | 24 layers on Metal |
| Generation | **1.0 tok/s** |
| Prompt eval | **1.8s** (2048 ctx) |
| GPU committed | ~14 GB (vs 31 GB for vanilla) |

### Why Hypura works

Hypura's custom GGML buffer type breaks the monolithic mmap into separate
allocations: Metal shared buffers for GPU layers (~14 GB), `posix_memalign`
for CPU layers (~9 GB). Metal only tracks the GPU portion, staying well
under the working set limit.

## Strategy 3: More Tokens on Existing Models

Current benchmarks generate only 10 tokens. Hypura's NVMe prefetch amortizes
startup costs over longer runs. Re-run with `--max-tokens 128` or `--max-tokens 256`
on models that already work:

```sh
hypura bench --max-tokens 128 ./test-models/qwen2.5-14b-q4_k_m.gguf
hypura bench --max-tokens 128 ./test-models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf
```

This won't show a speedup over baseline (Qwen fits in GPU) but will give
more stable tok/s numbers and show Hypura sustaining throughput.

## Priority Order

1. ~~**Strategy 2**~~ — DONE. Vanilla llama.cpp OOMs; Hypura runs at 0.8 tok/s.
2. **Strategy 1** — test Qwen 2.5 32B (22 GB, already downloaded) for direct A/B with --baseline
3. **Strategy 4 (NEW)** — large MoE model for expert optimization testing
4. **Strategy 3** — longer runs for more stable numbers

## Strategy 4: Large MoE Model for Expert Optimization Testing

The MoE expert-level optimizations (Phases 2-4) are implemented but untested in
streaming mode. Need a large MoE model that spills significantly to NVMe:

| Model | Approx Size | NVMe Spill | Why interesting |
|-------|-------------|------------|-----------------|
| Mixtral 8x7B Q8_0 | ~48 GB | ~24 GB | Same architecture, expert optimization directly comparable |
| DeepSeek-V2-Lite | ~16 GB | ~0 GB | MoE but too small to overflow |
| Mixtral 8x22B Q4_K_M | ~80 GB | ~56 GB | Way too large for 32 GB |

Mixtral Q8_0 is the best candidate: same 8x7B architecture, 2-of-8 expert routing,
but at ~48 GB it would spill ~24 GB to NVMe. Expert-aware loading would reduce
per-token NVMe I/O from ~24 GB to ~6 GB (75% reduction), targeting ~4x improvement
in the I/O-bound component.

## What Success Looks Like

```
  Model: Mixtral 8x7B Q8_0 (48 GB) on 32 GB M1 Max
  Without expert opt:  ~0.02 tok/s (load all 8 experts per layer)
  With expert opt:     ~0.08 tok/s (load 2/8 experts + neuron cache)
  Speedup:             ~4x on I/O-bound component
```

The Mixtral Q5_K_M comparison is complete:

```
  llama-cli (vanilla):   GPU OOM at any ngl > 0, unusable at ngl=0
  Hypura:                0.8 tok/s with 24 GPU layers
  Verdict:               runs a model that vanilla llama.cpp cannot
```
