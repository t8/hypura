```
 _   _
| | | |_   _ _ __  _   _ _ __ __ _
| |_| | | | | '_ \| | | | '__/ _` |
|  _  | |_| | |_) | |_| | | | (_| |
|_| |_|\__, | .__/ \__,_|_|  \__,_|
       |___/|_|
   メモリに収まらないモデルを動かす / Run models too big for your memory
```

---

## 概要

Hypura はストレージ階層を意識した LLM 推論スケジューラです。
モデルのテンソルを GPU・RAM・NVMe にアクセスパターン・帯域幅・ハードウェア特性に基づいて自動配置し、**物理メモリを超える大規模モデルをクラッシュなしに動作させます**。

- 31 GB の Mixtral 8x7B を 32 GB マシンで **2.2 tok/s** で実行
- 40 GB の Llama 3.3 70B を 32 GB マシンで **0.3 tok/s** で実行
- vanilla llama.cpp は両方 OOM でクラッシュ

**対応プラットフォーム:**

| プラットフォーム | GPU | NVMe I/O |
|---|---|---|
| macOS (Apple Silicon) | Metal (`F_NOCACHE` + `pread`) | ✅ |
| Windows ネイティブ | CUDA RTX 2060+ (`FILE_FLAG_NO_BUFFERING` + `ReadFile`) | ✅ |
| WSL2 (Windows) | CUDA RTX 2060+ (`posix_fadvise` + `pread`) | ✅ |
| Linux | CUDA RTX 2060+ (`posix_fadvise` + `pread`) | ✅ |

---

## なぜ必要か

コンシューマ向けハードウェア（MacBook Pro、RTX 3060 搭載 PC など）は高速な統合メモリや NVMe ストレージを搭載していますが、容量に限界があります。32 GB のマシンで 40 GB のモデルをナイーブに読み込もうとすると、OS がスワップを繰り返し OOM キラーが介入します。

Hypura はモデルアーキテクチャを理解することでこの問題を解決します:

- **Norms・Embeddings** — 小さいが毎トークンアクセスされる → GPU に固定
- **MoE エキスパートルーティング** — スパース性を利用。8 エキスパート中 2 つしか発火しない。ルーターインターセプションで選択されたエキスパートを識別し、GGUF ファイルから必要なストライドのみ NVMe ストリーミング（I/O 75% 削減）。ニューロンキャッシュが時間局所性を活かし 99.5% ヒット率を達成。共活性化追跡で投機的プリフェッチを実現
- **Dense FFN ウェイト** — gate/up/down ウェイト（モデルサイズの約 60%）をプールバッファ経由で NVMe からストリーミング。アテンション・Norms は GPU 常駐

---

## 動作原理

Hypura は GGUF ファイルを読み込み、ハードウェアをプロファイリング（GPU 作業セット、RAM、NVMe 帯域幅）し、すべてのテンソルを最適な階層に割り当てる配置最適化を解きます。

**推論モードの自動選択:**

| モード | 条件 | 説明 |
|---|---|---|
| **Full-resident** | モデルが GPU+RAM に収まる | NVMe I/O なし。フル GPU 速度 |
| **Expert-streaming** | MoE モデル（Mixtral 等）| 非エキスパートテンソル（~1 GB）のみ GPU。エキスパートは NVMe から on-demand ストリーミング |
| **Dense-FFN-streaming** | 大規模 Dense モデル（Llama 70B 等）| アテンション+Norms を GPU に（~8 GB）。FFN テンソルは NVMe からストリーミング |

プールバッファサイズ・プリフェッチ深度・メモリバジェットはハードウェアプロファイルから自動計算されます。

---

## NVMe ストリーミング — Windows 対応

Windows でも macOS と同等の NVMe キャッシュバイパス読み出しが動作します。

| 機能 | macOS | Linux/WSL2 | Windows |
|---|---|---|---|
| キャッシュバイパスオープン | `F_NOCACHE` | `O_DIRECT` | `FILE_FLAG_NO_BUFFERING` |
| ランダムオフセット読み出し | `pread(2)` | `pread(2)` | `ReadFile` + `OVERLAPPED` |
| 匿名ページ割り当て | `mmap(MAP_ANON)` | `mmap(MAP_ANON)` | `VirtualAlloc` |
| ページ解放ヒント | `madvise(MADV_FREE)` | `madvise(MADV_DONTNEED)` | `VirtualFree(MEM_DECOMMIT)` |

これらの違いは `src/io/compat.rs` の統一 API の背後に隠蔽されており、上位レイヤー（IoPool、NvmePrefetcher、iobench）はプラットフォームを意識しません。

---

## パフォーマンス

**M1 Max、32 GB 統合メモリ、NVMe シーケンシャル読み取り ~5.1 GB/s での計測値**

| モデル | サイズ | GPU | NVMe | モード | Hypura | llama.cpp | 備考 |
|---|---|---|---|---|---|---|---|
| Qwen 2.5 14B Q4_K_M | 8.4 GB | 8.4 GB | — | full-resident | **21 tok/s** | ~21 tok/s | GPU 収容、オーバーヘッドなし |
| Mixtral 8x7B Q5_K_M | 30.9 GB | 1.1 GB | 29.8 GB | expert-streaming | **2.2 tok/s** | **OOM** | 全層 Metal、キャッシュヒット率 99.5% |
| Llama 3.3 70B Q4_K_M | 39.6 GB | 7.8 GB | 31.8 GB | dense-FFN-streaming | **0.3 tok/s** | **OOM** | 全層 Metal、24 スロット動的プール、7 層プリフェッチ |

---

## インストール

Rust 1.75+ と CMake が必要です（vendored llama.cpp のビルドに使用）。

### macOS (Apple Silicon)

```sh
git clone --recurse-submodules https://github.com/zapabob/hypura.git
cd hypura
cargo build --release
```

### WSL2 (Windows)

CUDA ツールキット（12.x 推奨）がインストールされていることを確認してください。

```sh
# WSL2 ターミナル内で実行
git clone --recurse-submodules https://github.com/zapabob/hypura.git
cd hypura
cargo build --release
```

RTX 3060 以上では CUDA アーキテクチャ `sm_86` がデフォルトターゲットです（20xx: sm_75、40xx: sm_89、H100: sm_90）。

### Windows ネイティブ

```powershell
# PowerShell または Git Bash
git clone --recurse-submodules https://github.com/zapabob/hypura.git
cd hypura

# CUDA_PATH 環境変数を設定（例: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4）
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

cargo build --release
```

バイナリは `target/release/hypura.exe` に生成されます。

---

## クイックスタート

```sh
# ハードウェアプロファイリング（初回のみ、結果はキャッシュされる）
hypura profile

# GGUF モデルで推論
hypura run ./model.gguf --prompt "Hello, world"

# インタラクティブチャット
hypura run ./model.gguf --interactive

# ベンチマーク: Hypura vs ナイーブベースライン
hypura bench ./model.gguf

# モデル配置プランの確認（ロードなし）
hypura inspect ./model.gguf

# NVMe I/O マイクロベンチマーク
hypura iobench ./model.gguf --read-gb 1.0

# エキスパートレイアウト最適化（Mixtral 等の MoE モデル）
hypura optimize ./model.gguf
```

未テストモデルでは最初に `--max-tokens 10` から始めることを推奨します。

---

## Ollama 互換サーバー

Hypura は Ollama 互換 HTTP API を公開しており、Ollama に対応したツール（[OpenClaw](https://github.com/openclaw/openclaw) など）のドロップイン代替として機能します。

```sh
hypura serve ./model.gguf
# Hypura serving Mixtral 8x7B Instruct v0.1
#   Endpoint: http://127.0.0.1:8080
#   Ollama-compatible API: /api/generate, /api/chat, /api/tags
```

### エンドポイント

| エンドポイント | 説明 |
|---|---|
| `GET /` | ヘルスチェック |
| `GET /api/tags` | 読み込み済みモデル一覧 |
| `GET /api/version` | サーバーバージョン |
| `POST /api/show` | モデルメタデータ |
| `POST /api/generate` | テキスト補完（ストリーミング NDJSON または単一レスポンス）|
| `POST /api/chat` | チャット補完（ストリーミング NDJSON または単一レスポンス）|

### OpenClaw との連携

```sh
openclaw config set models.providers.ollama.baseUrl "http://127.0.0.1:8080"
```

---

## アーキテクチャ

Hypura は2つのクレートからなる Cargo ワークスペースです。

- **`hypura`** — メインバイナリ＋ライブラリ。CLI は `src/main.rs`、ロジックは `src/lib.rs` モジュール群
- **`hypura-sys`** — llama.cpp の FFI バインディング（`vendor/llama.cpp/` に vendored、CMake でビルド）

### 主要モジュール

| モジュール | 目的 |
|---|---|
| `io/compat.rs` | プラットフォーム抽象化（macOS/Linux/Windows の I/O プリミティブ統一）|
| `scheduler/placement.rs` | LP + greedy テンソル配置最適化（GPU/RAM/NVMe 階層）|
| `compute/inference.rs` | 推論エンジン: `generate_blocking`、`generate_with_nvme_scheduling` |
| `compute/nvme_backend.rs` | カスタム GGML バッファ型、エキスパート/FFN ストリーミング、ニューロンキャッシュ |
| `server/routes.rs` | Ollama 互換 API の Axum HTTP ハンドラ |
| `profiler/` | ハードウェア検出（CPU/GPU/メモリ帯域幅/NVMe スループット）|
| `cli/bench.rs` | A/B ベンチマークハーネス |
| `model/tensor_role.rs` | テンソル分類（配置スコアリング用）|
| `cache/coactivation.rs` | エキスパート共活性化追跡（投機的プリフェッチ用）|
| `cache/neuron_cache.rs` | 読み込み済みエキスパートスライスの LRU キャッシュ |

---

## FAQ

### SSD が壊れませんか?

いいえ。**Hypura は推論中に SSD への書き込みを一切行いません。**

SSD の劣化は書き込みサイクル（NAND フラッシュの P/E サイクル）によって生じます。読み取りはフラッシュセルを劣化させません。Hypura の NVMe I/O パスは読み取り専用で、GGUF ファイルからテンソルウェイトを RAM/GPU メモリプールにストリーミングするだけです。SSD はコールドストレージとして使用されます。

唯一の書き込みは無視できる程度です: ベンチマーク結果 JSON（~KB）、共活性化統計（~KB）、`hypura optimize` コマンド（任意実行時の1回限り）。通常の推論では SSD 書き込みはゼロです。

### Windows でも NVMe ストリーミングは動きますか?

はい。`FILE_FLAG_NO_BUFFERING` + `ReadFile(OVERLAPPED)` が macOS の `F_NOCACHE` + `pread` と同等の機能を提供します。詳細は `_docs/windows-wsl2-port.md` を参照してください。

---

## 安全上の注意

- `bench --baseline` はモデルが RAM − 4 GB ヘッドルームを超える場合にブロックされます。`--force` で上書き可能ですが自己責任で
- 未テストモデルでは必ず `--max-tokens 10` から始めてください
- テストモデルは `./test-models/` に置いてください（リポジトリには含めない）

---

## ライセンス

MIT

---

## Ethics

このリポジトリのコードは私が自分で書いたものではありません。このプロジェクトは LLM を使って私の指示に基づいてタスクを実行するという探求です。NVMe を活用した推論はメモリの一形態として（低速ではあるが）十分に有効であるにもかかわらず、未活用であるという直感から始まりました。

---
---

## Overview

Hypura is a storage-tier-aware LLM inference scheduler.
It places model tensors across GPU, RAM, and NVMe tiers based on access patterns, bandwidth costs, and hardware capabilities — enabling **models larger than physical memory to run without crashing**.

- Run a 31 GB Mixtral 8x7B on a 32 GB machine at **2.2 tok/s**
- Run a 40 GB Llama 3.3 70B on a 32 GB machine at **0.3 tok/s**
- Vanilla llama.cpp OOMs on both

**Supported platforms:**

| Platform | GPU backend | NVMe direct I/O |
|---|---|---|
| macOS (Apple Silicon) | Metal (`F_NOCACHE` + `pread`) | ✅ |
| Windows native | CUDA RTX 2060+ (`FILE_FLAG_NO_BUFFERING` + `ReadFile`) | ✅ |
| WSL2 (Windows) | CUDA RTX 2060+ (`posix_fadvise` + `pread`) | ✅ |
| Linux | CUDA RTX 2060+ (`posix_fadvise` + `pread`) | ✅ |

---

## Why does this matter?

Consumer hardware ships with fast GPU/unified memory and NVMe storage, but limited capacity. A 32 GB machine can't naively load a 40 GB model — the OS will swap-thrash until the OOM killer intervenes.

Hypura solves this by understanding model architecture:

- **Norms and embeddings** are tiny but accessed every token — pinned to GPU
- **MoE expert routing** exploits sparsity — only 2 of 8 experts fire per token. Router interception identifies selected experts in the eval callback, then loads only the needed expert strides from NVMe (75% I/O reduction). A neuron cache tracks loaded expert slices, achieving 99.5% hit rate from temporal locality. Co-activation tracking predicts next experts for speculative prefetch.
- **Dense FFN weights** (gate, up, down — ~60% of model size) stream from NVMe through a dynamically-sized pool buffer while attention + norms stay GPU-resident. Prefetch lookahead scales with available memory.

---

## How it works

Hypura reads the GGUF file, profiles your hardware (GPU working set, RAM, NVMe bandwidth), and solves a placement optimization assigning every tensor to a tier:

- **GPU** — Attention, norms, embeddings. Fastest access, limited by GPU working set.
- **RAM** — Overflow layers that don't fit in the GPU working set.
- **NVMe** — Remaining layers loaded on-demand via direct I/O, prefetched ahead of the forward pass.

**Inference mode selection (automatic):**

| Mode | Condition | Description |
|---|---|---|
| **Full-resident** | Model fits in GPU+RAM | No NVMe I/O. Full GPU speed. |
| **Expert-streaming** | MoE models (Mixtral, etc.) | Only non-expert tensors (~1 GB) on GPU. Expert tensors stream from NVMe on demand. |
| **Dense-FFN-streaming** | Large dense models (Llama 70B, etc.) | Attention+norms on GPU (~8 GB). FFN tensors stream from NVMe. |

Pool buffer sizes, prefetch depth, and memory budgets are computed automatically from your hardware profile.

---

## NVMe Streaming — Windows Support

Windows supports the same cache-bypass NVMe reads as macOS, implemented via `FILE_FLAG_NO_BUFFERING` + `ReadFile(OVERLAPPED)`.

| Feature | macOS | Linux/WSL2 | Windows |
|---|---|---|---|
| Cache-bypass open | `F_NOCACHE` | `O_DIRECT` | `FILE_FLAG_NO_BUFFERING` |
| Positional read | `pread(2)` | `pread(2)` | `ReadFile` + `OVERLAPPED` |
| Anonymous pages | `mmap(MAP_ANON)` | `mmap(MAP_ANON)` | `VirtualAlloc` |
| Page release hint | `madvise(MADV_FREE)` | `madvise(MADV_DONTNEED)` | `VirtualFree(MEM_DECOMMIT)` |

All platform differences are hidden behind the unified API in `src/io/compat.rs`. Upper layers (IoPool, NvmePrefetcher, iobench) are platform-agnostic.

---

## Performance

**All benchmarks on M1 Max, 32 GB unified memory, ~5.1 GB/s NVMe sequential read.**

| Model | Size | GPU | NVMe | Mode | Hypura | llama.cpp | Notes |
|---|---|---|---|---|---|---|---|
| Qwen 2.5 14B Q4_K_M | 8.4 GB | 8.4 GB | — | full-resident | **21 tok/s** | ~21 tok/s | Fits in GPU; no overhead |
| Mixtral 8x7B Q5_K_M | 30.9 GB | 1.1 GB | 29.8 GB | expert-streaming | **2.2 tok/s** | **OOM** | All layers on Metal; 99.5% cache hit rate |
| Llama 3.3 70B Q4_K_M | 39.6 GB | 7.8 GB | 31.8 GB | dense-FFN-streaming | **0.3 tok/s** | **OOM** | All layers on Metal; 24-slot dynamic pool, 7-layer prefetch |

---

## Install

Requires Rust 1.75+ and CMake (for vendored llama.cpp).

### macOS (Apple Silicon)

```sh
git clone --recurse-submodules https://github.com/zapabob/hypura.git
cd hypura
cargo build --release
```

### WSL2 (Windows)

Ensure CUDA Toolkit (12.x recommended) is installed.

```sh
git clone --recurse-submodules https://github.com/zapabob/hypura.git
cd hypura
cargo build --release
```

Default CUDA target is `sm_86` (RTX 3060 / Ampere). Supported: `sm_75` (RTX 20xx), `sm_86` (RTX 30xx), `sm_89` (RTX 40xx), `sm_90` (H100). Override with `HYPURA_CUDA_ARCHITECTURES=75;86;89;90`.

### Windows Native

```powershell
git clone --recurse-submodules https://github.com/zapabob/hypura.git
cd hypura

# Set CUDA Toolkit path
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

cargo build --release
```

Binary is at `target\release\hypura.exe`.

---

## Quick start

```sh
# Profile your hardware (runs once, results cached)
hypura profile

# Run inference on a GGUF model
hypura run ./model.gguf --prompt "Hello, world"

# Interactive chat
hypura run ./model.gguf --interactive

# Benchmark: Hypura scheduling vs naive baseline
hypura bench ./model.gguf

# Inspect model placement plan without loading
hypura inspect ./model.gguf

# NVMe I/O microbenchmark
hypura iobench ./model.gguf --read-gb 1.0

# Expert layout optimization (for MoE models like Mixtral)
hypura optimize ./model.gguf
```

Start with `--max-tokens 10` on untested models.

---

## Ollama-compatible server

Hypura exposes an Ollama-compatible HTTP API — a drop-in replacement for any tool that talks to Ollama, including [OpenClaw](https://github.com/openclaw/openclaw).

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

```sh
openclaw config set models.providers.ollama.baseUrl "http://127.0.0.1:8080"
```

---

## Architecture

Hypura is a Cargo workspace with two crates:

- **`hypura`** — Main binary and library. CLI in `src/main.rs`, all logic in `src/lib.rs` modules.
- **`hypura-sys`** — FFI bindings to llama.cpp (vendored at `vendor/llama.cpp/`, built via CMake).

### Key modules

| Module | Purpose |
|---|---|
| `io/compat.rs` | Platform abstraction (unifies macOS/Linux/Windows I/O primitives) |
| `scheduler/placement.rs` | LP + greedy tensor placement across GPU/RAM/NVMe tiers |
| `compute/inference.rs` | Inference engine: `generate_blocking`, `generate_with_nvme_scheduling` |
| `compute/nvme_backend.rs` | Custom GGML buffer type, pool-based expert/FFN streaming, neuron cache |
| `server/routes.rs` | Axum HTTP handlers for Ollama-compatible API |
| `profiler/` | Hardware detection (CPU, GPU, memory bandwidth, NVMe throughput) |
| `cli/bench.rs` | A/B benchmark harness |
| `model/tensor_role.rs` | Tensor classification for placement scoring |
| `cache/coactivation.rs` | Expert co-activation tracking for speculative prefetch |
| `cache/neuron_cache.rs` | LRU cache for loaded expert slices |

---

## FAQ

### Will this kill my SSD?

No. **Hypura only reads from your SSD during inference — it never writes to it.**

SSD wear is caused by write cycles. Reads do not degrade flash cells. Hypura's entire NVMe I/O path uses read-only calls (`pread` / `ReadFile`) with cache bypass to stream tensor weights from the GGUF file into RAM/GPU memory pools. The SSD is used as cold storage, not as working memory.

The only writes Hypura performs are negligible: benchmark result JSON files (~KB), co-activation statistics (~KB), and the one-time `hypura optimize` command. Normal inference generates zero SSD writes.

### Does NVMe streaming work on Windows?

Yes. `FILE_FLAG_NO_BUFFERING` + `ReadFile(OVERLAPPED)` provides the same functionality as `F_NOCACHE` + `pread` on macOS. See `_docs/windows-wsl2-port.md` for details.

---

## Safety notes

- `bench --baseline` is blocked when the model exceeds RAM minus 4 GB headroom. Use `--force` to override.
- Always start with `--max-tokens 10` on untested models.
- Test models belong in `./test-models/` (not checked in).

---

## License

MIT

---

## Ethics

I feel morally obligated to say I did *not* write the code in this repository myself. This project is an exploration of using LLMs to carry out tasks based on my direction. The majority of prompts I used to get here were derived using the socratic method, genuine curiosity, and a hunch that NVMe-supporting inference is underutilized despite being a (slow but) perfectly valid form of memory.
