# Hypura 実装ログ / Implementation Log

時系列順の実装記録。各フェーズの設計決定・根拠・重要な詳細を含む。

---

## Phase 1: スキャフォールド・FFI

**実装内容:**
- Cargo ワークスペース構成（`hypura` クレート + `hypura-sys` クレート）
- llama.cpp を `vendor/llama.cpp/` に git submodule として vendored
- `hypura-sys/build.rs` で CMake ビルド統合
- `hypura-sys/src/lib.rs` で llama.cpp C API の Rust bindgen バインディング
- カスタム GGML バッファ型 `hypura_buft.c` のスキャフォールド

**設計決定:**
- `hypura-sys` を別クレートに分離: bindgen の再生成と C/Rust 境界を明確に分離するため
- CMake + vendored llama.cpp: システムインストールの llama.cpp に依存しないようにするため

---

## Phase 2: ハードウェアプロファイラ

**実装内容: `src/profiler/`**

- `cpu.rs`: CPU モデル名・コア数・SIMD 機能（AVX2/AVX-512）の検出
  - macOS: `sysctlbyname` で `hw.model`、`hw.physicalcpu`、`hw.perflevel0.logicalcpu`
  - Linux: `/proc/cpuinfo` でモデル名、`sysinfo` クレートでコア数
  - Windows: `sysinfo` クレートで両方
  - x86_64: `std::is_x86_feature_detected!("avx2")` / `"avx512f"`

- `gpu.rs`: GPU スペック検出
  - macOS: Metal device API で `recommendedMaxWorkingSetSize`、`maxTransferRate`
  - CUDA (non-macOS): NVIDIA GPU スペック DB（RTX 20/30/40/50 + A/H シリーズ）

- `storage.rs`: NVMe スループット計測（キャッシュバイパス読み取り）
  - macOS: `F_NOCACHE` + `pread`
  - Linux/WSL2: `posix_fadvise(DONTNEED)` + `pread`
  - Windows: `std::io::Read + Seek`（後の Windows ポートで `FILE_FLAG_NO_BUFFERING` に更新予定）

- `mod.rs`: プロファイル集約、`data_dir()` パス解決
  - macOS/Linux: `~/.hypura/`
  - Windows: `%APPDATA%\Hypura\`

**設計決定:**
- プロファイル結果は JSON でキャッシュ: 毎起動の計測オーバーヘッドを避けるため
- NVIDIA GPU スペック DB: CUDA には Metal のような動的な帯域幅クエリ API がないため

---

## Phase 3: テンソル配置最適化

**実装内容: `src/scheduler/placement.rs`**

- LP（線形計画）+ greedy フォールバックによるテンソル → 階層割り当て
- `good_lp` クレート + HiGHS ソルバを使用
- テンソルスコアリング: `src/model/tensor_role.rs` でテンソルの役割（norm、attention、MoE expert、FFN等）を分類
- プラットフォーム別定数:
  ```rust
  #[cfg(target_os = "macos")] const OS_OVERHEAD: u64 = 2 * (1 << 30);     // 2 GB
  #[cfg(target_os = "windows")] const OS_OVERHEAD: u64 = 4 * (1 << 30);   // 4 GB
  #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))] const OS_OVERHEAD: u64 = 1 * (1 << 30);
  #[cfg(target_os = "macos")] const GPU_RUNTIME_OVERHEAD: u64 = 1 << 30;   // Metal: 1 GB
  #[cfg(not(target_os = "macos"))] const GPU_RUNTIME_OVERHEAD: u64 = 512 * (1 << 20); // CUDA: 512 MB
  ```

**設計決定:**
- Windows の OS_OVERHEAD を 4 GB に設定: VRAM オーバーレイドライバや DirectX ランタイムが macOS/Linux より多くのシステムメモリを消費するため

---

## Phase 4: NVMe バックエンド

**実装内容: `src/compute/nvme_backend.rs`**

- `IoPool`: ワーカースレッド群、各ワーカーが専用のダイレクト I/O ファイルディスクリプタを保持
- バリアベースの完了同期
- リージョンをワーカー間で分割して並列読み取り
- カスタム GGML バッファ型: llama.cpp のバッファアロケーションにフック、テンソルローディングをインターセプト
- `hypura_buft.c`: `platform_alloc_pages` / `platform_free_pages` でプラットフォーム抽象化

**設計決定:**
- ワーカーごとに個別の fd: macOS の `F_NOCACHE` は fd レベルの属性であり、スレッド間で共有不可のため
- ダイレクト I/O: OS ページキャッシュをバイパスすることで、ページキャッシュが LLM ウェイトで汚染されるのを防ぐ

---

## Phase 5: キャッシュ層

**実装内容:**

- `src/cache/coactivation.rs`: エキスパート共活性化追跡
  - 同一層・クロスレイヤー共活性化行列
  - `~/.hypura/coactivation/` への永続化
  - 投機的プリフェッチへの統合

- `src/cache/kv_cache.rs`: ウィンドウ付き KV キャッシュ圧縮
  - GPU バジェットが逼迫時の自動 Q8 選択
  - `llama_memory_seq_rm` によるウィンドウ圧縮

- `src/cache/neuron_cache.rs`: 読み込み済みエキスパートスライスの LRU キャッシュ
  - Mixtral での 99.5% ヒット率を達成

---

## Phase 6: 推論エンジン

**実装内容: `src/compute/inference.rs`**

- `generate_blocking`: ベースライン推論（mmap またはフル GPU 常駐）
- `generate_with_nvme_scheduling`: 階層推論（NVMe ストリーミング + GPU 常駐）
- サーバー向け: `load_model` / `generate_from_loaded`（モデルを常駐させて複数リクエストを処理）

**プラットフォーム対応:**
```rust
fn total_physical_memory() -> u64 {
    #[cfg(target_os = "macos")]
    { /* hw.memsize sysctl */ }
    #[cfg(not(target_os = "macos"))]
    { sysinfo::System::new_all().total_memory() }
}
```

---

## Phase 7: Ollama 互換サーバー

**実装内容: `src/server/routes.rs`, `src/cli/serve.rs`**

- Axum HTTP フレームワーク
- `POST /api/generate`, `POST /api/chat` でストリーミング NDJSON
- `GET /api/tags`, `GET /api/version`, `POST /api/show`
- CORS 対応（tower-http）

---

## Phase 8: `hypura optimize` — TSP エキスパートレイアウト最適化

**実装内容: `src/cli/optimize.rs`**

- Greedy TSP でエキスパートテンソルを共活性化順に並べ替え
- サイドカー `.permutations.json` で元ファイルを変更せずにレイアウトを記録
- 共活性化行列が蓄積された後に実行することで効果を最大化

---

## Phase 9: Windows/WSL2/CUDA ポート (2026-03-23)

**背景:** Hypura は当初 macOS/Apple Silicon 専用として設計されていた。Windows ネイティブ + WSL2 + CUDA（RTX 3060 以上）への対応を実施。

**詳細:** `windows-wsl2-port.md` を参照。

**主な変更ファイル:**

| ファイル | 変更内容 |
|---|---|
| `src/io/compat.rs` | 新規作成: プラットフォーム抽象化 API |
| `src/io/mod.rs` | `pub mod compat;` 追加 |
| `src/io/aligned_buffer.rs` | `std::alloc::Layout` ベースに書き換え（`posix_memalign` 廃止）|
| `hypura-sys/build.rs` | CUDA/Metal/CPU 対応の三分岐ビルドロジック |
| `hypura-sys/src/hypura_buft.c` | `#ifdef _WIN32` による `VirtualAlloc/VirtualFree` 対応 |
| `src/profiler/cpu.rs` | macOS sysctl / Linux procfs / Windows sysinfo の三分岐 |
| `src/profiler/gpu.rs` | NVIDIA GPU スペック DB 追加（RTX 20/30/40/50 + A/H シリーズ）|
| `src/profiler/storage.rs` | Windows `std::io::Read+Seek` フォールバック追加 |
| `src/compute/nvme_backend.rs` | 全 libc I/O を `compat` モジュール経由に置換 |
| `src/compute/inference.rs` | sysctl 呼び出しを非 macOS フォールバックに置換 |
| `src/io/async_reader.rs` | `compat` モジュール使用に書き換え |
| `src/cli/iobench.rs` | `compat` モジュール使用に書き換え |
| `src/scheduler/placement.rs` | OS_OVERHEAD / GPU_RUNTIME_OVERHEAD を OS 別に分岐 |
| `Cargo.toml` | `windows-sys` 条件付き依存関係を追加 |
