# 2026-03-23 Windows/WSL2/CUDA クロスプラットフォーム移植

**実装AI:** Claude Sonnet 4.6
**日付:** 2026-03-23
**カテゴリ:** アーキテクチャ移植・プラットフォーム対応

---

## 背景・動機

Hypura はもともと macOS/Apple Silicon 専用として設計されており、以下の macOS 固有 API に強依存していた:

- `F_NOCACHE` (キャッシュバイパス I/O)
- `pread(2)` (オフセット指定読み取り)
- `mmap(MAP_ANON|MAP_PRIVATE)` / `munmap` / `madvise(MADV_FREE)` (匿名メモリ管理)
- `sysctlbyname` (ハードウェア情報取得)
- `posix_memalign` (アライメント付きメモリ確保)
- Metal GPU API

ユーザーが Windows (RTX 3060+) / WSL2 環境での動作を要望し、**macOS サポートを維持したまま**クロスプラットフォーム対応を実施した。

---

## 実装内容

### 1. `src/io/compat.rs` — プラットフォーム抽象化レイヤー (新規作成)

全プラットフォーム固有 I/O を単一 API の背後に隠蔽する核心ファイル。

```
上位レイヤー (IoPool / NvmePrefetcher / iobench)
    ↓
src/io/compat.rs [統一 API]
    ↓ 条件コンパイル分岐
macOS impl | Linux/WSL2 impl | Windows impl
```

**実装した API:**

| 関数 | macOS | Linux/WSL2 | Windows |
|---|---|---|---|
| `open_direct_fd` | `F_NOCACHE` + `fcntl` | `O_DIRECT` | `FILE_FLAG_NO_BUFFERING` + `CreateFileW` |
| `read_at_fd` | `pread(2)` | `pread(2)` | `ReadFile` + `OVERLAPPED` |
| `alloc_pages` | `mmap(MAP_ANON)` | `mmap(MAP_ANON)` | `VirtualAlloc` |
| `free_pages` | `munmap` | `munmap` | `VirtualFree(MEM_RELEASE)` |
| `advise_free_pages` | `madvise(MADV_FREE)` | `madvise(MADV_DONTNEED)` | `VirtualFree(MEM_DECOMMIT)` |

**型エイリアス:**
```rust
#[cfg(unix)]   pub type NativeFd = i32;
#[cfg(windows)] pub type NativeFd = isize; // HANDLE
```

### 2. `src/io/aligned_buffer.rs` — クロスプラットフォーム書き換え

`posix_memalign`（POSIX 専用）→ `std::alloc::Layout` + `std::alloc::alloc/dealloc`（全 OS 対応）

```rust
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}
```

`FILE_FLAG_NO_BUFFERING` はバッファアドレス/サイズがセクターサイズ（4096 バイト）アライメントを要求するため、`AlignedBuffer::new(size, 4096)` がこれを保証する。

### 3. `hypura-sys/build.rs` — CMake ビルドの三分岐

```
target_os == "macos"  → Metal (GGML_METAL=ON)
非 macOS + CUDA 検出  → CUDA (GGML_CUDA=ON, sm_75;86;89;90)
非 macOS + CUDA なし  → CPU のみ
```

**CUDA 検出順序:**
1. 環境変数 `CUDA_PATH`
2. `/usr/local/cuda`
3. `/usr/cuda`
4. PATH 内の `nvcc` の親ディレクトリ

**CUDA アーキテクチャ:**
```
sm_75 → RTX 20xx (Turing)
sm_86 → RTX 30xx (Ampere): RTX 3060, 3070, 3080, 3090
sm_89 → RTX 40xx (Ada): RTX 4070, 4080, 4090
sm_90 → H100 (Hopper)
```
`HYPURA_CUDA_ARCHITECTURES=86` 環境変数で特定アーキテクチャのみビルド可能。

### 4. `hypura-sys/src/hypura_buft.c` — Windows メモリ対応

```c
#ifdef _WIN32
static void *platform_alloc_pages(size_t size) {
    return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}
static void platform_free_pages(void *addr, size_t size) {
    (void)size; if (addr) VirtualFree(addr, 0, MEM_RELEASE);
}
#else
static void *platform_alloc_pages(size_t size) {
    void *p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, -1, 0);
    return (p == MAP_FAILED) ? NULL : p;
}
#endif
```

### 5. `src/profiler/cpu.rs` — CPU 検出の三分岐

| OS | モデル名取得 | コア数取得 |
|---|---|---|
| macOS | `sysctlbyname("hw.model")` | `sysctlbyname("hw.physicalcpu")` |
| Linux | `/proc/cpuinfo` | `sysinfo` クレート |
| Windows | `sysinfo` クレート | `sysinfo` クレート |

x86_64: `std::is_x86_feature_detected!("avx2")` / `"avx512f"` で SIMD 検出

### 6. `src/profiler/gpu.rs` — NVIDIA GPU スペック DB 追加

CUDA には Metal のような動的帯域幅クエリ API がないため、既知 GPU の理論値テーブルをハードコード:

```rust
NvidiaSpec { pattern: "RTX 3080 Ti",   bandwidth_gb_s: 912.0, fp16_tflops: 65.0 },
NvidiaSpec { pattern: "RTX 3080 12GB", bandwidth_gb_s: 912.0, fp16_tflops: 60.0 },
NvidiaSpec { pattern: "RTX 3080",      bandwidth_gb_s: 760.0, fp16_tflops: 59.0 },
NvidiaSpec { pattern: "RTX 3060",      bandwidth_gb_s: 360.0, fp16_tflops: 25.4 },
// ... RTX 20/30/40/50, A100, H100, L40S
```

### 7. `src/scheduler/placement.rs` — OS 別オーバーヘッド定数

```rust
#[cfg(target_os = "macos")]   const OS_OVERHEAD: u64 = 2 * (1 << 30); // 2 GB
#[cfg(target_os = "windows")] const OS_OVERHEAD: u64 = 4 * (1 << 30); // 4 GB (VRAM ドライバ等)
#[cfg(all(not(...), not(...)))] const OS_OVERHEAD: u64 = 1 * (1 << 30); // 1 GB
#[cfg(target_os = "macos")]   const GPU_RUNTIME_OVERHEAD: u64 = 1 << 30; // Metal: 1 GB
#[cfg(not(target_os = "macos"))] const GPU_RUNTIME_OVERHEAD: u64 = 512 * (1 << 20); // CUDA: 512 MB
```

### 8. `Cargo.toml` — Windows 条件付き依存関係

```toml
[target.'cfg(windows)'.dependencies]
windows-sys = { version = "0.59", features = [
    "Win32_Foundation", "Win32_Storage_FileSystem",
    "Win32_System_IO", "Win32_System_Memory",
] }
```

### 9. 各モジュールの compat 移行

| ファイル | 変更内容 |
|---|---|
| `src/compute/nvme_backend.rs` | 全 libc I/O → `compat` モジュール、`NativeFd` 型使用 |
| `src/io/async_reader.rs` | `F_NOCACHE` + `pread` → `compat::open_direct_fd` + `read_at_fd` |
| `src/cli/iobench.rs` | 全テストバリアントを `compat` ベースに書き換え |
| `src/compute/inference.rs` | `sysctl` 呼び出し → `sysinfo` クレートベースに非 macOS 分岐 |
| `src/profiler/storage.rs` | Unix/Windows ストレージ計測を分岐 |
| `src/profiler/mod.rs` | `data_dir()`: Windows `%APPDATA%\Hypura` / Unix `~/.hypura` |

---

## 動作確認

- CUDA Toolkit 検出: RTX 3060 (sm_86) で確認
- `highs-sys` (HiGHS LP ソルバ): Windows MSVC でビルド成功
- Rust コードレベル警告: 0 件

---

## 既知の制限事項

- `vendor/llama.cpp` サブモジュール初期化後に `git submodule update --init --recursive` が必要
- AMD ROCm / Intel Arc 未対応（将来対応可能な設計）
- WSL2 側は `posix_fadvise(POSIX_FADV_DONTNEED)` パス（Linux コードパス）
