# Windows/WSL2 + CUDA ポート 詳細設計ドキュメント

**実施日:** 2026-03-23
**担当:** LLM-directed development (Claude Sonnet 4.6)

---

## 動機

Hypura は当初 macOS/Apple Silicon 専用として設計されていた。具体的には以下の macOS 固有 API に強く依存していた:

- `F_NOCACHE` フラグ（キャッシュバイパス I/O）
- `pread(2)` システムコール（オフセット指定読み取り）
- `mmap(MAP_ANON|MAP_PRIVATE)` / `munmap`（匿名メモリ）
- `madvise(MADV_FREE)`（ページ解放ヒント）
- `sysctlbyname`（ハードウェア情報取得）
- `posix_memalign`（アライメント付きメモリ確保）
- Metal GPU API

目標: **Windows ネイティブおよび WSL2 上での CUDA (RTX 3060 以上) での動作**を、既存の macOS サポートを破壊することなく実現する。

---

## 設計方針

### 1. 単一の抽象化レイヤー

すべてのプラットフォーム固有 I/O を `src/io/compat.rs` の単一 API の背後に隠蔽する。上位レイヤー（IoPool、NvmePrefetcher、iobench）はこの API のみを使用する。

```
IoPool / NvmePrefetcher / iobench
    ↓ uses
src/io/compat.rs  [NativeFd, open_direct_fd, read_at_fd, alloc_pages, ...]
    ↓ branches to
macOS impl  |  Linux/WSL2 impl  |  Windows impl
```

### 2. 条件コンパイルの粒度

- `#[cfg(target_os = "macos")]` / `#[cfg(unix)]` / `#[cfg(windows)]` を `compat.rs` 内に集約
- 上位レイヤーでの `#[cfg]` ブロックは最小限に抑える（`placement.rs` のオーバーヘッド定数のみ）

### 3. 型安全な fd ラッパー

```rust
#[cfg(unix)]
pub type NativeFd = i32;

#[cfg(windows)]
pub type NativeFd = isize; // HANDLE
```

`-1` (Unix の無効 fd) と `INVALID_HANDLE_VALUE` (Windows の無効ハンドル) を型レベルで区別。

---

## プラットフォーム抽象化 API (`src/io/compat.rs`)

### `open_direct_fd(path: &Path) -> std::io::Result<NativeFd>`

キャッシュバイパスモードでファイルを開く。

| OS | 実装 |
|---|---|
| macOS | `open(O_RDONLY)` + `fcntl(F_NOCACHE, 1)` |
| Linux/WSL2 | `open(O_RDONLY \| O_DIRECT)` |
| Windows | `CreateFileW(FILE_FLAG_NO_BUFFERING \| FILE_FLAG_OVERLAPPED)` |

### `close_fd(fd: NativeFd)`

ファイルディスクリプタ/ハンドルを閉じる。

| OS | 実装 |
|---|---|
| Unix | `libc::close(fd)` |
| Windows | `CloseHandle(fd as HANDLE)` |

### `read_at_fd(fd: NativeFd, dst: *mut u8, size: usize, offset: u64) -> isize`

指定オフセットから `size` バイト読み取る（非シーク型、スレッドセーフ）。

| OS | 実装 |
|---|---|
| Unix | `libc::pread(fd, dst, size, offset)` |
| Windows | `ReadFile` + `OVERLAPPED { Offset, OffsetHigh }` |

**Windows の注意点:** `FILE_FLAG_NO_BUFFERING` を使用する場合、バッファのアドレスとサイズはセクターサイズ（通常 512 バイトまたは 4096 バイト）のアライメントが必要。`AlignedBuffer` が 4096 バイトアライメントを保証する。

### `alloc_pages(size: usize) -> *mut u8`

匿名メモリページを確保する（NVMe バッファ用）。

| OS | 実装 |
|---|---|
| macOS/Linux | `mmap(NULL, size, PROT_READ\|PROT_WRITE, MAP_ANON\|MAP_PRIVATE, -1, 0)` |
| Windows | `VirtualAlloc(NULL, size, MEM_COMMIT \| MEM_RESERVE, PAGE_READWRITE)` |

### `free_pages(ptr: *mut u8, size: usize)`

| OS | 実装 |
|---|---|
| Unix | `munmap(ptr, size)` |
| Windows | `VirtualFree(ptr, 0, MEM_RELEASE)` |

### `advise_free_pages(ptr: *mut u8, size: usize)`

ページを OS に返却するヒントを与える（レイヤー解放後に呼び出す）。

| OS | 実装 |
|---|---|
| macOS | `madvise(ptr, size, MADV_FREE)` |
| Linux/WSL2 | `madvise(ptr, size, MADV_DONTNEED)` |
| Windows | `VirtualFree(ptr, size, MEM_DECOMMIT)` |

**重要:** `MEM_DECOMMIT` は物理ページをコミット解除するが仮想アドレス範囲は保持する。次回アクセス時にページフォルトが発生し、OS がゼロページを割り当てる。これが `MADV_FREE`/`MADV_DONTNEED` のセマンティクスに最も近い Windows の等価物。

---

## AlignedBuffer の書き換え

**変更前:** `posix_memalign`（POSIX 専用）
**変更後:** `std::alloc::Layout` + `std::alloc::alloc` / `dealloc`（クロスプラットフォーム）

```rust
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    pub fn new(len: usize, alignment: usize) -> std::io::Result<Self> {
        let layout = Layout::from_size_align(len, alignment)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(std::io::Error::new(std::io::ErrorKind::OutOfMemory, "alloc failed"));
        }
        Ok(Self { ptr, len, layout })
    }
}
```

---

## CUDA ビルド設定 (`hypura-sys/build.rs`)

### 検出フロー

```
target_os == "macos"
  → Metal ビルド (GGML_METAL=ON, framework Metal/Foundation/QuartzCore)
else
  → CUDA 検出:
    1. env CUDA_PATH
    2. /usr/local/cuda (Linux/WSL2 デフォルト)
    3. /usr/cuda
    4. PATH 内の nvcc の親ディレクトリ
  → CUDA 利用可能
    → GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES="75;86;89;90"
  → CUDA 不可
    → CPU のみ (GGML_CUDA=OFF, GGML_METAL=OFF)
```

### CUDA アーキテクチャ

| GPU シリーズ | sm_ | 代表例 |
|---|---|---|
| RTX 20xx (Turing) | sm_75 | RTX 2060, 2070, 2080 |
| RTX 30xx (Ampere) | sm_86 | RTX 3060, 3070, 3080, 3090 |
| RTX 40xx (Ada) | sm_89 | RTX 4070, 4080, 4090 |
| H100 (Hopper) | sm_90 | H100 SXM/PCIe |
| A100 (Ampere) | sm_80 | A100 |
| L40S (Ada) | sm_89 | L40S |

デフォルト: `"75;86;89;90"`（RTX 2060 以上のすべてをカバー）

カスタマイズ: `HYPURA_CUDA_ARCHITECTURES=86` 環境変数で特定のアーキテクチャのみビルド（ビルド時間短縮）

---

## NVIDIA GPU スペック DB (`src/profiler/gpu.rs`)

CUDA には Metal のような動的な GPU 帯域幅クエリ API がない。そのため、既知の GPU モデルの理論値テーブルをハードコードしている。

`lookup_nvidia_gpu(name: &str) -> Option<(bandwidth_bytes_per_sec, fp16_tflops)>`

名前マッチングは部分文字列検索（`contains`）で行う。未知の GPU は `estimate_nvidia_gpu(vram_bytes)` で VRAM 容量から推定値を返す。

**RTX 3060 の仕様:**
- メモリ帯域幅: 360 GB/s
- FP16 演算性能: 25.4 TFLOPS

---

## Windows NVMe ストリーミング 動作フロー

```
1. IoPool::new(model_path, num_workers)
   → compat::open_direct_fd(model_path)
      → Windows: CreateFileW(..., FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, ...)
   → 各ワーカースレッドが専用 HANDLE を保持

2. IoPool::read_region(file_offset, size)
   → リージョンを num_workers に分割
   → 各ワーカーが担当領域を pread_region() で読み取り
      → compat::read_at_fd(handle, dst, size, offset)
         → Windows:
            OVERLAPPED ol = { .Offset = offset & 0xFFFFFFFF,
                               .OffsetHigh = offset >> 32 };
            ReadFile(handle, dst, size, &bytes_read, &ol);
            GetOverlappedResult(handle, &ol, &bytes_read, TRUE); // 同期待機

3. バリアで全ワーカーの完了を待機

4. 推論完了後のレイヤー解放:
   → compat::advise_free_pages(ptr, size)
      → Windows: VirtualFree(ptr, size, MEM_DECOMMIT)
         → 物理ページを OS に返却、仮想アドレスは保持
         → 次回アクセス時にゼロページが割り当てられ再読み込み
```

---

## 既知の制限事項

1. **`cargo test --lib` の一部テストが GGUF ファイルを必要とする** — テストモデルは `test-models/` に配置する必要がある（リポジトリには含まれない）

2. **WSL2 での `vendor/llama.cpp` サブモジュール** — 初回ビルド前に `git submodule update --init --recursive` が必要

3. **Windows ネイティブでの `FILE_FLAG_NO_BUFFERING` のアライメント要件** — 読み取りサイズとバッファアドレスはセクターサイズの倍数である必要がある。`AlignedBuffer::new(size, 4096)` がこれを保証している。ただし、リードサイズがセクターサイズに満たない場合（最後のチャンク等）は内部でパディングが発生する

4. **CUDA のみ対応** — AMD ROCm / Intel Arc は未対応（将来対応可能な設計にはなっている）
