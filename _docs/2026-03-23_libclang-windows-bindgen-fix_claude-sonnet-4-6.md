# 2026-03-23 Windows 環境での libclang 未検出エラー修正 & pre-generated bindings サポート

**実装AI:** Claude Sonnet 4.6
**日付:** 2026-03-23
**カテゴリ:** バグ修正・Windows ビルド

---

## 症状

Windows ネイティブ環境で `cargo build` を実行すると `hypura-sys` のビルドが失敗:

```
thread 'main' panicked at ...bindgen...
Unable to find libclang: "couldn't find any valid shared libraries matching:
['clang.dll', 'libclang.dll'], searching paths: [...]"
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
warning: build failed, waiting for other jobs to finish...
```

---

## 根本原因

`hypura-sys/build.rs` の bindgen (v0.71.1) は `clang-sys` クレート経由で実行時に
`clang.dll` / `libclang.dll` を動的ロードする。

Windows に LLVM/Clang がインストールされておらず、かつ `.cargo/config.toml` に
`LIBCLANG_PATH` が設定されていなかったため、パニックしてビルドが中断していた。

---

## 修正内容

### 1. `.cargo/config.toml` に `LIBCLANG_PATH` を追加

**ファイル:** `.cargo/config.toml`

```toml
[env]
LIBCLANG_PATH = "C:\\Program Files\\LLVM\\bin"
```

`winget install LLVM.LLVM` でインストールされる標準パスを指定。
これにより `bindgen` が `libclang.dll` を自動検出できるようになる。

### 2. `hypura-sys/build.rs` に pre-generated bindings fallback を追加

**ファイル:** `hypura-sys/build.rs` — bindgen セクション (旧行 123〜171)

bindgen を呼ぶ前に以下の優先順で既存の bindings.rs を探す仕組みを追加:

1. 環境変数 `HYPURA_PREGENERATED_BINDINGS=/path/to/bindings.rs` が設定されている場合
2. `hypura-sys/bindings.rs` がソースツリーに存在する場合
3. 上記いずれも無ければ従来通り bindgen で生成

```rust
let pregenerated = env::var("HYPURA_PREGENERATED_BINDINGS")
    .map(PathBuf::from)
    .ok()
    .or_else(|| {
        let p = PathBuf::from(&manifest_dir).join("bindings.rs");
        if p.exists() { Some(p) } else { None }
    });

if let Some(src) = pregenerated {
    std::fs::copy(&src, out_path.join("bindings.rs"))
        .expect("Failed to copy pre-generated bindings");
    println!("cargo:warning=Using pre-generated bindings from {}", src.display());
} else {
    // ... bindgen::Builder::default()...generate() ...
}
```

また、bindgen 失敗時のパニックメッセージを改善:

```
Failed to generate bindings — install LLVM and set LIBCLANG_PATH,
or provide HYPURA_PREGENERATED_BINDINGS=/path/to/bindings.rs
```

---

## 運用フロー (pre-generated bindings をコミットする場合)

LLVM を一度インストールしてビルドが通った後、以下の手順で bindings.rs をコミットしておくと
LLVM なしの環境 (CI / 他開発者) でもビルド可能になる:

```sh
# OUT_DIR を調べる
cargo build --message-format=json 2>/dev/null \
  | grep -o '"out_dir":"[^"]*"' | head -1

# bindings.rs をソースツリーにコピー
cp <OUT_DIR>/bindings.rs hypura-sys/bindings.rs

# コミット
git add hypura-sys/bindings.rs
git commit -m "feat(build): commit pre-generated bindings for LLVM-free builds"
```

---

## 影響範囲

| ファイル | 変更種別 | 内容 |
|----------|----------|------|
| `.cargo/config.toml` | 追加 | `[env] LIBCLANG_PATH` セクション |
| `hypura-sys/build.rs` | 変更 | pre-generated bindings fallback ロジック + エラーメッセージ改善 |

macOS / Metal ビルドへの影響なし。CUDA ビルドへの影響なし。
既存の bindgen による生成フローは変更なし (libclang が存在する場合は従来通り動作)。

---

## 前提条件

- LLVM インストール: `winget install LLVM.LLVM`
- インストール先: `C:\Program Files\LLVM\` (デフォルト)
- `libclang.dll` の場所: `C:\Program Files\LLVM\bin\libclang.dll`
