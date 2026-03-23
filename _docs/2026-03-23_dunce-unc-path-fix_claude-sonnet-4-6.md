# 2026-03-23 dunce クレートによる Windows UNC パス問題修正

**実装AI:** Claude Sonnet 4.6
**日付:** 2026-03-23
**カテゴリ:** バグ修正・Windows ビルド

---

## 症状

Windows ネイティブ環境で `cargo check` を実行すると `hypura-sys` のビルドが失敗:

```
error C1083: ソース ファイルを開けません。
'\\?\C:\Users\...\vendor\llama.cpp\ggml\src\gguf.cpp': No such file or directory
```

ファイルは実際に存在しており、パスの `\\?\` プレフィックスが問題の原因。

---

## 根本原因

`hypura-sys/build.rs` の以下のコード:

```rust
let llama_dir = PathBuf::from(&manifest_dir).join("../vendor/llama.cpp");
let llama_dir = llama_dir.canonicalize().expect("...");
```

**`std::fs::canonicalize()` は Windows で `\\?\C:\...` 形式の拡張長パス (Extended-Length Path) を返す。**

これが CMake の生成する `*.vcxproj` ファイル内のソースパスとして埋め込まれると、MSBuild がそのパスを解釈できずにソースファイルが見つからないと判断する。

```
通常パス:  C:\Users\...\vendor\llama.cpp\ggml\src\gguf.cpp  ← MSBuild OK
UNC パス: \\?\C:\Users\...\vendor\llama.cpp\ggml\src\gguf.cpp ← MSBuild NG
```

---

## 修正内容

### `hypura-sys/Cargo.toml`

```toml
[build-dependencies]
cmake = "0.1"
cc = "1"
bindgen = "0.71"
dunce = "1"          # 追加
```

### `hypura-sys/build.rs`

```rust
// Before:
let llama_dir = llama_dir.canonicalize().expect("...");

// After:
// dunce::canonicalize strips the \\?\ UNC prefix that std::fs::canonicalize
// adds on Windows, which would otherwise cause MSBuild to reject source paths.
let llama_dir = dunce::canonicalize(&llama_dir).expect("...");
```

---

## `dunce` クレートについて

[`dunce`](https://crates.io/crates/dunce) は Windows の `\\?\` UNC プレフィックスを通常のパスに変換するユーティリティ。

```rust
// dunce::canonicalize の動作
// Windows: C:\foo\bar  (\\?\ を除去)
// Unix:    /foo/bar    (std::fs::canonicalize と同じ)
```

macOS/Linux では `std::fs::canonicalize` と同一の動作をするため、クロスプラットフォーム安全。

---

## 影響範囲

`llama_dir` を起点に生成される全パス（`include`、`ggml/src`、`vendor/cpp-httplib` 等）が正しい形式になる。bindgen の `-I` フラグや `cc::Build` の `.include()` パスも同様に修正される。

---

## 関連する環境問題

### Avast アンチウイルスのブロック

この修正とは別に、Avast の Real-Time Protection が新しくコンパイルされた `.exe`（cargo build scripts）の実行をブロックする問題が発生した。

**症状:**
```
error: failed to run custom build command for `proc-macro2 v1.0.106`
アクセスが拒否されました。 (os error 5)
```

**解決策:** Avast の除外設定にプロジェクトの `target\` ディレクトリを追加:
```
C:\Users\<user>\Desktop\hypura-main\hypura-main\target\
```

---

## 検証

`dunce` 修正後の `cargo check`:
- `highs-sys` (HiGHS LP ソルバ): ビルド成功
- `hypura-sys` (llama.cpp + CUDA): CMake 設定成功、MSBuild 実行中
- 全 Rust クレート (~220): チェック完了
