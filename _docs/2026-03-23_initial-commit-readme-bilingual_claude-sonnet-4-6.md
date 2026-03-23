# 2026-03-23 初期コミット・README 日英併記リライト・_docs 作成

**実装AI:** Claude Sonnet 4.6
**日付:** 2026-03-23
**カテゴリ:** ドキュメント・Git 管理

---

## 背景・動機

Windows/WSL2 + CUDA 移植完了後、リポジトリを `zapabob/hypura` に公開するにあたり:
1. Git リポジトリを初期化してコミット可能な状態にする
2. README を macOS 限定から全プラットフォーム対応に更新し、日英併記で書き直す
3. 設計決定・実装詳細を `_docs/` に記録する
4. 型定義警告を 0 件にする

---

## 実装内容

### 1. 型警告 0 修正

**`src/compute/nvme_backend.rs`**
- `#[cfg(unix)] use std::os::unix::io::IntoRawFd;` を削除
- `IntoRawFd` は `src/io/compat.rs` に移動済みで `nvme_backend.rs` では不使用

**`src/cli/iobench.rs`**
- `fn read_sequential(...)` (dead function) を削除 — 全テスト関数が直接 `read_full` を呼ぶため
- `test_mt_nocache` 内の `struct RawPtr(*mut u8)` インライン構造体ハックを削除
  - Before: `unsafe { struct RawPtr(*mut u8); ... RawPtr(my_buf) }.as_mut_ptr()`
  - After: 直接 `my_buf` を使用

### 2. `.gitignore` 作成

```
/target/
vendor/llama.cpp/build/
vendor/llama.cpp/build-*/
benchmarks/results/*.json
test-models/
*.gguf
.DS_Store
```

### 3. `.gitmodules` 作成

```ini
[submodule "vendor/llama.cpp"]
    path = vendor/llama.cpp
    url = https://github.com/ggerganov/llama.cpp.git
    branch = master
```

### 4. Git 初期化・コミット

```sh
git init
git remote add origin https://github.com/zapabob/hypura.git
git add ...  # 79 ファイル
git commit -m "feat: initial commit — cross-platform LLM inference scheduler"
```

コミット結果: **79 ファイル、16,482 行**

### 5. README.md 全面リライト (日英併記)

**構造:**
```
[ASCII art + bilingual tagline]
日本語セクション (概要 / なぜ必要か / 動作原理 / NVMe / パフォーマンス / インストール / ...)
---
English section (Overview / Why / How / NVMe / Performance / Install / ...)
```

**主な更新内容:**
- "Apple Silicon のみ" → 4 プラットフォーム対応表（macOS Metal / Windows CUDA / WSL2 CUDA / Linux CUDA）
- Windows NVMe セクション新設: `FILE_FLAG_NO_BUFFERING` + `ReadFile(OVERLAPPED)` = macOS `F_NOCACHE` + `pread`
- プラットフォーム別インストール手順（macOS / WSL2 / Windows ネイティブ）
- CUDA アーキテクチャ説明（sm_86 = RTX 3060 ベース）
- `hypura iobench` コマンド追加

### 6. `_docs/` 実装ログ作成

| ファイル | 内容 |
|---|---|
| `_docs/README.md` | ディレクトリ索引（日英） |
| `_docs/implementation-log.md` | Phase 1〜9 の時系列実装ログ |
| `_docs/windows-wsl2-port.md` | Windows ポート詳細設計（compat API 仕様、CUDA 検出、NVMe フロー） |

---

## 成果物

- `git log --oneline`: `de08889 feat: initial commit — cross-platform LLM inference scheduler`
- `git remote -v`: `origin https://github.com/zapabob/hypura.git`
- Rust コードレベル警告: 0 件（cargo check フィルタ確認済み）
