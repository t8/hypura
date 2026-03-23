# Cursor 引き継ぎ書 — hypura Windows ビルド & OpenClaw 統合

**作成日:** 2026-03-23
**作成AI:** Claude Sonnet 4.6
**対象リポジトリ:** `C:\Users\downl\Desktop\hypura-main\hypura-main`
**GitHub:** `https://github.com/zapabob/hypura` (fork) / upstream: `https://github.com/t8/hypura`

---

## 現状サマリー

| 項目 | 状態 |
|------|------|
| Windows ビルド | 🔧 修正中 (cmake ツール無効化で再ビルド必要) |
| PR #2 (upstream) | ✅ オープン中 |
| OpenClaw 統合 | ✅ Skill + Provider Extension 作成済み |
| CUDA 実機 | RTX 3060 + RTX 3080 (sm_86) 搭載済み |

---

## ビルドエラーの原因と修正

### 原因
`hypura-sys/build.rs` で `LLAMA_BUILD_TOOLS=OFF` が未設定だったため、
llama.cpp の `tools/mtmd/` 配下のマルチモーダル CLI ツール
(llama-llava-cli, llama-gemma3-cli, llama-minicpmv-cli, llama-qwen2vl-cli) が
ビルドされ、Windows で依存ライブラリ不足でコケていた。

### 修正済み (build.rs:21)
```rust
// hypura-sys/build.rs
cmake_config
    .define("LLAMA_BUILD_TESTS", "OFF")
    .define("LLAMA_BUILD_EXAMPLES", "OFF")
    .define("LLAMA_BUILD_TOOLS", "OFF")   // ← 追加済み
    .define("LLAMA_BUILD_SERVER", "OFF")
```

### 次のアクション
```sh
# target/debug/build/hypura-sys-* を削除してクリーンビルド
cargo clean -p hypura-sys
cargo build
```

---

## 環境設定 (Windows)

### `.cargo/config.toml` (git 管理外)
```toml
[build]
rustc-wrapper = "sccache"
jobs = 6

[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "incremental=yes"]

[env]
LIBCLANG_PATH = "C:\\Program Files\\LLVM\\bin"
HYPURA_CUDA_ARCHITECTURES = "86"   # RTX 3060/3080 専用 sm_86
```

### 必要ツール
- LLVM: `winget install LLVM.LLVM` (libclang.dll が必要)
- CUDA Toolkit: RTX 3060/3080 対応版
- sccache: `cargo install sccache`
- Visual Studio 2022 (MSBuild)

---

## 実装済み変更一覧

### hypura-sys/build.rs
1. `dunce::canonicalize` で Windows UNC パス (`\\?\`) 問題を修正
2. `LIBCLANG_PATH` による libclang 検出 (`.cargo/config.toml` 経由)
3. pre-generated bindings fallback:
   - `HYPURA_PREGENERATED_BINDINGS=/path/to/bindings.rs` env var
   - または `hypura-sys/bindings.rs` をソースツリーに置く
4. `LLAMA_BUILD_TOOLS=OFF` でマルチモーダルCLI無効化 ← 今回追加

### .gitignore
```gitignore
.cargo/config.toml    # machine-specific (LIBCLANG_PATH, CUDA arch)
.claude/              # Claude Code local settings
```

---

## PR #2 (t8/hypura) の状態

**URL:** https://github.com/t8/hypura/pull/2
**ブランチ:** `zapabob:windows-cuda-port` → `t8/hypura:main`
**内容:** Windows/WSL2/Linux + CUDA クロスプラットフォームポート
**状態:** オープン、レビュー待ち

### PR に含まれていない修正 (ローカルのみ)
- `LLAMA_BUILD_TOOLS=OFF` ← **PR に push すること**
- `.cargo/config.toml` は機械固有なので PR に含めない

---

## OpenClaw 統合

**clawdbot パス:** `C:\Users\downl\Desktop\clawdbot-main3\clawdbot-main`
**引き継ぎドキュメント:** `_docs/2026-03-23_hypura-provider-integration_claude-sonnet-4-6.md`

### 作成済みファイル
| ファイル | 説明 |
|---------|------|
| `skills/hypura/SKILL.md` | hypura CLI コマンド・OpenClaw 連携手順 |
| `extensions/hypura-provider/index.ts` | Ollama 互換 API プロバイダー (自動検出) |
| `extensions/hypura-provider/package.json` | `@openclaw/hypura-provider` |
| `extensions/hypura-provider/openclaw.plugin.json` | OpenClaw プラグインメタデータ |

### 使い方
```sh
# hypura ビルド後
hypura serve --model ./model.gguf --port 8080

# OpenClaw が http://127.0.0.1:8080 を自動検出
# または手動設定
openclaw config set models.providers.hypura.baseUrl=http://127.0.0.1:8080
```

---

## hypura serve API (Ollama 互換)

| エンドポイント | メソッド | 説明 |
|--------------|--------|------|
| `/` | GET | ヘルスチェック |
| `/api/tags` | GET | ロード済みモデル一覧 |
| `/api/generate` | POST | テキスト生成 (NDJSON ストリーミング) |
| `/api/chat` | POST | チャット補完 (NDJSON ストリーミング) |

---

## ベンチマーク結果 (M1 Max 32GB 参考値)

| モデル | モード | tok/s |
|--------|--------|-------|
| Mixtral 8x7B Q5_K_M (30.9 GB) | expert-streaming | 2.19 |
| Llama 3.3 70B Q4_K_M (39.6 GB) | dense-FFN-streaming | 0.30 |

---

## TODO (優先順)

- [ ] `cargo clean -p hypura-sys && cargo build` でビルド通過確認
- [ ] `LLAMA_BUILD_TOOLS=OFF` を PR #2 ブランチに push
- [ ] `hypura serve` 実動作テスト (RTX 3060/3080)
- [ ] bindings.rs を生成してコミット (LLVM不要ビルドのため)
- [ ] clawdbot: `pnpm check` で TypeScript 型確認
- [ ] clawdbot: `docs/providers/hypura.mdx` 作成
- [ ] C ドライブ空き容量回復: AppData/Temp (14GB) + npm-cache (5.3GB) 削除

---

## Cursor 実施ログ追記 (Codex)

### 実施日時
- 2026-03-23 (JST)

### このセッションで実際に行ったこと
1. `windows-cuda-port` ブランチで `LLAMA_BUILD_TOOLS=OFF` を確認し、PR差分を整理
2. `.gitignore` にローカル生成物除外を追加
   - `yes/`
   - `target-codex/`
3. Conventional Commit でコミット作成・push 実施
   - commit: `1fac4fd`
   - message: `fix(build): disable llama tools on Windows`
4. PR #2 へ進捗コメントを投稿
   - https://github.com/t8/hypura/pull/2#issuecomment-4111136329
5. Cドライブ掃除を実行し、空き容量を回復
   - Temp: 13.479 GB -> 0.126 GB
   - npm cache: 4.977 GB -> 0 GB
   - 合計回復: 約 18.33 GB
6. `hypura serve` 実行のために複数のビルド不整合を暫定修正
   - `hypura-sys/build.rs`: bindgen layout test を無効化 (`layout_tests(false)`)
   - `src/io/compat.rs`: Windows Handle 型まわりを `windows-sys` 0.59 系に合わせて調整
   - `src/profiler/cpu.rs`: `physical_cpu_count` -> `physical_core_count`

### 追加で判明したブロッカー
- `cargo` が頻繁に `Blocking waiting for file lock on artifact directory` で停滞し、
  同一 run の終了ステータスや最終行が安定して取得できない。
- その結果、`hypura-sys/build.rs` に入れたデバッグログが再現 run で生成されず、
  CRT 不整合 (`__imp__CrtDbgReport`, `__imp__calloc_dbg`) の改善効果を
  連続した新規証拠で確定できていない。
- 取得できる `LNK2001` 群は古い terminal 出力に残っている証拠が中心。

### 重要な現状
- `E0515` (`cannot return value referencing function parameter s`) は該当コードの
  所有権修正 (`map(|l| l.to_string())`) でコード上は解消済み。
- ただしビルド全体は lock 競合で観測しづらく、最終 `Finished` の安定確認は未完。

### 次に ClaudeCode で優先すべき手順
1. まず build 系プロセスを全停止 (`cargo/rustc/cmake/msbuild/sccache/hypura`)
2. 単一 PowerShell で環境変数を設定して実行
   - `$env:HYPURA_DEBUG_RUN_ID="<new-id>"`
   - `$env:CARGO_TARGET_DIR="target-codex-<new-id>"`
3. `cargo clean -p hypura-sys` -> `cargo run --release -- serve --model "F:\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf" --port 8080`
4. `hypura-sys/debug-4ee339.log` (or 新 run id のログ) が生成されることを最優先確認
5. 生成されたログで以下を確認
   - `PROFILE` / target env
   - CMake cache の `CMAKE_BUILD_TYPE`
   - `CMAKE_CONFIGURATION_TYPES`
   - `CMAKE_MSVC_RUNTIME_LIBRARY`
6. その後に `/`, `/api/tags`, `/api/generate` の順で smoke
