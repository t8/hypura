# _docs — Hypura 実装ドキュメント / Implementation Documentation

このディレクトリには Hypura の設計決定・実装詳細・移植記録が含まれています。

This directory contains design decisions, implementation details, and porting records for Hypura.

---

## ファイル一覧 / Files

### 設計ドキュメント

| ファイル | 内容 |
|---|---|
| [implementation-log.md](./implementation-log.md) | 全実装の時系列ログ（スキャフォールドから Windows ポートまで）|
| [windows-wsl2-port.md](./windows-wsl2-port.md) | Windows/WSL2 + CUDA ポートの詳細設計ドキュメント |

### 実装ログ (`yyyy-mm-dd_{内容}_{実装AI}.md`)

| ファイル | 内容 | 実装AI |
|---|---|---|
| [2026-03-23_windows-wsl2-cuda-port_claude-sonnet-4-6.md](./2026-03-23_windows-wsl2-cuda-port_claude-sonnet-4-6.md) | Windows/WSL2/CUDA クロスプラットフォーム移植 | Claude Sonnet 4.6 |
| [2026-03-23_initial-commit-readme-bilingual_claude-sonnet-4-6.md](./2026-03-23_initial-commit-readme-bilingual_claude-sonnet-4-6.md) | 初期コミット・README 日英併記リライト・型警告 0 | Claude Sonnet 4.6 |
| [2026-03-23_dunce-unc-path-fix_claude-sonnet-4-6.md](./2026-03-23_dunce-unc-path-fix_claude-sonnet-4-6.md) | Windows UNC パス問題 (dunce) + Avast ブロック対処 | Claude Sonnet 4.6 |

---

## 目的 / Purpose

- 将来の開発者（および LLM）が設計決定の背景を理解できるようにする
- プラットフォーム固有の実装詳細を一箇所に集約する
- ポート作業のアーキテクチャ上の理由を記録する

These docs exist so future developers (and LLMs) can understand the *why* behind design decisions, have a single place for platform-specific implementation details, and trace the architectural rationale of porting work.
