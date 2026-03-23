# RTX30 Release Publish Log

## Scope
- Branch: `release/rtx30-stable-2026-03-24`
- Goal: create stable RTX30 release branch/tag, publish `tar.gz` artifact, and provide bilingual usage docs.

## Implemented
1. Updated `.gitignore` with generated/large/local artifacts:
   - `target-codex*/`, `_docs/logs/`, `debug-*.log`, `.specstory/`, `.cargo/`
2. Rewrote top section of `README.md` for RTX30 stable flow (JA/EN).
3. Added bilingual release usage guide:
   - `_docs/2026-03-24_rtx30-stable-release-guide.md`
4. Packaged artifact:
   - `dist/hypura-rtx30-windows-stable-2026-03-24.tar.gz`
5. Created and pushed tag:
   - `v0.1.0-rtx30-stable-20260324`
6. Published GitHub release with uploaded artifact and JA/EN notes:
   - `https://github.com/zapabob/hypura/releases/tag/v0.1.0-rtx30-stable-20260324`

## Notes
- Fresh full rebuild with isolated target failed once due disk space (`No space left on device`) in `highs-sys` cmake stage.
- Release artifact was produced from validated existing `target-codex/release/hypura.exe` after cleanup.
