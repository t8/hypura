# hypura Windows build restart log

- Date: 2026-03-23
- Timestamp (local): 2026-03-23T22:50:32+09:00
- Branch: `windows-cuda-port`
- Scope: `hypura` repo only (`clawdbot` excluded this run)

## What was verified

1. `hypura-sys/build.rs` already contains:
   - `.define("LLAMA_BUILD_TOOLS", "OFF")`
2. Git branch state:
   - `windows-cuda-port` is ahead of `upstream/main`
3. PR-relevant diffs are isolated to:
   - `hypura-sys/build.rs`
   - `.gitignore` (local artifact ignore additions)

## Build attempts and observations

Commands run:

```powershell
cargo clean -p hypura-sys; cargo build
$env:CARGO_TARGET_DIR='target-codex'; cargo clean -p hypura-sys; cargo build
$env:CARGO_TARGET_DIR='target-codex'; cargo build -p hypura-sys -q
```

Observed behavior:

- Default `target` path repeatedly reported lock waits (`artifact directory` / `package cache`).
- Isolated target directory build progressed through many crates, including:
  - `hypura-sys`
  - `highs-sys`
  - `good_lp`
- Terminal multiplexing still showed stale/running states for some background jobs, so final success footer could not be conclusively captured in this run log.

## PR #2 candidate changes

Current intended commit candidates:

```diff
M hypura-sys/build.rs   # LLAMA_BUILD_TOOLS=OFF
M .gitignore            # yes/, target-codex/
```

Out of scope / not to include in PR:

- `Cargo.lock` (unless explicitly required by maintainer policy)
- local docs and generated artifacts

## Next actions

1. Re-run build in a single clean terminal and wait for final `Finished` line.
2. Commit only PR-intended files with Conventional Commit message.
3. Push `windows-cuda-port` and update PR #2.
4. Run `hypura serve` smoke test after confirmed build.

## Disk cleanup results

- Target paths:
  - `%LOCALAPPDATA%\\Temp`
  - `npm cache` path (`npm config get cache`)
- Before cleanup:
  - Temp: `13.479 GB`
  - npm cache: `4.977 GB`
- After cleanup / re-measure:
  - Temp: `0.126 GB`
  - npm cache: `0 GB`
- Reclaimed total (approx): `18.33 GB`

## Additional debug timeline (Codex)

### Branch/PR activity
- Pushed `windows-cuda-port` with commit `1fac4fd`
- PR #2 update comment posted:
  - https://github.com/t8/hypura/pull/2#issuecomment-4111136329

### Source changes made during smoke-test debugging
- `hypura-sys/build.rs`
  - Added `bindgen` fallback tuning: `.layout_tests(false)`
  - Added temporary runtime instrumentation for build-time diagnostics
  - Added CMake profile/runtime attempts:
    - `cmake_config.profile("Release")`
    - `CMAKE_BUILD_TYPE=Release`
    - `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` (Windows path)
  - Added CMake cache extraction block for:
    - `CMAKE_BUILD_TYPE`
    - `CMAKE_CONFIGURATION_TYPES`
    - `CMAKE_MSVC_RUNTIME_LIBRARY`
- `hypura-sys/Cargo.toml`
  - Added build dependency: `serde_json = "1"` for structured debug logging
- `src/io/compat.rs`
  - Windows HANDLE path adjusted to match `windows-sys` type expectations
  - `CloseHandle` / `ReadFile` argument casts normalized
- `src/profiler/cpu.rs`
  - `physical_cpu_count()` updated to `physical_core_count()`

### Runtime findings (important)
- Frequent lock contention:
  - `Blocking waiting for file lock on artifact directory`
- This lock issue prevented consistent collection of a complete end-to-end run log.
- Existing `LNK2001` evidence (`__imp__CrtDbgReport`, `__imp__calloc_dbg`) is confirmed in historical runs,
  but newer runs were repeatedly inconclusive due to lock stalls before reliable termination output.

### Verification status by item
- `LLAMA_BUILD_TOOLS=OFF` change:
  - Implemented and pushed.
- PR minimal-diff policy (`Cargo.lock` excluded from commit):
  - Applied in commit `1fac4fd` (left local `Cargo.lock` untouched).
- Disk cleanup:
  - Completed with measured before/after values.
- `hypura serve` smoke:
  - Not fully verified yet due to build lock contention + unstable run observability.

### Known open blockers for next operator
1. Build artifact lock contention in this environment
2. Inconsistent terminal completion/exit visibility for long cargo runs
3. Need one clean, single-shell run to conclusively verify:
   - final link health on Windows
   - `/`, `/api/tags`, `/api/generate` smoke endpoints
