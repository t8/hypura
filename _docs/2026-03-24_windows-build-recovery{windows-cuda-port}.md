# 2026-03-24 Windows Build Recovery Log

## Context
- Repository: `C:\Users\downl\Desktop\hypura-main\hypura-main`
- Branch: `windows-cuda-port`
- Plan: Hypura Windows 復旧・検証計画

## Phase 1: Single-lane build environment
- Stopped/verified absence of build-related processes (`cargo`, `rustc`, `cmake`, `msbuild`, `sccache`, `hypura`).
- Switched to isolated run IDs and target directories:
  - `run-20260324-0124` -> `target-codex-run-20260324-0124`
  - `run-20260324-0215` -> `target-codex-run-20260324-0215`
  - `run-20260324-0245` -> `target-codex-run-20260324-0245`

## Phase 2: Repro build and debug evidence
- Reproduced prior failure once:
  - `rustversion v1.0.22` build script execute failure
  - `os error 5` (Access denied)
- Confirmed executable existence and direct execution capability:
  - `target-codex-run-20260324-0215\release\build\rustversion-... \build-script-build.exe`
  - Direct run executes and panics with `OUT_DIR not set` (expected when not run under Cargo), proving file is executable.
- Captured `build.rs` debug evidence in `debug-4ee339.log` for run IDs:
  - `run-20260324-0215`
  - `run-20260324-0245`
- Verified key runtime-related fields in debug log:
  - `PROFILE=release` for release runs
  - `CMAKE_BUILD_TYPE=Release`
  - `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`
  - `LLAMA_BUILD_TOOLS=OFF`

## Phase 3: PR delta classification
- Shareable code deltas (candidate for PR):
  - `hypura-sys/build.rs`
  - `src/io/compat.rs`
  - `src/profiler/cpu.rs`
  - `hypura-sys/Cargo.toml`, `Cargo.lock` (if required by dependency/build changes)
- Local-only / non-PR artifacts:
  - `_docs/*` handoff/restart notes
  - `debug-4ee339.log`
  - `target-codex-*` directories
  - `.specstory/`
- Branch/remotes confirmed:
  - branch: `windows-cuda-port`
  - remotes: `origin`, `upstream`

## Phase 4: Smoke test (`/`, `/api/tags`, `/api/generate`)
- Started `serve` using existing binary:
  - `target-codex\release\hypura.exe serve <MODEL> --port 8080`
- Observed hardware profiling and model loading logs, including CUDA detection and tensor loading.
- Also tried reduced context:
  - `--context 1024`
- Result:
  - API bind confirmation (`Hypura serving ... Endpoint: http://127.0.0.1:8080`) was not reached.
  - `Invoke-WebRequest http://127.0.0.1:8080/` returned connection failure during observed runs.
  - Process terminated during/after model load in both attempts.

## Current blockers
- Intermittent artifact lock behavior still occurs on reused target dirs (`Blocking waiting for file lock on artifact directory`).
- `serve` process exits before API endpoint becomes reachable (requires next-step root-cause capture around post-load stage).

## Suggested next actions
1. Force strictly fresh target dir per attempt and avoid reusing a locked `target-codex-run-*`.
2. Capture final crash reason for `serve` right after tensor-load completion (stdout/stderr redirection to dedicated file).
3. Once API bind line appears, run smoke in order:
   - `/`
   - `/api/tags`
   - `/api/generate`

## Follow-up execution (same day)
- Fresh target build attempt executed with:
  - `run-20260324-022336`
  - `target-codex-run-20260324-022336`
  - build logs: `_docs/logs/build-run-20260324-022336.log` and `_docs/logs/build-run-20260324-022336.err.log`
- Serve was re-run with stdout/stderr redirection to dedicated files:
  - `_docs/logs/serve-run-serve-024443.out.log`
  - `_docs/logs/serve-run-serve-024443.err.log`
- Crash hypothesis update:
  - No process crash observed on redirected run.
  - `Hypura serving ... Endpoint: http://127.0.0.1:8080` was confirmed in output.
  - Earlier "terminated during model load" behavior was not reproduced in this redirected run.

## Final smoke result (`/` -> `/api/tags` -> `/api/generate`)
- Smoke run ID: `run-smoke-024730`
- Logs:
  - `_docs/logs/serve-run-smoke-024730.out.log`
  - `_docs/logs/serve-run-smoke-024730.err.log`
- Endpoint checks:
  - `/` -> `200` with body `{"status":"ok"}`
  - `/api/tags` -> `200` and loaded model listed
  - `/api/generate` -> success response (`done: true`)
- Note:
  - Generation content quality is not part of this smoke; API transport and completion path were confirmed.
