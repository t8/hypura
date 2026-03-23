---
name: Windows/WSL2 + CUDA port
description: Cross-platform porting work — macOS/Metal → Windows/WSL2/CUDA
type: project
---

Completed cross-platform port of Hypura from macOS/Apple Silicon to Windows/WSL2 with CUDA (RTX 3060+) support.

**Why:** User requested Windows/WSL2 support with RTX 3060 as the base GPU target.

**How to apply:** When touching platform-specific code, follow the established patterns below.

## Architecture: `src/io/compat.rs`
New platform abstraction module providing:
- `NativeFd` type alias (i32 on Unix, isize/HANDLE on Windows)
- `open_direct_fd(path)` — cache-bypass file open
- `close_fd(fd)` — close handle
- `read_at_fd(fd, dst, size, offset)` — positional read (pread / ReadFile)
- `alloc_pages(size)` / `free_pages(ptr, size)` — anonymous memory (mmap / VirtualAlloc)
- `advise_free_pages(ptr, size)` — MADV_FREE / MEM_DECOMMIT

## Key changes made
- `hypura-sys/build.rs`: CUDA detection (CUDA_PATH, /usr/local/cuda, nvcc), sm_75;86;89;90 architectures
- `hypura-sys/src/hypura_buft.c`: `#ifdef _WIN32` VirtualAlloc/VirtualFree replacing mmap/munmap
- `src/profiler/cpu.rs`: sysctl on macOS only; sysinfo + /proc/cpuinfo on Linux; AVX2/AVX512 via is_x86_feature_detected!
- `src/profiler/gpu.rs`: Metal on macOS; CUDA backend + NVIDIA GPU spec DB (RTX 20/30/40/50, A/H series)
- `src/profiler/storage.rs`: mount point detection per-platform; F_NOCACHE on macOS, posix_fadvise on Linux, std::io on Windows
- `src/profiler/mod.rs`: APPDATA on Windows, ~/.hypura elsewhere; cross-platform os_version/machine_model
- `src/compute/inference.rs`: total_physical_memory() uses sysinfo on non-macOS
- `src/compute/nvme_backend.rs`: all libc I/O replaced with compat module
- `src/io/aligned_buffer.rs`: rewritten with std::alloc::Layout (works everywhere)
- `src/io/async_reader.rs`: rewritten with compat module
- `src/cli/iobench.rs`: rewritten with compat module
- `src/scheduler/placement.rs`: OS_OVERHEAD: macOS=2GB, Windows=4GB, Linux=1GB; GPU_RUNTIME_OVERHEAD: macOS=1GB, others=512MB
- `Cargo.toml`: windows-sys 0.59 added as conditional Windows dependency

## CUDA architectures targeted
sm_75 (RTX 20xx), sm_86 (RTX 3060 base target), sm_89 (RTX 40xx), sm_90 (H100)
Override via env: HYPURA_CUDA_ARCHITECTURES="75;86;89;90"
Disable CUDA: HYPURA_NO_CUDA=1
Force CUDA: HYPURA_CUDA=1
