/// Cross-platform I/O primitives for the NVMe streaming backend.
///
/// Provides:
/// - `NativeFd` — file descriptor (Unix) or HANDLE (Windows)
/// - `open_direct_fd` — open with OS-cache bypass
/// - `close_fd` — close native handle
/// - `read_at_fd` — positional read (pread on Unix, ReadFile on Windows)
/// - `alloc_pages` / `free_pages` — anonymous page-backed memory

// ─────────────────────────────────────────────────────────────
// Type alias
// ─────────────────────────────────────────────────────────────

/// Unix: raw file descriptor (i32).
/// Windows: Win32 HANDLE stored as isize.
#[cfg(unix)]
pub type NativeFd = i32;

#[cfg(windows)]
pub type NativeFd = isize;

// ─────────────────────────────────────────────────────────────
// Unix implementation
// ─────────────────────────────────────────────────────────────

#[cfg(unix)]
mod imp {
    use super::NativeFd;
    use std::os::unix::io::IntoRawFd;

    /// Open `path` for direct (cache-bypass) sequential reads.
    ///
    /// - macOS: `F_NOCACHE` via `fcntl`
    /// - Linux: `POSIX_FADV_DONTNEED` advisory + `POSIX_FADV_SEQUENTIAL`
    pub fn open_direct_fd(path: &std::path::Path) -> std::io::Result<NativeFd> {
        let file = std::fs::File::open(path)?;
        let fd = file.into_raw_fd();

        #[cfg(target_os = "macos")]
        unsafe {
            // Disable the unified buffer cache for this fd.
            libc::fcntl(fd, libc::F_NOCACHE, 1);
        }

        #[cfg(target_os = "linux")]
        unsafe {
            // Ask the kernel to drop cached pages after reads (best-effort).
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL);
        }

        Ok(fd)
    }

    pub fn close_fd(fd: NativeFd) {
        unsafe {
            libc::close(fd);
        }
    }

    /// Positional read — equivalent to `pread(2)`.
    /// Returns bytes read (≥0) or -1 on error.
    pub fn read_at_fd(fd: NativeFd, dst: *mut u8, size: usize, file_offset: u64) -> isize {
        unsafe {
            libc::pread(
                fd,
                dst as *mut libc::c_void,
                size,
                file_offset as libc::off_t,
            )
        }
    }

    /// Allocate `size` bytes of anonymous, lazily-committed memory.
    /// Returns null on failure.
    pub fn alloc_pages(size: usize) -> *mut u8 {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            std::ptr::null_mut()
        } else {
            p as *mut u8
        }
    }

    pub fn free_pages(ptr: *mut u8, size: usize) {
        if !ptr.is_null() {
            unsafe {
                libc::munmap(ptr as *mut libc::c_void, size);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Windows implementation
// ─────────────────────────────────────────────────────────────

#[cfg(windows)]
mod imp {
    use super::NativeFd;
    use std::os::windows::io::IntoRawHandle;

    pub fn open_direct_fd(path: &std::path::Path) -> std::io::Result<NativeFd> {
        // Keep Windows path simple and robust across windows-sys versions.
        // We intentionally avoid CreateFileW flags here to prevent API
        // signature/feature drift from breaking compilation.
        let file = std::fs::File::open(path)?;
        Ok(file.into_raw_handle() as NativeFd)
    }

    pub fn close_fd(fd: NativeFd) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(fd as _);
        }
    }

    pub fn read_at_fd(fd: NativeFd, dst: *mut u8, size: usize, file_offset: u64) -> isize {
        use windows_sys::Win32::Storage::FileSystem::ReadFile;
        use windows_sys::Win32::System::IO::OVERLAPPED;

        let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
        // OVERLAPPED.Anonymous.Anonymous.{Offset, OffsetHigh}
        overlapped.Anonymous.Anonymous.Offset = file_offset as u32;
        overlapped.Anonymous.Anonymous.OffsetHigh = (file_offset >> 32) as u32;

        let read_size = size.min(u32::MAX as usize) as u32;
        let mut bytes_read: u32 = 0;

        let ok = unsafe {
            ReadFile(
                fd as _,
                dst as *mut _,
                read_size,
                &mut bytes_read,
                &mut overlapped as *mut _,
            )
        };

        if ok == 0 {
            let err = unsafe { windows_sys::Win32::Foundation::GetLastError() };
            const ERROR_HANDLE_EOF: u32 = 38;
            if err == ERROR_HANDLE_EOF {
                return 0;
            }
            return -1;
        }
        bytes_read as isize
    }

    pub fn alloc_pages(size: usize) -> *mut u8 {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        unsafe {
            VirtualAlloc(
                std::ptr::null(),
                size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            ) as *mut u8
        }
    }

    pub fn free_pages(ptr: *mut u8, _size: usize) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        if !ptr.is_null() {
            unsafe {
                VirtualFree(ptr as *mut _, 0, MEM_RELEASE);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Page advisory (hint to OS that pages can be recycled)
// ─────────────────────────────────────────────────────────────

/// Advise the OS that the pages at `[ptr, ptr+size)` are no longer needed
/// and can be reclaimed. The virtual address range remains valid.
///
/// - Unix: `madvise(MADV_FREE)` (macOS) / `madvise(MADV_DONTNEED)` (Linux)
/// - Windows: `VirtualFree(MEM_DECOMMIT)` — decommits physical pages but
///            keeps the virtual reservation alive.
#[cfg(unix)]
pub fn advise_free_pages(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    unsafe {
        #[cfg(target_os = "macos")]
        libc::madvise(ptr as *mut libc::c_void, size, libc::MADV_FREE);

        #[cfg(target_os = "linux")]
        libc::madvise(ptr as *mut libc::c_void, size, libc::MADV_DONTNEED);
    }
}

#[cfg(windows)]
pub fn advise_free_pages(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    unsafe {
        // Decommit physical backing pages while keeping the virtual range reserved.
        windows_sys::Win32::System::Memory::VirtualFree(
            ptr as *mut _,
            size,
            windows_sys::Win32::System::Memory::MEM_DECOMMIT,
        );
    }
}

// ─────────────────────────────────────────────────────────────
// Re-export
// ─────────────────────────────────────────────────────────────
pub use imp::{alloc_pages, close_fd, free_pages, open_direct_fd, read_at_fd};
