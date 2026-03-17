use std::ffi::c_void;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use hypura::model::gguf::GgufFile;

const BLOCK_SIZE: usize = 4 * 1024 * 1024; // 4 MiB pread chunks (matches typical tensor size)
const PAGE_SIZE: usize = 4096;

pub fn run(model_path: &str, read_gb: f64) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    let gguf = GgufFile::open(path)?;
    let file_size = std::fs::metadata(path)?.len();
    let data_start = gguf.data_offset & !(PAGE_SIZE as u64 - 1); // page-align down
    let available = file_size.saturating_sub(data_start) as usize;
    let test_bytes = ((read_gb * (1u64 << 30) as f64) as usize).min(available);

    let model_name = path.file_name().unwrap_or_default().to_string_lossy();

    println!("Hypura I/O Microbenchmark: {model_name}");
    println!("────────────────────────────────────────────────");
    println!(
        "  File: {:.1} GB tensor data starting at offset {:#X}",
        available as f64 / (1u64 << 30) as f64,
        data_start
    );
    println!(
        "  Test region: {:.1} GB, {BLOCK_SIZE} byte blocks",
        test_bytes as f64 / (1u64 << 30) as f64
    );
    println!();

    // Run F_NOCACHE variants first to avoid page cache contamination from variant A.
    let bw_b = test_nocache_sequential(path, data_start, test_bytes)?;
    let bw_c = test_nocache_madvfree_cycle(path, data_start, test_bytes)?;
    let bw_d2 = test_mt_nocache(path, data_start, test_bytes, 2)?;
    let bw_d4 = test_mt_nocache(path, data_start, test_bytes, 4)?;
    let bw_e2 = test_mt_nocache_madvfree(path, data_start, test_bytes, 2)?;
    let bw_e4 = test_mt_nocache_madvfree(path, data_start, test_bytes, 4)?;
    let bw_f = test_scattered_reads(path, &gguf, test_bytes)?;
    // Variant A last (populates page cache)
    let bw_a = test_raw_sequential(path, data_start, test_bytes)?;

    println!("  Results:");
    println!();
    let fmt = |label: &str, bw: f64| {
        let pct = (bw / bw_a - 1.0) * 100.0;
        let sign = if pct >= 0.0 { "+" } else { "" };
        if (pct.abs()) < 0.5 {
            println!("  {label:<42} {:.2} GB/s", bw / 1e9);
        } else {
            println!(
                "  {label:<42} {:.2} GB/s  ({sign}{:.1}%)",
                bw / 1e9,
                pct
            );
        }
    };

    fmt("A. Raw sequential pread (baseline)", bw_a);
    fmt("B. pread + F_NOCACHE", bw_b);
    println!();
    println!("  C. F_NOCACHE + MADV_FREE cycle:");
    for (i, &bw) in bw_c.iter().enumerate() {
        fmt(&format!("     Pass {} (re-read after release)", i + 1), bw);
    }
    println!();
    fmt("D. Multi-threaded F_NOCACHE (2 threads)", bw_d2);
    fmt("   Multi-threaded F_NOCACHE (4 threads)", bw_d4);
    println!();
    fmt("E. MT + MADV_FREE (2 threads)", bw_e2);
    fmt("   MT + MADV_FREE (4 threads)", bw_e4);
    println!();
    fmt("F. Scattered per-tensor reads", bw_f);

    // Diagnosis
    println!();
    let nocache_impact = (1.0 - bw_b / bw_a) * 100.0;
    let madvfree_impact = (1.0 - bw_c[0] / bw_b) * 100.0;
    let scatter_impact = (1.0 - bw_f / bw_b) * 100.0;
    let mt_gain = (bw_d4 / bw_b - 1.0) * 100.0;

    println!("  Diagnosis:");
    if madvfree_impact > 30.0 {
        println!("    >> MADV_FREE re-fault is a major bottleneck ({madvfree_impact:.0}% throughput loss)");
    }
    if nocache_impact > 20.0 {
        println!("    >> F_NOCACHE is a significant bottleneck ({nocache_impact:.0}% throughput loss)");
    }
    if scatter_impact > 30.0 {
        println!("    >> Per-tensor scattered reads cost {scatter_impact:.0}% throughput vs sequential");
    }
    if mt_gain > 10.0 {
        println!("    >> Multi-threading helps: +{mt_gain:.0}% with 4 threads");
    } else {
        println!("    >> Multi-threading provides minimal benefit ({mt_gain:+.0}%)");
    }
    println!();

    Ok(())
}

/// Allocate a page-aligned buffer via posix_memalign.
fn alloc_aligned(size: usize) -> *mut u8 {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        libc::posix_memalign(&mut ptr, PAGE_SIZE, size);
    }
    ptr as *mut u8
}

/// Read `size` bytes from `fd` at `file_offset` into `dst`. Handles partial reads.
fn pread_full(fd: i32, dst: *mut u8, size: usize, file_offset: u64) {
    let mut read = 0usize;
    while read < size {
        let n = unsafe {
            libc::pread(
                fd,
                dst.add(read) as *mut c_void,
                size - read,
                (file_offset + read as u64) as libc::off_t,
            )
        };
        if n <= 0 {
            break;
        }
        read += n as usize;
    }
}

/// Sequential pread through a region in BLOCK_SIZE chunks. Returns bytes read.
fn pread_sequential(fd: i32, buf: *mut u8, file_start: u64, total: usize) -> usize {
    let mut offset = 0;
    while offset < total {
        let chunk = BLOCK_SIZE.min(total - offset);
        pread_full(fd, unsafe { buf.add(offset) }, chunk, file_start + offset as u64);
        offset += chunk;
    }
    offset
}

/// Variant A: raw sequential pread, no F_NOCACHE, no MADV_FREE.
fn test_raw_sequential(path: &Path, data_start: u64, test_bytes: usize) -> anyhow::Result<f64> {
    let file = std::fs::File::open(path)?;
    let fd = file.as_raw_fd();
    let buf = alloc_aligned(BLOCK_SIZE);

    // Warmup
    pread_sequential(fd, buf, data_start, BLOCK_SIZE.min(test_bytes));

    let start = Instant::now();
    let mut total = 0usize;
    let mut off = 0usize;
    while off < test_bytes {
        let chunk = BLOCK_SIZE.min(test_bytes - off);
        pread_full(fd, buf, chunk, data_start + off as u64);
        total += chunk;
        off += chunk;
    }
    let elapsed = start.elapsed().as_secs_f64();

    unsafe { libc::free(buf as *mut c_void) };
    Ok(total as f64 / elapsed)
}

/// Variant B: pread + F_NOCACHE.
fn test_nocache_sequential(
    path: &Path,
    data_start: u64,
    test_bytes: usize,
) -> anyhow::Result<f64> {
    let file = std::fs::File::open(path)?;
    let fd = file.as_raw_fd();
    unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
    let buf = alloc_aligned(BLOCK_SIZE);

    let start = Instant::now();
    let mut total = 0usize;
    let mut off = 0usize;
    while off < test_bytes {
        let chunk = BLOCK_SIZE.min(test_bytes - off);
        pread_full(fd, buf, chunk, data_start + off as u64);
        total += chunk;
        off += chunk;
    }
    let elapsed = start.elapsed().as_secs_f64();

    unsafe { libc::free(buf as *mut c_void) };
    Ok(total as f64 / elapsed)
}

/// Variant C: F_NOCACHE + MADV_FREE cycle (3 passes).
/// Returns throughput for each re-read pass.
fn test_nocache_madvfree_cycle(
    path: &Path,
    data_start: u64,
    test_bytes: usize,
) -> anyhow::Result<Vec<f64>> {
    let file = std::fs::File::open(path)?;
    let fd = file.as_raw_fd();
    unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };

    // Full-size buffer (like Hypura's NVMe buffer)
    let buf = alloc_aligned(test_bytes);

    // Prime: initial read to commit pages
    pread_sequential(fd, buf, data_start, test_bytes);

    let mut results = Vec::new();

    for _ in 0..3 {
        // Release pages (matches release_layer in nvme_backend.rs)
        unsafe {
            libc::madvise(buf as *mut c_void, test_bytes, libc::MADV_FREE);
        }

        // Force page reclaim: allocate + touch a pressure buffer
        let pressure_size = 8usize << 30; // 8 GB
        let pressure = alloc_aligned(pressure_size);
        if !pressure.is_null() {
            // Touch every page to force OS to reclaim MADV_FREE pages
            for i in (0..pressure_size).step_by(PAGE_SIZE) {
                unsafe { *pressure.add(i) = 1 };
            }
            unsafe { libc::free(pressure as *mut c_void) };
        }

        // Re-read (timed)
        let start = Instant::now();
        pread_sequential(fd, buf, data_start, test_bytes);
        let elapsed = start.elapsed().as_secs_f64();

        results.push(test_bytes as f64 / elapsed);
    }

    unsafe { libc::free(buf as *mut c_void) };
    Ok(results)
}

/// Variant D: multi-threaded pread with F_NOCACHE.
fn test_mt_nocache(
    path: &Path,
    data_start: u64,
    test_bytes: usize,
    num_threads: usize,
) -> anyhow::Result<f64> {
    let buf = alloc_aligned(test_bytes);
    let buf_addr = buf as usize; // safe to send across threads
    let barrier = Arc::new(Barrier::new(num_threads + 1));

    let chunk_per_thread = (test_bytes + num_threads - 1) / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let file = std::fs::File::open(path).unwrap();
            let fd = file.as_raw_fd();
            unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
            std::mem::forget(file);

            let barrier = barrier.clone();
            let start_off = i * chunk_per_thread;
            let end_off = (start_off + chunk_per_thread).min(test_bytes);
            let thread_bytes = end_off - start_off;

            std::thread::spawn(move || {
                barrier.wait();
                let my_buf = (buf_addr + start_off) as *mut u8;
                pread_sequential(fd, my_buf, data_start + start_off as u64, thread_bytes);
                unsafe { libc::close(fd) };
            })
        })
        .collect();

    barrier.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();

    unsafe { libc::free(buf as *mut c_void) };
    Ok(test_bytes as f64 / elapsed)
}

/// Variant E: multi-threaded + MADV_FREE cycle.
fn test_mt_nocache_madvfree(
    path: &Path,
    data_start: u64,
    test_bytes: usize,
    num_threads: usize,
) -> anyhow::Result<f64> {
    let buf = alloc_aligned(test_bytes);

    // Prime
    {
        let file = std::fs::File::open(path)?;
        let fd = file.as_raw_fd();
        unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
        pread_sequential(fd, buf, data_start, test_bytes);
    }

    // Release
    unsafe {
        libc::madvise(buf as *mut c_void, test_bytes, libc::MADV_FREE);
    }

    // Pressure to force reclaim
    let pressure_size = 8usize << 30;
    let pressure = alloc_aligned(pressure_size);
    if !pressure.is_null() {
        for i in (0..pressure_size).step_by(PAGE_SIZE) {
            unsafe { *pressure.add(i) = 1 };
        }
        unsafe { libc::free(pressure as *mut c_void) };
    }

    // Multi-threaded re-read
    let buf_addr = buf as usize;
    let barrier = Arc::new(Barrier::new(num_threads + 1));
    let chunk_per_thread = (test_bytes + num_threads - 1) / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let file = std::fs::File::open(path).unwrap();
            let fd = file.as_raw_fd();
            unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
            std::mem::forget(file);

            let barrier = barrier.clone();
            let start_off = i * chunk_per_thread;
            let end_off = (start_off + chunk_per_thread).min(test_bytes);
            let thread_bytes = end_off - start_off;

            std::thread::spawn(move || {
                barrier.wait();
                let my_buf = (buf_addr + start_off) as *mut u8;
                pread_sequential(fd, my_buf, data_start + start_off as u64, thread_bytes);
                unsafe { libc::close(fd) };
            })
        })
        .collect();

    barrier.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();

    unsafe { libc::free(buf as *mut c_void) };
    Ok(test_bytes as f64 / elapsed)
}

/// Variant F: scattered per-tensor reads (simulates Hypura's actual pattern).
fn test_scattered_reads(
    path: &Path,
    gguf: &GgufFile,
    max_bytes: usize,
) -> anyhow::Result<f64> {
    let file = std::fs::File::open(path)?;
    let fd = file.as_raw_fd();
    unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
    let buf = alloc_aligned(BLOCK_SIZE);

    // Build list of (file_offset, size) from GGUF tensors
    let mut regions: Vec<(u64, usize)> = gguf
        .tensors
        .iter()
        .map(|t| (gguf.data_offset + t.offset, t.size_bytes as usize))
        .collect();
    regions.sort_by_key(|r| r.0);

    let start = Instant::now();
    let mut total = 0usize;

    for &(file_off, size) in &regions {
        if total + size > max_bytes {
            break;
        }
        // Read into the same buffer (we don't care about the data, just I/O throughput)
        let read_size = size.min(BLOCK_SIZE);
        pread_full(fd, buf, read_size, file_off);
        total += read_size;
    }
    let elapsed = start.elapsed().as_secs_f64();

    unsafe { libc::free(buf as *mut c_void) };
    Ok(total as f64 / elapsed)
}
