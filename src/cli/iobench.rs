use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use hypura::io::aligned_buffer::AlignedBuffer;
use hypura::io::compat::{self, NativeFd};
use hypura::model::gguf::GgufFile;

const BLOCK_SIZE: usize = 4 * 1024 * 1024; // 4 MiB read chunks (matches typical tensor size)
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

    // Run cache-bypass variants first to avoid page cache contamination from variant A.
    let bw_b = test_nocache_sequential(path, data_start, test_bytes)?;
    let bw_c = test_nocache_advfree_cycle(path, data_start, test_bytes)?;
    let bw_d2 = test_mt_nocache(path, data_start, test_bytes, 2)?;
    let bw_d4 = test_mt_nocache(path, data_start, test_bytes, 4)?;
    let bw_e2 = test_mt_nocache_advfree(path, data_start, test_bytes, 2)?;
    let bw_e4 = test_mt_nocache_advfree(path, data_start, test_bytes, 4)?;
    let bw_f = test_scattered_reads(path, &gguf, test_bytes)?;
    // Variant A last (populates page cache)
    let bw_a = test_raw_sequential(path, data_start, test_bytes)?;

    println!("  Results:");
    println!();
    let fmt = |label: &str, bw: f64| {
        let pct = (bw / bw_a - 1.0) * 100.0;
        let sign = if pct >= 0.0 { "+" } else { "" };
        if pct.abs() < 0.5 {
            println!("  {label:<42} {:.2} GB/s", bw / 1e9);
        } else {
            println!(
                "  {label:<42} {:.2} GB/s  ({sign}{:.1}%)",
                bw / 1e9,
                pct
            );
        }
    };

    fmt("A. Raw sequential read (baseline)", bw_a);
    fmt("B. Cache-bypass sequential read", bw_b);
    println!();
    println!("  C. Cache-bypass + advise-free cycle:");
    for (i, &bw) in bw_c.iter().enumerate() {
        fmt(&format!("     Pass {} (re-read after release)", i + 1), bw);
    }
    println!();
    fmt("D. Multi-threaded cache-bypass (2 threads)", bw_d2);
    fmt("   Multi-threaded cache-bypass (4 threads)", bw_d4);
    println!();
    fmt("E. MT + advise-free (2 threads)", bw_e2);
    fmt("   MT + advise-free (4 threads)", bw_e4);
    println!();
    fmt("F. Scattered per-tensor reads", bw_f);

    // Diagnosis
    println!();
    let nocache_impact = (1.0 - bw_b / bw_a) * 100.0;
    let advfree_impact = (1.0 - bw_c[0] / bw_b) * 100.0;
    let scatter_impact = (1.0 - bw_f / bw_b) * 100.0;
    let mt_gain = (bw_d4 / bw_b - 1.0) * 100.0;

    println!("  Diagnosis:");
    if advfree_impact > 30.0 {
        println!("    >> Page-release re-fault is a major bottleneck ({advfree_impact:.0}% throughput loss)");
    }
    if nocache_impact > 20.0 {
        println!("    >> Cache-bypass is a significant bottleneck ({nocache_impact:.0}% throughput loss)");
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

// ── Low-level helpers ─────────────────────────────────────────────────────────

/// Read `size` bytes from `fd` at `file_offset` into `dst`. Handles partial reads.
fn read_full(fd: NativeFd, dst: *mut u8, size: usize, file_offset: u64) {
    let mut done = 0usize;
    while done < size {
        let n = compat::read_at_fd(fd, unsafe { dst.add(done) }, size - done, file_offset + done as u64);
        if n <= 0 { break; }
        done += n as usize;
    }
}

// ── Variant A: raw sequential read (baseline) ─────────────────────────────────

fn test_raw_sequential(path: &Path, data_start: u64, test_bytes: usize) -> anyhow::Result<f64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(data_start))?;
    let mut buf = vec![0u8; BLOCK_SIZE];

    // Warmup
    let _ = file.read(&mut buf[..BLOCK_SIZE.min(test_bytes)]);
    file.seek(SeekFrom::Start(data_start))?;

    let start = Instant::now();
    let mut total = 0usize;
    while total < test_bytes {
        let chunk = BLOCK_SIZE.min(test_bytes - total);
        let n = file.read(&mut buf[..chunk])?;
        if n == 0 { break; }
        total += n;
    }
    let elapsed = start.elapsed().as_secs_f64();
    Ok(total as f64 / elapsed)
}

// ── Variant B: cache-bypass sequential read ───────────────────────────────────

fn test_nocache_sequential(path: &Path, data_start: u64, test_bytes: usize) -> anyhow::Result<f64> {
    let fd = compat::open_direct_fd(path)?;
    let mut buf = AlignedBuffer::new(BLOCK_SIZE, PAGE_SIZE)?;

    let start = Instant::now();
    let mut total = 0usize;
    let mut off = 0usize;
    while off < test_bytes {
        let chunk = BLOCK_SIZE.min(test_bytes - off);
        read_full(fd, buf.as_mut_ptr(), chunk, data_start + off as u64);
        total += chunk;
        off += chunk;
    }
    let elapsed = start.elapsed().as_secs_f64();
    compat::close_fd(fd);
    Ok(total as f64 / elapsed)
}

// ── Variant C: cache-bypass + advise-free cycle ───────────────────────────────

fn test_nocache_advfree_cycle(
    path: &Path,
    data_start: u64,
    test_bytes: usize,
) -> anyhow::Result<Vec<f64>> {
    let fd = compat::open_direct_fd(path)?;

    // Allocate a full-size buffer (like Hypura's NVMe buffer)
    let buf_ptr = compat::alloc_pages(test_bytes);
    anyhow::ensure!(!buf_ptr.is_null(), "alloc_pages failed for {test_bytes} bytes");

    // Prime: initial read to commit pages
    let mut off = 0usize;
    while off < test_bytes {
        let chunk = BLOCK_SIZE.min(test_bytes - off);
        read_full(fd, unsafe { buf_ptr.add(off) }, chunk, data_start + off as u64);
        off += chunk;
    }

    let mut results = Vec::new();
    for _ in 0..3 {
        // Release pages back to OS (matches release_layer in nvme_backend.rs)
        compat::advise_free_pages(buf_ptr, test_bytes);

        // Re-read (timed)
        let start = Instant::now();
        let mut off2 = 0usize;
        while off2 < test_bytes {
            let chunk = BLOCK_SIZE.min(test_bytes - off2);
            read_full(fd, unsafe { buf_ptr.add(off2) }, chunk, data_start + off2 as u64);
            off2 += chunk;
        }
        let elapsed = start.elapsed().as_secs_f64();
        results.push(test_bytes as f64 / elapsed);
    }

    compat::free_pages(buf_ptr, test_bytes);
    compat::close_fd(fd);
    Ok(results)
}

// ── Variant D: multi-threaded cache-bypass ────────────────────────────────────

fn test_mt_nocache(
    path: &Path,
    data_start: u64,
    test_bytes: usize,
    num_threads: usize,
) -> anyhow::Result<f64> {
    let buf_ptr = compat::alloc_pages(test_bytes);
    anyhow::ensure!(!buf_ptr.is_null(), "alloc_pages failed");
    let buf_addr = buf_ptr as usize;
    let barrier = Arc::new(Barrier::new(num_threads + 1));
    let chunk_per_thread = (test_bytes + num_threads - 1) / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let fd = compat::open_direct_fd(path).expect("open_direct_fd failed");
            let barrier = barrier.clone();
            let start_off = i * chunk_per_thread;
            let end_off = (start_off + chunk_per_thread).min(test_bytes);
            let thread_bytes = end_off - start_off;

            std::thread::spawn(move || {
                barrier.wait();
                let my_buf = (buf_addr + start_off) as *mut u8;
                let mut off = 0usize;
                while off < thread_bytes {
                    let chunk = BLOCK_SIZE.min(thread_bytes - off);
                    read_full(fd, unsafe { my_buf.add(off) }, chunk, data_start + (start_off + off) as u64);
                    off += chunk;
                }
                compat::close_fd(fd);
            })
        })
        .collect();

    barrier.wait();
    let start = Instant::now();
    for h in handles { h.join().unwrap(); }
    let elapsed = start.elapsed().as_secs_f64();

    compat::free_pages(buf_ptr, test_bytes);
    Ok(test_bytes as f64 / elapsed)
}

// ── Variant E: multi-threaded + advise-free ───────────────────────────────────

fn test_mt_nocache_advfree(
    path: &Path,
    data_start: u64,
    test_bytes: usize,
    num_threads: usize,
) -> anyhow::Result<f64> {
    let buf_ptr = compat::alloc_pages(test_bytes);
    anyhow::ensure!(!buf_ptr.is_null(), "alloc_pages failed");

    // Prime
    {
        let fd = compat::open_direct_fd(path)?;
        let mut off = 0usize;
        while off < test_bytes {
            let chunk = BLOCK_SIZE.min(test_bytes - off);
            read_full(fd, unsafe { buf_ptr.add(off) }, chunk, data_start + off as u64);
            off += chunk;
        }
        compat::close_fd(fd);
    }

    // Release
    compat::advise_free_pages(buf_ptr, test_bytes);

    // Multi-threaded re-read
    let buf_addr = buf_ptr as usize;
    let barrier = Arc::new(Barrier::new(num_threads + 1));
    let chunk_per_thread = (test_bytes + num_threads - 1) / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let fd = compat::open_direct_fd(path).expect("open_direct_fd failed");
            let barrier = barrier.clone();
            let start_off = i * chunk_per_thread;
            let end_off = (start_off + chunk_per_thread).min(test_bytes);
            let thread_bytes = end_off - start_off;

            std::thread::spawn(move || {
                barrier.wait();
                let my_buf = (buf_addr + start_off) as *mut u8;
                let mut off = 0usize;
                while off < thread_bytes {
                    let chunk = BLOCK_SIZE.min(thread_bytes - off);
                    read_full(fd, unsafe { my_buf.add(off) }, chunk, data_start + (start_off + off) as u64);
                    off += chunk;
                }
                compat::close_fd(fd);
            })
        })
        .collect();

    barrier.wait();
    let start = Instant::now();
    for h in handles { h.join().unwrap(); }
    let elapsed = start.elapsed().as_secs_f64();

    compat::free_pages(buf_ptr, test_bytes);
    Ok(test_bytes as f64 / elapsed)
}

// ── Variant F: scattered per-tensor reads ────────────────────────────────────

fn test_scattered_reads(
    path: &Path,
    gguf: &GgufFile,
    max_bytes: usize,
) -> anyhow::Result<f64> {
    let fd = compat::open_direct_fd(path)?;
    let mut buf = AlignedBuffer::new(BLOCK_SIZE, PAGE_SIZE)?;

    let mut regions: Vec<(u64, usize)> = gguf
        .tensors
        .iter()
        .map(|t| (gguf.data_offset + t.offset, t.size_bytes as usize))
        .collect();
    regions.sort_by_key(|r| r.0);

    let start = Instant::now();
    let mut total = 0usize;

    for &(file_off, size) in &regions {
        if total + size > max_bytes { break; }
        let read_size = size.min(BLOCK_SIZE);
        read_full(fd, buf.as_mut_ptr(), read_size, file_off);
        total += read_size;
    }
    let elapsed = start.elapsed().as_secs_f64();

    compat::close_fd(fd);
    Ok(total as f64 / elapsed)
}
