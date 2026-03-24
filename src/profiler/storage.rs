use std::io::Write;
use std::time::Instant;

use crate::profiler::types::{BandwidthCurve, StorageProfile, StorageType};

const BLOCK_SIZES: &[usize] = &[4096, 65536, 131_072, 1_048_576, 4_194_304];
const SEQUENTIAL_PASSES: usize = 3;
const RANDOM_IOPS_READS: usize = 10_000;

pub fn profile_storage() -> anyhow::Result<Vec<StorageProfile>> {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut profiles = Vec::new();

    for disk in disks.list() {
        let mount = disk.mount_point().to_string_lossy().to_string();

        if !is_primary_volume(&mount) {
            continue;
        }

        let device_path = disk.name().to_string_lossy().to_string();
        let capacity_bytes = disk.total_space();
        let free_bytes = disk.available_space();

        tracing::info!("Benchmarking storage at {mount}...");

        let (sequential_read, random_read_iops) = match benchmark_storage(&mount, free_bytes) {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!("Storage benchmark failed for {mount}: {e}");
                continue;
            }
        };

        profiles.push(StorageProfile {
            device_path,
            mount_point: mount,
            device_type: detect_storage_type(disk),
            capacity_bytes,
            free_bytes,
            sequential_read,
            random_read_iops,
            pcie_gen: None,
            wear_level: None,
        });

        break; // Benchmark only the first matching volume
    }

    anyhow::ensure!(!profiles.is_empty(), "No storage devices found to benchmark");
    Ok(profiles)
}

/// Returns true for the volume that should be benchmarked.
fn is_primary_volume(mount: &str) -> bool {
    #[cfg(target_os = "macos")]
    {
        mount == "/" || mount == "/System/Volumes/Data"
    }
    #[cfg(target_os = "linux")]
    {
        // On Linux / WSL2, benchmark the root filesystem
        mount == "/"
    }
    #[cfg(target_os = "windows")]
    {
        // On Windows, take the first fixed drive (usually C:\)
        mount.ends_with('\\') && mount.len() == 3
    }
}

fn detect_storage_type(disk: &sysinfo::Disk) -> StorageType {
    use sysinfo::DiskKind;
    match disk.kind() {
        DiskKind::SSD | DiskKind::Unknown(_) => StorageType::NvmePcie,
        DiskKind::HDD => StorageType::Sata,
    }
}

fn benchmark_storage(mount_point: &str, free_bytes: u64) -> anyhow::Result<(BandwidthCurve, u64)> {
    let file_size: usize = if free_bytes > 5 * (1 << 30) {
        1 << 30 // 1 GiB
    } else {
        256 << 20 // 256 MiB
    };

    let temp_dir = pick_temp_dir(mount_point);
    let temp_path = temp_dir.join(".hypura_bench_tmp");

    // Write test data
    {
        let mut f = std::fs::File::create(&temp_path)?;
        let pattern = vec![0xA5u8; 1 << 20]; // 1 MiB chunks
        let chunks = file_size / pattern.len();
        for _ in 0..chunks {
            f.write_all(&pattern)?;
        }
        f.sync_all()?;
    }

    let result = (|| -> anyhow::Result<(BandwidthCurve, u64)> {
        let sequential = benchmark_sequential(&temp_path, file_size)?;
        let iops = benchmark_random_4k(&temp_path, file_size)?;
        Ok((sequential, iops))
    })();

    let _ = std::fs::remove_file(&temp_path);
    result
}

fn pick_temp_dir(mount_point: &str) -> std::path::PathBuf {
    #[cfg(target_os = "macos")]
    if mount_point == "/System/Volumes/Data" {
        return std::env::temp_dir();
    }

    let candidate = std::path::PathBuf::from(mount_point);
    if candidate.exists() {
        candidate
    } else {
        std::env::temp_dir()
    }
}

// ── Sequential read benchmark ─────────────────────────────────────────────────

fn benchmark_sequential(
    path: &std::path::Path,
    file_size: usize,
) -> anyhow::Result<BandwidthCurve> {
    let mut points = Vec::new();
    let mut peak_sequential: u64 = 0;

    for &block_size in BLOCK_SIZES {
        if block_size > file_size {
            continue;
        }

        let mut trial_bandwidths = Vec::with_capacity(SEQUENTIAL_PASSES);

        for _ in 0..SEQUENTIAL_PASSES {
            let bw = read_sequential_pass(path, file_size, block_size)?;
            if bw > 0 {
                trial_bandwidths.push(bw);
            }
        }

        if !trial_bandwidths.is_empty() {
            trial_bandwidths.sort_unstable();
            let median = trial_bandwidths[trial_bandwidths.len() / 2];
            points.push((block_size as u64, median));
            peak_sequential = peak_sequential.max(median);
        }
    }

    Ok(BandwidthCurve { points, peak_sequential })
}

/// Platform-specific sequential read pass.
fn read_sequential_pass(
    path: &std::path::Path,
    file_size: usize,
    block_size: usize,
) -> anyhow::Result<u64> {
    #[cfg(unix)]
    {
        read_sequential_unix(path, file_size, block_size)
    }
    #[cfg(windows)]
    {
        read_sequential_windows(path, file_size, block_size)
    }
}

#[cfg(unix)]
fn read_sequential_unix(
    path: &std::path::Path,
    file_size: usize,
    block_size: usize,
) -> anyhow::Result<u64> {
    use crate::io::aligned_buffer::AlignedBuffer;
    use std::os::unix::io::AsRawFd;

    let file = std::fs::File::open(path)?;
    let fd = file.as_raw_fd();

    // macOS: disable unified buffer cache; Linux: use advisory fadvise
    #[cfg(target_os = "macos")]
    unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1); }
    #[cfg(target_os = "linux")]
    unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED); }

    let mut buf = AlignedBuffer::new(block_size, 4096)?;
    let mut total_read: usize = 0;

    let start = Instant::now();
    while total_read < file_size {
        let to_read = block_size.min(file_size - total_read);
        let n = unsafe {
            libc::pread(
                fd,
                buf.as_mut_ptr() as *mut libc::c_void,
                to_read,
                total_read as libc::off_t,
            )
        };
        if n <= 0 { break; }
        total_read += n as usize;
    }
    let elapsed = start.elapsed().as_secs_f64();

    if elapsed > 0.0 {
        Ok((total_read as f64 / elapsed) as u64)
    } else {
        Ok(0)
    }
}

#[cfg(windows)]
fn read_sequential_windows(
    path: &std::path::Path,
    file_size: usize,
    block_size: usize,
) -> anyhow::Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    use crate::io::aligned_buffer::AlignedBuffer;

    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(0))?;
    let mut buf = AlignedBuffer::new(block_size, 4096)?;
    let mut total_read: usize = 0;

    let start = Instant::now();
    while total_read < file_size {
        let to_read = block_size.min(file_size - total_read);
        let n = file.read(&mut buf[..to_read])?;
        if n == 0 { break; }
        total_read += n;
    }
    let elapsed = start.elapsed().as_secs_f64();

    if elapsed > 0.0 {
        Ok((total_read as f64 / elapsed) as u64)
    } else {
        Ok(0)
    }
}

// ── Random 4K IOPS benchmark ──────────────────────────────────────────────────

fn benchmark_random_4k(path: &std::path::Path, file_size: usize) -> anyhow::Result<u64> {
    #[cfg(unix)]
    return benchmark_random_4k_unix(path, file_size);
    #[cfg(windows)]
    return benchmark_random_4k_windows(path, file_size);
}

#[cfg(unix)]
fn benchmark_random_4k_unix(path: &std::path::Path, file_size: usize) -> anyhow::Result<u64> {
    use crate::io::aligned_buffer::AlignedBuffer;
    use std::os::unix::io::AsRawFd;

    let file = std::fs::File::open(path)?;
    let fd = file.as_raw_fd();

    #[cfg(target_os = "macos")]
    unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1); }
    #[cfg(target_os = "linux")]
    unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_RANDOM); }

    let mut buf = AlignedBuffer::new(4096, 4096)?;
    let max_offset = (file_size / 4096) as u64;

    let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
    let start = Instant::now();
    for _ in 0..RANDOM_IOPS_READS {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let block_idx = (rng >> 32) % max_offset;
        let offset = block_idx * 4096;

        let n = unsafe {
            libc::pread(fd, buf.as_mut_ptr() as *mut libc::c_void, 4096, offset as libc::off_t)
        };
        if n <= 0 { break; }
    }
    let elapsed = start.elapsed().as_secs_f64();

    Ok(if elapsed > 0.0 { (RANDOM_IOPS_READS as f64 / elapsed) as u64 } else { 0 })
}

#[cfg(windows)]
fn benchmark_random_4k_windows(path: &std::path::Path, file_size: usize) -> anyhow::Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    use crate::io::aligned_buffer::AlignedBuffer;

    let mut file = std::fs::File::open(path)?;
    let mut buf = AlignedBuffer::new(4096, 4096)?;
    let max_offset = (file_size / 4096) as u64;

    let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
    let start = Instant::now();
    for _ in 0..RANDOM_IOPS_READS {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let block_idx = (rng >> 32) % max_offset;
        let offset = block_idx * 4096;
        file.seek(SeekFrom::Start(offset))?;
        let _ = file.read(&mut buf[..4096]);
    }
    let elapsed = start.elapsed().as_secs_f64();

    Ok(if elapsed > 0.0 { (RANDOM_IOPS_READS as f64 / elapsed) as u64 } else { 0 })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_storage() {
        let profiles = profile_storage().unwrap();
        assert!(!profiles.is_empty());
        let p = &profiles[0];
        assert!(p.capacity_bytes > 0);
        assert!(p.sequential_read.peak_sequential > 50_000_000); // > 50 MB/s
        assert!(p.random_read_iops > 0);
    }
}
