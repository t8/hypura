use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

use crate::profiler::types::{BandwidthCurve, StorageProfile, StorageType};

const BLOCK_SIZES: &[usize] = &[4096, 65536, 131072, 1_048_576, 4_194_304];
const SEQUENTIAL_PASSES: usize = 3;
const RANDOM_IOPS_READS: usize = 10_000;

pub fn profile_storage() -> anyhow::Result<Vec<StorageProfile>> {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut profiles = Vec::new();

    for disk in disks.list() {
        let mount = disk.mount_point().to_string_lossy().to_string();
        // Only benchmark the root volume (or Data volume on APFS)
        if mount != "/" && mount != "/System/Volumes/Data" {
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
            device_type: StorageType::NvmePcie, // All internal Apple Silicon storage is NVMe
            capacity_bytes,
            free_bytes,
            sequential_read,
            random_read_iops,
            pcie_gen: None,
            wear_level: None,
        });

        break; // Only benchmark the first root volume
    }

    anyhow::ensure!(!profiles.is_empty(), "No storage devices found to benchmark");
    Ok(profiles)
}

fn benchmark_storage(mount_point: &str, free_bytes: u64) -> anyhow::Result<(BandwidthCurve, u64)> {
    // Size temp file: 1 GiB if space allows, 256 MiB otherwise
    let file_size: usize = if free_bytes > 5 * (1 << 30) {
        1 << 30 // 1 GiB
    } else {
        256 << 20 // 256 MiB
    };

    // Create temp file with data
    let temp_dir = if mount_point == "/System/Volumes/Data" {
        std::env::temp_dir()
    } else {
        std::path::PathBuf::from(mount_point).join("tmp")
    };
    let temp_path = temp_dir.join(".hypura_bench_tmp");

    // Write test data
    {
        let mut f = std::fs::File::create(&temp_path)?;
        let pattern = vec![0xA5u8; 1 << 20]; // 1 MiB pattern
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

    // Clean up
    let _ = std::fs::remove_file(&temp_path);

    result
}

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
            let file = std::fs::File::open(path)?;
            let fd = file.as_raw_fd();

            // Bypass filesystem cache
            let ret = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
            anyhow::ensure!(ret != -1, "F_NOCACHE failed: {}", std::io::Error::last_os_error());

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
                if n <= 0 {
                    break;
                }
                total_read += n as usize;
            }
            let elapsed = start.elapsed().as_secs_f64();

            if elapsed > 0.0 {
                let bandwidth = (total_read as f64 / elapsed) as u64;
                trial_bandwidths.push(bandwidth);
            }
        }

        if !trial_bandwidths.is_empty() {
            trial_bandwidths.sort();
            let median = trial_bandwidths[trial_bandwidths.len() / 2];
            points.push((block_size as u64, median));
            peak_sequential = peak_sequential.max(median);
        }
    }

    Ok(BandwidthCurve {
        points,
        peak_sequential,
    })
}

fn benchmark_random_4k(
    path: &std::path::Path,
    file_size: usize,
) -> anyhow::Result<u64> {
    let file = std::fs::File::open(path)?;
    let fd = file.as_raw_fd();

    let ret = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
    anyhow::ensure!(ret != -1, "F_NOCACHE failed: {}", std::io::Error::last_os_error());

    let mut buf = AlignedBuffer::new(4096, 4096)?;
    let max_offset = (file_size / 4096) as u64;

    // Simple LCG for pseudo-random offsets (avoids rand crate dependency)
    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_BABEu64;

    let start = Instant::now();
    for _ in 0..RANDOM_IOPS_READS {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let block_idx = (rng_state >> 32) % max_offset;
        let offset = block_idx * 4096;

        let n = unsafe {
            libc::pread(
                fd,
                buf.as_mut_ptr() as *mut libc::c_void,
                4096,
                offset as libc::off_t,
            )
        };
        if n <= 0 {
            break;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    let iops = if elapsed > 0.0 {
        (RANDOM_IOPS_READS as f64 / elapsed) as u64
    } else {
        0
    };

    Ok(iops)
}

/// Page-aligned buffer for direct I/O.
struct AlignedBuffer {
    ptr: *mut u8,
    _len: usize,
}

impl AlignedBuffer {
    fn new(size: usize, alignment: usize) -> anyhow::Result<Self> {
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let ret = unsafe { libc::posix_memalign(&mut ptr, alignment, size) };
        anyhow::ensure!(ret == 0, "posix_memalign failed: error code {ret}");
        Ok(Self {
            ptr: ptr as *mut u8,
            _len: size,
        })
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::free(self.ptr as *mut libc::c_void);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer() {
        let mut buf = AlignedBuffer::new(4096, 4096).unwrap();
        assert!(!buf.as_mut_ptr().is_null());
        assert_eq!(buf.as_mut_ptr() as usize % 4096, 0);
    }

    #[test]
    fn test_profile_storage() {
        let profiles = profile_storage().unwrap();
        assert!(!profiles.is_empty());
        let p = &profiles[0];
        assert!(p.capacity_bytes > 0);
        assert!(p.sequential_read.peak_sequential > 100_000_000); // > 100 MB/s
        assert!(p.random_read_iops > 0);
    }
}
