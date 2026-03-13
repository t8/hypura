use std::time::{Duration, Instant};

use crate::profiler::types::MemoryProfile;

const BUFFER_SIZE: usize = 256 * 1024 * 1024; // 256 MiB
const BENCHMARK_DURATION: Duration = Duration::from_secs(2);
const NUM_TRIALS: usize = 3;

pub fn profile_memory() -> anyhow::Result<MemoryProfile> {
    let mut sys = sysinfo::System::new_all();
    sys.refresh_memory();

    let total_bytes = sys.total_memory();
    // available_memory can return 0 on some macOS versions; fall back to free_memory
    let available_bytes = {
        let avail = sys.available_memory();
        if avail > 0 { avail } else { sys.free_memory() }
    };
    let bandwidth_bytes_per_sec = measure_memory_bandwidth();
    let is_unified = cfg!(all(target_os = "macos", target_arch = "aarch64"));

    Ok(MemoryProfile {
        total_bytes,
        available_bytes,
        bandwidth_bytes_per_sec,
        is_unified,
    })
}

fn measure_memory_bandwidth() -> u64 {
    let mut trials = Vec::with_capacity(NUM_TRIALS);

    // Allocate and initialize buffers to fault all pages
    let mut src = vec![0xAAu8; BUFFER_SIZE];
    let mut dst = vec![0x55u8; BUFFER_SIZE];

    // Warm up: one pass
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), BUFFER_SIZE);
    }

    for _ in 0..NUM_TRIALS {
        let mut iterations: u64 = 0;
        let start = Instant::now();

        while start.elapsed() < BENCHMARK_DURATION {
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), BUFFER_SIZE);
            }
            iterations += 1;

            // Prevent the compiler from optimizing away the copy
            std::hint::black_box(&dst);
        }

        let elapsed = start.elapsed().as_secs_f64();
        // Each iteration copies BUFFER_SIZE bytes (read + write = 2x bandwidth)
        let bandwidth = (iterations as f64 * BUFFER_SIZE as f64 * 2.0 / elapsed) as u64;
        trials.push(bandwidth);

        // Swap src pattern to prevent any caching tricks
        src.fill(iterations as u8);
    }

    trials.sort();
    trials[NUM_TRIALS / 2] // median
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_memory() {
        let mem = profile_memory().unwrap();
        assert!(mem.total_bytes > 0);
        assert!(mem.available_bytes <= mem.total_bytes);
        // Bandwidth should be between 10 GB/s and 1 TB/s
        assert!(mem.bandwidth_bytes_per_sec > 10_000_000_000);
        assert!(mem.bandwidth_bytes_per_sec < 1_000_000_000_000);
    }
}
