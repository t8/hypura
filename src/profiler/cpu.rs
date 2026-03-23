use crate::profiler::types::CpuProfile;

pub fn profile_cpu() -> anyhow::Result<CpuProfile> {
    let model_name = get_cpu_model_name();
    let (cores_performance, cores_efficiency) = get_core_counts();

    let is_apple_silicon = cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && model_name.contains("Apple");

    let int8_gflops = estimate_int8_gflops(&model_name, cores_performance);

    Ok(CpuProfile {
        model_name,
        cores_performance,
        cores_efficiency,
        has_amx: is_apple_silicon,
        has_neon: cfg!(target_arch = "aarch64"),
        has_avx512: detect_avx512(),
        has_avx2: detect_avx2(),
        int8_gflops,
    })
}

// ── CPU model name ────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn get_cpu_model_name() -> String {
    sysctl_string("machdep.cpu.brand_string").unwrap_or_else(|_| "Unknown".to_string())
}

#[cfg(target_os = "linux")]
fn get_cpu_model_name() -> String {
    // /proc/cpuinfo is authoritative on Linux / WSL2
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in info.lines() {
            if line.starts_with("model name") {
                if let Some(name) = line.splitn(2, ':').nth(1) {
                    return name.trim().to_string();
                }
            }
        }
    }
    // Fallback: sysinfo brand string
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_all();
    sys.cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

#[cfg(target_os = "windows")]
fn get_cpu_model_name() -> String {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_all();
    sys.cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

// ── Core counts ───────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn get_core_counts() -> (u32, u32) {
    let total = sysctl_u32("hw.ncpu").unwrap_or(1);
    let perf = sysctl_u32("hw.perflevel0.physicalcpu").unwrap_or(total);
    let eff = sysctl_u32("hw.perflevel1.physicalcpu").unwrap_or(0);
    (perf, eff)
}

#[cfg(not(target_os = "macos"))]
fn get_core_counts() -> (u32, u32) {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_all();
    let physical = sys.physical_core_count().unwrap_or(
        std::thread::available_parallelism()
            .map(|n| n.get() / 2)
            .unwrap_or(2),
    ) as u32;
    // Windows/Linux don't expose P/E core split in a portable way
    (physical, 0)
}

// ── ISA extension detection ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn detect_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}
#[cfg(not(target_arch = "x86_64"))]
fn detect_avx2() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn detect_avx512() -> bool {
    std::is_x86_feature_detected!("avx512f")
}
#[cfg(not(target_arch = "x86_64"))]
fn detect_avx512() -> bool {
    false
}

// ── INT8 GFLOPS estimate ──────────────────────────────────────────────────────

fn estimate_int8_gflops(model_name: &str, physical_cores: u32) -> f64 {
    // Apple Silicon lookup (ordered specific-first so "M2 Max" > "M2")
    let apple_specs: &[(&str, f64)] = &[
        ("M5 Ultra", 44.0),
        ("M5 Max", 22.0),
        ("M5 Pro", 11.0),
        ("M5", 5.5),
        ("M4 Ultra", 40.0),
        ("M4 Max", 20.0),
        ("M4 Pro", 10.0),
        ("M4", 5.0),
        ("M3 Ultra", 32.0),
        ("M3 Max", 16.0),
        ("M3 Pro", 8.0),
        ("M3", 4.0),
        ("M2 Ultra", 24.0),
        ("M2 Max", 12.0),
        ("M2 Pro", 6.0),
        ("M2", 3.0),
        ("M1 Ultra", 20.0),
        ("M1 Max", 10.0),
        ("M1 Pro", 4.0),
        ("M1", 2.0),
    ];
    for (pattern, gflops) in apple_specs {
        if model_name.contains(pattern) {
            return *gflops;
        }
    }

    // x86: estimate from ISA width and core count
    // AVX-512 VNNI: ~16 INT8 ops/cycle/core; AVX2 VPDPBUSD: ~8; baseline: ~4
    let cores = physical_cores.max(1) as f64;
    if detect_avx512() {
        // e.g., Intel Cascade Lake / Ice Lake: ~16 GFLOPS INT8/core @ 4 GHz
        cores * 16.0
    } else if detect_avx2() {
        // Ryzen 5000, Intel 10th gen+: ~8 GFLOPS INT8/core @ 4 GHz
        cores * 8.0
    } else {
        // Older or unknown CPU
        cores * 4.0
    }
}

// ── macOS sysctl helpers (compiled only on macOS) ─────────────────────────────

#[cfg(target_os = "macos")]
pub(crate) fn sysctl_string(name: &str) -> anyhow::Result<String> {
    use std::ffi::CStr;
    let c_name = std::ffi::CString::new(name)?;
    let mut size: libc::size_t = 0;

    let ret = unsafe {
        libc::sysctlbyname(c_name.as_ptr(), std::ptr::null_mut(), &mut size, std::ptr::null_mut(), 0)
    };
    anyhow::ensure!(ret == 0, "sysctlbyname({name}) failed: {}", std::io::Error::last_os_error());

    let mut buf = vec![0u8; size];
    let ret = unsafe {
        libc::sysctlbyname(
            c_name.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    anyhow::ensure!(ret == 0, "sysctlbyname({name}) read failed: {}", std::io::Error::last_os_error());

    let cstr = CStr::from_bytes_until_nul(&buf)
        .unwrap_or_else(|_| unsafe { CStr::from_ptr(buf.as_ptr() as *const i8) });
    Ok(cstr.to_string_lossy().to_string())
}

#[cfg(target_os = "macos")]
pub(crate) fn sysctl_u32(name: &str) -> anyhow::Result<u32> {
    let c_name = std::ffi::CString::new(name)?;
    let mut value: u32 = 0;
    let mut size = std::mem::size_of::<u32>() as libc::size_t;

    let ret = unsafe {
        libc::sysctlbyname(
            c_name.as_ptr(),
            &mut value as *mut u32 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    anyhow::ensure!(ret == 0, "sysctlbyname({name}) failed: {}", std::io::Error::last_os_error());
    Ok(value)
}

/// Stub sysctl functions for non-macOS — callers already have `unwrap_or` fallbacks.
#[cfg(not(target_os = "macos"))]
pub(crate) fn sysctl_string(_name: &str) -> anyhow::Result<String> {
    anyhow::bail!("sysctl is not available on this platform")
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn sysctl_u32(_name: &str) -> anyhow::Result<u32> {
    anyhow::bail!("sysctl is not available on this platform")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_cpu() {
        let cpu = profile_cpu().unwrap();
        assert!(!cpu.model_name.is_empty());
        assert!(cpu.cores_performance > 0);
        assert!(cpu.int8_gflops > 0.0);
        #[cfg(target_arch = "aarch64")]
        assert!(cpu.has_neon);
    }

    #[test]
    fn test_apple_silicon_lookup() {
        // Only meaningful on macOS; on other platforms model_name won't match
        assert_eq!(estimate_int8_gflops("Apple M2 Max", 12), 12.0);
        assert_eq!(estimate_int8_gflops("Apple M1", 8), 2.0);
    }

    #[test]
    fn test_avx_detection_x86() {
        #[cfg(target_arch = "x86_64")]
        {
            // AVX2 is present on virtually all CPUs from 2013+, but we can't
            // assert true here — just make sure it doesn't panic.
            let _ = detect_avx2();
            let _ = detect_avx512();
        }
    }
}
