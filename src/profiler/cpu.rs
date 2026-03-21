use std::ffi::CStr;

use crate::profiler::types::CpuProfile;

pub fn profile_cpu() -> anyhow::Result<CpuProfile> {
    let model_name = sysctl_string("machdep.cpu.brand_string")
        .unwrap_or_else(|_| "Unknown".to_string());

    let total_cores = sysctl_u32("hw.ncpu").unwrap_or(1);
    let cores_performance = sysctl_u32("hw.perflevel0.physicalcpu").unwrap_or(total_cores);
    let cores_efficiency = sysctl_u32("hw.perflevel1.physicalcpu").unwrap_or(0);

    let is_apple_silicon = cfg!(target_arch = "aarch64") && model_name.contains("Apple");

    let int8_gflops = estimate_int8_gflops(&model_name);

    Ok(CpuProfile {
        model_name,
        cores_performance,
        cores_efficiency,
        has_amx: is_apple_silicon,
        has_neon: cfg!(target_arch = "aarch64"),
        has_avx512: false, // Not on Apple Silicon
        has_avx2: false,   // Not on Apple Silicon
        int8_gflops,
    })
}

fn estimate_int8_gflops(model_name: &str) -> f64 {
    // Ordered specific-first so "M4 Max" matches before "M4"
    let specs: &[(&str, f64)] = &[
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

    for (pattern, gflops) in specs {
        if model_name.contains(pattern) {
            return *gflops;
        }
    }

    tracing::warn!("Unknown CPU model '{model_name}', using conservative INT8 GFLOPS estimate");
    2.0
}

pub(crate) fn sysctl_string(name: &str) -> anyhow::Result<String> {
    let c_name = std::ffi::CString::new(name)?;
    let mut size: libc::size_t = 0;

    // First call to get size
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
    fn test_lookup_ordering() {
        assert_eq!(estimate_int8_gflops("Apple M2 Max"), 12.0);
        assert_eq!(estimate_int8_gflops("Apple M2 Pro"), 6.0);
        assert_eq!(estimate_int8_gflops("Apple M2"), 3.0);
        assert_eq!(estimate_int8_gflops("Apple M1 Max"), 10.0);
    }
}
