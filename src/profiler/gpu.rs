use std::ffi::CStr;

use crate::profiler::types::{GpuBackend, GpuProfile};

pub fn profile_gpu() -> anyhow::Result<Option<GpuProfile>> {
    #[cfg(target_os = "macos")]
    return profile_gpu_metal();

    #[cfg(not(target_os = "macos"))]
    return profile_gpu_cuda_or_cpu();
}

// ── macOS / Metal ─────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn profile_gpu_metal() -> anyhow::Result<Option<GpuProfile>> {
    let (name, vram_bytes) = match query_metal_device() {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("No Metal GPU detected: {e}");
            return Ok(None);
        }
    };

    let (bandwidth, tflops) = match lookup_apple_silicon(&name) {
        Some(spec) => ((spec.bandwidth_gb_s * 1e9) as u64, spec.fp16_tflops),
        None => {
            tracing::warn!("Unknown Apple GPU '{name}', using conservative estimates");
            (68_250_000_000u64, 2.6)
        }
    };

    Ok(Some(GpuProfile {
        name,
        vram_bytes,
        bandwidth_bytes_per_sec: bandwidth,
        fp16_tflops: tflops,
        backend: GpuBackend::Metal,
    }))
}

#[cfg(target_os = "macos")]
fn query_metal_device() -> anyhow::Result<(String, u64)> {
    unsafe { hypura_sys::llama_backend_init() };

    let result = (|| -> anyhow::Result<(String, u64)> {
        let reg_count = unsafe { hypura_sys::ggml_backend_reg_count() };
        let mut reg = std::ptr::null_mut();

        for i in 0..reg_count {
            let r = unsafe { hypura_sys::ggml_backend_reg_get(i) };
            if r.is_null() { continue; }
            let name_ptr = unsafe { hypura_sys::ggml_backend_reg_name(r) };
            if !name_ptr.is_null() {
                let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy();
                if name.contains("MTL") || name.contains("Metal") {
                    reg = r;
                    break;
                }
            }
        }
        anyhow::ensure!(!reg.is_null(), "Metal backend not found");

        let dev_count = unsafe { hypura_sys::ggml_backend_reg_dev_count(reg) };
        anyhow::ensure!(dev_count > 0, "No Metal devices found");

        let device = unsafe { hypura_sys::ggml_backend_reg_dev_get(reg, 0) };
        anyhow::ensure!(!device.is_null(), "Metal device is null");

        let desc_ptr = unsafe { hypura_sys::ggml_backend_dev_description(device) };
        let name = if desc_ptr.is_null() {
            "Unknown Metal GPU".to_string()
        } else {
            unsafe { CStr::from_ptr(desc_ptr) }.to_string_lossy().to_string()
        };

        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe { hypura_sys::ggml_backend_dev_memory(device, &mut free, &mut total) };

        Ok((name, total as u64))
    })();

    unsafe { hypura_sys::llama_backend_free() };
    result
}

// ── Non-macOS: CUDA / CPU-only ────────────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
fn profile_gpu_cuda_or_cpu() -> anyhow::Result<Option<GpuProfile>> {
    // Query via the ggml CUDA backend if available
    let gpu = query_cuda_device();
    if let Some((name, vram_bytes)) = gpu {
        let (bandwidth, tflops) = lookup_nvidia_gpu(&name)
            .unwrap_or_else(|| estimate_nvidia_gpu(vram_bytes));
        return Ok(Some(GpuProfile {
            name,
            vram_bytes,
            bandwidth_bytes_per_sec: bandwidth,
            fp16_tflops: tflops,
            backend: GpuBackend::Cuda,
        }));
    }
    tracing::debug!("No CUDA GPU detected; running CPU-only");
    Ok(None)
}

#[cfg(not(target_os = "macos"))]
fn query_cuda_device() -> Option<(String, u64)> {
    unsafe { hypura_sys::llama_backend_init() };

    let result = (|| -> Option<(String, u64)> {
        let reg_count = unsafe { hypura_sys::ggml_backend_reg_count() };
        let mut cuda_reg = std::ptr::null_mut();

        for i in 0..reg_count {
            let r = unsafe { hypura_sys::ggml_backend_reg_get(i) };
            if r.is_null() { continue; }
            let name_ptr = unsafe { hypura_sys::ggml_backend_reg_name(r) };
            if !name_ptr.is_null() {
                let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy();
                if name.contains("CUDA") || name.contains("NVIDIA") {
                    cuda_reg = r;
                    break;
                }
            }
        }
        if cuda_reg.is_null() {
            return None;
        }

        let dev_count = unsafe { hypura_sys::ggml_backend_reg_dev_count(cuda_reg) };
        if dev_count == 0 {
            return None;
        }

        let device = unsafe { hypura_sys::ggml_backend_reg_dev_get(cuda_reg, 0) };
        if device.is_null() {
            return None;
        }

        let desc_ptr = unsafe { hypura_sys::ggml_backend_dev_description(device) };
        let name = if desc_ptr.is_null() {
            "Unknown NVIDIA GPU".to_string()
        } else {
            unsafe { CStr::from_ptr(desc_ptr) }.to_string_lossy().to_string()
        };

        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe { hypura_sys::ggml_backend_dev_memory(device, &mut free, &mut total) };

        Some((name, total as u64))
    })();

    unsafe { hypura_sys::llama_backend_free() };
    result
}

// ── NVIDIA GPU spec database ──────────────────────────────────────────────────
// (bandwidth_gb_s, fp16_tflops) — ordered most-specific first to avoid
// "RTX 3060" matching before "RTX 3060 Ti".

struct NvidiaSpec {
    pattern: &'static str,
    bandwidth_gb_s: f64,
    fp16_tflops: f64,
}

const NVIDIA_SPECS: &[NvidiaSpec] = &[
    // ── Blackwell (RTX 50xx) ────────────────────────────────────────────────
    NvidiaSpec { pattern: "RTX 5090",      bandwidth_gb_s: 1792.0, fp16_tflops: 838.0  },
    NvidiaSpec { pattern: "RTX 5080",      bandwidth_gb_s: 960.0,  fp16_tflops: 464.0  },
    NvidiaSpec { pattern: "RTX 5070 Ti",   bandwidth_gb_s: 896.0,  fp16_tflops: 228.0  },
    NvidiaSpec { pattern: "RTX 5070",      bandwidth_gb_s: 672.0,  fp16_tflops: 176.0  },
    NvidiaSpec { pattern: "RTX 5060 Ti",   bandwidth_gb_s: 576.0,  fp16_tflops: 129.0  },
    NvidiaSpec { pattern: "RTX 5060",      bandwidth_gb_s: 448.0,  fp16_tflops: 92.0   },
    // ── Ada Lovelace (RTX 40xx) ─────────────────────────────────────────────
    NvidiaSpec { pattern: "RTX 4090",      bandwidth_gb_s: 1008.0, fp16_tflops: 165.2  },
    NvidiaSpec { pattern: "RTX 4080 Super",bandwidth_gb_s: 736.0,  fp16_tflops: 103.9  },
    NvidiaSpec { pattern: "RTX 4080",      bandwidth_gb_s: 717.0,  fp16_tflops: 97.5   },
    NvidiaSpec { pattern: "RTX 4070 Ti Super", bandwidth_gb_s: 672.0, fp16_tflops: 88.9 },
    NvidiaSpec { pattern: "RTX 4070 Ti",   bandwidth_gb_s: 504.0,  fp16_tflops: 80.8   },
    NvidiaSpec { pattern: "RTX 4070 Super",bandwidth_gb_s: 504.0,  fp16_tflops: 71.2   },
    NvidiaSpec { pattern: "RTX 4070",      bandwidth_gb_s: 504.0,  fp16_tflops: 58.0   },
    NvidiaSpec { pattern: "RTX 4060 Ti",   bandwidth_gb_s: 288.0,  fp16_tflops: 45.2   },
    NvidiaSpec { pattern: "RTX 4060",      bandwidth_gb_s: 272.0,  fp16_tflops: 30.1   },
    NvidiaSpec { pattern: "RTX 4050",      bandwidth_gb_s: 192.0,  fp16_tflops: 24.2   },
    // ── Ampere (RTX 30xx) ───────────────────────────────────────────────────
    NvidiaSpec { pattern: "RTX 3090 Ti",   bandwidth_gb_s: 1008.0, fp16_tflops: 80.0   },
    NvidiaSpec { pattern: "RTX 3090",      bandwidth_gb_s: 936.0,  fp16_tflops: 71.0   },
    NvidiaSpec { pattern: "RTX 3080 Ti",   bandwidth_gb_s: 912.0,  fp16_tflops: 65.0   },
    NvidiaSpec { pattern: "RTX 3080 12GB", bandwidth_gb_s: 912.0,  fp16_tflops: 60.0   },
    NvidiaSpec { pattern: "RTX 3080",      bandwidth_gb_s: 760.0,  fp16_tflops: 59.0   },
    NvidiaSpec { pattern: "RTX 3070 Ti",   bandwidth_gb_s: 608.0,  fp16_tflops: 43.5   },
    NvidiaSpec { pattern: "RTX 3070",      bandwidth_gb_s: 448.0,  fp16_tflops: 32.0   },
    NvidiaSpec { pattern: "RTX 3060 Ti",   bandwidth_gb_s: 448.0,  fp16_tflops: 29.4   },
    NvidiaSpec { pattern: "RTX 3060 12GB", bandwidth_gb_s: 360.0,  fp16_tflops: 25.4   },
    NvidiaSpec { pattern: "RTX 3060",      bandwidth_gb_s: 360.0,  fp16_tflops: 25.4   }, // base target
    NvidiaSpec { pattern: "RTX 3050",      bandwidth_gb_s: 224.0,  fp16_tflops: 16.0   },
    // ── Turing (RTX 20xx) ───────────────────────────────────────────────────
    NvidiaSpec { pattern: "RTX 2080 Ti",   bandwidth_gb_s: 616.0,  fp16_tflops: 53.8   },
    NvidiaSpec { pattern: "RTX 2080 Super",bandwidth_gb_s: 496.0,  fp16_tflops: 43.6   },
    NvidiaSpec { pattern: "RTX 2080",      bandwidth_gb_s: 448.0,  fp16_tflops: 40.5   },
    NvidiaSpec { pattern: "RTX 2070 Super",bandwidth_gb_s: 448.0,  fp16_tflops: 36.9   },
    NvidiaSpec { pattern: "RTX 2070",      bandwidth_gb_s: 448.0,  fp16_tflops: 28.9   },
    NvidiaSpec { pattern: "RTX 2060 Super",bandwidth_gb_s: 448.0,  fp16_tflops: 26.6   },
    NvidiaSpec { pattern: "RTX 2060",      bandwidth_gb_s: 336.0,  fp16_tflops: 21.2   },
    // ── Data centre / professional ──────────────────────────────────────────
    NvidiaSpec { pattern: "H200",          bandwidth_gb_s: 4800.0, fp16_tflops: 1979.0 },
    NvidiaSpec { pattern: "H100 SXM",      bandwidth_gb_s: 3350.0, fp16_tflops: 1979.0 },
    NvidiaSpec { pattern: "H100",          bandwidth_gb_s: 2000.0, fp16_tflops: 1979.0 },
    NvidiaSpec { pattern: "A100 SXM 80GB", bandwidth_gb_s: 2000.0, fp16_tflops: 312.0  },
    NvidiaSpec { pattern: "A100 80GB",     bandwidth_gb_s: 1935.0, fp16_tflops: 312.0  },
    NvidiaSpec { pattern: "A100",          bandwidth_gb_s: 1555.0, fp16_tflops: 312.0  },
    NvidiaSpec { pattern: "L40S",          bandwidth_gb_s: 864.0,  fp16_tflops: 366.0  },
    NvidiaSpec { pattern: "L40",           bandwidth_gb_s: 864.0,  fp16_tflops: 181.0  },
    NvidiaSpec { pattern: "A40",           bandwidth_gb_s: 696.0,  fp16_tflops: 149.7  },
];

fn lookup_nvidia_gpu(name: &str) -> Option<(u64, f64)> {
    NVIDIA_SPECS.iter().find(|s| name.contains(s.pattern)).map(|s| {
        ((s.bandwidth_gb_s * 1e9) as u64, s.fp16_tflops)
    })
}

/// Conservative estimate when the GPU model isn't in our database.
fn estimate_nvidia_gpu(vram_bytes: u64) -> (u64, f64) {
    // Very rough: ~1 TB/s bandwidth per 24 GB VRAM, ~100 TFLOPS FP16 per 24 GB
    let gb = vram_bytes as f64 / 1e9;
    let bw = (gb / 24.0 * 1_000_000_000_000.0) as u64;
    let tflops = gb / 24.0 * 100.0;
    (bw.max(200_000_000_000), tflops.max(5.0))
}

// ── Apple Silicon spec database ───────────────────────────────────────────────

#[cfg(target_os = "macos")]
struct AppleSiliconSpec {
    pattern: &'static str,
    bandwidth_gb_s: f64,
    fp16_tflops: f64,
}

#[cfg(target_os = "macos")]
const APPLE_SILICON_SPECS: &[AppleSiliconSpec] = &[
    AppleSiliconSpec { pattern: "M5 Ultra", bandwidth_gb_s: 900.0,  fp16_tflops: 40.0  },
    AppleSiliconSpec { pattern: "M5 Max",   bandwidth_gb_s: 600.0,  fp16_tflops: 20.0  },
    AppleSiliconSpec { pattern: "M5 Pro",   bandwidth_gb_s: 300.0,  fp16_tflops: 10.0  },
    AppleSiliconSpec { pattern: "M5",       bandwidth_gb_s: 120.0,  fp16_tflops: 4.5   },
    AppleSiliconSpec { pattern: "M4 Ultra", bandwidth_gb_s: 819.0,  fp16_tflops: 36.0  },
    AppleSiliconSpec { pattern: "M4 Max",   bandwidth_gb_s: 546.0,  fp16_tflops: 18.0  },
    AppleSiliconSpec { pattern: "M4 Pro",   bandwidth_gb_s: 273.0,  fp16_tflops: 9.0   },
    AppleSiliconSpec { pattern: "M4",       bandwidth_gb_s: 120.0,  fp16_tflops: 4.0   },
    AppleSiliconSpec { pattern: "M3 Ultra", bandwidth_gb_s: 800.0,  fp16_tflops: 28.0  },
    AppleSiliconSpec { pattern: "M3 Max",   bandwidth_gb_s: 400.0,  fp16_tflops: 14.0  },
    AppleSiliconSpec { pattern: "M3 Pro",   bandwidth_gb_s: 150.0,  fp16_tflops: 7.0   },
    AppleSiliconSpec { pattern: "M3",       bandwidth_gb_s: 100.0,  fp16_tflops: 3.5   },
    AppleSiliconSpec { pattern: "M2 Ultra", bandwidth_gb_s: 800.0,  fp16_tflops: 27.2  },
    AppleSiliconSpec { pattern: "M2 Max",   bandwidth_gb_s: 400.0,  fp16_tflops: 13.6  },
    AppleSiliconSpec { pattern: "M2 Pro",   bandwidth_gb_s: 200.0,  fp16_tflops: 6.8   },
    AppleSiliconSpec { pattern: "M2",       bandwidth_gb_s: 100.0,  fp16_tflops: 3.6   },
    AppleSiliconSpec { pattern: "M1 Ultra", bandwidth_gb_s: 800.0,  fp16_tflops: 20.8  },
    AppleSiliconSpec { pattern: "M1 Max",   bandwidth_gb_s: 400.0,  fp16_tflops: 10.4  },
    AppleSiliconSpec { pattern: "M1 Pro",   bandwidth_gb_s: 200.0,  fp16_tflops: 5.2   },
    AppleSiliconSpec { pattern: "M1",       bandwidth_gb_s:  68.25, fp16_tflops: 2.6   },
];

#[cfg(target_os = "macos")]
fn lookup_apple_silicon(name: &str) -> Option<&'static AppleSiliconSpec> {
    APPLE_SILICON_SPECS.iter().find(|s| name.contains(s.pattern))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_rtx3060() {
        let spec = lookup_nvidia_gpu("NVIDIA GeForce RTX 3060");
        assert!(spec.is_some());
        let (bw, tflops) = spec.unwrap();
        assert!(bw > 300_000_000_000); // > 300 GB/s
        assert!(tflops > 20.0);
    }

    #[test]
    fn test_lookup_rtx4090() {
        let (bw, tflops) = lookup_nvidia_gpu("NVIDIA GeForce RTX 4090").unwrap();
        assert!(tflops > 100.0);
        assert!(bw > 900_000_000_000);
    }

    #[test]
    fn test_lookup_unknown() {
        // Unknown GPU falls back to estimate
        let (bw, tflops) = estimate_nvidia_gpu(12 * 1024 * 1024 * 1024);
        assert!(bw > 0);
        assert!(tflops > 0.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_lookup_apple() {
        let spec = lookup_apple_silicon("Apple M2 Max").unwrap();
        assert!((spec.fp16_tflops - 13.6).abs() < 0.01);
    }
}
