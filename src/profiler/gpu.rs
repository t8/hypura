use std::ffi::CStr;

use crate::profiler::types::{GpuBackend, GpuProfile};

struct AppleSiliconSpec {
    pattern: &'static str,
    bandwidth_gb_s: f64,
    fp16_tflops: f64,
}

// Ordered specific-first: "M2 Max" before "M2"
const APPLE_SILICON_SPECS: &[AppleSiliconSpec] = &[
    AppleSiliconSpec { pattern: "M5 Ultra", bandwidth_gb_s: 900.0, fp16_tflops: 40.0 },
    AppleSiliconSpec { pattern: "M5 Max",   bandwidth_gb_s: 600.0, fp16_tflops: 20.0 },
    AppleSiliconSpec { pattern: "M5 Pro",   bandwidth_gb_s: 300.0, fp16_tflops: 10.0 },
    AppleSiliconSpec { pattern: "M5",       bandwidth_gb_s: 120.0, fp16_tflops: 4.5  },
    AppleSiliconSpec { pattern: "M4 Ultra", bandwidth_gb_s: 819.0, fp16_tflops: 36.0 },
    AppleSiliconSpec { pattern: "M4 Max",   bandwidth_gb_s: 546.0, fp16_tflops: 18.0 },
    AppleSiliconSpec { pattern: "M4 Pro",   bandwidth_gb_s: 273.0, fp16_tflops: 9.0  },
    AppleSiliconSpec { pattern: "M4",       bandwidth_gb_s: 120.0, fp16_tflops: 4.0  },
    AppleSiliconSpec { pattern: "M3 Ultra", bandwidth_gb_s: 800.0, fp16_tflops: 28.0 },
    AppleSiliconSpec { pattern: "M3 Max",   bandwidth_gb_s: 400.0, fp16_tflops: 14.0 },
    AppleSiliconSpec { pattern: "M3 Pro",   bandwidth_gb_s: 150.0, fp16_tflops: 7.0  },
    AppleSiliconSpec { pattern: "M3",       bandwidth_gb_s: 100.0, fp16_tflops: 3.5  },
    AppleSiliconSpec { pattern: "M2 Ultra", bandwidth_gb_s: 800.0, fp16_tflops: 27.2 },
    AppleSiliconSpec { pattern: "M2 Max",   bandwidth_gb_s: 400.0, fp16_tflops: 13.6 },
    AppleSiliconSpec { pattern: "M2 Pro",   bandwidth_gb_s: 200.0, fp16_tflops: 6.8  },
    AppleSiliconSpec { pattern: "M2",       bandwidth_gb_s: 100.0, fp16_tflops: 3.6  },
    AppleSiliconSpec { pattern: "M1 Ultra", bandwidth_gb_s: 800.0, fp16_tflops: 20.8 },
    AppleSiliconSpec { pattern: "M1 Max",   bandwidth_gb_s: 400.0, fp16_tflops: 10.4 },
    AppleSiliconSpec { pattern: "M1 Pro",   bandwidth_gb_s: 200.0, fp16_tflops: 5.2  },
    AppleSiliconSpec { pattern: "M1",       bandwidth_gb_s: 68.25, fp16_tflops: 2.6  },
];

pub fn profile_gpu() -> anyhow::Result<Option<GpuProfile>> {
    let (name, vram_bytes) = match query_metal_device() {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("No Metal GPU detected: {e}");
            return Ok(None);
        }
    };

    let (bandwidth, tflops) = match lookup_apple_silicon(&name) {
        Some(spec) => (
            (spec.bandwidth_gb_s * 1e9) as u64,
            spec.fp16_tflops,
        ),
        None => {
            tracing::warn!("Unknown GPU '{name}', using conservative estimates");
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

fn query_metal_device() -> anyhow::Result<(String, u64)> {
    unsafe {
        hypura_sys::llama_backend_init();
    }

    let result = (|| -> anyhow::Result<(String, u64)> {
        // The Metal backend registers as "MTL", not "Metal"
        let reg_count = unsafe { hypura_sys::ggml_backend_reg_count() };
        let mut reg = std::ptr::null_mut();

        for i in 0..reg_count {
            let r = unsafe { hypura_sys::ggml_backend_reg_get(i) };
            if r.is_null() {
                continue;
            }
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

        // Get device description (GPU name)
        let desc_ptr = unsafe { hypura_sys::ggml_backend_dev_description(device) };
        let name = if desc_ptr.is_null() {
            "Unknown Metal GPU".to_string()
        } else {
            unsafe { CStr::from_ptr(desc_ptr) }.to_string_lossy().to_string()
        };

        // Get device memory
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            hypura_sys::ggml_backend_dev_memory(device, &mut free, &mut total);
        }

        Ok((name, total as u64))
    })();

    unsafe {
        hypura_sys::llama_backend_free();
    }

    result
}

fn lookup_apple_silicon(name: &str) -> Option<&'static AppleSiliconSpec> {
    APPLE_SILICON_SPECS.iter().find(|spec| name.contains(spec.pattern))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_specificity() {
        let spec = lookup_apple_silicon("Apple M2 Max").unwrap();
        assert_eq!(spec.pattern, "M2 Max");
        assert!((spec.fp16_tflops - 13.6).abs() < 0.01);

        let spec = lookup_apple_silicon("Apple M2").unwrap();
        assert_eq!(spec.pattern, "M2");

        let spec = lookup_apple_silicon("Apple M1 Pro").unwrap();
        assert_eq!(spec.pattern, "M1 Pro");
    }

    #[test]
    fn test_lookup_unknown() {
        assert!(lookup_apple_silicon("NVIDIA RTX 4090").is_none());
    }

    #[test]
    fn test_profile_gpu_returns_some() {
        let gpu = profile_gpu().unwrap();
        // On Apple Silicon, we should always get a GPU
        if cfg!(target_arch = "aarch64") {
            assert!(gpu.is_some());
            let gpu = gpu.unwrap();
            assert!(!gpu.name.is_empty());
            assert!(gpu.vram_bytes > 0);
        }
    }
}
