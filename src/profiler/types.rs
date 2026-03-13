use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub timestamp: DateTime<Utc>,
    pub system: SystemInfo,
    pub memory: MemoryProfile,
    pub gpu: Option<GpuProfile>,
    pub storage: Vec<StorageProfile>,
    pub cpu: CpuProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub machine_model: String,
    pub total_cores: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub total_bytes: u64,
    pub available_bytes: u64,
    /// Measured via memcpy benchmark (bytes/sec)
    pub bandwidth_bytes_per_sec: u64,
    /// On Apple Silicon, unified memory is shared with GPU
    pub is_unified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    pub name: String,
    pub vram_bytes: u64,
    /// Measured via matmul benchmark (bytes/sec)
    pub bandwidth_bytes_per_sec: u64,
    /// TFLOPS for FP16 matmul
    pub fp16_tflops: f64,
    pub backend: GpuBackend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuBackend {
    Metal,
    Cuda,
    Rocm,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageProfile {
    pub device_path: String,
    pub mount_point: String,
    pub device_type: StorageType,
    pub capacity_bytes: u64,
    pub free_bytes: u64,
    /// Measured at various block sizes
    pub sequential_read: BandwidthCurve,
    /// IOPS at 4K random read
    pub random_read_iops: u64,
    /// Detected PCIe generation
    pub pcie_gen: Option<u8>,
    /// S.M.A.R.T. wear indicator (0.0 = new, 1.0 = end of life)
    pub wear_level: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthCurve {
    /// (block_size_bytes, measured_bandwidth_bytes_per_sec)
    pub points: Vec<(u64, u64)>,
    /// Best sustained sequential read (bytes/sec)
    pub peak_sequential: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    NvmePcie,
    Sata,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    pub model_name: String,
    pub cores_performance: u32,
    pub cores_efficiency: u32,
    pub has_amx: bool,
    pub has_neon: bool,
    pub has_avx512: bool,
    pub has_avx2: bool,
    /// Measured FLOPS for INT8 matmul
    pub int8_gflops: f64,
}
