pub mod cpu;
pub mod gpu;
pub mod memory;
pub mod storage;
pub mod types;

use std::path::PathBuf;

use chrono::Utc;

use crate::profiler::types::{HardwareProfile, SystemInfo};

/// Run the full hardware profiling suite.
pub fn run_full_profile() -> anyhow::Result<HardwareProfile> {
    tracing::info!("Profiling CPU...");
    let cpu = cpu::profile_cpu()?;

    tracing::info!("Profiling memory...");
    let memory_profile = memory::profile_memory()?;

    tracing::info!("Profiling GPU...");
    let gpu = gpu::profile_gpu()?;

    tracing::info!("Profiling storage...");
    let storage = storage::profile_storage()?;

    let total_cores = cpu::sysctl_u32("hw.ncpu").unwrap_or(1);
    let machine_model = cpu::sysctl_string("hw.model").unwrap_or_else(|_| "Unknown".into());

    let system = SystemInfo {
        os: format!("{} {}", std::env::consts::OS, os_version()),
        arch: std::env::consts::ARCH.to_string(),
        machine_model,
        total_cores,
    };

    Ok(HardwareProfile {
        timestamp: Utc::now(),
        system,
        memory: memory_profile,
        gpu,
        storage,
        cpu,
    })
}

/// Returns the path to `~/.hypura/`, creating it if necessary.
pub fn profile_dir() -> anyhow::Result<PathBuf> {
    let dir = dirs_path();
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }
    Ok(dir)
}

/// Save a hardware profile to `~/.hypura/hardware_profile.json`.
pub fn save_profile(profile: &HardwareProfile) -> anyhow::Result<PathBuf> {
    let dir = profile_dir()?;
    let path = dir.join("hardware_profile.json");
    let json = serde_json::to_string_pretty(profile)?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Load a cached hardware profile, if one exists.
pub fn load_cached_profile() -> anyhow::Result<Option<HardwareProfile>> {
    let path = dirs_path().join("hardware_profile.json");
    if !path.exists() {
        return Ok(None);
    }
    let json = std::fs::read_to_string(&path)?;
    let profile: HardwareProfile = serde_json::from_str(&json)?;
    Ok(Some(profile))
}

/// Returns true if the profile is older than 30 days.
pub fn is_profile_stale(profile: &HardwareProfile) -> bool {
    let age = Utc::now() - profile.timestamp;
    age.num_days() > 30
}

fn dirs_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".hypura")
}

fn os_version() -> String {
    cpu::sysctl_string("kern.osproductversion").unwrap_or_else(|_| "unknown".into())
}
