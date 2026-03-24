pub mod cpu;
pub mod gpu;
pub mod memory;
pub mod storage;
pub mod types;

use chrono::Utc;

use crate::profiler::types::{HardwareProfile, SystemInfo};

/// Run the full hardware profiling suite.
pub fn run_full_profile() -> anyhow::Result<HardwareProfile> {
    tracing::info!("Profiling CPU...");
    let cpu_profile = cpu::profile_cpu()?;

    tracing::info!("Profiling memory...");
    let memory_profile = memory::profile_memory()?;

    tracing::info!("Profiling GPU...");
    let gpu = gpu::profile_gpu()?;

    tracing::info!("Profiling storage...");
    let storage = storage::profile_storage()?;

    let system = SystemInfo {
        os: format!("{} {}", std::env::consts::OS, os_version()),
        arch: std::env::consts::ARCH.to_string(),
        machine_model: machine_model(),
        total_cores: total_cpu_count(),
    };

    Ok(HardwareProfile {
        timestamp: Utc::now(),
        system,
        memory: memory_profile,
        gpu,
        storage,
        cpu: cpu_profile,
    })
}

/// Returns the path to the Hypura data directory, creating it if necessary.
///
/// - Windows: `%APPDATA%\Hypura`
/// - macOS / Linux: `~/.hypura`
pub fn profile_dir() -> anyhow::Result<std::path::PathBuf> {
    let dir = data_dir();
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }
    Ok(dir)
}

/// Save a hardware profile to `<data_dir>/hardware_profile.json`.
pub fn save_profile(profile: &HardwareProfile) -> anyhow::Result<std::path::PathBuf> {
    let dir = profile_dir()?;
    let path = dir.join("hardware_profile.json");
    let json = serde_json::to_string_pretty(profile)?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Load a cached hardware profile, if one exists.
pub fn load_cached_profile() -> anyhow::Result<Option<HardwareProfile>> {
    let path = data_dir().join("hardware_profile.json");
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

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Platform-specific data directory.
fn data_dir() -> std::path::PathBuf {
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| "C:\\Users\\Default\\AppData\\Roaming".into());
        std::path::PathBuf::from(appdata).join("Hypura")
    }
    #[cfg(not(target_os = "windows"))]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        std::path::PathBuf::from(home).join(".hypura")
    }
}

/// Total logical CPU count.
fn total_cpu_count() -> u32 {
    #[cfg(target_os = "macos")]
    {
        cpu::sysctl_u32("hw.ncpu").unwrap_or(1)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut sys = sysinfo::System::new();
        sys.refresh_cpu_all();
        sys.cpus().len() as u32
    }
}

/// Human-readable machine / system model string.
fn machine_model() -> String {
    #[cfg(target_os = "macos")]
    {
        cpu::sysctl_string("hw.model").unwrap_or_else(|_| "Unknown".into())
    }
    #[cfg(target_os = "linux")]
    {
        // WSL2 exposes DMI product name in /sys
        if let Ok(model) = std::fs::read_to_string("/sys/class/dmi/id/product_name") {
            let trimmed = model.trim().to_string();
            if !trimmed.is_empty() && trimmed != "None" {
                return trimmed;
            }
        }
        "Unknown Linux machine".to_string()
    }
    #[cfg(target_os = "windows")]
    {
        // Could query WMI here; for now return a static string
        "Windows PC".to_string()
    }
}

/// OS version string.
fn os_version() -> String {
    #[cfg(target_os = "macos")]
    {
        cpu::sysctl_string("kern.osproductversion").unwrap_or_else(|_| "unknown".into())
    }
    #[cfg(target_os = "linux")]
    {
        // /etc/os-release is the standard on modern Linux distros
        if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
            for line in content.lines() {
                if let Some(val) = line.strip_prefix("PRETTY_NAME=") {
                    return val.trim_matches('"').to_string();
                }
            }
        }
        // Fallback: kernel version
        let mut uname = sysinfo::System::kernel_version()
            .unwrap_or_else(|| "unknown".to_string());
        uname
    }
    #[cfg(target_os = "windows")]
    {
        sysinfo::System::os_version().unwrap_or_else(|| "unknown".to_string())
    }
}
