use hypura::profiler;
use hypura::profiler::types::HardwareProfile;

use super::fmt_util::{format_bandwidth, format_bytes};

pub fn run(force: bool) -> anyhow::Result<()> {
    if !force {
        if let Ok(Some(cached)) = profiler::load_cached_profile() {
            if !profiler::is_profile_stale(&cached) {
                print_profile(&cached);
                println!();
                println!("Profile is up to date. Use --force to re-profile.");
                return Ok(());
            }
            println!("Profile is stale (>30 days). Re-profiling...");
        }
    }

    println!("Running hardware profiler...");
    let profile = profiler::run_full_profile()?;
    let path = profiler::save_profile(&profile)?;

    print_profile(&profile);
    println!();
    println!("Saved to {}", path.display());

    Ok(())
}

fn print_profile(p: &HardwareProfile) {
    println!();
    println!("Hardware Profile ({})", p.timestamp.format("%Y-%m-%dT%H:%M:%SZ"));
    println!("{}", "─".repeat(48));

    println!();
    println!("  System");
    println!("    Machine:     {} ({} {})", p.system.machine_model, p.system.os, p.system.arch);
    println!(
        "    CPU:         {} ({}P + {}E cores)",
        p.cpu.model_name, p.cpu.cores_performance, p.cpu.cores_efficiency
    );

    println!();
    println!("  Memory");
    println!(
        "    Total:       {}{}",
        format_bytes(p.memory.total_bytes),
        if p.memory.is_unified { " (unified)" } else { "" }
    );
    println!("    Available:   {}", format_bytes(p.memory.available_bytes));
    println!("    Bandwidth:   {}", format_bandwidth(p.memory.bandwidth_bytes_per_sec));

    if let Some(ref gpu) = p.gpu {
        println!();
        println!("  GPU ({:?})", gpu.backend);
        println!("    Name:        {}", gpu.name);
        println!(
            "    VRAM:        {}{}",
            format_bytes(gpu.vram_bytes),
            if p.memory.is_unified { " (shared)" } else { "" }
        );
        println!("    Bandwidth:   {}", format_bandwidth(gpu.bandwidth_bytes_per_sec));
        println!("    FP16:        {:.1} TFLOPS", gpu.fp16_tflops);
    }

    for s in &p.storage {
        println!();
        println!("  Storage ({})", s.mount_point);
        println!("    Type:        {:?}", s.device_type);
        println!(
            "    Capacity:    {} ({} free)",
            format_bytes(s.capacity_bytes),
            format_bytes(s.free_bytes)
        );
        println!(
            "    Sequential:  {} (peak @ {} blocks)",
            format_bandwidth(s.sequential_read.peak_sequential),
            format_bytes(
                s.sequential_read
                    .points
                    .iter()
                    .max_by_key(|(_, bw)| bw)
                    .map(|(bs, _)| *bs)
                    .unwrap_or(0)
            ),
        );
        println!("    Random 4K:   {} IOPS", format_iops(s.random_read_iops));
    }
}

fn format_iops(iops: u64) -> String {
    if iops >= 1_000_000 {
        format!("{:.1}M", iops as f64 / 1e6)
    } else if iops >= 1_000 {
        format!("{},{:03}", iops / 1000, iops % 1000)
    } else {
        format!("{iops}")
    }
}
