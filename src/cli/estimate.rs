use std::path::Path;

use hypura::model::{gguf::GgufFile, metadata::ModelMetadata};
use hypura::profiler;
use hypura::scheduler::estimator::{estimate_performance, EstimateConfidence};
use hypura::scheduler::placement::{compute_placement, summarize_placement};

use super::fmt_util::{format_bytes, format_params};

pub fn run(model_path: &str) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    anyhow::ensure!(path.exists(), "Model file not found: {model_path}");

    // Load or create hardware profile
    let hardware = match profiler::load_cached_profile()? {
        Some(p) if !profiler::is_profile_stale(&p) => {
            println!("Using cached hardware profile.");
            p
        }
        _ => {
            println!("No hardware profile found. Running profiler...");
            let p = profiler::run_full_profile()?;
            profiler::save_profile(&p)?;
            p
        }
    };

    // Parse model
    println!("Parsing model...");
    let gguf = GgufFile::open(path)?;
    let metadata = ModelMetadata::from_gguf(&gguf)?;

    // Compute placement
    println!("Computing optimal placement...");
    let placement = compute_placement(&gguf, &hardware)?;

    // Estimate performance
    let estimate = estimate_performance(&gguf, &metadata, &hardware, &placement)?;
    let summary = summarize_placement(&placement.tier_assignments, &gguf.tensors);

    // Print results
    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    println!();
    println!("Hypura Estimate: {filename}");
    println!("{}", "─".repeat(48));

    println!();
    println!("  Model");
    println!(
        "    Architecture:   {} ({} params)",
        metadata.architecture,
        format_params(metadata.parameter_count)
    );
    if let Some(ref q) = metadata.quantization {
        println!("    Quantization:   {q}");
    }
    println!(
        "    Size:           {}",
        format_bytes(gguf.total_tensor_bytes())
    );
    println!("    Context:        {} tokens", metadata.context_length);
    if metadata.is_moe {
        println!(
            "    MoE:            {} experts, {} active",
            metadata.num_experts.unwrap_or(0),
            metadata.num_experts_used.unwrap_or(0)
        );
    }

    println!();
    println!("  Placement Plan");
    if summary.total_gpu_bytes > 0 {
        println!(
            "    GPU (Metal):    {:>10}   ({} layers)",
            format_bytes(summary.total_gpu_bytes),
            summary.layers_on_gpu
        );
    }
    if summary.total_ram_bytes > 0 {
        println!(
            "    RAM:            {:>10}   ({} layers)",
            format_bytes(summary.total_ram_bytes),
            summary.layers_in_ram
        );
    }
    if summary.total_nvme_bytes > 0 {
        println!(
            "    NVMe:           {:>10}   ({} layers)",
            format_bytes(summary.total_nvme_bytes),
            summary.layers_on_nvme
        );
    }

    println!();
    println!("  KV Cache");
    println!(
        "    Hot window:     {} tokens ({:?}, FP16)    {}",
        placement.kv_cache_plan.hot_window_tokens,
        placement.kv_cache_plan.hot_tier,
        format_bytes(placement.kv_cache_plan.hot_bytes),
    );
    println!(
        "    Warm window:    {} tokens ({:?}, Q8)      {}",
        placement.kv_cache_plan.warm_window_tokens,
        placement.kv_cache_plan.warm_tier,
        format_bytes(placement.kv_cache_plan.warm_bytes),
    );
    println!(
        "    Spill context:  >{} tokens",
        estimate.max_context_before_spill
    );

    println!();
    println!("  Performance Estimate");
    println!(
        "    Interactive:    {:.1} tok/s        [{}]",
        estimate.estimated_tok_per_sec_interactive,
        estimate.experience_tier.label(),
    );
    println!(
        "    Batched (x8):   {:.1} tok/s",
        estimate.estimated_tok_per_sec_batched,
    );
    if estimate.disk_read_per_token_bytes > 0 {
        println!(
            "    Disk I/O:       {}/token",
            format_bytes(estimate.disk_read_per_token_bytes),
        );
    }
    println!(
        "    Confidence:     {}",
        match estimate.confidence {
            EstimateConfidence::Measured => "Measured",
            EstimateConfidence::Predicted => "Predicted",
            EstimateConfidence::Interpolated => "Interpolated",
        }
    );

    println!();
    println!(
        "  Experience: {} — {}",
        estimate.experience_tier.label(),
        estimate.experience_tier.description()
    );

    Ok(())
}
