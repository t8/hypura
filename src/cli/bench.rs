use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use hypura::compute::ffi::SamplingParams;
use hypura::compute::inference::*;
use hypura::model::gguf::GgufFile;
use hypura::model::metadata::ModelMetadata;
use hypura::profiler;
use hypura::scheduler::placement::{compute_placement_with_context, summarize_placement};
use hypura::scheduler::types::StorageTier;
use hypura::telemetry::metrics::TelemetryEmitter;

use super::fmt_util::{format_bytes, format_params};

const DEFAULT_PROMPT: &str = "Write a short paragraph about artificial intelligence.";

pub fn run(
    model_path: &str,
    baseline: bool,
    context: u32,
    max_tokens: u32,
    prompt: Option<&str>,
    force: bool,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(model_path, baseline, context, max_tokens, prompt, force))
}

async fn run_async(
    model_path: &str,
    run_baseline: bool,
    context: u32,
    max_tokens: u32,
    prompt: Option<&str>,
    force: bool,
) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    anyhow::ensure!(path.exists(), "Model file not found: {model_path}");

    let prompt_text = prompt.unwrap_or(DEFAULT_PROMPT);

    // Load hardware profile
    let hardware = match profiler::load_cached_profile()? {
        Some(p) if !profiler::is_profile_stale(&p) => p,
        _ => {
            println!("Running hardware profiler...");
            let p = profiler::run_full_profile()?;
            profiler::save_profile(&p)?;
            p
        }
    };

    // Parse model
    let gguf = GgufFile::open(path)?;
    let metadata = ModelMetadata::from_gguf(&gguf)?;
    let plan = compute_placement_with_context(&gguf, &hardware, context)?;
    let summary = summarize_placement(&plan.tier_assignments, &gguf.tensors);
    let gpu_budget = compute_gpu_budget(&hardware, &metadata, context);
    let n_gpu_layers = gpu_layers_from_placement(&plan, &gguf, gpu_budget);

    let has_nvme = plan
        .tier_assignments
        .values()
        .any(|t| *t == StorageTier::Nvme);

    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Print header
    println!();
    println!("Hypura Benchmark: {filename}");
    println!("{}", "─".repeat(56));
    println!(
        "  Model: {} ({} params, {}, {})",
        metadata.architecture,
        format_params(metadata.parameter_count),
        metadata.quantization.as_deref().unwrap_or("unknown"),
        format_bytes(gguf.total_tensor_bytes()),
    );
    println!(
        "  Hardware: {}, {} unified",
        hardware.cpu.model_name,
        format_bytes(hardware.memory.total_bytes),
    );
    println!(
        "  Placement: {} GPU | {} RAM | {} NVMe",
        format_bytes(summary.total_gpu_bytes),
        format_bytes(summary.total_ram_bytes),
        format_bytes(summary.total_nvme_bytes),
    );
    println!(
        "  Config: context={}, max_tokens={}, n_gpu_layers={}",
        context, max_tokens, n_gpu_layers,
    );
    println!();

    // Memory safety check for the baseline run only.
    // The Hypura run is designed to handle models larger than RAM via NVMe scheduling —
    // that's the whole point. But the baseline (naive mmap) has no such mechanism and
    // will cause severe swap thrashing if the full model exceeds physical memory.
    let model_total_bytes = gguf.total_tensor_bytes();
    let total_ram = hardware.memory.total_bytes;
    let min_headroom: u64 = 4 * (1 << 30);
    let baseline_safe = model_total_bytes <= total_ram.saturating_sub(min_headroom);

    let config = InferenceConfig {
        n_ctx: context,
        n_batch: 512,
        n_threads: InferenceConfig::default().n_threads,
        sampling: SamplingParams {
            max_tokens,
            seed: 42, // fixed seed for reproducibility
            ..SamplingParams::default()
        },
    };

    // --- Baseline run (naive mmap) ---
    let mut baseline_result = None;
    if run_baseline {
        if !baseline_safe {
            let model_gb = model_total_bytes as f64 / (1u64 << 30) as f64;
            let total_gb = total_ram as f64 / (1u64 << 30) as f64;
            if force {
                println!("  WARNING: model ({model_gb:.1} GB) exceeds RAM ({total_gb:.1} GB) — baseline forced, expect heavy swap pressure.");
            } else {
                println!("  Skipping baseline: model ({model_gb:.1} GB) exceeds RAM ({total_gb:.1} GB).");
                println!("  Naive mmap of this model will cause severe swap thrashing / OOM.");
                println!("  Use --force to run baseline anyway.");
                println!();
            }
        }

        if baseline_safe || force {
            println!("  Run 1: Naive mmap (baseline)");
            println!("    Running...");

            let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel();
            let path_c = path.to_path_buf();
            let prompt_c = prompt_text.to_string();
            let config_c = config.clone();
            let telemetry = Arc::new(TelemetryEmitter::new(64));

            let wall_start = Instant::now();
            let handle = tokio::task::spawn_blocking(move || {
                generate_blocking(&path_c, &prompt_c, &config_c, n_gpu_layers, token_tx, telemetry)
            });

            // Drain tokens (discard output)
            while token_rx.recv().await.is_some() {}

            match handle.await {
                Ok(Ok(result)) => {
                    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
                    print_run_result("Naive mmap (baseline)", &result, wall_ms);
                    baseline_result = Some((result, wall_ms));
                }
                Ok(Err(e)) => {
                    println!("    Baseline failed: {e}");
                    println!("    Continuing with Hypura run...");
                    println!();
                }
                Err(e) => {
                    println!("    Baseline panicked: {e}");
                    println!("    Continuing with Hypura run...");
                    println!();
                }
            }

            // When model exceeds RAM, give the OS time to reclaim pages from the
            // baseline's GPU/mmap buffers before loading again for the Hypura run.
            // Without this, back-to-back loads can spike to 2x model size and OOM.
            if !baseline_safe && has_nvme {
                println!("  Reclaiming memory before Hypura run...");
                std::thread::sleep(std::time::Duration::from_secs(3));
            }
        }
    }

    // --- Hypura run ---
    let run_label = if has_nvme {
        "Hypura NVMe scheduling"
    } else {
        "Hypura (all in GPU+RAM)"
    };
    println!("  {}: {run_label}", if run_baseline { "Run 2" } else { "Run" });
    println!("    Running...");

    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel();
    let path_c = path.to_path_buf();
    let prompt_c = prompt_text.to_string();
    let config_c = config.clone();
    let plan_c = Arc::new(plan.clone());
    let gguf_c = Arc::new(gguf.clone());
    let telemetry = Arc::new(TelemetryEmitter::new(64));

    let wall_start = Instant::now();
    let handle = tokio::task::spawn_blocking(move || {
        generate_with_nvme_scheduling(
            &path_c, &prompt_c, &config_c, n_gpu_layers, &plan_c, &gguf_c, token_tx, telemetry,
        )
    });

    while token_rx.recv().await.is_some() {}
    let hypura_result = handle.await??;
    let hypura_wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

    print_run_result(run_label, &hypura_result, hypura_wall_ms);

    // --- Speedup ---
    if let Some((ref base, _)) = baseline_result {
        if base.tok_per_sec_avg > 0.0 {
            let speedup = hypura_result.tok_per_sec_avg / base.tok_per_sec_avg;
            println!();
            println!("  Speedup: {speedup:.1}x (generation tok/s)");
        }
    }

    // --- Save results ---
    let bench_result = BenchmarkResult {
        timestamp: chrono::Utc::now().to_rfc3339(),
        model: ModelInfo {
            name: filename.trim_end_matches(".gguf").to_string(),
            architecture: metadata.architecture.clone(),
            params: format_params(metadata.parameter_count),
            quant: metadata.quantization.clone().unwrap_or_default(),
            size_gb: gguf.total_tensor_bytes() as f64 / (1u64 << 30) as f64,
        },
        hardware: HardwareInfo {
            cpu: hardware.cpu.model_name.clone(),
            ram_gb: hardware.memory.total_bytes as f64 / (1u64 << 30) as f64,
            gpu: hardware.gpu.as_ref().map(|g| g.name.clone()).unwrap_or_default(),
            nvme_seq_gbps: hardware
                .storage
                .first()
                .map(|s| s.sequential_read.peak_sequential as f64 / 1e9)
                .unwrap_or(0.0),
        },
        placement: PlacementInfo {
            gpu_gb: summary.total_gpu_bytes as f64 / (1u64 << 30) as f64,
            ram_gb: summary.total_ram_bytes as f64 / (1u64 << 30) as f64,
            nvme_gb: summary.total_nvme_bytes as f64 / (1u64 << 30) as f64,
        },
        config: BenchConfig {
            context,
            max_tokens,
            prompt: prompt_text.to_string(),
            n_gpu_layers,
        },
        baseline: baseline_result.as_ref().map(|(r, wall)| RunResult {
            prompt_eval_ms: r.prompt_eval_ms,
            tok_per_sec: r.tok_per_sec_avg,
            tokens_generated: r.tokens_generated,
            wall_time_ms: *wall,
        }),
        hypura: RunResult {
            prompt_eval_ms: hypura_result.prompt_eval_ms,
            tok_per_sec: hypura_result.tok_per_sec_avg,
            tokens_generated: hypura_result.tokens_generated,
            wall_time_ms: hypura_wall_ms,
        },
        speedup: baseline_result
            .as_ref()
            .and_then(|(b, _)| {
                if b.tok_per_sec_avg > 0.0 {
                    Some(hypura_result.tok_per_sec_avg / b.tok_per_sec_avg)
                } else {
                    None
                }
            }),
    };

    save_benchmark_result(&bench_result)?;
    println!();
    println!("  Results saved to benchmarks/results/");

    Ok(())
}

fn print_run_result(_label: &str, result: &GenerationResult, wall_ms: f64) {
    let prompt_tps = if result.prompt_eval_ms > 0.0 {
        result.prompt_tokens as f64 / (result.prompt_eval_ms / 1000.0)
    } else {
        0.0
    };
    println!("    Prompt eval:    {:.1}s ({:.1} tok/s)", result.prompt_eval_ms / 1000.0, prompt_tps);
    println!("    Generation:     {:.1} tok/s ({} tokens)", result.tok_per_sec_avg, result.tokens_generated);
    println!("    Wall time:      {:.1}s", wall_ms / 1000.0);
    println!();
}

// --- Serializable result types ---

#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    timestamp: String,
    model: ModelInfo,
    hardware: HardwareInfo,
    placement: PlacementInfo,
    config: BenchConfig,
    baseline: Option<RunResult>,
    hypura: RunResult,
    speedup: Option<f64>,
}

#[derive(Serialize, Deserialize)]
struct ModelInfo {
    name: String,
    architecture: String,
    params: String,
    quant: String,
    size_gb: f64,
}

#[derive(Serialize, Deserialize)]
struct HardwareInfo {
    cpu: String,
    ram_gb: f64,
    gpu: String,
    nvme_seq_gbps: f64,
}

#[derive(Serialize, Deserialize)]
struct PlacementInfo {
    gpu_gb: f64,
    ram_gb: f64,
    nvme_gb: f64,
}

#[derive(Serialize, Deserialize)]
struct BenchConfig {
    context: u32,
    max_tokens: u32,
    prompt: String,
    n_gpu_layers: i32,
}

#[derive(Serialize, Deserialize)]
struct RunResult {
    prompt_eval_ms: f64,
    tok_per_sec: f64,
    tokens_generated: u32,
    wall_time_ms: f64,
}

fn save_benchmark_result(result: &BenchmarkResult) -> anyhow::Result<()> {
    let dir = Path::new("benchmarks/results");
    std::fs::create_dir_all(dir)?;

    // Save JSON
    let ts = result.timestamp.replace(':', "-").replace('.', "-");
    let filename = format!("{}_{}.json", &ts[..19], result.model.name);
    let json_path = dir.join(&filename);
    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(&json_path, &json)?;

    // Update benchmarks/README.md
    update_readme(result)?;

    Ok(())
}

fn update_readme(result: &BenchmarkResult) -> anyhow::Result<()> {
    let readme_path = Path::new("benchmarks/README.md");

    let header = "# Hypura Benchmarks\n\n\
        | Date | Model | Hardware | GPU | RAM | NVMe | Baseline tok/s | Hypura tok/s | Speedup |\n\
        |------|-------|----------|-----|-----|------|----------------|--------------|---------|";

    let existing = if readme_path.exists() {
        std::fs::read_to_string(readme_path)?
    } else {
        format!("{}\n", header)
    };

    let baseline_str = result
        .baseline
        .as_ref()
        .map(|b| format!("{:.1}", b.tok_per_sec))
        .unwrap_or_else(|| "—".to_string());

    let speedup_str = result
        .speedup
        .map(|s| format!("{s:.1}x"))
        .unwrap_or_else(|| "—".to_string());

    let date = &result.timestamp[..10];
    let row = format!(
        "| {date} | {} {} | {} {}GB | {:.1} GB | {:.1} GB | {:.1} GB | {baseline_str} | {:.1} | {speedup_str} |",
        result.model.name,
        result.model.quant,
        result.hardware.cpu,
        result.hardware.ram_gb as u32,
        result.placement.gpu_gb,
        result.placement.ram_gb,
        result.placement.nvme_gb,
        result.hypura.tok_per_sec,
    );

    let content = format!("{}\n{}\n", existing.trim_end(), row);
    std::fs::write(readme_path, content)?;

    Ok(())
}
