use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;

use hypura::compute::inference::*;
use hypura::model::gguf::GgufFile;
use hypura::profiler;
use hypura::scheduler::placement::{compute_placement, summarize_placement};
use hypura::scheduler::types::{PlacementSummary, StorageTier};
use hypura::telemetry::metrics::TelemetryEmitter;

use super::fmt_util::format_bytes;

pub fn run(
    model_path: &str,
    context: u32,
    prompt: Option<&str>,
    interactive: bool,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(model_path, context, prompt, interactive))
}

async fn run_async(
    model_path: &str,
    context: u32,
    prompt: Option<&str>,
    interactive: bool,
) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    anyhow::ensure!(path.exists(), "Model file not found: {model_path}");

    // Load or create hardware profile
    let hardware = match profiler::load_cached_profile()? {
        Some(p) if !profiler::is_profile_stale(&p) => p,
        _ => {
            println!("No hardware profile found. Running profiler...");
            let p = profiler::run_full_profile()?;
            profiler::save_profile(&p)?;
            p
        }
    };

    // Parse GGUF header for placement
    let gguf = GgufFile::open(path)?;
    let plan = compute_placement(&gguf, &hardware)?;
    let summary = summarize_placement(&plan.tier_assignments, &gguf.tensors);
    let n_gpu_layers = gpu_layers_from_placement(&plan, &gguf);

    let has_nvme = plan.tier_assignments.values().any(|t| *t == StorageTier::Nvme);
    if has_nvme {
        println!("  NVMe scheduling: ENABLED ({} tensors on SSD)",
            plan.tier_assignments.values().filter(|t| **t == StorageTier::Nvme).count());
    }

    print_placement_header(&summary, &plan, n_gpu_layers);

    let telemetry = Arc::new(TelemetryEmitter::new(256));
    let config = InferenceConfig {
        n_ctx: context,
        ..InferenceConfig::default()
    };

    // Clone what we need for the blocking thread
    let plan = Arc::new(plan);
    let gguf = Arc::new(gguf);

    if interactive {
        run_interactive(path, &config, n_gpu_layers, &plan, &gguf, telemetry).await
    } else if let Some(prompt_text) = prompt {
        run_single_prompt(path, prompt_text, &config, n_gpu_layers, &plan, &gguf, telemetry).await
    } else {
        run_interactive(path, &config, n_gpu_layers, &plan, &gguf, telemetry).await
    }
}

async fn run_single_prompt(
    model_path: &Path,
    prompt: &str,
    config: &InferenceConfig,
    n_gpu_layers: i32,
    plan: &Arc<hypura::scheduler::types::PlacementPlan>,
    gguf: &Arc<GgufFile>,
    telemetry: Arc<TelemetryEmitter>,
) -> anyhow::Result<()> {
    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel();

    let path = model_path.to_path_buf();
    let prompt_owned = prompt.to_string();
    let config_clone = config.clone();
    let plan_clone = plan.clone();
    let gguf_clone = gguf.clone();

    println!("Loading model...");
    let handle = tokio::task::spawn_blocking(move || {
        generate_with_nvme_scheduling(
            &path,
            &prompt_owned,
            &config_clone,
            n_gpu_layers,
            &plan_clone,
            &gguf_clone,
            token_tx,
            telemetry,
        )
    });

    // Stream tokens to stdout
    println!();
    while let Some(token) = token_rx.recv().await {
        print!("{}", token.text);
        io::stdout().flush().ok();
    }

    let result = handle.await??;

    println!();
    println!();
    print_generation_stats(&result);

    Ok(())
}

async fn run_interactive(
    model_path: &Path,
    config: &InferenceConfig,
    n_gpu_layers: i32,
    plan: &Arc<hypura::scheduler::types::PlacementPlan>,
    gguf: &Arc<GgufFile>,
    telemetry: Arc<TelemetryEmitter>,
) -> anyhow::Result<()> {
    println!("Hypura Interactive Mode");
    println!("Type your message, then press Enter. Type /quit or Ctrl-D to exit.");
    println!();

    let stdin = io::stdin();
    let mut history: Vec<(String, String)> = Vec::new();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF (Ctrl-D)
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" || input == "/exit" {
            break;
        }

        history.push(("user".into(), input.to_string()));
        let full_prompt = format_chat_prompt(&history);

        let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel();
        let path = model_path.to_path_buf();
        let prompt = full_prompt;
        let cfg = config.clone();
        let telem = telemetry.clone();
        let plan_c = plan.clone();
        let gguf_c = gguf.clone();

        let handle = tokio::task::spawn_blocking(move || {
            generate_with_nvme_scheduling(
                &path, &prompt, &cfg, n_gpu_layers, &plan_c, &gguf_c, token_tx, telem,
            )
        });

        let mut response = String::new();
        while let Some(token) = token_rx.recv().await {
            print!("{}", token.text);
            io::stdout().flush().ok();
            response.push_str(&token.text);
        }
        println!();

        let result = handle.await??;
        history.push(("assistant".into(), response));

        println!(
            "  [{:.1} tok/s, {} tokens]",
            result.tok_per_sec_avg, result.tokens_generated
        );
        println!();
    }

    Ok(())
}

/// Simple ChatML-style prompt formatting.
fn format_chat_prompt(history: &[(String, String)]) -> String {
    let mut prompt = String::new();
    for (role, content) in history {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn print_placement_header(
    summary: &PlacementSummary,
    plan: &hypura::scheduler::types::PlacementPlan,
    n_gpu_layers: i32,
) {
    println!();
    println!("Hypura: Loading model");
    println!("{}", "─".repeat(48));
    if summary.total_gpu_bytes > 0 {
        println!(
            "  GPU (Metal):  {} ({} layers, n_gpu_layers={})",
            format_bytes(summary.total_gpu_bytes),
            summary.layers_on_gpu,
            n_gpu_layers
        );
    }
    if summary.total_ram_bytes > 0 {
        println!(
            "  RAM:          {} ({} layers)",
            format_bytes(summary.total_ram_bytes),
            summary.layers_in_ram
        );
    }
    if summary.total_nvme_bytes > 0 {
        println!(
            "  NVMe:         {} ({} layers)",
            format_bytes(summary.total_nvme_bytes),
            summary.layers_on_nvme
        );
    }
    println!(
        "  Experience:   {} — {}",
        plan.experience_tier.label(),
        plan.experience_tier.description()
    );
}

fn print_generation_stats(result: &GenerationResult) {
    println!("Generation complete:");
    println!("  Prompt tokens:      {}", result.prompt_tokens);
    println!("  Generated tokens:   {}", result.tokens_generated);
    println!(
        "  Prompt eval:        {:.1} ms ({:.1} tok/s)",
        result.prompt_eval_ms,
        if result.prompt_eval_ms > 0.0 {
            result.prompt_tokens as f64 / (result.prompt_eval_ms / 1000.0)
        } else {
            0.0
        }
    );
    println!(
        "  Generation:         {:.1} tok/s (avg)",
        result.tok_per_sec_avg
    );
}
