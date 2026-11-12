use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use anyhow::Result;
use tokio::sync::mpsc;

use hypura::model::pull::{
    detect_default_models_dir, download_model_with_progress, resolve_pull_spec,
};
use hypura::server::ollama_types::PullResponseChunk;

pub async fn run(
    model: String,
    models_dir: Option<PathBuf>,
    force: bool,
) -> Result<()> {
    let spec = resolve_pull_spec(&model)?;
    let target_dir = models_dir.unwrap_or_else(detect_default_models_dir);

    println!();
    println!("Hypura Model Pull");
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("  Model Alias:  {}", spec.alias);
    println!("  Filename:     {}", spec.filename);
    println!("  Target Dir:   {}", target_dir.display());
    println!("  Source URL:   {}", spec.download_url);
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!();

    let (tx, mut rx) = mpsc::channel::<PullResponseChunk>(32);

    let spec_clone = spec.clone();
    let target_clone = target_dir.clone();
    let download_task = tokio::spawn(async move {
        download_model_with_progress(&spec_clone, &target_clone, force, Some(tx)).await
    });

    let start_time = Instant::now();
    let mut last_completed = 0u64;
    let mut last_time = Instant::now();

    while let Some(prog) = rx.recv().await {
        if let (Some(total), Some(completed)) = (prog.total, prog.completed) {
            let pct = if total > 0 {
                (completed as f64 / total as f64) * 100.0
            } else {
                0.0
            };

            let now = Instant::now();
            let dt = now.duration_since(last_time).as_secs_f64();
            let speed_mb = if dt > 0.5 {
                let bytes_diff = completed.saturating_sub(last_completed);
                last_completed = completed;
                last_time = now;
                (bytes_diff as f64 / dt) / (1024.0 * 1024.0)
            } else {
                0.0
            };

            let completed_gb = completed as f64 / (1024.0 * 1024.0 * 1024.0);
            let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);

            // 30-character progress bar
            let bar_len = 30;
            let filled = ((pct / 100.0) * bar_len as f64).round() as usize;
            let bar: String = (0..bar_len)
                .map(|i| if i < filled { '=' } else { ' ' })
                .collect();

            print!(
                "\r  [{}] {:>5.1}%  ({:.2} GB / {:.2} GB)  {:>6.1} MB/s",
                bar, pct, completed_gb, total_gb, speed_mb
            );
            let _ = std::io::stdout().flush();
        } else {
            println!("  • {}", prog.status);
        }
    }

    println!();
    let final_path = download_task.await??;
    let elapsed = start_time.elapsed().as_secs_f64();

    println!();
    println!("✓ Successfully pulled '{}' in {:.1}s", spec.alias, elapsed);
    println!("  Saved to: {}", final_path.display());
    println!();
    println!("To start serving:");
    println!("  hypura serve --model {}", spec.alias);
    println!();

    Ok(())
}
