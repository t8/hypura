use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use hypura::compute::inference;
use hypura::model::gguf::GgufFile;
use hypura::model::metadata::ModelMetadata;
use hypura::profiler;
use hypura::scheduler::placement::compute_placement_with_context;
use hypura::server::ollama_types::GgufInfo;
use hypura::server::routes::{self, AppState};
use hypura::telemetry::metrics::TelemetryEmitter;

pub fn run(model_path: &str, host: &str, port: u16, context: u32) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(model_path, host, port, context))
}

async fn run_async(model_path: &str, host: &str, port: u16, context: u32) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    anyhow::ensure!(path.exists(), "Model file not found: {model_path}");

    // Load hardware profile
    let hardware = match profiler::load_cached_profile()? {
        Some(p) if !profiler::is_profile_stale(&p) => p,
        _ => {
            println!("No hardware profile found. Running profiler...");
            let p = profiler::run_full_profile()?;
            profiler::save_profile(&p)?;
            p
        }
    };

    // Parse GGUF and compute placement
    let gguf = GgufFile::open(path)?;
    let metadata = ModelMetadata::from_gguf(&gguf)?;
    let file_size = std::fs::metadata(path)?.len();

    let gguf_info = GgufInfo {
        file_size,
        architecture: metadata.architecture.clone(),
        parameter_count: metadata.parameter_count,
        quantization: metadata.quantization.clone().unwrap_or_else(|| "unknown".into()),
        context_length: metadata.context_length,
    };

    let plan = compute_placement_with_context(&gguf, &hardware, context)?;
    let gpu_budget = inference::compute_gpu_budget(&hardware, &metadata, context);
    let n_gpu_layers = inference::gpu_layers_from_placement(&plan, &gguf, gpu_budget);

    let config = inference::InferenceConfig {
        n_ctx: context,
        ..inference::InferenceConfig::default()
    };

    // Load model on a blocking thread
    println!("Loading model...");
    let load_start = Instant::now();

    let path_owned = path.to_path_buf();
    let plan_arc = Arc::new(plan);
    let gguf_arc = Arc::new(gguf);
    let plan_for_load = plan_arc.clone();
    let gguf_for_load = gguf_arc.clone();

    let loaded = tokio::task::spawn_blocking(move || {
        inference::load_model(
            &path_owned,
            &config,
            n_gpu_layers,
            &plan_for_load,
            &gguf_for_load,
        )
    })
    .await??;

    let load_duration_ns = load_start.elapsed().as_nanos() as u64;
    let model_name = loaded.model_name.clone();

    let telemetry = Arc::new(TelemetryEmitter::new(256));

    let state = Arc::new(AppState {
        loaded_model: Arc::new(std::sync::Mutex::new(loaded)),
        model_name: model_name.clone(),
        gguf_info,
        load_duration_ns,
        telemetry,
    });

    let app = routes::router(state);
    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    println!();
    println!("Hypura serving {model_name}");
    println!("  Endpoint: http://{bind_addr}");
    println!("  Ollama-compatible API: /api/generate, /api/chat, /api/tags");
    println!();

    axum::serve(listener, app).await?;
    Ok(())
}
