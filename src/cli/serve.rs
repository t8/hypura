use std::path::PathBuf;
use std::sync::Arc;

use hypura::server::manager::ModelManager;
use hypura::server::registry::ModelRegistry;
use hypura::server::routes::{self, AppState};
use hypura::telemetry::metrics::TelemetryEmitter;

pub fn run(
    model: Option<&str>,
    host: &str,
    port: u16,
    context: u32,
    models_dir: Option<PathBuf>,
    ollama_dir: Option<PathBuf>,
    idle_timeout_secs: u64,
    kv_quant: Option<String>,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(model, host, port, context, models_dir, ollama_dir, idle_timeout_secs, kv_quant))
}

async fn run_async(
    model: Option<&str>,
    host: &str,
    port: u16,
    context: u32,
    models_dir: Option<PathBuf>,
    ollama_dir: Option<PathBuf>,
    idle_timeout_secs: u64,
    kv_quant: Option<String>,
) -> anyhow::Result<()> {
    let parsed_kv = kv_quant.as_deref().and_then(|s| match s.to_ascii_lowercase().as_str() {
        "q8_0" | "q8" => Some(hypura::scheduler::types::KvQuantization::Q8_0),
        "q4_0" | "q4" => Some(hypura::scheduler::types::KvQuantization::Q4_0),
        _ => None,
    });
    let registry = ModelRegistry::new(models_dir, ollama_dir);
    let manager = ModelManager::new(registry, context, model, parsed_kv)?;
    let telemetry = Arc::new(TelemetryEmitter::new(256));

    let state = Arc::new(AppState {
        manager: Arc::new(std::sync::Mutex::new(manager)),
        telemetry,
    });

    if idle_timeout_secs > 0 {
        let state_clone = state.clone();
        let timeout = std::time::Duration::from_secs(idle_timeout_secs);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
            loop {
                interval.tick().await;
                let mut mgr = state_clone.manager.lock().unwrap();
                mgr.maybe_unload_idle(timeout);
            }
        });
    }

    let app = routes::router(state.clone());
    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    println!();
    println!("Hypura Server running in dynamic multi-model mode");
    println!("  Endpoint: http://{bind_addr}");
    println!("  Ollama-compatible API: /api/tags, /api/show, /api/chat, /api/generate");
    println!("  OpenAI-compatible API: /v1/models, /v1/chat/completions, /v1/completions");
    if idle_timeout_secs > 0 {
        println!("  Idle memory timeout: {}s (auto-unloads from VRAM when inactive)", idle_timeout_secs);
    }
    if let Some(kv) = parsed_kv {
        println!("  KV cache quantization: {:?}", kv);
    }

    let count = {
        let mgr = state.manager.lock().unwrap();
        mgr.registry.models.len()
    };
    println!("  Discovered {count} models (Local + Ollama).");
    if let Some(m) = model {
        println!("  Pre-warmed model: {m}");
    }
    println!();

    axum::serve(listener, app).await?;
    Ok(())
}
