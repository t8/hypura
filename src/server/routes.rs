use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::{mpsc, oneshot};

use crate::compute::inference::{GenerateFromLoadedParams, GenerationResult, LoadedModel};
use crate::server::chat::format_chat_prompt;
use crate::server::ollama_types::*;
use crate::server::streaming;
use crate::telemetry::metrics::TelemetryEmitter;

pub struct AppState {
    pub loaded_model: Arc<std::sync::Mutex<LoadedModel>>,
    pub model_name: String,
    pub gguf_info: GgufInfo,
    pub load_duration_ns: u64,
    pub telemetry: Arc<TelemetryEmitter>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(health_handler))
        .route("/api/version", get(version_handler))
        .route("/api/tags", get(tags_handler))
        .route("/api/show", post(show_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/chat", post(chat_handler))
        .with_state(state)
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn version_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"version": env!("CARGO_PKG_VERSION")}))
}

async fn tags_handler(State(state): State<Arc<AppState>>) -> Json<TagsResponse> {
    let info = &state.gguf_info;
    Json(TagsResponse {
        models: vec![ModelTag {
            name: state.model_name.clone(),
            model: state.model_name.clone(),
            size: info.file_size,
            details: ModelDetails {
                format: "gguf".into(),
                family: info.architecture.clone(),
                parameter_size: format_parameter_size(info.parameter_count),
                quantization_level: info.quantization.clone(),
            },
        }],
    })
}

async fn show_handler(
    State(state): State<Arc<AppState>>,
    Json(_req): Json<ShowRequest>,
) -> Json<ShowResponse> {
    let info = &state.gguf_info;
    Json(ShowResponse {
        details: ModelDetails {
            format: "gguf".into(),
            family: info.architecture.clone(),
            parameter_size: format_parameter_size(info.parameter_count),
            quantization_level: info.quantization.clone(),
        },
        model_info: serde_json::json!({
            "general.architecture": info.architecture,
            "general.context_length": info.context_length,
            "general.parameter_count": info.parameter_count,
        }),
    })
}

async fn generate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    let sampling = build_sampling(&req.options);
    let prompt = req.prompt;
    let model_name = state.model_name.clone();

    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();

    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        match result {
            Ok(gen_result) => {
                let _ = result_tx.send(gen_result);
            }
            Err(e) => {
                tracing::error!("Generation error: {e}");
            }
        }
    });

    if req.stream {
        let body =
            streaming::ndjson_generate_stream(model_name, token_rx, result_rx, request_start, load_duration_ns);
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            body,
        )
            .into_response()
    } else {
        // Non-streaming: collect all tokens, return single JSON
        let result = collect_generate(model_name, token_rx, result_rx, request_start, load_duration_ns).await;
        Json(result).into_response()
    }
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    let sampling = build_sampling(&req.options);
    let prompt = format_chat_prompt(&req.messages);
    let model_name = state.model_name.clone();

    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();

    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        match result {
            Ok(gen_result) => {
                let _ = result_tx.send(gen_result);
            }
            Err(e) => {
                tracing::error!("Chat generation error: {e}");
            }
        }
    });

    if req.stream {
        let body =
            streaming::ndjson_chat_stream(model_name, token_rx, result_rx, request_start, load_duration_ns);
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            body,
        )
            .into_response()
    } else {
        let result = collect_chat(model_name, token_rx, result_rx, request_start, load_duration_ns).await;
        Json(result).into_response()
    }
}

// ── Helpers ──

fn build_sampling(opts: &GenerateOptions) -> crate::compute::ffi::SamplingParams {
    let mut s = crate::compute::ffi::SamplingParams::default();
    if let Some(t) = opts.temperature {
        s.temperature = t;
    }
    if let Some(k) = opts.top_k {
        s.top_k = k;
    }
    if let Some(p) = opts.top_p {
        s.top_p = p;
    }
    if let Some(n) = opts.num_predict {
        s.max_tokens = n;
    }
    if let Some(seed) = opts.seed {
        s.seed = seed;
    }
    s
}

async fn collect_generate(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<crate::compute::inference::GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> GenerateResponseChunk {
    let mut full_response = String::new();
    while let Some(token) = token_rx.recv().await {
        full_response.push_str(&token.text);
    }
    let total_ns = request_start.elapsed().as_nanos() as u64;
    let result = result_rx.await.ok();

    GenerateResponseChunk {
        model: model_name,
        created_at: now_rfc3339(),
        response: full_response,
        done: true,
        done_reason: Some("stop".into()),
        total_duration: Some(total_ns),
        load_duration: Some(load_duration_ns),
        prompt_eval_count: result.as_ref().map(|r| r.prompt_tokens),
        prompt_eval_duration: result
            .as_ref()
            .map(|r| (r.prompt_eval_ms * 1_000_000.0) as u64),
        eval_count: result.as_ref().map(|r| r.tokens_generated),
        eval_duration: result.as_ref().map(|r| {
            if r.tok_per_sec_avg > 0.0 {
                (r.tokens_generated as f64 / r.tok_per_sec_avg * 1e9) as u64
            } else {
                0
            }
        }),
    }
}

async fn collect_chat(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<crate::compute::inference::GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> ChatResponseChunk {
    let mut full_response = String::new();
    while let Some(token) = token_rx.recv().await {
        full_response.push_str(&token.text);
    }
    let total_ns = request_start.elapsed().as_nanos() as u64;
    let result = result_rx.await.ok();

    ChatResponseChunk {
        model: model_name,
        created_at: now_rfc3339(),
        message: ChatMessage {
            role: "assistant".into(),
            content: full_response,
        },
        done: true,
        done_reason: Some("stop".into()),
        total_duration: Some(total_ns),
        load_duration: Some(load_duration_ns),
        prompt_eval_count: result.as_ref().map(|r| r.prompt_tokens),
        prompt_eval_duration: result
            .as_ref()
            .map(|r| (r.prompt_eval_ms * 1_000_000.0) as u64),
        eval_count: result.as_ref().map(|r| r.tokens_generated),
        eval_duration: result.as_ref().map(|r| {
            if r.tok_per_sec_avg > 0.0 {
                (r.tokens_generated as f64 / r.tok_per_sec_avg * 1e9) as u64
            } else {
                0
            }
        }),
    }
}

fn format_parameter_size(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1e6)
    } else {
        format!("{params}")
    }
}
