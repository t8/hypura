use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::{mpsc, oneshot};

use crate::compute::inference::{GenerateFromLoadedParams, GenerationResult};
use crate::server::chat::format_chat_prompt;
use crate::server::manager::ModelManager;
use crate::server::ollama_types::*;
use crate::server::openai_types::*;
use crate::server::streaming;
use crate::telemetry::metrics::TelemetryEmitter;

use tower_http::cors::{Any, CorsLayer};

pub struct AppState {
    pub manager: Arc<std::sync::Mutex<ModelManager>>,
    pub telemetry: Arc<TelemetryEmitter>,
}

pub fn router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::PUT,
            axum::http::Method::DELETE,
            axum::http::Method::OPTIONS,
            axum::http::Method::HEAD,
        ])
        .allow_headers(Any)
        .expose_headers(Any);

    Router::new()
        .route("/", get(health_handler))
        // Ollama API
        .route("/api/version", get(version_handler))
        .route("/api/tags", get(tags_handler))
        .route("/api/ps", get(ps_handler))
        .route("/api/show", post(show_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/pull", post(pull_handler))
        .route("/api/delete", axum::routing::delete(delete_handler))
        // OpenAI API (with /v1/ prefix and without /v1/ for clients with custom base_url handling)
        .route("/v1/models", get(v1_models_handler))
        .route("/models", get(v1_models_handler))
        .route("/v1/chat/completions", post(v1_chat_completions_handler))
        .route("/chat/completions", post(v1_chat_completions_handler))
        .route("/v1/completions", post(v1_completions_handler))
        .route("/completions", post(v1_completions_handler))
        .layer(cors)
        .with_state(state)
}

async fn ps_handler(State(state): State<Arc<AppState>>) -> Json<PsResponse> {
    let models = {
        let manager = state.manager.lock().unwrap();
        if let Some(ref active) = manager.active_model {
            vec![ProcessModel {
                name: active.name.clone(),
                model: active.name.clone(),
                size: active.gguf_info.file_size,
                digest: "".into(),
                details: ModelDetails {
                    format: "gguf".into(),
                    family: active.gguf_info.architecture.clone(),
                    parameter_size: format_parameter_size(active.gguf_info.parameter_count),
                    quantization_level: active.gguf_info.quantization.clone(),
                },
                expires_at: "".into(),
                size_vram: active.gguf_info.file_size,
                context_size: active.context_size,
            }]
        } else {
            vec![]
        }
    };
    Json(PsResponse { models })
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn version_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"version": env!("CARGO_PKG_VERSION")}))
}

async fn tags_handler(State(state): State<Arc<AppState>>) -> Json<TagsResponse> {
    let models = {
        let mut manager = state.manager.lock().unwrap();
        manager.registry.refresh();
        manager.registry.list_models()
    };
    Json(TagsResponse { models })
}

async fn show_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShowRequest>,
) -> Response {
    let info_opt = {
        let manager = state.manager.lock().unwrap();
        manager
            .registry
            .resolve_model(&req.model)
            .map(|m| m.to_gguf_info())
    };

    match info_opt {
        Some(info) => Json(ShowResponse {
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
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("model '{}' not found", req.model)})),
        )
            .into_response(),
    }
}

async fn delete_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeleteRequest>,
) -> Response {
    let custom_dir_override = req.models_dir.map(std::path::PathBuf::from);
    let mut manager = state.manager.lock().unwrap();

    // If active model is being deleted, unload it first
    if let Some(ref active) = manager.active_model {
        if active.name.eq_ignore_ascii_case(&req.name)
            || active.name.to_lowercase().contains(&req.name.to_lowercase())
        {
            manager.active_model = None;
        }
    }

    let target_dir = custom_dir_override
        .or_else(|| manager.registry.custom_dir.clone());

    match crate::model::pull::delete_model_files(&req.name, target_dir.as_deref(), &mut manager.registry) {
        Ok((name, bytes_freed)) => {
            tracing::info!("Deleted model '{}' (freed {} bytes)", name, bytes_freed);
            StatusCode::OK.into_response()
        }
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn pull_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PullRequest>,
) -> Response {
    let spec = match crate::model::pull::resolve_pull_spec(&req.name) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    let target_models_dir = req
        .models_dir
        .map(std::path::PathBuf::from)
        .or_else(|| {
            let manager = state.manager.lock().unwrap();
            manager.registry.custom_dir.clone()
        })
        .unwrap_or_else(crate::model::pull::detect_default_models_dir);

    if !req.stream {
        // Non-streaming: await completion
        match crate::model::pull::download_model_with_progress(
            &spec,
            &target_models_dir,
            false,
            None,
        )
        .await
        {
            Ok(_) => {
                let mut manager = state.manager.lock().unwrap();
                manager.registry.refresh();
                Json(serde_json::json!({"status": "success"})).into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response(),
        }
    } else {
        // Streaming: NDJSON progress chunks
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let state_clone = state.clone();
        let target_dir_clone = target_models_dir.clone();

        tokio::spawn(async move {
            let res = crate::model::pull::download_model_with_progress(
                &spec,
                &target_dir_clone,
                false,
                Some(tx.clone()),
            )
            .await;

            if res.is_ok() {
                let mut manager = state_clone.manager.lock().unwrap();
                manager.registry.refresh();
            } else if let Err(e) = res {
                let _ = tx
                    .send(crate::server::ollama_types::PullResponseChunk {
                        status: format!("error: {e}"),
                        digest: None,
                        total: None,
                        completed: None,
                    })
                    .await;
            }
        });

        let body = streaming::ndjson_pull_stream(rx);
        (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/x-ndjson"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::CONNECTION, "keep-alive"),
                (header::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response()
    }
}

async fn generate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    let request_start = Instant::now();

    let (loaded, model_name, _info) = {
        let mut manager = state.manager.lock().unwrap();
        match manager.get_or_load(&req.model, req.options.num_ctx) {
            Ok(res) => res,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response();
            }
        }
    };

    let sampling = build_sampling(&req.options);
    let prompt = req.prompt;

    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();
    let telemetry = state.telemetry.clone();

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        if token_tx.is_closed() {
            tracing::info!("Client disconnected while waiting for model lock; skipping generate");
            return;
        }
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
        let body = streaming::ndjson_generate_stream(
            model_name,
            token_rx,
            result_rx,
            request_start,
            0,
        );
        (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/x-ndjson"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::CONNECTION, "keep-alive"),
                (axum::http::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response()
    } else {
        let result =
            collect_generate(model_name, token_rx, result_rx, request_start, 0).await;
        Json(result).into_response()
    }
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let request_start = Instant::now();
    tracing::info!(
        "POST /api/chat: model={:?}, stream={}, messages={}, tools={}",
        req.model,
        req.stream,
        req.messages.len(),
        req.tools
            .as_ref()
            .and_then(|t| t.as_array())
            .map_or(0, |a| a.len())
    );

    let (loaded, model_name, info) = {
        let mut manager = state.manager.lock().unwrap();
        match manager.get_or_load(&req.model, req.options.num_ctx) {
            Ok(res) => res,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response();
            }
        }
    };

    let sampling = build_sampling(&req.options);
    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();
    let telemetry = state.telemetry.clone();
    let req_messages = req.messages;
    let req_tools = req.tools;
    let is_streaming = req.stream;
    let arch = info.architecture.clone();
    let model_req_name = req.model.clone();
    let active_model_name = model_name.clone();

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        if token_tx.is_closed() {
            tracing::info!("Client disconnected while waiting for model lock; skipping chat");
            return;
        }
        let tmpl = model.model.chat_template().unwrap_or_default();
        let full_hint = format!("{} {} {} {}", arch, active_model_name, model_req_name, tmpl);
        let prompt = format_chat_prompt(&req_messages, req_tools.as_ref(), Some(&full_hint));
        if !is_streaming && prompt.len() > 12000 {
            let estimated_tokens = prompt.len() / 4;
            tracing::warn!(
                "Non-streaming request received with large prompt (~{} tokens, {} chars); prefill may take 60-120s. Ensure client socket read timeout is >= 300s.",
                estimated_tokens,
                prompt.len()
            );
        }
        if prompt.ends_with("<think>\n") {
            let _ = token_tx.send(crate::compute::inference::GeneratedToken {
                text: "<think>\n".into(),
                token_id: 0,
                tok_per_sec: 0.0,
                is_eog: false,
            });
        }
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
            streaming::ndjson_chat_stream(model_name, token_rx, result_rx, request_start, 0);
        (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/x-ndjson"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::CONNECTION, "keep-alive"),
                (axum::http::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response()
    } else {
        let result =
            collect_chat(model_name, token_rx, result_rx, request_start, 0).await;
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
    if let Some(rp) = opts.repeat_penalty {
        s.repeat_penalty = rp;
    }
    if let Some(n) = opts.num_predict {
        s.max_tokens = n;
    }
    if let Some(seed) = opts.seed {
        s.seed = seed;
    }
    if let Some(ref stops) = opts.stop {
        s.stop_sequences = stops.clone();
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
    tracing::info!("CHAT RAW RESPONSE: {:?}", full_response);
    let (content, tool_calls) = crate::server::chat::parse_tool_calls(&full_response);
    tracing::info!("PARSED TOOL CALLS: {:?}, CLEANED CONTENT: {:?}", tool_calls, content);
    let total_ns = request_start.elapsed().as_nanos() as u64;
    let result = result_rx.await.ok();

    ChatResponseChunk {
        model: model_name,
        created_at: now_rfc3339(),
        message: ChatMessage {
            role: "assistant".into(),
            content,
            tool_calls,
            ..Default::default()
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

// ── OpenAI V1 Handlers ──

async fn v1_models_handler(State(state): State<Arc<AppState>>) -> Json<OpenAIModelList> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let models = {
        let mut manager = state.manager.lock().unwrap();
        manager.registry.refresh();
        manager
            .registry
            .list_models()
            .into_iter()
            .map(|m| OpenAIModelEntry {
                id: m.name,
                object: "model",
                created: now,
                owned_by: "hypura",
            })
            .collect()
    };

    Json(OpenAIModelList {
        object: "list",
        data: models,
    })
}

async fn v1_chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAIChatCompletionRequest>,
) -> Response {
    let _request_start = Instant::now();
    tracing::info!(
        "POST /v1/chat/completions: model={:?}, stream={}, messages={}, tools={}, max_tokens={:?}",
        req.model,
        req.stream,
        req.messages.len(),
        req.tools
            .as_ref()
            .and_then(|t| t.as_array())
            .map_or(0, |a| a.len()),
        req.max_tokens
    );
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let chat_id = format!("chatcmpl-{}", uuid_simple());

    let (loaded, model_name, info) = {
        let mut manager = state.manager.lock().unwrap();
        match manager.get_or_load(&req.model, req.requested_context()) {
            Ok(res) => res,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": {
                            "message": e.to_string(),
                            "type": "invalid_request_error",
                            "code": "model_not_found"
                        }
                    })),
                )
                    .into_response();
            }
        }
    };

    let mut sampling = crate::compute::ffi::SamplingParams::default();
    if let Some(t) = req.temperature {
        sampling.temperature = t;
    }
    if let Some(p) = req.top_p {
        sampling.top_p = p;
    }
    if let Some(m) = req.max_tokens {
        sampling.max_tokens = m;
    }
    if let Some(ref stops) = req.stop {
        sampling.stop_sequences = stops.clone();
    }
    if let Some(seed) = req.seed {
        sampling.seed = seed;
    }
    if let Some(rp) = req.presence_penalty {
        sampling.repeat_penalty = rp;
    }

    // Convert OpenAIChatMessage to internal ChatMessage while resolving tool_call_id -> function name
    let mut id_to_tool_name = std::collections::HashMap::<String, String>::new();
    let mut internal_messages = Vec::with_capacity(req.messages.len());
    for m in req.messages {
        let converted_calls = m.tool_calls.map(|tcs| {
            tcs.into_iter()
                .map(|tc| {
                    id_to_tool_name.insert(tc.id.clone(), tc.function.name.clone());
                    ToolCall {
                        id: Some(tc.id),
                        call_type: Some(tc.call_type),
                        function: FunctionCall {
                            name: tc.function.name,
                            arguments: serde_json::from_str(&tc.function.arguments)
                                .unwrap_or_else(|_| serde_json::Value::String(tc.function.arguments)),
                        },
                    }
                })
                .collect()
        });
        let resolved_tool_name = m.name.or_else(|| {
            m.tool_call_id
                .as_ref()
                .and_then(|cid| id_to_tool_name.get(cid).cloned())
        });
        internal_messages.push(ChatMessage {
            role: m.role,
            content: m.content.unwrap_or_default(),
            tool_calls: converted_calls,
            tool_call_id: m.tool_call_id,
            tool_name: resolved_tool_name,
        });
    }

    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();
    let telemetry = state.telemetry.clone();
    let tools_disabled = req
        .tool_choice
        .as_ref()
        .and_then(|v| v.as_str())
        .map_or(false, |s| s.eq_ignore_ascii_case("none"));
    let req_tools = if tools_disabled { None } else { req.tools };
    let is_streaming = req.stream;
    let arch = info.architecture.clone();
    let model_req_name = req.model.clone();
    let active_model_name = model_name.clone();

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        if token_tx.is_closed() {
            tracing::info!("Client disconnected while waiting for model lock; skipping OpenAI chat");
            return;
        }
        let tmpl = model.model.chat_template().unwrap_or_default();
        let full_hint = format!("{} {} {} {}", arch, active_model_name, model_req_name, tmpl);
        let prompt = format_chat_prompt(&internal_messages, req_tools.as_ref(), Some(&full_hint));
        if !is_streaming && prompt.len() > 12000 {
            let estimated_tokens = prompt.len() / 4;
            tracing::warn!(
                "Non-streaming request received with large prompt (~{} tokens, {} chars); prefill may take 60-120s. Ensure client socket read timeout is >= 300s.",
                estimated_tokens,
                prompt.len()
            );
        }
        if prompt.ends_with("<think>\n") {
            let _ = token_tx.send(crate::compute::inference::GeneratedToken {
                text: "<think>\n".into(),
                token_id: 0,
                tok_per_sec: 0.0,
                is_eog: false,
            });
        }
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
                tracing::error!("OpenAI Chat generation error: {e}");
            }
        }
    });

    if req.stream {
        let body = streaming::sse_openai_chat_stream(
            model_name,
            token_rx,
            result_rx,
            chat_id,
            created,
        );
        (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::CONNECTION, "keep-alive"),
                (axum::http::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response()
    } else {
        let mut full_response = String::new();
        let mut rx = token_rx;
        while let Some(token) = rx.recv().await {
            full_response.push_str(&token.text);
        }
        let result = result_rx.await.ok();
        let (content, tool_calls) = crate::server::chat::parse_tool_calls(&full_response);

        let openai_tool_calls: Option<Vec<OpenAIToolCall>> = tool_calls.map(|tcs| {
            tcs.into_iter()
                .map(|tc| OpenAIToolCall {
                    index: None,
                    id: tc
                        .id
                        .unwrap_or_else(crate::server::chat::generate_tool_call_id),
                    call_type: "function".into(),
                    function: OpenAIFunctionCall {
                        name: tc.function.name,
                        arguments: serde_json::to_string(&tc.function.arguments)
                            .unwrap_or_else(|_| "{}".into()),
                    },
                })
                .collect()
        });

        let has_tool_calls = openai_tool_calls.is_some();
        let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };
        let message_content = if has_tool_calls && content.is_empty() {
            None
        } else {
            Some(content)
        };

        let prompt_tokens = result.as_ref().map(|r| r.prompt_tokens).unwrap_or(0);
        let completion_tokens = result.as_ref().map(|r| r.tokens_generated).unwrap_or(0);

        let resp = OpenAIChatCompletionResponse {
            id: chat_id,
            object: "chat.completion",
            created,
            model: model_name,
            choices: vec![OpenAIChatChoice {
                index: 0,
                message: OpenAIChatMessage {
                    role: "assistant".into(),
                    content: message_content,
                    name: None,
                    tool_calls: openai_tool_calls,
                    tool_call_id: None,
                },
                finish_reason: Some(finish_reason.into()),
            }],
            usage: OpenAIUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        Json(resp).into_response()
    }
}

async fn v1_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAICompletionRequest>,
) -> Response {
    let _request_start = Instant::now();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let completion_id = format!("cmpl-{}", uuid_simple());

    let (loaded, model_name, _info) = {
        let mut manager = state.manager.lock().unwrap();
        match manager.get_or_load(&req.model, req.requested_context()) {
            Ok(res) => res,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": {
                            "message": e.to_string(),
                            "type": "invalid_request_error",
                            "code": "model_not_found"
                        }
                    })),
                )
                    .into_response();
            }
        }
    };

    let mut sampling = crate::compute::ffi::SamplingParams::default();
    if let Some(t) = req.temperature {
        sampling.temperature = t;
    }
    if let Some(p) = req.top_p {
        sampling.top_p = p;
    }
    if let Some(m) = req.max_tokens {
        sampling.max_tokens = m;
    }
    if let Some(ref stops) = req.stop {
        sampling.stop_sequences = stops.clone();
    }

    let prompt = req.prompt;
    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();
    let telemetry = state.telemetry.clone();

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        if token_tx.is_closed() {
            tracing::info!("Client disconnected while waiting for model lock; skipping OpenAI completion");
            return;
        }
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
                tracing::error!("OpenAI Completion error: {e}");
            }
        }
    });

    if req.stream {
        let body = streaming::sse_openai_completion_stream(
            model_name,
            token_rx,
            result_rx,
            completion_id,
            created,
        );
        (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::CONNECTION, "keep-alive"),
                (axum::http::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response()
    } else {
        let mut full_response = String::new();
        let mut rx = token_rx;
        while let Some(token) = rx.recv().await {
            full_response.push_str(&token.text);
        }
        let result = result_rx.await.ok();
        let prompt_tokens = result.as_ref().map(|r| r.prompt_tokens).unwrap_or(0);
        let completion_tokens = result.as_ref().map(|r| r.tokens_generated).unwrap_or(0);

        let resp = OpenAICompletionResponse {
            id: completion_id,
            object: "text_completion",
            created,
            model: model_name,
            choices: vec![OpenAICompletionChoice {
                index: 0,
                text: full_response,
                finish_reason: Some("stop".into()),
            }],
            usage: OpenAIUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        Json(resp).into_response()
    }
}

fn uuid_simple() -> String {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", nanos)
}
