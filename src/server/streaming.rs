use std::time::{Duration, Instant};

use axum::body::Body;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval_at, Instant as TokioInstant, MissedTickBehavior};
use tokio_stream::wrappers::ReceiverStream;

use crate::compute::inference::{GeneratedToken, GenerationResult};
use crate::server::ollama_types::*;

/// Interval for keep-alive heartbeats during long prompt prefills, `<thought>` blocks,
/// and buffered tool-call generations so Tailscale Funnel and coding-agent HTTP clients
/// never hit their ~120s idle read timeout.
const STREAM_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);

fn new_keepalive_timer() -> tokio::time::Interval {
    let mut timer = interval_at(
        TokioInstant::now() + STREAM_KEEPALIVE_INTERVAL,
        STREAM_KEEPALIVE_INTERVAL,
    );
    timer.set_missed_tick_behavior(MissedTickBehavior::Delay);
    timer
}

/// Convert a token channel into an NDJSON streaming body for `/api/generate`.
pub fn ndjson_generate_stream(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> Body {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut keepalive = new_keepalive_timer();

        // Stream token chunks with periodic keep-alives & disconnect detection
        loop {
            let token = tokio::select! {
                biased;
                _ = tx.closed() => {
                    tracing::info!("Client disconnected from /api/generate stream; aborting generation");
                    return;
                }
                _ = keepalive.tick() => {
                    let hb = GenerateResponseChunk {
                        model: model_name.clone(),
                        created_at: now_rfc3339(),
                        response: String::new(),
                        done: false,
                        done_reason: None,
                        total_duration: None,
                        load_duration: None,
                        prompt_eval_count: None,
                        prompt_eval_duration: None,
                        eval_count: None,
                        eval_duration: None,
                    };
                    let mut line = serde_json::to_string(&hb).unwrap_or_default();
                    line.push('\n');
                    if tx.send(Ok(line)).await.is_err() {
                        return;
                    }
                    continue;
                }
                maybe_tok = token_rx.recv() => match maybe_tok {
                    Some(t) => t,
                    None => break,
                },
            };

            let chunk = GenerateResponseChunk {
                model: model_name.clone(),
                created_at: now_rfc3339(),
                response: token.text,
                done: false,
                done_reason: None,
                total_duration: None,
                load_duration: None,
                prompt_eval_count: None,
                prompt_eval_duration: None,
                eval_count: None,
                eval_duration: None,
            };
            let mut line = serde_json::to_string(&chunk).unwrap_or_default();
            line.push('\n');
            if tx.send(Ok(line)).await.is_err() {
                return;
            }
        }

        // Final chunk with timing
        let total_ns = request_start.elapsed().as_nanos() as u64;
        let result = result_rx.await.ok();
        let final_chunk = GenerateResponseChunk {
            model: model_name,
            created_at: now_rfc3339(),
            response: String::new(),
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
        };
        let mut line = serde_json::to_string(&final_chunk).unwrap_or_default();
        line.push('\n');
        let _ = tx.send(Ok(line)).await;
    });

    Body::from_stream(ReceiverStream::new(rx))
}

/// Convert a token channel into an NDJSON streaming body for `/api/chat`.
pub fn ndjson_chat_stream(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> Body {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut full_response = String::new();
        let mut prefix_buffer = String::new();
        let mut is_tool_call = false;
        let mut prefix_checked = false;
        let mut keepalive = new_keepalive_timer();

        loop {
            let token = tokio::select! {
                biased;
                _ = tx.closed() => {
                    tracing::info!("Client disconnected from /api/chat stream; aborting generation");
                    return;
                }
                _ = keepalive.tick() => {
                    let hb = make_chat_chunk(&model_name, String::new(), false);
                    if tx.send(Ok(hb)).await.is_err() {
                        return;
                    }
                    continue;
                }
                maybe_tok = token_rx.recv() => match maybe_tok {
                    Some(t) => t,
                    None => break,
                },
            };

            full_response.push_str(&token.text);

            if !prefix_checked {
                prefix_buffer.push_str(&token.text);
                let trimmed = prefix_buffer.trim_start();

                // 1. Strip thought channels like <|channel>thought\n...<channel|> or <thought>...</thought> or <think>...</think>
                if trimmed.starts_with("<|channel>thought")
                    || trimmed.starts_with("<thought>")
                    || trimmed.starts_with("<think>")
                {
                    if let Some(end_idx) = trimmed
                        .find("<channel|>")
                        .map(|i| i + "<channel|>".len())
                        .or_else(|| trimmed.find("</thought>").map(|i| i + "</thought>".len()))
                        .or_else(|| trimmed.find("</think>").map(|i| i + "</think>".len()))
                    {
                        prefix_buffer = trimmed[end_idx..].trim_start().to_string();
                    } else {
                        // Still inside thought channel, continue buffering
                        continue;
                    }
                }

                let trimmed = prefix_buffer.trim_start();

                // 2. Detect tool calls or reasoning channels early to prevent streaming raw JSON/reasoning to UI
                if trimmed.starts_with("<tool_call")
                    || trimmed.starts_with("<|tool_call")
                    || trimmed.starts_with("```tool_call")
                    || trimmed.starts_with("```json")
                    || trimmed.starts_with("<atem:function_calls")
                    || trimmed.starts_with("to=functions.")
                    || trimmed.starts_with("[TOOL_CALLS]")
                    || (trimmed.starts_with('{') && (trimmed.contains("\"name\"") || trimmed.contains("\"function\"") || trimmed.contains("\"arguments\"")))
                    || (trimmed.starts_with('[') && trimmed.contains("\"name\""))
                    || trimmed.starts_with("call:")
                    || (trimmed.starts_with("to=") && !trimmed.starts_with("to=user"))
                    || (trimmed.starts_with("<|start|>assistant to=")
                        && !trimmed.starts_with("<|start|>assistant to=user"))
                {
                    is_tool_call = true;
                    prefix_checked = true;
                    continue;
                }

                // 3. Strip Harmony channel headers like `to=user<|message|>` or `to=user\n`
                if let Some(msg_idx) = trimmed.find("<|message|>") {
                    let user_text = &trimmed[msg_idx + "<|message|>".len()..];
                    if !user_text.is_empty() {
                        let chunk = make_chat_chunk(&model_name, user_text.to_string(), false);
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    prefix_checked = true;
                    continue;
                }

                if trimmed.starts_with("to=user\n") {
                    let user_text = &trimmed["to=user\n".len()..];
                    if !user_text.is_empty() {
                        let chunk = make_chat_chunk(&model_name, user_text.to_string(), false);
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    prefix_checked = true;
                    continue;
                }

                // If not matching any known protocol prefix, flush buffer and stream normally
                if prefix_buffer.len() > 30
                    || (!trimmed.starts_with("to=")
                        && !trimmed.starts_with("<|")
                        && !trimmed.starts_with("<tool")
                        && !trimmed.starts_with('{')
                        && !trimmed.starts_with('['))
                {
                    let chunk = make_chat_chunk(&model_name, prefix_buffer.clone(), false);
                    if tx.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                    prefix_checked = true;
                }
                continue;
            }

            if !is_tool_call {
                let chunk = make_chat_chunk(&model_name, token.text, false);
                if tx.send(Ok(chunk)).await.is_err() {
                    return;
                }
            }
        }

        let total_ns = request_start.elapsed().as_nanos() as u64;
        let result = result_rx.await.ok();
        tracing::info!("STREAM CHAT RAW RESPONSE: {:?}", full_response);
        let (cleaned_content, tool_calls) = crate::server::chat::parse_tool_calls(&full_response);
        tracing::info!("STREAM PARSED TOOL CALLS: {:?}, CLEANED CONTENT: {:?}", tool_calls, cleaned_content);
        let final_chunk = ChatResponseChunk {
            model: model_name,
            created_at: now_rfc3339(),
            message: ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls,
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
        };
        let mut line = serde_json::to_string(&final_chunk).unwrap_or_default();
        line.push('\n');
        let _ = tx.send(Ok(line)).await;
    });

    Body::from_stream(ReceiverStream::new(rx))
}

fn make_chat_chunk(model_name: &str, content: String, done: bool) -> String {
    let chunk = ChatResponseChunk {
        model: model_name.to_string(),
        created_at: now_rfc3339(),
        message: ChatMessage {
            role: "assistant".into(),
            content,
            tool_calls: None,
        },
        done,
        done_reason: None,
        total_duration: None,
        load_duration: None,
        prompt_eval_count: None,
        prompt_eval_duration: None,
        eval_count: None,
        eval_duration: None,
    };
    let mut line = serde_json::to_string(&chunk).unwrap_or_default();
    line.push('\n');
    line
}

/// Convert a token channel into an SSE streaming body for OpenAI `/v1/chat/completions`.
pub fn sse_openai_chat_stream(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    chat_id: String,
    created: u64,
) -> Body {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut full_response = String::new();
        let mut prefix_buffer = String::new();
        let mut is_tool_call = false;
        let mut prefix_checked = false;

        // Send initial role chunk
        let initial_chunk = crate::server::openai_types::OpenAIChatChunkResponse {
            id: chat_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_name.clone(),
            choices: vec![crate::server::openai_types::OpenAIChatChunkChoice {
                index: 0,
                delta: crate::server::openai_types::OpenAIChatDelta {
                    role: Some("assistant".into()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        let init_line = format!(
            "data: {}\n\n",
            serde_json::to_string(&initial_chunk).unwrap_or_default()
        );
        if tx.send(Ok(init_line)).await.is_err() {
            return;
        }

        let mut keepalive = new_keepalive_timer();

        loop {
            let token = tokio::select! {
                biased;
                _ = tx.closed() => {
                    tracing::info!("Client disconnected from /v1/chat/completions stream; aborting generation");
                    return;
                }
                _ = keepalive.tick() => {
                    if tx.send(Ok(": keep-alive\n\n".into())).await.is_err() {
                        return;
                    }
                    continue;
                }
                maybe_tok = token_rx.recv() => match maybe_tok {
                    Some(t) => t,
                    None => break,
                },
            };

            full_response.push_str(&token.text);

            if !prefix_checked {
                prefix_buffer.push_str(&token.text);
                let trimmed = prefix_buffer.trim_start();

                // Strip thought channels
                if trimmed.starts_with("<|channel>thought")
                    || trimmed.starts_with("<thought>")
                    || trimmed.starts_with("<think>")
                {
                    if let Some(end_idx) = trimmed
                        .find("<channel|>")
                        .map(|i| i + "<channel|>".len())
                        .or_else(|| trimmed.find("</thought>").map(|i| i + "</thought>".len()))
                        .or_else(|| trimmed.find("</think>").map(|i| i + "</think>".len()))
                    {
                        prefix_buffer = trimmed[end_idx..].trim_start().to_string();
                    } else {
                        continue;
                    }
                }

                let trimmed = prefix_buffer.trim_start();

                if trimmed.starts_with("<tool_call")
                    || trimmed.starts_with("<|tool_call")
                    || trimmed.starts_with("```tool_call")
                    || trimmed.starts_with("```json")
                    || trimmed.starts_with("<atem:function_calls")
                    || trimmed.starts_with("to=functions.")
                    || trimmed.starts_with("[TOOL_CALLS]")
                    || (trimmed.starts_with('{') && (trimmed.contains("\"name\"") || trimmed.contains("\"function\"") || trimmed.contains("\"arguments\"")))
                    || (trimmed.starts_with('[') && trimmed.contains("\"name\""))
                    || trimmed.starts_with("call:")
                    || (trimmed.starts_with("to=") && !trimmed.starts_with("to=user"))
                    || (trimmed.starts_with("<|start|>assistant to=")
                        && !trimmed.starts_with("<|start|>assistant to=user"))
                {
                    is_tool_call = true;
                    prefix_checked = true;
                    continue;
                }

                if let Some(msg_idx) = trimmed.find("<|message|>") {
                    let user_text = &trimmed[msg_idx + "<|message|>".len()..];
                    if !user_text.is_empty() {
                        let chunk = make_openai_chat_chunk(&chat_id, &model_name, created, Some(user_text.to_string()), None, None);
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    prefix_checked = true;
                    continue;
                }

                if trimmed.starts_with("to=user\n") {
                    let user_text = &trimmed["to=user\n".len()..];
                    if !user_text.is_empty() {
                        let chunk = make_openai_chat_chunk(&chat_id, &model_name, created, Some(user_text.to_string()), None, None);
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    prefix_checked = true;
                    continue;
                }

                if prefix_buffer.len() > 30
                    || (!trimmed.starts_with("to=")
                        && !trimmed.starts_with("<|")
                        && !trimmed.starts_with("<tool")
                        && !trimmed.starts_with('{')
                        && !trimmed.starts_with('['))
                {
                    let chunk = make_openai_chat_chunk(&chat_id, &model_name, created, Some(prefix_buffer.clone()), None, None);
                    if tx.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                    prefix_checked = true;
                }
                continue;
            }

            if !is_tool_call {
                let chunk = make_openai_chat_chunk(&chat_id, &model_name, created, Some(token.text), None, None);
                if tx.send(Ok(chunk)).await.is_err() {
                    return;
                }
            }
        }

        let _ = result_rx.await;
        let (_cleaned_content, tool_calls) = crate::server::chat::parse_tool_calls(&full_response);

        if let Some(calls) = tool_calls {
            let openai_calls: Vec<crate::server::openai_types::OpenAIToolCall> = calls
                .into_iter()
                .enumerate()
                .map(|(i, tc)| crate::server::openai_types::OpenAIToolCall {
                    id: tc.id.unwrap_or_else(|| format!("call_{}_{}", chat_id, i)),
                    call_type: "function".into(),
                    function: crate::server::openai_types::OpenAIFunctionCall {
                        name: tc.function.name,
                        arguments: serde_json::to_string(&tc.function.arguments).unwrap_or_default(),
                    },
                })
                .collect();

            let tool_chunk = make_openai_chat_chunk(&chat_id, &model_name, created, None, Some(openai_calls), Some("tool_calls".into()));
            let _ = tx.send(Ok(tool_chunk)).await;
        } else {
            let finish_chunk = make_openai_chat_chunk(&chat_id, &model_name, created, None, None, Some("stop".into()));
            let _ = tx.send(Ok(finish_chunk)).await;
        }

        let _ = tx.send(Ok("data: [DONE]\n\n".into())).await;
    });

    Body::from_stream(ReceiverStream::new(rx))
}

fn make_openai_chat_chunk(
    chat_id: &str,
    model_name: &str,
    created: u64,
    content: Option<String>,
    tool_calls: Option<Vec<crate::server::openai_types::OpenAIToolCall>>,
    finish_reason: Option<String>,
) -> String {
    let chunk = crate::server::openai_types::OpenAIChatChunkResponse {
        id: chat_id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model_name.to_string(),
        choices: vec![crate::server::openai_types::OpenAIChatChunkChoice {
            index: 0,
            delta: crate::server::openai_types::OpenAIChatDelta {
                role: None,
                content,
                tool_calls,
            },
            finish_reason,
        }],
    };
    format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap_or_default())
}

/// Convert a token channel into an SSE streaming body for OpenAI `/v1/completions`.
pub fn sse_openai_completion_stream(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    completion_id: String,
    created: u64,
) -> Body {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut keepalive = new_keepalive_timer();

        loop {
            let token = tokio::select! {
                biased;
                _ = tx.closed() => {
                    tracing::info!("Client disconnected from /v1/completions stream; aborting generation");
                    return;
                }
                _ = keepalive.tick() => {
                    if tx.send(Ok(": keep-alive\n\n".into())).await.is_err() {
                        return;
                    }
                    continue;
                }
                maybe_tok = token_rx.recv() => match maybe_tok {
                    Some(t) => t,
                    None => break,
                },
            };

            let chunk = crate::server::openai_types::OpenAICompletionChunkResponse {
                id: completion_id.clone(),
                object: "text_completion",
                created,
                model: model_name.clone(),
                choices: vec![crate::server::openai_types::OpenAICompletionChunkChoice {
                    index: 0,
                    text: token.text,
                    finish_reason: None,
                }],
            };
            let line = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap_or_default());
            if tx.send(Ok(line)).await.is_err() {
                return;
            }
        }

        let _ = result_rx.await;
        let final_chunk = crate::server::openai_types::OpenAICompletionChunkResponse {
            id: completion_id,
            object: "text_completion",
            created,
            model: model_name,
            choices: vec![crate::server::openai_types::OpenAICompletionChunkChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some("stop".into()),
            }],
        };
        let _ = tx.send(Ok(format!("data: {}\n\n", serde_json::to_string(&final_chunk).unwrap_or_default()))).await;
        let _ = tx.send(Ok("data: [DONE]\n\n".into())).await;
    });

    Body::from_stream(ReceiverStream::new(rx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sse_chat_stream_disconnect_closes_token_channel() {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (_result_tx, result_rx) = oneshot::channel::<GenerationResult>();

        let body = sse_openai_chat_stream(
            "gemma4-26b".into(),
            token_rx,
            result_rx,
            "chatcmpl-test".into(),
            1234567890,
        );

        // Simulate model emitting `<|channel>thought` tokens (buffered silently)
        token_tx
            .send(GeneratedToken {
                text: "<|channel>thought\nanalyzing project files...".into(),
                token_id: 1,
                tok_per_sec: 12.0,
                is_eog: false,
            })
            .unwrap();

        // Client disconnects (drops HTTP response body) while thought buffering is active
        drop(body);

        // Stream task must detect tx.closed() and drop token_rx promptly
        tokio::time::timeout(Duration::from_millis(250), token_tx.closed())
            .await
            .expect("token_rx should be dropped promptly when client disconnects");
    }
}
