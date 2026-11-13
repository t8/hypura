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
const STREAM_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(10);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamFilterState {
    Prefix,
    StreamingProse,
    SuppressToolCall,
}

/// Stateful token-stream filter shared by `/api/chat` and `/v1/chat/completions`
/// so reasoning blocks, raw tool-call XML/JSON/tags, and turn-end control tokens
/// never leak into user-visible `content` deltas.
struct ChatStreamFilter {
    state: StreamFilterState,
    buffer: String,
}

impl ChatStreamFilter {
    fn new() -> Self {
        Self {
            state: StreamFilterState::Prefix,
            buffer: String::new(),
        }
    }

    fn strip_control_suffixes(mut s: String) -> String {
        for tok in [
            "<turn|>",
            "<|turn|>",
            "<|im_end|>",
            "<|im_start|>",
            "<|end_of_text|>",
            "<|end_of_turn|>",
            "<end_of_turn>",
            "<SPECIAL_12>",
            "<SPECIAL_11>",
            "</s>",
        ] {
            if s.contains(tok) {
                s = s.replace(tok, "");
            }
        }
        s
    }

    fn is_possible_prefix_of_protocol_tag(trimmed: &str) -> bool {
        const PROTOCOL_PREFIXES: &[&str] = &[
            "<|channel>thought",
            "<|channel|>thought",
            "<|channel|>analysis",
            "<|thought>",
            "<thought>",
            "<think>",
            "[THINK]",
            "<tool_call",
            "<|tool_call",
            "<toolcall",
            "<TOOLCALL",
            "<function=",
            "<|call>",
            "```tool_call",
            "```json",
            "<atem:function_calls",
            "[TOOL_CALLS]",
            "call:",
            "to=functions.",
            "to=user\n",
            "<|start|>assistant",
        ];
        for p in PROTOCOL_PREFIXES {
            if p.starts_with(trimmed) {
                return true;
            }
        }
        // Hold back short potential protocol prefixes (`<...`, `[...`, `` `... ``, `to=...`)
        if trimmed.len() < 28
            && (trimmed.starts_with('<')
                || trimmed.starts_with("```")
                || trimmed.starts_with('`')
                || trimmed.starts_with("to="))
        {
            return true;
        }
        // Hold back `{...` or `[...` briefly in case it's a raw JSON tool call `{"name": ...}`
        if trimmed.len() < 64 && (trimmed.starts_with('{') || trimmed.starts_with('[')) {
            return true;
        }
        false
    }

    fn find_midstream_tool_marker(s: &str) -> Option<usize> {
        const MIDSTREAM_TOOL_MARKERS: &[&str] = &[
            "<tool_call",
            "<|tool_call",
            "<toolcall",
            "<TOOLCALL",
            "<function=",
            "<|call>",
            "[TOOL_CALLS]",
            "```tool_call",
            "<atem:function_calls",
            "to=functions.",
        ];
        MIDSTREAM_TOOL_MARKERS
            .iter()
            .filter_map(|m| s.find(m))
            .min()
    }

    fn trailing_holdback_len(s: &str) -> usize {
        const HOLD_TAGS: &[&str] = &[
            "<tool_call",
            "<|tool_call",
            "<toolcall",
            "<TOOLCALL",
            "<function=",
            "<|call>",
            "[TOOL_CALLS]",
            "```tool_call",
            "<atem:function_calls",
            "to=functions.",
            "<turn|>",
            "<|turn|>",
            "<|im_end|>",
            "<|end_of_text|>",
            "<end_of_turn>",
            "<SPECIAL_12>",
            "</s>",
        ];
        let max_check = s.len().min(20);
        let start_from = s.len() - max_check;
        for (offset, _) in s[start_from..].char_indices() {
            let idx = start_from + offset;
            let suffix = &s[idx..];
            if HOLD_TAGS.iter().any(|tag| tag.starts_with(suffix)) {
                return s.len() - idx;
            }
        }
        0
    }

    fn push(&mut self, piece: &str) -> Option<String> {
        match self.state {
            StreamFilterState::SuppressToolCall => None,
            StreamFilterState::Prefix => {
                self.buffer.push_str(piece);

                // 1. Strip any completed leading thought blocks; hold back while inside an unclosed thought block.
                let thought_pairs = [
                    ("<|channel>thought", "<channel|>"),
                    ("<|channel|>thought", "<|end|>"),
                    ("<|channel|>analysis", "<|end|>"),
                    ("<|thought>", "<thought|>"),
                    ("<thought>", "</thought>"),
                    ("<think>", "</think>"),
                    ("[THINK]", "[/THINK]"),
                ];

                loop {
                    let trimmed = self.buffer.trim_start();
                    if trimmed.is_empty() {
                        return None;
                    }
                    let mut stripped_one = false;
                    for (open_tag, close_tag) in thought_pairs {
                        if trimmed.starts_with(open_tag) {
                            let after_open = &trimmed[open_tag.len()..];
                            let close_match = after_open
                                .find(close_tag)
                                .map(|i| i + close_tag.len())
                                .or_else(|| {
                                    if open_tag == "<|channel|>thought" {
                                        after_open.find("<channel|>").map(|i| i + "<channel|>".len())
                                    } else if open_tag == "<|thought>" {
                                        after_open.find("</thought|>").map(|i| i + "</thought|>".len())
                                    } else {
                                        None
                                    }
                                });

                            if let Some(end_rel) = close_match {
                                self.buffer = after_open[end_rel..].trim_start().to_string();
                                stripped_one = true;
                                break;
                            } else {
                                // Still inside active reasoning block
                                return None;
                            }
                        }
                    }
                    if !stripped_one {
                        break;
                    }
                }

                let trimmed = self.buffer.trim_start();
                if trimmed.is_empty() {
                    return None;
                }

                // 2. Detect tool calls early so raw XML/JSON/Mistral/Gemma tags never leak to content
                if trimmed.starts_with("<tool_call")
                    || trimmed.starts_with("<|tool_call")
                    || trimmed.starts_with("<toolcall")
                    || trimmed.starts_with("<TOOLCALL")
                    || trimmed.starts_with("<function=")
                    || trimmed.starts_with("<|call>")
                    || trimmed.starts_with("```tool_call")
                    || trimmed.starts_with("```json")
                    || trimmed.starts_with("<atem:function_calls")
                    || trimmed.starts_with("to=functions.")
                    || trimmed.starts_with("[TOOL_CALLS]")
                    || (trimmed.starts_with('{')
                        && (trimmed.contains("\"name\"")
                            || trimmed.contains("\"function\"")
                            || trimmed.contains("\"arguments\"")))
                    || (trimmed.starts_with('[') && trimmed.contains("\"name\""))
                    || trimmed.starts_with("call:")
                    || (trimmed.starts_with("to=") && !trimmed.starts_with("to=user"))
                    || (trimmed.starts_with("<|start|>assistant to=")
                        && !trimmed.starts_with("<|start|>assistant to=user"))
                {
                    self.state = StreamFilterState::SuppressToolCall;
                    self.buffer.clear();
                    return None;
                }

                // 3. Strip Harmony channel headers like `to=user<|message|>` or `to=user\n`
                if let Some(msg_idx) = trimmed.find("<|message|>") {
                    let user_text = trimmed[msg_idx + "<|message|>".len()..].to_string();
                    self.buffer.clear();
                    self.state = StreamFilterState::StreamingProse;
                    return if user_text.is_empty() {
                        None
                    } else {
                        self.push(&user_text)
                    };
                }

                if let Some(user_text) = trimmed.strip_prefix("to=user\n") {
                    let rem = user_text.to_string();
                    self.buffer.clear();
                    self.state = StreamFilterState::StreamingProse;
                    return if rem.is_empty() {
                        None
                    } else {
                        self.push(&rem)
                    };
                }

                // 4. Hold back while buffer could still complete into a protocol/tool tag
                if Self::is_possible_prefix_of_protocol_tag(trimmed) {
                    return None;
                }

                // Confirmed prose: transition to StreamingProse and run mid-stream tail check
                let initial = std::mem::take(&mut self.buffer);
                self.state = StreamFilterState::StreamingProse;
                self.push(&initial)
            }
            StreamFilterState::StreamingProse => {
                self.buffer.push_str(piece);

                // If a tool call starts mid-stream after preamble prose, emit only the prose before it
                if let Some(tool_idx) = Self::find_midstream_tool_marker(&self.buffer) {
                    let before = Self::strip_control_suffixes(self.buffer[..tool_idx].to_string());
                    self.buffer.clear();
                    self.state = StreamFilterState::SuppressToolCall;
                    return if before.is_empty() { None } else { Some(before) };
                }

                self.buffer = Self::strip_control_suffixes(std::mem::take(&mut self.buffer));
                let hold = Self::trailing_holdback_len(&self.buffer);
                let safe_len = self.buffer.len().saturating_sub(hold);
                if safe_len == 0 {
                    return None;
                }
                let emit = self.buffer[..safe_len].to_string();
                self.buffer.drain(..safe_len);
                Some(emit)
            }
        }
    }

    fn finish(&mut self) -> Option<String> {
        match self.state {
            StreamFilterState::SuppressToolCall => {
                self.buffer.clear();
                None
            }
            StreamFilterState::Prefix | StreamFilterState::StreamingProse => {
                if self.buffer.is_empty() {
                    return None;
                }
                let rem = Self::strip_control_suffixes(std::mem::take(&mut self.buffer));
                if rem.trim().is_empty() {
                    None
                } else {
                    Some(rem)
                }
            }
        }
    }
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
        let mut filter = ChatStreamFilter::new();
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

            if let Some(visible) = filter.push(&token.text) {
                let chunk = make_chat_chunk(&model_name, visible, false);
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

        if tool_calls.is_none() {
            if let Some(rem) = filter.finish() {
                let chunk = make_chat_chunk(&model_name, rem, false);
                if tx.send(Ok(chunk)).await.is_err() {
                    return;
                }
            }
        }

        let final_chunk = ChatResponseChunk {
            model: model_name,
            created_at: now_rfc3339(),
            message: ChatMessage {
                role: "assistant".into(),
                content: String::new(),
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
            ..Default::default()
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
        let mut filter = ChatStreamFilter::new();

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
                    // Emit an empty-content delta chunk (`data: {...}\n\n`) rather than only
                    // an SSE `: keep-alive` comment so OpenAI SDK / coding-agent stream parsers
                    // yield a chunk and reset their application-level per-chunk inactivity timer
                    // while `<think>...</think>` and `<tool_call>...</tool_call>` are buffered.
                    let hb_chunk = make_openai_chat_chunk(
                        &chat_id,
                        &model_name,
                        created,
                        Some(String::new()),
                        None,
                        None,
                    );
                    if tx.send(Ok(hb_chunk)).await.is_err() {
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

            if let Some(visible) = filter.push(&token.text) {
                let chunk = make_openai_chat_chunk(
                    &chat_id,
                    &model_name,
                    created,
                    Some(visible),
                    None,
                    None,
                );
                if tx.send(Ok(chunk)).await.is_err() {
                    return;
                }
            }
        }

        let _ = result_rx.await;
        let (_cleaned_content, tool_calls) = crate::server::chat::parse_tool_calls(&full_response);

        if tool_calls.is_none() {
            if let Some(rem) = filter.finish() {
                let chunk = make_openai_chat_chunk(
                    &chat_id,
                    &model_name,
                    created,
                    Some(rem),
                    None,
                    None,
                );
                let _ = tx.send(Ok(chunk)).await;
            }
        }

        if let Some(calls) = tool_calls {
            let openai_calls: Vec<crate::server::openai_types::OpenAIToolCall> = calls
                .into_iter()
                .enumerate()
                .map(|(i, tc)| crate::server::openai_types::OpenAIToolCall {
                    index: Some(i as u32),
                    id: tc
                        .id
                        .unwrap_or_else(crate::server::chat::generate_tool_call_id),
                    call_type: "function".into(),
                    function: crate::server::openai_types::OpenAIFunctionCall {
                        name: tc.function.name,
                        arguments: serde_json::to_string(&tc.function.arguments)
                            .unwrap_or_else(|_| "{}".into()),
                    },
                })
                .collect();

            // Per OpenAI SSE spec, emit the `delta.tool_calls` chunk with `finish_reason: null`
            // first, followed by an empty-delta chunk carrying `finish_reason: "tool_calls"`.
            let tool_delta_chunk = make_openai_chat_chunk(
                &chat_id,
                &model_name,
                created,
                None,
                Some(openai_calls),
                None,
            );
            let _ = tx.send(Ok(tool_delta_chunk)).await;

            let finish_chunk = make_openai_chat_chunk(
                &chat_id,
                &model_name,
                created,
                None,
                None,
                Some("tool_calls".into()),
            );
            let _ = tx.send(Ok(finish_chunk)).await;
        } else {
            let finish_chunk = make_openai_chat_chunk(
                &chat_id,
                &model_name,
                created,
                None,
                None,
                Some("stop".into()),
            );
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

/// Convert a pull progress receiver into an NDJSON streaming body for `/api/pull`.
pub fn ndjson_pull_stream(
    mut progress_rx: mpsc::Receiver<crate::server::ollama_types::PullResponseChunk>,
) -> Body {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(32);

    tokio::spawn(async move {
        while let Some(chunk) = progress_rx.recv().await {
            let mut line = serde_json::to_string(&chunk).unwrap_or_default();
            line.push('\n');
            if tx.send(Ok(line)).await.is_err() {
                break;
            }
        }
    });

    Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx))
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

    #[tokio::test]
    async fn test_sse_openai_tool_calls_include_index_and_separate_finish_chunk() {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();

        let body = sse_openai_chat_stream(
            "qwen3.8-26b".into(),
            token_rx,
            result_rx,
            "chatcmpl-tool".into(),
            1700000000,
        );

        // Simulate split tokens: prefilled `<think>\n`, thinking mentioning `</function>`,
        // closing `</think>`, optional preamble, and XML `<tool_call><function=...>`
        for tok in [
            "<think>\n",
            "Should call <function=list_directory> on root",
            "</think>\n",
            "Checking root directory.",
            "\n<tool",
            "_call>\n<function=list_directory>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>",
        ] {
            token_tx
                .send(GeneratedToken {
                    text: tok.into(),
                    token_id: 1,
                    tok_per_sec: 25.0,
                    is_eog: false,
                })
                .unwrap();
        }
        drop(token_tx);
        drop(result_tx);

        let bytes = axum::body::to_bytes(body, 64 * 1024).await.unwrap();
        let sse_text = String::from_utf8(bytes.to_vec()).unwrap();

        // Raw XML tags and thought blocks must never leak into streamed content deltas
        assert!(!sse_text.contains("<think>"), "leaked <think> in SSE: {sse_text}");
        assert!(!sse_text.contains("<tool_call"), "leaked <tool_call in SSE: {sse_text}");
        assert!(!sse_text.contains("<function="), "leaked <function= in SSE: {sse_text}");

        // Tool call delta chunk must include `"index":0` and `finish_reason:null`,
        // followed by a separate chunk with `"finish_reason":"tool_calls"`.
        assert!(
            sse_text.contains("\"tool_calls\":[{\"index\":0,"),
            "missing index:0 in delta.tool_calls: {sse_text}"
        );
        assert!(
            sse_text.contains("\"name\":\"list_directory\""),
            "missing function name in delta.tool_calls: {sse_text}"
        );
        assert!(
            sse_text.contains("\"delta\":{},\"finish_reason\":\"tool_calls\""),
            "missing separate finish_reason:tool_calls chunk: {sse_text}"
        );
        assert!(sse_text.ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn test_chat_stream_filter_holds_back_codefence_and_mistral_think() {
        // Token 1 is "```" followed by "json\n{\"name\":...}" -> must not leak "```"
        let mut f1 = ChatStreamFilter::new();
        assert_eq!(f1.push("```"), None);
        assert_eq!(f1.push("json\n{\"name\":\"ls\",\"arguments\":{}}```"), None);
        assert_eq!(f1.finish(), None);

        // Mistral `[THINK]...[/THINK][TOOL_CALLS]ls[ARGS]{}` split across tokens
        let mut f2 = ChatStreamFilter::new();
        assert_eq!(f2.push("[TH"), None);
        assert_eq!(f2.push("INK]reasoning"), None);
        assert_eq!(f2.push("[/THINK][TOOL_"), None);
        assert_eq!(f2.push("CALLS]ls[ARGS]{}"), None);
        assert_eq!(f2.finish(), None);
    }
}
