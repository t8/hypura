use serde::{Deserialize, Serialize};

// ── Request Types ──

#[derive(Debug, Deserialize)]
pub struct OpenAIChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAIChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_stop_sequences")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub num_ctx: Option<u32>,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
}

impl OpenAIChatCompletionRequest {
    pub fn requested_context(&self) -> Option<u32> {
        if let Some(ctx) = self.num_ctx {
            return Some(ctx);
        }
        if let Some(ref opts) = self.options {
            if let Some(ctx) = opts.get("num_ctx").and_then(|v| v.as_u64()) {
                return Some(ctx as u32);
            }
        }
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIChatMessage {
    pub role: String,
    #[serde(
        default,
        deserialize_with = "deserialize_openai_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

fn deserialize_openai_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val = Option::<serde_json::Value>::deserialize(deserializer)?;
    match val {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(serde_json::Value::Array(parts)) => {
            let mut joined = String::new();
            for part in parts {
                match part {
                    serde_json::Value::String(s) => joined.push_str(&s),
                    serde_json::Value::Object(map) => {
                        if let Some(text) = map
                            .get("text")
                            .or_else(|| map.get("content"))
                            .and_then(|v| v.as_str())
                        {
                            if !joined.is_empty() && !joined.ends_with('\n') && !text.starts_with('\n') {
                                joined.push('\n');
                            }
                            joined.push_str(text);
                        }
                    }
                    other => joined.push_str(&other.to_string()),
                }
            }
            Ok(Some(joined))
        }
        Some(other) => Ok(Some(other.to_string())),
    }
}

fn deserialize_stop_sequences<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val = Option::<serde_json::Value>::deserialize(deserializer)?;
    match val {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(vec![s])),
        Some(serde_json::Value::Array(arr)) => {
            let stops = arr
                .into_iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            Ok(Some(stops))
        }
        Some(_) => Ok(None),
    }
}

#[derive(Debug, Deserialize)]
pub struct OpenAICompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_stop_sequences")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub num_ctx: Option<u32>,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
}

impl OpenAICompletionRequest {
    pub fn requested_context(&self) -> Option<u32> {
        if let Some(ctx) = self.num_ctx {
            return Some(ctx);
        }
        if let Some(ref opts) = self.options {
            if let Some(ctx) = opts.get("num_ctx").and_then(|v| v.as_u64()) {
                return Some(ctx as u32);
            }
        }
        None
    }
}

// ── Response Types ──

#[derive(Debug, Serialize)]
pub struct OpenAIModelList {
    pub object: &'static str,
    pub data: Vec<OpenAIModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIModelEntry {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChatChoice>,
    pub usage: OpenAIUsage,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChatChoice {
    pub index: u32,
    pub message: OpenAIChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChatChunkResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChatChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChatChunkChoice {
    pub index: u32,
    pub delta: OpenAIChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct OpenAIChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct OpenAICompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAICompletionChoice>,
    pub usage: OpenAIUsage,
}

#[derive(Debug, Serialize)]
pub struct OpenAICompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenAICompletionChunkResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAICompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct OpenAICompletionChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
