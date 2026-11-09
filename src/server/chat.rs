use std::sync::atomic::{AtomicU32, Ordering};
use crate::server::ollama_types::{ChatMessage, FunctionCall, ToolCall};

static TOOL_CALL_SEQ: AtomicU32 = AtomicU32::new(1);

/// Generate a unique 9-character alphanumeric tool call ID (`"call"` + 5 base-36/hex chars).
/// Satisfies both OpenAI's string `tool_call.id` requirement and Mistral Small 3.2's strict
/// 9-character alphanumeric `[CALL_ID]` template constraint.
pub fn generate_tool_call_id() -> String {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let seq = TOOL_CALL_SEQ.fetch_add(1, Ordering::Relaxed);
    let mixed = nanos.wrapping_mul(1_664_525).wrapping_add(seq.wrapping_mul(1_013_904_223));
    format!("call{:05x}", mixed & 0xFFFFF)
}

/// Normalize any incoming/historical tool call ID into a deterministic 9-character
/// alphanumeric string for Mistral Small 3.2 `[CALL_ID]` / `[TOOL_RESULTS]` compatibility.
fn normalize_mistral_call_id(raw_id: &str) -> String {
    let alnum: String = raw_id.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if alnum.len() == 9 {
        return alnum;
    }
    if alnum.len() > 9 {
        // Keep trailing 9 chars (most entropy lives at the suffix of `call_..._0` / UUIDs)
        return alnum[alnum.len() - 9..].to_string();
    }
    // Deterministic FNV-1a pad to 9 chars
    let mut hash: u32 = 0x811c_9dc5;
    for b in raw_id.as_bytes() {
        hash ^= *b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    let combined = format!("{}{:08x}", alnum, hash);
    combined[..9].to_string()
}

/// Format chat messages (and optional tool definitions) into a prompt string
/// customized for the underlying model architecture (e.g. GPT-OSS / Harmony vs ChatML).
pub fn format_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
    hint: Option<&str>,
) -> String {
    let l = hint.unwrap_or_default().to_lowercase();

    let is_gpt_oss = l.contains("gptoss") || l.contains("gpt-oss") || l.contains("openai_moe");
    if is_gpt_oss {
        return format_gptoss_chat_prompt(messages, tools);
    }

    let is_qwen = l.contains("qwen");
    if is_qwen {
        return format_qwen_chat_prompt(messages, tools, &l);
    }

    let is_nemotron = l.contains("nemotron")
        || l.contains("<special_10>")
        || l.contains("<toolcall>");
    if is_nemotron {
        return format_nemotron_chat_prompt(messages, tools, &l);
    }

    let is_granite = l.contains("granite") || l.contains("<|start_of_role|>");
    if is_granite {
        return format_granite_chat_prompt(messages, tools);
    }

    let is_glm = l.contains("glm") || l.contains("[gmask]");
    if is_glm {
        return format_glm_chat_prompt(messages, tools);
    }

    let is_gemma4 = l.contains("gemma4")
        || l.contains("gemma-4")
        || l.contains("gemma_4")
        || l.contains("<|turn>")
        || l.contains("<|tool>declaration:");
    if is_gemma4 {
        return format_gemma4_chat_prompt(messages, tools);
    }

    let is_gemma = l.contains("gemma") || l.contains("<start_of_turn>");
    if is_gemma {
        return format_gemma_chat_prompt(messages, tools);
    }

    let is_mistral = l.contains("mistral")
        || l.contains("ministral")
        || l.contains("codestral")
        || l.contains("devstral")
        || l.contains("magistral")
        || l.contains("pixtral")
        || (l.contains("nemo") && !l.contains("nemotron"))
        || l.contains("[available_tools]")
        || l.contains("[tool_calls]")
        || l.contains("[system_prompt]");
    if is_mistral {
        return format_mistral_chat_prompt(messages, tools, &l);
    }

    let mut prompt = String::new();

    // Default ChatML format
    let system_msg = messages.iter().find(|m| m.role == "system");
    let base_system = system_msg.map(|m| m.content.as_str());

    let tools_array = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(val),
        _ => None,
    });

    if let Some(tools_json) = tools_array {
        let clean_tools: Vec<&serde_json::Value> = match tools_json {
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|t| t.get("function").unwrap_or(t))
                .collect(),
            _ => vec![tools_json],
        };
        let tools_str = serde_json::to_string_pretty(&clean_tools).unwrap_or_default();
        prompt.push_str("<|im_start|>system\n");
        if let Some(sys) = base_system {
            prompt.push_str(sys);
            prompt.push_str("\n\n");
        }
        prompt.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n");
        prompt.push_str("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        prompt.push_str(&tools_str);
        prompt.push_str("\n</tools>\n\nIMPORTANT TOOL CALL RULES:\n");
        prompt.push_str("1. If a function requires parameters (such as latitude, longitude, or ID) that are unknown or missing in the user query, ALWAYS call the prerequisite lookup tool FIRST (e.g., call `get_coordinates` or geocoding before searching for devices around a location).\n");
        prompt.push_str("2. NEVER invent, guess, or hallucinate latitude, longitude, or coordinate numbers.\n");
        prompt.push_str("3. Return each function call as a JSON object within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"arg_name\": \"arg_value\"}}\n</tool_call>\n");
        prompt.push_str("4. When answering the user directly after tool results, provide the final answer in markdown and STOP immediately.<|im_end|>\n");
    } else if let Some(sys) = base_system {
        prompt.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }

    for msg in messages {
        if msg.role == "system" {
            continue;
        }
        if msg.role == "tool" {
            prompt.push_str("<|im_start|>tool\n");
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
            continue;
        }
        prompt.push_str(&format!("<|im_start|>{}\n", msg.role));
        if !msg.content.is_empty() {
            prompt.push_str(&msg.content);
        }
        if let Some(ref tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                let call_obj = serde_json::json!({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                });
                prompt.push_str(&format!(
                    "<tool_call>\n{}\n</tool_call>",
                    serde_json::to_string(&call_obj).unwrap_or_default()
                ));
            }
        }
        prompt.push_str("<|im_end|>\n");
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Specialized prompt formatting for OpenAI Harmony / GPT-OSS models.
fn format_gptoss_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
) -> String {
    let mut prompt = String::new();

    let tools_array = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(val),
        _ => None,
    });

    let system_msg = messages.iter().find(|m| m.role == "system");
    let base_system = system_msg
        .map(|m| m.content.as_str())
        .unwrap_or("You are a helpful assistant.");

    if let Some(tools_json) = tools_array {
        let tools_str = serde_json::to_string_pretty(tools_json).unwrap_or_default();
        prompt.push_str("<|start|>system<|message|>");
        prompt.push_str(base_system);
        prompt.push_str("\n\nYou have access to the following functions:\n```json\n");
        prompt.push_str(&tools_str);
        prompt.push_str("\n```\n\nTo call a function, respond strictly in this format:\n\
            to=functions.<function_name><|channel|>commentary<|message|>{\"arg_name\": \"arg_value\"}<|call|>\n\
            When no function call is needed, respond with standard text.<|end|>\n");
    } else if let Some(sys) = system_msg {
        prompt.push_str(&format!("<|start|>system<|message|>{}<|end|>\n", sys.content));
    }

    for msg in messages {
        if msg.role == "system" {
            continue;
        }

        match msg.role.as_str() {
            "user" => {
                prompt.push_str(&format!("<|start|>user<|message|>{}<|end|>\n", msg.content));
            }
            "assistant" => {
                if let Some(ref tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        let args_str =
                            serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                        prompt.push_str(&format!(
                            "<|start|>assistant to=functions.{}<|channel|>commentary<|message|>{args_str}<|call|>\n",
                            tc.function.name
                        ));
                    }
                }
                if !msg.content.is_empty() {
                    prompt.push_str(&format!(
                        "<|start|>assistant<|message|>{}<|return|>\n",
                        msg.content
                    ));
                }
            }
            "tool" => {
                prompt.push_str(&format!("<|start|>tool<|message|>{}<|end|>\n", msg.content));
            }
            other => {
                prompt.push_str(&format!("<|start|>{other}<|message|>{}<|end|>\n", msg.content));
            }
        }
    }

    prompt.push_str("<|start|>assistant");
    prompt
}

/// Format XML `<function=...><parameter=...>...</parameter></function>` tool call block
/// shared by Qwen 3.5 / 3.8 / Qwen3-Coder and NVIDIA Nemotron-3 30B.
fn push_xml_function_tool_call(prompt: &mut String, tc: &ToolCall) {
    prompt.push_str("<tool_call>\n<function=");
    prompt.push_str(&tc.function.name);
    prompt.push_str(">\n");
    if let Some(obj) = tc.function.arguments.as_object() {
        for (k, v) in obj {
            prompt.push_str("<parameter=");
            prompt.push_str(k);
            prompt.push_str(">\n");
            match v {
                serde_json::Value::String(s) => prompt.push_str(s),
                other => prompt.push_str(&serde_json::to_string(other).unwrap_or_default()),
            }
            prompt.push_str("\n</parameter>\n");
        }
    }
    prompt.push_str("</function>\n</tool_call>");
}

/// Render tool definitions as `<function><name>...</name>...</function>` XML inside `<tools>`.
fn push_xml_tool_definitions(prompt: &mut String, tools_arr: &[serde_json::Value]) {
    prompt.push_str("<tools>");
    for tool_entry in tools_arr {
        let func = tool_entry.get("function").unwrap_or(tool_entry);
        let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
        prompt.push_str("\n<function>\n<name>");
        prompt.push_str(name);
        prompt.push_str("</name>");
        if let Some(desc) = func.get("description").and_then(|v| v.as_str()) {
            prompt.push_str("\n<description>");
            prompt.push_str(desc.trim());
            prompt.push_str("</description>");
        }
        prompt.push_str("\n<parameters>");
        if let Some(params) = func.get("parameters").and_then(|v| v.as_object()) {
            if let Some(props) = params.get("properties").and_then(|v| v.as_object()) {
                for (p_name, p_fields) in props {
                    prompt.push_str("\n<parameter>\n<name>");
                    prompt.push_str(p_name);
                    prompt.push_str("</name>");
                    if let Some(p_type) = p_fields.get("type") {
                        prompt.push_str("\n<type>");
                        if let Some(s) = p_type.as_str() {
                            prompt.push_str(s);
                        } else {
                            prompt.push_str(&p_type.to_string());
                        }
                        prompt.push_str("</type>");
                    }
                    if let Some(p_desc) = p_fields.get("description").and_then(|v| v.as_str()) {
                        prompt.push_str("\n<description>");
                        prompt.push_str(p_desc.trim());
                        prompt.push_str("</description>");
                    }
                    if let Some(p_enum) = p_fields.get("enum") {
                        prompt.push_str("\n<enum>");
                        prompt.push_str(&serde_json::to_string(p_enum).unwrap_or_default());
                        prompt.push_str("</enum>");
                    }
                    prompt.push_str("\n</parameter>");
                }
            }
            if let Some(req) = params.get("required") {
                prompt.push_str("\n<required>");
                prompt.push_str(&serde_json::to_string(req).unwrap_or_default());
                prompt.push_str("</required>");
            }
        }
        prompt.push_str("\n</parameters>\n</function>");
    }
    prompt.push_str("\n</tools>");
}

/// Specialized prompt formatting for Qwen 3.5 / 3.8 / Qwen3-Coder / Qwen 2.5 models.
fn format_qwen_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
    hint_lower: &str,
) -> String {
    let uses_xml_tools = hint_lower.contains("<function=")
        || hint_lower.contains("qwen35")
        || hint_lower.contains("qwen3.5")
        || hint_lower.contains("qwen3_5")
        || hint_lower.contains("qwen3.8")
        || hint_lower.contains("qwen38")
        || hint_lower.contains("qwen3_8")
        || hint_lower.contains("qwen3next")
        || hint_lower.contains("qwen3-coder")
        || hint_lower.contains("qwen3_coder");

    if uses_xml_tools {
        let mut prompt = String::new();
        let tools_slice = tools.and_then(|val| match val {
            serde_json::Value::Array(arr) if !arr.is_empty() => Some(arr.as_slice()),
            _ => None,
        });

        let system_msg = messages.iter().find(|m| m.role == "system");
        let base_system = system_msg.map(|m| m.content.trim()).filter(|s| !s.is_empty());

        let explicit_think = messages
            .iter()
            .any(|m| (m.role == "user" || m.role == "system") && m.content.contains("/think"));
        let is_dense_heavy_tool_turn = tools_slice.is_some()
            && !explicit_think
            && !hint_lower.contains("moe")
            && (hint_lower.contains("27b") || hint_lower.contains("32b"));
        let no_think = messages
            .iter()
            .any(|m| (m.role == "user" || m.role == "system") && m.content.contains("/no_think"))
            || is_dense_heavy_tool_turn;
        let is_coder_non_thinking = (hint_lower.contains("qwen3-coder") || hint_lower.contains("qwen3_coder"))
            && !hint_lower.contains("<think>");

        if let Some(arr) = tools_slice {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for tool in arr {
                let normalized = if tool.get("function").is_some() {
                    serde_json::to_string(tool).unwrap_or_default()
                } else {
                    serde_json::to_string(&serde_json::json!({
                        "type": "function",
                        "function": tool,
                    }))
                    .unwrap_or_default()
                };
                prompt.push('\n');
                prompt.push_str(&normalized);
            }
            prompt.push_str("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
            if let Some(sys) = base_system {
                prompt.push_str("\n\n");
                prompt.push_str(sys);
            }
            prompt.push_str("<|im_end|>\n");
        } else if let Some(sys) = base_system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(sys);
            prompt.push_str("<|im_end|>\n");
        }

        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            if msg.role == "system" {
                i += 1;
                continue;
            }
            if msg.role == "tool" {
                prompt.push_str("<|im_start|>user");
                while i < messages.len() && messages[i].role == "tool" {
                    prompt.push_str("\n<tool_response>\n");
                    prompt.push_str(messages[i].content.trim());
                    prompt.push_str("\n</tool_response>");
                    i += 1;
                }
                prompt.push_str("<|im_end|>\n");
                continue;
            }
            if msg.role == "assistant" {
                prompt.push_str("<|im_start|>assistant\n");
                let cleaned = strip_thought_blocks(&msg.content);
                let cleaned_trimmed = cleaned.trim();
                if !cleaned_trimmed.is_empty() {
                    prompt.push_str(cleaned_trimmed);
                } else if is_dense_heavy_tool_turn && msg.tool_calls.is_some() {
                    prompt.push_str("<think>\n\n</think>\n\n");
                }
                if let Some(ref tool_calls) = msg.tool_calls {
                    for (tc_i, tc) in tool_calls.iter().enumerate() {
                        if tc_i == 0 {
                            if !cleaned_trimmed.is_empty() || (is_dense_heavy_tool_turn && cleaned_trimmed.is_empty()) {
                                prompt.push_str("\n\n");
                            }
                        } else {
                            prompt.push('\n');
                        }
                        push_xml_function_tool_call(&mut prompt, tc);
                    }
                }
                prompt.push_str("<|im_end|>\n");
                i += 1;
                continue;
            }

            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content.trim()));
            i += 1;
        }

        prompt.push_str("<|im_start|>assistant\n");
        if !is_coder_non_thinking {
            if no_think {
                prompt.push_str("<think>\n\n</think>\n\n");
            } else {
                prompt.push_str("<think>\n");
            }
        }
        return prompt;
    }

    let mut prompt = String::new();

    let tools_array = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(val),
        _ => None,
    });

    let system_msg = messages.iter().find(|m| m.role == "system");
    let base_system = system_msg.map(|m| m.content.as_str());

    if let Some(tools_json) = tools_array {
        let clean_tools: Vec<&serde_json::Value> = match tools_json {
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|t| t.get("function").unwrap_or(t))
                .collect(),
            _ => vec![tools_json],
        };
        let tools_str = serde_json::to_string_pretty(&clean_tools).unwrap_or_default();
        prompt.push_str("<|im_start|>system\n");
        if let Some(sys) = base_system {
            prompt.push_str(sys);
            prompt.push_str("\n\n");
        }
        prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        prompt.push_str(&tools_str);
        prompt.push_str("\n</tools>\n\nTo call a function, reply with a JSON object within <tool_call></tool_call> tags:\n<tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"param_name\": \"param_value\"}}\n</tool_call>\n\nWhen providing the final answer to the user after tool execution, present your answer directly in markdown and STOP immediately. Do NOT call any further functions.<|im_end|>\n");
    } else if let Some(sys) = base_system {
        prompt.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }

    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        if msg.role == "system" {
            i += 1;
            continue;
        }
        if msg.role == "tool" {
            prompt.push_str("<|im_start|>user");
            while i < messages.len() && messages[i].role == "tool" {
                prompt.push_str("\n<tool_response>\n");
                prompt.push_str(&messages[i].content);
                prompt.push_str("\n</tool_response>");
                i += 1;
            }
            prompt.push_str("<|im_end|>\n");
            continue;
        }
        prompt.push_str(&format!("<|im_start|>{}\n", msg.role));
        if !msg.content.is_empty() {
            prompt.push_str(&msg.content);
            if !msg.content.ends_with('\n') {
                prompt.push('\n');
            }
        }
        if let Some(ref tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                let call_obj = serde_json::json!({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                });
                prompt.push_str(&format!(
                    "<tool_call>\n{}\n</tool_call>\n",
                    serde_json::to_string(&call_obj).unwrap_or_default()
                ));
            }
        }
        prompt.push_str("<|im_end|>\n");
        i += 1;
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Specialized prompt formatting for NVIDIA Nemotron-3 (30B MoE / ChatML XML) and Nemotron-Nano-v2.
fn format_nemotron_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
    hint_lower: &str,
) -> String {
    let tools_slice = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(arr.as_slice()),
        _ => None,
    });
    let no_think = messages
        .iter()
        .any(|m| (m.role == "user" || m.role == "system") && m.content.contains("/no_think"));

    // Nemotron-Nano-v2 (`<SPECIAL_10>` / `<TOOLCALL>[...]</TOOLCALL>`)
    if hint_lower.contains("<special_10>") || hint_lower.contains("<toolcall>") {
        let mut prompt = String::new();
        let sys_content = messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.replace("/think", "").replace("/no_think", ""))
            .unwrap_or_default();
        let sys_trimmed = sys_content.trim();

        prompt.push_str("<SPECIAL_10>System\n");
        if !sys_trimmed.is_empty() {
            prompt.push_str(sys_trimmed);
        }
        if let Some(arr) = tools_slice {
            if !sys_trimmed.is_empty() {
                prompt.push_str("\n\n");
            }
            let clean_tools: Vec<&serde_json::Value> = arr
                .iter()
                .map(|t| t.get("function").unwrap_or(t))
                .collect();
            prompt.push_str("You can use the following tools to assist the user if required:\n<AVAILABLE_TOOLS>");
            prompt.push_str(&serde_json::to_string(&clean_tools).unwrap_or_else(|_| "[]".into()));
            prompt.push_str("</AVAILABLE_TOOLS>\n\nIf you decide to call any tool(s), use the following format:\n<TOOLCALL>[{\"name\": \"tool_name1\", \"arguments\": {\"arg\": \"val\"}}]</TOOLCALL>\n\nThe user will execute tool-calls and return responses from tool(s) in this format:\n<TOOL_RESPONSE>[{\"tool_response1\"}]</TOOL_RESPONSE>\n\nBased on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user.");
        }
        prompt.push_str("\n\n");

        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            if msg.role == "system" {
                i += 1;
                continue;
            }
            if msg.role == "tool" {
                prompt.push_str("<SPECIAL_11>User\n<TOOL_RESPONSE>[");
                let mut first = true;
                while i < messages.len() && messages[i].role == "tool" {
                    if !first {
                        prompt.push_str(", ");
                    }
                    first = false;
                    prompt.push_str(&messages[i].content);
                    i += 1;
                }
                prompt.push_str("]</TOOL_RESPONSE>");
                continue;
            }
            if msg.role == "assistant" {
                let cleaned = strip_thought_blocks(&msg.content);
                let cleaned_trimmed = cleaned.trim();
                prompt.push_str("<SPECIAL_11>Assistant\n");
                prompt.push_str(cleaned_trimmed);
                if let Some(ref tool_calls) = msg.tool_calls {
                    if !cleaned_trimmed.is_empty() {
                        prompt.push('\n');
                    }
                    let arr: Vec<serde_json::Value> = tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            })
                        })
                        .collect();
                    prompt.push_str("<TOOLCALL>");
                    prompt.push_str(&serde_json::to_string(&arr).unwrap_or_else(|_| "[]".into()));
                    prompt.push_str("</TOOLCALL>");
                }
                prompt.push_str("\n<SPECIAL_12>\n");
                i += 1;
                continue;
            }
            let clean_user = msg.content.replace("/think", "").replace("/no_think", "");
            prompt.push_str("<SPECIAL_11>User\n");
            prompt.push_str(clean_user.trim());
            prompt.push('\n');
            i += 1;
        }

        prompt.push_str("<SPECIAL_11>Assistant\n");
        if no_think {
            prompt.push_str("<think></think>");
        } else {
            prompt.push_str("<think>\n");
        }
        return prompt;
    }

    // Default Nemotron-3 30B format (matches `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja`)
    let mut prompt = String::new();
    let sys_msg = messages.iter().find(|m| m.role == "system");
    let base_system = sys_msg.map(|m| m.content.as_str()).unwrap_or("");

    if sys_msg.is_some() || tools_slice.is_some() {
        prompt.push_str("<|im_start|>system\n");
        if !base_system.is_empty() {
            prompt.push_str(base_system);
        }
        if let Some(arr) = tools_slice {
            if !base_system.is_empty() {
                prompt.push_str("\n\n");
            }
            prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n");
            push_xml_tool_definitions(&mut prompt, arr);
            prompt.push_str("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
        }
        prompt.push_str("<|im_end|>\n");
    }

    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        if msg.role == "system" {
            i += 1;
            continue;
        }
        if msg.role == "tool" {
            prompt.push_str("<|im_start|>user\n");
            while i < messages.len() && messages[i].role == "tool" {
                prompt.push_str("<tool_response>\n");
                prompt.push_str(&messages[i].content);
                prompt.push_str("\n</tool_response>\n");
                i += 1;
            }
            prompt.push_str("<|im_end|>\n");
            continue;
        }
        if msg.role == "assistant" {
            let cleaned = strip_thought_blocks(&msg.content);
            let cleaned_trimmed = cleaned.trim();
            prompt.push_str("<|im_start|>assistant\n");
            if let Some(ref tool_calls) = msg.tool_calls {
                if !tool_calls.is_empty() {
                    prompt.push_str("<think></think>");
                    if !cleaned_trimmed.is_empty() {
                        prompt.push_str(cleaned_trimmed);
                        prompt.push('\n');
                    }
                    for tc in tool_calls {
                        push_xml_function_tool_call(&mut prompt, tc);
                        prompt.push('\n');
                    }
                    prompt.push_str("<|im_end|>\n");
                    i += 1;
                    continue;
                }
            }
            if !cleaned_trimmed.is_empty() {
                prompt.push_str("<think></think>");
                prompt.push_str(cleaned_trimmed);
            }
            prompt.push_str("<|im_end|>\n");
            i += 1;
            continue;
        }
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        i += 1;
    }

    if no_think {
        prompt.push_str("<|im_start|>assistant\n<think></think>");
    } else {
        prompt.push_str("<|im_start|>assistant\n<think>\n");
    }
    prompt
}

/// Specialized prompt formatting for IBM Granite 3.x / 4.x models.
fn format_granite_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
) -> String {
    let mut prompt = String::new();

    let tools_array = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(val),
        _ => None,
    });

    let system_msg = messages.iter().find(|m| m.role == "system");
    let base_system = system_msg.map(|m| m.content.as_str());

    if let Some(tools_json) = tools_array {
        let clean_tools: Vec<&serde_json::Value> = match tools_json {
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|t| t.get("function").unwrap_or(t))
                .collect(),
            _ => vec![tools_json],
        };
        let tools_str = serde_json::to_string_pretty(&clean_tools).unwrap_or_default();
        prompt.push_str("<|start_of_role|>system<|end_of_role|>");
        if let Some(sys) = base_system {
            prompt.push_str(sys);
            prompt.push_str("\n\n");
        }
        prompt.push_str("You are a helpful assistant with access to the following tools:\n");
        prompt.push_str(&tools_str);
        prompt.push_str("\n\nTo call a tool, respond strictly with:\n<|tool_call|>[{\"name\": \"function_name\", \"arguments\": {\"arg\": \"val\"}}]\n\nWhen you have received the tool results and are answering the user, summarize the answer directly in markdown and STOP immediately.<|end_of_text|>\n");
    } else if let Some(sys) = base_system {
        prompt.push_str(&format!("<|start_of_role|>system<|end_of_role|>{sys}<|end_of_text|>\n"));
    }

    for msg in messages {
        if msg.role == "system" {
            continue;
        }

        match msg.role.as_str() {
            "tool" => {
                prompt.push_str("<|start_of_role|>tool_response<|end_of_role|>");
                prompt.push_str(&msg.content);
                prompt.push_str("<|end_of_text|>\n");
            }
            "assistant" => {
                prompt.push_str("<|start_of_role|>assistant<|end_of_role|>");
                if let Some(ref tool_calls) = msg.tool_calls {
                    let calls_arr: Vec<serde_json::Value> = tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            })
                        })
                        .collect();
                    prompt.push_str(&format!(
                        "<|tool_call|>{}\n",
                        serde_json::to_string(&calls_arr).unwrap_or_default()
                    ));
                }
                if !msg.content.is_empty() {
                    prompt.push_str(&msg.content);
                }
                prompt.push_str("<|end_of_text|>\n");
            }
            "user" => {
                prompt.push_str("<|start_of_role|>user<|end_of_role|>");
                prompt.push_str(&msg.content);
                prompt.push_str("<|end_of_text|>\n");
            }
            other => {
                prompt.push_str(&format!(
                    "<|start_of_role|>{other}<|end_of_role|>{}<|end_of_text|>\n",
                    msg.content
                ));
            }
        }
    }

    prompt.push_str("<|start_of_role|>assistant<|end_of_role|>");
    prompt
}

/// Specialized prompt formatting for GLM-4 / GLM-4 MoE Lite models.
fn format_glm_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
) -> String {
    let mut prompt = String::new();
    prompt.push_str("[gMASK]<sop>");

    let system_msg = messages.iter().find(|m| m.role == "system");
    let base_system = system_msg.map(|m| m.content.as_str());

    let tools_array = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(val),
        _ => None,
    });

    if let Some(tools_json) = tools_array {
        let clean_tools: Vec<&serde_json::Value> = match tools_json {
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|t| t.get("function").unwrap_or(t))
                .collect(),
            _ => vec![tools_json],
        };
        let tools_str = serde_json::to_string_pretty(&clean_tools).unwrap_or_default();
        prompt.push_str("<|system|>\n");
        if let Some(sys) = base_system {
            prompt.push_str(sys);
            prompt.push_str("\n\n");
        }
        prompt.push_str("# Tools\n\nYou have access to the following functions:\n```json\n");
        prompt.push_str(&tools_str);
        prompt.push_str("\n```\n\nWhen you need to call a function, reply with:\n<tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"arg_name\": \"arg_value\"}}\n</tool_call>\n\nWhen you receive tool responses, summarize and answer the user query directly. Do NOT repeat previous tool calls.");
    } else if let Some(sys) = base_system {
        prompt.push_str(&format!("<|system|>\n{sys}"));
    }

    for msg in messages {
        if msg.role == "system" {
            continue;
        }
        if msg.role == "tool" {
            prompt.push_str("<|observation|>\n");
            prompt.push_str(&msg.content);
            continue;
        }
        match msg.role.as_str() {
            "user" => {
                prompt.push_str(&format!("<|user|>\n{}", msg.content));
            }
            "assistant" => {
                prompt.push_str("<|assistant|>\n");
                if !msg.content.is_empty() {
                    prompt.push_str(&msg.content);
                }
                if let Some(ref tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        let call_obj = serde_json::json!({
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        });
                        prompt.push_str(&format!(
                            "<tool_call>\n{}\n</tool_call>",
                            serde_json::to_string(&call_obj).unwrap_or_default()
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Format a JSON value into Gemma 4's `<|"|>...<|"|>` argument/schema literal syntax.
fn format_gemma4_arg(val: &serde_json::Value, escape_keys: bool) -> String {
    match val {
        serde_json::Value::String(s) => format!("<|\"|>{s}<|\"|>"),
        serde_json::Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Null => "null".into(),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr
                .iter()
                .map(|item| format_gemma4_arg(item, escape_keys))
                .collect();
            format!("[{}]", items.join(","))
        }
        serde_json::Value::Object(map) => {
            // Sort keys alphabetically (`dictsort` in `google-gemma-4-31B-it.jinja`)
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut parts = Vec::with_capacity(keys.len());
            for k in keys {
                let v = &map[k];
                let key_str = if escape_keys {
                    format!("<|\"|>{k}<|\"|>")
                } else {
                    k.to_string()
                };
                parts.push(format!("{key_str}:{}", format_gemma4_arg(v, escape_keys)));
            }
            format!("{{{}}}", parts.join(","))
        }
    }
}

/// Extract uppercased Gemma 4 schema type + nullable flag (handles `"string"` and `["string", "null"]`).
fn extract_gemma4_schema_type(val: &serde_json::Value) -> (String, bool) {
    let mut nullable = val
        .get("nullable")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let ty = match val.get("type") {
        Some(serde_json::Value::String(s)) => s.to_uppercase(),
        Some(serde_json::Value::Array(arr)) => {
            let mut chosen = String::new();
            for item in arr {
                if let Some(s) = item.as_str() {
                    if s.eq_ignore_ascii_case("null") {
                        nullable = true;
                    } else if chosen.is_empty() {
                        chosen = s.to_uppercase();
                    }
                }
            }
            if chosen.is_empty() { "STRING".into() } else { chosen }
        }
        _ => {
            if val.get("properties").is_some() {
                "OBJECT".into()
            } else if val.get("items").is_some() {
                "ARRAY".into()
            } else {
                "STRING".into()
            }
        }
    };
    (ty, nullable)
}

fn format_gemma4_properties(props: &serde_json::Map<String, serde_json::Value>) -> String {
    let standard_keys = ["description", "type", "properties", "required", "nullable"];
    let mut keys: Vec<&String> = props
        .keys()
        .filter(|k| !standard_keys.contains(&k.as_str()))
        .collect();
    keys.sort();

    let mut rendered = Vec::with_capacity(keys.len());
    for key in keys {
        let val = &props[key];
        let Some(obj) = val.as_object() else {
            continue;
        };
        let (ty_upper, nullable) = extract_gemma4_schema_type(val);
        let mut fields = Vec::new();

        if let Some(desc) = obj.get("description").and_then(|d| d.as_str()) {
            if !desc.is_empty() {
                fields.push(format!("description:<|\"|>{desc}<|\"|>"));
            }
        }
        if nullable {
            fields.push("nullable:true".into());
        }

        match ty_upper.as_str() {
            "STRING" => {
                if let Some(en) = obj.get("enum") {
                    fields.push(format!("enum:{}", format_gemma4_arg(en, true)));
                }
            }
            "OBJECT" => {
                let inner = obj
                    .get("properties")
                    .and_then(|p| p.as_object())
                    .map(format_gemma4_properties)
                    .unwrap_or_default();
                fields.push(format!("properties:{{{inner}}}"));
                if let Some(req) = obj.get("required").and_then(|r| r.as_array()) {
                    if !req.is_empty() {
                        let req_items: Vec<String> = req
                            .iter()
                            .filter_map(|r| r.as_str().map(|s| format!("<|\"|>{s}<|\"|>")))
                            .collect();
                        fields.push(format!("required:[{}]", req_items.join(",")));
                    }
                }
            }
            "ARRAY" => {
                if let Some(items_obj) = obj.get("items").and_then(|it| it.as_object()) {
                    if !items_obj.is_empty() {
                        let mut item_keys: Vec<&String> = items_obj.keys().collect();
                        item_keys.sort();
                        let mut item_parts = Vec::new();
                        for ik in item_keys {
                            let iv = &items_obj[ik];
                            if iv.is_null() {
                                continue;
                            }
                            if ik == "properties" {
                                let inner = iv
                                    .as_object()
                                    .map(format_gemma4_properties)
                                    .unwrap_or_default();
                                item_parts.push(format!("properties:{{{inner}}}"));
                            } else if ik == "required" {
                                if let Some(req) = iv.as_array() {
                                    let req_items: Vec<String> = req
                                        .iter()
                                        .filter_map(|r| r.as_str().map(|s| format!("<|\"|>{s}<|\"|>")))
                                        .collect();
                                    item_parts.push(format!("required:[{}]", req_items.join(",")));
                                }
                            } else if ik == "type" {
                                if let Some(ts) = iv.as_str() {
                                    item_parts.push(format!("type:<|\"|>{}<|\"|>", ts.to_uppercase()));
                                } else {
                                    item_parts.push(format!("type:{}", format_gemma4_arg(iv, true)));
                                }
                            } else {
                                item_parts.push(format!("{ik}:{}", format_gemma4_arg(iv, true)));
                            }
                        }
                        fields.push(format!("items:{{{}}}", item_parts.join(",")));
                    }
                }
            }
            _ => {}
        }

        fields.push(format!("type:<|\"|>{ty_upper}<|\"|>"));
        rendered.push(format!("{key}:{{{}}}", fields.join(",")));
    }

    rendered.join(",")
}

fn format_gemma4_tool_declaration(tool: &serde_json::Value) -> String {
    let func = tool.get("function").unwrap_or(tool);
    let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
    let desc = func.get("description").and_then(|d| d.as_str()).unwrap_or("");
    let mut out = format!("declaration:{name}{{description:<|\"|>{desc}<|\"|>");

    if let Some(params) = func.get("parameters").and_then(|p| p.as_object()) {
        if !params.is_empty() {
            out.push_str(",parameters:{");
            if let Some(props) = params.get("properties").and_then(|p| p.as_object()) {
                if !props.is_empty() {
                    out.push_str("properties:{");
                    out.push_str(&format_gemma4_properties(props));
                    out.push_str("},");
                }
            }
            if let Some(req) = params.get("required").and_then(|r| r.as_array()) {
                if !req.is_empty() {
                    let req_items: Vec<String> = req
                        .iter()
                        .filter_map(|r| r.as_str().map(|s| format!("<|\"|>{s}<|\"|>")))
                        .collect();
                    out.push_str("required:[");
                    out.push_str(&req_items.join(","));
                    out.push_str("],");
                }
            }
            let ptype = params
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("OBJECT")
                .to_uppercase();
            out.push_str(&format!("type:<|\"|>{ptype}<|\"|>}}"));
        }
    }

    out.push('}');
    out
}

/// Specialized prompt formatting for Gemma 4 (26B MoE / 31B / E2B / E4B) matching
/// `google-gemma-4-31B-it.jinja` and `llama.cpp`'s `gemma4_model_turn_builder`.
fn format_gemma4_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
) -> String {
    let mut prompt = String::new();
    let tools_slice = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(arr.as_slice()),
        _ => None,
    });

    let no_think = messages.iter().any(|m| {
        (m.role == "user" || m.role == "system" || m.role == "developer")
            && m.content.contains("/no_think")
    });
    let enable_thinking = !no_think;

    let first_is_sys = messages
        .first()
        .map(|m| m.role == "system" || m.role == "developer")
        .unwrap_or(false);

    if enable_thinking || tools_slice.is_some() || first_is_sys {
        prompt.push_str("<|turn>system\n");
        if enable_thinking {
            prompt.push_str("<|think|>");
        }
        if first_is_sys {
            let sys_clean = messages[0].content.replace("/no_think", "");
            prompt.push_str(sys_clean.trim());
        }
        if let Some(arr) = tools_slice {
            for t in arr {
                prompt.push_str("<|tool>");
                prompt.push_str(&format_gemma4_tool_declaration(t));
                prompt.push_str("<tool|>");
            }
        }
        prompt.push_str("<turn|>\n");
    }

    let mut i = if first_is_sys { 1 } else { 0 };
    let mut trailing_open_tool_response = false;

    while i < messages.len() {
        let msg = &messages[i];
        trailing_open_tool_response = false;

        if msg.role == "assistant" {
            if let Some(ref tcs) = msg.tool_calls {
                if !tcs.is_empty() {
                    prompt.push_str("<|turn>model\n");
                    for tc in tcs {
                        prompt.push_str("<|tool_call>call:");
                        prompt.push_str(&tc.function.name);
                        if let Some(obj) = tc.function.arguments.as_object() {
                            let mut keys: Vec<&String> = obj.keys().collect();
                            keys.sort();
                            prompt.push('{');
                            for (ki, k) in keys.iter().enumerate() {
                                if ki > 0 {
                                    prompt.push(',');
                                }
                                prompt.push_str(k);
                                prompt.push(':');
                                prompt.push_str(&format_gemma4_arg(&obj[*k], false));
                            }
                            prompt.push('}');
                        } else {
                            prompt.push_str("{}");
                        }
                        prompt.push_str("<tool_call|>");
                    }
                    i += 1;

                    // Collect consecutive `tool` responses belonging to this model turn
                    let mut resp_idx = 0usize;
                    let mut had_tool_responses = false;
                    while i < messages.len() && messages[i].role == "tool" {
                        had_tool_responses = true;
                        let tmsg = &messages[i];
                        let resolved_name = tmsg
                            .tool_name
                            .as_deref()
                            .filter(|s| !s.is_empty())
                            .or_else(|| {
                                tmsg.tool_call_id.as_deref().and_then(|cid| {
                                    tcs.iter()
                                        .find(|tc| tc.id.as_deref() == Some(cid))
                                        .map(|tc| tc.function.name.as_str())
                                })
                            })
                            .or_else(|| tcs.get(resp_idx).map(|tc| tc.function.name.as_str()))
                            .unwrap_or("unknown");

                        prompt.push_str("<|tool_response>response:");
                        prompt.push_str(resolved_name);
                        if let Ok(serde_json::Value::Object(map)) =
                            serde_json::from_str::<serde_json::Value>(&tmsg.content)
                        {
                            let mut keys: Vec<&String> = map.keys().collect();
                            keys.sort();
                            prompt.push('{');
                            for (ki, k) in keys.iter().enumerate() {
                                if ki > 0 {
                                    prompt.push(',');
                                }
                                prompt.push_str(k);
                                prompt.push(':');
                                prompt.push_str(&format_gemma4_arg(&map[*k], false));
                            }
                            prompt.push('}');
                        } else if let Ok(val) = serde_json::from_str::<serde_json::Value>(&tmsg.content) {
                            prompt.push_str("{value:");
                            prompt.push_str(&format_gemma4_arg(&val, false));
                            prompt.push('}');
                        } else {
                            prompt.push_str("{value:");
                            prompt.push_str(&format_gemma4_arg(
                                &serde_json::Value::String(tmsg.content.clone()),
                                false,
                            ));
                            prompt.push('}');
                        }
                        prompt.push_str("<tool_response|>");
                        resp_idx += 1;
                        i += 1;
                    }

                    // If the next message is the final assistant answer for this tool chain, fold it in
                    if i < messages.len()
                        && messages[i].role == "assistant"
                        && messages[i]
                            .tool_calls
                            .as_ref()
                            .map_or(true, |v| v.is_empty())
                        && !messages[i].content.trim().is_empty()
                    {
                        let cleaned = strip_thought_blocks(&messages[i].content);
                        prompt.push_str(cleaned.trim());
                        prompt.push_str("<turn|>\n");
                        i += 1;
                    } else if had_tool_responses && i == messages.len() {
                        // Leave `<|turn>model\n...<tool_response|>` open so Gemma 4 continues
                        // generating directly after the tool response (`google-gemma-4-31B-it.jinja` L254-261).
                        trailing_open_tool_response = true;
                    } else {
                        let cleaned = strip_thought_blocks(&msg.content);
                        if !cleaned.trim().is_empty() {
                            prompt.push_str(cleaned.trim());
                        }
                        prompt.push_str("<turn|>\n");
                    }
                    continue;
                }
            }

            // Regular assistant message without tool calls
            let cleaned = strip_thought_blocks(&msg.content);
            prompt.push_str("<|turn>model\n");
            prompt.push_str(cleaned.trim());
            prompt.push_str("<turn|>\n");
            i += 1;
            continue;
        }

        if msg.role == "tool" {
            // Orphan tool response fallback
            let rname = msg.tool_name.as_deref().unwrap_or("unknown");
            prompt.push_str("<|turn>model\n<|tool_response>response:");
            prompt.push_str(rname);
            prompt.push_str("{value:");
            prompt.push_str(&format_gemma4_arg(
                &serde_json::Value::String(msg.content.clone()),
                false,
            ));
            prompt.push_str("}<tool_response|>");
            i += 1;
            if i == messages.len() {
                trailing_open_tool_response = true;
            } else {
                prompt.push_str("<turn|>\n");
            }
            continue;
        }

        let role = if msg.role == "developer" { "system" } else { msg.role.as_str() };
        prompt.push_str("<|turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(msg.content.trim());
        prompt.push_str("<turn|>\n");
        i += 1;
    }

    if !trailing_open_tool_response {
        prompt.push_str("<|turn>model\n");
    }
    if !enable_thinking {
        prompt.push_str("<|channel>thought\n<channel|>");
    }
    prompt
}

/// Specialized prompt formatting for Gemma 2 / 3 models.
fn format_gemma_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
) -> String {
    let mut prompt = String::new();

    let tools_array = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(val),
        _ => None,
    });

    let system_msg = messages.iter().find(|m| m.role == "system");
    let base_system = system_msg.map(|m| m.content.as_str());

    let mut first_user = true;

    for msg in messages {
        if msg.role == "system" {
            continue;
        }

        if msg.role == "tool" {
            prompt.push_str("<start_of_turn>user\n<tool_response>\n");
            prompt.push_str(&msg.content);
            prompt.push_str("\n</tool_response><end_of_turn>\n");
            continue;
        }

        let role_name = if msg.role == "assistant" { "model" } else { "user" };
        prompt.push_str(&format!("<start_of_turn>{role_name}\n"));

        if role_name == "user" && first_user {
            if let Some(sys) = base_system {
                prompt.push_str(sys);
                prompt.push_str("\n\n");
            }
            if let Some(tools_json) = tools_array {
                let clean_tools: Vec<&serde_json::Value> = match tools_json {
                    serde_json::Value::Array(arr) => arr
                        .iter()
                        .map(|t| t.get("function").unwrap_or(t))
                        .collect(),
                    _ => vec![tools_json],
                };
                let tools_str = serde_json::to_string_pretty(&clean_tools).unwrap_or_default();
                prompt.push_str("# Available Tools\n\nYou have access to the following tools:\n\n<tools>\n");
                prompt.push_str(&tools_str);
                prompt.push_str("\n</tools>\n\nTo call a tool, reply in the following JSON format:\n```json\n{\n  \"name\": \"function_name\",\n  \"arguments\": {\"param_name\": \"param_value\"}\n}\n```\n\nWhen providing the final answer to the user, present your answer directly in markdown and do not call any further tools.\n\n");
            }
            first_user = false;
        }

        if !msg.content.is_empty() {
            prompt.push_str(&msg.content);
        }

        if let Some(ref tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                let call_obj = serde_json::json!({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                });
                prompt.push_str(&format!(
                    "\n```json\n{}\n```\n",
                    serde_json::to_string_pretty(&call_obj).unwrap_or_default()
                ));
            }
        }

        prompt.push_str("<end_of_turn>\n");
    }

    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Specialized prompt formatting for Mistral Small 3.x (24B/26B), Ministral 3, Devstral,
/// Magistral, Codestral, and legacy Mistral v0.3 / NeMo models.
fn format_mistral_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
    hint_lower: &str,
) -> String {
    let uses_call_id = hint_lower.contains("[call_id]")
        || hint_lower.contains("2506")
        || hint_lower.contains("2507")
        || hint_lower.contains("small-3.2")
        || hint_lower.contains("small_3.2")
        || hint_lower.contains("magistral");

    let uses_args_tag = uses_call_id
        || hint_lower.contains("[args]")
        || hint_lower.contains("[system_prompt]")
        || hint_lower.contains("mistral-small")
        || hint_lower.contains("mistral_small")
        || hint_lower.contains("ministral-3")
        || hint_lower.contains("ministral_3")
        || hint_lower.contains("devstral")
        || hint_lower.contains("24b")
        || hint_lower.contains("26b")
        || hint_lower.contains("2501")
        || hint_lower.contains("2503")
        || hint_lower.contains("2512");

    if uses_args_tag {
        let mut prompt = String::new();
        let tools_slice = tools.and_then(|val| match val {
            serde_json::Value::Array(arr) if !arr.is_empty() => Some(arr.as_slice()),
            _ => None,
        });

        let system_msg = messages.iter().find(|m| m.role == "system");
        if let Some(sys) = system_msg {
            prompt.push_str("[SYSTEM_PROMPT]");
            prompt.push_str(&sys.content);
            prompt.push_str("[/SYSTEM_PROMPT]");
        } else if hint_lower.contains("[think]") {
            prompt.push_str("[SYSTEM_PROMPT]First draft your thinking process inside [THINK]...[/THINK], then provide your response or tool call.[/SYSTEM_PROMPT]");
        } else {
            prompt.push_str("[SYSTEM_PROMPT]You are a helpful coding assistant. Use the available tools whenever needed to inspect files or perform actions accurately.[/SYSTEM_PROMPT]");
        }

        let tools_serialized = tools_slice.map(|arr| {
            let normalized: Vec<serde_json::Value> = arr
                .iter()
                .map(|t| {
                    if t.get("function").is_some() {
                        t.clone()
                    } else {
                        serde_json::json!({
                            "type": "function",
                            "function": t,
                        })
                    }
                })
                .collect();
            format!(
                "[AVAILABLE_TOOLS]{}[/AVAILABLE_TOOLS]",
                serde_json::to_string(&normalized).unwrap_or_else(|_| "[]".into())
            )
        });

        // Ministral-3 puts `[AVAILABLE_TOOLS]` right after `[/SYSTEM_PROMPT]`;
        // Mistral Small 3.2 puts `[AVAILABLE_TOOLS]` right before the last user `[INST]`.
        let last_user_idx = messages.iter().rposition(|m| m.role == "user");
        if !uses_call_id {
            if let Some(ref t_block) = tools_serialized {
                prompt.push_str(t_block);
            }
        }

        let mut last_call_ids: Vec<String> = Vec::new();
        let mut tool_resp_cursor: usize = 0;

        for (idx, msg) in messages.iter().enumerate() {
            if msg.role == "system" {
                continue;
            }
            match msg.role.as_str() {
                "user" => {
                    if uses_call_id && Some(idx) == last_user_idx {
                        if let Some(ref t_block) = tools_serialized {
                            prompt.push_str(t_block);
                        }
                    }
                    prompt.push_str("[INST]");
                    prompt.push_str(&msg.content);
                    prompt.push_str("[/INST]");
                }
                "assistant" => {
                    last_call_ids.clear();
                    tool_resp_cursor = 0;
                    if let Some(ref tool_calls) = msg.tool_calls {
                        if !tool_calls.is_empty() {
                            if !uses_call_id && !msg.content.trim().is_empty() {
                                prompt.push_str(&msg.content);
                            }
                            for (tc_i, tc) in tool_calls.iter().enumerate() {
                                prompt.push_str("[TOOL_CALLS]");
                                prompt.push_str(&tc.function.name);
                                if uses_call_id {
                                    let raw_id = tc
                                        .id
                                        .clone()
                                        .unwrap_or_else(|| format!("call{:05x}", tc_i + 1));
                                    let cid = normalize_mistral_call_id(&raw_id);
                                    last_call_ids.push(cid.clone());
                                    prompt.push_str("[CALL_ID]");
                                    prompt.push_str(&cid);
                                }
                                prompt.push_str("[ARGS]");
                                let args_str = serde_json::to_string(&tc.function.arguments)
                                    .unwrap_or_else(|_| "{}".into());
                                prompt.push_str(&args_str);
                            }
                            prompt.push_str("</s>");
                            continue;
                        }
                    }
                    prompt.push_str(&msg.content);
                    prompt.push_str("</s>");
                }
                "tool" => {
                    if uses_call_id {
                        let cid = msg
                            .tool_call_id
                            .as_deref()
                            .map(normalize_mistral_call_id)
                            .or_else(|| last_call_ids.get(tool_resp_cursor).cloned())
                            .unwrap_or_else(|| normalize_mistral_call_id("call00001"));
                        tool_resp_cursor += 1;
                        prompt.push_str("[TOOL_RESULTS]");
                        prompt.push_str(&cid);
                        prompt.push_str("[TOOL_CONTENT]");
                        prompt.push_str(&msg.content);
                        prompt.push_str("[/TOOL_RESULTS]");
                    } else {
                        prompt.push_str("[TOOL_RESULTS]");
                        prompt.push_str(&msg.content);
                        prompt.push_str("[/TOOL_RESULTS]");
                    }
                }
                _ => {}
            }
        }

        return prompt;
    }

    let mut prompt = String::new();

    let tools_array = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(val),
        _ => None,
    });

    let system_msg = messages.iter().find(|m| m.role == "system");
    let base_system = system_msg.map(|m| m.content.as_str());

    prompt.push_str("[INST] ");
    if let Some(sys) = base_system {
        prompt.push_str(sys);
        prompt.push_str("\n\n");
    }

    if let Some(tools_json) = tools_array {
        let clean_tools: Vec<&serde_json::Value> = match tools_json {
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|t| t.get("function").unwrap_or(t))
                .collect(),
            _ => vec![tools_json],
        };
        let tools_str = serde_json::to_string_pretty(&clean_tools).unwrap_or_default();
        prompt.push_str("[AVAILABLE_TOOLS] ");
        prompt.push_str(&tools_str);
        prompt.push_str(" [/AVAILABLE_TOOLS]\n\n");
        prompt.push_str("IMPORTANT RULES:\n1. If a function requires parameters that are unknown in the query, call the prerequisite lookup tool FIRST.\n2. NEVER invent or hallucinate coordinates or numerical parameters.\n3. Return function calls strictly inside [TOOL_CALLS] [{\"name\": \"...\", \"arguments\": {...}}] [/TOOL_CALLS]\n4. When you receive [TOOL_RESULTS], summarize the results and answer the user directly. Do NOT repeat previous tool calls.\n\n");
    }

    let mut inside_inst = true;

    for msg in messages {
        if msg.role == "system" {
            continue;
        }

        match msg.role.as_str() {
            "user" => {
                if !inside_inst {
                    prompt.push_str("[INST] ");
                }
                prompt.push_str(&msg.content);
                prompt.push_str(" [/INST]\n");
                inside_inst = false;
            }
            "tool" => {
                if !inside_inst {
                    prompt.push_str("[INST] ");
                }
                prompt.push_str("[TOOL_RESULTS] {\"content\": ");
                if msg.content.starts_with('{') || msg.content.starts_with('[') {
                    prompt.push_str(&msg.content);
                } else {
                    prompt.push_str(&serde_json::to_string(&msg.content).unwrap_or_default());
                }
                prompt.push_str("} [/TOOL_RESULTS] [/INST]\n");
                inside_inst = false;
            }
            "assistant" => {
                if inside_inst {
                    prompt.push_str(" [/INST]\n");
                    inside_inst = false;
                }
                if let Some(ref tool_calls) = msg.tool_calls {
                    let mut calls_arr = Vec::new();
                    for tc in tool_calls {
                        calls_arr.push(serde_json::json!({
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }));
                    }
                    prompt.push_str(&format!(
                        "[TOOL_CALLS] {} [/TOOL_CALLS]\n",
                        serde_json::to_string(&calls_arr).unwrap_or_default()
                    ));
                }
                if !msg.content.is_empty() {
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n");
                }
            }
            _ => {}
        }
    }

    if inside_inst {
        prompt.push_str(" [/INST]\n");
    }

    prompt
}

/// Strip reasoning / thought blocks (<think>...</think>, <thought>...</thought>, [THINK]...[/THINK], etc.)
/// before parsing tool calls.
fn strip_thought_blocks(text: &str) -> String {
    let mut result = text.to_string();

    // Handle prefilled thought blocks where the opening tag was part of the prompt
    // and only the closing tag (`</think>`, `[/THINK]`, `<channel|>`) appears in raw output.
    if !result.contains("<think>") {
        if let Some(end_idx) = result.find("</think>") {
            result.replace_range(..end_idx + "</think>".len(), "");
        }
    }
    if !result.contains("[THINK]") {
        if let Some(end_idx) = result.find("[/THINK]") {
            result.replace_range(..end_idx + "[/THINK]".len(), "");
        }
    }

    let thought_patterns = [
        ("<|channel>thought\n", "<channel|>"),
        ("<|channel>thought", "<channel|>"),
        ("<|channel|>thought\n", "<|end|>"),
        ("<|channel|>thought", "<|end|>"),
        ("<|channel|>thought\n", "<channel|>"),
        ("<|channel|>thought", "<channel|>"),
        ("<|channel|>analysis\n", "<|end|>"),
        ("<|channel|>analysis", "<|end|>"),
        ("<|thought>\n", "<thought|>"),
        ("<|thought>", "<thought|>"),
        ("<|thought>", "</thought|>"),
        ("<thought>", "</thought>"),
        ("<think>", "</think>"),
        ("[THINK]\n", "[/THINK]"),
        ("[THINK]", "[/THINK]"),
    ];

    for (start_tag, end_tag) in thought_patterns {
        while let Some(start_idx) = result.find(start_tag) {
            let after_start = &result[start_idx + start_tag.len()..];
            if let Some(end_idx) = after_start.find(end_tag) {
                let full_end_idx = start_idx + start_tag.len() + end_idx + end_tag.len();
                result.replace_range(start_idx..full_end_idx, "");
            } else {
                result.replace_range(start_idx..start_idx + start_tag.len(), "");
            }
        }
    }
    result
}

/// Parse tool calls and clean message content from the model's raw generated text.
pub fn parse_tool_calls(raw_output: &str) -> (String, Option<Vec<ToolCall>>) {
    let mut tool_calls = Vec::new();
    // 0. Clean up thought channels first: <think>...</think>, <thought>...</thought>, etc.
    // This prevents thinking deliberations/hypothetical examples inside <think> from being parsed as real tool calls.
    let mut cleaned_text = strip_thought_blocks(raw_output);

    // 1. Try parsing GPT-OSS / Harmony tool call syntax:
    // e.g. "to=functions.<name><|channel|>commentary<|message|>{...}<|call|>"
    // or "<|start|>assistant to=functions.<name>..."
    while let Some(to_idx) = cleaned_text.find("to=functions.") {
        let after_to = &cleaned_text[to_idx + "to=functions.".len()..];
        let name_end = after_to
            .find(|c: char| c == '<' || c == ' ' || c == '\n' || c == '{')
            .unwrap_or(after_to.len());
        let func_name = after_to[..name_end].trim().to_string();

        let remainder = &after_to[name_end..];
        let args_start_offset = if let Some(msg_idx) = remainder.find("<|message|>") {
            Some(msg_idx + "<|message|>".len())
        } else {
            remainder.find('{')
        };

        if !func_name.is_empty() {
            if let Some(args_offset) = args_start_offset {
                let args_text = &remainder[args_offset..];
                let end_tag_offset = args_text
                    .find("<|call|>")
                    .or_else(|| args_text.find("<|end|>"))
                    .or_else(|| args_text.find("<|return|>"));

                let (raw_args, full_match_len) = if let Some(end_o) = end_tag_offset {
                    (
                        &args_text[..end_o],
                        to_idx + "to=functions.".len() + name_end + args_offset + end_o + 8,
                    )
                } else if let Some(json_end) = find_matching_brace(args_text) {
                    (
                        &args_text[..json_end],
                        to_idx + "to=functions.".len() + name_end + args_offset + json_end,
                    )
                } else {
                    (args_text.trim(), cleaned_text.len())
                };

                if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw_args.trim()) {
                    if val.is_object() {
                        tool_calls.push(ToolCall {
                            id: Some(generate_tool_call_id()),
                            call_type: Some("function".into()),
                            function: FunctionCall {
                                name: func_name,
                                arguments: val,
                            },
                        });
                        let end_pos = full_match_len.min(cleaned_text.len());
                        cleaned_text.replace_range(to_idx..end_pos, "");
                        continue;
                    }
                }
            }
        }
        break;
    }

    // 2. Try parsing Gemma 4 / standard call syntax: <|tool_call>call:name{...}<tool_call|> or call:name{...}
    let gemma_markers = [
        ("<|tool_call>call:", "<tool_call|>"),
        ("<tool_call>call:", "</tool_call>"),
        ("<|tool_call>call:", "<|end|>"),
        ("<|call>call:", "<call|>"),
        ("<|call>", "<call|>"),
    ];

    for (start_tag, end_tag) in gemma_markers {
        while let Some(start_idx) = cleaned_text.find(start_tag) {
            let after_start = &cleaned_text[start_idx + start_tag.len()..];
            if let Some(end_idx) = after_start.find(end_tag) {
                let call_str = &after_start[..end_idx];
                if let Some((name, args_val)) = parse_gemma_call(call_str) {
                    tool_calls.push(ToolCall {
                        id: Some(generate_tool_call_id()),
                        call_type: Some("function".into()),
                        function: FunctionCall {
                            name,
                            arguments: args_val,
                        },
                    });
                }
                let full_end_idx = start_idx + start_tag.len() + end_idx + end_tag.len();
                cleaned_text.replace_range(start_idx..full_end_idx, "");
            } else {
                break;
            }
        }
    }

    // Fallback: search for un-tagged `call:func_name{...}` anywhere in text
    while let Some(call_idx) = cleaned_text.find("call:") {
        let after_call = &cleaned_text[call_idx + 5..];
        if let Some(end_brace) = find_matching_brace_gemma(after_call) {
            let call_str = &after_call[..end_brace];
            if let Some((name, args_val)) = parse_gemma_call(call_str) {
                tool_calls.push(ToolCall {
                    id: Some(generate_tool_call_id()),
                    call_type: Some("function".into()),
                    function: FunctionCall {
                        name,
                        arguments: args_val,
                    },
                });
                cleaned_text.replace_range(call_idx..call_idx + 5 + end_brace, "");
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // 3. Try parsing Qwen 3.5 / 3.8 & Nemotron-3 XML syntax:
    // <tool_call>
    // <function=name>
    // <parameter=key>value</parameter>
    // </function>
    // </tool_call>
    while let Some(start_idx) = cleaned_text.find("<function=") {
        let after_func = &cleaned_text[start_idx + "<function=".len()..];
        if let Some(func_name_end) = after_func.find('>') {
            let func_name = after_func[..func_name_end].trim().to_string();
            let remainder = &after_func[func_name_end + 1..];

            let func_end_idx = remainder.find("</function>").unwrap_or(remainder.len());
            let func_body = &remainder[..func_end_idx];

            let mut arguments = serde_json::Map::new();
            let mut param_search = func_body;
            while let Some(p_start) = param_search.find("<parameter=") {
                let after_p = &param_search[p_start + "<parameter=".len()..];
                if let Some(p_name_end) = after_p.find('>') {
                    let p_name = after_p[..p_name_end].trim().to_string();
                    let p_val_after = &after_p[p_name_end + 1..];
                    if let Some(p_val_end) = p_val_after.find("</parameter>") {
                        let p_val_slice = &p_val_after[..p_val_end];
                        let p_val_trimmed = p_val_slice.trim();
                        // Strip at most a single leading and trailing newline emitted by XML templates
                        // so multi-line code parameters preserve indentation.
                        let p_val_str = p_val_slice
                            .strip_prefix("\r\n")
                            .or_else(|| p_val_slice.strip_prefix('\n'))
                            .unwrap_or(p_val_slice);
                        let p_val_str = p_val_str
                            .strip_suffix("\r\n")
                            .or_else(|| p_val_str.strip_suffix('\n'))
                            .unwrap_or(p_val_str);

                        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(p_val_trimmed) {
                            arguments.insert(p_name, json_val);
                        } else {
                            arguments.insert(p_name, serde_json::Value::String(p_val_str.to_string()));
                        }
                        param_search = &p_val_after[p_val_end + "</parameter>".len()..];
                        continue;
                    }
                }
                break;
            }

            tool_calls.push(ToolCall {
                id: Some(generate_tool_call_id()),
                call_type: Some("function".into()),
                function: FunctionCall {
                    name: func_name,
                    arguments: serde_json::Value::Object(arguments),
                },
            });

            // Clean up surrounding tags if present
            let full_start = if let Some(tc_open) = cleaned_text[..start_idx].rfind("<tool_call>") {
                let between = &cleaned_text[tc_open + "<tool_call>".len()..start_idx];
                if between.trim().is_empty() {
                    tc_open
                } else {
                    start_idx
                }
            } else {
                start_idx
            };
            let end_offset = start_idx + "<function=".len() + func_name_end + 1 + func_end_idx + "</function>".len();
            let full_end = if let Some(tc_close) = cleaned_text[end_offset..].find("</tool_call>") {
                end_offset + tc_close + "</tool_call>".len()
            } else {
                end_offset
            };
            cleaned_text.replace_range(full_start..full_end.min(cleaned_text.len()), "");
        } else {
            break;
        }
    }

    let json_markers = [
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call|>", "<|tool_call|>"),
        ("<|tool_call>", "<tool_call|>"),
        ("<toolcall>", "</toolcall>"),
        ("<TOOLCALL>", "</TOOLCALL>"),
        ("```tool_call", "```"),
        ("[TOOL_CALLS]", "[/TOOL_CALLS]"),
    ];

    for (start_tag, end_tag) in json_markers {
        while let Some(start_idx) = cleaned_text.find(start_tag) {
            let after_start = &cleaned_text[start_idx + start_tag.len()..];
            let found_end = after_start.find(end_tag).map(|idx| (idx, end_tag.len())).or_else(|| {
                if start_tag == "<tool_call>" {
                    // Handle typos emitted by local models e.g. </tool__call> or unclosed tag with valid JSON
                    after_start.find("</tool__call>").map(|idx| (idx, "</tool__call>".len()))
                        .or_else(|| after_start.find("</toolcall>").map(|idx| (idx, "</toolcall>".len())))
                        .or_else(|| find_matching_brace(after_start).map(|idx| (idx, 0)))
                } else if start_tag == "<|tool_call|>" || start_tag == "<|tool_call>" {
                    after_start.find("<|im_end|>").map(|idx| (idx, 0))
                        .or_else(|| after_start.find("<|end_of_text|>").map(|idx| (idx, 0)))
                        .or_else(|| after_start.find("\n\n").map(|idx| (idx, 0)))
                } else {
                    None
                }
            });

            if let Some((end_idx, matched_end_len)) = found_end {
                let raw_inner = after_start[..end_idx].trim();
                // Preserve JSON arrays `[...]` (e.g. `<TOOLCALL>[...]</TOOLCALL>` or `<|tool_call|>[...]`),
                // otherwise strip corrupted trailing tags around `{...}` e.g. `</arg_value>`.
                let inner_clean = if raw_inner.starts_with('[') {
                    if let Some(bracket_len) = find_matching_bracket(raw_inner) {
                        &raw_inner[..bracket_len]
                    } else {
                        raw_inner
                    }
                } else if let Some(brace_start) = raw_inner.find('{') {
                    let from_brace = &raw_inner[brace_start..];
                    if let Some(brace_len) = find_matching_brace(from_brace) {
                        &from_brace[..brace_len]
                    } else {
                        raw_inner
                    }
                } else {
                    raw_inner
                };

                if let Some(tc) = parse_json_tool_call(inner_clean) {
                    tool_calls.push(tc);
                } else if let Ok(val) = serde_json::from_str::<serde_json::Value>(inner_clean) {
                    if let Some(arr) = val.as_array() {
                        for item in arr {
                            if let Some(tc) = parse_json_tool_call(&item.to_string()) {
                                tool_calls.push(tc);
                            }
                        }
                    }
                }
                let full_end_idx = start_idx + start_tag.len() + end_idx + matched_end_len;
                cleaned_text.replace_range(start_idx..full_end_idx, "");
            } else {
                break;
            }
        }
    }

    // 4. Try parsing Mistral `[TOOL_CALLS]` syntax:
    //    - `[TOOL_CALLS] [{"name": ..., "arguments": ...}]`
    //    - `[TOOL_CALLS] {"name": ..., "arguments": ...}`
    //    - `[TOOL_CALLS]func_name[ARGS]{"k": "v"}` (Ministral 3 / Devstral)
    //    - `[TOOL_CALLS]func_name[CALL_ID]abc123456[ARGS]{"k": "v"}` (Mistral Small 3.2 24B/26B)
    while let Some(tc_idx) = cleaned_text.find("[TOOL_CALLS]") {
        let after_tc = &cleaned_text[tc_idx + "[TOOL_CALLS]".len()..];
        let trimmed_after = after_tc.trim_start();
        let leading_ws = after_tc.len() - trimmed_after.len();

        if trimmed_after.starts_with('[') && !trimmed_after.starts_with("[ARGS]") && !trimmed_after.starts_with("[CALL_ID]") {
            if let Some(bracket_len) = find_matching_bracket(trimmed_after) {
                let candidate = &trimmed_after[..bracket_len];
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(candidate) {
                    if let Some(arr) = val.as_array() {
                        for item in arr {
                            if let Some(tc) = parse_json_tool_call(&item.to_string()) {
                                tool_calls.push(tc);
                            }
                        }
                    }
                }
                let full_end = tc_idx + "[TOOL_CALLS]".len() + leading_ws + bracket_len;
                cleaned_text.replace_range(tc_idx..full_end, "");
                continue;
            }
        } else if trimmed_after.starts_with('{') {
            if let Some(brace_len) = find_matching_brace(trimmed_after) {
                let candidate = &trimmed_after[..brace_len];
                if let Some(tc) = parse_json_tool_call(candidate) {
                    tool_calls.push(tc);
                }
                let full_end = tc_idx + "[TOOL_CALLS]".len() + leading_ws + brace_len;
                cleaned_text.replace_range(tc_idx..full_end, "");
                continue;
            }
        } else if let Some(args_rel) = trimmed_after.find("[ARGS]") {
            let next_tc_rel = trimmed_after.find("[TOOL_CALLS]").unwrap_or(usize::MAX);
            if args_rel < next_tc_rel {
                let header = trimmed_after[..args_rel].trim();
                let (func_name, explicit_id) = if let Some((fname, cid)) = header.split_once("[CALL_ID]") {
                    let cid_trimmed = cid.trim();
                    (
                        fname.trim(),
                        if cid_trimmed.is_empty() {
                            None
                        } else {
                            Some(cid_trimmed.to_string())
                        },
                    )
                } else {
                    (header, None)
                };

                let after_args = &trimmed_after[args_rel + "[ARGS]".len()..];
                let trimmed_args = after_args.trim_start();
                let args_ws = after_args.len() - trimmed_args.len();

                if !func_name.is_empty() && trimmed_args.starts_with('{') {
                    if let Some(brace_len) = find_matching_brace(trimmed_args) {
                        let args_json = &trimmed_args[..brace_len];
                        let parsed_args = serde_json::from_str::<serde_json::Value>(args_json)
                            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
                        tool_calls.push(ToolCall {
                            id: Some(explicit_id.unwrap_or_else(generate_tool_call_id)),
                            call_type: Some("function".into()),
                            function: FunctionCall {
                                name: func_name.to_string(),
                                arguments: parsed_args,
                            },
                        });
                        let full_end = tc_idx
                            + "[TOOL_CALLS]".len()
                            + leading_ws
                            + args_rel
                            + "[ARGS]".len()
                            + args_ws
                            + brace_len;
                        cleaned_text.replace_range(tc_idx..full_end, "");
                        continue;
                    }
                }
            }
        }
        break;
    }

    // 5. Try parsing raw JSON tool calls: arrays [ {...}, ... ] or individual objects { "name": ... }
    let mut search_idx = 0;
    while search_idx < cleaned_text.len() {
        if let Some(arr_start) = cleaned_text[search_idx..].find('[') {
            let actual_start = search_idx + arr_start;
            let from_bracket = &cleaned_text[actual_start..];
            if let Some(bracket_len) = find_matching_bracket(from_bracket) {
                let candidate = &from_bracket[..bracket_len];
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(candidate) {
                    if let Some(arr) = val.as_array() {
                        let mut parsed = Vec::new();
                        for item in arr {
                            if let Some(tc) = parse_json_tool_call(&item.to_string()) {
                                parsed.push(tc);
                            } else {
                                parsed.clear();
                                break;
                            }
                        }
                        if !parsed.is_empty() {
                            tool_calls.extend(parsed);
                            cleaned_text.replace_range(actual_start..actual_start + bracket_len, "");
                            continue;
                        }
                    }
                }
                search_idx = actual_start + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    let mut obj_search_idx = 0;
    while obj_search_idx < cleaned_text.len() {
        if let Some(brace_start) = cleaned_text[obj_search_idx..].find('{') {
            let actual_start = obj_search_idx + brace_start;
            let from_brace = &cleaned_text[actual_start..];
            if let Some(brace_len) = find_matching_brace(from_brace) {
                let candidate = &from_brace[..brace_len];
                if let Some(tc) = parse_json_tool_call(candidate) {
                    tool_calls.push(tc);
                    cleaned_text.replace_range(actual_start..actual_start + brace_len, "");
                    continue;
                }
                obj_search_idx = actual_start + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if !tool_calls.is_empty() {
        cleaned_text = strip_empty_code_fences(&cleaned_text);
    }

    // 6. Clean up thought channels and special control tags
    cleaned_text = strip_channel_tags(&cleaned_text);

    let final_tool_calls = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    let trimmed = cleaned_text.trim().to_string();

    (trimmed, final_tool_calls)
}

/// Strip empty markdown code fences (e.g. ```` ```json\n\n``` ````) left behind after extracting JSON tool calls.
fn strip_empty_code_fences(text: &str) -> String {
    let mut out = text.to_string();
    for opener in ["```tool_call", "```json", "```JSON", "```"] {
        loop {
            let Some(start) = out.find(opener) else {
                break;
            };
            let after = &out[start + opener.len()..];
            let Some(close_rel) = after.find("```") else {
                break;
            };
            if after[..close_rel].trim().is_empty() {
                let end = start + opener.len() + close_rel + 3;
                out.replace_range(start..end, "");
            } else {
                break;
            }
        }
    }
    out
}

/// Convert Gemma 4 `<|"|>...<|"|>` string literals into properly JSON-escaped `"..."` strings
/// while leaving surrounding structure intact.
fn gemma4_literals_to_json(input: &str) -> String {
    const DELIM: &str = "<|\"|>";
    if !input.contains(DELIM) {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len() + 16);
    let mut rest = input;
    while let Some(open_idx) = rest.find(DELIM) {
        out.push_str(&rest[..open_idx]);
        let after_open = &rest[open_idx + DELIM.len()..];
        if let Some(close_idx) = after_open.find(DELIM) {
            let raw_str = &after_open[..close_idx];
            if let Ok(escaped) = serde_json::to_string(raw_str) {
                out.push_str(&escaped);
            } else {
                out.push('"');
                out.push_str(&raw_str.replace('\\', "\\\\").replace('"', "\\\""));
                out.push('"');
            }
            rest = &after_open[close_idx + DELIM.len()..];
        } else {
            // Unclosed delimiter fallback
            if let Ok(escaped) = serde_json::to_string(after_open) {
                out.push_str(&escaped);
            }
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out
}

/// Find matching closing brace `}` accounting for both `"..."` and Gemma 4 `<|"|>...<|"|>` strings.
fn find_matching_brace_gemma(s: &str) -> Option<usize> {
    let normalized = gemma4_literals_to_json(s);
    if normalized == s {
        return find_matching_brace(s);
    }
    // Walk raw `s` directly while skipping `<|"|>...<|"|>` spans
    const DELIM: &str = "<|\"|>";
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut i = start;
    let mut depth = 0i32;
    let mut in_json_str = false;
    let mut escape = false;

    while i < bytes.len() {
        if !in_json_str && s[i..].starts_with(DELIM) {
            i += DELIM.len();
            if let Some(close_rel) = s[i..].find(DELIM) {
                i += close_rel + DELIM.len();
                continue;
            } else {
                return None;
            }
        }
        let c = s[i..].chars().next()?;
        let clen = c.len_utf8();
        if in_json_str {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_json_str = false;
            }
        } else if c == '"' {
            in_json_str = true;
        } else if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(i + clen);
            }
        }
        i += clen;
    }
    None
}

/// Helper: parse Gemma 4 `function_name{param:<|"|>val<|"|>}` or `function_name{param: "val"}`
fn parse_gemma_call(call_str: &str) -> Option<(String, serde_json::Value)> {
    let stripped = call_str.trim().strip_prefix("call:").unwrap_or(call_str.trim());
    let brace_idx = stripped.find('{')?;
    let func_name = stripped[..brace_idx].trim().to_string();
    if func_name.is_empty() {
        return None;
    }

    let args_slice = stripped[brace_idx..].trim();
    // Safely convert Gemma 4 `<|"|>...<|"|>` literals into RFC-8259 JSON strings
    // (properly escaping inner `"`, `\`, `\n`, etc.)
    let normalized = gemma4_literals_to_json(args_slice);

    // Attempt direct JSON parse
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&normalized) {
        if val.is_object() {
            return Some((func_name, val));
        }
    }

    // Quote unquoted keys (`{key: "val"}` -> `{"key": "val"}`) including nested objects
    let fixed_json = quote_unquoted_keys(&normalized);
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&fixed_json) {
        if val.is_object() {
            return Some((func_name, val));
        }
    }

    // Fallback: manual key-value parser for simple {key: "val", key2: 123}
    let manual_map = parse_simple_kv(&normalized);
    Some((func_name, serde_json::Value::Object(manual_map)))
}

/// Helper: parse a JSON block containing tool call object
fn parse_json_tool_call(json_str: &str) -> Option<ToolCall> {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
        // Direct object: {"name": "...", "arguments": ...}
        if let Some(name) = val.get("name").and_then(|n| n.as_str()) {
            let arguments = match val.get("arguments").or_else(|| val.get("parameters")) {
                Some(serde_json::Value::String(s)) => {
                    serde_json::from_str(s).unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
                }
                Some(obj @ serde_json::Value::Object(_)) => obj.clone(),
                _ => serde_json::Value::Object(serde_json::Map::new()),
            };
            return Some(ToolCall {
                id: Some(generate_tool_call_id()),
                call_type: Some("function".into()),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            });
        }

        // Nested in "function": {"function": {"name": "...", "arguments": ...}}
        if let Some(func_obj) = val.get("function") {
            if let Some(name) = func_obj.get("name").and_then(|n| n.as_str()) {
                let arguments = match func_obj.get("arguments").or_else(|| func_obj.get("parameters")) {
                    Some(serde_json::Value::String(s)) => {
                        serde_json::from_str(s).unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
                    }
                    Some(obj @ serde_json::Value::Object(_)) => obj.clone(),
                    _ => serde_json::Value::Object(serde_json::Map::new()),
                };
                return Some(ToolCall {
                    id: Some(generate_tool_call_id()),
                    call_type: Some("function".into()),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments,
                    },
                });
            }
        }
    }
    None
}

/// Convert unquoted keys `{foo: "bar"}` to `{"foo": "bar"}`
fn quote_unquoted_keys(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 16);
    let mut chars = input.chars().peekable();
    let mut in_string = false;
    let mut escape = false;

    while let Some(c) = chars.next() {
        if in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            out.push(c);
        } else {
            if c == '"' {
                in_string = true;
                out.push(c);
            } else if c.is_alphabetic() || c == '_' {
                // Potential unquoted key
                let mut ident = String::new();
                ident.push(c);
                while let Some(&next_c) = chars.peek() {
                    if next_c.is_alphanumeric() || next_c == '_' {
                        ident.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                // Check if followed by ':'
                let mut ws = String::new();
                while let Some(&next_c) = chars.peek() {
                    if next_c.is_whitespace() {
                        ws.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                if let Some(&':') = chars.peek() {
                    out.push('"');
                    out.push_str(&ident);
                    out.push('"');
                    out.push_str(&ws);
                } else {
                    out.push_str(&ident);
                    out.push_str(&ws);
                }
            } else {
                out.push(c);
            }
        }
    }
    out
}

/// Fallback manual key-value parser
fn parse_simple_kv(input: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    let trimmed = input.trim_matches(|c| c == '{' || c == '}');
    for pair in trimmed.split(',') {
        if let Some((k, v)) = pair.split_once(':') {
            let key = k.trim().trim_matches('"').to_string();
            let raw_val = v.trim().trim_matches('"');
            if let Ok(num) = raw_val.parse::<i64>() {
                map.insert(key, serde_json::Value::Number(num.into()));
            } else if let Ok(b) = raw_val.parse::<bool>() {
                map.insert(key, serde_json::Value::Bool(b));
            } else {
                map.insert(key, serde_json::Value::String(raw_val.to_string()));
            }
        }
    }
    map
}

/// Helper: find matching closing bracket `]` accounting for strings and nesting
fn find_matching_bracket(s: &str) -> Option<usize> {
    let start = s.find('[')?;
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;

    for (i, c) in s[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if c == '[' {
            depth += 1;
        } else if c == ']' {
            depth -= 1;
            if depth == 0 {
                return Some(start + i + 1);
            }
        }
    }
    None
}

/// Helper: find matching closing brace `}` accounting for strings and nesting
fn find_matching_brace(s: &str) -> Option<usize> {
    let start = s.find('{')?;
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;

    for (i, c) in s[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(start + i + 1);
            }
        }
    }
    None
}

/// Strip thought channels and special control tags from model output
fn strip_channel_tags(text: &str) -> String {
    let mut result = text.to_string();

    let channel_patterns = [
        ("to=self<|message|>", "<|eom|>"),
        ("to=self<|message|>", "<|eot|>"),
        ("to=self<|message|>", "<|start|>"),
        ("<|start|>assistant to=self<|message|>", "<|eom|>"),
        ("<|start|>assistant to=self<|message|>", "<|eot|>"),
        ("<|start|>assistant to=self<|message|>", "<|start|>"),
        ("<|channel>thought\n", "<channel|>"),
        ("<|channel>thought", "<channel|>"),
        ("<|channel|>thought\n", "<|end|>"),
        ("<|channel|>thought", "<|end|>"),
        ("<|channel|>thought\n", "<channel|>"),
        ("<|channel|>thought", "<channel|>"),
        ("<|channel|>analysis\n", "<|end|>"),
        ("<|channel|>analysis", "<|end|>"),
        ("<|channel|>commentary\n", "<|end|>"),
        ("<|channel|>commentary", "<|end|>"),
        ("<|thought>\n", "<thought|>"),
        ("<|thought>", "<thought|>"),
        ("<|thought>", "</thought|>"),
        ("<thought>", "</thought>"),
        ("<think>", "</think>"),
        ("[THINK]\n", "[/THINK]"),
        ("[THINK]", "[/THINK]"),
    ];

    for (start_tag, end_tag) in channel_patterns {
        while let Some(start_idx) = result.find(start_tag) {
            let after_start = &result[start_idx + start_tag.len()..];
            if let Some(end_idx) = after_start.find(end_tag) {
                let full_end_idx = start_idx + start_tag.len() + end_idx + end_tag.len();
                result.replace_range(start_idx..full_end_idx, "");
            } else {
                result.replace_range(start_idx..start_idx + start_tag.len(), "");
            }
        }
    }

    // Strip remaining to=<recipient><|message|> or to=<recipient>\n headers
    while let Some(to_idx) = result.find("to=") {
        let after_to = &result[to_idx..];
        if let Some(msg_idx) = after_to.find("<|message|>") {
            let full_end = to_idx + msg_idx + "<|message|>".len();
            result.replace_range(to_idx..full_end, "");
        } else if let Some(nl_idx) = after_to.find('\n') {
            let full_end = to_idx + nl_idx + 1;
            result.replace_range(to_idx..full_end, "");
        } else {
            result.replace_range(to_idx..result.len(), "");
        }
    }

    // Clean stray tokens
    let stray_tokens = [
        "<|channel|>final<|message|>",
        "<|channel|>final",
        "<|channel>thought",
        "<channel|>",
        "<|thought>",
        "<thought|>",
        "</thought|>",
        "<turn|>",
        "<|turn|>",
        "<start_of_turn>model",
        "<start_of_turn>user",
        "<start_of_turn>",
        "<end_of_turn>",
        "<|start|>assistant",
        "<|start|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<call|>",
        "<|message|>",
        "<|header_start|>assistant<|header_end|>",
        "<|header_start|>",
        "<|header_end|>",
        "<|eot|>",
        "<|im_start|>assistant",
        "<|im_start|>",
        "<|im_end|>",
        "<SPECIAL_10>",
        "<SPECIAL_11>",
        "<SPECIAL_12>",
        "<|start_of_role|>system<|end_of_role|>",
        "<|start_of_role|>assistant<|end_of_role|>",
        "<|start_of_role|>user<|end_of_role|>",
        "<|start_of_role|>tool_response<|end_of_role|>",
        "<|start_of_role|>tools<|end_of_role|>",
        "<|start_of_role|>",
        "<|end_of_role|>",
        "<|end_of_text|>",
        "<|end_of_turn|>",
        "to=user",
        "to=self",
        "<atem:function_calls>",
        "</atem:function_calls>",
        "[SYSTEM_PROMPT]",
        "[/SYSTEM_PROMPT]",
        "[INST]",
        "[/INST]",
        "[AVAILABLE_TOOLS]",
        "[/AVAILABLE_TOOLS]",
        "[TOOL_RESULTS]",
        "[/TOOL_RESULTS]",
        "[TOOL_CONTENT]",
        "[THINK]",
        "[/THINK]",
        "<s>",
        "</s>",
    ];

    for tok in stray_tokens {
        result = result.replace(tok, "");
    }

    result
}









#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gptoss_tool_call() {
        let raw = "<|channel|>thought\nNeed station data\n<|end|><|start|>assistant to=functions.get_station_data<|channel|>commentary<|message|>{\"station_id\": 104, \"sensor\": \"temperature\"}<|call|>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_station_data");
        assert_eq!(
            tcs[0].function.arguments["station_id"],
            serde_json::json!(104)
        );
        assert_eq!(
            tcs[0].function.arguments["sensor"],
            serde_json::json!("temperature")
        );
    }

    #[test]
    fn test_parse_gemma4_tool_call() {
        let raw = "<|channel>thought\nNeed to get weather for Paris\n<channel|><|tool_call>call:get_current_weather{location:<|\"|>Paris<|\"|>,unit:<|\"|>celsius<|\"|>}<tool_call|>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_current_weather");
        assert_eq!(
            tcs[0].function.arguments["location"],
            serde_json::json!("Paris")
        );
        assert_eq!(
            tcs[0].function.arguments["unit"],
            serde_json::json!("celsius")
        );
    }

    #[test]
    fn test_parse_special_pipe_tool_call() {
        let raw = "<|tool_call|>\n{\"name\": \"getgroupid\", \"arguments\": {}}<|im_end|>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "getgroupid");
    }

    #[test]
    fn test_parse_corrupted_closing_tag() {
        let raw = "<tool_call>\n{\"name\": \"getgroupid\", \"arguments\": {}}\n</tool__call>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "getgroupid");
    }

    #[test]
    fn test_parse_json_tool_call() {
        let raw = "Let me check that for you.\n<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}\n</tool_call>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "Let me check that for you.");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs[0].function.name, "calculator");
        assert_eq!(tcs[0].function.arguments["expression"], serde_json::json!("2+2"));
    }

    #[test]
    fn test_parse_mistral_tool_call() {
        let raw = "[TOOL_CALLS] [{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Tokyo\"}}]";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_current_weather");
        assert_eq!(tcs[0].function.arguments["location"], serde_json::json!("Tokyo"));
    }

    #[test]
    fn test_parse_qwen38_xml_tool_call() {
        let raw = "<think>\nThinking about weather...\n</think>\n<tool_call>\n<function=get_forecast>\n<parameter=location>\nGraz, Austria\n</parameter>\n<parameter=hours>\n12\n</parameter>\n</function>\n</tool_call>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_forecast");
        assert_eq!(tcs[0].function.arguments["location"], serde_json::json!("Graz, Austria"));
        assert_eq!(tcs[0].function.arguments["hours"], serde_json::json!(12));
    }

    #[test]
    fn test_parse_tool_call_with_thinking_cross_contamination() {
        let raw = "<think>\nWait, the prompt shows \n<tool_call>\n<function=get_devices_around_position>... twice? ... I'll just make one tool call properly: <function=get_coordinates><parameter=location>Göttingen</parameter></function>\n</think>\n\n<tool_call>\n<function=get_coordinates>\n<parameter=location>\nGöttingen\n</parameter>\n</function>\n</tool_call>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_coordinates");
        assert_eq!(tcs[0].function.arguments["location"], serde_json::json!("Göttingen"));
    }

    #[test]
    fn test_format_chat_prompt_with_tools() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "What is 2+2?".into(),
            ..Default::default()
        }];
        let tools = serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Evaluate math expression"
                }
            }
        ]);
        let prompt = format_chat_prompt(&messages, Some(&tools), None);
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("\"name\": \"calc\""));
        assert!(prompt.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"));

        let gpt_prompt = format_chat_prompt(&messages, Some(&tools), Some("gptoss"));
        assert!(gpt_prompt.contains("<|start|>system<|message|>"));
        assert!(gpt_prompt.contains("to=functions.<function_name>"));
        assert!(gpt_prompt.contains("<|start|>user<|message|>What is 2+2?<|end|>"));
        assert!(gpt_prompt.ends_with("<|start|>assistant"));

        let mistral_prompt = format_chat_prompt(&messages, Some(&tools), Some("mistral"));
        assert!(mistral_prompt.contains("[INST]"));
        assert!(mistral_prompt.contains("[AVAILABLE_TOOLS]"));
        assert!(mistral_prompt.contains("[/INST]"));

        let granite_prompt = format_chat_prompt(&messages, Some(&tools), Some("granite"));
        assert!(granite_prompt.contains("<|start_of_role|>system<|end_of_role|>"));
        assert!(granite_prompt.contains("<|start_of_role|>user<|end_of_role|>What is 2+2?<|end_of_text|>"));
        assert!(granite_prompt.ends_with("<|start_of_role|>assistant<|end_of_role|>"));
    }

    #[test]
    fn test_qwen_tool_response_no_forced_think() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "get temperature".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(vec![ToolCall {
                    id: Some("call_1".into()),
                    call_type: Some("function".into()),
                    function: FunctionCall {
                        name: "weathertool".into(),
                        arguments: serde_json::json!({}),
                    },
                }]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                content: "[{\"station\": \"A\", \"temp\": -5}]".into(),
                ..Default::default()
            },
        ];
        let prompt = format_chat_prompt(&messages, None, Some("qwen"));
        assert!(!prompt.contains("<think>\n\n</think>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_parse_glm_tool_call_with_corrupted_trailing_tag() {
        let raw = "<tool_call>{\"name\": \"get_devices_around_position\", \"arguments\": {\"place\": \"Göttingen, Germany\", \"km\": 20}}</arg_value></tool_call>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_devices_around_position");
        assert_eq!(tcs[0].function.arguments["place"], serde_json::json!("Göttingen, Germany"));
        assert_eq!(tcs[0].function.arguments["km"], serde_json::json!(20));
    }

    #[test]
    fn test_parse_ministral_multiple_raw_arrays() {
        let raw = "[\n  {\n    \"name\": \"get_devices_by_name\",\n    \"arguments\": {\n      \"dluName\": \"Lengden\",\n      \"fields\": \"id,name,channels\"\n    }\n  }\n][\n  {\n    \"name\": \"get_excel_measurement_data_by_name\",\n    \"arguments\": {\n      \"dluName\": \"Lengden\",\n      \"channels\": \"ALL\",\n      \"hour\": 12,\n      \"fileName\": \"Lengden_Weather_Data\"\n    }\n  }\n]Here is the Excel file";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "Here is the Excel file");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].function.name, "get_devices_by_name");
        assert_eq!(tcs[0].function.arguments["dluName"], serde_json::json!("Lengden"));
        assert_eq!(tcs[1].function.name, "get_excel_measurement_data_by_name");
        assert_eq!(tcs[1].function.arguments["hour"], serde_json::json!(12));
    }

    #[test]
    fn test_format_gemma_chat_prompt() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
            ..Default::default()
        }];
        let tools = serde_json::json!([{
            "name": "get_weather",
            "description": "Fetch weather",
            "parameters": {}
        }]);
        // Gemma 2 / 3 uses `<start_of_turn>`
        let prompt = format_chat_prompt(&messages, Some(&tools), Some("gemma3"));
        assert!(prompt.contains("<start_of_turn>user"));
        assert!(prompt.contains("# Available Tools"));
        assert!(prompt.contains("<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_format_and_parse_gemma4_26b_multiturn_tools() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to file" }
                    },
                    "required": ["path"]
                }
            }
        }]);

        let turn1_msgs = vec![ChatMessage {
            role: "user".into(),
            content: "Analyze this project".into(),
            ..Default::default()
        }];
        let p1 = format_chat_prompt(&turn1_msgs, Some(&tools), Some("gemma4-26b"));
        assert!(p1.starts_with("<|turn>system\n<|think|>"));
        assert!(p1.contains("<|tool>declaration:read_file{description:<|\"|>Read file content<|\"|>,parameters:{properties:{path:{description:<|\"|>Path to file<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"));
        assert!(p1.contains("<|turn>user\nAnalyze this project<turn|>\n"));
        assert!(p1.ends_with("<|turn>model\n"));

        // Parse Gemma 4 tool call with embedded quotes and newlines inside <|"|>...<|"|>
        let raw_gemma4 = "<|channel>thought\nLet me inspect Cargo.toml\n<channel|><|tool_call>call:read_file{path:<|\"|>src/\"quoted\".rs<|\"|>}<tool_call|><turn|>";
        let (content, tcs) = parse_tool_calls(raw_gemma4);
        assert_eq!(content, "");
        let tcs = tcs.expect("expected Gemma 4 tool call");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "read_file");
        assert_eq!(tcs[0].function.arguments["path"], serde_json::json!("src/\"quoted\".rs"));
        assert_eq!(tcs[0].id.as_ref().unwrap().len(), 9);

        // Turn 2: after tool execution, `<|turn>model\n...<tool_response|>` remains open
        let turn2_msgs = vec![
            turn1_msgs[0].clone(),
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(tcs.clone()),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                content: "[package]\nname = \"hypura\"".into(),
                tool_call_id: tcs[0].id.clone(),
                ..Default::default()
            },
        ];
        let p2 = format_chat_prompt(&turn2_msgs, Some(&tools), Some("gemma4-26b"));
        assert!(p2.contains("<|tool_call>call:read_file{path:<|\"|>src/\"quoted\".rs<|\"|>}<tool_call|>"));
        assert!(p2.ends_with("<|tool_response>response:read_file{value:<|\"|>[package]\nname = \"hypura\"<|\"|>}<tool_response|>"));
    }

    #[test]
    fn test_nemotron_30b_not_routed_to_mistral_and_parses_toolcalls() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "List files".into(),
            ..Default::default()
        }];
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in directory",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }
            }
        }]);

        let prompt = format_chat_prompt(&messages, Some(&tools), Some("nemotron_h Nemotron-3-Nano-30B-A3B"));
        assert!(!prompt.contains("[INST]"), "Nemotron must not be routed to Mistral [INST]");
        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("<function>\n<name>list_files</name>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));

        // Uppercase <TOOLCALL>[...]</TOOLCALL> (Nemotron-Nano-v2)
        let raw_upper = "<think>\nChecking dir\n</think>\n<TOOLCALL>[{\"name\": \"list_files\", \"arguments\": {\"path\": \".\"}}]</TOOLCALL>";
        let (c1, tc1) = parse_tool_calls(raw_upper);
        assert_eq!(c1, "");
        let tc1 = tc1.expect("expected Nemotron TOOLCALL");
        assert_eq!(tc1[0].function.name, "list_files");
        assert_eq!(tc1[0].function.arguments["path"], serde_json::json!("."));

        // Prefilled `<think>\n` orphan `</think>` + XML `<function=...>` (Nemotron-3 30B / Qwen 3.8 26B)
        let raw_prefilled = "Need to call <function=fake> in thought</think>\n<tool_call>\n<function=list_files>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>";
        let (c2, tc2) = parse_tool_calls(raw_prefilled);
        assert_eq!(c2, "");
        let tc2 = tc2.expect("expected XML tool call after orphan </think>");
        assert_eq!(tc2.len(), 1);
        assert_eq!(tc2[0].function.name, "list_files");
        assert_eq!(tc2[0].function.arguments["path"], serde_json::json!("."));
    }

    #[test]
    fn test_mistral_small_26b_args_and_call_id_format_and_parse() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List directory",
                "parameters": { "type": "object", "properties": { "path": { "type": "string" } } }
            }
        }]);
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: "Inspect repo".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(vec![ToolCall {
                    id: Some("call_long_uuid_12345".into()),
                    call_type: Some("function".into()),
                    function: FunctionCall {
                        name: "list_dir".into(),
                        arguments: serde_json::json!({"path": "."}),
                    },
                }]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                content: "src/\nCargo.toml".into(),
                tool_call_id: Some("call_long_uuid_12345".into()),
                ..Default::default()
            },
        ];

        let prompt = format_chat_prompt(
            &msgs,
            Some(&tools),
            Some("mistral Mistral-Small-3.2-24B-Instruct-2506 [SYSTEM_PROMPT] [CALL_ID] [ARGS]"),
        );
        assert!(prompt.starts_with("[SYSTEM_PROMPT]"));
        assert!(prompt.contains("[AVAILABLE_TOOLS][{\"function\":"));
        assert!(prompt.contains("[TOOL_CALLS]list_dir[CALL_ID]uuid12345[ARGS]{\"path\":\".\"}</s>"));
        assert!(prompt.contains("[TOOL_RESULTS]uuid12345[TOOL_CONTENT]src/\nCargo.toml[/TOOL_RESULTS]"));

        // Parse `[THINK]...[/THINK][TOOL_CALLS]name[CALL_ID]id[ARGS]{...}` and parallel `[TOOL_CALLS]name[ARGS]{...}`
        let raw = "[THINK]Let me list files and read Cargo.toml[/THINK][TOOL_CALLS]list_dir[CALL_ID]abc123456[ARGS]{\"path\": \".\"}[TOOL_CALLS]read_file[ARGS]{\"path\": \"Cargo.toml\"}</s>";
        let (content, tcs) = parse_tool_calls(raw);
        assert_eq!(content, "");
        let tcs = tcs.expect("expected Mistral Small tool calls");
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].function.name, "list_dir");
        assert_eq!(tcs[0].id.as_deref(), Some("abc123456"));
        assert_eq!(tcs[0].function.arguments["path"], serde_json::json!("."));
        assert_eq!(tcs[1].function.name, "read_file");
        assert_eq!(tcs[1].function.arguments["path"], serde_json::json!("Cargo.toml"));
    }

    #[test]
    fn test_qwen38_26b_xml_prompt_and_multiturn_format() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List directory contents",
                "parameters": { "type": "object", "properties": { "path": { "type": "string" } } }
            }
        }]);
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: "Analyze project".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(vec![ToolCall {
                    id: Some("call00001".into()),
                    call_type: Some("function".into()),
                    function: FunctionCall {
                        name: "list_directory".into(),
                        arguments: serde_json::json!({"path": "."}),
                    },
                }]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                content: "Cargo.toml\nsrc/".into(),
                tool_call_id: Some("call00001".into()),
                ..Default::default()
            },
        ];

        let prompt = format_chat_prompt(&msgs, Some(&tools), Some("qwen35moe qwen3.8-26b <function="));
        assert!(prompt.contains("<function=example_function_name>"));
        assert!(prompt.contains("<tool_call>\n<function=list_directory>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call><|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\n<tool_response>\nCargo.toml\nsrc/\n</tool_response><|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
    }
}
