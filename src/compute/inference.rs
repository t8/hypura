use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::compute::ffi::*;
use crate::compute::nvme_backend::{
    build_override_patterns, eval_callback, HypuraBuftController, LayerStatus, PrefetchState,
};
use crate::model::gguf::GgufFile;
use crate::model::metadata::ModelMetadata;
use crate::model::tensor_role::TensorRole;
use crate::profiler::types::HardwareProfile;
use crate::scheduler::types::*;
use crate::telemetry::metrics::{TelemetryEmitter, TelemetryEvent};

/// A token emitted during generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToken {
    pub text: String,
    pub token_id: i32,
    pub tok_per_sec: f64,
    pub is_eog: bool,
}

/// Configuration for an inference session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
    pub sampling: SamplingParams,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: num_performance_cores(),
            sampling: SamplingParams::default(),
        }
    }
}

/// Result returned after generation completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: u32,
    pub prompt_tokens: u32,
    pub tok_per_sec_avg: f64,
    pub prompt_eval_ms: f64,
    pub perf: PerfData,
}

/// Persistent inference context & token history cached across HTTP requests for prefix KV reuse.
struct CachedSession {
    ctx: LlamaContext,
    n_ctx: u32,
    tokens: Vec<i32>,
}

/// A loaded model that can serve multiple generation requests.
///
/// Holds the llama.cpp backend, model, persistent KV session, and NVMe scheduling state.
/// Note: `cached_session` is declared first so `LlamaContext` drops before `LlamaModel`.
pub struct LoadedModel {
    cached_session: Option<CachedSession>,
    pub _backend: LlamaBackend,
    pub model: LlamaModel,
    pub config: InferenceConfig,
    pub n_gpu_layers: i32,
    pub model_name: String,
    pub kv_quantization: Option<crate::scheduler::types::KvQuantization>,
    // NVMe scheduling state (None when all tensors fit in GPU+RAM)
    _controller: Option<Box<HypuraBuftController>>,
    prefetch_state: Option<Arc<PrefetchState>>,
    keep_resident: bool,
}

// SAFETY: All access to LoadedModel is serialized under std::sync::Mutex
// and only happens from spawn_blocking threads.
unsafe impl Send for LoadedModel {}

impl Drop for LoadedModel {
    fn drop(&mut self) {
        // Free LlamaContext before stopping the NVMe I/O pool or freeing LlamaModel
        self.cached_session = None;
        if let Some(ref state) = self.prefetch_state {
            state.stop_io_pool();
        }
    }
}

/// Parameters for `generate_from_loaded`.
pub struct GenerateFromLoadedParams<'a> {
    pub prompt: &'a str,
    pub sampling: &'a SamplingParams,
    pub token_tx: mpsc::UnboundedSender<GeneratedToken>,
    pub telemetry: Arc<TelemetryEmitter>,
}

/// Load a model once for repeated generation (server use case).
///
/// Extracts the model loading logic from `generate_with_nvme_scheduling` so the
/// heavy work (GGUF parse, placement, buffer setup, model load, prefetch init)
/// happens once at startup.
pub fn load_model(
    model_path: &Path,
    config: &InferenceConfig,
    n_gpu_layers: i32,
    plan: &PlacementPlan,
    gguf: &GgufFile,
) -> anyhow::Result<LoadedModel> {
    let backend = LlamaBackend::init();

    let has_nvme = plan
        .tier_assignments
        .values()
        .any(|t| *t == StorageTier::Nvme);

    // Derive model name from GGUF metadata or filename
    let model_name = gguf
        .get_string("general.name")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            model_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".into())
        });

    if !has_nvme {
        let model = LlamaModel::load(model_path, n_gpu_layers, true)?;
        return Ok(LoadedModel {
            cached_session: None,
            _backend: backend,
            model,
            config: config.clone(),
            n_gpu_layers,
            model_name,
            kv_quantization: plan.kv_cache_plan.kv_quantization,
            _controller: None,
            prefetch_state: None,
            keep_resident: false,
        });
    }

    // NVMe path: create custom buffer type
    let controller = HypuraBuftController::new(model_path, gguf);
    let (_patterns, overrides) =
        build_override_patterns(plan, gguf, controller.buft_ptr(), n_gpu_layers);

    let nvme_count = plan
        .tier_assignments
        .values()
        .filter(|t| **t == StorageTier::Nvme)
        .count();
    tracing::info!("NVMe scheduling: {nvme_count} tensors on custom buffer type");

    let model =
        LlamaModel::load_with_overrides(model_path, n_gpu_layers, true, overrides.as_ptr())?;

    let num_layers = model.n_layers() as u32;

    let nvme_layers: std::collections::HashSet<u32> = gguf
        .tensors
        .iter()
        .filter(|t| plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme))
        .filter_map(|t| t.layer_index)
        .collect();

    let prefetch_state = controller.build_prefetch_state(gguf, num_layers, nvme_layers);

    // Determine keep-resident mode
    let total_ram = total_physical_memory();
    let nvme_bytes: u64 = gguf
        .tensors
        .iter()
        .filter(|t| plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme))
        .map(|t| t.size_bytes)
        .sum();

    let model_total_bytes = gguf.total_tensor_bytes();
    let buffer_bytes: u64 = gguf
        .tensors
        .iter()
        .filter(|t| {
            let tier = plan.tier_assignments.get(&t.name);
            tier == Some(&StorageTier::Nvme) || tier == Some(&StorageTier::Ram)
        })
        .map(|t| t.size_bytes)
        .sum();
    let gpu_bytes = model_total_bytes.saturating_sub(buffer_bytes);
    let gpu_committed_estimate = gpu_bytes * 60 / 100;
    let runtime_overhead: u64 = 5 * (1 << 29);
    let estimated_committed = gpu_committed_estimate + buffer_bytes + runtime_overhead;
    let headroom: u64 = 4 * (1 << 30);

    // Enforce 90% maximum unified memory limit (leave 10% for macOS OS/system safety)
    let max_ram_limit = (total_ram as f64 * 0.90) as u64;
    let effective_max_ram = max_ram_limit.min(total_ram.saturating_sub(headroom));

    let keep_resident = nvme_bytes > 0
        && (estimated_committed + nvme_bytes) <= effective_max_ram;

    let should_preload = keep_resident
        && (estimated_committed + nvme_bytes) <= effective_max_ram.saturating_sub(2 * (1 << 30));

    if keep_resident {
        tracing::info!(
            "NVMe keep-resident mode: {:.2} GB NVMe spill, est. committed {:.1}/{:.1} GB",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            (estimated_committed + nvme_bytes) as f64 / (1u64 << 30) as f64,
            total_ram as f64 / (1u64 << 30) as f64,
        );
    } else if nvme_bytes > 0 {
        tracing::info!(
            "NVMe streaming mode: {:.2} GB NVMe spill",
            nvme_bytes as f64 / (1u64 << 30) as f64,
        );
    }
    prefetch_state
        .keep_nvme_resident
        .store(keep_resident, std::sync::atomic::Ordering::Relaxed);

    // Start multi-threaded I/O pool
    let num_io_workers = (num_performance_cores() as usize / 2).clamp(2, 4);
    prefetch_state.start_io_pool(num_io_workers)?;

    if should_preload {
        prefetch_state.preload_ram_layers();
    } else if keep_resident {
        tracing::info!(
            "Skipping preload: model {:.1} GB exceeds preload threshold (will load lazily)",
            model_total_bytes as f64 / (1u64 << 30) as f64,
        );
    }

    // Initial NVMe prefetch
    prefetch_state.prefetch_all_nvme();

    if keep_resident {
        prefetch_state
            .prefetch_enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    Ok(LoadedModel {
        cached_session: None,
        _backend: backend,
        model,
        config: config.clone(),
        n_gpu_layers,
        model_name,
        kv_quantization: plan.kv_cache_plan.kv_quantization,
        _controller: Some(controller),
        prefetch_state: Some(prefetch_state),
        keep_resident,
    })
}

/// Helper: find matching closing bracket `]` accounting for strings and nesting
pub fn find_matching_bracket(s: &str) -> Option<usize> {
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
pub fn find_matching_brace(s: &str) -> Option<usize> {
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

/// Check if generation should terminate early due to stop tokens, turn boundaries, or tool call completion.
pub fn should_stop_early(generated_text: &str, custom_stops: &[String]) -> bool {
    // 1. User/client-specified stop sequences
    for stop in custom_stops {
        if !stop.is_empty() && generated_text.contains(stop.as_str()) {
            return true;
        }
    }

    // 2. ChatML & Qwen stop / turn boundary tokens
    if generated_text.contains("<|im_end|>")
        || generated_text.contains("<|im_start|>")
        || generated_text.contains("<|endoftext|>")
        || generated_text.contains("<|eot_id|>")
    {
        return true;
    }

    // 3. Mistral stop / turn boundary tokens
    if generated_text.contains("</s>")
        || generated_text.contains("[INST]")
        || generated_text.contains("[AVAILABLE_TOOLS]")
        || generated_text.contains("[/TOOL_CALLS]")
        || generated_text.contains("[TOOL_RESULTS]")
    {
        return true;
    }

    // 4. Granite & Gemma stop / turn boundary tokens
    if generated_text.contains("<|end_of_text|>")
        || generated_text.contains("<|end_of_turn|>")
        || generated_text.contains("<|start_of_role|>")
        || generated_text.contains("<end_of_turn>")
        || generated_text.contains("<start_of_turn>")
        || generated_text.contains("<turn|>")
        || generated_text.contains("<|turn|>")
    {
        return true;
    }

    // 5. Tool call closure tags or complete JSON tool call objects
    if generated_text.contains("</tool_call>")
        || generated_text.contains("</tool__call>")
        || generated_text.contains("</toolcall>")
        || generated_text.contains("<tool_call|>")
        || generated_text.contains("</function>")
        || generated_text.contains("[/TOOL_CALLS]")
        || (generated_text.contains("<|tool_call|>") && (generated_text.ends_with('}') || generated_text.ends_with(']')))
    {
        return true;
    }

    let trimmed = generated_text.trim_start();

    // Check complete [TOOL_CALLS] [...] syntax
    if let Some(tc_idx) = trimmed.find("[TOOL_CALLS]") {
        let after_tc = trimmed[tc_idx + "[TOOL_CALLS]".len()..].trim_start();
        if after_tc.starts_with('[') {
            if let Some(bracket_len) = find_matching_bracket(after_tc) {
                let candidate = &after_tc[..bracket_len];
                if candidate.contains("\"name\"") && serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                    return true;
                }
            }
        } else if after_tc.starts_with('{') {
            if let Some(brace_len) = find_matching_brace(after_tc) {
                let candidate = &after_tc[..brace_len];
                if candidate.contains("\"name\"") && serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                    return true;
                }
            }
        }
    }

    // Check complete raw JSON array [ {"name": ...} ]
    if trimmed.starts_with('[')
        && !trimmed.starts_with("[TOOL_CALLS]")
        && !trimmed.starts_with("[INST]")
        && !trimmed.starts_with("[AVAILABLE_TOOLS]")
    {
        if let Some(bracket_len) = find_matching_bracket(trimmed) {
            let candidate = &trimmed[..bracket_len];
            if candidate.contains("\"name\"") && serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return true;
            }
        }
    }

    // Check complete raw JSON object { "name": ... }
    if trimmed.starts_with('{') {
        if let Some(brace_len) = find_matching_brace(trimmed) {
            let candidate = &trimmed[..brace_len];
            if candidate.contains("\"name\"") && serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return true;
            }
        }
    }

    // 6. Prevent reasoning over-thinking loops (e.g. model finished thinking, gave answer, then reopened <think>)
    if generated_text.contains("</think>") {
        if let Some(think_end) = generated_text.find("</think>") {
            let after_think = &generated_text[think_end + "</think>".len()..];
            if after_think.contains("<think>") || after_think.contains("<thought>") {
                return true;
            }
        }
    }
    if generated_text.contains("</thought>") {
        if let Some(thought_end) = generated_text.find("</thought>") {
            let after_thought = &generated_text[thought_end + "</thought>".len()..];
            if after_thought.contains("<thought>") || after_thought.contains("<think>") {
                return true;
            }
        }
    }

    false
}

/// Generate text from a pre-loaded model.
///
/// Reuses a persistent `LlamaContext` and cached prompt KV prefix across requests
/// when compatible, and aborts promptly if the HTTP client disconnects.
pub fn generate_from_loaded(
    loaded: &mut LoadedModel,
    params: GenerateFromLoadedParams<'_>,
) -> anyhow::Result<GenerationResult> {
    let GenerateFromLoadedParams {
        prompt,
        sampling,
        token_tx,
        telemetry,
    } = params;

    if token_tx.is_closed() {
        anyhow::bail!("Generation cancelled: client disconnected before start");
    }

    let n_ctx_limit = loaded.config.n_ctx;
    let n_batch = loaded.config.n_batch;
    let n_threads = loaded.config.n_threads;

    let tokens = loaded.model.tokenize(prompt, true, true);
    let prompt_len = tokens.len() as u32;
    anyhow::ensure!(!tokens.is_empty(), "Prompt tokenized to zero tokens");

    // Dynamic context calculation: keep within config.n_ctx bounds to ensure
    // KV cache allocations stay within the reserved Metal working set limit.
    let gen_headroom = sampling.max_tokens.min(4096).max(512);
    let effective_ctx = (prompt_len + gen_headroom).min(n_ctx_limit).max(1024);

    let need_new_ctx = loaded
        .cached_session
        .as_ref()
        .map_or(true, |s| s.n_ctx < effective_ctx);

    if need_new_ctx {
        // Release previous Metal KV buffers first so old and new contexts never coexist
        loaded.cached_session = None;

        // Allocate with a 4k growth cushion (capped at n_ctx_limit) so multi-turn
        // coding-agent sessions don't reallocate KV buffers on every turn.
        let alloc_ctx = effective_ctx
            .saturating_add(4096)
            .min(n_ctx_limit)
            .max(effective_ctx);

        let ctx = if let Some(ref prefetch_state) = loaded.prefetch_state {
            let state_ptr = Arc::into_raw(prefetch_state.clone()) as *mut std::ffi::c_void;
            let ctx_res = LlamaContext::new_with_callback_and_kv(
                &loaded.model,
                alloc_ctx,
                n_batch,
                n_threads,
                Some(eval_callback),
                state_ptr,
                loaded.kv_quantization,
            );
            // Immediately convert back to avoid leak — the PrefetchState is kept alive
            // by the Arc in LoadedModel, not by this raw pointer.
            unsafe {
                Arc::from_raw(state_ptr as *const PrefetchState);
            }
            ctx_res?
        } else {
            LlamaContext::new_with_callback_and_kv(
                &loaded.model,
                alloc_ctx,
                n_batch,
                n_threads,
                None,
                std::ptr::null_mut(),
                loaded.kv_quantization,
            )?
        };

        loaded.cached_session = Some(CachedSession {
            ctx,
            n_ctx: alloc_ctx,
            tokens: Vec::new(),
        });
    }

    let mut sampler = LlamaSampler::new(sampling);

    // Prefetch NVMe layers before prompt eval
    if let Some(ref state) = loaded.prefetch_state {
        state.prefetch_all_nvme();
    }

    // Process prompt (ensure prompt fits within effective context minus generation headroom)
    let max_prompt_tokens = effective_ctx.saturating_sub(64) as usize;
    let eval_tokens = if tokens.len() > max_prompt_tokens {
        tracing::warn!(
            "Prompt length ({} tokens) exceeds context limit ({}). Truncating prompt from the beginning...",
            tokens.len(),
            max_prompt_tokens
        );
        &tokens[tokens.len() - max_prompt_tokens..]
    } else {
        &tokens[..]
    };

    let n_swa = loaded.model.n_swa();
    let session = loaded.cached_session.as_mut().unwrap();

    // Reuse longest valid common token prefix already present in KV memory,
    // leaving at least 1 token to decode so logits are freshly computed for sampling.
    let common_len = session
        .tokens
        .iter()
        .zip(eval_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let desired_reuse = common_len.min(eval_tokens.len().saturating_sub(1));
    let reused_len = session.ctx.trim_kv_to_prefix(desired_reuse, n_swa);
    session.tokens.truncate(reused_len);

    if reused_len > 0 {
        tracing::info!(
            "KV prefix cache hit: reused {}/{} prompt tokens (decoding {} new tokens)",
            reused_len,
            eval_tokens.len(),
            eval_tokens.len() - reused_len
        );
    }

    let prompt_start = Instant::now();
    let batch_size = n_batch as usize;
    for chunk in eval_tokens[reused_len..].chunks(batch_size) {
        if token_tx.is_closed() {
            tracing::info!("Client disconnected during prompt prefill; aborting generation");
            anyhow::bail!("Generation cancelled: client disconnected");
        }
        if let Err(e) = session.ctx.decode(chunk) {
            session.ctx.clear_memory();
            session.tokens.clear();
            return Err(e);
        }
        session.tokens.extend_from_slice(chunk);
    }
    let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;

    if token_tx.is_closed() {
        tracing::info!("Client disconnected after prompt prefill; aborting generation");
        anyhow::bail!("Generation cancelled: client disconnected");
    }

    // Generation loop
    let mut generated_text = String::new();
    let mut n_generated: u32 = 0;
    let gen_start = Instant::now();

    for _ in 0..sampling.max_tokens {
        let token_id = sampler.sample(&mut session.ctx, -1);
        let is_eog = loaded.model.is_eog(token_id);
        let is_control = loaded.model.is_control(token_id);
        let piece = loaded.model.token_to_piece(token_id);

        n_generated += 1;
        generated_text.push_str(&piece);

        let elapsed = gen_start.elapsed().as_secs_f64();
        let tok_per_sec = if elapsed > 0.0 {
            n_generated as f64 / elapsed
        } else {
            0.0
        };

        telemetry.emit(TelemetryEvent::TokenGenerated {
            tok_per_sec,
            token: piece.clone(),
        });

        if token_tx
            .send(GeneratedToken {
                text: piece.clone(),
                token_id,
                tok_per_sec,
                is_eog,
            })
            .is_err()
        {
            break;
        }

        if is_eog {
            break;
        }

        if is_control
            && (piece.contains("end")
                || piece.contains("im_end")
                || piece.contains("eot")
                || piece.contains("stop")
                || piece.contains("start_of_role")
                || piece.contains("im_start")
                || piece.contains("INST"))
        {
            break;
        }

        if should_stop_early(&generated_text, &sampling.stop_sequences) {
            break;
        }

        if let Err(e) = session.ctx.decode(&[token_id]) {
            session.ctx.clear_memory();
            session.tokens.clear();
            return Err(e);
        }
        session.tokens.push(token_id);
    }

    let perf = session.ctx.perf();
    let total_gen_time = gen_start.elapsed().as_secs_f64();
    let avg_tps = if total_gen_time > 0.0 {
        n_generated as f64 / total_gen_time
    } else {
        0.0
    };

    Ok(GenerationResult {
        text: generated_text,
        tokens_generated: n_generated,
        prompt_tokens: prompt_len,
        tok_per_sec_avg: avg_tps,
        prompt_eval_ms: prompt_ms,
        perf,
    })
}

/// Compute GPU budget for model weights (bytes) after reserving space for
/// KV cache and compute buffers within the Metal working set.
pub fn compute_gpu_budget(hw: &HardwareProfile, metadata: &ModelMetadata, context_length: u32) -> u64 {
    let raw_vram = hw.gpu.as_ref().map_or(0, |g| g.vram_bytes);
    let max_memory_limit = (hw.memory.total_bytes as f64 * 0.90) as u64;
    let gpu_working_set = raw_vram.min(max_memory_limit);

    // KV cache on GPU: for MLA models (like GLM-4 MoE Lite, DeepSeek), KV is compressed (576 dims per token)
    let arch_lower = metadata.architecture.to_lowercase();
    let is_mla = arch_lower.contains("glm4moe") || arch_lower.contains("deepseek2") || arch_lower.contains("deepseek3");

    let raw_kv_on_gpu = if is_mla {
        // 576 bytes per layer * Q8_0 (1 byte/elem) * layers * context
        576 * metadata.num_layers as u64 * context_length as u64
    } else {
        let head_dim = if metadata.num_heads > 0 {
            metadata.embedding_dim as u64 / metadata.num_heads as u64
        } else {
            0
        };
        2 * metadata.num_layers as u64
            * metadata.num_kv_heads as u64
            * head_dim
            * 2
            * context_length as u64
    };

    // Scale GPU KV reservation by the auto-selected KV cache quantization (Q4_0 / Q8_0),
    // matching compute_tier_capacities and compute_kv_cache_plan in placement.rs.
    let kv_scale = if is_mla {
        1.0 // already compressed above
    } else if context_length >= 32768 || raw_kv_on_gpu > (4u64 << 30) {
        0.28 // Q4_0
    } else if context_length >= 16384 || raw_kv_on_gpu > (500 * (1 << 20)) {
        0.53 // Q8_0
    } else {
        1.0 // F16
    };
    let kv_on_gpu = (raw_kv_on_gpu as f64 * kv_scale) as u64;

    // Reserve headroom for Metal compute graph splits (150+ splits on dense models), SSM recurrent state buffers, and display compositor
    let is_ssm_hybrid = arch_lower.contains("gemma")
        || arch_lower.contains("delta")
        || arch_lower.contains("mamba")
        || arch_lower.contains("rwkv")
        || arch_lower.contains("jamba");

    let runtime_overhead: u64 = if is_ssm_hybrid {
        // Gated Delta Net and recurrent state buffers create substantial Metal intermediate allocations
        5800 * 1024 * 1024 // 5.8 GB safety buffer on unified memory
    } else if hw.memory.total_bytes <= 24 * 1024 * 1024 * 1024 {
        // On 24 GB machines, macOS WindowServer + system memory pressure requires 4.8 GB safety buffer
        4800 * 1024 * 1024
    } else {
        3800 * 1024 * 1024
    };

    gpu_working_set
        .saturating_sub(kv_on_gpu)
        .saturating_sub(runtime_overhead)
}

/// Derive `n_gpu_layers` from a PlacementPlan.
///
/// In expert-streaming mode, expert tensors are on the Hypura NVMe buffer (not Metal),
/// so they don't count against GPU working set or the NVMe-layer cutoff. All layers
/// can be offloaded to Metal — the eval_callback loads expert data on demand.
///
/// In other modes, caps at the minimum of:
/// 1. The first NVMe layer (layers with NVMe tensors must stay on CPU)
/// 2. The GPU working set capacity
pub fn gpu_layers_from_placement(
    plan: &PlacementPlan,
    gguf: &GgufFile,
    gpu_budget_bytes: u64,
) -> i32 {
    // SparseMoeMmap: if model fits entirely in GPU, offload all layers. If not,
    // offload as many layers as fit in the GPU budget rather than falling back to CPU-only (ngl=0).
    if plan.inference_mode == InferenceMode::SparseMoeMmap {
        let total_bytes = gguf.total_tensor_bytes();
        if total_bytes <= gpu_budget_bytes {
            let max_layer = gguf.tensors.iter().filter_map(|t| t.layer_index).max().unwrap_or(0);
            return max_layer as i32 + 1 + 1; // all layers + output
        }
        // Fall through to compute max fitting layers within gpu_budget_bytes below
    }

    let expert_streaming = plan.inference_mode == InferenceMode::ExpertStreaming;
    let dense_ffn_streaming = plan.inference_mode == InferenceMode::DenseFfnStreaming;
    let mut max_layer: i32 = -1;
    let mut first_non_gpu_layer: Option<u32> = None;

    // Compute per-layer sizes (excluding streamed tensors from GPU budget)
    let mut layer_sizes: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();

    for t in &gguf.tensors {
        if let Some(layer_idx) = t.layer_index {
            max_layer = max_layer.max(layer_idx as i32);

            // In streaming modes, NVMe-streamed tensors go to the Hypura pool buffer,
            // not Metal shared buffers — don't count them for GPU budget/NVMe cutoff.
            if expert_streaming {
                let role = TensorRole::from_name(&t.name);
                if matches!(role, TensorRole::MoeFusedExperts) {
                    continue;
                }
            }
            if dense_ffn_streaming {
                let role = TensorRole::from_name(&t.name);
                if matches!(role, TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown) {
                    continue;
                }
            }

            *layer_sizes.entry(layer_idx).or_default() += t.size_bytes;
            if plan.tier_assignments.get(&t.name) != Some(&StorageTier::Gpu) {
                first_non_gpu_layer = Some(match first_non_gpu_layer {
                    Some(existing) => existing.min(layer_idx),
                    None => layer_idx,
                });
            }
        }
    }

    if max_layer < 0 {
        return 0;
    }

    // Cap by plan cutoff (first layer with non-GPU tensors stays on CPU)
    let from_plan = match first_non_gpu_layer {
        Some(non_gpu_start) => non_gpu_start as i32,
        None => max_layer + 1 + 1,
    };

    // Cap by GPU working set: sum layers until budget exhausted.
    let non_layer_gpu_size: u64 = gguf
        .tensors
        .iter()
        .filter(|t| t.layer_index.is_none())
        .map(|t| t.size_bytes)
        .sum();
    let mut cumulative: u64 = non_layer_gpu_size;
    let mut max_fitting: i32 = 0;
    for (&layer_idx, &size) in &layer_sizes {
        cumulative += size;
        if cumulative <= gpu_budget_bytes {
            max_fitting = layer_idx as i32 + 1;
        } else {
            break;
        }
    }
    let from_capacity = if max_fitting == max_layer + 1 {
        max_fitting + 1
    } else {
        max_fitting
    };

    from_plan.min(from_capacity)
}

/// Run inference on a blocking thread. Streams tokens via `token_tx`.
pub fn generate_blocking(
    model_path: &Path,
    prompt: &str,
    config: &InferenceConfig,
    n_gpu_layers: i32,
    token_tx: mpsc::UnboundedSender<GeneratedToken>,
    telemetry: Arc<TelemetryEmitter>,
) -> anyhow::Result<GenerationResult> {
    let _backend = LlamaBackend::init();

    let model = LlamaModel::load(model_path, n_gpu_layers, true)?;
    let mut ctx = LlamaContext::new(&model, config.n_ctx, config.n_batch, config.n_threads)?;
    let mut sampler = LlamaSampler::new(&config.sampling);

    // Tokenize prompt
    let tokens = model.tokenize(prompt, true, true);
    let prompt_len = tokens.len() as u32;
    anyhow::ensure!(!tokens.is_empty(), "Prompt tokenized to zero tokens");

    // Process prompt
    let prompt_start = Instant::now();
    // Decode in batches if prompt is longer than n_batch
    let batch_size = config.n_batch as usize;
    for chunk in tokens.chunks(batch_size) {
        ctx.decode(chunk)?;
    }
    let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;

    // Generation loop
    let mut generated_text = String::new();
    let mut n_generated: u32 = 0;
    let gen_start = Instant::now();

    for _ in 0..config.sampling.max_tokens {
        let token_id = sampler.sample(&mut ctx, -1);
        let is_eog = model.is_eog(token_id);
        let is_control = model.is_control(token_id);
        let piece = model.token_to_piece(token_id);

        n_generated += 1;
        generated_text.push_str(&piece);

        let elapsed = gen_start.elapsed().as_secs_f64();
        let tok_per_sec = if elapsed > 0.0 {
            n_generated as f64 / elapsed
        } else {
            0.0
        };

        telemetry.emit(TelemetryEvent::TokenGenerated {
            tok_per_sec,
            token: piece.clone(),
        });

        let gen_token = GeneratedToken {
            text: piece.clone(),
            token_id,
            tok_per_sec,
            is_eog,
        };

        if token_tx.send(gen_token).is_err() {
            break;
        }

        if is_eog {
            break;
        }

        if is_control
            && (piece.contains("end")
                || piece.contains("im_end")
                || piece.contains("eot")
                || piece.contains("stop")
                || piece.contains("start_of_role")
                || piece.contains("im_start")
                || piece.contains("INST"))
        {
            break;
        }

        if should_stop_early(&generated_text, &config.sampling.stop_sequences) {
            break;
        }

        ctx.decode(&[token_id])?;
    }

    let perf = ctx.perf();
    let total_gen_time = gen_start.elapsed().as_secs_f64();
    let avg_tps = if total_gen_time > 0.0 {
        n_generated as f64 / total_gen_time
    } else {
        0.0
    };

    Ok(GenerationResult {
        text: generated_text,
        tokens_generated: n_generated,
        prompt_tokens: prompt_len,
        tok_per_sec_avg: avg_tps,
        prompt_eval_ms: prompt_ms,
        perf,
    })
}

/// Run inference with NVMe-aware tensor scheduling.
/// Uses custom buffer type for NVMe-tier tensors + cb_eval for layer tracking.
pub fn generate_with_nvme_scheduling(
    model_path: &Path,
    prompt: &str,
    config: &InferenceConfig,
    n_gpu_layers: i32,
    plan: &PlacementPlan,
    gguf: &GgufFile,
    token_tx: mpsc::UnboundedSender<GeneratedToken>,
    telemetry: Arc<TelemetryEmitter>,
) -> anyhow::Result<GenerationResult> {
    let _backend = LlamaBackend::init();

    // Check if there are any NVMe tensors
    let has_nvme = plan
        .tier_assignments
        .values()
        .any(|t| *t == StorageTier::Nvme);

    if !has_nvme {
        return generate_blocking(model_path, prompt, config, n_gpu_layers, token_tx, telemetry);
    }

    // Create custom buffer type for NVMe-tier tensors
    let controller = HypuraBuftController::new(model_path, gguf);
    let (_patterns, overrides) =
        build_override_patterns(plan, gguf, controller.buft_ptr(), n_gpu_layers);

    let nvme_count = plan
        .tier_assignments
        .values()
        .filter(|t| **t == StorageTier::Nvme)
        .count();
    tracing::info!("NVMe scheduling: {nvme_count} tensors on custom buffer type");

    // Expert/dense-FFN streaming: use_mmap=false so Metal creates individual GPU buffers
    // for resident tensors instead of one giant MTLBuffer for the entire model file.
    let use_mmap = !matches!(
        plan.inference_mode,
        InferenceMode::ExpertStreaming | InferenceMode::DenseFfnStreaming
    );

    // Dense FFN streaming: redirect FFN tensor fread to a small scratch buffer during
    // model loading. Without this, fread commits ~22 GB of anonymous mmap pages for
    // FFN tensors, causing OOM on 32 GB machines.
    if plan.inference_mode == InferenceMode::DenseFfnStreaming {
        controller.enable_dense_ffn_scratch(gguf);
    }

    let model = LlamaModel::load_with_overrides(
        model_path,
        n_gpu_layers,
        use_mmap,
        overrides.as_ptr(),
    )?;

    // Build prefetch state with layer groupings and file offsets
    let num_layers = model.n_layers() as u32;

    // Determine which layers are NVMe (released after use) vs RAM (loaded once, kept)
    let nvme_layers: std::collections::HashSet<u32> = gguf
        .tensors
        .iter()
        .filter(|t| plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme))
        .filter_map(|t| t.layer_index)
        .collect();

    let prefetch_state = controller.build_prefetch_state(gguf, num_layers, nvme_layers);

    // Determine if NVMe layers can stay resident in physical memory.
    // When the NVMe spill is modest relative to total RAM, we keep NVMe-tier data
    // loaded after the first forward pass — eliminating all NVMe I/O for subsequent
    // tokens. This is the key optimization for "barely overflows" models like
    // Mixtral 30.9GB on 32GB: the 2GB NVMe spill stays resident, so only the first
    // forward pass incurs NVMe I/O.
    let total_ram = total_physical_memory();
    let nvme_bytes: u64 = gguf
        .tensors
        .iter()
        .filter(|t| plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme))
        .map(|t| t.size_bytes)
        .sum();
    // Keep-resident mode: keep NVMe data loaded after the first forward pass,
    // eliminating all NVMe I/O for subsequent tokens.
    //
    // Estimate actual committed memory: GPU layers via mmap commit ~60% of their
    // size on Apple Silicon (demand paging), our buffer commits its full allocation,
    // plus ~2.5 GB for KV cache + Metal overhead.
    //
    // Keep-resident when: estimated_committed + nvme_bytes fits in RAM with 4 GB
    // headroom for page cache. This lets small-spill models (Mixtral 2 GB on 32 GB)
    // stay resident while large-spill models (Llama 70B, 16 GB NVMe) use streaming.
    let model_total_bytes = gguf.total_tensor_bytes();
    let buffer_bytes: u64 = gguf
        .tensors
        .iter()
        .filter(|t| {
            let tier = plan.tier_assignments.get(&t.name);
            tier == Some(&StorageTier::Nvme) || tier == Some(&StorageTier::Ram)
        })
        .map(|t| t.size_bytes)
        .sum();
    let gpu_bytes = model_total_bytes.saturating_sub(buffer_bytes);
    let gpu_committed_estimate = gpu_bytes * 60 / 100; // ~60% committed via mmap
    let runtime_overhead: u64 = 5 * (1 << 29); // ~2.5 GB for KV cache + Metal
    let estimated_committed = gpu_committed_estimate + buffer_bytes + runtime_overhead;
    let headroom: u64 = 4 * (1 << 30); // 4 GB for page cache + system

    let keep_resident = nvme_bytes > 0
        && (estimated_committed + nvme_bytes) <= total_ram.saturating_sub(headroom);

    // Preloading is separate: only preload when estimated committed memory
    // (including all buffer layers pre-loaded) fits with 6 GB headroom.
    let should_preload = keep_resident
        && (estimated_committed + nvme_bytes) <= total_ram.saturating_sub(6 * (1 << 30));

    if keep_resident {
        tracing::info!(
            "NVMe keep-resident mode: {:.2} GB NVMe spill, est. committed {:.1}/{:.1} GB",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            (estimated_committed + nvme_bytes) as f64 / (1u64 << 30) as f64,
            total_ram as f64 / (1u64 << 30) as f64,
        );
    } else if nvme_bytes > 0 {
        tracing::info!(
            "NVMe streaming mode: {:.2} GB NVMe spill, est. committed {:.1} GB + {:.1} GB NVMe > {:.1} GB limit",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            estimated_committed as f64 / (1u64 << 30) as f64,
            nvme_bytes as f64 / (1u64 << 30) as f64,
            total_ram.saturating_sub(headroom) as f64 / (1u64 << 30) as f64,
        );
    }
    let expert_streaming = plan.inference_mode == InferenceMode::ExpertStreaming;
    let dense_ffn_streaming = plan.inference_mode == InferenceMode::DenseFfnStreaming;
    let any_streaming = expert_streaming || dense_ffn_streaming;

    // Unified memory budget for pool sizing and residency decisions.
    // Streaming modes use use_mmap=false → GPU tensors 100% committed.
    let metadata_for_budget = ModelMetadata::from_gguf(gguf).ok();
    let head_dim = metadata_for_budget
        .as_ref()
        .map(|m| {
            if m.num_heads > 0 {
                m.embedding_dim as u64 / m.num_heads as u64
            } else {
                0
            }
        })
        .unwrap_or(0);
    let num_kv_heads = metadata_for_budget
        .as_ref()
        .map(|m| m.num_kv_heads)
        .unwrap_or(0);

    let memory_budget = MemoryBudget::compute(
        total_ram,
        gpu_bytes,
        !any_streaming, // use_mmap=false for streaming modes
        num_layers,
        num_kv_heads,
        head_dim,
        config.n_ctx,
        plan.kv_cache_plan.kv_quantization,
    );

    tracing::info!(
        "Memory budget: {:.1} GB committed ({:.1} GPU + {:.1} KV + {:.1} Metal + {:.1} OS), {:.1} GB available",
        memory_budget.total_committed as f64 / 1e9,
        memory_budget.gpu_committed as f64 / 1e9,
        memory_budget.kv_cache_bytes as f64 / 1e9,
        memory_budget.metal_overhead as f64 / 1e9,
        memory_budget.os_overhead as f64 / 1e9,
        memory_budget.available as f64 / 1e9,
    );

    if expert_streaming {
        // Expert-streaming: non-expert tensors on GPU/Metal, experts on NVMe buffer.
        let num_experts = gguf.get_u32("expert_count").unwrap_or(8);
        // Dynamic pool slots: each slot = largest fused expert tensor (~385 MB for Mixtral).
        // Min 6 (2 layers × 3 tensors), max 18 (6 layers).
        let expert_slot_size = gguf
            .tensors
            .iter()
            .filter(|t| matches!(TensorRole::from_name(&t.name), TensorRole::MoeFusedExperts))
            .map(|t| t.size_bytes)
            .max()
            .unwrap_or(1);
        let num_slots = memory_budget.pool_slots(expert_slot_size, 6, 18);
        let pool = controller.activate_expert_pool(gguf, num_experts, num_slots)?;
        let memory_budget = memory_budget.with_pool(pool.pool_size as u64);
        tracing::info!(
            "Expert-streaming mode: {:.2} GB expert tensors on NVMe, {:.0} MB pool ({} slots), {:.1} GB available",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            pool.pool_size as f64 / 1e6,
            pool.num_slots,
            memory_budget.available as f64 / 1e9,
        );

        *prefetch_state.expert_pool.lock().unwrap() = Some(pool);
        let tensor_ptrs = controller.take_tensor_ptrs();
        let state_mut = unsafe {
            &mut *(Arc::as_ptr(&prefetch_state) as *mut PrefetchState)
        };
        state_mut.fused_tensor_ptrs = tensor_ptrs;

        prefetch_state
            .expert_streaming
            .store(true, std::sync::atomic::Ordering::Relaxed);
    } else if dense_ffn_streaming {
        // Dense FFN-streaming: attention+norms on GPU, FFN on NVMe pool buffer.
        // Dynamic pool slots: each slot = largest FFN tensor (~193 MB for Llama 70B).
        // Min 6 (2 layers × 3 tensors), max 24 (8 layers).
        let ffn_slot_size = gguf
            .tensors
            .iter()
            .filter(|t| {
                matches!(
                    TensorRole::from_name(&t.name),
                    TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown
                )
            })
            .map(|t| t.size_bytes)
            .max()
            .unwrap_or(1);
        let num_slots = memory_budget.pool_slots(ffn_slot_size, 6, 24);
        let pool = controller.activate_dense_ffn_pool(gguf, num_slots)?;
        let _memory_budget = memory_budget.with_pool(pool.pool_size as u64);
        tracing::info!(
            "Dense FFN-streaming mode: {:.2} GB FFN tensors on NVMe, {:.0} MB pool ({} slots), {:.1} GB available",
            nvme_bytes as f64 / (1u64 << 30) as f64,
            pool.pool_size as f64 / 1e6,
            pool.num_slots,
            _memory_budget.available as f64 / 1e9,
        );

        // Prefetch lookahead = (pool_slots / 3) - 1 (reserve current layer)
        let lookahead = ((num_slots / 3).saturating_sub(1)).max(1) as u32;
        *prefetch_state.expert_pool.lock().unwrap() = Some(pool);
        let tensor_ptrs = controller.take_tensor_ptrs();
        let state_mut = unsafe {
            &mut *(Arc::as_ptr(&prefetch_state) as *mut PrefetchState)
        };
        state_mut.fused_tensor_ptrs = tensor_ptrs;
        state_mut.dense_ffn_lookahead = lookahead;

        prefetch_state
            .dense_ffn_streaming
            .store(true, std::sync::atomic::Ordering::Relaxed);
    } else {
        prefetch_state
            .keep_nvme_resident
            .store(keep_resident, std::sync::atomic::Ordering::Relaxed);
    }

    // Start multi-threaded I/O pool (opens per-worker F_NOCACHE fds)
    let num_io_workers = (num_performance_cores() as usize / 2).clamp(2, 4);
    prefetch_state.start_io_pool(num_io_workers)?;

    // Only preload when the full model fits with 6 GB headroom.
    // For "barely overflows" models (Mixtral on 32GB), skip preloading —
    // layers will be loaded lazily via ensure_layer_loaded on first use,
    // then kept resident (not released) for subsequent tokens.
    // Expert-streaming never preloads — experts are loaded on demand.
    let any_streaming = expert_streaming || dense_ffn_streaming;
    if !any_streaming && should_preload {
        prefetch_state.preload_ram_layers();
    } else if keep_resident && !any_streaming {
        tracing::info!(
            "Skipping preload: model {:.1} GB exceeds preload threshold {:.1} GB (will load lazily)",
            model_total_bytes as f64 / (1u64 << 30) as f64,
            (total_ram.saturating_sub(6 * (1 << 30))) as f64 / (1u64 << 30) as f64,
        );
    }

    let nvme_tensor_count = prefetch_state.tensor_map.len();
    let nvme_layer_count = prefetch_state.layer_regions.len();
    tracing::info!(
        "Prefetch state: {nvme_tensor_count} tensors across {nvme_layer_count} layers"
    );

    // Pass PrefetchState to cb_eval callback
    let state_ptr = Arc::into_raw(prefetch_state.clone()) as *mut std::ffi::c_void;

    let kv_quant = plan.kv_cache_plan.kv_quantization;
    // Expert-streaming with use_mmap=false + pool buffer keeps Metal working set small
    // enough for full batch size. No n_batch reduction needed.
    let effective_batch = config.n_batch;
    let mut ctx = LlamaContext::new_with_callback_and_kv(
        &model,
        config.n_ctx,
        effective_batch,
        config.n_threads,
        Some(eval_callback),
        state_ptr,
        kv_quant,
    )?;

    let mut sampler = LlamaSampler::new(&config.sampling);

    // KV cache manager for windowed compaction (Phase 3b)
    let mut kv_manager = if plan.kv_cache_plan.warm_window_tokens > 0 {
        Some(crate::cache::kv_cache::KvCacheManager::new(
            plan.kv_cache_plan.hot_window_tokens,
        ))
    } else {
        None
    };

    // Tokenize prompt
    let tokens = model.tokenize(prompt, true, true);
    let prompt_len = tokens.len() as u32;
    anyhow::ensure!(!tokens.is_empty(), "Prompt tokenized to zero tokens");

    // Enable I/O tracing for streaming/expert-streaming diagnostics.
    if !keep_resident || any_streaming {
        prefetch_state.enable_trace();
    }

    // Hybrid residency: keep first N layers' FFN permanently in RAM.
    // Uses the unified MemoryBudget (already accounts for GPU, KV, Metal, OS, pool).
    if dense_ffn_streaming {
        let pool_size = prefetch_state
            .expert_pool
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, |p| p.pool_size as u64);
        let budget = MemoryBudget::compute(
            total_ram,
            gpu_bytes,
            false, // use_mmap=false for dense FFN streaming
            num_layers,
            num_kv_heads,
            head_dim,
            config.n_ctx,
            plan.kv_cache_plan.kv_quantization,
        )
        .with_pool(pool_size);
        let resident_budget = budget.available;

        // Compute per-layer FFN size from first layer with layouts
        let per_layer_ffn: u64 = prefetch_state
            .dense_ffn_layouts
            .values()
            .next()
            .map(|layouts| layouts.iter().map(|l| l.size as u64).sum())
            .unwrap_or(0);

        if per_layer_ffn > 0 {
            let max_resident = (resident_budget / per_layer_ffn) as u32;
            // Cap conservatively: Metal needs significant headroom for compute
            // buffers, KV cache, and page cache. On 32 GB M1 Max, 10+ resident
            // layers causes memory pressure that slows Metal compute more than
            // the I/O savings. Scale with available memory.
            let num_resident = max_resident.min(num_layers / 4);
            // Only activate residency when available memory is >50% of total RAM.
            // On 32 GB M1 Max, residency causes memory pressure that slows Metal
            // compute more than the I/O savings. Needs 64 GB+ to be beneficial.
            // Residency needs abundant headroom — Metal compute buffers, KV cache,
            // and page cache all compete with resident data. On 32 GB, even 7 GB
            // of resident data causes measurable compute slowdown.
            let min_available_for_residency = total_ram * 75 / 100;
            if num_resident >= 4 && budget.available > min_available_for_residency {
                if let Some((base, size, layers, offsets)) =
                    prefetch_state.activate_resident_ffn(num_resident)
                {
                    let state_mut = unsafe {
                        &mut *(Arc::as_ptr(&prefetch_state) as *mut PrefetchState)
                    };
                    state_mut.resident_ffn_base = base;
                    state_mut.resident_ffn_size = size;
                    state_mut.resident_ffn_layers = layers;
                    state_mut.resident_ffn_offsets = offsets;
                }
            }
        }
    }

    // Eagerly prefetch NVMe layers before the first forward pass.
    if expert_streaming {
        prefetch_state.warm_cache_from_coactivation();
    } else if dense_ffn_streaming {
        // Pre-load the first streaming layers' FFN data so the initial eval_callback
        // doesn't stall. Skip resident layers (already loaded).
        let first_streaming = prefetch_state.resident_ffn_layers.len() as u32;
        for layer in first_streaming..(first_streaming + 4).min(num_layers) {
            if prefetch_state.dense_ffn_layouts.contains_key(&layer) {
                prefetch_state.prefetch_dense_ffn(layer);
            }
        }
    } else {
        prefetch_state.prefetch_all_nvme();
    }

    if keep_resident && !any_streaming {
        // Keep-resident mode: disable the eval callback immediately. Layer data
        // is provided by llama.cpp's own mmap mechanism (use_mmap=true) — our
        // buffer's posix_memalign pages are overlaid with mmap file pages during
        // model loading. Calling pread would REPLACE these efficient file-backed
        // pages with anonymous pages, increasing memory pressure.
        //
        // The prefetch_all_nvme above still runs to ensure NVMe layers are loaded
        // via pread (those might not have mmap pages committed). The callback is
        // disabled so it won't block the forward pass.
        tracing::info!("Keep-resident mode; disabling eval callback");
        prefetch_state
            .prefetch_enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    // Process prompt
    let prompt_start = Instant::now();
    let prompt_batch = effective_batch as usize;
    for chunk in tokens.chunks(prompt_batch) {
        ctx.decode(chunk)?;
    }
    let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;

    // Set KV cache manager position after prompt processing
    if let Some(ref mut kv_mgr) = kv_manager {
        kv_mgr.set_position(prompt_len);
    }

    // Generation loop with active prefetch/release
    let mut generated_text = String::new();
    let mut n_generated: u32 = 0;
    let gen_start = Instant::now();

    for _ in 0..config.sampling.max_tokens {
        let token_id = sampler.sample(&mut ctx, -1);
        let is_eog = model.is_eog(token_id);
        let is_control = model.is_control(token_id);
        let piece = model.token_to_piece(token_id);

        n_generated += 1;
        generated_text.push_str(&piece);

        let elapsed = gen_start.elapsed().as_secs_f64();
        let tok_per_sec = if elapsed > 0.0 {
            n_generated as f64 / elapsed
        } else {
            0.0
        };

        telemetry.emit(TelemetryEvent::TokenGenerated {
            tok_per_sec,
            token: piece.clone(),
        });

        if token_tx
            .send(GeneratedToken {
                text: piece.clone(),
                token_id,
                tok_per_sec,
                is_eog,
            })
            .is_err()
        {
            break;
        }

        if is_eog {
            break;
        }

        if is_control
            && (piece.contains("end")
                || piece.contains("im_end")
                || piece.contains("eot")
                || piece.contains("stop")
                || piece.contains("start_of_role")
                || piece.contains("im_start")
                || piece.contains("INST"))
        {
            break;
        }

        if should_stop_early(&generated_text, &config.sampling.stop_sequences) {
            break;
        }

        // KV cache compaction for long-context inference
        if let Some(ref mut kv_mgr) = kv_manager {
            kv_mgr.advance(&ctx);
        }

        // In standard streaming mode, request NVMe layers before next forward pass.
        // Keep-resident and expert-streaming modes don't need this — keep-resident
        // has all data loaded, expert-streaming loads experts via eval_callback.
        if !keep_resident && !any_streaming {
            prefetch_state.prefetch_all_nvme();
        }

        // Dense FFN-streaming: pre-submit early STREAMING layer prefetches before the
        // next forward pass. At the token boundary, the I/O pipe goes idle. Priming
        // the first streaming layers keeps the pipe busy through the sampling gap.
        // Resident layers are skipped (always ready).
        if dense_ffn_streaming {
            let first_streaming = prefetch_state.resident_ffn_layers.len() as u32;
            let status = prefetch_state.layer_status.lock().unwrap();
            let mut to_prefetch = Vec::new();
            for layer in first_streaming..(first_streaming + 4).min(num_layers) {
                if !prefetch_state.dense_ffn_layouts.contains_key(&layer) {
                    continue;
                }
                let s = status.get(&layer).copied();
                if s != Some(LayerStatus::Loaded) && s != Some(LayerStatus::Loading) {
                    to_prefetch.push(layer);
                }
            }
            drop(status);
            for layer in to_prefetch {
                prefetch_state.prefetch_dense_ffn(layer);
            }
        }

        let decode_start = std::time::Instant::now();
        ctx.decode(&[token_id])?;
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        prefetch_state.record_decode(decode_ms);
    }

    // Print I/O trace summary (streaming mode only)
    prefetch_state.print_trace_summary();

    // Stop I/O pool and clean up
    prefetch_state.stop_io_pool();

    // Clean up the Arc we leaked into the callback
    unsafe {
        Arc::from_raw(state_ptr as *const PrefetchState);
    }

    let perf = ctx.perf();
    let total_gen_time = gen_start.elapsed().as_secs_f64();
    let avg_tps = if total_gen_time > 0.0 {
        n_generated as f64 / total_gen_time
    } else {
        0.0
    };

    Ok(GenerationResult {
        text: generated_text,
        tokens_generated: n_generated,
        prompt_tokens: prompt_len,
        tok_per_sec_avg: avg_tps,
        prompt_eval_ms: prompt_ms,
        perf,
    })
}

/// Query total physical RAM via sysctl (macOS).
fn total_physical_memory() -> u64 {
    unsafe {
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = b"hw.memsize\0";
        libc::sysctlbyname(
            name.as_ptr() as *const i8,
            &mut size as *mut u64 as *mut libc::c_void,
            &mut len as *mut usize,
            std::ptr::null_mut(),
            0,
        );
        if size == 0 { 32 * (1 << 30) } else { size }
    }
}

fn num_performance_cores() -> i32 {
    crate::profiler::cpu::sysctl_u32("hw.perflevel0.logicalcpu")
        .map(|n| n as i32)
        .unwrap_or_else(|_| {
            std::thread::available_parallelism()
                .map(|n| (n.get() / 2).max(1) as i32)
                .unwrap_or(4)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::model::gguf::{GgmlType, TensorInfo};

    fn make_gguf(layers: u32, tensors_per_layer: u32) -> GgufFile {
        let mut tensors = Vec::new();
        for l in 0..layers {
            for i in 0..tensors_per_layer {
                let name = format!("blk.{l}.tensor_{i}.weight");
                tensors.push(TensorInfo {
                    name,
                    dimensions: vec![4096, 4096],
                    dtype: GgmlType::Q4K,
                    offset: 0,
                    size_bytes: 1 << 26,
                    layer_index: Some(l),
                });
            }
        }
        GgufFile {
            version: 3,
            metadata: Default::default(),
            tensors,
            data_offset: 0,
        }
    }

    fn make_plan(assignments: HashMap<String, StorageTier>) -> PlacementPlan {
        PlacementPlan {
            model_id: "test".into(),
            hardware_profile_hash: "".into(),
            tier_assignments: assignments,
            prefetch_schedule: PrefetchSchedule { layer_prefetches: vec![] },
            estimated_tok_per_sec: 0.0,
            estimated_time_to_first_token: 0.0,
            kv_cache_plan: KvCachePlan {
                hot_window_tokens: 0, warm_window_tokens: 0,
                hot_tier: StorageTier::Gpu, warm_tier: StorageTier::Ram,
                hot_bytes: 0, warm_bytes: 0, kv_quantization: None,
            },
            experience_tier: ExperienceTier::Fast,
            inference_mode: InferenceMode::FullStreaming,
        }
    }

    #[test]
    fn test_gpu_layers_offloads_all() {
        // On unified memory, all layers should be GPU-offloaded
        // regardless of individual tensor tier assignments
        let gguf = make_gguf(10, 3);
        let mut assignments = HashMap::new();
        for t in &gguf.tensors {
            assignments.insert(t.name.clone(), StorageTier::Gpu);
        }
        let plan = make_plan(assignments);
        // 10 layers (0-9) + 1 output = 11
        assert_eq!(gpu_layers_from_placement(&plan, &gguf, u64::MAX), 11);
    }

    #[test]
    fn test_gpu_layers_mixed_tiers_stops_at_nvme() {
        // Only offload layers before the first NVMe layer
        // Layers 0-5 on GPU, layers 6-9 on NVMe
        let gguf = make_gguf(10, 3);
        let mut assignments = HashMap::new();
        for t in &gguf.tensors {
            let tier = if t.layer_index.unwrap() < 6 {
                StorageTier::Gpu
            } else {
                StorageTier::Nvme
            };
            assignments.insert(t.name.clone(), tier);
        }
        let plan = make_plan(assignments);
        // Layers 0-5 on GPU (6 layers)
        assert_eq!(gpu_layers_from_placement(&plan, &gguf, u64::MAX), 6);
    }

    #[test]
    fn test_gpu_layers_empty() {
        let gguf = GgufFile {
            version: 3,
            metadata: Default::default(),
            tensors: vec![],
            data_offset: 0,
        };
        let plan = make_plan(HashMap::new());
        assert_eq!(gpu_layers_from_placement(&plan, &gguf, u64::MAX), 0);
    }

    #[test]
    fn test_should_stop_early() {
        let empty_stops: Vec<String> = vec![];
        // ChatML / Qwen
        assert!(should_stop_early("Here is the answer.<|im_end|>", &empty_stops));
        assert!(should_stop_early("Here is the answer.\n<|im_start|>user", &empty_stops));
        // Mistral
        assert!(should_stop_early("Result: 42</s>", &empty_stops));
        assert!(should_stop_early("Result: 42\n[INST]", &empty_stops));
        // Granite
        assert!(should_stop_early("Result: 42<|end_of_text|>", &empty_stops));
        assert!(should_stop_early("Result: 42<|start_of_role|>user", &empty_stops));
        // Tool call closure
        assert!(should_stop_early("<tool_call>\n{\"name\": \"test\", \"arguments\": {}}\n</tool_call>", &empty_stops));
        // Tool call partial should NOT stop
        assert!(!should_stop_early("[TOOL_CALLS][\n   {\n     \"name\":", &empty_stops));
        assert!(!should_stop_early("[TOOL_CALLS][\n   {\n     \"name\": \"get_devices_by_name\",\n     \"arguments\": {", &empty_stops));
        // Complete [TOOL_CALLS] array should stop
        assert!(should_stop_early("[TOOL_CALLS][\n   {\n     \"name\": \"get_devices_by_name\",\n     \"arguments\": {}\n   }\n]", &empty_stops));
        // Thinking loop prevention
        assert!(should_stop_early("<think>analyzing</think>\nDone!\n<think>wait", &empty_stops));
        // Custom stop
        let custom_stops = vec!["### Human:".to_string()];
        assert!(should_stop_early("Done.\n### Human:", &custom_stops));
        // Normal text should not stop
        assert!(!should_stop_early("Here are 3 weather stations with the lowest temperature:\n1. Station A", &empty_stops));
    }

    #[test]
    fn test_compute_gpu_budget_applies_kv_quantization_at_32k() {
        use crate::profiler::types::{CpuProfile, GpuBackend, GpuProfile, MemoryProfile, SystemInfo};
        use chrono::Utc;

        let hw = HardwareProfile {
            timestamp: Utc::now(),
            system: SystemInfo {
                os: "macos".into(),
                arch: "aarch64".into(),
                machine_model: "Mac16,7".into(),
                total_cores: 14,
            },
            memory: MemoryProfile {
                total_bytes: 24 * (1 << 30),
                available_bytes: 20 * (1 << 30),
                bandwidth_bytes_per_sec: 273_000_000_000,
                is_unified: true,
            },
            gpu: Some(GpuProfile {
                name: "Apple M4 Pro".into(),
                vram_bytes: 19069 * 1024 * 1024, // 19069 MiB Metal working set on 24 GB M4 Pro
                bandwidth_bytes_per_sec: 273_000_000_000,
                fp16_tflops: 20.0,
                backend: GpuBackend::Metal,
            }),
            storage: vec![],
            cpu: CpuProfile {
                model_name: "Apple M4 Pro".into(),
                cores_performance: 10,
                cores_efficiency: 4,
                has_amx: true,
                has_neon: true,
                has_avx512: false,
                has_avx2: false,
                int8_gflops: 1000.0,
            },
        };

        // 26B Gemma-class metadata where unquantized 32k FP16 KV would be ~6.44 GiB
        let meta = ModelMetadata {
            architecture: "gemma4".into(),
            parameter_count: 26_000_000_000,
            context_length: 131072,
            embedding_dim: 4096,
            num_layers: 42,
            num_heads: 32,
            num_kv_heads: 16,
            vocab_size: 262144,
            quantization: Some("Q4_K_M".into()),
            is_moe: false,
            num_experts: None,
            num_experts_used: None,
        };

        let budget_32k = compute_gpu_budget(&hw, &meta, 32768);
        // Raw FP16 KV at 32k here is 10.5 GiB, which previously crushed GPU budget down to ~2.4 GiB.
        // With Q4_0 KV scaling (0.28x -> ~2.94 GiB), GPU budget stays above 9.5 GiB on 24 GB M4 Pro.
        assert!(
            budget_32k > (95 * (1 << 30)) / 10,
            "Expected > 9.5 GiB GPU budget at 32k context with Q4_0 KV, got {} bytes",
            budget_32k
        );
    }
}
