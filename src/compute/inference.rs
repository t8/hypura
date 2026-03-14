use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::compute::ffi::*;
use crate::model::gguf::GgufFile;
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

/// Derive `n_gpu_layers` from a PlacementPlan.
///
/// Only offload layers to GPU where NO tensor is on NVMe. Layers with any
/// NVMe tensor stay on CPU — the custom buffer type + eval callback handles
/// prefetch/release for those layers. This keeps Metal's working set within
/// `recommendedMaxWorkingSetSize` and prevents GPU OOM on models that exceed
/// physical memory.
///
/// Since llama.cpp's `n_gpu_layers` is sequential (layers 0..N go to GPU),
/// we offload up to (but not including) the first layer with an NVMe tensor.
pub fn gpu_layers_from_placement(plan: &PlacementPlan, gguf: &GgufFile) -> i32 {
    let mut max_layer: i32 = -1;
    let mut first_nvme_layer: Option<u32> = None;

    for t in &gguf.tensors {
        if let Some(layer_idx) = t.layer_index {
            max_layer = max_layer.max(layer_idx as i32);
            if plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme) {
                first_nvme_layer = Some(match first_nvme_layer {
                    Some(existing) => existing.min(layer_idx),
                    None => layer_idx,
                });
            }
        }
    }

    if max_layer < 0 {
        return 0;
    }

    match first_nvme_layer {
        Some(nvme_start) => {
            // Offload layers 0..(nvme_start-1) to GPU.
            // +1 for the output layer llama.cpp counts separately.
            nvme_start as i32 + 1
        }
        None => {
            // No NVMe tensors — offload everything.
            max_layer + 1 + 1
        }
    }
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
    let tokens = model.tokenize(prompt, true);
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
            text: piece,
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
    use crate::compute::nvme_backend::{
        build_override_patterns, eval_callback, HypuraBuftController, PrefetchState,
    };

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
    let (_patterns, overrides) = build_override_patterns(plan, gguf, controller.buft_ptr());

    let nvme_count = plan
        .tier_assignments
        .values()
        .filter(|t| **t == StorageTier::Nvme)
        .count();
    tracing::info!("NVMe scheduling: {nvme_count} tensors on custom buffer type");

    // Load model with overrides
    let model = LlamaModel::load_with_overrides(
        model_path,
        n_gpu_layers,
        true,
        overrides.as_ptr(),
    )?;

    // Build prefetch state with layer groupings and file offsets
    let num_layers = model.n_layers() as u32;
    let prefetch_state = controller.build_prefetch_state(gguf, num_layers);

    // Open NVMe file descriptor for F_NOCACHE reads during prefetch
    prefetch_state.open_nvme_fd()?;

    // Start background prefetch thread for async layer preloading
    let prefetch_handle = prefetch_state.start_prefetch_thread();

    let nvme_tensor_count = prefetch_state.tensor_map.len();
    let nvme_layer_count = prefetch_state.layer_regions.len();
    tracing::info!(
        "Prefetch state: {nvme_tensor_count} tensors across {nvme_layer_count} layers"
    );

    // Pass PrefetchState to cb_eval callback
    let state_ptr = Arc::into_raw(prefetch_state.clone()) as *mut std::ffi::c_void;

    let mut ctx = LlamaContext::new_with_callback(
        &model,
        config.n_ctx,
        config.n_batch,
        config.n_threads,
        Some(eval_callback),
        state_ptr,
    )?;

    let mut sampler = LlamaSampler::new(&config.sampling);

    // Tokenize prompt
    let tokens = model.tokenize(prompt, true);
    let prompt_len = tokens.len() as u32;
    anyhow::ensure!(!tokens.is_empty(), "Prompt tokenized to zero tokens");

    // Process prompt
    let prompt_start = Instant::now();
    let batch_size = config.n_batch as usize;
    for chunk in tokens.chunks(batch_size) {
        ctx.decode(chunk)?;
    }
    let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;

    // Generation loop with active prefetch/release
    let mut generated_text = String::new();
    let mut n_generated: u32 = 0;
    let gen_start = Instant::now();

    for _ in 0..config.sampling.max_tokens {
        let token_id = sampler.sample(&mut ctx, -1);
        let is_eog = model.is_eog(token_id);
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
                text: piece,
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

        ctx.decode(&[token_id])?;
    }

    // Stop prefetch thread and clean up
    prefetch_state.stop_prefetch_thread();
    if let Some(handle) = prefetch_handle {
        let _ = handle.join();
    }

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
                hot_bytes: 0, warm_bytes: 0,
            },
            experience_tier: ExperienceTier::Fast,
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
        assert_eq!(gpu_layers_from_placement(&plan, &gguf), 11);
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
        // Layers 0-5 on GPU (6 layers) + 1 output = 7
        assert_eq!(gpu_layers_from_placement(&plan, &gguf), 7);
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
        assert_eq!(gpu_layers_from_placement(&plan, &gguf), 0);
    }
}
