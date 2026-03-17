use serde::{Deserialize, Serialize};

use crate::model::gguf::GgufFile;
use crate::model::metadata::ModelMetadata;
use crate::model::tensor_role::TensorRole;
use crate::profiler::types::HardwareProfile;
use crate::scheduler::placement::summarize_placement;
use crate::scheduler::types::*;

const MOE_CACHE_HIT_RATE: f64 = 0.965; // From PowerInfer-2 research
const SYNC_OVERHEAD_US: f64 = 50.0; // CPU-GPU sync per layer

/// Pre-download performance prediction for a model on specific hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypuraEstimate {
    pub model_id: String,
    pub estimated_tok_per_sec_interactive: f64,
    pub estimated_tok_per_sec_batched: f64,
    pub placement_summary: PlacementSummary,
    pub max_context_before_spill: u32,
    pub disk_read_per_token_bytes: u64,
    pub experience_tier: ExperienceTier,
    pub confidence: EstimateConfidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EstimateConfidence {
    Measured,
    Predicted,
    Interpolated,
}

pub fn estimate_performance(
    model: &GgufFile,
    metadata: &ModelMetadata,
    hardware: &HardwareProfile,
    placement: &PlacementPlan,
) -> anyhow::Result<HypuraEstimate> {
    let gpu_bw = hardware
        .gpu
        .as_ref()
        .map_or(1e9, |g| g.bandwidth_bytes_per_sec as f64);
    let ram_bw = hardware.memory.bandwidth_bytes_per_sec as f64;
    let nvme_bw = hardware
        .storage
        .first()
        .map(|s| s.sequential_read.peak_sequential as f64)
        .unwrap_or(3e9);

    let experts_per_token = metadata.num_experts_used.unwrap_or(1);
    let total_experts = metadata.num_experts.unwrap_or(1);
    let num_layers = metadata.num_layers;

    // Group tensors by layer and compute per-layer latency
    let mut total_latency = 0.0;
    let mut total_nvme_bytes_per_token: f64 = 0.0;
    let mut prev_compute_time = 0.0;

    // Process layer by layer
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("blk.{layer_idx}.");
        let layer_tensors: Vec<_> = model
            .tensors
            .iter()
            .filter(|t| t.name.starts_with(&layer_prefix))
            .collect();

        let mut layer_compute_time = 0.0;
        let mut layer_io_time = 0.0;

        for t in &layer_tensors {
            let role = TensorRole::from_name(&t.name);
            let freq = role.access_frequency(experts_per_token, total_experts);
            let tier = placement
                .tier_assignments
                .get(&t.name)
                .unwrap_or(&StorageTier::Nvme);

            match tier {
                StorageTier::Gpu => {
                    layer_compute_time += t.size_bytes as f64 * freq / gpu_bw;
                }
                StorageTier::Ram => {
                    layer_compute_time += t.size_bytes as f64 * freq / ram_bw;
                }
                StorageTier::Nvme => {
                    let effective_freq = if metadata.is_moe {
                        match role {
                            TensorRole::MoeExpert { .. } | TensorRole::MoeFusedExperts => {
                                freq * (1.0 - MOE_CACHE_HIT_RATE)
                            }
                            _ => freq,
                        }
                    } else {
                        freq
                    };

                    let io = t.size_bytes as f64 * effective_freq / nvme_bw;
                    total_nvme_bytes_per_token += t.size_bytes as f64 * effective_freq;

                    // Check if prefetchable (previous layer's compute hides the I/O)
                    if prev_compute_time >= io {
                        // I/O hidden by prefetch — don't add to io_time
                    } else {
                        layer_io_time += io - prev_compute_time.min(io);
                    }
                }
            }
        }

        // Per-token latency for this layer: max(compute, io) + sync
        let layer_latency = layer_compute_time.max(layer_io_time) + SYNC_OVERHEAD_US * 1e-6;
        total_latency += layer_latency;
        prev_compute_time = layer_compute_time;
    }

    // Also account for non-layer tensors (embedding, output head)
    for t in &model.tensors {
        if t.layer_index.is_some() {
            continue;
        }
        let role = TensorRole::from_name(&t.name);
        let freq = role.access_frequency(experts_per_token, total_experts);
        let tier = placement
            .tier_assignments
            .get(&t.name)
            .unwrap_or(&StorageTier::Nvme);

        let transfer = match tier {
            StorageTier::Gpu => t.size_bytes as f64 * freq / gpu_bw,
            StorageTier::Ram => t.size_bytes as f64 * freq / ram_bw,
            StorageTier::Nvme => {
                total_nvme_bytes_per_token += t.size_bytes as f64 * freq;
                t.size_bytes as f64 * freq / nvme_bw
            }
        };
        total_latency += transfer;
    }

    let interactive_tps = if total_latency > 0.0 {
        1.0 / total_latency
    } else {
        0.0
    };

    // Batched: compute scales linearly, I/O stays constant
    let batch_size = 8.0;
    let batched_tps = interactive_tps * batch_size * 0.7; // ~70% efficiency for batching

    // Max context before KV spill to NVMe
    let kv_per_token = estimate_kv_per_token(metadata);
    let available_for_kv = hardware.memory.total_bytes.saturating_sub(
        model.total_tensor_bytes().min(
            hardware.memory.total_bytes.saturating_sub(OS_OVERHEAD_BYTES),
        ),
    );
    let max_context = if kv_per_token > 0 {
        (available_for_kv / kv_per_token) as u32
    } else {
        metadata.context_length
    };

    let placement_summary = summarize_placement(&placement.tier_assignments, &model.tensors);
    let experience_tier = ExperienceTier::from_tok_per_sec(interactive_tps);

    Ok(HypuraEstimate {
        model_id: placement.model_id.clone(),
        estimated_tok_per_sec_interactive: interactive_tps,
        estimated_tok_per_sec_batched: batched_tps,
        placement_summary,
        max_context_before_spill: max_context,
        disk_read_per_token_bytes: total_nvme_bytes_per_token as u64,
        experience_tier,
        confidence: EstimateConfidence::Predicted,
    })
}

const OS_OVERHEAD_BYTES: u64 = 2 * (1 << 30);

fn estimate_kv_per_token(metadata: &ModelMetadata) -> u64 {
    if metadata.num_heads == 0 || metadata.embedding_dim == 0 {
        return 0;
    }
    let head_dim = metadata.embedding_dim as u64 / metadata.num_heads as u64;
    2 * metadata.num_layers as u64 * metadata.num_kv_heads as u64 * head_dim * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_per_token() {
        let metadata = ModelMetadata {
            architecture: "llama".into(),
            parameter_count: 7_000_000_000,
            context_length: 4096,
            embedding_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            quantization: None,
            is_moe: false,
            num_experts: None,
            num_experts_used: None,
        };

        let kv = estimate_kv_per_token(&metadata);
        // 2 * 32 layers * 8 kv_heads * 128 head_dim * 2 bytes = 131072
        assert_eq!(kv, 131072);
    }
}
