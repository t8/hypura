use serde::{Deserialize, Serialize};

use crate::model::gguf::GgufFile;

/// Unified model metadata extracted from any supported format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub architecture: String,
    pub parameter_count: u64,
    pub context_length: u32,
    pub embedding_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub vocab_size: u32,
    pub quantization: Option<String>,
    pub is_moe: bool,
    pub num_experts: Option<u32>,
    pub num_experts_used: Option<u32>,
}

impl ModelMetadata {
    /// Extract metadata from a parsed GGUF file.
    pub fn from_gguf(gguf: &GgufFile) -> anyhow::Result<Self> {
        let arch = gguf
            .get_string("general.architecture")
            .unwrap_or("unknown")
            .to_string();

        let num_layers = gguf
            .get_u32(&format!("{arch}.block_count"))
            .or_else(|| gguf.get_u32("block_count"))
            .unwrap_or(0);

        let embedding_dim = gguf
            .get_u32(&format!("{arch}.embedding_length"))
            .or_else(|| gguf.get_u32("embedding_length"))
            .unwrap_or(0);

        let num_heads = gguf
            .get_u32(&format!("{arch}.attention.head_count"))
            .or_else(|| gguf.get_u32("attention.head_count"))
            .unwrap_or(0);

        let num_kv_heads = gguf
            .get_u32(&format!("{arch}.attention.head_count_kv"))
            .or_else(|| gguf.get_u32("attention.head_count_kv"))
            .unwrap_or(num_heads);

        let vocab_size = gguf
            .get_u32(&format!("{arch}.vocab_size"))
            .or_else(|| gguf.get_u32("vocab_size"))
            .unwrap_or(0);

        let context_length = gguf
            .get_u32(&format!("{arch}.context_length"))
            .or_else(|| gguf.get_u32("context_length"))
            .unwrap_or(0);

        let num_experts = gguf
            .get_u32(&format!("{arch}.expert_count"))
            .or_else(|| gguf.get_u32("expert_count"));

        let num_experts_used = gguf
            .get_u32(&format!("{arch}.expert_used_count"))
            .or_else(|| gguf.get_u32("expert_used_count"));

        let is_moe = num_experts.map_or(false, |n| n > 1);

        // Determine quantization from the dominant tensor type
        let quantization = detect_quantization(gguf);

        // Estimate parameter count from tensor sizes and quantization
        let parameter_count = estimate_parameters(gguf);

        Ok(Self {
            architecture: arch,
            parameter_count,
            context_length,
            embedding_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            quantization,
            is_moe,
            num_experts,
            num_experts_used,
        })
    }
}

fn detect_quantization(gguf: &GgufFile) -> Option<String> {
    // Find the most common tensor dtype (excluding small tensors like norms)
    let mut type_counts = std::collections::HashMap::new();
    for t in &gguf.tensors {
        if t.size_bytes > 1024 * 1024 {
            // Only consider tensors > 1MB
            *type_counts.entry(format!("{:?}", t.dtype)).or_insert(0u32) += 1;
        }
    }
    type_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(dtype, _)| dtype)
}

fn estimate_parameters(gguf: &GgufFile) -> u64 {
    // Sum up total elements across all tensors
    gguf.tensors
        .iter()
        .map(|t| t.dimensions.iter().product::<u64>())
        .sum()
}
