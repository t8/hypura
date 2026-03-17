use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Memory/storage tier for tensor placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageTier {
    /// Metal GPU (fastest, most limited)
    Gpu,
    /// System RAM
    Ram,
    /// NVMe SSD (largest, slowest)
    Nvme,
}

/// User experience classification based on predicted tok/s.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExperienceTier {
    /// >10 tok/s — real-time chat
    Fast,
    /// 2-10 tok/s — usable for coding, Q&A
    Usable,
    /// <2 tok/s — background tasks only
    Slow,
}

impl ExperienceTier {
    pub fn from_tok_per_sec(tps: f64) -> Self {
        if tps > 10.0 {
            Self::Fast
        } else if tps > 2.0 {
            Self::Usable
        } else {
            Self::Slow
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Fast => "FAST",
            Self::Usable => "USABLE",
            Self::Slow => "SLOW",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Fast => "suitable for real-time chat",
            Self::Usable => "suitable for coding, Q&A. Not real-time chat.",
            Self::Slow => "background tasks only",
        }
    }
}

/// Inference mode determined by model size, MoE structure, and hardware capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceMode {
    /// All tensors fit in GPU+RAM. No NVMe I/O.
    FullResident,
    /// NVMe spill is small enough to keep loaded after first pass.
    KeepResident,
    /// MoE model: non-expert tensors resident, experts streamed from NVMe on demand.
    ExpertStreaming,
    /// Heavy NVMe spill: all NVMe layers streamed, loaded/released per token.
    FullStreaming,
}

/// Complete placement plan for a model on specific hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub model_id: String,
    pub hardware_profile_hash: String,
    /// tensor_name → tier
    pub tier_assignments: HashMap<String, StorageTier>,
    pub prefetch_schedule: PrefetchSchedule,
    pub estimated_tok_per_sec: f64,
    pub estimated_time_to_first_token: f64,
    pub kv_cache_plan: KvCachePlan,
    pub experience_tier: ExperienceTier,
    pub inference_mode: InferenceMode,
}

/// Schedule for prefetching tensors from slower tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchSchedule {
    /// For each layer, which tensors to begin loading during the previous layer's compute.
    pub layer_prefetches: Vec<Vec<PrefetchOp>>,
}

/// A single prefetch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchOp {
    pub tensor_name: String,
    pub source_tier: StorageTier,
    pub target_tier: StorageTier,
    pub size_bytes: u64,
    /// How many layers ahead this prefetch should start.
    pub lead_time_layers: u32,
}

/// KV cache quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvQuantization {
    F16,
    Q8_0,
    Q4_0,
}

impl KvQuantization {
    /// Memory scale factor relative to F16.
    pub fn memory_scale(&self) -> f64 {
        match self {
            Self::F16 => 1.0,
            Self::Q8_0 => 0.53,
            Self::Q4_0 => 0.28,
        }
    }
}

/// Plan for KV cache allocation across tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCachePlan {
    pub hot_window_tokens: u32,
    pub warm_window_tokens: u32,
    pub hot_tier: StorageTier,
    pub warm_tier: StorageTier,
    pub hot_bytes: u64,
    pub warm_bytes: u64,
    /// Auto-selected KV quantization (None = F16 default).
    pub kv_quantization: Option<KvQuantization>,
}

/// Summary of placement for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementSummary {
    pub layers_on_gpu: u32,
    pub layers_in_ram: u32,
    pub layers_on_nvme: u32,
    pub total_gpu_bytes: u64,
    pub total_ram_bytes: u64,
    pub total_nvme_bytes: u64,
}
