use serde::{Deserialize, Serialize};

use crate::scheduler::types::{ExperienceTier, PlacementSummary};

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
    /// Based on crowdsourced data for this exact (hardware, model) pair
    Measured,
    /// Based on the analytical model only
    Predicted,
    /// Interpolated from similar hardware
    Interpolated,
}

pub fn estimate() -> anyhow::Result<HypuraEstimate> {
    anyhow::bail!("Not yet implemented: estimator")
}
