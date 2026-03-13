use serde::{Deserialize, Serialize};

/// Classification of a tensor's role in the model, derived from its name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TensorRole {
    Embedding,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    FfnGate,
    FfnUp,
    FfnDown,
    MoeRouter,
    MoeExpert { expert_id: u32 },
    Norm,
    OutputHead,
    Other(String),
}

impl TensorRole {
    /// Parse tensor role from its GGUF name.
    ///
    /// Common patterns:
    ///   - `token_embd.weight` → Embedding
    ///   - `blk.N.attn_q.weight` → AttentionQuery
    ///   - `blk.N.attn_k.weight` → AttentionKey
    ///   - `blk.N.attn_v.weight` → AttentionValue
    ///   - `blk.N.attn_output.weight` → AttentionOutput
    ///   - `blk.N.ffn_gate.weight` → FfnGate
    ///   - `blk.N.ffn_up.weight` → FfnUp
    ///   - `blk.N.ffn_down.weight` → FfnDown
    ///   - `blk.N.ffn_gate_inp.weight` → MoeRouter
    ///   - `blk.N.ffn_gate_exps.weight` → MoeExpert
    ///   - `blk.N.attn_norm.weight` → Norm
    ///   - `output.weight` → OutputHead
    pub fn from_name(name: &str) -> Self {
        if name.contains("token_embd") {
            return Self::Embedding;
        }
        if name == "output.weight" || name == "lm_head.weight" {
            return Self::OutputHead;
        }
        if name.contains("_norm") || name.contains(".norm") {
            return Self::Norm;
        }
        if name.contains("ffn_gate_inp") || name.contains("ffn_gate_shexp") {
            return Self::MoeRouter;
        }
        if name.contains("_exps.") {
            let expert_id = parse_expert_id(name).unwrap_or(0);
            return Self::MoeExpert { expert_id };
        }
        if name.contains("attn_q") || name.contains("attn.q") {
            return Self::AttentionQuery;
        }
        if name.contains("attn_k") || name.contains("attn.k") {
            return Self::AttentionKey;
        }
        if name.contains("attn_v") || name.contains("attn.v") {
            return Self::AttentionValue;
        }
        if name.contains("attn_output") || name.contains("attn.o") {
            return Self::AttentionOutput;
        }
        if name.contains("ffn_gate") {
            return Self::FfnGate;
        }
        if name.contains("ffn_up") {
            return Self::FfnUp;
        }
        if name.contains("ffn_down") {
            return Self::FfnDown;
        }

        Self::Other(name.to_string())
    }

    /// Whether this tensor is accessed every token (vs. probabilistically for MoE experts).
    pub fn access_frequency(&self, experts_per_token: u32, total_experts: u32) -> f64 {
        match self {
            Self::MoeExpert { .. } => {
                if total_experts > 0 {
                    experts_per_token as f64 / total_experts as f64
                } else {
                    1.0
                }
            }
            _ => 1.0,
        }
    }
}

fn parse_expert_id(name: &str) -> Option<u32> {
    // Patterns like "blk.0.ffn_gate_exps.0.weight" or expert ID embedded in name
    name.split('.')
        .filter_map(|part| part.parse::<u32>().ok())
        .nth(1) // second numeric component (first is layer index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_classification() {
        assert_eq!(
            TensorRole::from_name("token_embd.weight"),
            TensorRole::Embedding
        );
        assert_eq!(
            TensorRole::from_name("blk.0.attn_q.weight"),
            TensorRole::AttentionQuery
        );
        assert_eq!(
            TensorRole::from_name("blk.5.ffn_gate.weight"),
            TensorRole::FfnGate
        );
        assert_eq!(
            TensorRole::from_name("blk.3.attn_norm.weight"),
            TensorRole::Norm
        );
        assert_eq!(
            TensorRole::from_name("output.weight"),
            TensorRole::OutputHead
        );
        assert_eq!(
            TensorRole::from_name("blk.0.ffn_gate_inp.weight"),
            TensorRole::MoeRouter
        );
    }
}
