/// Layout of a fused expert tensor — tracks per-expert regions within a single
/// tensor that contains all experts (e.g., Mixtral's `ffn_gate_exps.weight`
/// with dims [hidden, intermediate, num_experts]).
///
/// Each expert occupies a contiguous stride of `expert_stride` bytes starting
/// at `file_offset + expert_id * expert_stride` in the GGUF file, and at
/// `buffer_offset + expert_id * expert_stride` in the posix_memalign buffer.
#[derive(Debug, Clone)]
pub struct ExpertLayout {
    pub tensor_name: String,
    pub layer_index: u32,
    pub num_experts: u32,
    /// Size of one expert's data: `total_size / num_experts`
    pub expert_stride: usize,
    /// Absolute byte offset of the fused tensor in the GGUF file
    pub file_offset: u64,
    /// Offset of the fused tensor within the posix_memalign buffer
    pub buffer_offset: usize,
    /// Total size of the fused tensor (all experts)
    pub total_size: usize,
    /// Inverse permutation from optimized model: maps logical expert ID → physical
    /// position in the file. None means natural order. Set by loading a sidecar
    /// `.permutations.json` file produced by `hypura optimize`.
    pub expert_permutation: Option<Vec<u32>>,
}

/// Identifies which type of expert FFN tensor this is. Used as a cache key
/// dimension. Most architectures (Mixtral-style) split into 3 separate fused
/// tensors (Gate/Up/Down). Some architectures (gemma4) fuse Gate and Up into
/// a single tensor, leaving 2 tensors per layer (GateUp/Down).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertTensorType {
    Gate,
    Up,
    Down,
    /// Fused gate+up tensor (e.g., gemma4's `ffn_gate_up_exps.weight`).
    GateUp,
}

impl ExpertTensorType {
    /// Parse from a tensor name like `blk.N.ffn_gate_exps.weight`.
    /// Note: `ffn_gate_up_exps` must be checked before `ffn_gate_exps` and
    /// `ffn_up_exps` to avoid mismatches (the substring `ffn_gate_exps` does
    /// not appear in `ffn_gate_up_exps`, but being explicit guards against
    /// future naming variants).
    pub fn from_name(name: &str) -> Option<Self> {
        if name.contains("ffn_gate_up_exps") {
            Some(Self::GateUp)
        } else if name.contains("ffn_gate_exps") {
            Some(Self::Gate)
        } else if name.contains("ffn_up_exps") {
            Some(Self::Up)
        } else if name.contains("ffn_down_exps") {
            Some(Self::Down)
        } else {
            None
        }
    }
}

impl ExpertLayout {
    /// Absolute file offset for a specific expert's data.
    /// Applies permutation if the model was optimized with `hypura optimize`.
    pub fn expert_file_offset(&self, expert_id: u32) -> u64 {
        let physical = self.map_expert_id(expert_id);
        self.file_offset + (physical as u64) * (self.expert_stride as u64)
    }

    /// Buffer offset for a specific expert's data.
    /// Applies permutation if the model was optimized with `hypura optimize`.
    pub fn expert_buffer_offset(&self, expert_id: u32) -> usize {
        let physical = self.map_expert_id(expert_id);
        self.buffer_offset + (physical as usize) * self.expert_stride
    }

    /// Map logical expert ID to physical position via permutation.
    fn map_expert_id(&self, expert_id: u32) -> u32 {
        if let Some(ref perm) = self.expert_permutation {
            perm.get(expert_id as usize).copied().unwrap_or(expert_id)
        } else {
            expert_id
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_offsets() {
        let layout = ExpertLayout {
            tensor_name: "blk.30.ffn_gate_exps.weight".into(),
            layer_index: 30,
            num_experts: 8,
            expert_stride: 1024,
            file_offset: 10000,
            buffer_offset: 5000,
            total_size: 8192,
            expert_permutation: None,
        };

        assert_eq!(layout.expert_file_offset(0), 10000);
        assert_eq!(layout.expert_file_offset(3), 10000 + 3 * 1024);
        assert_eq!(layout.expert_buffer_offset(0), 5000);
        assert_eq!(layout.expert_buffer_offset(7), 5000 + 7 * 1024);
    }

    #[test]
    fn test_expert_tensor_type() {
        assert_eq!(
            ExpertTensorType::from_name("blk.0.ffn_gate_exps.weight"),
            Some(ExpertTensorType::Gate)
        );
        assert_eq!(
            ExpertTensorType::from_name("blk.0.ffn_up_exps.weight"),
            Some(ExpertTensorType::Up)
        );
        assert_eq!(
            ExpertTensorType::from_name("blk.0.ffn_down_exps.weight"),
            Some(ExpertTensorType::Down)
        );
        assert_eq!(
            ExpertTensorType::from_name("blk.0.ffn_gate_up_exps.weight"),
            Some(ExpertTensorType::GateUp)
        );
        assert_eq!(
            ExpertTensorType::from_name("blk.0.ffn_gate.weight"),
            None
        );
    }
}
