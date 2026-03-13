use std::collections::HashMap;

use crate::model::gguf::TensorInfo;
use crate::model::metadata::ModelMetadata;
use crate::profiler::types::HardwareProfile;
use crate::scheduler::types::*;

/// Build a prefetch schedule based on tensor placement.
///
/// For each layer, determines which NVMe-resident tensors should be prefetched
/// and how many layers ahead the prefetch should start.
pub fn build_prefetch_schedule(
    tier_assignments: &HashMap<String, StorageTier>,
    tensors: &[TensorInfo],
    metadata: &ModelMetadata,
    hardware: &HardwareProfile,
) -> PrefetchSchedule {
    let num_layers = metadata.num_layers as usize;
    if num_layers == 0 {
        return PrefetchSchedule {
            layer_prefetches: vec![],
        };
    }

    let gpu_bw = hardware
        .gpu
        .as_ref()
        .map_or(1e9, |g| g.bandwidth_bytes_per_sec as f64);
    let nvme_bw = hardware
        .storage
        .first()
        .map(|s| s.sequential_read.peak_sequential as f64)
        .unwrap_or(3e9);

    // Group tensors by layer
    let mut layer_tensors: Vec<Vec<&TensorInfo>> = vec![Vec::new(); num_layers];
    let mut non_layer_tensors = Vec::new();

    for t in tensors {
        match t.layer_index {
            Some(idx) if (idx as usize) < num_layers => {
                layer_tensors[idx as usize].push(t);
            }
            _ => non_layer_tensors.push(t),
        }
    }

    // Estimate compute time per layer (GPU tensors / GPU bandwidth)
    let compute_time_per_layer: Vec<f64> = layer_tensors
        .iter()
        .map(|layer| {
            let gpu_bytes: u64 = layer
                .iter()
                .filter(|t| tier_assignments.get(&t.name) == Some(&StorageTier::Gpu))
                .map(|t| t.size_bytes)
                .sum();
            gpu_bytes as f64 / gpu_bw
        })
        .collect();

    let mut layer_prefetches = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let mut ops = Vec::new();

        for t in &layer_tensors[layer_idx] {
            let tier = tier_assignments.get(&t.name).unwrap_or(&StorageTier::Nvme);
            if *tier != StorageTier::Nvme {
                continue;
            }

            let load_time = t.size_bytes as f64 / nvme_bw;

            // How many layers ahead do we need to start prefetching?
            let lead_time = if layer_idx > 0 {
                let prev_compute = compute_time_per_layer[layer_idx - 1];
                if prev_compute >= load_time {
                    1 // Previous layer's compute hides this load
                } else if prev_compute > 0.0 {
                    (load_time / prev_compute).ceil() as u32
                } else {
                    1
                }
            } else {
                0 // First layer — must be loaded at start
            };

            ops.push(PrefetchOp {
                tensor_name: t.name.clone(),
                source_tier: StorageTier::Nvme,
                target_tier: StorageTier::Ram,
                size_bytes: t.size_bytes,
                lead_time_layers: lead_time,
            });
        }

        layer_prefetches.push(ops);
    }

    PrefetchSchedule { layer_prefetches }
}
