use std::collections::HashMap;

use good_lp::{constraint, default_solver, variable, variables, Expression, Solution, SolverModel};

use crate::model::gguf::{GgufFile, TensorInfo};
use crate::model::metadata::ModelMetadata;
use crate::model::tensor_role::TensorRole;
use crate::profiler::types::HardwareProfile;
use crate::scheduler::prefetch::build_prefetch_schedule;
use crate::scheduler::types::*;

const OS_OVERHEAD: u64 = 2 * (1 << 30); // 2 GiB reserved for macOS
const SYNC_OVERHEAD_PER_LAYER_US: f64 = 50.0; // 50μs CPU-GPU sync per layer

pub fn compute_placement(
    model: &GgufFile,
    hardware: &HardwareProfile,
) -> anyhow::Result<PlacementPlan> {
    compute_placement_with_context(model, hardware, 0)
}

/// Compute placement with a specific context length for KV cache headroom.
/// If `context_length` is 0, uses a sensible default (capped at 8192 to avoid
/// over-reserving KV headroom for models with very large max context).
pub fn compute_placement_with_context(
    model: &GgufFile,
    hardware: &HardwareProfile,
    context_length: u32,
) -> anyhow::Result<PlacementPlan> {
    let metadata = ModelMetadata::from_gguf(model)?;
    // Cap KV headroom context — the model's max context (e.g., 131K) is the ceiling,
    // not the operating point. Use a practical default for placement planning.
    let context_length = if context_length > 0 {
        context_length.min(metadata.context_length.max(2048))
    } else {
        metadata.context_length.max(2048).min(8192)
    };
    let capacities = compute_tier_capacities(hardware, &metadata, context_length);

    // Score and sort tensors
    let scored = score_tensors(&model.tensors, &metadata);

    // Try LP first, fall back to greedy
    let tier_assignments = match lp_assign(&scored, &capacities, hardware) {
        Ok(assignments) => {
            tracing::debug!("LP solver produced optimal placement");
            assignments
        }
        Err(e) => {
            tracing::debug!("LP solver failed ({e}), using greedy placement");
            greedy_assign(&scored, &capacities)
        }
    };

    // Build prefetch schedule
    let prefetch_schedule =
        build_prefetch_schedule(&tier_assignments, &model.tensors, &metadata, hardware);

    // KV cache plan
    let kv_cache_plan = compute_kv_cache_plan(&metadata, context_length, &tier_assignments, &capacities);

    // Quick tok/s estimate for the plan
    let estimated_tok_per_sec = quick_estimate(
        &tier_assignments,
        &model.tensors,
        &metadata,
        hardware,
        &prefetch_schedule,
    );

    let experience_tier = ExperienceTier::from_tok_per_sec(estimated_tok_per_sec);

    // Compute profile hash for cache invalidation
    let profile_hash = format!("{:x}", {
        let json = serde_json::to_string(hardware).unwrap_or_default();
        let mut h: u64 = 0xcbf29ce484222325;
        for b in json.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    });

    let model_id = model
        .get_string("general.name")
        .unwrap_or("unknown")
        .to_string();

    Ok(PlacementPlan {
        model_id,
        hardware_profile_hash: profile_hash,
        tier_assignments,
        prefetch_schedule,
        estimated_tok_per_sec,
        estimated_time_to_first_token: 0.5, // rough estimate
        kv_cache_plan,
        experience_tier,
    })
}

struct TierCapacities {
    gpu_bytes: u64,
    ram_bytes: u64,
    /// Total unified limit (gpu + ram must not exceed this)
    unified_limit: u64,
    nvme_peak_bw: u64,
}

fn compute_tier_capacities(
    hw: &HardwareProfile,
    metadata: &ModelMetadata,
    context_length: u32,
) -> TierCapacities {
    let total_ram = hw.memory.total_bytes;
    let usable = total_ram.saturating_sub(OS_OVERHEAD);

    let gpu_max = hw.gpu.as_ref().map_or(0, |g| g.vram_bytes);
    let gpu_bytes = gpu_max.min(usable);

    // Reserve space for KV cache
    let kv_headroom = estimate_kv_bytes(metadata, context_length);
    let ram_bytes = usable.saturating_sub(gpu_bytes).saturating_sub(kv_headroom);
    let unified_limit = usable.saturating_sub(kv_headroom);

    let nvme_peak_bw = hw
        .storage
        .first()
        .map(|s| s.sequential_read.peak_sequential)
        .unwrap_or(3_000_000_000); // 3 GB/s fallback

    TierCapacities {
        gpu_bytes,
        ram_bytes,
        unified_limit,
        nvme_peak_bw,
    }
}

fn estimate_kv_bytes(metadata: &ModelMetadata, context_length: u32) -> u64 {
    if metadata.num_heads == 0 || metadata.embedding_dim == 0 {
        return 0;
    }
    let head_dim = metadata.embedding_dim as u64 / metadata.num_heads as u64;
    // 2 (key+value) * layers * kv_heads * head_dim * 2 (FP16 bytes) * context
    2 * metadata.num_layers as u64
        * metadata.num_kv_heads as u64
        * head_dim
        * 2
        * context_length as u64
}

struct ScoredTensor {
    name: String,
    size_bytes: u64,
    score: f64,
    access_freq: f64,
    layer_index: Option<u32>,
}

fn score_tensors(tensors: &[TensorInfo], metadata: &ModelMetadata) -> Vec<ScoredTensor> {
    let experts_per_token = metadata.num_experts_used.unwrap_or(1);
    let total_experts = metadata.num_experts.unwrap_or(1);

    tensors
        .iter()
        .map(|t| {
            let role = TensorRole::from_name(&t.name);
            let access_freq = role.access_frequency(experts_per_token, total_experts);

            // Bonus for small, always-accessed tensors
            let bonus = match role {
                TensorRole::Norm => 10.0,
                TensorRole::Embedding | TensorRole::OutputHead => 5.0,
                _ => 1.0,
            };

            let size_gb = t.size_bytes as f64 / (1u64 << 30) as f64;
            let score = access_freq * bonus / (1.0 + size_gb);

            ScoredTensor {
                name: t.name.clone(),
                size_bytes: t.size_bytes,
                score,
                access_freq,
                layer_index: t.layer_index,
            }
        })
        .collect()
}

fn greedy_assign(
    tensors: &[ScoredTensor],
    caps: &TierCapacities,
) -> HashMap<String, StorageTier> {
    let mut assignments = HashMap::new();

    // Separate layer tensors from non-layer tensors (embedding, output head, etc.)
    let mut layer_tensors: std::collections::BTreeMap<u32, Vec<&ScoredTensor>> =
        std::collections::BTreeMap::new();
    let mut non_layer_tensors: Vec<&ScoredTensor> = Vec::new();

    for t in tensors {
        if let Some(layer) = t.layer_index {
            layer_tensors.entry(layer).or_default().push(t);
        } else {
            non_layer_tensors.push(t);
        }
    }

    // Assign non-layer tensors to GPU/RAM first (these are always accessed)
    non_layer_tensors.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut gpu_remaining = caps.gpu_bytes;
    let mut unified_remaining = caps.unified_limit;

    for t in &non_layer_tensors {
        if t.size_bytes <= gpu_remaining && t.size_bytes <= unified_remaining {
            gpu_remaining -= t.size_bytes;
            unified_remaining -= t.size_bytes;
            assignments.insert(t.name.clone(), StorageTier::Gpu);
        } else if t.size_bytes <= unified_remaining {
            unified_remaining -= t.size_bytes;
            assignments.insert(t.name.clone(), StorageTier::Ram);
        } else {
            assignments.insert(t.name.clone(), StorageTier::Nvme);
        }
    }

    // Assign layers contiguously: lowest layers to GPU/RAM, rest to NVMe.
    // Once a layer doesn't fit, all remaining layers go to NVMe (contiguity).
    let mut nvme_cutoff = false;

    for (_layer_idx, layer_ts) in &layer_tensors {
        let layer_size: u64 = layer_ts.iter().map(|t| t.size_bytes).sum();

        if !nvme_cutoff && layer_size <= unified_remaining {
            unified_remaining -= layer_size;

            // Within this layer, split GPU vs RAM by tensor score
            let mut sorted_layer: Vec<&&ScoredTensor> = layer_ts.iter().collect();
            sorted_layer.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

            for t in sorted_layer {
                if t.size_bytes <= gpu_remaining {
                    gpu_remaining -= t.size_bytes;
                    assignments.insert(t.name.clone(), StorageTier::Gpu);
                } else {
                    assignments.insert(t.name.clone(), StorageTier::Ram);
                }
            }
        } else {
            nvme_cutoff = true;
            for t in layer_ts {
                assignments.insert(t.name.clone(), StorageTier::Nvme);
            }
        }
    }

    assignments
}

fn lp_assign(
    tensors: &[ScoredTensor],
    caps: &TierCapacities,
    hw: &HardwareProfile,
) -> anyhow::Result<HashMap<String, StorageTier>> {
    let n = tensors.len();
    if n == 0 {
        return Ok(HashMap::new());
    }

    let gpu_bw = hw
        .gpu
        .as_ref()
        .map_or(1e9, |g| g.bandwidth_bytes_per_sec as f64);
    let ram_bw = hw.memory.bandwidth_bytes_per_sec as f64;
    let nvme_bw = caps.nvme_peak_bw as f64;

    // Collect unique layer indices (sorted ascending) for contiguity constraints
    let mut layer_set = std::collections::BTreeSet::new();
    for t in tensors {
        if let Some(l) = t.layer_index {
            layer_set.insert(l);
        }
    }
    let layer_indices: Vec<u32> = layer_set.into_iter().collect();
    let layer_pos: HashMap<u32, usize> = layer_indices
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();
    let num_layers = layer_indices.len();

    // Create binary variables (MIP — no rounding needed)
    variables! {
        vars:
    }

    let mut x_gpu = Vec::with_capacity(n);
    let mut x_ram = Vec::with_capacity(n);
    let mut x_nvme = Vec::with_capacity(n);

    for _ in 0..n {
        x_gpu.push(vars.add(variable().binary()));
        x_ram.push(vars.add(variable().binary()));
        x_nvme.push(vars.add(variable().binary()));
    }

    // Binary variable per layer: 1 = NVMe, 0 = GPU/RAM
    let mut layer_nvme = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layer_nvme.push(vars.add(variable().binary()));
    }

    // Objective: minimize weighted latency
    let mut objective = Expression::from(0.0);
    for (i, t) in tensors.iter().enumerate() {
        let weight = t.size_bytes as f64 * t.access_freq;
        objective += x_gpu[i] * (weight / gpu_bw);
        objective += x_ram[i] * (weight / ram_bw);
        objective += x_nvme[i] * (weight / nvme_bw);
    }

    let mut problem = vars.minimise(objective).using(default_solver);

    // Constraint: each tensor on exactly one tier
    for i in 0..n {
        problem = problem.with(constraint!(x_gpu[i] + x_ram[i] + x_nvme[i] == 1.0));
    }

    // GPU capacity
    let mut gpu_sum = Expression::from(0.0);
    for (i, t) in tensors.iter().enumerate() {
        gpu_sum += x_gpu[i] * t.size_bytes as f64;
    }
    problem = problem.with(constraint!(gpu_sum <= caps.gpu_bytes as f64));

    // RAM capacity
    let mut ram_sum = Expression::from(0.0);
    for (i, t) in tensors.iter().enumerate() {
        ram_sum += x_ram[i] * t.size_bytes as f64;
    }
    problem = problem.with(constraint!(ram_sum <= caps.ram_bytes as f64));

    // Unified memory constraint: GPU + RAM combined
    let mut unified_sum = Expression::from(0.0);
    for (i, t) in tensors.iter().enumerate() {
        unified_sum += (x_gpu[i] + x_ram[i]) * t.size_bytes as f64;
    }
    problem = problem.with(constraint!(unified_sum <= caps.unified_limit as f64));

    // Layer contiguity: monotonicity — once NVMe starts, it stays NVMe
    // layer_nvme[L] >= layer_nvme[L-1] for consecutive layers
    for j in 1..num_layers {
        problem = problem.with(constraint!(layer_nvme[j] >= layer_nvme[j - 1]));
    }

    // Link tensor NVMe vars to layer NVMe vars:
    // All tensors in a layer share the same NVMe/non-NVMe assignment
    for (i, t) in tensors.iter().enumerate() {
        if let Some(layer) = t.layer_index {
            if let Some(&j) = layer_pos.get(&layer) {
                problem = problem.with(constraint!(x_nvme[i] == layer_nvme[j]));
            }
        }
    }

    let solution = problem.solve()?;

    // Extract assignments — MIP gives integer solution, no rounding needed
    let mut assignments = HashMap::new();
    for (i, t) in tensors.iter().enumerate() {
        let g = solution.value(x_gpu[i]);
        let r = solution.value(x_ram[i]);
        let _v = solution.value(x_nvme[i]);

        let tier = if g > 0.5 {
            StorageTier::Gpu
        } else if r > 0.5 {
            StorageTier::Ram
        } else {
            StorageTier::Nvme
        };
        assignments.insert(t.name.clone(), tier);
    }

    Ok(assignments)
}

fn compute_kv_cache_plan(
    metadata: &ModelMetadata,
    context_length: u32,
    _assignments: &HashMap<String, StorageTier>,
    caps: &TierCapacities,
) -> KvCachePlan {
    let kv_per_token = estimate_kv_bytes(metadata, 1);
    if kv_per_token == 0 {
        return KvCachePlan {
            hot_window_tokens: context_length,
            warm_window_tokens: 0,
            hot_tier: StorageTier::Gpu,
            warm_tier: StorageTier::Ram,
            hot_bytes: 0,
            warm_bytes: 0,
        };
    }

    // 20% of GPU budget for hot KV cache
    let gpu_kv_budget = caps.gpu_bytes / 5;
    let hot_tokens = (gpu_kv_budget / kv_per_token).min(context_length as u64) as u32;
    let warm_tokens = context_length.saturating_sub(hot_tokens);

    // Q8 warm cache uses ~half the bytes of FP16
    let kv_per_token_q8 = kv_per_token / 2;

    KvCachePlan {
        hot_window_tokens: hot_tokens,
        warm_window_tokens: warm_tokens,
        hot_tier: StorageTier::Gpu,
        warm_tier: StorageTier::Ram,
        hot_bytes: hot_tokens as u64 * kv_per_token,
        warm_bytes: warm_tokens as u64 * kv_per_token_q8,
    }
}

fn quick_estimate(
    assignments: &HashMap<String, StorageTier>,
    tensors: &[TensorInfo],
    metadata: &ModelMetadata,
    hw: &HardwareProfile,
    _prefetch: &PrefetchSchedule,
) -> f64 {
    let gpu_bw = hw
        .gpu
        .as_ref()
        .map_or(1e9, |g| g.bandwidth_bytes_per_sec as f64);
    let ram_bw = hw.memory.bandwidth_bytes_per_sec as f64;
    let nvme_bw = hw
        .storage
        .first()
        .map(|s| s.sequential_read.peak_sequential as f64)
        .unwrap_or(3e9);

    let experts_per_token = metadata.num_experts_used.unwrap_or(1);
    let total_experts = metadata.num_experts.unwrap_or(1);

    let mut total_latency_secs = 0.0;

    for t in tensors {
        let role = TensorRole::from_name(&t.name);
        let freq = role.access_frequency(experts_per_token, total_experts);
        let tier = assignments.get(&t.name).unwrap_or(&StorageTier::Nvme);

        let transfer_time = match tier {
            StorageTier::Gpu => t.size_bytes as f64 / gpu_bw,
            StorageTier::Ram => t.size_bytes as f64 / ram_bw,
            StorageTier::Nvme => t.size_bytes as f64 / nvme_bw,
        };

        total_latency_secs += transfer_time * freq;
    }

    // Add sync overhead
    total_latency_secs += metadata.num_layers as f64 * SYNC_OVERHEAD_PER_LAYER_US * 1e-6;

    if total_latency_secs > 0.0 {
        1.0 / total_latency_secs
    } else {
        0.0
    }
}

/// Compute a summary of where tensors are placed.
pub fn summarize_placement(
    assignments: &HashMap<String, StorageTier>,
    tensors: &[TensorInfo],
) -> PlacementSummary {
    let mut summary = PlacementSummary {
        layers_on_gpu: 0,
        layers_in_ram: 0,
        layers_on_nvme: 0,
        total_gpu_bytes: 0,
        total_ram_bytes: 0,
        total_nvme_bytes: 0,
    };

    // Count unique layers per tier
    let mut gpu_layers = std::collections::HashSet::new();
    let mut ram_layers = std::collections::HashSet::new();
    let mut nvme_layers = std::collections::HashSet::new();

    for t in tensors {
        let tier = assignments.get(&t.name).unwrap_or(&StorageTier::Nvme);
        match tier {
            StorageTier::Gpu => {
                summary.total_gpu_bytes += t.size_bytes;
                if let Some(l) = t.layer_index {
                    gpu_layers.insert(l);
                }
            }
            StorageTier::Ram => {
                summary.total_ram_bytes += t.size_bytes;
                if let Some(l) = t.layer_index {
                    ram_layers.insert(l);
                }
            }
            StorageTier::Nvme => {
                summary.total_nvme_bytes += t.size_bytes;
                if let Some(l) = t.layer_index {
                    nvme_layers.insert(l);
                }
            }
        }
    }

    summary.layers_on_gpu = gpu_layers.len() as u32;
    summary.layers_in_ram = ram_layers.len() as u32;
    summary.layers_on_nvme = nvme_layers.len() as u32;

    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::gguf::{GgmlType, TensorInfo};

    fn make_tensors(count: usize, size: u64) -> Vec<TensorInfo> {
        (0..count)
            .map(|i| TensorInfo {
                name: format!("blk.{i}.attn_q.weight"),
                dimensions: vec![4096, 4096],
                dtype: GgmlType::Q4K,
                offset: 0,
                size_bytes: size,
                layer_index: Some(i as u32),
            })
            .collect()
    }

    fn make_metadata(layers: u32) -> ModelMetadata {
        ModelMetadata {
            architecture: "llama".into(),
            parameter_count: 7_000_000_000,
            context_length: 4096,
            embedding_dim: 4096,
            num_layers: layers,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            quantization: Some("Q4K".into()),
            is_moe: false,
            num_experts: None,
            num_experts_used: None,
        }
    }

    #[test]
    fn test_greedy_respects_capacity() {
        let tensors: Vec<ScoredTensor> = (0..10)
            .map(|i| ScoredTensor {
                name: format!("tensor_{i}"),
                size_bytes: 1 << 30, // 1 GB each
                score: 10.0 - i as f64,
                access_freq: 1.0,
                layer_index: None,
            })
            .collect();

        let caps = TierCapacities {
            gpu_bytes: 3 << 30,       // 3 GB GPU
            ram_bytes: 4 << 30,       // 4 GB RAM
            unified_limit: 7 << 30,   // 7 GB unified
            nvme_peak_bw: 5_000_000_000,
        };

        let assignments = greedy_assign(&tensors, &caps);

        let gpu_count = assignments.values().filter(|t| **t == StorageTier::Gpu).count();
        let ram_count = assignments.values().filter(|t| **t == StorageTier::Ram).count();
        let nvme_count = assignments.values().filter(|t| **t == StorageTier::Nvme).count();

        assert_eq!(gpu_count, 3);
        assert_eq!(ram_count, 4);
        assert_eq!(nvme_count, 3);
    }

    #[test]
    fn test_greedy_layer_contiguity() {
        let tensors: Vec<ScoredTensor> = (0..10)
            .map(|i| ScoredTensor {
                name: format!("blk.{i}.attn_q.weight"),
                size_bytes: 1 << 30,
                score: 10.0 - i as f64,
                access_freq: 1.0,
                layer_index: Some(i as u32),
            })
            .collect();

        let caps = TierCapacities {
            gpu_bytes: 3 << 30,
            ram_bytes: 4 << 30,
            unified_limit: 7 << 30,
            nvme_peak_bw: 5_000_000_000,
        };

        let assignments = greedy_assign(&tensors, &caps);

        // Verify contiguity: no NVMe tensor has a lower layer than any GPU/RAM tensor
        let max_resident_layer = assignments
            .iter()
            .filter(|(_, t)| **t != StorageTier::Nvme)
            .filter_map(|(name, _)| test_parse_layer(name))
            .max();

        let min_nvme_layer = assignments
            .iter()
            .filter(|(_, t)| **t == StorageTier::Nvme)
            .filter_map(|(name, _)| test_parse_layer(name))
            .min();

        if let (Some(max_r), Some(min_n)) = (max_resident_layer, min_nvme_layer) {
            assert!(
                max_r < min_n,
                "Contiguity violated: resident up to layer {max_r} but NVMe from {min_n}"
            );
        }

        // 7 layers fit in 7 GB unified, 3 on NVMe
        let nvme_count = assignments
            .values()
            .filter(|t| **t == StorageTier::Nvme)
            .count();
        assert_eq!(nvme_count, 3);
    }

    fn test_parse_layer(name: &str) -> Option<u32> {
        if name.starts_with("blk.") {
            let rest = &name[4..];
            rest.split('.').next()?.parse().ok()
        } else {
            None
        }
    }

    #[test]
    fn test_score_prioritizes_norms() {
        let tensors = vec![
            TensorInfo {
                name: "blk.0.attn_q.weight".into(),
                dimensions: vec![4096, 4096],
                dtype: GgmlType::Q4K,
                offset: 0,
                size_bytes: 1 << 30,
                layer_index: Some(0),
            },
            TensorInfo {
                name: "blk.0.attn_norm.weight".into(),
                dimensions: vec![4096],
                dtype: GgmlType::F32,
                offset: 0,
                size_bytes: 16384,
                layer_index: Some(0),
            },
        ];

        let metadata = make_metadata(1);
        let scored = score_tensors(&tensors, &metadata);

        // Norm should have higher score (small + 10x bonus)
        assert!(scored[1].score > scored[0].score);
    }
}
