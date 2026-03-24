use std::collections::HashMap;

use good_lp::{constraint, default_solver, variable, variables, Expression, Solution, SolverModel};

use crate::model::gguf::{GgufFile, TensorInfo};
use crate::model::metadata::ModelMetadata;
use crate::model::tensor_role::TensorRole;
use crate::profiler::types::HardwareProfile;
use crate::scheduler::prefetch::build_prefetch_schedule;
use crate::scheduler::types::*;

/// RAM reserved for the OS and background processes.
///
/// macOS: ~2 GiB (kernel + system agents).
/// Windows: ~4 GiB (kernel + system processes tend to use more).
/// Linux / WSL2: ~1 GiB conservative estimate.
#[cfg(target_os = "macos")]
const OS_OVERHEAD: u64 = 2 * (1 << 30); // 2 GiB
#[cfg(target_os = "windows")]
const OS_OVERHEAD: u64 = 4 * (1 << 30); // 4 GiB
#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
const OS_OVERHEAD: u64 = 1 * (1 << 30); // 1 GiB (Linux / WSL2)

/// GPU runtime overhead: CUDA/Metal framework + compute buffer pool.
///
/// CUDA: ~0.5 GiB driver + context overhead on Ampere+.
/// Metal: ~1 GiB.
#[cfg(target_os = "macos")]
const GPU_RUNTIME_OVERHEAD: u64 = 1 << 30; // 1 GiB (Metal)
#[cfg(not(target_os = "macos"))]
const GPU_RUNTIME_OVERHEAD: u64 = 512 * (1 << 20); // 512 MiB (CUDA)
const SYNC_OVERHEAD_PER_LAYER_US: f64 = 50.0; // 50μs CPU-GPU sync per layer
const MOE_CACHE_HIT_RATE: f64 = 0.965; // From PowerInfer-2, matches estimator.rs

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

    // Try streaming modes: sparse MoE mmap, expert-streaming, then dense FFN-streaming
    let (tier_assignments, inference_mode) =
        if let Some(assignments) = try_sparse_moe_mmap(&scored, &capacities, &metadata) {
            tracing::info!("Sparse MoE mmap: active working set fits in RAM, using OS page cache");
            (assignments, InferenceMode::SparseMoeMmap)
        } else if let Some(es) = try_expert_streaming_assign(&scored, &capacities, &metadata) {
            tracing::info!("Expert-streaming placement: non-expert tensors on GPU/RAM, experts on NVMe");
            (es, InferenceMode::ExpertStreaming)
        } else if let Some(ds) = try_dense_ffn_streaming_assign(&scored, &capacities, &metadata) {
            tracing::info!("Dense FFN-streaming placement: attention+norms on GPU, FFN on NVMe");
            (ds, InferenceMode::DenseFfnStreaming)
        } else {
            // Try LP first, fall back to greedy
            let assignments = match lp_assign(&scored, &capacities, hardware, &metadata) {
                Ok(a) => {
                    tracing::debug!("LP solver produced optimal placement");
                    a
                }
                Err(e) => {
                    tracing::debug!("LP solver failed ({e}), using greedy placement");
                    greedy_assign(&scored, &capacities, hardware, &metadata)
                }
            };
            let has_nvme = assignments.values().any(|t| *t == StorageTier::Nvme);
            let mode = if !has_nvme {
                InferenceMode::FullResident
            } else {
                // Determined later by inference engine (keep-resident vs full-streaming)
                InferenceMode::FullStreaming
            };
            (assignments, mode)
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
        inference_mode,
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

    // Reserve space for KV cache and GPU runtime (compute buffers, Metal overhead).
    // On unified memory (Apple Silicon), KV cache + compute live in the same
    // working set as model weights — must subtract from gpu_bytes too.
    let kv_headroom = estimate_kv_bytes(metadata, context_length);
    let gpu_bytes = gpu_max
        .min(usable)
        .saturating_sub(kv_headroom)
        .saturating_sub(GPU_RUNTIME_OVERHEAD);
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
    role: TensorRole,
}

/// Effective I/O frequency for NVMe placement, applying MoE cache-hit discount.
fn effective_io_freq(role: &TensorRole, base_freq: f64, is_moe: bool) -> f64 {
    if is_moe {
        match role {
            TensorRole::MoeExpert { .. } | TensorRole::MoeFusedExperts => {
                base_freq * (1.0 - MOE_CACHE_HIT_RATE)
            }
            _ => base_freq,
        }
    } else {
        base_freq
    }
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
                role,
            }
        })
        .collect()
}

/// Try expert-streaming placement for MoE models: non-expert tensors on GPU/RAM,
/// all expert (MoeFusedExperts) tensors on NVMe. Returns None if the model is not
/// MoE or non-expert tensors don't fit in unified memory.
/// Ultra-sparse MoE: if the active working set (experts_used/expert_count × model_size)
/// fits comfortably in RAM, skip the pool/callback machinery entirely. Use mmap and let
/// the OS page cache handle sparsity — only the active expert pages get loaded into
/// physical memory. This avoids eval callback overhead, pool slot allocation, and tensor
/// pointer rewriting, which dominate latency for very sparse models (e.g. 2% activation).
fn try_sparse_moe_mmap(
    tensors: &[ScoredTensor],
    caps: &TierCapacities,
    metadata: &ModelMetadata,
) -> Option<HashMap<String, StorageTier>> {
    if !metadata.is_moe {
        return None;
    }

    let experts_total = metadata.num_experts.unwrap_or(1).max(1) as u64;
    let experts_used = metadata.num_experts_used.unwrap_or(1).max(1) as u64;
    let activation_ratio = experts_used as f64 / experts_total as f64;

    // Only use this path for sparse models (activation ratio < 15%)
    if activation_ratio >= 0.15 {
        return None;
    }

    let total_bytes: u64 = tensors.iter().map(|t| t.size_bytes).sum();
    let active_bytes = (total_bytes as f64 * activation_ratio) as u64;

    // Active working set must fit in 30% of unified memory limit
    if active_bytes > caps.unified_limit * 30 / 100 {
        return None;
    }

    // All tensors go to GPU tier — mmap handles data.
    // If the model exceeds Metal's working set, gpu_layers_from_placement will
    // detect this and return ngl=0 (CPU-only). The OS page cache still works
    // because only ~2% of pages are active per token.
    let mut assignments = HashMap::new();
    for t in tensors {
        assignments.insert(t.name.clone(), StorageTier::Gpu);
    }

    let fits_gpu = total_bytes <= caps.gpu_bytes + GPU_RUNTIME_OVERHEAD;
    tracing::info!(
        "Sparse MoE mmap: {:.0}% activation ({}/{} experts), {:.1} GB active of {:.1} GB total{}",
        activation_ratio * 100.0,
        experts_used,
        experts_total,
        active_bytes as f64 / (1u64 << 30) as f64,
        total_bytes as f64 / (1u64 << 30) as f64,
        if fits_gpu { "" } else { " (exceeds GPU, will use CPU-only)" },
    );

    Some(assignments)
}

fn try_expert_streaming_assign(
    tensors: &[ScoredTensor],
    caps: &TierCapacities,
    metadata: &ModelMetadata,
) -> Option<HashMap<String, StorageTier>> {
    if !metadata.is_moe {
        return None;
    }

    let non_expert_bytes: u64 = tensors
        .iter()
        .filter(|t| !matches!(t.role, TensorRole::MoeFusedExperts))
        .map(|t| t.size_bytes)
        .sum();
    let expert_bytes: u64 = tensors
        .iter()
        .filter(|t| matches!(t.role, TensorRole::MoeFusedExperts))
        .map(|t| t.size_bytes)
        .sum();

    // Only use expert-streaming if: non-experts fit in memory AND experts would overflow.
    // Also: expert buffer virtual allocation must fit in Metal's address space.
    // On unified memory, Metal sees all process memory — a 30 GB posix_memalign buffer
    // causes Metal OOM even if physical pages aren't committed. Cap expert buffer at
    // (gpu_max - non_expert_bytes - 1 GB headroom) to stay within Metal limits.
    let total = non_expert_bytes + expert_bytes;
    if non_expert_bytes > caps.unified_limit || total <= caps.unified_limit {
        return None; // Either doesn't fit at all, or everything fits (use keep-resident)
    }

    // With the pool buffer, Metal only sees the small pool (~1 GB), not the full expert
    // tensor size. Check that non-expert tensors + pool overhead fit in GPU budget.
    // Pool size is approximately 14 slots × largest fused tensor size.
    // Conservative estimate: pool ≈ 1.5 GB.
    let estimated_pool: u64 = 3 * (1 << 30) / 2; // 1.5 GB
    if non_expert_bytes + estimated_pool > caps.gpu_bytes + GPU_RUNTIME_OVERHEAD {
        tracing::debug!(
            "Expert-streaming skipped: non-expert ({:.1} GB) + pool ({:.1} GB) exceeds GPU budget ({:.1} GB)",
            non_expert_bytes as f64 / (1u64 << 30) as f64,
            estimated_pool as f64 / (1u64 << 30) as f64,
            (caps.gpu_bytes + GPU_RUNTIME_OVERHEAD) as f64 / (1u64 << 30) as f64,
        );
        return None;
    }

    let mut assignments = HashMap::new();
    let mut gpu_remaining = caps.gpu_bytes;

    // Assign non-expert tensors to GPU/RAM by score
    let mut non_experts: Vec<&ScoredTensor> = tensors
        .iter()
        .filter(|t| !matches!(t.role, TensorRole::MoeFusedExperts))
        .collect();
    non_experts.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    for t in non_experts {
        if t.size_bytes <= gpu_remaining {
            gpu_remaining -= t.size_bytes;
            assignments.insert(t.name.clone(), StorageTier::Gpu);
        } else {
            assignments.insert(t.name.clone(), StorageTier::Ram);
        }
    }

    // All expert tensors to NVMe
    for t in tensors
        .iter()
        .filter(|t| matches!(t.role, TensorRole::MoeFusedExperts))
    {
        assignments.insert(t.name.clone(), StorageTier::Nvme);
    }

    tracing::info!(
        "Expert-streaming: {:.1} GB non-expert resident, {:.1} GB expert on NVMe",
        non_expert_bytes as f64 / (1u64 << 30) as f64,
        expert_bytes as f64 / (1u64 << 30) as f64,
    );

    Some(assignments)
}

/// Try dense FFN streaming for non-MoE models: attention + norms on GPU/RAM,
/// FFN tensors (gate, up, down) on NVMe streamed through pool buffer.
fn try_dense_ffn_streaming_assign(
    tensors: &[ScoredTensor],
    caps: &TierCapacities,
    metadata: &ModelMetadata,
) -> Option<HashMap<String, StorageTier>> {
    if metadata.is_moe {
        return None;
    }

    let is_ffn = |role: &TensorRole| {
        matches!(
            role,
            TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown
        )
    };

    let non_ffn_bytes: u64 = tensors
        .iter()
        .filter(|t| !is_ffn(&t.role))
        .map(|t| t.size_bytes)
        .sum();
    let ffn_bytes: u64 = tensors.iter().filter(|t| is_ffn(&t.role)).map(|t| t.size_bytes).sum();

    let total = non_ffn_bytes + ffn_bytes;
    if non_ffn_bytes > caps.unified_limit || total <= caps.unified_limit {
        return None; // Either doesn't fit at all, or everything fits
    }

    if ffn_bytes == 0 {
        return None;
    }

    // Pool size: ~6 slots (2 layers × 3 FFN tensors)
    let max_ffn_tensor: u64 = tensors
        .iter()
        .filter(|t| is_ffn(&t.role))
        .map(|t| t.size_bytes)
        .max()
        .unwrap_or(0);
    let estimated_pool = max_ffn_tensor * 6;
    if non_ffn_bytes + estimated_pool > caps.gpu_bytes + GPU_RUNTIME_OVERHEAD {
        tracing::debug!(
            "Dense FFN streaming skipped: non-FFN ({:.1} GB) + pool ({:.1} GB) exceeds GPU budget",
            non_ffn_bytes as f64 / (1u64 << 30) as f64,
            estimated_pool as f64 / (1u64 << 30) as f64,
        );
        return None;
    }

    let mut assignments = HashMap::new();
    let mut gpu_remaining = caps.gpu_bytes;

    let mut non_ffn: Vec<&ScoredTensor> = tensors.iter().filter(|t| !is_ffn(&t.role)).collect();
    non_ffn.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    for t in non_ffn {
        if t.size_bytes <= gpu_remaining {
            gpu_remaining -= t.size_bytes;
            assignments.insert(t.name.clone(), StorageTier::Gpu);
        } else {
            assignments.insert(t.name.clone(), StorageTier::Ram);
        }
    }

    for t in tensors.iter().filter(|t| is_ffn(&t.role)) {
        assignments.insert(t.name.clone(), StorageTier::Nvme);
    }

    tracing::info!(
        "Dense FFN streaming: {:.1} GB non-FFN resident, {:.1} GB FFN on NVMe",
        non_ffn_bytes as f64 / (1u64 << 30) as f64,
        ffn_bytes as f64 / (1u64 << 30) as f64,
    );

    Some(assignments)
}

fn greedy_assign(
    tensors: &[ScoredTensor],
    caps: &TierCapacities,
    hw: &HardwareProfile,
    metadata: &ModelMetadata,
) -> HashMap<String, StorageTier> {
    let mut assignments = HashMap::new();

    let ram_bw = hw.memory.bandwidth_bytes_per_sec as f64;
    let nvme_bw = caps.nvme_peak_bw as f64;

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

    // Try every possible NVMe cutoff point — pick the one that minimizes
    // total latency under the overlap model: sum of max(compute, io) per layer,
    // where NVMe I/O is reduced by the previous layer's compute (prefetch).
    let layer_list: Vec<(u32, &Vec<&ScoredTensor>)> =
        layer_tensors.iter().map(|(&k, v)| (k, v)).collect();
    let max_cutoff = layer_list.len();

    let mut best_cutoff = 0;
    let mut best_cost = f64::MAX;

    for cutoff in 0..=max_cutoff {
        // Check if layers 0..cutoff fit in unified memory
        let resident_bytes: u64 = layer_list[..cutoff]
            .iter()
            .flat_map(|(_, ts)| ts.iter())
            .map(|t| t.size_bytes)
            .sum();
        if resident_bytes > unified_remaining {
            break; // This and higher cutoffs won't fit
        }

        // Cost of resident layers (GPU/RAM transfer time)
        let resident_cost: f64 = layer_list[..cutoff]
            .iter()
            .flat_map(|(_, ts)| ts.iter())
            .map(|t| t.size_bytes as f64 * t.access_freq / ram_bw)
            .sum();

        // Cost of NVMe layers with overlap model
        let mut nvme_cost = 0.0;
        let mut prev_compute = if cutoff > 0 {
            // Last resident layer provides compute time for prefetch
            layer_list[cutoff - 1]
                .1
                .iter()
                .map(|t| t.size_bytes as f64 * t.access_freq / ram_bw)
                .sum()
        } else {
            0.0
        };

        for &(_, ts) in &layer_list[cutoff..] {
            let layer_io: f64 = ts
                .iter()
                .map(|t| {
                    let eff = effective_io_freq(&t.role, t.access_freq, metadata.is_moe);
                    t.size_bytes as f64 * eff / nvme_bw
                })
                .sum();
            let effective_io = (layer_io - prev_compute).max(0.0);
            nvme_cost += effective_io + SYNC_OVERHEAD_PER_LAYER_US * 1e-6;
            prev_compute = 0.0; // NVMe layers have no compute to overlap with
        }

        let total = resident_cost + nvme_cost;
        if total < best_cost {
            best_cost = total;
            best_cutoff = cutoff;
        }
    }

    // Apply the best cutoff
    for (i, &(_, ts)) in layer_list.iter().enumerate() {
        if i < best_cutoff {
            let mut sorted_layer: Vec<&&ScoredTensor> = ts.iter().collect();
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
            for t in ts {
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
    metadata: &ModelMetadata,
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

    // Continuous auxiliary variable per layer: max(compute, io) with prefetch overlap
    let mut layer_cost = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layer_cost.push(vars.add(variable().min(0.0)));
    }

    // Objective: sum of per-layer max(compute, io) + non-layer transfer costs
    let mut objective = Expression::from(0.0);
    for j in 0..num_layers {
        objective += layer_cost[j];
    }

    // Non-layer tensors: additive (no overlap possible — embedding, output head)
    for (i, t) in tensors.iter().enumerate() {
        if t.layer_index.is_some() {
            continue;
        }
        let weight = t.size_bytes as f64 * t.access_freq;
        objective += x_gpu[i] * (weight / gpu_bw);
        objective += x_ram[i] * (weight / ram_bw);
        objective += x_nvme[i] * (weight / nvme_bw);
    }

    let mut problem = vars.minimise(objective).using(default_solver);

    // Build per-layer compute and I/O expressions, then constrain layer_cost
    let mut compute_exprs = Vec::with_capacity(num_layers);
    let mut io_exprs = Vec::with_capacity(num_layers);

    for j in 0..num_layers {
        let layer_idx = layer_indices[j];
        let mut compute_expr = Expression::from(0.0);
        let mut io_expr = Expression::from(0.0);

        for (i, t) in tensors.iter().enumerate() {
            if t.layer_index != Some(layer_idx) {
                continue;
            }
            let weight = t.size_bytes as f64 * t.access_freq;

            // GPU and RAM transfers contribute to compute time
            compute_expr += x_gpu[i] * (weight / gpu_bw);
            compute_expr += x_ram[i] * (weight / ram_bw);

            // NVMe transfers contribute to I/O time (with MoE cache-hit discount)
            let eff_weight =
                t.size_bytes as f64 * effective_io_freq(&t.role, t.access_freq, metadata.is_moe);
            io_expr += x_nvme[i] * (eff_weight / nvme_bw);
        }

        // layer_cost[j] >= compute_time (linearization of max)
        problem = problem.with(constraint!(layer_cost[j] >= compute_expr.clone()));
        // layer_cost[j] >= io_time (for layer 0, no prefetch possible)
        if j == 0 {
            problem = problem.with(constraint!(layer_cost[j] >= io_expr.clone()));
        }

        compute_exprs.push(compute_expr);
        io_exprs.push(io_expr);
    }

    // Cross-layer prefetch overlap: I/O of layer j is reduced by compute of layer j-1.
    // layer_cost[j] >= io_j - compute_{j-1}
    // Combined with layer_cost[j] >= 0 (from variable bound), this correctly models
    // that prefetch starts during the previous layer's compute.
    for j in 1..num_layers {
        problem = problem.with(constraint!(
            layer_cost[j] >= io_exprs[j].clone() - compute_exprs[j - 1].clone()
        ));
    }

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
    let kv_per_token_fp16 = estimate_kv_bytes(metadata, 1);
    if kv_per_token_fp16 == 0 {
        return KvCachePlan {
            hot_window_tokens: context_length,
            warm_window_tokens: 0,
            hot_tier: StorageTier::Gpu,
            warm_tier: StorageTier::Ram,
            hot_bytes: 0,
            warm_bytes: 0,
            kv_quantization: None,
        };
    }

    let total_fp16_kv = kv_per_token_fp16 * context_length as u64;

    // Auto-select Q8 KV when GPU budget is tight:
    // FP16 KV exceeds 40% of GPU budget, but Q8 KV fits in 25%
    let kv_quantization = if caps.gpu_bytes > 0
        && total_fp16_kv > caps.gpu_bytes * 2 / 5
        && (total_fp16_kv / 2) <= caps.gpu_bytes / 4
    {
        tracing::info!(
            "Auto-selecting Q8 KV: FP16 KV {:.1} GB > 40% of GPU {:.1} GB",
            total_fp16_kv as f64 / (1u64 << 30) as f64,
            caps.gpu_bytes as f64 / (1u64 << 30) as f64,
        );
        Some(KvQuantization::Q8_0)
    } else {
        None
    };

    let kv_per_token = if kv_quantization.is_some() {
        kv_per_token_fp16 / 2
    } else {
        kv_per_token_fp16
    };

    // 20% of GPU budget for hot KV cache
    let gpu_kv_budget = caps.gpu_bytes / 5;
    let hot_tokens = (gpu_kv_budget / kv_per_token).min(context_length as u64) as u32;
    let warm_tokens = context_length.saturating_sub(hot_tokens);

    // Q8 warm cache uses ~half the bytes of FP16
    let kv_per_token_q8 = kv_per_token_fp16 / 2;

    KvCachePlan {
        hot_window_tokens: hot_tokens,
        warm_window_tokens: warm_tokens,
        hot_tier: StorageTier::Gpu,
        warm_tier: StorageTier::Ram,
        hot_bytes: hot_tokens as u64 * kv_per_token,
        warm_bytes: warm_tokens as u64 * kv_per_token_q8,
        kv_quantization,
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
    let mut prev_compute_time = 0.0;

    // Per-layer overlap model: max(compute, io - prev_compute)
    for layer_idx in 0..metadata.num_layers {
        let mut layer_compute = 0.0;
        let mut layer_io = 0.0;

        for t in tensors.iter().filter(|t| t.layer_index == Some(layer_idx)) {
            let role = TensorRole::from_name(&t.name);
            let freq = role.access_frequency(experts_per_token, total_experts);
            let tier = assignments.get(&t.name).unwrap_or(&StorageTier::Nvme);

            match tier {
                StorageTier::Gpu => layer_compute += t.size_bytes as f64 * freq / gpu_bw,
                StorageTier::Ram => layer_compute += t.size_bytes as f64 * freq / ram_bw,
                StorageTier::Nvme => {
                    let eff = effective_io_freq(&role, freq, metadata.is_moe);
                    layer_io += t.size_bytes as f64 * eff / nvme_bw;
                }
            }
        }

        let effective_io = (layer_io - prev_compute_time).max(0.0);
        let layer_latency = layer_compute.max(effective_io) + SYNC_OVERHEAD_PER_LAYER_US * 1e-6;
        total_latency_secs += layer_latency;
        prev_compute_time = layer_compute;
    }

    // Non-layer tensors (embedding, output head) — additive, no overlap
    for t in tensors.iter().filter(|t| t.layer_index.is_none()) {
        let role = TensorRole::from_name(&t.name);
        let freq = role.access_frequency(experts_per_token, total_experts);
        let tier = assignments.get(&t.name).unwrap_or(&StorageTier::Nvme);

        let transfer = match tier {
            StorageTier::Gpu => t.size_bytes as f64 * freq / gpu_bw,
            StorageTier::Ram => t.size_bytes as f64 * freq / ram_bw,
            StorageTier::Nvme => t.size_bytes as f64 * freq / nvme_bw,
        };
        total_latency_secs += transfer;
    }

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

    fn make_hw() -> HardwareProfile {
        use crate::profiler::types::*;
        HardwareProfile {
            timestamp: chrono::Utc::now(),
            system: SystemInfo {
                os: "test".into(),
                arch: "arm64".into(),
                machine_model: "test".into(),
                total_cores: 10,
            },
            cpu: CpuProfile {
                model_name: "Test CPU".into(),
                cores_performance: 8,
                cores_efficiency: 2,
                has_amx: false,
                has_neon: true,
                has_avx512: false,
                has_avx2: false,
                int8_gflops: 0.0,
            },
            gpu: Some(GpuProfile {
                name: "Test GPU".into(),
                vram_bytes: 16 << 30,
                bandwidth_bytes_per_sec: 200_000_000_000,
                fp16_tflops: 0.0,
                backend: GpuBackend::Metal,
            }),
            memory: MemoryProfile {
                total_bytes: 32 << 30,
                available_bytes: 28 << 30,
                bandwidth_bytes_per_sec: 200_000_000_000,
                is_unified: true,
            },
            storage: vec![StorageProfile {
                device_path: "/dev/disk0".into(),
                mount_point: "/".into(),
                device_type: StorageType::NvmePcie,
                capacity_bytes: 1 << 40,
                free_bytes: 500 << 30,
                sequential_read: BandwidthCurve {
                    points: vec![],
                    peak_sequential: 5_000_000_000,
                },
                random_read_iops: 0,
                pcie_gen: None,
                wear_level: None,
            }],
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
                role: TensorRole::Other(format!("tensor_{i}")),
            })
            .collect();

        let caps = TierCapacities {
            gpu_bytes: 3 << 30,       // 3 GB GPU
            ram_bytes: 4 << 30,       // 4 GB RAM
            unified_limit: 7 << 30,   // 7 GB unified
            nvme_peak_bw: 5_000_000_000,
        };

        let hw = make_hw();
        let meta = make_metadata(0);
        let assignments = greedy_assign(&tensors, &caps, &hw, &meta);

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
                role: TensorRole::AttentionQuery,
            })
            .collect();

        let caps = TierCapacities {
            gpu_bytes: 3 << 30,
            ram_bytes: 4 << 30,
            unified_limit: 7 << 30,
            nvme_peak_bw: 5_000_000_000,
        };

        let hw = make_hw();
        let meta = make_metadata(10);
        let assignments = greedy_assign(&tensors, &caps, &hw, &meta);

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
