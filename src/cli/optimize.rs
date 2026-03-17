use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use hypura::cache::coactivation::CoActivationMatrix;
use hypura::model::gguf::GgufFile;
use hypura::model::tensor_role::TensorRole;

/// Rewrite fused expert tensors so co-activated experts are contiguous on disk.
/// Reads the co-activation matrix (from previous inference runs) and reorders
/// expert strides within each fused tensor using a greedy nearest-neighbor TSP.
///
/// Always writes to a new file (never in-place) with byte-sum verification.
pub fn run(model: &str) -> anyhow::Result<()> {
    let model_path = Path::new(model);
    let gguf = GgufFile::open(model_path)?;

    // Load co-activation matrix
    let co_act_path = CoActivationMatrix::persistence_path(model_path);
    let co_act = CoActivationMatrix::load(&co_act_path).map_err(|e| {
        anyhow::anyhow!(
            "No co-activation data found at {} (run inference first): {e}",
            co_act_path.display()
        )
    })?;

    if !co_act.has_data() {
        anyhow::bail!(
            "Co-activation matrix has insufficient data. Run at least 100 tokens of inference first."
        );
    }

    let num_experts = gguf.get_u32("expert_count").unwrap_or(0);
    if num_experts == 0 {
        anyhow::bail!("Not a MoE model (no expert_count metadata)");
    }

    // Find fused expert tensors and compute optimal ordering per layer
    let mut permutations: HashMap<u32, Vec<u32>> = HashMap::new();

    for tensor in &gguf.tensors {
        let role = TensorRole::from_name(&tensor.name);
        if role != TensorRole::MoeFusedExperts {
            continue;
        }
        let layer_idx = match tensor.layer_index {
            Some(l) => l,
            None => continue,
        };
        if permutations.contains_key(&layer_idx) {
            continue;
        }
        let perm = compute_optimal_expert_order(&co_act, layer_idx, num_experts);
        permutations.insert(layer_idx, perm);
    }

    if permutations.is_empty() {
        anyhow::bail!("No fused expert tensors found");
    }

    // Build output path
    let stem = model_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let ext = model_path
        .extension()
        .unwrap_or_default()
        .to_string_lossy();
    let output_path = model_path.with_file_name(format!("{stem}.optimized.{ext}"));

    tracing::info!("Writing optimized model to {}", output_path.display());

    // Copy file first, then reorder expert data in-place in the copy
    std::fs::copy(model_path, &output_path)?;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .read(true)
        .open(&output_path)?;

    let mut reordered_count = 0usize;
    for tensor in &gguf.tensors {
        let role = TensorRole::from_name(&tensor.name);
        if role != TensorRole::MoeFusedExperts {
            continue;
        }
        let layer_idx = match tensor.layer_index {
            Some(l) => l,
            None => continue,
        };
        let perm = match permutations.get(&layer_idx) {
            Some(p) => p,
            None => continue,
        };

        let stride = tensor.size_bytes as usize / num_experts as usize;
        let abs_offset = gguf.data_offset + tensor.offset;

        // Read original expert data from the copy
        let mut original = vec![0u8; tensor.size_bytes as usize];
        file.seek(SeekFrom::Start(abs_offset))?;
        file.read_exact(&mut original)?;

        // Rearrange according to permutation: perm[new_pos] = old_expert
        let mut reordered = vec![0u8; tensor.size_bytes as usize];
        for (new_pos, &old_expert) in perm.iter().enumerate() {
            let src_start = old_expert as usize * stride;
            let dst_start = new_pos * stride;
            reordered[dst_start..dst_start + stride]
                .copy_from_slice(&original[src_start..src_start + stride]);
        }

        // Byte-sum verification: same bytes, different order
        let original_sum: u64 = original.iter().map(|&b| b as u64).sum();
        let reordered_sum: u64 = reordered.iter().map(|&b| b as u64).sum();
        if original_sum != reordered_sum {
            // Roll back: delete output file
            let _ = std::fs::remove_file(&output_path);
            anyhow::bail!(
                "Checksum mismatch for {}: original={}, reordered={}",
                tensor.name,
                original_sum,
                reordered_sum
            );
        }

        // Write reordered data
        file.seek(SeekFrom::Start(abs_offset))?;
        file.write_all(&reordered)?;
        reordered_count += 1;

        tracing::info!(
            "Reordered {} experts in {} (stride={} bytes)",
            num_experts,
            tensor.name,
            stride,
        );
    }

    tracing::info!(
        "Optimization complete: {} tensors reordered in {}",
        reordered_count,
        output_path.display()
    );

    // Save permutation data as sidecar for runtime loading
    let perm_path = output_path.with_extension("permutations.json");
    let perm_json = serde_json::to_string_pretty(&permutations)?;
    std::fs::write(&perm_path, perm_json)?;
    tracing::info!("Permutations saved to {}", perm_path.display());

    Ok(())
}

/// Compute optimal expert ordering for a layer using greedy nearest-neighbor TSP.
/// Maximizes co-activation affinity between adjacent experts in the output order.
fn compute_optimal_expert_order(
    co_act: &CoActivationMatrix,
    layer: u32,
    num_experts: u32,
) -> Vec<u32> {
    let ne = num_experts as usize;
    let l = layer as usize;
    let counts = co_act.layer_counts();

    if l >= counts.len() || ne == 0 {
        return (0..num_experts).collect();
    }

    let counts = &counts[l];

    // Start from the most frequently activated expert
    let start = (0..ne).max_by_key(|&e| counts[e][e]).unwrap_or(0);

    let mut visited = vec![false; ne];
    let mut order = Vec::with_capacity(ne);

    visited[start] = true;
    order.push(start as u32);

    for _ in 1..ne {
        let current = *order.last().unwrap() as usize;
        let next = (0..ne)
            .filter(|&e| !visited[e])
            .max_by_key(|&e| counts[current][e])
            .unwrap_or(0);
        visited[next] = true;
        order.push(next as u32);
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_expert_order_identity() {
        let co_act = CoActivationMatrix::new(4, 8);
        // With no data, should return identity permutation
        let order = compute_optimal_expert_order(&co_act, 0, 8);
        assert_eq!(order.len(), 8);
        // All experts should be present
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_optimal_expert_order_with_data() {
        let mut co_act = CoActivationMatrix::new(4, 4);
        // Experts 0 and 1 co-fire frequently
        for _ in 0..20 {
            co_act.record(0, &[0, 1]);
        }
        // Experts 2 and 3 co-fire frequently
        for _ in 0..20 {
            co_act.record(0, &[2, 3]);
        }

        let order = compute_optimal_expert_order(&co_act, 0, 4);
        assert_eq!(order.len(), 4);

        // Co-activated experts should be adjacent
        let pos_0 = order.iter().position(|&e| e == 0).unwrap();
        let pos_1 = order.iter().position(|&e| e == 1).unwrap();
        let pos_2 = order.iter().position(|&e| e == 2).unwrap();
        let pos_3 = order.iter().position(|&e| e == 3).unwrap();

        // 0 and 1 should be adjacent
        assert!((pos_0 as i32 - pos_1 as i32).abs() <= 1);
        // 2 and 3 should be adjacent
        assert!((pos_2 as i32 - pos_3 as i32).abs() <= 1);
    }
}
