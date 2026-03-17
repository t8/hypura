use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Tracks expert co-activation patterns for speculative prefetch.
///
/// Maintains two matrices per layer:
/// - Same-layer: which experts tend to fire together (for top-k routing with k>1)
/// - Cross-layer: which experts at layer N predict which experts at layer N+1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoActivationMatrix {
    /// Per-layer symmetric co-activation counts: `[layer][expert_a][expert_b]`
    layer_counts: Vec<Vec<Vec<u32>>>,
    /// Cross-layer counts: `[layer][expert_at_N][expert_at_N+1]`
    cross_layer_counts: Vec<Vec<Vec<u32>>>,
    /// Total observations per layer
    observation_counts: Vec<u32>,
    num_layers: usize,
    num_experts: usize,
}

impl CoActivationMatrix {
    pub fn new(num_layers: u32, num_experts: u32) -> Self {
        let nl = num_layers as usize;
        let ne = num_experts as usize;
        Self {
            layer_counts: vec![vec![vec![0u32; ne]; ne]; nl],
            cross_layer_counts: vec![vec![vec![0u32; ne]; ne]; nl],
            observation_counts: vec![0u32; nl],
            num_layers: nl,
            num_experts: ne,
        }
    }

    /// Record which experts were selected at a given layer.
    pub fn record(&mut self, layer: u32, selected_experts: &[u32]) {
        let l = layer as usize;
        if l >= self.num_layers {
            return;
        }
        self.observation_counts[l] += 1;

        for i in 0..selected_experts.len() {
            let a = selected_experts[i] as usize;
            if a >= self.num_experts {
                continue;
            }
            self.layer_counts[l][a][a] += 1;
            for j in (i + 1)..selected_experts.len() {
                let b = selected_experts[j] as usize;
                if b >= self.num_experts {
                    continue;
                }
                self.layer_counts[l][a][b] += 1;
                self.layer_counts[l][b][a] += 1;
            }
        }
    }

    /// Record cross-layer correlation: experts at layer N predicting layer N+1.
    pub fn record_cross_layer(&mut self, layer: u32, experts_n: &[u32], experts_n1: &[u32]) {
        let l = layer as usize;
        if l >= self.num_layers {
            return;
        }
        for &a in experts_n {
            let a = a as usize;
            if a >= self.num_experts {
                continue;
            }
            for &b in experts_n1 {
                let b = b as usize;
                if b >= self.num_experts {
                    continue;
                }
                self.cross_layer_counts[l][a][b] += 1;
            }
        }
    }

    /// Predict which experts are likely to co-fire with `observed` at `layer`.
    pub fn predict_same_layer(&self, layer: u32, observed: &[u32], top_k: usize) -> Vec<u32> {
        let l = layer as usize;
        if l >= self.num_layers || self.observation_counts[l] == 0 {
            return Vec::new();
        }
        let mut scores = vec![0u32; self.num_experts];
        for &eid in observed {
            let e = eid as usize;
            if e < self.num_experts {
                for (i, &count) in self.layer_counts[l][e].iter().enumerate() {
                    scores[i] += count;
                }
            }
        }
        for &eid in observed {
            if (eid as usize) < self.num_experts {
                scores[eid as usize] = 0;
            }
        }
        let mut indexed: Vec<(u32, u32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i as u32, s))
            .filter(|(_, s)| *s > 0)
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.truncate(top_k);
        indexed.into_iter().map(|(id, _)| id).collect()
    }

    /// Predict which experts will fire at layer N+1 given experts at layer N.
    pub fn predict_next_layer(&self, layer: u32, observed: &[u32], top_k: usize) -> Vec<u32> {
        let l = layer as usize;
        if l >= self.num_layers || self.observation_counts[l] == 0 {
            return Vec::new();
        }
        let mut scores = vec![0u32; self.num_experts];
        for &eid in observed {
            let e = eid as usize;
            if e < self.num_experts {
                for (i, &count) in self.cross_layer_counts[l][e].iter().enumerate() {
                    scores[i] += count;
                }
            }
        }
        let mut indexed: Vec<(u32, u32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i as u32, s))
            .filter(|(_, s)| *s > 0)
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.truncate(top_k);
        indexed.into_iter().map(|(id, _)| id).collect()
    }

    pub fn has_data(&self) -> bool {
        self.observation_counts.iter().any(|&c| c > 10)
    }

    pub fn layer_counts(&self) -> &Vec<Vec<Vec<u32>>> {
        &self.layer_counts
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let matrix: Self = serde_json::from_str(&json)?;
        Ok(matrix)
    }

    pub fn persistence_path(model_path: &Path) -> PathBuf {
        let model_name = model_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".into());
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home)
            .join(".hypura")
            .join("coactivation")
            .join(format!("{model_name}.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_predict_same_layer() {
        let mut matrix = CoActivationMatrix::new(4, 8);
        for _ in 0..20 {
            matrix.record(0, &[0, 1]);
        }
        for _ in 0..5 {
            matrix.record(0, &[0, 2]);
        }
        let predicted = matrix.predict_same_layer(0, &[0], 2);
        assert!(!predicted.is_empty());
        assert_eq!(predicted[0], 1);
    }

    #[test]
    fn test_cross_layer_prediction() {
        let mut matrix = CoActivationMatrix::new(4, 8);
        matrix.observation_counts[0] = 10;
        for _ in 0..10 {
            matrix.record_cross_layer(0, &[0, 1], &[2, 3]);
        }
        let predicted = matrix.predict_next_layer(0, &[0, 1], 2);
        assert_eq!(predicted.len(), 2);
        assert!(predicted.contains(&2));
        assert!(predicted.contains(&3));
    }

    #[test]
    fn test_persistence() {
        let mut matrix = CoActivationMatrix::new(2, 4);
        matrix.record(0, &[0, 1]);
        matrix.record(1, &[2, 3]);

        let dir = std::env::temp_dir().join("hypura_test_coactivation");
        let path = dir.join("test_model.json");
        matrix.save(&path).unwrap();

        let loaded = CoActivationMatrix::load(&path).unwrap();
        assert_eq!(loaded.num_layers, 2);
        assert_eq!(loaded.num_experts, 4);
        assert_eq!(loaded.observation_counts[0], 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_empty_predictions() {
        let matrix = CoActivationMatrix::new(4, 8);
        assert!(matrix.predict_same_layer(0, &[0], 3).is_empty());
        assert!(matrix.predict_next_layer(0, &[0], 3).is_empty());
    }
}
