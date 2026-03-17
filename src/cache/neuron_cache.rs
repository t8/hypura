use std::collections::{HashMap, VecDeque};

use crate::io::expert_layout::ExpertTensorType;

/// Cache key: (layer_index, expert_id, tensor_type).
type CacheKey = (u32, u32, ExpertTensorType);

/// LRU cache tracking which expert slices are currently loaded in the
/// posix_memalign buffer. Expert slices live at fixed positions in the buffer —
/// this cache does NOT allocate separate memory. It only tracks which positions
/// contain valid data (loaded via pread) vs stale data (released via MADV_FREE).
///
/// When a cache entry is evicted, the corresponding buffer region is released.
/// When a cache hit occurs, no I/O is needed — the data is already in place.
///
/// Typical capacity for Mixtral 8x7B on M1 Max 32GB:
/// - 2 NVMe layers × 3 tensor types × 3 hot experts = 18 entries (~640 MB)
/// - Expected hit rate: ~96.5% (2/8 experts used per token, strong temporal locality)
pub struct NeuronCache {
    /// Set of currently-loaded expert slices.
    entries: HashMap<CacheKey, ()>,
    /// LRU order: front = least recently used, back = most recently used.
    lru_order: VecDeque<CacheKey>,
    /// Maximum number of expert slices to keep loaded.
    capacity: usize,
    /// Cache statistics.
    pub hits: u64,
    pub misses: u64,
}

impl NeuronCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            lru_order: VecDeque::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    /// Check if an expert slice is currently loaded (cache hit).
    /// Promotes the entry to most-recently-used on hit.
    pub fn is_loaded(&mut self, layer: u32, expert_id: u32, tensor_type: ExpertTensorType) -> bool {
        let key = (layer, expert_id, tensor_type);
        if self.entries.contains_key(&key) {
            self.hits += 1;
            // Promote to MRU
            self.lru_order.retain(|k| *k != key);
            self.lru_order.push_back(key);
            true
        } else {
            self.misses += 1;
            false
        }
    }

    /// Mark an expert slice as loaded. Evicts LRU entry if at capacity.
    /// Returns the evicted key if one was removed (caller should MADV_FREE it).
    pub fn mark_loaded(
        &mut self,
        layer: u32,
        expert_id: u32,
        tensor_type: ExpertTensorType,
    ) -> Option<CacheKey> {
        let key = (layer, expert_id, tensor_type);

        // Already loaded — just promote
        if self.entries.contains_key(&key) {
            self.lru_order.retain(|k| *k != key);
            self.lru_order.push_back(key);
            return None;
        }

        // Evict LRU if at capacity
        let evicted = if self.entries.len() >= self.capacity {
            self.evict_lru()
        } else {
            None
        };

        self.entries.insert(key, ());
        self.lru_order.push_back(key);
        evicted
    }

    /// Evict the least recently used entry. Returns the evicted key.
    fn evict_lru(&mut self) -> Option<CacheKey> {
        if let Some(key) = self.lru_order.pop_front() {
            self.entries.remove(&key);
            Some(key)
        } else {
            None
        }
    }

    /// Evict all entries for a given layer (called when releasing a layer's pages).
    pub fn evict_layer(&mut self, layer: u32) {
        self.entries.retain(|&(l, _, _), _| l != layer);
        self.lru_order.retain(|&(l, _, _)| l != layer);
    }

    /// Current number of loaded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Cache hit rate as a fraction [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let mut cache = NeuronCache::new(4);

        // Miss on first access
        assert!(!cache.is_loaded(0, 0, ExpertTensorType::Gate));
        assert_eq!(cache.misses, 1);

        // Mark loaded
        cache.mark_loaded(0, 0, ExpertTensorType::Gate);
        assert!(cache.is_loaded(0, 0, ExpertTensorType::Gate));
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = NeuronCache::new(3);

        cache.mark_loaded(0, 0, ExpertTensorType::Gate);
        cache.mark_loaded(0, 1, ExpertTensorType::Gate);
        cache.mark_loaded(0, 2, ExpertTensorType::Gate);
        assert_eq!(cache.len(), 3);

        // Adding a 4th should evict the LRU (0, 0, Gate)
        let evicted = cache.mark_loaded(0, 3, ExpertTensorType::Gate);
        assert_eq!(evicted, Some((0, 0, ExpertTensorType::Gate)));
        assert_eq!(cache.len(), 3);

        // Evicted entry should be a miss
        assert!(!cache.is_loaded(0, 0, ExpertTensorType::Gate));
        // Others should still be loaded
        assert!(cache.is_loaded(0, 1, ExpertTensorType::Gate));
        assert!(cache.is_loaded(0, 2, ExpertTensorType::Gate));
        assert!(cache.is_loaded(0, 3, ExpertTensorType::Gate));
    }

    #[test]
    fn test_cache_lru_promotion() {
        let mut cache = NeuronCache::new(3);

        cache.mark_loaded(0, 0, ExpertTensorType::Gate);
        cache.mark_loaded(0, 1, ExpertTensorType::Gate);
        cache.mark_loaded(0, 2, ExpertTensorType::Gate);

        // Access (0, 0) to promote it — now (0, 1) is LRU
        cache.is_loaded(0, 0, ExpertTensorType::Gate);

        let evicted = cache.mark_loaded(0, 3, ExpertTensorType::Gate);
        assert_eq!(evicted, Some((0, 1, ExpertTensorType::Gate)));
    }

    #[test]
    fn test_cache_evict_layer() {
        let mut cache = NeuronCache::new(10);

        cache.mark_loaded(0, 0, ExpertTensorType::Gate);
        cache.mark_loaded(0, 1, ExpertTensorType::Up);
        cache.mark_loaded(1, 0, ExpertTensorType::Gate);
        cache.mark_loaded(1, 1, ExpertTensorType::Down);
        assert_eq!(cache.len(), 4);

        cache.evict_layer(0);
        assert_eq!(cache.len(), 2);
        assert!(!cache.is_loaded(0, 0, ExpertTensorType::Gate));
        assert!(cache.is_loaded(1, 0, ExpertTensorType::Gate));
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = NeuronCache::new(10);

        cache.mark_loaded(0, 0, ExpertTensorType::Gate);

        // 3 hits
        cache.is_loaded(0, 0, ExpertTensorType::Gate);
        cache.is_loaded(0, 0, ExpertTensorType::Gate);
        cache.is_loaded(0, 0, ExpertTensorType::Gate);

        // 1 miss
        cache.is_loaded(0, 1, ExpertTensorType::Gate);

        assert_eq!(cache.hits, 3);
        assert_eq!(cache.misses, 1);
        assert!((cache.hit_rate() - 0.75).abs() < 0.001);
    }
}
