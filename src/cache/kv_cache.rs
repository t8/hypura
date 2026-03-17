use crate::compute::ffi::LlamaContext;

/// Manages KV cache windowing to bound memory usage for long contexts.
///
/// When context exceeds the hot window, evicts oldest positions using
/// `llama_memory_seq_rm`. This keeps KV cache bounded regardless of
/// total context length, enabling 16K+ context on memory-constrained devices.
pub struct KvCacheManager {
    pub hot_window_tokens: u32,
    current_pos: u32,
    compact_interval: u32,
    tokens_since_check: u32,
}

impl KvCacheManager {
    pub fn new(hot_window_tokens: u32) -> Self {
        Self {
            hot_window_tokens,
            current_pos: 0,
            compact_interval: 256,
            tokens_since_check: 0,
        }
    }

    /// Advance position by one token. Returns true if compaction was performed.
    pub fn advance(&mut self, ctx: &LlamaContext) -> bool {
        self.current_pos += 1;
        self.tokens_since_check += 1;

        if self.tokens_since_check >= self.compact_interval {
            self.tokens_since_check = 0;
            return self.check_and_compact(ctx);
        }
        false
    }

    /// Check if the KV cache needs compaction and perform it if so.
    pub fn check_and_compact(&mut self, ctx: &LlamaContext) -> bool {
        if self.hot_window_tokens == 0 || self.current_pos <= self.hot_window_tokens {
            return false;
        }

        let evict_before = (self.current_pos - self.hot_window_tokens) as i32;

        let mem = unsafe { hypura_sys::llama_get_memory(ctx.as_ptr()) };
        if mem.is_null() {
            return false;
        }

        let pos_min = unsafe { hypura_sys::llama_memory_seq_pos_min(mem, 0) };
        if pos_min >= evict_before {
            return false;
        }

        tracing::debug!(
            "KV compaction: evicting positions {} to {} (keeping {} to {})",
            pos_min,
            evict_before - 1,
            evict_before,
            self.current_pos
        );

        let removed = unsafe { hypura_sys::llama_memory_seq_rm(mem, 0, pos_min, evict_before) };
        if removed {
            tracing::trace!(
                "KV cache compacted: removed positions [{}, {})",
                pos_min,
                evict_before
            );
        }
        removed
    }

    pub fn set_position(&mut self, pos: u32) {
        self.current_pos = pos;
    }

    pub fn position(&self) -> u32 {
        self.current_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_manager_new() {
        let mgr = KvCacheManager::new(4096);
        assert_eq!(mgr.hot_window_tokens, 4096);
        assert_eq!(mgr.current_pos, 0);
    }

    #[test]
    fn test_position_tracking() {
        let mut mgr = KvCacheManager::new(4096);
        mgr.set_position(100);
        assert_eq!(mgr.position(), 100);
    }
}
