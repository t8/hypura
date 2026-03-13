use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::scheduler::types::StorageTier;

/// Real-time telemetry events emitted during inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEvent {
    TokenGenerated {
        tok_per_sec: f64,
        token: String,
    },
    PrefetchStatus {
        hit_rate: f64,
        nvme_mbps: f64,
    },
    KvCacheUpdate {
        hot_tokens: u32,
        warm_tokens: u32,
    },
    LayerComputed {
        layer_idx: u32,
        tier: StorageTier,
        duration_us: u64,
    },
    TierRead {
        tier: StorageTier,
        bytes: u64,
        latency_us: u64,
    },
}

/// Emitter for telemetry events. Subscribers receive events via broadcast channel.
pub struct TelemetryEmitter {
    tx: broadcast::Sender<TelemetryEvent>,
}

impl TelemetryEmitter {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn emit(&self, event: TelemetryEvent) {
        // Ignore errors (no subscribers)
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<TelemetryEvent> {
        self.tx.subscribe()
    }
}

/// Aggregated telemetry for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySummary {
    pub tokens_per_second: f64,
    pub gpu_bandwidth_utilization: f64,
    pub ram_bandwidth_utilization: f64,
    pub nvme_read_mbps: f64,
    pub prefetch_hit_rate: f64,
    pub kv_cache_hot_tokens: u32,
    pub kv_cache_warm_tokens: u32,
}
