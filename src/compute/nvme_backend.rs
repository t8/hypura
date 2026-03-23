use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use crate::io::compat::{self, NativeFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::cache::coactivation::CoActivationMatrix;
use crate::cache::neuron_cache::NeuronCache;
use crate::io::expert_layout::{ExpertLayout, ExpertTensorType};
use crate::model::gguf::GgufFile;
use crate::model::tensor_role::TensorRole;
use crate::scheduler::types::*;

/// Metadata about a tensor in our custom buffer.
#[derive(Debug, Clone)]
pub struct TensorLocation {
    pub offset_in_buffer: usize,
    pub size: usize,
    pub file_offset: u64,
    pub layer_index: Option<u32>,
}

/// Status of a layer's tensor data in physical memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerStatus {
    NotLoaded,
    Loading,
    Loaded,
}

/// Message types for prefetch requests (high-level API).
pub enum PrefetchRequest {
    /// Load an entire layer's data from NVMe.
    Layer(u32),
    /// Load specific expert slices for a layer (speculative prefetch).
    ExpertSlices {
        layer_idx: u32,
        expert_ids: Vec<u32>,
    },
}

// --- I/O Pool types (Phase 1) ---

/// A single unit of I/O work for the pool workers.
struct IoPoolTask {
    /// Regions to pread: (buffer_offset, size, file_offset)
    regions: Vec<(usize, usize, u64)>,
    /// Buffer base pointer
    base: *mut u8,
    /// Shared completion tracking
    completion: Arc<IoCompletion>,
}

// SAFETY: base pointer is a stable posix_memalign allocation shared via IoPool protocol.
// Workers write to non-overlapping regions guaranteed by LoadUnit decomposition.
unsafe impl Send for IoPoolTask {}

/// Tracks completion of a multi-unit layer load.
struct IoCompletion {
    /// Number of remaining tasks. When 0, all tasks are complete.
    remaining: AtomicUsize,
    /// Which layer this load is for.
    layer_idx: u32,
    /// Whether to update LayerStatus::Loaded on completion.
    update_status: bool,
}

/// Multi-threaded I/O pool for NVMe reads. Manages worker threads,
/// each with its own F_NOCACHE fd to the model file.
pub struct IoPool {
    /// Channel for submitting tasks (None when pool is stopping).
    tx: Option<std::sync::mpsc::Sender<IoPoolTask>>,
    /// Worker thread handles.
    handles: Vec<std::thread::JoinHandle<()>>,
    /// Per-worker file descriptors / handles (for cleanup).
    worker_fds: Vec<NativeFd>,
    /// Throughput tracking: total bytes loaded by all workers.
    pub bytes_loaded: Arc<AtomicU64>,
    /// Throughput tracking: total load time in nanoseconds.
    pub load_time_ns: Arc<AtomicU64>,
}

impl IoPool {
    fn new(
        model_path: &Path,
        num_workers: usize,
        state: Arc<PrefetchState>,
    ) -> anyhow::Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<IoPoolTask>();
        let rx = Arc::new(Mutex::new(rx));

        let bytes_loaded = Arc::new(AtomicU64::new(0));
        let load_time_ns = Arc::new(AtomicU64::new(0));

        let mut worker_fds = Vec::with_capacity(num_workers);
        let mut handles = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            let fd = compat::open_direct_fd(model_path)?;
            worker_fds.push(fd);

            let rx = rx.clone();
            let state = state.clone();
            let bytes_loaded = bytes_loaded.clone();
            let load_time_ns = load_time_ns.clone();

            let handle = std::thread::Builder::new()
                .name(format!("hypura-io-{i}"))
                .spawn(move || io_worker(fd, rx, state, bytes_loaded, load_time_ns))
                .expect("failed to spawn I/O worker");
            handles.push(handle);
        }

        tracing::info!("I/O pool started: {} workers", num_workers);

        Ok(Self {
            tx: Some(tx),
            handles,
            worker_fds,
            bytes_loaded,
            load_time_ns,
        })
    }

    pub fn num_workers(&self) -> usize {
        self.worker_fds.len()
    }

    /// Measured throughput in bytes per second.
    pub fn throughput_bps(&self) -> f64 {
        let bytes = self.bytes_loaded.load(Ordering::Relaxed) as f64;
        let ns = self.load_time_ns.load(Ordering::Relaxed) as f64;
        if ns > 0.0 {
            bytes / (ns / 1e9)
        } else {
            0.0
        }
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        // Close channel to signal workers to exit
        self.tx.take();

        // Wait for workers to finish
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }

        // Close per-worker fds / handles
        for fd in &self.worker_fds {
            compat::close_fd(*fd);
        }
    }
}

/// I/O worker thread: pulls tasks from shared channel, executes pread / ReadFile.
fn io_worker(
    fd: NativeFd,
    rx: Arc<Mutex<std::sync::mpsc::Receiver<IoPoolTask>>>,
    state: Arc<PrefetchState>,
    bytes_loaded: Arc<AtomicU64>,
    load_time_ns: Arc<AtomicU64>,
) {
    loop {
        let task = {
            let rx = rx.lock().unwrap();
            match rx.recv() {
                Ok(task) => task,
                Err(_) => return, // Channel closed
            }
        };

        let start = Instant::now();
        let mut total = 0u64;

        for &(offset, size, file_offset) in &task.regions {
            pread_region(fd, task.base, offset, size, file_offset);
            total += size as u64;
        }

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        bytes_loaded.fetch_add(total, Ordering::Relaxed);
        load_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

        // Decrement completion counter
        let prev = task.completion.remaining.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            if state.trace_enabled.load(Ordering::Relaxed) {
                state.trace.record(TraceEvent::LoadComplete {
                    layer: task.completion.layer_idx,
                    bytes: total,
                    io_ms: elapsed_ns as f64 / 1e6,
                });
            }
            if task.completion.update_status {
                // Last task for this layer — mark complete
                let mut status = state.layer_status.lock().unwrap();
                status.insert(task.completion.layer_idx, LayerStatus::Loaded);
                state.layer_notify.notify_all();
            }
        }
    }
}

/// Perform positional I/O for a single region (pread on Unix, ReadFile on Windows).
fn pread_region(fd: NativeFd, base: *mut u8, offset: usize, size: usize, file_offset: u64) {
    let dst = unsafe { base.add(offset) };
    let mut read = 0usize;
    while read < size {
        let n = {
            compat::read_at_fd(fd, unsafe { dst.add(read) }, size - read, file_offset + read as u64)
        };
        if n <= 0 {
            break;
        }
        read += n as usize;
    }
}

/// Trace event for I/O timeline analysis.
#[derive(Debug, Clone)]
pub enum TraceEvent {
    /// ensure_layer_loaded found the layer already loaded (no I/O wait).
    LayerHit(u32),
    /// ensure_layer_loaded had to wait for I/O to complete.
    LayerStall { layer: u32, wait_ms: f64 },
    /// I/O pool completed loading a layer.
    LoadComplete { layer: u32, bytes: u64, io_ms: f64 },
    /// Layer pages released via MADV_FREE.
    Released(u32),
    /// ctx.decode completed for one token.
    DecodeComplete { decode_ms: f64 },
}

/// Collects timestamped trace events during inference for post-hoc analysis.
pub struct IoTrace {
    start: Instant,
    events: Mutex<Vec<(f64, TraceEvent)>>,
}

impl IoTrace {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            events: Mutex::new(Vec::with_capacity(4096)),
        }
    }

    fn record(&self, event: TraceEvent) {
        let ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.events.lock().unwrap().push((ms, event));
    }
}

/// Shared state between the eval callback and the inference engine.
/// This struct is passed as `user_data` to `cb_eval`.
pub struct PrefetchState {
    pub current_layer: AtomicI32,
    pub tensor_map: HashMap<String, TensorLocation>,
    pub model_path: PathBuf,
    /// Buffer base pointer (set during model loading via callback)
    pub buffer_base: Mutex<*mut u8>,
    /// Layers grouped by index: layer_idx -> Vec<(offset_in_buffer, size, file_offset)>
    pub layer_regions: HashMap<u32, Vec<(usize, usize, u64)>>,
    /// Track layer load status for async prefetch coordination
    pub layer_status: Mutex<HashMap<u32, LayerStatus>>,
    /// Notify waiters when a layer finishes loading
    pub layer_notify: Condvar,
    /// Total number of layers
    pub num_layers: u32,
    /// Whether prefetch is enabled
    pub prefetch_enabled: AtomicBool,
    /// Multi-threaded I/O pool (replaces single prefetch thread + nvme_fd)
    pub io_pool: Mutex<Option<IoPool>>,
    /// Layer indices that are on NVMe (released after use, re-loaded as needed).
    pub nvme_layers: std::collections::HashSet<u32>,
    /// When true, NVMe layers are kept in physical memory after first load
    pub keep_nvme_resident: AtomicBool,
    /// NVMe layer indices sorted ascending for sequential prefetch ordering.
    pub sorted_nvme_layers: Vec<u32>,

    // --- Expert-level data structures ---

    /// Per-layer expert tensor layouts for fused expert tensors.
    pub expert_layouts: HashMap<u32, Vec<ExpertLayout>>,
    /// Per-layer non-expert tensor regions (norms, router weights, etc.).
    pub non_expert_regions: HashMap<u32, Vec<(usize, usize, u64)>>,

    // --- Router interception ---

    /// Per-layer expert selection from the most recent router output.
    pub selected_experts: Mutex<HashMap<u32, Vec<u32>>>,
    pub num_experts_used: u32,
    pub num_experts_total: u32,

    // --- Neuron cache ---

    pub neuron_cache: Mutex<NeuronCache>,
    pub debug_logged_tensors: AtomicI32,

    // --- Co-activation tracking (Phase 2) ---

    pub co_activation: Mutex<CoActivationMatrix>,
    /// Previous layer expert selections for cross-layer tracking.
    pub prev_layer_experts: Mutex<Option<(u32, Vec<u32>)>>,

    // --- I/O tracing ---

    /// Expert-streaming mode: non-expert tensors GPU-resident, experts loaded on demand.
    pub expert_streaming: AtomicBool,
    /// Dense FFN-streaming mode: attention+norms GPU-resident, FFN streamed from NVMe.
    pub dense_ffn_streaming: AtomicBool,
    /// Per-layer dense FFN tensor info for pool-based loading.
    pub dense_ffn_layouts: HashMap<u32, Vec<DenseFfnLayout>>,
    /// Prefetch lookahead depth for dense FFN streaming (scales with pool slots).
    pub dense_ffn_lookahead: u32,
    /// Pool buffer for expert-streaming (slot allocator + pool base).
    pub expert_pool: Mutex<Option<ExpertPool>>,
    /// Captured ggml_tensor* pointers for fused expert tensors (for data pointer rewriting).
    pub fused_tensor_ptrs: HashMap<String, *mut hypura_sys::ggml_tensor>,

    // --- Hybrid residency (dense FFN) ---

    /// Layers whose FFN data is permanently resident in memory (not pool-managed).
    pub resident_ffn_layers: std::collections::HashSet<u32>,
    /// Base pointer of the resident FFN buffer (separate from pool).
    pub resident_ffn_base: *mut u8,
    /// Total size of the resident FFN buffer (for munmap on drop).
    pub resident_ffn_size: usize,
    /// Per-resident-layer tensor: tensor_name → offset within resident buffer.
    pub resident_ffn_offsets: HashMap<String, usize>,

    /// When true, record timestamped I/O events for post-hoc analysis.
    pub trace_enabled: AtomicBool,
    /// Trace event log. Only populated when `trace_enabled` is true.
    pub trace: IoTrace,
}

// SAFETY: buffer_base accessed under Mutex; raw pointers managed by IoPool protocol
unsafe impl Send for PrefetchState {}
unsafe impl Sync for PrefetchState {}

impl PrefetchState {
    /// Submit a layer load to the I/O pool. Decomposes into tasks based on
    /// expert-aware loading when router selections are available.
    fn submit_layer_load(&self, layer_idx: u32) {
        let base = *self.buffer_base.lock().unwrap();
        if base.is_null() {
            return;
        }

        let maybe_experts = self
            .selected_experts
            .lock()
            .unwrap()
            .get(&layer_idx)
            .cloned();
        let has_expert_layouts = self.expert_layouts.contains_key(&layer_idx);

        let mut task_regions: Vec<Vec<(usize, usize, u64)>> = Vec::new();

        if let (Some(ref experts), true) = (&maybe_experts, has_expert_layouts) {
            tracing::trace!(
                "Expert-aware load: layer {} experts {:?}",
                layer_idx,
                experts
            );

            // Non-expert regions as one task
            if let Some(regions) = self.non_expert_regions.get(&layer_idx) {
                if !regions.is_empty() {
                    task_regions.push(regions.clone());
                }
            }

            // Each fused expert tensor as a separate task
            let mut cache = self.neuron_cache.lock().unwrap();
            if let Some(layouts) = self.expert_layouts.get(&layer_idx) {
                for layout in layouts {
                    let tensor_type = ExpertTensorType::from_name(&layout.tensor_name)
                        .unwrap_or(ExpertTensorType::Gate);
                    let mut regions = Vec::new();
                    for &eid in experts {
                        if eid >= layout.num_experts {
                            continue;
                        }
                        if cache.is_loaded(layer_idx, eid, tensor_type) {
                            continue;
                        }
                        regions.push((
                            layout.expert_buffer_offset(eid),
                            layout.expert_stride,
                            layout.expert_file_offset(eid),
                        ));
                        cache.mark_loaded(layer_idx, eid, tensor_type);
                    }
                    if !regions.is_empty() {
                        task_regions.push(regions);
                    }
                }
            }
        } else {
            // Full layer load — split regions across workers for parallel I/O.
            // Each worker reads different file offsets concurrently, which saturates
            // the NVMe controller better than a single sequential reader.
            if let Some(regions) = self.layer_regions.get(&layer_idx) {
                if !regions.is_empty() {
                    let num_workers = self
                        .io_pool
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map_or(1, |p| p.num_workers().max(1));
                    let chunk_size = (regions.len() + num_workers - 1) / num_workers;
                    for chunk in regions.chunks(chunk_size) {
                        task_regions.push(chunk.to_vec());
                    }
                }
            }
        }

        if task_regions.is_empty() {
            // Nothing to load — mark as Loaded immediately
            let mut status = self.layer_status.lock().unwrap();
            status.insert(layer_idx, LayerStatus::Loaded);
            self.layer_notify.notify_all();
            return;
        }

        let completion = Arc::new(IoCompletion {
            remaining: AtomicUsize::new(task_regions.len()),
            layer_idx,
            update_status: true,
        });

        let pool = self.io_pool.lock().unwrap();
        if let Some(ref pool) = *pool {
            if let Some(ref tx) = pool.tx {
                for regions in task_regions {
                    let _ = tx.send(IoPoolTask {
                        regions,
                        base,
                        completion: completion.clone(),
                    });
                }
            }
        } else {
            // No pool — mark as Loaded to avoid deadlock
            tracing::warn!("IoPool not started, cannot load layer {}", layer_idx);
            drop(pool);
            let mut status = self.layer_status.lock().unwrap();
            status.insert(layer_idx, LayerStatus::Loaded);
            self.layer_notify.notify_all();
        }
    }

    /// Submit expert-only loads for speculative prefetch (no layer status change).
    fn submit_expert_load(&self, layer_idx: u32, expert_ids: &[u32]) {
        if !self.expert_layouts.contains_key(&layer_idx) {
            return;
        }

        let base = *self.buffer_base.lock().unwrap();
        if base.is_null() {
            return;
        }

        let mut task_regions: Vec<Vec<(usize, usize, u64)>> = Vec::new();

        // Non-expert regions
        if let Some(regions) = self.non_expert_regions.get(&layer_idx) {
            if !regions.is_empty() {
                task_regions.push(regions.clone());
            }
        }

        // Expert strides
        {
            let mut cache = self.neuron_cache.lock().unwrap();
            if let Some(layouts) = self.expert_layouts.get(&layer_idx) {
                for layout in layouts {
                    let tensor_type = ExpertTensorType::from_name(&layout.tensor_name)
                        .unwrap_or(ExpertTensorType::Gate);
                    let mut regions = Vec::new();
                    for &eid in expert_ids {
                        if eid >= layout.num_experts {
                            continue;
                        }
                        if cache.is_loaded(layer_idx, eid, tensor_type) {
                            continue;
                        }
                        regions.push((
                            layout.expert_buffer_offset(eid),
                            layout.expert_stride,
                            layout.expert_file_offset(eid),
                        ));
                        cache.mark_loaded(layer_idx, eid, tensor_type);
                    }
                    if !regions.is_empty() {
                        task_regions.push(regions);
                    }
                }
            }
        }

        if task_regions.is_empty() {
            return;
        }

        // Expert-only loads don't update layer status
        let completion = Arc::new(IoCompletion {
            remaining: AtomicUsize::new(task_regions.len()),
            layer_idx,
            update_status: false,
        });

        let pool = self.io_pool.lock().unwrap();
        if let Some(ref pool) = *pool {
            if let Some(ref tx) = pool.tx {
                for regions in task_regions {
                    let _ = tx.send(IoPoolTask {
                        regions,
                        base,
                        completion: completion.clone(),
                    });
                }
            }
        }
    }

    /// Ensure a layer's tensor data is loaded in physical memory.
    /// Submits to the I/O pool and waits for completion.
    pub fn ensure_layer_loaded(&self, layer_idx: u32) {
        let tracing = self.trace_enabled.load(Ordering::Relaxed);
        let mut status = self.layer_status.lock().unwrap();
        loop {
            match status.get(&layer_idx).copied() {
                Some(LayerStatus::Loaded) => {
                    if tracing {
                        self.trace.record(TraceEvent::LayerHit(layer_idx));
                    }
                    return;
                }
                Some(LayerStatus::Loading) => {
                    // I/O pool or another thread is loading — wait
                    let wait_start = Instant::now();
                    status = self.layer_notify.wait(status).unwrap();
                    if tracing
                        && status.get(&layer_idx).copied() == Some(LayerStatus::Loaded)
                    {
                        let wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
                        self.trace.record(TraceEvent::LayerStall {
                            layer: layer_idx,
                            wait_ms,
                        });
                    }
                }
                _ => {
                    // Not loaded — submit to I/O pool and wait
                    let wait_start = Instant::now();
                    status.insert(layer_idx, LayerStatus::Loading);
                    drop(status);

                    self.submit_layer_load(layer_idx);

                    // Wait for completion (pool workers set Loaded + notify)
                    let mut status = self.layer_status.lock().unwrap();
                    while status.get(&layer_idx).copied() != Some(LayerStatus::Loaded) {
                        status = self.layer_notify.wait(status).unwrap();
                    }
                    if tracing {
                        let wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
                        self.trace.record(TraceEvent::LayerStall {
                            layer: layer_idx,
                            wait_ms,
                        });
                    }
                    return;
                }
            }
        }
    }

    /// Release physical pages for a layer's tensors.
    pub fn release_layer(&self, layer_idx: u32) {
        let regions = match self.layer_regions.get(&layer_idx) {
            Some(r) => r,
            None => return,
        };

        let base = *self.buffer_base.lock().unwrap();
        if base.is_null() {
            return;
        }

        for &(offset, size, _) in regions {
            let ptr = unsafe { base.add(offset) };
            compat::advise_free_pages(ptr, size);
        }

        // Invalidate neuron cache entries for this layer
        self.neuron_cache.lock().unwrap().evict_layer(layer_idx);

        self.layer_status
            .lock()
            .unwrap()
            .insert(layer_idx, LayerStatus::NotLoaded);

        if self.trace_enabled.load(Ordering::Relaxed) {
            self.trace.record(TraceEvent::Released(layer_idx));
        }
    }

    /// Pre-load all RAM-tier layers via the I/O pool.
    pub fn preload_ram_layers(&self) {
        let ram_layers: Vec<u32> = self
            .layer_regions
            .keys()
            .filter(|l| !self.nvme_layers.contains(l))
            .copied()
            .collect();

        if ram_layers.is_empty() {
            return;
        }
        tracing::info!("Pre-loading {} RAM-tier layers", ram_layers.len());

        // Submit all layers to pool
        for &layer_idx in &ram_layers {
            self.request_prefetch(PrefetchRequest::Layer(layer_idx));
        }

        // Wait for all to complete
        for &layer_idx in &ram_layers {
            let mut status = self.layer_status.lock().unwrap();
            while status.get(&layer_idx).copied() != Some(LayerStatus::Loaded) {
                status = self.layer_notify.wait(status).unwrap();
            }
        }
    }

    /// Request prefetch for all NVMe layers (sorted for sequential I/O).
    pub fn prefetch_all_nvme(&self) {
        for &layer in &self.sorted_nvme_layers {
            self.request_prefetch(PrefetchRequest::Layer(layer));
        }
    }

    /// Send a non-blocking prefetch request to the I/O pool.
    pub fn request_prefetch(&self, request: PrefetchRequest) {
        match request {
            PrefetchRequest::Layer(layer_idx) => {
                let mut status = self.layer_status.lock().unwrap();
                match status.get(&layer_idx).copied() {
                    Some(LayerStatus::Loaded) | Some(LayerStatus::Loading) => return,
                    _ => {}
                }
                status.insert(layer_idx, LayerStatus::Loading);
                drop(status);
                self.submit_layer_load(layer_idx);
            }
            PrefetchRequest::ExpertSlices {
                layer_idx,
                expert_ids,
            } => {
                self.submit_expert_load(layer_idx, &expert_ids);
            }
        }
    }

    /// Start the multi-threaded I/O pool.
    pub fn start_io_pool(self: &Arc<Self>, num_workers: usize) -> anyhow::Result<()> {
        let pool = IoPool::new(&self.model_path, num_workers, self.clone())?;
        *self.io_pool.lock().unwrap() = Some(pool);
        Ok(())
    }

    /// Stop the I/O pool and wait for workers to finish.
    pub fn stop_io_pool(&self) {
        // Take the pool — IoPool::drop closes channel, joins workers, closes fds
        self.io_pool.lock().unwrap().take();
    }

    /// Compute adaptive prefetch lookahead based on measured I/O throughput.
    fn adaptive_lookahead(&self) -> u32 {
        let pool = self.io_pool.lock().unwrap();
        let throughput = pool.as_ref().map_or(3e9, |p| {
            let t = p.throughput_bps();
            if t > 0.0 {
                t
            } else {
                3e9 // Default 3 GB/s
            }
        });
        drop(pool);

        if self.sorted_nvme_layers.is_empty() {
            return 4;
        }

        let avg_layer_bytes: u64 = self
            .sorted_nvme_layers
            .iter()
            .filter_map(|l| self.layer_regions.get(l))
            .map(|regions| regions.iter().map(|r| r.1 as u64).sum::<u64>())
            .sum::<u64>()
            .checked_div(self.sorted_nvme_layers.len() as u64)
            .unwrap_or(0);

        if avg_layer_bytes == 0 {
            return 4;
        }

        // Conservative: assume 100ms compute per layer
        let compute_time_secs = 0.1;
        let loadable_per_compute = throughput * compute_time_secs;
        let needed = (avg_layer_bytes as f64 / loadable_per_compute).ceil() as u32;
        needed.clamp(2, 8)
    }

    /// Load selected experts for a layer, blocking until complete.
    /// Uses neuron cache to skip already-loaded expert strides.
    /// In pool mode, allocates pool slots, rewrites tensor->data, and preads into pool.
    pub fn ensure_experts_loaded(&self, layer_idx: u32, expert_ids: &[u32]) {
        if expert_ids.is_empty() || !self.expert_layouts.contains_key(&layer_idx) {
            return;
        }

        let tracing_on = self.trace_enabled.load(Ordering::Relaxed);
        let use_pool = self.expert_pool.lock().unwrap().is_some();

        // Check which experts need loading (cache misses)
        let needs_load = {
            let mut cache = self.neuron_cache.lock().unwrap();
            let mut missing = false;
            if let Some(layouts) = self.expert_layouts.get(&layer_idx) {
                for &eid in expert_ids {
                    for layout in layouts {
                        let tt = ExpertTensorType::from_name(&layout.tensor_name)
                            .unwrap_or(ExpertTensorType::Gate);
                        if !cache.is_loaded(layer_idx, eid, tt) {
                            missing = true;
                            break;
                        }
                    }
                    if missing {
                        break;
                    }
                }
            }
            missing
        };

        if !needs_load {
            // Even cache hits need tensor->data rewriting in pool mode
            if use_pool {
                self.rewrite_tensor_ptrs_for_layer(layer_idx);
            }
            if tracing_on {
                self.trace.record(TraceEvent::LayerHit(layer_idx));
            }
            return;
        }

        let wait_start = Instant::now();

        if use_pool {
            // Pool mode: allocate slots, rewrite tensor->data, pread into pool
            self.load_experts_pooled(layer_idx, expert_ids);
        } else {
            // Standard mode: pread into the main buffer at fixed offsets
            {
                let mut status = self.layer_status.lock().unwrap();
                status.insert(layer_idx, LayerStatus::Loading);
            }
            self.submit_expert_load_with_status(layer_idx, expert_ids);
            let mut status = self.layer_status.lock().unwrap();
            while status.get(&layer_idx).copied() != Some(LayerStatus::Loaded) {
                status = self.layer_notify.wait(status).unwrap();
            }
        }

        if tracing_on {
            let wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
            self.trace
                .record(TraceEvent::LayerStall { layer: layer_idx, wait_ms });
        }
    }

    /// Pool mode: allocate slots for a layer, rewrite tensor->data, pread expert strides.
    fn load_experts_pooled(&self, layer_idx: u32, expert_ids: &[u32]) {
        let layouts = match self.expert_layouts.get(&layer_idx) {
            Some(l) => l,
            None => return,
        };

        // Allocate pool slots
        let slot_offsets = {
            let mut pool = self.expert_pool.lock().unwrap();
            let pool = pool.as_mut().expect("pool not initialized");
            // Evict old layer's neuron cache entries if pool reclaims slots
            let evicted_layers: Vec<u32> = pool
                .layer_slots
                .keys()
                .copied()
                .filter(|&l| l != layer_idx && pool.free_slots.len() < layouts.len())
                .collect();
            for el in &evicted_layers {
                self.neuron_cache.lock().unwrap().evict_layer(*el);
            }
            pool.allocate_layer(layer_idx, layouts.len())
        };

        // Rewrite tensor->data pointers to pool slots
        for (i, layout) in layouts.iter().enumerate() {
            if let Some(&tensor_ptr) = self.fused_tensor_ptrs.get(&layout.tensor_name) {
                if !tensor_ptr.is_null() {
                    let pool = self.expert_pool.lock().unwrap();
                    let pool = pool.as_ref().unwrap();
                    unsafe {
                        (*tensor_ptr).data =
                            pool.pool_base.add(slot_offsets[i]) as *mut c_void;
                    }
                }
            }
        }

        // Build pread regions targeting pool slots
        let pool_base = {
            let pool = self.expert_pool.lock().unwrap();
            pool.as_ref().unwrap().pool_base
        };

        let mut task_regions: Vec<Vec<(usize, usize, u64)>> = Vec::new();
        {
            let mut cache = self.neuron_cache.lock().unwrap();
            for (i, layout) in layouts.iter().enumerate() {
                let tensor_type = ExpertTensorType::from_name(&layout.tensor_name)
                    .unwrap_or(ExpertTensorType::Gate);
                let mut regions = Vec::new();
                for &eid in expert_ids {
                    if eid >= layout.num_experts {
                        continue;
                    }
                    if cache.is_loaded(layer_idx, eid, tensor_type) {
                        continue;
                    }
                    // Pool-relative: slot_offset + expert_id * stride
                    let mapped_eid = layout
                        .expert_permutation
                        .as_ref()
                        .and_then(|p| p.get(eid as usize).copied())
                        .unwrap_or(eid);
                    let pool_dest =
                        slot_offsets[i] + (mapped_eid as usize) * layout.expert_stride;
                    regions.push((
                        pool_dest,
                        layout.expert_stride,
                        layout.expert_file_offset(eid),
                    ));
                    cache.mark_loaded(layer_idx, eid, tensor_type);
                }
                if !regions.is_empty() {
                    task_regions.push(regions);
                }
            }
        }

        if task_regions.is_empty() {
            return;
        }

        // Submit to I/O pool and wait
        let completion = Arc::new(IoCompletion {
            remaining: AtomicUsize::new(task_regions.len()),
            layer_idx,
            update_status: true,
        });

        {
            let mut status = self.layer_status.lock().unwrap();
            status.insert(layer_idx, LayerStatus::Loading);
        }

        let pool_lock = self.io_pool.lock().unwrap();
        if let Some(ref pool) = *pool_lock {
            if let Some(ref tx) = pool.tx {
                for regions in task_regions {
                    let _ = tx.send(IoPoolTask {
                        regions,
                        base: pool_base,
                        completion: completion.clone(),
                    });
                }
            }
        }
        drop(pool_lock);

        // Wait for I/O completion
        let mut status = self.layer_status.lock().unwrap();
        while status.get(&layer_idx).copied() != Some(LayerStatus::Loaded) {
            status = self.layer_notify.wait(status).unwrap();
        }
    }

    /// Rewrite tensor->data for a layer's fused tensors to their current pool slots.
    /// Called on neuron cache hits when pool mode is active.
    fn rewrite_tensor_ptrs_for_layer(&self, layer_idx: u32) {
        let layouts = match self.expert_layouts.get(&layer_idx) {
            Some(l) => l,
            None => return,
        };

        let pool = self.expert_pool.lock().unwrap();
        let pool = match pool.as_ref() {
            Some(p) => p,
            None => return,
        };

        let slot_offsets: Vec<usize> = match pool.layer_slots.get(&layer_idx) {
            Some(slots) => slots.iter().map(|&s| s * pool.slot_size).collect(),
            None => return,
        };

        for (i, layout) in layouts.iter().enumerate() {
            if i >= slot_offsets.len() {
                break;
            }
            if let Some(&tensor_ptr) = self.fused_tensor_ptrs.get(&layout.tensor_name) {
                if !tensor_ptr.is_null() {
                    unsafe {
                        (*tensor_ptr).data =
                            pool.pool_base.add(slot_offsets[i]) as *mut c_void;
                    }
                }
            }
        }
    }

    /// Submit expert-only loads that update layer status on completion (for blocking waits).
    fn submit_expert_load_with_status(&self, layer_idx: u32, expert_ids: &[u32]) {
        if !self.expert_layouts.contains_key(&layer_idx) {
            return;
        }

        let base = *self.buffer_base.lock().unwrap();
        if base.is_null() {
            return;
        }

        let mut task_regions: Vec<Vec<(usize, usize, u64)>> = Vec::new();

        // Non-expert regions for this layer (if any are in the buffer)
        if let Some(regions) = self.non_expert_regions.get(&layer_idx) {
            if !regions.is_empty() {
                task_regions.push(regions.clone());
            }
        }

        // Expert strides
        {
            let mut cache = self.neuron_cache.lock().unwrap();
            if let Some(layouts) = self.expert_layouts.get(&layer_idx) {
                for layout in layouts {
                    let tensor_type = ExpertTensorType::from_name(&layout.tensor_name)
                        .unwrap_or(ExpertTensorType::Gate);
                    let mut regions = Vec::new();
                    for &eid in expert_ids {
                        if eid >= layout.num_experts {
                            continue;
                        }
                        if cache.is_loaded(layer_idx, eid, tensor_type) {
                            continue;
                        }
                        regions.push((
                            layout.expert_buffer_offset(eid),
                            layout.expert_stride,
                            layout.expert_file_offset(eid),
                        ));
                        // MADV_FREE evicted expert's pages on cache eviction
                        if let Some((ev_l, ev_e, ev_t)) =
                            cache.mark_loaded(layer_idx, eid, tensor_type)
                        {
                            if let Some(ev_layouts) = self.expert_layouts.get(&ev_l) {
                                for ev_layout in ev_layouts {
                                    if ExpertTensorType::from_name(&ev_layout.tensor_name)
                                        == Some(ev_t)
                                    {
                                        let offset = ev_layout.expert_buffer_offset(ev_e);
                                        let ptr = unsafe { base.add(offset) };
                                        compat::advise_free_pages(ptr, ev_layout.expert_stride);
                                    }
                                }
                            }
                        }
                    }
                    if !regions.is_empty() {
                        task_regions.push(regions);
                    }
                }
            }
        }

        if task_regions.is_empty() {
            let mut status = self.layer_status.lock().unwrap();
            status.insert(layer_idx, LayerStatus::Loaded);
            self.layer_notify.notify_all();
            return;
        }

        // update_status: true so completion triggers LayerStatus::Loaded + notify
        let completion = Arc::new(IoCompletion {
            remaining: AtomicUsize::new(task_regions.len()),
            layer_idx,
            update_status: true,
        });

        let pool = self.io_pool.lock().unwrap();
        if let Some(ref pool) = *pool {
            if let Some(ref tx) = pool.tx {
                for regions in task_regions {
                    let _ = tx.send(IoPoolTask {
                        regions,
                        base,
                        completion: completion.clone(),
                    });
                }
            }
        } else {
            drop(pool);
            let mut status = self.layer_status.lock().unwrap();
            status.insert(layer_idx, LayerStatus::Loaded);
            self.layer_notify.notify_all();
        }
    }

    /// Load all dense FFN tensors for a layer into pool slots, blocking until complete.
    pub fn ensure_dense_ffn_loaded(&self, layer_idx: u32) {
        let layouts = match self.dense_ffn_layouts.get(&layer_idx) {
            Some(l) if !l.is_empty() => l,
            _ => return,
        };

        let tracing_on = self.trace_enabled.load(Ordering::Relaxed);

        // Check if already loaded
        {
            let status = self.layer_status.lock().unwrap();
            if status.get(&layer_idx).copied() == Some(LayerStatus::Loaded) {
                // Already loaded — just rewrite tensor pointers
                self.rewrite_dense_ffn_ptrs(layer_idx);
                if tracing_on {
                    self.trace.record(TraceEvent::LayerHit(layer_idx));
                }
                return;
            }
        }

        let wait_start = Instant::now();

        // Allocate pool slots
        let slot_offsets = {
            let mut pool = self.expert_pool.lock().unwrap();
            let pool = pool.as_mut().expect("pool not initialized");
            pool.allocate_layer(layer_idx, layouts.len())
        };

        // Rewrite tensor->data pointers
        let pool_base = {
            let pool = self.expert_pool.lock().unwrap();
            pool.as_ref().unwrap().pool_base
        };

        for (i, layout) in layouts.iter().enumerate() {
            if let Some(&tensor_ptr) = self.fused_tensor_ptrs.get(&layout.tensor_name) {
                if !tensor_ptr.is_null() {
                    unsafe {
                        (*tensor_ptr).data =
                            pool_base.add(slot_offsets[i]) as *mut c_void;
                    }
                }
            }
        }

        // Build pread regions
        let mut task_regions: Vec<Vec<(usize, usize, u64)>> = Vec::new();
        for (i, layout) in layouts.iter().enumerate() {
            task_regions.push(vec![(slot_offsets[i], layout.size, layout.file_offset)]);
        }

        // Submit and wait
        {
            let mut status = self.layer_status.lock().unwrap();
            status.insert(layer_idx, LayerStatus::Loading);
        }

        let completion = Arc::new(IoCompletion {
            remaining: AtomicUsize::new(task_regions.len()),
            layer_idx,
            update_status: true,
        });

        {
            let pool_lock = self.io_pool.lock().unwrap();
            if let Some(ref pool) = *pool_lock {
                if let Some(ref tx) = pool.tx {
                    for regions in task_regions {
                        let _ = tx.send(IoPoolTask {
                            regions,
                            base: pool_base,
                            completion: completion.clone(),
                        });
                    }
                }
            }
        }

        let mut status = self.layer_status.lock().unwrap();
        while status.get(&layer_idx).copied() != Some(LayerStatus::Loaded) {
            status = self.layer_notify.wait(status).unwrap();
        }

        if tracing_on {
            let wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
            self.trace
                .record(TraceEvent::LayerStall { layer: layer_idx, wait_ms });
        }
    }

    /// Non-blocking prefetch of a layer's dense FFN tensors into pool slots.
    pub fn prefetch_dense_ffn(&self, layer_idx: u32) {
        let layouts = match self.dense_ffn_layouts.get(&layer_idx) {
            Some(l) if !l.is_empty() => l,
            _ => return,
        };

        let slot_offsets = {
            let mut pool = self.expert_pool.lock().unwrap();
            let pool = pool.as_mut().expect("pool not initialized");
            pool.allocate_layer(layer_idx, layouts.len())
        };

        let pool_base = {
            let pool = self.expert_pool.lock().unwrap();
            pool.as_ref().unwrap().pool_base
        };

        // Rewrite tensor pointers for when compute reaches this layer
        for (i, layout) in layouts.iter().enumerate() {
            if let Some(&tensor_ptr) = self.fused_tensor_ptrs.get(&layout.tensor_name) {
                if !tensor_ptr.is_null() {
                    unsafe {
                        (*tensor_ptr).data =
                            pool_base.add(slot_offsets[i]) as *mut c_void;
                    }
                }
            }
        }

        let mut task_regions: Vec<Vec<(usize, usize, u64)>> = Vec::new();
        for (i, layout) in layouts.iter().enumerate() {
            task_regions.push(vec![(slot_offsets[i], layout.size, layout.file_offset)]);
        }

        if task_regions.is_empty() {
            return;
        }

        {
            let mut status = self.layer_status.lock().unwrap();
            status.insert(layer_idx, LayerStatus::Loading);
        }

        let completion = Arc::new(IoCompletion {
            remaining: AtomicUsize::new(task_regions.len()),
            layer_idx,
            update_status: true,
        });

        let pool_lock = self.io_pool.lock().unwrap();
        if let Some(ref pool) = *pool_lock {
            if let Some(ref tx) = pool.tx {
                for regions in task_regions {
                    let _ = tx.send(IoPoolTask {
                        regions,
                        base: pool_base,
                        completion: completion.clone(),
                    });
                }
            }
        }
    }

    /// Allocate a resident buffer for the first `num_layers` FFN layers and load their
    /// data from the model file. These layers' tensors point into this permanent buffer
    /// instead of the pool, eliminating all I/O during inference.
    ///
    /// SAFETY: must be called after `activate_dense_ffn_pool` and `start_io_pool`.
    /// Mutates self through unsafe cast (same pattern as expert pool activation).
    pub fn activate_resident_ffn(
        &self,
        num_resident: u32,
    ) -> Option<(
        *mut u8,
        usize,
        std::collections::HashSet<u32>,
        HashMap<String, usize>,
    )> {
        if num_resident == 0 {
            return None;
        }

        // Compute total size and build offset map
        let mut total_size: usize = 0;
        let mut offsets: HashMap<String, usize> = HashMap::new();
        let mut resident_layers = std::collections::HashSet::new();

        for layer in 0..num_resident {
            if let Some(layouts) = self.dense_ffn_layouts.get(&layer) {
                for layout in layouts {
                    offsets.insert(layout.tensor_name.clone(), total_size);
                    total_size += layout.size;
                    // Align to page boundary
                    total_size = (total_size + 4095) & !4095;
                }
                resident_layers.insert(layer);
            }
        }

        if total_size == 0 {
            return None;
        }

        // Allocate resident buffer (anonymous pages, lazily committed)
        let base = compat::alloc_pages(total_size);
        if base.is_null() {
            tracing::warn!(
                "Failed to allocate resident FFN buffer ({:.1} GB)",
                total_size as f64 / 1e9,
            );
            return None;
        }

        tracing::info!(
            "Resident FFN buffer: {:.1} GB for {} layers (layers 0-{})",
            total_size as f64 / 1e9,
            num_resident,
            num_resident - 1,
        );

        // Load data via I/O pool (blocking)
        let mut all_regions: Vec<Vec<(usize, usize, u64)>> = Vec::new();
        for layer in 0..num_resident {
            if let Some(layouts) = self.dense_ffn_layouts.get(&layer) {
                for layout in layouts {
                    if let Some(&offset) = offsets.get(&layout.tensor_name) {
                        all_regions.push(vec![(offset, layout.size, layout.file_offset)]);
                    }
                }
            }
        }

        if !all_regions.is_empty() {
            let load_start = Instant::now();
            // Use a sentinel layer index for the completion tracker
            let sentinel_layer = self.num_layers + 1;
            {
                let mut status = self.layer_status.lock().unwrap();
                status.insert(sentinel_layer, LayerStatus::Loading);
            }

            let completion = Arc::new(IoCompletion {
                remaining: AtomicUsize::new(all_regions.len()),
                layer_idx: sentinel_layer,
                update_status: true,
            });

            {
                let pool_lock = self.io_pool.lock().unwrap();
                if let Some(ref pool) = *pool_lock {
                    if let Some(ref tx) = pool.tx {
                        for regions in all_regions {
                            let _ = tx.send(IoPoolTask {
                                regions,
                                base,
                                completion: completion.clone(),
                            });
                        }
                    }
                }
            }

            // Wait for all I/O to complete
            let mut status = self.layer_status.lock().unwrap();
            while status.get(&sentinel_layer).copied() != Some(LayerStatus::Loaded) {
                status = self.layer_notify.wait(status).unwrap();
            }
            status.remove(&sentinel_layer);

            let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
            tracing::info!(
                "Resident FFN loaded: {:.1} GB in {:.1}s ({:.1} GB/s)",
                total_size as f64 / 1e9,
                load_ms / 1000.0,
                total_size as f64 / 1e9 / (load_ms / 1000.0),
            );
        }

        // Rewrite tensor->data for resident layers
        for (name, &offset) in &offsets {
            if let Some(&tensor_ptr) = self.fused_tensor_ptrs.get(name) {
                if !tensor_ptr.is_null() {
                    unsafe {
                        (*tensor_ptr).data = base.add(offset) as *mut c_void;
                    }
                }
            }
        }

        Some((base, total_size, resident_layers, offsets))
    }

    /// Rewrite tensor->data for resident FFN layers to the permanent resident buffer.
    fn rewrite_resident_ffn_ptrs(&self, layer_idx: u32) {
        let layouts = match self.dense_ffn_layouts.get(&layer_idx) {
            Some(l) => l,
            None => return,
        };

        if self.resident_ffn_base.is_null() {
            return;
        }

        for layout in layouts {
            if let Some(&offset) = self.resident_ffn_offsets.get(&layout.tensor_name) {
                if let Some(&tensor_ptr) = self.fused_tensor_ptrs.get(&layout.tensor_name) {
                    if !tensor_ptr.is_null() {
                        unsafe {
                            (*tensor_ptr).data =
                                self.resident_ffn_base.add(offset) as *mut c_void;
                        }
                    }
                }
            }
        }
    }

    /// Rewrite tensor->data for dense FFN tensors to their current pool slots.
    fn rewrite_dense_ffn_ptrs(&self, layer_idx: u32) {
        let layouts = match self.dense_ffn_layouts.get(&layer_idx) {
            Some(l) => l,
            None => return,
        };

        let pool = self.expert_pool.lock().unwrap();
        let pool = match pool.as_ref() {
            Some(p) => p,
            None => return,
        };

        let slot_offsets: Vec<usize> = match pool.layer_slots.get(&layer_idx) {
            Some(slots) => slots.iter().map(|&s| s * pool.slot_size).collect(),
            None => return,
        };

        for (i, layout) in layouts.iter().enumerate() {
            if i >= slot_offsets.len() {
                break;
            }
            if let Some(&tensor_ptr) = self.fused_tensor_ptrs.get(&layout.tensor_name) {
                if !tensor_ptr.is_null() {
                    unsafe {
                        (*tensor_ptr).data =
                            pool.pool_base.add(slot_offsets[i]) as *mut c_void;
                    }
                }
            }
        }
    }

    /// Warm the neuron cache by pre-loading the most frequently activated experts
    /// from co-activation data. Reduces first-token stalls in expert-streaming mode.
    pub fn warm_cache_from_coactivation(&self) {
        let co_act = self.co_activation.lock().unwrap();
        if !co_act.has_data() {
            tracing::info!("No co-activation data available for cache warming");
            return;
        }

        let num_to_load = self.num_experts_used as usize;
        let mut layers_warmed = 0u32;
        let mut experts_loaded = 0u32;

        for (&layer_idx, _) in &self.expert_layouts {
            // Get top-K experts by self-activation frequency (diagonal of layer_counts)
            let layer_counts = co_act.layer_counts();
            let l = layer_idx as usize;
            if l >= layer_counts.len() {
                continue;
            }

            let mut freq: Vec<(u32, u32)> = (0..self.num_experts_total)
                .map(|e| {
                    let count = if (e as usize) < layer_counts[l].len() {
                        layer_counts[l][e as usize][e as usize]
                    } else {
                        0
                    };
                    (e, count)
                })
                .collect();
            freq.sort_by(|a, b| b.1.cmp(&a.1));

            let top_experts: Vec<u32> = freq
                .iter()
                .take(num_to_load)
                .filter(|(_, count)| *count > 0)
                .map(|(eid, _)| *eid)
                .collect();

            if top_experts.is_empty() {
                continue;
            }

            // Store as predicted experts so the eval_callback uses them
            self.selected_experts
                .lock()
                .unwrap()
                .insert(layer_idx, top_experts.clone());

            // Pre-load into pool
            self.ensure_experts_loaded(layer_idx, &top_experts);

            experts_loaded += top_experts.len() as u32;
            layers_warmed += 1;
        }

        if layers_warmed > 0 {
            tracing::info!(
                "Cache warmed: {experts_loaded} experts across {layers_warmed} layers from co-activation data"
            );
        }
    }

    /// Enable I/O tracing for diagnostic analysis.
    pub fn enable_trace(&self) {
        self.trace_enabled.store(true, Ordering::Relaxed);
    }

    /// Record a decode completion event (called from inference loop).
    pub fn record_decode(&self, decode_ms: f64) {
        if self.trace_enabled.load(Ordering::Relaxed) {
            self.trace.record(TraceEvent::DecodeComplete { decode_ms });
        }
    }

    /// Print a summary of the I/O trace after inference.
    pub fn print_trace_summary(&self) {
        let events = self.trace.events.lock().unwrap();
        if events.is_empty() {
            return;
        }

        let mut total_stall_ms = 0.0;
        let mut total_decode_ms = 0.0;
        let mut stall_count = 0u32;
        let mut hit_count = 0u32;
        let mut total_io_bytes = 0u64;
        let mut total_io_ms = 0.0;
        let mut decode_count = 0u32;
        let mut release_count = 0u32;

        for (_, event) in events.iter() {
            match event {
                TraceEvent::LayerHit(_) => hit_count += 1,
                TraceEvent::LayerStall { wait_ms, .. } => {
                    stall_count += 1;
                    total_stall_ms += wait_ms;
                }
                TraceEvent::LoadComplete { bytes, io_ms, .. } => {
                    total_io_bytes += bytes;
                    total_io_ms += io_ms;
                }
                TraceEvent::Released(_) => release_count += 1,
                TraceEvent::DecodeComplete { decode_ms } => {
                    decode_count += 1;
                    total_decode_ms += decode_ms;
                }
            }
        }

        let total_wall_ms = events.last().map_or(0.0, |(ms, _)| *ms);
        let accounted_ms = total_stall_ms + total_decode_ms;
        let dead_ms = (total_wall_ms - accounted_ms).max(0.0);
        let effective_bw = if total_io_ms > 0.0 {
            total_io_bytes as f64 / (total_io_ms / 1000.0) / 1e9
        } else {
            0.0
        };

        println!();
        println!("I/O Trace Summary ({decode_count} tokens):");
        println!("────────────────────────────────────────────────");
        println!(
            "  Layer requests: {} hit (prefetch ready), {} stalled (had to wait)",
            hit_count, stall_count
        );
        if stall_count > 0 {
            println!(
                "  Avg stall per layer:  {:.1} ms",
                total_stall_ms / stall_count as f64
            );
        }
        println!("  Total I/O stall:      {:.1} ms ({:.0}%)",
            total_stall_ms,
            if total_wall_ms > 0.0 { total_stall_ms / total_wall_ms * 100.0 } else { 0.0 }
        );
        println!("  Total decode (compute):{:.1} ms ({:.0}%)",
            total_decode_ms,
            if total_wall_ms > 0.0 { total_decode_ms / total_wall_ms * 100.0 } else { 0.0 }
        );
        println!("  Dead time (other):    {:.1} ms ({:.0}%)",
            dead_ms,
            if total_wall_ms > 0.0 { dead_ms / total_wall_ms * 100.0 } else { 0.0 }
        );
        println!("  Layers released:      {release_count}");
        println!(
            "  I/O pool throughput:  {:.2} GB/s ({:.1} GB in {:.1}ms worker time)",
            effective_bw,
            total_io_bytes as f64 / 1e9,
            total_io_ms
        );
        println!("  Wall time:            {:.1} ms", total_wall_ms);

        // Per-token breakdown for first 3 tokens
        let mut token_events: Vec<Vec<&(f64, TraceEvent)>> = Vec::new();
        let mut current_token: Vec<&(f64, TraceEvent)> = Vec::new();
        for entry in events.iter() {
            current_token.push(entry);
            if matches!(entry.1, TraceEvent::DecodeComplete { .. }) {
                token_events.push(std::mem::take(&mut current_token));
            }
        }

        let show_tokens = token_events.len().min(3);
        if show_tokens > 0 {
            println!();
            println!("  Per-token detail (first {show_tokens}):");
        }
        for (tok_i, tok) in token_events.iter().take(show_tokens).enumerate() {
            let tok_stalls: Vec<_> = tok
                .iter()
                .filter_map(|(_, e)| match e {
                    TraceEvent::LayerStall { layer, wait_ms } => Some((*layer, *wait_ms)),
                    _ => None,
                })
                .collect();
            let tok_hits = tok
                .iter()
                .filter(|(_, e)| matches!(e, TraceEvent::LayerHit(_)))
                .count();
            let tok_decode = tok.iter().find_map(|(_, e)| match e {
                TraceEvent::DecodeComplete { decode_ms } => Some(*decode_ms),
                _ => None,
            });
            let tok_stall_total: f64 = tok_stalls.iter().map(|(_, ms)| ms).sum();

            println!(
                "    Token {}: decode={:.0}ms, stalls={} ({:.0}ms total), hits={}",
                tok_i + 1,
                tok_decode.unwrap_or(0.0),
                tok_stalls.len(),
                tok_stall_total,
                tok_hits,
            );
            // Show individual stalls for first token
            if tok_i == 0 {
                for (layer, wait_ms) in &tok_stalls {
                    let layer_bytes: u64 = self
                        .layer_regions
                        .get(layer)
                        .map_or(0, |r| r.iter().map(|x| x.1 as u64).sum());
                    let bw = if *wait_ms > 0.0 {
                        layer_bytes as f64 / (*wait_ms / 1000.0) / 1e9
                    } else {
                        0.0
                    };
                    println!(
                        "      Layer {layer}: stall {wait_ms:.0}ms ({:.0} MB, {bw:.2} GB/s)",
                        layer_bytes as f64 / 1e6,
                    );
                }
            }
        }
        println!();
    }
}

impl Drop for PrefetchState {
    fn drop(&mut self) {
        // Save co-activation data on shutdown
        let co_activation = self.co_activation.lock().unwrap();
        if co_activation.has_data() {
            let path = CoActivationMatrix::persistence_path(&self.model_path);
            if let Err(e) = co_activation.save(&path) {
                tracing::warn!("Failed to save co-activation data: {e}");
            } else {
                tracing::info!("Co-activation data saved to {}", path.display());
            }
        }
        // Release resident FFN buffer
        if !self.resident_ffn_base.is_null() && self.resident_ffn_size > 0 {
            compat::free_pages(self.resident_ffn_base, self.resident_ffn_size);
        }
    }
}

/// Expert pool for expert-streaming mode: small mmap'd buffer where expert
/// tensor data is dynamically loaded into reusable slots.
pub struct ExpertPool {
    pub pool_base: *mut u8,
    pub pool_size: usize,
    pub slot_size: usize,
    pub num_slots: usize,
    /// Which slot each layer's tensors currently occupy: layer -> [slot indices]
    pub layer_slots: HashMap<u32, Vec<usize>>,
    /// Free slot indices (LIFO)
    pub free_slots: Vec<usize>,
}

unsafe impl Send for ExpertPool {}
unsafe impl Sync for ExpertPool {}

impl ExpertPool {
    pub fn new(pool_base: *mut u8, pool_size: usize, slot_size: usize) -> Self {
        let num_slots = pool_size / slot_size;
        let free_slots = (0..num_slots).rev().collect();
        Self {
            pool_base,
            pool_size,
            slot_size,
            num_slots,
            layer_slots: HashMap::new(),
            free_slots,
        }
    }

    /// Allocate slots for a layer. Returns pool byte offsets for each slot.
    pub fn allocate_layer(&mut self, layer_idx: u32, num_tensors: usize) -> Vec<usize> {
        if let Some(existing) = self.layer_slots.get(&layer_idx) {
            return existing.iter().map(|&s| s * self.slot_size).collect();
        }

        let mut slots = Vec::with_capacity(num_tensors);
        for _ in 0..num_tensors {
            if let Some(slot) = self.free_slots.pop() {
                slots.push(slot);
            } else {
                // Pool full — evict oldest layer
                let oldest = self
                    .layer_slots
                    .keys()
                    .filter(|&&l| l != layer_idx)
                    .copied()
                    .min()
                    .unwrap_or(0);
                self.release_layer(oldest);
                let slot = self.free_slots.pop().expect("pool exhausted after eviction");
                slots.push(slot);
            }
        }
        let offsets: Vec<usize> = slots.iter().map(|&s| s * self.slot_size).collect();
        self.layer_slots.insert(layer_idx, slots);
        offsets
    }

    /// Release a layer's slots back to the free list.
    pub fn release_layer(&mut self, layer_idx: u32) {
        if let Some(slots) = self.layer_slots.remove(&layer_idx) {
            for slot in slots {
                // Release the slot's physical pages back to the OS
                compat::advise_free_pages(
                    unsafe { self.pool_base.add(slot * self.slot_size) },
                    self.slot_size,
                );
                self.free_slots.push(slot);
            }
        }
    }

    /// Get pool byte offset for a specific slot index.
    pub fn slot_offset(&self, slot_idx: usize) -> usize {
        slot_idx * self.slot_size
    }
}

/// Layout of a dense FFN tensor for pool-based streaming.
#[derive(Debug, Clone)]
pub struct DenseFfnLayout {
    pub tensor_name: String,
    pub layer_index: u32,
    pub file_offset: u64,
    pub size: usize,
}

/// Controls the custom Hypura buffer type for NVMe-tier tensors.
pub struct HypuraBuftController {
    buft_ptr: hypura_sys::ggml_backend_buffer_type_t,
    tensor_map: Arc<Mutex<HashMap<String, TensorLocation>>>,
    model_path: PathBuf,
    gguf_data_offset: u64,
    /// Buffer base pointer captured from C callback during model loading
    buffer_base: Mutex<*mut u8>,
    /// Captured ggml_tensor* pointers for fused expert tensors (for data pointer rewriting)
    tensor_ptrs: Mutex<HashMap<String, *mut hypura_sys::ggml_tensor>>,
    /// Scratch buffer for dense FFN streaming: FFN tensor fread goes here instead of
    /// committing 22+ GB of anonymous mmap pages during model loading.
    loading_scratch: Mutex<*mut u8>,
    loading_scratch_size: AtomicUsize,
}

// SAFETY: buft_ptr used only from the creating thread; buffer_base accessed under Mutex
unsafe impl Send for HypuraBuftController {}
unsafe impl Sync for HypuraBuftController {}

impl HypuraBuftController {
    pub fn new(model_path: &Path, gguf: &GgufFile) -> Box<Self> {
        let tensor_map = Arc::new(Mutex::new(HashMap::new()));

        let mut controller = Box::new(Self {
            buft_ptr: std::ptr::null_mut(),
            tensor_map,
            model_path: model_path.to_path_buf(),
            gguf_data_offset: gguf.data_offset,
            buffer_base: Mutex::new(std::ptr::null_mut()),
            tensor_ptrs: Mutex::new(HashMap::new()),
            loading_scratch: Mutex::new(std::ptr::null_mut()),
            loading_scratch_size: AtomicUsize::new(0),
        });

        let rust_ctx = &*controller as *const Self as *mut c_void;
        let buft_ptr = unsafe {
            hypura_sys::hypura_buft_create(
                Some(on_tensor_loaded_cb),
                Some(on_tensor_init_cb),
                rust_ctx,
            )
        };
        controller.buft_ptr = buft_ptr;

        controller
    }

    pub fn buft_ptr(&self) -> hypura_sys::ggml_backend_buffer_type_t {
        self.buft_ptr
    }

    /// Enable scratch buffer for dense FFN streaming. Must be called BEFORE model loading.
    /// Allocates a scratch region sized to the largest FFN tensor. During init_tensor,
    /// FFN tensor->data is redirected here so fread doesn't commit ~22 GB of mmap pages.
    pub fn enable_dense_ffn_scratch(&self, gguf: &crate::model::gguf::GgufFile) {
        let max_ffn_size = gguf
            .tensors
            .iter()
            .filter(|t| {
                let role = TensorRole::from_name(&t.name);
                matches!(
                    role,
                    TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown
                )
            })
            .map(|t| t.size_bytes as usize)
            .max()
            .unwrap_or(0);

        if max_ffn_size == 0 {
            return;
        }

        let aligned = (max_ffn_size + 4095) & !4095;
        let ptr = compat::alloc_pages(aligned);
        if ptr.is_null() {
            tracing::warn!("Failed to allocate FFN loading scratch ({aligned} bytes)");
            return;
        }

        *self.loading_scratch.lock().unwrap() = ptr;
        self.loading_scratch_size
            .store(aligned, Ordering::Relaxed);
        tracing::info!(
            "Dense FFN scratch: {:.1} MB (redirects fread away from {:.1} GB loading buffer)",
            aligned as f64 / 1e6,
            gguf.tensors
                .iter()
                .filter(|t| {
                    let role = TensorRole::from_name(&t.name);
                    matches!(role, TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown)
                })
                .map(|t| t.size_bytes)
                .sum::<u64>() as f64
                / 1e9,
        );
    }

    /// Free the loading scratch buffer (called after pool activation).
    fn release_scratch(&self) {
        let mut scratch = self.loading_scratch.lock().unwrap();
        let size = self.loading_scratch_size.swap(0, Ordering::Relaxed);
        if !scratch.is_null() && size > 0 {
            compat::free_pages(*scratch, size);
            *scratch = std::ptr::null_mut();
        }
    }

    /// After model loading, correlate tensor map with GGUF file offsets
    /// and build the PrefetchState for use during inference.
    pub fn build_prefetch_state(
        &self,
        gguf: &GgufFile,
        num_layers: u32,
        nvme_layers: std::collections::HashSet<u32>,
    ) -> Arc<PrefetchState> {
        let mut map = self.tensor_map.lock().unwrap();

        // Fill in file offsets and layer indices from GGUF metadata
        for tensor_info in &gguf.tensors {
            if let Some(loc) = map.get_mut(&tensor_info.name) {
                loc.file_offset = self.gguf_data_offset + tensor_info.offset;
                loc.layer_index = tensor_info.layer_index;
            }
        }

        // Group by layer
        let mut layer_regions: HashMap<u32, Vec<(usize, usize, u64)>> = HashMap::new();
        for loc in map.values() {
            if let Some(layer) = loc.layer_index {
                layer_regions
                    .entry(layer)
                    .or_default()
                    .push((loc.offset_in_buffer, loc.size, loc.file_offset));
            }
        }

        // Sort regions within each layer by file offset for sequential I/O
        for regions in layer_regions.values_mut() {
            regions.sort_by_key(|&(_, _, file_offset)| file_offset);
        }

        // --- Build expert layouts and non-expert region maps ---
        let mut expert_layouts: HashMap<u32, Vec<ExpertLayout>> = HashMap::new();
        let mut non_expert_regions: HashMap<u32, Vec<(usize, usize, u64)>> = HashMap::new();
        let mut dense_ffn_layouts: HashMap<u32, Vec<DenseFfnLayout>> = HashMap::new();

        let num_experts_total = gguf.get_u32("expert_count").unwrap_or(0);
        let num_experts_used = gguf.get_u32("expert_used_count").unwrap_or(0);

        // Try to load expert permutations from sidecar file
        let perm_path = self.model_path.with_extension("permutations.json");
        let permutations: HashMap<u32, Vec<u32>> = if perm_path.exists() {
            match std::fs::read_to_string(&perm_path) {
                Ok(json) => {
                    let forward: HashMap<u32, Vec<u32>> =
                        serde_json::from_str(&json).unwrap_or_default();
                    if !forward.is_empty() {
                        tracing::info!(
                            "Loaded expert permutations from {}",
                            perm_path.display()
                        );
                    }
                    // Invert: forward[phys_pos] = logical_id → inverse[logical_id] = phys_pos
                    forward
                        .into_iter()
                        .map(|(layer, fwd)| {
                            let mut inv = vec![0u32; fwd.len()];
                            for (phys, &logical) in fwd.iter().enumerate() {
                                if (logical as usize) < inv.len() {
                                    inv[logical as usize] = phys as u32;
                                }
                            }
                            (layer, inv)
                        })
                        .collect()
                }
                Err(_) => HashMap::new(),
            }
        } else {
            HashMap::new()
        };

        for tensor_info in &gguf.tensors {
            let layer_idx = match tensor_info.layer_index {
                Some(l) => l,
                None => continue,
            };

            let loc = match map.get(&tensor_info.name) {
                Some(l) => l,
                None => continue,
            };

            let role = TensorRole::from_name(&tensor_info.name);

            match role {
                TensorRole::MoeFusedExperts if num_experts_total > 0 => {
                    let expert_stride = loc.size / num_experts_total as usize;
                    expert_layouts.entry(layer_idx).or_default().push(ExpertLayout {
                        tensor_name: tensor_info.name.clone(),
                        layer_index: layer_idx,
                        num_experts: num_experts_total,
                        expert_stride,
                        file_offset: loc.file_offset,
                        buffer_offset: loc.offset_in_buffer,
                        total_size: loc.size,
                        expert_permutation: permutations.get(&layer_idx).cloned(),
                    });
                }
                TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown
                    if num_experts_total == 0 =>
                {
                    // Dense FFN tensor — track for pool-based streaming
                    dense_ffn_layouts
                        .entry(layer_idx)
                        .or_default()
                        .push(DenseFfnLayout {
                            tensor_name: tensor_info.name.clone(),
                            layer_index: layer_idx,
                            file_offset: loc.file_offset,
                            size: loc.size,
                        });
                }
                _ => {
                    non_expert_regions
                        .entry(layer_idx)
                        .or_default()
                        .push((loc.offset_in_buffer, loc.size, loc.file_offset));
                }
            }
        }

        non_expert_regions.retain(|layer, _| expert_layouts.contains_key(layer));

        for regions in non_expert_regions.values_mut() {
            regions.sort_by_key(|&(_, _, file_offset)| file_offset);
        }

        if !expert_layouts.is_empty() {
            let total_expert_tensors: usize = expert_layouts.values().map(|v| v.len()).sum();
            tracing::info!(
                "MoE expert layouts: {} fused tensors across {} layers ({} experts, {} used/token)",
                total_expert_tensors,
                expert_layouts.len(),
                num_experts_total,
                num_experts_used,
            );
        }

        let layer_status: HashMap<u32, LayerStatus> = layer_regions
            .keys()
            .map(|&k| (k, LayerStatus::NotLoaded))
            .collect();

        let buffer_base = *self.buffer_base.lock().unwrap();

        let mut sorted_nvme_layers: Vec<u32> = nvme_layers.iter().copied().collect();
        sorted_nvme_layers.sort();

        let moe_nvme_layer_count = sorted_nvme_layers
            .iter()
            .filter(|l| expert_layouts.contains_key(l))
            .count();
        let expert_tensor_types = 3;
        let hot_experts = 3;
        let cache_capacity = moe_nvme_layer_count * expert_tensor_types * hot_experts;
        let cache_capacity = cache_capacity.max(16);

        // Load or create co-activation matrix
        let co_activation = if num_experts_total > 0 {
            let co_path = CoActivationMatrix::persistence_path(&self.model_path);
            match CoActivationMatrix::load(&co_path) {
                Ok(matrix) => {
                    tracing::info!("Loaded co-activation data from {}", co_path.display());
                    matrix
                }
                Err(_) => CoActivationMatrix::new(num_layers, num_experts_total),
            }
        } else {
            CoActivationMatrix::new(num_layers, num_experts_total.max(1))
        };

        Arc::new(PrefetchState {
            current_layer: AtomicI32::new(-1),
            tensor_map: map.clone(),
            model_path: self.model_path.clone(),
            buffer_base: Mutex::new(buffer_base),
            layer_regions,
            layer_status: Mutex::new(layer_status),
            layer_notify: Condvar::new(),
            num_layers,
            prefetch_enabled: AtomicBool::new(true),
            io_pool: Mutex::new(None),
            nvme_layers,
            keep_nvme_resident: AtomicBool::new(false),
            sorted_nvme_layers,
            expert_layouts,
            non_expert_regions,
            selected_experts: Mutex::new(HashMap::new()),
            num_experts_used,
            num_experts_total,
            neuron_cache: Mutex::new(NeuronCache::new(cache_capacity)),
            debug_logged_tensors: AtomicI32::new(0),
            co_activation: Mutex::new(co_activation),
            prev_layer_experts: Mutex::new(None),
            expert_streaming: AtomicBool::new(false),
            dense_ffn_streaming: AtomicBool::new(false),
            dense_ffn_layouts,
            dense_ffn_lookahead: 3, // default, updated after pool activation
            expert_pool: Mutex::new(None),
            fused_tensor_ptrs: HashMap::new(),
            resident_ffn_layers: std::collections::HashSet::new(),
            resident_ffn_base: std::ptr::null_mut(),
            resident_ffn_size: 0,
            resident_ffn_offsets: HashMap::new(),
            trace_enabled: AtomicBool::new(false),
            trace: IoTrace::new(),
        })
    }

    pub fn tensor_map(&self) -> Arc<Mutex<HashMap<String, TensorLocation>>> {
        self.tensor_map.clone()
    }

    /// Activate pool buffer for expert-streaming mode. Allocates a small pool,
    /// rewrites fused expert tensor->data pointers to pool slots, then releases
    /// the original large loading buffer. Returns the ExpertPool.
    pub fn activate_expert_pool(
        &self,
        gguf: &GgufFile,
        _num_experts: u32,
        num_slots: usize,
    ) -> anyhow::Result<ExpertPool> {
        let buffer = unsafe { hypura_sys::hypura_buft_get_last_buffer(self.buft_ptr) };
        anyhow::ensure!(!buffer.is_null(), "No buffer allocated");

        // Determine slot size from the largest fused expert tensor
        let slot_size = gguf
            .tensors
            .iter()
            .filter(|t| {
                let role = TensorRole::from_name(&t.name);
                matches!(role, TensorRole::MoeFusedExperts)
            })
            .map(|t| t.size_bytes as usize)
            .max()
            .unwrap_or(0);

        anyhow::ensure!(slot_size > 0, "No fused expert tensors found");

        // Caller provides slot count based on available memory.
        let pool_size = num_slots * slot_size;

        tracing::info!(
            "Expert pool: {} slots × {:.1} MB = {:.1} MB total",
            num_slots,
            slot_size as f64 / 1e6,
            pool_size as f64 / 1e6,
        );

        // Initialize pool in C
        let ret = unsafe { hypura_sys::hypura_buffer_init_pool(buffer, pool_size) };
        anyhow::ensure!(ret == 0, "Failed to allocate expert pool ({pool_size} bytes)");

        let pool_base =
            unsafe { hypura_sys::hypura_buffer_get_pool_base(buffer) } as *mut u8;
        anyhow::ensure!(!pool_base.is_null(), "Pool base is null");

        // Rewrite tensor->data for all fused expert tensors to point into the pool.
        // Initially all point to slot 0 (arbitrary valid address within pool).
        // The eval_callback will assign real slots before compute.
        let tensor_ptrs = self.tensor_ptrs.lock().unwrap();
        for (name, &ptr) in tensor_ptrs.iter() {
            if !ptr.is_null() {
                unsafe {
                    (*ptr).data = pool_base as *mut c_void;
                }
                tracing::trace!("Rewrote tensor->data for {name} to pool base");
            }
        }

        // Release the original large loading buffer
        unsafe { hypura_sys::hypura_buffer_release_loading_buffer(buffer) };
        tracing::info!("Released loading buffer, pool active");

        let pool = ExpertPool::new(pool_base, pool_size, slot_size);

        // Store tensor_ptrs in the pool for later data pointer updates
        // (handled via PrefetchState.tensor_ptrs instead)

        Ok(pool)
    }

    /// Activate pool buffer for dense FFN streaming. Same as expert pool but sized
    /// for individual FFN tensors (gate, up, down) rather than fused expert tensors.
    pub fn activate_dense_ffn_pool(
        &self,
        gguf: &GgufFile,
        num_slots: usize,
    ) -> anyhow::Result<ExpertPool> {
        let buffer = unsafe { hypura_sys::hypura_buft_get_last_buffer(self.buft_ptr) };
        anyhow::ensure!(!buffer.is_null(), "No buffer allocated");

        // Slot size = largest FFN tensor
        let slot_size = gguf
            .tensors
            .iter()
            .filter(|t| {
                let role = TensorRole::from_name(&t.name);
                matches!(
                    role,
                    TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown
                )
            })
            .map(|t| t.size_bytes as usize)
            .max()
            .unwrap_or(0);

        anyhow::ensure!(slot_size > 0, "No FFN tensors found");

        // Caller provides slot count based on available memory.
        let pool_size = num_slots * slot_size;

        tracing::info!(
            "Dense FFN pool: {} slots × {:.1} MB = {:.1} MB total",
            num_slots,
            slot_size as f64 / 1e6,
            pool_size as f64 / 1e6,
        );

        let ret = unsafe { hypura_sys::hypura_buffer_init_pool(buffer, pool_size) };
        anyhow::ensure!(ret == 0, "Failed to allocate FFN pool ({pool_size} bytes)");

        let pool_base =
            unsafe { hypura_sys::hypura_buffer_get_pool_base(buffer) } as *mut u8;
        anyhow::ensure!(!pool_base.is_null(), "Pool base is null");

        // Rewrite tensor->data for all FFN tensors to pool base (temporary valid address)
        let tensor_ptrs = self.tensor_ptrs.lock().unwrap();
        for (name, &ptr) in tensor_ptrs.iter() {
            let role = TensorRole::from_name(name);
            if matches!(
                role,
                TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown
            ) && !ptr.is_null()
            {
                unsafe {
                    (*ptr).data = pool_base as *mut c_void;
                }
            }
        }

        unsafe { hypura_sys::hypura_buffer_release_loading_buffer(buffer) };
        self.release_scratch();
        tracing::info!("Released loading buffer + scratch, FFN pool active");

        Ok(ExpertPool::new(pool_base, pool_size, slot_size))
    }

    /// Get captured tensor pointers (for transfer to PrefetchState).
    pub fn take_tensor_ptrs(&self) -> HashMap<String, *mut hypura_sys::ggml_tensor> {
        std::mem::take(&mut *self.tensor_ptrs.lock().unwrap())
    }
}

impl Drop for HypuraBuftController {
    fn drop(&mut self) {
        self.release_scratch();
        if !self.buft_ptr.is_null() {
            unsafe { hypura_sys::hypura_buft_free(self.buft_ptr) }
        }
    }
}

/// Build `tensor_buft_overrides` patterns from a PlacementPlan.
pub fn build_override_patterns(
    plan: &PlacementPlan,
    gguf: &GgufFile,
    buft_ptr: hypura_sys::ggml_backend_buffer_type_t,
    n_gpu_layers: i32,
) -> (Vec<CString>, Vec<hypura_sys::llama_model_tensor_buft_override>) {
    // Streaming modes: only override tensors explicitly assigned to NVMe.
    // Non-streamed tensors in the same layer stay on Metal.
    // - ExpertStreaming: fused expert tensors on NVMe, attention/norms on Metal
    // - DenseFfnStreaming: FFN (gate/up/down) on NVMe, attention/norms on Metal
    if matches!(
        plan.inference_mode,
        InferenceMode::ExpertStreaming | InferenceMode::DenseFfnStreaming
    ) {
        let mut patterns = Vec::new();
        for t in &gguf.tensors {
            if plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme) {
                let escaped = regex_escape(&t.name);
                patterns.push(format!("^{}$", escaped));
            }
        }

        let c_patterns: Vec<CString> = patterns
            .iter()
            .map(|p| CString::new(p.as_str()).unwrap())
            .collect();

        let mut overrides: Vec<hypura_sys::llama_model_tensor_buft_override> = c_patterns
            .iter()
            .map(|p| hypura_sys::llama_model_tensor_buft_override {
                pattern: p.as_ptr(),
                buft: buft_ptr,
            })
            .collect();

        overrides.push(hypura_sys::llama_model_tensor_buft_override {
            pattern: std::ptr::null(),
            buft: std::ptr::null_mut(),
        });

        return (c_patterns, overrides);
    }

    // Standard mode: override entire non-GPU layers
    let first_non_gpu_layer = if n_gpu_layers > 0 {
        (n_gpu_layers - 1) as u32
    } else {
        0
    };

    let mut layer_counts: HashMap<u32, (usize, usize)> = HashMap::new();

    for t in &gguf.tensors {
        if let Some(layer) = t.layer_index {
            if layer < first_non_gpu_layer {
                continue;
            }
            let entry = layer_counts.entry(layer).or_insert((0, 0));
            entry.1 += 1;
            entry.0 += 1;
        }
    }

    let mut patterns = Vec::new();

    for (layer, (non_gpu, total)) in &layer_counts {
        if *non_gpu == *total && *non_gpu > 0 {
            patterns.push(format!("^blk\\.{}\\.", layer));
        }
    }

    for t in &gguf.tensors {
        if let Some(layer) = t.layer_index {
            if layer < first_non_gpu_layer {
                continue;
            }
            let (non_gpu, total) = layer_counts[&layer];
            if non_gpu < total {
                let escaped = regex_escape(&t.name);
                patterns.push(format!("^{}$", escaped));
            }
        }
    }

    let c_patterns: Vec<CString> = patterns
        .iter()
        .map(|p| CString::new(p.as_str()).unwrap())
        .collect();

    let mut overrides: Vec<hypura_sys::llama_model_tensor_buft_override> = c_patterns
        .iter()
        .map(|p| hypura_sys::llama_model_tensor_buft_override {
            pattern: p.as_ptr(),
            buft: buft_ptr,
        })
        .collect();

    overrides.push(hypura_sys::llama_model_tensor_buft_override {
        pattern: std::ptr::null(),
        buft: std::ptr::null_mut(),
    });

    (c_patterns, overrides)
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
            | '\\' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// cb_eval callback — tracks layer transitions, intercepts router output,
/// and triggers expert-aware prefetch/release with co-activation predictions.
pub extern "C" fn eval_callback(
    tensor: *mut hypura_sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    if tensor.is_null() || user_data.is_null() {
        return true;
    }

    let state = unsafe { &*(user_data as *const PrefetchState) };

    if !state.prefetch_enabled.load(Ordering::Relaxed) {
        return true;
    }

    let name = unsafe { CStr::from_ptr((*tensor).name.as_ptr()) };
    let name_str = match name.to_str() {
        Ok(s) => s,
        Err(_) => return true,
    };

    // Router interception — detect ffn_moe_argsort post-compute
    if !ask && name_str.starts_with("ffn_moe_argsort-") {
        if let Some(layer_idx) = parse_layer_from_graph_name(name_str) {
            intercept_router_output(state, tensor, layer_idx);
        }
        return true;
    }

    let layer_idx = match parse_layer_from_graph_name(name_str) {
        Some(l) => l,
        None => match parse_layer_from_name(name_str) {
            Some(l) => l,
            None => return true,
        },
    };

    if !state.layer_regions.contains_key(&layer_idx) {
        return true;
    }

    // Expert-streaming mode: load only selected experts, keep non-expert tensors resident
    if state.expert_streaming.load(Ordering::Relaxed) {
        return eval_callback_expert_streaming(state, layer_idx, ask);
    }

    // Dense FFN-streaming mode: load FFN tensors on demand, attention is GPU-resident
    if state.dense_ffn_streaming.load(Ordering::Relaxed) {
        return eval_callback_dense_ffn_streaming(state, layer_idx, ask);
    }

    if ask {
        state.ensure_layer_loaded(layer_idx);
    } else {
        let prev = state.current_layer.swap(layer_idx as i32, Ordering::Relaxed);
        let prev_layer = prev as u32;

        if prev >= 0 && prev_layer != layer_idx && prev_layer < layer_idx {
            if state.keep_nvme_resident.load(Ordering::Relaxed) {
                // Keep-resident mode: no release, no prefetch
            } else {
                // Streaming mode: release old NVMe layers, prefetch ahead
                if state.nvme_layers.contains(&prev_layer) {
                    state.release_layer(prev_layer);
                }

                // Adaptive prefetch lookahead
                let lookahead_max = state.adaptive_lookahead();
                for lookahead in 2..=lookahead_max {
                    let target = layer_idx + lookahead;
                    if target < state.num_layers && state.nvme_layers.contains(&target) {
                        state.request_prefetch(PrefetchRequest::Layer(target));
                    }
                }

                // Speculative expert prefetch with co-activation predictions
                if state.expert_layouts.contains_key(&layer_idx) {
                    if let Some(experts) =
                        state.selected_experts.lock().unwrap().get(&layer_idx).cloned()
                    {
                        // Record in co-activation matrix
                        state.co_activation.lock().unwrap().record(layer_idx, &experts);

                        // Record cross-layer correlation
                        {
                            let mut prev_exp = state.prev_layer_experts.lock().unwrap();
                            if let Some((prev_l, ref prev_e)) = *prev_exp {
                                state
                                    .co_activation
                                    .lock()
                                    .unwrap()
                                    .record_cross_layer(prev_l, prev_e, &experts);
                            }
                            *prev_exp = Some((layer_idx, experts.clone()));
                        }

                        // Use co-activation to predict next layer's experts
                        let predicted = {
                            let co_act = state.co_activation.lock().unwrap();
                            if co_act.has_data() {
                                co_act.predict_next_layer(layer_idx, &experts, 3)
                            } else {
                                experts.clone() // Fallback: same experts
                            }
                        };

                        // Union predicted + observed, cap at 4
                        let mut prefetch_experts = experts;
                        for p in predicted {
                            if !prefetch_experts.contains(&p) {
                                prefetch_experts.push(p);
                            }
                        }
                        prefetch_experts.truncate(4);

                        for lookahead in 1..4 {
                            let target = layer_idx + lookahead;
                            if target < state.num_layers
                                && state.nvme_layers.contains(&target)
                                && state.expert_layouts.contains_key(&target)
                            {
                                state.request_prefetch(PrefetchRequest::ExpertSlices {
                                    layer_idx: target,
                                    expert_ids: prefetch_experts.clone(),
                                });
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    true
}

/// Dense FFN-streaming eval_callback path: load FFN tensors on demand per layer.
fn eval_callback_dense_ffn_streaming(state: &PrefetchState, layer_idx: u32, ask: bool) -> bool {
    let is_resident = state.resident_ffn_layers.contains(&layer_idx);

    if ask {
        // Before computation: load FFN tensors for this layer
        if is_resident {
            // Resident layer: rewrite tensor->data to permanent buffer, zero I/O
            state.rewrite_resident_ffn_ptrs(layer_idx);
        } else if state.dense_ffn_layouts.contains_key(&layer_idx) {
            // Streaming layer: load from NVMe into pool slot
            state.ensure_dense_ffn_loaded(layer_idx);
        }
    } else {
        // After computation: release old layer, prefetch next
        let prev = state
            .current_layer
            .swap(layer_idx as i32, Ordering::Relaxed);

        if prev >= 0 {
            let prev_layer = prev as u32;
            if prev_layer != layer_idx
                && prev_layer < layer_idx
                && !state.resident_ffn_layers.contains(&prev_layer)
                && state.dense_ffn_layouts.contains_key(&prev_layer)
            {
                // Release previous STREAMING layer's pool slots and status
                {
                    let mut pool = state.expert_pool.lock().unwrap();
                    if let Some(pool) = pool.as_mut() {
                        pool.release_layer(prev_layer);
                    }
                }
                let mut status = state.layer_status.lock().unwrap();
                status.insert(prev_layer, LayerStatus::NotLoaded);
            }
        }

        // Prefetch ahead: only submit for streaming (non-resident) layers.
        // Lookahead depth scales with pool size.
        {
            let status = state.layer_status.lock().unwrap();
            let mut to_prefetch = Vec::new();
            let max_lookahead = state.dense_ffn_lookahead;
            for lookahead in 1..=max_lookahead {
                let target = layer_idx + lookahead;
                if target >= state.num_layers {
                    break;
                }
                if state.resident_ffn_layers.contains(&target) {
                    continue;
                }
                if !state.dense_ffn_layouts.contains_key(&target) {
                    continue;
                }
                let s = status.get(&target).copied();
                if s != Some(LayerStatus::Loaded) && s != Some(LayerStatus::Loading) {
                    to_prefetch.push(target);
                }
            }
            drop(status);
            for target in to_prefetch {
                state.prefetch_dense_ffn(target);
            }
        }
    }
    true
}

/// Expert-streaming eval_callback path: load only selected experts on demand.
fn eval_callback_expert_streaming(state: &PrefetchState, layer_idx: u32, ask: bool) -> bool {
    if ask {
        // Before computation: load selected experts for this MoE layer
        if !state.expert_layouts.contains_key(&layer_idx) {
            // Non-MoE layer within the model — data is GPU-resident, nothing to do
            return true;
        }

        let experts = state
            .selected_experts
            .lock()
            .unwrap()
            .get(&layer_idx)
            .cloned();

        if let Some(expert_ids) = experts {
            state.ensure_experts_loaded(layer_idx, &expert_ids);
        } else {
            // No router output yet — use co-activation prediction or load all
            let predicted = {
                let co_act = state.co_activation.lock().unwrap();
                if co_act.has_data() {
                    co_act.predict_next_layer(layer_idx, &[], state.num_experts_used as usize)
                } else {
                    Vec::new()
                }
            };

            if predicted.is_empty() {
                // True cold start — load all experts for this layer
                state.ensure_layer_loaded(layer_idx);
            } else {
                state.ensure_experts_loaded(layer_idx, &predicted);
            }
        }
    } else {
        // After computation: track layer, record co-activation, prefetch next
        let prev = state
            .current_layer
            .swap(layer_idx as i32, Ordering::Relaxed);

        if state.expert_layouts.contains_key(&layer_idx) {
            if let Some(experts) = state
                .selected_experts
                .lock()
                .unwrap()
                .get(&layer_idx)
                .cloned()
            {
                state
                    .co_activation
                    .lock()
                    .unwrap()
                    .record(layer_idx, &experts);

                {
                    let mut prev_exp = state.prev_layer_experts.lock().unwrap();
                    if let Some((prev_l, ref prev_e)) = *prev_exp {
                        state
                            .co_activation
                            .lock()
                            .unwrap()
                            .record_cross_layer(prev_l, prev_e, &experts);
                    }
                    *prev_exp = Some((layer_idx, experts.clone()));
                }

                // Predict and prefetch next MoE layer's experts
                let predicted = {
                    let co_act = state.co_activation.lock().unwrap();
                    if co_act.has_data() {
                        co_act.predict_next_layer(layer_idx, &experts, 3)
                    } else {
                        experts.clone()
                    }
                };

                let mut prefetch_experts = experts;
                for p in predicted {
                    if !prefetch_experts.contains(&p) {
                        prefetch_experts.push(p);
                    }
                }
                prefetch_experts.truncate(4);

                for lookahead in 1..4 {
                    let target = layer_idx + lookahead;
                    if target < state.num_layers
                        && state.expert_layouts.contains_key(&target)
                    {
                        state.request_prefetch(PrefetchRequest::ExpertSlices {
                            layer_idx: target,
                            expert_ids: prefetch_experts.clone(),
                        });
                        break;
                    }
                }
            }
        }

        // Reset layer status to NotLoaded so next token can re-load experts.
        // (Expert data may have been evicted by neuron cache LRU.)
        if state.expert_layouts.contains_key(&layer_idx) && prev >= 0 {
            let prev_layer = prev as u32;
            if prev_layer != layer_idx && prev_layer < layer_idx {
                let mut status = state.layer_status.lock().unwrap();
                status.insert(prev_layer, LayerStatus::NotLoaded);
            }
        }
    }

    true
}

/// Read selected expert indices from the `ffn_moe_argsort` tensor.
fn intercept_router_output(
    state: &PrefetchState,
    tensor: *mut hypura_sys::ggml_tensor,
    layer_idx: u32,
) {
    let t = unsafe { &*tensor };

    if t.data.is_null() {
        return;
    }

    let n_experts = t.ne[0] as usize;
    let n_tokens = t.ne[1].max(1) as usize;
    if n_experts == 0 || n_experts > 64 {
        return;
    }

    let k = (state.num_experts_used as usize).min(n_experts);
    if k == 0 {
        return;
    }

    let data = t.data as *const i32;
    let mut expert_ids = Vec::with_capacity(k * n_tokens);

    for token in 0..n_tokens {
        let row_start = token * n_experts;
        for i in 0..k {
            let id = unsafe { *data.add(row_start + i) };
            if id >= 0 && (id as u32) < state.num_experts_total {
                expert_ids.push(id as u32);
            }
        }
    }

    if !expert_ids.is_empty() {
        expert_ids.sort_unstable();
        expert_ids.dedup();
        tracing::trace!(
            "Router intercepted: layer {} experts {:?} (from {} tokens)",
            layer_idx,
            expert_ids,
            n_tokens,
        );
        state
            .selected_experts
            .lock()
            .unwrap()
            .insert(layer_idx, expert_ids);
    }
}

fn parse_layer_from_name(name: &str) -> Option<u32> {
    if name.starts_with("blk.") {
        let rest = &name[4..];
        if let Some(dot_pos) = rest.find('.') {
            return rest[..dot_pos].parse().ok();
        }
    }
    None
}

fn parse_layer_from_graph_name(name: &str) -> Option<u32> {
    let dash_pos = name.rfind('-')?;
    let suffix = &name[dash_pos + 1..];
    suffix.parse().ok()
}

// C callbacks — signature must match typedef in hypura_buft.h

extern "C" fn on_tensor_loaded_cb(
    rust_ctx: *mut c_void,
    name: *const std::os::raw::c_char,
    offset: usize,
    size: usize,
    buffer_base: *mut c_void,
) {
    if rust_ctx.is_null() || name.is_null() {
        return;
    }
    let controller = unsafe { &*(rust_ctx as *const HypuraBuftController) };
    let name_str = unsafe { CStr::from_ptr(name) }
        .to_str()
        .unwrap_or("")
        .to_string();

    if !name_str.is_empty() {
        controller.tensor_map.lock().unwrap().insert(
            name_str,
            TensorLocation {
                offset_in_buffer: offset,
                size,
                file_offset: 0,
                layer_index: None,
            },
        );
    }

    if !buffer_base.is_null() {
        *controller.buffer_base.lock().unwrap() = buffer_base as *mut u8;
    }
}

extern "C" fn on_tensor_init_cb(
    rust_ctx: *mut c_void,
    tensor: *mut hypura_sys::ggml_tensor,
    name: *const std::os::raw::c_char,
) {
    if rust_ctx.is_null() || tensor.is_null() || name.is_null() {
        return;
    }
    let controller = unsafe { &*(rust_ctx as *const HypuraBuftController) };
    let name_str = unsafe { CStr::from_ptr(name) }
        .to_str()
        .unwrap_or("")
        .to_string();

    // Capture tensor pointers for pool data-pointer rewriting
    // (fused expert tensors for MoE, FFN tensors for dense models)
    let role = TensorRole::from_name(&name_str);
    if matches!(
        role,
        TensorRole::MoeFusedExperts
            | TensorRole::FfnGate
            | TensorRole::FfnUp
            | TensorRole::FfnDown
    ) {
        controller
            .tensor_ptrs
            .lock()
            .unwrap()
            .insert(name_str, tensor);

        // Dense FFN streaming: redirect FFN tensor->data to scratch buffer so
        // llama.cpp's fread doesn't commit ~22 GB of anonymous mmap pages.
        // on_tensor_loaded already captured the correct buffer offset above.
        if matches!(
            role,
            TensorRole::FfnGate | TensorRole::FfnUp | TensorRole::FfnDown
        ) {
            let scratch = controller.loading_scratch.lock().unwrap();
            if !scratch.is_null() {
                unsafe {
                    (*tensor).data = *scratch as *mut c_void;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_escape() {
        assert_eq!(
            regex_escape("blk.0.attn_q.weight"),
            "blk\\.0\\.attn_q\\.weight"
        );
        assert_eq!(regex_escape("simple"), "simple");
    }

    #[test]
    fn test_parse_layer() {
        assert_eq!(parse_layer_from_name("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_layer_from_name("blk.15.ffn_gate.weight"), Some(15));
        assert_eq!(parse_layer_from_name("token_embd.weight"), None);
        assert_eq!(parse_layer_from_name("output.weight"), None);
    }

    #[test]
    fn test_parse_layer_moe_topk() {
        assert_eq!(parse_layer_from_name("blk.5.ffn_moe_topk"), Some(5));
        assert_eq!(parse_layer_from_name("blk.31.ffn_moe_topk"), Some(31));
    }

    #[test]
    fn test_layer_status_transitions() {
        let status = Mutex::new(HashMap::new());
        let notify = Condvar::new();

        assert_eq!(status.lock().unwrap().get(&0), None);

        status.lock().unwrap().insert(0, LayerStatus::Loading);
        assert_eq!(
            status.lock().unwrap().get(&0).copied(),
            Some(LayerStatus::Loading)
        );

        status.lock().unwrap().insert(0, LayerStatus::Loaded);
        notify.notify_all();
        assert_eq!(
            status.lock().unwrap().get(&0).copied(),
            Some(LayerStatus::Loaded)
        );

        status.lock().unwrap().insert(0, LayerStatus::NotLoaded);
        assert_eq!(
            status.lock().unwrap().get(&0).copied(),
            Some(LayerStatus::NotLoaded)
        );
    }
}
