use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::os::unix::io::AsRawFd;
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
    /// Per-worker file descriptors (for cleanup).
    worker_fds: Vec<i32>,
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
            let file = std::fs::File::open(model_path)?;
            let fd = file.as_raw_fd();
            unsafe {
                libc::fcntl(fd, libc::F_NOCACHE, 1);
            }
            std::mem::forget(file);
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

        // Close per-worker fds
        for fd in &self.worker_fds {
            unsafe {
                libc::close(*fd);
            }
        }
    }
}

/// I/O worker thread: pulls tasks from shared channel, executes pread.
fn io_worker(
    fd: i32,
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

/// Perform pread I/O for a single region. Standalone function used by IoPool workers.
fn pread_region(fd: i32, base: *mut u8, offset: usize, size: usize, file_offset: u64) {
    let dst = unsafe { base.add(offset) };
    let mut read = 0usize;
    while read < size {
        let n = unsafe {
            libc::pread(
                fd,
                dst.add(read) as *mut c_void,
                size - read,
                (file_offset + read as u64) as libc::off_t,
            )
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
            unsafe {
                libc::madvise(ptr as *mut c_void, size, libc::MADV_FREE);
            }
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
    }
}

/// Controls the custom Hypura buffer type for NVMe-tier tensors.
pub struct HypuraBuftController {
    buft_ptr: hypura_sys::ggml_backend_buffer_type_t,
    tensor_map: Arc<Mutex<HashMap<String, TensorLocation>>>,
    model_path: PathBuf,
    gguf_data_offset: u64,
    /// Buffer base pointer captured from C callback during model loading
    buffer_base: Mutex<*mut u8>,
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
            trace_enabled: AtomicBool::new(false),
            trace: IoTrace::new(),
        })
    }

    pub fn tensor_map(&self) -> Arc<Mutex<HashMap<String, TensorLocation>>> {
        self.tensor_map.clone()
    }
}

impl Drop for HypuraBuftController {
    fn drop(&mut self) {
        if !self.buft_ptr.is_null() {
            unsafe { hypura_sys::hypura_buft_free(self.buft_ptr) }
        }
    }
}

/// Build `tensor_buft_overrides` patterns from a PlacementPlan.
pub fn build_override_patterns(
    _plan: &PlacementPlan,
    gguf: &GgufFile,
    buft_ptr: hypura_sys::ggml_backend_buffer_type_t,
    n_gpu_layers: i32,
) -> (Vec<CString>, Vec<hypura_sys::llama_model_tensor_buft_override>) {
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
    _rust_ctx: *mut c_void,
    _tensor: *mut hypura_sys::ggml_tensor,
    _name: *const std::os::raw::c_char,
) {
    // Reserved for future tensor pointer registry
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
