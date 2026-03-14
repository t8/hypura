use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::model::gguf::GgufFile;
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
    /// File descriptor for F_NOCACHE reads (opened once, reused)
    pub nvme_fd: Mutex<Option<i32>>,
    /// Channel sender for background prefetch thread
    pub prefetch_tx: Mutex<Option<std::sync::mpsc::Sender<u32>>>,
}

// SAFETY: buffer_base and nvme_fd are accessed under Mutex
unsafe impl Send for PrefetchState {}
unsafe impl Sync for PrefetchState {}

impl PrefetchState {
    /// Open the model file with F_NOCACHE for direct NVMe reads.
    pub fn open_nvme_fd(&self) -> anyhow::Result<()> {
        let file = std::fs::File::open(&self.model_path)?;
        let fd = file.as_raw_fd();
        unsafe {
            libc::fcntl(fd, libc::F_NOCACHE, 1);
        }
        // Prevent the file from being closed when `file` drops
        std::mem::forget(file);
        *self.nvme_fd.lock().unwrap() = Some(fd);
        Ok(())
    }

    /// Perform the raw pread I/O for a layer's tensor regions.
    fn load_layer_data(&self, layer_idx: u32) {
        let regions = match self.layer_regions.get(&layer_idx) {
            Some(r) => r,
            None => return,
        };

        let base = *self.buffer_base.lock().unwrap();
        if base.is_null() {
            return;
        }

        let fd = match *self.nvme_fd.lock().unwrap() {
            Some(fd) => fd,
            None => return,
        };

        for &(offset, size, file_offset) in regions {
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
    }

    /// Ensure a layer's tensor data is loaded in physical memory.
    /// Blocks until loaded — waits on background thread or does synchronous pread fallback.
    pub fn ensure_layer_loaded(&self, layer_idx: u32) {
        let mut status = self.layer_status.lock().unwrap();
        loop {
            match status.get(&layer_idx).copied() {
                Some(LayerStatus::Loaded) => return,
                Some(LayerStatus::Loading) => {
                    // Background thread is loading — wait for completion
                    status = self.layer_notify.wait(status).unwrap();
                }
                _ => {
                    // Not loaded — synchronous fallback
                    status.insert(layer_idx, LayerStatus::Loading);
                    drop(status);

                    self.load_layer_data(layer_idx);

                    let mut status = self.layer_status.lock().unwrap();
                    status.insert(layer_idx, LayerStatus::Loaded);
                    self.layer_notify.notify_all();
                    return;
                }
            }
        }
    }

    /// Release physical pages for a layer's tensors.
    /// Virtual address space is preserved; data must be reloaded before next use.
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

        self.layer_status
            .lock()
            .unwrap()
            .insert(layer_idx, LayerStatus::NotLoaded);
    }

    /// Send a non-blocking prefetch request to the background thread.
    pub fn request_prefetch(&self, layer_idx: u32) {
        if let Some(tx) = self.prefetch_tx.lock().unwrap().as_ref() {
            let _ = tx.send(layer_idx);
        }
    }

    /// Spawn the background prefetch thread. Returns JoinHandle.
    pub fn start_prefetch_thread(self: &Arc<Self>) -> Option<std::thread::JoinHandle<()>> {
        let (tx, rx) = std::sync::mpsc::channel::<u32>();
        *self.prefetch_tx.lock().unwrap() = Some(tx);

        let state = self.clone();
        Some(
            std::thread::Builder::new()
                .name("hypura-prefetch".into())
                .spawn(move || prefetch_worker(&state, rx))
                .expect("failed to spawn prefetch thread"),
        )
    }

    /// Stop the background prefetch thread by dropping the sender.
    pub fn stop_prefetch_thread(&self) {
        self.prefetch_tx.lock().unwrap().take();
    }
}

/// Background prefetch worker: receives layer indices, preloads data from NVMe.
fn prefetch_worker(state: &PrefetchState, rx: std::sync::mpsc::Receiver<u32>) {
    while let Ok(layer_idx) = rx.recv() {
        let mut status = state.layer_status.lock().unwrap();
        match status.get(&layer_idx).copied() {
            Some(LayerStatus::Loaded) | Some(LayerStatus::Loading) => continue,
            _ => {}
        }
        status.insert(layer_idx, LayerStatus::Loading);
        drop(status);

        state.load_layer_data(layer_idx);

        let mut status = state.layer_status.lock().unwrap();
        status.insert(layer_idx, LayerStatus::Loaded);
        state.layer_notify.notify_all();
    }
}

impl Drop for PrefetchState {
    fn drop(&mut self) {
        if let Some(fd) = self.nvme_fd.lock().unwrap().take() {
            unsafe {
                libc::close(fd);
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
    pub fn build_prefetch_state(&self, gguf: &GgufFile, num_layers: u32) -> Arc<PrefetchState> {
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

        // All layers start as Loaded (data was written during model init)
        let layer_status: HashMap<u32, LayerStatus> = layer_regions
            .keys()
            .map(|&k| (k, LayerStatus::Loaded))
            .collect();

        // Copy buffer_base from controller (set during model loading callbacks)
        let buffer_base = *self.buffer_base.lock().unwrap();

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
            nvme_fd: Mutex::new(None),
            prefetch_tx: Mutex::new(None),
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
    plan: &PlacementPlan,
    gguf: &GgufFile,
    buft_ptr: hypura_sys::ggml_backend_buffer_type_t,
) -> (Vec<CString>, Vec<hypura_sys::llama_model_tensor_buft_override>) {
    let mut layer_counts: HashMap<u32, (usize, usize)> = HashMap::new();

    for t in &gguf.tensors {
        if let Some(layer) = t.layer_index {
            let entry = layer_counts.entry(layer).or_insert((0, 0));
            entry.1 += 1;
            if plan.tier_assignments.get(&t.name) == Some(&StorageTier::Nvme) {
                entry.0 += 1;
            }
        }
    }

    let mut patterns = Vec::new();

    for (layer, (nvme, total)) in &layer_counts {
        if *nvme == *total && *nvme > 0 {
            patterns.push(format!("^blk\\.{}\\.", layer));
        }
    }

    for t in &gguf.tensors {
        let tier = plan.tier_assignments.get(&t.name);
        if tier != Some(&StorageTier::Nvme) {
            continue;
        }
        if let Some(layer) = t.layer_index {
            let (nvme, total) = layer_counts[&layer];
            if nvme < total {
                let escaped = regex_escape(&t.name);
                patterns.push(format!("^{}$", escaped));
            }
        } else {
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

/// cb_eval callback — tracks layer transitions and triggers prefetch/release.
///
/// When `ask=true` (before computing a tensor): ensure that layer's NVMe data is loaded.
/// When `ask=false` (after computing): if we've moved to a new layer, release old layer's
/// pages and request async prefetch for upcoming layers.
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

    let layer_idx = match parse_layer_from_name(name_str) {
        Some(l) => l,
        None => return true,
    };

    if ask {
        // Before computing: ensure this layer's tensors are in physical memory
        state.ensure_layer_loaded(layer_idx);
    } else {
        // After computing: track layer transition
        let prev = state.current_layer.swap(layer_idx as i32, Ordering::Relaxed);
        let prev_layer = prev as u32;

        if prev >= 0 && prev_layer != layer_idx && prev_layer < layer_idx {
            // We've moved forward — release the completed layer's pages
            state.release_layer(prev_layer);

            // Async prefetch: request layer+2 and layer+3 (non-blocking)
            let target2 = layer_idx + 2;
            if target2 < state.num_layers {
                state.request_prefetch(target2);
            }
            let target3 = layer_idx + 3;
            if target3 < state.num_layers {
                state.request_prefetch(target3);
            }
        }
    }

    true
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

    // Capture buffer_base (same for all tensors in a buffer, set once is sufficient)
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
    fn test_layer_status_transitions() {
        let status = Mutex::new(HashMap::new());
        let notify = Condvar::new();

        // Initially empty (not tracked)
        assert_eq!(status.lock().unwrap().get(&0), None);

        // Mark as loading
        status.lock().unwrap().insert(0, LayerStatus::Loading);
        assert_eq!(
            status.lock().unwrap().get(&0).copied(),
            Some(LayerStatus::Loading)
        );

        // Mark as loaded
        status.lock().unwrap().insert(0, LayerStatus::Loaded);
        notify.notify_all();
        assert_eq!(
            status.lock().unwrap().get(&0).copied(),
            Some(LayerStatus::Loaded)
        );

        // Release
        status.lock().unwrap().insert(0, LayerStatus::NotLoaded);
        assert_eq!(
            status.lock().unwrap().get(&0).copied(),
            Some(LayerStatus::NotLoaded)
        );
    }
}
