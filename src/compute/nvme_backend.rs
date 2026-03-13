use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

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

/// Controls the custom Hypura buffer type for NVMe-tier tensors.
/// Manages the lifecycle of the C-side buffer type and tracks tensor metadata.
pub struct HypuraBuftController {
    buft_ptr: hypura_sys::ggml_backend_buffer_type_t,
    tensor_map: Arc<Mutex<HashMap<String, TensorLocation>>>,
    current_layer: Arc<AtomicI32>,
    model_path: PathBuf,
    gguf_data_offset: u64,
}

impl HypuraBuftController {
    /// Create a new controller with a custom buffer type.
    /// SAFETY: The returned controller must not be moved after creation,
    /// because the C-side buffer type holds a pointer to it.
    /// Use `Box::pin` or store in a stable location.
    pub fn new(model_path: &Path, gguf: &GgufFile) -> Box<Self> {
        let tensor_map = Arc::new(Mutex::new(HashMap::new()));
        let current_layer = Arc::new(AtomicI32::new(-1));

        let mut controller = Box::new(Self {
            buft_ptr: std::ptr::null_mut(),
            tensor_map,
            current_layer,
            model_path: model_path.to_path_buf(),
            gguf_data_offset: gguf.data_offset,
        });

        // Create the C-side buffer type with callbacks pointing to this controller
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

    /// Get the raw buffer type pointer for passing to llama.cpp.
    pub fn buft_ptr(&self) -> hypura_sys::ggml_backend_buffer_type_t {
        self.buft_ptr
    }

    /// After model loading, correlate tensor map with GGUF file offsets.
    pub fn finalize_tensor_map(&self, gguf: &GgufFile) {
        let mut map = self.tensor_map.lock().unwrap();
        for tensor_info in &gguf.tensors {
            if let Some(loc) = map.get_mut(&tensor_info.name) {
                loc.file_offset = self.gguf_data_offset + tensor_info.offset;
                loc.layer_index = tensor_info.layer_index;
            }
        }
    }

    /// Get the tensor location map (for prefetch scheduling).
    pub fn tensor_map(&self) -> Arc<Mutex<HashMap<String, TensorLocation>>> {
        self.tensor_map.clone()
    }

    /// Get current layer tracker (for cb_eval integration).
    pub fn current_layer(&self) -> Arc<AtomicI32> {
        self.current_layer.clone()
    }

    /// Get the model file path (for NVMe reads).
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Get tensors belonging to a specific layer.
    pub fn layer_tensors(&self, layer_idx: u32) -> Vec<(String, TensorLocation)> {
        let map = self.tensor_map.lock().unwrap();
        map.iter()
            .filter(|(_, loc)| loc.layer_index == Some(layer_idx))
            .map(|(name, loc)| (name.clone(), loc.clone()))
            .collect()
    }

    /// Release physical pages for a layer's tensors via madvise(MADV_FREE).
    /// The virtual address space is preserved; pages will be zero-filled on next access.
    pub unsafe fn release_layer_pages(&self, layer_idx: u32, buffer_base: *mut u8) {
        let map = self.tensor_map.lock().unwrap();
        for (_, loc) in map.iter().filter(|(_, l)| l.layer_index == Some(layer_idx)) {
            let ptr = buffer_base.add(loc.offset_in_buffer);
            // MADV_FREE: pages can be reclaimed by the kernel when under memory pressure
            libc::madvise(
                ptr as *mut c_void,
                loc.size,
                libc::MADV_FREE,
            );
        }
    }

    /// Reload a layer's tensors from NVMe using pread with F_NOCACHE.
    pub fn reload_layer_from_nvme(
        &self,
        layer_idx: u32,
        buffer_base: *mut u8,
    ) -> anyhow::Result<u64> {
        let map = self.tensor_map.lock().unwrap();
        let layer_tensors: Vec<_> = map
            .iter()
            .filter(|(_, l)| l.layer_index == Some(layer_idx))
            .collect();

        if layer_tensors.is_empty() {
            return Ok(0);
        }

        let file = std::fs::File::open(&self.model_path)?;
        let fd = std::os::unix::io::AsRawFd::as_raw_fd(&file);
        unsafe {
            libc::fcntl(fd, libc::F_NOCACHE, 1);
        }

        let mut total_bytes = 0u64;
        for (_, loc) in &layer_tensors {
            let dst = unsafe { buffer_base.add(loc.offset_in_buffer) };
            let mut read = 0usize;
            while read < loc.size {
                let n = unsafe {
                    libc::pread(
                        fd,
                        dst.add(read) as *mut c_void,
                        loc.size - read,
                        (loc.file_offset + read as u64) as libc::off_t,
                    )
                };
                if n <= 0 {
                    break;
                }
                read += n as usize;
            }
            total_bytes += read as u64;
        }

        Ok(total_bytes)
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
/// Returns (CString patterns, override structs). The CStrings must outlive the overrides.
pub fn build_override_patterns(
    plan: &PlacementPlan,
    gguf: &GgufFile,
    buft_ptr: hypura_sys::ggml_backend_buffer_type_t,
) -> (Vec<CString>, Vec<hypura_sys::llama_model_tensor_buft_override>) {
    // Group NVMe tensors by layer
    let mut layer_counts: HashMap<u32, (usize, usize)> = HashMap::new(); // (nvme_count, total_count)

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

    // Full-layer patterns where all tensors in the layer are NVMe
    for (layer, (nvme, total)) in &layer_counts {
        if *nvme == *total && *nvme > 0 {
            patterns.push(format!("^blk\\.{}\\.", layer));
        }
    }

    // Per-tensor patterns for layers where only some tensors are NVMe
    for t in &gguf.tensors {
        let tier = plan.tier_assignments.get(&t.name);
        if tier != Some(&StorageTier::Nvme) {
            continue;
        }
        if let Some(layer) = t.layer_index {
            let (nvme, total) = layer_counts[&layer];
            if nvme < total {
                // Not a full layer — need per-tensor pattern
                let escaped = regex_escape(&t.name);
                patterns.push(format!("^{}$", escaped));
            }
        } else {
            // Non-layer tensor (embedding, output head)
            let escaped = regex_escape(&t.name);
            patterns.push(format!("^{}$", escaped));
        }
    }

    // Convert to CStrings and build override array
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

    // NULL terminator
    overrides.push(hypura_sys::llama_model_tensor_buft_override {
        pattern: std::ptr::null(),
        buft: std::ptr::null_mut(),
    });

    (c_patterns, overrides)
}

/// Escape special regex characters in a tensor name.
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

/// cb_eval callback — tracks layer transitions during inference.
pub extern "C" fn eval_callback(
    tensor: *mut hypura_sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    if tensor.is_null() || user_data.is_null() {
        return true;
    }

    let current_layer = unsafe { &*(user_data as *const AtomicI32) };
    let name = unsafe { CStr::from_ptr((*tensor).name.as_ptr()) };

    if let Ok(name_str) = name.to_str() {
        if let Some(layer_idx) = parse_layer_from_name(name_str) {
            let prev = current_layer.load(Ordering::Relaxed);
            if prev != layer_idx as i32 {
                current_layer.store(layer_idx as i32, Ordering::Relaxed);
                if !ask {
                    tracing::trace!("Layer transition: {} -> {}", prev, layer_idx);
                }
            }
        }
    }

    true // always continue
}

fn parse_layer_from_name(name: &str) -> Option<u32> {
    // Match "blk.N." pattern
    if name.starts_with("blk.") {
        let rest = &name[4..];
        if let Some(dot_pos) = rest.find('.') {
            return rest[..dot_pos].parse().ok();
        }
    }
    None
}

// C callbacks
extern "C" fn on_tensor_loaded_cb(
    rust_ctx: *mut c_void,
    name: *const std::os::raw::c_char,
    offset: usize,
    size: usize,
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
        controller
            .tensor_map
            .lock()
            .unwrap()
            .insert(name_str, TensorLocation {
                offset_in_buffer: offset,
                size,
                file_offset: 0, // filled in by finalize_tensor_map
                layer_index: None,
            });
    }
}

extern "C" fn on_tensor_init_cb(
    _rust_ctx: *mut c_void,
    _tensor: *mut hypura_sys::ggml_tensor,
    _name: *const std::os::raw::c_char,
) {
    // Currently unused — tensor pointer registry for future use
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_escape() {
        assert_eq!(regex_escape("blk.0.attn_q.weight"), "blk\\.0\\.attn_q\\.weight");
        assert_eq!(regex_escape("simple"), "simple");
    }

    #[test]
    fn test_parse_layer() {
        assert_eq!(parse_layer_from_name("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_layer_from_name("blk.15.ffn_gate.weight"), Some(15));
        assert_eq!(parse_layer_from_name("token_embd.weight"), None);
        assert_eq!(parse_layer_from_name("output.weight"), None);
    }
}
