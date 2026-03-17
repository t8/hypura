use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use serde::{Deserialize, Serialize};

/// Sampling parameters for token generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u32,
    pub max_tokens: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            seed: 42,
            max_tokens: 512,
        }
    }
}

/// Performance data from llama.cpp context.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerfData {
    pub t_load_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_p_eval: i32,
    pub n_eval: i32,
}

/// RAII guard for llama.cpp backend initialization.
/// Must be created before any other llama.cpp calls.
pub struct LlamaBackend(());

impl LlamaBackend {
    pub fn init() -> Self {
        unsafe { hypura_sys::llama_backend_init() }
        Self(())
    }
}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        unsafe { hypura_sys::llama_backend_free() }
    }
}

/// Safe wrapper around `llama_model *`.
pub struct LlamaModel {
    ptr: *mut hypura_sys::llama_model,
    vocab: *const hypura_sys::llama_vocab,
}

impl LlamaModel {
    /// Load a model from a GGUF file.
    pub fn load(path: &Path, n_gpu_layers: i32, use_mmap: bool) -> anyhow::Result<Self> {
        let c_path = CString::new(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid path encoding"))?,
        )?;

        let mut params = unsafe { hypura_sys::llama_model_default_params() };
        params.n_gpu_layers = n_gpu_layers;
        params.use_mmap = use_mmap;

        let ptr = unsafe { hypura_sys::llama_model_load_from_file(c_path.as_ptr(), params) };
        anyhow::ensure!(!ptr.is_null(), "Failed to load model from {}", path.display());

        let vocab = unsafe { hypura_sys::llama_model_get_vocab(ptr) };
        anyhow::ensure!(!vocab.is_null(), "Failed to get vocab from model");

        Ok(Self { ptr, vocab })
    }

    /// Load a model with custom tensor buffer type overrides.
    /// `overrides` is a NULL-terminated array of pattern/buft pairs.
    pub fn load_with_overrides(
        path: &Path,
        n_gpu_layers: i32,
        use_mmap: bool,
        overrides: *const hypura_sys::llama_model_tensor_buft_override,
    ) -> anyhow::Result<Self> {
        let c_path = CString::new(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid path encoding"))?,
        )?;

        let mut params = unsafe { hypura_sys::llama_model_default_params() };
        params.n_gpu_layers = n_gpu_layers;
        params.use_mmap = use_mmap;
        params.tensor_buft_overrides = overrides;

        let ptr = unsafe { hypura_sys::llama_model_load_from_file(c_path.as_ptr(), params) };
        anyhow::ensure!(!ptr.is_null(), "Failed to load model from {}", path.display());

        let vocab = unsafe { hypura_sys::llama_model_get_vocab(ptr) };
        anyhow::ensure!(!vocab.is_null(), "Failed to get vocab from model");

        Ok(Self { ptr, vocab })
    }

    pub fn as_ptr(&self) -> *mut hypura_sys::llama_model {
        self.ptr
    }

    pub fn n_layers(&self) -> i32 {
        unsafe { hypura_sys::llama_model_n_layer(self.ptr) }
    }

    /// Get the model's embedded chat template, if any.
    pub fn chat_template(&self) -> Option<String> {
        let ptr = unsafe { hypura_sys::llama_model_chat_template(self.ptr, ptr::null()) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string())
        }
    }

    /// Check if a token signals end of generation.
    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { hypura_sys::llama_vocab_is_eog(self.vocab, token) }
    }

    /// Tokenize text into token IDs.
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let c_text = CString::new(text).unwrap_or_default();
        let max = (text.len() as i32 + 32).max(64);
        let mut tokens = vec![0i32; max as usize];

        let n = unsafe {
            hypura_sys::llama_tokenize(
                self.vocab,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                max,
                add_bos,
                false,
            )
        };

        if n < 0 {
            // Buffer too small, retry
            tokens.resize((-n) as usize, 0);
            let n2 = unsafe {
                hypura_sys::llama_tokenize(
                    self.vocab,
                    c_text.as_ptr(),
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    -n,
                    add_bos,
                    false,
                )
            };
            tokens.truncate(n2.max(0) as usize);
        } else {
            tokens.truncate(n as usize);
        }
        tokens
    }

    /// Convert a token ID to its text piece.
    pub fn token_to_piece(&self, token: i32) -> String {
        let mut buf = vec![0u8; 128];
        let n = unsafe {
            hypura_sys::llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
                0,
                false,
            )
        };

        if n < 0 {
            buf.resize((-n) as usize, 0);
            let n2 = unsafe {
                hypura_sys::llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr() as *mut i8,
                    buf.len() as i32,
                    0,
                    false,
                )
            };
            buf.truncate(n2.max(0) as usize);
        } else {
            buf.truncate(n as usize);
        }
        String::from_utf8_lossy(&buf).to_string()
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hypura_sys::llama_model_free(self.ptr) }
        }
    }
}

/// Safe wrapper around `llama_context *`.
pub struct LlamaContext {
    ptr: *mut hypura_sys::llama_context,
}

impl LlamaContext {
    /// Create a new inference context from a loaded model.
    pub fn new(model: &LlamaModel, n_ctx: u32, n_batch: u32, n_threads: i32) -> anyhow::Result<Self> {
        Self::new_inner(model, n_ctx, n_batch, n_threads, None, std::ptr::null_mut(), None)
    }

    /// Create a context with a cb_eval callback for layer tracking.
    /// `callback_data` must outlive the context.
    pub fn new_with_callback(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
    ) -> anyhow::Result<Self> {
        Self::new_inner(model, n_ctx, n_batch, n_threads, cb_eval, callback_data, None)
    }

    /// Create a context with callback and KV cache quantization.
    pub fn new_with_callback_and_kv(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
        kv_quant: Option<crate::scheduler::types::KvQuantization>,
    ) -> anyhow::Result<Self> {
        Self::new_inner(model, n_ctx, n_batch, n_threads, cb_eval, callback_data, kv_quant)
    }

    fn new_inner(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
        kv_quant: Option<crate::scheduler::types::KvQuantization>,
    ) -> anyhow::Result<Self> {
        use crate::scheduler::types::KvQuantization;

        let mut params = unsafe { hypura_sys::llama_context_default_params() };
        params.n_ctx = n_ctx;
        params.n_batch = n_batch;
        params.n_ubatch = n_batch;
        params.n_threads = n_threads;
        params.n_threads_batch = n_threads;
        params.offload_kqv = true;

        if let Some(kv) = kv_quant {
            let ggml_type = match kv {
                KvQuantization::F16 => hypura_sys::ggml_type_GGML_TYPE_F16,
                KvQuantization::Q8_0 => hypura_sys::ggml_type_GGML_TYPE_Q8_0,
                KvQuantization::Q4_0 => hypura_sys::ggml_type_GGML_TYPE_Q4_0,
            };
            params.type_k = ggml_type;
            params.type_v = ggml_type;
            tracing::info!("KV cache quantization: {:?}", kv);
        }

        if cb_eval.is_some() {
            params.cb_eval = cb_eval;
            params.cb_eval_user_data = callback_data;
        }

        let ptr = unsafe { hypura_sys::llama_init_from_model(model.as_ptr(), params) };
        anyhow::ensure!(!ptr.is_null(), "Failed to create llama context");

        Ok(Self { ptr })
    }

    /// Decode a batch of tokens (prompt processing or single-token step).
    pub fn decode(&mut self, tokens: &[i32]) -> anyhow::Result<()> {
        let batch = unsafe {
            hypura_sys::llama_batch_get_one(tokens.as_ptr() as *mut i32, tokens.len() as i32)
        };
        let ret = unsafe { hypura_sys::llama_decode(self.ptr, batch) };
        match ret {
            0 => Ok(()),
            1 => anyhow::bail!("KV cache full — try reducing context or batch size"),
            _ => anyhow::bail!("llama_decode failed with code {ret}"),
        }
    }

    /// Get performance counters.
    pub fn perf(&self) -> PerfData {
        let data = unsafe { hypura_sys::llama_perf_context(self.ptr) };
        PerfData {
            t_load_ms: data.t_load_ms,
            t_p_eval_ms: data.t_p_eval_ms,
            t_eval_ms: data.t_eval_ms,
            n_p_eval: data.n_p_eval,
            n_eval: data.n_eval,
        }
    }

    pub fn as_ptr(&self) -> *mut hypura_sys::llama_context {
        self.ptr
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hypura_sys::llama_free(self.ptr) }
        }
    }
}

/// Safe wrapper around a llama sampler chain.
pub struct LlamaSampler {
    ptr: *mut hypura_sys::llama_sampler,
}

impl LlamaSampler {
    /// Build a sampler chain from parameters.
    pub fn new(params: &SamplingParams) -> Self {
        let chain_params = hypura_sys::llama_sampler_chain_params { no_perf: false };
        let ptr = unsafe { hypura_sys::llama_sampler_chain_init(chain_params) };

        unsafe {
            hypura_sys::llama_sampler_chain_add(ptr, hypura_sys::llama_sampler_init_top_k(params.top_k));
            hypura_sys::llama_sampler_chain_add(
                ptr,
                hypura_sys::llama_sampler_init_top_p(params.top_p, 1),
            );
            hypura_sys::llama_sampler_chain_add(
                ptr,
                hypura_sys::llama_sampler_init_min_p(params.min_p, 1),
            );
            hypura_sys::llama_sampler_chain_add(
                ptr,
                hypura_sys::llama_sampler_init_temp(params.temperature),
            );
            hypura_sys::llama_sampler_chain_add(
                ptr,
                hypura_sys::llama_sampler_init_dist(params.seed),
            );
        }

        Self { ptr }
    }

    /// Sample the next token. `idx = -1` means last token in context.
    pub fn sample(&mut self, ctx: &mut LlamaContext, idx: i32) -> i32 {
        unsafe { hypura_sys::llama_sampler_sample(self.ptr, ctx.as_ptr(), idx) }
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hypura_sys::llama_sampler_free(self.ptr) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_lifecycle() {
        let _backend = LlamaBackend::init();
        // Drop cleans up
    }

    #[test]
    fn test_sampling_params_default() {
        let p = SamplingParams::default();
        assert!(p.temperature > 0.0);
        assert!(p.top_k > 0);
        assert!(p.max_tokens > 0);
    }
}
