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
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u32,
    pub max_tokens: u32,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 42,
            max_tokens: 512,
            stop_sequences: Vec::new(),
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

    /// Sliding-window attention size (`0` if the model does not use SWA).
    pub fn n_swa(&self) -> i32 {
        unsafe { hypura_sys::llama_model_n_swa(self.ptr) }
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

    /// Check if a token is a control / special token.
    pub fn is_control(&self, token: i32) -> bool {
        unsafe { hypura_sys::llama_vocab_is_control(self.vocab, token) }
    }

    /// Tokenize text into token IDs.
    ///
    /// `add_bos`: ask the tokenizer to add BOS/EOS as appropriate (controlled
    /// by `tokenizer.ggml.add_bos_token` in the GGUF metadata).
    /// `parse_special`: when true, special tokens like `<start_of_turn>` in the
    /// input text are matched against the vocab and tokenized as their special
    /// IDs. When false, they are tokenized as raw bytes (which produces
    /// gibberish when feeding chat-templated prompts to instruction models).
    pub fn tokenize(&self, text: &str, add_bos: bool, parse_special: bool) -> Vec<i32> {
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
                parse_special,
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
                    parse_special,
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
                true,
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
                    true,
                )
            };
            buf.truncate(n2.max(0) as usize);
        } else {
            buf.truncate(n as usize);
        }
        String::from_utf8_lossy(&buf).to_string()
    }

    /// Apply a chat template to a series of `(role, content)` message tuples using llama.cpp's
    /// built-in template engine.
    ///
    /// If `custom_template` is `None`, uses the model's embedded template.
    /// `add_assistant_prompt` controls whether to append the assistant turn header (e.g. `<|im_start|>assistant\n`).
    pub fn apply_chat_template(
        &self,
        custom_template: Option<&str>,
        messages: &[(&str, &str)],
        add_assistant_prompt: bool,
    ) -> Option<String> {
        let custom_c = custom_template.and_then(|t| std::ffi::CString::new(t).ok());
        let tmpl_ptr = if let Some(ref c) = custom_c {
            c.as_ptr()
        } else {
            unsafe { hypura_sys::llama_model_chat_template(self.ptr, std::ptr::null()) }
        };

        if tmpl_ptr.is_null() {
            return None;
        }

        let mut role_cstrings = Vec::with_capacity(messages.len());
        let mut content_cstrings = Vec::with_capacity(messages.len());
        let mut chat_msgs = Vec::with_capacity(messages.len());

        for (role, content) in messages {
            let r = std::ffi::CString::new(*role).ok()?;
            let c = std::ffi::CString::new(*content).ok()?;
            chat_msgs.push(hypura_sys::llama_chat_message {
                role: r.as_ptr(),
                content: c.as_ptr(),
            });
            role_cstrings.push(r);
            content_cstrings.push(c);
        }

        // Measure formatted buffer size
        let needed_size = unsafe {
            hypura_sys::llama_chat_apply_template(
                tmpl_ptr,
                chat_msgs.as_ptr(),
                chat_msgs.len(),
                add_assistant_prompt,
                std::ptr::null_mut(),
                0,
            )
        };

        if needed_size < 0 {
            return None;
        }

        let mut buf = vec![0u8; (needed_size as usize) + 1];
        let res = unsafe {
            hypura_sys::llama_chat_apply_template(
                tmpl_ptr,
                chat_msgs.as_ptr(),
                chat_msgs.len(),
                add_assistant_prompt,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
            )
        };

        if res < 0 {
            return None;
        }

        buf.truncate(res as usize);
        String::from_utf8(buf).ok()
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
        Self::new_inner(model, n_ctx, n_batch, n_batch, n_threads, None, std::ptr::null_mut(), None, true)
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
        Self::new_inner(model, n_ctx, n_batch, n_batch, n_threads, cb_eval, callback_data, None, false)
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
        Self::new_inner(
            model,
            n_ctx,
            n_batch,
            n_batch,
            n_threads,
            cb_eval,
            callback_data,
            kv_quant,
            cb_eval.is_none(),
        )
    }

    /// Create a context with explicit `n_ubatch` and `op_offload` control.
    /// Disabling `op_offload` and capping `n_ubatch` on partially GPU-offloaded models
    /// avoids hundreds of CPU<->Metal graph splits and prevents staging CPU weights into
    /// Metal command buffers under tight unified memory pressure.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_options(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_ubatch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
        kv_quant: Option<crate::scheduler::types::KvQuantization>,
        op_offload: bool,
    ) -> anyhow::Result<Self> {
        Self::new_inner(
            model,
            n_ctx,
            n_batch,
            n_ubatch,
            n_threads,
            cb_eval,
            callback_data,
            kv_quant,
            op_offload,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_ubatch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
        kv_quant: Option<crate::scheduler::types::KvQuantization>,
        op_offload: bool,
    ) -> anyhow::Result<Self> {
        use crate::scheduler::types::KvQuantization;

        let mut params = unsafe { hypura_sys::llama_context_default_params() };
        params.n_ctx = n_ctx;
        params.n_batch = n_batch;
        params.n_ubatch = n_ubatch.min(n_batch).max(1);
        params.n_threads = n_threads;
        params.n_threads_batch = n_threads;
        params.offload_kqv = true;
        params.op_offload = op_offload;

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

    /// Save context/KV state to a disk file.
    pub fn save_state_to_file(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid unicode path"))?;
        let c_path = std::ffi::CString::new(path_str)?;

        let ok = unsafe { hypura_sys::llama_state_save_file(self.ptr, c_path.as_ptr(), std::ptr::null(), 0) };
        if !ok {
            anyhow::bail!("Failed to save llama context state to {}", path.display());
        }
        tracing::info!("Saved KV cache state to {}", path.display());
        Ok(())
    }

    /// Load context/KV state from a disk file.
    pub fn load_state_from_file(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        if !path.exists() {
            anyhow::bail!("State file {} does not exist", path.display());
        }
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid unicode path"))?;
        let c_path = std::ffi::CString::new(path_str)?;

        let ok = unsafe { hypura_sys::llama_state_load_file(self.ptr, c_path.as_ptr(), std::ptr::null_mut(), 0, std::ptr::null_mut()) };
        if !ok {
            anyhow::bail!("Failed to load llama context state from {}", path.display());
        }
        tracing::info!("Loaded KV cache state from {}", path.display());
        Ok(())
    }

    /// Returns the total serialized size (in bytes) of this context's state.
    pub fn state_size(&self) -> usize {
        unsafe { hypura_sys::llama_state_get_size(self.ptr) }
    }

    /// Serializes the entire context state (KV cache and recurrent hidden states)
    /// into an in-memory buffer.
    pub fn save_state_to_memory(&self) -> anyhow::Result<Vec<u8>> {
        let size = self.state_size();
        if size == 0 {
            anyhow::bail!("Failed to determine context state size");
        }
        let mut buf = vec![0u8; size];
        let copied = unsafe {
            hypura_sys::llama_state_get_data(self.ptr, buf.as_mut_ptr(), size)
        };
        if copied == 0 {
            anyhow::bail!("Failed to copy context state to memory buffer");
        }
        buf.truncate(copied);
        Ok(buf)
    }

    /// Deserializes and restores context state (KV cache and recurrent hidden states)
    /// from an in-memory buffer.
    pub fn load_state_from_memory(&mut self, data: &[u8]) -> anyhow::Result<()> {
        let read = unsafe {
            hypura_sys::llama_state_set_data(self.ptr, data.as_ptr(), data.len())
        };
        if read == 0 {
            anyhow::bail!("Failed to restore context state from memory buffer");
        }
        Ok(())
    }

    pub fn as_ptr(&self) -> *mut hypura_sys::llama_context {
        self.ptr
    }

    /// Clear all KV / recurrent sequence metadata and buffers so decoding starts fresh from position 0.
    pub fn clear_memory(&mut self) {
        let mem = unsafe { hypura_sys::llama_get_memory(self.ptr) };
        if !mem.is_null() {
            unsafe { hypura_sys::llama_memory_clear(mem, true) };
        }
    }

    /// Trim sequence 0 KV memory to retain `[0, keep_len)` and remove `[keep_len, inf)`.
    ///
    /// Returns `keep_len` if the prefix remains valid (including SWA window & recurrent state
    /// checks), or `0` after clearing memory if partial sequence removal is unsupported or
    /// would invalidate the sliding-window cache.
    pub fn trim_kv_to_prefix(&mut self, keep_len: usize, n_swa: i32) -> usize {
        if keep_len == 0 {
            self.clear_memory();
            return 0;
        }

        let mem = unsafe { hypura_sys::llama_get_memory(self.ptr) };
        if mem.is_null() {
            return 0;
        }

        let pos_max = unsafe { hypura_sys::llama_memory_seq_pos_max(mem, 0) };
        let target_pos_max = (keep_len as i32) - 1;

        if pos_max < target_pos_max {
            self.clear_memory();
            return 0;
        }

        if pos_max > target_pos_max {
            let ok = unsafe { hypura_sys::llama_memory_seq_rm(mem, 0, keep_len as i32, -1) };
            if !ok {
                self.clear_memory();
                return 0;
            }
        }

        let new_min = unsafe { hypura_sys::llama_memory_seq_pos_min(mem, 0) };
        let new_max = unsafe { hypura_sys::llama_memory_seq_pos_max(mem, 0) };
        let swa_valid = n_swa <= 0 || new_min == 0 || new_min <= (keep_len as i32 - n_swa);

        if new_min < 0 || new_max != target_pos_max || !swa_valid {
            self.clear_memory();
            return 0;
        }

        keep_len
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
            if params.repeat_penalty > 1.0 {
                hypura_sys::llama_sampler_chain_add(
                    ptr,
                    hypura_sys::llama_sampler_init_penalties(
                        params.repeat_last_n,
                        params.repeat_penalty,
                        0.0,
                        0.0,
                    ),
                );
            }
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
        let token = unsafe { hypura_sys::llama_sampler_sample(self.ptr, ctx.as_ptr(), idx) };
        unsafe {
            hypura_sys::llama_sampler_accept(self.ptr, token);
        }
        token
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

    #[test]
    fn test_llama_chat_apply_template() {
        let _backend = LlamaBackend::init();
        let tmpl = std::ffi::CString::new("chatml").unwrap();
        let r1 = std::ffi::CString::new("user").unwrap();
        let c1 = std::ffi::CString::new("hello world").unwrap();
        let msgs = [hypura_sys::llama_chat_message {
            role: r1.as_ptr(),
            content: c1.as_ptr(),
        }];

        let len = unsafe {
            hypura_sys::llama_chat_apply_template(
                tmpl.as_ptr(),
                msgs.as_ptr(),
                msgs.len(),
                true,
                std::ptr::null_mut(),
                0,
            )
        };
        assert!(len > 0);
        let mut buf = vec![0u8; (len + 1) as usize];
        let res = unsafe {
            hypura_sys::llama_chat_apply_template(
                tmpl.as_ptr(),
                msgs.as_ptr(),
                msgs.len(),
                true,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
            )
        };
        assert_eq!(res, len);
        buf.truncate(res as usize);
        let formatted = String::from_utf8(buf).unwrap();
        assert!(formatted.contains("<|im_start|>user\nhello world<|im_end|>"));
        assert!(formatted.contains("<|im_start|>assistant\n"));
    }
}
