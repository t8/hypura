use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::compute::inference::{
    compute_gpu_budget, gpu_layers_from_placement, load_model, InferenceConfig, LoadedModel,
};
use crate::model::gguf::GgufFile;
use crate::profiler;
use crate::server::ollama_types::GgufInfo;
use crate::server::registry::{ModelRegistry, RegisteredModel};

pub struct ActiveModel {
    pub name: String,
    pub path: PathBuf,
    pub loaded: Arc<std::sync::Mutex<LoadedModel>>,
    pub gguf_info: GgufInfo,
    pub context_size: u32,
    pub last_used: Instant,
    pub entry: RegisteredModel,
}

pub struct ModelManager {
    pub registry: ModelRegistry,
    pub active_model: Option<ActiveModel>,
    pub default_context: u32,
    pub default_kv_quant: Option<crate::scheduler::types::KvQuantization>,
}

impl ModelManager {
    pub fn new(
        registry: ModelRegistry,
        default_context: u32,
        initial_model_path: Option<&str>,
        default_kv_quant: Option<crate::scheduler::types::KvQuantization>,
    ) -> anyhow::Result<Self> {
        let mut manager = Self {
            registry,
            active_model: None,
            default_context,
            default_kv_quant,
        };

        if let Some(initial_name_or_path) = initial_model_path {
            manager.get_or_load(initial_name_or_path, Some(default_context))?;
        }

        Ok(manager)
    }

    /// Load or reuse an active model based on request model name and context size.
    /// If an active model is loaded but requested context exceeds its loaded capacity,
    /// dynamically reloads the model with adapted layer offloading to fit GPU memory.
    pub fn get_or_load(
        &mut self,
        requested_name: &str,
        requested_ctx: Option<u32>,
    ) -> anyhow::Result<(Arc<std::sync::Mutex<LoadedModel>>, String, GgufInfo)> {
        // 1. Check if an active model matches requested_name
        let active_match = self.active_model.as_ref().map(|active| {
            let a_lower = active.name.to_lowercase();
            let r_lower = requested_name.to_lowercase();
            a_lower == r_lower
                || requested_name.is_empty()
                || a_lower.starts_with(&r_lower)
                || r_lower.starts_with(&a_lower)
                || active.path == std::path::Path::new(requested_name)
        }).unwrap_or(false);

        if active_match {
            let active = self.active_model.as_mut().unwrap();
            let max_model_ctx = if active.entry.metadata.context_length > 0 {
                active.entry.metadata.context_length
            } else {
                131072
            };

            // Desired context: if client requested a context, clamp to model native limit;
            // otherwise use default_context or currently loaded context_size
            let needed_ctx = requested_ctx
                .map(|c| c.min(max_model_ctx))
                .unwrap_or(self.default_context);

            // If current model context capacity already covers the needed context, reuse it directly
            if active.context_size >= needed_ctx {
                active.last_used = Instant::now();
                return Ok((
                    active.loaded.clone(),
                    active.name.clone(),
                    active.gguf_info.clone(),
                ));
            }

            // Otherwise, we need to adapt layer placement to the larger context!
            tracing::info!(
                "Adapting active model '{}' context capacity: current {} -> requested {} (recalculating GPU layer offload)",
                active.name,
                active.context_size,
                needed_ctx
            );
            let entry = active.entry.clone();
            return self.load_model_entry(&entry, needed_ctx);
        }

        // 2. Resolve model path from registry or direct path
        let resolved = if let Some(reg_model) = self.registry.resolve_model(requested_name) {
            reg_model.clone()
        } else if Path::new(requested_name).exists() {
            // Direct file path fallback
            let p = Path::new(requested_name);
            let file_size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            let gguf = GgufFile::open(p)?;
            let metadata = crate::model::metadata::ModelMetadata::from_gguf(&gguf)?;
            RegisteredModel {
                name: p
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".into()),
                path: p.to_path_buf(),
                source: "Direct".into(),
                size: file_size,
                metadata,
            }
        } else {
            anyhow::bail!(
                "Model '{}' not found. Run 'hypura list' to see available models.",
                requested_name
            );
        };

        let max_model_ctx = if resolved.metadata.context_length > 0 {
            resolved.metadata.context_length
        } else {
            131072
        };
        let target_context = requested_ctx
            .map(|c| c.min(max_model_ctx))
            .unwrap_or(self.default_context);

        // 3. Unload old model and load the requested model
        self.load_model_entry(&resolved, target_context)
    }

    /// Load a model from a RegisteredModel entry
    fn load_model_entry(
        &mut self,
        model_entry: &RegisteredModel,
        context_size: u32,
    ) -> anyhow::Result<(Arc<std::sync::Mutex<LoadedModel>>, String, GgufInfo)> {
        // Drop current model first to reclaim Metal GPU memory
        self.active_model = None;

        tracing::info!(
            "Loading model '{}' ({}) with context {}...",
            model_entry.name,
            model_entry.path.display(),
            context_size
        );

        let gguf = GgufFile::open(&model_entry.path)?;
        let hardware = match profiler::load_cached_profile()? {
            Some(p) if !profiler::is_profile_stale(&p) => p,
            _ => {
                let p = profiler::run_full_profile()?;
                let _ = profiler::save_profile(&p);
                p
            }
        };

        let plan = crate::scheduler::placement::compute_placement_with_context_and_kv_quant(
            &gguf,
            &hardware,
            context_size,
            self.default_kv_quant,
        )?;
        let gpu_budget = compute_gpu_budget(&hardware, &model_entry.metadata, context_size);
        let n_gpu_layers = gpu_layers_from_placement(&plan, &gguf, gpu_budget);

        let config = InferenceConfig {
            n_ctx: context_size,
            ..InferenceConfig::default()
        };

        let loaded = load_model(&model_entry.path, &config, n_gpu_layers, &plan, &gguf)?;
        let gguf_info = model_entry.to_gguf_info();
        let loaded_arc = Arc::new(std::sync::Mutex::new(loaded));

        self.active_model = Some(ActiveModel {
            name: model_entry.name.clone(),
            path: model_entry.path.clone(),
            loaded: loaded_arc.clone(),
            gguf_info: gguf_info.clone(),
            context_size,
            last_used: Instant::now(),
            entry: model_entry.clone(),
        });

        tracing::info!(
            "Model '{}' loaded successfully ({} layers offloaded to GPU)",
            model_entry.name,
            n_gpu_layers
        );

        Ok((loaded_arc, model_entry.name.clone(), gguf_info))
    }

    /// Load a model directly by file path
    #[allow(dead_code)]
    fn load_model_by_path(
        &mut self,
        path: &Path,
        context_size: u32,
    ) -> anyhow::Result<(Arc<std::sync::Mutex<LoadedModel>>, String, GgufInfo)> {
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let gguf = GgufFile::open(path)?;
        let metadata = crate::model::metadata::ModelMetadata::from_gguf(&gguf)?;
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());

        let entry = RegisteredModel {
            name,
            path: path.to_path_buf(),
            source: "Direct".into(),
            size: file_size,
            metadata,
        };

        self.load_model_entry(&entry, context_size)
    }

    /// Check if the active model has been idle for longer than `idle_timeout` and unload it if so.
    /// Returns true if a model was unloaded.
    pub fn maybe_unload_idle(&mut self, idle_timeout: std::time::Duration) -> bool {
        if let Some(ref mut active) = self.active_model {
            // Ensure we never unload a model while a request is in flight:
            // 1. Arc::strong_count(&active.loaded) > 1 means an active route handler currently holds a reference.
            // 2. active.loaded.try_lock().is_err() means a thread is actively executing inference inside the lock.
            let is_busy = Arc::strong_count(&active.loaded) > 1 || active.loaded.try_lock().is_err();
            if is_busy {
                active.last_used = Instant::now();
                return false;
            }

            if active.last_used.elapsed() >= idle_timeout {
                tracing::info!(
                    "Active model '{}' has been idle for {:.1}s (timeout: {:.1}s); unloading to reclaim unified memory",
                    active.name,
                    active.last_used.elapsed().as_secs_f64(),
                    idle_timeout.as_secs_f64()
                );
                self.active_model = None;
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_unload_idle_timeout() {
        let registry = ModelRegistry::new(None, None);
        let mut manager = ModelManager {
            registry,
            active_model: None,
            default_context: 2048,
            default_kv_quant: None,
        };

        // No model loaded -> returns false
        assert!(!manager.maybe_unload_idle(std::time::Duration::from_secs(60)));
    }
}
