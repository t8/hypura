use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::model::gguf::GgufFile;
use crate::model::metadata::ModelMetadata;
use crate::server::ollama_types::{GgufInfo, ModelDetails, ModelTag};

/// A registered model available for on-demand loading.
#[derive(Debug, Clone)]
pub struct RegisteredModel {
    pub name: String,
    pub path: PathBuf,
    pub source: String, // "Local", "Ollama", or "Custom"
    pub size: u64,
    pub metadata: ModelMetadata,
}

impl RegisteredModel {
    pub fn to_model_tag(&self) -> ModelTag {
        ModelTag {
            name: self.name.clone(),
            model: self.name.clone(),
            size: self.size,
            details: ModelDetails {
                format: "gguf".into(),
                family: self.metadata.architecture.clone(),
                parameter_size: format_parameter_size(self.metadata.parameter_count),
                quantization_level: self
                    .metadata
                    .quantization
                    .clone()
                    .unwrap_or_else(|| "unknown".into()),
            },
        }
    }

    pub fn to_gguf_info(&self) -> GgufInfo {
        GgufInfo {
            file_size: self.size,
            architecture: self.metadata.architecture.clone(),
            parameter_count: self.metadata.parameter_count,
            quantization: self
                .metadata
                .quantization
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            context_length: self.metadata.context_length,
        }
    }
}

/// Discovers and tracks all local and Ollama models.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    pub models: HashMap<String, RegisteredModel>,
    pub custom_dir: Option<PathBuf>,
    pub ollama_dir: Option<PathBuf>,
}

impl ModelRegistry {
    /// Create a new registry scanning default directories and optional custom dirs.
    pub fn new(custom_dir: Option<PathBuf>, ollama_dir: Option<PathBuf>) -> Self {
        let mut registry = Self {
            models: HashMap::new(),
            custom_dir,
            ollama_dir,
        };
        registry.refresh();
        registry
    }

    /// Set or override the custom models directory at runtime and refresh.
    pub fn set_custom_dir(&mut self, dir: Option<PathBuf>) {
        self.custom_dir = dir;
        self.refresh();
    }

    /// Refresh and scan all directories for models.
    pub fn refresh(&mut self) {
        self.models.clear();

        // 1. Scan custom directory (if specified, or fallback to environment/default)
        let custom_dir = self.custom_dir.clone().or_else(|| {
            if let Ok(env_val) = std::env::var("HYPURA_MODELS_DIR") {
                let p = PathBuf::from(env_val);
                if p.is_dir() {
                    return Some(p);
                }
            }
            let realtek = PathBuf::from("/Volumes/RealtekData/models");
            if realtek.is_dir() {
                return Some(realtek);
            }
            None
        });

        if let Some(ref dir) = custom_dir {
            self.scan_directory(dir, "Custom");
        }

        // 2. Scan current working directory for *.gguf
        if let Ok(cwd) = std::env::current_dir() {
            self.scan_directory(&cwd, "Local");
        }

        // 3. Scan Ollama models directory
        let ollama_path = self.ollama_dir.clone().or_else(detect_ollama_dir);
        if let Some(ref o_dir) = ollama_path {
            self.scan_ollama_models(o_dir);
        }
    }

    /// Scan a directory for GGUF files (recursively up to depth 4).
    fn scan_directory(&mut self, dir: &Path, source_label: &str) {
        self.scan_directory_recursive(dir, dir, source_label, 0, 4);
    }

    fn scan_directory_recursive(
        &mut self,
        root_dir: &Path,
        current_dir: &Path,
        source_label: &str,
        depth: usize,
        max_depth: usize,
    ) {
        if !current_dir.is_dir() || depth > max_depth {
            return;
        }

        let entries = match fs::read_dir(current_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() || path.is_symlink() {
                if let Some(ext) = path.extension() {
                    if ext == "gguf" {
                        let stem = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default();

                        // For multi-part sharded models, skip non-first shards (00002-of-*, etc.)
                        if is_non_first_shard(&stem) {
                            continue;
                        }

                        if let Some(model) = self.inspect_gguf_file(&path, source_label) {
                            // 1. Primary key: exact filename stem
                            self.models.insert(model.name.clone(), model.clone());

                            // 2. Base stem without shard suffix (e.g. Qwen3.8-Flash-Next-UD-IQ3_XXS)
                            if let Some(pos) = stem.find("-00001-of-") {
                                let base_stem = stem[..pos].to_string();
                                self.models.entry(base_stem).or_insert_with(|| model.clone());
                            }

                            // 3. Parent directory aliases (e.g. "qwen3.8-flash-next", "UD-IQ3_XXS")
                            let mut parent = current_dir;
                            while parent != root_dir {
                                if let Some(folder_name) = parent.file_name() {
                                    let folder_str = folder_name.to_string_lossy().to_string();
                                    self.models.entry(folder_str).or_insert_with(|| model.clone());
                                }
                                if let Some(p) = parent.parent() {
                                    parent = p;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            } else if path.is_dir() && depth < max_depth {
                let name = path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
                // Skip hidden directories like .cache, .git
                if !name.starts_with('.') {
                    self.scan_directory_recursive(root_dir, &path, source_label, depth + 1, max_depth);
                }
            }
        }
    }

    /// Scan Ollama manifest repository: `~/.ollama/models/manifests/...`
    fn scan_ollama_models(&mut self, ollama_dir: &Path) {
        let manifests_dir = ollama_dir.join("manifests");
        let blobs_dir = ollama_dir.join("blobs");

        if !manifests_dir.exists() || !blobs_dir.exists() {
            return;
        }

        let mut manifest_files = Vec::new();
        collect_files_recursive(&manifests_dir, &mut manifest_files);

        for manifest_path in manifest_files {
            if let Some((model_name, blob_path)) =
                parse_ollama_manifest(&manifests_dir, &manifest_path, &blobs_dir)
            {
                if blob_path.exists() {
                    if let Some(mut model) = self.inspect_gguf_file(&blob_path, "Ollama") {
                        model.name = model_name.clone();
                        self.models.insert(model_name.clone(), model.clone());

                        // Also alias without `:latest` tag if tag is latest
                        if let Some(base_name) = model_name.strip_suffix(":latest") {
                            let mut aliased = model;
                            aliased.name = base_name.to_string();
                            self.models.insert(base_name.to_string(), aliased);
                        }
                    }
                }
            }
        }
    }

    /// Read GGUF header to extract metadata.
    fn inspect_gguf_file(&self, path: &Path, source_label: &str) -> Option<RegisteredModel> {
        let file_size = compute_total_shard_size(path);
        let gguf = GgufFile::open(path).ok()?;
        let metadata = ModelMetadata::from_gguf(&gguf).ok()?;

        let filename_stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());

        Some(RegisteredModel {
            name: filename_stem,
            path: path.to_path_buf(),
            source: source_label.into(),
            size: file_size,
            metadata,
        })
    }

    /// List all discovered models.
    pub fn list_models(&self) -> Vec<ModelTag> {
        let mut tags: Vec<ModelTag> = self.models.values().map(|m| m.to_model_tag()).collect();
        tags.sort_by(|a, b| a.name.cmp(&b.name));
        tags.dedup_by(|a, b| a.name == b.name);
        tags
    }

    /// Resolve a model by query (name, tag, alias, or file path).
    pub fn resolve_model(&self, query: &str) -> Option<&RegisteredModel> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            // Default to first available model if any
            return self.models.values().next();
        }

        // 1. Direct key match
        if let Some(model) = self.models.get(trimmed) {
            return Some(model);
        }

        // 2. Strip / add `:latest`
        if let Some(base) = trimmed.strip_suffix(":latest") {
            if let Some(model) = self.models.get(base) {
                return Some(model);
            }
        } else {
            let with_latest = format!("{trimmed}:latest");
            if let Some(model) = self.models.get(&with_latest) {
                return Some(model);
            }
        }

        // 3. Normalized `:` vs `-` (e.g. `mistral-small3.2-24b` vs `mistral-small3.2:24b`)
        let alt_name = if trimmed.contains(':') {
            trimmed.replace(':', "-")
        } else {
            trimmed.replace('-', ":")
        };
        if let Some(model) = self.models.get(&alt_name) {
            return Some(model);
        }

        // 4. Match path directly if provided as valid file path
        let query_path = Path::new(trimmed);
        if query_path.exists() {
            for model in self.models.values() {
                if model.path == query_path || model.path.ends_with(query_path) {
                    return Some(model);
                }
            }
        }

        // 5. Case-insensitive substring match
        let query_lower = trimmed.to_lowercase();
        for (name, model) in &self.models {
            if name.to_lowercase().contains(&query_lower) {
                return Some(model);
            }
        }

        None
    }
}

/// Helper: Resolve a model path by checking local filesystem first, then ModelRegistry
pub fn resolve_model_path(query: &str) -> anyhow::Result<PathBuf> {
    let direct_path = Path::new(query);
    if direct_path.exists() {
        return Ok(direct_path.to_path_buf());
    }

    let registry = ModelRegistry::new(None, None);
    if let Some(model) = registry.resolve_model(query) {
        return Ok(model.path.clone());
    }

    anyhow::bail!(
        "Model not found: '{query}'. Run `hypura list` to see available models."
    )
}

/// Detect default Ollama models directory
pub fn detect_ollama_dir() -> Option<PathBuf> {
    if let Ok(env_val) = std::env::var("OLLAMA_MODELS") {
        let p = PathBuf::from(env_val);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        let default_ollama = PathBuf::from(home).join(".ollama/models");
        if default_ollama.exists() {
            return Some(default_ollama);
        }
    }

    None
}

/// Parse an Ollama manifest file to extract (model_name, blob_path)
fn parse_ollama_manifest(
    manifests_root: &Path,
    manifest_file: &Path,
    blobs_root: &Path,
) -> Option<(String, PathBuf)> {
    let content = fs::read_to_string(manifest_file).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Find the layer with mediaType "application/vnd.ollama.image.model"
    let layers = json.get("layers")?.as_array()?;
    let mut model_digest = None;

    for layer in layers {
        let media_type = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
        if media_type == "application/vnd.ollama.image.model" {
            model_digest = layer.get("digest").and_then(|d| d.as_str());
            break;
        }
    }

    let digest = model_digest?;
    let blob_filename = digest.replace(':', "-");
    let blob_path = blobs_root.join(blob_filename);

    // Compute relative model name from manifests directory
    // e.g. manifests_root/registry.ollama.ai/library/mistral-small3.2/24b -> mistral-small3.2:24b
    let rel = manifest_file.strip_prefix(manifests_root).ok()?;
    let components: Vec<String> = rel
        .iter()
        .map(|c| c.to_string_lossy().to_string())
        .collect();

    let model_name = match components.as_slice() {
        // e.g. ["registry.ollama.ai", "library", "mistral-small3.2", "24b"]
        [_, namespace, model, tag] => {
            if namespace == "library" {
                format!("{model}:{tag}")
            } else {
                format!("{namespace}/{model}:{tag}")
            }
        }
        // e.g. ["library", "mistral-small3.2", "24b"]
        [namespace, model, tag] => {
            if namespace == "library" {
                format!("{model}:{tag}")
            } else {
                format!("{namespace}/{model}:{tag}")
            }
        }
        // fallback to file name or parent
        _ => manifest_file
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into()),
    };

    Some((model_name, blob_path))
}

/// Recursively collect all files in a directory
fn collect_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_files_recursive(&path, out);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
}

/// Check if a GGUF filename stem is a non-first shard (e.g. `*-00002-of-00004`).
pub fn is_non_first_shard(stem: &str) -> bool {
    if let Some(pos) = stem.find("-of-") {
        let prefix = &stem[..pos];
        if let Some(dash) = prefix.rfind('-') {
            let shard_num_str = &prefix[dash + 1..];
            if let Ok(num) = shard_num_str.parse::<u32>() {
                return num > 1;
            }
        }
    }
    false
}

/// Compute total size of all shards if this is the first shard of a multi-file model.
pub fn compute_total_shard_size(path: &Path) -> u64 {
    let stem = match path.file_stem().and_then(|s| s.to_str()) {
        Some(s) => s,
        None => return fs::metadata(path).map(|m| m.len()).unwrap_or(0),
    };
    if let Some(pos) = stem.find("-00001-of-") {
        let base_prefix = &stem[..pos];
        let total_str = &stem[pos + "-00001-of-".len()..];
        if let Ok(total_shards) = total_str.parse::<u32>() {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let mut total_bytes = 0u64;
            for shard_idx in 1..=total_shards {
                let shard_name = format!("{}-{:05}-of-{:05}.gguf", base_prefix, shard_idx, total_shards);
                let shard_path = parent.join(shard_name);
                if let Ok(meta) = fs::metadata(shard_path) {
                    total_bytes += meta.len();
                }
            }
            if total_bytes > 0 {
                return total_bytes;
            }
        }
    }
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn format_parameter_size(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1e6)
    } else {
        format!("{params}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_param_size() {
        assert_eq!(format_parameter_size(24_000_000_000), "24.0B");
        assert_eq!(format_parameter_size(1_500_000_000), "1.5B");
        assert_eq!(format_parameter_size(500_000_000), "500M");
    }

    #[test]
    fn test_is_non_first_shard() {
        assert!(!is_non_first_shard("model-00001-of-00004"));
        assert!(is_non_first_shard("model-00002-of-00004"));
        assert!(is_non_first_shard("model-00003-of-00004"));
        assert!(is_non_first_shard("model-00004-of-00004"));
        assert!(!is_non_first_shard("single-model"));
    }
}
