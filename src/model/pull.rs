use std::path::{Path, PathBuf};
use std::time::Duration;
use anyhow::{Context, Result};
use tokio::sync::mpsc;
use crate::server::ollama_types::PullResponseChunk;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullSpec {
    pub alias: String,
    pub filename: String,
    pub download_url: String,
    pub expected_size_bytes: Option<u64>,
}

/// Detect default models directory with priority:
/// 1. `HYPURA_MODELS_DIR` environment variable
/// 2. `/Volumes/RealtekData/models` if exists
/// 3. `~/.hypura/models` if exists
/// 4. `./models` in current working directory
pub fn detect_default_models_dir() -> PathBuf {
    if let Ok(env_val) = std::env::var("HYPURA_MODELS_DIR") {
        let p = PathBuf::from(env_val);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }

    let realtek = PathBuf::from("/Volumes/RealtekData/models");
    if realtek.is_dir() {
        return realtek;
    }

    if let Ok(home) = std::env::var("HOME") {
        let hypura_dir = PathBuf::from(home).join(".hypura/models");
        if hypura_dir.is_dir() {
            return hypura_dir;
        }
    }

    PathBuf::from("models")
}

/// Resolve a user-supplied model query (short alias, HuggingFace repo/tag, or URL)
/// into a concrete `PullSpec`.
pub fn resolve_pull_spec(query: &str) -> Result<PullSpec> {
    let q = query.trim();
    if q.is_empty() {
        anyhow::bail!("Model name or URL cannot be empty");
    }

    // 1. Direct HTTP/HTTPS URL
    if q.starts_with("http://") || q.starts_with("https://") {
        let clean_url = q.split('?').next().unwrap_or(q);
        let raw_filename = clean_url.split('/').next_back().unwrap_or("model.gguf");
        let filename = if raw_filename.is_empty() {
            "model.gguf"
        } else {
            raw_filename
        }.to_string();

        let alias = filename
            .strip_suffix(".gguf")
            .unwrap_or(&filename)
            .to_lowercase();
        return Ok(PullSpec {
            alias,
            filename,
            download_url: q.to_string(),
            expected_size_bytes: None,
        });
    }

    let q_lower = q.to_lowercase();

    // 2. Predefined high-performance aliases for 2026 / 2025 models
    match q_lower.as_str() {
        // Gemma 4 26B (Mixture / Active-4B routing)
        "gemma-4:26b" | "gemma4:26b" | "gemma-4-26b" | "gemma4-26b" | "gemma-4-26b-a4b" | "gemma4:26b-a4b" => {
            return Ok(PullSpec {
                alias: "gemma-4-26b-a4b".into(),
                filename: "google_gemma-4-26B-A4B-it-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF/resolve/main/google_gemma-4-26B-A4B-it-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(15_800_000_000),
            });
        }
        // Gemma 4 31B (Dense flagship)
        "gemma-4:31b" | "gemma4:31b" | "gemma-4-31b" | "gemma4-31b" => {
            return Ok(PullSpec {
                alias: "gemma-4-31b".into(),
                filename: "google_gemma-4-31B-it-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/bartowski/google_gemma-4-31B-it-GGUF/resolve/main/google_gemma-4-31B-it-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(19_200_000_000),
            });
        }
        // Qwen 3.8 27B
        "qwen3.8:27b" | "qwen3.8-27b" | "qwen-3.8:27b" | "qwen-3.8-27b" => {
            return Ok(PullSpec {
                alias: "qwen3.8-27b".into(),
                filename: "Qwen3.8-27B-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(16_200_000_000),
            });
        }
        // Qwen 2.5 Coder 32B Instruct
        "qwen2.5-coder:32b" | "qwen2.5-coder-32b" | "qwen-coder:32b" | "qwen-2.5-coder:32b" => {
            return Ok(PullSpec {
                alias: "qwen2.5-coder-32b-q3km".into(),
                filename: "qwen2.5-coder-32b-instruct-q3_k_m.gguf".into(),
                download_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q3_k_m.gguf".into(),
                expected_size_bytes: Some(15_400_000_000),
            });
        }
        // Mistral Small 24B Instruct
        "mistral-small:24b" | "mistral-small-24b" | "mistral-small-2501" | "mistral-24b" => {
            return Ok(PullSpec {
                alias: "mistral-small-24b-q4km".into(),
                filename: "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF/resolve/main/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(14_300_000_000),
            });
        }
        // DeepSeek Coder V2 Lite MoE (16B total, 2.4B active)
        "deepseek-coder-v2-lite" | "deepseek-coder:16b" | "deepseek-coder-lite" => {
            return Ok(PullSpec {
                alias: "deepseek-coder-v2-lite-q4km".into(),
                filename: "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(9_500_000_000),
            });
        }
        _ => {}
    }

    // 3. Hugging Face pattern: owner/repo[:file_or_quant] or hf.co/owner/repo[:file_or_quant]
    let stripped = q
        .strip_prefix("hf.co/")
        .or_else(|| q.strip_prefix("huggingface.co/"))
        .unwrap_or(q);

    if stripped.contains('/') {
        let (repo_part, tag_part) = match stripped.split_once(':') {
            Some((r, t)) => (r, Some(t)),
            None => (stripped, None),
        };

        let parts: Vec<&str> = repo_part.split('/').collect();
        if parts.len() == 2 {
            let owner = parts[0];
            let repo = parts[1];
            let filename = if let Some(tag) = tag_part {
                if tag.ends_with(".gguf") {
                    tag.to_string()
                } else {
                    format!("{}-{}.gguf", repo, tag)
                }
            } else {
                format!("{}-Q4_K_M.gguf", repo)
            };

            let download_url = format!(
                "https://huggingface.co/{}/{}/resolve/main/{}",
                owner, repo, filename
            );
            let alias = repo.to_lowercase();
            return Ok(PullSpec {
                alias,
                filename,
                download_url,
                expected_size_bytes: None,
            });
        }
    }

    anyhow::bail!(
        "Unknown model alias or repository '{query}'. Supported aliases: gemma-4:26b, gemma-4:31b, qwen3.8:27b, qwen2.5-coder:32b, mistral-small:24b, or specify HuggingFace 'owner/repo:quant'."
    )
}

/// Execute download via system `curl` with progress streaming, resume, and atomic completion.
pub async fn download_model_with_progress(
    spec: &PullSpec,
    models_dir: &Path,
    force: bool,
    progress_tx: Option<mpsc::Sender<PullResponseChunk>>,
) -> Result<PathBuf> {
    let target_dir = models_dir.join(&spec.alias);
    tokio::fs::create_dir_all(&target_dir).await.with_context(|| {
        format!("Failed to create target directory {}", target_dir.display())
    })?;

    let final_path = target_dir.join(&spec.filename);
    let part_path = target_dir.join(format!("{}.part", spec.filename));

    // If final file already exists and force is false, return immediately
    if final_path.exists() && !force {
        if let Some(ref tx) = progress_tx {
            let size = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
            let _ = tx
                .send(PullResponseChunk {
                    status: "already downloaded".into(),
                    digest: None,
                    total: Some(size),
                    completed: Some(size),
                })
                .await;
            let _ = tx
                .send(PullResponseChunk {
                    status: "success".into(),
                    digest: None,
                    total: Some(size),
                    completed: Some(size),
                })
                .await;
        }
        return Ok(final_path);
    }

    if let Some(ref tx) = progress_tx {
        let _ = tx
            .send(PullResponseChunk {
                status: "pulling manifest".into(),
                digest: None,
                total: spec.expected_size_bytes,
                completed: Some(0),
            })
            .await;
    }

    // Try to get total size via curl HEAD request
    let total_bytes = match get_remote_content_length(&spec.download_url).await {
        Ok(len) if len > 0 => Some(len),
        _ => spec.expected_size_bytes,
    };

    if let Some(ref tx) = progress_tx {
        let completed = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
        let _ = tx
            .send(PullResponseChunk {
                status: format!("downloading {}", spec.filename),
                digest: None,
                total: total_bytes,
                completed: Some(completed),
            })
            .await;
    }

    // Spawn curl with resume (-C -), follow-redirects (-L), silent but show error (-sS)
    let mut cmd = tokio::process::Command::new("curl");
    cmd.arg("-L")
        .arg("-C")
        .arg("-")
        .arg("-f")
        .arg("-sS")
        .arg("-o")
        .arg(&part_path)
        .arg(&spec.download_url);

    let mut child = cmd.spawn().with_context(|| {
        format!(
            "Failed to launch curl process for download from {}",
            spec.download_url
        )
    })?;

    // Monitor file size while curl runs in background
    let part_clone = part_path.clone();
    let tx_clone = progress_tx.clone();
    let monitor_handle = tokio::spawn(async move {
        let mut last_completed = 0u64;
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if let Ok(meta) = tokio::fs::metadata(&part_clone).await {
                let current = meta.len();
                if current != last_completed {
                    last_completed = current;
                    if let Some(ref tx) = tx_clone {
                        if tx
                            .send(PullResponseChunk {
                                status: "downloading".into(),
                                digest: None,
                                total: total_bytes,
                                completed: Some(current),
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        }
    });

    let status = child.wait().await?;
    monitor_handle.abort();

    if !status.success() {
        anyhow::bail!(
            "Download failed: curl exited with status {:?}. Partial download kept at {}",
            status.code(),
            part_path.display()
        );
    }

    // Atomic rename .part -> final .gguf
    tokio::fs::rename(&part_path, &final_path)
        .await
        .with_context(|| {
            format!(
                "Failed to rename {} to {}",
                part_path.display(),
                final_path.display()
            )
        })?;

    let final_size = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);

    if let Some(ref tx) = progress_tx {
        let _ = tx
            .send(PullResponseChunk {
                status: "verifying sha256 digest".into(),
                digest: None,
                total: Some(final_size),
                completed: Some(final_size),
            })
            .await;
        let _ = tx
            .send(PullResponseChunk {
                status: "success".into(),
                digest: None,
                total: Some(final_size),
                completed: Some(final_size),
            })
            .await;
    }

    Ok(final_path)
}

/// Helper: query HTTP Content-Length header via curl HEAD request
async fn get_remote_content_length(url: &str) -> Result<u64> {
    let output = tokio::process::Command::new("curl")
        .arg("-sI")
        .arg("-L")
        .arg(url)
        .output()
        .await?;

    if !output.status.success() {
        anyhow::bail!("HEAD request failed");
    }

    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines().rev() {
        let line_lower = line.to_lowercase();
        if line_lower.starts_with("content-length:") {
            if let Some(val_str) = line.split(':').nth(1) {
                if let Ok(num) = val_str.trim().parse::<u64>() {
                    return Ok(num);
                }
            }
        }
    }

    anyhow::bail!("Content-Length header not found");
}

/// Delete a model and its associated files (including all shards if split).
/// Returns the model display name and total bytes freed.
pub fn delete_model_files(
    query: &str,
    models_dir: Option<&Path>,
    registry: &mut crate::server::registry::ModelRegistry,
) -> Result<(String, u64)> {
    // If a custom models_dir was provided, set it on registry and refresh
    if let Some(dir) = models_dir {
        registry.set_custom_dir(Some(dir.to_path_buf()));
    } else {
        registry.refresh();
    }

    let model_entry = registry
        .resolve_model(query)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", query))?;

    let main_path = model_entry.path.clone();
    let model_name = model_entry.name.clone();
    let mut total_freed = 0u64;

    if !main_path.exists() {
        anyhow::bail!("Model file does not exist at {}", main_path.display());
    }

    // Check if sharded model (e.g. *-00001-of-00004.gguf)
    let parent_dir = main_path.parent();
    let filename_str = main_path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    if let Some(pos) = filename_str.find("-00001-of-") {
        let prefix = &filename_str[..pos];
        if let Some(dir) = parent_dir {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with(prefix) && name.ends_with(".gguf") {
                            if let Ok(meta) = std::fs::metadata(&p) {
                                total_freed += meta.len();
                            }
                            let _ = std::fs::remove_file(&p);
                        }
                    }
                }
            }
        }
    } else {
        if let Ok(meta) = std::fs::metadata(&main_path) {
            total_freed += meta.len();
        }
        std::fs::remove_file(&main_path)
            .with_context(|| format!("Failed to delete {}", main_path.display()))?;
    }

    // If parent directory is empty or only had this model, attempt to clean up empty dir
    if let Some(dir) = parent_dir {
        if let Some(custom_root) = models_dir.or(registry.custom_dir.as_deref()) {
            if dir != custom_root {
                let _ = std::fs::remove_dir(dir); // only succeeds if empty
            }
        }
    }

    // Refresh registry to update tags and loaded state
    registry.refresh();

    Ok((model_name, total_freed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_pull_spec_aliases() {
        let gemma = resolve_pull_spec("gemma-4:26b").unwrap();
        assert_eq!(gemma.alias, "gemma-4-26b-a4b");
        assert!(gemma.filename.contains("gemma-4-26B-A4B"));
        assert!(gemma.download_url.contains("bartowski"));

        let qwen = resolve_pull_spec("qwen3.8:27b").unwrap();
        assert_eq!(qwen.alias, "qwen3.8-27b");
        assert!(qwen.filename.contains("Qwen3.8-27B-Q4_K_M"));

        let coder = resolve_pull_spec("qwen2.5-coder:32b").unwrap();
        assert_eq!(coder.alias, "qwen2.5-coder-32b-q3km");
        assert!(coder.filename.contains("q3_k_m"));

        let mistral = resolve_pull_spec("mistral-small:24b").unwrap();
        assert_eq!(mistral.alias, "mistral-small-24b-q4km");
    }

    #[test]
    fn test_resolve_pull_spec_direct_url() {
        let spec = resolve_pull_spec("https://example.com/models/test-model-q4.gguf").unwrap();
        assert_eq!(spec.filename, "test-model-q4.gguf");
        assert_eq!(spec.alias, "test-model-q4");
    }

    #[test]
    fn test_resolve_pull_spec_huggingface_repo() {
        let spec = resolve_pull_spec("unsloth/Qwen3.8-27B-GGUF:Q4_K_M").unwrap();
        assert_eq!(spec.alias, "qwen3.8-27b-gguf");
        assert_eq!(spec.filename, "Qwen3.8-27B-GGUF-Q4_K_M.gguf");
        assert!(spec.download_url.contains("unsloth/Qwen3.8-27B-GGUF/resolve/main/"));
    }
}
