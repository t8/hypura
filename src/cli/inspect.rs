use std::path::Path;

use hypura::model::{gguf::GgufFile, metadata::ModelMetadata, tensor_role::TensorRole};

pub fn run(model_path: &str, show_tensors: bool) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    anyhow::ensure!(path.exists(), "Model file not found: {model_path}");

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "gguf" => inspect_gguf(path, show_tensors),
        "safetensors" => anyhow::bail!("Safetensors inspect not yet implemented"),
        _ => anyhow::bail!("Unsupported model format: .{ext}"),
    }
}

fn inspect_gguf(path: &Path, show_tensors: bool) -> anyhow::Result<()> {
    let gguf = GgufFile::open(path)?;
    let metadata = ModelMetadata::from_gguf(&gguf)?;

    println!("Model: {}", path.display());
    println!("  Format: GGUF v{}", gguf.version);
    println!("  Architecture: {}", metadata.architecture);
    println!("  Parameters: {}", format_params(metadata.parameter_count));
    println!("  Layers: {}", metadata.num_layers);
    println!("  Embedding dim: {}", metadata.embedding_dim);
    println!("  Attention heads: {}", metadata.num_heads);
    println!("  KV heads: {}", metadata.num_kv_heads);
    println!("  Vocab size: {}", metadata.vocab_size);
    println!("  Context length: {}", metadata.context_length);
    if let Some(ref q) = metadata.quantization {
        println!("  Quantization: {q}");
    }
    if metadata.is_moe {
        println!(
            "  MoE: {} experts, {} active per token",
            metadata.num_experts.unwrap_or(0),
            metadata.num_experts_used.unwrap_or(0)
        );
    }
    println!(
        "  Total size: {:.1} GB",
        gguf.total_tensor_bytes() as f64 / (1u64 << 30) as f64
    );
    println!("  Tensors: {}", gguf.tensors.len());

    if show_tensors {
        println!();
        println!("  {:50} {:>12} {:>8} {:>10}", "Name", "Size", "Type", "Role");
        println!("  {}", "-".repeat(84));
        for t in &gguf.tensors {
            let role = TensorRole::from_name(&t.name);
            println!(
                "  {:50} {:>12} {:>8} {:>10}",
                t.name,
                format_bytes(t.size_bytes),
                format!("{:?}", t.dtype),
                format!("{role:?}"),
            );
        }
    }

    Ok(())
}

fn format_params(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1e6)
    } else {
        format!("{count}")
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2} GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KB", bytes as f64 / (1u64 << 10) as f64)
    } else {
        format!("{bytes} B")
    }
}
