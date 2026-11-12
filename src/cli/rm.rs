use std::path::PathBuf;
use anyhow::Result;

use hypura::model::pull::delete_model_files;
use hypura::server::registry::ModelRegistry;

pub fn run(
    model: String,
    models_dir: Option<PathBuf>,
) -> Result<()> {
    let mut registry = ModelRegistry::new(models_dir.clone(), None);

    let (deleted_name, bytes_freed) =
        delete_model_files(&model, models_dir.as_deref(), &mut registry)?;

    let freed_gb = bytes_freed as f64 / (1024.0 * 1024.0 * 1024.0);

    println!();
    println!("✓ Deleted model '{}'", deleted_name);
    println!("  Space freed: {:.2} GB", freed_gb);
    println!();

    Ok(())
}
