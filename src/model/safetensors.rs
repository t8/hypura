use std::path::Path;

use crate::model::metadata::ModelMetadata;

pub fn parse_header(_path: &Path) -> anyhow::Result<ModelMetadata> {
    anyhow::bail!("Not yet implemented: safetensors parser")
}
