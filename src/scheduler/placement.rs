use crate::model::gguf::GgufFile;
use crate::profiler::types::HardwareProfile;
use crate::scheduler::types::PlacementPlan;

pub fn compute_placement(
    _model: &GgufFile,
    _hardware: &HardwareProfile,
) -> anyhow::Result<PlacementPlan> {
    anyhow::bail!("Not yet implemented: placement optimizer")
}
