use std::num::NonZeroUsize;

use stwo_backend_cuda::gpu_default_pool_memory;

use super::{ResidentRuntimeError, ResidentSessionError, ResidentSessionTelemetry};
use crate::fixed_table_materializer::PEDERSEN_POINTS_18_EVALUATION_BYTES;
use crate::graphs::GraphWorkspace;
use crate::memory_ledger::{AllocatorPoolCheckpoint, PhysicalMemoryInputs};

pub(super) fn capture_pool_checkpoint(
    workspace: &GraphWorkspace,
    telemetry: &mut ResidentSessionTelemetry,
) -> Result<(), ResidentSessionError> {
    let arena_bytes = workspace
        .plan()
        .total_words()
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(ResidentSessionError::SizeOverflow)?;
    let default_attributed_bytes = workspace
        .plan()
        .requires_registered_pedersen_table()
        .then_some(PEDERSEN_POINTS_18_EVALUATION_BYTES)
        .unwrap_or(0);
    let isolated = workspace
        .arena()
        .context()
        .pool_memory()
        .map_err(ResidentRuntimeError::from)?;
    let default = gpu_default_pool_memory().map_err(ResidentRuntimeError::from)?;
    let checkpoint = AllocatorPoolCheckpoint::try_new(
        isolated.used_bytes,
        isolated.reserved_bytes,
        arena_bytes,
        default.used_bytes,
        default.reserved_bytes,
        default_attributed_bytes,
    )
    .map_err(ResidentSessionError::PhysicalMemory)?;
    let inputs = telemetry
        .physical_memory_inputs
        .clone()
        .with_allocator_pool_checkpoint(checkpoint)
        .map_err(ResidentSessionError::PhysicalMemory)?;
    telemetry.allocator_pool_checkpoint = Some(checkpoint);
    telemetry.physical_memory_inputs = inputs;
    Ok(())
}

pub(super) fn policy_inputs(
    operational_safety_reserve_bytes: Option<NonZeroUsize>,
) -> Result<PhysicalMemoryInputs, ResidentSessionError> {
    match operational_safety_reserve_bytes {
        Some(bytes) => PhysicalMemoryInputs::default()
            .with_operational_safety_reserve(bytes)
            .map_err(ResidentSessionError::PhysicalMemory),
        None => Ok(PhysicalMemoryInputs::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_ledger::PhysicalAllocationId;

    #[test]
    fn operational_reserve_policy_is_retained_without_native_measurements() {
        let reserve = NonZeroUsize::new(4096).unwrap();
        let inputs = policy_inputs(Some(reserve)).unwrap();
        assert_eq!(
            inputs.get(PhysicalAllocationId::OperationalSafetyReserve),
            Some(4096)
        );
        assert_eq!(inputs.missing_allocation_ids().len(), 5);
        assert_eq!(
            policy_inputs(None).unwrap().missing_allocation_ids().len(),
            6
        );
    }
}
