//! Exact linked static-wrapper projection for one witness-input gather.

use super::*;
use crate::compiled_proof::{LaunchGeometry, StaticCudaLaunchIdentity};

pub(super) fn linked(
    arena: &ProofArenaPlan,
    values: &adapter::SemanticValueMap,
    id: StaticCudaWrapperId,
    linked: &stwo_backend_cuda::WitnessInputGatherLinkedContract,
    lowered: &LoweredWitnessInputGather,
) -> Result<LinkedWitnessInputGatherExecution, InvocationShapeError> {
    validate(arena, values, lowered)?;
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    require_receipt(linked, &lowered.contract)?;

    let launch = lowered.contract.wrapper_launch();
    let launch = StaticCudaLaunchIdentity::new(
        launch.audited_internal_kernel_symbol().as_bytes().to_vec(),
        LaunchGeometry {
            grid: launch.grid,
            block: launch.block,
            cluster: None,
            dynamic_shared_bytes: launch.dynamic_shared_bytes,
            cooperative: launch.cooperative,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    let wrapper = StaticCudaWrapperAuthority::new(
        id,
        linked.module_build_identity(),
        linked.target_sm(),
        lowered.contract.abi().entry_symbol().as_bytes().to_vec(),
        lowered.contract.abi_identity(),
        lowered.contract.effect_identity(),
        lowered.contract.identity(),
        linked.identity(),
        vec![launch],
        lowered
            .invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?,
        lowered.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    Ok(LinkedWitnessInputGatherExecution { wrapper })
}

fn require_receipt(
    linked: &stwo_backend_cuda::WitnessInputGatherLinkedContract,
    contract: &WitnessInputGatherContract,
) -> Result<(), InvocationShapeError> {
    if linked.contract_identity() != contract.identity()
        || linked.target_sm() < 10
        || [
            linked.module_build_identity(),
            linked.static_build_source_identity(),
            linked.static_build_identity(),
            linked.sm_identity(),
            linked.identity(),
        ]
        .contains(&[0; 32])
    {
        Err(InvocationShapeError::InvalidProductionBaseAuthority)
    } else {
        Ok(())
    }
}
