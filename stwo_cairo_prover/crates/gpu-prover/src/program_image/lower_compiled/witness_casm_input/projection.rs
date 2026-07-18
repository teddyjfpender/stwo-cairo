//! Exact linked static-wrapper projection.

use super::*;
use crate::compiled_proof::{LaunchGeometry, StaticCudaLaunchIdentity};

pub(super) fn linked(
    id: StaticCudaWrapperId,
    linked: &WitnessCasmInputLinkedContract,
    lowered: &LoweredWitnessCasmInput,
) -> Result<LinkedWitnessCasmInputExecution, InvocationShapeError> {
    validate_lowered(lowered)?;
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    require_receipt(linked, &lowered.contract)?;
    let launch = lowered.contract.launch();
    let launch = StaticCudaLaunchIdentity::new(
        launch.symbol().as_bytes().to_vec(),
        LaunchGeometry {
            grid: launch.grid,
            block: launch.block,
            cluster: launch.cluster,
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
    Ok(LinkedWitnessCasmInputExecution { wrapper })
}

pub(super) fn validate_lowered(
    lowered: &LoweredWitnessCasmInput,
) -> Result<(), InvocationShapeError> {
    let effect = semantic::effect(&lowered.staging, &lowered.outputs)?;
    let invocation = semantic::invocation(&lowered.contract, &lowered.staging, &lowered.outputs)?;
    semantic::validate_exact_bindings(&invocation, &effect)?;
    if effect == lowered.effect && invocation == lowered.invocation {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidProductionBaseAuthority)
    }
}

fn require_receipt(
    linked: &WitnessCasmInputLinkedContract,
    contract: &WitnessCasmInputContract,
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
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(())
}
