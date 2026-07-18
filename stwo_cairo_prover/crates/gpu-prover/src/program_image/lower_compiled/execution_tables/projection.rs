//! Linked static-wrapper projection for one ordered split stage.

use super::*;
use crate::compiled_proof::{LaunchGeometry, StaticCudaLaunchIdentity};

pub(super) fn linked(
    id: StaticCudaWrapperId,
    linked: &ExecutionTablesLinkedContract,
    lowered: &LoweredExecutionTables,
    stage: ExecutionTablesStage,
) -> Result<LinkedExecutionTableStage, InvocationShapeError> {
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    require_receipt(linked, &lowered.contract)?;
    let lowered_stage = lowered
        .stages
        .iter()
        .find(|candidate| candidate.stage == stage)
        .ok_or(InvocationShapeError::InvalidStructuredAbi)?;
    let contract_stage = stage_contract(&lowered.contract, stage)?;
    let launch = contract_stage.launch();
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
        contract_stage.abi().entry_symbol().as_bytes().to_vec(),
        lowered.contract.abi_identity(),
        lowered.contract.effect_identity(),
        lowered.contract.identity(),
        linked.identity(),
        vec![launch],
        lowered_stage
            .invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?,
        lowered_stage.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    Ok(LinkedExecutionTableStage { wrapper })
}

fn require_receipt(
    linked: &ExecutionTablesLinkedContract,
    contract: &ExecutionTablesContract,
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
