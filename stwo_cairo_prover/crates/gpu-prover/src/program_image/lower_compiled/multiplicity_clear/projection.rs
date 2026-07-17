//! Linked static-wrapper projection for the exact clear.

use std::collections::BTreeSet;

use super::*;
use crate::compiled_proof::{LaunchGeometry, StaticCudaLaunchIdentity};

pub(super) fn linked(
    id: StaticCudaWrapperId,
    linked: &WitnessFeedClearLinkedContract,
    lowered: &LoweredMultiplicityClear,
) -> Result<LinkedMultiplicityClearExecution, InvocationShapeError> {
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
        lowered.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    Ok(LinkedMultiplicityClearExecution { wrapper })
}

pub(super) fn validate_lowered(
    lowered: &LoweredMultiplicityClear,
) -> Result<(), InvocationShapeError> {
    lowered
        .contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    require_contract_identity(&lowered.contract)?;
    validate_metadata(lowered)?;
    let effect = semantic::effect(
        &lowered.contract,
        &lowered.destinations,
        lowered.lengths_value,
        lowered.lengths_binding,
    )?;
    let invocation = semantic::invocation(
        &lowered.contract,
        &lowered.destinations,
        lowered.lengths_value,
        lowered.lengths_binding,
    )?;
    semantic::validate_exact_bindings(
        &invocation,
        &effect,
        lowered.lengths_value,
        lowered.lengths_binding,
    )?;
    if effect == lowered.effect && invocation == lowered.invocation {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidProductionBaseAuthority)
    }
}

fn validate_metadata(lowered: &LoweredMultiplicityClear) -> Result<(), InvocationShapeError> {
    let requirements = lowered.contract.requirements();
    let relocations = lowered.relocations;
    if relocations.destination_pointers.len_words != requirements.destination_pointer_words
        || relocations.destination_lengths.len_words != requirements.destination_length_words
        || relocations.destination_pointers.logical == relocations.destination_lengths.logical
        || relocations.destination_pointers.physical == relocations.destination_lengths.physical
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    let mut names = BTreeSet::new();
    let mut logicals = BTreeSet::new();
    let mut physicals = BTreeSet::new();
    let mut values = BTreeSet::new();
    let mut versions = BTreeSet::new();
    for destination in &lowered.destinations {
        if destination.value != ArenaCatalogValueId(destination.arena.logical.0)
            || destination.arena.len_words != destination.elements.len()
            || !names.insert(destination.name)
            || !logicals.insert(destination.arena.logical)
            || !physicals.insert(destination.arena.physical)
            || !values.insert(destination.value)
            || !versions.insert(destination.version)
            || [
                relocations.destination_pointers.physical,
                relocations.destination_lengths.physical,
            ]
            .contains(&destination.arena.physical)
        {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
    }
    Ok(())
}

fn require_receipt(
    linked: &WitnessFeedClearLinkedContract,
    contract: &WitnessFeedClearContract,
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
