//! Linked static-wrapper projection for one generic feed.

use std::collections::BTreeSet;

use super::*;
use crate::compiled_proof::{LaunchGeometry, StaticCudaLaunchIdentity};

pub(super) fn linked(
    id: StaticCudaWrapperId,
    linked: &WitnessFeedLinkedContract,
    lowered: &LoweredMultiplicityFeed,
) -> Result<LinkedMultiplicityFeedExecution, InvocationShapeError> {
    validate_lowered(lowered)?;
    validate_metadata(lowered)?;
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
    Ok(LinkedMultiplicityFeedExecution { wrapper })
}

fn validate_metadata(lowered: &LoweredMultiplicityFeed) -> Result<(), InvocationShapeError> {
    let requirements = lowered.contract.requirements();
    let relocations = lowered.relocations;
    if lowered.invocation.arguments.len() != 6
        || relocations.descriptor_workspace.len_words != requirements.descriptor_words
        || relocations.lut_pointers.len_words != requirements.lut_pointer_words
        || relocations.multiplicity_pointers.len_words != requirements.multiplicity_pointer_words
        || [
            relocations.descriptor_workspace.logical,
            relocations.lut_pointers.logical,
            relocations.multiplicity_pointers.logical,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .len()
            != 3
        || [
            relocations.descriptor_workspace.physical,
            relocations.lut_pointers.physical,
            relocations.multiplicity_pointers.physical,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .len()
            != 3
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }

    let mut values = BTreeSet::new();
    let mut physical = BTreeSet::new();
    let mut effect_bindings = BTreeSet::new();
    let inputs = std::iter::once(&lowered.source).chain(lowered.luts.iter().map(|lut| &lut.input));
    for input in inputs {
        if input.value != ArenaCatalogValueId(input.arena.logical.0)
            || input.arena.len_words < input.elements.end
            || !values.insert(input.value)
            || !physical.insert(input.arena.physical)
            || !effect_bindings.insert(input.binding)
        {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
    }
    let mut aliases = BTreeSet::new();
    let mut names = BTreeSet::new();
    for destination in &lowered.destinations {
        if destination.value != ArenaCatalogValueId(destination.arena.logical.0)
            || destination.arena.len_words != destination.elements.len()
            || destination.elements.start != 0
            || !values.insert(destination.value)
            || !physical.insert(destination.arena.physical)
            || !effect_bindings.insert(destination.binding)
            || !aliases.insert(destination.alias.id)
            || !names.insert(destination.name)
        {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
    }
    if physical.iter().any(|slot| {
        [
            relocations.descriptor_workspace.physical,
            relocations.lut_pointers.physical,
            relocations.multiplicity_pointers.physical,
        ]
        .contains(slot)
    }) {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    match lowered.owner {
        MultiplicityFeedOwner::Recorded { component, part } => {
            if component == "__public_memory__"
                || part != TracePartId::Main
                || lowered.luts.is_empty() != requirements.lut_words.is_empty()
            {
                return Err(InvocationShapeError::InvalidProductionBaseAuthority);
            }
        }
        MultiplicityFeedOwner::PublicMemorySeed => {
            let names = lowered
                .destinations
                .iter()
                .map(|destination| destination.name)
                .collect::<Vec<_>>();
            if !lowered.luts.is_empty()
                || names
                    != [
                        "memory_address_to_id",
                        "memory_id_to_big",
                        "memory_id_to_big#small",
                    ]
            {
                return Err(InvocationShapeError::InvalidProductionBaseAuthority);
            }
        }
    }
    Ok(())
}

fn require_receipt(
    linked: &WitnessFeedLinkedContract,
    contract: &WitnessFeedContract,
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
