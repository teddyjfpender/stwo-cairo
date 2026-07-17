//! Exact semantic lowering for the proof-wide multiplicity clear.
//!
//! The clear is one ordered, write-only operation over the canonical
//! multiplicity destination list. Each complete zero write produces the
//! slab's catalog-first semantic value; later additive owners, not this
//! initializer, advance it. The pointer table is relocation metadata; the
//! dereferenced length vector is one immutable semantic value.

use std::collections::BTreeSet;

use stwo_backend_cuda::{WitnessFeedClearContract, WitnessFeedClearLinkedContract};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::*;
use crate::arena_plan::{ArenaBinding, BufferPurpose, ProofArenaPlan};
use crate::compiled_proof::{
    AotInvocation, EffectBindingId, EffectContract, ElementRange, StaticCudaWrapperAuthority,
    StaticCudaWrapperId, ValueVersion,
};

mod projection;
mod semantic;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MultiplicityClearRelocations {
    pub(super) destination_pointers: ArenaBinding,
    pub(super) destination_lengths: ArenaBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MultiplicityClearDestinationBinding {
    pub(super) ordinal: u32,
    pub(super) name: &'static str,
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
    /// Catalog-first output of this complete zero write.
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredMultiplicityClear {
    pub(super) contract: WitnessFeedClearContract,
    pub(super) relocations: MultiplicityClearRelocations,
    pub(super) destinations: Vec<MultiplicityClearDestinationBinding>,
    pub(super) lengths_value: ValueVersion,
    pub(super) lengths_binding: EffectBindingId,
    pub(super) invocation: AotInvocation,
    pub(super) effect: EffectContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LinkedMultiplicityClearExecution {
    pub(super) wrapper: StaticCudaWrapperAuthority,
}

/// Lower the one clear operation transactionally.
///
/// A completed stage is idempotent. A partially allocated destination set or
/// any transitioned destination is rejected.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredMultiplicityClear, InvocationShapeError> {
    let multiplicity = arena
        .multiplicity()
        .ok_or(InvocationShapeError::MultiplicityNeedsSemanticVersions)?;
    let contract = WitnessFeedClearContract::compile(&multiplicity.clear_requirements)
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    require_contract_identity(&contract)?;

    let catalog = BaseProducerCatalog::compile(arena)?;
    let relocations = exact_relocations(arena, &catalog, multiplicity, &contract)?;
    let exact = exact_destinations(arena, &catalog, multiplicity, &contract)?;

    let mut next_values = values.clone();
    let versions = semantic::exact_clear_outputs(
        &mut next_values,
        exact.iter().map(|destination| destination.value.id),
    )?;
    let destinations = exact
        .into_iter()
        .zip(versions)
        .enumerate()
        .map(|(index, (destination, version))| {
            Ok(MultiplicityClearDestinationBinding {
                ordinal: to_u32(index)?,
                name: destination.name,
                arena: destination.arena,
                value: destination.value.id,
                elements: ElementRange::new(0, destination.value.words)
                    .ok_or(InvocationShapeError::InvalidStructuredAbi)?,
                binding: EffectBindingId(to_u32(index)?),
                version,
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    let lengths_value =
        next_values.register_fixed_u32(contract.effect_geometry().destination_lengths.to_vec())?;
    let lengths_binding = EffectBindingId(to_u32(destinations.len())?);
    let effect = semantic::effect(&contract, &destinations, lengths_value, lengths_binding)?;
    let invocation =
        semantic::invocation(&contract, &destinations, lengths_value, lengths_binding)?;
    semantic::validate_exact_bindings(&invocation, &effect, lengths_value, lengths_binding)?;
    validate_value_classes(&next_values, &destinations, lengths_value)?;

    *values = next_values;
    Ok(LoweredMultiplicityClear {
        contract,
        relocations,
        destinations,
        lengths_value,
        lengths_binding,
        invocation,
        effect,
    })
}

pub(super) fn validate(
    arena: &ProofArenaPlan,
    values: &adapter::SemanticValueMap,
    supplied: &LoweredMultiplicityClear,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = values.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    if exact == *supplied && &exact_values == values {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

pub(super) fn project_static_wrapper(
    id: StaticCudaWrapperId,
    linked: &WitnessFeedClearLinkedContract,
    lowered: &LoweredMultiplicityClear,
) -> Result<LinkedMultiplicityClearExecution, InvocationShapeError> {
    projection::linked(id, linked, lowered)
}

#[derive(Clone, Copy)]
struct ExactDestination<'a> {
    name: &'static str,
    value: &'a BaseCatalogValue,
    arena: ArenaBinding,
}

fn exact_destinations<'a>(
    arena: &ProofArenaPlan,
    catalog: &'a BaseProducerCatalog,
    multiplicity: &crate::arena_plan::PlannedGraphAMultiplicityWorkspace,
    contract: &WitnessFeedClearContract,
) -> Result<Vec<ExactDestination<'a>>, InvocationShapeError> {
    let geometry = contract.effect_geometry();
    if multiplicity.multiplicities.len() != geometry.destinations.len()
        || multiplicity.multiplicities.len() != geometry.destination_lengths.len()
        || multiplicity.multiplicities.len() != contract.requirements().destination_words.len()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let mut logicals = BTreeSet::new();
    let mut physicals = BTreeSet::new();
    multiplicity
        .multiplicities
        .iter()
        .zip(&geometry.destinations)
        .enumerate()
        .map(|(index, (&(name, binding), effect))| {
            let value = catalog.value(ArenaCatalogValueId(binding.logical.0))?;
            let expected_words = contract.requirements().destination_words[index];
            if effect.destination_ordinal as usize != index
                || effect.write_start_words != 0
                || effect.write_len_words != expected_words
                || geometry.destination_lengths[index] as usize != expected_words
                || binding.len_words != expected_words
                || value.logical != binding.logical
                || value.physical != binding.physical
                || value.words != expected_words
                || arena.binding(value.logical) != Some(binding)
                || !logicals.insert(value.logical)
                || !physicals.insert(value.physical)
            {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            validate_destination_role(value, name, index)?;
            Ok(ExactDestination {
                name,
                value,
                arena: binding,
            })
        })
        .collect()
}

fn validate_destination_role(
    value: &BaseCatalogValue,
    name: &'static str,
    clear_ordinal: usize,
) -> Result<(), InvocationShapeError> {
    let valid = match value.purpose {
        BufferPurpose::FixedMultiplicity => {
            value.component == Some(name)
                && value.part == Some(TracePartId::Main)
                && value.ordinal == 0
        }
        BufferPurpose::RuntimeMultiplicity => {
            let (component, part) = match name {
                "memory_address_to_id" => (Some("memory_address_to_id"), Some(TracePartId::Main)),
                "memory_id_to_big" | "memory_id_to_big#small" => (Some("memory_id_to_big"), None),
                _ => return Err(InvocationShapeError::InvalidScheduledProducerBinding),
            };
            value.component == component
                && value.part == part
                && value.ordinal as usize == clear_ordinal
        }
        _ => false,
    };
    valid
        .then_some(())
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)
}

fn exact_relocations(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    multiplicity: &crate::arena_plan::PlannedGraphAMultiplicityWorkspace,
    contract: &WitnessFeedClearContract,
) -> Result<MultiplicityClearRelocations, InvocationShapeError> {
    let destination_pointers = exact_global(
        arena,
        catalog,
        multiplicity.clear_slots.destination_pointers,
        BufferPurpose::FixedMultiplicityClearPointers,
        contract.requirements().destination_pointer_words,
    )?;
    let destination_lengths = exact_global(
        arena,
        catalog,
        multiplicity.clear_slots.destination_lengths,
        BufferPurpose::FixedMultiplicityClearLengths,
        contract.requirements().destination_length_words,
    )?;
    if destination_pointers.physical == destination_lengths.physical
        || multiplicity.multiplicities.iter().any(|(_, destination)| {
            [destination_pointers.physical, destination_lengths.physical]
                .contains(&destination.physical)
        })
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(MultiplicityClearRelocations {
        destination_pointers,
        destination_lengths,
    })
}

fn exact_global(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    physical: ArenaSlotId,
    purpose: BufferPurpose,
    words: usize,
) -> Result<ArenaBinding, InvocationShapeError> {
    let mut matches = catalog.values.iter().filter(|value| {
        value.physical == physical
            && value.purpose == purpose
            && value.component.is_none()
            && value.part.is_none()
            && value.ordinal == 0
            && value.words == words
    });
    let value = matches
        .next()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let binding = ArenaBinding {
        logical: value.logical,
        physical: value.physical,
        len_words: value.words,
    };
    if arena.binding(value.logical) == Some(binding) {
        Ok(binding)
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

fn validate_value_classes(
    values: &adapter::SemanticValueMap,
    destinations: &[MultiplicityClearDestinationBinding],
    lengths: ValueVersion,
) -> Result<(), InvocationShapeError> {
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    if !fixed.contains(&lengths)
        || catalog_first.contains(&lengths)
        || transitions.contains(&lengths)
        || destinations.iter().any(|destination| {
            !catalog_first.contains(&destination.version)
                || transitions.contains(&destination.version)
                || fixed.contains(&destination.version)
                || values.version(destination.value) != Ok(destination.version)
                || values.versions_for(destination.value).collect::<Vec<_>>()
                    != [destination.version]
        })
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(())
}

fn require_contract_identity(
    contract: &WitnessFeedClearContract,
) -> Result<(), InvocationShapeError> {
    if [
        contract.static_source_identity(),
        contract.wrapper_source_identity(),
        contract.source_identity(),
        contract.requirements_identity(),
        contract.abi_identity(),
        contract.effect_identity(),
        contract.launch_identity(),
        contract.identity(),
    ]
    .contains(&[0; 32])
    {
        Err(InvocationShapeError::InvalidStructuredAbi)
    } else {
        Ok(())
    }
}

fn to_u32(value: usize) -> Result<u32, InvocationShapeError> {
    u32::try_from(value).map_err(|_| InvocationShapeError::SizeOverflow)
}
