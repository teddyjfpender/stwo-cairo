//! Semantic lowering for generic multiplicity feeds.
//!
//! The coordinator owns schedule order and canonical LUT construction. This
//! module proves one recorded writer/feed edge or the public-memory seed:
//! pointer tables remain relocation metadata, descriptors are one immutable
//! constant, LUTs are catalog-first immutable values, and every count slab is
//! one full-range in-place atomic SSA transition.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::{WitnessFeedContract, WitnessFeedLaunchMode, WitnessFeedLinkedContract};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::producer_prefix::LoweredRecordedWitnessProducer;
use super::*;
use crate::arena_plan::{
    ArenaBinding, PlannedGenericRecordedMultiplicityFeedGraph, ProofArenaPlan,
};
use crate::compiled_proof::{
    AotInvocation, EffectBindingId, EffectContract, ElementRange, InPlaceAliasAuthority,
    StaticCudaWrapperAuthority, StaticCudaWrapperId, ValueVersion,
};

mod bindings;
mod projection;
mod semantic;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MultiplicityFeedOwner {
    Recorded {
        component: &'static str,
        part: TracePartId,
    },
    PublicMemorySeed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MultiplicityFeedRelocations {
    pub(super) descriptor_workspace: ArenaBinding,
    pub(super) lut_pointers: ArenaBinding,
    pub(super) multiplicity_pointers: ArenaBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MultiplicityFeedValueBinding {
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MultiplicityFeedLutBinding {
    pub(super) family: &'static str,
    pub(super) input: MultiplicityFeedValueBinding,
    pub(super) content_identity: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MultiplicityFeedTransition {
    pub(super) ordinal: u32,
    pub(super) name: &'static str,
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
    pub(super) source: ValueVersion,
    pub(super) destination: ValueVersion,
    pub(super) alias: InPlaceAliasAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredMultiplicityFeed {
    pub(super) owner: MultiplicityFeedOwner,
    pub(super) contract: WitnessFeedContract,
    pub(super) relocations: MultiplicityFeedRelocations,
    pub(super) source: MultiplicityFeedValueBinding,
    pub(super) descriptor_value: ValueVersion,
    pub(super) descriptor_binding: EffectBindingId,
    pub(super) luts: Vec<MultiplicityFeedLutBinding>,
    pub(super) destinations: Vec<MultiplicityFeedTransition>,
    pub(super) invocation: AotInvocation,
    pub(super) effect: EffectContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LinkedMultiplicityFeedExecution {
    pub(super) wrapper: StaticCudaWrapperAuthority,
}

pub(super) fn project_static_wrapper(
    id: StaticCudaWrapperId,
    linked: &WitnessFeedLinkedContract,
    lowered: &LoweredMultiplicityFeed,
) -> Result<LinkedMultiplicityFeedExecution, InvocationShapeError> {
    projection::linked(id, linked, lowered)
}

/// Lower the feed immediately owned by one recorded writer.
///
/// The source must be a current full-range `Write` output of `writer`; it is
/// never allocated here. Publication is transactional.
pub(super) fn lower_recorded(
    arena: &ProofArenaPlan,
    plan: &PlannedGenericRecordedMultiplicityFeedGraph,
    writer: &LoweredRecordedWitnessProducer,
    canonical_luts: &BTreeMap<&'static str, Vec<u32>>,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredMultiplicityFeed, InvocationShapeError> {
    let owner = MultiplicityFeedOwner::Recorded {
        component: writer.producer.component,
        part: writer
            .producer
            .part
            .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?,
    };
    if owner
        != (MultiplicityFeedOwner::Recorded {
            component: plan.plan.producer,
            part: TracePartId::Main,
        })
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    bindings::require_recorded_membership(arena, plan)?;
    let luts = bindings::ordered_lut_words(plan, canonical_luts)?;
    lower(arena, plan, owner, Some(writer), &luts, values)
}

/// Lower the optional claim-bound public-memory multiplicity seed.
///
/// This is the only feed allowed to allocate its source catalog value. It has
/// no LUTs and must target exactly the three runtime memory slabs.
pub(super) fn lower_public_memory_seed(
    arena: &ProofArenaPlan,
    plan: &PlannedGenericRecordedMultiplicityFeedGraph,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredMultiplicityFeed, InvocationShapeError> {
    bindings::require_public_seed_membership(arena, plan)?;
    lower(
        arena,
        plan,
        MultiplicityFeedOwner::PublicMemorySeed,
        None,
        &[],
        values,
    )
}

fn lower(
    arena: &ProofArenaPlan,
    plan: &PlannedGenericRecordedMultiplicityFeedGraph,
    owner: MultiplicityFeedOwner,
    writer: Option<&LoweredRecordedWitnessProducer>,
    lut_words: &[Vec<u32>],
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredMultiplicityFeed, InvocationShapeError> {
    let contract = WitnessFeedContract::compile(
        &plan.plan.requirements,
        &plan.plan.descriptors,
        lut_words,
        arena.protocol_identity().witness_feed_launch_mode,
    )
    .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    require_contract_identity(&contract)?;

    let catalog = BaseProducerCatalog::compile(arena)?;
    let exact = bindings::exact(arena, &catalog, plan, owner, &contract)?;
    let mut next_values = values.clone();
    let source = bindings::bind_source(
        arena,
        &exact,
        owner,
        writer,
        &mut next_values,
        EffectBindingId(0),
    )?;

    next_values.extend_ordered(exact.luts.iter().map(|lut| lut.id))?;
    let descriptor_value = next_values.register_fixed_u32(plan.plan.descriptors.clone())?;
    let descriptor_binding = EffectBindingId(1);
    let mut next_binding = 2u32;
    let luts = exact
        .luts
        .into_iter()
        .zip(contract.effect_geometry().lut_reads.iter())
        .zip(plan.plan.lut_families.iter().copied())
        .map(|((value, read), family)| {
            let binding = EffectBindingId(next_binding);
            next_binding = next_binding
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            let input = bindings::bind_immutable(
                arena,
                value,
                read.read_start_words,
                read.read_len_words,
                binding,
                &next_values,
            )?;
            Ok(MultiplicityFeedLutBinding {
                family,
                input,
                content_identity: read.content_identity,
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    bindings::require_immutable_luts(&next_values, &luts)?;

    let destinations = exact
        .destinations
        .into_iter()
        .zip(contract.effect_geometry().destinations.iter())
        .enumerate()
        .map(|(ordinal, (destination, geometry))| {
            let binding = EffectBindingId(next_binding);
            next_binding = next_binding
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            semantic::transition_destination(
                destination,
                geometry,
                ordinal,
                binding,
                &mut next_values,
            )
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    let effect = semantic::effect(
        &contract,
        &source,
        descriptor_value,
        descriptor_binding,
        &luts,
        &destinations,
    )?;
    let invocation = semantic::invocation(
        &contract,
        &source,
        descriptor_value,
        descriptor_binding,
        &luts,
        &destinations,
    )?;
    semantic::validate_exact_bindings(
        &invocation,
        &effect,
        descriptor_value,
        descriptor_binding,
        luts.is_empty(),
    )?;
    let lowered = LoweredMultiplicityFeed {
        owner,
        contract,
        relocations: exact.relocations,
        source,
        descriptor_value,
        descriptor_binding,
        luts,
        destinations,
        invocation,
        effect,
    };
    validate_lowered(&lowered)?;
    *values = next_values;
    Ok(lowered)
}

pub(super) fn validate_lowered(
    lowered: &LoweredMultiplicityFeed,
) -> Result<(), InvocationShapeError> {
    lowered
        .contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    require_contract_identity(&lowered.contract)?;
    semantic::validate_geometry(lowered)?;
    let effect = semantic::effect(
        &lowered.contract,
        &lowered.source,
        lowered.descriptor_value,
        lowered.descriptor_binding,
        &lowered.luts,
        &lowered.destinations,
    )?;
    let invocation = semantic::invocation(
        &lowered.contract,
        &lowered.source,
        lowered.descriptor_value,
        lowered.descriptor_binding,
        &lowered.luts,
        &lowered.destinations,
    )?;
    semantic::validate_exact_bindings(
        &invocation,
        &effect,
        lowered.descriptor_value,
        lowered.descriptor_binding,
        lowered.luts.is_empty(),
    )?;
    if effect == lowered.effect && invocation == lowered.invocation {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidAdapterEffect)
    }
}

fn require_contract_identity(contract: &WitnessFeedContract) -> Result<(), InvocationShapeError> {
    if contract.launch_mode()
        != match contract.abi() {
            stwo_backend_cuda::WitnessFeedAbi::GlobalAtomicsV1 => {
                WitnessFeedLaunchMode::GlobalAtomics
            }
            stwo_backend_cuda::WitnessFeedAbi::PrivatizedV1 => WitnessFeedLaunchMode::Privatized,
        }
        || [
            contract.static_source_identity(),
            contract.wrapper_source_identity(),
            contract.source_identity(),
            contract.requirements_identity(),
            contract.descriptor_identity(),
            contract.lut_identity(),
            contract.content_identity(),
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
