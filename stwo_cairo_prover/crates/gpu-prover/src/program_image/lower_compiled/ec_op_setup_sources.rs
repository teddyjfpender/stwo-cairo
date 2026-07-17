//! Exact source ownership at the native EC-op Base boundary.
//!
//! This closes the storage/SSA identity of the one ingested segment scalar and
//! the four zero-before-accumulation multiplicity slabs. It intentionally does
//! not emit origins or operations: `CompiledProof` still lacks typed execution
//! authorities for host ingest, the batched clear graph, public-memory seed,
//! and generic witness-feed graphs.

use std::collections::BTreeMap;

use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::adapter::SemanticValueMap;
use super::producer_prefix::SemanticBaseProducer;
use super::*;
use crate::arena_plan::{
    ArenaBinding, BufferPurpose, PlannedGraphAMultiplicityWorkspace,
    PlannedRecordedMultiplicityFeedGraph, ProofArenaPlan,
};
use crate::compiled_proof::{EffectBindingId, InPlaceAliasAuthority, ValueVersion};
use crate::multiplicity_pipeline::blake_g_fused_feed_binding;
use crate::resident_runtime::producer_schedule::{
    BaseProducerSchedule, BaseProducerStep, WitnessProducerKind,
};

pub(super) const EC_OP_SETUP_SOURCE_COUNT: usize = 5;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum EcOpMultiplicityRole {
    AddressCounts,
    BigCounts,
    SmallCounts,
    RangeCheck8Counts,
}

impl EcOpMultiplicityRole {
    const fn destination(self) -> &'static str {
        match self {
            Self::AddressCounts => "memory_address_to_id",
            Self::BigCounts => "memory_id_to_big",
            Self::SmallCounts => "memory_id_to_big#small",
            Self::RangeCheck8Counts => "range_check_8",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EcOpMultiplicityAdditiveOwner {
    PublicMemorySeed,
    WitnessFeed(&'static str),
    NativeEcOp,
}

/// The exact executable-origin layer still absent from `CompiledProof`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MissingEcOpSetupOriginAuthority {
    SegmentStartExternalInput,
    BatchedMultiplicityClear,
    PublicMemorySeed,
    GenericWitnessFeeds,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct EcOpSegmentStartSource {
    pub(super) value: ArenaCatalogRange,
    pub(super) source: ValueVersion,
    pub(super) native_binding: EffectBindingId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct EcOpMultiplicitySource {
    pub(super) role: EcOpMultiplicityRole,
    pub(super) value: ArenaCatalogRange,
    pub(super) clear_destination_index: usize,
    pub(super) source: ValueVersion,
    pub(super) native_destination: ValueVersion,
    pub(super) native_binding: EffectBindingId,
    pub(super) native_alias: InPlaceAliasAuthority,
    /// Ordered ownership classes: optional seed, scheduled witness feeds, then
    /// the native EC-op atomic transition represented by `native_destination`.
    pub(super) additive_owners: Vec<EcOpMultiplicityAdditiveOwner>,
}

/// Role-complete source receipt. The five values are identified without
/// relying on generated `ValueVersion`, catalog, or logical-buffer numbers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct EcOpSetupSourceAuthority {
    pub(super) segment_start: EcOpSegmentStartSource,
    pub(super) multiplicities: [EcOpMultiplicitySource; 4],
}

impl EcOpSetupSourceAuthority {
    pub(super) const fn source_count(&self) -> usize {
        1 + self.multiplicities.len()
    }

    /// Every mapped value still needs an origin-emitting adapter. The mapping
    /// is complete; executable ingest/clear/seed/feed authority is not.
    pub(super) const fn unresolved_origin_count(&self) -> usize {
        self.source_count()
    }

    pub(super) fn missing_origin_authorities(&self) -> Vec<MissingEcOpSetupOriginAuthority> {
        let has_seed = self.multiplicities.iter().any(|source| {
            source
                .additive_owners
                .contains(&EcOpMultiplicityAdditiveOwner::PublicMemorySeed)
        });
        let has_feeds = self.multiplicities.iter().any(|source| {
            source
                .additive_owners
                .iter()
                .any(|owner| matches!(owner, EcOpMultiplicityAdditiveOwner::WitnessFeed(_)))
        });
        [
            Some(MissingEcOpSetupOriginAuthority::SegmentStartExternalInput),
            Some(MissingEcOpSetupOriginAuthority::BatchedMultiplicityClear),
            has_seed.then_some(MissingEcOpSetupOriginAuthority::PublicMemorySeed),
            has_feeds.then_some(MissingEcOpSetupOriginAuthority::GenericWitnessFeeds),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

/// Bind the setup sources to the exact lowered EC invocation and arena plan.
///
/// This is a projection of existing authorities, not a second execution plan:
/// the resident Base schedule owns order, the multiplicity workspace owns
/// clear/seed/feed routing, and the native contract owns the final atomics.
pub(super) fn bind(
    arena: &ProofArenaPlan,
    producers: &[SemanticBaseProducer],
    values: &SemanticValueMap,
) -> Result<EcOpSetupSourceAuthority, InvocationShapeError> {
    validate_setup_order(arena)?;
    let catalog = BaseProducerCatalog::compile(arena)?;
    let multiplicity = arena
        .multiplicity()
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
    let ec_op = arena
        .ec_op()
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
    let contract = unique_native_ec_op(producers)?;
    if super::ec_op_prefix::exact_effect(&contract.invocation)? != contract.effect {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }

    let (catalog_first, transitions, fixed) = values.allocation_classes();
    let segment = &contract.invocation.segment_start;
    let (Some(native_binding), Some(source)) = (segment.binding, segment.version) else {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    };
    let segment_catalog = catalog.value(segment.value.value)?;
    if segment.value.value_words != (0..1)
        || segment_catalog.physical != ec_op.slots.segment_start
        || segment_catalog.component != Some("ec_op_builtin")
        || segment_catalog.part != Some(TracePartId::Main)
        || segment_catalog.purpose != BufferPurpose::EcOpSegmentStart
        || segment_catalog.ordinal != 0
        || segment_catalog.words != 1
        || values.version(segment.value.value)? != source
        || !catalog_first.contains(&source)
        || transitions.contains(&source)
        || fixed.contains(&source)
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }

    let feed_owners = feed_owners(multiplicity)?;
    let specs = [
        (
            EcOpMultiplicityRole::AddressCounts,
            ec_op.slots.address_counts,
            ec_op.requirements.address_count_words,
            BufferPurpose::RuntimeMultiplicity,
            Some("memory_address_to_id"),
            Some(TracePartId::Main),
        ),
        (
            EcOpMultiplicityRole::BigCounts,
            ec_op.slots.big_counts,
            ec_op.requirements.big_count_words,
            BufferPurpose::RuntimeMultiplicity,
            Some("memory_id_to_big"),
            None,
        ),
        (
            EcOpMultiplicityRole::SmallCounts,
            ec_op.slots.small_counts,
            ec_op.requirements.small_count_words,
            BufferPurpose::RuntimeMultiplicity,
            Some("memory_id_to_big"),
            None,
        ),
        (
            EcOpMultiplicityRole::RangeCheck8Counts,
            ec_op.slots.range_check_8_counts,
            ec_op.requirements.range_check_8_count_words,
            BufferPurpose::FixedMultiplicity,
            Some("range_check_8"),
            Some(TracePartId::Main),
        ),
    ];
    if contract.invocation.multiplicities.len() != specs.len() {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }

    let mut bound = Vec::with_capacity(specs.len());
    for (atomic, (role, slot, words, purpose, component, part)) in
        contract.invocation.multiplicities.iter().zip(specs)
    {
        let catalog_value = catalog.value(atomic.value.value)?;
        let (clear_destination_index, clear_binding) =
            exact_multiplicity(multiplicity, role.destination())?;
        if atomic.value.value_words != (0..words)
            || catalog_value.physical != slot
            || catalog_value.logical != clear_binding.logical
            || catalog_value.component != component
            || catalog_value.part != part
            || catalog_value.purpose != purpose
            || catalog_value.words != words
            || clear_binding.physical != slot
            || clear_binding.len_words != words
            || multiplicity
                .clear_requirements
                .destination_words
                .get(clear_destination_index)
                != Some(&words)
            || !catalog_first.contains(&atomic.source)
            || transitions.contains(&atomic.source)
            || fixed.contains(&atomic.source)
            || !transitions.contains(&atomic.destination)
            || values.version(atomic.value.value)? != atomic.destination
        {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        let mut additive_owners = feed_owners
            .get(role.destination())
            .cloned()
            .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
        additive_owners.push(EcOpMultiplicityAdditiveOwner::NativeEcOp);
        bound.push(EcOpMultiplicitySource {
            role,
            value: atomic.value.clone(),
            clear_destination_index,
            source: atomic.source,
            native_destination: atomic.destination,
            native_binding: atomic.binding,
            native_alias: atomic.alias,
            additive_owners,
        });
    }
    let multiplicities = bound
        .try_into()
        .map_err(|_| InvocationShapeError::InvalidNativeEcOpBinding)?;
    let authority = EcOpSetupSourceAuthority {
        segment_start: EcOpSegmentStartSource {
            value: segment.value.clone(),
            source,
            native_binding,
        },
        multiplicities,
    };
    if authority.source_count() != EC_OP_SETUP_SOURCE_COUNT {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    Ok(authority)
}

fn unique_native_ec_op(
    producers: &[SemanticBaseProducer],
) -> Result<&super::ec_op_prefix::LoweredNativeEcOpContract, InvocationShapeError> {
    let mut matches = producers.iter().filter_map(|producer| match producer {
        SemanticBaseProducer::NativeEcOp {
            producer, contract, ..
        } if producer.component == "ec_op_builtin"
            // Native producers are schedule nodes, not recorded trace parts.
            // The EC contract separately binds its arena values to Main.
            && producer.part.is_none()
            && producer.kind == WitnessProducerKind::NativeEcOp =>
        {
            Some(contract)
        }
        _ => None,
    });
    let contract = matches
        .next()
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    Ok(contract)
}

fn validate_setup_order(arena: &ProofArenaPlan) -> Result<(), InvocationShapeError> {
    let schedule = BaseProducerSchedule::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    let steps = schedule.steps();
    let seed = arena
        .multiplicity()
        .is_some_and(|multiplicity| multiplicity.public_memory_seed.is_some());
    let witness_index = 2 + usize::from(seed);
    if steps.first() != Some(&BaseProducerStep::ExecutionTables)
        || steps.get(1) != Some(&BaseProducerStep::MultiplicityClear)
        || seed && steps.get(2) != Some(&BaseProducerStep::PublicMemorySeed)
        || !matches!(
            steps.get(witness_index),
            Some(BaseProducerStep::WitnessDag { .. })
        )
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(())
}

fn exact_multiplicity(
    multiplicity: &PlannedGraphAMultiplicityWorkspace,
    name: &'static str,
) -> Result<(usize, ArenaBinding), InvocationShapeError> {
    let mut matches = multiplicity
        .multiplicities
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, (candidate, _))| *candidate == name);
    let exact = matches
        .next()
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    let (index, (_, binding)) = exact;
    Ok((index, binding))
}

fn feed_owners(
    multiplicity: &PlannedGraphAMultiplicityWorkspace,
) -> Result<BTreeMap<&'static str, Vec<EcOpMultiplicityAdditiveOwner>>, InvocationShapeError> {
    if multiplicity.multiplicities.len() != multiplicity.clear_requirements.destination_words.len()
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    let mut owners = BTreeMap::new();
    for &(name, _) in &multiplicity.multiplicities {
        if owners.insert(name, Vec::new()).is_some() {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
    }

    if let Some(seed) = &multiplicity.public_memory_seed {
        if seed.plan.destination_components
            != [
                "memory_address_to_id_state",
                "memory_id_to_big_state",
                "memory_id_to_big_state#small",
            ]
        {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        record_destinations(
            multiplicity,
            &mut owners,
            &[
                "memory_address_to_id",
                "memory_id_to_big",
                "memory_id_to_big#small",
            ],
            &seed.slots.multiplicity_destinations,
            &seed.plan.requirements.multiplicity_words,
            EcOpMultiplicityAdditiveOwner::PublicMemorySeed,
        )?;
    }
    for feed in &multiplicity.feeds {
        match feed {
            PlannedRecordedMultiplicityFeedGraph::Generic(feed) => record_destinations(
                multiplicity,
                &mut owners,
                &feed.plan.destination_components,
                &feed.slots.multiplicity_destinations,
                &feed.plan.requirements.multiplicity_words,
                EcOpMultiplicityAdditiveOwner::WitnessFeed(feed.plan.producer),
            )?,
            PlannedRecordedMultiplicityFeedGraph::BlakeGFused {
                plan,
                multiplicity_destinations,
                ..
            } => {
                let binding = blake_g_fused_feed_binding(plan)
                    .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
                let names = binding
                    .destination_indices
                    .map(|index| plan.destination_components[index]);
                let words = binding
                    .destination_indices
                    .map(|index| plan.requirements.multiplicity_words[index]);
                let slots = multiplicity_destinations.map(|binding| binding.physical);
                record_destinations(
                    multiplicity,
                    &mut owners,
                    &names,
                    &slots,
                    &words,
                    EcOpMultiplicityAdditiveOwner::WitnessFeed(plan.producer),
                )?;
            }
        }
    }
    Ok(owners)
}

fn record_destinations(
    multiplicity: &PlannedGraphAMultiplicityWorkspace,
    owners: &mut BTreeMap<&'static str, Vec<EcOpMultiplicityAdditiveOwner>>,
    names: &[&'static str],
    slots: &[stwo_backend_cuda::ArenaSlotId],
    words: &[usize],
    owner: EcOpMultiplicityAdditiveOwner,
) -> Result<(), InvocationShapeError> {
    if names.len() != slots.len() || names.len() != words.len() {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    for ((&name, &slot), &expected_words) in names.iter().zip(slots).zip(words) {
        let (_, binding) = exact_multiplicity(multiplicity, name)?;
        let destinations = owners
            .get_mut(name)
            .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
        if binding.physical != slot
            || binding.len_words != expected_words
            || destinations.contains(&owner)
        {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        destinations.push(owner);
    }
    Ok(())
}

#[cfg(test)]
#[path = "ec_op_setup_sources_tests.rs"]
mod tests;
