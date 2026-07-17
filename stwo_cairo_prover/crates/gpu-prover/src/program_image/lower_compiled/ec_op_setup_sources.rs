//! Exact source ownership at the native EC-op Base boundary.
//!
//! This closes the storage/SSA identity of the one ingested segment scalar and
//! the four zero-before-accumulation multiplicity slabs. Clear, seed, feed, and
//! native transitions are reconstructed in schedule order. Only the ingested
//! segment scalar still lacks an origin operation.

use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::adapter::SemanticValueMap;
use super::multiplicity_feed::{LoweredMultiplicityFeed, MultiplicityFeedOwner};
use super::producer_prefix::{BaseProducerAuthority, SemanticBaseProducer};
use super::*;
use crate::arena_plan::{
    ArenaBinding, BufferPurpose, PlannedGraphAMultiplicityWorkspace, ProofArenaPlan,
};
use crate::compiled_proof::{EffectBindingId, ElementRange, InPlaceAliasAuthority, ValueVersion};
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

/// The one executable origin still absent from the canonical Base prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MissingEcOpSetupOriginAuthority {
    SegmentStartExternalInput,
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
    /// Ordered ownership classes for the complete multiplicity lineage. The
    /// native EC-op transition appears at its producer-schedule position and
    /// may be followed by later witness feeds.
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

    /// Clear, seed, and feed origins are canonical operations. Only the
    /// statement-varying segment scalar still needs an external-input origin.
    pub(super) const fn unresolved_origin_count(&self) -> usize {
        1
    }

    pub(super) fn missing_origin_authorities(&self) -> Vec<MissingEcOpSetupOriginAuthority> {
        vec![MissingEcOpSetupOriginAuthority::SegmentStartExternalInput]
    }
}

/// Bind the setup sources to the exact lowered EC invocation and arena plan.
///
/// This is a projection of existing authorities, not a second execution plan:
/// the resident Base schedule owns order, the multiplicity workspace owns
/// clear/seed/feed routing, and the native contract owns the final atomics.
pub(super) fn bind(
    arena: &ProofArenaPlan,
    authority: &BaseProducerAuthority,
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
    let contract = unique_native_ec_op(&authority.producers)?;
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
        {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        let clear = authority
            .multiplicity
            .clear
            .destinations
            .get(clear_destination_index)
            .filter(|destination| {
                destination.name == role.destination()
                    && destination.value == atomic.value.value
                    && destination.arena == clear_binding
            })
            .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
        let additive_owners = exact_multiplicity_lineage(
            authority,
            values,
            clear,
            atomic,
            &catalog_first,
            &transitions,
            &fixed,
        )?;
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

fn exact_multiplicity_lineage(
    authority: &BaseProducerAuthority,
    values: &SemanticValueMap,
    clear: &super::multiplicity_clear::MultiplicityClearDestinationBinding,
    native: &super::ec_op_prefix::NativeEcOpAtomicBinding,
    catalog_first: &std::collections::BTreeSet<ValueVersion>,
    transitions: &std::collections::BTreeSet<ValueVersion>,
    fixed: &std::collections::BTreeSet<ValueVersion>,
) -> Result<Vec<EcOpMultiplicityAdditiveOwner>, InvocationShapeError> {
    if !catalog_first.contains(&clear.version)
        || transitions.contains(&clear.version)
        || fixed.contains(&clear.version)
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    let mut lineage = vec![clear.version];
    let mut owners = Vec::new();
    if let Some(seed) = &authority.multiplicity.public_memory_seed {
        if seed.owner != MultiplicityFeedOwner::PublicMemorySeed {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        append_feed_transition(
            seed,
            clear,
            &mut lineage,
            &mut owners,
            EcOpMultiplicityAdditiveOwner::PublicMemorySeed,
        )?;
    }
    if authority.multiplicity.after_producer.len() != authority.producers.len() {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    let mut native_count = 0usize;
    for (producer, feed) in authority
        .producers
        .iter()
        .zip(&authority.multiplicity.after_producer)
    {
        match producer {
            SemanticBaseProducer::Recorded(recorded) => {
                let feed = feed
                    .as_ref()
                    .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
                if feed.owner
                    != (MultiplicityFeedOwner::Recorded {
                        component: recorded.producer.component,
                        part: recorded
                            .producer
                            .part
                            .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?,
                    })
                {
                    return Err(InvocationShapeError::InvalidNativeEcOpBinding);
                }
                append_feed_transition(
                    feed,
                    clear,
                    &mut lineage,
                    &mut owners,
                    EcOpMultiplicityAdditiveOwner::WitnessFeed(recorded.producer.component),
                )?;
            }
            SemanticBaseProducer::NativeBlakeGDirect {
                producer, contract, ..
            } => {
                if feed.is_some() {
                    return Err(InvocationShapeError::InvalidNativeEcOpBinding);
                }
                let matching = contract
                    .invocation
                    .counts
                    .iter()
                    .filter(|transition| transition.value.value == clear.value)
                    .collect::<Vec<_>>();
                if matching.len() > 1 {
                    return Err(InvocationShapeError::InvalidNativeEcOpBinding);
                }
                if let Some(transition) = matching.first() {
                    append_transition(
                        clear,
                        transition.value.value,
                        element_range(&transition.value)?,
                        transition.source,
                        transition.destination,
                        &mut lineage,
                    )?;
                    owners.push(EcOpMultiplicityAdditiveOwner::WitnessFeed(
                        producer.component,
                    ));
                }
            }
            SemanticBaseProducer::NativeEcOp { contract, .. } => {
                if feed.is_some() {
                    return Err(InvocationShapeError::InvalidNativeEcOpBinding);
                }
                let matching = contract
                    .invocation
                    .multiplicities
                    .iter()
                    .filter(|transition| transition.value.value == clear.value)
                    .collect::<Vec<_>>();
                if matching.len() != 1 || matching[0] != native {
                    return Err(InvocationShapeError::InvalidNativeEcOpBinding);
                }
                append_transition(
                    clear,
                    native.value.value,
                    element_range(&native.value)?,
                    native.source,
                    native.destination,
                    &mut lineage,
                )?;
                owners.push(EcOpMultiplicityAdditiveOwner::NativeEcOp);
                native_count += 1;
            }
        }
    }
    let post = authority
        .multiplicity
        .post_witness_current
        .iter()
        .filter(|entry| entry.value == clear.value)
        .collect::<Vec<_>>();
    let native_edges = lineage
        .windows(2)
        .filter(|edge| edge[0] == native.source && edge[1] == native.destination)
        .count();
    if native_count != 1
        || owners
            .iter()
            .filter(|owner| **owner == EcOpMultiplicityAdditiveOwner::NativeEcOp)
            .count()
            != 1
        || native_edges != 1
        || lineage.len() != owners.len() + 1
        || lineage.iter().skip(1).any(|version| {
            !transitions.contains(version)
                || catalog_first.contains(version)
                || fixed.contains(version)
        })
        || values.versions_for(clear.value).collect::<Vec<_>>() != lineage
        || lineage.last().copied() != Some(values.version(clear.value)?)
        || post.len() != 1
        || post[0].ordinal != clear.ordinal
        || post[0].name != clear.name
        || lineage.last().copied() != Some(post[0].current)
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    Ok(owners)
}

fn append_feed_transition(
    feed: &LoweredMultiplicityFeed,
    clear: &super::multiplicity_clear::MultiplicityClearDestinationBinding,
    lineage: &mut Vec<ValueVersion>,
    owners: &mut Vec<EcOpMultiplicityAdditiveOwner>,
    owner: EcOpMultiplicityAdditiveOwner,
) -> Result<(), InvocationShapeError> {
    let matching = feed
        .destinations
        .iter()
        .filter(|transition| transition.value == clear.value)
        .collect::<Vec<_>>();
    if matching.len() > 1 {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    if let Some(transition) = matching.first() {
        if transition.name != clear.name {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        append_transition(
            clear,
            transition.value,
            transition.elements,
            transition.source,
            transition.destination,
            lineage,
        )?;
        owners.push(owner);
    }
    Ok(())
}

fn append_transition(
    clear: &super::multiplicity_clear::MultiplicityClearDestinationBinding,
    value: ArenaCatalogValueId,
    elements: ElementRange,
    source: ValueVersion,
    destination: ValueVersion,
    lineage: &mut Vec<ValueVersion>,
) -> Result<(), InvocationShapeError> {
    if value != clear.value
        || elements != clear.elements
        || lineage.last() != Some(&source)
        || source == destination
        || lineage.contains(&destination)
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    lineage.push(destination);
    Ok(())
}

fn element_range(value: &ArenaCatalogRange) -> Result<ElementRange, InvocationShapeError> {
    ElementRange::new(value.value_words.start, value.value_words.end)
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)
}

#[cfg(test)]
#[path = "ec_op_setup_sources_tests.rs"]
mod tests;
