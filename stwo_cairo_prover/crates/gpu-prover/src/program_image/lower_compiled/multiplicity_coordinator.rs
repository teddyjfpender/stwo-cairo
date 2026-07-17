//! Schedule-ordered multiplicity SSA for the Base witness DAG.
//!
//! This coordinator is deliberately narrower than Base lowering. It owns the
//! clear, optional public-memory seed, and every witness/native count
//! transition through the end of `WitnessDag`. Memory-base and fixed-table
//! producers still follow, so the returned current versions are post-witness,
//! never final Base versions.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use stwo_backend_cuda::BlakeGDirectLutContentIdentity;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::witness::device_feed::canonical_count_lut;
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::adapter::SemanticValueMap;
use super::multiplicity_clear::{self, LoweredMultiplicityClear};
use super::multiplicity_feed::{self, LoweredMultiplicityFeed};
use super::producer_prefix::{ProducerSchedulePosition, SemanticBaseProducer};
use super::*;
use crate::arena_plan::{ArenaBinding, PlannedRecordedMultiplicityFeedGraph, ProofArenaPlan};
use crate::compiled_proof::{ElementRange, ValueVersion};
use crate::multiplicity_pipeline::blake_g_fused_feed_binding;
use crate::resident_runtime::producer_schedule::{
    BaseProducerSchedule, BaseProducerStep, WitnessProducer, WitnessProducerKind,
};

mod lineage;

#[cfg(test)]
#[path = "multiplicity_coordinator/tests.rs"]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PostWitnessMultiplicity {
    pub(super) ordinal: u32,
    pub(super) name: &'static str,
    pub(super) value: ArenaCatalogValueId,
    pub(super) current: ValueVersion,
}

/// Exact multiplicity slice through the end of the witness DAG.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredBaseMultiplicity {
    pub(super) clear: LoweredMultiplicityClear,
    pub(super) public_memory_seed: Option<LoweredMultiplicityFeed>,
    /// Aligned one-for-one with the flattened `BaseProducerSchedule`.
    /// Generic recorded writers have `Some(feed)`; native fused count writers
    /// and the native EC-op have `None`.
    pub(super) after_producer: Vec<Option<LoweredMultiplicityFeed>>,
    /// Current versions after the last witness node. Later Base producers may
    /// transition these values again.
    pub(super) post_witness_current: Vec<PostWitnessMultiplicity>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeedPlanKind {
    Generic,
    BlakeGFused,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FeedPlanRef {
    index: usize,
    kind: FeedPlanKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TransitionReceipt {
    value: ArenaCatalogValueId,
    elements: ElementRange,
    source: ValueVersion,
    destination: ValueVersion,
}

/// Stateful gate called immediately after each lowered Base witness producer.
///
/// Every mutating method operates on clones and commits only after all checks
/// pass, so a caller can also wrap the complete producer loop in one outer
/// transaction.
#[derive(Clone)]
pub(super) struct BaseMultiplicityCoordinator<'arena> {
    arena: &'arena ProofArenaPlan,
    scheduled: Vec<(ProducerSchedulePosition, WitnessProducer)>,
    next: usize,
    feeds_by_producer: BTreeMap<&'static str, FeedPlanRef>,
    clear: LoweredMultiplicityClear,
    public_memory_seed: Option<LoweredMultiplicityFeed>,
    after_producer: Vec<Option<LoweredMultiplicityFeed>>,
    transitions_after_step: Vec<Vec<TransitionReceipt>>,
    canonical_luts: Arc<BTreeMap<&'static str, Vec<u32>>>,
    used_lut_families: BTreeSet<&'static str>,
}

impl<'arena> BaseMultiplicityCoordinator<'arena> {
    /// Clear every slab, apply the optional seed, and seal the exact witness
    /// schedule before admitting its first producer.
    pub(super) fn begin(
        arena: &'arena ProofArenaPlan,
        schedule: &BaseProducerSchedule,
        variant: PreProcessedTraceVariant,
        values: &mut SemanticValueMap,
    ) -> Result<Self, InvocationShapeError> {
        let scheduled = exact_schedule(arena, schedule)?;
        let (feeds_by_producer, expected_lut_words) = exact_feed_alignment(arena, &scheduled)?;
        let canonical_luts = Arc::new(canonical_luts(variant, &expected_lut_words)?);

        let mut next_values = values.clone();
        let clear = multiplicity_clear::lower_stage(arena, &mut next_values)?;
        let public_memory_seed = arena
            .multiplicity()
            .and_then(|multiplicity| multiplicity.public_memory_seed.as_ref())
            .map(|seed| multiplicity_feed::lower_public_memory_seed(arena, seed, &mut next_values))
            .transpose()?;

        let coordinator = Self {
            arena,
            scheduled,
            next: 0,
            feeds_by_producer,
            clear,
            public_memory_seed,
            after_producer: Vec::new(),
            transitions_after_step: Vec::new(),
            canonical_luts,
            used_lut_families: BTreeSet::new(),
        };
        coordinator.validate_prefix_lineages(&next_values)?;
        *values = next_values;
        Ok(coordinator)
    }

    /// Admit exactly the next schedule node and, for a recorded writer, lower
    /// its generic feed before any later producer can be admitted.
    pub(super) fn after_producer(
        &mut self,
        producer: &SemanticBaseProducer,
        values: &mut SemanticValueMap,
    ) -> Result<(), InvocationShapeError> {
        let mut next = self.clone();
        let mut next_values = values.clone();
        next.apply_after_producer(producer, &mut next_values)?;
        *self = next;
        *values = next_values;
        Ok(())
    }

    /// Close only the witness-DAG portion of Base.
    pub(super) fn finish_witness(
        self,
        values: &SemanticValueMap,
    ) -> Result<LoweredBaseMultiplicity, InvocationShapeError> {
        if self.next != self.scheduled.len()
            || self.after_producer.len() != self.scheduled.len()
            || self.transitions_after_step.len() != self.scheduled.len()
            || !self.feeds_by_producer.is_empty()
            || self.used_lut_families
                != self.canonical_luts.keys().copied().collect::<BTreeSet<_>>()
        {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
        let post_witness_current = lineage::exact_post_witness_current(
            &self.clear,
            self.public_memory_seed.as_ref(),
            &self.transitions_after_step,
            values,
        )?;
        Ok(LoweredBaseMultiplicity {
            clear: self.clear,
            public_memory_seed: self.public_memory_seed,
            after_producer: self.after_producer,
            post_witness_current,
        })
    }

    fn apply_after_producer(
        &mut self,
        producer: &SemanticBaseProducer,
        values: &mut SemanticValueMap,
    ) -> Result<(), InvocationShapeError> {
        let &(expected_position, expected_producer) = self
            .scheduled
            .get(self.next)
            .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
        if producer.position() != expected_position || producer.producer() != expected_producer {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }

        let (feed, transitions) = match producer {
            SemanticBaseProducer::Recorded(writer) => {
                let plan = self.take_plan(writer.producer.component, FeedPlanKind::Generic)?;
                let PlannedRecordedMultiplicityFeedGraph::Generic(plan) = plan else {
                    return Err(InvocationShapeError::InvalidScheduledProducerBinding);
                };
                let lowered = multiplicity_feed::lower_recorded(
                    self.arena,
                    plan,
                    writer,
                    self.canonical_luts.as_ref(),
                    values,
                )?;
                self.record_lut_families(&plan.plan.lut_families)?;
                let transitions = feed_transitions(&lowered);
                (Some(lowered), transitions)
            }
            SemanticBaseProducer::NativeEcOp { contract, .. } => {
                if self
                    .feeds_by_producer
                    .contains_key(expected_producer.component)
                {
                    return Err(InvocationShapeError::InvalidScheduledProducerBinding);
                }
                (None, ec_op_transitions(contract)?)
            }
            SemanticBaseProducer::NativeBlakeGDirect { contract, .. } => {
                let plan =
                    self.take_plan(expected_producer.component, FeedPlanKind::BlakeGFused)?;
                let PlannedRecordedMultiplicityFeedGraph::BlakeGFused {
                    plan, lut_tables, ..
                } = plan
                else {
                    return Err(InvocationShapeError::InvalidScheduledProducerBinding);
                };
                self.validate_direct_luts(plan, lut_tables, contract, values)?;
                self.record_lut_families(&plan.lut_families)?;
                (None, blake_g_transitions(contract)?)
            }
        };
        self.after_producer.push(feed);
        self.transitions_after_step.push(transitions);
        self.next = self
            .next
            .checked_add(1)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        self.validate_prefix_lineages(values)
    }

    fn take_plan(
        &mut self,
        producer: &'static str,
        expected_kind: FeedPlanKind,
    ) -> Result<&'arena PlannedRecordedMultiplicityFeedGraph, InvocationShapeError> {
        let plan_ref = self
            .feeds_by_producer
            .remove(producer)
            .filter(|plan| plan.kind == expected_kind)
            .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
        self.arena
            .multiplicity()
            .and_then(|multiplicity| multiplicity.feeds.get(plan_ref.index))
            .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)
    }

    fn record_lut_families(
        &mut self,
        families: &[&'static str],
    ) -> Result<(), InvocationShapeError> {
        for &family in families {
            if !self.canonical_luts.contains_key(family) {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            self.used_lut_families.insert(family);
        }
        Ok(())
    }

    fn validate_direct_luts(
        &self,
        plan: &crate::multiplicity_pipeline::PlannedRecordedMultiplicityFeed,
        lut_tables: &[ArenaBinding; 4],
        contract: &super::blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
        values: &SemanticValueMap,
    ) -> Result<(), InvocationShapeError> {
        let binding = blake_g_fused_feed_binding(plan)
            .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
        let families = binding.lut_indices.map(|index| plan.lut_families[index]);
        let luts = families
            .map(|family| {
                self.canonical_luts
                    .get(family)
                    .map(Vec::as_slice)
                    .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let identity =
            BlakeGDirectLutContentIdentity::from_host_words([luts[0], luts[1], luts[2], luts[3]])
                .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
        if !super::blake_g_direct_execution_authority::canonical_lut_content_is_exact(&identity) {
            return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
        }
        for (actual, &expected) in contract.invocation.luts.iter().zip(lut_tables) {
            let value = ArenaCatalogValueId(expected.logical.0);
            if actual.value.value != value
                || actual.value.value_words != (0..expected.len_words)
                || values.version(value)? != actual.version
            {
                return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
            }
        }
        Ok(())
    }

    fn validate_prefix_lineages(
        &self,
        values: &SemanticValueMap,
    ) -> Result<(), InvocationShapeError> {
        lineage::validate_prefix(
            &self.clear,
            self.public_memory_seed.as_ref(),
            &self.transitions_after_step,
            values,
        )
    }
}

fn exact_schedule(
    arena: &ProofArenaPlan,
    supplied: &BaseProducerSchedule,
) -> Result<Vec<(ProducerSchedulePosition, WitnessProducer)>, InvocationShapeError> {
    let exact = BaseProducerSchedule::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    if &exact != supplied {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let scheduled = supplied
        .witness_levels()
        .iter()
        .enumerate()
        .flat_map(|(level, lanes)| {
            lanes
                .iter()
                .copied()
                .enumerate()
                .map(move |(lane, producer)| (level, lane, producer))
        })
        .enumerate()
        .map(|(ordinal, (level, lane, producer))| {
            Ok((
                ProducerSchedulePosition {
                    level: u32::try_from(level).map_err(|_| InvocationShapeError::SizeOverflow)?,
                    lane: u32::try_from(lane).map_err(|_| InvocationShapeError::SizeOverflow)?,
                    ordinal: u32::try_from(ordinal)
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                },
                producer,
            ))
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    validate_step_prefix(arena, supplied, &scheduled)?;
    Ok(scheduled)
}

fn validate_step_prefix(
    arena: &ProofArenaPlan,
    schedule: &BaseProducerSchedule,
    scheduled: &[(ProducerSchedulePosition, WitnessProducer)],
) -> Result<(), InvocationShapeError> {
    let execution_tables = usize::from(arena.execution_tables().is_some());
    let seed = arena
        .multiplicity()
        .is_some_and(|multiplicity| multiplicity.public_memory_seed.is_some());
    let clear_index = execution_tables;
    let seed_index = clear_index
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let witness_index = seed_index
        .checked_add(usize::from(seed))
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let expected_witness =
        BaseProducerStep::witness(schedule.witness_levels().len(), scheduled.len())
            .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    let steps = schedule.steps();
    if steps
        .iter()
        .filter(|step| matches!(step, BaseProducerStep::ExecutionTables))
        .count()
        != execution_tables
        || execution_tables == 1 && steps.first() != Some(&BaseProducerStep::ExecutionTables)
        || steps.get(clear_index) != Some(&BaseProducerStep::MultiplicityClear)
        || seed && steps.get(seed_index) != Some(&BaseProducerStep::PublicMemorySeed)
        || steps.get(witness_index) != Some(&expected_witness)
        || steps
            .iter()
            .filter(|step| matches!(step, BaseProducerStep::MultiplicityClear))
            .count()
            != 1
        || steps
            .iter()
            .filter(|step| matches!(step, BaseProducerStep::WitnessDag { .. }))
            .count()
            != 1
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(())
}

fn exact_feed_alignment(
    arena: &ProofArenaPlan,
    scheduled: &[(ProducerSchedulePosition, WitnessProducer)],
) -> Result<
    (
        BTreeMap<&'static str, FeedPlanRef>,
        BTreeMap<&'static str, usize>,
    ),
    InvocationShapeError,
> {
    let multiplicity = arena
        .multiplicity()
        .filter(|multiplicity| multiplicity.coverage_complete() && multiplicity.blockers.is_empty())
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    // V1 admits only a closed producer/feed bijection: every recorded writer
    // owns one nonempty generic feed. A future zero-descriptor writer needs an
    // explicit no-feed schedule receipt rather than being inferred here.
    let mut feeds = BTreeMap::new();
    let mut lut_words = BTreeMap::new();
    for (index, feed) in multiplicity.feeds.iter().enumerate() {
        let (plan, kind) = match feed {
            PlannedRecordedMultiplicityFeedGraph::Generic(feed) => {
                (&feed.plan, FeedPlanKind::Generic)
            }
            PlannedRecordedMultiplicityFeedGraph::BlakeGFused { plan, .. } => {
                (plan, FeedPlanKind::BlakeGFused)
            }
        };
        if plan.lut_families.len() != plan.requirements.lut_words.len()
            || feeds
                .insert(plan.producer, FeedPlanRef { index, kind })
                .is_some()
        {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
        for (&family, &words) in plan.lut_families.iter().zip(&plan.requirements.lut_words) {
            if lut_words
                .insert(family, words)
                .is_some_and(|old| old != words)
            {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
        }
    }
    let scheduled_by_name = scheduled
        .iter()
        .map(|(_, producer)| (producer.component, *producer))
        .collect::<BTreeMap<_, _>>();
    if scheduled_by_name.len() != scheduled.len() {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    for &(_, producer) in scheduled {
        let plan = feeds.get(producer.component);
        let valid = match producer.kind {
            WitnessProducerKind::Recorded => {
                producer.part == Some(TracePartId::Main)
                    && plan.is_some_and(|plan| plan.kind == FeedPlanKind::Generic)
            }
            WitnessProducerKind::BlakeGDirect => {
                producer.part == Some(TracePartId::Main)
                    && plan.is_some_and(|plan| plan.kind == FeedPlanKind::BlakeGFused)
            }
            WitnessProducerKind::NativeEcOp => plan.is_none(),
            WitnessProducerKind::BlakeGFused => false,
        };
        if !valid {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
    }
    if feeds
        .keys()
        .any(|producer| !scheduled_by_name.contains_key(producer))
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok((feeds, lut_words))
}

fn canonical_luts(
    variant: PreProcessedTraceVariant,
    expected_words: &BTreeMap<&'static str, usize>,
) -> Result<BTreeMap<&'static str, Vec<u32>>, InvocationShapeError> {
    if expected_words.is_empty() {
        return Ok(BTreeMap::new());
    }
    let trace = Arc::new(variant.to_preprocessed_trace());
    expected_words
        .iter()
        .map(|(&family, &words)| {
            let lut = canonical_count_lut(family, Arc::clone(&trace))
                .map_err(|_| InvocationShapeError::InvalidScheduledProducerBinding)?;
            if lut.len() != words {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            Ok((family, lut))
        })
        .collect()
}

fn feed_transitions(feed: &LoweredMultiplicityFeed) -> Vec<TransitionReceipt> {
    feed.destinations
        .iter()
        .map(|transition| TransitionReceipt {
            value: transition.value,
            elements: transition.elements,
            source: transition.source,
            destination: transition.destination,
        })
        .collect()
}

fn ec_op_transitions(
    contract: &super::ec_op_prefix::LoweredNativeEcOpContract,
) -> Result<Vec<TransitionReceipt>, InvocationShapeError> {
    contract
        .invocation
        .multiplicities
        .iter()
        .map(|transition| {
            let elements = ElementRange::new(
                transition.value.value_words.start,
                transition.value.value_words.end,
            )
            .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
            Ok(TransitionReceipt {
                value: transition.value.value,
                elements,
                source: transition.source,
                destination: transition.destination,
            })
        })
        .collect()
}

fn blake_g_transitions(
    contract: &super::blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
) -> Result<Vec<TransitionReceipt>, InvocationShapeError> {
    contract
        .invocation
        .counts
        .iter()
        .map(|transition| {
            let elements = ElementRange::new(
                transition.value.value_words.start,
                transition.value.value_words.end,
            )
            .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
            Ok(TransitionReceipt {
                value: transition.value.value,
                elements,
                source: transition.source,
                destination: transition.destination,
            })
        })
        .collect()
}
