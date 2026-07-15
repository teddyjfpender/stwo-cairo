//! Production-order-exact physical receipt for Composition output ownership.

use std::collections::BTreeMap;

use super::{
    checked_pow2, ArenaBinding, ArenaPlanError, BufferLifetime, BufferPurpose, CommitmentTreeId,
    LogicalBuffer, LogicalBufferId, PlannedCommitment, ProofArenaPlan, ProofEpoch,
};
use crate::prepared_composition::CompositionOutputMode;
use stwo_cairo_prover::witness::proof_shape::TracePartId;

/// Absolute arena facts for one otherwise-identical Composition output policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionSlabArenaFootprint {
    pub total_words: usize,
    pub whole_slot_total_words: usize,
    pub raw_peak_words: usize,
    pub excess_over_raw_peak_words: usize,
    pub range_view_count: usize,
    pub range_view_words: usize,
    /// Sum of live logical words at each epoch; not a physical address extent.
    pub epoch_live_words: Vec<(ProofEpoch, usize)>,
}

/// Production Direct plan versus a complete production-order fallback build.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionSlabArenaCounterfactual {
    pub direct: CompositionSlabArenaFootprint,
    pub forced_coefficient_fallback: CompositionSlabArenaFootprint,
    pub fallback_coefficient_buffers: usize,
    pub fallback_coefficient_words: usize,
    pub fallback_coefficient_insertion_index: usize,
    pub identical_non_output_logical_buffers: usize,
    pub transition_alias_pairs: usize,
    pub released_commitment_alias_pairs: usize,
    pub fallback_coefficient_alias_pairs: usize,
}

impl ProofArenaPlan {
    /// Compare two complete arena builds made from identical inputs. `self` is
    /// the production Direct build; `fallback` traversed the same builder with
    /// only its Composition output choice forced to CoefficientSplit at the
    /// production insertion point.
    pub fn composition_slab_arena_counterfactual(
        &self,
        fallback: &Self,
    ) -> Result<CompositionSlabArenaCounterfactual, ArenaPlanError> {
        if self.composition.output_plan.mode() != CompositionOutputMode::DirectRetainedEvaluations
            || fallback.composition.output_plan.mode() != CompositionOutputMode::CoefficientSplit
            || self
                .logical
                .iter()
                .any(|buffer| buffer.purpose == BufferPurpose::CompositionCoefficients)
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "Composition slab receipt requires Direct and forced CoefficientSplit builds",
            ));
        }
        if self.shape_key != fallback.shape_key
            || self.protocol_key != fallback.protocol_key
            || self.protocol_identity != fallback.protocol_identity
            || self.late_coefficient_ownership.entries()
                != fallback.late_coefficient_ownership.entries()
            || self.composition.plan != fallback.composition.plan
            || self.composition.requirements != fallback.composition.requirements
            || self.composition.direct_retention != fallback.composition.direct_retention
            || !same_commitment_geometry(&self.commitments, &fallback.commitments)
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "forced Composition fallback changed non-output protocol geometry",
            ));
        }

        let coefficients = fallback
            .logical
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.purpose == BufferPurpose::CompositionCoefficients)
            .collect::<Vec<_>>();
        let composition = self.commitment(CommitmentTreeId::Composition).ok_or(
            ArenaPlanError::InvalidProtocolGeometry("missing composition commitment geometry"),
        )?;
        let logs = composition
            .grouped_column_log_sizes
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        if coefficients.len() != 8
            || logs.len() != coefficients.len()
            || fallback.logical.len() != self.logical.len() + coefficients.len()
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "forced Composition fallback does not contain exactly eight added outputs",
            ));
        }
        let insertion_index = coefficients[0].0;
        let mut coefficient_words = 0usize;
        for (ordinal, ((index, buffer), log_size)) in coefficients.iter().zip(logs).enumerate() {
            let ordinal_u32 = u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?;
            let expected_lifetime = BufferLifetime::new(
                ProofEpoch::Composition,
                fallback
                    .late_coefficient_ownership
                    .final_consumer(super::OpenedColumnSource::Composition {
                        ordinal: ordinal_u32,
                    })
                    .map_err(ArenaPlanError::InvalidProtocolGeometry)?,
            )?;
            if *index != insertion_index + ordinal
                || buffer.id.0 as usize != *index
                || buffer.component.is_some()
                || buffer.part.is_some()
                || buffer.ordinal != ordinal_u32
                || buffer.len_words != checked_pow2(log_size)?
                || buffer.lifetime != expected_lifetime
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "forced Composition coefficient output geometry drifted",
                ));
            }
            coefficient_words = coefficient_words
                .checked_add(buffer.len_words)
                .ok_or(ArenaPlanError::SizeOverflow)?;
        }

        let direct_non_output = self
            .logical
            .iter()
            .filter(|buffer| buffer.purpose != BufferPurpose::CompositionCoefficients);
        let fallback_non_output = fallback
            .logical
            .iter()
            .filter(|buffer| buffer.purpose != BufferPurpose::CompositionCoefficients);
        if direct_non_output
            .zip(fallback_non_output)
            .any(|(direct, fallback)| !same_logical_geometry(direct, fallback))
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "forced Composition fallback changed non-output logical geometry",
            ));
        }
        if fallback.range_view_count != self.range_view_count + coefficients.len()
            || fallback.range_view_words
                != self
                    .range_view_words
                    .checked_add(coefficient_words)
                    .ok_or(ArenaPlanError::SizeOverflow)?
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "forced Composition fallback range-view accounting drifted",
            ));
        }
        for epoch in ProofEpoch::ALL {
            let expected = if matches!(
                epoch,
                ProofEpoch::Composition | ProofEpoch::CompositionCommit
            ) {
                coefficient_words
            } else {
                0
            };
            if fallback.high_water_words(epoch)
                != self
                    .high_water_words(epoch)
                    .checked_add(expected)
                    .ok_or(ArenaPlanError::SizeOverflow)?
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "forced Composition fallback changed an unrelated epoch live set",
                ));
            }
        }

        let direct_aliases = semantic_aliases(self)?;
        let fallback_aliases = semantic_aliases(fallback)?;
        if direct_aliases.non_output != fallback_aliases.non_output
            || direct_aliases.released_commitments != fallback_aliases.released_commitments
            || !direct_aliases.coefficient_involving.is_empty()
            || fallback_aliases
                .coefficient_involving
                .iter()
                .any(|pair| !pair.is_valid_coefficient_alias())
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "forced Composition fallback changed non-output semantic aliases",
            ));
        }
        let transition_alias_pairs = direct_aliases
            .non_output
            .len()
            .checked_sub(direct_aliases.released_commitments.len())
            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "released commitment aliases exceed all non-output semantic aliases",
            ))?;
        Ok(CompositionSlabArenaCounterfactual {
            direct: footprint(self),
            forced_coefficient_fallback: footprint(fallback),
            fallback_coefficient_buffers: coefficients.len(),
            fallback_coefficient_words: coefficient_words,
            fallback_coefficient_insertion_index: insertion_index,
            identical_non_output_logical_buffers: self.logical.len(),
            transition_alias_pairs,
            released_commitment_alias_pairs: direct_aliases.released_commitments.len(),
            fallback_coefficient_alias_pairs: fallback_aliases.coefficient_involving.len(),
        })
    }
}

fn same_commitment_geometry(direct: &[PlannedCommitment], fallback: &[PlannedCommitment]) -> bool {
    direct.len() == fallback.len()
        && direct.iter().zip(fallback).all(|(direct, fallback)| {
            direct.id == fallback.id
                && direct.storage_mode == fallback.storage_mode
                && direct.commit_program == fallback.commit_program
                && direct.domain_cooperative_program == fallback.domain_cooperative_program
                && direct.compact_domain_program == fallback.compact_domain_program
                && direct.direct_retained_b2n_program == fallback.direct_retained_b2n_program
                && direct.direct_compact_terminal == fallback.direct_compact_terminal
                && direct.config == fallback.config
                && direct.grouped_column_log_sizes == fallback.grouped_column_log_sizes
                && direct.grouped_column_sources == fallback.grouped_column_sources
                && direct.requirements == fallback.requirements
                && same_binding_group_shape(
                    &direct.retained_evaluation_groups,
                    &fallback.retained_evaluation_groups,
                )
                && same_binding_group_shape(
                    &direct.evaluation_output_groups,
                    &fallback.evaluation_output_groups,
                )
                && same_binding_group_shape(
                    &direct.direct_composition_evaluation_groups,
                    &fallback.direct_composition_evaluation_groups,
                )
                && same_binding_group_shape(
                    &direct.numerator_evaluation_groups,
                    &fallback.numerator_evaluation_groups,
                )
                && direct.interpolation_mode == fallback.interpolation_mode
                && direct.interpolation_batches.len() == fallback.interpolation_batches.len()
                && direct
                    .interpolation_batches
                    .iter()
                    .zip(&fallback.interpolation_batches)
                    .all(|(direct, fallback)| {
                        direct.log_size == fallback.log_size && direct.sources == fallback.sources
                    })
        })
}

fn same_binding_group_shape(
    direct: &[Option<Vec<ArenaBinding>>],
    fallback: &[Option<Vec<ArenaBinding>>],
) -> bool {
    direct.len() == fallback.len()
        && direct.iter().zip(fallback).all(|(direct, fallback)| {
            direct.as_ref().map(Vec::len) == fallback.as_ref().map(Vec::len)
        })
}

fn same_logical_geometry(direct: &LogicalBuffer, fallback: &LogicalBuffer) -> bool {
    direct.component == fallback.component
        && direct.part == fallback.part
        && direct.purpose == fallback.purpose
        && direct.ordinal == fallback.ordinal
        && direct.len_words == fallback.len_words
        && direct.lifetime == fallback.lifetime
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SemanticTracePart {
    None,
    Main,
    MemoryBig(u32),
    MemorySmall,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BufferLifetimeKey {
    first: ProofEpoch,
    last: ProofEpoch,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SemanticBuffer {
    component: Option<&'static str>,
    part: SemanticTracePart,
    purpose: BufferPurpose,
    ordinal: u32,
    len_words: usize,
    lifetime: BufferLifetimeKey,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SemanticAliasPair(SemanticBuffer, SemanticBuffer);

impl SemanticAliasPair {
    fn new(left: &LogicalBuffer, right: &LogicalBuffer) -> Self {
        let left = semantic_buffer(left);
        let right = semantic_buffer(right);
        if left <= right {
            Self(left, right)
        } else {
            Self(right, left)
        }
    }

    fn coefficient_count(self) -> usize {
        usize::from(self.0.purpose == BufferPurpose::CompositionCoefficients)
            + usize::from(self.1.purpose == BufferPurpose::CompositionCoefficients)
    }

    fn is_valid_coefficient_alias(self) -> bool {
        self.coefficient_count() == 1
            && (self.0.lifetime.last < self.1.lifetime.first
                || self.1.lifetime.last < self.0.lifetime.first)
    }

    fn is_valid_transition_alias(self) -> bool {
        let purposes = (self.0.purpose, self.1.purpose);
        matches!(
            purposes,
            (BufferPurpose::BaseTrace, BufferPurpose::BaseCoefficients)
                | (BufferPurpose::BaseCoefficients, BufferPurpose::BaseTrace)
                | (
                    BufferPurpose::InteractionTrace,
                    BufferPurpose::InteractionCoefficients
                )
                | (
                    BufferPurpose::InteractionCoefficients,
                    BufferPurpose::InteractionTrace
                )
        ) && self.0.component == self.1.component
            && self.0.part == self.1.part
            && self.0.ordinal == self.1.ordinal
            && self.0.len_words == self.1.len_words
            && (self.0.lifetime.last < self.1.lifetime.first
                || self.1.lifetime.last < self.0.lifetime.first)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct SemanticAliases {
    non_output: Vec<SemanticAliasPair>,
    released_commitments: Vec<SemanticAliasPair>,
    coefficient_involving: Vec<SemanticAliasPair>,
}

fn semantic_aliases(plan: &ProofArenaPlan) -> Result<SemanticAliases, ArenaPlanError> {
    let mut groups = BTreeMap::new();
    for binding in &plan.bindings {
        groups
            .entry(binding.physical)
            .or_insert_with(Vec::new)
            .push(binding.logical);
    }
    if groups.values().any(|group| group.len() > 2) {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "arena slot contains more than one semantic alias pair",
        ));
    }
    let mut non_output = Vec::new();
    let mut coefficient_involving = Vec::new();
    for group in groups.values().filter(|group| group.len() == 2) {
        let pair = semantic_alias_pair(plan, group[0], group[1])?;
        if pair.coefficient_count() == 0 {
            non_output.push(pair);
        } else {
            coefficient_involving.push(pair);
        }
    }
    non_output.sort_unstable();
    coefficient_involving.sort_unstable();

    let mut released_commitments = Vec::new();
    for role in &plan.quotient_numerator.staged_overflows {
        if role.released_slab.physical != role.staging.physical {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "released commitment alias is absent from physical bindings",
            ));
        }
        let pair = semantic_alias_pair(plan, role.released_slab.logical, role.staging.logical)?;
        if pair.coefficient_count() != 0 || non_output.binary_search(&pair).is_err() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "released commitment alias is absent from normalized semantic aliases",
            ));
        }
        released_commitments.push(pair);
    }
    released_commitments.sort_unstable();
    if released_commitments
        .windows(2)
        .any(|pair| pair[0] == pair[1])
    {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "released commitment semantic alias is duplicated",
        ));
    }
    if non_output.iter().any(|pair| {
        released_commitments.binary_search(pair).is_err() && !pair.is_valid_transition_alias()
    }) {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "physical arena slot does not identify a typed semantic alias",
        ));
    }
    Ok(SemanticAliases {
        non_output,
        released_commitments,
        coefficient_involving,
    })
}

fn semantic_alias_pair(
    plan: &ProofArenaPlan,
    left: LogicalBufferId,
    right: LogicalBufferId,
) -> Result<SemanticAliasPair, ArenaPlanError> {
    let find = |id: LogicalBufferId| {
        plan.logical
            .get(id.0 as usize)
            .filter(|buffer| buffer.id == id)
            .ok_or(ArenaPlanError::MissingBinding(id))
    };
    Ok(SemanticAliasPair::new(find(left)?, find(right)?))
}

fn semantic_buffer(buffer: &LogicalBuffer) -> SemanticBuffer {
    let part = match buffer.part {
        None => SemanticTracePart::None,
        Some(TracePartId::Main) => SemanticTracePart::Main,
        Some(TracePartId::MemoryBig(index)) => SemanticTracePart::MemoryBig(index),
        Some(TracePartId::MemorySmall) => SemanticTracePart::MemorySmall,
    };
    SemanticBuffer {
        component: buffer.component,
        part,
        purpose: buffer.purpose,
        ordinal: buffer.ordinal,
        len_words: buffer.len_words,
        lifetime: BufferLifetimeKey {
            first: buffer.lifetime.first,
            last: buffer.lifetime.last,
        },
    }
}

fn footprint(plan: &ProofArenaPlan) -> CompositionSlabArenaFootprint {
    CompositionSlabArenaFootprint {
        total_words: plan.total_words(),
        whole_slot_total_words: plan.whole_slot_total_words,
        raw_peak_words: plan.raw_peak_words,
        excess_over_raw_peak_words: plan.excess_over_raw_peak_words,
        range_view_count: plan.range_view_count,
        range_view_words: plan.range_view_words,
        epoch_live_words: plan.high_water_words.clone(),
    }
}
