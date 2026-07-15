//! Pure, CPU-only physical receipt for the direct Composition slab deletion.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    append_composition_output_buffers, color_logical_buffers_with_released_commitments,
    plan_range_arena_with_released_commitments, validate_aliases, ArenaBinding, ArenaLayout,
    ArenaPlanError, BufferPurpose, CommitmentGeometry, CommitmentTreeId, CompositionOutputMode,
    CompositionOutputPlan, LogicalBufferId, LogicalCompositionOutput, ProofArenaPlan, ProofEpoch,
    ReleasedCommitmentAlias,
};

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

/// Direct output versus an exact forced coefficient-output counterfactual.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionSlabArenaCounterfactual {
    pub direct: CompositionSlabArenaFootprint,
    pub forced_coefficient_fallback: CompositionSlabArenaFootprint,
    pub fallback_coefficient_buffers: usize,
    pub fallback_coefficient_words: usize,
    pub identical_logical_prefix_buffers: usize,
    pub transition_alias_pairs: usize,
    pub released_commitment_alias_pairs: usize,
}

impl ProofArenaPlan {
    /// Re-color this exact Direct plan after adding only the production
    /// CoefficientSplit outputs. No policy, commitment geometry, lifetime, or
    /// semantic alias changes. The current plan is first reconstructed and
    /// required to reproduce its bindings, slot ranges, and accounting exactly.
    pub fn composition_slab_arena_counterfactual(
        &self,
    ) -> Result<CompositionSlabArenaCounterfactual, ArenaPlanError> {
        if self.composition.output_plan.mode() != CompositionOutputMode::DirectRetainedEvaluations
            || self
                .logical
                .iter()
                .any(|buffer| buffer.purpose == BufferPurpose::CompositionCoefficients)
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "composition slab receipt requires a coefficient-free Direct plan",
            ));
        }

        let (transition_aliases, released_aliases) = self.reconstruct_aliases()?;
        let (_, whole_specs, whole_slot_total_words) =
            color_logical_buffers_with_released_commitments(
                &self.logical,
                &transition_aliases,
                &released_aliases,
            )?;
        ArenaLayout::new(whole_slot_total_words, &whole_specs).map_err(ArenaPlanError::Arena)?;
        let reconstructed = plan_range_arena_with_released_commitments(
            &self.logical,
            &transition_aliases,
            &released_aliases,
        )?;
        self.require_exact_reconstruction(&reconstructed, whole_slot_total_words)?;

        let mut fallback_logical = self.logical.clone();
        let prefix_len = fallback_logical.len();
        let commitment = self.commitment(CommitmentTreeId::Composition).ok_or(
            ArenaPlanError::InvalidProtocolGeometry("missing composition commitment geometry"),
        )?;
        let geometry = CommitmentGeometry {
            id: commitment.id,
            created: ProofEpoch::CompositionCommit,
            config: commitment.config,
            grouped_column_log_sizes: commitment.grouped_column_log_sizes.clone(),
            grouped_column_sources: commitment.grouped_column_sources.clone(),
            retained_evaluation_groups: selected_groups(&commitment.retained_evaluation_groups),
            direct_composition_evaluation_groups: selected_groups(
                &commitment.direct_composition_evaluation_groups,
            ),
            numerator_evaluation_groups: selected_groups(&commitment.numerator_evaluation_groups),
        };
        let output = append_composition_output_buffers(
            &mut fallback_logical,
            &geometry,
            &self.composition.requirements,
            CompositionOutputPlan::CoefficientSplit,
            &self.late_coefficient_ownership,
        )?;
        let LogicalCompositionOutput::CoefficientSplit(coefficient_ids) = output else {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "forced Composition coefficient fallback did not append coefficients",
            ));
        };
        if fallback_logical[..prefix_len] != self.logical[..]
            || fallback_logical.len() != prefix_len + coefficient_ids.len()
            || coefficient_ids.iter().any(|id| {
                fallback_logical[id.0 as usize].purpose != BufferPurpose::CompositionCoefficients
            })
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "forced Composition fallback changed non-output logical geometry",
            ));
        }
        let fallback_coefficient_words = coefficient_ids.iter().try_fold(0usize, |sum, id| {
            sum.checked_add(fallback_logical[id.0 as usize].len_words)
                .ok_or(ArenaPlanError::SizeOverflow)
        })?;

        let (_, fallback_whole_specs, fallback_whole_slot_total_words) =
            color_logical_buffers_with_released_commitments(
                &fallback_logical,
                &transition_aliases,
                &released_aliases,
            )?;
        ArenaLayout::new(fallback_whole_slot_total_words, &fallback_whole_specs)
            .map_err(ArenaPlanError::Arena)?;
        let fallback = plan_range_arena_with_released_commitments(
            &fallback_logical,
            &transition_aliases,
            &released_aliases,
        )?;
        validate_aliases(&fallback_logical, &fallback.bindings)?;

        Ok(CompositionSlabArenaCounterfactual {
            direct: footprint(
                self.total_words(),
                self.whole_slot_total_words,
                self.raw_peak_words,
                self.excess_over_raw_peak_words,
                self.range_view_count,
                self.range_view_words,
                self.high_water_words.clone(),
            ),
            forced_coefficient_fallback: footprint(
                fallback.layout.total_words(),
                fallback_whole_slot_total_words,
                fallback.raw_peak_words,
                fallback.excess_over_raw_peak_words,
                fallback.range_view_count,
                fallback.range_view_words,
                fallback.high_water_words,
            ),
            fallback_coefficient_buffers: coefficient_ids.len(),
            fallback_coefficient_words,
            identical_logical_prefix_buffers: prefix_len,
            transition_alias_pairs: transition_aliases.len(),
            released_commitment_alias_pairs: released_aliases.len(),
        })
    }

    fn reconstruct_aliases(
        &self,
    ) -> Result<
        (
            Vec<(LogicalBufferId, LogicalBufferId)>,
            Vec<ReleasedCommitmentAlias>,
        ),
        ArenaPlanError,
    > {
        let by_id = self
            .logical
            .iter()
            .map(|buffer| (buffer.id, buffer))
            .collect::<BTreeMap<_, _>>();
        let released = self
            .quotient_numerator
            .staged_overflows
            .iter()
            .map(|role| ReleasedCommitmentAlias {
                commitment: role.commitment,
                released_slab: role.released_slab.logical,
                quotient_staging: role.staging.logical,
                used_words: role.used_words,
            })
            .collect::<Vec<_>>();
        let mut expected_released = released
            .iter()
            .map(|alias| canonical_pair(alias.released_slab, alias.quotient_staging))
            .collect::<BTreeSet<_>>();
        if expected_released.len() != released.len() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "released commitment alias identities are not unique",
            ));
        }

        let mut by_physical = BTreeMap::new();
        for binding in &self.bindings {
            by_physical
                .entry(binding.physical)
                .or_insert_with(Vec::new)
                .push(binding.logical);
        }
        let mut transitions = Vec::new();
        for group in by_physical.values() {
            if group.len() == 1 {
                continue;
            }
            let [left, right] = group.as_slice() else {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "physical arena slot has more than one semantic alias pair",
                ));
            };
            if expected_released.remove(&canonical_pair(*left, *right)) {
                continue;
            }
            let left = by_id
                .get(left)
                .ok_or(ArenaPlanError::MissingBinding(*left))?;
            let right = by_id
                .get(right)
                .ok_or(ArenaPlanError::MissingBinding(*right))?;
            let pair = match (left.purpose, right.purpose) {
                (BufferPurpose::BaseTrace, BufferPurpose::BaseCoefficients)
                | (BufferPurpose::InteractionTrace, BufferPurpose::InteractionCoefficients) => {
                    (left.id, right.id)
                }
                (BufferPurpose::BaseCoefficients, BufferPurpose::BaseTrace)
                | (BufferPurpose::InteractionCoefficients, BufferPurpose::InteractionTrace) => {
                    (right.id, left.id)
                }
                _ => {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "physical arena slot does not identify a typed semantic alias",
                    ));
                }
            };
            transitions.push(pair);
        }
        if !expected_released.is_empty() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "released commitment alias is absent from current physical bindings",
            ));
        }
        Ok((transitions, released))
    }

    fn require_exact_reconstruction(
        &self,
        reconstructed: &crate::range_arena::PlannedRangeArena,
        whole_slot_total_words: usize,
    ) -> Result<(), ArenaPlanError> {
        if reconstructed.bindings != self.bindings
            || reconstructed.layout.total_words() != self.total_words()
            || reconstructed.high_water_words != self.high_water_words
            || whole_slot_total_words != self.whole_slot_total_words
            || reconstructed.raw_peak_words != self.raw_peak_words
            || reconstructed.excess_over_raw_peak_words != self.excess_over_raw_peak_words
            || reconstructed.range_view_count != self.range_view_count
            || reconstructed.range_view_words != self.range_view_words
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "reconstructed direct arena accounting differs from the committed plan",
            ));
        }
        for physical in self
            .bindings
            .iter()
            .map(|binding| binding.physical)
            .collect::<BTreeSet<_>>()
        {
            if reconstructed.layout.slot(physical) != self.layout.slot(physical) {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "reconstructed direct arena slot range differs from the committed plan",
                ));
            }
        }
        Ok(())
    }
}

fn selected_groups(groups: &[Option<Vec<ArenaBinding>>]) -> Vec<bool> {
    groups.iter().map(Option::is_some).collect()
}

fn canonical_pair(
    left: LogicalBufferId,
    right: LogicalBufferId,
) -> (LogicalBufferId, LogicalBufferId) {
    (left.min(right), left.max(right))
}

fn footprint(
    total_words: usize,
    whole_slot_total_words: usize,
    raw_peak_words: usize,
    excess_over_raw_peak_words: usize,
    range_view_count: usize,
    range_view_words: usize,
    epoch_live_words: Vec<(ProofEpoch, usize)>,
) -> CompositionSlabArenaFootprint {
    CompositionSlabArenaFootprint {
        total_words,
        whole_slot_total_words,
        raw_peak_words,
        excess_over_raw_peak_words,
        range_view_count,
        range_view_words,
        epoch_live_words,
    }
}
