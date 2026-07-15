//! Proof-carrying bridge from logical lifetimes to the reused CUDA arena.
//!
//! The range allocator owns placement. This module owns the semantic boundary:
//! only an exact evaluation-to-coefficient transition may collapse two logical
//! identities into one stable CUDA slot. All other logical values retain unique
//! slot identities even when their disjoint lifetimes reuse the same address.

use std::collections::BTreeMap;

use stwo_backend_cuda::{ArenaLayout, ArenaRangeSpec, ArenaSlotId, ArenaSlotSpec};

use crate::arena_plan::{
    ArenaBinding, ArenaPlanError, BufferPurpose, CommitmentTreeId, LogicalBuffer, LogicalBufferId,
    ProofEpoch, ARENA_ALIGNMENT_WORDS,
};
use crate::range_allocator::{
    allocate_ranges, validate_range_layout, AliasGroupId, RangeId, RangeRequest,
};

#[derive(Clone, Debug)]
pub(crate) struct ValidatedTransitionAliases {
    partner_by_index: BTreeMap<usize, usize>,
    group_by_logical: BTreeMap<LogicalBufferId, AliasGroupId>,
    slot_owner_by_logical: BTreeMap<LogicalBufferId, LogicalBufferId>,
    pair_count: usize,
}

impl ValidatedTransitionAliases {
    pub(crate) fn partner_index(&self, index: usize) -> Option<usize> {
        self.partner_by_index.get(&index).copied()
    }

    pub(crate) fn group(&self, logical: LogicalBufferId) -> Option<AliasGroupId> {
        self.group_by_logical.get(&logical).copied()
    }

    fn slot_owner(&self, logical: LogicalBufferId) -> LogicalBufferId {
        self.slot_owner_by_logical
            .get(&logical)
            .copied()
            .unwrap_or(logical)
    }

    pub(crate) const fn pair_count(&self) -> usize {
        self.pair_count
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PlannedRangeArena {
    pub(crate) bindings: Vec<ArenaBinding>,
    pub(crate) layout: ArenaLayout,
    pub(crate) high_water_words: Vec<(ProofEpoch, usize)>,
    pub(crate) raw_peak_words: usize,
    pub(crate) excess_over_raw_peak_words: usize,
    pub(crate) range_view_count: usize,
    pub(crate) range_view_words: usize,
}

/// Typed non-adjacent reuse owned by replacement-v1. The quotient role is
/// padded to the full released slab; `used_words` remains the manifest extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReleasedCommitmentAlias {
    pub(crate) commitment: CommitmentTreeId,
    pub(crate) released_slab: LogicalBufferId,
    pub(crate) quotient_staging: LogicalBufferId,
    pub(crate) used_words: usize,
}

/// Validate the only semantic alias admitted by the resident prover: an
/// adjacent, equal-shaped interpolation transition that overwrites retired
/// evaluations with coefficients in place.
pub(crate) fn validate_transition_aliases(
    logical: &[LogicalBuffer],
    transition_aliases: &[(LogicalBufferId, LogicalBufferId)],
) -> Result<ValidatedTransitionAliases, ArenaPlanError> {
    let mut index_by_id = BTreeMap::new();
    for (index, buffer) in logical.iter().enumerate() {
        if index_by_id.insert(buffer.id, index).is_some() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "duplicate logical buffer id",
            ));
        }
    }

    let mut partner_by_index = BTreeMap::new();
    let mut group_by_logical = BTreeMap::new();
    let mut slot_owner_by_logical = BTreeMap::new();
    for (pair_index, &(evaluations, coefficients)) in transition_aliases.iter().enumerate() {
        let evaluation_index = *index_by_id
            .get(&evaluations)
            .ok_or(ArenaPlanError::MissingBinding(evaluations))?;
        let coefficient_index = *index_by_id
            .get(&coefficients)
            .ok_or(ArenaPlanError::MissingBinding(coefficients))?;
        let evaluation = &logical[evaluation_index];
        let coefficient = &logical[coefficient_index];
        let valid_purpose = matches!(
            (evaluation.purpose, coefficient.purpose),
            (BufferPurpose::BaseTrace, BufferPurpose::BaseCoefficients)
                | (
                    BufferPurpose::InteractionTrace,
                    BufferPurpose::InteractionCoefficients
                )
        );
        let unused = evaluation_index != coefficient_index
            && !partner_by_index.contains_key(&evaluation_index)
            && !partner_by_index.contains_key(&coefficient_index);
        if !valid_purpose
            || evaluation.component != coefficient.component
            || evaluation.part != coefficient.part
            || evaluation.ordinal != coefficient.ordinal
            || evaluation.len_words != coefficient.len_words
            || evaluation.lifetime.overlaps(coefficient.lifetime)
            || (evaluation.lifetime.last as u8).checked_add(1)
                != Some(coefficient.lifetime.first as u8)
            || !unused
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "invalid interpolation transition alias",
            ));
        }

        let group = AliasGroupId(
            u32::try_from(pair_index)
                .map_err(|_| ArenaPlanError::SizeOverflow)?
                .checked_add(1)
                .ok_or(ArenaPlanError::SizeOverflow)?,
        );
        let owner = evaluations.min(coefficients);
        partner_by_index.insert(evaluation_index, coefficient_index);
        partner_by_index.insert(coefficient_index, evaluation_index);
        group_by_logical.insert(evaluations, group);
        group_by_logical.insert(coefficients, group);
        slot_owner_by_logical.insert(evaluations, owner);
        slot_owner_by_logical.insert(coefficients, owner);
    }
    Ok(ValidatedTransitionAliases {
        partner_by_index,
        group_by_logical,
        slot_owner_by_logical,
        pair_count: transition_aliases.len(),
    })
}

pub(crate) fn validate_arena_aliases(
    logical: &[LogicalBuffer],
    transition_aliases: &[(LogicalBufferId, LogicalBufferId)],
    released_commitment_aliases: &[ReleasedCommitmentAlias],
) -> Result<ValidatedTransitionAliases, ArenaPlanError> {
    let mut validated = validate_transition_aliases(logical, transition_aliases)?;
    let index_by_id = logical
        .iter()
        .enumerate()
        .map(|(index, buffer)| (buffer.id, index))
        .collect::<BTreeMap<_, _>>();
    for (alias_index, alias) in released_commitment_aliases.iter().enumerate() {
        let released_index = *index_by_id
            .get(&alias.released_slab)
            .ok_or(ArenaPlanError::MissingBinding(alias.released_slab))?;
        let staging_index = *index_by_id
            .get(&alias.quotient_staging)
            .ok_or(ArenaPlanError::MissingBinding(alias.quotient_staging))?;
        let released = &logical[released_index];
        let staging = &logical[staging_index];
        let expected_epoch = match alias.commitment {
            CommitmentTreeId::Preprocessed => ProofEpoch::Ingest,
            CommitmentTreeId::Base => ProofEpoch::BaseCommit,
            CommitmentTreeId::Interaction => ProofEpoch::InteractionCommit,
            CommitmentTreeId::Composition => ProofEpoch::CompositionCommit,
            CommitmentTreeId::Fri(_) => {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "FRI storage cannot own quotient staging",
                ));
            }
        };
        let unused = released_index != staging_index
            && !validated.partner_by_index.contains_key(&released_index)
            && !validated.partner_by_index.contains_key(&staging_index);
        if released.purpose != BufferPurpose::CommitProgressiveStatePing
            || staging.purpose != BufferPurpose::QuotientNumeratorLdeTile
            || staging.ordinal == 0
            || released.len_words != staging.len_words
            || alias.used_words == 0
            || alias.used_words > staging.len_words
            || released.lifetime != crate::arena_plan::BufferLifetime::at(expected_epoch)
            || staging.lifetime != crate::arena_plan::BufferLifetime::at(ProofEpoch::Quotient)
            || released.lifetime.overlaps(staging.lifetime)
            || released.lifetime.last as u8 >= staging.lifetime.first as u8
            || !unused
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "invalid released-commitment quotient-staging alias",
            ));
        }
        let group_index = transition_aliases
            .len()
            .checked_add(alias_index)
            .and_then(|index| index.checked_add(1))
            .ok_or(ArenaPlanError::SizeOverflow)?;
        let group =
            AliasGroupId(u32::try_from(group_index).map_err(|_| ArenaPlanError::SizeOverflow)?);
        let owner = alias.released_slab.min(alias.quotient_staging);
        validated
            .partner_by_index
            .insert(released_index, staging_index);
        validated
            .partner_by_index
            .insert(staging_index, released_index);
        validated
            .group_by_logical
            .insert(alias.released_slab, group);
        validated
            .group_by_logical
            .insert(alias.quotient_staging, group);
        validated
            .slot_owner_by_logical
            .insert(alias.released_slab, owner);
        validated
            .slot_owner_by_logical
            .insert(alias.quotient_staging, owner);
        validated.pair_count += 1;
    }
    Ok(validated)
}

#[cfg(test)]
pub(crate) fn plan_range_arena(
    logical: &[LogicalBuffer],
    transition_aliases: &[(LogicalBufferId, LogicalBufferId)],
) -> Result<PlannedRangeArena, ArenaPlanError> {
    plan_range_arena_with_released_commitments(logical, transition_aliases, &[])
}

pub(crate) fn plan_range_arena_with_released_commitments(
    logical: &[LogicalBuffer],
    transition_aliases: &[(LogicalBufferId, LogicalBufferId)],
    released_commitment_aliases: &[ReleasedCommitmentAlias],
) -> Result<PlannedRangeArena, ArenaPlanError> {
    let aliases = validate_arena_aliases(logical, transition_aliases, released_commitment_aliases)?;
    let requests = logical
        .iter()
        .map(|buffer| RangeRequest {
            id: RangeId(buffer.id.0),
            len_words: buffer.len_words,
            alignment_words: ARENA_ALIGNMENT_WORDS,
            live_mask: buffer.lifetime.epoch_mask(),
            must_alias: aliases.group(buffer.id),
        })
        .collect::<Vec<_>>();
    let range_layout =
        allocate_ranges(&requests, ARENA_ALIGNMENT_WORDS, None).map_err(ArenaPlanError::Range)?;
    // Keep the independent validator at this trust boundary even though the
    // allocator currently self-validates. Future placement changes cannot
    // silently weaken the runtime proof.
    validate_range_layout(&requests, ARENA_ALIGNMENT_WORDS, None, &range_layout)
        .map_err(ArenaPlanError::Range)?;

    let mut bindings = Vec::with_capacity(logical.len());
    let mut specs = BTreeMap::<ArenaSlotId, ArenaRangeSpec>::new();
    for buffer in logical {
        let range = range_layout
            .binding(RangeId(buffer.id.0))
            .ok_or(ArenaPlanError::MissingBinding(buffer.id))?;
        let physical = slot_id(aliases.slot_owner(buffer.id))?;
        bindings.push(ArenaBinding {
            logical: buffer.id,
            physical,
            len_words: buffer.len_words,
        });
        let candidate = ArenaRangeSpec {
            slot: ArenaSlotSpec {
                id: physical,
                offset_words: range.offset_words,
                len_words: range.len_words,
                alignment_words: ARENA_ALIGNMENT_WORDS,
            },
            live_mask: buffer.lifetime.epoch_mask(),
        };
        match specs.get_mut(&physical) {
            Some(spec) => {
                if spec.slot != candidate.slot || spec.live_mask & candidate.live_mask != 0 {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "transition alias range view does not match exactly",
                    ));
                }
                spec.live_mask |= candidate.live_mask;
            }
            None => {
                specs.insert(physical, candidate);
            }
        }
    }
    bindings.sort_unstable_by_key(|binding| binding.logical);
    let specs = specs.into_values().collect::<Vec<_>>();
    let range_view_words = specs.iter().try_fold(0usize, |total, spec| {
        total
            .checked_add(spec.slot.len_words)
            .ok_or(ArenaPlanError::SizeOverflow)
    })?;

    // SAFETY: every mask is derived directly from the inclusive BufferLifetime
    // used by resident execution. `validate_range_layout` independently proves
    // exact lengths, alignment, bounds, required transition offsets, and no
    // overlap at any live epoch. The semantic validator above collapses only
    // equal-shaped adjacent interpolation transitions; every other logical
    // range keeps a unique slot id. Resident execution establishes a CUDA
    // happens-before edge across lifetime boundaries: forked lanes are joined
    // by events before their successor, and graph segments enqueue in protocol
    // order on the workspace stream. Thus disjoint masks cannot access reused
    // bytes concurrently even though an epoch boundary need not host-sync.
    let layout = unsafe { ArenaLayout::new_reused(range_layout.total_words(), &specs) }
        .map_err(ArenaPlanError::Arena)?;
    validate_runtime_views(logical, &bindings, &specs, &aliases)?;
    let high_water_words = ProofEpoch::ALL
        .into_iter()
        .map(|epoch| {
            let words = logical
                .iter()
                .filter(|buffer| buffer.lifetime.contains(epoch))
                .try_fold(0usize, |total, buffer| {
                    total
                        .checked_add(buffer.len_words)
                        .ok_or(ArenaPlanError::SizeOverflow)
                })?;
            Ok((epoch, words))
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    Ok(PlannedRangeArena {
        bindings,
        layout,
        high_water_words,
        raw_peak_words: range_layout.raw_peak_words(),
        excess_over_raw_peak_words: range_layout.excess_over_raw_peak_words(),
        range_view_count: specs.len(),
        range_view_words,
    })
}

fn slot_id(owner: LogicalBufferId) -> Result<ArenaSlotId, ArenaPlanError> {
    Ok(ArenaSlotId(
        owner.0.checked_add(1).ok_or(ArenaPlanError::SizeOverflow)?,
    ))
}

fn validate_runtime_views(
    logical: &[LogicalBuffer],
    bindings: &[ArenaBinding],
    specs: &[ArenaRangeSpec],
    aliases: &ValidatedTransitionAliases,
) -> Result<(), ArenaPlanError> {
    let specs = specs
        .iter()
        .map(|spec| (spec.slot.id, spec))
        .collect::<BTreeMap<_, _>>();
    if bindings
        .windows(2)
        .any(|pair| pair[0].logical >= pair[1].logical)
    {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "range arena bindings are not canonical and unique",
        ));
    }
    let bindings_by_id = bindings
        .iter()
        .map(|binding| (binding.logical, binding))
        .collect::<BTreeMap<_, _>>();
    if bindings_by_id.len() != logical.len() {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "range arena bindings are not canonical and unique",
        ));
    }
    for buffer in logical {
        let binding = bindings_by_id
            .get(&buffer.id)
            .ok_or(ArenaPlanError::MissingBinding(buffer.id))?;
        let spec = specs
            .get(&binding.physical)
            .ok_or(ArenaPlanError::MissingBinding(buffer.id))?;
        if binding.len_words != buffer.len_words
            || spec.slot.len_words != binding.len_words
            || spec.live_mask & buffer.lifetime.epoch_mask() != buffer.lifetime.epoch_mask()
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "range arena binding does not match its logical value",
            ));
        }
        let expected = slot_id(aliases.slot_owner(buffer.id))?;
        if binding.physical != expected {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "range arena slot identity drifted",
            ));
        }
    }
    if specs.len() != logical.len() - aliases.pair_count() {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "range arena view cardinality drifted",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "range_arena_tests.rs"]
mod tests;
