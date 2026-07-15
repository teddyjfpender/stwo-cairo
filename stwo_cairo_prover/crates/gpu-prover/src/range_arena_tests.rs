use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::*;
use crate::arena_plan::BufferLifetime;

fn buffer(
    id: u32,
    purpose: BufferPurpose,
    ordinal: u32,
    len_words: usize,
    first: ProofEpoch,
    last: ProofEpoch,
) -> LogicalBuffer {
    LogicalBuffer {
        id: LogicalBufferId(id),
        component: Some("component"),
        part: Some(TracePartId::Main),
        purpose,
        ordinal,
        len_words,
        lifetime: BufferLifetime::new(first, last).unwrap(),
    }
}

fn binding(plan: &PlannedRangeArena, id: u32) -> ArenaBinding {
    plan.bindings
        .iter()
        .find(|binding| binding.logical == LogicalBufferId(id))
        .copied()
        .unwrap()
}

fn offset(plan: &PlannedRangeArena, id: u32) -> usize {
    plan.layout
        .slot(binding(plan, id).physical)
        .unwrap()
        .offset_words
}

fn assert_live_address_union(plan: &PlannedRangeArena, logical: &[LogicalBuffer]) {
    for epoch in ProofEpoch::ALL {
        let mut ranges = logical
            .iter()
            .filter(|buffer| buffer.lifetime.contains(epoch))
            .map(|buffer| {
                let slot = plan
                    .layout
                    .slot(binding(plan, buffer.id.0).physical)
                    .unwrap();
                (slot.offset_words, slot.offset_words + slot.len_words)
            })
            .collect::<Vec<_>>();
        ranges.sort_unstable();
        assert!(ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0));
        let union_words = ranges.iter().map(|(start, end)| end - start).sum::<usize>();
        assert_eq!(
            union_words,
            plan.high_water_words
                .iter()
                .find_map(|&(candidate, words)| (candidate == epoch).then_some(words))
                .unwrap()
        );
    }
}

#[test]
fn unrelated_disjoint_ranges_reuse_address_not_identity() {
    let logical = vec![
        buffer(
            0,
            BufferPurpose::CommitLdeTile,
            0,
            128,
            ProofEpoch::Witness,
            ProofEpoch::Witness,
        ),
        buffer(
            1,
            BufferPurpose::CommitLdeTile,
            1,
            96,
            ProofEpoch::Interaction,
            ProofEpoch::Interaction,
        ),
        buffer(
            2,
            BufferPurpose::CommitLdeTile,
            2,
            64,
            ProofEpoch::Witness,
            ProofEpoch::Interaction,
        ),
    ];
    let plan = plan_range_arena(&logical, &[]).unwrap();

    assert_ne!(binding(&plan, 0).physical, binding(&plan, 1).physical);
    assert_eq!(offset(&plan, 0), offset(&plan, 1));
    assert_eq!(plan.range_view_count, logical.len());
    assert_eq!(plan.range_view_words, 128 + 96 + 64);
    assert_eq!(plan.raw_peak_words, 128 + 64);
    assert_eq!(plan.layout.total_words(), 128 + 64);
    for buffer in &logical {
        let binding = binding(&plan, buffer.id.0);
        let slot = plan.layout.slot(binding.physical).unwrap();
        assert_eq!(binding.len_words, buffer.len_words);
        assert_eq!(slot.len_words, buffer.len_words);
        assert_eq!(slot.offset_words % ARENA_ALIGNMENT_WORDS, 0);
    }
    assert_live_address_union(&plan, &logical);
}

#[test]
fn only_in_place_transition_shares_identity_and_address() {
    let logical = vec![
        buffer(
            0,
            BufferPurpose::BaseTrace,
            7,
            128,
            ProofEpoch::Witness,
            ProofEpoch::Witness,
        ),
        buffer(
            1,
            BufferPurpose::BaseCoefficients,
            7,
            128,
            ProofEpoch::BaseCommit,
            ProofEpoch::Decommit,
        ),
        // This evaluation is retained into Interaction, so its coefficient
        // transition is not authorized to overwrite it in place.
        buffer(
            2,
            BufferPurpose::BaseTrace,
            8,
            128,
            ProofEpoch::Witness,
            ProofEpoch::Interaction,
        ),
        buffer(
            3,
            BufferPurpose::BaseCoefficients,
            8,
            128,
            ProofEpoch::BaseCommit,
            ProofEpoch::Decommit,
        ),
    ];
    let plan = plan_range_arena(&logical, &[(LogicalBufferId(0), LogicalBufferId(1))]).unwrap();

    assert_eq!(binding(&plan, 0).physical, binding(&plan, 1).physical);
    assert_eq!(offset(&plan, 0), offset(&plan, 1));
    assert_ne!(binding(&plan, 2).physical, binding(&plan, 3).physical);
    assert_ne!(offset(&plan, 2), offset(&plan, 3));
    assert_eq!(plan.range_view_count, logical.len() - 1);
    assert_live_address_union(&plan, &logical);

    assert!(matches!(
        plan_range_arena(
            &logical,
            &[
                (LogicalBufferId(0), LogicalBufferId(1)),
                (LogicalBufferId(2), LogicalBufferId(3)),
            ],
        ),
        Err(ArenaPlanError::InvalidProtocolGeometry(
            "invalid interpolation transition alias"
        ))
    ));
}

#[test]
fn transition_union_masks_prevent_cross_pair_overlap() {
    let logical = vec![
        buffer(
            0,
            BufferPurpose::BaseTrace,
            0,
            128,
            ProofEpoch::Witness,
            ProofEpoch::Witness,
        ),
        buffer(
            1,
            BufferPurpose::BaseCoefficients,
            0,
            128,
            ProofEpoch::BaseCommit,
            ProofEpoch::Interaction,
        ),
        buffer(
            2,
            BufferPurpose::InteractionTrace,
            1,
            128,
            ProofEpoch::Interaction,
            ProofEpoch::Interaction,
        ),
        buffer(
            3,
            BufferPurpose::InteractionCoefficients,
            1,
            128,
            ProofEpoch::InteractionCommit,
            ProofEpoch::Oods,
        ),
    ];
    let plan = plan_range_arena(
        &logical,
        &[
            (LogicalBufferId(0), LogicalBufferId(1)),
            (LogicalBufferId(2), LogicalBufferId(3)),
        ],
    )
    .unwrap();

    assert_eq!(binding(&plan, 0).physical, binding(&plan, 1).physical);
    assert_eq!(binding(&plan, 2).physical, binding(&plan, 3).physical);
    assert_ne!(offset(&plan, 0), offset(&plan, 2));
    assert_live_address_union(&plan, &logical);
}

#[test]
fn malformed_transition_shape_is_rejected_before_placement() {
    let logical = vec![
        buffer(
            0,
            BufferPurpose::BaseTrace,
            4,
            64,
            ProofEpoch::Witness,
            ProofEpoch::Witness,
        ),
        buffer(
            1,
            BufferPurpose::BaseCoefficients,
            4,
            96,
            ProofEpoch::BaseCommit,
            ProofEpoch::BaseCommit,
        ),
    ];
    assert!(matches!(
        plan_range_arena(&logical, &[(LogicalBufferId(0), LogicalBufferId(1))]),
        Err(ArenaPlanError::InvalidProtocolGeometry(
            "invalid interpolation transition alias"
        ))
    ));
}

#[test]
fn named_released_commitment_alias_has_exact_identity_offset_and_extent() {
    let logical = vec![
        buffer(
            0,
            BufferPurpose::CommitProgressiveStatePing,
            0,
            256,
            ProofEpoch::BaseCommit,
            ProofEpoch::BaseCommit,
        ),
        buffer(
            1,
            BufferPurpose::QuotientNumeratorLdeTile,
            1,
            256,
            ProofEpoch::Quotient,
            ProofEpoch::Quotient,
        ),
    ];
    let alias = ReleasedCommitmentAlias {
        commitment: CommitmentTreeId::Base,
        released_slab: LogicalBufferId(0),
        quotient_staging: LogicalBufferId(1),
        used_words: 192,
    };
    let plan = plan_range_arena_with_released_commitments(&logical, &[], &[alias]).unwrap();
    assert_eq!(binding(&plan, 0).physical, binding(&plan, 1).physical);
    assert_eq!(offset(&plan, 0), offset(&plan, 1));
    assert_eq!(
        plan.layout
            .slot(binding(&plan, 0).physical)
            .unwrap()
            .len_words,
        256
    );
}

#[test]
fn named_released_commitment_alias_rejects_swaps_overlap_and_extent_drift() {
    let valid = vec![
        buffer(
            0,
            BufferPurpose::CommitProgressiveStatePing,
            0,
            256,
            ProofEpoch::BaseCommit,
            ProofEpoch::BaseCommit,
        ),
        buffer(
            1,
            BufferPurpose::QuotientNumeratorLdeTile,
            1,
            256,
            ProofEpoch::Quotient,
            ProofEpoch::Quotient,
        ),
    ];
    let alias = ReleasedCommitmentAlias {
        commitment: CommitmentTreeId::Base,
        released_slab: LogicalBufferId(0),
        quotient_staging: LogicalBufferId(1),
        used_words: 192,
    };
    let mut swapped = alias;
    swapped.commitment = CommitmentTreeId::Interaction;
    assert!(plan_range_arena_with_released_commitments(&valid, &[], &[swapped]).is_err());

    let mut overlapping = valid.clone();
    overlapping[0].lifetime =
        BufferLifetime::new(ProofEpoch::BaseCommit, ProofEpoch::Quotient).unwrap();
    assert!(plan_range_arena_with_released_commitments(&overlapping, &[], &[alias]).is_err());

    let mut wrong_extent = valid;
    wrong_extent[1].len_words = 255;
    assert!(plan_range_arena_with_released_commitments(&wrong_extent, &[], &[alias]).is_err());
}

#[test]
fn identical_input_rebuilds_identical_ids_offsets_and_lengths() {
    let logical = vec![
        buffer(
            7,
            BufferPurpose::CommitLdeTile,
            7,
            65,
            ProofEpoch::Witness,
            ProofEpoch::BaseCommit,
        ),
        buffer(
            2,
            BufferPurpose::CommitLdeTile,
            2,
            33,
            ProofEpoch::Composition,
            ProofEpoch::Oods,
        ),
        buffer(
            11,
            BufferPurpose::CommitLdeTile,
            11,
            97,
            ProofEpoch::Fri,
            ProofEpoch::Assemble,
        ),
    ];
    let first = plan_range_arena(&logical, &[]).unwrap();
    let second = plan_range_arena(&logical, &[]).unwrap();

    assert_eq!(first.bindings, second.bindings);
    assert_eq!(first.layout.total_words(), second.layout.total_words());
    assert_eq!(first.high_water_words, second.high_water_words);
    for binding in &first.bindings {
        assert_eq!(
            first.layout.slot(binding.physical),
            second.layout.slot(binding.physical)
        );
    }
}
