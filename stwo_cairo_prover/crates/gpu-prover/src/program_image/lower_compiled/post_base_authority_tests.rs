//! Exact generated-SN2 post-Base shape and transcript receipt.
//!
//! This is deliberately not called a complete execution authority: outside the
//! Base commitment, several plans below seal geometry or traffic but not every
//! ABI, effect, invocation, and child operation needed by `CompiledProof`.

use stwo_backend_cuda::{
    blake_g_inputs_batch_is_exact, relation_batch_fused_eligible, DecommitTreeGeometry,
    DecommitTreeRequirements, FriFoldLaunchMode, ProgressiveCommitStorageMode, RelationLaunchMode,
    RelationTailMode, TraceTreeRole,
};

use crate::arena_plan::{
    BufferLifetime, BufferPurpose, CommitmentTreeId, ProofEpoch, QuotientNumeratorSchedule,
};
use crate::prepared_composition::{
    CompositionExecutionReceipt, CompositionLaunchMode, CompositionOutputMode,
};
use crate::transcript_plan::{
    CairoTranscriptBoundary as Boundary, CairoTranscriptInput as Input,
    CairoTranscriptOutput as Output, CairoTranscriptSegment as Segment,
};

#[test]
fn generated_sn2_post_base_shape_receipt_is_exact() {
    let executable = super::tests::generated_sn2_replacement();
    let arena = executable.arena();
    let protocol = arena.protocol_identity();

    let relation = arena.relation();
    let program = relation.execution.kernel_program();
    assert_eq!(relation.launch_mode, RelationLaunchMode::Fused);
    assert_eq!(protocol.relation_tail_mode, RelationTailMode::Segmented);
    assert_eq!(program.batches.len(), 68);
    assert_eq!(
        program
            .batches
            .iter()
            .map(|batch| batch.instances.len())
            .sum::<usize>(),
        45
    );
    assert_eq!(
        program
            .batches
            .iter()
            .map(|batch| batch.columns.len())
            .sum::<usize>(),
        807
    );
    assert_eq!(relation.execution.template_use_count, 1_566);
    assert_eq!(relation.source_plan.len(), 45);
    let relation_classes = program.batches.iter().fold(
        (0usize, 0usize, 0usize),
        |(fused, blake, fallback), batch| {
            let count = batch.instances.len();
            if blake_g_inputs_batch_is_exact(batch) {
                (fused, blake + count, fallback)
            } else if relation_batch_fused_eligible(batch) {
                (fused + count, blake, fallback)
            } else {
                (fused, blake, fallback + count)
            }
        },
    );
    assert_eq!(relation_classes, (45, 0, 0));

    let composition = arena.composition();
    assert_eq!(
        protocol.composition_launch_mode,
        CompositionLaunchMode::Wave
    );
    assert_eq!(composition.plan.components.len(), 45);
    assert_eq!(
        composition
            .plan
            .components
            .iter()
            .map(|component| component.kernels.len())
            .sum::<usize>(),
        123
    );
    assert_eq!(composition.plan.wave_kernels.len(), 14);
    assert_eq!(composition.plan.total_constraints, 1_053);
    assert_eq!(composition.plan.max_evaluation_log_size, 24);
    assert_eq!(composition.requirements.dynamic_ext_param_count, 4_782);
    assert_eq!(composition.requirements.claimed_sum_count, 45);
    assert_eq!(composition.requirements.accumulator_words, 78_577_280);
    assert_eq!(composition.requirements.accumulators.len(), 14);
    assert_eq!(composition.requirements.waves.len(), 14);
    assert_eq!(
        composition.requirements.execution_receipt,
        Some(CompositionExecutionReceipt {
            part_count: 123,
            wave_count: 14,
        })
    );
    assert_eq!(
        composition.output_plan.mode(),
        CompositionOutputMode::DirectRetainedEvaluations
    );
    let split = composition
        .output_plan
        .direct_program()
        .expect("SN2 replacement must retain direct Composition evaluations");
    assert_eq!(split.schedule().evaluation_log_size, 24);
    assert_eq!(split.schedule().inverse_intervals, 3);
    assert_eq!(split.traffic().fused_kernel_launches, 5);

    let oods = arena.oods();
    assert_eq!(oods.columns.len(), 4_524);
    let collapse = oods
        .pass_collapse
        .as_ref()
        .expect("SN2 replacement must select the OODS collapse");
    let receipt = collapse.receipt();
    assert_eq!(receipt.coefficient_group_count, 0);
    assert_eq!(receipt.evaluation_group_count, 28);
    assert_eq!(receipt.covered_evaluation_group_count, 28);
    assert_eq!(receipt.evaluation_sample_count, 4_673);
    assert_eq!(receipt.unchanged_coefficient_kernel_launches, 0);
    assert_eq!(receipt.unchanged_evaluation_kernel_launches, 84);
    assert_eq!(receipt.legacy_weight_kernel_launches, 112);
    assert_eq!(receipt.collapsed_weight_kernel_launches, 15);
    assert_eq!(receipt.legacy_total_kernel_launches, 196);
    assert_eq!(receipt.collapsed_total_kernel_launches, 99);
    assert_eq!(receipt.kernel_launches_removed, 97);
    assert_eq!(receipt.logical_bytes_removed, 4_400_328_576);
    assert_eq!(receipt.workspace_bytes_removed, 268_435_360);
    assert_eq!(receipt.retained_weight_bytes, 268_435_456);
    let (descriptor_offsets_logical, descriptor_offsets) = arena
        .find(None, None, BufferPurpose::OodsBarycentricScales, 0)
        .expect("collapsed OODS descriptor storage must be planned");
    assert_eq!(
        descriptor_offsets_logical.lifetime,
        BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble).unwrap()
    );
    assert_eq!(
        arena
            .bindings()
            .iter()
            .filter(|binding| binding.physical == descriptor_offsets.physical)
            .count(),
        1,
        "collapsed OODS descriptor storage must not physically alias"
    );
    assert_eq!(
        descriptor_offsets.len_words,
        collapse.collapsed_requirements().barycentric_scale_words
    );
    assert_eq!(descriptor_offsets.len_words, 28);

    let numerator = arena.quotient_numerator();
    assert_eq!(
        numerator.schedule,
        QuotientNumeratorSchedule::StagedRunSumOrPacked
    );
    assert_eq!(numerator.columns.len(), 4_524);
    assert_eq!(numerator.requirements.groups.len(), 15);
    assert_eq!(numerator.requirements.batches.len(), 14);
    assert_eq!(numerator.requirements.term_count, 4_853);
    let staged = numerator
        .staged_single_write
        .as_ref()
        .expect("SN2 replacement must select staged single-write numerator");
    assert!(staged.coefficient_ldes().is_empty());
    assert_eq!(staged.operations().len(), 1);
    let report = staged.report();
    assert_eq!(report.group_count, 15);
    assert_eq!(report.term_count, 4_853);
    assert_eq!(report.source_count, 4_493);
    assert_eq!(report.coefficient_source_count, 0);
    assert_eq!(report.output_rows, 18_210_768);
    assert_eq!(report.rectangular_launch_rows, 125_829_120);
    assert_eq!(report.inactive_rectangular_launch_rows, 107_618_352);
    assert_eq!(report.useful_row_terms, 39_242_841_600);

    let quotient = arena.quotient();
    assert_eq!(quotient.config.lifting_log_size, 24);
    assert_eq!(quotient.config.log_blowup_factor, 1);
    assert_eq!(quotient.requirements.subdomain_log_size, 23);
    assert_eq!(quotient.requirements.sample_count, 15);
    let producer_b2n = quotient
        .producer_b2n
        .as_ref()
        .expect("canonical SN2 must select the exact log23 producer/B2N fusion");
    let receipt = producer_b2n.receipt();
    assert_eq!(receipt.schedule.lifting_log_size, 24);
    assert_eq!(receipt.schedule.subdomain_log_size, 23);
    assert_eq!(receipt.schedule.sample_count, 15);
    assert_eq!(receipt.traffic.eliminated_kernel_launches, 21);
    assert_eq!(receipt.traffic.eliminated_logical_bytes, 5_637_144_576);

    let fri = arena.fri();
    assert_eq!(protocol.fri_fold_launch_mode, FriFoldLaunchMode::PerFold);
    assert_eq!(fri.config.circle_log_size, 24);
    assert_eq!(fri.config.fri.log_blowup_factor, 1);
    assert_eq!(fri.config.fri.log_last_layer_degree_bound, 0);
    assert_eq!(fri.config.fri.n_queries, 3);
    assert_eq!(fri.config.fri.fold_step, 1);
    assert_eq!(fri.requirements.last_layer_log_size, 1);
    assert_eq!(fri.requirements.trees.len(), 23);
    assert_eq!(fri.requirements.rounds.len(), 23);
    for (index, tree) in fri.requirements.trees.iter().enumerate() {
        assert_eq!(tree.evaluation_log_size, 24 - index as u32);
        assert_eq!(tree.outgoing_fold_step, 1);
        assert_eq!(tree.log_rows_per_leaf, 0);
    }
    for (index, round) in fri.requirements.rounds.iter().enumerate() {
        assert_eq!(round.input_log_size, 24 - index as u32);
        assert_eq!(round.fold_step, 1);
        assert_eq!(round.output_log_size, 23 - index as u32);
        assert_eq!(round.output_tree, (index < 22).then_some(index + 1));
    }
    assert_eq!(
        fri.requirements
            .trees
            .iter()
            .map(|tree| tree.layers_bottom_up.len())
            .sum::<usize>(),
        322
    );
    assert_eq!((fri.requirements.trees.len() - 1) * 4, 88);

    let final_pow = arena.final_fri_pow();
    assert_eq!(final_pow.final_requirements.evaluation_log_size, 1);
    assert_eq!(final_pow.final_requirements.evaluation_words, 8);
    assert_eq!(final_pow.final_requirements.coefficient_words, 8);
    assert_eq!(final_pow.final_requirements.transcript_words, 4);
    assert_eq!(
        final_pow.final_requirements.inverse_twiddle_words,
        8_388_608
    );
    assert_eq!(final_pow.interaction_pow_bits, 24);
    assert_eq!(final_pow.query_pow_bits, 10);
    let pow = final_pow.pow_requirements;
    assert_eq!(
        (
            pow.state_words,
            pow.nonce_words,
            pow.best_nonce_words,
            pow.completed_blocks_words,
            pow.prefix_digest_words,
        ),
        (16, 2, 2, 1, 8)
    );

    let decommit = arena.decommit();
    assert_eq!(decommit.config.query_log_size, 24);
    assert_eq!(decommit.config.n_queries, 3);
    assert_eq!(decommit.config.trees.len(), 27);
    assert_eq!(decommit.requirements.trees.len(), 27);
    assert_eq!(
        (
            decommit.requirements.unique_query_words,
            decommit.requirements.mapped_query_words,
            decommit.requirements.walk_query_words,
            decommit.requirements.expanded_position_words,
            decommit.requirements.sparse_index_words,
            decommit.requirements.sparse_hash_words,
            decommit.requirements.count_words,
            decommit.requirements.assembly_words,
        ),
        (3, 3, 6, 6, 90, 720, 8, 73_805)
    );
    let mut trace_receipt = Vec::new();
    let mut derived_trace_wrapper_calls = 0usize;
    for (geometry, requirements) in decommit
        .config
        .trees
        .iter()
        .zip(&decommit.requirements.trees)
    {
        match (geometry, requirements) {
            (
                DecommitTreeGeometry::Trace(geometry),
                DecommitTreeRequirements::Trace(requirements),
            ) => {
                let lde_batches = requirements
                    .groups
                    .iter()
                    .map(|group| group.lde_batches.len())
                    .sum::<usize>();
                let wrapper_calls = 2
                    + lde_batches
                    + requirements.groups.len()
                    + requirements.groups.len()
                        * usize::from(geometry.unretained_bottom_layers > 0)
                    + geometry.unretained_bottom_layers.saturating_sub(1) as usize;
                derived_trace_wrapper_calls += wrapper_calls;
                trace_receipt.push((
                    geometry.role,
                    requirements.groups.len(),
                    requirements.column_count,
                    geometry.unretained_bottom_layers,
                    lde_batches,
                    wrapper_calls,
                ));
            }
            (DecommitTreeGeometry::Fri(_), DecommitTreeRequirements::Fri(_)) => {}
            _ => panic!("decommit geometry and requirements must agree"),
        }
    }
    assert_eq!(
        trace_receipt,
        [
            (TraceTreeRole::Preprocessed, 11, 161, 4, 0, 27),
            (TraceTreeRole::Base, 165, 2_627, 4, 0, 335),
            (TraceTreeRole::Interaction, 108, 1_728, 4, 0, 221),
            (TraceTreeRole::Composition, 1, 8, 4, 0, 7),
        ]
    );
    assert_eq!(derived_trace_wrapper_calls, 590);
    for (index, (geometry, requirements)) in decommit
        .config
        .trees
        .iter()
        .zip(&decommit.requirements.trees)
        .skip(4)
        .enumerate()
    {
        let (DecommitTreeGeometry::Fri(geometry), DecommitTreeRequirements::Fri(_)) =
            (geometry, requirements)
        else {
            panic!("FRI decommit trees must follow the four trace trees");
        };
        assert_eq!(geometry.fri_tree_index, index as u32);
        assert_eq!(geometry.evaluation_log_size, 24 - index as u32);
        assert_eq!(geometry.cumulative_fold, index as u32);
        assert_eq!(geometry.outgoing_fold_step, 1);
        assert_eq!(geometry.log_rows_per_leaf, 0);
    }
    // One normalize wrapper + 590 trace wrappers + prepare/assemble per FRI tree.
    assert_eq!(1 + derived_trace_wrapper_calls + 2 * 23, 637);

    let commitments: Vec<_> = arena
        .commitments()
        .iter()
        .map(|commitment| {
            (
                commitment.id,
                commitment.storage_mode,
                commitment
                    .commit_program
                    .as_ref()
                    .map_or(0, |program| program.steps().len()),
                commitment.direct_retained_b2n_program.is_some(),
                commitment.interpolation_batches.len(),
                commitment
                    .retained_evaluation_groups
                    .iter()
                    .flatten()
                    .map(Vec::len)
                    .sum::<usize>(),
                commitment.retained_layers_bottom_up.len(),
            )
        })
        .collect();
    assert_eq!(
        commitments,
        [
            (
                CommitmentTreeId::Preprocessed,
                ProgressiveCommitStorageMode::InPlaceSlab,
                86,
                false,
                0,
                161,
                23,
            ),
            (
                CommitmentTreeId::Base,
                ProgressiveCommitStorageMode::InPlaceSlab,
                96,
                true,
                0,
                2_627,
                21,
            ),
            (
                CommitmentTreeId::Interaction,
                ProgressiveCommitStorageMode::InPlaceSlab,
                98,
                true,
                0,
                1_728,
                21,
            ),
            (
                CommitmentTreeId::Composition,
                ProgressiveCommitStorageMode::InPlaceSlab,
                17,
                false,
                0,
                8,
                21,
            ),
        ]
    );

    assert_transcript_receipt(&executable);
    assert_cross_stage_aliases(&executable);
    assert_proof_bundle_layout(&decommit.proof_bundle_layout);
}

fn assert_transcript_receipt(executable: &crate::shape_executable::ShapeExecutable) {
    let transcript = executable.transcript();
    assert_eq!(transcript.schedule().operations().len(), 69);

    let mut expected_inputs = vec![
        (Input::ChannelSalt, 4),
        (Input::PcsConfig, 8),
        (Input::PreprocessedRoot, 8),
        (Input::ClaimComponentCount, 4),
        (Input::ClaimEnableBits, 84),
        (Input::ClaimLogSizes, 48),
        (Input::ClaimProgramLength, 4),
        (Input::ClaimPublicData, 584),
        (Input::ClaimOutputRoot, 8),
        (Input::ClaimProgramRoot, 8),
        (Input::BaseRoot, 8),
        (Input::InteractionPowNonce, 2),
        (Input::InteractionClaim, 180),
        (Input::InteractionRoot, 8),
        (Input::CompositionRoot, 8),
        (Input::OodsSampledValues, 18_692),
    ];
    expected_inputs.extend((0..23).map(|index| (Input::FriLayerRoot(index), 8)));
    expected_inputs.extend([
        (Input::FriLastLayerPolynomial, 4),
        (Input::QueryPowNonce, 2),
    ]);
    assert_eq!(
        transcript
            .inputs()
            .iter()
            .map(|input| (input.semantic, input.min_words))
            .collect::<Vec<_>>(),
        expected_inputs
    );

    let mut expected_outputs = vec![
        (Output::CommonLookupElements, 8),
        (Output::CompositionRandomCoefficient, 4),
        (Output::OodsPointParameter, 4),
        (Output::QuotientRandomCoefficient, 4),
    ];
    expected_outputs.extend((0..23).map(|index| (Output::FriFoldingChallenge(index), 4)));
    expected_outputs.push((Output::QueryPositions, 3));
    assert_eq!(
        transcript
            .outputs()
            .iter()
            .map(|output| (output.semantic, output.min_words))
            .collect::<Vec<_>>(),
        expected_outputs
    );

    let segments = transcript.segments();
    assert_eq!(segments.len(), 30);
    assert_eq!(
        (
            segments[0].segment,
            segments[0].operation_range.clone(),
            segments[0].starts_after,
            segments[0].ends_at,
        ),
        (
            Segment::BootstrapThroughBase,
            0..11,
            None,
            Boundary::BaseRoot
        )
    );
    assert_eq!(
        (
            segments[1].segment,
            segments[1].operation_range.clone(),
            segments[1].starts_after,
            segments[1].ends_at,
        ),
        (
            Segment::InteractionPowAndLookup,
            11..13,
            Some(Boundary::BaseRoot),
            Boundary::CommonLookupElements,
        )
    );
    assert_eq!(
        (
            segments[2].segment,
            segments[2].operation_range.clone(),
            segments[2].starts_after,
            segments[2].ends_at,
        ),
        (
            Segment::InteractionAndComposition,
            13..16,
            Some(Boundary::CommonLookupElements),
            Boundary::CompositionRandomCoefficient,
        )
    );
    assert_eq!(
        (
            segments[3].segment,
            segments[3].operation_range.clone(),
            segments[3].starts_after,
            segments[3].ends_at,
        ),
        (
            Segment::CompositionAndOods,
            16..18,
            Some(Boundary::CompositionRandomCoefficient),
            Boundary::OodsPoint,
        )
    );
    assert_eq!(
        (
            segments[4].segment,
            segments[4].operation_range.clone(),
            segments[4].starts_after,
            segments[4].ends_at,
        ),
        (
            Segment::OodsAndQuotient,
            18..20,
            Some(Boundary::OodsPoint),
            Boundary::QuotientRandomCoefficient,
        )
    );
    for index in 0..23 {
        let segment = &segments[5 + index];
        assert_eq!(segment.segment, Segment::FriLayer(index as u32));
        assert_eq!(segment.operation_range, 20 + 2 * index..22 + 2 * index);
        assert_eq!(
            segment.starts_after,
            Some(if index == 0 {
                Boundary::QuotientRandomCoefficient
            } else {
                Boundary::FriFoldingChallenge(index as u32 - 1)
            })
        );
        assert_eq!(segment.ends_at, Boundary::FriFoldingChallenge(index as u32));
    }
    assert_eq!(
        (
            segments[28].segment,
            segments[28].operation_range.clone(),
            segments[28].starts_after,
            segments[28].ends_at,
        ),
        (
            Segment::FriLastLayer,
            66..67,
            Some(Boundary::FriFoldingChallenge(22)),
            Boundary::FriLastLayerPolynomial,
        )
    );
    assert_eq!(
        (
            segments[29].segment,
            segments[29].operation_range.clone(),
            segments[29].starts_after,
            segments[29].ends_at,
        ),
        (
            Segment::QueryPowAndPositions,
            67..69,
            Some(Boundary::FriLastLayerPolynomial),
            Boundary::QueryPositions,
        )
    );
}

fn assert_cross_stage_aliases(executable: &crate::shape_executable::ShapeExecutable) {
    let arena = executable.arena();
    let transcript = arena.transcript();
    let transcript_input = |semantic: Input| {
        let id = semantic.id().expect("valid Cairo transcript input");
        transcript
            .inputs
            .iter()
            .find_map(|(candidate, binding)| (*candidate == id).then_some(*binding))
            .expect("planned transcript input")
    };
    let transcript_output = |semantic: Output| {
        let id = semantic.id().expect("valid Cairo transcript output");
        transcript
            .outputs
            .iter()
            .find_map(|(candidate, binding)| (*candidate == id).then_some(*binding))
            .expect("planned transcript output")
    };

    let composition = arena.composition();
    let oods = arena.oods();
    let numerator = arena.quotient_numerator();
    let quotient = arena.quotient();
    let fri = arena.fri();
    let decommit = arena.decommit();
    assert_eq!(
        composition.random_coefficient,
        transcript_output(Output::CompositionRandomCoefficient)
    );
    assert_eq!(
        oods.oods_point_parameter,
        transcript_output(Output::OodsPointParameter)
    );
    assert_eq!(
        oods.sampled_values,
        transcript_input(Input::OodsSampledValues)
    );
    assert_eq!(numerator.oods_sample_points, oods.sample_points);
    assert_eq!(numerator.oods_sampled_values, oods.sampled_values);
    assert_eq!(
        numerator.random_coefficient,
        transcript_output(Output::QuotientRandomCoefficient)
    );
    assert_eq!(numerator.sample_points_destination, quotient.sample_points);
    assert_eq!(
        numerator.first_linear_terms_destination,
        quotient.first_linear_terms
    );
    assert_eq!(
        numerator.destinations.len(),
        quotient.partial_numerators.len()
    );
    for (numerator, quotient) in numerator
        .destinations
        .iter()
        .zip(&quotient.partial_numerators)
    {
        assert_eq!(numerator.log_size, quotient.log_size);
        assert_eq!(numerator.coordinates, quotient.coordinates);
    }
    assert_eq!(quotient.output_values, fri.input_values);
    assert_eq!(
        decommit.raw_queries,
        transcript_output(Output::QueryPositions)
    );
}

fn assert_proof_bundle_layout(layout: &crate::proof_bundle::ResidentProofBundleLayout) {
    assert_eq!(layout.commitments, 0..32);
    assert_eq!(layout.interaction_claim, 32..212);
    assert_eq!(layout.interaction_pow, 212..214);
    assert_eq!(layout.sampled_values, 214..18_906);
    assert_eq!(layout.fri_commitments, 18_906..19_090);
    assert_eq!(layout.final_line_poly, 19_090..19_094);
    assert_eq!(layout.query_pow, 19_094..19_096);
    assert_eq!(layout.decommitment, 19_096..92_901);
    assert_eq!(layout.total_words, 92_901);
}
