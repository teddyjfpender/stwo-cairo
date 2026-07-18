use std::sync::{Arc, OnceLock};

use stwo_backend_cuda::{
    BaseCommitProgramAuthority, InteractionCommitAuthorityError, InteractionCommitProgramAuthority,
};

use super::*;
use crate::arena_plan::{BufferLifetime, BufferPurpose, ProofEpoch};
use crate::compiled_proof::StaticCudaWrapperId;
use crate::shape_executable::ShapeExecutable;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    inventory: base_commit_projection::BaseCommitInventory,
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: LoweredInteractionCommit,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let planned = executable
            .arena()
            .commitment(CommitmentTreeId::Interaction)
            .unwrap();
        let authority = InteractionCommitProgramAuthority::compile(
            planned.commit_program.as_ref().unwrap(),
            planned.direct_retained_b2n_program.as_ref().unwrap(),
        )
        .unwrap();
        let inventory = base_commit_projection::BaseCommitInventory::compile_for(
            CommitInventoryKind::Interaction,
            executable.arena(),
            planned,
            authority.canonical(),
        )
        .unwrap();
        let before =
            adapter::SemanticValueMap::allocate_ordered(inventory.source_catalogs()).unwrap();
        let mut after = before.clone();
        let lowered = lower_stage(executable.arena(), executable.transcript(), &mut after).unwrap();
        Fixture {
            executable,
            inventory,
            before,
            after,
            lowered,
        }
    })
}

#[test]
fn generated_sn2_binds_every_operation_to_interaction_owned_values() {
    let fixture = fixture();
    let arena = fixture.executable.arena();
    let planned = arena.commitment(CommitmentTreeId::Interaction).unwrap();
    let authority = fixture.lowered.authority();
    assert_eq!(authority.role(), TraceTreeRole::Interaction);
    let mut kinds = [0usize; 8];
    for operation in authority.operations() {
        kinds[match operation.kind {
            stwo_backend_cuda::BaseCommitOperationKind::DirectB2n { .. } => 0,
            stwo_backend_cuda::BaseCommitOperationKind::DirectN2b { .. } => 1,
            stwo_backend_cuda::BaseCommitOperationKind::StateInit { .. } => 2,
            stwo_backend_cuda::BaseCommitOperationKind::StateExpandInPlace { .. } => 3,
            stwo_backend_cuda::BaseCommitOperationKind::StateAbsorb { .. } => 4,
            stwo_backend_cuda::BaseCommitOperationKind::StateFinalizeInPlace { .. } => 5,
            stwo_backend_cuda::BaseCommitOperationKind::MerkleLayerInPlace { .. } => 6,
            stwo_backend_cuda::BaseCommitOperationKind::MerkleLayer { .. } => 7,
        }] += 1;
    }
    assert_eq!(kinds, [55, 55, 1, 13, 55, 1, 3, 21]);
    assert_eq!(authority.operations().len(), 204);
    assert_eq!(
        fixture.lowered.operations().len(),
        authority.operations().len()
    );
    assert_ne!(fixture.lowered.digest(), [0; 32]);

    let catalog = BaseProducerCatalog::compile(arena).unwrap();
    for source in fixture.inventory.source_catalogs() {
        let value = catalog.value(source).unwrap();
        assert_eq!(value.purpose, BufferPurpose::InteractionTrace);
        assert_eq!(
            fixture.before.version(source).unwrap(),
            fixture.after.version(source).unwrap()
        );
    }
    let state_role = authority
        .layouts()
        .iter()
        .find_map(|layout| {
            matches!(
                layout.role,
                stwo_backend_cuda::BaseCommitValueRole::State { .. }
            )
            .then_some(layout.role)
        })
        .unwrap();
    let (_, state) = fixture.inventory.role(state_role).unwrap();
    let state_logical = &arena.logical_buffers()[state.logical.0 as usize];
    assert_eq!(
        state_logical.purpose,
        BufferPurpose::CommitProgressiveStatePing
    );
    assert_eq!(
        state_logical.lifetime,
        BufferLifetime::at(ProofEpoch::InteractionCommit)
    );

    let mut canonical = 0u32;
    for ((sources, outputs), logs) in planned
        .grouped_column_sources
        .iter()
        .zip(&planned.evaluation_output_groups)
        .zip(&planned.grouped_column_log_sizes)
    {
        let outputs = outputs.as_ref().unwrap();
        for ((source, output), log_size) in sources.iter().zip(outputs).zip(logs) {
            assert!(matches!(
                source,
                crate::arena_plan::CommitmentColumnSource::Trace {
                    purpose: BufferPurpose::InteractionCoefficients,
                    ..
                }
            ));
            let (_, retained) = fixture
                .inventory
                .role(stwo_backend_cuda::BaseCommitValueRole::RetainedEvaluation {
                    canonical_column: canonical,
                })
                .unwrap();
            assert_eq!(&retained, output);
            assert_eq!(
                retained.len_words,
                1usize << (log_size + planned.config.log_blowup_factor)
            );
            canonical += 1;
        }
    }
    assert_eq!(canonical as usize, authority.retained_evaluations().len());
}

#[test]
fn transcript_stage_seals_claim_root_and_next_challenge_in_order() {
    let fixture = fixture();
    let stage = fixture.lowered.stage();
    assert_eq!(
        stage.schedule_key,
        fixture.executable.transcript().schedule_key()
    );
    assert_eq!(
        usize::try_from(stage.interaction_claim_felts).unwrap() * 4,
        fixture
            .executable
            .transcript()
            .inputs()
            .iter()
            .find(|requirement| requirement.semantic == CairoTranscriptInput::InteractionClaim)
            .unwrap()
            .min_words
    );
    assert_eq!(stage.operation_end - stage.operation_start, 3);
    assert_eq!(stage.interaction_claim, stage.operation_start);
    assert_eq!(stage.interaction_root, stage.operation_start + 1);
    assert_eq!(
        stage.composition_random_coefficient,
        stage.operation_start + 2
    );

    let segment = fixture
        .executable
        .transcript()
        .segments()
        .iter()
        .find(|segment| segment.segment == CairoTranscriptSegment::InteractionAndComposition)
        .unwrap();
    let mut wrong_start = segment.clone();
    wrong_start.starts_after = Some(CairoTranscriptBoundary::BaseRoot);
    assert_eq!(
        InteractionCommitTranscriptStage::compile_segment(
            fixture.executable.transcript(),
            &wrong_start,
        ),
        Err(InvocationShapeError::InvalidInteractionCommitBinding)
    );
    let mut truncated = segment.clone();
    truncated.operation_range.end -= 1;
    assert_eq!(
        InteractionCommitTranscriptStage::compile_segment(
            fixture.executable.transcript(),
            &truncated,
        ),
        Err(InvocationShapeError::InvalidInteractionCommitBinding)
    );
}

#[test]
fn role_tree_and_epoch_substitution_fail_closed() {
    let fixture = fixture();
    let arena = fixture.executable.arena();
    let base = arena.commitment(CommitmentTreeId::Base).unwrap();
    let interaction = arena.commitment(CommitmentTreeId::Interaction).unwrap();
    assert_eq!(
        InteractionCommitProgramAuthority::compile(
            base.commit_program.as_ref().unwrap(),
            base.direct_retained_b2n_program.as_ref().unwrap(),
        ),
        Err(InteractionCommitAuthorityError::UnsupportedRole(
            TraceTreeRole::Base
        ))
    );
    assert_eq!(
        base_commit_projection::BaseCommitInventory::compile_for(
            CommitInventoryKind::Interaction,
            arena,
            base,
            fixture.lowered.authority().canonical(),
        )
        .unwrap_err(),
        InvocationShapeError::InvalidBaseCommitBinding
    );
    assert_eq!(
        base_commit_projection::BaseCommitInventory::compile_for(
            CommitInventoryKind::Base,
            arena,
            interaction,
            fixture.lowered.authority().canonical(),
        )
        .unwrap_err(),
        InvocationShapeError::InvalidBaseCommitBinding
    );
}

#[test]
fn projection_is_transactional_and_receipt_tamper_is_rejected() {
    let fixture = fixture();
    assert!(validate_from(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        &fixture.before,
        &fixture.after,
        &fixture.lowered,
    )
    .is_ok());

    let mut empty =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    let unchanged = empty.clone();
    assert!(lower_stage(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        &mut empty,
    )
    .is_err());
    assert_eq!(empty, unchanged);

    let mut reordered = fixture.lowered.clone();
    reordered.operations.swap(0, 1);
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            &fixture.before,
            &fixture.after,
            &reordered,
        ),
        Err(InvocationShapeError::InvalidInteractionCommitBinding)
    );

    let mut forged_stage = fixture.lowered.clone();
    forged_stage.stage.interaction_root ^= 1;
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            &fixture.before,
            &fixture.after,
            &forged_stage,
        ),
        Err(InvocationShapeError::InvalidInteractionCommitBinding)
    );

    let mut forged_digest = fixture.lowered.clone();
    forged_digest.digest[0] ^= 1;
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            &fixture.before,
            &fixture.after,
            &forged_digest,
        ),
        Err(InvocationShapeError::InvalidInteractionCommitBinding)
    );
}

#[test]
fn static_wrapper_binds_outer_interaction_link_not_canonical_base_link() {
    let fixture = fixture();
    let arena = fixture.executable.arena();
    let base_planned = arena.commitment(CommitmentTreeId::Base).unwrap();
    let base = BaseCommitProgramAuthority::compile(
        base_planned.commit_program.as_ref().unwrap(),
        base_planned.direct_retained_b2n_program.as_ref().unwrap(),
    )
    .unwrap();
    let role_bound = base_commit_projection::project_wrapper_parts(
        StaticCudaWrapperId(1),
        [7; 32],
        89,
        fixture.lowered.authority().identity(),
        fixture.lowered.authority().canonical(),
        fixture.lowered.operations(),
        0,
    )
    .unwrap();
    let canonical_substitute = base_commit_projection::project_wrapper_parts(
        StaticCudaWrapperId(1),
        [7; 32],
        89,
        base.identity(),
        fixture.lowered.authority().canonical(),
        fixture.lowered.operations(),
        0,
    )
    .unwrap();
    assert_eq!(
        role_bound.linked_module_identity(),
        &fixture.lowered.authority().identity()
    );
    assert_eq!(
        canonical_substitute.linked_module_identity(),
        &base.identity()
    );
    assert_ne!(role_bound, canonical_substitute);
    assert_ne!(role_bound.digest(), canonical_substitute.digest());

    let mut tested = false;
    for target_sm in [80, 86, 89, 90] {
        let Some(interaction_link) = fixture
            .lowered
            .authority()
            .bind_static_build(target_sm)
            .ok()
            .flatten()
        else {
            continue;
        };
        let base_link = base.bind_static_build(target_sm).unwrap().unwrap();
        let interaction_wrapper = resolve_static_wrapper(
            StaticCudaWrapperId(1),
            target_sm,
            arena,
            fixture.executable.transcript(),
            &fixture.lowered,
            0,
        )
        .unwrap()
        .unwrap();
        let canonical_substitute = base_commit_projection::project_wrapper_parts(
            StaticCudaWrapperId(1),
            base_link.module_build_identity(),
            base_link.target_sm(),
            base_link.identity(),
            fixture.lowered.authority().canonical(),
            fixture.lowered.operations(),
            0,
        )
        .unwrap();

        assert_eq!(
            interaction_wrapper.linked_module_identity(),
            &interaction_link.identity()
        );
        assert_eq!(
            canonical_substitute.linked_module_identity(),
            &base_link.identity()
        );
        assert_ne!(interaction_link.identity(), base_link.identity());
        assert_ne!(interaction_wrapper, canonical_substitute);
        assert_ne!(interaction_wrapper.digest(), canonical_substitute.digest());
        tested = true;
        break;
    }
    assert!(tested || !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT);
}
