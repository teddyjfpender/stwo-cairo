use super::*;
use crate::program_image::lower_compiled::{interaction_commit_projection, relation_projection};

fn ready_post_relation() -> (
    std::sync::Arc<crate::shape_executable::ShapeExecutable>,
    CompiledBaseDagBuilder,
) {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
    let mut builder = base_commit::ready_post_base(executable.arena());
    builder
        .append_bootstrap_root_stage(executable.arena(), executable.transcript())
        .unwrap();
    builder
        .append_bootstrap_transcript(executable.arena(), executable.transcript())
        .unwrap();
    builder
        .append_interaction_transcript(executable.arena(), executable.transcript())
        .unwrap();
    builder
        .append_relation_semantics(executable.arena(), executable.transcript())
        .unwrap();
    builder
        .emit_relation_operations_using(
            executable.arena(),
            executable.transcript(),
            fake_relation_wrapper,
        )
        .unwrap();
    (executable, builder)
}

#[test]
fn generated_sn2_publishes_exact_interaction_commit_transactionally() {
    let (executable, mut builder) = ready_post_relation();
    let old_operations = builder.prefix.operations.len();
    let old_wrappers = builder.prefix.static_wrappers.len();
    let old_effects = builder.prefix.effects.len();
    let roots = builder.prefix.causal_external_roots.clone();

    builder
        .append_interaction_semantics(executable.arena(), executable.transcript())
        .unwrap();
    let commit_operations = builder
        .interaction_stage
        .as_ref()
        .unwrap()
        .commit()
        .operations()
        .len();
    builder
        .emit_interaction_operations_using(
            executable.arena(),
            executable.transcript(),
            fake_interaction_commit_wrapper,
        )
        .unwrap();

    assert!(builder.has_complete_interaction_stage());
    assert_eq!(
        builder.prefix.operations.len(),
        old_operations + commit_operations + 2
    );
    assert_eq!(
        builder.prefix.static_wrappers.len(),
        old_wrappers + commit_operations
    );
    assert_eq!(
        builder.prefix.effects.len(),
        old_effects
            + commit_operations
            + builder
                .interaction_stage
                .as_ref()
                .unwrap()
                .claim()
                .child_effects()
                .len()
            + 2
    );
    assert_eq!(builder.prefix.causal_external_roots, roots);
    assert!(builder.prefix.operations[old_operations..]
        .iter()
        .all(|operation| operation.stage
            == ProofStage::BeforeTranscript(CairoTranscriptSegment::InteractionAndComposition)));
    let structural_sources = builder.released_transcript_sources().unwrap();
    assert_eq!(
        validate_causal_value_closure_with_sources(
            &builder.prefix.values,
            &builder.prefix.effects,
            &builder.prefix.operations,
            &structural_sources,
        )
        .unwrap(),
        roots.unwrap()
    );
}

#[test]
fn missing_interaction_wrapper_leaves_publication_retryable() {
    let (executable, mut builder) = ready_post_relation();
    builder
        .append_interaction_semantics(executable.arena(), executable.transcript())
        .unwrap();
    let operations = builder.prefix.operations.clone();
    let wrappers = builder.prefix.static_wrappers.clone();
    let effects = builder.prefix.effects.clone();
    let roots = builder.prefix.causal_external_roots.clone();

    assert!(matches!(
        builder.emit_interaction_operations_using(
            executable.arena(),
            executable.transcript(),
            miss_last_interaction_wrapper,
        ),
        Err(
            CompiledBaseDagAppendError::MissingInteractionCommitStaticWrapper {
                operation_ordinal
            }
        ) if operation_ordinal > 0
    ));
    assert_eq!(builder.prefix.operations, operations);
    assert_eq!(builder.prefix.static_wrappers, wrappers);
    assert_eq!(builder.prefix.effects, effects);
    assert_eq!(builder.prefix.causal_external_roots, roots);
    assert!(!builder.has_complete_interaction_stage());

    builder
        .emit_interaction_operations_using(
            executable.arena(),
            executable.transcript(),
            fake_interaction_commit_wrapper,
        )
        .unwrap();
    assert!(builder.has_complete_interaction_stage());
}

fn fake_relation_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &relation_projection::LoweredRelation,
    ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    let (invocation, effect) = match ordinal {
        0 => (
            lowered.challenge().invocation(),
            lowered.challenge().effect(),
        ),
        1 | 2 => {
            let wrapper = lowered
                .wrappers()
                .get(ordinal - 1)
                .ok_or(InvocationShapeError::InvalidRelationBinding)?;
            (wrapper.invocation(), wrapper.effect())
        }
        _ => return Err(InvocationShapeError::InvalidRelationBinding),
    };
    fake_wrapper(id, target_sm, b"test_relation_wrapper", invocation, effect)
}

fn fake_interaction_commit_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    _arena: &ProofArenaPlan,
    _transcript: &crate::transcript_plan::CairoBlake2sTranscriptPlan,
    lowered: &interaction_commit_projection::LoweredInteractionCommit,
    ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    let operation = lowered
        .operations()
        .get(ordinal)
        .ok_or(InvocationShapeError::InvalidInteractionCommitBinding)?;
    fake_wrapper(
        id,
        target_sm,
        b"test_interaction_commit_wrapper",
        operation.invocation(),
        operation.effect(),
    )
}

fn miss_last_interaction_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    arena: &ProofArenaPlan,
    transcript: &crate::transcript_plan::CairoBlake2sTranscriptPlan,
    lowered: &interaction_commit_projection::LoweredInteractionCommit,
    ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    if ordinal + 1 == lowered.operations().len() {
        Ok(None)
    } else {
        fake_interaction_commit_wrapper(id, target_sm, arena, transcript, lowered, ordinal)
    }
}

fn fake_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    symbol: &[u8],
    invocation: &crate::compiled_proof::AotInvocation,
    effect: &crate::compiled_proof::EffectContract,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    let launch = StaticCudaLaunchIdentity::new(
        symbol.to_vec(),
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidInteractionCommitAuthority)?;
    StaticCudaWrapperAuthority::new(
        id,
        [0x81; 32],
        target_sm,
        symbol.to_vec(),
        [0x82; 32],
        [0x83; 32],
        [0x84; 32],
        [0x85; 32],
        vec![launch],
        invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidInteractionCommitAuthority)?,
        effect.id(),
    )
    .map(Some)
    .map_err(|_| InvocationShapeError::InvalidInteractionCommitAuthority)
}
