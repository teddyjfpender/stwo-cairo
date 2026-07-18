use stwo_backend_cuda::RelationExecutionStage;

use super::*;
use crate::program_image::lower_compiled::relation_projection;
use crate::transcript_plan::CairoTranscriptOutput;

fn ready_post_interaction() -> (
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
    (executable, builder)
}

#[test]
fn generated_sn2_publishes_exact_relation_sequence_after_lookup_transcript() {
    let (executable, mut builder) = ready_post_interaction();
    let transcript = builder.interaction_transcript.as_ref().unwrap().lowered();
    assert!(builder.has_complete_interaction_transcript());
    assert_eq!(
        transcript.segment(),
        CairoTranscriptSegment::InteractionPowAndLookup
    );
    assert_eq!(transcript.operation_range(), &(11..13));
    assert_eq!(transcript.inputs().len(), 1);
    assert_eq!(transcript.outputs().len(), 1);
    assert!(transcript
        .output(CairoTranscriptOutput::CommonLookupElements)
        .is_some());

    let old_operations = builder.prefix.operations.len();
    let old_wrappers = builder.prefix.static_wrappers.len();
    let old_effects = builder.prefix.effects.len();
    let old_roots = builder.prefix.causal_external_roots.clone().unwrap();
    builder
        .append_relation_semantics(executable.arena(), executable.transcript())
        .unwrap();
    assert_eq!(builder.prefix.operations.len(), old_operations);
    assert_eq!(builder.prefix.static_wrappers.len(), old_wrappers);
    assert_eq!(builder.prefix.effects.len(), old_effects);

    let relation = builder.relation.as_ref().unwrap();
    assert_eq!(relation.lowered().wrappers().len(), 2);
    assert_eq!(
        relation.lowered().wrappers()[0].authority().stage,
        RelationExecutionStage::FusedBody
    );
    assert_eq!(
        relation.lowered().wrappers()[1].authority().stage,
        RelationExecutionStage::SegmentedTail
    );
    assert_eq!(relation.after(), &builder.prefix.values);

    builder
        .emit_relation_operations_using(
            executable.arena(),
            executable.transcript(),
            fake_relation_wrapper,
        )
        .unwrap();
    assert!(builder.has_complete_relation_stage());
    assert_eq!(builder.prefix.operations.len(), old_operations + 3);
    assert_eq!(builder.prefix.static_wrappers.len(), old_wrappers + 3);
    assert_eq!(builder.prefix.effects.len(), old_effects + 3);
    assert_eq!(
        builder.prefix.causal_external_roots.as_ref().unwrap().len(),
        old_roots.len() + 1
    );
    assert!(builder.prefix.operations[old_operations..]
        .iter()
        .all(|operation| operation.stage
            == ProofStage::BeforeTranscript(CairoTranscriptSegment::InteractionAndComposition)));
    assert_eq!(
        validate_causal_value_closure_with_sources(
            &builder.prefix.values,
            &builder.prefix.effects,
            &builder.prefix.operations,
            &builder.released_transcript_sources().unwrap(),
        )
        .unwrap(),
        builder.prefix.causal_external_roots.clone().unwrap()
    );
}

#[test]
fn missing_or_drifted_relation_authority_is_retryable_without_partial_publication() {
    let (executable, mut builder) = ready_post_interaction();
    builder
        .append_relation_semantics(executable.arena(), executable.transcript())
        .unwrap();
    let operations = builder.prefix.operations.clone();
    let wrappers = builder.prefix.static_wrappers.clone();
    let effects = builder.prefix.effects.clone();
    let roots = builder.prefix.causal_external_roots.clone();

    assert_eq!(
        builder.emit_relation_operations_using(
            executable.arena(),
            executable.transcript(),
            miss_fused_relation_wrapper,
        ),
        Err(CompiledBaseDagAppendError::MissingRelationStaticWrapper {
            operation_ordinal: 1,
        })
    );
    assert_eq!(builder.prefix.operations, operations);
    assert_eq!(builder.prefix.static_wrappers, wrappers);
    assert_eq!(builder.prefix.effects, effects);
    assert_eq!(builder.prefix.causal_external_roots, roots);
    assert!(!builder.has_complete_relation_stage());

    let root = *builder
        .prefix
        .causal_external_roots
        .as_ref()
        .unwrap()
        .iter()
        .next()
        .unwrap();
    builder
        .prefix
        .causal_external_roots
        .as_mut()
        .unwrap()
        .remove(&root);
    assert_eq!(
        builder.emit_relation_operations_using(
            executable.arena(),
            executable.transcript(),
            fake_relation_wrapper,
        ),
        Err(CompiledBaseDagAppendError::Lowering)
    );
    assert_eq!(builder.prefix.operations, operations);
    assert_eq!(builder.prefix.static_wrappers, wrappers);
    assert_eq!(builder.prefix.effects, effects);
    assert!(!builder.has_complete_relation_stage());

    builder.prefix.causal_external_roots = roots;
    builder
        .emit_relation_operations_using(
            executable.arena(),
            executable.transcript(),
            fake_relation_wrapper,
        )
        .unwrap();
    assert!(builder.has_complete_relation_stage());
}

fn fake_relation_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &relation_projection::LoweredRelation,
    ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    let (invocation, effect) = relation_operation(lowered, ordinal)?;
    let launch = StaticCudaLaunchIdentity::new(
        b"test_relation_kernel".to_vec(),
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    StaticCudaWrapperAuthority::new(
        id,
        [0x91; 32],
        target_sm,
        b"test_relation_wrapper".to_vec(),
        [0x92; 32],
        [0x93; 32],
        [0x94; 32],
        [0x95; 32],
        vec![launch],
        invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?,
        effect.id(),
    )
    .map(Some)
    .map_err(|_| InvocationShapeError::InvalidRelationAuthority)
}

fn miss_fused_relation_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &relation_projection::LoweredRelation,
    ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    if ordinal == 1 {
        Ok(None)
    } else {
        fake_relation_wrapper(id, target_sm, lowered, ordinal)
    }
}

fn relation_operation(
    lowered: &relation_projection::LoweredRelation,
    ordinal: usize,
) -> Result<
    (
        &crate::compiled_proof::AotInvocation,
        &crate::compiled_proof::EffectContract,
    ),
    InvocationShapeError,
> {
    match ordinal {
        0 => Ok((
            lowered.challenge().invocation(),
            lowered.challenge().effect(),
        )),
        1 | 2 => lowered
            .wrappers()
            .get(ordinal - 1)
            .map(|wrapper| (wrapper.invocation(), wrapper.effect()))
            .ok_or(InvocationShapeError::InvalidRelationBinding),
        _ => Err(InvocationShapeError::InvalidRelationBinding),
    }
}
