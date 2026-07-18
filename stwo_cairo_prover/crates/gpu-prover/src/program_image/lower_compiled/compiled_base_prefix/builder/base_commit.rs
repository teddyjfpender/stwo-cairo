use super::*;
use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::{
    ExecutionPrimitive, LaunchGeometry, ProofStage, StaticCudaLaunchIdentity,
    StaticCudaWrapperAuthority, StaticCudaWrapperId,
};
use crate::program_image::lower_compiled::base_commit_projection;
use crate::transcript_plan::CairoTranscriptSegment;

fn ready_post_fixed(arena: &ProofArenaPlan) -> CompiledBaseDagBuilder {
    let mut builder = CompiledBaseDagBuilder::from_complete_witness(complete_prefix()).unwrap();
    builder.append_memory_semantics(arena).unwrap();
    builder
        .emit_memory_operations_using(fake_memory_wrapper)
        .unwrap();
    builder.append_fixed_table_semantics(arena).unwrap();
    builder
        .emit_fixed_table_operations_using(fake_fixed_table_wrapper)
        .unwrap();
    assert_eq!(builder.prefix.operations.len(), 102);
    assert_eq!(builder.prefix.static_wrappers.len(), 71);
    builder
}

pub(super) fn ready_post_base(arena: &ProofArenaPlan) -> CompiledBaseDagBuilder {
    let mut builder = ready_post_fixed(arena);
    builder.append_base_commit_semantics(arena).unwrap();
    builder
        .emit_base_commit_operations_using(arena, fake_base_commit_wrapper)
        .unwrap();
    builder
}

fn fake_base_commit_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &base_commit_projection::LoweredBaseCommit,
    operation_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    let operation = lowered
        .operations()
        .get(operation_ordinal)
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    let launch = StaticCudaLaunchIdentity::new(
        b"test_base_commit_kernel".to_vec(),
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?;
    StaticCudaWrapperAuthority::new(
        id,
        [0x71; 32],
        target_sm,
        b"test_base_commit_wrapper".to_vec(),
        [0x72; 32],
        [0x73; 32],
        [0x74; 32],
        [0x75; 32],
        vec![launch],
        operation
            .invocation()
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?,
        operation.effect().id(),
    )
    .map(Some)
    .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)
}

fn miss_last_base_commit_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &base_commit_projection::LoweredBaseCommit,
    operation_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    if operation_ordinal + 1 == lowered.operations().len() {
        Ok(None)
    } else {
        fake_base_commit_wrapper(id, target_sm, lowered, operation_ordinal)
    }
}

#[test]
fn generated_sn2_publishes_exact_300_operation_base_checkpoint() {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
    let mut builder = ready_post_fixed(executable.arena());
    let operations = builder.prefix.operations.clone();
    let wrappers = builder.prefix.static_wrappers.clone();
    let effects = builder.prefix.effects.clone();
    let roots = builder.prefix.causal_external_roots.clone().unwrap();
    let before = builder.prefix.values.clone();

    assert_eq!(
        builder.emit_base_commit_operations(executable.arena()),
        Err(CompiledBaseDagAppendError::InvalidStage)
    );
    builder
        .append_base_commit_semantics(executable.arena())
        .unwrap();
    assert_eq!(builder.prefix.operations, operations);
    assert_eq!(builder.prefix.static_wrappers, wrappers);
    assert_eq!(builder.prefix.effects, effects);
    assert_eq!(builder.prefix.causal_external_roots.as_ref(), Some(&roots));

    let sealed = builder.base_commit.as_ref().unwrap();
    assert_ne!(builder.prefix.values, before);
    assert_eq!(sealed.lowered.operations().len(), 198);
    assert_eq!(
        sealed.lowered.digest(),
        decode32("51f1f517f661575765ba886c3031dee25dde222390c206551c56c31d04e91e7e")
    );
    assert_eq!(sealed.external_roots.len(), 2);
    assert!(!sealed.emitted);
    assert_eq!(sealed.checkpoint_digest, None);

    builder
        .emit_base_commit_operations_using(executable.arena(), fake_base_commit_wrapper)
        .unwrap();
    let sealed = builder.base_commit.as_ref().unwrap();
    assert!(builder.has_complete_base_commit_stage());
    assert!(sealed.emitted);
    assert_eq!(builder.prefix.operations.len(), 300);
    assert_eq!(builder.prefix.static_wrappers.len(), 269);
    assert_eq!(builder.prefix.effects.len(), 300);
    assert_eq!(
        builder
            .prefix
            .operations
            .iter()
            .filter(|operation| {
                operation.stage
                    == ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
            })
            .count(),
        300
    );

    for (ordinal, local) in sealed.lowered.operations().iter().enumerate() {
        let operation = &builder.prefix.operations[102 + ordinal];
        let wrapper = &builder.prefix.static_wrappers[71 + ordinal];
        assert_eq!(operation.id.0 as usize, 102 + ordinal);
        assert_eq!(operation.semantic_id.0 as usize, 103 + ordinal);
        assert_eq!(wrapper.id().0 as usize, 72 + ordinal);
        assert_eq!(
            operation.primitive,
            ExecutionPrimitive::StaticCudaWrapper {
                wrapper: wrapper.id()
            }
        );
        assert_eq!(operation.invocation.as_ref(), Some(local.invocation()));
        assert_eq!(operation.effect, local.effect().id());
        assert_eq!(wrapper.accepted_effect(), local.effect().id());
        assert_eq!(
            wrapper.accepted_invocation(),
            local.invocation().contract_id().unwrap()
        );
        assert_eq!(wrapper.consumer_target_sm(), TARGET_SM);
        assert!(builder
            .prefix
            .effects
            .iter()
            .any(|effect| effect == local.effect()));
    }

    let actual_roots = validate_causal_value_closure(
        &builder.prefix.values,
        &builder.prefix.effects,
        &builder.prefix.operations,
    )
    .unwrap();
    assert_eq!(actual_roots.len(), 90);
    assert_eq!(actual_roots.len(), roots.len() + 2);
    assert!(sealed
        .external_roots
        .iter()
        .all(|root| actual_roots.contains(root)));
    assert_eq!(
        builder.prefix.causal_external_roots.as_ref(),
        Some(&actual_roots)
    );
    assert_eq!(
        sealed.checkpoint_digest,
        Some(decode32(
            "ba78d646298225c05755bbdcc6f10df4c2d30a9295de0a595ccabd2340a3c20d"
        ))
    );
    assert_eq!(
        builder.emit_base_commit_operations(executable.arena()),
        Err(CompiledBaseDagAppendError::InvalidStage)
    );
}

#[test]
fn base_commit_publication_rolls_back_missing_authority_and_tampering() {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
    let mut missing = ready_post_fixed(executable.arena());
    missing
        .append_base_commit_semantics(executable.arena())
        .unwrap();
    let operations = missing.prefix.operations.clone();
    let wrappers = missing.prefix.static_wrappers.clone();
    let effects = missing.prefix.effects.clone();
    let roots = missing.prefix.causal_external_roots.clone();
    let values = missing.prefix.values.clone();
    assert_eq!(
        missing
            .emit_base_commit_operations_using(executable.arena(), miss_last_base_commit_wrapper,),
        Err(CompiledBaseDagAppendError::MissingBaseCommitStaticWrapper {
            operation_ordinal: 197,
        })
    );
    assert_eq!(missing.prefix.operations, operations);
    assert_eq!(missing.prefix.static_wrappers, wrappers);
    assert_eq!(missing.prefix.effects, effects);
    assert_eq!(missing.prefix.causal_external_roots, roots);
    assert_eq!(missing.prefix.values, values);
    assert!(!missing.base_commit.as_ref().unwrap().emitted);
    assert_eq!(
        missing.base_commit.as_ref().unwrap().checkpoint_digest,
        None
    );
    base_commit_projection::tamper_receipt_digest_for_test(
        &mut missing.base_commit.as_mut().unwrap().lowered,
    );
    assert_eq!(
        missing.emit_base_commit_operations_using(executable.arena(), fake_base_commit_wrapper),
        Err(CompiledBaseDagAppendError::Lowering)
    );
    assert_eq!(missing.prefix.operations, operations);
    assert_eq!(missing.prefix.static_wrappers, wrappers);
    assert_eq!(missing.prefix.effects, effects);
    assert_eq!(missing.prefix.causal_external_roots, roots);
    assert!(!missing.base_commit.as_ref().unwrap().emitted);
    assert_eq!(
        missing.base_commit.as_ref().unwrap().checkpoint_digest,
        None
    );
}

fn decode32(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64);
    let mut bytes = [0u8; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
    }
    bytes
}
