use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

use super::*;
use crate::compiled_proof::{
    ExecutionPrimitive, LaunchGeometry, ProofStage, StaticCudaLaunchIdentity,
    StaticCudaWrapperAuthority,
};
use crate::program_image::lower_compiled::compiled_base_prefix::{
    emit_recorded_witness_writer_prefix_for_test, CompiledWitnessWriterPrefix,
};
use crate::program_image::lower_compiled::compiled_base_prefix_tests::{
    exact_fields, resolve_all_static_for_prefix, MANIFEST, TARGET_SM,
};
use crate::transcript_plan::CairoTranscriptSegment;

mod base_commit;
mod interaction_stage;
mod relation_stage;

#[test]
fn generated_sn2_releases_the_exact_bootstrap_transcript_after_base() {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
    let mut builder = base_commit::ready_post_base(executable.arena());
    let values = builder.prefix.values.clone();
    let operations = builder.prefix.operations.clone();
    let effects = builder.prefix.effects.clone();
    let wrappers = builder.prefix.static_wrappers.clone();
    let roots = builder.prefix.causal_external_roots.clone().unwrap();

    builder
        .append_bootstrap_root_stage(executable.arena(), executable.transcript())
        .unwrap();
    assert!(builder.has_complete_bootstrap_root_stage());
    assert_eq!(
        builder.prefix.values.entries().count(),
        values.entries().count() + 3
    );
    assert_eq!(builder.prefix.operations.len(), operations.len() + 2);
    assert_eq!(builder.prefix.effects.len(), effects.len() + 2);
    assert_eq!(builder.prefix.static_wrappers, wrappers);
    assert_eq!(
        builder.prefix.causal_external_roots.as_ref().unwrap().len(),
        roots.len() + 1
    );
    assert!(builder.prefix.operations[operations.len()..]
        .iter()
        .all(|operation| {
            operation.primitive == ExecutionPrimitive::DeviceCopyD2D { bytes: 32 }
                && operation.stage
                    == ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
        }));

    builder
        .append_bootstrap_transcript(executable.arena(), executable.transcript())
        .unwrap();
    let sealed = builder.bootstrap_transcript.as_ref().unwrap();
    let lowered = sealed.lowered();
    assert!(builder.has_complete_bootstrap_transcript());
    assert_eq!(
        lowered.segment(),
        CairoTranscriptSegment::BootstrapThroughBase
    );
    assert_eq!(lowered.operation_range(), &(0..11));
    assert_eq!(lowered.inputs().len(), 11);
    assert!(lowered.outputs().is_empty());
    assert_eq!(
        builder.prefix.values.entries().count(),
        values.entries().count() + 12
    );
    for semantic in crate::transcript_plan::CAIRO_STATIC_TRANSCRIPT_INPUTS {
        let id = semantic.id().unwrap();
        let logical = executable
            .arena()
            .transcript()
            .inputs
            .iter()
            .find_map(|(candidate, binding)| (*candidate == id).then_some(binding.logical))
            .unwrap();
        let catalog = crate::program_image::ArenaCatalogValueId(logical.0);
        assert!(values.version(catalog).is_err());
        assert!(builder.prefix.values.version(catalog).is_ok());
    }
    assert_eq!(
        builder.prefix.causal_external_roots.as_ref().unwrap().len(),
        roots.len() + 10
    );
    assert_eq!(&builder.prefix.operations[..operations.len()], operations);
    assert_eq!(builder.prefix.static_wrappers, wrappers);
    assert_eq!(
        builder.append_bootstrap_transcript(executable.arena(), executable.transcript()),
        Err(CompiledBaseDagAppendError::InvalidStage)
    );
}

#[test]
fn bootstrap_release_rejects_missing_or_tampered_base_transactionally() {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
    let mut missing = CompiledBaseDagBuilder::from_complete_witness(complete_prefix()).unwrap();
    let values = missing.prefix.values.clone();
    assert_eq!(
        missing.append_bootstrap_transcript(executable.arena(), executable.transcript()),
        Err(CompiledBaseDagAppendError::InvalidStage)
    );
    assert_eq!(missing.prefix.values, values);
    assert!(missing.bootstrap_transcript.is_none());

    let mut missing_root = base_commit::ready_post_base(executable.arena());
    missing_root
        .append_bootstrap_root_stage(executable.arena(), executable.transcript())
        .unwrap();
    let base_roots = missing_root
        .base_commit
        .as_ref()
        .unwrap()
        .causal_roots
        .as_ref()
        .unwrap();
    let prepared_root = *missing_root
        .prefix
        .causal_external_roots
        .as_ref()
        .unwrap()
        .difference(base_roots)
        .next()
        .unwrap();
    missing_root
        .prefix
        .causal_external_roots
        .as_mut()
        .unwrap()
        .remove(&prepared_root);
    let values = missing_root.prefix.values.clone();
    let operations = missing_root.prefix.operations.clone();
    assert_eq!(
        missing_root.append_bootstrap_transcript(executable.arena(), executable.transcript()),
        Err(CompiledBaseDagAppendError::Lowering)
    );
    assert_eq!(missing_root.prefix.values, values);
    assert_eq!(missing_root.prefix.operations, operations);
    assert!(missing_root.bootstrap_transcript.is_none());

    let mut tampered = base_commit::ready_post_base(executable.arena());
    tampered
        .base_commit
        .as_mut()
        .unwrap()
        .checkpoint_digest
        .as_mut()
        .unwrap()[0] ^= 1;
    let values = tampered.prefix.values.clone();
    assert_eq!(
        tampered.append_bootstrap_root_stage(executable.arena(), executable.transcript()),
        Err(CompiledBaseDagAppendError::Lowering)
    );
    assert_eq!(tampered.prefix.values, values);
    assert!(tampered.bootstrap_roots.is_none());
    assert!(tampered.bootstrap_transcript.is_none());
}

fn complete_prefix() -> CompiledWitnessWriterPrefix {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
    emit_recorded_witness_writer_prefix_for_test(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
        MANIFEST,
        TARGET_SM,
        |source| Ok(exact_fields(source)),
        resolve_all_static_for_prefix,
    )
    .unwrap()
}

fn fake_memory_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &memory_base_trace::LoweredMemoryBaseTrace,
    step_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    let step = lowered
        .steps()
        .get(step_ordinal)
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?;
    let launch = StaticCudaLaunchIdentity::new(
        b"test_memory_base_kernel".to_vec(),
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    StaticCudaWrapperAuthority::new(
        id,
        [0x51; 32],
        target_sm,
        b"test_memory_base_wrapper".to_vec(),
        [0x52; 32],
        [0x53; 32],
        [0x54; 32],
        [0x55; 32],
        vec![launch],
        step.invocation()
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?,
        step.effect().id(),
    )
    .map(Some)
    .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)
}

fn miss_second_memory_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &memory_base_trace::LoweredMemoryBaseTrace,
    step_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    if step_ordinal == 1 {
        Ok(None)
    } else {
        fake_memory_wrapper(id, target_sm, lowered, step_ordinal)
    }
}

fn fake_fixed_table_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &fixed_table_materialization::LoweredFixedTableStage,
    table_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    let table = lowered
        .tables()
        .get(table_ordinal)
        .ok_or(InvocationShapeError::InvalidFixedTableBinding)?;
    let launch = StaticCudaLaunchIdentity::new(
        b"test_fixed_table_kernel".to_vec(),
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
    StaticCudaWrapperAuthority::new(
        id,
        [0x61; 32],
        target_sm,
        b"test_fixed_table_wrapper".to_vec(),
        [0x62; 32],
        [0x63; 32],
        [0x64; 32],
        [0x65; 32],
        vec![launch],
        table
            .invocation()
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?,
        table.effect().id(),
    )
    .map(Some)
    .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)
}

fn miss_second_fixed_table_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &fixed_table_materialization::LoweredFixedTableStage,
    table_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    if table_ordinal == 1 {
        Ok(None)
    } else {
        fake_fixed_table_wrapper(id, target_sm, lowered, table_ordinal)
    }
}

#[test]
fn cursor_rejects_an_incomplete_witness_prefix_without_losing_it() {
    let mut prefix = complete_prefix();
    prefix.next_producer -= 1;
    let operation_count = prefix.operations.len();
    let CompiledBaseDagStartError::IncompleteWitnessPrefix(returned) =
        CompiledBaseDagBuilder::from_complete_witness(prefix).unwrap_err();
    assert_eq!(returned.operations.len(), operation_count);
    assert_eq!(
        returned.next_producer + 1,
        returned.base_authority.producers.len()
    );
}

#[test]
fn semantic_appends_reject_arena_authority_drift_transactionally() {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();

    let mut memory = CompiledBaseDagBuilder::from_complete_witness(complete_prefix()).unwrap();
    memory.prefix.base_authority.producers.pop().unwrap();
    let values = memory.prefix.values.clone();
    assert_eq!(
        memory.append_memory_semantics(executable.arena()),
        Err(CompiledBaseDagAppendError::Lowering)
    );
    assert_eq!(memory.prefix.values, values);
    assert!(memory.memory.is_none());

    let mut fixed = CompiledBaseDagBuilder::from_complete_witness(complete_prefix()).unwrap();
    fixed.append_memory_semantics(executable.arena()).unwrap();
    fixed
        .emit_memory_operations_using(fake_memory_wrapper)
        .unwrap();
    fixed.prefix.base_authority.producers.pop().unwrap();
    let values = fixed.prefix.values.clone();
    assert_eq!(
        fixed.append_fixed_table_semantics(executable.arena()),
        Err(CompiledBaseDagAppendError::Lowering)
    );
    assert_eq!(fixed.prefix.values, values);
    assert!(fixed.fixed_tables.is_none());
}

#[test]
fn memory_receipt_and_operations_are_append_only_and_transactional() {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
    let prefix = complete_prefix();
    let witness_roots = prefix.causal_external_roots.clone().unwrap();
    let old_operations = prefix.operations.clone();
    let old_wrappers = prefix.static_wrappers.clone();
    let old_effects = prefix.effects.clone();
    let old_direct_b2n = prefix.base_authority.direct_retained_b2n.clone();
    let before = prefix.values.clone();
    let mut builder = CompiledBaseDagBuilder::from_complete_witness(prefix).unwrap();

    assert_eq!(
        builder.emit_memory_operations(),
        Err(CompiledBaseDagAppendError::InvalidStage)
    );
    builder.append_memory_semantics(executable.arena()).unwrap();
    assert_eq!(builder.prefix.operations, old_operations);
    assert_eq!(builder.prefix.static_wrappers, old_wrappers);
    assert_eq!(builder.prefix.effects, old_effects);
    assert_eq!(
        builder.prefix.base_authority.direct_retained_b2n,
        old_direct_b2n
    );
    let missing_kind = {
        let sealed = builder.memory.as_ref().unwrap();
        let lowered = sealed.lowered.as_ref().unwrap();
        assert_eq!(lowered.steps().len(), 5);
        assert!(!sealed.emitted);
        memory_base_trace::validate_from(
            executable.arena(),
            &before,
            &builder.prefix.values,
            &sealed.lowered,
        )
        .unwrap();
        lowered.steps()[1].kind()
    };

    let post_memory_values = builder.prefix.values.clone();
    assert_eq!(
        builder.append_memory_semantics(executable.arena()),
        Err(CompiledBaseDagAppendError::InvalidStage)
    );
    assert_eq!(builder.prefix.values, post_memory_values);

    assert_eq!(
        builder.emit_memory_operations_using(miss_second_memory_wrapper),
        Err(CompiledBaseDagAppendError::MissingMemoryStaticWrapper {
            step_ordinal: 1,
            kind: missing_kind,
        })
    );
    assert_eq!(builder.prefix.operations, old_operations);
    assert_eq!(builder.prefix.static_wrappers, old_wrappers);
    assert_eq!(builder.prefix.effects, old_effects);
    assert_eq!(builder.prefix.values, post_memory_values);
    assert!(!builder.memory.as_ref().unwrap().emitted);

    let old_operation_count = builder.prefix.operations.len();
    let old_wrapper_count = builder.prefix.static_wrappers.len();
    builder
        .emit_memory_operations_using(fake_memory_wrapper)
        .unwrap();
    let lowered = builder.memory.as_ref().unwrap().lowered.as_ref().unwrap();
    assert_eq!(
        builder.prefix.operations.len(),
        old_operation_count + lowered.steps().len()
    );
    assert_eq!(
        builder.prefix.static_wrappers.len(),
        old_wrapper_count + lowered.steps().len()
    );
    for (ordinal, step) in lowered.steps().iter().enumerate() {
        let operation = &builder.prefix.operations[old_operation_count + ordinal];
        let wrapper = &builder.prefix.static_wrappers[old_wrapper_count + ordinal];
        assert_eq!(
            operation.primitive,
            ExecutionPrimitive::StaticCudaWrapper {
                wrapper: wrapper.id()
            }
        );
        assert_eq!(operation.invocation.as_ref(), Some(step.invocation()));
        assert_eq!(operation.effect, step.effect().id());
        assert_eq!(
            operation.stage,
            ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
        );
        assert_eq!(wrapper.accepted_effect(), step.effect().id());
        assert_eq!(wrapper.consumer_target_sm(), TARGET_SM);
        assert!(builder
            .prefix
            .effects
            .iter()
            .any(|effect| effect == step.effect()));
    }
    assert!(builder.has_complete_memory_stage());
    assert_eq!(builder.prefix.values, post_memory_values);
    assert_eq!(
        builder.prefix.causal_external_roots.as_ref(),
        Some(&witness_roots)
    );
    assert_eq!(
        builder.prefix.base_authority.direct_retained_b2n,
        old_direct_b2n
    );
    assert_eq!(
        builder.emit_memory_operations(),
        Err(CompiledBaseDagAppendError::InvalidStage)
    );
}

#[test]
fn post_fixed_pre_base_commit_publishes_transactionally_and_preserves_mixed_sources() {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
    let mut builder = CompiledBaseDagBuilder::from_complete_witness(complete_prefix()).unwrap();
    assert_eq!(
        builder.append_fixed_table_semantics(executable.arena()),
        Err(CompiledBaseDagAppendError::InvalidStage)
    );
    builder.append_memory_semantics(executable.arena()).unwrap();
    builder
        .emit_memory_operations_using(fake_memory_wrapper)
        .unwrap();
    let before = builder.prefix.values.clone();
    builder
        .append_fixed_table_semantics(executable.arena())
        .unwrap();
    let sealed = builder.fixed_tables.as_ref().unwrap();
    fixed_table_materialization::validate_from(
        executable.arena(),
        &before,
        &builder.prefix.values,
        &sealed.lowered,
    )
    .unwrap();
    assert_eq!(
        (
            sealed.fixed_image_roots.occurrence_count(),
            sealed.fixed_image_roots.distinct_count(),
            *sealed.fixed_image_roots.digest(),
        ),
        (
            72,
            71,
            [
                94, 120, 13, 23, 171, 20, 167, 225, 151, 11, 40, 225, 169, 227, 137, 214, 140, 136,
                111, 219, 226, 182, 40, 206, 78, 182, 168, 37, 38, 101, 75, 29,
            ],
        )
    );
    let mixed = sealed
        .lowered
        .tables()
        .iter()
        .find(|table| {
            table.sources().iter().any(|source| {
                matches!(
                    source,
                    fixed_table_materialization::LoweredFixedTableSource::Arena { .. }
                )
            }) && table.sources().iter().any(|source| {
                matches!(
                    source,
                    fixed_table_materialization::LoweredFixedTableSource::Registered { .. }
                )
            })
        })
        .expect("Pedersen-18 must retain seq_23 followed by registered columns");
    assert!(matches!(
        mixed.invocation().arguments[0].value,
        crate::compiled_proof::AotArgumentValue::DeviceMixedFixedSourcePointerTable(_)
    ));
    let (ordinary_catalog, ordinary_version) = mixed
        .sources()
        .iter()
        .find_map(|source| match source {
            fixed_table_materialization::LoweredFixedTableSource::Arena {
                value, version, ..
            } => Some((*value, *version)),
            fixed_table_materialization::LoweredFixedTableSource::Registered { .. } => None,
        })
        .expect("mixed table must retain its ordinary preprocessed source");
    assert!(before.version(ordinary_catalog).is_err());
    assert_eq!(
        builder.prefix.values.version(ordinary_catalog),
        Ok(ordinary_version)
    );

    let old_operations = builder.prefix.operations.clone();
    let old_wrappers = builder.prefix.static_wrappers.clone();
    let old_effects = builder.prefix.effects.clone();
    assert!(matches!(
        builder.emit_fixed_table_operations_using(miss_second_fixed_table_wrapper),
        Err(CompiledBaseDagAppendError::MissingFixedTableStaticWrapper {
            table_ordinal: 1,
            ..
        })
    ));
    assert_eq!(builder.prefix.operations, old_operations);
    assert_eq!(builder.prefix.static_wrappers, old_wrappers);
    assert_eq!(builder.prefix.effects, old_effects);
    assert!(!builder.fixed_tables.as_ref().unwrap().emitted);

    let tables = builder
        .fixed_tables
        .as_ref()
        .unwrap()
        .lowered
        .tables()
        .len();
    assert_eq!(tables, 19);
    builder
        .emit_fixed_table_operations_using(fake_fixed_table_wrapper)
        .unwrap();
    assert_eq!(
        builder.prefix.operations.len(),
        old_operations.len() + tables
    );
    assert_eq!(
        builder.prefix.static_wrappers.len(),
        old_wrappers.len() + tables
    );
    assert_eq!(builder.prefix.operations.len(), 102);
    assert_eq!(builder.prefix.static_wrappers.len(), 71);
    assert_eq!(builder.prefix.kernels.len(), 22);
    assert_eq!(builder.prefix.effects.len(), 102);
    assert_eq!(
        builder
            .prefix
            .operations
            .iter()
            .filter(|operation| matches!(
                operation.primitive,
                ExecutionPrimitive::StatementHostIngress { .. }
            ))
            .count(),
        9
    );
    assert!(builder.has_complete_fixed_table_stage());
    let roots = validate_causal_value_closure(
        &builder.prefix.values,
        &builder.prefix.effects,
        &builder.prefix.operations,
    )
    .unwrap();
    assert_eq!(roots.len(), 88);
    assert_eq!(builder.prefix.causal_external_roots.as_ref(), Some(&roots));
    assert_eq!(
        builder.emit_fixed_table_operations(),
        Err(CompiledBaseDagAppendError::InvalidStage)
    );
}

#[test]
fn fixed_publication_rejects_unread_allocations_and_tampered_root_receipts() {
    let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
    let ready = || {
        let mut builder = CompiledBaseDagBuilder::from_complete_witness(complete_prefix()).unwrap();
        builder.append_memory_semantics(executable.arena()).unwrap();
        builder
            .emit_memory_operations_using(fake_memory_wrapper)
            .unwrap();
        builder
            .append_fixed_table_semantics(executable.arena())
            .unwrap();
        builder
    };

    let mut unread = ready();
    unread.prefix.values = unread.prefix.values.with_unused_catalog_for_test().unwrap();
    assert_eq!(
        unread.emit_fixed_table_operations_using(fake_fixed_table_wrapper),
        Err(CompiledBaseDagAppendError::Lowering)
    );
    assert!(!unread.fixed_tables.as_ref().unwrap().emitted);

    let mut omitted = ready();
    omitted
        .fixed_tables
        .as_mut()
        .unwrap()
        .fixed_image_roots
        .omit_last_distinct_for_test();
    assert_eq!(
        omitted.emit_fixed_table_operations_using(fake_fixed_table_wrapper),
        Err(CompiledBaseDagAppendError::Lowering)
    );
    assert!(!omitted.fixed_tables.as_ref().unwrap().emitted);
}
