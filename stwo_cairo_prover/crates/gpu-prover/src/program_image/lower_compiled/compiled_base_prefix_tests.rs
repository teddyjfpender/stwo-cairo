use std::collections::BTreeSet;

use stwo_backend_cuda::aot::{AotKernelAbiSchema, AotKernelModuleGlobals, AotKernelSchemaScope};
use stwo_backend_cuda::TraceTreeRole;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::compiled_base_prefix::emission::{ResolvedStaticExecution, StaticWrapperRequest};
use super::compiled_base_prefix::test_support::assert_witness_writer_def_use_is_ordered;
use super::compiled_base_prefix::{
    emit_recorded_witness_writer_prefix_for_test, CompiledWitnessWriterPrefixError,
    MissingBaseAdapter, StaticWrapperKind,
};
use super::producer_prefix::SemanticBaseProducer;
use super::resolved_recorded_build_authority::ResolvedRecordedBuildAuthority;
use super::*;
use crate::arena_plan::BufferPurpose;
use crate::compiled_proof::{
    AotArgumentValue, EffectAccess, EffectBindingId, EffectContract, ExecutionPrimitive,
    FixedSourcePointerEntry, FixedValueInitializer, LaunchGeometry, PartitionAuthority, ProofStage,
    StatementHostEncoding, StatementHostPart, StatementHostSourceKind, StaticCudaLaunchIdentity,
    StaticCudaWrapperAuthority, StaticCudaWrapperId, ValueRange, ValueVersion,
};

pub(super) const MANIFEST: [u8; 32] = [0x4d; 32];
pub(super) const TARGET_SM: u32 = 89;

fn resolve_static_for_prefix(
    request: StaticWrapperRequest<'_>,
    id: StaticCudaWrapperId,
    target_sm: u32,
    _arena: &ProofArenaPlan,
    _values: &adapter::SemanticValueMap,
) -> Result<Option<ResolvedStaticExecution>, InvocationShapeError> {
    if request.kind() == StaticWrapperKind::NativeEcOp {
        return Ok(None);
    }
    fake_static_authority(request, id, target_sm).map(Some)
}

pub(super) fn resolve_all_static_for_prefix(
    request: StaticWrapperRequest<'_>,
    id: StaticCudaWrapperId,
    target_sm: u32,
    _arena: &ProofArenaPlan,
    _values: &adapter::SemanticValueMap,
) -> Result<Option<ResolvedStaticExecution>, InvocationShapeError> {
    fake_static_authority(request, id, target_sm).map(Some)
}

fn resolve_without_generic_feed(
    request: StaticWrapperRequest<'_>,
    id: StaticCudaWrapperId,
    target_sm: u32,
    _arena: &ProofArenaPlan,
    _values: &adapter::SemanticValueMap,
) -> Result<Option<ResolvedStaticExecution>, InvocationShapeError> {
    if request.kind() == StaticWrapperKind::MultiplicityFeed {
        return Ok(None);
    }
    fake_static_authority(request, id, target_sm).map(Some)
}

fn resolve_without_compact_setup(
    request: StaticWrapperRequest<'_>,
    id: StaticCudaWrapperId,
    target_sm: u32,
    _arena: &ProofArenaPlan,
    _values: &adapter::SemanticValueMap,
) -> Result<Option<ResolvedStaticExecution>, InvocationShapeError> {
    if request.kind() == StaticWrapperKind::WitnessInputCompact {
        return Ok(None);
    }
    fake_static_authority(request, id, target_sm).map(Some)
}

fn fake_static_authority(
    request: StaticWrapperRequest<'_>,
    id: StaticCudaWrapperId,
    target_sm: u32,
) -> Result<ResolvedStaticExecution, InvocationShapeError> {
    let launch = StaticCudaLaunchIdentity::new(
        b"test_base_static_kernel".to_vec(),
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    let (invocation, execution_steps) = match request {
        StaticWrapperRequest::WitnessInputCompact(compact) => {
            super::witness_input_seed_compact::compact_test_execution(compact, 4, 8)?
        }
        _ => (
            request.invocation()?,
            vec![crate::compiled_proof::StaticCudaExecutionStepIdentity::KernelLaunch(launch)],
        ),
    };
    let aggregate_contract_identity = match request {
        StaticWrapperRequest::WitnessCasmScatter(lowered) => lowered.contract.identity(),
        _ => [0x44; 32],
    };
    let wrapper = StaticCudaWrapperAuthority::new_with_execution_steps(
        id,
        [0x41; 32],
        target_sm,
        b"test_base_static_wrapper".to_vec(),
        [0x42; 32],
        [0x43; 32],
        aggregate_contract_identity,
        [0x45; 32],
        execution_steps,
        invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?,
        request.effect()?.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    Ok(ResolvedStaticExecution {
        wrapper,
        invocation,
    })
}

#[test]
fn resolved_recorded_build_authority_rejects_every_offline_identity_mutation() {
    let executable = super::tests::generated_sn2_replacement();
    let authority = BaseProducerAuthority::compile_replacement(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
    )
    .unwrap();
    let recorded = authority
        .producers
        .iter()
        .find_map(|producer| match producer {
            SemanticBaseProducer::Recorded(recorded) => Some(recorded),
            _ => None,
        })
        .unwrap();
    let exact = exact_fields(&recorded.source);
    exact
        .validate(&recorded.source, MANIFEST, TARGET_SM)
        .unwrap();

    let mutations: [fn(&mut ResolvedRecordedBuildAuthority); 13] = [
        |fields| fields.manifest_identity = [0; 32],
        |fields| fields.program_identity[0] ^= 1,
        |fields| fields.abi_schema = None,
        |fields| fields.abi_schema_identity[0] ^= 1,
        |fields| fields.schema_scope = AotKernelSchemaScope::ExportedSymbolOnly,
        |fields| fields.kernel_symbol.push('x'),
        |fields| fields.semantic_hash ^= 1,
        |fields| fields.cache_key ^= 1,
        |fields| fields.target_sm += 1,
        |fields| fields.source_identity[0] ^= 1,
        |fields| fields.cubin_identity = [0; 32],
        |fields| fields.authority_identity = [0; 32],
        |fields| fields.module_globals = AotKernelModuleGlobals::WitnessPedersenV1,
    ];
    for mutate in mutations {
        let mut changed = exact.clone();
        mutate(&mut changed);
        assert!(changed
            .validate(&recorded.source, MANIFEST, TARGET_SM)
            .is_err());
    }
    assert!(exact
        .validate(&recorded.source, [0; 32], TARGET_SM)
        .is_err());
    assert!(exact.validate(&recorded.source, MANIFEST, 0).is_err());

    let stateful = authority
        .producers
        .iter()
        .find_map(|producer| match producer {
            SemanticBaseProducer::Recorded(recorded)
                if recorded.source.deduce.module_state.is_some() =>
            {
                Some(recorded)
            }
            _ => None,
        })
        .expect("generated SN2 must contain a Pedersen recorded deduce");
    let exact = exact_fields(&stateful.source);
    exact
        .validate(&stateful.source, MANIFEST, TARGET_SM)
        .unwrap();
    let mut missing = exact;
    missing.module_globals = AotKernelModuleGlobals::None;
    assert!(missing
        .validate(&stateful.source, MANIFEST, TARGET_SM)
        .is_err());
}

#[test]
fn recorded_witness_writer_prefix_emits_real_ops_and_stops_at_first_native_wrapper() {
    let executable = super::tests::generated_sn2_replacement();
    let authority = BaseProducerAuthority::compile_replacement(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
    )
    .unwrap();
    let ec_op = authority
        .producers
        .iter()
        .find_map(|producer| match producer {
            SemanticBaseProducer::NativeEcOp { contract, .. } => Some(contract),
            _ => None,
        })
        .expect("generated SN2 must schedule native EC-op");
    let invocation = static_wrapper_invocation::ec_op(ec_op).unwrap();
    assert_eq!(invocation.arguments.len(), 18);
    assert!(invocation
        .arguments
        .iter()
        .enumerate()
        .all(|(ordinal, argument)| argument.ordinal as usize == ordinal));
    static_wrapper_invocation::validate_ec_op_invocation_for_test(ec_op, &invocation).unwrap();
    for ordinal in [2, 3] {
        let mut changed = invocation.clone();
        let AotArgumentValue::U32(value) = &changed.arguments[ordinal].value else {
            panic!("EC-op table geometry must be u32")
        };
        changed.arguments[ordinal].value = AotArgumentValue::U32(*value ^ 1);
        assert!(
            static_wrapper_invocation::validate_ec_op_invocation_for_test(ec_op, &changed).is_err()
        );
    }
    let mut changed = invocation.clone();
    changed.arguments[6].value = invocation.arguments[8].value.clone();
    changed.arguments[8].value = invocation.arguments[6].value.clone();
    assert!(
        static_wrapper_invocation::validate_ec_op_invocation_for_test(ec_op, &changed).is_err()
    );
    const COUNT_ARGUMENT_PAIRS: [(usize, usize); 4] = [(10, 11), (12, 13), (14, 15), (16, 17)];
    for (index, &(pointer, words)) in COUNT_ARGUMENT_PAIRS.iter().enumerate() {
        let (next_pointer, next_words) =
            COUNT_ARGUMENT_PAIRS[(index + 1) % COUNT_ARGUMENT_PAIRS.len()];
        let mut changed = invocation.clone();
        changed.arguments[pointer].value = invocation.arguments[next_pointer].value.clone();
        changed.arguments[words].value = invocation.arguments[next_words].value.clone();
        assert!(
            static_wrapper_invocation::validate_ec_op_invocation_for_test(ec_op, &changed).is_err()
        );
        let mut changed = invocation.clone();
        let AotArgumentValue::U32(value) = &changed.arguments[words].value else {
            panic!("EC-op count extent must be u32")
        };
        changed.arguments[words].value = AotArgumentValue::U32(*value ^ 1);
        assert!(
            static_wrapper_invocation::validate_ec_op_invocation_for_test(ec_op, &changed).is_err()
        );
    }
    let authoritative = ec_op.authority.abi().arguments();
    assert!(static_wrapper_invocation::ec_op_using_abi_for_test(
        ec_op,
        &authoritative[..authoritative.len() - 1],
    )
    .is_err());
    let mut rotated = authoritative.to_vec();
    rotated.rotate_right(1);
    assert!(static_wrapper_invocation::ec_op_using_abi_for_test(ec_op, &rotated).is_err());
    for index in 0..authoritative.len() {
        let mut changed = authoritative.to_vec();
        changed[index].ordinal ^= 0x80;
        assert!(static_wrapper_invocation::ec_op_using_abi_for_test(ec_op, &changed).is_err());
        let mut changed = authoritative.to_vec();
        changed[index].name = "wrong_role";
        assert!(static_wrapper_invocation::ec_op_using_abi_for_test(ec_op, &changed).is_err());
        let mut changed = authoritative.to_vec();
        changed[index].kind =
            if changed[index].kind == stwo_backend_cuda::EcOpAbiArgumentKind::CudaStream {
                stwo_backend_cuda::EcOpAbiArgumentKind::U32
            } else {
                stwo_backend_cuda::EcOpAbiArgumentKind::CudaStream
            };
        assert!(static_wrapper_invocation::ec_op_using_abi_for_test(ec_op, &changed).is_err());
        let mut changed = authoritative.to_vec();
        changed[index].access =
            if changed[index].access == stwo_backend_cuda::EcOpAbiAccess::OrderedExecutionStream {
                stwo_backend_cuda::EcOpAbiAccess::Read
            } else {
                stwo_backend_cuda::EcOpAbiAccess::OrderedExecutionStream
            };
        assert!(static_wrapper_invocation::ec_op_using_abi_for_test(ec_op, &changed).is_err());
    }
    let mut malformed = ec_op.clone();
    malformed.invocation.multiplicities.pop();
    assert!(static_wrapper_invocation::ec_op(&malformed).is_err());
    let recorded = authority
        .producers
        .iter()
        .find_map(|producer| match producer {
            SemanticBaseProducer::Recorded(recorded) => Some(recorded),
            _ => None,
        })
        .unwrap();
    let fields = exact_fields(&recorded.source);
    let distinct_effect = authority
        .producers
        .iter()
        .map(|producer| producer.effect().id())
        .find(|effect| *effect != recorded.effect.id())
        .expect("generated SN2 must have more than one Base effect");
    let (kernels, first, second) =
        super::compiled_base_prefix::install_recorded_kernel_pair_for_test(
            &recorded.source,
            &fields,
            &recorded.invocation,
            recorded.effect.id(),
            &recorded.source,
            &fields,
            &recorded.invocation,
            distinct_effect,
        )
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(kernels.len(), 1);
    let invocation = recorded.invocation.contract_id().unwrap();
    let mut accepted = vec![
        (recorded.effect.id(), invocation),
        (distinct_effect, invocation),
    ];
    accepted.sort_unstable();
    assert_eq!(
        kernels[0].accepted_executions(),
        accepted
            .into_iter()
            .map(|(effect, invocation)| {
                (effect, PartitionAuthority::monolithic().id(), invocation)
            })
            .collect::<Vec<_>>()
    );
    let mut drifted_source = recorded.source.clone();
    drifted_source.semantic_hash ^= 1;
    assert!(
        super::compiled_base_prefix::install_recorded_kernel_pair_for_test(
            &recorded.source,
            &fields,
            &recorded.invocation,
            recorded.effect.id(),
            &drifted_source,
            &fields,
            &recorded.invocation,
            distinct_effect,
        )
        .is_err()
    );

    let error = emit_recorded_witness_writer_prefix_for_test(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
        MANIFEST,
        TARGET_SM,
        |source| Ok(exact_fields(source)),
        resolve_static_for_prefix,
    )
    .unwrap_err();
    let CompiledWitnessWriterPrefixError::MissingTypedAdapter { missing, prefix } = error else {
        panic!("recorded Base must stop only at the first native wrapper")
    };

    assert_eq!(missing.adapter, MissingBaseAdapter::NativeEcOpStaticWrapper);
    assert_eq!(missing.producer.component, "ec_op_builtin");
    assert_eq!(missing.schedule_ordinal, 7);
    assert_eq!(prefix.execution_manifest_identity, MANIFEST);
    assert_eq!(prefix.target_sm, TARGET_SM);
    assert_eq!(prefix.kernels.len(), 7);
    assert_eq!(prefix.recorded_kernel_authorities().len(), 7);
    assert_eq!(prefix.effects.len(), prefix.operations.len());
    assert_eq!(prefix.base_authority(), &authority);
    assert_eq!(prefix.next_producer(), missing.schedule_ordinal as usize);
    assert_eq!(
        prefix.pending_producers(),
        &authority.producers[missing.schedule_ordinal as usize..]
    );
    assert_eq!(prefix.partitions, vec![PartitionAuthority::monolithic()]);
    assert_eq!(prefix.direct_retained_b2n().role(), TraceTreeRole::Base);
    assert!(prefix
        .effects
        .windows(2)
        .all(|pair| pair[0].id() < pair[1].id()));
    assert_ne!(prefix.operations.len(), prefix.next_producer());

    let monolithic = PartitionAuthority::monolithic().id();
    let mut aot_count = 0usize;
    let mut host_count = 0usize;
    let mut wrapper_ids = Vec::new();
    for (index, operation) in prefix.operations.iter().enumerate() {
        assert_eq!(operation.id.0 as usize, index);
        assert_eq!(operation.semantic_id.0 as usize, index + 1);
        assert_eq!(
            operation.stage,
            ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
        );
        assert_eq!(operation.partition, monolithic);
        match &operation.primitive {
            ExecutionPrimitive::AotKernel { kernel, launch } => {
                aot_count += 1;
                assert_eq!(kernel.0 as usize, aot_count);
                assert!(!launch.grid.contains(&0));
                assert!(!launch.block.contains(&0));
                let authority = prefix
                    .kernels
                    .iter()
                    .find(|authority| authority.id() == *kernel)
                    .unwrap();
                assert_eq!(
                    authority.accepted_executions(),
                    &[(
                        operation.effect,
                        monolithic,
                        operation
                            .invocation
                            .as_ref()
                            .unwrap()
                            .contract_id()
                            .unwrap(),
                    )]
                );
            }
            ExecutionPrimitive::StaticCudaWrapper { wrapper } => {
                wrapper_ids.push(*wrapper);
                let authority = prefix
                    .static_wrappers
                    .iter()
                    .find(|authority| authority.id() == *wrapper)
                    .unwrap();
                assert_eq!(authority.accepted_effect(), operation.effect);
                assert_eq!(
                    authority.accepted_invocation(),
                    operation
                        .invocation
                        .as_ref()
                        .unwrap()
                        .contract_id()
                        .unwrap()
                );
                assert_eq!(authority.consumer_target_sm(), TARGET_SM);
            }
            ExecutionPrimitive::StatementHostIngress { .. } => {
                host_count += 1;
                assert!(operation.invocation.is_none());
            }
            _ => panic!("Base prefix operation has the wrong primitive"),
        }
        let effect = prefix
            .effects
            .iter()
            .find(|effect| effect.id() == operation.effect)
            .unwrap();
        if let Some(invocation) = operation.invocation.as_ref() {
            assert_invocation_covers_exact_bindings(invocation, effect);
        }
    }
    assert_eq!(aot_count, 7);
    assert_eq!(
        wrapper_ids,
        (1..=u32::try_from(prefix.static_wrappers.len()).unwrap())
            .map(StaticCudaWrapperId)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        prefix.operations.len(),
        aot_count + host_count + wrapper_ids.len()
    );
    assert!(prefix.operations[..3].iter().all(|operation| matches!(
        &operation.primitive,
        ExecutionPrimitive::StaticCudaWrapper { .. }
    )));
    assert_witness_writer_def_use_is_ordered(prefix.base_authority());
    let required = super::compiled_base_prefix::validate_witness_writer_transitions_for_test(
        prefix.base_authority(),
        prefix.values(),
    )
    .unwrap();
    assert!(!required.is_empty());
    assert_eq!(&required, prefix.required_preproducer_versions());
    let fixed_versions = prefix
        .fixed_value_versions()
        .into_iter()
        .map(|value| value.version)
        .collect::<BTreeSet<_>>();
    assert!(!fixed_versions.is_empty());
    assert!(required.is_disjoint(&fixed_versions));
    super::compiled_base_prefix::validate_sealed_prefix_for_test(
        &prefix,
        prefix.next_producer(),
        &prefix.operations,
    )
    .unwrap();
    assert!(
        super::compiled_base_prefix::validate_sealed_prefix_for_test(
            &prefix,
            prefix.next_producer() + 1,
            &prefix.operations,
        )
        .is_err()
    );
    let mut wrong_kernel_use = prefix.operations.clone();
    let first_aot = wrong_kernel_use
        .iter_mut()
        .find(|operation| matches!(&operation.primitive, ExecutionPrimitive::AotKernel { .. }))
        .unwrap();
    let ExecutionPrimitive::AotKernel { kernel, .. } = &mut first_aot.primitive else {
        unreachable!();
    };
    *kernel = prefix.kernels[1].id();
    assert!(
        super::compiled_base_prefix::validate_sealed_prefix_for_test(
            &prefix,
            prefix.next_producer(),
            &wrong_kernel_use,
        )
        .is_err()
    );
    let mut wrong_ids = prefix.operations.clone();
    wrong_ids[0].id.0 += 1;
    assert!(
        super::compiled_base_prefix::validate_sealed_prefix_for_test(
            &prefix,
            prefix.next_producer(),
            &wrong_ids,
        )
        .is_err()
    );
    let mut wrong_semantic_ids = prefix.operations.clone();
    wrong_semantic_ids[0].semantic_id.0 += 1;
    assert!(
        super::compiled_base_prefix::validate_sealed_prefix_for_test(
            &prefix,
            prefix.next_producer(),
            &wrong_semantic_ids,
        )
        .is_err()
    );
    let mut wrong_wrapper = prefix.operations.clone();
    let first_wrapper = wrong_wrapper
        .iter_mut()
        .find_map(|operation| match &mut operation.primitive {
            ExecutionPrimitive::StaticCudaWrapper { wrapper } => Some(wrapper),
            _ => None,
        })
        .unwrap();
    *first_wrapper = StaticCudaWrapperId(2);
    assert!(
        super::compiled_base_prefix::validate_sealed_prefix_for_test(
            &prefix,
            prefix.next_producer(),
            &wrong_wrapper,
        )
        .is_err()
    );
    let unconsumed = prefix
        .values()
        .with_unconsumed_transition_for_test()
        .unwrap();
    assert!(
        super::compiled_base_prefix::validate_witness_writer_transitions_for_test(
            prefix.base_authority(),
            &unconsumed,
        )
        .is_err()
    );

    let first_written = first_destination(prefix.base_authority().producers[0].effect());
    let future_written = first_destination(prefix.base_authority().producers[1].effect());
    let mut future_read = prefix.base_authority().clone();
    replace_first_source(&mut future_read.producers[0], future_written);
    assert!(
        super::compiled_base_prefix::validate_witness_writer_transitions_for_test(
            &future_read,
            prefix.values(),
        )
        .is_err()
    );
    let mut duplicate_destination = prefix.base_authority().clone();
    replace_first_destination(&mut duplicate_destination.producers[1], first_written);
    assert!(
        super::compiled_base_prefix::validate_witness_writer_transitions_for_test(
            &duplicate_destination,
            prefix.values(),
        )
        .is_err()
    );

    assert_fixed_values_are_exact(&prefix);
    assert!(
        prefix
            .values()
            .entries()
            .any(|(catalog, version)| catalog.0 != version.0),
        "encounter-order semantic versions must not be catalog-ID casts"
    );
}

#[test]
fn generated_sn2_full_prefix_and_missing_feed_are_exact() {
    let executable = super::tests::generated_sn2_replacement();
    let full = emit_recorded_witness_writer_prefix_for_test(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
        MANIFEST,
        TARGET_SM,
        |source| Ok(exact_fields(source)),
        resolve_all_static_for_prefix,
    )
    .unwrap();
    assert_eq!(full.operations.len(), 78);
    assert_eq!(full.static_wrappers.len(), 47);
    assert_eq!(full.kernels.len(), 22);
    assert_eq!(full.effects.len(), 78);
    assert_eq!(full.module_global_initializers.len(), 4);
    assert_eq!(full.next_producer(), 23);
    assert_eq!(full.causal_setup_counts(), (5, 7, 9));
    assert_eq!(full.required_preproducer_versions().len(), 249);
    let roots = full
        .causal_external_roots()
        .expect("complete causal prefix must retain its external roots");
    assert_eq!(roots.len(), 17);
    assert_eq!(roots, &full.expected_causal_external_roots());
    let seed_roots = full.causal_seed_scalar_roots();
    assert_eq!(seed_roots.len(), 4);
    assert!(seed_roots.is_subset(roots));
    assert_eq!(
        roots
            .intersection(full.required_preproducer_versions())
            .count(),
        13
    );
    let non_witness_roles = roots
        .intersection(full.required_preproducer_versions())
        .map(|root| {
            let catalog = full
                .values()
                .entries()
                .find_map(|(catalog, version)| (version == *root).then_some(catalog))
                .unwrap();
            let logical = &executable.arena().logical_buffers()[catalog.0 as usize];
            assert_eq!(logical.id.0, catalog.0);
            (
                logical.purpose,
                logical.component,
                logical.part.map(|part| match part {
                    TracePartId::Main => (0, 0),
                    TracePartId::MemoryBig(ordinal) => (1, ordinal),
                    TracePartId::MemorySmall => (2, 0),
                }),
                logical.ordinal,
            )
        })
        .collect::<BTreeSet<_>>();
    let mut expected_non_witness_roles = BTreeSet::from([
        (BufferPurpose::ExecutionTableRawAddressToId, None, None, 0),
        (BufferPurpose::ExecutionTableRawF252Words, None, None, 0),
        (BufferPurpose::ExecutionTableRawSmallWords, None, None, 0),
        (
            BufferPurpose::EcOpSegmentStart,
            Some("ec_op_builtin"),
            Some((0, 0)),
            0,
        ),
    ]);
    expected_non_witness_roles
        .extend((0..9).map(|ordinal| (BufferPurpose::WitnessFeedLut, None, None, ordinal)));
    assert_eq!(non_witness_roles, expected_non_witness_roles);
    for root in &seed_roots {
        let catalog = full
            .values()
            .entries()
            .find_map(|(catalog, version)| (version == *root).then_some(catalog))
            .unwrap();
        assert_eq!(
            executable.arena().logical_buffers()[catalog.0 as usize].purpose,
            BufferPurpose::WitnessInputSeedScalars
        );
    }
    assert_casm_ingress_chain_is_exact(&full);
    let unused = full.values().with_unused_catalog_for_test().unwrap();
    assert!(
        super::compiled_base_prefix::validate_causal_value_closure_for_test(
            &unused,
            &full.effects,
            &full.operations,
        )
        .is_err()
    );
    assert_eq!(
        super::compiled_base_prefix::emission::resolve_setup_counts_for_test(
            executable.arena(),
            &full,
            resolve_all_static_for_prefix,
        ),
        Ok((12, 9))
    );
    let operation_count = full.operations.len();
    assert_eq!(
        super::compiled_base_prefix::emission::resolve_setup_counts_for_test(
            executable.arena(),
            &full,
            resolve_without_compact_setup,
        ),
        Err(
            super::compiled_base_prefix::emission::ResolveSetupError::Missing(
                MissingBaseAdapter::WitnessInputCompactStaticWrapper,
            )
        )
    );
    assert_eq!(full.operations.len(), operation_count);
    assert!(full.pending_producers().is_empty());
    assert_eq!(full.partitions, vec![PartitionAuthority::monolithic()]);

    let error = emit_recorded_witness_writer_prefix_for_test(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
        MANIFEST,
        TARGET_SM,
        |source| Ok(exact_fields(source)),
        resolve_without_generic_feed,
    )
    .unwrap_err();
    let CompiledWitnessWriterPrefixError::MissingTypedAdapter { missing, prefix } = error else {
        panic!("missing first feed must return the complete prelude only")
    };
    assert_eq!(
        missing.adapter,
        MissingBaseAdapter::MultiplicityFeedStaticWrapper
    );
    assert_eq!(missing.schedule_ordinal, 0);
    assert_eq!(prefix.next_producer(), 0);
    assert_eq!(prefix.operations.len(), 3);
    assert_eq!(prefix.static_wrappers.len(), 3);
    assert!(prefix.kernels.is_empty());
    assert!(prefix.module_global_initializers.is_empty());

    let error = emit_recorded_witness_writer_prefix_for_test(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
        MANIFEST,
        TARGET_SM,
        |source| Ok(exact_fields(source)),
        resolve_without_compact_setup,
    )
    .unwrap_err();
    let CompiledWitnessWriterPrefixError::MissingTypedAdapter { missing, prefix } = error else {
        panic!("missing compact setup must return the preceding complete causal groups")
    };
    assert_eq!(
        missing.adapter,
        MissingBaseAdapter::WitnessInputCompactStaticWrapper
    );
    assert_eq!(prefix.next_producer(), missing.schedule_ordinal as usize);
    assert!(prefix.causal_external_roots().is_none());
    super::compiled_base_prefix::validate_sealed_prefix_for_test(
        &prefix,
        prefix.next_producer(),
        &prefix.operations,
    )
    .unwrap();
}

fn assert_casm_ingress_chain_is_exact(
    prefix: &super::compiled_base_prefix::CompiledWitnessWriterPrefix,
) {
    let ingress = prefix
        .operations
        .iter()
        .enumerate()
        .filter(|(_, operation)| {
            matches!(
                operation.primitive,
                ExecutionPrimitive::StatementHostIngress { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(ingress.len(), 9);
    let mut setup = prefix.causal_casm_setup().iter().collect::<Vec<_>>();
    setup.sort_by_key(|setup| setup.position.ordinal);
    let mut previous_destination: Option<ValueRange> = None;
    for ((index, operation), setup) in ingress.into_iter().zip(setup) {
        let ExecutionPrimitive::StatementHostIngress {
            source,
            predecessor,
        } = &operation.primitive
        else {
            unreachable!()
        };
        let requirements = setup.contract.requirements();
        assert_eq!(source.kind, StatementHostSourceKind::WitnessCasm);
        assert_eq!(source.producer_ordinal, setup.position.ordinal);
        assert_eq!(source.component.as_ref(), setup.component);
        assert_eq!(
            source.part,
            match setup.part {
                TracePartId::Main => StatementHostPart::Main,
                TracePartId::MemoryBig(ordinal) => StatementHostPart::MemoryBig(ordinal),
                TracePartId::MemorySmall => StatementHostPart::MemorySmall,
            }
        );
        assert_eq!(source.encoding, StatementHostEncoding::RowMajorU32);
        assert_eq!(source.words, requirements.staging_words);
        assert_eq!(source.real_rows, requirements.n_real_rows);
        assert_eq!(source.consumer_rows, requirements.consumer_rows);
        assert_eq!(source.include_iota, requirements.include_iota);
        assert_eq!(source.casm_contract_identity, setup.contract.identity());
        let expected_predecessor = previous_destination.as_ref().map(|previous| ValueRange {
            version: previous.version,
            elements: setup.staging.elements,
        });
        assert_eq!(predecessor, &expected_predecessor);
        assert!(operation.invocation.is_none());
        let effect = prefix
            .effects
            .iter()
            .find(|effect| effect.id() == operation.effect)
            .unwrap();
        assert_eq!(effect.accesses().len(), 1);
        let destination = effect
            .accesses()
            .iter()
            .find_map(EffectAccess::destination)
            .expect("host ingress must write its exact staging version")
            .value;
        assert_eq!(
            destination,
            ValueRange {
                version: setup.staging.version,
                elements: setup.staging.elements,
            }
        );
        let access = effect.accesses()[0].destination().unwrap();
        assert_eq!(access.binding, setup.staging.binding);
        let scatter = &prefix.operations[index + 1];
        let ExecutionPrimitive::StaticCudaWrapper { wrapper } = &scatter.primitive else {
            panic!("host ingress must be immediately followed by its CASM scatter")
        };
        assert_eq!(
            prefix.static_wrappers[wrapper.0 as usize - 1].aggregate_contract_identity(),
            &setup.contract.identity()
        );
        assert!(scatter.invocation.is_some());
        let scatter_effect = prefix
            .effects
            .iter()
            .find(|effect| effect.id() == scatter.effect)
            .unwrap();
        assert!(scatter_effect.accesses().iter().any(|access| {
            access
                .source()
                .is_some_and(|source| source.value == destination)
        }));
        previous_destination = Some(destination);
    }

    let mut mutated = prefix.operations.clone();
    let ingress = mutated
        .iter_mut()
        .find(|operation| {
            matches!(
                operation.primitive,
                ExecutionPrimitive::StatementHostIngress { .. }
            )
        })
        .unwrap();
    let ExecutionPrimitive::StatementHostIngress { predecessor, .. } = &mut ingress.primitive
    else {
        unreachable!()
    };
    *predecessor = previous_destination;
    assert!(
        super::compiled_base_prefix::validate_sealed_prefix_for_test(
            prefix,
            prefix.next_producer(),
            &mutated,
        )
        .is_err()
    );
}

fn first_destination(effect: &EffectContract) -> ValueVersion {
    effect
        .accesses()
        .iter()
        .find_map(EffectAccess::destination)
        .map(|destination| destination.value.version)
        .expect("fixture producer must write a Base value")
}

fn replace_first_source(producer: &mut SemanticBaseProducer, version: ValueVersion) {
    replace_effect(producer, |accesses| {
        let source = accesses
            .iter_mut()
            .find_map(EffectAccess::source_mut)
            .expect("fixture producer must read a Base value");
        source.value.version = version;
    });
}

fn replace_first_destination(producer: &mut SemanticBaseProducer, version: ValueVersion) {
    replace_effect(producer, |accesses| {
        let destination = accesses
            .iter_mut()
            .find_map(|access| match access {
                EffectAccess::Write { destination }
                | EffectAccess::ReadWrite { destination, .. }
                | EffectAccess::Atomic { destination, .. } => Some(destination),
                EffectAccess::Read { .. } => None,
            })
            .expect("fixture producer must write a Base value");
        destination.value.version = version;
    });
}

fn replace_effect(
    producer: &mut SemanticBaseProducer,
    mutate: impl FnOnce(&mut Vec<EffectAccess>),
) {
    let current = producer.effect();
    let mut accesses = current.accesses().to_vec();
    let globals = current.module_globals().to_vec();
    mutate(&mut accesses);
    let replacement = EffectContract::new(accesses, globals).unwrap();
    match producer {
        SemanticBaseProducer::Recorded(recorded) => recorded.effect = replacement,
        SemanticBaseProducer::NativeBlakeGDirect { contract, .. } => {
            contract.effect = replacement;
        }
        SemanticBaseProducer::NativeEcOp { contract, .. } => contract.effect = replacement,
    }
}

pub(super) fn exact_fields(
    source: &RecordedWitnessInvocationShape,
) -> ResolvedRecordedBuildAuthority {
    ResolvedRecordedBuildAuthority {
        manifest_identity: MANIFEST,
        program_identity: source.program_identity,
        abi_schema: Some(AotKernelAbiSchema::RecordedWitnessV1),
        abi_schema_identity: source.abi_schema_identity,
        schema_scope: AotKernelSchemaScope::StructuredAbi,
        kernel_symbol: source.kernel_symbol.clone(),
        semantic_hash: source.semantic_hash,
        cache_key: source.cache_key,
        target_sm: TARGET_SM,
        source_identity: source.deduce.source_identity,
        cubin_identity: identity(b"cubin", source),
        authority_identity: identity(b"authority", source),
        module_globals: if source.deduce.module_state.is_some() {
            AotKernelModuleGlobals::WitnessPedersenV1
        } else {
            AotKernelModuleGlobals::None
        },
    }
}

fn identity(domain: &[u8], source: &RecordedWitnessInvocationShape) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&source.program_identity);
    hasher.update(&source.cache_key.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn assert_invocation_covers_exact_bindings(
    invocation: &crate::compiled_proof::AotInvocation,
    effect: &crate::compiled_proof::EffectContract,
) {
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>();
    let actual = invocation
        .arguments
        .iter()
        .flat_map(|argument| match &argument.value {
            AotArgumentValue::U32(_)
            | AotArgumentValue::Usize(_)
            | AotArgumentValue::DevicePointer(None) => Vec::new(),
            AotArgumentValue::DevicePointer(Some(binding)) => vec![*binding],
            AotArgumentValue::DevicePointerTable(entries) => {
                entries.iter().flatten().copied().collect()
            }
            AotArgumentValue::DevicePointerTableValue(table) => {
                table.entries.iter().flatten().copied().collect()
            }
            AotArgumentValue::DeviceNestedPointerTableValue { entries } => entries
                .iter()
                .flat_map(|entry| entry.entries.iter().flatten().copied())
                .collect(),
            AotArgumentValue::HostFixedU32(_) => Vec::new(),
            AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(_) => Vec::new(),
            AotArgumentValue::DeviceMixedFixedSourcePointerTable(entries) => entries
                .iter()
                .filter_map(|entry| match entry {
                    FixedSourcePointerEntry::EffectBinding(binding) => Some(*binding),
                    FixedSourcePointerEntry::Registered(_) => None,
                })
                .collect(),
            AotArgumentValue::DeviceFixedU32 { binding, .. } => vec![*binding],
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), expected.len());
}

fn assert_fixed_values_are_exact(
    prefix: &super::compiled_base_prefix::CompiledWitnessWriterPrefix,
) {
    let fixed = prefix.fixed_values();
    let versions = prefix.fixed_value_versions();
    assert!(!fixed.is_empty());
    assert_eq!(fixed.len(), versions.len());
    assert!(fixed.iter().enumerate().all(|(index, value)| {
        value.constant().0 as usize == index
            && versions[index].version == value.value()
            && matches!(value.initializer(), FixedValueInitializer::InlineU32(_))
    }));

    for operation in &prefix.operations {
        let Some(invocation) = operation.invocation.as_ref() else {
            assert!(matches!(
                operation.primitive,
                ExecutionPrimitive::StatementHostIngress { .. }
            ));
            continue;
        };
        let effect = prefix
            .effects
            .iter()
            .find(|effect| effect.id() == operation.effect)
            .unwrap();
        for argument in &invocation.arguments {
            let AotArgumentValue::DeviceFixedU32 { value, binding } = &argument.value else {
                continue;
            };
            assert!(fixed.iter().any(|fixed| fixed.value() == *value));
            assert!(effect.accesses().iter().any(|access| {
                access.source().is_some_and(|source| {
                    source.binding == *binding && source.value.version == *value
                })
            }));
        }
    }

    let fixed_bindings = prefix
        .operations
        .iter()
        .flat_map(|operation| {
            operation
                .invocation
                .iter()
                .flat_map(|invocation| invocation.arguments.iter())
        })
        .filter_map(|argument| match &argument.value {
            AotArgumentValue::DeviceFixedU32 { binding, .. } => Some(*binding),
            _ => None,
        })
        .collect::<Vec<EffectBindingId>>();
    assert!(!fixed_bindings.is_empty());
}
