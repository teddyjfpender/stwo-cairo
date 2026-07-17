use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::aot::{AotKernelAbiSchema, AotKernelModuleGlobals, AotKernelSchemaScope};
use stwo_backend_cuda::TraceTreeRole;

use super::compiled_base_prefix::{
    emit_recorded_witness_writer_prefix_for_test, CompiledWitnessWriterPrefixError,
    MissingBaseAdapter,
};
use super::loaded_authority::LoadedAuthorityFields;
use super::producer_prefix::SemanticBaseProducer;
use super::*;
use crate::compiled_proof::{
    AotArgumentValue, EffectAccess, EffectBindingId, EffectContract, ExecutionPrimitive,
    FixedValueInitializer, PartitionAuthority, ProofStage, ValueVersion,
};

const MANIFEST: [u8; 32] = [0x4d; 32];
const TARGET_SM: u32 = 89;

#[test]
fn recorded_witness_writer_prefix_emits_real_ops_and_stops_at_first_native_wrapper() {
    let executable = super::tests::generated_sn2_replacement();
    let authority = BaseProducerAuthority::compile_replacement(executable.arena()).unwrap();
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
            recorded.effect.id(),
            &recorded.source,
            &fields,
            distinct_effect,
        )
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(kernels.len(), 1);
    let mut accepted = vec![recorded.effect.id(), distinct_effect];
    accepted.sort_unstable();
    assert_eq!(
        kernels[0].accepted_executions(),
        accepted
            .into_iter()
            .map(|effect| (effect, PartitionAuthority::monolithic().id()))
            .collect::<Vec<_>>()
    );
    let mut drifted_source = recorded.source.clone();
    drifted_source.semantic_hash ^= 1;
    assert!(
        super::compiled_base_prefix::install_recorded_kernel_pair_for_test(
            &recorded.source,
            &fields,
            recorded.effect.id(),
            &drifted_source,
            &fields,
            distinct_effect,
        )
        .is_err()
    );

    let error = emit_recorded_witness_writer_prefix_for_test(
        executable.arena(),
        MANIFEST,
        TARGET_SM,
        |source| Ok(exact_fields(source)),
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
    assert_eq!(prefix.operations.len(), 7);
    assert_eq!(prefix.kernels.len(), 7);
    assert_eq!(prefix.recorded_kernel_authorities().len(), 7);
    assert_eq!(prefix.effects.len(), 7);
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

    let monolithic = PartitionAuthority::monolithic().id();
    for (index, operation) in prefix.operations.iter().enumerate() {
        assert_eq!(operation.id.0 as usize, index);
        assert_eq!(operation.semantic_id.0 as usize, index + 1);
        assert_eq!(
            operation.stage,
            ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
        );
        assert_eq!(operation.partition, monolithic);
        let ExecutionPrimitive::AotKernel { kernel, launch } = &operation.primitive else {
            panic!("ordinary recorded Base producer must be one AOT kernel")
        };
        assert_eq!(kernel.0 as usize, index + 1);
        assert!(!launch.grid.contains(&0));
        assert!(!launch.block.contains(&0));
        let authority = &prefix.kernels[index];
        assert_eq!(authority.id(), *kernel);
        assert_eq!(
            authority.accepted_executions(),
            &[(operation.effect, monolithic)]
        );
        let effect = prefix
            .effects
            .iter()
            .find(|effect| effect.id() == operation.effect)
            .unwrap();
        assert_invocation_covers_exact_bindings(operation.invocation.as_ref().unwrap(), effect);
    }
    let planned_effects = authority
        .producers
        .iter()
        .map(|producer| producer.effect().id())
        .collect::<Vec<_>>();
    let resumed_effects = prefix
        .operations
        .iter()
        .map(|operation| operation.effect)
        .chain(
            prefix
                .pending_producers()
                .iter()
                .map(|producer| producer.effect().id()),
        )
        .collect::<Vec<_>>();
    assert_eq!(resumed_effects, planned_effects);
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
    let ExecutionPrimitive::AotKernel { kernel, .. } = &mut wrong_kernel_use[0].primitive else {
        unreachable!()
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

fn assert_witness_writer_def_use_is_ordered(
    authority: &super::producer_prefix::BaseProducerAuthority,
) {
    let mut destination_producer = BTreeMap::new();
    for (producer_index, producer) in authority.producers.iter().enumerate() {
        for destination in producer
            .effect()
            .accesses()
            .iter()
            .filter_map(|access| access.destination())
        {
            if let Some(previous) =
                destination_producer.insert(destination.value.version, producer_index)
            {
                assert_eq!(
                    previous, producer_index,
                    "one semantic version cannot be produced by different witness writers"
                );
            }
        }
    }
    for (producer_index, producer) in authority.producers.iter().enumerate() {
        for source in producer
            .effect()
            .accesses()
            .iter()
            .filter_map(|access| access.source())
        {
            if let Some(&source_producer) = destination_producer.get(&source.value.version) {
                assert!(
                    source_producer < producer_index,
                    "writer-produced source version must come from an earlier writer"
                );
            }
        }
    }
}

fn exact_fields(source: &RecordedWitnessInvocationShape) -> LoadedAuthorityFields {
    LoadedAuthorityFields {
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
        module_globals: AotKernelModuleGlobals::None,
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
            AotArgumentValue::U32(_) | AotArgumentValue::DevicePointer(None) => Vec::new(),
            AotArgumentValue::DevicePointer(Some(binding)) => vec![*binding],
            AotArgumentValue::DevicePointerTable(entries) => {
                entries.iter().flatten().copied().collect()
            }
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
        let invocation = operation.invocation.as_ref().unwrap();
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
        .flat_map(|operation| operation.invocation.as_ref().unwrap().arguments.iter())
        .filter_map(|argument| match &argument.value {
            AotArgumentValue::DeviceFixedU32 { binding, .. } => Some(*binding),
            _ => None,
        })
        .collect::<Vec<EffectBindingId>>();
    assert!(!fixed_bindings.is_empty());
}
