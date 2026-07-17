use std::collections::BTreeSet;

use stwo_backend_cuda::aot::{AotKernelAbiSchema, AotKernelModuleGlobals, AotKernelSchemaScope};
use stwo_backend_cuda::TraceTreeRole;

use super::compiled_base_prefix::{
    emit_recorded_base_prefix_for_test, CompiledBasePrefixError, MissingBaseAdapter,
};
use super::loaded_authority::LoadedAuthorityFields;
use super::*;
use crate::compiled_proof::{
    AotArgumentValue, EffectBindingId, ExecutionPrimitive, FixedValueInitializer,
    PartitionAuthority, ProofStage,
};

const MANIFEST: [u8; 32] = [0x4d; 32];
const TARGET_SM: u32 = 89;

#[test]
fn recorded_base_emits_real_ops_and_stops_at_first_native_wrapper() {
    let executable = super::tests::generated_sn2();
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
    let mut malformed = ec_op.clone();
    malformed.invocation.multiplicities.pop();
    assert!(static_wrapper_invocation::ec_op(&malformed).is_err());

    let error =
        emit_recorded_base_prefix_for_test(executable.arena(), MANIFEST, TARGET_SM, |source| {
            Ok(exact_fields(source))
        })
        .unwrap_err();
    let CompiledBasePrefixError::MissingTypedAdapter { missing, prefix } = error else {
        panic!("recorded Base must stop only at the first native wrapper")
    };

    assert_eq!(missing.adapter, MissingBaseAdapter::NativeEcOpStaticWrapper);
    assert_eq!(missing.producer.component, "ec_op_builtin");
    assert_eq!(missing.schedule_ordinal, 7);
    assert_eq!(prefix.execution_manifest_identity, MANIFEST);
    assert_eq!(prefix.target_sm, TARGET_SM);
    assert_eq!(prefix.operations.len(), 7);
    assert_eq!(prefix.kernels.len(), 7);
    assert_eq!(prefix.effects.len(), 7);
    assert_eq!(prefix.partitions, vec![PartitionAuthority::monolithic()]);
    assert_eq!(prefix.direct_retained_b2n.role(), TraceTreeRole::Base);
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

    assert_fixed_values_are_exact(&prefix);
    assert!(
        prefix
            .values
            .entries()
            .any(|(catalog, version)| catalog.0 != version.0),
        "encounter-order semantic versions must not be catalog-ID casts"
    );
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

fn assert_fixed_values_are_exact(prefix: &super::compiled_base_prefix::CompiledBasePrefix) {
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
