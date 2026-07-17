use stwo_backend_cuda::jit_witness::isa::{DeduceKind, WitnessInst, WitnessOp};

use super::*;
use crate::compiled_proof::{AotArgumentValue, EffectAccess};
use crate::shape_executable::ShapeExecutable;

pub(super) fn assert_generated_partial_authority(
    executable: &ShapeExecutable,
    catalog: &BaseProducerCatalog,
    mapped: &producer_prefix::BaseProducerBindingFrontier,
) {
    let partial = mapped.bound[14].recorded().unwrap();
    assert_eq!(partial.position.ordinal, 14);
    assert_eq!(partial.producer.component, "partial_ec_mul_generic");
    assert_ne!(partial.source.program_identity, [0; 32]);
    assert_ne!(partial.source.semantic_hash, 0);
    assert_ne!(partial.source.cache_key, 0);
    assert_eq!(partial.source.source_arguments.len(), 8);
    assert_eq!(partial.invocation.arguments.len(), 8);
    assert_ne!(*partial.effect.id().as_bytes(), [0; 32]);
    let mut kinds = partial.source.deduce.kinds.clone();
    kinds.sort_by_key(|kind| *kind as u32);
    assert_eq!(
        kinds.as_slice(),
        &[
            DeduceKind::FeltAdd,
            DeduceKind::FeltSub,
            DeduceKind::FeltMul,
            DeduceKind::FeltDiv,
        ]
    );
    assert_ne!(partial.source.deduce.source_identity, [0; 32]);

    let SourceArgument::PointerTable {
        descriptor_access,
        entries,
        ..
    } = &partial.source.source_arguments[4]
    else {
        panic!("partial EC multiplicity argument must be its pointer table")
    };
    assert_eq!(*descriptor_access, InvocationAccess::Inactive);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].target.access, InvocationAccess::Inactive);
    let AotArgumentValue::DevicePointerTable(entries) = &partial.invocation.arguments[4].value
    else {
        panic!("partial EC multiplicity argument must lower as a pointer table")
    };
    assert_eq!(entries, &[None]);
    assert!(partial
        .effect
        .accesses()
        .iter()
        .all(|access| !matches!(access, EffectAccess::Atomic { .. })));
    validate_invocation(
        &partial.source,
        catalog,
        executable.arena(),
        partial.producer,
    )
    .unwrap();

    let planned = planned_recorded_component(executable.arena(), partial.producer).unwrap();
    validate_multiplicity_free(planned).unwrap();
    let mut fabricated = planned.clone();
    fabricated.program.n_mult_tables = 1;
    assert_eq!(
        validate_multiplicity_free(&fabricated),
        Err(InvocationShapeError::MultiplicityNeedsSemanticVersions)
    );
    let mut fabricated = planned.clone();
    fabricated
        .program
        .insts
        .push(WitnessInst::new(WitnessOp::MultPush, 0, 0, 0, 0));
    assert_eq!(
        validate_multiplicity_free(&fabricated),
        Err(InvocationShapeError::MultiplicityNeedsSemanticVersions)
    );

    let mut mutated = partial.source.clone();
    mutated.deduce.source_identity[0] ^= 1;
    assert_eq!(
        validate_invocation(&mutated, catalog, executable.arena(), partial.producer),
        Err(InvocationShapeError::InvocationMismatch)
    );

    let emitted = aot::witness_kernel_source(&planned.program).unwrap();
    assert_eq!(
        emitted.program_identity,
        Some(partial.source.program_identity)
    );
    assert_eq!(emitted.semantic_hash, partial.source.semantic_hash);
    assert_eq!(emitted.cache_key, partial.source.cache_key);
    assert_eq!(emitted.kernel_name, partial.source.kernel_symbol);
    assert_eq!(
        aot::emitted_source_identity(&emitted.source),
        partial.source.deduce.source_identity
    );
    let mut forged_source = emitted.source;
    forged_source.push_str("#define STWO_WIT_NEEDS_PEDERSEN 1\n");
    assert_eq!(
        recorded_deduce_authority::bind(&planned.program, &forged_source),
        Err(InvocationShapeError::SourceEmitterRejected)
    );
}

pub(super) fn assert_generated_pedersen_state_authority(
    executable: &ShapeExecutable,
    catalog: &BaseProducerCatalog,
    mapped: &producer_prefix::BaseProducerBindingFrontier,
) {
    let stateful = mapped
        .bound
        .iter()
        .filter_map(producer_prefix::LoweredBaseProducer::recorded)
        .filter(|producer| producer.source.deduce.module_state.is_some())
        .collect::<Vec<_>>();
    assert_eq!(
        stateful
            .iter()
            .map(|producer| (producer.position.ordinal, producer.producer.component))
            .collect::<Vec<_>>(),
        [
            (15, "pedersen_aggregator_window_bits_18"),
            (18, "partial_ec_mul_window_bits_18"),
        ]
    );
    assert_eq!(
        stateful[0].source.deduce.module_state,
        stateful[1].source.deduce.module_state
    );
    for producer in stateful {
        assert_eq!(
            producer.source.deduce.module_state,
            Some(recorded_deduce_authority::PedersenTableColumnsAndRowsV1::CANONICAL)
        );
        // These private inventory effects cover ordinary values only. The
        // production compiler must add real module-global authority after the
        // loaded CUmodule publication receipt is available.
        assert!(producer.effect.module_globals().is_empty());
        validate_invocation(
            &producer.source,
            catalog,
            executable.arena(),
            producer.producer,
        )
        .unwrap();
    }
}
