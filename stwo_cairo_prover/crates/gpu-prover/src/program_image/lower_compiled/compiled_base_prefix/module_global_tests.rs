use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

use super::super::compiled_base_prefix_tests::{
    exact_fields, resolve_all_static_for_prefix, MANIFEST, TARGET_SM,
};
use super::super::producer_prefix::SemanticBaseProducer;
use super::super::RecordedWitnessInvocationShape;
use super::*;
use crate::compiled_proof::{
    module_global_initializer_structure_identity_for_test, ByteRange, EffectContract,
    ExecutionPrimitive, ModuleGlobalEffect, ModuleGlobalInitializer, ModuleGlobalInitializerAtom,
    ModuleIdentity, RegisteredFixedSourceAuthority,
};

#[test]
fn full_sn2_order_and_module_global_authority_are_independent_and_exact() {
    let executable = super::super::tests::generated_sn2_replacement();
    let prefix = emit_recorded_witness_writer_prefix_for_test(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
        MANIFEST,
        TARGET_SM,
        |source| Ok(exact_fields(source)),
        resolve_all_static_for_prefix,
    )
    .unwrap();
    assert_eq!(prefix.operations.len(), 78);
    assert_eq!(prefix.static_wrappers.len(), 47);
    assert_eq!(prefix.kernels.len(), 22);
    assert_eq!(
        causal_prefix_fixture_digest(&prefix),
        "a4257ab48618e3e09bb99e3073740cf8c3f58ef0224d9586256961dbf710d39f"
    );
    assert_full_sn2_order_without_emission_oracle(&prefix);
    assert_recorded_module_globals_are_exact(&prefix);
}

fn causal_prefix_fixture_digest(prefix: &CompiledWitnessWriterPrefix) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"stwo-cairo.sn2.causal-base-fixture.v1\0");
    for operation in &prefix.operations {
        let encoded = format!("{operation:?}");
        hasher.update(&(encoded.len() as u64).to_le_bytes());
        hasher.update(encoded.as_bytes());
        let effect = prefix
            .effects
            .iter()
            .find(|effect| effect.id() == operation.effect)
            .unwrap();
        hasher.update(&(effect.canonical_encoding().len() as u64).to_le_bytes());
        hasher.update(effect.canonical_encoding());
    }
    hasher.finalize().to_hex().to_string()
}

fn assert_full_sn2_order_without_emission_oracle(prefix: &CompiledWitnessWriterPrefix) {
    #[derive(Clone, Copy)]
    enum ExpectedPrimitive {
        Aot,
        Static,
        HostIngress,
    }

    let authority = prefix.base_authority();
    let mut expected = Vec::<(ExpectedPrimitive, EffectContract, bool)>::new();
    let execution = authority
        .execution_tables
        .as_ref()
        .expect("SN2 must lower both execution-table stages");
    expected.extend(
        execution
            .stages
            .iter()
            .map(|stage| (ExpectedPrimitive::Static, stage.effect.clone(), false)),
    );
    expected.push((
        ExpectedPrimitive::Static,
        authority.multiplicity.clear.effect.clone(),
        false,
    ));
    if let Some(seed) = &authority.multiplicity.public_memory_seed {
        expected.push((ExpectedPrimitive::Static, seed.effect.clone(), false));
    }
    for (ordinal, (producer, feed)) in authority
        .producers
        .iter()
        .zip(&authority.multiplicity.after_producer)
        .enumerate()
    {
        assert_eq!(producer.position().ordinal as usize, ordinal);
        let ordinal = u32::try_from(ordinal).unwrap();
        let mut setup_count = 0usize;
        for setup in prefix
            .causal_setup
            .gathers
            .iter()
            .filter(|setup| setup.position.ordinal == ordinal)
        {
            expected.push((ExpectedPrimitive::Static, setup.effect.clone(), false));
            setup_count += 1;
        }
        for setup in prefix
            .causal_setup
            .seed_compact
            .iter()
            .filter(|setup| setup.position().ordinal == ordinal)
        {
            expected.push((ExpectedPrimitive::Static, setup.effect().clone(), false));
            setup_count += 1;
        }
        for setup in prefix
            .causal_setup
            .casm
            .iter()
            .filter(|setup| setup.position.ordinal == ordinal)
        {
            let (_, ingress) = super::emission::statement_host_ingress_for_test(setup).unwrap();
            expected.push((ExpectedPrimitive::HostIngress, ingress, false));
            expected.push((ExpectedPrimitive::Static, setup.effect.clone(), false));
            setup_count += 1;
        }
        assert!(setup_count <= 1);

        let stateful = matches!(
            producer,
            SemanticBaseProducer::Recorded(recorded)
                if recorded.source.deduce.module_state.is_some()
        );
        expected.push((
            if matches!(producer, SemanticBaseProducer::Recorded(_)) {
                ExpectedPrimitive::Aot
            } else {
                ExpectedPrimitive::Static
            },
            producer.effect().clone(),
            stateful,
        ));
        if let Some(feed) = feed {
            expected.push((ExpectedPrimitive::Static, feed.effect.clone(), false));
        }
    }
    assert_eq!(expected.len(), 78);

    let mut next_aot = 1u32;
    let mut next_wrapper = 1u32;
    let mut host_ingress = 0usize;
    for (index, (operation, (primitive, expected_effect, stateful))) in
        prefix.operations.iter().zip(expected).enumerate()
    {
        assert_eq!(operation.id.0 as usize, index);
        assert_eq!(operation.semantic_id.0 as usize, index + 1);
        let effect = prefix
            .effects
            .iter()
            .find(|effect| effect.id() == operation.effect)
            .unwrap();
        assert_eq!(effect.accesses(), expected_effect.accesses());
        if stateful {
            assert_eq!(effect.module_globals().len(), 2);
        } else {
            assert_eq!(effect, &expected_effect);
        }
        match (&operation.primitive, primitive) {
            (ExecutionPrimitive::AotKernel { kernel, .. }, ExpectedPrimitive::Aot) => {
                assert_eq!(kernel.0, next_aot);
                next_aot += 1;
            }
            (ExecutionPrimitive::StaticCudaWrapper { wrapper }, ExpectedPrimitive::Static) => {
                assert_eq!(wrapper.0, next_wrapper);
                next_wrapper += 1;
            }
            (ExecutionPrimitive::StatementHostIngress { .. }, ExpectedPrimitive::HostIngress) => {
                assert!(operation.invocation.is_none());
                host_ingress += 1;
            }
            _ => panic!("SN2 operation order changed primitive class"),
        }
    }
    assert_eq!(next_aot, 23);
    assert_eq!(next_wrapper, 48);
    assert_eq!(host_ingress, 9);
}

fn assert_recorded_module_globals_are_exact(prefix: &CompiledWitnessWriterPrefix) {
    let stateful = prefix
        .base_authority()
        .producers
        .iter()
        .filter_map(|producer| match producer {
            SemanticBaseProducer::Recorded(recorded)
                if recorded.source.deduce.module_state.is_some() =>
            {
                Some(recorded)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(stateful.len(), 2);
    assert_eq!(
        prefix
            .module_global_initializers
            .iter()
            .enumerate()
            .map(|(index, initializer)| initializer.id().0 as usize == index)
            .collect::<Vec<_>>(),
        vec![true; 4]
    );
    for (pair, recorded) in prefix
        .module_global_initializers
        .chunks_exact(2)
        .zip(&stateful)
    {
        let state = recorded.source.deduce.module_state.unwrap();
        assert_eq!(pair[0].symbol(), state.column_pointers.symbol.as_bytes());
        assert_eq!(pair[0].bytes(), state.column_pointers.symbol_bytes as usize);
        assert_eq!(pair[1].symbol(), state.row_count.symbol.as_bytes());
        assert_eq!(pair[1].bytes(), state.row_count.symbol_bytes as usize);
        let [ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
            destination,
            source,
        }] = pair[0].atoms()
        else {
            panic!("Pedersen column global must use one registered-source relocation")
        };
        assert_eq!(
            *destination,
            ByteRange::new(0, state.column_pointers.symbol_bytes as usize).unwrap()
        );
        assert_eq!(
            source.source_rows(),
            state.resource.registered_source_rows as usize
        );
        assert_eq!(
            source.padded_rows(),
            state.resource.registered_padded_rows as usize
        );
        assert_eq!(
            source.element_bytes(),
            state.resource.element_bytes as usize
        );
        assert_eq!(
            source
                .columns()
                .map(|column| column.to_vec())
                .collect::<Vec<_>>(),
            (0..state.resource.columns)
                .map(|offset| state.resource.first_column + offset)
                .map(|column| format!("{}{}", state.resource.column_identity_prefix, column))
                .map(String::into_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            pair[1].atoms(),
            &[ModuleGlobalInitializerAtom::Literal {
                destination: ByteRange::new(0, state.row_count.symbol_bytes as usize).unwrap(),
                bytes: state
                    .row_count
                    .value
                    .to_le_bytes()
                    .to_vec()
                    .into_boxed_slice(),
            }]
        );
    }

    let effect = prefix
        .effects
        .iter()
        .find(|effect| {
            effect.module_globals().first().is_some_and(|global| {
                global.initializer == prefix.module_global_initializers[0].id()
            })
        })
        .unwrap();
    let source = &stateful[0].source;
    let module = prefix.module_global_initializers[0].module();
    validate_recorded_module_global_effect_for_test(
        source,
        module,
        effect,
        &prefix.module_global_initializers,
    )
    .unwrap();
    assert_module_global_mutations_are_rejected(prefix, source, module, effect);

    let second_effect = prefix
        .effects
        .iter()
        .find(|effect| {
            effect.module_globals().first().is_some_and(|global| {
                global.initializer == prefix.module_global_initializers[2].id()
            })
        })
        .unwrap();
    assert!(validate_recorded_module_global_effect_for_test(
        source,
        prefix.module_global_initializers[2].module(),
        effect,
        &prefix.module_global_initializers,
    )
    .is_err());
    assert!(validate_recorded_module_global_effect_for_test(
        source,
        module,
        second_effect,
        &prefix.module_global_initializers,
    )
    .is_err());
    let stateful_operations = prefix
        .operations
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            prefix
                .effects
                .iter()
                .find(|effect| effect.id() == operation.effect)
                .filter(|effect| !effect.module_globals().is_empty())
                .map(|_| index)
        })
        .collect::<Vec<_>>();
    assert_eq!(stateful_operations.len(), 2);
    let mut swapped = prefix.operations.clone();
    let first = swapped[stateful_operations[0]].effect;
    swapped[stateful_operations[0]].effect = swapped[stateful_operations[1]].effect;
    swapped[stateful_operations[1]].effect = first;
    assert!(validate_sealed_prefix_for_test(prefix, prefix.next_producer(), &swapped).is_err());
}

fn assert_module_global_mutations_are_rejected(
    prefix: &CompiledWitnessWriterPrefix,
    source: &RecordedWitnessInvocationShape,
    module: &ModuleIdentity,
    effect: &EffectContract,
) {
    let validate = |initializers: &[ModuleGlobalInitializer], effect: &EffectContract| {
        validate_recorded_module_global_effect_for_test(source, module, effect, initializers)
    };
    let mut changed = prefix.module_global_initializers.clone();
    let current = changed[0].clone();
    changed[0] = rebuild_initializer(
        &current,
        ModuleIdentity::new(b"wrong-recorded-module".to_vec()).unwrap(),
        current.symbol().to_vec(),
        current.atoms().to_vec(),
    );
    assert!(validate(&changed, effect).is_err());

    let mut changed = prefix.module_global_initializers.clone();
    let current = changed[0].clone();
    changed[0] = rebuild_initializer(
        &current,
        current.module().clone(),
        b"wrong_pedersen_columns".to_vec(),
        current.atoms().to_vec(),
    );
    assert!(validate(&changed, effect).is_err());

    let mut globals = effect.module_globals().to_vec();
    globals[0].bytes.end -= core::mem::size_of::<u64>();
    let changed_effect = EffectContract::new(effect.accesses().to_vec(), globals).unwrap();
    assert_ne!(changed_effect.id(), effect.id());
    assert!(validate(&prefix.module_global_initializers, &changed_effect).is_err());

    let mut globals = effect.module_globals().to_vec();
    globals[1] = ModuleGlobalEffect {
        initializer: prefix.module_global_initializers[2].id(),
        bytes: globals[1].bytes,
    };
    let changed_effect = EffectContract::new(effect.accesses().to_vec(), globals).unwrap();
    assert_ne!(changed_effect.id(), effect.id());
    assert!(validate(&prefix.module_global_initializers, &changed_effect).is_err());

    let mut changed = prefix.module_global_initializers.clone();
    let ModuleGlobalInitializerAtom::Literal { destination, bytes } = &changed[1].atoms()[0] else {
        unreachable!()
    };
    let destination = *destination;
    let mut wrong_rows = bytes.to_vec();
    wrong_rows[0] ^= 1;
    let current = changed[1].clone();
    changed[1] = rebuild_initializer(
        &current,
        current.module().clone(),
        current.symbol().to_vec(),
        vec![ModuleGlobalInitializerAtom::Literal {
            destination,
            bytes: wrong_rows.into_boxed_slice(),
        }],
    );
    assert_ne!(
        changed[1].content_or_recipe_digest(),
        prefix.module_global_initializers[1].content_or_recipe_digest()
    );
    assert!(validate(&changed, effect).is_err());

    let mut globals = effect.module_globals().to_vec();
    globals[1].bytes.end -= 1;
    let changed_effect = EffectContract::new(effect.accesses().to_vec(), globals).unwrap();
    assert_ne!(changed_effect.id(), effect.id());
    assert!(validate(&prefix.module_global_initializers, &changed_effect).is_err());

    let ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
        destination,
        source: fixed,
    } = &prefix.module_global_initializers[0].atoms()[0]
    else {
        unreachable!()
    };
    let columns = fixed
        .columns()
        .map(|column| column.to_vec())
        .collect::<Vec<_>>();
    let mut identities = Vec::new();
    let mut recipe = *fixed.recipe_identity();
    recipe[0] ^= 1;
    identities.push(
        RegisteredFixedSourceAuthority::new(
            recipe,
            fixed.source_rows(),
            fixed.padded_rows(),
            fixed.element_bytes(),
            columns.clone(),
        )
        .unwrap(),
    );
    identities.push(
        RegisteredFixedSourceAuthority::new(
            *fixed.recipe_identity(),
            fixed.source_rows() - 1,
            fixed.padded_rows(),
            fixed.element_bytes(),
            columns.clone(),
        )
        .unwrap(),
    );
    identities.push(
        RegisteredFixedSourceAuthority::new(
            *fixed.recipe_identity(),
            fixed.source_rows(),
            fixed.padded_rows() * 2,
            fixed.element_bytes(),
            columns.clone(),
        )
        .unwrap(),
    );
    identities.push(
        RegisteredFixedSourceAuthority::new(
            *fixed.recipe_identity(),
            fixed.source_rows(),
            fixed.padded_rows(),
            fixed.element_bytes() * 2,
            columns.clone(),
        )
        .unwrap(),
    );
    let mut reordered = columns;
    reordered.swap(0, 1);
    identities.push(
        RegisteredFixedSourceAuthority::new(
            *fixed.recipe_identity(),
            fixed.source_rows(),
            fixed.padded_rows(),
            fixed.element_bytes(),
            reordered.clone(),
        )
        .unwrap(),
    );
    let mut substituted = reordered.clone();
    substituted[0] = b"pedersen_points_999".to_vec();
    identities.push(
        RegisteredFixedSourceAuthority::new(
            *fixed.recipe_identity(),
            fixed.source_rows(),
            fixed.padded_rows(),
            fixed.element_bytes(),
            substituted,
        )
        .unwrap(),
    );
    let mut removed = reordered;
    removed.pop();
    let removed = RegisteredFixedSourceAuthority::new(
        *fixed.recipe_identity(),
        fixed.source_rows(),
        fixed.padded_rows(),
        fixed.element_bytes(),
        removed,
    )
    .unwrap();
    for fixed in identities {
        assert_ne!(fixed.identity(), fixed_source_identity(prefix));
        let mut changed = prefix.module_global_initializers.clone();
        let current = changed[0].clone();
        changed[0] = rebuild_initializer(
            &current,
            current.module().clone(),
            current.symbol().to_vec(),
            vec![
                ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
                    destination: *destination,
                    source: fixed,
                },
            ],
        );
        assert_ne!(
            changed[0].content_or_recipe_digest(),
            prefix.module_global_initializers[0].content_or_recipe_digest()
        );
        assert_ne!(
            module_global_initializer_structure_identity_for_test(&changed[0]).unwrap(),
            module_global_initializer_structure_identity_for_test(
                &prefix.module_global_initializers[0]
            )
            .unwrap()
        );
        assert!(validate(&changed, effect).is_err());
    }

    assert_ne!(removed.identity(), fixed_source_identity(prefix));
    let removed_bytes = removed.columns().len() * core::mem::size_of::<u64>();
    let current = &prefix.module_global_initializers[0];
    let removed_initializer = ModuleGlobalInitializer::new(
        current.id(),
        current.module().clone(),
        current.symbol().to_vec(),
        removed_bytes,
        current.alignment(),
        current.immutable(),
        vec![
            ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
                destination: ByteRange::new(0, removed_bytes).unwrap(),
                source: removed,
            },
        ],
    )
    .unwrap();
    assert_ne!(
        removed_initializer.content_or_recipe_digest(),
        current.content_or_recipe_digest()
    );
    assert_ne!(
        module_global_initializer_structure_identity_for_test(&removed_initializer).unwrap(),
        module_global_initializer_structure_identity_for_test(current).unwrap()
    );
    let mut changed = prefix.module_global_initializers.clone();
    changed[0] = removed_initializer;
    assert!(validate(&changed, effect).is_err());

    assert!(ModuleGlobalInitializer::new(
        prefix.module_global_initializers[0].id(),
        prefix.module_global_initializers[0].module().clone(),
        prefix.module_global_initializers[0].symbol().to_vec(),
        prefix.module_global_initializers[0].bytes(),
        prefix.module_global_initializers[0].alignment(),
        true,
        vec![
            ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
                destination: ByteRange::new(0, destination.end - 8).unwrap(),
                source: fixed.clone(),
            },
        ],
    )
    .is_err());
}

fn fixed_source_identity(prefix: &CompiledWitnessWriterPrefix) -> &[u8; 32] {
    let ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses { source, .. } =
        &prefix.module_global_initializers[0].atoms()[0]
    else {
        unreachable!()
    };
    source.identity()
}

#[test]
fn registered_fixed_source_rejects_every_structural_boundary() {
    let exact = || {
        RegisteredFixedSourceAuthority::new(
            [7; 32],
            8,
            8,
            4,
            vec![b"column_0".to_vec(), b"column_1".to_vec()],
        )
    };
    exact().unwrap();
    assert!(
        RegisteredFixedSourceAuthority::new([0; 32], 8, 8, 4, vec![b"column_0".to_vec()]).is_err()
    );
    assert!(
        RegisteredFixedSourceAuthority::new([7; 32], 0, 8, 4, vec![b"column_0".to_vec()]).is_err()
    );
    assert!(
        RegisteredFixedSourceAuthority::new([7; 32], 9, 8, 4, vec![b"column_0".to_vec()]).is_err()
    );
    assert!(
        RegisteredFixedSourceAuthority::new([7; 32], 8, 12, 4, vec![b"column_0".to_vec()]).is_err()
    );
    assert!(
        RegisteredFixedSourceAuthority::new([7; 32], 8, 8, 0, vec![b"column_0".to_vec()]).is_err()
    );
    for columns in [
        vec![],
        vec![vec![]],
        vec![b"column\0x".to_vec()],
        vec![b"column_0".to_vec(), b"column_0".to_vec()],
    ] {
        assert!(RegisteredFixedSourceAuthority::new([7; 32], 8, 8, 4, columns).is_err());
    }
}

fn rebuild_initializer(
    source: &ModuleGlobalInitializer,
    module: ModuleIdentity,
    symbol: Vec<u8>,
    atoms: Vec<ModuleGlobalInitializerAtom>,
) -> ModuleGlobalInitializer {
    ModuleGlobalInitializer::new(
        source.id(),
        module,
        symbol,
        source.bytes(),
        source.alignment(),
        source.immutable(),
        atoms,
    )
    .unwrap()
}
