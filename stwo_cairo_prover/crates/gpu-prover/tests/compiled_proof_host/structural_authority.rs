use super::*;

fn install_fixed_u32(input: &mut CompiledProofInput, words: Vec<u32>) -> ValueVersion {
    let fixed = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        fixed,
        words.len(),
        ValueOrigin::Constant(ConstantId(0)),
        Region::FixedData,
    ));
    input.fixed_values = vec![FixedValueDesc::inline_u32(ConstantId(0), fixed, words)];

    let mut accesses = input.effects[0].accesses().to_vec();
    let binding = EffectBindingId(accesses.len() as u32);
    accesses.push(EffectAccess::Read {
        source: bound(
            binding.0,
            value_range(
                fixed,
                input.values[fixed.0 as usize]
                    .layout
                    .element_count()
                    .unwrap(),
            ),
        ),
    });
    let contract = EffectContract::new(accesses, vec![]).unwrap();
    install_effect(input, contract);
    let invocation = input.operations[0].invocation.as_mut().unwrap();
    let AotArgumentValue::DevicePointerTable(entries) = &mut invocation.arguments[0].value else {
        panic!("fixture must use a pointer table")
    };
    assert_eq!(entries.pop(), Some(Some(binding)));
    invocation.arguments.push(AotArgumentBinding {
        ordinal: 1,
        value: AotArgumentValue::DeviceFixedU32 {
            value: fixed,
            binding,
        },
    });
    refresh_kernel_invocation_authorities(input);
    fixed
}

#[test]
fn fixed_values_are_bijective_content_bound_and_the_only_literal_channel() {
    let mut first = valid_input();
    let fixed = install_fixed_u32(&mut first, vec![7, 11]);
    let compiled = CompiledProof::compile(first.clone(), transcript()).unwrap();
    assert_eq!(compiled.fixed_values()[0].value(), fixed);

    let mut changed = valid_input();
    install_fixed_u32(&mut changed, vec![7, 12]);
    let changed = CompiledProof::compile(changed, transcript()).unwrap();
    assert_ne!(compiled.identity(), changed.identity());

    let mut missing = first.clone();
    missing.fixed_values.clear();
    assert!(matches!(
        CompiledProof::compile(missing, transcript()),
        Err(CompiledProofError::NonCanonicalFixedValues)
    ));

    let mut misaligned = first.clone();
    misaligned.values[fixed.0 as usize].alignment = 1;
    assert!(matches!(
        CompiledProof::compile(misaligned, transcript()),
        Err(CompiledProofError::InvalidFixedValue)
    ));

    let mut wrong_reference = first;
    let AotArgumentValue::DeviceFixedU32 { value, .. } = &mut wrong_reference.operations[0]
        .invocation
        .as_mut()
        .unwrap()
        .arguments[1]
        .value
    else {
        panic!("fixture must bind the fixed u32 value")
    };
    *value = ValueVersion(0);
    refresh_kernel_invocation_authorities(&mut wrong_reference);
    assert!(matches!(
        CompiledProof::compile(wrong_reference, transcript()),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
}

#[test]
fn fixed_address_relocations_are_address_free_and_module_exact() {
    let mut input = valid_input();
    let fixed = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        fixed,
        2,
        ValueOrigin::Constant(ConstantId(0)),
        Region::FixedData,
    ));
    input.fixed_values = vec![FixedValueDesc::inline_u32(ConstantId(0), fixed, vec![1, 2])];
    input.module_global_initializers = vec![ModuleGlobalInitializer::new(
        ModuleGlobalInitializerId(0),
        module(),
        b"FIXED_TABLE_POINTER".to_vec(),
        8,
        8,
        true,
        vec![ModuleGlobalInitializerAtom::FixedValueAddress {
            destination: ByteRange::new(0, 8).unwrap(),
            value: fixed,
            source_byte_offset: 0,
        }],
    )
    .unwrap()];
    let contract = EffectContract::new(
        input.effects[0].accesses().to_vec(),
        vec![ModuleGlobalEffect {
            initializer: ModuleGlobalInitializerId(0),
            bytes: ByteRange::new(0, 8).unwrap(),
        }],
    )
    .unwrap();
    install_effect(&mut input, contract);
    CompiledProof::compile(input.clone(), transcript()).unwrap();

    input.module_global_initializers = vec![ModuleGlobalInitializer::new(
        ModuleGlobalInitializerId(0),
        module(),
        b"FIXED_TABLE_POINTER".to_vec(),
        8,
        8,
        true,
        vec![ModuleGlobalInitializerAtom::FixedValueAddress {
            destination: ByteRange::new(0, 8).unwrap(),
            value: ValueVersion(0),
            source_byte_offset: 0,
        }],
    )
    .unwrap()];
    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::InvalidModuleGlobalInitializer)
    ));
}

#[test]
fn v4_partition_authority_is_monolithic_canonical_and_used_once() {
    CompiledProof::compile(valid_input(), transcript()).unwrap();
    let mut duplicate = valid_input();
    duplicate.partitions.push(duplicate.partitions[0].clone());
    assert!(matches!(
        CompiledProof::compile(duplicate, transcript()),
        Err(CompiledProofError::NonCanonicalPartitionAuthority)
    ));
}

#[test]
fn transcript_states_output_fragments_and_finalizer_are_exact() {
    let mut state = valid_input();
    state.transcript_segments[1].entry_state = TranscriptStateVersion(99);
    assert!(matches!(
        CompiledProof::compile(state, transcript()),
        Err(CompiledProofError::TranscriptSegmentBinding { index: 1 })
    ));

    let mut consumed = valid_input();
    consumed.transcript_segments[0].consumed.pop();
    assert!(matches!(
        CompiledProof::compile(consumed, transcript()),
        Err(CompiledProofError::TranscriptSegmentBinding { index: 0 })
    ));

    let mut ordinal = valid_input();
    ordinal.output.fragments[2].ordinal = 7;
    assert!(matches!(
        CompiledProof::compile(ordinal, transcript()),
        Err(CompiledProofError::ProofFragmentOrdinal { index: 2 })
    ));

    let mut destination = valid_input();
    destination.output.fragments[2].destination.end -= 1;
    assert!(matches!(
        CompiledProof::compile(destination, transcript()),
        Err(CompiledProofError::ProofSectionOrder { index: 2 })
    ));

    let mut stale = valid_input();
    stale.identity =
        ProofIdentity::new(b"other-semantics".to_vec(), b"other-build".to_vec()).unwrap();
    assert!(matches!(
        CompiledProof::compile(stale, transcript()),
        Err(CompiledProofError::InvalidHostFinalizer)
    ));

    let mut wrong_shape = valid_input();
    let mut shape = fixture::proof_assembly_shape();
    shape.n_queries += 1;
    wrong_shape.host_finalizer = fixture::host_finalizer_with_shape(
        &wrong_shape.identity,
        wrong_shape.output.codec.clone(),
        shape,
    );
    assert!(matches!(
        CompiledProof::compile(wrong_shape, transcript()),
        Err(CompiledProofError::InvalidHostFinalizer)
    ));

    let mut invalid_pcs = valid_input();
    let mut pcs = fixture::pcs_config();
    pcs.fri_config.log_blowup_factor = 0;
    invalid_pcs.host_finalizer = fixture::host_finalizer_with_pcs(
        &invalid_pcs.identity,
        invalid_pcs.output.codec.clone(),
        fixture::proof_assembly_shape(),
        pcs,
    );
    assert!(matches!(
        CompiledProof::compile(invalid_pcs, transcript()),
        Err(CompiledProofError::InvalidHostFinalizer)
    ));
}

#[test]
fn module_global_effects_bind_initializer_module_symbol_range_and_access() {
    let mut input = valid_input();
    let mut accesses = input.effects[0].accesses().to_vec();
    input.module_global_initializers = vec![module_initializer(module(), b"ROUND_CONSTANTS", 256)];
    let global = ModuleGlobalEffect {
        initializer: ModuleGlobalInitializerId(0),
        bytes: ByteRange::new(0, 256).unwrap(),
    };
    let with_global = EffectContract::new(accesses.clone(), vec![global]).unwrap();
    install_effect(&mut input, with_global);
    let compiled = CompiledProof::compile(input, transcript()).unwrap();
    assert_eq!(compiled.effects()[0].module_globals().len(), 1);

    let mut mismatched = valid_input();
    mismatched.module_global_initializers = vec![module_initializer(
        ModuleIdentity::new(b"different-module".to_vec()).unwrap(),
        b"ROUND_CONSTANTS",
        256,
    )];
    let global = ModuleGlobalEffect {
        initializer: ModuleGlobalInitializerId(0),
        bytes: ByteRange::new(0, 256).unwrap(),
    };
    let contract =
        EffectContract::new(mismatched.effects[0].accesses().to_vec(), vec![global]).unwrap();
    install_effect(&mut mismatched, contract);
    assert!(matches!(
        CompiledProof::compile(mismatched, transcript()),
        Err(CompiledProofError::ModuleGlobalAuthorityMismatch { .. })
    ));

    let source = accesses[0].source().unwrap().value;
    *accesses[0].source_mut().unwrap() = bound(9, source);
    assert!(matches!(
        EffectContract::new(accesses, vec![]),
        Err(CompiledProofError::NonCanonicalEffectBindings)
    ));

    for symbol in [vec![], b"ROUND\0CONSTANTS".to_vec()] {
        assert!(matches!(
            ModuleGlobalInitializer::new(
                ModuleGlobalInitializerId(0),
                module(),
                symbol,
                1,
                1,
                true,
                vec![],
            ),
            Err(CompiledProofError::InvalidModuleGlobalInitializer)
        ));
    }

    assert!(matches!(
        EffectContract::new(
            valid_input().effects[0].accesses().to_vec(),
            vec![
                ModuleGlobalEffect {
                    initializer: ModuleGlobalInitializerId(0),
                    bytes: ByteRange::new(0, 8).unwrap(),
                },
                ModuleGlobalEffect {
                    initializer: ModuleGlobalInitializerId(0),
                    bytes: ByteRange::new(8, 16).unwrap(),
                },
            ],
        ),
        Err(CompiledProofError::NonCanonicalModuleGlobals)
    ));

    let mut duplicate_symbol = valid_input();
    duplicate_symbol.module_global_initializers = vec![
        module_initializer(module(), b"ROUND_CONSTANTS", 8),
        ModuleGlobalInitializer::new(
            ModuleGlobalInitializerId(1),
            module(),
            b"ROUND_CONSTANTS".to_vec(),
            8,
            8,
            true,
            vec![ModuleGlobalInitializerAtom::Literal {
                destination: ByteRange::new(0, 8).unwrap(),
                bytes: vec![0; 8].into_boxed_slice(),
            }],
        )
        .unwrap(),
    ];
    let contract = EffectContract::new(
        duplicate_symbol.effects[0].accesses().to_vec(),
        vec![
            ModuleGlobalEffect {
                initializer: ModuleGlobalInitializerId(0),
                bytes: ByteRange::new(0, 8).unwrap(),
            },
            ModuleGlobalEffect {
                initializer: ModuleGlobalInitializerId(1),
                bytes: ByteRange::new(0, 8).unwrap(),
            },
        ],
    )
    .unwrap();
    install_effect(&mut duplicate_symbol, contract);
    assert!(matches!(
        CompiledProof::compile(duplicate_symbol, transcript()),
        Err(CompiledProofError::NonCanonicalModuleGlobalInitializers)
    ));
}
