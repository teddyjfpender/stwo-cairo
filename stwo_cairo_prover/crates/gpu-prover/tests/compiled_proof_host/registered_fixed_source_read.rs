use super::*;

fn source(recipe: u8) -> RegisteredFixedSourceAuthority {
    RegisteredFixedSourceAuthority::new(
        [recipe; 32],
        6,
        8,
        core::mem::size_of::<u32>(),
        vec![b"column_0".to_vec(), b"column_1".to_vec()],
    )
    .unwrap()
}

fn read(
    source: &RegisteredFixedSourceAuthority,
    column: usize,
    start: usize,
    end: usize,
) -> RegisteredFixedSourceRead {
    RegisteredFixedSourceRead::new(source.clone(), column, ElementRange { start, end }).unwrap()
}

fn install_registered_reads(
    input: &mut CompiledProofInput,
    effect_reads: Vec<RegisteredFixedSourceRead>,
    invocation_reads: Option<Vec<RegisteredFixedSourceRead>>,
) {
    let old = &input.effects[0];
    let effect = EffectContract::new_with_registered_fixed_source_reads(
        old.accesses().to_vec(),
        old.module_globals().to_vec(),
        effect_reads,
    )
    .unwrap();
    let effect_id = effect.id();
    input.operations[0].invocation = invocation_with_registered(&effect, invocation_reads);
    input.operations[0].effect = effect_id;
    input.effects = vec![effect];
    let invocation = input.operations[0].invocation.clone().unwrap();
    input.kernels = vec![kernel(
        module(),
        vec![(effect_id, invocation)],
        b"registered-read-build-v1",
    )];
}

fn invocation_with_registered(
    effect: &EffectContract,
    reads: Option<Vec<RegisteredFixedSourceRead>>,
) -> Option<AotInvocation> {
    let mut arguments = invocation(effect).unwrap().arguments;
    if let Some(reads) = reads {
        arguments.push(AotArgumentBinding {
            ordinal: arguments.len() as u8,
            value: AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(reads),
        });
    }
    Some(AotInvocation { arguments })
}

fn compile_with(
    effect_reads: Vec<RegisteredFixedSourceRead>,
    invocation_reads: Option<Vec<RegisteredFixedSourceRead>>,
) -> Result<CompiledProof, CompiledProofError> {
    let mut input = valid_input();
    install_registered_reads(&mut input, effect_reads, invocation_reads);
    CompiledProof::compile(input, transcript())
}

#[test]
fn aot_and_static_wrappers_admit_the_exact_registered_read_table() {
    let source = source(0x31);
    let reads = vec![read(&source, 0, 0, 8), read(&source, 1, 2, 6)];
    assert!(compile_with(reads.clone(), Some(reads.clone())).is_ok());

    let mut input = valid_input();
    install_registered_reads(&mut input, reads.clone(), Some(reads));
    let effect = input.effects[0].id();
    input.operations[0].primitive = ExecutionPrimitive::StaticCudaWrapper {
        wrapper: StaticCudaWrapperId(1),
    };
    input.kernels.clear();
    let invocation = input.operations[0]
        .invocation
        .as_ref()
        .unwrap()
        .contract_id()
        .unwrap();
    input.static_wrappers = vec![StaticCudaWrapperAuthority::new(
        StaticCudaWrapperId(1),
        [1; 32],
        89,
        b"registered_source_wrapper".to_vec(),
        [2; 32],
        [3; 32],
        [4; 32],
        [5; 32],
        vec![StaticCudaLaunchIdentity::new(
            b"registered_source_kernel".to_vec(),
            LaunchGeometry {
                grid: [1, 1, 1],
                block: [128, 1, 1],
                cluster: None,
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
        )
        .unwrap()],
        invocation,
        effect,
    )
    .unwrap()];
    assert!(CompiledProof::compile(input, transcript()).is_ok());
}

#[test]
fn invocation_requires_every_registered_read_exactly_once() {
    let source = source(0x32);
    let first = read(&source, 0, 0, 8);
    let second = read(&source, 1, 0, 8);

    assert!(matches!(
        compile_with(vec![first.clone()], None),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
    assert!(matches!(
        compile_with(vec![], Some(vec![])),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
    assert!(matches!(
        compile_with(
            vec![first.clone()],
            Some(vec![first.clone(), first.clone()])
        ),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
    assert!(matches!(
        compile_with(vec![first.clone(), second], Some(vec![first])),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
}

#[test]
fn effect_reads_reject_invalid_geometry_and_noncanonical_ranges() {
    let source = source(0x33);
    assert!(matches!(
        RegisteredFixedSourceRead::new(source.clone(), 2, ElementRange { start: 0, end: 1 }),
        Err(CompiledProofError::InvalidRegisteredFixedSourceRead)
    ));
    for elements in [
        ElementRange { start: 1, end: 1 },
        ElementRange { start: 0, end: 9 },
    ] {
        assert!(matches!(
            RegisteredFixedSourceRead::new(source.clone(), 0, elements),
            Err(CompiledProofError::InvalidRegisteredFixedSourceRead)
        ));
    }

    let input = valid_input();
    assert!(input.effects[0].registered_fixed_source_reads().is_empty());
    let accesses = input.effects[0].accesses().to_vec();
    let first = read(&source, 0, 0, 2);
    let second = read(&source, 0, 2, 4);
    assert!(matches!(
        EffectContract::new_with_registered_fixed_source_reads(
            accesses.clone(),
            vec![],
            vec![second, first]
        ),
        Err(CompiledProofError::NonCanonicalRegisteredFixedSourceReads)
    ));
    let duplicate = read(&source, 0, 0, 2);
    assert!(matches!(
        EffectContract::new_with_registered_fixed_source_reads(
            accesses.clone(),
            vec![],
            vec![duplicate.clone(), duplicate]
        ),
        Err(CompiledProofError::NonCanonicalRegisteredFixedSourceReads)
    ));
    let overlap = vec![read(&source, 0, 0, 4), read(&source, 0, 3, 6)];
    assert!(matches!(
        EffectContract::new_with_registered_fixed_source_reads(accesses, vec![], overlap),
        Err(CompiledProofError::NonCanonicalRegisteredFixedSourceReads)
    ));
}

#[test]
fn module_global_relocation_cannot_substitute_for_a_registered_read() {
    let source = source(0x34);
    let read = read(&source, 0, 0, 8);
    let mut input = valid_input();
    let bytes = source.columns().len() * core::mem::size_of::<u64>();
    let initializer = ModuleGlobalInitializer::new(
        ModuleGlobalInitializerId(0),
        module(),
        b"registered_columns".to_vec(),
        bytes,
        core::mem::align_of::<u64>(),
        true,
        vec![
            ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
                destination: ByteRange::new(0, bytes).unwrap(),
                source,
            },
        ],
    )
    .unwrap();
    let old = &input.effects[0];
    let effect = EffectContract::new(
        old.accesses().to_vec(),
        vec![ModuleGlobalEffect {
            initializer: initializer.id(),
            bytes: ByteRange::new(0, bytes).unwrap(),
        }],
    )
    .unwrap();
    let effect_id = effect.id();
    let mut arguments = invocation(&effect).unwrap().arguments;
    arguments.push(AotArgumentBinding {
        ordinal: arguments.len() as u8,
        value: AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(vec![read]),
    });
    input.module_global_initializers = vec![initializer];
    input.operations[0].effect = effect_id;
    let malformed = AotInvocation { arguments };
    input.operations[0].invocation = Some(malformed.clone());
    input.effects = vec![effect];
    input.kernels = vec![kernel(
        module(),
        vec![(effect_id, malformed)],
        b"module-substitution-v1",
    )];

    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
}

#[test]
fn copy_and_memset_cannot_claim_registered_source_reads() {
    let source = source(0x35);
    let registered = vec![read(&source, 0, 0, 8)];

    let mut copy = valid_input();
    let bundle = copy.output.sections[0].value;
    let words = copy
        .values
        .iter()
        .find(|value| value.version == bundle)
        .unwrap()
        .layout
        .element_count()
        .unwrap();
    let source_value = ValueVersion(copy.values.len() as u32);
    copy.values.push(u32_value(
        source_value,
        words,
        ValueOrigin::ExternalInput(ExternalInputId(u32::MAX)),
        Region::Input,
    ));
    let copy_effect = EffectContract::new_with_registered_fixed_source_reads(
        vec![
            EffectAccess::Read {
                source: bound(0, value_range(source_value, words)),
            },
            EffectAccess::Write {
                destination: bound(1, value_range(bundle, words)),
            },
        ],
        vec![],
        registered.clone(),
    )
    .unwrap();
    copy.operations[0].primitive = ExecutionPrimitive::DeviceCopyD2D { bytes: words * 4 };
    copy.operations[0].invocation = None;
    copy.operations[0].effect = copy_effect.id();
    copy.effects = vec![copy_effect];
    copy.kernels.clear();
    assert!(matches!(
        CompiledProof::compile(copy, transcript()),
        Err(CompiledProofError::PrimitiveEffectMismatch(OpId(0)))
    ));

    let mut memset = valid_input();
    let bundle = memset.output.sections[0].value;
    let words = memset
        .values
        .iter()
        .find(|value| value.version == bundle)
        .unwrap()
        .layout
        .element_count()
        .unwrap();
    let memset_effect = EffectContract::new_with_registered_fixed_source_reads(
        vec![EffectAccess::Write {
            destination: bound(0, value_range(bundle, words)),
        }],
        vec![],
        registered,
    )
    .unwrap();
    memset.operations[0].primitive = ExecutionPrimitive::DeviceMemsetByte {
        bytes: words * 4,
        value: 0,
    };
    memset.operations[0].invocation = None;
    memset.operations[0].effect = memset_effect.id();
    memset.effects = vec![memset_effect];
    memset.kernels.clear();
    assert!(matches!(
        CompiledProof::compile(memset, transcript()),
        Err(CompiledProofError::PrimitiveEffectMismatch(OpId(0)))
    ));
}

fn composite_input(outer_reads: Vec<RegisteredFixedSourceRead>) -> CompiledProofInput {
    let source = source(0x36);
    let first = read(&source, 0, 0, 8);
    let second = read(&source, 1, 0, 8);
    let mut input = valid_input();
    let base_accesses = input.effects[0].accesses().to_vec();
    let first_effect = EffectContract::new_with_registered_fixed_source_reads(
        base_accesses.clone(),
        vec![],
        vec![first.clone()],
    )
    .unwrap();
    let second_effect = EffectContract::new_with_registered_fixed_source_reads(
        vec![base_accesses[0].clone()],
        vec![],
        vec![second.clone()],
    )
    .unwrap();
    let outer =
        EffectContract::new_with_registered_fixed_source_reads(base_accesses, vec![], outer_reads)
            .unwrap();
    let launch = match &input.operations[0].primitive {
        ExecutionPrimitive::AotKernel { launch, .. } => *launch,
        _ => unreachable!(),
    };
    input.operations[0].primitive = ExecutionPrimitive::OrderedComposite {
        children: vec![
            ExecutableStep {
                primitive: ExecutionPrimitive::AotKernel {
                    kernel: AotKernelId(1),
                    launch,
                },
                invocation: invocation_with_registered(&first_effect, Some(vec![first.clone()])),
                effect: first_effect.id(),
            },
            ExecutableStep {
                primitive: ExecutionPrimitive::AotKernel {
                    kernel: AotKernelId(1),
                    launch,
                },
                invocation: invocation_with_registered(&second_effect, Some(vec![second.clone()])),
                effect: second_effect.id(),
            },
        ]
        .into_boxed_slice(),
    };
    input.operations[0].invocation = None;
    input.operations[0].effect = outer.id();
    let first_invocation = invocation_with_registered(&first_effect, Some(vec![first])).unwrap();
    let second_invocation = invocation_with_registered(&second_effect, Some(vec![second])).unwrap();
    let mut accepted = vec![
        (first_effect.id(), first_invocation),
        (second_effect.id(), second_invocation),
    ];
    accepted.sort_by_key(|(effect, invocation)| (*effect, invocation.contract_id().unwrap()));
    input.kernels = vec![kernel(module(), accepted, b"registered-composite-v1")];
    input.effects = vec![first_effect, second_effect, outer];
    input.effects.sort_by_key(EffectContract::id);
    input
}

#[test]
fn ordered_composite_boundary_is_the_canonical_registered_read_union() {
    let source = source(0x36);
    let full = vec![read(&source, 0, 0, 8), read(&source, 1, 0, 8)];
    assert!(CompiledProof::compile(composite_input(full), transcript()).is_ok());

    // A full first read is byte-identical to `first_effect`, so that malformed
    // fixture is correctly rejected earlier as a duplicate effect authority.
    let duplicate_outer = vec![read(&source, 0, 0, 8)];
    assert!(matches!(
        CompiledProof::compile(composite_input(duplicate_outer), transcript()),
        Err(CompiledProofError::NonCanonicalEffectAuthority)
    ));

    // Keep the outer authority distinct while omitting part of the canonical
    // child-read union; this must reach and fail the composite boundary gate.
    let missing = vec![read(&source, 0, 0, 7)];
    assert!(matches!(
        CompiledProof::compile(composite_input(missing), transcript()),
        Err(CompiledProofError::CompositeBoundaryEffectMismatch(OpId(0)))
    ));
}

#[test]
fn pointer_order_must_match_effect_and_read_identity_mutations_change_compiled_identity() {
    let registered_source = source(0x37);
    let first = read(&registered_source, 0, 0, 8);
    let second = read(&registered_source, 1, 0, 8);
    let effect_reads = vec![first.clone(), second.clone()];
    let baseline = compile_with(effect_reads.clone(), Some(effect_reads.clone())).unwrap();
    assert!(matches!(
        compile_with(
            effect_reads.clone(),
            Some(vec![second.clone(), first.clone()])
        ),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));

    let changed_range = vec![read(&registered_source, 0, 0, 7), second];
    let changed_range = compile_with(changed_range.clone(), Some(changed_range)).unwrap();
    assert_ne!(baseline.identity(), changed_range.identity());

    let other_source = source(0x38);
    let changed_source = vec![read(&other_source, 0, 0, 8), read(&other_source, 1, 0, 8)];
    let changed_source = compile_with(changed_source.clone(), Some(changed_source)).unwrap();
    assert_ne!(baseline.identity(), changed_source.identity());
}

fn compile_mixed(
    registered_source: &RegisteredFixedSourceAuthority,
    mutate: impl FnOnce(&mut Vec<FixedSourcePointerEntry>),
) -> Result<CompiledProof, CompiledProofError> {
    let mut input = valid_input();
    let read = read(registered_source, 0, 0, 8);
    let old = &input.effects[0];
    let effect = EffectContract::new_with_registered_fixed_source_reads(
        old.accesses().to_vec(),
        old.module_globals().to_vec(),
        vec![read.clone()],
    )
    .unwrap();
    let mut entries = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| FixedSourcePointerEntry::EffectBinding(range.binding))
        .collect::<Vec<_>>();
    entries.insert(1, FixedSourcePointerEntry::Registered(read));
    mutate(&mut entries);
    let invocation = AotInvocation {
        arguments: vec![AotArgumentBinding {
            ordinal: 0,
            value: AotArgumentValue::DeviceMixedFixedSourcePointerTable(entries),
        }],
    };
    input.operations[0].invocation = Some(invocation.clone());
    input.operations[0].effect = effect.id();
    input.kernels = vec![kernel(
        module(),
        vec![(effect.id(), invocation)],
        b"mixed-fixed-source-v1",
    )];
    input.effects = vec![effect];
    CompiledProof::compile(input, transcript())
}

#[test]
fn mixed_pointer_table_requires_exact_ordinary_and_registered_membership() {
    let registered_source = source(0x39);
    assert!(compile_mixed(&registered_source, |_| {}).is_ok());
    assert!(matches!(
        compile_mixed(&registered_source, |entries| {
            entries.retain(|entry| !matches!(entry, FixedSourcePointerEntry::Registered(_)));
        }),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
    assert!(matches!(
        compile_mixed(&registered_source, |entries| {
            entries.push(entries[0].clone());
        }),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
    assert!(matches!(
        compile_mixed(&registered_source, |entries| {
            let registered = entries
                .iter()
                .find(|entry| matches!(entry, FixedSourcePointerEntry::Registered(_)))
                .cloned()
                .unwrap();
            entries.push(registered);
        }),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
}
