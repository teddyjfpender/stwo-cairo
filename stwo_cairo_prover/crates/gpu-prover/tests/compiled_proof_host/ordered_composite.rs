use super::*;

fn composite_input() -> CompiledProofInput {
    let mut input = valid_input();
    let bundle = input.output.sections[0].value;
    let words = input.output.layout.total_words;
    let mut add_scratch = || {
        let version = ValueVersion(input.values.len() as u32);
        input.values.push(u32_value(
            version,
            words,
            ValueOrigin::OpOutput(OpId(0)),
            Region::Dynamic,
        ));
        version
    };
    let scratch_a = add_scratch();
    let scratch_b = add_scratch();
    let scratch_c = add_scratch();
    let scratch_d = add_scratch();
    let scratch_e = add_scratch();

    let challenge_reads = input.effects[0]
        .accesses()
        .iter()
        .filter(|access| matches!(access, EffectAccess::Read { .. }))
        .cloned()
        .collect::<Vec<_>>();
    let memset = |version, value| {
        let effect = EffectContract::new(
            vec![EffectAccess::Write {
                destination: bound(0, value_range(version, words)),
            }],
            vec![],
        )
        .unwrap();
        let step = ExecutableStep {
            primitive: ExecutionPrimitive::DeviceMemsetByte {
                bytes: words * core::mem::size_of::<u32>(),
                value,
            },
            invocation: None,
            effect: effect.id(),
        };
        (effect, step)
    };
    let (memset_a, step_a) = memset(scratch_a, 0x5a);
    let (memset_b, step_b) = memset(scratch_b, 0xa5);

    let mut aot_accesses = challenge_reads.clone();
    for version in [scratch_a, scratch_b] {
        aot_accesses.push(EffectAccess::Read {
            source: bound(
                u32::try_from(aot_accesses.len()).unwrap(),
                value_range(version, words),
            ),
        });
    }
    aot_accesses.push(EffectAccess::Write {
        destination: bound(
            u32::try_from(aot_accesses.len()).unwrap(),
            value_range(scratch_c, words),
        ),
    });
    let aot_effect = EffectContract::new(aot_accesses, vec![]).unwrap();
    let launch = match &input.operations[0].primitive {
        ExecutionPrimitive::AotKernel { launch, .. } => *launch,
        _ => unreachable!("fixture starts with an AOT operation"),
    };
    let aot_step = ExecutableStep {
        primitive: ExecutionPrimitive::AotKernel {
            kernel: AotKernelId(1),
            launch,
        },
        invocation: invocation(&aot_effect),
        effect: aot_effect.id(),
    };

    let read_write_effect = EffectContract::new(
        vec![EffectAccess::ReadWrite {
            source: bound(0, value_range(scratch_c, words)),
            destination: bound(0, value_range(scratch_d, words)),
            in_place: Some(InPlaceAliasAuthority {
                id: InPlaceAliasId(0),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
            }),
        }],
        vec![],
    )
    .unwrap();
    let read_write_step = ExecutableStep {
        primitive: ExecutionPrimitive::AotKernel {
            kernel: AotKernelId(1),
            launch,
        },
        invocation: invocation(&read_write_effect),
        effect: read_write_effect.id(),
    };

    let atomic_effect = EffectContract::new(
        vec![EffectAccess::Atomic {
            source: bound(0, value_range(scratch_d, words)),
            destination: bound(0, value_range(scratch_e, words)),
            operation: AtomicOperation::AddU32,
            in_place: InPlaceAliasAuthority {
                id: InPlaceAliasId(0),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
            },
        }],
        vec![],
    )
    .unwrap();
    let atomic_step = ExecutableStep {
        primitive: ExecutionPrimitive::AotKernel {
            kernel: AotKernelId(1),
            launch,
        },
        invocation: invocation(&atomic_effect),
        effect: atomic_effect.id(),
    };

    let copy_effect = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(0, value_range(scratch_e, words)),
            },
            EffectAccess::Write {
                destination: bound(1, value_range(bundle, words)),
            },
        ],
        vec![],
    )
    .unwrap();
    let copy_step = ExecutableStep {
        primitive: ExecutionPrimitive::DeviceCopyD2D {
            bytes: words * core::mem::size_of::<u32>(),
        },
        invocation: None,
        effect: copy_effect.id(),
    };

    let mut boundary_accesses = challenge_reads;
    for version in [scratch_a, scratch_b] {
        boundary_accesses.push(EffectAccess::Write {
            destination: bound(
                u32::try_from(boundary_accesses.len()).unwrap(),
                value_range(version, words),
            ),
        });
    }
    for version in [scratch_a, scratch_b] {
        boundary_accesses.push(EffectAccess::Read {
            source: bound(
                u32::try_from(boundary_accesses.len()).unwrap(),
                value_range(version, words),
            ),
        });
    }
    boundary_accesses.push(EffectAccess::Write {
        destination: bound(
            u32::try_from(boundary_accesses.len()).unwrap(),
            value_range(scratch_c, words),
        ),
    });
    let read_write_binding = u32::try_from(boundary_accesses.len()).unwrap();
    boundary_accesses.push(EffectAccess::ReadWrite {
        source: bound(read_write_binding, value_range(scratch_c, words)),
        destination: bound(read_write_binding, value_range(scratch_d, words)),
        in_place: Some(InPlaceAliasAuthority {
            id: InPlaceAliasId(0),
            requirement: InPlaceAliasRequirement::Required,
            discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
        }),
    });
    let atomic_binding = u32::try_from(boundary_accesses.len()).unwrap();
    boundary_accesses.push(EffectAccess::Atomic {
        source: bound(atomic_binding, value_range(scratch_d, words)),
        destination: bound(atomic_binding, value_range(scratch_e, words)),
        operation: AtomicOperation::AddU32,
        in_place: InPlaceAliasAuthority {
            id: InPlaceAliasId(1),
            requirement: InPlaceAliasRequirement::Required,
            discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
        },
    });
    boundary_accesses.push(EffectAccess::Read {
        source: bound(
            u32::try_from(boundary_accesses.len()).unwrap(),
            value_range(scratch_e, words),
        ),
    });
    boundary_accesses.push(EffectAccess::Write {
        destination: bound(
            u32::try_from(boundary_accesses.len()).unwrap(),
            value_range(bundle, words),
        ),
    });
    let boundary = EffectContract::new(boundary_accesses, vec![]).unwrap();

    input.operations[0].primitive = ExecutionPrimitive::OrderedComposite {
        children: vec![
            step_a,
            step_b,
            aot_step,
            read_write_step,
            atomic_step,
            copy_step,
        ]
        .into_boxed_slice(),
    };
    input.operations[0].invocation = None;
    input.operations[0].effect = boundary.id();
    let mut accepted = vec![aot_effect.id(), read_write_effect.id(), atomic_effect.id()];
    accepted.sort_unstable();
    input.kernels = vec![kernel(module(), accepted, b"ordered-composite-aot-v1")];
    input.effects = vec![
        memset_a,
        memset_b,
        aot_effect,
        read_write_effect,
        atomic_effect,
        copy_effect,
        boundary,
    ];
    input.effects.sort_by_key(EffectContract::id);
    input
}

fn children(input: &CompiledProofInput) -> &[ExecutableStep] {
    let ExecutionPrimitive::OrderedComposite { children } = &input.operations[0].primitive else {
        panic!("fixture must remain composite");
    };
    children
}

fn children_mut(input: &mut CompiledProofInput) -> &mut [ExecutableStep] {
    let ExecutionPrimitive::OrderedComposite { children } = &mut input.operations[0].primitive
    else {
        panic!("fixture must remain composite");
    };
    children
}

fn replace_outer_effect(input: &mut CompiledProofInput, replacement: EffectContract) {
    let old = input.operations[0].effect;
    input.effects.retain(|effect| effect.id() != old);
    input.operations[0].effect = replacement.id();
    input.effects.push(replacement);
    input.effects.sort_by_key(EffectContract::id);
}

#[test]
fn ordered_composite_binds_exact_child_order_and_leaf_authority() {
    let input = composite_input();
    let compiled = CompiledProof::compile(input.clone(), transcript()).unwrap();
    assert_eq!(children(compiled.input()).len(), 6);

    let mut reordered = input;
    children_mut(&mut reordered).swap(0, 1);
    let reordered = CompiledProof::compile(reordered, transcript()).unwrap();
    assert_ne!(compiled.identity(), reordered.identity());
}

#[test]
fn ordered_composite_rejects_boundary_drift_and_bad_child_order() {
    let mut boundary_drift = composite_input();
    let outer = boundary_drift
        .effects
        .iter()
        .find(|effect| effect.id() == boundary_drift.operations[0].effect)
        .unwrap();
    let mut accesses = outer.accesses().to_vec();
    accesses.pop();
    replace_outer_effect(
        &mut boundary_drift,
        EffectContract::new(accesses, vec![]).unwrap(),
    );
    assert!(matches!(
        CompiledProof::compile(boundary_drift, transcript()),
        Err(CompiledProofError::CompositeBoundaryEffectMismatch(OpId(0)))
    ));

    let mut uninitialized = composite_input();
    children_mut(&mut uninitialized).swap(0, 2);
    assert!(matches!(
        CompiledProof::compile(uninitialized, transcript()),
        Err(CompiledProofError::CompositeUninitializedRead {
            operation: OpId(0),
            child: 0,
            ..
        })
    ));

    let mut overlapping = composite_input();
    let obsolete_effect = children(&overlapping)[1].effect;
    let repeated = children(&overlapping)[0].clone();
    children_mut(&mut overlapping)[1] = repeated;
    overlapping
        .effects
        .retain(|effect| effect.id() != obsolete_effect);
    let result = CompiledProof::compile(overlapping, transcript());
    assert!(
        matches!(result, Err(CompiledProofError::OverlappingWrite { .. })),
        "{result:?}"
    );
}

#[test]
fn ordered_composite_rejects_nested_steps_hidden_globals_and_stage_bypass() {
    let mut empty = composite_input();
    let outer_effect = empty.operations[0].effect;
    empty.operations[0].primitive = ExecutionPrimitive::OrderedComposite {
        children: Box::new([]),
    };
    empty.kernels.clear();
    empty.effects.retain(|effect| effect.id() == outer_effect);
    let result = CompiledProof::compile(empty, transcript());
    assert!(
        matches!(
            result,
            Err(CompiledProofError::InvalidOrderedComposite {
                operation: OpId(0),
                child: None,
            })
        ),
        "{result:?}"
    );

    let mut outer_invocation = composite_input();
    outer_invocation.operations[0].invocation = Some(AotInvocation { arguments: vec![] });
    assert!(matches!(
        CompiledProof::compile(outer_invocation, transcript()),
        Err(CompiledProofError::InvalidOrderedComposite {
            operation: OpId(0),
            child: None,
        })
    ));

    let mut nested = composite_input();
    let leaf = children(&nested)[0].clone();
    children_mut(&mut nested)[0].primitive = ExecutionPrimitive::OrderedComposite {
        children: vec![leaf].into_boxed_slice(),
    };
    assert!(matches!(
        CompiledProof::compile(nested, transcript()),
        Err(CompiledProofError::InvalidOrderedComposite {
            operation: OpId(0),
            child: Some(0),
        })
    ));

    let mut global = composite_input();
    let aot_id = children(&global)[2].effect;
    let aot = global
        .effects
        .iter()
        .find(|effect| effect.id() == aot_id)
        .unwrap();
    let global_effect = EffectContract::new(
        aot.accesses().to_vec(),
        vec![ModuleGlobalEffect {
            initializer: ModuleGlobalInitializerId(0),
            bytes: ByteRange::new(0, 8).unwrap(),
        }],
    )
    .unwrap();
    global.module_global_initializers = vec![module_initializer(module(), b"HIDDEN", 8)];
    global.effects.retain(|effect| effect.id() != aot_id);
    children_mut(&mut global)[2].effect = global_effect.id();
    global.effects.push(global_effect);
    global.effects.sort_by_key(EffectContract::id);
    let mut accepted = children(&global)
        .iter()
        .filter_map(|child| {
            matches!(&child.primitive, ExecutionPrimitive::AotKernel { .. }).then_some(child.effect)
        })
        .collect::<Vec<_>>();
    accepted.sort_unstable();
    global.kernels = vec![kernel(
        module(),
        accepted,
        b"ordered-composite-hidden-global",
    )];
    assert!(matches!(
        CompiledProof::compile(global, transcript()),
        Err(CompiledProofError::CompositeBoundaryEffectMismatch(OpId(0)))
    ));

    let mut early = composite_input();
    early.operations[0].stage =
        ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase);
    assert!(matches!(
        CompiledProof::compile(early, transcript()),
        Err(CompiledProofError::TranscriptCausality { .. })
    ));
}

#[test]
fn ordered_composite_reuses_exact_aot_copy_and_memset_validation() {
    let mut bad_memset = composite_input();
    children_mut(&mut bad_memset)[0].invocation = Some(AotInvocation { arguments: vec![] });
    assert!(matches!(
        CompiledProof::compile(bad_memset, transcript()),
        Err(CompiledProofError::PrimitiveEffectMismatch(OpId(0)))
    ));

    let mut bad_aot = composite_input();
    let ExecutionPrimitive::AotKernel { launch, .. } = &mut children_mut(&mut bad_aot)[2].primitive
    else {
        unreachable!()
    };
    launch.grid[0] = 0;
    assert!(matches!(
        CompiledProof::compile(bad_aot, transcript()),
        Err(CompiledProofError::InvalidLaunchGeometry(OpId(0)))
    ));

    let mut bad_copy = composite_input();
    let ExecutionPrimitive::DeviceCopyD2D { bytes } = &mut children_mut(&mut bad_copy)[5].primitive
    else {
        unreachable!()
    };
    *bytes -= core::mem::size_of::<u32>();
    assert!(matches!(
        CompiledProof::compile(bad_copy, transcript()),
        Err(CompiledProofError::PrimitiveEffectMismatch(OpId(0)))
    ));
}
