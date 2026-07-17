use super::*;

#[derive(Clone, Copy)]
enum Tail {
    Copy,
    RequiredAlias,
}

fn rebind(access: &EffectAccess, binding: u32) -> EffectAccess {
    match access {
        EffectAccess::Read { source } => EffectAccess::Read {
            source: bound(binding, source.value),
        },
        EffectAccess::Write { destination } => EffectAccess::Write {
            destination: bound(binding, destination.value),
        },
        _ => unreachable!("the base fleet fixture has only reads and one write"),
    }
}

fn composite_input(tail: Tail) -> (CompiledProofInput, ValueVersion, ValueVersion) {
    let (base, _, scratch) = compiled_proof();
    let mut input = base.input().clone();
    input.values[scratch.0 as usize].region = Region::Dynamic;
    let output = ValueVersion(input.values.len() as u32);
    let mut output_desc = input.values[scratch.0 as usize].clone();
    output_desc.version = output;
    output_desc.region = Region::Output;
    input.values.push(output_desc);
    for section in &mut input.output.sections {
        section.value = output;
    }
    for fragment in &mut input.output.fragments {
        fragment.source.version = output;
    }

    let operation = input.operations[0].clone();
    let first = ExecutableStep {
        primitive: operation.primitive,
        invocation: operation.invocation,
        effect: operation.effect,
    };
    let words = input.values[output.0 as usize]
        .layout
        .element_count()
        .unwrap();
    let alias = InPlaceAliasAuthority {
        id: InPlaceAliasId(0),
        requirement: InPlaceAliasRequirement::Required,
        discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
    };
    let tail_access = match tail {
        Tail::Copy => vec![
            EffectAccess::Read {
                source: bound(0, value_range(scratch, words)),
            },
            EffectAccess::Write {
                destination: bound(1, value_range(output, words)),
            },
        ],
        Tail::RequiredAlias => vec![EffectAccess::ReadWrite {
            source: bound(0, value_range(scratch, words)),
            destination: bound(0, value_range(output, words)),
            in_place: Some(alias),
        }],
    };
    let tail_effect = EffectContract::new(tail_access, vec![]).unwrap();
    let tail_step = match tail {
        Tail::Copy => ExecutableStep {
            primitive: ExecutionPrimitive::DeviceCopyD2D {
                bytes: words * size_of::<u32>(),
            },
            invocation: None,
            effect: tail_effect.id(),
        },
        Tail::RequiredAlias => {
            let ExecutionPrimitive::AotKernel { kernel, launch } = &first.primitive else {
                unreachable!("the base fleet fixture starts with one AOT operation")
            };
            ExecutableStep {
                primitive: ExecutionPrimitive::AotKernel {
                    kernel: *kernel,
                    launch: *launch,
                },
                invocation: invocation(&tail_effect),
                effect: tail_effect.id(),
            }
        }
    };

    let first_effect = input
        .effects
        .iter()
        .find(|effect| effect.id() == first.effect)
        .unwrap();
    let mut outer_accesses = first_effect
        .accesses()
        .iter()
        .enumerate()
        .map(|(index, access)| rebind(access, index as u32))
        .collect::<Vec<_>>();
    let next = outer_accesses.len() as u32;
    match tail {
        Tail::Copy => outer_accesses.extend([
            EffectAccess::Read {
                source: bound(next, value_range(scratch, words)),
            },
            EffectAccess::Write {
                destination: bound(next + 1, value_range(output, words)),
            },
        ]),
        Tail::RequiredAlias => outer_accesses.push(EffectAccess::ReadWrite {
            source: bound(next, value_range(scratch, words)),
            destination: bound(next, value_range(output, words)),
            in_place: Some(alias),
        }),
    }
    let outer = EffectContract::new(outer_accesses, vec![]).unwrap();
    input.operations[0].primitive = ExecutionPrimitive::OrderedComposite {
        children: vec![first, tail_step].into_boxed_slice(),
    };
    input.operations[0].invocation = None;
    input.operations[0].effect = outer.id();
    if matches!(tail, Tail::RequiredAlias) {
        let kernel = &input.kernels[0];
        let mut accepted = vec![operation.effect, tail_effect.id()];
        accepted.sort_unstable();
        input.kernels[0] = AotKernelAuthority::new(
            kernel.id(),
            kernel.module().clone(),
            kernel.semantic_encoding().to_vec(),
            kernel.execution_build_encoding().to_vec(),
            accepted,
        )
        .unwrap();
    }
    input.effects.push(tail_effect);
    input.effects.push(outer);
    input.effects.sort_by_key(EffectContract::id);
    (input, scratch, output)
}

fn composite_fixture(tail: Tail) -> Fixture {
    let (input, _, output) = composite_input(tail);
    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    let mut result = fixture();
    let value = compiled.value(output).unwrap();
    let bytes = value.layout.logical_bytes().unwrap();
    let live = result.placement.operations[0].during;
    result.placement.owners.push(FleetOwnerPlacement {
        value: value_range(output, value.layout.element_count().unwrap()),
        worker: WorkerId(0),
        live: ScheduleRange::new(live.start, result.placement.terminal_step).unwrap(),
    });
    let storage = StorageId(output.0);
    result.placement.storages.push(StorageDesc {
        id: storage,
        worker: WorkerId(0),
        bytes,
        alignment_bytes: value.alignment,
    });
    result
        .placement
        .storage_bindings
        .extend(
            compiled
                .output()
                .sections
                .iter()
                .map(|section| FleetStoragePlacement {
                    storage,
                    value: ValueRange {
                        version: section.value,
                        elements: section.elements,
                    },
                    offset_bytes: section.elements.start * size_of::<u32>(),
                }),
        );
    result.placement.output_storage = storage;
    result.placement.topology.workers[0].capacity_bytes += bytes;
    result.shape = shape_identity_for_test(
        b"fleet-composite-topology-v1",
        b"fleet-composite-workspace-v1",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    result.compiled = compiled;
    result.output_value = output;
    result
}

fn external_alias_then_read_fixture(requirement: InPlaceAliasRequirement) -> Fixture {
    let (base, _, scratch) = compiled_proof();
    let mut input = base.input().clone();
    input.values[scratch.0 as usize].region = Region::Dynamic;
    let words = input.values[scratch.0 as usize]
        .layout
        .element_count()
        .unwrap();

    let old = ValueVersion(input.values.len() as u32);
    let mut old_desc = input.values[scratch.0 as usize].clone();
    old_desc.version = old;
    old_desc.origin = ValueOrigin::ExternalInput(ExternalInputId(2_000_001));
    old_desc.region = Region::Input;
    input.values.push(old_desc);

    let output = ValueVersion(input.values.len() as u32);
    let mut output_desc = input.values[scratch.0 as usize].clone();
    output_desc.version = output;
    output_desc.region = Region::Output;
    input.values.push(output_desc);
    for section in &mut input.output.sections {
        section.value = output;
    }
    for fragment in &mut input.output.fragments {
        fragment.source.version = output;
    }

    let alias = InPlaceAliasAuthority {
        id: InPlaceAliasId(0),
        requirement,
        discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
    };
    let destination_binding = match requirement {
        InPlaceAliasRequirement::Required => 0,
        InPlaceAliasRequirement::Permitted => 1,
    };
    let alias_effect = EffectContract::new(
        vec![EffectAccess::ReadWrite {
            source: bound(0, value_range(old, words)),
            destination: bound(destination_binding, value_range(scratch, words)),
            in_place: Some(alias),
        }],
        vec![],
    )
    .unwrap();
    let copy_effect = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(0, value_range(old, words)),
            },
            EffectAccess::Write {
                destination: bound(1, value_range(output, words)),
            },
        ],
        vec![],
    )
    .unwrap();
    let next_binding = destination_binding + 1;
    let outer = EffectContract::new(
        vec![
            EffectAccess::ReadWrite {
                source: bound(0, value_range(old, words)),
                destination: bound(destination_binding, value_range(scratch, words)),
                in_place: Some(alias),
            },
            EffectAccess::Read {
                source: bound(next_binding, value_range(old, words)),
            },
            EffectAccess::Write {
                destination: bound(next_binding + 1, value_range(output, words)),
            },
        ],
        vec![],
    )
    .unwrap();
    let operation = input.operations[0].clone();
    let ExecutionPrimitive::AotKernel { kernel, launch } = operation.primitive else {
        unreachable!("base fixture operation is AOT")
    };
    input.operations[0].primitive = ExecutionPrimitive::OrderedComposite {
        children: vec![
            ExecutableStep {
                primitive: ExecutionPrimitive::AotKernel { kernel, launch },
                invocation: invocation(&alias_effect),
                effect: alias_effect.id(),
            },
            ExecutableStep {
                primitive: ExecutionPrimitive::DeviceCopyD2D {
                    bytes: words * size_of::<u32>(),
                },
                invocation: None,
                effect: copy_effect.id(),
            },
        ]
        .into_boxed_slice(),
    };
    input.operations[0].invocation = None;
    input.operations[0].effect = outer.id();
    let authority = input.kernels[0].clone();
    input.kernels[0] = AotKernelAuthority::new(
        authority.id(),
        authority.module().clone(),
        authority.semantic_encoding().to_vec(),
        authority.execution_build_encoding().to_vec(),
        vec![alias_effect.id()],
    )
    .unwrap();
    input.effects = vec![alias_effect, copy_effect, outer];
    input.effects.sort_by_key(EffectContract::id);
    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());

    let mut result = fixture();
    let operation_live = result.placement.operations[0].during;
    let terminal = result.placement.terminal_step;
    result.placement.owners.push(FleetOwnerPlacement {
        value: value_range(old, words),
        worker: WorkerId(0),
        live: ScheduleRange::new(ScheduleStep(0), operation_live.end).unwrap(),
    });
    result.placement.owners.push(FleetOwnerPlacement {
        value: value_range(output, words),
        worker: WorkerId(0),
        live: ScheduleRange::new(operation_live.start, terminal).unwrap(),
    });

    let alias_storage = StorageId(scratch.0);
    result
        .placement
        .storage_bindings
        .retain(|binding| binding.storage != alias_storage);
    result.placement.storage_bindings.extend([
        FleetStoragePlacement {
            storage: alias_storage,
            value: value_range(scratch, words),
            offset_bytes: 0,
        },
        FleetStoragePlacement {
            storage: alias_storage,
            value: value_range(old, words),
            offset_bytes: 0,
        },
    ]);
    result.placement.in_place_aliases = vec![InPlaceAliasPlacement {
        operation: OP_ASSEMBLE,
        alias: InPlaceAliasId(0),
        storage: alias_storage,
        offset_bytes: 0,
    }];

    let output_storage = StorageId(result.placement.storages.len() as u32);
    let bytes = compiled
        .value(output)
        .unwrap()
        .layout
        .logical_bytes()
        .unwrap();
    result.placement.storages.push(StorageDesc {
        id: output_storage,
        worker: WorkerId(0),
        bytes,
        alignment_bytes: compiled.value(output).unwrap().alignment,
    });
    result
        .placement
        .storage_bindings
        .extend(
            compiled
                .output()
                .sections
                .iter()
                .map(|section| FleetStoragePlacement {
                    storage: output_storage,
                    value: ValueRange {
                        version: section.value,
                        elements: section.elements,
                    },
                    offset_bytes: section.elements.start * size_of::<u32>(),
                }),
        );
    result.placement.output_storage = output_storage;
    result.placement.topology.workers[0].capacity_bytes += bytes;
    result.shape = shape_identity_for_test(
        b"fleet-composite-alias-topology-v1",
        b"fleet-composite-alias-workspace-v1",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    result.compiled = compiled;
    result.output_value = output;
    result
}

#[test]
fn ordered_composite_internal_read_is_available_atomically() {
    compile(composite_fixture(Tail::Copy)).unwrap();
}

#[test]
fn ordinary_operation_cannot_read_its_own_output() {
    let (mut input, scratch, output) = composite_input(Tail::Copy);
    let effect = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(0, value_range(scratch, input.output.layout.total_words)),
            },
            EffectAccess::Write {
                destination: bound(1, value_range(output, input.output.layout.total_words)),
            },
        ],
        vec![],
    )
    .unwrap();
    input.operations[0].primitive = ExecutionPrimitive::DeviceCopyD2D {
        bytes: input.output.layout.total_words * size_of::<u32>(),
    };
    input.operations[0].invocation = None;
    input.operations[0].effect = effect.id();
    input.effects = vec![effect];
    input.kernels.clear();
    assert_eq!(
        CompiledProof::compile(input, transcript()).unwrap_err(),
        CompiledProofError::ProducerAfterConsumer {
            value: scratch,
            consumer: OP_ASSEMBLE,
        }
    );
}

#[test]
fn required_child_internal_alias_fails_closed_without_physical_authority() {
    assert_eq!(
        compile(composite_fixture(Tail::RequiredAlias)).unwrap_err(),
        FleetPlanError::InvalidInPlaceAlias {
            operation: OP_ASSEMBLE,
            alias: InPlaceAliasId(0),
        }
    );
}

#[test]
fn ordered_composite_alias_cannot_overwrite_a_later_childs_source() {
    for requirement in [
        InPlaceAliasRequirement::Required,
        InPlaceAliasRequirement::Permitted,
    ] {
        assert_eq!(
            compile(external_alias_then_read_fixture(requirement)).unwrap_err(),
            FleetPlanError::InvalidInPlaceAlias {
                operation: OP_ASSEMBLE,
                alias: InPlaceAliasId(0),
            }
        );
    }
}
