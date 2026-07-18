use super::*;

const EXACT_OPERATION: OpId = OpId(0);

fn exact_read_fixture() -> (Fixture, ValueVersion, ValueVersion) {
    let mut fixture = super::operation_execution::exact_fixture();
    let mut input = fixture.compiled.input().clone();
    let destination = ValueVersion(input.values.len() as u32 - 1);
    let sliced = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        sliced,
        8,
        4,
        ValueOrigin::ExternalInput(ExternalInputId(2_100_000)),
        Region::Input,
    ));
    let replicated = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        replicated,
        4,
        4,
        ValueOrigin::Constant(ConstantId(0)),
        Region::FixedData,
    ));
    input.fixed_values = vec![FixedValueDesc::inline_u32(
        ConstantId(0),
        replicated,
        vec![7; 4],
    )];

    let effect = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(0, value_range(sliced, 8)),
            },
            EffectAccess::Read {
                source: bound(1, value_range(replicated, 4)),
            },
            EffectAccess::Write {
                destination: bound(2, value_range(destination, 8)),
            },
        ],
        vec![],
    )
    .unwrap();
    let authority = ExactPartitionAuthority::new(
        0,
        range(0, 8),
        2,
        4,
        vec![
            PartitionEffectProjection::ContiguousAxisSlice {
                binding: EffectBindingId(0),
            },
            PartitionEffectProjection::ReplicatedRead {
                binding: EffectBindingId(1),
            },
            PartitionEffectProjection::ContiguousAxisSlice {
                binding: EffectBindingId(2),
            },
        ],
        PartitionLaunchDerivation::new(1, 2, PartitionGridAxis::X, 2).unwrap(),
    )
    .unwrap();
    let partition = PartitionAuthority::exact(authority).unwrap();
    input.operations[0].effect = effect.id();
    input.operations[0].partition = partition.id();
    input.operations[0].invocation = Some(AotInvocation {
        arguments: vec![
            AotArgumentBinding {
                ordinal: 0,
                value: AotArgumentValue::DevicePointerTable(vec![
                    Some(EffectBindingId(0)),
                    Some(EffectBindingId(1)),
                    Some(EffectBindingId(2)),
                ]),
            },
            AotArgumentBinding {
                ordinal: 1,
                value: AotArgumentValue::U32(0),
            },
            AotArgumentBinding {
                ordinal: 2,
                value: AotArgumentValue::U32(8),
            },
        ],
    });

    let kernel = input.kernels[0].clone();
    let monolithic = (
        input.operations[1].effect,
        input.operations[1].partition,
        input.operations[1]
            .invocation
            .as_ref()
            .unwrap()
            .contract_id()
            .unwrap(),
    );
    let mut accepted = vec![
        (
            effect.id(),
            partition.id(),
            input.operations[0]
                .invocation
                .as_ref()
                .unwrap()
                .contract_id()
                .unwrap(),
        ),
        monolithic,
    ];
    accepted.sort_unstable();
    input.kernels[0] = AotKernelAuthority::new_with_accepted_executions(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        accepted,
    )
    .unwrap();
    let monolithic_effect = input
        .effects
        .iter()
        .find(|candidate| candidate.id() == monolithic.0)
        .unwrap()
        .clone();
    let monolithic_partition = input
        .partitions
        .iter()
        .find(|candidate| candidate.id() == monolithic.1)
        .unwrap()
        .clone();
    input.effects = vec![effect, monolithic_effect];
    input.effects.sort_unstable_by_key(EffectContract::id);
    input.partitions = vec![partition, monolithic_partition];
    input
        .partitions
        .sort_unstable_by_key(PartitionAuthority::id);

    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    fixture.compiled = compiled;

    let terminal = fixture.placement.terminal_step;
    fixture.placement.owners.extend([
        FleetOwnerPlacement {
            value: ValueRange {
                version: sliced,
                elements: range(0, 2),
            },
            worker: WorkerId(0),
            live: during(0, terminal.0),
        },
        FleetOwnerPlacement {
            value: ValueRange {
                version: sliced,
                elements: range(2, 4),
            },
            worker: WorkerId(0),
            live: during(0, terminal.0),
        },
        FleetOwnerPlacement {
            value: ValueRange {
                version: sliced,
                elements: range(4, 6),
            },
            worker: WorkerId(1),
            live: during(0, terminal.0),
        },
        FleetOwnerPlacement {
            value: ValueRange {
                version: sliced,
                elements: range(6, 8),
            },
            worker: WorkerId(1),
            live: during(0, terminal.0),
        },
        FleetOwnerPlacement {
            value: ValueRange {
                version: replicated,
                elements: range(0, 2),
            },
            worker: WorkerId(0),
            live: during(0, terminal.0),
        },
        FleetOwnerPlacement {
            value: ValueRange {
                version: replicated,
                elements: range(2, 4),
            },
            worker: WorkerId(1),
            live: during(0, terminal.0),
        },
    ]);
    add_affine_storage(
        &mut fixture,
        sliced,
        WorkerId(0),
        &[range(0, 2), range(2, 4)],
    );
    add_affine_storage(
        &mut fixture,
        sliced,
        WorkerId(1),
        &[range(4, 6), range(6, 8)],
    );
    add_affine_storage(
        &mut fixture,
        replicated,
        WorkerId(0),
        &[range(0, 2), range(2, 4)],
    );
    add_affine_storage(
        &mut fixture,
        replicated,
        WorkerId(1),
        &[range(0, 2), range(2, 4)],
    );
    let layout = fixture.compiled.value(replicated).unwrap().layout.clone();
    fixture.placement.replicas.extend([
        FleetReplicaPlacement {
            id: ReplicaId(0),
            value: ValueRange {
                version: replicated,
                elements: range(2, 4),
            },
            canonical_worker: WorkerId(1),
            worker: WorkerId(0),
            layout: layout.clone(),
            origin: ReplicaOrigin::InstalledFixed,
            live: during(0, terminal.0),
        },
        FleetReplicaPlacement {
            id: ReplicaId(1),
            value: ValueRange {
                version: replicated,
                elements: range(0, 2),
            },
            canonical_worker: WorkerId(0),
            worker: WorkerId(1),
            layout,
            origin: ReplicaOrigin::InstalledFixed,
            live: during(0, terminal.0),
        },
    ]);
    fixture.placement.topology.workers[0].capacity_bytes += 32;
    fixture.placement.topology.workers[1].capacity_bytes += 32;
    (fixture, sliced, replicated)
}

fn add_affine_storage(
    fixture: &mut Fixture,
    version: ValueVersion,
    worker: WorkerId,
    ranges: &[ElementRange],
) {
    let base = ranges.first().unwrap().start;
    let end = ranges.last().unwrap().end;
    let storage = StorageId(fixture.placement.storages.len() as u32);
    fixture.placement.storages.push(StorageDesc {
        id: storage,
        worker,
        bytes: (end - base) * size_of::<u32>(),
        alignment_bytes: 4,
    });
    fixture
        .placement
        .storage_bindings
        .extend(ranges.iter().map(|&elements| FleetStoragePlacement {
            storage,
            value: ValueRange { version, elements },
            offset_bytes: (elements.start - base) * size_of::<u32>(),
        }));
}

#[test]
fn exact_sliced_and_replicated_reads_are_available_locally() {
    let (fixture, ..) = exact_read_fixture();
    compile(fixture).unwrap();
}

#[test]
fn exact_read_rejects_gap_free_union_across_distinct_storages() {
    let (mut fixture, sliced, _) = exact_read_fixture();
    let split = range(2, 4);
    let binding_index = fixture
        .placement
        .storage_bindings
        .iter()
        .position(|binding| {
            binding.value
                == ValueRange {
                    version: sliced,
                    elements: split,
                }
        })
        .unwrap();
    let original = fixture.placement.storage_bindings[binding_index].storage;
    fixture.placement.storages[original.0 as usize].bytes = 2 * size_of::<u32>();

    let separate = StorageId(fixture.placement.storages.len() as u32);
    fixture.placement.storages.push(StorageDesc {
        id: separate,
        worker: WorkerId(0),
        bytes: split.len() * size_of::<u32>(),
        alignment_bytes: 4,
    });
    fixture.placement.storage_bindings[binding_index].storage = separate;
    fixture.placement.storage_bindings[binding_index].offset_bytes = 0;

    assert_eq!(
        compile(fixture).unwrap_err(),
        FleetPlanError::UndeclaredRead {
            operation: EXACT_OPERATION,
            value: sliced,
        }
    );
}

#[test]
fn exact_sliced_read_rejects_source_shards_on_the_wrong_workers() {
    let (mut fixture, sliced, _) = exact_read_fixture();
    for owner in fixture
        .placement
        .owners
        .iter_mut()
        .filter(|owner| owner.value.version == sliced)
    {
        owner.worker = swap(owner.worker);
    }
    let storage_ids = fixture
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| binding.value.version == sliced)
        .map(|binding| binding.storage)
        .collect::<Vec<_>>();
    for storage in fixture
        .placement
        .storages
        .iter_mut()
        .filter(|storage| storage_ids.contains(&storage.id))
    {
        storage.worker = swap(storage.worker);
    }
    assert_eq!(
        compile(fixture).unwrap_err(),
        FleetPlanError::UndeclaredRead {
            operation: EXACT_OPERATION,
            value: sliced,
        }
    );
}

#[test]
fn exact_replicated_read_requires_a_gap_free_local_union() {
    let (mut fixture, _, replicated) = exact_read_fixture();
    fixture.placement.replicas.clear();
    assert_eq!(
        compile(fixture).unwrap_err(),
        FleetPlanError::UndeclaredRead {
            operation: EXACT_OPERATION,
            value: replicated,
        }
    );
}

#[test]
fn exact_read_rejects_overlapping_local_replica_fragments() {
    let (mut fixture, ..) = exact_read_fixture();
    let mut overlap = fixture.placement.replicas[0].clone();
    overlap.id = ReplicaId(2);
    overlap.value.elements = range(3, 4);
    fixture.placement.replicas.push(overlap);
    assert_eq!(
        compile(fixture).unwrap_err(),
        FleetPlanError::InvalidReplica(ReplicaId(2))
    );
}

#[test]
fn exact_sliced_read_rejects_a_stale_local_fragment() {
    let (mut fixture, sliced, _) = exact_read_fixture();
    fixture
        .placement
        .owners
        .iter_mut()
        .find(|owner| {
            owner.worker == WorkerId(0)
                && owner.value.version == sliced
                && owner.value.elements == range(2, 4)
        })
        .unwrap()
        .live = during(0, 1);
    assert_eq!(
        compile(fixture).unwrap_err(),
        FleetPlanError::UndeclaredRead {
            operation: EXACT_OPERATION,
            value: sliced,
        }
    );
}

#[test]
fn transcript_input_accepts_fragmented_local_ownership() {
    let mut fixture = fixture();
    let binding = fixture
        .compiled
        .transcript_inputs()
        .iter()
        .find(|binding| binding.elements.len() > 1)
        .unwrap();
    let version = binding.value;
    let split = binding.elements.start + 1;
    split_owner_and_storage(&mut fixture, version, split);
    compile(fixture).unwrap();
}

#[test]
fn transcript_input_rejects_a_stale_fragment() {
    let mut fixture = fixture();
    let binding = fixture
        .compiled
        .transcript_inputs()
        .iter()
        .find(|binding| binding.elements.len() > 1)
        .unwrap();
    let version = binding.value;
    let split = binding.elements.start + 1;
    split_owner_and_storage(&mut fixture, version, split);
    fixture
        .placement
        .owners
        .iter_mut()
        .find(|owner| owner.value.version == version && owner.value.elements.start == split)
        .unwrap()
        .live = during(0, 1);
    assert_eq!(
        compile(fixture).unwrap_err(),
        FleetPlanError::TranscriptValueCausality(version)
    );
}

#[test]
fn transcript_output_rejects_gap_free_union_across_distinct_storages() {
    let mut fixture = fixture();
    let transcript_binding = fixture
        .compiled
        .transcript_outputs()
        .iter()
        .find(|binding| binding.elements.len() > 1)
        .copied()
        .unwrap();
    let version = transcript_binding.value;
    let split = transcript_binding.elements.start + 1;
    let mut input = fixture.compiled.input().clone();
    let operation = &mut input.operations[0];
    let previous_effect = input
        .effects
        .iter()
        .find(|effect| effect.id() == operation.effect)
        .unwrap();
    let effect = EffectContract::new(
        previous_effect
            .accesses()
            .iter()
            .filter(|access| {
                access
                    .source()
                    .is_none_or(|source| source.value.version != version)
            })
            .enumerate()
            .map(|(binding, access)| match access {
                EffectAccess::Read { source } => EffectAccess::Read {
                    source: bound(binding as u32, source.value),
                },
                EffectAccess::Write { destination } => EffectAccess::Write {
                    destination: bound(binding as u32, destination.value),
                },
                _ => unreachable!("the base fixture effect contains only reads and one write"),
            })
            .collect(),
        previous_effect.module_globals().to_vec(),
    )
    .unwrap();
    operation.effect = effect.id();
    operation.invocation = invocation(&effect);
    let kernel = input.kernels[0].clone();
    input.kernels[0] = AotKernelAuthority::new(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        vec![(
            effect.id(),
            operation
                .invocation
                .as_ref()
                .unwrap()
                .contract_id()
                .unwrap(),
        )],
    )
    .unwrap();
    input.effects = vec![effect];
    fixture.compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        fixture.compiled.transcript_encoding(),
        fixture.compiled.identity().canonical_encoding(),
    )
    .unwrap();

    let binding_index = fixture
        .placement
        .storage_bindings
        .iter()
        .position(|binding| {
            binding.value.version == version
                && binding.value.elements == transcript_binding.elements
        })
        .unwrap();
    let mut tail = fixture.placement.storage_bindings[binding_index];
    fixture.placement.storage_bindings[binding_index]
        .value
        .elements
        .end = split;
    tail.value.elements.start = split;
    tail.offset_bytes = 0;

    let storage = StorageId(fixture.placement.storages.len() as u32);
    let value = fixture.compiled.value(version).unwrap();
    fixture.placement.storages.push(StorageDesc {
        id: storage,
        worker: WorkerId(0),
        bytes: tail.value.elements.len() * value.layout.element.bytes,
        alignment_bytes: value.alignment,
    });
    tail.storage = storage;
    fixture.placement.storage_bindings.push(tail);

    assert_eq!(
        compile(fixture).unwrap_err(),
        FleetPlanError::TranscriptValueCausality(version)
    );
}

fn split_owner_and_storage(fixture: &mut Fixture, version: ValueVersion, split: usize) {
    let owner = fixture
        .placement
        .owners
        .iter_mut()
        .find(|owner| owner.value.version == version)
        .unwrap();
    let mut tail = *owner;
    owner.value.elements.end = split;
    tail.value.elements.start = split;
    fixture.placement.owners.push(tail);

    let binding = fixture
        .placement
        .storage_bindings
        .iter_mut()
        .find(|binding| binding.value.version == version)
        .unwrap();
    let mut tail = binding.clone();
    binding.value.elements.end = split;
    tail.value.elements.start = split;
    tail.offset_bytes = split * size_of::<u32>();
    fixture.placement.storage_bindings.push(tail);
}

fn swap(worker: WorkerId) -> WorkerId {
    match worker {
        WorkerId(0) => WorkerId(1),
        WorkerId(1) => WorkerId(0),
        worker => worker,
    }
}
