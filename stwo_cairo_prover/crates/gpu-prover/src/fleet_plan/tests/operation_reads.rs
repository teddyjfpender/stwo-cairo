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
        1,
        4,
        ValueOrigin::Constant(ConstantId(0)),
        Region::FixedData,
    ));
    input.fixed_values = vec![FixedValueDesc::inline_u32(
        ConstantId(0),
        replicated,
        vec![7],
    )];

    let effect = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(0, value_range(sliced, 8)),
            },
            EffectAccess::Read {
                source: bound(1, value_range(replicated, 1)),
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
    let monolithic = (input.operations[1].effect, input.operations[1].partition);
    let mut accepted = vec![(effect.id(), partition.id()), monolithic];
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
                elements: range(0, 4),
            },
            worker: WorkerId(0),
            live: during(0, terminal.0),
        },
        FleetOwnerPlacement {
            value: ValueRange {
                version: sliced,
                elements: range(4, 8),
            },
            worker: WorkerId(1),
            live: during(0, terminal.0),
        },
        FleetOwnerPlacement {
            value: value_range(replicated, 1),
            worker: WorkerId(0),
            live: during(0, terminal.0),
        },
    ]);
    add_storage(&mut fixture, sliced, WorkerId(0), range(0, 4));
    add_storage(&mut fixture, sliced, WorkerId(1), range(4, 8));
    add_storage(&mut fixture, replicated, WorkerId(0), range(0, 1));
    add_storage(&mut fixture, replicated, WorkerId(1), range(0, 1));
    fixture.placement.replicas.push(FleetReplicaPlacement {
        id: ReplicaId(0),
        value: value_range(replicated, 1),
        canonical_worker: WorkerId(0),
        worker: WorkerId(1),
        layout: fixture.compiled.value(replicated).unwrap().layout.clone(),
        origin: ReplicaOrigin::InstalledFixed,
        live: during(0, terminal.0),
    });
    fixture.placement.topology.workers[0].capacity_bytes += 20;
    fixture.placement.topology.workers[1].capacity_bytes += 20;
    (fixture, sliced, replicated)
}

fn add_storage(
    fixture: &mut Fixture,
    version: ValueVersion,
    worker: WorkerId,
    elements: ElementRange,
) {
    let storage = StorageId(fixture.placement.storages.len() as u32);
    fixture.placement.storages.push(StorageDesc {
        id: storage,
        worker,
        bytes: elements.len() * size_of::<u32>(),
        alignment_bytes: 4,
    });
    fixture
        .placement
        .storage_bindings
        .push(FleetStoragePlacement {
            storage,
            value: ValueRange { version, elements },
            offset_bytes: 0,
        });
}

#[test]
fn exact_sliced_and_replicated_reads_are_available_locally() {
    let (fixture, ..) = exact_read_fixture();
    compile(fixture).unwrap();
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
    for binding in fixture
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| binding.value.version == sliced)
    {
        let storage = &mut fixture.placement.storages[binding.storage.0 as usize];
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
fn exact_replicated_read_requires_a_full_local_replica() {
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

fn swap(worker: WorkerId) -> WorkerId {
    match worker {
        WorkerId(0) => WorkerId(1),
        WorkerId(1) => WorkerId(0),
        worker => worker,
    }
}
