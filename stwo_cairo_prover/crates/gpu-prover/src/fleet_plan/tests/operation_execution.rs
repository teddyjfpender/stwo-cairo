use super::*;

const EXACT_OPERATION: OpId = OpId(0);
const MONOLITHIC_OPERATION: OpId = OpId(1);

pub(super) fn exact_fixture() -> Fixture {
    let mut fixture = fixture();
    let mut input = fixture.compiled.input().clone();
    for value in &mut input.values {
        if value.origin == ValueOrigin::OpOutput(OP_ASSEMBLE) {
            value.origin = ValueOrigin::OpOutput(MONOLITHIC_OPERATION);
        }
    }

    let exact_value = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        exact_value,
        8,
        4,
        ValueOrigin::OpOutput(EXACT_OPERATION),
        Region::Dynamic,
    ));
    let exact_effect = EffectContract::new(
        vec![EffectAccess::Write {
            destination: bound(0, value_range(exact_value, 8)),
        }],
        vec![],
    )
    .unwrap();
    let exact_authority = ExactPartitionAuthority::new(
        0,
        range(0, 8),
        2,
        4,
        vec![PartitionEffectProjection::ContiguousAxisSlice {
            binding: EffectBindingId(0),
        }],
        PartitionLaunchDerivation::new(1, 2, PartitionGridAxis::X, 2).unwrap(),
    )
    .unwrap();
    let exact_partition = PartitionAuthority::exact(exact_authority).unwrap();

    let mut monolithic = input.operations[0].clone();
    monolithic.id = MONOLITHIC_OPERATION;
    let exact_invocation = AotInvocation {
        arguments: vec![
            AotArgumentBinding {
                ordinal: 0,
                value: AotArgumentValue::DevicePointerTable(vec![Some(EffectBindingId(0))]),
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
    };
    let kernel = input.kernels[0].clone();
    let mut accepted = kernel.accepted_executions().to_vec();
    accepted.push((
        exact_effect.id(),
        exact_partition.id(),
        exact_invocation.contract_id().unwrap(),
    ));
    accepted.sort_unstable();
    input.kernels[0] = AotKernelAuthority::new_with_accepted_executions(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        accepted,
    )
    .unwrap();
    input.operations = vec![
        OpNode {
            id: EXACT_OPERATION,
            semantic_id: SemanticOpId(2),
            primitive: ExecutionPrimitive::AotKernel {
                kernel: kernel.id(),
                launch: LaunchGeometry {
                    grid: [4, 1, 1],
                    block: [128, 1, 1],
                    cluster: None,
                    dynamic_shared_bytes: 0,
                    cooperative: false,
                },
            },
            invocation: Some(exact_invocation),
            effect: exact_effect.id(),
            partition: exact_partition.id(),
            stage: ProofStage::BeforeTranscript(transcript().segments()[0].segment),
        },
        monolithic,
    ];
    input.effects.push(exact_effect);
    input.effects.sort_unstable_by_key(EffectContract::id);
    input.partitions.push(exact_partition);
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

    fixture.placement.operations[0].operation = MONOLITHIC_OPERATION;
    fixture.placement.operations.push(FleetOperationPlacement {
        operation: EXACT_OPERATION,
        during: during(1, 2),
        executions: vec![
            FleetOperationExecution {
                worker: WorkerId(0),
                domain: OperationDomain::Exact(range(0, 4)),
            },
            FleetOperationExecution {
                worker: WorkerId(1),
                domain: OperationDomain::Exact(range(4, 8)),
            },
        ],
    });
    add_second_worker(&mut fixture);

    let live = during(1, 3);
    fixture.placement.owners.extend([
        FleetOwnerPlacement {
            value: ValueRange {
                version: exact_value,
                elements: range(0, 4),
            },
            worker: WorkerId(0),
            live,
        },
        FleetOwnerPlacement {
            value: ValueRange {
                version: exact_value,
                elements: range(4, 8),
            },
            worker: WorkerId(1),
            live,
        },
    ]);
    add_exact_storage(&mut fixture, exact_value, WorkerId(0), range(0, 4));
    add_exact_storage(&mut fixture, exact_value, WorkerId(1), range(4, 8));
    fixture.placement.topology.workers[0].capacity_bytes += 16;
    fixture
}

fn exact_required_alias_fixture() -> Fixture {
    let mut fixture = exact_fixture();
    let mut input = fixture.compiled.input().clone();
    let destination = ValueVersion(input.values.len() as u32 - 1);
    let source = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        source,
        8,
        4,
        ValueOrigin::ExternalInput(ExternalInputId(2_000_000)),
        Region::Input,
    ));
    let required = EffectContract::new(
        vec![EffectAccess::ReadWrite {
            source: bound(0, value_range(source, 8)),
            destination: bound(1, value_range(destination, 8)),
            in_place: Some(InPlaceAliasAuthority {
                id: InPlaceAliasId(0),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
            }),
        }],
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
            PartitionEffectProjection::ContiguousAxisSlice {
                binding: EffectBindingId(1),
            },
        ],
        PartitionLaunchDerivation::new(1, 2, PartitionGridAxis::X, 2).unwrap(),
    )
    .unwrap();
    let partition = PartitionAuthority::exact(authority).unwrap();
    input.operations[0].effect = required.id();
    input.operations[0].partition = partition.id();
    input.operations[0].invocation = Some(AotInvocation {
        arguments: vec![
            AotArgumentBinding {
                ordinal: 0,
                value: AotArgumentValue::DevicePointerTable(vec![
                    Some(EffectBindingId(0)),
                    Some(EffectBindingId(1)),
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
            required.id(),
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
        .find(|effect| effect.id() == monolithic.0)
        .unwrap()
        .clone();
    let monolithic_partition = input
        .partitions
        .iter()
        .find(|candidate| candidate.id() == monolithic.1)
        .unwrap()
        .clone();
    input.effects = vec![required, monolithic_effect];
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
    fixture
}

fn add_second_worker(fixture: &mut Fixture) {
    fixture.placement.topology.workers.push(WorkerSpec {
        id: WorkerId(1),
        capacity_bytes: 1024,
        exchange_reserve_bytes: 0,
    });
    for (ordinal, release) in fixture.placement.barrier_steps.iter().copied().enumerate() {
        fixture.placement.barrier_arrivals.push(BarrierArrival {
            barrier_ordinal: ordinal as u32,
            worker: WorkerId(1),
            ready_step: ScheduleStep(release.0 - 1),
        });
    }
    fixture.placement.barrier_arrivals.push(BarrierArrival {
        barrier_ordinal: fixture.placement.barrier_steps.len() as u32,
        worker: WorkerId(1),
        ready_step: *fixture.placement.barrier_steps.last().unwrap(),
    });
}

fn add_exact_storage(
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

fn expect_domain_error(fixture: Fixture, operation: OpId) {
    assert_eq!(
        compile(fixture).unwrap_err(),
        FleetPlanError::InvalidOperationDomain(operation)
    );
}

#[test]
fn exact_execution_is_a_canonical_two_worker_union() {
    let baseline = compile(exact_fixture()).unwrap();
    let executions = &baseline.placement().operations[EXACT_OPERATION.0 as usize].executions;
    assert_eq!(executions.len(), 2);
    assert_eq!(executions[0].domain, OperationDomain::Exact(range(0, 4)));
    assert_eq!(executions[1].domain, OperationDomain::Exact(range(4, 8)));

    let mut reordered = exact_fixture();
    reordered.placement.operations[1].executions.reverse();
    let reordered = compile(reordered).unwrap();
    assert_eq!(baseline.identity(), reordered.identity());
    assert_eq!(
        baseline.canonical_bytes().unwrap(),
        reordered.canonical_bytes().unwrap()
    );
}

#[test]
fn exact_write_rejects_gap_free_union_across_distinct_storages() {
    let mut fixture = exact_fixture();
    let exact_value = ValueVersion(fixture.compiled.values().len() as u32 - 1);
    let binding_index = fixture
        .placement
        .storage_bindings
        .iter()
        .position(|binding| binding.value == value_range(exact_value, 4))
        .unwrap();
    let original = fixture.placement.storage_bindings[binding_index].storage;
    fixture.placement.storage_bindings[binding_index]
        .value
        .elements
        .end = 2;
    fixture.placement.storages[original.0 as usize].bytes = 2 * size_of::<u32>();

    let separate = StorageId(fixture.placement.storages.len() as u32);
    fixture.placement.storages.push(StorageDesc {
        id: separate,
        worker: WorkerId(0),
        bytes: 2 * size_of::<u32>(),
        alignment_bytes: 4,
    });
    fixture
        .placement
        .storage_bindings
        .push(FleetStoragePlacement {
            storage: separate,
            value: ValueRange {
                version: exact_value,
                elements: range(2, 4),
            },
            offset_bytes: 0,
        });

    assert_eq!(
        compile(fixture).unwrap_err(),
        FleetPlanError::InvalidProducer(exact_value)
    );
}

#[test]
fn exact_execution_identity_binds_workers_to_domains() {
    let baseline = compile(exact_fixture()).unwrap();
    let mut execution_only = baseline.clone();
    execution_only.placement.operations[0].executions[0].worker = WorkerId(1);
    execution_only.placement.operations[0].executions[1].worker = WorkerId(0);
    assert_ne!(
        baseline.canonical_bytes().unwrap(),
        super::super::identity::encode(&execution_only).unwrap()
    );

    let mut swapped = exact_fixture();
    let exact_value = ValueVersion(swapped.compiled.values().len() as u32 - 1);
    for execution in &mut swapped.placement.operations[1].executions {
        execution.worker = match execution.worker {
            WorkerId(0) => WorkerId(1),
            WorkerId(1) => WorkerId(0),
            worker => worker,
        };
    }
    for owner in swapped
        .placement
        .owners
        .iter_mut()
        .filter(|owner| owner.value.version == exact_value)
    {
        owner.worker = match owner.worker {
            WorkerId(0) => WorkerId(1),
            WorkerId(1) => WorkerId(0),
            worker => worker,
        };
    }
    for binding in swapped
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| binding.value.version == exact_value)
    {
        let storage = &mut swapped.placement.storages[binding.storage.0 as usize];
        storage.worker = match storage.worker {
            WorkerId(0) => WorkerId(1),
            WorkerId(1) => WorkerId(0),
            worker => worker,
        };
    }
    let swapped = compile(swapped).unwrap();
    assert_ne!(baseline.identity(), swapped.identity());
    assert_ne!(
        baseline.canonical_bytes().unwrap(),
        swapped.canonical_bytes().unwrap()
    );
}

#[test]
fn exact_execution_rejects_owner_bound_to_the_wrong_worker_domain() {
    let mut mismatched = exact_fixture();
    let exact_value = ValueVersion(mismatched.compiled.values().len() as u32 - 1);
    mismatched.placement.operations[1].executions[0].worker = WorkerId(1);
    mismatched.placement.operations[1].executions[1].worker = WorkerId(0);
    assert_eq!(
        compile(mismatched).unwrap_err(),
        FleetPlanError::InvalidProducer(exact_value)
    );
}

#[test]
fn exact_execution_rejects_missing_gap_and_overlap() {
    let mut missing = exact_fixture();
    missing.placement.operations[1].executions.clear();
    expect_domain_error(missing, EXACT_OPERATION);

    let mut gap = exact_fixture();
    gap.placement.operations[1].executions[0].domain = OperationDomain::Exact(range(0, 2));
    expect_domain_error(gap, EXACT_OPERATION);

    let mut overlap = exact_fixture();
    overlap.placement.operations[1].executions[0].domain = OperationDomain::Exact(range(0, 6));
    expect_domain_error(overlap, EXACT_OPERATION);
}

#[test]
fn exact_execution_rejects_misalignment_and_duplicate_worker() {
    let mut misaligned = exact_fixture();
    misaligned.placement.operations[1].executions[0].domain = OperationDomain::Exact(range(0, 3));
    misaligned.placement.operations[1].executions[1].domain = OperationDomain::Exact(range(3, 8));
    expect_domain_error(misaligned, EXACT_OPERATION);

    let mut duplicate = exact_fixture();
    duplicate.placement.operations[1].executions[1].worker = WorkerId(0);
    expect_domain_error(duplicate, EXACT_OPERATION);
}

#[test]
fn exact_execution_rejects_mixed_domain_kinds() {
    let mut mixed = exact_fixture();
    mixed.placement.operations[1].executions[0].domain = OperationDomain::Monolithic;
    expect_domain_error(mixed, EXACT_OPERATION);
}

#[test]
fn monolithic_execution_requires_exactly_one_coordinator_execution() {
    let mut missing = exact_fixture();
    missing.placement.operations[0].executions.clear();
    expect_domain_error(missing, MONOLITHIC_OPERATION);

    let mut duplicate = exact_fixture();
    duplicate.placement.operations[0]
        .executions
        .push(FleetOperationExecution {
            worker: WorkerId(0),
            domain: OperationDomain::Monolithic,
        });
    expect_domain_error(duplicate, MONOLITHIC_OPERATION);

    let mut away = exact_fixture();
    away.placement.operations[0].executions[0].worker = WorkerId(1);
    expect_domain_error(away, MONOLITHIC_OPERATION);
}

#[test]
fn exact_execution_rejects_required_in_place_alias() {
    expect_domain_error(exact_required_alias_fixture(), EXACT_OPERATION);
}

#[test]
fn exact_spill_reclaim_conflicts_are_projected_to_each_worker_shard() {
    let plan = compile(exact_fixture()).unwrap();
    let operation = &plan.placement().operations[EXACT_OPERATION.0 as usize];
    let version = ValueVersion(plan.compiled().values().len() as u32 - 1);
    let shard = |start, end| ValueRange {
        version,
        elements: range(start, end),
    };

    assert!(super::super::validate::operation_access_during(
        &plan,
        WorkerId(0),
        shard(0, 4),
        operation.during,
    )
    .unwrap());
    assert!(!super::super::validate::operation_access_during(
        &plan,
        WorkerId(0),
        shard(4, 8),
        operation.during,
    )
    .unwrap());
    assert!(super::super::validate::operation_access_during(
        &plan,
        WorkerId(1),
        shard(4, 8),
        operation.during,
    )
    .unwrap());
    assert!(!super::super::validate::operation_access_during(
        &plan,
        WorkerId(1),
        shard(0, 4),
        operation.during,
    )
    .unwrap());
}
