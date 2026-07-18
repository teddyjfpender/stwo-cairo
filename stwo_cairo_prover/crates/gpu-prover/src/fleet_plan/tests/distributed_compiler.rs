use stwo_backend_cuda::IPC_EXCHANGE_ALLOCATION_ALIGNMENT;

use super::*;

fn compile_partitioned(fixture: Fixture) -> Result<FleetProofPlan, FleetCompileError> {
    FleetProofPlan::compile_track_a_partitioned(
        fixture.compiled,
        fixture.shape,
        fixture.placement.topology,
        fixture.placement.pow,
        transcript(),
    )
}

pub(super) fn transfer_fixture() -> Fixture {
    transfer_fixture_with_reads(None)
}

fn transfer_fixture_with_reads(reads: Option<[ElementRange; 2]>) -> Fixture {
    let mut fixture = super::operation_execution::exact_fixture();
    let mut input = fixture.compiled.input().clone();
    let exact = input.operations[0].clone();
    let mut consumer = input.operations[1].clone();
    let exact_value = input.values.last().unwrap().version;
    let exact_words = input.values.last().unwrap().layout.element_count().unwrap();
    let [consumer_read, observer_read] =
        reads.unwrap_or([range(0, exact_words), range(0, exact_words)]);
    let previous_effect = input
        .effects
        .iter()
        .find(|effect| effect.id() == consumer.effect)
        .unwrap();
    let mut accesses = previous_effect.accesses().to_vec();
    accesses.push(EffectAccess::Read {
        source: bound(
            accesses.len() as u32,
            ValueRange {
                version: exact_value,
                elements: consumer_read,
            },
        ),
    });
    let consumer_effect =
        EffectContract::new(accesses, previous_effect.module_globals().to_vec()).unwrap();
    consumer.effect = consumer_effect.id();
    consumer.invocation = invocation(&consumer_effect);
    let observer_effect = EffectContract::new(
        vec![EffectAccess::Read {
            source: bound(
                0,
                ValueRange {
                    version: exact_value,
                    elements: observer_read,
                },
            ),
        }],
        vec![],
    )
    .unwrap();
    let mut observer = consumer.clone();
    observer.id = OpId(2);
    observer.semantic_id = SemanticOpId(3);
    observer.effect = observer_effect.id();
    observer.invocation = invocation(&observer_effect);

    let exact_effect = input
        .effects
        .iter()
        .find(|effect| effect.id() == exact.effect)
        .unwrap()
        .clone();
    input.effects = vec![exact_effect, consumer_effect, observer_effect];
    input.effects.sort_unstable_by_key(EffectContract::id);
    input.operations = vec![exact.clone(), consumer.clone(), observer.clone()];

    let kernel = input.kernels[0].clone();
    let mut accepted = vec![
        (
            exact.effect,
            exact.partition,
            exact.invocation.as_ref().unwrap().contract_id().unwrap(),
        ),
        (
            consumer.effect,
            consumer.partition,
            consumer.invocation.as_ref().unwrap().contract_id().unwrap(),
        ),
        (
            observer.effect,
            observer.partition,
            observer.invocation.as_ref().unwrap().contract_id().unwrap(),
        ),
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

    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    fixture.compiled = compiled;

    fixture.placement.topology.links = vec![FleetLink {
        id: FleetLinkId(0),
        source: WorkerId(1),
        destination: WorkerId(0),
        max_transfer_bytes: exact_words * size_of::<u32>(),
    }];
    let reserve = 8 * IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[1].exchange_reserve_bytes = reserve;
    fixture.placement.topology.workers[1].capacity_bytes += reserve;
    fixture
}

fn route_limited_tail_fixture() -> (Fixture, ValueVersion, usize) {
    let mut fixture = super::operation_execution::exact_fixture();
    let mut input = fixture.compiled.input().clone();
    let exact_value = input.values.last().unwrap().version;
    let prior_output = input.output.sections[0].value;
    let words = input.output.layout.total_words;
    input.values[exact_value.0 as usize].layout.axes[0].extent = words;
    input.values[exact_value.0 as usize].region = Region::Output;
    input.values[prior_output.0 as usize].region = Region::Dynamic;
    for (section, fragment) in input
        .output
        .sections
        .iter_mut()
        .zip(&mut input.output.fragments)
    {
        section.value = exact_value;
        section.elements = range(fragment.destination.start, fragment.destination.end);
        fragment.source = ValueRange {
            version: exact_value,
            elements: section.elements,
        };
    }

    let exact_effect = EffectContract::new(
        vec![EffectAccess::Write {
            destination: bound(0, value_range(exact_value, words)),
        }],
        vec![],
    )
    .unwrap();
    let exact_authority = ExactPartitionAuthority::new(
        0,
        range(0, words),
        1,
        size_of::<u32>(),
        vec![PartitionEffectProjection::ContiguousAxisSlice {
            binding: EffectBindingId(0),
        }],
        PartitionLaunchDerivation::new(1, 2, PartitionGridAxis::X, 1).unwrap(),
    )
    .unwrap();
    let exact_partition = PartitionAuthority::exact(exact_authority).unwrap();
    input.operations[0].effect = exact_effect.id();
    input.operations[0].partition = exact_partition.id();
    input.operations[0].invocation = Some(AotInvocation {
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
                value: AotArgumentValue::U32(words as u32),
            },
        ],
    });
    let ExecutionPrimitive::AotKernel { launch, .. } = &mut input.operations[0].primitive else {
        unreachable!()
    };
    launch.grid[0] = words as u32;

    let monolithic = input.operations[1].clone();
    let monolithic_effect = input
        .effects
        .iter()
        .find(|effect| effect.id() == monolithic.effect)
        .unwrap()
        .clone();
    let monolithic_partition = input
        .partitions
        .iter()
        .find(|partition| partition.id() == monolithic.partition)
        .unwrap()
        .clone();
    input.effects = vec![exact_effect, monolithic_effect];
    input.effects.sort_unstable_by_key(EffectContract::id);
    input.partitions = vec![exact_partition, monolithic_partition];
    input
        .partitions
        .sort_unstable_by_key(PartitionAuthority::id);
    let kernel = input.kernels[0].clone();
    let mut accepted = input
        .operations
        .iter()
        .map(|operation| {
            (
                operation.effect,
                operation.partition,
                operation
                    .invocation
                    .as_ref()
                    .unwrap()
                    .contract_id()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    accepted.sort_unstable();
    input.kernels[0] = AotKernelAuthority::new_with_accepted_executions(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        accepted,
    )
    .unwrap();

    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    fixture.compiled = compiled;
    let route_words = words.div_ceil(4);
    fixture.placement.topology.links = vec![FleetLink {
        id: FleetLinkId(0),
        source: WorkerId(1),
        destination: WorkerId(0),
        max_transfer_bytes: route_words * size_of::<u32>(),
    }];
    let reserve = 8 * IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    for worker in &mut fixture.placement.topology.workers {
        worker.capacity_bytes += words * size_of::<u32>() + reserve;
    }
    fixture.placement.topology.workers[1].exchange_reserve_bytes = reserve;
    (fixture, exact_value, route_words)
}

#[test]
fn partitioned_compiler_segments_a_coalesced_demand_to_the_route_limit() {
    let mut fixture = transfer_fixture_with_reads(Some([range(4, 6), range(6, 8)]));
    fixture.placement.topology.links[0].max_transfer_bytes = 2 * size_of::<u32>();

    let plan = compile_partitioned(fixture).unwrap();
    assert_eq!(
        plan.placement()
            .transitions
            .iter()
            .map(|transition| transition.value.elements)
            .collect::<Vec<_>>(),
        [range(4, 6), range(6, 8)]
    );
    assert!(plan
        .placement()
        .replicas
        .iter()
        .all(|replica| replica.live.end == plan.placement().operations[2].during.end));
}

#[test]
fn partitioned_compiler_refines_output_bindings_at_segmented_tail_boundaries() {
    let (fixture, output, route_words) = route_limited_tail_fixture();
    let plan = compile_partitioned(fixture).unwrap();
    let tail = plan
        .placement()
        .transitions
        .iter()
        .filter(|transition| {
            transition.value.version == output
                && transition.interval == ExecutionInterval::AfterFinalBarrier
        })
        .collect::<Vec<_>>();
    assert!(tail.len() > 1);
    for transition in tail {
        assert!(transition.value.elements.len() <= route_words);
        let mut bindings = plan
            .placement()
            .storage_bindings
            .iter()
            .filter(|binding| {
                binding.storage == plan.placement().output_storage
                    && binding.value.version == transition.value.version
                    && binding.value.elements.start >= transition.value.elements.start
                    && binding.value.elements.end <= transition.value.elements.end
            })
            .collect::<Vec<_>>();
        bindings.sort_unstable_by_key(|binding| binding.value.elements.start);
        let mut cursor = transition.value.elements.start;
        for binding in bindings {
            assert_eq!(binding.value.elements.start, cursor);
            assert_eq!(
                binding.offset_bytes,
                binding.value.elements.start * size_of::<u32>()
            );
            cursor = binding.value.elements.end;
        }
        assert_eq!(cursor, transition.value.elements.end);
    }
}

#[test]
fn partitioned_compiler_deterministically_splits_exact_work_across_two_workers() {
    let baseline = compile_partitioned(super::operation_execution::exact_fixture()).unwrap();
    let exact = &baseline.placement().operations[0];
    assert_eq!(
        exact.executions,
        [
            FleetOperationExecution {
                worker: WorkerId(0),
                domain: OperationDomain::Exact(range(0, 4)),
            },
            FleetOperationExecution {
                worker: WorkerId(1),
                domain: OperationDomain::Exact(range(4, 8)),
            },
        ]
    );

    let mut reordered = super::operation_execution::exact_fixture();
    reordered.placement.topology.workers.reverse();
    reordered.placement.topology.links.reverse();
    let reordered = compile_partitioned(reordered).unwrap();
    assert_eq!(baseline.identity(), reordered.identity());
    assert_eq!(
        baseline.canonical_bytes().unwrap(),
        reordered.canonical_bytes().unwrap()
    );
}

#[test]
fn partitioned_compiler_rejects_missing_route_and_short_exchange_reserve() {
    let plan = compile_partitioned(transfer_fixture()).unwrap();
    assert_eq!(
        plan.placement().transitions.len(),
        1,
        "one immutable remote slice must be transferred once and reused"
    );
    assert_eq!(
        plan.placement().replicas[0].live.end,
        plan.placement().operations[2].during.end
    );
    let required = plan
        .runtime_view()
        .unwrap()
        .exchange_reserves()
        .iter()
        .find(|reserve| reserve.worker == WorkerId(1))
        .unwrap()
        .required_bytes;
    assert!(required > 0);

    let mut missing_route = transfer_fixture();
    missing_route.placement.topology.links.clear();
    assert!(compile_partitioned(missing_route).is_err());

    let mut short = transfer_fixture();
    short.placement.topology.workers[1].exchange_reserve_bytes = required - 1;
    assert!(compile_partitioned(short).is_err());
}

#[test]
fn partitioned_compiler_matches_monolithic_lowering_on_one_worker() {
    let mut fixture = super::operation_execution::exact_fixture();
    fixture.placement.topology.workers.truncate(1);
    fixture.placement.topology.links.clear();

    let partitioned = compile_partitioned(fixture.clone()).unwrap();
    let monolithic = FleetProofPlan::compile_track_a_monolithic(
        fixture.compiled,
        fixture.shape,
        fixture.placement.topology,
        fixture.placement.pow,
        transcript(),
    )
    .unwrap();
    assert_eq!(partitioned.identity(), monolithic.identity());
    assert_eq!(
        partitioned.canonical_bytes().unwrap(),
        monolithic.canonical_bytes().unwrap()
    );
}
