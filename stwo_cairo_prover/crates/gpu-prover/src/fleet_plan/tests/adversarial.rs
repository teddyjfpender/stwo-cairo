use super::*;
use stwo_backend_cuda::IPC_EXCHANGE_ALLOCATION_ALIGNMENT;

fn add_workers(fixture: &mut Fixture, count: usize) {
    let releases = fixture.placement.barrier_steps.clone();
    for ordinal in 1..count {
        let worker = WorkerId(ordinal as u16);
        fixture.placement.topology.workers.push(WorkerSpec {
            id: worker,
            capacity_bytes: 1024,
            exchange_reserve_bytes: 0,
        });
        fixture
            .placement
            .barrier_arrivals
            .extend(
                releases
                    .iter()
                    .enumerate()
                    .map(|(barrier_ordinal, release)| BarrierArrival {
                        barrier_ordinal: barrier_ordinal as u32,
                        worker,
                        ready_step: ScheduleStep(release.0 - 1),
                    }),
            );
        fixture.placement.barrier_arrivals.push(BarrierArrival {
            barrier_ordinal: releases.len() as u32,
            worker,
            ready_step: *releases.last().unwrap(),
        });
    }
}

fn transition_fixture() -> Fixture {
    let mut fixture = fixture();
    add_workers(&mut fixture, 2);
    fixture.placement.topology.workers[0].exchange_reserve_bytes =
        IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[0].capacity_bytes += IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    let value = fixture.compiled.value(fixture.spill_value).unwrap();
    let elements = range(0, value.layout.element_count().unwrap());
    fixture.placement.topology.links.push(FleetLink {
        id: FleetLinkId(0),
        source: WorkerId(0),
        destination: WorkerId(1),
        max_transfer_bytes: value.layout.logical_bytes().unwrap(),
    });
    fixture.placement.replicas.push(FleetReplicaPlacement {
        id: ReplicaId(0),
        value: ValueRange {
            version: value.version,
            elements,
        },
        canonical_worker: WorkerId(0),
        worker: WorkerId(1),
        layout: value.layout.clone(),
        origin: ReplicaOrigin::Transition(LayoutTransitionId(0)),
        live: during(1, 50),
    });
    fixture
        .placement
        .transitions
        .push(FleetTransitionPlacement {
            id: LayoutTransitionId(0),
            value: ValueRange {
                version: value.version,
                elements,
            },
            source_worker: WorkerId(0),
            destination_replica: ReplicaId(0),
            axes: value
                .layout
                .axes
                .iter()
                .map(|axis| AxisMap {
                    source: axis.tag,
                    destination: axis.tag,
                })
                .collect(),
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(1, 2),
            scratch_bytes: 8,
            scratch_worker: WorkerId(1),
            route: FleetLinkId(0),
        });
    let storage = StorageId(fixture.placement.storages.len() as u32);
    fixture.placement.storages.push(StorageDesc {
        id: storage,
        worker: WorkerId(1),
        bytes: value.layout.logical_bytes().unwrap(),
        alignment_bytes: value.alignment,
    });
    fixture
        .placement
        .storage_bindings
        .push(FleetStoragePlacement {
            storage,
            value: ValueRange {
                version: value.version,
                elements,
            },
            offset_bytes: 0,
        });
    fixture
}

#[test]
fn homogeneous_worker_matrix_accepts_only_supported_fleet_widths() {
    for count in [1, 2, 4, 8, 16] {
        let mut candidate = fixture();
        add_workers(&mut candidate, count);
        let plan = compile(candidate).unwrap();
        assert_eq!(plan.workers().len(), count);
    }

    let mut unsupported = fixture();
    add_workers(&mut unsupported, 3);
    assert_eq!(
        compile(unsupported).unwrap_err(),
        FleetPlanError::EmptyTopology
    );
}

#[test]
fn topology_and_barrier_contracts_fail_closed() {
    let mut zero_module = fixture();
    zero_module.placement.topology.module_pack_identity = [0; 32];
    assert_eq!(
        compile(zero_module).unwrap_err(),
        FleetPlanError::InvalidHomogeneousTopology
    );

    let mut oversized = fixture();
    oversized.placement.topology.workers[0].capacity_bytes = (21usize << 30) + 1;
    assert_eq!(
        compile(oversized).unwrap_err(),
        FleetPlanError::NonDenseWorkers
    );

    let mut missing_arrival = fixture();
    missing_arrival.placement.barrier_arrivals.pop();
    assert_eq!(
        compile(missing_arrival).unwrap_err(),
        FleetPlanError::TranscriptMismatch
    );

    let mut early_terminal = fixture();
    let terminal = early_terminal.placement.barrier_steps.len() as u32;
    let arrival = early_terminal
        .placement
        .barrier_arrivals
        .iter_mut()
        .find(|arrival| arrival.barrier_ordinal == terminal)
        .unwrap();
    arrival.ready_step = *early_terminal.placement.barrier_steps.last().unwrap();
    assert_eq!(
        compile(early_terminal).unwrap_err(),
        FleetPlanError::TranscriptMismatch
    );

    let mut wrong_interval = fixture();
    wrong_interval.placement.operations[0].during = during(1, 2);
    assert_eq!(
        compile(wrong_interval).unwrap_err(),
        FleetPlanError::InvalidSchedule
    );
}

#[test]
fn transition_routes_layouts_and_scratch_are_exact() {
    compile(transition_fixture()).unwrap();

    let mut undersized = transition_fixture();
    undersized.placement.topology.links[0].max_transfer_bytes -= 1;
    assert_eq!(
        compile(undersized).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );

    let mut reversed = transition_fixture();
    reversed.placement.topology.links[0].source = WorkerId(1);
    reversed.placement.topology.links[0].destination = WorkerId(0);
    assert_eq!(
        compile(reversed).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );

    let mut transformed = transition_fixture();
    transformed.placement.replicas[0].layout.axes[0].tag = 9;
    assert_eq!(
        compile(transformed).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );

    let mut phantom_scratch = transition_fixture();
    phantom_scratch.placement.transitions[0].scratch_worker = WorkerId(9);
    assert_eq!(
        compile(phantom_scratch).unwrap_err(),
        FleetPlanError::UnknownWorker(WorkerId(9))
    );
}
