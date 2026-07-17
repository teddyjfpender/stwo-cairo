use stwo_backend_cuda::IPC_EXCHANGE_ALLOCATION_ALIGNMENT;

use super::*;

fn add_second_worker(fixture: &mut Fixture) {
    let worker = WorkerId(1);
    fixture.placement.topology.workers.push(WorkerSpec {
        id: worker,
        capacity_bytes: 1024,
        exchange_reserve_bytes: 0,
    });
    let releases = fixture.placement.barrier_steps.clone();
    fixture
        .placement
        .barrier_arrivals
        .extend(
            releases
                .iter()
                .enumerate()
                .map(|(ordinal, release)| BarrierArrival {
                    barrier_ordinal: ordinal as u32,
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

pub(super) fn transfer_fixture() -> Fixture {
    let mut fixture = fixture();
    add_second_worker(&mut fixture);
    let value = fixture.compiled.value(fixture.spill_value).unwrap();
    let elements = range(0, value.layout.element_count().unwrap());
    let logical_bytes = value.layout.logical_bytes().unwrap();
    fixture.placement.topology.links.push(FleetLink {
        id: FleetLinkId(0),
        source: WorkerId(0),
        destination: WorkerId(1),
        max_transfer_bytes: logical_bytes,
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
    let destination_storage = StorageId(fixture.placement.storages.len() as u32);
    fixture.placement.storages.push(StorageDesc {
        id: destination_storage,
        worker: WorkerId(1),
        bytes: logical_bytes,
        alignment_bytes: value.alignment,
    });
    fixture
        .placement
        .storage_bindings
        .push(FleetStoragePlacement {
            storage: destination_storage,
            value: ValueRange {
                version: value.version,
                elements,
            },
            offset_bytes: 0,
        });
    fixture.placement.topology.workers[0].exchange_reserve_bytes =
        IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[0].capacity_bytes += IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture
}

pub(super) fn split_transfer_fixture() -> Fixture {
    let mut fixture = transfer_fixture();
    let source_storage = StorageId(fixture.spill_value.0);
    let destination_storage = fixture.placement.storages.last().unwrap().id;
    let version = fixture.spill_value;

    fixture.placement.storage_bindings.retain(|binding| {
        !(binding.value.version == version
            && (binding.storage == source_storage || binding.storage == destination_storage))
    });
    fixture.placement.storage_bindings.extend([
        FleetStoragePlacement {
            storage: source_storage,
            value: ValueRange {
                version,
                elements: range(0, 8),
            },
            offset_bytes: 0,
        },
        FleetStoragePlacement {
            storage: source_storage,
            value: ValueRange {
                version,
                elements: range(8, 16),
            },
            offset_bytes: 64,
        },
        FleetStoragePlacement {
            storage: destination_storage,
            value: ValueRange {
                version,
                elements: range(0, 4),
            },
            offset_bytes: 0,
        },
        FleetStoragePlacement {
            storage: destination_storage,
            value: ValueRange {
                version,
                elements: range(4, 12),
            },
            offset_bytes: 64,
        },
        FleetStoragePlacement {
            storage: destination_storage,
            value: ValueRange {
                version,
                elements: range(12, 16),
            },
            offset_bytes: 128,
        },
    ]);

    let source = fixture
        .placement
        .storages
        .iter_mut()
        .find(|storage| storage.id == source_storage)
        .unwrap();
    fixture.placement.topology.workers[0].capacity_bytes += 96 - source.bytes;
    source.bytes = 96;
    let destination = fixture
        .placement
        .storages
        .iter_mut()
        .find(|storage| storage.id == destination_storage)
        .unwrap();
    fixture.placement.topology.workers[1].capacity_bytes += 144 - destination.bytes;
    destination.bytes = 144;

    let old_reserve = fixture.placement.topology.workers[0].exchange_reserve_bytes;
    let new_reserve = 4 * IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[0].exchange_reserve_bytes = new_reserve;
    fixture.placement.topology.workers[0].capacity_bytes += new_reserve - old_reserve;
    fixture
}

#[test]
fn one_transition_derives_one_exact_span_and_owner_reserve() {
    let plan = compile(transfer_fixture()).unwrap();
    let view = plan.runtime_view().unwrap();
    let span = view.spans()[0];
    assert_eq!(view.plan_identity(), plan.identity());
    assert_eq!(view.spans().len(), 1);
    assert_eq!(span.edge_ordinal, 0);
    assert_eq!(span.transition, LayoutTransitionId(0));
    assert_eq!(span.span_ordinal, 0);
    assert_eq!(span.route, FleetLinkId(0));
    assert_eq!((span.owner, span.peer), (WorkerId(0), WorkerId(1)));
    assert_eq!(span.elements, range(0, 16));
    assert_eq!(span.logical_bytes(), 64);
    assert_eq!(
        span.source.storage,
        StorageId(plan.compiled().values().len() as u32 - 2)
    );
    assert_eq!(span.source.offset_bytes, 0);
    assert_eq!(span.destination.offset_bytes, 0);
    assert_eq!(
        view.exchange_reserves(),
        [
            FleetExchangeReserve {
                worker: WorkerId(0),
                required_bytes: IPC_EXCHANGE_ALLOCATION_ALIGNMENT,
                declared_bytes: IPC_EXCHANGE_ALLOCATION_ALIGNMENT,
            },
            FleetExchangeReserve {
                worker: WorkerId(1),
                required_bytes: 0,
                declared_bytes: 0,
            },
        ]
    );
}

#[test]
fn common_refinement_is_dense_and_independent_of_binding_order() {
    let plan = compile(split_transfer_fixture()).unwrap();
    let expected = plan.runtime_view().unwrap();
    assert_eq!(expected.spans().len(), 4);
    assert_eq!(
        expected
            .spans()
            .iter()
            .map(|span| {
                (
                    span.edge_ordinal,
                    span.span_ordinal,
                    span.elements,
                    span.source.offset_bytes,
                    span.destination.offset_bytes,
                    span.logical_bytes(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (0, 0, range(0, 4), 0, 0, 16),
            (1, 1, range(4, 8), 16, 64, 16),
            (2, 2, range(8, 12), 64, 80, 16),
            (3, 3, range(12, 16), 80, 128, 16),
        ]
    );

    let mut reordered = plan.clone();
    reordered.placement.storage_bindings.reverse();
    reordered.placement.transitions.reverse();
    assert_eq!(reordered.runtime_view().unwrap(), expected);
}

#[test]
fn missing_and_ambiguous_windows_fail_closed() {
    let plan = compile(split_transfer_fixture()).unwrap();
    let source_storage = StorageId(plan.compiled().values().len() as u32 - 2);

    let mut gap = plan.clone();
    gap.placement.storage_bindings.retain(|binding| {
        !(binding.storage == source_storage && binding.value.elements == range(8, 16))
    });
    assert!(matches!(
        gap.runtime_view(),
        Err(FleetRuntimeViewError::MissingStorageWindow {
            transition: LayoutTransitionId(0),
            worker: WorkerId(0),
            ..
        })
    ));

    let mut ambiguous = plan;
    let duplicate = *ambiguous
        .placement
        .storage_bindings
        .iter()
        .find(|binding| binding.storage == source_storage)
        .unwrap();
    ambiguous.placement.storage_bindings.push(duplicate);
    assert!(matches!(
        ambiguous.runtime_view(),
        Err(FleetRuntimeViewError::AmbiguousStorageWindow {
            transition: LayoutTransitionId(0),
            worker: WorkerId(0),
            ..
        })
    ));
}

#[test]
fn rounded_exchange_reserve_is_exact_and_near_miss_rejects() {
    let plan = compile(split_transfer_fixture()).unwrap();
    let view = plan.runtime_view().unwrap();
    assert_eq!(
        view.exchange_reserves()[0].required_bytes,
        4 * IPC_EXCHANGE_ALLOCATION_ALIGNMENT
    );

    let mut short = plan;
    short.placement.topology.workers[0].exchange_reserve_bytes =
        4 * IPC_EXCHANGE_ALLOCATION_ALIGNMENT - 1;
    assert_eq!(
        short.runtime_view().unwrap_err(),
        FleetRuntimeViewError::ExchangeReserveExceeded {
            worker: WorkerId(0),
            required: 4 * IPC_EXCHANGE_ALLOCATION_ALIGNMENT,
            declared: 4 * IPC_EXCHANGE_ALLOCATION_ALIGNMENT - 1,
        }
    );
}

#[test]
fn plan_construction_enforces_the_derived_exchange_reserve() {
    let exact = split_transfer_fixture();
    let required = 4 * IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    assert_eq!(
        exact.placement.topology.workers[0].exchange_reserve_bytes,
        required
    );
    compile(exact).unwrap();

    let mut short = split_transfer_fixture();
    short.placement.topology.workers[0].exchange_reserve_bytes = required - 1;
    assert_eq!(
        compile(short).unwrap_err(),
        FleetPlanError::RuntimeView(FleetRuntimeViewError::ExchangeReserveExceeded {
            worker: WorkerId(0),
            required,
            declared: required - 1,
        })
    );
}
