use stwo_backend_cuda::IPC_EXCHANGE_ALLOCATION_ALIGNMENT;

use super::runtime_view::transfer_fixture;
use super::*;

const PROOF_GENERATION: u64 = 41;

fn view() -> FleetRuntimeView {
    compile(transfer_fixture()).unwrap().runtime_view().unwrap()
}

fn two_wave_view() -> FleetRuntimeView {
    let mut fixture = transfer_fixture();
    let second = fixture
        .placement
        .owners
        .iter()
        .find(|owner| {
            owner.live.start == ScheduleStep(0) && owner.value.version != fixture.spill_value
        })
        .unwrap()
        .value;
    let value = fixture.compiled.value(second.version).unwrap();
    let layout = value.layout.clone();
    let alignment = value.alignment;
    let bytes = layout.logical_bytes().unwrap();
    let storage = StorageId(fixture.placement.storages.len() as u32);

    fixture.placement.topology.links[0].max_transfer_bytes = fixture.placement.topology.links[0]
        .max_transfer_bytes
        .max(bytes);
    fixture.placement.topology.workers[0].exchange_reserve_bytes +=
        IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[0].capacity_bytes += IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[1].capacity_bytes += bytes;
    fixture.placement.replicas.push(FleetReplicaPlacement {
        id: ReplicaId(1),
        value: second,
        canonical_worker: WorkerId(0),
        worker: WorkerId(1),
        layout: layout.clone(),
        origin: ReplicaOrigin::Transition(LayoutTransitionId(1)),
        live: during(3, 50),
    });
    fixture
        .placement
        .transitions
        .push(FleetTransitionPlacement {
            id: LayoutTransitionId(1),
            value: second,
            source_worker: WorkerId(0),
            destination_replica: ReplicaId(1),
            axes: layout
                .axes
                .iter()
                .map(|axis| AxisMap {
                    source: axis.tag,
                    destination: axis.tag,
                })
                .collect(),
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(3, 4),
            scratch_bytes: 8,
            scratch_worker: WorkerId(1),
            route: FleetLinkId(0),
        });
    fixture.placement.storages.push(StorageDesc {
        id: storage,
        worker: WorkerId(1),
        bytes,
        alignment_bytes: alignment,
    });
    fixture
        .placement
        .storage_bindings
        .push(FleetStoragePlacement {
            storage,
            value: second,
            offset_bytes: 0,
        });
    compile(fixture).unwrap().runtime_view().unwrap()
}

fn receipt(view: &FleetRuntimeView, edge: usize, phase: FleetIpcPhase) -> FleetIpcPhaseReceipt {
    FleetIpcPhaseReceipt::for_span(
        view.plan_identity(),
        PROOF_GENERATION,
        view.spans()[edge],
        phase,
    )
    .unwrap()
}

fn poisoned_by(
    mutate: impl FnOnce(&mut FleetIpcPhaseReceipt),
) -> (FleetIpcCursorError, FleetIpcCoordinatorCursor) {
    let view = view();
    let mut cursor = FleetIpcCoordinatorCursor::new(&view, PROOF_GENERATION).unwrap();
    let mut receipt = receipt(&view, 0, FleetIpcPhase::Published);
    mutate(&mut receipt);
    let error = cursor.accept(receipt).unwrap_err();
    assert_eq!(cursor.state(), FleetIpcAttemptState::Poisoned);
    (error, cursor)
}

#[test]
fn exact_four_phase_sequence_completes_and_armed_uses_next_generation() {
    let view = view();
    let mut cursor = FleetIpcCoordinatorCursor::new(&view, PROOF_GENERATION).unwrap();
    assert_eq!(cursor.active_wave_edges(), [0]);
    for phase in [
        FleetIpcPhase::Published,
        FleetIpcPhase::Consumed,
        FleetIpcPhase::Reclaimed,
    ] {
        assert_eq!(
            cursor.accept(receipt(&view, 0, phase)).unwrap(),
            FleetIpcCursorProgress::WavePending {
                step: ScheduleStep(1)
            }
        );
    }
    let armed = receipt(&view, 0, FleetIpcPhase::Armed);
    assert_eq!(armed.binding.generation, PROOF_GENERATION + 1);
    assert_eq!(
        cursor.accept(armed).unwrap(),
        FleetIpcCursorProgress::Complete {
            completed: ScheduleStep(1)
        }
    );
    assert_eq!(cursor.state(), FleetIpcAttemptState::Complete);
}

#[test]
fn same_step_spans_are_one_wave_without_edge_ordering() {
    let plan = compile(super::runtime_view::split_transfer_fixture()).unwrap();
    let view = plan.runtime_view().unwrap();
    let mut cursor = FleetIpcCoordinatorCursor::new(&view, PROOF_GENERATION).unwrap();
    assert_eq!(cursor.active_wave_edges(), [0, 1, 2, 3]);

    for phase in [
        FleetIpcPhase::Published,
        FleetIpcPhase::Consumed,
        FleetIpcPhase::Reclaimed,
        FleetIpcPhase::Armed,
    ] {
        for edge in [3, 1, 0, 2] {
            let progress = cursor.accept(receipt(&view, edge, phase)).unwrap();
            if phase != FleetIpcPhase::Armed || edge != 2 {
                assert_eq!(
                    progress,
                    FleetIpcCursorProgress::WavePending {
                        step: ScheduleStep(1)
                    }
                );
            }
        }
    }
    assert_eq!(cursor.state(), FleetIpcAttemptState::Complete);
}

#[test]
fn future_wave_poisoned_and_next_wave_opens_only_after_current_wave_arms() {
    let view = two_wave_view();
    assert_eq!(view.spans().len(), 2);

    let mut future = FleetIpcCoordinatorCursor::new(&view, PROOF_GENERATION).unwrap();
    assert_eq!(future.active_wave_edges(), [0]);
    assert_eq!(
        future
            .accept(receipt(&view, 1, FleetIpcPhase::Published))
            .unwrap_err(),
        FleetIpcCursorError::WrongWave {
            edge: 1,
            active_step: ScheduleStep(1),
        }
    );
    assert_eq!(future.state(), FleetIpcAttemptState::Poisoned);

    let mut cursor = FleetIpcCoordinatorCursor::new(&view, PROOF_GENERATION).unwrap();
    for phase in [
        FleetIpcPhase::Published,
        FleetIpcPhase::Consumed,
        FleetIpcPhase::Reclaimed,
    ] {
        cursor.accept(receipt(&view, 0, phase)).unwrap();
    }
    assert_eq!(
        cursor
            .accept(receipt(&view, 0, FleetIpcPhase::Armed))
            .unwrap(),
        FleetIpcCursorProgress::WaveComplete {
            completed: ScheduleStep(1),
            next: ScheduleStep(3),
        }
    );
    assert_eq!(cursor.active_wave_edges(), [1]);
}

#[test]
fn stale_future_and_duplicate_phases_poison_the_attempt() {
    let view = view();

    let mut future_receipt = receipt(&view, 0, FleetIpcPhase::Consumed);
    future_receipt.binding.generation += 1;
    future_receipt.worker = WorkerId(0);
    let mut future = FleetIpcCoordinatorCursor::new(&view, PROOF_GENERATION).unwrap();
    assert!(matches!(
        future.accept(future_receipt),
        Err(FleetIpcCursorError::OutOfOrderPhase {
            expected: Some(FleetIpcPhase::Published),
            actual: FleetIpcPhase::Consumed,
            ..
        })
    ));
    assert_eq!(future.state(), FleetIpcAttemptState::Poisoned);

    let mut duplicate = FleetIpcCoordinatorCursor::new(&view, PROOF_GENERATION).unwrap();
    let published = receipt(&view, 0, FleetIpcPhase::Published);
    duplicate.accept(published).unwrap();
    assert!(matches!(
        duplicate.accept(published),
        Err(FleetIpcCursorError::OutOfOrderPhase {
            expected: Some(FleetIpcPhase::Consumed),
            actual: FleetIpcPhase::Published,
            ..
        })
    ));
    assert_eq!(duplicate.state(), FleetIpcAttemptState::Poisoned);

    let (error, _) = poisoned_by(|receipt| receipt.binding.generation -= 1);
    assert!(matches!(
        error,
        FleetIpcCursorError::GenerationMismatch { .. }
    ));
    let (error, _) = poisoned_by(|receipt| receipt.binding.generation += 1);
    assert!(matches!(
        error,
        FleetIpcCursorError::GenerationMismatch { .. }
    ));
}

#[test]
fn wrong_rank_edge_and_attempt_poison_globally() {
    let (error, mut cursor) = poisoned_by(|receipt| receipt.worker = WorkerId(1));
    assert!(matches!(error, FleetIpcCursorError::WrongWorker { .. }));
    let view = view();
    assert_eq!(
        cursor
            .accept(receipt(&view, 0, FleetIpcPhase::Published))
            .unwrap_err(),
        FleetIpcCursorError::AttemptPoisoned
    );

    let (error, _) = poisoned_by(|receipt| receipt.binding.plan_identity[0] ^= 1);
    assert_eq!(error, FleetIpcCursorError::WrongAttempt);
    let (error, _) = poisoned_by(|receipt| receipt.binding.proof_generation += 1);
    assert_eq!(error, FleetIpcCursorError::WrongAttempt);
    let (error, _) = poisoned_by(|receipt| receipt.binding.transition = LayoutTransitionId(9));
    assert!(matches!(error, FleetIpcCursorError::WrongEdgeIdentity(0)));
    let (error, _) = poisoned_by(|receipt| receipt.binding.span_ordinal += 1);
    assert!(matches!(error, FleetIpcCursorError::WrongEdgeIdentity(0)));
    let (error, _) = poisoned_by(|receipt| receipt.binding.owner = WorkerId(1));
    assert!(matches!(error, FleetIpcCursorError::WrongEdgeIdentity(0)));
    let (error, _) = poisoned_by(|receipt| receipt.binding.peer = WorkerId(0));
    assert!(matches!(error, FleetIpcCursorError::WrongEdgeIdentity(0)));
    let (error, _) = poisoned_by(|receipt| receipt.binding.edge_ordinal = 9);
    assert_eq!(error, FleetIpcCursorError::UnknownEdge(9));
}

#[test]
fn generation_that_cannot_arm_is_rejected_before_attempt_creation() {
    assert!(matches!(
        FleetIpcCoordinatorCursor::new(&view(), u64::MAX),
        Err(FleetIpcCursorError::GenerationOverflow)
    ));
}
