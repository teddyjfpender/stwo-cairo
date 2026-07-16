use super::runtime_view::transfer_fixture;
use super::*;

const PROOF_GENERATION: u64 = 41;

fn view() -> FleetRuntimeView {
    compile(transfer_fixture()).unwrap().runtime_view().unwrap()
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
fn stale_future_and_duplicate_phases_poison_the_attempt() {
    let view = view();

    let mut future = FleetIpcCoordinatorCursor::new(&view, PROOF_GENERATION).unwrap();
    assert!(matches!(
        future.accept(receipt(&view, 0, FleetIpcPhase::Consumed)),
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
