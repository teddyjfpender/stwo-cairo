use super::*;
use crate::fleet_barrier::{
    ArrivalState, BarrierReceipt, CoordinatorBarrierCursor, FleetBarrierError, WorkerBarrierCursor,
};

#[test]
fn coordinator_binds_each_arrival_to_its_exact_ready_step() {
    let plan = compile(fixture()).unwrap();
    let arrival = plan.placement().barrier_arrivals[0];
    let mut cursor = CoordinatorBarrierCursor::new(&plan, 9).unwrap();
    let receipt = BarrierReceipt {
        plan_identity: plan.identity(),
        proof_generation: 9,
        barrier_ordinal: arrival.barrier_ordinal,
        worker: arrival.worker,
        ready_step: ScheduleStep(arrival.ready_step.0 + 1),
    };

    assert_eq!(
        cursor.arrive(receipt),
        Err(FleetBarrierError::UnexpectedReadyStep {
            worker: arrival.worker,
            expected: arrival.ready_step,
            actual: receipt.ready_step,
        })
    );
    assert_eq!(
        cursor.arrive(BarrierReceipt {
            ready_step: arrival.ready_step,
            ..receipt
        }),
        Ok(ArrivalState::Ready)
    );
}

#[test]
fn coordinator_indexes_ready_steps_by_barrier_and_worker() {
    let mut fixture = runtime_view::transfer_fixture();
    let worker_one = fixture
        .placement
        .barrier_arrivals
        .iter_mut()
        .find(|arrival| arrival.barrier_ordinal == 0 && arrival.worker == WorkerId(1))
        .unwrap();
    worker_one.ready_step = ScheduleStep(worker_one.ready_step.0 - 1);
    let plan = compile(fixture).unwrap();
    let arrivals = &plan.placement().barrier_arrivals[..2];
    let mut cursor = CoordinatorBarrierCursor::new(&plan, 11).unwrap();

    assert_eq!(
        cursor.arrive(BarrierReceipt {
            plan_identity: plan.identity(),
            proof_generation: 11,
            barrier_ordinal: 0,
            worker: WorkerId(2),
            ready_step: arrivals[0].ready_step,
        }),
        Err(FleetBarrierError::UnknownWorker(WorkerId(2)))
    );
    assert_eq!(
        cursor.arrive(BarrierReceipt {
            plan_identity: plan.identity(),
            proof_generation: 11,
            barrier_ordinal: 0,
            worker: WorkerId(1),
            ready_step: arrivals[0].ready_step,
        }),
        Err(FleetBarrierError::UnexpectedReadyStep {
            worker: WorkerId(1),
            expected: arrivals[1].ready_step,
            actual: arrivals[0].ready_step,
        })
    );
    assert_eq!(
        cursor.arrive(BarrierReceipt {
            plan_identity: plan.identity(),
            proof_generation: 11,
            barrier_ordinal: 0,
            worker: WorkerId(1),
            ready_step: arrivals[1].ready_step,
        }),
        Ok(ArrivalState::Waiting { remaining: 1 })
    );
}

#[test]
fn terminal_fence_releases_only_after_every_planned_arrival() {
    let plan = compile(fixture()).unwrap();
    let generation = 13;
    let mut coordinator = CoordinatorBarrierCursor::new(&plan, generation).unwrap();
    let mut worker = WorkerBarrierCursor::new(&plan, generation).unwrap();
    let fence_count = plan.fence_count().unwrap();

    for ordinal in 0..fence_count {
        let arrival = plan
            .placement()
            .barrier_arrivals
            .iter()
            .find(|arrival| arrival.barrier_ordinal == ordinal)
            .unwrap();
        assert!(worker.can_start_segment(ordinal));
        assert_eq!(
            coordinator.arrive(BarrierReceipt {
                plan_identity: plan.identity(),
                proof_generation: generation,
                barrier_ordinal: ordinal,
                worker: arrival.worker,
                ready_step: arrival.ready_step,
            }),
            Ok(ArrivalState::Ready)
        );
        let release = coordinator.release(WorkerId(0)).unwrap();
        assert_eq!(release.barrier_ordinal, ordinal);
        worker.accept(release).unwrap();
    }

    assert_eq!(fence_count, plan.barriers().len() as u32 + 1);
    assert!(coordinator.is_complete());
    assert!(!worker.can_start_segment(fence_count));
    assert_eq!(
        coordinator.release(WorkerId(0)),
        Err(FleetBarrierError::Complete)
    );
}
