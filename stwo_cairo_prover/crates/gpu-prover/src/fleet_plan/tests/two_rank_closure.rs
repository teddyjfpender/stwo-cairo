use super::*;
use crate::fleet_barrier::FleetBarrierError;

const GENERATION: u64 = 73;

fn compile_partitioned(fixture: Fixture) -> FleetProofPlan {
    FleetProofPlan::compile_track_a_partitioned(
        fixture.compiled,
        fixture.shape,
        fixture.placement.topology,
        fixture.placement.pow,
        transcript(),
    )
    .unwrap()
}

fn transfer_plan(split: bool) -> FleetProofPlan {
    let mut fixture = super::distributed_compiler::transfer_fixture();
    if split {
        fixture.placement.topology.links[0].max_transfer_bytes /= 4;
    }
    compile_partitioned(fixture)
}

#[test]
fn partitioned_exact_shards_real_transfer_and_terminal_fence_close() {
    let plan = compile_partitioned(super::distributed_compiler::transfer_fixture());
    let receipt = plan
        .simulate_two_rank_structural_closure(GENERATION)
        .unwrap();

    assert_eq!(receipt.plan_identity(), plan.identity());
    assert_eq!(receipt.proof_generation(), GENERATION);
    assert_eq!(receipt.exact_shards(), 2);
    assert!(receipt.transfer_spans() > 0);
    assert_eq!(receipt.fence_releases(), plan.fence_count().unwrap());
    assert_eq!(
        receipt.synthesized_transcript_phases(),
        plan.barriers().len() as u32
    );
    assert_eq!(
        receipt.worker_executions(),
        plan.placement()
            .operations
            .iter()
            .map(|operation| operation.executions.len() as u64)
            .sum::<u64>()
    );
}

#[test]
fn transfer_and_same_step_segmented_wave_close_at_declared_end() {
    let one = transfer_plan(false);
    let one_receipt = one
        .simulate_two_rank_structural_closure(GENERATION)
        .unwrap();
    assert_eq!(one_receipt.transfer_spans(), 1);
    assert_eq!(one_receipt.ipc_phase_receipts(), 4);

    let split = transfer_plan(true);
    let split_receipt = split
        .simulate_two_rank_structural_closure(GENERATION)
        .unwrap();
    assert!(split_receipt.transfer_spans() > one_receipt.transfer_spans());
    assert_eq!(
        split_receipt.ipc_phase_receipts(),
        split_receipt.transfer_spans() * 4
    );
    assert_eq!(split_receipt.fence_releases(), split.fence_count().unwrap());
}

#[test]
fn empty_next_interval_may_arrive_at_the_previous_release() {
    let mut plan = compile_partitioned(super::operation_execution::exact_fixture());
    let ordinal = (1..plan.barriers().len())
        .find(|&ordinal| {
            let segment = plan.barriers()[ordinal].segment;
            !plan.placement().operations.iter().any(|placement| {
                plan.compiled()
                    .operation(placement.operation)
                    .unwrap()
                    .stage
                    == ProofStage::BeforeTranscript(segment)
            }) && !plan.placement().transitions.iter().any(|transition| {
                transition.interval == ExecutionInterval::BeforeBarrier(ordinal as u32)
            })
        })
        .unwrap();
    let ready_step = plan.barriers()[ordinal - 1].release_step;
    for arrival in plan
        .placement
        .barrier_arrivals
        .iter_mut()
        .filter(|arrival| arrival.barrier_ordinal == ordinal as u32)
    {
        arrival.ready_step = ready_step;
    }
    plan.identity = super::super::identity::compute(&plan).unwrap();

    plan.simulate_two_rank_structural_closure(GENERATION)
        .unwrap();
}

#[test]
fn one_rank_generation_overflow_and_uninstalled_scratch_fail_closed() {
    let one_rank = compile(fixture()).unwrap();
    assert!(matches!(
        one_rank.simulate_two_rank_structural_closure(GENERATION),
        Err(FleetTwoRankStructuralClosureError::WorkerCount { actual: 1 })
    ));

    let zero_span = compile_partitioned(super::operation_execution::exact_fixture());
    let zero_receipt = zero_span
        .simulate_two_rank_structural_closure(GENERATION)
        .unwrap();
    assert_eq!(zero_receipt.exact_shards(), 2);
    assert_eq!(zero_receipt.transfer_spans(), 0);
    assert_eq!(zero_receipt.ipc_phase_receipts(), 0);
    assert!(matches!(
        zero_span.simulate_two_rank_structural_closure(u64::MAX),
        Err(FleetTwoRankStructuralClosureError::Ipc(
            FleetIpcCursorError::GenerationOverflow
        ))
    ));
    transfer_plan(false)
        .simulate_two_rank_structural_closure(u64::MAX - 1)
        .unwrap();

    let mut idle = super::runtime_view::transfer_fixture();
    idle.placement.transitions[0].scratch_bytes = 0;
    assert_eq!(
        compile(idle)
            .unwrap()
            .simulate_two_rank_structural_closure(GENERATION)
            .unwrap_err(),
        FleetTwoRankStructuralClosureError::IdleWorker {
            worker: WorkerId(1),
        }
    );

    let scratch = compile(super::runtime_view::transfer_fixture()).unwrap();
    assert!(matches!(
        scratch.simulate_two_rank_structural_closure(GENERATION),
        Err(FleetTwoRankStructuralClosureError::Install(
            FleetWorkerInstallError::UnsupportedScratch(LayoutTransitionId(0))
        ))
    ));
}

#[test]
fn stale_identity_and_current_fence_same_step_arrivals_are_rejected() {
    let mut stale = transfer_plan(false);
    stale.placement.topology.workers[0].capacity_bytes -= 1;
    assert!(matches!(
        stale.simulate_two_rank_structural_closure(GENERATION),
        Err(FleetTwoRankStructuralClosureError::PlanIdentityMismatch)
    ));

    for worker in [WorkerId(0), WorkerId(1)] {
        let mut early = transfer_plan(false);
        let release = early.barriers()[0].release_step;
        early
            .placement
            .barrier_arrivals
            .iter_mut()
            .find(|arrival| arrival.barrier_ordinal == 0 && arrival.worker == worker)
            .unwrap()
            .ready_step = release;
        early.identity = super::super::identity::compute(&early).unwrap();
        assert_eq!(
            early
                .simulate_two_rank_structural_closure(GENERATION)
                .unwrap_err(),
            FleetTwoRankStructuralClosureError::Barrier(FleetBarrierError::IncompleteBarrier)
        );
    }

    for worker in [WorkerId(0), WorkerId(1)] {
        let mut terminal = transfer_plan(false);
        let ordinal = terminal.barriers().len() as u32;
        let release = terminal.terminal_step();
        terminal
            .placement
            .barrier_arrivals
            .iter_mut()
            .find(|arrival| arrival.barrier_ordinal == ordinal && arrival.worker == worker)
            .unwrap()
            .ready_step = release;
        terminal.identity = super::super::identity::compute(&terminal).unwrap();
        assert_eq!(
            terminal
                .simulate_two_rank_structural_closure(GENERATION)
                .unwrap_err(),
            FleetTwoRankStructuralClosureError::Barrier(FleetBarrierError::IncompleteBarrier)
        );
    }
}

#[test]
fn work_must_finish_strictly_before_its_closing_fence() {
    let mut execution = compile_partitioned(super::operation_execution::exact_fixture());
    execution.placement.operations[0].during.end = execution.barriers()[0].release_step;
    execution.identity = super::super::identity::compute(&execution).unwrap();
    assert_eq!(
        execution
            .simulate_two_rank_structural_closure(GENERATION)
            .unwrap_err(),
        FleetTwoRankStructuralClosureError::ExecutionClosure
    );

    let mut transfer = transfer_plan(false);
    transfer.placement.transitions[0].during.end = transfer.barriers()[0].release_step;
    transfer.identity = super::super::identity::compute(&transfer).unwrap();
    assert!(matches!(
        transfer.simulate_two_rank_structural_closure(GENERATION),
        Err(FleetTwoRankStructuralClosureError::TransferClosure { .. })
    ));
}

#[test]
fn transcript_install_and_runtime_span_projection_fail_closed() {
    let plan = transfer_plan(false);
    let install = plan.worker_install_plan(WorkerId(0)).unwrap();
    let coordinator = install.coordinator().unwrap();

    let mut missing = coordinator.clone();
    missing.transcript.pop();
    assert_eq!(
        super::super::two_rank_closure::validation::validate_transcript_install(
            &plan, &install, &missing,
        ),
        Err(FleetTwoRankStructuralClosureError::TranscriptClosure)
    );

    let mut duplicate = coordinator.clone();
    duplicate.transcript.push(duplicate.transcript[0]);
    assert_eq!(
        super::super::two_rank_closure::validation::validate_transcript_install(
            &plan, &install, &duplicate,
        ),
        Err(FleetTwoRankStructuralClosureError::TranscriptClosure)
    );

    let view = plan.runtime_view().unwrap();
    assert!(matches!(
        super::super::two_rank_closure::validation::validate_view_completeness(
            &plan,
            &view.spans()[1..],
        ),
        Err(FleetTwoRankStructuralClosureError::TransferClosure { .. })
    ));
    let mut span = view.spans()[0];
    span.during.end = ScheduleStep(span.during.end.0 - 1);
    assert!(matches!(
        super::super::two_rank_closure::validation::validate_span_projection(&plan, span),
        Err(FleetTwoRankStructuralClosureError::TransferClosure { .. })
    ));
}

#[test]
fn sparse_near_u32_max_schedule_is_event_driven_and_non_admitting() {
    let mut plan = compile_partitioned(super::operation_execution::exact_fixture());
    let delta = u32::MAX - plan.terminal_step().0 - 1;
    let shift = |step: &mut ScheduleStep| step.0 += delta;
    plan.placement.barrier_steps.iter_mut().for_each(shift);
    plan.placement
        .barrier_arrivals
        .iter_mut()
        .for_each(|arrival| shift(&mut arrival.ready_step));
    plan.placement.operations.iter_mut().for_each(|operation| {
        shift(&mut operation.during.start);
        shift(&mut operation.during.end);
    });
    plan.placement.owners.iter_mut().for_each(|owner| {
        shift(&mut owner.live.start);
        shift(&mut owner.live.end);
    });
    plan.placement.replicas.iter_mut().for_each(|replica| {
        shift(&mut replica.live.start);
        shift(&mut replica.live.end);
    });
    plan.barriers
        .iter_mut()
        .for_each(|barrier| shift(&mut barrier.release_step));
    shift(&mut plan.placement.terminal_step);
    plan.identity = super::super::identity::compute(&plan).unwrap();

    let receipt = plan
        .simulate_two_rank_structural_closure(GENERATION)
        .unwrap();
    assert_eq!(plan.terminal_step(), ScheduleStep(u32::MAX - 1));
    assert!(receipt.event_steps() < 100);
    assert_eq!(
        plan.require_real_sn_runtime(),
        Err(FleetRuntimeAdmissionError::MissingInstalledRuntime)
    );
}
