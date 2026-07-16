use super::*;

#[test]
fn topology_rejects_card_caps_and_empty_identities() {
    for (class, limit) in [
        (ConsumerGpuClass::Rtx3090Sm86, 21usize << 30),
        (ConsumerGpuClass::Rtx4090Sm89, 21usize << 30),
        (ConsumerGpuClass::Rtx5090Sm120, 29usize << 30),
    ] {
        let mut broken = one_worker_input();
        broken.topology.gpu_class = class;
        broken.topology.workers[0].capacity_bytes = limit + 1;
        assert_eq!(
            compile(broken).unwrap_err(),
            FleetPlanError::NonDenseWorkers
        );
    }

    let mut broken = one_worker_input();
    broken.topology.module_pack_identity = [0; 32];
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidHomogeneousTopology
    );

    let mut broken = one_worker_input();
    broken.topology.executable_identity.clear();
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidHomogeneousTopology
    );
}

#[test]
fn transition_rejects_reversed_and_undersized_links() {
    let mut reversed = two_worker_input();
    reversed.topology.links[0].source = WorkerId(1);
    reversed.topology.links[0].destination = WorkerId(0);
    assert_eq!(
        compile(reversed).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );

    let mut undersized = two_worker_input();
    undersized.topology.links[0].max_transfer_bytes = 31;
    assert_eq!(
        compile(undersized).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );
}

#[test]
fn static_barriers_reject_missing_and_duplicate_arrivals() {
    let mut missing = two_worker_input();
    missing.barrier_arrivals.pop();
    assert_eq!(
        compile(missing).unwrap_err(),
        FleetPlanError::TranscriptMismatch
    );

    let mut duplicate = two_worker_input();
    duplicate.barrier_arrivals[1] = duplicate.barrier_arrivals[0];
    assert_eq!(
        compile(duplicate).unwrap_err(),
        FleetPlanError::TranscriptMismatch
    );
}

#[test]
fn runtime_barriers_reject_early_wrong_and_stale_releases() {
    let plan = compile(two_worker_input()).unwrap();
    let mut coordinator = CoordinatorBarrierCursor::new(&plan, 17).unwrap();
    assert_eq!(
        coordinator.release(WorkerId(0)).unwrap_err(),
        FleetBarrierError::IncompleteBarrier
    );

    for worker in [WorkerId(0), WorkerId(1)] {
        assert!(matches!(
            coordinator
                .arrive(BarrierReceipt {
                    plan_identity: plan.identity(),
                    proof_generation: 17,
                    barrier_ordinal: 0,
                    worker,
                })
                .unwrap(),
            ArrivalState::Waiting { .. } | ArrivalState::Ready
        ));
    }
    let release = coordinator.release(WorkerId(0)).unwrap();
    let mut worker = WorkerBarrierCursor::new(&plan, 17).unwrap();

    let mut wrong = release;
    wrong.coordinator = WorkerId(1);
    assert_eq!(
        worker.accept(wrong).unwrap_err(),
        FleetBarrierError::WrongCoordinator(WorkerId(1))
    );

    let mut stale = release;
    stale.proof_generation = 16;
    assert_eq!(
        worker.accept(stale).unwrap_err(),
        FleetBarrierError::StaleOrFutureRelease
    );
    worker.accept(release).unwrap();
    assert_eq!(
        worker.accept(release).unwrap_err(),
        FleetBarrierError::StaleOrFutureRelease
    );
}

#[test]
fn pow_oracle_requires_true_local_minimum_and_exhaustive_none() {
    let pow = FleetPowPlan {
        workers_per_rank: 2,
        indices_per_attempt: 8,
    };
    let identity = [3; 32];
    let nonce = stwo_backend_cuda::pow_index_to_nonce;
    let receipt = |rank, candidate_nonce| PowRankReceipt {
        site: FleetPowSite::Interaction,
        plan_identity: identity,
        rank,
        proof_generation: 11,
        attempt_ordinal: 0,
        candidate_nonce,
    };
    let valid = |candidate| [nonce(0), nonce(2), nonce(4)].contains(&candidate);
    let exact = [
        receipt(WorkerId(0), Some(nonce(0))),
        receipt(WorkerId(1), Some(nonce(2))),
    ];
    assert_eq!(
        pow.verify_winner(FleetPowSite::Interaction, 2, identity, 11, &exact, valid,)
            .unwrap(),
        nonce(0)
    );

    let not_local_minimum = [
        receipt(WorkerId(0), Some(nonce(4))),
        receipt(WorkerId(1), Some(nonce(2))),
    ];
    assert_eq!(
        pow.verify_winner(
            FleetPowSite::Interaction,
            2,
            identity,
            11,
            &not_local_minimum,
            valid,
        )
        .unwrap_err(),
        FleetPowError::InvalidReceipt(WorkerId(0))
    );

    let none = [receipt(WorkerId(0), None), receipt(WorkerId(1), None)];
    assert_eq!(
        pow.verify_winner(FleetPowSite::Interaction, 2, identity, 11, &none, |_| false,)
            .unwrap_err(),
        FleetPowError::NoWinner
    );
    assert_eq!(
        pow.attempt_bounds(1u64 << 49).unwrap_err(),
        FleetPowError::SearchExhausted
    );
}

#[test]
fn transition_scratch_requires_a_real_worker_and_owned_capacity() {
    let mut unknown = two_worker_input();
    unknown.transitions[0].scratch_worker = WorkerId(2);
    assert_eq!(
        compile(unknown).unwrap_err(),
        FleetPlanError::UnknownWorker(WorkerId(2))
    );

    let mut oversized = two_worker_input();
    oversized.transitions[0].scratch_bytes = 65;
    assert_eq!(
        compile(oversized).unwrap_err(),
        FleetPlanError::CapacityExceeded {
            worker: WorkerId(1),
            required: 105,
            capacity: 96,
        }
    );
}

#[test]
fn same_numa_spills_are_charged_in_aggregate() {
    let mut broken = two_worker_input();
    broken.spills[1].store.numa_node = 0;
    broken.spills[1].ring.numa_node = 0;
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::HostCapacityExceeded(0)
    );
}

#[test]
fn empty_spill_plan_cannot_reserve_resources() {
    let mut broken = SpillPlan::empty(WorkerId(0));
    broken.store.capacity_bytes = 1;
    assert_eq!(
        broken.validate().unwrap_err(),
        SpillPlanError::OrphanedResource
    );
}

#[test]
fn overlapping_full_spill_cycles_reject_a_d2h_from_unavailable_data() {
    let mut broken = two_worker_input();
    let spill = &mut broken.spills[0];
    spill.store.capacity_bytes = 64;
    spill.store.extents.push(StoreExtent {
        id: StoreExtentId(1),
        offset_bytes: 32,
        len_bytes: 32,
    });
    spill.ring.capacity_bytes = 64;
    spill.ring.memlock_limit_bytes = 64;
    spill.ring.slots.push(RingSlot {
        id: RingSlotId(1),
        offset_bytes: 32,
        len_bytes: 32,
    });
    let mut chunk = spill.chunks[0].clone();
    chunk.id = SpillChunkId(1);
    chunk.store_extent = StoreExtentId(1);
    chunk.ring_slot = RingSlotId(1);
    spill.chunks.push(chunk);

    for (offset, (kind, start)) in [
        (SpillTransitionKind::DeviceToRing, 40),
        (SpillTransitionKind::RingToStore, 41),
        (SpillTransitionKind::StoreToRing, 50),
        (SpillTransitionKind::RingToDevice, 51),
    ]
    .into_iter()
    .enumerate()
    {
        spill.transitions.push(SpillTransition {
            id: SpillTransitionId(4 + offset as u32),
            chunk: SpillChunkId(1),
            kind,
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(start, start + 1),
            bytes: 32,
        });
    }

    // Chunk 1 D2H [40,41) reads while chunk 0 is unavailable [21,42).
    // The former gap-only windows [21,41) and [41,51) merely touch.
    spill.validate().unwrap();
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::SpillValue(stwo_cairo_gpu_prover::fleet_plan::SpillChunkIdForError(1))
    );
}

#[test]
fn post_transcript_tail_is_bounded_by_a_terminal_wait_all_fence() {
    let mut input = one_worker_input();
    let final_release = *input.barrier_steps.last().unwrap();
    input.operations[0].interval = ExecutionInterval::AfterFinalBarrier;
    input.operations[0].during = during(final_release.0, final_release.0 + 1);
    input.owners[0].live = during(0, input.terminal_step.0);
    input.owners[1].ready_at = ScheduleStep(final_release.0 + 1);
    input.owners[1].live = during(final_release.0, input.terminal_step.0);

    let plan = compile(input.clone()).unwrap();
    let mut coordinator = CoordinatorBarrierCursor::new(&plan, 29).unwrap();
    let mut worker = WorkerBarrierCursor::new(&plan, 29).unwrap();
    let fence_count = plan.fence_count().unwrap();
    for ordinal in 0..fence_count {
        assert_eq!(
            coordinator
                .arrive(BarrierReceipt {
                    plan_identity: plan.identity(),
                    proof_generation: 29,
                    barrier_ordinal: ordinal,
                    worker: WorkerId(0),
                })
                .unwrap(),
            ArrivalState::Ready
        );
        let release = coordinator.release(WorkerId(0)).unwrap();
        worker.accept(release).unwrap();
        if ordinal + 1 < fence_count {
            assert!(worker.can_start_segment(ordinal + 1));
        }
    }
    assert!(coordinator.is_complete());
    assert!(!worker.can_start_segment(fence_count));

    let mut early = input.clone();
    early.operations[0].during = during(final_release.0 - 1, final_release.0 + 1);
    assert_eq!(compile(early).unwrap_err(), FleetPlanError::InvalidSchedule);

    let mut missing_terminal_arrival = input;
    missing_terminal_arrival.barrier_arrivals.pop();
    assert_eq!(
        compile(missing_terminal_arrival).unwrap_err(),
        FleetPlanError::TranscriptMismatch
    );
}

#[test]
fn fixed_image_replicas_need_no_fake_per_proof_transfer() {
    let mut input = two_worker_input();
    input.values[0].origin = ValueOrigin::FixedImage(7);
    input.owners[0].live = during(0, input.terminal_step.0);
    input.replicas[0].origin = ReplicaOrigin::InstalledFixed;
    input.replicas[0].layout = canonical_layout();
    input.replicas[0].ready_at = ScheduleStep(0);
    input.replicas[0].live = during(0, input.terminal_step.0);
    input.operations[0].reads[0].layout = canonical_layout();
    input.transitions.clear();
    input.topology.links.clear();
    compile(input.clone()).unwrap();

    let mut external = input.clone();
    external.values[0].origin = ValueOrigin::ExternalInput(0);
    assert_eq!(
        compile(external).unwrap_err(),
        FleetPlanError::InvalidReplica(ReplicaId(0))
    );

    let mut unbound_image = input.clone();
    unbound_image.topology.fixed_image_identity = [0; 32];
    assert_eq!(
        compile(unbound_image).unwrap_err(),
        FleetPlanError::InvalidHomogeneousTopology
    );

    input.owners[0].live.end.0 -= 1;
    assert_eq!(
        compile(input).unwrap_err(),
        FleetPlanError::InvalidProducer(V_INPUT)
    );
}

#[test]
fn physical_work_is_interval_bound_and_included_in_wait_all_arrivals() {
    let mut crossing = two_worker_input();
    crossing.transitions[0].during = during(99, 100);
    assert_eq!(
        compile(crossing).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );

    let mut wrong_interval = two_worker_input();
    wrong_interval.spills[0].transitions[0].interval = ExecutionInterval::BeforeBarrier(1);
    assert_eq!(
        compile(wrong_interval).unwrap_err(),
        FleetPlanError::SpillValue(stwo_cairo_gpu_prover::fleet_plan::SpillChunkIdForError(0))
    );

    let mut early_arrival = two_worker_input();
    early_arrival
        .barrier_arrivals
        .iter_mut()
        .find(|arrival| arrival.barrier_ordinal == 0 && arrival.worker == WorkerId(1))
        .unwrap()
        .ready_step = ScheduleStep(40);
    assert_eq!(
        compile(early_arrival).unwrap_err(),
        FleetPlanError::TranscriptMismatch
    );
}
