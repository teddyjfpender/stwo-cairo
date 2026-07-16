use std::sync::OnceLock;

use cairo_air::air::PublicData;
use cairo_air::claims::CairoClaim;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_gpu_prover::fleet_barrier::{
    ArrivalState, BarrierReceipt, CoordinatorBarrierCursor, FleetBarrierError, WorkerBarrierCursor,
};
use stwo_cairo_gpu_prover::fleet_plan::{
    AxisMap, BarrierArrival, ConsumerGpuClass, DeclaredReplica, EffectContractId, ElementRange,
    ElementType, ExecutionInterval, FleetLink, FleetLinkId, FleetPlanError, FleetPlanInput,
    FleetProofPlan, FleetTopology, HostNumaCapacity, InPlaceAlias, LayoutAxis, LayoutTransition,
    LayoutTransitionId, OperationAssignment, OperationDesc, OperationId, OwnedValueRange,
    ReplicaId, ReplicaOrigin, ScheduleRange, ScheduleStep, StorageBinding, StorageDesc, StorageId,
    TranscriptInputValueBinding, TranscriptOutputValueBinding, ValueDesc, ValueId, ValueLayout,
    ValueOrigin, ValueUse, WorkerId, WorkerSpec,
};
use stwo_cairo_gpu_prover::fleet_pow::{
    FleetPowError, FleetPowPlan, FleetPowSchedule, FleetPowSite, PowRankReceipt,
};
use stwo_cairo_gpu_prover::fleet_spill::{
    DmaRing, HostSpillStore, RingSlot, RingSlotId, SpillChunk, SpillChunkId, SpillPlan,
    SpillPlanError, SpillTransition, SpillTransitionId, SpillTransitionKind, StoreExtent,
    StoreExtentId,
};
use stwo_cairo_gpu_prover::transcript_plan::{
    plan_cairo_blake2s_transcript, CairoBlake2sTranscriptPlan, DynamicTranscriptShape,
};

#[path = "fleet_plan_host/adversarial.rs"]
mod adversarial;
#[path = "fleet_plan_host/matrix.rs"]
mod matrix;
#[path = "fleet_plan_host/pow_bounds.rs"]
mod pow_bounds;
#[path = "fleet_plan_host/storage_alias.rs"]
mod storage_alias;
#[path = "fleet_plan_host/transcript_causality.rs"]
mod transcript_causality;

const V_INPUT: ValueId = ValueId(0);
const V_OUTPUT: ValueId = ValueId(1);
const OP_COPY: OperationId = OperationId(0);

fn transcript() -> &'static CairoBlake2sTranscriptPlan {
    static PLAN: OnceLock<CairoBlake2sTranscriptPlan> = OnceLock::new();
    PLAN.get_or_init(|| {
        let claim: CairoClaim = serde_json::from_value(serde_json::json!({
            "public_data": PublicData::default(),
            "add_opcode": { "log_size": 4 },
            "memory_id_to_big": { "big_log_sizes": [] }
        }))
        .unwrap();
        plan_cairo_blake2s_transcript(
            &claim,
            PcsConfig {
                pow_bits: 0,
                fri_config: FriConfig::new(2, 1, 13, 2),
                lifting_log_size: Some(10),
            },
            10,
            DynamicTranscriptShape {
                interaction_claim_felts: Some(5),
                oods_sampled_values_felts: Some(7),
            },
        )
        .unwrap()
    })
}

fn range(start: usize, end: usize) -> ElementRange {
    ElementRange::new(start, end).unwrap()
}

fn during(start: u32, end: u32) -> ScheduleRange {
    ScheduleRange::new(ScheduleStep(start), ScheduleStep(end)).unwrap()
}

fn canonical_layout() -> ValueLayout {
    ValueLayout {
        element: ElementType { tag: 1, bytes: 4 },
        axes: vec![
            LayoutAxis {
                tag: 0,
                extent: 4,
                stride_bytes: 4,
            },
            LayoutAxis {
                tag: 1,
                extent: 2,
                stride_bytes: 16,
            },
        ],
    }
}

fn transposed_layout() -> ValueLayout {
    ValueLayout {
        element: ElementType { tag: 1, bytes: 4 },
        axes: vec![
            LayoutAxis {
                tag: 1,
                extent: 2,
                stride_bytes: 4,
            },
            LayoutAxis {
                tag: 0,
                extent: 4,
                stride_bytes: 8,
            },
        ],
    }
}

fn barrier_steps() -> Vec<ScheduleStep> {
    (1..=transcript().segments().len())
        .map(|ordinal| ScheduleStep(u32::try_from(ordinal * 100).unwrap()))
        .collect()
}

fn terminal_step() -> ScheduleStep {
    ScheduleStep(barrier_steps().last().unwrap().0 + 100)
}

fn barrier_arrivals(workers: usize) -> Vec<BarrierArrival> {
    let mut steps = barrier_steps();
    steps.push(terminal_step());
    steps
        .into_iter()
        .enumerate()
        .flat_map(|(ordinal, release)| {
            (0..workers).map(move |worker| BarrierArrival {
                barrier_ordinal: u32::try_from(ordinal).unwrap(),
                worker: WorkerId(u16::try_from(worker).unwrap()),
                ready_step: ScheduleStep(release.0 - 1),
            })
        })
        .collect()
}

fn worker(id: u16) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id),
        capacity_bytes: 96,
        exchange_reserve_bytes: 8,
    }
}

fn values() -> Vec<ValueDesc> {
    vec![
        ValueDesc {
            id: V_INPUT,
            layout: canonical_layout(),
            alignment_bytes: 16,
            origin: ValueOrigin::ExternalInput(0),
        },
        ValueDesc {
            id: V_OUTPUT,
            layout: canonical_layout(),
            alignment_bytes: 16,
            origin: ValueOrigin::Operation,
        },
    ]
}

fn spill(worker: WorkerId, value: ValueId, base: u32) -> SpillPlan {
    let transitions = [
        SpillTransitionKind::DeviceToRing,
        SpillTransitionKind::RingToStore,
        SpillTransitionKind::StoreToRing,
        SpillTransitionKind::RingToDevice,
    ]
    .into_iter()
    .enumerate()
    .map(|(index, kind)| {
        let start = base + [0, 1, 20, 21][index];
        SpillTransition {
            id: SpillTransitionId(index as u32),
            chunk: SpillChunkId(0),
            kind,
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(start, start + 1),
            bytes: 32,
        }
    })
    .collect();
    SpillPlan {
        store: HostSpillStore {
            worker,
            capacity_bytes: 32,
            alignment_bytes: 16,
            numa_node: u32::from(worker.0),
            extents: vec![StoreExtent {
                id: StoreExtentId(0),
                offset_bytes: 0,
                len_bytes: 32,
            }],
        },
        ring: DmaRing {
            worker,
            numa_node: u32::from(worker.0),
            capacity_bytes: 32,
            memlock_limit_bytes: 32,
            alignment_bytes: 16,
            slots: vec![RingSlot {
                id: RingSlotId(0),
                offset_bytes: 0,
                len_bytes: 32,
            }],
        },
        chunks: vec![SpillChunk {
            id: SpillChunkId(0),
            value,
            elements: range(0, 8),
            worker,
            bytes: 32,
            store_extent: StoreExtentId(0),
            ring_slot: RingSlotId(0),
        }],
        transitions,
    }
}

fn one_worker_input() -> FleetPlanInput {
    let layout = canonical_layout();
    let (storages, storage_bindings) = storage_alias::one_worker_storage();
    let mut input = FleetPlanInput {
        topology: FleetTopology {
            gpu_class: ConsumerGpuClass::Rtx4090Sm89,
            module_pack_identity: [7; 32],
            fixed_image_identity: [8; 32],
            executable_identity: b"synthetic-shape-v0".to_vec(),
            coordinator: WorkerId(0),
            workers: vec![worker(0)],
            links: vec![],
            host_numa: vec![],
        },
        pow: FleetPowSchedule {
            interaction: FleetPowPlan {
                workers_per_rank: 2,
                indices_per_attempt: 8,
            },
            query: FleetPowPlan {
                workers_per_rank: 2,
                indices_per_attempt: 8,
            },
        },
        barrier_steps: barrier_steps(),
        terminal_step: terminal_step(),
        barrier_arrivals: barrier_arrivals(1),
        transcript_inputs: vec![],
        transcript_outputs: vec![],
        values: values(),
        operations: vec![OperationDesc {
            id: OP_COPY,
            semantic: b"copy".to_vec(),
            effect_identity: storage_alias::effect(1),
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(1, 2),
            reads: vec![ValueUse {
                value: V_INPUT,
                elements: range(0, 8),
                layout: layout.clone(),
            }],
            writes: vec![ValueUse {
                value: V_OUTPUT,
                elements: range(0, 8),
                layout,
            }],
        }],
        assignments: vec![OperationAssignment {
            operation: OP_COPY,
            worker: WorkerId(0),
        }],
        owners: vec![
            OwnedValueRange {
                value: V_INPUT,
                elements: range(0, 8),
                worker: WorkerId(0),
                producer: None,
                ready_at: ScheduleStep(0),
                live: during(0, 90),
            },
            OwnedValueRange {
                value: V_OUTPUT,
                elements: range(0, 8),
                worker: WorkerId(0),
                producer: Some(OP_COPY),
                ready_at: ScheduleStep(2),
                live: during(1, 90),
            },
        ],
        replicas: vec![],
        transitions: vec![],
        spills: vec![SpillPlan::empty(WorkerId(0))],
        storages,
        storage_bindings,
        in_place_aliases: vec![],
    };
    transcript_causality::bind_transcript_values(&mut input);
    input
}

fn two_worker_input() -> FleetPlanInput {
    let source = canonical_layout();
    let destination = transposed_layout();
    let (storages, storage_bindings) = storage_alias::two_worker_storage();
    let mut input = FleetPlanInput {
        topology: FleetTopology {
            gpu_class: ConsumerGpuClass::Rtx4090Sm89,
            module_pack_identity: [7; 32],
            fixed_image_identity: [8; 32],
            executable_identity: b"synthetic-shape-v0".to_vec(),
            coordinator: WorkerId(0),
            workers: vec![worker(0), worker(1)],
            links: vec![FleetLink {
                id: FleetLinkId(0),
                source: WorkerId(0),
                destination: WorkerId(1),
                max_transfer_bytes: 32,
            }],
            host_numa: vec![
                HostNumaCapacity {
                    numa_node: 0,
                    store_capacity_bytes: 32,
                    memlock_limit_bytes: 32,
                },
                HostNumaCapacity {
                    numa_node: 1,
                    store_capacity_bytes: 32,
                    memlock_limit_bytes: 32,
                },
            ],
        },
        pow: FleetPowSchedule {
            interaction: FleetPowPlan {
                workers_per_rank: 2,
                indices_per_attempt: 8,
            },
            query: FleetPowPlan {
                workers_per_rank: 2,
                indices_per_attempt: 8,
            },
        },
        barrier_steps: barrier_steps(),
        terminal_step: terminal_step(),
        barrier_arrivals: barrier_arrivals(2),
        transcript_inputs: vec![],
        transcript_outputs: vec![],
        values: values(),
        operations: vec![OperationDesc {
            id: OP_COPY,
            semantic: b"remote-copy".to_vec(),
            effect_identity: storage_alias::effect(2),
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(5, 6),
            reads: vec![ValueUse {
                value: V_INPUT,
                elements: range(0, 8),
                layout: destination.clone(),
            }],
            writes: vec![ValueUse {
                value: V_OUTPUT,
                elements: range(0, 8),
                layout: source.clone(),
            }],
        }],
        assignments: vec![OperationAssignment {
            operation: OP_COPY,
            worker: WorkerId(1),
        }],
        owners: vec![
            OwnedValueRange {
                value: V_INPUT,
                elements: range(0, 8),
                worker: WorkerId(0),
                producer: None,
                ready_at: ScheduleStep(0),
                live: during(0, 90),
            },
            OwnedValueRange {
                value: V_OUTPUT,
                elements: range(0, 8),
                worker: WorkerId(1),
                producer: Some(OP_COPY),
                ready_at: ScheduleStep(6),
                live: during(5, 90),
            },
        ],
        replicas: vec![DeclaredReplica {
            id: ReplicaId(0),
            value: V_INPUT,
            elements: range(0, 8),
            canonical_worker: WorkerId(0),
            worker: WorkerId(1),
            layout: destination.clone(),
            origin: ReplicaOrigin::Transition(LayoutTransitionId(0)),
            ready_at: ScheduleStep(2),
            live: during(1, 10),
        }],
        transitions: vec![LayoutTransition {
            id: LayoutTransitionId(0),
            value: V_INPUT,
            elements: range(0, 8),
            source_worker: WorkerId(0),
            destination_replica: ReplicaId(0),
            source_layout: source,
            destination_layout: destination,
            axes: vec![
                AxisMap {
                    source: 0,
                    destination: 0,
                },
                AxisMap {
                    source: 1,
                    destination: 1,
                },
            ],
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(1, 2),
            bytes: 32,
            scratch_bytes: 8,
            scratch_worker: WorkerId(1),
            route: FleetLinkId(0),
        }],
        spills: vec![
            spill(WorkerId(0), V_INPUT, 20),
            spill(WorkerId(1), V_OUTPUT, 30),
        ],
        storages,
        storage_bindings,
        in_place_aliases: vec![],
    };
    transcript_causality::bind_transcript_values(&mut input);
    input
}

fn compile(input: FleetPlanInput) -> Result<FleetProofPlan, FleetPlanError> {
    FleetProofPlan::compile_explicit(input, transcript())
}

#[test]
fn real_transcript_accepts_explicit_one_and_two_worker_plans() {
    let one_input = one_worker_input();
    let two_input = two_worker_input();
    let two_worker_0 = storage_alias::reserved(&two_input, WorkerId(0));
    let two_worker_1 = storage_alias::reserved(&two_input, WorkerId(1)) + 8;
    let one = compile(one_input).unwrap();
    let two = compile(two_input).unwrap();
    assert_eq!(one.workers().len(), 1);
    assert_eq!(two.workers().len(), 2);
    assert_eq!(two.barriers().len(), transcript().segments().len());
    assert_eq!(two.workers()[0].peak_live_bytes, two_worker_0);
    assert_eq!(two.workers()[1].peak_live_bytes, two_worker_1);
}

#[test]
fn ownership_producer_and_read_layout_fail_closed() {
    let mut broken = two_worker_input();
    broken.owners[0].elements.end = 7;
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::OwnershipCoverage(V_INPUT)
    );

    let mut broken = two_worker_input();
    broken.owners[1].ready_at = ScheduleStep(5);
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidProducer(V_OUTPUT)
    );

    let mut broken = two_worker_input();
    broken.owners[0].elements = ElementRange { start: 6, end: 4 };
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidRange(V_INPUT)
    );

    let mut broken = two_worker_input();
    broken.operations[0].writes.clear();
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidProducer(V_OUTPUT)
    );

    let mut broken = two_worker_input();
    broken.operations[0].reads[0].layout = canonical_layout();
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::UndeclaredRead {
            operation: OP_COPY,
            value: V_INPUT,
        }
    );
}

#[test]
fn transition_replica_and_capacity_fail_closed() {
    let mut broken = two_worker_input();
    broken.transitions[0].bytes -= 4;
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );

    let mut broken = two_worker_input();
    broken.replicas[0].ready_at = ScheduleStep(3);
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );

    let mut broken = two_worker_input();
    broken.transitions[0].axes[0].source = 99;
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );

    let mut broken = two_worker_input();
    broken.transitions[0].elements = range(0, 4);
    broken.transitions[0].bytes = 16;
    broken.replicas[0].elements = range(0, 4);
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidTransition(LayoutTransitionId(0))
    );

    let mut broken = two_worker_input();
    let mut phantom = broken.replicas[0].clone();
    phantom.id = ReplicaId(1);
    phantom.live = during(10, 20);
    broken.replicas.push(phantom);
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidReplica(ReplicaId(1))
    );

    let mut broken = two_worker_input();
    broken.topology.workers[1].capacity_bytes = 71;
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::CapacityExceeded {
            worker: WorkerId(1),
            required: 80,
            capacity: 71,
        }
    );
}

#[test]
fn spill_store_and_dma_ring_are_owned_and_bounded_per_rank() {
    let input = two_worker_input();
    for (rank, spill) in input.spills.iter().enumerate() {
        assert_eq!(spill.store.worker, WorkerId(rank as u16));
        assert_eq!(spill.ring.worker, WorkerId(rank as u16));
        spill.validate().unwrap();
    }
    compile(input).unwrap();

    let mut broken = two_worker_input();
    broken.spills[1].ring.worker = WorkerId(0);
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::Spill(SpillPlanError::WrongOwner)
    );

    let mut broken = two_worker_input();
    broken.spills[0].ring.memlock_limit_bytes = 31;
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::Spill(SpillPlanError::MemlockExceeded)
    );

    let mut broken = two_worker_input();
    broken.topology.host_numa[0].memlock_limit_bytes = 31;
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::HostCapacityExceeded(0)
    );

    let mut broken = two_worker_input();
    broken.transitions[0].during = during(30, 31);
    broken.replicas[0].live = during(30, 60);
    broken.replicas[0].ready_at = ScheduleStep(31);
    broken.operations[0].during = during(50, 51);
    broken.owners[1].live = during(50, 90);
    broken.owners[1].ready_at = ScheduleStep(51);
    broken
        .spills
        .retain(|spill| spill.store.worker == WorkerId(0));
    assert!(matches!(
        compile(broken).unwrap_err(),
        FleetPlanError::SpillValue(_)
    ));

    let mut reused = spill(WorkerId(0), V_INPUT, 10);
    reused.ring.capacity_bytes = 64;
    reused.ring.memlock_limit_bytes = 64;
    reused.ring.slots.push(RingSlot {
        id: RingSlotId(1),
        offset_bytes: 32,
        len_bytes: 32,
    });
    let mut second = reused.chunks[0].clone();
    second.id = SpillChunkId(1);
    second.ring_slot = RingSlotId(1);
    reused.chunks.push(second);
    let second_transitions = reused.transitions[..4]
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, mut transition)| {
            transition.id = SpillTransitionId(4 + index as u32);
            transition.chunk = SpillChunkId(1);
            transition.during.start.0 += 40;
            transition.during.end.0 += 40;
            transition
        })
        .collect::<Vec<_>>();
    reused.transitions.extend(second_transitions);
    reused.validate().unwrap();

    let mut overlap = reused;
    for transition in overlap
        .transitions
        .iter_mut()
        .filter(|transition| transition.chunk == SpillChunkId(1))
    {
        transition.during.start.0 -= 30;
        transition.during.end.0 -= 30;
    }
    assert_eq!(
        overlap.validate().unwrap_err(),
        SpillPlanError::StoreExtentOverlap
    );
}

#[test]
fn canonical_transcript_barriers_gate_work_and_cursor_tokens() {
    let plan = compile(two_worker_input()).unwrap();
    let mut coordinator = CoordinatorBarrierCursor::new(&plan, 41).unwrap();
    let mut worker = WorkerBarrierCursor::new(&plan, 41).unwrap();
    let receipt = |rank| BarrierReceipt {
        plan_identity: plan.identity(),
        proof_generation: 41,
        barrier_ordinal: 0,
        worker: rank,
    };
    assert_eq!(
        coordinator.arrive(receipt(WorkerId(1))).unwrap(),
        ArrivalState::Waiting { remaining: 1 }
    );
    assert_eq!(
        coordinator.arrive(receipt(WorkerId(1))).unwrap_err(),
        FleetBarrierError::DuplicateWorker(WorkerId(1))
    );
    assert_eq!(
        coordinator.arrive(receipt(WorkerId(0))).unwrap(),
        ArrivalState::Ready
    );
    assert!(!worker.can_start_segment(1));
    let release = coordinator.release(WorkerId(0)).unwrap();
    worker.accept(release).unwrap();
    assert!(worker.can_start_segment(1));
    assert!(!worker.can_start_segment(0));
    assert_eq!(
        coordinator.arrive(receipt(WorkerId(0))).unwrap_err(),
        FleetBarrierError::StaleOrFutureReceipt
    );

    let mut broken = two_worker_input();
    broken.barrier_steps[1] = broken.barrier_steps[0];
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::TranscriptMismatch
    );

    let mut broken = two_worker_input();
    broken.operations[0].interval = ExecutionInterval::BeforeBarrier(1);
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidSchedule
    );
}

#[test]
fn identity_is_order_independent_but_semantics_bound() {
    let baseline = compile(two_worker_input()).unwrap();
    let mut reordered = two_worker_input();
    reordered.topology.workers.reverse();
    reordered.values.reverse();
    reordered.owners.reverse();
    reordered.spills.reverse();
    reordered.storages.reverse();
    reordered.storage_bindings.reverse();
    for spill in &mut reordered.spills {
        spill.transitions.reverse();
    }
    let reordered = compile(reordered).unwrap();
    assert_eq!(baseline.identity(), reordered.identity());
    assert_eq!(
        baseline.canonical_bytes().unwrap(),
        reordered.canonical_bytes().unwrap()
    );

    let mut changed = two_worker_input();
    changed.operations[0].semantic.push(b'!');
    assert_ne!(baseline.identity(), compile(changed).unwrap().identity());
}
