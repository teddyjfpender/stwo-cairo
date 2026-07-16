use super::*;

fn matrix_input(worker_count: usize) -> FleetPlanInput {
    let layout = ValueLayout {
        element: ElementType { tag: 7, bytes: 4 },
        axes: vec![LayoutAxis {
            tag: 0,
            extent: 16,
            stride_bytes: 4,
        }],
    };
    let shard = 16 / worker_count;
    let (storages, storage_bindings) = storage_alias::matrix_storage(worker_count, shard);
    let operations = (0..worker_count)
        .map(|rank| OperationDesc {
            id: OperationId(rank as u32),
            semantic: vec![b'm', rank as u8],
            effect_identity: storage_alias::effect(rank as u8 + 3),
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(1, 2),
            reads: vec![ValueUse {
                value: V_INPUT,
                elements: range(rank * shard, (rank + 1) * shard),
                layout: layout.clone(),
            }],
            writes: vec![ValueUse {
                value: V_OUTPUT,
                elements: range(rank * shard, (rank + 1) * shard),
                layout: layout.clone(),
            }],
        })
        .collect::<Vec<_>>();
    let assignments = (0..worker_count)
        .map(|rank| OperationAssignment {
            operation: OperationId(rank as u32),
            worker: WorkerId(rank as u16),
        })
        .collect::<Vec<_>>();
    let mut owners = Vec::with_capacity(worker_count * 2);
    for rank in 0..worker_count {
        let elements = range(rank * shard, (rank + 1) * shard);
        owners.extend([
            OwnedValueRange {
                value: V_INPUT,
                elements,
                worker: WorkerId(rank as u16),
                producer: None,
                ready_at: ScheduleStep(0),
                live: during(0, 90),
            },
            OwnedValueRange {
                value: V_OUTPUT,
                elements,
                worker: WorkerId(rank as u16),
                producer: Some(OperationId(rank as u32)),
                ready_at: ScheduleStep(2),
                live: during(1, 90),
            },
        ]);
    }
    let mut input = FleetPlanInput {
        topology: FleetTopology {
            gpu_class: ConsumerGpuClass::Rtx4090Sm89,
            module_pack_identity: [7; 32],
            fixed_image_identity: [8; 32],
            executable_identity: b"matrix-shape-v0".to_vec(),
            coordinator: WorkerId(0),
            workers: (0..worker_count)
                .map(|rank| WorkerSpec {
                    id: WorkerId(rank as u16),
                    capacity_bytes: 160,
                    exchange_reserve_bytes: 8,
                })
                .collect(),
            links: vec![],
            host_numa: vec![],
        },
        pow: FleetPowSchedule {
            interaction: FleetPowPlan {
                workers_per_rank: 2,
                indices_per_attempt: 64,
            },
            query: FleetPowPlan {
                workers_per_rank: 2,
                indices_per_attempt: 64,
            },
        },
        barrier_steps: barrier_steps(),
        terminal_step: terminal_step(),
        barrier_arrivals: barrier_arrivals(worker_count),
        transcript_inputs: vec![],
        transcript_outputs: vec![],
        values: vec![
            ValueDesc {
                id: V_INPUT,
                layout: layout.clone(),
                alignment_bytes: 4,
                origin: ValueOrigin::ExternalInput(0),
            },
            ValueDesc {
                id: V_OUTPUT,
                layout,
                alignment_bytes: 4,
                origin: ValueOrigin::Operation,
            },
        ],
        operations,
        assignments,
        owners,
        replicas: vec![],
        transitions: vec![],
        spills: vec![],
        storages,
        storage_bindings,
        in_place_aliases: vec![],
    };
    transcript_causality::bind_transcript_values(&mut input);
    input
}

#[test]
fn homogeneous_worker_matrix_covers_every_shard_and_capacity() {
    let mut identities = Vec::new();
    for worker_count in [1usize, 2, 4, 8, 16] {
        let input = matrix_input(worker_count);
        let expected = (0..worker_count)
            .map(|worker| storage_alias::reserved(&input, WorkerId(worker as u16)))
            .collect::<Vec<_>>();
        let plan = compile(input).unwrap();
        assert_eq!(plan.workers().len(), worker_count);
        assert!(plan
            .workers()
            .iter()
            .enumerate()
            .all(|(worker, plan)| { plan.peak_live_bytes == expected[worker] }));
        identities.push(plan.identity());
    }
    identities.sort_unstable();
    identities.dedup();
    assert_eq!(identities.len(), 5);
}
