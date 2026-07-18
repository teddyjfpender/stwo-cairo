use stwo_backend_cuda::IPC_EXCHANGE_ALLOCATION_ALIGNMENT;

use super::*;
use crate::fleet_spill::{
    DmaRing, HostSpillStore, RingSlot, RingSlotId, SpillChunk, SpillChunkId, SpillPlan,
    SpillPlanError, SpillTransition, SpillTransitionId, SpillTransitionKind, StoreExtent,
    StoreExtentId, VmmReclaim, VmmTransition,
};

fn spill_plan(
    value: ValueRange,
    storage: StorageId,
    interval: ExecutionInterval,
    times: [u32; 8],
    reclaim: bool,
) -> SpillPlan {
    let bytes = value.elements.len() * size_of::<u32>();
    let transitions = [
        SpillTransitionKind::DeviceToRing,
        SpillTransitionKind::RingToStore,
        SpillTransitionKind::StoreToRing,
        SpillTransitionKind::RingToDevice,
    ]
    .into_iter()
    .enumerate()
    .map(|(ordinal, kind)| SpillTransition {
        id: SpillTransitionId(ordinal as u32),
        chunk: SpillChunkId(0),
        tile_ordinal: 0,
        chunk_offset_bytes: 0,
        len_bytes: bytes,
        ring_slot: RingSlotId(0),
        kind,
        interval,
        during: during(times[ordinal * 2], times[ordinal * 2 + 1]),
    })
    .collect();
    SpillPlan {
        store: HostSpillStore {
            worker: WorkerId(0),
            capacity_bytes: bytes,
            alignment_bytes: bytes,
            numa_node: 0,
            extents: vec![StoreExtent {
                id: StoreExtentId(0),
                offset_bytes: 0,
                len_bytes: bytes,
            }],
        },
        ring: DmaRing {
            worker: WorkerId(0),
            numa_node: 0,
            capacity_bytes: bytes,
            memlock_limit_bytes: bytes,
            alignment_bytes: bytes,
            slots: vec![RingSlot {
                id: RingSlotId(0),
                offset_bytes: 0,
                len_bytes: bytes,
            }],
        },
        chunks: vec![SpillChunk {
            id: SpillChunkId(0),
            value,
            worker: WorkerId(0),
            storage,
            store_extent: StoreExtentId(0),
            len_bytes: bytes,
        }],
        transitions,
        vmm_reclaims: reclaim
            .then_some(VmmReclaim {
                chunk: SpillChunkId(0),
                storage,
                allocation_granularity_bytes: bytes,
                unmap: VmmTransition {
                    interval,
                    during: during(times[3] + 1, times[3] + 2),
                },
                remap: VmmTransition {
                    interval,
                    during: during(times[4] - 2, times[4] - 1),
                },
                remap_generation: 1,
            })
            .into_iter()
            .collect(),
    }
}

fn with_spill(reclaim: bool) -> Fixture {
    let mut fixture = fixture();
    let value = value_range(fixture.spill_value, 16);
    let storage = StorageId(fixture.spill_value.0);
    fixture.placement.spills = vec![spill_plan(
        value,
        storage,
        ExecutionInterval::BeforeBarrier(0),
        [10, 11, 12, 13, 30, 31, 32, 33],
        reclaim,
    )];
    fixture.placement.topology.host_numa = vec![HostNumaCapacity {
        numa_node: 0,
        store_capacity_bytes: 64,
        memlock_limit_bytes: 64,
    }];
    fixture
}

fn with_tiled_spill() -> Fixture {
    let mut fixture = with_spill(true);
    let spill = &mut fixture.placement.spills[0];
    spill.ring.capacity_bytes = 16;
    spill.ring.memlock_limit_bytes = 16;
    spill.ring.alignment_bytes = 16;
    spill.ring.slots[0].len_bytes = 16;
    spill.transitions = (0..4_u32)
        .flat_map(|tile| {
            let spill_start = 10 + tile * 2;
            let restore_start = 30 + tile * 2;
            [
                (SpillTransitionKind::DeviceToRing, spill_start),
                (SpillTransitionKind::RingToStore, spill_start + 1),
                (SpillTransitionKind::StoreToRing, restore_start),
                (SpillTransitionKind::RingToDevice, restore_start + 1),
            ]
            .into_iter()
            .enumerate()
            .map(move |(phase, (kind, start))| SpillTransition {
                id: SpillTransitionId(tile * 4 + phase as u32),
                chunk: SpillChunkId(0),
                tile_ordinal: tile,
                chunk_offset_bytes: tile as usize * 16,
                len_bytes: 16,
                ring_slot: RingSlotId(0),
                kind,
                interval: ExecutionInterval::BeforeBarrier(0),
                during: during(start, start + 1),
            })
        })
        .collect();
    spill.vmm_reclaims[0].unmap.during = during(19, 20);
    fixture.placement.topology.host_numa[0].memlock_limit_bytes = 16;
    fixture
}

fn add_scratch_transition(fixture: &mut Fixture) {
    let value = fixture.compiled.values()[0].clone();
    let elements = range(0, value.layout.element_count().unwrap());
    let worker = WorkerId(1);
    fixture.placement.topology.workers[0].exchange_reserve_bytes =
        IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[0].capacity_bytes += IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers.push(WorkerSpec {
        id: worker,
        capacity_bytes: 1024,
        exchange_reserve_bytes: 0,
    });
    for (ordinal, release) in fixture.placement.barrier_steps.iter().enumerate() {
        fixture.placement.barrier_arrivals.push(BarrierArrival {
            barrier_ordinal: ordinal as u32,
            worker,
            ready_step: ScheduleStep(release.0 - 1),
        });
    }
    fixture.placement.barrier_arrivals.push(BarrierArrival {
        barrier_ordinal: fixture.placement.barrier_steps.len() as u32,
        worker,
        ready_step: *fixture.placement.barrier_steps.last().unwrap(),
    });
    fixture.placement.topology.links.push(FleetLink {
        id: FleetLinkId(0),
        source: WorkerId(0),
        destination: worker,
        max_transfer_bytes: value.layout.logical_bytes().unwrap(),
    });
    fixture.placement.replicas.push(FleetReplicaPlacement {
        id: ReplicaId(0),
        value: ValueRange {
            version: value.version,
            elements,
        },
        canonical_worker: WorkerId(0),
        worker,
        layout: value.layout.clone(),
        origin: ReplicaOrigin::Transition(LayoutTransitionId(0)),
        live: during(20, 50),
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
            axes: vec![AxisMap {
                source: 0,
                destination: 0,
            }],
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(20, 21),
            scratch_bytes: 128,
            scratch_worker: WorkerId(0),
            route: FleetLinkId(0),
        });
    let storage = StorageId(fixture.placement.storages.len() as u32);
    fixture.placement.storages.push(StorageDesc {
        id: storage,
        worker,
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
}

fn bind_spill_read(fixture: &mut Fixture) {
    let mut input = fixture.compiled.input().clone();
    let old = input.effects[0].clone();
    let mut accesses = old.accesses().to_vec();
    accesses.push(EffectAccess::Read {
        source: bound(accesses.len() as u32, value_range(fixture.spill_value, 16)),
    });
    let effect = EffectContract::new(accesses, vec![]).unwrap();
    input.effects = vec![effect.clone()];
    input.operations[0].invocation = invocation(&effect);
    input.operations[0].effect = effect.id();
    input.kernels = vec![AotKernelAuthority::new(
        AotKernelId(1),
        input.kernels[0].module().clone(),
        b"fleet-spill-read-semantics-v2".to_vec(),
        b"fleet-spill-read-build-v2".to_vec(),
        vec![(
            effect.id(),
            input.operations[0]
                .invocation
                .as_ref()
                .unwrap()
                .contract_id()
                .unwrap(),
        )],
    )
    .unwrap()];
    input.identity = ProofIdentity::new(
        b"fleet-spill-read-program-v2".to_vec(),
        b"fleet-spill-read-aot-v2".to_vec(),
    )
    .unwrap();
    input.host_finalizer = host_finalizer(&input.identity, input.output.codec.clone());
    fixture.compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        fixture.compiled.transcript_encoding(),
        fixture.compiled.identity().canonical_encoding(),
    )
    .unwrap();
}

#[test]
fn ordinary_spill_and_whole_storage_vmm_reclaim_are_valid() {
    let ordinary = compile(with_spill(false)).unwrap();
    let reclaimed = compile(with_spill(true)).unwrap();
    assert_eq!(
        ordinary.workers()[0].peak_resident_bytes,
        reclaimed.workers()[0].peak_resident_bytes
    );

    let mut ordinary_peak = with_spill(false);
    add_scratch_transition(&mut ordinary_peak);
    let mut reclaimed_peak = with_spill(true);
    add_scratch_transition(&mut reclaimed_peak);
    let ordinary_peak = compile(ordinary_peak).unwrap().workers()[0].peak_resident_bytes;
    let reclaimed_peak = compile(reclaimed_peak).unwrap().workers()[0].peak_resident_bytes;
    assert_eq!(ordinary_peak - reclaimed_peak, 64);
}

#[test]
fn vmm_reclaim_streams_whole_storage_through_a_bounded_pinned_tile() {
    let full_ring = compile(with_spill(true)).unwrap();
    let tiled = compile(with_tiled_spill()).unwrap();
    assert_ne!(tiled.identity(), full_ring.identity());
    assert_eq!(tiled.workers(), full_ring.workers());
    assert_eq!(tiled.placement().spills[0].ring.slots[0].len_bytes, 16);
    assert_eq!(tiled.placement().spills[0].store.extents[0].len_bytes, 64);
    assert!(tiled.placement().spills[0]
        .transitions
        .windows(2)
        .all(|pair| pair[0].during.start <= pair[1].during.start));
}

#[test]
fn tiled_spill_rejects_range_stage_and_slot_schedule_drift() {
    let expected = |error| FleetPlanError::Spill(error);

    let mut gap = with_tiled_spill();
    for transition in gap.placement.spills[0]
        .transitions
        .iter_mut()
        .filter(|transition| transition.tile_ordinal == 1)
    {
        transition.chunk_offset_bytes += 1;
    }
    assert_eq!(
        compile(gap).unwrap_err(),
        expected(SpillPlanError::IncompleteChain(SpillChunkId(0)))
    );

    let mut mismatched_stage = with_tiled_spill();
    mismatched_stage.placement.spills[0]
        .transitions
        .iter_mut()
        .find(|transition| {
            transition.tile_ordinal == 2 && transition.kind == SpillTransitionKind::RingToStore
        })
        .unwrap()
        .len_bytes = 15;
    assert_eq!(
        compile(mismatched_stage).unwrap_err(),
        expected(SpillPlanError::IncompleteChain(SpillChunkId(0)))
    );

    let mut overlapping_slot = with_tiled_spill();
    for transition in overlapping_slot.placement.spills[0]
        .transitions
        .iter_mut()
        .filter(|transition| transition.tile_ordinal == 1)
    {
        transition.during.start.0 -= 1;
        transition.during.end.0 -= 1;
    }
    assert_eq!(
        compile(overlapping_slot).unwrap_err(),
        expected(SpillPlanError::RingSlotOverlap)
    );
}

#[test]
fn spill_chunk_rejects_split_device_storage_even_when_coverage_is_complete() {
    let mut split = with_tiled_spill();
    let storage = split.placement.spills[0].chunks[0].storage;
    let binding = split
        .placement
        .storage_bindings
        .iter_mut()
        .find(|binding| binding.storage == storage)
        .unwrap();
    let mut second = binding.clone();
    binding.value.elements.end = 8;
    second.value.elements.start = 8;
    second.offset_bytes = 32;
    split.placement.storage_bindings.push(second);

    assert_eq!(
        compile(split).unwrap_err(),
        FleetPlanError::SpillValue(SpillChunkId(0))
    );
}

#[test]
fn vmm_reclaim_requires_one_exact_whole_storage_generation() {
    let storage = StorageId(fixture().spill_value.0);

    let mut partial = with_spill(true);
    partial.placement.spills[0].chunks[0].value.elements.end = 8;
    partial.placement.spills[0].chunks[0].len_bytes = 32;
    for transition in &mut partial.placement.spills[0].transitions {
        transition.len_bytes = 32;
    }
    let binding = partial
        .placement
        .storage_bindings
        .iter_mut()
        .find(|binding| binding.storage == storage)
        .unwrap();
    binding.value.elements.end = 8;
    let remainder_storage = StorageId(partial.placement.storages.len() as u32);
    partial.placement.storages.push(StorageDesc {
        id: remainder_storage,
        worker: WorkerId(0),
        bytes: 32,
        alignment_bytes: 64,
    });
    partial
        .placement
        .storage_bindings
        .push(FleetStoragePlacement {
            storage: remainder_storage,
            value: ValueRange {
                version: partial.spill_value,
                elements: range(8, 16),
            },
            offset_bytes: 0,
        });
    assert_eq!(
        compile(partial).unwrap_err(),
        FleetPlanError::InvalidVmmReclaim(storage)
    );

    let mut extra_binding = with_spill(true);
    let binding = extra_binding
        .placement
        .storage_bindings
        .iter()
        .find(|binding| binding.storage == storage)
        .unwrap()
        .clone();
    extra_binding.placement.storage_bindings.push(binding);
    assert_eq!(
        compile(extra_binding).unwrap_err(),
        FleetPlanError::SpillValue(SpillChunkId(0))
    );

    let mut aliased = with_spill(true);
    aliased
        .placement
        .in_place_aliases
        .push(InPlaceAliasPlacement {
            operation: OP_ASSEMBLE,
            alias: InPlaceAliasId(99),
            storage,
            offset_bytes: 0,
        });
    assert_eq!(
        compile(aliased).unwrap_err(),
        FleetPlanError::InvalidVmmReclaim(storage)
    );

    let mut bad_granularity = with_spill(true);
    bad_granularity.placement.spills[0].vmm_reclaims[0].allocation_granularity_bytes = 128;
    assert_eq!(
        compile(bad_granularity).unwrap_err(),
        FleetPlanError::InvalidVmmReclaim(storage)
    );

    let mut generation = with_spill(true);
    generation.placement.spills[0].vmm_reclaims[0].remap_generation = 2;
    assert_eq!(
        compile(generation).unwrap_err(),
        FleetPlanError::Spill(SpillPlanError::InvalidVmmReclaim(SpillChunkId(0)))
    );
}

#[test]
fn vmm_transition_order_and_unmapped_access_fail_closed() {
    let storage = StorageId(fixture().spill_value.0);
    let mut early_unmap = with_spill(true);
    early_unmap.placement.spills[0].vmm_reclaims[0].unmap.during = during(10, 11);
    assert_eq!(
        compile(early_unmap).unwrap_err(),
        FleetPlanError::InvalidVmmReclaim(storage)
    );

    let mut late_remap = with_spill(true);
    late_remap.placement.spills[0].vmm_reclaims[0].remap.during = during(32, 33);
    assert_eq!(
        compile(late_remap).unwrap_err(),
        FleetPlanError::InvalidVmmReclaim(storage)
    );

    let mut operation_touch = fixture();
    bind_spill_read(&mut operation_touch);
    let final_release = operation_touch.placement.barrier_steps.last().unwrap().0;
    operation_touch.placement.spills = vec![spill_plan(
        value_range(operation_touch.spill_value, 16),
        storage,
        ExecutionInterval::AfterFinalBarrier,
        [
            final_release + 1,
            final_release + 2,
            final_release + 3,
            final_release + 4,
            final_release + 40,
            final_release + 41,
            final_release + 42,
            final_release + 43,
        ],
        true,
    )];
    operation_touch.placement.topology.host_numa = vec![HostNumaCapacity {
        numa_node: 0,
        store_capacity_bytes: 64,
        memlock_limit_bytes: 64,
    }];
    assert!(matches!(
        compile(operation_touch),
        Err(FleetPlanError::SpillValue(SpillChunkId(0)))
    ));

    let mut other_spill = with_spill(true);
    let spill = &mut other_spill.placement.spills[0];
    spill.store.capacity_bytes = 128;
    spill.store.extents.push(StoreExtent {
        id: StoreExtentId(1),
        offset_bytes: 64,
        len_bytes: 64,
    });
    spill.ring.capacity_bytes = 128;
    spill.ring.memlock_limit_bytes = 128;
    spill.ring.slots.push(RingSlot {
        id: RingSlotId(1),
        offset_bytes: 64,
        len_bytes: 64,
    });
    spill.chunks.push(SpillChunk {
        id: SpillChunkId(1),
        value: value_range(other_spill.spill_value, 16),
        worker: WorkerId(0),
        storage,
        store_extent: StoreExtentId(1),
        len_bytes: 64,
    });
    spill.transitions.extend(
        [
            SpillTransitionKind::DeviceToRing,
            SpillTransitionKind::RingToStore,
            SpillTransitionKind::StoreToRing,
            SpillTransitionKind::RingToDevice,
        ]
        .into_iter()
        .zip([(20, 21), (21, 22), (23, 24), (24, 25)])
        .enumerate()
        .map(|(ordinal, (kind, (start, end)))| SpillTransition {
            id: SpillTransitionId(4 + ordinal as u32),
            chunk: SpillChunkId(1),
            tile_ordinal: 0,
            chunk_offset_bytes: 0,
            len_bytes: 64,
            ring_slot: RingSlotId(1),
            kind,
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(start, end),
        }),
    );
    other_spill.placement.topology.host_numa[0].store_capacity_bytes = 128;
    other_spill.placement.topology.host_numa[0].memlock_limit_bytes = 128;
    assert_eq!(
        compile(other_spill).unwrap_err(),
        FleetPlanError::UnmappedStorageAccess(storage)
    );

    let mut transition_touch = with_spill(true);
    add_scratch_transition(&mut transition_touch);
    let spill_value = transition_touch
        .compiled
        .value(transition_touch.spill_value)
        .unwrap()
        .clone();
    let elements = range(0, 16);
    transition_touch.placement.replicas[0].value = ValueRange {
        version: spill_value.version,
        elements,
    };
    transition_touch.placement.replicas[0].layout = spill_value.layout.clone();
    transition_touch.placement.transitions[0].value = ValueRange {
        version: spill_value.version,
        elements,
    };
    transition_touch.placement.topology.links[0].max_transfer_bytes = 64;
    let replica_storage = transition_touch.placement.storages.last_mut().unwrap();
    replica_storage.bytes = 64;
    replica_storage.alignment_bytes = 64;
    let replica_binding = transition_touch
        .placement
        .storage_bindings
        .last_mut()
        .unwrap();
    replica_binding.value = ValueRange {
        version: spill_value.version,
        elements,
    };
    assert!(matches!(
        compile(transition_touch),
        Err(FleetPlanError::SpillValue(SpillChunkId(0)))
    ));
}
