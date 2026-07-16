use super::*;

pub(super) const fn effect(tag: u8) -> EffectContractId {
    EffectContractId([tag; 32])
}

pub(super) fn one_worker_storage() -> (Vec<StorageDesc>, Vec<StorageBinding>) {
    (
        vec![
            StorageDesc {
                id: StorageId(0),
                worker: WorkerId(0),
                bytes: 32,
                alignment_bytes: 16,
            },
            StorageDesc {
                id: StorageId(1),
                worker: WorkerId(0),
                bytes: 32,
                alignment_bytes: 16,
            },
        ],
        vec![
            StorageBinding {
                storage: StorageId(0),
                value: V_INPUT,
                elements: range(0, 8),
                worker: WorkerId(0),
                offset_bytes: 0,
                bytes: 32,
            },
            StorageBinding {
                storage: StorageId(1),
                value: V_OUTPUT,
                elements: range(0, 8),
                worker: WorkerId(0),
                offset_bytes: 0,
                bytes: 32,
            },
        ],
    )
}

pub(super) fn two_worker_storage() -> (Vec<StorageDesc>, Vec<StorageBinding>) {
    (
        vec![
            StorageDesc {
                id: StorageId(0),
                worker: WorkerId(0),
                bytes: 32,
                alignment_bytes: 16,
            },
            StorageDesc {
                id: StorageId(1),
                worker: WorkerId(1),
                bytes: 32,
                alignment_bytes: 16,
            },
            StorageDesc {
                id: StorageId(2),
                worker: WorkerId(1),
                bytes: 32,
                alignment_bytes: 16,
            },
        ],
        vec![
            StorageBinding {
                storage: StorageId(0),
                value: V_INPUT,
                elements: range(0, 8),
                worker: WorkerId(0),
                offset_bytes: 0,
                bytes: 32,
            },
            StorageBinding {
                storage: StorageId(1),
                value: V_INPUT,
                elements: range(0, 8),
                worker: WorkerId(1),
                offset_bytes: 0,
                bytes: 32,
            },
            StorageBinding {
                storage: StorageId(2),
                value: V_OUTPUT,
                elements: range(0, 8),
                worker: WorkerId(1),
                offset_bytes: 0,
                bytes: 32,
            },
        ],
    )
}

pub(super) fn matrix_storage(
    worker_count: usize,
    shard_elements: usize,
) -> (Vec<StorageDesc>, Vec<StorageBinding>) {
    let mut storages = Vec::with_capacity(worker_count * 2);
    let mut bindings = Vec::with_capacity(worker_count * 2);
    let bytes = shard_elements * 4;
    for rank in 0..worker_count {
        let worker = WorkerId(rank as u16);
        let elements = range(rank * shard_elements, (rank + 1) * shard_elements);
        for (kind, value) in [V_INPUT, V_OUTPUT].into_iter().enumerate() {
            let storage = StorageId((rank * 2 + kind) as u32);
            storages.push(StorageDesc {
                id: storage,
                worker,
                bytes,
                alignment_bytes: 4,
            });
            bindings.push(StorageBinding {
                storage,
                value,
                elements,
                worker,
                offset_bytes: 0,
                bytes,
            });
        }
    }
    (storages, bindings)
}

fn alias_input_output(input: &mut FleetPlanInput) {
    input.owners[0].live.end = input.operations[0].during.end;
    input.storages.truncate(1);
    input.storage_bindings[1].storage = StorageId(0);
    input.in_place_aliases.push(InPlaceAlias {
        operation: OP_COPY,
        effect: input.operations[0].effect_identity,
        source: V_INPUT,
        source_elements: range(0, 8),
        destination: V_OUTPUT,
        destination_elements: range(0, 8),
        storage: StorageId(0),
        offset_bytes: 0,
        bytes: 32,
    });
}

pub(super) fn reserved(input: &FleetPlanInput, worker: WorkerId) -> usize {
    input.topology.workers[worker.0 as usize].exchange_reserve_bytes
        + input
            .storages
            .iter()
            .filter(|storage| storage.worker == worker)
            .map(|storage| storage.bytes)
            .sum::<usize>()
}

#[test]
fn exact_same_operation_alias_counts_one_physical_allocation() {
    let mut input = one_worker_input();
    alias_input_output(&mut input);
    let expected = reserved(&input, WorkerId(0));
    let plan = compile(input).unwrap();
    assert_eq!(plan.workers()[0].peak_live_bytes, expected);
}

#[test]
fn same_operation_alias_requires_exact_effect_and_byte_range() {
    let mut stale_effect = one_worker_input();
    alias_input_output(&mut stale_effect);
    stale_effect.in_place_aliases[0].effect = effect(99);
    assert_eq!(
        compile(stale_effect).unwrap_err(),
        FleetPlanError::InvalidInPlaceAlias(OP_COPY)
    );

    let mut shifted = one_worker_input();
    alias_input_output(&mut shifted);
    shifted.in_place_aliases[0].offset_bytes = 4;
    assert!(matches!(
        compile(shifted).unwrap_err(),
        FleetPlanError::InvalidStorageBinding { .. }
    ));
}

#[test]
fn overlapping_versions_need_an_explicit_alias() {
    let mut input = one_worker_input();
    input.owners[0].live.end = input.operations[0].during.end;
    input.storages.truncate(1);
    input.storage_bindings[1].storage = StorageId(0);
    assert_eq!(
        compile(input).unwrap_err(),
        FleetPlanError::IllegalStorageReuse(StorageId(0))
    );
}

#[test]
fn simultaneous_disjoint_ranges_share_one_slab_without_aliasing() {
    let mut input = one_worker_input();
    input.storages.truncate(1);
    input.storage_bindings[1].storage = StorageId(0);
    input.storage_bindings[1].offset_bytes = 32;
    let expected = reserved(&input, WorkerId(0));
    let plan = compile(input).unwrap();
    assert_eq!(plan.workers()[0].peak_live_bytes, expected);
}

#[test]
fn in_place_alias_rejects_a_concurrent_old_version_consumer() {
    let mut input = one_worker_input();
    alias_input_output(&mut input);
    let next_value = ValueId(input.values.len() as u32);
    let next_operation = OperationId(input.operations.len() as u32);
    input.values.push(ValueDesc {
        id: next_value,
        layout: canonical_layout(),
        alignment_bytes: 16,
        origin: ValueOrigin::Operation,
    });
    input.operations.push(OperationDesc {
        id: next_operation,
        semantic: b"concurrent-reader".to_vec(),
        effect_identity: effect(3),
        interval: ExecutionInterval::BeforeBarrier(0),
        during: during(1, 2),
        reads: vec![ValueUse {
            value: V_INPUT,
            elements: range(0, 8),
            layout: canonical_layout(),
        }],
        writes: vec![ValueUse {
            value: next_value,
            elements: range(0, 8),
            layout: canonical_layout(),
        }],
    });
    input.assignments.push(OperationAssignment {
        operation: next_operation,
        worker: WorkerId(0),
    });
    input.owners.push(OwnedValueRange {
        value: next_value,
        elements: range(0, 8),
        worker: WorkerId(0),
        producer: Some(next_operation),
        ready_at: ScheduleStep(2),
        live: during(1, 90),
    });
    input.storages.push(StorageDesc {
        id: StorageId(1),
        worker: WorkerId(0),
        bytes: 32,
        alignment_bytes: 16,
    });
    input.storage_bindings.push(StorageBinding {
        storage: StorageId(1),
        value: next_value,
        elements: range(0, 8),
        worker: WorkerId(0),
        offset_bytes: 0,
        bytes: 32,
    });
    assert_eq!(
        compile(input).unwrap_err(),
        FleetPlanError::InvalidInPlaceAlias(OP_COPY)
    );
}

fn add_second_output(input: &mut FleetPlanInput, storage: StorageId) {
    let value = ValueId(input.values.len() as u32);
    let operation = OperationId(input.operations.len() as u32);
    input.owners[1].live.end = ScheduleStep(3);
    input.values.push(ValueDesc {
        id: value,
        layout: canonical_layout(),
        alignment_bytes: 16,
        origin: ValueOrigin::Operation,
    });
    input.operations.push(OperationDesc {
        id: operation,
        semantic: b"second-copy".to_vec(),
        effect_identity: effect(3),
        interval: ExecutionInterval::BeforeBarrier(0),
        during: during(3, 4),
        reads: vec![ValueUse {
            value: V_INPUT,
            elements: range(0, 8),
            layout: canonical_layout(),
        }],
        writes: vec![ValueUse {
            value,
            elements: range(0, 8),
            layout: canonical_layout(),
        }],
    });
    input.assignments.push(OperationAssignment {
        operation,
        worker: WorkerId(0),
    });
    input.owners.push(OwnedValueRange {
        value,
        elements: range(0, 8),
        worker: WorkerId(0),
        producer: Some(operation),
        ready_at: ScheduleStep(4),
        live: during(3, 90),
    });
    input.storage_bindings.push(StorageBinding {
        storage,
        value,
        elements: range(0, 8),
        worker: WorkerId(0),
        offset_bytes: 0,
        bytes: 32,
    });
}

#[test]
fn disjoint_value_lifetimes_can_reuse_one_storage() {
    let mut input = one_worker_input();
    add_second_output(&mut input, StorageId(1));
    let expected = reserved(&input, WorkerId(0));
    assert_eq!(
        compile(input).unwrap().workers()[0].peak_live_bytes,
        expected
    );
}

#[test]
fn sequential_distinct_storages_still_coexist_in_vram() {
    let mut input = one_worker_input();
    input.storages.push(StorageDesc {
        id: StorageId(2),
        worker: WorkerId(0),
        bytes: 32,
        alignment_bytes: 16,
    });
    add_second_output(&mut input, StorageId(2));
    let expected = reserved(&input, WorkerId(0));
    input.topology.workers[0].capacity_bytes = expected;
    assert_eq!(
        compile(input).unwrap().workers()[0].peak_live_bytes,
        expected
    );
}

#[test]
fn storage_alignment_range_coverage_and_identity_fail_closed() {
    let mut zero_effect = one_worker_input();
    zero_effect.operations[0].effect_identity = EffectContractId([0; 32]);
    assert_eq!(
        compile(zero_effect).unwrap_err(),
        FleetPlanError::InvalidEffectContract(OP_COPY)
    );

    let mut bad_alignment = one_worker_input();
    bad_alignment.storages[0].alignment_bytes = 3;
    assert_eq!(
        compile(bad_alignment).unwrap_err(),
        FleetPlanError::InvalidStorage(StorageId(0))
    );

    let mut under_aligned = one_worker_input();
    under_aligned.storages[0].alignment_bytes = 8;
    assert!(matches!(
        compile(under_aligned).unwrap_err(),
        FleetPlanError::InvalidStorageBinding { .. }
    ));

    let mut out_of_range = one_worker_input();
    out_of_range.storage_bindings[0].offset_bytes = 4;
    assert!(matches!(
        compile(out_of_range).unwrap_err(),
        FleetPlanError::InvalidStorageBinding { .. }
    ));

    let baseline = compile(one_worker_input()).unwrap();
    let mut changed = one_worker_input();
    changed.storages[0].alignment_bytes = 32;
    assert_ne!(baseline.identity(), compile(changed).unwrap().identity());

    let mut changed = one_worker_input();
    changed.operations[0].effect_identity = effect(77);
    assert_ne!(baseline.identity(), compile(changed).unwrap().identity());

    let mut changed = one_worker_input();
    changed.values[0].alignment_bytes = 32;
    changed.storages[0].alignment_bytes = 32;
    assert_ne!(baseline.identity(), compile(changed).unwrap().identity());
}
