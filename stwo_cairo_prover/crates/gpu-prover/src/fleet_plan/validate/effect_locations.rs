use std::collections::BTreeMap;

use super::{
    operation_placement, owner_ready_at, projected_range, range_covered, replica_ready_at,
};
use crate::compiled_proof::{ElementRange, ExecutionPrimitive, ValueOrigin, ValueRange};
use crate::fleet_plan::*;

pub(super) fn validate(plan: &FleetProofPlan) -> Result<(), FleetPlanError> {
    for operation in plan.compiled.operations() {
        let placement = operation_placement(plan, operation.id)?;
        let effect = plan
            .compiled
            .effect_for(operation.id)
            .ok_or(FleetPlanError::InvalidOperation(operation.id))?;
        for execution in &placement.executions {
            for access in effect.accesses() {
                if let Some(read) = access.source() {
                    let range = projected_range(&plan.compiled, operation, *read, execution)?;
                    if !read_available(plan, range, placement, execution) {
                        return Err(FleetPlanError::UndeclaredRead {
                            operation: operation.id,
                            value: range.version,
                        });
                    }
                }
                if let Some(write) = access.destination() {
                    let range = projected_range(&plan.compiled, operation, *write, execution)?;
                    if !write_available(plan, range, placement, execution)? {
                        return Err(FleetPlanError::InvalidProducer(range.version));
                    }
                }
            }
        }
    }
    Ok(())
}

fn read_available(
    plan: &FleetProofPlan,
    range: ValueRange,
    operation: &FleetOperationPlacement,
    execution: &FleetOperationExecution,
) -> bool {
    let composite_internal = plan.compiled.value(range.version).is_some_and(|value| {
        value.origin == ValueOrigin::OpOutput(operation.operation)
            && plan
                .compiled
                .operation(operation.operation)
                .is_some_and(|operation| {
                    matches!(
                        operation.primitive,
                        ExecutionPrimitive::OrderedComposite { .. }
                    )
                })
    });
    local_read_union_available(
        plan,
        range,
        execution.worker,
        |owner| {
            owner.live.contains(operation.during)
                && (composite_internal
                    || owner_ready_at(plan, owner)
                        .is_ok_and(|ready| ready <= operation.during.start))
        },
        |replica| {
            replica.live.contains(operation.during)
                && replica_ready_at(plan, replica)
                    .is_ok_and(|ready| ready <= operation.during.start)
        },
    )
}

pub(super) fn local_read_union_available(
    plan: &FleetProofPlan,
    range: ValueRange,
    worker: WorkerId,
    owner_available: impl Fn(&FleetOwnerPlacement) -> bool,
    replica_available: impl Fn(&FleetReplicaPlacement) -> bool,
) -> bool {
    let mut fragments = plan
        .placement
        .owners
        .iter()
        .filter(|owner| {
            owner.worker == worker && owner.value.version == range.version && owner_available(owner)
        })
        .map(|owner| owner.value)
        .collect::<Vec<_>>();
    fragments.extend(
        plan.placement
            .replicas
            .iter()
            .filter(|replica| {
                replica.worker == worker
                    && replica.value.version == range.version
                    && replica_available(replica)
                    && plan
                        .compiled
                        .value(range.version)
                        .is_some_and(|value| replica.layout == value.layout)
            })
            .map(|replica| replica.value),
    );
    range_covered(range.elements, &fragments)
        && one_affine_storage_covers(plan, range, worker, &fragments)
}

pub(super) fn one_affine_storage_covers(
    plan: &FleetProofPlan,
    range: ValueRange,
    worker: WorkerId,
    available: &[ValueRange],
) -> bool {
    let Some(value) = plan.compiled.value(range.version) else {
        return false;
    };
    let element_bytes = value.layout.element.bytes;
    let storage_bytes = plan
        .placement
        .storages
        .iter()
        .filter(|storage| storage.worker == worker)
        .map(|storage| (storage.id, storage.bytes))
        .collect::<BTreeMap<_, _>>();
    let mut windows_by_storage = BTreeMap::<StorageId, Vec<(ElementRange, usize)>>::new();
    for binding in plan.placement.storage_bindings.iter().filter(|binding| {
        storage_bytes.contains_key(&binding.storage)
            && binding.value.version == range.version
            && binding.value.elements.overlaps(range.elements)
            && available
                .iter()
                .any(|location| location.elements.contains(binding.value.elements))
    }) {
        let start = binding.value.elements.start.max(range.elements.start);
        let end = binding.value.elements.end.min(range.elements.end);
        let Some(offset) = start
            .checked_sub(binding.value.elements.start)
            .and_then(|skipped| skipped.checked_mul(element_bytes))
            .and_then(|skipped| binding.offset_bytes.checked_add(skipped))
        else {
            continue;
        };
        let Some(elements) = ElementRange::new(start, end) else {
            continue;
        };
        windows_by_storage
            .entry(binding.storage)
            .or_default()
            .push((elements, offset));
    }
    for (storage, mut windows) in windows_by_storage {
        let storage_bytes = storage_bytes[&storage];
        windows.sort_unstable_by_key(|(elements, offset)| (elements.start, elements.end, *offset));
        let mut cursor = range.elements.start;
        let mut base_offset = None;
        let mut valid = true;
        for (elements, offset) in windows {
            let Some(relative) = elements
                .start
                .checked_sub(range.elements.start)
                .and_then(|elements| elements.checked_mul(element_bytes))
            else {
                valid = false;
                break;
            };
            let base = if let Some(base) = base_offset {
                base
            } else {
                let Some(base) = offset.checked_sub(relative) else {
                    valid = false;
                    break;
                };
                base_offset = Some(base);
                base
            };
            let Some(expected) = base.checked_add(relative) else {
                valid = false;
                break;
            };
            let Some(end) = elements
                .len()
                .checked_mul(element_bytes)
                .and_then(|bytes| offset.checked_add(bytes))
            else {
                valid = false;
                break;
            };
            if elements.start != cursor || offset != expected || end > storage_bytes {
                valid = false;
                break;
            }
            cursor = elements.end;
        }
        if valid && cursor == range.elements.end {
            return true;
        }
    }
    false
}

fn write_available(
    plan: &FleetProofPlan,
    range: ValueRange,
    operation: &FleetOperationPlacement,
    execution: &FleetOperationExecution,
) -> Result<bool, FleetPlanError> {
    let owners = plan
        .placement
        .owners
        .iter()
        .filter(|owner| {
            owner.worker == execution.worker
                && owner.value.version == range.version
                && owner.live.contains(operation.during)
                && owner_ready_at(plan, owner) == Ok(operation.during.end)
        })
        .map(|owner| owner.value)
        .collect::<Vec<_>>();
    Ok(owners
        .iter()
        .any(|owner| owner.elements.contains(range.elements))
        && one_affine_storage_covers(plan, range, execution.worker, &owners))
}
