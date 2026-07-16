use std::collections::{BTreeMap, BTreeSet};

use super::super::*;
use super::{execution_interval_contains, require_worker, validate_element_range};
use crate::fleet_spill::SpillChunkId;

pub(super) fn validate_spill(
    plan: &FleetProofPlan,
    values: &BTreeMap<ValueId, &ValueDesc>,
    operations: &BTreeMap<OperationId, &OperationDesc>,
    assignments: &BTreeMap<OperationId, &OperationAssignment>,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    let mut spill_workers = BTreeSet::new();
    let mut host_usage = BTreeMap::<u32, (usize, usize)>::new();
    for spill in &plan.input.spills {
        spill.validate()?;
        require_worker(workers, spill.store.worker)?;
        if !spill.chunks.is_empty() {
            let usage = host_usage.entry(spill.store.numa_node).or_default();
            usage.0 = usage
                .0
                .checked_add(spill.store.capacity_bytes)
                .ok_or(FleetPlanError::SizeOverflow)?;
            usage.1 = usage
                .1
                .checked_add(spill.ring.capacity_bytes)
                .ok_or(FleetPlanError::SizeOverflow)?;
        }
        if !spill_workers.insert(spill.store.worker) {
            return Err(FleetPlanError::Spill(SpillPlanError::WrongOwner));
        }
        for chunk in &spill.chunks {
            let value = values
                .get(&chunk.value)
                .ok_or(FleetPlanError::UnknownValue(chunk.value))?;
            validate_element_range(chunk.value, chunk.elements, value.layout.element_count()?)?;
            let expected = chunk
                .elements
                .len()
                .checked_mul(value.layout.element.bytes)
                .ok_or(FleetPlanError::SizeOverflow)?;
            let owner = plan.input.owners.iter().find(|owner| {
                owner.value == chunk.value
                    && owner.worker == chunk.worker
                    && owner.elements.contains(chunk.elements)
            });
            let Some(owner) = owner else {
                return Err(spill_value_error(chunk.id));
            };
            let chain = spill.chain(chunk.id)?;
            if chain
                .iter()
                .any(|stage| !execution_interval_contains(plan, stage.interval, stage.during))
            {
                return Err(spill_value_error(chunk.id));
            }
            let [d2h, _, _, h2d] = chain;
            let unavailable = ScheduleRange::new(d2h.during.end, h2d.during.end)
                .ok_or(FleetPlanError::InvalidSchedule)?;
            if expected != chunk.bytes
                || !owner.live.contains(d2h.during)
                || !owner.live.contains(h2d.during)
                || owner.ready_at > d2h.during.start
            {
                return Err(spill_value_error(chunk.id));
            }
            for operation in operations.values().filter(|operation| {
                assignments[&operation.id].worker == chunk.worker
                    && operation.reads.iter().any(|read| {
                        read.value == chunk.value && read.elements.overlaps(chunk.elements)
                    })
            }) {
                if operation.during.end > d2h.during.start
                    && operation.during.start < h2d.during.end
                {
                    return Err(spill_value_error(chunk.id));
                }
            }
            if plan.input.transitions.iter().any(|transition| {
                transition.value == chunk.value
                    && transition.source_worker == chunk.worker
                    && transition.elements.overlaps(chunk.elements)
                    && transition.during.overlaps(unavailable)
            }) {
                return Err(spill_value_error(chunk.id));
            }
        }
        for (index, left) in spill.chunks.iter().enumerate() {
            let [left_d2h, _, _, left_h2d] = spill.chain(left.id)?;
            let left_cycle = ScheduleRange::new(left_d2h.during.start, left_h2d.during.end)
                .ok_or(FleetPlanError::InvalidSchedule)?;
            for right in &spill.chunks[index + 1..] {
                let [right_d2h, _, _, right_h2d] = spill.chain(right.id)?;
                let right_cycle = ScheduleRange::new(right_d2h.during.start, right_h2d.during.end)
                    .ok_or(FleetPlanError::InvalidSchedule)?;
                if left.value == right.value
                    && left.elements.overlaps(right.elements)
                    && left_cycle.overlaps(right_cycle)
                {
                    return Err(spill_value_error(right.id));
                }
            }
        }
    }
    for (numa_node, (store, memlock)) in host_usage {
        let capacity = plan
            .input
            .topology
            .host_numa
            .iter()
            .find(|capacity| capacity.numa_node == numa_node)
            .ok_or(FleetPlanError::UnknownNuma(numa_node))?;
        if store > capacity.store_capacity_bytes || memlock > capacity.memlock_limit_bytes {
            return Err(FleetPlanError::HostCapacityExceeded(numa_node));
        }
    }
    Ok(())
}

pub(super) fn measure_workers(
    plan: &FleetProofPlan,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<Vec<FleetWorkerPlan>, FleetPlanError> {
    let mut result = Vec::with_capacity(workers.len());
    for worker in workers.values() {
        let reserved_storage = super::super::storage::reserved_bytes(plan, worker.id)?;
        let mut steps = BTreeSet::from([ScheduleStep(0)]);
        steps.extend(
            plan.input
                .owners
                .iter()
                .filter(|x| x.worker == worker.id)
                .map(|x| x.live.start),
        );
        for spill in plan
            .input
            .spills
            .iter()
            .filter(|spill| spill.store.worker == worker.id)
        {
            for chunk in &spill.chunks {
                let [d2h, _, _, h2d] = spill.chain(chunk.id)?;
                steps.extend([d2h.during.end, h2d.during.start]);
            }
        }
        steps.extend(
            plan.input
                .replicas
                .iter()
                .filter(|x| x.worker == worker.id)
                .map(|x| x.live.start),
        );
        steps.extend(
            plan.input
                .transitions
                .iter()
                .filter(|x| x.scratch_worker == worker.id)
                .map(|x| x.during.start),
        );
        let mut peak = worker.exchange_reserve_bytes;
        for step in steps {
            let point = ScheduleRange {
                start: step,
                end: ScheduleStep(step.0.checked_add(1).ok_or(FleetPlanError::SizeOverflow)?),
            };
            let mut live = worker
                .exchange_reserve_bytes
                .checked_add(reserved_storage)
                .ok_or(FleetPlanError::SizeOverflow)?;
            for transition in plan
                .input
                .transitions
                .iter()
                .filter(|x| x.during.overlaps(point))
            {
                if transition.scratch_worker == worker.id {
                    live = live
                        .checked_add(transition.scratch_bytes)
                        .ok_or(FleetPlanError::SizeOverflow)?;
                }
            }
            peak = peak.max(live);
        }
        if peak > worker.capacity_bytes {
            return Err(FleetPlanError::CapacityExceeded {
                worker: worker.id,
                required: peak,
                capacity: worker.capacity_bytes,
            });
        }
        result.push(FleetWorkerPlan {
            worker: worker.id,
            peak_live_bytes: peak,
            capacity_bytes: worker.capacity_bytes,
        });
    }
    Ok(result)
}

fn spill_value_error(id: SpillChunkId) -> FleetPlanError {
    FleetPlanError::SpillValue(SpillChunkIdForError(id.0))
}
