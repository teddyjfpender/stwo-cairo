use std::collections::{BTreeMap, BTreeSet};

use super::super::*;
use super::transcript_values::{release_for_input, release_for_output};
use super::{execution_interval_contains, owner_ready_at, require_worker, validate_value_range};
use crate::compiled_proof::ValueRange;
use crate::fleet_spill::{SpillChunk, SpillPlan, VmmReclaim};
use crate::transcript_plan::CairoBlake2sTranscriptPlan;

pub(super) fn validate_spill(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    let mut spill_workers = BTreeSet::new();
    let mut host_usage = BTreeMap::<u32, (usize, usize)>::new();
    for spill in &plan.placement.spills {
        spill.validate()?;
        require_worker(workers, spill.store.worker)?;
        if !spill_workers.insert(spill.store.worker) {
            return Err(FleetPlanError::Spill(SpillPlanError::WrongOwner));
        }
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
        for chunk in &spill.chunks {
            validate_chunk(plan, transcript, spill, chunk)?;
        }
        for reclaim in &spill.vmm_reclaims {
            validate_vmm_reclaim(plan, transcript, spill, reclaim)?;
        }
        reject_overlapping_spill_cycles(spill)?;
    }
    validate_host_capacity(plan, host_usage)?;
    Ok(())
}

fn validate_chunk(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    spill: &SpillPlan,
    chunk: &SpillChunk,
) -> Result<(), FleetPlanError> {
    let value = validate_value_range(plan, chunk.value)?;
    let bytes = super::super::storage::range_bytes(value, chunk.value.elements)?;
    let extent = spill
        .store
        .extents
        .iter()
        .find(|extent| extent.id == chunk.store_extent)
        .ok_or(FleetPlanError::SpillValue(chunk.id))?;
    let storage = super::super::storage::storage(plan, chunk.storage)
        .ok_or(FleetPlanError::SpillValue(chunk.id))?;
    let bindings = plan
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| {
            super::super::storage::storage(plan, binding.storage).is_some_and(|candidate| {
                candidate.worker == chunk.worker && ranges_overlap(binding.value, chunk.value)
            })
        })
        .collect::<Vec<_>>();
    let exact_binding = plan
        .placement
        .storage_bindings
        .iter()
        .find(|binding| binding.storage == chunk.storage && binding.value == chunk.value)
        .ok_or(FleetPlanError::SpillValue(chunk.id))?;
    let owner = plan.placement.owners.iter().find(|owner| {
        owner.worker == chunk.worker
            && owner.value.version == chunk.value.version
            && owner.value.elements.contains(chunk.value.elements)
    });
    let Some(owner) = owner else {
        return Err(FleetPlanError::SpillValue(chunk.id));
    };
    let chain = spill.bounds(chunk.id)?;
    if spill
        .transitions
        .iter()
        .filter(|stage| stage.chunk == chunk.id)
        .any(|stage| !execution_interval_contains(plan, stage.interval, stage.during))
        || storage.worker != chunk.worker
        || bindings.len() != 1
        || bindings[0] != exact_binding
        || chunk.len_bytes != bytes
        || extent.len_bytes < bytes
    {
        return Err(FleetPlanError::SpillValue(chunk.id));
    }
    let [d2h, _, _, h2d] = chain;
    if !owner.live.contains(d2h.during)
        || !owner.live.contains(h2d.during)
        || owner_ready_at(plan, owner)? > d2h.during.start
    {
        return Err(FleetPlanError::SpillValue(chunk.id));
    }
    let unavailable = ScheduleRange::new(d2h.during.start, h2d.during.end)
        .ok_or(FleetPlanError::InvalidSchedule)?;
    if operation_access_during(plan, chunk.worker, chunk.value, unavailable)?
        || transition_access_during(plan, chunk.worker, chunk.value, unavailable)?
        || transcript_access_during(plan, transcript, chunk.worker, chunk.value, unavailable)?
    {
        return Err(FleetPlanError::SpillValue(chunk.id));
    }
    Ok(())
}

fn validate_vmm_reclaim(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    spill: &SpillPlan,
    reclaim: &VmmReclaim,
) -> Result<(), FleetPlanError> {
    let invalid = || FleetPlanError::InvalidVmmReclaim(reclaim.storage);
    let chunk = spill
        .chunks
        .iter()
        .find(|chunk| chunk.id == reclaim.chunk)
        .ok_or_else(invalid)?;
    let storage = super::super::storage::storage(plan, reclaim.storage).ok_or_else(invalid)?;
    let chain = spill.bounds(chunk.id)?;
    let [_, retire, prefetch, _] = chain;
    if storage.worker != chunk.worker
        || chunk.storage != reclaim.storage
        || storage.alignment_bytes < reclaim.allocation_granularity_bytes
        || storage.bytes % reclaim.allocation_granularity_bytes != 0
        || !execution_interval_contains(plan, reclaim.unmap.interval, reclaim.unmap.during)
        || !execution_interval_contains(plan, reclaim.remap.interval, reclaim.remap.during)
        || retire.during.end > reclaim.unmap.during.start
        || reclaim.unmap.during.end > reclaim.remap.during.start
        || reclaim.remap.during.end > prefetch.during.start
    {
        return Err(invalid());
    }
    let bindings = plan
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| binding.storage == reclaim.storage)
        .collect::<Vec<_>>();
    let binding = bindings.first().copied().ok_or_else(invalid)?;
    let bytes = super::super::storage::range_bytes(
        validate_value_range(plan, binding.value)?,
        binding.value.elements,
    )?;
    if bindings.len() != 1
        || binding.offset_bytes != 0
        || binding.value != chunk.value
        || bytes != storage.bytes
        || plan
            .placement
            .in_place_aliases
            .iter()
            .any(|alias| alias.storage == reclaim.storage)
    {
        return Err(invalid());
    }
    let unmapped = ScheduleRange::new(reclaim.unmap.during.start, reclaim.remap.during.end)
        .ok_or_else(invalid)?;
    if operation_access_during(plan, storage.worker, binding.value, unmapped)?
        || transition_access_during(plan, storage.worker, binding.value, unmapped)?
        || transcript_access_during(plan, transcript, storage.worker, binding.value, unmapped)?
        || other_spill_access_during(spill, chunk, unmapped)?
    {
        return Err(FleetPlanError::UnmappedStorageAccess(reclaim.storage));
    }
    Ok(())
}

fn operation_access_during(
    plan: &FleetProofPlan,
    worker: WorkerId,
    range: ValueRange,
    window: ScheduleRange,
) -> Result<bool, FleetPlanError> {
    for placement in plan
        .placement
        .operations
        .iter()
        .filter(|placement| placement.worker == worker && placement.during.overlaps(window))
    {
        let effect = plan
            .compiled
            .effect_for(placement.operation)
            .ok_or(FleetPlanError::InvalidOperation(placement.operation))?;
        if effect.accesses().iter().any(|access| {
            [access.source(), access.destination()]
                .into_iter()
                .flatten()
                .any(|bound| ranges_overlap(bound.value, range))
        }) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn transition_access_during(
    plan: &FleetProofPlan,
    worker: WorkerId,
    range: ValueRange,
    window: ScheduleRange,
) -> Result<bool, FleetPlanError> {
    for transition in plan.placement.transitions.iter().filter(|transition| {
        ranges_overlap(transition.value, range) && transition.during.overlaps(window)
    }) {
        let destination = plan
            .placement
            .replicas
            .get(transition.destination_replica.0 as usize)
            .filter(|replica| replica.id == transition.destination_replica)
            .ok_or(FleetPlanError::InvalidTransition(transition.id))?;
        if transition.source_worker == worker || destination.worker == worker {
            return Ok(true);
        }
    }
    Ok(false)
}

fn transcript_access_during(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    worker: WorkerId,
    range: ValueRange,
    window: ScheduleRange,
) -> Result<bool, FleetPlanError> {
    if worker != plan.placement.topology.coordinator {
        return Ok(false);
    }
    for binding in plan.compiled.transcript_inputs() {
        let bound = ValueRange {
            version: binding.value,
            elements: binding.elements,
        };
        if ranges_overlap(bound, range)
            && point_in(window, release_for_input(plan, transcript, binding.id)?)
        {
            return Ok(true);
        }
    }
    for binding in plan.compiled.transcript_outputs() {
        let bound = ValueRange {
            version: binding.value,
            elements: binding.elements,
        };
        if ranges_overlap(bound, range)
            && point_in(window, release_for_output(plan, transcript, binding.id)?)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn other_spill_access_during(
    spill: &SpillPlan,
    reclaimed: &SpillChunk,
    window: ScheduleRange,
) -> Result<bool, FleetPlanError> {
    for chunk in spill
        .chunks
        .iter()
        .filter(|chunk| chunk.id != reclaimed.id && ranges_overlap(chunk.value, reclaimed.value))
    {
        let [d2h, _, _, h2d] = spill.bounds(chunk.id)?;
        let cycle = ScheduleRange::new(d2h.during.start, h2d.during.end)
            .ok_or(FleetPlanError::InvalidSchedule)?;
        if cycle.overlaps(window) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn reject_overlapping_spill_cycles(spill: &SpillPlan) -> Result<(), FleetPlanError> {
    for (index, left) in spill.chunks.iter().enumerate() {
        let [left_d2h, _, _, left_h2d] = spill.bounds(left.id)?;
        let left_cycle = ScheduleRange::new(left_d2h.during.start, left_h2d.during.end)
            .ok_or(FleetPlanError::InvalidSchedule)?;
        for right in &spill.chunks[index + 1..] {
            let [right_d2h, _, _, right_h2d] = spill.bounds(right.id)?;
            let right_cycle = ScheduleRange::new(right_d2h.during.start, right_h2d.during.end)
                .ok_or(FleetPlanError::InvalidSchedule)?;
            if ranges_overlap(left.value, right.value) && left_cycle.overlaps(right_cycle) {
                return Err(FleetPlanError::SpillValue(right.id));
            }
        }
    }
    Ok(())
}

fn validate_host_capacity(
    plan: &FleetProofPlan,
    usage: BTreeMap<u32, (usize, usize)>,
) -> Result<(), FleetPlanError> {
    for (numa, (store, memlock)) in usage {
        let capacity = plan
            .placement
            .topology
            .host_numa
            .iter()
            .find(|capacity| capacity.numa_node == numa)
            .ok_or(FleetPlanError::UnknownNuma(numa))?;
        if store > capacity.store_capacity_bytes || memlock > capacity.memlock_limit_bytes {
            return Err(FleetPlanError::HostCapacityExceeded(numa));
        }
    }
    Ok(())
}

pub(super) fn measure_workers(
    plan: &FleetProofPlan,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<Vec<FleetWorkerPlan>, FleetPlanError> {
    let reclaims = plan
        .placement
        .spills
        .iter()
        .flat_map(|spill| &spill.vmm_reclaims)
        .map(|reclaim| (reclaim.storage, reclaim))
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::with_capacity(workers.len());
    for worker in workers.values() {
        let mut steps = BTreeSet::from([ScheduleStep(0)]);
        for reclaim in reclaims.values() {
            steps.extend([reclaim.unmap.during.end, reclaim.remap.during.start]);
        }
        steps.extend(
            plan.placement
                .transitions
                .iter()
                .filter(|transition| transition.scratch_worker == worker.id)
                .map(|transition| transition.during.start),
        );
        let mut peak = worker.exchange_reserve_bytes;
        for step in steps {
            let mut resident = worker.exchange_reserve_bytes;
            for storage in plan
                .placement
                .storages
                .iter()
                .filter(|storage| storage.worker == worker.id)
            {
                let mapped = reclaims.get(&storage.id).is_none_or(|reclaim| {
                    step < reclaim.unmap.during.end || step >= reclaim.remap.during.start
                });
                if mapped {
                    resident = resident
                        .checked_add(storage.bytes)
                        .ok_or(FleetPlanError::SizeOverflow)?;
                }
            }
            for transition in plan.placement.transitions.iter().filter(|transition| {
                transition.scratch_worker == worker.id
                    && transition.during.start <= step
                    && step < transition.during.end
            }) {
                resident = resident
                    .checked_add(transition.scratch_bytes)
                    .ok_or(FleetPlanError::SizeOverflow)?;
            }
            peak = peak.max(resident);
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
            peak_resident_bytes: peak,
            capacity_bytes: worker.capacity_bytes,
        });
    }
    Ok(result)
}

fn ranges_overlap(left: ValueRange, right: ValueRange) -> bool {
    left.version == right.version && left.elements.overlaps(right.elements)
}

fn point_in(window: ScheduleRange, point: ScheduleStep) -> bool {
    window.start <= point && point < window.end
}
