use std::collections::{BTreeMap, BTreeSet};

mod memory;
mod transcript_values;

use memory::{measure_workers, validate_spill};
use transcript_values::validate_transcript_values;

use super::*;
use crate::compiled_proof::{
    ExecutionPrimitive, OpNode, ProofStage, ValueDesc, ValueOrigin, ValueRange,
};
use crate::transcript_plan::CairoBlake2sTranscriptPlan;

pub(super) fn validate_and_measure(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<Vec<FleetWorkerPlan>, FleetPlanError> {
    let workers = validate_topology(&plan.placement.topology)?;
    plan.placement.pow.validate(workers.len())?;
    validate_transcript(plan, transcript)?;
    if plan.compiled.operations().is_empty() || plan.compiled.values().is_empty() {
        return Err(FleetPlanError::EmptyProgram);
    }
    validate_operations(plan, &workers)?;
    validate_owners(plan, &workers)?;
    validate_replicas_and_transitions(plan, &workers)?;
    validate_effect_locations(plan)?;
    validate_transcript_values(plan, transcript)?;
    validate_spill(plan, transcript, &workers)?;
    super::storage::validate(plan, &workers)?;
    validate_barrier_arrivals(plan, &workers)?;
    measure_workers(plan, &workers)
}

fn validate_operations(
    plan: &FleetProofPlan,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    if plan.placement.operations.len() != plan.compiled.operations().len() {
        let missing = plan
            .compiled
            .operations()
            .iter()
            .find(|operation| operation_placement(plan, operation.id).is_err())
            .map_or(OpId(u32::MAX), |operation| operation.id);
        return Err(FleetPlanError::MissingOperation(missing));
    }
    for (index, placement) in plan.placement.operations.iter().enumerate() {
        let expected = OpId(u32::try_from(index).map_err(|_| FleetPlanError::SizeOverflow)?);
        if placement.operation != expected {
            return Err(FleetPlanError::DuplicateOperation(placement.operation));
        }
        let operation = operation(plan, placement.operation)?;
        require_worker(workers, placement.worker)?;
        let interval = operation_interval(plan, operation)?;
        if !execution_interval_contains(plan, interval, placement.during) {
            return Err(FleetPlanError::InvalidSchedule);
        }
    }
    Ok(())
}

fn validate_owners(
    plan: &FleetProofPlan,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    for value in plan.compiled.values() {
        let total = value
            .layout
            .element_count()
            .map_err(|_| FleetPlanError::SizeOverflow)?;
        let mut owners = plan
            .placement
            .owners
            .iter()
            .filter(|owner| owner.value.version == value.version)
            .collect::<Vec<_>>();
        owners.sort_unstable_by_key(|owner| {
            (
                owner.value.elements.start,
                owner.value.elements.end,
                owner.worker,
            )
        });
        let mut cursor = 0usize;
        for owner in owners {
            require_worker(workers, owner.worker)?;
            validate_value_range(plan, owner.value)?;
            if owner.value.elements.start != cursor
                || !owner.live.is_valid()
                || owner.live.end > plan.placement.terminal_step
                || owner_ready_at(plan, owner)? >= owner.live.end
            {
                return Err(FleetPlanError::OwnershipCoverage(value.version));
            }
            validate_owner_origin(plan, owner, value)?;
            cursor = owner.value.elements.end;
        }
        if cursor != total {
            return Err(FleetPlanError::OwnershipCoverage(value.version));
        }
    }
    if let Some(owner) = plan
        .placement
        .owners
        .iter()
        .find(|owner| plan.compiled.value(owner.value.version).is_none())
    {
        return Err(FleetPlanError::UnknownValue(owner.value.version));
    }
    Ok(())
}

fn validate_owner_origin(
    plan: &FleetProofPlan,
    owner: &FleetOwnerPlacement,
    value: &ValueDesc,
) -> Result<(), FleetPlanError> {
    let valid = match value.origin {
        ValueOrigin::ExternalInput(_) => owner.live.start == ScheduleStep(0),
        ValueOrigin::Constant(_) => {
            owner.live.start == ScheduleStep(0) && owner.live.end == plan.placement.terminal_step
        }
        ValueOrigin::TranscriptOutput(_) => owner.worker == plan.placement.topology.coordinator,
        ValueOrigin::OpOutput(producer) => {
            let placement = operation_placement(plan, producer)?;
            let effect = plan
                .compiled
                .effect_for(producer)
                .ok_or(FleetPlanError::InvalidOperation(producer))?;
            let destinations = effect
                .accesses()
                .iter()
                .filter_map(|access| access.destination().map(|range| range.value))
                .filter(|range| range.version == value.version)
                .collect::<Vec<_>>();
            placement.worker == owner.worker
                && owner.live.start == placement.during.start
                && owner.live.contains(placement.during)
                && range_covered(owner.value.elements, &destinations)
        }
    };
    if valid {
        Ok(())
    } else {
        Err(FleetPlanError::InvalidProducer(value.version))
    }
}

fn validate_replicas_and_transitions(
    plan: &FleetProofPlan,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    if plan
        .placement
        .replicas
        .iter()
        .enumerate()
        .any(|(index, replica)| replica.id.0 as usize != index)
        || plan
            .placement
            .transitions
            .iter()
            .enumerate()
            .any(|(index, transition)| transition.id.0 as usize != index)
    {
        return Err(FleetPlanError::NonDenseIds);
    }
    for transition in &plan.placement.transitions {
        validate_transition(plan, transition, workers)?;
    }
    for replica in &plan.placement.replicas {
        validate_replica(plan, replica, workers)?;
    }
    for (index, left) in plan.placement.replicas.iter().enumerate() {
        if let Some(right) = plan.placement.replicas[index + 1..].iter().find(|right| {
            left.worker == right.worker
                && ranges_overlap(left.value, right.value)
                && left.live.overlaps(right.live)
        }) {
            return Err(FleetPlanError::InvalidReplica(right.id));
        }
    }
    Ok(())
}

fn validate_transition(
    plan: &FleetProofPlan,
    transition: &FleetTransitionPlacement,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    let value = validate_value_range(plan, transition.value)?;
    require_worker(workers, transition.source_worker)?;
    require_worker(workers, transition.scratch_worker)?;
    let replica = replica(plan, transition.destination_replica)?;
    let link = plan
        .placement
        .topology
        .links
        .iter()
        .find(|link| link.id == transition.route)
        .ok_or(FleetPlanError::InvalidLink(transition.route))?;
    let owner = plan.placement.owners.iter().find(|owner| {
        owner.worker == transition.source_worker
            && owner.value.version == transition.value.version
            && owner.value.elements.contains(transition.value.elements)
    });
    let Some(owner) = owner else {
        return Err(FleetPlanError::InvalidTransition(transition.id));
    };
    let bytes = super::storage::range_bytes(value, transition.value.elements)?;
    if !execution_interval_contains(plan, transition.interval, transition.during)
        || link.source != transition.source_worker
        || link.destination != replica.worker
        || bytes > link.max_transfer_bytes
        || transition.scratch_worker != transition.source_worker
            && transition.scratch_worker != replica.worker
        || replica.value != transition.value
        || replica.canonical_worker != transition.source_worker
        || replica.origin != ReplicaOrigin::Transition(transition.id)
        || replica.live.start != transition.during.start
        || !replica.live.contains(transition.during)
        || transition.during.end >= replica.live.end
        || owner_ready_at(plan, owner)? > transition.during.start
        || !owner.live.contains(transition.during)
    {
        return Err(FleetPlanError::InvalidTransition(transition.id));
    }
    validate_axis_map(value, &replica.layout, &transition.axes)
        .map_err(|_| FleetPlanError::InvalidTransition(transition.id))
}

fn validate_replica(
    plan: &FleetProofPlan,
    replica: &FleetReplicaPlacement,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    let value = validate_value_range(plan, replica.value)?;
    require_worker(workers, replica.worker)?;
    require_worker(workers, replica.canonical_worker)?;
    validate_layout(replica.value.version, &replica.layout)?;
    let origin_valid = match replica.origin {
        ReplicaOrigin::InstalledFixed => {
            let canonical_owner = plan.placement.owners.iter().any(|owner| {
                owner.worker == replica.canonical_worker
                    && owner.value.version == replica.value.version
                    && owner.value.elements.contains(replica.value.elements)
            });
            matches!(value.origin, ValueOrigin::Constant(_))
                && canonical_owner
                && replica.layout == value.layout
                && replica.live.start == ScheduleStep(0)
                && replica.live.end == plan.placement.terminal_step
        }
        ReplicaOrigin::Transition(id) => plan
            .placement
            .transitions
            .get(id.0 as usize)
            .is_some_and(|transition| transition.destination_replica == replica.id),
    };
    if replica.worker == replica.canonical_worker
        || !replica.live.is_valid()
        || replica.live.end > plan.placement.terminal_step
        || !origin_valid
    {
        return Err(FleetPlanError::InvalidReplica(replica.id));
    }
    Ok(())
}

fn validate_effect_locations(plan: &FleetProofPlan) -> Result<(), FleetPlanError> {
    for operation in plan.compiled.operations() {
        let placement = operation_placement(plan, operation.id)?;
        let effect = plan
            .compiled
            .effect_for(operation.id)
            .ok_or(FleetPlanError::InvalidOperation(operation.id))?;
        for access in effect.accesses() {
            if let Some(read) = access.source() {
                if !read_available(plan, read.value, placement) {
                    return Err(FleetPlanError::UndeclaredRead {
                        operation: operation.id,
                        value: read.value.version,
                    });
                }
            }
            if let Some(write) = access.destination() {
                if !write_available(plan, write.value, placement)? {
                    return Err(FleetPlanError::InvalidProducer(write.value.version));
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
    let canonical = plan.placement.owners.iter().any(|owner| {
        owner.worker == operation.worker
            && owner.value.version == range.version
            && owner.value.elements.contains(range.elements)
            && owner.live.contains(operation.during)
            && (composite_internal
                || owner_ready_at(plan, owner).is_ok_and(|ready| ready <= operation.during.start))
    });
    let replica = plan.placement.replicas.iter().any(|replica| {
        replica.worker == operation.worker
            && replica.value.version == range.version
            && replica.value.elements.contains(range.elements)
            && replica.live.contains(operation.during)
            && replica_ready_at(plan, replica).is_ok_and(|ready| ready <= operation.during.start)
            && plan
                .compiled
                .value(range.version)
                .is_some_and(|value| replica.layout == value.layout)
    });
    canonical || replica
}

fn write_available(
    plan: &FleetProofPlan,
    range: ValueRange,
    operation: &FleetOperationPlacement,
) -> Result<bool, FleetPlanError> {
    Ok(plan.placement.owners.iter().any(|owner| {
        owner.worker == operation.worker
            && owner.value.version == range.version
            && owner.value.elements.contains(range.elements)
            && owner.live.contains(operation.during)
            && owner_ready_at(plan, owner) == Ok(operation.during.end)
    }))
}

fn validate_barrier_arrivals(
    plan: &FleetProofPlan,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    let expected = plan
        .barriers
        .len()
        .checked_add(1)
        .and_then(|count| count.checked_mul(workers.len()))
        .ok_or(FleetPlanError::SizeOverflow)?;
    if plan.placement.barrier_arrivals.len() != expected {
        return Err(FleetPlanError::TranscriptMismatch);
    }
    let arrivals = index_unique(&plan.placement.barrier_arrivals, |arrival| {
        (arrival.barrier_ordinal, arrival.worker)
    })
    .ok_or(FleetPlanError::TranscriptMismatch)?;
    for barrier in &plan.barriers {
        let released_after = barrier
            .ordinal
            .checked_sub(1)
            .map_or(ScheduleStep(0), |previous| {
                plan.barriers[previous as usize].release_step
            });
        for worker in workers.keys() {
            let arrival = arrivals
                .get(&(barrier.ordinal, *worker))
                .ok_or(FleetPlanError::TranscriptMismatch)?;
            let completed = worker_interval_completion(
                plan,
                *worker,
                ExecutionInterval::BeforeBarrier(barrier.ordinal),
                released_after,
            );
            if arrival.ready_step < completed
                || arrival.ready_step < released_after
                || arrival.ready_step >= barrier.release_step
            {
                return Err(FleetPlanError::TranscriptMismatch);
            }
        }
    }
    let terminal = u32::try_from(plan.barriers.len()).map_err(|_| FleetPlanError::SizeOverflow)?;
    let final_release = plan
        .barriers
        .last()
        .ok_or(FleetPlanError::TranscriptMismatch)?
        .release_step;
    for worker in workers.keys() {
        let arrival = arrivals
            .get(&(terminal, *worker))
            .ok_or(FleetPlanError::TranscriptMismatch)?;
        let completed = worker_interval_completion(
            plan,
            *worker,
            ExecutionInterval::AfterFinalBarrier,
            final_release,
        );
        if arrival.ready_step < completed
            || arrival.ready_step < final_release
            || arrival.ready_step >= plan.placement.terminal_step
        {
            return Err(FleetPlanError::TranscriptMismatch);
        }
    }
    Ok(())
}

fn worker_interval_completion(
    plan: &FleetProofPlan,
    worker: WorkerId,
    interval: ExecutionInterval,
    released_after: ScheduleStep,
) -> ScheduleStep {
    let operations = plan.placement.operations.iter().filter_map(|placement| {
        let operation = operation(plan, placement.operation).ok()?;
        (placement.worker == worker && operation_interval(plan, operation).ok()? == interval)
            .then_some(placement.during.end)
    });
    let transitions = plan.placement.transitions.iter().filter_map(|transition| {
        let destination = replica(plan, transition.destination_replica).ok()?.worker;
        (transition.interval == interval
            && (transition.source_worker == worker
                || transition.scratch_worker == worker
                || destination == worker))
            .then_some(transition.during.end)
    });
    let spill = plan
        .placement
        .spills
        .iter()
        .filter(|spill| spill.store.worker == worker)
        .flat_map(|spill| {
            spill
                .transitions
                .iter()
                .filter(move |transition| transition.interval == interval)
                .map(|transition| transition.during.end)
                .chain(spill.vmm_reclaims.iter().flat_map(move |reclaim| {
                    [reclaim.unmap, reclaim.remap]
                        .into_iter()
                        .filter(move |transition| transition.interval == interval)
                        .map(|transition| transition.during.end)
                }))
        });
    operations
        .chain(transitions)
        .chain(spill)
        .max()
        .unwrap_or(released_after)
}

fn validate_topology(
    topology: &FleetPlacementTopology,
) -> Result<BTreeMap<WorkerId, &WorkerSpec>, FleetPlanError> {
    if !matches!(topology.workers.len(), 1 | 2 | 4 | 8 | 16) {
        return Err(FleetPlanError::EmptyTopology);
    }
    if topology.module_pack_identity == [0; 32] || topology.fixed_image_identity == [0; 32] {
        return Err(FleetPlanError::InvalidHomogeneousTopology);
    }
    let capacity_limit = match topology.gpu_class {
        ConsumerGpuClass::Rtx3090Sm86 | ConsumerGpuClass::Rtx4090Sm89 => 21usize << 30,
        ConsumerGpuClass::Rtx5090Sm120 => 29usize << 30,
    };
    let mut workers = BTreeMap::new();
    for (rank, worker) in topology.workers.iter().enumerate() {
        if worker.id.0 as usize != rank
            || worker.capacity_bytes == 0
            || worker.capacity_bytes > capacity_limit
            || worker.exchange_reserve_bytes > worker.capacity_bytes
        {
            return Err(FleetPlanError::NonDenseWorkers);
        }
        workers.insert(worker.id, worker);
    }
    if topology.coordinator != WorkerId(0) || !workers.contains_key(&topology.coordinator) {
        return Err(FleetPlanError::InvalidCoordinator);
    }
    let mut pairs = BTreeSet::new();
    for (ordinal, link) in topology.links.iter().enumerate() {
        if link.id.0 as usize != ordinal
            || link.source == link.destination
            || link.max_transfer_bytes == 0
            || !workers.contains_key(&link.source)
            || !workers.contains_key(&link.destination)
            || !pairs.insert((link.source, link.destination))
        {
            return Err(FleetPlanError::InvalidLink(link.id));
        }
    }
    let mut numa = BTreeSet::new();
    for capacity in &topology.host_numa {
        if !numa.insert(capacity.numa_node)
            || capacity.store_capacity_bytes == 0 && capacity.memlock_limit_bytes == 0
        {
            return Err(FleetPlanError::InvalidHomogeneousTopology);
        }
    }
    Ok(workers)
}

fn validate_transcript(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), FleetPlanError> {
    if plan.compiled.transcript_encoding()
        != transcript
            .canonical_encoding()
            .map_err(|_| FleetPlanError::TranscriptMismatch)?
        || plan.barriers.len() != transcript.segments().len()
        || plan.placement.barrier_steps.len() != transcript.segments().len()
    {
        return Err(FleetPlanError::TranscriptMismatch);
    }
    let mut previous = None;
    for (ordinal, ((barrier, segment), release)) in plan
        .barriers
        .iter()
        .zip(transcript.segments())
        .zip(&plan.placement.barrier_steps)
        .enumerate()
    {
        if barrier.ordinal as usize != ordinal
            || barrier.coordinator != plan.placement.topology.coordinator
            || barrier.segment != segment.segment
            || barrier.operation_range != segment.operation_range
            || barrier.starts_after != segment.starts_after
            || barrier.ends_at != segment.ends_at
            || barrier.release_step != *release
            || previous.is_some_and(|step| step >= barrier.release_step)
        {
            return Err(FleetPlanError::TranscriptMismatch);
        }
        previous = Some(barrier.release_step);
    }
    if previous.is_none_or(|release| release >= plan.placement.terminal_step) {
        return Err(FleetPlanError::TranscriptMismatch);
    }
    Ok(())
}

pub(super) fn execution_interval_contains(
    plan: &FleetProofPlan,
    interval: ExecutionInterval,
    during: ScheduleRange,
) -> bool {
    let bounds = match interval {
        ExecutionInterval::BeforeBarrier(ordinal) => {
            plan.barriers.get(ordinal as usize).map(|barrier| {
                let start = ordinal.checked_sub(1).map_or(ScheduleStep(0), |previous| {
                    plan.barriers[previous as usize].release_step
                });
                (start, barrier.release_step)
            })
        }
        ExecutionInterval::AfterFinalBarrier => plan
            .barriers
            .last()
            .map(|barrier| (barrier.release_step, plan.placement.terminal_step)),
    };
    bounds
        .is_some_and(|(start, end)| during.is_valid() && during.start >= start && during.end < end)
}

pub(super) fn operation_interval(
    plan: &FleetProofPlan,
    operation: &OpNode,
) -> Result<ExecutionInterval, FleetPlanError> {
    match operation.stage {
        ProofStage::BeforeTranscript(segment) => plan
            .barriers
            .iter()
            .position(|barrier| barrier.segment == segment)
            .and_then(|ordinal| u32::try_from(ordinal).ok())
            .map(ExecutionInterval::BeforeBarrier)
            .ok_or(FleetPlanError::InvalidSegment(operation.id)),
        ProofStage::AfterTranscript => Ok(ExecutionInterval::AfterFinalBarrier),
    }
}

pub(super) fn owner_ready_at(
    plan: &FleetProofPlan,
    owner: &FleetOwnerPlacement,
) -> Result<ScheduleStep, FleetPlanError> {
    let value = plan
        .compiled
        .value(owner.value.version)
        .ok_or(FleetPlanError::UnknownValue(owner.value.version))?;
    match value.origin {
        ValueOrigin::ExternalInput(_) | ValueOrigin::Constant(_) => Ok(ScheduleStep(0)),
        ValueOrigin::TranscriptOutput(_) => Ok(owner.live.start),
        ValueOrigin::OpOutput(producer) => Ok(operation_placement(plan, producer)?.during.end),
    }
}

pub(super) fn replica_ready_at(
    plan: &FleetProofPlan,
    replica: &FleetReplicaPlacement,
) -> Result<ScheduleStep, FleetPlanError> {
    match replica.origin {
        ReplicaOrigin::InstalledFixed => Ok(ScheduleStep(0)),
        ReplicaOrigin::Transition(id) => Ok(plan
            .placement
            .transitions
            .get(id.0 as usize)
            .filter(|transition| transition.id == id)
            .ok_or(FleetPlanError::InvalidReplica(replica.id))?
            .during
            .end),
    }
}

pub(super) fn validate_value_range(
    plan: &FleetProofPlan,
    range: ValueRange,
) -> Result<&ValueDesc, FleetPlanError> {
    let value = plan
        .compiled
        .value(range.version)
        .ok_or(FleetPlanError::UnknownValue(range.version))?;
    let total = value
        .layout
        .element_count()
        .map_err(|_| FleetPlanError::SizeOverflow)?;
    if range.elements.is_empty() || range.elements.end > total {
        return Err(FleetPlanError::InvalidRange(range.version));
    }
    Ok(value)
}

pub(super) fn require_worker<T>(
    workers: &BTreeMap<WorkerId, T>,
    worker: WorkerId,
) -> Result<(), FleetPlanError> {
    workers
        .contains_key(&worker)
        .then_some(())
        .ok_or(FleetPlanError::UnknownWorker(worker))
}

fn operation(plan: &FleetProofPlan, id: OpId) -> Result<&OpNode, FleetPlanError> {
    plan.compiled
        .operations()
        .get(id.0 as usize)
        .filter(|operation| operation.id == id)
        .ok_or(FleetPlanError::UnknownOperation(id))
}

pub(super) fn operation_placement(
    plan: &FleetProofPlan,
    id: OpId,
) -> Result<&FleetOperationPlacement, FleetPlanError> {
    plan.placement
        .operations
        .get(id.0 as usize)
        .filter(|placement| placement.operation == id)
        .ok_or(FleetPlanError::MissingOperation(id))
}

fn replica(plan: &FleetProofPlan, id: ReplicaId) -> Result<&FleetReplicaPlacement, FleetPlanError> {
    plan.placement
        .replicas
        .get(id.0 as usize)
        .filter(|replica| replica.id == id)
        .ok_or(FleetPlanError::InvalidReplica(id))
}

fn validate_layout(version: ValueVersion, layout: &ValueLayout) -> Result<(), FleetPlanError> {
    if layout.element.bytes == 0 {
        return Err(FleetPlanError::InvalidRange(version));
    }
    let mut tags = BTreeSet::new();
    let mut stride = layout.element.bytes;
    for axis in &layout.axes {
        if axis.extent == 0 || axis.stride_bytes != stride || !tags.insert(axis.tag) {
            return Err(FleetPlanError::InvalidRange(version));
        }
        stride = stride
            .checked_mul(axis.extent)
            .ok_or(FleetPlanError::SizeOverflow)?;
    }
    Ok(())
}

fn validate_axis_map(
    value: &ValueDesc,
    destination: &ValueLayout,
    axes: &[AxisMap],
) -> Result<(), FleetPlanError> {
    if destination != &value.layout || axes.len() != value.layout.axes.len() {
        return Err(FleetPlanError::InvalidRange(value.version));
    }
    let expected = value
        .layout
        .axes
        .iter()
        .map(|axis| axis.tag)
        .collect::<BTreeSet<_>>();
    let actual = axes
        .iter()
        .filter(|axis| axis.source == axis.destination)
        .map(|axis| axis.source)
        .collect::<BTreeSet<_>>();
    if actual.len() != axes.len() || actual != expected {
        return Err(FleetPlanError::InvalidRange(value.version));
    }
    Ok(())
}

fn range_covered(target: ElementRange, ranges: &[ValueRange]) -> bool {
    let mut ranges = ranges
        .iter()
        .map(|range| range.elements)
        .filter(|range| target.overlaps(*range))
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| (range.start, range.end));
    let mut cursor = target.start;
    for range in ranges {
        let start = range.start.max(target.start);
        let end = range.end.min(target.end);
        if start != cursor {
            return false;
        }
        cursor = end;
    }
    cursor == target.end
}

fn ranges_overlap(left: ValueRange, right: ValueRange) -> bool {
    left.version == right.version && left.elements.overlaps(right.elements)
}

fn index_unique<T, K: Ord + Copy>(values: &[T], key: impl Fn(&T) -> K) -> Option<BTreeMap<K, &T>> {
    let mut indexed = BTreeMap::new();
    for value in values {
        if indexed.insert(key(value), value).is_some() {
            return None;
        }
    }
    Some(indexed)
}
