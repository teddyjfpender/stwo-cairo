use std::collections::{BTreeMap, BTreeSet};

mod memory;
mod transcript_values;

use memory::{measure_workers, validate_spill};
use transcript_values::validate_transcript_values;

use super::*;
use crate::transcript_plan::CairoBlake2sTranscriptPlan;

pub(super) fn validate_and_measure(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<Vec<FleetWorkerPlan>, FleetPlanError> {
    let workers = validate_topology(&plan.input.topology)?;
    plan.input.pow.validate(workers.len())?;
    validate_transcript(plan, transcript)?;
    if plan.input.values.is_empty() || plan.input.operations.is_empty() {
        return Err(FleetPlanError::EmptyProgram);
    }

    let values = index_unique(
        &plan.input.values,
        |value| value.id,
        FleetPlanError::DuplicateValue,
    )?;
    for value in &plan.input.values {
        validate_layout(value.id, &value.layout)?;
    }
    let operations = index_unique(
        &plan.input.operations,
        |operation| operation.id,
        FleetPlanError::DuplicateOperation,
    )?;
    let assignments = index_unique(
        &plan.input.assignments,
        |assignment| assignment.operation,
        FleetPlanError::DuplicateAssignment,
    )?;
    if plan
        .input
        .values
        .iter()
        .enumerate()
        .any(|(index, value)| value.id.0 as usize != index)
        || plan
            .input
            .operations
            .iter()
            .enumerate()
            .any(|(index, operation)| operation.id.0 as usize != index)
        || plan
            .input
            .replicas
            .iter()
            .enumerate()
            .any(|(index, replica)| replica.id.0 as usize != index)
        || plan
            .input
            .transitions
            .iter()
            .enumerate()
            .any(|(index, transition)| transition.id.0 as usize != index)
    {
        return Err(FleetPlanError::NonDenseIds);
    }
    for operation in &plan.input.operations {
        if !operation.during.is_valid() || operation.semantic.is_empty() {
            return Err(FleetPlanError::InvalidOperation(operation.id));
        }
        let (released_after, release_before) = match operation.interval {
            ExecutionInterval::BeforeBarrier(segment) => {
                let segment = usize::try_from(segment)
                    .map_err(|_| FleetPlanError::InvalidSegment(operation.id))?;
                let barrier = plan
                    .barriers
                    .get(segment)
                    .ok_or(FleetPlanError::InvalidSegment(operation.id))?;
                let released_after = segment.checked_sub(1).map_or(ScheduleStep(0), |previous| {
                    plan.barriers[previous].release_step
                });
                (released_after, barrier.release_step)
            }
            ExecutionInterval::AfterFinalBarrier => {
                let released_after = plan
                    .barriers
                    .last()
                    .ok_or(FleetPlanError::TranscriptMismatch)?
                    .release_step;
                (released_after, plan.input.terminal_step)
            }
        };
        if operation.during.start < released_after || operation.during.end >= release_before {
            return Err(FleetPlanError::InvalidSchedule);
        }
        let assignment = assignments
            .get(&operation.id)
            .ok_or(FleetPlanError::MissingAssignment(operation.id))?;
        require_worker(&workers, assignment.worker)?;
        for read in &operation.reads {
            let value = values
                .get(&read.value)
                .ok_or(FleetPlanError::UnknownValue(read.value))?;
            validate_element_range(read.value, read.elements, value.layout.element_count()?)?;
            validate_layout(read.value, &read.layout)?;
        }
        for write in &operation.writes {
            let value = values
                .get(&write.value)
                .ok_or(FleetPlanError::UnknownValue(write.value))?;
            validate_element_range(write.value, write.elements, value.layout.element_count()?)?;
            if write.layout != value.layout {
                return Err(FleetPlanError::InvalidProducer(write.value));
            }
        }
        for (index, left) in operation.writes.iter().enumerate() {
            if operation.writes[index + 1..]
                .iter()
                .any(|right| left.value == right.value && left.elements.overlaps(right.elements))
            {
                return Err(FleetPlanError::InvalidProducer(left.value));
            }
        }
    }
    if assignments.len() != operations.len() {
        let Some(unexpected) = assignments
            .keys()
            .find(|id| !operations.contains_key(id))
            .copied()
        else {
            return Err(FleetPlanError::NonDenseIds);
        };
        return Err(FleetPlanError::UnknownOperation(unexpected));
    }
    validate_transcript_values(plan, transcript, &values, &operations)?;
    validate_owners(plan, &values, &operations, &assignments, &workers)?;
    validate_replicas_and_transitions(plan, &values, &workers)?;
    validate_reads(plan, &values, &operations, &assignments)?;
    validate_spill(plan, &values, &operations, &assignments, &workers)?;
    super::storage::validate(plan, &values, &operations, &assignments, &workers)?;
    validate_barrier_arrivals(plan, &assignments, &workers)?;
    measure_workers(plan, &workers)
}

fn validate_barrier_arrivals(
    plan: &FleetProofPlan,
    assignments: &BTreeMap<OperationId, &OperationAssignment>,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    let expected = plan
        .barriers
        .len()
        .checked_add(1)
        .ok_or(FleetPlanError::SizeOverflow)?
        .checked_mul(workers.len())
        .ok_or(FleetPlanError::SizeOverflow)?;
    if plan.input.barrier_arrivals.len() != expected {
        return Err(FleetPlanError::TranscriptMismatch);
    }
    let arrivals = index_unique(
        &plan.input.barrier_arrivals,
        |arrival| (arrival.barrier_ordinal, arrival.worker),
        |_| FleetPlanError::TranscriptMismatch,
    )?;
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
            let work_completed = worker_interval_completion(
                plan,
                assignments,
                *worker,
                ExecutionInterval::BeforeBarrier(barrier.ordinal),
                released_after,
            );
            if arrival.ready_step < work_completed
                || arrival.ready_step < released_after
                || arrival.ready_step >= barrier.release_step
            {
                return Err(FleetPlanError::TranscriptMismatch);
            }
        }
    }
    let terminal_ordinal =
        u32::try_from(plan.barriers.len()).map_err(|_| FleetPlanError::SizeOverflow)?;
    let final_release = plan
        .barriers
        .last()
        .ok_or(FleetPlanError::TranscriptMismatch)?
        .release_step;
    for worker in workers.keys() {
        let arrival = arrivals
            .get(&(terminal_ordinal, *worker))
            .ok_or(FleetPlanError::TranscriptMismatch)?;
        let work_completed = worker_interval_completion(
            plan,
            assignments,
            *worker,
            ExecutionInterval::AfterFinalBarrier,
            final_release,
        );
        if arrival.ready_step < work_completed
            || arrival.ready_step < final_release
            || arrival.ready_step >= plan.input.terminal_step
        {
            return Err(FleetPlanError::TranscriptMismatch);
        }
    }
    Ok(())
}

fn worker_interval_completion(
    plan: &FleetProofPlan,
    assignments: &BTreeMap<OperationId, &OperationAssignment>,
    worker: WorkerId,
    interval: ExecutionInterval,
    released_after: ScheduleStep,
) -> ScheduleStep {
    let operations = plan
        .input
        .operations
        .iter()
        .filter(|operation| {
            operation.interval == interval && assignments[&operation.id].worker == worker
        })
        .map(|operation| operation.during.end);
    let transitions = plan
        .input
        .transitions
        .iter()
        .filter(|transition| {
            let destination = plan
                .input
                .replicas
                .iter()
                .find(|replica| replica.id == transition.destination_replica)
                .map(|replica| replica.worker);
            transition.interval == interval
                && (transition.source_worker == worker
                    || transition.scratch_worker == worker
                    || destination == Some(worker))
        })
        .map(|transition| transition.during.end);
    let spills = plan
        .input
        .spills
        .iter()
        .filter(|spill| spill.store.worker == worker)
        .flat_map(|spill| &spill.transitions)
        .filter(|transition| transition.interval == interval)
        .map(|transition| transition.during.end);
    operations
        .chain(transitions)
        .chain(spills)
        .max()
        .unwrap_or(released_after)
}

fn validate_topology(
    topology: &FleetTopology,
) -> Result<BTreeMap<WorkerId, &WorkerSpec>, FleetPlanError> {
    if !matches!(topology.workers.len(), 1 | 2 | 4 | 8 | 16) {
        return Err(FleetPlanError::EmptyTopology);
    }
    if topology.module_pack_identity == [0; 32]
        || topology.fixed_image_identity == [0; 32]
        || topology.executable_identity.is_empty()
    {
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
    let mut numa_nodes = BTreeSet::new();
    for capacity in &topology.host_numa {
        if !numa_nodes.insert(capacity.numa_node)
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
    if plan.schedule_key != transcript.schedule_key()
        || plan.transcript_encoding != super::identity::encode_transcript(transcript)?
        || plan.barriers.len() != transcript.segments().len()
        || plan.input.barrier_steps.len() != transcript.segments().len()
    {
        return Err(FleetPlanError::TranscriptMismatch);
    }
    let mut previous_release = None;
    for (ordinal, ((barrier, segment), release_step)) in plan
        .barriers
        .iter()
        .zip(transcript.segments())
        .zip(&plan.input.barrier_steps)
        .enumerate()
    {
        if barrier.ordinal as usize != ordinal
            || barrier.coordinator != plan.input.topology.coordinator
            || barrier.segment != segment.segment
            || barrier.operation_range != segment.operation_range
            || barrier.starts_after != segment.starts_after
            || barrier.ends_at != segment.ends_at
            || barrier.release_step != *release_step
            || previous_release.is_some_and(|previous| previous >= barrier.release_step)
        {
            return Err(FleetPlanError::TranscriptMismatch);
        }
        previous_release = Some(barrier.release_step);
    }
    if previous_release.is_none_or(|release| release >= plan.input.terminal_step) {
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
        ExecutionInterval::BeforeBarrier(segment) => {
            let segment = segment as usize;
            plan.barriers.get(segment).map(|barrier| {
                let released_after = segment.checked_sub(1).map_or(ScheduleStep(0), |previous| {
                    plan.barriers[previous].release_step
                });
                (released_after, barrier.release_step)
            })
        }
        ExecutionInterval::AfterFinalBarrier => plan
            .barriers
            .last()
            .map(|barrier| (barrier.release_step, plan.input.terminal_step)),
    };
    bounds.is_some_and(|(released_after, release_before)| {
        during.is_valid() && during.start >= released_after && during.end < release_before
    })
}

fn validate_layout(value: ValueId, layout: &ValueLayout) -> Result<(), FleetPlanError> {
    if layout.element.bytes == 0 {
        return Err(FleetPlanError::InvalidLayout(value));
    }
    let mut tags = BTreeSet::new();
    let mut expected_stride = layout.element.bytes;
    for axis in &layout.axes {
        if axis.extent == 0
            || axis.stride_bytes == 0
            || axis.stride_bytes % layout.element.bytes != 0
            || !tags.insert(axis.tag)
            || axis.stride_bytes != expected_stride
        {
            return Err(FleetPlanError::InvalidLayout(value));
        }
        expected_stride = expected_stride
            .checked_mul(axis.extent)
            .ok_or(FleetPlanError::SizeOverflow)?;
    }
    if expected_stride != layout.logical_bytes()? {
        return Err(FleetPlanError::InvalidLayout(value));
    }
    Ok(())
}

fn validate_owners(
    plan: &FleetProofPlan,
    values: &BTreeMap<ValueId, &ValueDesc>,
    operations: &BTreeMap<OperationId, &OperationDesc>,
    assignments: &BTreeMap<OperationId, &OperationAssignment>,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    for value in values.values() {
        let total = value.layout.element_count()?;
        let mut owners = plan
            .input
            .owners
            .iter()
            .filter(|owner| owner.value == value.id)
            .collect::<Vec<_>>();
        owners.sort_unstable_by_key(|owner| (owner.elements.start, owner.elements.end));
        let mut cursor = 0usize;
        for owner in owners {
            require_worker(workers, owner.worker)?;
            validate_element_range(value.id, owner.elements, total)?;
            if !owner.live.is_valid()
                || owner.live.end > plan.input.terminal_step
                || owner.elements.start != cursor
                || owner.elements.end > total
            {
                return Err(FleetPlanError::OwnershipCoverage(value.id));
            }
            cursor = owner.elements.end;
            match (value.origin, owner.producer) {
                (ValueOrigin::ExternalInput(_), None)
                    if owner.live.start == ScheduleStep(0) && owner.ready_at == ScheduleStep(0) => {
                }
                (ValueOrigin::FixedImage(_), None)
                    if owner.live.start == ScheduleStep(0)
                        && owner.live.end == plan.input.terminal_step
                        && owner.ready_at == ScheduleStep(0) => {}
                (ValueOrigin::Operation, Some(producer)) => {
                    let operation = operations
                        .get(&producer)
                        .ok_or(FleetPlanError::UnknownOperation(producer))?;
                    let assignment = assignments
                        .get(&producer)
                        .ok_or(FleetPlanError::MissingAssignment(producer))?;
                    let declares_write = operation.writes.iter().any(|write| {
                        write.value == owner.value && write.elements == owner.elements
                    });
                    if assignment.worker != owner.worker
                        || owner.ready_at != operation.during.end
                        || !owner.live.contains(operation.during)
                        || !declares_write
                    {
                        return Err(FleetPlanError::InvalidProducer(value.id));
                    }
                }
                (ValueOrigin::TranscriptOutput(_), None) => {}
                _ => return Err(FleetPlanError::InvalidProducer(value.id)),
            }
        }
        if cursor != total {
            return Err(FleetPlanError::OwnershipCoverage(value.id));
        }
    }
    for operation in operations.values() {
        let worker = assignments[&operation.id].worker;
        for write in &operation.writes {
            let matches = plan
                .input
                .owners
                .iter()
                .filter(|owner| {
                    owner.value == write.value
                        && owner.elements == write.elements
                        && owner.worker == worker
                        && owner.producer == Some(operation.id)
                })
                .count();
            if matches != 1 {
                return Err(FleetPlanError::InvalidProducer(write.value));
            }
        }
    }
    if let Some(owner) = plan
        .input
        .owners
        .iter()
        .find(|owner| !values.contains_key(&owner.value))
    {
        return Err(FleetPlanError::UnknownValue(owner.value));
    }
    Ok(())
}

fn validate_replicas_and_transitions(
    plan: &FleetProofPlan,
    values: &BTreeMap<ValueId, &ValueDesc>,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    let replicas = index_unique(
        &plan.input.replicas,
        |replica| replica.id,
        FleetPlanError::InvalidReplica,
    )?;
    let transitions = index_unique(
        &plan.input.transitions,
        |transition| transition.id,
        FleetPlanError::InvalidTransition,
    )?;
    for transition in &plan.input.transitions {
        let value = values
            .get(&transition.value)
            .ok_or(FleetPlanError::UnknownValue(transition.value))?;
        require_worker(workers, transition.source_worker)?;
        require_worker(workers, transition.scratch_worker)?;
        let replica = replicas
            .get(&transition.destination_replica)
            .ok_or(FleetPlanError::InvalidTransition(transition.id))?;
        let link = plan
            .input
            .topology
            .links
            .iter()
            .find(|link| link.id == transition.route)
            .ok_or(FleetPlanError::InvalidLink(transition.route))?;
        validate_element_range(
            transition.value,
            transition.elements,
            value.layout.element_count()?,
        )?;
        let owner = plan.input.owners.iter().find(|owner| {
            owner.value == transition.value
                && owner.worker == transition.source_worker
                && owner.elements.contains(transition.elements)
        });
        let Some(owner) = owner else {
            return Err(FleetPlanError::InvalidTransition(transition.id));
        };
        let expected_bytes = transition
            .elements
            .len()
            .checked_mul(value.layout.element.bytes)
            .ok_or(FleetPlanError::SizeOverflow)?;
        let changes_layout = transition.source_layout != transition.destination_layout;
        let covers_full_value = transition.elements.start == 0
            && transition.elements.end == value.layout.element_count()?;
        if !transition.during.is_valid()
            || !execution_interval_contains(plan, transition.interval, transition.during)
            || transition.elements.is_empty()
            || transition.bytes != expected_bytes
            || link.source != transition.source_worker
            || link.destination != replica.worker
            || transition.bytes > link.max_transfer_bytes
            || transition.scratch_worker != transition.source_worker
                && transition.scratch_worker != replica.worker
            || transition.source_layout != value.layout
            || replica.value != transition.value
            || replica.elements != transition.elements
            || replica.canonical_worker != transition.source_worker
            || replica.origin != ReplicaOrigin::Transition(transition.id)
            || replica.layout != transition.destination_layout
            || replica.live.start != transition.during.start
            || replica.ready_at != transition.during.end
            || !replica.live.contains(transition.during)
            || replica.ready_at >= replica.live.end
            || !owner.live.contains(transition.during)
            || owner.ready_at > transition.during.start
            || changes_layout && !covers_full_value
        {
            return Err(FleetPlanError::InvalidTransition(transition.id));
        }
        validate_axis_map(transition)?;
    }
    for replica in &plan.input.replicas {
        require_worker(workers, replica.worker)?;
        require_worker(workers, replica.canonical_worker)?;
        let value = values
            .get(&replica.value)
            .ok_or(FleetPlanError::UnknownValue(replica.value))?;
        validate_element_range(
            replica.value,
            replica.elements,
            value.layout.element_count()?,
        )?;
        validate_layout(replica.value, &replica.layout)?;
        let origin_is_valid = match replica.origin {
            ReplicaOrigin::Transition(transition) => transitions
                .get(&transition)
                .is_some_and(|transition| transition.destination_replica == replica.id),
            ReplicaOrigin::InstalledFixed => {
                let owner = plan.input.owners.iter().any(|owner| {
                    owner.value == replica.value
                        && owner.worker == replica.canonical_worker
                        && owner.elements.contains(replica.elements)
                });
                matches!(value.origin, ValueOrigin::FixedImage(_))
                    && owner
                    && replica.layout == value.layout
                    && replica.ready_at == ScheduleStep(0)
                    && replica.live.start == ScheduleStep(0)
                    && replica.live.end == plan.input.terminal_step
            }
        };
        if !replica.live.is_valid()
            || replica.live.end > plan.input.terminal_step
            || replica.worker == replica.canonical_worker
            || !origin_is_valid
        {
            return Err(FleetPlanError::InvalidReplica(replica.id));
        }
    }
    for pair in plan
        .input
        .replicas
        .iter()
        .enumerate()
        .flat_map(|(index, left)| {
            plan.input.replicas[index + 1..]
                .iter()
                .map(move |right| (left, right))
        })
    {
        if pair.0.value == pair.1.value
            && pair.0.worker == pair.1.worker
            && pair.0.elements.overlaps(pair.1.elements)
            && pair.0.live.overlaps(pair.1.live)
        {
            return Err(FleetPlanError::InvalidReplica(pair.1.id));
        }
    }
    Ok(())
}

fn validate_axis_map(transition: &LayoutTransition) -> Result<(), FleetPlanError> {
    validate_layout(transition.value, &transition.source_layout)?;
    validate_layout(transition.value, &transition.destination_layout)?;
    if transition.source_layout.element != transition.destination_layout.element
        || transition.axes.len() != transition.source_layout.axes.len()
    {
        return Err(FleetPlanError::InvalidTransition(transition.id));
    }
    let source = transition
        .source_layout
        .axes
        .iter()
        .map(|axis| (axis.tag, axis.extent))
        .collect::<BTreeMap<_, _>>();
    let destination = transition
        .destination_layout
        .axes
        .iter()
        .map(|axis| (axis.tag, axis.extent))
        .collect::<BTreeMap<_, _>>();
    let mut used_source = BTreeSet::new();
    let mut used_destination = BTreeSet::new();
    for axis in &transition.axes {
        let extents_match = matches!(
            (source.get(&axis.source), destination.get(&axis.destination)),
            (Some(source_extent), Some(destination_extent)) if source_extent == destination_extent
        );
        if axis.source != axis.destination
            || !extents_match
            || !used_source.insert(axis.source)
            || !used_destination.insert(axis.destination)
        {
            return Err(FleetPlanError::InvalidTransition(transition.id));
        }
    }
    if used_source.len() != source.len() || used_destination.len() != destination.len() {
        return Err(FleetPlanError::InvalidTransition(transition.id));
    }
    Ok(())
}

fn validate_reads(
    plan: &FleetProofPlan,
    values: &BTreeMap<ValueId, &ValueDesc>,
    operations: &BTreeMap<OperationId, &OperationDesc>,
    assignments: &BTreeMap<OperationId, &OperationAssignment>,
) -> Result<(), FleetPlanError> {
    for operation in operations.values() {
        let worker = assignments[&operation.id].worker;
        for read in &operation.reads {
            let canonical = plan.input.owners.iter().any(|owner| {
                owner.value == read.value
                    && owner.worker == worker
                    && owner.elements.contains(read.elements)
                    && owner.live.contains(operation.during)
                    && owner.ready_at <= operation.during.start
                    && read.layout == values[&read.value].layout
            });
            let replica = plan.input.replicas.iter().any(|replica| {
                replica.value == read.value
                    && replica.worker == worker
                    && replica.elements.contains(read.elements)
                    && replica.live.contains(operation.during)
                    && replica.ready_at <= operation.during.start
                    && replica.layout == read.layout
            });
            if !canonical && !replica {
                return Err(FleetPlanError::UndeclaredRead {
                    operation: operation.id,
                    value: read.value,
                });
            }
        }
    }
    Ok(())
}

fn validate_element_range(
    value: ValueId,
    range: ElementRange,
    total: usize,
) -> Result<(), FleetPlanError> {
    if range.is_empty() || range.end > total {
        return Err(FleetPlanError::InvalidRange(value));
    }
    Ok(())
}

fn require_worker<T>(
    workers: &BTreeMap<WorkerId, T>,
    worker: WorkerId,
) -> Result<(), FleetPlanError> {
    workers
        .contains_key(&worker)
        .then_some(())
        .ok_or(FleetPlanError::UnknownWorker(worker))
}

fn index_unique<'a, T, K: Ord + Copy>(
    values: &'a [T],
    key: impl Fn(&T) -> K,
    duplicate: impl Fn(K) -> FleetPlanError,
) -> Result<BTreeMap<K, &'a T>, FleetPlanError> {
    let mut indexed = BTreeMap::new();
    for value in values {
        let key = key(value);
        if indexed.insert(key, value).is_some() {
            return Err(duplicate(key));
        }
    }
    Ok(indexed)
}
