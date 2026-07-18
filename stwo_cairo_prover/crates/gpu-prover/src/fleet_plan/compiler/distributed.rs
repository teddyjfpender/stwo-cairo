//! Conservative deterministic lowering for one proof cooperatively executed
//! by a homogeneous worker set.
//!
//! This compiler deliberately spends memory to keep the first distributed
//! authority simple: one affine storage window per worker/value, except for
//! validated coordinator statement-ingress lineages; exact contiguous shards;
//! explicit point-to-point transfers; and no host bounce.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::*;
use crate::compiled_proof::{
    CompiledProof, OpNode, ProofStage, ValueDesc, ValueOrigin, ValueRange, ValueVersion,
};
use crate::fleet_pow::FleetPowSchedule;
use crate::shape_executable::ShapeExecutableIdentity;
use crate::transcript_plan::CairoBlake2sTranscriptPlan;

mod schedule;
mod storage;

use schedule::DistributedSchedule;
use storage::compile_partitioned_storage;

impl FleetProofPlan {
    /// Derive a deterministic exact-shard Track-A placement.
    ///
    /// Monolithic operations stay on rank zero. Exact operations split their
    /// authority granules across ascending workers. Every remote read,
    /// transcript gather, and proof-tail gather becomes an explicit replica
    /// and transition. Passing this compiler remains host-only validation.
    pub fn compile_track_a_partitioned(
        compiled: Arc<CompiledProof>,
        shape: ShapeExecutableIdentity,
        mut topology: FleetPlacementTopology,
        pow: FleetPowSchedule,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, FleetCompileError> {
        if topology.workers.len() == 1 {
            return Self::compile_track_a_monolithic(compiled, shape, topology, pow, transcript);
        }
        topology.workers.sort_unstable_by_key(|worker| worker.id);
        topology.links.sort_unstable_by_key(|link| link.id);
        topology
            .host_numa
            .sort_unstable_by_key(|capacity| capacity.numa_node);
        pow.validate(topology.workers.len())
            .map_err(FleetCompileError::Pow)?;

        let schedule = DistributedSchedule::compile(&compiled, &topology, transcript)?;
        let mut owners = compile_partitioned_owners(&compiled, &topology, &schedule)?;
        prepare_required_alias_lifetimes(
            &compiled,
            topology.coordinator,
            &mut owners,
            &schedule.operations,
        )?;
        let required_aliases = compile_required_aliases(&compiled, &owners, &schedule.operations)?;
        let materialized = materialize_remote_demands(&compiled, &topology, &schedule, &owners)?;
        let (storages, storage_bindings, in_place_aliases, output_storage) =
            compile_partitioned_storage(
                &compiled,
                topology.coordinator,
                &owners,
                &materialized.replicas,
                &materialized.output_bindings,
                &required_aliases,
            )?;
        let placement = FleetPlacementInput {
            topology,
            pow,
            barrier_steps: schedule.barrier_steps,
            terminal_step: schedule.terminal_step,
            barrier_arrivals: schedule.barrier_arrivals,
            operations: schedule.operations,
            owners,
            replicas: materialized.replicas,
            transitions: materialized.transitions,
            spills: vec![],
            storages,
            storage_bindings,
            in_place_aliases,
            output_storage,
        };
        Self::lower_compiled(compiled, shape, placement, transcript)
            .map_err(FleetCompileError::Lowering)
    }
}

fn prepare_required_alias_lifetimes(
    compiled: &CompiledProof,
    coordinator: WorkerId,
    owners: &mut [FleetOwnerPlacement],
    operations: &[FleetOperationPlacement],
) -> Result<(), FleetCompileError> {
    for operation in compiled.operations() {
        let effect = compiled
            .effect_for(operation.id)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        let placement = operation_placement(operations, operation.id)?;
        for access in effect.accesses() {
            let Some(alias) = access
                .in_place()
                .filter(|alias| alias.requirement == InPlaceAliasRequirement::Required)
            else {
                continue;
            };
            if !matches!(
                placement.executions.as_slice(),
                [FleetOperationExecution {
                    worker,
                    domain: OperationDomain::Monolithic,
                }] if *worker == coordinator
            ) {
                continue;
            }
            let invalid = || FleetCompileError::RequiredAlias {
                operation: operation.id,
                alias: alias.id,
            };
            let source = access.source().ok_or_else(invalid)?.value;
            let destination = access.destination().ok_or_else(invalid)?.value;
            for range in [source, destination] {
                let value = compiled.value(range.version).ok_or_else(invalid)?;
                let full = full_range(value)?;
                let matching = owners
                    .iter()
                    .enumerate()
                    .filter_map(|(index, owner)| {
                        (owner.value.version == range.version).then_some((index, owner))
                    })
                    .collect::<Vec<_>>();
                let [(index, owner)] = matching.as_slice() else {
                    return Err(invalid());
                };
                if owner.worker != coordinator || owner.value.elements != full {
                    return Err(invalid());
                }
                if range.version == source.version {
                    owners[*index].live.end = placement.during.end;
                }
            }
        }
    }
    Ok(())
}

fn compile_partitioned_owners(
    compiled: &CompiledProof,
    topology: &FleetPlacementTopology,
    schedule: &DistributedSchedule,
) -> Result<Vec<FleetOwnerPlacement>, FleetCompileError> {
    let mut owners = Vec::new();
    for value in compiled.values() {
        let start = match value.origin {
            ValueOrigin::ExternalInput(_) | ValueOrigin::Constant(_) => ScheduleStep(0),
            ValueOrigin::TranscriptOutput(_) => {
                transcript_output_release(compiled, &schedule.barrier_steps, value.version)?
            }
            ValueOrigin::OpOutput(producer) => {
                operation_placement(&schedule.operations, producer)?
                    .during
                    .start
            }
        };
        let ranges = match value.origin {
            ValueOrigin::OpOutput(producer) => {
                output_owner_ranges(compiled, schedule, producer, value)?
            }
            _ => vec![(topology.coordinator, full_range(value)?)],
        };
        for (worker, elements) in ranges {
            owners.push(FleetOwnerPlacement {
                value: ValueRange {
                    version: value.version,
                    elements,
                },
                worker,
                live: ScheduleRange::new(start, schedule.terminal_step)
                    .ok_or(FleetCompileError::InvalidSemanticSchedule)?,
            });
        }
    }
    for reuse in statement_host_reuses(compiled)? {
        let overwrite = operation_placement(&schedule.operations, reuse.operation)?
            .during
            .start;
        let matching = owners
            .iter()
            .enumerate()
            .filter_map(|(index, owner)| {
                (owner.value == reuse.predecessor && owner.worker == topology.coordinator)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        let [owner] = matching.as_slice() else {
            return Err(FleetCompileError::InvalidOwnership(
                reuse.predecessor.version,
            ));
        };
        owners[*owner].live.end = overwrite;
    }
    Ok(owners)
}

fn output_owner_ranges(
    compiled: &CompiledProof,
    schedule: &DistributedSchedule,
    producer: crate::compiled_proof::OpId,
    value: &ValueDesc,
) -> Result<Vec<(WorkerId, ElementRange)>, FleetCompileError> {
    let operation = compiled
        .operation(producer)
        .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
    let placement = operation_placement(&schedule.operations, producer)?;
    let effect = compiled
        .effect_for(producer)
        .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
    let mut ranges = Vec::new();
    for execution in &placement.executions {
        for access in effect.accesses() {
            let Some(destination) = access.destination() else {
                continue;
            };
            let range = crate::fleet_plan::validate::projected_range(
                compiled,
                operation,
                *destination,
                execution,
            )
            .map_err(FleetCompileError::Projection)?;
            let carried = (execution.domain == OperationDomain::Monolithic)
                .then(|| exact_partial_atomic_carry_forward(compiled.values(), access))
                .flatten()
                .filter(|carry| carry.full_destination.version == value.version);
            if let Some(carry) = carried {
                ranges.push((execution.worker, carry.full_destination.elements));
            } else if range.version == value.version {
                ranges.push((execution.worker, range.elements));
            }
        }
    }
    canonical_owner_union(value.version, full_range(value)?, ranges)
}

fn canonical_owner_union(
    version: ValueVersion,
    full: ElementRange,
    mut ranges: Vec<(WorkerId, ElementRange)>,
) -> Result<Vec<(WorkerId, ElementRange)>, FleetCompileError> {
    ranges.sort_unstable_by_key(|(worker, range)| (range.start, range.end, *worker));
    let mut cursor = full.start;
    let mut merged: Vec<(WorkerId, ElementRange)> = Vec::new();
    for (worker, range) in ranges {
        if range.start != cursor || range.end > full.end {
            return Err(FleetCompileError::InvalidOwnership(version));
        }
        if let Some((previous_worker, previous)) = merged.last_mut() {
            if *previous_worker == worker && previous.end == range.start {
                previous.end = range.end;
            } else {
                merged.push((worker, range));
            }
        } else {
            merged.push((worker, range));
        }
        cursor = range.end;
    }
    if cursor != full.end {
        return Err(FleetCompileError::InvalidOwnership(version));
    }
    Ok(merged)
}

struct Materialized {
    replicas: Vec<FleetReplicaPlacement>,
    transitions: Vec<FleetTransitionPlacement>,
    output_bindings: Vec<OutputBinding>,
}

#[derive(Clone, Copy)]
struct OutputBinding {
    value: ValueRange,
    offset_bytes: usize,
}

#[derive(Clone)]
struct Demand {
    value: ValueRange,
    source: WorkerId,
    destination: WorkerId,
    during: ScheduleRange,
    interval: ExecutionInterval,
    live_end: ScheduleStep,
}

fn materialize_remote_demands(
    compiled: &CompiledProof,
    topology: &FleetPlacementTopology,
    schedule: &DistributedSchedule,
    owners: &[FleetOwnerPlacement],
) -> Result<Materialized, FleetCompileError> {
    let mut fixed = BTreeSet::new();
    let mut demands = Vec::new();

    for operation in compiled.operations() {
        let placement = operation_placement(&schedule.operations, operation.id)?;
        let pre = schedule.pre_operation[operation.id.0 as usize];
        let interval = stage_interval(operation, compiled)?;
        let effect = compiled
            .effect_for(operation.id)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        for execution in &placement.executions {
            for source in effect
                .accesses()
                .iter()
                .filter_map(|access| access.source())
            {
                let range = crate::fleet_plan::validate::projected_range(
                    compiled, operation, *source, execution,
                )
                .map_err(FleetCompileError::Projection)?;
                add_remote_demand(
                    compiled,
                    owners,
                    range,
                    execution.worker,
                    pre,
                    interval,
                    placement.during.end,
                    &mut fixed,
                    &mut demands,
                )?;
            }
        }
    }

    for (ordinal, segment) in compiled.transcript_segments().iter().enumerate() {
        let gather = schedule.transcript_gathers[ordinal];
        let release = schedule.barrier_steps[ordinal];
        for &range in &segment.consumed {
            add_remote_demand(
                compiled,
                owners,
                range,
                topology.coordinator,
                gather,
                ExecutionInterval::BeforeBarrier(
                    u32::try_from(ordinal).map_err(|_| FleetCompileError::SizeOverflow)?,
                ),
                release,
                &mut fixed,
                &mut demands,
            )?;
        }
    }

    let mut output_bindings = Vec::new();
    for fragment in &compiled.output().fragments {
        let chunks = owner_intersections(owners, fragment.source)?;
        for (owner, elements) in chunks {
            let value = ValueRange {
                version: fragment.source.version,
                elements,
            };
            if owner != topology.coordinator {
                add_remote_demand(
                    compiled,
                    owners,
                    value,
                    topology.coordinator,
                    schedule.tail_gather,
                    ExecutionInterval::AfterFinalBarrier,
                    schedule.terminal_step,
                    &mut fixed,
                    &mut demands,
                )?;
            }
            let relative = elements
                .start
                .checked_sub(fragment.source.elements.start)
                .ok_or(FleetCompileError::InvalidOwnership(value.version))?;
            let word = fragment
                .destination
                .start
                .checked_add(relative)
                .ok_or(FleetCompileError::SizeOverflow)?;
            output_bindings.push(OutputBinding {
                value,
                offset_bytes: word
                    .checked_mul(core::mem::size_of::<u32>())
                    .ok_or(FleetCompileError::SizeOverflow)?,
            });
        }
    }

    demands = segment_demands(compiled, topology, canonical_demands(demands)?)?;
    output_bindings = refine_output_bindings(
        compiled,
        topology.coordinator,
        schedule.terminal_step,
        output_bindings,
        &demands,
    )?;
    let mut replicas = Vec::new();
    for (version, worker) in fixed {
        let value = compiled
            .value(version)
            .ok_or(FleetCompileError::InvalidOwnership(version))?;
        replicas.push(FleetReplicaPlacement {
            id: next_replica_id(replicas.len())?,
            value: ValueRange {
                version,
                elements: full_range(value)?,
            },
            canonical_worker: topology.coordinator,
            worker,
            layout: value.layout.clone(),
            origin: ReplicaOrigin::InstalledFixed,
            live: ScheduleRange::new(ScheduleStep(0), schedule.terminal_step)
                .ok_or(FleetCompileError::InvalidSemanticSchedule)?,
        });
    }
    let mut transitions = Vec::new();
    for demand in demands {
        let value = compiled
            .value(demand.value.version)
            .ok_or(FleetCompileError::InvalidOwnership(demand.value.version))?;
        let transition_id = LayoutTransitionId(
            u32::try_from(transitions.len()).map_err(|_| FleetCompileError::SizeOverflow)?,
        );
        let replica_id = next_replica_id(replicas.len())?;
        let route = route(
            topology,
            demand.source,
            demand.destination,
            value,
            demand.value,
        )?;
        replicas.push(FleetReplicaPlacement {
            id: replica_id,
            value: demand.value,
            canonical_worker: demand.source,
            worker: demand.destination,
            layout: value.layout.clone(),
            origin: ReplicaOrigin::Transition(transition_id),
            live: ScheduleRange::new(demand.during.start, demand.live_end)
                .ok_or(FleetCompileError::InvalidSemanticSchedule)?,
        });
        transitions.push(FleetTransitionPlacement {
            id: transition_id,
            value: demand.value,
            source_worker: demand.source,
            destination_replica: replica_id,
            axes: value
                .layout
                .axes
                .iter()
                .map(|axis| AxisMap {
                    source: axis.tag,
                    destination: axis.tag,
                })
                .collect(),
            interval: demand.interval,
            during: demand.during,
            scratch_bytes: 0,
            scratch_worker: demand.source,
            route,
        });
    }
    Ok(Materialized {
        replicas,
        transitions,
        output_bindings,
    })
}

#[allow(clippy::too_many_arguments)]
fn add_remote_demand(
    compiled: &CompiledProof,
    owners: &[FleetOwnerPlacement],
    range: ValueRange,
    destination: WorkerId,
    during: ScheduleRange,
    interval: ExecutionInterval,
    live_end: ScheduleStep,
    fixed: &mut BTreeSet<(ValueVersion, WorkerId)>,
    demands: &mut Vec<Demand>,
) -> Result<(), FleetCompileError> {
    let value = compiled
        .value(range.version)
        .ok_or(FleetCompileError::InvalidOwnership(range.version))?;
    for (source, elements) in owner_intersections(owners, range)? {
        if source == destination {
            continue;
        }
        if matches!(value.origin, ValueOrigin::Constant(_)) {
            fixed.insert((range.version, destination));
        } else {
            demands.push(Demand {
                value: ValueRange {
                    version: range.version,
                    elements,
                },
                source,
                destination,
                during,
                interval,
                live_end,
            });
        }
    }
    Ok(())
}

fn owner_intersections(
    owners: &[FleetOwnerPlacement],
    range: ValueRange,
) -> Result<Vec<(WorkerId, ElementRange)>, FleetCompileError> {
    let mut chunks = owners
        .iter()
        .filter(|owner| {
            owner.value.version == range.version && owner.value.elements.overlaps(range.elements)
        })
        .map(|owner| {
            (
                owner.worker,
                ElementRange {
                    start: owner.value.elements.start.max(range.elements.start),
                    end: owner.value.elements.end.min(range.elements.end),
                },
            )
        })
        .collect::<Vec<_>>();
    chunks.sort_unstable_by_key(|(worker, elements)| (elements.start, elements.end, *worker));
    let mut cursor = range.elements.start;
    for (_, elements) in &chunks {
        if elements.start != cursor {
            return Err(FleetCompileError::InvalidOwnership(range.version));
        }
        cursor = elements.end;
    }
    if cursor != range.elements.end {
        return Err(FleetCompileError::InvalidOwnership(range.version));
    }
    Ok(chunks)
}

fn canonical_demands(mut demands: Vec<Demand>) -> Result<Vec<Demand>, FleetCompileError> {
    demands.sort_unstable_by_key(|demand| {
        (
            demand.destination,
            demand.value.version,
            demand.source,
            demand.value.elements.start,
            demand.value.elements.end,
            demand.during.start,
            demand.during.end,
            demand.live_end,
        )
    });
    let mut merged: Vec<Demand> = Vec::new();
    for demand in demands {
        if let Some(previous) = merged.last_mut() {
            let same_location = previous.destination == demand.destination
                && previous.value.version == demand.value.version
                && previous.source == demand.source;
            if same_location && demand.value.elements.start <= previous.value.elements.end {
                previous.value.elements.end =
                    previous.value.elements.end.max(demand.value.elements.end);
                if demand.during.start < previous.during.start {
                    previous.during = demand.during;
                    previous.interval = demand.interval;
                }
                previous.live_end = previous.live_end.max(demand.live_end);
                continue;
            }
        }
        merged.push(demand);
    }
    Ok(merged)
}

fn segment_demands(
    compiled: &CompiledProof,
    topology: &FleetPlacementTopology,
    demands: Vec<Demand>,
) -> Result<Vec<Demand>, FleetCompileError> {
    let mut segmented = Vec::new();
    for demand in demands {
        let value = compiled
            .value(demand.value.version)
            .ok_or(FleetCompileError::InvalidOwnership(demand.value.version))?;
        let link = topology
            .links
            .iter()
            .find(|link| link.source == demand.source && link.destination == demand.destination)
            .ok_or(FleetCompileError::MissingRoute {
                source: demand.source,
                destination: demand.destination,
            })?;
        let element_bytes = value.layout.element.bytes;
        let demand_bytes = demand
            .value
            .elements
            .len()
            .checked_mul(element_bytes)
            .ok_or(FleetCompileError::SizeOverflow)?;
        if demand_bytes <= link.max_transfer_bytes {
            segmented.push(demand);
            continue;
        }
        let boundary_elements =
            value.alignment / greatest_common_divisor(value.alignment, element_bytes);
        let max_elements = link.max_transfer_bytes / element_bytes;
        let segment_elements = max_elements / boundary_elements * boundary_elements;
        if segment_elements == 0 {
            return Err(FleetCompileError::TransferTooLarge {
                route: link.id,
                bytes: demand_bytes,
                limit: link.max_transfer_bytes,
            });
        }
        let mut cursor = demand.value.elements.start;
        while cursor < demand.value.elements.end {
            let end = cursor
                .saturating_add(segment_elements)
                .min(demand.value.elements.end);
            let mut segment = demand.clone();
            segment.value.elements = ElementRange::new(cursor, end)
                .ok_or(FleetCompileError::InvalidOwnership(demand.value.version))?;
            segmented.push(segment);
            cursor = end;
        }
    }
    Ok(segmented)
}

fn refine_output_bindings(
    compiled: &CompiledProof,
    coordinator: WorkerId,
    terminal: ScheduleStep,
    bindings: Vec<OutputBinding>,
    demands: &[Demand],
) -> Result<Vec<OutputBinding>, FleetCompileError> {
    let mut refined = Vec::new();
    for binding in bindings {
        let value = compiled
            .value(binding.value.version)
            .ok_or(FleetCompileError::InvalidOwnership(binding.value.version))?;
        let mut cuts = BTreeSet::from([binding.value.elements.start, binding.value.elements.end]);
        for demand in demands.iter().filter(|demand| {
            demand.destination == coordinator
                && demand.live_end == terminal
                && demand.value.version == binding.value.version
                && demand.value.elements.overlaps(binding.value.elements)
        }) {
            cuts.insert(
                demand
                    .value
                    .elements
                    .start
                    .max(binding.value.elements.start),
            );
            cuts.insert(demand.value.elements.end.min(binding.value.elements.end));
        }
        for (&start, &end) in cuts.iter().zip(cuts.iter().skip(1)) {
            refined.push(OutputBinding {
                value: ValueRange {
                    version: binding.value.version,
                    elements: ElementRange::new(start, end)
                        .ok_or(FleetCompileError::InvalidOwnership(binding.value.version))?,
                },
                offset_bytes: binding
                    .offset_bytes
                    .checked_add(
                        start
                            .checked_sub(binding.value.elements.start)
                            .and_then(|elements| elements.checked_mul(value.layout.element.bytes))
                            .ok_or(FleetCompileError::SizeOverflow)?,
                    )
                    .ok_or(FleetCompileError::SizeOverflow)?,
            });
        }
    }
    Ok(refined)
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn route(
    topology: &FleetPlacementTopology,
    source: WorkerId,
    destination: WorkerId,
    value: &ValueDesc,
    range: ValueRange,
) -> Result<FleetLinkId, FleetCompileError> {
    let link = topology
        .links
        .iter()
        .find(|link| link.source == source && link.destination == destination)
        .ok_or(FleetCompileError::MissingRoute {
            source,
            destination,
        })?;
    let bytes = range
        .elements
        .len()
        .checked_mul(value.layout.element.bytes)
        .ok_or(FleetCompileError::SizeOverflow)?;
    if bytes > link.max_transfer_bytes {
        return Err(FleetCompileError::TransferTooLarge {
            route: link.id,
            bytes,
            limit: link.max_transfer_bytes,
        });
    }
    Ok(link.id)
}

fn stage_interval(
    operation: &OpNode,
    compiled: &CompiledProof,
) -> Result<ExecutionInterval, FleetCompileError> {
    match operation.stage {
        ProofStage::BeforeTranscript(segment) => compiled
            .transcript_segments()
            .iter()
            .position(|candidate| candidate.segment == segment)
            .and_then(|ordinal| u32::try_from(ordinal).ok())
            .map(ExecutionInterval::BeforeBarrier)
            .ok_or(FleetCompileError::InvalidSemanticSchedule),
        ProofStage::AfterTranscript => Ok(ExecutionInterval::AfterFinalBarrier),
    }
}

fn next_replica_id(index: usize) -> Result<ReplicaId, FleetCompileError> {
    Ok(ReplicaId(
        u32::try_from(index).map_err(|_| FleetCompileError::SizeOverflow)?,
    ))
}
