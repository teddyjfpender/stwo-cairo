//! Pure preflight for the address-free facts consumed by the closure replay.

use std::collections::BTreeSet;

use super::*;
use crate::compiled_proof::{OpId, ProofStage, ValueRange};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TranscriptRecord {
    binding: FleetTranscriptBinding,
    value: ValueRange,
    barrier_ordinal: u32,
    release_step: ScheduleStep,
}

pub(super) fn require_supported_topology(
    plan: &FleetProofPlan,
) -> Result<usize, FleetTwoRankStructuralClosureError> {
    let workers = &plan.placement().topology.workers;
    if !matches!(workers.len(), 2 | 4 | 8 | 16) {
        return Err(FleetTwoRankStructuralClosureError::WorkerCount {
            actual: workers.len(),
        });
    }
    if workers
        .iter()
        .enumerate()
        .any(|(rank, worker)| worker.id.0 as usize != rank)
        || plan.placement().topology.coordinator != WorkerId(0)
    {
        return Err(FleetTwoRankStructuralClosureError::InstallClosure {
            worker: WorkerId(0),
        });
    }
    Ok(workers.len())
}

pub(super) fn validate_install_closure(
    plan: &FleetProofPlan,
    view: &FleetRuntimeView,
    installs: &[FleetWorkerInstallPlan],
) -> Result<BTreeSet<u32>, FleetTwoRankStructuralClosureError> {
    if installs.len() != plan.placement().topology.workers.len() {
        return Err(FleetTwoRankStructuralClosureError::InstallClosure {
            worker: WorkerId(0),
        });
    }
    let mut installed_arrivals = BTreeSet::new();
    let mut inbound = BTreeSet::new();
    let mut outbound = BTreeSet::new();

    for (rank, install) in installs.iter().enumerate() {
        let worker = WorkerId(
            u16::try_from(rank).map_err(|_| FleetTwoRankStructuralClosureError::SizeOverflow)?,
        );
        if install.executions().is_empty() {
            return Err(FleetTwoRankStructuralClosureError::IdleWorker { worker });
        }
        validate_target_and_capacity(plan, view, install, rank, worker)?;
        for &arrival in install.barrier_arrivals() {
            if arrival.worker != worker
                || !installed_arrivals.insert((
                    arrival.barrier_ordinal,
                    arrival.worker,
                    arrival.ready_step,
                ))
            {
                return Err(FleetTwoRankStructuralClosureError::InstallClosure { worker });
            }
        }
        for &span in install.inbound() {
            validate_endpoint(view, span)?;
            if span.peer != worker || !inbound.insert((worker, span.edge_ordinal)) {
                return Err(FleetTwoRankStructuralClosureError::TransferClosure {
                    edge: span.edge_ordinal,
                });
            }
        }
        for &span in install.outbound() {
            validate_endpoint(view, span)?;
            if span.owner != worker || !outbound.insert((worker, span.edge_ordinal)) {
                return Err(FleetTwoRankStructuralClosureError::TransferClosure {
                    edge: span.edge_ordinal,
                });
            }
        }
    }

    let expected_arrivals = plan
        .placement()
        .barrier_arrivals
        .iter()
        .map(|arrival| (arrival.barrier_ordinal, arrival.worker, arrival.ready_step))
        .collect::<BTreeSet<_>>();
    if installed_arrivals != expected_arrivals {
        return Err(FleetTwoRankStructuralClosureError::InstallClosure {
            worker: WorkerId(0),
        });
    }

    validate_view_completeness(plan, view.spans())?;
    for &span in view.spans() {
        validate_span_projection(plan, span)?;
        if !outbound.contains(&(span.owner, span.edge_ordinal))
            || !inbound.contains(&(span.peer, span.edge_ordinal))
            || span.source.worker != span.owner
            || span.destination.worker != span.peer
            || span.source.bytes == 0
            || span.source.bytes != span.destination.bytes
            || installs
                .get(usize::from(span.owner.0))
                .is_none_or(|install| !installed_window_contains(install, span.source))
            || installs
                .get(usize::from(span.peer.0))
                .is_none_or(|install| !installed_window_contains(install, span.destination))
        {
            return Err(FleetTwoRankStructuralClosureError::TransferClosure {
                edge: span.edge_ordinal,
            });
        }
    }

    let Some(coordinator) = installs[0].coordinator() else {
        return Err(FleetTwoRankStructuralClosureError::InstallClosure {
            worker: WorkerId(0),
        });
    };
    if installs
        .iter()
        .skip(1)
        .any(|install| install.coordinator().is_some())
        || coordinator.transcript_barriers != plan.barriers()
        || coordinator.output.storage != plan.placement().output_storage
    {
        return Err(FleetTwoRankStructuralClosureError::InstallClosure {
            worker: WorkerId(0),
        });
    }
    validate_transcript_install(plan, &installs[0], coordinator)
}

pub(in crate::fleet_plan) fn validate_view_completeness(
    plan: &FleetProofPlan,
    spans: &[FleetTransferSpan],
) -> Result<(), FleetTwoRankStructuralClosureError> {
    for transition in &plan.placement().transitions {
        let mut projected = spans
            .iter()
            .filter(|span| span.transition == transition.id)
            .collect::<Vec<_>>();
        projected.sort_unstable_by_key(|span| (span.elements.start, span.elements.end));
        if projected.is_empty() {
            return Err(FleetTwoRankStructuralClosureError::TransferClosure { edge: u64::MAX });
        }
        let mut cursor = transition.value.elements.start;
        for (ordinal, span) in projected.into_iter().enumerate() {
            let expected_ordinal = u32::try_from(ordinal)
                .map_err(|_| FleetTwoRankStructuralClosureError::SizeOverflow)?;
            if span.span_ordinal != expected_ordinal
                || span.elements.start != cursor
                || span.elements.end <= span.elements.start
            {
                return Err(FleetTwoRankStructuralClosureError::TransferClosure {
                    edge: span.edge_ordinal,
                });
            }
            cursor = span.elements.end;
        }
        if cursor != transition.value.elements.end {
            return Err(FleetTwoRankStructuralClosureError::TransferClosure { edge: u64::MAX });
        }
    }
    Ok(())
}

fn validate_target_and_capacity(
    plan: &FleetProofPlan,
    view: &FleetRuntimeView,
    install: &FleetWorkerInstallPlan,
    rank: usize,
    worker: WorkerId,
) -> Result<(), FleetTwoRankStructuralClosureError> {
    let topology = &plan.placement().topology;
    let spec = &topology.workers[rank];
    let reserve = view
        .exchange_reserves()
        .iter()
        .find(|reserve| reserve.worker == worker)
        .ok_or(FleetTwoRankStructuralClosureError::InstallClosure { worker })?;
    let target = install.target();
    let capacity = install.capacity();
    let packed = capacity
        .slab_bytes
        .checked_add(reserve.declared_bytes)
        .ok_or(FleetTwoRankStructuralClosureError::SizeOverflow)?;
    if install.plan_identity() != plan.identity()
        || target.worker != worker
        || target.gpu_class != topology.gpu_class
        || target.module_pack_identity != topology.module_pack_identity
        || target.fixed_image_identity != topology.fixed_image_identity
        || capacity.required_exchange_bytes != reserve.required_bytes
        || capacity.exchange_reserve_bytes != reserve.declared_bytes
        || reserve.declared_bytes != spec.exchange_reserve_bytes
        || capacity.packed_slab_and_exchange_bytes != packed
        || packed > spec.capacity_bytes
        || capacity.capacity_bytes != spec.capacity_bytes
    {
        return Err(FleetTwoRankStructuralClosureError::InstallClosure { worker });
    }
    Ok(())
}

pub(in crate::fleet_plan) fn validate_span_projection(
    plan: &FleetProofPlan,
    span: FleetTransferSpan,
) -> Result<(), FleetTwoRankStructuralClosureError> {
    let transition = plan
        .placement()
        .transitions
        .get(span.transition.0 as usize)
        .filter(|transition| transition.id == span.transition);
    let replica = transition.and_then(|transition| {
        plan.placement()
            .replicas
            .get(transition.destination_replica.0 as usize)
            .filter(|replica| replica.id == transition.destination_replica)
            .map(|replica| (transition, replica))
    });
    if replica.is_none_or(|(transition, replica)| {
        span.during != transition.during
            || span.route != transition.route
            || span.owner != transition.source_worker
            || span.peer != replica.worker
    }) {
        return Err(FleetTwoRankStructuralClosureError::TransferClosure {
            edge: span.edge_ordinal,
        });
    }
    Ok(())
}

fn validate_endpoint(
    view: &FleetRuntimeView,
    actual: FleetTransferSpan,
) -> Result<(), FleetTwoRankStructuralClosureError> {
    let index = usize::try_from(actual.edge_ordinal).map_err(|_| {
        FleetTwoRankStructuralClosureError::TransferClosure {
            edge: actual.edge_ordinal,
        }
    })?;
    if view.spans().get(index) != Some(&actual) {
        return Err(FleetTwoRankStructuralClosureError::TransferClosure {
            edge: actual.edge_ordinal,
        });
    }
    Ok(())
}

fn installed_window_contains(install: &FleetWorkerInstallPlan, window: FleetStorageWindow) -> bool {
    install.storages().iter().any(|storage| {
        storage.storage == window.storage
            && window
                .offset_bytes
                .checked_add(window.bytes)
                .is_some_and(|end| end <= storage.bytes)
    })
}

pub(in crate::fleet_plan) fn validate_transcript_install(
    plan: &FleetProofPlan,
    install: &FleetWorkerInstallPlan,
    coordinator: &FleetCoordinatorInstall,
) -> Result<BTreeSet<u32>, FleetTwoRankStructuralClosureError> {
    let expected = plan
        .compiled()
        .transcript_inputs()
        .iter()
        .map(|binding| {
            transcript_record(
                plan,
                FleetTranscriptBinding::Input(binding.id),
                ValueRange {
                    version: binding.value,
                    elements: binding.elements,
                },
                false,
            )
        })
        .chain(plan.compiled().transcript_outputs().iter().map(|binding| {
            transcript_record(
                plan,
                FleetTranscriptBinding::Output(binding.id),
                ValueRange {
                    version: binding.value,
                    elements: binding.elements,
                },
                true,
            )
        }))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut actual = BTreeSet::new();
    for record in &coordinator.transcript {
        let canonical = transcript_record(
            plan,
            record.binding,
            record.value,
            matches!(record.binding, FleetTranscriptBinding::Output(_)),
        )?;
        let bytes = record
            .value
            .elements
            .len()
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(FleetTwoRankStructuralClosureError::SizeOverflow)?;
        let storage = install
            .storages()
            .iter()
            .find(|storage| storage.storage == record.window.storage);
        let installed = TranscriptRecord {
            binding: record.binding,
            value: record.value,
            barrier_ordinal: record.barrier_ordinal,
            release_step: record.release_step,
        };
        if canonical != installed
            || bytes == 0
            || record.window.bytes != bytes
            || storage.is_none_or(|storage| {
                record.window.slab_offset_bytes != storage.slab_offset_bytes
                    || record
                        .window
                        .offset_bytes
                        .checked_add(bytes)
                        .is_none_or(|end| end > storage.bytes)
            })
            || !actual.insert(canonical)
        {
            return Err(FleetTwoRankStructuralClosureError::TranscriptClosure);
        }
    }
    if actual != expected {
        return Err(FleetTwoRankStructuralClosureError::TranscriptClosure);
    }
    Ok(plan
        .barriers()
        .iter()
        .map(|barrier| barrier.ordinal)
        .collect())
}

fn transcript_record(
    plan: &FleetProofPlan,
    binding: FleetTranscriptBinding,
    value: ValueRange,
    produced: bool,
) -> Result<TranscriptRecord, FleetTwoRankStructuralClosureError> {
    let mut matches = plan
        .compiled()
        .transcript_segments()
        .iter()
        .filter(|segment| {
            let values = if produced {
                &segment.produced
            } else {
                &segment.consumed
            };
            values.contains(&value)
        });
    let segment = matches
        .next()
        .filter(|_| matches.next().is_none())
        .ok_or(FleetTwoRankStructuralClosureError::TranscriptClosure)?;
    let barrier = plan
        .barriers()
        .iter()
        .find(|barrier| barrier.segment == segment.segment)
        .ok_or(FleetTwoRankStructuralClosureError::TranscriptClosure)?;
    Ok(TranscriptRecord {
        binding,
        value,
        barrier_ordinal: barrier.ordinal,
        release_step: barrier.release_step,
    })
}

pub(super) fn inside_interval(plan: &FleetProofPlan, segment: u32, during: ScheduleRange) -> bool {
    let index = segment as usize;
    let released_after = index
        .checked_sub(1)
        .and_then(|prior| plan.barriers().get(prior))
        .map_or(ScheduleStep(0), |barrier| barrier.release_step);
    let closed_before = plan
        .barriers()
        .get(index)
        .map_or(plan.terminal_step(), |barrier| barrier.release_step);
    index <= plan.barriers().len() && during.start >= released_after && during.end < closed_before
}

pub(super) fn operation_segment(
    plan: &FleetProofPlan,
    operation: OpId,
) -> Result<u32, FleetTwoRankStructuralClosureError> {
    let operation = plan
        .compiled()
        .operation(operation)
        .ok_or(FleetTwoRankStructuralClosureError::ExecutionClosure)?;
    match operation.stage {
        ProofStage::BeforeTranscript(segment) => plan
            .barriers()
            .iter()
            .position(|barrier| barrier.segment == segment)
            .and_then(|ordinal| u32::try_from(ordinal).ok())
            .ok_or(FleetTwoRankStructuralClosureError::ExecutionClosure),
        ProofStage::AfterTranscript => u32::try_from(plan.barriers().len())
            .map_err(|_| FleetTwoRankStructuralClosureError::SizeOverflow),
    }
}

pub(super) fn transition_segment(
    plan: &FleetProofPlan,
    transition: LayoutTransitionId,
) -> Result<u32, FleetTwoRankStructuralClosureError> {
    let transition = plan
        .placement()
        .transitions
        .get(transition.0 as usize)
        .filter(|candidate| candidate.id == transition)
        .ok_or(FleetTwoRankStructuralClosureError::TransferClosure { edge: u64::MAX })?;
    match transition.interval {
        ExecutionInterval::BeforeBarrier(ordinal) if (ordinal as usize) < plan.barriers().len() => {
            Ok(ordinal)
        }
        ExecutionInterval::AfterFinalBarrier => u32::try_from(plan.barriers().len())
            .map_err(|_| FleetTwoRankStructuralClosureError::SizeOverflow),
        _ => Err(FleetTwoRankStructuralClosureError::TransferClosure { edge: u64::MAX }),
    }
}
