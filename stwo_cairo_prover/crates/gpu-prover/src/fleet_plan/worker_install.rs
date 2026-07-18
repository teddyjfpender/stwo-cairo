//! Projects a [`FleetProofPlan`] into deterministic address-free worker slab
//! geometry and executable windows; it installs no CUDA runtime resources.

use std::collections::BTreeMap;

use super::*;
use crate::compiled_proof::{
    EffectBindingId, EffectContractId, ExecutionPrimitive, OpId, PartitionAuthorityKind,
    PartitionEffectProjection, TranscriptInputId, TranscriptOutputId, ValueRange,
};

mod error;
mod statement_host_ingress;

pub use error::FleetWorkerInstallError;
pub use statement_host_ingress::FleetStatementHostIngress;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetWorkerStorage {
    pub storage: StorageId,
    pub slab_offset_bytes: usize,
    pub bytes: usize,
    pub alignment_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetInstallWindow {
    pub storage: StorageId,
    pub offset_bytes: usize,
    pub slab_offset_bytes: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetEffectBinding {
    pub binding: EffectBindingId,
    pub source: Option<ValueRange>,
    pub destination: Option<ValueRange>,
    pub window: FleetInstallWindow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetWorkerExecutable {
    /// Composite children carry a stable ordinal; ordinary operations do not.
    pub child_ordinal: Option<u32>,
    pub effects: Vec<FleetEffectBinding>,
    pub statement_host_ingress: Option<FleetStatementHostIngress>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetWorkerExecution {
    pub step: ScheduleStep,
    pub operation: OpId,
    pub domain: OperationDomain,
    pub during: ScheduleRange,
    pub executables: Vec<FleetWorkerExecutable>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetCoordinatorInstall {
    pub transcript_barriers: Vec<TranscriptBarrier>,
    pub transcript: Vec<FleetTranscriptInstall>,
    pub output: FleetInstallWindow,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FleetTranscriptBinding {
    Input(TranscriptInputId),
    Output(TranscriptOutputId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetTranscriptInstall {
    pub binding: FleetTranscriptBinding,
    pub value: ValueRange,
    pub barrier_ordinal: u32,
    pub release_step: ScheduleStep,
    pub window: FleetInstallWindow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetWorkerInstallTarget {
    pub worker: WorkerId,
    pub gpu_class: ConsumerGpuClass,
    pub module_pack_identity: [u8; 32],
    pub fixed_image_identity: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetWorkerInstallCapacity {
    pub slab_bytes: usize,
    pub slab_alignment_bytes: usize,
    pub required_exchange_bytes: usize,
    pub exchange_reserve_bytes: usize,
    pub packed_slab_and_exchange_bytes: usize,
    pub capacity_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetWorkerInstallPlan {
    plan_identity: [u8; 32],
    target: FleetWorkerInstallTarget,
    storages: Vec<FleetWorkerStorage>,
    capacity: FleetWorkerInstallCapacity,
    executions: Vec<FleetWorkerExecution>,
    inbound: Vec<FleetTransferSpan>,
    outbound: Vec<FleetTransferSpan>,
    barrier_arrivals: Vec<BarrierArrival>,
    coordinator: Option<FleetCoordinatorInstall>,
}

impl FleetWorkerInstallPlan {
    pub const fn plan_identity(&self) -> [u8; 32] {
        self.plan_identity
    }

    pub const fn target(&self) -> &FleetWorkerInstallTarget {
        &self.target
    }

    pub fn storages(&self) -> &[FleetWorkerStorage] {
        &self.storages
    }

    pub const fn capacity(&self) -> &FleetWorkerInstallCapacity {
        &self.capacity
    }

    pub fn executions(&self) -> &[FleetWorkerExecution] {
        &self.executions
    }

    pub fn inbound(&self) -> &[FleetTransferSpan] {
        &self.inbound
    }

    pub fn outbound(&self) -> &[FleetTransferSpan] {
        &self.outbound
    }

    pub fn barrier_arrivals(&self) -> &[BarrierArrival] {
        &self.barrier_arrivals
    }

    pub const fn coordinator(&self) -> Option<&FleetCoordinatorInstall> {
        self.coordinator.as_ref()
    }
}

impl FleetProofPlan {
    pub fn worker_install_plan(
        &self,
        worker: WorkerId,
    ) -> Result<FleetWorkerInstallPlan, FleetWorkerInstallError> {
        if !self.placement().spills.is_empty() {
            return Err(FleetWorkerInstallError::UnsupportedSpill);
        }
        if let Some(transition) = self
            .placement()
            .transitions
            .iter()
            .find(|transition| transition.scratch_bytes != 0)
        {
            return Err(FleetWorkerInstallError::UnsupportedScratch(transition.id));
        }
        for value in self.compiled().values() {
            if value.layout.element != ElementType::U32 {
                return Err(FleetWorkerInstallError::UnsupportedElement(value.version));
            }
        }

        let spec = self
            .placement()
            .topology
            .workers
            .iter()
            .find(|candidate| candidate.id == worker)
            .ok_or(FleetWorkerInstallError::UnknownWorker(worker))?;
        let (storages, slab_bytes, slab_alignment_bytes) = worker_storages(self, worker)?;
        let storage_by_id = storages
            .iter()
            .map(|storage| (storage.storage, *storage))
            .collect::<BTreeMap<_, _>>();
        let executions = worker_executions(self, worker, &storage_by_id)?;
        let view = self.runtime_view()?;
        let inbound = view
            .spans()
            .iter()
            .copied()
            .filter(|span| span.peer == worker)
            .collect::<Vec<_>>();
        let outbound = view
            .spans()
            .iter()
            .copied()
            .filter(|span| span.owner == worker)
            .collect::<Vec<_>>();
        let reserve = view
            .exchange_reserves()
            .iter()
            .find(|reserve| reserve.worker == worker)
            .ok_or(FleetWorkerInstallError::UnknownWorker(worker))?;
        let packed_slab_and_exchange_bytes = slab_bytes
            .checked_add(spec.exchange_reserve_bytes)
            .ok_or(FleetWorkerInstallError::SizeOverflow)?;
        if packed_slab_and_exchange_bytes > spec.capacity_bytes {
            return Err(FleetWorkerInstallError::CapacityExceeded {
                worker,
                required: packed_slab_and_exchange_bytes,
                capacity: spec.capacity_bytes,
            });
        }

        let mut barrier_arrivals = self
            .placement()
            .barrier_arrivals
            .iter()
            .copied()
            .filter(|arrival| arrival.worker == worker)
            .collect::<Vec<_>>();
        barrier_arrivals.sort_unstable_by_key(|arrival| arrival.barrier_ordinal);
        let coordinator = (worker == self.placement().topology.coordinator)
            .then(|| coordinator_install(self, &storage_by_id))
            .transpose()?;

        Ok(FleetWorkerInstallPlan {
            plan_identity: self.identity(),
            target: FleetWorkerInstallTarget {
                worker,
                gpu_class: self.placement().topology.gpu_class,
                module_pack_identity: self.placement().topology.module_pack_identity,
                fixed_image_identity: self.placement().topology.fixed_image_identity,
            },
            storages,
            capacity: FleetWorkerInstallCapacity {
                slab_bytes,
                slab_alignment_bytes,
                required_exchange_bytes: reserve.required_bytes,
                exchange_reserve_bytes: spec.exchange_reserve_bytes,
                packed_slab_and_exchange_bytes,
                capacity_bytes: spec.capacity_bytes,
            },
            executions,
            inbound,
            outbound,
            barrier_arrivals,
            coordinator,
        })
    }
}

fn worker_storages(
    plan: &FleetProofPlan,
    worker: WorkerId,
) -> Result<(Vec<FleetWorkerStorage>, usize, usize), FleetWorkerInstallError> {
    let mut local = plan
        .placement()
        .storages
        .iter()
        .copied()
        .filter(|storage| storage.worker == worker)
        .collect::<Vec<_>>();
    local.sort_unstable_by_key(|storage| storage.id);

    let mut cursor = 0usize;
    let mut slab_alignment_bytes = 1usize;
    let mut output = Vec::with_capacity(local.len());
    for storage in local {
        if storage.bytes == 0
            || storage.alignment_bytes == 0
            || !storage.alignment_bytes.is_power_of_two()
        {
            return Err(FleetWorkerInstallError::InvalidStorage(storage.id));
        }
        let slab_offset_bytes = align_up(cursor, storage.alignment_bytes)?;
        slab_alignment_bytes = slab_alignment_bytes.max(storage.alignment_bytes);
        cursor = slab_offset_bytes
            .checked_add(storage.bytes)
            .ok_or(FleetWorkerInstallError::SizeOverflow)?;
        output.push(FleetWorkerStorage {
            storage: storage.id,
            slab_offset_bytes,
            bytes: storage.bytes,
            alignment_bytes: storage.alignment_bytes,
        });
    }
    Ok((output, cursor, slab_alignment_bytes))
}

fn worker_executions(
    plan: &FleetProofPlan,
    worker: WorkerId,
    storages: &BTreeMap<StorageId, FleetWorkerStorage>,
) -> Result<Vec<FleetWorkerExecution>, FleetWorkerInstallError> {
    let mut output = Vec::new();
    for placement in &plan.placement().operations {
        let operation = plan.compiled().operation(placement.operation).ok_or(
            FleetWorkerInstallError::InvalidOperation(placement.operation),
        )?;
        for execution in placement
            .executions
            .iter()
            .filter(|execution| execution.worker == worker)
        {
            let executable_effects = match &operation.primitive {
                ExecutionPrimitive::OrderedComposite { children } => children
                    .iter()
                    .enumerate()
                    .map(|(ordinal, child)| {
                        Ok((
                            Some(
                                u32::try_from(ordinal)
                                    .map_err(|_| FleetWorkerInstallError::SizeOverflow)?,
                            ),
                            &child.primitive,
                            child.effect,
                        ))
                    })
                    .collect::<Result<Vec<_>, FleetWorkerInstallError>>()?,
                primitive => vec![(None, primitive, operation.effect)],
            };
            let executables = executable_effects
                .into_iter()
                .map(|(child_ordinal, primitive, effect)| {
                    Ok(FleetWorkerExecutable {
                        child_ordinal,
                        effects: effect_bindings(
                            plan, worker, operation, execution, effect, storages,
                        )?,
                        statement_host_ingress: statement_host_ingress::project(
                            plan, worker, operation, execution, primitive, storages,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, FleetWorkerInstallError>>()?;
            output.push(FleetWorkerExecution {
                step: placement.during.start,
                operation: operation.id,
                domain: execution.domain,
                during: placement.during,
                executables,
            });
        }
    }
    output.sort_unstable_by_key(|execution| {
        let domain = match execution.domain {
            OperationDomain::Monolithic => (0, 0, 0),
            OperationDomain::Exact(range) => (1, range.start, range.end),
        };
        (execution.step, execution.operation, domain)
    });
    Ok(output)
}

fn effect_bindings(
    plan: &FleetProofPlan,
    worker: WorkerId,
    operation: &crate::compiled_proof::OpNode,
    execution: &FleetOperationExecution,
    effect: EffectContractId,
    storages: &BTreeMap<StorageId, FleetWorkerStorage>,
) -> Result<Vec<FleetEffectBinding>, FleetWorkerInstallError> {
    let effect = plan
        .compiled()
        .effect(effect)
        .ok_or(FleetWorkerInstallError::InvalidOperation(operation.id))?;
    let mut effects = BTreeMap::<EffectBindingId, FleetEffectBinding>::new();
    for access in effect.accesses() {
        if let Some(source) = access.source() {
            let range = validate::projected_range(plan.compiled(), operation, *source, execution)
                .map_err(FleetWorkerInstallError::Plan)?;
            merge_effect(
                &mut effects,
                operation.id,
                source.binding,
                range,
                true,
                resolve_window(plan, worker, range, storages)?,
            )?;
        }
        if let Some(destination) = access.destination() {
            let range =
                validate::projected_range(plan.compiled(), operation, *destination, execution)
                    .map_err(FleetWorkerInstallError::Plan)?;
            merge_effect(
                &mut effects,
                operation.id,
                destination.binding,
                range,
                false,
                resolve_window(plan, worker, range, storages)?,
            )?;
        }
    }
    validate_effect_alignment(plan, operation.id, execution.domain, &effects)?;
    Ok(effects.into_values().collect())
}

fn validate_effect_alignment(
    plan: &FleetProofPlan,
    operation: OpId,
    domain: OperationDomain,
    effects: &BTreeMap<EffectBindingId, FleetEffectBinding>,
) -> Result<(), FleetWorkerInstallError> {
    let required_alignment = execution_alignment(plan, operation, domain, effects)?;
    for effect in effects.values() {
        let absolute = effect
            .window
            .slab_offset_bytes
            .checked_add(effect.window.offset_bytes)
            .ok_or(FleetWorkerInstallError::SizeOverflow)?;
        let alignment = required_alignment
            .get(&effect.binding)
            .copied()
            .unwrap_or(core::mem::align_of::<u32>());
        if absolute % alignment != 0 {
            return Err(FleetWorkerInstallError::MisalignedEffect {
                operation,
                binding: effect.binding,
            });
        }
    }
    Ok(())
}

fn merge_effect(
    effects: &mut BTreeMap<EffectBindingId, FleetEffectBinding>,
    operation: OpId,
    binding: EffectBindingId,
    range: ValueRange,
    source: bool,
    window: FleetInstallWindow,
) -> Result<(), FleetWorkerInstallError> {
    let effect = effects.entry(binding).or_insert(FleetEffectBinding {
        binding,
        source: None,
        destination: None,
        window,
    });
    let slot = if source {
        &mut effect.source
    } else {
        &mut effect.destination
    };
    if effect.window.storage != window.storage
        || effect.window.offset_bytes != window.offset_bytes
        || effect.window.slab_offset_bytes != window.slab_offset_bytes
        || slot.replace(range).is_some()
    {
        return Err(FleetWorkerInstallError::AmbiguousEffect { operation, binding });
    }
    effect.window.bytes = effect.window.bytes.max(window.bytes);
    Ok(())
}

fn resolve_window(
    plan: &FleetProofPlan,
    worker: WorkerId,
    target: ValueRange,
    storages: &BTreeMap<StorageId, FleetWorkerStorage>,
) -> Result<FleetInstallWindow, FleetWorkerInstallError> {
    let mut matches = Vec::new();
    for storage in storages.values() {
        if let Some(offset_bytes) = affine_offset(plan, storage, target)? {
            let bytes = target
                .elements
                .len()
                .checked_mul(core::mem::size_of::<u32>())
                .ok_or(FleetWorkerInstallError::SizeOverflow)?;
            matches.push(FleetInstallWindow {
                storage: storage.storage,
                offset_bytes,
                slab_offset_bytes: storage.slab_offset_bytes,
                bytes,
            });
        }
    }
    match matches.as_slice() {
        [window] => Ok(*window),
        [] => Err(FleetWorkerInstallError::MissingEffectWindow {
            worker,
            value: target,
        }),
        _ => Err(FleetWorkerInstallError::AmbiguousEffectWindow {
            worker,
            value: target,
        }),
    }
}

fn affine_offset(
    plan: &FleetProofPlan,
    storage: &FleetWorkerStorage,
    target: ValueRange,
) -> Result<Option<usize>, FleetWorkerInstallError> {
    let mut windows = plan
        .placement()
        .storage_bindings
        .iter()
        .filter(|binding| {
            binding.storage == storage.storage
                && binding.value.version == target.version
                && binding.value.elements.overlaps(target.elements)
        })
        .map(|binding| {
            let start = binding.value.elements.start.max(target.elements.start);
            let end = binding.value.elements.end.min(target.elements.end);
            let skipped = start
                .checked_sub(binding.value.elements.start)
                .and_then(|elements| elements.checked_mul(core::mem::size_of::<u32>()))
                .ok_or(FleetWorkerInstallError::SizeOverflow)?;
            Ok((
                ElementRange { start, end },
                binding
                    .offset_bytes
                    .checked_add(skipped)
                    .ok_or(FleetWorkerInstallError::SizeOverflow)?,
            ))
        })
        .collect::<Result<Vec<_>, FleetWorkerInstallError>>()?;
    if windows.is_empty() {
        return Ok(None);
    }
    windows.sort_unstable_by_key(|(elements, offset)| (elements.start, elements.end, *offset));

    let mut cursor = target.elements.start;
    let mut base_offset = None;
    for (elements, offset) in windows {
        let relative = elements
            .start
            .checked_sub(target.elements.start)
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<u32>()))
            .ok_or(FleetWorkerInstallError::SizeOverflow)?;
        let base = match base_offset {
            Some(base) => base,
            None => {
                let base = offset
                    .checked_sub(relative)
                    .ok_or(FleetWorkerInstallError::InvalidStorage(storage.storage))?;
                base_offset = Some(base);
                base
            }
        };
        let expected = base
            .checked_add(relative)
            .ok_or(FleetWorkerInstallError::SizeOverflow)?;
        let end = offset
            .checked_add(
                elements
                    .len()
                    .checked_mul(core::mem::size_of::<u32>())
                    .ok_or(FleetWorkerInstallError::SizeOverflow)?,
            )
            .ok_or(FleetWorkerInstallError::SizeOverflow)?;
        if elements.start != cursor || offset != expected || end > storage.bytes {
            return Err(FleetWorkerInstallError::InvalidStorage(storage.storage));
        }
        cursor = elements.end;
    }
    if cursor != target.elements.end {
        return Err(FleetWorkerInstallError::InvalidStorage(storage.storage));
    }
    Ok(base_offset)
}

fn execution_alignment(
    plan: &FleetProofPlan,
    operation: OpId,
    domain: OperationDomain,
    effects: &BTreeMap<EffectBindingId, FleetEffectBinding>,
) -> Result<BTreeMap<EffectBindingId, usize>, FleetWorkerInstallError> {
    let operation = plan
        .compiled()
        .operation(operation)
        .ok_or(FleetWorkerInstallError::InvalidOperation(operation))?;
    let partition = plan
        .compiled()
        .partitions()
        .iter()
        .find(|partition| partition.id() == operation.partition)
        .ok_or(FleetWorkerInstallError::InvalidOperation(operation.id))?;
    let mut output = BTreeMap::new();
    for (&binding, effect) in effects {
        let value = effect
            .source
            .or(effect.destination)
            .and_then(|range| plan.compiled().value(range.version))
            .ok_or(FleetWorkerInstallError::InvalidOperation(operation.id))?;
        let alignment = match (partition.kind(), domain) {
            (PartitionAuthorityKind::Monolithic, OperationDomain::Monolithic) => value.alignment,
            (PartitionAuthorityKind::Exact(authority), OperationDomain::Exact(_)) => {
                match authority
                    .projections()
                    .iter()
                    .find(|projection| projection.binding() == binding)
                    .ok_or(FleetWorkerInstallError::InvalidOperation(operation.id))?
                {
                    PartitionEffectProjection::ReplicatedRead { .. } => value.alignment,
                    PartitionEffectProjection::ContiguousAxisSlice { .. } => {
                        authority.alignment_bytes()
                    }
                }
            }
            _ => return Err(FleetWorkerInstallError::InvalidOperation(operation.id)),
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(FleetWorkerInstallError::InvalidOperation(operation.id));
        }
        output.insert(binding, alignment);
    }
    Ok(output)
}

fn coordinator_install(
    plan: &FleetProofPlan,
    storages: &BTreeMap<StorageId, FleetWorkerStorage>,
) -> Result<FleetCoordinatorInstall, FleetWorkerInstallError> {
    let worker = plan.placement().topology.coordinator;
    let storage = storages
        .get(&plan.placement().output_storage)
        .ok_or(FleetWorkerInstallError::InvalidCoordinator)?;
    let mut transcript = plan
        .compiled()
        .transcript_inputs()
        .iter()
        .map(|binding| {
            transcript_install(
                plan,
                worker,
                storages,
                FleetTranscriptBinding::Input(binding.id),
                ValueRange {
                    version: binding.value,
                    elements: binding.elements,
                },
            )
        })
        .chain(plan.compiled().transcript_outputs().iter().map(|binding| {
            transcript_install(
                plan,
                worker,
                storages,
                FleetTranscriptBinding::Output(binding.id),
                ValueRange {
                    version: binding.value,
                    elements: binding.elements,
                },
            )
        }))
        .collect::<Result<Vec<_>, FleetWorkerInstallError>>()?;
    transcript.sort_unstable_by_key(|binding| binding.binding);
    Ok(FleetCoordinatorInstall {
        transcript_barriers: plan.barriers().to_vec(),
        transcript,
        output: FleetInstallWindow {
            storage: storage.storage,
            offset_bytes: 0,
            slab_offset_bytes: storage.slab_offset_bytes,
            bytes: storage.bytes,
        },
    })
}

fn transcript_install(
    plan: &FleetProofPlan,
    worker: WorkerId,
    storages: &BTreeMap<StorageId, FleetWorkerStorage>,
    binding: FleetTranscriptBinding,
    value: ValueRange,
) -> Result<FleetTranscriptInstall, FleetWorkerInstallError> {
    let (barrier_ordinal, release_step) = transcript_release(
        plan,
        value,
        matches!(binding, FleetTranscriptBinding::Output(_)),
    )?;
    Ok(FleetTranscriptInstall {
        binding,
        value,
        barrier_ordinal,
        release_step,
        window: resolve_window(plan, worker, value, storages)?,
    })
}

fn transcript_release(
    plan: &FleetProofPlan,
    value: ValueRange,
    produced: bool,
) -> Result<(u32, ScheduleStep), FleetWorkerInstallError> {
    let mut matched = None;
    for segment in plan.compiled().transcript_segments() {
        let values = if produced {
            &segment.produced
        } else {
            &segment.consumed
        };
        for _ in values.iter().filter(|candidate| **candidate == value) {
            if matched.replace(segment.segment).is_some() {
                return Err(FleetWorkerInstallError::InvalidCoordinator);
            }
        }
    }
    let segment = matched.ok_or(FleetWorkerInstallError::InvalidCoordinator)?;
    let barrier = plan
        .barriers()
        .iter()
        .find(|barrier| barrier.segment == segment)
        .ok_or(FleetWorkerInstallError::InvalidCoordinator)?;
    Ok((barrier.ordinal, barrier.release_step))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, FleetWorkerInstallError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(FleetWorkerInstallError::SizeOverflow)
}
