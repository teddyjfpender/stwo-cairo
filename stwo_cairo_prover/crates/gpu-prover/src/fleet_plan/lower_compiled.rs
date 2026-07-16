//! One-way lowering from sealed proof semantics into fleet placement.
//!
//! The caller chooses where and when work runs. Values, operation edges,
//! effects, layouts, transcript bindings and executable identity come only
//! from [`CompiledProof`] and [`ShapeExecutableIdentity`]. The MVP keeps every
//! operation use and physical placement whole-value; sharding needs a later
//! typed semantic partition authority. Passing this host contract is zero
//! real-SN or MHz evidence.

use std::collections::BTreeMap;

use super::*;
use crate::compiled_proof::{
    CompiledProof, OpId as CompiledOpId, OpNode, ProofStage, ValueDesc as CompiledValueDesc,
    ValueId as CompiledValueId, ValueOrigin as CompiledValueOrigin,
};
use crate::fleet_spill::{
    DmaRing, HostSpillStore, RingSlotId, SpillChunk, SpillChunkId, SpillPlan, SpillTransition,
    SpillTransitionId, SpillTransitionKind, StoreExtentId,
};
use crate::shape_executable::ShapeExecutableIdentity;
use crate::transcript_plan::{CairoBlake2sTranscriptPlan, TranscriptPlanError};

const OP_REFERENCE_DOMAIN: &[u8] = b"stwo-cairo.compiled-operation-reference.v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetPlacementTopology {
    pub gpu_class: ConsumerGpuClass,
    pub module_pack_identity: [u8; 32],
    pub fixed_image_identity: [u8; 32],
    pub coordinator: WorkerId,
    pub workers: Vec<WorkerSpec>,
    pub links: Vec<FleetLink>,
    pub host_numa: Vec<HostNumaCapacity>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetOperationPlacement {
    pub operation: CompiledOpId,
    pub worker: WorkerId,
    pub during: ScheduleRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetOwnerPlacement {
    pub value: CompiledValueId,
    pub worker: WorkerId,
    pub ready_at: ScheduleStep,
    pub live: ScheduleRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetReplicaPlacement {
    pub id: ReplicaId,
    pub value: CompiledValueId,
    pub canonical_worker: WorkerId,
    pub worker: WorkerId,
    pub origin: ReplicaOrigin,
    pub ready_at: ScheduleStep,
    pub live: ScheduleRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetTransitionPlacement {
    pub id: LayoutTransitionId,
    pub value: CompiledValueId,
    pub source_worker: WorkerId,
    pub destination_replica: ReplicaId,
    pub interval: ExecutionInterval,
    pub during: ScheduleRange,
    pub scratch_bytes: usize,
    pub scratch_worker: WorkerId,
    pub route: FleetLinkId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetStoragePlacement {
    pub storage: StorageId,
    pub value: CompiledValueId,
    pub worker: WorkerId,
    pub offset_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InPlaceAliasPlacement {
    pub operation: CompiledOpId,
    pub source: CompiledValueId,
    pub destination: CompiledValueId,
    pub storage: StorageId,
    pub offset_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetSpillChunkPlacement {
    pub id: SpillChunkId,
    pub value: CompiledValueId,
    pub worker: WorkerId,
    pub store_extent: StoreExtentId,
    pub ring_slot: RingSlotId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetSpillTransitionPlacement {
    pub id: SpillTransitionId,
    pub chunk: SpillChunkId,
    pub kind: SpillTransitionKind,
    pub interval: ExecutionInterval,
    pub during: ScheduleRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetSpillPlacement {
    pub store: HostSpillStore,
    pub ring: DmaRing,
    pub chunks: Vec<FleetSpillChunkPlacement>,
    pub transitions: Vec<FleetSpillTransitionPlacement>,
}

/// Placement-only input. It intentionally has no semantic DAG, value layout,
/// transcript binding, effect token or executable-identity field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetPlacementInput {
    pub topology: FleetPlacementTopology,
    pub pow: FleetPowSchedule,
    pub barrier_steps: Vec<ScheduleStep>,
    pub terminal_step: ScheduleStep,
    pub barrier_arrivals: Vec<BarrierArrival>,
    pub operations: Vec<FleetOperationPlacement>,
    pub owners: Vec<FleetOwnerPlacement>,
    pub replicas: Vec<FleetReplicaPlacement>,
    pub transitions: Vec<FleetTransitionPlacement>,
    pub spills: Vec<FleetSpillPlacement>,
    pub storages: Vec<StorageDesc>,
    pub storage_bindings: Vec<FleetStoragePlacement>,
    pub in_place_aliases: Vec<InPlaceAliasPlacement>,
}

impl FleetProofPlan {
    pub fn lower_compiled(
        compiled: &CompiledProof,
        shape: &ShapeExecutableIdentity,
        placement: FleetPlacementInput,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, FleetLoweringError> {
        verify_identities(compiled, shape, transcript)?;
        let operations = operation_placements(compiled, &placement.operations)?;
        let owners = owner_placements(compiled, &placement.owners)?;
        let values = lower_values(compiled);
        let lowered_operations = lower_operations(compiled, transcript, &operations)?;
        let assignments = compiled
            .operations()
            .iter()
            .map(|operation| OperationAssignment {
                operation: OperationId(operation.id.0),
                worker: operations[&operation.id].worker,
            })
            .collect();
        let input = FleetPlanInput {
            topology: lower_topology(placement.topology, shape),
            pow: placement.pow,
            barrier_steps: placement.barrier_steps,
            terminal_step: placement.terminal_step,
            barrier_arrivals: placement.barrier_arrivals,
            transcript_inputs: lower_transcript_inputs(compiled),
            transcript_outputs: lower_transcript_outputs(compiled),
            values,
            operations: lowered_operations,
            assignments,
            owners: lower_owners(compiled, &owners)?,
            replicas: lower_replicas(compiled, placement.replicas)?,
            transitions: lower_transitions(compiled, placement.transitions)?,
            spills: lower_spills(compiled, placement.spills)?,
            storages: placement.storages,
            storage_bindings: lower_storage(compiled, placement.storage_bindings)?,
            in_place_aliases: lower_aliases(compiled, placement.in_place_aliases)?,
        };
        Self::compile_explicit(input, transcript).map_err(FleetLoweringError::Plan)
    }

    /// A lowered contract is not a launchable real-SN program until a typed
    /// emitter binds execution primitives and launch geometry. No fallback
    /// kernel, transcript kernel or assembly kernel is fabricated here.
    pub const fn require_real_sn_runtime(&self) -> Result<(), FleetRuntimeAdmissionError> {
        Err(FleetRuntimeAdmissionError::MissingTypedExecutionPrimitives)
    }
}

fn verify_identities(
    compiled: &CompiledProof,
    shape: &ShapeExecutableIdentity,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), FleetLoweringError> {
    if shape.compiled_proof_encoding() != compiled.identity().canonical_encoding() {
        return Err(FleetLoweringError::ShapeCompiledProofMismatch);
    }
    let transcript_encoding = transcript.canonical_encoding()?;
    if shape.transcript_encoding() != transcript_encoding {
        return Err(FleetLoweringError::ShapeTranscriptMismatch);
    }
    Ok(())
}

fn operation_placements<'a>(
    compiled: &CompiledProof,
    placements: &'a [FleetOperationPlacement],
) -> Result<BTreeMap<CompiledOpId, &'a FleetOperationPlacement>, FleetLoweringError> {
    let mut by_id = BTreeMap::new();
    for placement in placements {
        if compiled
            .operations()
            .get(placement.operation.0 as usize)
            .is_none_or(|operation| operation.id != placement.operation)
        {
            return Err(FleetLoweringError::UnknownOperation(placement.operation));
        }
        if by_id.insert(placement.operation, placement).is_some() {
            return Err(FleetLoweringError::DuplicateOperationPlacement(
                placement.operation,
            ));
        }
    }
    for operation in compiled.operations() {
        if !by_id.contains_key(&operation.id) {
            return Err(FleetLoweringError::MissingOperationPlacement(operation.id));
        }
    }
    Ok(by_id)
}

fn owner_placements<'a>(
    compiled: &CompiledProof,
    placements: &'a [FleetOwnerPlacement],
) -> Result<BTreeMap<CompiledValueId, &'a FleetOwnerPlacement>, FleetLoweringError> {
    let mut by_id = BTreeMap::new();
    for placement in placements {
        compiled_value(compiled, placement.value)?;
        if by_id.insert(placement.value, placement).is_some() {
            return Err(FleetLoweringError::DuplicateOwnerPlacement(placement.value));
        }
    }
    for value in compiled.values() {
        if !by_id.contains_key(&value.id) {
            return Err(FleetLoweringError::MissingOwnerPlacement(value.id));
        }
    }
    Ok(by_id)
}

fn lower_topology(
    topology: FleetPlacementTopology,
    shape: &ShapeExecutableIdentity,
) -> FleetTopology {
    FleetTopology {
        gpu_class: topology.gpu_class,
        module_pack_identity: topology.module_pack_identity,
        fixed_image_identity: topology.fixed_image_identity,
        executable_identity: shape.canonical_encoding().to_vec(),
        coordinator: topology.coordinator,
        workers: topology.workers,
        links: topology.links,
        host_numa: topology.host_numa,
    }
}

fn lower_values(compiled: &CompiledProof) -> Vec<ValueDesc> {
    compiled
        .values()
        .iter()
        .map(|value| ValueDesc {
            id: ValueId(value.id.0),
            layout: value.layout.clone(),
            alignment_bytes: value.alignment,
            origin: match value.origin {
                CompiledValueOrigin::ExternalInput(id) => ValueOrigin::ExternalInput(id.0),
                CompiledValueOrigin::Constant(id) => ValueOrigin::FixedImage(id.0),
                CompiledValueOrigin::OpOutput(_) => ValueOrigin::Operation,
                CompiledValueOrigin::TranscriptOutput(id) => ValueOrigin::TranscriptOutput(id),
            },
        })
        .collect()
}

fn lower_operations(
    compiled: &CompiledProof,
    transcript: &CairoBlake2sTranscriptPlan,
    placements: &BTreeMap<CompiledOpId, &FleetOperationPlacement>,
) -> Result<Vec<OperationDesc>, FleetLoweringError> {
    compiled
        .operations()
        .iter()
        .map(|operation| {
            let placement = placements[&operation.id];
            Ok(OperationDesc {
                id: OperationId(operation.id.0),
                semantic: operation_reference(operation),
                effect_identity: operation.effects,
                interval: lower_stage(operation, transcript)?,
                during: placement.during,
                reads: lower_uses(compiled, &operation.inputs)?,
                writes: lower_uses(compiled, &operation.outputs)?,
            })
        })
        .collect()
}

fn operation_reference(operation: &OpNode) -> Vec<u8> {
    let mut out = Vec::with_capacity(OP_REFERENCE_DOMAIN.len() + 8);
    out.extend_from_slice(OP_REFERENCE_DOMAIN);
    out.extend_from_slice(&operation.semantic_id.0.to_le_bytes());
    out.extend_from_slice(&operation.kernel_id.0.to_le_bytes());
    out
}

fn lower_stage(
    operation: &OpNode,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<ExecutionInterval, FleetLoweringError> {
    match operation.stage {
        ProofStage::BeforeTranscript(segment) => transcript
            .segments()
            .iter()
            .position(|candidate| candidate.segment == segment)
            .ok_or(FleetLoweringError::UnknownStage(operation.id))
            .and_then(|ordinal| {
                u32::try_from(ordinal)
                    .map(ExecutionInterval::BeforeBarrier)
                    .map_err(|_| FleetLoweringError::SizeOverflow)
            }),
        ProofStage::AfterTranscript => Ok(ExecutionInterval::AfterFinalBarrier),
    }
}

fn lower_uses(
    compiled: &CompiledProof,
    ids: &[CompiledValueId],
) -> Result<Vec<ValueUse>, FleetLoweringError> {
    ids.iter()
        .map(|&id| {
            let value = compiled_value(compiled, id)?;
            Ok(ValueUse {
                value: ValueId(id.0),
                elements: whole_range(value)?,
                layout: value.layout.clone(),
            })
        })
        .collect()
}

fn lower_owners(
    compiled: &CompiledProof,
    placements: &BTreeMap<CompiledValueId, &FleetOwnerPlacement>,
) -> Result<Vec<OwnedValueRange>, FleetLoweringError> {
    compiled
        .values()
        .iter()
        .map(|value| {
            let placement = placements[&value.id];
            Ok(OwnedValueRange {
                value: ValueId(value.id.0),
                elements: whole_range(value)?,
                worker: placement.worker,
                producer: match value.origin {
                    CompiledValueOrigin::OpOutput(id) => Some(OperationId(id.0)),
                    _ => None,
                },
                ready_at: placement.ready_at,
                live: placement.live,
            })
        })
        .collect()
}

fn lower_replicas(
    compiled: &CompiledProof,
    placements: Vec<FleetReplicaPlacement>,
) -> Result<Vec<DeclaredReplica>, FleetLoweringError> {
    placements
        .into_iter()
        .map(|placement| {
            let value = compiled_value(compiled, placement.value)?;
            Ok(DeclaredReplica {
                id: placement.id,
                value: ValueId(value.id.0),
                elements: whole_range(value)?,
                canonical_worker: placement.canonical_worker,
                worker: placement.worker,
                layout: value.layout.clone(),
                origin: placement.origin,
                ready_at: placement.ready_at,
                live: placement.live,
            })
        })
        .collect()
}

fn lower_transitions(
    compiled: &CompiledProof,
    placements: Vec<FleetTransitionPlacement>,
) -> Result<Vec<LayoutTransition>, FleetLoweringError> {
    placements
        .into_iter()
        .map(|placement| {
            let value = compiled_value(compiled, placement.value)?;
            Ok(LayoutTransition {
                id: placement.id,
                value: ValueId(value.id.0),
                elements: whole_range(value)?,
                source_worker: placement.source_worker,
                destination_replica: placement.destination_replica,
                source_layout: value.layout.clone(),
                destination_layout: value.layout.clone(),
                axes: value
                    .layout
                    .axes
                    .iter()
                    .map(|axis| AxisMap {
                        source: axis.tag,
                        destination: axis.tag,
                    })
                    .collect(),
                interval: placement.interval,
                during: placement.during,
                bytes: value.layout.logical_bytes()?,
                scratch_bytes: placement.scratch_bytes,
                scratch_worker: placement.scratch_worker,
                route: placement.route,
            })
        })
        .collect()
}

fn lower_spills(
    compiled: &CompiledProof,
    placements: Vec<FleetSpillPlacement>,
) -> Result<Vec<SpillPlan>, FleetLoweringError> {
    placements
        .into_iter()
        .map(|placement| {
            let mut bytes_by_chunk = BTreeMap::new();
            let chunks = placement
                .chunks
                .into_iter()
                .map(|chunk| {
                    let value = compiled_value(compiled, chunk.value)?;
                    let bytes = value.layout.logical_bytes()?;
                    if bytes_by_chunk.insert(chunk.id, bytes).is_some() {
                        return Err(FleetLoweringError::DuplicateSpillChunk(chunk.id));
                    }
                    Ok(SpillChunk {
                        id: chunk.id,
                        value: ValueId(value.id.0),
                        elements: whole_range(value)?,
                        worker: chunk.worker,
                        bytes,
                        store_extent: chunk.store_extent,
                        ring_slot: chunk.ring_slot,
                    })
                })
                .collect::<Result<Vec<_>, FleetLoweringError>>()?;
            let transitions = placement
                .transitions
                .into_iter()
                .map(|transition| {
                    let bytes = bytes_by_chunk
                        .get(&transition.chunk)
                        .copied()
                        .ok_or(FleetLoweringError::UnknownSpillChunk(transition.chunk))?;
                    Ok(SpillTransition {
                        id: transition.id,
                        chunk: transition.chunk,
                        kind: transition.kind,
                        interval: transition.interval,
                        during: transition.during,
                        bytes,
                    })
                })
                .collect::<Result<Vec<_>, FleetLoweringError>>()?;
            Ok(SpillPlan {
                store: placement.store,
                ring: placement.ring,
                chunks,
                transitions,
            })
        })
        .collect()
}

fn lower_storage(
    compiled: &CompiledProof,
    placements: Vec<FleetStoragePlacement>,
) -> Result<Vec<StorageBinding>, FleetLoweringError> {
    placements
        .into_iter()
        .map(|placement| {
            let value = compiled_value(compiled, placement.value)?;
            Ok(StorageBinding {
                storage: placement.storage,
                value: ValueId(value.id.0),
                elements: whole_range(value)?,
                worker: placement.worker,
                offset_bytes: placement.offset_bytes,
                bytes: value.layout.logical_bytes()?,
            })
        })
        .collect()
}

fn lower_aliases(
    compiled: &CompiledProof,
    placements: Vec<InPlaceAliasPlacement>,
) -> Result<Vec<InPlaceAlias>, FleetLoweringError> {
    placements
        .into_iter()
        .map(|placement| {
            let operation = compiled
                .operations()
                .get(placement.operation.0 as usize)
                .filter(|operation| operation.id == placement.operation)
                .ok_or(FleetLoweringError::UnknownOperation(placement.operation))?;
            let source = compiled_value(compiled, placement.source)?;
            let destination = compiled_value(compiled, placement.destination)?;
            let bytes = source.layout.logical_bytes()?;
            if destination.layout.logical_bytes()? != bytes {
                return Err(FleetLoweringError::AliasSizeMismatch {
                    source: source.id,
                    destination: destination.id,
                });
            }
            Ok(InPlaceAlias {
                operation: OperationId(operation.id.0),
                effect: operation.effects,
                source: ValueId(source.id.0),
                source_elements: whole_range(source)?,
                destination: ValueId(destination.id.0),
                destination_elements: whole_range(destination)?,
                storage: placement.storage,
                offset_bytes: placement.offset_bytes,
                bytes,
            })
        })
        .collect()
}

fn lower_transcript_inputs(compiled: &CompiledProof) -> Vec<TranscriptInputValueBinding> {
    compiled
        .input()
        .transcript_inputs
        .iter()
        .map(|binding| TranscriptInputValueBinding {
            id: binding.id,
            value: ValueId(binding.value.0),
            elements: ElementRange {
                start: binding.value_words.start,
                end: binding.value_words.end,
            },
        })
        .collect()
}

fn lower_transcript_outputs(compiled: &CompiledProof) -> Vec<TranscriptOutputValueBinding> {
    compiled
        .input()
        .transcript_outputs
        .iter()
        .map(|binding| TranscriptOutputValueBinding {
            id: binding.id,
            value: ValueId(binding.value.0),
            elements: ElementRange {
                start: binding.value_words.start,
                end: binding.value_words.end,
            },
        })
        .collect()
}

fn compiled_value(
    compiled: &CompiledProof,
    id: CompiledValueId,
) -> Result<&CompiledValueDesc, FleetLoweringError> {
    compiled
        .values()
        .get(id.0 as usize)
        .filter(|value| value.id == id)
        .ok_or(FleetLoweringError::UnknownValue(id))
}

fn whole_range(value: &CompiledValueDesc) -> Result<ElementRange, FleetLoweringError> {
    let end = value.layout.element_count()?;
    ElementRange::new(0, end).ok_or(FleetLoweringError::UnknownValue(value.id))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FleetRuntimeAdmissionError {
    MissingTypedExecutionPrimitives,
}

impl core::fmt::Display for FleetRuntimeAdmissionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fleet runtime admission failed: {self:?}")
    }
}

impl std::error::Error for FleetRuntimeAdmissionError {}

#[derive(Debug)]
pub enum FleetLoweringError {
    ShapeCompiledProofMismatch,
    ShapeTranscriptMismatch,
    UnknownOperation(CompiledOpId),
    DuplicateOperationPlacement(CompiledOpId),
    MissingOperationPlacement(CompiledOpId),
    UnknownValue(CompiledValueId),
    DuplicateOwnerPlacement(CompiledValueId),
    MissingOwnerPlacement(CompiledValueId),
    DuplicateSpillChunk(SpillChunkId),
    UnknownSpillChunk(SpillChunkId),
    AliasSizeMismatch {
        source: CompiledValueId,
        destination: CompiledValueId,
    },
    UnknownStage(CompiledOpId),
    SizeOverflow,
    Transcript(TranscriptPlanError),
    Plan(FleetPlanError),
}

impl core::fmt::Display for FleetLoweringError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid compiled-proof fleet placement: {self:?}")
    }
}

impl std::error::Error for FleetLoweringError {}

impl From<FleetPlanError> for FleetLoweringError {
    fn from(value: FleetPlanError) -> Self {
        Self::Plan(value)
    }
}

impl From<TranscriptPlanError> for FleetLoweringError {
    fn from(value: TranscriptPlanError) -> Self {
        Self::Transcript(value)
    }
}

#[cfg(test)]
mod tests;
