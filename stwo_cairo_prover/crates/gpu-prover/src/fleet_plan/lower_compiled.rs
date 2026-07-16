//! One-way admission from sealed proof semantics into physical fleet placement.

use std::sync::Arc;

use super::*;
use crate::compiled_proof::{CompiledProof, InPlaceAliasId, OpId, ValueLayout, ValueRange};
use crate::fleet_pow::FleetPowSchedule;
use crate::fleet_spill::SpillPlan;
use crate::shape_executable::ShapeExecutableIdentity;
use crate::transcript_plan::{CairoBlake2sTranscriptPlan, TranscriptPlanError};

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
    pub operation: OpId,
    pub worker: WorkerId,
    pub during: ScheduleRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetOwnerPlacement {
    pub value: ValueRange,
    pub worker: WorkerId,
    /// Allocation lifetime. Semantic readiness is derived from `CompiledProof`.
    pub live: ScheduleRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetReplicaPlacement {
    pub id: ReplicaId,
    pub value: ValueRange,
    pub canonical_worker: WorkerId,
    pub worker: WorkerId,
    pub layout: ValueLayout,
    pub origin: ReplicaOrigin,
    pub live: ScheduleRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetTransitionPlacement {
    pub id: LayoutTransitionId,
    pub value: ValueRange,
    pub source_worker: WorkerId,
    pub destination_replica: ReplicaId,
    pub axes: Vec<AxisMap>,
    pub interval: ExecutionInterval,
    pub during: ScheduleRange,
    pub scratch_bytes: usize,
    pub scratch_worker: WorkerId,
    pub route: FleetLinkId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetStoragePlacement {
    pub storage: StorageId,
    pub value: ValueRange,
    pub offset_bytes: usize,
}

/// Physical realization of one alias authority already sealed in an effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InPlaceAliasPlacement {
    pub operation: OpId,
    pub alias: InPlaceAliasId,
    pub storage: StorageId,
    pub offset_bytes: usize,
}

/// Placement-only input. It has no semantic DAG, effect body, value layout,
/// transcript binding or executable identity field.
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
    pub spills: Vec<SpillPlan>,
    pub storages: Vec<StorageDesc>,
    pub storage_bindings: Vec<FleetStoragePlacement>,
    pub in_place_aliases: Vec<InPlaceAliasPlacement>,
    /// One packed canonical proof bundle, copied D2H as a single byte range.
    pub output_storage: StorageId,
}

impl FleetProofPlan {
    pub fn lower_compiled(
        compiled: Arc<CompiledProof>,
        shape: ShapeExecutableIdentity,
        placement: FleetPlacementInput,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, FleetLoweringError> {
        verify_identities(&compiled, &shape, transcript)?;
        Self::assemble(
            compiled,
            shape.canonical_encoding().into(),
            placement,
            transcript,
        )
        .map_err(FleetLoweringError::Plan)
    }

    /// No launch is admitted until a real emitter binds these typed primitives
    /// to installed modules, buffers, graph nodes and hardware receipts.
    pub const fn require_real_sn_runtime(&self) -> Result<(), FleetRuntimeAdmissionError> {
        Err(FleetRuntimeAdmissionError::MissingInstalledRuntime)
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
    if compiled.transcript_encoding() != transcript_encoding
        || shape.transcript_encoding() != transcript_encoding
    {
        return Err(FleetLoweringError::ShapeTranscriptMismatch);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FleetRuntimeAdmissionError {
    MissingInstalledRuntime,
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
    Transcript(TranscriptPlanError),
    Plan(FleetPlanError),
}

impl core::fmt::Display for FleetLoweringError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid compiled-proof fleet placement: {self:?}")
    }
}

impl std::error::Error for FleetLoweringError {}

impl From<TranscriptPlanError> for FleetLoweringError {
    fn from(value: TranscriptPlanError) -> Self {
        Self::Transcript(value)
    }
}
