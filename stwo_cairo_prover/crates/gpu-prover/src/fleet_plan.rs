//! Address-free contract for one proof cooperatively executed by a GPU fleet.
//!
//! [`CompiledProof`] is the sole semantic authority. This module owns only
//! worker placement, schedule, storage, transfer and spill facts. Passing this
//! host validator is not runtime admission or benchmark evidence.

use core::ops::Range;
use std::sync::Arc;

use crate::compiled_proof::{CompiledProof, InPlaceAliasId, OpId, ValueVersion};
pub use crate::compiled_proof::{ElementRange, ElementType, LayoutAxis, ValueLayout};
use crate::fleet_pow::{FleetPowError, FleetPowSite, PowRankReceipt};
use crate::fleet_spill::{SpillChunkId, SpillPlanError};
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptSegment, TranscriptSegmentPlan,
};

mod identity;
mod lower_compiled;
mod storage;
mod validate;

#[cfg(test)]
mod tests;

pub use lower_compiled::{
    FleetLoweringError, FleetOperationPlacement, FleetOwnerPlacement, FleetPlacementInput,
    FleetPlacementTopology, FleetReplicaPlacement, FleetRuntimeAdmissionError,
    FleetStoragePlacement, FleetTransitionPlacement, InPlaceAliasPlacement,
};
pub use storage::{StorageDesc, StorageId};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct WorkerId(pub u16);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ReplicaId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LayoutTransitionId(pub u32);

/// Deterministic event ordinal in the address-free schedule.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ScheduleStep(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduleRange {
    pub start: ScheduleStep,
    pub end: ScheduleStep,
}

impl ScheduleRange {
    pub const fn new(start: ScheduleStep, end: ScheduleStep) -> Option<Self> {
        if start.0 < end.0 {
            Some(Self { start, end })
        } else {
            None
        }
    }

    pub const fn is_valid(self) -> bool {
        self.start.0 < self.end.0
    }

    pub const fn contains(self, other: Self) -> bool {
        self.start.0 <= other.start.0 && other.end.0 <= self.end.0
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.start.0 < other.end.0 && other.start.0 < self.end.0
    }
}

/// Causal transcript interval containing physical work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionInterval {
    BeforeBarrier(u32),
    AfterFinalBarrier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplicaOrigin {
    InstalledFixed,
    Transition(LayoutTransitionId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AxisMap {
    pub source: u16,
    pub destination: u16,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FleetLinkId(pub u16);

/// Directed P2P route. Host bounce is represented only by an explicit spill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetLink {
    pub id: FleetLinkId,
    pub source: WorkerId,
    pub destination: WorkerId,
    pub max_transfer_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsumerGpuClass {
    Rtx3090Sm86,
    Rtx4090Sm89,
    Rtx5090Sm120,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkerSpec {
    pub id: WorkerId,
    pub capacity_bytes: usize,
    pub exchange_reserve_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostNumaCapacity {
    pub numa_node: u32,
    pub store_capacity_bytes: usize,
    pub memlock_limit_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptBarrier {
    pub ordinal: u32,
    pub coordinator: WorkerId,
    pub segment: CairoTranscriptSegment,
    pub operation_range: Range<usize>,
    pub starts_after: Option<crate::transcript_plan::CairoTranscriptBoundary>,
    pub ends_at: crate::transcript_plan::CairoTranscriptBoundary,
    pub release_step: ScheduleStep,
}

impl TranscriptBarrier {
    fn from_segment(
        ordinal: usize,
        coordinator: WorkerId,
        plan: &TranscriptSegmentPlan,
        release_step: ScheduleStep,
    ) -> Result<Self, FleetPlanError> {
        Ok(Self {
            ordinal: u32::try_from(ordinal).map_err(|_| FleetPlanError::SizeOverflow)?,
            coordinator,
            segment: plan.segment,
            operation_range: plan.operation_range.clone(),
            starts_after: plan.starts_after,
            ends_at: plan.ends_at,
            release_step,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BarrierArrival {
    pub barrier_ordinal: u32,
    pub worker: WorkerId,
    pub ready_step: ScheduleStep,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetWorkerPlan {
    pub worker: WorkerId,
    pub peak_resident_bytes: usize,
    pub capacity_bytes: usize,
}

/// Validated immutable host contract. Runtime installation remains fail closed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetProofPlan {
    compiled: Arc<CompiledProof>,
    shape_encoding: Box<[u8]>,
    placement: FleetPlacementInput,
    barriers: Vec<TranscriptBarrier>,
    workers: Vec<FleetWorkerPlan>,
    identity: [u8; 32],
}

impl FleetProofPlan {
    pub(super) fn assemble(
        compiled: Arc<CompiledProof>,
        shape_encoding: Box<[u8]>,
        mut placement: FleetPlacementInput,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, FleetPlanError> {
        canonicalize(&mut placement);
        if placement.barrier_steps.len() != transcript.segments().len() {
            return Err(FleetPlanError::TranscriptMismatch);
        }
        let barriers = transcript
            .segments()
            .iter()
            .zip(&placement.barrier_steps)
            .enumerate()
            .map(|(ordinal, (segment, &release))| {
                TranscriptBarrier::from_segment(
                    ordinal,
                    placement.topology.coordinator,
                    segment,
                    release,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut plan = Self {
            compiled,
            shape_encoding,
            placement,
            barriers,
            workers: Vec::new(),
            identity: [0; 32],
        };
        plan.workers = validate::validate_and_measure(&plan, transcript)?;
        plan.identity = identity::compute(&plan)?;
        Ok(plan)
    }

    pub fn validate(&self, transcript: &CairoBlake2sTranscriptPlan) -> Result<(), FleetPlanError> {
        let measured = validate::validate_and_measure(self, transcript)?;
        if measured != self.workers || identity::compute(self)? != self.identity {
            return Err(FleetPlanError::IdentityMismatch);
        }
        Ok(())
    }

    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FleetPlanError> {
        identity::encode(self)
    }

    pub fn compiled(&self) -> &CompiledProof {
        &self.compiled
    }

    pub fn placement(&self) -> &FleetPlacementInput {
        &self.placement
    }

    pub fn shape_encoding(&self) -> &[u8] {
        &self.shape_encoding
    }

    pub fn barriers(&self) -> &[TranscriptBarrier] {
        &self.barriers
    }

    pub const fn terminal_step(&self) -> ScheduleStep {
        self.placement.terminal_step
    }

    pub fn fence_count(&self) -> Result<u32, FleetPlanError> {
        let count = self
            .barriers
            .len()
            .checked_add(1)
            .ok_or(FleetPlanError::SizeOverflow)?;
        u32::try_from(count).map_err(|_| FleetPlanError::SizeOverflow)
    }

    pub fn workers(&self) -> &[FleetWorkerPlan] {
        &self.workers
    }

    pub fn verify_pow_winner(
        &self,
        site: FleetPowSite,
        proof_generation: u64,
        receipts: &[PowRankReceipt],
        is_valid_nonce: impl Fn(u64) -> bool,
    ) -> Result<u64, FleetPowError> {
        self.placement.pow.plan(site).verify_winner(
            site,
            self.placement.topology.workers.len(),
            self.identity,
            proof_generation,
            receipts,
            is_valid_nonce,
        )
    }
}

fn canonicalize(input: &mut FleetPlacementInput) {
    input
        .topology
        .workers
        .sort_unstable_by_key(|worker| worker.id);
    input.topology.links.sort_unstable_by_key(|link| link.id);
    input
        .topology
        .host_numa
        .sort_unstable_by_key(|capacity| capacity.numa_node);
    input
        .barrier_arrivals
        .sort_unstable_by_key(|arrival| (arrival.barrier_ordinal, arrival.worker));
    input
        .operations
        .sort_unstable_by_key(|operation| operation.operation);
    input.owners.sort_unstable_by_key(|owner| {
        (
            owner.value.version,
            owner.value.elements.start,
            owner.value.elements.end,
            owner.worker,
        )
    });
    input.replicas.sort_unstable_by_key(|replica| replica.id);
    input
        .transitions
        .sort_unstable_by_key(|transition| transition.id);
    for transition in &mut input.transitions {
        transition
            .axes
            .sort_unstable_by_key(|axis| (axis.source, axis.destination));
    }
    input
        .spills
        .sort_unstable_by_key(|spill| spill.store.worker);
    for spill in &mut input.spills {
        spill.canonicalize();
    }
    input.storages.sort_unstable_by_key(|storage| storage.id);
    input.storage_bindings.sort_unstable_by_key(|binding| {
        (
            binding.storage,
            binding.offset_bytes,
            binding.value.version,
            binding.value.elements.start,
            binding.value.elements.end,
        )
    });
    input.in_place_aliases.sort_unstable_by_key(|alias| {
        (
            alias.operation,
            alias.alias,
            alias.storage,
            alias.offset_bytes,
        )
    });
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FleetPlanError {
    EmptyTopology,
    EmptyProgram,
    InvalidCoordinator,
    NonDenseWorkers,
    NonDenseIds,
    InvalidHomogeneousTopology,
    InvalidLink(FleetLinkId),
    UnknownNuma(u32),
    HostCapacityExceeded(u32),
    DuplicateOperation(OpId),
    MissingOperation(OpId),
    UnknownWorker(WorkerId),
    UnknownValue(ValueVersion),
    UnknownOperation(OpId),
    InvalidOperation(OpId),
    InvalidRange(ValueVersion),
    InvalidSchedule,
    InvalidSegment(OpId),
    OwnershipCoverage(ValueVersion),
    InvalidProducer(ValueVersion),
    UndeclaredRead {
        operation: OpId,
        value: ValueVersion,
    },
    InvalidReplica(ReplicaId),
    InvalidTransition(LayoutTransitionId),
    InvalidStorage(StorageId),
    InvalidStorageBinding {
        value: ValueVersion,
        storage: StorageId,
    },
    StorageCoverage(ValueVersion),
    IllegalStorageReuse(StorageId),
    InvalidInPlaceAlias {
        operation: OpId,
        alias: InPlaceAliasId,
    },
    InvalidProofOutput(StorageId),
    TranscriptValueCausality(ValueVersion),
    CapacityExceeded {
        worker: WorkerId,
        required: usize,
        capacity: usize,
    },
    TranscriptMismatch,
    SpillValue(SpillChunkId),
    InvalidVmmReclaim(StorageId),
    UnmappedStorageAccess(StorageId),
    Spill(SpillPlanError),
    Pow(FleetPowError),
    SizeOverflow,
    IdentityMismatch,
}

impl core::fmt::Display for FleetPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid fleet proof plan: {self:?}")
    }
}

impl std::error::Error for FleetPlanError {}

impl From<SpillPlanError> for FleetPlanError {
    fn from(value: SpillPlanError) -> Self {
        Self::Spill(value)
    }
}

impl From<FleetPowError> for FleetPlanError {
    fn from(value: FleetPowError) -> Self {
        Self::Pow(value)
    }
}
