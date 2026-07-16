//! Pure, address-free contract for one proof cooperatively executed by a GPU fleet.
//!
//! This is deliberately a validation boundary, not a runtime. Callers must
//! provide typed values and producer/consumer edges explicitly; today's arena
//! metadata is too coarse to infer them soundly. No constructor accepts a
//! `ProofArenaPlan`, and no method allocates a GPU, opens IPC, or installs work.

use core::ops::Range;

use crate::fleet_pow::{FleetPowError, FleetPowSchedule, FleetPowSite, PowRankReceipt};
use crate::fleet_spill::{SpillPlan, SpillPlanError};
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptSegment, TranscriptSegmentPlan,
};

mod identity;
mod validate;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct WorkerId(pub u16);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct OperationId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ReplicaId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LayoutTransitionId(pub u32);

/// A deterministic event ordinal in the address-free schedule.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ScheduleStep(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduleRange {
    pub start: ScheduleStep,
    pub end: ScheduleStep,
}

impl ScheduleRange {
    pub fn new(start: ScheduleStep, end: ScheduleStep) -> Option<Self> {
        (start < end).then_some(Self { start, end })
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ElementRange {
    pub start: usize,
    pub end: usize,
}

impl ElementRange {
    pub fn new(start: usize, end: usize) -> Option<Self> {
        (start < end).then_some(Self { start, end })
    }

    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }

    pub const fn contains(self, other: Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ElementType {
    /// Stable semantic ABI tag chosen by the future ProgramImage emitter.
    pub tag: u32,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayoutAxis {
    /// Stable semantic axis tag; order in `axes` is the physical nesting order.
    pub tag: u16,
    pub extent: usize,
    pub stride_bytes: usize,
}

/// A dense, non-aliasing physical layout. Arbitrary padded/aliased views are
/// intentionally rejected by the MVP contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueLayout {
    pub element: ElementType,
    pub axes: Vec<LayoutAxis>,
}

impl ValueLayout {
    pub fn element_count(&self) -> Result<usize, FleetPlanError> {
        self.axes.iter().try_fold(1usize, |count, axis| {
            count
                .checked_mul(axis.extent)
                .ok_or(FleetPlanError::SizeOverflow)
        })
    }

    pub fn logical_bytes(&self) -> Result<usize, FleetPlanError> {
        self.element_count()?
            .checked_mul(self.element.bytes)
            .ok_or(FleetPlanError::SizeOverflow)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueDesc {
    pub id: ValueId,
    pub layout: ValueLayout,
    pub origin: ValueOrigin,
}

/// Semantic source of one logical value version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueOrigin {
    /// Proof-varying input slot in the ProgramImage input codec.
    ExternalInput(u32),
    /// Immutable constant slot in the content-addressed FixedImage.
    FixedImage(u32),
    /// Value produced by one or more explicitly declared operation shards.
    Operation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueUse {
    pub value: ValueId,
    pub elements: ElementRange,
    pub layout: ValueLayout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationDesc {
    pub id: OperationId,
    /// Canonical address-free operation encoding emitted by the future ProgramImage compiler.
    pub semantic: Vec<u8>,
    pub interval: ExecutionInterval,
    pub during: ScheduleRange,
    pub reads: Vec<ValueUse>,
    pub writes: Vec<ValueUse>,
}

/// Causal interval containing an operation. The final interval is explicit so
/// query decommitment and proof assembly cannot be hidden inside the last
/// Fiat-Shamir segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionInterval {
    BeforeBarrier(u32),
    AfterFinalBarrier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperationAssignment {
    pub operation: OperationId,
    pub worker: WorkerId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedValueRange {
    pub value: ValueId,
    pub elements: ElementRange,
    pub worker: WorkerId,
    pub producer: Option<OperationId>,
    pub ready_at: ScheduleStep,
    pub live: ScheduleRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredReplica {
    pub id: ReplicaId,
    pub value: ValueId,
    pub elements: ElementRange,
    pub canonical_worker: WorkerId,
    pub worker: WorkerId,
    pub layout: ValueLayout,
    pub origin: ReplicaOrigin,
    pub ready_at: ScheduleStep,
    pub live: ScheduleRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplicaOrigin {
    /// FixedImage bytes installed independently under the topology identity.
    InstalledFixed,
    /// Proof-varying bytes materialized by this declared transfer/transpose.
    Transition(LayoutTransitionId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AxisMap {
    pub source: u16,
    pub destination: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayoutTransition {
    pub id: LayoutTransitionId,
    pub value: ValueId,
    pub elements: ElementRange,
    pub source_worker: WorkerId,
    pub destination_replica: ReplicaId,
    pub source_layout: ValueLayout,
    pub destination_layout: ValueLayout,
    pub axes: Vec<AxisMap>,
    pub interval: ExecutionInterval,
    pub during: ScheduleRange,
    pub bytes: usize,
    pub scratch_bytes: usize,
    pub scratch_worker: WorkerId,
    pub route: FleetLinkId,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FleetLinkId(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Declared directed P2P link. Host-bounce transport is intentionally absent
/// until it owns bounded pinned slots and exact NUMA copy lifetimes.
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
pub struct FleetTopology {
    /// Declared logical identity only. Hardware probes must independently
    /// attest every rank and directed link before a runtime may install it.
    /// One exact card/SM class for every rank; mixed fleets are not admitted.
    pub gpu_class: ConsumerGpuClass,
    /// Hash of the complete ordered module, ABI and initializer pack on every rank.
    pub module_pack_identity: [u8; 32],
    /// Content identity of the immutable twiddle/table/preprocessed FixedImage.
    pub fixed_image_identity: [u8; 32],
    /// Full canonical address-free ShapeExecutable identity, not a short digest.
    pub executable_identity: Vec<u8>,
    pub coordinator: WorkerId,
    pub workers: Vec<WorkerSpec>,
    pub links: Vec<FleetLink>,
    pub host_numa: Vec<HostNumaCapacity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptBarrier {
    pub ordinal: u32,
    pub coordinator: WorkerId,
    pub segment: CairoTranscriptSegment,
    pub operation_range: Range<usize>,
    pub starts_after: Option<crate::transcript_plan::CairoTranscriptBoundary>,
    pub ends_at: crate::transcript_plan::CairoTranscriptBoundary,
    /// Coordinator broadcast completion for this exact transcript segment.
    pub release_step: ScheduleStep,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BarrierArrival {
    pub barrier_ordinal: u32,
    pub worker: WorkerId,
    /// Rank completion receipt consumed before the coordinator release step.
    pub ready_step: ScheduleStep,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetPlanInput {
    pub topology: FleetTopology,
    pub pow: FleetPowSchedule,
    /// One coordinator-broadcast completion step per canonical transcript segment.
    pub barrier_steps: Vec<ScheduleStep>,
    /// Wait-all completion fence after query decommitment and proof assembly.
    pub terminal_step: ScheduleStep,
    /// Includes one arrival per worker for every transcript barrier and the terminal fence.
    pub barrier_arrivals: Vec<BarrierArrival>,
    pub values: Vec<ValueDesc>,
    pub operations: Vec<OperationDesc>,
    pub assignments: Vec<OperationAssignment>,
    pub owners: Vec<OwnedValueRange>,
    pub replicas: Vec<DeclaredReplica>,
    pub transitions: Vec<LayoutTransition>,
    pub spills: Vec<SpillPlan>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetWorkerPlan {
    pub worker: WorkerId,
    pub peak_live_bytes: usize,
    pub capacity_bytes: usize,
}

/// Validated host contract only. Absence of a runtime/install method is a
/// deliberate admission fence until a real typed ProgramImage emitter exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetProofPlan {
    input: FleetPlanInput,
    schedule_key: u64,
    transcript_encoding: Vec<u8>,
    barriers: Vec<TranscriptBarrier>,
    workers: Vec<FleetWorkerPlan>,
    identity: [u8; 32],
}

impl FleetProofPlan {
    pub fn compile_explicit(
        mut input: FleetPlanInput,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, FleetPlanError> {
        canonicalize(&mut input);
        if input.barrier_steps.len() != transcript.segments().len() {
            return Err(FleetPlanError::TranscriptMismatch);
        }
        let barriers = transcript
            .segments()
            .iter()
            .zip(&input.barrier_steps)
            .enumerate()
            .map(|(ordinal, (segment, &release_step))| {
                TranscriptBarrier::from_segment(
                    ordinal,
                    input.topology.coordinator,
                    segment,
                    release_step,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let transcript_encoding = identity::encode_transcript(transcript)?;
        let mut plan = Self {
            input,
            schedule_key: transcript.schedule_key(),
            transcript_encoding,
            barriers,
            workers: Vec::new(),
            identity: [0; 32],
        };
        plan.workers = validate::validate_and_measure(&plan, transcript)?;
        plan.identity = identity::compute(&plan)?;
        plan.validate(transcript)?;
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

    pub fn input(&self) -> &FleetPlanInput {
        &self.input
    }

    pub fn barriers(&self) -> &[TranscriptBarrier] {
        &self.barriers
    }

    pub const fn terminal_step(&self) -> ScheduleStep {
        self.input.terminal_step
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
        self.input.pow.plan(site).verify_winner(
            site,
            self.input.topology.workers.len(),
            self.identity,
            proof_generation,
            receipts,
            is_valid_nonce,
        )
    }
}

fn canonicalize(input: &mut FleetPlanInput) {
    input
        .topology
        .workers
        .sort_unstable_by_key(|worker| worker.id);
    input.topology.links.sort_unstable_by_key(|link| link.id);
    input
        .topology
        .host_numa
        .sort_unstable_by_key(|capacity| capacity.numa_node);
    input.values.sort_unstable_by_key(|value| value.id);
    input
        .barrier_arrivals
        .sort_unstable_by_key(|arrival| (arrival.barrier_ordinal, arrival.worker));
    input
        .operations
        .sort_unstable_by_key(|operation| operation.id);
    input
        .assignments
        .sort_unstable_by_key(|assignment| assignment.operation);
    input.owners.sort_unstable_by_key(|owner| {
        (
            owner.value,
            owner.elements.start,
            owner.elements.end,
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
    DuplicateValue(ValueId),
    DuplicateOperation(OperationId),
    DuplicateAssignment(OperationId),
    MissingAssignment(OperationId),
    UnknownWorker(WorkerId),
    UnknownValue(ValueId),
    UnknownOperation(OperationId),
    InvalidOperation(OperationId),
    InvalidLayout(ValueId),
    InvalidRange(ValueId),
    InvalidSchedule,
    InvalidSegment(OperationId),
    OwnershipCoverage(ValueId),
    InvalidProducer(ValueId),
    UndeclaredRead {
        operation: OperationId,
        value: ValueId,
    },
    InvalidReplica(ReplicaId),
    InvalidTransition(LayoutTransitionId),
    CapacityExceeded {
        worker: WorkerId,
        required: usize,
        capacity: usize,
    },
    TranscriptMismatch,
    SpillValue(SpillChunkIdForError),
    Spill(SpillPlanError),
    Pow(FleetPowError),
    SizeOverflow,
    IdentityMismatch,
}

/// Keeps the public error independent of the spill module's internal lookup maps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpillChunkIdForError(pub u32);

impl core::fmt::Display for FleetPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid explicit fleet proof plan: {self:?}")
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
