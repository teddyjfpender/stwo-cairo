//! Diagnostic arena inventory plus the production ReplacementV1 Base authority.
//!
//! [`ArenaProgramInventory`] remains a test-only migration aid: the arena is
//! authoritative for value extents, alignment, transcript I/O and
//! the packed proof ABI. It is not authoritative for operation effects. This
//! module therefore emits the complete arena inventory and selects the first
//! pending entry in canonical arena order whose producer contract is missing.
//! That selector is not proven execution or topological order. The inventory
//! never fabricates an `AotKernelId` and is not the source-generated production
//! `ProgramImage` required by the replacement backend.
//!
//! One inventory entry is emitted per arena `LogicalBufferId`. These entries
//! are not semantic SSA values or `compiled_proof::ValueVersion` authority:
//! arena reuse can place several entries in one allocation, while one entry
//! can still require several semantic versions in the compiled program. The
//! separate Base authority lowering is production input for ReplacementV1; it
//! does not promote this inventory or claim the later proof DAG is compiled.

use core::ops::Range;

use stwo_backend_cuda::{TranscriptInputId, TranscriptOutputId};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use crate::arena_plan::{BufferLifetime, BufferPurpose, LogicalBufferId, ProofEpoch};
use crate::compiled_proof::{CompiledProof, ProofBundleSection, ProofCodecIdentity, ValueLayout};
use crate::proof_bundle::ResidentProofBundleLayout;
use crate::transcript_plan::{CairoBlake2sTranscriptPlan, CairoTranscriptSegment};

mod emitter;
mod identity;
mod lower_compiled;

use identity::ArenaProgramInventoryIdentity;
pub(crate) use lower_compiled::{
    bind_replacement_base_authority, compile_replacement_base_authority, BaseProducerAuthority,
    BaseProducerAuthorityError, LoadedBaseProducerAuthority, PreparedBaseProducerInventory,
    PreparedBlakeGDirectKernel, PreparedEcOpSegment, PreparedGenericMultiplicityFeed,
    PreparedPublicMemorySeed, PreparedRecordedKernel,
};

/// Dense ID for one arena inventory entry. This is deliberately not named a
/// `ValueId`: its 1:1 mapping to `LogicalBufferId` is storage inventory, not
/// semantic `compiled_proof::ValueVersion` authority.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ArenaCatalogValueId(pub u32);

/// Honest origin state at the current lowering boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramValueOrigin {
    ExternalInput(LogicalBufferId),
    FixedImage(LogicalBufferId),
    TranscriptOutput(TranscriptOutputId),
    /// A typed producer/initializer contract has not yet been supplied. This
    /// is intentionally not converted to `compiled_proof::ValueOrigin`.
    PendingOperationContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramValueDesc {
    pub id: ArenaCatalogValueId,
    pub logical: LogicalBufferId,
    pub component: Option<&'static str>,
    pub part: Option<TracePartId>,
    pub purpose: BufferPurpose,
    pub ordinal: u32,
    pub layout: ValueLayout,
    pub alignment: usize,
    pub lifetime: BufferLifetime,
    pub origin: ProgramValueOrigin,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramTranscriptInputBinding {
    pub id: TranscriptInputId,
    pub value: ArenaCatalogValueId,
    pub value_words: Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramTranscriptOutputBinding {
    pub id: TranscriptOutputId,
    pub value: ArenaCatalogValueId,
    pub value_words: Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramProofOutputSection {
    pub section: ProofBundleSection,
    pub value: ArenaCatalogValueId,
    pub value_words: Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramProofOutput {
    pub codec: ProofCodecIdentity,
    pub layout: ResidentProofBundleLayout,
    pub sections: Vec<ProgramProofOutputSection>,
}

/// Minimal execution-IR distinction required before lowering real operations.
/// Copies, transcript work and assembly are deliberately not kernel IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionPrimitiveClass {
    KernelLaunch,
    DeviceCopy,
    Transcript,
    ProofAssembly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingOperationField {
    PrimitiveAuthority,
    LaunchGeometry,
    PartitionAuthority,
    ReadValueRanges,
    CompleteWriteSet,
    EffectContract,
}

/// Source-free strict-AOT candidate derivable from the recorded witness
/// program and the current u64 manifest receipt. This is not collision-
/// resistant primitive authority and cannot populate an `AotKernelId`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessKernelCandidate {
    pub label: String,
    pub kernel_name: String,
    pub semantic_hash: u64,
    pub cache_key: u64,
    pub aot_manifest_hash: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelLaunchCandidate {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_shared_bytes: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArenaCatalogRange {
    pub value: ArenaCatalogValueId,
    pub value_words: Range<usize>,
}

/// Rule that a future typed partition authority must preserve. Recorded
/// witness rows are one-row-per-thread candidates, but any shared global
/// multiplicity atomics must remain coordinator-owned or be replaced by local
/// partials followed by a canonical reduction. They make naive row sharding
/// incorrect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlobalMultiplicityPartitionRule {
    CoordinatorOwnedOrCanonicalReduction,
}

/// Evidence useful for closing the partition contract. It is not partition
/// authority: legal axes, alignment and granularity must ultimately come from
/// the same typed primitive consumed by execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessPartitionCandidate {
    pub row_count: usize,
    pub row_granularity: usize,
    pub global_multiplicity_outputs: u32,
    pub multiplicity_rule: GlobalMultiplicityPartitionRule,
}

/// First pending arena entry in canonical inventory order whose producer is
/// not typed. This is not asserted to be execution or topological order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationFrontier {
    pub value: ArenaCatalogValueId,
    pub logical: LogicalBufferId,
    pub component: Option<&'static str>,
    pub part: Option<TracePartId>,
    pub purpose: BufferPurpose,
    pub ordinal: u32,
    pub producer_epoch: ProofEpoch,
    pub earliest_transcript_stage: CairoTranscriptSegment,
    pub primitive: ExecutionPrimitiveClass,
    /// Present only for the ordinary recorded-witness launch whose source-free
    /// strict-AOT candidate is reconstructible without CUDA.
    pub witness_kernel_candidate: Option<WitnessKernelCandidate>,
    /// Candidate reconstructed from the current launch implementation. It is
    /// not sealed launch authority because the 256-thread rule is duplicated
    /// across the Rust emitter, generated source and CUDA launch shim.
    pub launch_candidate: Option<KernelLaunchCandidate>,
    pub partition_candidate: Option<WitnessPartitionCandidate>,
    /// Exact direct input columns already named by the witness plan. Table and
    /// module-global reads remain absent until they become typed ValueIds.
    pub known_reads: Vec<ArenaCatalogRange>,
    /// Complete direct writer destinations for the ordinary recorded writer,
    /// admitted only after its multiplicity-free contract is rechecked.
    pub known_writes: Vec<ArenaCatalogRange>,
    pub missing: Vec<MissingOperationField>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArenaProgramInventoryReceipt {
    pub catalog_values: usize,
    pub catalog_words: usize,
    pub external_values: usize,
    pub fixed_values: usize,
    pub transcript_output_values: usize,
    pub pending_operation_values: usize,
    pub transcript_inputs: usize,
    pub transcript_outputs: usize,
    pub proof_bundle_words: usize,
    pub lowered_operations: usize,
    pub frontier: OperationFrontier,
}

/// Complete real-shape arena inventory. This diagnostic is intentionally not a
/// production `ProgramImage`; it can only report the first canonical missing-
/// producer frontier and fail promotion.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ArenaProgramInventory {
    identity: ArenaProgramInventoryIdentity,
    values: Vec<ProgramValueDesc>,
    transcript_inputs: Vec<ProgramTranscriptInputBinding>,
    transcript_outputs: Vec<ProgramTranscriptOutputBinding>,
    output: ProgramProofOutput,
    frontier: OperationFrontier,
}

impl ArenaProgramInventory {
    /// Build the address-free diagnostic from the same locals used during
    /// shape compilation. No production executable accepts this inventory.
    pub fn from_planned_parts(
        topology: &crate::shape_executable::TopologyKey,
        transcript: &CairoBlake2sTranscriptPlan,
        arena: &crate::arena_plan::ProofArenaPlan,
    ) -> Result<Self, ArenaProgramInventoryError> {
        emitter::from_planned_parts(topology, transcript, arena)
    }

    pub const fn identity(&self) -> &ArenaProgramInventoryIdentity {
        &self.identity
    }

    pub const fn output(&self) -> &ProgramProofOutput {
        &self.output
    }

    pub const fn frontier(&self) -> &OperationFrontier {
        &self.frontier
    }

    pub fn receipt(&self) -> Result<ArenaProgramInventoryReceipt, ArenaProgramInventoryError> {
        let catalog_words = self.values.iter().try_fold(0usize, |sum, value| {
            value
                .layout
                .element_count()
                .ok()
                .and_then(|words| sum.checked_add(words))
                .ok_or(ArenaProgramInventoryError::SizeOverflow)
        })?;
        let count = |origin: fn(ProgramValueOrigin) -> bool| {
            self.values
                .iter()
                .filter(|value| origin(value.origin))
                .count()
        };
        Ok(ArenaProgramInventoryReceipt {
            catalog_values: self.values.len(),
            catalog_words,
            external_values: count(|origin| matches!(origin, ProgramValueOrigin::ExternalInput(_))),
            fixed_values: count(|origin| matches!(origin, ProgramValueOrigin::FixedImage(_))),
            transcript_output_values: count(|origin| {
                matches!(origin, ProgramValueOrigin::TranscriptOutput(_))
            }),
            pending_operation_values: count(|origin| {
                origin == ProgramValueOrigin::PendingOperationContract
            }),
            transcript_inputs: self.transcript_inputs.len(),
            transcript_outputs: self.transcript_outputs.len(),
            proof_bundle_words: self.output.layout.total_words,
            lowered_operations: 0,
            frontier: self.frontier.clone(),
        })
    }

    /// Promotion gate. The current real arena lacks the complete typed contract
    /// named by `frontier`, so conversion fails before constructing any fake
    /// operation, authority, or identity.
    pub fn try_promote_to_compiled_proof(
        &self,
    ) -> Result<CompiledProof, ArenaProgramInventoryError> {
        Err(ArenaProgramInventoryError::MissingTypedProducerFrontier(
            self.frontier.clone(),
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ArenaProgramInventoryError {
    SizeOverflow,
    InvalidArena(&'static str),
    NonDenseLogicalValue {
        expected: LogicalBufferId,
        actual: LogicalBufferId,
    },
    MissingArenaBinding(LogicalBufferId),
    MissingArenaSlot(LogicalBufferId),
    BindingLength(LogicalBufferId),
    InvalidAlignment(LogicalBufferId),
    Transcript(&'static str),
    ProofBundle(&'static str),
    MissingTypedProducerFrontier(OperationFrontier),
    TranscriptPlan(crate::transcript_plan::TranscriptPlanError),
    BundleLayout(crate::proof_bundle::ResidentProofBundleError),
}

impl core::fmt::Display for ArenaProgramInventoryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "real arena inventory failed: {self:?}")
    }
}

impl std::error::Error for ArenaProgramInventoryError {}

impl From<crate::transcript_plan::TranscriptPlanError> for ArenaProgramInventoryError {
    fn from(value: crate::transcript_plan::TranscriptPlanError) -> Self {
        Self::TranscriptPlan(value)
    }
}

impl From<crate::proof_bundle::ResidentProofBundleError> for ArenaProgramInventoryError {
    fn from(value: crate::proof_bundle::ResidentProofBundleError) -> Self {
        Self::BundleLayout(value)
    }
}

#[cfg(test)]
mod tests;
