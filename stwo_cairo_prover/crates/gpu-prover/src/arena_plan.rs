//! Checked device-memory liveness and physical arena coloring.
//!
//! The proof shape describes logical columns. This module turns those columns plus
//! protocol scratch geometry into one stable-address [`ArenaLayout`]. Logical
//! buffers may share a physical slot only when their inclusive proof-epoch
//! lifetimes are disjoint. The resulting alias proof is computed before CUDA is
//! touched; graph capture therefore never discovers residency by allocation luck.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use stwo::core::circle::CirclePoint;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::jit_witness::isa::WitnessProgram;
use stwo_backend_cuda::{
    blake2s_pow_workspace_requirements, blake_g_fusion_program_is_exact,
    commit_workspace_requirements, decommit_workspace_requirements, ec_op_workspace_requirements,
    execution_tables_workspace_requirements, fri_final_workspace_requirements,
    fri_workspace_requirements, oods_workspace_requirements,
    progressive_commit_workspace_requirements_for_mode, quotient_numerator_hybrid_plan,
    quotient_numerator_workspace_requirements, quotient_workspace_requirements,
    witness_input_compact_requirements, witness_input_gather_requirements,
    witness_workspace_requirements, ArenaError, ArenaLayout, ArenaSlotId, ArenaSlotSpec,
    Blake2sFriAssemblyShape, Blake2sPowWorkspaceRequirements, Blake2sPowWorkspaceSlots,
    Blake2sProofAssemblyShape, Blake2sTraceAssemblyShape, Blake2sTranscriptRequirements,
    Blake2sTranscriptWorkspaceSlots, CommitBatchRequirements, CommitBatchSlots, CommitGroupSlots,
    CommitWorkspaceConfig, CommitWorkspaceSlots, CudaExecContext, DecommitColumnGeometry,
    DecommitSourceMode, DecommitTreeGeometry, DecommitTreeRequirements, DecommitTreeSlots,
    DecommitWorkspaceConfig, DecommitWorkspaceRequirements, DecommitWorkspaceSlots, DeviceArena,
    DeviceTranscriptError, EcOpMultiplicityGeometry, EcOpWorkspaceRequirements, EcOpWorkspaceSlots,
    ExecutionTablesWorkspaceRequirements, ExecutionTablesWorkspaceSlots,
    FixedTableContiguousWorkspaceSlots, FriDecommitGeometry, FriDecommitSlots,
    FriFinalWorkspaceRequirements, FriFinalWorkspaceSlots, FriFoldLaunchMode, FriMerkleTreeSlots,
    FriWorkspaceConfig, FriWorkspaceRequirements, FriWorkspaceSlots, InterpolationLaunchMode,
    MerkleFromLeavesSlots, ModeAwareCommitWorkspaceRequirements, ModeAwareCommitWorkspaceSlots,
    OodsColumnTopology, OodsSourceKind, OodsWorkspaceConfig, OodsWorkspaceRequirements,
    OodsWorkspaceSlots, PreparedBlake2sPowError, PreparedCommitError, PreparedDecommitError,
    PreparedExecutionTablesError, PreparedFixedTableError, PreparedFriError, PreparedFriFinalError,
    PreparedOodsError, PreparedProgressiveCommitError, PreparedQuotientError,
    PreparedQuotientNumeratorError, PreparedWitnessError, PreparedWitnessFeedError,
    PreparedWitnessInputGatherError, ProgressiveBatchRequirements, ProgressiveBatchSlots,
    ProgressiveCommitGeometry, ProgressiveCommitGroupGeometry, ProgressiveCommitMode,
    ProgressiveCommitWorkspaceSlots, ProgressiveLeafWorkspaceSlots,
    QuotientNumeratorColumnTopology, QuotientNumeratorSingleWriteError,
    QuotientNumeratorSourceKind, QuotientNumeratorWorkspaceConfig,
    QuotientNumeratorWorkspaceRequirements, QuotientNumeratorWorkspaceSlots, QuotientOodsSample,
    QuotientWorkspaceConfig, QuotientWorkspaceRequirements, QuotientWorkspaceSlots,
    RelationGraphError, RelationGraphRequirements, RelationGraphSlots, RelationInstanceSlots,
    RelationLaunchMode, RelationTailMode, TraceDecommitGeometry, TraceDecommitSlots,
    TraceSourceGroupGeometry, TraceSourceGroupSlots, TraceTreeRole, TranscriptInputId,
    TranscriptOutputId, WitnessFeedClearWorkspaceRequirements, WitnessFeedClearWorkspaceSlots,
    WitnessFeedLaunchMode, WitnessFeedWorkspaceSlots, WitnessInputCompactLayout,
    WitnessInputCompactRequirements, WitnessInputCompactSlots, WitnessInputGatherEdge,
    WitnessInputGatherRequirements, WitnessInputGatherSlots, WitnessInputSeedRequirements,
    WitnessInputSeedSlots, WitnessWorkspaceRequirements, WitnessWorkspaceSlots,
    EXECUTION_TABLE_BIG_LIMBS, EXECUTION_TABLE_SMALL_LIMBS,
};
use stwo_cairo_prover::witness::jit_prove_backend::{BlakeGRecordedLane, BuiltinLaneSpec};
use stwo_cairo_prover::witness::proof_shape::{
    ProofShapeError, RowResolution, TracePartId, TracePartShape,
};

use crate::composition_plan::{CompositionExtParamSource, CompositionPlan};
use crate::direct_composition_retention::{
    derive_direct_composition_consumers, direct_bitmap_hash,
    validate_direct_composition_retention_plan, DirectCompositionRetentionMode,
    DirectCompositionRetentionPlan,
};
use crate::fixed_table_materializer::{
    pedersen_points_18_column_index, PEDERSEN_POINTS_18_LOG_SIZE, PEDERSEN_POINTS_18_ROW_COUNT,
};
use crate::multiplicity_pipeline::{
    blake_g_fused_feed_binding, plan_graph_a_multiplicities, plan_public_memory_multiplicity_seed,
    FixedMultiplicityCoverageGap, GraphAMultiplicityPlan, GraphAMultiplicityPlanError,
    MultiplicityFeedBlocker, PlannedMemoryBaseTraces,
};
use crate::plan::ProofPlan;
use crate::prepared_composition::{
    composition_workspace_requirements_with_retention, CompositionCoefficientSource,
    CompositionExtParamBinding, CompositionLaunchMode, CompositionTraceTopology,
    CompositionWorkspaceRequirements, CompositionWorkspaceSlots, PreparedCompositionError,
};
use crate::proof_bundle::{ResidentProofBundleError, ResidentProofBundleLayout};
use crate::range_allocator::RangeAllocationError;
use crate::range_arena::{plan_range_arena, validate_transition_aliases};
use crate::relation::RelationTracePart;
use crate::relation_execution::{
    RelationExecutionError, RelationExecutionPlan, RelationInstanceSourcePlan, RelationSourcePlane,
};
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::schedule::{InputEdge, TraceColumnCount, WitnessWriterKind};
use crate::source_ownership::LateCoefficientOwnershipPlan;
use crate::transcript_plan::{CairoTranscriptInput, CairoTranscriptOutput};

/// Every arena range is at least 128-byte aligned. Kernel code may rely on this.
pub const ARENA_ALIGNMENT_WORDS: usize = 128 / core::mem::size_of::<u32>();
const BLAKE2S_HASH_WORDS: usize = 8;
const SECURE_FIELD_WORDS: usize = 4;
const CAIRO_RELATION_LAUNCH_MODE: RelationLaunchMode = RelationLaunchMode::Fused;
// Replacement-v1 seals a fixed maximum coefficient-LDE batch into its protocol
// identity. Exact arena plus physical-reserve admission still decides whether a
// shape fits; there is no runtime width fallback. Legacy keeps one column.
const REPLACEMENT_NUMERATOR_LDE_TILE_COLUMNS: usize = 32;

fn retain_fixed_preprocessed_evaluation(identity: &str) -> bool {
    pedersen_points_18_column_index(identity).is_none()
}

fn preprocessed_requires_registered_pedersen_table(columns: &[PlannedPreprocessedColumn]) -> bool {
    columns
        .iter()
        .any(|column| pedersen_points_18_column_index(&column.identity).is_some())
}

/// Coarse protocol epochs. Lifetimes are inclusive because a producer and a
/// consumer executing in the same epoch must not alias.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum ProofEpoch {
    Ingest,
    Witness,
    BaseCommit,
    Interaction,
    InteractionCommit,
    Composition,
    CompositionCommit,
    Oods,
    Quotient,
    Fri,
    Decommit,
    Assemble,
}

impl ProofEpoch {
    pub const ALL: [Self; 12] = [
        Self::Ingest,
        Self::Witness,
        Self::BaseCommit,
        Self::Interaction,
        Self::InteractionCommit,
        Self::Composition,
        Self::CompositionCommit,
        Self::Oods,
        Self::Quotient,
        Self::Fri,
        Self::Decommit,
        Self::Assemble,
    ];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferLifetime {
    pub first: ProofEpoch,
    pub last: ProofEpoch,
}

impl BufferLifetime {
    pub fn new(first: ProofEpoch, last: ProofEpoch) -> Result<Self, ArenaPlanError> {
        if first > last {
            return Err(ArenaPlanError::InvalidLifetime { first, last });
        }
        Ok(Self { first, last })
    }

    pub const fn at(epoch: ProofEpoch) -> Self {
        Self {
            first: epoch,
            last: epoch,
        }
    }

    pub const fn contains(self, epoch: ProofEpoch) -> bool {
        self.first as u8 <= epoch as u8 && epoch as u8 <= self.last as u8
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.first as u8 <= other.last as u8 && other.first as u8 <= self.last as u8
    }

    /// Bitmask with one bit per [`ProofEpoch`] discriminant, set for every
    /// epoch in this inclusive lifetime.
    ///
    /// A lifetime is exactly the contiguous range `first..=last` (`new`
    /// rejects `first > last`, `at` sets both). For two contiguous inclusive
    /// ranges `A = [a1, a2]` and `B = [b1, b2]`:
    ///
    /// `mask(A) & mask(B) != 0`
    ///   iff some epoch bit `e` is set in both masks
    ///   iff some epoch `e` satisfies `a1 <= e <= a2` and `b1 <= e <= b2`
    ///   iff `a1 <= b2 && b1 <= a2`
    ///   iff `A.overlaps(B)`.
    ///
    /// Because bitwise-or distributes over "has a common bit", a union of
    /// masks intersects `mask(B)` iff at least one constituent mask does, so a
    /// per-slot mask union replaces a per-lifetime overlap scan exactly.
    pub fn epoch_mask(self) -> u16 {
        // 12 proof epochs fit bits 0..=11; the shift below needs headroom.
        const _: () = assert!(ProofEpoch::ALL.len() <= 16);
        debug_assert!(self.first as u8 <= self.last as u8);
        ((1u32 << (self.last as u8 + 1)) - (1u32 << (self.first as u8))) as u16
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BufferPurpose {
    /// Persistent fixed-column coefficients, indexed in canonical
    /// preprocessed-tree proof order.
    PreprocessedCoefficients,
    /// Immutable fixed-column evaluations retained for Graph-A materializers.
    PreprocessedEvaluations,
    /// Setup-only same-log inverse-NTT pointer tables for fixed columns.
    PreprocessedInterpolationPointers,
    /// Full max-domain inverse twiddles used only while interpolating the
    /// preprocessed tree. This stays distinct from the smaller dynamic FRI
    /// inverse table when fixed columns have a taller commitment domain.
    PreprocessedInverseTwiddles,
    /// Canonical base-trace evaluations consumed by lookup/relation kernels.
    BaseTrace,
    /// Compact, statement-varying columns consumed by a prepared recorded
    /// witness kernel. These are the only per-proof witness H2D destinations.
    WitnessInput,
    /// Immutable prepared-witness ABI descriptors. They persist for the entire
    /// workspace lifetime because captured graphs dereference them on replay.
    WitnessInputPointers,
    WitnessInputGatherSourcePointers,
    WitnessInputGatherDescriptors,
    WitnessInputGatherOutputPointers,
    WitnessInputSeedScalars,
    WitnessInputSeedOutputPointers,
    WitnessInputCompactSourcePointers,
    WitnessInputCompactDescriptors,
    WitnessInputCompactOutputPointers,
    WitnessInputCompactTupleScratch,
    WitnessInputCompactSortKey,
    WitnessInputCompactSortIndex,
    WitnessInputCompactRunHeads,
    WitnessInputCompactRunPositions,
    WitnessInputCompactUniqueCount,
    WitnessInputCompactSortTemp,
    WitnessInputCompactScanTemp,
    ExecutionTableRawAddressToId,
    ExecutionTableRawF252Words,
    ExecutionTableRawSmallWords,
    ExecutionTableBigLimb,
    ExecutionTableSmallLimb,
    ExecutionTablePointers,
    ExecutionTableStrides,
    EcOpSegmentStart,
    EcOpPartialIota,
    WitnessExecutionTablePointers,
    WitnessExecutionTableStrides,
    WitnessOutputPointers,
    WitnessMultiplicityPointers,
    WitnessMultiplicityDummy,
    WitnessLookupDummy,
    WitnessSubDummy,
    FixedMultiplicity,
    RuntimeMultiplicity,
    FixedMultiplicityClearPointers,
    FixedMultiplicityClearLengths,
    WitnessFeedDescriptors,
    WitnessFeedLut,
    WitnessFeedLutPointers,
    WitnessFeedMultiplicityPointers,
    PublicMemoryMultiplicitySeed,
    FixedTableSourcePointers,
    FixedTableMultiplicityPointers,
    FixedTableTraceMultiplicityColumns,
    FixedTableTraceOutputPointers,
    FixedTableLookupDescriptors,
    FixedTableLookupOutputPointers,
    /// Interpolated base-trace coefficients consumed by commit, OODS,
    /// composition extension, quotient accumulation and decommit recomputation.
    BaseCoefficients,
    LookupInputs,
    SubcomponentInputs,
    /// Canonical interaction-trace evaluations emitted by the relation graph.
    InteractionTrace,
    /// Interpolated interaction coefficients consumed by every later PCS stage.
    InteractionCoefficients,
    /// Forward circle-FFT twiddles shared by every trace/composition commit and
    /// by queried-LDE recomputation during decommitment.
    ForwardTwiddles,
    /// Inverse circle-FFT twiddles consumed by the FRI fold chain.
    InverseTwiddles,
    /// Extracted inverse twiddles for the quotient subdomain. These are not the
    /// full-domain FRI inverse twiddles and must remain a separate value.
    QuotientInverseTwiddles,
    CommitLdeTile,
    MerkleLeafState,
    CommitProgressiveStatePing,
    CommitProgressiveStatePong,
    MerkleLayerScratch,
    CommitColumnPointers,
    CommitColumnLogSizes,
    CommitCoefficientPointers,
    InterpolationInputPointers,
    InterpolationOutputPointers,
    CommitCoefficientSizes,
    CommitOutputPointers,
    CommitRetainedEvaluation,
    MerkleTailPointers,
    RetainedMerkleLayers,
    FriMerkleLayer,
    CompositionDescriptors,
    CompositionLdeTile,
    CompositionAccumulators,
    CompositionRandomCoefficientPowers,
    CompositionExtParams,
    OodsSourcePointers,
    OodsOffsetPoints,
    OodsFoldCounts,
    OodsOutputIndices,
    OodsFoldingFactors,
    OodsScratchA,
    OodsScratchB,
    OodsSamplePoints,
    OodsEvaluationPoints,
    OodsBarycentricNumerators,
    OodsBarycentricWeights,
    OodsBarycentricScales,
    OodsBarycentricPartials,
    QuotientNumeratorRuntimeTerms,
    QuotientNumeratorGroupTermIndices,
    QuotientNumeratorGroupOffsets,
    QuotientNumeratorLineCoefficients,
    QuotientNumeratorTermPoints,
    QuotientNumeratorBatchTerms,
    QuotientNumeratorBatchGroupOffsets,
    QuotientNumeratorBatchSourcePointers,
    QuotientNumeratorOutputPointers,
    QuotientNumeratorOutputLogSizes,
    QuotientNumeratorCoefficientPointers,
    QuotientNumeratorCoefficientSizes,
    QuotientNumeratorCoefficientOutputPointers,
    QuotientNumeratorLdeTile,
    /// Four coordinates per canonical sample; ordinal = sample * 4 + coordinate.
    QuotientPartialNumerator,
    QuotientSamplePoints,
    QuotientFirstLinearTerms,
    QuotientPartialLogSizes,
    QuotientPartialCoordinatePointers,
    QuotientSubdomainCoordinatePointers,
    QuotientOutputCoordinatePointers,
    QuotientCoefficientSizes,
    QuotientSubdomainValues,
    QuotientTile,
    FriPing,
    FriPong,
    FriRetainedEvaluation,
    FriInputCoordinatePointers,
    FriPingCoordinatePointers,
    FriPongCoordinatePointers,
    FriRetainedCoordinatePointers,
    FriFoldingChallenge,
    FriFinalCoefficients,
    FriFinalDegreeError,
    PowBestNonce,
    PowCompletedBlocks,
    PowPrefixDigest,
    /// Eight split M31 coefficient columns committed as the composition tree.
    CompositionCoefficients,
    RelationDescriptors,
    RelationAlphaPowers,
    RelationZ,
    RelationInverseScratch,
    RelationReductionA,
    RelationReductionB,
    RelationScanEvalScratch,
    RelationScanTempScratch,
    RelationScanDescriptors,
    RelationFractionPointers,
    RelationFractionGeometry,
    RelationSourcePointers,
    RelationOutputPointers,
    RelationDenominators,
    RelationClaimedSum,
    QueryIndices,
    DecommitHashes,
    DecommitUniqueQueries,
    DecommitMappedQueries,
    DecommitWalkQueries,
    DecommitWalkScratch,
    DecommitExpandedPositions,
    DecommitSparseIndices,
    DecommitSparseHashes,
    DecommitCounts,
    DecommitTraceRetainedPointers,
    DecommitTraceSparseOffsets,
    DecommitTraceEvaluationPointers,
    DecommitTraceEvaluationLogs,
    DecommitTraceCoefficientPointers,
    DecommitTraceCoefficientSizes,
    DecommitTraceLdeOutputPointers,
    DecommitTraceLdeTile,
    DecommitFriCoordinatePointers,
    DecommitFriRetainedPointers,
    TranscriptState,
    TranscriptBoundarySnapshots,
    TranscriptInputSnapshots,
    TranscriptOutputSnapshots,
    TranscriptInput,
    TranscriptOutput,
    ProofBytes,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LogicalBufferId(pub u32);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalBuffer {
    pub id: LogicalBufferId,
    pub component: Option<&'static str>,
    pub part: Option<TracePartId>,
    pub purpose: BufferPurpose,
    /// Distinguishes repeated protocol buffers of the same purpose (for example,
    /// one retained Merkle range per commitment tree).
    pub ordinal: u32,
    pub len_words: usize,
    pub lifetime: BufferLifetime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArenaBinding {
    pub logical: LogicalBufferId,
    pub physical: ArenaSlotId,
    pub len_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CommitmentTreeId {
    Preprocessed,
    Base,
    Interaction,
    Composition,
    Fri(u8),
}

/// One exact lifted commitment topology. `grouped_column_log_sizes` is the
/// canonical global leaf order split into 16-wide update groups and a final
/// group of at most 16 columns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitmentColumnSource {
    /// Fixed preprocessed coefficient in canonical preprocessed-tree proof
    /// order. These slots are populated and committed once per exact workspace.
    Preprocessed {
        ordinal: u32,
    },
    Trace {
        component: &'static str,
        part: TracePartId,
        purpose: BufferPurpose,
        ordinal: u32,
    },
    Composition {
        ordinal: u32,
    },
}

/// One committed polynomial in canonical PCS opening order. This is
/// intentionally separate from [`CommitmentColumnSource`]: prepared commits
/// stably sort columns by log size, while OODS sampling and Fiat-Shamir powers
/// retain preprocessed/base/interaction/composition proof order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OpenedColumnSource {
    Preprocessed {
        ordinal: u32,
    },
    Trace {
        component: &'static str,
        part: TracePartId,
        purpose: BufferPurpose,
        ordinal: u32,
    },
    Composition {
        ordinal: u32,
    },
}

impl From<CommitmentColumnSource> for OpenedColumnSource {
    fn from(source: CommitmentColumnSource) -> Self {
        match source {
            CommitmentColumnSource::Preprocessed { ordinal } => Self::Preprocessed { ordinal },
            CommitmentColumnSource::Trace {
                component,
                part,
                purpose,
                ordinal,
            } => Self::Trace {
                component,
                part,
                purpose,
                ordinal,
            },
            CommitmentColumnSource::Composition { ordinal } => Self::Composition { ordinal },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitmentGeometry {
    pub id: CommitmentTreeId,
    pub created: ProofEpoch,
    pub config: CommitWorkspaceConfig,
    pub grouped_column_log_sizes: Vec<Vec<u32>>,
    /// Exact stable arena source paired one-for-one with every canonical log.
    pub grouped_column_sources: Vec<Vec<CommitmentColumnSource>>,
    /// Explicit commitment/opening policy in canonical group order.
    pub retained_evaluation_groups: Vec<bool>,
    /// Exact-native evaluations retained only through composition. These are
    /// separate from decommit intent; allocation uses the union.
    pub direct_composition_evaluation_groups: Vec<bool>,
    /// Exact-native evaluations retained through quotient-numerator evaluation.
    /// This intent is independent of decommit and direct composition.
    pub numerator_evaluation_groups: Vec<bool>,
}

fn commitment_workspace_requirements(
    mode: ProgressiveCommitMode,
    commitment: &CommitmentGeometry,
) -> Result<ModeAwareCommitWorkspaceRequirements, PreparedCommitError> {
    match mode {
        ProgressiveCommitMode::FullLifting => Ok(
            ModeAwareCommitWorkspaceRequirements::FullLifting(commit_workspace_requirements(
                commitment.config,
                &commitment.grouped_column_log_sizes,
            )?),
        ),
        ProgressiveCommitMode::DomainProgressive => {
            let groups = commitment
                .grouped_column_log_sizes
                .iter()
                .zip(&commitment.retained_evaluation_groups)
                .zip(&commitment.direct_composition_evaluation_groups)
                .zip(&commitment.numerator_evaluation_groups)
                .map(
                    |(((logs, &retain_decommit), &retain_direct), &retain_numerator)| {
                        ProgressiveCommitGroupGeometry {
                            coefficient_log_sizes: logs.clone(),
                            retain_evaluations: retain_decommit
                                || retain_direct
                                || retain_numerator,
                        }
                    },
                )
                .collect();
            progressive_commit_workspace_requirements_for_mode(
                mode,
                commitment.config,
                ProgressiveCommitGeometry {
                    lifting_log_size: commitment.config.lifting_log_size,
                    log_blowup_factor: commitment.config.log_blowup_factor,
                    groups,
                },
            )
            .map(ModeAwareCommitWorkspaceRequirements::DomainProgressive)
            .map_err(|_| PreparedCommitError::SlotShapeMismatch {
                role: "progressive_commit_geometry",
                expected: 1,
                actual: 0,
            })
        }
    }
}

fn commitment_group_evaluation_bytes(
    commitment: &CommitmentGeometry,
    group: usize,
) -> Result<usize, ArenaPlanError> {
    commitment
        .grouped_column_log_sizes
        .get(group)
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "retained evaluation group index is out of range",
        ))?
        .iter()
        .try_fold(0usize, |bytes, &coefficient_log| {
            let evaluation_log = coefficient_log
                .checked_add(commitment.config.log_blowup_factor)
                .ok_or(ArenaPlanError::SizeOverflow)?;
            let words = 1usize
                .checked_shl(evaluation_log)
                .ok_or(ArenaPlanError::SizeOverflow)?;
            bytes
                .checked_add(
                    words
                        .checked_mul(core::mem::size_of::<u32>())
                        .ok_or(ArenaPlanError::SizeOverflow)?,
                )
                .ok_or(ArenaPlanError::SizeOverflow)
        })
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum DecommitStrategy {
    RetainAllLde = 0,
    RecomputeQueriedLde = 1,
    HybridByGroup = 2,
}

/// Resident implementation generation selected before topology planning.
/// Different generations never share a workspace or captured graph identity.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum ResidentBackend {
    #[default]
    LegacyResident = 0,
    ReplacementV1 = 1,
}

impl ResidentBackend {
    pub const fn cli_name(self) -> &'static str {
        match self {
            Self::LegacyResident => "legacy-resident",
            Self::ReplacementV1 => "replacement-v1",
        }
    }
}

/// Quotient-numerator launch topology sealed into the graph key.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum QuotientNumeratorSchedule {
    #[default]
    LegacyBatches = 0,
    HybridSingleWrite = 1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum QuotientNumeratorSourcePolicy {
    CoefficientsOnly = 0,
    ReuseRetainedEvaluations = 1,
}

impl QuotientNumeratorSourcePolicy {
    pub fn from_env() -> Self {
        static POLICY: std::sync::OnceLock<QuotientNumeratorSourcePolicy> =
            std::sync::OnceLock::new();
        *POLICY.get_or_init(|| {
            if crate::flags::flag_on("STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS") {
                Self::ReuseRetainedEvaluations
            } else {
                Self::CoefficientsOnly
            }
        })
    }
}

/// Proof-format and generated-code identity not implied by raw buffer sizes.
/// Any change here gets a separate graph/workspace cache entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProtocolIdentity {
    pub pow_bits: u32,
    pub log_blowup_factor: u32,
    pub log_last_layer_degree_bound: u32,
    pub fri_fold_step: u32,
    /// Stable channel/hash-suite tag, not a Rust type hash.
    pub channel_tag: u64,
    pub relation_graph_hash: u64,
    pub preprocessed_binding_hash: u64,
    /// Exact proof-order OODS column/mask topology. This prevents two claims
    /// with size-compatible arenas but different opening powers from reusing a
    /// captured graph.
    pub oods_topology_hash: u64,
    pub composition_plan_hash: u64,
    pub kernel_manifest_hash: u64,
    pub decommit_strategy: DecommitStrategy,
    pub interpolation_mode: InterpolationLaunchMode,
    pub blake2s_interior_fused: bool,
    pub composition_launch_mode: CompositionLaunchMode,
    pub relation_tail_mode: RelationTailMode,
    pub fri_fold_launch_mode: FriFoldLaunchMode,
    pub witness_feed_launch_mode: WitnessFeedLaunchMode,
    pub resident_backend: ResidentBackend,
    pub quotient_numerator_schedule: QuotientNumeratorSchedule,
    pub quotient_numerator_source_policy: QuotientNumeratorSourcePolicy,
    pub commit_mode: stwo_backend_cuda::ProgressiveCommitMode,
    pub direct_composition_retention_mode: DirectCompositionRetentionMode,
    pub direct_composition_planner_key: u64,
    pub direct_composition_occurrence_bitmap_hash: u64,
    pub direct_composition_group_rounded_bytes: usize,
    pub numerator_evaluation_group_rounded_bytes: usize,
    pub retained_evaluation_union_bytes: usize,
}

impl ProtocolIdentity {
    pub fn from_pcs(
        pcs: &PcsConfig,
        channel_tag: u64,
        relation_graph_hash: u64,
        preprocessed_binding_hash: u64,
        oods_topology_hash: u64,
        composition_plan_hash: u64,
        kernel_manifest_hash: u64,
        decommit_strategy: DecommitStrategy,
        interpolation_mode: InterpolationLaunchMode,
        blake2s_interior_fused: bool,
        composition_launch_mode: CompositionLaunchMode,
        relation_tail_mode: RelationTailMode,
        fri_fold_launch_mode: FriFoldLaunchMode,
        witness_feed_launch_mode: WitnessFeedLaunchMode,
        resident_backend: ResidentBackend,
        quotient_numerator_schedule: QuotientNumeratorSchedule,
        commit_mode: stwo_backend_cuda::ProgressiveCommitMode,
        direct_composition_retention_mode: DirectCompositionRetentionMode,
        direct_composition_planner_key: u64,
        direct_composition_occurrence_bitmap_hash: u64,
        direct_composition_group_rounded_bytes: usize,
        quotient_numerator_source_policy: QuotientNumeratorSourcePolicy,
        numerator_evaluation_group_rounded_bytes: usize,
        retained_evaluation_union_bytes: usize,
    ) -> Self {
        Self {
            pow_bits: pcs.pow_bits,
            log_blowup_factor: pcs.fri_config.log_blowup_factor,
            log_last_layer_degree_bound: pcs.fri_config.log_last_layer_degree_bound,
            fri_fold_step: pcs.fri_config.fold_step,
            channel_tag,
            relation_graph_hash,
            preprocessed_binding_hash,
            oods_topology_hash,
            composition_plan_hash,
            kernel_manifest_hash,
            decommit_strategy,
            interpolation_mode,
            blake2s_interior_fused,
            composition_launch_mode,
            relation_tail_mode,
            fri_fold_launch_mode,
            witness_feed_launch_mode,
            resident_backend,
            quotient_numerator_schedule,
            quotient_numerator_source_policy,
            commit_mode,
            direct_composition_retention_mode,
            direct_composition_planner_key,
            direct_composition_occurrence_bitmap_hash,
            direct_composition_group_rounded_bytes,
            numerator_evaluation_group_rounded_bytes,
            retained_evaluation_union_bytes,
        }
    }
}

/// Exact device-transcript storage and semantic I/O contract. The schedule
/// itself remains in `transcript_plan`; this projection is sufficient to key
/// and allocate a stable proof workspace before capture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptGeometry {
    pub schedule_key: u64,
    pub requirements: Blake2sTranscriptRequirements,
}

impl TranscriptGeometry {
    pub fn validate(&self) -> Result<(), ArenaPlanError> {
        let requirements = &self.requirements;
        if self.schedule_key == 0
            || requirements.state_words == 0
            || requirements.boundary_snapshot_words == 0
            || requirements.input_snapshot_words == 0
            || requirements.output_snapshot_words == 0
            || requirements.inputs.is_empty()
            || requirements.outputs.is_empty()
            || requirements
                .inputs
                .iter()
                .any(|requirement| requirement.min_words == 0)
            || requirements
                .outputs
                .iter()
                .any(|requirement| requirement.min_words == 0)
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "empty or unbound device transcript geometry",
            ));
        }
        if requirements.input_snapshot_used_words > requirements.input_snapshot_words
            || requirements.output_snapshot_used_words > requirements.output_snapshot_words
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "device transcript snapshot use exceeds capacity",
            ));
        }
        if requirements
            .inputs
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
            || requirements
                .outputs
                .windows(2)
                .any(|pair| pair[0].id >= pair[1].id)
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "device transcript I/O ids are duplicated or non-canonical",
            ));
        }
        Ok(())
    }
}

/// Exact quotient numerator topology. The vector order is the canonical sample
/// order used by the OODS constants and by the prepared quotient descriptors.
/// Each entry is the post-blowup (coefficient/subdomain) log size after lifting
/// all numerator groups for that sample point; the vector length is therefore
/// the exact partial-numerator source count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuotientGeometry {
    pub partial_numerator_log_sizes: Vec<u32>,
}

impl QuotientGeometry {
    pub fn partial_numerator_count(&self) -> usize {
        self.partial_numerator_log_sizes.len()
    }
}

/// One polynomial and its fixed mask in canonical four-tree opening order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OodsColumnGeometry {
    pub source: OpenedColumnSource,
    pub coefficient_log_size: u32,
    pub evaluation_log_size: u32,
    pub shape_points: Vec<CirclePoint<SecureField>>,
    pub offset_points: Vec<CirclePoint<BaseField>>,
}

/// Address-free OODS and quotient-numerator topology. Source representation is
/// compiled from the exact commitment-group ownership policy; proof order and
/// opening geometry remain independent of that physical choice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OodsGeometry {
    pub mask_log_size: u32,
    pub sampled_values_input: TranscriptInputId,
    pub point_parameter_output: TranscriptOutputId,
    pub quotient_random_coefficient_output: TranscriptOutputId,
    pub columns: Vec<OodsColumnGeometry>,
}

impl OodsGeometry {
    pub fn sample_count(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.shape_points.len())
            .sum()
    }

    pub fn column_topologies(
        &self,
        source_kinds: &[OodsSourceKind],
    ) -> Result<Vec<OodsColumnTopology<'_>>, ArenaPlanError> {
        if source_kinds.len() != self.columns.len() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS source policy width mismatch",
            ));
        }
        Ok(self
            .columns
            .iter()
            .zip(source_kinds)
            .map(|(column, source_kind)| match source_kind {
                OodsSourceKind::Coefficients => OodsColumnTopology::coefficient_offset_points(
                    column.coefficient_log_size,
                    column.evaluation_log_size,
                    &column.offset_points,
                ),
                OodsSourceKind::Evaluations => OodsColumnTopology::evaluation_offset_points(
                    column.evaluation_log_size,
                    &column.offset_points,
                ),
            })
            .collect())
    }

    pub fn quotient_numerator_topologies(
        &self,
        source_kinds: &[QuotientNumeratorSourceKind],
    ) -> Result<Vec<QuotientNumeratorColumnTopology>, ArenaPlanError> {
        if source_kinds.len() != self.columns.len() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "quotient numerator source policy width mismatch",
            ));
        }
        let mut sample_index = 0usize;
        self.columns
            .iter()
            .zip(source_kinds)
            .map(|(column, &source_kind)| {
                let samples = column
                    .shape_points
                    .iter()
                    .map(|&shape_point| {
                        let input_index = u32::try_from(sample_index)
                            .map_err(|_| ArenaPlanError::SizeOverflow)?;
                        sample_index = sample_index
                            .checked_add(1)
                            .ok_or(ArenaPlanError::SizeOverflow)?;
                        Ok(QuotientOodsSample {
                            input_index,
                            shape_point,
                        })
                    })
                    .collect::<Result<Vec<_>, ArenaPlanError>>()?;
                Ok(QuotientNumeratorColumnTopology {
                    coefficient_log_size: column.coefficient_log_size,
                    source_kind,
                    samples,
                })
            })
            .collect()
    }

    pub fn topology_hash(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        feed_hash(&mut hash, b"stwo-cairo-oods-topology-v1\0");
        feed_hash(&mut hash, &self.mask_log_size.to_le_bytes());
        feed_hash(&mut hash, &self.sampled_values_input.0.to_le_bytes());
        feed_hash(&mut hash, &self.point_parameter_output.0.to_le_bytes());
        feed_hash(
            &mut hash,
            &self.quotient_random_coefficient_output.0.to_le_bytes(),
        );
        feed_hash(&mut hash, &(self.columns.len() as u64).to_le_bytes());
        for column in &self.columns {
            feed_opened_source(&mut hash, column.source);
            feed_hash(&mut hash, &column.coefficient_log_size.to_le_bytes());
            feed_hash(&mut hash, &column.evaluation_log_size.to_le_bytes());
            feed_hash(&mut hash, &(column.shape_points.len() as u64).to_le_bytes());
            for point in &column.shape_points {
                for coordinate in [point.x, point.y] {
                    for felt in coordinate.to_m31_array() {
                        feed_hash(&mut hash, &felt.0.to_le_bytes());
                    }
                }
            }
            feed_hash(
                &mut hash,
                &(column.offset_points.len() as u64).to_le_bytes(),
            );
            for point in &column.offset_points {
                feed_hash(&mut hash, &point.x.0.to_le_bytes());
                feed_hash(&mut hash, &point.y.0.to_le_bytes());
            }
        }
        hash
    }
}

/// Exact protocol scratch geometry for one graph-template key. Component column
/// sizes come from [`ProofPlan`]; this carries the PCS/FRI dimensions that are not
/// component-local.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProtocolGeometry {
    pub identity: ProtocolIdentity,
    /// Canonical preprocessed-tree column identities in commitment ordinal
    /// order. The identity hash already seals the same list; retaining the
    /// names lets fixed-table materializers bind exact evaluation columns.
    pub preprocessed_column_ids: Vec<String>,
    pub max_domain_log_size: u32,
    pub lifting_log_size: u32,
    pub n_queries: usize,
    pub total_opened_columns: usize,
    pub proof_capacity_words: usize,
    pub transcript: TranscriptGeometry,
    pub composition_random_coefficient_output: TranscriptOutputId,
    pub oods: OodsGeometry,
    pub quotient: QuotientGeometry,
    pub commitments: Vec<CommitmentGeometry>,
    pub direct_composition_retention: Option<DirectCompositionRetentionPlan>,
    /// Commitment-tree leaf log sizes opened by the PCS, in proof tree order.
    /// The first tree is the cold-once resident preprocessed commitment.
    pub opened_tree_log_sizes: Vec<u32>,
    /// Fully retained FRI Merkle tree leaf log sizes, in transcript order.
    pub fri_layer_log_sizes: Vec<u32>,
}

impl ProtocolGeometry {
    pub fn oods_source_kinds(&self) -> Result<Vec<OodsSourceKind>, ArenaPlanError> {
        self.oods
            .columns
            .iter()
            .map(|column| {
                let tree = match column.source {
                    OpenedColumnSource::Preprocessed { .. } => {
                        return Ok(OodsSourceKind::Coefficients);
                    }
                    OpenedColumnSource::Trace {
                        purpose: BufferPurpose::BaseCoefficients,
                        ..
                    } => CommitmentTreeId::Base,
                    OpenedColumnSource::Trace {
                        purpose: BufferPurpose::InteractionCoefficients,
                        ..
                    } => CommitmentTreeId::Interaction,
                    OpenedColumnSource::Composition { .. } => CommitmentTreeId::Composition,
                    OpenedColumnSource::Trace { .. } => {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "OODS source is not a committed coefficient column",
                        ));
                    }
                };
                let commitment = self
                    .commitments
                    .iter()
                    .find(|commitment| commitment.id == tree)
                    .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                        "OODS source tree is missing",
                    ))?;
                let mut available_at_oods = None;
                for (group, (sources, logs)) in commitment
                    .grouped_column_sources
                    .iter()
                    .zip(&commitment.grouped_column_log_sizes)
                    .enumerate()
                {
                    for (&source, &log_size) in sources.iter().zip(logs) {
                        if OpenedColumnSource::from(source) != column.source {
                            continue;
                        }
                        if available_at_oods.is_some() || log_size != column.coefficient_log_size {
                            return Err(ArenaPlanError::InvalidProtocolGeometry(
                                "OODS retained source mapping is ambiguous or has the wrong log",
                            ));
                        }
                        let retained_for_decommit = commitment
                            .retained_evaluation_groups
                            .get(group)
                            .copied()
                            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                                "OODS retained source group is missing",
                            ))?;
                        let retained_for_numerator = commitment
                            .numerator_evaluation_groups
                            .get(group)
                            .copied()
                            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                                "OODS numerator source group is missing",
                            ))?;
                        available_at_oods = Some(retained_for_decommit || retained_for_numerator);
                    }
                }
                match available_at_oods {
                    Some(true) => Ok(OodsSourceKind::Evaluations),
                    Some(false) => Ok(OodsSourceKind::Coefficients),
                    None => Err(ArenaPlanError::InvalidProtocolGeometry(
                        "OODS source is absent from its commitment",
                    )),
                }
            })
            .collect()
    }

    pub fn quotient_numerator_source_kinds(
        &self,
    ) -> Result<Vec<QuotientNumeratorSourceKind>, ArenaPlanError> {
        self.oods
            .columns
            .iter()
            .map(|column| {
                if self.identity.quotient_numerator_source_policy
                    == QuotientNumeratorSourcePolicy::CoefficientsOnly
                    || column.shape_points.is_empty()
                {
                    return Ok(QuotientNumeratorSourceKind::Coefficients);
                }
                let tree = match column.source {
                    OpenedColumnSource::Preprocessed { .. } => {
                        return Ok(QuotientNumeratorSourceKind::Coefficients);
                    }
                    OpenedColumnSource::Trace {
                        purpose: BufferPurpose::BaseCoefficients,
                        ..
                    } => CommitmentTreeId::Base,
                    OpenedColumnSource::Trace {
                        purpose: BufferPurpose::InteractionCoefficients,
                        ..
                    } => CommitmentTreeId::Interaction,
                    OpenedColumnSource::Composition { .. } => CommitmentTreeId::Composition,
                    OpenedColumnSource::Trace { .. } => {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "quotient numerator source is not a committed coefficient column",
                        ));
                    }
                };
                let commitment = self
                    .commitments
                    .iter()
                    .find(|commitment| commitment.id == tree)
                    .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                        "quotient numerator source tree is missing",
                    ))?;
                let mut retained = None;
                for (group_index, (sources, logs)) in commitment
                    .grouped_column_sources
                    .iter()
                    .zip(&commitment.grouped_column_log_sizes)
                    .enumerate()
                {
                    for (&source, &log_size) in sources.iter().zip(logs) {
                        if OpenedColumnSource::from(source) != column.source {
                            continue;
                        }
                        if retained.is_some() || log_size != column.coefficient_log_size {
                            return Err(ArenaPlanError::InvalidProtocolGeometry(
                                "quotient numerator retained source mapping is ambiguous or has the wrong log",
                            ));
                        }
                        retained = Some(
                            commitment
                                .numerator_evaluation_groups
                                .get(group_index)
                                .copied()
                                .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                                    "quotient numerator retained source group is missing",
                                ))?,
                        );
                    }
                }
                match retained {
                    Some(true) => Ok(QuotientNumeratorSourceKind::Evaluation),
                    Some(false) => Ok(QuotientNumeratorSourceKind::Coefficients),
                    None => Err(ArenaPlanError::InvalidProtocolGeometry(
                        "quotient numerator source is absent from its commitment",
                    )),
                }
            })
            .collect()
    }

    pub fn quotient_numerator_topologies(
        &self,
    ) -> Result<Vec<QuotientNumeratorColumnTopology>, ArenaPlanError> {
        self.oods
            .quotient_numerator_topologies(&self.quotient_numerator_source_kinds()?)
    }

    pub fn oods_workspace_config(&self) -> OodsWorkspaceConfig {
        OodsWorkspaceConfig {
            lifting_log_size: self.lifting_log_size,
            mask_log_size: self.oods.mask_log_size,
        }
    }

    fn quotient_numerator_lde_tile_columns(&self) -> usize {
        match self.identity.resident_backend {
            ResidentBackend::LegacyResident => 1,
            ResidentBackend::ReplacementV1 => REPLACEMENT_NUMERATOR_LDE_TILE_COLUMNS,
        }
    }

    pub fn quotient_numerator_workspace_config(
        &self,
    ) -> Result<QuotientNumeratorWorkspaceConfig, ArenaPlanError> {
        Ok(QuotientNumeratorWorkspaceConfig {
            lifting_log_size: self.lifting_log_size,
            log_blowup_factor: self.identity.log_blowup_factor,
            max_lde_tile_words: checked_pow2(self.lifting_log_size)?
                .checked_mul(self.quotient_numerator_lde_tile_columns())
                .ok_or(ArenaPlanError::SizeOverflow)?,
        })
    }

    pub fn quotient_workspace_config(&self) -> QuotientWorkspaceConfig {
        QuotientWorkspaceConfig {
            lifting_log_size: self.lifting_log_size,
            log_blowup_factor: self.identity.log_blowup_factor,
        }
    }

    pub fn fri_workspace_config(&self) -> Result<FriWorkspaceConfig, ArenaPlanError> {
        Ok(FriWorkspaceConfig {
            fri: FriConfig {
                log_blowup_factor: self.identity.log_blowup_factor,
                log_last_layer_degree_bound: self.identity.log_last_layer_degree_bound,
                n_queries: self.n_queries,
                fold_step: self.identity.fri_fold_step,
            },
            circle_log_size: self.lifting_log_size,
            twiddle_log_size: self
                .lifting_log_size
                .checked_sub(1)
                .ok_or(ArenaPlanError::SizeOverflow)?,
        })
    }

    /// Canonical device decommit geometry: the four Starknet trace trees in
    /// PCS order followed by every committed FRI tree in transcript order.
    pub fn decommit_workspace_config(&self) -> Result<DecommitWorkspaceConfig, ArenaPlanError> {
        if self.commitments.len() != 4 || self.opened_tree_log_sizes.len() != 4 {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "decommitment requires exactly four trace trees",
            ));
        }
        let roles = [
            TraceTreeRole::Preprocessed,
            TraceTreeRole::Base,
            TraceTreeRole::Interaction,
            TraceTreeRole::Composition,
        ];
        let mut trees = self
            .commitments
            .iter()
            .zip(&self.opened_tree_log_sizes)
            .zip(roles)
            .map(|((commitment, &tree_query_log_size), role)| {
                if commitment.config.lifting_log_size != tree_query_log_size
                    || (role != TraceTreeRole::Preprocessed
                        && tree_query_log_size != self.lifting_log_size)
                {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "trace decommit query height disagrees with its commitment",
                    ));
                }
                if commitment.retained_evaluation_groups.len()
                    != commitment.grouped_column_log_sizes.len()
                {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "commitment opening policy does not match its groups",
                    ));
                }
                let groups = commitment
                    .grouped_column_log_sizes
                    .iter()
                    .enumerate()
                    .map(|(group_index, logs)| {
                        let retained = commitment
                            .retained_evaluation_groups
                            .get(group_index)
                            .copied()
                            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                                "commitment opening policy does not match its groups",
                            ))?;
                        if self.identity.decommit_strategy == DecommitStrategy::RecomputeQueriedLde
                            && retained
                        {
                            return Err(ArenaPlanError::InvalidProtocolGeometry(
                                "recompute-only opening policy retained an LDE group",
                            ));
                        }
                        Ok(TraceSourceGroupGeometry {
                            mode: if retained {
                                DecommitSourceMode::ResidentEvaluations
                            } else {
                                DecommitSourceMode::RecomputeQueriedLde
                            },
                            columns: logs
                                .iter()
                                .map(|&coefficient_log_size| {
                                    Ok(DecommitColumnGeometry {
                                        coefficient_log_size,
                                        evaluation_log_size: coefficient_log_size
                                            .checked_add(commitment.config.log_blowup_factor)
                                            .ok_or(ArenaPlanError::SizeOverflow)?,
                                    })
                                })
                                .collect::<Result<Vec<_>, ArenaPlanError>>()?,
                        })
                    })
                    .collect::<Result<Vec<_>, ArenaPlanError>>()?;
                Ok(DecommitTreeGeometry::Trace(TraceDecommitGeometry {
                    role,
                    tree_query_log_size,
                    leaf_log_size: commitment.config.lifting_log_size,
                    unretained_bottom_layers: commitment.config.unretained_bottom_layers,
                    groups,
                }))
            })
            .collect::<Result<Vec<_>, ArenaPlanError>>()?;

        let fri = fri_workspace_requirements(self.fri_workspace_config()?)
            .map_err(ArenaPlanError::Fri)?;
        for (fri_tree_index, tree) in fri.trees.iter().enumerate() {
            trees.push(DecommitTreeGeometry::Fri(FriDecommitGeometry {
                fri_tree_index: u32::try_from(fri_tree_index)
                    .map_err(|_| ArenaPlanError::SizeOverflow)?,
                evaluation_log_size: tree.evaluation_log_size,
                cumulative_fold: self
                    .lifting_log_size
                    .checked_sub(tree.evaluation_log_size)
                    .ok_or(ArenaPlanError::SizeOverflow)?,
                outgoing_fold_step: tree.outgoing_fold_step,
                log_rows_per_leaf: tree.log_rows_per_leaf,
            }));
        }
        Ok(DecommitWorkspaceConfig {
            query_log_size: self.lifting_log_size,
            n_queries: u32::try_from(self.n_queries).map_err(|_| ArenaPlanError::SizeOverflow)?,
            trees,
        })
    }

    /// Host adapter geometry sealed by the same source identities used by the
    /// prepared commit and decommit graphs.
    pub fn proof_assembly_shape(&self) -> Result<Blake2sProofAssemblyShape, ArenaPlanError> {
        let decommit = self.decommit_workspace_config()?;
        let mut trace_trees = Vec::with_capacity(4);
        for (tree_index, (commitment, geometry)) in self
            .commitments
            .iter()
            .zip(&decommit.trees)
            .take(4)
            .enumerate()
        {
            let DecommitTreeGeometry::Trace(geometry) = geometry else {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "trace decommit tree was replaced by FRI geometry",
                ));
            };
            let proof_columns = self
                .oods
                .columns
                .iter()
                .filter(|column| opened_source_tree(column.source) == tree_index)
                .collect::<Vec<_>>();
            let commit_to_proof_column = commitment
                .grouped_column_sources
                .iter()
                .flatten()
                .map(|&source| {
                    let source = OpenedColumnSource::from(source);
                    proof_columns
                        .iter()
                        .position(|column| column.source == source)
                        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                            "commit/decommit source is absent from OODS proof order",
                        ))
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()?;
            trace_trees.push(Blake2sTraceAssemblyShape {
                role: geometry.role,
                leaf_log_size: geometry.leaf_log_size,
                query_log_size: geometry.tree_query_log_size,
                oods_samples_per_column: proof_columns
                    .iter()
                    .map(|column| column.shape_points.len())
                    .collect(),
                commit_to_proof_column,
            });
        }
        let fri_trees = decommit
            .trees
            .iter()
            .skip(4)
            .map(|tree| match tree {
                DecommitTreeGeometry::Fri(tree) => Ok(Blake2sFriAssemblyShape {
                    evaluation_log_size: tree.evaluation_log_size,
                    cumulative_fold: tree.cumulative_fold,
                    outgoing_fold_step: tree.outgoing_fold_step,
                    log_rows_per_leaf: tree.log_rows_per_leaf,
                }),
                DecommitTreeGeometry::Trace(_) => Err(ArenaPlanError::InvalidProtocolGeometry(
                    "FRI decommit suffix contains a trace tree",
                )),
            })
            .collect::<Result<Vec<_>, ArenaPlanError>>()?;
        Ok(Blake2sProofAssemblyShape {
            query_log_size: self.lifting_log_size,
            n_queries: self.n_queries,
            trace_trees,
            fri_trees,
        })
    }

    fn validate(&self) -> Result<(), ArenaPlanError> {
        self.transcript.validate()?;
        if self.max_domain_log_size >= usize::BITS
            || self.lifting_log_size >= usize::BITS
            || self.lifting_log_size == 0
            || self.max_domain_log_size == 0
            || self.n_queries == 0
            || self.total_opened_columns == 0
            || self.proof_capacity_words == 0
            || self.commitments.is_empty()
            || self.opened_tree_log_sizes.is_empty()
            || self.fri_layer_log_sizes.is_empty()
            || self.identity.log_blowup_factor == 0
            || self.identity.fri_fold_step == 0
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "zero or overflowing PCS/FRI dimension",
            ));
        }
        if self.lifting_log_size > self.max_domain_log_size {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "FRI lifting domain exceeds the maximal commitment domain",
            ));
        }
        let commitment_max_domain = self
            .commitments
            .iter()
            .map(|commitment| commitment.config.lifting_log_size)
            .max()
            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "protocol has no commitment domain",
            ))?;
        if commitment_max_domain != self.max_domain_log_size {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "maximal commitment domain disagrees with commitment workspaces",
            ));
        }
        if self.identity.oods_topology_hash != self.oods.topology_hash() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS topology hash disagrees with protocol identity",
            ));
        }
        if !self
            .transcript
            .requirements
            .outputs
            .iter()
            .any(|requirement| {
                requirement.id == self.composition_random_coefficient_output
                    && requirement.min_words >= 4
            })
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "composition random-coefficient transcript output is missing or too small",
            ));
        }
        if self.oods.columns.len() != self.total_opened_columns {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS column count disagrees with opened-column count",
            ));
        }
        if self
            .oods
            .columns
            .iter()
            .any(|column| column.shape_points.len() != column.offset_points.len())
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS shape-point and offset-point arity differs",
            ));
        }
        let mut seen_sources = Vec::with_capacity(self.oods.columns.len());
        let mut previous_tree = 0u8;
        let mut preprocessed_ordinal = 0u32;
        let mut composition_ordinal = 0u32;
        for column in &self.oods.columns {
            if column.evaluation_log_size
                != column
                    .coefficient_log_size
                    .checked_add(self.identity.log_blowup_factor)
                    .ok_or(ArenaPlanError::SizeOverflow)?
                || (!column.shape_points.is_empty()
                    && column.evaluation_log_size > self.lifting_log_size)
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "OODS evaluation domain disagrees with coefficient log and PCS blowup",
                ));
            }
            if seen_sources.contains(&column.source) {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "duplicate OODS opened-column source",
                ));
            }
            seen_sources.push(column.source);
            let tree = match column.source {
                OpenedColumnSource::Preprocessed { ordinal } => {
                    if ordinal != preprocessed_ordinal {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "preprocessed OODS columns are not in ordinal order",
                        ));
                    }
                    preprocessed_ordinal = preprocessed_ordinal
                        .checked_add(1)
                        .ok_or(ArenaPlanError::SizeOverflow)?;
                    0
                }
                OpenedColumnSource::Trace {
                    purpose: BufferPurpose::BaseCoefficients,
                    ..
                } => 1,
                OpenedColumnSource::Trace {
                    purpose: BufferPurpose::InteractionCoefficients,
                    ..
                } => 2,
                OpenedColumnSource::Composition { ordinal } => {
                    if ordinal != composition_ordinal || ordinal >= 8 {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "composition OODS columns are not the eight canonical ordinals",
                        ));
                    }
                    composition_ordinal = composition_ordinal
                        .checked_add(1)
                        .ok_or(ArenaPlanError::SizeOverflow)?;
                    3
                }
                OpenedColumnSource::Trace { .. } => {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "OODS trace source is not a coefficient column",
                    ));
                }
            };
            if tree < previous_tree {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "OODS columns are not in four-tree proof order",
                ));
            }
            previous_tree = tree;
        }
        if preprocessed_ordinal == 0 || composition_ordinal != 8 {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS topology is missing preprocessed or composition columns",
            ));
        }
        let oods_source_kinds = self.oods_source_kinds()?;
        let oods_topologies = self.oods.column_topologies(&oods_source_kinds)?;
        let oods_requirements =
            oods_workspace_requirements(self.oods_workspace_config(), &oods_topologies)
                .map_err(ArenaPlanError::Oods)?;
        if oods_requirements.sample_count != self.oods.sample_count() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS workspace sample count disagrees with topology",
            ));
        }
        let sampled_values_words = oods_requirements.sampled_value_words;
        if !self
            .transcript
            .requirements
            .inputs
            .iter()
            .any(|requirement| {
                requirement.id == self.oods.sampled_values_input
                    && requirement.min_words == sampled_values_words
            })
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS sampled-value transcript input is missing or has the wrong size",
            ));
        }
        for output in [
            self.oods.point_parameter_output,
            self.oods.quotient_random_coefficient_output,
        ] {
            if !self
                .transcript
                .requirements
                .outputs
                .iter()
                .any(|requirement| requirement.id == output && requirement.min_words >= 4)
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "OODS transcript output is missing or too small",
                ));
            }
        }
        let numerator_topologies = self.quotient_numerator_topologies()?;
        let numerator_requirements = quotient_numerator_workspace_requirements(
            self.quotient_numerator_workspace_config()?,
            &numerator_topologies,
        )
        .map_err(ArenaPlanError::QuotientNumerator)?;
        if self.identity.quotient_numerator_schedule == QuotientNumeratorSchedule::HybridSingleWrite
        {
            let hybrid = quotient_numerator_hybrid_plan(
                self.quotient_numerator_workspace_config()?,
                &numerator_topologies,
            )
            .map_err(ArenaPlanError::QuotientNumeratorSchedule)?;
            if hybrid.requirements() != &numerator_requirements {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "hybrid numerator requirements drifted from the canonical workspace",
                ));
            }
        }
        let numerator_logs = numerator_requirements
            .groups
            .iter()
            .map(|group| group.log_size)
            .collect::<Vec<_>>();
        if numerator_logs != self.quotient.partial_numerator_log_sizes {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "quotient numerator groups disagree with prepared quotient sources",
            ));
        }
        quotient_workspace_requirements(
            self.quotient_workspace_config(),
            &self.quotient.partial_numerator_log_sizes,
        )
        .map_err(ArenaPlanError::Quotient)?;
        let expected_commitment_order = [
            CommitmentTreeId::Preprocessed,
            CommitmentTreeId::Base,
            CommitmentTreeId::Interaction,
            CommitmentTreeId::Composition,
        ];
        if self
            .commitments
            .iter()
            .map(|commitment| commitment.id)
            .collect::<Vec<_>>()
            != expected_commitment_order
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "commitments are not the canonical four Starknet trees",
            ));
        }
        // Validated after the canonical-tree-order check so a missing
        // Preprocessed tree reports as the tree-shape error, not as an
        // identity-count mismatch.
        let preprocessed_columns = self
            .commitments
            .iter()
            .find(|commitment| commitment.id == CommitmentTreeId::Preprocessed)
            .map(|commitment| {
                commitment
                    .grouped_column_sources
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>()
            })
            .unwrap_or(0);
        if self.preprocessed_column_ids.len() != preprocessed_columns
            || self.preprocessed_column_ids.iter().any(String::is_empty)
            || self
                .preprocessed_column_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != self.preprocessed_column_ids.len()
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "preprocessed column identities are incomplete or duplicated",
            ));
        }
        let mut ids = Vec::new();
        let mut committed_opened_sources = Vec::with_capacity(self.total_opened_columns);
        match (
            self.identity.direct_composition_retention_mode,
            &self.direct_composition_retention,
        ) {
            (DirectCompositionRetentionMode::Disabled, None)
                if self.identity.direct_composition_planner_key == 0
                    && self.identity.direct_composition_occurrence_bitmap_hash == 0
                    && self.identity.direct_composition_group_rounded_bytes == 0 => {}
            (DirectCompositionRetentionMode::ExactNative, Some(plan))
                if self.identity.commit_mode == ProgressiveCommitMode::DomainProgressive
                    && self.identity.direct_composition_planner_key == plan.cache_key => {}
            _ => {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "direct composition retention identity or mode drifted",
                ));
            }
        }
        for commitment in &self.commitments {
            if ids.contains(&commitment.id) {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "duplicate commitment geometry",
                ));
            }
            ids.push(commitment.id);
            match commitment.id {
                CommitmentTreeId::Preprocessed
                    if commitment.config.lifting_log_size > self.max_domain_log_size =>
                {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "preprocessed commitment exceeds the maximal commitment domain",
                    ));
                }
                CommitmentTreeId::Base
                | CommitmentTreeId::Interaction
                | CommitmentTreeId::Composition
                    if commitment.config.lifting_log_size != self.lifting_log_size =>
                {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "dynamic commitment height disagrees with the FRI query domain",
                    ));
                }
                _ => {}
            }
            BufferLifetime::new(commitment.created, ProofEpoch::Decommit)?;
            if commitment.grouped_column_sources.len() != commitment.grouped_column_log_sizes.len()
                || commitment.direct_composition_evaluation_groups.len()
                    != commitment.grouped_column_log_sizes.len()
                || commitment.retained_evaluation_groups.len()
                    != commitment.grouped_column_log_sizes.len()
                || commitment.numerator_evaluation_groups.len()
                    != commitment.grouped_column_log_sizes.len()
                || commitment
                    .grouped_column_sources
                    .iter()
                    .zip(&commitment.grouped_column_log_sizes)
                    .any(|(sources, logs)| sources.len() != logs.len())
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "commitment source/log group shape mismatch",
                ));
            }
            let mut seen_sources = Vec::new();
            for source in commitment.grouped_column_sources.iter().flatten() {
                if seen_sources.contains(source) {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "duplicate commitment column source",
                    ));
                }
                let valid_tree = matches!(
                    (commitment.id, source),
                    (
                        CommitmentTreeId::Preprocessed,
                        CommitmentColumnSource::Preprocessed { .. }
                    ) | (
                        CommitmentTreeId::Base,
                        CommitmentColumnSource::Trace {
                            purpose: BufferPurpose::BaseCoefficients,
                            ..
                        }
                    ) | (
                        CommitmentTreeId::Interaction,
                        CommitmentColumnSource::Trace {
                            purpose: BufferPurpose::InteractionCoefficients,
                            ..
                        }
                    ) | (
                        CommitmentTreeId::Composition,
                        CommitmentColumnSource::Composition { .. }
                    )
                );
                if !valid_tree {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "commitment source belongs to the wrong proof tree",
                    ));
                }
                match source {
                    CommitmentColumnSource::Preprocessed { .. } => {}
                    CommitmentColumnSource::Trace { purpose, .. }
                        if !matches!(
                            purpose,
                            BufferPurpose::BaseCoefficients
                                | BufferPurpose::InteractionCoefficients
                        ) =>
                    {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "commitment trace source is not a base/interaction coefficient column",
                        ));
                    }
                    CommitmentColumnSource::Composition { ordinal } if *ordinal >= 8 => {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "composition source ordinal exceeds the eight split coordinates",
                        ));
                    }
                    _ => {}
                }
                seen_sources.push(*source);
                committed_opened_sources.push((*source).into());
            }
            commitment_workspace_requirements(self.identity.commit_mode, commitment)
                .map_err(ArenaPlanError::Commit)?;
        }
        match &self.direct_composition_retention {
            None => {
                if self.commitments.iter().any(|commitment| {
                    commitment
                        .direct_composition_evaluation_groups
                        .iter()
                        .any(|&selected| selected)
                }) {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "disabled direct composition retention selected a group",
                    ));
                }
            }
            Some(plan) => {
                if self.identity.direct_composition_occurrence_bitmap_hash
                    != direct_bitmap_hash(&plan.direct_bitmap)
                {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "direct composition occurrence bitmap identity drifted",
                    ));
                }
                let mut expected = self
                    .commitments
                    .iter()
                    .map(|commitment| vec![false; commitment.grouped_column_log_sizes.len()])
                    .collect::<Vec<_>>();
                for binding in plan.bindings.iter().filter(|binding| binding.direct) {
                    let column = plan.columns.get(binding.column).ok_or(
                        ArenaPlanError::InvalidProtocolGeometry(
                            "direct composition binding column is out of range",
                        ),
                    )?;
                    let commitment = self
                        .commitments
                        .iter()
                        .position(|commitment| commitment.id == column.tree)
                        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                            "direct composition binding tree is missing",
                        ))?;
                    let selected = expected[commitment].get_mut(column.group).ok_or(
                        ArenaPlanError::InvalidProtocolGeometry(
                            "direct composition binding group is out of range",
                        ),
                    )?;
                    *selected = true;
                }
                if self
                    .commitments
                    .iter()
                    .zip(&expected)
                    .any(|(commitment, expected)| {
                        commitment.direct_composition_evaluation_groups != *expected
                    })
                {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "direct composition group closure has missing or extra groups",
                    ));
                }
            }
        }
        let mut expected_numerator = self
            .commitments
            .iter()
            .map(|commitment| vec![false; commitment.grouped_column_log_sizes.len()])
            .collect::<Vec<_>>();
        if self.identity.quotient_numerator_source_policy
            == QuotientNumeratorSourcePolicy::ReuseRetainedEvaluations
        {
            if self.identity.commit_mode != ProgressiveCommitMode::DomainProgressive {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "quotient numerator retention requires progressive commit",
                ));
            }
            for column in self
                .oods
                .columns
                .iter()
                .filter(|column| !column.shape_points.is_empty())
            {
                let tree = match column.source {
                    OpenedColumnSource::Preprocessed { .. } => continue,
                    OpenedColumnSource::Trace {
                        purpose: BufferPurpose::BaseCoefficients,
                        ..
                    } => CommitmentTreeId::Base,
                    OpenedColumnSource::Trace {
                        purpose: BufferPurpose::InteractionCoefficients,
                        ..
                    } => CommitmentTreeId::Interaction,
                    OpenedColumnSource::Composition { .. } => CommitmentTreeId::Composition,
                    OpenedColumnSource::Trace { .. } => {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "quotient numerator source is not a committed coefficient column",
                        ))
                    }
                };
                let commitment_index = self
                    .commitments
                    .iter()
                    .position(|commitment| commitment.id == tree)
                    .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                        "quotient numerator source tree is missing",
                    ))?;
                let commitment = &self.commitments[commitment_index];
                let mut matched = None;
                for (group, (sources, logs)) in commitment
                    .grouped_column_sources
                    .iter()
                    .zip(&commitment.grouped_column_log_sizes)
                    .enumerate()
                {
                    for (&source, &log) in sources.iter().zip(logs) {
                        if OpenedColumnSource::from(source) == column.source {
                            if matched.is_some() || log != column.coefficient_log_size {
                                return Err(ArenaPlanError::InvalidProtocolGeometry(
                                    "quotient numerator source mapping is ambiguous or has the wrong log",
                                ));
                            }
                            matched = Some(group);
                        }
                    }
                }
                let group = matched.ok_or(ArenaPlanError::InvalidProtocolGeometry(
                    "quotient numerator source is absent from its commitment",
                ))?;
                expected_numerator[commitment_index][group] = true;
            }
        }
        if self
            .commitments
            .iter()
            .zip(&expected_numerator)
            .any(|(commitment, expected)| commitment.numerator_evaluation_groups != *expected)
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "quotient numerator group closure has missing or extra groups",
            ));
        }
        let mut direct_bytes = 0usize;
        let mut numerator_bytes = 0usize;
        let mut union_bytes = 0usize;
        for commitment in &self.commitments {
            for group in 0..commitment.grouped_column_log_sizes.len() {
                let direct = commitment.direct_composition_evaluation_groups[group];
                let numerator = commitment.numerator_evaluation_groups[group];
                let decommit = commitment.retained_evaluation_groups[group];
                if direct || numerator || decommit {
                    let bytes = commitment_group_evaluation_bytes(commitment, group)?;
                    union_bytes = union_bytes
                        .checked_add(bytes)
                        .ok_or(ArenaPlanError::SizeOverflow)?;
                    if direct {
                        direct_bytes = direct_bytes
                            .checked_add(bytes)
                            .ok_or(ArenaPlanError::SizeOverflow)?;
                    }
                    if numerator {
                        numerator_bytes = numerator_bytes
                            .checked_add(bytes)
                            .ok_or(ArenaPlanError::SizeOverflow)?;
                    }
                }
            }
        }
        let identity_union_bytes = if self.direct_composition_retention.is_some()
            || self.identity.quotient_numerator_source_policy
                == QuotientNumeratorSourcePolicy::ReuseRetainedEvaluations
        {
            union_bytes
        } else {
            0
        };
        if self.identity.direct_composition_group_rounded_bytes != direct_bytes
            || self.identity.numerator_evaluation_group_rounded_bytes != numerator_bytes
            || self.identity.retained_evaluation_union_bytes != identity_union_bytes
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "retained evaluation physical byte identity drifted",
            ));
        }
        if committed_opened_sources.len() != self.oods.columns.len()
            || self
                .oods
                .columns
                .iter()
                .any(|column| !committed_opened_sources.contains(&column.source))
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "commitment sources do not cover the exact opened-column set",
            ));
        }
        if self
            .opened_tree_log_sizes
            .first()
            .is_some_and(|&log_size| log_size > self.max_domain_log_size)
            || self
                .opened_tree_log_sizes
                .iter()
                .skip(1)
                .any(|&log_size| log_size != self.lifting_log_size)
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "opened commitment tree disagrees with its query domain",
            ));
        }
        let mut previous = None;
        for &log_size in &self.fri_layer_log_sizes {
            if log_size > self.lifting_log_size
                || previous.is_some_and(|previous| log_size > previous)
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "FRI tree log sizes increase or exceed the lifting domain",
                ));
            }
            previous = Some(log_size);
        }
        let fri = fri_workspace_requirements(self.fri_workspace_config()?)
            .map_err(ArenaPlanError::Fri)?;
        let prepared_logs: Vec<_> = fri
            .trees
            .iter()
            .map(|tree| {
                tree.layers_bottom_up
                    .first()
                    .expect("prepared FRI trees always contain a leaf")
                    .log_size
            })
            .collect();
        if prepared_logs != self.fri_layer_log_sizes {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "FRI leaf schedule disagrees with prepared workspace",
            ));
        }
        let decommit = self.decommit_workspace_config()?;
        decommit_workspace_requirements(decommit).map_err(ArenaPlanError::Decommit)?;
        self.proof_assembly_shape()?;
        Ok(())
    }

    /// Stable protocol-topology key. Together with `ProofShapeKey` this covers
    /// every arena and CUDA-graph dimension owned by this plan.
    pub fn key(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        };
        feed(b"stwo-cairo-protocol-geometry-v13\0");
        feed(&self.identity.pow_bits.to_le_bytes());
        feed(&self.identity.log_blowup_factor.to_le_bytes());
        feed(&self.identity.log_last_layer_degree_bound.to_le_bytes());
        feed(&self.identity.fri_fold_step.to_le_bytes());
        feed(&self.identity.channel_tag.to_le_bytes());
        feed(&self.identity.relation_graph_hash.to_le_bytes());
        feed(&self.identity.preprocessed_binding_hash.to_le_bytes());
        feed(&self.identity.oods_topology_hash.to_le_bytes());
        feed(&self.identity.composition_plan_hash.to_le_bytes());
        feed(&self.identity.kernel_manifest_hash.to_le_bytes());
        feed(&[self.identity.interpolation_mode as u8]);
        feed(&[u8::from(self.identity.blake2s_interior_fused)]);
        feed(&[self.identity.composition_launch_mode as u8]);
        feed(&[self.identity.relation_tail_mode as u8]);
        feed(&[self.identity.fri_fold_launch_mode as u8]);
        feed(&[self.identity.witness_feed_launch_mode as u8]);
        feed(&[self.identity.resident_backend as u8]);
        feed(&(self.quotient_numerator_lde_tile_columns() as u64).to_le_bytes());
        feed(&[self.identity.quotient_numerator_schedule as u8]);
        feed(&[self.identity.quotient_numerator_source_policy as u8]);
        feed(&[self.identity.commit_mode as u8]);
        feed(&[self.identity.direct_composition_retention_mode as u8]);
        feed(&self.identity.direct_composition_planner_key.to_le_bytes());
        feed(
            &self
                .identity
                .direct_composition_occurrence_bitmap_hash
                .to_le_bytes(),
        );
        feed(&(self.identity.direct_composition_group_rounded_bytes as u64).to_le_bytes());
        feed(&(self.identity.retained_evaluation_union_bytes as u64).to_le_bytes());
        if self.identity.quotient_numerator_source_policy
            == QuotientNumeratorSourcePolicy::ReuseRetainedEvaluations
        {
            feed(&(self.identity.numerator_evaluation_group_rounded_bytes as u64).to_le_bytes());
        }
        for identity in &self.preprocessed_column_ids {
            feed(&(identity.len() as u64).to_le_bytes());
            feed(identity.as_bytes());
        }
        feed(&[self.identity.decommit_strategy as u8]);
        feed(&self.max_domain_log_size.to_le_bytes());
        feed(&self.lifting_log_size.to_le_bytes());
        feed(&(self.n_queries as u64).to_le_bytes());
        feed(&(self.total_opened_columns as u64).to_le_bytes());
        feed(&(self.proof_capacity_words as u64).to_le_bytes());
        feed(&(self.quotient.partial_numerator_count() as u64).to_le_bytes());
        for log_size in &self.quotient.partial_numerator_log_sizes {
            feed(&log_size.to_le_bytes());
        }
        feed(&self.oods.topology_hash().to_le_bytes());
        feed(&self.composition_random_coefficient_output.0.to_le_bytes());
        feed(&self.transcript.schedule_key.to_le_bytes());
        let transcript = &self.transcript.requirements;
        for words in [
            transcript.state_words,
            transcript.boundary_snapshot_words,
            transcript.input_snapshot_words,
            transcript.input_snapshot_used_words,
            transcript.output_snapshot_words,
            transcript.output_snapshot_used_words,
        ] {
            feed(&(words as u64).to_le_bytes());
        }
        for requirement in &transcript.inputs {
            feed(&requirement.id.0.to_le_bytes());
            feed(&(requirement.min_words as u64).to_le_bytes());
        }
        for requirement in &transcript.outputs {
            feed(&requirement.id.0.to_le_bytes());
            feed(&(requirement.min_words as u64).to_le_bytes());
        }
        for commitment in &self.commitments {
            match commitment.id {
                CommitmentTreeId::Preprocessed => feed(&[0]),
                CommitmentTreeId::Base => feed(&[1]),
                CommitmentTreeId::Interaction => feed(&[2]),
                CommitmentTreeId::Composition => feed(&[3]),
                CommitmentTreeId::Fri(layer) => feed(&[4, layer]),
            }
            feed(&[commitment.created as u8]);
            feed(&commitment.config.log_blowup_factor.to_le_bytes());
            feed(&commitment.config.lifting_log_size.to_le_bytes());
            feed(&commitment.config.unretained_bottom_layers.to_le_bytes());
            feed(&commitment.config.max_fused_tail_levels.to_le_bytes());
            feed(&(commitment.retained_evaluation_groups.len() as u64).to_le_bytes());
            for &retained in &commitment.retained_evaluation_groups {
                feed(&[u8::from(retained)]);
            }
            feed(&(commitment.direct_composition_evaluation_groups.len() as u64).to_le_bytes());
            for &retained in &commitment.direct_composition_evaluation_groups {
                feed(&[u8::from(retained)]);
            }
            if self.identity.quotient_numerator_source_policy
                == QuotientNumeratorSourcePolicy::ReuseRetainedEvaluations
            {
                feed(&(commitment.numerator_evaluation_groups.len() as u64).to_le_bytes());
                for &retained in &commitment.numerator_evaluation_groups {
                    feed(&[u8::from(retained)]);
                }
            }
            for group in &commitment.grouped_column_log_sizes {
                feed(&(group.len() as u64).to_le_bytes());
                for log_size in group {
                    feed(&log_size.to_le_bytes());
                }
            }
            for group in &commitment.grouped_column_sources {
                for source in group {
                    match source {
                        CommitmentColumnSource::Preprocessed { ordinal } => {
                            feed(&[0]);
                            feed(&ordinal.to_le_bytes());
                        }
                        CommitmentColumnSource::Trace {
                            component,
                            part,
                            purpose,
                            ordinal,
                        } => {
                            feed(&[1]);
                            feed(component.as_bytes());
                            feed(&[0]);
                            match part {
                                TracePartId::Main => feed(&[0]),
                                TracePartId::MemoryBig(index) => {
                                    feed(&[1]);
                                    feed(&index.to_le_bytes());
                                }
                                TracePartId::MemorySmall => feed(&[2]),
                            }
                            feed(&[match purpose {
                                BufferPurpose::BaseCoefficients => 0,
                                BufferPurpose::InteractionCoefficients => 1,
                                _ => u8::MAX,
                            }]);
                            feed(&ordinal.to_le_bytes());
                        }
                        CommitmentColumnSource::Composition { ordinal } => {
                            feed(&[2]);
                            feed(&ordinal.to_le_bytes());
                        }
                    }
                }
            }
            if self.identity.commit_mode == ProgressiveCommitMode::DomainProgressive {
                let ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements) =
                    commitment_workspace_requirements(self.identity.commit_mode, commitment)
                        .expect("validated commitment geometry")
                else {
                    unreachable!();
                };
                feed(&requirements.leaves.plan.cache_key.to_le_bytes());
            }
        }
        feed(&(self.opened_tree_log_sizes.len() as u64).to_le_bytes());
        for log_size in &self.opened_tree_log_sizes {
            feed(&log_size.to_le_bytes());
        }
        feed(&(self.fri_layer_log_sizes.len() as u64).to_le_bytes());
        for log_size in &self.fri_layer_log_sizes {
            feed(&log_size.to_le_bytes());
        }
        hash
    }
}

fn opened_source_tree(source: OpenedColumnSource) -> usize {
    match source {
        OpenedColumnSource::Preprocessed { .. } => 0,
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::BaseCoefficients,
            ..
        } => 1,
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::InteractionCoefficients,
            ..
        } => 2,
        OpenedColumnSource::Composition { .. } => 3,
        OpenedColumnSource::Trace { .. } => usize::MAX,
    }
}

fn feed_hash(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn execution_tables_protocol_key(mut protocol_key: u64, geometry: ExecutionTableGeometry) -> u64 {
    feed_hash(&mut protocol_key, b"resident-execution-tables-v1\0");
    for dimension in [
        geometry.n_addrs,
        geometry.n_big,
        geometry.n_small,
        geometry.public_memory_entries,
    ] {
        feed_hash(&mut protocol_key, &(dimension as u64).to_le_bytes());
    }
    protocol_key
}

fn graph_a_multiplicity_protocol_key(mut protocol_key: u64, topology_hash: u64) -> u64 {
    feed_hash(&mut protocol_key, b"resident-graph-a-multiplicity-v1\0");
    feed_hash(&mut protocol_key, &topology_hash.to_le_bytes());
    protocol_key
}

fn relation_protocol_key(mut protocol_key: u64, mode: RelationLaunchMode) -> u64 {
    feed_hash(&mut protocol_key, b"resident-relation-launch-v1\0");
    feed_hash(&mut protocol_key, &(mode as u32).to_le_bytes());
    protocol_key
}

fn feed_opened_source(hash: &mut u64, source: OpenedColumnSource) {
    match source {
        OpenedColumnSource::Preprocessed { ordinal } => {
            feed_hash(hash, &[0]);
            feed_hash(hash, &ordinal.to_le_bytes());
        }
        OpenedColumnSource::Trace {
            component,
            part,
            purpose,
            ordinal,
        } => {
            feed_hash(hash, &[1]);
            feed_hash(hash, component.as_bytes());
            feed_hash(hash, &[0]);
            match part {
                TracePartId::Main => feed_hash(hash, &[0]),
                TracePartId::MemoryBig(index) => {
                    feed_hash(hash, &[1]);
                    feed_hash(hash, &index.to_le_bytes());
                }
                TracePartId::MemorySmall => feed_hash(hash, &[2]),
            }
            feed_hash(
                hash,
                &[match purpose {
                    BufferPurpose::BaseCoefficients => 0,
                    BufferPurpose::InteractionCoefficients => 1,
                    _ => u8::MAX,
                }],
            );
            feed_hash(hash, &ordinal.to_le_bytes());
        }
        OpenedColumnSource::Composition { ordinal } => {
            feed_hash(hash, &[2]);
            feed_hash(hash, &ordinal.to_le_bytes());
        }
    }
}

#[derive(Clone, Debug)]
struct LogicalPreprocessedColumn {
    identity: String,
    ordinal: u32,
    log_size: u32,
    evaluations: Option<LogicalBufferId>,
    coefficients: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalPreprocessedInterpolationBatch {
    log_size: u32,
    column_ordinals: Vec<u32>,
    coefficient_pointers: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalPreprocessedWorkspace {
    columns: Vec<LogicalPreprocessedColumn>,
    interpolation_batches: Vec<LogicalPreprocessedInterpolationBatch>,
    inverse_twiddles: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalOodsColumn {
    geometry: OodsColumnGeometry,
    coefficients: LogicalBufferId,
    source_kind: OodsSourceKind,
    source_binding: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalCompositionTraceColumn {
    source: OpenedColumnSource,
    log_size: u32,
    coefficients: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalCompositionExtParams {
    component: &'static str,
    instance: usize,
    sources: Vec<CompositionExtParamSource>,
    values: Vec<SecureField>,
    binding: Option<LogicalBufferId>,
}

#[derive(Clone, Debug)]
struct LogicalCompositionWorkspace {
    plan: CompositionPlan,
    requirements: CompositionWorkspaceRequirements,
    trace_trees: Vec<Vec<LogicalCompositionTraceColumn>>,
    random_coefficient: LogicalBufferId,
    forward_twiddles: LogicalBufferId,
    inverse_twiddles: LogicalBufferId,
    ext_params: Vec<LogicalCompositionExtParams>,
    descriptors: LogicalBufferId,
    lde_tile: LogicalBufferId,
    accumulators: LogicalBufferId,
    random_coefficient_powers: LogicalBufferId,
    composition_coefficients: [LogicalBufferId; 8],
    direct_retention: Option<DirectCompositionRetentionPlan>,
    direct_bindings: Vec<LogicalDirectCompositionBinding>,
}

#[derive(Clone, Debug)]
struct LogicalDirectCompositionBinding {
    consumer: usize,
    plan_column: usize,
    source: OpenedColumnSource,
    tree: CommitmentTreeId,
    proof_column: usize,
    group: usize,
    column_in_group: usize,
    canonical_column: usize,
    evaluation_log_size: u32,
    evaluation: Option<LogicalBufferId>,
}

#[derive(Clone, Debug)]
struct LogicalOodsWorkspace {
    config: OodsWorkspaceConfig,
    requirements: OodsWorkspaceRequirements,
    columns: Vec<LogicalOodsColumn>,
    oods_point_parameter: LogicalBufferId,
    source_pointers: LogicalBufferId,
    offset_points: LogicalBufferId,
    fold_counts: LogicalBufferId,
    output_indices: LogicalBufferId,
    folding_factors: LogicalBufferId,
    scratch_a: LogicalBufferId,
    scratch_b: LogicalBufferId,
    sample_points: LogicalBufferId,
    sampled_values: LogicalBufferId,
    evaluation_points: LogicalBufferId,
    barycentric_numerators: LogicalBufferId,
    barycentric_weights: LogicalBufferId,
    barycentric_scales: LogicalBufferId,
    barycentric_partials: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalQuotientNumeratorColumn {
    source: OpenedColumnSource,
    topology: QuotientNumeratorColumnTopology,
    coefficients: LogicalBufferId,
    numerator_source: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalQuotientNumeratorWorkspace {
    schedule: QuotientNumeratorSchedule,
    config: QuotientNumeratorWorkspaceConfig,
    requirements: QuotientNumeratorWorkspaceRequirements,
    columns: Vec<LogicalQuotientNumeratorColumn>,
    oods_sample_points: LogicalBufferId,
    oods_sampled_values: LogicalBufferId,
    random_coefficient: LogicalBufferId,
    sample_points_destination: LogicalBufferId,
    first_linear_terms_destination: LogicalBufferId,
    destinations: Vec<LogicalQuotientNumeratorSource>,
    forward_twiddles: LogicalBufferId,
    runtime_terms: LogicalBufferId,
    group_term_indices: LogicalBufferId,
    group_offsets: LogicalBufferId,
    line_coefficients: LogicalBufferId,
    term_points: LogicalBufferId,
    batch_terms: LogicalBufferId,
    batch_group_offsets: LogicalBufferId,
    batch_source_ptrs: LogicalBufferId,
    output_ptrs: LogicalBufferId,
    output_log_sizes: LogicalBufferId,
    coefficient_ptrs: Option<LogicalBufferId>,
    coefficient_sizes: Option<LogicalBufferId>,
    coefficient_output_ptrs: Option<LogicalBufferId>,
    lde_tile: Option<LogicalBufferId>,
}

#[derive(Clone, Debug)]
struct LogicalCommitWorkspace {
    id: CommitmentTreeId,
    interpolation_mode: InterpolationLaunchMode,
    config: CommitWorkspaceConfig,
    grouped_column_log_sizes: Vec<Vec<u32>>,
    grouped_column_sources: Vec<Vec<CommitmentColumnSource>>,
    requirements: ModeAwareCommitWorkspaceRequirements,
    twiddles: LogicalBufferId,
    leaf_state: LogicalBufferId,
    merkle_scratch: Option<LogicalBufferId>,
    retained_layers: Vec<LogicalBufferId>,
    tail_level_ptrs: Option<LogicalBufferId>,
    tail_outputs: Vec<LogicalBufferId>,
    retained_evaluations: Vec<Option<Vec<LogicalBufferId>>>,
    decommit_evaluation_groups: Vec<bool>,
    direct_composition_evaluation_groups: Vec<bool>,
    numerator_evaluation_groups: Vec<bool>,
    leaf_workspace: LogicalCommitLeafWorkspace,
    interpolation_batches: Vec<LogicalInterpolationBatch>,
}

#[derive(Clone, Debug)]
enum LogicalCommitLeafWorkspace {
    FullLifting {
        lde_tile: LogicalBufferId,
        groups: Vec<LogicalCommitGroupSlots>,
    },
    DomainProgressive {
        lde_scratch: Option<LogicalBufferId>,
        state_ping: LogicalBufferId,
        state_pong: Option<LogicalBufferId>,
        batches: Vec<LogicalCommitBatchSlots>,
    },
}

#[derive(Clone, Debug)]
struct LogicalInterpolationBatch {
    log_size: u32,
    sources: Vec<CommitmentColumnSource>,
    input_pointers: LogicalBufferId,
    output_pointers: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalCommitGroupSlots {
    column_ptrs: LogicalBufferId,
    column_log_sizes: LogicalBufferId,
    batches: Vec<LogicalCommitBatchSlots>,
}

#[derive(Clone, Copy, Debug)]
struct LogicalCommitBatchSlots {
    coefficient_ptrs: LogicalBufferId,
    coefficient_sizes: LogicalBufferId,
    output_ptrs: LogicalBufferId,
}

fn allocate_commit_batch(
    logical: &mut Vec<LogicalBuffer>,
    ordinal: &mut impl FnMut() -> Result<u32, ArenaPlanError>,
    batch: &CommitBatchRequirements,
    lifetime: BufferLifetime,
) -> Result<LogicalCommitBatchSlots, ArenaPlanError> {
    allocate_commit_batch_words(
        logical,
        ordinal,
        batch.coefficient_pointer_words,
        batch.coefficient_size_words,
        batch.output_pointer_words,
        lifetime,
    )
}

fn allocate_progressive_commit_batch(
    logical: &mut Vec<LogicalBuffer>,
    ordinal: &mut impl FnMut() -> Result<u32, ArenaPlanError>,
    batch: &ProgressiveBatchRequirements,
    lifetime: BufferLifetime,
) -> Result<LogicalCommitBatchSlots, ArenaPlanError> {
    allocate_commit_batch_words(
        logical,
        ordinal,
        batch.coefficient_pointer_words,
        batch.coefficient_size_words,
        batch.output_pointer_words,
        lifetime,
    )
}

fn allocate_commit_batch_words(
    logical: &mut Vec<LogicalBuffer>,
    ordinal: &mut impl FnMut() -> Result<u32, ArenaPlanError>,
    coefficient_pointer_words: usize,
    coefficient_size_words: usize,
    output_pointer_words: usize,
    lifetime: BufferLifetime,
) -> Result<LogicalCommitBatchSlots, ArenaPlanError> {
    Ok(LogicalCommitBatchSlots {
        coefficient_ptrs: push_buffer_id(
            logical,
            None,
            None,
            BufferPurpose::CommitCoefficientPointers,
            ordinal()?,
            coefficient_pointer_words,
            lifetime,
        )?,
        coefficient_sizes: push_buffer_id(
            logical,
            None,
            None,
            BufferPurpose::CommitCoefficientSizes,
            ordinal()?,
            coefficient_size_words,
            lifetime,
        )?,
        output_ptrs: push_buffer_id(
            logical,
            None,
            None,
            BufferPurpose::CommitOutputPointers,
            ordinal()?,
            output_pointer_words,
            lifetime,
        )?,
    })
}

#[derive(Clone, Debug)]
struct LogicalFriWorkspace {
    config: FriWorkspaceConfig,
    requirements: FriWorkspaceRequirements,
    twiddles: LogicalBufferId,
    input_values: LogicalBufferId,
    evaluation_ping: LogicalBufferId,
    evaluation_pong: LogicalBufferId,
    input_coordinate_ptrs: LogicalBufferId,
    ping_coordinate_ptrs: LogicalBufferId,
    pong_coordinate_ptrs: LogicalBufferId,
    retained_tree_evaluations: Vec<LogicalBufferId>,
    retained_tree_coordinate_ptrs: Vec<LogicalBufferId>,
    folding_challenges: Vec<LogicalBufferId>,
    trees: Vec<Vec<LogicalBufferId>>,
}

#[derive(Clone, Debug)]
struct LogicalFinalFriPowWorkspace {
    final_requirements: FriFinalWorkspaceRequirements,
    final_coefficients: LogicalBufferId,
    final_degree_error: LogicalBufferId,
    pow_requirements: Blake2sPowWorkspaceRequirements,
    interaction_pow_bits: u32,
    interaction_pow_best_nonce: LogicalBufferId,
    interaction_pow_completed_blocks: LogicalBufferId,
    interaction_pow_prefix_digest: LogicalBufferId,
    query_pow_bits: u32,
    query_pow_best_nonce: LogicalBufferId,
    query_pow_completed_blocks: LogicalBufferId,
    query_pow_prefix_digest: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalTraceDecommitSlots {
    evaluation_ptrs: LogicalBufferId,
    evaluation_log_sizes: LogicalBufferId,
    retained_layers_by_log: LogicalBufferId,
    sparse_level_offsets: LogicalBufferId,
    groups: Vec<LogicalTraceDecommitGroupSlots>,
}

#[derive(Clone, Debug)]
struct LogicalTraceDecommitGroupSlots {
    coefficient_ptrs: Option<LogicalBufferId>,
    coefficient_sizes: Option<LogicalBufferId>,
    lde_output_ptrs: Option<LogicalBufferId>,
    lde_tile: Option<LogicalBufferId>,
}

#[derive(Clone, Debug)]
struct LogicalFriDecommitSlots {
    coordinate_ptrs: LogicalBufferId,
    retained_layers_by_log: LogicalBufferId,
}

#[derive(Clone, Debug)]
enum LogicalDecommitTreeSlots {
    Trace(LogicalTraceDecommitSlots),
    Fri(LogicalFriDecommitSlots),
}

#[derive(Clone, Debug)]
struct LogicalDecommitWorkspace {
    config: DecommitWorkspaceConfig,
    requirements: DecommitWorkspaceRequirements,
    proof_shape: Blake2sProofAssemblyShape,
    raw_queries: LogicalBufferId,
    lde_twiddles: LogicalBufferId,
    unique_queries: LogicalBufferId,
    mapped_queries: LogicalBufferId,
    walk_queries: LogicalBufferId,
    walk_scratch: LogicalBufferId,
    expanded_positions: LogicalBufferId,
    sparse_indices: LogicalBufferId,
    sparse_hashes: LogicalBufferId,
    counts: LogicalBufferId,
    proof_bundle_layout: ResidentProofBundleLayout,
    proof_bundle: LogicalBufferId,
    trees: Vec<LogicalDecommitTreeSlots>,
}

#[derive(Clone, Debug)]
struct LogicalQuotientNumeratorSource {
    log_size: u32,
    coordinates: [LogicalBufferId; 4],
}

#[derive(Clone, Debug)]
struct LogicalQuotientWorkspace {
    config: QuotientWorkspaceConfig,
    requirements: QuotientWorkspaceRequirements,
    forward_twiddles: LogicalBufferId,
    inverse_subdomain_twiddles: LogicalBufferId,
    partial_numerators: Vec<LogicalQuotientNumeratorSource>,
    sample_points: LogicalBufferId,
    first_linear_terms: LogicalBufferId,
    partial_log_sizes: LogicalBufferId,
    partial_coordinate_ptrs: LogicalBufferId,
    subdomain_coordinate_ptrs: LogicalBufferId,
    output_coordinate_ptrs: LogicalBufferId,
    coefficient_sizes: LogicalBufferId,
    subdomain_values: LogicalBufferId,
    output_values: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalTranscriptWorkspace {
    schedule_key: u64,
    requirements: Blake2sTranscriptRequirements,
    state: LogicalBufferId,
    boundary_snapshots: LogicalBufferId,
    input_snapshots: LogicalBufferId,
    output_snapshots: LogicalBufferId,
    inputs: Vec<(TranscriptInputId, LogicalBufferId)>,
    outputs: Vec<(TranscriptOutputId, LogicalBufferId)>,
}

/// Exact compact execution-memory shape owned by one resident workspace.
/// Contents may change between proofs, but a shape change must select a new
/// arena/cache key because every captured limb destination has a fixed extent.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionTableGeometry {
    pub n_addrs: usize,
    pub n_big: usize,
    pub n_small: usize,
    pub public_memory_entries: usize,
}

impl ExecutionTableGeometry {
    pub const fn new(n_addrs: usize, n_big: usize, n_small: usize) -> Self {
        Self {
            n_addrs,
            n_big,
            n_small,
            public_memory_entries: 0,
        }
    }

    pub const fn with_public_memory_entries(mut self, entries: usize) -> Self {
        self.public_memory_entries = entries;
        self
    }
}

#[derive(Clone, Debug)]
struct LogicalExecutionTablesWorkspace {
    requirements: ExecutionTablesWorkspaceRequirements,
    raw_addr_to_id: LogicalBufferId,
    raw_f252_words: LogicalBufferId,
    raw_small_words: LogicalBufferId,
    big_limbs: Vec<LogicalBufferId>,
    small_limbs: Vec<LogicalBufferId>,
    table_pointers: LogicalBufferId,
    table_strides: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalEcOpWorkspace {
    requirements: EcOpWorkspaceRequirements,
    trace_columns: Vec<LogicalBufferId>,
    lookup_words: LogicalBufferId,
    partial_input_columns: Vec<LogicalBufferId>,
    segment_start: LogicalBufferId,
    address_counts: LogicalBufferId,
    big_counts: LogicalBufferId,
    small_counts: LogicalBufferId,
    range_check_8_counts: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalWitnessInputGather {
    requirements: WitnessInputGatherRequirements,
    producers: Vec<&'static str>,
    sources: Vec<LogicalBufferId>,
    source_pointers: LogicalBufferId,
    descriptors: LogicalBufferId,
    output_pointers: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalWitnessInputSeed {
    requirements: WitnessInputSeedRequirements,
    scalar_values: LogicalBufferId,
    output_pointers: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalWitnessInputCompact {
    requirements: WitnessInputCompactRequirements,
    sources: Vec<LogicalBufferId>,
    source_pointers: LogicalBufferId,
    descriptors: LogicalBufferId,
    output_pointers: LogicalBufferId,
    tuple_scratch: LogicalBufferId,
    sort_keys_a: LogicalBufferId,
    sort_keys_b: LogicalBufferId,
    sort_indices_a: LogicalBufferId,
    sort_indices_b: LogicalBufferId,
    run_heads: LogicalBufferId,
    run_positions: LogicalBufferId,
    n_unique: LogicalBufferId,
    sort_temp: LogicalBufferId,
    scan_temp: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalWitnessComponent {
    component: &'static str,
    part: TracePartId,
    n_real_rows: usize,
    blake_g_fused: bool,
    native_input_producer: Option<&'static str>,
    program: WitnessProgram,
    requirements: WitnessWorkspaceRequirements,
    input_columns: Vec<LogicalBufferId>,
    input_pointers: LogicalBufferId,
    execution_table_pointers: Option<LogicalBufferId>,
    execution_table_strides: Option<LogicalBufferId>,
    output_columns: Vec<LogicalBufferId>,
    output_pointers: LogicalBufferId,
    multiplicity_columns: Vec<LogicalBufferId>,
    multiplicity_pointers: LogicalBufferId,
    multiplicity_dummy: Option<LogicalBufferId>,
    lookup_words: LogicalBufferId,
    sub_words: LogicalBufferId,
    input_gather: Option<LogicalWitnessInputGather>,
    input_seed: Option<LogicalWitnessInputSeed>,
    input_compact: Option<LogicalWitnessInputCompact>,
}

#[derive(Clone, Debug, Default)]
struct LogicalWitnessWorkspace {
    components: Vec<LogicalWitnessComponent>,
}

#[derive(Clone, Debug)]
struct LogicalGenericRecordedMultiplicityFeed {
    plan: crate::multiplicity_pipeline::PlannedRecordedMultiplicityFeed,
    source: LogicalBufferId,
    descriptors: LogicalBufferId,
    lut_tables: Vec<LogicalBufferId>,
    lut_pointers: LogicalBufferId,
    multiplicity_destinations: Vec<LogicalBufferId>,
    multiplicity_pointers: LogicalBufferId,
}

#[derive(Clone, Debug)]
enum LogicalRecordedMultiplicityFeed {
    Generic(LogicalGenericRecordedMultiplicityFeed),
    BlakeGFused {
        plan: crate::multiplicity_pipeline::PlannedRecordedMultiplicityFeed,
        lut_tables: [LogicalBufferId; 4],
        multiplicity_destinations: [LogicalBufferId; 5],
    },
}

#[derive(Clone, Debug)]
struct LogicalFixedTableMaterializer {
    plan: crate::multiplicity_pipeline::PlannedFixedMultiplicity,
    sources: Vec<LogicalFixedTableSource>,
    multiplicity: LogicalBufferId,
    source_pointers: Option<LogicalBufferId>,
    multiplicity_pointers: LogicalBufferId,
    trace_multiplicity_columns: LogicalBufferId,
    trace_outputs: Vec<LogicalBufferId>,
    trace_output_pointers: LogicalBufferId,
    lookup_descriptors: LogicalBufferId,
    lookup_output: LogicalBufferId,
    lookup_output_pointers: LogicalBufferId,
}

#[derive(Clone, Copy, Debug)]
enum LogicalFixedTableSource {
    Arena(LogicalBufferId),
    RegisteredPedersen18 { column: usize },
}

#[derive(Clone, Debug)]
struct LogicalGraphAMultiplicityWorkspace {
    topology_hash: u64,
    coverage_gaps: Vec<FixedMultiplicityCoverageGap>,
    blockers: Vec<MultiplicityFeedBlocker>,
    multiplicities: Vec<(&'static str, LogicalBufferId)>,
    clear_requirements: WitnessFeedClearWorkspaceRequirements,
    clear_pointers: LogicalBufferId,
    clear_lengths: LogicalBufferId,
    feeds: Vec<LogicalRecordedMultiplicityFeed>,
    public_memory_seed: Option<LogicalGenericRecordedMultiplicityFeed>,
    fixed_tables: Vec<LogicalFixedTableMaterializer>,
    memory_traces: Option<LogicalMemoryBaseTraces>,
}

#[derive(Clone, Debug)]
struct LogicalMemoryTracePart {
    part: TracePartId,
    source_offset: usize,
    row_count: usize,
    outputs: Vec<LogicalBufferId>,
}

#[derive(Clone, Debug)]
struct LogicalMemoryBaseTraces {
    plan: PlannedMemoryBaseTraces,
    address_outputs: Vec<LogicalBufferId>,
    big_parts: Vec<LogicalMemoryTracePart>,
    small_part: LogicalMemoryTracePart,
    rc99_lut: LogicalBufferId,
    rc99_counts: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalRelationInstanceSlots {
    source_pointers: LogicalBufferId,
    output_pointers: LogicalBufferId,
    output_coordinates: Vec<LogicalBufferId>,
    denominators: LogicalBufferId,
    claimed_sum: LogicalBufferId,
}

#[derive(Clone, Debug)]
struct LogicalRelationWorkspace {
    execution: RelationExecutionPlan,
    source_plan: Vec<RelationInstanceSourcePlan>,
    launch_mode: RelationLaunchMode,
    requirements: RelationGraphRequirements,
    descriptors: LogicalBufferId,
    alphas: LogicalBufferId,
    z: LogicalBufferId,
    inverse_scratch: LogicalBufferId,
    reduction_a: LogicalBufferId,
    reduction_b: LogicalBufferId,
    scan_eval_scratch: LogicalBufferId,
    scan_temp_scratch: LogicalBufferId,
    scan_descriptors: LogicalBufferId,
    fraction_pointers: LogicalBufferId,
    fraction_geometry: LogicalBufferId,
    instances: Vec<LogicalRelationInstanceSlots>,
}

#[derive(Clone, Debug)]
pub struct PlannedWitnessInputGather {
    pub requirements: WitnessInputGatherRequirements,
    pub producers: Vec<&'static str>,
    pub sources: Vec<ArenaBinding>,
    pub slots: WitnessInputGatherSlots,
}

#[derive(Clone, Debug)]
pub struct PlannedWitnessInputSeed {
    pub requirements: WitnessInputSeedRequirements,
    pub slots: WitnessInputSeedSlots,
}

#[derive(Clone, Debug)]
pub struct PlannedWitnessInputCompact {
    pub requirements: WitnessInputCompactRequirements,
    pub sources: Vec<ArenaBinding>,
    pub slots: WitnessInputCompactSlots,
}

#[derive(Clone, Debug)]
pub struct PlannedExecutionTablesWorkspace {
    pub requirements: ExecutionTablesWorkspaceRequirements,
    pub slots: ExecutionTablesWorkspaceSlots,
}

#[derive(Clone, Debug)]
pub struct PlannedEcOpWorkspace {
    pub requirements: EcOpWorkspaceRequirements,
    pub slots: EcOpWorkspaceSlots,
}

#[derive(Clone, Debug)]
pub struct PlannedWitnessComponent {
    pub component: &'static str,
    pub part: TracePartId,
    pub n_real_rows: usize,
    pub blake_g_fused: bool,
    pub native_input_producer: Option<&'static str>,
    pub program: WitnessProgram,
    pub requirements: WitnessWorkspaceRequirements,
    pub slots: WitnessWorkspaceSlots,
    pub input_gather: Option<PlannedWitnessInputGather>,
    pub input_seed: Option<PlannedWitnessInputSeed>,
    pub input_compact: Option<PlannedWitnessInputCompact>,
}

#[derive(Clone, Debug, Default)]
pub struct PlannedWitnessWorkspace {
    pub components: Vec<PlannedWitnessComponent>,
}

#[derive(Clone, Debug)]
pub struct PlannedGenericRecordedMultiplicityFeedGraph {
    pub plan: crate::multiplicity_pipeline::PlannedRecordedMultiplicityFeed,
    pub source: ArenaBinding,
    pub slots: WitnessFeedWorkspaceSlots,
}

#[derive(Clone, Debug)]
pub enum PlannedRecordedMultiplicityFeedGraph {
    Generic(PlannedGenericRecordedMultiplicityFeedGraph),
    BlakeGFused {
        plan: crate::multiplicity_pipeline::PlannedRecordedMultiplicityFeed,
        lut_tables: [ArenaBinding; 4],
        multiplicity_destinations: [ArenaBinding; 5],
    },
}

#[derive(Clone, Debug)]
pub struct PlannedFixedTableMaterializer {
    pub plan: crate::multiplicity_pipeline::PlannedFixedMultiplicity,
    pub sources: Vec<PlannedFixedTableSource>,
    pub multiplicity: ArenaBinding,
    pub slots: FixedTableContiguousWorkspaceSlots,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlannedFixedTableSource {
    Arena(ArenaBinding),
    RegisteredPedersen18 { column: usize },
}

#[derive(Clone, Debug)]
pub struct PlannedGraphAMultiplicityWorkspace {
    pub topology_hash: u64,
    pub coverage_gaps: Vec<FixedMultiplicityCoverageGap>,
    pub blockers: Vec<MultiplicityFeedBlocker>,
    pub multiplicities: Vec<(&'static str, ArenaBinding)>,
    pub clear_requirements: WitnessFeedClearWorkspaceRequirements,
    pub clear_slots: WitnessFeedClearWorkspaceSlots,
    pub feeds: Vec<PlannedRecordedMultiplicityFeedGraph>,
    pub public_memory_seed: Option<PlannedGenericRecordedMultiplicityFeedGraph>,
    pub fixed_tables: Vec<PlannedFixedTableMaterializer>,
    pub memory_traces: Option<PlannedMemoryBaseTraceWorkspace>,
}

#[derive(Clone, Debug)]
pub struct PlannedMemoryTracePartWorkspace {
    pub part: TracePartId,
    pub source_offset: usize,
    pub row_count: usize,
    pub outputs: Vec<ArenaBinding>,
}

#[derive(Clone, Debug)]
pub struct PlannedMemoryBaseTraceWorkspace {
    pub plan: PlannedMemoryBaseTraces,
    pub address_outputs: Vec<ArenaBinding>,
    pub big_parts: Vec<PlannedMemoryTracePartWorkspace>,
    pub small_part: PlannedMemoryTracePartWorkspace,
    pub rc99_lut: ArenaBinding,
    pub rc99_counts: ArenaBinding,
}

impl PlannedGraphAMultiplicityWorkspace {
    pub fn coverage_complete(&self) -> bool {
        self.coverage_gaps.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct PlannedPreprocessedColumn {
    pub identity: String,
    pub ordinal: u32,
    pub log_size: u32,
    pub evaluations: Option<ArenaBinding>,
    pub coefficients: ArenaBinding,
}

#[derive(Clone, Debug)]
pub struct PlannedPreprocessedInterpolationBatch {
    pub log_size: u32,
    pub column_ordinals: Vec<u32>,
    pub coefficient_pointers: ArenaBinding,
}

/// Fixed-column residency initialized once per exact workspace key.
#[derive(Clone, Debug)]
pub struct PlannedPreprocessedWorkspace {
    pub columns: Vec<PlannedPreprocessedColumn>,
    pub interpolation_batches: Vec<PlannedPreprocessedInterpolationBatch>,
    pub inverse_twiddles: ArenaBinding,
}

#[derive(Clone, Debug)]
pub struct PlannedOodsColumn {
    pub source: OpenedColumnSource,
    pub coefficient_log_size: u32,
    pub evaluation_log_size: u32,
    pub shape_points: Vec<CirclePoint<SecureField>>,
    pub offset_points: Vec<CirclePoint<BaseField>>,
    pub source_kind: OodsSourceKind,
    pub source_binding: ArenaBinding,
}

#[derive(Clone, Debug)]
pub struct PlannedCompositionTraceColumn {
    pub source: OpenedColumnSource,
    pub log_size: u32,
    pub coefficients: ArenaBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedDirectCompositionBinding {
    pub consumer: usize,
    pub plan_column: usize,
    pub source: OpenedColumnSource,
    pub tree: CommitmentTreeId,
    pub proof_column: usize,
    pub group: usize,
    pub column_in_group: usize,
    pub canonical_column: usize,
    pub evaluation_log_size: u32,
    pub evaluation: Option<ArenaBinding>,
}

#[derive(Clone, Debug)]
pub struct PlannedCompositionExtParams {
    pub component: &'static str,
    pub instance: usize,
    pub sources: Vec<CompositionExtParamSource>,
    pub values: Vec<SecureField>,
    pub binding: Option<ArenaBinding>,
}

#[derive(Clone, Debug)]
pub struct PlannedCompositionWorkspace {
    pub plan: CompositionPlan,
    pub requirements: CompositionWorkspaceRequirements,
    pub trace_trees: Vec<Vec<PlannedCompositionTraceColumn>>,
    pub random_coefficient: ArenaBinding,
    pub forward_twiddles: ArenaBinding,
    pub inverse_twiddles: ArenaBinding,
    pub ext_params: Vec<PlannedCompositionExtParams>,
    pub slots: CompositionWorkspaceSlots,
    pub direct_retention: Option<DirectCompositionRetentionPlan>,
    pub direct_bindings: Vec<PlannedDirectCompositionBinding>,
}

impl PlannedCompositionWorkspace {
    pub fn trace_topology(&self) -> CompositionTraceTopology {
        CompositionTraceTopology {
            trees: self
                .trace_trees
                .iter()
                .map(|tree| {
                    tree.iter()
                        .map(|column| CompositionCoefficientSource {
                            slot: column.coefficients.physical,
                            log_size: column.log_size,
                        })
                        .collect()
                })
                .collect(),
        }
    }

    pub fn ext_param_bindings(&self) -> Vec<Option<CompositionExtParamBinding>> {
        self.ext_params
            .iter()
            .map(|params| {
                params.binding.map(|binding| CompositionExtParamBinding {
                    slot: binding.physical,
                    offset_words: 0,
                })
            })
            .collect()
    }
}

impl PlannedOodsColumn {
    pub fn topology(&self) -> OodsColumnTopology<'_> {
        match self.source_kind {
            OodsSourceKind::Coefficients => OodsColumnTopology::coefficient_offset_points(
                self.coefficient_log_size,
                self.evaluation_log_size,
                &self.offset_points,
            ),
            OodsSourceKind::Evaluations => OodsColumnTopology::evaluation_offset_points(
                self.evaluation_log_size,
                &self.offset_points,
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlannedOodsWorkspace {
    pub config: OodsWorkspaceConfig,
    pub requirements: OodsWorkspaceRequirements,
    pub columns: Vec<PlannedOodsColumn>,
    pub oods_point_parameter: ArenaBinding,
    pub slots: OodsWorkspaceSlots,
    pub sample_points: ArenaBinding,
    pub sampled_values: ArenaBinding,
}

#[derive(Clone, Debug)]
pub struct PlannedQuotientNumeratorColumn {
    pub source: OpenedColumnSource,
    pub topology: QuotientNumeratorColumnTopology,
    pub coefficients: ArenaBinding,
    pub numerator_source: ArenaBinding,
}

#[derive(Clone, Debug)]
pub struct PlannedQuotientNumeratorWorkspace {
    pub schedule: QuotientNumeratorSchedule,
    pub config: QuotientNumeratorWorkspaceConfig,
    pub requirements: QuotientNumeratorWorkspaceRequirements,
    pub columns: Vec<PlannedQuotientNumeratorColumn>,
    pub oods_sample_points: ArenaBinding,
    pub oods_sampled_values: ArenaBinding,
    pub random_coefficient: ArenaBinding,
    pub sample_points_destination: ArenaBinding,
    pub first_linear_terms_destination: ArenaBinding,
    pub destinations: Vec<PlannedQuotientNumeratorSource>,
    pub forward_twiddles: ArenaBinding,
    pub slots: QuotientNumeratorWorkspaceSlots,
}

/// Exact prepared-commit inputs resolved to physical arena slots after liveness
/// coloring. The canonical log groups are retained so the caller can bind the
/// corresponding coefficient columns without rediscovering ordering.
#[derive(Clone, Debug)]
pub struct PlannedCommitment {
    pub id: CommitmentTreeId,
    pub config: CommitWorkspaceConfig,
    pub grouped_column_log_sizes: Vec<Vec<u32>>,
    pub grouped_column_sources: Vec<Vec<CommitmentColumnSource>>,
    pub requirements: ModeAwareCommitWorkspaceRequirements,
    pub twiddles: ArenaBinding,
    pub slots: ModeAwareCommitWorkspaceSlots,
    pub retained_evaluation_groups: Vec<Option<Vec<ArenaBinding>>>,
    pub evaluation_output_groups: Vec<Option<Vec<ArenaBinding>>>,
    pub direct_composition_evaluation_groups: Vec<Option<Vec<ArenaBinding>>>,
    pub numerator_evaluation_groups: Vec<Option<Vec<ArenaBinding>>>,
    /// Exact root and retained decommit layers, bound independently of the
    /// ephemeral `PreparedCommitGraph` value used for the cold fixed commit.
    pub root: ArenaBinding,
    pub retained_layers_bottom_up: Vec<ArenaBinding>,
    pub interpolation_mode: InterpolationLaunchMode,
    pub interpolation_batches: Vec<PlannedInterpolationBatch>,
}

#[derive(Clone, Debug)]
pub struct PlannedInterpolationBatch {
    pub log_size: u32,
    pub sources: Vec<CommitmentColumnSource>,
    pub input_pointers: ArenaSlotId,
    pub output_pointers: ArenaSlotId,
}

#[derive(Clone, Debug)]
pub struct PlannedFriWorkspace {
    pub config: FriWorkspaceConfig,
    pub requirements: FriWorkspaceRequirements,
    pub twiddles: ArenaBinding,
    /// The contiguous four-coordinate quotient output consumed by FRI.
    pub input_values: ArenaBinding,
    pub slots: FriWorkspaceSlots,
}

/// Arena-owned scratch at the two Fiat-Shamir PoW edges plus final FRI
/// interpolation. Transcript state/nonces and LinePoly output remain the typed
/// transcript bindings; this plan owns only allocation-free replay workspace.
#[derive(Clone, Debug)]
pub struct PlannedFinalFriPowWorkspace {
    pub final_requirements: FriFinalWorkspaceRequirements,
    pub final_slots: FriFinalWorkspaceSlots,
    pub pow_requirements: Blake2sPowWorkspaceRequirements,
    pub interaction_pow_bits: u32,
    pub interaction_pow_slots: Blake2sPowWorkspaceSlots,
    pub query_pow_bits: u32,
    pub query_pow_slots: Blake2sPowWorkspaceSlots,
}

#[derive(Clone, Debug)]
pub struct PlannedDecommitWorkspace {
    pub config: DecommitWorkspaceConfig,
    pub requirements: DecommitWorkspaceRequirements,
    pub proof_shape: Blake2sProofAssemblyShape,
    /// Exact device-transcript query output; no intermediate query copy exists.
    pub raw_queries: ArenaBinding,
    pub lde_twiddles: ArenaBinding,
    pub slots: DecommitWorkspaceSlots,
    pub proof_bundle_layout: ResidentProofBundleLayout,
    pub proof_bundle: ArenaBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedQuotientNumeratorSource {
    pub log_size: u32,
    pub coordinates: [ArenaBinding; 4],
}

#[derive(Clone, Debug)]
pub struct PlannedQuotientWorkspace {
    pub config: QuotientWorkspaceConfig,
    pub requirements: QuotientWorkspaceRequirements,
    pub forward_twiddles: ArenaBinding,
    pub inverse_subdomain_twiddles: ArenaBinding,
    pub partial_numerators: Vec<PlannedQuotientNumeratorSource>,
    pub slots: QuotientWorkspaceSlots,
    pub sample_points: ArenaBinding,
    pub first_linear_terms: ArenaBinding,
    /// Binding form of `slots.output_values`; this preserves the logical
    /// QuotientTile identity and its exact contiguous length for FRI setup.
    pub output_values: ArenaBinding,
}

#[derive(Clone, Debug)]
pub struct PlannedTranscriptWorkspace {
    pub schedule_key: u64,
    pub requirements: Blake2sTranscriptRequirements,
    pub slots: Blake2sTranscriptWorkspaceSlots,
    pub inputs: Vec<(TranscriptInputId, ArenaBinding)>,
    pub outputs: Vec<(TranscriptOutputId, ArenaBinding)>,
}

#[derive(Clone, Debug)]
pub struct PlannedRelationWorkspace {
    pub execution: RelationExecutionPlan,
    pub source_plan: Vec<RelationInstanceSourcePlan>,
    pub launch_mode: RelationLaunchMode,
    pub requirements: RelationGraphRequirements,
    pub slots: RelationGraphSlots,
}

#[derive(Clone, Debug)]
pub struct ProofArenaPlan {
    pub shape_key: stwo_cairo_prover::witness::proof_shape::ProofShapeKey,
    pub protocol_key: u64,
    protocol_identity: ProtocolIdentity,
    logical: Vec<LogicalBuffer>,
    bindings: Vec<ArenaBinding>,
    layout: ArenaLayout,
    high_water_words: Vec<(ProofEpoch, usize)>,
    whole_slot_total_words: usize,
    raw_peak_words: usize,
    excess_over_raw_peak_words: usize,
    range_view_count: usize,
    range_view_words: usize,
    late_coefficient_ownership: LateCoefficientOwnershipPlan,
    preprocessed: PlannedPreprocessedWorkspace,
    commitments: Vec<PlannedCommitment>,
    composition: PlannedCompositionWorkspace,
    oods: PlannedOodsWorkspace,
    quotient_numerator: PlannedQuotientNumeratorWorkspace,
    quotient: PlannedQuotientWorkspace,
    fri: PlannedFriWorkspace,
    final_fri_pow: PlannedFinalFriPowWorkspace,
    decommit: PlannedDecommitWorkspace,
    transcript: PlannedTranscriptWorkspace,
    execution_tables: Option<PlannedExecutionTablesWorkspace>,
    ec_op: Option<PlannedEcOpWorkspace>,
    witness: PlannedWitnessWorkspace,
    multiplicity: Option<PlannedGraphAMultiplicityWorkspace>,
    relation: PlannedRelationWorkspace,
}

impl ProofArenaPlan {
    pub fn build(
        plan: &ProofPlan,
        protocol: &ProtocolGeometry,
        composition: &CompositionPlan,
    ) -> Result<Self, ArenaPlanError> {
        Self::build_inner(plan, protocol, composition, None)
    }

    pub fn build_with_execution_tables(
        plan: &ProofPlan,
        protocol: &ProtocolGeometry,
        composition: &CompositionPlan,
        geometry: ExecutionTableGeometry,
    ) -> Result<Self, ArenaPlanError> {
        Self::build_inner(plan, protocol, composition, Some(geometry))
    }

    fn build_inner(
        plan: &ProofPlan,
        protocol: &ProtocolGeometry,
        composition: &CompositionPlan,
        execution_table_geometry: Option<ExecutionTableGeometry>,
    ) -> Result<Self, ArenaPlanError> {
        plan.proof_shape()
            .require_arena_ready()
            .map_err(ArenaPlanError::Shape)?;
        protocol.validate()?;
        if protocol.identity.composition_plan_hash != composition.key() {
            return Err(ArenaPlanError::CompositionPlanMismatch {
                protocol: protocol.identity.composition_plan_hash,
                planned: composition.key(),
            });
        }
        if let Some(direct) = &protocol.direct_composition_retention {
            let consumers = derive_direct_composition_consumers(
                &protocol.oods,
                composition,
                protocol.identity.composition_launch_mode,
            )
            .map_err(|_| {
                ArenaPlanError::InvalidProtocolGeometry(
                    "direct composition consumer derivation failed",
                )
            })?;
            validate_direct_composition_retention_plan(protocol, &consumers, direct).map_err(
                |_| {
                    ArenaPlanError::InvalidProtocolGeometry(
                        "direct composition retention plan drifted",
                    )
                },
            )?;
        }
        if protocol.identity.relation_graph_hash != plan.relation_graph_hash {
            return Err(ArenaPlanError::RelationGraphMismatch {
                planned: plan.relation_graph_hash,
                protocol: protocol.identity.relation_graph_hash,
            });
        }

        let relation_execution =
            RelationExecutionPlan::from_proof_plan(plan, &CAIRO_RELATION_GRAPH)
                .map_err(ArenaPlanError::RelationExecution)?;
        let retained_base_trace = relation_execution
            .source_plan()
            .map_err(ArenaPlanError::RelationExecution)?
            .into_iter()
            .filter(|source| source.plane == RelationSourcePlane::BaseTrace)
            .flat_map(|source| {
                (0..source.column_count)
                    .map(move |ordinal| (source.batch.component, source.part, ordinal))
            })
            .collect::<HashSet<_>>();
        let late_coefficient_ownership = LateCoefficientOwnershipPlan::compile(protocol)
            .map_err(ArenaPlanError::InvalidProtocolGeometry)?;
        let multiplicity_plan = execution_table_geometry
            .is_some()
            .then(|| plan_graph_a_multiplicities(plan).map_err(ArenaPlanError::MultiplicityPlan))
            .transpose()?;
        let blake_g_fused = multiplicity_plan.as_ref().is_some_and(|multiplicity| {
            let feed_is_exact = multiplicity
                .feeds
                .iter()
                .find(|feed| feed.producer == "blake_g")
                .and_then(blake_g_fused_feed_binding)
                .is_some();
            if !feed_is_exact {
                return false;
            }
            let recording = BlakeGRecordedLane::record();
            recording.poisoned_cols.is_empty()
                && recording.poisoned_lookup_words.is_empty()
                && recording.poisoned_sub_words.is_empty()
                && recording.poison_ops.is_empty()
                && blake_g_fusion_program_is_exact(&recording.program)
        });
        let mut logical = Vec::new();
        let mut transition_aliases = Vec::new();
        for component in &plan.components {
            let parts = capacity_parts(&component.runtime.rows)?;
            for part in parts {
                let max_domain_rows = checked_pow2(protocol.max_domain_log_size)? as u64;
                if part.padded_rows > max_domain_rows {
                    return Err(ArenaPlanError::ComponentExceedsProtocolDomain {
                        component: component.node.id,
                        padded_rows: part.padded_rows,
                        max_domain_log_size: protocol.max_domain_log_size,
                    });
                }
                let trace_columns = trace_width(component.node.facts.trace_columns, part.part)?;
                let trace_words =
                    usize::try_from(part.padded_rows).map_err(|_| ArenaPlanError::SizeOverflow)?;
                for ordinal in 0..trace_columns {
                    let retained =
                        retained_base_trace.contains(&(component.node.id, part.part, ordinal));
                    let evaluations = push_buffer_id(
                        &mut logical,
                        Some(component.node.id),
                        Some(part.part),
                        BufferPurpose::BaseTrace,
                        ordinal,
                        trace_words,
                        if retained {
                            BufferLifetime::new(ProofEpoch::Witness, ProofEpoch::Interaction)?
                        } else {
                            BufferLifetime::at(ProofEpoch::Witness)
                        },
                    )?;
                    let coefficient_source = OpenedColumnSource::Trace {
                        component: component.node.id,
                        part: part.part,
                        purpose: BufferPurpose::BaseCoefficients,
                        ordinal,
                    };
                    let coefficients = push_buffer_id(
                        &mut logical,
                        Some(component.node.id),
                        Some(part.part),
                        BufferPurpose::BaseCoefficients,
                        ordinal,
                        trace_words,
                        BufferLifetime::new(
                            ProofEpoch::BaseCommit,
                            late_coefficient_ownership
                                .final_consumer(coefficient_source)
                                .map_err(ArenaPlanError::InvalidProtocolGeometry)?,
                        )?,
                    )?;
                    if !retained {
                        transition_aliases.push((evaluations, coefficients));
                    }
                }
                if let Some(words) = component.node.facts.lookup_words {
                    push_component_flat_buffer(
                        &mut logical,
                        component.node.id,
                        part.part,
                        BufferPurpose::LookupInputs,
                        words,
                        part.padded_rows,
                        BufferLifetime::new(ProofEpoch::Witness, ProofEpoch::Interaction)?,
                    )?;
                }
                if let Some(words) = component
                    .node
                    .facts
                    .sub_words
                    .filter(|_| component.node.id != "blake_g" || !blake_g_fused)
                {
                    push_component_flat_buffer(
                        &mut logical,
                        component.node.id,
                        part.part,
                        BufferPurpose::SubcomponentInputs,
                        words,
                        part.padded_rows,
                        BufferLifetime::at(ProofEpoch::Witness),
                    )?;
                }
                if let Some(columns) = component.node.facts.logup_columns {
                    let coordinate_columns =
                        columns.checked_mul(4).ok_or(ArenaPlanError::SizeOverflow)?;
                    for ordinal in 0..coordinate_columns {
                        let evaluations = push_buffer_id(
                            &mut logical,
                            Some(component.node.id),
                            Some(part.part),
                            BufferPurpose::InteractionTrace,
                            ordinal,
                            trace_words,
                            BufferLifetime::at(ProofEpoch::Interaction),
                        )?;
                        let coefficient_source = OpenedColumnSource::Trace {
                            component: component.node.id,
                            part: part.part,
                            purpose: BufferPurpose::InteractionCoefficients,
                            ordinal,
                        };
                        let coefficients = push_buffer_id(
                            &mut logical,
                            Some(component.node.id),
                            Some(part.part),
                            BufferPurpose::InteractionCoefficients,
                            ordinal,
                            trace_words,
                            BufferLifetime::new(
                                ProofEpoch::InteractionCommit,
                                late_coefficient_ownership
                                    .final_consumer(coefficient_source)
                                    .map_err(ArenaPlanError::InvalidProtocolGeometry)?,
                            )?,
                        )?;
                        transition_aliases.push((evaluations, coefficients));
                    }
                }
            }
        }
        let logical_execution_tables = execution_table_geometry
            .map(|geometry| append_execution_table_buffers(&mut logical, geometry))
            .transpose()?;
        let logical_witness = append_witness_buffers(
            &mut logical,
            plan,
            logical_execution_tables.is_some(),
            blake_g_fused,
        )?;
        let logical_relation = append_relation_buffers(
            &mut logical,
            relation_execution,
            &late_coefficient_ownership,
            &mut transition_aliases,
        )?;
        let retained_preprocessed_evaluations = multiplicity_plan.as_ref().map(|multiplicity| {
            multiplicity
                .fixed
                .iter()
                .flat_map(|fixed| fixed.materializer.preprocessed_sources().iter().copied())
                .filter(|identity| retain_fixed_preprocessed_evaluation(identity))
                .collect::<BTreeSet<_>>()
        });
        let (
            logical_preprocessed,
            logical_commitments,
            logical_composition,
            logical_oods,
            logical_quotient_numerator,
            logical_quotient,
            logical_fri,
            logical_final_fri_pow,
            logical_decommit,
            logical_transcript,
        ) = append_protocol_buffers(
            &mut logical,
            protocol,
            composition,
            retained_preprocessed_evaluations.as_ref(),
            &late_coefficient_ownership,
        )?;
        let logical_multiplicity = if let Some(multiplicity_plan) = multiplicity_plan {
            if multiplicity_plan.fixed.is_empty() && multiplicity_plan.runtime.is_empty() {
                None
            } else {
                Some(append_graph_a_multiplicity_buffers(
                    &mut logical,
                    &logical_preprocessed,
                    multiplicity_plan,
                    execution_table_geometry.map_or(0, |geometry| geometry.public_memory_entries),
                    blake_g_fused,
                )?)
            }
        } else {
            None
        };
        let logical_ec_op = append_ec_op_buffers(
            &mut logical,
            plan,
            logical_execution_tables.as_ref(),
            &logical_witness,
            logical_multiplicity.as_ref(),
        )?;
        validate_commitment_sources(&logical, &logical_commitments)?;

        // Retain the old whole-slot result as an exact same-shape comparator,
        // but bind production execution to the checked range-packed layout.
        let (_, whole_slot_specs, whole_slot_total_words) =
            color_logical_buffers(&logical, &transition_aliases)?;
        ArenaLayout::new(whole_slot_total_words, &whole_slot_specs)
            .map_err(ArenaPlanError::Arena)?;
        let range_arena = plan_range_arena(&logical, &transition_aliases)?;
        let bindings = range_arena.bindings;
        let layout = range_arena.layout;
        validate_aliases(&logical, &bindings)?;
        let high_water_words = range_arena.high_water_words;
        if high_water_words
            .iter()
            .any(|&(epoch, words)| words != high_water_at(epoch, &logical, &bindings))
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "range-address and slot-identity high-water disagree",
            ));
        }
        let commitments: Vec<PlannedCommitment> = logical_commitments
            .into_iter()
            .map(|commitment| resolve_commitment_slots(commitment, &bindings))
            .collect::<Result<Vec<_>, _>>()?;
        let preprocessed = resolve_preprocessed_slots(logical_preprocessed, &bindings)?;
        let composition = resolve_composition_slots(logical_composition, &bindings)?;
        let oods = resolve_oods_slots(logical_oods, &bindings)?;
        let quotient_numerator =
            resolve_quotient_numerator_slots(logical_quotient_numerator, &bindings)?;
        let quotient = resolve_quotient_slots(logical_quotient, &bindings)?;
        let fri = resolve_fri_slots(logical_fri, &bindings)?;
        let final_fri_pow = resolve_final_fri_pow_slots(logical_final_fri_pow, &bindings)?;
        let decommit = resolve_decommit_slots(logical_decommit, &bindings)?;
        validate_decommit_group_bindings(&decommit.config, &commitments)?;
        if quotient.output_values != fri.input_values {
            return Err(ArenaPlanError::QuotientFriInputMismatch {
                quotient: quotient.output_values,
                fri: fri.input_values,
            });
        }
        if quotient_numerator.sample_points_destination != quotient.sample_points
            || quotient_numerator.first_linear_terms_destination != quotient.first_linear_terms
            || quotient_numerator.destinations != quotient.partial_numerators
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "quotient numerator outputs do not alias prepared quotient inputs exactly",
            ));
        }
        let transcript = resolve_transcript_slots(logical_transcript, &bindings)?;
        let transcript_oods_values = transcript
            .inputs
            .iter()
            .find_map(|(id, binding)| {
                (*id == protocol.oods.sampled_values_input).then_some(*binding)
            })
            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "missing OODS sampled-values transcript binding",
            ))?;
        let transcript_composition_random = transcript
            .outputs
            .iter()
            .find_map(|(id, binding)| {
                (*id == protocol.composition_random_coefficient_output).then_some(*binding)
            })
            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "missing composition random-coefficient transcript binding",
            ))?;
        if composition.random_coefficient != transcript_composition_random {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "composition random coefficient does not alias its transcript output exactly",
            ));
        }
        if oods.sampled_values != transcript_oods_values
            || quotient_numerator.oods_sampled_values != transcript_oods_values
            || quotient_numerator.oods_sample_points != oods.sample_points
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS outputs do not alias transcript/numerator inputs exactly",
            ));
        }
        if decommit.raw_queries
            != transcript
                .outputs
                .iter()
                .find_map(|(id, binding)| {
                    (Some(*id) == CairoTranscriptOutput::QueryPositions.id().ok())
                        .then_some(*binding)
                })
                .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                    "query-position transcript output is missing",
                ))?
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "decommit raw queries do not alias the transcript output exactly",
            ));
        }
        let execution_tables = logical_execution_tables
            .map(|logical| resolve_execution_table_slots(logical, &bindings))
            .transpose()?;
        let witness = resolve_witness_slots(logical_witness, execution_tables.as_ref(), &bindings)?;
        let multiplicity = logical_multiplicity
            .map(|logical| resolve_graph_a_multiplicity_slots(logical, &bindings))
            .transpose()?;
        let ec_op = logical_ec_op
            .map(|logical| resolve_ec_op_slots(logical, &bindings))
            .transpose()?;
        let relation = resolve_relation_slots(logical_relation, &bindings)?;

        let mut protocol_key = execution_table_geometry.map_or_else(
            || protocol.key(),
            |geometry| execution_tables_protocol_key(protocol.key(), geometry),
        );
        if let Some(multiplicity) = &multiplicity {
            protocol_key =
                graph_a_multiplicity_protocol_key(protocol_key, multiplicity.topology_hash);
        }
        protocol_key = relation_protocol_key(protocol_key, relation.launch_mode);

        Ok(Self {
            shape_key: plan.shape_key,
            protocol_key,
            protocol_identity: protocol.identity,
            logical,
            bindings,
            layout,
            high_water_words,
            whole_slot_total_words,
            raw_peak_words: range_arena.raw_peak_words,
            excess_over_raw_peak_words: range_arena.excess_over_raw_peak_words,
            range_view_count: range_arena.range_view_count,
            range_view_words: range_arena.range_view_words,
            late_coefficient_ownership,
            preprocessed,
            commitments,
            composition,
            oods,
            quotient_numerator,
            quotient,
            fri,
            final_fri_pow,
            decommit,
            transcript,
            execution_tables,
            ec_op,
            witness,
            multiplicity,
            relation,
        })
    }

    pub fn layout(&self) -> &ArenaLayout {
        &self.layout
    }

    pub const fn protocol_identity(&self) -> ProtocolIdentity {
        self.protocol_identity
    }

    pub fn logical_buffers(&self) -> &[LogicalBuffer] {
        &self.logical
    }

    pub fn bindings(&self) -> &[ArenaBinding] {
        &self.bindings
    }

    pub fn binding(&self, logical: LogicalBufferId) -> Option<ArenaBinding> {
        find_binding(&self.bindings, logical).ok()
    }

    pub fn high_water_words(&self, epoch: ProofEpoch) -> usize {
        self.high_water_words
            .iter()
            .find_map(|&(candidate, words)| (candidate == epoch).then_some(words))
            .expect("all proof epochs have a high-water entry")
    }

    pub fn total_words(&self) -> usize {
        self.layout.total_words()
    }

    /// Bytes omitted from this arena because the immutable process-lifetime
    /// Pedersen registration is the sole owner of those evaluation columns.
    pub fn process_owned_pedersen_evaluation_bytes(&self) -> usize {
        let columns = self
            .multiplicity
            .iter()
            .flat_map(|multiplicity| &multiplicity.fixed_tables)
            .flat_map(|fixed| &fixed.sources)
            .filter_map(|source| match source {
                PlannedFixedTableSource::RegisteredPedersen18 { column } => Some(*column),
                PlannedFixedTableSource::Arena(_) => None,
            })
            .collect::<BTreeSet<_>>();
        columns.len() * PEDERSEN_POINTS_18_ROW_COUNT * core::mem::size_of::<u32>()
    }

    pub fn requires_registered_pedersen_table(&self) -> bool {
        preprocessed_requires_registered_pedersen_table(&self.preprocessed.columns)
    }

    /// Exact allocation produced by the superseded whole-slot colorer for the
    /// same logical shape and transition aliases.
    pub fn whole_slot_total_words(&self) -> usize {
        self.whole_slot_total_words
    }

    /// Hard lower bound from the largest sum of simultaneously-live logical
    /// ranges. This is not an allocator optimality claim.
    pub fn raw_peak_words(&self) -> usize {
        self.raw_peak_words
    }

    /// Exact distance between this range-packed slab and `raw_peak_words`.
    pub fn excess_over_raw_peak_words(&self) -> usize {
        self.excess_over_raw_peak_words
    }

    /// Stable runtime range views. This is not a device-allocation count: all
    /// views belong to the one shape arena allocation.
    pub fn range_view_count(&self) -> usize {
        self.range_view_count
    }

    /// Sum of view lengths before spatial reuse; may exceed `total_words`.
    pub fn range_view_words(&self) -> usize {
        self.range_view_words
    }

    pub fn late_coefficient_ownership(&self) -> &LateCoefficientOwnershipPlan {
        &self.late_coefficient_ownership
    }

    pub fn commitments(&self) -> &[PlannedCommitment] {
        &self.commitments
    }

    pub fn preprocessed(&self) -> &PlannedPreprocessedWorkspace {
        &self.preprocessed
    }

    pub fn preprocessed_coefficients(&self) -> &[PlannedPreprocessedColumn] {
        &self.preprocessed.columns
    }

    pub fn oods(&self) -> &PlannedOodsWorkspace {
        &self.oods
    }

    pub fn composition(&self) -> &PlannedCompositionWorkspace {
        &self.composition
    }

    pub fn quotient_numerator(&self) -> &PlannedQuotientNumeratorWorkspace {
        &self.quotient_numerator
    }

    pub fn commitment(&self, id: CommitmentTreeId) -> Option<&PlannedCommitment> {
        self.commitments
            .iter()
            .find(|commitment| commitment.id == id)
    }

    pub fn fri(&self) -> &PlannedFriWorkspace {
        &self.fri
    }

    pub fn final_fri_pow(&self) -> &PlannedFinalFriPowWorkspace {
        &self.final_fri_pow
    }

    pub fn decommit(&self) -> &PlannedDecommitWorkspace {
        &self.decommit
    }

    pub fn quotient(&self) -> &PlannedQuotientWorkspace {
        &self.quotient
    }

    pub fn transcript(&self) -> &PlannedTranscriptWorkspace {
        &self.transcript
    }

    pub fn execution_tables(&self) -> Option<&PlannedExecutionTablesWorkspace> {
        self.execution_tables.as_ref()
    }

    pub fn ec_op(&self) -> Option<&PlannedEcOpWorkspace> {
        self.ec_op.as_ref()
    }

    pub fn witness(&self) -> &PlannedWitnessWorkspace {
        &self.witness
    }

    pub fn multiplicity(&self) -> Option<&PlannedGraphAMultiplicityWorkspace> {
        self.multiplicity.as_ref()
    }

    pub fn relation(&self) -> &PlannedRelationWorkspace {
        &self.relation
    }

    pub fn fri_merkle_layer(
        &self,
        tree_index: usize,
        log_size: u32,
    ) -> Option<(&LogicalBuffer, ArenaBinding)> {
        let ordinal = u32::try_from(tree_index)
            .ok()?
            .checked_shl(16)?
            .checked_add(log_size)?;
        self.find(None, None, BufferPurpose::FriMerkleLayer, ordinal)
    }

    /// Allocate the one stable slab after every logical range and alias has been
    /// validated. No component-level allocation occurs here or later.
    pub fn allocate(&self, context: CudaExecContext) -> Result<DeviceArena, ArenaPlanError> {
        DeviceArena::new(context, self.layout.clone()).map_err(ArenaPlanError::Arena)
    }

    pub fn find(
        &self,
        component: Option<&str>,
        part: Option<TracePartId>,
        purpose: BufferPurpose,
        ordinal: u32,
    ) -> Option<(&LogicalBuffer, ArenaBinding)> {
        let logical = self.logical.iter().find(|buffer| {
            buffer.component == component
                && buffer.part == part
                && buffer.purpose == purpose
                && buffer.ordinal == ordinal
        })?;
        Some((logical, self.binding(logical.id)?))
    }

    pub fn validate_aliases(&self) -> Result<(), ArenaPlanError> {
        validate_aliases(&self.logical, &self.bindings)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArenaPlanError {
    Shape(ProofShapeError),
    PendingRows,
    InvalidTracePart {
        part: TracePartId,
        trace_columns: TraceColumnCount,
    },
    InvalidLifetime {
        first: ProofEpoch,
        last: ProofEpoch,
    },
    InvalidProtocolGeometry(&'static str),
    ComponentExceedsProtocolDomain {
        component: &'static str,
        padded_rows: u64,
        max_domain_log_size: u32,
    },
    RelationGraphMismatch {
        planned: u64,
        protocol: u64,
    },
    MissingWitnessRecording(&'static str),
    UnsupportedWitnessMultiplicities {
        component: &'static str,
        tables: u32,
    },
    WitnessProgramShapeMismatch {
        component: &'static str,
        role: &'static str,
        expected: u32,
        actual: u32,
    },
    MissingWitnessBuffer {
        component: &'static str,
        purpose: BufferPurpose,
        ordinal: u32,
    },
    CompositionPlanMismatch {
        protocol: u64,
        planned: u64,
    },
    QuotientFriInputMismatch {
        quotient: ArenaBinding,
        fri: ArenaBinding,
    },
    RelationExecution(RelationExecutionError),
    Relation(RelationGraphError),
    ExecutionTables(PreparedExecutionTablesError),
    Witness(PreparedWitnessError),
    WitnessInputGather(PreparedWitnessInputGatherError),
    WitnessFeed(PreparedWitnessFeedError),
    FixedTable(PreparedFixedTableError),
    MultiplicityPlan(GraphAMultiplicityPlanError),
    WitnessInputGatherGeometry {
        component: &'static str,
        expected_real_rows: usize,
        actual_real_rows: usize,
        expected_padded_rows: usize,
        actual_padded_rows: usize,
    },
    WitnessInputGatherProgramWidth {
        component: &'static str,
        expected_inputs: usize,
        actual_inputs: usize,
    },
    MissingRelationOutput {
        component: &'static str,
        part: TracePartId,
        ordinal: u32,
    },
    RelationOutputCountMismatch {
        component: &'static str,
        part: TracePartId,
        expected: usize,
        actual: usize,
    },
    RelationOutputTooSmall {
        component: &'static str,
        part: TracePartId,
        ordinal: u32,
        required_words: usize,
        actual_words: usize,
    },
    MissingCommitmentSource {
        tree: CommitmentTreeId,
        source: CommitmentColumnSource,
    },
    CommitmentSourceTreeMismatch {
        tree: CommitmentTreeId,
        source: CommitmentColumnSource,
    },
    CommitmentSourceSizeMismatch {
        tree: CommitmentTreeId,
        source: CommitmentColumnSource,
        expected_words: usize,
        actual_words: usize,
    },
    SizeOverflow,
    EmptyLogicalBuffer(LogicalBufferId),
    MissingBinding(LogicalBufferId),
    AliasedLiveBuffers {
        physical: ArenaSlotId,
        first: LogicalBufferId,
        second: LogicalBufferId,
    },
    Commit(PreparedCommitError),
    ProgressiveCommit(PreparedProgressiveCommitError),
    Composition(PreparedCompositionError),
    Oods(PreparedOodsError),
    QuotientNumerator(PreparedQuotientNumeratorError),
    QuotientNumeratorSchedule(QuotientNumeratorSingleWriteError),
    Quotient(PreparedQuotientError),
    Fri(PreparedFriError),
    FriFinal(PreparedFriFinalError),
    Pow(PreparedBlake2sPowError),
    Decommit(PreparedDecommitError),
    ProofBundle(ResidentProofBundleError),
    Transcript(DeviceTranscriptError),
    Range(RangeAllocationError),
    Arena(ArenaError),
}

impl core::fmt::Display for ArenaPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid proof arena plan: {self:?}")
    }
}

impl std::error::Error for ArenaPlanError {}

fn capacity_parts(rows: &RowResolution) -> Result<Vec<TracePartShape>, ArenaPlanError> {
    match rows {
        RowResolution::Absent => Ok(Vec::new()),
        RowResolution::Resolved(parts) => Ok(parts.clone()),
        RowResolution::Bounded { bound, .. } => Ok(vec![TracePartShape {
            part: TracePartId::Main,
            n_real_rows: bound.observed_rows,
            padded_rows: bound.padded_capacity,
        }]),
        RowResolution::Pending { .. } => Err(ArenaPlanError::PendingRows),
    }
}

fn trace_width(columns: TraceColumnCount, part: TracePartId) -> Result<u32, ArenaPlanError> {
    match (columns, part) {
        (TraceColumnCount::Fixed(width), TracePartId::Main) => Ok(width),
        (TraceColumnCount::SplitMemory { big, .. }, TracePartId::MemoryBig(_)) => Ok(big),
        (TraceColumnCount::SplitMemory { small, .. }, TracePartId::MemorySmall) => Ok(small),
        (trace_columns, part) => Err(ArenaPlanError::InvalidTracePart {
            part,
            trace_columns,
        }),
    }
}

fn push_component_flat_buffer(
    logical: &mut Vec<LogicalBuffer>,
    component: &'static str,
    part: TracePartId,
    purpose: BufferPurpose,
    words_per_row: u32,
    rows: u64,
    lifetime: BufferLifetime,
) -> Result<(), ArenaPlanError> {
    let len_words = usize::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(words_per_row as usize))
        .ok_or(ArenaPlanError::SizeOverflow)?;
    push_buffer(
        logical,
        Some(component),
        Some(part),
        purpose,
        0,
        len_words,
        lifetime,
    )
}

fn push_buffer(
    logical: &mut Vec<LogicalBuffer>,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    purpose: BufferPurpose,
    ordinal: u32,
    len_words: usize,
    lifetime: BufferLifetime,
) -> Result<(), ArenaPlanError> {
    push_buffer_id(
        logical, component, part, purpose, ordinal, len_words, lifetime,
    )
    .map(|_| ())
}

fn push_buffer_id(
    logical: &mut Vec<LogicalBuffer>,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    purpose: BufferPurpose,
    ordinal: u32,
    len_words: usize,
    lifetime: BufferLifetime,
) -> Result<LogicalBufferId, ArenaPlanError> {
    let id =
        LogicalBufferId(u32::try_from(logical.len()).map_err(|_| ArenaPlanError::SizeOverflow)?);
    if len_words == 0 {
        return Err(ArenaPlanError::EmptyLogicalBuffer(id));
    }
    logical.push(LogicalBuffer {
        id,
        component,
        part,
        purpose,
        ordinal,
        len_words,
        lifetime,
    });
    Ok(id)
}

fn logical_buffer_id(
    logical: &[LogicalBuffer],
    component: &'static str,
    part: TracePartId,
    purpose: BufferPurpose,
    ordinal: u32,
) -> Result<LogicalBufferId, ArenaPlanError> {
    logical
        .iter()
        .find(|buffer| {
            buffer.component == Some(component)
                && buffer.part == Some(part)
                && buffer.purpose == purpose
                && buffer.ordinal == ordinal
        })
        .map(|buffer| buffer.id)
        .ok_or(ArenaPlanError::MissingWitnessBuffer {
            component,
            purpose,
            ordinal,
        })
}

fn append_execution_table_buffers(
    logical: &mut Vec<LogicalBuffer>,
    geometry: ExecutionTableGeometry,
) -> Result<LogicalExecutionTablesWorkspace, ArenaPlanError> {
    let requirements =
        execution_tables_workspace_requirements(geometry.n_addrs, geometry.n_big, geometry.n_small)
            .map_err(ArenaPlanError::ExecutionTables)?;
    let ingest_to_witness = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Witness)?;
    let persistent = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?;
    let raw_addr_to_id = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::ExecutionTableRawAddressToId,
        0,
        requirements.raw_addr_to_id_words,
        ingest_to_witness,
    )?;
    let raw_f252_words = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::ExecutionTableRawF252Words,
        0,
        requirements.raw_f252_words,
        ingest_to_witness,
    )?;
    let raw_small_words = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::ExecutionTableRawSmallWords,
        0,
        requirements.raw_small_words,
        ingest_to_witness,
    )?;
    let big_limbs = (0..EXECUTION_TABLE_BIG_LIMBS)
        .map(|ordinal| {
            push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::ExecutionTableBigLimb,
                u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?,
                requirements.big_column_words,
                ingest_to_witness,
            )
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let small_limbs = (0..EXECUTION_TABLE_SMALL_LIMBS)
        .map(|ordinal| {
            push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::ExecutionTableSmallLimb,
                u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?,
                requirements.small_column_words,
                ingest_to_witness,
            )
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let table_pointers = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::ExecutionTablePointers,
        0,
        requirements.table_pointer_words,
        persistent,
    )?;
    let table_strides = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::ExecutionTableStrides,
        0,
        requirements.table_stride_words,
        persistent,
    )?;
    Ok(LogicalExecutionTablesWorkspace {
        requirements,
        raw_addr_to_id,
        raw_f252_words,
        raw_small_words,
        big_limbs,
        small_limbs,
        table_pointers,
        table_strides,
    })
}

fn append_witness_buffers(
    logical: &mut Vec<LogicalBuffer>,
    proof: &ProofPlan,
    shared_execution_tables: bool,
    blake_g_fused: bool,
) -> Result<LogicalWitnessWorkspace, ArenaPlanError> {
    let mut recordings = BTreeMap::new();
    for (label, program) in stwo_cairo_prover::witness::jit_prove_backend::all_lane_recordings() {
        if recordings.insert(label, program).is_some() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "duplicate recorded witness label",
            ));
        }
    }

    let persistent = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?;
    let input_lifetime = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Witness)?;
    let witness_lifetime = BufferLifetime::at(ProofEpoch::Witness);
    let mut components = Vec::new();
    for component in topological_component_order(proof)? {
        if component.node.facts.witness_writer.kind != WitnessWriterKind::RecordedAot
            || !component.runtime.is_present()
        {
            continue;
        }
        let program = recordings
            .remove(component.node.id)
            .ok_or(ArenaPlanError::MissingWitnessRecording(component.node.id))?;
        let parts = capacity_parts(&component.runtime.rows)?;
        if parts.len() != 1 || parts[0].part != TracePartId::Main {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "recorded witness must have one main trace part",
            ));
        }
        let part = parts[0];
        let trace_columns = trace_width(component.node.facts.trace_columns, part.part)?;
        for (role, expected, actual) in [
            ("trace", trace_columns, program.n_cols),
            (
                "lookup",
                component.node.facts.lookup_words.unwrap_or(0),
                program.n_lookup_words,
            ),
            (
                "subcomponent",
                component.node.facts.sub_words.unwrap_or(0),
                program.n_sub_words,
            ),
        ] {
            if expected != actual {
                return Err(ArenaPlanError::WitnessProgramShapeMismatch {
                    component: component.node.id,
                    role,
                    expected,
                    actual,
                });
            }
        }
        if program.n_mult_tables != 0 {
            return Err(ArenaPlanError::UnsupportedWitnessMultiplicities {
                component: component.node.id,
                tables: program.n_mult_tables,
            });
        }
        let rows = usize::try_from(part.padded_rows).map_err(|_| ArenaPlanError::SizeOverflow)?;
        let requirements =
            witness_workspace_requirements(&program, rows, &[]).map_err(ArenaPlanError::Witness)?;

        let input_columns = requirements
            .input_column_words
            .iter()
            .enumerate()
            .map(|(ordinal, &words)| {
                push_buffer_id(
                    logical,
                    Some(component.node.id),
                    Some(part.part),
                    BufferPurpose::WitnessInput,
                    u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?,
                    words,
                    input_lifetime,
                )
            })
            .collect::<Result<Vec<_>, ArenaPlanError>>()?;
        let input_pointers = push_buffer_id(
            logical,
            Some(component.node.id),
            Some(part.part),
            BufferPurpose::WitnessInputPointers,
            0,
            requirements.input_pointer_words,
            persistent,
        )?;
        let execution_table_pointers = (!shared_execution_tables)
            .then(|| {
                push_buffer_id(
                    logical,
                    Some(component.node.id),
                    Some(part.part),
                    BufferPurpose::WitnessExecutionTablePointers,
                    0,
                    requirements.execution_table_pointer_words,
                    persistent,
                )
            })
            .transpose()?;
        let execution_table_strides = (!shared_execution_tables)
            .then(|| {
                push_buffer_id(
                    logical,
                    Some(component.node.id),
                    Some(part.part),
                    BufferPurpose::WitnessExecutionTableStrides,
                    0,
                    requirements.execution_table_stride_words,
                    persistent,
                )
            })
            .transpose()?;
        let output_columns = (0..program.n_cols)
            .map(|ordinal| {
                logical_buffer_id(
                    logical,
                    component.node.id,
                    part.part,
                    BufferPurpose::BaseTrace,
                    ordinal,
                )
            })
            .collect::<Result<Vec<_>, ArenaPlanError>>()?;
        let output_pointers = push_buffer_id(
            logical,
            Some(component.node.id),
            Some(part.part),
            BufferPurpose::WitnessOutputPointers,
            0,
            requirements.output_pointer_words,
            persistent,
        )?;
        let multiplicity_pointers = push_buffer_id(
            logical,
            Some(component.node.id),
            Some(part.part),
            BufferPurpose::WitnessMultiplicityPointers,
            0,
            requirements.multiplicity_pointer_words,
            persistent,
        )?;
        let multiplicity_dummy = requirements
            .multiplicity_dummy_words
            .map(|words| {
                push_buffer_id(
                    logical,
                    Some(component.node.id),
                    Some(part.part),
                    BufferPurpose::WitnessMultiplicityDummy,
                    0,
                    words,
                    witness_lifetime,
                )
            })
            .transpose()?;
        let lookup_words = match component.node.facts.lookup_words {
            Some(_) => logical_buffer_id(
                logical,
                component.node.id,
                part.part,
                BufferPurpose::LookupInputs,
                0,
            )?,
            None => push_buffer_id(
                logical,
                Some(component.node.id),
                Some(part.part),
                BufferPurpose::WitnessLookupDummy,
                0,
                requirements.lookup_words,
                witness_lifetime,
            )?,
        };
        let component_blake_g_fused = blake_g_fused && component.node.id == "blake_g";
        let sub_words = match (component.node.facts.sub_words, component_blake_g_fused) {
            (Some(_), true) => {
                multiplicity_dummy.ok_or(ArenaPlanError::InvalidProtocolGeometry(
                    "fused blake_g is missing its shared one-word dummy",
                ))?
            }
            (Some(_), false) => logical_buffer_id(
                logical,
                component.node.id,
                part.part,
                BufferPurpose::SubcomponentInputs,
                0,
            )?,
            (None, _) => push_buffer_id(
                logical,
                Some(component.node.id),
                Some(part.part),
                BufferPurpose::WitnessSubDummy,
                0,
                requirements.sub_words,
                witness_lifetime,
            )?,
        };
        let native_input_producer = (component.node.id == "partial_ec_mul_generic"
            && proof.components.iter().any(|candidate| {
                candidate.node.id == "ec_op_builtin" && candidate.runtime.is_present()
            }))
        .then_some("ec_op_builtin");
        let input_compact = if native_input_producer.is_none() {
            append_witness_input_compact(
                logical,
                proof,
                component.node,
                &program,
                part,
                &input_columns,
                persistent,
            )?
        } else {
            None
        };
        let input_gather = if input_compact.is_none() && native_input_producer.is_none() {
            append_witness_input_gather(
                logical,
                proof,
                component.node,
                &program,
                part,
                &input_columns,
                persistent,
            )?
        } else {
            None
        };
        let input_seed = if native_input_producer.is_none() {
            append_witness_input_seed(
                logical,
                component.node,
                &program,
                part,
                &input_columns,
                persistent,
            )?
        } else {
            None
        };
        if usize::from(input_gather.is_some())
            + usize::from(input_seed.is_some())
            + usize::from(input_compact.is_some())
            > 1
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "recorded witness has multiple device input materializers",
            ));
        }
        components.push(LogicalWitnessComponent {
            component: component.node.id,
            part: part.part,
            n_real_rows: usize::try_from(part.n_real_rows)
                .map_err(|_| ArenaPlanError::SizeOverflow)?,
            blake_g_fused: component_blake_g_fused,
            native_input_producer,
            program,
            requirements,
            input_columns,
            input_pointers,
            execution_table_pointers,
            execution_table_strides,
            output_columns,
            output_pointers,
            multiplicity_columns: Vec::new(),
            multiplicity_pointers,
            multiplicity_dummy,
            lookup_words,
            sub_words,
            input_gather,
            input_seed,
            input_compact,
        });
    }
    Ok(LogicalWitnessWorkspace { components })
}

fn append_ec_op_buffers(
    logical: &mut Vec<LogicalBuffer>,
    proof: &ProofPlan,
    execution_tables: Option<&LogicalExecutionTablesWorkspace>,
    witness: &LogicalWitnessWorkspace,
    multiplicity: Option<&LogicalGraphAMultiplicityWorkspace>,
) -> Result<Option<LogicalEcOpWorkspace>, ArenaPlanError> {
    let Some(component) = proof
        .components
        .iter()
        .find(|component| component.node.id == "ec_op_builtin" && component.runtime.is_present())
    else {
        return Ok(None);
    };
    if execution_tables.is_none() {
        // The sealed-witness compatibility plan has no execution-table arena
        // and keeps using its already materialized EC-op columns.
        return Ok(None);
    }
    let multiplicity = multiplicity.ok_or(ArenaPlanError::InvalidProtocolGeometry(
        "resident EC-op requires Graph-A multiplicity destinations",
    ))?;
    let parts = capacity_parts(&component.runtime.rows)?;
    let [part] = parts.as_slice() else {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "resident EC-op must have one main trace part",
        ));
    };
    if part.part != TracePartId::Main {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "resident EC-op must have one main trace part",
        ));
    }
    let row_count = usize::try_from(part.padded_rows).map_err(|_| ArenaPlanError::SizeOverflow)?;
    let trace_columns = (0..cairo_air::components::ec_op_builtin::N_TRACE_COLUMNS)
        .map(|ordinal| {
            logical_buffer_id(
                logical,
                "ec_op_builtin",
                TracePartId::Main,
                BufferPurpose::BaseTrace,
                u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?,
            )
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let lookup_words = logical_buffer_id(
        logical,
        "ec_op_builtin",
        TracePartId::Main,
        BufferPurpose::LookupInputs,
        0,
    )?;
    let partial = witness
        .components
        .iter()
        .find(|candidate| candidate.component == "partial_ec_mul_generic")
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "resident EC-op is missing partial_ec_mul_generic consumer",
        ))?;
    let count = |name: &'static str| {
        multiplicity
            .multiplicities
            .iter()
            .find_map(|(candidate, id)| (*candidate == name).then_some(*id))
            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "resident EC-op multiplicity destination is missing",
            ))
    };
    let address_counts = count("memory_address_to_id")?;
    let big_counts = count("memory_id_to_big")?;
    let small_counts = count("memory_id_to_big#small")?;
    let range_check_8_counts = count("range_check_8")?;
    let requirements = ec_op_workspace_requirements(
        row_count,
        EcOpMultiplicityGeometry {
            address_count_words: logical[address_counts.0 as usize].len_words,
            big_count_words: logical[big_counts.0 as usize].len_words,
            small_count_words: logical[small_counts.0 as usize].len_words,
            range_check_8_count_words: logical[range_check_8_counts.0 as usize].len_words,
        },
    )
    .map_err(|_| ArenaPlanError::InvalidProtocolGeometry("invalid resident EC-op geometry"))?;
    // The native ec_op writer materializes 127 partial-input columns: the 126
    // the consumer recording binds (data + enabler) plus the one-past-end iota
    // column the witness kernel computes in-kernel and never reads. The
    // consumer's witness plan therefore owns 126 slots and the plan adds an
    // ec_op-owned slot for the writer's iota column so the producer ABI stays
    // intact (see recorded_witness_inputs' 126-input contract for the same
    // seam on the consumer side).
    if partial.input_columns.len() + 1 != requirements.partial_input_column_words.len()
        || partial
            .input_columns
            .iter()
            .any(|id| logical[id.0 as usize].len_words != requirements.partial_row_count)
    {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "EC-op direct partial_ec_mul_generic input geometry mismatch",
        ));
    }
    let iota_words = *requirements.partial_input_column_words.last().ok_or(
        ArenaPlanError::InvalidProtocolGeometry("EC-op partial input requirements are empty"),
    )?;
    let iota_column = push_buffer_id(
        logical,
        Some("ec_op_builtin"),
        Some(TracePartId::Main),
        BufferPurpose::EcOpPartialIota,
        0,
        iota_words,
        BufferLifetime::at(ProofEpoch::Witness),
    )?;
    let mut partial_input_columns = partial.input_columns.clone();
    partial_input_columns.push(iota_column);
    let segment_start = push_buffer_id(
        logical,
        Some("ec_op_builtin"),
        Some(TracePartId::Main),
        BufferPurpose::EcOpSegmentStart,
        0,
        1,
        BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?,
    )?;
    Ok(Some(LogicalEcOpWorkspace {
        requirements,
        trace_columns,
        lookup_words,
        partial_input_columns,
        segment_start,
        address_counts,
        big_counts,
        small_counts,
        range_check_8_counts,
    }))
}

fn append_witness_input_seed(
    logical: &mut Vec<LogicalBuffer>,
    node: &'static crate::schedule::ComponentNode,
    program: &WitnessProgram,
    part: TracePartShape,
    input_columns: &[LogicalBufferId],
    persistent: BufferLifetime,
) -> Result<Option<LogicalWitnessInputSeed>, ArenaPlanError> {
    let Some(scalar_words) =
        stwo_cairo_prover::witness::jit_prove_backend::recorded_device_seed_scalar_count(node.id)
    else {
        return Ok(None);
    };
    let geometry = stwo_cairo_prover::witness::jit_prove_backend::recorded_input_geometry(node.id)
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "device-seeded witness has no input geometry",
        ))?;
    let n_inputs = usize::try_from(program.n_inputs).map_err(|_| ArenaPlanError::SizeOverflow)?;
    let include_enabler = geometry.enabler_slot.is_some_and(|slot| slot < n_inputs);
    let include_iota = geometry.iota_slot.is_some_and(|slot| slot < n_inputs);
    let n_real_rows =
        usize::try_from(part.n_real_rows).map_err(|_| ArenaPlanError::SizeOverflow)?;
    let consumer_rows =
        usize::try_from(part.padded_rows).map_err(|_| ArenaPlanError::SizeOverflow)?;
    let requirements = stwo_backend_cuda::witness_input_seed_requirements(
        scalar_words,
        n_real_rows,
        consumer_rows,
        include_enabler,
        include_iota,
    )
    .map_err(ArenaPlanError::WitnessInputGather)?;
    if requirements.consumer_input_column_words.len() != n_inputs
        || input_columns.len() != n_inputs
        || geometry.enabler_slot != include_enabler.then_some(scalar_words)
        || geometry.iota_slot != include_iota.then_some(scalar_words + usize::from(include_enabler))
    {
        return Err(ArenaPlanError::WitnessInputGatherProgramWidth {
            component: node.id,
            expected_inputs: n_inputs,
            actual_inputs: requirements.consumer_input_column_words.len(),
        });
    }
    let scalar_values = push_buffer_id(
        logical,
        Some(node.id),
        Some(part.part),
        BufferPurpose::WitnessInputSeedScalars,
        0,
        requirements.scalar_words,
        persistent,
    )?;
    let output_pointers = push_buffer_id(
        logical,
        Some(node.id),
        Some(part.part),
        BufferPurpose::WitnessInputSeedOutputPointers,
        0,
        requirements.output_pointer_words,
        persistent,
    )?;
    Ok(Some(LogicalWitnessInputSeed {
        requirements,
        scalar_values,
        output_pointers,
    }))
}

fn append_graph_a_multiplicity_buffers(
    logical: &mut Vec<LogicalBuffer>,
    preprocessed: &LogicalPreprocessedWorkspace,
    plan: GraphAMultiplicityPlan,
    public_memory_entries: usize,
    blake_g_fused: bool,
) -> Result<LogicalGraphAMultiplicityWorkspace, ArenaPlanError> {
    let persistent = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?;
    let witness = BufferLifetime::at(ProofEpoch::Witness);

    let mut multiplicities = Vec::with_capacity(plan.fixed.len() + plan.runtime.len());
    let mut multiplicity_word_sizes = Vec::with_capacity(plan.fixed.len() + plan.runtime.len());
    let mut multiplicity_ids = BTreeMap::new();
    for fixed in &plan.fixed {
        let id = push_buffer_id(
            logical,
            Some(fixed.component),
            Some(TracePartId::Main),
            BufferPurpose::FixedMultiplicity,
            0,
            fixed.slab_words,
            witness,
        )?;
        multiplicities.push((fixed.component, id));
        multiplicity_word_sizes.push(fixed.slab_words);
        multiplicity_ids.insert(fixed.component, id);
    }
    for runtime in &plan.runtime {
        let (component, part) = match runtime.destination {
            "memory_address_to_id" => (Some("memory_address_to_id"), Some(TracePartId::Main)),
            "memory_id_to_big" | "memory_id_to_big#small" => (Some("memory_id_to_big"), None),
            _ => {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "unknown runtime multiplicity destination",
                ))
            }
        };
        let id = push_buffer_id(
            logical,
            component,
            part,
            BufferPurpose::RuntimeMultiplicity,
            u32::try_from(multiplicities.len()).map_err(|_| ArenaPlanError::SizeOverflow)?,
            runtime.words,
            witness,
        )?;
        if multiplicity_ids.insert(runtime.destination, id).is_some() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "duplicate runtime multiplicity destination",
            ));
        }
        multiplicities.push((runtime.destination, id));
        multiplicity_word_sizes.push(runtime.words);
    }
    let clear_requirements =
        stwo_backend_cuda::witness_feed_clear_workspace_requirements(&multiplicity_word_sizes)
            .map_err(ArenaPlanError::WitnessFeed)?;
    let clear_pointers = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FixedMultiplicityClearPointers,
        0,
        clear_requirements.destination_pointer_words,
        persistent,
    )?;
    let clear_lengths = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FixedMultiplicityClearLengths,
        0,
        clear_requirements.destination_length_words,
        persistent,
    )?;

    let mut lut_ids = BTreeMap::new();
    for (ordinal, lut) in plan.luts.iter().enumerate() {
        let id = push_buffer_id(
            logical,
            None,
            None,
            BufferPurpose::WitnessFeedLut,
            u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?,
            lut.words,
            persistent,
        )?;
        lut_ids.insert(lut.state_param, id);
    }

    let feed_count = plan.feeds.len();
    let mut feeds = Vec::with_capacity(feed_count);
    for (ordinal, feed) in plan.feeds.into_iter().enumerate() {
        let ordinal = u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?;
        let lut_tables = feed
            .lut_families
            .iter()
            .map(|family| {
                lut_ids
                    .get(family)
                    .copied()
                    .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                        "missing canonical witness-feed LUT",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let multiplicity_destinations = feed
            .destination_components
            .iter()
            .map(|component| {
                multiplicity_ids.get(component).copied().ok_or(
                    ArenaPlanError::InvalidProtocolGeometry(
                        "missing fixed multiplicity destination",
                    ),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(binding) = blake_g_fused
            .then(|| blake_g_fused_feed_binding(&feed))
            .flatten()
        {
            feeds.push(LogicalRecordedMultiplicityFeed::BlakeGFused {
                plan: feed,
                lut_tables: binding.lut_indices.map(|index| lut_tables[index]),
                multiplicity_destinations: binding
                    .destination_indices
                    .map(|index| multiplicity_destinations[index]),
            });
            continue;
        }
        let source = logical_buffer_id(
            logical,
            feed.producer,
            TracePartId::Main,
            BufferPurpose::SubcomponentInputs,
            0,
        )?;
        let descriptors = push_buffer_id(
            logical,
            Some(feed.producer),
            Some(TracePartId::Main),
            BufferPurpose::WitnessFeedDescriptors,
            ordinal,
            feed.requirements.descriptor_words,
            persistent,
        )?;
        let lut_pointers = push_buffer_id(
            logical,
            Some(feed.producer),
            Some(TracePartId::Main),
            BufferPurpose::WitnessFeedLutPointers,
            ordinal,
            feed.requirements.lut_pointer_words,
            persistent,
        )?;
        let multiplicity_pointers = push_buffer_id(
            logical,
            Some(feed.producer),
            Some(TracePartId::Main),
            BufferPurpose::WitnessFeedMultiplicityPointers,
            ordinal,
            feed.requirements.multiplicity_pointer_words,
            persistent,
        )?;
        feeds.push(LogicalRecordedMultiplicityFeed::Generic(
            LogicalGenericRecordedMultiplicityFeed {
                plan: feed,
                source,
                descriptors,
                lut_tables,
                lut_pointers,
                multiplicity_destinations,
                multiplicity_pointers,
            },
        ));
    }

    let public_memory_seed = (public_memory_entries != 0)
        .then(|| {
            let address = *multiplicity_ids.get("memory_address_to_id").ok_or(
                ArenaPlanError::InvalidProtocolGeometry("missing public-memory address counts"),
            )?;
            let big = *multiplicity_ids.get("memory_id_to_big").ok_or(
                ArenaPlanError::InvalidProtocolGeometry("missing public-memory big counts"),
            )?;
            let small = *multiplicity_ids.get("memory_id_to_big#small").ok_or(
                ArenaPlanError::InvalidProtocolGeometry("missing public-memory small counts"),
            )?;
            let seed = plan_public_memory_multiplicity_seed(
                public_memory_entries,
                logical[address.0 as usize].len_words,
                logical[big.0 as usize].len_words,
                logical[small.0 as usize].len_words,
            )
            .map_err(ArenaPlanError::MultiplicityPlan)?;
            if !seed.lut_families.is_empty()
                || seed.destination_components
                    != [
                        "memory_address_to_id_state",
                        "memory_id_to_big_state",
                        "memory_id_to_big_state#small",
                    ]
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "public-memory multiplicity seed descriptor routing drifted",
                ));
            }
            let ordinal = u32::try_from(feed_count).map_err(|_| ArenaPlanError::SizeOverflow)?;
            let source = push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::PublicMemoryMultiplicitySeed,
                0,
                seed.requirements.source_words,
                persistent,
            )?;
            let descriptors = push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::WitnessFeedDescriptors,
                ordinal,
                seed.requirements.descriptor_words,
                persistent,
            )?;
            let lut_pointers = push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::WitnessFeedLutPointers,
                ordinal,
                seed.requirements.lut_pointer_words,
                persistent,
            )?;
            let multiplicity_pointers = push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::WitnessFeedMultiplicityPointers,
                ordinal,
                seed.requirements.multiplicity_pointer_words,
                persistent,
            )?;
            Ok::<_, ArenaPlanError>(LogicalGenericRecordedMultiplicityFeed {
                plan: seed,
                source,
                descriptors,
                lut_tables: Vec::new(),
                lut_pointers,
                multiplicity_destinations: vec![address, big, small],
                multiplicity_pointers,
            })
        })
        .transpose()?;

    let mut fixed_tables = Vec::with_capacity(plan.fixed.len());
    for fixed in plan.fixed {
        let requirements = fixed.materializer.requirements();
        let sources = fixed
            .materializer
            .preprocessed_sources()
            .iter()
            .map(|identity| {
                let column = preprocessed
                    .columns
                    .iter()
                    .find(|column| column.identity == *identity)
                    .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                        "fixed table references an unknown preprocessed column",
                    ))?;
                if let Some(index) = pedersen_points_18_column_index(identity) {
                    if column.log_size != PEDERSEN_POINTS_18_LOG_SIZE
                        || fixed.row_count != PEDERSEN_POINTS_18_ROW_COUNT
                        || column.evaluations.is_some()
                    {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "process-owned pedersen source has invalid geometry or arena retention",
                        ));
                    }
                    return Ok(LogicalFixedTableSource::RegisteredPedersen18 { column: index });
                }
                let evaluation =
                    column
                        .evaluations
                        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                            "fixed table preprocessed evaluations were not retained",
                        ))?;
                if logical[evaluation.0 as usize].len_words != fixed.row_count {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "fixed-table preprocessed source has the wrong row count",
                    ));
                }
                Ok(LogicalFixedTableSource::Arena(evaluation))
            })
            .collect::<Result<Vec<_>, ArenaPlanError>>()?;
        let multiplicity = *multiplicity_ids.get(fixed.component).ok_or(
            ArenaPlanError::InvalidProtocolGeometry("missing fixed-table multiplicity slab"),
        )?;
        let source_pointers = (requirements.source_pointer_words != 0)
            .then(|| {
                push_buffer_id(
                    logical,
                    Some(fixed.component),
                    Some(TracePartId::Main),
                    BufferPurpose::FixedTableSourcePointers,
                    0,
                    requirements.source_pointer_words,
                    persistent,
                )
            })
            .transpose()?;
        let multiplicity_pointers = push_buffer_id(
            logical,
            Some(fixed.component),
            Some(TracePartId::Main),
            BufferPurpose::FixedTableMultiplicityPointers,
            0,
            requirements.multiplicity_pointer_words,
            persistent,
        )?;
        let trace_multiplicity_columns = push_buffer_id(
            logical,
            Some(fixed.component),
            Some(TracePartId::Main),
            BufferPurpose::FixedTableTraceMultiplicityColumns,
            0,
            requirements.trace_mapping_words,
            persistent,
        )?;
        let trace_outputs = (0..requirements.trace_output_count)
            .map(|ordinal| {
                logical_buffer_id(
                    logical,
                    fixed.component,
                    TracePartId::Main,
                    BufferPurpose::BaseTrace,
                    u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?,
                )
            })
            .collect::<Result<Vec<_>, ArenaPlanError>>()?;
        let trace_output_pointers = push_buffer_id(
            logical,
            Some(fixed.component),
            Some(TracePartId::Main),
            BufferPurpose::FixedTableTraceOutputPointers,
            0,
            requirements.trace_pointer_words,
            persistent,
        )?;
        let lookup_descriptors = push_buffer_id(
            logical,
            Some(fixed.component),
            Some(TracePartId::Main),
            BufferPurpose::FixedTableLookupDescriptors,
            0,
            requirements.lookup_descriptor_words,
            persistent,
        )?;
        let lookup_output = logical_buffer_id(
            logical,
            fixed.component,
            TracePartId::Main,
            BufferPurpose::LookupInputs,
            0,
        )?;
        let lookup_output_pointers = push_buffer_id(
            logical,
            Some(fixed.component),
            Some(TracePartId::Main),
            BufferPurpose::FixedTableLookupOutputPointers,
            0,
            requirements.lookup_pointer_words,
            persistent,
        )?;
        fixed_tables.push(LogicalFixedTableMaterializer {
            plan: fixed,
            sources,
            multiplicity,
            source_pointers,
            multiplicity_pointers,
            trace_multiplicity_columns,
            trace_outputs,
            trace_output_pointers,
            lookup_descriptors,
            lookup_output,
            lookup_output_pointers,
        });
    }

    let memory_traces = plan
        .memory_traces
        .map(|memory| {
            let outputs = |part, count: usize| {
                (0..count)
                    .map(|ordinal| {
                        logical_buffer_id(
                            logical,
                            match part {
                                TracePartId::Main => "memory_address_to_id",
                                TracePartId::MemoryBig(_) | TracePartId::MemorySmall => {
                                    "memory_id_to_big"
                                }
                            },
                            part,
                            BufferPurpose::BaseTrace,
                            u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?,
                        )
                    })
                    .collect::<Result<Vec<_>, ArenaPlanError>>()
            };
            let address_outputs = outputs(
                TracePartId::Main,
                cairo_air::components::memory_address_to_id::N_TRACE_COLUMNS,
            )?;
            let big_parts = memory
                .big_parts
                .iter()
                .map(|part| {
                    Ok(LogicalMemoryTracePart {
                        part: part.part,
                        source_offset: part.source_offset,
                        row_count: part.row_count,
                        outputs: outputs(
                            part.part,
                            cairo_air::components::memory_id_to_big::BIG_N_COLUMNS,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()?;
            let small_part = LogicalMemoryTracePart {
                part: memory.small_part.part,
                source_offset: memory.small_part.source_offset,
                row_count: memory.small_part.row_count,
                outputs: outputs(
                    memory.small_part.part,
                    cairo_air::components::memory_id_to_small::N_TRACE_COLUMNS,
                )?,
            };
            let rc99_lut = *lut_ids.get("range_check_9_9_state").ok_or(
                ArenaPlanError::InvalidProtocolGeometry("missing canonical rc9_9 LUT"),
            )?;
            let rc99_counts = *multiplicity_ids.get("range_check_9_9").ok_or(
                ArenaPlanError::InvalidProtocolGeometry("missing rc9_9 multiplicity slab"),
            )?;
            if logical[rc99_lut.0 as usize].len_words != memory.rc99_lut_words
                || logical[rc99_counts.0 as usize].len_words != memory.rc99_count_words
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "rc9_9 memory-feed binding geometry drifted",
                ));
            }
            Ok::<_, ArenaPlanError>(LogicalMemoryBaseTraces {
                plan: memory,
                address_outputs,
                big_parts,
                small_part,
                rc99_lut,
                rc99_counts,
            })
        })
        .transpose()?;

    Ok(LogicalGraphAMultiplicityWorkspace {
        topology_hash: plan.topology_hash,
        coverage_gaps: plan.coverage_gaps,
        blockers: plan.blockers,
        multiplicities,
        clear_requirements,
        clear_pointers,
        clear_lengths,
        feeds,
        public_memory_seed,
        fixed_tables,
        memory_traces,
    })
}

pub(crate) fn topological_component_order(
    proof: &ProofPlan,
) -> Result<Vec<&crate::plan::ComponentPlan>, ArenaPlanError> {
    let known = proof
        .components
        .iter()
        .map(|component| component.node.id)
        .collect::<BTreeSet<_>>();
    let mut placed = BTreeSet::new();
    let mut ordered = Vec::with_capacity(proof.components.len());
    while ordered.len() < proof.components.len() {
        let level = proof
            .components
            .iter()
            .filter(|component| !placed.contains(component.node.id))
            .filter(|component| {
                component.node.inputs.iter().all(|input| match input {
                    InputEdge::Producer { of, .. } => known.contains(of) && placed.contains(of),
                    InputEdge::ExecTables | InputEdge::DeviceTable(_) => true,
                }) && component
                    .node
                    .capacity_inputs
                    .iter()
                    .all(|feed| known.contains(feed.from) && placed.contains(feed.from))
            })
            .collect::<Vec<_>>();
        if level.is_empty() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "proof witness dependency graph is cyclic or dangling",
            ));
        }
        for component in level {
            placed.insert(component.node.id);
            ordered.push(component);
        }
    }
    Ok(ordered)
}

fn append_witness_input_compact(
    logical: &mut Vec<LogicalBuffer>,
    proof: &ProofPlan,
    node: &'static crate::schedule::ComponentNode,
    program: &WitnessProgram,
    part: TracePartShape,
    input_columns: &[LogicalBufferId],
    persistent: BufferLifetime,
) -> Result<Option<LogicalWitnessInputCompact>, ArenaPlanError> {
    let Some(compaction) =
        stwo_cairo_prover::witness::jit_prove_backend::recorded_input_compaction_geometry(node.id)
    else {
        return Ok(None);
    };
    let input_geometry = stwo_cairo_prover::witness::jit_prove_backend::recorded_input_geometry(
        node.id,
    )
    .ok_or(ArenaPlanError::InvalidProtocolGeometry(
        "recorded compact input has no recorder geometry",
    ))?;
    let n_inputs = usize::try_from(program.n_inputs).map_err(|_| ArenaPlanError::SizeOverflow)?;
    let layout = WitnessInputCompactLayout {
        tuple_words: compaction.tuple_words,
        key_words: compaction.key_words,
        consumer_input_count: n_inputs,
        enabler_slot: input_geometry.enabler_slot,
        iota_slot: input_geometry.iota_slot,
        multiplicity_slot: compaction.multiplicity_slot,
    };

    let mut sources = Vec::new();
    let mut edges = Vec::new();
    for edge in node.inputs {
        let InputEdge::Producer {
            of,
            word_base,
            words_per_instance,
            n_instances,
        } = edge
        else {
            continue;
        };
        let Some(producer_plan) = proof
            .components
            .iter()
            .find(|component| component.node.id == *of && component.runtime.is_present())
        else {
            continue;
        };
        let producer_parts = capacity_parts(&producer_plan.runtime.rows)?;
        if producer_parts.len() != 1 || producer_parts[0].part != TracePartId::Main {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "recorded compact input producer is not one main trace",
            ));
        }
        let producer_rows = usize::try_from(producer_parts[0].padded_rows)
            .map_err(|_| ArenaPlanError::SizeOverflow)?;
        sources.push(logical_buffer_id(
            logical,
            *of,
            TracePartId::Main,
            BufferPurpose::SubcomponentInputs,
            0,
        )?);
        edges.push(WitnessInputGatherEdge {
            producer_rows,
            word_base: *word_base as usize,
            words_per_instance: *words_per_instance as usize,
            n_instances: *n_instances as usize,
        });
    }
    let consumer_rows =
        usize::try_from(part.padded_rows).map_err(|_| ArenaPlanError::SizeOverflow)?;
    let requirements = witness_input_compact_requirements(&edges, layout, consumer_rows)
        .map_err(ArenaPlanError::WitnessInputGather)?;
    if input_columns.len() != n_inputs || requirements.consumer_input_column_words.len() != n_inputs
    {
        return Err(ArenaPlanError::WitnessInputGatherProgramWidth {
            component: node.id,
            expected_inputs: n_inputs,
            actual_inputs: requirements.consumer_input_column_words.len(),
        });
    }
    for (source, edge) in sources.iter().zip(&requirements.edges) {
        let source_words = logical
            .iter()
            .find(|buffer| buffer.id == *source)
            .map(|buffer| buffer.len_words)
            .ok_or(ArenaPlanError::MissingBinding(*source))?;
        if source_words < edge.required_source_words {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "recorded compact input exceeds producer sub words",
            ));
        }
    }

    macro_rules! push_persistent {
        ($purpose:expr, $ordinal:expr, $words:expr) => {
            push_buffer_id(
                logical,
                Some(node.id),
                Some(part.part),
                $purpose,
                $ordinal,
                $words,
                persistent,
            )?
        };
    }
    macro_rules! push_scratch {
        ($purpose:expr, $ordinal:expr, $words:expr) => {
            push_buffer_id(
                logical,
                Some(node.id),
                Some(part.part),
                $purpose,
                $ordinal,
                $words,
                BufferLifetime::at(ProofEpoch::Witness),
            )?
        };
    }
    let source_pointers = push_persistent!(
        BufferPurpose::WitnessInputCompactSourcePointers,
        0,
        requirements.source_pointer_words
    );
    let descriptors = push_persistent!(
        BufferPurpose::WitnessInputCompactDescriptors,
        0,
        requirements.descriptor_words
    );
    let output_pointers = push_persistent!(
        BufferPurpose::WitnessInputCompactOutputPointers,
        0,
        requirements.output_pointer_words
    );
    let tuple_scratch = push_scratch!(
        BufferPurpose::WitnessInputCompactTupleScratch,
        0,
        requirements.tuple_scratch_words
    );
    let sort_keys_a = push_scratch!(
        BufferPurpose::WitnessInputCompactSortKey,
        0,
        requirements.sort_key_words
    );
    let sort_keys_b = push_scratch!(
        BufferPurpose::WitnessInputCompactSortKey,
        1,
        requirements.sort_key_words
    );
    let sort_indices_a = push_scratch!(
        BufferPurpose::WitnessInputCompactSortIndex,
        0,
        requirements.sort_index_words
    );
    let sort_indices_b = push_scratch!(
        BufferPurpose::WitnessInputCompactSortIndex,
        1,
        requirements.sort_index_words
    );
    let run_heads = push_scratch!(
        BufferPurpose::WitnessInputCompactRunHeads,
        0,
        requirements.run_words
    );
    let run_positions = push_scratch!(
        BufferPurpose::WitnessInputCompactRunPositions,
        0,
        requirements.run_words
    );
    let n_unique = push_scratch!(BufferPurpose::WitnessInputCompactUniqueCount, 0, 1);
    let sort_temp = push_scratch!(
        BufferPurpose::WitnessInputCompactSortTemp,
        0,
        requirements.sort_temp_words
    );
    let scan_temp = push_scratch!(
        BufferPurpose::WitnessInputCompactScanTemp,
        0,
        requirements.scan_temp_words
    );
    Ok(Some(LogicalWitnessInputCompact {
        requirements,
        sources,
        source_pointers,
        descriptors,
        output_pointers,
        tuple_scratch,
        sort_keys_a,
        sort_keys_b,
        sort_indices_a,
        sort_indices_b,
        run_heads,
        run_positions,
        n_unique,
        sort_temp,
        scan_temp,
    }))
}

fn append_witness_input_gather(
    logical: &mut Vec<LogicalBuffer>,
    proof: &ProofPlan,
    node: &'static crate::schedule::ComponentNode,
    program: &WitnessProgram,
    part: TracePartShape,
    input_columns: &[LogicalBufferId],
    persistent: BufferLifetime,
) -> Result<Option<LogicalWitnessInputGather>, ArenaPlanError> {
    let producer_edges =
        node.inputs
            .iter()
            .filter_map(|edge| match edge {
                InputEdge::Producer {
                    of,
                    word_base,
                    words_per_instance,
                    n_instances,
                } if proof.components.iter().any(|component| {
                    component.node.id == *of && component.runtime.is_present()
                }) =>
                {
                    Some((*of, *word_base, *words_per_instance, *n_instances))
                }
                InputEdge::ExecTables | InputEdge::DeviceTable(_) => None,
                InputEdge::Producer { .. } => None,
            })
            .collect::<Vec<_>>();
    if producer_edges.is_empty() {
        return Ok(None);
    }

    let input_geometry =
        stwo_cairo_prover::witness::jit_prove_backend::recorded_input_geometry(node.id).ok_or(
            ArenaPlanError::InvalidProtocolGeometry("recorded producer edge has no input geometry"),
        )?;
    let n_inputs = usize::try_from(program.n_inputs).map_err(|_| ArenaPlanError::SizeOverflow)?;
    let include_enabler = input_geometry
        .enabler_slot
        .is_some_and(|ordinal| ordinal < n_inputs);
    let include_iota = input_geometry
        .iota_slot
        .is_some_and(|ordinal| ordinal < n_inputs);

    let mut producers = Vec::with_capacity(producer_edges.len());
    let mut sources = Vec::with_capacity(producer_edges.len());
    let mut edges = Vec::with_capacity(producer_edges.len());
    for (producer, word_base, words_per_instance, n_instances) in producer_edges {
        let producer_plan = proof
            .components
            .iter()
            .find(|component| component.node.id == producer)
            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "recorded witness input edge has no producer plan",
            ))?;
        let producer_parts = capacity_parts(&producer_plan.runtime.rows)?;
        if producer_parts.len() != 1 || producer_parts[0].part != TracePartId::Main {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "recorded witness input edge producer is not one main trace",
            ));
        }
        let producer_rows = usize::try_from(producer_parts[0].padded_rows)
            .map_err(|_| ArenaPlanError::SizeOverflow)?;
        producers.push(producer);
        sources.push(logical_buffer_id(
            logical,
            producer,
            TracePartId::Main,
            BufferPurpose::SubcomponentInputs,
            0,
        )?);
        edges.push(WitnessInputGatherEdge {
            producer_rows,
            word_base: word_base as usize,
            words_per_instance: words_per_instance as usize,
            n_instances: n_instances as usize,
        });
    }

    let requirements = witness_input_gather_requirements(&edges, include_enabler, include_iota)
        .map_err(ArenaPlanError::WitnessInputGather)?;
    let expected_real_rows =
        usize::try_from(part.n_real_rows).map_err(|_| ArenaPlanError::SizeOverflow)?;
    let expected_padded_rows =
        usize::try_from(part.padded_rows).map_err(|_| ArenaPlanError::SizeOverflow)?;
    if requirements.total_real_rows != expected_real_rows
        || requirements.consumer_rows != expected_padded_rows
    {
        return Err(ArenaPlanError::WitnessInputGatherGeometry {
            component: node.id,
            expected_real_rows,
            actual_real_rows: requirements.total_real_rows,
            expected_padded_rows,
            actual_padded_rows: requirements.consumer_rows,
        });
    }
    if requirements.consumer_input_column_words.len() != n_inputs
        || input_columns.len() != n_inputs
        || input_geometry
            .enabler_slot
            .filter(|&ordinal| ordinal < n_inputs)
            != include_enabler.then_some(requirements.input_width)
        || input_geometry
            .iota_slot
            .filter(|&ordinal| ordinal < n_inputs)
            != include_iota.then_some(requirements.input_width + usize::from(include_enabler))
    {
        return Err(ArenaPlanError::WitnessInputGatherProgramWidth {
            component: node.id,
            expected_inputs: n_inputs,
            actual_inputs: requirements.consumer_input_column_words.len(),
        });
    }
    for (source, edge) in sources.iter().zip(&requirements.edges) {
        let source_words = logical
            .iter()
            .find(|buffer| buffer.id == *source)
            .map(|buffer| buffer.len_words)
            .ok_or(ArenaPlanError::MissingBinding(*source))?;
        if source_words < edge.required_source_words {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "recorded witness input edge exceeds producer sub words",
            ));
        }
    }

    let source_pointers = push_buffer_id(
        logical,
        Some(node.id),
        Some(part.part),
        BufferPurpose::WitnessInputGatherSourcePointers,
        0,
        requirements.source_pointer_words,
        persistent,
    )?;
    let descriptors = push_buffer_id(
        logical,
        Some(node.id),
        Some(part.part),
        BufferPurpose::WitnessInputGatherDescriptors,
        0,
        requirements.descriptor_words,
        persistent,
    )?;
    let output_pointers = push_buffer_id(
        logical,
        Some(node.id),
        Some(part.part),
        BufferPurpose::WitnessInputGatherOutputPointers,
        0,
        requirements.output_pointer_words,
        persistent,
    )?;
    Ok(Some(LogicalWitnessInputGather {
        requirements,
        producers,
        sources,
        source_pointers,
        descriptors,
        output_pointers,
    }))
}

fn append_relation_buffers(
    logical: &mut Vec<LogicalBuffer>,
    execution: RelationExecutionPlan,
    late_coefficient_ownership: &LateCoefficientOwnershipPlan,
    transition_aliases: &mut Vec<(LogicalBufferId, LogicalBufferId)>,
) -> Result<LogicalRelationWorkspace, ArenaPlanError> {
    let launch_mode = CAIRO_RELATION_LAUNCH_MODE;
    let requirements =
        execution
            .requirements_for_mode(launch_mode)
            .map_err(|error| match error {
                RelationExecutionError::BackendPlan(error) => ArenaPlanError::Relation(error),
                other => ArenaPlanError::RelationExecution(other),
            })?;
    let source_plan = execution
        .source_plan()
        .map_err(ArenaPlanError::RelationExecution)?;
    if source_plan.len() != requirements.instances.len() {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "relation source plan count drifted from requirements",
        ));
    }
    // Captured relation kernels dereference these immutable tables on every
    // warm replay, so their lifetime crosses the Assemble -> Ingest cycle.
    let persistent = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?;
    let interaction = BufferLifetime::at(ProofEpoch::Interaction);
    // Relation preparation uploads placeholder challenge values before replay;
    // the transcript overwrites them with the real values at Interaction.
    let challenge = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Composition)?;
    let claimed = BufferLifetime::new(ProofEpoch::Interaction, ProofEpoch::Composition)?;
    let descriptors = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationDescriptors,
        0,
        requirements.descriptor_words.max(1),
        persistent,
    )?;
    let alphas = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationAlphaPowers,
        0,
        requirements.alpha_words.max(1),
        challenge,
    )?;
    let z = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationZ,
        0,
        requirements.z_words.max(1),
        challenge,
    )?;
    let inverse_scratch = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationInverseScratch,
        0,
        requirements.inverse_words.max(1),
        interaction,
    )?;
    let reduction_a = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationReductionA,
        0,
        requirements.reduction_words.max(1),
        interaction,
    )?;
    let reduction_b = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationReductionB,
        0,
        requirements.reduction_words.max(1),
        interaction,
    )?;
    let scan_eval_scratch = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationScanEvalScratch,
        0,
        requirements.scan_eval_words.max(1),
        interaction,
    )?;
    let scan_temp_scratch = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationScanTempScratch,
        0,
        requirements.scan_temp_words.max(1),
        interaction,
    )?;
    let scan_descriptors = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationScanDescriptors,
        0,
        requirements.scan_descriptor_words.max(1),
        interaction,
    )?;
    let fraction_pointers = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationFractionPointers,
        0,
        requirements.fraction_pointer_words.max(1),
        persistent,
    )?;
    let fraction_geometry = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::RelationFractionGeometry,
        0,
        requirements.fraction_geometry_words.max(1),
        persistent,
    )?;

    let mut instances = Vec::with_capacity(requirements.instances.len());
    for (ordinal, (requirement, source)) in
        requirements.instances.iter().zip(&source_plan).enumerate()
    {
        let batch = execution.batches.get(requirement.batch_index).ok_or(
            ArenaPlanError::InvalidProtocolGeometry(
                "relation requirement references an unknown batch",
            ),
        )?;
        if *batch != source.batch || requirement.instance_index != source.instance_index {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "relation source plan order drifted from requirements",
            ));
        }
        let part = source.part;
        let mut output_coordinates: Vec<_> = logical
            .iter()
            .filter(|buffer| {
                buffer.component == Some(batch.component)
                    && buffer.part == Some(part)
                    && buffer.purpose == BufferPurpose::InteractionTrace
            })
            .map(|buffer| (buffer.ordinal, buffer.id, buffer.len_words))
            .collect();
        output_coordinates.sort_unstable_by_key(|(ordinal, ..)| *ordinal);

        // `memory_id_to_big` predates the generic component logup metadata: its
        // big and small interaction widths are expressed only by the generated
        // relation graph. Materialize those graph-derived coordinates directly
        // in the arena so the relation kernels write the columns committed by
        // PreparedCommitGraph without an intermediate scatter/copy.
        let graph_derived_outputs = output_coordinates.is_empty()
            && matches!(
                batch.trace_part,
                RelationTracePart::EachMemoryBig | RelationTracePart::MemorySmall
            );
        let output_coordinates = if graph_derived_outputs {
            (0..requirement.output_coordinate_count)
                .map(|coordinate| {
                    push_buffer_id(
                        logical,
                        Some(batch.component),
                        Some(part),
                        BufferPurpose::InteractionTrace,
                        u32::try_from(coordinate).map_err(|_| ArenaPlanError::SizeOverflow)?,
                        requirement.output_coordinate_words,
                        BufferLifetime::at(ProofEpoch::Interaction),
                    )
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()?
        } else {
            if output_coordinates.len() != requirement.output_coordinate_count {
                return Err(ArenaPlanError::RelationOutputCountMismatch {
                    component: batch.component,
                    part,
                    expected: requirement.output_coordinate_count,
                    actual: output_coordinates.len(),
                });
            }
            output_coordinates
                .into_iter()
                .enumerate()
                .map(|(coordinate, (actual_ordinal, id, len_words))| {
                    let ordinal =
                        u32::try_from(coordinate).map_err(|_| ArenaPlanError::SizeOverflow)?;
                    if actual_ordinal != ordinal {
                        return Err(ArenaPlanError::MissingRelationOutput {
                            component: batch.component,
                            part,
                            ordinal,
                        });
                    }
                    if len_words < requirement.output_coordinate_words {
                        return Err(ArenaPlanError::RelationOutputTooSmall {
                            component: batch.component,
                            part,
                            ordinal,
                            required_words: requirement.output_coordinate_words,
                            actual_words: len_words,
                        });
                    }
                    Ok(id)
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()?
        };
        // Every relation output is committed as a polynomial. Materialize the
        // logical coefficient identity even when the selected interpolation
        // path transforms a dead evaluation in place.
        let mut coefficient_coordinates: Vec<_> = logical
            .iter()
            .filter(|buffer| {
                buffer.component == Some(batch.component)
                    && buffer.part == Some(part)
                    && buffer.purpose == BufferPurpose::InteractionCoefficients
            })
            .map(|buffer| (buffer.ordinal, buffer.id, buffer.len_words))
            .collect();
        coefficient_coordinates.sort_unstable_by_key(|(ordinal, ..)| *ordinal);
        let coefficient_coordinates = if coefficient_coordinates.is_empty()
            && matches!(
                batch.trace_part,
                RelationTracePart::EachMemoryBig | RelationTracePart::MemorySmall
            ) {
            (0..requirement.output_coordinate_count)
                .map(|coordinate| {
                    let ordinal =
                        u32::try_from(coordinate).map_err(|_| ArenaPlanError::SizeOverflow)?;
                    push_buffer_id(
                        logical,
                        Some(batch.component),
                        Some(part),
                        BufferPurpose::InteractionCoefficients,
                        ordinal,
                        requirement.output_coordinate_words,
                        BufferLifetime::new(
                            ProofEpoch::InteractionCommit,
                            late_coefficient_ownership
                                .final_consumer(OpenedColumnSource::Trace {
                                    component: batch.component,
                                    part,
                                    purpose: BufferPurpose::InteractionCoefficients,
                                    ordinal,
                                })
                                .map_err(ArenaPlanError::InvalidProtocolGeometry)?,
                        )?,
                    )
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()?
        } else {
            if coefficient_coordinates.len() != requirement.output_coordinate_count {
                return Err(ArenaPlanError::RelationOutputCountMismatch {
                    component: batch.component,
                    part,
                    expected: requirement.output_coordinate_count,
                    actual: coefficient_coordinates.len(),
                });
            }
            coefficient_coordinates
                .into_iter()
                .enumerate()
                .map(|(coordinate, (actual_ordinal, id, len_words))| {
                    let ordinal =
                        u32::try_from(coordinate).map_err(|_| ArenaPlanError::SizeOverflow)?;
                    if actual_ordinal != ordinal {
                        return Err(ArenaPlanError::MissingRelationOutput {
                            component: batch.component,
                            part,
                            ordinal,
                        });
                    }
                    if len_words < requirement.output_coordinate_words {
                        return Err(ArenaPlanError::RelationOutputTooSmall {
                            component: batch.component,
                            part,
                            ordinal,
                            required_words: requirement.output_coordinate_words,
                            actual_words: len_words,
                        });
                    }
                    Ok(id)
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()?
        };
        if graph_derived_outputs {
            transition_aliases.extend(
                output_coordinates
                    .iter()
                    .copied()
                    .zip(coefficient_coordinates.iter().copied()),
            );
        }
        let ordinal = u32::try_from(ordinal).map_err(|_| ArenaPlanError::SizeOverflow)?;
        instances.push(LogicalRelationInstanceSlots {
            source_pointers: push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::RelationSourcePointers,
                ordinal,
                requirement.source_pointer_words.max(1),
                persistent,
            )?,
            output_pointers: push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::RelationOutputPointers,
                ordinal,
                requirement.output_pointer_words.max(1),
                persistent,
            )?,
            output_coordinates,
            denominators: push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::RelationDenominators,
                ordinal,
                requirement.denominator_words.max(1),
                interaction,
            )?,
            claimed_sum: push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::RelationClaimedSum,
                ordinal,
                requirement.claimed_sum_words.max(1),
                claimed,
            )?,
        });
    }

    Ok(LogicalRelationWorkspace {
        execution,
        source_plan,
        launch_mode,
        requirements,
        descriptors,
        alphas,
        z,
        inverse_scratch,
        reduction_a,
        reduction_b,
        scan_eval_scratch,
        scan_temp_scratch,
        scan_descriptors,
        fraction_pointers,
        fraction_geometry,
        instances,
    })
}

fn validate_commitment_sources(
    logical: &[LogicalBuffer],
    commitments: &[LogicalCommitWorkspace],
) -> Result<(), ArenaPlanError> {
    for commitment in commitments {
        for (sources, log_sizes) in commitment
            .grouped_column_sources
            .iter()
            .zip(&commitment.grouped_column_log_sizes)
        {
            for (&source, &log_size) in sources.iter().zip(log_sizes) {
                let valid_tree = matches!(
                    (commitment.id, source),
                    (
                        CommitmentTreeId::Preprocessed,
                        CommitmentColumnSource::Preprocessed { .. }
                    ) | (
                        CommitmentTreeId::Base,
                        CommitmentColumnSource::Trace {
                            purpose: BufferPurpose::BaseCoefficients,
                            ..
                        }
                    ) | (
                        CommitmentTreeId::Interaction,
                        CommitmentColumnSource::Trace {
                            purpose: BufferPurpose::InteractionCoefficients,
                            ..
                        }
                    ) | (
                        CommitmentTreeId::Composition,
                        CommitmentColumnSource::Composition { .. }
                    )
                );
                if !valid_tree {
                    return Err(ArenaPlanError::CommitmentSourceTreeMismatch {
                        tree: commitment.id,
                        source,
                    });
                }
                let buffer = logical
                    .iter()
                    .find(|buffer| match source {
                        CommitmentColumnSource::Preprocessed { ordinal } => {
                            buffer.component.is_none()
                                && buffer.part.is_none()
                                && buffer.purpose == BufferPurpose::PreprocessedCoefficients
                                && buffer.ordinal == ordinal
                        }
                        CommitmentColumnSource::Trace {
                            component,
                            part,
                            purpose,
                            ordinal,
                        } => {
                            buffer.component == Some(component)
                                && buffer.part == Some(part)
                                && buffer.purpose == purpose
                                && buffer.ordinal == ordinal
                        }
                        CommitmentColumnSource::Composition { ordinal } => {
                            buffer.component.is_none()
                                && buffer.part.is_none()
                                && buffer.purpose == BufferPurpose::CompositionCoefficients
                                && buffer.ordinal == ordinal
                        }
                    })
                    .ok_or(ArenaPlanError::MissingCommitmentSource {
                        tree: commitment.id,
                        source,
                    })?;
                let expected_words = checked_pow2(log_size)?;
                if buffer.len_words != expected_words {
                    return Err(ArenaPlanError::CommitmentSourceSizeMismatch {
                        tree: commitment.id,
                        source,
                        expected_words,
                        actual_words: buffer.len_words,
                    });
                }
            }
        }
    }
    Ok(())
}

fn opened_source_logical_id(
    logical: &[LogicalBuffer],
    source: OpenedColumnSource,
) -> Option<LogicalBufferId> {
    logical
        .iter()
        .find(|buffer| match source {
            OpenedColumnSource::Preprocessed { ordinal } => {
                buffer.component.is_none()
                    && buffer.part.is_none()
                    && buffer.purpose == BufferPurpose::PreprocessedCoefficients
                    && buffer.ordinal == ordinal
            }
            OpenedColumnSource::Trace {
                component,
                part,
                purpose,
                ordinal,
            } => {
                buffer.component == Some(component)
                    && buffer.part == Some(part)
                    && buffer.purpose == purpose
                    && buffer.ordinal == ordinal
            }
            OpenedColumnSource::Composition { ordinal } => {
                buffer.component.is_none()
                    && buffer.part.is_none()
                    && buffer.purpose == BufferPurpose::CompositionCoefficients
                    && buffer.ordinal == ordinal
            }
        })
        .map(|buffer| buffer.id)
}

fn address_free_composition_trace(
    oods: &OodsGeometry,
) -> Result<CompositionTraceTopology, ArenaPlanError> {
    let mut trees = vec![Vec::new(), Vec::new(), Vec::new()];
    for (flat, column) in oods.columns.iter().enumerate() {
        let tree = match column.source {
            OpenedColumnSource::Preprocessed { .. } => 0,
            OpenedColumnSource::Trace {
                purpose: BufferPurpose::BaseCoefficients,
                ..
            } => 1,
            OpenedColumnSource::Trace {
                purpose: BufferPurpose::InteractionCoefficients,
                ..
            } => 2,
            OpenedColumnSource::Composition { .. } => continue,
            OpenedColumnSource::Trace { .. } => {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "composition trace source is not a coefficient column",
                ));
            }
        };
        trees[tree].push(CompositionCoefficientSource {
            slot: ArenaSlotId(
                u32::try_from(flat)
                    .map_err(|_| ArenaPlanError::SizeOverflow)?
                    .checked_add(1)
                    .ok_or(ArenaPlanError::SizeOverflow)?,
            ),
            log_size: column.coefficient_log_size,
        });
    }
    if trees.iter().any(Vec::is_empty) {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "composition trace topology has an empty opened tree",
        ));
    }
    Ok(CompositionTraceTopology { trees })
}

fn append_transcript_buffers(
    logical: &mut Vec<LogicalBuffer>,
    protocol: &ProtocolGeometry,
) -> Result<LogicalTranscriptWorkspace, ArenaPlanError> {
    let live = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?;
    let requirements = protocol.transcript.requirements.clone();
    let state = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::TranscriptState,
        0,
        requirements.state_words,
        live,
    )?;
    let boundary_snapshots = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::TranscriptBoundarySnapshots,
        0,
        requirements.boundary_snapshot_words,
        live,
    )?;
    let input_snapshots = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::TranscriptInputSnapshots,
        0,
        requirements.input_snapshot_words,
        live,
    )?;
    let output_snapshots = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::TranscriptOutputSnapshots,
        0,
        requirements.output_snapshot_words,
        live,
    )?;
    let inputs = requirements
        .inputs
        .iter()
        .map(|requirement| {
            Ok((
                requirement.id,
                push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::TranscriptInput,
                    requirement.id.0,
                    requirement.min_words,
                    live,
                )?,
            ))
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let outputs = requirements
        .outputs
        .iter()
        .map(|requirement| {
            Ok((
                requirement.id,
                push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::TranscriptOutput,
                    requirement.id.0,
                    requirement.min_words,
                    live,
                )?,
            ))
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    Ok(LogicalTranscriptWorkspace {
        schedule_key: protocol.transcript.schedule_key,
        requirements,
        state,
        boundary_snapshots,
        input_snapshots,
        output_snapshots,
        inputs,
        outputs,
    })
}

fn interpolation_batch_geometry(
    geometry: &CommitmentGeometry,
    mode: InterpolationLaunchMode,
) -> Result<Vec<(u32, Vec<CommitmentColumnSource>)>, ArenaPlanError> {
    if !matches!(
        geometry.id,
        CommitmentTreeId::Base | CommitmentTreeId::Interaction
    ) {
        return Ok(Vec::new());
    }
    match mode {
        InterpolationLaunchMode::StageWiseCopyThenInPlace => {
            let mut output = Vec::new();
            for (sources, logs) in geometry
                .grouped_column_sources
                .iter()
                .zip(&geometry.grouped_column_log_sizes)
            {
                let mut start = 0usize;
                while start < logs.len() {
                    let mut end = start + 1;
                    while end < logs.len() && logs[end] == logs[start] {
                        end += 1;
                    }
                    output.push((logs[start], sources[start..end].to_vec()));
                    start = end;
                }
            }
            Ok(output)
        }
        InterpolationLaunchMode::StageFusedOutOfPlace => {
            let mut by_log = BTreeMap::<u32, Vec<CommitmentColumnSource>>::new();
            for (sources, logs) in geometry
                .grouped_column_sources
                .iter()
                .zip(&geometry.grouped_column_log_sizes)
            {
                if sources.len() != logs.len() {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "commitment interpolation source/log width mismatch",
                    ));
                }
                for (&source, &log) in sources.iter().zip(logs) {
                    by_log.entry(log).or_default().push(source);
                }
            }
            Ok(by_log.into_iter().collect())
        }
    }
}

fn retained_oods_source(
    commitments: &[LogicalCommitWorkspace],
    source: OpenedColumnSource,
    coefficient_log_size: u32,
) -> Result<LogicalBufferId, ArenaPlanError> {
    let tree = match source {
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::BaseCoefficients,
            ..
        } => CommitmentTreeId::Base,
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::InteractionCoefficients,
            ..
        } => CommitmentTreeId::Interaction,
        OpenedColumnSource::Composition { .. } => CommitmentTreeId::Composition,
        _ => {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "evaluation-backed OODS source is not dynamic",
            ));
        }
    };
    let commitment = commitments
        .iter()
        .find(|commitment| commitment.id == tree)
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "evaluation-backed OODS commitment is missing",
        ))?;
    let mut selected = None;
    for (group, (sources, logs)) in commitment
        .grouped_column_sources
        .iter()
        .zip(&commitment.grouped_column_log_sizes)
        .enumerate()
    {
        for (column, (&candidate, &log_size)) in sources.iter().zip(logs).enumerate() {
            if OpenedColumnSource::from(candidate) != source {
                continue;
            }
            if selected.is_some() || log_size != coefficient_log_size {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "evaluation-backed OODS source mapping drifted",
                ));
            }
            let retained_for_decommit = commitment
                .decommit_evaluation_groups
                .get(group)
                .copied()
                .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "OODS retained-evaluation group closure is missing",
            ))?;
            let retained_for_numerator = commitment
                .numerator_evaluation_groups
                .get(group)
                .copied()
                .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                    "OODS retained-evaluation group closure is missing",
                ))?;
            if !retained_for_decommit && !retained_for_numerator {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "OODS retained-evaluation group closure is missing",
                ));
            }
            selected = Some(
                commitment
                    .retained_evaluations
                    .get(group)
                    .and_then(Option::as_ref)
                    .and_then(|columns| columns.get(column))
                    .copied()
                    .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                        "planned retained OODS evaluation is missing",
                    ))?,
            );
        }
    }
    selected.ok_or(ArenaPlanError::InvalidProtocolGeometry(
        "planned retained OODS source is absent",
    ))
}

fn retained_quotient_numerator_source(
    commitments: &[LogicalCommitWorkspace],
    source: OpenedColumnSource,
    coefficient_log_size: u32,
) -> Result<LogicalBufferId, ArenaPlanError> {
    let tree = match source {
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::BaseCoefficients,
            ..
        } => CommitmentTreeId::Base,
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::InteractionCoefficients,
            ..
        } => CommitmentTreeId::Interaction,
        OpenedColumnSource::Composition { .. } => CommitmentTreeId::Composition,
        _ => {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "evaluation-backed quotient numerator source is not dynamic",
            ));
        }
    };
    let commitment = commitments
        .iter()
        .find(|commitment| commitment.id == tree)
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "evaluation-backed quotient numerator commitment is missing",
        ))?;
    let mut selected = None;
    for (group_index, (sources, logs)) in commitment
        .grouped_column_sources
        .iter()
        .zip(&commitment.grouped_column_log_sizes)
        .enumerate()
    {
        for (column_index, (&candidate, &log_size)) in sources.iter().zip(logs).enumerate() {
            if OpenedColumnSource::from(candidate) != source {
                continue;
            }
            if selected.is_some() || log_size != coefficient_log_size {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "evaluation-backed quotient numerator source mapping drifted",
                ));
            }
            if !commitment
                .numerator_evaluation_groups
                .get(group_index)
                .copied()
                .unwrap_or(false)
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "quotient numerator group closure is missing",
                ));
            }
            let retained = commitment
                .retained_evaluations
                .get(group_index)
                .and_then(Option::as_ref)
                .and_then(|columns| columns.get(column_index))
                .copied()
                .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                    "planned retained quotient numerator evaluation is missing",
                ))?;
            selected = Some(retained);
        }
    }
    selected.ok_or(ArenaPlanError::InvalidProtocolGeometry(
        "planned retained quotient numerator source is absent",
    ))
}

fn direct_composition_logical_source(
    commitments: &[LogicalCommitWorkspace],
    column: &crate::direct_composition_retention::DirectCompositionColumn,
) -> Result<LogicalBufferId, ArenaPlanError> {
    let commitment = commitments
        .iter()
        .find(|commitment| commitment.id == column.tree)
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "direct composition commitment is missing",
        ))?;
    if !commitment
        .direct_composition_evaluation_groups
        .get(column.group)
        .copied()
        .unwrap_or(false)
    {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "direct composition group closure is missing",
        ));
    }
    commitment
        .retained_evaluations
        .get(column.group)
        .and_then(Option::as_ref)
        .and_then(|group| group.get(column.column_in_group))
        .copied()
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "direct composition evaluation output is missing",
        ))
}

fn append_protocol_buffers(
    logical: &mut Vec<LogicalBuffer>,
    protocol: &ProtocolGeometry,
    composition: &CompositionPlan,
    retained_preprocessed_evaluations: Option<&BTreeSet<&'static str>>,
    late_coefficient_ownership: &LateCoefficientOwnershipPlan,
) -> Result<
    (
        LogicalPreprocessedWorkspace,
        Vec<LogicalCommitWorkspace>,
        LogicalCompositionWorkspace,
        LogicalOodsWorkspace,
        LogicalQuotientNumeratorWorkspace,
        LogicalQuotientWorkspace,
        LogicalFriWorkspace,
        LogicalFinalFriPowWorkspace,
        LogicalDecommitWorkspace,
        LogicalTranscriptWorkspace,
    ),
    ArenaPlanError,
> {
    let commit_requirements = protocol
        .commitments
        .iter()
        .map(|commitment| {
            commitment_workspace_requirements(protocol.identity.commit_mode, commitment)
                .map_err(ArenaPlanError::Commit)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fri_config = protocol.fri_workspace_config()?;
    let fri_requirements = fri_workspace_requirements(fri_config).map_err(ArenaPlanError::Fri)?;
    let final_fri_requirements =
        fri_final_workspace_requirements(fri_config).map_err(ArenaPlanError::FriFinal)?;
    let pow_requirements = blake2s_pow_workspace_requirements();
    let quotient_config = protocol.quotient_workspace_config();
    let quotient_requirements = quotient_workspace_requirements(
        quotient_config,
        &protocol.quotient.partial_numerator_log_sizes,
    )
    .map_err(ArenaPlanError::Quotient)?;
    let oods_config = protocol.oods_workspace_config();
    let oods_source_kinds = protocol.oods_source_kinds()?;
    let oods_topologies = protocol.oods.column_topologies(&oods_source_kinds)?;
    let oods_requirements =
        oods_workspace_requirements(oods_config, &oods_topologies).map_err(ArenaPlanError::Oods)?;
    let quotient_numerator_config = protocol.quotient_numerator_workspace_config()?;
    let quotient_numerator_topologies = protocol.quotient_numerator_topologies()?;
    let quotient_numerator_requirements = quotient_numerator_workspace_requirements(
        quotient_numerator_config,
        &quotient_numerator_topologies,
    )
    .map_err(ArenaPlanError::QuotientNumerator)?;
    let composition_trace_shape = address_free_composition_trace(&protocol.oods)?;
    let composition_requirements = composition_workspace_requirements_with_retention(
        composition,
        &composition_trace_shape,
        protocol.identity.composition_launch_mode,
        protocol.direct_composition_retention.as_ref(),
    )
    .map_err(ArenaPlanError::Composition)?;
    let max_commitment_twiddle_words = commit_requirements
        .iter()
        .map(|requirements| match requirements {
            ModeAwareCommitWorkspaceRequirements::FullLifting(requirements) => {
                requirements.twiddle_words
            }
            ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements) => {
                requirements.leaves.twiddle_words
            }
        })
        .max()
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "proof has no commitment workspace",
        ))?;
    let forward_twiddle_words =
        max_commitment_twiddle_words.max(quotient_requirements.forward_twiddle_words);
    // These three trees are staged once per workspace and reused by every warm
    // proof. Their semantic lifetime crosses the Assemble -> Ingest cycle, so
    // model the full cycle explicitly instead of relying on incidental overlap
    // with the current proof-assembly buffers.
    let cached_workspace = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?;
    let forward_twiddles = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::ForwardTwiddles,
        0,
        forward_twiddle_words,
        cached_workspace,
    )?;
    let inverse_twiddles = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::InverseTwiddles,
        0,
        fri_requirements.twiddle_words,
        cached_workspace,
    )?;
    let preprocessed_inverse_twiddles = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::PreprocessedInverseTwiddles,
        0,
        max_commitment_twiddle_words,
        BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Decommit)?,
    )?;
    let quotient_inverse_twiddles = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientInverseTwiddles,
        0,
        quotient_requirements.inverse_twiddle_words,
        cached_workspace,
    )?;
    let logical_transcript = append_transcript_buffers(logical, protocol)?;

    let preprocessed_columns = protocol
        .oods
        .columns
        .iter()
        .filter_map(|column| match column.source {
            OpenedColumnSource::Preprocessed { ordinal } => {
                Some((ordinal, column.coefficient_log_size))
            }
            _ => None,
        })
        .map(|(ordinal, log_size)| {
            let identity = protocol
                .preprocessed_column_ids
                .get(ordinal as usize)
                .cloned()
                .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                    "missing preprocessed column identity",
                ))?;
            let retain_evaluation = retained_preprocessed_evaluations
                .is_some_and(|retained| retained.contains(identity.as_str()));
            Ok(LogicalPreprocessedColumn {
                identity,
                ordinal,
                log_size,
                evaluations: retain_evaluation
                    .then(|| {
                        push_buffer_id(
                            logical,
                            None,
                            None,
                            BufferPurpose::PreprocessedEvaluations,
                            ordinal,
                            checked_pow2(log_size)?,
                            BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?,
                        )
                    })
                    .transpose()?,
                coefficients: push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::PreprocessedCoefficients,
                    ordinal,
                    checked_pow2(log_size)?,
                    cached_workspace,
                )?,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    if preprocessed_columns.is_empty() {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "OODS topology has no preprocessed columns",
        ));
    }
    let mut interpolation_groups = std::collections::BTreeMap::<u32, Vec<u32>>::new();
    for column in &preprocessed_columns {
        interpolation_groups
            .entry(column.log_size)
            .or_default()
            .push(column.ordinal);
    }
    let interpolation_batches = interpolation_groups
        .into_iter()
        .enumerate()
        .map(|(batch, (log_size, column_ordinals))| {
            Ok(LogicalPreprocessedInterpolationBatch {
                log_size,
                coefficient_pointers: push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::PreprocessedInterpolationPointers,
                    u32::try_from(batch).map_err(|_| ArenaPlanError::SizeOverflow)?,
                    column_ordinals
                        .len()
                        .checked_mul(
                            core::mem::size_of::<*mut u32>().div_ceil(core::mem::size_of::<u32>()),
                        )
                        .ok_or(ArenaPlanError::SizeOverflow)?,
                    BufferLifetime::at(ProofEpoch::Ingest),
                )?,
                column_ordinals,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let logical_preprocessed = LogicalPreprocessedWorkspace {
        columns: preprocessed_columns,
        interpolation_batches,
        inverse_twiddles: preprocessed_inverse_twiddles,
    };

    let mut logical_commitments = Vec::with_capacity(protocol.commitments.len());
    for (commitment_index, (geometry, requirements)) in protocol
        .commitments
        .iter()
        .zip(commit_requirements)
        .enumerate()
    {
        let at = BufferLifetime::at(geometry.created);
        // The fixed commitment is produced once and cache-hit on every later
        // proof. All of its retained Merkle/LDE outputs must survive the full
        // workspace cycle; dynamic commitments are regenerated each replay.
        let retained = if geometry.id == CommitmentTreeId::Preprocessed {
            cached_workspace
        } else {
            BufferLifetime::new(geometry.created, ProofEpoch::Decommit)?
        };
        // Captured kernels dereference these tables on every warm replay. Their
        // contents are commitment-specific and therefore cannot alias another
        // segment's descriptors even though the compute epochs are disjoint.
        let descriptor = cached_workspace;
        let mut next_ordinal = 0u32;
        let mut ordinal = || {
            let local = next_ordinal;
            next_ordinal = next_ordinal
                .checked_add(1)
                .ok_or(ArenaPlanError::SizeOverflow)?;
            let prefix = u32::try_from(commitment_index)
                .map_err(|_| ArenaPlanError::SizeOverflow)?
                .checked_shl(20)
                .ok_or(ArenaPlanError::SizeOverflow)?;
            prefix
                .checked_add(local)
                .ok_or(ArenaPlanError::SizeOverflow)
        };
        let (
            leaf_words,
            merkle_scratch_words,
            retained_layer_requirements,
            tail_pointer_words,
            tail_output_requirements,
        ) = match &requirements {
            ModeAwareCommitWorkspaceRequirements::FullLifting(requirements) => (
                requirements.leaf_state_words,
                requirements.merkle_scratch_words,
                requirements.retained_layers.as_slice(),
                requirements.tail_pointer_words,
                requirements.tail_outputs.as_slice(),
            ),
            ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements) => (
                requirements.merkle.leaf_words,
                requirements.merkle.merkle_scratch_words,
                requirements.merkle.retained_layers.as_slice(),
                requirements.merkle.tail_pointer_words,
                requirements.merkle.tail_outputs.as_slice(),
            ),
        };
        let leaf_state = push_buffer_id(
            logical,
            None,
            None,
            BufferPurpose::MerkleLeafState,
            ordinal()?,
            leaf_words,
            if geometry.config.unretained_bottom_layers == 0 {
                retained
            } else {
                at
            },
        )?;
        let merkle_scratch = merkle_scratch_words
            .map(|words| {
                push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::MerkleLayerScratch,
                    ordinal()?,
                    words,
                    at,
                )
            })
            .transpose()?;
        let retained_layers = retained_layer_requirements
            .iter()
            .map(|layer| {
                push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::RetainedMerkleLayers,
                    ordinal()?,
                    layer.words,
                    retained,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tail_level_ptrs = tail_pointer_words
            .map(|words| {
                push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::MerkleTailPointers,
                    ordinal()?,
                    words,
                    descriptor,
                )
            })
            .transpose()?;
        let tail_outputs = tail_output_requirements
            .iter()
            .map(|layer| {
                push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::RetainedMerkleLayers,
                    ordinal()?,
                    layer.words,
                    retained,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if geometry.retained_evaluation_groups.len() != geometry.grouped_column_log_sizes.len() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "commitment opening policy does not match its groups",
            ));
        }
        if geometry.direct_composition_evaluation_groups.len()
            != geometry.grouped_column_log_sizes.len()
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "direct composition retention policy does not match its groups",
            ));
        }
        if geometry.numerator_evaluation_groups.len() != geometry.grouped_column_log_sizes.len() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "quotient numerator retention policy does not match its groups",
            ));
        }
        let retained_evaluations = geometry
            .grouped_column_log_sizes
            .iter()
            .zip(&geometry.retained_evaluation_groups)
            .zip(&geometry.direct_composition_evaluation_groups)
            .zip(&geometry.numerator_evaluation_groups)
            .map(
                |(((logs, &keep_decommit), &keep_direct), &keep_numerator)| {
                    (keep_decommit || keep_direct || keep_numerator)
                        .then(|| {
                            logs.iter()
                                .map(|&log_size| {
                                    let evaluation_log = log_size
                                        .checked_add(geometry.config.log_blowup_factor)
                                        .ok_or(ArenaPlanError::SizeOverflow)?;
                                    push_buffer_id(
                                        logical,
                                        None,
                                        None,
                                        BufferPurpose::CommitRetainedEvaluation,
                                        ordinal()?,
                                        checked_pow2(evaluation_log)?,
                                        if geometry.id == CommitmentTreeId::Preprocessed {
                                            cached_workspace
                                        } else if keep_decommit {
                                            retained
                                        } else if keep_numerator {
                                            BufferLifetime::new(
                                                geometry.created,
                                                ProofEpoch::Quotient,
                                            )?
                                        } else {
                                            BufferLifetime::new(
                                                geometry.created,
                                                ProofEpoch::Composition,
                                            )?
                                        },
                                    )
                                })
                                .collect::<Result<Vec<_>, ArenaPlanError>>()
                        })
                        .transpose()
                },
            )
            .collect::<Result<Vec<_>, ArenaPlanError>>()?;
        let leaf_workspace = match &requirements {
            ModeAwareCommitWorkspaceRequirements::FullLifting(requirements) => {
                let lde_tile = push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::CommitLdeTile,
                    ordinal()?,
                    requirements.lde_tile_words,
                    at,
                )?;
                let groups = requirements
                    .groups
                    .iter()
                    .map(|group| {
                        let column_ptrs = push_buffer_id(
                            logical,
                            None,
                            None,
                            BufferPurpose::CommitColumnPointers,
                            ordinal()?,
                            group.column_pointer_words,
                            descriptor,
                        )?;
                        let column_log_sizes = push_buffer_id(
                            logical,
                            None,
                            None,
                            BufferPurpose::CommitColumnLogSizes,
                            ordinal()?,
                            group.column_log_size_words,
                            descriptor,
                        )?;
                        let batches = group
                            .batches
                            .iter()
                            .map(|batch| {
                                allocate_commit_batch(logical, &mut ordinal, batch, descriptor)
                            })
                            .collect::<Result<Vec<_>, ArenaPlanError>>()?;
                        Ok(LogicalCommitGroupSlots {
                            column_ptrs,
                            column_log_sizes,
                            batches,
                        })
                    })
                    .collect::<Result<Vec<_>, ArenaPlanError>>()?;
                LogicalCommitLeafWorkspace::FullLifting { lde_tile, groups }
            }
            ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements) => {
                let lde_scratch = requirements
                    .leaves
                    .lde_scratch_words
                    .map(|words| {
                        push_buffer_id(
                            logical,
                            None,
                            None,
                            BufferPurpose::CommitLdeTile,
                            ordinal()?,
                            words,
                            at,
                        )
                    })
                    .transpose()?;
                let state_ping = push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::CommitProgressiveStatePing,
                    ordinal()?,
                    requirements.leaves.state_ping_words,
                    at,
                )?;
                let state_pong = requirements
                    .leaves
                    .state_pong_words
                    .map(|words| {
                        push_buffer_id(
                            logical,
                            None,
                            None,
                            BufferPurpose::CommitProgressiveStatePong,
                            ordinal()?,
                            words,
                            at,
                        )
                    })
                    .transpose()?;
                let batches = requirements
                    .leaves
                    .batches
                    .iter()
                    .map(|batch| {
                        allocate_progressive_commit_batch(logical, &mut ordinal, batch, descriptor)
                    })
                    .collect::<Result<Vec<_>, ArenaPlanError>>()?;
                LogicalCommitLeafWorkspace::DomainProgressive {
                    lde_scratch,
                    state_ping,
                    state_pong,
                    batches,
                }
            }
        };
        let interpolation_batches =
            interpolation_batch_geometry(geometry, protocol.identity.interpolation_mode)?
                .into_iter()
                .map(|(log_size, sources)| {
                    let pointer_words = sources
                        .len()
                        .checked_mul(
                            core::mem::size_of::<usize>().div_ceil(core::mem::size_of::<u32>()),
                        )
                        .ok_or(ArenaPlanError::SizeOverflow)?;
                    Ok(LogicalInterpolationBatch {
                        log_size,
                        sources,
                        input_pointers: push_buffer_id(
                            logical,
                            None,
                            None,
                            BufferPurpose::InterpolationInputPointers,
                            ordinal()?,
                            pointer_words,
                            descriptor,
                        )?,
                        output_pointers: push_buffer_id(
                            logical,
                            None,
                            None,
                            BufferPurpose::InterpolationOutputPointers,
                            ordinal()?,
                            pointer_words,
                            descriptor,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()?;
        logical_commitments.push(LogicalCommitWorkspace {
            id: geometry.id,
            interpolation_mode: protocol.identity.interpolation_mode,
            config: geometry.config,
            grouped_column_log_sizes: geometry.grouped_column_log_sizes.clone(),
            grouped_column_sources: geometry.grouped_column_sources.clone(),
            requirements,
            twiddles: forward_twiddles,
            leaf_state,
            merkle_scratch,
            retained_layers,
            tail_level_ptrs,
            tail_outputs,
            retained_evaluations,
            decommit_evaluation_groups: geometry.retained_evaluation_groups.clone(),
            direct_composition_evaluation_groups: geometry
                .direct_composition_evaluation_groups
                .clone(),
            numerator_evaluation_groups: geometry.numerator_evaluation_groups.clone(),
            leaf_workspace,
            interpolation_batches,
        });
    }

    let output_values = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientTile,
        0,
        quotient_requirements.output_value_words,
        BufferLifetime::new(ProofEpoch::Quotient, ProofEpoch::Decommit)?,
    )?;
    let quotient_live = BufferLifetime::at(ProofEpoch::Quotient);
    let quotient_descriptor = cached_workspace;
    let partial_numerators = protocol
        .quotient
        .partial_numerator_log_sizes
        .iter()
        .enumerate()
        .map(|(source, &log_size)| {
            let first_ordinal = u32::try_from(source)
                .ok()
                .and_then(|source| source.checked_mul(4))
                .ok_or(ArenaPlanError::SizeOverflow)?;
            let source_words = checked_pow2(log_size)?;
            let mut coordinate = |coordinate: u32| {
                push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::QuotientPartialNumerator,
                    first_ordinal
                        .checked_add(coordinate)
                        .ok_or(ArenaPlanError::SizeOverflow)?,
                    source_words,
                    quotient_live,
                )
            };
            Ok(LogicalQuotientNumeratorSource {
                log_size,
                coordinates: [
                    coordinate(0)?,
                    coordinate(1)?,
                    coordinate(2)?,
                    coordinate(3)?,
                ],
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    // Generic quotient preparation uploads placeholder constants into these
    // destinations before the resident numerator overwrites them at Quotient.
    // Their declared lifetime must cover that real setup-time write; otherwise
    // arena coloring may alias them with a retained direct-composition input
    // that is still live and silently corrupt the first proof before replay.
    let quotient_prepare_live = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Quotient)?;
    let sample_points = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientSamplePoints,
        0,
        quotient_requirements.sample_point_words,
        quotient_prepare_live,
    )?;
    let first_linear_terms = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientFirstLinearTerms,
        0,
        quotient_requirements.first_linear_term_words,
        quotient_prepare_live,
    )?;
    let partial_log_sizes = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientPartialLogSizes,
        0,
        quotient_requirements.partial_log_size_words,
        quotient_descriptor,
    )?;
    let partial_coordinate_ptrs = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientPartialCoordinatePointers,
        0,
        quotient_requirements.partial_pointer_words,
        quotient_descriptor,
    )?;
    let subdomain_coordinate_ptrs = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientSubdomainCoordinatePointers,
        0,
        quotient_requirements.coordinate_pointer_words,
        quotient_descriptor,
    )?;
    let output_coordinate_ptrs = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientOutputCoordinatePointers,
        0,
        quotient_requirements.coordinate_pointer_words,
        quotient_descriptor,
    )?;
    let coefficient_sizes = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientCoefficientSizes,
        0,
        quotient_requirements.coefficient_size_words,
        quotient_descriptor,
    )?;
    let subdomain_values = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientSubdomainValues,
        0,
        quotient_requirements.subdomain_value_words,
        quotient_live,
    )?;
    let logical_quotient = LogicalQuotientWorkspace {
        config: quotient_config,
        requirements: quotient_requirements,
        forward_twiddles,
        inverse_subdomain_twiddles: quotient_inverse_twiddles,
        partial_numerators: partial_numerators.clone(),
        sample_points,
        first_linear_terms,
        partial_log_sizes,
        partial_coordinate_ptrs,
        subdomain_coordinate_ptrs,
        output_coordinate_ptrs,
        coefficient_sizes,
        subdomain_values,
        output_values,
    };
    let fri_scratch = BufferLifetime::at(ProofEpoch::Fri);
    let fri_live = BufferLifetime::new(ProofEpoch::Fri, ProofEpoch::Decommit)?;
    let fri_descriptor = cached_workspace;
    let evaluation_ping = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FriPing,
        0,
        fri_requirements.evaluation_ping_words,
        fri_scratch,
    )?;
    let evaluation_pong = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FriPong,
        0,
        fri_requirements.evaluation_pong_words,
        fri_scratch,
    )?;
    let input_coordinate_ptrs = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FriInputCoordinatePointers,
        0,
        fri_requirements.coordinate_pointer_words,
        fri_descriptor,
    )?;
    let ping_coordinate_ptrs = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FriPingCoordinatePointers,
        0,
        fri_requirements.coordinate_pointer_words,
        fri_descriptor,
    )?;
    let pong_coordinate_ptrs = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FriPongCoordinatePointers,
        0,
        fri_requirements.coordinate_pointer_words,
        fri_descriptor,
    )?;
    let retained_tree_evaluations = fri_requirements
        .trees
        .iter()
        .enumerate()
        .skip(1)
        .map(|(tree_index, tree)| {
            push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::FriRetainedEvaluation,
                u32::try_from(tree_index).map_err(|_| ArenaPlanError::SizeOverflow)?,
                tree.evaluation_words,
                fri_live,
            )
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let retained_tree_coordinate_ptrs = fri_requirements
        .trees
        .iter()
        .enumerate()
        .skip(1)
        .map(|(tree_index, _)| {
            push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::FriRetainedCoordinatePointers,
                u32::try_from(tree_index).map_err(|_| ArenaPlanError::SizeOverflow)?,
                fri_requirements.coordinate_pointer_words,
                fri_descriptor,
            )
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let folding_challenges = fri_requirements
        .rounds
        .iter()
        .enumerate()
        .map(|(round, _)| {
            push_buffer_id(
                logical,
                None,
                None,
                BufferPurpose::FriFoldingChallenge,
                u32::try_from(round).map_err(|_| ArenaPlanError::SizeOverflow)?,
                stwo_backend_cuda::FRI_CHALLENGE_WORDS,
                fri_live,
            )
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let fri_trees = fri_requirements
        .trees
        .iter()
        .enumerate()
        .map(|(tree_index, tree)| {
            tree.layers_bottom_up
                .iter()
                .map(|layer| {
                    let ordinal = u32::try_from(tree_index)
                        .ok()
                        .and_then(|tree| tree.checked_shl(16))
                        .and_then(|tree| tree.checked_add(layer.log_size))
                        .ok_or(ArenaPlanError::SizeOverflow)?;
                    push_buffer_id(
                        logical,
                        None,
                        None,
                        BufferPurpose::FriMerkleLayer,
                        ordinal,
                        layer.words,
                        fri_live,
                    )
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let logical_fri = LogicalFriWorkspace {
        config: fri_config,
        requirements: fri_requirements,
        twiddles: inverse_twiddles,
        input_values: output_values,
        evaluation_ping,
        evaluation_pong,
        input_coordinate_ptrs,
        ping_coordinate_ptrs,
        pong_coordinate_ptrs,
        retained_tree_evaluations,
        retained_tree_coordinate_ptrs,
        folding_challenges,
        trees: fri_trees,
    };
    let final_fri_live = BufferLifetime::at(ProofEpoch::Fri);
    let final_fri_coefficients = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FriFinalCoefficients,
        0,
        final_fri_requirements.coefficient_words,
        final_fri_live,
    )?;
    let final_fri_degree_error = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FriFinalDegreeError,
        0,
        1,
        final_fri_live,
    )?;
    let interaction_pow_best_nonce = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::PowBestNonce,
        0,
        pow_requirements.best_nonce_words,
        BufferLifetime::at(ProofEpoch::BaseCommit),
    )?;
    let interaction_pow_completed_blocks = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::PowCompletedBlocks,
        0,
        pow_requirements.completed_blocks_words,
        BufferLifetime::at(ProofEpoch::BaseCommit),
    )?;
    let interaction_pow_prefix_digest = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::PowPrefixDigest,
        0,
        pow_requirements.prefix_digest_words,
        BufferLifetime::at(ProofEpoch::BaseCommit),
    )?;
    let query_pow_best_nonce = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::PowBestNonce,
        1,
        pow_requirements.best_nonce_words,
        BufferLifetime::at(ProofEpoch::Fri),
    )?;
    let query_pow_completed_blocks = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::PowCompletedBlocks,
        1,
        pow_requirements.completed_blocks_words,
        BufferLifetime::at(ProofEpoch::Fri),
    )?;
    let query_pow_prefix_digest = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::PowPrefixDigest,
        1,
        pow_requirements.prefix_digest_words,
        BufferLifetime::at(ProofEpoch::Fri),
    )?;
    let logical_final_fri_pow = LogicalFinalFriPowWorkspace {
        final_requirements: final_fri_requirements,
        final_coefficients: final_fri_coefficients,
        final_degree_error: final_fri_degree_error,
        pow_requirements,
        interaction_pow_bits: cairo_air::verifier::INTERACTION_POW_BITS,
        interaction_pow_best_nonce,
        interaction_pow_completed_blocks,
        interaction_pow_prefix_digest,
        query_pow_bits: protocol.identity.pow_bits,
        query_pow_best_nonce,
        query_pow_completed_blocks,
        query_pow_prefix_digest,
    };
    let composition_commitment = protocol
        .commitments
        .iter()
        .find(|commitment| commitment.id == CommitmentTreeId::Composition)
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "missing composition commitment geometry",
        ))?;
    let mut composition_coefficients = Vec::with_capacity(8);
    for (composition_column, &log_size) in composition_commitment
        .grouped_column_log_sizes
        .iter()
        .flatten()
        .enumerate()
    {
        let ordinal =
            u32::try_from(composition_column).map_err(|_| ArenaPlanError::SizeOverflow)?;
        composition_coefficients.push(push_buffer_id(
            logical,
            None,
            None,
            BufferPurpose::CompositionCoefficients,
            ordinal,
            checked_pow2(log_size)?,
            BufferLifetime::new(
                ProofEpoch::Composition,
                late_coefficient_ownership
                    .final_consumer(OpenedColumnSource::Composition { ordinal })
                    .map_err(ArenaPlanError::InvalidProtocolGeometry)?,
            )?,
        )?);
    }
    let composition_coefficients: [LogicalBufferId; 8] =
        composition_coefficients.try_into().map_err(|_| {
            ArenaPlanError::InvalidProtocolGeometry(
                "composition commitment must contain eight M31 coordinate polynomials",
            )
        })?;
    if composition_coefficients.iter().any(|id| {
        logical[id.0 as usize].len_words < composition_requirements.output_coefficient_words
    }) {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "composition coefficient output is smaller than the prepared composition result",
        ));
    }
    let mut composition_trace_trees = vec![Vec::new(), Vec::new(), Vec::new()];
    for column in &protocol.oods.columns {
        let tree = match column.source {
            OpenedColumnSource::Preprocessed { .. } => 0,
            OpenedColumnSource::Trace {
                purpose: BufferPurpose::BaseCoefficients,
                ..
            } => 1,
            OpenedColumnSource::Trace {
                purpose: BufferPurpose::InteractionCoefficients,
                ..
            } => 2,
            OpenedColumnSource::Composition { .. } => continue,
            OpenedColumnSource::Trace { .. } => {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "composition trace source is not a coefficient column",
                ));
            }
        };
        let coefficients = opened_source_logical_id(logical, column.source).ok_or(
            ArenaPlanError::InvalidProtocolGeometry(
                "composition coefficient source is absent from the arena",
            ),
        )?;
        composition_trace_trees[tree].push(LogicalCompositionTraceColumn {
            source: column.source,
            log_size: column.coefficient_log_size,
            coefficients,
        });
    }
    let composition_random_coefficient = logical_transcript
        .outputs
        .iter()
        .find_map(|(id, logical)| {
            (*id == protocol.composition_random_coefficient_output).then_some(*logical)
        })
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "missing composition random-coefficient transcript output",
        ))?;
    let composition_descriptor = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?;
    let composition_live = BufferLifetime::at(ProofEpoch::Composition);
    let descriptors = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::CompositionDescriptors,
        0,
        composition_requirements.descriptor_words,
        composition_descriptor,
    )?;
    let composition_lde_tile = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::CompositionLdeTile,
        0,
        composition_requirements.lde_tile_words,
        composition_live,
    )?;
    let composition_accumulators = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::CompositionAccumulators,
        0,
        composition_requirements.accumulator_words,
        composition_live,
    )?;
    let composition_random_powers = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::CompositionRandomCoefficientPowers,
        0,
        composition_requirements.random_power_words,
        composition_live,
    )?;
    let composition_ext_params = composition
        .components
        .iter()
        .enumerate()
        .map(|(component_index, component)| {
            let words = component
                .ext_param_sources
                .len()
                .checked_mul(SECURE_FIELD_WORDS)
                .ok_or(ArenaPlanError::SizeOverflow)?;
            let binding = (words != 0)
                .then(|| {
                    push_buffer_id(
                        logical,
                        None,
                        None,
                        BufferPurpose::CompositionExtParams,
                        u32::try_from(component_index).map_err(|_| ArenaPlanError::SizeOverflow)?,
                        words,
                        BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?,
                    )
                })
                .transpose()?;
            Ok(LogicalCompositionExtParams {
                component: component.component,
                instance: component.instance,
                sources: component.ext_param_sources.clone(),
                values: component.ext_param_values.clone(),
                binding,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let direct_bindings = protocol
        .direct_composition_retention
        .as_ref()
        .map(|plan| {
            plan.bindings
                .iter()
                .enumerate()
                .map(|(consumer, binding)| {
                    if binding.consumer != consumer {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "direct composition occurrence order drifted",
                        ));
                    }
                    let column = plan.columns.get(binding.column).ok_or(
                        ArenaPlanError::InvalidProtocolGeometry(
                            "direct composition column index drifted",
                        ),
                    )?;
                    Ok(LogicalDirectCompositionBinding {
                        consumer,
                        plan_column: binding.column,
                        source: column.source,
                        tree: column.tree,
                        proof_column: column.proof_column,
                        group: column.group,
                        column_in_group: column.column_in_group,
                        canonical_column: column.canonical_column,
                        evaluation_log_size: column.evaluation_log_size,
                        evaluation: binding
                            .direct
                            .then(|| {
                                direct_composition_logical_source(&logical_commitments, column)
                            })
                            .transpose()?,
                    })
                })
                .collect()
        })
        .transpose()?
        .unwrap_or_default();
    let logical_composition = LogicalCompositionWorkspace {
        plan: composition.clone(),
        requirements: composition_requirements,
        trace_trees: composition_trace_trees,
        random_coefficient: composition_random_coefficient,
        forward_twiddles,
        inverse_twiddles,
        ext_params: composition_ext_params,
        descriptors,
        lde_tile: composition_lde_tile,
        accumulators: composition_accumulators,
        random_coefficient_powers: composition_random_powers,
        composition_coefficients,
        direct_retention: protocol.direct_composition_retention.clone(),
        direct_bindings,
    };
    let opened_columns = protocol
        .oods
        .columns
        .iter()
        .zip(&oods_source_kinds)
        .map(|(geometry, &source_kind)| {
            let coefficients = opened_source_logical_id(logical, geometry.source).ok_or(
                ArenaPlanError::InvalidProtocolGeometry(
                    "OODS column coefficient identity is absent from the arena",
                ),
            )?;
            if logical[coefficients.0 as usize].len_words
                != checked_pow2(geometry.coefficient_log_size)?
            {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "OODS coefficient identity size disagrees with topology",
                ));
            }
            let source_binding = match source_kind {
                OodsSourceKind::Coefficients => coefficients,
                OodsSourceKind::Evaluations => retained_oods_source(
                    &logical_commitments,
                    geometry.source,
                    geometry.coefficient_log_size,
                )?,
            };
            let source_log_size = match source_kind {
                OodsSourceKind::Coefficients => geometry.coefficient_log_size,
                OodsSourceKind::Evaluations => geometry.evaluation_log_size,
            };
            let expected_words = checked_pow2(source_log_size)?;
            let actual_words = logical[source_binding.0 as usize].len_words;
            if actual_words != expected_words {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "OODS source size disagrees with its representation topology",
                ));
            }
            Ok(LogicalOodsColumn {
                geometry: geometry.clone(),
                coefficients,
                source_kind,
                source_binding,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let transcript_input = |id| {
        logical_transcript
            .inputs
            .iter()
            .find_map(|(candidate, logical)| (*candidate == id).then_some(*logical))
            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "missing required transcript input",
            ))
    };
    let transcript_output = |id| {
        logical_transcript
            .outputs
            .iter()
            .find_map(|(candidate, logical)| (*candidate == id).then_some(*logical))
            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "missing required transcript output",
            ))
    };
    let oods_sampled_values = transcript_input(protocol.oods.sampled_values_input)?;
    let oods_point_parameter = transcript_output(protocol.oods.point_parameter_output)?;
    let quotient_random_coefficient =
        transcript_output(protocol.oods.quotient_random_coefficient_output)?;
    let oods_live = BufferLifetime::at(ProofEpoch::Oods);
    let oods_sample_points_live = BufferLifetime::new(ProofEpoch::Oods, ProofEpoch::Quotient)?;
    let oods_descriptor = cached_workspace;
    let mut oods_slot = |purpose, ordinal, words, lifetime| {
        push_buffer_id(logical, None, None, purpose, ordinal, words, lifetime)
    };
    let oods_source_pointers = oods_slot(
        BufferPurpose::OodsSourcePointers,
        0,
        oods_requirements.source_pointer_words,
        oods_descriptor,
    )?;
    let oods_offset_points = oods_slot(
        BufferPurpose::OodsOffsetPoints,
        0,
        oods_requirements.offset_point_words,
        oods_descriptor,
    )?;
    let oods_fold_counts = oods_slot(
        BufferPurpose::OodsFoldCounts,
        0,
        oods_requirements.fold_count_words,
        oods_descriptor,
    )?;
    let oods_output_indices = oods_slot(
        BufferPurpose::OodsOutputIndices,
        0,
        oods_requirements.output_index_words,
        oods_descriptor,
    )?;
    let oods_folding_factors = oods_slot(
        BufferPurpose::OodsFoldingFactors,
        0,
        oods_requirements.factor_words,
        oods_live,
    )?;
    let oods_scratch_a = oods_slot(
        BufferPurpose::OodsScratchA,
        0,
        oods_requirements.scratch_a_words,
        oods_live,
    )?;
    let oods_scratch_b = oods_slot(
        BufferPurpose::OodsScratchB,
        0,
        oods_requirements.scratch_b_words,
        oods_live,
    )?;
    let oods_sample_points = oods_slot(
        BufferPurpose::OodsSamplePoints,
        0,
        oods_requirements.sample_point_words,
        oods_sample_points_live,
    )?;
    let oods_evaluation_points = oods_slot(
        BufferPurpose::OodsEvaluationPoints,
        0,
        oods_requirements.evaluation_point_words,
        oods_live,
    )?;
    let oods_barycentric_numerators = oods_slot(
        BufferPurpose::OodsBarycentricNumerators,
        0,
        oods_requirements.barycentric_numerator_words,
        oods_live,
    )?;
    let oods_barycentric_weights = oods_slot(
        BufferPurpose::OodsBarycentricWeights,
        0,
        oods_requirements.barycentric_weight_words,
        oods_live,
    )?;
    let oods_barycentric_scales = oods_slot(
        BufferPurpose::OodsBarycentricScales,
        0,
        oods_requirements.barycentric_scale_words,
        oods_live,
    )?;
    let oods_barycentric_partials = oods_slot(
        BufferPurpose::OodsBarycentricPartials,
        0,
        oods_requirements.barycentric_partial_words,
        oods_live,
    )?;
    let logical_oods = LogicalOodsWorkspace {
        config: oods_config,
        requirements: oods_requirements,
        columns: opened_columns.clone(),
        oods_point_parameter,
        source_pointers: oods_source_pointers,
        offset_points: oods_offset_points,
        fold_counts: oods_fold_counts,
        output_indices: oods_output_indices,
        folding_factors: oods_folding_factors,
        scratch_a: oods_scratch_a,
        scratch_b: oods_scratch_b,
        sample_points: oods_sample_points,
        sampled_values: oods_sampled_values,
        evaluation_points: oods_evaluation_points,
        barycentric_numerators: oods_barycentric_numerators,
        barycentric_weights: oods_barycentric_weights,
        barycentric_scales: oods_barycentric_scales,
        barycentric_partials: oods_barycentric_partials,
    };

    let numerator_live = BufferLifetime::at(ProofEpoch::Quotient);
    let numerator_descriptor = cached_workspace;
    let mut numerator_slot =
        |purpose, words, lifetime| push_buffer_id(logical, None, None, purpose, 0, words, lifetime);
    let numerator_runtime_terms = numerator_slot(
        BufferPurpose::QuotientNumeratorRuntimeTerms,
        quotient_numerator_requirements.runtime_term_words,
        numerator_descriptor,
    )?;
    let numerator_group_term_indices = numerator_slot(
        BufferPurpose::QuotientNumeratorGroupTermIndices,
        quotient_numerator_requirements.group_term_index_words,
        numerator_descriptor,
    )?;
    let numerator_group_offsets = numerator_slot(
        BufferPurpose::QuotientNumeratorGroupOffsets,
        quotient_numerator_requirements.group_offset_words,
        numerator_descriptor,
    )?;
    let numerator_line_coefficients = numerator_slot(
        BufferPurpose::QuotientNumeratorLineCoefficients,
        quotient_numerator_requirements.line_coefficient_words,
        numerator_live,
    )?;
    let numerator_term_points = numerator_slot(
        BufferPurpose::QuotientNumeratorTermPoints,
        quotient_numerator_requirements.term_point_words,
        numerator_live,
    )?;
    let numerator_batch_terms = numerator_slot(
        BufferPurpose::QuotientNumeratorBatchTerms,
        quotient_numerator_requirements.batch_term_words,
        numerator_descriptor,
    )?;
    let numerator_batch_group_offsets = numerator_slot(
        BufferPurpose::QuotientNumeratorBatchGroupOffsets,
        quotient_numerator_requirements.batch_group_offset_words,
        numerator_descriptor,
    )?;
    let numerator_batch_source_ptrs = numerator_slot(
        BufferPurpose::QuotientNumeratorBatchSourcePointers,
        quotient_numerator_requirements.batch_source_pointer_words,
        numerator_descriptor,
    )?;
    let numerator_output_ptrs = numerator_slot(
        BufferPurpose::QuotientNumeratorOutputPointers,
        quotient_numerator_requirements.output_pointer_words,
        numerator_descriptor,
    )?;
    let numerator_output_log_sizes = numerator_slot(
        BufferPurpose::QuotientNumeratorOutputLogSizes,
        quotient_numerator_requirements.output_log_size_words,
        numerator_descriptor,
    )?;
    let optional_numerator_slot = |logical: &mut Vec<LogicalBuffer>,
                                   purpose,
                                   words,
                                   lifetime|
     -> Result<Option<LogicalBufferId>, ArenaPlanError> {
        (words != 0)
            .then(|| push_buffer_id(logical, None, None, purpose, 0, words, lifetime))
            .transpose()
    };
    let numerator_coefficient_ptrs = optional_numerator_slot(
        logical,
        BufferPurpose::QuotientNumeratorCoefficientPointers,
        quotient_numerator_requirements.coefficient_pointer_words,
        numerator_descriptor,
    )?;
    let numerator_coefficient_sizes = optional_numerator_slot(
        logical,
        BufferPurpose::QuotientNumeratorCoefficientSizes,
        quotient_numerator_requirements.coefficient_size_words,
        numerator_descriptor,
    )?;
    let numerator_coefficient_output_ptrs = optional_numerator_slot(
        logical,
        BufferPurpose::QuotientNumeratorCoefficientOutputPointers,
        quotient_numerator_requirements.coefficient_output_pointer_words,
        numerator_descriptor,
    )?;
    let numerator_lde_tile = optional_numerator_slot(
        logical,
        BufferPurpose::QuotientNumeratorLdeTile,
        quotient_numerator_requirements.lde_tile_words,
        numerator_live,
    )?;
    let numerator_columns = opened_columns
        .into_iter()
        .zip(quotient_numerator_topologies)
        .map(|(column, topology)| {
            let numerator_source = match topology.source_kind {
                QuotientNumeratorSourceKind::Coefficients => column.coefficients,
                QuotientNumeratorSourceKind::Evaluation => retained_quotient_numerator_source(
                    &logical_commitments,
                    column.geometry.source,
                    topology.coefficient_log_size,
                )?,
            };
            Ok(LogicalQuotientNumeratorColumn {
                source: column.geometry.source,
                topology,
                coefficients: column.coefficients,
                numerator_source,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let logical_quotient_numerator = LogicalQuotientNumeratorWorkspace {
        schedule: protocol.identity.quotient_numerator_schedule,
        config: quotient_numerator_config,
        requirements: quotient_numerator_requirements,
        columns: numerator_columns,
        oods_sample_points,
        oods_sampled_values,
        random_coefficient: quotient_random_coefficient,
        sample_points_destination: sample_points,
        first_linear_terms_destination: first_linear_terms,
        destinations: partial_numerators.clone(),
        forward_twiddles,
        runtime_terms: numerator_runtime_terms,
        group_term_indices: numerator_group_term_indices,
        group_offsets: numerator_group_offsets,
        line_coefficients: numerator_line_coefficients,
        term_points: numerator_term_points,
        batch_terms: numerator_batch_terms,
        batch_group_offsets: numerator_batch_group_offsets,
        batch_source_ptrs: numerator_batch_source_ptrs,
        output_ptrs: numerator_output_ptrs,
        output_log_sizes: numerator_output_log_sizes,
        coefficient_ptrs: numerator_coefficient_ptrs,
        coefficient_sizes: numerator_coefficient_sizes,
        coefficient_output_ptrs: numerator_coefficient_output_ptrs,
        lde_tile: numerator_lde_tile,
    };
    let logical_decommit =
        append_decommit_buffers(logical, protocol, &logical_transcript, forward_twiddles)?;
    Ok((
        logical_preprocessed,
        logical_commitments,
        logical_composition,
        logical_oods,
        logical_quotient_numerator,
        logical_quotient,
        logical_fri,
        logical_final_fri_pow,
        logical_decommit,
        logical_transcript,
    ))
}

fn append_decommit_buffers(
    logical: &mut Vec<LogicalBuffer>,
    protocol: &ProtocolGeometry,
    transcript: &LogicalTranscriptWorkspace,
    lde_twiddles: LogicalBufferId,
) -> Result<LogicalDecommitWorkspace, ArenaPlanError> {
    let config = protocol.decommit_workspace_config()?;
    let requirements =
        decommit_workspace_requirements(config.clone()).map_err(ArenaPlanError::Decommit)?;
    let proof_shape = protocol.proof_assembly_shape()?;
    let query_id = CairoTranscriptOutput::QueryPositions.id().map_err(|_| {
        ArenaPlanError::InvalidProtocolGeometry("query-position transcript ABI is unavailable")
    })?;
    let raw_queries = transcript
        .outputs
        .iter()
        .find_map(|(id, logical)| (*id == query_id).then_some(*logical))
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "query-position transcript output is missing",
        ))?;
    let transcript_input = |semantic: CairoTranscriptInput| {
        let id = semantic.id().map_err(|_| {
            ArenaPlanError::InvalidProtocolGeometry("proof-bundle transcript ABI is unavailable")
        })?;
        transcript
            .inputs
            .iter()
            .find_map(|(candidate, logical)| (*candidate == id).then_some(*logical))
            .ok_or(ArenaPlanError::InvalidProtocolGeometry(
                "proof-bundle transcript input is missing",
            ))
    };
    let interaction_claim = transcript_input(CairoTranscriptInput::InteractionClaim)?;
    let sampled_values = transcript_input(CairoTranscriptInput::OodsSampledValues)?;
    let final_line_poly = transcript_input(CairoTranscriptInput::FriLastLayerPolynomial)?;
    let interaction_claim_words = logical[interaction_claim.0 as usize].len_words;
    let sampled_value_words = logical[sampled_values.0 as usize].len_words;
    let final_line_poly_words = logical[final_line_poly.0 as usize].len_words;
    let proof_bundle_layout = ResidentProofBundleLayout::new(
        interaction_claim_words,
        sampled_value_words,
        proof_shape.fri_trees.len(),
        final_line_poly_words,
        requirements.assembly_words,
    )
    .map_err(ArenaPlanError::ProofBundle)?;
    let scratch = BufferLifetime::at(ProofEpoch::Decommit);
    let descriptor = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?;
    // The proof bundle itself owns the decommit assembly tail. It becomes live
    // before the first decommit write and stays live through final assembly;
    // no second logical buffer or cross-buffer alias is admitted.
    let bundle_live = BufferLifetime::new(ProofEpoch::Decommit, ProofEpoch::Assemble)?;
    let mut allocate = |purpose, ordinal, words, lifetime| {
        push_buffer_id(logical, None, None, purpose, ordinal, words, lifetime)
    };
    let unique_queries = allocate(
        BufferPurpose::DecommitUniqueQueries,
        0,
        requirements.unique_query_words,
        scratch,
    )?;
    let mapped_queries = allocate(
        BufferPurpose::DecommitMappedQueries,
        0,
        requirements.mapped_query_words,
        scratch,
    )?;
    let walk_queries = allocate(
        BufferPurpose::DecommitWalkQueries,
        0,
        requirements.walk_query_words,
        scratch,
    )?;
    let walk_scratch = allocate(
        BufferPurpose::DecommitWalkScratch,
        0,
        requirements.walk_query_words,
        scratch,
    )?;
    let expanded_positions = allocate(
        BufferPurpose::DecommitExpandedPositions,
        0,
        requirements.expanded_position_words,
        scratch,
    )?;
    let sparse_indices = allocate(
        BufferPurpose::DecommitSparseIndices,
        0,
        requirements.sparse_index_words,
        scratch,
    )?;
    let sparse_hashes = allocate(
        BufferPurpose::DecommitSparseHashes,
        0,
        requirements.sparse_hash_words,
        scratch,
    )?;
    let counts = allocate(
        BufferPurpose::DecommitCounts,
        0,
        requirements.count_words,
        scratch,
    )?;
    let proof_bundle = allocate(
        BufferPurpose::ProofBytes,
        0,
        proof_bundle_layout.total_words,
        bundle_live,
    )?;
    let shared_lde_tile = requirements
        .trees
        .iter()
        .filter_map(|tree| match tree {
            DecommitTreeRequirements::Trace(tree) => tree
                .groups
                .iter()
                .filter_map(|group| group.lde_tile_words)
                .max(),
            DecommitTreeRequirements::Fri(_) => None,
        })
        .max()
        .map(|words| allocate(BufferPurpose::DecommitTraceLdeTile, 0, words, scratch))
        .transpose()?;
    let trees = requirements
        .trees
        .iter()
        .enumerate()
        .map(|(tree_index, tree)| {
            let tree_ordinal = u32::try_from(tree_index)
                .map_err(|_| ArenaPlanError::SizeOverflow)?
                .checked_shl(16)
                .ok_or(ArenaPlanError::SizeOverflow)?;
            match tree {
                DecommitTreeRequirements::Trace(tree) => {
                    let groups = tree
                        .groups
                        .iter()
                        .enumerate()
                        .map(|(group_index, group)| {
                            let ordinal = tree_ordinal
                                .checked_add(
                                    u32::try_from(group_index)
                                        .map_err(|_| ArenaPlanError::SizeOverflow)?,
                                )
                                .ok_or(ArenaPlanError::SizeOverflow)?;
                            Ok(LogicalTraceDecommitGroupSlots {
                                coefficient_ptrs: group
                                    .coefficient_pointer_words
                                    .map(|words| {
                                        allocate(
                                            BufferPurpose::DecommitTraceCoefficientPointers,
                                            ordinal,
                                            words,
                                            descriptor,
                                        )
                                    })
                                    .transpose()?,
                                coefficient_sizes: group
                                    .coefficient_size_words
                                    .map(|words| {
                                        allocate(
                                            BufferPurpose::DecommitTraceCoefficientSizes,
                                            ordinal,
                                            words,
                                            descriptor,
                                        )
                                    })
                                    .transpose()?,
                                lde_output_ptrs: group
                                    .coefficient_pointer_words
                                    .map(|words| {
                                        allocate(
                                            BufferPurpose::DecommitTraceLdeOutputPointers,
                                            ordinal,
                                            words,
                                            descriptor,
                                        )
                                    })
                                    .transpose()?,
                                lde_tile: group.lde_tile_words.map(|_| {
                                    shared_lde_tile.expect(
                                        "a recompute group guarantees a shared decommit LDE tile",
                                    )
                                }),
                            })
                        })
                        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
                    Ok(LogicalDecommitTreeSlots::Trace(LogicalTraceDecommitSlots {
                        evaluation_ptrs: allocate(
                            BufferPurpose::DecommitTraceEvaluationPointers,
                            tree_ordinal,
                            tree.evaluation_pointer_words,
                            descriptor,
                        )?,
                        evaluation_log_sizes: allocate(
                            BufferPurpose::DecommitTraceEvaluationLogs,
                            tree_ordinal,
                            tree.evaluation_log_words,
                            descriptor,
                        )?,
                        retained_layers_by_log: allocate(
                            BufferPurpose::DecommitTraceRetainedPointers,
                            tree_ordinal,
                            tree.retained_pointer_words,
                            descriptor,
                        )?,
                        sparse_level_offsets: allocate(
                            BufferPurpose::DecommitTraceSparseOffsets,
                            tree_ordinal,
                            tree.sparse_level_offsets.len().max(1),
                            descriptor,
                        )?,
                        groups,
                    }))
                }
                DecommitTreeRequirements::Fri(tree) => {
                    Ok(LogicalDecommitTreeSlots::Fri(LogicalFriDecommitSlots {
                        coordinate_ptrs: allocate(
                            BufferPurpose::DecommitFriCoordinatePointers,
                            tree_ordinal,
                            tree.coordinate_pointer_words,
                            descriptor,
                        )?,
                        retained_layers_by_log: allocate(
                            BufferPurpose::DecommitFriRetainedPointers,
                            tree_ordinal,
                            tree.retained_pointer_words,
                            descriptor,
                        )?,
                    }))
                }
            }
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    Ok(LogicalDecommitWorkspace {
        config,
        requirements,
        proof_shape,
        raw_queries,
        lde_twiddles,
        unique_queries,
        mapped_queries,
        walk_queries,
        walk_scratch,
        expanded_positions,
        sparse_indices,
        sparse_hashes,
        counts,
        proof_bundle_layout,
        proof_bundle,
        trees,
    })
}

/// Binding lookup shared by the resolve/validate phases. `color_logical_buffers`
/// returns bindings sorted by logical id (one binding per logical buffer), so a
/// binary search finds the unique entry in `O(log n)`; if a caller ever passes
/// an unsorted slice, fall back to the original linear scan so the result is
/// identical in every case.
fn find_binding(
    bindings: &[ArenaBinding],
    id: LogicalBufferId,
) -> Result<ArenaBinding, ArenaPlanError> {
    if let Ok(index) = bindings.binary_search_by_key(&id, |binding| binding.logical) {
        return Ok(bindings[index]);
    }
    bindings
        .iter()
        .find(|binding| binding.logical == id)
        .copied()
        .ok_or(ArenaPlanError::MissingBinding(id))
}

fn resolve_commit_batches(
    batches: Vec<LogicalCommitBatchSlots>,
    physical: &impl Fn(LogicalBufferId) -> Result<ArenaSlotId, ArenaPlanError>,
) -> Result<Vec<CommitBatchSlots>, ArenaPlanError> {
    batches
        .into_iter()
        .map(|batch| {
            Ok(CommitBatchSlots {
                coefficient_ptrs: physical(batch.coefficient_ptrs)?,
                coefficient_sizes: physical(batch.coefficient_sizes)?,
                output_ptrs: physical(batch.output_ptrs)?,
            })
        })
        .collect()
}

fn resolve_commitment_slots(
    logical: LogicalCommitWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedCommitment, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let leaf_state = binding(logical.leaf_state)?;
    let retained_layers = logical
        .retained_layers
        .iter()
        .copied()
        .map(binding)
        .collect::<Result<Vec<_>, _>>()?;
    let tail_outputs = logical
        .tail_outputs
        .iter()
        .copied()
        .map(binding)
        .collect::<Result<Vec<_>, _>>()?;
    let evaluation_output_groups = logical
        .retained_evaluations
        .iter()
        .map(|group| {
            group
                .as_ref()
                .map(|columns| columns.iter().copied().map(binding).collect())
                .transpose()
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let retained_evaluation_groups = evaluation_output_groups
        .iter()
        .zip(&logical.decommit_evaluation_groups)
        .map(|(group, &selected)| selected.then(|| group.clone().expect("union output exists")))
        .collect();
    let direct_composition_evaluation_groups = evaluation_output_groups
        .iter()
        .zip(&logical.direct_composition_evaluation_groups)
        .map(|(group, &selected)| selected.then(|| group.clone().expect("union output exists")))
        .collect();
    let numerator_evaluation_groups = evaluation_output_groups
        .iter()
        .zip(&logical.numerator_evaluation_groups)
        .map(|(group, &selected)| selected.then(|| group.clone().expect("union output exists")))
        .collect();
    let interpolation_batches = logical
        .interpolation_batches
        .iter()
        .map(|batch| {
            Ok(PlannedInterpolationBatch {
                log_size: batch.log_size,
                sources: batch.sources.clone(),
                input_pointers: physical(batch.input_pointers)?,
                output_pointers: physical(batch.output_pointers)?,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let mut retained_layers_bottom_up = Vec::new();
    if logical.config.unretained_bottom_layers == 0 {
        retained_layers_bottom_up.push(leaf_state);
    }
    retained_layers_bottom_up.extend(retained_layers.iter().copied());
    retained_layers_bottom_up.extend(tail_outputs.iter().copied());
    let root = retained_layers_bottom_up.last().copied().ok_or(
        ArenaPlanError::InvalidProtocolGeometry("commitment workspace has no retained root"),
    )?;
    if root.len_words != BLAKE2S_HASH_WORDS {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "commitment root binding is not one Blake2s hash",
        ));
    }
    let merkle_slots = MerkleFromLeavesSlots {
        leaves: leaf_state.physical,
        merkle_scratch: logical.merkle_scratch.map(physical).transpose()?,
        retained_layers: retained_layers
            .iter()
            .map(|binding| binding.physical)
            .collect(),
        tail_level_ptrs: logical.tail_level_ptrs.map(physical).transpose()?,
        tail_outputs: tail_outputs
            .iter()
            .map(|binding| binding.physical)
            .collect(),
    };
    let slots = match logical.leaf_workspace {
        LogicalCommitLeafWorkspace::FullLifting { lde_tile, groups } => {
            let groups = groups
                .into_iter()
                .map(|group| {
                    Ok(CommitGroupSlots {
                        column_ptrs: physical(group.column_ptrs)?,
                        column_log_sizes: physical(group.column_log_sizes)?,
                        batches: resolve_commit_batches(group.batches, &physical)?,
                    })
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()?;
            ModeAwareCommitWorkspaceSlots::FullLifting(CommitWorkspaceSlots {
                lde_tile: physical(lde_tile)?,
                leaf_state: merkle_slots.leaves,
                merkle_scratch: merkle_slots.merkle_scratch,
                retained_layers: merkle_slots.retained_layers.clone(),
                tail_level_ptrs: merkle_slots.tail_level_ptrs,
                tail_outputs: merkle_slots.tail_outputs.clone(),
                groups,
            })
        }
        LogicalCommitLeafWorkspace::DomainProgressive {
            lde_scratch,
            state_ping,
            state_pong,
            batches,
        } => ModeAwareCommitWorkspaceSlots::DomainProgressive(ProgressiveCommitWorkspaceSlots {
            leaves: ProgressiveLeafWorkspaceSlots {
                lde_scratch: lde_scratch.map(physical).transpose()?,
                state_ping: physical(state_ping)?,
                state_pong: state_pong.map(physical).transpose()?,
                leaf_hashes: merkle_slots.leaves,
                batches: resolve_commit_batches(batches, &physical)?
                    .into_iter()
                    .map(|batch| ProgressiveBatchSlots {
                        coefficient_ptrs: batch.coefficient_ptrs,
                        coefficient_sizes: batch.coefficient_sizes,
                        output_ptrs: batch.output_ptrs,
                    })
                    .collect(),
            },
            merkle: merkle_slots,
        }),
    };
    logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::ProgressiveCommit)?;
    Ok(PlannedCommitment {
        id: logical.id,
        config: logical.config,
        grouped_column_log_sizes: logical.grouped_column_log_sizes,
        grouped_column_sources: logical.grouped_column_sources,
        requirements: logical.requirements,
        twiddles: binding(logical.twiddles)?,
        slots,
        retained_evaluation_groups,
        evaluation_output_groups,
        direct_composition_evaluation_groups,
        numerator_evaluation_groups,
        root,
        retained_layers_bottom_up,
        interpolation_mode: logical.interpolation_mode,
        interpolation_batches,
    })
}

fn resolve_preprocessed_slots(
    logical: LogicalPreprocessedWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedPreprocessedWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let columns = logical
        .columns
        .into_iter()
        .map(|column| {
            Ok(PlannedPreprocessedColumn {
                identity: column.identity,
                ordinal: column.ordinal,
                log_size: column.log_size,
                evaluations: column.evaluations.map(binding).transpose()?,
                coefficients: binding(column.coefficients)?,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let interpolation_batches = logical
        .interpolation_batches
        .into_iter()
        .map(|batch| {
            Ok(PlannedPreprocessedInterpolationBatch {
                log_size: batch.log_size,
                column_ordinals: batch.column_ordinals,
                coefficient_pointers: binding(batch.coefficient_pointers)?,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    Ok(PlannedPreprocessedWorkspace {
        columns,
        interpolation_batches,
        inverse_twiddles: binding(logical.inverse_twiddles)?,
    })
}

fn resolve_composition_slots(
    logical: LogicalCompositionWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedCompositionWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let slots = CompositionWorkspaceSlots {
        descriptors: physical(logical.descriptors)?,
        lde_tile: physical(logical.lde_tile)?,
        accumulators: physical(logical.accumulators)?,
        random_coefficient_powers: physical(logical.random_coefficient_powers)?,
        composition_coefficients: logical
            .composition_coefficients
            .map(physical)
            .into_iter()
            .collect::<Result<Vec<_>, ArenaPlanError>>()?
            .try_into()
            .expect("exactly eight composition coefficient slots"),
    };
    let workspace_ids = logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::Composition)?
        .into_iter()
        .map(|requirement| requirement.id)
        .collect::<std::collections::BTreeSet<_>>();
    let trace_trees = logical
        .trace_trees
        .into_iter()
        .map(|tree| {
            tree.into_iter()
                .map(|column| {
                    Ok(PlannedCompositionTraceColumn {
                        source: column.source,
                        log_size: column.log_size,
                        coefficients: binding(column.coefficients)?,
                    })
                })
                .collect::<Result<Vec<_>, ArenaPlanError>>()
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let ext_params = logical
        .ext_params
        .into_iter()
        .map(|params| {
            Ok(PlannedCompositionExtParams {
                component: params.component,
                instance: params.instance,
                sources: params.sources,
                values: params.values,
                binding: params.binding.map(binding).transpose()?,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let direct_bindings = logical
        .direct_bindings
        .iter()
        .map(|direct| {
            Ok(PlannedDirectCompositionBinding {
                consumer: direct.consumer,
                plan_column: direct.plan_column,
                source: direct.source,
                tree: direct.tree,
                proof_column: direct.proof_column,
                group: direct.group,
                column_in_group: direct.column_in_group,
                canonical_column: direct.canonical_column,
                evaluation_log_size: direct.evaluation_log_size,
                evaluation: direct.evaluation.map(binding).transpose()?,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let random_coefficient = binding(logical.random_coefficient)?;
    let forward_twiddles = binding(logical.forward_twiddles)?;
    let inverse_twiddles = binding(logical.inverse_twiddles)?;
    let external = trace_trees
        .iter()
        .flatten()
        .map(|column| column.coefficients)
        .chain(ext_params.iter().filter_map(|params| params.binding))
        .chain(
            direct_bindings
                .iter()
                .filter_map(|direct| direct.evaluation),
        )
        .chain([random_coefficient, forward_twiddles, inverse_twiddles]);
    if external
        .into_iter()
        .any(|binding| workspace_ids.contains(&binding.physical))
    {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "composition input aliases a live writable workspace slot",
        ));
    }
    let trace_topology = CompositionTraceTopology {
        trees: trace_trees
            .iter()
            .map(|tree| {
                tree.iter()
                    .map(|column| CompositionCoefficientSource {
                        slot: column.coefficients.physical,
                        log_size: column.log_size,
                    })
                    .collect()
            })
            .collect(),
    };
    let rebound_requirements = composition_workspace_requirements_with_retention(
        &logical.plan,
        &trace_topology,
        logical.requirements.mode,
        logical.direct_retention.as_ref(),
    )
    .map_err(ArenaPlanError::Composition)?;
    let address_free = |mut requirements: CompositionWorkspaceRequirements| {
        for source in requirements
            .components
            .iter_mut()
            .flat_map(|component| &mut component.sources)
        {
            source.source.slot = ArenaSlotId(0);
        }
        requirements
    };
    if address_free(rebound_requirements) != address_free(logical.requirements.clone()) {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "composition workspace changed while binding physical trace sources",
        ));
    }
    Ok(PlannedCompositionWorkspace {
        plan: logical.plan,
        requirements: logical.requirements,
        trace_trees,
        random_coefficient,
        forward_twiddles,
        inverse_twiddles,
        ext_params,
        slots,
        direct_retention: logical.direct_retention,
        direct_bindings,
    })
}

fn resolve_oods_slots(
    logical: LogicalOodsWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedOodsWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let slots = OodsWorkspaceSlots {
        source_pointers: physical(logical.source_pointers)?,
        offset_points: physical(logical.offset_points)?,
        fold_counts: physical(logical.fold_counts)?,
        output_indices: physical(logical.output_indices)?,
        folding_factors: physical(logical.folding_factors)?,
        scratch_a: physical(logical.scratch_a)?,
        scratch_b: physical(logical.scratch_b)?,
        sample_points: physical(logical.sample_points)?,
        sampled_values: physical(logical.sampled_values)?,
        evaluation_points: physical(logical.evaluation_points)?,
        barycentric_numerators: physical(logical.barycentric_numerators)?,
        barycentric_weights: physical(logical.barycentric_weights)?,
        barycentric_scales: physical(logical.barycentric_scales)?,
        barycentric_partials: physical(logical.barycentric_partials)?,
    };
    let workspace_ids = logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::Oods)?
        .into_iter()
        .map(|requirement| requirement.id)
        .collect::<std::collections::BTreeSet<_>>();
    let columns = logical
        .columns
        .into_iter()
        .map(|column| {
            Ok(PlannedOodsColumn {
                source: column.geometry.source,
                coefficient_log_size: column.geometry.coefficient_log_size,
                evaluation_log_size: column.geometry.evaluation_log_size,
                shape_points: column.geometry.shape_points,
                offset_points: column.geometry.offset_points,
                source_kind: column.source_kind,
                source_binding: binding(column.source_binding)?,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    for column in &columns {
        validate_oods_source_binding(column)?;
    }
    let rebound_topologies = columns
        .iter()
        .map(PlannedOodsColumn::topology)
        .collect::<Vec<_>>();
    let rebound_requirements = oods_workspace_requirements(logical.config, &rebound_topologies)
        .map_err(ArenaPlanError::Oods)?;
    if rebound_requirements != logical.requirements {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "OODS workspace changed while binding physical polynomial sources",
        ));
    }
    let oods_point_parameter = binding(logical.oods_point_parameter)?;
    if columns
        .iter()
        .map(|column| column.source_binding)
        .chain([oods_point_parameter])
        .any(|external| workspace_ids.contains(&external.physical))
    {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "OODS source aliases a live workspace slot",
        ));
    }
    Ok(PlannedOodsWorkspace {
        config: logical.config,
        requirements: logical.requirements,
        columns,
        oods_point_parameter,
        sample_points: binding(logical.sample_points)?,
        sampled_values: binding(logical.sampled_values)?,
        slots,
    })
}

fn validate_oods_source_binding(column: &PlannedOodsColumn) -> Result<(), ArenaPlanError> {
    let source_log_size = match column.source_kind {
        OodsSourceKind::Coefficients => column.coefficient_log_size,
        OodsSourceKind::Evaluations => column.evaluation_log_size,
    };
    if column.source_binding.len_words != checked_pow2(source_log_size)? {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "OODS source binding has the wrong representation extent",
        ));
    }
    Ok(())
}

fn resolve_quotient_numerator_slots(
    logical: LogicalQuotientNumeratorWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedQuotientNumeratorWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let slots = QuotientNumeratorWorkspaceSlots {
        runtime_terms: physical(logical.runtime_terms)?,
        group_term_indices: physical(logical.group_term_indices)?,
        group_offsets: physical(logical.group_offsets)?,
        line_coefficients: physical(logical.line_coefficients)?,
        term_points: physical(logical.term_points)?,
        batch_terms: physical(logical.batch_terms)?,
        batch_group_offsets: physical(logical.batch_group_offsets)?,
        batch_source_ptrs: physical(logical.batch_source_ptrs)?,
        output_ptrs: physical(logical.output_ptrs)?,
        output_log_sizes: physical(logical.output_log_sizes)?,
        coefficient_ptrs: logical.coefficient_ptrs.map(physical).transpose()?,
        coefficient_sizes: logical.coefficient_sizes.map(physical).transpose()?,
        coefficient_output_ptrs: logical.coefficient_output_ptrs.map(physical).transpose()?,
        lde_tile: logical.lde_tile.map(physical).transpose()?,
    };
    let workspace_ids = logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::QuotientNumerator)?
        .into_iter()
        .map(|requirement| requirement.id)
        .collect::<std::collections::BTreeSet<_>>();
    let columns = logical
        .columns
        .into_iter()
        .map(|column| {
            let coefficients = binding(column.coefficients)?;
            let numerator_source = binding(column.numerator_source)?;
            validate_quotient_numerator_source_binding(
                logical.config,
                &column.topology,
                coefficients,
                numerator_source,
            )?;
            Ok(PlannedQuotientNumeratorColumn {
                source: column.source,
                topology: column.topology,
                coefficients,
                numerator_source,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let destinations = logical
        .destinations
        .into_iter()
        .map(|destination| {
            Ok(PlannedQuotientNumeratorSource {
                log_size: destination.log_size,
                coordinates: [
                    binding(destination.coordinates[0])?,
                    binding(destination.coordinates[1])?,
                    binding(destination.coordinates[2])?,
                    binding(destination.coordinates[3])?,
                ],
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let oods_sample_points = binding(logical.oods_sample_points)?;
    let oods_sampled_values = binding(logical.oods_sampled_values)?;
    let random_coefficient = binding(logical.random_coefficient)?;
    let sample_points_destination = binding(logical.sample_points_destination)?;
    let first_linear_terms_destination = binding(logical.first_linear_terms_destination)?;
    let forward_twiddles = binding(logical.forward_twiddles)?;
    if quotient_numerator_sources_alias_workspace(&columns, &workspace_ids) {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "quotient numerator external binding aliases a live workspace slot",
        ));
    }
    let external = [
        oods_sample_points,
        oods_sampled_values,
        random_coefficient,
        sample_points_destination,
        first_linear_terms_destination,
        forward_twiddles,
    ]
    .into_iter()
    .chain(
        destinations
            .iter()
            .flat_map(|destination| destination.coordinates),
    );
    if external
        .into_iter()
        .any(|binding| workspace_ids.contains(&binding.physical))
    {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "quotient numerator external binding aliases a live workspace slot",
        ));
    }
    Ok(PlannedQuotientNumeratorWorkspace {
        schedule: logical.schedule,
        config: logical.config,
        requirements: logical.requirements,
        columns,
        oods_sample_points,
        oods_sampled_values,
        random_coefficient,
        sample_points_destination,
        first_linear_terms_destination,
        destinations,
        forward_twiddles,
        slots,
    })
}

fn quotient_numerator_sources_alias_workspace(
    columns: &[PlannedQuotientNumeratorColumn],
    workspace_ids: &BTreeSet<ArenaSlotId>,
) -> bool {
    columns
        .iter()
        .any(|column| workspace_ids.contains(&column.numerator_source.physical))
}

fn validate_quotient_numerator_source_binding(
    config: QuotientNumeratorWorkspaceConfig,
    topology: &QuotientNumeratorColumnTopology,
    coefficients: ArenaBinding,
    numerator_source: ArenaBinding,
) -> Result<(), ArenaPlanError> {
    let source_log_size = match topology.source_kind {
        QuotientNumeratorSourceKind::Coefficients => topology.coefficient_log_size,
        QuotientNumeratorSourceKind::Evaluation => topology
            .coefficient_log_size
            .checked_add(config.log_blowup_factor)
            .ok_or(ArenaPlanError::SizeOverflow)?,
    };
    if numerator_source.len_words != checked_pow2(source_log_size)?
        || (topology.source_kind == QuotientNumeratorSourceKind::Coefficients
            && numerator_source != coefficients)
    {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "quotient numerator source binding has the wrong kind or extent",
        ));
    }
    Ok(())
}

fn resolve_quotient_slots(
    logical: LogicalQuotientWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedQuotientWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let slots = QuotientWorkspaceSlots {
        sample_points: physical(logical.sample_points)?,
        first_linear_terms: physical(logical.first_linear_terms)?,
        partial_log_sizes: physical(logical.partial_log_sizes)?,
        partial_coordinate_ptrs: physical(logical.partial_coordinate_ptrs)?,
        subdomain_coordinate_ptrs: physical(logical.subdomain_coordinate_ptrs)?,
        output_coordinate_ptrs: physical(logical.output_coordinate_ptrs)?,
        coefficient_sizes: physical(logical.coefficient_sizes)?,
        subdomain_values: physical(logical.subdomain_values)?,
        output_values: physical(logical.output_values)?,
    };
    let slot_requirements = logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::Quotient)?;
    let workspace_ids = slot_requirements
        .iter()
        .map(|requirement| requirement.id)
        .collect::<std::collections::BTreeSet<_>>();
    let forward_twiddles = binding(logical.forward_twiddles)?;
    let inverse_subdomain_twiddles = binding(logical.inverse_subdomain_twiddles)?;
    if forward_twiddles.physical == inverse_subdomain_twiddles.physical {
        return Err(ArenaPlanError::Quotient(
            PreparedQuotientError::AliasedTwiddles(forward_twiddles.physical),
        ));
    }
    for twiddles in [forward_twiddles, inverse_subdomain_twiddles] {
        if workspace_ids.contains(&twiddles.physical) {
            return Err(ArenaPlanError::Quotient(
                PreparedQuotientError::SourceAliasesWorkspace(twiddles.physical),
            ));
        }
    }
    let partial_numerators = logical
        .partial_numerators
        .into_iter()
        .map(|source| {
            Ok(PlannedQuotientNumeratorSource {
                log_size: source.log_size,
                coordinates: [
                    binding(source.coordinates[0])?,
                    binding(source.coordinates[1])?,
                    binding(source.coordinates[2])?,
                    binding(source.coordinates[3])?,
                ],
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let mut source_ids = std::collections::BTreeSet::from([
        forward_twiddles.physical,
        inverse_subdomain_twiddles.physical,
    ]);
    for source in &partial_numerators {
        for coordinate in source.coordinates {
            if workspace_ids.contains(&coordinate.physical) {
                return Err(ArenaPlanError::Quotient(
                    PreparedQuotientError::SourceAliasesWorkspace(coordinate.physical),
                ));
            }
            if !source_ids.insert(coordinate.physical) {
                return Err(ArenaPlanError::Quotient(
                    PreparedQuotientError::AliasedSourceSlot(coordinate.physical),
                ));
            }
        }
    }
    let output_values = binding(logical.output_values)?;
    debug_assert_eq!(output_values.physical, slots.output_values);
    let sample_points = binding(logical.sample_points)?;
    let first_linear_terms = binding(logical.first_linear_terms)?;
    debug_assert_eq!(sample_points.physical, slots.sample_points);
    debug_assert_eq!(first_linear_terms.physical, slots.first_linear_terms);
    Ok(PlannedQuotientWorkspace {
        config: logical.config,
        requirements: logical.requirements,
        forward_twiddles,
        inverse_subdomain_twiddles,
        partial_numerators,
        slots,
        sample_points,
        first_linear_terms,
        output_values,
    })
}

fn resolve_fri_slots(
    logical: LogicalFriWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedFriWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let slots = FriWorkspaceSlots {
        evaluation_ping: physical(logical.evaluation_ping)?,
        evaluation_pong: physical(logical.evaluation_pong)?,
        input_coordinate_ptrs: physical(logical.input_coordinate_ptrs)?,
        ping_coordinate_ptrs: physical(logical.ping_coordinate_ptrs)?,
        pong_coordinate_ptrs: physical(logical.pong_coordinate_ptrs)?,
        retained_tree_evaluations: logical
            .retained_tree_evaluations
            .into_iter()
            .map(physical)
            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
        retained_tree_coordinate_ptrs: logical
            .retained_tree_coordinate_ptrs
            .into_iter()
            .map(physical)
            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
        folding_challenges: logical
            .folding_challenges
            .into_iter()
            .map(physical)
            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
        trees: logical
            .trees
            .into_iter()
            .map(|layers| {
                Ok(FriMerkleTreeSlots {
                    layers_bottom_up: layers
                        .into_iter()
                        .map(physical)
                        .collect::<Result<Vec<_>, ArenaPlanError>>()?,
                })
            })
            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
    };
    let slot_requirements = logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::Fri)?;
    let workspace_ids = slot_requirements
        .iter()
        .map(|requirement| requirement.id)
        .collect::<std::collections::BTreeSet<_>>();
    let twiddles = binding(logical.twiddles)?;
    let input_values = binding(logical.input_values)?;
    for source in [twiddles, input_values] {
        if workspace_ids.contains(&source.physical) {
            return Err(ArenaPlanError::Fri(
                PreparedFriError::SourceAliasesWorkspace(source.physical),
            ));
        }
    }
    if twiddles.physical == input_values.physical {
        return Err(ArenaPlanError::Fri(PreparedFriError::AliasedSourceSlot(
            input_values.physical,
        )));
    }
    Ok(PlannedFriWorkspace {
        config: logical.config,
        requirements: logical.requirements,
        twiddles,
        input_values,
        slots,
    })
}

fn resolve_final_fri_pow_slots(
    logical: LogicalFinalFriPowWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedFinalFriPowWorkspace, ArenaPlanError> {
    let physical = |id: LogicalBufferId| find_binding(bindings, id).map(|binding| binding.physical);
    let final_slots = FriFinalWorkspaceSlots {
        coefficients: physical(logical.final_coefficients)?,
        degree_error: physical(logical.final_degree_error)?,
    };
    logical
        .final_requirements
        .arena_slot_requirements(final_slots)
        .map_err(ArenaPlanError::FriFinal)?;
    let interaction_pow_slots = Blake2sPowWorkspaceSlots {
        best_nonce: physical(logical.interaction_pow_best_nonce)?,
        completed_blocks: physical(logical.interaction_pow_completed_blocks)?,
        prefix_digest: physical(logical.interaction_pow_prefix_digest)?,
    };
    logical
        .pow_requirements
        .arena_slot_requirements(interaction_pow_slots)
        .map_err(ArenaPlanError::Pow)?;
    let query_pow_slots = Blake2sPowWorkspaceSlots {
        best_nonce: physical(logical.query_pow_best_nonce)?,
        completed_blocks: physical(logical.query_pow_completed_blocks)?,
        prefix_digest: physical(logical.query_pow_prefix_digest)?,
    };
    logical
        .pow_requirements
        .arena_slot_requirements(query_pow_slots)
        .map_err(ArenaPlanError::Pow)?;
    Ok(PlannedFinalFriPowWorkspace {
        final_requirements: logical.final_requirements,
        final_slots,
        pow_requirements: logical.pow_requirements,
        interaction_pow_bits: logical.interaction_pow_bits,
        interaction_pow_slots,
        query_pow_bits: logical.query_pow_bits,
        query_pow_slots,
    })
}

fn resolve_decommit_slots(
    logical: LogicalDecommitWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedDecommitWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let proof_bundle = binding(logical.proof_bundle)?;
    let decommitment = logical
        .proof_bundle_layout
        .direct_decommitment_destination();
    if proof_bundle.len_words != logical.proof_bundle_layout.total_words
        || decommitment.len_words != logical.requirements.assembly_words
        || decommitment
            .offset_words
            .checked_add(decommitment.len_words)
            != Some(logical.proof_bundle_layout.total_words)
    {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "proof bundle direct decommit range is not the exact canonical tail",
        ));
    }
    let trees = logical
        .trees
        .into_iter()
        .map(|tree| match tree {
            LogicalDecommitTreeSlots::Trace(tree) => {
                Ok(DecommitTreeSlots::Trace(TraceDecommitSlots {
                    evaluation_ptrs: physical(tree.evaluation_ptrs)?,
                    evaluation_log_sizes: physical(tree.evaluation_log_sizes)?,
                    retained_layers_by_log: physical(tree.retained_layers_by_log)?,
                    sparse_level_offsets: physical(tree.sparse_level_offsets)?,
                    groups: tree
                        .groups
                        .into_iter()
                        .map(|group| {
                            Ok(TraceSourceGroupSlots {
                                coefficient_ptrs: group
                                    .coefficient_ptrs
                                    .map(physical)
                                    .transpose()?,
                                coefficient_sizes: group
                                    .coefficient_sizes
                                    .map(physical)
                                    .transpose()?,
                                lde_output_ptrs: group.lde_output_ptrs.map(physical).transpose()?,
                                lde_tile: group.lde_tile.map(physical).transpose()?,
                            })
                        })
                        .collect::<Result<Vec<_>, ArenaPlanError>>()?,
                }))
            }
            LogicalDecommitTreeSlots::Fri(tree) => Ok(DecommitTreeSlots::Fri(FriDecommitSlots {
                coordinate_ptrs: physical(tree.coordinate_ptrs)?,
                retained_layers_by_log: physical(tree.retained_layers_by_log)?,
            })),
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let slots = DecommitWorkspaceSlots {
        unique_queries: physical(logical.unique_queries)?,
        mapped_queries: physical(logical.mapped_queries)?,
        walk_queries: physical(logical.walk_queries)?,
        walk_scratch: physical(logical.walk_scratch)?,
        expanded_positions: physical(logical.expanded_positions)?,
        sparse_indices: physical(logical.sparse_indices)?,
        sparse_hashes: physical(logical.sparse_hashes)?,
        counts: physical(logical.counts)?,
        // The proof bundle owns the exact final decommit range. Giving the
        // prepared tail this physical identity makes aliasing explicit while
        // the typed subrange below preserves its canonical byte offset.
        assembly: proof_bundle.physical,
        trees,
    };
    let slot_requirements = logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::Decommit)?;
    let workspace_ids = slot_requirements
        .iter()
        .map(|requirement| requirement.id)
        .collect::<std::collections::BTreeSet<_>>();
    let raw_queries = binding(logical.raw_queries)?;
    let lde_twiddles = binding(logical.lde_twiddles)?;
    if raw_queries.len_words != logical.config.n_queries as usize {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "transcript query output width disagrees with decommitment",
        ));
    }
    for source in [raw_queries, lde_twiddles] {
        if workspace_ids.contains(&source.physical) {
            return Err(ArenaPlanError::Decommit(
                PreparedDecommitError::SourceAliasesWorkspace(source.physical),
            ));
        }
    }
    Ok(PlannedDecommitWorkspace {
        config: logical.config,
        requirements: logical.requirements,
        proof_shape: logical.proof_shape,
        raw_queries,
        lde_twiddles,
        slots,
        proof_bundle_layout: logical.proof_bundle_layout,
        proof_bundle,
    })
}

fn validate_decommit_group_bindings(
    config: &DecommitWorkspaceConfig,
    commitments: &[PlannedCommitment],
) -> Result<(), ArenaPlanError> {
    if commitments.len() != 4 || config.trees.len() < commitments.len() {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "decommit group bindings require the canonical trace prefix",
        ));
    }
    for (tree, commitment) in config.trees.iter().zip(commitments) {
        let DecommitTreeGeometry::Trace(trace) = tree else {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "decommit group bindings disagree with trace topology",
            ));
        };
        if trace.groups.len() != commitment.retained_evaluation_groups.len() {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "decommit group bindings disagree with opening groups",
            ));
        }
        for (group, retained) in trace
            .groups
            .iter()
            .zip(&commitment.retained_evaluation_groups)
        {
            let matches = match (group.mode, retained) {
                (DecommitSourceMode::RecomputeQueriedLde, None) => true,
                (DecommitSourceMode::ResidentEvaluations, Some(bindings)) => {
                    bindings.len() == group.columns.len()
                }
                _ => false,
            };
            if !matches {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "decommit source mode disagrees with retained evaluation bindings",
                ));
            }
        }
    }
    Ok(())
}

fn resolve_transcript_slots(
    logical: LogicalTranscriptWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedTranscriptWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let slots = Blake2sTranscriptWorkspaceSlots {
        state: physical(logical.state)?,
        boundary_snapshots: physical(logical.boundary_snapshots)?,
        input_snapshots: physical(logical.input_snapshots)?,
        output_snapshots: physical(logical.output_snapshots)?,
    };
    logical
        .requirements
        .arena_slot_requirements(slots)
        .map_err(ArenaPlanError::Transcript)?;
    Ok(PlannedTranscriptWorkspace {
        schedule_key: logical.schedule_key,
        requirements: logical.requirements,
        slots,
        inputs: logical
            .inputs
            .into_iter()
            .map(|(id, logical)| Ok((id, binding(logical)?)))
            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
        outputs: logical
            .outputs
            .into_iter()
            .map(|(id, logical)| Ok((id, binding(logical)?)))
            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
    })
}

fn resolve_execution_table_slots(
    logical: LogicalExecutionTablesWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedExecutionTablesWorkspace, ArenaPlanError> {
    let physical = |id: LogicalBufferId| find_binding(bindings, id).map(|binding| binding.physical);
    let slots = ExecutionTablesWorkspaceSlots {
        raw_addr_to_id: physical(logical.raw_addr_to_id)?,
        raw_f252_words: physical(logical.raw_f252_words)?,
        raw_small_words: physical(logical.raw_small_words)?,
        big_limbs: logical
            .big_limbs
            .into_iter()
            .map(physical)
            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
        small_limbs: logical
            .small_limbs
            .into_iter()
            .map(physical)
            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
        table_pointers: physical(logical.table_pointers)?,
        table_strides: physical(logical.table_strides)?,
    };
    logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::ExecutionTables)?;
    Ok(PlannedExecutionTablesWorkspace {
        requirements: logical.requirements,
        slots,
    })
}

fn resolve_ec_op_slots(
    logical: LogicalEcOpWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedEcOpWorkspace, ArenaPlanError> {
    let physical = |id: LogicalBufferId| find_binding(bindings, id).map(|binding| binding.physical);
    let slots = EcOpWorkspaceSlots {
        trace_columns: logical
            .trace_columns
            .into_iter()
            .map(physical)
            .collect::<Result<Vec<_>, _>>()?,
        lookup_words: physical(logical.lookup_words)?,
        partial_input_columns: logical
            .partial_input_columns
            .into_iter()
            .map(physical)
            .collect::<Result<Vec<_>, _>>()?,
        segment_start: physical(logical.segment_start)?,
        address_counts: physical(logical.address_counts)?,
        big_counts: physical(logical.big_counts)?,
        small_counts: physical(logical.small_counts)?,
        range_check_8_counts: physical(logical.range_check_8_counts)?,
    };
    logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(|_| ArenaPlanError::InvalidProtocolGeometry("invalid resident EC-op slots"))?;
    Ok(PlannedEcOpWorkspace {
        requirements: logical.requirements,
        slots,
    })
}

fn resolve_witness_slots(
    logical: LogicalWitnessWorkspace,
    execution_tables: Option<&PlannedExecutionTablesWorkspace>,
    bindings: &[ArenaBinding],
) -> Result<PlannedWitnessWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let components = logical
        .components
        .into_iter()
        .map(|component| {
            let (execution_table_pointers, execution_table_strides) = match execution_tables {
                Some(execution_tables) => (
                    execution_tables.slots.table_pointers,
                    execution_tables.slots.table_strides,
                ),
                None => (
                    physical(component.execution_table_pointers.ok_or(
                        ArenaPlanError::InvalidProtocolGeometry(
                            "legacy witness execution-table pointers are missing",
                        ),
                    )?)?,
                    physical(component.execution_table_strides.ok_or(
                        ArenaPlanError::InvalidProtocolGeometry(
                            "legacy witness execution-table strides are missing",
                        ),
                    )?)?,
                ),
            };
            let input_column_slots = component
                .input_columns
                .iter()
                .copied()
                .map(|id| physical(id))
                .collect::<Result<Vec<_>, ArenaPlanError>>()?;
            let slots = WitnessWorkspaceSlots {
                input_columns: input_column_slots.clone(),
                input_pointers: physical(component.input_pointers)?,
                execution_table_pointers,
                execution_table_strides,
                output_columns: component
                    .output_columns
                    .into_iter()
                    .map(|id| physical(id))
                    .collect::<Result<Vec<_>, ArenaPlanError>>()?,
                output_pointers: physical(component.output_pointers)?,
                multiplicity_columns: component
                    .multiplicity_columns
                    .into_iter()
                    .map(|id| physical(id))
                    .collect::<Result<Vec<_>, ArenaPlanError>>()?,
                multiplicity_pointers: physical(component.multiplicity_pointers)?,
                multiplicity_dummy: component
                    .multiplicity_dummy
                    .map(|id| physical(id))
                    .transpose()?,
                lookup_words: physical(component.lookup_words)?,
                sub_words: physical(component.sub_words)?,
            };
            if component.blake_g_fused {
                if execution_tables.is_none() {
                    return Err(ArenaPlanError::InvalidProtocolGeometry(
                        "fused blake_g requires prepared execution tables",
                    ));
                }
                component
                    .requirements
                    .arena_slot_requirements_for_blake_g_fusion_with_prepared_execution_tables(
                        &slots,
                    )
                    .map_err(ArenaPlanError::Witness)?;
            } else if execution_tables.is_some() {
                component
                    .requirements
                    .arena_slot_requirements_with_prepared_execution_tables(&slots)
                    .map_err(ArenaPlanError::Witness)?;
            } else {
                component
                    .requirements
                    .arena_slot_requirements(&slots)
                    .map_err(ArenaPlanError::Witness)?;
            }
            let input_gather = component
                .input_gather
                .map(|gather| {
                    let slots = WitnessInputGatherSlots {
                        source_pointers: physical(gather.source_pointers)?,
                        descriptors: physical(gather.descriptors)?,
                        consumer_input_columns: input_column_slots.clone(),
                        output_pointers: physical(gather.output_pointers)?,
                    };
                    gather
                        .requirements
                        .arena_slot_requirements(&slots)
                        .map_err(ArenaPlanError::WitnessInputGather)?;
                    Ok::<_, ArenaPlanError>(PlannedWitnessInputGather {
                        requirements: gather.requirements,
                        producers: gather.producers,
                        sources: gather
                            .sources
                            .into_iter()
                            .map(|id| binding(id))
                            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
                        slots,
                    })
                })
                .transpose()?;
            let input_seed = component
                .input_seed
                .map(|seed| {
                    let slots = WitnessInputSeedSlots {
                        scalar_values: physical(seed.scalar_values)?,
                        consumer_input_columns: input_column_slots.clone(),
                        output_pointers: physical(seed.output_pointers)?,
                    };
                    seed.requirements
                        .arena_slot_requirements(&slots)
                        .map_err(ArenaPlanError::WitnessInputGather)?;
                    Ok::<_, ArenaPlanError>(PlannedWitnessInputSeed {
                        requirements: seed.requirements,
                        slots,
                    })
                })
                .transpose()?;
            let input_compact = component
                .input_compact
                .map(|compact| {
                    let slots = WitnessInputCompactSlots {
                        source_pointers: physical(compact.source_pointers)?,
                        descriptors: physical(compact.descriptors)?,
                        consumer_input_columns: input_column_slots.clone(),
                        output_pointers: physical(compact.output_pointers)?,
                        tuple_scratch: physical(compact.tuple_scratch)?,
                        sort_keys_a: physical(compact.sort_keys_a)?,
                        sort_keys_b: physical(compact.sort_keys_b)?,
                        sort_indices_a: physical(compact.sort_indices_a)?,
                        sort_indices_b: physical(compact.sort_indices_b)?,
                        run_heads: physical(compact.run_heads)?,
                        run_positions: physical(compact.run_positions)?,
                        n_unique: physical(compact.n_unique)?,
                        sort_temp: physical(compact.sort_temp)?,
                        scan_temp: physical(compact.scan_temp)?,
                    };
                    compact
                        .requirements
                        .arena_slot_requirements(&slots)
                        .map_err(ArenaPlanError::WitnessInputGather)?;
                    Ok::<_, ArenaPlanError>(PlannedWitnessInputCompact {
                        requirements: compact.requirements,
                        sources: compact
                            .sources
                            .into_iter()
                            .map(|id| binding(id))
                            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
                        slots,
                    })
                })
                .transpose()?;
            Ok(PlannedWitnessComponent {
                component: component.component,
                part: component.part,
                n_real_rows: component.n_real_rows,
                blake_g_fused: component.blake_g_fused,
                native_input_producer: component.native_input_producer,
                program: component.program,
                requirements: component.requirements,
                slots,
                input_gather,
                input_seed,
                input_compact,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    Ok(PlannedWitnessWorkspace { components })
}

fn resolve_graph_a_multiplicity_slots(
    logical: LogicalGraphAMultiplicityWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedGraphAMultiplicityWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let multiplicities = logical
        .multiplicities
        .into_iter()
        .map(|(component, id)| Ok((component, binding(id)?)))
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let clear_slots = WitnessFeedClearWorkspaceSlots {
        destination_pointers: physical(logical.clear_pointers)?,
        destination_lengths: physical(logical.clear_lengths)?,
    };
    logical
        .clear_requirements
        .arena_slot_requirements(clear_slots)
        .map_err(ArenaPlanError::WitnessFeed)?;
    let resolve_generic_feed = |feed: LogicalGenericRecordedMultiplicityFeed| {
        let slots = WitnessFeedWorkspaceSlots {
            descriptors: physical(feed.descriptors)?,
            lut_tables: feed
                .lut_tables
                .into_iter()
                .map(physical)
                .collect::<Result<Vec<_>, _>>()?,
            lut_pointers: physical(feed.lut_pointers)?,
            multiplicity_destinations: feed
                .multiplicity_destinations
                .into_iter()
                .map(physical)
                .collect::<Result<Vec<_>, _>>()?,
            multiplicity_pointers: physical(feed.multiplicity_pointers)?,
        };
        feed.plan
            .requirements
            .arena_slot_requirements(&slots)
            .map_err(ArenaPlanError::WitnessFeed)?;
        Ok::<_, ArenaPlanError>(PlannedGenericRecordedMultiplicityFeedGraph {
            plan: feed.plan,
            source: binding(feed.source)?,
            slots,
        })
    };
    let feeds = logical
        .feeds
        .into_iter()
        .map(|feed| match feed {
            LogicalRecordedMultiplicityFeed::Generic(feed) => {
                resolve_generic_feed(feed).map(PlannedRecordedMultiplicityFeedGraph::Generic)
            }
            LogicalRecordedMultiplicityFeed::BlakeGFused {
                plan,
                lut_tables,
                multiplicity_destinations,
            } => Ok(PlannedRecordedMultiplicityFeedGraph::BlakeGFused {
                plan,
                lut_tables: [
                    binding(lut_tables[0])?,
                    binding(lut_tables[1])?,
                    binding(lut_tables[2])?,
                    binding(lut_tables[3])?,
                ],
                multiplicity_destinations: [
                    binding(multiplicity_destinations[0])?,
                    binding(multiplicity_destinations[1])?,
                    binding(multiplicity_destinations[2])?,
                    binding(multiplicity_destinations[3])?,
                    binding(multiplicity_destinations[4])?,
                ],
            }),
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let public_memory_seed = logical
        .public_memory_seed
        .map(resolve_generic_feed)
        .transpose()?;
    let fixed_tables = logical
        .fixed_tables
        .into_iter()
        .map(|fixed| {
            let slots = FixedTableContiguousWorkspaceSlots {
                source_pointers: fixed.source_pointers.map(physical).transpose()?,
                multiplicity_pointers: physical(fixed.multiplicity_pointers)?,
                trace_multiplicity_columns: physical(fixed.trace_multiplicity_columns)?,
                trace_outputs: fixed
                    .trace_outputs
                    .into_iter()
                    .map(physical)
                    .collect::<Result<Vec<_>, _>>()?,
                trace_output_pointers: physical(fixed.trace_output_pointers)?,
                lookup_descriptors: physical(fixed.lookup_descriptors)?,
                lookup_output: physical(fixed.lookup_output)?,
                lookup_output_pointers: physical(fixed.lookup_output_pointers)?,
            };
            fixed
                .plan
                .materializer
                .requirements()
                .arena_slot_requirements_contiguous(&slots)
                .map_err(ArenaPlanError::FixedTable)?;
            Ok(PlannedFixedTableMaterializer {
                plan: fixed.plan,
                sources: fixed
                    .sources
                    .into_iter()
                    .map(|source| match source {
                        LogicalFixedTableSource::Arena(logical) => {
                            binding(logical).map(PlannedFixedTableSource::Arena)
                        }
                        LogicalFixedTableSource::RegisteredPedersen18 { column } => {
                            Ok(PlannedFixedTableSource::RegisteredPedersen18 { column })
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                multiplicity: binding(fixed.multiplicity)?,
                slots,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let memory_traces = logical
        .memory_traces
        .map(|memory| {
            let resolve_part = |part: LogicalMemoryTracePart| {
                Ok::<_, ArenaPlanError>(PlannedMemoryTracePartWorkspace {
                    part: part.part,
                    source_offset: part.source_offset,
                    row_count: part.row_count,
                    outputs: part
                        .outputs
                        .into_iter()
                        .map(binding)
                        .collect::<Result<Vec<_>, _>>()?,
                })
            };
            Ok::<_, ArenaPlanError>(PlannedMemoryBaseTraceWorkspace {
                plan: memory.plan,
                address_outputs: memory
                    .address_outputs
                    .into_iter()
                    .map(binding)
                    .collect::<Result<Vec<_>, _>>()?,
                big_parts: memory
                    .big_parts
                    .into_iter()
                    .map(resolve_part)
                    .collect::<Result<Vec<_>, _>>()?,
                small_part: resolve_part(memory.small_part)?,
                rc99_lut: binding(memory.rc99_lut)?,
                rc99_counts: binding(memory.rc99_counts)?,
            })
        })
        .transpose()?;
    Ok(PlannedGraphAMultiplicityWorkspace {
        topology_hash: logical.topology_hash,
        coverage_gaps: logical.coverage_gaps,
        blockers: logical.blockers,
        multiplicities,
        clear_requirements: logical.clear_requirements,
        clear_slots,
        feeds,
        public_memory_seed,
        fixed_tables,
        memory_traces,
    })
}

fn resolve_relation_slots(
    logical: LogicalRelationWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedRelationWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| find_binding(bindings, id);
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let slots = RelationGraphSlots {
        descriptors: physical(logical.descriptors)?,
        alphas: physical(logical.alphas)?,
        z: physical(logical.z)?,
        inverse_scratch: physical(logical.inverse_scratch)?,
        reduction_a: physical(logical.reduction_a)?,
        reduction_b: physical(logical.reduction_b)?,
        scan_eval_scratch: physical(logical.scan_eval_scratch)?,
        scan_temp_scratch: physical(logical.scan_temp_scratch)?,
        scan_descriptors: physical(logical.scan_descriptors)?,
        fraction_pointers: physical(logical.fraction_pointers)?,
        fraction_geometry: physical(logical.fraction_geometry)?,
        instances: logical
            .instances
            .into_iter()
            .map(|instance| {
                Ok(RelationInstanceSlots {
                    source_pointers: physical(instance.source_pointers)?,
                    output_pointers: physical(instance.output_pointers)?,
                    output_coordinates: instance
                        .output_coordinates
                        .into_iter()
                        .map(physical)
                        .collect::<Result<Vec<_>, ArenaPlanError>>()?,
                    denominators: physical(instance.denominators)?,
                    claimed_sum: physical(instance.claimed_sum)?,
                })
            })
            .collect::<Result<Vec<_>, ArenaPlanError>>()?,
    };
    logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::Relation)?;
    Ok(PlannedRelationWorkspace {
        execution: logical.execution,
        source_plan: logical.source_plan,
        launch_mode: logical.launch_mode,
        requirements: logical.requirements,
        slots,
    })
}

fn checked_pow2(log_size: u32) -> Result<usize, ArenaPlanError> {
    1usize
        .checked_shl(log_size)
        .ok_or(ArenaPlanError::SizeOverflow)
}

#[derive(Debug)]
struct ColoredSlot {
    id: ArenaSlotId,
    len_words: usize,
    /// Union of [`BufferLifetime::epoch_mask`] over every buffer pooled into
    /// this slot. A candidate buffer is compatible iff its own epoch mask is
    /// disjoint from this union: the union intersects the candidate's mask iff
    /// at least one pooled lifetime's mask does, and per-lifetime mask
    /// intersection is exactly [`BufferLifetime::overlaps`] (see the proof on
    /// `epoch_mask`). This makes the compatibility check `O(1)` per slot
    /// instead of `O(lifetimes)` while selecting the identical candidate.
    occupied_epochs: u16,
    /// Kept only to cross-check the mask filter against the original
    /// per-lifetime overlap scan in debug builds.
    #[cfg(debug_assertions)]
    lifetimes: Vec<(LogicalBufferId, BufferLifetime)>,
}

/// Number of leading buffers (in coloring order) whose slot selection is
/// re-derived with the original per-lifetime overlap scan and asserted equal
/// to the epoch-mask selection in debug builds.
#[cfg(debug_assertions)]
const COLORING_REFERENCE_CHECK_BUFFERS: usize = 256;

#[derive(Debug)]
struct ColorUnit {
    members: Vec<usize>,
    first: ProofEpoch,
    id: LogicalBufferId,
    len_words: usize,
    occupied_epochs: u16,
}

fn color_logical_buffers(
    logical: &[LogicalBuffer],
    transition_aliases: &[(LogicalBufferId, LogicalBufferId)],
) -> Result<(Vec<ArenaBinding>, Vec<ArenaSlotSpec>, usize), ArenaPlanError> {
    let aliases = validate_transition_aliases(logical, transition_aliases)?;

    let mut units = Vec::with_capacity(logical.len() - aliases.pair_count());
    let mut visited = BTreeSet::new();
    for index in 0..logical.len() {
        if !visited.insert(index) {
            continue;
        }
        let mut members = vec![index];
        if let Some(other) = aliases.partner_index(index) {
            if !visited.insert(other) {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "logical buffer appears in multiple transition aliases",
                ));
            }
            members.push(other);
        }
        let first = members
            .iter()
            .map(|&member| logical[member].lifetime.first)
            .min()
            .expect("color unit is non-empty");
        let id = members
            .iter()
            .map(|&member| logical[member].id)
            .min()
            .expect("color unit is non-empty");
        let len_words = members
            .iter()
            .map(|&member| logical[member].len_words)
            .max()
            .expect("color unit is non-empty");
        let occupied_epochs = members.iter().fold(0, |mask, &member| {
            mask | logical[member].lifetime.epoch_mask()
        });
        units.push(ColorUnit {
            members,
            first,
            id,
            len_words,
            occupied_epochs,
        });
    }
    // Admit larger units first so a slot's capacity is fixed by its first
    // occupant and can never grow later. First-compatible placement is a
    // deterministic memory heuristic, not a general optimal-coloring claim;
    // exact PIE preflight measures the resulting arena before admission.
    // `first` and `id` make equal-capacity ties deterministic.
    units.sort_unstable_by_key(|unit| (core::cmp::Reverse(unit.len_words), unit.first, unit.id));

    let mut slots: Vec<ColoredSlot> = Vec::new();
    let mut bindings = Vec::with_capacity(logical.len());
    for unit in units {
        let candidate = slots
            .iter()
            .position(|slot| slot.occupied_epochs & unit.occupied_epochs == 0);
        // The mask filter must admit exactly the slots the per-lifetime
        // overlap scan admitted. Cross-check first-fit selection for the first
        // buffers against the original scan in debug builds.
        #[cfg(debug_assertions)]
        if bindings.len() < COLORING_REFERENCE_CHECK_BUFFERS {
            let reference = slots.iter().position(|slot| {
                unit.members.iter().all(|&member| {
                    slot.lifetimes
                        .iter()
                        .all(|&(_, lifetime)| !lifetime.overlaps(logical[member].lifetime))
                })
            });
            debug_assert_eq!(
                candidate, reference,
                "epoch-mask slot selection diverged from the per-lifetime overlap scan"
            );
        }
        let slot_index = match candidate {
            Some(index) => index,
            None => {
                let id = ArenaSlotId(
                    u32::try_from(slots.len() + 1).map_err(|_| ArenaPlanError::SizeOverflow)?,
                );
                slots.push(ColoredSlot {
                    id,
                    len_words: 0,
                    occupied_epochs: 0,
                    #[cfg(debug_assertions)]
                    lifetimes: Vec::new(),
                });
                slots.len() - 1
            }
        };
        let slot = &mut slots[slot_index];
        debug_assert!(slot.len_words == 0 || slot.len_words >= unit.len_words);
        slot.len_words = slot.len_words.max(unit.len_words);
        slot.occupied_epochs |= unit.occupied_epochs;
        #[cfg(debug_assertions)]
        slot.lifetimes.extend(unit.members.iter().map(|&member| {
            let buffer = &logical[member];
            (buffer.id, buffer.lifetime)
        }));
        bindings.extend(unit.members.into_iter().map(|member| {
            let buffer = &logical[member];
            ArenaBinding {
                logical: buffer.id,
                physical: slot.id,
                len_words: buffer.len_words,
            }
        }));
    }
    bindings.sort_unstable_by_key(|binding| binding.logical);

    let mut offset = 0usize;
    let mut specs = Vec::with_capacity(slots.len());
    for slot in slots {
        offset = align_up(offset, ARENA_ALIGNMENT_WORDS)?;
        specs.push(ArenaSlotSpec {
            id: slot.id,
            offset_words: offset,
            len_words: slot.len_words,
            alignment_words: ARENA_ALIGNMENT_WORDS,
        });
        offset = offset
            .checked_add(slot.len_words)
            .ok_or(ArenaPlanError::SizeOverflow)?;
    }
    let total_words = align_up(offset, ARENA_ALIGNMENT_WORDS)?;
    Ok((bindings, specs, total_words))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, ArenaPlanError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(ArenaPlanError::SizeOverflow)
}

fn validate_aliases(
    logical: &[LogicalBuffer],
    bindings: &[ArenaBinding],
) -> Result<(), ArenaPlanError> {
    // Semantically identical to the previous all-pairs scan (for each pair
    // `(i, j)` with `i < j` in `logical` order, error on the lexicographically
    // first same-slot live overlap), but grouped per physical slot so the work
    // is `O(n log n + sum(group^2))` instead of `O(n^2)` pairs each paying a
    // linear binding lookup. Valid colorings bound every group by the epoch
    // count (disjoint inclusive ranges over 12 epochs), so the pairwise stage
    // is effectively linear.
    let Some(first) = logical.first() else {
        return Ok(());
    };
    // The old scan resolved buffer 0's binding, then each later binding right
    // before its `(0, j)` pair check. Reproduce that order: a missing binding
    // at position `m` surfaces as `MissingBinding(m)` unless an aliased
    // `(0, k)` pair with `k < m` precedes it.
    let first_binding = find_binding(bindings, first.id)?;
    let mut resolved = Vec::with_capacity(logical.len());
    resolved.push(first_binding);
    for buffer in &logical[1..] {
        match find_binding(bindings, buffer.id) {
            Ok(binding) => resolved.push(binding),
            Err(missing) => {
                for (candidate, binding) in logical[1..resolved.len()].iter().zip(&resolved[1..]) {
                    if binding.physical == first_binding.physical
                        && first.lifetime.overlaps(candidate.lifetime)
                    {
                        return Err(ArenaPlanError::AliasedLiveBuffers {
                            physical: first_binding.physical,
                            first: first.id,
                            second: candidate.id,
                        });
                    }
                }
                return Err(missing);
            }
        }
    }
    let mut groups: BTreeMap<ArenaSlotId, Vec<usize>> = BTreeMap::new();
    for (index, binding) in resolved.iter().enumerate() {
        groups.entry(binding.physical).or_default().push(index);
    }
    // Scanning each group's indices (ascending, as inserted) yields that
    // group's lexicographically first violating pair; the winner across groups
    // is the pair the all-pairs scan reported.
    let mut earliest: Option<(usize, usize)> = None;
    for group in groups.values() {
        'group: for (position, &first_index) in group.iter().enumerate() {
            for &second_index in &group[position + 1..] {
                if logical[first_index]
                    .lifetime
                    .overlaps(logical[second_index].lifetime)
                {
                    if earliest.is_none_or(|pair| (first_index, second_index) < pair) {
                        earliest = Some((first_index, second_index));
                    }
                    break 'group;
                }
            }
        }
    }
    if let Some((first_index, second_index)) = earliest {
        return Err(ArenaPlanError::AliasedLiveBuffers {
            physical: resolved[first_index].physical,
            first: logical[first_index].id,
            second: logical[second_index].id,
        });
    }
    Ok(())
}

fn high_water_at(epoch: ProofEpoch, logical: &[LogicalBuffer], bindings: &[ArenaBinding]) -> usize {
    // Identical to summing, over the distinct physical slots with a live
    // buffer at `epoch`, the largest bound length on each slot — but with the
    // per-slot maxima computed in one pass instead of rescanning `bindings`
    // for every newly seen slot.
    let mut slot_capacity = BTreeMap::<ArenaSlotId, usize>::new();
    for binding in bindings {
        let capacity = slot_capacity.entry(binding.physical).or_insert(0);
        *capacity = (*capacity).max(binding.len_words);
    }
    let mut physical = BTreeSet::<ArenaSlotId>::new();
    let mut words = 0usize;
    for buffer in logical
        .iter()
        .filter(|buffer| buffer.lifetime.contains(epoch))
    {
        let binding = find_binding(bindings, buffer.id)
            .expect("bindings are complete before high-water computation");
        if physical.insert(binding.physical) {
            words += slot_capacity
                .get(&binding.physical)
                .expect("the active binding belongs to its physical slot");
        }
    }
    words
}

#[cfg(test)]
mod tests {
    use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
    use stwo_cairo_prover::witness::proof_shape::{
        ProofShape, RuntimeComponentShape, TracePartShape,
    };

    use super::*;
    use crate::relation_table::CAIRO_RELATION_GRAPH;
    use crate::schedule_table::CAIRO_SCHEDULE;

    fn test_buffer(id: u32, words: usize, first: ProofEpoch, last: ProofEpoch) -> LogicalBuffer {
        LogicalBuffer {
            id: LogicalBufferId(id),
            component: None,
            part: None,
            purpose: BufferPurpose::CommitLdeTile,
            ordinal: id,
            len_words: words,
            lifetime: BufferLifetime::new(first, last).unwrap(),
        }
    }

    #[test]
    fn process_owned_pedersen_removes_exactly_one_evaluation_table_from_arena() {
        let omitted_words = (0..crate::fixed_table_materializer::PEDERSEN_POINTS_18_COLUMN_COUNT)
            .map(|index| format!("pedersen_points_{index}"))
            .filter(|identity| !retain_fixed_preprocessed_evaluation(identity))
            .map(|_| PEDERSEN_POINTS_18_ROW_COUNT)
            .sum::<usize>();
        assert_eq!(
            omitted_words * core::mem::size_of::<u32>(),
            crate::fixed_table_materializer::PEDERSEN_POINTS_18_EVALUATION_BYTES
        );
        assert_eq!(
            crate::fixed_table_materializer::PEDERSEN_POINTS_18_EVALUATION_BYTES,
            1_879_048_192
        );
        assert!(retain_fixed_preprocessed_evaluation("seq_23"));
        assert!(retain_fixed_preprocessed_evaluation(
            "pedersen_points_small_0"
        ));
    }

    #[test]
    fn registration_requirement_is_driven_by_preprocessed_identity() {
        let binding = ArenaBinding {
            logical: LogicalBufferId(0),
            physical: ArenaSlotId(0),
            len_words: PEDERSEN_POINTS_18_ROW_COUNT,
        };
        let mut columns = vec![PlannedPreprocessedColumn {
            identity: "pedersen_points_55".to_owned(),
            ordinal: 0,
            log_size: PEDERSEN_POINTS_18_LOG_SIZE,
            evaluations: None,
            coefficients: binding,
        }];
        assert!(preprocessed_requires_registered_pedersen_table(&columns));

        for identity in [
            "seq_23",
            "pedersen_points_small_0",
            "pedersen_points_055",
            "pedersen_points_56",
        ] {
            columns[0].identity = identity.to_owned();
            assert!(
                !preprocessed_requires_registered_pedersen_table(&columns),
                "forged/non-W18 identity {identity} must not request the global table"
            );
        }
    }

    #[test]
    fn disjoint_epochs_alias_but_live_ranges_never_do() {
        let logical = vec![
            test_buffer(0, 128, ProofEpoch::Witness, ProofEpoch::Witness),
            test_buffer(1, 96, ProofEpoch::Interaction, ProofEpoch::Interaction),
            test_buffer(2, 64, ProofEpoch::Witness, ProofEpoch::Interaction),
        ];
        let (bindings, specs, total) = color_logical_buffers(&logical, &[]).unwrap();
        ArenaLayout::new(total, &specs).unwrap();
        validate_aliases(&logical, &bindings).unwrap();

        let slot = |id| {
            bindings
                .iter()
                .find(|binding| binding.logical == LogicalBufferId(id))
                .unwrap()
                .physical
        };
        assert_eq!(slot(0), slot(1));
        assert_ne!(slot(0), slot(2));
        assert_ne!(slot(1), slot(2));
    }

    #[test]
    fn transition_alias_reserves_one_slot_for_both_values() {
        let mut evaluations = test_buffer(0, 128, ProofEpoch::Witness, ProofEpoch::Witness);
        evaluations.component = Some("component");
        evaluations.part = Some(TracePartId::Main);
        evaluations.purpose = BufferPurpose::BaseTrace;
        evaluations.ordinal = 7;
        let mut coefficients = test_buffer(1, 128, ProofEpoch::BaseCommit, ProofEpoch::Decommit);
        coefficients.component = evaluations.component;
        coefficients.part = evaluations.part;
        coefficients.purpose = BufferPurpose::BaseCoefficients;
        coefficients.ordinal = evaluations.ordinal;
        let blocker = test_buffer(2, 128, ProofEpoch::Interaction, ProofEpoch::Composition);
        let logical = vec![evaluations, coefficients, blocker];

        let (bindings, specs, total) =
            color_logical_buffers(&logical, &[(LogicalBufferId(0), LogicalBufferId(1))]).unwrap();
        ArenaLayout::new(total, &specs).unwrap();
        validate_aliases(&logical, &bindings).unwrap();
        let physical = |id| {
            find_binding(&bindings, LogicalBufferId(id))
                .unwrap()
                .physical
        };
        assert_eq!(physical(0), physical(1));
        assert_ne!(physical(0), physical(2));
    }

    #[test]
    fn transition_alias_rejects_a_still_live_evaluation() {
        let mut evaluations = test_buffer(0, 128, ProofEpoch::Witness, ProofEpoch::Interaction);
        evaluations.component = Some("component");
        evaluations.part = Some(TracePartId::Main);
        evaluations.purpose = BufferPurpose::BaseTrace;
        let mut coefficients = test_buffer(1, 128, ProofEpoch::BaseCommit, ProofEpoch::Decommit);
        coefficients.component = evaluations.component;
        coefficients.part = evaluations.part;
        coefficients.purpose = BufferPurpose::BaseCoefficients;
        assert!(matches!(
            color_logical_buffers(
                &[evaluations, coefficients],
                &[(LogicalBufferId(0), LogicalBufferId(1))],
            ),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "invalid interpolation transition alias"
            ))
        ));
    }

    #[test]
    fn transition_alias_rejects_an_epoch_gap() {
        let mut evaluations = test_buffer(0, 128, ProofEpoch::Witness, ProofEpoch::Witness);
        evaluations.component = Some("component");
        evaluations.part = Some(TracePartId::Main);
        evaluations.purpose = BufferPurpose::BaseTrace;
        let mut coefficients = test_buffer(1, 128, ProofEpoch::Interaction, ProofEpoch::Decommit);
        coefficients.component = evaluations.component;
        coefficients.part = evaluations.part;
        coefficients.purpose = BufferPurpose::BaseCoefficients;
        assert!(matches!(
            color_logical_buffers(
                &[evaluations, coefficients],
                &[(LogicalBufferId(0), LogicalBufferId(1))],
            ),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "invalid interpolation transition alias"
            ))
        ));
    }

    #[test]
    fn physical_slots_are_128_byte_aligned() {
        let logical = vec![
            test_buffer(0, 33, ProofEpoch::Witness, ProofEpoch::Witness),
            test_buffer(1, 65, ProofEpoch::Witness, ProofEpoch::Witness),
        ];
        let (_, specs, total) = color_logical_buffers(&logical, &[]).unwrap();
        assert_eq!(total % ARENA_ALIGNMENT_WORDS, 0);
        assert!(specs
            .iter()
            .all(|spec| spec.offset_words % ARENA_ALIGNMENT_WORDS == 0));
    }

    /// The colorer's `O(1)` epoch-bitmask compatibility check must equal the
    /// per-lifetime `BufferLifetime::overlaps` scan it replaced, for every
    /// pair of valid inclusive epoch ranges (exhaustive over first/last in
    /// `ProofEpoch::ALL` on both sides, i.e. the full 12x12x12x12 space; the
    /// inverted first > last combinations are exactly the ones
    /// `BufferLifetime::new` rejects).
    #[test]
    fn epoch_mask_intersection_equals_lifetime_overlap_for_all_ranges() {
        let mut ranges = Vec::new();
        for first in ProofEpoch::ALL {
            for last in ProofEpoch::ALL {
                match BufferLifetime::new(first, last) {
                    Ok(lifetime) => ranges.push(lifetime),
                    Err(_) => assert!(first > last),
                }
            }
        }
        assert_eq!(ranges.len(), 12 * 13 / 2);
        for &a in &ranges {
            for &b in &ranges {
                assert_eq!(
                    a.epoch_mask() & b.epoch_mask() != 0,
                    a.overlaps(b),
                    "mask intersection diverges from overlaps for {a:?} vs {b:?}"
                );
            }
        }
    }

    /// Per-lifetime reference for the descending-capacity colorer: compatibility
    /// scans every lifetime already pooled into a slot, while production uses
    /// the equivalent epoch-union mask.
    fn reference_color_logical_buffers(
        logical: &[LogicalBuffer],
    ) -> (Vec<ArenaBinding>, Vec<ArenaSlotSpec>, usize) {
        struct ReferenceSlot {
            id: ArenaSlotId,
            len_words: usize,
            lifetimes: Vec<BufferLifetime>,
        }
        let mut order: Vec<usize> = (0..logical.len()).collect();
        order.sort_unstable_by_key(|&index| {
            let buffer = &logical[index];
            (
                core::cmp::Reverse(buffer.len_words),
                buffer.lifetime.first,
                buffer.id,
            )
        });
        let mut slots: Vec<ReferenceSlot> = Vec::new();
        let mut bindings = Vec::with_capacity(logical.len());
        for index in order {
            let buffer = &logical[index];
            let candidate = slots.iter().position(|slot| {
                slot.lifetimes
                    .iter()
                    .all(|&lifetime| !lifetime.overlaps(buffer.lifetime))
            });
            let slot_index = candidate.unwrap_or_else(|| {
                slots.push(ReferenceSlot {
                    id: ArenaSlotId(u32::try_from(slots.len() + 1).unwrap()),
                    len_words: 0,
                    lifetimes: Vec::new(),
                });
                slots.len() - 1
            });
            let slot = &mut slots[slot_index];
            assert!(slot.len_words == 0 || slot.len_words >= buffer.len_words);
            slot.len_words = slot.len_words.max(buffer.len_words);
            slot.lifetimes.push(buffer.lifetime);
            bindings.push(ArenaBinding {
                logical: buffer.id,
                physical: slot.id,
                len_words: buffer.len_words,
            });
        }
        bindings.sort_unstable_by_key(|binding| binding.logical);
        let mut offset = 0usize;
        let mut specs = Vec::with_capacity(slots.len());
        for slot in slots {
            offset = align_up(offset, ARENA_ALIGNMENT_WORDS).unwrap();
            specs.push(ArenaSlotSpec {
                id: slot.id,
                offset_words: offset,
                len_words: slot.len_words,
                alignment_words: ARENA_ALIGNMENT_WORDS,
            });
            offset += slot.len_words;
        }
        (
            bindings,
            specs,
            align_up(offset, ARENA_ALIGNMENT_WORDS).unwrap(),
        )
    }

    /// The optimized colorer must reproduce the per-lifetime first-fit
    /// reference exactly across a deterministic population that exercises
    /// pooling, ties, and every epoch range shape past the debug check window.
    #[test]
    fn bitmask_coloring_matches_reference_coloring_exactly() {
        let mut state = 0x243f_6a88_85a3_08d3u64; // deterministic LCG
        let mut next = move |bound: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % bound
        };
        let mut logical = Vec::new();
        for id in 0..2048u32 {
            let first = ProofEpoch::ALL[next(12) as usize];
            let last = ProofEpoch::ALL[(first as u64 + next(12 - first as u64)) as usize];
            // Small word-length pool so best-fit ties are common.
            let words = 1 + next(7) as usize * 64;
            logical.push(LogicalBuffer {
                id: LogicalBufferId(id),
                component: None,
                part: None,
                purpose: BufferPurpose::CommitLdeTile,
                ordinal: id,
                len_words: words,
                lifetime: BufferLifetime::new(first, last).unwrap(),
            });
        }
        let (bindings, specs, total) = color_logical_buffers(&logical, &[]).unwrap();
        let (expected_bindings, expected_specs, expected_total) =
            reference_color_logical_buffers(&logical);
        assert_eq!(bindings, expected_bindings);
        assert_eq!(specs, expected_specs);
        assert_eq!(total, expected_total);
        validate_aliases(&logical, &bindings).unwrap();
    }

    #[test]
    fn generated_proof_shape_builds_a_complete_alias_checked_arena() {
        let default_shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let mut components = default_shape.components().to_vec();
        // The resident multiplicity plan requires both memory tables present
        // together, so the address table gets a shape alongside the split
        // value table.
        *components
            .iter_mut()
            .find(|component| component.id == "memory_address_to_id")
            .unwrap() = RuntimeComponentShape::uniform("memory_address_to_id", 5, 16).unwrap();
        *components
            .iter_mut()
            .find(|component| component.id == "memory_id_to_big")
            .unwrap() = RuntimeComponentShape::parts(
            "memory_id_to_big",
            vec![
                TracePartShape {
                    part: TracePartId::MemoryBig(0),
                    n_real_rows: 33,
                    padded_rows: 64,
                },
                TracePartShape {
                    part: TracePartId::MemorySmall,
                    n_real_rows: 17,
                    padded_rows: 32,
                },
            ],
        )
        .unwrap();
        let rc99_rows = 1u64 << cairo_air::components::range_check_9_9::LOG_SIZE;
        *components
            .iter_mut()
            .find(|component| component.id == "range_check_9_9")
            .unwrap() =
            RuntimeComponentShape::uniform("range_check_9_9", rc99_rows, rc99_rows).unwrap();
        let shape = ProofShape::new(components).unwrap();
        let proof =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let group_columns = |mut columns: Vec<(u32, CommitmentColumnSource)>| {
            columns.sort_by_key(|(log_size, _)| *log_size);
            let logs: Vec<Vec<u32>> = columns
                .chunks(16)
                .map(|group| group.iter().map(|(log_size, _)| *log_size).collect())
                .collect();
            let sources: Vec<Vec<CommitmentColumnSource>> = columns
                .chunks(16)
                .map(|group| group.iter().map(|(_, source)| *source).collect())
                .collect();
            (logs, sources)
        };
        let dynamic_geometry = |id| {
            let component = proof
                .components
                .iter()
                .find(|component| component.node.id == id)
                .unwrap();
            let parts = capacity_parts(&component.runtime.rows).unwrap();
            assert_eq!(parts.len(), 1);
            assert_eq!(parts[0].part, TracePartId::Main);
            (
                parts[0].padded_rows.ilog2(),
                trace_width(component.node.facts.trace_columns, TracePartId::Main).unwrap(),
                component.node.facts.logup_columns.unwrap() as usize * SECURE_FIELD_WORDS,
            )
        };
        let (address_log_size, address_base_columns, address_interaction_columns) =
            dynamic_geometry("memory_address_to_id");
        let (rc99_log_size, rc99_base_columns, rc99_interaction_columns) =
            dynamic_geometry("range_check_9_9");
        let (base_logs, base_sources) = group_columns(
            (0..address_base_columns)
                .map(|ordinal| {
                    (
                        address_log_size,
                        CommitmentColumnSource::Trace {
                            component: "memory_address_to_id",
                            part: TracePartId::Main,
                            purpose: BufferPurpose::BaseCoefficients,
                            ordinal,
                        },
                    )
                })
                .chain(
                    (0..cairo_air::components::memory_id_to_big::BIG_N_COLUMNS as u32).map(
                        |ordinal| {
                            (
                                6,
                                CommitmentColumnSource::Trace {
                                    component: "memory_id_to_big",
                                    part: TracePartId::MemoryBig(0),
                                    purpose: BufferPurpose::BaseCoefficients,
                                    ordinal,
                                },
                            )
                        },
                    ),
                )
                .chain(
                    (0..cairo_air::components::memory_id_to_small::N_TRACE_COLUMNS as u32).map(
                        |ordinal| {
                            (
                                5,
                                CommitmentColumnSource::Trace {
                                    component: "memory_id_to_big",
                                    part: TracePartId::MemorySmall,
                                    purpose: BufferPurpose::BaseCoefficients,
                                    ordinal,
                                },
                            )
                        },
                    ),
                )
                .chain((0..rc99_base_columns).map(|ordinal| {
                    (
                        rc99_log_size,
                        CommitmentColumnSource::Trace {
                            component: "range_check_9_9",
                            part: TracePartId::Main,
                            purpose: BufferPurpose::BaseCoefficients,
                            ordinal,
                        },
                    )
                }))
                .collect(),
        );
        let (interaction_logs, interaction_sources) = group_columns(
            (0..u32::try_from(address_interaction_columns).unwrap())
                .map(|ordinal| {
                    (
                        address_log_size,
                        CommitmentColumnSource::Trace {
                            component: "memory_address_to_id",
                            part: TracePartId::Main,
                            purpose: BufferPurpose::InteractionCoefficients,
                            ordinal,
                        },
                    )
                })
                .chain((0..32).map(|ordinal| {
                    (
                        6,
                        CommitmentColumnSource::Trace {
                            component: "memory_id_to_big",
                            part: TracePartId::MemoryBig(0),
                            purpose: BufferPurpose::InteractionCoefficients,
                            ordinal,
                        },
                    )
                }))
                .chain((0..12).map(|ordinal| {
                    (
                        5,
                        CommitmentColumnSource::Trace {
                            component: "memory_id_to_big",
                            part: TracePartId::MemorySmall,
                            purpose: BufferPurpose::InteractionCoefficients,
                            ordinal,
                        },
                    )
                }))
                .chain(
                    (0..u32::try_from(rc99_interaction_columns).unwrap()).map(|ordinal| {
                        (
                            rc99_log_size,
                            CommitmentColumnSource::Trace {
                                component: "range_check_9_9",
                                part: TracePartId::Main,
                                purpose: BufferPurpose::InteractionCoefficients,
                                ordinal,
                            },
                        )
                    }),
                )
                .collect(),
        );
        let mut oods_columns = vec![
            OodsColumnGeometry {
                source: OpenedColumnSource::Preprocessed { ordinal: 0 },
                coefficient_log_size: 25,
                evaluation_log_size: 26,
                shape_points: Vec::new(),
                offset_points: Vec::new(),
            },
            OodsColumnGeometry {
                source: OpenedColumnSource::Preprocessed { ordinal: 1 },
                coefficient_log_size: 18,
                evaluation_log_size: 19,
                shape_points: Vec::new(),
                offset_points: Vec::new(),
            },
            OodsColumnGeometry {
                source: OpenedColumnSource::Preprocessed { ordinal: 2 },
                coefficient_log_size: 18,
                evaluation_log_size: 19,
                shape_points: Vec::new(),
                offset_points: Vec::new(),
            },
        ];
        oods_columns.extend(
            base_logs
                .iter()
                .flatten()
                .copied()
                .zip(base_sources.iter().flatten().copied())
                .map(|(coefficient_log_size, source)| OodsColumnGeometry {
                    source: source.into(),
                    coefficient_log_size,
                    evaluation_log_size: coefficient_log_size + 1,
                    shape_points: Vec::new(),
                    offset_points: Vec::new(),
                }),
        );
        oods_columns.extend(
            interaction_logs
                .iter()
                .flatten()
                .copied()
                .zip(interaction_sources.iter().flatten().copied())
                .map(|(coefficient_log_size, source)| OodsColumnGeometry {
                    source: source.into(),
                    coefficient_log_size,
                    evaluation_log_size: coefficient_log_size + 1,
                    shape_points: Vec::new(),
                    offset_points: Vec::new(),
                }),
        );
        let offset_step = stwo::core::poly::circle::CanonicCoset::new(25).step();
        let offsets = (1..=3)
            .map(|multiple| offset_step.mul(multiple))
            .collect::<Vec<_>>();
        let shape_points = offsets
            .iter()
            .map(|&offset| {
                stwo::core::circle::SECURE_FIELD_CIRCLE_GEN + offset.into_ef::<SecureField>()
            })
            .collect::<Vec<_>>();
        oods_columns.extend((0..8).map(|ordinal| {
            OodsColumnGeometry {
                source: OpenedColumnSource::Composition { ordinal },
                coefficient_log_size: 25,
                evaluation_log_size: 26,
                shape_points: (ordinal == 0)
                    .then(|| shape_points.clone())
                    .unwrap_or_default(),
                offset_points: (ordinal == 0).then(|| offsets.clone()).unwrap_or_default(),
            }
        }));
        let oods = OodsGeometry {
            mask_log_size: 25,
            sampled_values_input: TranscriptInputId(25),
            point_parameter_output: TranscriptOutputId(3),
            quotient_random_coefficient_output: TranscriptOutputId(4),
            columns: oods_columns,
        };
        let transcript_schedule = stwo_backend_cuda::Blake2sTranscriptSchedule::new(
            stwo_backend_cuda::TranscriptStart::Default,
            vec![
                stwo_backend_cuda::TranscriptOperation::MixFelts {
                    boundary: stwo_backend_cuda::TranscriptBoundaryId(1),
                    source: stwo_backend_cuda::TranscriptInputId(1),
                    n_felts: 1,
                },
                stwo_backend_cuda::TranscriptOperation::DrawSecureFelt {
                    boundary: stwo_backend_cuda::TranscriptBoundaryId(2),
                    output: stwo_backend_cuda::TranscriptOutputId(1),
                },
                stwo_backend_cuda::TranscriptOperation::DrawSecureFelt {
                    boundary: stwo_backend_cuda::TranscriptBoundaryId(3),
                    output: stwo_backend_cuda::TranscriptOutputId(2),
                },
                stwo_backend_cuda::TranscriptOperation::MixFelts {
                    boundary: stwo_backend_cuda::TranscriptBoundaryId(4),
                    source: stwo_backend_cuda::TranscriptInputId(22),
                    n_felts: 1,
                },
                stwo_backend_cuda::TranscriptOperation::DrawSecureFelt {
                    boundary: stwo_backend_cuda::TranscriptBoundaryId(5),
                    output: stwo_backend_cuda::TranscriptOutputId(3),
                },
                stwo_backend_cuda::TranscriptOperation::MixFelts {
                    boundary: stwo_backend_cuda::TranscriptBoundaryId(6),
                    source: stwo_backend_cuda::TranscriptInputId(25),
                    n_felts: 3,
                },
                stwo_backend_cuda::TranscriptOperation::DrawSecureFelt {
                    boundary: stwo_backend_cuda::TranscriptBoundaryId(7),
                    output: stwo_backend_cuda::TranscriptOutputId(4),
                },
                stwo_backend_cuda::TranscriptOperation::MixFelts {
                    boundary: stwo_backend_cuda::TranscriptBoundaryId(8),
                    source: stwo_backend_cuda::TranscriptInputId(30),
                    n_felts: 1,
                },
                stwo_backend_cuda::TranscriptOperation::DrawQueries {
                    boundary: stwo_backend_cuda::TranscriptBoundaryId(9),
                    output: stwo_backend_cuda::TranscriptOutputId(5),
                    log_domain_size: 26,
                    n_queries: 70,
                },
            ],
            8,
        )
        .unwrap();
        let constant = SecureField::from(7u32);
        let composition = CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 1,
            max_evaluation_log_size: 26,
            components: vec![crate::composition_plan::CompositionComponentPlan {
                component: "memory_id_to_big",
                instance: 0,
                trace_locations: vec![
                    stwo::core::pcs::TreeSubspan {
                        tree_index: 0,
                        col_start: 0,
                        col_end: 0,
                    },
                    stwo::core::pcs::TreeSubspan {
                        tree_index: 1,
                        col_start: 0,
                        col_end: 1,
                    },
                    stwo::core::pcs::TreeSubspan {
                        tree_index: 2,
                        col_start: 0,
                        col_end: 1,
                    },
                ],
                preprocessed_column_indices: vec![0],
                trace_log_size: 25,
                evaluation_log_size: 26,
                n_constraints: 1,
                random_coefficient_offset: 0,
                denominator_inverses: vec![BaseField::from(1); 2],
                base_param_values: Vec::new(),
                ext_param_values: vec![
                    SecureField::default(),
                    SecureField::default(),
                    SecureField::default(),
                    constant,
                ],
                ext_param_sources: vec![
                    CompositionExtParamSource::LookupZ,
                    CompositionExtParamSource::LookupAlphaPower(1),
                    CompositionExtParamSource::ClaimedSumScaled,
                    CompositionExtParamSource::Constant(constant),
                ],
                kernels: vec![crate::composition_plan::CompositionKernelPart {
                    kernel_name: "kernel".to_owned(),
                    cache_key: 7,
                    semantic_hash: 9,
                    source: "extern \"C\" __global__ void kernel() {}".to_owned(),
                    rc_base: 0,
                }],
            }],
        };
        let protocol = ProtocolGeometry {
            identity: ProtocolIdentity {
                pow_bits: 26,
                log_blowup_factor: 1,
                log_last_layer_degree_bound: 0,
                fri_fold_step: 1,
                channel_tag: 1,
                relation_graph_hash: proof.relation_graph_hash,
                preprocessed_binding_hash: 3,
                oods_topology_hash: oods.topology_hash(),
                composition_plan_hash: composition.key(),
                kernel_manifest_hash: 4,
                decommit_strategy: DecommitStrategy::RecomputeQueriedLde,
                interpolation_mode: InterpolationLaunchMode::StageWiseCopyThenInPlace,
                blake2s_interior_fused: false,
                composition_launch_mode: CompositionLaunchMode::Serial,
                relation_tail_mode: RelationTailMode::Segmented,
                fri_fold_launch_mode: FriFoldLaunchMode::PerFold,
                witness_feed_launch_mode: WitnessFeedLaunchMode::GlobalAtomics,
                resident_backend: ResidentBackend::LegacyResident,
                quotient_numerator_schedule: QuotientNumeratorSchedule::LegacyBatches,
                quotient_numerator_source_policy: QuotientNumeratorSourcePolicy::CoefficientsOnly,
                commit_mode: ProgressiveCommitMode::FullLifting,
                direct_composition_retention_mode: DirectCompositionRetentionMode::Disabled,
                direct_composition_planner_key: 0,
                direct_composition_occurrence_bitmap_hash: 0,
                direct_composition_group_rounded_bytes: 0,
                numerator_evaluation_group_rounded_bytes: 0,
                retained_evaluation_union_bytes: 0,
            },
            preprocessed_column_ids: vec![
                "test_preprocessed".to_owned(),
                "range_check_9_9_column_0".to_owned(),
                "range_check_9_9_column_1".to_owned(),
            ],
            max_domain_log_size: 26,
            lifting_log_size: 26,
            n_queries: 70,
            total_opened_columns: oods.columns.len(),
            proof_capacity_words: 1 << 20,
            transcript: TranscriptGeometry {
                schedule_key: transcript_schedule.protocol_key(),
                requirements: transcript_schedule.requirements().clone(),
            },
            composition_random_coefficient_output: TranscriptOutputId(2),
            oods,
            quotient: QuotientGeometry {
                partial_numerator_log_sizes: vec![25, 25, 25],
            },
            commitments: vec![
                CommitmentGeometry {
                    id: CommitmentTreeId::Preprocessed,
                    created: ProofEpoch::Ingest,
                    config: CommitWorkspaceConfig {
                        log_blowup_factor: 1,
                        lifting_log_size: 26,
                        unretained_bottom_layers: 4,
                        max_fused_tail_levels: 12,
                    },
                    grouped_column_log_sizes: vec![vec![18, 18, 25]],
                    grouped_column_sources: vec![vec![
                        CommitmentColumnSource::Preprocessed { ordinal: 1 },
                        CommitmentColumnSource::Preprocessed { ordinal: 2 },
                        CommitmentColumnSource::Preprocessed { ordinal: 0 },
                    ]],
                    retained_evaluation_groups: vec![false],
                    direct_composition_evaluation_groups: vec![false],
                    numerator_evaluation_groups: vec![false],
                },
                CommitmentGeometry {
                    id: CommitmentTreeId::Base,
                    created: ProofEpoch::BaseCommit,
                    config: CommitWorkspaceConfig {
                        log_blowup_factor: 1,
                        lifting_log_size: 26,
                        unretained_bottom_layers: 4,
                        max_fused_tail_levels: 12,
                    },
                    retained_evaluation_groups: vec![false; base_logs.len()],
                    direct_composition_evaluation_groups: vec![false; base_logs.len()],
                    numerator_evaluation_groups: vec![false; base_logs.len()],
                    grouped_column_log_sizes: base_logs,
                    grouped_column_sources: base_sources,
                },
                CommitmentGeometry {
                    id: CommitmentTreeId::Interaction,
                    created: ProofEpoch::InteractionCommit,
                    config: CommitWorkspaceConfig {
                        log_blowup_factor: 1,
                        lifting_log_size: 26,
                        unretained_bottom_layers: 4,
                        max_fused_tail_levels: 12,
                    },
                    retained_evaluation_groups: vec![false; interaction_logs.len()],
                    direct_composition_evaluation_groups: vec![false; interaction_logs.len()],
                    numerator_evaluation_groups: vec![false; interaction_logs.len()],
                    grouped_column_log_sizes: interaction_logs,
                    grouped_column_sources: interaction_sources,
                },
                CommitmentGeometry {
                    id: CommitmentTreeId::Composition,
                    created: ProofEpoch::CompositionCommit,
                    config: CommitWorkspaceConfig {
                        log_blowup_factor: 1,
                        lifting_log_size: 26,
                        unretained_bottom_layers: 4,
                        max_fused_tail_levels: 12,
                    },
                    grouped_column_log_sizes: vec![vec![25; 8]],
                    grouped_column_sources: vec![(0..8)
                        .map(|ordinal| CommitmentColumnSource::Composition { ordinal })
                        .collect()],
                    retained_evaluation_groups: vec![false],
                    direct_composition_evaluation_groups: vec![false],
                    numerator_evaluation_groups: vec![false],
                },
            ],
            direct_composition_retention: None,
            opened_tree_log_sizes: vec![26, 26, 26, 26],
            fri_layer_log_sizes: (2..=26).rev().collect(),
        };
        assert_eq!(
            protocol
                .quotient_numerator_workspace_config()
                .unwrap()
                .max_lde_tile_words,
            1 << 26,
            "legacy graph identity keeps one maximal lifted column"
        );
        let mut replacement_tile = protocol.clone();
        replacement_tile.identity.resident_backend = ResidentBackend::ReplacementV1;
        assert_eq!(
            replacement_tile
                .quotient_numerator_workspace_config()
                .unwrap()
                .max_lde_tile_words,
            REPLACEMENT_NUMERATOR_LDE_TILE_COLUMNS * (1 << 26),
            "replacement graph identity seals the wider coefficient tile"
        );

        let mut changed_quotient = protocol.clone();
        changed_quotient
            .quotient
            .partial_numerator_log_sizes
            .push(23);
        assert_ne!(protocol.key(), changed_quotient.key());
        let mut fused_interpolation = protocol.clone();
        fused_interpolation.identity.interpolation_mode =
            InterpolationLaunchMode::StageFusedOutOfPlace;
        assert_ne!(protocol.key(), fused_interpolation.key());
        let identity_mutations: [fn(&mut ProtocolGeometry); 7] = [
            |changed| changed.identity.blake2s_interior_fused = true,
            |changed| changed.identity.composition_launch_mode = CompositionLaunchMode::Wide,
            |changed| changed.identity.relation_tail_mode = RelationTailMode::Scan,
            |changed| changed.identity.fri_fold_launch_mode = FriFoldLaunchMode::FusedTriple,
            |changed| changed.identity.witness_feed_launch_mode = WitnessFeedLaunchMode::Privatized,
            |changed| changed.identity.resident_backend = ResidentBackend::ReplacementV1,
            |changed| {
                changed.identity.quotient_numerator_schedule =
                    QuotientNumeratorSchedule::HybridSingleWrite
            },
        ];
        for mutate in identity_mutations {
            let mut changed = protocol.clone();
            mutate(&mut changed);
            assert_ne!(protocol.key(), changed.key());
        }
        let mut retained_numerator = protocol.clone();
        retained_numerator.identity.commit_mode = ProgressiveCommitMode::DomainProgressive;
        retained_numerator.identity.quotient_numerator_source_policy =
            QuotientNumeratorSourcePolicy::ReuseRetainedEvaluations;
        let composition_commitment = retained_numerator
            .commitments
            .iter_mut()
            .find(|commitment| commitment.id == CommitmentTreeId::Composition)
            .unwrap();
        composition_commitment.numerator_evaluation_groups[0] = true;
        let numerator_bytes = commitment_group_evaluation_bytes(composition_commitment, 0).unwrap();
        retained_numerator
            .identity
            .numerator_evaluation_group_rounded_bytes = numerator_bytes;
        retained_numerator.identity.retained_evaluation_union_bytes = numerator_bytes;
        assert_ne!(protocol.key(), retained_numerator.key());
        let retained_kinds = retained_numerator
            .quotient_numerator_source_kinds()
            .unwrap();
        assert_eq!(
            retained_kinds
                .iter()
                .filter(|&&kind| kind == QuotientNumeratorSourceKind::Evaluation)
                .count(),
            1,
            "only the sampled retained composition column is eligible"
        );
        let mut sampled_preprocessed = retained_numerator.clone();
        let sampled_shape = sampled_preprocessed
            .oods
            .columns
            .iter()
            .find(|column| !column.shape_points.is_empty())
            .unwrap()
            .shape_points
            .clone();
        sampled_preprocessed.oods.columns[0].shape_points = sampled_shape;
        assert!(!sampled_preprocessed.commitments[0].numerator_evaluation_groups[0]);
        assert_eq!(
            sampled_preprocessed
                .quotient_numerator_source_kinds()
                .unwrap()[0],
            QuotientNumeratorSourceKind::Coefficients
        );
        let coefficient_requirements = quotient_numerator_workspace_requirements(
            protocol.quotient_numerator_workspace_config().unwrap(),
            &protocol.quotient_numerator_topologies().unwrap(),
        )
        .unwrap();
        let retained_requirements = quotient_numerator_workspace_requirements(
            retained_numerator
                .quotient_numerator_workspace_config()
                .unwrap(),
            &retained_numerator.quotient_numerator_topologies().unwrap(),
        )
        .unwrap();
        assert_eq!(
            coefficient_requirements
                .groups
                .iter()
                .map(|group| (group.shape_point, group.log_size, group.value_words))
                .collect::<Vec<_>>(),
            retained_requirements
                .groups
                .iter()
                .map(|group| (group.shape_point, group.log_size, group.value_words))
                .collect::<Vec<_>>()
        );
        assert!(coefficient_requirements
            .groups
            .iter()
            .all(|group| group.coefficient_source_count == 1));
        assert!(retained_requirements
            .groups
            .iter()
            .all(|group| group.coefficient_source_count == 0));
        assert_eq!(
            coefficient_requirements.term_count,
            retained_requirements.term_count
        );
        assert_eq!(
            coefficient_requirements.output_pointer_words,
            retained_requirements.output_pointer_words
        );
        assert!(
            retained_requirements.coefficient_pointer_words
                < coefficient_requirements.coefficient_pointer_words
        );
        retained_numerator.validate().unwrap();
        let retained_arena =
            ProofArenaPlan::build(&proof, &retained_numerator, &composition).unwrap();
        retained_arena.validate_aliases().unwrap();
        let retained_commitment = retained_arena
            .commitments()
            .iter()
            .find(|commitment| commitment.id == CommitmentTreeId::Composition)
            .unwrap();
        assert_eq!(
            retained_commitment.numerator_evaluation_groups[0],
            retained_commitment.evaluation_output_groups[0]
        );
        assert!(retained_commitment.retained_evaluation_groups[0].is_none());
        assert!(retained_commitment.direct_composition_evaluation_groups[0].is_none());
        for binding in retained_commitment.numerator_evaluation_groups[0]
            .as_ref()
            .unwrap()
        {
            let logical = retained_arena
                .logical_buffers()
                .iter()
                .find(|buffer| buffer.id == binding.logical)
                .unwrap();
            assert_eq!(logical.lifetime.last, ProofEpoch::Quotient);
        }
        let mut missing_numerator_group = retained_numerator.clone();
        missing_numerator_group
            .commitments
            .iter_mut()
            .find(|commitment| commitment.id == CommitmentTreeId::Composition)
            .unwrap()
            .numerator_evaluation_groups[0] = false;
        assert!(matches!(
            missing_numerator_group.validate(),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "quotient numerator group closure has missing or extra groups"
            ))
        ));
        let mut drifted_numerator_identity = retained_numerator.clone();
        drifted_numerator_identity
            .identity
            .numerator_evaluation_group_rounded_bytes += 4;
        assert!(matches!(
            drifted_numerator_identity.validate(),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "retained evaluation physical byte identity drifted"
            ))
        ));
        let mut invalid_oods_evaluation_log = protocol.clone();
        invalid_oods_evaluation_log.oods.columns[0].evaluation_log_size =
            invalid_oods_evaluation_log.oods.columns[0].coefficient_log_size;
        // Seal the mutated topology so validation reaches the domain relation
        // instead of rejecting the stale topology hash first.
        invalid_oods_evaluation_log.identity.oods_topology_hash =
            invalid_oods_evaluation_log.oods.topology_hash();
        assert_eq!(
            invalid_oods_evaluation_log.validate(),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS evaluation domain disagrees with coefficient log and PCS blowup"
            ))
        );
        let mut missing_fixed_tree = protocol.clone();
        missing_fixed_tree.commitments.remove(0);
        let missing_fixed_tree_result =
            ProofArenaPlan::build(&proof, &missing_fixed_tree, &composition);
        assert!(
            matches!(
                missing_fixed_tree_result,
                Err(ArenaPlanError::InvalidProtocolGeometry(
                    "commitments are not the canonical four Starknet trees"
                ))
            ),
            "unexpected missing-tree result: {missing_fixed_tree_result:?}"
        );
        let arena = ProofArenaPlan::build(&proof, &protocol, &composition).unwrap();
        let memory_ledger = crate::memory_ledger::PhysicalMemoryLedger::from_plan(&arena).unwrap();
        assert_eq!(
            memory_ledger.arena_allocation_bytes,
            arena.total_words() * core::mem::size_of::<u32>()
        );
        assert_eq!(memory_ledger.epochs.len(), ProofEpoch::ALL.len());
        arena.validate_aliases().unwrap();
        assert_eq!(
            arena.protocol_key,
            relation_protocol_key(protocol.key(), CAIRO_RELATION_LAUNCH_MODE)
        );

        let mut progressive = protocol.clone();
        progressive.identity.commit_mode = ProgressiveCommitMode::DomainProgressive;
        assert_ne!(protocol.key(), progressive.key());
        let progressive_arena = ProofArenaPlan::build(&proof, &progressive, &composition).unwrap();
        progressive_arena.validate_aliases().unwrap();
        assert_ne!(arena.protocol_key, progressive_arena.protocol_key);
        assert!(arena.commitments().iter().all(|commitment| matches!(
            (&commitment.requirements, &commitment.slots),
            (
                ModeAwareCommitWorkspaceRequirements::FullLifting(_),
                ModeAwareCommitWorkspaceSlots::FullLifting(_)
            )
        )));
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| matches!(
                    buffer.purpose,
                    BufferPurpose::CommitProgressiveStatePing
                        | BufferPurpose::CommitProgressiveStatePong
                ))
                .count(),
            0,
            "flags-off arena must not allocate progressive-only workspace"
        );
        assert!(progressive_arena
            .commitments()
            .iter()
            .all(|commitment| matches!(
                (&commitment.requirements, &commitment.slots),
                (
                    ModeAwareCommitWorkspaceRequirements::DomainProgressive(_),
                    ModeAwareCommitWorkspaceSlots::DomainProgressive(_)
                )
            )));
        assert_eq!(
            progressive_arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.purpose == BufferPurpose::CommitColumnPointers)
                .count(),
            0,
            "progressive arena must not allocate legacy leaf pointer groups"
        );
        assert!(progressive_arena
            .logical_buffers()
            .iter()
            .any(|buffer| { buffer.purpose == BufferPurpose::CommitProgressiveStatePing }));
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.purpose == BufferPurpose::MerkleLeafState)
                .count(),
            progressive_arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.purpose == BufferPurpose::MerkleLeafState)
                .count(),
            "both modes own exactly one common Merkle leaf layer per tree"
        );

        assert!(arena.composition().direct_bindings.is_empty());
        assert!(arena.commitments().iter().all(|commitment| commitment
            .direct_composition_evaluation_groups
            .iter()
            .all(Option::is_none)));
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.purpose == BufferPurpose::CommitRetainedEvaluation)
                .count(),
            0,
            "flags-off layout must not allocate direct-retention outputs"
        );

        let mut keyed = Vec::new();
        for mode in [
            DirectCompositionRetentionMode::Disabled,
            DirectCompositionRetentionMode::ExactNative,
        ] {
            for planner_key in [0, 1] {
                let mut candidate = protocol.clone();
                candidate.identity.direct_composition_retention_mode = mode;
                candidate.identity.direct_composition_planner_key = planner_key;
                keyed.push(candidate.key());
            }
        }
        assert_eq!(keyed.iter().copied().collect::<BTreeSet<_>>().len(), 4);

        let consumers = derive_direct_composition_consumers(
            &protocol.oods,
            &composition,
            protocol.identity.composition_launch_mode,
        )
        .unwrap();
        let direct_plan = crate::direct_composition_retention::plan_direct_composition_retention(
            &protocol, &consumers,
        )
        .unwrap();
        assert!(direct_plan.bindings.iter().any(|binding| binding.direct));
        let mut direct_protocol = protocol.clone();
        direct_protocol.identity.commit_mode = ProgressiveCommitMode::DomainProgressive;
        direct_protocol.identity.decommit_strategy = DecommitStrategy::HybridByGroup;
        direct_protocol.identity.direct_composition_retention_mode =
            DirectCompositionRetentionMode::ExactNative;
        direct_protocol.identity.direct_composition_planner_key = direct_plan.cache_key;
        direct_protocol
            .identity
            .direct_composition_occurrence_bitmap_hash =
            direct_bitmap_hash(&direct_plan.direct_bitmap);
        direct_protocol.direct_composition_retention = Some(direct_plan.clone());
        for binding in direct_plan.bindings.iter().filter(|binding| binding.direct) {
            let column = direct_plan.columns[binding.column];
            direct_protocol
                .commitments
                .iter_mut()
                .find(|commitment| commitment.id == column.tree)
                .unwrap()
                .direct_composition_evaluation_groups[column.group] = true;
        }
        let first_direct = direct_plan
            .bindings
            .iter()
            .find(|binding| binding.direct)
            .map(|binding| direct_plan.columns[binding.column])
            .unwrap();
        direct_protocol
            .commitments
            .iter_mut()
            .find(|commitment| commitment.id == first_direct.tree)
            .unwrap()
            .retained_evaluation_groups[first_direct.group] = true;
        direct_protocol
            .commitments
            .iter_mut()
            .find(|commitment| commitment.id == CommitmentTreeId::Composition)
            .unwrap()
            .retained_evaluation_groups[0] = true;
        let mut direct_bytes = 0usize;
        let mut union_bytes = 0usize;
        for commitment in &direct_protocol.commitments {
            for ((logs, &decommit), &direct) in commitment
                .grouped_column_log_sizes
                .iter()
                .zip(&commitment.retained_evaluation_groups)
                .zip(&commitment.direct_composition_evaluation_groups)
            {
                let bytes = logs
                    .iter()
                    .map(|log| 1usize << (log + commitment.config.log_blowup_factor))
                    .sum::<usize>()
                    * core::mem::size_of::<u32>();
                if direct {
                    direct_bytes += bytes;
                }
                if direct || decommit {
                    union_bytes += bytes;
                }
            }
        }
        direct_protocol
            .identity
            .direct_composition_group_rounded_bytes = direct_bytes;
        direct_protocol.identity.retained_evaluation_union_bytes = union_bytes;
        let mut changed_direct_bytes = direct_protocol.clone();
        changed_direct_bytes
            .identity
            .direct_composition_group_rounded_bytes += core::mem::size_of::<u32>();
        assert_ne!(direct_protocol.key(), changed_direct_bytes.key());
        assert!(matches!(
            ProofArenaPlan::build_inner(&proof, &changed_direct_bytes, &composition, None),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "retained evaluation physical byte identity drifted"
            ))
        ));
        let mut changed_union_bytes = direct_protocol.clone();
        changed_union_bytes.identity.retained_evaluation_union_bytes += core::mem::size_of::<u32>();
        assert!(matches!(
            ProofArenaPlan::build_inner(&proof, &changed_union_bytes, &composition, None),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "retained evaluation physical byte identity drifted"
            ))
        ));
        let mut missing_direct_group = direct_protocol.clone();
        missing_direct_group
            .commitments
            .iter_mut()
            .find(|commitment| commitment.id == first_direct.tree)
            .unwrap()
            .direct_composition_evaluation_groups[first_direct.group] = false;
        assert!(matches!(
            ProofArenaPlan::build_inner(&proof, &missing_direct_group, &composition, None),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "direct composition group closure has missing or extra groups"
            ))
        ));
        let (extra_tree, extra_group) = direct_protocol
            .commitments
            .iter()
            .find_map(|commitment| {
                commitment
                    .direct_composition_evaluation_groups
                    .iter()
                    .position(|&selected| !selected)
                    .map(|group| (commitment.id, group))
            })
            .unwrap();
        let mut extra_direct_group = direct_protocol.clone();
        extra_direct_group
            .commitments
            .iter_mut()
            .find(|commitment| commitment.id == extra_tree)
            .unwrap()
            .direct_composition_evaluation_groups[extra_group] = true;
        assert!(matches!(
            ProofArenaPlan::build_inner(&proof, &extra_direct_group, &composition, None),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "direct composition group closure has missing or extra groups"
            ))
        ));
        let mut changed_bitmap_hash = direct_protocol.clone();
        changed_bitmap_hash
            .identity
            .direct_composition_occurrence_bitmap_hash ^= 1;
        assert!(matches!(
            ProofArenaPlan::build_inner(&proof, &changed_bitmap_hash, &composition, None),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "direct composition occurrence bitmap identity drifted"
            ))
        ));
        let direct_arena = ProofArenaPlan::build(&proof, &direct_protocol, &composition).unwrap();
        direct_arena.validate_aliases().unwrap();

        let mut combined_protocol = direct_protocol.clone();
        combined_protocol.identity.quotient_numerator_source_policy =
            QuotientNumeratorSourcePolicy::ReuseRetainedEvaluations;
        let combined_composition = combined_protocol
            .commitments
            .iter_mut()
            .find(|commitment| commitment.id == CommitmentTreeId::Composition)
            .unwrap();
        combined_composition.numerator_evaluation_groups[0] = true;
        combined_protocol
            .identity
            .numerator_evaluation_group_rounded_bytes =
            commitment_group_evaluation_bytes(combined_composition, 0).unwrap();
        assert_eq!(
            combined_protocol.transcript, protocol.transcript,
            "retention policies must not move a Fiat-Shamir boundary"
        );
        combined_protocol.validate().unwrap();
        let combined_arena =
            ProofArenaPlan::build(&proof, &combined_protocol, &composition).unwrap();
        combined_arena.validate_aliases().unwrap();
        assert_eq!(
            combined_arena
                .composition()
                .requirements
                .direct_retention_plan_key,
            Some(direct_plan.cache_key),
            "normalized composition requirements must rebound the exact direct plan"
        );
        let combined_commitment = combined_arena
            .commitment(CommitmentTreeId::Composition)
            .unwrap();
        assert_eq!(
            combined_commitment.numerator_evaluation_groups[0],
            combined_commitment.evaluation_output_groups[0]
        );
        for binding in combined_commitment.numerator_evaluation_groups[0]
            .as_ref()
            .unwrap()
        {
            assert_eq!(
                combined_arena.logical_buffers()[binding.logical.0 as usize]
                    .lifetime
                    .last,
                ProofEpoch::Decommit,
                "hybrid decommit plus numerator union must retain through decommit"
            );
        }
        assert_eq!(
            combined_protocol.identity.retained_evaluation_union_bytes,
            direct_protocol.identity.retained_evaluation_union_bytes,
            "adding numerator intent to an already-retained group must not double-count the union"
        );
        assert_eq!(
            direct_arena.composition().direct_retention.as_ref(),
            Some(&direct_plan),
            "resident preparation must receive the exact protocol retention plan"
        );
        assert_eq!(
            direct_arena
                .logical_buffers()
                .iter()
                .find(|buffer| buffer.purpose == BufferPurpose::CompositionDescriptors)
                .unwrap()
                .len_words,
            direct_arena.composition().requirements.descriptor_words,
            "the arena must allocate the exact retention-aware descriptor extent"
        );
        assert!(
            direct_arena.composition().requirements.lde_tile_words
                < progressive_arena.composition().requirements.lde_tile_words,
            "direct sources must shrink the composition fallback LDE workspace"
        );
        assert_eq!(
            direct_arena.composition().direct_bindings.len(),
            direct_plan.bindings.len()
        );
        for (planned, oracle) in direct_arena
            .composition()
            .direct_bindings
            .iter()
            .zip(&direct_plan.bindings)
        {
            let column = direct_plan.columns[oracle.column];
            assert_eq!(
                (
                    planned.consumer,
                    planned.plan_column,
                    planned.source,
                    planned.tree,
                    planned.proof_column,
                    planned.group,
                    planned.column_in_group,
                    planned.canonical_column,
                ),
                (
                    oracle.consumer,
                    oracle.column,
                    column.source,
                    column.tree,
                    column.proof_column,
                    column.group,
                    column.column_in_group,
                    column.canonical_column,
                )
            );
            assert_eq!(planned.evaluation.is_some(), oracle.direct);
            if let Some(evaluation) = planned.evaluation {
                let commitment = direct_arena.commitment(planned.tree).unwrap();
                assert_eq!(
                    commitment.evaluation_output_groups[planned.group]
                        .as_ref()
                        .unwrap()[planned.column_in_group],
                    evaluation,
                    "composition must bind the exact progressive producer output"
                );
                let lifetime =
                    direct_arena.logical_buffers()[evaluation.logical.0 as usize].lifetime;
                assert!(lifetime.contains(ProofEpoch::Composition));
                assert!(
                    direct_protocol
                        .commitments
                        .iter()
                        .find(|commitment| commitment.id == planned.tree)
                        .unwrap()
                        .created
                        < ProofEpoch::Composition,
                    "every direct evaluation producer must precede composition capture/replay"
                );
            }
        }
        let both = direct_arena
            .composition()
            .direct_bindings
            .iter()
            .find(|binding| {
                binding.tree == first_direct.tree
                    && binding.group == first_direct.group
                    && binding.evaluation.is_some()
            })
            .unwrap()
            .evaluation
            .unwrap();
        assert_eq!(
            direct_arena.logical_buffers()[both.logical.0 as usize]
                .lifetime
                .last,
            if first_direct.tree == CommitmentTreeId::Preprocessed {
                ProofEpoch::Assemble
            } else {
                ProofEpoch::Decommit
            },
            "cached fixed outputs must persist; dynamic decommit outputs end at decommit"
        );
        let mut direct_only_protocol = direct_protocol.clone();
        direct_only_protocol
            .commitments
            .iter_mut()
            .find(|commitment| commitment.id == first_direct.tree)
            .unwrap()
            .retained_evaluation_groups[first_direct.group] = false;
        let direct_only_arena =
            ProofArenaPlan::build_inner(&proof, &direct_only_protocol, &composition, None).unwrap();
        let direct_only = direct_only_arena
            .composition()
            .direct_bindings
            .iter()
            .find(|binding| {
                binding.evaluation.is_some()
                    && binding.tree == first_direct.tree
                    && binding.group == first_direct.group
            })
            .unwrap()
            .evaluation
            .unwrap();
        assert_eq!(
            direct_only_arena.logical_buffers()[direct_only.logical.0 as usize]
                .lifetime
                .last,
            if first_direct.tree == CommitmentTreeId::Preprocessed {
                ProofEpoch::Assemble
            } else {
                ProofEpoch::Composition
            },
            "cached fixed outputs must persist; dynamic direct-only outputs end at composition"
        );
        let composition_commitment = direct_arena
            .commitment(CommitmentTreeId::Composition)
            .unwrap();
        assert!(composition_commitment.direct_composition_evaluation_groups[0].is_none());
        let decommit_only = composition_commitment.evaluation_output_groups[0]
            .as_ref()
            .unwrap()[0];
        assert_eq!(
            direct_arena.logical_buffers()[decommit_only.logical.0 as usize]
                .lifetime
                .last,
            ProofEpoch::Decommit,
            "decommit-only intent must retain through decommit"
        );

        let mut drifted_direct = direct_protocol.clone();
        drifted_direct
            .direct_composition_retention
            .as_mut()
            .unwrap()
            .direct_bitmap[0] ^= 1;
        assert!(matches!(
            ProofArenaPlan::build_inner(&proof, &drifted_direct, &composition, None),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "direct composition occurrence bitmap identity drifted"
            ))
        ));

        let retained_numerator_arena =
            ProofArenaPlan::build(&proof, &retained_numerator, &composition).unwrap();
        let numerator_only_commitment = retained_numerator_arena
            .commitment(CommitmentTreeId::Composition)
            .unwrap();
        assert!(numerator_only_commitment.retained_evaluation_groups[0].is_none());
        let numerator_only_group = numerator_only_commitment.numerator_evaluation_groups[0]
            .as_ref()
            .unwrap();
        let numerator_only_oods_columns = retained_numerator_arena
            .oods()
            .columns
            .iter()
            .filter_map(|column| match column.source {
                OpenedColumnSource::Composition { ordinal } => Some((ordinal, column)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(numerator_only_oods_columns.len(), 8);
        for (ordinal, column) in numerator_only_oods_columns {
            assert_eq!(column.source_kind, OodsSourceKind::Evaluations);
            assert_eq!(
                column.source_binding,
                numerator_only_group[usize::try_from(ordinal).unwrap()],
                "numerator-retained evaluations already alive at OODS must be reused exactly"
            );
        }
        assert!(retained_numerator_arena
            .logical_buffers()
            .iter()
            .filter(|buffer| buffer.purpose == BufferPurpose::CompositionCoefficients)
            .all(|buffer| buffer.lifetime.last == ProofEpoch::Decommit));
        assert!(retained_numerator_arena
            .late_coefficient_ownership()
            .entries()
            .iter()
            .filter(|ownership| matches!(ownership.source, OpenedColumnSource::Composition { .. }))
            .all(|ownership| {
                !ownership.composition_reads_coefficients
                    && !ownership.oods_reads_coefficients
                    && !ownership.quotient_reads_coefficients
                    && ownership.decommit_reads_coefficients
                    && ownership.final_consumer == ProofEpoch::Decommit
            }));
        let numerator_columns = &retained_numerator_arena.quotient_numerator().columns;
        let evaluation_column = numerator_columns
            .iter()
            .find(|column| column.topology.source_kind == QuotientNumeratorSourceKind::Evaluation)
            .unwrap();
        assert_eq!(
            evaluation_column.source,
            OpenedColumnSource::Composition { ordinal: 0 }
        );
        assert_eq!(evaluation_column.numerator_source.len_words, 1 << 26);
        assert_ne!(
            evaluation_column.numerator_source,
            evaluation_column.coefficients
        );
        assert!(numerator_columns.iter().all(|column| {
            column.topology.source_kind == QuotientNumeratorSourceKind::Evaluation
                || column.numerator_source == column.coefficients
        }));
        let retained_source_buffer = retained_numerator_arena
            .logical_buffers()
            .iter()
            .find(|buffer| buffer.id == evaluation_column.numerator_source.logical)
            .unwrap();
        assert_eq!(
            retained_source_buffer.purpose,
            BufferPurpose::CommitRetainedEvaluation
        );
        assert_eq!(retained_source_buffer.lifetime.last, ProofEpoch::Quotient);
        let mut exclusive_late = retained_numerator.clone();
        exclusive_late.identity.decommit_strategy = DecommitStrategy::HybridByGroup;
        exclusive_late
            .commitments
            .iter_mut()
            .find(|commitment| commitment.id == CommitmentTreeId::Composition)
            .unwrap()
            .retained_evaluation_groups[0] = true;
        let exclusive_late_arena =
            ProofArenaPlan::build(&proof, &exclusive_late, &composition).unwrap();
        assert!(exclusive_late_arena
            .logical_buffers()
            .iter()
            .filter(|buffer| buffer.purpose == BufferPurpose::CompositionCoefficients)
            .all(|buffer| buffer.lifetime.last == ProofEpoch::CompositionCommit));
        let exclusive_commitment = exclusive_late_arena
            .commitment(CommitmentTreeId::Composition)
            .unwrap();
        let retained_composition = exclusive_commitment.retained_evaluation_groups[0]
            .as_ref()
            .unwrap();
        let evaluation_oods_columns = exclusive_late_arena
            .oods()
            .columns
            .iter()
            .filter_map(|column| match column.source {
                OpenedColumnSource::Composition { ordinal } => Some((ordinal, column)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(evaluation_oods_columns.len(), 8);
        for (ordinal, column) in evaluation_oods_columns {
            assert_eq!(column.source_kind, OodsSourceKind::Evaluations);
            assert_eq!(column.coefficient_log_size, 25);
            assert_eq!(column.topology().log_size, 26);
            assert_eq!(
                column.source_binding,
                retained_composition[usize::try_from(ordinal).unwrap()]
            );
        }
        assert!(exclusive_late_arena
            .late_coefficient_ownership()
            .entries()
            .iter()
            .filter(|ownership| matches!(ownership.source, OpenedColumnSource::Composition { .. }))
            .all(|ownership| {
                !ownership.composition_reads_coefficients
                    && !ownership.oods_reads_coefficients
                    && !ownership.quotient_reads_coefficients
                    && !ownership.decommit_reads_coefficients
                    && ownership.final_consumer == ProofEpoch::CompositionCommit
            }));
        assert!(retained_composition.iter().all(|binding| {
            exclusive_late_arena.logical_buffers()[binding.logical.0 as usize]
                .lifetime
                .last
                == ProofEpoch::Decommit
        }));
        let sampled_composition = exclusive_late_arena
            .oods()
            .columns
            .iter()
            .find(|column| column.source == OpenedColumnSource::Composition { ordinal: 0 })
            .unwrap();
        assert_eq!(
            sampled_composition.source_binding,
            exclusive_late_arena
                .quotient_numerator()
                .columns
                .iter()
                .find(|column| column.source == sampled_composition.source)
                .unwrap()
                .numerator_source
        );
        let mut short_oods_source = sampled_composition.clone();
        short_oods_source.source_binding.len_words -= 1;
        assert_eq!(
            validate_oods_source_binding(&short_oods_source),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "OODS source binding has the wrong representation extent"
            ))
        );
        let mut short_source = evaluation_column.numerator_source;
        short_source.len_words -= 1;
        assert_eq!(
            validate_quotient_numerator_source_binding(
                retained_numerator_arena.quotient_numerator().config,
                &evaluation_column.topology,
                evaluation_column.coefficients,
                short_source,
            ),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "quotient numerator source binding has the wrong kind or extent"
            ))
        );
        let coefficient_column = numerator_columns
            .iter()
            .find(|column| column.topology.source_kind == QuotientNumeratorSourceKind::Coefficients)
            .unwrap();
        assert_eq!(
            validate_quotient_numerator_source_binding(
                retained_numerator_arena.quotient_numerator().config,
                &coefficient_column.topology,
                coefficient_column.coefficients,
                evaluation_column.numerator_source,
            ),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "quotient numerator source binding has the wrong kind or extent"
            ))
        );
        let source_alias = BTreeSet::from([evaluation_column.numerator_source.physical]);
        assert!(quotient_numerator_sources_alias_workspace(
            std::slice::from_ref(evaluation_column),
            &source_alias
        ));
        let coefficient_source_alias = BTreeSet::from([coefficient_column.coefficients.physical]);
        assert!(quotient_numerator_sources_alias_workspace(
            std::slice::from_ref(coefficient_column),
            &coefficient_source_alias
        ));
        let retired_coefficient_reuse = BTreeSet::from([evaluation_column.coefficients.physical]);
        assert!(!quotient_numerator_sources_alias_workspace(
            std::slice::from_ref(evaluation_column),
            &retired_coefficient_reuse
        ));

        let mut direct_only_source_policy = protocol.clone();
        direct_only_source_policy.commitments[1].direct_composition_evaluation_groups[0] = true;
        let direct_only_source_kinds = direct_only_source_policy.oods_source_kinds().unwrap();
        for &source in &direct_only_source_policy.commitments[1].grouped_column_sources[0] {
            let source = OpenedColumnSource::from(source);
            let proof_column = direct_only_source_policy
                .oods
                .columns
                .iter()
                .position(|column| column.source == source)
                .unwrap();
            assert_eq!(
                direct_only_source_kinds[proof_column],
                OodsSourceKind::Coefficients,
                "a direct-only evaluation expires before OODS and is not a valid source"
            );
        }

        let mut hybrid = protocol.clone();
        hybrid.identity.decommit_strategy = DecommitStrategy::HybridByGroup;
        hybrid.commitments[1].retained_evaluation_groups[0] = true;
        let hybrid_arena = ProofArenaPlan::build(&proof, &hybrid, &composition).unwrap();
        let hybrid_config = hybrid.decommit_workspace_config().unwrap();
        let DecommitTreeGeometry::Trace(base_decommit) = &hybrid_config.trees[1] else {
            panic!("base trace")
        };
        assert_eq!(
            base_decommit.groups[0].mode,
            DecommitSourceMode::ResidentEvaluations
        );
        assert_eq!(
            base_decommit.groups[1].mode,
            DecommitSourceMode::RecomputeQueriedLde
        );
        let hybrid_base = hybrid_arena.commitment(CommitmentTreeId::Base).unwrap();
        assert!(hybrid_base.retained_evaluation_groups[0].is_some());
        assert!(hybrid_base.retained_evaluation_groups[1].is_none());
        assert_eq!(
            hybrid_arena
                .oods()
                .columns
                .iter()
                .map(|column| column.source)
                .collect::<Vec<_>>(),
            hybrid
                .oods
                .columns
                .iter()
                .map(|column| column.source)
                .collect::<Vec<_>>(),
            "source grouping must not perturb canonical proof order"
        );
        let hybrid_base_geometry = hybrid
            .commitments
            .iter()
            .find(|commitment| commitment.id == CommitmentTreeId::Base)
            .unwrap();
        let retained_base_group = hybrid_base.retained_evaluation_groups[0].as_ref().unwrap();
        for (column_in_group, &source) in hybrid_base_geometry.grouped_column_sources[0]
            .iter()
            .enumerate()
        {
            let source = OpenedColumnSource::from(source);
            let planned = hybrid_arena
                .oods()
                .columns
                .iter()
                .find(|column| column.source == source)
                .unwrap();
            assert_eq!(planned.source_kind, OodsSourceKind::Evaluations);
            assert_eq!(
                planned.source_binding, retained_base_group[column_in_group],
                "a retained group must bind the exact commitment output column"
            );
            let ownership = hybrid_arena
                .late_coefficient_ownership()
                .get(source)
                .unwrap();
            assert!(ownership.composition_reads_coefficients);
            assert!(!ownership.oods_reads_coefficients);
            assert!(!ownership.quotient_reads_coefficients);
            assert!(!ownership.decommit_reads_coefficients);
            assert_eq!(ownership.final_consumer, ProofEpoch::Composition);
            let OpenedColumnSource::Trace {
                component,
                part,
                purpose,
                ordinal,
            } = source
            else {
                panic!("base commitment contains a non-trace source")
            };
            assert_eq!(
                hybrid_arena
                    .find(Some(component), Some(part), purpose, ordinal)
                    .unwrap()
                    .0
                    .lifetime
                    .last,
                ProofEpoch::Composition,
                "retained evaluation must let the coefficient die after its last real reader"
            );
        }
        for &source in hybrid_base_geometry
            .grouped_column_sources
            .iter()
            .skip(1)
            .flatten()
        {
            let source = OpenedColumnSource::from(source);
            let planned = hybrid_arena
                .oods()
                .columns
                .iter()
                .find(|column| column.source == source)
                .unwrap();
            assert_eq!(planned.source_kind, OodsSourceKind::Coefficients);
            let OpenedColumnSource::Trace {
                component,
                part,
                purpose,
                ordinal,
            } = source
            else {
                panic!("base commitment contains a non-trace source")
            };
            let (coefficient, coefficient_binding) = hybrid_arena
                .find(Some(component), Some(part), purpose, ordinal)
                .unwrap();
            assert_eq!(planned.source_binding, coefficient_binding);
            assert_eq!(coefficient.lifetime.last, ProofEpoch::Decommit);
            let ownership = hybrid_arena
                .late_coefficient_ownership()
                .get(source)
                .unwrap();
            assert!(ownership.composition_reads_coefficients);
            assert!(!ownership.oods_reads_coefficients);
            assert!(!ownership.quotient_reads_coefficients);
            assert!(ownership.decommit_reads_coefficients);
            assert_eq!(ownership.final_consumer, ProofEpoch::Decommit);
        }
        validate_decommit_group_bindings(&hybrid_config, &hybrid_arena.commitments).unwrap();

        let mut missing_retained_binding = hybrid_arena.commitments.clone();
        missing_retained_binding[1].retained_evaluation_groups[0] = None;
        assert!(matches!(
            validate_decommit_group_bindings(&hybrid_config, &missing_retained_binding),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "decommit source mode disagrees with retained evaluation bindings"
            ))
        ));
        let mut wrong_mode = hybrid_config.clone();
        let DecommitTreeGeometry::Trace(base) = &mut wrong_mode.trees[1] else {
            panic!("base trace")
        };
        base.groups[1].mode = DecommitSourceMode::ResidentEvaluations;
        assert!(matches!(
            validate_decommit_group_bindings(&wrong_mode, &hybrid_arena.commitments),
            Err(ArenaPlanError::InvalidProtocolGeometry(
                "decommit source mode disagrees with retained evaluation bindings"
            ))
        ));
        let mut wrong_binding_width = hybrid_arena.commitments.clone();
        wrong_binding_width[1].retained_evaluation_groups[0]
            .as_mut()
            .unwrap()
            .pop();
        assert!(validate_decommit_group_bindings(&hybrid_config, &wrong_binding_width).is_err());

        let fused_arena =
            ProofArenaPlan::build(&proof, &fused_interpolation, &composition).unwrap();
        let fused_base = fused_arena.commitment(CommitmentTreeId::Base).unwrap();
        let fused_retained_base = fused_arena
            .relation()
            .source_plan
            .iter()
            .filter(|source| source.plane == RelationSourcePlane::BaseTrace)
            .flat_map(|source| {
                (0..source.column_count)
                    .map(move |ordinal| (source.batch.component, source.part, ordinal))
            })
            .collect::<HashSet<_>>();
        assert!(!fused_retained_base.is_empty());
        for evaluation in fused_arena.logical_buffers().iter().filter(|buffer| {
            matches!(
                buffer.purpose,
                BufferPurpose::BaseTrace | BufferPurpose::InteractionTrace
            )
        }) {
            let coefficient_purpose = match evaluation.purpose {
                BufferPurpose::BaseTrace => BufferPurpose::BaseCoefficients,
                BufferPurpose::InteractionTrace => BufferPurpose::InteractionCoefficients,
                _ => unreachable!(),
            };
            let coefficient = fused_arena
                .find(
                    evaluation.component,
                    evaluation.part,
                    coefficient_purpose,
                    evaluation.ordinal,
                )
                .expect("every evaluation has one coefficient destination")
                .1;
            let should_alias = match evaluation.purpose {
                BufferPurpose::BaseTrace => !fused_retained_base.contains(&(
                    evaluation.component.unwrap(),
                    evaluation.part.unwrap(),
                    evaluation.ordinal,
                )),
                BufferPurpose::InteractionTrace => true,
                _ => unreachable!(),
            };
            assert_eq!(
                fused_arena.binding(evaluation.id).unwrap().physical == coefficient.physical,
                should_alias,
                "fused interpolation transition alias drifted from evaluation liveness"
            );
        }
        let distinct_base_logs = fused_interpolation
            .commitments
            .iter()
            .find(|commitment| commitment.id == CommitmentTreeId::Base)
            .unwrap()
            .grouped_column_log_sizes
            .iter()
            .flatten()
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(
            fused_base.interpolation_batches.len(),
            distinct_base_logs.len()
        );
        assert!(fused_base
            .interpolation_batches
            .windows(2)
            .all(|pair| pair[0].log_size < pair[1].log_size));
        let base_geometry = fused_interpolation
            .commitments
            .iter()
            .find(|commitment| commitment.id == CommitmentTreeId::Base)
            .unwrap();
        let mut expected_by_log = BTreeMap::<u32, Vec<CommitmentColumnSource>>::new();
        for (sources, logs) in base_geometry
            .grouped_column_sources
            .iter()
            .zip(&base_geometry.grouped_column_log_sizes)
        {
            for (&source, &log_size) in sources.iter().zip(logs) {
                expected_by_log.entry(log_size).or_default().push(source);
            }
        }
        assert_eq!(
            fused_base
                .interpolation_batches
                .iter()
                .map(|batch| (batch.log_size, batch.sources.clone()))
                .collect::<Vec<_>>(),
            expected_by_log.into_iter().collect::<Vec<_>>()
        );
        assert!(arena.execution_tables().is_none());
        let resident_arena = ProofArenaPlan::build_with_execution_tables(
            &proof,
            &protocol,
            &composition,
            ExecutionTableGeometry::new(19, 17, 5),
        )
        .unwrap();
        let execution_tables = resident_arena.execution_tables().unwrap();
        assert_eq!(
            (
                execution_tables.requirements.n_addrs,
                execution_tables.requirements.n_big,
                execution_tables.requirements.n_small,
            ),
            (19, 17, 5)
        );
        assert_ne!(resident_arena.protocol_key, arena.protocol_key);
        let changed_execution_shape = ProofArenaPlan::build_with_execution_tables(
            &proof,
            &protocol,
            &composition,
            ExecutionTableGeometry::new(20, 17, 5),
        )
        .unwrap();
        assert_ne!(
            resident_arena.protocol_key,
            changed_execution_shape.protocol_key
        );
        let public_seed_arena = ProofArenaPlan::build_with_execution_tables(
            &proof,
            &protocol,
            &composition,
            ExecutionTableGeometry::new(19, 17, 5).with_public_memory_entries(3),
        )
        .unwrap();
        assert_ne!(resident_arena.protocol_key, public_seed_arena.protocol_key);
        let public_seed = public_seed_arena
            .multiplicity()
            .unwrap()
            .public_memory_seed
            .as_ref()
            .unwrap();
        assert_eq!(public_seed.plan.row_count, 3);
        assert_eq!(public_seed.source.len_words, 6);
        assert_eq!(arena.shape_key, proof.shape_key);
        assert_eq!(arena.commitments().len(), 4);
        assert_eq!(arena.composition().plan, composition);
        assert_eq!(arena.composition().ext_params.len(), 1);
        assert!(arena.composition().ext_params[0].binding.is_some());
        let base = arena.commitment(CommitmentTreeId::Base).unwrap();
        let interaction = arena.commitment(CommitmentTreeId::Interaction).unwrap();
        let preprocessed = arena.commitment(CommitmentTreeId::Preprocessed).unwrap();
        let base_slots = match &base.slots {
            ModeAwareCommitWorkspaceSlots::FullLifting(slots) => slots,
            ModeAwareCommitWorkspaceSlots::DomainProgressive(_) => {
                panic!("default arena unexpectedly uses progressive commit slots")
            }
        };
        let interaction_slots = match &interaction.slots {
            ModeAwareCommitWorkspaceSlots::FullLifting(slots) => slots,
            ModeAwareCommitWorkspaceSlots::DomainProgressive(_) => {
                panic!("default arena unexpectedly uses progressive commit slots")
            }
        };
        assert_eq!(preprocessed.root.len_words, BLAKE2S_HASH_WORDS);
        assert_eq!(
            preprocessed.root,
            *preprocessed.retained_layers_bottom_up.last().unwrap(),
            "fixed tree root must stay bound through decommit"
        );
        assert_ne!(
            base_slots.groups[0].column_ptrs, interaction_slots.groups[0].column_ptrs,
            "captured commitment descriptor tables must persist independently"
        );
        assert_eq!(arena.fri().requirements.trees.len(), 25);
        assert_eq!(
            arena.fri().slots.retained_tree_evaluations.len(),
            arena.fri().requirements.trees.len() - 1
        );
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.purpose == BufferPurpose::FriRetainedEvaluation)
                .map(|buffer| buffer.len_words)
                .sum::<usize>(),
            arena
                .fri()
                .requirements
                .trees
                .iter()
                .skip(1)
                .map(|tree| tree.evaluation_words)
                .sum::<usize>(),
            "the CPU plan must retain every committed inner FRI codeword exactly once"
        );
        assert!(arena.logical_buffers().iter().all(|buffer| {
            buffer.purpose != BufferPurpose::FriRetainedEvaluation
                || buffer.lifetime
                    == BufferLifetime::new(ProofEpoch::Fri, ProofEpoch::Decommit).unwrap()
        }));
        assert!(arena.logical_buffers().iter().all(|buffer| {
            !matches!(
                buffer.purpose,
                BufferPurpose::FriPing | BufferPurpose::FriPong
            ) || buffer.lifetime == BufferLifetime::at(ProofEpoch::Fri)
        }));
        assert_eq!(arena.decommit().requirements.trees.len(), 29);
        assert_eq!(arena.decommit().proof_shape.trace_trees.len(), 4);
        assert_eq!(arena.decommit().proof_shape.fri_trees.len(), 25);
        let decommitment = arena
            .decommit()
            .proof_bundle_layout
            .direct_decommitment_destination();
        assert_eq!(
            arena.decommit().slots.assembly,
            arena.decommit().proof_bundle.physical,
            "the decommit producer must own the proof bundle's physical identity"
        );
        assert_eq!(
            decommitment.len_words,
            arena.decommit().requirements.assembly_words
        );
        assert_eq!(
            decommitment
                .offset_words
                .checked_add(decommitment.len_words),
            Some(arena.decommit().proof_bundle_layout.total_words),
            "decommitment must remain the canonical final proof-bundle range"
        );
        let proof_bundle = arena
            .logical_buffers()
            .iter()
            .find(|buffer| buffer.purpose == BufferPurpose::ProofBytes)
            .unwrap();
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.purpose == BufferPurpose::ProofBytes)
                .count(),
            1,
            "one proof bundle must be the sole final assembly owner"
        );
        assert_eq!(
            proof_bundle.lifetime,
            BufferLifetime::new(ProofEpoch::Decommit, ProofEpoch::Assemble).unwrap()
        );
        assert_eq!(
            proof_bundle.len_words,
            arena.decommit().proof_bundle_layout.total_words
        );
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.purpose == BufferPurpose::DecommitTraceLdeTile)
                .count(),
            1,
            "serial trace opens must share one max-sized queried-LDE tile"
        );
        assert_eq!(
            arena.decommit().raw_queries,
            arena
                .transcript()
                .outputs
                .iter()
                .find(|(id, _)| *id == TranscriptOutputId(5))
                .unwrap()
                .1,
            "decommit must consume device transcript queries without a copy"
        );
        let mut proof_order_protocol = protocol.clone();
        let base_columns = proof_order_protocol.commitments[1]
            .grouped_column_sources
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        let preprocessed_columns = proof_order_protocol.commitments[0]
            .grouped_column_sources
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        proof_order_protocol.oods.columns.swap(
            preprocessed_columns,
            preprocessed_columns + base_columns - 1,
        );
        let proof_order_shape = proof_order_protocol.proof_assembly_shape().unwrap();
        assert_ne!(
            proof_order_shape.trace_trees[1].commit_to_proof_column,
            (0..base_columns).collect::<Vec<_>>(),
            "mixed-log Merkle order must retain an explicit mapping back to PCS proof order"
        );
        let mut permutation = proof_order_shape.trace_trees[1]
            .commit_to_proof_column
            .clone();
        permutation.sort_unstable();
        assert_eq!(permutation, (0..base_columns).collect::<Vec<_>>());
        assert_eq!(arena.quotient().requirements.sample_count, 3);
        let quotient_pass_bytes = arena.quotient().requirements.combine_pass_bytes;
        assert_eq!(quotient_pass_bytes.rows, 1 << 25);
        assert_eq!(quotient_pass_bytes.samples, 3);
        assert_eq!(quotient_pass_bytes.denominator_inversions, 100_663_296);
        assert_eq!(quotient_pass_bytes.eliminated_scratch_bytes, 805_306_368);
        assert_eq!(
            quotient_pass_bytes.eliminated_logical_traffic_bytes,
            1_610_612_736
        );
        assert_eq!(quotient_pass_bytes.denominator_global_passes, 0);
        assert_eq!(quotient_pass_bytes.output_write_bytes, 536_870_912);
        assert_eq!(arena.quotient().partial_numerators.len(), 3);
        assert_eq!(arena.preprocessed_coefficients().len(), 3);
        assert_eq!(arena.preprocessed().interpolation_batches.len(), 2);
        let preprocessed_inverse = arena
            .find(None, None, BufferPurpose::PreprocessedInverseTwiddles, 0)
            .unwrap()
            .1;
        let fri_inverse = arena
            .find(None, None, BufferPurpose::InverseTwiddles, 0)
            .unwrap()
            .1;
        assert_eq!(arena.preprocessed().inverse_twiddles, preprocessed_inverse);
        assert_ne!(
            preprocessed_inverse.physical, fri_inverse.physical,
            "preprocessed and FRI inverse-twiddle oracles must remain distinct"
        );
        let pointer_buffer = arena
            .logical_buffers()
            .iter()
            .find(|buffer| buffer.purpose == BufferPurpose::PreprocessedInterpolationPointers)
            .unwrap();
        assert_eq!(
            pointer_buffer.lifetime,
            BufferLifetime::at(ProofEpoch::Ingest)
        );
        assert_eq!(
            arena.preprocessed_coefficients()[0].coefficients,
            arena
                .find(None, None, BufferPurpose::PreprocessedCoefficients, 0,)
                .unwrap()
                .1
        );
        assert_eq!(arena.oods().requirements.sample_count, 3);
        assert_eq!(
            arena.oods().sampled_values,
            arena
                .transcript()
                .inputs
                .iter()
                .find(|(id, _)| *id == TranscriptInputId(25))
                .unwrap()
                .1,
            "OODS values must write directly into the transcript input"
        );
        assert_eq!(
            arena.quotient_numerator().oods_sample_points,
            arena.oods().sample_points
        );
        assert_eq!(
            arena.quotient_numerator().sample_points_destination,
            arena.quotient().sample_points
        );
        assert_eq!(
            arena.quotient_numerator().first_linear_terms_destination,
            arena.quotient().first_linear_terms
        );
        assert_eq!(
            arena.quotient_numerator().destinations,
            arena.quotient().partial_numerators
        );
        let address_ranges_for = |purposes: &[BufferPurpose]| {
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| purposes.contains(&buffer.purpose))
                .map(|buffer| {
                    let binding = arena.binding(buffer.id).unwrap();
                    let slot = arena.layout().slot(binding.physical).unwrap();
                    (
                        binding.physical,
                        buffer.lifetime,
                        slot.offset_words,
                        slot.offset_words + binding.len_words,
                    )
                })
                .collect::<Vec<_>>()
        };
        let oods_ephemeral = address_ranges_for(&[
            BufferPurpose::OodsFoldingFactors,
            BufferPurpose::OodsScratchA,
            BufferPurpose::OodsScratchB,
            BufferPurpose::OodsEvaluationPoints,
            BufferPurpose::OodsBarycentricNumerators,
            BufferPurpose::OodsBarycentricWeights,
            BufferPurpose::OodsBarycentricScales,
            BufferPurpose::OodsBarycentricPartials,
        ]);
        let numerator_ephemeral = address_ranges_for(&[
            BufferPurpose::QuotientNumeratorLineCoefficients,
            BufferPurpose::QuotientNumeratorTermPoints,
            BufferPurpose::QuotientNumeratorLdeTile,
        ]);
        // OODS and numerator preparation are sequential. Keep the semantic
        // liveness fact explicit, but do not pin a particular physical alias:
        // retiring a large scratch slab can legitimately change the optimal
        // range packing chosen for every smaller buffer around it.
        assert!(oods_ephemeral.iter().all(|oods| numerator_ephemeral
            .iter()
            .all(|numerator| !oods.1.overlaps(numerator.1))));
        assert!(
            !numerator_ephemeral.is_empty(),
            "numerator scratch must be planned"
        );
        assert_eq!(
            arena.quotient().output_values,
            arena.fri().input_values,
            "prepared quotient output must be the exact contiguous FRI input binding"
        );
        assert_eq!(
            arena.quotient().output_values,
            arena
                .find(None, None, BufferPurpose::QuotientTile, 0)
                .expect("quotient output keeps the canonical tile identity")
                .1
        );
        assert_eq!(
            arena.quotient().output_values.len_words,
            arena.quotient().requirements.output_value_words
        );
        assert_ne!(
            arena.quotient().forward_twiddles.physical,
            arena.quotient().inverse_subdomain_twiddles.physical
        );
        assert_ne!(
            arena.quotient().inverse_subdomain_twiddles.physical,
            arena.fri().twiddles.physical
        );
        assert_ne!(
            arena.quotient().forward_twiddles.physical,
            arena.fri().twiddles.physical
        );
        let quotient_slots = &arena.quotient().slots;
        let quotient_workspace_ids = [
            quotient_slots.sample_points,
            quotient_slots.first_linear_terms,
            quotient_slots.partial_log_sizes,
            quotient_slots.partial_coordinate_ptrs,
            quotient_slots.subdomain_coordinate_ptrs,
            quotient_slots.output_coordinate_ptrs,
            quotient_slots.coefficient_sizes,
            quotient_slots.subdomain_values,
            quotient_slots.output_values,
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(quotient_workspace_ids.len(), 9);
        let source_ids = arena
            .quotient()
            .partial_numerators
            .iter()
            .flat_map(|source| source.coordinates)
            .map(|binding| binding.physical)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(source_ids.len(), 12);
        assert!(source_ids.is_disjoint(&quotient_workspace_ids));
        assert_eq!(arena.transcript().requirements.inputs.len(), 4);
        assert_eq!(arena.transcript().requirements.outputs.len(), 5);
        assert_ne!(
            arena.fri().slots.input_coordinate_ptrs,
            base_slots.groups[0].column_ptrs
        );
        let composition_sources: Vec<_> = (0..8)
            .map(|ordinal| {
                arena
                    .find(None, None, BufferPurpose::CompositionCoefficients, ordinal)
                    .expect("all split composition coordinates are arena-resident")
                    .1
                    .physical
            })
            .collect();
        assert_eq!(
            composition_sources
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            8,
            "composition coefficient sources overlap during their live range"
        );
        assert_eq!(arena.relation().execution.template_use_count, 1566);
        let retained_base = arena
            .relation()
            .source_plan
            .iter()
            .filter(|source| source.plane == RelationSourcePlane::BaseTrace)
            .flat_map(|source| {
                (0..source.column_count)
                    .map(move |ordinal| (source.batch.component, source.part, ordinal))
            })
            .collect::<HashSet<_>>();
        for evaluation in arena
            .logical_buffers()
            .iter()
            .filter(|buffer| buffer.purpose == BufferPurpose::BaseTrace)
        {
            let coefficients = arena
                .find(
                    evaluation.component,
                    evaluation.part,
                    BufferPurpose::BaseCoefficients,
                    evaluation.ordinal,
                )
                .expect("every base evaluation has one coefficient destination")
                .1;
            let evaluations = arena.binding(evaluation.id).unwrap();
            let retained = retained_base.contains(&(
                evaluation.component.unwrap(),
                evaluation.part.unwrap(),
                evaluation.ordinal,
            ));
            assert_eq!(evaluations.physical == coefficients.physical, !retained);
        }
        for evaluation in arena
            .logical_buffers()
            .iter()
            .filter(|buffer| buffer.purpose == BufferPurpose::InteractionTrace)
        {
            let coefficients = arena
                .find(
                    evaluation.component,
                    evaluation.part,
                    BufferPurpose::InteractionCoefficients,
                    evaluation.ordinal,
                )
                .expect("every interaction evaluation has one coefficient destination")
                .1;
            assert_eq!(
                arena.binding(evaluation.id).unwrap().physical,
                coefficients.physical
            );
        }
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| {
                    buffer.component == Some("memory_id_to_big")
                        && buffer.part == Some(TracePartId::MemoryBig(0))
                        && buffer.purpose == BufferPurpose::InteractionTrace
                })
                .count(),
            32
        );
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| {
                    buffer.component == Some("memory_id_to_big")
                        && buffer.part == Some(TracePartId::MemoryBig(0))
                        && buffer.purpose == BufferPurpose::InteractionCoefficients
                })
                .count(),
            32
        );
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| {
                    buffer.component == Some("memory_id_to_big")
                        && buffer.part == Some(TracePartId::MemorySmall)
                        && buffer.purpose == BufferPurpose::InteractionTrace
                })
                .count(),
            12
        );
        assert_eq!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| {
                    buffer.component == Some("memory_id_to_big")
                        && buffer.part == Some(TracePartId::MemorySmall)
                        && buffer.purpose == BufferPurpose::InteractionCoefficients
                })
                .count(),
            12
        );
        assert!(arena.fri_merkle_layer(0, 26).is_some());
        assert_eq!(arena.total_words() % ARENA_ALIGNMENT_WORDS, 0);
        assert!(arena.total_words() >= arena.high_water_words(ProofEpoch::Fri));
    }
}
