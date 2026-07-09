//! Checked device-memory liveness and physical arena coloring.
//!
//! The proof shape describes logical columns. This module turns those columns plus
//! protocol scratch geometry into one stable-address [`ArenaLayout`]. Logical
//! buffers may share a physical slot only when their inclusive proof-epoch
//! lifetimes are disjoint. The resulting alias proof is computed before CUDA is
//! touched; graph capture therefore never discovers residency by allocation luck.

use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::{
    commit_workspace_requirements, fri_workspace_requirements, quotient_workspace_requirements,
    ArenaError, ArenaLayout, ArenaSlotId, ArenaSlotSpec, Blake2sTranscriptRequirements,
    Blake2sTranscriptWorkspaceSlots, CommitBatchSlots, CommitGroupSlots, CommitWorkspaceConfig,
    CommitWorkspaceRequirements, CommitWorkspaceSlots, CudaExecContext, DeviceArena,
    DeviceTranscriptError, FriMerkleTreeSlots, FriWorkspaceConfig, FriWorkspaceRequirements,
    FriWorkspaceSlots, PreparedCommitError, PreparedFriError, PreparedQuotientError,
    QuotientWorkspaceConfig, QuotientWorkspaceRequirements, QuotientWorkspaceSlots,
    RelationGraphError, RelationGraphRequirements, RelationGraphSlots, RelationInstanceSlots,
    TranscriptInputId, TranscriptOutputId,
};
use stwo_cairo_prover::witness::proof_shape::{
    ProofShapeError, RowResolution, TracePartId, TracePartShape,
};

use crate::plan::ProofPlan;
use crate::relation::RelationTracePart;
use crate::relation_execution::{RelationExecutionError, RelationExecutionPlan};
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::schedule::TraceColumnCount;

/// Every arena range is at least 128-byte aligned. Kernel code may rely on this.
pub const ARENA_ALIGNMENT_WORDS: usize = 128 / core::mem::size_of::<u32>();
const BLAKE2S_HASH_WORDS: usize = 8;

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
    Quotient,
    Fri,
    Decommit,
    Assemble,
}

impl ProofEpoch {
    pub const ALL: [Self; 11] = [
        Self::Ingest,
        Self::Witness,
        Self::BaseCommit,
        Self::Interaction,
        Self::InteractionCommit,
        Self::Composition,
        Self::CompositionCommit,
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
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BufferPurpose {
    BaseTrace,
    LookupInputs,
    SubcomponentInputs,
    InteractionTrace,
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
    MerkleLayerScratch,
    CommitColumnPointers,
    CommitColumnLogSizes,
    CommitCoefficientPointers,
    CommitCoefficientSizes,
    CommitOutputPointers,
    MerkleTailPointers,
    RetainedMerkleLayers,
    FriMerkleLayer,
    CompositionTile,
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
    QuotientDenominatorScratch,
    FriPing,
    FriPong,
    FriInputCoordinatePointers,
    FriPingCoordinatePointers,
    FriPongCoordinatePointers,
    FriFoldingChallenge,
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
    RelationSourcePointers,
    RelationOutputPointers,
    RelationDenominators,
    RelationClaimedSum,
    QueryIndices,
    DecommitValues,
    DecommitHashes,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitmentGeometry {
    pub id: CommitmentTreeId,
    pub created: ProofEpoch,
    pub config: CommitWorkspaceConfig,
    pub grouped_column_log_sizes: Vec<Vec<u32>>,
    /// Exact stable arena source paired one-for-one with every canonical log.
    pub grouped_column_sources: Vec<Vec<CommitmentColumnSource>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum DecommitStrategy {
    RetainAllLde = 0,
    RecomputeQueriedLde = 1,
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
    pub kernel_manifest_hash: u64,
    pub decommit_strategy: DecommitStrategy,
}

impl ProtocolIdentity {
    pub fn from_pcs(
        pcs: &PcsConfig,
        channel_tag: u64,
        relation_graph_hash: u64,
        preprocessed_binding_hash: u64,
        kernel_manifest_hash: u64,
        decommit_strategy: DecommitStrategy,
    ) -> Self {
        Self {
            pow_bits: pcs.pow_bits,
            log_blowup_factor: pcs.fri_config.log_blowup_factor,
            log_last_layer_degree_bound: pcs.fri_config.log_last_layer_degree_bound,
            fri_fold_step: pcs.fri_config.fold_step,
            channel_tag,
            relation_graph_hash,
            preprocessed_binding_hash,
            kernel_manifest_hash,
            decommit_strategy,
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

/// Exact protocol scratch geometry for one graph-template key. Component column
/// sizes come from [`ProofPlan`]; this carries the PCS/FRI dimensions that are not
/// component-local.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProtocolGeometry {
    pub identity: ProtocolIdentity,
    pub max_domain_log_size: u32,
    pub lifting_log_size: u32,
    pub n_queries: usize,
    pub total_opened_columns: usize,
    pub proof_capacity_words: usize,
    pub transcript: TranscriptGeometry,
    pub quotient: QuotientGeometry,
    pub commitments: Vec<CommitmentGeometry>,
    /// Commitment-tree leaf log sizes opened by the PCS, in proof tree order.
    /// Persistent preprocessed trees have no per-proof prepared-commit workspace,
    /// but their authentication paths still require decommit capacity.
    pub opened_tree_log_sizes: Vec<u32>,
    /// Fully retained FRI Merkle tree leaf log sizes, in transcript order.
    pub fri_layer_log_sizes: Vec<u32>,
}

impl ProtocolGeometry {
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
        if self.lifting_log_size < self.max_domain_log_size {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "lifting domain is smaller than the maximal commitment domain",
            ));
        }
        quotient_workspace_requirements(
            self.quotient_workspace_config(),
            &self.quotient.partial_numerator_log_sizes,
        )
        .map_err(ArenaPlanError::Quotient)?;
        let mut ids = Vec::new();
        for commitment in &self.commitments {
            if ids.contains(&commitment.id) {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "duplicate commitment geometry",
                ));
            }
            ids.push(commitment.id);
            if commitment.config.lifting_log_size > self.lifting_log_size {
                return Err(ArenaPlanError::InvalidProtocolGeometry(
                    "commitment exceeds the maximal lifting domain",
                ));
            }
            BufferLifetime::new(commitment.created, ProofEpoch::Decommit)?;
            if commitment.grouped_column_sources.len() != commitment.grouped_column_log_sizes.len()
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
                match source {
                    CommitmentColumnSource::Trace { purpose, .. }
                        if !matches!(
                            purpose,
                            BufferPurpose::BaseTrace | BufferPurpose::InteractionTrace
                        ) =>
                    {
                        return Err(ArenaPlanError::InvalidProtocolGeometry(
                            "commitment trace source is not a base/interaction column",
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
            }
            commit_workspace_requirements(commitment.config, &commitment.grouped_column_log_sizes)
                .map_err(ArenaPlanError::Commit)?;
        }
        if self
            .opened_tree_log_sizes
            .iter()
            .any(|&log_size| log_size > self.lifting_log_size)
        {
            return Err(ArenaPlanError::InvalidProtocolGeometry(
                "opened commitment tree exceeds the lifting domain",
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
        feed(b"stwo-cairo-protocol-geometry-v2\0");
        feed(&self.identity.pow_bits.to_le_bytes());
        feed(&self.identity.log_blowup_factor.to_le_bytes());
        feed(&self.identity.log_last_layer_degree_bound.to_le_bytes());
        feed(&self.identity.fri_fold_step.to_le_bytes());
        feed(&self.identity.channel_tag.to_le_bytes());
        feed(&self.identity.relation_graph_hash.to_le_bytes());
        feed(&self.identity.preprocessed_binding_hash.to_le_bytes());
        feed(&self.identity.kernel_manifest_hash.to_le_bytes());
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
            for group in &commitment.grouped_column_log_sizes {
                feed(&(group.len() as u64).to_le_bytes());
                for log_size in group {
                    feed(&log_size.to_le_bytes());
                }
            }
            for group in &commitment.grouped_column_sources {
                for source in group {
                    match source {
                        CommitmentColumnSource::Trace {
                            component,
                            part,
                            purpose,
                            ordinal,
                        } => {
                            feed(&[0]);
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
                                BufferPurpose::BaseTrace => 0,
                                BufferPurpose::InteractionTrace => 1,
                                _ => u8::MAX,
                            }]);
                            feed(&ordinal.to_le_bytes());
                        }
                        CommitmentColumnSource::Composition { ordinal } => {
                            feed(&[1]);
                            feed(&ordinal.to_le_bytes());
                        }
                    }
                }
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

#[derive(Clone, Debug)]
struct LogicalCommitWorkspace {
    id: CommitmentTreeId,
    config: CommitWorkspaceConfig,
    grouped_column_log_sizes: Vec<Vec<u32>>,
    grouped_column_sources: Vec<Vec<CommitmentColumnSource>>,
    requirements: CommitWorkspaceRequirements,
    twiddles: LogicalBufferId,
    lde_tile: LogicalBufferId,
    leaf_state: LogicalBufferId,
    merkle_scratch: Option<LogicalBufferId>,
    retained_layers: Vec<LogicalBufferId>,
    tail_level_ptrs: Option<LogicalBufferId>,
    tail_outputs: Vec<LogicalBufferId>,
    groups: Vec<LogicalCommitGroupSlots>,
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
    folding_challenges: Vec<LogicalBufferId>,
    trees: Vec<Vec<LogicalBufferId>>,
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
    denominator_scratch: LogicalBufferId,
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
    requirements: RelationGraphRequirements,
    descriptors: LogicalBufferId,
    alphas: LogicalBufferId,
    z: LogicalBufferId,
    inverse_scratch: LogicalBufferId,
    reduction_a: LogicalBufferId,
    reduction_b: LogicalBufferId,
    scan_eval_scratch: LogicalBufferId,
    scan_temp_scratch: LogicalBufferId,
    instances: Vec<LogicalRelationInstanceSlots>,
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
    pub requirements: CommitWorkspaceRequirements,
    pub twiddles: ArenaBinding,
    pub slots: CommitWorkspaceSlots,
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

#[derive(Clone, Debug)]
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
    pub requirements: RelationGraphRequirements,
    pub slots: RelationGraphSlots,
}

#[derive(Clone, Debug)]
pub struct ProofArenaPlan {
    pub shape_key: stwo_cairo_prover::witness::proof_shape::ProofShapeKey,
    pub protocol_key: u64,
    logical: Vec<LogicalBuffer>,
    bindings: Vec<ArenaBinding>,
    layout: ArenaLayout,
    high_water_words: Vec<(ProofEpoch, usize)>,
    commitments: Vec<PlannedCommitment>,
    quotient: PlannedQuotientWorkspace,
    fri: PlannedFriWorkspace,
    transcript: PlannedTranscriptWorkspace,
    relation: PlannedRelationWorkspace,
}

impl ProofArenaPlan {
    pub fn build(plan: &ProofPlan, protocol: &ProtocolGeometry) -> Result<Self, ArenaPlanError> {
        plan.proof_shape()
            .require_arena_ready()
            .map_err(ArenaPlanError::Shape)?;
        protocol.validate()?;
        if protocol.identity.relation_graph_hash != plan.relation_graph_hash {
            return Err(ArenaPlanError::RelationGraphMismatch {
                planned: plan.relation_graph_hash,
                protocol: protocol.identity.relation_graph_hash,
            });
        }

        let mut logical = Vec::new();
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
                push_component_columns(
                    &mut logical,
                    component.node.id,
                    part.part,
                    BufferPurpose::BaseTrace,
                    trace_columns,
                    part.padded_rows,
                    BufferLifetime::new(ProofEpoch::Witness, ProofEpoch::Decommit)?,
                )?;
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
                if let Some(words) = component.node.facts.sub_words {
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
                    push_component_columns(
                        &mut logical,
                        component.node.id,
                        part.part,
                        BufferPurpose::InteractionTrace,
                        coordinate_columns,
                        part.padded_rows,
                        BufferLifetime::new(ProofEpoch::Interaction, ProofEpoch::Decommit)?,
                    )?;
                }
            }
        }
        let logical_relation = append_relation_buffers(&mut logical, plan)?;
        let (logical_commitments, logical_quotient, logical_fri, logical_transcript) =
            append_protocol_buffers(&mut logical, protocol)?;
        validate_commitment_sources(&logical, &logical_commitments)?;

        let (bindings, specs, total_words) = color_logical_buffers(&logical)?;
        let layout = ArenaLayout::new(total_words, &specs).map_err(ArenaPlanError::Arena)?;
        validate_aliases(&logical, &bindings)?;
        let high_water_words = ProofEpoch::ALL
            .into_iter()
            .map(|epoch| (epoch, high_water_at(epoch, &logical, &bindings)))
            .collect();
        let commitments = logical_commitments
            .into_iter()
            .map(|commitment| resolve_commitment_slots(commitment, &bindings))
            .collect::<Result<_, _>>()?;
        let quotient = resolve_quotient_slots(logical_quotient, &bindings)?;
        let fri = resolve_fri_slots(logical_fri, &bindings)?;
        if quotient.output_values != fri.input_values {
            return Err(ArenaPlanError::QuotientFriInputMismatch {
                quotient: quotient.output_values,
                fri: fri.input_values,
            });
        }
        let transcript = resolve_transcript_slots(logical_transcript, &bindings)?;
        let relation = resolve_relation_slots(logical_relation, &bindings)?;

        Ok(Self {
            shape_key: plan.shape_key,
            protocol_key: protocol.key(),
            logical,
            bindings,
            layout,
            high_water_words,
            commitments,
            quotient,
            fri,
            transcript,
            relation,
        })
    }

    pub fn layout(&self) -> &ArenaLayout {
        &self.layout
    }

    pub fn logical_buffers(&self) -> &[LogicalBuffer] {
        &self.logical
    }

    pub fn bindings(&self) -> &[ArenaBinding] {
        &self.bindings
    }

    pub fn binding(&self, logical: LogicalBufferId) -> Option<ArenaBinding> {
        self.bindings
            .iter()
            .find(|binding| binding.logical == logical)
            .copied()
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

    pub fn commitments(&self) -> &[PlannedCommitment] {
        &self.commitments
    }

    pub fn commitment(&self, id: CommitmentTreeId) -> Option<&PlannedCommitment> {
        self.commitments
            .iter()
            .find(|commitment| commitment.id == id)
    }

    pub fn fri(&self) -> &PlannedFriWorkspace {
        &self.fri
    }

    pub fn quotient(&self) -> &PlannedQuotientWorkspace {
        &self.quotient
    }

    pub fn transcript(&self) -> &PlannedTranscriptWorkspace {
        &self.transcript
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
    QuotientFriInputMismatch {
        quotient: ArenaBinding,
        fri: ArenaBinding,
    },
    RelationExecution(RelationExecutionError),
    Relation(RelationGraphError),
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
    Quotient(PreparedQuotientError),
    Fri(PreparedFriError),
    Transcript(DeviceTranscriptError),
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

fn push_component_columns(
    logical: &mut Vec<LogicalBuffer>,
    component: &'static str,
    part: TracePartId,
    purpose: BufferPurpose,
    columns: u32,
    rows: u64,
    lifetime: BufferLifetime,
) -> Result<(), ArenaPlanError> {
    let len_words = usize::try_from(rows).map_err(|_| ArenaPlanError::SizeOverflow)?;
    for column in 0..columns {
        push_buffer(
            logical,
            Some(component),
            Some(part),
            purpose,
            column,
            len_words,
            lifetime,
        )?;
    }
    Ok(())
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

fn append_relation_buffers(
    logical: &mut Vec<LogicalBuffer>,
    proof: &ProofPlan,
) -> Result<LogicalRelationWorkspace, ArenaPlanError> {
    let execution = RelationExecutionPlan::from_proof_plan(proof, &CAIRO_RELATION_GRAPH)
        .map_err(ArenaPlanError::RelationExecution)?;
    let requirements = execution.requirements().map_err(|error| match error {
        RelationExecutionError::BackendPlan(error) => ArenaPlanError::Relation(error),
        other => ArenaPlanError::RelationExecution(other),
    })?;
    let persistent = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Decommit)?;
    let interaction = BufferLifetime::at(ProofEpoch::Interaction);
    let challenge = BufferLifetime::new(ProofEpoch::Interaction, ProofEpoch::Composition)?;
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

    let mut instances = Vec::with_capacity(requirements.instances.len());
    for (ordinal, requirement) in requirements.instances.iter().enumerate() {
        let batch = execution.batches.get(requirement.batch_index).ok_or(
            ArenaPlanError::InvalidProtocolGeometry(
                "relation requirement references an unknown batch",
            ),
        )?;
        let part = match batch.trace_part {
            RelationTracePart::Component => TracePartId::Main,
            RelationTracePart::EachMemoryBig => TracePartId::MemoryBig(
                u32::try_from(requirement.instance_index)
                    .map_err(|_| ArenaPlanError::SizeOverflow)?,
            ),
            RelationTracePart::MemorySmall => TracePartId::MemorySmall,
        };
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
        let output_coordinates = if output_coordinates.is_empty()
            && matches!(
                batch.trace_part,
                RelationTracePart::EachMemoryBig | RelationTracePart::MemorySmall
            ) {
            (0..requirement.output_coordinate_count)
                .map(|coordinate| {
                    push_buffer_id(
                        logical,
                        Some(batch.component),
                        Some(part),
                        BufferPurpose::InteractionTrace,
                        u32::try_from(coordinate).map_err(|_| ArenaPlanError::SizeOverflow)?,
                        requirement.output_coordinate_words,
                        BufferLifetime::new(ProofEpoch::Interaction, ProofEpoch::Decommit)?,
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
        requirements,
        descriptors,
        alphas,
        z,
        inverse_scratch,
        reduction_a,
        reduction_b,
        scan_eval_scratch,
        scan_temp_scratch,
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
                        CommitmentTreeId::Base,
                        CommitmentColumnSource::Trace {
                            purpose: BufferPurpose::BaseTrace,
                            ..
                        }
                    ) | (
                        CommitmentTreeId::Interaction,
                        CommitmentColumnSource::Trace {
                            purpose: BufferPurpose::InteractionTrace,
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

fn append_protocol_buffers(
    logical: &mut Vec<LogicalBuffer>,
    protocol: &ProtocolGeometry,
) -> Result<
    (
        Vec<LogicalCommitWorkspace>,
        LogicalQuotientWorkspace,
        LogicalFriWorkspace,
        LogicalTranscriptWorkspace,
    ),
    ArenaPlanError,
> {
    let domain_rows = checked_pow2(protocol.max_domain_log_size)?;
    let commit_requirements = protocol
        .commitments
        .iter()
        .map(|commitment| {
            commit_workspace_requirements(commitment.config, &commitment.grouped_column_log_sizes)
                .map_err(ArenaPlanError::Commit)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fri_config = protocol.fri_workspace_config()?;
    let fri_requirements = fri_workspace_requirements(fri_config).map_err(ArenaPlanError::Fri)?;
    let quotient_config = protocol.quotient_workspace_config();
    let quotient_requirements = quotient_workspace_requirements(
        quotient_config,
        &protocol.quotient.partial_numerator_log_sizes,
    )
    .map_err(ArenaPlanError::Quotient)?;
    let forward_twiddle_words = commit_requirements
        .iter()
        .map(|requirements| requirements.twiddle_words)
        .max()
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "proof has no commitment workspace",
        ))?
        .max(quotient_requirements.forward_twiddle_words);
    // Forward and inverse twiddles are distinct persistent values. Sharing one
    // physical range and overwriting it at the FRI boundary would invalidate
    // the later queried-LDE recomputation used by decommitment.
    let forward_twiddles = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::ForwardTwiddles,
        0,
        forward_twiddle_words,
        BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Decommit)?,
    )?;
    let inverse_twiddles = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::InverseTwiddles,
        0,
        fri_requirements.twiddle_words,
        BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Decommit)?,
    )?;
    let quotient_inverse_twiddles = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientInverseTwiddles,
        0,
        quotient_requirements.inverse_twiddle_words,
        BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Decommit)?,
    )?;

    let mut logical_commitments = Vec::with_capacity(protocol.commitments.len());
    for (commitment_index, (geometry, requirements)) in protocol
        .commitments
        .iter()
        .zip(commit_requirements)
        .enumerate()
    {
        let at = BufferLifetime::at(geometry.created);
        let retained = BufferLifetime::new(geometry.created, ProofEpoch::Decommit)?;
        // Captured kernels dereference these tables on every warm replay. Their
        // contents are commitment-specific and therefore cannot alias another
        // segment's descriptors even though the compute epochs are disjoint.
        let descriptor = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Decommit)?;
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
        let lde_tile = push_buffer_id(
            logical,
            None,
            None,
            BufferPurpose::CommitLdeTile,
            ordinal()?,
            requirements.lde_tile_words,
            at,
        )?;
        let leaf_state = push_buffer_id(
            logical,
            None,
            None,
            BufferPurpose::MerkleLeafState,
            ordinal()?,
            requirements.leaf_state_words,
            if geometry.config.unretained_bottom_layers == 0 {
                retained
            } else {
                at
            },
        )?;
        let merkle_scratch = requirements
            .merkle_scratch_words
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
        let retained_layers = requirements
            .retained_layers
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
        let tail_level_ptrs = requirements
            .tail_pointer_words
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
        let tail_outputs = requirements
            .tail_outputs
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
                        Ok(LogicalCommitBatchSlots {
                            coefficient_ptrs: push_buffer_id(
                                logical,
                                None,
                                None,
                                BufferPurpose::CommitCoefficientPointers,
                                ordinal()?,
                                batch.coefficient_pointer_words,
                                descriptor,
                            )?,
                            coefficient_sizes: push_buffer_id(
                                logical,
                                None,
                                None,
                                BufferPurpose::CommitCoefficientSizes,
                                ordinal()?,
                                batch.coefficient_size_words,
                                descriptor,
                            )?,
                            output_ptrs: push_buffer_id(
                                logical,
                                None,
                                None,
                                BufferPurpose::CommitOutputPointers,
                                ordinal()?,
                                batch.output_pointer_words,
                                descriptor,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, ArenaPlanError>>()?;
                Ok(LogicalCommitGroupSlots {
                    column_ptrs,
                    column_log_sizes,
                    batches,
                })
            })
            .collect::<Result<Vec<_>, ArenaPlanError>>()?;
        logical_commitments.push(LogicalCommitWorkspace {
            id: geometry.id,
            config: geometry.config,
            grouped_column_log_sizes: geometry.grouped_column_log_sizes.clone(),
            grouped_column_sources: geometry.grouped_column_sources.clone(),
            requirements,
            twiddles: forward_twiddles,
            lde_tile,
            leaf_state,
            merkle_scratch,
            retained_layers,
            tail_level_ptrs,
            tail_outputs,
            groups,
        });
    }

    let secure_domain_words = domain_rows
        .checked_mul(4)
        .ok_or(ArenaPlanError::SizeOverflow)?;
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::CompositionTile,
        0,
        secure_domain_words,
        BufferLifetime::new(ProofEpoch::Composition, ProofEpoch::CompositionCommit)?,
    )?;
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
    let quotient_descriptor = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Decommit)?;
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
    let sample_points = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientSamplePoints,
        0,
        quotient_requirements.sample_point_words,
        quotient_live,
    )?;
    let first_linear_terms = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientFirstLinearTerms,
        0,
        quotient_requirements.first_linear_term_words,
        quotient_live,
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
    let denominator_scratch = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::QuotientDenominatorScratch,
        0,
        quotient_requirements.denominator_words,
        quotient_live,
    )?;
    let logical_quotient = LogicalQuotientWorkspace {
        config: quotient_config,
        requirements: quotient_requirements,
        forward_twiddles,
        inverse_subdomain_twiddles: quotient_inverse_twiddles,
        partial_numerators,
        sample_points,
        first_linear_terms,
        partial_log_sizes,
        partial_coordinate_ptrs,
        subdomain_coordinate_ptrs,
        output_coordinate_ptrs,
        coefficient_sizes,
        subdomain_values,
        output_values,
        denominator_scratch,
    };
    let fri_live = BufferLifetime::new(ProofEpoch::Fri, ProofEpoch::Decommit)?;
    let fri_descriptor = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Decommit)?;
    let evaluation_ping = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FriPing,
        0,
        fri_requirements.evaluation_ping_words,
        fri_live,
    )?;
    let evaluation_pong = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::FriPong,
        0,
        fri_requirements.evaluation_pong_words,
        fri_live,
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
        folding_challenges,
        trees: fri_trees,
    };
    let composition = protocol
        .commitments
        .iter()
        .find(|commitment| commitment.id == CommitmentTreeId::Composition)
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "missing composition commitment geometry",
        ))?;
    let mut composition_column = 0u32;
    for &log_size in composition.grouped_column_log_sizes.iter().flatten() {
        push_buffer(
            logical,
            None,
            None,
            BufferPurpose::CompositionCoefficients,
            composition_column,
            checked_pow2(log_size)?,
            BufferLifetime::new(ProofEpoch::Composition, ProofEpoch::Decommit)?,
        )?;
        composition_column = composition_column
            .checked_add(1)
            .ok_or(ArenaPlanError::SizeOverflow)?;
    }
    if composition_column != 8 {
        return Err(ArenaPlanError::InvalidProtocolGeometry(
            "composition commitment must contain eight M31 coordinate polynomials",
        ));
    }
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::QueryIndices,
        0,
        protocol.n_queries,
        BufferLifetime::new(ProofEpoch::Fri, ProofEpoch::Decommit)?,
    )?;
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::DecommitValues,
        0,
        protocol
            .n_queries
            .checked_mul(protocol.total_opened_columns)
            .ok_or(ArenaPlanError::SizeOverflow)?,
        BufferLifetime::new(ProofEpoch::Decommit, ProofEpoch::Assemble)?,
    )?;
    let trace_path_hashes =
        protocol
            .opened_tree_log_sizes
            .iter()
            .try_fold(0usize, |total, &log_size| {
                total
                    .checked_add(log_size as usize)
                    .ok_or(ArenaPlanError::SizeOverflow)
            })?;
    let fri_path_hashes =
        protocol
            .fri_layer_log_sizes
            .iter()
            .try_fold(0usize, |total, &log_size| {
                total
                    .checked_add(log_size as usize)
                    .ok_or(ArenaPlanError::SizeOverflow)
            })?;
    let path_hashes = trace_path_hashes
        .checked_add(fri_path_hashes)
        .ok_or(ArenaPlanError::SizeOverflow)?;
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::DecommitHashes,
        0,
        protocol
            .n_queries
            .checked_mul(path_hashes)
            .and_then(|hashes| hashes.checked_mul(BLAKE2S_HASH_WORDS))
            .ok_or(ArenaPlanError::SizeOverflow)?,
        BufferLifetime::new(ProofEpoch::Decommit, ProofEpoch::Assemble)?,
    )?;
    let transcript_live = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?;
    let transcript_requirements = protocol.transcript.requirements.clone();
    let transcript_state = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::TranscriptState,
        0,
        transcript_requirements.state_words,
        transcript_live,
    )?;
    let transcript_boundaries = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::TranscriptBoundarySnapshots,
        0,
        transcript_requirements.boundary_snapshot_words,
        transcript_live,
    )?;
    let transcript_input_snapshots = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::TranscriptInputSnapshots,
        0,
        transcript_requirements.input_snapshot_words,
        transcript_live,
    )?;
    let transcript_output_snapshots = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::TranscriptOutputSnapshots,
        0,
        transcript_requirements.output_snapshot_words,
        transcript_live,
    )?;
    let transcript_inputs = transcript_requirements
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
                    transcript_live,
                )?,
            ))
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let transcript_outputs = transcript_requirements
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
                    transcript_live,
                )?,
            ))
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let logical_transcript = LogicalTranscriptWorkspace {
        schedule_key: protocol.transcript.schedule_key,
        requirements: transcript_requirements,
        state: transcript_state,
        boundary_snapshots: transcript_boundaries,
        input_snapshots: transcript_input_snapshots,
        output_snapshots: transcript_output_snapshots,
        inputs: transcript_inputs,
        outputs: transcript_outputs,
    };
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::ProofBytes,
        0,
        protocol.proof_capacity_words,
        BufferLifetime::at(ProofEpoch::Assemble),
    )?;
    Ok((
        logical_commitments,
        logical_quotient,
        logical_fri,
        logical_transcript,
    ))
}

fn resolve_commitment_slots(
    logical: LogicalCommitWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedCommitment, ArenaPlanError> {
    let binding = |id: LogicalBufferId| {
        bindings
            .iter()
            .find(|binding| binding.logical == id)
            .copied()
            .ok_or(ArenaPlanError::MissingBinding(id))
    };
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let groups = logical
        .groups
        .into_iter()
        .map(|group| {
            Ok(CommitGroupSlots {
                column_ptrs: physical(group.column_ptrs)?,
                column_log_sizes: physical(group.column_log_sizes)?,
                batches: group
                    .batches
                    .into_iter()
                    .map(|batch| {
                        Ok(CommitBatchSlots {
                            coefficient_ptrs: physical(batch.coefficient_ptrs)?,
                            coefficient_sizes: physical(batch.coefficient_sizes)?,
                            output_ptrs: physical(batch.output_ptrs)?,
                        })
                    })
                    .collect::<Result<Vec<_>, ArenaPlanError>>()?,
            })
        })
        .collect::<Result<Vec<_>, ArenaPlanError>>()?;
    let slots = CommitWorkspaceSlots {
        lde_tile: physical(logical.lde_tile)?,
        leaf_state: physical(logical.leaf_state)?,
        merkle_scratch: logical.merkle_scratch.map(physical).transpose()?,
        retained_layers: logical
            .retained_layers
            .into_iter()
            .map(physical)
            .collect::<Result<Vec<_>, _>>()?,
        tail_level_ptrs: logical.tail_level_ptrs.map(physical).transpose()?,
        tail_outputs: logical
            .tail_outputs
            .into_iter()
            .map(physical)
            .collect::<Result<Vec<_>, _>>()?,
        groups,
    };
    logical
        .requirements
        .arena_slot_requirements(&slots)
        .map_err(ArenaPlanError::Commit)?;
    Ok(PlannedCommitment {
        id: logical.id,
        config: logical.config,
        grouped_column_log_sizes: logical.grouped_column_log_sizes,
        grouped_column_sources: logical.grouped_column_sources,
        requirements: logical.requirements,
        twiddles: binding(logical.twiddles)?,
        slots,
    })
}

fn resolve_quotient_slots(
    logical: LogicalQuotientWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedQuotientWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| {
        bindings
            .iter()
            .find(|binding| binding.logical == id)
            .copied()
            .ok_or(ArenaPlanError::MissingBinding(id))
    };
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
        denominator_scratch: physical(logical.denominator_scratch)?,
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
    Ok(PlannedQuotientWorkspace {
        config: logical.config,
        requirements: logical.requirements,
        forward_twiddles,
        inverse_subdomain_twiddles,
        partial_numerators,
        slots,
        output_values,
    })
}

fn resolve_fri_slots(
    logical: LogicalFriWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedFriWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| {
        bindings
            .iter()
            .find(|binding| binding.logical == id)
            .copied()
            .ok_or(ArenaPlanError::MissingBinding(id))
    };
    let physical = |id| Ok::<_, ArenaPlanError>(binding(id)?.physical);
    let slots = FriWorkspaceSlots {
        evaluation_ping: physical(logical.evaluation_ping)?,
        evaluation_pong: physical(logical.evaluation_pong)?,
        input_coordinate_ptrs: physical(logical.input_coordinate_ptrs)?,
        ping_coordinate_ptrs: physical(logical.ping_coordinate_ptrs)?,
        pong_coordinate_ptrs: physical(logical.pong_coordinate_ptrs)?,
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

fn resolve_transcript_slots(
    logical: LogicalTranscriptWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedTranscriptWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| {
        bindings
            .iter()
            .find(|binding| binding.logical == id)
            .copied()
            .ok_or(ArenaPlanError::MissingBinding(id))
    };
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

fn resolve_relation_slots(
    logical: LogicalRelationWorkspace,
    bindings: &[ArenaBinding],
) -> Result<PlannedRelationWorkspace, ArenaPlanError> {
    let binding = |id: LogicalBufferId| {
        bindings
            .iter()
            .find(|binding| binding.logical == id)
            .copied()
            .ok_or(ArenaPlanError::MissingBinding(id))
    };
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
    lifetimes: Vec<(LogicalBufferId, BufferLifetime)>,
}

fn color_logical_buffers(
    logical: &[LogicalBuffer],
) -> Result<(Vec<ArenaBinding>, Vec<ArenaSlotSpec>, usize), ArenaPlanError> {
    let mut order: Vec<usize> = (0..logical.len()).collect();
    order.sort_unstable_by_key(|&index| {
        let buffer = &logical[index];
        (
            buffer.lifetime.first,
            core::cmp::Reverse(buffer.len_words),
            buffer.id,
        )
    });

    let mut slots: Vec<ColoredSlot> = Vec::new();
    let mut bindings = Vec::with_capacity(logical.len());
    for index in order {
        let buffer = &logical[index];
        let candidate = slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                slot.lifetimes
                    .iter()
                    .all(|&(_, lifetime)| !lifetime.overlaps(buffer.lifetime))
            })
            .min_by_key(|(_, slot)| {
                (
                    buffer.len_words.saturating_sub(slot.len_words),
                    slot.len_words.max(buffer.len_words),
                    slot.id,
                )
            })
            .map(|(index, _)| index);
        let slot_index = match candidate {
            Some(index) => index,
            None => {
                let id = ArenaSlotId(
                    u32::try_from(slots.len() + 1).map_err(|_| ArenaPlanError::SizeOverflow)?,
                );
                slots.push(ColoredSlot {
                    id,
                    len_words: 0,
                    lifetimes: Vec::new(),
                });
                slots.len() - 1
            }
        };
        let slot = &mut slots[slot_index];
        slot.len_words = slot.len_words.max(buffer.len_words);
        slot.lifetimes.push((buffer.id, buffer.lifetime));
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
    for (index, first) in logical.iter().enumerate() {
        let first_binding = bindings
            .iter()
            .find(|binding| binding.logical == first.id)
            .ok_or(ArenaPlanError::MissingBinding(first.id))?;
        for second in &logical[index + 1..] {
            let second_binding = bindings
                .iter()
                .find(|binding| binding.logical == second.id)
                .ok_or(ArenaPlanError::MissingBinding(second.id))?;
            if first_binding.physical == second_binding.physical
                && first.lifetime.overlaps(second.lifetime)
            {
                return Err(ArenaPlanError::AliasedLiveBuffers {
                    physical: first_binding.physical,
                    first: first.id,
                    second: second.id,
                });
            }
        }
    }
    Ok(())
}

fn high_water_at(epoch: ProofEpoch, logical: &[LogicalBuffer], bindings: &[ArenaBinding]) -> usize {
    let mut physical = Vec::<ArenaSlotId>::new();
    let mut words = 0usize;
    for buffer in logical
        .iter()
        .filter(|buffer| buffer.lifetime.contains(epoch))
    {
        let binding = bindings
            .iter()
            .find(|binding| binding.logical == buffer.id)
            .expect("bindings are complete before high-water computation");
        if !physical.contains(&binding.physical) {
            physical.push(binding.physical);
            words += bindings
                .iter()
                .filter(|candidate| candidate.physical == binding.physical)
                .map(|candidate| candidate.len_words)
                .max()
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
    fn disjoint_epochs_alias_but_live_ranges_never_do() {
        let logical = vec![
            test_buffer(0, 128, ProofEpoch::Witness, ProofEpoch::Witness),
            test_buffer(1, 96, ProofEpoch::Interaction, ProofEpoch::Interaction),
            test_buffer(2, 64, ProofEpoch::Witness, ProofEpoch::Interaction),
        ];
        let (bindings, specs, total) = color_logical_buffers(&logical).unwrap();
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
    fn physical_slots_are_128_byte_aligned() {
        let logical = vec![
            test_buffer(0, 33, ProofEpoch::Witness, ProofEpoch::Witness),
            test_buffer(1, 65, ProofEpoch::Witness, ProofEpoch::Witness),
        ];
        let (_, specs, total) = color_logical_buffers(&logical).unwrap();
        assert_eq!(total % ARENA_ALIGNMENT_WORDS, 0);
        assert!(specs
            .iter()
            .all(|spec| spec.offset_words % ARENA_ALIGNMENT_WORDS == 0));
    }

    #[test]
    fn generated_proof_shape_builds_a_complete_alias_checked_arena() {
        let default_shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let mut components = default_shape.components().to_vec();
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
        let shape = ProofShape::new(components).unwrap();
        let proof =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let group_columns = |mut columns: Vec<(u32, CommitmentColumnSource)>| {
            columns.sort_by_key(|(log_size, _)| *log_size);
            let logs = columns
                .chunks(16)
                .map(|group| group.iter().map(|(log_size, _)| *log_size).collect())
                .collect();
            let sources = columns
                .chunks(16)
                .map(|group| group.iter().map(|(_, source)| *source).collect())
                .collect();
            (logs, sources)
        };
        let (base_logs, base_sources) = group_columns(
            (0..cairo_air::components::memory_id_to_big::BIG_N_COLUMNS as u32)
                .map(|ordinal| {
                    (
                        6,
                        CommitmentColumnSource::Trace {
                            component: "memory_id_to_big",
                            part: TracePartId::MemoryBig(0),
                            purpose: BufferPurpose::BaseTrace,
                            ordinal,
                        },
                    )
                })
                .chain(
                    (0..cairo_air::components::memory_id_to_small::N_TRACE_COLUMNS as u32).map(
                        |ordinal| {
                            (
                                5,
                                CommitmentColumnSource::Trace {
                                    component: "memory_id_to_big",
                                    part: TracePartId::MemorySmall,
                                    purpose: BufferPurpose::BaseTrace,
                                    ordinal,
                                },
                            )
                        },
                    ),
                )
                .collect(),
        );
        let (interaction_logs, interaction_sources) = group_columns(
            (0..32)
                .map(|ordinal| {
                    (
                        6,
                        CommitmentColumnSource::Trace {
                            component: "memory_id_to_big",
                            part: TracePartId::MemoryBig(0),
                            purpose: BufferPurpose::InteractionTrace,
                            ordinal,
                        },
                    )
                })
                .chain((0..12).map(|ordinal| {
                    (
                        5,
                        CommitmentColumnSource::Trace {
                            component: "memory_id_to_big",
                            part: TracePartId::MemorySmall,
                            purpose: BufferPurpose::InteractionTrace,
                            ordinal,
                        },
                    )
                }))
                .collect(),
        );
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
            ],
            8,
        )
        .unwrap();
        let protocol = ProtocolGeometry {
            identity: ProtocolIdentity {
                pow_bits: 26,
                log_blowup_factor: 1,
                log_last_layer_degree_bound: 0,
                fri_fold_step: 1,
                channel_tag: 1,
                relation_graph_hash: proof.relation_graph_hash,
                preprocessed_binding_hash: 3,
                kernel_manifest_hash: 4,
                decommit_strategy: DecommitStrategy::RecomputeQueriedLde,
            },
            max_domain_log_size: 26,
            lifting_log_size: 26,
            n_queries: 70,
            total_opened_columns: 1024,
            proof_capacity_words: 1 << 20,
            transcript: TranscriptGeometry {
                schedule_key: transcript_schedule.protocol_key(),
                requirements: transcript_schedule.requirements().clone(),
            },
            quotient: QuotientGeometry {
                partial_numerator_log_sizes: vec![25, 24, 25],
            },
            commitments: vec![
                CommitmentGeometry {
                    id: CommitmentTreeId::Base,
                    created: ProofEpoch::BaseCommit,
                    config: CommitWorkspaceConfig {
                        log_blowup_factor: 1,
                        lifting_log_size: 26,
                        unretained_bottom_layers: 4,
                        max_fused_tail_levels: 12,
                    },
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
                },
            ],
            opened_tree_log_sizes: vec![26, 26, 26, 26],
            fri_layer_log_sizes: (2..=26).rev().collect(),
        };
        let mut changed_quotient = protocol.clone();
        changed_quotient
            .quotient
            .partial_numerator_log_sizes
            .push(23);
        assert_ne!(protocol.key(), changed_quotient.key());
        let arena = ProofArenaPlan::build(&proof, &protocol).unwrap();
        arena.validate_aliases().unwrap();
        assert_eq!(arena.protocol_key, protocol.key());
        assert_eq!(arena.shape_key, proof.shape_key);
        assert_eq!(arena.commitments().len(), 3);
        let base = arena.commitment(CommitmentTreeId::Base).unwrap();
        let interaction = arena.commitment(CommitmentTreeId::Interaction).unwrap();
        assert_ne!(
            base.slots.groups[0].column_ptrs, interaction.slots.groups[0].column_ptrs,
            "captured commitment descriptor tables must persist independently"
        );
        assert_eq!(arena.fri().requirements.trees.len(), 25);
        assert_eq!(arena.quotient().requirements.sample_count, 3);
        assert_eq!(arena.quotient().partial_numerators.len(), 3);
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
            quotient_slots.denominator_scratch,
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(quotient_workspace_ids.len(), 10);
        let source_ids = arena
            .quotient()
            .partial_numerators
            .iter()
            .flat_map(|source| source.coordinates)
            .map(|binding| binding.physical)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(source_ids.len(), 12);
        assert!(source_ids.is_disjoint(&quotient_workspace_ids));
        assert_eq!(arena.transcript().requirements.inputs.len(), 1);
        assert_eq!(arena.transcript().requirements.outputs.len(), 1);
        assert_ne!(
            arena.fri().slots.input_coordinate_ptrs,
            base.slots.groups[0].column_ptrs
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
                        && buffer.part == Some(TracePartId::MemorySmall)
                        && buffer.purpose == BufferPurpose::InteractionTrace
                })
                .count(),
            12
        );
        assert!(arena.fri_merkle_layer(0, 26).is_some());
        assert_eq!(arena.total_words() % ARENA_ALIGNMENT_WORDS, 0);
        assert!(arena.total_words() >= arena.high_water_words(ProofEpoch::Fri));
    }
}
