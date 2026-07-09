//! Checked device-memory liveness and physical arena coloring.
//!
//! The proof shape describes logical columns. This module turns those columns plus
//! protocol scratch geometry into one stable-address [`ArenaLayout`]. Logical
//! buffers may share a physical slot only when their inclusive proof-epoch
//! lifetimes are disjoint. The resulting alias proof is computed before CUDA is
//! touched; graph capture therefore never discovers residency by allocation luck.

use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::{
    commit_workspace_requirements, ArenaError, ArenaLayout, ArenaSlotId, ArenaSlotSpec,
    CommitBatchSlots, CommitGroupSlots, CommitWorkspaceConfig, CommitWorkspaceRequirements,
    CommitWorkspaceSlots, CudaExecContext, DeviceArena, PreparedCommitError,
};
use stwo_cairo_prover::witness::proof_shape::{
    ProofShapeError, RowResolution, TracePartId, TracePartShape,
};

use crate::plan::ProofPlan;
use crate::schedule::TraceColumnCount;

/// Every arena range is at least 128-byte aligned. Kernel code may rely on this.
pub const ARENA_ALIGNMENT_WORDS: usize = 128 / core::mem::size_of::<u32>();
const BLAKE2S_HASH_WORDS: usize = 8;
const TRANSCRIPT_STATE_WORDS: usize = 64;

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
    Fri,
    Decommit,
    Assemble,
}

impl ProofEpoch {
    pub const ALL: [Self; 10] = [
        Self::Ingest,
        Self::Witness,
        Self::BaseCommit,
        Self::Interaction,
        Self::InteractionCommit,
        Self::Composition,
        Self::CompositionCommit,
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BufferPurpose {
    BaseTrace,
    LookupInputs,
    SubcomponentInputs,
    InteractionTrace,
    Twiddles,
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
    QuotientTile,
    FriPing,
    FriPong,
    QueryIndices,
    DecommitValues,
    DecommitHashes,
    TranscriptState,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitmentGeometry {
    pub id: CommitmentTreeId,
    pub created: ProofEpoch,
    pub config: CommitWorkspaceConfig,
    pub grouped_column_log_sizes: Vec<Vec<u32>>,
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
    pub commitments: Vec<CommitmentGeometry>,
    /// Commitment-tree leaf log sizes opened by the PCS, in proof tree order.
    /// Persistent preprocessed trees have no per-proof prepared-commit workspace,
    /// but their authentication paths still require decommit capacity.
    pub opened_tree_log_sizes: Vec<u32>,
    /// Fully retained FRI Merkle tree leaf log sizes, in transcript order.
    pub fri_layer_log_sizes: Vec<u32>,
}

impl ProtocolGeometry {
    fn validate(&self) -> Result<(), ArenaPlanError> {
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
        feed(b"stwo-cairo-protocol-geometry-v1\0");
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

/// Exact prepared-commit inputs resolved to physical arena slots after liveness
/// coloring. The canonical log groups are retained so the caller can bind the
/// corresponding coefficient columns without rediscovering ordering.
#[derive(Clone, Debug)]
pub struct PlannedCommitment {
    pub id: CommitmentTreeId,
    pub config: CommitWorkspaceConfig,
    pub grouped_column_log_sizes: Vec<Vec<u32>>,
    pub requirements: CommitWorkspaceRequirements,
    pub twiddles: ArenaBinding,
    pub slots: CommitWorkspaceSlots,
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
        let logical_commitments = append_protocol_buffers(&mut logical, protocol)?;

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

        Ok(Self {
            shape_key: plan.shape_key,
            protocol_key: protocol.key(),
            logical,
            bindings,
            layout,
            high_water_words,
            commitments,
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
    SizeOverflow,
    EmptyLogicalBuffer(LogicalBufferId),
    MissingBinding(LogicalBufferId),
    AliasedLiveBuffers {
        physical: ArenaSlotId,
        first: LogicalBufferId,
        second: LogicalBufferId,
    },
    Commit(PreparedCommitError),
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

fn append_protocol_buffers(
    logical: &mut Vec<LogicalBuffer>,
    protocol: &ProtocolGeometry,
) -> Result<Vec<LogicalCommitWorkspace>, ArenaPlanError> {
    let domain_rows = checked_pow2(protocol.max_domain_log_size)?;
    let requirements = protocol
        .commitments
        .iter()
        .map(|commitment| {
            commit_workspace_requirements(commitment.config, &commitment.grouped_column_log_sizes)
                .map_err(ArenaPlanError::Commit)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let twiddle_words = requirements
        .iter()
        .map(|requirements| requirements.twiddle_words)
        .max()
        .ok_or(ArenaPlanError::InvalidProtocolGeometry(
            "proof has no commitment workspace",
        ))?;
    let twiddles = push_buffer_id(
        logical,
        None,
        None,
        BufferPurpose::Twiddles,
        0,
        twiddle_words,
        BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Fri)?,
    )?;

    let mut logical_commitments = Vec::with_capacity(protocol.commitments.len());
    for (commitment_index, (geometry, requirements)) in
        protocol.commitments.iter().zip(requirements).enumerate()
    {
        let at = BufferLifetime::at(geometry.created);
        let retained = BufferLifetime::new(geometry.created, ProofEpoch::Decommit)?;
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
                    at,
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
                    at,
                )?;
                let column_log_sizes = push_buffer_id(
                    logical,
                    None,
                    None,
                    BufferPurpose::CommitColumnLogSizes,
                    ordinal()?,
                    group.column_log_size_words,
                    at,
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
                                at,
                            )?,
                            coefficient_sizes: push_buffer_id(
                                logical,
                                None,
                                None,
                                BufferPurpose::CommitCoefficientSizes,
                                ordinal()?,
                                batch.coefficient_size_words,
                                at,
                            )?,
                            output_ptrs: push_buffer_id(
                                logical,
                                None,
                                None,
                                BufferPurpose::CommitOutputPointers,
                                ordinal()?,
                                batch.output_pointer_words,
                                at,
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
            requirements,
            twiddles,
            lde_tile,
            leaf_state,
            merkle_scratch,
            retained_layers,
            tail_level_ptrs,
            tail_outputs,
            groups,
        });
    }

    for (tree_index, &leaf_log_size) in protocol.fri_layer_log_sizes.iter().enumerate() {
        for log_size in 0..=leaf_log_size {
            let ordinal = u32::try_from(tree_index)
                .ok()
                .and_then(|tree| tree.checked_shl(16))
                .and_then(|tree| tree.checked_add(log_size))
                .ok_or(ArenaPlanError::SizeOverflow)?;
            push_buffer(
                logical,
                None,
                None,
                BufferPurpose::FriMerkleLayer,
                ordinal,
                checked_pow2(log_size)?
                    .checked_mul(BLAKE2S_HASH_WORDS)
                    .ok_or(ArenaPlanError::SizeOverflow)?,
                BufferLifetime::new(ProofEpoch::Fri, ProofEpoch::Decommit)?,
            )?;
        }
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
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::QuotientTile,
        0,
        secure_domain_words,
        BufferLifetime::new(ProofEpoch::Composition, ProofEpoch::Fri)?,
    )?;
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::FriPing,
        0,
        secure_domain_words,
        BufferLifetime::new(ProofEpoch::Fri, ProofEpoch::Decommit)?,
    )?;
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::FriPong,
        0,
        (domain_rows / 2)
            .checked_mul(4)
            .ok_or(ArenaPlanError::SizeOverflow)?,
        BufferLifetime::new(ProofEpoch::Fri, ProofEpoch::Decommit)?,
    )?;
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
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::TranscriptState,
        0,
        TRANSCRIPT_STATE_WORDS,
        BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble)?,
    )?;
    push_buffer(
        logical,
        None,
        None,
        BufferPurpose::ProofBytes,
        0,
        protocol.proof_capacity_words,
        BufferLifetime::at(ProofEpoch::Assemble),
    )?;
    Ok(logical_commitments)
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
        requirements: logical.requirements,
        twiddles: binding(logical.twiddles)?,
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
        let shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let proof =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
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
                    grouped_column_log_sizes: vec![vec![20; 16], vec![21; 3]],
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
                    grouped_column_log_sizes: vec![vec![20; 16], vec![22; 2]],
                },
            ],
            opened_tree_log_sizes: vec![26, 26, 26, 26],
            fri_layer_log_sizes: vec![26, 25, 24],
        };
        let arena = ProofArenaPlan::build(&proof, &protocol).unwrap();
        arena.validate_aliases().unwrap();
        assert_eq!(arena.protocol_key, protocol.key());
        assert_eq!(arena.shape_key, proof.shape_key);
        assert_eq!(arena.commitments().len(), 2);
        assert!(arena.commitment(CommitmentTreeId::Base).is_some());
        assert!(arena.fri_merkle_layer(0, 26).is_some());
        assert_eq!(arena.total_words() % ARENA_ALIGNMENT_WORDS, 0);
        assert!(arena.total_words() >= arena.high_water_words(ProofEpoch::Fri));
    }
}
