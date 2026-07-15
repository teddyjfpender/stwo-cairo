//! Address-free oracle for retaining commitment evaluations consumed directly
//! by composition. Production selection and device binding are intentionally
//! outside this module.

use std::collections::{BTreeSet, HashSet};

use stwo_backend_cuda::ArenaSlotId;
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use crate::arena_plan::{
    BufferLifetime, BufferPurpose, CommitmentGeometry, CommitmentTreeId, OodsColumnGeometry,
    OodsGeometry, OpenedColumnSource, ProofEpoch, ProtocolGeometry,
};
use crate::composition_plan::CompositionPlan;
use crate::prepared_composition::{
    composition_workspace_requirements_with_mode, CompositionCoefficientSource,
    CompositionLaunchMode, CompositionTraceTopology, PreparedCompositionError,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();

pub(crate) fn direct_bitmap_hash(words: &[u64]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in b"direct-composition-occurrence-bitmap-v1\0" {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for byte in words.iter().flat_map(|word| word.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum DirectCompositionRetentionMode {
    #[default]
    Disabled = 0,
    ExactNative = 1,
}

impl DirectCompositionRetentionMode {
    pub fn from_env() -> Self {
        static MODE: std::sync::OnceLock<DirectCompositionRetentionMode> =
            std::sync::OnceLock::new();
        *MODE.get_or_init(|| {
            if crate::flags::flag_on("STWO_CUDA_COMPOSITION_DIRECT_RETENTION") {
                Self::ExactNative
            } else {
                Self::Disabled
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectCompositionConsumer {
    pub source: OpenedColumnSource,
    pub evaluation_log_size: u32,
    pub force_direct: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectCompositionColumn {
    pub source: OpenedColumnSource,
    pub tree: CommitmentTreeId,
    pub proof_column: usize,
    pub group: usize,
    pub column_in_group: usize,
    pub canonical_column: usize,
    pub coefficient_log_size: u32,
    pub evaluation_log_size: u32,
    pub lifetime: BufferLifetime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectCompositionBinding {
    pub consumer: usize,
    pub column: usize,
    pub consumer_evaluation_log_size: u32,
    pub direct: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectCompositionBucket {
    pub tree: CommitmentTreeId,
    pub evaluation_log_size: u32,
    pub column_count: usize,
    pub bytes: usize,
    pub hash: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectCompositionRetentionPlan {
    pub columns: Vec<DirectCompositionColumn>,
    pub bindings: Vec<DirectCompositionBinding>,
    pub direct_bitmap: Vec<u64>,
    pub buckets: Vec<DirectCompositionBucket>,
    pub direct_column_count: usize,
    pub direct_bytes: usize,
    pub cache_key: u64,
}

impl DirectCompositionRetentionPlan {
    pub fn by_proof_column(
        &self,
        tree: CommitmentTreeId,
        proof_column: usize,
    ) -> Option<&DirectCompositionColumn> {
        self.columns
            .iter()
            .find(|column| column.tree == tree && column.proof_column == proof_column)
    }

    pub fn by_canonical_column(
        &self,
        tree: CommitmentTreeId,
        canonical_column: usize,
    ) -> Option<&DirectCompositionColumn> {
        self.columns
            .iter()
            .find(|column| column.tree == tree && column.canonical_column == canonical_column)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectCompositionRetentionError {
    DuplicateOodsSource(OpenedColumnSource),
    UnsupportedOodsSource(OpenedColumnSource),
    MissingCommitmentTree(CommitmentTreeId),
    AmbiguousCommitmentTree(CommitmentTreeId),
    WrongCommitmentTree {
        source: OpenedColumnSource,
        expected: CommitmentTreeId,
        actual: CommitmentTreeId,
    },
    MissingCommitmentSource(OpenedColumnSource),
    AmbiguousCommitmentSource(OpenedColumnSource),
    CommitmentGroupShape(CommitmentTreeId),
    SourceLogMismatch {
        source: OpenedColumnSource,
        oods: u32,
        commitment: u32,
    },
    EvaluationLogMismatch {
        source: OpenedColumnSource,
        oods: u32,
        native: u32,
    },
    MissingConsumerSource(OpenedColumnSource),
    ForcedDirectLogMismatch {
        consumer: usize,
        source: OpenedColumnSource,
        consumer_log: u32,
        native_log: u32,
    },
    InvalidLifetime {
        tree: CommitmentTreeId,
        first: ProofEpoch,
        last: ProofEpoch,
    },
    SizeOverflow,
    PlanDrift,
    CountDrift,
    BitmapDrift,
    CacheIdentityDrift,
    Composition(PreparedCompositionError),
    MissingProofColumn {
        tree: usize,
        column: usize,
    },
}

impl core::fmt::Display for DirectCompositionRetentionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid direct-composition retention plan: {self:?}")
    }
}

impl std::error::Error for DirectCompositionRetentionError {}

impl From<PreparedCompositionError> for DirectCompositionRetentionError {
    fn from(value: PreparedCompositionError) -> Self {
        Self::Composition(value)
    }
}

/// Derive composition consumers from the same mode-explicit requirements that
/// own the prepared component/source order. No AIR traversal or ambient launch
/// policy is duplicated here.
pub fn derive_direct_composition_consumers(
    oods: &OodsGeometry,
    composition: &CompositionPlan,
) -> Result<Vec<DirectCompositionConsumer>, DirectCompositionRetentionError> {
    let mut proof_sources = [Vec::new(), Vec::new(), Vec::new()];
    let mut trace_trees = vec![Vec::new(), Vec::new(), Vec::new()];
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
            source => {
                return Err(DirectCompositionRetentionError::UnsupportedOodsSource(
                    source,
                ));
            }
        };
        proof_sources[tree].push(column.source);
        trace_trees[tree].push(CompositionCoefficientSource {
            slot: ArenaSlotId(
                u32::try_from(flat)
                    .map_err(|_| DirectCompositionRetentionError::SizeOverflow)?
                    .checked_add(1)
                    .ok_or(DirectCompositionRetentionError::SizeOverflow)?,
            ),
            log_size: column.coefficient_log_size,
        });
    }
    let requirements = composition_workspace_requirements_with_mode(
        composition,
        &CompositionTraceTopology { trees: trace_trees },
        // Source selection and canonical consumer order are independent of
        // launch topology. Retention must be planned before Wave can prove
        // every source direct, so use the source-only Serial seam here.
        CompositionLaunchMode::Serial,
    )?;
    let mut consumers = Vec::new();
    for component in &requirements.components {
        for source in &component.sources {
            let opened = proof_sources
                .get(source.tree)
                .and_then(|tree| tree.get(source.column))
                .copied()
                .ok_or(DirectCompositionRetentionError::MissingProofColumn {
                    tree: source.tree,
                    column: source.column,
                })?;
            consumers.push(DirectCompositionConsumer {
                source: opened,
                evaluation_log_size: component.evaluation_log_size,
                force_direct: false,
            });
        }
    }
    Ok(consumers)
}

pub(crate) fn plan_direct_composition_retention_from_parts(
    commitments: &[CommitmentGeometry],
    oods_columns: &[OodsColumnGeometry],
    log_blowup_factor: u32,
    consumers: &[DirectCompositionConsumer],
) -> Result<DirectCompositionRetentionPlan, DirectCompositionRetentionError> {
    plan_topology(commitments, oods_columns, log_blowup_factor, consumers)
}

pub fn plan_direct_composition_retention(
    protocol: &ProtocolGeometry,
    consumers: &[DirectCompositionConsumer],
) -> Result<DirectCompositionRetentionPlan, DirectCompositionRetentionError> {
    plan_topology(
        &protocol.commitments,
        &protocol.oods.columns,
        protocol.identity.log_blowup_factor,
        consumers,
    )
}

pub fn validate_direct_composition_retention_plan(
    protocol: &ProtocolGeometry,
    consumers: &[DirectCompositionConsumer],
    plan: &DirectCompositionRetentionPlan,
) -> Result<(), DirectCompositionRetentionError> {
    let expected = plan_direct_composition_retention(protocol, consumers)?;
    validate_against(plan, &expected)
}

fn validate_against(
    actual: &DirectCompositionRetentionPlan,
    expected: &DirectCompositionRetentionPlan,
) -> Result<(), DirectCompositionRetentionError> {
    if actual.direct_bitmap != expected.direct_bitmap {
        return Err(DirectCompositionRetentionError::BitmapDrift);
    }
    if actual.buckets != expected.buckets
        || actual.direct_column_count != expected.direct_column_count
        || actual.direct_bytes != expected.direct_bytes
    {
        return Err(DirectCompositionRetentionError::CountDrift);
    }
    if actual.columns != expected.columns || actual.bindings != expected.bindings {
        return Err(DirectCompositionRetentionError::PlanDrift);
    }
    if actual.cache_key != expected.cache_key {
        return Err(DirectCompositionRetentionError::CacheIdentityDrift);
    }
    Ok(())
}

fn plan_topology(
    commitments: &[CommitmentGeometry],
    oods_columns: &[OodsColumnGeometry],
    log_blowup_factor: u32,
    consumers: &[DirectCompositionConsumer],
) -> Result<DirectCompositionRetentionPlan, DirectCompositionRetentionError> {
    for commitment in commitments.iter().filter(|commitment| {
        matches!(
            commitment.id,
            CommitmentTreeId::Preprocessed | CommitmentTreeId::Base | CommitmentTreeId::Interaction
        )
    }) {
        validate_commitment_groups(commitment)?;
    }
    let mut seen = HashSet::new();
    let mut proof_counts = [0usize; 3];
    let mut columns = Vec::new();

    for oods in oods_columns {
        if !seen.insert(oods.source) {
            return Err(DirectCompositionRetentionError::DuplicateOodsSource(
                oods.source,
            ));
        }
        let Some(tree) = relevant_tree(oods.source)? else {
            continue;
        };
        let tree_index = tree_index(tree).expect("relevant tree has an index");
        let proof_column = proof_counts[tree_index];
        proof_counts[tree_index] = proof_counts[tree_index]
            .checked_add(1)
            .ok_or(DirectCompositionRetentionError::SizeOverflow)?;
        let commitment = unique_commitment(commitments, tree)?;
        let matched = match_source(commitments, commitment, tree, oods.source)?;
        if matched.log_size != oods.coefficient_log_size {
            return Err(DirectCompositionRetentionError::SourceLogMismatch {
                source: oods.source,
                oods: oods.coefficient_log_size,
                commitment: matched.log_size,
            });
        }
        let evaluation_log_size = matched
            .log_size
            .checked_add(log_blowup_factor)
            .ok_or(DirectCompositionRetentionError::SizeOverflow)?;
        if oods.evaluation_log_size != evaluation_log_size {
            return Err(DirectCompositionRetentionError::EvaluationLogMismatch {
                source: oods.source,
                oods: oods.evaluation_log_size,
                native: evaluation_log_size,
            });
        }
        let lifetime =
            BufferLifetime::new(commitment.created, ProofEpoch::Composition).map_err(|_| {
                DirectCompositionRetentionError::InvalidLifetime {
                    tree,
                    first: commitment.created,
                    last: ProofEpoch::Composition,
                }
            })?;
        columns.push(DirectCompositionColumn {
            source: oods.source,
            tree,
            proof_column,
            group: matched.group,
            column_in_group: matched.column_in_group,
            canonical_column: matched.canonical_column,
            coefficient_log_size: matched.log_size,
            evaluation_log_size,
            lifetime,
        });
    }

    let mut bindings = Vec::with_capacity(consumers.len());
    let mut direct_bitmap = vec![0u64; consumers.len().div_ceil(64)];
    let mut retained = BTreeSet::new();
    for (consumer_index, consumer) in consumers.iter().enumerate() {
        let column_index = columns
            .iter()
            .position(|column| column.source == consumer.source)
            .ok_or(DirectCompositionRetentionError::MissingConsumerSource(
                consumer.source,
            ))?;
        let native_log = columns[column_index].evaluation_log_size;
        let direct = native_log == consumer.evaluation_log_size;
        if consumer.force_direct && !direct {
            return Err(DirectCompositionRetentionError::ForcedDirectLogMismatch {
                consumer: consumer_index,
                source: consumer.source,
                consumer_log: consumer.evaluation_log_size,
                native_log,
            });
        }
        if direct {
            direct_bitmap[consumer_index / 64] |= 1u64 << (consumer_index % 64);
            retained.insert(column_index);
        }
        bindings.push(DirectCompositionBinding {
            consumer: consumer_index,
            column: column_index,
            consumer_evaluation_log_size: consumer.evaluation_log_size,
            direct,
        });
    }

    let (buckets, direct_bytes) = buckets(&columns, &retained)?;
    let direct_column_count = retained.len();
    let mut plan = DirectCompositionRetentionPlan {
        columns,
        bindings,
        direct_bitmap,
        buckets,
        direct_column_count,
        direct_bytes,
        cache_key: 0,
    };
    plan.cache_key = plan_hash(&plan);
    Ok(plan)
}

#[derive(Clone, Copy)]
struct SourceMatch {
    group: usize,
    column_in_group: usize,
    canonical_column: usize,
    log_size: u32,
}

fn unique_commitment(
    commitments: &[CommitmentGeometry],
    tree: CommitmentTreeId,
) -> Result<&CommitmentGeometry, DirectCompositionRetentionError> {
    let mut matches = commitments
        .iter()
        .filter(|commitment| commitment.id == tree);
    let commitment = matches
        .next()
        .ok_or(DirectCompositionRetentionError::MissingCommitmentTree(tree))?;
    if matches.next().is_some() {
        return Err(DirectCompositionRetentionError::AmbiguousCommitmentTree(
            tree,
        ));
    }
    Ok(commitment)
}

fn match_source(
    commitments: &[CommitmentGeometry],
    commitment: &CommitmentGeometry,
    tree: CommitmentTreeId,
    source: OpenedColumnSource,
) -> Result<SourceMatch, DirectCompositionRetentionError> {
    validate_commitment_groups(commitment)?;
    let mut canonical_column = 0usize;
    let mut selected = None;
    for (group, (sources, logs)) in commitment
        .grouped_column_sources
        .iter()
        .zip(&commitment.grouped_column_log_sizes)
        .enumerate()
    {
        for (column_in_group, (&candidate, &log_size)) in sources.iter().zip(logs).enumerate() {
            if (group, column_in_group) != (canonical_column / 16, canonical_column % 16) {
                return Err(DirectCompositionRetentionError::CommitmentGroupShape(tree));
            }
            if OpenedColumnSource::from(candidate) == source {
                if selected.is_some() {
                    return Err(DirectCompositionRetentionError::AmbiguousCommitmentSource(
                        source,
                    ));
                }
                selected = Some(SourceMatch {
                    group,
                    column_in_group,
                    canonical_column,
                    log_size,
                });
            }
            canonical_column = canonical_column
                .checked_add(1)
                .ok_or(DirectCompositionRetentionError::SizeOverflow)?;
        }
    }
    if let Some(selected) = selected {
        return Ok(selected);
    }
    for other in commitments.iter().filter(|other| other.id != tree) {
        if other
            .grouped_column_sources
            .iter()
            .flatten()
            .copied()
            .map(OpenedColumnSource::from)
            .any(|candidate| candidate == source)
        {
            return Err(DirectCompositionRetentionError::WrongCommitmentTree {
                source,
                expected: tree,
                actual: other.id,
            });
        }
    }
    Err(DirectCompositionRetentionError::MissingCommitmentSource(
        source,
    ))
}

fn validate_commitment_groups(
    commitment: &CommitmentGeometry,
) -> Result<(), DirectCompositionRetentionError> {
    let group_count = commitment.grouped_column_sources.len();
    if group_count == 0 || group_count != commitment.grouped_column_log_sizes.len() {
        return Err(DirectCompositionRetentionError::CommitmentGroupShape(
            commitment.id,
        ));
    }
    for (group, (sources, logs)) in commitment
        .grouped_column_sources
        .iter()
        .zip(&commitment.grouped_column_log_sizes)
        .enumerate()
    {
        let width = sources.len();
        if width == 0
            || width > 16
            || width != logs.len()
            || (group + 1 != group_count && width != 16)
        {
            return Err(DirectCompositionRetentionError::CommitmentGroupShape(
                commitment.id,
            ));
        }
    }
    Ok(())
}

fn relevant_tree(
    source: OpenedColumnSource,
) -> Result<Option<CommitmentTreeId>, DirectCompositionRetentionError> {
    match source {
        OpenedColumnSource::Preprocessed { .. } => Ok(Some(CommitmentTreeId::Preprocessed)),
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::BaseCoefficients,
            ..
        } => Ok(Some(CommitmentTreeId::Base)),
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::InteractionCoefficients,
            ..
        } => Ok(Some(CommitmentTreeId::Interaction)),
        OpenedColumnSource::Composition { .. } => Ok(None),
        source => Err(DirectCompositionRetentionError::UnsupportedOodsSource(
            source,
        )),
    }
}

fn tree_index(tree: CommitmentTreeId) -> Option<usize> {
    match tree {
        CommitmentTreeId::Preprocessed => Some(0),
        CommitmentTreeId::Base => Some(1),
        CommitmentTreeId::Interaction => Some(2),
        CommitmentTreeId::Composition | CommitmentTreeId::Fri(_) => None,
    }
}

fn buckets(
    columns: &[DirectCompositionColumn],
    retained: &BTreeSet<usize>,
) -> Result<(Vec<DirectCompositionBucket>, usize), DirectCompositionRetentionError> {
    let mut keys = BTreeSet::new();
    for &column in retained {
        let column = columns
            .get(column)
            .ok_or(DirectCompositionRetentionError::PlanDrift)?;
        keys.insert((
            tree_index(column.tree).expect("retained tree has an index"),
            column.evaluation_log_size,
        ));
    }
    let mut total_bytes = 0usize;
    let mut buckets = Vec::with_capacity(keys.len());
    for (tree_index, evaluation_log_size) in keys {
        let tree = [
            CommitmentTreeId::Preprocessed,
            CommitmentTreeId::Base,
            CommitmentTreeId::Interaction,
        ][tree_index];
        let selected = retained.iter().copied().filter(|&index| {
            columns[index].tree == tree && columns[index].evaluation_log_size == evaluation_log_size
        });
        let mut column_count = 0usize;
        let mut hash = 0xcbf29ce484222325u64;
        feed(&mut hash, b"direct-composition-retention-bucket-v1\0");
        feed_tree(&mut hash, tree);
        feed(&mut hash, &evaluation_log_size.to_le_bytes());
        for index in selected {
            column_count = column_count
                .checked_add(1)
                .ok_or(DirectCompositionRetentionError::SizeOverflow)?;
            feed_column(&mut hash, &columns[index]);
        }
        let column_bytes = words_for_log(evaluation_log_size)?
            .checked_mul(WORD_BYTES)
            .ok_or(DirectCompositionRetentionError::SizeOverflow)?;
        let bytes = column_count
            .checked_mul(column_bytes)
            .ok_or(DirectCompositionRetentionError::SizeOverflow)?;
        feed(&mut hash, &(column_count as u64).to_le_bytes());
        feed(&mut hash, &(bytes as u64).to_le_bytes());
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or(DirectCompositionRetentionError::SizeOverflow)?;
        buckets.push(DirectCompositionBucket {
            tree,
            evaluation_log_size,
            column_count,
            bytes,
            hash,
        });
    }
    Ok((buckets, total_bytes))
}

fn words_for_log(log_size: u32) -> Result<usize, DirectCompositionRetentionError> {
    1usize
        .checked_shl(log_size)
        .ok_or(DirectCompositionRetentionError::SizeOverflow)
}

fn plan_hash(plan: &DirectCompositionRetentionPlan) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    feed(&mut hash, b"direct-composition-retention-plan-v1\0");
    feed(&mut hash, &(plan.columns.len() as u64).to_le_bytes());
    for column in &plan.columns {
        feed_column(&mut hash, column);
    }
    feed(&mut hash, &(plan.bindings.len() as u64).to_le_bytes());
    for binding in &plan.bindings {
        feed(&mut hash, &(binding.consumer as u64).to_le_bytes());
        feed(&mut hash, &(binding.column as u64).to_le_bytes());
        feed(
            &mut hash,
            &binding.consumer_evaluation_log_size.to_le_bytes(),
        );
        feed(&mut hash, &[u8::from(binding.direct)]);
    }
    for word in &plan.direct_bitmap {
        feed(&mut hash, &word.to_le_bytes());
    }
    for bucket in &plan.buckets {
        feed_tree(&mut hash, bucket.tree);
        feed(&mut hash, &bucket.evaluation_log_size.to_le_bytes());
        feed(&mut hash, &(bucket.column_count as u64).to_le_bytes());
        feed(&mut hash, &(bucket.bytes as u64).to_le_bytes());
        feed(&mut hash, &bucket.hash.to_le_bytes());
    }
    feed(&mut hash, &(plan.direct_column_count as u64).to_le_bytes());
    feed(&mut hash, &(plan.direct_bytes as u64).to_le_bytes());
    hash
}

pub fn direct_composition_plan_key(plan: &DirectCompositionRetentionPlan) -> u64 {
    plan_hash(plan)
}

fn feed_column(hash: &mut u64, column: &DirectCompositionColumn) {
    feed_source(hash, column.source);
    feed_tree(hash, column.tree);
    for value in [
        column.proof_column,
        column.group,
        column.column_in_group,
        column.canonical_column,
    ] {
        feed(hash, &(value as u64).to_le_bytes());
    }
    feed(hash, &column.coefficient_log_size.to_le_bytes());
    feed(hash, &column.evaluation_log_size.to_le_bytes());
    feed(
        hash,
        &[column.lifetime.first as u8, column.lifetime.last as u8],
    );
}

fn feed_source(hash: &mut u64, source: OpenedColumnSource) {
    match source {
        OpenedColumnSource::Preprocessed { ordinal } => {
            feed(hash, &[0]);
            feed(hash, &ordinal.to_le_bytes());
        }
        OpenedColumnSource::Trace {
            component,
            part,
            purpose,
            ordinal,
        } => {
            feed(hash, &[1]);
            feed(hash, component.as_bytes());
            feed(hash, &[0]);
            match part {
                TracePartId::Main => feed(hash, &[0]),
                TracePartId::MemoryBig(index) => {
                    feed(hash, &[1]);
                    feed(hash, &index.to_le_bytes());
                }
                TracePartId::MemorySmall => feed(hash, &[2]),
            }
            feed(
                hash,
                &[match purpose {
                    BufferPurpose::BaseCoefficients => 0,
                    BufferPurpose::InteractionCoefficients => 1,
                    _ => u8::MAX,
                }],
            );
            feed(hash, &ordinal.to_le_bytes());
        }
        OpenedColumnSource::Composition { ordinal } => {
            feed(hash, &[2]);
            feed(hash, &ordinal.to_le_bytes());
        }
    }
}

fn feed_tree(hash: &mut u64, tree: CommitmentTreeId) {
    match tree {
        CommitmentTreeId::Preprocessed => feed(hash, &[0]),
        CommitmentTreeId::Base => feed(hash, &[1]),
        CommitmentTreeId::Interaction => feed(hash, &[2]),
        CommitmentTreeId::Composition => feed(hash, &[3]),
        CommitmentTreeId::Fri(index) => feed(hash, &[4, index]),
    }
}

fn feed(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::BaseField;
    use stwo::core::pcs::TreeSubspan;
    use stwo_backend_cuda::{CommitWorkspaceConfig, TranscriptInputId, TranscriptOutputId};
    use stwo_cairo_prover::witness::proof_shape::TracePartId;

    use super::*;
    use crate::arena_plan::CommitmentColumnSource;
    use crate::composition_plan::{CompositionComponentPlan, CompositionKernelPart};

    fn source(ordinal: u32, purpose: BufferPurpose) -> CommitmentColumnSource {
        CommitmentColumnSource::Trace {
            component: "component",
            part: TracePartId::Main,
            purpose,
            ordinal,
        }
    }

    fn oods_with_eval(
        source: OpenedColumnSource,
        coefficient_log_size: u32,
        evaluation_log_size: u32,
    ) -> OodsColumnGeometry {
        OodsColumnGeometry {
            source,
            coefficient_log_size,
            evaluation_log_size,
            shape_points: Vec::new(),
            offset_points: Vec::new(),
        }
    }

    fn oods(source: OpenedColumnSource, log: u32) -> OodsColumnGeometry {
        oods_with_eval(source, log, log + 1)
    }

    fn commitment(
        id: CommitmentTreeId,
        created: ProofEpoch,
        columns: Vec<(CommitmentColumnSource, u32)>,
    ) -> CommitmentGeometry {
        CommitmentGeometry {
            id,
            created,
            config: CommitWorkspaceConfig {
                lifting_log_size: 12,
                log_blowup_factor: 1,
                unretained_bottom_layers: 0,
                max_fused_tail_levels: 0,
            },
            grouped_column_log_sizes: columns
                .chunks(16)
                .map(|group| group.iter().map(|column| column.1).collect())
                .collect(),
            grouped_column_sources: columns
                .chunks(16)
                .map(|group| group.iter().map(|column| column.0).collect())
                .collect(),
            retained_evaluation_groups: Vec::new(),
            direct_composition_evaluation_groups: Vec::new(),
            numerator_evaluation_groups: Vec::new(),
        }
    }

    fn topology_with_count(count: u32) -> (Vec<CommitmentGeometry>, Vec<OodsColumnGeometry>) {
        let mut proof_columns = (0..count)
            .map(|ordinal| (ordinal, if ordinal % 2 == 0 { 5 } else { 6 }))
            .collect::<Vec<_>>();
        let oods_columns = proof_columns
            .iter()
            .map(|&(ordinal, log)| {
                oods(
                    OpenedColumnSource::from(source(ordinal, BufferPurpose::BaseCoefficients)),
                    log,
                )
            })
            .collect();
        proof_columns.sort_by_key(|&(_, log)| log);
        let base = commitment(
            CommitmentTreeId::Base,
            ProofEpoch::BaseCommit,
            proof_columns
                .into_iter()
                .map(|(ordinal, log)| (source(ordinal, BufferPurpose::BaseCoefficients), log))
                .collect(),
        );
        (vec![base], oods_columns)
    }

    fn topology() -> (Vec<CommitmentGeometry>, Vec<OodsColumnGeometry>) {
        topology_with_count(18)
    }

    fn plan(
        commitments: &[CommitmentGeometry],
        oods: &[OodsColumnGeometry],
        consumers: &[DirectCompositionConsumer],
    ) -> Result<DirectCompositionRetentionPlan, DirectCompositionRetentionError> {
        plan_topology(commitments, oods, 1, consumers)
    }

    #[test]
    fn group_boundaries_and_mixed_log_mapping_roundtrip() {
        for (count, expected_group, expected_in_group) in [(15, 0, 14), (16, 0, 15), (17, 1, 0)] {
            let (commitments, oods) = topology_with_count(count);
            let plan = plan(&commitments, &oods, &[]).unwrap();
            let last = plan
                .by_canonical_column(CommitmentTreeId::Base, count as usize - 1)
                .unwrap();
            assert_eq!(
                (last.group, last.column_in_group),
                (expected_group, expected_in_group)
            );
            assert!(plan.columns.iter().all(|column| {
                (column.group, column.column_in_group)
                    == (column.canonical_column / 16, column.canonical_column % 16)
            }));
        }

        let (commitments, oods) = topology();
        let plan = plan(&commitments, &oods, &[]).unwrap();
        assert_eq!(plan.columns.len(), 18);
        assert_eq!(plan.columns[1].canonical_column, 9);
        assert_eq!(plan.columns[16].canonical_column, 8);
        assert_eq!(
            (plan.columns[17].group, plan.columns[17].column_in_group),
            (1, 1)
        );
        for column in &plan.columns {
            assert_eq!(
                plan.by_proof_column(column.tree, column.proof_column),
                Some(column)
            );
            assert_eq!(
                plan.by_canonical_column(column.tree, column.canonical_column),
                Some(column)
            );
            assert_eq!(column.lifetime.first, ProofEpoch::BaseCommit);
            assert_eq!(column.lifetime.last, ProofEpoch::Composition);
        }
    }

    #[test]
    fn maps_preprocessed_and_interaction_with_inclusive_lifetimes() {
        let preprocessed_source = CommitmentColumnSource::Preprocessed { ordinal: 0 };
        let interaction_source = source(0, BufferPurpose::InteractionCoefficients);
        let commitments = [
            commitment(
                CommitmentTreeId::Preprocessed,
                ProofEpoch::Ingest,
                vec![(preprocessed_source, 4)],
            ),
            commitment(
                CommitmentTreeId::Interaction,
                ProofEpoch::InteractionCommit,
                vec![(interaction_source, 6)],
            ),
        ];
        let oods = [
            oods(OpenedColumnSource::from(preprocessed_source), 4),
            oods(OpenedColumnSource::from(interaction_source), 6),
        ];
        let plan = plan(&commitments, &oods, &[]).unwrap();
        assert_eq!(plan.columns[0].tree, CommitmentTreeId::Preprocessed);
        assert_eq!(plan.columns[0].proof_column, 0);
        assert_eq!(plan.columns[0].lifetime.first, ProofEpoch::Ingest);
        assert_eq!(plan.columns[1].tree, CommitmentTreeId::Interaction);
        assert_eq!(plan.columns[1].proof_column, 0);
        assert_eq!(
            plan.columns[1].lifetime.first,
            ProofEpoch::InteractionCommit
        );
        assert!(plan
            .columns
            .iter()
            .all(|column| column.lifetime.last == ProofEpoch::Composition));
    }

    #[test]
    fn shared_sources_deduplicate_and_consumers_mix_direct_with_fallback() {
        let (commitments, oods) = topology();
        let shared = oods[0].source;
        let consumers = [
            DirectCompositionConsumer {
                source: shared,
                evaluation_log_size: 6,
                force_direct: false,
            },
            DirectCompositionConsumer {
                source: shared,
                evaluation_log_size: 7,
                force_direct: false,
            },
            DirectCompositionConsumer {
                source: oods[1].source,
                evaluation_log_size: 7,
                force_direct: true,
            },
        ];
        let plan = plan(&commitments, &oods, &consumers).unwrap();
        assert_eq!(plan.columns.len(), 18);
        assert_eq!(plan.bindings[0].column, plan.bindings[1].column);
        assert_eq!(plan.direct_bitmap, vec![0b101]);
        assert_eq!(plan.direct_column_count, 2);
        assert_eq!(plan.buckets.len(), 2);
        assert!(plan.buckets.iter().all(|bucket| {
            bucket.tree == CommitmentTreeId::Base && bucket.column_count == 1 && bucket.hash != 0
        }));
        assert_eq!(plan.direct_bytes, (1usize << 6) * 4 + (1usize << 7) * 4);
    }

    #[test]
    fn bitmap_boundaries_keep_duplicate_consumers_on_one_retained_column() {
        let (commitments, oods) = topology_with_count(1);
        for (count, expected) in [
            (63, vec![u64::MAX >> 1]),
            (64, vec![u64::MAX]),
            (65, vec![u64::MAX, 1]),
        ] {
            let consumers = vec![
                DirectCompositionConsumer {
                    source: oods[0].source,
                    evaluation_log_size: 6,
                    force_direct: false,
                };
                count
            ];
            let plan = plan(&commitments, &oods, &consumers).unwrap();
            assert_eq!(plan.direct_bitmap, expected);
            assert_eq!(plan.direct_column_count, 1);
            assert_eq!(plan.buckets[0].column_count, 1);
            assert!(plan.bindings.iter().all(|binding| binding.column == 0));
        }
    }

    #[test]
    fn rejects_malformed_commitment_group_widths() {
        let (valid, oods) = topology_with_count(16);
        let mut malformed = Vec::new();

        let mut no_groups = valid[0].clone();
        no_groups.grouped_column_sources.clear();
        no_groups.grouped_column_log_sizes.clear();
        malformed.push(no_groups);

        let mut short_non_final = valid[0].clone();
        let final_source = short_non_final.grouped_column_sources[0].split_off(15);
        let final_log = short_non_final.grouped_column_log_sizes[0].split_off(15);
        short_non_final.grouped_column_sources.push(final_source);
        short_non_final.grouped_column_log_sizes.push(final_log);
        malformed.push(short_non_final);

        let (overwide, _) = topology_with_count(17);
        let mut overwide = overwide[0].clone();
        let tail_sources = overwide.grouped_column_sources.pop().unwrap();
        let tail_logs = overwide.grouped_column_log_sizes.pop().unwrap();
        overwide.grouped_column_sources[0].extend(tail_sources);
        overwide.grouped_column_log_sizes[0].extend(tail_logs);
        malformed.push(overwide);

        let mut empty_final = valid[0].clone();
        empty_final.grouped_column_sources.push(Vec::new());
        empty_final.grouped_column_log_sizes.push(Vec::new());
        malformed.push(empty_final);

        let mut unequal = valid[0].clone();
        unequal.grouped_column_log_sizes[0].pop();
        malformed.push(unequal);

        for commitment in malformed {
            assert_eq!(
                plan(&[commitment], &oods, &[]),
                Err(DirectCompositionRetentionError::CommitmentGroupShape(
                    CommitmentTreeId::Base
                ))
            );
        }
    }

    #[test]
    fn rejects_source_tree_and_log_drift() {
        let (mut commitments, mut oods) = topology();
        oods.push(oods[0].clone());
        assert!(matches!(
            plan(&commitments, &oods, &[]),
            Err(DirectCompositionRetentionError::DuplicateOodsSource(_))
        ));
        oods.pop();

        let saved = commitments[0].grouped_column_sources[0][0];
        commitments[0].grouped_column_sources[0][0] = source(99, BufferPurpose::BaseCoefficients);
        assert!(matches!(
            plan(&commitments, &oods, &[]),
            Err(DirectCompositionRetentionError::MissingCommitmentSource(_))
        ));
        commitments[0].grouped_column_sources[0][0] = saved;

        let displaced = commitments[0].grouped_column_sources[0][1];
        commitments[0].grouped_column_sources[0][1] = saved;
        assert!(matches!(
            plan(&commitments, &oods, &[]),
            Err(DirectCompositionRetentionError::AmbiguousCommitmentSource(
                _
            ))
        ));
        commitments[0].grouped_column_sources[0][1] = displaced;

        oods[0].coefficient_log_size += 1;
        assert!(matches!(
            plan(&commitments, &oods, &[]),
            Err(DirectCompositionRetentionError::SourceLogMismatch { .. })
        ));
        oods[0].coefficient_log_size -= 1;

        oods[0].evaluation_log_size += 1;
        assert!(matches!(
            plan(&commitments, &oods, &[]),
            Err(DirectCompositionRetentionError::EvaluationLogMismatch { .. })
        ));
        oods[0].evaluation_log_size -= 1;

        let unsupported = OpenedColumnSource::from(source(0, BufferPurpose::BaseTrace));
        let mut unsupported_oods = oods.clone();
        unsupported_oods[0].source = unsupported;
        assert_eq!(
            plan(&commitments, &unsupported_oods, &[]),
            Err(DirectCompositionRetentionError::UnsupportedOodsSource(
                unsupported
            ))
        );

        let duplicate_tree = commitments[0].clone();
        assert_eq!(
            plan(&[commitments[0].clone(), duplicate_tree], &oods, &[]),
            Err(DirectCompositionRetentionError::AmbiguousCommitmentTree(
                CommitmentTreeId::Base
            ))
        );

        let mut misplaced = commitments[0].clone();
        misplaced.id = CommitmentTreeId::Interaction;
        assert_eq!(
            plan(&[misplaced.clone()], &oods, &[]),
            Err(DirectCompositionRetentionError::MissingCommitmentTree(
                CommitmentTreeId::Base
            ))
        );
        let unrelated_base = commitment(
            CommitmentTreeId::Base,
            ProofEpoch::BaseCommit,
            vec![(source(99, BufferPurpose::BaseCoefficients), 5)],
        );
        assert!(matches!(
            plan(&[unrelated_base, misplaced], &oods, &[]),
            Err(DirectCompositionRetentionError::WrongCommitmentTree {
                expected: CommitmentTreeId::Base,
                actual: CommitmentTreeId::Interaction,
                ..
            })
        ));

        let forced = [DirectCompositionConsumer {
            source: oods[0].source,
            evaluation_log_size: 7,
            force_direct: true,
        }];
        assert!(matches!(
            plan(&commitments, &oods, &forced),
            Err(DirectCompositionRetentionError::ForcedDirectLogMismatch { .. })
        ));
    }

    #[test]
    fn rejects_missing_consumers_and_unrepresentable_sizes() {
        let (commitments, oods) = topology();
        let missing = source(999, BufferPurpose::BaseCoefficients).into();
        assert_eq!(
            plan(
                &commitments,
                &oods,
                &[DirectCompositionConsumer {
                    source: missing,
                    evaluation_log_size: 6,
                    force_direct: false,
                }]
            ),
            Err(DirectCompositionRetentionError::MissingConsumerSource(
                missing
            ))
        );

        for (coefficient_log, evaluation_log) in [(62, 63), (63, 64)] {
            let column_source = source(0, BufferPurpose::BaseCoefficients);
            let large_commitment = commitment(
                CommitmentTreeId::Base,
                ProofEpoch::BaseCommit,
                vec![(column_source, coefficient_log)],
            );
            let large_oods = oods_with_eval(column_source.into(), coefficient_log, evaluation_log);
            assert_eq!(
                plan(
                    &[large_commitment],
                    &[large_oods],
                    &[DirectCompositionConsumer {
                        source: column_source.into(),
                        evaluation_log_size: evaluation_log,
                        force_direct: true,
                    }]
                ),
                Err(DirectCompositionRetentionError::SizeOverflow)
            );
        }

        let overflow_source = source(0, BufferPurpose::BaseCoefficients);
        assert_eq!(
            plan_topology(
                &[commitment(
                    CommitmentTreeId::Base,
                    ProofEpoch::BaseCommit,
                    vec![(overflow_source, u32::MAX)],
                )],
                &[oods_with_eval(overflow_source.into(), u32::MAX, 0)],
                1,
                &[]
            ),
            Err(DirectCompositionRetentionError::SizeOverflow)
        );
    }

    #[test]
    fn detects_plan_count_bitmap_cache_and_real_topology_drift() {
        let (commitments, oods) = topology();
        let consumers = [DirectCompositionConsumer {
            source: oods[0].source,
            evaluation_log_size: 6,
            force_direct: false,
        }];
        let expected = plan(&commitments, &oods, &consumers).unwrap();
        assert_eq!(expected, plan(&commitments, &oods, &consumers).unwrap());
        let fallback = plan(
            &commitments,
            &oods,
            &[DirectCompositionConsumer {
                evaluation_log_size: 7,
                ..consumers[0]
            }],
        )
        .unwrap();
        assert_ne!(expected.cache_key, fallback.cache_key);

        let mut source_commitments = commitments.clone();
        let mut source_oods = oods.clone();
        let replacement = source(100, BufferPurpose::BaseCoefficients);
        source_commitments[0].grouped_column_sources[0][0] = replacement;
        source_oods[0].source = replacement.into();
        let source_changed = plan(
            &source_commitments,
            &source_oods,
            &[DirectCompositionConsumer {
                source: replacement.into(),
                ..consumers[0]
            }],
        )
        .unwrap();
        assert_ne!(expected.cache_key, source_changed.cache_key);
        assert_ne!(expected.buckets[0].hash, source_changed.buckets[0].hash);

        let mut log_commitments = commitments.clone();
        let mut log_oods = oods.clone();
        log_commitments[0].grouped_column_log_sizes[0][0] = 6;
        log_oods[0].coefficient_log_size = 6;
        log_oods[0].evaluation_log_size = 7;
        let log_changed = plan(
            &log_commitments,
            &log_oods,
            &[DirectCompositionConsumer {
                evaluation_log_size: 7,
                ..consumers[0]
            }],
        )
        .unwrap();
        assert_ne!(expected.cache_key, log_changed.cache_key);
        assert_ne!(expected.buckets[0], log_changed.buckets[0]);

        let mut drift = expected.clone();
        drift.direct_bitmap[0] = 0;
        assert_eq!(
            validate_against(&drift, &expected),
            Err(DirectCompositionRetentionError::BitmapDrift)
        );
        drift = expected.clone();
        drift.buckets[0].column_count += 1;
        assert_eq!(
            validate_against(&drift, &expected),
            Err(DirectCompositionRetentionError::CountDrift)
        );
        drift = expected.clone();
        drift.columns[0].canonical_column += 1;
        assert_eq!(
            validate_against(&drift, &expected),
            Err(DirectCompositionRetentionError::PlanDrift)
        );
        drift = expected.clone();
        drift.cache_key ^= 1;
        assert_eq!(
            validate_against(&drift, &expected),
            Err(DirectCompositionRetentionError::CacheIdentityDrift)
        );
    }

    #[test]
    fn consumers_follow_baseline_component_and_source_order_exactly() {
        let sources = [
            (0..3)
                .map(|ordinal| OpenedColumnSource::Preprocessed { ordinal })
                .collect::<Vec<_>>(),
            (0..4)
                .map(|ordinal| source(ordinal, BufferPurpose::BaseCoefficients).into())
                .collect(),
            (0..3)
                .map(|ordinal| source(ordinal, BufferPurpose::InteractionCoefficients).into())
                .collect(),
        ];
        let oods = OodsGeometry {
            mask_log_size: 7,
            sampled_values_input: TranscriptInputId(1),
            point_parameter_output: TranscriptOutputId(2),
            quotient_random_coefficient_output: TranscriptOutputId(3),
            columns: sources
                .iter()
                .flatten()
                .copied()
                .map(|source| oods(source, 4))
                .collect(),
        };
        let component = |name: &'static str,
                         eval_log: u32,
                         offset: usize,
                         preprocessed: Vec<usize>,
                         base: core::ops::Range<usize>,
                         interaction: core::ops::Range<usize>| {
            CompositionComponentPlan {
                component: name,
                instance: 0,
                trace_locations: vec![
                    TreeSubspan {
                        tree_index: 0,
                        col_start: 0,
                        col_end: 0,
                    },
                    TreeSubspan {
                        tree_index: 1,
                        col_start: base.start,
                        col_end: base.end,
                    },
                    TreeSubspan {
                        tree_index: 2,
                        col_start: interaction.start,
                        col_end: interaction.end,
                    },
                ],
                preprocessed_column_indices: preprocessed,
                trace_log_size: 4,
                evaluation_log_size: eval_log,
                n_constraints: 1,
                random_coefficient_offset: offset,
                denominator_inverses: vec![BaseField::from(1); 1 << (eval_log - 4)],
                base_param_values: Vec::new(),
                ext_param_values: Vec::new(),
                ext_param_sources: Vec::new(),
                kernels: vec![CompositionKernelPart {
                    kernel_name: "kernel".to_owned(),
                    cache_key: 7,
                    semantic_hash: 9,
                    source: "kernel".to_owned(),
                    rc_base: 0,
                }],
            }
        };
        let composition = CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 2,
            max_evaluation_log_size: 7,
            components: vec![
                component("a", 7, 0, vec![2, 0], 1..4, 0..2),
                component("b", 6, 1, vec![1], 0..2, 1..3),
            ],
            wave_kernels: Vec::new(),
        };
        let consumers = derive_direct_composition_consumers(&oods, &composition).unwrap();
        let expected = [
            sources[0][2],
            sources[0][0],
            sources[1][1],
            sources[1][2],
            sources[1][3],
            sources[2][0],
            sources[2][1],
            sources[0][1],
            sources[1][0],
            sources[1][1],
            sources[2][1],
            sources[2][2],
        ];
        assert_eq!(
            consumers
                .iter()
                .map(|consumer| consumer.source)
                .collect::<Vec<_>>(),
            expected
        );
        assert!(consumers[..7]
            .iter()
            .all(|consumer| consumer.evaluation_log_size == 7));
        assert!(consumers[7..]
            .iter()
            .all(|consumer| consumer.evaluation_log_size == 6));
    }
}
