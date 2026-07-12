//! Exact PCS/FRI geometry derived from one Cairo claim.
//!
//! This is the protocol half of the proof-plan IR.  It contains no CUDA calls:
//! the claim, fixed-table binding, PCS security parameters, commitment order and
//! FRI folding schedule are validated before a device arena is allocated.

use cairo_air::claims::CairoClaim;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::{
    fri_workspace_requirements, CommitWorkspaceConfig, FriWorkspaceConfig, PreparedFriError,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedTrace, PreProcessedTraceVariant,
};
use stwo_cairo_prover::witness::proof_shape::{RowResolution, TracePartId, TracePartShape};

use crate::arena_plan::{
    BufferPurpose, CommitmentColumnSource, CommitmentGeometry, CommitmentTreeId, DecommitStrategy,
    OodsColumnGeometry, OodsGeometry, OpenedColumnSource, ProofEpoch, ProtocolGeometry,
    ProtocolIdentity, QuotientGeometry, TranscriptGeometry,
};
use crate::composition_plan::CompositionPlan;
use crate::direct_composition_retention::{
    derive_direct_composition_consumers, direct_bitmap_hash,
    plan_direct_composition_retention_from_parts, DirectCompositionRetentionError,
    DirectCompositionRetentionMode, DirectCompositionRetentionPlan,
};
use crate::plan::ProofPlan;
use crate::protocol_discovery::ProtocolTranscriptDiscovery;
use crate::relation::RelationTracePart;
use crate::relation_execution::{RelationExecutionError, RelationExecutionPlan};
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::schedule::TraceColumnCount;
use crate::schedule_table::CAIRO_COMMITMENT_COMPONENT_ORDER;
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptInput, CairoTranscriptOutput,
};

/// Stable identity for the ordinary Blake2s channel/proof format used by the
/// Starknet block benchmark.  It is intentionally not a Rust `TypeId` or hash.
pub const BLAKE2S_MERKLE_CHANNEL_TAG: u64 = 0x424c_414b_4532_5331;

/// Runtime choices that affect graph topology or opening residency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProtocolPlanPolicy {
    pub channel_tag: u64,
    /// Hash of the AOT manifest actually loaded by the runtime.  Zero is never
    /// accepted: graph reuse without a bound kernel library is unsafe.
    pub kernel_manifest_hash: u64,
    pub composition_max_kernel_instrs: usize,
    pub decommit_strategy: DecommitStrategy,
    /// Maximum persistent canonical LDE storage selected by the hybrid opener.
    pub retained_lde_budget_bytes: usize,
    pub unretained_bottom_layers: u32,
    pub max_fused_tail_levels: u32,
    pub commit_mode: stwo_backend_cuda::ProgressiveCommitMode,
    pub direct_composition_retention_mode: DirectCompositionRetentionMode,
}

impl ProtocolPlanPolicy {
    pub const fn starknet_blake2s(
        kernel_manifest_hash: u64,
        composition_max_kernel_instrs: usize,
    ) -> Self {
        Self {
            channel_tag: BLAKE2S_MERKLE_CHANNEL_TAG,
            kernel_manifest_hash,
            composition_max_kernel_instrs,
            decommit_strategy: DecommitStrategy::HybridByGroup,
            retained_lde_budget_bytes: 8 * 1024 * 1024 * 1024,
            unretained_bottom_layers: 4,
            max_fused_tail_levels: 12,
            commit_mode: stwo_backend_cuda::ProgressiveCommitMode::FullLifting,
            direct_composition_retention_mode: DirectCompositionRetentionMode::Disabled,
        }
    }

    /// Bind the plan to the AOT pack embedded in the running binary. Stub builds
    /// and binaries with no generated pack are rejected before CUDA allocation.
    pub fn loaded_starknet_blake2s() -> Result<Self, ProtocolPlanError> {
        let hash = stwo_backend_cuda::aot::loaded_manifest_hash();
        let composition_max_kernel_instrs = stwo_backend_cuda::aot::loaded_constraint_max_instrs();
        if hash == 0 {
            return Err(ProtocolPlanError::UnboundKernelManifest);
        }
        if composition_max_kernel_instrs == 0 {
            return Err(ProtocolPlanError::UnboundCompositionKernelCap);
        }
        let mut policy = Self::starknet_blake2s(hash, composition_max_kernel_instrs);
        policy.commit_mode = stwo_backend_cuda::ProgressiveCommitMode::from_env();
        policy.direct_composition_retention_mode = DirectCompositionRetentionMode::from_env();
        if policy.direct_composition_retention_mode == DirectCompositionRetentionMode::ExactNative {
            if policy.commit_mode != stwo_backend_cuda::ProgressiveCommitMode::DomainProgressive {
                return Err(ProtocolPlanError::DirectRetentionRequiresProgressiveCommit);
            }
            return Err(ProtocolPlanError::DirectRetentionExecutionUnsupported);
        }
        if let Ok(value) = crate::flags::env_value("STWO_CUDA_RETAINED_LDE_BUDGET_BYTES") {
            policy.retained_lde_budget_bytes = value
                .parse()
                .map_err(|_| ProtocolPlanError::InvalidRetainedLdeBudget)?;
        }
        Ok(policy)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProtocolPlanError {
    UnboundChannel,
    UnboundKernelManifest,
    UnboundCompositionKernelCap,
    InvalidRetainedLdeBudget,
    UnsupportedRetainAllLde,
    DirectRetentionRequiresProgressiveCommit,
    DirectRetentionExecutionUnsupported,
    DirectRetentionCompositionMissing,
    DirectRetentionBudgetExceeded {
        required_bytes: usize,
        budget_bytes: usize,
    },
    DirectRetention(DirectCompositionRetentionError),
    CompositionKernelCapMismatch {
        policy: usize,
        plan: usize,
    },
    InvalidClaimTreeCount(usize),
    EmptyClaimTree(usize),
    EmptyPreprocessedTrace,
    InvalidLiftingLogSize {
        lifting: u32,
        required: u32,
    },
    DynamicTreeLiftingMismatch {
        tree: CommitmentTreeId,
        tree_lifting: u32,
        composition_lifting: u32,
    },
    InvalidFriGeometry {
        lifting: u32,
        first_fold: u32,
        last_domain: u32,
    },
    DiscoveryLiftingMismatch {
        planned: u32,
        discovered: u32,
    },
    Fri(PreparedFriError),
    Relation(RelationExecutionError),
    ProofShapeNotExact,
    MissingOrderedComponent(&'static str),
    DuplicateOrderedComponent(&'static str),
    OrderedComponentCoverage {
        expected: usize,
        actual: usize,
    },
    MissingRelationOutput {
        component: &'static str,
        part: TracePartId,
    },
    DuplicateRelationOutput {
        component: &'static str,
        part: TracePartId,
    },
    RelationOutputRowsMismatch {
        component: &'static str,
        part: TracePartId,
        expected: u64,
        actual: u32,
    },
    UnusedRelationOutput {
        component: &'static str,
        part: TracePartId,
    },
    InvalidTracePart {
        component: &'static str,
        part: TracePartId,
    },
    CommitmentColumnCountMismatch {
        tree: CommitmentTreeId,
        expected: usize,
        actual: usize,
    },
    CommitmentColumnLogMismatch {
        tree: CommitmentTreeId,
        column: usize,
        expected: u32,
        actual: u32,
    },
    InvalidOodsTreeCount(usize),
    OodsTreeColumnCountMismatch {
        tree: usize,
        expected: usize,
        actual: usize,
    },
    OodsColumnOrderMismatch {
        flat_column: usize,
        expected_tree: usize,
        expected_column: usize,
        actual_tree: usize,
        actual_column: usize,
    },
    OodsColumnLogMismatch {
        tree: usize,
        column: usize,
        expected: u32,
        actual: u32,
    },
    OodsMaskArityMismatch {
        tree: usize,
        column: usize,
        shape_points: usize,
        offset_points: usize,
    },
    OodsSampleCountMismatch {
        expected: usize,
        actual: usize,
    },
    TranscriptAbi,
    SizeOverflow,
}

impl core::fmt::Display for ProtocolPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid CUDA protocol plan: {self:?}")
    }
}

impl std::error::Error for ProtocolPlanError {}

/// Build the exact protocol geometry consumed by [`crate::arena_plan::ProofArenaPlan`].
///
/// Claim tree order is protocol data: base then interaction.  Composition is
/// eight M31 coordinate polynomials (two secure halves), and the persistent
/// preprocessed tree is the first canonical commitment and is materialized once
/// per exact resident workspace.
pub fn plan_protocol_geometry(
    proof_plan: &ProofPlan,
    claim: &CairoClaim,
    preprocessed_trace: &PreProcessedTrace,
    pcs: &PcsConfig,
    include_all_preprocessed_columns: bool,
    policy: ProtocolPlanPolicy,
    transcript_plan: &CairoBlake2sTranscriptPlan,
    discovery: &ProtocolTranscriptDiscovery,
    composition: &CompositionPlan,
) -> Result<ProtocolGeometry, ProtocolPlanError> {
    if composition.max_kernel_instrs != policy.composition_max_kernel_instrs {
        return Err(ProtocolPlanError::CompositionKernelCapMismatch {
            policy: policy.composition_max_kernel_instrs,
            plan: composition.max_kernel_instrs,
        });
    }
    let claim_log_sizes = claim.log_sizes();
    plan_protocol_from_logs(
        proof_plan,
        &claim_log_sizes,
        preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        policy,
        TranscriptGeometry {
            schedule_key: transcript_plan.schedule_key(),
            requirements: transcript_plan.schedule().requirements().clone(),
        },
        discovery,
        composition.key(),
        Some(composition),
    )
}

/// One trace column in the exact unsorted order emitted by
/// `CairoClaimGenerator::write_trace` and consumed by `CairoClaim::log_sizes`.
///
/// Prepared commitments reorder these columns stably by log size.  Keeping the
/// pre-sort identity public is the checked hand-off from witness output into the
/// arena: a caller may not infer component ownership from a raw column index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraceCommitmentColumn {
    pub source: CommitmentColumnSource,
    pub log_size: u32,
}

/// Exact claim-order layouts for the two Cairo trace commitments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TraceCommitmentLayout {
    pub base: Vec<TraceCommitmentColumn>,
    pub interaction: Vec<TraceCommitmentColumn>,
}

#[derive(Clone, Copy, Debug)]
struct PlannedRelationOutput {
    component: &'static str,
    part: TracePartId,
    padded_rows: u32,
    coordinates: usize,
    consumed: bool,
}

pub fn trace_commitment_layout(
    proof_plan: &ProofPlan,
) -> Result<TraceCommitmentLayout, ProtocolPlanError> {
    if !proof_plan.capture_ready() {
        return Err(ProtocolPlanError::ProofShapeNotExact);
    }
    if CAIRO_COMMITMENT_COMPONENT_ORDER.len() != proof_plan.components.len() {
        return Err(ProtocolPlanError::OrderedComponentCoverage {
            expected: proof_plan.components.len(),
            actual: CAIRO_COMMITMENT_COMPONENT_ORDER.len(),
        });
    }
    for (index, &component) in CAIRO_COMMITMENT_COMPONENT_ORDER.iter().enumerate() {
        if CAIRO_COMMITMENT_COMPONENT_ORDER[..index].contains(&component) {
            return Err(ProtocolPlanError::DuplicateOrderedComponent(component));
        }
        if !proof_plan
            .components
            .iter()
            .any(|candidate| candidate.node.id == component)
        {
            return Err(ProtocolPlanError::MissingOrderedComponent(component));
        }
    }

    let relation_execution =
        RelationExecutionPlan::from_proof_plan(proof_plan, &CAIRO_RELATION_GRAPH)
            .map_err(ProtocolPlanError::Relation)?;
    let relation_requirements = relation_execution
        .requirements()
        .map_err(ProtocolPlanError::Relation)?;
    let mut relation_outputs = Vec::with_capacity(relation_requirements.instances.len());
    for requirement in &relation_requirements.instances {
        let batch = relation_execution
            .batches
            .get(requirement.batch_index)
            .ok_or(ProtocolPlanError::SizeOverflow)?;
        let part = match batch.trace_part {
            RelationTracePart::Component => TracePartId::Main,
            RelationTracePart::EachMemoryBig => TracePartId::MemoryBig(
                u32::try_from(requirement.instance_index)
                    .map_err(|_| ProtocolPlanError::SizeOverflow)?,
            ),
            RelationTracePart::MemorySmall => TracePartId::MemorySmall,
        };
        relation_outputs.push(PlannedRelationOutput {
            component: batch.component,
            part,
            padded_rows: requirement.row_capacity,
            coordinates: requirement.output_coordinate_count,
            consumed: false,
        });
    }

    let mut base = Vec::new();
    let mut interaction = Vec::new();
    for &component_id in CAIRO_COMMITMENT_COMPONENT_ORDER {
        let component = proof_plan
            .components
            .iter()
            .find(|component| component.node.id == component_id)
            .ok_or(ProtocolPlanError::MissingOrderedComponent(component_id))?;
        let RowResolution::Resolved(parts) = &component.runtime.rows else {
            if matches!(component.runtime.rows, RowResolution::Absent) {
                continue;
            }
            return Err(ProtocolPlanError::ProofShapeNotExact);
        };
        let mut parts = parts.clone();
        parts.sort_unstable_by_key(|part| match part.part {
            TracePartId::Main => (0u8, 0u32),
            TracePartId::MemoryBig(index) => (1, index),
            TracePartId::MemorySmall => (2, 0),
        });
        for part in parts {
            let log_size = exact_log_size(component_id, part)?;
            let base_columns = match (component.node.facts.trace_columns, part.part) {
                (TraceColumnCount::Fixed(columns), TracePartId::Main) => columns,
                (TraceColumnCount::SplitMemory { big, .. }, TracePartId::MemoryBig(_)) => big,
                (TraceColumnCount::SplitMemory { small, .. }, TracePartId::MemorySmall) => small,
                _ => {
                    return Err(ProtocolPlanError::InvalidTracePart {
                        component: component_id,
                        part: part.part,
                    })
                }
            };
            base.extend((0..base_columns).map(|ordinal| TraceCommitmentColumn {
                source: CommitmentColumnSource::Trace {
                    component: component_id,
                    part: part.part,
                    purpose: BufferPurpose::BaseCoefficients,
                    ordinal,
                },
                log_size,
            }));

            let mut matching = relation_outputs
                .iter_mut()
                .filter(|output| output.component == component_id && output.part == part.part);
            let Some(output) = matching.next() else {
                if component.node.facts.logup_columns.is_some()
                    || matches!(
                        component.node.facts.trace_columns,
                        TraceColumnCount::SplitMemory { .. }
                    )
                {
                    return Err(ProtocolPlanError::MissingRelationOutput {
                        component: component_id,
                        part: part.part,
                    });
                }
                continue;
            };
            if matching.next().is_some() {
                return Err(ProtocolPlanError::DuplicateRelationOutput {
                    component: component_id,
                    part: part.part,
                });
            }
            if u64::from(output.padded_rows) != part.padded_rows {
                return Err(ProtocolPlanError::RelationOutputRowsMismatch {
                    component: component_id,
                    part: part.part,
                    expected: part.padded_rows,
                    actual: output.padded_rows,
                });
            }
            output.consumed = true;
            interaction.extend(
                (0..output.coordinates)
                    .map(|coordinate| {
                        Ok(TraceCommitmentColumn {
                            source: CommitmentColumnSource::Trace {
                                component: component_id,
                                part: part.part,
                                purpose: BufferPurpose::InteractionCoefficients,
                                ordinal: u32::try_from(coordinate)
                                    .map_err(|_| ProtocolPlanError::SizeOverflow)?,
                            },
                            log_size,
                        })
                    })
                    .collect::<Result<Vec<_>, ProtocolPlanError>>()?,
            );
        }
    }
    if let Some(output) = relation_outputs.iter().find(|output| !output.consumed) {
        return Err(ProtocolPlanError::UnusedRelationOutput {
            component: output.component,
            part: output.part,
        });
    }
    Ok(TraceCommitmentLayout { base, interaction })
}

fn exact_log_size(component: &'static str, part: TracePartShape) -> Result<u32, ProtocolPlanError> {
    if part.padded_rows == 0 || !part.padded_rows.is_power_of_two() {
        return Err(ProtocolPlanError::InvalidTracePart {
            component,
            part: part.part,
        });
    }
    Ok(part.padded_rows.ilog2())
}

fn validate_claim_logs(
    tree: CommitmentTreeId,
    planned: &[TraceCommitmentColumn],
    actual: &[u32],
) -> Result<(), ProtocolPlanError> {
    if planned.len() != actual.len() {
        return Err(ProtocolPlanError::CommitmentColumnCountMismatch {
            tree,
            expected: planned.len(),
            actual: actual.len(),
        });
    }
    for (column, (planned, &actual)) in planned.iter().zip(actual).enumerate() {
        if planned.log_size != actual {
            return Err(ProtocolPlanError::CommitmentColumnLogMismatch {
                tree,
                column,
                expected: planned.log_size,
                actual,
            });
        }
    }
    Ok(())
}

fn canonical_commit_columns(
    mut columns: Vec<TraceCommitmentColumn>,
) -> (Vec<Vec<u32>>, Vec<Vec<CommitmentColumnSource>>) {
    columns.sort_by_key(|column| column.log_size);
    let logs = columns
        .chunks(16)
        .map(|group| group.iter().map(|column| column.log_size).collect())
        .collect();
    let sources = columns
        .chunks(16)
        .map(|group| group.iter().map(|column| column.source).collect())
        .collect();
    (logs, sources)
}

fn plan_oods_geometry(
    discovery: &ProtocolTranscriptDiscovery,
    preprocessed_logs: &[u32],
    base_columns: &[TraceCommitmentColumn],
    interaction_columns: &[TraceCommitmentColumn],
    composition_coefficient_log: u32,
    log_blowup_factor: u32,
) -> Result<OodsGeometry, ProtocolPlanError> {
    let expected = vec![
        preprocessed_logs
            .iter()
            .enumerate()
            .map(|(ordinal, &log_size)| {
                Ok((
                    OpenedColumnSource::Preprocessed {
                        ordinal: u32::try_from(ordinal)
                            .map_err(|_| ProtocolPlanError::SizeOverflow)?,
                    },
                    log_size,
                ))
            })
            .collect::<Result<Vec<_>, ProtocolPlanError>>()?,
        base_columns
            .iter()
            .map(|column| (column.source.into(), column.log_size))
            .collect(),
        interaction_columns
            .iter()
            .map(|column| (column.source.into(), column.log_size))
            .collect(),
        (0..8)
            .map(|ordinal| {
                (
                    OpenedColumnSource::Composition { ordinal },
                    composition_coefficient_log,
                )
            })
            .collect(),
    ];
    if discovery.oods_topology.tree_column_counts.len() != expected.len() {
        return Err(ProtocolPlanError::InvalidOodsTreeCount(
            discovery.oods_topology.tree_column_counts.len(),
        ));
    }
    for (tree, (expected_columns, &actual)) in expected
        .iter()
        .zip(&discovery.oods_topology.tree_column_counts)
        .enumerate()
    {
        if expected_columns.len() != actual {
            return Err(ProtocolPlanError::OodsTreeColumnCountMismatch {
                tree,
                expected: expected_columns.len(),
                actual,
            });
        }
    }
    let expected_column_count = expected.iter().map(Vec::len).sum::<usize>();
    if discovery.oods_topology.columns.len() != expected_column_count {
        return Err(ProtocolPlanError::OodsTreeColumnCountMismatch {
            tree: expected.len(),
            expected: expected_column_count,
            actual: discovery.oods_topology.columns.len(),
        });
    }

    let mut flat_column = 0usize;
    let mut sample_count = 0usize;
    let mut columns = Vec::with_capacity(expected_column_count);
    for (tree, expected_columns) in expected.into_iter().enumerate() {
        for (column, (source, log_size)) in expected_columns.into_iter().enumerate() {
            let discovered = &discovery.oods_topology.columns[flat_column];
            if discovered.tree != tree || discovered.column != column {
                return Err(ProtocolPlanError::OodsColumnOrderMismatch {
                    flat_column,
                    expected_tree: tree,
                    expected_column: column,
                    actual_tree: discovered.tree,
                    actual_column: discovered.column,
                });
            }
            if discovered.coefficient_log_size != log_size {
                return Err(ProtocolPlanError::OodsColumnLogMismatch {
                    tree,
                    column,
                    expected: log_size,
                    actual: discovered.coefficient_log_size,
                });
            }
            if discovered.shape_points.len() != discovered.offset_points.len() {
                return Err(ProtocolPlanError::OodsMaskArityMismatch {
                    tree,
                    column,
                    shape_points: discovered.shape_points.len(),
                    offset_points: discovered.offset_points.len(),
                });
            }
            sample_count = sample_count
                .checked_add(discovered.shape_points.len())
                .ok_or(ProtocolPlanError::SizeOverflow)?;
            columns.push(OodsColumnGeometry {
                source,
                coefficient_log_size: log_size,
                evaluation_log_size: log_size
                    .checked_add(log_blowup_factor)
                    .ok_or(ProtocolPlanError::SizeOverflow)?,
                shape_points: discovered.shape_points.clone(),
                offset_points: discovered.offset_points.clone(),
            });
            flat_column += 1;
        }
    }
    if sample_count != discovery.oods_sampled_value_felts {
        return Err(ProtocolPlanError::OodsSampleCountMismatch {
            expected: discovery.oods_sampled_value_felts,
            actual: sample_count,
        });
    }
    Ok(OodsGeometry {
        mask_log_size: discovery.max_log_degree_bound,
        sampled_values_input: CairoTranscriptInput::OodsSampledValues
            .id()
            .map_err(|_| ProtocolPlanError::TranscriptAbi)?,
        point_parameter_output: CairoTranscriptOutput::OodsPointParameter
            .id()
            .map_err(|_| ProtocolPlanError::TranscriptAbi)?,
        quotient_random_coefficient_output: CairoTranscriptOutput::QuotientRandomCoefficient
            .id()
            .map_err(|_| ProtocolPlanError::TranscriptAbi)?,
        columns,
    })
}

fn plan_protocol_from_logs(
    proof_plan: &ProofPlan,
    claim_log_sizes: &[Vec<u32>],
    preprocessed_trace: &PreProcessedTrace,
    pcs: &PcsConfig,
    include_all_preprocessed_columns: bool,
    policy: ProtocolPlanPolicy,
    transcript: TranscriptGeometry,
    discovery: &ProtocolTranscriptDiscovery,
    composition_plan_hash: u64,
    composition: Option<&CompositionPlan>,
) -> Result<ProtocolGeometry, ProtocolPlanError> {
    if policy.channel_tag == 0 {
        return Err(ProtocolPlanError::UnboundChannel);
    }
    if policy.kernel_manifest_hash == 0 {
        return Err(ProtocolPlanError::UnboundKernelManifest);
    }
    if policy.composition_max_kernel_instrs == 0 {
        return Err(ProtocolPlanError::UnboundCompositionKernelCap);
    }
    if policy.direct_composition_retention_mode == DirectCompositionRetentionMode::ExactNative
        && policy.commit_mode != stwo_backend_cuda::ProgressiveCommitMode::DomainProgressive
    {
        return Err(ProtocolPlanError::DirectRetentionRequiresProgressiveCommit);
    }
    if claim_log_sizes.len() != 2 {
        return Err(ProtocolPlanError::InvalidClaimTreeCount(
            claim_log_sizes.len(),
        ));
    }
    for (tree, logs) in claim_log_sizes.iter().enumerate() {
        if logs.is_empty() {
            return Err(ProtocolPlanError::EmptyClaimTree(tree));
        }
    }
    let preprocessed_logs = preprocessed_trace.log_sizes();
    if preprocessed_logs.is_empty() {
        return Err(ProtocolPlanError::EmptyPreprocessedTrace);
    }

    let blowup = pcs.fri_config.log_blowup_factor;
    // STWO samples FRI queries on the split composition tree. With implicit
    // lifting, that height is derived from the two dynamic trace trees only;
    // a taller preprocessed tree is opened by remapping those query positions.
    let max_dynamic_trace_log = claim_log_sizes
        .iter()
        .flatten()
        .copied()
        .max()
        .ok_or(ProtocolPlanError::EmptyPreprocessedTrace)?;
    let required_lifting = max_dynamic_trace_log
        .checked_add(blowup.max(1))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let lifting = pcs.lifting_log_size.unwrap_or(required_lifting);
    if lifting < required_lifting {
        return Err(ProtocolPlanError::InvalidLiftingLogSize {
            lifting,
            required: required_lifting,
        });
    }
    if lifting != discovery.lifting_log_size {
        return Err(ProtocolPlanError::DiscoveryLiftingMismatch {
            planned: lifting,
            discovered: discovery.lifting_log_size,
        });
    }

    let tree_lifting = |logs: &[u32]| -> Result<u32, ProtocolPlanError> {
        let max = logs
            .iter()
            .copied()
            .max()
            .ok_or(ProtocolPlanError::EmptyPreprocessedTrace)?;
        let required = max
            .checked_add(blowup)
            .ok_or(ProtocolPlanError::SizeOverflow)?;
        let actual = pcs.lifting_log_size.unwrap_or(required);
        if actual < required {
            return Err(ProtocolPlanError::InvalidLiftingLogSize {
                lifting: actual,
                required,
            });
        }
        Ok(actual)
    };

    let preprocessed_lifting = tree_lifting(&preprocessed_logs)?;
    let base_lifting = tree_lifting(&claim_log_sizes[0])?;
    let interaction_lifting = tree_lifting(&claim_log_sizes[1])?;
    for (tree, tree_lifting) in [
        (CommitmentTreeId::Base, base_lifting),
        (CommitmentTreeId::Interaction, interaction_lifting),
    ] {
        if tree_lifting != lifting {
            return Err(ProtocolPlanError::DynamicTreeLiftingMismatch {
                tree,
                tree_lifting,
                composition_lifting: lifting,
            });
        }
    }
    if include_all_preprocessed_columns && preprocessed_lifting > lifting {
        return Err(ProtocolPlanError::InvalidLiftingLogSize {
            lifting,
            required: preprocessed_lifting,
        });
    }
    let composition_coefficient_log = lifting
        .checked_sub(blowup)
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let TraceCommitmentLayout {
        base: base_columns,
        interaction: interaction_columns,
    } = trace_commitment_layout(proof_plan)?;
    validate_claim_logs(CommitmentTreeId::Base, &base_columns, &claim_log_sizes[0])?;
    validate_claim_logs(
        CommitmentTreeId::Interaction,
        &interaction_columns,
        &claim_log_sizes[1],
    )?;
    let oods = plan_oods_geometry(
        discovery,
        &preprocessed_logs,
        &base_columns,
        &interaction_columns,
        composition_coefficient_log,
        blowup,
    )?;
    let (base_logs, base_sources) = canonical_commit_columns(base_columns);
    let (interaction_logs, interaction_sources) = canonical_commit_columns(interaction_columns);
    let (grouped_preprocessed_logs, preprocessed_sources) = canonical_commit_columns(
        preprocessed_logs
            .iter()
            .copied()
            .enumerate()
            .map(|(ordinal, log_size)| {
                Ok(TraceCommitmentColumn {
                    source: CommitmentColumnSource::Preprocessed {
                        ordinal: u32::try_from(ordinal)
                            .map_err(|_| ProtocolPlanError::SizeOverflow)?,
                    },
                    log_size,
                })
            })
            .collect::<Result<Vec<_>, ProtocolPlanError>>()?,
    );
    let composition_sources = (0..8)
        .map(|ordinal| CommitmentColumnSource::Composition { ordinal })
        .collect();

    let commit_config = |tree_lifting| CommitWorkspaceConfig {
        log_blowup_factor: blowup,
        lifting_log_size: tree_lifting,
        unretained_bottom_layers: policy.unretained_bottom_layers,
        max_fused_tail_levels: policy.max_fused_tail_levels,
    };
    let mut commitments = vec![
        CommitmentGeometry {
            id: CommitmentTreeId::Preprocessed,
            created: ProofEpoch::Ingest,
            config: commit_config(preprocessed_lifting),
            grouped_column_log_sizes: grouped_preprocessed_logs,
            grouped_column_sources: preprocessed_sources,
            retained_evaluation_groups: Vec::new(),
            direct_composition_evaluation_groups: Vec::new(),
        },
        CommitmentGeometry {
            id: CommitmentTreeId::Base,
            created: ProofEpoch::BaseCommit,
            config: commit_config(base_lifting),
            grouped_column_log_sizes: base_logs,
            grouped_column_sources: base_sources,
            retained_evaluation_groups: Vec::new(),
            direct_composition_evaluation_groups: Vec::new(),
        },
        CommitmentGeometry {
            id: CommitmentTreeId::Interaction,
            created: ProofEpoch::InteractionCommit,
            config: commit_config(interaction_lifting),
            grouped_column_log_sizes: interaction_logs,
            grouped_column_sources: interaction_sources,
            retained_evaluation_groups: Vec::new(),
            direct_composition_evaluation_groups: Vec::new(),
        },
        CommitmentGeometry {
            id: CommitmentTreeId::Composition,
            created: ProofEpoch::CompositionCommit,
            config: commit_config(lifting),
            grouped_column_log_sizes: vec![vec![composition_coefficient_log; 8]],
            grouped_column_sources: vec![composition_sources],
            retained_evaluation_groups: Vec::new(),
            direct_composition_evaluation_groups: Vec::new(),
        },
    ];
    let direct_composition_retention = match policy.direct_composition_retention_mode {
        DirectCompositionRetentionMode::Disabled => None,
        DirectCompositionRetentionMode::ExactNative => {
            let composition =
                composition.ok_or(ProtocolPlanError::DirectRetentionCompositionMissing)?;
            let consumers = derive_direct_composition_consumers(&oods, composition)
                .map_err(ProtocolPlanError::DirectRetention)?;
            Some(
                plan_direct_composition_retention_from_parts(
                    &commitments,
                    &oods.columns,
                    blowup,
                    &consumers,
                )
                .map_err(ProtocolPlanError::DirectRetention)?,
            )
        }
    };
    let retention = select_retained_evaluation_groups(
        &mut commitments,
        policy.decommit_strategy,
        policy.retained_lde_budget_bytes,
        direct_composition_retention.as_ref(),
    )?;

    let fri_layer_log_sizes = fri_merkle_log_sizes(lifting, pcs.fri_config)?;
    let opened_tree_log_sizes = vec![
        preprocessed_lifting,
        base_lifting,
        interaction_lifting,
        lifting,
    ];
    let total_opened_columns = preprocessed_logs
        .len()
        .checked_add(claim_log_sizes[0].len())
        .and_then(|total| total.checked_add(claim_log_sizes[1].len()))
        .and_then(|total| total.checked_add(8))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let proof_capacity_words = proof_capacity_words(
        pcs,
        total_opened_columns,
        &opened_tree_log_sizes,
        &fri_layer_log_sizes,
    )?;
    let preprocessed_binding_hash =
        preprocessed_binding_hash(preprocessed_trace, pcs, include_all_preprocessed_columns);

    Ok(ProtocolGeometry {
        identity: ProtocolIdentity::from_pcs(
            pcs,
            policy.channel_tag,
            proof_plan.relation_graph_hash,
            preprocessed_binding_hash,
            oods.topology_hash(),
            composition_plan_hash,
            policy.kernel_manifest_hash,
            policy.decommit_strategy,
            policy.commit_mode,
            policy.direct_composition_retention_mode,
            direct_composition_retention
                .as_ref()
                .map_or(0, |plan| plan.cache_key),
            direct_composition_retention
                .as_ref()
                .map_or(0, |plan| direct_bitmap_hash(&plan.direct_bitmap)),
            retention.direct_group_rounded_bytes,
            direct_composition_retention
                .as_ref()
                .map_or(0, |_| retention.union_group_rounded_bytes),
        ),
        preprocessed_column_ids: preprocessed_trace
            .ids()
            .into_iter()
            .map(|identity| identity.id)
            .collect(),
        max_domain_log_size: preprocessed_lifting.max(lifting),
        lifting_log_size: lifting,
        n_queries: pcs.fri_config.n_queries,
        total_opened_columns,
        proof_capacity_words,
        transcript,
        composition_random_coefficient_output: CairoTranscriptOutput::CompositionRandomCoefficient
            .id()
            .map_err(|_| ProtocolPlanError::TranscriptAbi)?,
        oods,
        quotient: QuotientGeometry {
            partial_numerator_log_sizes: discovery.partial_numerator_log_sizes.clone(),
        },
        commitments,
        direct_composition_retention,
        opened_tree_log_sizes,
        fri_layer_log_sizes,
    })
}

fn select_retained_evaluation_groups(
    commitments: &mut [CommitmentGeometry],
    strategy: DecommitStrategy,
    budget_bytes: usize,
    direct: Option<&DirectCompositionRetentionPlan>,
) -> Result<RetentionSelection, ProtocolPlanError> {
    for commitment in commitments.iter_mut() {
        commitment.retained_evaluation_groups =
            vec![false; commitment.grouped_column_log_sizes.len()];
        commitment.direct_composition_evaluation_groups =
            vec![false; commitment.grouped_column_log_sizes.len()];
    }
    if let Some(plan) = direct {
        for binding in plan.bindings.iter().filter(|binding| binding.direct) {
            let column =
                plan.columns
                    .get(binding.column)
                    .ok_or(ProtocolPlanError::DirectRetention(
                        DirectCompositionRetentionError::PlanDrift,
                    ))?;
            let commitment = commitments
                .iter_mut()
                .find(|commitment| commitment.id == column.tree)
                .ok_or(ProtocolPlanError::DirectRetention(
                    DirectCompositionRetentionError::MissingCommitmentTree(column.tree),
                ))?;
            let selected = commitment
                .direct_composition_evaluation_groups
                .get_mut(column.group)
                .ok_or(ProtocolPlanError::DirectRetention(
                    DirectCompositionRetentionError::PlanDrift,
                ))?;
            *selected = true;
        }
    }

    let direct_words = commitments.iter().try_fold(0usize, |total, commitment| {
        commitment
            .grouped_column_log_sizes
            .iter()
            .zip(&commitment.direct_composition_evaluation_groups)
            .try_fold(total, |total, (logs, &selected)| {
                if selected {
                    total
                        .checked_add(retained_group_words(commitment, logs)?)
                        .ok_or(ProtocolPlanError::SizeOverflow)
                } else {
                    Ok(total)
                }
            })
    })?;
    let budget_words = budget_bytes / core::mem::size_of::<u32>();
    if direct_words > budget_words {
        return Err(ProtocolPlanError::DirectRetentionBudgetExceeded {
            required_bytes: direct_words
                .checked_mul(core::mem::size_of::<u32>())
                .ok_or(ProtocolPlanError::SizeOverflow)?,
            budget_bytes,
        });
    }

    match strategy {
        DecommitStrategy::RecomputeQueriedLde => {}
        DecommitStrategy::RetainAllLde => return Err(ProtocolPlanError::UnsupportedRetainAllLde),
        DecommitStrategy::HybridByGroup => {
            select_decommit_groups(commitments, budget_words - direct_words)?;
        }
    }

    let union_words = commitments.iter().try_fold(0usize, |total, commitment| {
        commitment
            .grouped_column_log_sizes
            .iter()
            .zip(&commitment.retained_evaluation_groups)
            .zip(&commitment.direct_composition_evaluation_groups)
            .try_fold(total, |total, ((logs, &decommit), &direct)| {
                if decommit || direct {
                    total
                        .checked_add(retained_group_words(commitment, logs)?)
                        .ok_or(ProtocolPlanError::SizeOverflow)
                } else {
                    Ok(total)
                }
            })
    })?;
    Ok(RetentionSelection {
        direct_group_rounded_bytes: direct_words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ProtocolPlanError::SizeOverflow)?,
        union_group_rounded_bytes: union_words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ProtocolPlanError::SizeOverflow)?,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RetentionSelection {
    direct_group_rounded_bytes: usize,
    union_group_rounded_bytes: usize,
}

fn retained_group_words(
    commitment: &CommitmentGeometry,
    logs: &[u32],
) -> Result<usize, ProtocolPlanError> {
    logs.iter().try_fold(0usize, |words, &log_size| {
        let evaluation_log = log_size
            .checked_add(commitment.config.log_blowup_factor)
            .ok_or(ProtocolPlanError::SizeOverflow)?;
        words
            .checked_add(
                1usize
                    .checked_shl(evaluation_log)
                    .ok_or(ProtocolPlanError::SizeOverflow)?,
            )
            .ok_or(ProtocolPlanError::SizeOverflow)
    })
}

fn select_decommit_groups(
    commitments: &mut [CommitmentGeometry],
    mut remaining_words: usize,
) -> Result<(), ProtocolPlanError> {
    #[derive(Clone, Copy)]
    struct Candidate {
        commitment: usize,
        group: usize,
        words: usize,
        weighted_log: u128,
    }

    let mut candidates = Vec::new();
    // The fixed preprocessed commitment has its own process-persistent cache.
    // Hybrid per-proof storage is reserved for the three dynamic trees.
    for (commitment_index, commitment) in commitments.iter().enumerate().skip(1) {
        for (group_index, logs) in commitment.grouped_column_log_sizes.iter().enumerate() {
            let words = retained_group_words(commitment, logs)?;
            let mut weighted_log = 0u128;
            for &log_size in logs {
                let evaluation_log = log_size
                    .checked_add(commitment.config.log_blowup_factor)
                    .ok_or(ProtocolPlanError::SizeOverflow)?;
                let column_words = 1usize
                    .checked_shl(evaluation_log)
                    .ok_or(ProtocolPlanError::SizeOverflow)?;
                weighted_log = weighted_log
                    .checked_add((column_words as u128) * u128::from(evaluation_log))
                    .ok_or(ProtocolPlanError::SizeOverflow)?;
            }
            candidates.push(Candidate {
                commitment: commitment_index,
                group: group_index,
                words,
                weighted_log,
            });
        }
    }
    // Saved FFT work per retained word is approximately log(domain). Stable
    // ties prefer the smaller group, then proof order, to use the full budget.
    candidates.sort_by(|left, right| {
        (right.weighted_log * left.words as u128)
            .cmp(&(left.weighted_log * right.words as u128))
            .then_with(|| left.words.cmp(&right.words))
            .then_with(|| left.commitment.cmp(&right.commitment))
            .then_with(|| left.group.cmp(&right.group))
    });

    for candidate in candidates {
        let additional = if commitments[candidate.commitment].direct_composition_evaluation_groups
            [candidate.group]
        {
            0
        } else {
            candidate.words
        };
        if additional <= remaining_words {
            commitments[candidate.commitment].retained_evaluation_groups[candidate.group] = true;
            remaining_words -= additional;
        }
    }
    Ok(())
}

/// Stable ascending leaf order, split into full 16-column Blake2s blocks and
/// one final block.  Sorting is stable so equal-log columns retain claim order.
#[cfg(test)]
fn canonical_commit_groups(log_sizes: &[u32]) -> Vec<Vec<u32>> {
    let mut sorted = log_sizes.to_vec();
    sorted.sort_by_key(|log| *log);
    sorted.chunks(16).map(<[u32]>::to_vec).collect()
}

/// Merkle leaf heights for the first and inner FRI commitments, in transcript
/// order.  Packed leaves remove two bits.  A final one-bit fold can therefore
/// produce the same Merkle height as its predecessor; order, not strict height,
/// identifies the graph segment.
pub fn fri_merkle_log_sizes(
    lifting_log_size: u32,
    config: FriConfig,
) -> Result<Vec<u32>, ProtocolPlanError> {
    let twiddle_log_size = lifting_log_size
        .checked_sub(1)
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let requirements = fri_workspace_requirements(FriWorkspaceConfig {
        fri: config,
        circle_log_size: lifting_log_size,
        twiddle_log_size,
    })
    .map_err(ProtocolPlanError::Fri)?;
    Ok(requirements
        .trees
        .iter()
        .map(|tree| {
            tree.layers_bottom_up
                .first()
                .expect("prepared FRI trees always contain a leaf")
                .log_size
        })
        .collect())
}

fn proof_capacity_words(
    pcs: &PcsConfig,
    total_opened_columns: usize,
    opened_tree_log_sizes: &[u32],
    fri_layer_log_sizes: &[u32],
) -> Result<usize, ProtocolPlanError> {
    let queries = pcs.fri_config.n_queries;
    let queried_values = queries
        .checked_mul(total_opened_columns)
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let path_height = opened_tree_log_sizes
        .iter()
        .chain(fri_layer_log_sizes)
        .try_fold(0usize, |sum, &log| {
            sum.checked_add(log as usize)
                .ok_or(ProtocolPlanError::SizeOverflow)
        })?;
    let auth_paths = queries
        .checked_mul(path_height)
        .and_then(|words| words.checked_mul(8))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let fold_width = 1usize
        .checked_shl(pcs.fri_config.fold_step)
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let fri_witness = queries
        .checked_mul(fri_layer_log_sizes.len())
        .and_then(|words| words.checked_mul(fold_width))
        .and_then(|words| words.checked_mul(4))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    // Sixteen secure mask points per column is a deliberately conservative
    // serialization bound; actual Cairo masks are smaller.
    let sampled_values = total_opened_columns
        .checked_mul(16)
        .and_then(|words| words.checked_mul(4))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let last_layer = 1usize
        .checked_shl(pcs.fri_config.log_last_layer_degree_bound)
        .and_then(|words| words.checked_mul(4))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    queried_values
        .checked_add(auth_paths)
        .and_then(|words| words.checked_add(fri_witness))
        .and_then(|words| words.checked_add(sampled_values))
        .and_then(|words| words.checked_add(last_layer))
        .and_then(|words| words.checked_add(4096))
        .ok_or(ProtocolPlanError::SizeOverflow)
}

pub fn preprocessed_binding_hash(
    trace: &PreProcessedTrace,
    pcs: &PcsConfig,
    include_all_preprocessed_columns: bool,
) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    feed(b"stwo-cairo-preprocessed-binding-v1\0");
    feed(&[match trace.variant {
        PreProcessedTraceVariant::Canonical => 0,
        PreProcessedTraceVariant::CanonicalWithoutPedersen => 1,
        PreProcessedTraceVariant::CanonicalSmall => 2,
    }]);
    feed(&[u8::from(include_all_preprocessed_columns)]);
    feed(&pcs.fri_config.log_blowup_factor.to_le_bytes());
    feed(&pcs.lifting_log_size.unwrap_or(0).to_le_bytes());
    for (id, log_size) in trace.ids().into_iter().zip(trace.log_sizes()) {
        feed(&(id.id.len() as u64).to_le_bytes());
        feed(id.id.as_bytes());
        feed(&log_size.to_le_bytes());
    }
    hash
}

#[cfg(test)]
mod tests {
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::PcsConfig;
    use stwo_backend_cuda::{
        Blake2sTranscriptSchedule, TranscriptBoundaryId, TranscriptInputId, TranscriptOperation,
        TranscriptOutputId, TranscriptStart,
    };
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
    use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
    use stwo_cairo_prover::witness::proof_shape::{
        ProofShape, RuntimeComponentShape, TracePartShape,
    };

    use super::*;
    use crate::plan::ProofPlan;
    use crate::protocol_discovery::{DiscoveredOodsColumn, DiscoveredOodsTopology};
    use crate::relation_table::CAIRO_RELATION_GRAPH;
    use crate::schedule_table::CAIRO_SCHEDULE;

    fn pcs(fold_step: u32) -> PcsConfig {
        PcsConfig {
            pow_bits: 26,
            fri_config: FriConfig::new(0, 1, 70, fold_step),
            lifting_log_size: None,
        }
    }

    #[test]
    fn hybrid_opening_budget_prefers_highest_fft_work_per_word() {
        let config = CommitWorkspaceConfig {
            log_blowup_factor: 1,
            lifting_log_size: 21,
            unretained_bottom_layers: 4,
            max_fused_tail_levels: 12,
        };
        let mut commitments = vec![
            CommitmentGeometry {
                id: CommitmentTreeId::Preprocessed,
                created: ProofEpoch::Ingest,
                config,
                grouped_column_log_sizes: vec![vec![20]],
                grouped_column_sources: vec![Vec::new()],
                retained_evaluation_groups: Vec::new(),
                direct_composition_evaluation_groups: Vec::new(),
            },
            CommitmentGeometry {
                id: CommitmentTreeId::Base,
                created: ProofEpoch::BaseCommit,
                config,
                grouped_column_log_sizes: vec![vec![10], vec![12]],
                grouped_column_sources: vec![Vec::new(), Vec::new()],
                retained_evaluation_groups: Vec::new(),
                direct_composition_evaluation_groups: Vec::new(),
            },
            CommitmentGeometry {
                id: CommitmentTreeId::Interaction,
                created: ProofEpoch::InteractionCommit,
                config,
                grouped_column_log_sizes: vec![vec![11]],
                grouped_column_sources: vec![Vec::new()],
                retained_evaluation_groups: Vec::new(),
                direct_composition_evaluation_groups: Vec::new(),
            },
        ];
        select_retained_evaluation_groups(
            &mut commitments,
            DecommitStrategy::HybridByGroup,
            (1usize << 13) * core::mem::size_of::<u32>(),
            None,
        )
        .unwrap();
        assert_eq!(commitments[0].retained_evaluation_groups, [false]);
        assert_eq!(commitments[1].retained_evaluation_groups, [false, true]);
        assert_eq!(commitments[2].retained_evaluation_groups, [false]);
    }

    fn direct_plan(
        tree: CommitmentTreeId,
        group: usize,
        column_in_group: usize,
    ) -> DirectCompositionRetentionPlan {
        let source = OpenedColumnSource::Trace {
            component: "test",
            part: TracePartId::Main,
            purpose: BufferPurpose::BaseCoefficients,
            ordinal: u32::try_from(group * 16 + column_in_group).unwrap(),
        };
        DirectCompositionRetentionPlan {
            columns: vec![
                crate::direct_composition_retention::DirectCompositionColumn {
                    source,
                    tree,
                    proof_column: group * 16 + column_in_group,
                    group,
                    column_in_group,
                    canonical_column: group * 16 + column_in_group,
                    coefficient_log_size: 4,
                    evaluation_log_size: 5,
                    lifetime: crate::arena_plan::BufferLifetime::new(
                        ProofEpoch::BaseCommit,
                        ProofEpoch::Composition,
                    )
                    .unwrap(),
                },
            ],
            bindings: vec![
                crate::direct_composition_retention::DirectCompositionBinding {
                    consumer: 0,
                    column: 0,
                    consumer_evaluation_log_size: 5,
                    direct: true,
                },
            ],
            direct_bitmap: vec![1],
            buckets: Vec::new(),
            direct_column_count: 1,
            direct_bytes: 1 << 7,
            cache_key: 7,
        }
    }

    fn grouped_base(column_count: usize) -> CommitmentGeometry {
        let columns = (0..column_count)
            .map(|ordinal| CommitmentColumnSource::Trace {
                component: "test",
                part: TracePartId::Main,
                purpose: BufferPurpose::BaseCoefficients,
                ordinal: ordinal as u32,
            })
            .collect::<Vec<_>>();
        CommitmentGeometry {
            id: CommitmentTreeId::Base,
            created: ProofEpoch::BaseCommit,
            config: CommitWorkspaceConfig {
                log_blowup_factor: 1,
                lifting_log_size: 5,
                unretained_bottom_layers: 0,
                max_fused_tail_levels: 0,
            },
            grouped_column_log_sizes: columns
                .chunks(16)
                .map(|group| vec![4; group.len()])
                .collect(),
            grouped_column_sources: columns
                .chunks(16)
                .map(<[CommitmentColumnSource]>::to_vec)
                .collect(),
            retained_evaluation_groups: Vec::new(),
            direct_composition_evaluation_groups: Vec::new(),
        }
    }

    fn retention_fixture(column_count: usize) -> Vec<CommitmentGeometry> {
        let mut preprocessed = grouped_base(1);
        preprocessed.id = CommitmentTreeId::Preprocessed;
        preprocessed.created = ProofEpoch::Ingest;
        vec![preprocessed, grouped_base(column_count)]
    }

    #[test]
    fn direct_group_closure_rounds_15_16_17_and_shares_budget_by_intent() {
        for count in [15usize, 16, 17] {
            let group = (count - 1) / 16;
            let in_group = (count - 1) % 16;
            let direct = direct_plan(CommitmentTreeId::Base, group, in_group);
            let rounded_columns = if count == 17 { 1 } else { count };
            let rounded_bytes = rounded_columns * (1usize << 5) * core::mem::size_of::<u32>();

            let mut direct_only = retention_fixture(count);
            let selected = select_retained_evaluation_groups(
                &mut direct_only,
                DecommitStrategy::RecomputeQueriedLde,
                rounded_bytes,
                Some(&direct),
            )
            .unwrap();
            assert!(direct_only[1].direct_composition_evaluation_groups[group]);
            assert!(!direct_only[1].retained_evaluation_groups[group]);
            assert_eq!(selected.direct_group_rounded_bytes, rounded_bytes);
            assert_eq!(selected.union_group_rounded_bytes, rounded_bytes);

            let mut both = retention_fixture(count);
            select_retained_evaluation_groups(
                &mut both,
                DecommitStrategy::HybridByGroup,
                rounded_bytes,
                Some(&direct),
            )
            .unwrap();
            assert!(both[1].direct_composition_evaluation_groups[group]);
            assert!(both[1].retained_evaluation_groups[group]);

            let mut decommit_only = retention_fixture(count);
            select_retained_evaluation_groups(
                &mut decommit_only,
                DecommitStrategy::HybridByGroup,
                rounded_bytes,
                None,
            )
            .unwrap();
            assert!(!decommit_only[1].direct_composition_evaluation_groups[group]);
            assert!(decommit_only[1].retained_evaluation_groups[group]);

            let mut insufficient = retention_fixture(count);
            assert_eq!(
                select_retained_evaluation_groups(
                    &mut insufficient,
                    DecommitStrategy::RecomputeQueriedLde,
                    rounded_bytes - 1,
                    Some(&direct),
                ),
                Err(ProtocolPlanError::DirectRetentionBudgetExceeded {
                    required_bytes: rounded_bytes,
                    budget_bytes: rounded_bytes - 1,
                })
            );
        }
    }

    fn memory_plan() -> ProofPlan {
        let default = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let mut components = default.components().to_vec();
        *components
            .iter_mut()
            .find(|component| component.id == "memory_id_to_big")
            .unwrap() = RuntimeComponentShape::parts(
            "memory_id_to_big",
            vec![
                TracePartShape {
                    part: TracePartId::MemoryBig(0),
                    n_real_rows: 17,
                    padded_rows: 32,
                },
                TracePartShape {
                    part: TracePartId::MemorySmall,
                    n_real_rows: 9,
                    padded_rows: 16,
                },
            ],
        )
        .unwrap();
        let shape = ProofShape::new(components).unwrap();
        ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap()
    }

    fn transcript_geometry() -> TranscriptGeometry {
        let schedule = Blake2sTranscriptSchedule::new(
            TranscriptStart::Default,
            vec![
                TranscriptOperation::MixFelts {
                    boundary: TranscriptBoundaryId(1),
                    source: TranscriptInputId(1),
                    n_felts: 1,
                },
                TranscriptOperation::DrawSecureFelt {
                    boundary: TranscriptBoundaryId(2),
                    output: TranscriptOutputId(1),
                },
            ],
            8,
        )
        .unwrap();
        TranscriptGeometry {
            schedule_key: schedule.protocol_key(),
            requirements: schedule.requirements().clone(),
        }
    }

    fn discovery_for(
        claim_logs: &[Vec<u32>],
        preprocessed: &PreProcessedTrace,
        pcs: &PcsConfig,
    ) -> ProtocolTranscriptDiscovery {
        let blowup = pcs.fri_config.log_blowup_factor;
        let required = claim_logs.iter().flatten().copied().max().unwrap() + blowup.max(1);
        let lifting_log_size = pcs.lifting_log_size.unwrap_or(required);
        let max_log_degree_bound = lifting_log_size - blowup;
        let logs_by_tree = vec![
            preprocessed.log_sizes(),
            claim_logs[0].clone(),
            claim_logs[1].clone(),
            vec![max_log_degree_bound; 8],
        ];
        let tree_column_counts = logs_by_tree.iter().map(Vec::len).collect::<Vec<_>>();
        let mut columns = Vec::new();
        for (tree, logs) in logs_by_tree.into_iter().enumerate() {
            for (column, coefficient_log_size) in logs.into_iter().enumerate() {
                let sampled = tree == 0 && column == 0;
                columns.push(DiscoveredOodsColumn {
                    tree,
                    column,
                    coefficient_log_size,
                    shape_points: sampled
                        .then_some(stwo::core::circle::SECURE_FIELD_CIRCLE_GEN)
                        .into_iter()
                        .collect(),
                    offset_points: sampled
                        .then_some(stwo::core::circle::CirclePoint {
                            x: stwo::core::fields::m31::BaseField::from(1),
                            y: stwo::core::fields::m31::BaseField::from(0),
                        })
                        .into_iter()
                        .collect(),
                });
            }
        }
        ProtocolTranscriptDiscovery {
            interaction_claim_felts: 1,
            oods_sampled_value_felts: 1,
            sampled_value_felts_by_tree: vec![1, 0, 0, 0],
            partial_numerator_log_sizes: vec![preprocessed.log_sizes()[0]],
            oods_topology: DiscoveredOodsTopology {
                tree_column_counts,
                columns,
            },
            lifting_log_size,
            max_log_degree_bound,
        }
    }

    #[test]
    fn canonical_groups_are_full_blocks_plus_one_tail() {
        let logs: Vec<_> = (0..35).rev().map(|index| 4 + index % 7).collect();
        let groups = canonical_commit_groups(&logs);
        assert_eq!(groups.iter().map(Vec::len).collect::<Vec<_>>(), [16, 16, 3]);
        let flattened: Vec<_> = groups.into_iter().flatten().collect();
        assert!(flattened.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn canonical_commit_sort_preserves_source_order_within_each_log() {
        let source = |ordinal| CommitmentColumnSource::Trace {
            component: "test",
            part: TracePartId::Main,
            purpose: BufferPurpose::BaseCoefficients,
            ordinal,
        };
        let columns = vec![
            TraceCommitmentColumn {
                source: source(2),
                log_size: 6,
            },
            TraceCommitmentColumn {
                source: source(9),
                log_size: 5,
            },
            TraceCommitmentColumn {
                source: source(0),
                log_size: 6,
            },
            TraceCommitmentColumn {
                source: source(1),
                log_size: 6,
            },
        ];
        let (logs, sources) = canonical_commit_columns(columns);
        assert_eq!(logs.into_iter().flatten().collect::<Vec<_>>(), [5, 6, 6, 6]);
        assert_eq!(
            sources.into_iter().flatten().collect::<Vec<_>>(),
            [source(9), source(2), source(0), source(1)]
        );
    }

    #[test]
    fn fri_fold_three_matches_starknet_tree_schedule() {
        assert_eq!(
            fri_merkle_log_sizes(26, FriConfig::new(0, 1, 70, 3)).unwrap(),
            vec![24, 21, 18, 15, 12, 9, 6, 3, 2]
        );
    }

    #[test]
    fn packed_to_scalar_transition_can_repeat_a_merkle_height() {
        assert_eq!(
            fri_merkle_log_sizes(8, FriConfig::new(0, 1, 70, 2)).unwrap(),
            vec![6, 4, 2, 2]
        );
    }

    #[test]
    fn exact_claim_logs_produce_all_four_opening_trees() {
        let plan = memory_plan();
        let TraceCommitmentLayout { base, interaction } = trace_commitment_layout(&plan).unwrap();
        assert!(base.iter().all(|column| matches!(
            column.source,
            CommitmentColumnSource::Trace {
                purpose: BufferPurpose::BaseCoefficients,
                ..
            }
        )));
        assert!(interaction.iter().all(|column| matches!(
            column.source,
            CommitmentColumnSource::Trace {
                purpose: BufferPurpose::InteractionCoefficients,
                ..
            }
        )));
        let claim_logs = vec![
            base.iter().map(|column| column.log_size).collect(),
            interaction.iter().map(|column| column.log_size).collect(),
        ];
        let preprocessed = PreProcessedTrace::canonical();
        let discovery = discovery_for(&claim_logs, &preprocessed, &pcs(3));
        let mut invalid_direct = ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048);
        invalid_direct.direct_composition_retention_mode =
            DirectCompositionRetentionMode::ExactNative;
        assert_eq!(
            plan_protocol_from_logs(
                &plan,
                &claim_logs,
                &preprocessed,
                &pcs(3),
                false,
                invalid_direct,
                transcript_geometry(),
                &discovery,
                0x5678,
                None,
            ),
            Err(ProtocolPlanError::DirectRetentionRequiresProgressiveCommit)
        );
        let geometry = plan_protocol_from_logs(
            &plan,
            &claim_logs,
            &preprocessed,
            &pcs(3),
            false,
            ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048),
            transcript_geometry(),
            &discovery,
            0x5678,
            None,
        )
        .unwrap();
        let blowup = pcs(3).fri_config.log_blowup_factor;
        let preprocessed_height = preprocessed.log_sizes().into_iter().max().unwrap() + blowup;
        let base_height = claim_logs[0].iter().copied().max().unwrap() + blowup;
        let interaction_height = claim_logs[1].iter().copied().max().unwrap() + blowup;
        assert_eq!(base_height, interaction_height);
        assert!(
            preprocessed_height > base_height,
            "regression fixture must exercise a taller preprocessed tree"
        );
        assert_eq!(geometry.lifting_log_size, base_height);
        assert_eq!(geometry.max_domain_log_size, preprocessed_height);
        assert_eq!(
            geometry.opened_tree_log_sizes,
            [
                preprocessed_height,
                base_height,
                interaction_height,
                base_height,
            ]
        );
        assert!(
            geometry.decommit_workspace_config().is_ok(),
            "preprocessed height may exceed the dynamic FRI query height"
        );
        assert_eq!(
            plan_protocol_from_logs(
                &plan,
                &claim_logs,
                &preprocessed,
                &pcs(3),
                true,
                ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048),
                transcript_geometry(),
                &discovery,
                0x5678,
                None,
            ),
            Err(ProtocolPlanError::InvalidLiftingLogSize {
                lifting: base_height,
                required: preprocessed_height,
            }),
            "including every fixed column must preserve STWO's explicit lifting check"
        );
        assert_eq!(geometry.commitments.len(), 4);
        assert_eq!(
            geometry
                .commitments
                .iter()
                .map(|commitment| commitment.id)
                .collect::<Vec<_>>(),
            [
                CommitmentTreeId::Preprocessed,
                CommitmentTreeId::Base,
                CommitmentTreeId::Interaction,
                CommitmentTreeId::Composition,
            ]
        );
        let committed_preprocessed = &geometry.commitments[0];
        assert_eq!(committed_preprocessed.created, ProofEpoch::Ingest);
        let mut expected_preprocessed = preprocessed
            .log_sizes()
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();
        expected_preprocessed.sort_by_key(|(_, log_size)| *log_size);
        assert_eq!(
            committed_preprocessed
                .grouped_column_sources
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            expected_preprocessed
                .into_iter()
                .map(|(ordinal, _)| CommitmentColumnSource::Preprocessed {
                    ordinal: ordinal as u32,
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(geometry.opened_tree_log_sizes.len(), 4);
        assert_eq!(
            geometry.total_opened_columns,
            preprocessed.log_sizes().len() + base.len() + interaction.len() + 8
        );
        assert!(geometry.commitments.iter().all(|commitment| commitment
            .grouped_column_sources
            .iter()
            .zip(&commitment.grouped_column_log_sizes)
            .all(|(sources, logs)| sources.len() == logs.len())));
        assert_eq!(
            geometry.identity.relation_graph_hash,
            plan.relation_graph_hash
        );
        assert_ne!(geometry.identity.preprocessed_binding_hash, 0);
        assert_eq!(
            geometry.identity.oods_topology_hash,
            geometry.oods.topology_hash()
        );
        let preprocessed_count = preprocessed.log_sizes().len();
        let opened_base =
            &geometry.oods.columns[preprocessed_count..preprocessed_count + base.len()];
        assert_eq!(
            opened_base
                .iter()
                .map(|column| column.source)
                .collect::<Vec<_>>(),
            base.iter()
                .map(|column| OpenedColumnSource::from(column.source))
                .collect::<Vec<_>>(),
            "OODS must retain base claim order"
        );
        let committed_base = geometry
            .commitments
            .iter()
            .find(|commitment| commitment.id == CommitmentTreeId::Base)
            .unwrap()
            .grouped_column_sources
            .iter()
            .flatten()
            .copied()
            .map(OpenedColumnSource::from)
            .collect::<Vec<_>>();
        assert_ne!(
            opened_base
                .iter()
                .map(|column| column.source)
                .collect::<Vec<_>>(),
            committed_base,
            "opening order must remain distinct from log-sorted commit leaf order"
        );
        let mut changed_topology = geometry.clone();
        let sampled = changed_topology
            .oods
            .columns
            .iter_mut()
            .find(|column| !column.shape_points.is_empty())
            .unwrap();
        sampled.shape_points[0] = sampled.shape_points[0].conjugate();
        assert_ne!(geometry.key(), changed_topology.key());
    }

    #[test]
    fn exact_native_progressive_protocol_seals_physical_retention_identity() {
        let proof_plan = memory_plan();
        let TraceCommitmentLayout { base, interaction } =
            trace_commitment_layout(&proof_plan).unwrap();
        let claim_logs = vec![
            base.iter().map(|column| column.log_size).collect(),
            interaction.iter().map(|column| column.log_size).collect(),
        ];
        let preprocessed = PreProcessedTrace::canonical();
        let pcs = pcs(3);
        let discovery = discovery_for(&claim_logs, &preprocessed, &pcs);
        let blowup = pcs.fri_config.log_blowup_factor;
        let preprocessed_logs = preprocessed.log_sizes();
        let (preprocessed_index, &preprocessed_log_size) = preprocessed_logs
            .iter()
            .enumerate()
            .max_by_key(|(_, log_size)| *log_size)
            .unwrap();
        let evaluation_log_size = preprocessed_log_size + blowup;
        let trace_log_size = evaluation_log_size - blowup;
        let composition = crate::composition_plan::CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 1,
            max_evaluation_log_size: evaluation_log_size,
            components: vec![crate::composition_plan::CompositionComponentPlan {
                component: "phase2a_protocol_fixture",
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
                        col_end: base.len(),
                    },
                    stwo::core::pcs::TreeSubspan {
                        tree_index: 2,
                        col_start: 0,
                        col_end: interaction.len(),
                    },
                ],
                preprocessed_column_indices: vec![preprocessed_index],
                trace_log_size,
                evaluation_log_size,
                n_constraints: 1,
                random_coefficient_offset: 0,
                denominator_inverses: vec![
                    stwo::core::fields::m31::BaseField::from(1);
                    1usize << blowup
                ],
                ext_param_values: Vec::new(),
                ext_param_sources: Vec::new(),
                kernels: vec![crate::composition_plan::CompositionKernelPart {
                    kernel_name: "phase2a_protocol_fixture".to_owned(),
                    cache_key: 7,
                    semantic_hash: 11,
                    source: "phase2a_protocol_fixture".to_owned(),
                    rc_base: 0,
                }],
            }],
        };
        let mut policy = ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048);
        policy.commit_mode = stwo_backend_cuda::ProgressiveCommitMode::DomainProgressive;
        policy.decommit_strategy = DecommitStrategy::RecomputeQueriedLde;
        policy.direct_composition_retention_mode = DirectCompositionRetentionMode::ExactNative;
        let geometry = plan_protocol_from_logs(
            &proof_plan,
            &claim_logs,
            &preprocessed,
            &pcs,
            false,
            policy,
            transcript_geometry(),
            &discovery,
            composition.key(),
            Some(&composition),
        )
        .unwrap();
        let direct = geometry.direct_composition_retention.as_ref().unwrap();
        assert!(direct.bindings.iter().any(|binding| {
            binding.direct
                && matches!(
                    direct.columns[binding.column].source,
                    OpenedColumnSource::Preprocessed { ordinal }
                        if ordinal as usize == preprocessed_index
                )
        }));

        let mut expected_closure = geometry
            .commitments
            .iter()
            .map(|commitment| vec![false; commitment.grouped_column_log_sizes.len()])
            .collect::<Vec<_>>();
        for binding in direct.bindings.iter().filter(|binding| binding.direct) {
            let column = direct.columns[binding.column];
            let tree = geometry
                .commitments
                .iter()
                .position(|commitment| commitment.id == column.tree)
                .unwrap();
            expected_closure[tree][column.group] = true;
        }
        assert!(geometry.commitments.iter().zip(&expected_closure).all(
            |(commitment, expected)| {
                commitment.direct_composition_evaluation_groups == *expected
            }
        ));
        let direct_words = geometry
            .commitments
            .iter()
            .try_fold(0usize, |total, commitment| {
                commitment
                    .grouped_column_log_sizes
                    .iter()
                    .zip(&commitment.direct_composition_evaluation_groups)
                    .try_fold(total, |total, (logs, &selected)| {
                        if selected {
                            total
                                .checked_add(retained_group_words(commitment, logs).unwrap())
                                .ok_or(())
                        } else {
                            Ok(total)
                        }
                    })
            })
            .unwrap();
        let direct_bytes = direct_words * core::mem::size_of::<u32>();
        assert_eq!(
            geometry.identity.direct_composition_occurrence_bitmap_hash,
            direct_bitmap_hash(&direct.direct_bitmap)
        );
        assert_eq!(
            geometry.identity.direct_composition_group_rounded_bytes,
            direct_bytes
        );
        assert_eq!(
            geometry.identity.retained_evaluation_union_bytes,
            direct_bytes
        );
        assert_eq!(
            geometry.identity.direct_composition_planner_key,
            direct.cache_key
        );

        let identity_mutations: [fn(&mut ProtocolIdentity); 4] = [
            |identity: &mut ProtocolIdentity| identity.direct_composition_planner_key ^= 1,
            |identity: &mut ProtocolIdentity| {
                identity.direct_composition_occurrence_bitmap_hash ^= 1
            },
            |identity: &mut ProtocolIdentity| {
                identity.direct_composition_group_rounded_bytes += core::mem::size_of::<u32>()
            },
            |identity: &mut ProtocolIdentity| {
                identity.retained_evaluation_union_bytes += core::mem::size_of::<u32>()
            },
        ];
        for mutate in identity_mutations {
            let mut drifted = geometry.clone();
            mutate(&mut drifted.identity);
            assert_ne!(geometry.key(), drifted.key());
        }
    }

    #[test]
    fn fixed_table_policy_and_manifest_are_part_of_identity() {
        let trace = PreProcessedTrace::canonical_small();
        let config = pcs(3);
        assert_ne!(
            preprocessed_binding_hash(&trace, &config, false),
            preprocessed_binding_hash(&trace, &config, true)
        );
        let shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let claim_logs = [vec![18], vec![18]];
        let discovery = discovery_for(&claim_logs, &trace, &config);
        assert_eq!(
            plan_protocol_from_logs(
                &plan,
                &claim_logs,
                &trace,
                &config,
                false,
                ProtocolPlanPolicy::starknet_blake2s(0, 2048),
                transcript_geometry(),
                &discovery,
                0x5678,
                None,
            ),
            Err(ProtocolPlanError::UnboundKernelManifest)
        );
    }
}
