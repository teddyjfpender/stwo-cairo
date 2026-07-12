//! Checked migration hand-off from witness-owned CUDA columns to arena slots.
//!
//! The end-state witness DAG writes these slots directly.  Until every generated
//! writer accepts an arena destination, this module performs exactly one D2D
//! copy per base coefficient column.  The copy is identity driven: raw vector
//! position is first validated against the generated claim-order layout and is
//! then resolved to `(component, trace part, purpose, ordinal)` in the arena.

use std::sync::Arc;

use stwo::core::fields::m31::BaseField;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::twiddles::{TwiddleBuffer, TwiddleTree};
use stwo::prover::poly::BitReversedOrder;
use stwo_backend_cuda::{
    synchronize_legacy_stream_for_arena_handoff, ArenaSlice, BaseFieldVec, CommitCoefficientColumn,
    CommitCoefficientGroup, CudaBackend, CudaRuntimeError, InterpolationBatch, InterpolationColumn,
    ModeAwareCommitWorkspaceRequirements, ModeAwareCommitWorkspaceSlots, PreparedCommitError,
    PreparedCommitGraph, PreparedInterpolationError, PreparedInterpolationGraph,
    PreparedProgressiveCommitError, PreparedProgressiveCommitGraph,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_cairo_prover::witness::base_trace::BaseTrace;
use stwo_cairo_prover::witness::preprocessed_trace_backend::GenPreprocessedTrace;
use stwo_cairo_prover::witness::proof_shape::ProofShapeKey;
use stwo_cairo_prover::witness::relation_sources::{
    CairoRelationSourceSet, DeviceRelationWord, RelationLookupTransfer, RelationSourceEncoding,
    RelationSourceId,
};

use crate::arena_plan::{
    BufferPurpose, CommitmentColumnSource, CommitmentTreeId, PlannedCommitment,
};
use crate::graphs::{GraphError, GraphWorkspace};
use crate::plan::ProofPlan;
use crate::protocol_plan::{trace_commitment_layout, ProtocolPlanError, TraceCommitmentColumn};
use crate::transcript_plan::CairoTranscriptInput;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentSourceStageReport {
    pub base_columns: usize,
    pub direct_base_columns: usize,
    pub migrated_base_columns: usize,
    pub base_words: usize,
    pub twiddle_words: usize,
    pub preprocessed_inverse_twiddle_words: usize,
    pub quotient_inverse_twiddle_words: usize,
    pub d2d_words: usize,
    pub d2d_bytes: usize,
    pub d2d_copies: usize,
    /// Remains true until every base witness producer is arena-native.
    pub used_migration_copy: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BaseTraceResidency {
    pub columns: usize,
    pub direct_columns: usize,
    pub migrated_columns: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentLookupStageReport {
    pub sources: usize,
    pub resident_device_sources: usize,
    pub resident_device_words: usize,
    pub host_words: usize,
    pub device_words: usize,
    pub filled_words: usize,
    pub host_copies: usize,
    pub device_copies: usize,
    pub fill_calls: usize,
    pub staged_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentPreprocessedStageReport {
    pub cache_hit: bool,
    pub columns: usize,
    pub coefficient_words: usize,
    pub evaluation_words: usize,
    pub d2d_bytes: usize,
    pub d2d_copies: usize,
    pub descriptor_h2d_bytes: usize,
    pub descriptor_h2d_copies: usize,
    pub interpolation_batches: usize,
    pub commitment_launches: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentTwiddleStageReport {
    pub cache_hit: bool,
    pub forward_words: usize,
    pub preprocessed_inverse_words: usize,
    pub inverse_words: usize,
    pub quotient_inverse_words: usize,
    pub d2d_bytes: usize,
    pub d2d_copies: usize,
    pub sync_calls: usize,
}

#[derive(Debug)]
pub enum ResidentSourceStageError {
    WorkspaceShapeMismatch {
        expected: ProofShapeKey,
        actual: ProofShapeKey,
    },
    Protocol(ProtocolPlanError),
    ColumnCountMismatch {
        expected: usize,
        actual: usize,
    },
    ColumnLogMismatch {
        column: usize,
        expected: u32,
        actual: u32,
    },
    ColumnValueSizeMismatch {
        column: usize,
        expected_words: usize,
        actual_words: usize,
    },
    BaseEvaluationsUnavailable,
    PreprocessedColumnCountMismatch {
        expected: usize,
        actual: usize,
    },
    PreprocessedColumnLogMismatch {
        column: usize,
        expected: u32,
        actual: u32,
    },
    MissingPreprocessedColumn(u32),
    FixedTwiddlesNotReady,
    MissingPreprocessedRootInput,
    PreprocessedRootBindingMismatch {
        expected_words: usize,
        actual_words: usize,
    },
    PreprocessedCommitBindingMismatch,
    MissingCommitment(CommitmentTreeId),
    InterpolationBatchShapeMismatch(CommitmentTreeId),
    InvalidInterpolationSource {
        tree: CommitmentTreeId,
        source: CommitmentColumnSource,
    },
    MissingArenaSource(CommitmentColumnSource),
    MissingGlobalArenaSource(BufferPurpose),
    ArenaSourceSizeMismatch {
        source: CommitmentColumnSource,
        expected_words: usize,
        actual_words: usize,
    },
    TwiddleSourceSizeMismatch {
        purpose: BufferPurpose,
        expected_words: usize,
        actual_words: usize,
    },
    InvalidTwiddleSubdomain {
        purpose: BufferPurpose,
        source_words: usize,
        domain_log_size: u32,
        subdomain_log_size: u32,
        expected_words: usize,
    },
    MissingLookupDestination(RelationSourceId),
    UnexpectedLookupDestination(RelationSourceId),
    LookupGeometryMismatch {
        id: RelationSourceId,
        expected_words: usize,
        actual_words: usize,
    },
    DuplicateLookupSource(RelationSourceId),
    UnstagedLookupSource {
        component: &'static str,
        part: stwo_cairo_prover::witness::proof_shape::TracePartId,
    },
    SizeOverflow,
    Graph(GraphError),
    Runtime(CudaRuntimeError),
    Interpolation(PreparedInterpolationError),
    Commit(PreparedCommitError),
    ProgressiveCommit(PreparedProgressiveCommitError),
}

impl core::fmt::Display for ResidentSourceStageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident witness source staging rejected: {self:?}")
    }
}

impl std::error::Error for ResidentSourceStageError {}

impl From<ProtocolPlanError> for ResidentSourceStageError {
    fn from(value: ProtocolPlanError) -> Self {
        Self::Protocol(value)
    }
}

impl From<GraphError> for ResidentSourceStageError {
    fn from(value: GraphError) -> Self {
        Self::Graph(value)
    }
}

impl From<CudaRuntimeError> for ResidentSourceStageError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<PreparedInterpolationError> for ResidentSourceStageError {
    fn from(value: PreparedInterpolationError) -> Self {
        Self::Interpolation(value)
    }
}

impl From<PreparedCommitError> for ResidentSourceStageError {
    fn from(value: PreparedCommitError) -> Self {
        Self::Commit(value)
    }
}

impl From<PreparedProgressiveCommitError> for ResidentSourceStageError {
    fn from(value: PreparedProgressiveCommitError) -> Self {
        Self::ProgressiveCommit(value)
    }
}

/// Consume canonical base evaluations and populate both resident trace forms.
///
/// `BaseTrace::Polys` is rejected: already-interpolated coefficients cannot
/// reconstruct the canonical evaluations required by memory/xor relation
/// kernels. All source identities and logs are validated before the first
/// migration copy. Evaluation-to-coefficient interpolation then runs in place in
/// the distinct coefficient slots on the arena's explicit stream.
pub fn stage_base_trace_coefficients(
    workspace: &mut GraphWorkspace,
    proof_plan: &ProofPlan,
    trace: BaseTrace<CudaBackend>,
    twiddles: &'static TwiddleTree<CudaBackend>,
) -> Result<ResidentSourceStageReport, ResidentSourceStageError> {
    if workspace.plan().shape_key != proof_plan.shape_key {
        return Err(ResidentSourceStageError::WorkspaceShapeMismatch {
            expected: proof_plan.shape_key,
            actual: workspace.plan().shape_key,
        });
    }

    let residency = inspect_base_trace_residency(workspace, proof_plan, &trace)?;
    let columns = trace_commitment_layout(proof_plan)?.base;
    let evals = require_base_evaluations(trace)?;
    validate_evaluations(&columns, &evals)?;

    let stage_fixed_twiddles = !workspace.fixed_twiddles_ready();
    let interpolation = prepare_commitment_interpolation(workspace, CommitmentTreeId::Base)?;

    let mut copies = Vec::with_capacity(columns.len() + 4);
    let mut base_words = 0usize;
    let mut direct_base_columns = 0usize;
    let mut migration_base_words = 0usize;
    for (column, eval) in columns.iter().copied().zip(&evals) {
        // The migration source is an evaluation. Stage it into the evaluation
        // slot; `PreparedInterpolationGraph` then copies that value into the
        // distinct coefficient slot before applying the inverse transform.
        let (destination, _) = bind_trace_pair(workspace, CommitmentTreeId::Base, column.source)?;
        let logical = destination.len_words();
        let expected_words = checked_words(column.log_size)?;
        if logical != expected_words {
            return Err(ResidentSourceStageError::ArenaSourceSizeMismatch {
                source: column.source,
                expected_words,
                actual_words: logical,
            });
        }
        base_words = base_words
            .checked_add(expected_words)
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        if destination.as_u32_ptr().cast_const() == eval.values.device_ptr {
            direct_base_columns += 1;
        } else {
            migration_base_words = migration_base_words
                .checked_add(expected_words)
                .ok_or(ResidentSourceStageError::SizeOverflow)?;
            copies.push((destination, eval.values.device_ptr, expected_words));
        }
    }

    let mut twiddle_words = 0usize;
    let mut preprocessed_inverse_words = 0usize;
    let mut quotient_inverse_words = 0usize;
    // These buffers are protocol-keyed fixed oracles. Recopying roughly a full
    // lifting domain on every warm proof would defeat residency, so a cached
    // workspace stages them once and retries only after a failed setup.
    let (protocol_inverse, quotient_inverse) = if stage_fixed_twiddles {
        let (forward_destination, forward_words) =
            bind_global_source(workspace, BufferPurpose::ForwardTwiddles)?;
        if twiddles.twiddles.size != forward_words {
            return Err(ResidentSourceStageError::TwiddleSourceSizeMismatch {
                purpose: BufferPurpose::ForwardTwiddles,
                expected_words: forward_words,
                actual_words: twiddles.twiddles.size,
            });
        }
        twiddle_words = twiddle_words
            .checked_add(forward_words)
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        copies.push((
            forward_destination,
            twiddles.twiddles.device_ptr,
            forward_words,
        ));

        let (preprocessed_inverse_destination, expected_preprocessed_inverse_words) =
            bind_global_source(workspace, BufferPurpose::PreprocessedInverseTwiddles)?;
        if twiddles.itwiddles.size != expected_preprocessed_inverse_words {
            return Err(ResidentSourceStageError::TwiddleSourceSizeMismatch {
                purpose: BufferPurpose::PreprocessedInverseTwiddles,
                expected_words: expected_preprocessed_inverse_words,
                actual_words: twiddles.itwiddles.size,
            });
        }
        preprocessed_inverse_words = expected_preprocessed_inverse_words;
        twiddle_words = twiddle_words
            .checked_add(preprocessed_inverse_words)
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        copies.push((
            preprocessed_inverse_destination,
            twiddles.itwiddles.device_ptr,
            preprocessed_inverse_words,
        ));

        let (inverse_destination, inverse_words) =
            bind_global_source(workspace, BufferPurpose::InverseTwiddles)?;
        let inverse_log_size = workspace.plan().fri().config.circle_log_size;
        let protocol_inverse = extract_inverse_subdomain(
            &twiddles.itwiddles,
            BufferPurpose::InverseTwiddles,
            inverse_log_size,
            inverse_log_size,
            inverse_words,
        )?;
        twiddle_words = twiddle_words
            .checked_add(inverse_words)
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        copies.push((
            inverse_destination,
            protocol_inverse.device_ptr,
            inverse_words,
        ));

        // Quotient interpolation runs on the first canonic subdomain. Its
        // inverse twiddles are a layer-wise extraction, not a prefix.
        let quotient = workspace.plan().quotient();
        quotient_inverse_words = quotient.requirements.inverse_twiddle_words;
        let (destination, expected_words) =
            bind_global_source(workspace, BufferPurpose::QuotientInverseTwiddles)?;
        if quotient_inverse_words != expected_words {
            return Err(ResidentSourceStageError::TwiddleSourceSizeMismatch {
                purpose: BufferPurpose::QuotientInverseTwiddles,
                expected_words,
                actual_words: quotient_inverse_words,
            });
        }
        let quotient_inverse = extract_inverse_subdomain(
            &twiddles.itwiddles,
            BufferPurpose::QuotientInverseTwiddles,
            quotient.config.lifting_log_size,
            quotient.requirements.subdomain_log_size,
            quotient_inverse_words,
        )?;
        twiddle_words = twiddle_words
            .checked_add(quotient_inverse_words)
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        copies.push((
            destination,
            quotient_inverse.device_ptr,
            quotient_inverse_words,
        ));
        (Some(protocol_inverse), Some(quotient_inverse))
    } else {
        (None, None)
    };

    // Interpolation copies each resident evaluation into its distinct
    // coefficient slot once. Only non-aliased source columns add a migration
    // copy into BaseTrace first.
    let d2d_words = base_words
        .checked_add(migration_base_words)
        .and_then(|words| words.checked_add(twiddle_words))
        .ok_or(ResidentSourceStageError::SizeOverflow)?;

    synchronize_legacy_stream_for_arena_handoff();
    for (destination, source, words) in copies {
        let bytes = words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        // SAFETY: the metadata pass proved both live ranges contain `words`
        // u32s. Sources are owned by `polys` until the final context sync and
        // destinations are stable, non-overlapping arena ranges.
        unsafe {
            workspace.arena().context().memcpy_d2d_async(
                destination.as_void_ptr(),
                source.cast(),
                bytes,
            )?;
        }
    }
    interpolation.launch()?;
    workspace.arena().context().sync()?;
    drop(interpolation);
    drop(protocol_inverse);
    drop(quotient_inverse);
    if stage_fixed_twiddles {
        workspace.mark_fixed_twiddles_ready();
    }
    debug_assert_eq!(direct_base_columns, residency.direct_columns);

    Ok(ResidentSourceStageReport {
        base_columns: columns.len(),
        direct_base_columns,
        migrated_base_columns: columns.len() - direct_base_columns,
        base_words,
        twiddle_words,
        preprocessed_inverse_twiddle_words: preprocessed_inverse_words,
        quotient_inverse_twiddle_words: quotient_inverse_words,
        d2d_words,
        d2d_bytes: d2d_words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ResidentSourceStageError::SizeOverflow)?,
        d2d_copies: columns.len()
            + (columns.len() - direct_base_columns)
            + usize::from(stage_fixed_twiddles) * 4,
        used_migration_copy: direct_base_columns != columns.len(),
    })
}

/// Validate the entire canonical base trace and classify arena aliases without
/// issuing a copy. Strict callers use this before staging so a detached writer
/// is rejected before the first migration operation is enqueued.
pub fn inspect_base_trace_residency(
    workspace: &GraphWorkspace,
    proof_plan: &ProofPlan,
    trace: &BaseTrace<CudaBackend>,
) -> Result<BaseTraceResidency, ResidentSourceStageError> {
    if workspace.plan().shape_key != proof_plan.shape_key {
        return Err(ResidentSourceStageError::WorkspaceShapeMismatch {
            expected: proof_plan.shape_key,
            actual: workspace.plan().shape_key,
        });
    }
    let columns = trace_commitment_layout(proof_plan)?.base;
    let BaseTrace::Evals(evals) = trace else {
        return Err(ResidentSourceStageError::BaseEvaluationsUnavailable);
    };
    validate_evaluations(&columns, evals)?;
    let mut direct_columns = 0usize;
    for (column, eval) in columns.iter().zip(evals) {
        let (destination, _) = bind_trace_pair(workspace, CommitmentTreeId::Base, column.source)?;
        if destination.as_u32_ptr().cast_const() == eval.values.device_ptr {
            direct_columns += 1;
        }
    }
    Ok(residency_counts(columns.len(), direct_columns))
}

fn residency_counts(columns: usize, direct_columns: usize) -> BaseTraceResidency {
    BaseTraceResidency {
        columns,
        direct_columns,
        migrated_columns: columns - direct_columns,
    }
}

/// Stage the protocol-keyed twiddle oracle without requiring a materialized
/// base trace. Strict Graph A calls this before witness capture; all dynamic
/// base evaluations are born later in the arena and interpolation is part of
/// the captured graph.
pub fn stage_protocol_twiddles(
    workspace: &mut GraphWorkspace,
    twiddles: &'static TwiddleTree<CudaBackend>,
) -> Result<ResidentTwiddleStageReport, ResidentSourceStageError> {
    if workspace.fixed_twiddles_ready() {
        return Ok(ResidentTwiddleStageReport {
            cache_hit: true,
            ..ResidentTwiddleStageReport::default()
        });
    }

    let mut copies = Vec::with_capacity(4);
    let mut report = ResidentTwiddleStageReport::default();
    let (forward_destination, forward_words) =
        bind_global_source(workspace, BufferPurpose::ForwardTwiddles)?;
    if twiddles.twiddles.size != forward_words {
        return Err(ResidentSourceStageError::TwiddleSourceSizeMismatch {
            purpose: BufferPurpose::ForwardTwiddles,
            expected_words: forward_words,
            actual_words: twiddles.twiddles.size,
        });
    }
    report.forward_words = forward_words;
    copies.push((
        forward_destination,
        twiddles.twiddles.device_ptr,
        forward_words,
    ));

    let (preprocessed_inverse_destination, preprocessed_inverse_words) =
        bind_global_source(workspace, BufferPurpose::PreprocessedInverseTwiddles)?;
    if twiddles.itwiddles.size != preprocessed_inverse_words {
        return Err(ResidentSourceStageError::TwiddleSourceSizeMismatch {
            purpose: BufferPurpose::PreprocessedInverseTwiddles,
            expected_words: preprocessed_inverse_words,
            actual_words: twiddles.itwiddles.size,
        });
    }
    report.preprocessed_inverse_words = preprocessed_inverse_words;
    copies.push((
        preprocessed_inverse_destination,
        twiddles.itwiddles.device_ptr,
        preprocessed_inverse_words,
    ));

    let (inverse_destination, inverse_words) =
        bind_global_source(workspace, BufferPurpose::InverseTwiddles)?;
    let inverse_log_size = workspace.plan().fri().config.circle_log_size;
    let protocol_inverse = extract_inverse_subdomain(
        &twiddles.itwiddles,
        BufferPurpose::InverseTwiddles,
        inverse_log_size,
        inverse_log_size,
        inverse_words,
    )?;
    report.inverse_words = inverse_words;
    copies.push((
        inverse_destination,
        protocol_inverse.device_ptr,
        inverse_words,
    ));

    let quotient = workspace.plan().quotient();
    report.quotient_inverse_words = quotient.requirements.inverse_twiddle_words;
    let (quotient_destination, expected_words) =
        bind_global_source(workspace, BufferPurpose::QuotientInverseTwiddles)?;
    if report.quotient_inverse_words != expected_words {
        return Err(ResidentSourceStageError::TwiddleSourceSizeMismatch {
            purpose: BufferPurpose::QuotientInverseTwiddles,
            expected_words,
            actual_words: report.quotient_inverse_words,
        });
    }
    let quotient_inverse = extract_inverse_subdomain(
        &twiddles.itwiddles,
        BufferPurpose::QuotientInverseTwiddles,
        quotient.config.lifting_log_size,
        quotient.requirements.subdomain_log_size,
        expected_words,
    )?;
    copies.push((
        quotient_destination,
        quotient_inverse.device_ptr,
        expected_words,
    ));

    synchronize_legacy_stream_for_arena_handoff();
    for (destination, source, words) in &copies {
        let bytes = words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        unsafe {
            workspace.arena().context().memcpy_d2d_async(
                destination.as_void_ptr(),
                source.cast(),
                bytes,
            )?;
        }
        report.d2d_bytes = report
            .d2d_bytes
            .checked_add(bytes)
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
    }
    workspace.arena().context().sync()?;
    report.d2d_copies = copies.len();
    report.sync_calls = 1;
    drop(protocol_inverse);
    drop(quotient_inverse);
    workspace.mark_fixed_twiddles_ready();
    Ok(report)
}

/// Populate and commit the workspace's fixed coefficient oracle exactly once.
///
/// Fixed evaluations may be generated by the existing persistent-table setup
/// lane, but no per-proof consumer may retain those detached allocations. This
/// handoff copies each canonical column into its protocol-keyed arena slot,
/// interpolates same-log batches, commits the canonical tree and stages its root
/// into the device transcript. The ready bit is set only after the final
/// synchronization, so a failed partial setup is retried.
pub fn stage_preprocessed_commitment(
    workspace: &mut GraphWorkspace,
    trace: Arc<PreProcessedTrace>,
) -> Result<ResidentPreprocessedStageReport, ResidentSourceStageError> {
    let planned = workspace.plan().preprocessed().clone();
    let commitment = workspace
        .plan()
        .commitment(CommitmentTreeId::Preprocessed)
        .cloned()
        .ok_or(ResidentSourceStageError::MissingCommitment(
            CommitmentTreeId::Preprocessed,
        ))?;
    if workspace.preprocessed_commitment_ready() {
        return Ok(ResidentPreprocessedStageReport {
            cache_hit: true,
            columns: planned.columns.len(),
            coefficient_words: planned
                .columns
                .iter()
                .try_fold(0usize, |sum, column| {
                    sum.checked_add(column.coefficients.len_words)
                })
                .ok_or(ResidentSourceStageError::SizeOverflow)?,
            evaluation_words: planned.columns.iter().try_fold(0usize, |sum, column| {
                column.evaluations.map_or(Ok(sum), |evaluation| {
                    sum.checked_add(evaluation.len_words)
                        .ok_or(ResidentSourceStageError::SizeOverflow)
                })
            })?,
            ..ResidentPreprocessedStageReport::default()
        });
    }
    if !workspace.fixed_twiddles_ready() {
        return Err(ResidentSourceStageError::FixedTwiddlesNotReady);
    }

    let evaluations = <CudaBackend as GenPreprocessedTrace>::gen_preprocessed_trace(trace);
    if evaluations.len() != planned.columns.len() {
        return Err(ResidentSourceStageError::PreprocessedColumnCountMismatch {
            expected: planned.columns.len(),
            actual: evaluations.len(),
        });
    }

    let mut coefficient_words = 0usize;
    let mut evaluation_words = 0usize;
    let mut destinations = Vec::with_capacity(planned.columns.len());
    let mut evaluation_destinations = Vec::with_capacity(planned.columns.len());
    for (column, evaluation) in planned.columns.iter().zip(&evaluations) {
        let actual_log = evaluation.domain.log_size();
        if actual_log != column.log_size {
            return Err(ResidentSourceStageError::PreprocessedColumnLogMismatch {
                column: column.ordinal as usize,
                expected: column.log_size,
                actual: actual_log,
            });
        }
        let expected_words = checked_words(column.log_size)?;
        if evaluation.values.size != expected_words
            || column.coefficients.len_words != expected_words
        {
            return Err(ResidentSourceStageError::ColumnValueSizeMismatch {
                column: column.ordinal as usize,
                expected_words,
                actual_words: evaluation.values.size,
            });
        }
        coefficient_words = coefficient_words
            .checked_add(expected_words)
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        destinations.push(workspace.bind(column.coefficients.logical)?.0);
        let retained = column
            .evaluations
            .map(|binding| workspace.bind(binding.logical).map(|bound| bound.0))
            .transpose()?;
        if let Some(retained) = retained {
            if retained.len_words() != expected_words {
                return Err(ResidentSourceStageError::ColumnValueSizeMismatch {
                    column: column.ordinal as usize,
                    expected_words,
                    actual_words: retained.len_words(),
                });
            }
            evaluation_words = evaluation_words
                .checked_add(expected_words)
                .ok_or(ResidentSourceStageError::SizeOverflow)?;
        }
        evaluation_destinations.push(retained);
    }

    synchronize_legacy_stream_for_arena_handoff();
    for ((column, evaluation), destination) in
        planned.columns.iter().zip(&evaluations).zip(&destinations)
    {
        let bytes = checked_words(column.log_size)?
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        unsafe {
            workspace.arena().context().memcpy_d2d_async(
                destination.as_void_ptr(),
                evaluation.values.device_ptr.cast(),
                bytes,
            )?;
        }
    }
    for (evaluation, destination) in evaluations.iter().zip(&evaluation_destinations) {
        let Some(destination) = destination else {
            continue;
        };
        let bytes = evaluation
            .values
            .size
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        unsafe {
            workspace.arena().context().memcpy_d2d_async(
                destination.as_void_ptr(),
                evaluation.values.device_ptr.cast(),
                bytes,
            )?;
        }
    }

    let inverse_twiddles = workspace.bind(planned.inverse_twiddles.logical)?.0;
    let inverse_words = u32::try_from(inverse_twiddles.len_words())
        .map_err(|_| ResidentSourceStageError::SizeOverflow)?;
    let stream = workspace.arena().context().stream_raw().as_ptr();
    let mut descriptor_storage = Vec::with_capacity(planned.interpolation_batches.len());
    let mut descriptor_h2d_bytes = 0usize;
    for batch in &planned.interpolation_batches {
        let pointers = batch
            .column_ordinals
            .iter()
            .map(|&ordinal| {
                planned
                    .columns
                    .iter()
                    .position(|column| column.ordinal == ordinal)
                    .map(|index| destinations[index].as_u32_ptr() as usize)
                    .ok_or(ResidentSourceStageError::MissingPreprocessedColumn(ordinal))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pointer_table = workspace.bind(batch.coefficient_pointers.logical)?.0;
        let bytes = pointers
            .len()
            .checked_mul(core::mem::size_of::<usize>())
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        descriptor_h2d_bytes = descriptor_h2d_bytes
            .checked_add(bytes)
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        unsafe {
            workspace.arena().context().memcpy_h2d_async(
                pointer_table.as_void_ptr(),
                pointers.as_ptr().cast(),
                bytes,
            )?;
        }
        descriptor_storage.push(pointers);
        let count = u32::try_from(batch.column_ordinals.len())
            .map_err(|_| ResidentSourceStageError::SizeOverflow)?;
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_ntt_b2n_columns_on(
                pointer_table.as_u32_ptr().cast::<*mut u32>(),
                batch.log_size,
                count,
                inverse_twiddles.as_u32_ptr(),
                inverse_words,
                1u32 << (batch.log_size - 1),
                stream,
            )
        };
        if code != 0 {
            return Err(ResidentSourceStageError::Runtime(CudaRuntimeError::Cuda {
                operation: "preprocessed_interpolation",
                code,
            }));
        }
    }
    let groups = commitment_groups(workspace, &commitment)?;
    let twiddles = workspace.bind(commitment.twiddles.logical)?.0;
    let (root, retained) = match (&commitment.requirements, &commitment.slots) {
        (
            ModeAwareCommitWorkspaceRequirements::FullLifting(_),
            ModeAwareCommitWorkspaceSlots::FullLifting(slots),
        ) => {
            let commit = PreparedCommitGraph::prepare(
                workspace.arena(),
                commitment.config,
                &groups,
                twiddles,
                slots,
            )?;
            commit.launch()?;
            (
                commit.root_slice(),
                commit.retained_layers_bottom_up().to_vec(),
            )
        }
        (
            ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements),
            ModeAwareCommitWorkspaceSlots::DomainProgressive(slots),
        ) => {
            let coefficients = groups
                .iter()
                .flat_map(|group| group.columns.iter().copied())
                .collect::<Vec<_>>();
            let retained_outputs = vec![None; coefficients.len()];
            let commit = PreparedProgressiveCommitGraph::prepare(
                workspace.arena(),
                commitment.config,
                requirements,
                slots,
                &coefficients,
                &retained_outputs,
                twiddles,
            )?;
            commit.launch()?;
            (
                commit.root_slice(),
                commit.retained_layers_bottom_up().to_vec(),
            )
        }
        _ => return Err(ResidentSourceStageError::PreprocessedCommitBindingMismatch),
    };
    if root.id() != commitment.root.physical
        || retained.len() != commitment.retained_layers_bottom_up.len()
        || retained
            .iter()
            .zip(&commitment.retained_layers_bottom_up)
            .any(|(actual, expected)| {
                actual.id() != expected.physical || actual.len_words() < expected.len_words
            })
    {
        return Err(ResidentSourceStageError::PreprocessedCommitBindingMismatch);
    }
    let root_input = CairoTranscriptInput::PreprocessedRoot
        .id()
        .map_err(|_| ResidentSourceStageError::MissingPreprocessedRootInput)?;
    let root_binding = workspace
        .plan()
        .transcript()
        .inputs
        .iter()
        .find_map(|(id, binding)| (*id == root_input).then_some(*binding))
        .ok_or(ResidentSourceStageError::MissingPreprocessedRootInput)?;
    if root_binding.len_words != 8 {
        return Err(ResidentSourceStageError::PreprocessedRootBindingMismatch {
            expected_words: 8,
            actual_words: root_binding.len_words,
        });
    }
    let root_destination = workspace.bind(root_binding.logical)?.0;
    unsafe {
        workspace.arena().context().memcpy_d2d_async(
            root_destination.as_void_ptr(),
            root.as_void_ptr().cast_const(),
            8 * core::mem::size_of::<u32>(),
        )?;
    }
    workspace.arena().context().sync()?;
    drop(descriptor_storage);
    drop(evaluations);
    let (commit_descriptor_bytes, commit_descriptor_copies) =
        commitment_descriptor_transfers(&commitment)?;
    let report = ResidentPreprocessedStageReport {
        cache_hit: false,
        columns: planned.columns.len(),
        coefficient_words,
        evaluation_words,
        d2d_bytes: coefficient_words
            .checked_add(evaluation_words)
            .and_then(|words| words.checked_mul(core::mem::size_of::<u32>()))
            .and_then(|bytes| bytes.checked_add(8 * core::mem::size_of::<u32>()))
            .ok_or(ResidentSourceStageError::SizeOverflow)?,
        d2d_copies: planned
            .columns
            .len()
            .checked_add(
                planned
                    .columns
                    .iter()
                    .filter(|column| column.evaluations.is_some())
                    .count(),
            )
            .and_then(|copies| copies.checked_add(1))
            .ok_or(ResidentSourceStageError::SizeOverflow)?,
        descriptor_h2d_bytes: descriptor_h2d_bytes
            .checked_add(commit_descriptor_bytes)
            .ok_or(ResidentSourceStageError::SizeOverflow)?,
        descriptor_h2d_copies: planned
            .interpolation_batches
            .len()
            .checked_add(commit_descriptor_copies)
            .ok_or(ResidentSourceStageError::SizeOverflow)?,
        interpolation_batches: planned.interpolation_batches.len(),
        commitment_launches: 1,
    };
    workspace.mark_preprocessed_commitment_ready();
    Ok(report)
}

/// Populate every generated `LookupWords` arena source in one checked batch.
/// Non-lookup special relations (memory and xor tables) are intentionally left
/// on their base-trace identities and are validated to have no lookup slot.
pub fn stage_relation_lookup_sources(
    workspace: &GraphWorkspace,
    sources: &CairoRelationSourceSet,
) -> Result<ResidentLookupStageReport, ResidentSourceStageError> {
    let mut staged = Vec::<RelationSourceId>::new();
    let mut report = ResidentLookupStageReport::default();
    let mut has_device_sources = false;

    // Validate every source and destination before issuing the first transfer.
    for source in sources.as_slice() {
        let destination = workspace.plan().find(
            Some(source.id.component),
            Some(source.id.part),
            BufferPurpose::LookupInputs,
            0,
        );
        if source.encoding != RelationSourceEncoding::LookupWords {
            if destination.is_some() {
                return Err(ResidentSourceStageError::UnexpectedLookupDestination(
                    source.id,
                ));
            }
            continue;
        }
        if staged.contains(&source.id) {
            return Err(ResidentSourceStageError::DuplicateLookupSource(source.id));
        }
        let (logical, _) = destination.ok_or(
            ResidentSourceStageError::MissingLookupDestination(source.id),
        )?;
        let expected_words = usize::try_from(source.padded_rows)
            .ok()
            .and_then(|rows| rows.checked_mul(source.words_per_row))
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        if logical.len_words != expected_words {
            return Err(ResidentSourceStageError::LookupGeometryMismatch {
                id: source.id,
                expected_words,
                actual_words: logical.len_words,
            });
        }
        match &source.transfer {
            RelationLookupTransfer::HostWordMajor(words) => {
                if words.len() != expected_words {
                    return Err(ResidentSourceStageError::LookupGeometryMismatch {
                        id: source.id,
                        expected_words,
                        actual_words: words.len(),
                    });
                }
            }
            RelationLookupTransfer::DeviceWordMajor(words) => {
                has_device_sources = true;
                if words.size != expected_words {
                    return Err(ResidentSourceStageError::LookupGeometryMismatch {
                        id: source.id,
                        expected_words,
                        actual_words: words.size,
                    });
                }
            }
            RelationLookupTransfer::DeviceProjectedWords(words) => {
                has_device_sources = true;
                if words.len() != source.words_per_row {
                    return Err(ResidentSourceStageError::LookupGeometryMismatch {
                        id: source.id,
                        expected_words: source.words_per_row,
                        actual_words: words.len(),
                    });
                }
            }
        }
        staged.push(source.id);
    }
    for logical in workspace
        .plan()
        .logical_buffers()
        .iter()
        .filter(|buffer| buffer.purpose == BufferPurpose::LookupInputs)
    {
        let id = RelationSourceId {
            component: logical
                .component
                .expect("component lookup buffers are component-owned"),
            part: logical
                .part
                .expect("component lookup buffers have a trace part"),
        };
        if !staged.contains(&id) {
            return Err(ResidentSourceStageError::UnstagedLookupSource {
                component: id.component,
                part: id.part,
            });
        }
    }

    if has_device_sources {
        synchronize_legacy_stream_for_arena_handoff();
    }
    for source in sources
        .as_slice()
        .iter()
        .filter(|source| source.encoding == RelationSourceEncoding::LookupWords)
    {
        let (logical, _) = workspace
            .plan()
            .find(
                Some(source.id.component),
                Some(source.id.part),
                BufferPurpose::LookupInputs,
                0,
            )
            .ok_or(ResidentSourceStageError::MissingLookupDestination(
                source.id,
            ))?;
        let (destination, logical_words) = workspace.bind(logical.id)?;
        let rows = usize::try_from(source.padded_rows)
            .map_err(|_| ResidentSourceStageError::SizeOverflow)?;
        match &source.transfer {
            RelationLookupTransfer::HostWordMajor(words) => {
                // SAFETY: the validation pass proved the host and logical arena
                // ranges have the same exact word count; the final sync keeps
                // pageable/pinned host storage alive through the transfer.
                unsafe {
                    workspace.arena().context().memcpy_h2d_async(
                        destination.as_void_ptr(),
                        words.as_ptr().cast(),
                        logical_words * core::mem::size_of::<u32>(),
                    )?;
                }
                report.host_words = report
                    .host_words
                    .checked_add(logical_words)
                    .ok_or(ResidentSourceStageError::SizeOverflow)?;
                report.host_copies += 1;
            }
            RelationLookupTransfer::DeviceWordMajor(words) => {
                if destination.as_u32_ptr().cast_const() == words.device_ptr {
                    report.resident_device_sources += 1;
                    report.resident_device_words = report
                        .resident_device_words
                        .checked_add(logical_words)
                        .ok_or(ResidentSourceStageError::SizeOverflow)?;
                } else {
                    unsafe {
                        workspace.arena().context().memcpy_d2d_async(
                            destination.as_void_ptr(),
                            words.device_ptr.cast(),
                            logical_words * core::mem::size_of::<u32>(),
                        )?;
                    }
                    report.device_words = report
                        .device_words
                        .checked_add(logical_words)
                        .ok_or(ResidentSourceStageError::SizeOverflow)?;
                    report.device_copies += 1;
                }
            }
            RelationLookupTransfer::DeviceProjectedWords(words) => {
                for (word, source_word) in words.iter().enumerate() {
                    let offset = word
                        .checked_mul(rows)
                        .ok_or(ResidentSourceStageError::SizeOverflow)?;
                    // SAFETY: word-major geometry was validated before launch;
                    // each projected destination is one disjoint row column.
                    let output = unsafe { destination.as_u32_ptr().add(offset) };
                    match source_word {
                        DeviceRelationWord::Column(column) => {
                            unsafe {
                                workspace.arena().context().memcpy_d2d_async(
                                    output.cast(),
                                    column.device_ptr.cast(),
                                    rows * core::mem::size_of::<u32>(),
                                )?;
                            }
                            report.device_words = report
                                .device_words
                                .checked_add(rows)
                                .ok_or(ResidentSourceStageError::SizeOverflow)?;
                            report.device_copies += 1;
                        }
                        DeviceRelationWord::Constant(value) => {
                            unsafe {
                                workspace
                                    .arena()
                                    .context()
                                    .fill_u32_async(output, *value, rows)?;
                            }
                            report.filled_words = report
                                .filled_words
                                .checked_add(rows)
                                .ok_or(ResidentSourceStageError::SizeOverflow)?;
                            report.fill_calls += 1;
                        }
                    }
                }
            }
        }
        report.sources += 1;
    }
    workspace.arena().context().sync()?;
    report.staged_bytes = report
        .host_words
        .checked_add(report.device_words)
        .and_then(|words| words.checked_add(report.filled_words))
        .and_then(|words| words.checked_mul(core::mem::size_of::<u32>()))
        .ok_or(ResidentSourceStageError::SizeOverflow)?;
    Ok(report)
}

type CudaBaseEvaluation = CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>;

fn require_base_evaluations(
    trace: BaseTrace<CudaBackend>,
) -> Result<Vec<CudaBaseEvaluation>, ResidentSourceStageError> {
    match trace {
        BaseTrace::Evals(evals) => Ok(evals),
        BaseTrace::Polys { .. } => Err(ResidentSourceStageError::BaseEvaluationsUnavailable),
    }
}

fn validate_evaluations(
    columns: &[TraceCommitmentColumn],
    evals: &[CudaBaseEvaluation],
) -> Result<(), ResidentSourceStageError> {
    if columns.len() != evals.len() {
        return Err(ResidentSourceStageError::ColumnCountMismatch {
            expected: columns.len(),
            actual: evals.len(),
        });
    }
    for (column, (expected, eval)) in columns.iter().zip(evals).enumerate() {
        if expected.log_size != eval.domain.log_size() {
            return Err(ResidentSourceStageError::ColumnLogMismatch {
                column,
                expected: expected.log_size,
                actual: eval.domain.log_size(),
            });
        }
        let expected_words = checked_words(expected.log_size)?;
        if eval.values.size != expected_words {
            return Err(ResidentSourceStageError::ColumnValueSizeMismatch {
                column,
                expected_words,
                actual_words: eval.values.size,
            });
        }
    }
    Ok(())
}

pub(crate) fn commitment_groups(
    workspace: &GraphWorkspace,
    planned: &PlannedCommitment,
) -> Result<Vec<CommitCoefficientGroup>, ResidentSourceStageError> {
    planned
        .grouped_column_sources
        .iter()
        .zip(&planned.grouped_column_log_sizes)
        .map(|(sources, logs)| {
            let columns = sources
                .iter()
                .zip(logs)
                .map(|(&source, &log_size)| {
                    let binding = match source {
                        CommitmentColumnSource::Preprocessed { ordinal } => workspace
                            .plan()
                            .find(None, None, BufferPurpose::PreprocessedCoefficients, ordinal)
                            .map(|(_, binding)| binding),
                        CommitmentColumnSource::Trace {
                            component,
                            part,
                            purpose,
                            ordinal,
                        } => workspace
                            .plan()
                            .find(Some(component), Some(part), purpose, ordinal)
                            .map(|(_, binding)| binding),
                        CommitmentColumnSource::Composition { ordinal } => workspace
                            .plan()
                            .find(None, None, BufferPurpose::CompositionCoefficients, ordinal)
                            .map(|(_, binding)| binding),
                    }
                    .ok_or(ResidentSourceStageError::MissingArenaSource(source))?;
                    let expected_words = checked_words(log_size)?;
                    if binding.len_words != expected_words {
                        return Err(ResidentSourceStageError::ArenaSourceSizeMismatch {
                            source,
                            expected_words,
                            actual_words: binding.len_words,
                        });
                    }
                    Ok(CommitCoefficientColumn {
                        coefficients: workspace.bind(binding.logical)?.0,
                        log_size,
                    })
                })
                .collect::<Result<Vec<_>, ResidentSourceStageError>>()?;
            Ok(CommitCoefficientGroup { columns })
        })
        .collect()
}

fn commitment_descriptor_transfers(
    planned: &PlannedCommitment,
) -> Result<(usize, usize), ResidentSourceStageError> {
    let word_bytes = core::mem::size_of::<u32>();
    let (tail_words, groups, progressive_batches) = match &planned.requirements {
        ModeAwareCommitWorkspaceRequirements::FullLifting(requirements) => (
            requirements.tail_pointer_words,
            Some(requirements.groups.as_slice()),
            None,
        ),
        ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements) => (
            requirements.merkle.tail_pointer_words,
            None,
            Some(requirements.leaves.batches.as_slice()),
        ),
    };
    let mut bytes = tail_words
        .unwrap_or_default()
        .checked_mul(word_bytes)
        .ok_or(ResidentSourceStageError::SizeOverflow)?;
    let mut copies = usize::from(tail_words.is_some());
    if let Some(groups) = groups {
        for group in groups {
            copies = copies
                .checked_add(2)
                .ok_or(ResidentSourceStageError::SizeOverflow)?;
            bytes = bytes
                .checked_add(
                    group
                        .column_pointer_words
                        .checked_mul(word_bytes)
                        .ok_or(ResidentSourceStageError::SizeOverflow)?,
                )
                .and_then(|bytes| {
                    group
                        .column_log_size_words
                        .checked_mul(word_bytes)
                        .and_then(|next| bytes.checked_add(next))
                })
                .ok_or(ResidentSourceStageError::SizeOverflow)?;
            for batch in &group.batches {
                let (next_bytes, next_copies) = descriptor_batch_transfer(
                    bytes,
                    copies,
                    batch.coefficient_pointer_words,
                    batch.coefficient_size_words,
                    batch.output_pointer_words,
                    word_bytes,
                )?;
                bytes = next_bytes;
                copies = next_copies;
            }
        }
    }
    if let Some(batches) = progressive_batches {
        for batch in batches {
            let (next_bytes, next_copies) = descriptor_batch_transfer(
                bytes,
                copies,
                batch.coefficient_pointer_words,
                batch.coefficient_size_words,
                batch.output_pointer_words,
                word_bytes,
            )?;
            bytes = next_bytes;
            copies = next_copies;
        }
    }
    Ok((bytes, copies))
}

fn descriptor_batch_transfer(
    bytes: usize,
    copies: usize,
    coefficient_pointer_words: usize,
    coefficient_size_words: usize,
    output_pointer_words: usize,
    word_bytes: usize,
) -> Result<(usize, usize), ResidentSourceStageError> {
    let copies = copies
        .checked_add(3)
        .ok_or(ResidentSourceStageError::SizeOverflow)?;
    let bytes = [
        coefficient_pointer_words,
        coefficient_size_words,
        output_pointer_words,
    ]
    .into_iter()
    .try_fold(bytes, |total, words| {
        words
            .checked_mul(word_bytes)
            .and_then(|next| total.checked_add(next))
            .ok_or(ResidentSourceStageError::SizeOverflow)
    })?;
    Ok((bytes, copies))
}

/// Bind the planner's sealed same-log interpolation batches. Stage-wise mode
/// preserves the historical commitment-group partition; fused mode coalesces
/// equal logs in canonical encounter order. Both own immutable input/output
/// pointer tables, so capture never depends on host allocation.
pub(crate) fn commitment_interpolation_batches(
    workspace: &GraphWorkspace,
    tree: CommitmentTreeId,
) -> Result<Vec<InterpolationBatch>, ResidentSourceStageError> {
    let planned = workspace
        .plan()
        .commitment(tree)
        .ok_or(ResidentSourceStageError::MissingCommitment(tree))?;
    let expected_purpose = match tree {
        CommitmentTreeId::Base => BufferPurpose::BaseCoefficients,
        CommitmentTreeId::Interaction => BufferPurpose::InteractionCoefficients,
        _ => {
            return Err(ResidentSourceStageError::InterpolationBatchShapeMismatch(
                tree,
            ))
        }
    };
    planned
        .interpolation_batches
        .iter()
        .map(|batch| {
            let columns = batch
                .sources
                .iter()
                .map(|&source| {
                    if !matches!(
                        source,
                        CommitmentColumnSource::Trace { purpose, .. }
                            if purpose == expected_purpose
                    ) {
                        return Err(ResidentSourceStageError::InvalidInterpolationSource {
                            tree,
                            source,
                        });
                    }
                    let (evaluations, coefficients) = bind_trace_pair(workspace, tree, source)?;
                    Ok(InterpolationColumn {
                        evaluations,
                        coefficients,
                        log_size: batch.log_size,
                    })
                })
                .collect::<Result<Vec<_>, ResidentSourceStageError>>()?;
            Ok(InterpolationBatch {
                columns,
                input_pointers: batch.input_pointers,
                output_pointers: batch.output_pointers,
            })
        })
        .collect()
}

pub(crate) fn prepare_commitment_interpolation<'a>(
    workspace: &'a GraphWorkspace,
    tree: CommitmentTreeId,
) -> Result<PreparedInterpolationGraph<'a>, ResidentSourceStageError> {
    let batches = commitment_interpolation_batches(workspace, tree)?;
    let (inverse_twiddles, _) = bind_global_source(workspace, BufferPurpose::InverseTwiddles)?;
    Ok(PreparedInterpolationGraph::prepare(
        workspace.arena(),
        &batches,
        inverse_twiddles,
        workspace
            .plan()
            .commitment(tree)
            .ok_or(ResidentSourceStageError::MissingCommitment(tree))?
            .interpolation_mode,
    )?)
}

fn bind_trace_pair(
    workspace: &GraphWorkspace,
    tree: CommitmentTreeId,
    source: CommitmentColumnSource,
) -> Result<(ArenaSlice, ArenaSlice), ResidentSourceStageError> {
    let CommitmentColumnSource::Trace {
        component,
        part,
        purpose,
        ordinal,
    } = source
    else {
        return Err(ResidentSourceStageError::InvalidInterpolationSource { tree, source });
    };
    let (evaluation_purpose, coefficient_purpose) = match tree {
        CommitmentTreeId::Base => (BufferPurpose::BaseTrace, BufferPurpose::BaseCoefficients),
        CommitmentTreeId::Interaction => (
            BufferPurpose::InteractionTrace,
            BufferPurpose::InteractionCoefficients,
        ),
        _ => return Err(ResidentSourceStageError::InvalidInterpolationSource { tree, source }),
    };
    if purpose != coefficient_purpose {
        return Err(ResidentSourceStageError::InvalidInterpolationSource { tree, source });
    }
    let (evaluation, _) = workspace
        .plan()
        .find(Some(component), Some(part), evaluation_purpose, ordinal)
        .ok_or(ResidentSourceStageError::MissingArenaSource(source))?;
    let (coefficient, _) = workspace
        .plan()
        .find(Some(component), Some(part), coefficient_purpose, ordinal)
        .ok_or(ResidentSourceStageError::MissingArenaSource(source))?;
    let (evaluation, evaluation_words) = workspace.bind(evaluation.id)?;
    let (coefficient, coefficient_words) = workspace.bind(coefficient.id)?;
    if evaluation_words != coefficient_words {
        return Err(ResidentSourceStageError::ArenaSourceSizeMismatch {
            source,
            expected_words: evaluation_words,
            actual_words: coefficient_words,
        });
    }
    Ok((evaluation, coefficient))
}

fn bind_global_source(
    workspace: &GraphWorkspace,
    purpose: BufferPurpose,
) -> Result<(ArenaSlice, usize), ResidentSourceStageError> {
    let (logical, _) = workspace
        .plan()
        .find(None, None, purpose, 0)
        .ok_or(ResidentSourceStageError::MissingGlobalArenaSource(purpose))?;
    let (slice, logical_words) = workspace.bind(logical.id)?;
    Ok((slice, logical_words))
}

/// Select the exact inverse-twiddle tower for one protocol domain from a
/// possibly larger setup tree. A max-domain `TwiddleTree` is shared by the
/// cold setup lane, but FRI and quotient interpolation index their buffers
/// relative to their own domain. Passing the larger tree through unchanged
/// therefore selects different layers even when it has enough words.
fn extract_inverse_subdomain(
    source: &BaseFieldVec,
    purpose: BufferPurpose,
    domain_log_size: u32,
    subdomain_log_size: u32,
    expected_words: usize,
) -> Result<BaseFieldVec, ResidentSourceStageError> {
    validate_inverse_subdomain(
        source.size,
        purpose,
        domain_log_size,
        subdomain_log_size,
        expected_words,
    )?;
    Ok(
        <_ as TwiddleBuffer<BitReversedOrder>>::extract_subdomain_twiddles(
            source,
            domain_log_size,
            subdomain_log_size,
        ),
    )
}

fn validate_inverse_subdomain(
    source_words: usize,
    purpose: BufferPurpose,
    domain_log_size: u32,
    subdomain_log_size: u32,
    expected_words: usize,
) -> Result<(), ResidentSourceStageError> {
    let Some(domain_half_log_size) = domain_log_size.checked_sub(1) else {
        return Err(ResidentSourceStageError::InvalidTwiddleSubdomain {
            purpose,
            source_words,
            domain_log_size,
            subdomain_log_size,
            expected_words,
        });
    };
    let Some(subdomain_half_log_size) = subdomain_log_size.checked_sub(1) else {
        return Err(ResidentSourceStageError::InvalidTwiddleSubdomain {
            purpose,
            source_words,
            domain_log_size,
            subdomain_log_size,
            expected_words,
        });
    };
    let Some(actual_words) = 1usize.checked_shl(subdomain_half_log_size) else {
        return Err(ResidentSourceStageError::InvalidTwiddleSubdomain {
            purpose,
            source_words,
            domain_log_size,
            subdomain_log_size,
            expected_words,
        });
    };
    if !source_words.is_power_of_two()
        || subdomain_half_log_size > domain_half_log_size
        || domain_half_log_size > source_words.ilog2()
    {
        return Err(ResidentSourceStageError::InvalidTwiddleSubdomain {
            purpose,
            source_words,
            domain_log_size,
            subdomain_log_size,
            expected_words,
        });
    }
    if actual_words != expected_words {
        return Err(ResidentSourceStageError::TwiddleSourceSizeMismatch {
            purpose,
            expected_words,
            actual_words,
        });
    }
    Ok(())
}

fn checked_words(log_size: u32) -> Result<usize, ResidentSourceStageError> {
    1usize
        .checked_shl(log_size)
        .ok_or(ResidentSourceStageError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_words_rejects_host_width_overflow() {
        assert_eq!(checked_words(0).unwrap(), 1);
        assert_eq!(checked_words(17).unwrap(), 1 << 17);
        assert!(matches!(
            checked_words(usize::BITS),
            Err(ResidentSourceStageError::SizeOverflow)
        ));
    }

    #[test]
    fn coefficient_only_base_trace_is_rejected() {
        let result = require_base_evaluations(BaseTrace::Polys {
            polys: Vec::new(),
            tree_ptr: 0,
        });
        assert!(matches!(
            result,
            Err(ResidentSourceStageError::BaseEvaluationsUnavailable)
        ));
    }

    #[test]
    fn residency_counts_distinguish_direct_columns_from_migrations() {
        assert_eq!(
            residency_counts(7, 5),
            BaseTraceResidency {
                columns: 7,
                direct_columns: 5,
                migrated_columns: 2,
            }
        );
    }

    #[test]
    fn inverse_twiddle_subdomains_accept_a_larger_setup_tree() {
        assert!(validate_inverse_subdomain(
            1 << 25,
            BufferPurpose::InverseTwiddles,
            21,
            21,
            1 << 20,
        )
        .is_ok());
        assert!(validate_inverse_subdomain(
            1 << 25,
            BufferPurpose::QuotientInverseTwiddles,
            21,
            18,
            1 << 17,
        )
        .is_ok());
    }

    #[test]
    fn inverse_twiddle_subdomains_fail_closed_on_invalid_geometry() {
        for result in [
            validate_inverse_subdomain(1 << 19, BufferPurpose::InverseTwiddles, 21, 21, 1 << 20),
            validate_inverse_subdomain(1 << 25, BufferPurpose::InverseTwiddles, 21, 22, 1 << 21),
            validate_inverse_subdomain(
                (1 << 25) - 1,
                BufferPurpose::InverseTwiddles,
                21,
                21,
                1 << 20,
            ),
        ] {
            assert!(matches!(
                result,
                Err(ResidentSourceStageError::InvalidTwiddleSubdomain { .. })
            ));
        }
        assert!(matches!(
            validate_inverse_subdomain(1 << 25, BufferPurpose::InverseTwiddles, 21, 21, 1 << 19,),
            Err(ResidentSourceStageError::TwiddleSourceSizeMismatch {
                expected_words,
                actual_words,
                ..
            }) if expected_words == 1 << 19 && actual_words == 1 << 20
        ));
    }
}
