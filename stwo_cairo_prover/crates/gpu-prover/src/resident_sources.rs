//! Checked migration hand-off from witness-owned CUDA columns to arena slots.
//!
//! The end-state witness DAG writes these slots directly.  Until every generated
//! writer accepts an arena destination, this module performs exactly one D2D
//! copy per base coefficient column.  The copy is identity driven: raw vector
//! position is first validated against the generated claim-order layout and is
//! then resolved to `(component, trace part, purpose, ordinal)` in the arena.

use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_backend_cuda::{
    synchronize_legacy_stream_for_arena_handoff, ArenaSlice, CudaBackend, CudaRuntimeError,
};
use stwo_cairo_prover::witness::base_trace::BaseTrace;
use stwo_cairo_prover::witness::proof_shape::ProofShapeKey;
use stwo_cairo_prover::witness::relation_sources::{
    CairoRelationSourceSet, DeviceRelationWord, RelationLookupTransfer, RelationSourceEncoding,
    RelationSourceId,
};

use crate::arena_plan::{BufferPurpose, CommitmentColumnSource};
use crate::graphs::{GraphError, GraphWorkspace};
use crate::plan::ProofPlan;
use crate::protocol_plan::{trace_commitment_layout, ProtocolPlanError, TraceCommitmentColumn};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentSourceStageReport {
    pub base_columns: usize,
    pub base_words: usize,
    pub twiddle_words: usize,
    pub d2d_words: usize,
    pub d2d_bytes: usize,
    /// Remains true until every base witness producer is arena-native.
    pub used_migration_copy: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentLookupStageReport {
    pub sources: usize,
    pub host_words: usize,
    pub device_words: usize,
    pub filled_words: usize,
}

#[derive(Debug)]
pub enum ResidentSourceStageError {
    WorkspaceShapeMismatch {
        expected: ProofShapeKey,
        actual: ProofShapeKey,
    },
    Protocol(ProtocolPlanError),
    InvalidBaseSource(CommitmentColumnSource),
    ColumnCountMismatch {
        expected: usize,
        actual: usize,
    },
    ColumnLogMismatch {
        column: usize,
        expected: u32,
        actual: u32,
    },
    TwiddleIdentityMismatch {
        expected: usize,
        actual: usize,
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

/// Consume the base trace and populate every arena-owned base coefficient slot.
///
/// All metadata is validated before the first copy is issued.  One legacy-stream
/// fence precedes the batch and one isolated-context fence follows it, so source
/// allocations may be dropped immediately on return without racing CUDA work.
pub fn stage_base_trace_coefficients(
    workspace: &GraphWorkspace,
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

    let columns = trace_commitment_layout(proof_plan)?.base;
    let polys = into_coefficients(trace, twiddles)?;
    validate_coefficients(&columns, &polys)?;

    let mut copies = Vec::with_capacity(columns.len() + 2);
    let mut base_words = 0usize;
    for (column, poly) in columns.iter().copied().zip(&polys) {
        let (logical, destination) = bind_base_source(workspace, column.source)?;
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
        copies.push((destination, poly.coeffs.device_ptr, expected_words));
    }

    let mut twiddle_words = 0usize;
    for (purpose, source, source_words) in [
        (
            BufferPurpose::ForwardTwiddles,
            twiddles.twiddles.device_ptr,
            twiddles.twiddles.size,
        ),
        (
            BufferPurpose::InverseTwiddles,
            twiddles.itwiddles.device_ptr,
            twiddles.itwiddles.size,
        ),
    ] {
        let (destination, expected_words) = bind_global_source(workspace, purpose)?;
        if source_words != expected_words {
            return Err(ResidentSourceStageError::TwiddleSourceSizeMismatch {
                purpose,
                expected_words,
                actual_words: source_words,
            });
        }
        twiddle_words = twiddle_words
            .checked_add(expected_words)
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        copies.push((destination, source, expected_words));
    }

    let d2d_words = base_words
        .checked_add(twiddle_words)
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
    workspace.arena().context().sync()?;

    Ok(ResidentSourceStageReport {
        base_columns: columns.len(),
        base_words,
        twiddle_words,
        d2d_words,
        d2d_bytes: d2d_words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ResidentSourceStageError::SizeOverflow)?,
        used_migration_copy: !columns.is_empty(),
    })
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
            }
            RelationLookupTransfer::DeviceWordMajor(words) => {
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
                        }
                    }
                }
            }
        }
        report.sources += 1;
    }
    workspace.arena().context().sync()?;
    Ok(report)
}

fn into_coefficients(
    trace: BaseTrace<CudaBackend>,
    twiddles: &'static TwiddleTree<CudaBackend>,
) -> Result<Vec<CircleCoefficients<CudaBackend>>, ResidentSourceStageError> {
    match trace {
        BaseTrace::Evals(evals) => Ok(CudaBackend::interpolate_columns(evals, twiddles)),
        BaseTrace::Polys { polys, tree_ptr } => {
            let expected = twiddles as *const TwiddleTree<CudaBackend> as usize;
            if tree_ptr != expected {
                return Err(ResidentSourceStageError::TwiddleIdentityMismatch {
                    expected,
                    actual: tree_ptr,
                });
            }
            Ok(polys)
        }
    }
}

fn validate_coefficients(
    columns: &[TraceCommitmentColumn],
    polys: &[CircleCoefficients<CudaBackend>],
) -> Result<(), ResidentSourceStageError> {
    if columns.len() != polys.len() {
        return Err(ResidentSourceStageError::ColumnCountMismatch {
            expected: columns.len(),
            actual: polys.len(),
        });
    }
    for (column, (expected, poly)) in columns.iter().zip(polys).enumerate() {
        if expected.log_size != poly.log_size() {
            return Err(ResidentSourceStageError::ColumnLogMismatch {
                column,
                expected: expected.log_size,
                actual: poly.log_size(),
            });
        }
    }
    Ok(())
}

fn bind_base_source(
    workspace: &GraphWorkspace,
    source: CommitmentColumnSource,
) -> Result<(usize, ArenaSlice), ResidentSourceStageError> {
    let CommitmentColumnSource::Trace {
        component,
        part,
        purpose: BufferPurpose::BaseTrace,
        ordinal,
    } = source
    else {
        return Err(ResidentSourceStageError::InvalidBaseSource(source));
    };
    let (logical, _) = workspace
        .plan()
        .find(
            Some(component),
            Some(part),
            BufferPurpose::BaseTrace,
            ordinal,
        )
        .ok_or(ResidentSourceStageError::MissingArenaSource(source))?;
    let (slice, logical_words) = workspace.bind(logical.id)?;
    Ok((logical_words, slice))
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
}
