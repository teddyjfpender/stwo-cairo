//! Pure lowering from the generated semantic relation graph plus proof shape to
//! backend-CUDA descriptor batches. Device buffers are bound later by
//! `PreparedRelationGraph`; this pass is deterministic and CUDA-free.

use stwo_backend_cuda::{
    RelationBatchProgram, RelationColumnDescriptor, RelationGraphError, RelationGraphRequirements,
    RelationKernelProgram, RelationLaunchMode, RelationMultiplicityKind, RelationRowExtent,
    RelationSourceLayout, RelationTupleKind, RelationUseDescriptor,
};
use stwo_cairo_prover::witness::proof_shape::{RowResolution, TracePartId};

use crate::plan::ProofPlan;
use crate::relation::{
    ComponentRelationPlan, MultiplicitySign, MultiplicitySource, RelationGraph, RelationTracePart,
    RelationUse, TupleSource,
};
use crate::schedule::ComponentId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelationBatchKey {
    pub component: ComponentId,
    pub trace_part: RelationTracePart,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationSourcePlane {
    LookupWords,
    BaseTrace,
}

/// Canonical source columns for one lowered relation instance.  Arena liveness
/// and runtime binding consume this same plan so a retained evaluation cannot
/// silently drift from the kernel ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelationInstanceSourcePlan {
    pub batch: RelationBatchKey,
    pub instance_index: usize,
    pub part: TracePartId,
    pub plane: RelationSourcePlane,
    pub column_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationExecutionPlan {
    pub relation_graph_hash: u64,
    pub template_use_count: usize,
    pub batches: Vec<RelationBatchKey>,
    kernel_program: RelationKernelProgram,
}

impl RelationExecutionPlan {
    pub fn from_proof_plan(
        proof_plan: &ProofPlan,
        relation_graph: &'static RelationGraph,
    ) -> Result<Self, RelationExecutionError> {
        if proof_plan.relation_graph_hash != relation_graph.expected_hash {
            return Err(RelationExecutionError::RelationGraphHashMismatch {
                proof_plan: proof_plan.relation_graph_hash,
                graph: relation_graph.expected_hash,
            });
        }
        let mut batches = Vec::new();
        let mut kernel_batches = Vec::new();
        let mut template_use_count = 0usize;
        let mut max_alpha_powers = 0u32;
        for component in relation_graph.components {
            let runtime = proof_plan
                .components
                .iter()
                .find(|candidate| candidate.node.id == component.component)
                .ok_or(RelationExecutionError::MissingProofComponent(
                    component.component,
                ))?;
            for trace in component.traces {
                let source_layout = source_layout(component, trace.part, trace.columns)?;
                let columns: Vec<_> = trace
                    .columns
                    .iter()
                    .map(|column| {
                        let uses: Vec<_> = column
                            .uses
                            .iter()
                            .map(|relation_use| {
                                max_alpha_powers =
                                    max_alpha_powers.max(relation_use.denominator.tuple.words);
                                lower_use(source_layout, *relation_use)
                            })
                            .collect::<Result<_, _>>()?;
                        template_use_count = template_use_count
                            .checked_add(uses.len())
                            .ok_or(RelationExecutionError::SizeOverflow)?;
                        Ok(RelationColumnDescriptor { uses })
                    })
                    .collect::<Result<_, RelationExecutionError>>()?;
                let instances =
                    lower_row_extents(component.component, trace.part, &runtime.runtime.rows)?;
                batches.push(RelationBatchKey {
                    component: component.component,
                    trace_part: trace.part,
                });
                kernel_batches.push(RelationBatchProgram {
                    source_layout,
                    columns,
                    instances,
                });
            }
        }
        let expected_uses = relation_graph
            .components
            .iter()
            .flat_map(|component| component.traces)
            .flat_map(|trace| trace.columns)
            .map(|column| column.uses.len())
            .sum();
        if template_use_count != expected_uses {
            return Err(RelationExecutionError::UseCoverageMismatch {
                expected: expected_uses,
                actual: template_use_count,
            });
        }
        let kernel_program = RelationKernelProgram {
            relation_graph_hash: relation_graph.expected_hash,
            template_use_count,
            max_alpha_powers,
            batches: kernel_batches,
        };
        kernel_program
            .validate()
            .map_err(RelationExecutionError::BackendPlan)?;
        Ok(Self {
            relation_graph_hash: relation_graph.expected_hash,
            template_use_count,
            batches,
            kernel_program,
        })
    }

    pub fn kernel_program(&self) -> &RelationKernelProgram {
        &self.kernel_program
    }

    pub fn requirements(&self) -> Result<RelationGraphRequirements, RelationExecutionError> {
        self.kernel_program
            .requirements()
            .map_err(RelationExecutionError::BackendPlan)
    }

    pub fn requirements_for_mode(
        &self,
        mode: RelationLaunchMode,
    ) -> Result<RelationGraphRequirements, RelationExecutionError> {
        self.kernel_program
            .requirements_for_mode(mode)
            .map_err(RelationExecutionError::BackendPlan)
    }

    pub fn source_plan(&self) -> Result<Vec<RelationInstanceSourcePlan>, RelationExecutionError> {
        let mut sources = Vec::new();
        for (batch_index, kernel_batch) in self.kernel_program.batches.iter().enumerate() {
            let batch = *self
                .batches
                .get(batch_index)
                .ok_or(RelationExecutionError::SourcePlanDrift)?;
            let (plane, column_count) = match kernel_batch.source_layout {
                RelationSourceLayout::LookupWords { .. } => (RelationSourcePlane::LookupWords, 1),
                RelationSourceLayout::MemoryAddress { chunks } => (
                    RelationSourcePlane::BaseTrace,
                    chunks
                        .checked_mul(2)
                        .ok_or(RelationExecutionError::SizeOverflow)?,
                ),
                RelationSourceLayout::MemoryBig { value_words }
                | RelationSourceLayout::MemorySmall { value_words } => (
                    RelationSourcePlane::BaseTrace,
                    value_words
                        .checked_add(1)
                        .ok_or(RelationExecutionError::SizeOverflow)?,
                ),
                RelationSourceLayout::BitwiseXor12 {
                    multiplicity_columns,
                } => (RelationSourcePlane::BaseTrace, multiplicity_columns),
            };
            for instance_index in 0..kernel_batch.instances.len() {
                let part = match batch.trace_part {
                    RelationTracePart::Component => TracePartId::Main,
                    RelationTracePart::EachMemoryBig => TracePartId::MemoryBig(
                        u32::try_from(instance_index)
                            .map_err(|_| RelationExecutionError::SizeOverflow)?,
                    ),
                    RelationTracePart::MemorySmall => TracePartId::MemorySmall,
                };
                sources.push(RelationInstanceSourcePlan {
                    batch,
                    instance_index,
                    part,
                    plane,
                    column_count,
                });
            }
        }
        if sources.len() != self.requirements()?.instances.len() {
            return Err(RelationExecutionError::SourcePlanDrift);
        }
        Ok(sources)
    }
}

fn lower_row_extents(
    component: ComponentId,
    trace_part: RelationTracePart,
    rows: &RowResolution,
) -> Result<Vec<RelationRowExtent>, RelationExecutionError> {
    match rows {
        RowResolution::Absent => Ok(Vec::new()),
        RowResolution::Pending { .. } => Err(RelationExecutionError::PendingRows(component)),
        RowResolution::Bounded { bound, .. } => {
            if trace_part != RelationTracePart::Component {
                return Err(RelationExecutionError::BoundedSplitRows(component));
            }
            Ok(vec![RelationRowExtent::Bounded {
                observed_rows: to_u32(component, bound.observed_rows)?,
                max_rows: to_u32(component, bound.max_rows)?,
                padded_capacity: to_u32(component, bound.padded_capacity)?,
            }])
        }
        RowResolution::Resolved(parts) => match trace_part {
            RelationTracePart::Component => {
                let part = parts
                    .iter()
                    .find(|part| part.part == TracePartId::Main)
                    .ok_or(RelationExecutionError::MissingTracePart {
                        component,
                        part: trace_part,
                    })?;
                Ok(vec![exact_extent(component, *part, 0)?])
            }
            RelationTracePart::EachMemoryBig => {
                let mut big: Vec<_> = parts
                    .iter()
                    .filter_map(|part| match part.part {
                        TracePartId::MemoryBig(index) => Some((index, *part)),
                        _ => None,
                    })
                    .collect();
                big.sort_unstable_by_key(|(index, _)| *index);
                let mut offset = 0u32;
                let mut output = Vec::with_capacity(big.len());
                for (expected_index, (index, part)) in big.into_iter().enumerate() {
                    if index != expected_index as u32 {
                        return Err(RelationExecutionError::NonContiguousMemoryParts(component));
                    }
                    output.push(exact_extent(component, part, offset)?);
                    offset = offset
                        .checked_add(to_u32(component, part.padded_rows)?)
                        .ok_or(RelationExecutionError::SizeOverflow)?;
                }
                Ok(output)
            }
            RelationTracePart::MemorySmall => {
                let part = parts
                    .iter()
                    .find(|part| part.part == TracePartId::MemorySmall)
                    .ok_or(RelationExecutionError::MissingTracePart {
                        component,
                        part: trace_part,
                    })?;
                Ok(vec![exact_extent(component, *part, 0)?])
            }
        },
    }
}

fn exact_extent(
    component: ComponentId,
    part: stwo_cairo_prover::witness::proof_shape::TracePartShape,
    source_offset_rows: u32,
) -> Result<RelationRowExtent, RelationExecutionError> {
    Ok(RelationRowExtent::Exact {
        n_real_rows: to_u32(component, part.n_real_rows)?,
        padded_rows: to_u32(component, part.padded_rows)?,
        source_offset_rows,
    })
}

fn to_u32(component: ComponentId, value: u64) -> Result<u32, RelationExecutionError> {
    u32::try_from(value).map_err(|_| RelationExecutionError::RowsExceedU32 { component, value })
}

fn source_layout(
    component: &ComponentRelationPlan,
    part: RelationTracePart,
    columns: &[crate::relation::LogupColumnPlan],
) -> Result<RelationSourceLayout, RelationExecutionError> {
    let sources: Vec<_> = columns
        .iter()
        .flat_map(|column| column.uses)
        .map(|relation_use| relation_use.denominator.tuple.source)
        .collect();
    let first = *sources
        .first()
        .ok_or(RelationExecutionError::EmptyRelationTrace(
            component.component,
        ))?;
    match first {
        TupleSource::LookupWords { .. } => {
            if !sources
                .iter()
                .all(|source| matches!(source, TupleSource::LookupWords { .. }))
            {
                return Err(RelationExecutionError::MixedSourceLayout(
                    component.component,
                ));
            }
            Ok(RelationSourceLayout::LookupWords {
                words: component
                    .lookup_words
                    .ok_or(RelationExecutionError::MissingLookupWidth(
                        component.component,
                    ))?,
            })
        }
        TupleSource::MemoryAddressChunk { .. } => {
            let chunks = sources
                .iter()
                .map(|source| match source {
                    TupleSource::MemoryAddressChunk { chunk } => Ok(*chunk + 1),
                    _ => Err(RelationExecutionError::MixedSourceLayout(
                        component.component,
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap();
            Ok(RelationSourceLayout::MemoryAddress { chunks })
        }
        TupleSource::MemoryBigLimbs { .. } | TupleSource::MemoryBigValue => {
            if part != RelationTracePart::EachMemoryBig {
                return Err(RelationExecutionError::MixedSourceLayout(
                    component.component,
                ));
            }
            let value_words = columns
                .iter()
                .flat_map(|column| column.uses)
                .find(|relation_use| {
                    relation_use.denominator.tuple.source == TupleSource::MemoryBigValue
                })
                .and_then(|relation_use| relation_use.denominator.tuple.words.checked_sub(2))
                .ok_or(RelationExecutionError::MissingMemoryValueTuple(
                    component.component,
                ))?;
            Ok(RelationSourceLayout::MemoryBig { value_words })
        }
        TupleSource::MemorySmallLimbs { .. } | TupleSource::MemorySmallValue => {
            if part != RelationTracePart::MemorySmall {
                return Err(RelationExecutionError::MixedSourceLayout(
                    component.component,
                ));
            }
            let value_words = columns
                .iter()
                .flat_map(|column| column.uses)
                .find(|relation_use| {
                    relation_use.denominator.tuple.source == TupleSource::MemorySmallValue
                })
                .and_then(|relation_use| relation_use.denominator.tuple.words.checked_sub(2))
                .ok_or(RelationExecutionError::MissingMemoryValueTuple(
                    component.component,
                ))?;
            Ok(RelationSourceLayout::MemorySmall { value_words })
        }
        TupleSource::BitwiseXor12 { .. } => {
            let multiplicity_columns = sources
                .iter()
                .map(|source| match source {
                    TupleSource::BitwiseXor12 {
                        multiplicity_column,
                    } => Ok(*multiplicity_column + 1),
                    _ => Err(RelationExecutionError::MixedSourceLayout(
                        component.component,
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap();
            Ok(RelationSourceLayout::BitwiseXor12 {
                multiplicity_columns,
            })
        }
    }
}

fn lower_use(
    layout: RelationSourceLayout,
    relation_use: RelationUse,
) -> Result<RelationUseDescriptor, RelationExecutionError> {
    let (tuple_kind, tuple_arg) = match relation_use.denominator.tuple.source {
        TupleSource::LookupWords { word_offset } => (RelationTupleKind::LookupWords, word_offset),
        TupleSource::MemoryAddressChunk { chunk } => (RelationTupleKind::MemoryAddressChunk, chunk),
        TupleSource::MemoryBigLimbs { first_limb } => {
            (RelationTupleKind::MemoryBigLimbs, first_limb)
        }
        TupleSource::MemoryBigValue => (RelationTupleKind::MemoryBigValue, 0),
        TupleSource::MemorySmallLimbs { first_limb } => {
            (RelationTupleKind::MemorySmallLimbs, first_limb)
        }
        TupleSource::MemorySmallValue => (RelationTupleKind::MemorySmallValue, 0),
        TupleSource::BitwiseXor12 {
            multiplicity_column,
        } => (RelationTupleKind::BitwiseXor12, multiplicity_column),
    };
    let (multiplicity_kind, multiplicity_arg) = match relation_use.multiplicity.source {
        MultiplicitySource::One => (RelationMultiplicityKind::One, 0),
        MultiplicitySource::Enabler => (RelationMultiplicityKind::Enabler, 0),
        MultiplicitySource::LookupWord { word_offset } => {
            (RelationMultiplicityKind::LookupWord, word_offset)
        }
        MultiplicitySource::MemoryAddressChunk { chunk } => {
            (RelationMultiplicityKind::MemoryAddressChunk, chunk)
        }
        MultiplicitySource::MemoryBig => (
            RelationMultiplicityKind::MemoryBig,
            match layout {
                RelationSourceLayout::MemoryBig { value_words } => value_words,
                _ => return Err(RelationExecutionError::MultiplicityLayoutMismatch),
            },
        ),
        MultiplicitySource::MemorySmall => (
            RelationMultiplicityKind::MemorySmall,
            match layout {
                RelationSourceLayout::MemorySmall { value_words } => value_words,
                _ => return Err(RelationExecutionError::MultiplicityLayoutMismatch),
            },
        ),
        MultiplicitySource::BitwiseXor12 {
            multiplicity_column,
        } => (RelationMultiplicityKind::BitwiseXor12, multiplicity_column),
    };
    Ok(RelationUseDescriptor {
        tuple_kind,
        tuple_arg,
        tuple_words: relation_use.denominator.tuple.words,
        relation_id: relation_use.relation.id,
        multiplicity_kind,
        multiplicity_arg,
        negative: relation_use.multiplicity.sign == MultiplicitySign::Negative,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelationExecutionError {
    RelationGraphHashMismatch {
        proof_plan: u64,
        graph: u64,
    },
    MissingProofComponent(ComponentId),
    PendingRows(ComponentId),
    BoundedSplitRows(ComponentId),
    MissingTracePart {
        component: ComponentId,
        part: RelationTracePart,
    },
    NonContiguousMemoryParts(ComponentId),
    RowsExceedU32 {
        component: ComponentId,
        value: u64,
    },
    EmptyRelationTrace(ComponentId),
    MixedSourceLayout(ComponentId),
    MissingLookupWidth(ComponentId),
    MissingMemoryValueTuple(ComponentId),
    MultiplicityLayoutMismatch,
    UseCoverageMismatch {
        expected: usize,
        actual: usize,
    },
    SourcePlanDrift,
    SizeOverflow,
    BackendPlan(RelationGraphError),
}

impl core::fmt::Display for RelationExecutionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RelationExecutionError {}

#[cfg(test)]
mod tests {
    use stwo_backend_cuda::{
        relation_batch_fused_eligible, relation_batch_one_read_eligible, RELATION_FUSED_MAX_COLUMNS,
    };
    use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
    use stwo_cairo_prover::witness::proof_shape::{
        ProofShape, RuntimeComponentShape, TracePartShape,
    };

    use super::*;
    use crate::plan::ProofPlan;
    use crate::relation_table::CAIRO_RELATION_GRAPH;
    use crate::schedule_table::CAIRO_SCHEDULE;

    const LEGACY_NARROW_MAX_TUPLE_WORDS: u32 = 32;
    const WORD_BYTES: u64 = core::mem::size_of::<u32>() as u64;
    const QM31_BYTES: u64 = 16;

    // Derived from the sealed SN3 adapted input (285,299,888 bytes, SHA-256
    // cd823275e92b4b224251565791ac0aa794a04377f43fd1198798a0d0c1454bed)
    // by `real_sn3_capacity_fixture_rederives_relation_accounting` below. These are
    // planned capacity extents, matching resident-arena preflight accounting.
    const SN3_NARROW_INSTANCES: usize = 48;
    const SN3_NARROW_FRACTIONS: u64 = 274_250_768;
    const SN3_NARROW_DENOMINATOR_BYTES: u64 = 4_388_012_288;
    const SN3_NEWLY_WIDE_INSTANCES: usize = 10;
    const SN3_NEWLY_WIDE_FRACTIONS: u64 = 264_177_664;
    const SN3_NEWLY_WIDE_DENOMINATOR_BYTES: u64 = 4_226_842_624;

    fn legacy_narrow_fused_eligible(batch: &RelationBatchProgram) -> bool {
        batch.columns.len() <= RELATION_FUSED_MAX_COLUMNS
            && batch.columns.iter().all(|column| {
                column
                    .uses
                    .iter()
                    .all(|relation_use| relation_use.tuple_words <= LEGACY_NARROW_MAX_TUPLE_WORDS)
            })
    }

    #[derive(Debug, Default, Eq, PartialEq)]
    struct RelationAccounting {
        narrow_instances: usize,
        narrow_fractions: u64,
        narrow_denominator_bytes: u64,
        newly_wide_instances: usize,
        newly_wide_fractions: u64,
        newly_wide_denominator_bytes: u64,
    }

    fn relation_accounting(
        execution: &RelationExecutionPlan,
        full: &RelationGraphRequirements,
    ) -> RelationAccounting {
        let mut accounting = RelationAccounting::default();
        for requirement in &full.instances {
            assert_eq!(requirement.denominator_words, requirement.output_words);
            let denominator_bytes =
                u64::try_from(requirement.denominator_words).unwrap() * WORD_BYTES;
            assert_eq!(denominator_bytes % QM31_BYTES, 0);
            let fractions = denominator_bytes / QM31_BYTES;
            let batch = &execution.kernel_program.batches[requirement.batch_index];
            if legacy_narrow_fused_eligible(batch) {
                accounting.narrow_instances += 1;
                accounting.narrow_fractions += fractions;
                accounting.narrow_denominator_bytes += denominator_bytes;
            } else {
                assert!(
                    relation_batch_fused_eligible(batch),
                    "SN3 batch {} is neither legacy-narrow nor newly-wide eligible",
                    requirement.batch_index
                );
                accounting.newly_wide_instances += 1;
                accounting.newly_wide_fractions += fractions;
                accounting.newly_wide_denominator_bytes += denominator_bytes;
            }
        }
        accounting
    }

    #[test]
    fn all_generated_uses_lower_once() {
        let shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let proof =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let execution =
            RelationExecutionPlan::from_proof_plan(&proof, &CAIRO_RELATION_GRAPH).unwrap();
        assert_eq!(execution.template_use_count, 1566);
        assert_eq!(execution.batches.len(), 68);
        assert_eq!(
            execution
                .kernel_program
                .batches
                .iter()
                .map(|batch| batch.columns.len())
                .sum::<usize>(),
            807
        );
        assert_eq!(
            execution
                .kernel_program
                .batches
                .iter()
                .map(|batch| batch.columns.len())
                .max(),
            Some(157),
            "generated relation shape outgrew the 512-fraction one-read tile"
        );
        assert_eq!(execution.relation_graph_hash, 0x7396_3831_c53d_f4a2);
        let newly_wide_batches = execution
            .kernel_program
            .batches
            .iter()
            .filter(|batch| !legacy_narrow_fused_eligible(batch))
            .count();
        assert!(
            newly_wide_batches > 0,
            "wide production coverage is vacuous"
        );
        assert!(execution
            .kernel_program
            .batches
            .iter()
            .all(relation_batch_fused_eligible));
        assert!(execution
            .kernel_program
            .batches
            .iter()
            .all(relation_batch_one_read_eligible));
        execution.requirements().unwrap();
        let sources = execution.source_plan().unwrap();
        let requirements = execution.requirements().unwrap();
        assert_eq!(sources.len(), requirements.instances.len());
        for (source, requirement) in sources.iter().zip(&requirements.instances) {
            assert_eq!(source.batch, execution.batches[requirement.batch_index]);
            assert_eq!(source.instance_index, requirement.instance_index);
        }
    }

    #[test]
    fn real_sn3_relation_accounting_baselines_are_not_conflated() {
        let total_instances = SN3_NARROW_INSTANCES + SN3_NEWLY_WIDE_INSTANCES;
        let total_fractions = SN3_NARROW_FRACTIONS + SN3_NEWLY_WIDE_FRACTIONS;
        let full_legacy_denominator_bytes =
            SN3_NARROW_DENOMINATOR_BYTES + SN3_NEWLY_WIDE_DENOMINATOR_BYTES;
        assert_eq!(total_instances, 58);
        assert_eq!(total_fractions, 538_428_432);
        assert_eq!(full_legacy_denominator_bytes, 8_614_854_912);
        assert_eq!(
            SN3_NARROW_FRACTIONS * QM31_BYTES,
            SN3_NARROW_DENOMINATOR_BYTES
        );
        assert_eq!(
            SN3_NEWLY_WIDE_FRACTIONS * QM31_BYTES,
            SN3_NEWLY_WIDE_DENOMINATOR_BYTES
        );

        // The CURRENT <=32-word fused baseline already holds one four-byte
        // sentinel for each narrow instance and a full slab only for the ten
        // wide instances. The new adaptive lane leaves one sentinel for all
        // 58. This is logical arena accounting, not aligned physical slots.
        let current_baseline_denominator_bytes =
            SN3_NARROW_INSTANCES as u64 * WORD_BYTES + SN3_NEWLY_WIDE_DENOMINATOR_BYTES;
        let new_denominator_bytes = total_instances as u64 * WORD_BYTES;
        assert_eq!(current_baseline_denominator_bytes, 4_226_842_816);
        assert_eq!(new_denominator_bytes, 232);
        assert_eq!(
            current_baseline_denominator_bytes - new_denominator_bytes,
            4_226_842_584,
            "incremental arena retirement versus the current narrow-fused baseline"
        );
        assert_eq!(
            full_legacy_denominator_bytes - new_denominator_bytes,
            8_614_854_680,
            "arena retirement versus a full three-stage legacy baseline"
        );

        // Post-source-evaluation logical pass bytes per fraction: three-stage
        // writes pairs (32), reads+writes inverse (32), then reads fractions
        // (32) and writes output (16) = 112. The suffix/recompute lane stages,
        // rereads and overwrites output = 48. The one-read lane writes only the
        // final output = 16. Tuple-source/descriptor reads are deliberately
        // excluded; this is an exact logical pass model, not measured DRAM.
        const THREE_STAGE_BYTES: u64 = 112;
        const NARROW_FUSED_BYTES: u64 = 48;
        const ONE_READ_BYTES: u64 = 16;
        let current_baseline_body_bytes = SN3_NARROW_FRACTIONS * NARROW_FUSED_BYTES
            + SN3_NEWLY_WIDE_FRACTIONS * THREE_STAGE_BYTES;
        let new_body_bytes =
            SN3_NARROW_FRACTIONS * NARROW_FUSED_BYTES + SN3_NEWLY_WIDE_FRACTIONS * ONE_READ_BYTES;
        let full_legacy_body_bytes = total_fractions * THREE_STAGE_BYTES;
        assert_eq!(current_baseline_body_bytes, 42_751_935_232);
        assert_eq!(new_body_bytes, 17_390_879_488);
        assert_eq!(
            current_baseline_body_bytes - new_body_bytes,
            25_361_055_744,
            "incremental pass bytes retired by the new wide lane"
        );
        assert_eq!(full_legacy_body_bytes, 60_303_984_384);
        assert_eq!(
            full_legacy_body_bytes - new_body_bytes,
            42_913_104_896,
            "pass bytes retired versus an all-three-stage legacy implementation"
        );

        // Every generated relation batch has at most 157 columns, so the
        // shape-routed source sends all 538,428,432 SN3 fractions through the
        // existing one-read tile. This is the exact additional logical-body
        // reduction relative to the preceding adaptive-wide implementation.
        let all_one_read_body_bytes = total_fractions * ONE_READ_BYTES;
        let additional_retired_body_bytes = new_body_bytes - all_one_read_body_bytes;
        assert_eq!(all_one_read_body_bytes, 8_614_854_912);
        assert_eq!(additional_retired_body_bytes, 8_776_024_576);
        assert_eq!(
            additional_retired_body_bytes * 1_000_000 / new_body_bytes,
            504_633,
            "additional one-read logical-body reduction in integer ppm"
        );
    }

    #[test]
    #[ignore = "requires STWO_SN3_INPUT pointing to the sealed 285 MB adapted fixture"]
    fn real_sn3_capacity_fixture_rederives_relation_accounting() {
        let input_path = std::env::var("STWO_SN3_INPUT").unwrap();
        let input_bytes = std::fs::read(input_path).unwrap();
        assert_eq!(input_bytes.len(), 285_299_888);
        let input: stwo_cairo_adapter::ProverInput = bincode::deserialize(&input_bytes).unwrap();
        let ingest = crate::phases::ingest::run(
            input,
            stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant::Canonical,
            None,
        );
        let proof = ingest.proof_plan;
        let execution =
            RelationExecutionPlan::from_proof_plan(&proof, &CAIRO_RELATION_GRAPH).unwrap();
        let ineligible = execution
            .kernel_program
            .batches
            .iter()
            .enumerate()
            .filter_map(|(index, batch)| {
                (!relation_batch_fused_eligible(batch)).then_some(execution.batches[index])
            })
            .collect::<Vec<_>>();
        assert!(
            ineligible.is_empty(),
            "generated production relation batches escaped fused coverage: {ineligible:?}"
        );
        let not_one_read = execution
            .kernel_program
            .batches
            .iter()
            .enumerate()
            .filter_map(|(index, batch)| {
                (!relation_batch_one_read_eligible(batch)).then_some(execution.batches[index])
            })
            .collect::<Vec<_>>();
        assert!(
            not_one_read.is_empty(),
            "generated SN3 relation batches escaped one-read coverage: {not_one_read:?}"
        );

        let full = execution
            .requirements_for_mode(RelationLaunchMode::ThreeStage)
            .unwrap();
        let compact = execution
            .requirements_for_mode(RelationLaunchMode::Fused)
            .unwrap();
        assert_eq!(full.instances.len(), compact.instances.len());
        assert_eq!(full.instances.len(), 58, "real SN3 coverage is non-vacuous");
        assert!(compact
            .instances
            .iter()
            .all(|instance| instance.denominator_words == 1));
        assert_eq!(full.instances.len() * 16, 928, "claimed-sum geometry");
        assert_eq!(full.instances.len() * 44, 2_552, "launch geometry");
        assert_eq!(
            relation_accounting(&execution, &full),
            RelationAccounting {
                narrow_instances: SN3_NARROW_INSTANCES,
                narrow_fractions: SN3_NARROW_FRACTIONS,
                narrow_denominator_bytes: SN3_NARROW_DENOMINATOR_BYTES,
                newly_wide_instances: SN3_NEWLY_WIDE_INSTANCES,
                newly_wide_fractions: SN3_NEWLY_WIDE_FRACTIONS,
                newly_wide_denominator_bytes: SN3_NEWLY_WIDE_DENOMINATOR_BYTES,
            },
        );
    }

    #[test]
    fn special_sources_lower_to_descriptor_layouts() {
        let shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let proof =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let execution =
            RelationExecutionPlan::from_proof_plan(&proof, &CAIRO_RELATION_GRAPH).unwrap();
        let key_index = |component, trace_part| {
            execution
                .batches
                .iter()
                .position(|key| key.component == component && key.trace_part == trace_part)
                .unwrap()
        };
        assert_eq!(
            execution.kernel_program.batches
                [key_index("memory_address_to_id", RelationTracePart::Component)]
            .source_layout,
            RelationSourceLayout::MemoryAddress { chunks: 16 }
        );
        assert_eq!(
            execution.kernel_program.batches
                [key_index("verify_bitwise_xor_12", RelationTracePart::Component)]
            .source_layout,
            RelationSourceLayout::BitwiseXor12 {
                multiplicity_columns: 16
            }
        );
    }

    #[test]
    fn source_plan_pins_every_source_layout_to_its_trace_plane() {
        let default_shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let mut components = default_shape.components().to_vec();
        let mut set = |shape: RuntimeComponentShape| {
            let destination = components
                .iter_mut()
                .find(|component| component.id == shape.id)
                .unwrap();
            *destination = shape;
        };
        set(RuntimeComponentShape::uniform("add_ap_opcode", 5, 16).unwrap());
        set(RuntimeComponentShape::uniform("memory_address_to_id", 5, 16).unwrap());
        set(RuntimeComponentShape::parts(
            "memory_id_to_big",
            vec![
                TracePartShape {
                    part: TracePartId::MemoryBig(0),
                    n_real_rows: 33,
                    padded_rows: 64,
                },
                TracePartShape {
                    part: TracePartId::MemoryBig(1),
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
        .unwrap());
        let xor12_rows = 1u64 << 20;
        set(
            RuntimeComponentShape::uniform("verify_bitwise_xor_12", xor12_rows, xor12_rows)
                .unwrap(),
        );
        let shape = ProofShape::new(components).unwrap();
        let proof =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let execution =
            RelationExecutionPlan::from_proof_plan(&proof, &CAIRO_RELATION_GRAPH).unwrap();
        let sources = execution.source_plan().unwrap();
        let find = |component, trace_part| {
            sources
                .iter()
                .filter(|source| {
                    source.batch.component == component && source.batch.trace_part == trace_part
                })
                .collect::<Vec<_>>()
        };

        let lookup = find("add_ap_opcode", RelationTracePart::Component);
        assert_eq!(lookup.len(), 1);
        assert_eq!(lookup[0].part, TracePartId::Main);
        assert_eq!(lookup[0].plane, RelationSourcePlane::LookupWords);
        assert_eq!(lookup[0].column_count, 1);

        let address = find("memory_address_to_id", RelationTracePart::Component);
        assert_eq!(address.len(), 1);
        assert_eq!(address[0].part, TracePartId::Main);
        assert_eq!(address[0].plane, RelationSourcePlane::BaseTrace);
        assert_eq!(address[0].column_count, 32);

        let big = find("memory_id_to_big", RelationTracePart::EachMemoryBig);
        assert!(!big.is_empty());
        for (index, source) in big.into_iter().enumerate() {
            assert_eq!(source.instance_index, index);
            assert_eq!(source.part, TracePartId::MemoryBig(index as u32));
            assert_eq!(source.plane, RelationSourcePlane::BaseTrace);
            assert_eq!(source.column_count, 29);
        }

        let small = find("memory_id_to_big", RelationTracePart::MemorySmall);
        assert_eq!(small.len(), 1);
        assert_eq!(small[0].part, TracePartId::MemorySmall);
        assert_eq!(small[0].plane, RelationSourcePlane::BaseTrace);
        assert_eq!(small[0].column_count, 9);

        let xor12 = find("verify_bitwise_xor_12", RelationTracePart::Component);
        assert_eq!(xor12.len(), 1);
        assert_eq!(xor12[0].part, TracePartId::Main);
        assert_eq!(xor12[0].plane, RelationSourcePlane::BaseTrace);
        assert_eq!(xor12[0].column_count, 16);
    }
}
