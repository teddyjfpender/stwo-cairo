//! Pure lowering from the generated semantic relation graph plus proof shape to
//! backend-CUDA descriptor batches. Device buffers are bound later by
//! `PreparedRelationGraph`; this pass is deterministic and CUDA-free.

use stwo_backend_cuda::{
    RelationBatchProgram, RelationColumnDescriptor, RelationGraphError, RelationGraphRequirements,
    RelationKernelProgram, RelationMultiplicityKind, RelationRowExtent, RelationSourceLayout,
    RelationTupleKind, RelationUseDescriptor,
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
    use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;

    use super::*;
    use crate::plan::ProofPlan;
    use crate::relation_table::CAIRO_RELATION_GRAPH;
    use crate::schedule_table::CAIRO_SCHEDULE;

    #[test]
    fn all_generated_uses_lower_once() {
        let shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let proof =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let execution =
            RelationExecutionPlan::from_proof_plan(&proof, &CAIRO_RELATION_GRAPH).unwrap();
        assert_eq!(execution.template_use_count, 1566);
        assert_eq!(execution.batches.len(), 68);
        assert_eq!(execution.relation_graph_hash, 0x7396_3831_c53d_f4a2);
        execution.requirements().unwrap();
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
}
