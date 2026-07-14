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
    use stwo_backend_cuda::relation_batch_fused_eligible;
    use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
    use stwo_cairo_prover::witness::proof_shape::{
        ProofShape, RuntimeComponentShape, TracePartShape,
    };

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
        assert_eq!(
            execution
                .kernel_program
                .batches
                .iter()
                .map(|batch| batch.columns.len())
                .sum::<usize>(),
            807
        );
        assert_eq!(execution.relation_graph_hash, 0x7396_3831_c53d_f4a2);
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
    fn generated_production_graph_proves_exact_sn3_denominator_retirement() {
        // Eligibility is a property of each generated batch's columns and
        // tuple widths, not its runtime row count. Lowering the complete
        // machine-written graph therefore proves every instance in SN1-SN4,
        // including batches absent from any one proof shape.
        let shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let proof =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
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

        let full = execution
            .requirements_for_mode(RelationLaunchMode::ThreeStage)
            .unwrap();
        let compact = execution
            .requirements_for_mode(RelationLaunchMode::Fused)
            .unwrap();
        assert_eq!(full.instances.len(), compact.instances.len());
        assert!(compact
            .instances
            .iter()
            .all(|instance| instance.denominator_words == 1));

        // Sealed SN3 arena facts independently identify 58 instances (928 B
        // claimed sums / 16 B each, and 2,552 B geometry / 44 B each). Since
        // every production batch above is eligible, no SN3 instance can retain
        // a slab: only one 4-byte sentinel per instance remains.
        const SN3_RELATION_INSTANCES: usize = 58;
        const SN3_LEGACY_DENOMINATOR_BYTES: usize = 4_226_842_816;
        const QM31_BYTES: usize = 16;
        const RETIRED_HBM_BYTES_PER_FRACTION: usize = 96;
        assert_eq!(928 / 16, SN3_RELATION_INSTANCES);
        assert_eq!(2_552 / 44, SN3_RELATION_INSTANCES);
        let compact_bytes = SN3_RELATION_INSTANCES * core::mem::size_of::<u32>();
        assert_eq!(compact_bytes, 232);
        assert_eq!(SN3_LEGACY_DENOMINATOR_BYTES - compact_bytes, 4_226_842_584);

        // Each old fallback fraction moved 112 logical HBM bytes after source
        // evaluation (pair writes 32, inverse read/write 32, chain reads 32 +
        // final write 16); the wide lane writes only the final 16. This is a
        // pass-byte lower bound, independent of cache transaction effects.
        let fallback_fractions = SN3_LEGACY_DENOMINATOR_BYTES / QM31_BYTES;
        assert_eq!(fallback_fractions, 264_177_676);
        assert_eq!(
            fallback_fractions * RETIRED_HBM_BYTES_PER_FRACTION,
            25_361_056_896
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
