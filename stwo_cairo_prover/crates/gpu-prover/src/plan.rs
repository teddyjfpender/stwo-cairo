//! Pure proof-plan validation: generated static component facts joined with one
//! pre-witness runtime [`ProofShape`]. No CUDA allocation or PCS policy lives here.

use std::collections::BTreeMap;

use stwo::prover::backend::simd::m31::N_LANES;
use stwo_cairo_prover::witness::proof_shape::{
    CapacityBound, PendingRowsReason, ProofShape, ProofShapeError, ProofShapeKey, RowResolution,
    RuntimeComponentShape, TracePartId,
};

use crate::relation::{RelationGraph, RelationPlanError};
use crate::schedule::{
    ComponentId, ComponentNode, ComponentRowSource, Schedule, ScheduleError, TraceColumnCount,
};

#[derive(Clone, Debug)]
pub struct ComponentPlan {
    pub node: &'static ComponentNode,
    pub runtime: RuntimeComponentShape,
}

#[derive(Clone, Debug)]
pub struct ProofPlan {
    pub components: Vec<ComponentPlan>,
    pub shape_key: ProofShapeKey,
    pub relation_graph_hash: u64,
    proof_shape: ProofShape,
}

impl ProofPlan {
    pub fn from_schedule(
        schedule: &'static Schedule,
        relation_graph: &'static RelationGraph,
        shape: &ProofShape,
    ) -> Result<Self, ProofPlanError> {
        schedule.validate().map_err(ProofPlanError::Schedule)?;
        let relation_plan = relation_graph
            .plan(schedule)
            .map_err(ProofPlanError::Relations)?;

        let mut runtime: BTreeMap<ComponentId, RuntimeComponentShape> = BTreeMap::new();
        for component in shape.components() {
            runtime.insert(component.id, component.clone());
        }

        for node in schedule.nodes {
            let shape = runtime
                .get(node.id)
                .ok_or(ProofPlanError::MissingRuntimeComponent(node.id))?;
            validate_row_source(node, shape)?;
            validate_output_bounds(node)?;
        }
        if let Some((&id, _)) = runtime
            .iter()
            .find(|(id, _)| !schedule.nodes.iter().any(|node| node.id == **id))
        {
            return Err(ProofPlanError::UnknownRuntimeComponent(id));
        }

        resolve_capacity_bounds(schedule, &mut runtime)?;
        let proof_shape =
            ProofShape::new(runtime.values().cloned().collect()).map_err(ProofPlanError::Shape)?;
        let mut components = Vec::with_capacity(schedule.nodes.len());
        for node in schedule.nodes {
            components.push(ComponentPlan {
                node,
                runtime: proof_shape
                    .component(node.id)
                    .expect("proof shape completeness checked")
                    .clone(),
            });
        }

        Ok(Self {
            components,
            shape_key: proof_shape.key(),
            relation_graph_hash: relation_plan.relation_graph_hash(),
            proof_shape,
        })
    }

    pub fn proof_shape(&self) -> &ProofShape {
        &self.proof_shape
    }

    pub fn into_proof_shape(self) -> ProofShape {
        self.proof_shape
    }

    pub fn arena_capacity_ready(&self) -> bool {
        self.proof_shape.require_arena_ready().is_ok()
    }

    pub fn capture_ready(&self) -> bool {
        self.proof_shape.require_capture_ready().is_ok()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ProofPlanError {
    Schedule(ScheduleError),
    Relations(RelationPlanError),
    Shape(ProofShapeError),
    MissingRuntimeComponent(ComponentId),
    UnknownRuntimeComponent(ComponentId),
    PendingRowsForFinalSource(ComponentId),
    FixedLogSizeMismatch {
        component: ComponentId,
        expected_rows: u64,
        actual_rows: u64,
    },
    FixedTraceHasMultipleParts(ComponentId),
    SplitMemoryPartMismatch(ComponentId),
    MissingSubWidth(ComponentId),
    EdgeOutOfBounds {
        producer: ComponentId,
        consumer: ComponentId,
        end_word: u64,
        sub_words: u32,
    },
    EdgeGeometryOverflow {
        producer: ComponentId,
        consumer: ComponentId,
    },
    MissingCapacityFeed(ComponentId),
    CapacityArithmeticOverflow(ComponentId),
    CapacityUsesSplitProducer {
        producer: ComponentId,
        consumer: ComponentId,
    },
}

impl std::fmt::Display for ProofPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ProofPlanError {}

fn validate_row_source(
    node: &ComponentNode,
    runtime: &RuntimeComponentShape,
) -> Result<(), ProofPlanError> {
    if !runtime.is_present() {
        return Ok(());
    }

    if matches!(
        runtime.rows,
        RowResolution::Pending { .. } | RowResolution::Bounded { .. }
    ) && !matches!(
        node.facts.row_source,
        ComponentRowSource::WitnessRelationFeeds
    ) {
        return Err(ProofPlanError::PendingRowsForFinalSource(node.id));
    }

    let RowResolution::Resolved(parts) = &runtime.rows else {
        return Ok(());
    };
    match node.facts.trace_columns {
        TraceColumnCount::Fixed(_) if parts.len() != 1 || parts[0].part != TracePartId::Main => {
            return Err(ProofPlanError::FixedTraceHasMultipleParts(node.id));
        }
        TraceColumnCount::SplitMemory { .. }
            if parts.is_empty()
                || !parts
                    .iter()
                    .any(|part| part.part == TracePartId::MemorySmall)
                || parts.iter().any(|part| part.part == TracePartId::Main) =>
        {
            return Err(ProofPlanError::SplitMemoryPartMismatch(node.id));
        }
        _ => {}
    }
    if let ComponentRowSource::FixedLogSize(log_size) = node.facts.row_source {
        let expected_rows = 1u64.checked_shl(log_size).unwrap_or(0);
        if parts[0].padded_rows != expected_rows {
            return Err(ProofPlanError::FixedLogSizeMismatch {
                component: node.id,
                expected_rows,
                actual_rows: parts[0].padded_rows,
            });
        }
    }
    Ok(())
}

fn resolve_capacity_bounds(
    schedule: &'static Schedule,
    runtime: &mut BTreeMap<ComponentId, RuntimeComponentShape>,
) -> Result<(), ProofPlanError> {
    loop {
        let pending: Vec<_> = runtime
            .iter()
            .filter_map(|(&id, component)| {
                matches!(component.rows, RowResolution::Pending { .. }).then_some(id)
            })
            .collect();
        if pending.is_empty() {
            return Ok(());
        }

        let mut progress = false;
        for id in &pending {
            let node = schedule
                .nodes
                .iter()
                .find(|node| node.id == *id)
                .expect("schedule/runtime completeness checked");
            if node.capacity_inputs.is_empty() {
                continue;
            }
            let mut max_rows = 0u64;
            let mut blocked = false;
            for feed in node.capacity_inputs {
                let producer = runtime
                    .get(feed.from)
                    .expect("schedule validation checked capacity producer");
                let producer_rows = match &producer.rows {
                    RowResolution::Absent => 0,
                    RowResolution::Resolved(parts) if parts.len() == 1 => parts[0].padded_rows,
                    RowResolution::Resolved(_) => {
                        return Err(ProofPlanError::CapacityUsesSplitProducer {
                            producer: feed.from,
                            consumer: node.id,
                        });
                    }
                    RowResolution::Bounded { bound, .. } => bound.padded_capacity,
                    RowResolution::Pending { .. } => {
                        blocked = true;
                        break;
                    }
                };
                max_rows = max_rows
                    .checked_add(
                        producer_rows
                            .checked_mul(u64::from(feed.n_instances))
                            .ok_or(ProofPlanError::CapacityArithmeticOverflow(node.id))?,
                    )
                    .ok_or(ProofPlanError::CapacityArithmeticOverflow(node.id))?;
            }
            if blocked || max_rows == 0 {
                continue;
            }
            let observed_rows = match runtime.get(*id).unwrap().rows {
                RowResolution::Pending {
                    observed_n_real_rows,
                    ..
                } => observed_n_real_rows,
                _ => unreachable!(),
            };
            let padded_capacity = max_rows
                .checked_next_power_of_two()
                .ok_or(ProofPlanError::CapacityArithmeticOverflow(node.id))?
                .max(N_LANES as u64);
            runtime.insert(
                *id,
                RuntimeComponentShape::bounded(
                    *id,
                    PendingRowsReason::WitnessRelationFeeds,
                    CapacityBound {
                        observed_rows,
                        max_rows,
                        padded_capacity,
                    },
                ),
            );
            progress = true;
        }
        if !progress {
            return Err(ProofPlanError::MissingCapacityFeed(pending[0]));
        }
    }
}

fn validate_output_bounds(node: &ComponentNode) -> Result<(), ProofPlanError> {
    if node.outputs.is_empty() {
        return Ok(());
    }
    let sub_words = node
        .facts
        .sub_words
        .ok_or(ProofPlanError::MissingSubWidth(node.id))?;
    for output in node.outputs {
        let width = u64::from(output.words_per_instance)
            .checked_mul(u64::from(output.n_instances))
            .ok_or(ProofPlanError::EdgeGeometryOverflow {
                producer: node.id,
                consumer: output.to,
            })?;
        let end_word = u64::from(output.word_base).checked_add(width).ok_or(
            ProofPlanError::EdgeGeometryOverflow {
                producer: node.id,
                consumer: output.to,
            },
        )?;
        if end_word > u64::from(sub_words) {
            return Err(ProofPlanError::EdgeOutOfBounds {
                producer: node.id,
                consumer: output.to,
                end_word,
                sub_words,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
    use stwo_cairo_prover::witness::proof_shape::{
        PendingRowsReason, ProofShape, RowResolution, RuntimeComponentShape,
    };

    use super::*;
    use crate::relation_table::CAIRO_RELATION_GRAPH;
    use crate::schedule_table::CAIRO_SCHEDULE;

    fn with_components(
        replacements: Vec<RuntimeComponentShape>,
    ) -> stwo_cairo_prover::witness::proof_shape::ProofShape {
        let default = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let mut components = default.components().to_vec();
        for replacement in replacements {
            let slot = components
                .iter_mut()
                .find(|component| component.id == replacement.id)
                .unwrap();
            *slot = replacement;
        }
        ProofShape::new(components).unwrap()
    }

    #[test]
    fn generated_feed_bound_is_arena_ready_but_not_capture_ready() {
        // blake_compress emits ten blake_round inputs per padded producer row.
        let shape = with_components(vec![
            RuntimeComponentShape::uniform("blake_compress_opcode", 33, 64).unwrap(),
            RuntimeComponentShape::pending(
                "blake_round",
                PendingRowsReason::WitnessRelationFeeds,
                0,
            ),
        ]);
        let plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        assert!(plan.arena_capacity_ready());
        assert!(!plan.capture_ready());

        let blake_round = plan.proof_shape().component("blake_round").unwrap();
        let RowResolution::Bounded { bound, .. } = blake_round.rows else {
            panic!("blake_round was not capacity-bounded")
        };
        assert_eq!(bound.observed_rows, 0);
        assert_eq!(bound.max_rows, 64 * 10);
        assert_eq!(bound.padded_capacity, 1024);
    }

    #[test]
    fn exact_shape_passes_capture_gate() {
        let shape = with_components(vec![
            RuntimeComponentShape::uniform("blake_compress_opcode", 33, 64).unwrap(),
            RuntimeComponentShape::uniform("blake_round", 640, 1024).unwrap(),
        ]);
        let plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        assert!(plan.arena_capacity_ready());
        assert!(plan.capture_ready());
    }
}
