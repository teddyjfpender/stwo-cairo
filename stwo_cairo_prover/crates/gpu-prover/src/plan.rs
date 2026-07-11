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

    /// Resolve generated witness-feed capacities to the exact row geometry of
    /// the strict device DAG before materialization. Every producer contributes
    /// its full padded row domain `n_instances` times; this is the same formula
    /// the live device-edge ledger checks after the writers run. The post-write
    /// seal remains mandatory and rejects any disagreement.
    ///
    /// The capacity formula is exact only for PLAIN feed consumers, whose
    /// device writers process every padded producer row (a padded producer row
    /// times a power-of-two instance count keeps the same padded power of
    /// two). A DEVICE-COMPACTED consumer (RLE multiset compaction, e.g.
    /// `verify_instruction` and the aggregators) deduplicates the gathered
    /// feed, so its exact row count is a property of the witness data and is
    /// NOT derivable from producer capacities. Promoting the capacity bound
    /// for such a component produced a claim whose log size diverges from the
    /// host reference AND armed the compact kernel's fail-closed device trap
    /// (observed on hardware as a SIGSEGV inside `cudaGraphLaunch`); this now
    /// fails closed at plan time instead.
    pub fn strict_resident_exact(
        &self,
        schedule: &'static Schedule,
        relation_graph: &'static RelationGraph,
    ) -> Result<Self, ProofPlanError> {
        let components = self
            .proof_shape
            .components()
            .iter()
            .map(|component| {
                let rows = match &component.rows {
                    RowResolution::Bounded { bound, .. } => {
                        if stwo_cairo_prover::witness::jit_prove_backend::recorded_input_compaction_geometry(
                            component.id,
                        )
                        .is_some()
                        {
                            return Err(ProofPlanError::StrictResidentCompactedRowsUnresolved {
                                component: component.id,
                                observed_rows: bound.observed_rows,
                                max_rows: bound.max_rows,
                                padded_capacity: bound.padded_capacity,
                            });
                        }
                        RowResolution::Resolved(vec![
                            stwo_cairo_prover::witness::proof_shape::TracePartShape {
                                part: TracePartId::Main,
                                n_real_rows: bound.max_rows,
                                padded_rows: bound.padded_capacity,
                            },
                        ])
                    }
                    RowResolution::Pending { .. } => {
                        return Err(ProofPlanError::StrictResidentRowsUnresolved(component.id));
                    }
                    _ => component.rows.clone(),
                };
                Ok(RuntimeComponentShape {
                    id: component.id,
                    rows,
                })
            })
            .collect::<Result<Vec<_>, ProofPlanError>>()?;
        let shape = ProofShape::new(components).map_err(ProofPlanError::Shape)?;
        let exact = Self::from_schedule(schedule, relation_graph, &shape)?;
        debug_assert!(exact.capture_ready());
        Ok(exact)
    }

    /// Rebuilds the generated plan from the post-witness exact row ledger while
    /// proving that no component, trace width/order, or preallocated capacity
    /// changed. CUDA graph preparation must only consume the returned plan.
    pub fn seal_exact_shape(
        &self,
        schedule: &'static Schedule,
        relation_graph: &'static RelationGraph,
        exact_shape: &ProofShape,
    ) -> Result<Self, ProofPlanError> {
        exact_shape
            .require_capture_ready()
            .map_err(ProofPlanError::Shape)?;

        for component in &self.components {
            let exact = exact_shape
                .component(component.node.id)
                .ok_or(ProofPlanError::MissingRuntimeComponent(component.node.id))?;
            validate_sealed_runtime(&component.runtime, exact)?;
        }

        let sealed = Self::from_schedule(schedule, relation_graph, exact_shape)?;
        if sealed.relation_graph_hash != self.relation_graph_hash {
            return Err(ProofPlanError::SealedRelationGraphChanged {
                expected: self.relation_graph_hash,
                actual: sealed.relation_graph_hash,
            });
        }
        if sealed.components.len() != self.components.len() {
            return Err(ProofPlanError::SealedComponentCountChanged {
                expected: self.components.len(),
                actual: sealed.components.len(),
            });
        }
        for (index, (expected, actual)) in
            self.components.iter().zip(&sealed.components).enumerate()
        {
            if expected.node.id != actual.node.id {
                return Err(ProofPlanError::SealedComponentOrderChanged {
                    index,
                    expected: expected.node.id,
                    actual: actual.node.id,
                });
            }
            if expected.node.facts != actual.node.facts {
                return Err(ProofPlanError::SealedTraceGeometryChanged(expected.node.id));
            }
        }
        debug_assert!(sealed.capture_ready());
        Ok(sealed)
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
    SealedExpectedRowsPending(ComponentId),
    SealedActualRowsNotExact(ComponentId),
    SealedPresenceChanged(ComponentId),
    SealedResolvedGeometryChanged(ComponentId),
    SealedRowsBelowObserved {
        component: ComponentId,
        observed_rows: u64,
        final_rows: u64,
    },
    SealedRowsExceedCapacity {
        component: ComponentId,
        final_rows: u64,
        max_rows: u64,
    },
    SealedPaddingExceedsCapacity {
        component: ComponentId,
        final_padding: u64,
        padded_capacity: u64,
    },
    SealedComponentCountChanged {
        expected: usize,
        actual: usize,
    },
    SealedComponentOrderChanged {
        index: usize,
        expected: ComponentId,
        actual: ComponentId,
    },
    SealedTraceGeometryChanged(ComponentId),
    StrictResidentRowsUnresolved(ComponentId),
    /// A capacity bound cannot stand in for exact rows on a device-compacted
    /// consumer. The RLE compaction shrinks the gathered feed by the
    /// input-dependent number of duplicate tuples, so the exact row count is
    /// unknowable from producer capacities alone. Hardware-verified on the
    /// SN2-profile fixture: the compact finalize kernel's fail-closed trap
    /// fired with 174 unique tuples (256 padded rows) against this
    /// capacity-planned consumer size, poisoning the CUDA context in the
    /// middle of `cudaGraphLaunch` (SIGSEGV inside the driver). Failing here,
    /// at plan time, replaces that undefined behavior with a checked error.
    StrictResidentCompactedRowsUnresolved {
        component: ComponentId,
        observed_rows: u64,
        max_rows: u64,
        padded_capacity: u64,
    },
    SealedRelationGraphChanged {
        expected: u64,
        actual: u64,
    },
}

impl std::fmt::Display for ProofPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ProofPlanError {}

/// TEST-ONLY stand-in for the exact rows of device-compacted consumers.
/// [`ProofPlan::strict_resident_exact`] fails closed on their capacity bounds,
/// and the pre-witness exact-count derivation does not exist yet, so tests of
/// lane/arena geometry substitute the capacity geometry explicitly before the
/// strict resolution. Production code must never take this shortcut: the
/// substituted row counts are upper bounds, not the deduplicated truth.
#[cfg(test)]
pub(crate) fn resolve_compacted_capacity_for_test(
    plan: &ProofPlan,
    schedule: &'static Schedule,
    relation_graph: &'static RelationGraph,
) -> ProofPlan {
    let components = plan
        .proof_shape()
        .components()
        .iter()
        .map(|component| {
            let rows = match &component.rows {
                RowResolution::Bounded { bound, .. }
                    if stwo_cairo_prover::witness::jit_prove_backend::recorded_input_compaction_geometry(
                        component.id,
                    )
                    .is_some() =>
                {
                    RowResolution::Resolved(vec![
                        stwo_cairo_prover::witness::proof_shape::TracePartShape {
                            part: TracePartId::Main,
                            n_real_rows: bound.max_rows,
                            padded_rows: bound.padded_capacity,
                        },
                    ])
                }
                _ => component.rows.clone(),
            };
            RuntimeComponentShape {
                id: component.id,
                rows,
            }
        })
        .collect();
    let shape = ProofShape::new(components).expect("test capacity substitution kept a valid shape");
    ProofPlan::from_schedule(schedule, relation_graph, &shape)
        .expect("test capacity substitution kept a plannable shape")
}

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

fn validate_sealed_runtime(
    expected: &RuntimeComponentShape,
    exact: &RuntimeComponentShape,
) -> Result<(), ProofPlanError> {
    match (&expected.rows, &exact.rows) {
        (RowResolution::Absent, RowResolution::Absent) => Ok(()),
        (RowResolution::Absent, _) | (_, RowResolution::Absent) => {
            Err(ProofPlanError::SealedPresenceChanged(expected.id))
        }
        (RowResolution::Resolved(_), RowResolution::Resolved(_)) => {
            if expected == exact {
                Ok(())
            } else {
                Err(ProofPlanError::SealedResolvedGeometryChanged(expected.id))
            }
        }
        (RowResolution::Pending { .. }, _) => {
            Err(ProofPlanError::SealedExpectedRowsPending(expected.id))
        }
        (RowResolution::Bounded { bound, .. }, RowResolution::Resolved(parts)) => {
            if parts.len() != 1 || parts[0].part != TracePartId::Main {
                return Err(ProofPlanError::SealedResolvedGeometryChanged(expected.id));
            }
            let part = parts[0];
            if part.n_real_rows < bound.observed_rows {
                return Err(ProofPlanError::SealedRowsBelowObserved {
                    component: expected.id,
                    observed_rows: bound.observed_rows,
                    final_rows: part.n_real_rows,
                });
            }
            if part.n_real_rows > bound.max_rows {
                return Err(ProofPlanError::SealedRowsExceedCapacity {
                    component: expected.id,
                    final_rows: part.n_real_rows,
                    max_rows: bound.max_rows,
                });
            }
            if part.padded_rows > bound.padded_capacity {
                return Err(ProofPlanError::SealedPaddingExceedsCapacity {
                    component: expected.id,
                    final_padding: part.padded_rows,
                    padded_capacity: bound.padded_capacity,
                });
            }
            Ok(())
        }
        (RowResolution::Bounded { .. }, _) => {
            Err(ProofPlanError::SealedActualRowsNotExact(expected.id))
        }
        (_, RowResolution::Pending { .. } | RowResolution::Bounded { .. }) => {
            Err(ProofPlanError::SealedActualRowsNotExact(expected.id))
        }
    }
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

    /// A device-compacted consumer deduplicates its gathered feed, so a
    /// capacity bound can never stand in for its exact rows. Promoting it
    /// armed the compact kernel's fail-closed device trap (hardware SIGSEGV
    /// inside `cudaGraphLaunch` on the SN2-profile fixture); the strict
    /// resident plan must reject it at plan time instead.
    #[test]
    fn strict_resident_exact_rejects_capacity_bounded_compacted_consumer() {
        let shape = with_components(vec![
            RuntimeComponentShape::uniform("ret_opcode", 33, 64).unwrap(),
            RuntimeComponentShape::pending(
                "verify_instruction",
                PendingRowsReason::WitnessRelationFeeds,
                0,
            ),
        ]);
        let plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let verify_instruction = plan.proof_shape().component("verify_instruction").unwrap();
        assert!(
            matches!(verify_instruction.rows, RowResolution::Bounded { .. }),
            "test setup: verify_instruction must be capacity-bounded"
        );
        assert!(matches!(
            plan.strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH),
            Err(ProofPlanError::StrictResidentCompactedRowsUnresolved {
                component: "verify_instruction",
                ..
            })
        ));
    }

    #[test]
    fn post_witness_seal_replaces_bound_with_exact_rows() {
        let pre_shape = with_components(vec![
            RuntimeComponentShape::uniform("blake_compress_opcode", 33, 64).unwrap(),
            RuntimeComponentShape::pending(
                "blake_round",
                PendingRowsReason::WitnessRelationFeeds,
                0,
            ),
        ]);
        let capacity_plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &pre_shape).unwrap();
        let exact_shape = with_components(vec![
            RuntimeComponentShape::uniform("blake_compress_opcode", 33, 64).unwrap(),
            RuntimeComponentShape::uniform("blake_round", 640, 1024).unwrap(),
        ]);

        let sealed = capacity_plan
            .seal_exact_shape(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &exact_shape)
            .unwrap();
        assert!(sealed.capture_ready());
        assert_eq!(sealed.proof_shape(), &exact_shape);
    }

    #[test]
    fn post_witness_seal_rejects_rows_beyond_generated_capacity() {
        let pre_shape = with_components(vec![
            RuntimeComponentShape::uniform("blake_compress_opcode", 33, 64).unwrap(),
            RuntimeComponentShape::pending(
                "blake_round",
                PendingRowsReason::WitnessRelationFeeds,
                0,
            ),
        ]);
        let capacity_plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &pre_shape).unwrap();
        let exact_shape = with_components(vec![
            RuntimeComponentShape::uniform("blake_compress_opcode", 33, 64).unwrap(),
            RuntimeComponentShape::uniform("blake_round", 641, 1024).unwrap(),
        ]);

        assert!(matches!(
            capacity_plan.seal_exact_shape(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &exact_shape),
            Err(ProofPlanError::SealedRowsExceedCapacity {
                component: "blake_round",
                final_rows: 641,
                max_rows: 640,
            })
        ));
    }

    #[test]
    fn strict_resident_plan_resolves_generated_feed_rows_before_witness() {
        let pre_shape = with_components(vec![
            RuntimeComponentShape::uniform("blake_compress_opcode", 33, 64).unwrap(),
            RuntimeComponentShape::pending(
                "blake_round",
                PendingRowsReason::WitnessRelationFeeds,
                0,
            ),
        ]);
        let capacity =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &pre_shape).unwrap();
        let exact = capacity
            .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
            .unwrap();
        assert_eq!(
            exact.proof_shape().component("blake_round").unwrap().rows,
            RowResolution::Resolved(vec![
                stwo_cairo_prover::witness::proof_shape::TracePartShape {
                    part: TracePartId::Main,
                    n_real_rows: 640,
                    padded_rows: 1024,
                }
            ])
        );
        assert!(exact.capture_ready());
    }
}
