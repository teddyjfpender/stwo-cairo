//! Device-born base-witness bindings.
//!
//! This is the only adapter between the generated witness layer and the proof
//! arena. Recorded CUDA writers receive borrowed views of their canonical
//! BaseTrace/LookupInputs/SubcomponentInputs slots and launch on the owning
//! workspace stream. No writer may infer column order from vector position.

use cairo_air::air::PublicData;
use cairo_air::claims::CairoClaim;
use serde_json::{Map, Value};
use stwo_backend_cuda::BaseFieldVec;
use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
use stwo_cairo_prover::witness::exec_context::{ResidentWitnessDestination, ResidentWitnessPlan};
use stwo_cairo_prover::witness::proof_shape::{RowResolution, TracePartId};

use crate::arena_plan::BufferPurpose;
use crate::graphs::{GraphError, GraphWorkspace};
use crate::plan::ProofPlan;
use crate::schedule::{ComponentRowSource, TraceColumnCount};

#[derive(Debug)]
pub enum ResidentWitnessPlanError {
    UnsupportedComponents(Vec<&'static str>),
    ShapeNotExact(&'static str),
    InvalidTracePart {
        component: &'static str,
        part: TracePartId,
    },
    MissingBuffer {
        component: &'static str,
        part: TracePartId,
        purpose: BufferPurpose,
        ordinal: u32,
    },
    BufferSizeMismatch {
        component: &'static str,
        part: TracePartId,
        purpose: BufferPurpose,
        expected: usize,
        actual: usize,
    },
    SizeOverflow,
    ClaimSerialization(serde_json::Error),
    Graph(GraphError),
}

impl core::fmt::Display for ResidentWitnessPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident witness plan rejected: {self:?}")
    }
}

impl std::error::Error for ResidentWitnessPlanError {}

impl From<GraphError> for ResidentWitnessPlanError {
    fn from(value: GraphError) -> Self {
        Self::Graph(value)
    }
}

impl From<serde_json::Error> for ResidentWitnessPlanError {
    fn from(value: serde_json::Error) -> Self {
        Self::ClaimSerialization(value)
    }
}

/// Fail closed before witness generation if any present component lacks an
/// arena-native CUDA writer. Public schedule entries cannot hide behind Rust's
/// dead-code lints, so strict coverage is derived from the exact proof plan on
/// every proof rather than from a manually maintained allowlist.
pub fn require_strict_resident_witness_coverage(
    proof_plan: &ProofPlan,
) -> Result<(), ResidentWitnessPlanError> {
    let unsupported = proof_plan
        .components
        .iter()
        .filter(|component| component.runtime.is_present())
        .filter(|component| !component.node.facts.witness_writer.is_capture_safe())
        .map(|component| component.node.id)
        .collect::<Vec<_>>();
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(ResidentWitnessPlanError::UnsupportedComponents(unsupported))
    }
}

/// Build the statement claim from the exact pre-witness plan. Cairo component
/// claims contain only presence plus their committed log size (or are empty
/// fixed-table structs); the post-write claim is serialized and compared with
/// this value before transcript use.
pub fn planned_cairo_claim(
    generator: &CairoClaimGenerator,
    proof_plan: &ProofPlan,
) -> Result<CairoClaim, ResidentWitnessPlanError> {
    planned_cairo_claim_from_public_data(&generator.public_data, proof_plan)
}

pub fn planned_cairo_claim_from_public_data(
    public_data: &PublicData,
    proof_plan: &ProofPlan,
) -> Result<CairoClaim, ResidentWitnessPlanError> {
    let mut claim = Map::new();
    claim.insert(
        "public_data".to_string(),
        serde_json::to_value(public_data)?,
    );
    let mut memory_small = Value::Null;
    for component in &proof_plan.components {
        let value = match &component.runtime.rows {
            RowResolution::Absent => Value::Null,
            RowResolution::Resolved(parts) => match component.node.facts.row_source {
                ComponentRowSource::FixedLogSize(_) => Value::Object(Map::new()),
                ComponentRowSource::MemoryIdToBig => {
                    let mut big = parts
                        .iter()
                        .filter_map(|part| match part.part {
                            TracePartId::MemoryBig(index) => {
                                Some((index, part.padded_rows.ilog2()))
                            }
                            TracePartId::MemorySmall => {
                                memory_small = serde_json::json!({
                                    "log_size": part.padded_rows.ilog2()
                                });
                                None
                            }
                            TracePartId::Main => None,
                        })
                        .collect::<Vec<_>>();
                    big.sort_unstable_by_key(|(index, _)| *index);
                    serde_json::json!({
                        "big_log_sizes": big.into_iter().map(|(_, log)| log).collect::<Vec<_>>()
                    })
                }
                _ => {
                    if parts.len() != 1 || parts[0].part != TracePartId::Main {
                        return Err(ResidentWitnessPlanError::InvalidTracePart {
                            component: component.node.id,
                            part: parts.first().map_or(TracePartId::Main, |part| part.part),
                        });
                    }
                    serde_json::json!({"log_size": parts[0].padded_rows.ilog2()})
                }
            },
            RowResolution::Pending { .. } | RowResolution::Bounded { .. } => {
                return Err(ResidentWitnessPlanError::ShapeNotExact(component.node.id));
            }
        };
        claim.insert(component.node.id.to_string(), value);
    }
    claim.insert("memory_id_to_small".to_string(), memory_small);
    Ok(serde_json::from_value(Value::Object(claim))?)
}

pub fn cairo_claims_match(planned: &CairoClaim, actual: &CairoClaim) -> bool {
    serde_json::to_value(planned).ok() == serde_json::to_value(actual).ok()
}

/// Bind every recorded CUDA witness component directly to the exact workspace.
/// Components without a recorded kernel remain on the explicitly measured
/// migration path; strict session staging rejects any detached base column.
pub fn bind_resident_witness_plan(
    workspace: &GraphWorkspace,
    proof_plan: &ProofPlan,
    strict: bool,
) -> Result<ResidentWitnessPlan, ResidentWitnessPlanError> {
    let mut destinations = Vec::new();
    for component in &proof_plan.components {
        if !component.node.facts.witness_writer.uses_arena_destination() {
            continue;
        }
        let RowResolution::Resolved(parts) = &component.runtime.rows else {
            if matches!(component.runtime.rows, RowResolution::Absent) {
                continue;
            }
            return Err(ResidentWitnessPlanError::ShapeNotExact(component.node.id));
        };
        for part in parts {
            let rows = usize::try_from(part.padded_rows)
                .map_err(|_| ResidentWitnessPlanError::SizeOverflow)?;
            let trace_columns = trace_width(component.node.facts.trace_columns, part.part)?;
            let mut trace = Vec::with_capacity(trace_columns as usize);
            for ordinal in 0..trace_columns {
                trace.push(bind_column(
                    workspace,
                    component.node.id,
                    part.part,
                    BufferPurpose::BaseTrace,
                    ordinal,
                    rows,
                )?);
            }
            let dummy = trace
                .first()
                .expect("every Cairo witness component has at least one base column")
                .device_ptr;
            let lookup = bind_flat_or_dummy(
                workspace,
                component.node.id,
                part.part,
                BufferPurpose::LookupInputs,
                component.node.facts.lookup_words,
                rows,
                dummy,
            )?;
            let sub = bind_flat_or_dummy(
                workspace,
                component.node.id,
                part.part,
                BufferPurpose::SubcomponentInputs,
                component.node.facts.sub_words,
                rows,
                dummy,
            )?;
            destinations.push(ResidentWitnessDestination {
                component: component.node.id,
                part: part.part,
                trace,
                lookup,
                sub,
            });
        }
    }
    Ok(ResidentWitnessPlan {
        strict,
        context: workspace.arena().context().launch_context(),
        destinations,
    })
}

fn trace_width(
    columns: TraceColumnCount,
    part: TracePartId,
) -> Result<u32, ResidentWitnessPlanError> {
    match (columns, part) {
        (TraceColumnCount::Fixed(columns), TracePartId::Main) => Ok(columns),
        (TraceColumnCount::SplitMemory { big, .. }, TracePartId::MemoryBig(_)) => Ok(big),
        (TraceColumnCount::SplitMemory { small, .. }, TracePartId::MemorySmall) => Ok(small),
        _ => Err(ResidentWitnessPlanError::InvalidTracePart {
            component: "<trace-width>",
            part,
        }),
    }
}

fn bind_column(
    workspace: &GraphWorkspace,
    component: &'static str,
    part: TracePartId,
    purpose: BufferPurpose,
    ordinal: u32,
    expected_words: usize,
) -> Result<BaseFieldVec, ResidentWitnessPlanError> {
    let (logical, _) = workspace
        .plan()
        .find(Some(component), Some(part), purpose, ordinal)
        .ok_or(ResidentWitnessPlanError::MissingBuffer {
            component,
            part,
            purpose,
            ordinal,
        })?;
    let (slice, logical_words) = workspace.bind(logical.id)?;
    if logical_words != expected_words {
        return Err(ResidentWitnessPlanError::BufferSizeMismatch {
            component,
            part,
            purpose,
            expected: expected_words,
            actual: logical_words,
        });
    }
    Ok(BaseFieldVec::from_borrowed_ptr(
        slice.as_u32_ptr(),
        logical_words,
    ))
}

fn bind_flat_or_dummy(
    workspace: &GraphWorkspace,
    component: &'static str,
    part: TracePartId,
    purpose: BufferPurpose,
    words_per_row: Option<u32>,
    rows: usize,
    dummy: *const u32,
) -> Result<BaseFieldVec, ResidentWitnessPlanError> {
    let Some(words_per_row) = words_per_row else {
        return Ok(BaseFieldVec::from_borrowed_ptr(dummy, 1));
    };
    let expected_words = usize::try_from(words_per_row)
        .ok()
        .and_then(|words| words.checked_mul(rows))
        .ok_or(ResidentWitnessPlanError::SizeOverflow)?;
    bind_column(workspace, component, part, purpose, 0, expected_words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::{
        KernelIdentitySource, WitnessWriterKind, WitnessWriterReadiness, WitnessWriterSpec,
    };
    use crate::schedule_table::CAIRO_SCHEDULE;

    #[test]
    fn recorded_witness_selection_is_schedule_derived() {
        let selected: Vec<_> = CAIRO_SCHEDULE
            .nodes
            .iter()
            .filter(|node| node.facts.kernel_identity == KernelIdentitySource::RecordedWitness)
            .map(|node| node.id)
            .collect();
        assert!(selected.contains(&"add_opcode"));
        assert!(selected.contains(&"add_ap_opcode"));
        assert!(selected.contains(&"triple_xor_32"));
        assert!(selected.contains(&"pedersen_aggregator_window_bits_18"));
        assert!(!selected.contains(&"range_check_6"));
    }

    #[test]
    fn writer_kind_and_readiness_are_distinct_contracts() {
        let selected = CAIRO_SCHEDULE
            .nodes
            .iter()
            .filter(|node| node.facts.witness_writer.kind == WitnessWriterKind::NativeCuda)
            .map(|node| node.id)
            .collect::<Vec<_>>();
        assert_eq!(
            selected,
            ["ec_op_builtin", "memory_address_to_id", "memory_id_to_big"]
        );
        let recorded = CAIRO_SCHEDULE
            .nodes
            .iter()
            .find(|node| node.id == "blake_g")
            .unwrap()
            .facts
            .witness_writer;
        assert_eq!(recorded.kind, WitnessWriterKind::RecordedAot);
        assert!(recorded.uses_arena_destination());
        assert!(recorded.is_capture_safe());
        for component in selected {
            let native = CAIRO_SCHEDULE
                .nodes
                .iter()
                .find(|node| node.id == component)
                .unwrap()
                .facts
                .witness_writer;
            assert!(native.uses_arena_destination(), "{component}");
            assert!(native.is_capture_safe(), "{component}");
        }
        assert!(WitnessWriterSpec {
            kind: WitnessWriterKind::FixedTableCuda,
            readiness: WitnessWriterReadiness::CaptureSafe,
        }
        .is_capture_safe());
        assert!(!WitnessWriterSpec::HOST.uses_arena_destination());
    }

    #[test]
    fn planned_default_claim_roundtrips_from_exact_shape() {
        use crate::relation_table::CAIRO_RELATION_GRAPH;

        let generator = CairoClaimGenerator::default();
        let shape = generator.proof_shape(None).unwrap();
        let plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let claim = planned_cairo_claim(&generator, &plan).unwrap();
        let value = serde_json::to_value(claim).unwrap();
        assert!(value["add_opcode"].is_null());
        assert!(value["memory_id_to_small"].is_null());
    }

    #[test]
    fn planned_fixed_table_claims_use_their_empty_struct_schema() {
        use stwo_cairo_prover::witness::proof_shape::{ProofShape, RuntimeComponentShape};

        use crate::relation_table::CAIRO_RELATION_GRAPH;

        let generator = CairoClaimGenerator::default();
        let default = generator.proof_shape(None).unwrap();
        let mut components = default.components().to_vec();
        for node in CAIRO_SCHEDULE.nodes {
            let ComponentRowSource::FixedLogSize(log_size) = node.facts.row_source else {
                continue;
            };
            let slot = components
                .iter_mut()
                .find(|component| component.id == node.id)
                .unwrap();
            let rows = 1u64 << log_size;
            *slot = RuntimeComponentShape::uniform(node.id, rows, rows).unwrap();
        }
        let shape = ProofShape::new(components).unwrap();
        let plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let claim = serde_json::to_value(planned_cairo_claim(&generator, &plan).unwrap()).unwrap();
        assert_eq!(claim["blake_round_sigma"], serde_json::json!({}));
        assert_eq!(claim["range_check_20"], serde_json::json!({}));
    }

    #[test]
    fn strict_coverage_accepts_prepared_recorded_writers() {
        use stwo_cairo_prover::witness::proof_shape::{ProofShape, RuntimeComponentShape};

        use crate::relation_table::CAIRO_RELATION_GRAPH;

        let generator = CairoClaimGenerator::default();
        let default = generator.proof_shape(None).unwrap();
        let mut components = default.components().to_vec();
        for (id, rows) in [("add_ap_opcode", 16), ("add_opcode", 16)] {
            let slot = components
                .iter_mut()
                .find(|component| component.id == id)
                .unwrap();
            *slot = RuntimeComponentShape::uniform(id, rows, rows).unwrap();
        }
        let shape = ProofShape::new(components).unwrap();
        let plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        require_strict_resident_witness_coverage(&plan).unwrap();
    }
}
