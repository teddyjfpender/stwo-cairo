//! Ingest: adapter output → preprocessed trace + claim generator (design §5.1).

use std::sync::Arc;

use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::witness::cairo::create_cairo_claim_generator;
use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
use stwo_cairo_prover::witness::jit_prove_backend::{
    planned_compacted_consumer_shape, recorded_input_compaction_geometry,
};
use stwo_cairo_prover::witness::proof_shape::{ProofShape, RowResolution};
use tracing::{span, Level};

use crate::plan::ProofPlan;
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::schedule_table::CAIRO_SCHEDULE;
use crate::state::IngestOutput;

pub fn run(
    input: ProverInput,
    variant: PreProcessedTraceVariant,
    opt_n_id_to_big_components: Option<usize>,
) -> IngestOutput {
    let span = span!(Level::INFO, "Write Preprocessed trace").entered();
    let preprocessed_trace = Arc::new(variant.to_preprocessed_trace());
    span.exit();

    let generator = create_cairo_claim_generator(input, preprocessed_trace.clone());
    let observed_shape = generator
        .proof_shape(opt_n_id_to_big_components)
        .expect("pre-witness proof shape is invalid");
    // First pass: the generated capacity contract over the observed shape.
    // The device-compacted consumers come out of it CAPACITY-BOUNDED (their
    // exact rows are witness-data-dependent RLE dedups), and those bounds are
    // the fail-closed ceiling for the host derivation sealed below.
    let capacity_plan =
        ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &observed_shape)
            .expect("pre-witness proof shape disagrees with generated component facts");
    // Timed: the verify_instruction arm sorts+dedups one u32 per executed
    // step, so its host cost must stay visible in the phase ledger for
    // SN-scale inputs (plan of record: measure before optimizing).
    let seal_span = span!(Level::INFO, "Seal compacted consumer rows").entered();
    let sealed_shape = seal_compacted_consumer_rows(&generator, &observed_shape, &capacity_plan);
    seal_span.exit();
    // Second pass: the sealed shape keeps every OTHER witness-feed consumer
    // PENDING (it splices into the observed shape, not the capacity plan), so
    // this resolution recomputes every downstream feed bound from the sealed
    // exact padded domains — which is what makes the strict plan's plain-feed
    // promotions exact for consumers downstream of the compacted trio.
    let proof_plan = Arc::new(
        ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &sealed_shape)
            .expect("host-sealed compacted consumer rows disagree with generated component facts"),
    );
    IngestOutput {
        preprocessed_trace,
        generator,
        proof_plan,
    }
}

/// Replace every device-compacted consumer's pending rows in the OBSERVED
/// shape with the exact host-derived row resolution, pre-witness. Every other
/// witness-feed consumer is deliberately left `Pending` so the subsequent
/// `ProofPlan::from_schedule` recomputes its capacity bound from the sealed
/// exact padded domains (splicing into the already-bounded capacity plan
/// instead would freeze downstream bounds at the unsealed capacities).
///
/// INVARIANT (soundness-critical): for every device-compacted consumer
/// (`verify_instruction`, `pedersen_aggregator_window_bits_18`,
/// `poseidon_aggregator`), the sealed row resolution MUST equal the row count
/// the SIMD `CairoClaimGenerator` realizes for the same ProverInput — same
/// dedup semantics, same padding to the next power of two. The derivation
/// itself (`planned_compacted_consumer_shape`) reuses the components' own
/// `AddInputs` dedup and `FinalComponentShape` projection; this function only
/// splices the result into the shape and fail-closes if a derived count ever
/// exceeds the generated capacity bound (impossible unless the dedup logic
/// drifts from the schedule's feed formula). Backstops downstream: the
/// witness-time `FinalShapeLedger` equality check, the device
/// compact-finalize trap, and whole-proof byte identity.
pub fn seal_compacted_consumer_rows(
    generator: &CairoClaimGenerator,
    observed_shape: &ProofShape,
    capacity_plan: &ProofPlan,
) -> ProofShape {
    let components = observed_shape
        .components()
        .iter()
        .map(|component| {
            if recorded_input_compaction_geometry(component.id).is_none() {
                return component.clone();
            }
            match &component.rows {
                RowResolution::Absent => component.clone(),
                RowResolution::Pending { .. } => {
                    let derived = planned_compacted_consumer_shape(generator, component.id)
                        .unwrap_or_else(|error| {
                            panic!(
                                "compacted consumer {} row derivation failed closed: {error}",
                                component.id
                            )
                        })
                        .unwrap_or_else(|| {
                            panic!(
                                "compacted consumer {} is present in the proof shape but the \
                                 host derivation reported it absent",
                                component.id
                            )
                        });
                    let RowResolution::Resolved(parts) = &derived.rows else {
                        panic!(
                            "compacted consumer {} host derivation returned unresolved rows",
                            component.id
                        );
                    };
                    let capacity = capacity_plan
                        .proof_shape()
                        .component(component.id)
                        .unwrap_or_else(|| {
                            panic!(
                                "compacted consumer {} is missing from the capacity plan",
                                component.id
                            )
                        });
                    let RowResolution::Bounded { bound, .. } = &capacity.rows else {
                        panic!(
                            "compacted consumer {} is not capacity-bounded in the capacity plan: \
                             {:?}",
                            component.id, capacity.rows
                        );
                    };
                    assert!(
                        parts.len() == 1
                            && parts[0].n_real_rows <= bound.max_rows
                            && parts[0].padded_rows <= bound.padded_capacity,
                        "compacted consumer {} host-derived rows {:?} exceed the generated \
                         capacity bound {bound:?}: the host dedup derivation drifted from the \
                         schedule's feed formula",
                        component.id,
                        parts,
                    );
                    derived
                }
                other => panic!(
                    "compacted consumer {} entered sealing with unexpected pre-witness rows \
                     {other:?}",
                    component.id
                ),
            }
        })
        .collect();
    ProofShape::new(components).expect("host-sealed compacted consumer rows kept the shape valid")
}
