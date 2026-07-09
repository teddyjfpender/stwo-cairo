//! Ingest: adapter output → preprocessed trace + claim generator (design §5.1).

use std::sync::Arc;

use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::witness::cairo::create_cairo_claim_generator;
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
    let proof_plan = Arc::new(
        ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &observed_shape)
            .expect("pre-witness proof shape disagrees with generated component facts"),
    );
    IngestOutput {
        preprocessed_trace,
        generator,
        proof_plan,
    }
}
