//! Ingest: adapter output → preprocessed trace + claim generator (design §5.1).

use std::sync::Arc;

use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::witness::cairo::create_cairo_claim_generator;
use tracing::{span, Level};

use crate::state::IngestOutput;

pub fn run(input: ProverInput, variant: PreProcessedTraceVariant) -> IngestOutput {
    let span = span!(Level::INFO, "Write Preprocessed trace").entered();
    let preprocessed_trace = Arc::new(variant.to_preprocessed_trace());
    span.exit();

    let generator = create_cairo_claim_generator(input, preprocessed_trace.clone());
    IngestOutput {
        preprocessed_trace,
        generator,
    }
}
