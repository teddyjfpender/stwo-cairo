//! Witness: base-trace generation (design §5.2).
//!
//! M1: delegates to the claim generator's `write_trace` — the device lanes
//! (witness JIT, count feeds, edges) engage inside it exactly as on the legacy
//! path. M2 moves ownership here: the schedule table drives the component DAG
//! and this module becomes the launch issuer.

use stwo::prover::poly::twiddles::TwiddleTree;
use tracing::{span, Level};

use crate::prover::CairoWitnessBackend;
use crate::state::WitnessOutput;

pub fn run<B: CairoWitnessBackend>(
    generator: stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator,
    opt_n_id_to_big_components: Option<usize>,
    pipeline_twiddles: Option<&'static TwiddleTree<B>>,
) -> WitnessOutput<B> {
    let span = span!(Level::INFO, "Write Base trace").entered();
    let (trace, claim, interaction_generator) =
        generator.write_trace::<B>(opt_n_id_to_big_components, pipeline_twiddles);
    span.exit();
    WitnessOutput {
        trace,
        claim,
        interaction_generator,
    }
}
