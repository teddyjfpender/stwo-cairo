//! Interaction (logup) trace generation (design §5.3).
//!
//! M1: delegates to the interaction generator — the §6a device-interaction lanes
//! engage inside it exactly as on the legacy path. The channel operations around
//! this call (grind, draw, mix) live in `prove()`: they are the transcript spine.

use cairo_air::claims::CairoInteractionClaim;
use cairo_air::relations::CommonLookupElements;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use tracing::{span, Level};

use crate::prover::CairoWitnessBackend;

#[allow(clippy::type_complexity)]
pub fn run<B: CairoWitnessBackend>(
    interaction_generator: stwo_cairo_prover::witness::cairo_claim_generator::CairoInteractionClaimGenerator<B>,
    interaction_elements: &CommonLookupElements,
) -> (
    Vec<CircleEvaluation<B, stwo::core::fields::m31::BaseField, BitReversedOrder>>,
    CairoInteractionClaim,
) {
    let _span = span!(Level::INFO, "Write interaction trace").entered();
    interaction_generator.write_interaction_trace(interaction_elements)
}
