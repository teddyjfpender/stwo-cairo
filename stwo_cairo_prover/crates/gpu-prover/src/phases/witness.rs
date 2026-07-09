//! Witness: base-trace generation (design §5.2).
//!
//! M1: delegates to the claim generator's `write_trace` — the device lanes
//! (witness JIT, count feeds, edges) engage inside it exactly as on the legacy
//! path. M2 moves ownership here: the schedule table drives the component DAG
//! and this module becomes the launch issuer.

use std::sync::Arc;

use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_cairo_prover::witness::exec_context::WitnessArtifactPlan;
use tracing::{span, Level};

use crate::plan::ProofPlan;
use crate::prover::CairoWitnessBackend;
use crate::state::{DeviceProofState, WitnessOutput};

pub fn run<B: CairoWitnessBackend>(
    generator: stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator,
    witness_artifact_plan: Arc<WitnessArtifactPlan>,
    proof_plan: Arc<ProofPlan>,
    opt_n_id_to_big_components: Option<usize>,
    pipeline_twiddles: Option<&'static TwiddleTree<B>>,
) -> WitnessOutput<B> {
    let span = span!(Level::INFO, "Write Base trace").entered();
    let device = DeviceProofState::new(witness_artifact_plan, proof_plan);
    let (trace, claim, interaction_generator) = generator.write_trace::<B>(
        &device.witness_exec_context,
        opt_n_id_to_big_components,
        pipeline_twiddles,
    );
    device.witness_exec_context.assert_witness_drained();
    span.exit();
    WitnessOutput {
        trace,
        claim,
        interaction_generator,
        device,
    }
}
