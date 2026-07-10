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
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::schedule_table::CAIRO_SCHEDULE;
use crate::state::{DeviceProofState, WitnessOutput};

pub fn run<
    B: CairoWitnessBackend + stwo_cairo_prover::witness::jit_prove_backend::RecordedFlatWitness,
>(
    generator: stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator,
    witness_artifact_plan: Arc<WitnessArtifactPlan>,
    proof_plan: Arc<ProofPlan>,
    opt_n_id_to_big_components: Option<usize>,
    pipeline_twiddles: Option<&'static TwiddleTree<B>>,
) -> WitnessOutput<B> {
    let device = DeviceProofState::new(witness_artifact_plan, proof_plan);
    run_with_device(
        generator,
        device,
        opt_n_id_to_big_components,
        pipeline_twiddles,
    )
}

pub fn run_with_device<
    B: CairoWitnessBackend + stwo_cairo_prover::witness::jit_prove_backend::RecordedFlatWitness,
>(
    generator: stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator,
    mut device: DeviceProofState,
    opt_n_id_to_big_components: Option<usize>,
    pipeline_twiddles: Option<&'static TwiddleTree<B>>,
) -> WitnessOutput<B> {
    let span = span!(Level::INFO, "Write Base trace").entered();
    let (trace, claim, interaction_generator) = generator.write_trace::<B>(
        &device.witness_exec_context,
        opt_n_id_to_big_components,
        pipeline_twiddles,
    );
    device.witness_exec_context.assert_witness_drained();
    device
        .witness_exec_context
        .assert_resident_witness_complete();
    let exact_shape = device
        .witness_exec_context
        .seal_final_proof_shape()
        .expect("post-witness component shape ledger is incomplete");
    device.proof_plan = Arc::new(
        device
            .proof_plan
            .seal_exact_shape(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &exact_shape)
            .expect("post-witness component geometry violated the preplanned capacity"),
    );
    assert!(
        device.proof_plan.capture_ready(),
        "device proof plan must be exact before graph preparation"
    );
    span.exit();
    WitnessOutput {
        trace,
        claim,
        interaction_generator,
        device,
    }
}
