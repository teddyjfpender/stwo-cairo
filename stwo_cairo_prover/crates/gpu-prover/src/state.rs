//! Per-proof state (design §3.2).
//!
//! The state retains the generated proof plan and witness artifacts together.
//! CUDA graph workspaces are keyed from this plan and own the stable-address
//! arena; no phase is allowed to reconstruct row geometry from host vectors.

use std::sync::Arc;

use cairo_air::claims::CairoClaim;
use stwo::prover::poly::circle::PolyOps;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_cairo_prover::witness::base_trace::BaseTrace;
use stwo_cairo_prover::witness::blake_g_witness_backend::BlakeGWitness;
use stwo_cairo_prover::witness::cairo_claim_generator::{
    CairoClaimGenerator, CairoInteractionClaimGenerator,
};
use stwo_cairo_prover::witness::exec_context::{
    ResidentWitnessPlan, WitnessArtifactPlan, WitnessExecContext,
};
use stwo_cairo_prover::witness::memory_witness_backend::MemoryIdToBigWitness;

use crate::plan::ProofPlan;
use crate::resident_input::ResidentProverInputOwner;

/// Output of the ingest phase: the preprocessed trace and the claim generator
/// (adapter output digested into per-component packed inputs).
pub struct IngestOutput {
    pub preprocessed_trace: Arc<PreProcessedTrace>,
    pub generator: CairoClaimGenerator,
    /// The generated component/relation capacity contract. It is retained for
    /// the whole proof instead of being collapsed to a shape key after ingest.
    pub proof_plan: Arc<ProofPlan>,
}

/// Generator-free ingest result owned by the ReplacementV1 session. Every
/// adapter allocation is moved once into `input`; no witness slab is rebuilt.
pub struct ReplacementIngestOutput {
    pub preprocessed_trace: Arc<PreProcessedTrace>,
    pub input: ResidentProverInputOwner,
    pub proof_plan: Arc<ProofPlan>,
}

/// Per-proof runtime state. Device witness artifacts live here from witness
/// through interaction; `proof_plan` is the allocation/scheduling contract used
/// by every later graph segment.
pub struct DeviceProofState {
    pub witness_exec_context: WitnessExecContext,
    pub proof_plan: Arc<ProofPlan>,
}

impl DeviceProofState {
    pub fn new(
        witness_artifact_plan: Arc<WitnessArtifactPlan>,
        proof_plan: Arc<ProofPlan>,
    ) -> Self {
        let planned_shape = proof_plan.proof_shape().clone();
        Self {
            witness_exec_context: WitnessExecContext::planned_with_shape(
                witness_artifact_plan,
                planned_shape,
            ),
            proof_plan,
        }
    }

    pub fn new_resident(
        witness_artifact_plan: Arc<WitnessArtifactPlan>,
        proof_plan: Arc<ProofPlan>,
        resident_witness: ResidentWitnessPlan,
    ) -> Self {
        let planned_shape = proof_plan.proof_shape().clone();
        Self {
            witness_exec_context: WitnessExecContext::planned_with_resident_witness(
                witness_artifact_plan,
                planned_shape,
                resident_witness,
            ),
            proof_plan,
        }
    }
}

/// Output of the witness phase: the base trace (device-resident columns on CUDA),
/// the claim, and the interaction generator holding the per-component lookup state.
pub struct WitnessOutput<B: PolyOps + MemoryIdToBigWitness + BlakeGWitness> {
    pub trace: BaseTrace<B>,
    pub claim: CairoClaim,
    pub interaction_generator: CairoInteractionClaimGenerator<B>,
    pub device: DeviceProofState,
}
