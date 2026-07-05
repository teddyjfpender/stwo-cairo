//! Per-proof state (design §3.2).
//!
//! M1 scope: the typed artifacts that flow between phases — host-side handles to
//! (already device-resident, on CUDA) columns. The identity-slot arena that makes
//! this a true `DeviceProofState` (stable device addresses per capture epoch,
//! design §16.2) lands with the graphs work (M5); the phase-output types here are
//! its seam.

use std::sync::Arc;

use cairo_air::claims::CairoClaim;
use stwo::prover::poly::circle::PolyOps;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_cairo_prover::witness::base_trace::BaseTrace;
use stwo_cairo_prover::witness::blake_g_witness_backend::BlakeGWitness;
use stwo_cairo_prover::witness::cairo_claim_generator::{
    CairoClaimGenerator, CairoInteractionClaimGenerator,
};
use stwo_cairo_prover::witness::memory_witness_backend::MemoryIdToBigWitness;

/// Output of the ingest phase: the preprocessed trace and the claim generator
/// (adapter output digested into per-component packed inputs).
pub struct IngestOutput {
    pub preprocessed_trace: Arc<PreProcessedTrace>,
    pub generator: CairoClaimGenerator,
}

/// Output of the witness phase: the base trace (device-resident columns on CUDA),
/// the claim, and the interaction generator holding the per-component lookup state.
pub struct WitnessOutput<B: PolyOps + MemoryIdToBigWitness + BlakeGWitness> {
    pub trace: BaseTrace<B>,
    pub claim: CairoClaim,
    pub interaction_generator: CairoInteractionClaimGenerator<B>,
}
