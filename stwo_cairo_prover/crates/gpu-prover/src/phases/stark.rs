//! STARK core: composition + FRI + PoW + decommit (design §5.5–§5.6).
//!
//! M1: one call into stwo's `prove_ex` — the constraint lanes (JIT today, AOT at
//! M3), quotients, FRI folds, device grind and batched decommit gathers all run
//! inside it. M4/M5 take orchestration below this boundary (per-phase graphs,
//! single-graph FRI with the device channel); until then the boundary is exactly
//! the legacy pipeline's, which is what the parity gate wants.

use cairo_air::cairo_components::CairoComponents;
use cairo_air::claims::{CairoClaim, CairoInteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::channel::MerkleChannel;
use stwo::core::proof::ExtendedStarkProof;
use stwo::prover::{prove_ex, CommitmentSchemeProver, ProvingError};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_cairo_prover::utils::cairo_provers;
use tracing::{span, Level};

use crate::prover::CairoBackend;

#[allow(clippy::too_many_arguments)]
pub fn run<B, MC>(
    claim: &CairoClaim,
    interaction_elements: &CommonLookupElements,
    interaction_claim: &CairoInteractionClaim,
    preprocessed_trace: &PreProcessedTrace,
    channel: &mut MC::C,
    commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
    include_all_preprocessed_columns: bool,
) -> Result<ExtendedStarkProof<MC::H>, ProvingError>
where
    B: CairoBackend<MC>,
    MC: MerkleChannel + 'static,
{
    let component_builder = CairoComponents::new(
        claim,
        interaction_elements,
        interaction_claim,
        &preprocessed_trace.ids(),
    );
    let components = cairo_provers::<B>(&component_builder);

    let span = span!(Level::INFO, "Prove STARKs").entered();
    let proof = prove_ex::<B, _>(
        &components,
        channel,
        commitment_scheme,
        include_all_preprocessed_columns,
    )?;
    span.exit();

    tracing::event!(name: "component_info", Level::DEBUG, "Components: {}", component_builder);
    Ok(proof)
}
