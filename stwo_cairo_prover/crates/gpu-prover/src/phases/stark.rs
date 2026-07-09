//! STARK core: composition + FRI + PoW + decommit (design §5.5–§5.6).
//!
//! The Cairo phase prepares composition through stwo, then hands the final PCS/FRI
//! protocol to the concrete CUDA driver. The configuration is caller-owned so a
//! materialized proof workspace can bind its arena and graph segments explicitly;
//! there is no hidden runtime context.

use cairo_air::cairo_components::CairoComponents;
use cairo_air::claims::{CairoClaim, CairoInteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::channel::MerkleChannel;
use stwo::core::proof::ExtendedStarkProof;
use stwo::prover::{prove_ex_with_pcs_driver, CommitmentSchemeProver, ProveExWithPcsDriverError};
use stwo_backend_cuda::{
    prove_cuda_pcs_values, CudaBackend, CudaPcsDriverConfig, CudaPcsDriverError,
    CudaPcsDriverTelemetry,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_cairo_prover::utils::cairo_provers;
use tracing::{span, Level};

use crate::prover::CairoBackend;

#[allow(clippy::too_many_arguments)]
pub fn run<MC>(
    claim: &CairoClaim,
    interaction_elements: &CommonLookupElements,
    interaction_claim: &CairoInteractionClaim,
    preprocessed_trace: &PreProcessedTrace,
    channel: &mut MC::C,
    commitment_scheme: CommitmentSchemeProver<'_, CudaBackend, MC>,
    include_all_preprocessed_columns: bool,
    pcs_driver_config: &mut CudaPcsDriverConfig<'_>,
) -> Result<
    (ExtendedStarkProof<MC::H>, CudaPcsDriverTelemetry),
    ProveExWithPcsDriverError<CudaPcsDriverError>,
>
where
    MC: MerkleChannel + 'static,
    CudaBackend: CairoBackend<MC>,
{
    let component_builder = CairoComponents::new(
        claim,
        interaction_elements,
        interaction_claim,
        &preprocessed_trace.ids(),
    );
    let components = cairo_provers::<CudaBackend>(&component_builder);

    let span = span!(Level::INFO, "Prove STARKs").entered();
    let output = prove_ex_with_pcs_driver(
        &components,
        channel,
        commitment_scheme,
        include_all_preprocessed_columns,
        |commitment_scheme, sample_points, channel| {
            let output = prove_cuda_pcs_values(
                commitment_scheme,
                sample_points,
                channel,
                pcs_driver_config,
            )?;
            Ok((output.proof, output.telemetry))
        },
    )?;
    span.exit();

    tracing::event!(name: "component_info", Level::DEBUG, "Components: {}", component_builder);
    Ok(output)
}
