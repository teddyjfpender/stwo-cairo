//! Commit-path helpers (design §5.4).
//!
//! M1: domain sizing + preprocessed-tree construction, mirroring the legacy path
//! byte-for-byte. The tree_builder extend/commit calls stay in `prove()` — they
//! are transcript operations. The fused commit rebuild (stage-fused NTT, twiddle
//! regen, hash-from-registers) is M4 and lands inside stwo's CUDA backend behind
//! the same conformance gates either engine consumes.

use std::sync::Arc;

use cairo_air::claims::CairoClaim;
use stwo::core::channel::MerkleChannel;
use stwo::core::pcs::utils::InvalidLiftingLogSizeError;
use stwo::core::pcs::PcsConfig;
use stwo::prover::mempool::BaseColumnPool;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::{CommitmentTreeProver, ProvingError};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedTrace, PreProcessedTraceVariant,
};
use tracing::{span, Level};

use crate::prover::CairoBackend;

/// The maximal committed-domain log size for this claim (identical formula to the
/// legacy pipeline, including the lifting-log-size validation).
pub fn max_domain_log_size(
    claim: &CairoClaim,
    variant: PreProcessedTraceVariant,
    pcs_config: &PcsConfig,
) -> Result<u32, ProvingError> {
    let max_log_trace_size = claim
        .log_sizes()
        .iter()
        .flatten()
        .fold(variant.max_log_trace_size(), |max, &size| max.max(size));

    let cairo_air_log_degree_bound = 1;
    let mut max_domain_log_size = max_log_trace_size
        + std::cmp::max(
            cairo_air_log_degree_bound,
            pcs_config.fri_config.log_blowup_factor,
        );

    if let Some(lifting_log_size) = pcs_config.lifting_log_size {
        if lifting_log_size < max_domain_log_size {
            return Err(ProvingError::InvalidLiftingLogSize(
                InvalidLiftingLogSizeError {
                    lifting_log_size,
                    min_log_size: max_domain_log_size,
                },
            ));
        }
        max_domain_log_size = lifting_log_size;
    }
    Ok(max_domain_log_size)
}

/// Build the preprocessed commitment tree (design §5.1: the columns are generated
/// by the backend — on device for CUDA — then interpolated and committed).
pub fn build_preprocessed_tree<B, MC>(
    preprocessed_trace: Arc<PreProcessedTrace>,
    twiddles: &'static TwiddleTree<B>,
    pcs_config: &PcsConfig,
    store_polynomials_coefficients: bool,
    base_column_pool: &BaseColumnPool<B>,
) -> CommitmentTreeProver<B, MC>
where
    B: CairoBackend<MC>,
    MC: MerkleChannel + 'static,
{
    let _span = span!(Level::INFO, "Compute preprocessed trace commitment").entered();
    let preprocessed_trace_polys =
        B::interpolate_columns(B::gen_preprocessed_trace(preprocessed_trace), twiddles);
    CommitmentTreeProver::<B, MC>::new(
        preprocessed_trace_polys,
        pcs_config.fri_config.log_blowup_factor,
        twiddles,
        store_polynomials_coefficients,
        pcs_config.lifting_log_size,
        base_column_pool,
    )
}
