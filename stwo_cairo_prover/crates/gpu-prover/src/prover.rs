//! The GPU-native prover: persistent context + the prove() transcript spine
//! (design §3.2, §16.1).

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use cairo_air::claims::lookup_sum;
use cairo_air::relations::CommonLookupElements;
use cairo_air::verifier::INTERACTION_POW_BITS;
use cairo_air::CairoProof;
use num_traits::Zero;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::MaybeOwned;
use stwo::prover::backend::{BackendForChannel, FromSimdColumns};
use stwo::prover::mempool::BaseColumnPool;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::{CommitmentSchemeProver, CommitmentTreeProver, ProvingError};
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_prover::prover::ProverParameters;
use stwo_cairo_prover::witness::base_trace::BaseTrace;
use stwo_cairo_prover::witness::blake_g_witness_backend::BlakeGWitness;
use stwo_cairo_prover::witness::blake_round_witness_backend::BlakeRoundWitness;
use stwo_cairo_prover::witness::jit_prove_backend::{Cube252Witness, OpcodeJitBackend};
use stwo_cairo_prover::witness::memory_witness_backend::MemoryIdToBigWitness;
use stwo_cairo_prover::witness::pedersen_witness_backend::{
    PartialEcMulGenericWitness, PartialEcMulWindowBits18Witness,
    PedersenAggregatorWindowBits18Witness,
};
use stwo_cairo_prover::witness::preprocessed_trace_backend::GenPreprocessedTrace;
use stwo_cairo_prover::witness::utils::witness_trace_cells;
use stwo_constraint_framework::{FrameworkBackend, LogupFinalizeBackend};
use tracing::{span, Level};

use crate::state::{IngestOutput, WitnessOutput};
use crate::{flags, phases};

/// The witness-side backend bounds (everything `write_trace` and the interaction
/// generator require; no channel involved).
pub trait CairoWitnessBackend:
    PolyOps
    + MemoryIdToBigWitness
    + BlakeGWitness
    + OpcodeJitBackend
    + BlakeRoundWitness
    + Cube252Witness
    + PartialEcMulGenericWitness
    + PartialEcMulWindowBits18Witness
    + PedersenAggregatorWindowBits18Witness
{
}
impl<B> CairoWitnessBackend for B where
    B: PolyOps
        + MemoryIdToBigWitness
        + BlakeGWitness
        + OpcodeJitBackend
        + BlakeRoundWitness
        + Cube252Witness
        + PartialEcMulGenericWitness
        + PartialEcMulWindowBits18Witness
        + PedersenAggregatorWindowBits18Witness
{
}

/// The full backend contract of the pipeline (design §16.1): witness bounds plus
/// commitment/constraint/grind capabilities for the chosen Merkle channel. This is
/// the formal statement of what a backend must provide to prove Cairo — the same
/// set `prove_cairo` requires, named once.
pub trait CairoBackend<MC: MerkleChannel>:
    CairoWitnessBackend
    + BackendForChannel<MC>
    + FrameworkBackend
    + FromSimdColumns
    + LogupFinalizeBackend
    + GenPreprocessedTrace
    + 'static
{
}
impl<MC: MerkleChannel, B> CairoBackend<MC> for B where
    B: CairoWitnessBackend
        + BackendForChannel<MC>
        + FrameworkBackend
        + FromSimdColumns
        + LogupFinalizeBackend
        + GenPreprocessedTrace
        + 'static
{
}

#[derive(Debug)]
pub enum GpuError {
    /// Invalid configuration (e.g. a pipeline depth this milestone doesn't support).
    Config(String),
    Proving(ProvingError),
}

impl From<ProvingError> for GpuError {
    fn from(e: ProvingError) -> Self {
        GpuError::Proving(e)
    }
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Config(msg) => write!(f, "gpu-prover config error: {msg}"),
            GpuError::Proving(e) => write!(f, "gpu-prover proving error: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Fiat-Shamir channel placement (design §5.7). `DeviceMirrored` lands at M5
/// behind transcript byte-equality + human approval (U4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChannelMode {
    #[default]
    Host,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuProverConfig {
    /// CUDA device ordinal. Reserved: the backend currently binds the default
    /// device; multi-device selection lands with the fleet work.
    pub device: u32,
    /// VRAM ceiling driving diet mode (M4). `None` = card total.
    pub vram_budget: Option<usize>,
    /// Proofs in flight (M6 unlocks 2 with admission control, design §8).
    pub pipeline_depth: usize,
    pub channel: ChannelMode,
    /// Post-M6: no fallbacks, any device failure aborts the prove (U3).
    pub strict: bool,
}

impl Default for GpuProverConfig {
    fn default() -> Self {
        Self {
            device: 0,
            vram_budget: None,
            pipeline_depth: 1,
            channel: ChannelMode::Host,
            strict: false,
        }
    }
}

/// Persistent per-device prover context (design §3.2): caches that outlive a proof
/// — twiddle trees and preprocessed commitment trees today; the AOT kernel
/// registry (M3), graph cache and identity-slot arena (M5) join here.
///
/// Cached trees are intentionally leaked (`&'static`), matching the legacy
/// pipeline's process-global caches: bounded by the number of distinct sizes and
/// preprocessed configurations per process, shared read-only across proves.
pub struct GpuCairoProver<B, MC>
where
    B: CairoBackend<MC>,
    MC: MerkleChannel + 'static,
{
    config: GpuProverConfig,
    twiddles: HashMap<u32, &'static TwiddleTree<B>>,
    preprocessed_trees: HashMap<u64, &'static CommitmentTreeProver<B, MC>>,
}

impl<B, MC> GpuCairoProver<B, MC>
where
    B: CairoBackend<MC>,
    MC: MerkleChannel + 'static,
{
    pub fn new(config: GpuProverConfig) -> Result<Self, GpuError> {
        if config.pipeline_depth != 1 {
            return Err(GpuError::Config(format!(
                "pipeline_depth {} unsupported until M6 (two-proof pipelining)",
                config.pipeline_depth
            )));
        }
        // The gpu-native engine defaults to the composed device configuration
        // (explicit env, including =0 kill switches, always wins) — design §3.
        crate::flags::apply_gpu_native_defaults();
        Ok(Self {
            config,
            twiddles: HashMap::new(),
            preprocessed_trees: HashMap::new(),
        })
    }

    pub fn config(&self) -> &GpuProverConfig {
        &self.config
    }

    /// Prove one Cairo execution. Byte-identical to `prove_cairo::<B, MC>` on the
    /// same input and parameters — the parity gate (design §9) holds at every
    /// milestone; only WHERE and WHEN values are computed changes as the pipeline
    /// deepens.
    ///
    /// The transcript spine (every channel operation, in Fiat-Shamir order) lives
    /// in this function by design: the phase modules do the heavy lifting, this
    /// function IS the proof protocol.
    pub fn prove(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
    ) -> Result<CairoProof<MC::H>, GpuError> {
        // Same top-level span name as the legacy engine: the phase-ledger tooling
        // keys on span names; the bench record's `engine` field disambiguates.
        let _span = span!(Level::INFO, "prove_cairo").entered();
        let ProverParameters {
            channel_hash: _,
            channel_salt,
            pcs_config,
            preprocessed_trace: preprocessed_trace_variant,
            store_polynomials_coefficients,
            include_all_preprocessed_columns,
            opt_n_id_to_big_components,
        } = params;

        // ── Phase: ingest ────────────────────────────────────────────────────
        let IngestOutput {
            preprocessed_trace,
            generator,
        } = phases::ingest::run(input, preprocessed_trace_variant);

        // ── Phase: witness ───────────────────────────────────────────────────
        // M1 passes no pipelined-commit twiddles: the byte-identical Evals path.
        let WitnessOutput {
            trace,
            claim,
            interaction_generator,
        } = phases::witness::run::<B>(generator, opt_n_id_to_big_components, None);

        // ── Domain sizing + persistent caches ────────────────────────────────
        let max_domain_log_size =
            phases::commit::max_domain_log_size(&claim, preprocessed_trace_variant, &pcs_config)?;
        let twiddles = self.twiddle_tree(max_domain_log_size);

        let base_column_pool = BaseColumnPool::new();
        let low_memory = flags::flag_on("STWO_CAIRO_LOW_MEMORY");
        let stream_lde = flags::flag_on("STWO_CAIRO_STREAM_LDE");
        // Owned rebuild under the memory-diet modes (compaction wants ownership;
        // a borrowed cached tree would pin its evaluations for the whole prove),
        // cached+borrowed otherwise — the legacy semantics exactly.
        let preprocessed_tree: MaybeOwned<'_, CommitmentTreeProver<B, MC>> =
            if low_memory || stream_lde {
                MaybeOwned::Owned(phases::commit::build_preprocessed_tree(
                    preprocessed_trace.clone(),
                    twiddles,
                    &pcs_config,
                    store_polynomials_coefficients,
                    &base_column_pool,
                ))
            } else {
                MaybeOwned::Borrowed(self.preprocessed_tree(
                    &preprocessed_trace,
                    twiddles,
                    &pcs_config,
                    store_polynomials_coefficients,
                    &base_column_pool,
                ))
            };

        // ── Transcript spine ─────────────────────────────────────────────────
        let channel = &mut MC::C::default();
        channel.mix_felts(&[channel_salt.into()]);
        pcs_config.mix_into(channel);
        let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::with_memory_pool(
            pcs_config,
            twiddles,
            &base_column_pool,
        );
        if low_memory {
            commitment_scheme.set_low_memory();
        }
        if store_polynomials_coefficients {
            commitment_scheme.set_store_polynomials_coefficients();
        }
        if stream_lde {
            commitment_scheme.set_stream_lde();
        }

        commitment_scheme.commit_tree(preprocessed_tree, channel);

        claim.mix_into::<MC>(channel);
        let span = span!(Level::INFO, "Compute base trace commitment").entered();
        let mut tree_builder = commitment_scheme.tree_builder();
        match trace {
            BaseTrace::Evals(evals) => {
                tree_builder.extend_evals(evals);
            }
            BaseTrace::Polys { .. } => {
                // write_trace was called with pipeline_twiddles=None above.
                unreachable!("gpu-native M1 requested the Evals path");
            }
        }
        tree_builder.commit(channel);
        span.exit();

        let interaction_pow = B::grind(channel, INTERACTION_POW_BITS);
        channel.mix_u64(interaction_pow);
        let interaction_elements = CommonLookupElements::draw(channel);

        // ── Phase: interaction ───────────────────────────────────────────────
        let (interaction_trace_evals, interaction_claim) =
            phases::interaction::run(interaction_generator, &interaction_elements);

        tracing::info!(
            "Witness trace cells: {:?}",
            witness_trace_cells(&claim, &preprocessed_trace)
        );
        debug_assert_eq!(
            lookup_sum(&claim, &interaction_elements, &interaction_claim),
            SecureField::zero()
        );
        interaction_claim.mix_into(channel);

        let span = span!(Level::INFO, "Compute interaction trace commitment").entered();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(interaction_trace_evals);
        tree_builder.commit(channel);
        span.exit();

        // ── Phase: STARK core (composition + FRI + PoW + decommit) ──────────
        let proof = phases::stark::run(
            &claim,
            &interaction_elements,
            &interaction_claim,
            &preprocessed_trace,
            channel,
            commitment_scheme,
            include_all_preprocessed_columns,
        )?;

        Ok(CairoProof {
            claim,
            interaction_pow,
            interaction_claim,
            extended_stark_proof: proof,
            channel_salt,
            preprocessed_trace_variant,
        })
    }

    /// The twiddle tree for `log_size`, built once per prover instance and leaked
    /// (`&'static` — required by downstream borrows and the M6 committer pattern).
    fn twiddle_tree(&mut self, log_size: u32) -> &'static TwiddleTree<B> {
        let _span = span!(Level::INFO, "Precompute Twiddles").entered();
        *self.twiddles.entry(log_size).or_insert_with(|| {
            Box::leak(Box::new(B::precompute_twiddles(
                CanonicCoset::new(log_size).circle_domain().half_coset,
            )))
        })
    }

    /// The cached preprocessed commitment tree, keyed exactly like the legacy
    /// pipeline: column ids + log sizes + blowup + lifting + store-coefficients.
    fn preprocessed_tree(
        &mut self,
        preprocessed_trace: &Arc<
            stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace,
        >,
        twiddles: &'static TwiddleTree<B>,
        pcs_config: &stwo::core::pcs::PcsConfig,
        store_polynomials_coefficients: bool,
        base_column_pool: &BaseColumnPool<B>,
    ) -> &'static CommitmentTreeProver<B, MC> {
        let mut hasher = DefaultHasher::new();
        for id in preprocessed_trace.ids() {
            id.id.hash(&mut hasher);
        }
        preprocessed_trace.log_sizes().hash(&mut hasher);
        pcs_config.fri_config.log_blowup_factor.hash(&mut hasher);
        pcs_config.lifting_log_size.hash(&mut hasher);
        store_polynomials_coefficients.hash(&mut hasher);
        let key = hasher.finish();

        if let Some(tree) = self.preprocessed_trees.get(&key) {
            return tree;
        }
        let tree = phases::commit::build_preprocessed_tree(
            preprocessed_trace.clone(),
            twiddles,
            pcs_config,
            store_polynomials_coefficients,
            base_column_pool,
        );
        let leaked: &'static CommitmentTreeProver<B, MC> = Box::leak(Box::new(tree));
        self.preprocessed_trees.insert(key, leaked);
        leaked
    }
}
