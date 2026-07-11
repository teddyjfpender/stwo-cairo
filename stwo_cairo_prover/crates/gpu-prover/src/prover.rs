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
use stwo::core::proof_of_work::GrindOps;
use stwo::core::utils::MaybeOwned;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::prover::backend::{BackendForChannel, FromSimdColumns};
use stwo::prover::mempool::BaseColumnPool;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::{
    CommitmentSchemeProver, CommitmentTreeProver, ProveExWithPcsDriverError, ProvingError,
};
use stwo_backend_cuda::{
    aot, assemble_blake2s_stark_proof, Blake2sProofAssemblyError, Blake2sProofAssemblyInput,
    CudaBackend, CudaExecTelemetry, CudaPcsDriverConfig, CudaPcsDriverError,
    CudaPcsDriverTelemetry, CudaPcsRuntimeMode, CudaRuntimeError, TranscriptMirrorReport,
};
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_prover::prover::{ChannelHash, ProverParameters};
use stwo_cairo_prover::witness::base_trace::BaseTrace;
use stwo_cairo_prover::witness::blake_g_witness_backend::BlakeGWitness;
use stwo_cairo_prover::witness::blake_round_witness_backend::BlakeRoundWitness;
use stwo_cairo_prover::witness::exec_context::WitnessArtifactPlan;
use stwo_cairo_prover::witness::jit_prove_backend::{
    Cube252Witness, OpcodeJitBackend, RecordedFlatWitness,
};
use stwo_cairo_prover::witness::memory_witness_backend::MemoryIdToBigWitness;
use stwo_cairo_prover::witness::pedersen_witness_backend::{
    PartialEcMulGenericWitness, PartialEcMulWindowBits18Witness,
    PedersenAggregatorWindowBits18Witness,
};
use stwo_cairo_prover::witness::preprocessed_trace_backend::GenPreprocessedTrace;
use stwo_cairo_prover::witness::utils::witness_trace_cells;
use stwo_constraint_framework::{FrameworkBackend, LogupFinalizeBackend};
use tracing::{span, Level};

use crate::arena_plan::ProofArenaPlan;
use crate::graphs::{GraphError, GraphWorkspace};
use crate::protocol_discovery::interaction_claim_from_flattened;
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::resident_runtime::{ResidentGraphRuntime, ResidentHotPathBudget, ResidentRuntimeError};
use crate::resident_session::{
    resident_max_domain_log_size, with_resident_session, with_resident_session_from_generator,
    ResidentExecutionReadiness, ResidentPreWitnessSessionRequest, ResidentPreparationState,
    ResidentSessionArtifacts, ResidentSessionError, ResidentSessionRequest,
    ResidentSessionTelemetry,
};
use crate::resident_witness::{planned_cairo_claim, require_strict_resident_witness_coverage};
use crate::schedule::ScheduleError;
use crate::schedule_table::CAIRO_SCHEDULE;
use crate::state::{IngestOutput, WitnessOutput};
use crate::workspace_cache::{
    WorkspaceCache, WorkspaceCacheError, WorkspaceKey, WorkspaceMaterialization,
};
use crate::{flags, phases};

/// Per-phase VRAM attribution (design §1.1 R5): when `STWO_VRAM_PHASES=1`,
/// log the pool high-water since the previous mark, then reset — the ledger
/// that ranks the diet's targets (stream-LDE was falsified by exactly this
/// kind of measurement).
fn vram_phase_mark(phase: &str) {
    if crate::flags::flag_on("STWO_VRAM_PHASES") {
        let (used, reserved) = stwo_backend_cuda::gpu_pool_highwater();
        eprintln!(
            "vram_phase[{phase}]: used_high={:.2}GB reserved_high={:.2}GB",
            used as f64 / 1e9,
            reserved as f64 / 1e9
        );
        stwo_backend_cuda::gpu_pool_highwater_reset();
    }
}

/// The witness-side backend bounds (everything `write_trace` and the interaction
/// generator require; no channel involved).
pub trait CairoWitnessBackend:
    PolyOps
    + MemoryIdToBigWitness
    + BlakeGWitness
    + OpcodeJitBackend
    + RecordedFlatWitness
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
        + RecordedFlatWitness
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
    Schedule(ScheduleError),
    Proving(ProvingError),
    PcsDriver(CudaPcsDriverError),
    Runtime(CudaRuntimeError),
    Graph(GraphError),
    WorkspaceCache(WorkspaceCacheError),
    ResidentSession(ResidentSessionError),
    ProofAssembly(Blake2sProofAssemblyError),
}

/// Machine-readable evidence from the opt-in U4 transcript migration gate.
/// `performance_admissible` is permanently false: the mirror deliberately adds
/// compact D2H reads, one synchronization and host Blake2s replay after the
/// normal resident hot-path budget has already been checked.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentTranscriptMirrorTelemetry {
    pub report: TranscriptMirrorReport,
    pub mirror_d2h_bytes: u64,
    pub mirror_sync_calls: u64,
    pub performance_admissible: bool,
}

impl ResidentTranscriptMirrorTelemetry {
    pub const fn performance_claim_admissible(&self) -> bool {
        false
    }
}

/// A correctness-only resident proof. Its distinct return type prevents a
/// mirrored run from entering the ordinary MHz benchmark path by accident.
pub struct MirroredResidentBlake2sProof {
    pub proof: CairoProof<Blake2sMerkleHasher>,
    pub transcript_mirror: ResidentTranscriptMirrorTelemetry,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentTranscriptMode {
    DeviceOnly,
    DeviceMirrored,
}

fn resident_hot_path_budget(
    mode: ResidentTranscriptMode,
    expected_graph_launches: u64,
    max_kernel_launches: u64,
    d2h_bytes: u64,
) -> ResidentHotPathBudget {
    let mut budget = ResidentHotPathBudget::final_bundle(
        expected_graph_launches,
        max_kernel_launches,
        d2h_bytes,
    );
    if mode == ResidentTranscriptMode::DeviceMirrored {
        budget.max_graph_submit_gap_ns = u64::MAX;
    }
    budget
}

fn transcript_mirror_telemetry(
    report: TranscriptMirrorReport,
    before: CudaExecTelemetry,
    after: CudaExecTelemetry,
) -> Result<ResidentTranscriptMirrorTelemetry, ResidentRuntimeError> {
    Ok(ResidentTranscriptMirrorTelemetry {
        report,
        mirror_d2h_bytes: after
            .d2h_bytes
            .checked_sub(before.d2h_bytes)
            .ok_or(ResidentRuntimeError::SizeOverflow)?,
        mirror_sync_calls: after
            .sync_calls
            .checked_sub(before.sync_calls)
            .ok_or(ResidentRuntimeError::SizeOverflow)?,
        performance_admissible: false,
    })
}

impl From<ProvingError> for GpuError {
    fn from(e: ProvingError) -> Self {
        GpuError::Proving(e)
    }
}

impl From<ScheduleError> for GpuError {
    fn from(e: ScheduleError) -> Self {
        GpuError::Schedule(e)
    }
}

impl From<CudaRuntimeError> for GpuError {
    fn from(e: CudaRuntimeError) -> Self {
        GpuError::Runtime(e)
    }
}

impl From<CudaPcsDriverError> for GpuError {
    fn from(e: CudaPcsDriverError) -> Self {
        GpuError::PcsDriver(e)
    }
}

impl From<ProveExWithPcsDriverError<CudaPcsDriverError>> for GpuError {
    fn from(e: ProveExWithPcsDriverError<CudaPcsDriverError>) -> Self {
        match e {
            ProveExWithPcsDriverError::Proving(error) => GpuError::Proving(error),
            ProveExWithPcsDriverError::PcsDriver(error) => GpuError::PcsDriver(error),
        }
    }
}

impl From<GraphError> for GpuError {
    fn from(e: GraphError) -> Self {
        GpuError::Graph(e)
    }
}

impl From<WorkspaceCacheError> for GpuError {
    fn from(e: WorkspaceCacheError) -> Self {
        GpuError::WorkspaceCache(e)
    }
}

impl From<ResidentSessionError> for GpuError {
    fn from(e: ResidentSessionError) -> Self {
        GpuError::ResidentSession(e)
    }
}

impl From<Blake2sProofAssemblyError> for GpuError {
    fn from(e: Blake2sProofAssemblyError) -> Self {
        GpuError::ProofAssembly(e)
    }
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Config(msg) => write!(f, "gpu-prover config error: {msg}"),
            GpuError::Schedule(e) => write!(f, "gpu-prover schedule error: {e}"),
            GpuError::Proving(e) => write!(f, "gpu-prover proving error: {e}"),
            GpuError::PcsDriver(e) => write!(f, "gpu-prover CUDA PCS driver error: {e}"),
            GpuError::Runtime(e) => write!(f, "gpu-prover CUDA runtime error: {e}"),
            GpuError::Graph(e) => write!(f, "gpu-prover CUDA graph error: {e}"),
            GpuError::WorkspaceCache(e) => write!(f, "gpu-prover workspace cache error: {e}"),
            GpuError::ResidentSession(e) => write!(f, "gpu-prover resident session error: {e}"),
            GpuError::ProofAssembly(e) => write!(f, "gpu-prover proof assembly error: {e}"),
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
    /// Maximum exact shape/protocol workspaces retained. Capacity exhaustion
    /// fails closed; captured graphs are never evicted implicitly.
    pub workspace_cache_capacity: usize,
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
            workspace_cache_capacity: 1,
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
pub struct GpuCairoProver<MC>
where
    MC: MerkleChannel + 'static,
    CudaBackend: CairoBackend<MC>,
{
    config: GpuProverConfig,
    /// Stable, exact-key workspaces. Each materialization owns an isolated CUDA
    /// context inside its arena; merely caching one does not activate resident
    /// execution for a proof.
    workspace_cache: WorkspaceCache,
    /// Architecture proof that the last successful gpu-native call used the
    /// concrete CUDA PCS state machine and completed every stage exactly once.
    last_pcs_telemetry: Option<CudaPcsDriverTelemetry>,
    /// Full setup/ingest evidence for the last strict resident proof. Kept
    /// separate from replay-only CUDA counters so architecture admission
    /// cannot hide a legacy writer before telemetry reset.
    last_resident_session_telemetry: Option<ResidentSessionTelemetry>,
    /// Provenance of every generated CUDA kernel lookup during the last proof.
    /// Strict mode accepts only embedded-AOT loads/hits.
    last_aot_stats: Option<aot::RuntimeStats>,
    witness_artifact_plan: Arc<WitnessArtifactPlan>,
    twiddles: HashMap<u32, &'static TwiddleTree<CudaBackend>>,
    preprocessed_trees: HashMap<u64, &'static CommitmentTreeProver<CudaBackend, MC>>,
}

impl<MC> GpuCairoProver<MC>
where
    MC: MerkleChannel + 'static,
    CudaBackend: CairoBackend<MC>,
{
    pub fn new(config: GpuProverConfig) -> Result<Self, GpuError> {
        if config.pipeline_depth != 1 {
            return Err(GpuError::Config(format!(
                "pipeline_depth {} unsupported until M6 (two-proof pipelining)",
                config.pipeline_depth
            )));
        }
        if config.strict {
            if std::env::var("STWO_CUDA_PCS_REFERENCE").as_deref() == Ok("1") {
                return Err(GpuError::Config(
                    "strict GPU-native mode rejects the migration-only PCS reference escape hatch"
                        .to_string(),
                ));
            }
            if std::env::var("STWO_CUDA_DECOMMIT_GATHER_REFERENCE").as_deref() == Ok("1") {
                return Err(GpuError::Config(
                    "strict GPU-native mode rejects the migration-only decommit gather escape hatch"
                        .to_string(),
                ));
            }
            let manifest_hash = aot::loaded_manifest_hash();
            if manifest_hash == 0 {
                return Err(GpuError::Config(
                    "strict GPU-native mode requires a non-empty embedded AOT kernel pack"
                        .to_string(),
                ));
            }
            aot::require_loaded_kernels();
        }
        let witness_artifact_plan = Arc::new(CAIRO_SCHEDULE.artifact_plan()?);
        // The gpu-native engine defaults to the composed device configuration
        // (explicit env, including =0 kill switches, always wins) — design §3.
        crate::flags::apply_gpu_native_defaults();
        let workspace_cache = WorkspaceCache::new(config.workspace_cache_capacity)?;
        Ok(Self {
            config,
            workspace_cache,
            last_pcs_telemetry: None,
            last_resident_session_telemetry: None,
            last_aot_stats: None,
            witness_artifact_plan,
            twiddles: HashMap::new(),
            preprocessed_trees: HashMap::new(),
        })
    }

    pub fn config(&self) -> &GpuProverConfig {
        &self.config
    }

    pub fn last_pcs_telemetry(&self) -> Option<&CudaPcsDriverTelemetry> {
        self.last_pcs_telemetry.as_ref()
    }

    pub fn last_resident_session_telemetry(&self) -> Option<&ResidentSessionTelemetry> {
        self.last_resident_session_telemetry.as_ref()
    }

    pub fn last_aot_stats(&self) -> Option<aot::RuntimeStats> {
        self.last_aot_stats
    }

    pub fn workspace_cache(&self) -> &WorkspaceCache {
        &self.workspace_cache
    }

    pub fn workspace_cache_mut(&mut self) -> &mut WorkspaceCache {
        &mut self.workspace_cache
    }

    /// Compatibility view for the former single-workspace API. Multi-key
    /// callers must use [`Self::graph_workspace_for`].
    pub fn graph_workspace(&self) -> Option<&GraphWorkspace> {
        self.workspace_cache.only()
    }

    pub fn graph_workspace_for(&self, key: WorkspaceKey) -> Option<&GraphWorkspace> {
        self.workspace_cache.get(key)
    }

    pub fn graph_workspace_for_mut(&mut self, key: WorkspaceKey) -> Option<&mut GraphWorkspace> {
        self.workspace_cache.get_mut(key)
    }

    /// Temporarily move the workspace out so real graph hooks may borrow its
    /// captured segments while [`Self::prove_with_pcs_driver_config`] mutably
    /// drives the prover. Reinstall it with [`Self::install_graph_workspace`].
    pub fn take_graph_workspace(&mut self) -> Option<GraphWorkspace> {
        self.workspace_cache.take_only()
    }

    pub fn take_graph_workspace_for(&mut self, key: WorkspaceKey) -> Option<GraphWorkspace> {
        self.workspace_cache.take(key)
    }

    pub fn install_graph_workspace(&mut self, workspace: GraphWorkspace) -> Result<(), GpuError> {
        self.workspace_cache.install(workspace)?;
        Ok(())
    }

    /// Materialize or reuse the exact shape/protocol workspace. This establishes
    /// stable ownership only; resident execution starts only when an explicit
    /// arena-bound PCS configuration is passed to the prove path.
    pub fn materialize_graph_workspace(
        &mut self,
        plan: Arc<ProofArenaPlan>,
    ) -> Result<(), GpuError> {
        self.materialize_or_reuse_graph_workspace(plan)?;
        Ok(())
    }

    pub fn materialize_or_reuse_graph_workspace(
        &mut self,
        plan: Arc<ProofArenaPlan>,
    ) -> Result<WorkspaceMaterialization, GpuError> {
        let (_, materialization) = self.workspace_cache.materialize_or_reuse(plan)?;
        Ok(materialization)
    }

    /// Execute the production resident hand-off while the exact cached arena is
    /// borrowed. Strict mode has no detached fallback: any discovery, staging,
    /// preparation or callback failure aborts this path.
    pub fn with_strict_resident_session<R>(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
        run: impl FnOnce(
            &mut ResidentGraphRuntime<'_>,
            ResidentSessionArtifacts<'_>,
        ) -> Result<R, ResidentRuntimeError>,
    ) -> Result<(R, ResidentSessionTelemetry), GpuError> {
        if !self.config.strict {
            return Err(GpuError::Config(
                "resident session entrypoint requires strict GPU-native mode".to_string(),
            ));
        }
        if !matches!(params.channel_hash, ChannelHash::Blake2s) {
            return Err(GpuError::Config(
                "resident device transcript currently requires the Blake2s channel".to_string(),
            ));
        }

        let IngestOutput {
            preprocessed_trace,
            generator,
            proof_plan,
        } = phases::ingest::run(
            input,
            params.preprocessed_trace,
            params.opt_n_id_to_big_components,
        );
        let exact_plan = proof_plan
            .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
            .map_err(ResidentSessionError::ProofPlan)?;
        require_strict_resident_witness_coverage(&exact_plan)
            .map_err(ResidentSessionError::ResidentWitness)?;
        let planned_claim = planned_cairo_claim(&generator, &exact_plan)
            .map_err(ResidentSessionError::ResidentWitness)?;
        let max_domain =
            resident_max_domain_log_size(&planned_claim, &preprocessed_trace, params.pcs_config)?;
        let twiddles = self.twiddle_tree(max_domain);
        Ok(with_resident_session_from_generator(
            &mut self.workspace_cache,
            ResidentPreWitnessSessionRequest {
                preprocessed_trace,
                generator,
                capacity_plan: proof_plan,
                channel_salt: params.channel_salt,
                pcs: params.pcs_config,
                include_all_preprocessed_columns: params.include_all_preprocessed_columns,
                twiddles,
            },
            run,
        )?)
    }

    /// Lower-level entrypoint for callers that already own the sealed witness
    /// output. The generated interaction state is consumed into resident lookup
    /// sources and cannot fall back to the legacy interaction writer afterwards.
    pub fn with_strict_resident_session_after_witness<R>(
        &mut self,
        preprocessed_trace: Arc<
            stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace,
        >,
        witness: WitnessOutput<CudaBackend>,
        channel_salt: u32,
        pcs: stwo::core::pcs::PcsConfig,
        include_all_preprocessed_columns: bool,
        run: impl FnOnce(
            &mut ResidentGraphRuntime<'_>,
            ResidentSessionArtifacts<'_>,
        ) -> Result<R, ResidentRuntimeError>,
    ) -> Result<(R, ResidentSessionTelemetry), GpuError> {
        if !self.config.strict {
            return Err(GpuError::Config(
                "resident session entrypoint requires strict GPU-native mode".to_string(),
            ));
        }
        let max_domain = resident_max_domain_log_size(&witness.claim, &preprocessed_trace, pcs)?;
        let twiddles = self.twiddle_tree(max_domain);
        Ok(with_resident_session(
            &mut self.workspace_cache,
            ResidentSessionRequest {
                preprocessed_trace,
                witness,
                channel_salt,
                pcs,
                include_all_preprocessed_columns,
                twiddles,
            },
            run,
        )?)
    }

    /// Prepare the strict resident architecture through runtime construction.
    /// Prepare and validate the complete resident proof runtime without
    /// launching or assembling a proof. Whole-proof callers use
    /// [`Self::prove_resident_blake2s`] after this fail-closed readiness gate.
    pub fn prepare_strict_resident(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
    ) -> Result<ResidentPreparationState, GpuError> {
        let (_, telemetry) = self.with_strict_resident_session(input, params, |_, _| Ok(()))?;
        Ok(ResidentPreparationState {
            telemetry,
            readiness: ResidentExecutionReadiness::ReadyForResidentProofReplay,
        })
    }

    /// Prove one Cairo execution. Byte-identical to `prove_cairo::<CudaBackend, MC>` on the
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
        if self.config.strict {
            return Err(GpuError::Config(
                "strict GPU-native mode requires an explicit arena-bound PCS configuration"
                    .to_string(),
            ));
        }
        // Cache residency is ownership, not execution selection. Until the
        // caller supplies arena-bound hooks, this remains the detached legacy
        // path even when one or more warm workspaces have been materialized.
        let mut pcs_driver_config = CudaPcsDriverConfig::detached_eager();
        self.prove_with_pcs_driver_config(input, params, &mut pcs_driver_config)
    }

    /// Explicit PCS-driver entry point. Resident execution constructs
    /// `CudaPcsDriverConfig::arena_graph` with real captured-segment hooks and
    /// enters here; the default [`Self::prove`] always uses detached eager mode.
    pub fn prove_with_pcs_driver_config(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
        pcs_driver_config: &mut CudaPcsDriverConfig<'_>,
    ) -> Result<CairoProof<MC::H>, GpuError> {
        if self.config.strict && pcs_driver_config.runtime_mode() != CudaPcsRuntimeMode::ArenaGraph
        {
            return Err(GpuError::Config(
                "strict GPU-native mode rejects detached PCS execution".to_string(),
            ));
        }
        self.last_pcs_telemetry = None;
        self.last_resident_session_telemetry = None;
        self.last_aot_stats = None;
        aot::reset_runtime_stats();
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

        // Streamed-LDE composition half: when the streaming leaf commit (or
        // stream_lde) is on, the committed trace evaluations are released, so
        // composition must regenerate them from coefficients (ExtendToEvalDomain)
        // rather than read the released buffers. Set before the composition phase
        // reads it; the witness/commit phases ignore this flag.
        // SAFETY: single set on the main thread before any concurrent getenv of
        // this key (composition runs later). The `!= Ok("1")` guard makes this a
        // no-op when the caller already set the var on the main thread before
        // spawning prove threads (the resident-concurrent harness does exactly
        // this) — so no set_var ever races a sibling thread's getenv.
        if (flags::flag_on("STWO_CUDA_STREAM_LEAF_COMMIT")
            || flags::flag_on("STWO_CAIRO_STREAM_LDE"))
            && std::env::var("STWO_FORCE_EXTEND_EVAL_MODE").as_deref() != Ok("1")
        {
            unsafe {
                std::env::set_var("STWO_FORCE_EXTEND_EVAL_MODE", "1");
            }
        }

        // ── Phase: ingest ────────────────────────────────────────────────────
        let IngestOutput {
            preprocessed_trace,
            generator,
            proof_plan,
        } = phases::ingest::run(
            input,
            preprocessed_trace_variant,
            opt_n_id_to_big_components,
        );

        // ── Phase: witness ───────────────────────────────────────────────────
        // Pipelined commit (M5b): on a WARM process the largest cached twiddle
        // tree is the commitment tree, so the witness phase interpolates the
        // opcode prefix AND each finished builtin lane on a committer thread
        // while later arms still generate. Cold prove (empty cache) or flag off
        // → None → the byte-identical Evals path. The commit site verifies the
        // tree by identity and fails closed if the trace size changed.
        let pipeline_twiddles = if flags::flag_on("STWO_CUDA_PIPELINED_COMMIT") {
            self.twiddles
                .iter()
                .max_by_key(|(log_size, _)| **log_size)
                .map(|(_, tree)| *tree)
        } else {
            None
        };
        let WitnessOutput {
            trace,
            claim,
            interaction_generator,
            device,
        } = phases::witness::run::<CudaBackend>(
            generator,
            Arc::clone(&self.witness_artifact_plan),
            proof_plan,
            opt_n_id_to_big_components,
            pipeline_twiddles,
        );
        vram_phase_mark("witness");

        // ── Domain sizing + persistent caches ────────────────────────────────
        let max_domain_log_size =
            phases::commit::max_domain_log_size(&claim, preprocessed_trace_variant, &pcs_config)?;
        let twiddles = self.twiddle_tree(max_domain_log_size);

        let base_column_pool = BaseColumnPool::new();
        let low_memory = flags::flag_on("STWO_CAIRO_LOW_MEMORY");
        // Streaming leaf commit (the VRAM diet) produces coeffs-retained,
        // evals-released trees, so it REQUIRES the stream_lde downstream (quotients
        // + decommit regenerate evaluations from coefficients) and store_coeffs.
        // Force both when it is on so the pieces are consistent.
        let stream_leaf_commit = flags::flag_on("STWO_CUDA_STREAM_LEAF_COMMIT");
        let stream_lde = flags::flag_on("STWO_CAIRO_STREAM_LDE") || stream_leaf_commit;
        let store_polynomials_coefficients =
            store_polynomials_coefficients || stream_leaf_commit || stream_lde;
        // Approach-B: under the diet (stream_lde) the preprocessed tree is a CACHED,
        // borrowed persistent artifact — coeffs retained, evals size-0-released, Merkle
        // layers/root owned+leaked. Its decommit routes through the coeff-regen path (the
        // Borrowed arm in the pcs compaction match), never gathering the released evals, so
        // caching it is byte-identical and removes the ~1.2s/proof rebuild. An Owned
        // per-prove rebuild is kept only for pure low_memory (no diet — needs the eval
        // release for tightest VRAM) or the STWO_DIET_REBUILD_PREPROCESSED kill switch
        // (the A/B baseline + fallback). The cache key includes max_domain_log_size so a
        // cached artifact is only ever served to a prove re-LDE'ing with the identical
        // twiddle tree that built its coeffs+Merkle (the soundness guard).
        let rebuild_owned = (low_memory && !stream_lde)
            || (stream_lde && flags::flag_on("STWO_DIET_REBUILD_PREPROCESSED"));
        let preprocessed_tree: MaybeOwned<'_, CommitmentTreeProver<CudaBackend, MC>> =
            if rebuild_owned {
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
                    max_domain_log_size,
                ))
            };

        // ── Transcript spine ─────────────────────────────────────────────────
        let channel = &mut MC::C::default();
        channel.mix_felts(&[channel_salt.into()]);
        pcs_config.mix_into(channel);
        let mut commitment_scheme = CommitmentSchemeProver::<CudaBackend, MC>::with_memory_pool(
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

        vram_phase_mark("preprocessed_tree");
        commitment_scheme.commit_tree(preprocessed_tree, channel);

        claim.mix_into::<MC>(channel);
        let span = span!(Level::INFO, "Compute base trace commitment").entered();
        let mut tree_builder = commitment_scheme.tree_builder();
        match trace {
            BaseTrace::Evals(evals) => {
                tree_builder.extend_evals(evals);
            }
            BaseTrace::Polys { polys, tree_ptr } => {
                // Byte-identity requires the committer to have interpolated with
                // THIS exact tree — verify by identity, fail closed on a
                // mid-process trace-size change (a stale tree would silently
                // fork the proof).
                assert_eq!(
                    tree_ptr, twiddles as *const TwiddleTree<CudaBackend> as usize,
                    "STWO_CUDA_PIPELINED_COMMIT: committer tree is not the commitment tree \
                     (trace size changed mid-process)"
                );
                tree_builder.extend_polys(polys);
            }
        }
        tree_builder.commit(channel);
        span.exit();
        vram_phase_mark("base_commit");

        let interaction_pow = CudaBackend::grind(channel, INTERACTION_POW_BITS);
        channel.mix_u64(interaction_pow);
        let interaction_elements = CommonLookupElements::draw(channel);

        // ── Phase: interaction ───────────────────────────────────────────────
        let (interaction_trace_evals, interaction_claim) =
            phases::interaction::run(interaction_generator, &device, &interaction_elements);
        vram_phase_mark("interaction_write");

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
        vram_phase_mark("interaction_commit");

        // ── Phase: STARK core (composition + FRI + PoW + decommit) ──────────
        let (proof, pcs_telemetry) = phases::stark::run(
            &claim,
            &interaction_elements,
            &interaction_claim,
            &preprocessed_trace,
            channel,
            commitment_scheme,
            include_all_preprocessed_columns,
            pcs_driver_config,
        )?;

        tracing::info!(
            architecture = pcs_telemetry.architecture,
            runtime_mode = ?pcs_telemetry.runtime_mode,
            stage_finished = ?pcs_telemetry.stage_finished,
            batched_tree_decommit = pcs_telemetry.batched_tree_decommit,
            "CUDA PCS architecture telemetry"
        );
        self.last_pcs_telemetry = Some(pcs_telemetry);

        let aot_stats = aot::runtime_stats();
        if self.config.strict
            && (aot_stats.aot_misses != 0
                || aot_stats.runtime_loads != 0
                || aot_stats.runtime_cache_hits != 0
                || aot_stats.strict_rejections != 0)
        {
            return Err(GpuError::Config(format!(
                "strict GPU-native AOT provenance failed: {aot_stats:?}"
            )));
        }
        tracing::info!(?aot_stats, "CUDA AOT provenance telemetry");
        self.last_aot_stats = Some(aot_stats);

        vram_phase_mark("stark_core");

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
    fn twiddle_tree(&mut self, log_size: u32) -> &'static TwiddleTree<CudaBackend> {
        let _span = span!(Level::INFO, "Precompute Twiddles").entered();
        *self.twiddles.entry(log_size).or_insert_with(|| {
            Box::leak(Box::new(CudaBackend::precompute_twiddles(
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
        twiddles: &'static TwiddleTree<CudaBackend>,
        pcs_config: &stwo::core::pcs::PcsConfig,
        store_polynomials_coefficients: bool,
        base_column_pool: &BaseColumnPool<CudaBackend>,
        max_domain_log_size: u32,
    ) -> &'static CommitmentTreeProver<CudaBackend, MC> {
        let mut hasher = DefaultHasher::new();
        for id in preprocessed_trace.ids() {
            id.id.hash(&mut hasher);
        }
        preprocessed_trace.log_sizes().hash(&mut hasher);
        pcs_config.fri_config.log_blowup_factor.hash(&mut hasher);
        pcs_config.lifting_log_size.hash(&mut hasher);
        store_polynomials_coefficients.hash(&mut hasher);
        // SOUNDNESS (approach-B): the cached artifact's coeffs + Merkle root/layers were
        // built with the twiddle tree of THIS max_domain_log_size, and decommit re-LDEs
        // from those coeffs using the same (log-size-keyed, leaked) twiddle tree. Keying on
        // it guarantees the artifact is only ever served to a prove re-LDE'ing with the
        // identical twiddle tree — a differently-sized prove is a cache MISS, never a
        // decommit against a stale root (independent of unproven cross-size CUDA twiddle
        // extraction; consistent with the tree_ptr fail-closed invariant in prove()).
        max_domain_log_size.hash(&mut hasher);
        // The diet determines the cached tree's eval state (size-0-released vs resident);
        // never serve an evals-released artifact to a non-diet prove or vice versa.
        flags::flag_on("STWO_CUDA_STREAM_LEAF_COMMIT").hash(&mut hasher);
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
        // Crash guard + invariant 1: under the diet the cached artifact MUST carry
        // coefficients (so decommit routes through the coeff-regen Borrowed arm, never the
        // size-0 resident evals that caused the approach-A illegal-address crash) with its
        // evals released. A bulk-fallback build leaving resident evals, or a coeff-less
        // column, would silently re-enter the crash path on the cache hit.
        #[cfg(debug_assertions)]
        if flags::flag_on("STWO_CUDA_STREAM_LEAF_COMMIT") {
            use stwo::prover::backend::Column;
            for poly in &tree.polynomials {
                debug_assert!(
                    poly.coeffs.is_some(),
                    "cached preprocessed artifact column missing coefficients under the diet"
                );
                debug_assert!(
                    poly.evals.values.is_empty(),
                    "cached preprocessed artifact retains resident evals under the diet"
                );
            }
        }
        let leaked: &'static CommitmentTreeProver<CudaBackend, MC> = Box::leak(Box::new(tree));
        self.preprocessed_trees.insert(key, leaked);
        leaked
    }
}

impl GpuCairoProver<Blake2sMerkleChannel> {
    /// Production Starknet resident path. Every soundness-critical prover stage
    /// runs through the sealed arena graph; the host receives one compact proof
    /// bundle and only decodes it into the existing verifier-facing type.
    pub fn prove_resident_blake2s(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
    ) -> Result<CairoProof<Blake2sMerkleHasher>, GpuError> {
        let (proof, transcript_mirror) = self.prove_resident_blake2s_with_mode(
            input,
            params,
            ResidentTranscriptMode::DeviceOnly,
        )?;
        debug_assert!(transcript_mirror.is_none());
        Ok(proof)
    }

    /// Opt-in U4 migration gate. It proves through the same strict resident
    /// graphs, first enforces the ordinary structural hot-path budget, and only
    /// then reads compact transcript snapshots back for a boundary-by-boundary
    /// host replay. Submit-gap timing is ignored because this correctness-only
    /// mode is explicitly performance-inadmissible.
    /// The distinct result and `performance_admissible=false` telemetry make
    /// this correctness run ineligible for MHz reporting by construction.
    pub fn prove_resident_blake2s_with_transcript_mirror(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
    ) -> Result<MirroredResidentBlake2sProof, GpuError> {
        let (proof, transcript_mirror) = self.prove_resident_blake2s_with_mode(
            input,
            params,
            ResidentTranscriptMode::DeviceMirrored,
        )?;
        let transcript_mirror = transcript_mirror.ok_or_else(|| {
            GpuError::Config("resident transcript mirror completed without telemetry".to_string())
        })?;
        Ok(MirroredResidentBlake2sProof {
            proof,
            transcript_mirror,
        })
    }

    fn prove_resident_blake2s_with_mode(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
        transcript_mode: ResidentTranscriptMode,
    ) -> Result<
        (
            CairoProof<Blake2sMerkleHasher>,
            Option<ResidentTranscriptMirrorTelemetry>,
        ),
        GpuError,
    > {
        if !self.config.strict {
            return Err(GpuError::Config(
                "resident Blake2s proving requires strict GPU-native mode".to_string(),
            ));
        }
        self.last_pcs_telemetry = None;
        self.last_aot_stats = None;
        aot::reset_runtime_stats();

        let ((claim, bundle, shape, exec, transcript_mirror), session_telemetry) = self
            .with_strict_resident_session(input, params, |runtime, artifacts| {
                runtime.require_prepared_witness_coverage()?;
                runtime.capture_all_prepared_subgraphs()?;
                let expected_graphs = u64::try_from(runtime.captured_graph_count())
                    .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(usize::MAX))?;
                let expected_kernel_launches = runtime.captured_graph_kernel_node_count()?;
                let bundle_bytes = runtime
                    .workspace_proof_bundle_bytes()
                    .ok_or(ResidentRuntimeError::TranscriptRequirementsMismatch)?;
                runtime.begin_hot_path_telemetry();
                runtime.replay_all_prepared_subgraphs(2)?;
                let bundle = runtime.read_proof_bundle_once()?;
                let exec = runtime.require_hot_path_budget(resident_hot_path_budget(
                    transcript_mode,
                    expected_graphs,
                    expected_kernel_launches,
                    bundle_bytes,
                ))?;
                let transcript_mirror = match transcript_mode {
                    ResidentTranscriptMode::DeviceOnly => None,
                    ResidentTranscriptMode::DeviceMirrored => {
                        // U4 correctness work is intentionally outside the hot
                        // telemetry accepted immediately above.
                        let before = runtime.hot_path_telemetry();
                        let report = runtime.verify_transcript_mirror_correctness_only()?;
                        let after = runtime.hot_path_telemetry();
                        Some(transcript_mirror_telemetry(report, before, after)?)
                    }
                };
                Ok((
                    artifacts.claim.clone(),
                    bundle,
                    runtime.proof_assembly_shape().clone(),
                    exec,
                    transcript_mirror,
                ))
            })?;
        session_telemetry.require_strict_graph_a()?;

        let interaction_claim = interaction_claim_from_flattened(&claim, &bundle.interaction_claim)
            .map_err(ResidentSessionError::Discovery)?;
        let proof = assemble_blake2s_stark_proof(Blake2sProofAssemblyInput {
            config: params.pcs_config,
            shape,
            commitments: bundle.commitments,
            sampled_values: bundle.sampled_values,
            raw_queries: bundle.decommitment.raw_queries().to_vec(),
            proof_of_work: bundle.query_pow,
            final_line_poly_words: bundle.final_line_poly_words,
            fri_commitments: bundle.fri_commitments,
            decommitment: bundle.decommitment,
        })?;

        self.last_pcs_telemetry = Some(CudaPcsDriverTelemetry::completed_arena_graph(exec));
        self.last_resident_session_telemetry = Some(session_telemetry);
        let aot_stats = aot::runtime_stats();
        if aot_stats.aot_misses != 0
            || aot_stats.runtime_loads != 0
            || aot_stats.runtime_cache_hits != 0
            || aot_stats.strict_rejections != 0
        {
            return Err(GpuError::Config(format!(
                "strict GPU-native AOT provenance failed: {aot_stats:?}"
            )));
        }
        self.last_aot_stats = Some(aot_stats);

        Ok((
            CairoProof {
                claim,
                interaction_pow: bundle.interaction_pow,
                interaction_claim,
                extended_stark_proof: proof,
                channel_salt: params.channel_salt,
                preprocessed_trace_variant: params.preprocessed_trace,
            },
            transcript_mirror,
        ))
    }
}

#[cfg(test)]
mod resident_transcript_mirror_tests {
    use stwo::core::vcs::blake2_hash::Blake2sHash;

    use super::*;

    fn report() -> TranscriptMirrorReport {
        TranscriptMirrorReport {
            protocol_key: 0x1234,
            boundaries_verified: 17,
            output_words_verified: 41,
            final_digest: Blake2sHash::default(),
            final_n_draws: 9,
        }
    }

    #[test]
    fn mirrored_run_is_explicitly_performance_inadmissible() {
        let before = CudaExecTelemetry {
            d2h_bytes: 128,
            sync_calls: 1,
            ..CudaExecTelemetry::default()
        };
        let after = CudaExecTelemetry {
            d2h_bytes: 640,
            sync_calls: 2,
            ..before
        };
        let telemetry = transcript_mirror_telemetry(report(), before, after).unwrap();
        assert_eq!(telemetry.mirror_d2h_bytes, 512);
        assert_eq!(telemetry.mirror_sync_calls, 1);
        assert!(!telemetry.performance_admissible);
        assert!(!telemetry.performance_claim_admissible());
        assert_eq!(telemetry.report.boundaries_verified, 17);
    }

    #[test]
    fn mirrored_budget_relaxes_only_submit_gap_timing() {
        let production =
            resident_hot_path_budget(ResidentTranscriptMode::DeviceOnly, 29, 7_859, 371_604);
        assert_eq!(
            production,
            ResidentHotPathBudget::final_bundle(29, 7_859, 371_604)
        );

        let mut expected_mirrored = production;
        expected_mirrored.max_graph_submit_gap_ns = u64::MAX;
        assert_eq!(
            resident_hot_path_budget(ResidentTranscriptMode::DeviceMirrored, 29, 7_859, 371_604,),
            expected_mirrored
        );
    }

    #[test]
    fn mirror_telemetry_counter_regression_fails_closed() {
        let before = CudaExecTelemetry {
            d2h_bytes: 2,
            sync_calls: 2,
            ..CudaExecTelemetry::default()
        };
        assert!(matches!(
            transcript_mirror_telemetry(report(), before, CudaExecTelemetry::default()),
            Err(ResidentRuntimeError::SizeOverflow)
        ));
    }
}
