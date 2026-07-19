//! The GPU-native prover: persistent context + the prove() transcript spine
//! (design §3.2, §16.1).

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use cairo_air::cairo_components::CairoComponents;
use cairo_air::claims::{lookup_sum, CairoClaim, CairoInteractionClaim};
use cairo_air::relations::CommonLookupElements;
use cairo_air::verifier::INTERACTION_POW_BITS;
use cairo_air::CairoProof;
use num_traits::Zero;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::circle::CirclePoint;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::{CommitmentSchemeVerifier, TreeVec};
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
    aot, assemble_blake2s_stark_proof, cuda_device_snapshot, Blake2sProofAssemblyError,
    Blake2sProofAssemblyInput, CudaBackend, CudaDeviceSnapshot, CudaExecTelemetry,
    CudaPcsDriverConfig, CudaPcsDriverError, CudaPcsDriverTelemetry, CudaPcsRuntimeMode,
    CudaRuntimeError, TranscriptMirrorReport,
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

use crate::arena_plan::ResidentBackend;
use crate::fleet_pow::FleetPowSchedule;
use crate::fleet_pow_replay::{replay_fleet_pow_split, FleetPowReplayReceipt};
use crate::fleet_pow_runtime::{FleetPowTransport, TwoRankFleetPowCoordinator};
use crate::graphs::{GraphError, GraphWorkspace, ResidentGraphTopology};
use crate::protocol_discovery::interaction_claim_from_flattened;
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::replacement_host_cache::{
    ReplacementHostCache, ReplacementHostCacheError, ReplacementHostCacheTelemetry,
};
use crate::resident_runtime::{
    ResidentGraphRuntime, ResidentHotPathBudget, ResidentRuntimeError,
    SealedResidentExecutionConfig,
};
use crate::resident_session::{
    with_resident_pre_witness_session_for_topology, with_resident_session,
    ResidentExecutionReadiness, ResidentIngressAudit, ResidentPreWitnessInput,
    ResidentPreWitnessSessionRequest, ResidentPreparationState, ResidentSessionArtifacts,
    ResidentSessionError, ResidentSessionRequest, ResidentSessionTelemetry,
};
use crate::resident_shape::RawResidentShapeError;
use crate::schedule::ScheduleError;
use crate::schedule_table::CAIRO_SCHEDULE;
use crate::shape_executable::{ShapeExecutable, ShapeExecutableCache};
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
    RawResidentShape(RawResidentShapeError),
    ReplacementHostCache(ReplacementHostCacheError),
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

/// Correctness result for the first cooperative two-rank resident path.
///
/// This result is deliberately excluded from headline benchmarking until the
/// fleet-specific copy, synchronization, and transport budgets are qualified.
pub struct FleetResidentBlake2sProof {
    pub proof: CairoProof<Blake2sMerkleHasher>,
    pub telemetry: FleetResidentProofTelemetry,
}

#[derive(Clone, Debug)]
pub struct FleetResidentProofTelemetry {
    pub proof_generation: u64,
    pub plan_identity: [u8; 32],
    pub pow: FleetPowReplayReceipt,
    pub execution: CudaExecTelemetry,
    pub expected_graph_launches: u64,
    pub expected_captured_kernel_launches: u64,
    pub performance_admissible: bool,
}

impl FleetResidentProofTelemetry {
    pub const fn performance_claim_admissible(&self) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentTranscriptMode {
    DeviceOnly,
    DeviceMirrored,
}

fn resident_hot_path_budget(
    mode: ResidentTranscriptMode,
    allow_slow_graph_submit_diagnostic: bool,
    expected_graph_launches: u64,
    expected_kernel_launches: u64,
    d2h_bytes: u64,
) -> ResidentHotPathBudget {
    let mut budget = ResidentHotPathBudget::final_bundle(
        expected_graph_launches,
        expected_kernel_launches,
        d2h_bytes,
    );
    if mode == ResidentTranscriptMode::DeviceMirrored || allow_slow_graph_submit_diagnostic {
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

fn require_composition_oods_consistency(
    oods_point: CirclePoint<SecureField>,
    max_log_degree_bound: u32,
    sampled_values: &TreeVec<Vec<Vec<SecureField>>>,
    evaluate_from_trace: impl FnOnce(&TreeVec<Vec<Vec<SecureField>>>) -> SecureField,
) -> Result<(), GpuError> {
    match stwo::core::proof::validate_composition_oods(
        sampled_values,
        oods_point,
        max_log_degree_bound,
        || evaluate_from_trace(sampled_values),
    ) {
        Ok(()) => Ok(()),
        Err(stwo::core::proof::CompositionOodsValidationError::InvalidStructure) => Err(
            GpuError::Config("malformed composition OODS opening".to_string()),
        ),
        Err(stwo::core::proof::CompositionOodsValidationError::Mismatch) => {
            Err(GpuError::Proving(ProvingError::ConstraintsNotSatisfied))
        }
    }
}

fn validate_resident_composition_oods(
    claim: &CairoClaim,
    interaction_claim: &CairoInteractionClaim,
    interaction_pow: u64,
    proof: &stwo::core::proof::ExtendedStarkProof<Blake2sMerkleHasher>,
    params: &ProverParameters,
    lifting_log_size: u32,
) -> Result<(), GpuError> {
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    channel.mix_felts(&[params.channel_salt.into()]);
    params.pcs_config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(params.pcs_config);
    let preprocessed_trace = params.preprocessed_trace.to_preprocessed_trace();
    let mut log_sizes = claim.log_sizes();
    log_sizes.insert(0, preprocessed_trace.log_sizes());
    let stark_proof = &proof.proof.0;
    let [preprocessed_root, base_root, interaction_root, composition_root] =
        stark_proof.commitments.as_slice()
    else {
        return Err(GpuError::Config(
            "malformed resident commitment root structure".to_string(),
        ));
    };
    let [preprocessed_logs, base_logs, interaction_logs] = log_sizes.as_slice() else {
        return Err(GpuError::Config(
            "malformed resident trace log-size structure".to_string(),
        ));
    };
    commitment_scheme.commit(*preprocessed_root, preprocessed_logs, channel);
    claim.mix_into::<Blake2sMerkleChannel>(channel);
    commitment_scheme.commit(*base_root, base_logs, channel);
    channel.mix_u64(interaction_pow);
    let interaction_elements = CommonLookupElements::draw(channel);
    interaction_claim.mix_into(channel);
    commitment_scheme.commit(*interaction_root, interaction_logs, channel);

    let component_generator = CairoComponents::new(
        claim,
        &interaction_elements,
        interaction_claim,
        &preprocessed_trace.ids(),
    );
    let components = stwo::core::air::Components {
        components: component_generator.components(),
        n_preprocessed_columns: preprocessed_logs.len(),
    };
    let max_log_degree_bound = lifting_log_size
        .checked_sub(params.pcs_config.fri_config.log_blowup_factor)
        .ok_or(ProvingError::ConstraintsNotSatisfied)?;
    let random_coefficient = channel.draw_secure_felt();
    commitment_scheme.commit(
        *composition_root,
        &[max_log_degree_bound; 2 * SECURE_EXTENSION_DEGREE],
        channel,
    );
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
    let mut trace_evaluation = None;
    let validation = require_composition_oods_consistency(
        oods_point,
        max_log_degree_bound,
        &stark_proof.sampled_values,
        |sampled_values| {
            let evaluation = components.eval_composition_polynomial_at_point(
                oods_point,
                sampled_values,
                random_coefficient,
                max_log_degree_bound,
            );
            trace_evaluation = Some(evaluation);
            evaluation
        },
    );
    if validation.is_err() && flags::flag_on("STWO_RESIDENT_OODS_DIAGNOSTIC") {
        let topology = stark_proof
            .sampled_values
            .iter()
            .map(|tree| {
                (
                    tree.len(),
                    tree.iter().map(|column| column.len()).sum::<usize>(),
                )
            })
            .collect::<Vec<_>>();
        eprintln!(
            "resident_oods_diagnostic: point={oods_point:?} random_coefficient={random_coefficient:?} max_log_degree_bound={max_log_degree_bound} trace_evaluation={trace_evaluation:?} composition_mask_coordinates={:?} topology(columns,total_samples)={topology:?}",
            stark_proof.sampled_values.last(),
        );
    }
    validation
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

impl From<RawResidentShapeError> for GpuError {
    fn from(e: RawResidentShapeError) -> Self {
        GpuError::RawResidentShape(e)
    }
}

impl From<ReplacementHostCacheError> for GpuError {
    fn from(e: ReplacementHostCacheError) -> Self {
        GpuError::ReplacementHostCache(e)
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
            GpuError::RawResidentShape(e) => write!(f, "gpu-prover raw resident shape error: {e}"),
            GpuError::ReplacementHostCache(e) => {
                write!(f, "gpu-prover replacement host cache error: {e}")
            }
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
    /// Logical CUDA device ordinal. Replacement-v1 currently requires one
    /// visible device and therefore enforces ordinal zero.
    pub device: u32,
    /// VRAM ceiling driving diet mode (M4). `None` = card total.
    pub vram_budget: Option<usize>,
    /// Proofs in flight (M6 unlocks 2 with admission control, design §8).
    pub pipeline_depth: usize,
    /// Maximum exact shape/protocol workspaces retained. Capacity exhaustion
    /// fails closed; captured graphs are never evicted implicitly.
    pub workspace_cache_capacity: usize,
    /// Deployment-owned headroom reserved in addition to measured allocations.
    /// `None` keeps physical admission incomplete; callers must opt into an
    /// explicit non-zero policy rather than infer one from remaining VRAM.
    pub operational_safety_reserve_bytes: Option<core::num::NonZeroUsize>,
    pub channel: ChannelMode,
    /// Immutable resident implementation generation. The replacement is
    /// admitted only through strict, no-fallback execution.
    pub resident_backend: ResidentBackend,
    /// Post-M6: no fallbacks, any device failure aborts the prove (U3).
    pub strict: bool,
    /// Temporary relaxation of the host graph-submit gap abort. Diagnostic runs
    /// are non-formal; capture runs may be re-admitted only after the observed
    /// strict gap gate passes. Structural/copy/synchronization budgets stay exact.
    pub allow_slow_graph_submit_diagnostic: bool,
    /// Record per-graph CUDA-event intervals for a diagnostic run. This is
    /// deliberately separate from the soft graph-gap capture used by formal
    /// timing runs, which must remain free of event instrumentation.
    pub record_graph_replay_intervals_diagnostic: bool,
    /// Correctness-only vertical checkpoint for the compiled Composition
    /// authority. The ordinary captured replay remains the default; this lane
    /// is never eligible for a formal performance claim.
    pub compiled_composition_vertical_checkpoint: bool,
}

impl Default for GpuProverConfig {
    fn default() -> Self {
        Self {
            device: 0,
            vram_budget: None,
            pipeline_depth: 1,
            workspace_cache_capacity: 1,
            operational_safety_reserve_bytes: None,
            channel: ChannelMode::Host,
            resident_backend: ResidentBackend::LegacyResident,
            strict: false,
            allow_slow_graph_submit_diagnostic: false,
            record_graph_replay_intervals_diagnostic: false,
            compiled_composition_vertical_checkpoint: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentProofExecution {
    CapturedGraph,
    CompiledCompositionVerticalCheckpoint,
}

impl ResidentProofExecution {
    const fn from_config(config: &GpuProverConfig) -> Self {
        if config.compiled_composition_vertical_checkpoint {
            Self::CompiledCompositionVerticalCheckpoint
        } else {
            Self::CapturedGraph
        }
    }
}

fn validate_resident_proof_execution_config(config: &GpuProverConfig) -> Result<(), GpuError> {
    if !config.compiled_composition_vertical_checkpoint {
        return Ok(());
    }
    if config.resident_backend != ResidentBackend::ReplacementV1 || !config.strict {
        return Err(GpuError::Config(
            "compiled Composition vertical checkpoint requires strict replacement-v1".to_string(),
        ));
    }
    if config.allow_slow_graph_submit_diagnostic || config.record_graph_replay_intervals_diagnostic
    {
        return Err(GpuError::Config(
            "compiled Composition vertical checkpoint cannot be combined with graph-submit diagnostics"
                .to_string(),
        ));
    }
    Ok(())
}

const REPLACEMENT_V1_REQUIRED_ENV: &[(&str, &str)] = &[
    ("STWO_CUDA_WITNESS_JIT_PROVE", "1"),
    ("STWO_CUDA_WITNESS_JIT_MAX_INSTRS", "20000"),
    ("STWO_CUDA_DEVICE_INTERACTION", "1"),
    ("STWO_CUDA_WITNESS_EDGES", "1"),
    ("STWO_CUDA_MEM_COUNT_FEEDS", "1"),
    ("STWO_CUDA_STREAM_FANOUT", "1"),
];

const REPLACEMENT_V1_DEFAULT_OFF_ENV: &[&str] = &[
    "STWO_CAIRO_LOW_MEMORY",
    "STWO_CAIRO_STREAM_LDE",
    "STWO_CUDA_STREAM_LEAF_COMMIT",
    "STWO_CUDA_PIPELINED_COMMIT",
    "STWO_DIET_REBUILD_PREPROCESSED",
    "STWO_FORCE_EXTEND_EVAL_MODE",
    "STWO_STORE_COEFFS",
];

const REPLACEMENT_V1_FORBIDDEN_ENV: &[&str] = &[
    "PREPROCESSED_TRACE_GPU_GENERATE",
    "STWO_CUDA_RETAINED_LDE_BUDGET_BYTES",
    "STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS",
    "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE",
    "STWO_CUDA_COMPOSITION_DIRECT_RETENTION",
    "STWO_CUDA_B2N_STAGE_FUSED",
    "STWO_CUDA_BLAKE2S_INTERIOR_FUSED",
    "STWO_CUDA_COMPOSITION_WIDE",
    "STWO_CUDA_RELATION_SCAN_TAIL",
    "STWO_CUDA_FRI_FOLD_FUSED",
    "STWO_CUDA_FEED_PRIVATIZED",
];

fn validate_replacement_fixed_env_value(
    name: &str,
    observed: Option<&str>,
    expected: &str,
) -> Result<(), GpuError> {
    if let Some(observed) = observed {
        if observed != expected {
            return Err(GpuError::Config(format!(
                "replacement-v1 requires {name} to be unset or exactly {expected}, got {observed:?}"
            )));
        }
    }
    Ok(())
}

fn validate_replacement_fixed_environment() -> Result<(), GpuError> {
    if flags::GPU_NATIVE_DEFAULTS != REPLACEMENT_V1_REQUIRED_ENV {
        return Err(GpuError::Config(
            "replacement-v1 required environment no longer matches GPU_NATIVE_DEFAULTS".to_string(),
        ));
    }
    for &(name, expected) in REPLACEMENT_V1_REQUIRED_ENV {
        validate_replacement_fixed_environment_value(name, expected)?;
    }
    for &name in REPLACEMENT_V1_DEFAULT_OFF_ENV {
        validate_replacement_fixed_environment_value(name, "0")?;
    }
    Ok(())
}

fn validate_replacement_fixed_environment_value(
    name: &str,
    expected: &str,
) -> Result<(), GpuError> {
    match std::env::var(name) {
        Ok(observed) => validate_replacement_fixed_env_value(name, Some(&observed), expected),
        Err(std::env::VarError::NotPresent) => {
            validate_replacement_fixed_env_value(name, None, expected)
        }
        Err(std::env::VarError::NotUnicode(_)) => Err(GpuError::Config(format!(
            "replacement-v1 requires {name} to be unset or valid UTF-8 equal to {expected}"
        ))),
    }
}

fn validate_replacement_forbidden_env_value(name: &str, present: bool) -> Result<(), GpuError> {
    if present {
        return Err(GpuError::Config(format!(
            "replacement-v1 rejects legacy topology override {name}; unset it and use the immutable backend selector"
        )));
    }
    Ok(())
}

fn replacement_execution_config_from_environment(
) -> Result<SealedResidentExecutionConfig, GpuError> {
    validate_replacement_fixed_environment()?;
    for &name in REPLACEMENT_V1_FORBIDDEN_ENV {
        validate_replacement_forbidden_env_value(name, std::env::var_os(name).is_some())?;
    }
    Ok(SealedResidentExecutionConfig::replacement_v1())
}

fn packed_numerator_measurement_policy(
    loaded: ProtocolPlanPolicy,
) -> Result<ProtocolPlanPolicy, GpuError> {
    let expected = ProtocolPlanPolicy::replacement_v1(
        loaded.kernel_manifest_hash,
        loaded.composition_max_kernel_instrs,
    );
    if loaded != expected {
        return Err(GpuError::Config(
            "packed numerator measurement control requires the exact loaded replacement-v1 tuple"
                .to_string(),
        ));
    }
    Ok(
        ProtocolPlanPolicy::replacement_v1_packed_numerator_measurement_control(
            loaded.kernel_manifest_hash,
            loaded.composition_max_kernel_instrs,
        ),
    )
}

fn validate_replacement_device_admission(
    configured: u32,
    snapshot: CudaDeviceSnapshot,
    aot_arch_supported: bool,
) -> Result<(), GpuError> {
    if snapshot.count != 1 {
        return Err(GpuError::Config(format!(
            "replacement-v1 requires exactly one CUDA-visible device, found {}",
            snapshot.count
        )));
    }
    if configured != 0 || snapshot.current != 0 {
        return Err(GpuError::Config(format!(
            "replacement-v1 requires configured and current CUDA ordinals to both be 0, got configured={configured} current={}",
            snapshot.current
        )));
    }
    if !aot_arch_supported {
        return Err(GpuError::Config(format!(
            "replacement-v1 AOT pack does not support sm_{}{}",
            snapshot.sm_major, snapshot.sm_minor
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProverExecutionEntry {
    PreWitnessResident,
    AfterWitnessResident,
    LegacyPcs,
}

fn validate_execution_entry(
    backend: ResidentBackend,
    entry: ProverExecutionEntry,
) -> Result<(), GpuError> {
    if backend == ResidentBackend::ReplacementV1
        && entry != ProverExecutionEntry::PreWitnessResident
    {
        return Err(GpuError::Config(
            "replacement-v1 executes only through the pre-witness strict resident entrypoint"
                .to_string(),
        ));
    }
    Ok(())
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
    resident_execution_config: SealedResidentExecutionConfig,
    /// Stable, exact-key workspaces. Each materialization owns an isolated CUDA
    /// context inside its arena; merely caching one does not activate resident
    /// execution for a proof.
    workspace_cache: WorkspaceCache,
    /// Persistent typed host executable cache. Workspace reuse is admitted
    /// only when this cache supplies the exact topology and arena identity.
    shape_executable_cache: ShapeExecutableCache,
    /// Replacement-only immutable plan/claim/lane cache. LegacyResident never
    /// constructs or consults it.
    replacement_host_cache: Option<ReplacementHostCache>,
    /// Architecture proof that the last successful gpu-native call used the
    /// concrete CUDA PCS state machine and completed every stage exactly once.
    last_pcs_telemetry: Option<CudaPcsDriverTelemetry>,
    /// Full setup/ingest evidence for the last strict resident proof. Kept
    /// separate from replay-only CUDA counters so architecture admission
    /// cannot hide a legacy writer before telemetry reset.
    last_resident_session_telemetry: Option<ResidentSessionTelemetry>,
    /// Exact topology contract resolved once, before any shape is compiled.
    /// Non-strict execution does not enter the resident selector.
    resident_protocol_policy: Option<ProtocolPlanPolicy>,
    /// Provenance of every generated CUDA kernel lookup during the last proof.
    /// Strict mode accepts only embedded-AOT loads/hits.
    last_aot_stats: Option<aot::RuntimeStats>,
    /// A captured workspace has one exact graph topology. Reject mode changes
    /// before leasing it so a comparison run cannot poison the cache.
    resident_graph_topology: Option<ResidentGraphTopology>,
    witness_artifact_plan: Arc<WitnessArtifactPlan>,
    twiddles: HashMap<u32, &'static TwiddleTree<CudaBackend>>,
    preprocessed_trees: HashMap<u64, &'static CommitmentTreeProver<CudaBackend, MC>>,
}

/// Fully validated constructor state with no semantic mode, environment, or
/// AOT latch committed. Checked default-pool admission may already have cached
/// a retryable process resource; `commit` is deliberately infallible and is the
/// only point that changes proof-selection state.
struct PendingGpuCairoProver {
    config: GpuProverConfig,
    resident_execution_config: SealedResidentExecutionConfig,
    workspace_cache: WorkspaceCache,
    shape_executable_cache: ShapeExecutableCache,
    replacement_host_cache: Option<ReplacementHostCache>,
    resident_protocol_policy: Option<ProtocolPlanPolicy>,
    witness_artifact_plan: Arc<WitnessArtifactPlan>,
}

pub(crate) struct PreparedResidentIngest {
    pub preprocessed_trace:
        Arc<stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace>,
    pub input: ResidentPreWitnessInput,
    pub audit: ResidentIngressAudit,
}

/// The sole production backend dispatch before a resident session. The input
/// is consumed exactly once; the replacement arm cannot name or construct a
/// `CairoClaimGenerator`.
pub(crate) fn prepare_resident_ingest(
    backend: ResidentBackend,
    replacement_host_cache: Option<&mut ReplacementHostCache>,
    input: ProverInput,
    variant: stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant,
    opt_n_id_to_big_components: Option<usize>,
) -> Result<PreparedResidentIngest, GpuError> {
    let start = Instant::now();
    let before = stwo_cairo_prover::witness::cairo::claim_generator_constructions();
    let mut replacement_host_cache_audit = None;
    let (preprocessed_trace, input) = match backend {
        ResidentBackend::LegacyResident => {
            let IngestOutput {
                preprocessed_trace,
                generator,
                proof_plan,
            } = phases::ingest::run(input, variant, opt_n_id_to_big_components);
            (
                preprocessed_trace,
                ResidentPreWitnessInput::LegacyResident {
                    generator,
                    capacity_plan: proof_plan,
                },
            )
        }
        ResidentBackend::ReplacementV1 => {
            let cache = replacement_host_cache.ok_or_else(|| {
                GpuError::Config("ReplacementV1 host cache was not constructed".to_owned())
            })?;
            let input = phases::ingest::encode_replacement(input);
            let selection =
                cache.compile_or_bind(&input, variant, opt_n_id_to_big_components)?;
            let preprocessed_trace = Arc::clone(selection.template.preprocessed_trace());
            replacement_host_cache_audit = Some(selection.audit);
            (
                preprocessed_trace,
                ResidentPreWitnessInput::ReplacementV1 {
                    input,
                    template: selection.template,
                },
            )
        }
    };
    let claim_generator_constructions =
        stwo_cairo_prover::witness::cairo::claim_generator_constructions().saturating_sub(before);
    if backend == ResidentBackend::ReplacementV1 && claim_generator_constructions != 0 {
        return Err(GpuError::Config(
            "ReplacementV1 ingest constructed a CairoClaimGenerator".to_owned(),
        ));
    }
    Ok(PreparedResidentIngest {
        preprocessed_trace,
        input,
        audit: ResidentIngressAudit {
            ingest_ns: start.elapsed().as_nanos(),
            claim_generator_constructions,
            replacement_host_cache: replacement_host_cache_audit,
        },
    })
}

impl PendingGpuCairoProver {
    fn commit<MC>(self) -> GpuCairoProver<MC>
    where
        MC: MerkleChannel + 'static,
        CudaBackend: CairoBackend<MC>,
    {
        if self.config.strict {
            aot::require_loaded_kernels();
        }
        if self.config.resident_backend == ResidentBackend::LegacyResident {
            crate::flags::apply_gpu_native_defaults();
        }
        GpuCairoProver {
            config: self.config,
            resident_execution_config: self.resident_execution_config,
            workspace_cache: self.workspace_cache,
            shape_executable_cache: self.shape_executable_cache,
            replacement_host_cache: self.replacement_host_cache,
            last_pcs_telemetry: None,
            last_resident_session_telemetry: None,
            resident_protocol_policy: self.resident_protocol_policy,
            last_aot_stats: None,
            resident_graph_topology: None,
            witness_artifact_plan: self.witness_artifact_plan,
            twiddles: HashMap::new(),
            preprocessed_trees: HashMap::new(),
        }
    }
}

impl<MC> GpuCairoProver<MC>
where
    MC: MerkleChannel + 'static,
    CudaBackend: CairoBackend<MC>,
{
    /// Construct the exact packed numerator baseline used for replacement-v1
    /// A/B measurements. The normal constructor and production selector remain
    /// group-direct; this path accepts no caller-supplied policy.
    pub fn new_packed_numerator_measurement_control(
        config: GpuProverConfig,
    ) -> Result<Self, GpuError> {
        if config.resident_backend != ResidentBackend::ReplacementV1 || !config.strict {
            return Err(GpuError::Config(
                "packed numerator measurement control requires strict replacement-v1".to_string(),
            ));
        }
        let mut prover = Self::new(config)?;
        let loaded = prover.resident_protocol_policy.ok_or_else(|| {
            GpuError::Config(
                "packed numerator measurement control requires a loaded replacement policy"
                    .to_string(),
            )
        })?;
        prover.resident_protocol_policy = Some(packed_numerator_measurement_policy(loaded)?);
        Ok(prover)
    }

    pub fn new(config: GpuProverConfig) -> Result<Self, GpuError> {
        if config.pipeline_depth != 1 {
            return Err(GpuError::Config(format!(
                "pipeline_depth {} unsupported until M6 (two-proof pipelining)",
                config.pipeline_depth
            )));
        }
        if config.resident_backend == ResidentBackend::ReplacementV1 && !config.strict {
            return Err(GpuError::Config(
                "replacement-v1 requires strict GPU-native resident execution".to_string(),
            ));
        }
        if config.record_graph_replay_intervals_diagnostic
            && (!config.strict || !config.allow_slow_graph_submit_diagnostic)
        {
            return Err(GpuError::Config(
                "graph replay interval timing requires strict slow-submit diagnostic mode"
                    .to_string(),
            ));
        }
        validate_resident_proof_execution_config(&config)?;
        let resident_execution_config = match config.resident_backend {
            ResidentBackend::LegacyResident => SealedResidentExecutionConfig::legacy_strict(),
            ResidentBackend::ReplacementV1 => replacement_execution_config_from_environment()?,
        };
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
        }
        if config.resident_backend == ResidentBackend::ReplacementV1 {
            let device = cuda_device_snapshot()?;
            validate_replacement_device_admission(
                config.device,
                device,
                aot::supports_arch(device.sm_major, device.sm_minor),
            )?;
            stwo_backend_cuda::ensure_gpu_default_pool()?;
        }
        let witness_artifact_plan = Arc::new(CAIRO_SCHEDULE.artifact_plan()?);
        let resident_protocol_policy = config
            .strict
            .then(|| ProtocolPlanPolicy::loaded_starknet_blake2s_for(config.resident_backend))
            .transpose()
            .map_err(ResidentSessionError::from)?;
        let workspace_cache = WorkspaceCache::new(config.workspace_cache_capacity)?;
        let shape_executable_cache = ShapeExecutableCache::new(config.workspace_cache_capacity)
            .map_err(ResidentSessionError::from)?;
        let replacement_host_cache = (config.resident_backend == ResidentBackend::ReplacementV1)
            .then(|| ReplacementHostCache::new(config.workspace_cache_capacity))
            .transpose()?;
        Ok(PendingGpuCairoProver {
            config,
            resident_execution_config,
            workspace_cache,
            shape_executable_cache,
            replacement_host_cache,
            resident_protocol_policy,
            witness_artifact_plan,
        }
        .commit())
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

    pub fn shape_executable_cache(&self) -> &ShapeExecutableCache {
        &self.shape_executable_cache
    }

    pub fn replacement_host_cache_telemetry(&self) -> Option<ReplacementHostCacheTelemetry> {
        self.replacement_host_cache
            .as_ref()
            .map(ReplacementHostCache::telemetry)
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
        executable: &ShapeExecutable,
    ) -> Result<(), GpuError> {
        self.materialize_or_reuse_graph_workspace(executable)?;
        Ok(())
    }

    pub fn materialize_or_reuse_graph_workspace(
        &mut self,
        executable: &ShapeExecutable,
    ) -> Result<WorkspaceMaterialization, GpuError> {
        let (_, materialization) = self.workspace_cache.materialize_or_reuse(executable)?;
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
        self.with_strict_resident_session_for_topology(
            input,
            params,
            ResidentGraphTopology::Monolithic,
            run,
        )
    }

    fn with_strict_resident_session_for_topology<R>(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
        graph_topology: ResidentGraphTopology,
        run: impl FnOnce(
            &mut ResidentGraphRuntime<'_>,
            ResidentSessionArtifacts<'_>,
        ) -> Result<R, ResidentRuntimeError>,
    ) -> Result<(R, ResidentSessionTelemetry), GpuError> {
        match self.resident_graph_topology {
            None => self.resident_graph_topology = Some(graph_topology),
            Some(current) if current == graph_topology => {}
            Some(current) => {
                return Err(GpuError::Config(format!(
                    "resident prover graph topology is sealed as {current:?}, requested \
                     {graph_topology:?}; use a separate prover instance for parity comparison"
                )));
            }
        }
        validate_execution_entry(
            self.config.resident_backend,
            ProverExecutionEntry::PreWitnessResident,
        )?;
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

        let PreparedResidentIngest {
            preprocessed_trace,
            input,
            audit,
        } = prepare_resident_ingest(
            self.config.resident_backend,
            self.replacement_host_cache.as_mut(),
            input,
            params.preprocessed_trace,
            params.opt_n_id_to_big_components,
        )?;
        let protocol_policy = self.resident_protocol_policy.ok_or_else(|| {
            GpuError::Config("strict resident protocol policy was not resolved".to_string())
        })?;
        Ok(with_resident_pre_witness_session_for_topology(
            &mut self.shape_executable_cache,
            &mut self.workspace_cache,
            ResidentPreWitnessSessionRequest {
                preprocessed_trace,
                input,
                ingress_audit: audit,
                channel_salt: params.channel_salt,
                pcs: params.pcs_config,
                include_all_preprocessed_columns: params.include_all_preprocessed_columns,
                operational_safety_reserve_bytes: self.config.operational_safety_reserve_bytes,
                protocol_policy,
                execution_config: self.resident_execution_config,
            },
            graph_topology,
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
        validate_execution_entry(
            self.config.resident_backend,
            ProverExecutionEntry::AfterWitnessResident,
        )?;
        if !self.config.strict {
            return Err(GpuError::Config(
                "resident session entrypoint requires strict GPU-native mode".to_string(),
            ));
        }
        let protocol_policy = self.resident_protocol_policy.ok_or_else(|| {
            GpuError::Config("strict resident protocol policy was not resolved".to_string())
        })?;
        Ok(with_resident_session(
            &mut self.shape_executable_cache,
            &mut self.workspace_cache,
            ResidentSessionRequest {
                preprocessed_trace,
                witness,
                channel_salt,
                pcs,
                include_all_preprocessed_columns,
                operational_safety_reserve_bytes: self.config.operational_safety_reserve_bytes,
                protocol_policy,
                execution_config: self.resident_execution_config,
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
        let (_, telemetry) = self.with_strict_resident_session(input, params, |runtime, _| {
            if !runtime.prepared_capture_ready()? {
                runtime.capture_all_prepared_subgraphs()?;
            }
            runtime.require_complete_captured_topology()?;
            Ok(())
        })?;
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

    /// Legacy migration PCS-driver entry point. Replacement-v1 has a separate
    /// sealed whole-proof runtime and is rejected here before any env-driven
    /// witness choice can execute.
    pub fn prove_with_pcs_driver_config(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
        pcs_driver_config: &mut CudaPcsDriverConfig<'_>,
    ) -> Result<CairoProof<MC::H>, GpuError> {
        validate_execution_entry(
            self.config.resident_backend,
            ProverExecutionEntry::LegacyPcs,
        )?;
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
    /// Production Starknet resident path. The default execution runs every
    /// soundness-critical stage through the sealed arena graph. The explicit
    /// compiled-Composition checkpoint selector instead runs the non-formal
    /// eager vertical; both routes return one verifier-facing proof bundle.
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

    /// Prove through the real resident DAG while ranks 0 and 1 cooperatively
    /// search both transcript PoW boundaries.
    pub fn prove_resident_blake2s_with_fleet_pow<T: FleetPowTransport>(
        &mut self,
        input: ProverInput,
        params: ProverParameters,
        schedule: FleetPowSchedule,
        proof_generation: u64,
        transport: &mut T,
    ) -> Result<FleetResidentBlake2sProof, GpuError> {
        if self.config.record_graph_replay_intervals_diagnostic {
            return Err(GpuError::Config(
                "fleet PoW replay timing is not yet topology-qualified".to_string(),
            ));
        }
        self.last_pcs_telemetry = None;
        self.last_resident_session_telemetry = None;
        self.last_aot_stats = None;
        if !self.config.strict {
            return Err(GpuError::Config(
                "resident Blake2s proving requires strict GPU-native mode".to_string(),
            ));
        }
        aot::reset_runtime_stats();
        let capture_graph_replay_timing = self.config.record_graph_replay_intervals_diagnostic;

        let (
            (
                claim,
                bundle,
                shape,
                lifting_log_size,
                plan_identity,
                pow,
                execution,
                expected_graphs,
                expected_kernel_launches,
            ),
            session_telemetry,
        ) = self.with_strict_resident_session_for_topology(
            input,
            params,
            ResidentGraphTopology::FleetPowSplit,
            |runtime, artifacts| {
                runtime.require_prepared_witness_coverage()?;
                if artifacts
                    .telemetry
                    .prepared_runtime_materialization
                    .is_none()
                {
                    return Err(ResidentRuntimeError::MissingPreparedRuntimeMaterialization);
                }
                if !runtime.prepared_capture_ready_for(ResidentGraphTopology::FleetPowSplit)? {
                    runtime
                        .capture_all_prepared_subgraphs_for(ResidentGraphTopology::FleetPowSplit)?;
                }
                let expected_graphs =
                    u64::try_from(runtime.require_complete_captured_topology_for(
                        ResidentGraphTopology::FleetPowSplit,
                    )?)
                    .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(usize::MAX))?;
                let expected_kernel_launches = runtime.captured_graph_kernel_node_count()?;
                runtime
                    .workspace_proof_bundle_bytes()
                    .ok_or(ResidentRuntimeError::TranscriptRequirementsMismatch)?;
                let plan_identity = artifacts
                    .telemetry
                    .shape_executable_topology_digest
                    .ok_or(ResidentRuntimeError::TranscriptRequirementsMismatch)?;
                let mut coordinator = TwoRankFleetPowCoordinator::new(
                    schedule,
                    plan_identity,
                    proof_generation,
                    &mut *transport,
                )
                .map_err(|error| ResidentRuntimeError::FleetPowControl(error.to_string()))?;

                runtime.begin_hot_path_telemetry();
                if capture_graph_replay_timing {
                    runtime.begin_graph_replay_timing()?;
                }
                let pow = replay_fleet_pow_split(runtime, &mut coordinator)
                    .map_err(|error| error.into_resident())?;
                let bundle = runtime.read_proof_bundle_once()?;
                if capture_graph_replay_timing {
                    runtime.finish_graph_replay_timing()?;
                }
                let execution = runtime.hot_path_telemetry();
                Ok((
                    artifacts.claim.clone(),
                    bundle,
                    runtime.proof_assembly_shape().clone(),
                    artifacts.discovery.lifting_log_size,
                    plan_identity,
                    pow,
                    execution,
                    expected_graphs,
                    expected_kernel_launches,
                ))
            },
        )?;
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
        validate_resident_composition_oods(
            &claim,
            &interaction_claim,
            bundle.interaction_pow,
            &proof,
            &params,
            lifting_log_size,
        )?;

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

        let proof = CairoProof {
            claim,
            interaction_pow: bundle.interaction_pow,
            interaction_claim,
            extended_stark_proof: proof,
            channel_salt: params.channel_salt,
            preprocessed_trace_variant: params.preprocessed_trace,
        };
        self.last_resident_session_telemetry = Some(session_telemetry);
        self.last_aot_stats = Some(aot_stats);

        Ok(FleetResidentBlake2sProof {
            proof,
            telemetry: FleetResidentProofTelemetry {
                proof_generation,
                plan_identity,
                pow,
                execution,
                expected_graph_launches: expected_graphs,
                expected_captured_kernel_launches: expected_kernel_launches,
                performance_admissible: false,
            },
        })
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
        self.last_pcs_telemetry = None;
        self.last_resident_session_telemetry = None;
        self.last_aot_stats = None;
        if !self.config.strict {
            return Err(GpuError::Config(
                "resident Blake2s proving requires strict GPU-native mode".to_string(),
            ));
        }
        aot::reset_runtime_stats();
        let allow_slow_graph_submit_diagnostic = self.config.allow_slow_graph_submit_diagnostic;
        let capture_graph_replay_timing = self.config.record_graph_replay_intervals_diagnostic;
        let resident_proof_execution = ResidentProofExecution::from_config(&self.config);
        if resident_proof_execution == ResidentProofExecution::CompiledCompositionVerticalCheckpoint
            && transcript_mode != ResidentTranscriptMode::DeviceOnly
        {
            return Err(GpuError::Config(
                "compiled Composition vertical checkpoint requires device-only transcript mode"
                    .to_string(),
            ));
        }

        let (
            (
                claim,
                bundle,
                shape,
                lifting_log_size,
                exec,
                expected_graphs,
                expected_kernel_launches,
                transcript_mirror,
            ),
            session_telemetry,
        ) = self.with_strict_resident_session(input, params, |runtime, artifacts| {
            runtime.require_prepared_witness_coverage()?;
            if artifacts
                .telemetry
                .prepared_runtime_materialization
                .is_none()
            {
                return Err(ResidentRuntimeError::MissingPreparedRuntimeMaterialization);
            }
            if !runtime.prepared_capture_ready()? {
                runtime.capture_all_prepared_subgraphs()?;
            }
            let expected_graphs = u64::try_from(runtime.require_complete_captured_topology()?)
                .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(usize::MAX))?;
            let expected_kernel_launches = runtime.captured_graph_kernel_node_count()?;
            let bundle_bytes = runtime
                .workspace_proof_bundle_bytes()
                .ok_or(ResidentRuntimeError::TranscriptRequirementsMismatch)?;
            if resident_proof_execution
                == ResidentProofExecution::CompiledCompositionVerticalCheckpoint
            {
                runtime.admit_compiled_composition_eager()?;
            }
            runtime.begin_hot_path_telemetry();
            if capture_graph_replay_timing {
                runtime.begin_graph_replay_timing()?;
            }
            match resident_proof_execution {
                ResidentProofExecution::CapturedGraph => runtime.replay_all_prepared_subgraphs()?,
                ResidentProofExecution::CompiledCompositionVerticalCheckpoint => {
                    runtime.launch_compiled_eager_vertical()?
                }
            }
            let bundle = runtime.read_proof_bundle_once()?;
            if capture_graph_replay_timing {
                runtime.finish_graph_replay_timing()?;
            }
            let exec = match resident_proof_execution {
                ResidentProofExecution::CapturedGraph => {
                    runtime.require_hot_path_budget(resident_hot_path_budget(
                        transcript_mode,
                        allow_slow_graph_submit_diagnostic,
                        expected_graphs,
                        expected_kernel_launches,
                        bundle_bytes,
                    ))?
                }
                ResidentProofExecution::CompiledCompositionVerticalCheckpoint => {
                    runtime.require_compiled_eager_boundary(bundle_bytes)?
                }
            };
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
                artifacts.discovery.lifting_log_size,
                exec,
                expected_graphs,
                expected_kernel_launches,
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
        validate_resident_composition_oods(
            &claim,
            &interaction_claim,
            bundle.interaction_pow,
            &proof,
            &params,
            lifting_log_size,
        )?;

        // Eager execution is deliberately not represented as ArenaGraph
        // telemetry; the typed surface stays truthful.
        let pcs_telemetry = (resident_proof_execution == ResidentProofExecution::CapturedGraph)
            .then(|| {
                CudaPcsDriverTelemetry::completed_arena_graph(
                    exec,
                    expected_graphs,
                    expected_kernel_launches,
                )
            });
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

        let proof = CairoProof {
            claim,
            interaction_pow: bundle.interaction_pow,
            interaction_claim,
            extended_stark_proof: proof,
            channel_salt: params.channel_salt,
            preprocessed_trace_variant: params.preprocessed_trace,
        };
        self.last_pcs_telemetry = pcs_telemetry;
        self.last_resident_session_telemetry = Some(session_telemetry);
        self.last_aot_stats = Some(aot_stats);

        Ok((proof, transcript_mirror))
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

    fn device(count: u32, current: u32) -> CudaDeviceSnapshot {
        CudaDeviceSnapshot {
            count,
            current,
            sm_major: 9,
            sm_minor: 0,
        }
    }

    #[test]
    fn compiled_composition_vertical_checkpoint_is_opt_in() {
        let default = GpuProverConfig::default();
        assert!(!default.compiled_composition_vertical_checkpoint);
        assert_eq!(
            ResidentProofExecution::from_config(&default),
            ResidentProofExecution::CapturedGraph
        );

        let enabled = GpuProverConfig {
            compiled_composition_vertical_checkpoint: true,
            resident_backend: ResidentBackend::ReplacementV1,
            strict: true,
            ..default
        };
        assert_eq!(
            ResidentProofExecution::from_config(&enabled),
            ResidentProofExecution::CompiledCompositionVerticalCheckpoint
        );
        validate_resident_proof_execution_config(&enabled).unwrap();
    }

    #[test]
    fn compiled_composition_vertical_checkpoint_rejects_non_strict_or_mixed_diagnostics() {
        let valid = GpuProverConfig {
            compiled_composition_vertical_checkpoint: true,
            resident_backend: ResidentBackend::ReplacementV1,
            strict: true,
            ..GpuProverConfig::default()
        };
        for invalid in [
            GpuProverConfig {
                resident_backend: ResidentBackend::LegacyResident,
                ..valid
            },
            GpuProverConfig {
                strict: false,
                ..valid
            },
            GpuProverConfig {
                allow_slow_graph_submit_diagnostic: true,
                ..valid
            },
            GpuProverConfig {
                allow_slow_graph_submit_diagnostic: true,
                record_graph_replay_intervals_diagnostic: true,
                ..valid
            },
        ] {
            assert!(validate_resident_proof_execution_config(&invalid).is_err());
        }
    }

    #[test]
    fn packed_numerator_measurement_policy_accepts_only_loaded_direct_tuple() {
        let direct = ProtocolPlanPolicy::replacement_v1(0x1234, 2048);
        let packed = packed_numerator_measurement_policy(direct).unwrap();
        assert_eq!(
            packed,
            ProtocolPlanPolicy::replacement_v1_packed_numerator_measurement_control(0x1234, 2048,)
        );
        assert_eq!(
            ProtocolPlanPolicy::replacement_v1(0x1234, 2048).quotient_numerator_schedule,
            crate::arena_plan::QuotientNumeratorSchedule::StagedGroupDirect
        );

        let mut drifted = direct;
        drifted.retained_lde_budget_bytes -= 1;
        assert!(packed_numerator_measurement_policy(drifted).is_err());
    }

    #[test]
    fn packed_numerator_measurement_constructor_rejects_non_strict_or_legacy() {
        let legacy = GpuProverConfig::default();
        assert!(
            GpuCairoProver::<Blake2sMerkleChannel>::new_packed_numerator_measurement_control(
                legacy,
            )
            .is_err()
        );
        let non_strict = GpuProverConfig {
            resident_backend: ResidentBackend::ReplacementV1,
            ..legacy
        };
        assert!(
            GpuCairoProver::<Blake2sMerkleChannel>::new_packed_numerator_measurement_control(
                non_strict,
            )
            .is_err()
        );
    }

    #[test]
    fn replacement_device_admission_accepts_one_visible_supported_device() {
        validate_replacement_device_admission(0, device(1, 0), true).unwrap();
    }

    #[test]
    fn replacement_device_admission_rejects_ambient_device_or_arch_drift() {
        for (configured, snapshot, arch_supported) in [
            (0, device(0, 0), true),
            (0, device(2, 0), true),
            (1, device(1, 0), true),
            (0, device(1, 1), true),
            (0, device(1, 0), false),
        ] {
            assert!(
                validate_replacement_device_admission(configured, snapshot, arch_supported)
                    .is_err()
            );
        }
    }

    #[test]
    fn replacement_fixed_ambient_accepts_unset_or_exact_admission() {
        assert_eq!(flags::GPU_NATIVE_DEFAULTS, REPLACEMENT_V1_REQUIRED_ENV);
        for &(name, expected) in REPLACEMENT_V1_REQUIRED_ENV {
            validate_replacement_fixed_env_value(name, None, expected).unwrap();
            validate_replacement_fixed_env_value(name, Some(expected), expected).unwrap();
        }
        for &name in REPLACEMENT_V1_DEFAULT_OFF_ENV {
            validate_replacement_fixed_env_value(name, None, "0").unwrap();
            validate_replacement_fixed_env_value(name, Some("0"), "0").unwrap();
        }
    }

    #[test]
    fn replacement_fixed_ambient_rejects_required_on_and_default_off_drift() {
        for &(name, expected) in REPLACEMENT_V1_REQUIRED_ENV {
            let drift = if expected == "1" { "0" } else { "19999" };
            assert!(validate_replacement_fixed_env_value(name, Some(drift), expected).is_err());
        }
        for &name in REPLACEMENT_V1_DEFAULT_OFF_ENV {
            assert!(validate_replacement_fixed_env_value(name, Some("1"), "0").is_err());
        }
    }

    #[test]
    fn replacement_ambient_rejects_every_forbidden_override() {
        for &name in REPLACEMENT_V1_FORBIDDEN_ENV {
            validate_replacement_forbidden_env_value(name, false).unwrap();
            assert!(validate_replacement_forbidden_env_value(name, true).is_err());
        }
    }

    #[test]
    fn replacement_admits_only_the_pre_witness_resident_entrypoint() {
        assert!(validate_execution_entry(
            ResidentBackend::ReplacementV1,
            ProverExecutionEntry::PreWitnessResident,
        )
        .is_ok());
        for entry in [
            ProverExecutionEntry::AfterWitnessResident,
            ProverExecutionEntry::LegacyPcs,
        ] {
            assert!(validate_execution_entry(ResidentBackend::ReplacementV1, entry).is_err());
            assert!(validate_execution_entry(ResidentBackend::LegacyResident, entry).is_ok());
        }
    }

    #[test]
    fn composition_oods_consistency_rejects_corrupted_trace_or_composition_opening() {
        let zero = SecureField::default();
        let one = SecureField::from(1u32);
        let point = CirclePoint { x: one, y: zero };
        let mut sampled_values = TreeVec(vec![
            vec![vec![zero]],
            vec![vec![zero]; 2 * SECURE_EXTENSION_DEGREE],
        ]);
        require_composition_oods_consistency(point, 2, &sampled_values, |_| zero).unwrap();

        let mut trace_evaluations = 0;
        assert!(matches!(
            require_composition_oods_consistency(point, 2, &sampled_values, |_| {
                trace_evaluations += 1;
                one
            }),
            Err(GpuError::Proving(ProvingError::ConstraintsNotSatisfied))
        ));
        assert_eq!(trace_evaluations, 1);
        sampled_values.last_mut().unwrap()[0][0] = one;
        assert!(matches!(
            require_composition_oods_consistency(point, 2, &sampled_values, |_| zero),
            Err(GpuError::Proving(ProvingError::ConstraintsNotSatisfied))
        ));
        sampled_values[1].pop();
        assert!(matches!(
            require_composition_oods_consistency(point, 2, &sampled_values, |_| zero),
            Err(GpuError::Config(message)) if message == "malformed composition OODS opening"
        ));
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
            resident_hot_path_budget(ResidentTranscriptMode::DeviceOnly, false, 29, 123, 371_604);
        assert_eq!(
            production,
            ResidentHotPathBudget::final_bundle(29, 123, 371_604)
        );

        let mut expected_mirrored = production;
        expected_mirrored.max_graph_submit_gap_ns = u64::MAX;
        assert_eq!(
            resident_hot_path_budget(
                ResidentTranscriptMode::DeviceMirrored,
                false,
                29,
                123,
                371_604,
            ),
            expected_mirrored
        );
    }

    #[test]
    fn graph_submit_diagnostic_relaxes_only_submit_gap_timing() {
        let strict = resident_hot_path_budget(
            ResidentTranscriptMode::DeviceOnly,
            false,
            14,
            14_205,
            8_410_304,
        );
        let mut expected_diagnostic = strict;
        expected_diagnostic.max_graph_submit_gap_ns = u64::MAX;
        assert_eq!(
            resident_hot_path_budget(
                ResidentTranscriptMode::DeviceOnly,
                true,
                14,
                14_205,
                8_410_304,
            ),
            expected_diagnostic
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
