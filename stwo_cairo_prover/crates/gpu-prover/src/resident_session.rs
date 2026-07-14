//! Production hand-off from a sealed Cairo witness to one resident CUDA runtime.
//!
//! The runtime borrows its stable arena, so it is deliberately scoped to a
//! callback while the exact workspace remains owned by the cache. This avoids a
//! self-referential session object and makes every failure leave the cache in a
//! valid, reusable state.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use cairo_air::claims::CairoClaim;
use num_traits::Zero;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::poly::circle::PolyOps;
use stwo_backend_cuda::{
    CudaBackend, ExecutionTablesHostData, PreparedEcOpIngestTelemetry,
    PreparedExecutionTablesIngestTelemetry, PreparedNumeratorSchedule, RelationChallenges,
    WitnessInputGatherRequirements,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
use stwo_cairo_prover::witness::exec_context::WitnessResidencyReport;
use stwo_cairo_prover::witness::relation_sources::RelationSourceError;

use crate::arena_plan::{ArenaPlanError, ExecutionTableGeometry, ProofArenaPlan};
use crate::composition_plan::{CompositionPlan, CompositionPlanError, CompositionProofBindings};
use crate::fixed_table_materializer::{
    PEDERSEN_POINTS_18_COLUMN_COUNT, PEDERSEN_POINTS_18_ROW_COUNT,
};
use crate::graphs::GraphWorkspace;
use crate::memory_ledger::{AllocatorPoolCheckpoint, PhysicalMemoryInputs};
use crate::plan::{ProofPlan, ProofPlanError};
use crate::protocol_discovery::{ProtocolDiscoveryError, ProtocolTranscriptDiscovery};
use crate::protocol_plan::{ProtocolPlanError, ProtocolPlanPolicy};
use crate::recorded_witness_inputs::{
    recorded_witness_inputs_for_plan, DeviceCompactColumn, DeviceEdgeColumn, DeviceEdgeSourceKind,
    DeviceGatherColumn, DeviceNativeColumn, DeviceSeedColumn, PlannedRecordedWitnessInputs,
    RecordedInputColumnProvenance, RecordedWitnessPlanError,
};
use crate::resident_runtime::{
    ResidentGraphRuntime, ResidentRuntimeError, ResidentWitnessIngestReport, ResidentWitnessInput,
    ResidentWitnessInputColumn, ResidentWorkspaceIdentity, SealedResidentExecutionConfig,
};
use crate::resident_sources::{
    inspect_base_trace_residency, stage_base_trace_coefficients, stage_preprocessed_commitment,
    stage_protocol_twiddles, stage_relation_lookup_sources, BaseTraceResidency,
    ResidentLookupStageReport, ResidentPreprocessedStageReport, ResidentSourceStageError,
    ResidentSourceStageReport, ResidentTwiddleStageReport,
};
use crate::resident_witness::{
    planned_cairo_claim, require_strict_resident_witness_coverage, ResidentWitnessPlanError,
};
use crate::shape_executable::{
    ShapeCompileRequest, ShapeExecutable, ShapeExecutableCache, ShapeExecutableCacheTelemetry,
    ShapeExecutableError, ShapeExecutableMaterialization, ShapeExecutableSelection,
};
use crate::state::{DeviceProofState, WitnessOutput};
use crate::transcript_plan::{
    encode_static_transcript_inputs, CairoBlake2sTranscriptPlan, TranscriptPlanError,
};
use crate::workspace_cache::{
    WorkspaceCache, WorkspaceCacheError, WorkspaceCacheTelemetry, WorkspaceKey,
    WorkspaceMaterialization,
};

mod physical_memory;
use physical_memory::{capture_pool_checkpoint, policy_inputs};

/// Everything whose value changes the resident graph or its stable pointers.
pub struct ResidentSessionRequest {
    pub preprocessed_trace: Arc<PreProcessedTrace>,
    pub witness: WitnessOutput<CudaBackend>,
    pub channel_salt: u32,
    pub pcs: PcsConfig,
    pub include_all_preprocessed_columns: bool,
    pub operational_safety_reserve_bytes: Option<NonZeroUsize>,
    pub protocol_policy: ProtocolPlanPolicy,
    pub execution_config: SealedResidentExecutionConfig,
}

/// Strict device-born entry: claim/shape/protocol planning happens before the
/// first base-witness output is materialized, so the exact GraphWorkspace owns
/// every recorded CUDA writer destination from birth.
pub struct ResidentPreWitnessSessionRequest {
    pub preprocessed_trace: Arc<PreProcessedTrace>,
    pub generator: CairoClaimGenerator,
    pub capacity_plan: Arc<ProofPlan>,
    pub channel_salt: u32,
    pub pcs: PcsConfig,
    pub include_all_preprocessed_columns: bool,
    pub operational_safety_reserve_bytes: Option<NonZeroUsize>,
    pub protocol_policy: ProtocolPlanPolicy,
    pub execution_config: SealedResidentExecutionConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentExecutionReadiness {
    /// Every proof stage, dynamic decommit source and the one-copy host bundle
    /// is bound to the exact workspace. Strict coverage may still reject a
    /// statement whose witness writers are not all arena-native.
    ReadyForResidentProofReplay,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentPreparationState {
    pub telemetry: ResidentSessionTelemetry,
    pub readiness: ResidentExecutionReadiness,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResidentSessionTelemetry {
    pub shape_executable_materialization: Option<ShapeExecutableMaterialization>,
    pub shape_executable_cache: ShapeExecutableCacheTelemetry,
    pub shape_executable_topology_digest: Option<[u8; 32]>,
    pub workspace_key: Option<WorkspaceKey>,
    pub workspace_materialization: Option<WorkspaceMaterialization>,
    pub cache: WorkspaceCacheTelemetry,
    pub arena_words: usize,
    pub transcript_segments: usize,
    pub protocol_policy: Option<ProtocolPlanPolicy>,
    pub prepared_numerator_schedule: Option<PreparedNumeratorSchedule>,
    pub base: ResidentSourceStageReport,
    pub twiddles: ResidentTwiddleStageReport,
    pub preprocessed: ResidentPreprocessedStageReport,
    pub lookups: ResidentLookupStageReport,
    pub transcript_ingest_bytes: usize,
    pub transcript_ingest_copies: usize,
    pub witness: WitnessResidencyReport,
    pub execution_tables_ingest: Option<PreparedExecutionTablesIngestTelemetry>,
    pub ec_op_ingest: Option<PreparedEcOpIngestTelemetry>,
    pub recorded_witness_ingest: ResidentWitnessIngestReport,
    pub allocator_pool_checkpoint: Option<AllocatorPoolCheckpoint>,
    pub physical_memory_inputs: PhysicalMemoryInputs,
    /// A failed native checkpoint leaves its rows absent so admission remains
    /// false without discarding an otherwise valid proof.
    pub physical_memory_checkpoint_error: Option<String>,
}

impl ResidentSessionTelemetry {
    pub fn cache_hit(&self) -> bool {
        self.workspace_materialization == Some(WorkspaceMaterialization::Reused)
    }

    pub fn staged_bytes(&self) -> Result<usize, ResidentSessionError> {
        let execution_table_bytes = self
            .execution_tables_ingest
            .map(|ingest| {
                ingest
                    .compact_h2d_bytes
                    .checked_add(ingest.descriptor_h2d_bytes)
                    .and_then(|bytes| usize::try_from(bytes).ok())
                    .ok_or(ResidentSessionError::SizeOverflow)
            })
            .transpose()?
            .unwrap_or(0);
        let ec_op_bytes = self
            .ec_op_ingest
            .map_or(0, |ingest| ingest.h2d_bytes as usize);
        self.base
            .d2d_bytes
            .checked_add(self.preprocessed.d2d_bytes)
            .and_then(|bytes| bytes.checked_add(self.preprocessed.descriptor_h2d_bytes))
            .and_then(|bytes| bytes.checked_add(self.lookups.staged_bytes))
            .and_then(|bytes| bytes.checked_add(self.transcript_ingest_bytes))
            .and_then(|bytes| bytes.checked_add(self.twiddles.d2d_bytes))
            .and_then(|bytes| bytes.checked_add(execution_table_bytes))
            .and_then(|bytes| bytes.checked_add(ec_op_bytes))
            .and_then(|bytes| bytes.checked_add(self.recorded_witness_ingest.h2d_bytes))
            .ok_or(ResidentSessionError::SizeOverflow)
    }

    pub fn staged_copies(&self) -> Result<usize, ResidentSessionError> {
        let execution_table_copies = self
            .execution_tables_ingest
            .map(|ingest| {
                ingest
                    .compact_h2d_copies
                    .checked_add(ingest.descriptor_h2d_copies)
                    .and_then(|copies| usize::try_from(copies).ok())
                    .ok_or(ResidentSessionError::SizeOverflow)
            })
            .transpose()?
            .unwrap_or(0);
        let ec_op_calls = self.ec_op_ingest.map_or(0, |ingest| {
            ingest.h2d_copies.saturating_add(ingest.fill_calls) as usize
        });
        self.base
            .d2d_copies
            .checked_add(self.preprocessed.d2d_copies)
            .and_then(|copies| copies.checked_add(self.preprocessed.descriptor_h2d_copies))
            .and_then(|copies| copies.checked_add(self.lookups.host_copies))
            .and_then(|copies| copies.checked_add(self.lookups.device_copies))
            .and_then(|copies| copies.checked_add(self.lookups.fill_calls))
            .and_then(|copies| copies.checked_add(self.transcript_ingest_copies))
            .and_then(|copies| copies.checked_add(self.twiddles.d2d_copies))
            .and_then(|copies| copies.checked_add(execution_table_copies))
            .and_then(|copies| copies.checked_add(ec_op_calls))
            .and_then(|copies| copies.checked_add(self.recorded_witness_ingest.h2d_copies))
            .ok_or(ResidentSessionError::SizeOverflow)
    }

    /// Admission check for the strict Graph-A path. Fixed twiddle and
    /// preprocessed setup plus compact recorded-input ingest are allowed;
    /// executing/staging any legacy base or interaction writer is not.
    pub fn require_strict_graph_a(&self) -> Result<(), ResidentSessionError> {
        let policy =
            self.protocol_policy
                .ok_or(ResidentSessionError::StrictArchitectureTelemetry(
                    "resident protocol policy was not reported",
                ))?;
        if self.workspace_key.is_none() || self.shape_executable_topology_digest.is_none() {
            return Err(ResidentSessionError::StrictArchitectureTelemetry(
                "resident topology identity was not reported",
            ));
        }
        if policy.resident_backend == crate::arena_plan::ResidentBackend::ReplacementV1
            && policy
                != ProtocolPlanPolicy::replacement_v1(
                    policy.kernel_manifest_hash,
                    policy.composition_max_kernel_instrs,
                )
        {
            return Err(ResidentSessionError::StrictArchitectureTelemetry(
                "replacement-v1 policy tuple drifted",
            ));
        }
        let prepared = self.prepared_numerator_schedule.ok_or(
            ResidentSessionError::StrictArchitectureTelemetry(
                "prepared quotient-numerator schedule was not reported",
            ),
        )?;
        let schedule_matches = match (policy.quotient_numerator_schedule, prepared) {
            (
                crate::arena_plan::QuotientNumeratorSchedule::LegacyBatches,
                PreparedNumeratorSchedule::LegacyBatches,
            ) => true,
            (
                crate::arena_plan::QuotientNumeratorSchedule::HybridSingleWrite,
                PreparedNumeratorSchedule::HybridCandidate {
                    eligible_groups, ..
                },
            ) => eligible_groups != 0,
            _ => false,
        };
        if !schedule_matches {
            return Err(ResidentSessionError::StrictArchitectureTelemetry(
                "planned and prepared quotient-numerator schedules differ",
            ));
        }
        if self.base != ResidentSourceStageReport::default() {
            return Err(ResidentSessionError::StrictArchitectureTelemetry(
                "legacy base trace staging executed",
            ));
        }
        if self.lookups != ResidentLookupStageReport::default() {
            return Err(ResidentSessionError::StrictArchitectureTelemetry(
                "legacy relation source staging executed",
            ));
        }
        if self.witness != WitnessResidencyReport::default() {
            return Err(ResidentSessionError::StrictArchitectureTelemetry(
                "legacy witness execution context executed",
            ));
        }
        let execution_tables = self.execution_tables_ingest.ok_or(
            ResidentSessionError::StrictArchitectureTelemetry(
                "prepared execution tables were not ingested",
            ),
        )?;
        if execution_tables.sync_calls != 1
            || execution_tables.compact_h2d_copies > 3
            || execution_tables.descriptor_h2d_copies > 2
        {
            return Err(ResidentSessionError::StrictArchitectureTelemetry(
                "prepared execution-table ingest exceeded its one-fence contract",
            ));
        }
        if self.ec_op_ingest.is_some_and(|ingest| {
            ingest.h2d_bytes != 0
                || ingest.h2d_copies != 0
                || ingest.fill_calls != 1
                || ingest.sync_calls != 0
        }) {
            return Err(ResidentSessionError::StrictArchitectureTelemetry(
                "EC-op segment seed violated its fill-only ingest contract",
            ));
        }
        if self.recorded_witness_ingest.sync_calls > 1
            || self.recorded_witness_ingest.columns != self.recorded_witness_ingest.h2d_copies
        {
            return Err(ResidentSessionError::StrictArchitectureTelemetry(
                "recorded witness ingest exceeded its one-fence contract",
            ));
        }
        Ok(())
    }
}

/// Exact protocol artifacts available while the resident runtime is borrowed.
pub struct ResidentSessionArtifacts<'a> {
    pub claim: &'a CairoClaim,
    pub proof_plan: &'a crate::plan::ProofPlan,
    pub discovery: &'a ProtocolTranscriptDiscovery,
    pub transcript_plan: &'a CairoBlake2sTranscriptPlan,
    pub composition_plan: &'a CompositionPlan,
    pub telemetry: &'a ResidentSessionTelemetry,
}

#[derive(Debug)]
pub enum ResidentSessionError {
    UnsealedProofPlan,
    EmptyTraceGeometry,
    InvalidLiftingLogSize {
        lifting: u32,
        required: u32,
    },
    InvalidRelationAlphaWords(usize),
    SizeOverflow,
    Discovery(ProtocolDiscoveryError),
    CompositionPlan(CompositionPlanError),
    Transcript(TranscriptPlanError),
    Protocol(ProtocolPlanError),
    Arena(ArenaPlanError),
    Cache(WorkspaceCacheError),
    Source(ResidentSourceStageError),
    RelationSource(RelationSourceError),
    Runtime(ResidentRuntimeError),
    ProofPlan(ProofPlanError),
    ResidentWitness(ResidentWitnessPlanError),
    RecordedWitness(RecordedWitnessPlanError),
    ShapeExecutable(ShapeExecutableError),
    RecordedPedersenTableUnavailable,
    RecordedWitnessInputRoute {
        component: &'static str,
        ordinal: usize,
    },
    PhysicalMemory(&'static str),
    StrictArchitectureTelemetry(&'static str),
    PlannedClaimMismatch,
    PublicMemoryMultiplicitySeed(&'static str),
    PlannedShapeMismatch {
        context: &'static str,
        component: Option<&'static str>,
        expected: usize,
        actual: usize,
    },
    DetachedBaseWitness {
        migrated_columns: usize,
    },
}

impl core::fmt::Display for ResidentSessionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident CUDA session preparation failed: {self:?}")
    }
}

impl std::error::Error for ResidentSessionError {}

macro_rules! convert_error {
    ($from:ty, $variant:ident) => {
        impl From<$from> for ResidentSessionError {
            fn from(value: $from) -> Self {
                Self::$variant(value)
            }
        }
    };
}

convert_error!(ProtocolDiscoveryError, Discovery);
convert_error!(CompositionPlanError, CompositionPlan);
convert_error!(TranscriptPlanError, Transcript);
convert_error!(ProtocolPlanError, Protocol);
convert_error!(ArenaPlanError, Arena);
convert_error!(WorkspaceCacheError, Cache);
convert_error!(ResidentSourceStageError, Source);
convert_error!(RelationSourceError, RelationSource);
convert_error!(ResidentRuntimeError, Runtime);
convert_error!(ProofPlanError, ProofPlan);
convert_error!(ResidentWitnessPlanError, ResidentWitness);
convert_error!(RecordedWitnessPlanError, RecordedWitness);
convert_error!(ShapeExecutableError, ShapeExecutable);

/// The exact dynamic composition/FRI log size used by STWO's lifting decision.
/// The reference prover derives this from the split composition commitment,
/// whose degree is bounded by the base and interaction traces.  A taller
/// preprocessed tree is opened through remapped query positions and must not
/// inflate this value.
pub fn resident_lifting_log_size(
    claim: &CairoClaim,
    pcs: PcsConfig,
) -> Result<u32, ResidentSessionError> {
    let max_trace_log = claim
        .log_sizes()
        .iter()
        .flatten()
        .copied()
        .max()
        .ok_or(ResidentSessionError::EmptyTraceGeometry)?;
    lifting_log_size_from_max(max_trace_log, pcs)
}

/// Largest domain needed by any commitment.  This sizes the shared twiddle
/// table and may exceed [`resident_lifting_log_size`] only for the preprocessed
/// tree when the PCS uses implicit lifting.
pub fn resident_max_domain_log_size(
    claim: &CairoClaim,
    preprocessed_trace: &PreProcessedTrace,
    pcs: PcsConfig,
) -> Result<u32, ResidentSessionError> {
    let max_trace_log = claim
        .log_sizes()
        .iter()
        .flatten()
        .copied()
        .chain(preprocessed_trace.log_sizes())
        .max()
        .ok_or(ResidentSessionError::EmptyTraceGeometry)?;
    lifting_log_size_from_max(max_trace_log, pcs)
}

fn lifting_log_size_from_max(
    max_trace_log: u32,
    pcs: PcsConfig,
) -> Result<u32, ResidentSessionError> {
    let required = max_trace_log
        .checked_add(pcs.fri_config.log_blowup_factor.max(1))
        .ok_or(ResidentSessionError::SizeOverflow)?;
    let lifting = pcs.lifting_log_size.unwrap_or(required);
    if lifting < required {
        return Err(ResidentSessionError::InvalidLiftingLogSize { lifting, required });
    }
    Ok(lifting)
}

fn select_resident_executable(
    cache: &mut ShapeExecutableCache,
    claim: &CairoClaim,
    proof_plan: &ProofPlan,
    preprocessed_trace: &PreProcessedTrace,
    pcs: PcsConfig,
    include_all_preprocessed_columns: bool,
    execution_tables: Option<ExecutionTableGeometry>,
    policy: ProtocolPlanPolicy,
) -> Result<ShapeExecutableSelection, ResidentSessionError> {
    Ok(cache.compile_or_bind(ShapeCompileRequest {
        claim,
        proof_plan,
        preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        execution_tables,
        policy,
    })?)
}

#[allow(clippy::too_many_arguments)]
fn run_materialized_session<R>(
    workspace: &mut GraphWorkspace,
    workspace_materialization: WorkspaceMaterialization,
    preprocessed_trace: Arc<PreProcessedTrace>,
    witness: WitnessOutput<CudaBackend>,
    channel_salt: u32,
    pcs: PcsConfig,
    executable: &ShapeExecutable,
    composition_bindings: &CompositionProofBindings,
    shape_executable_materialization: ShapeExecutableMaterialization,
    shape_executable_cache: ShapeExecutableCacheTelemetry,
    require_device_born: bool,
    operational_safety_reserve_bytes: Option<NonZeroUsize>,
    execution_config: SealedResidentExecutionConfig,
    run: impl FnOnce(
        &mut ResidentGraphRuntime<'_>,
        ResidentSessionArtifacts<'_>,
    ) -> Result<R, ResidentRuntimeError>,
) -> Result<(R, ResidentSessionTelemetry), ResidentSessionError> {
    let WitnessOutput {
        trace,
        claim,
        interaction_generator,
        device,
    } = witness;
    let DeviceProofState {
        witness_exec_context,
        proof_plan,
    } = device;
    if !proof_plan.capture_ready() {
        return Err(ResidentSessionError::UnsealedProofPlan);
    }
    let witness_report = witness_exec_context.witness_residency_report();
    if require_device_born {
        let residency = inspect_base_trace_residency(workspace, &proof_plan, &trace)?;
        require_device_born_base(residency)?;
    }
    let twiddles = (!workspace.fixed_twiddles_ready()).then(|| {
        CudaBackend::precompute_twiddles(
            CanonicCoset::new(resident_twiddle_log_size(workspace))
                .circle_domain()
                .half_coset,
        )
    });
    let base = stage_base_trace_coefficients(workspace, &proof_plan, trace, twiddles.as_ref())?;
    // `stage_base_trace_coefficients` synchronizes its arena hand-off before
    // returning. No replay node reads the detached source tree afterwards.
    drop(twiddles);
    let preprocessed = stage_preprocessed_commitment(workspace, Arc::clone(&preprocessed_trace))?;
    let relation_sources =
        interaction_generator.into_relation_lookup_sources(&witness_exec_context)?;
    let lookups = stage_relation_lookup_sources(workspace, &relation_sources)?;
    drop(relation_sources);

    let alpha_words = workspace.plan().relation().requirements.alpha_words;
    if alpha_words % SECURE_EXTENSION_DEGREE != 0 {
        return Err(ResidentSessionError::InvalidRelationAlphaWords(alpha_words));
    }
    let setup_alphas = vec![SecureField::zero(); alpha_words / SECURE_EXTENSION_DEGREE];
    let mut runtime = ResidentGraphRuntime::prepare(
        workspace,
        ResidentWorkspaceIdentity::of(workspace),
        execution_config,
        RelationChallenges {
            alpha_powers: &setup_alphas,
            z: SecureField::zero(),
        },
        executable.transcript(),
        executable.composition(),
        composition_bindings,
        None,
        None,
        None,
        None,
    )?;
    let transcript_inputs = encode_static_transcript_inputs(channel_salt, pcs, &claim)?;
    runtime.upload_transcript_inputs_at_ingest(&transcript_inputs)?;
    let transcript_ingest_bytes =
        transcript_inputs
            .iter()
            .try_fold(0usize, |bytes, (_, words)| {
                words
                    .len()
                    .checked_mul(core::mem::size_of::<u32>())
                    .and_then(|next| bytes.checked_add(next))
                    .ok_or(ResidentSessionError::SizeOverflow)
            })?;
    let mut telemetry = ResidentSessionTelemetry {
        shape_executable_materialization: Some(shape_executable_materialization),
        shape_executable_cache,
        shape_executable_topology_digest: Some(executable.topology().digest()),
        workspace_key: Some(executable.workspace_key()),
        workspace_materialization: Some(workspace_materialization),
        cache: WorkspaceCacheTelemetry::default(),
        arena_words: workspace.plan().total_words(),
        transcript_segments: executable.transcript().segments().len(),
        protocol_policy: Some(executable.protocol_policy()),
        prepared_numerator_schedule: Some(runtime.prepared_numerator_schedule()),
        base,
        twiddles: ResidentTwiddleStageReport::default(),
        preprocessed,
        lookups,
        transcript_ingest_bytes,
        transcript_ingest_copies: transcript_inputs.len(),
        witness: witness_report,
        execution_tables_ingest: None,
        ec_op_ingest: None,
        recorded_witness_ingest: ResidentWitnessIngestReport::default(),
        allocator_pool_checkpoint: None,
        physical_memory_inputs: policy_inputs(operational_safety_reserve_bytes)?,
        physical_memory_checkpoint_error: None,
    };
    let result = run(
        &mut runtime,
        ResidentSessionArtifacts {
            claim: &claim,
            proof_plan: &proof_plan,
            discovery: executable.discovery(),
            transcript_plan: executable.transcript(),
            composition_plan: executable.composition(),
            telemetry: &telemetry,
        },
    )?;
    if let Err(error) = capture_pool_checkpoint(workspace, &mut telemetry) {
        telemetry.physical_memory_checkpoint_error = Some(error.to_string());
    }
    Ok((result, telemetry))
}

/// Largest commitment domain backed by the workspace's shared forward tree.
///
/// The fixed preprocessed tree may be taller than the dynamic quotient/FRI
/// domain. Generating only the quotient-sized tree would make the cold stage
/// reject that valid protocol geometry, so derive this from the exact compiled
/// commitment plans rather than from one consumer.
fn resident_twiddle_log_size(workspace: &GraphWorkspace) -> u32 {
    let quotient = workspace.plan().quotient().config.lifting_log_size;
    max_twiddle_log_size(
        quotient,
        workspace
            .plan()
            .commitments()
            .iter()
            .map(|commitment| commitment.config.lifting_log_size),
    )
}

fn max_twiddle_log_size(
    quotient_log_size: u32,
    commitment_log_sizes: impl IntoIterator<Item = u32>,
) -> u32 {
    commitment_log_sizes
        .into_iter()
        .fold(quotient_log_size, u32::max)
}

fn require_device_born_base(residency: BaseTraceResidency) -> Result<(), ResidentSessionError> {
    if residency.migrated_columns == 0 && residency.direct_columns == residency.columns {
        Ok(())
    } else {
        Err(ResidentSessionError::DetachedBaseWitness {
            migrated_columns: residency.migrated_columns,
        })
    }
}

fn public_memory_multiplicity_seed_words(
    claim: &CairoClaim,
    memory: &stwo_cairo_adapter::memory::Memory,
) -> Result<Vec<u32>, ResidentSessionError> {
    let entries = claim.public_data.public_memory.get_entries(
        claim.public_data.initial_state.pc.0,
        claim.public_data.initial_state.ap.0,
        claim.public_data.final_state.ap.0,
    );
    let mut addresses = Vec::new();
    let mut ids = Vec::new();
    for (address, id, _) in entries {
        if address == 0 {
            return Err(ResidentSessionError::PublicMemoryMultiplicitySeed(
                "public address zero cannot be represented by the address-minus-one descriptor",
            ));
        }
        let actual = memory.address_to_id.get(address as usize).ok_or(
            ResidentSessionError::PublicMemoryMultiplicitySeed(
                "public address is outside execution memory",
            ),
        )?;
        if actual.0 != id {
            return Err(ResidentSessionError::PublicMemoryMultiplicitySeed(
                "claim public memory disagrees with execution memory",
            ));
        }
        if !public_memory_id_is_valid(id, memory.f252_values.len(), memory.small_values.len()) {
            return Err(ResidentSessionError::PublicMemoryMultiplicitySeed(
                "claim public memory contains an invalid encoded ID",
            ));
        }
        addresses.push(address);
        ids.push(id);
    }
    if addresses.is_empty() {
        return Err(ResidentSessionError::PublicMemoryMultiplicitySeed(
            "claim contains no public memory",
        ));
    }
    addresses.extend(ids);
    Ok(addresses)
}

fn public_memory_id_is_valid(id: u32, n_f252: usize, n_small: usize) -> bool {
    use stwo_cairo_adapter::memory::DEFAULT_ID;

    let index = (id & 0x3fff_ffff) as usize;
    match id >> 30 {
        0 => id != DEFAULT_ID && index < n_small,
        1 => index < n_f252,
        _ => false,
    }
}

fn ensure_process_owned_pedersen_table(
    executable: &ShapeExecutable,
) -> Result<(), ResidentSessionError> {
    if !executable.arena().requires_registered_pedersen_table() {
        return Ok(());
    }
    if !stwo_cairo_prover::witness::jit_prove_backend::ensure_device_pedersen_table() {
        return Err(ResidentSessionError::RecordedPedersenTableUnavailable);
    }
    let table = stwo_backend_cuda::pedersen_table::registered_borrowed_pedersen_table()
        .ok_or(ResidentSessionError::RecordedPedersenTableUnavailable)?;
    if !table.has_exact_rows(PEDERSEN_POINTS_18_ROW_COUNT)
        || table.columns().len() != PEDERSEN_POINTS_18_COLUMN_COUNT
        || !table.columns().iter().enumerate().all(|(index, column)| {
            column.index() == index && column.len_words() == PEDERSEN_POINTS_18_ROW_COUNT
        })
    {
        return Err(ResidentSessionError::RecordedPedersenTableUnavailable);
    }
    Ok(())
}

/// Compatibility hand-off for callers that already own a sealed witness. It
/// still enforces arena-born base columns; a detached legacy witness is rejected
/// before staging. New callers use [`with_resident_session_from_generator`].
pub fn with_resident_session<R>(
    executable_cache: &mut ShapeExecutableCache,
    cache: &mut WorkspaceCache,
    request: ResidentSessionRequest,
    run: impl FnOnce(
        &mut ResidentGraphRuntime<'_>,
        ResidentSessionArtifacts<'_>,
    ) -> Result<R, ResidentRuntimeError>,
) -> Result<(R, ResidentSessionTelemetry), ResidentSessionError> {
    let ResidentSessionRequest {
        preprocessed_trace,
        witness,
        channel_salt,
        pcs,
        include_all_preprocessed_columns,
        operational_safety_reserve_bytes,
        protocol_policy,
        execution_config,
    } = request;
    let selection = select_resident_executable(
        executable_cache,
        &witness.claim,
        &witness.device.proof_plan,
        &preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        None,
        protocol_policy,
    )?;
    let shape_executable_cache = executable_cache.telemetry();
    let ShapeExecutableSelection {
        executable,
        bindings,
        materialization: executable_materialization,
    } = selection;
    ensure_process_owned_pedersen_table(&executable)?;
    let (result, mut telemetry) = {
        let (workspace, materialization) = cache.materialize_or_reuse(&executable)?;
        run_materialized_session(
            workspace,
            materialization,
            preprocessed_trace,
            witness,
            channel_salt,
            pcs,
            &executable,
            &bindings,
            executable_materialization,
            shape_executable_cache,
            true,
            operational_safety_reserve_bytes,
            execution_config,
            run,
        )?
    };
    telemetry.cache = cache.telemetry();
    Ok((result, telemetry))
}

struct ResidentWitnessInputRoute<'a> {
    columns: Vec<ResidentWitnessInputColumn<'a>>,
    seed_scalars: Vec<u32>,
}

fn recorded_input_matches_gather(
    source: &RecordedInputColumnProvenance,
    ordinal: usize,
    producers: &[&'static str],
    requirements: &WitnessInputGatherRequirements,
) -> bool {
    if producers.len() != requirements.edges.len() {
        return false;
    }
    let edge_matches = |column: &DeviceEdgeColumn, edge_index: usize| {
        let plan = &requirements.edges[edge_index];
        column.producer == producers[edge_index]
            && column.source_kind == DeviceEdgeSourceKind::SubcomponentWords
            && usize::try_from(column.source_word).ok() == plan.edge.word_base.checked_add(ordinal)
            && column.words_per_instance as usize == plan.edge.words_per_instance
            && column.n_instances as usize == plan.edge.n_instances
    };

    if ordinal < requirements.input_width {
        return match source {
            RecordedInputColumnProvenance::DeviceEdge(column) if requirements.edges.len() == 1 => {
                edge_matches(column, 0)
            }
            RecordedInputColumnProvenance::DeviceGather(DeviceGatherColumn::Data { edges })
                if requirements.edges.len() > 1 && edges.len() == requirements.edges.len() =>
            {
                edges
                    .iter()
                    .enumerate()
                    .all(|(edge_index, column)| edge_matches(column, edge_index))
            }
            _ => false,
        };
    }

    let device_tail = requirements.edges.len() > 1;
    let enabler_ordinal = requirements
        .include_enabler
        .then_some(requirements.input_width);
    if Some(ordinal) == enabler_ordinal {
        return match source {
            RecordedInputColumnProvenance::DeviceGather(DeviceGatherColumn::Enabler) => device_tail,
            RecordedInputColumnProvenance::Host(words) => {
                !device_tail
                    && words.len() == requirements.consumer_rows
                    && words
                        .iter()
                        .enumerate()
                        .all(|(row, &value)| value == u32::from(row < requirements.total_real_rows))
            }
            _ => false,
        };
    }
    let iota_ordinal = requirements
        .include_iota
        .then_some(requirements.input_width + usize::from(requirements.include_enabler));
    if Some(ordinal) == iota_ordinal {
        return match source {
            RecordedInputColumnProvenance::DeviceGather(DeviceGatherColumn::Iota) => device_tail,
            RecordedInputColumnProvenance::Host(words) => {
                !device_tail
                    && words.len() == requirements.consumer_rows
                    && words
                        .iter()
                        .enumerate()
                        .all(|(row, &value)| u32::try_from(row).ok() == Some(value))
            }
            _ => false,
        };
    }
    false
}

fn resident_host_witness_inputs<'a>(
    recorded: &'a PlannedRecordedWitnessInputs,
    arena: &ProofArenaPlan,
) -> Result<Vec<ResidentWitnessInputRoute<'a>>, ResidentSessionError> {
    if recorded.lanes.len() != arena.witness().components.len() {
        return Err(ResidentSessionError::PlannedShapeMismatch {
            context: "recorded lane count vs arena witness components",
            component: None,
            expected: arena.witness().components.len(),
            actual: recorded.lanes.len(),
        });
    }
    recorded
        .lanes
        .iter()
        .zip(&arena.witness().components)
        .map(|(lane, component)| {
            if lane.component != component.component {
                return Err(ResidentSessionError::PlannedShapeMismatch {
                    context: "recorded lane order vs arena witness order",
                    component: Some(lane.component),
                    expected: 0,
                    actual: 0,
                });
            }
            if lane.columns.len() != component.program.n_inputs as usize {
                return Err(ResidentSessionError::PlannedShapeMismatch {
                    context: "recorded lane column count vs recording n_inputs",
                    component: Some(lane.component),
                    expected: component.program.n_inputs as usize,
                    actual: lane.columns.len(),
                });
            }
            if let Some(producer) = component.native_input_producer {
                if lane.columns.iter().enumerate().all(|(ordinal, source)| {
                    matches!(
                        source,
                        RecordedInputColumnProvenance::DeviceNative(DeviceNativeColumn {
                            producer: source_producer,
                            ordinal: source_ordinal,
                        }) if *source_producer == producer && *source_ordinal == ordinal
                    )
                }) {
                    return Ok(ResidentWitnessInputRoute {
                        columns: Vec::new(),
                        seed_scalars: Vec::new(),
                    });
                }
                return Err(ResidentSessionError::RecordedWitnessInputRoute {
                    component: lane.component,
                    ordinal: lane
                        .columns
                        .iter()
                        .enumerate()
                        .find_map(|(ordinal, source)| {
                            (!matches!(source, RecordedInputColumnProvenance::DeviceNative(_)))
                                .then_some(ordinal)
                        })
                        .unwrap_or(0),
                });
            }
            if let Some(gather) = &component.input_gather {
                for (ordinal, source) in lane.columns.iter().enumerate() {
                    if !recorded_input_matches_gather(
                        source,
                        ordinal,
                        &gather.producers,
                        &gather.requirements,
                    ) {
                        return Err(ResidentSessionError::RecordedWitnessInputRoute {
                            component: lane.component,
                            ordinal,
                        });
                    }
                }
                // The gather kernel writes both producer columns and its
                // mechanically generated enabler/iota tail; no host copy may
                // race those destinations before capture.
                return Ok(ResidentWitnessInputRoute {
                    columns: Vec::new(),
                    seed_scalars: Vec::new(),
                });
            }

            if let Some(compact) = &component.input_compact {
                let layout = compact.requirements.layout;
                for (ordinal, source) in lane.columns.iter().enumerate() {
                    let valid = if ordinal < layout.tuple_words {
                        matches!(
                            source,
                            RecordedInputColumnProvenance::DeviceCompact(
                                DeviceCompactColumn::Tuple { word }
                            ) if *word == ordinal
                        )
                    } else if Some(ordinal) == layout.enabler_slot {
                        matches!(
                            source,
                            RecordedInputColumnProvenance::DeviceCompact(
                                DeviceCompactColumn::Enabler
                            )
                        )
                    } else if Some(ordinal) == layout.iota_slot {
                        matches!(
                            source,
                            RecordedInputColumnProvenance::DeviceCompact(DeviceCompactColumn::Iota)
                        )
                    } else if ordinal == layout.multiplicity_slot {
                        matches!(
                            source,
                            RecordedInputColumnProvenance::DeviceCompact(
                                DeviceCompactColumn::Multiplicity
                            )
                        )
                    } else {
                        false
                    };
                    if !valid {
                        return Err(ResidentSessionError::RecordedWitnessInputRoute {
                            component: lane.component,
                            ordinal,
                        });
                    }
                }
                return Ok(ResidentWitnessInputRoute {
                    columns: Vec::new(),
                    seed_scalars: Vec::new(),
                });
            }

            if component.input_seed.is_some() {
                let mut seed_scalars = Vec::new();
                for (ordinal, source) in lane.columns.iter().enumerate() {
                    match source {
                        RecordedInputColumnProvenance::DeviceSeed(DeviceSeedColumn::Scalar {
                            index,
                            value,
                        }) if *index == seed_scalars.len() && ordinal == *index => {
                            seed_scalars.push(*value);
                        }
                        RecordedInputColumnProvenance::DeviceSeed(DeviceSeedColumn::Enabler)
                        | RecordedInputColumnProvenance::DeviceSeed(DeviceSeedColumn::Iota) => {}
                        _ => {
                            return Err(ResidentSessionError::RecordedWitnessInputRoute {
                                component: lane.component,
                                ordinal,
                            })
                        }
                    }
                }
                let expected_scalars = component
                    .input_seed
                    .as_ref()
                    .expect("seed presence checked")
                    .requirements
                    .scalar_words;
                if seed_scalars.len() != expected_scalars {
                    return Err(ResidentSessionError::PlannedShapeMismatch {
                        context: "recorded seed scalars vs planned seed words",
                        component: Some(lane.component),
                        expected: expected_scalars,
                        actual: seed_scalars.len(),
                    });
                }
                return Ok(ResidentWitnessInputRoute {
                    columns: Vec::new(),
                    seed_scalars,
                });
            }

            let columns = lane
                .columns
                .iter()
                .enumerate()
                .map(|(ordinal, source)| match source {
                    RecordedInputColumnProvenance::Host(words) => {
                        Ok(ResidentWitnessInputColumn { ordinal, words })
                    }
                    RecordedInputColumnProvenance::DeviceEdge(_)
                    | RecordedInputColumnProvenance::DeviceGather(_)
                    | RecordedInputColumnProvenance::DeviceNative(_)
                    | RecordedInputColumnProvenance::Unresolved(_) => {
                        Err(ResidentSessionError::RecordedWitnessInputRoute {
                            component: lane.component,
                            ordinal,
                        })
                    }
                    RecordedInputColumnProvenance::DeviceSeed(_) => {
                        Err(ResidentSessionError::RecordedWitnessInputRoute {
                            component: lane.component,
                            ordinal,
                        })
                    }
                    RecordedInputColumnProvenance::DeviceCompact(_) => {
                        Err(ResidentSessionError::RecordedWitnessInputRoute {
                            component: lane.component,
                            ordinal,
                        })
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ResidentWitnessInputRoute {
                columns,
                seed_scalars: Vec::new(),
            })
        })
        .collect()
}

/// Strict Graph-A hand-off. Planning and compact input extraction happen before
/// workspace materialization; the legacy witness writer and host interaction
/// generator are never executed on this path.
pub fn with_resident_session_from_generator<R>(
    executable_cache: &mut ShapeExecutableCache,
    cache: &mut WorkspaceCache,
    request: ResidentPreWitnessSessionRequest,
    run: impl FnOnce(
        &mut ResidentGraphRuntime<'_>,
        ResidentSessionArtifacts<'_>,
    ) -> Result<R, ResidentRuntimeError>,
) -> Result<(R, ResidentSessionTelemetry), ResidentSessionError> {
    let ResidentPreWitnessSessionRequest {
        preprocessed_trace,
        generator,
        capacity_plan,
        channel_salt,
        pcs,
        include_all_preprocessed_columns,
        operational_safety_reserve_bytes,
        protocol_policy,
        execution_config,
    } = request;
    let exact_plan = Arc::new(capacity_plan.strict_resident_exact(
        &crate::schedule_table::CAIRO_SCHEDULE,
        &crate::relation_table::CAIRO_RELATION_GRAPH,
    )?);
    let ec_op_segment_start = generator
        .ec_op_builtin
        .as_ref()
        .map(|ec_op| ec_op.ec_op_builtin_segment_start as usize);
    require_strict_resident_witness_coverage(&exact_plan)?;
    let planned_claim = planned_cairo_claim(&generator, &exact_plan)?;
    let recorded = recorded_witness_inputs_for_plan(&generator, &exact_plan)?;
    recorded.require_resolved()?;
    let memory = &recorded.execution_memory;
    let public_memory_seed = public_memory_multiplicity_seed_words(&planned_claim, memory)?;
    let public_memory_entries = public_memory_seed.len() / 2;
    let selection = select_resident_executable(
        executable_cache,
        &planned_claim,
        &exact_plan,
        &preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        Some(
            ExecutionTableGeometry::new(
                memory.address_to_id.len(),
                memory.f252_values.len(),
                memory.small_values.len(),
            )
            .with_public_memory_entries(public_memory_entries),
        ),
        protocol_policy,
    )?;
    let shape_executable_cache = executable_cache.telemetry();
    let ShapeExecutableSelection {
        executable,
        bindings: composition_bindings,
        materialization: executable_materialization,
    } = selection;

    ensure_process_owned_pedersen_table(&executable)?;
    let raw_address_to_id = memory
        .address_to_id
        .iter()
        .map(|encoded| encoded.0)
        .collect::<Vec<_>>();
    let host_columns = resident_host_witness_inputs(&recorded, executable.arena())?;
    let witness_inputs = recorded
        .lanes
        .iter()
        .zip(&host_columns)
        .map(|(lane, route)| ResidentWitnessInput {
            component: lane.component,
            columns: &route.columns,
            seed_scalars: (!route.seed_scalars.is_empty()).then_some(route.seed_scalars.as_slice()),
        })
        .collect::<Vec<_>>();

    let (result, mut telemetry) = {
        let (workspace, materialization) = cache.materialize_or_reuse(&executable)?;
        let twiddle_report = if workspace.fixed_twiddles_ready() {
            ResidentTwiddleStageReport {
                cache_hit: true,
                ..ResidentTwiddleStageReport::default()
            }
        } else {
            let twiddles = CudaBackend::precompute_twiddles(
                CanonicCoset::new(resident_twiddle_log_size(workspace))
                    .circle_domain()
                    .half_coset,
            );
            let report = stage_protocol_twiddles(workspace, &twiddles)?;
            // The stage synchronizes the arena copies before returning.
            drop(twiddles);
            report
        };
        let preprocessed =
            stage_preprocessed_commitment(workspace, Arc::clone(&preprocessed_trace))?;

        let alpha_words = workspace.plan().relation().requirements.alpha_words;
        if alpha_words % SECURE_EXTENSION_DEGREE != 0 {
            return Err(ResidentSessionError::InvalidRelationAlphaWords(alpha_words));
        }
        let setup_alphas = vec![SecureField::zero(); alpha_words / SECURE_EXTENSION_DEGREE];
        let mut runtime = ResidentGraphRuntime::prepare(
            workspace,
            ResidentWorkspaceIdentity::of(workspace),
            execution_config,
            RelationChallenges {
                alpha_powers: &setup_alphas,
                z: SecureField::zero(),
            },
            executable.transcript(),
            executable.composition(),
            &composition_bindings,
            Some(ExecutionTablesHostData {
                addr_to_id: &raw_address_to_id,
                f252_values: &memory.f252_values,
                small_values: &memory.small_values,
            }),
            ec_op_segment_start,
            Some(Arc::clone(&preprocessed_trace)),
            Some(&public_memory_seed),
        )?;
        let execution_tables_ingest = runtime.execution_tables_ingest_telemetry();
        let ec_op_ingest = runtime.ec_op_ingest_telemetry();
        let witness_ingest = runtime.upload_witness_inputs_at_ingest(&witness_inputs)?;
        let transcript_inputs = encode_static_transcript_inputs(channel_salt, pcs, &planned_claim)?;
        runtime.upload_transcript_inputs_at_ingest(&transcript_inputs)?;
        let transcript_ingest_bytes =
            transcript_inputs
                .iter()
                .try_fold(0usize, |bytes, (_, words)| {
                    words
                        .len()
                        .checked_mul(core::mem::size_of::<u32>())
                        .and_then(|next| bytes.checked_add(next))
                        .ok_or(ResidentSessionError::SizeOverflow)
                })?;
        let mut telemetry = ResidentSessionTelemetry {
            shape_executable_materialization: Some(executable_materialization),
            shape_executable_cache,
            shape_executable_topology_digest: Some(executable.topology().digest()),
            workspace_key: Some(executable.workspace_key()),
            workspace_materialization: Some(materialization),
            cache: WorkspaceCacheTelemetry::default(),
            arena_words: workspace.plan().total_words(),
            transcript_segments: executable.transcript().segments().len(),
            protocol_policy: Some(executable.protocol_policy()),
            prepared_numerator_schedule: Some(runtime.prepared_numerator_schedule()),
            base: ResidentSourceStageReport::default(),
            twiddles: twiddle_report,
            preprocessed,
            lookups: ResidentLookupStageReport::default(),
            transcript_ingest_bytes,
            transcript_ingest_copies: transcript_inputs.len(),
            witness: WitnessResidencyReport::default(),
            execution_tables_ingest,
            ec_op_ingest,
            recorded_witness_ingest: witness_ingest,
            allocator_pool_checkpoint: None,
            physical_memory_inputs: policy_inputs(operational_safety_reserve_bytes)?,
            physical_memory_checkpoint_error: None,
        };
        let result = run(
            &mut runtime,
            ResidentSessionArtifacts {
                claim: &planned_claim,
                proof_plan: &exact_plan,
                discovery: executable.discovery(),
                transcript_plan: executable.transcript(),
                composition_plan: executable.composition(),
                telemetry: &telemetry,
            },
        )?;
        if let Err(error) = capture_pool_checkpoint(workspace, &mut telemetry) {
            telemetry.physical_memory_checkpoint_error = Some(error.to_string());
        }
        (result, telemetry)
    };
    telemetry.cache = cache.telemetry();
    Ok((result, telemetry))
}

/// How the preflight bound the [`ProtocolPlanPolicy`]: to the AOT pack embedded
/// in this binary (exact, what an H100 proving run would use) or to the probe
/// placeholder used when the binary carries no pack (arena geometry is
/// unaffected by the manifest hash; the composition kernel cap is recorded so a
/// divergence from the loaded pack is visible in the report).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreflightManifestPolicy {
    Loaded {
        kernel_manifest_hash: u64,
        composition_max_kernel_instrs: usize,
    },
    Fake {
        kernel_manifest_hash: u64,
        composition_max_kernel_instrs: usize,
    },
}

/// Failure surface of [`plan_resident_preflight`]: every variant is exactly the
/// fail-closed error the strict resident session (or its Graph-A multiplicity
/// prepare) would raise on hardware.
#[derive(Debug)]
pub enum ResidentPreflightError {
    Session(ResidentSessionError),
    Multiplicity(crate::multiplicity_pipeline::GraphAMultiplicityPlanError),
}

impl From<ResidentSessionError> for ResidentPreflightError {
    fn from(value: ResidentSessionError) -> Self {
        Self::Session(value)
    }
}

impl From<crate::multiplicity_pipeline::GraphAMultiplicityPlanError> for ResidentPreflightError {
    fn from(value: crate::multiplicity_pipeline::GraphAMultiplicityPlanError) -> Self {
        Self::Multiplicity(value)
    }
}

/// Host-only output of [`plan_resident_preflight`]: everything the resident
/// session plans before the first CUDA allocation, for VRAM/coverage triage.
pub struct ResidentPreflightReport {
    /// Component ids present in the exact plan, in plan order.
    pub present_components: Vec<&'static str>,
    /// Present components whose witness writer is capture-safe (strict
    /// coverage passed, so this equals `present_components` — retained so the
    /// report stays honest if coverage semantics ever widen).
    pub capture_safe_components: Vec<&'static str>,
    /// Recorded (AOT) witness lanes, in ingest order.
    pub recorded_lanes: Vec<&'static str>,
    /// Graph-A multiplicity/feed plan; `coverage_gaps`/`blockers` are the
    /// fail-closed facts the runtime enforces at prepare.
    pub multiplicities: crate::multiplicity_pipeline::GraphAMultiplicityPlan,
    /// The full arena plan the workspace cache would materialize.
    pub arena: Arc<ProofArenaPlan>,
    pub transcript_segments: usize,
    /// Persistent host-control-plane result used by the real preflight path.
    pub shape_executable_materialization: ShapeExecutableMaterialization,
    pub shape_executable_cache: ShapeExecutableCacheTelemetry,
    pub shape_executable_topology_digest: [u8; 32],
    pub shape_executable_control_plane_ns: u128,
    /// Proof-varying values rebound without regenerating CUDA source.
    pub composition_bindings: CompositionProofBindings,
    pub manifest_policy: PreflightManifestPolicy,
    /// Exact selected topology/residency policy modeled by this host plan.
    pub protocol_policy: ProtocolPlanPolicy,
    pub interpolation_mode: stwo_backend_cuda::InterpolationLaunchMode,
}

/// Plan the strict resident session end-to-end WITHOUT touching CUDA: the same
/// fail-closed pipeline as [`with_resident_session_from_generator`] up to (and
/// including) the full arena plan — ingest artifacts in, exact plan, strict
/// witness coverage, planned claim, recorded witness inputs (`require_resolved`),
/// Graph-A multiplicity plan, protocol/arena plan. Every failure is fail-closed;
/// cached planning wraps its underlying planner error in `ShapeExecutable`.
/// Dev tooling only (`arena_preflight`); proving sessions never call this.
pub fn plan_resident_preflight(
    generator: &CairoClaimGenerator,
    capacity_plan: &ProofPlan,
    preprocessed_trace: &PreProcessedTrace,
    pcs: PcsConfig,
    include_all_preprocessed_columns: bool,
) -> Result<ResidentPreflightReport, ResidentPreflightError> {
    plan_resident_preflight_for(
        generator,
        capacity_plan,
        preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        crate::arena_plan::ResidentBackend::LegacyResident,
    )
}

/// Explicit resident-generation preflight. Replacement generations never read
/// legacy topology flags, even when a loaded AOT pack is unavailable locally.
pub fn plan_resident_preflight_for(
    generator: &CairoClaimGenerator,
    capacity_plan: &ProofPlan,
    preprocessed_trace: &PreProcessedTrace,
    pcs: PcsConfig,
    include_all_preprocessed_columns: bool,
    resident_backend: crate::arena_plan::ResidentBackend,
) -> Result<ResidentPreflightReport, ResidentPreflightError> {
    let mut executable_cache = ShapeExecutableCache::new(1).map_err(ResidentSessionError::from)?;
    plan_resident_preflight_with_cache_for(
        &mut executable_cache,
        generator,
        capacity_plan,
        preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        resident_backend,
    )
}

/// The reusable host-only preflight seam. Repeated exact-topology calls use
/// the same shape executable and rebind only statement-dependent parameters.
pub fn plan_resident_preflight_with_cache(
    executable_cache: &mut ShapeExecutableCache,
    generator: &CairoClaimGenerator,
    capacity_plan: &ProofPlan,
    preprocessed_trace: &PreProcessedTrace,
    pcs: PcsConfig,
    include_all_preprocessed_columns: bool,
) -> Result<ResidentPreflightReport, ResidentPreflightError> {
    plan_resident_preflight_with_cache_for(
        executable_cache,
        generator,
        capacity_plan,
        preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        crate::arena_plan::ResidentBackend::LegacyResident,
    )
}

pub fn plan_resident_preflight_with_cache_for(
    executable_cache: &mut ShapeExecutableCache,
    generator: &CairoClaimGenerator,
    capacity_plan: &ProofPlan,
    preprocessed_trace: &PreProcessedTrace,
    pcs: PcsConfig,
    include_all_preprocessed_columns: bool,
    resident_backend: crate::arena_plan::ResidentBackend,
) -> Result<ResidentPreflightReport, ResidentPreflightError> {
    let exact_plan = capacity_plan
        .strict_resident_exact(
            &crate::schedule_table::CAIRO_SCHEDULE,
            &crate::relation_table::CAIRO_RELATION_GRAPH,
        )
        .map_err(ResidentSessionError::from)?;
    require_strict_resident_witness_coverage(&exact_plan).map_err(ResidentSessionError::from)?;
    let planned_claim =
        planned_cairo_claim(generator, &exact_plan).map_err(ResidentSessionError::from)?;
    let recorded = recorded_witness_inputs_for_plan(generator, &exact_plan)
        .map_err(ResidentSessionError::from)?;
    recorded
        .require_resolved()
        .map_err(ResidentSessionError::from)?;
    let public_memory_entries =
        public_memory_multiplicity_seed_words(&planned_claim, &recorded.execution_memory)
            .map_err(ResidentPreflightError::from)?
            .len()
            / 2;
    let multiplicities = crate::multiplicity_pipeline::plan_graph_a_multiplicities(&exact_plan)?;

    let (protocol_policy, manifest_policy) =
        match ProtocolPlanPolicy::loaded_starknet_blake2s_for(resident_backend) {
            Ok(policy) => (
                policy,
                PreflightManifestPolicy::Loaded {
                    kernel_manifest_hash: policy.kernel_manifest_hash,
                    composition_max_kernel_instrs: policy.composition_max_kernel_instrs,
                },
            ),
            Err(ProtocolPlanError::UnboundKernelManifest)
            | Err(ProtocolPlanError::UnboundCompositionKernelCap) => {
                // Off-CUDA probe trick (see the fixture parity test above): geometry
                // never reads the manifest hash, only the composition kernel cap.
                let policy = match resident_backend {
                    crate::arena_plan::ResidentBackend::LegacyResident => {
                        ProtocolPlanPolicy::starknet_blake2s_from_env(0x1234, 2048)
                            .map_err(ResidentSessionError::from)?
                    }
                    crate::arena_plan::ResidentBackend::ReplacementV1 => {
                        ProtocolPlanPolicy::replacement_v1(0x1234, 2048)
                    }
                };
                (
                    policy,
                    PreflightManifestPolicy::Fake {
                        kernel_manifest_hash: policy.kernel_manifest_hash,
                        composition_max_kernel_instrs: policy.composition_max_kernel_instrs,
                    },
                )
            }
            Err(other) => return Err(ResidentSessionError::from(other).into()),
        };
    let memory = &recorded.execution_memory;
    let control_plane_start = Instant::now();
    let selection = executable_cache
        .compile_or_bind(ShapeCompileRequest {
            claim: &planned_claim,
            proof_plan: &exact_plan,
            preprocessed_trace,
            pcs,
            include_all_preprocessed_columns,
            execution_tables: Some(
                ExecutionTableGeometry::new(
                    memory.address_to_id.len(),
                    memory.f252_values.len(),
                    memory.small_values.len(),
                )
                .with_public_memory_entries(public_memory_entries),
            ),
            policy: protocol_policy,
        })
        .map_err(ResidentSessionError::from)?;
    let shape_executable_control_plane_ns = control_plane_start.elapsed().as_nanos();

    let present_components: Vec<&'static str> = exact_plan
        .components
        .iter()
        .filter(|component| component.runtime.is_present())
        .map(|component| component.node.id)
        .collect();
    let capture_safe_components: Vec<&'static str> = exact_plan
        .components
        .iter()
        .filter(|component| component.runtime.is_present())
        .filter(|component| component.node.facts.witness_writer.is_capture_safe())
        .map(|component| component.node.id)
        .collect();
    let recorded_lanes: Vec<&'static str> =
        recorded.lanes.iter().map(|lane| lane.component).collect();
    Ok(ResidentPreflightReport {
        present_components,
        capture_safe_components,
        recorded_lanes,
        multiplicities,
        arena: Arc::clone(selection.executable.arena()),
        transcript_segments: selection.executable.transcript().segments().len(),
        shape_executable_materialization: selection.materialization,
        shape_executable_cache: executable_cache.telemetry(),
        shape_executable_topology_digest: selection.executable.topology().digest(),
        shape_executable_control_plane_ns,
        composition_bindings: selection.bindings,
        manifest_policy,
        protocol_policy,
        interpolation_mode: protocol_policy.interpolation_mode,
    })
}

#[cfg(test)]
mod tests {
    use cairo_air::relations::CommonLookupElements;
    use stwo::core::fri::FriConfig;
    use stwo_backend_cuda::{witness_input_gather_requirements, WitnessInputGatherEdge};

    use super::*;
    use crate::composition_plan::plan_cairo_composition;
    use crate::protocol_discovery::{
        discover_protocol_transcript_shape, schema_zero_interaction_claim_for_composition,
    };
    use crate::protocol_plan::plan_protocol_geometry;
    use crate::transcript_plan::plan_cairo_blake2s_transcript;

    #[test]
    fn public_memory_seed_rejects_invalid_tags_and_out_of_bounds_ids() {
        use stwo_cairo_adapter::memory::DEFAULT_ID;

        assert!(public_memory_id_is_valid(1, 3, 2));
        assert!(public_memory_id_is_valid((1 << 30) | 2, 3, 2));
        assert!(!public_memory_id_is_valid(DEFAULT_ID, 3, usize::MAX));
        assert!(!public_memory_id_is_valid(2, 3, 2));
        assert!(!public_memory_id_is_valid((1 << 30) | 3, 3, 2));
        assert!(!public_memory_id_is_valid(2 << 30, usize::MAX, usize::MAX));
        assert!(!public_memory_id_is_valid(3 << 30, usize::MAX, usize::MAX));
    }

    #[test]
    fn claim_public_entries_equal_adapter_public_address_multiset() {
        use cairo_vm::types::layout_name::LayoutName;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
        use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
        use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

        let input = run_and_adapt(
            &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
            ProgramType::Json,
            LayoutName::all_cairo_stwo,
            None,
        )
        .unwrap();
        let mut expected = input
            .public_memory_addresses
            .iter()
            .map(|&address| (address, input.memory.get_raw_id(address)))
            .collect::<Vec<_>>();
        let ingest = crate::phases::ingest::run(input, PreProcessedTraceVariant::Canonical, None);
        let exact = ingest
            .proof_plan
            .strict_resident_exact(
                &crate::schedule_table::CAIRO_SCHEDULE,
                &crate::relation_table::CAIRO_RELATION_GRAPH,
            )
            .unwrap();
        let claim = planned_cairo_claim(&ingest.generator, &exact).unwrap();
        let recorded = recorded_witness_inputs_for_plan(&ingest.generator, &exact).unwrap();
        let words =
            public_memory_multiplicity_seed_words(&claim, &recorded.execution_memory).unwrap();
        let n = words.len() / 2;
        let mut actual = words[..n]
            .iter()
            .copied()
            .zip(words[n..].iter().copied())
            .collect::<Vec<_>>();
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(
            actual, expected,
            "public-memory pair multiset incl. duplicates"
        );
    }

    /// Host-side replication of the pre-witness session planning
    /// (`with_resident_session_from_generator` up to the
    /// `resident_host_witness_inputs` shape check), so a recorded-lane vs
    /// arena-plan drift is caught on any machine instead of surfacing as an
    /// opaque `PlannedShapeMismatch` twenty minutes into an H100 parity run.
    #[test]
    fn poseidon_fixture_recorded_lanes_match_arena_witness_plan() {
        use cairo_vm::types::layout_name::LayoutName;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
        use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
        use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

        use crate::schedule::WitnessWriterKind;

        let input = run_and_adapt(
            &get_compiled_cairo_program_path("test_prove_verify_poseidon_builtin"),
            ProgramType::Json,
            LayoutName::all_cairo_stwo,
            None,
        )
        .unwrap();
        let ingest = crate::phases::ingest::run(
            input,
            PreProcessedTraceVariant::CanonicalWithoutPedersen,
            None,
        );
        // Ingest seals the compacted consumers' exact host-derived rows into
        // the plan, so the strict resolution runs on the production plan
        // directly — exactly the session pipeline.
        let exact_plan = ingest
            .proof_plan
            .strict_resident_exact(
                &crate::schedule_table::CAIRO_SCHEDULE,
                &crate::relation_table::CAIRO_RELATION_GRAPH,
            )
            .unwrap();
        require_strict_resident_witness_coverage(&exact_plan).unwrap();
        let planned_claim = planned_cairo_claim(&ingest.generator, &exact_plan).unwrap();
        let recorded = recorded_witness_inputs_for_plan(&ingest.generator, &exact_plan).unwrap();
        recorded.require_resolved().unwrap();

        // Full protocol planning needs the embedded kernel manifest (CUDA
        // builds only), but the arena's witness component list is exactly the
        // topological RecordedAot filter — compare against that directly. The
        // aggregator must precede the chains it feeds even though plan order
        // sorts the chains first; the zip in resident_host_witness_inputs
        // fails closed on any drift.
        let lanes: Vec<_> = recorded.lanes.iter().map(|lane| lane.component).collect();
        let expected: Vec<_> = crate::arena_plan::topological_component_order(&exact_plan)
            .unwrap()
            .into_iter()
            .filter(|component| {
                component.runtime.is_present()
                    && component.node.facts.witness_writer.kind == WitnessWriterKind::RecordedAot
            })
            .map(|component| component.node.id)
            .collect();
        assert_eq!(
            lanes, expected,
            "recorded witness lanes disagree with the arena's topological order"
        );
        let aggregator = lanes
            .iter()
            .position(|id| *id == "poseidon_aggregator")
            .unwrap();
        let partial_chain = lanes
            .iter()
            .position(|id| *id == "poseidon_3_partial_rounds_chain")
            .unwrap();
        assert!(
            aggregator < partial_chain,
            "producer must be planned before its consumer"
        );

        // The runtime fails closed on any fixed-multiplicity coverage gap or
        // feed blocker; both are plan-level facts, so pin them here instead of
        // twenty minutes into a hardware parity run.
        let multiplicities =
            crate::multiplicity_pipeline::plan_graph_a_multiplicities(&exact_plan).unwrap();
        assert!(
            multiplicities.coverage_gaps.is_empty(),
            "fixed-multiplicity coverage gaps: {:?}",
            multiplicities.coverage_gaps
        );
        assert!(
            multiplicities.blockers.is_empty(),
            "multiplicity feed blockers: {:?}",
            multiplicities.blockers
        );

        // Production policy loading is manifest-blocked off-CUDA (the
        // embedded AOT pack hashes to zero), but arena geometry never depends
        // on that hash: rebuild the identical protocol/arena plan with a fake
        // nonzero policy and replicate the physical slot-length validation that
        // `PreparedWitnessInput{Gather,Seed,Compact}::prepare` performs on
        // hardware. This pins the plan↔kernel-ABI slot contract host-side, so
        // a drift fails here in seconds instead of surfacing as an opaque
        // `SlotSizeMismatch` at H100 session prep.
        let session_pcs = PcsConfig::default();
        let lifting_log_size = resident_lifting_log_size(&planned_claim, session_pcs).unwrap();
        let discovery = discover_protocol_transcript_shape(
            &planned_claim,
            &exact_plan,
            &ingest.preprocessed_trace,
            &session_pcs,
            lifting_log_size,
            false,
        )
        .unwrap();
        let transcript = plan_cairo_blake2s_transcript(
            &planned_claim,
            session_pcs,
            discovery.lifting_log_size,
            discovery.dynamic_transcript_shape(),
        )
        .unwrap();
        let policy = ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048);
        let zero_interaction_claim =
            schema_zero_interaction_claim_for_composition(&planned_claim).unwrap();
        let composition = plan_cairo_composition(
            &planned_claim,
            &CommonLookupElements::dummy(),
            &zero_interaction_claim,
            &ingest.preprocessed_trace.ids(),
            policy.composition_max_kernel_instrs,
        )
        .unwrap();
        let protocol = plan_protocol_geometry(
            &exact_plan,
            &planned_claim,
            &ingest.preprocessed_trace,
            &session_pcs,
            false,
            policy,
            &transcript,
            &discovery,
            &composition,
        )
        .unwrap();
        let memory = &recorded.execution_memory;
        let public_memory_entries = public_memory_multiplicity_seed_words(&planned_claim, memory)
            .unwrap()
            .len()
            / 2;
        let arena = ProofArenaPlan::build_with_execution_tables(
            &exact_plan,
            &protocol,
            &composition,
            ExecutionTableGeometry::new(
                memory.address_to_id.len(),
                memory.f252_values.len(),
                memory.small_values.len(),
            )
            .with_public_memory_entries(public_memory_entries),
        )
        .unwrap();
        assert_witness_input_slots_satisfy_prepare(&arena);
        assert_witness_input_compact_lifetimes(&arena);

        // Parity with the packaged preflight: `plan_resident_preflight` (the
        // pipeline the `arena_preflight` binary runs) must reproduce this
        // manual plan exactly — same fake-policy fallback off-CUDA, same
        // lanes, same coverage verdicts, same arena geometry — so the JSON
        // the tool prints for an SN-scale input is the plan the H100 session
        // would materialize.
        let preflight = plan_resident_preflight(
            &ingest.generator,
            &ingest.proof_plan,
            &ingest.preprocessed_trace,
            session_pcs,
            false,
        )
        .unwrap();
        assert_eq!(
            preflight.manifest_policy,
            PreflightManifestPolicy::Fake {
                kernel_manifest_hash: 0x1234,
                composition_max_kernel_instrs: 2048,
            },
            "off-CUDA the preflight must bind the probe placeholder policy"
        );
        assert_eq!(preflight.recorded_lanes, lanes);
        assert_eq!(
            preflight.present_components, preflight.capture_safe_components,
            "strict coverage passed, so every present component is capture-safe"
        );
        assert!(preflight.multiplicities.coverage_gaps.is_empty());
        assert!(preflight.multiplicities.blockers.is_empty());
        assert_eq!(preflight.arena.total_words(), arena.total_words());
        for epoch in crate::arena_plan::ProofEpoch::ALL {
            assert_eq!(
                preflight.arena.high_water_words(epoch),
                arena.high_water_words(epoch),
                "high-water drift at {epoch:?}"
            );
        }
        assert_eq!(preflight.arena.bindings().len(), arena.bindings().len());
        assert_eq!(
            preflight.arena.logical_buffers().len(),
            arena.logical_buffers().len()
        );
    }

    fn assert_witness_input_compact_lifetimes(arena: &ProofArenaPlan) {
        use crate::arena_plan::{BufferLifetime, BufferPurpose, ProofEpoch};

        let persistent = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble).unwrap();
        let scratch = BufferLifetime::at(ProofEpoch::Witness);
        let mut compact_components = 0;
        for component in &arena.witness().components {
            let Some(compact) = &component.input_compact else {
                continue;
            };
            compact_components += 1;
            let persistent_slots = [
                (
                    BufferPurpose::WitnessInputCompactSourcePointers,
                    0,
                    compact.slots.source_pointers,
                ),
                (
                    BufferPurpose::WitnessInputCompactDescriptors,
                    0,
                    compact.slots.descriptors,
                ),
                (
                    BufferPurpose::WitnessInputCompactOutputPointers,
                    0,
                    compact.slots.output_pointers,
                ),
            ];
            let scratch_slots = [
                (
                    BufferPurpose::WitnessInputCompactTupleScratch,
                    0,
                    compact.slots.tuple_scratch,
                ),
                (
                    BufferPurpose::WitnessInputCompactSortKey,
                    0,
                    compact.slots.sort_keys_a,
                ),
                (
                    BufferPurpose::WitnessInputCompactSortKey,
                    1,
                    compact.slots.sort_keys_b,
                ),
                (
                    BufferPurpose::WitnessInputCompactSortIndex,
                    0,
                    compact.slots.sort_indices_a,
                ),
                (
                    BufferPurpose::WitnessInputCompactSortIndex,
                    1,
                    compact.slots.sort_indices_b,
                ),
                (
                    BufferPurpose::WitnessInputCompactRunHeads,
                    0,
                    compact.slots.run_heads,
                ),
                (
                    BufferPurpose::WitnessInputCompactRunPositions,
                    0,
                    compact.slots.run_positions,
                ),
                (
                    BufferPurpose::WitnessInputCompactUniqueCount,
                    0,
                    compact.slots.n_unique,
                ),
                (
                    BufferPurpose::WitnessInputCompactSortTemp,
                    0,
                    compact.slots.sort_temp,
                ),
                (
                    BufferPurpose::WitnessInputCompactScanTemp,
                    0,
                    compact.slots.scan_temp,
                ),
            ];
            for (purpose, ordinal, physical) in persistent_slots {
                let (logical, binding) = arena
                    .find(
                        Some(component.component),
                        Some(component.part),
                        purpose,
                        ordinal,
                    )
                    .unwrap();
                assert_eq!(binding.physical, physical);
                assert_eq!(logical.lifetime, persistent);
            }
            for (purpose, ordinal, physical) in scratch_slots {
                let (logical, binding) = arena
                    .find(
                        Some(component.component),
                        Some(component.part),
                        purpose,
                        ordinal,
                    )
                    .unwrap();
                assert_eq!(binding.physical, physical);
                assert_eq!(logical.lifetime, scratch);
            }
        }
        assert!(compact_components > 0, "fixture must exercise compaction");
    }

    #[test]
    fn progressive_direct_fixture_closes_producer_to_consumer_bindings() {
        use cairo_vm::types::layout_name::LayoutName;
        use stwo_backend_cuda::{ModeAwareCommitWorkspaceRequirements, ProgressiveCommitMode};
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
        use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
        use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

        use crate::arena_plan::{BufferLifetime, BufferPurpose, CommitmentTreeId, ProofEpoch};
        use crate::direct_composition_retention::DirectCompositionRetentionMode;

        let input = run_and_adapt(
            &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
            ProgramType::Json,
            LayoutName::all_cairo_stwo,
            None,
        )
        .unwrap();
        let ingest = crate::phases::ingest::run(input, PreProcessedTraceVariant::Canonical, None);
        let exact_plan = ingest
            .proof_plan
            .strict_resident_exact(
                &crate::schedule_table::CAIRO_SCHEDULE,
                &crate::relation_table::CAIRO_RELATION_GRAPH,
            )
            .unwrap();
        let planned_claim = planned_cairo_claim(&ingest.generator, &exact_plan).unwrap();
        let recorded = recorded_witness_inputs_for_plan(&ingest.generator, &exact_plan).unwrap();
        recorded.require_resolved().unwrap();
        let memory = &recorded.execution_memory;
        let public_memory_entries = public_memory_multiplicity_seed_words(&planned_claim, memory)
            .unwrap()
            .len()
            / 2;
        let mut policy = ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048);
        policy.commit_mode = ProgressiveCommitMode::DomainProgressive;
        policy.direct_composition_retention_mode = DirectCompositionRetentionMode::ExactNative;
        let mut executable_cache = ShapeExecutableCache::new(1).unwrap();
        let selection = executable_cache
            .compile_or_bind(ShapeCompileRequest {
                claim: &planned_claim,
                proof_plan: &exact_plan,
                preprocessed_trace: &ingest.preprocessed_trace,
                pcs: PcsConfig::default(),
                include_all_preprocessed_columns: false,
                execution_tables: Some(
                    ExecutionTableGeometry::new(
                        memory.address_to_id.len(),
                        memory.f252_values.len(),
                        memory.small_values.len(),
                    )
                    .with_public_memory_entries(public_memory_entries),
                ),
                policy,
            })
            .unwrap();
        let arena = selection.executable.arena().as_ref();
        let retention = arena
            .composition()
            .direct_retention
            .as_ref()
            .expect("explicit direct policy must seal a retention plan");
        assert!(retention.direct_column_count > 0);

        for binding in arena
            .composition()
            .direct_bindings
            .iter()
            .filter(|binding| binding.evaluation.is_some())
        {
            let column = retention.columns[binding.plan_column];
            assert_eq!(
                column.canonical_column,
                column.group * 16 + column.column_in_group
            );
            let commitment = arena.commitment(column.tree).unwrap();
            let ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements) =
                &commitment.requirements
            else {
                panic!("direct retention must use a progressive producer");
            };
            let producer = requirements
                .leaves
                .plan
                .columns
                .get(column.canonical_column)
                .unwrap();
            assert_eq!(producer.canonical_index, column.canonical_column);
            assert_eq!(producer.group_index, column.group);
            assert_eq!(producer.column_in_group, column.column_in_group);
            assert_eq!(producer.coefficient_log_size, column.coefficient_log_size);
            assert_eq!(producer.evaluation_log_size, column.evaluation_log_size);
            assert!(producer.retained_evaluation);

            let runtime_flattened = commitment
                .grouped_column_log_sizes
                .iter()
                .zip(&commitment.evaluation_output_groups)
                .flat_map(|(logs, outputs)| match outputs {
                    Some(outputs) => outputs.iter().copied().map(Some).collect::<Vec<_>>(),
                    None => vec![None; logs.len()],
                })
                .collect::<Vec<_>>();
            let evaluation = binding.evaluation.unwrap();
            assert_eq!(
                runtime_flattened[column.canonical_column],
                Some(evaluation),
                "runtime producer flattening must resolve the consumer's exact slot"
            );
            assert_eq!(
                commitment.evaluation_output_groups[column.group]
                    .as_ref()
                    .unwrap()[column.column_in_group],
                evaluation
            );

            let producer_occurrences = requirements
                .leaves
                .plan
                .lde_batches
                .iter()
                .flat_map(|batch| &batch.columns)
                .filter(|&&canonical| canonical == column.canonical_column)
                .count();
            assert_eq!(
                producer_occurrences, 1,
                "every canonical column must have exactly one progressive LDE producer"
            );
            let (batch, batch_column) = requirements
                .leaves
                .plan
                .lde_batches
                .iter()
                .find_map(|batch| {
                    batch
                        .columns
                        .iter()
                        .position(|&canonical| canonical == column.canonical_column)
                        .map(|position| (batch, position))
                })
                .expect("every progressive column must occur in one LDE batch");
            assert_eq!(batch.evaluation_log_size, column.evaluation_log_size);
            assert_eq!(
                batch.retained_columns[batch_column],
                Some((column.group, column.column_in_group))
            );
        }

        let fixed = arena.commitment(CommitmentTreeId::Preprocessed).unwrap();
        let prepare_written = [
            BufferPurpose::QuotientSamplePoints,
            BufferPurpose::QuotientFirstLinearTerms,
        ]
        .map(|purpose| {
            let (logical, binding) = arena.find(None, None, purpose, 0).unwrap();
            assert_eq!(
                logical.lifetime,
                BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Quotient).unwrap(),
                "prepare-time quotient writes must be live from Ingest"
            );
            binding.physical
        });
        for purpose in [BufferPurpose::RelationAlphaPowers, BufferPurpose::RelationZ] {
            let (logical, _) = arena.find(None, None, purpose, 0).unwrap();
            assert_eq!(
                logical.lifetime,
                BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Composition).unwrap(),
                "prepare-time relation challenge writes must be live from Ingest"
            );
        }
        let fixed_cached = fixed
            .evaluation_output_groups
            .iter()
            .flatten()
            .flatten()
            .chain(fixed.retained_layers_bottom_up.iter())
            .collect::<Vec<_>>();
        for binding in &fixed_cached {
            assert!(
                !prepare_written.contains(&binding.physical),
                "quotient preparation must not overwrite a live fixed commitment output"
            );
        }
        for binding in fixed_cached {
            assert_eq!(
                arena.logical_buffers()[binding.logical.0 as usize]
                    .lifetime
                    .last,
                ProofEpoch::Assemble,
                "cached preprocessed commitment data must survive the next proof cycle"
            );
        }
        assert!(arena
            .logical_buffers()
            .iter()
            .filter(|buffer| {
                matches!(
                    buffer.purpose,
                    BufferPurpose::PreprocessedEvaluations
                        | BufferPurpose::PreprocessedCoefficients
                        | BufferPurpose::ForwardTwiddles
                        | BufferPurpose::InverseTwiddles
                        | BufferPurpose::QuotientInverseTwiddles
                )
            })
            .all(|buffer| buffer.lifetime.last == ProofEpoch::Assemble));
        let ingest_to_decommit =
            BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Decommit).unwrap();
        let finite_setup = arena
            .logical_buffers()
            .iter()
            .filter(|buffer| buffer.lifetime == ingest_to_decommit)
            .collect::<Vec<_>>();
        assert_eq!(finite_setup.len(), 1);
        assert_eq!(
            finite_setup[0].purpose,
            BufferPurpose::PreprocessedInverseTwiddles,
            "only the cold fixed interpolation twiddles may end before Assemble"
        );
        let preprocessed_root_id = crate::transcript_plan::CairoTranscriptInput::PreprocessedRoot
            .id()
            .unwrap();
        let root_input = arena
            .transcript()
            .inputs
            .iter()
            .find(|(id, _)| *id == preprocessed_root_id)
            .unwrap()
            .1;
        assert_eq!(
            arena.logical_buffers()[root_input.logical.0 as usize]
                .lifetime
                .last,
            ProofEpoch::Assemble
        );
    }

    /// Host mirror of the slot validation in stwo's
    /// `PreparedWitnessInput{Gather,Seed,Compact}::prepare` and the recorded
    /// host-column ingest: every workspace view handed to the witness-input
    /// kernels is bound to its checked stable range. Disjoint-lifetime views
    /// retain distinct identities and may reuse overlapping arena addresses,
    /// so the contract is CAPACITY — each view must hold exactly the kernel-ABI
    /// requirement recomputed at runtime. Any shortfall here
    /// is exactly the `SlotSizeMismatch`/`SourceRowsMismatch`/`SourceTooSmall`
    /// the H100 raises during `ResidentGraphRuntime::prepare`.
    fn assert_witness_input_slots_satisfy_prepare(arena: &ProofArenaPlan) {
        use stwo_backend_cuda::ArenaSlotId;

        let mut failures = Vec::new();
        let mut check_slot = |failures: &mut Vec<String>,
                              component: &str,
                              kind: &str,
                              label: String,
                              id: ArenaSlotId,
                              required: usize| {
            match arena.layout().slot(id) {
                None => failures.push(format!(
                    "{component} {kind} {label}: slot {id:?} missing from the arena layout"
                )),
                Some(spec) if spec.len_words < required => failures.push(format!(
                    "{component} {kind} {label}: slot {id:?} has {} words, \
                         requirements need at least {required}",
                    spec.len_words
                )),
                Some(_) => {}
            }
        };
        let check_sources =
            |failures: &mut Vec<String>,
             component: &str,
             kind: &str,
             sources: &[crate::arena_plan::ArenaBinding],
             edges: &[stwo_backend_cuda::WitnessInputGatherEdgePlan]| {
                for (index, (source, plan)) in sources.iter().zip(edges).enumerate() {
                    let Some(spec) = arena.layout().slot(source.physical) else {
                        failures.push(format!(
                            "{component} {kind} source[{index}]: slot {:?} missing from the arena \
                         layout",
                            source.physical
                        ));
                        continue;
                    };
                    if spec.len_words % plan.edge.producer_rows != 0 {
                        failures.push(format!(
                            "{component} {kind} source[{index}]: slot {:?} has {} words, not a \
                         multiple of {} producer rows",
                            source.physical, spec.len_words, plan.edge.producer_rows
                        ));
                    }
                    if spec.len_words < plan.required_source_words {
                        failures.push(format!(
                            "{component} {kind} source[{index}]: slot {:?} has {} words, edge \
                         requires {}",
                            source.physical, spec.len_words, plan.required_source_words
                        ));
                    }
                }
            };

        for component in &arena.witness().components {
            // The prepared writer (and, for host-fed components, the recorded
            // column ingest) reads exactly `input_column_words[i]` words from
            // each input column's slot base regardless of how the column is
            // materialized.
            for (ordinal, (&id, &words)) in component
                .slots
                .input_columns
                .iter()
                .zip(&component.requirements.input_column_words)
                .enumerate()
            {
                check_slot(
                    &mut failures,
                    component.component,
                    "writer",
                    format!("input_column[{ordinal}]"),
                    id,
                    words,
                );
            }
            if let Some(gather) = &component.input_gather {
                let requirements = &gather.requirements;
                check_slot(
                    &mut failures,
                    component.component,
                    "input_gather",
                    "source_pointers".to_owned(),
                    gather.slots.source_pointers,
                    requirements.source_pointer_words,
                );
                check_slot(
                    &mut failures,
                    component.component,
                    "input_gather",
                    "descriptors".to_owned(),
                    gather.slots.descriptors,
                    requirements.descriptor_words,
                );
                check_slot(
                    &mut failures,
                    component.component,
                    "input_gather",
                    "output_pointers".to_owned(),
                    gather.slots.output_pointers,
                    requirements.output_pointer_words,
                );
                for (ordinal, (&id, &words)) in gather
                    .slots
                    .consumer_input_columns
                    .iter()
                    .zip(&requirements.consumer_input_column_words)
                    .enumerate()
                {
                    check_slot(
                        &mut failures,
                        component.component,
                        "input_gather",
                        format!("input_column[{ordinal}]"),
                        id,
                        words,
                    );
                }
                check_sources(
                    &mut failures,
                    component.component,
                    "input_gather",
                    &gather.sources,
                    &requirements.edges,
                );
            }
            if let Some(seed) = &component.input_seed {
                let requirements = &seed.requirements;
                check_slot(
                    &mut failures,
                    component.component,
                    "input_seed",
                    "scalar_values".to_owned(),
                    seed.slots.scalar_values,
                    requirements.scalar_words,
                );
                check_slot(
                    &mut failures,
                    component.component,
                    "input_seed",
                    "output_pointers".to_owned(),
                    seed.slots.output_pointers,
                    requirements.output_pointer_words,
                );
                for (ordinal, (&id, &words)) in seed
                    .slots
                    .consumer_input_columns
                    .iter()
                    .zip(&requirements.consumer_input_column_words)
                    .enumerate()
                {
                    check_slot(
                        &mut failures,
                        component.component,
                        "input_seed",
                        format!("input_column[{ordinal}]"),
                        id,
                        words,
                    );
                }
            }
            if let Some(compact) = &component.input_compact {
                let requirements = &compact.requirements;
                let scratch = [
                    (
                        "source_pointers",
                        compact.slots.source_pointers,
                        requirements.source_pointer_words,
                    ),
                    (
                        "descriptors",
                        compact.slots.descriptors,
                        requirements.descriptor_words,
                    ),
                    (
                        "output_pointers",
                        compact.slots.output_pointers,
                        requirements.output_pointer_words,
                    ),
                    (
                        "tuple_scratch",
                        compact.slots.tuple_scratch,
                        requirements.tuple_scratch_words,
                    ),
                    (
                        "sort_keys_a",
                        compact.slots.sort_keys_a,
                        requirements.sort_key_words,
                    ),
                    (
                        "sort_keys_b",
                        compact.slots.sort_keys_b,
                        requirements.sort_key_words,
                    ),
                    (
                        "sort_indices_a",
                        compact.slots.sort_indices_a,
                        requirements.sort_index_words,
                    ),
                    (
                        "sort_indices_b",
                        compact.slots.sort_indices_b,
                        requirements.sort_index_words,
                    ),
                    ("run_heads", compact.slots.run_heads, requirements.run_words),
                    (
                        "run_positions",
                        compact.slots.run_positions,
                        requirements.run_words,
                    ),
                    ("n_unique", compact.slots.n_unique, 1),
                    (
                        "sort_temp",
                        compact.slots.sort_temp,
                        requirements.sort_temp_words,
                    ),
                    (
                        "scan_temp",
                        compact.slots.scan_temp,
                        requirements.scan_temp_words,
                    ),
                ];
                for (label, id, words) in scratch {
                    check_slot(
                        &mut failures,
                        component.component,
                        "input_compact",
                        label.to_owned(),
                        id,
                        words,
                    );
                }
                for (ordinal, (&id, &words)) in compact
                    .slots
                    .consumer_input_columns
                    .iter()
                    .zip(&requirements.consumer_input_column_words)
                    .enumerate()
                {
                    check_slot(
                        &mut failures,
                        component.component,
                        "input_compact",
                        format!("input_column[{ordinal}]"),
                        id,
                        words,
                    );
                }
                check_sources(
                    &mut failures,
                    component.component,
                    "input_compact",
                    &compact.sources,
                    &requirements.edges,
                );
            }
        }
        assert!(
            failures.is_empty(),
            "witness input slots would fail resident prepare on hardware:\n{}",
            failures.join("\n")
        );
    }

    fn pcs(lifting_log_size: Option<u32>) -> PcsConfig {
        PcsConfig {
            pow_bits: 0,
            fri_config: FriConfig::new(1, 2, 13, 1),
            lifting_log_size,
        }
    }

    #[test]
    fn lifting_is_exact_and_rejects_an_undersized_override() {
        assert_eq!(lifting_log_size_from_max(20, pcs(None)).unwrap(), 22);
        assert_eq!(lifting_log_size_from_max(20, pcs(Some(24))).unwrap(), 24);
        assert!(matches!(
            lifting_log_size_from_max(20, pcs(Some(21))),
            Err(ResidentSessionError::InvalidLiftingLogSize {
                lifting: 21,
                required: 22
            })
        ));
    }

    #[test]
    fn fixed_twiddle_tree_covers_the_tallest_commitment_domain() {
        assert_eq!(max_twiddle_log_size(24, [21, 24, 23]), 24);
        assert_eq!(max_twiddle_log_size(24, [27, 24, 23]), 27);
    }

    #[test]
    fn telemetry_reports_cache_and_all_staging_calls() {
        let telemetry = ResidentSessionTelemetry {
            workspace_materialization: Some(WorkspaceMaterialization::Reused),
            base: ResidentSourceStageReport {
                d2d_bytes: 40,
                d2d_copies: 3,
                ..Default::default()
            },
            lookups: ResidentLookupStageReport {
                staged_bytes: 24,
                host_copies: 2,
                device_copies: 4,
                fill_calls: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(telemetry.cache_hit());
        assert_eq!(telemetry.staged_bytes().unwrap(), 64);
        assert_eq!(telemetry.staged_copies().unwrap(), 10);
    }

    #[test]
    fn strict_telemetry_admits_only_the_prepared_replacement_schedule() {
        let policy = ProtocolPlanPolicy::replacement_v1(0x1234, 2048);
        let valid = ResidentSessionTelemetry {
            shape_executable_topology_digest: Some([7; 32]),
            workspace_key: Some(WorkspaceKey::new(
                stwo_cairo_prover::witness::proof_shape::ProofShapeKey(9),
                11,
            )),
            protocol_policy: Some(policy),
            prepared_numerator_schedule: Some(PreparedNumeratorSchedule::HybridCandidate {
                eligible_groups: 18,
                legacy_groups: 1,
            }),
            execution_tables_ingest: Some(PreparedExecutionTablesIngestTelemetry {
                compact_h2d_bytes: 0,
                compact_h2d_copies: 0,
                descriptor_h2d_bytes: 0,
                descriptor_h2d_copies: 0,
                sync_calls: 1,
            }),
            ..ResidentSessionTelemetry::default()
        };
        assert!(valid.require_strict_graph_a().is_ok());

        let mut wrong_schedule = valid.clone();
        wrong_schedule.prepared_numerator_schedule = Some(PreparedNumeratorSchedule::LegacyBatches);
        assert!(wrong_schedule.require_strict_graph_a().is_err());

        let mut drifted_policy = valid;
        drifted_policy
            .protocol_policy
            .as_mut()
            .unwrap()
            .retained_lde_budget_bytes -= 1;
        assert!(drifted_policy.require_strict_graph_a().is_err());
    }

    #[test]
    fn strict_base_gate_accepts_only_zero_migration_witnesses() {
        require_device_born_base(BaseTraceResidency {
            columns: 11,
            direct_columns: 11,
            migrated_columns: 0,
        })
        .unwrap();
        assert!(matches!(
            require_device_born_base(BaseTraceResidency {
                columns: 11,
                direct_columns: 10,
                migrated_columns: 1,
            }),
            Err(ResidentSessionError::DetachedBaseWitness {
                migrated_columns: 1
            })
        ));
    }

    #[test]
    fn multi_edge_route_requires_exact_order_and_device_generated_tail() {
        let requirements = witness_input_gather_requirements(
            &[
                WitnessInputGatherEdge {
                    producer_rows: 16,
                    word_base: 282,
                    words_per_instance: 10,
                    n_instances: 2,
                },
                WitnessInputGatherEdge {
                    producer_rows: 16,
                    word_base: 1,
                    words_per_instance: 10,
                    n_instances: 3,
                },
            ],
            true,
            true,
        )
        .unwrap();
        let producers = ["poseidon_aggregator", "poseidon_3_partial_rounds_chain"];
        let edge = |producer, source_word, n_instances| DeviceEdgeColumn {
            producer,
            source_kind: DeviceEdgeSourceKind::SubcomponentWords,
            source_word,
            words_per_instance: 10,
            n_instances,
        };
        let exact = RecordedInputColumnProvenance::DeviceGather(DeviceGatherColumn::Data {
            edges: vec![
                edge("poseidon_aggregator", 285, 2),
                edge("poseidon_3_partial_rounds_chain", 4, 3),
            ],
        });
        assert!(recorded_input_matches_gather(
            &exact,
            3,
            &producers,
            &requirements,
        ));

        let reversed = RecordedInputColumnProvenance::DeviceGather(DeviceGatherColumn::Data {
            edges: vec![
                edge("poseidon_3_partial_rounds_chain", 4, 3),
                edge("poseidon_aggregator", 285, 2),
            ],
        });
        assert!(!recorded_input_matches_gather(
            &reversed,
            3,
            &producers,
            &requirements,
        ));
        assert!(recorded_input_matches_gather(
            &RecordedInputColumnProvenance::DeviceGather(DeviceGatherColumn::Enabler),
            10,
            &producers,
            &requirements,
        ));
        assert!(recorded_input_matches_gather(
            &RecordedInputColumnProvenance::DeviceGather(DeviceGatherColumn::Iota),
            11,
            &producers,
            &requirements,
        ));
        assert!(!recorded_input_matches_gather(
            &RecordedInputColumnProvenance::Host(vec![1; requirements.consumer_rows]),
            10,
            &producers,
            &requirements,
        ));
    }
}
