//! Production hand-off from a sealed Cairo witness to one resident CUDA runtime.
//!
//! The runtime borrows its stable arena, so it is deliberately scoped to a
//! callback while the exact workspace remains owned by the cache. This avoids a
//! self-referential session object and makes every failure leave the cache in a
//! valid, reusable state.

use std::sync::Arc;

use cairo_air::claims::CairoClaim;
use cairo_air::relations::CommonLookupElements;
use num_traits::Zero;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::PcsConfig;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_backend_cuda::{
    CudaBackend, ExecutionTablesHostData, PreparedEcOpIngestTelemetry,
    PreparedExecutionTablesIngestTelemetry, RelationChallenges, WitnessInputGatherRequirements,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
use stwo_cairo_prover::witness::exec_context::WitnessResidencyReport;
use stwo_cairo_prover::witness::relation_sources::RelationSourceError;

use crate::arena_plan::{ArenaPlanError, ExecutionTableGeometry, ProofArenaPlan};
use crate::composition_plan::{plan_cairo_composition, CompositionPlan, CompositionPlanError};
use crate::graphs::GraphWorkspace;
use crate::plan::{ProofPlan, ProofPlanError};
use crate::protocol_discovery::{
    discover_protocol_transcript_shape, schema_zero_interaction_claim_for_composition,
    ProtocolDiscoveryError, ProtocolTranscriptDiscovery,
};
use crate::protocol_plan::{plan_protocol_geometry, ProtocolPlanError, ProtocolPlanPolicy};
use crate::recorded_witness_inputs::{
    recorded_witness_inputs_for_plan, DeviceCompactColumn, DeviceEdgeColumn, DeviceEdgeSourceKind,
    DeviceGatherColumn, DeviceNativeColumn, DeviceSeedColumn, PlannedRecordedWitnessInputs,
    RecordedInputColumnProvenance, RecordedWitnessPlanError,
};
use crate::resident_runtime::{
    ResidentGraphRuntime, ResidentRuntimeError, ResidentWitnessIngestReport, ResidentWitnessInput,
    ResidentWitnessInputColumn, ResidentWorkspaceIdentity,
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
use crate::state::{DeviceProofState, WitnessOutput};
use crate::transcript_plan::{
    encode_static_transcript_inputs, plan_cairo_blake2s_transcript, CairoBlake2sTranscriptPlan,
    TranscriptPlanError,
};
use crate::workspace_cache::{
    WorkspaceCache, WorkspaceCacheError, WorkspaceCacheTelemetry, WorkspaceKey,
    WorkspaceMaterialization,
};

/// Everything whose value changes the resident graph or its stable pointers.
pub struct ResidentSessionRequest {
    pub preprocessed_trace: Arc<PreProcessedTrace>,
    pub witness: WitnessOutput<CudaBackend>,
    pub channel_salt: u32,
    pub pcs: PcsConfig,
    pub include_all_preprocessed_columns: bool,
    pub twiddles: &'static TwiddleTree<CudaBackend>,
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
    pub twiddles: &'static TwiddleTree<CudaBackend>,
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
    pub workspace_key: Option<WorkspaceKey>,
    pub workspace_materialization: Option<WorkspaceMaterialization>,
    pub cache: WorkspaceCacheTelemetry,
    pub arena_words: usize,
    pub transcript_segments: usize,
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
    RecordedPedersenTableUnavailable,
    RecordedWitnessInputRoute {
        component: &'static str,
        ordinal: usize,
    },
    StrictArchitectureTelemetry(&'static str),
    PlannedClaimMismatch,
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

struct PlannedResidentProtocol {
    discovery: ProtocolTranscriptDiscovery,
    transcript: CairoBlake2sTranscriptPlan,
    composition: CompositionPlan,
    arena: Arc<ProofArenaPlan>,
    workspace_key: WorkspaceKey,
}

fn plan_resident_protocol(
    claim: &CairoClaim,
    proof_plan: &ProofPlan,
    preprocessed_trace: &PreProcessedTrace,
    pcs: PcsConfig,
    include_all_preprocessed_columns: bool,
    execution_table_geometry: Option<ExecutionTableGeometry>,
) -> Result<PlannedResidentProtocol, ResidentSessionError> {
    let protocol_policy = ProtocolPlanPolicy::loaded_starknet_blake2s()?;
    plan_resident_protocol_with_policy(
        claim,
        proof_plan,
        preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        execution_table_geometry,
        protocol_policy,
    )
}

/// [`plan_resident_protocol`] with the protocol-plan policy supplied by the
/// caller instead of resolved from the embedded AOT pack. The proving sessions
/// always go through [`plan_resident_protocol`] (loaded manifest, fail-closed);
/// this seam exists for the host-only preflight planner, which reuses the exact
/// planning path on machines whose binary carries no AOT pack (the manifest
/// hash never feeds arena geometry — see
/// `poseidon_fixture_recorded_lanes_match_arena_witness_plan`).
fn plan_resident_protocol_with_policy(
    claim: &CairoClaim,
    proof_plan: &ProofPlan,
    preprocessed_trace: &PreProcessedTrace,
    pcs: PcsConfig,
    include_all_preprocessed_columns: bool,
    execution_table_geometry: Option<ExecutionTableGeometry>,
    protocol_policy: ProtocolPlanPolicy,
) -> Result<PlannedResidentProtocol, ResidentSessionError> {
    let lifting_log_size = resident_lifting_log_size(claim, pcs)?;
    let discovery = discover_protocol_transcript_shape(
        claim,
        proof_plan,
        preprocessed_trace,
        &pcs,
        lifting_log_size,
        include_all_preprocessed_columns,
    )?;
    let transcript = plan_cairo_blake2s_transcript(
        claim,
        pcs,
        discovery.lifting_log_size,
        discovery.dynamic_transcript_shape(),
    )?;
    let zero_interaction_claim = schema_zero_interaction_claim_for_composition(claim)?;
    let composition = plan_cairo_composition(
        claim,
        &CommonLookupElements::dummy(),
        &zero_interaction_claim,
        &preprocessed_trace.ids(),
        protocol_policy.composition_max_kernel_instrs,
    )?;
    let protocol = plan_protocol_geometry(
        proof_plan,
        claim,
        preprocessed_trace,
        &pcs,
        include_all_preprocessed_columns,
        protocol_policy,
        &transcript,
        &discovery,
        &composition,
    )?;
    let arena = Arc::new(match execution_table_geometry {
        Some(geometry) => ProofArenaPlan::build_with_execution_tables(
            proof_plan,
            &protocol,
            &composition,
            geometry,
        )?,
        None => ProofArenaPlan::build(proof_plan, &protocol, &composition)?,
    });
    let workspace_key = WorkspaceKey::from_plan(&arena);
    Ok(PlannedResidentProtocol {
        discovery,
        transcript,
        composition,
        arena,
        workspace_key,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_materialized_session<R>(
    workspace: &mut GraphWorkspace,
    workspace_materialization: WorkspaceMaterialization,
    preprocessed_trace: Arc<PreProcessedTrace>,
    witness: WitnessOutput<CudaBackend>,
    channel_salt: u32,
    pcs: PcsConfig,
    twiddles: &'static TwiddleTree<CudaBackend>,
    planned: &PlannedResidentProtocol,
    require_device_born: bool,
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
    let base = stage_base_trace_coefficients(workspace, &proof_plan, trace, twiddles)?;
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
        RelationChallenges {
            alpha_powers: &setup_alphas,
            z: SecureField::zero(),
        },
        &planned.transcript,
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
    let telemetry = ResidentSessionTelemetry {
        workspace_key: Some(planned.workspace_key),
        workspace_materialization: Some(workspace_materialization),
        cache: WorkspaceCacheTelemetry::default(),
        arena_words: workspace.plan().total_words(),
        transcript_segments: planned.transcript.segments().len(),
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
    };
    let result = run(
        &mut runtime,
        ResidentSessionArtifacts {
            claim: &claim,
            proof_plan: &proof_plan,
            discovery: &planned.discovery,
            transcript_plan: &planned.transcript,
            composition_plan: &planned.composition,
            telemetry: &telemetry,
        },
    )?;
    Ok((result, telemetry))
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

/// Compatibility hand-off for callers that already own a sealed witness. It
/// still enforces arena-born base columns; a detached legacy witness is rejected
/// before staging. New callers use [`with_resident_session_from_generator`].
pub fn with_resident_session<R>(
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
        twiddles,
    } = request;
    let planned = plan_resident_protocol(
        &witness.claim,
        &witness.device.proof_plan,
        &preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        None,
    )?;
    let (result, mut telemetry) = {
        let (workspace, materialization) =
            cache.materialize_or_reuse(Arc::clone(&planned.arena))?;
        run_materialized_session(
            workspace,
            materialization,
            preprocessed_trace,
            witness,
            channel_salt,
            pcs,
            twiddles,
            &planned,
            true,
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
        twiddles,
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
    let planned = plan_resident_protocol(
        &planned_claim,
        &exact_plan,
        &preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        Some(ExecutionTableGeometry::new(
            memory.address_to_id.len(),
            memory.f252_values.len(),
            memory.small_values.len(),
        )),
    )?;

    if recorded
        .lanes
        .iter()
        .any(|lane| lane.tables.host_pedersen_points_18)
        && !stwo_cairo_prover::witness::jit_prove_backend::ensure_device_pedersen_table()
    {
        return Err(ResidentSessionError::RecordedPedersenTableUnavailable);
    }
    let raw_address_to_id = memory
        .address_to_id
        .iter()
        .map(|encoded| encoded.0)
        .collect::<Vec<_>>();
    let host_columns = resident_host_witness_inputs(&recorded, &planned.arena)?;
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
        let (workspace, materialization) =
            cache.materialize_or_reuse(Arc::clone(&planned.arena))?;
        let twiddle_report = stage_protocol_twiddles(workspace, twiddles)?;
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
            RelationChallenges {
                alpha_powers: &setup_alphas,
                z: SecureField::zero(),
            },
            &planned.transcript,
            Some(ExecutionTablesHostData {
                addr_to_id: &raw_address_to_id,
                f252_values: &memory.f252_values,
                small_values: &memory.small_values,
            }),
            ec_op_segment_start,
            Some(Arc::clone(&preprocessed_trace)),
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
        let telemetry = ResidentSessionTelemetry {
            workspace_key: Some(planned.workspace_key),
            workspace_materialization: Some(materialization),
            cache: WorkspaceCacheTelemetry::default(),
            arena_words: workspace.plan().total_words(),
            transcript_segments: planned.transcript.segments().len(),
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
        };
        let result = run(
            &mut runtime,
            ResidentSessionArtifacts {
                claim: &planned_claim,
                proof_plan: &exact_plan,
                discovery: &planned.discovery,
                transcript_plan: &planned.transcript,
                composition_plan: &planned.composition,
                telemetry: &telemetry,
            },
        )?;
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
    pub manifest_policy: PreflightManifestPolicy,
}

/// Plan the strict resident session end-to-end WITHOUT touching CUDA: the same
/// fail-closed pipeline as [`with_resident_session_from_generator`] up to (and
/// including) the full arena plan — ingest artifacts in, exact plan, strict
/// witness coverage, planned claim, recorded witness inputs (`require_resolved`),
/// Graph-A multiplicity plan, protocol/arena plan. Any `Err` is byte-for-byte
/// the error the session would fail closed with on hardware. Dev tooling only
/// (`arena_preflight`); proving sessions never call this.
pub fn plan_resident_preflight(
    generator: &CairoClaimGenerator,
    capacity_plan: &ProofPlan,
    preprocessed_trace: &PreProcessedTrace,
    pcs: PcsConfig,
    include_all_preprocessed_columns: bool,
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
    let multiplicities = crate::multiplicity_pipeline::plan_graph_a_multiplicities(&exact_plan)?;

    let (protocol_policy, manifest_policy) = match ProtocolPlanPolicy::loaded_starknet_blake2s() {
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
            let policy = ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048);
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
    let planned = plan_resident_protocol_with_policy(
        &planned_claim,
        &exact_plan,
        preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        Some(ExecutionTableGeometry::new(
            memory.address_to_id.len(),
            memory.f252_values.len(),
            memory.small_values.len(),
        )),
        protocol_policy,
    )?;

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
        arena: planned.arena,
        transcript_segments: planned.transcript.segments().len(),
        manifest_policy,
    })
}

#[cfg(test)]
mod tests {
    use stwo::core::fri::FriConfig;
    use stwo_backend_cuda::{witness_input_gather_requirements, WitnessInputGatherEdge};

    use super::*;

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
        // The strict resolution fails closed on the compacted consumers'
        // capacity bounds; this lane-order test substitutes the capacity
        // geometry explicitly (see resolve_compacted_capacity_for_test).
        let capacity_for_test = crate::plan::resolve_compacted_capacity_for_test(
            &ingest.proof_plan,
            &crate::schedule_table::CAIRO_SCHEDULE,
            &crate::relation_table::CAIRO_RELATION_GRAPH,
        );
        let exact_plan = capacity_for_test
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

        // `plan_resident_protocol` itself is manifest-blocked off-CUDA (the
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
        let arena = ProofArenaPlan::build_with_execution_tables(
            &exact_plan,
            &protocol,
            &composition,
            ExecutionTableGeometry::new(
                memory.address_to_id.len(),
                memory.f252_values.len(),
                memory.small_values.len(),
            ),
        )
        .unwrap();
        assert_witness_input_slots_satisfy_prepare(&arena);

        // Parity with the packaged preflight: `plan_resident_preflight` (the
        // pipeline the `arena_preflight` binary runs) must reproduce this
        // manual plan exactly — same fake-policy fallback off-CUDA, same
        // lanes, same coverage verdicts, same arena geometry — so the JSON
        // the tool prints for an SN-scale input is the plan the H100 session
        // would materialize.
        let preflight = plan_resident_preflight(
            &ingest.generator,
            &capacity_for_test,
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

    /// Host mirror of the slot validation in stwo's
    /// `PreparedWitnessInput{Gather,Seed,Compact}::prepare` and the recorded
    /// host-column ingest: every workspace slot handed to the witness-input
    /// kernels is `DeviceArena::bind`-ed as a WHOLE physical slot. The arena
    /// pools disjoint-lifetime buffers into one slot sized to the largest
    /// sharer, so the contract is CAPACITY — each slot must hold at least the
    /// kernel-ABI requirement recomputed at runtime (the kernels touch exactly
    /// the required extent and never the pooled surplus). Any shortfall here
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
