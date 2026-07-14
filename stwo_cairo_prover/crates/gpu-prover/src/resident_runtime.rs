//! Fail-closed binding and replay for the arena-backed CUDA proof primitives.
//!
//! This module does not infer Cairo trace order. Commitment sources come from
//! the plan's canonical identities; generated relation layouts bind their own
//! trace/lookup columns. OODS sampling, numerator construction and quotient-to-
//! FRI binding are one checked resident pipeline.

use core::ffi::c_void;
use std::collections::BTreeMap;
use std::sync::Arc;

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo_backend_cuda::{
    ArenaError, ArenaSlice, ArenaSlotId, Blake2sProofAssemblyShape, CommitEvaluationGroup,
    CudaExecTelemetry, CudaRuntimeError, DecommitAssembly, DecommitColumnSource,
    DecommitTreeGeometry, DecommitTreeSources, DeviceTranscriptError, ExecutionTablesHostData,
    FixedTableSourceColumn, FriDecommitOwnedSources, MemoryBaseTracePart,
    ModeAwareCommitWorkspaceRequirements, ModeAwareCommitWorkspaceSlots, PreparedBlake2sPowError,
    PreparedBlake2sPowGraph, PreparedBlake2sTranscript, PreparedCommitError, PreparedCommitGraph,
    PreparedDecommitError, PreparedDecommitGraph, PreparedEcOpError, PreparedEcOpGraph,
    PreparedEcOpIngestTelemetry, PreparedExecutionTablesError, PreparedExecutionTablesGraph,
    PreparedExecutionTablesIngestTelemetry, PreparedFixedTableError, PreparedFixedTableGraph,
    PreparedFriError, PreparedFriFinalError, PreparedFriFinalGraph, PreparedFriGraph,
    PreparedInterpolationError, PreparedInterpolationGraph, PreparedMemoryBaseTraceError,
    PreparedMemoryBaseTraceGraph, PreparedNumeratorSchedule, PreparedProgressiveCommitError,
    PreparedProgressiveCommitGraph, PreparedRelationGraph, PreparedWitnessError,
    PreparedWitnessFeedClearGraph, PreparedWitnessFeedError, PreparedWitnessFeedGraph,
    PreparedWitnessGraph, PreparedWitnessInputCompactGraph, PreparedWitnessInputGatherError,
    PreparedWitnessInputGatherGraph, PreparedWitnessInputSeedGraph, PreparedWitnessMode,
    RelationChallenges, RelationGraphError, RelationInstanceSources, TraceDecommitSources,
    TraceSourceGroup, TranscriptInputBinding, TranscriptInputId, TranscriptMirrorReport,
    TranscriptOutputBinding, TranscriptOutputId, TranscriptSegmentCursor, TranscriptSegmentStart,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
use stwo_cairo_prover::witness::device_feed::canonical_count_lut;
use stwo_cairo_prover::witness::proof_shape::{ProofShapeKey, TracePartId};

use crate::arena_plan::{
    BufferPurpose, CommitmentColumnSource, CommitmentTreeId, PlannedFixedTableSource,
};
use crate::composition_plan::CompositionProofBindings;
use crate::fixed_table_materializer::PEDERSEN_POINTS_18_ROW_COUNT;
use crate::graphs::{
    bind_arena_binding, GraphCaptureStatus, GraphError, GraphSegment, GraphWorkspace,
};
use crate::multiplicity_pipeline::{FixedMultiplicityCoverageGap, MultiplicityFeedBlocker};
use crate::proof_bundle::{
    ResidentProofBundle, ResidentProofBundleError, ResidentProofBundleLayout,
};
use crate::relation::RelationTracePart;
use crate::relation_execution::{RelationBatchKey, RelationSourcePlane};
use crate::resident_composition::{prepare_resident_composition, ResidentCompositionError};
use crate::resident_oods::{ResidentOodsError, ResidentOodsPipeline};
use crate::resident_sources::{
    commitment_groups, prepare_commitment_interpolation, ResidentSourceStageError,
};
use crate::schedule_table::CAIRO_SCHEDULE;
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptInput, CairoTranscriptOutput,
    CairoTranscriptSegment, TranscriptPlanError, TranscriptSegmentPlan,
};
use crate::{PreparedCompositionError, PreparedCompositionGraph};

/// Complete cache identity of one materialized resident workspace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentWorkspaceIdentity {
    pub shape_key: ProofShapeKey,
    pub protocol_key: u64,
    pub arena_base: usize,
}

impl ResidentWorkspaceIdentity {
    pub fn of(workspace: &GraphWorkspace) -> Self {
        Self {
            shape_key: workspace.plan().shape_key,
            protocol_key: workspace.plan().protocol_key,
            arena_base: workspace.arena().base_ptr().as_ptr() as usize,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentClaimedSum {
    pub batch: RelationBatchKey,
    pub instance_index: usize,
    pub value: SecureField,
}

/// The only interaction D2H boundary: one root plus four words per relation
/// instance, drained with a single stream synchronization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InteractionTranscriptBoundary {
    pub root: Blake2sHash,
    pub claimed_sums: Vec<ResidentClaimedSum>,
}

struct ResidentProofBundleSources {
    commitments: [ArenaSlice; 4],
    interaction_claim: ArenaSlice,
    interaction_pow: ArenaSlice,
    sampled_values: ArenaSlice,
    fri_commitments: Vec<ArenaSlice>,
    final_line_poly: ArenaSlice,
    query_pow: ArenaSlice,
    decommitment: ArenaSlice,
}

pub struct ResidentWitnessInput<'a> {
    pub component: &'static str,
    pub columns: &'a [ResidentWitnessInputColumn<'a>],
    pub seed_scalars: Option<&'a [u32]>,
}

pub struct ResidentWitnessInputColumn<'a> {
    pub ordinal: usize,
    pub words: &'a [u32],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentWitnessIngestReport {
    pub components: usize,
    pub columns: usize,
    pub h2d_bytes: usize,
    pub h2d_copies: usize,
    pub sync_calls: usize,
}

/// Machine-checkable host-boundary budget for one warm resident replay. Setup,
/// compact-input ingest and the final proof copy are measured separately; the
/// transcript-bounded graph hot path itself must not cross PCIe or synchronize.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentHotPathBudget {
    pub expected_graph_launches: u64,
    /// Exact kernel-node total enumerated from the captured CUDA graphs. Graph-only
    /// callers that do not yet have a complete capture may leave this unbounded.
    pub expected_kernel_launches: Option<u64>,
    pub expected_sync_calls: u64,
    pub max_h2d_bytes: u64,
    pub expected_d2h_bytes: u64,
    pub max_allocations: u64,
    pub max_frees: u64,
    pub max_graph_submit_gap_ns: u64,
}

impl ResidentHotPathBudget {
    pub const fn graph_only(expected_graph_launches: u64) -> Self {
        Self {
            expected_graph_launches,
            expected_kernel_launches: None,
            expected_sync_calls: 0,
            max_h2d_bytes: 0,
            expected_d2h_bytes: 0,
            max_allocations: 0,
            max_frees: 0,
            max_graph_submit_gap_ns: u64::MAX,
        }
    }

    pub const fn final_bundle(
        expected_graph_launches: u64,
        expected_kernel_launches: u64,
        d2h_bytes: u64,
    ) -> Self {
        Self {
            expected_graph_launches,
            expected_kernel_launches: Some(expected_kernel_launches),
            expected_sync_calls: 1,
            max_h2d_bytes: 0,
            expected_d2h_bytes: d2h_bytes,
            max_allocations: 0,
            max_frees: 0,
            max_graph_submit_gap_ns: 49_999_999,
        }
    }

    fn accepts(self, actual: CudaExecTelemetry) -> bool {
        actual.graph_launches == self.expected_graph_launches
            && actual.graph_launches != 0
            && actual.graph_launches < 100
            && actual.kernel_launches != 0
            && self
                .expected_kernel_launches
                .is_none_or(|expected| actual.kernel_launches == expected)
            && actual.sync_calls == self.expected_sync_calls
            && actual.h2d_bytes <= self.max_h2d_bytes
            && actual.d2h_bytes == self.expected_d2h_bytes
            && actual.allocations <= self.max_allocations
            && actual.allocation_bytes == 0
            && actual.frees <= self.max_frees
            && actual.memset_bytes == 0
            && actual.fill_words == 0
            && actual.d2d_bytes == 0
            && actual.capture_begins == 0
            && actual.capture_finishes == 0
            && actual.capture_aborts == 0
            && actual.lane_forks == 0
            && actual.lane_joins == 0
            && actual.graph_submit_gap_ns_max <= self.max_graph_submit_gap_ns
    }
}

#[derive(Debug)]
pub enum ResidentRuntimeError {
    WorkspaceIdentityMismatch {
        expected: ResidentWorkspaceIdentity,
        actual: ResidentWorkspaceIdentity,
    },
    SourceOutsideArena {
        slot: stwo_backend_cuda::ArenaSlotId,
    },
    MissingCommitmentSource {
        id: CommitmentTreeId,
        source: CommitmentColumnSource,
    },
    MissingRelationSource {
        batch: RelationBatchKey,
        instance_index: usize,
        ordinal: u32,
    },
    MissingPreparedCommitment(CommitmentTreeId),
    FixedPreprocessedCommitmentNotReady,
    TranscriptScheduleMismatch {
        expected: u64,
        actual: u64,
    },
    TranscriptRequirementsMismatch,
    MissingTranscriptInput(TranscriptInputId),
    MissingTranscriptOutput(TranscriptOutputId),
    InvalidTranscriptSegment(usize),
    MissingTranscriptSegment(CairoTranscriptSegment),
    TranscriptBindingTooSmall {
        role: &'static str,
        required_words: usize,
        actual_words: usize,
    },
    TranscriptClaimWidthMismatch {
        expected_words: usize,
        actual_words: usize,
    },
    UnknownTranscriptRelationComponent(&'static str),
    WitnessInputCoverage {
        expected: usize,
        actual: usize,
    },
    PreparedWitnessCoverage {
        expected: usize,
        actual: usize,
    },
    PreparedWitnessCaptureContract {
        component: &'static str,
        role: &'static str,
    },
    PreparedFixedTableCoverage {
        expected: usize,
        actual: usize,
    },
    PreparedFixedTableCaptureContract {
        component: &'static str,
        role: &'static str,
    },
    RegisteredPedersenTableUnavailable,
    RegisteredPedersenTableRows {
        expected: usize,
        actual: usize,
    },
    RegisteredPedersenTableColumn(usize),
    IncompleteFixedMultiplicityCoverage(Vec<FixedMultiplicityCoverageGap>),
    UnsupportedMultiplicityFeeds(Vec<MultiplicityFeedBlocker>),
    MissingPreprocessedTraceForMultiplicity,
    CanonicalMultiplicityLut(&'static str),
    PublicMemoryMultiplicitySeed(&'static str),
    MissingPreparedExecutionTables,
    UnexpectedPreparedExecutionTables,
    MissingPreparedEcOpSegment,
    UnexpectedPreparedEcOpSegment,
    PreparedEcOpCoverage(&'static str),
    DuplicateWitnessInput(&'static str),
    DuplicateWitnessInputColumn {
        component: &'static str,
        ordinal: usize,
    },
    UnexpectedGatheredWitnessHostInput(&'static str),
    MissingPreparedWitness(&'static str),
    WitnessInputColumnCount {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    WitnessInputRowCount {
        component: &'static str,
        column: usize,
        expected: usize,
        actual: usize,
    },
    StaleRelationChallenges,
    StaleFriChallenge(usize),
    HotPathBudgetExceeded {
        budget: ResidentHotPathBudget,
        actual: CudaExecTelemetry,
    },
    CapturedGraphTopology {
        fri_rounds: usize,
        transcript_segments: usize,
        expected: usize,
        actual: usize,
    },
    FriRoundOutOfOrder {
        expected: usize,
        actual: usize,
    },
    FriRoundIndexTooLarge(usize),
    SizeOverflow,
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
    Graph(GraphError),
    Commit(PreparedCommitError),
    ProgressiveCommit(PreparedProgressiveCommitError),
    CommitModeMismatch,
    Interpolation(PreparedInterpolationError),
    SourceStage(ResidentSourceStageError),
    CompositionBinding(ResidentCompositionError),
    Composition(PreparedCompositionError),
    Oods(ResidentOodsError),
    Fri(PreparedFriError),
    FriFinal(PreparedFriFinalError),
    Pow(PreparedBlake2sPowError),
    Decommit(PreparedDecommitError),
    ProofBundle(ResidentProofBundleError),
    DecommitTopologyMismatch(&'static str),
    Relation(RelationGraphError),
    ExecutionTables(PreparedExecutionTablesError),
    EcOp(PreparedEcOpError),
    WitnessInputGather(PreparedWitnessInputGatherError),
    Witness(PreparedWitnessError),
    WitnessFeed(PreparedWitnessFeedError),
    FixedTable(PreparedFixedTableError),
    MemoryBaseTrace(PreparedMemoryBaseTraceError),
    DeviceTranscript(DeviceTranscriptError),
    TranscriptPlan(TranscriptPlanError),
}

impl core::fmt::Display for ResidentRuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident CUDA runtime rejected proof binding: {self:?}")
    }
}

impl std::error::Error for ResidentRuntimeError {}

impl From<GraphError> for ResidentRuntimeError {
    fn from(value: GraphError) -> Self {
        Self::Graph(value)
    }
}

impl From<ArenaError> for ResidentRuntimeError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for ResidentRuntimeError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

impl From<PreparedCommitError> for ResidentRuntimeError {
    fn from(value: PreparedCommitError) -> Self {
        Self::Commit(value)
    }
}

impl From<PreparedProgressiveCommitError> for ResidentRuntimeError {
    fn from(value: PreparedProgressiveCommitError) -> Self {
        Self::ProgressiveCommit(value)
    }
}

impl From<PreparedInterpolationError> for ResidentRuntimeError {
    fn from(value: PreparedInterpolationError) -> Self {
        Self::Interpolation(value)
    }
}

impl From<ResidentSourceStageError> for ResidentRuntimeError {
    fn from(value: ResidentSourceStageError) -> Self {
        Self::SourceStage(value)
    }
}

impl From<ResidentCompositionError> for ResidentRuntimeError {
    fn from(value: ResidentCompositionError) -> Self {
        Self::CompositionBinding(value)
    }
}

impl From<PreparedCompositionError> for ResidentRuntimeError {
    fn from(value: PreparedCompositionError) -> Self {
        Self::Composition(value)
    }
}

impl From<PreparedFriError> for ResidentRuntimeError {
    fn from(value: PreparedFriError) -> Self {
        Self::Fri(value)
    }
}

impl From<PreparedFriFinalError> for ResidentRuntimeError {
    fn from(value: PreparedFriFinalError) -> Self {
        Self::FriFinal(value)
    }
}

impl From<PreparedBlake2sPowError> for ResidentRuntimeError {
    fn from(value: PreparedBlake2sPowError) -> Self {
        Self::Pow(value)
    }
}

impl From<PreparedDecommitError> for ResidentRuntimeError {
    fn from(value: PreparedDecommitError) -> Self {
        Self::Decommit(value)
    }
}

impl From<ResidentProofBundleError> for ResidentRuntimeError {
    fn from(value: ResidentProofBundleError) -> Self {
        Self::ProofBundle(value)
    }
}

impl From<ResidentOodsError> for ResidentRuntimeError {
    fn from(value: ResidentOodsError) -> Self {
        Self::Oods(value)
    }
}

impl From<RelationGraphError> for ResidentRuntimeError {
    fn from(value: RelationGraphError) -> Self {
        Self::Relation(value)
    }
}

impl From<PreparedExecutionTablesError> for ResidentRuntimeError {
    fn from(value: PreparedExecutionTablesError) -> Self {
        Self::ExecutionTables(value)
    }
}

impl From<PreparedEcOpError> for ResidentRuntimeError {
    fn from(value: PreparedEcOpError) -> Self {
        Self::EcOp(value)
    }
}

impl From<PreparedWitnessError> for ResidentRuntimeError {
    fn from(value: PreparedWitnessError) -> Self {
        Self::Witness(value)
    }
}

impl From<PreparedWitnessInputGatherError> for ResidentRuntimeError {
    fn from(value: PreparedWitnessInputGatherError) -> Self {
        Self::WitnessInputGather(value)
    }
}

impl From<PreparedWitnessFeedError> for ResidentRuntimeError {
    fn from(value: PreparedWitnessFeedError) -> Self {
        Self::WitnessFeed(value)
    }
}

impl From<PreparedFixedTableError> for ResidentRuntimeError {
    fn from(value: PreparedFixedTableError) -> Self {
        Self::FixedTable(value)
    }
}

impl From<PreparedMemoryBaseTraceError> for ResidentRuntimeError {
    fn from(value: PreparedMemoryBaseTraceError) -> Self {
        Self::MemoryBaseTrace(value)
    }
}

impl From<DeviceTranscriptError> for ResidentRuntimeError {
    fn from(value: DeviceTranscriptError) -> Self {
        Self::DeviceTranscript(value)
    }
}

impl From<TranscriptPlanError> for ResidentRuntimeError {
    fn from(value: TranscriptPlanError) -> Self {
        Self::TranscriptPlan(value)
    }
}

#[derive(Debug)]
enum ResidentLaunchError {
    Commit(PreparedCommitError),
    ProgressiveCommit(PreparedProgressiveCommitError),
    Interpolation(PreparedInterpolationError),
    Composition(PreparedCompositionError),
    Oods(ResidentOodsError),
    Fri(PreparedFriError),
    FriFinal(PreparedFriFinalError),
    Pow(PreparedBlake2sPowError),
    Decommit(PreparedDecommitError),
    Relation(RelationGraphError),
    ExecutionTables(PreparedExecutionTablesError),
    WitnessFeed(PreparedWitnessFeedError),
    FixedTable(PreparedFixedTableError),
    MemoryBaseTrace(PreparedMemoryBaseTraceError),
    WitnessLanes(WitnessLaneLaunchError),
    Transcript(DeviceTranscriptError),
    Cuda(CudaRuntimeError),
    Binding(&'static str),
}

impl core::fmt::Display for ResidentLaunchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Commit(error) => write!(f, "resident commitment launch rejected: {error}"),
            Self::ProgressiveCommit(error) => {
                write!(
                    f,
                    "resident progressive commitment launch rejected: {error}"
                )
            }
            Self::Interpolation(error) => {
                write!(f, "resident interpolation launch rejected: {error}")
            }
            Self::Composition(error) => {
                write!(f, "resident composition launch rejected: {error}")
            }
            Self::Oods(error) => write!(f, "resident OODS/quotient launch rejected: {error}"),
            Self::Fri(error) => write!(f, "resident FRI launch rejected: {error}"),
            Self::FriFinal(error) => write!(f, "resident final FRI launch rejected: {error}"),
            Self::Pow(error) => write!(f, "resident Blake2s PoW launch rejected: {error}"),
            Self::Decommit(error) => write!(f, "resident decommit launch rejected: {error}"),
            Self::Relation(error) => write!(f, "resident relation launch rejected: {error}"),
            Self::ExecutionTables(error) => {
                write!(f, "resident execution-table launch rejected: {error}")
            }
            Self::WitnessFeed(error) => {
                write!(f, "resident witness multiplicity feed rejected: {error}")
            }
            Self::FixedTable(error) => {
                write!(f, "resident fixed-table materializer rejected: {error}")
            }
            Self::MemoryBaseTrace(error) => {
                write!(f, "resident memory base trace rejected: {error}")
            }
            Self::WitnessLanes(error) => write!(f, "resident witness lanes rejected: {error}"),
            Self::Transcript(error) => write!(f, "resident transcript launch rejected: {error}"),
            Self::Cuda(error) => write!(f, "resident CUDA handoff rejected: {error}"),
            Self::Binding(role) => write!(f, "resident transcript binding rejected: {role}"),
        }
    }
}

impl std::error::Error for ResidentLaunchError {}

impl From<ResidentLaunchError> for ResidentRuntimeError {
    fn from(value: ResidentLaunchError) -> Self {
        Self::Graph(GraphError::Enqueue(Box::new(value)))
    }
}

/// Prepared proof primitives bound to workspace-owned transcript subgraphs.
/// The workspace outlives the runtime and keeps exact-key graph executables
/// resident across proof sessions.
struct PreparedResidentWitness<'a> {
    component: &'static str,
    native_input_producer: Option<&'static str>,
    input_gather: Option<PreparedWitnessInputGatherGraph<'a>>,
    input_seed: Option<PreparedWitnessInputSeedGraph<'a>>,
    input_compact: Option<PreparedWitnessInputCompactGraph<'a>>,
    writer: PreparedWitnessGraph<'a>,
}

#[derive(Debug)]
enum WitnessLaneLaunchError {
    Cuda(CudaRuntimeError),
    EcOp(PreparedEcOpError),
    Input(PreparedWitnessInputGatherError),
    Writer(PreparedWitnessError),
    Feed(PreparedWitnessFeedError),
}

impl core::fmt::Display for WitnessLaneLaunchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for WitnessLaneLaunchError {}

impl From<WitnessLaneLaunchError> for ResidentRuntimeError {
    fn from(value: WitnessLaneLaunchError) -> Self {
        match value {
            WitnessLaneLaunchError::Cuda(error) => Self::Cuda(error),
            WitnessLaneLaunchError::EcOp(error) => Self::EcOp(error),
            WitnessLaneLaunchError::Input(error) => Self::WitnessInputGather(error),
            WitnessLaneLaunchError::Writer(error) => Self::Witness(error),
            WitnessLaneLaunchError::Feed(error) => Self::WitnessFeed(error),
        }
    }
}

/// Pack each dependency level onto fixed proof-owned streams. Long writers are
/// placed first on the least-loaded lane; ties are stable by component name.
/// The result is capture topology, not runtime scheduling policy.
fn pack_weighted_lane_level(
    mut components: Vec<(usize, u64, &'static str)>,
    lane_count: usize,
) -> Vec<Vec<usize>> {
    components.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.2.cmp(right.2)));
    let mut lanes = vec![Vec::new(); lane_count];
    let mut loads = vec![0u64; lane_count];
    for (component, work, _) in components {
        let lane = loads
            .iter()
            .enumerate()
            .min_by_key(|&(lane, &load)| (load, lane))
            .map(|(lane, _)| lane)
            .expect("lane count checked");
        loads[lane] = loads[lane].saturating_add(work);
        lanes[lane].push(component);
    }
    lanes
}

fn plan_witness_lane_levels(
    witness: &[PreparedResidentWitness<'_>],
    ec_op: Option<&PreparedEcOpGraph<'_>>,
    lane_count: usize,
) -> Result<Vec<Vec<Vec<usize>>>, ResidentRuntimeError> {
    if witness.is_empty() && ec_op.is_none() {
        return Ok(Vec::new());
    }
    if lane_count == 0 {
        return Err(ResidentRuntimeError::PreparedWitnessCaptureContract {
            component: witness
                .first()
                .map_or("ec_op_builtin", |entry| entry.component),
            role: "proof execution context has no component lanes",
        });
    }

    let mut by_component = witness
        .iter()
        .enumerate()
        .map(|(index, prepared)| (prepared.component, index))
        .collect::<BTreeMap<_, _>>();
    let native_ec_op_index = witness.len();
    if ec_op.is_some()
        && by_component
            .insert("ec_op_builtin", native_ec_op_index)
            .is_some()
    {
        return Err(ResidentRuntimeError::PreparedWitnessCaptureContract {
            component: "ec_op_builtin",
            role: "native EC-op duplicates a recorded witness component",
        });
    }
    let task_count = witness.len() + usize::from(ec_op.is_some());
    if by_component.len() != task_count {
        return Err(ResidentRuntimeError::PreparedWitnessCaptureContract {
            component: "<duplicate>",
            role: "prepared witness component appears more than once",
        });
    }

    let schedule_levels = CAIRO_SCHEDULE.levels().map_err(|_| {
        ResidentRuntimeError::PreparedWitnessCaptureContract {
            component: "<schedule>",
            role: "generated component dependency schedule is invalid",
        }
    })?;
    let mut seen = vec![false; task_count];
    let mut result = Vec::new();
    for level in schedule_levels {
        let components = level
            .into_iter()
            .filter_map(|component| by_component.get(component).copied())
            .collect::<Vec<_>>();
        if components.is_empty() {
            continue;
        }
        let lanes = pack_weighted_lane_level(
            components
                .iter()
                .map(|&component| {
                    if component == native_ec_op_index {
                        (
                            component,
                            ec_op.expect("native task was indexed").estimated_work(),
                            "ec_op_builtin",
                        )
                    } else {
                        (
                            component,
                            witness[component].writer.estimated_work(),
                            witness[component].component,
                        )
                    }
                })
                .collect(),
            lane_count,
        );
        for component in components {
            seen[component] = true;
        }
        result.push(lanes);
    }
    if seen.iter().any(|seen| !seen) {
        let component = seen
            .iter()
            .position(|seen| !seen)
            .map(|index| {
                if index == native_ec_op_index {
                    "ec_op_builtin"
                } else {
                    witness[index].component
                }
            })
            .unwrap_or("<missing>");
        return Err(ResidentRuntimeError::PreparedWitnessCaptureContract {
            component,
            role: "prepared witness component is absent from generated schedule",
        });
    }
    Ok(result)
}

fn enqueue_witness_lane_levels(
    arena: &stwo_backend_cuda::DeviceArena,
    witness: &[PreparedResidentWitness<'_>],
    ec_op: Option<&PreparedEcOpGraph<'_>>,
    levels: &[Vec<Vec<usize>>],
    multiplicity: Option<&PreparedResidentMultiplicity<'_>>,
) -> Result<(), WitnessLaneLaunchError> {
    let context = arena.context();
    for level in levels {
        let active_lanes = level
            .iter()
            .enumerate()
            .filter_map(|(lane, components)| (!components.is_empty()).then_some(lane))
            .collect::<Vec<_>>();
        let mut forked = Vec::with_capacity(active_lanes.len());
        let mut first_error = None;
        for &lane in &active_lanes {
            match context.fork_lane(lane) {
                Ok(launch) => forked.push((lane, launch)),
                Err(error) => {
                    first_error = Some(WitnessLaneLaunchError::Cuda(error));
                    break;
                }
            }
        }
        if first_error.is_none() {
            for &(lane, launch) in &forked {
                for &component in &level[lane] {
                    if component == witness.len() {
                        let launched = ec_op
                            .expect("native EC-op task was planned")
                            .launch_on(launch)
                            .map_err(WitnessLaneLaunchError::EcOp);
                        if let Err(error) = launched {
                            first_error = Some(error);
                            break;
                        }
                        continue;
                    }
                    let graph = &witness[component];
                    let launched = (|| {
                        if let Some(seed) = &graph.input_seed {
                            seed.launch_on(launch)
                                .map_err(WitnessLaneLaunchError::Input)?;
                        }
                        if let Some(gather) = &graph.input_gather {
                            gather
                                .launch_on(launch)
                                .map_err(WitnessLaneLaunchError::Input)?;
                        }
                        if let Some(compact) = &graph.input_compact {
                            compact
                                .launch_on(launch)
                                .map_err(WitnessLaneLaunchError::Input)?;
                        }
                        graph
                            .writer
                            .launch_on(launch)
                            .map_err(WitnessLaneLaunchError::Writer)?;
                        if let Some((_, feed)) = multiplicity.and_then(|multiplicity| {
                            multiplicity
                                .feeds
                                .iter()
                                .find(|(producer, _)| *producer == graph.component)
                        }) {
                            feed.launch_on(launch)
                                .map_err(WitnessLaneLaunchError::Feed)?;
                        }
                        Ok(())
                    })();
                    if let Err(error) = launched {
                        first_error = Some(error);
                        break;
                    }
                }
                if first_error.is_some() {
                    break;
                }
            }
        }
        for (lane, _) in forked {
            if let Err(error) = context.join_lane(lane) {
                if first_error.is_none() {
                    first_error = Some(WitnessLaneLaunchError::Cuda(error));
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
    }
    Ok(())
}

struct PreparedResidentMultiplicity<'a> {
    clear: PreparedWitnessFeedClearGraph<'a>,
    public_memory_seed: Option<PreparedWitnessFeedGraph<'a>>,
    feeds: Vec<(&'static str, PreparedWitnessFeedGraph<'a>)>,
    fixed_tables: Vec<PreparedFixedTableGraph<'a>>,
    memory_traces: Option<PreparedMemoryBaseTraceGraph<'a>>,
}

fn slice_matches_slot(slice: ArenaSlice, slot: ArenaSlotId, required_words: usize) -> bool {
    slice.id() == slot && slice.len_words() >= required_words
}

fn slices_match_slots(
    slices: &[ArenaSlice],
    slots: &[ArenaSlotId],
    required_words: &[usize],
) -> bool {
    slices.len() == slots.len()
        && slices.len() == required_words.len()
        && slices
            .iter()
            .zip(slots)
            .zip(required_words)
            .all(|((&slice, &slot), &words)| slice_matches_slot(slice, slot, words))
}

enum PreparedResidentCommitment<'a> {
    Full(PreparedCommitGraph<'a>),
    Progressive {
        graph: PreparedProgressiveCommitGraph<'a>,
        retained_evaluations: Vec<Option<Vec<ArenaSlice>>>,
    },
}

impl PreparedResidentCommitment<'_> {
    fn launch(&self) -> Result<(), ResidentLaunchError> {
        match self {
            Self::Full(graph) => graph.launch().map_err(ResidentLaunchError::Commit),
            Self::Progressive { graph, .. } => graph
                .launch()
                .map_err(ResidentLaunchError::ProgressiveCommit),
        }
    }

    fn root_slice(&self) -> ArenaSlice {
        match self {
            Self::Full(graph) => graph.root_slice(),
            Self::Progressive { graph, .. } => graph.root_slice(),
        }
    }

    fn retained_layers_bottom_up(&self) -> &[ArenaSlice] {
        match self {
            Self::Full(graph) => graph.retained_layers_bottom_up(),
            Self::Progressive { graph, .. } => graph.retained_layers_bottom_up(),
        }
    }

    fn retained_evaluations(&self) -> &[Option<Vec<ArenaSlice>>] {
        match self {
            Self::Full(graph) => graph.retained_evaluations(),
            Self::Progressive {
                retained_evaluations,
                ..
            } => retained_evaluations,
        }
    }

    fn read_root_at_transcript_boundary(&self) -> Result<Blake2sHash, ResidentRuntimeError> {
        match self {
            Self::Full(graph) => Ok(graph.read_root_at_transcript_boundary()?),
            Self::Progressive { graph, .. } => Ok(graph.read_root_at_transcript_boundary()?),
        }
    }
}

pub struct ResidentGraphRuntime<'a> {
    execution_tables: Option<PreparedExecutionTablesGraph<'a>>,
    execution_tables_ingest: Option<PreparedExecutionTablesIngestTelemetry>,
    ec_op: Option<PreparedEcOpGraph<'a>>,
    ec_op_ingest: Option<PreparedEcOpIngestTelemetry>,
    witness: Vec<PreparedResidentWitness<'a>>,
    witness_lane_levels: Vec<Vec<Vec<usize>>>,
    multiplicity: Option<PreparedResidentMultiplicity<'a>>,
    commitments: Vec<(CommitmentTreeId, PreparedResidentCommitment<'a>)>,
    base_interpolation: PreparedInterpolationGraph<'a>,
    fixed_preprocessed_root: ArenaSlice,
    fixed_preprocessed_retained_layers: Vec<ArenaSlice>,
    relation: PreparedRelationGraph<'a>,
    interaction_interpolation: PreparedInterpolationGraph<'a>,
    interaction_claim_sources: Vec<ArenaSlice>,
    composition: PreparedCompositionGraph<'a>,
    oods: ResidentOodsPipeline<'a>,
    fri: PreparedFriGraph<'a>,
    fri_final: PreparedFriFinalGraph<'a>,
    interaction_pow: PreparedBlake2sPowGraph<'a>,
    query_pow: PreparedBlake2sPowGraph<'a>,
    decommit: PreparedDecommitGraph<'a>,
    proof_bundle: ArenaSlice,
    transcript: PreparedBlake2sTranscript<'a>,
    transcript_inputs: Vec<(TranscriptInputId, ArenaSlice)>,
    transcript_outputs: Vec<(TranscriptOutputId, ArenaSlice)>,
    transcript_segments: Vec<TranscriptSegmentPlan>,
    transcript_cursor: TranscriptSegmentCursor,
    workspace: &'a GraphWorkspace,
    identity: ResidentWorkspaceIdentity,
    relation_challenge_generation: u64,
    launched_relation_challenge_generation: u64,
    fri_challenge_generations: Vec<u64>,
    launched_fri_challenge_generations: Vec<u64>,
    next_fri_round: Option<usize>,
}

impl<'a> ResidentGraphRuntime<'a> {
    /// Validate all identities and source order, upload immutable descriptor
    /// tables, and bind every launch to the workspace's isolated CUDA context.
    /// `setup_relation_challenges` initializes device storage only; interaction
    /// launch remains blocked until [`Self::upload_relation_challenges`] records
    /// the real post-base-commit transcript challenge.
    pub fn prepare(
        workspace: &'a GraphWorkspace,
        expected_identity: ResidentWorkspaceIdentity,
        setup_relation_challenges: RelationChallenges<'_>,
        transcript_plan: &CairoBlake2sTranscriptPlan,
        current_composition: &crate::composition_plan::CompositionPlan,
        composition_bindings: &CompositionProofBindings,
        execution_tables_host: Option<ExecutionTablesHostData<'_>>,
        ec_op_segment_start: Option<usize>,
        preprocessed_trace: Option<Arc<PreProcessedTrace>>,
        public_memory_seed_host: Option<&[u32]>,
    ) -> Result<Self, ResidentRuntimeError> {
        let actual_identity = ResidentWorkspaceIdentity::of(workspace);
        if expected_identity != actual_identity {
            return Err(ResidentRuntimeError::WorkspaceIdentityMismatch {
                expected: expected_identity,
                actual: actual_identity,
            });
        }
        let protocol_identity = workspace.plan().protocol_identity();
        if !workspace.preprocessed_commitment_ready() {
            return Err(ResidentRuntimeError::FixedPreprocessedCommitmentNotReady);
        }
        let public_memory_seed_planned = workspace
            .plan()
            .multiplicity()
            .and_then(|planned| planned.public_memory_seed.as_ref())
            .is_some();
        if public_memory_seed_planned != public_memory_seed_host.is_some() {
            return Err(ResidentRuntimeError::PublicMemoryMultiplicitySeed(
                "claim-bound source and planned seed disagree",
            ));
        }
        if let Some(planned) = workspace.plan().multiplicity() {
            if !planned.coverage_complete() {
                return Err(ResidentRuntimeError::IncompleteFixedMultiplicityCoverage(
                    planned.coverage_gaps.clone(),
                ));
            }
            if !planned.blockers.is_empty() {
                return Err(ResidentRuntimeError::UnsupportedMultiplicityFeeds(
                    planned.blockers.clone(),
                ));
            }
            if preprocessed_trace.is_none() {
                return Err(ResidentRuntimeError::MissingPreprocessedTraceForMultiplicity);
            }
        }

        let relation_sources = arena_relation_sources(workspace)?;
        for source in relation_sources
            .iter()
            .flat_map(|instance| &instance.columns)
        {
            require_resident_source(workspace, *source)?;
        }

        let arena = workspace.arena();
        let (execution_tables, execution_tables_ingest, execution_tables_view) =
            match (workspace.plan().execution_tables(), execution_tables_host) {
                (Some(planned), Some(host)) => {
                    let prepared = PreparedExecutionTablesGraph::prepare(
                        arena,
                        &planned.requirements,
                        &planned.slots,
                    )?;
                    let ingest = prepared.ingest(host)?;
                    let view = prepared.view()?;
                    (Some(prepared), Some(ingest), Some(view))
                }
                (Some(_), None) => {
                    return Err(ResidentRuntimeError::MissingPreparedExecutionTables)
                }
                (None, Some(_)) => {
                    return Err(ResidentRuntimeError::UnexpectedPreparedExecutionTables)
                }
                (None, None) => (None, None, None),
            };
        let witness = match execution_tables_view {
            Some(tables) => workspace
                .plan()
                .witness()
                .components
                .iter()
                .map(|component| {
                    let input_gather = component
                        .input_gather
                        .as_ref()
                        .map(|gather| {
                            let sources = gather
                                .sources
                                .iter()
                                .map(|&binding| bind_arena_binding(arena, binding))
                                .collect::<Result<Vec<_>, _>>()?;
                            let edges = gather
                                .requirements
                                .edges
                                .iter()
                                .map(|edge| edge.edge)
                                .collect::<Vec<_>>();
                            PreparedWitnessInputGatherGraph::prepare(
                                arena,
                                &sources,
                                &edges,
                                gather.requirements.include_enabler,
                                gather.requirements.include_iota,
                                &gather.slots,
                            )
                            .map_err(ResidentRuntimeError::from)
                        })
                        .transpose()?;
                    let input_seed = component
                        .input_seed
                        .as_ref()
                        .map(|seed| {
                            PreparedWitnessInputSeedGraph::prepare(
                                arena,
                                &seed.requirements,
                                &seed.slots,
                            )
                            .map_err(ResidentRuntimeError::from)
                        })
                        .transpose()?;
                    let input_compact = component
                        .input_compact
                        .as_ref()
                        .map(|compact| {
                            let sources = compact
                                .sources
                                .iter()
                                .map(|&binding| bind_arena_binding(arena, binding))
                                .collect::<Result<Vec<_>, _>>()?;
                            PreparedWitnessInputCompactGraph::prepare(
                                arena,
                                &sources,
                                &compact.requirements,
                                &compact.slots,
                            )
                            .map_err(ResidentRuntimeError::from)
                        })
                        .transpose()?;
                    let writer = PreparedWitnessGraph::prepare_with_execution_tables(
                        arena,
                        &component.program,
                        component.requirements.row_count,
                        &component.requirements.multiplicity_column_words,
                        tables,
                        &component.slots,
                        PreparedWitnessMode::RequireEmbeddedAot,
                    )
                    .map_err(ResidentRuntimeError::from)?;
                    Ok::<_, ResidentRuntimeError>(PreparedResidentWitness {
                        component: component.component,
                        native_input_producer: component.native_input_producer,
                        input_gather,
                        input_seed,
                        input_compact,
                        writer,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };
        let multiplicity = match workspace.plan().multiplicity() {
            Some(planned) => {
                let preprocessed_trace = preprocessed_trace
                    .ok_or(ResidentRuntimeError::MissingPreprocessedTraceForMultiplicity)?;
                let destinations = planned
                    .multiplicities
                    .iter()
                    .map(|&(_, binding)| bind_arena_binding(arena, binding))
                    .collect::<Result<Vec<_>, _>>()?;
                let clear = PreparedWitnessFeedClearGraph::prepare(
                    arena,
                    &destinations,
                    planned.clear_slots,
                )?;
                let feeds = planned
                    .feeds
                    .iter()
                    .map(|feed| {
                        let luts = feed
                            .plan
                            .lut_families
                            .iter()
                            .map(|&family| {
                                canonical_count_lut(family, Arc::clone(&preprocessed_trace))
                                    .map_err(|_| {
                                        ResidentRuntimeError::CanonicalMultiplicityLut(family)
                                    })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let graph = PreparedWitnessFeedGraph::prepare_with_mode(
                            arena,
                            bind_arena_binding(arena, feed.source)?,
                            feed.plan.row_count,
                            feed.plan.sub_words_per_row,
                            &feed.plan.descriptors,
                            &luts,
                            &feed.plan.requirements.multiplicity_words,
                            &feed.slots,
                            protocol_identity.witness_feed_launch_mode,
                        )?;
                        Ok::<_, ResidentRuntimeError>((feed.plan.producer, graph))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let public_memory_seed = planned
                    .public_memory_seed
                    .as_ref()
                    .map(|feed| {
                        let graph = PreparedWitnessFeedGraph::prepare_with_mode(
                            arena,
                            bind_arena_binding(arena, feed.source)?,
                            feed.plan.row_count,
                            feed.plan.sub_words_per_row,
                            &feed.plan.descriptors,
                            &[],
                            &feed.plan.requirements.multiplicity_words,
                            &feed.slots,
                            protocol_identity.witness_feed_launch_mode,
                        )?;
                        let words = public_memory_seed_host.ok_or(
                            ResidentRuntimeError::PublicMemoryMultiplicitySeed(
                                "planned seed has no claim-bound source",
                            ),
                        )?;
                        graph.upload_source(words)?;
                        Ok::<_, ResidentRuntimeError>(graph)
                    })
                    .transpose()?;
                if planned.public_memory_seed.is_none() && public_memory_seed_host.is_some() {
                    return Err(ResidentRuntimeError::PublicMemoryMultiplicitySeed(
                        "claim-bound source has no planned seed",
                    ));
                }
                let fixed_tables = planned
                    .fixed_tables
                    .iter()
                    .map(|fixed| {
                        let sources = fixed
                            .sources
                            .iter()
                            .map(|source| match *source {
                                PlannedFixedTableSource::Arena(binding) => {
                                    bind_arena_binding(arena, binding)
                                        .map(FixedTableSourceColumn::from)
                                        .map_err(ResidentRuntimeError::from)
                                }
                                PlannedFixedTableSource::RegisteredPedersen18 { column } => {
                                    let table = stwo_backend_cuda::pedersen_table::registered_borrowed_pedersen_table()
                                        .ok_or(ResidentRuntimeError::RegisteredPedersenTableUnavailable)?;
                                    if !table.has_exact_rows(PEDERSEN_POINTS_18_ROW_COUNT) {
                                        return Err(ResidentRuntimeError::RegisteredPedersenTableRows {
                                            expected: PEDERSEN_POINTS_18_ROW_COUNT,
                                            actual: table.n_rows(),
                                        });
                                    }
                                    let source = table
                                        .column(column)
                                        .filter(|source| {
                                            source.index() == column
                                                && source.len_words()
                                                    == PEDERSEN_POINTS_18_ROW_COUNT
                                        })
                                        .ok_or(
                                            ResidentRuntimeError::RegisteredPedersenTableColumn(
                                                column,
                                            ),
                                        )?;
                                    Ok(FixedTableSourceColumn::from(source))
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        PreparedFixedTableGraph::prepare_contiguous(
                            arena,
                            fixed.plan.materializer.config(),
                            &sources,
                            bind_arena_binding(arena, fixed.multiplicity)?,
                            &fixed.slots,
                        )
                        .map_err(ResidentRuntimeError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let memory_traces = planned
                    .memory_traces
                    .as_ref()
                    .map(|memory| {
                        let execution = execution_tables
                            .as_ref()
                            .ok_or(ResidentRuntimeError::MissingPreparedExecutionTables)?;
                        let multiplicity = |name| {
                            planned
                                .multiplicities
                                .iter()
                                .find(|(candidate, _)| *candidate == name)
                                .ok_or(ResidentRuntimeError::PreparedFixedTableCaptureContract {
                                    component: "memory_id_to_big",
                                    role: "runtime multiplicity destination",
                                })
                                .and_then(|(_, binding)| {
                                    bind_arena_binding(arena, *binding)
                                        .map_err(ResidentRuntimeError::from)
                                })
                        };
                        let address_outputs = memory
                            .address_outputs
                            .iter()
                            .map(|&binding| bind_arena_binding(arena, binding))
                            .collect::<Result<Vec<_>, _>>()?;
                        let big_outputs = memory
                            .big_parts
                            .iter()
                            .map(|part| {
                                part.outputs
                                    .iter()
                                    .map(|&binding| bind_arena_binding(arena, binding))
                                    .collect::<Result<Vec<_>, _>>()
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let big_parts = memory
                            .big_parts
                            .iter()
                            .zip(&big_outputs)
                            .map(|(part, outputs)| MemoryBaseTracePart {
                                source_offset: part.source_offset,
                                row_count: part.row_count,
                                outputs,
                            })
                            .collect::<Vec<_>>();
                        let small_outputs = memory
                            .small_part
                            .outputs
                            .iter()
                            .map(|&binding| bind_arena_binding(arena, binding))
                            .collect::<Result<Vec<_>, _>>()?;
                        let graph = PreparedMemoryBaseTraceGraph::prepare(
                            arena,
                            execution,
                            multiplicity("memory_address_to_id")?,
                            memory.plan.address_count_words,
                            memory.plan.address_rows,
                            &address_outputs,
                            multiplicity("memory_id_to_big")?,
                            memory.plan.big_count_words,
                            &big_parts,
                            multiplicity("memory_id_to_big#small")?,
                            memory.plan.small_count_words,
                            MemoryBaseTracePart {
                                source_offset: memory.small_part.source_offset,
                                row_count: memory.small_part.row_count,
                                outputs: &small_outputs,
                            },
                            bind_arena_binding(arena, memory.rc99_lut)?,
                            memory.plan.rc99_lut_words,
                            bind_arena_binding(arena, memory.rc99_counts)?,
                        )?;
                        let rc99_lut = canonical_count_lut(
                            "range_check_9_9_state",
                            Arc::clone(&preprocessed_trace),
                        )
                        .map_err(|_| {
                            ResidentRuntimeError::CanonicalMultiplicityLut("range_check_9_9_state")
                        })?;
                        graph.upload_rc99_lut(&rc99_lut)?;
                        Ok::<_, ResidentRuntimeError>(graph)
                    })
                    .transpose()?;
                Some(PreparedResidentMultiplicity {
                    clear,
                    public_memory_seed,
                    feeds,
                    fixed_tables,
                    memory_traces,
                })
            }
            None => None,
        };
        let (ec_op, ec_op_ingest) = match (workspace.plan().ec_op(), ec_op_segment_start) {
            (Some(planned), Some(segment_start)) => {
                let execution = execution_tables
                    .as_ref()
                    .ok_or(ResidentRuntimeError::MissingPreparedExecutionTables)?;
                let prepared = PreparedEcOpGraph::prepare(
                    arena,
                    execution.view()?,
                    &planned.requirements,
                    &planned.slots,
                )?;
                let ingest = prepared.ingest_segment_start(segment_start)?;
                (Some(prepared), Some(ingest))
            }
            (Some(_), None) => return Err(ResidentRuntimeError::MissingPreparedEcOpSegment),
            (None, Some(_)) => return Err(ResidentRuntimeError::UnexpectedPreparedEcOpSegment),
            (None, None) => (None, None),
        };
        let witness_lane_levels =
            plan_witness_lane_levels(&witness, ec_op.as_ref(), arena.context().lane_count())?;
        let planned_transcript = workspace.plan().transcript();
        if planned_transcript.schedule_key != transcript_plan.schedule_key() {
            return Err(ResidentRuntimeError::TranscriptScheduleMismatch {
                expected: planned_transcript.schedule_key,
                actual: transcript_plan.schedule_key(),
            });
        }
        if planned_transcript.requirements != *transcript_plan.schedule().requirements() {
            return Err(ResidentRuntimeError::TranscriptRequirementsMismatch);
        }
        let (transcript_inputs, transcript_outputs) = transcript_bindings(workspace)?;
        let transcript = PreparedBlake2sTranscript::prepare(
            arena,
            transcript_plan.schedule().clone(),
            planned_transcript.slots,
            &transcript_inputs
                .iter()
                .map(|&(id, slice)| TranscriptInputBinding { id, slice })
                .collect::<Vec<_>>(),
            &transcript_outputs
                .iter()
                .map(|&(id, slice)| TranscriptOutputBinding { id, slice })
                .collect::<Vec<_>>(),
        )?;
        let transcript_cursor = transcript.segment_cursor();
        let relation_plan = workspace.plan().relation();
        let relation = PreparedRelationGraph::prepare_with_mode(
            arena,
            relation_plan.execution.kernel_program(),
            relation_plan.launch_mode,
            &relation_plan.slots,
            &relation_sources,
            setup_relation_challenges,
        )?;
        let interaction_claim_sources = interaction_outputs_in_cairo_order(workspace, &relation)?;
        let composition = prepare_resident_composition(
            workspace,
            &relation,
            current_composition,
            composition_bindings,
        )?;

        let fixed_preprocessed = workspace
            .plan()
            .commitment(CommitmentTreeId::Preprocessed)
            .ok_or(ResidentRuntimeError::MissingPreparedCommitment(
                CommitmentTreeId::Preprocessed,
            ))?;
        let fixed_preprocessed_root = bind_arena_binding(arena, fixed_preprocessed.root)?;
        let fixed_preprocessed_retained_layers = fixed_preprocessed
            .retained_layers_bottom_up
            .iter()
            .map(|&binding| bind_arena_binding(arena, binding))
            .collect::<Result<Vec<_>, _>>()?;
        let mut commitments = Vec::with_capacity(workspace.plan().commitments().len() - 1);
        for planned in workspace
            .plan()
            .commitments()
            .iter()
            .filter(|planned| planned.id != CommitmentTreeId::Preprocessed)
        {
            let groups = commitment_groups(workspace, planned)?;
            let twiddles = bind_arena_binding(arena, planned.twiddles)?;
            let retained_evaluations = planned
                .retained_evaluation_groups
                .iter()
                .map(|group| {
                    group
                        .as_ref()
                        .map(|columns| {
                            columns
                                .iter()
                                .map(|&binding| bind_arena_binding(arena, binding))
                                .collect::<Result<Vec<_>, _>>()
                                .map(|columns| CommitEvaluationGroup { columns })
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>, ArenaError>>()?;
            let evaluation_outputs = planned
                .evaluation_output_groups
                .iter()
                .map(|group| {
                    group
                        .as_ref()
                        .map(|columns| {
                            columns
                                .iter()
                                .map(|&binding| bind_arena_binding(arena, binding))
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>, ArenaError>>()?;
            let prepared = match (&planned.requirements, &planned.slots) {
                (
                    ModeAwareCommitWorkspaceRequirements::FullLifting(_),
                    ModeAwareCommitWorkspaceSlots::FullLifting(slots),
                ) => PreparedResidentCommitment::Full(
                    PreparedCommitGraph::prepare_with_retained_evaluations(
                        arena,
                        planned.config,
                        &groups,
                        twiddles,
                        slots,
                        &retained_evaluations,
                    )?,
                ),
                (
                    ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements),
                    ModeAwareCommitWorkspaceSlots::DomainProgressive(slots),
                ) => {
                    let coefficients = groups
                        .iter()
                        .flat_map(|group| group.columns.iter().copied())
                        .collect::<Vec<_>>();
                    let flat_retained = groups
                        .iter()
                        .zip(&evaluation_outputs)
                        .flat_map(|(group, retained)| match retained {
                            Some(retained) => {
                                retained.iter().copied().map(Some).collect::<Vec<_>>()
                            }
                            None => vec![None; group.columns.len()],
                        })
                        .collect::<Vec<_>>();
                    let grouped_retained = retained_evaluations
                        .iter()
                        .map(|group| group.as_ref().map(|group| group.columns.clone()))
                        .collect();
                    PreparedResidentCommitment::Progressive {
                        graph: PreparedProgressiveCommitGraph::prepare_with_modes(
                            arena,
                            planned.config,
                            requirements,
                            slots,
                            &coefficients,
                            &flat_retained,
                            twiddles,
                            protocol_identity.commit_mode,
                            protocol_identity.blake2s_interior_fused,
                        )?,
                        retained_evaluations: grouped_retained,
                    }
                }
                _ => return Err(ResidentRuntimeError::CommitModeMismatch),
            };
            commitments.push((planned.id, prepared));
        }
        let base_interpolation =
            prepare_commitment_interpolation(workspace, CommitmentTreeId::Base)?;
        let interaction_interpolation =
            prepare_commitment_interpolation(workspace, CommitmentTreeId::Interaction)?;

        let oods = ResidentOodsPipeline::prepare(workspace)?;

        let fri_plan = workspace.plan().fri();
        let fri_input = bind_arena_binding(arena, fri_plan.input_values)?;
        if fri_input.id() != oods.quotient().output_evaluation().id()
            || fri_input.len_words() != oods.quotient().output_evaluation().len_words()
        {
            return Err(ResidentRuntimeError::TranscriptRequirementsMismatch);
        }
        let fri = PreparedFriGraph::prepare(
            arena,
            fri_plan.config,
            oods.quotient().output_evaluation(),
            bind_arena_binding(arena, fri_plan.twiddles)?,
            &fri_plan.slots,
        )?;
        let transcript_input = |semantic: CairoTranscriptInput| {
            let id = semantic.id()?;
            transcript_inputs
                .iter()
                .find_map(|&(candidate, slice)| (candidate == id).then_some(slice))
                .ok_or(ResidentRuntimeError::MissingTranscriptInput(id))
        };
        let final_plan = workspace.plan().final_fri_pow();
        let fri_final = PreparedFriFinalGraph::prepare(
            arena,
            fri_plan.config,
            fri.final_evaluation(),
            bind_arena_binding(arena, fri_plan.twiddles)?,
            transcript_input(CairoTranscriptInput::FriLastLayerPolynomial)?,
            final_plan.final_slots,
        )?;
        let interaction_pow = PreparedBlake2sPowGraph::prepare(
            arena,
            transcript.state(),
            final_plan.interaction_pow_bits,
            transcript_input(CairoTranscriptInput::InteractionPowNonce)?,
            final_plan.interaction_pow_slots,
        )?;
        let query_pow = PreparedBlake2sPowGraph::prepare(
            arena,
            transcript.state(),
            final_plan.query_pow_bits,
            transcript_input(CairoTranscriptInput::QueryPowNonce)?,
            final_plan.query_pow_slots,
        )?;
        let decommit_plan = workspace.plan().decommit();
        let raw_queries = bind_arena_binding(arena, decommit_plan.raw_queries)?;
        let query_output_id = CairoTranscriptOutput::QueryPositions.id()?;
        let transcript_queries = transcript_outputs
            .iter()
            .find_map(|&(id, slice)| (id == query_output_id).then_some(slice))
            .ok_or(ResidentRuntimeError::MissingTranscriptOutput(
                query_output_id,
            ))?;
        require_same_slice(
            "decommit queries are not the transcript output",
            raw_queries,
            transcript_queries,
        )?;
        let decommit_sources = resident_decommit_sources(
            workspace,
            &commitments,
            fixed_preprocessed_root,
            &fixed_preprocessed_retained_layers,
            &fri,
        )?;
        let decommit = PreparedDecommitGraph::prepare(
            arena,
            decommit_plan.config.clone(),
            raw_queries,
            Some(bind_arena_binding(arena, decommit_plan.lde_twiddles)?),
            &decommit_sources,
            &decommit_plan.slots,
        )?;
        require_same_slice(
            "decommit assembly does not match the planned final ABI",
            decommit.assembly_slice(),
            bind_arena_binding(arena, decommit_plan.assembly)?,
        )?;
        let proof_bundle = bind_arena_binding(arena, decommit_plan.proof_bundle)?;
        if decommit_plan.proof_bundle.len_words != decommit_plan.proof_bundle_layout.total_words
            || proof_bundle.len_words() < decommit_plan.proof_bundle_layout.total_words
        {
            return Err(ResidentRuntimeError::TranscriptBindingTooSmall {
                role: "resident proof bundle",
                required_words: decommit_plan.proof_bundle_layout.total_words,
                actual_words: proof_bundle.len_words(),
            });
        }
        let fri_rounds = fri.round_count();

        let runtime = Self {
            execution_tables,
            execution_tables_ingest,
            ec_op,
            ec_op_ingest,
            witness,
            witness_lane_levels,
            multiplicity,
            commitments,
            base_interpolation,
            fixed_preprocessed_root,
            fixed_preprocessed_retained_layers,
            relation,
            interaction_interpolation,
            interaction_claim_sources,
            composition,
            oods,
            fri,
            fri_final,
            interaction_pow,
            query_pow,
            decommit,
            proof_bundle,
            transcript,
            transcript_inputs,
            transcript_outputs,
            transcript_segments: transcript_plan.segments().to_vec(),
            transcript_cursor,
            workspace,
            identity: actual_identity,
            // Setup values only initialize the stable challenge slots. The
            // transcript boundary must explicitly publish generation one.
            relation_challenge_generation: 0,
            launched_relation_challenge_generation: 0,
            fri_challenge_generations: vec![0; fri_rounds],
            launched_fri_challenge_generations: vec![0; fri_rounds],
            next_fri_round: None,
        };
        // Preparing every descriptor is not enough: require the fully bound
        // runtime to match the strict schedule one-for-one before it can escape
        // this constructor.  This includes producer gather/compaction sources,
        // writer destinations and embedded-AOT identity.
        runtime.require_prepared_witness_coverage()?;
        Ok(runtime)
    }

    pub const fn identity(&self) -> ResidentWorkspaceIdentity {
        self.identity
    }

    pub const fn execution_tables_ingest_telemetry(
        &self,
    ) -> Option<PreparedExecutionTablesIngestTelemetry> {
        self.execution_tables_ingest
    }

    pub const fn ec_op_ingest_telemetry(&self) -> Option<PreparedEcOpIngestTelemetry> {
        self.ec_op_ingest
    }

    pub fn prepared_numerator_schedule(&self) -> PreparedNumeratorSchedule {
        self.oods.numerator_schedule()
    }

    /// Upload the complete compact input set before capture/replay. Every copy
    /// targets a stable arena column; one setup fence protects the borrowed host
    /// vectors, and hot-path telemetry is reset only after this boundary.
    pub fn upload_witness_inputs_at_ingest(
        &self,
        inputs: &[ResidentWitnessInput<'_>],
    ) -> Result<ResidentWitnessIngestReport, ResidentRuntimeError> {
        if inputs.len() != self.witness.len() {
            return Err(ResidentRuntimeError::WitnessInputCoverage {
                expected: self.witness.len(),
                actual: inputs.len(),
            });
        }
        let mut seen = Vec::with_capacity(inputs.len());
        let mut report = ResidentWitnessIngestReport {
            components: inputs.len(),
            ..ResidentWitnessIngestReport::default()
        };
        for input in inputs {
            if seen.contains(&input.component) {
                return Err(ResidentRuntimeError::DuplicateWitnessInput(input.component));
            }
            seen.push(input.component);
            let prepared = self
                .witness
                .iter()
                .find(|prepared| prepared.component == input.component)
                .ok_or(ResidentRuntimeError::MissingPreparedWitness(
                    input.component,
                ))?;
            let destinations = prepared.writer.input_columns();
            if (prepared.input_gather.is_some()
                || prepared.input_seed.is_some()
                || prepared.input_compact.is_some()
                || prepared.native_input_producer.is_some())
                && !input.columns.is_empty()
            {
                return Err(ResidentRuntimeError::UnexpectedGatheredWitnessHostInput(
                    input.component,
                ));
            }
            if prepared.input_gather.is_none()
                && prepared.input_seed.is_none()
                && prepared.input_compact.is_none()
                && prepared.native_input_producer.is_none()
                && destinations.len() != input.columns.len()
            {
                return Err(ResidentRuntimeError::WitnessInputColumnCount {
                    component: input.component,
                    expected: destinations.len(),
                    actual: input.columns.len(),
                });
            }
            match (&prepared.input_seed, input.seed_scalars) {
                (Some(seed), Some(values)) => {
                    seed.ingest_scalars(values)?;
                    let bytes = values
                        .len()
                        .checked_mul(core::mem::size_of::<u32>())
                        .ok_or(ResidentRuntimeError::SizeOverflow)?;
                    report.columns += values.len();
                    report.h2d_copies += usize::from(!values.is_empty());
                    report.h2d_bytes = report
                        .h2d_bytes
                        .checked_add(bytes)
                        .ok_or(ResidentRuntimeError::SizeOverflow)?;
                }
                (None, None) => {}
                _ => {
                    return Err(ResidentRuntimeError::WitnessInputColumnCount {
                        component: input.component,
                        expected: prepared
                            .input_seed
                            .as_ref()
                            .map_or(0, |seed| seed.requirements().scalar_words),
                        actual: input.seed_scalars.map_or(0, <[u32]>::len),
                    })
                }
            }
            let mut seen_columns = Vec::with_capacity(input.columns.len());
            for source in input.columns {
                if source.ordinal >= destinations.len() {
                    return Err(ResidentRuntimeError::WitnessInputColumnCount {
                        component: input.component,
                        expected: destinations.len(),
                        actual: source.ordinal + 1,
                    });
                }
                if seen_columns.contains(&source.ordinal) {
                    return Err(ResidentRuntimeError::DuplicateWitnessInputColumn {
                        component: input.component,
                        ordinal: source.ordinal,
                    });
                }
                seen_columns.push(source.ordinal);
                let destination = destinations[source.ordinal];
                // The prepared writer reads exactly `row_count` words from each
                // input column's slot base, and the arena may pool the backing
                // slot larger than that (whole-slot binds over disjoint
                // lifetimes). Pin the host payload to the writer requirement —
                // not the pooled slot length — and fail closed on capacity.
                let required_words = prepared.writer.row_count();
                if source.words.len() != required_words {
                    return Err(ResidentRuntimeError::WitnessInputRowCount {
                        component: input.component,
                        column: source.ordinal,
                        expected: required_words,
                        actual: source.words.len(),
                    });
                }
                if destination.len_words() < required_words {
                    return Err(ResidentRuntimeError::WitnessInputRowCount {
                        component: input.component,
                        column: source.ordinal,
                        expected: required_words,
                        actual: destination.len_words(),
                    });
                }
                let bytes = source
                    .words
                    .len()
                    .checked_mul(core::mem::size_of::<u32>())
                    .ok_or(ResidentRuntimeError::SizeOverflow)?;
                unsafe {
                    self.workspace.arena().context().memcpy_h2d_async(
                        destination.as_void_ptr(),
                        source.words.as_ptr().cast(),
                        bytes,
                    )?;
                }
                report.columns += 1;
                report.h2d_copies += 1;
                report.h2d_bytes = report
                    .h2d_bytes
                    .checked_add(bytes)
                    .ok_or(ResidentRuntimeError::SizeOverflow)?;
            }
        }
        self.workspace.arena().context().sync()?;
        report.sync_calls = 1;
        Ok(report)
    }

    /// Reset counters after setup/input staging and immediately before a warm
    /// replay. This makes the resident acceptance gate independent of one-time
    /// descriptor uploads and graph instantiation.
    pub fn begin_hot_path_telemetry(&self) {
        self.workspace.arena().context().reset_telemetry();
    }

    pub fn hot_path_telemetry(&self) -> CudaExecTelemetry {
        self.workspace.arena().context().telemetry()
    }

    pub fn require_hot_path_budget(
        &self,
        budget: ResidentHotPathBudget,
    ) -> Result<CudaExecTelemetry, ResidentRuntimeError> {
        let actual = self.hot_path_telemetry();
        if !budget.accepts(actual) {
            return Err(ResidentRuntimeError::HotPathBudgetExceeded { budget, actual });
        }
        Ok(actual)
    }

    pub fn begin_transcript_generation(
        &mut self,
        generation: u64,
    ) -> Result<(), ResidentRuntimeError> {
        self.transcript_cursor.begin_generation(generation)?;
        Ok(())
    }

    pub fn transcript_segment_count(&self) -> usize {
        self.transcript_segments.len()
    }

    pub fn launch_transcript_segment_eager(
        &mut self,
        segment_index: usize,
    ) -> Result<(), ResidentRuntimeError> {
        let segment = self.transcript_segments.get(segment_index).ok_or(
            ResidentRuntimeError::InvalidTranscriptSegment(segment_index),
        )?;
        let start = if segment_index == 0 {
            TranscriptSegmentStart::Initialize
        } else {
            TranscriptSegmentStart::Resume
        };
        let generation = self.transcript_cursor.generation();
        self.transcript.launch_segment(
            &mut self.transcript_cursor,
            generation,
            segment.operation_range.clone(),
            start,
        )?;
        Ok(())
    }

    /// Admit the corresponding already-captured transcript range immediately
    /// before graph replay. The same cursor rejects skipped, duplicated or stale
    /// transcript segments in eager and captured execution.
    pub fn admit_transcript_segment_replay(
        &mut self,
        segment_index: usize,
    ) -> Result<(), ResidentRuntimeError> {
        let segment = self.transcript_segments.get(segment_index).ok_or(
            ResidentRuntimeError::InvalidTranscriptSegment(segment_index),
        )?;
        let start = if segment_index == 0 {
            TranscriptSegmentStart::Initialize
        } else {
            TranscriptSegmentStart::Resume
        };
        let generation = self.transcript_cursor.generation();
        self.transcript_cursor.admit_segment(
            self.transcript.schedule(),
            generation,
            segment.operation_range.clone(),
            start,
        )?;
        Ok(())
    }

    pub fn require_transcript_complete(&self) -> Result<(), ResidentRuntimeError> {
        self.transcript_cursor.require_complete()?;
        Ok(())
    }

    /// Correctness-only U4 gate. This performs compact D2H snapshot reads and a
    /// host Blake2s replay, so callers must invoke it after (and outside) the
    /// resident hot-path telemetry window. The backend verifier checks every
    /// scheduled boundary and every device-drawn output fail-closed.
    pub fn verify_transcript_mirror_correctness_only(
        &self,
    ) -> Result<TranscriptMirrorReport, ResidentRuntimeError> {
        let report = self.transcript.verify_mirror()?;
        if report.boundaries_verified != self.transcript.schedule().operations().len() {
            return Err(ResidentRuntimeError::TranscriptRequirementsMismatch);
        }
        Ok(report)
    }

    pub fn transcript_input(
        &self,
        semantic: CairoTranscriptInput,
    ) -> Result<ArenaSlice, ResidentRuntimeError> {
        let id = semantic.id()?;
        self.transcript_inputs
            .iter()
            .find_map(|&(candidate, slice)| (candidate == id).then_some(slice))
            .ok_or(ResidentRuntimeError::MissingTranscriptInput(id))
    }

    pub fn transcript_output(
        &self,
        semantic: CairoTranscriptOutput,
    ) -> Result<ArenaSlice, ResidentRuntimeError> {
        let id = semantic.id()?;
        self.transcript_outputs
            .iter()
            .find_map(|&(candidate, slice)| (candidate == id).then_some(slice))
            .ok_or(ResidentRuntimeError::MissingTranscriptOutput(id))
    }

    /// Upload the compact host-owned transcript inputs permitted at ingest
    /// (salt, PCS parameters, public claim material and persistent roots) in one
    /// batch and one synchronization. Large proof data is never accepted here.
    pub fn upload_transcript_inputs_at_ingest(
        &self,
        inputs: &[(CairoTranscriptInput, Vec<u32>)],
    ) -> Result<(), ResidentRuntimeError> {
        let mut seen = Vec::with_capacity(inputs.len());
        let mut uploads = Vec::with_capacity(inputs.len());
        for (semantic, words) in inputs {
            let id = semantic.id()?;
            if seen.contains(&id) {
                return Err(ResidentRuntimeError::TranscriptRequirementsMismatch);
            }
            seen.push(id);
            let expected = self
                .transcript
                .schedule()
                .requirements()
                .inputs
                .iter()
                .find_map(|requirement| (requirement.id == id).then_some(requirement.min_words))
                .ok_or(ResidentRuntimeError::MissingTranscriptInput(id))?;
            if words.len() != expected {
                return Err(ResidentRuntimeError::TranscriptBindingTooSmall {
                    role: "ingest_transcript_input",
                    required_words: expected,
                    actual_words: words.len(),
                });
            }
            let destination = self.transcript_input(*semantic)?;
            uploads.push((destination, words.as_slice()));
        }
        for (destination, words) in uploads {
            // SAFETY: `words` remains borrowed through the one sync below and
            // the exact logical destination width was checked above.
            unsafe {
                self.workspace.arena().context().memcpy_h2d_async(
                    destination.as_void_ptr(),
                    words.as_ptr().cast(),
                    words.len() * core::mem::size_of::<u32>(),
                )?;
            }
        }
        self.workspace.arena().context().sync()?;
        Ok(())
    }

    /// Enqueue a device-to-device root handoff into the semantic transcript
    /// input slot. No root crosses PCIe and the copy is capture-safe.
    pub fn stage_commitment_root_for_transcript(
        &self,
        commitment: CommitmentTreeId,
        semantic: CairoTranscriptInput,
    ) -> Result<(), ResidentRuntimeError> {
        let source = self.commitment_root_slice(commitment)?;
        let destination = self.transcript_input(semantic)?;
        self.copy_transcript_words("commitment_root", source, destination, 8)
    }

    pub fn stage_fri_root_for_transcript(
        &self,
        tree_index: usize,
        layer_index: u32,
    ) -> Result<(), ResidentRuntimeError> {
        let source = self.fri.tree_root(tree_index)?;
        let destination = self.transcript_input(CairoTranscriptInput::FriLayerRoot(layer_index))?;
        self.copy_transcript_words("fri_root", source, destination, 8)
    }

    /// Gather relation claimed sums in the canonical Cairo claim order into the
    /// one contiguous transcript input. Each sum stays on device.
    pub fn stage_interaction_claim_for_transcript(&self) -> Result<(), ResidentRuntimeError> {
        let destination = self.transcript_input(CairoTranscriptInput::InteractionClaim)?;
        let required_words = self.interaction_claim_sources.len().checked_mul(4).ok_or(
            ResidentRuntimeError::TranscriptClaimWidthMismatch {
                expected_words: usize::MAX,
                actual_words: destination.len_words(),
            },
        )?;
        if destination.len_words() != required_words {
            return Err(ResidentRuntimeError::TranscriptClaimWidthMismatch {
                expected_words: required_words,
                actual_words: destination.len_words(),
            });
        }
        for (index, source) in self.interaction_claim_sources.iter().copied().enumerate() {
            let offset =
                index
                    .checked_mul(4)
                    .ok_or(ResidentRuntimeError::TranscriptClaimWidthMismatch {
                        expected_words: usize::MAX,
                        actual_words: destination.len_words(),
                    })?;
            // SAFETY: exact destination width was checked above; each source is
            // a four-word claimed sum owned by the same arena/context.
            unsafe {
                self.workspace.arena().context().memcpy_d2d_async(
                    destination.as_u32_ptr().add(offset).cast(),
                    source.as_void_ptr().cast_const(),
                    4 * core::mem::size_of::<u32>(),
                )?;
            }
        }
        Ok(())
    }

    /// Make the device channel's `[z, alpha]` output authoritative for relation
    /// execution without reconstructing CommonLookupElements on the host.
    pub fn publish_relation_challenges_from_transcript(
        &mut self,
    ) -> Result<(), ResidentRuntimeError> {
        let drawn = self.transcript_output(CairoTranscriptOutput::CommonLookupElements)?;
        self.relation.expand_challenges_from_transcript(drawn)?;
        self.relation_challenge_generation = self
            .relation_challenge_generation
            .checked_add(1)
            .ok_or(ResidentRuntimeError::StaleRelationChallenges)?;
        Ok(())
    }

    pub fn publish_fri_challenge_from_transcript(
        &mut self,
        round_index: usize,
    ) -> Result<(), ResidentRuntimeError> {
        let layer = u32::try_from(round_index)
            .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(round_index))?;
        let source = self.transcript_output(CairoTranscriptOutput::FriFoldingChallenge(layer))?;
        let destination = self.fri_round_challenge_destination(round_index)?;
        self.copy_transcript_words("fri_challenge", source, destination, 4)?;
        self.mark_fri_round_challenge_ready(round_index)
    }

    pub fn capture_base_commit_only(&mut self) -> Result<(), ResidentRuntimeError> {
        let commitment_index = self
            .commitments
            .iter()
            .position(|(id, _)| *id == CommitmentTreeId::Base)
            .ok_or(ResidentRuntimeError::MissingPreparedCommitment(
                CommitmentTreeId::Base,
            ))?;
        let bootstrap_segment =
            self.transcript_segment_index(CairoTranscriptSegment::BootstrapThroughBase)?;
        let pow_segment =
            self.transcript_segment_index(CairoTranscriptSegment::InteractionPowAndLookup)?;
        let bootstrap_range = self.transcript_segments[bootstrap_segment]
            .operation_range
            .clone();
        let pow_range = self.transcript_segments[pow_segment]
            .operation_range
            .clone();
        let root_destination = self.transcript_input(CairoTranscriptInput::BaseRoot)?;
        let lookup_output = self.transcript_output(CairoTranscriptOutput::CommonLookupElements)?;
        let execution_tables = self.execution_tables.as_ref();
        let ec_op = self.ec_op.as_ref();
        let witness = &self.witness;
        let witness_lane_levels = &self.witness_lane_levels;
        let multiplicity = self.multiplicity.as_ref();
        let interpolation = &self.base_interpolation;
        let commitment = &self.commitments[commitment_index].1;
        let root_source = commitment.root_slice();
        let transcript = &self.transcript;
        let interaction_pow = &self.interaction_pow;
        let relation = &self.relation;
        let workspace = self.workspace;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let capture = capture_with_cursor_rollback(cursor, |cursor| {
            workspace.capture_segment(GraphSegment::IngestWitnessBaseCommit, |arena| {
                if let Some(execution_tables) = execution_tables {
                    execution_tables
                        .launch()
                        .map_err(ResidentLaunchError::ExecutionTables)?;
                }
                if let Some(multiplicity) = multiplicity {
                    multiplicity
                        .clear
                        .launch()
                        .map_err(ResidentLaunchError::WitnessFeed)?;
                    if let Some(seed) = &multiplicity.public_memory_seed {
                        seed.launch().map_err(ResidentLaunchError::WitnessFeed)?;
                    }
                }
                enqueue_witness_lane_levels(
                    arena,
                    witness,
                    ec_op,
                    witness_lane_levels,
                    multiplicity,
                )
                .map_err(ResidentLaunchError::WitnessLanes)?;
                if let Some(multiplicity) = multiplicity {
                    if let Some(memory) = &multiplicity.memory_traces {
                        memory
                            .launch()
                            .map_err(ResidentLaunchError::MemoryBaseTrace)?;
                    }
                    for fixed in &multiplicity.fixed_tables {
                        fixed.launch().map_err(ResidentLaunchError::FixedTable)?;
                    }
                }
                interpolation
                    .launch()
                    .map_err(ResidentLaunchError::Interpolation)?;
                commitment.launch()?;
                enqueue_copy_words(arena, root_source, root_destination, 8)?;
                transcript
                    .launch_segment(
                        cursor,
                        generation,
                        bootstrap_range,
                        TranscriptSegmentStart::Initialize,
                    )
                    .map_err(ResidentLaunchError::Transcript)?;
                interaction_pow.launch().map_err(ResidentLaunchError::Pow)?;
                transcript
                    .launch_segment(
                        cursor,
                        generation,
                        pow_range,
                        TranscriptSegmentStart::Resume,
                    )
                    .map_err(ResidentLaunchError::Transcript)?;
                relation
                    .expand_challenges_from_transcript(lookup_output)
                    .map_err(ResidentLaunchError::Relation)
            })
        })?;
        self.admit_reused_transcript_segment(capture, bootstrap_segment)?;
        self.admit_reused_transcript_segment(capture, pow_segment)?;
        Ok(())
    }

    pub fn capture_interaction_relation_and_commit(&mut self) -> Result<(), ResidentRuntimeError> {
        let commitment_index = self
            .commitments
            .iter()
            .position(|(id, _)| *id == CommitmentTreeId::Interaction)
            .ok_or(ResidentRuntimeError::MissingPreparedCommitment(
                CommitmentTreeId::Interaction,
            ))?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::InteractionAndComposition)?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let claim_destination = self.transcript_input(CairoTranscriptInput::InteractionClaim)?;
        let root_destination = self.transcript_input(CairoTranscriptInput::InteractionRoot)?;
        let claim_sources = &self.interaction_claim_sources;
        let relation = &self.relation;
        let interpolation = &self.interaction_interpolation;
        let commitment = &self.commitments[commitment_index].1;
        let root_source = commitment.root_slice();
        let transcript = &self.transcript;
        let workspace = self.workspace;
        let relation_launch_mode = workspace.plan().relation().launch_mode;
        let relation_tail_mode = workspace.plan().protocol_identity().relation_tail_mode;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let capture = capture_with_cursor_rollback(cursor, |cursor| {
            workspace.capture_segment(GraphSegment::InteractionCommit, |arena| {
                relation
                    .launch_with_modes(relation_launch_mode, relation_tail_mode)
                    .map_err(ResidentLaunchError::Relation)?;
                interpolation
                    .launch()
                    .map_err(ResidentLaunchError::Interpolation)?;
                commitment.launch()?;
                enqueue_claimed_sums(arena, claim_sources, claim_destination)?;
                enqueue_copy_words(arena, root_source, root_destination, 8)?;
                transcript
                    .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                    .map_err(ResidentLaunchError::Transcript)
            })
        })?;
        self.admit_reused_transcript_segment(capture, transcript_segment)?;
        Ok(())
    }

    pub fn capture_composition_commit_only(&mut self) -> Result<(), ResidentRuntimeError> {
        let commitment_index = self
            .commitments
            .iter()
            .position(|(id, _)| *id == CommitmentTreeId::Composition)
            .ok_or(ResidentRuntimeError::MissingPreparedCommitment(
                CommitmentTreeId::Composition,
            ))?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::CompositionAndOods)?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let root_destination = self.transcript_input(CairoTranscriptInput::CompositionRoot)?;
        let commitment = &self.commitments[commitment_index].1;
        let root_source = commitment.root_slice();
        let transcript = &self.transcript;
        let composition = &self.composition;
        let oods = &self.oods;
        let workspace = self.workspace;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let capture = capture_with_cursor_rollback(cursor, |cursor| {
            workspace.capture_segment(GraphSegment::CompositionQuotientCommit, |arena| {
                composition
                    .launch()
                    .map_err(ResidentLaunchError::Composition)?;
                commitment.launch()?;
                enqueue_copy_words(arena, root_source, root_destination, 8)?;
                transcript
                    .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                    .map_err(ResidentLaunchError::Transcript)?;
                oods.launch_oods().map_err(ResidentLaunchError::Oods)
            })
        })?;
        self.admit_reused_transcript_segment(capture, transcript_segment)?;
        Ok(())
    }

    /// Capture the OODS-value absorb and quotient challenge boundary. The OODS
    /// evaluator is bound into this segment once its prepared graph is installed;
    /// today the exact device input slot is already stable and mandatory.
    pub fn capture_oods_transcript_boundary(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::OodsAndQuotient)?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let transcript = &self.transcript;
        let oods = &self.oods;
        let workspace = self.workspace;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let capture = capture_with_cursor_rollback(cursor, |cursor| {
            workspace.capture_segment(GraphSegment::OodsEvaluation, |_| {
                transcript
                    .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                    .map_err(ResidentLaunchError::Transcript)?;
                oods.launch_numerator_and_quotient()
                    .map_err(ResidentLaunchError::Oods)
            })
        })?;
        self.admit_reused_transcript_segment(capture, transcript_segment)?;
        Ok(())
    }

    pub fn capture_fri_first_tree(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLayer(0))?;
        let range = self.transcript_segments[transcript_segment]
            .operation_range
            .clone();
        let root_source = self.fri.tree_root(0)?;
        let root_destination = self.transcript_input(CairoTranscriptInput::FriLayerRoot(0))?;
        let challenge_source =
            self.transcript_output(CairoTranscriptOutput::FriFoldingChallenge(0))?;
        let challenge_destination = self.fri.round_challenge_slice(0)?;
        let fri = &self.fri;
        let transcript = &self.transcript;
        let workspace = self.workspace;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let capture = capture_with_cursor_rollback(cursor, |cursor| {
            workspace.capture_segment(GraphSegment::FriLayer(0), |arena| {
                fri.launch_first_tree().map_err(ResidentLaunchError::Fri)?;
                enqueue_copy_words(arena, root_source, root_destination, 8)?;
                transcript
                    .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                    .map_err(ResidentLaunchError::Transcript)?;
                enqueue_copy_words(arena, challenge_source, challenge_destination, 4)
            })
        })?;
        self.admit_reused_transcript_segment(capture, transcript_segment)?;
        Ok(())
    }

    pub fn capture_fri_round(&mut self, round_index: usize) -> Result<(), ResidentRuntimeError> {
        // Validate the round before beginning stream capture.
        let _ = self.fri.round_challenge_slice(round_index)?;
        let segment = fri_round_segment(round_index)?;
        let output_tree = self
            .fri
            .requirements()
            .rounds
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?
            .output_tree;
        let transcript_tail = output_tree
            .map(|tree_index| {
                let layer = u32::try_from(tree_index)
                    .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(tree_index))?;
                let transcript_segment =
                    self.transcript_segment_index(CairoTranscriptSegment::FriLayer(layer))?;
                Ok::<_, ResidentRuntimeError>((
                    transcript_segment,
                    self.transcript_segments[transcript_segment]
                        .operation_range
                        .clone(),
                    self.fri.tree_root(tree_index)?,
                    self.transcript_input(CairoTranscriptInput::FriLayerRoot(layer))?,
                    self.transcript_output(CairoTranscriptOutput::FriFoldingChallenge(layer))?,
                    self.fri.round_challenge_slice(tree_index)?,
                ))
            })
            .transpose()?;
        let fri = &self.fri;
        let transcript = &self.transcript;
        let workspace = self.workspace;
        let fri_fold_launch_mode = workspace.plan().protocol_identity().fri_fold_launch_mode;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let reused_transcript_segment = transcript_tail.as_ref().map(|tail| tail.0);
        let capture = capture_with_cursor_rollback(cursor, |cursor| {
            workspace.capture_segment(segment, |arena| {
                fri.launch_round_with_mode(round_index, fri_fold_launch_mode)
                    .map(|_| ())
                    .map_err(ResidentLaunchError::Fri)?;
                if let Some((
                    _,
                    range,
                    root_source,
                    root_destination,
                    challenge_source,
                    challenge_destination,
                )) = transcript_tail
                {
                    enqueue_copy_words(arena, root_source, root_destination, 8)?;
                    transcript
                        .launch_segment(cursor, generation, range, TranscriptSegmentStart::Resume)
                        .map_err(ResidentLaunchError::Transcript)?;
                    enqueue_copy_words(arena, challenge_source, challenge_destination, 4)?;
                }
                Ok::<(), ResidentLaunchError>(())
            })
        })?;
        if let Some(transcript_segment) = reused_transcript_segment {
            self.admit_reused_transcript_segment(capture, transcript_segment)?;
        }
        Ok(())
    }

    pub fn capture_final_transcript_boundary(&mut self) -> Result<(), ResidentRuntimeError> {
        let last_layer_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLastLayer)?;
        let query_segment =
            self.transcript_segment_index(CairoTranscriptSegment::QueryPowAndPositions)?;
        let last_layer_range = self.transcript_segments[last_layer_segment]
            .operation_range
            .clone();
        let query_range = self.transcript_segments[query_segment]
            .operation_range
            .clone();
        let transcript = &self.transcript;
        let fri_final = &self.fri_final;
        let query_pow = &self.query_pow;
        let fri = &self.fri;
        let decommit = &self.decommit;
        let proof_bundle_sources = self.proof_bundle_sources()?;
        let proof_bundle = self.proof_bundle;
        let proof_bundle_layout = self.workspace.plan().decommit().proof_bundle_layout.clone();
        let trace_tree_count = self.workspace.plan().commitments().len();
        let workspace = self.workspace;
        let cursor = &mut self.transcript_cursor;
        let generation = cursor.generation();
        let capture = capture_with_cursor_rollback(cursor, |cursor| {
            workspace.capture_segment(GraphSegment::OodsQueriesDecommitAssemble, |arena| {
                fri_final.launch().map_err(ResidentLaunchError::FriFinal)?;
                transcript
                    .launch_segment(
                        cursor,
                        generation,
                        last_layer_range,
                        TranscriptSegmentStart::Resume,
                    )
                    .map_err(ResidentLaunchError::Transcript)?;
                query_pow.launch().map_err(ResidentLaunchError::Pow)?;
                transcript
                    .launch_segment(
                        cursor,
                        generation,
                        query_range,
                        TranscriptSegmentStart::Resume,
                    )
                    .map_err(ResidentLaunchError::Transcript)?;
                enqueue_decommit_tail(fri, decommit, trace_tree_count)?;
                enqueue_proof_bundle(
                    arena,
                    &proof_bundle_sources,
                    proof_bundle,
                    &proof_bundle_layout,
                )
            })
        })?;
        self.admit_reused_transcript_segment(capture, last_layer_segment)?;
        self.admit_reused_transcript_segment(capture, query_segment)?;
        Ok(())
    }

    pub fn capture_all_prepared_subgraphs(&mut self) -> Result<(), ResidentRuntimeError> {
        let generation = next_capture_generation(&self.transcript_cursor)?;
        self.begin_transcript_generation(generation)?;
        self.capture_base_commit_only()?;
        self.capture_interaction_relation_and_commit()?;
        self.capture_composition_commit_only()?;
        self.capture_oods_transcript_boundary()?;
        self.capture_fri_first_tree()?;
        for round in 0..self.fri.round_count() {
            self.capture_fri_round(round)?;
        }
        self.capture_final_transcript_boundary()?;
        self.require_transcript_complete()?;
        Ok(())
    }

    pub fn replay_base_commit_only(&mut self) -> Result<(), ResidentRuntimeError> {
        let bootstrap =
            self.transcript_segment_index(CairoTranscriptSegment::BootstrapThroughBase)?;
        let pow = self.transcript_segment_index(CairoTranscriptSegment::InteractionPowAndLookup)?;
        self.admit_transcript_segment_replay(bootstrap)?;
        self.admit_transcript_segment_replay(pow)?;
        self.replay(GraphSegment::IngestWitnessBaseCommit)?;
        self.relation_challenge_generation = self
            .relation_challenge_generation
            .checked_add(1)
            .ok_or(ResidentRuntimeError::StaleRelationChallenges)?;
        Ok(())
    }

    pub fn replay_interaction_relation_and_commit(&mut self) -> Result<(), ResidentRuntimeError> {
        if self.relation_challenge_generation <= self.launched_relation_challenge_generation {
            return Err(ResidentRuntimeError::StaleRelationChallenges);
        }
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::InteractionAndComposition)?;
        self.admit_transcript_segment_replay(transcript_segment)?;
        self.replay(GraphSegment::InteractionCommit)?;
        self.launched_relation_challenge_generation = self.relation_challenge_generation;
        Ok(())
    }

    pub fn replay_composition_commit_only(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::CompositionAndOods)?;
        self.admit_transcript_segment_replay(transcript_segment)?;
        self.replay(GraphSegment::CompositionQuotientCommit)
    }

    pub fn replay_oods_transcript_boundary(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::OodsAndQuotient)?;
        self.admit_transcript_segment_replay(transcript_segment)?;
        self.replay(GraphSegment::OodsEvaluation)
    }

    pub fn replay_fri_first_tree(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLayer(0))?;
        self.admit_transcript_segment_replay(transcript_segment)?;
        self.replay(GraphSegment::FriLayer(0))?;
        self.mark_fri_round_challenge_ready(0)?;
        self.next_fri_round = Some(0);
        Ok(())
    }

    pub fn replay_fri_round(&mut self, round_index: usize) -> Result<(), ResidentRuntimeError> {
        self.require_next_fri_round(round_index)?;
        let generation = *self
            .fri_challenge_generations
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?;
        let launched = self.launched_fri_challenge_generations[round_index];
        if generation <= launched {
            return Err(ResidentRuntimeError::StaleFriChallenge(round_index));
        }
        let output_tree = self
            .fri
            .requirements()
            .rounds
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?
            .output_tree;
        if let Some(tree_index) = output_tree {
            let layer = u32::try_from(tree_index)
                .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(tree_index))?;
            let transcript_segment =
                self.transcript_segment_index(CairoTranscriptSegment::FriLayer(layer))?;
            self.admit_transcript_segment_replay(transcript_segment)?;
        }
        self.replay(fri_round_segment(round_index)?)?;
        self.launched_fri_challenge_generations[round_index] = generation;
        if let Some(tree_index) = output_tree {
            self.mark_fri_round_challenge_ready(tree_index)?;
        }
        self.next_fri_round = Some(round_index + 1);
        Ok(())
    }

    pub fn replay_final_transcript_boundary(&mut self) -> Result<(), ResidentRuntimeError> {
        let last_layer = self.transcript_segment_index(CairoTranscriptSegment::FriLastLayer)?;
        let queries =
            self.transcript_segment_index(CairoTranscriptSegment::QueryPowAndPositions)?;
        self.admit_transcript_segment_replay(last_layer)?;
        self.admit_transcript_segment_replay(queries)?;
        self.replay(GraphSegment::OodsQueriesDecommitAssemble)?;
        self.require_transcript_complete()
    }

    /// Replay the complete transcript-bounded proof DAG. The only host loop is
    /// over true FRI challenge boundaries; component and relation work remains
    /// inside the captured graphs.
    pub fn replay_all_prepared_subgraphs(
        &mut self,
        generation: u64,
    ) -> Result<(), ResidentRuntimeError> {
        self.begin_transcript_generation(generation)?;
        self.replay_base_commit_only()?;
        self.replay_interaction_relation_and_commit()?;
        self.replay_composition_commit_only()?;
        self.replay_oods_transcript_boundary()?;
        self.replay_fri_first_tree()?;
        for round in 0..self.fri.round_count() {
            self.replay_fri_round(round)?;
        }
        self.replay_final_transcript_boundary()
    }

    pub fn captured_graph_count(&self) -> usize {
        self.workspace.graph_count()
    }

    /// Require the protocol topology independently of the replay counters: six
    /// fixed transcript-boundary graphs plus one graph per FRI fold round.
    pub fn require_complete_captured_topology(&self) -> Result<usize, ResidentRuntimeError> {
        require_complete_captured_topology(
            self.fri.round_count(),
            self.captured_graph_count(),
            self.transcript_segment_count(),
        )
    }

    pub fn captured_graph_kernel_node_count(&self) -> Result<u64, ResidentRuntimeError> {
        self.workspace
            .graph_kernel_node_count()
            .ok_or(ResidentRuntimeError::SizeOverflow)
    }

    pub fn prepared_witness_graph_count(&self) -> usize {
        self.witness.len()
    }

    pub fn require_prepared_witness_coverage(&self) -> Result<(), ResidentRuntimeError> {
        let planned = &self.workspace.plan().witness().components;
        if self.witness.len() != planned.len() {
            return Err(ResidentRuntimeError::PreparedWitnessCoverage {
                expected: planned.len(),
                actual: self.witness.len(),
            });
        }
        for (prepared, planned) in self.witness.iter().zip(planned) {
            let reject = |role| ResidentRuntimeError::PreparedWitnessCaptureContract {
                component: planned.component,
                role,
            };
            if prepared.component != planned.component {
                return Err(reject("component identity"));
            }
            if prepared.native_input_producer != planned.native_input_producer {
                return Err(reject("native input provenance"));
            }
            if prepared.writer.kernel_identity().mode != PreparedWitnessMode::RequireEmbeddedAot {
                return Err(reject("embedded AOT mode"));
            }
            if !slices_match_slots(
                prepared.writer.output_columns(),
                &planned.slots.output_columns,
                &planned.requirements.output_column_words,
            ) {
                return Err(reject("base trace destinations"));
            }
            if !slice_matches_slot(
                prepared.writer.lookup_words(),
                planned.slots.lookup_words,
                planned.requirements.lookup_words,
            ) {
                return Err(reject("lookup destination"));
            }
            if !slice_matches_slot(
                prepared.writer.sub_words(),
                planned.slots.sub_words,
                planned.requirements.sub_words,
            ) {
                return Err(reject("subcomponent destination"));
            }
            match (
                &prepared.input_gather,
                &planned.input_gather,
                &prepared.input_seed,
                &planned.input_seed,
                &prepared.input_compact,
                &planned.input_compact,
            ) {
                (None, None, None, None, None, None) => {
                    if !slices_match_slots(
                        prepared.writer.input_columns(),
                        &planned.slots.input_columns,
                        &planned.requirements.input_column_words,
                    ) {
                        return Err(reject("ingested input columns"));
                    }
                }
                (Some(gather), Some(planned_gather), None, None, None, None) => {
                    if !slices_match_slots(
                        gather.consumer_input_columns(),
                        &planned.slots.input_columns,
                        &planned.requirements.input_column_words,
                    ) || gather
                        .consumer_input_columns()
                        .iter()
                        .map(|slice| slice.id())
                        .ne(prepared
                            .writer
                            .input_columns()
                            .iter()
                            .map(|slice| slice.id()))
                        || gather
                            .sources()
                            .iter()
                            .map(|slice| slice.id())
                            .ne(planned_gather.sources.iter().map(|source| source.physical))
                    {
                        return Err(reject("device-edge input gather"));
                    }
                }
                (None, None, Some(seed), Some(planned_seed), None, None) => {
                    let [seed_scalars, seed_pointers] = seed.descriptor_slices();
                    if !slices_match_slots(
                        seed.consumer_input_columns(),
                        &planned.slots.input_columns,
                        &planned.requirements.input_column_words,
                    ) || seed
                        .consumer_input_columns()
                        .iter()
                        .map(|slice| slice.id())
                        .ne(prepared
                            .writer
                            .input_columns()
                            .iter()
                            .map(|slice| slice.id()))
                        || seed.requirements() != &planned_seed.requirements
                        || seed_scalars.id() != planned_seed.slots.scalar_values
                        || seed_pointers.id() != planned_seed.slots.output_pointers
                    {
                        return Err(reject("device-seeded input columns"));
                    }
                }
                (None, None, None, None, Some(compact), Some(planned_compact)) => {
                    let [source_pointers, descriptors, output_pointers] =
                        compact.descriptor_slices();
                    let [tuple_scratch, sort_keys_a, sort_keys_b, sort_indices_a, sort_indices_b, run_heads, run_positions, n_unique, sort_temp, scan_temp] =
                        compact.scratch_slices();
                    let slots = &planned_compact.slots;
                    if !slices_match_slots(
                        compact.consumer_input_columns(),
                        &planned.slots.input_columns,
                        &planned.requirements.input_column_words,
                    ) || compact
                        .consumer_input_columns()
                        .iter()
                        .map(|slice| slice.id())
                        .ne(prepared
                            .writer
                            .input_columns()
                            .iter()
                            .map(|slice| slice.id()))
                        || compact.requirements() != &planned_compact.requirements
                        || compact
                            .sources()
                            .iter()
                            .map(|slice| slice.id())
                            .ne(planned_compact.sources.iter().map(|source| source.physical))
                        || source_pointers.id() != slots.source_pointers
                        || descriptors.id() != slots.descriptors
                        || output_pointers.id() != slots.output_pointers
                        || tuple_scratch.id() != slots.tuple_scratch
                        || sort_keys_a.id() != slots.sort_keys_a
                        || sort_keys_b.id() != slots.sort_keys_b
                        || sort_indices_a.id() != slots.sort_indices_a
                        || sort_indices_b.id() != slots.sort_indices_b
                        || run_heads.id() != slots.run_heads
                        || run_positions.id() != slots.run_positions
                        || n_unique.id() != slots.n_unique
                        || sort_temp.id() != slots.sort_temp
                        || scan_temp.id() != slots.scan_temp
                    {
                        return Err(reject("device-compacted input columns"));
                    }
                }
                _ => return Err(reject("input preparation presence")),
            }
        }

        match (&self.ec_op, self.workspace.plan().ec_op()) {
            (Some(prepared), Some(planned)) => {
                if !CAIRO_SCHEDULE.nodes.iter().any(|node| {
                    node.id == "ec_op_builtin"
                        && node.facts.witness_writer.kind
                            == crate::schedule::WitnessWriterKind::NativeCuda
                        && node.facts.witness_writer.is_capture_safe()
                }) {
                    return Err(ResidentRuntimeError::PreparedEcOpCoverage(
                        "schedule capability",
                    ));
                }
                if !slices_match_slots(
                    prepared.trace_columns(),
                    &planned.slots.trace_columns,
                    &planned.requirements.trace_column_words,
                ) || !slice_matches_slot(
                    prepared.lookup_words(),
                    planned.slots.lookup_words,
                    planned.requirements.lookup_words,
                ) || !slices_match_slots(
                    prepared.partial_input_columns(),
                    &planned.slots.partial_input_columns,
                    &planned.requirements.partial_input_column_words,
                ) || prepared.segment_start_source().id() != planned.slots.segment_start
                    || prepared
                        .multiplicity_destinations()
                        .into_iter()
                        .map(|slice| slice.id())
                        .ne([
                            planned.slots.address_counts,
                            planned.slots.big_counts,
                            planned.slots.small_counts,
                            planned.slots.range_check_8_counts,
                        ])
                {
                    return Err(ResidentRuntimeError::PreparedEcOpCoverage(
                        "arena destination binding",
                    ));
                }
                let partial = self
                    .workspace
                    .plan()
                    .witness()
                    .components
                    .iter()
                    .find(|component| component.component == "partial_ec_mul_generic")
                    .ok_or(ResidentRuntimeError::PreparedEcOpCoverage(
                        "partial_ec_mul_generic consumer",
                    ))?;
                // The ec_op writer materializes 127 partial-input columns:
                // the 126 the consumer recording binds (data + enabler) plus
                // the plan-owned one-past-end iota column the witness kernel
                // computes in-kernel and never binds (see the 126-input
                // contract in recorded_witness_inputs and the iota slot in
                // arena_plan's ec_op workspace).
                if partial.native_input_producer != Some("ec_op_builtin")
                    || planned.slots.partial_input_columns.len()
                        != partial.slots.input_columns.len() + 1
                    || planned.slots.partial_input_columns[..partial.slots.input_columns.len()]
                        != partial.slots.input_columns[..]
                {
                    return Err(ResidentRuntimeError::PreparedEcOpCoverage(
                        "direct partial_ec_mul_generic provenance",
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Err(ResidentRuntimeError::PreparedEcOpCoverage(
                    "prepared graph presence",
                ))
            }
        }

        let expected_fixed = self
            .workspace
            .plan()
            .multiplicity()
            .map_or(0, |multiplicity| multiplicity.fixed_tables.len());
        let actual_fixed = self
            .multiplicity
            .as_ref()
            .map_or(0, |multiplicity| multiplicity.fixed_tables.len());
        if actual_fixed != expected_fixed {
            return Err(ResidentRuntimeError::PreparedFixedTableCoverage {
                expected: expected_fixed,
                actual: actual_fixed,
            });
        }
        if let Some(planned_multiplicity) = self.workspace.plan().multiplicity() {
            let prepared_multiplicity = self.multiplicity.as_ref().ok_or(
                ResidentRuntimeError::PreparedFixedTableCoverage {
                    expected: expected_fixed,
                    actual: 0,
                },
            )?;
            for (prepared, planned) in prepared_multiplicity
                .fixed_tables
                .iter()
                .zip(&planned_multiplicity.fixed_tables)
            {
                let reject = |role| ResidentRuntimeError::PreparedFixedTableCaptureContract {
                    component: planned.plan.component,
                    role,
                };
                if prepared
                    .source_columns()
                    .iter()
                    .zip(&planned.sources)
                    .any(|(prepared, planned)| match planned {
                        PlannedFixedTableSource::Arena(binding) => {
                            prepared.arena_slot() != Some(binding.physical)
                        }
                        PlannedFixedTableSource::RegisteredPedersen18 { column } => {
                            prepared.registered_pedersen_index() != Some(*column)
                        }
                    })
                    || prepared.source_columns().len() != planned.sources.len()
                {
                    return Err(reject("preprocessed sources"));
                }
                if prepared.multiplicity_slab().is_none_or(|slice| {
                    !slice_matches_slot(
                        slice,
                        planned.multiplicity.physical,
                        planned.plan.slab_words,
                    )
                }) {
                    return Err(reject("multiplicity slab"));
                }
                if prepared
                    .trace_outputs()
                    .iter()
                    .map(|slice| slice.id())
                    .ne(planned.slots.trace_outputs.iter().copied())
                {
                    return Err(reject("base trace destinations"));
                }
                let lookup_words = planned
                    .plan
                    .row_count
                    .checked_mul(prepared.requirements().lookup_output_count)
                    .ok_or(ResidentRuntimeError::SizeOverflow)?;
                if prepared.lookup_output_slab().is_none_or(|slice| {
                    !slice_matches_slot(slice, planned.slots.lookup_output, lookup_words)
                }) {
                    return Err(reject("lookup destination"));
                }
            }
            let prepared_memory = prepared_multiplicity.memory_traces.is_some();
            let planned_memory = planned_multiplicity.memory_traces.is_some();
            if prepared_memory != planned_memory {
                return Err(ResidentRuntimeError::PreparedFixedTableCaptureContract {
                    component: "memory_address_to_id+memory_id_to_big",
                    role: "PreparedMemoryBaseTraceGraph presence",
                });
            }
            if let (Some(prepared), Some(planned)) = (
                prepared_multiplicity.memory_traces.as_ref(),
                planned_multiplicity.memory_traces.as_ref(),
            ) {
                let reject = |role| ResidentRuntimeError::PreparedFixedTableCaptureContract {
                    component: "memory_id_to_big->range_check_9_9",
                    role,
                };
                if !slice_matches_slot(
                    prepared.rc99_lut(),
                    planned.rc99_lut.physical,
                    planned.plan.rc99_lut_words,
                ) || prepared.rc99_table_size() != planned.plan.rc99_lut_words
                {
                    return Err(reject("canonical LUT binding"));
                }
                if !slice_matches_slot(
                    prepared.rc99_counts(),
                    planned.rc99_counts.physical,
                    planned.plan.rc99_count_words,
                ) {
                    return Err(reject("multiplicity slab binding"));
                }
                let Some(range_check) = planned_multiplicity
                    .fixed_tables
                    .iter()
                    .find(|fixed| fixed.plan.component == "range_check_9_9")
                else {
                    return Err(reject("downstream materializer"));
                };
                if range_check.multiplicity.physical != planned.rc99_counts.physical {
                    return Err(reject("shared downstream multiplicity slab"));
                }
            }
            if planned_memory
                && !["memory_address_to_id", "memory_id_to_big"]
                    .into_iter()
                    .all(|component| {
                        CAIRO_SCHEDULE.nodes.iter().any(|node| {
                            node.id == component
                                && node.facts.witness_writer.kind
                                    == crate::schedule::WitnessWriterKind::NativeCuda
                                && node.facts.witness_writer.is_capture_safe()
                        })
                    })
            {
                return Err(ResidentRuntimeError::PreparedFixedTableCaptureContract {
                    component: "memory_address_to_id+memory_id_to_big",
                    role: "schedule capability is not backed by PreparedMemoryBaseTraceGraph",
                });
            }
        }
        Ok(())
    }

    pub fn workspace_proof_bundle_bytes(&self) -> Option<u64> {
        self.workspace
            .plan()
            .decommit()
            .proof_bundle_layout
            .total_words
            .checked_mul(core::mem::size_of::<u32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
    }

    /// Host-channel migration path. Device transcript mode writes directly to
    /// [`Self::fri_round_challenge_destination`] then calls
    /// [`Self::mark_fri_round_challenge_ready`].
    pub fn upload_fri_round_challenge(
        &mut self,
        round_index: usize,
        value: SecureField,
    ) -> Result<(), ResidentRuntimeError> {
        self.fri
            .upload_round_challenge_at_transcript_boundary(round_index, value)?;
        self.mark_fri_round_challenge_ready(round_index)
    }

    pub fn fri_round_challenge_destination(
        &self,
        round_index: usize,
    ) -> Result<ArenaSlice, ResidentRuntimeError> {
        Ok(self.fri.round_challenge_slice(round_index)?)
    }

    pub fn mark_fri_round_challenge_ready(
        &mut self,
        round_index: usize,
    ) -> Result<(), ResidentRuntimeError> {
        let generation = self
            .fri_challenge_generations
            .get_mut(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?;
        *generation = generation
            .checked_add(1)
            .ok_or(ResidentRuntimeError::FriRoundIndexTooLarge(round_index))?;
        Ok(())
    }

    pub fn upload_relation_challenges(
        &mut self,
        challenges: RelationChallenges<'_>,
    ) -> Result<(), ResidentRuntimeError> {
        self.relation
            .upload_challenges_at_transcript_boundary(challenges)?;
        self.relation_challenge_generation = self
            .relation_challenge_generation
            .checked_add(1)
            .ok_or(ResidentRuntimeError::StaleRelationChallenges)?;
        Ok(())
    }

    pub fn launch_base_commit_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        if let Some(execution_tables) = &self.execution_tables {
            execution_tables.launch()?;
        }
        if let Some(multiplicity) = &self.multiplicity {
            multiplicity.clear.launch()?;
            if let Some(seed) = &multiplicity.public_memory_seed {
                seed.launch()?;
            }
        }
        enqueue_witness_lane_levels(
            self.workspace.arena(),
            &self.witness,
            self.ec_op.as_ref(),
            &self.witness_lane_levels,
            self.multiplicity.as_ref(),
        )?;
        if let Some(multiplicity) = &self.multiplicity {
            if let Some(memory) = &multiplicity.memory_traces {
                memory.launch()?;
            }
            for fixed in &multiplicity.fixed_tables {
                fixed.launch()?;
            }
        }
        self.base_interpolation.launch()?;
        self.commitment(CommitmentTreeId::Base)?.launch()?;
        self.stage_commitment_root_for_transcript(
            CommitmentTreeId::Base,
            CairoTranscriptInput::BaseRoot,
        )?;
        let bootstrap =
            self.transcript_segment_index(CairoTranscriptSegment::BootstrapThroughBase)?;
        let pow = self.transcript_segment_index(CairoTranscriptSegment::InteractionPowAndLookup)?;
        self.launch_transcript_segment_eager(bootstrap)?;
        self.interaction_pow.launch()?;
        self.launch_transcript_segment_eager(pow)?;
        self.publish_relation_challenges_from_transcript()?;
        Ok(())
    }

    pub fn launch_interaction_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        if self.relation_challenge_generation <= self.launched_relation_challenge_generation {
            return Err(ResidentRuntimeError::StaleRelationChallenges);
        }
        let identity = self.workspace.plan().protocol_identity();
        self.relation.launch_with_modes(
            self.workspace.plan().relation().launch_mode,
            identity.relation_tail_mode,
        )?;
        self.interaction_interpolation.launch()?;
        self.commitment(CommitmentTreeId::Interaction)?.launch()?;
        self.stage_interaction_claim_for_transcript()?;
        self.stage_commitment_root_for_transcript(
            CommitmentTreeId::Interaction,
            CairoTranscriptInput::InteractionRoot,
        )?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::InteractionAndComposition)?;
        self.launch_transcript_segment_eager(transcript_segment)?;
        self.launched_relation_challenge_generation = self.relation_challenge_generation;
        Ok(())
    }

    pub fn launch_composition_commit_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        self.composition.launch()?;
        self.commitment(CommitmentTreeId::Composition)?.launch()?;
        self.stage_commitment_root_for_transcript(
            CommitmentTreeId::Composition,
            CairoTranscriptInput::CompositionRoot,
        )?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::CompositionAndOods)?;
        self.launch_transcript_segment_eager(transcript_segment)?;
        self.oods.launch_oods()?;
        Ok(())
    }

    pub fn launch_oods_transcript_boundary_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::OodsAndQuotient)?;
        self.launch_transcript_segment_eager(transcript_segment)?;
        self.oods.launch_numerator_and_quotient()?;
        Ok(())
    }

    pub fn launch_fri_first_tree_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        self.fri.launch_first_tree()?;
        self.stage_fri_root_for_transcript(0, 0)?;
        let transcript_segment =
            self.transcript_segment_index(CairoTranscriptSegment::FriLayer(0))?;
        self.launch_transcript_segment_eager(transcript_segment)?;
        self.publish_fri_challenge_from_transcript(0)?;
        self.next_fri_round = Some(0);
        Ok(())
    }

    pub fn launch_fri_round_eager(
        &mut self,
        round_index: usize,
    ) -> Result<Option<usize>, ResidentRuntimeError> {
        self.require_next_fri_round(round_index)?;
        let fri_fold_launch_mode = self
            .workspace
            .plan()
            .protocol_identity()
            .fri_fold_launch_mode;
        let generation = *self
            .fri_challenge_generations
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?;
        let launched = &mut self.launched_fri_challenge_generations[round_index];
        if generation <= *launched {
            return Err(ResidentRuntimeError::StaleFriChallenge(round_index));
        }
        let tree = self
            .fri
            .launch_round_with_mode(round_index, fri_fold_launch_mode)?;
        *launched = generation;
        if let Some(tree_index) = tree {
            let layer = u32::try_from(tree_index)
                .map_err(|_| ResidentRuntimeError::FriRoundIndexTooLarge(tree_index))?;
            self.stage_fri_root_for_transcript(tree_index, layer)?;
            let transcript_segment =
                self.transcript_segment_index(CairoTranscriptSegment::FriLayer(layer))?;
            self.launch_transcript_segment_eager(transcript_segment)?;
            self.publish_fri_challenge_from_transcript(tree_index)?;
        }
        self.next_fri_round = Some(round_index + 1);
        Ok(tree)
    }

    pub fn launch_final_transcript_boundary_eager(&mut self) -> Result<(), ResidentRuntimeError> {
        let last_layer = self.transcript_segment_index(CairoTranscriptSegment::FriLastLayer)?;
        let queries =
            self.transcript_segment_index(CairoTranscriptSegment::QueryPowAndPositions)?;
        self.fri_final.launch()?;
        self.launch_transcript_segment_eager(last_layer)?;
        self.query_pow.launch()?;
        self.launch_transcript_segment_eager(queries)?;
        self.decommit.launch_query_normalization()?;
        let trace_tree_count = self.workspace.plan().commitments().len();
        for tree_index in 0..trace_tree_count {
            self.decommit.launch_trace_tree(tree_index)?;
        }
        for tree_index in 0..self.fri.tree_count() {
            self.decommit
                .launch_fri_tree(trace_tree_count + tree_index)?;
        }
        let sources = self.proof_bundle_sources()?;
        enqueue_proof_bundle(
            self.workspace.arena(),
            &sources,
            self.proof_bundle,
            &self.workspace.plan().decommit().proof_bundle_layout,
        )
        .map_err(|error| ResidentRuntimeError::Graph(GraphError::Enqueue(Box::new(error))))?;
        self.require_transcript_complete()
    }

    pub fn read_commitment_root(
        &self,
        id: CommitmentTreeId,
    ) -> Result<Blake2sHash, ResidentRuntimeError> {
        if id != CommitmentTreeId::Preprocessed {
            return Ok(self.commitment(id)?.read_root_at_transcript_boundary()?);
        }
        let mut root = Blake2sHash::default();
        unsafe {
            self.workspace.arena().context().memcpy_d2h_async(
                root.0.as_mut_ptr().cast(),
                self.fixed_preprocessed_root.as_void_ptr().cast_const(),
                core::mem::size_of::<Blake2sHash>(),
            )?;
        }
        self.workspace.arena().context().sync()?;
        Ok(root)
    }

    pub fn read_interaction_transcript_boundary(
        &self,
    ) -> Result<InteractionTranscriptBoundary, ResidentRuntimeError> {
        let interaction = self.commitment(CommitmentTreeId::Interaction)?;
        let outputs: Vec<_> = self.relation.outputs().collect();
        let mut words = vec![[0u32; 4]; outputs.len()];
        let mut root = Blake2sHash::default();
        for (output, destination) in outputs.iter().zip(&mut words) {
            unsafe {
                self.workspace.arena().context().memcpy_d2h_async(
                    destination.as_mut_ptr().cast(),
                    output.claimed_sum.as_void_ptr().cast_const(),
                    core::mem::size_of_val(destination),
                )?;
            }
        }
        unsafe {
            self.workspace.arena().context().memcpy_d2h_async(
                root.0.as_mut_ptr().cast::<c_void>(),
                interaction.root_slice().as_void_ptr().cast_const(),
                core::mem::size_of::<Blake2sHash>(),
            )?;
        }
        self.workspace.arena().context().sync()?;

        let claimed_sums = outputs
            .into_iter()
            .zip(words)
            .map(|(output, coordinates)| ResidentClaimedSum {
                batch: self.workspace.plan().relation().execution.batches[output.batch_index],
                instance_index: output.instance_index,
                value: SecureField::from_m31_array(coordinates.map(M31::from_u32_unchecked)),
            })
            .collect();
        Ok(InteractionTranscriptBoundary { root, claimed_sums })
    }

    pub fn read_fri_tree_root(
        &self,
        tree_index: usize,
    ) -> Result<Blake2sHash, ResidentRuntimeError> {
        Ok(self.fri.read_tree_root(tree_index)?)
    }

    pub fn fri_round_output_tree(
        &self,
        round_index: usize,
    ) -> Result<Option<usize>, ResidentRuntimeError> {
        Ok(self
            .fri
            .requirements()
            .rounds
            .get(round_index)
            .ok_or_else(|| PreparedFriError::InvalidRoundIndex(round_index))?
            .output_tree)
    }

    pub fn proof_assembly_shape(&self) -> &Blake2sProofAssemblyShape {
        &self.workspace.plan().decommit().proof_shape
    }

    fn proof_bundle_sources(&self) -> Result<ResidentProofBundleSources, ResidentRuntimeError> {
        let commitments = [
            self.commitment_root_slice(CommitmentTreeId::Preprocessed)?,
            self.commitment_root_slice(CommitmentTreeId::Base)?,
            self.commitment_root_slice(CommitmentTreeId::Interaction)?,
            self.commitment_root_slice(CommitmentTreeId::Composition)?,
        ];
        let fri_commitments = (0..self.fri.tree_count())
            .map(|tree| self.fri.tree_root(tree).map_err(ResidentRuntimeError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ResidentProofBundleSources {
            commitments,
            interaction_claim: self.transcript_input(CairoTranscriptInput::InteractionClaim)?,
            interaction_pow: self.interaction_pow.nonce_destination(),
            sampled_values: self.transcript_input(CairoTranscriptInput::OodsSampledValues)?,
            fri_commitments,
            final_line_poly: self.fri_final.transcript_destination(),
            query_pow: self.query_pow.nonce_destination(),
            decommitment: self.decommit.assembly_slice(),
        })
    }

    /// The production host boundary: one contiguous D2H copy followed by the
    /// proof stream's sole hot synchronization.
    pub fn read_proof_bundle_once(&self) -> Result<ResidentProofBundle, ResidentRuntimeError> {
        let planned = self.workspace.plan().decommit();
        let words = planned.proof_bundle_layout.total_words;
        let mut host = vec![0u32; words];
        unsafe {
            self.workspace.arena().context().memcpy_d2h_async(
                host.as_mut_ptr().cast(),
                self.proof_bundle.as_void_ptr().cast_const(),
                words.checked_mul(core::mem::size_of::<u32>()).ok_or(
                    ResidentRuntimeError::TranscriptBindingTooSmall {
                        role: "resident proof bundle",
                        required_words: words,
                        actual_words: self.proof_bundle.len_words(),
                    },
                )?,
            )?;
        }
        self.workspace.arena().context().sync()?;
        Ok(ResidentProofBundle::decode(
            host,
            &planned.proof_bundle_layout,
            &planned.proof_shape,
        )?)
    }

    /// Resident compact decommit ABI. The production final-bundle copier must
    /// include this slice in its one D2H transfer instead of calling the
    /// standalone read helper below.
    pub fn decommit_assembly_slice(&self) -> ArenaSlice {
        self.decommit.assembly_slice()
    }

    pub fn decommit_assembly_capacity_words(&self) -> usize {
        self.decommit.requirements().assembly_words
    }

    /// Migration/test boundary. Production proof assembly uses the single
    /// proof-bundle D2H and then calls `DecommitAssembly::decode` on its prefix.
    pub fn read_decommit_assembly_once(&self) -> Result<DecommitAssembly, ResidentRuntimeError> {
        Ok(self.decommit.read_assembly_once()?)
    }

    /// DIAGNOSTIC-ONLY replay of the witness-generation prefix without base
    /// interpolation or commitment, so the trace auditor can inspect the exact
    /// prepared witness writers. Call this before any base-commit replay: the
    /// ingest inputs are not live after `ProofEpoch::Witness` and are not
    /// re-uploaded by this diagnostic seam.
    pub fn replay_witness_only_for_diagnostics(&self) -> Result<(), ResidentRuntimeError> {
        if let Some(execution_tables) = &self.execution_tables {
            execution_tables.launch()?;
        }
        if let Some(multiplicity) = &self.multiplicity {
            multiplicity.clear.launch()?;
            if let Some(seed) = &multiplicity.public_memory_seed {
                seed.launch()?;
            }
        }
        enqueue_witness_lane_levels(
            self.workspace.arena(),
            &self.witness,
            self.ec_op.as_ref(),
            &self.witness_lane_levels,
            self.multiplicity.as_ref(),
        )?;
        if let Some(multiplicity) = &self.multiplicity {
            if let Some(memory) = &multiplicity.memory_traces {
                memory.launch()?;
            }
            for fixed in &multiplicity.fixed_tables {
                fixed.launch()?;
            }
        }
        self.workspace.arena().context().sync()?;
        Ok(())
    }

    /// DIAGNOSTIC-ONLY base-trace readback: every planned `BaseTrace`
    /// evaluation column of one component part (ordinals `0..width`, each
    /// `padded_rows` u32 words), copied D2H and drained with one stream
    /// synchronization.
    ///
    /// `pub` (rather than `pub(crate)`) solely for the diagnostic trace-audit
    /// runner (`tests/resident_trace_audit.rs`), exactly like the
    /// `interaction_claim_from_flattened` promotion — not a stable API
    /// surface. Production replay must never call this: it crosses PCIe and
    /// synchronizes, deliberately violating the resident hot-path budget.
    ///
    /// Content is only meaningful immediately after a pre-base-commit
    /// [`Self::replay_witness_only_for_diagnostics`]. Base interpolation may
    /// overwrite dead evaluations in place, and later epochs may reuse pooled
    /// arena slots.
    pub fn read_base_trace_columns_for_diagnostics(
        &self,
        component: &'static str,
        part: TracePartId,
    ) -> Result<Vec<Vec<u32>>, ResidentRuntimeError> {
        let missing = |ordinal: u32| ResidentRuntimeError::MissingCommitmentSource {
            id: CommitmentTreeId::Base,
            source: CommitmentColumnSource::Trace {
                component,
                part,
                purpose: BufferPurpose::BaseTrace,
                ordinal,
            },
        };
        let plan = self.workspace.plan();
        let mut columns: Vec<Vec<u32>> = Vec::new();
        loop {
            let ordinal = u32::try_from(columns.len()).map_err(|_| missing(u32::MAX))?;
            let Some((logical, binding)) = plan.find(
                Some(component),
                Some(part),
                BufferPurpose::BaseTrace,
                ordinal,
            ) else {
                break;
            };
            let words = logical.len_words;
            let slice = bind_arena_binding(self.workspace.arena(), binding)?;
            if slice.len_words() < words {
                return Err(ResidentRuntimeError::TranscriptBindingTooSmall {
                    role: "diagnostic base-trace column",
                    required_words: words,
                    actual_words: slice.len_words(),
                });
            }
            let bytes = words.checked_mul(core::mem::size_of::<u32>()).ok_or(
                ResidentRuntimeError::TranscriptBindingTooSmall {
                    role: "diagnostic base-trace column",
                    required_words: words,
                    actual_words: slice.len_words(),
                },
            )?;
            let mut host = vec![0u32; words];
            // SAFETY: `slice` is an arena-owned device range of at least
            // `words` words (checked above); `host` owns `words` writable
            // words and its heap allocation is address-stable across the move
            // into `columns`. The copy is drained by the sync below before
            // any host read.
            unsafe {
                self.workspace.arena().context().memcpy_d2h_async(
                    host.as_mut_ptr().cast(),
                    slice.as_void_ptr().cast_const(),
                    bytes,
                )?;
            }
            columns.push(host);
        }
        if columns.is_empty() {
            return Err(missing(0));
        }
        self.workspace.arena().context().sync()?;
        Ok(columns)
    }

    fn commitment(
        &self,
        id: CommitmentTreeId,
    ) -> Result<&PreparedResidentCommitment<'a>, ResidentRuntimeError> {
        self.commitments
            .iter()
            .find_map(|(candidate, graph)| (*candidate == id).then_some(graph))
            .ok_or(ResidentRuntimeError::MissingPreparedCommitment(id))
    }

    pub fn commitment_root_slice(
        &self,
        id: CommitmentTreeId,
    ) -> Result<ArenaSlice, ResidentRuntimeError> {
        if id == CommitmentTreeId::Preprocessed {
            Ok(self.fixed_preprocessed_root)
        } else {
            Ok(self.commitment(id)?.root_slice())
        }
    }

    pub fn commitment_retained_layers_bottom_up(
        &self,
        id: CommitmentTreeId,
    ) -> Result<&[ArenaSlice], ResidentRuntimeError> {
        if id == CommitmentTreeId::Preprocessed {
            Ok(&self.fixed_preprocessed_retained_layers)
        } else {
            Ok(self.commitment(id)?.retained_layers_bottom_up())
        }
    }

    fn transcript_segment_index(
        &self,
        semantic: CairoTranscriptSegment,
    ) -> Result<usize, ResidentRuntimeError> {
        self.transcript_segments
            .iter()
            .position(|segment| segment.segment == semantic)
            .ok_or(ResidentRuntimeError::MissingTranscriptSegment(semantic))
    }

    fn copy_transcript_words(
        &self,
        role: &'static str,
        source: ArenaSlice,
        destination: ArenaSlice,
        words: usize,
    ) -> Result<(), ResidentRuntimeError> {
        if source.len_words() < words {
            return Err(ResidentRuntimeError::TranscriptBindingTooSmall {
                role,
                required_words: words,
                actual_words: source.len_words(),
            });
        }
        if destination.len_words() < words {
            return Err(ResidentRuntimeError::TranscriptBindingTooSmall {
                role,
                required_words: words,
                actual_words: destination.len_words(),
            });
        }
        let bytes = words.checked_mul(core::mem::size_of::<u32>()).ok_or(
            ResidentRuntimeError::TranscriptBindingTooSmall {
                role,
                required_words: words,
                actual_words: destination.len_words(),
            },
        )?;
        // SAFETY: both slices are arena-owned and the word capacities above
        // cover the exact non-overlapping semantic handoff.
        unsafe {
            self.workspace.arena().context().memcpy_d2d_async(
                destination.as_void_ptr(),
                source.as_void_ptr().cast_const(),
                bytes,
            )?;
        }
        Ok(())
    }

    fn admit_reused_transcript_segment(
        &mut self,
        capture: GraphCaptureStatus,
        transcript_segment: usize,
    ) -> Result<(), ResidentRuntimeError> {
        admit_on_graph_reuse(capture, || {
            self.admit_transcript_segment_replay(transcript_segment)
        })
    }

    fn replay(&self, segment: GraphSegment) -> Result<(), ResidentRuntimeError> {
        self.workspace.replay_segment(segment)?;
        Ok(())
    }

    fn require_next_fri_round(&self, round_index: usize) -> Result<(), ResidentRuntimeError> {
        let expected = self.next_fri_round.unwrap_or(0);
        if self.next_fri_round.is_none() || round_index != expected {
            return Err(ResidentRuntimeError::FriRoundOutOfOrder {
                expected,
                actual: round_index,
            });
        }
        Ok(())
    }
}

fn admit_on_graph_reuse<E>(
    capture: GraphCaptureStatus,
    admit: impl FnOnce() -> Result<(), E>,
) -> Result<(), E> {
    if capture.is_reused() {
        admit()?;
    }
    Ok(())
}

fn capture_with_cursor_rollback<T, E>(
    cursor: &mut TranscriptSegmentCursor,
    capture: impl FnOnce(&mut TranscriptSegmentCursor) -> Result<T, E>,
) -> Result<T, E> {
    let checkpoint = cursor.clone();
    match capture(cursor) {
        Ok(value) => Ok(value),
        Err(error) => {
            *cursor = checkpoint;
            Err(error)
        }
    }
}

fn next_capture_generation(cursor: &TranscriptSegmentCursor) -> Result<u64, ResidentRuntimeError> {
    cursor
        .generation()
        .checked_add(1)
        .ok_or(ResidentRuntimeError::SizeOverflow)
}

fn enqueue_copy_words(
    arena: &stwo_backend_cuda::DeviceArena,
    source: ArenaSlice,
    destination: ArenaSlice,
    words: usize,
) -> Result<(), ResidentLaunchError> {
    if source.len_words() < words || destination.len_words() < words {
        return Err(ResidentLaunchError::Binding("device word copy capacity"));
    }
    let bytes = words
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(ResidentLaunchError::Binding("device word copy overflow"))?;
    // SAFETY: both arena ranges were capacity-checked and semantic producers
    // and consumers are distinct live slots.
    unsafe {
        arena
            .context()
            .memcpy_d2d_async(
                destination.as_void_ptr(),
                source.as_void_ptr().cast_const(),
                bytes,
            )
            .map_err(ResidentLaunchError::Cuda)?;
    }
    Ok(())
}

fn enqueue_proof_bundle(
    arena: &stwo_backend_cuda::DeviceArena,
    sources: &ResidentProofBundleSources,
    destination: ArenaSlice,
    layout: &ResidentProofBundleLayout,
) -> Result<(), ResidentLaunchError> {
    if destination.len_words() < layout.total_words
        || layout.commitments.len() != sources.commitments.len() * 8
        || layout.fri_commitments.len() != sources.fri_commitments.len() * 8
    {
        return Err(ResidentLaunchError::Binding("resident proof bundle layout"));
    }
    for (index, source) in sources.commitments.iter().copied().enumerate() {
        let start = layout.commitments.start + index * 8;
        enqueue_bundle_range(arena, source, destination, start, 8)?;
    }
    enqueue_bundle_range(
        arena,
        sources.interaction_claim,
        destination,
        layout.interaction_claim.start,
        layout.interaction_claim.len(),
    )?;
    enqueue_bundle_range(
        arena,
        sources.interaction_pow,
        destination,
        layout.interaction_pow.start,
        layout.interaction_pow.len(),
    )?;
    enqueue_bundle_range(
        arena,
        sources.sampled_values,
        destination,
        layout.sampled_values.start,
        layout.sampled_values.len(),
    )?;
    for (index, source) in sources.fri_commitments.iter().copied().enumerate() {
        let start = layout.fri_commitments.start + index * 8;
        enqueue_bundle_range(arena, source, destination, start, 8)?;
    }
    enqueue_bundle_range(
        arena,
        sources.final_line_poly,
        destination,
        layout.final_line_poly.start,
        layout.final_line_poly.len(),
    )?;
    enqueue_bundle_range(
        arena,
        sources.query_pow,
        destination,
        layout.query_pow.start,
        layout.query_pow.len(),
    )?;
    enqueue_bundle_range(
        arena,
        sources.decommitment,
        destination,
        layout.decommitment.start,
        layout.decommitment.len(),
    )
}

fn enqueue_bundle_range(
    arena: &stwo_backend_cuda::DeviceArena,
    source: ArenaSlice,
    destination: ArenaSlice,
    destination_offset_words: usize,
    words: usize,
) -> Result<(), ResidentLaunchError> {
    let destination_end = destination_offset_words
        .checked_add(words)
        .ok_or(ResidentLaunchError::Binding("proof bundle offset overflow"))?;
    if source.len_words() < words || destination.len_words() < destination_end {
        return Err(ResidentLaunchError::Binding("proof bundle copy capacity"));
    }
    let bytes = words
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(ResidentLaunchError::Binding("proof bundle byte overflow"))?;
    unsafe {
        arena
            .context()
            .memcpy_d2d_async(
                destination
                    .as_u32_ptr()
                    .add(destination_offset_words)
                    .cast(),
                source.as_void_ptr().cast_const(),
                bytes,
            )
            .map_err(ResidentLaunchError::Cuda)?;
    }
    Ok(())
}

fn enqueue_decommit_tail(
    fri: &PreparedFriGraph<'_>,
    decommit: &PreparedDecommitGraph<'_>,
    trace_tree_count: usize,
) -> Result<(), ResidentLaunchError> {
    decommit
        .launch_query_normalization()
        .map_err(ResidentLaunchError::Decommit)?;
    for tree_index in 0..trace_tree_count {
        decommit
            .launch_trace_tree(tree_index)
            .map_err(ResidentLaunchError::Decommit)?;
    }
    for tree_index in 0..fri.tree_count() {
        decommit
            .launch_fri_tree(trace_tree_count + tree_index)
            .map_err(ResidentLaunchError::Decommit)?;
    }
    Ok(())
}

fn enqueue_claimed_sums(
    arena: &stwo_backend_cuda::DeviceArena,
    sources: &[ArenaSlice],
    destination: ArenaSlice,
) -> Result<(), ResidentLaunchError> {
    let required_words = sources
        .len()
        .checked_mul(4)
        .ok_or(ResidentLaunchError::Binding("claimed-sum width overflow"))?;
    if destination.len_words() != required_words {
        return Err(ResidentLaunchError::Binding(
            "claimed-sum transcript width mismatch",
        ));
    }
    for (index, source) in sources.iter().copied().enumerate() {
        if source.len_words() < 4 {
            return Err(ResidentLaunchError::Binding(
                "claimed-sum source is too small",
            ));
        }
        let offset = index
            .checked_mul(4)
            .ok_or(ResidentLaunchError::Binding("claimed-sum offset overflow"))?;
        // SAFETY: exact aggregate width and each four-word source were checked.
        unsafe {
            arena
                .context()
                .memcpy_d2d_async(
                    destination.as_u32_ptr().add(offset).cast(),
                    source.as_void_ptr().cast_const(),
                    4 * core::mem::size_of::<u32>(),
                )
                .map_err(ResidentLaunchError::Cuda)?;
        }
    }
    Ok(())
}

fn interaction_outputs_in_cairo_order(
    workspace: &GraphWorkspace,
    relation: &PreparedRelationGraph<'_>,
) -> Result<Vec<ArenaSlice>, ResidentRuntimeError> {
    let execution = &workspace.plan().relation().execution;
    let mut outputs = relation.outputs().collect::<Vec<_>>();
    outputs.sort_by_key(|output| {
        let batch = execution.batches[output.batch_index];
        let component = crate::schedule_table::CAIRO_COMMITMENT_COMPONENT_ORDER
            .iter()
            .position(|candidate| *candidate == batch.component)
            .unwrap_or(usize::MAX);
        let (part, instance) = match batch.trace_part {
            RelationTracePart::Component => (0u8, output.instance_index),
            RelationTracePart::EachMemoryBig => (1, output.instance_index),
            RelationTracePart::MemorySmall => (2, output.instance_index),
        };
        (component, part, instance)
    });
    for output in &outputs {
        let batch = execution.batches[output.batch_index];
        if !crate::schedule_table::CAIRO_COMMITMENT_COMPONENT_ORDER.contains(&batch.component) {
            return Err(ResidentRuntimeError::UnknownTranscriptRelationComponent(
                batch.component,
            ));
        }
    }
    Ok(outputs
        .into_iter()
        .map(|output| output.claimed_sum)
        .collect())
}

fn resident_decommit_sources(
    workspace: &GraphWorkspace,
    commitments: &[(CommitmentTreeId, PreparedResidentCommitment<'_>)],
    fixed_preprocessed_root: ArenaSlice,
    fixed_preprocessed_layers: &[ArenaSlice],
    fri: &PreparedFriGraph<'_>,
) -> Result<Vec<DecommitTreeSources>, ResidentRuntimeError> {
    let arena = workspace.arena();
    let planned_decommit = workspace.plan().decommit();
    let mut sources = Vec::with_capacity(planned_decommit.config.trees.len());
    for (tree_index, planned) in workspace.plan().commitments().iter().enumerate() {
        let Some(DecommitTreeGeometry::Trace(geometry)) =
            planned_decommit.config.trees.get(tree_index)
        else {
            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                "trace commitment/decommit order differs",
            ));
        };
        let expected_role = match planned.id {
            CommitmentTreeId::Preprocessed => stwo_backend_cuda::TraceTreeRole::Preprocessed,
            CommitmentTreeId::Base => stwo_backend_cuda::TraceTreeRole::Base,
            CommitmentTreeId::Interaction => stwo_backend_cuda::TraceTreeRole::Interaction,
            CommitmentTreeId::Composition => stwo_backend_cuda::TraceTreeRole::Composition,
            CommitmentTreeId::Fri(_) => {
                return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                    "FRI commitment appeared in the trace prefix",
                ));
            }
        };
        if geometry.role != expected_role
            || geometry.leaf_log_size != planned.config.lifting_log_size
            || geometry.unretained_bottom_layers != planned.config.unretained_bottom_layers
            || geometry.groups.len() != planned.grouped_column_log_sizes.len()
        {
            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                "trace commitment/decommit geometry differs",
            ));
        }
        let prepared_commitment = commitments
            .iter()
            .find_map(|(id, graph)| (*id == planned.id).then_some(graph));
        if planned.id != CommitmentTreeId::Preprocessed && prepared_commitment.is_none() {
            return Err(ResidentRuntimeError::MissingPreparedCommitment(planned.id));
        }
        let commit_groups = commitment_groups(workspace, planned)?;
        let groups = commit_groups
            .into_iter()
            .zip(&geometry.groups)
            .enumerate()
            .map(|(group_index, (group, geometry))| {
                if group.columns.len() != geometry.columns.len()
                    || group
                        .columns
                        .iter()
                        .zip(&geometry.columns)
                        .any(|(column, geometry)| column.log_size != geometry.coefficient_log_size)
                {
                    return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                        "trace decommit columns differ from prepared commit columns",
                    ));
                }
                let planned_retained = planned.retained_evaluation_groups.get(group_index).ok_or(
                    ResidentRuntimeError::DecommitTopologyMismatch(
                        "trace retained-evaluation plan differs from decommit geometry",
                    ),
                )?;
                let columns = match geometry.mode {
                    stwo_backend_cuda::DecommitSourceMode::RecomputeQueriedLde => {
                        if planned_retained.is_some() {
                            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                                "recomputed trace group owns an unexpected retained evaluation",
                            ));
                        }
                        group
                            .columns
                            .into_iter()
                            .map(|column| DecommitColumnSource::Coefficients(column.coefficients))
                            .collect()
                    }
                    stwo_backend_cuda::DecommitSourceMode::ResidentEvaluations => {
                        let graph = prepared_commitment.ok_or(
                            ResidentRuntimeError::DecommitTopologyMismatch(
                                "fixed preprocessed tree cannot use per-proof retained LDEs",
                            ),
                        )?;
                        let actual = graph
                            .retained_evaluations()
                            .get(group_index)
                            .and_then(Option::as_ref)
                            .ok_or(ResidentRuntimeError::DecommitTopologyMismatch(
                                "prepared commitment did not retain its planned evaluation group",
                            ))?;
                        let expected = planned_retained.as_ref().ok_or(
                            ResidentRuntimeError::DecommitTopologyMismatch(
                                "resident trace group has no arena binding",
                            ),
                        )?;
                        if actual.len() != expected.len() {
                            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                                "retained trace evaluation width differs from the arena plan",
                            ));
                        }
                        for (&actual, expected) in actual.iter().zip(expected) {
                            require_same_slice(
                                "retained trace evaluation binding differs from commit output",
                                actual,
                                bind_arena_binding(arena, *expected)?,
                            )?;
                        }
                        actual
                            .iter()
                            .copied()
                            .map(DecommitColumnSource::ResidentEvaluation)
                            .collect()
                    }
                };
                Ok(TraceSourceGroup { columns })
            })
            .collect::<Result<Vec<_>, ResidentRuntimeError>>()?;
        let expected_layers = planned
            .retained_layers_bottom_up
            .iter()
            .map(|&binding| bind_arena_binding(arena, binding))
            .collect::<Result<Vec<_>, _>>()?;
        let (root, retained_layers_bottom_up) = if planned.id == CommitmentTreeId::Preprocessed {
            (fixed_preprocessed_root, fixed_preprocessed_layers.to_vec())
        } else {
            let graph = prepared_commitment
                .ok_or(ResidentRuntimeError::MissingPreparedCommitment(planned.id))?;
            (
                graph.root_slice(),
                graph.retained_layers_bottom_up().to_vec(),
            )
        };
        if retained_layers_bottom_up.len() != expected_layers.len() {
            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                "trace retained-layer count differs from the arena plan",
            ));
        }
        for (actual, expected) in retained_layers_bottom_up
            .iter()
            .copied()
            .zip(expected_layers)
        {
            require_same_slice(
                "trace retained-layer binding differs from prepared commitment",
                actual,
                expected,
            )?;
        }
        let Some(last_layer) = retained_layers_bottom_up.last().copied() else {
            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                "trace commitment has no retained root",
            ));
        };
        require_same_slice(
            "trace retained root differs from prepared commitment root",
            last_layer,
            root,
        )?;
        sources.push(DecommitTreeSources::Trace(TraceDecommitSources {
            groups,
            retained_layers_bottom_up,
        }));
    }

    let trace_count = workspace.plan().commitments().len();
    if fri.tree_count() + trace_count != planned_decommit.config.trees.len()
        || fri.tree_count() != workspace.plan().fri().requirements.trees.len()
    {
        return Err(ResidentRuntimeError::DecommitTopologyMismatch(
            "FRI tree count differs from decommitment",
        ));
    }
    for fri_tree_index in 0..fri.tree_count() {
        let Some(DecommitTreeGeometry::Fri(geometry)) = planned_decommit
            .config
            .trees
            .get(trace_count + fri_tree_index)
        else {
            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                "FRI decommit suffix order differs",
            ));
        };
        if geometry.fri_tree_index as usize != fri_tree_index {
            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                "FRI decommit ordinal differs",
            ));
        }
        let evaluation = fri.tree_evaluation(fri_tree_index)?;
        if evaluation.log_size != geometry.evaluation_log_size {
            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                "FRI evaluation height differs from decommitment",
            ));
        }
        let retained_layers_bottom_up = fri.tree_layers_bottom_up(fri_tree_index)?.to_vec();
        let planned_layers = &workspace.plan().fri().slots.trees[fri_tree_index].layers_bottom_up;
        let planned_layer_words =
            &workspace.plan().fri().requirements.trees[fri_tree_index].layers_bottom_up;
        if retained_layers_bottom_up.len() != planned_layers.len()
            || retained_layers_bottom_up.len() != planned_layer_words.len()
        {
            return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                "FRI retained-layer count differs from the arena plan",
            ));
        }
        for ((&actual, &planned_slot), layer) in retained_layers_bottom_up
            .iter()
            .zip(planned_layers)
            .zip(planned_layer_words)
        {
            // The physical slot may be pooled larger than the logical layer;
            // compare against the plan's logical extent, mirroring the
            // truncated slice the commit graph's binder returned.
            let planned = arena.bind(planned_slot)?;
            if planned.len_words() < layer.words {
                return Err(ResidentRuntimeError::DecommitTopologyMismatch(
                    "FRI retained-layer slot is smaller than its planned extent",
                ));
            }
            require_same_slice(
                "FRI retained-layer binding differs from its commit graph",
                actual,
                planned.truncated(layer.words),
            )?;
        }
        require_same_slice(
            "FRI retained root differs from its commit root",
            *retained_layers_bottom_up.last().ok_or(
                ResidentRuntimeError::DecommitTopologyMismatch(
                    "FRI commitment has no retained root",
                ),
            )?,
            fri.tree_root(fri_tree_index)?,
        )?;
        sources.push(DecommitTreeSources::Fri(FriDecommitOwnedSources {
            evaluation: evaluation.values,
            coordinate_stride: evaluation.coordinate_stride,
            retained_layers_bottom_up,
        }));
    }
    Ok(sources)
}

fn require_same_slice(
    role: &'static str,
    actual: ArenaSlice,
    expected: ArenaSlice,
) -> Result<(), ResidentRuntimeError> {
    if actual.id() != expected.id()
        || actual.as_u32_ptr() != expected.as_u32_ptr()
        || actual.len_words() != expected.len_words()
    {
        return Err(ResidentRuntimeError::DecommitTopologyMismatch(role));
    }
    Ok(())
}

fn arena_relation_sources(
    workspace: &GraphWorkspace,
) -> Result<Vec<RelationInstanceSources>, ResidentRuntimeError> {
    let relation = workspace.plan().relation();
    let mut ordered = Vec::with_capacity(relation.source_plan.len());
    for source_plan in &relation.source_plan {
        let purpose = match source_plan.plane {
            RelationSourcePlane::LookupWords => BufferPurpose::LookupInputs,
            RelationSourcePlane::BaseTrace => BufferPurpose::BaseTrace,
        };
        let columns = (0..source_plan.column_count)
            .map(|ordinal| {
                let (_, binding) = workspace
                    .plan()
                    .find(
                        Some(source_plan.batch.component),
                        Some(source_plan.part),
                        purpose,
                        ordinal,
                    )
                    .ok_or(ResidentRuntimeError::MissingRelationSource {
                        batch: source_plan.batch,
                        instance_index: source_plan.instance_index,
                        ordinal,
                    })?;
                let source = bind_arena_binding(workspace.arena(), binding)?;
                require_resident_source(workspace, source)?;
                Ok(source)
            })
            .collect::<Result<Vec<_>, ResidentRuntimeError>>()?;
        ordered.push(RelationInstanceSources { columns });
    }
    Ok(ordered)
}

fn transcript_bindings(
    workspace: &GraphWorkspace,
) -> Result<
    (
        Vec<(TranscriptInputId, ArenaSlice)>,
        Vec<(TranscriptOutputId, ArenaSlice)>,
    ),
    ResidentRuntimeError,
> {
    let planned = workspace.plan().transcript();
    let inputs = planned
        .inputs
        .iter()
        .map(|&(id, binding)| Ok((id, bind_arena_binding(workspace.arena(), binding)?)))
        .collect::<Result<Vec<_>, ResidentRuntimeError>>()?;
    let outputs = planned
        .outputs
        .iter()
        .map(|&(id, binding)| Ok((id, bind_arena_binding(workspace.arena(), binding)?)))
        .collect::<Result<Vec<_>, ResidentRuntimeError>>()?;
    Ok((inputs, outputs))
}

fn require_resident_source(
    workspace: &GraphWorkspace,
    source: ArenaSlice,
) -> Result<(), ResidentRuntimeError> {
    let arena_start = workspace.arena().base_ptr().as_ptr() as usize;
    let arena_end = workspace
        .plan()
        .total_words()
        .checked_mul(core::mem::size_of::<u32>())
        .and_then(|bytes| arena_start.checked_add(bytes));
    let source_start = source.as_u32_ptr() as usize;
    let source_end = source_start.checked_add(source.len_bytes());
    let (Some(arena_end), Some(source_end)) = (arena_end, source_end) else {
        return Err(ResidentRuntimeError::SourceOutsideArena { slot: source.id() });
    };
    if source_start < arena_start || source_end > arena_end {
        return Err(ResidentRuntimeError::SourceOutsideArena { slot: source.id() });
    }
    Ok(())
}

fn fri_round_segment(round_index: usize) -> Result<GraphSegment, ResidentRuntimeError> {
    let layer = round_index
        .checked_add(1)
        .and_then(|value| u8::try_from(value).ok())
        .ok_or(ResidentRuntimeError::FriRoundIndexTooLarge(round_index))?;
    Ok(GraphSegment::FriLayer(layer))
}

fn require_complete_captured_topology(
    fri_rounds: usize,
    actual: usize,
    transcript_segments: usize,
) -> Result<usize, ResidentRuntimeError> {
    let expected = fri_rounds
        .checked_add(6)
        .ok_or(ResidentRuntimeError::SizeOverflow)?;
    if actual != expected
        || actual
            .checked_add(1)
            .ok_or(ResidentRuntimeError::SizeOverflow)?
            != transcript_segments
    {
        return Err(ResidentRuntimeError::CapturedGraphTopology {
            fri_rounds,
            transcript_segments,
            expected,
            actual,
        });
    }
    Ok(actual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_bundle_budget_uses_the_captured_kernel_node_count() {
        let budget = ResidentHotPathBudget::final_bundle(29, 123, 371_604);
        assert_eq!(budget.expected_graph_launches, 29);
        assert_eq!(budget.expected_kernel_launches, Some(123));
        assert_eq!(budget.expected_d2h_bytes, 371_604);
    }

    #[test]
    fn captured_topology_is_six_fixed_graphs_plus_fri_rounds() {
        assert_eq!(require_complete_captured_topology(8, 14, 15).unwrap(), 14);
        assert_eq!(require_complete_captured_topology(23, 29, 30).unwrap(), 29);
        assert!(matches!(
            require_complete_captured_topology(8, 13, 15),
            Err(ResidentRuntimeError::CapturedGraphTopology {
                fri_rounds: 8,
                transcript_segments: 15,
                expected: 14,
                actual: 13,
            })
        ));
        assert!(matches!(
            require_complete_captured_topology(8, 14, 14),
            Err(ResidentRuntimeError::CapturedGraphTopology {
                fri_rounds: 8,
                transcript_segments: 14,
                expected: 14,
                actual: 14,
            })
        ));
    }

    #[test]
    fn final_bundle_budget_requires_exact_kernel_nodes_and_zero_frees() {
        let budget = ResidentHotPathBudget::final_bundle(29, 123, 371_604);
        let exact = CudaExecTelemetry {
            sync_calls: 1,
            d2h_bytes: 371_604,
            graph_launches: 29,
            kernel_launches: 123,
            ..CudaExecTelemetry::default()
        };
        assert!(budget.accepts(exact));
        macro_rules! reject_counter {
            ($field:ident, $value:expr) => {{
                let mut changed = exact;
                changed.$field = $value;
                assert!(!budget.accepts(changed), stringify!($field));
            }};
        }
        reject_counter!(graph_launches, 28);
        reject_counter!(graph_launches, 100);
        reject_counter!(kernel_launches, 0);
        reject_counter!(kernel_launches, 122);
        reject_counter!(kernel_launches, 124);
        reject_counter!(sync_calls, 0);
        reject_counter!(h2d_bytes, 1);
        reject_counter!(d2h_bytes, 371_603);
        reject_counter!(allocations, 1);
        reject_counter!(allocation_bytes, 4);
        reject_counter!(frees, 1);
        reject_counter!(memset_bytes, 4);
        reject_counter!(fill_words, 1);
        reject_counter!(d2d_bytes, 4);
        reject_counter!(capture_begins, 1);
        reject_counter!(capture_finishes, 1);
        reject_counter!(capture_aborts, 1);
        reject_counter!(lane_forks, 1);
        reject_counter!(lane_joins, 1);
        reject_counter!(graph_submit_gap_ns_max, 50_000_000);
    }

    #[test]
    fn witness_level_packing_is_deterministic_and_balances_long_writers() {
        assert_eq!(
            pack_weighted_lane_level(vec![(0, 10, "a"), (1, 8, "b"), (2, 7, "c"), (3, 3, "d")], 2,),
            vec![vec![0, 3], vec![1, 2]],
        );
        assert_eq!(
            pack_weighted_lane_level(vec![(7, 5, "z"), (6, 5, "a")], 2),
            vec![vec![6], vec![7]],
        );
    }

    #[test]
    fn fri_segments_reserve_zero_for_the_original_tree() {
        assert_eq!(fri_round_segment(0).unwrap(), GraphSegment::FriLayer(1));
        assert_eq!(fri_round_segment(254).unwrap(), GraphSegment::FriLayer(255));
        assert!(matches!(
            fri_round_segment(255),
            Err(ResidentRuntimeError::FriRoundIndexTooLarge(255))
        ));
    }

    #[test]
    fn workspace_identity_includes_arena_not_only_protocol_shape() {
        let first = ResidentWorkspaceIdentity {
            shape_key: ProofShapeKey(7),
            protocol_key: 11,
            arena_base: 13,
        };
        assert_ne!(
            first,
            ResidentWorkspaceIdentity {
                arena_base: 17,
                ..first
            }
        );
    }

    #[test]
    fn cached_capture_still_advances_host_transcript_admission() {
        let admissions = std::cell::Cell::new(0);
        admit_on_graph_reuse(GraphCaptureStatus::Captured, || {
            admissions.set(admissions.get() + 1);
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();
        admit_on_graph_reuse(GraphCaptureStatus::Reused, || {
            admissions.set(admissions.get() + 1);
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();
        assert_eq!(admissions.get(), 1);
    }

    #[test]
    fn failed_capture_rolls_back_transcript_cursor_and_can_retry() {
        use stwo_backend_cuda::{
            Blake2sTranscriptSchedule, TranscriptBoundaryId, TranscriptInputId,
            TranscriptOperation, TranscriptStart,
        };

        let schedule = Blake2sTranscriptSchedule::new(
            TranscriptStart::Default,
            vec![TranscriptOperation::MixU32s {
                boundary: TranscriptBoundaryId(1),
                source: TranscriptInputId(1),
                n_words: 1,
            }],
            8,
        )
        .unwrap();
        let mut cursor = TranscriptSegmentCursor::new(&schedule);
        let first_generation = next_capture_generation(&cursor).unwrap();
        assert_eq!(first_generation, 1);
        cursor.begin_generation(first_generation).unwrap();
        let checkpoint = cursor.clone();

        let failed = capture_with_cursor_rollback(&mut cursor, |cursor| {
            cursor
                .admit_segment(&schedule, 1, 0..1, TranscriptSegmentStart::Initialize)
                .unwrap();
            Err::<(), _>("injected post-transcript capture failure")
        });
        assert_eq!(failed, Err("injected post-transcript capture failure"));
        assert_eq!(cursor, checkpoint);

        let retry_generation = next_capture_generation(&cursor).unwrap();
        assert_eq!(retry_generation, 2);
        cursor.begin_generation(retry_generation).unwrap();
        capture_with_cursor_rollback(&mut cursor, |cursor| {
            cursor
                .admit_segment(
                    &schedule,
                    retry_generation,
                    0..1,
                    TranscriptSegmentStart::Initialize,
                )
                .map_err(|_| "retry admission failed")
        })
        .unwrap();
        assert!(cursor.is_complete());
    }
}
