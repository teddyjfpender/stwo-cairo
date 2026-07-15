//! Allocation-free resident Cairo composition graph.
//!
//! Preparation resolves every generated constraint kernel against the embedded
//! AOT pack and uploads only immutable pointer/log/denominator descriptors.
//! Replay evaluates each component's exact trace slice from resident
//! coefficients, accumulates with proof-global descending random powers, lifts
//! the per-log accumulators, interpolates the four secure coordinates, and
//! writes the eight split composition coefficient columns. No replay operation
//! allocates, uploads, downloads, synchronizes, or uses CUDA's default stream.

use core::ffi::c_void;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;

use stwo::core::fields::m31::M31;
use stwo_backend_cuda::{aot, ArenaError, ArenaSlice, ArenaSlotId, CudaRuntimeError, DeviceArena};
use stwo_backend_cuda_kernels::raw::{self, CudaSecureField};

use crate::arena_plan::{CommitmentTreeId, OpenedColumnSource};
use crate::composition_plan::{
    CompositionComponentPlan, CompositionExtParamSource, CompositionKernelPart, CompositionPlan,
    CompositionProofBindings, CompositionWaveKernelPlan,
};
use crate::direct_composition_retention::{
    direct_composition_plan_key, DirectCompositionRetentionPlan,
};

mod binding_refresh;
pub use binding_refresh::CompositionBindingRefreshTelemetry;

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const SECURE_WORDS: usize = 4;
const SECURE_COORDINATES: usize = 4;
const SPLIT_COORDINATES: usize = 8;
const TRACE_TREES: usize = 3;
const POINTER_WORDS: usize = core::mem::size_of::<usize>().div_ceil(WORD_BYTES);
const WAVE_PART_WORDS: usize =
    core::mem::size_of::<raw::CudaCompositionWavePart>().div_ceil(WORD_BYTES);

const _: () = assert!(WAVE_PART_WORDS == 12);

pub const COMPOSITION_POINTER_ALIGNMENT_WORDS: usize = core::mem::align_of::<usize>() / WORD_BYTES;

/// Static membership bound for the wide launch mode: a component is "small"
/// iff `evaluation_log_size <= 18` (at most 2^18 evaluation rows).
///
/// Rationale (documented per the Step-3.3 plan): the generated constraint
/// kernels launch 128-thread blocks, so a 2^18-row component is at most 2048
/// blocks — about one scheduling wave on an H100 (132 SMs x ~16 resident
/// 128-thread blocks ~= 2112 block slots). Such a kernel can never fill the
/// card alone and only benefits from co-scheduling with its peers; anything
/// larger runs multiple waves and saturates the card by itself, so it keeps
/// its dedicated serial launch exactly as today.
pub const COMPOSITION_WIDE_SMALL_MAX_EVALUATION_LOG: u32 = 18;

/// Selects the stream topology [`PreparedCompositionGraph::launch`] enqueues.
///
/// Byte identity: both modes launch the exact same kernels with the exact same
/// inputs. `Wide` changes only (a) which stream each small component's
/// LDE+eval pair is recorded on and (b) each small group's private LDE-tile
/// region. Every accumulator element still receives its contributions from
/// the same components in the same plan order (an accumulator is keyed by
/// `evaluation_log_size`, membership is decided by `evaluation_log_size`, and
/// one group is never split across lanes), so the per-element read-modify-
/// write sequence — and therefore every output byte — is unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionLaunchMode {
    /// Every component's LDE + eval kernels enqueue in plan order on the main
    /// stream through one shared LDE tile (today's behavior; default).
    Serial,
    /// Small components (see [`COMPOSITION_WIDE_SMALL_MAX_EVALUATION_LOG`])
    /// are grouped by `evaluation_log_size`, given private LDE-tile regions,
    /// and fanned across the context's component lanes via fork/join so their
    /// kernels co-schedule inside the captured graph. Large components keep
    /// their dedicated main-stream launches, overlapping the small lanes.
    /// Opt-in via `STWO_CUDA_COMPOSITION_WIDE=1`.
    Wide,
    /// One exact generated kernel owns every contribution to each evaluation-
    /// log accumulator. ReplacementV1 selects this only after all source
    /// columns are retained at their consumer log, so no fallback LDE or
    /// per-component accumulator writer may coexist with a wave.
    Wave,
}

/// `STWO_CUDA_COMPOSITION_WIDE=1` opts [`PreparedCompositionGraph::prepare`]
/// and the arena plan into the wide topology. Read once per process
/// (`OnceLock`) so plan sizing, eager launch, capture, and replay all observe
/// one mode. Default OFF: the serial topology stays the proven path.
pub fn default_composition_launch_mode() -> CompositionLaunchMode {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| crate::flags::flag_on("STWO_CUDA_COMPOSITION_WIDE")) {
        CompositionLaunchMode::Wide
    } else {
        CompositionLaunchMode::Serial
    }
}

/// One wide-mode scheduling unit: every small component sharing one
/// `evaluation_log_size` (and therefore one accumulator). Members are in plan
/// order — the accumulator's exact serial-mode read-modify-write order — and a
/// group is always assigned to a single lane, so the shared-accumulator
/// updates can never race and never reorder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionWideGroup {
    pub evaluation_log_size: u32,
    /// Component indices into [`CompositionWorkspaceRequirements::components`],
    /// in plan order.
    pub members: Vec<usize>,
    /// Deterministic packing weight: approximate words touched by the group
    /// (rows x (columns + accumulator coordinates) per member).
    pub weight_words: u64,
    /// Private LDE-tile region shared by the group's members (they serialize
    /// in-lane, exactly like today's global tile reuse).
    pub tile_offset_words: usize,
    pub tile_len_words: usize,
}

/// Deterministically pack wide groups onto `lane_count` component lanes:
/// heaviest group first onto the least-loaded lane, ties broken by
/// `evaluation_log_size` then lane index (the same greedy shape as the
/// witness segment's `pack_weighted_lane_level`). Returns `lanes[lane] ->
/// group indices` in assignment order.
pub fn pack_composition_wide_groups(
    groups: &[CompositionWideGroup],
    lane_count: usize,
) -> Vec<Vec<usize>> {
    let mut order = (0..groups.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        groups[right]
            .weight_words
            .cmp(&groups[left].weight_words)
            .then_with(|| {
                groups[left]
                    .evaluation_log_size
                    .cmp(&groups[right].evaluation_log_size)
            })
    });
    let mut lanes = vec![Vec::new(); lane_count];
    let mut loads = vec![0u64; lane_count];
    for group in order {
        let lane = loads
            .iter()
            .enumerate()
            .min_by_key(|&(lane, &load)| (load, lane))
            .map(|(lane, _)| lane)
            .expect("lane count checked nonzero");
        loads[lane] = loads[lane].saturating_add(groups[group].weight_words);
        lanes[lane].push(group);
    }
    lanes
}

/// One resident coefficient column. Logical slot identities keep the topology
/// address-free; [`PreparedCompositionGraph::prepare`] performs every binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionCoefficientSource {
    pub slot: ArenaSlotId,
    pub log_size: u32,
}

/// Global preprocessed/base/interaction coefficient trees in commitment order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionTraceTopology {
    pub trees: Vec<Vec<CompositionCoefficientSource>>,
}

/// A caller-populated device range containing QM31 values in the exact slot
/// order exposed by [`CompositionComponentPlan::ext_param_values`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionExtParamBinding {
    pub slot: ArenaSlotId,
    pub offset_words: usize,
}

/// Dynamic proof inputs. Values are produced by transcript/claim device stages;
/// this graph never materializes their host-side oracle values.
#[derive(Clone, Debug)]
pub struct CompositionDeviceInputs {
    pub random_coefficient: ArenaSlotId,
    /// Full logical twiddle-tree slices. Small-domain NTTs index relative to
    /// the END of these trees, so callers must preserve the planned logical
    /// length rather than truncate to a consumer's minimum requirement.
    pub forward_twiddles: ArenaSlice,
    pub inverse_twiddles: ArenaSlice,
    /// Stable challenge slices produced by `PreparedRelationGraph`. These are
    /// logically-truncated slices, not slot ids: the alpha-power count is
    /// derived from `relation_alpha_powers.len_words()`, which must be the
    /// logical challenge extent and never a pooled physical slot length.
    pub relation_z: ArenaSlice,
    pub relation_alpha_powers: ArenaSlice,
    /// One entry per Cairo component, in composition-plan order. A claimed-sum
    /// source is present exactly when that component has a
    /// [`CompositionExtParamSource::ClaimedSumScaled`] slot.
    pub claimed_sums: Vec<Option<ArenaSlotId>>,
    /// One entry per component. `None` is valid only for a parameter-free
    /// program; non-empty parameter tables must be supplied by the caller.
    pub ext_params: Vec<Option<CompositionExtParamBinding>>,
}

/// Caller-selected arena slots for all mutable graph workspace and final
/// composition outputs. Outputs are `[left coordinates 0..4, right 0..4]`, the
/// exact order passed to STWO's composition commitment tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionWorkspaceSlots {
    pub descriptors: ArenaSlotId,
    pub lde_tile: ArenaSlotId,
    pub accumulators: ArenaSlotId,
    pub random_coefficient_powers: ArenaSlotId,
    pub composition_coefficients: [ArenaSlotId; SPLIT_COORDINATES],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionArenaSlotRequirement {
    pub id: ArenaSlotId,
    pub len_words: usize,
    pub alignment_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionSourceRef {
    pub tree: usize,
    pub column: usize,
    pub source: CompositionCoefficientSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionSourceRetention {
    pub consumer: usize,
    pub plan_column: usize,
    pub source: OpenedColumnSource,
    pub tree: CommitmentTreeId,
    pub proof_column: usize,
    pub native_evaluation_log_size: u32,
    pub direct: bool,
    pub fallback_ordinal: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct CompositionDirectEvaluationBinding {
    pub plan_column: usize,
    pub evaluation: ArenaSlice,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionComponentRequirements {
    pub component: &'static str,
    pub instance: usize,
    pub trace_log_size: u32,
    pub evaluation_log_size: u32,
    pub row_count: usize,
    pub sources: Vec<CompositionSourceRef>,
    /// Empty on the historical all-fallback path. When a retention plan is
    /// supplied this is one-for-one with `sources`, in exact evaluator order.
    pub source_retention: Vec<CompositionSourceRetention>,
    pub fallback_count: usize,
    pub interaction_offsets: [u32; TRACE_TREES],
    pub denominator_words: usize,
    pub base_param_words: usize,
    pub ext_param_words: usize,
    pub random_coefficient_offset: usize,
    pub accumulator_offset_words: usize,
    /// Word offset of this component's evaluation region inside the shared
    /// LDE tile. Always zero in [`CompositionLaunchMode::Serial`]; in `Wide`,
    /// each small group gets a private disjoint region so concurrent lanes
    /// never alias, while large (serial) components keep offset zero.
    pub lde_tile_offset_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionAccumulatorRequirements {
    pub log_size: u32,
    pub offset_words: usize,
    pub len_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionWavePartRequirement {
    pub component: usize,
    pub kernel: usize,
    pub identity: aot::CompositionWaveKernelPartIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionWaveRequirements {
    pub evaluation_log_size: u32,
    pub row_count: usize,
    pub accumulator_offset_words: usize,
    pub descriptor_offset_words: usize,
    pub parts: Vec<CompositionWavePartRequirement>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ComponentDescriptorLayout {
    coefficient_pointers: usize,
    coefficient_sizes: usize,
    evaluation_pointers: usize,
    fallback_coefficient_pointers: Option<usize>,
    fallback_coefficient_sizes: Option<usize>,
    fallback_evaluation_pointers: Option<usize>,
    interaction_offsets: usize,
    denominator_inverses: usize,
    base_params: usize,
}

/// Pure shape/sizing result. It contains logical source/slot identities but no
/// process-local device addresses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionWorkspaceRequirements {
    pub total_constraints: usize,
    pub max_evaluation_log_size: u32,
    pub descriptor_words: usize,
    pub lde_tile_words: usize,
    pub accumulator_words: usize,
    pub random_power_words: usize,
    pub output_coefficient_words: usize,
    pub forward_twiddle_words: usize,
    pub inverse_twiddle_words: usize,
    pub dynamic_ext_param_count: usize,
    pub claimed_sum_count: usize,
    /// `None` preserves the historical descriptor bytes and launch topology.
    pub direct_retention_plan_key: Option<u64>,
    pub direct_retention_bitmap: Vec<u64>,
    /// The launch mode these requirements were computed for. `Wide` changes
    /// only `lde_tile_words`, per-component `lde_tile_offset_words`, and the
    /// scheduling metadata below; everything else is mode-independent.
    pub mode: CompositionLaunchMode,
    /// Wide-mode scheduling units (empty in `Serial` mode).
    pub wide_groups: Vec<CompositionWideGroup>,
    /// Components enqueued serially on the main stream, in plan order. In
    /// `Serial` mode this is every component; in `Wide` mode, the large ones.
    pub serial_components: Vec<usize>,
    /// Exact sole-owner waves. Nonempty only in [`CompositionLaunchMode::Wave`].
    pub waves: Vec<CompositionWaveRequirements>,
    pub components: Vec<CompositionComponentRequirements>,
    pub accumulators: Vec<CompositionAccumulatorRequirements>,
    zero_words: usize,
    final_coordinate_pointers: usize,
    dynamic_destination_pointers: usize,
    dynamic_source_kinds: usize,
    dynamic_source_indices: usize,
    dynamic_scales: usize,
    claimed_sum_pointers: usize,
    component_descriptors: Vec<ComponentDescriptorLayout>,
}

impl CompositionWorkspaceRequirements {
    pub fn arena_slot_requirements(
        &self,
        slots: &CompositionWorkspaceSlots,
    ) -> Result<Vec<CompositionArenaSlotRequirement>, PreparedCompositionError> {
        let mut requirements = vec![
            slot_requirement(
                slots.descriptors,
                self.descriptor_words,
                COMPOSITION_POINTER_ALIGNMENT_WORDS,
            ),
            slot_requirement(slots.lde_tile, self.lde_tile_words, 1),
            slot_requirement(slots.accumulators, self.accumulator_words, 1),
            slot_requirement(
                slots.random_coefficient_powers,
                self.random_power_words,
                SECURE_WORDS,
            ),
        ];
        requirements.extend(
            slots
                .composition_coefficients
                .map(|id| slot_requirement(id, self.output_coefficient_words, 1)),
        );
        let mut ids = BTreeSet::new();
        for requirement in &requirements {
            if !ids.insert(requirement.id) {
                return Err(PreparedCompositionError::DuplicateSlot(requirement.id));
            }
        }
        Ok(requirements)
    }
}

fn slot_requirement(
    id: ArenaSlotId,
    len_words: usize,
    alignment_words: usize,
) -> CompositionArenaSlotRequirement {
    CompositionArenaSlotRequirement {
        id,
        len_words,
        alignment_words,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedCompositionError {
    EmptyPlan,
    TraceTreeCount {
        expected: usize,
        actual: usize,
    },
    DuplicateSlot(ArenaSlotId),
    InvalidPlanTotal {
        declared: usize,
        actual: usize,
    },
    InvalidPlanMaxLog {
        declared: u32,
        actual: u32,
    },
    InvalidLogSize {
        component: usize,
        trace_log_size: u32,
        evaluation_log_size: u32,
    },
    SourceCannotFitEvaluationDomain {
        component: usize,
        tree: usize,
        column: usize,
        source_log_size: u32,
        evaluation_log_size: u32,
    },
    MissingTraceTree {
        component: usize,
        tree: usize,
    },
    DuplicateTraceTree {
        component: usize,
        tree: usize,
    },
    UnsupportedTraceTree {
        component: usize,
        tree: usize,
    },
    InvalidTraceRange {
        component: usize,
        tree: usize,
        start: usize,
        end: usize,
        columns: usize,
    },
    InvalidPreprocessedColumn {
        component: usize,
        column: usize,
        columns: usize,
    },
    EmptyComponentSources(usize),
    EmptyKernelProgram(usize),
    InvalidKernelIdentity {
        component: usize,
        kernel: usize,
    },
    InvalidKernelRandomOffset {
        component: usize,
        kernel: usize,
        rc_base: u32,
        constraints: usize,
    },
    RandomCoefficientOrder {
        component: usize,
        expected: usize,
        actual: usize,
    },
    DenominatorCount {
        component: usize,
        expected: usize,
        actual: usize,
    },
    ExtParamSourceCount {
        component: usize,
        values: usize,
        sources: usize,
    },
    ConstantExtParamMismatch {
        component: usize,
        slot: usize,
    },
    BaseParamBindingCount {
        expected: usize,
        actual: usize,
    },
    BaseParamBindingTotalWords {
        expected: usize,
        actual: usize,
    },
    BaseParamBindingIdentity(usize),
    BaseParamBindingWords {
        component: usize,
        expected: usize,
        actual: usize,
    },
    ExtParamBindingCount {
        expected: usize,
        actual: usize,
    },
    ClaimedSumBindingCount {
        expected: usize,
        actual: usize,
    },
    MissingExtParamBinding {
        component: usize,
        words: usize,
    },
    UnexpectedExtParamBinding(usize),
    DuplicateExtParamBinding(ArenaSlotId),
    MissingClaimedSumBinding(usize),
    UnexpectedClaimedSumBinding(usize),
    MisalignedExtParamBinding {
        component: usize,
        offset_words: usize,
    },
    SlotTooSmall {
        slot: ArenaSlotId,
        required_words: usize,
        actual_words: usize,
    },
    ContextMismatch(ArenaSlotId),
    SourceAliasesWritableWorkspace(ArenaSlotId),
    InputAliasesWritableWorkspace(ArenaSlotId),
    ForwardInverseTwiddlesAlias(ArenaSlotId),
    RelationChallengeSourcesAlias(ArenaSlotId),
    DirectRetentionBindingCount {
        expected: usize,
        actual: usize,
    },
    DirectRetentionPlanKeyDrift,
    DirectRetentionPlanDrift(&'static str),
    DirectEvaluationBindingCount {
        expected: usize,
        actual: usize,
    },
    MissingDirectEvaluation(usize),
    UnexpectedDirectEvaluation(usize),
    DuplicateDirectEvaluationColumn(usize),
    InconsistentDuplicateDirectEvaluation {
        first: usize,
        second: usize,
    },
    DirectEvaluationAliasesWritableWorkspace(ArenaSlotId),
    DirectEvaluationAliasesCoefficient(ArenaSlotId),
    DirectEvaluationAliasesUnrelatedInput(ArenaSlotId),
    AlphaPowerOutOfRange {
        component: usize,
        power: u32,
        available: usize,
    },
    AotPackUnavailable,
    AotKernelMiss {
        component: usize,
        kernel: usize,
        cache_key: u64,
    },
    KernelNameContainsNul {
        component: usize,
        kernel: usize,
    },
    KernelSourceContainsNul {
        component: usize,
        kernel: usize,
    },
    KernelLaunchMiss {
        component: usize,
        kernel: usize,
        cache_key: u64,
    },
    CompositionWaveRequiresAllDirect {
        component: usize,
        fallback_count: usize,
    },
    CompositionWavePlanDrift(&'static str),
    CompositionWaveAotMiss {
        wave: usize,
        cache_key: u64,
    },
    CompositionWaveNameContainsNul(usize),
    CompositionWaveLaunchMiss {
        wave: usize,
        cache_key: u64,
    },
    /// Wide mode requires at least one component lane on the execution
    /// context; fail closed rather than silently degrading the topology.
    NoComponentLanes,
    CudaStatus {
        operation: &'static str,
        status: i32,
    },
    SizeOverflow,
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for PreparedCompositionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid prepared CUDA composition graph: {self:?}")
    }
}

impl std::error::Error for PreparedCompositionError {}

impl From<ArenaError> for PreparedCompositionError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for PreparedCompositionError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

/// Compute exact workspace geometry and source selection without touching
/// CUDA, in the process-default launch mode (see
/// [`default_composition_launch_mode`]). The arena plan and
/// [`PreparedCompositionGraph::prepare`] both call this, so their geometry
/// always agrees within a process.
pub fn composition_workspace_requirements(
    plan: &CompositionPlan,
    trace: &CompositionTraceTopology,
) -> Result<CompositionWorkspaceRequirements, PreparedCompositionError> {
    composition_workspace_requirements_with_mode(plan, trace, default_composition_launch_mode())
}

/// Mode-explicit form of [`composition_workspace_requirements`]. Parity tests
/// use this to exercise both modes in one invocation; production callers go
/// through the env-default wrapper.
pub fn composition_workspace_requirements_with_mode(
    plan: &CompositionPlan,
    trace: &CompositionTraceTopology,
    mode: CompositionLaunchMode,
) -> Result<CompositionWorkspaceRequirements, PreparedCompositionError> {
    composition_workspace_requirements_with_retention(plan, trace, mode, None)
}

pub(crate) fn composition_workspace_requirements_with_retention(
    plan: &CompositionPlan,
    trace: &CompositionTraceTopology,
    mode: CompositionLaunchMode,
    direct_retention: Option<&DirectCompositionRetentionPlan>,
) -> Result<CompositionWorkspaceRequirements, PreparedCompositionError> {
    if plan.components.is_empty() || plan.total_constraints == 0 {
        return Err(PreparedCompositionError::EmptyPlan);
    }
    if trace.trees.len() != TRACE_TREES {
        return Err(PreparedCompositionError::TraceTreeCount {
            expected: TRACE_TREES,
            actual: trace.trees.len(),
        });
    }

    let actual_constraints = plan.components.iter().try_fold(0usize, |sum, component| {
        sum.checked_add(component.n_constraints)
    });
    let actual_constraints = actual_constraints.ok_or(PreparedCompositionError::SizeOverflow)?;
    if actual_constraints != plan.total_constraints {
        return Err(PreparedCompositionError::InvalidPlanTotal {
            declared: plan.total_constraints,
            actual: actual_constraints,
        });
    }
    let actual_max_log = plan
        .components
        .iter()
        .map(|component| component.evaluation_log_size)
        .max()
        .ok_or(PreparedCompositionError::EmptyPlan)?;
    if actual_max_log != plan.max_evaluation_log_size {
        return Err(PreparedCompositionError::InvalidPlanMaxLog {
            declared: plan.max_evaluation_log_size,
            actual: actual_max_log,
        });
    }
    let _ = u32::try_from(plan.total_constraints)
        .map_err(|_| PreparedCompositionError::SizeOverflow)?;

    let accumulator_logs = plan
        .components
        .iter()
        .map(|component| component.evaluation_log_size)
        .collect::<BTreeSet<_>>();
    let mut accumulator_words = 0usize;
    let mut accumulators = Vec::with_capacity(accumulator_logs.len());
    let mut accumulator_offsets = BTreeMap::new();
    for log_size in accumulator_logs {
        let len_words = pow2(log_size)?
            .checked_mul(SECURE_COORDINATES)
            .ok_or(PreparedCompositionError::SizeOverflow)?;
        accumulator_offsets.insert(log_size, accumulator_words);
        accumulators.push(CompositionAccumulatorRequirements {
            log_size,
            offset_words: accumulator_words,
            len_words,
        });
        accumulator_words = accumulator_words
            .checked_add(len_words)
            .ok_or(PreparedCompositionError::SizeOverflow)?;
    }

    let mut expected_random_offset = 0usize;
    let mut components = Vec::with_capacity(plan.components.len());
    for (component_index, component) in plan.components.iter().enumerate() {
        validate_component_program(component_index, component, expected_random_offset)?;
        expected_random_offset = expected_random_offset
            .checked_add(component.n_constraints)
            .ok_or(PreparedCompositionError::SizeOverflow)?;
        let sources = select_component_sources(component_index, component, trace)?;
        if sources.is_empty() {
            return Err(PreparedCompositionError::EmptyComponentSources(
                component_index,
            ));
        }
        for source in &sources {
            // The prepared LDE primitive stages at most half the target circle
            // domain, exactly matching a polynomial whose degree bound is
            // strictly below that target domain.
            if source.source.log_size >= component.evaluation_log_size {
                return Err(PreparedCompositionError::SourceCannotFitEvaluationDomain {
                    component: component_index,
                    tree: source.tree,
                    column: source.column,
                    source_log_size: source.source.log_size,
                    evaluation_log_size: component.evaluation_log_size,
                });
            }
        }
        let row_count = pow2(component.evaluation_log_size)?;
        let expected_denominators = pow2(component.evaluation_log_size - component.trace_log_size)?;
        if component.denominator_inverses.len() != expected_denominators {
            return Err(PreparedCompositionError::DenominatorCount {
                component: component_index,
                expected: expected_denominators,
                actual: component.denominator_inverses.len(),
            });
        }
        let interaction_offsets = interaction_offsets(&sources)?;
        components.push(CompositionComponentRequirements {
            component: component.component,
            instance: component.instance,
            trace_log_size: component.trace_log_size,
            evaluation_log_size: component.evaluation_log_size,
            row_count,
            sources,
            source_retention: Vec::new(),
            fallback_count: 0,
            interaction_offsets,
            denominator_words: expected_denominators,
            base_param_words: component.base_param_values.len(),
            ext_param_words: component
                .ext_param_values
                .len()
                .checked_mul(SECURE_WORDS)
                .ok_or(PreparedCompositionError::SizeOverflow)?,
            random_coefficient_offset: component.random_coefficient_offset,
            accumulator_offset_words: *accumulator_offsets
                .get(&component.evaluation_log_size)
                .expect("evaluation log was collected"),
            lde_tile_offset_words: 0,
        });
    }
    let (direct_retention_plan_key, direct_retention_bitmap) =
        apply_direct_retention(&mut components, direct_retention)?;
    let mut waves = composition_wave_requirements(plan, &components, mode, &accumulators)?;
    debug_assert_eq!(expected_random_offset, plan.total_constraints);
    let (mut lde_tile_words, wide_groups, serial_components) =
        lde_tile_layout(mode, &mut components)?;
    // The arena cannot materialize or bind a zero-length logical slot. Keep
    // one inert physical word in the all-direct case; semantic work remains
    // exactly `fallback_count == 0`, so no LDE kernel is enqueued.
    if direct_retention_plan_key.is_some() {
        lde_tile_words = lde_tile_words.max(1);
    }

    let mut descriptor = DescriptorAllocator::default();
    let zero_words = descriptor.take(SECURE_WORDS, SECURE_WORDS)?;
    let final_coordinate_pointers = descriptor.take(
        SECURE_COORDINATES
            .checked_mul(POINTER_WORDS)
            .ok_or(PreparedCompositionError::SizeOverflow)?,
        COMPOSITION_POINTER_ALIGNMENT_WORDS,
    )?;
    let dynamic_ext_param_count = plan
        .components
        .iter()
        .flat_map(|component| &component.ext_param_sources)
        .filter(|source| !matches!(source, CompositionExtParamSource::Constant(_)))
        .count();
    let claimed_sum_count = plan
        .components
        .iter()
        .flat_map(|component| &component.ext_param_sources)
        .filter(|source| matches!(source, CompositionExtParamSource::ClaimedSumScaled))
        .count();
    let dynamic_pointer_words = dynamic_ext_param_count
        .checked_mul(POINTER_WORDS)
        .ok_or(PreparedCompositionError::SizeOverflow)?;
    let claimed_pointer_words = claimed_sum_count
        .checked_mul(POINTER_WORDS)
        .ok_or(PreparedCompositionError::SizeOverflow)?;
    let dynamic_destination_pointers =
        descriptor.take(dynamic_pointer_words, COMPOSITION_POINTER_ALIGNMENT_WORDS)?;
    let dynamic_source_kinds = descriptor.take(dynamic_ext_param_count, 1)?;
    let dynamic_source_indices = descriptor.take(dynamic_ext_param_count, 1)?;
    let dynamic_scales = descriptor.take(dynamic_ext_param_count, 1)?;
    let claimed_sum_pointers =
        descriptor.take(claimed_pointer_words, COMPOSITION_POINTER_ALIGNMENT_WORDS)?;
    let mut component_descriptors = Vec::with_capacity(components.len());
    for component in &components {
        let pointer_words = component
            .sources
            .len()
            .checked_mul(POINTER_WORDS)
            .ok_or(PreparedCompositionError::SizeOverflow)?;
        let base_params = if component.base_param_words == 0 {
            zero_words
        } else {
            descriptor.take(component.base_param_words, 1)?
        };
        component_descriptors.push(ComponentDescriptorLayout {
            coefficient_pointers: descriptor
                .take(pointer_words, COMPOSITION_POINTER_ALIGNMENT_WORDS)?,
            coefficient_sizes: descriptor.take(component.sources.len(), 1)?,
            evaluation_pointers: descriptor
                .take(pointer_words, COMPOSITION_POINTER_ALIGNMENT_WORDS)?,
            fallback_coefficient_pointers: direct_retention_plan_key
                .map(|_| {
                    component
                        .fallback_count
                        .checked_mul(POINTER_WORDS)
                        .ok_or(PreparedCompositionError::SizeOverflow)
                })
                .transpose()?
                .map(|words| descriptor.take(words, COMPOSITION_POINTER_ALIGNMENT_WORDS))
                .transpose()?,
            fallback_coefficient_sizes: direct_retention_plan_key
                .map(|_| descriptor.take(component.fallback_count, 1))
                .transpose()?,
            fallback_evaluation_pointers: direct_retention_plan_key
                .map(|_| {
                    component
                        .fallback_count
                        .checked_mul(POINTER_WORDS)
                        .ok_or(PreparedCompositionError::SizeOverflow)
                })
                .transpose()?
                .map(|words| descriptor.take(words, COMPOSITION_POINTER_ALIGNMENT_WORDS))
                .transpose()?,
            interaction_offsets: descriptor.take(TRACE_TREES, 1)?,
            denominator_inverses: descriptor.take(component.denominator_words, 1)?,
            base_params,
        });
    }
    for wave in &mut waves {
        let descriptor_words = wave
            .parts
            .len()
            .checked_mul(WAVE_PART_WORDS)
            .ok_or(PreparedCompositionError::SizeOverflow)?;
        wave.descriptor_offset_words =
            descriptor.take(descriptor_words, COMPOSITION_POINTER_ALIGNMENT_WORDS)?;
    }

    let max_rows = pow2(plan.max_evaluation_log_size)?;
    Ok(CompositionWorkspaceRequirements {
        total_constraints: plan.total_constraints,
        max_evaluation_log_size: plan.max_evaluation_log_size,
        descriptor_words: descriptor.cursor,
        lde_tile_words,
        accumulator_words,
        random_power_words: plan
            .total_constraints
            .checked_mul(SECURE_WORDS)
            .ok_or(PreparedCompositionError::SizeOverflow)?,
        output_coefficient_words: max_rows / 2,
        forward_twiddle_words: max_rows / 2,
        inverse_twiddle_words: max_rows / 2,
        dynamic_ext_param_count,
        claimed_sum_count,
        direct_retention_plan_key,
        direct_retention_bitmap,
        mode,
        wide_groups,
        serial_components,
        waves,
        components,
        accumulators,
        zero_words,
        final_coordinate_pointers,
        dynamic_destination_pointers,
        dynamic_source_kinds,
        dynamic_source_indices,
        dynamic_scales,
        claimed_sum_pointers,
        component_descriptors,
    })
}

#[cfg(feature = "direct-retention-test-api")]
#[doc(hidden)]
pub fn composition_workspace_requirements_with_retention_for_test(
    plan: &CompositionPlan,
    trace: &CompositionTraceTopology,
    mode: CompositionLaunchMode,
    direct_retention: Option<&DirectCompositionRetentionPlan>,
) -> Result<CompositionWorkspaceRequirements, PreparedCompositionError> {
    composition_workspace_requirements_with_retention(plan, trace, mode, direct_retention)
}

fn component_lde_footprint_words(
    component: &CompositionComponentRequirements,
) -> Result<usize, PreparedCompositionError> {
    component
        .row_count
        .checked_mul(component.fallback_count)
        .ok_or(PreparedCompositionError::SizeOverflow)
}

fn apply_direct_retention(
    components: &mut [CompositionComponentRequirements],
    direct_retention: Option<&DirectCompositionRetentionPlan>,
) -> Result<(Option<u64>, Vec<u64>), PreparedCompositionError> {
    let Some(plan) = direct_retention else {
        for component in components {
            component.fallback_count = component.sources.len();
        }
        return Ok((None, Vec::new()));
    };
    if direct_composition_plan_key(plan) != plan.cache_key {
        return Err(PreparedCompositionError::DirectRetentionPlanKeyDrift);
    }
    let expected_bindings = components.iter().try_fold(0usize, |count, component| {
        count
            .checked_add(component.sources.len())
            .ok_or(PreparedCompositionError::SizeOverflow)
    })?;
    if plan.bindings.len() != expected_bindings {
        return Err(PreparedCompositionError::DirectRetentionBindingCount {
            expected: expected_bindings,
            actual: plan.bindings.len(),
        });
    }
    if plan.direct_bitmap.len() != expected_bindings.div_ceil(64) {
        return Err(PreparedCompositionError::DirectRetentionPlanDrift(
            "bitmap word count",
        ));
    }
    let used_tail_bits = expected_bindings % 64;
    if used_tail_bits != 0
        && plan
            .direct_bitmap
            .last()
            .is_some_and(|word| word >> used_tail_bits != 0)
    {
        return Err(PreparedCompositionError::DirectRetentionPlanDrift(
            "bitmap tail bits",
        ));
    }

    let mut consumer = 0usize;
    let mut seen_columns =
        Vec::<(CommitmentTreeId, usize, usize, OpenedColumnSource, u32, u32)>::new();
    for component in components {
        let mut fallback_ordinal = 0usize;
        let mut retention = Vec::with_capacity(component.sources.len());
        for source_ref in &component.sources {
            let binding = plan.bindings.get(consumer).ok_or(
                PreparedCompositionError::DirectRetentionPlanDrift("missing binding"),
            )?;
            if binding.consumer != consumer {
                return Err(PreparedCompositionError::DirectRetentionPlanDrift(
                    "consumer order",
                ));
            }
            if binding.consumer_evaluation_log_size != component.evaluation_log_size {
                return Err(PreparedCompositionError::DirectRetentionPlanDrift(
                    "consumer evaluation log",
                ));
            }
            let column = plan.columns.get(binding.column).ok_or(
                PreparedCompositionError::DirectRetentionPlanDrift("column index"),
            )?;
            let tree = composition_tree(source_ref.tree)?;
            if column.tree != tree {
                return Err(PreparedCompositionError::DirectRetentionPlanDrift(
                    "source tree",
                ));
            }
            if column.proof_column != source_ref.column {
                return Err(PreparedCompositionError::DirectRetentionPlanDrift(
                    "proof column",
                ));
            }
            if column.coefficient_log_size != source_ref.source.log_size {
                return Err(PreparedCompositionError::DirectRetentionPlanDrift(
                    "coefficient log",
                ));
            }
            // Tree/proof-column/log are the prepared engine's execution
            // identity. Full OpenedColumnSource equality is established by
            // the upstream phase-1 protocol planner and sealed plan key; this
            // address-free topology deliberately does not duplicate it.
            if source_tree(column.source)? != tree {
                return Err(PreparedCompositionError::DirectRetentionPlanDrift(
                    "source identity tree",
                ));
            }
            let expected_direct = column.evaluation_log_size == component.evaluation_log_size;
            let bitmap_direct = (plan.direct_bitmap[consumer / 64] >> (consumer % 64)) & 1 != 0;
            if binding.direct != expected_direct || bitmap_direct != binding.direct {
                return Err(PreparedCompositionError::DirectRetentionPlanDrift(
                    "direct bitmap or native log",
                ));
            }
            if let Some((_, _, plan_column, source, coefficient_log, evaluation_log)) =
                seen_columns.iter().find(|(seen_tree, seen_column, ..)| {
                    *seen_tree == tree && *seen_column == source_ref.column
                })
            {
                if *plan_column != binding.column {
                    return Err(PreparedCompositionError::DirectRetentionPlanDrift(
                        "logical source has multiple plan columns",
                    ));
                }
                if *source != column.source
                    || *coefficient_log != column.coefficient_log_size
                    || *evaluation_log != column.evaluation_log_size
                {
                    return Err(PreparedCompositionError::DirectRetentionPlanDrift(
                        "inconsistent duplicate source",
                    ));
                }
            } else {
                seen_columns.push((
                    tree,
                    source_ref.column,
                    binding.column,
                    column.source,
                    column.coefficient_log_size,
                    column.evaluation_log_size,
                ));
            }
            let fallback = (!binding.direct).then_some(fallback_ordinal);
            if fallback.is_some() {
                fallback_ordinal = fallback_ordinal
                    .checked_add(1)
                    .ok_or(PreparedCompositionError::SizeOverflow)?;
            }
            retention.push(CompositionSourceRetention {
                consumer,
                plan_column: binding.column,
                source: column.source,
                tree,
                proof_column: source_ref.column,
                native_evaluation_log_size: column.evaluation_log_size,
                direct: binding.direct,
                fallback_ordinal: fallback,
            });
            consumer += 1;
        }
        component.source_retention = retention;
        component.fallback_count = fallback_ordinal;
    }
    Ok((Some(plan.cache_key), plan.direct_bitmap.clone()))
}

fn composition_tree(tree: usize) -> Result<CommitmentTreeId, PreparedCompositionError> {
    match tree {
        0 => Ok(CommitmentTreeId::Preprocessed),
        1 => Ok(CommitmentTreeId::Base),
        2 => Ok(CommitmentTreeId::Interaction),
        _ => Err(PreparedCompositionError::DirectRetentionPlanDrift(
            "unsupported source tree",
        )),
    }
}

fn source_tree(source: OpenedColumnSource) -> Result<CommitmentTreeId, PreparedCompositionError> {
    match source {
        OpenedColumnSource::Preprocessed { .. } => Ok(CommitmentTreeId::Preprocessed),
        OpenedColumnSource::Trace {
            purpose: crate::arena_plan::BufferPurpose::BaseCoefficients,
            ..
        } => Ok(CommitmentTreeId::Base),
        OpenedColumnSource::Trace {
            purpose: crate::arena_plan::BufferPurpose::InteractionCoefficients,
            ..
        } => Ok(CommitmentTreeId::Interaction),
        _ => Err(PreparedCompositionError::DirectRetentionPlanDrift(
            "unsupported source identity",
        )),
    }
}

/// Size the shared LDE tile and (in wide mode) assign each small group its
/// private region. Serial: every offset stays zero and the tile is the single
/// largest footprint — exactly today's reuse. Wide: large components share
/// the leading `[0, max_large)` region (they stay serial on the main stream,
/// so reuse is safe), then each group of small components gets one region
/// sized for its largest member (members serialize in-lane, so in-group reuse
/// mirrors the serial tile discipline). Regions of distinct groups are
/// disjoint because their lanes execute concurrently.
fn lde_tile_layout(
    mode: CompositionLaunchMode,
    components: &mut [CompositionComponentRequirements],
) -> Result<(usize, Vec<CompositionWideGroup>, Vec<usize>), PreparedCompositionError> {
    let mut lde_tile_words = 0usize;
    if mode != CompositionLaunchMode::Wide {
        for component in components.iter() {
            lde_tile_words = lde_tile_words.max(component_lde_footprint_words(component)?);
        }
        return Ok((lde_tile_words, Vec::new(), (0..components.len()).collect()));
    }

    let mut serial_components = Vec::new();
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (index, component) in components.iter().enumerate() {
        if component.evaluation_log_size <= COMPOSITION_WIDE_SMALL_MAX_EVALUATION_LOG {
            grouped
                .entry(component.evaluation_log_size)
                .or_default()
                .push(index);
        } else {
            serial_components.push(index);
            lde_tile_words = lde_tile_words.max(component_lde_footprint_words(component)?);
        }
    }
    let mut wide_groups = Vec::with_capacity(grouped.len());
    for (evaluation_log_size, members) in grouped {
        let mut tile_len_words = 0usize;
        let mut weight_words = 0u64;
        for &member in &members {
            let component = &components[member];
            tile_len_words = tile_len_words.max(component_lde_footprint_words(component)?);
            weight_words = weight_words.saturating_add(
                (component.row_count as u64)
                    .saturating_mul((component.sources.len() + SECURE_COORDINATES) as u64),
            );
        }
        for &member in &members {
            components[member].lde_tile_offset_words = lde_tile_words;
        }
        wide_groups.push(CompositionWideGroup {
            evaluation_log_size,
            members,
            weight_words,
            tile_offset_words: lde_tile_words,
            tile_len_words,
        });
        lde_tile_words = lde_tile_words
            .checked_add(tile_len_words)
            .ok_or(PreparedCompositionError::SizeOverflow)?;
    }
    Ok((lde_tile_words, wide_groups, serial_components))
}

fn composition_wave_requirements(
    plan: &CompositionPlan,
    components: &[CompositionComponentRequirements],
    mode: CompositionLaunchMode,
    accumulators: &[CompositionAccumulatorRequirements],
) -> Result<Vec<CompositionWaveRequirements>, PreparedCompositionError> {
    if mode != CompositionLaunchMode::Wave {
        return Ok(Vec::new());
    }
    let canonical =
        crate::composition_wave::CompositionWaveProgram::from_plan(plan).map_err(|_| {
            PreparedCompositionError::CompositionWavePlanDrift("canonical wave program")
        })?;
    for (component_index, component) in components.iter().enumerate() {
        if component.fallback_count != 0 {
            return Err(PreparedCompositionError::CompositionWaveRequiresAllDirect {
                component: component_index,
                fallback_count: component.fallback_count,
            });
        }
    }
    if canonical.waves().len() != accumulators.len()
        || plan.wave_kernels.len() != canonical.waves().len()
    {
        return Err(PreparedCompositionError::CompositionWavePlanDrift(
            "wave/accumulator count",
        ));
    }

    let mut waves = Vec::with_capacity(canonical.waves().len());
    for (wave_index, (canonical_wave, wave)) in
        canonical.waves().iter().zip(&plan.wave_kernels).enumerate()
    {
        let evaluation_log_size = canonical_wave.evaluation_log_size;
        if wave.evaluation_log_size != evaluation_log_size {
            return Err(PreparedCompositionError::CompositionWavePlanDrift(
                "evaluation-log order",
            ));
        }
        let parts = canonical_wave
            .part_ordinals
            .iter()
            .map(|&ordinal| {
                let part = canonical.parts().get(ordinal).ok_or(
                    PreparedCompositionError::CompositionWavePlanDrift("canonical part ordinal"),
                )?;
                if components
                    .get(part.component_index)
                    .is_none_or(|component| component.evaluation_log_size != evaluation_log_size)
                {
                    return Err(PreparedCompositionError::CompositionWavePlanDrift(
                        "canonical component/log",
                    ));
                }
                Ok(CompositionWavePartRequirement {
                    component: part.component_index,
                    kernel: part.kernel_index,
                    identity: aot::CompositionWaveKernelPartIdentity {
                        semantic_hash: part.semantic_hash,
                        coefficient_start: u32::try_from(part.coefficient_start)
                            .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                        coefficient_end: u32::try_from(part.coefficient_end)
                            .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                    },
                })
            })
            .collect::<Result<Vec<_>, PreparedCompositionError>>()?;
        let identities = parts.iter().map(|part| part.identity).collect::<Vec<_>>();
        if wave.parts != identities {
            return Err(PreparedCompositionError::CompositionWavePlanDrift(
                "part identity/order",
            ));
        }
        let expected = aot::composition_wave_kernel_identity(evaluation_log_size, &identities)
            .ok_or(PreparedCompositionError::CompositionWavePlanDrift(
                "invalid identity",
            ))?;
        if expected.part_count != parts.len()
            || expected.kernel_name != wave.kernel_name
            || expected.cache_key != wave.cache_key
            || expected.semantic_hash != wave.semantic_hash
        {
            return Err(PreparedCompositionError::CompositionWavePlanDrift(
                "kernel identity",
            ));
        }
        let accumulator = accumulators
            .iter()
            .find(|accumulator| accumulator.log_size == evaluation_log_size)
            .ok_or(PreparedCompositionError::CompositionWavePlanDrift(
                "missing accumulator owner",
            ))?;
        let row_count = pow2(evaluation_log_size)?;
        if accumulator.len_words
            != row_count
                .checked_mul(SECURE_COORDINATES)
                .ok_or(PreparedCompositionError::SizeOverflow)?
        {
            return Err(PreparedCompositionError::CompositionWavePlanDrift(
                "accumulator extent",
            ));
        }
        debug_assert_eq!(wave_index, waves.len());
        waves.push(CompositionWaveRequirements {
            evaluation_log_size,
            row_count,
            accumulator_offset_words: accumulator.offset_words,
            descriptor_offset_words: 0,
            parts,
        });
    }
    Ok(waves)
}

fn validate_component_program(
    component_index: usize,
    component: &CompositionComponentPlan,
    expected_random_offset: usize,
) -> Result<(), PreparedCompositionError> {
    if component.ext_param_values.len() != component.ext_param_sources.len() {
        return Err(PreparedCompositionError::ExtParamSourceCount {
            component: component_index,
            values: component.ext_param_values.len(),
            sources: component.ext_param_sources.len(),
        });
    }
    for (slot, (&value, source)) in component
        .ext_param_values
        .iter()
        .zip(&component.ext_param_sources)
        .enumerate()
    {
        if let CompositionExtParamSource::Constant(constant) = source {
            if value != *constant {
                return Err(PreparedCompositionError::ConstantExtParamMismatch {
                    component: component_index,
                    slot,
                });
            }
        }
    }
    if !(2..=30).contains(&component.evaluation_log_size)
        || component.trace_log_size >= component.evaluation_log_size
    {
        return Err(PreparedCompositionError::InvalidLogSize {
            component: component_index,
            trace_log_size: component.trace_log_size,
            evaluation_log_size: component.evaluation_log_size,
        });
    }
    if component.random_coefficient_offset != expected_random_offset {
        return Err(PreparedCompositionError::RandomCoefficientOrder {
            component: component_index,
            expected: expected_random_offset,
            actual: component.random_coefficient_offset,
        });
    }
    if component.kernels.is_empty() {
        return Err(PreparedCompositionError::EmptyKernelProgram(
            component_index,
        ));
    }
    for (kernel_index, kernel) in component.kernels.iter().enumerate() {
        if kernel.cache_key == 0
            || kernel.semantic_hash == 0
            || kernel.kernel_name.is_empty()
            || kernel.source.is_empty()
        {
            return Err(PreparedCompositionError::InvalidKernelIdentity {
                component: component_index,
                kernel: kernel_index,
            });
        }
        if kernel.rc_base as usize >= component.n_constraints {
            return Err(PreparedCompositionError::InvalidKernelRandomOffset {
                component: component_index,
                kernel: kernel_index,
                rc_base: kernel.rc_base,
                constraints: component.n_constraints,
            });
        }
    }
    Ok(())
}

fn select_component_sources(
    component_index: usize,
    component: &CompositionComponentPlan,
    trace: &CompositionTraceTopology,
) -> Result<Vec<CompositionSourceRef>, PreparedCompositionError> {
    let mut spans = [None; TRACE_TREES];
    for span in &component.trace_locations {
        if span.tree_index >= TRACE_TREES {
            return Err(PreparedCompositionError::UnsupportedTraceTree {
                component: component_index,
                tree: span.tree_index,
            });
        }
        if spans[span.tree_index].replace(*span).is_some() {
            return Err(PreparedCompositionError::DuplicateTraceTree {
                component: component_index,
                tree: span.tree_index,
            });
        }
    }
    for (tree, span) in spans.iter().enumerate() {
        if span.is_none() {
            return Err(PreparedCompositionError::MissingTraceTree {
                component: component_index,
                tree,
            });
        }
    }

    let mut selected = Vec::new();
    for &column in &component.preprocessed_column_indices {
        let Some(&source) = trace.trees[0].get(column) else {
            return Err(PreparedCompositionError::InvalidPreprocessedColumn {
                component: component_index,
                column,
                columns: trace.trees[0].len(),
            });
        };
        selected.push(CompositionSourceRef {
            tree: 0,
            column,
            source,
        });
    }
    for tree in 1..TRACE_TREES {
        let span = spans[tree].expect("all spans validated");
        if span.col_start > span.col_end || span.col_end > trace.trees[tree].len() {
            return Err(PreparedCompositionError::InvalidTraceRange {
                component: component_index,
                tree,
                start: span.col_start,
                end: span.col_end,
                columns: trace.trees[tree].len(),
            });
        }
        selected.extend(
            trace.trees[tree][span.col_start..span.col_end]
                .iter()
                .copied()
                .enumerate()
                .map(|(offset, source)| CompositionSourceRef {
                    tree,
                    column: span.col_start + offset,
                    source,
                }),
        );
    }
    Ok(selected)
}

fn interaction_offsets(
    sources: &[CompositionSourceRef],
) -> Result<[u32; TRACE_TREES], PreparedCompositionError> {
    let mut counts = [0usize; TRACE_TREES];
    for source in sources {
        counts[source.tree] += 1;
    }
    let base = counts[0];
    let interaction = base
        .checked_add(counts[1])
        .ok_or(PreparedCompositionError::SizeOverflow)?;
    Ok([
        0,
        u32::try_from(base).map_err(|_| PreparedCompositionError::SizeOverflow)?,
        u32::try_from(interaction).map_err(|_| PreparedCompositionError::SizeOverflow)?,
    ])
}

#[derive(Default)]
struct DescriptorAllocator {
    cursor: usize,
}

impl DescriptorAllocator {
    fn take(
        &mut self,
        words: usize,
        alignment_words: usize,
    ) -> Result<usize, PreparedCompositionError> {
        self.cursor = self
            .cursor
            .checked_next_multiple_of(alignment_words)
            .ok_or(PreparedCompositionError::SizeOverflow)?;
        let offset = self.cursor;
        self.cursor = self
            .cursor
            .checked_add(words)
            .ok_or(PreparedCompositionError::SizeOverflow)?;
        Ok(offset)
    }
}

#[derive(Debug)]
struct PreparedKernel {
    source: CString,
    name: CString,
    cache_key: u64,
    rc_base: u32,
}

#[derive(Debug)]
struct PreparedComponent {
    evaluation_pointers: usize,
    fallback_coefficient_pointers: usize,
    fallback_coefficient_sizes: usize,
    fallback_evaluation_pointers: usize,
    interaction_offsets: usize,
    denominator_inverses: usize,
    base_params: usize,
    ext_params: *const u32,
    accumulator_offset_words: usize,
    trace_log_size: u32,
    evaluation_log_size: u32,
    row_count: u32,
    fallback_count: u32,
    kernels: Vec<PreparedKernel>,
}

#[derive(Debug)]
struct PreparedWave {
    name: CString,
    cache_key: u64,
    descriptor_offset_words: usize,
    accumulator_offset_words: usize,
    row_count: u32,
}

/// Stable resident composition launch object.
pub struct PreparedCompositionGraph<'a> {
    arena: &'a DeviceArena,
    requirements: CompositionWorkspaceRequirements,
    descriptors: ArenaSlice,
    // Retained as an explicit graph binding: immutable device pointer tables
    // refer into this reusable tile for the graph's entire lifetime.
    _lde_tile: ArenaSlice,
    accumulators: ArenaSlice,
    random_coefficient: ArenaSlice,
    random_coefficient_powers: ArenaSlice,
    forward_twiddles: ArenaSlice,
    inverse_twiddles: ArenaSlice,
    relation_z: ArenaSlice,
    relation_alpha_powers: ArenaSlice,
    _claimed_sums: Vec<ArenaSlice>,
    _direct_evaluations: Vec<ArenaSlice>,
    composition_coefficients: [ArenaSlice; SPLIT_COORDINATES],
    components: Vec<PreparedComponent>,
    waves: Vec<PreparedWave>,
    /// Wide-mode fanout: `lane_components[lane]` holds component indices in
    /// enqueue order (group-contiguous, members in plan order). Empty in
    /// serial mode, so the serial launch path performs no fork/join at all.
    lane_components: Vec<Vec<usize>>,
}

impl<'a> PreparedCompositionGraph<'a> {
    /// Prepare in the process-default launch mode (see
    /// [`default_composition_launch_mode`]).
    pub fn prepare(
        arena: &'a DeviceArena,
        plan: &CompositionPlan,
        trace: &CompositionTraceTopology,
        inputs: &CompositionDeviceInputs,
        slots: &CompositionWorkspaceSlots,
    ) -> Result<Self, PreparedCompositionError> {
        Self::prepare_with_mode(
            arena,
            plan,
            trace,
            inputs,
            slots,
            default_composition_launch_mode(),
        )
    }

    /// Mode-explicit form of [`PreparedCompositionGraph::prepare`]. Parity
    /// tests use this to run both stream topologies in one invocation.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_with_mode(
        arena: &'a DeviceArena,
        plan: &CompositionPlan,
        trace: &CompositionTraceTopology,
        inputs: &CompositionDeviceInputs,
        slots: &CompositionWorkspaceSlots,
        mode: CompositionLaunchMode,
    ) -> Result<Self, PreparedCompositionError> {
        Self::prepare_with_mode_and_retention(arena, plan, trace, inputs, slots, mode, None, &[])
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "direct-retention-test-api")]
    #[doc(hidden)]
    pub fn prepare_with_mode_and_retention_for_test(
        arena: &'a DeviceArena,
        plan: &CompositionPlan,
        trace: &CompositionTraceTopology,
        inputs: &CompositionDeviceInputs,
        slots: &CompositionWorkspaceSlots,
        mode: CompositionLaunchMode,
        direct_retention: Option<&DirectCompositionRetentionPlan>,
        direct_evaluations: &[CompositionDirectEvaluationBinding],
    ) -> Result<Self, PreparedCompositionError> {
        Self::prepare_with_mode_and_retention(
            arena,
            plan,
            trace,
            inputs,
            slots,
            mode,
            direct_retention,
            direct_evaluations,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_with_mode_and_retention(
        arena: &'a DeviceArena,
        plan: &CompositionPlan,
        trace: &CompositionTraceTopology,
        inputs: &CompositionDeviceInputs,
        slots: &CompositionWorkspaceSlots,
        mode: CompositionLaunchMode,
        direct_retention: Option<&DirectCompositionRetentionPlan>,
        direct_evaluations: &[CompositionDirectEvaluationBinding],
    ) -> Result<Self, PreparedCompositionError> {
        Self::prepare_impl(
            arena,
            plan,
            None,
            trace,
            inputs,
            slots,
            mode,
            direct_retention,
            direct_evaluations,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_with_proof_bindings(
        arena: &'a DeviceArena,
        plan: &CompositionPlan,
        proof_bindings: &CompositionProofBindings,
        trace: &CompositionTraceTopology,
        inputs: &CompositionDeviceInputs,
        slots: &CompositionWorkspaceSlots,
        mode: CompositionLaunchMode,
        direct_retention: Option<&DirectCompositionRetentionPlan>,
        direct_evaluations: &[CompositionDirectEvaluationBinding],
    ) -> Result<Self, PreparedCompositionError> {
        Self::prepare_impl(
            arena,
            plan,
            Some(proof_bindings),
            trace,
            inputs,
            slots,
            mode,
            direct_retention,
            direct_evaluations,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_impl(
        arena: &'a DeviceArena,
        plan: &CompositionPlan,
        proof_bindings: Option<&CompositionProofBindings>,
        trace: &CompositionTraceTopology,
        inputs: &CompositionDeviceInputs,
        slots: &CompositionWorkspaceSlots,
        mode: CompositionLaunchMode,
        direct_retention: Option<&DirectCompositionRetentionPlan>,
        direct_evaluations: &[CompositionDirectEvaluationBinding],
    ) -> Result<Self, PreparedCompositionError> {
        let requirements =
            composition_workspace_requirements_with_retention(plan, trace, mode, direct_retention)?;
        let lane_components = if requirements.wide_groups.is_empty() {
            Vec::new()
        } else {
            let lane_count = arena.context().lane_count();
            if lane_count == 0 {
                return Err(PreparedCompositionError::NoComponentLanes);
            }
            pack_composition_wide_groups(&requirements.wide_groups, lane_count)
                .into_iter()
                .map(|groups| {
                    groups
                        .into_iter()
                        .flat_map(|group| requirements.wide_groups[group].members.iter().copied())
                        .collect()
                })
                .collect()
        };
        if inputs.ext_params.len() != requirements.components.len() {
            return Err(PreparedCompositionError::ExtParamBindingCount {
                expected: requirements.components.len(),
                actual: inputs.ext_params.len(),
            });
        }
        if inputs.claimed_sums.len() != requirements.components.len() {
            return Err(PreparedCompositionError::ClaimedSumBindingCount {
                expected: requirements.components.len(),
                actual: inputs.claimed_sums.len(),
            });
        }
        if let Some(bindings) = proof_bindings {
            if bindings.component_count() != requirements.components.len() {
                return Err(PreparedCompositionError::BaseParamBindingCount {
                    expected: requirements.components.len(),
                    actual: bindings.component_count(),
                });
            }
            let expected_words = requirements
                .components
                .iter()
                .try_fold(0usize, |total, component| {
                    total.checked_add(component.base_param_words)
                })
                .ok_or(PreparedCompositionError::SizeOverflow)?;
            if bindings.base_param_word_count() != expected_words {
                return Err(PreparedCompositionError::BaseParamBindingTotalWords {
                    expected: expected_words,
                    actual: bindings.base_param_word_count(),
                });
            }
        }
        let slot_requirements = requirements.arena_slot_requirements(slots)?;
        let descriptor_requirement = slot_requirements[0];
        let descriptors = bind_slot(arena, descriptor_requirement)?;
        let lde_tile = bind_slot(arena, slot_requirements[1])?;
        let accumulators = bind_slot(arena, slot_requirements[2])?;
        let random_coefficient_powers = bind_slot(arena, slot_requirements[3])?;
        let composition_coefficients: [ArenaSlice; SPLIT_COORDINATES] = slot_requirements[4..]
            .iter()
            .map(|&requirement| bind_slot(arena, requirement))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .expect("exactly eight output requirements");
        let random_coefficient = bind_minimum(arena, inputs.random_coefficient, SECURE_WORDS)?;
        let forward_twiddles = require_input_min(
            arena,
            inputs.forward_twiddles,
            requirements.forward_twiddle_words,
        )?;
        let inverse_twiddles = require_input_min(
            arena,
            inputs.inverse_twiddles,
            requirements.inverse_twiddle_words,
        )?;
        if inputs.forward_twiddles.id() == inputs.inverse_twiddles.id() {
            return Err(PreparedCompositionError::ForwardInverseTwiddlesAlias(
                inputs.forward_twiddles.id(),
            ));
        }

        let core_writable_ids = slot_requirements
            .iter()
            .map(|requirement| requirement.id)
            .collect::<BTreeSet<_>>();
        let mut ext_param_ids = BTreeSet::new();
        for binding in inputs.ext_params.iter().flatten() {
            if !ext_param_ids.insert(binding.slot) {
                return Err(PreparedCompositionError::DuplicateExtParamBinding(
                    binding.slot,
                ));
            }
            if core_writable_ids.contains(&binding.slot) {
                return Err(PreparedCompositionError::InputAliasesWritableWorkspace(
                    binding.slot,
                ));
            }
        }
        let writable_ids = core_writable_ids
            .union(&ext_param_ids)
            .copied()
            .collect::<BTreeSet<_>>();
        let direct_columns = requirements
            .components
            .iter()
            .flat_map(|component| &component.source_retention)
            .filter(|source| source.direct)
            .map(|source| source.plan_column)
            .collect::<BTreeSet<_>>();
        let expected_direct_bindings = direct_columns.len();
        if direct_evaluations.len() != expected_direct_bindings {
            return Err(PreparedCompositionError::DirectEvaluationBindingCount {
                expected: expected_direct_bindings,
                actual: direct_evaluations.len(),
            });
        }
        let canonical_column_count = direct_retention.map_or(0, |plan| plan.columns.len());
        let mut direct_by_plan_column = vec![None; canonical_column_count];
        let unrelated_readonly_ids = [
            inputs.random_coefficient,
            inputs.forward_twiddles.id(),
            inputs.inverse_twiddles.id(),
            inputs.relation_z.id(),
            inputs.relation_alpha_powers.id(),
        ]
        .into_iter()
        .chain(inputs.claimed_sums.iter().flatten().copied())
        .collect::<BTreeSet<_>>();
        let mut direct_by_slot = Vec::<(ArenaSlotId, CommitmentTreeId, usize, usize)>::new();
        for (binding_index, binding) in direct_evaluations.iter().enumerate() {
            if !direct_columns.contains(&binding.plan_column) {
                return Err(PreparedCompositionError::UnexpectedDirectEvaluation(
                    binding.plan_column,
                ));
            }
            let destination = direct_by_plan_column.get_mut(binding.plan_column).ok_or(
                PreparedCompositionError::UnexpectedDirectEvaluation(binding.plan_column),
            )?;
            if destination.is_some() {
                return Err(PreparedCompositionError::DuplicateDirectEvaluationColumn(
                    binding.plan_column,
                ));
            }
            if !binding.evaluation.belongs_to(arena.context()) {
                return Err(PreparedCompositionError::ContextMismatch(
                    binding.evaluation.id(),
                ));
            }
            if writable_ids.contains(&binding.evaluation.id()) {
                return Err(
                    PreparedCompositionError::DirectEvaluationAliasesWritableWorkspace(
                        binding.evaluation.id(),
                    ),
                );
            }
            if unrelated_readonly_ids.contains(&binding.evaluation.id()) {
                return Err(
                    PreparedCompositionError::DirectEvaluationAliasesUnrelatedInput(
                        binding.evaluation.id(),
                    ),
                );
            }
            let column = direct_retention
                .and_then(|plan| plan.columns.get(binding.plan_column))
                .ok_or(PreparedCompositionError::UnexpectedDirectEvaluation(
                    binding.plan_column,
                ))?;
            let evaluation =
                require_input_min(arena, binding.evaluation, pow2(column.evaluation_log_size)?)?;
            if let Some((_, tree, proof_column, first)) = direct_by_slot
                .iter()
                .find(|(slot, ..)| *slot == evaluation.id())
            {
                if (*tree, *proof_column) != (column.tree, column.proof_column) {
                    return Err(
                        PreparedCompositionError::InconsistentDuplicateDirectEvaluation {
                            first: *first,
                            second: binding_index,
                        },
                    );
                }
            } else {
                direct_by_slot.push((
                    evaluation.id(),
                    column.tree,
                    column.proof_column,
                    binding_index,
                ));
            }
            *destination = Some(evaluation);
        }
        if let Some(&missing) = direct_columns
            .iter()
            .find(|&&column| direct_by_plan_column[column].is_none())
        {
            return Err(PreparedCompositionError::MissingDirectEvaluation(missing));
        }
        for tree in &trace.trees {
            for source in tree {
                if writable_ids.contains(&source.slot) {
                    return Err(PreparedCompositionError::SourceAliasesWritableWorkspace(
                        source.slot,
                    ));
                }
                let _ = bind_minimum(arena, source.slot, pow2(source.log_size)?)?;
                if direct_evaluations
                    .iter()
                    .any(|binding| binding.evaluation.id() == source.slot)
                {
                    return Err(
                        PreparedCompositionError::DirectEvaluationAliasesCoefficient(source.slot),
                    );
                }
            }
        }
        for input in [
            inputs.random_coefficient,
            inputs.forward_twiddles.id(),
            inputs.inverse_twiddles.id(),
            inputs.relation_z.id(),
            inputs.relation_alpha_powers.id(),
        ] {
            if writable_ids.contains(&input) {
                return Err(PreparedCompositionError::InputAliasesWritableWorkspace(
                    input,
                ));
            }
        }
        if inputs.relation_z.id() == inputs.relation_alpha_powers.id() {
            return Err(PreparedCompositionError::RelationChallengeSourcesAlias(
                inputs.relation_z.id(),
            ));
        }
        for &claimed_sum in inputs.claimed_sums.iter().flatten() {
            if writable_ids.contains(&claimed_sum) {
                return Err(PreparedCompositionError::InputAliasesWritableWorkspace(
                    claimed_sum,
                ));
            }
        }

        // The relation graph binds these to their logical challenge extents;
        // the kernel reads exactly one QM31 z value and
        // `len_words() / SECURE_WORDS` alpha powers, so the caller's slice
        // length is load-bearing and must never be a pooled slot length.
        let relation_z = require_input_min(arena, inputs.relation_z, SECURE_WORDS)?;
        let relation_alpha_powers =
            require_input_min(arena, inputs.relation_alpha_powers, SECURE_WORDS)?;
        if relation_alpha_powers.len_words() % SECURE_WORDS != 0 {
            return Err(PreparedCompositionError::SlotTooSmall {
                slot: relation_alpha_powers.id(),
                required_words: relation_alpha_powers
                    .len_words()
                    .next_multiple_of(SECURE_WORDS),
                actual_words: relation_alpha_powers.len_words(),
            });
        }
        let alpha_power_count = relation_alpha_powers.len_words() / SECURE_WORDS;

        if aot::loaded_manifest_hash() == 0 {
            return Err(PreparedCompositionError::AotPackUnavailable);
        }
        // Strictness is monotonic process-wide. A prior runtime-compiled entry
        // cannot satisfy this graph after this point.
        aot::require_loaded_kernels();

        let mut descriptor_words = vec![0u32; requirements.descriptor_words];
        let max_accumulator = requirements
            .accumulators
            .last()
            .expect("non-empty plan has an accumulator");
        let max_rows = pow2(max_accumulator.log_size)?;
        for coordinate in 0..SECURE_COORDINATES {
            let pointer = unsafe {
                accumulators
                    .as_u32_ptr()
                    .add(max_accumulator.offset_words + coordinate * max_rows)
            };
            write_pointer(
                &mut descriptor_words,
                requirements.final_coordinate_pointers + coordinate * POINTER_WORDS,
                pointer,
            );
        }

        let mut prepared_components = Vec::with_capacity(requirements.components.len());
        let mut ext_uploads = Vec::<(*mut c_void, Vec<u32>)>::new();
        let mut claimed_sums = Vec::with_capacity(requirements.claimed_sum_count);
        let mut dynamic_index = 0usize;
        let mut claimed_index = 0usize;
        let direct_evaluation_slices = direct_by_plan_column
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        for (
            component_index,
            ((((component_plan, component), descriptor), ext_binding), claimed_sum_binding),
        ) in plan
            .components
            .iter()
            .zip(&requirements.components)
            .zip(&requirements.component_descriptors)
            .zip(&inputs.ext_params)
            .zip(&inputs.claimed_sums)
            .enumerate()
        {
            let base_param_values = match proof_bindings {
                Some(bindings) => {
                    let (binding_component, binding_instance, binding_values) =
                        bindings.component(component_index).ok_or(
                            PreparedCompositionError::BaseParamBindingIdentity(component_index),
                        )?;
                    if binding_component != component_plan.component
                        || binding_instance != component_plan.instance
                    {
                        return Err(PreparedCompositionError::BaseParamBindingIdentity(
                            component_index,
                        ));
                    }
                    if binding_values.len() != component.base_param_words {
                        return Err(PreparedCompositionError::BaseParamBindingWords {
                            component: component_index,
                            expected: component.base_param_words,
                            actual: binding_values.len(),
                        });
                    }
                    binding_values
                }
                None => component_plan.base_param_values.as_slice(),
            };
            let row_count = component.row_count;
            for (source_index, source_ref) in component.sources.iter().enumerate() {
                let source = bind_minimum(
                    arena,
                    source_ref.source.slot,
                    pow2(source_ref.source.log_size)?,
                )?;
                write_pointer(
                    &mut descriptor_words,
                    descriptor.coefficient_pointers + source_index * POINTER_WORDS,
                    source.as_u32_ptr(),
                );
                descriptor_words[descriptor.coefficient_sizes + source_index] =
                    u32::try_from(pow2(source_ref.source.log_size)?)
                        .map_err(|_| PreparedCompositionError::SizeOverflow)?;
                let retention = component.source_retention.get(source_index);
                let evaluation = match retention {
                    None => unsafe {
                        lde_tile.as_u32_ptr().add(
                            source_index
                                .checked_mul(row_count)
                                .and_then(|words| {
                                    words.checked_add(component.lde_tile_offset_words)
                                })
                                .ok_or(PreparedCompositionError::SizeOverflow)?,
                        )
                    },
                    Some(retention) if retention.direct => {
                        let evaluation = direct_by_plan_column[retention.plan_column]
                            .expect("canonical direct bindings were validated");
                        evaluation.as_u32_ptr()
                    }
                    Some(retention) => {
                        let fallback = retention.fallback_ordinal.expect("fallback has ordinal");
                        let evaluation = unsafe {
                            lde_tile.as_u32_ptr().add(
                                fallback
                                    .checked_mul(row_count)
                                    .and_then(|words| {
                                        words.checked_add(component.lde_tile_offset_words)
                                    })
                                    .ok_or(PreparedCompositionError::SizeOverflow)?,
                            )
                        };
                        write_pointer(
                            &mut descriptor_words,
                            descriptor
                                .fallback_coefficient_pointers
                                .expect("retention descriptors")
                                + fallback * POINTER_WORDS,
                            source.as_u32_ptr(),
                        );
                        descriptor_words[descriptor
                            .fallback_coefficient_sizes
                            .expect("retention descriptors")
                            + fallback] = u32::try_from(pow2(source_ref.source.log_size)?)
                            .map_err(|_| PreparedCompositionError::SizeOverflow)?;
                        write_pointer(
                            &mut descriptor_words,
                            descriptor
                                .fallback_evaluation_pointers
                                .expect("retention descriptors")
                                + fallback * POINTER_WORDS,
                            evaluation,
                        );
                        evaluation
                    }
                };
                write_pointer(
                    &mut descriptor_words,
                    descriptor.evaluation_pointers + source_index * POINTER_WORDS,
                    evaluation,
                );
            }
            descriptor_words
                [descriptor.interaction_offsets..descriptor.interaction_offsets + TRACE_TREES]
                .copy_from_slice(&component.interaction_offsets);
            for (word, value) in descriptor_words[descriptor.denominator_inverses
                ..descriptor.denominator_inverses + component.denominator_words]
                .iter_mut()
                .zip(&component_plan.denominator_inverses)
            {
                *word = value.0;
            }
            if component.base_param_words != 0 {
                for (word, value) in descriptor_words
                    [descriptor.base_params..descriptor.base_params + component.base_param_words]
                    .iter_mut()
                    .zip(base_param_values)
                {
                    *word = value.0;
                }
            }

            let ext_params = match (component.ext_param_words, ext_binding) {
                (0, None) => unsafe { descriptors.as_u32_ptr().add(requirements.zero_words) },
                (0, Some(_)) => {
                    return Err(PreparedCompositionError::UnexpectedExtParamBinding(
                        component_index,
                    ));
                }
                (words, None) => {
                    return Err(PreparedCompositionError::MissingExtParamBinding {
                        component: component_index,
                        words,
                    });
                }
                (words, Some(binding)) => {
                    if binding.offset_words % SECURE_WORDS != 0 {
                        return Err(PreparedCompositionError::MisalignedExtParamBinding {
                            component: component_index,
                            offset_words: binding.offset_words,
                        });
                    }
                    if core_writable_ids.contains(&binding.slot) {
                        return Err(PreparedCompositionError::InputAliasesWritableWorkspace(
                            binding.slot,
                        ));
                    }
                    let required = binding
                        .offset_words
                        .checked_add(words)
                        .ok_or(PreparedCompositionError::SizeOverflow)?;
                    let values = bind_minimum(arena, binding.slot, required)?;
                    let ext_params = unsafe { values.as_u32_ptr().add(binding.offset_words) };
                    let mut upload_words = vec![0u32; words];
                    let needs_claimed_sum = component_plan.ext_param_sources.iter().any(|source| {
                        matches!(source, CompositionExtParamSource::ClaimedSumScaled)
                    });
                    let claimed_sum = match (needs_claimed_sum, claimed_sum_binding) {
                        (true, Some(slot)) => Some(bind_minimum(arena, *slot, SECURE_WORDS)?),
                        (true, None) => {
                            return Err(PreparedCompositionError::MissingClaimedSumBinding(
                                component_index,
                            ));
                        }
                        (false, Some(_)) => {
                            return Err(PreparedCompositionError::UnexpectedClaimedSumBinding(
                                component_index,
                            ));
                        }
                        (false, None) => None,
                    };
                    for (slot, source) in component_plan.ext_param_sources.iter().enumerate() {
                        let destination = unsafe { ext_params.add(slot * SECURE_WORDS) };
                        match *source {
                            CompositionExtParamSource::Constant(value) => {
                                upload_words[slot * SECURE_WORDS..(slot + 1) * SECURE_WORDS]
                                    .copy_from_slice(&secure_words(value));
                            }
                            CompositionExtParamSource::LookupZ => {
                                write_dynamic_ext_param(
                                    &mut descriptor_words,
                                    &requirements,
                                    dynamic_index,
                                    destination,
                                    0,
                                    0,
                                    M31::from_u32_unchecked(1).0,
                                );
                                dynamic_index += 1;
                            }
                            CompositionExtParamSource::LookupAlphaPower(power) => {
                                let power_index = usize::try_from(power)
                                    .map_err(|_| PreparedCompositionError::SizeOverflow)?;
                                if power_index >= alpha_power_count {
                                    return Err(PreparedCompositionError::AlphaPowerOutOfRange {
                                        component: component_index,
                                        power,
                                        available: alpha_power_count,
                                    });
                                }
                                write_dynamic_ext_param(
                                    &mut descriptor_words,
                                    &requirements,
                                    dynamic_index,
                                    destination,
                                    1,
                                    power,
                                    M31::from_u32_unchecked(1).0,
                                );
                                dynamic_index += 1;
                            }
                            CompositionExtParamSource::LookupAlphaPowerScaled { power, scale } => {
                                let power_index = usize::try_from(power)
                                    .map_err(|_| PreparedCompositionError::SizeOverflow)?;
                                if power_index >= alpha_power_count {
                                    return Err(PreparedCompositionError::AlphaPowerOutOfRange {
                                        component: component_index,
                                        power,
                                        available: alpha_power_count,
                                    });
                                }
                                write_dynamic_ext_param(
                                    &mut descriptor_words,
                                    &requirements,
                                    dynamic_index,
                                    destination,
                                    1,
                                    power,
                                    scale.0,
                                );
                                dynamic_index += 1;
                            }
                            CompositionExtParamSource::ClaimedSumScaled => {
                                let claimed_sum = claimed_sum
                                    .expect("claimed-sum binding was validated for component");
                                write_pointer(
                                    &mut descriptor_words,
                                    requirements.claimed_sum_pointers
                                        + claimed_index * POINTER_WORDS,
                                    claimed_sum.as_u32_ptr(),
                                );
                                write_dynamic_ext_param(
                                    &mut descriptor_words,
                                    &requirements,
                                    dynamic_index,
                                    destination,
                                    2,
                                    u32::try_from(claimed_index)
                                        .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                                    M31::from_u32_unchecked(1u32 << component.trace_log_size)
                                        .inverse()
                                        .0,
                                );
                                claimed_sums.push(claimed_sum);
                                claimed_index += 1;
                                dynamic_index += 1;
                            }
                        }
                    }
                    ext_uploads.push((ext_params.cast::<c_void>(), upload_words));
                    ext_params
                }
            };

            if component.ext_param_words == 0 && claimed_sum_binding.is_some() {
                return Err(PreparedCompositionError::UnexpectedClaimedSumBinding(
                    component_index,
                ));
            }

            let mut kernels = Vec::new();
            if mode != CompositionLaunchMode::Wave {
                kernels.reserve(component_plan.kernels.len());
                for (kernel_index, kernel) in component_plan.kernels.iter().enumerate() {
                    kernels.push(prepare_aot_kernel(component_index, kernel_index, kernel)?);
                }
            }
            prepared_components.push(PreparedComponent {
                evaluation_pointers: descriptor.evaluation_pointers,
                fallback_coefficient_pointers: descriptor
                    .fallback_coefficient_pointers
                    .unwrap_or(descriptor.coefficient_pointers),
                fallback_coefficient_sizes: descriptor
                    .fallback_coefficient_sizes
                    .unwrap_or(descriptor.coefficient_sizes),
                fallback_evaluation_pointers: descriptor
                    .fallback_evaluation_pointers
                    .unwrap_or(descriptor.evaluation_pointers),
                interaction_offsets: descriptor.interaction_offsets,
                denominator_inverses: descriptor.denominator_inverses,
                base_params: descriptor.base_params,
                ext_params,
                accumulator_offset_words: component.accumulator_offset_words,
                trace_log_size: component.trace_log_size,
                evaluation_log_size: component.evaluation_log_size,
                row_count: u32::try_from(component.row_count)
                    .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                fallback_count: u32::try_from(component.fallback_count)
                    .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                kernels,
            });
        }
        debug_assert_eq!(dynamic_index, requirements.dynamic_ext_param_count);
        debug_assert_eq!(claimed_index, requirements.claimed_sum_count);

        let descriptor_ptr = descriptors.as_u32_ptr();
        let mut prepared_waves = Vec::with_capacity(requirements.waves.len());
        for (wave_index, wave) in requirements.waves.iter().enumerate() {
            for (part_index, part) in wave.parts.iter().enumerate() {
                let component = &prepared_components[part.component];
                let base = wave
                    .descriptor_offset_words
                    .checked_add(
                        part_index
                            .checked_mul(WAVE_PART_WORDS)
                            .ok_or(PreparedCompositionError::SizeOverflow)?,
                    )
                    .ok_or(PreparedCompositionError::SizeOverflow)?;
                write_wave_part_descriptor(
                    &mut descriptor_words,
                    base,
                    descriptor_ptr,
                    component,
                    part.identity.coefficient_start,
                );
            }
            prepared_waves.push(prepare_aot_wave(
                wave_index,
                &plan.wave_kernels[wave_index],
                wave,
            )?);
        }

        // The host descriptor is immutable and may be dropped only after the
        // setup upload completes. No setup values are uploaded on replay.
        unsafe {
            arena.context().memcpy_h2d_async(
                descriptors.as_void_ptr(),
                descriptor_words.as_ptr().cast::<c_void>(),
                descriptor_words
                    .len()
                    .checked_mul(WORD_BYTES)
                    .ok_or(PreparedCompositionError::SizeOverflow)?,
            )?;
            for (destination, words) in &ext_uploads {
                arena.context().memcpy_h2d_async(
                    *destination,
                    words.as_ptr().cast::<c_void>(),
                    words
                        .len()
                        .checked_mul(WORD_BYTES)
                        .ok_or(PreparedCompositionError::SizeOverflow)?,
                )?;
            }
        }
        arena.context().sync()?;

        Ok(Self {
            arena,
            requirements,
            descriptors,
            _lde_tile: lde_tile,
            accumulators,
            random_coefficient,
            random_coefficient_powers,
            forward_twiddles,
            inverse_twiddles,
            relation_z,
            relation_alpha_powers,
            _claimed_sums: claimed_sums,
            _direct_evaluations: direct_evaluation_slices,
            composition_coefficients,
            components: prepared_components,
            waves: prepared_waves,
            lane_components,
        })
    }

    /// Enqueue the complete composition path. Eager execution and graph capture
    /// call this same method and therefore have identical launch topology.
    pub fn launch(&self) -> Result<(), PreparedCompositionError> {
        let context = self.arena.context();
        let stream = context.stream_raw().as_ptr();
        let descriptor_ptr = self.descriptors.as_u32_ptr();
        if self.requirements.dynamic_ext_param_count != 0 {
            check_status("composition_materialize_ext_params", unsafe {
                raw::stwo_composition_materialize_ext_params_on(
                    descriptor_ptr
                        .add(self.requirements.dynamic_destination_pointers)
                        .cast::<*mut CudaSecureField>(),
                    descriptor_ptr.add(self.requirements.dynamic_source_kinds),
                    descriptor_ptr.add(self.requirements.dynamic_source_indices),
                    descriptor_ptr.add(self.requirements.dynamic_scales),
                    u32::try_from(self.requirements.dynamic_ext_param_count)
                        .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                    self.relation_z
                        .as_u32_ptr()
                        .cast::<CudaSecureField>()
                        .cast_const(),
                    self.relation_alpha_powers
                        .as_u32_ptr()
                        .cast::<CudaSecureField>()
                        .cast_const(),
                    u32::try_from(self.relation_alpha_powers.len_words() / SECURE_WORDS)
                        .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                    descriptor_ptr
                        .add(self.requirements.claimed_sum_pointers)
                        .cast::<*const CudaSecureField>(),
                    u32::try_from(self.requirements.claimed_sum_count)
                        .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                    stream,
                )
            })?;
        }
        if self.requirements.mode != CompositionLaunchMode::Wave {
            unsafe {
                context.memset_async(
                    self.accumulators.as_void_ptr(),
                    0,
                    self.accumulators.len_bytes(),
                )?;
            }
        }
        let total_constraints = u32::try_from(self.requirements.total_constraints)
            .map_err(|_| PreparedCompositionError::SizeOverflow)?;
        check_status("composition_generate_descending_powers", unsafe {
            raw::stwo_composition_generate_descending_powers_on(
                self.random_coefficient
                    .as_u32_ptr()
                    .cast::<CudaSecureField>(),
                self.random_coefficient_powers
                    .as_u32_ptr()
                    .cast::<CudaSecureField>(),
                total_constraints,
                stream,
            )
        })?;

        if self.requirements.mode == CompositionLaunchMode::Wave {
            for (wave_index, wave) in self.waves.iter().enumerate() {
                self.enqueue_wave(wave_index, wave, stream)?;
            }
        } else if self.lane_components.iter().all(|lane| lane.is_empty()) {
            // Serial topology: identical call sequence to the historical
            // launch path — plan order on the main stream, no fork/join.
            for &component in &self.requirements.serial_components {
                self.enqueue_component(component, stream)?;
            }
        } else {
            // Wide topology: fork the active component lanes off the main
            // stream (recording the graph dependency on the prelude above),
            // enqueue each lane's small groups, run the large components on
            // the main stream so they overlap the lanes, then join every
            // forked lane before the accumulator lift below. Lanes are always
            // rejoined, even on a mid-enqueue error.
            let active_lanes = self
                .lane_components
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
                        first_error = Some(PreparedCompositionError::from(error));
                        break;
                    }
                }
            }
            if first_error.is_none() {
                'lanes: for &(lane, launch) in &forked {
                    for &component in &self.lane_components[lane] {
                        if let Err(error) =
                            self.enqueue_component(component, launch.stream_raw().as_ptr())
                        {
                            first_error = Some(error);
                            break 'lanes;
                        }
                    }
                }
            }
            if first_error.is_none() {
                for &component in &self.requirements.serial_components {
                    if let Err(error) = self.enqueue_component(component, stream) {
                        first_error = Some(error);
                        break;
                    }
                }
            }
            for (lane, _) in forked {
                if let Err(error) = context.join_lane(lane) {
                    if first_error.is_none() {
                        first_error = Some(PreparedCompositionError::from(error));
                    }
                }
            }
            if let Some(error) = first_error {
                return Err(error);
            }
        }

        for pair in self.requirements.accumulators.windows(2) {
            let previous = pair[0];
            let current = pair[1];
            check_status("composition_lift_accumulate", unsafe {
                raw::stwo_composition_lift_accumulate_on(
                    self.accumulators.as_u32_ptr().add(previous.offset_words),
                    previous.log_size,
                    self.accumulators.as_u32_ptr().add(current.offset_words),
                    current.log_size,
                    stream,
                )
            })?;
        }

        let max_accumulator = self
            .requirements
            .accumulators
            .last()
            .expect("non-empty plan has an accumulator");
        check_status("composition_interpolate", unsafe {
            raw::stwo_ntt_b2n_columns_on(
                self.descriptors
                    .as_u32_ptr()
                    .add(self.requirements.final_coordinate_pointers)
                    .cast::<*mut u32>(),
                max_accumulator.log_size,
                SECURE_COORDINATES as u32,
                self.inverse_twiddles.as_u32_ptr(),
                u32::try_from(self.inverse_twiddles.len_words())
                    .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                1u32 << (max_accumulator.log_size - 1),
                stream,
            )
        })?;

        let full_rows = pow2(max_accumulator.log_size)?;
        let half_rows = self.requirements.output_coefficient_words;
        let bytes = half_rows
            .checked_mul(WORD_BYTES)
            .ok_or(PreparedCompositionError::SizeOverflow)?;
        let max_coordinates = unsafe {
            self.accumulators
                .as_u32_ptr()
                .add(max_accumulator.offset_words)
        };
        for coordinate in 0..SECURE_COORDINATES {
            let source = unsafe { max_coordinates.add(coordinate * full_rows) };
            unsafe {
                context.memcpy_d2d_async(
                    self.composition_coefficients[coordinate].as_void_ptr(),
                    source.cast::<c_void>(),
                    bytes,
                )?;
                context.memcpy_d2d_async(
                    self.composition_coefficients[SECURE_COORDINATES + coordinate].as_void_ptr(),
                    source.add(half_rows).cast::<c_void>(),
                    bytes,
                )?;
            }
        }
        Ok(())
    }

    fn enqueue_wave(
        &self,
        wave_index: usize,
        wave: &PreparedWave,
        stream: *mut c_void,
    ) -> Result<(), PreparedCompositionError> {
        let row_count = wave.row_count as usize;
        let accumulator = unsafe {
            self.accumulators
                .as_u32_ptr()
                .add(wave.accumulator_offset_words)
        };
        let launched = unsafe {
            raw::stwo_cuda_jit_eval_composition_wave_on(
                core::ptr::null(),
                wave.name.as_ptr(),
                wave.cache_key,
                self.descriptors
                    .as_u32_ptr()
                    .add(wave.descriptor_offset_words)
                    .cast::<raw::CudaCompositionWavePart>(),
                self.random_coefficient_powers.as_u32_ptr(),
                accumulator,
                accumulator.add(row_count),
                accumulator.add(2 * row_count),
                accumulator.add(3 * row_count),
                wave.row_count,
                stream,
            )
        };
        if !launched {
            return Err(PreparedCompositionError::CompositionWaveLaunchMiss {
                wave: wave_index,
                cache_key: wave.cache_key,
            });
        }
        Ok(())
    }

    /// Enqueue one component's trace LDE and constraint-eval kernels on
    /// `stream`. Kernel arguments are identical regardless of the stream (the
    /// per-component evaluation pointers already carry the mode's tile
    /// offsets), so serial and wide launches differ only in stream placement.
    fn enqueue_component(
        &self,
        component_index: usize,
        stream: *mut c_void,
    ) -> Result<(), PreparedCompositionError> {
        let descriptor_ptr = self.descriptors.as_u32_ptr();
        let requirement = &self.requirements.components[component_index];
        let component = &self.components[component_index];
        let twiddle_words = u32::try_from(self.forward_twiddles.len_words())
            .map_err(|_| PreparedCompositionError::SizeOverflow)?;
        if component.fallback_count != 0 {
            check_status("composition_trace_lde", unsafe {
                raw::stwo_lde_n2b_columns_on(
                    descriptor_ptr
                        .add(component.fallback_coefficient_pointers)
                        .cast::<*const u32>(),
                    descriptor_ptr.add(component.fallback_coefficient_sizes),
                    descriptor_ptr
                        .add(component.fallback_evaluation_pointers)
                        .cast::<*mut u32>(),
                    component.evaluation_log_size,
                    component.fallback_count,
                    self.forward_twiddles.as_u32_ptr(),
                    twiddle_words,
                    1u32 << (component.evaluation_log_size - 1),
                    stream,
                )
            })?;
        }

        let row_count = component.row_count as usize;
        let accumulator = unsafe {
            self.accumulators
                .as_u32_ptr()
                .add(component.accumulator_offset_words)
        };
        for (kernel_index, kernel) in component.kernels.iter().enumerate() {
            let rc_base = requirement
                .random_coefficient_offset
                .checked_add(kernel.rc_base as usize)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or(PreparedCompositionError::SizeOverflow)?;
            let launched = unsafe {
                raw::stwo_cuda_jit_eval_fused_on(
                    kernel.source.as_ptr(),
                    kernel.name.as_ptr(),
                    kernel.cache_key,
                    descriptor_ptr.add(component.evaluation_pointers),
                    descriptor_ptr.add(component.interaction_offsets),
                    descriptor_ptr.add(component.base_params),
                    component.ext_params,
                    self.random_coefficient_powers.as_u32_ptr(),
                    descriptor_ptr.add(component.denominator_inverses),
                    accumulator,
                    accumulator.add(row_count),
                    accumulator.add(2 * row_count),
                    accumulator.add(3 * row_count),
                    component.row_count,
                    component.trace_log_size,
                    rc_base,
                    false,
                    stream,
                )
            };
            if !launched {
                return Err(PreparedCompositionError::KernelLaunchMiss {
                    component: component_index,
                    kernel: kernel_index,
                    cache_key: kernel.cache_key,
                });
            }
        }
        Ok(())
    }

    pub fn requirements(&self) -> &CompositionWorkspaceRequirements {
        &self.requirements
    }

    pub fn composition_coefficients(&self) -> [ArenaSlice; SPLIT_COORDINATES] {
        self.composition_coefficients
    }
}

fn prepare_aot_kernel(
    component_index: usize,
    kernel_index: usize,
    kernel: &CompositionKernelPart,
) -> Result<PreparedKernel, PreparedCompositionError> {
    let source = CString::new(kernel.source.as_bytes()).map_err(|_| {
        PreparedCompositionError::KernelSourceContainsNul {
            component: component_index,
            kernel: kernel_index,
        }
    })?;
    let name = CString::new(kernel.kernel_name.as_bytes()).map_err(|_| {
        PreparedCompositionError::KernelNameContainsNul {
            component: component_index,
            kernel: kernel_index,
        }
    })?;
    let found = unsafe {
        raw::stwo_cuda_jit_precompile(source.as_ptr(), name.as_ptr(), kernel.cache_key, false)
    };
    if !found {
        return Err(PreparedCompositionError::AotKernelMiss {
            component: component_index,
            kernel: kernel_index,
            cache_key: kernel.cache_key,
        });
    }
    Ok(PreparedKernel {
        source,
        name,
        cache_key: kernel.cache_key,
        rc_base: kernel.rc_base,
    })
}

fn prepare_aot_wave(
    wave_index: usize,
    wave: &CompositionWaveKernelPlan,
    requirements: &CompositionWaveRequirements,
) -> Result<PreparedWave, PreparedCompositionError> {
    let name = CString::new(wave.kernel_name.as_bytes())
        .map_err(|_| PreparedCompositionError::CompositionWaveNameContainsNul(wave_index))?;
    let found = unsafe {
        raw::stwo_cuda_jit_precompile(core::ptr::null(), name.as_ptr(), wave.cache_key, false)
    };
    if !found {
        return Err(PreparedCompositionError::CompositionWaveAotMiss {
            wave: wave_index,
            cache_key: wave.cache_key,
        });
    }
    Ok(PreparedWave {
        name,
        cache_key: wave.cache_key,
        descriptor_offset_words: requirements.descriptor_offset_words,
        accumulator_offset_words: requirements.accumulator_offset_words,
        row_count: u32::try_from(requirements.row_count)
            .map_err(|_| PreparedCompositionError::SizeOverflow)?,
    })
}

fn write_wave_part_descriptor(
    words: &mut [u32],
    base: usize,
    descriptor_ptr: *mut u32,
    component: &PreparedComponent,
    proof_global_rc_base: u32,
) {
    let field = |offset: usize| base + offset / WORD_BYTES;
    write_pointer(
        words,
        field(core::mem::offset_of!(
            raw::CudaCompositionWavePart,
            trace_cols
        )),
        unsafe { descriptor_ptr.add(component.evaluation_pointers) },
    );
    write_pointer(
        words,
        field(core::mem::offset_of!(
            raw::CudaCompositionWavePart,
            interaction_offsets
        )),
        unsafe { descriptor_ptr.add(component.interaction_offsets) },
    );
    write_pointer(
        words,
        field(core::mem::offset_of!(
            raw::CudaCompositionWavePart,
            base_params
        )),
        unsafe { descriptor_ptr.add(component.base_params) },
    );
    write_pointer(
        words,
        field(core::mem::offset_of!(
            raw::CudaCompositionWavePart,
            ext_params
        )),
        component.ext_params.cast_mut(),
    );
    write_pointer(
        words,
        field(core::mem::offset_of!(
            raw::CudaCompositionWavePart,
            denom_inv
        )),
        unsafe { descriptor_ptr.add(component.denominator_inverses) },
    );
    words[field(core::mem::offset_of!(
        raw::CudaCompositionWavePart,
        log_n_rows
    ))] = component.trace_log_size;
    words[field(core::mem::offset_of!(raw::CudaCompositionWavePart, rc_base))] =
        proof_global_rc_base;
}

fn bind_slot(
    arena: &DeviceArena,
    requirement: CompositionArenaSlotRequirement,
) -> Result<ArenaSlice, PreparedCompositionError> {
    let slice = arena.bind(requirement.id)?;
    if slice.len_words() < requirement.len_words {
        return Err(PreparedCompositionError::SlotTooSmall {
            slot: requirement.id,
            required_words: requirement.len_words,
            actual_words: slice.len_words(),
        });
    }
    if (slice.as_u32_ptr() as usize) % (requirement.alignment_words * WORD_BYTES) != 0 {
        return Err(PreparedCompositionError::Arena(ArenaError::Misaligned(
            requirement.id,
        )));
    }
    // Pooled slots may be larger than any single logical buffer; expose only
    // the logical extent so memsets and kernel sizes never see the surplus.
    Ok(slice.truncated(requirement.len_words))
}

fn bind_minimum(
    arena: &DeviceArena,
    id: ArenaSlotId,
    required_words: usize,
) -> Result<ArenaSlice, PreparedCompositionError> {
    let slice = arena.bind(id)?;
    if slice.len_words() < required_words {
        return Err(PreparedCompositionError::SlotTooSmall {
            slot: id,
            required_words,
            actual_words: slice.len_words(),
        });
    }
    // Pooled slots may be larger than any single logical buffer; expose only
    // the logical extent so twiddle sizes and kernel extents derived from
    // `len_words()` are the logical requirement, never the pooled maximum.
    Ok(slice.truncated(required_words))
}

/// Validate a caller-provided logical slice against a minimum extent. The
/// slice is returned as provided: its length is the caller's logical extent,
/// already truncated at bind time, and downstream counts derive from it.
fn require_input_min(
    arena: &DeviceArena,
    slice: ArenaSlice,
    required_words: usize,
) -> Result<ArenaSlice, PreparedCompositionError> {
    if slice.len_words() < required_words {
        return Err(PreparedCompositionError::SlotTooSmall {
            slot: slice.id(),
            required_words,
            actual_words: slice.len_words(),
        });
    }
    if !slice.belongs_to(arena.context()) {
        return Err(PreparedCompositionError::ContextMismatch(slice.id()));
    }
    let rebound = arena.bind(slice.id())?;
    if rebound.len_words() < slice.len_words() {
        return Err(PreparedCompositionError::SlotTooSmall {
            slot: slice.id(),
            required_words: slice.len_words(),
            actual_words: rebound.len_words(),
        });
    }
    Ok(rebound.truncated(slice.len_words()))
}

fn write_pointer(words: &mut [u32], offset_words: usize, pointer: *mut u32) {
    let bytes = (pointer as usize).to_ne_bytes();
    for (word, chunk) in words[offset_words..offset_words + POINTER_WORDS]
        .iter_mut()
        .zip(bytes.chunks(WORD_BYTES))
    {
        let mut encoded = [0u8; WORD_BYTES];
        encoded[..chunk.len()].copy_from_slice(chunk);
        *word = u32::from_ne_bytes(encoded);
    }
}

#[allow(clippy::too_many_arguments)]
fn write_dynamic_ext_param(
    words: &mut [u32],
    requirements: &CompositionWorkspaceRequirements,
    index: usize,
    destination: *mut u32,
    source_kind: u32,
    source_index: u32,
    scale: u32,
) {
    write_pointer(
        words,
        requirements.dynamic_destination_pointers + index * POINTER_WORDS,
        destination,
    );
    words[requirements.dynamic_source_kinds + index] = source_kind;
    words[requirements.dynamic_source_indices + index] = source_index;
    words[requirements.dynamic_scales + index] = scale;
}

fn secure_words(value: stwo::core::fields::qm31::SecureField) -> [u32; SECURE_WORDS] {
    value.to_m31_array().map(|coordinate| coordinate.0)
}

fn check_status(operation: &'static str, status: i32) -> Result<(), PreparedCompositionError> {
    if status == 0 {
        Ok(())
    } else {
        Err(PreparedCompositionError::CudaStatus { operation, status })
    }
}

fn pow2(log_size: u32) -> Result<usize, PreparedCompositionError> {
    1usize
        .checked_shl(log_size)
        .ok_or(PreparedCompositionError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::BaseField;
    use stwo::core::pcs::TreeSubspan;

    use super::*;
    use crate::arena_plan::{BufferLifetime, BufferPurpose, ProofEpoch};
    use crate::direct_composition_retention::{
        DirectCompositionBinding, DirectCompositionColumn, DirectCompositionRetentionPlan,
    };

    fn kernel(rc_base: u32) -> CompositionKernelPart {
        CompositionKernelPart {
            kernel_name: "kernel".to_owned(),
            cache_key: 7,
            semantic_hash: 9,
            source: "extern \"C\" __global__ void kernel() {}".to_owned(),
            rc_base,
        }
    }

    fn component(
        name: &'static str,
        trace_log: u32,
        eval_log: u32,
        constraints: usize,
        random_offset: usize,
        preprocessed: Vec<usize>,
        base: core::ops::Range<usize>,
        interaction: core::ops::Range<usize>,
    ) -> CompositionComponentPlan {
        CompositionComponentPlan {
            component: name,
            instance: 0,
            trace_locations: vec![
                TreeSubspan {
                    tree_index: 0,
                    col_start: 0,
                    col_end: 0,
                },
                TreeSubspan {
                    tree_index: 1,
                    col_start: base.start,
                    col_end: base.end,
                },
                TreeSubspan {
                    tree_index: 2,
                    col_start: interaction.start,
                    col_end: interaction.end,
                },
            ],
            preprocessed_column_indices: preprocessed,
            trace_log_size: trace_log,
            evaluation_log_size: eval_log,
            n_constraints: constraints,
            random_coefficient_offset: random_offset,
            denominator_inverses: vec![BaseField::from(1); 1usize << (eval_log - trace_log)],
            base_param_values: Vec::new(),
            ext_param_values: Vec::new(),
            ext_param_sources: Vec::new(),
            kernels: vec![kernel(0)],
        }
    }

    fn trace() -> CompositionTraceTopology {
        let tree = |base: u32, logs: &[u32]| {
            logs.iter()
                .enumerate()
                .map(|(index, &log_size)| CompositionCoefficientSource {
                    slot: ArenaSlotId(base + index as u32),
                    log_size,
                })
                .collect()
        };
        CompositionTraceTopology {
            trees: vec![
                tree(100, &[4, 5, 6]),
                tree(200, &[4, 4, 5, 5]),
                tree(300, &[4, 4, 5]),
            ],
        }
    }

    fn one_component_plan(preprocessed: Vec<usize>) -> CompositionPlan {
        CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 2,
            max_evaluation_log_size: 8,
            components: vec![component("a", 5, 8, 2, 0, preprocessed, 1..4, 0..2)],
            wave_kernels: Vec::new(),
        }
    }

    #[test]
    fn base_parameters_use_immutable_component_descriptor_storage() {
        let mut plan = one_component_plan(vec![0]);
        let without = composition_workspace_requirements(&plan, &trace()).unwrap();
        assert_eq!(without.components[0].base_param_words, 0);
        assert_eq!(
            without.component_descriptors[0].base_params,
            without.zero_words
        );

        plan.components[0].base_param_values = vec![
            BaseField::from_u32_unchecked(7),
            BaseField::from_u32_unchecked(11),
        ];
        let with = composition_workspace_requirements(&plan, &trace()).unwrap();
        assert_eq!(with.components[0].base_param_words, 2);
        let offset = with.component_descriptors[0].base_params;
        assert_ne!(offset, with.zero_words);
        assert!(offset + 2 <= with.descriptor_words);
        assert!(with.descriptor_words >= without.descriptor_words + 2);
    }

    fn opened_source(tree: usize, column: usize) -> OpenedColumnSource {
        match tree {
            0 => OpenedColumnSource::Preprocessed {
                ordinal: column as u32,
            },
            1 => OpenedColumnSource::Trace {
                component: "test",
                part: stwo_cairo_prover::witness::proof_shape::TracePartId::Main,
                purpose: BufferPurpose::BaseCoefficients,
                ordinal: column as u32,
            },
            2 => OpenedColumnSource::Trace {
                component: "test",
                part: stwo_cairo_prover::witness::proof_shape::TracePartId::Main,
                purpose: BufferPurpose::InteractionCoefficients,
                ordinal: column as u32,
            },
            _ => unreachable!(),
        }
    }

    fn retention_plan(
        requirements: &CompositionWorkspaceRequirements,
        direct_consumers: &[usize],
    ) -> DirectCompositionRetentionPlan {
        let mut columns = Vec::<DirectCompositionColumn>::new();
        let mut bindings = Vec::new();
        let mut direct_bitmap = vec![
            0u64;
            requirements
                .components
                .iter()
                .map(|component| component.sources.len())
                .sum::<usize>()
                .div_ceil(64)
        ];
        let mut consumer = 0usize;
        for component in &requirements.components {
            for source in &component.sources {
                let tree = composition_tree(source.tree).unwrap();
                let source_identity = opened_source(source.tree, source.column);
                let column = columns
                    .iter()
                    .position(|column| column.tree == tree && column.proof_column == source.column)
                    .unwrap_or_else(|| {
                        let direct = direct_consumers.contains(&consumer);
                        columns.push(DirectCompositionColumn {
                            source: source_identity,
                            tree,
                            proof_column: source.column,
                            group: 0,
                            column_in_group: source.column,
                            canonical_column: source.column,
                            coefficient_log_size: source.source.log_size,
                            evaluation_log_size: if direct {
                                component.evaluation_log_size
                            } else {
                                source.source.log_size + 1
                            },
                            lifetime: BufferLifetime::new(
                                match tree {
                                    CommitmentTreeId::Preprocessed => ProofEpoch::Ingest,
                                    CommitmentTreeId::Base => ProofEpoch::BaseCommit,
                                    CommitmentTreeId::Interaction => ProofEpoch::InteractionCommit,
                                    CommitmentTreeId::Composition | CommitmentTreeId::Fri(_) => {
                                        unreachable!()
                                    }
                                },
                                ProofEpoch::Composition,
                            )
                            .unwrap(),
                        });
                        columns.len() - 1
                    });
                let direct = columns[column].evaluation_log_size == component.evaluation_log_size;
                if direct {
                    direct_bitmap[consumer / 64] |= 1 << (consumer % 64);
                }
                bindings.push(DirectCompositionBinding {
                    consumer,
                    column,
                    consumer_evaluation_log_size: component.evaluation_log_size,
                    direct,
                });
                consumer += 1;
            }
        }
        let mut plan = DirectCompositionRetentionPlan {
            columns,
            bindings,
            direct_bitmap,
            buckets: Vec::new(),
            direct_column_count: 0,
            direct_bytes: 0,
            cache_key: 0,
        };
        plan.direct_column_count = plan
            .bindings
            .iter()
            .filter(|binding| binding.direct)
            .map(|binding| binding.column)
            .collect::<BTreeSet<_>>()
            .len();
        plan.cache_key = direct_composition_plan_key(&plan);
        plan
    }

    #[test]
    fn direct_retention_all_fallback_mixed_and_all_direct_preserve_evaluator_order() {
        for mode in [CompositionLaunchMode::Serial, CompositionLaunchMode::Wide] {
            let plan = one_component_plan(vec![2, 0]);
            let legacy =
                composition_workspace_requirements_with_mode(&plan, &trace(), mode).unwrap();
            let explicit_off =
                composition_workspace_requirements_with_retention(&plan, &trace(), mode, None)
                    .unwrap();
            assert_eq!(legacy, explicit_off, "flags-off requirements must be exact");
            assert_eq!(legacy.direct_retention_plan_key, None);
            assert!(legacy.direct_retention_bitmap.is_empty());

            let source_count = legacy.components[0].sources.len();
            for direct_consumers in [
                Vec::new(),
                vec![0, 2, source_count - 1],
                (0..source_count).collect::<Vec<_>>(),
            ] {
                let retention = retention_plan(&legacy, &direct_consumers);
                let requirements = composition_workspace_requirements_with_retention(
                    &plan,
                    &trace(),
                    mode,
                    Some(&retention),
                )
                .unwrap();
                let component = &requirements.components[0];
                assert_eq!(component.source_retention.len(), source_count);
                assert_eq!(
                    component
                        .source_retention
                        .iter()
                        .map(|retention| (retention.tree, retention.proof_column))
                        .collect::<Vec<_>>(),
                    component
                        .sources
                        .iter()
                        .map(|source| (composition_tree(source.tree).unwrap(), source.column))
                        .collect::<Vec<_>>()
                );
                let expected_fallbacks = source_count - direct_consumers.len();
                assert_eq!(component.fallback_count, expected_fallbacks);
                assert_eq!(
                    component
                        .source_retention
                        .iter()
                        .filter_map(|retention| retention.fallback_ordinal)
                        .collect::<Vec<_>>(),
                    (0..expected_fallbacks).collect::<Vec<_>>()
                );
                assert_eq!(
                    requirements.lde_tile_words,
                    (expected_fallbacks * (1usize << component.evaluation_log_size)).max(1)
                );
                assert_eq!(
                    requirements.direct_retention_plan_key,
                    Some(retention.cache_key)
                );
                assert_eq!(
                    requirements.direct_retention_bitmap,
                    retention.direct_bitmap
                );
                let expected_descriptor_delta = expected_fallbacks * (2 * POINTER_WORDS + 1)
                    + (POINTER_WORDS - expected_fallbacks % POINTER_WORDS) % POINTER_WORDS;
                assert_eq!(
                    requirements.descriptor_words,
                    legacy.descriptor_words + expected_descriptor_delta
                );
            }
        }
    }

    #[test]
    fn replacement_wave_requires_exact_all_direct_ownership_and_global_spans() {
        let mut plan = CompositionPlan {
            max_kernel_instrs: 192,
            total_constraints: 5,
            max_evaluation_log_size: 8,
            components: vec![
                component("a", 5, 8, 2, 0, vec![0], 0..1, 0..1),
                component("b", 5, 8, 3, 2, vec![1], 1..2, 1..2),
            ],
            wave_kernels: Vec::new(),
        };
        let identities = vec![
            aot::CompositionWaveKernelPartIdentity {
                semantic_hash: 9,
                coefficient_start: 0,
                coefficient_end: 2,
            },
            aot::CompositionWaveKernelPartIdentity {
                semantic_hash: 9,
                coefficient_start: 2,
                coefficient_end: 5,
            },
        ];
        let wave_identity = aot::composition_wave_kernel_identity(8, &identities).unwrap();
        plan.wave_kernels.push(CompositionWaveKernelPlan {
            evaluation_log_size: 8,
            parts: identities,
            kernel_name: wave_identity.kernel_name,
            cache_key: wave_identity.cache_key,
            semantic_hash: wave_identity.semantic_hash,
            // The strict warm binder resolves by name/key and must not require
            // a retained cold CUDA TU.
            source: String::new(),
        });

        let policy = crate::protocol_plan::ProtocolPlanPolicy::replacement_v1(1, 192);
        assert_eq!(policy.composition_launch_mode, CompositionLaunchMode::Wave);
        let source_requirements = composition_workspace_requirements_with_mode(
            &plan,
            &trace(),
            CompositionLaunchMode::Serial,
        )
        .unwrap();
        let source_count = source_requirements
            .components
            .iter()
            .map(|component| component.sources.len())
            .sum::<usize>();
        let all_direct =
            retention_plan(&source_requirements, &(0..source_count).collect::<Vec<_>>());
        let wave = composition_workspace_requirements_with_retention(
            &plan,
            &trace(),
            policy.composition_launch_mode,
            Some(&all_direct),
        )
        .unwrap();
        assert!(wave
            .components
            .iter()
            .all(|component| component.fallback_count == 0));
        assert_eq!(wave.waves.len(), 1);
        assert_eq!(wave.waves[0].parts.len(), 2);
        assert_eq!(wave.waves[0].parts[1].identity.coefficient_start, 2);
        assert_eq!(wave.waves[0].parts[1].identity.coefficient_end, 5);

        let one_fallback =
            retention_plan(&source_requirements, &(1..source_count).collect::<Vec<_>>());
        assert!(matches!(
            composition_workspace_requirements_with_retention(
                &plan,
                &trace(),
                CompositionLaunchMode::Wave,
                Some(&one_fallback),
            ),
            Err(PreparedCompositionError::CompositionWaveRequiresAllDirect { .. })
        ));

        plan.wave_kernels[0].parts[1].coefficient_start += 1;
        assert_eq!(
            composition_workspace_requirements_with_retention(
                &plan,
                &trace(),
                CompositionLaunchMode::Wave,
                Some(&all_direct),
            )
            .unwrap_err(),
            PreparedCompositionError::CompositionWavePlanDrift("part identity/order")
        );
    }

    #[test]
    fn wave_descriptor_writes_the_proof_global_coefficient_start() {
        let mut words = vec![0u32; WAVE_PART_WORDS + 4];
        let descriptor_ptr = 0x1000usize as *mut u32;
        let component = PreparedComponent {
            evaluation_pointers: 2,
            fallback_coefficient_pointers: 0,
            fallback_coefficient_sizes: 0,
            fallback_evaluation_pointers: 0,
            interaction_offsets: 4,
            denominator_inverses: 8,
            base_params: 12,
            ext_params: 0x2000usize as *const u32,
            accumulator_offset_words: 0,
            trace_log_size: 17,
            evaluation_log_size: 19,
            row_count: 1 << 19,
            fallback_count: 0,
            kernels: Vec::new(),
        };
        write_wave_part_descriptor(&mut words, 0, descriptor_ptr, &component, 73);
        assert_eq!(
            words[core::mem::offset_of!(raw::CudaCompositionWavePart, log_n_rows) / WORD_BYTES],
            17
        );
        assert_eq!(
            words[core::mem::offset_of!(raw::CudaCompositionWavePart, rc_base) / WORD_BYTES],
            73
        );
    }

    #[test]
    fn direct_retention_interleaving_and_duplicate_occurrences_are_sealed() {
        let plan = one_component_plan(vec![0, 0, 2]);
        let legacy = composition_workspace_requirements_with_mode(
            &plan,
            &trace(),
            CompositionLaunchMode::Serial,
        )
        .unwrap();
        let retention = retention_plan(&legacy, &[0, 1, 3, 6]);
        assert_eq!(retention.bindings[0].column, retention.bindings[1].column);
        let requirements = composition_workspace_requirements_with_retention(
            &plan,
            &trace(),
            CompositionLaunchMode::Serial,
            Some(&retention),
        )
        .unwrap();
        let metadata = &requirements.components[0].source_retention;
        assert_eq!(metadata[0].plan_column, metadata[1].plan_column);
        assert!(metadata[0].direct && metadata[1].direct);
        assert_eq!(
            metadata
                .iter()
                .filter_map(|entry| entry.fallback_ordinal)
                .collect::<Vec<_>>(),
            (0..requirements.components[0].fallback_count).collect::<Vec<_>>()
        );

        let mut ambiguous = retention.clone();
        ambiguous.columns.push(ambiguous.columns[0]);
        ambiguous.bindings[1].column = ambiguous.columns.len() - 1;
        ambiguous.cache_key = direct_composition_plan_key(&ambiguous);
        assert_eq!(
            composition_workspace_requirements_with_retention(
                &plan,
                &trace(),
                CompositionLaunchMode::Serial,
                Some(&ambiguous),
            )
            .unwrap_err(),
            PreparedCompositionError::DirectRetentionPlanDrift(
                "logical source has multiple plan columns"
            )
        );
    }

    #[test]
    fn direct_retention_rejects_cache_bitmap_order_source_tree_proof_and_log_drift() {
        let plan = one_component_plan(vec![2, 0]);
        let legacy = composition_workspace_requirements(&plan, &trace()).unwrap();
        let retention = retention_plan(&legacy, &[0, 2]);

        let mut drift = retention.clone();
        drift.cache_key ^= 1;
        assert_eq!(
            composition_workspace_requirements_with_retention(
                &plan,
                &trace(),
                CompositionLaunchMode::Serial,
                Some(&drift),
            )
            .unwrap_err(),
            PreparedCompositionError::DirectRetentionPlanKeyDrift
        );

        let mut mutations: Vec<(&str, Box<dyn Fn(&mut DirectCompositionRetentionPlan)>)> = vec![
            (
                "consumer order",
                Box::new(|plan| plan.bindings[0].consumer = 1),
            ),
            (
                "direct bitmap or native log",
                Box::new(|plan| plan.direct_bitmap[0] ^= 1),
            ),
            (
                "source tree",
                Box::new(|plan| plan.columns[0].tree = CommitmentTreeId::Base),
            ),
            (
                "proof column",
                Box::new(|plan| plan.columns[0].proof_column += 1),
            ),
            (
                "coefficient log",
                Box::new(|plan| plan.columns[0].coefficient_log_size += 1),
            ),
            (
                "source identity tree",
                Box::new(|plan| {
                    plan.columns[0].source = OpenedColumnSource::Trace {
                        component: "test",
                        part: stwo_cairo_prover::witness::proof_shape::TracePartId::Main,
                        purpose: BufferPurpose::BaseCoefficients,
                        ordinal: 0,
                    }
                }),
            ),
        ];
        for (expected, mutate) in mutations.drain(..) {
            let mut drift = retention.clone();
            mutate(&mut drift);
            drift.cache_key = direct_composition_plan_key(&drift);
            assert_eq!(
                composition_workspace_requirements_with_retention(
                    &plan,
                    &trace(),
                    CompositionLaunchMode::Serial,
                    Some(&drift),
                )
                .unwrap_err(),
                PreparedCompositionError::DirectRetentionPlanDrift(expected)
            );
        }

        let mut tail_drift = retention.clone();
        let occurrence_count = tail_drift.bindings.len();
        tail_drift.direct_bitmap[occurrence_count / 64] |= 1 << (occurrence_count % 64);
        tail_drift.cache_key = direct_composition_plan_key(&tail_drift);
        assert_eq!(
            composition_workspace_requirements_with_retention(
                &plan,
                &trace(),
                CompositionLaunchMode::Serial,
                Some(&tail_drift),
            )
            .unwrap_err(),
            PreparedCompositionError::DirectRetentionPlanDrift("bitmap tail bits")
        );
    }

    #[test]
    fn topology_preserves_tree_abi_global_random_order_and_workspace_reuse() {
        let plan = CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 5,
            max_evaluation_log_size: 7,
            components: vec![
                component("a", 5, 7, 2, 0, vec![2, 0], 1..4, 0..2),
                component("b", 4, 6, 3, 2, vec![1], 0..2, 1..3),
            ],
            wave_kernels: Vec::new(),
        };
        let requirements = composition_workspace_requirements_with_mode(
            &plan,
            &trace(),
            CompositionLaunchMode::Serial,
        )
        .unwrap();
        assert_eq!(requirements.components[0].interaction_offsets, [0, 2, 5]);
        assert_eq!(
            requirements.components[0]
                .sources
                .iter()
                .map(|source| (source.tree, source.column))
                .collect::<Vec<_>>(),
            vec![(0, 2), (0, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]
        );
        assert_eq!(
            requirements.lde_tile_words,
            7 * (1usize << 7),
            "one tile is reused and sized for the largest component footprint"
        );
        assert_eq!(
            requirements
                .accumulators
                .iter()
                .map(|accumulator| accumulator.log_size)
                .collect::<Vec<_>>(),
            vec![6, 7]
        );
        assert_eq!(requirements.random_power_words, 5 * SECURE_WORDS);
        assert_eq!(requirements.output_coefficient_words, 1 << 6);
        assert_eq!(requirements.serial_components, vec![0, 1]);
        assert!(requirements.wide_groups.is_empty());
        assert!(requirements
            .components
            .iter()
            .all(|component| component.lde_tile_offset_words == 0));
    }

    #[test]
    fn wide_mode_partitions_small_groups_with_disjoint_private_tile_regions() {
        // Eval logs 7, 6, 7: all small (<= COMPOSITION_WIDE_SMALL_MAX_EVALUATION_LOG),
        // so two groups form (log 6, log 7) and no serial component remains.
        let plan = CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 6,
            max_evaluation_log_size: 7,
            components: vec![
                component("a", 5, 7, 2, 0, vec![2, 0], 1..4, 0..2),
                component("b", 4, 6, 3, 2, vec![1], 0..2, 1..3),
                component("c", 5, 7, 1, 5, vec![0], 0..1, 0..1),
            ],
            wave_kernels: Vec::new(),
        };
        let serial = composition_workspace_requirements_with_mode(
            &plan,
            &trace(),
            CompositionLaunchMode::Serial,
        )
        .unwrap();
        let wide = composition_workspace_requirements_with_mode(
            &plan,
            &trace(),
            CompositionLaunchMode::Wide,
        )
        .unwrap();

        // Mode changes only tile layout and scheduling metadata.
        assert_eq!(wide.components.len(), serial.components.len());
        for (wide_component, serial_component) in wide.components.iter().zip(&serial.components) {
            let mut normalized = wide_component.clone();
            normalized.lde_tile_offset_words = 0;
            assert_eq!(&normalized, serial_component);
        }
        assert_eq!(wide.accumulators, serial.accumulators);
        assert_eq!(wide.descriptor_words, serial.descriptor_words);

        assert!(wide.serial_components.is_empty());
        assert_eq!(wide.wide_groups.len(), 2);
        let group_6 = &wide.wide_groups[0];
        let group_7 = &wide.wide_groups[1];
        assert_eq!(group_6.evaluation_log_size, 6);
        assert_eq!(group_6.members, vec![1]);
        assert_eq!(group_7.evaluation_log_size, 7);
        assert_eq!(group_7.members, vec![0, 2], "members stay in plan order");

        // Regions: group 6 first (BTreeMap ascending), sized by its largest
        // member; group 7 after it; every member carries its group's offset.
        assert_eq!(group_6.tile_offset_words, 0);
        // Component b selects 1 preprocessed + 2 base + 2 interaction columns.
        assert_eq!(group_6.tile_len_words, 5 * (1 << 6));
        assert_eq!(group_7.tile_offset_words, group_6.tile_len_words);
        assert_eq!(group_7.tile_len_words, 7 * (1 << 7));
        assert_eq!(wide.components[1].lde_tile_offset_words, 0);
        assert_eq!(
            wide.components[0].lde_tile_offset_words,
            group_7.tile_offset_words
        );
        assert_eq!(
            wide.components[2].lde_tile_offset_words,
            group_7.tile_offset_words
        );
        assert_eq!(
            wide.lde_tile_words,
            group_6.tile_len_words + group_7.tile_len_words
        );
        assert!(wide.lde_tile_words >= serial.lde_tile_words);
    }

    #[test]
    fn wide_mode_keeps_large_components_serial_on_the_shared_leading_region() {
        // Eval log 20 > threshold: stays serial at offset zero; the small
        // log-7 group is placed after the large footprint.
        let plan = CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 3,
            max_evaluation_log_size: 20,
            components: vec![
                component("large", 19, 20, 2, 0, vec![0], 0..1, 0..1),
                component("small", 5, 7, 1, 2, vec![0], 0..1, 0..1),
            ],
            wave_kernels: Vec::new(),
        };
        let wide = composition_workspace_requirements_with_mode(
            &plan,
            &trace(),
            CompositionLaunchMode::Wide,
        )
        .unwrap();
        assert_eq!(wide.serial_components, vec![0]);
        assert_eq!(wide.components[0].lde_tile_offset_words, 0);
        let large_footprint = 3 * (1usize << 20);
        assert_eq!(wide.wide_groups.len(), 1);
        assert_eq!(wide.wide_groups[0].members, vec![1]);
        assert_eq!(wide.wide_groups[0].tile_offset_words, large_footprint);
        assert_eq!(wide.components[1].lde_tile_offset_words, large_footprint);
        assert_eq!(
            wide.lde_tile_words,
            large_footprint + wide.wide_groups[0].tile_len_words
        );
    }

    #[test]
    fn wide_group_packing_is_deterministic_and_balances_by_weight() {
        let group = |log: u32, members: Vec<usize>, weight: u64| CompositionWideGroup {
            evaluation_log_size: log,
            members,
            weight_words: weight,
            tile_offset_words: 0,
            tile_len_words: 0,
        };
        let groups = vec![
            group(6, vec![0], 10),
            group(7, vec![1], 40),
            group(8, vec![2], 30),
            group(9, vec![3], 10),
        ];
        let lanes = pack_composition_wide_groups(&groups, 2);
        // Heaviest (40 -> lane 0), then 30 -> lane 1, then the two 10s: the
        // log-6 group precedes the log-9 group (tie broken by eval log), so
        // it lands on lane 1 (load 30 < 40) and log-9 lands on lane 0.
        assert_eq!(lanes, vec![vec![1, 3], vec![2, 0]]);
        assert_eq!(
            lanes,
            pack_composition_wide_groups(&groups, 2),
            "packing must be a pure function of its inputs"
        );
        // One lane: everything serializes in weight order.
        assert_eq!(
            pack_composition_wide_groups(&groups, 1),
            vec![vec![1, 2, 0, 3]]
        );
    }

    #[test]
    fn slot_requirements_are_address_free_unique_and_exact() {
        let plan = CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 2,
            max_evaluation_log_size: 7,
            components: vec![component("a", 5, 7, 2, 0, vec![0], 0..1, 0..1)],
            wave_kernels: Vec::new(),
        };
        let requirements = composition_workspace_requirements(&plan, &trace()).unwrap();
        let slots = CompositionWorkspaceSlots {
            descriptors: ArenaSlotId(1),
            lde_tile: ArenaSlotId(2),
            accumulators: ArenaSlotId(3),
            random_coefficient_powers: ArenaSlotId(4),
            composition_coefficients: std::array::from_fn(|index| ArenaSlotId(5 + index as u32)),
        };
        let arena = requirements.arena_slot_requirements(&slots).unwrap();
        assert_eq!(arena.len(), 12);
        assert_eq!(arena[3].len_words, 2 * SECURE_WORDS);
        assert!(arena[4..].iter().all(|slot| slot.len_words == 1usize << 6));

        let mut duplicate = slots;
        duplicate.composition_coefficients[7] = duplicate.descriptors;
        assert_eq!(
            requirements
                .arena_slot_requirements(&duplicate)
                .unwrap_err(),
            PreparedCompositionError::DuplicateSlot(duplicate.descriptors)
        );
    }

    #[test]
    fn topology_fails_closed_on_truncating_lde_or_random_order_drift() {
        let mut too_large = trace();
        too_large.trees[1][0].log_size = 7;
        let plan = CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 1,
            max_evaluation_log_size: 7,
            components: vec![component("a", 5, 7, 1, 0, vec![0], 0..1, 0..1)],
            wave_kernels: Vec::new(),
        };
        assert!(matches!(
            composition_workspace_requirements(&plan, &too_large),
            Err(PreparedCompositionError::SourceCannotFitEvaluationDomain {
                tree: 1,
                column: 0,
                ..
            })
        ));

        let drifted = CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 1,
            max_evaluation_log_size: 7,
            components: vec![component("a", 5, 7, 1, 1, vec![0], 0..1, 0..1)],
            wave_kernels: Vec::new(),
        };
        assert_eq!(
            composition_workspace_requirements(&drifted, &trace()).unwrap_err(),
            PreparedCompositionError::RandomCoefficientOrder {
                component: 0,
                expected: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn lift_index_matches_domain_evaluation_accumulator_reference() {
        for previous_log in 2..7 {
            for current_log in previous_log + 1..9 {
                let ratio = current_log - previous_log;
                let previous_size = 1usize << previous_log;
                for index in 0..1usize << current_log {
                    let expected = (index >> (ratio + 1) << 1) + (index & 1);
                    assert!(expected < previous_size);
                    assert_eq!(expected & 1, index & 1);
                }
            }
        }
    }
}
