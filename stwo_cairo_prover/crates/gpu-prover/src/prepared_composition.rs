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

use crate::composition_plan::{
    CompositionComponentPlan, CompositionExtParamSource, CompositionKernelPart, CompositionPlan,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const SECURE_WORDS: usize = 4;
const SECURE_COORDINATES: usize = 4;
const SPLIT_COORDINATES: usize = 8;
const TRACE_TREES: usize = 3;
const POINTER_WORDS: usize = core::mem::size_of::<usize>().div_ceil(WORD_BYTES);

pub const COMPOSITION_POINTER_ALIGNMENT_WORDS: usize = core::mem::align_of::<usize>() / WORD_BYTES;

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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionDeviceInputs {
    pub random_coefficient: ArenaSlotId,
    pub forward_twiddles: ArenaSlotId,
    pub inverse_twiddles: ArenaSlotId,
    /// Stable challenge slices produced by [`PreparedRelationGraph`].
    pub relation_z: ArenaSlotId,
    pub relation_alpha_powers: ArenaSlotId,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionComponentRequirements {
    pub component: &'static str,
    pub instance: usize,
    pub trace_log_size: u32,
    pub evaluation_log_size: u32,
    pub row_count: usize,
    pub sources: Vec<CompositionSourceRef>,
    pub interaction_offsets: [u32; TRACE_TREES],
    pub denominator_words: usize,
    pub ext_param_words: usize,
    pub random_coefficient_offset: usize,
    pub accumulator_offset_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionAccumulatorRequirements {
    pub log_size: u32,
    pub offset_words: usize,
    pub len_words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ComponentDescriptorLayout {
    coefficient_pointers: usize,
    coefficient_sizes: usize,
    evaluation_pointers: usize,
    interaction_offsets: usize,
    denominator_inverses: usize,
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
    SourceAliasesWritableWorkspace(ArenaSlotId),
    InputAliasesWritableWorkspace(ArenaSlotId),
    ForwardInverseTwiddlesAlias(ArenaSlotId),
    RelationChallengeSourcesAlias(ArenaSlotId),
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

/// Compute exact workspace geometry and source selection without touching CUDA.
pub fn composition_workspace_requirements(
    plan: &CompositionPlan,
    trace: &CompositionTraceTopology,
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
    let mut lde_tile_words = 0usize;
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
        lde_tile_words = lde_tile_words.max(
            row_count
                .checked_mul(sources.len())
                .ok_or(PreparedCompositionError::SizeOverflow)?,
        );
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
            interaction_offsets,
            denominator_words: expected_denominators,
            ext_param_words: component
                .ext_param_values
                .len()
                .checked_mul(SECURE_WORDS)
                .ok_or(PreparedCompositionError::SizeOverflow)?,
            random_coefficient_offset: component.random_coefficient_offset,
            accumulator_offset_words: *accumulator_offsets
                .get(&component.evaluation_log_size)
                .expect("evaluation log was collected"),
        });
    }
    debug_assert_eq!(expected_random_offset, plan.total_constraints);

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
        component_descriptors.push(ComponentDescriptorLayout {
            coefficient_pointers: descriptor
                .take(pointer_words, COMPOSITION_POINTER_ALIGNMENT_WORDS)?,
            coefficient_sizes: descriptor.take(component.sources.len(), 1)?,
            evaluation_pointers: descriptor
                .take(pointer_words, COMPOSITION_POINTER_ALIGNMENT_WORDS)?,
            interaction_offsets: descriptor.take(TRACE_TREES, 1)?,
            denominator_inverses: descriptor.take(component.denominator_words, 1)?,
        });
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
    coefficient_pointers: usize,
    coefficient_sizes: usize,
    evaluation_pointers: usize,
    interaction_offsets: usize,
    denominator_inverses: usize,
    ext_params: *const u32,
    accumulator_offset_words: usize,
    trace_log_size: u32,
    evaluation_log_size: u32,
    row_count: u32,
    column_count: u32,
    kernels: Vec<PreparedKernel>,
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
    composition_coefficients: [ArenaSlice; SPLIT_COORDINATES],
    components: Vec<PreparedComponent>,
}

impl<'a> PreparedCompositionGraph<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        arena: &'a DeviceArena,
        plan: &CompositionPlan,
        trace: &CompositionTraceTopology,
        inputs: &CompositionDeviceInputs,
        slots: &CompositionWorkspaceSlots,
    ) -> Result<Self, PreparedCompositionError> {
        let requirements = composition_workspace_requirements(plan, trace)?;
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
        let forward_twiddles = bind_minimum(
            arena,
            inputs.forward_twiddles,
            requirements.forward_twiddle_words,
        )?;
        let inverse_twiddles = bind_minimum(
            arena,
            inputs.inverse_twiddles,
            requirements.inverse_twiddle_words,
        )?;
        if inputs.forward_twiddles == inputs.inverse_twiddles {
            return Err(PreparedCompositionError::ForwardInverseTwiddlesAlias(
                inputs.forward_twiddles,
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
        for tree in &trace.trees {
            for source in tree {
                if writable_ids.contains(&source.slot) {
                    return Err(PreparedCompositionError::SourceAliasesWritableWorkspace(
                        source.slot,
                    ));
                }
                let _ = bind_minimum(arena, source.slot, pow2(source.log_size)?)?;
            }
        }
        for input in [
            inputs.random_coefficient,
            inputs.forward_twiddles,
            inputs.inverse_twiddles,
            inputs.relation_z,
            inputs.relation_alpha_powers,
        ] {
            if writable_ids.contains(&input) {
                return Err(PreparedCompositionError::InputAliasesWritableWorkspace(
                    input,
                ));
            }
        }
        if inputs.relation_z == inputs.relation_alpha_powers {
            return Err(PreparedCompositionError::RelationChallengeSourcesAlias(
                inputs.relation_z,
            ));
        }
        for &claimed_sum in inputs.claimed_sums.iter().flatten() {
            if writable_ids.contains(&claimed_sum) {
                return Err(PreparedCompositionError::InputAliasesWritableWorkspace(
                    claimed_sum,
                ));
            }
        }

        let relation_z = bind_minimum(arena, inputs.relation_z, SECURE_WORDS)?;
        let relation_alpha_powers =
            bind_minimum(arena, inputs.relation_alpha_powers, SECURE_WORDS)?;
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
                let evaluation = unsafe {
                    lde_tile.as_u32_ptr().add(
                        source_index
                            .checked_mul(row_count)
                            .ok_or(PreparedCompositionError::SizeOverflow)?,
                    )
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

            let mut kernels = Vec::with_capacity(component_plan.kernels.len());
            for (kernel_index, kernel) in component_plan.kernels.iter().enumerate() {
                kernels.push(prepare_aot_kernel(component_index, kernel_index, kernel)?);
            }
            prepared_components.push(PreparedComponent {
                coefficient_pointers: descriptor.coefficient_pointers,
                coefficient_sizes: descriptor.coefficient_sizes,
                evaluation_pointers: descriptor.evaluation_pointers,
                interaction_offsets: descriptor.interaction_offsets,
                denominator_inverses: descriptor.denominator_inverses,
                ext_params,
                accumulator_offset_words: component.accumulator_offset_words,
                trace_log_size: component.trace_log_size,
                evaluation_log_size: component.evaluation_log_size,
                row_count: u32::try_from(component.row_count)
                    .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                column_count: u32::try_from(component.sources.len())
                    .map_err(|_| PreparedCompositionError::SizeOverflow)?,
                kernels,
            });
        }
        debug_assert_eq!(dynamic_index, requirements.dynamic_ext_param_count);
        debug_assert_eq!(claimed_index, requirements.claimed_sum_count);

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
            composition_coefficients,
            components: prepared_components,
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
        unsafe {
            context.memset_async(
                self.accumulators.as_void_ptr(),
                0,
                self.accumulators.len_bytes(),
            )?;
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

        let twiddle_words = u32::try_from(self.forward_twiddles.len_words())
            .map_err(|_| PreparedCompositionError::SizeOverflow)?;
        for (component_index, (plan_component, component)) in self
            .requirements
            .components
            .iter()
            .zip(&self.components)
            .enumerate()
        {
            check_status("composition_trace_lde", unsafe {
                raw::stwo_lde_n2b_columns_on(
                    descriptor_ptr
                        .add(component.coefficient_pointers)
                        .cast::<*const u32>(),
                    descriptor_ptr.add(component.coefficient_sizes),
                    descriptor_ptr
                        .add(component.evaluation_pointers)
                        .cast::<*mut u32>(),
                    component.evaluation_log_size,
                    component.column_count,
                    self.forward_twiddles.as_u32_ptr(),
                    twiddle_words,
                    1u32 << (component.evaluation_log_size - 1),
                    stream,
                )
            })?;

            let row_count = component.row_count as usize;
            let accumulator = unsafe {
                self.accumulators
                    .as_u32_ptr()
                    .add(component.accumulator_offset_words)
            };
            for (kernel_index, kernel) in component.kernels.iter().enumerate() {
                let rc_base = plan_component
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
                        descriptor_ptr.add(self.requirements.zero_words),
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
    Ok(slice)
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
    Ok(slice)
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
        };
        let requirements = composition_workspace_requirements(&plan, &trace()).unwrap();
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
    }

    #[test]
    fn slot_requirements_are_address_free_unique_and_exact() {
        let plan = CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 2,
            max_evaluation_log_size: 7,
            components: vec![component("a", 5, 7, 2, 0, vec![0], 0..1, 0..1)],
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
