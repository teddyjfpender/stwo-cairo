//! Exact direct-retained comparator for ordinary installed stripes.
//!
//! The three range-check instances deliberately reuse one embedded ordinary
//! kernel at evaluation logs 7, 13, and 24. This is a diagnostic stress shape,
//! not an SN2 claim. It proves the candidate's direct-split bytes and captured
//! input rebinding before a real SN2/153 hardware gate may claim promotion.

#![cfg(stwo_cuda_link)]

use core::mem::{size_of, size_of_val};

#[path = "composition_stripe_direct/device.rs"]
mod device;
#[path = "composition_stripe_direct/receipt.rs"]
mod receipt;

use std::collections::BTreeMap;
use std::ffi::CString;

use cairo_air::components::range_check_6;
use cairo_air::relations::CommonLookupElements;
use stwo::core::air::Component;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse;
use stwo::prover::backend::cpu::circle::slow_precompute_twiddles;
use stwo_backend_cuda::{
    aot, ArenaSlice, ArenaSlotId, CompositionSplitPointerSlots, CompositionSplitProgram,
    DeviceArena, COMPOSITION_RETAINED_COLUMNS,
};
use stwo_backend_cuda_kernels::raw;
use stwo_cairo_gpu_prover::arena_plan::{
    BufferLifetime, BufferPurpose, CommitmentTreeId, OpenedColumnSource, ProofEpoch,
};
use stwo_cairo_gpu_prover::composition_plan::{
    CompositionComponentPlan, CompositionExtParamSource, CompositionKernelPart, CompositionPlan,
    CompositionWaveKernelPlan,
};
use stwo_cairo_gpu_prover::direct_composition_retention::{
    direct_composition_plan_key, DirectCompositionBinding, DirectCompositionColumn,
    DirectCompositionRetentionPlan,
};
use stwo_cairo_gpu_prover::prepared_composition::{
    CompositionDirectEvaluationBinding, CompositionDirectSplitBinding,
};
use stwo_cairo_gpu_prover::{
    composition_workspace_requirements_with_mode, CompositionCoefficientSource,
    CompositionDeviceInputs, CompositionLaunchMode, CompositionOutputSlots,
    CompositionTraceTopology, CompositionWorkspaceRequirements, CompositionWorkspaceSlots,
    PreparedCompositionGraph,
};
use stwo_cairo_prover::witness::proof_shape::TracePartId;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{FrameworkComponent, FrameworkEval, TraceLocationAllocator};

pub(super) use device::{ready, refresh};
pub(super) use receipt::{optional_captured_abba, publish_receipt};

const TRACE_LOGS: [u32; 3] = [6, 12, 23];
const EVALUATION_LOGS: [u32; 3] = [7, 13, 24];
const MAX_EVALUATION_LOG: u32 = 24;
const MODULUS: u32 = 0x7fff_ffff;
const READ_CHUNK_WORDS: usize = 1 << 20;

const WAVE_WORKSPACE_BASE: u32 = 1;
const WAVE_SOURCE_POINTERS: ArenaSlotId = ArenaSlotId(10);
const WAVE_RETAINED_POINTERS: ArenaSlotId = ArenaSlotId(11);
const STRIPE_WORKSPACE_BASE: u32 = 20;
const STRIPE_SOURCE_POINTERS: ArenaSlotId = ArenaSlotId(30);
const STRIPE_RETAINED_POINTERS: ArenaSlotId = ArenaSlotId(31);
const RANDOM: ArenaSlotId = ArenaSlotId(40);
const FORWARD: ArenaSlotId = ArenaSlotId(41);
const INVERSE: ArenaSlotId = ArenaSlotId(42);
const RELATION_Z: ArenaSlotId = ArenaSlotId(43);
const RELATION_ALPHA: ArenaSlotId = ArenaSlotId(44);
const TRACE_BASE: u32 = 1_000;
const DIRECT_BASE: u32 = 2_000;
const WAVE_RETAINED_BASE: u32 = 3_000;
const STRIPE_RETAINED_BASE: u32 = 3_100;
const EXT_BASE: u32 = 4_000;

type RangeCheckComponent = FrameworkComponent<range_check_6::Eval>;

pub(super) struct Fixture {
    pub(super) plan: CompositionPlan,
    pub(super) trace: CompositionTraceTopology,
    pub(super) retention: DirectCompositionRetentionPlan,
    pub(super) wave_slots: CompositionWorkspaceSlots,
    pub(super) stripe_slots: CompositionWorkspaceSlots,
    split_program: CompositionSplitProgram,
    wave_pointer_slots: CompositionSplitPointerSlots,
    stripe_pointer_slots: CompositionSplitPointerSlots,
}

pub(super) struct Twiddles {
    forward: Vec<u32>,
    inverse: Vec<u32>,
}

pub(super) struct Ready {
    pub(super) arena: DeviceArena,
    pub(super) inputs: CompositionDeviceInputs,
    pub(super) direct: Vec<CompositionDirectEvaluationBinding>,
    pub(super) wave_split: CompositionDirectSplitBinding,
    pub(super) stripe_split: CompositionDirectSplitBinding,
    wave_retained: [ArenaSlice; COMPOSITION_RETAINED_COLUMNS],
    stripe_retained: [ArenaSlice; COMPOSITION_RETAINED_COLUMNS],
}

fn slot(base: u32, index: usize) -> ArenaSlotId {
    ArenaSlotId(base + u32::try_from(index).unwrap())
}

fn workspace_slots(base: u32) -> CompositionWorkspaceSlots {
    CompositionWorkspaceSlots {
        descriptors: ArenaSlotId(base),
        lde_tile: ArenaSlotId(base + 1),
        accumulators: ArenaSlotId(base + 2),
        random_coefficient_powers: ArenaSlotId(base + 3),
        output: CompositionOutputSlots::DirectRetainedEvaluations,
    }
}

fn denominator_inverses(trace_log: u32, evaluation_log: u32) -> Vec<BaseField> {
    let mut values = (0..1usize << (evaluation_log - trace_log))
        .map(|index| {
            coset_vanishing(
                CanonicCoset::new(trace_log).coset(),
                CanonicCoset::new(evaluation_log).circle_domain().at(index),
            )
            .inverse()
        })
        .collect::<Vec<_>>();
    bit_reverse(&mut values);
    values
}

fn components() -> Vec<RangeCheckComponent> {
    let lookup = CommonLookupElements::from_z_alpha(
        SecureField::from_u32_unchecked(2, 3, 5, 7),
        SecureField::from_u32_unchecked(11, 13, 17, 19),
    );
    let mut allocator =
        TraceLocationAllocator::new_with_preprocessed_columns(&[PreProcessedColumnId {
            id: "seq_6".to_owned(),
        }]);
    (0..TRACE_LOGS.len())
        .map(|instance| {
            range_check_6::Component::new(
                &mut allocator,
                range_check_6::Eval {
                    claim: range_check_6::Claim {},
                    common_lookup_elements: lookup.clone(),
                },
                SecureField::from_u32_unchecked(
                    23 + instance as u32,
                    29 + instance as u32,
                    31 + instance as u32,
                    37 + instance as u32,
                ),
            )
        })
        .collect()
}

fn lower(
    component: &RangeCheckComponent,
    instance: usize,
    coefficient_start: usize,
) -> (
    CompositionComponentPlan,
    (
        aot::CompositionWaveKernelPartIdentity,
        aot::ConstraintWaveFragment,
    ),
) {
    let emitted = aot::constraint_program(
        component.evaluator(),
        3,
        component.claimed_sum(),
        range_check_6::LOG_SIZE,
        aot::loaded_constraint_max_instrs(),
    )
    .unwrap();
    assert_eq!(emitted.kernels.len(), 1);
    let base_param_values = emitted.base_param_values;
    let ext_param_values = emitted.ext_param_values;
    let ext_param_sources = ext_param_values
        .iter()
        .copied()
        .map(CompositionExtParamSource::Constant)
        .collect();
    let part = emitted.kernels.into_iter().next().unwrap();
    let identity = aot::CompositionWaveKernelPartIdentity {
        semantic_hash: part.kernel.semantic_hash,
        coefficient_start: coefficient_start as u32,
        coefficient_end: (coefficient_start + component.n_constraints()) as u32,
    };
    let kernel = CompositionKernelPart {
        kernel_name: part.kernel.kernel_name,
        cache_key: part.kernel.cache_key,
        semantic_hash: part.kernel.semantic_hash,
        source: part.kernel.source,
        rc_base: part.rc_base,
    };
    assert_eq!(kernel.cache_key, 0xfe80_08c0_8878_f25a);
    let trace_log_size = TRACE_LOGS[instance];
    let evaluation_log_size = EVALUATION_LOGS[instance];
    (
        CompositionComponentPlan {
            component: "range_check_6_direct_diag",
            instance,
            trace_locations: component.trace_locations().to_vec(),
            preprocessed_column_indices: vec![instance],
            trace_log_size,
            evaluation_log_size,
            n_constraints: component.n_constraints(),
            random_coefficient_offset: coefficient_start,
            denominator_inverses: denominator_inverses(trace_log_size, evaluation_log_size),
            base_param_values,
            ext_param_values,
            ext_param_sources,
            kernels: vec![kernel],
        },
        (identity, part.wave_fragment),
    )
}

fn plans() -> (CompositionPlan, CompositionTraceTopology) {
    let mut coefficient_start = 0usize;
    let mut plans = Vec::new();
    let mut waves = BTreeMap::new();
    for (instance, component) in components().iter().enumerate() {
        let (plan, wave_part) = lower(component, instance, coefficient_start);
        coefficient_start += plan.n_constraints;
        waves
            .entry(plan.evaluation_log_size)
            .or_insert_with(Vec::new)
            .push(wave_part);
        plans.push(plan);
    }
    let wave_kernels = waves
        .into_iter()
        .map(|(evaluation_log_size, parts)| {
            let identities = parts.iter().map(|part| part.0).collect::<Vec<_>>();
            let emitted = aot::composition_wave_kernel_source(evaluation_log_size, &parts).unwrap();
            let expected =
                aot::composition_wave_kernel_identity(evaluation_log_size, &identities).unwrap();
            assert_eq!(
                (
                    emitted.kernel_name.as_str(),
                    emitted.cache_key,
                    emitted.semantic_hash,
                ),
                (
                    expected.kernel_name.as_str(),
                    expected.cache_key,
                    expected.semantic_hash,
                )
            );
            CompositionWaveKernelPlan {
                evaluation_log_size,
                parts: identities,
                kernel_name: emitted.kernel_name,
                cache_key: emitted.cache_key,
                semantic_hash: emitted.semantic_hash,
                program_identity: emitted.program_identity.unwrap(),
                source: emitted.source,
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        wave_kernels
            .iter()
            .map(|wave| wave.evaluation_log_size)
            .collect::<Vec<_>>(),
        EVALUATION_LOGS
    );
    let source = |tree: usize, column: usize, log_size| CompositionCoefficientSource {
        slot: slot(TRACE_BASE + 100 * tree as u32, column),
        log_size,
    };
    let trace = CompositionTraceTopology {
        trees: vec![
            TRACE_LOGS
                .iter()
                .enumerate()
                .map(|(column, &log)| source(0, column, log))
                .collect(),
            TRACE_LOGS
                .iter()
                .enumerate()
                .map(|(column, &log)| source(1, column, log))
                .collect(),
            TRACE_LOGS
                .iter()
                .enumerate()
                .flat_map(|(instance, &log)| {
                    (0..4).map(move |ordinal| source(2, 4 * instance + ordinal, log))
                })
                .collect(),
        ],
    };
    (
        CompositionPlan {
            max_kernel_instrs: aot::loaded_constraint_max_instrs(),
            total_constraints: coefficient_start,
            max_evaluation_log_size: MAX_EVALUATION_LOG,
            components: plans,
            wave_kernels,
        },
        trace,
    )
}

fn retention(requirements: &CompositionWorkspaceRequirements) -> DirectCompositionRetentionPlan {
    let occurrence_count = requirements
        .components
        .iter()
        .map(|component| component.sources.len())
        .sum::<usize>();
    let mut columns = Vec::<DirectCompositionColumn>::new();
    let mut bindings = Vec::with_capacity(occurrence_count);
    let mut direct_bitmap = vec![0u64; occurrence_count.div_ceil(64)];
    for (consumer, (component, source)) in requirements
        .components
        .iter()
        .flat_map(|component| {
            component
                .sources
                .iter()
                .map(move |source| (component, source))
        })
        .enumerate()
    {
        let tree = match source.tree {
            0 => CommitmentTreeId::Preprocessed,
            1 => CommitmentTreeId::Base,
            2 => CommitmentTreeId::Interaction,
            _ => unreachable!(),
        };
        let column = columns
            .iter()
            .position(|column| column.tree == tree && column.proof_column == source.column)
            .unwrap_or_else(|| {
                let opened = match tree {
                    CommitmentTreeId::Preprocessed => OpenedColumnSource::Preprocessed {
                        ordinal: source.column as u32,
                    },
                    CommitmentTreeId::Base | CommitmentTreeId::Interaction => {
                        OpenedColumnSource::Trace {
                            component: "range_check_6_direct_diag",
                            part: TracePartId::Main,
                            purpose: if tree == CommitmentTreeId::Base {
                                BufferPurpose::BaseCoefficients
                            } else {
                                BufferPurpose::InteractionCoefficients
                            },
                            ordinal: source.column as u32,
                        }
                    }
                    CommitmentTreeId::Composition | CommitmentTreeId::Fri(_) => unreachable!(),
                };
                columns.push(DirectCompositionColumn {
                    source: opened,
                    tree,
                    proof_column: source.column,
                    group: 0,
                    column_in_group: source.column,
                    canonical_column: source.column,
                    coefficient_log_size: source.source.log_size,
                    evaluation_log_size: source.source.log_size + 1,
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
        assert_eq!(
            component.evaluation_log_size,
            columns[column].evaluation_log_size
        );
        direct_bitmap[consumer / 64] |= 1 << (consumer % 64);
        bindings.push(DirectCompositionBinding {
            consumer,
            column,
            consumer_evaluation_log_size: component.evaluation_log_size,
            direct: true,
        });
    }
    let direct_bytes = columns
        .iter()
        .map(|column| (1usize << column.evaluation_log_size) * size_of::<u32>())
        .sum();
    let mut plan = DirectCompositionRetentionPlan {
        direct_column_count: columns.len(),
        direct_bytes,
        columns,
        bindings,
        direct_bitmap,
        buckets: Vec::new(),
        cache_key: 0,
    };
    plan.cache_key = direct_composition_plan_key(&plan);
    plan
}

pub(super) fn fixture() -> Fixture {
    let (plan, trace) = plans();
    let baseline =
        composition_workspace_requirements_with_mode(&plan, &trace, CompositionLaunchMode::Serial)
            .unwrap();
    let retention = retention(&baseline);
    assert_eq!(retention.direct_column_count, 18);
    assert_eq!(plan.wave_kernels.len(), 3);
    Fixture {
        plan,
        trace,
        retention,
        wave_slots: workspace_slots(WAVE_WORKSPACE_BASE),
        stripe_slots: workspace_slots(STRIPE_WORKSPACE_BASE),
        split_program: CompositionSplitProgram::compile(MAX_EVALUATION_LOG).unwrap(),
        wave_pointer_slots: CompositionSplitPointerSlots {
            source_pointers: WAVE_SOURCE_POINTERS,
            retained_pointers: WAVE_RETAINED_POINTERS,
        },
        stripe_pointer_slots: CompositionSplitPointerSlots {
            source_pointers: STRIPE_SOURCE_POINTERS,
            retained_pointers: STRIPE_RETAINED_POINTERS,
        },
    }
}

pub(super) fn twiddles() -> Twiddles {
    let half_coset = CanonicCoset::new(MAX_EVALUATION_LOG)
        .circle_domain()
        .half_coset;
    Twiddles {
        forward: slow_precompute_twiddles(half_coset)
            .into_iter()
            .map(|value| value.0)
            .collect(),
        inverse: slow_precompute_twiddles(half_coset)
            .into_iter()
            .map(|value| value.inverse().0)
            .collect(),
    }
}

pub(super) fn install_wave_sources(plan: &CompositionPlan) {
    assert_eq!(plan.wave_kernels.len(), EVALUATION_LOGS.len());
    aot::reset_runtime_stats();
    for wave in &plan.wave_kernels {
        let source = CString::new(wave.source.as_bytes()).unwrap();
        let name = CString::new(wave.kernel_name.as_bytes()).unwrap();
        assert!(unsafe {
            raw::stwo_cuda_jit_precompile(source.as_ptr(), name.as_ptr(), wave.cache_key, false)
        });
        assert!(unsafe {
            raw::stwo_cuda_jit_precompile(core::ptr::null(), name.as_ptr(), wave.cache_key, false)
        });
    }
    let stats = aot::runtime_stats();
    assert_eq!(stats.runtime_loads, EVALUATION_LOGS.len() as u64);
    assert!(stats.runtime_cache_hits >= EVALUATION_LOGS.len() as u64);
    assert_eq!(stats.strict_rejections, 0);
}

fn read_chunk(arena: &DeviceArena, slice: ArenaSlice, words: &mut [u32]) {
    assert_eq!(slice.len_words(), words.len());
    unsafe {
        arena
            .context()
            .memcpy_d2h_async(
                words.as_mut_ptr().cast(),
                slice.as_void_ptr().cast_const(),
                size_of_val(words),
            )
            .unwrap();
    }
    arena.context().sync().unwrap();
}

pub(super) fn assert_retained_equal(ready: &Ready) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"stwo-composition-direct-retained-v1");
    let mut wave_words = vec![0u32; READ_CHUNK_WORDS];
    let mut stripe_words = vec![0u32; READ_CHUNK_WORDS];
    let rows = 1usize << MAX_EVALUATION_LOG;
    for column in 0..COMPOSITION_RETAINED_COLUMNS {
        for offset in (0..rows).step_by(READ_CHUNK_WORDS) {
            let count = READ_CHUNK_WORDS.min(rows - offset);
            read_chunk(
                &ready.arena,
                ready.wave_retained[column]
                    .checked_subslice(offset, count)
                    .unwrap(),
                &mut wave_words[..count],
            );
            read_chunk(
                &ready.arena,
                ready.stripe_retained[column]
                    .checked_subslice(offset, count)
                    .unwrap(),
                &mut stripe_words[..count],
            );
            assert_eq!(
                wave_words[..count],
                stripe_words[..count],
                "direct-retained byte mismatch at column {column}, word {offset}"
            );
            hasher.update(bytemuck::cast_slice(&wave_words[..count]));
        }
    }
    *hasher.finalize().as_bytes()
}
