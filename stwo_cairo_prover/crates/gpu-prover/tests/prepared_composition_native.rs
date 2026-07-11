//! Native eager/capture parity for the complete prepared composition graph.

#![cfg(stwo_cuda_link)]

use core::ffi::c_void;

use cairo_air::components::{range_check_6, range_check_8};
use cairo_air::relations::CommonLookupElements;
use stwo::core::air::Component;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse;
use stwo::prover::backend::cpu::circle::slow_precompute_twiddles;
use stwo::prover::backend::cpu::{CpuBackend, CpuCirclePoly};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo_backend_cuda::{
    aot, ArenaLayout, ArenaSlotId, ArenaSlotSpec, CudaExecContext, DeviceArena,
};
use stwo_cairo_gpu_prover::composition_plan::{
    CompositionComponentPlan, CompositionExtParamSource, CompositionKernelPart, CompositionPlan,
};
use stwo_cairo_gpu_prover::{
    composition_workspace_requirements, composition_workspace_requirements_with_mode,
    CompositionCoefficientSource, CompositionDeviceInputs, CompositionExtParamBinding,
    CompositionLaunchMode, CompositionTraceTopology, CompositionWorkspaceRequirements,
    CompositionWorkspaceSlots, PreparedCompositionGraph,
};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    accumulate_pointwise_cpu, FrameworkComponent, FrameworkEval, TraceLocationAllocator,
};

const PREPROCESSED: ArenaSlotId = ArenaSlotId(100);
const BASE: ArenaSlotId = ArenaSlotId(101);
const INTERACTION_0: ArenaSlotId = ArenaSlotId(102);
const INTERACTION_1: ArenaSlotId = ArenaSlotId(103);
const INTERACTION_2: ArenaSlotId = ArenaSlotId(104);
const INTERACTION_3: ArenaSlotId = ArenaSlotId(105);
const RANDOM_COEFFICIENT: ArenaSlotId = ArenaSlotId(106);
const FORWARD_TWIDDLES: ArenaSlotId = ArenaSlotId(107);
const INVERSE_TWIDDLES: ArenaSlotId = ArenaSlotId(108);
const RELATION_Z: ArenaSlotId = ArenaSlotId(109);
const RELATION_ALPHA_POWERS: ArenaSlotId = ArenaSlotId(110);
const EXT_PARAMS: ArenaSlotId = ArenaSlotId(111);
const TRACE_LOG_SIZE: u32 = range_check_6::LOG_SIZE;
const EVALUATION_LOG_SIZE: u32 = TRACE_LOG_SIZE + 1;
const SECURE_WORDS: usize = 4;

type RangeCheckComponent = FrameworkComponent<range_check_6::Eval>;

#[derive(Clone)]
struct Coefficients {
    preprocessed: Vec<BaseField>,
    base: Vec<BaseField>,
    interaction: [Vec<BaseField>; 4],
}

/// Lower one real Cairo component against the embedded AOT pack into a
/// composition-plan entry, with all extension parameters pinned as constants
/// (the differential lane's setup; the resident graph binds dynamic sources).
fn lower_component_plan<E: FrameworkEval>(
    name: &'static str,
    component: &FrameworkComponent<E>,
    random_coefficient_offset: usize,
) -> CompositionComponentPlan {
    let trace_log_size = component.evaluator().log_size();
    let evaluation_log_size = component.max_constraint_log_degree_bound();
    let emitted = aot::constraint_program(
        component.evaluator(),
        3,
        component.claimed_sum(),
        trace_log_size,
        aot::loaded_constraint_max_instrs(),
    )
    .expect("component must lower into the embedded AOT pack");
    let mut denominator_inverses = (0..1usize << (evaluation_log_size - trace_log_size))
        .map(|index| {
            coset_vanishing(
                CanonicCoset::new(trace_log_size).coset(),
                CanonicCoset::new(evaluation_log_size)
                    .circle_domain()
                    .at(index),
            )
            .inverse()
        })
        .collect::<Vec<_>>();
    bit_reverse(&mut denominator_inverses);
    let ext_param_values = emitted.ext_param_values;
    let ext_param_sources = ext_param_values
        .iter()
        .copied()
        .map(CompositionExtParamSource::Constant)
        .collect();
    let kernels = emitted
        .kernels
        .into_iter()
        .map(|part| CompositionKernelPart {
            kernel_name: part.kernel.kernel_name,
            cache_key: part.kernel.cache_key,
            semantic_hash: part.kernel.semantic_hash,
            source: part.kernel.source,
            rc_base: part.rc_base,
        })
        .collect();
    CompositionComponentPlan {
        component: name,
        instance: 0,
        trace_locations: component.trace_locations().to_vec(),
        preprocessed_column_indices: component.preprocessed_column_indices().to_vec(),
        trace_log_size,
        evaluation_log_size,
        n_constraints: component.n_constraints(),
        random_coefficient_offset,
        denominator_inverses,
        ext_param_values,
        ext_param_sources,
        kernels,
    }
}

fn real_component_and_plan() -> (RangeCheckComponent, CompositionPlan) {
    let lookup = CommonLookupElements::from_z_alpha(
        SecureField::from_u32_unchecked(2, 3, 5, 7),
        SecureField::from_u32_unchecked(11, 13, 17, 19),
    );
    let claimed_sum = SecureField::from_u32_unchecked(23, 29, 31, 37);
    let mut allocator =
        TraceLocationAllocator::new_with_preprocessed_columns(&[PreProcessedColumnId {
            id: "seq_6".to_owned(),
        }]);
    let component = range_check_6::Component::new(
        &mut allocator,
        range_check_6::Eval {
            claim: range_check_6::Claim {},
            common_lookup_elements: lookup,
        },
        claimed_sum,
    );
    let max_kernel_instrs = aot::loaded_constraint_max_instrs();
    assert_eq!(max_kernel_instrs, 2048, "unexpected embedded AOT cap");
    let component_plan = lower_component_plan("range_check_6", &component, 0);
    assert_eq!(component_plan.kernels.len(), 1);
    assert_eq!(component_plan.kernels[0].cache_key, 0x5754_f8a8_73a5_2740);
    let total_constraints = component.n_constraints();
    (
        component,
        CompositionPlan {
            max_kernel_instrs,
            total_constraints,
            max_evaluation_log_size: EVALUATION_LOG_SIZE,
            components: vec![component_plan],
        },
    )
}

fn topology() -> CompositionTraceTopology {
    let source = |slot| CompositionCoefficientSource {
        slot,
        log_size: TRACE_LOG_SIZE,
    };
    CompositionTraceTopology {
        trees: vec![
            vec![source(PREPROCESSED)],
            vec![source(BASE)],
            vec![
                source(INTERACTION_0),
                source(INTERACTION_1),
                source(INTERACTION_2),
                source(INTERACTION_3),
            ],
        ],
    }
}

fn workspace_slots() -> CompositionWorkspaceSlots {
    CompositionWorkspaceSlots {
        descriptors: ArenaSlotId(1),
        lde_tile: ArenaSlotId(2),
        accumulators: ArenaSlotId(3),
        random_coefficient_powers: ArenaSlotId(4),
        composition_coefficients: std::array::from_fn(|index| ArenaSlotId(5 + index as u32)),
    }
}

fn arena(
    requirements: &CompositionWorkspaceRequirements,
    slots: &CompositionWorkspaceSlots,
    extra_slots: &[(ArenaSlotId, usize, usize)],
) -> DeviceArena {
    let mut requested = requirements
        .arena_slot_requirements(slots)
        .unwrap()
        .into_iter()
        .map(|requirement| {
            (
                requirement.id,
                requirement.len_words,
                requirement.alignment_words,
            )
        })
        .collect::<Vec<_>>();
    requested.extend(extra_slots.iter().copied());
    let mut offset = 0usize;
    let mut specs = Vec::with_capacity(requested.len());
    for (id, len_words, alignment_words) in requested {
        offset = offset.next_multiple_of(alignment_words);
        specs.push(ArenaSlotSpec {
            id,
            offset_words: offset,
            len_words,
            alignment_words,
        });
        offset += len_words;
    }
    DeviceArena::new(
        CudaExecContext::new().unwrap(),
        ArenaLayout::new(offset, &specs).unwrap(),
    )
    .unwrap()
}

fn sequence_coefficients(log_size: u32) -> Vec<BaseField> {
    CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
        CanonicCoset::new(log_size).circle_domain(),
        (0..1usize << log_size).map(BaseField::from).collect(),
    )
    .interpolate()
    .coeffs
}

fn linear_column(log_size: u32, factor: u32, offset: u32) -> Vec<BaseField> {
    (0..1usize << log_size)
        .map(|index| {
            BaseField::from_u32_unchecked(
                factor.wrapping_mul(index as u32).wrapping_add(offset) & 0x7fff_ffff,
            )
        })
        .collect()
}

fn coefficients(seed: u32) -> Coefficients {
    coefficients_at(TRACE_LOG_SIZE, seed)
}

fn coefficients_at(log_size: u32, seed: u32) -> Coefficients {
    Coefficients {
        preprocessed: sequence_coefficients(log_size),
        base: linear_column(log_size, 17 + seed, 3 + seed),
        interaction: [
            linear_column(log_size, 29 + seed, 5),
            linear_column(log_size, 43 + seed, 7),
            linear_column(log_size, 71 + seed, 11),
            linear_column(log_size, 101 + seed, 13),
        ],
    }
}

fn upload<T>(arena: &DeviceArena, slot: ArenaSlotId, values: &[T]) {
    let destination = arena.bind(slot).unwrap();
    assert!(core::mem::size_of_val(values) <= destination.len_bytes());
    unsafe {
        arena
            .context()
            .memcpy_h2d_async(
                destination.as_void_ptr(),
                values.as_ptr().cast::<c_void>(),
                core::mem::size_of_val(values),
            )
            .unwrap();
    }
}

fn upload_coefficients(arena: &DeviceArena, coefficients: &Coefficients) {
    let words = |values: &[BaseField]| values.iter().map(|value| value.0).collect::<Vec<_>>();
    upload(arena, PREPROCESSED, &words(&coefficients.preprocessed));
    upload(arena, BASE, &words(&coefficients.base));
    for (slot, values) in [INTERACTION_0, INTERACTION_1, INTERACTION_2, INTERACTION_3]
        .into_iter()
        .zip(&coefficients.interaction)
    {
        upload(arena, slot, &words(values));
    }
}

fn expected_outputs(
    component: &RangeCheckComponent,
    plan: &CompositionPlan,
    coefficients: &Coefficients,
    random_coefficient: SecureField,
) -> [Vec<u32>; 8] {
    let evaluation_domain = CanonicCoset::new(EVALUATION_LOG_SIZE).circle_domain();
    let evaluate =
        |values: &[BaseField]| CpuCirclePoly::new(values.to_vec()).evaluate(evaluation_domain);
    let evaluations = TreeVec::new(vec![
        vec![evaluate(&coefficients.preprocessed)],
        vec![evaluate(&coefficients.base)],
        coefficients
            .interaction
            .iter()
            .map(|values| evaluate(values))
            .collect(),
    ]);
    let random_powers = (0..plan.total_constraints)
        .map(|index| random_coefficient.pow((plan.total_constraints - 1 - index) as u128))
        .collect::<Vec<_>>();
    let accumulation = accumulate_pointwise_cpu(
        component,
        evaluations.as_cols_ref(),
        EVALUATION_LOG_SIZE,
        TRACE_LOG_SIZE,
        plan.components[0].denominator_inverses.clone(),
        &random_powers,
        &SecureColumnByCoords::<CpuBackend>::zeros(1 << EVALUATION_LOG_SIZE),
    );
    interpolate_and_split(accumulation, EVALUATION_LOG_SIZE)
}

/// Interpolate the accumulated secure column over its evaluation domain and
/// split each coordinate's coefficients into the eight committed halves —
/// exactly the device graph's b2n + d2d tail.
fn interpolate_and_split(
    accumulation: SecureColumnByCoords<CpuBackend>,
    evaluation_log_size: u32,
) -> [Vec<u32>; 8] {
    let evaluation_domain = CanonicCoset::new(evaluation_log_size).circle_domain();
    let full = accumulation.columns.map(|values| {
        CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(evaluation_domain, values)
            .interpolate()
            .coeffs
            .into_iter()
            .map(|value| value.0)
            .collect::<Vec<_>>()
    });
    let half = 1usize << (evaluation_log_size - 1);
    std::array::from_fn(|index| {
        let coordinate = index % 4;
        let start = usize::from(index >= 4) * half;
        full[coordinate][start..start + half].to_vec()
    })
}

fn read_outputs(arena: &DeviceArena, slots: &CompositionWorkspaceSlots) -> [Vec<u32>; 8] {
    slots.composition_coefficients.map(|slot| {
        let source = arena.bind(slot).unwrap();
        let mut words = vec![0u32; source.len_words()];
        unsafe {
            arena
                .context()
                .memcpy_d2h_async(
                    words.as_mut_ptr().cast::<c_void>(),
                    source.as_void_ptr(),
                    core::mem::size_of_val(words.as_slice()),
                )
                .unwrap();
        }
        words
    })
}

#[test]
fn real_range_check_6_matches_cpu_eager_and_capture_replay() {
    const SHARED_TWIDDLE_LOG_SIZE: u32 = EVALUATION_LOG_SIZE + 2;

    let (component, plan) = real_component_and_plan();
    let trace = topology();
    let requirements = composition_workspace_requirements(&plan, &trace).unwrap();
    assert_eq!(requirements.dynamic_ext_param_count, 0);
    assert_eq!(
        requirements.output_coefficient_words,
        1 << (EVALUATION_LOG_SIZE - 1)
    );
    let slots = workspace_slots();
    let ext_param_words = plan.components[0].ext_param_values.len() * SECURE_WORDS;
    let arena = arena(
        &requirements,
        &slots,
        &[
            (PREPROCESSED, 1 << TRACE_LOG_SIZE, 1),
            (BASE, 1 << TRACE_LOG_SIZE, 1),
            (INTERACTION_0, 1 << TRACE_LOG_SIZE, 1),
            (INTERACTION_1, 1 << TRACE_LOG_SIZE, 1),
            (INTERACTION_2, 1 << TRACE_LOG_SIZE, 1),
            (INTERACTION_3, 1 << TRACE_LOG_SIZE, 1),
            (RANDOM_COEFFICIENT, SECURE_WORDS, SECURE_WORDS),
            (FORWARD_TWIDDLES, 1 << (SHARED_TWIDDLE_LOG_SIZE - 1), 1),
            (INVERSE_TWIDDLES, 1 << (SHARED_TWIDDLE_LOG_SIZE - 1), 1),
            (RELATION_Z, SECURE_WORDS, SECURE_WORDS),
            (RELATION_ALPHA_POWERS, SECURE_WORDS, SECURE_WORDS),
            (EXT_PARAMS, ext_param_words, SECURE_WORDS),
        ],
    );
    let random_coefficient = SecureField::from_u32_unchecked(107, 109, 113, 127);
    upload(
        &arena,
        RANDOM_COEFFICIENT,
        &random_coefficient
            .to_m31_array()
            .map(|coordinate| coordinate.0),
    );
    upload(&arena, RELATION_Z, &[0u32; SECURE_WORDS]);
    upload(&arena, RELATION_ALPHA_POWERS, &[0u32; SECURE_WORDS]);
    // Resident proofs share one larger twiddle tree across smaller composition
    // domains. CUDA selects the nested tree relative to the logical END, so
    // preparation must retain this caller-provided length.
    let domain = CanonicCoset::new(SHARED_TWIDDLE_LOG_SIZE).circle_domain();
    let forward = slow_precompute_twiddles(domain.half_coset)
        .into_iter()
        .map(|value| value.0)
        .collect::<Vec<_>>();
    let inverse = slow_precompute_twiddles(domain.half_coset)
        .into_iter()
        .map(|value| value.inverse().0)
        .collect::<Vec<_>>();
    upload(&arena, FORWARD_TWIDDLES, &forward);
    upload(&arena, INVERSE_TWIDDLES, &inverse);

    let foreign_arena = DeviceArena::new(
        CudaExecContext::new().unwrap(),
        ArenaLayout::new(
            forward.len(),
            &[ArenaSlotSpec {
                id: FORWARD_TWIDDLES,
                offset_words: 0,
                len_words: forward.len(),
                alignment_words: 1,
            }],
        )
        .unwrap(),
    )
    .unwrap();
    let foreign_result = PreparedCompositionGraph::prepare(
        &arena,
        &plan,
        &trace,
        &CompositionDeviceInputs {
            random_coefficient: RANDOM_COEFFICIENT,
            forward_twiddles: foreign_arena.bind(FORWARD_TWIDDLES).unwrap(),
            inverse_twiddles: arena.bind(INVERSE_TWIDDLES).unwrap(),
            relation_z: arena.bind(RELATION_Z).unwrap(),
            relation_alpha_powers: arena.bind(RELATION_ALPHA_POWERS).unwrap(),
            claimed_sums: vec![None],
            ext_params: vec![Some(CompositionExtParamBinding {
                slot: EXT_PARAMS,
                offset_words: 0,
            })],
        },
        &slots,
    );
    assert!(matches!(
        foreign_result,
        Err(stwo_cairo_gpu_prover::PreparedCompositionError::ContextMismatch(FORWARD_TWIDDLES))
    ));

    let prepared = PreparedCompositionGraph::prepare(
        &arena,
        &plan,
        &trace,
        &CompositionDeviceInputs {
            random_coefficient: RANDOM_COEFFICIENT,
            forward_twiddles: arena.bind(FORWARD_TWIDDLES).unwrap(),
            inverse_twiddles: arena.bind(INVERSE_TWIDDLES).unwrap(),
            relation_z: arena.bind(RELATION_Z).unwrap(),
            relation_alpha_powers: arena.bind(RELATION_ALPHA_POWERS).unwrap(),
            claimed_sums: vec![None],
            ext_params: vec![Some(CompositionExtParamBinding {
                slot: EXT_PARAMS,
                offset_words: 0,
            })],
        },
        &slots,
    )
    .unwrap();

    let eager_coefficients = coefficients(0);
    upload_coefficients(&arena, &eager_coefficients);
    arena.context().sync().unwrap();
    prepared.launch().unwrap();
    let eager = read_outputs(&arena, &slots);
    arena.context().sync().unwrap();
    assert_eq!(
        eager,
        expected_outputs(&component, &plan, &eager_coefficients, random_coefficient)
    );

    let capture = arena.context().capture().unwrap();
    prepared.launch().unwrap();
    let graph = capture.finish().unwrap();
    let replay_coefficients = coefficients(19);
    upload_coefficients(&arena, &replay_coefficients);
    arena.context().sync().unwrap();
    graph.launch(arena.context()).unwrap();
    let replay = read_outputs(&arena, &slots);
    arena.context().sync().unwrap();
    assert_eq!(
        replay,
        expected_outputs(&component, &plan, &replay_coefficients, random_coefficient)
    );
    assert_ne!(
        eager, replay,
        "captured graph must reread resident coefficients"
    );
}

fn component_evaluations(
    coefficients: &Coefficients,
    evaluation_log_size: u32,
) -> TreeVec<Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> {
    let evaluation_domain = CanonicCoset::new(evaluation_log_size).circle_domain();
    let evaluate =
        |values: &[BaseField]| CpuCirclePoly::new(values.to_vec()).evaluate(evaluation_domain);
    TreeVec::new(vec![
        vec![evaluate(&coefficients.preprocessed)],
        vec![evaluate(&coefficients.base)],
        coefficients
            .interaction
            .iter()
            .map(|values| evaluate(values))
            .collect(),
    ])
}

/// Both stream topologies of the two-component graph — serial (default) and
/// wide (`STWO_CUDA_COMPOSITION_WIDE=1`) — in one invocation: each mode must
/// match the CPU reference eagerly and under capture/replay, and the two
/// modes must be byte-identical to each other. The components' evaluation
/// logs differ (7 and 9), so the wide mode forms two groups on two distinct
/// lanes: the captured graph genuinely fans out.
#[test]
fn serial_and_wide_modes_match_cpu_and_each_other() {
    const SEQ_6: ArenaSlotId = ArenaSlotId(200);
    const SEQ_8: ArenaSlotId = ArenaSlotId(201);
    const BASE_6: ArenaSlotId = ArenaSlotId(202);
    const BASE_8: ArenaSlotId = ArenaSlotId(203);
    const INTERACTION_6: [ArenaSlotId; 4] = [
        ArenaSlotId(204),
        ArenaSlotId(205),
        ArenaSlotId(206),
        ArenaSlotId(207),
    ];
    const INTERACTION_8: [ArenaSlotId; 4] = [
        ArenaSlotId(208),
        ArenaSlotId(209),
        ArenaSlotId(210),
        ArenaSlotId(211),
    ];
    const RANDOM: ArenaSlotId = ArenaSlotId(212);
    const FORWARD: ArenaSlotId = ArenaSlotId(213);
    const INVERSE: ArenaSlotId = ArenaSlotId(214);
    const Z: ArenaSlotId = ArenaSlotId(215);
    const ALPHA: ArenaSlotId = ArenaSlotId(216);
    const EXT_6: ArenaSlotId = ArenaSlotId(217);
    const EXT_8: ArenaSlotId = ArenaSlotId(218);

    let lookup = CommonLookupElements::from_z_alpha(
        SecureField::from_u32_unchecked(2, 3, 5, 7),
        SecureField::from_u32_unchecked(11, 13, 17, 19),
    );
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(&[
        PreProcessedColumnId {
            id: "seq_6".to_owned(),
        },
        PreProcessedColumnId {
            id: "seq_8".to_owned(),
        },
    ]);
    let component_6 = range_check_6::Component::new(
        &mut allocator,
        range_check_6::Eval {
            claim: range_check_6::Claim {},
            common_lookup_elements: lookup.clone(),
        },
        SecureField::from_u32_unchecked(23, 29, 31, 37),
    );
    let component_8 = range_check_8::Component::new(
        &mut allocator,
        range_check_8::Eval {
            claim: range_check_8::Claim {},
            common_lookup_elements: lookup,
        },
        SecureField::from_u32_unchecked(41, 43, 47, 53),
    );
    let constraints_6 = component_6.n_constraints();
    let constraints_8 = component_8.n_constraints();
    let eval_log_6 = component_6.max_constraint_log_degree_bound();
    let eval_log_8 = component_8.max_constraint_log_degree_bound();
    assert_eq!((eval_log_6, eval_log_8), (7, 9));
    let plan = CompositionPlan {
        max_kernel_instrs: aot::loaded_constraint_max_instrs(),
        total_constraints: constraints_6 + constraints_8,
        max_evaluation_log_size: eval_log_8,
        components: vec![
            lower_component_plan("range_check_6", &component_6, 0),
            lower_component_plan("range_check_8", &component_8, constraints_6),
        ],
    };
    let source = |slot, log_size| CompositionCoefficientSource { slot, log_size };
    let trace = CompositionTraceTopology {
        trees: vec![
            vec![source(SEQ_6, 6), source(SEQ_8, 8)],
            vec![source(BASE_6, 6), source(BASE_8, 8)],
            vec![
                source(INTERACTION_6[0], 6),
                source(INTERACTION_6[1], 6),
                source(INTERACTION_6[2], 6),
                source(INTERACTION_6[3], 6),
                source(INTERACTION_8[0], 8),
                source(INTERACTION_8[1], 8),
                source(INTERACTION_8[2], 8),
                source(INTERACTION_8[3], 8),
            ],
        ],
    };
    // Wide geometry is a strict superset of serial (same slots, private tile
    // regions appended), so one wide-sized arena serves both prepared graphs.
    let wide_requirements =
        composition_workspace_requirements_with_mode(&plan, &trace, CompositionLaunchMode::Wide)
            .unwrap();
    let serial_requirements =
        composition_workspace_requirements_with_mode(&plan, &trace, CompositionLaunchMode::Serial)
            .unwrap();
    assert_eq!(wide_requirements.wide_groups.len(), 2);
    assert!(wide_requirements.serial_components.is_empty());
    assert!(wide_requirements.lde_tile_words >= serial_requirements.lde_tile_words);
    let slots = workspace_slots();
    let arena = arena(
        &wide_requirements,
        &slots,
        &[
            (SEQ_6, 1 << 6, 1),
            (SEQ_8, 1 << 8, 1),
            (BASE_6, 1 << 6, 1),
            (BASE_8, 1 << 8, 1),
            (INTERACTION_6[0], 1 << 6, 1),
            (INTERACTION_6[1], 1 << 6, 1),
            (INTERACTION_6[2], 1 << 6, 1),
            (INTERACTION_6[3], 1 << 6, 1),
            (INTERACTION_8[0], 1 << 8, 1),
            (INTERACTION_8[1], 1 << 8, 1),
            (INTERACTION_8[2], 1 << 8, 1),
            (INTERACTION_8[3], 1 << 8, 1),
            (RANDOM, SECURE_WORDS, SECURE_WORDS),
            (FORWARD, wide_requirements.forward_twiddle_words, 1),
            (INVERSE, wide_requirements.inverse_twiddle_words, 1),
            (Z, SECURE_WORDS, SECURE_WORDS),
            (ALPHA, SECURE_WORDS, SECURE_WORDS),
            (
                EXT_6,
                plan.components[0].ext_param_values.len() * SECURE_WORDS,
                SECURE_WORDS,
            ),
            (
                EXT_8,
                plan.components[1].ext_param_values.len() * SECURE_WORDS,
                SECURE_WORDS,
            ),
        ],
    );
    let random_coefficient = SecureField::from_u32_unchecked(107, 109, 113, 127);
    upload(
        &arena,
        RANDOM,
        &random_coefficient
            .to_m31_array()
            .map(|coordinate| coordinate.0),
    );
    upload(&arena, Z, &[0u32; SECURE_WORDS]);
    upload(&arena, ALPHA, &[0u32; SECURE_WORDS]);
    // One twiddle tree for the largest evaluation domain; the smaller
    // component's LDE addresses its nested subtree, exactly as the resident
    // runtime's shared twiddle slot does.
    let domain = CanonicCoset::new(eval_log_8).circle_domain();
    let forward = slow_precompute_twiddles(domain.half_coset)
        .into_iter()
        .map(|value| value.0)
        .collect::<Vec<_>>();
    let inverse = slow_precompute_twiddles(domain.half_coset)
        .into_iter()
        .map(|value| value.inverse().0)
        .collect::<Vec<_>>();
    upload(&arena, FORWARD, &forward);
    upload(&arena, INVERSE, &inverse);
    let coefficients_6 = coefficients_at(6, 0);
    let coefficients_8 = coefficients_at(8, 100);
    let words = |values: &[BaseField]| values.iter().map(|value| value.0).collect::<Vec<_>>();
    upload(&arena, SEQ_6, &words(&coefficients_6.preprocessed));
    upload(&arena, BASE_6, &words(&coefficients_6.base));
    for (slot, values) in INTERACTION_6.into_iter().zip(&coefficients_6.interaction) {
        upload(&arena, slot, &words(values));
    }
    upload(&arena, SEQ_8, &words(&coefficients_8.preprocessed));
    upload(&arena, BASE_8, &words(&coefficients_8.base));
    for (slot, values) in INTERACTION_8.into_iter().zip(&coefficients_8.interaction) {
        upload(&arena, slot, &words(values));
    }
    arena.context().sync().unwrap();

    // CPU reference: per-component pointwise accumulation with the global
    // descending random powers, the device lift's exact index map, then the
    // shared interpolate + split tail.
    let total_constraints = constraints_6 + constraints_8;
    let random_powers = (0..total_constraints)
        .map(|index| random_coefficient.pow((total_constraints - 1 - index) as u128))
        .collect::<Vec<_>>();
    let accumulation_6 = accumulate_pointwise_cpu(
        &component_6,
        component_evaluations(&coefficients_6, eval_log_6).as_cols_ref(),
        eval_log_6,
        range_check_6::LOG_SIZE,
        plan.components[0].denominator_inverses.clone(),
        &random_powers[..constraints_6],
        &SecureColumnByCoords::<CpuBackend>::zeros(1 << eval_log_6),
    );
    let mut accumulation = accumulate_pointwise_cpu(
        &component_8,
        component_evaluations(&coefficients_8, eval_log_8).as_cols_ref(),
        eval_log_8,
        range_check_8::LOG_SIZE,
        plan.components[1].denominator_inverses.clone(),
        &random_powers[constraints_6..],
        &SecureColumnByCoords::<CpuBackend>::zeros(1 << eval_log_8),
    );
    let log_ratio = eval_log_8 - eval_log_6;
    for index in 0..1usize << eval_log_8 {
        let lifted = (index >> (log_ratio + 1) << 1) + (index & 1);
        accumulation.set(index, accumulation.at(index) + accumulation_6.at(lifted));
    }
    let expected = interpolate_and_split(accumulation, eval_log_8);

    let inputs = CompositionDeviceInputs {
        random_coefficient: RANDOM,
        forward_twiddles: arena.bind(FORWARD).unwrap(),
        inverse_twiddles: arena.bind(INVERSE).unwrap(),
        relation_z: arena.bind(Z).unwrap(),
        relation_alpha_powers: arena.bind(ALPHA).unwrap(),
        claimed_sums: vec![None, None],
        ext_params: vec![
            Some(CompositionExtParamBinding {
                slot: EXT_6,
                offset_words: 0,
            }),
            Some(CompositionExtParamBinding {
                slot: EXT_8,
                offset_words: 0,
            }),
        ],
    };
    let mut replays = Vec::new();
    for mode in [CompositionLaunchMode::Serial, CompositionLaunchMode::Wide] {
        let prepared = PreparedCompositionGraph::prepare_with_mode(
            &arena, &plan, &trace, &inputs, &slots, mode,
        )
        .unwrap();
        prepared.launch().unwrap();
        let eager = read_outputs(&arena, &slots);
        arena.context().sync().unwrap();
        assert_eq!(eager, expected, "{mode:?} eager output mismatch");

        let capture = arena.context().capture().unwrap();
        prepared.launch().unwrap();
        let graph = capture.finish().unwrap();
        graph.launch(arena.context()).unwrap();
        let replay = read_outputs(&arena, &slots);
        arena.context().sync().unwrap();
        assert_eq!(replay, expected, "{mode:?} captured replay mismatch");
        replays.push(replay);
    }
    assert_eq!(
        replays[0], replays[1],
        "serial and wide modes must be byte-identical"
    );
}
