//! Native eager/capture parity for the complete prepared composition graph.

#![cfg(stwo_cuda_link)]

use core::ffi::c_void;

use cairo_air::components::range_check_6;
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
    composition_workspace_requirements, CompositionCoefficientSource, CompositionDeviceInputs,
    CompositionExtParamBinding, CompositionTraceTopology, CompositionWorkspaceRequirements,
    CompositionWorkspaceSlots, PreparedCompositionGraph,
};
use stwo_constraint_framework::{
    accumulate_pointwise_cpu, FrameworkComponent, TraceLocationAllocator,
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

fn real_component_and_plan() -> (RangeCheckComponent, CompositionPlan) {
    let lookup = CommonLookupElements::from_z_alpha(
        SecureField::from_u32_unchecked(2, 3, 5, 7),
        SecureField::from_u32_unchecked(11, 13, 17, 19),
    );
    let claimed_sum = SecureField::from_u32_unchecked(23, 29, 31, 37);
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(&[
        stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId {
            id: "seq_6".to_owned(),
        },
    ]);
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
    let emitted = aot::constraint_program(
        component.evaluator(),
        3,
        component.claimed_sum(),
        TRACE_LOG_SIZE,
        max_kernel_instrs,
    )
    .expect("range_check_6 must lower into the embedded AOT pack");
    assert_eq!(emitted.kernels.len(), 1);
    assert_eq!(emitted.kernels[0].kernel.cache_key, 0x5754_f8a8_73a5_2740);

    let mut denominator_inverses = (0..1usize << (EVALUATION_LOG_SIZE - TRACE_LOG_SIZE))
        .map(|index| {
            coset_vanishing(
                CanonicCoset::new(TRACE_LOG_SIZE).coset(),
                CanonicCoset::new(EVALUATION_LOG_SIZE)
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
    let total_constraints = component.n_constraints();
    let component_plan = CompositionComponentPlan {
        component: "range_check_6",
        instance: 0,
        trace_locations: component.trace_locations().to_vec(),
        preprocessed_column_indices: component.preprocessed_column_indices().to_vec(),
        trace_log_size: TRACE_LOG_SIZE,
        evaluation_log_size: component.max_constraint_log_degree_bound(),
        n_constraints: total_constraints,
        random_coefficient_offset: 0,
        denominator_inverses,
        ext_param_values,
        ext_param_sources,
        kernels,
    };
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
    ext_param_words: usize,
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
    requested.extend([
        (PREPROCESSED, 1 << TRACE_LOG_SIZE, 1),
        (BASE, 1 << TRACE_LOG_SIZE, 1),
        (INTERACTION_0, 1 << TRACE_LOG_SIZE, 1),
        (INTERACTION_1, 1 << TRACE_LOG_SIZE, 1),
        (INTERACTION_2, 1 << TRACE_LOG_SIZE, 1),
        (INTERACTION_3, 1 << TRACE_LOG_SIZE, 1),
        (RANDOM_COEFFICIENT, SECURE_WORDS, SECURE_WORDS),
        (FORWARD_TWIDDLES, requirements.forward_twiddle_words, 1),
        (INVERSE_TWIDDLES, requirements.inverse_twiddle_words, 1),
        (RELATION_Z, SECURE_WORDS, SECURE_WORDS),
        (RELATION_ALPHA_POWERS, SECURE_WORDS, SECURE_WORDS),
        (EXT_PARAMS, ext_param_words, SECURE_WORDS),
    ]);
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

fn sequence_coefficients() -> Vec<BaseField> {
    CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
        CanonicCoset::new(TRACE_LOG_SIZE).circle_domain(),
        (0..1usize << TRACE_LOG_SIZE).map(BaseField::from).collect(),
    )
    .interpolate()
    .coeffs
}

fn coefficients(seed: u32) -> Coefficients {
    let column = |factor: u32, offset: u32| {
        (0..1usize << TRACE_LOG_SIZE)
            .map(|index| {
                BaseField::from_u32_unchecked(
                    factor.wrapping_mul(index as u32).wrapping_add(offset) & 0x7fff_ffff,
                )
            })
            .collect::<Vec<_>>()
    };
    Coefficients {
        preprocessed: sequence_coefficients(),
        base: column(17 + seed, 3 + seed),
        interaction: [
            column(29 + seed, 5),
            column(43 + seed, 7),
            column(71 + seed, 11),
            column(101 + seed, 13),
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
    let full = accumulation.columns.map(|values| {
        CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(evaluation_domain, values)
            .interpolate()
            .coeffs
            .into_iter()
            .map(|value| value.0)
            .collect::<Vec<_>>()
    });
    let half = 1usize << (EVALUATION_LOG_SIZE - 1);
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
    let arena = arena(&requirements, &slots, ext_param_words);
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
    let domain = CanonicCoset::new(EVALUATION_LOG_SIZE).circle_domain();
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

    let prepared = PreparedCompositionGraph::prepare(
        &arena,
        &plan,
        &trace,
        &CompositionDeviceInputs {
            random_coefficient: RANDOM_COEFFICIENT,
            forward_twiddles: FORWARD_TWIDDLES,
            inverse_twiddles: INVERSE_TWIDDLES,
            relation_z: RELATION_Z,
            relation_alpha_powers: RELATION_ALPHA_POWERS,
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
