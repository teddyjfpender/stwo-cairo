//! Bounded source/JIT parity for an exact 18-wave composition topology.
//!
//! This is a correctness gate, not a benchmark. It binds the production Wave
//! path, compares every output byte with Serial and CPU, replays both captured
//! graphs after input mutation, and protects arena tails. The fixture explicitly
//! keeps strict AOT admission open and therefore grants no production-pack
//! promotion credit; the current-head SN pack has a separate source-free gate.

#![cfg(stwo_cuda_link)]

#[path = "prepared_composition_wave_fixture/device.rs"]
mod device;
use std::collections::BTreeMap;
use std::ffi::CString;

pub(crate) use device::{assert_guards, inputs, ready_arena, refresh};
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
    aot, ArenaLayout, ArenaSlice, ArenaSlotId, ArenaSlotSpec, CudaExecContext, DeviceArena,
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
use stwo_cairo_gpu_prover::prepared_composition::CompositionDirectEvaluationBinding;
use stwo_cairo_gpu_prover::{
    CompositionCoefficientSource, CompositionDeviceInputs, CompositionExtParamBinding,
    CompositionOutputSlots, CompositionTraceTopology, CompositionWorkspaceRequirements,
    CompositionWorkspaceSlots, PreparedCompositionGraph,
};
use stwo_cairo_prover::witness::proof_shape::TracePartId;
use stwo_constraint_framework::{
    accumulate_pointwise_cpu, EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator,
};

pub(super) const WAVE_COUNT: usize = 18;
const FIRST_TRACE_LOG: u32 = 2;
pub(super) const MAX_EVALUATION_LOG: u32 = FIRST_TRACE_LOG + WAVE_COUNT as u32;
const EXTRA_TRACE_LOGS: [u32; 4] = [6, 6, 10, 15];
const COMPONENT_COUNT: usize = WAVE_COUNT + EXTRA_TRACE_LOGS.len();
const CONSTRAINTS_PER_COMPONENT: usize = 2;
const GUARD_WORDS: usize = 64;
pub(super) const GUARD: u32 = 0xdead_beef;
const TRACE_BASE: u32 = 1_000;
const TRACE_INTERACTION: u32 = 1_100;
const DIRECT_BASE: u32 = 1_200;
const DIRECT_INTERACTION: u32 = 1_300;
pub(super) const RANDOM: ArenaSlotId = ArenaSlotId(1_400);
pub(super) const FORWARD: ArenaSlotId = ArenaSlotId(1_401);
pub(super) const INVERSE: ArenaSlotId = ArenaSlotId(1_402);
pub(super) const Z: ArenaSlotId = ArenaSlotId(1_403);
pub(super) const ALPHA: ArenaSlotId = ArenaSlotId(1_404);
const EXT_BASE: u32 = 1_500;
const JIT_MAX_KERNEL_INSTRS: usize = 2_048;
pub(super) const EAGER_Z: SecureField = SecureField::from_u32_unchecked(23, 29, 31, 37);
pub(super) const EAGER_ALPHA: SecureField = SecureField::from_u32_unchecked(41, 43, 47, 53);
pub(super) const REPLAY_Z: SecureField = SecureField::from_u32_unchecked(59, 61, 67, 71);
pub(super) const REPLAY_ALPHA: SecureField = SecureField::from_u32_unchecked(73, 79, 83, 89);

#[derive(Clone, Copy)]
struct BoundedEval {
    log_size: u32,
    z: SecureField,
    alpha_power: SecureField,
}

impl FrameworkEval for BoundedEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let base = eval.next_trace_mask();
        let [interaction] = eval.next_interaction_mask(2, [0]);
        eval.add_constraint(base * self.z);
        eval.add_constraint(interaction * self.alpha_power);
        eval
    }
}

type BoundedComponent = FrameworkComponent<BoundedEval>;

pub(super) struct Fixture {
    pub(super) plan: CompositionPlan,
    pub(super) trace: CompositionTraceTopology,
    pub(super) logs: Vec<u32>,
}

pub(super) struct Data {
    coefficients: [Vec<Vec<BaseField>>; 2],
    evaluations: [Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>; 2],
}

pub(super) struct ReadyArena {
    pub(super) arena: DeviceArena,
    pub(super) data: Data,
    pub(super) direct: Vec<CompositionDirectEvaluationBinding>,
}

pub(super) struct Twiddles {
    forward: Vec<u32>,
    inverse: Vec<u32>,
}

fn slot(base: u32, index: usize) -> ArenaSlotId {
    ArenaSlotId(base + index as u32)
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

fn component_logs() -> Vec<u32> {
    (FIRST_TRACE_LOG..FIRST_TRACE_LOG + WAVE_COUNT as u32)
        .chain(EXTRA_TRACE_LOGS)
        .collect()
}

fn components(logs: &[u32], z: SecureField, alpha_power: SecureField) -> Vec<BoundedComponent> {
    let mut allocator = TraceLocationAllocator::default();
    logs.iter()
        .map(|&log_size| {
            FrameworkComponent::new(
                &mut allocator,
                BoundedEval {
                    log_size,
                    z,
                    alpha_power,
                },
                SecureField::default(),
            )
        })
        .collect()
}

fn expected_wave_starts(evaluation_log_size: u32) -> Vec<u32> {
    match evaluation_log_size {
        7 => vec![8, 36, 38],
        11 => vec![16, 40],
        16 => vec![26, 42],
        log => vec![2 * (log - (FIRST_TRACE_LOG + 1))],
    }
}

fn lower(
    component: &BoundedComponent,
    instance: usize,
    coefficient_start: usize,
    cap: usize,
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
        component.evaluator().log_size(),
        cap,
    )
    .unwrap();
    assert_eq!(
        emitted.kernels.len(),
        1,
        "bounded component must remain one part"
    );
    let part = emitted.kernels.into_iter().next().unwrap();
    assert_eq!(part.rc_base, 0);
    let identity = aot::CompositionWaveKernelPartIdentity {
        semantic_hash: part.kernel.semantic_hash,
        coefficient_start: coefficient_start as u32,
        coefficient_end: (coefficient_start + component.n_constraints()) as u32,
    };
    assert_eq!(component.n_constraints(), CONSTRAINTS_PER_COMPONENT);
    assert_eq!(
        emitted.ext_param_values,
        vec![component.evaluator().z, component.evaluator().alpha_power]
    );
    let ext_param_sources = vec![
        CompositionExtParamSource::LookupZ,
        CompositionExtParamSource::LookupAlphaPower(0),
    ];
    let kernel = CompositionKernelPart {
        kernel_name: part.kernel.kernel_name,
        cache_key: part.kernel.cache_key,
        semantic_hash: part.kernel.semantic_hash,
        source: part.kernel.source,
        rc_base: part.rc_base,
    };
    let trace_log_size = component.evaluator().log_size();
    let evaluation_log_size = component.max_constraint_log_degree_bound();
    (
        CompositionComponentPlan {
            component: "bounded_wave",
            instance,
            trace_locations: component.trace_locations().to_vec(),
            preprocessed_column_indices: Vec::new(),
            trace_log_size,
            evaluation_log_size,
            n_constraints: component.n_constraints(),
            random_coefficient_offset: coefficient_start,
            denominator_inverses: denominator_inverses(trace_log_size, evaluation_log_size),
            base_param_values: emitted.base_param_values,
            ext_param_values: emitted.ext_param_values,
            ext_param_sources,
            kernels: vec![kernel],
        },
        (identity, part.wave_fragment),
    )
}

pub(super) fn fixture() -> Fixture {
    let cap = JIT_MAX_KERNEL_INSTRS;
    let logs = component_logs();
    let components = components(&logs, EAGER_Z, EAGER_ALPHA);
    let mut plans = Vec::with_capacity(COMPONENT_COUNT);
    let mut wave_parts = BTreeMap::<
        u32,
        Vec<(
            aot::CompositionWaveKernelPartIdentity,
            aot::ConstraintWaveFragment,
        )>,
    >::new();
    let mut coefficient_start = 0usize;
    for (instance, component) in components.iter().enumerate() {
        let (plan, part) = lower(component, instance, coefficient_start, cap);
        coefficient_start += plan.n_constraints;
        wave_parts
            .entry(plan.evaluation_log_size)
            .or_default()
            .push(part);
        plans.push(plan);
    }
    let waves = wave_parts
        .into_iter()
        .map(|(evaluation_log_size, parts)| {
            let identities = parts.iter().map(|part| part.0).collect::<Vec<_>>();
            assert_eq!(
                identities
                    .iter()
                    .map(|part| part.coefficient_start)
                    .collect::<Vec<_>>(),
                expected_wave_starts(evaluation_log_size),
                "hard-sealed part order at evaluation log {evaluation_log_size}"
            );
            let emitted = aot::composition_wave_kernel_source(evaluation_log_size, &parts).unwrap();
            let expected =
                aot::composition_wave_kernel_identity(evaluation_log_size, &identities).unwrap();
            assert_eq!(
                (
                    emitted.kernel_name.as_str(),
                    emitted.cache_key,
                    emitted.semantic_hash
                ),
                (
                    expected.kernel_name.as_str(),
                    expected.cache_key,
                    expected.semantic_hash
                )
            );
            assert_eq!(
                emitted.abi_schema,
                Some(aot::AotKernelAbiSchema::CompositionWaveV2)
            );
            let program_identity = emitted.program_identity.unwrap();
            assert_ne!(program_identity, [0; 32]);
            CompositionWaveKernelPlan {
                evaluation_log_size,
                parts: identities,
                kernel_name: emitted.kernel_name,
                cache_key: emitted.cache_key,
                semantic_hash: emitted.semantic_hash,
                program_identity,
                source: emitted.source,
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(waves.len(), WAVE_COUNT);
    assert!(waves.iter().any(|wave| wave.parts.len() > 1));
    let trace = CompositionTraceTopology {
        trees: vec![
            Vec::new(),
            logs.iter()
                .enumerate()
                .map(|(index, &log_size)| CompositionCoefficientSource {
                    slot: slot(TRACE_BASE, index),
                    log_size,
                })
                .collect(),
            logs.iter()
                .enumerate()
                .map(|(index, &log_size)| CompositionCoefficientSource {
                    slot: slot(TRACE_INTERACTION, index),
                    log_size,
                })
                .collect(),
        ],
    };
    Fixture {
        plan: CompositionPlan {
            max_kernel_instrs: cap,
            total_constraints: coefficient_start,
            max_evaluation_log_size: MAX_EVALUATION_LOG,
            components: plans,
            wave_kernels: waves,
        },
        trace,
        logs,
    }
}

pub(super) fn retention(logs: &[u32]) -> DirectCompositionRetentionPlan {
    let mut columns = Vec::with_capacity(2 * logs.len());
    let mut bindings = Vec::with_capacity(2 * logs.len());
    for (index, &log_size) in logs.iter().enumerate() {
        for (tree, purpose, first_epoch) in [
            (
                CommitmentTreeId::Base,
                BufferPurpose::BaseCoefficients,
                ProofEpoch::BaseCommit,
            ),
            (
                CommitmentTreeId::Interaction,
                BufferPurpose::InteractionCoefficients,
                ProofEpoch::InteractionCommit,
            ),
        ] {
            let column = columns.len();
            columns.push(DirectCompositionColumn {
                source: OpenedColumnSource::Trace {
                    component: "bounded_wave",
                    part: TracePartId::Main,
                    purpose,
                    ordinal: index as u32,
                },
                tree,
                proof_column: index,
                group: 0,
                column_in_group: index,
                canonical_column: index,
                coefficient_log_size: log_size,
                evaluation_log_size: log_size + 1,
                lifetime: BufferLifetime::new(first_epoch, ProofEpoch::Composition).unwrap(),
            });
            bindings.push(DirectCompositionBinding {
                consumer: bindings.len(),
                column,
                consumer_evaluation_log_size: log_size + 1,
                direct: true,
            });
        }
    }
    let occurrence_count = bindings.len();
    let direct_column_count = columns.len();
    assert!(occurrence_count < 64);
    let mut plan = DirectCompositionRetentionPlan {
        direct_bytes: columns
            .iter()
            .map(|column| (1usize << column.evaluation_log_size) * 4)
            .sum(),
        columns,
        bindings,
        direct_bitmap: vec![(1u64 << occurrence_count) - 1],
        buckets: Vec::new(),
        direct_column_count,
        cache_key: 0,
    };
    plan.cache_key = direct_composition_plan_key(&plan);
    plan
}

pub(super) fn workspace_slots() -> CompositionWorkspaceSlots {
    CompositionWorkspaceSlots {
        descriptors: ArenaSlotId(1),
        lde_tile: ArenaSlotId(2),
        accumulators: ArenaSlotId(3),
        random_coefficient_powers: ArenaSlotId(4),
        output: CompositionOutputSlots::CoefficientSplit(std::array::from_fn(|i| {
            ArenaSlotId(5 + i as u32)
        })),
    }
}

pub(super) fn arena(
    requirements: &CompositionWorkspaceRequirements,
    slots: &CompositionWorkspaceSlots,
    logs: &[u32],
) -> DeviceArena {
    let mut requested = requirements
        .arena_slot_requirements(slots)
        .unwrap()
        .into_iter()
        .map(|item| (item.id, item.len_words + GUARD_WORDS, item.alignment_words))
        .collect::<Vec<_>>();
    requested.extend(logs.iter().enumerate().flat_map(|(index, &log_size)| {
        [TRACE_BASE, TRACE_INTERACTION]
            .into_iter()
            .map(move |base| (slot(base, index), (1usize << log_size) + GUARD_WORDS, 1))
            .chain(
                [DIRECT_BASE, DIRECT_INTERACTION]
                    .into_iter()
                    .map(move |base| {
                        (
                            slot(base, index),
                            (1usize << (log_size + 1)) + GUARD_WORDS,
                            1,
                        )
                    }),
            )
    }));
    requested.extend((0..logs.len()).map(|index| (slot(EXT_BASE, index), 8 + GUARD_WORDS, 4)));
    requested.extend([
        (RANDOM, 4 + GUARD_WORDS, 4),
        (FORWARD, requirements.forward_twiddle_words + GUARD_WORDS, 1),
        (INVERSE, requirements.inverse_twiddle_words + GUARD_WORDS, 1),
        (Z, 4 + GUARD_WORDS, 4),
        (ALPHA, 4 + GUARD_WORDS, 4),
    ]);
    let mut offset = 0usize;
    let specs = requested
        .into_iter()
        .map(|(id, len_words, alignment_words)| {
            offset = offset.next_multiple_of(alignment_words);
            let spec = ArenaSlotSpec {
                id,
                offset_words: offset,
                len_words,
                alignment_words,
            };
            offset += len_words;
            spec
        })
        .collect::<Vec<_>>();
    DeviceArena::new(
        CudaExecContext::new().unwrap(),
        ArenaLayout::new(offset, &specs).unwrap(),
    )
    .unwrap()
}

pub(super) fn fill(arena: &DeviceArena, slice: ArenaSlice, word: u32) {
    unsafe {
        arena
            .context()
            .fill_u32_async(slice.as_u32_ptr(), word, slice.len_words())
            .unwrap()
    }
}

pub(super) fn upload(arena: &DeviceArena, slice: ArenaSlice, words: &[u32]) {
    assert!(words.len() <= slice.len_words());
    unsafe {
        arena
            .context()
            .memcpy_h2d_async(slice.as_void_ptr(), words.as_ptr().cast(), words.len() * 4)
            .unwrap()
    }
    // Callers deliberately use temporary host word vectors; finish the copy
    // before their backing storage can be dropped or reused.
    arena.context().sync().unwrap();
}

fn read(arena: &DeviceArena, slice: ArenaSlice) -> Vec<u32> {
    let mut words = vec![0; slice.len_words()];
    unsafe {
        arena
            .context()
            .memcpy_d2h_async(
                words.as_mut_ptr().cast(),
                slice.as_void_ptr().cast_const(),
                words.len() * 4,
            )
            .unwrap()
    }
    arena.context().sync().unwrap();
    words
}

pub(super) fn data(logs: &[u32], seed: u32) -> Data {
    let coefficients = std::array::from_fn(|tree| {
        logs.iter()
            .enumerate()
            .map(|(column, &log_size)| {
                (0..1usize << log_size)
                    .map(|row| {
                        BaseField::from_u32_unchecked(
                            ((row as u32)
                                .wrapping_mul(17 + column as u32 + 31 * tree as u32)
                                .wrapping_add(seed + 3 * column as u32 + 101 * tree as u32))
                                % 0x7fff_ffff,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    let evaluations = std::array::from_fn(|tree| {
        coefficients[tree]
            .iter()
            .zip(logs)
            .map(|(values, &log_size)| {
                CpuCirclePoly::new(values.clone())
                    .evaluate(CanonicCoset::new(log_size + 1).circle_domain())
            })
            .collect()
    });
    Data {
        coefficients,
        evaluations,
    }
}

pub(super) fn upload_data(
    arena: &DeviceArena,
    data: &Data,
    logs: &[u32],
) -> Vec<CompositionDirectEvaluationBinding> {
    for (tree, (trace_base, direct_base)) in [
        (TRACE_BASE, DIRECT_BASE),
        (TRACE_INTERACTION, DIRECT_INTERACTION),
    ]
    .into_iter()
    .enumerate()
    {
        for (index, (&log_size, (coefficients, evaluations))) in logs
            .iter()
            .zip(data.coefficients[tree].iter().zip(&data.evaluations[tree]))
            .enumerate()
        {
            let coefficient_slot = arena.bind(slot(trace_base, index)).unwrap();
            let direct_slot = arena.bind(slot(direct_base, index)).unwrap();
            fill(arena, coefficient_slot, GUARD);
            fill(arena, direct_slot, GUARD);
            upload(
                arena,
                coefficient_slot,
                &coefficients.iter().map(|value| value.0).collect::<Vec<_>>(),
            );
            upload(
                arena,
                direct_slot,
                &evaluations
                    .values
                    .iter()
                    .map(|value| value.0)
                    .collect::<Vec<_>>(),
            );
            debug_assert_eq!(evaluations.values.len(), 1usize << (log_size + 1));
        }
    }
    arena.context().sync().unwrap();
    logs.iter()
        .enumerate()
        .flat_map(|(index, &log_size)| {
            [DIRECT_BASE, DIRECT_INTERACTION]
                .into_iter()
                .enumerate()
                .map(move |(tree, base)| CompositionDirectEvaluationBinding {
                    plan_column: 2 * index + tree,
                    evaluation: arena
                        .bind(slot(base, index))
                        .unwrap()
                        .truncated(1usize << (log_size + 1)),
                })
        })
        .collect()
}

pub(super) fn expected(
    fixture: &Fixture,
    data: &Data,
    random: SecureField,
    z: SecureField,
    alpha_power: SecureField,
) -> [Vec<u32>; 8] {
    let powers = (0..fixture.plan.total_constraints)
        .map(|index| random.pow((fixture.plan.total_constraints - 1 - index) as u128))
        .collect::<Vec<_>>();
    let mut accumulation = SecureColumnByCoords::<CpuBackend>::zeros(1 << MAX_EVALUATION_LOG);
    for (index, component) in components(&fixture.logs, z, alpha_power).iter().enumerate() {
        let evaluation_log = fixture.logs[index] + 1;
        let columns = TreeVec::new(vec![
            Vec::new(),
            vec![data.evaluations[0][index].clone()],
            vec![data.evaluations[1][index].clone()],
        ]);
        let component_plan = &fixture.plan.components[index];
        let coefficient_end =
            component_plan.random_coefficient_offset + component_plan.n_constraints;
        let local = accumulate_pointwise_cpu(
            component,
            columns.as_cols_ref(),
            evaluation_log,
            fixture.logs[index],
            component_plan.denominator_inverses.clone(),
            &powers[component_plan.random_coefficient_offset..coefficient_end],
            &SecureColumnByCoords::<CpuBackend>::zeros(1 << evaluation_log),
        );
        let ratio = MAX_EVALUATION_LOG - evaluation_log;
        for row in 0..1usize << MAX_EVALUATION_LOG {
            let lifted = (row >> (ratio + 1) << 1) + (row & 1);
            accumulation.set(row, accumulation.at(row) + local.at(lifted));
        }
    }
    let domain = CanonicCoset::new(MAX_EVALUATION_LOG).circle_domain();
    let full = accumulation.columns.map(|values| {
        CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(domain, values)
            .interpolate()
            .coeffs
            .into_iter()
            .map(|value| value.0)
            .collect::<Vec<_>>()
    });
    let half = 1usize << (MAX_EVALUATION_LOG - 1);
    std::array::from_fn(|index| full[index % 4][usize::from(index >= 4) * half..][..half].to_vec())
}

pub(super) fn outputs(
    arena: &DeviceArena,
    prepared: &PreparedCompositionGraph<'_>,
) -> [Vec<u32>; 8] {
    prepared
        .composition_coefficients()
        .unwrap()
        .map(|slice| read(arena, slice))
}

pub(super) fn twiddles() -> Twiddles {
    let domain = CanonicCoset::new(MAX_EVALUATION_LOG).circle_domain();
    Twiddles {
        forward: slow_precompute_twiddles(domain.half_coset)
            .into_iter()
            .map(|value| value.0)
            .collect(),
        inverse: slow_precompute_twiddles(domain.half_coset)
            .into_iter()
            .map(|value| value.inverse().0)
            .collect(),
    }
}

pub(super) fn install_jit_wave_sources(plan: &CompositionPlan) {
    assert_eq!(plan.wave_kernels.len(), WAVE_COUNT);
    let logs = plan
        .wave_kernels
        .iter()
        .map(|wave| wave.evaluation_log_size)
        .collect::<Vec<_>>();
    assert_eq!(
        logs,
        (FIRST_TRACE_LOG + 1..=MAX_EVALUATION_LOG).collect::<Vec<_>>()
    );
    aot::reset_runtime_stats();
    for wave in &plan.wave_kernels {
        assert_eq!(
            wave.parts
                .iter()
                .map(|part| part.coefficient_start)
                .collect::<Vec<_>>(),
            expected_wave_starts(wave.evaluation_log_size)
        );
        let identity =
            aot::composition_wave_kernel_identity(wave.evaluation_log_size, &wave.parts).unwrap();
        assert_eq!(
            (
                wave.kernel_name.as_str(),
                wave.cache_key,
                wave.semantic_hash
            ),
            (
                identity.kernel_name.as_str(),
                identity.cache_key,
                identity.semantic_hash
            )
        );
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
    assert_eq!(stats.runtime_loads, WAVE_COUNT as u64);
    assert!(stats.runtime_cache_hits >= WAVE_COUNT as u64);
    assert_eq!(stats.strict_rejections, 0);
}
