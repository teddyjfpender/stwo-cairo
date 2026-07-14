//! Exact AOT composition-kernel plan in Cairo protocol order.
//!
//! This is the host-only preparation half of the resident composition graph. It
//! lowers each concrete Cairo evaluator once, records the embedded-AOT identity,
//! preserves its base- and extension-parameter slot order, and assigns the exact
//! descending random-coefficient range consumed by STWO's
//! `DomainEvaluationAccumulator`.

use cairo_air::cairo_components::CairoComponents;
use cairo_air::claims::{CairoClaim, CairoInteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::air::Component;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeSubspan;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse;
use stwo_backend_cuda::aot::{constraint_program, EmittedConstraintKernel};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{FrameworkComponent, FrameworkEval};

const LOOKUP_PROBE_Z: SecureField = SecureField::from_u32_unchecked(17, 29, 43, 71);
const LOOKUP_PROBE_ALPHA: SecureField = SecureField::from_u32_unchecked(101, 131, 173, 211);
const CLAIMED_SUM_PROBE: SecureField = SecureField::from_u32_unchecked(257, 263, 269, 271);

mod bindings;
mod schema;
pub use bindings::CompositionProofBindings;
pub(crate) use bindings::{
    bind_cairo_composition, compile_cairo_composition_binding_plan, CompositionBindingPlan,
};

/// One AOT launch part. Several parts may contribute to the same component
/// accumulation; setup resolves every key before any part may launch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionKernelPart {
    pub kernel_name: String,
    pub cache_key: u64,
    pub semantic_hash: u64,
    pub source: String,
    pub rc_base: u32,
}

/// Exact prepared inputs for one concrete Cairo component instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionComponentPlan {
    pub component: &'static str,
    pub instance: usize,
    pub trace_locations: Vec<TreeSubspan>,
    pub preprocessed_column_indices: Vec<usize>,
    pub trace_log_size: u32,
    pub evaluation_log_size: u32,
    pub n_constraints: usize,
    /// Offset into the proof-global descending power array. The first Cairo
    /// component starts at zero, matching the accumulator's split/reverse order.
    pub random_coefficient_offset: usize,
    pub denominator_inverses: Vec<BaseField>,
    /// Statement values for the base-field parameter slots encoded by every
    /// kernel. They are runtime bindings and intentionally do not affect the
    /// kernel identity or the reusable composition-plan key.
    pub base_param_values: Vec<BaseField>,
    /// Setup values used only by differential/reference lanes. Resident replay
    /// binds the corresponding typed sources below directly on device.
    pub ext_param_values: Vec<SecureField>,
    pub ext_param_sources: Vec<CompositionExtParamSource>,
    pub kernels: Vec<CompositionKernelPart>,
}

/// Provenance of one hoisted extension constant in the generated AOT ABI.
/// Two structurally identical recordings with independent lookup/claim probes
/// classify every statement-dependent slot; anything not invariant or one of
/// these exact sources fails planning instead of becoming a stale host value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionExtParamSource {
    Constant(SecureField),
    LookupZ,
    LookupAlphaPower(u32),
    LookupAlphaPowerScaled { power: u32, scale: BaseField },
    ClaimedSumScaled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionPlan {
    pub max_kernel_instrs: usize,
    pub total_constraints: usize,
    pub max_evaluation_log_size: u32,
    pub components: Vec<CompositionComponentPlan>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompositionPlanError {
    Empty,
    InvalidMaxKernelInstructions,
    ConstraintCountOverflow,
    KernelLowering {
        component: &'static str,
        instance: usize,
    },
    EmptyKernelProgram {
        component: &'static str,
        instance: usize,
    },
    ComponentOrderMismatch {
        expected: usize,
        actual: usize,
    },
    ProbeComponentPresence {
        component: &'static str,
        instance: usize,
    },
    ProbeKernelMismatch {
        component: &'static str,
        instance: usize,
    },
    ProbeBaseParamMismatch {
        component: &'static str,
        instance: usize,
    },
    AmbiguousBaseParamProbe {
        component: &'static str,
        instance: usize,
    },
    UnclassifiedBaseParam {
        component: &'static str,
        instance: usize,
        slot: usize,
        value: BaseField,
        first_probe_value: BaseField,
        second_probe_value: BaseField,
    },
    BaseParamProbeCountMismatch {
        component: &'static str,
        instance: usize,
        expected: usize,
        first_probe: usize,
        second_probe: usize,
    },
    MissingBaseParamSegment {
        component: &'static str,
        instance: usize,
        segment: &'static str,
    },
    BaseParamRecipeMismatch {
        sample: &'static str,
    },
    UnsupportedPublicDataBaseParam {
        component: &'static str,
        instance: usize,
        slot: usize,
        expected: BaseField,
        actual: BaseField,
    },
    BaseParamProbeValueExhausted,
    ProbeExtParamCountMismatch {
        component: &'static str,
        instance: usize,
        expected: usize,
        actual: usize,
    },
    AmbiguousExtParamProbe {
        component: &'static str,
        instance: usize,
    },
    UnclassifiedExtParam {
        component: &'static str,
        instance: usize,
        slot: usize,
        value: SecureField,
        probe_value: SecureField,
    },
    BindingTopologyDrift {
        component: &'static str,
        instance: usize,
    },
    BindingKernelMismatch {
        component: &'static str,
        instance: usize,
    },
}

impl core::fmt::Display for CompositionPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid resident composition plan: {self:?}")
    }
}

impl std::error::Error for CompositionPlanError {}

/// Build the exact composition program for one claim/interaction statement.
///
/// `max_kernel_instrs` must equal the AOT pack's generation cap. Strict runtime
/// admission checks each returned cache key against the loaded pack before graph
/// capture, so a fixture/shape absent from the pack fails before touching output.
pub fn plan_cairo_composition(
    claim: &CairoClaim,
    lookup_elements: &CommonLookupElements,
    interaction_claim: &CairoInteractionClaim,
    preprocessed_columns: &[PreProcessedColumnId],
    max_kernel_instrs: usize,
) -> Result<CompositionPlan, CompositionPlanError> {
    if max_kernel_instrs == 0 {
        return Err(CompositionPlanError::InvalidMaxKernelInstructions);
    }
    let cairo = CairoComponents::new(
        claim,
        lookup_elements,
        interaction_claim,
        preprocessed_columns,
    );
    let probe_lookup = CommonLookupElements::from_z_alpha(LOOKUP_PROBE_Z, LOOKUP_PROBE_ALPHA);
    let probe_cairo = CairoComponents::new(
        claim,
        &probe_lookup,
        interaction_claim,
        preprocessed_columns,
    );
    let erased = cairo.components();
    let total_constraints = erased.iter().try_fold(0usize, |total, component| {
        total.checked_add(component.n_constraints())
    });
    let total_constraints =
        total_constraints.ok_or(CompositionPlanError::ConstraintCountOverflow)?;
    if total_constraints == 0 {
        return Err(CompositionPlanError::Empty);
    }

    let mut components = Vec::with_capacity(erased.len());
    let mut consumed_constraints = 0usize;

    macro_rules! push_optional {
        ($( $field:ident ),+ $(,)?) => {
            $(
                if let Some(component) = &cairo.$field {
                    let probe_component = probe_cairo.$field.as_ref().ok_or(
                        CompositionPlanError::ProbeComponentPresence {
                            component: stringify!($field),
                            instance: 0,
                        },
                    )?;
                    push_component(
                        stringify!($field),
                        0,
                        component,
                        probe_component,
                        lookup_elements,
                        &probe_lookup,
                        max_kernel_instrs,
                        &mut consumed_constraints,
                        &mut components,
                    )?;
                } else if probe_cairo.$field.is_some() {
                    return Err(CompositionPlanError::ProbeComponentPresence {
                        component: stringify!($field),
                        instance: 0,
                    });
                }
            )+
        };
    }

    // This is deliberately byte-for-byte the order of
    // `CairoComponents::components`, which is also `cairo_provers` order.
    push_optional!(
        add_opcode,
        add_opcode_small,
        add_ap_opcode,
        assert_eq_opcode,
        assert_eq_opcode_imm,
        assert_eq_opcode_double_deref,
        blake_compress_opcode,
        call_opcode_abs,
        call_opcode_rel_imm,
        generic_opcode,
        jnz_opcode_non_taken,
        jnz_opcode_taken,
        jump_opcode_abs,
        jump_opcode_double_deref,
        jump_opcode_rel,
        jump_opcode_rel_imm,
        mul_opcode,
        mul_opcode_small,
        qm_31_add_mul_opcode,
        ret_opcode,
        verify_instruction,
        blake_round,
        blake_g,
        blake_round_sigma,
        triple_xor_32,
        verify_bitwise_xor_12,
        add_mod_builtin,
        bitwise_builtin,
        mul_mod_builtin,
        pedersen_builtin,
        pedersen_builtin_narrow_windows,
        poseidon_builtin,
        range_check96_builtin,
        range_check_builtin,
        ec_op_builtin,
        partial_ec_mul_generic,
        pedersen_aggregator_window_bits_18,
        partial_ec_mul_window_bits_18,
        pedersen_points_table_window_bits_18,
        pedersen_aggregator_window_bits_9,
        partial_ec_mul_window_bits_9,
        pedersen_points_table_window_bits_9,
        poseidon_aggregator,
        poseidon_3_partial_rounds_chain,
        poseidon_full_round_chain,
        cube_252,
        poseidon_round_keys,
        range_check_252_width_27,
        memory_address_to_id,
    );
    if cairo.memory_id_to_big.len() != probe_cairo.memory_id_to_big.len() {
        return Err(CompositionPlanError::ProbeComponentPresence {
            component: "memory_id_to_big",
            instance: cairo
                .memory_id_to_big
                .len()
                .min(probe_cairo.memory_id_to_big.len()),
        });
    }
    for (instance, (component, probe_component)) in cairo
        .memory_id_to_big
        .iter()
        .zip(&probe_cairo.memory_id_to_big)
        .enumerate()
    {
        push_component(
            "memory_id_to_big",
            instance,
            component,
            probe_component,
            lookup_elements,
            &probe_lookup,
            max_kernel_instrs,
            &mut consumed_constraints,
            &mut components,
        )?;
    }
    push_optional!(
        memory_id_to_small,
        range_check_6,
        range_check_8,
        range_check_11,
        range_check_12,
        range_check_18,
        range_check_20,
        range_check_4_3,
        range_check_4_4,
        range_check_9_9,
        range_check_7_2_5,
        range_check_3_6_6_3,
        range_check_4_4_4_4,
        range_check_3_3_3_3_3,
        verify_bitwise_xor_4,
        verify_bitwise_xor_7,
        verify_bitwise_xor_8,
        verify_bitwise_xor_9,
    );

    if components.len() != erased.len() || consumed_constraints != total_constraints {
        return Err(CompositionPlanError::ComponentOrderMismatch {
            expected: erased.len(),
            actual: components.len(),
        });
    }
    let max_evaluation_log_size = components
        .iter()
        .map(|component| component.evaluation_log_size)
        .max()
        .ok_or(CompositionPlanError::Empty)?;
    Ok(CompositionPlan {
        max_kernel_instrs,
        total_constraints,
        max_evaluation_log_size,
        components,
    })
}

impl CompositionPlan {
    /// Stable statement/topology identity for workspace and captured-graph
    /// reuse. Dynamic lookup elements and claimed sums are represented by their
    /// typed source tags; only truly constant extension parameters contribute
    /// values to this key.
    pub fn key(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        };
        feed(b"stwo-cairo-composition-plan-v2\0");
        feed(&(self.max_kernel_instrs as u64).to_le_bytes());
        feed(&(self.total_constraints as u64).to_le_bytes());
        feed(&self.max_evaluation_log_size.to_le_bytes());
        feed(&(self.components.len() as u64).to_le_bytes());
        for component in &self.components {
            feed(component.component.as_bytes());
            feed(&[0]);
            feed(&(component.instance as u64).to_le_bytes());
            feed(&component.trace_log_size.to_le_bytes());
            feed(&component.evaluation_log_size.to_le_bytes());
            feed(&(component.n_constraints as u64).to_le_bytes());
            feed(&(component.random_coefficient_offset as u64).to_le_bytes());
            for span in &component.trace_locations {
                feed(&(span.tree_index as u64).to_le_bytes());
                feed(&(span.col_start as u64).to_le_bytes());
                feed(&(span.col_end as u64).to_le_bytes());
            }
            feed(&[0xff]);
            for &column in &component.preprocessed_column_indices {
                feed(&(column as u64).to_le_bytes());
            }
            feed(&[0xfe]);
            for inverse in &component.denominator_inverses {
                feed(&inverse.0.to_le_bytes());
            }
            // The slot topology is load-bearing, while the statement values
            // are rebound whenever a resident session is prepared. Including
            // values here would prevent safe graph reuse across blocks.
            feed(&(component.base_param_values.len() as u64).to_le_bytes());
            for source in &component.ext_param_sources {
                match *source {
                    CompositionExtParamSource::Constant(value) => {
                        feed(&[0]);
                        for coordinate in value.to_m31_array() {
                            feed(&coordinate.0.to_le_bytes());
                        }
                    }
                    CompositionExtParamSource::LookupZ => feed(&[1]),
                    CompositionExtParamSource::LookupAlphaPower(power) => {
                        feed(&[2]);
                        feed(&power.to_le_bytes());
                    }
                    CompositionExtParamSource::ClaimedSumScaled => feed(&[3]),
                    CompositionExtParamSource::LookupAlphaPowerScaled { power, scale } => {
                        feed(&[4]);
                        feed(&power.to_le_bytes());
                        feed(&scale.0.to_le_bytes());
                    }
                }
            }
            feed(&[0xfd]);
            for kernel in &component.kernels {
                feed(kernel.kernel_name.as_bytes());
                feed(&[0]);
                feed(&kernel.cache_key.to_le_bytes());
                feed(&kernel.semantic_hash.to_le_bytes());
                feed(&kernel.rc_base.to_le_bytes());
                feed(&(kernel.source.len() as u64).to_le_bytes());
                feed(kernel.source.as_bytes());
            }
        }
        hash
    }
}

fn push_component<E: FrameworkEval>(
    name: &'static str,
    instance: usize,
    component: &FrameworkComponent<E>,
    probe_component: &FrameworkComponent<E>,
    lookup_elements: &CommonLookupElements,
    probe_lookup_elements: &CommonLookupElements,
    max_kernel_instrs: usize,
    consumed_constraints: &mut usize,
    output: &mut Vec<CompositionComponentPlan>,
) -> Result<(), CompositionPlanError> {
    let n_constraints = component.n_constraints();
    let program = constraint_program(
        component.evaluator(),
        3,
        component.claimed_sum(),
        component.evaluator().log_size(),
        max_kernel_instrs,
    )
    .ok_or(CompositionPlanError::KernelLowering {
        component: name,
        instance,
    })?;
    if program.kernels.is_empty() {
        return Err(CompositionPlanError::EmptyKernelProgram {
            component: name,
            instance,
        });
    }
    let probe_program = constraint_program(
        probe_component.evaluator(),
        3,
        CLAIMED_SUM_PROBE,
        probe_component.evaluator().log_size(),
        max_kernel_instrs,
    )
    .ok_or(CompositionPlanError::KernelLowering {
        component: name,
        instance,
    })?;
    if !same_kernel_program(&program.kernels, &probe_program.kernels) {
        return Err(CompositionPlanError::ProbeKernelMismatch {
            component: name,
            instance,
        });
    }
    if program.base_param_values != probe_program.base_param_values {
        return Err(CompositionPlanError::ProbeBaseParamMismatch {
            component: name,
            instance,
        });
    }
    let ext_param_sources = classify_ext_params(
        name,
        instance,
        component.evaluator().log_size(),
        component.claimed_sum(),
        lookup_elements,
        &program.ext_param_values,
        probe_lookup_elements,
        &probe_program.ext_param_values,
    )?;
    let trace_log_size = component.evaluator().log_size();
    let evaluation_log_size = component.max_constraint_log_degree_bound();
    let denominator_inverses = denominator_inverses(trace_log_size, evaluation_log_size);
    let kernels = program.kernels.into_iter().map(kernel_part).collect();
    output.push(CompositionComponentPlan {
        component: name,
        instance,
        trace_locations: component.trace_locations().to_vec(),
        preprocessed_column_indices: component.preprocessed_column_indices().to_vec(),
        trace_log_size,
        evaluation_log_size,
        n_constraints,
        random_coefficient_offset: *consumed_constraints,
        denominator_inverses,
        base_param_values: program.base_param_values,
        ext_param_values: program.ext_param_values,
        ext_param_sources,
        kernels,
    });
    *consumed_constraints = consumed_constraints
        .checked_add(n_constraints)
        .ok_or(CompositionPlanError::ConstraintCountOverflow)?;
    Ok(())
}

fn same_kernel_program(
    left: &[EmittedConstraintKernel],
    right: &[EmittedConstraintKernel],
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.rc_base == right.rc_base
                && left.kernel.cache_key == right.kernel.cache_key
                && left.kernel.semantic_hash == right.kernel.semantic_hash
                && left.kernel.kernel_name == right.kernel.kernel_name
                && left.kernel.source == right.kernel.source
        })
}

#[allow(clippy::too_many_arguments)]
fn classify_ext_params(
    component: &'static str,
    instance: usize,
    log_size: u32,
    claimed_sum: SecureField,
    lookup: &CommonLookupElements,
    values: &[SecureField],
    probe_lookup: &CommonLookupElements,
    probe_values: &[SecureField],
) -> Result<Vec<CompositionExtParamSource>, CompositionPlanError> {
    if values.len() != probe_values.len() {
        return Err(CompositionPlanError::ProbeExtParamCountMismatch {
            component,
            instance,
            expected: values.len(),
            actual: probe_values.len(),
        });
    }

    let mut dynamic = Vec::with_capacity(lookup.alpha_powers().len() + 2);
    dynamic.push((
        (lookup.z(), probe_lookup.z()),
        CompositionExtParamSource::LookupZ,
    ));
    for (power, (&value, &probe)) in lookup
        .alpha_powers()
        .iter()
        .zip(probe_lookup.alpha_powers())
        .enumerate()
        .skip(1)
    {
        dynamic.push((
            (value, probe),
            CompositionExtParamSource::LookupAlphaPower(
                u32::try_from(power).map_err(|_| CompositionPlanError::ConstraintCountOverflow)?,
            ),
        ));
    }
    let rows = BaseField::from_u32_unchecked(1u32 << log_size);
    dynamic.push((
        (claimed_sum / rows, CLAIMED_SUM_PROBE / rows),
        CompositionExtParamSource::ClaimedSumScaled,
    ));
    for (index, (pair, _)) in dynamic.iter().enumerate() {
        if pair.0 == pair.1
            || dynamic[..index]
                .iter()
                .any(|(candidate, _)| candidate == pair)
        {
            return Err(CompositionPlanError::AmbiguousExtParamProbe {
                component,
                instance,
            });
        }
    }

    values
        .iter()
        .copied()
        .zip(probe_values.iter().copied())
        .enumerate()
        .map(|(slot, pair)| {
            if pair.0 == pair.1 {
                return Ok(CompositionExtParamSource::Constant(pair.0));
            }
            let exact = dynamic
                .iter()
                .filter_map(|&(candidate, source)| (candidate == pair).then_some(source));
            let scaled = probe_lookup
                .alpha_powers()
                .iter()
                .copied()
                .enumerate()
                .skip(1)
                .filter_map(|(power_index, probe_power)| {
                    // Derive the scale from the fixed nonzero probe, then
                    // require the exact same embedded-M31 scale in the live
                    // recording. This accepts c*alpha^k but rejects general
                    // extension-field or affine functions of alpha.
                    if probe_power
                        .to_m31_array()
                        .iter()
                        .all(|coordinate| coordinate.0 == 0)
                    {
                        return None;
                    }
                    let coordinates = (pair.1 / probe_power).to_m31_array();
                    if coordinates[1..].iter().any(|coordinate| coordinate.0 != 0)
                        || coordinates[0].0 <= 1
                    {
                        return None;
                    }
                    let scale = coordinates[0];
                    let candidate = (
                        lookup.alpha_powers()[power_index] * scale,
                        probe_power * scale,
                    );
                    let power = u32::try_from(power_index).ok()?;
                    (candidate == pair).then_some(
                        CompositionExtParamSource::LookupAlphaPowerScaled { power, scale },
                    )
                });
            let mut matches = exact.chain(scaled).collect::<Vec<_>>();
            match matches.len() {
                0 => Err(CompositionPlanError::UnclassifiedExtParam {
                    component,
                    instance,
                    slot,
                    value: pair.0,
                    probe_value: pair.1,
                }),
                1 => Ok(matches.pop().expect("one classified source")),
                _ => Err(CompositionPlanError::AmbiguousExtParamProbe {
                    component,
                    instance,
                }),
            }
        })
        .collect()
}

fn kernel_part(part: EmittedConstraintKernel) -> CompositionKernelPart {
    CompositionKernelPart {
        kernel_name: part.kernel.kernel_name,
        cache_key: part.kernel.cache_key,
        semantic_hash: part.kernel.semantic_hash,
        source: part.kernel.source,
        rc_base: part.rc_base,
    }
}

fn denominator_inverses(trace_log_size: u32, evaluation_log_size: u32) -> Vec<BaseField> {
    let trace_domain = CanonicCoset::new(trace_log_size);
    let evaluation_domain = CanonicCoset::new(evaluation_log_size).circle_domain();
    let log_expand = evaluation_log_size - trace_log_size;
    let mut values = (0..1usize << log_expand)
        .map(|index| coset_vanishing(trace_domain.coset(), evaluation_domain.at(index)).inverse())
        .collect::<Vec<_>>();
    bit_reverse(&mut values);
    values
}

#[cfg(test)]
#[path = "composition_plan_tests.rs"]
mod tests;
