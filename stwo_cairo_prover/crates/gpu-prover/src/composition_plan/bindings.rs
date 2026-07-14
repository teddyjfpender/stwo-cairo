//! Cold-compiled statement binding for an immutable composition program.
//!
//! Cairo AIR evaluators expose only eight proof-varying base-field inputs: the
//! public builtin segment starts. `memory_id_to_big::BigEval::offset` is the
//! only other non-claim field and is derived solely from log sizes already in
//! `TopologyKey`. Cold compilation lowers two independently tagged statements
//! and admits a slot only when its value is invariant or is an exact tagged
//! segment start. A warm bind therefore reads the statement directly and never
//! reconstructs `CairoComponents` or records an evaluator.

use std::collections::BTreeSet;
use std::sync::Arc;

use cairo_air::cairo_components::CairoComponents;
use cairo_air::claims::{CairoClaim, CairoInteractionClaim};
use cairo_air::relations::CommonLookupElements;
use stwo::core::air::Component;
use stwo::core::fields::m31::BaseField;
use stwo_backend_cuda::aot::constraint_program_bindings;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{FrameworkComponent, FrameworkEval};

use super::schema::{assert_binding_schema_is_whitelisted, public_data_negative_control};
use super::{CompositionComponentPlan, CompositionPlan, CompositionPlanError};

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompositionComponentBinding {
    component: &'static str,
    instance: usize,
    base_param_offset: usize,
    base_param_words: usize,
}

/// The only proof-varying host values in an installed composition program.
/// Component descriptors are shared with the cold-compiled binding recipe;
/// each proof owns only its packed parameter words.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionProofBindings {
    components: Arc<[CompositionComponentBinding]>,
    base_param_values: Vec<BaseField>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SegmentStartSource {
    AddMod,
    Bitwise,
    MulMod,
    Pedersen,
    Poseidon,
    RangeCheck96,
    RangeCheck128,
    EcOp,
}

const SEGMENT_START_SOURCES: [SegmentStartSource; 8] = [
    SegmentStartSource::AddMod,
    SegmentStartSource::Bitwise,
    SegmentStartSource::MulMod,
    SegmentStartSource::Pedersen,
    SegmentStartSource::Poseidon,
    SegmentStartSource::RangeCheck96,
    SegmentStartSource::RangeCheck128,
    SegmentStartSource::EcOp,
];

impl SegmentStartSource {
    const fn name(self) -> &'static str {
        match self {
            Self::AddMod => "add_mod_builtin",
            Self::Bitwise => "bitwise_builtin",
            Self::MulMod => "mul_mod_builtin",
            Self::Pedersen => "pedersen_builtin",
            Self::Poseidon => "poseidon_builtin",
            Self::RangeCheck96 => "range_check96_builtin",
            Self::RangeCheck128 => "range_check_builtin",
            Self::EcOp => "ec_op_builtin",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BaseParamSource {
    Constant(BaseField),
    SegmentStart(SegmentStartSource),
}

#[derive(Clone, Copy)]
struct SegmentProbe {
    source: SegmentStartSource,
    original: BaseField,
    first: BaseField,
    second: BaseField,
}

/// Immutable base-parameter slot geometry and its fail-closed statement recipe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompositionBindingPlan {
    components: Arc<[CompositionComponentBinding]>,
    sources: Arc<[BaseParamSource]>,
}

impl CompositionProofBindings {
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    pub fn base_param_word_count(&self) -> usize {
        self.base_param_values.len()
    }

    pub(crate) fn component(&self, index: usize) -> Option<(&'static str, usize, &[BaseField])> {
        let binding = self.components.get(index)?;
        let end = binding
            .base_param_offset
            .checked_add(binding.base_param_words)?;
        Some((
            binding.component,
            binding.instance,
            self.base_param_values.get(binding.base_param_offset..end)?,
        ))
    }

    pub fn from_plan(plan: &CompositionPlan) -> Self {
        let (components, words) = component_bindings(plan)
            .expect("an allocated composition plan has representable binding geometry");
        let mut base_param_values = Vec::with_capacity(words);
        for component in &plan.components {
            base_param_values.extend_from_slice(&component.base_param_values);
        }
        Self {
            components,
            base_param_values,
        }
    }
}

impl CompositionBindingPlan {
    fn bind(&self, claim: &CairoClaim) -> Result<CompositionProofBindings, CompositionPlanError> {
        let mut base_param_values = Vec::with_capacity(self.sources.len());
        for source in self.sources.iter().copied() {
            base_param_values.push(match source {
                BaseParamSource::Constant(value) => value,
                BaseParamSource::SegmentStart(source) => segment_start(claim, source).ok_or(
                    CompositionPlanError::MissingBaseParamSegment {
                        component: source.name(),
                        instance: 0,
                        segment: source.name(),
                    },
                )?,
            });
        }
        Ok(CompositionProofBindings {
            components: Arc::clone(&self.components),
            base_param_values,
        })
    }
}

/// Bind only statement values to a shape already admitted by the full typed
/// `TopologyKey`. No evaluator construction, recording, lowering, or kernel
/// identity work is performed on this warm path.
pub(crate) fn bind_cairo_composition(
    claim: &CairoClaim,
    installed: &CompositionBindingPlan,
) -> Result<CompositionProofBindings, CompositionPlanError> {
    installed.bind(claim)
}

/// Compile and validate the direct binding recipe once with the shape.
///
/// Two independently tagged/permuted probe statements prove each changing
/// slot is exactly one known segment start. The recipe must then reconstruct
/// the original and both lowered probes word-for-word before it is installed.
pub(crate) fn compile_cairo_composition_binding_plan(
    claim: &CairoClaim,
    lookup_elements: &CommonLookupElements,
    interaction_claim: &CairoInteractionClaim,
    preprocessed_columns: &[PreProcessedColumnId],
    installed: &CompositionPlan,
) -> Result<CompositionBindingPlan, CompositionPlanError> {
    assert_binding_schema_is_whitelisted(claim);
    validate_required_segments(claim, installed)?;
    let original = CompositionProofBindings::from_plan(installed);
    let negative_control_claim = public_data_negative_control(claim);
    let negative_control = lower_cairo_composition_bindings(
        &negative_control_claim,
        lookup_elements,
        interaction_claim,
        preprocessed_columns,
        installed,
    )?;
    require_public_data_independence(&original, &negative_control)?;
    let (first_claim, second_claim, probes) = statement_probes(claim)?;
    let first = lower_cairo_composition_bindings(
        &first_claim,
        lookup_elements,
        interaction_claim,
        preprocessed_columns,
        installed,
    )?;
    let second = lower_cairo_composition_bindings(
        &second_claim,
        lookup_elements,
        interaction_claim,
        preprocessed_columns,
        installed,
    )?;
    let (components, words) = component_bindings(installed)?;
    let mut sources = Vec::with_capacity(words);
    for (index, component) in installed.components.iter().enumerate() {
        let (first_name, first_instance, first_values) =
            first
                .component(index)
                .ok_or(CompositionPlanError::BindingTopologyDrift {
                    component: component.component,
                    instance: component.instance,
                })?;
        let (second_name, second_instance, second_values) =
            second
                .component(index)
                .ok_or(CompositionPlanError::BindingTopologyDrift {
                    component: component.component,
                    instance: component.instance,
                })?;
        if (first_name, first_instance) != (component.component, component.instance)
            || (second_name, second_instance) != (component.component, component.instance)
        {
            return Err(CompositionPlanError::BindingTopologyDrift {
                component: component.component,
                instance: component.instance,
            });
        }
        sources.extend(classify_base_params(
            component.component,
            component.instance,
            &component.base_param_values,
            first_values,
            second_values,
            &probes,
        )?);
    }
    let recipe = CompositionBindingPlan {
        components,
        sources: sources.into(),
    };
    for (sample, sample_claim, expected) in [
        ("original", claim, original),
        ("first_probe", &first_claim, first),
        ("second_probe", &second_claim, second),
    ] {
        if recipe.bind(sample_claim)? != expected {
            return Err(CompositionPlanError::BaseParamRecipeMismatch { sample });
        }
    }
    Ok(recipe)
}

fn require_public_data_independence(
    expected: &CompositionProofBindings,
    actual: &CompositionProofBindings,
) -> Result<(), CompositionPlanError> {
    if expected == actual {
        return Ok(());
    }
    for index in 0..expected.component_count() {
        let Some((component, instance, expected_values)) = expected.component(index) else {
            break;
        };
        let Some((actual_component, actual_instance, actual_values)) = actual.component(index)
        else {
            break;
        };
        if (component, instance) != (actual_component, actual_instance) {
            break;
        }
        if let Some((slot, (&expected, &actual))) = expected_values
            .iter()
            .zip(actual_values)
            .enumerate()
            .find(|(_, (expected, actual))| expected != actual)
        {
            return Err(CompositionPlanError::UnsupportedPublicDataBaseParam {
                component,
                instance,
                slot,
                expected,
                actual,
            });
        }
    }
    Err(CompositionPlanError::BaseParamRecipeMismatch {
        sample: "public_data_negative_control",
    })
}

fn component_bindings(
    plan: &CompositionPlan,
) -> Result<(Arc<[CompositionComponentBinding]>, usize), CompositionPlanError> {
    let mut offset = 0usize;
    let mut components = Vec::with_capacity(plan.components.len());
    for component in &plan.components {
        let words = component.base_param_values.len();
        components.push(CompositionComponentBinding {
            component: component.component,
            instance: component.instance,
            base_param_offset: offset,
            base_param_words: words,
        });
        offset = offset
            .checked_add(words)
            .ok_or(CompositionPlanError::ConstraintCountOverflow)?;
    }
    Ok((components.into(), offset))
}

fn validate_required_segments(
    claim: &CairoClaim,
    installed: &CompositionPlan,
) -> Result<(), CompositionPlanError> {
    for component in &installed.components {
        if let Some(source) = component_segment_source(component.component) {
            if segment_start(claim, source).is_none() {
                return Err(CompositionPlanError::MissingBaseParamSegment {
                    component: component.component,
                    instance: component.instance,
                    segment: source.name(),
                });
            }
        }
    }
    Ok(())
}

fn component_segment_source(component: &str) -> Option<SegmentStartSource> {
    match component {
        "add_mod_builtin" => Some(SegmentStartSource::AddMod),
        "bitwise_builtin" => Some(SegmentStartSource::Bitwise),
        "mul_mod_builtin" => Some(SegmentStartSource::MulMod),
        "pedersen_builtin" | "pedersen_builtin_narrow_windows" => {
            Some(SegmentStartSource::Pedersen)
        }
        "poseidon_builtin" => Some(SegmentStartSource::Poseidon),
        "range_check96_builtin" => Some(SegmentStartSource::RangeCheck96),
        "range_check_builtin" => Some(SegmentStartSource::RangeCheck128),
        "ec_op_builtin" => Some(SegmentStartSource::EcOp),
        _ => None,
    }
}

fn statement_probes(
    claim: &CairoClaim,
) -> Result<(CairoClaim, CairoClaim, Vec<SegmentProbe>), CompositionPlanError> {
    let present = SEGMENT_START_SOURCES
        .into_iter()
        .filter_map(|source| segment_start(claim, source).map(|value| (source, value)))
        .collect::<Vec<_>>();
    let mut forbidden = present
        .iter()
        .map(|(_, value)| value.0)
        .collect::<BTreeSet<_>>();
    let first_values = allocate_probe_values(present.len(), 1_700_000_001, &mut forbidden)?;
    let mut second_values = allocate_probe_values(present.len(), 1_900_000_001, &mut forbidden)?;
    second_values.reverse();

    let mut first_claim = claim.clone();
    let mut second_claim = claim.clone();
    let probes = present
        .into_iter()
        .zip(first_values.into_iter().zip(second_values))
        .map(|((source, original), (first, second))| {
            set_segment_start(&mut first_claim, source, first.0);
            set_segment_start(&mut second_claim, source, second.0);
            SegmentProbe {
                source,
                original,
                first,
                second,
            }
        })
        .collect();
    Ok((first_claim, second_claim, probes))
}

fn allocate_probe_values(
    count: usize,
    mut candidate: u32,
    forbidden: &mut BTreeSet<u32>,
) -> Result<Vec<BaseField>, CompositionPlanError> {
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        if candidate >= 2_147_483_647 {
            return Err(CompositionPlanError::BaseParamProbeValueExhausted);
        }
        if forbidden.insert(candidate) {
            values.push(BaseField::from_u32_unchecked(candidate));
        }
        candidate = candidate
            .checked_add(1)
            .ok_or(CompositionPlanError::BaseParamProbeValueExhausted)?;
    }
    Ok(values)
}

fn segment_start(claim: &CairoClaim, source: SegmentStartSource) -> Option<BaseField> {
    let segments = &claim.public_data.public_memory.public_segments;
    let segment = match source {
        SegmentStartSource::AddMod => segments.add_mod,
        SegmentStartSource::Bitwise => segments.bitwise,
        SegmentStartSource::MulMod => segments.mul_mod,
        SegmentStartSource::Pedersen => segments.pedersen,
        SegmentStartSource::Poseidon => segments.poseidon,
        SegmentStartSource::RangeCheck96 => segments.range_check_96,
        SegmentStartSource::RangeCheck128 => segments.range_check_128,
        SegmentStartSource::EcOp => segments.ec_op,
    }?;
    Some(BaseField::from(segment.start_ptr.value))
}

fn set_segment_start(claim: &mut CairoClaim, source: SegmentStartSource, value: u32) {
    let segments = &mut claim.public_data.public_memory.public_segments;
    let segment = match source {
        SegmentStartSource::AddMod => &mut segments.add_mod,
        SegmentStartSource::Bitwise => &mut segments.bitwise,
        SegmentStartSource::MulMod => &mut segments.mul_mod,
        SegmentStartSource::Pedersen => &mut segments.pedersen,
        SegmentStartSource::Poseidon => &mut segments.poseidon,
        SegmentStartSource::RangeCheck96 => &mut segments.range_check_96,
        SegmentStartSource::RangeCheck128 => &mut segments.range_check_128,
        SegmentStartSource::EcOp => &mut segments.ec_op,
    };
    segment
        .as_mut()
        .expect("probe sources were collected only from present segments")
        .start_ptr
        .value = value;
}

fn classify_base_params(
    component: &'static str,
    instance: usize,
    values: &[BaseField],
    first_values: &[BaseField],
    second_values: &[BaseField],
    probes: &[SegmentProbe],
) -> Result<Vec<BaseParamSource>, CompositionPlanError> {
    if values.len() != first_values.len() || values.len() != second_values.len() {
        return Err(CompositionPlanError::BaseParamProbeCountMismatch {
            component,
            instance,
            expected: values.len(),
            first_probe: first_values.len(),
            second_probe: second_values.len(),
        });
    }
    for (index, probe) in probes.iter().enumerate() {
        let triple = (probe.original, probe.first, probe.second);
        if probe.original == probe.first
            || probe.original == probe.second
            || probe.first == probe.second
            || probes[..index]
                .iter()
                .any(|other| triple == (other.original, other.first, other.second))
        {
            return Err(CompositionPlanError::AmbiguousBaseParamProbe {
                component,
                instance,
            });
        }
    }
    values
        .iter()
        .copied()
        .zip(first_values.iter().copied())
        .zip(second_values.iter().copied())
        .enumerate()
        .map(|(slot, ((value, first), second))| {
            if value == first && value == second {
                return Ok(BaseParamSource::Constant(value));
            }
            let mut matches = probes.iter().filter(|probe| {
                (value, first, second) == (probe.original, probe.first, probe.second)
            });
            let source = matches.next().map(|probe| probe.source);
            if source.is_none() || matches.next().is_some() {
                return Err(CompositionPlanError::UnclassifiedBaseParam {
                    component,
                    instance,
                    slot,
                    value,
                    first_probe_value: first,
                    second_probe_value: second,
                });
            }
            Ok(BaseParamSource::SegmentStart(
                source.expect("one exact source"),
            ))
        })
        .collect()
}

/// Cold-only oracle: fully re-record one statement and prove it has the exact
/// installed topology, kernel identities, extension constants, and BASE ABI.
fn lower_cairo_composition_bindings(
    claim: &CairoClaim,
    lookup_elements: &CommonLookupElements,
    interaction_claim: &CairoInteractionClaim,
    preprocessed_columns: &[PreProcessedColumnId],
    installed: &CompositionPlan,
) -> Result<CompositionProofBindings, CompositionPlanError> {
    let cairo = CairoComponents::new(
        claim,
        lookup_elements,
        interaction_claim,
        preprocessed_columns,
    );
    let expected_count = cairo.components().len();
    let (components, words) = component_bindings(installed)?;
    let mut base_param_values = Vec::with_capacity(words);
    let mut component_index = 0usize;

    macro_rules! bind_component {
        ($name:expr, $instance:expr, $component:expr) => {{
            let expected = installed.components.get(component_index).ok_or(
                CompositionPlanError::BindingTopologyDrift {
                    component: $name,
                    instance: $instance,
                },
            )?;
            lower_component_values(
                $name,
                $instance,
                $component,
                expected,
                installed.max_kernel_instrs,
                &mut base_param_values,
            )?;
            component_index += 1;
        }};
    }
    macro_rules! bind_optional {
        ($( $field:ident ),+ $(,)?) => {
            $(
                if let Some(component) = &cairo.$field {
                    bind_component!(stringify!($field), 0, component);
                }
            )+
        };
    }

    bind_optional!(
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
    for (instance, component) in cairo.memory_id_to_big.iter().enumerate() {
        bind_component!("memory_id_to_big", instance, component);
    }
    bind_optional!(
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
    if expected_count != installed.components.len()
        || component_index != installed.components.len()
        || base_param_values.len() != words
    {
        return Err(CompositionPlanError::ComponentOrderMismatch {
            expected: installed.components.len(),
            actual: expected_count,
        });
    }
    Ok(CompositionProofBindings {
        components,
        base_param_values,
    })
}

fn lower_component_values<E: FrameworkEval>(
    name: &'static str,
    instance: usize,
    component: &FrameworkComponent<E>,
    installed: &CompositionComponentPlan,
    max_kernel_instrs: usize,
    packed_values: &mut Vec<BaseField>,
) -> Result<(), CompositionPlanError> {
    let topology_matches = installed.component == name
        && installed.instance == instance
        && installed.trace_locations == component.trace_locations()
        && installed.preprocessed_column_indices == component.preprocessed_column_indices()
        && installed.trace_log_size == component.evaluator().log_size()
        && installed.evaluation_log_size == component.max_constraint_log_degree_bound()
        && installed.n_constraints == component.n_constraints();
    if !topology_matches {
        return Err(CompositionPlanError::BindingTopologyDrift {
            component: name,
            instance,
        });
    }
    let binding = constraint_program_bindings(
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
    let kernels_match = binding.kernels.len() == installed.kernels.len()
        && binding
            .kernels
            .iter()
            .zip(&installed.kernels)
            .all(|(current, expected)| {
                current.cache_key == expected.cache_key
                    && current.semantic_hash == expected.semantic_hash
                    && current.rc_base == expected.rc_base
            });
    if !kernels_match {
        return Err(CompositionPlanError::BindingKernelMismatch {
            component: name,
            instance,
        });
    }
    if binding.base_param_values.len() != installed.base_param_values.len()
        || binding.ext_param_values != installed.ext_param_values
    {
        return Err(CompositionPlanError::BindingTopologyDrift {
            component: name,
            instance,
        });
    }
    packed_values.extend(binding.base_param_values);
    Ok(())
}

#[cfg(test)]
#[path = "bindings_tests.rs"]
mod tests;
