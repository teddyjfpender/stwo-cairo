use num_traits::Zero;

use super::*;

#[test]
fn denominator_layout_matches_constraint_framework_bit_reverse() {
    let values = denominator_inverses(4, 6);
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|value| *value != BaseField::zero()));
}

#[test]
fn ext_param_probe_classifies_only_exact_device_sources() {
    let lookup = CommonLookupElements::dummy();
    let probe = CommonLookupElements::from_z_alpha(LOOKUP_PROBE_Z, LOOKUP_PROBE_ALPHA);
    let log_size = 9;
    let claimed_sum = SecureField::from_u32_unchecked(3, 5, 7, 11);
    let rows = BaseField::from_u32_unchecked(1 << log_size);
    let constant = SecureField::from_u32_unchecked(13, 17, 19, 23);
    let values = vec![
        constant,
        lookup.z(),
        lookup.alpha_powers()[3],
        claimed_sum / rows,
    ];
    let probe_values = vec![
        constant,
        probe.z(),
        probe.alpha_powers()[3],
        CLAIMED_SUM_PROBE / rows,
    ];
    assert_eq!(
        classify_ext_params(
            "test",
            0,
            log_size,
            claimed_sum,
            &lookup,
            &values,
            &probe,
            &probe_values,
        )
        .unwrap(),
        vec![
            CompositionExtParamSource::Constant(constant),
            CompositionExtParamSource::LookupZ,
            CompositionExtParamSource::LookupAlphaPower(3),
            CompositionExtParamSource::ClaimedSumScaled,
        ]
    );
}

#[test]
fn add_ap_folded_relation_constant_is_scaled_alpha_power() {
    use cairo_air::components::add_ap_opcode;

    let lookup = CommonLookupElements::dummy();
    let probe = CommonLookupElements::from_z_alpha(LOOKUP_PROBE_Z, LOOKUP_PROBE_ALPHA);
    let log_size = 9;
    let claimed_sum = SecureField::from_u32_unchecked(3, 5, 7, 11);
    let primary = constraint_program(
        &add_ap_opcode::Eval {
            claim: add_ap_opcode::Claim { log_size },
            common_lookup_elements: lookup.clone(),
        },
        3,
        claimed_sum,
        log_size,
        usize::MAX,
    )
    .unwrap();
    let probed = constraint_program(
        &add_ap_opcode::Eval {
            claim: add_ap_opcode::Claim { log_size },
            common_lookup_elements: probe.clone(),
        },
        3,
        CLAIMED_SUM_PROBE,
        log_size,
        usize::MAX,
    )
    .unwrap();
    let scale = BaseField::from_u32_unchecked(32767);

    assert_eq!(
        primary.ext_param_values[2],
        lookup.alpha_powers()[2] * scale
    );
    assert_eq!(probed.ext_param_values[2], probe.alpha_powers()[2] * scale);
    let sources = classify_ext_params(
        "add_ap_opcode",
        0,
        log_size,
        claimed_sum,
        &lookup,
        &primary.ext_param_values,
        &probe,
        &probed.ext_param_values,
    )
    .unwrap();
    assert_eq!(
        sources[2],
        CompositionExtParamSource::LookupAlphaPowerScaled { power: 2, scale }
    );
}

#[test]
fn bitwise_statement_start_is_a_runtime_base_binding_not_kernel_identity() {
    use cairo_air::components::bitwise_builtin;

    let lookup = CommonLookupElements::dummy();
    let claimed_sum = SecureField::from_u32_unchecked(3, 5, 7, 11);
    let lower = |segment_start| {
        constraint_program(
            &bitwise_builtin::Eval {
                claim: bitwise_builtin::Claim { log_size: 4 },
                common_lookup_elements: lookup.clone(),
                bitwise_builtin_segment_start: segment_start,
            },
            3,
            claimed_sum,
            4,
            2048,
        )
        .unwrap()
    };
    let first_start = 7_711;
    let second_start = 6_606_534;
    let first = lower(first_start);
    let second = lower(second_start);

    assert!(same_kernel_program(&first.kernels, &second.kernels));
    assert_eq!(first.ext_param_values, second.ext_param_values);
    assert_eq!(
        first.base_param_values.len(),
        second.base_param_values.len()
    );
    let differences = first
        .base_param_values
        .iter()
        .zip(&second.base_param_values)
        .filter(|(left, right)| left != right)
        .collect::<Vec<_>>();
    assert_eq!(differences.len(), 5);
    assert!(differences.iter().all(|(left, right)| {
        **left == BaseField::from_u32_unchecked(first_start)
            && **right == BaseField::from_u32_unchecked(second_start)
    }));
}

#[test]
fn scaled_alpha_classifier_rejects_non_base_affine_and_mismatched_scales() {
    let lookup = CommonLookupElements::dummy();
    let probe = CommonLookupElements::from_z_alpha(LOOKUP_PROBE_Z, LOOKUP_PROBE_ALPHA);
    let log_size = 9;
    let claimed_sum = SecureField::from_u32_unchecked(3, 5, 7, 11);
    let alpha = lookup.alpha_powers()[2];
    let probe_alpha = probe.alpha_powers()[2];
    let non_base_scale = SecureField::from_u32_unchecked(2, 3, 5, 7);
    let base_scale = BaseField::from_u32_unchecked(32767);
    let other_scale = BaseField::from_u32_unchecked(32768);
    let one = SecureField::from(BaseField::from_u32_unchecked(1));

    for (value, probe_value) in [
        (alpha * non_base_scale, probe_alpha * non_base_scale),
        (alpha * base_scale + one, probe_alpha * base_scale + one),
        (alpha * base_scale, probe_alpha * other_scale),
    ] {
        assert!(matches!(
            classify_ext_params(
                "test",
                0,
                log_size,
                claimed_sum,
                &lookup,
                &[value],
                &probe,
                &[probe_value],
            ),
            Err(CompositionPlanError::UnclassifiedExtParam { slot: 0, .. })
        ));
    }
}

fn key_with_ext_source(source: CompositionExtParamSource) -> u64 {
    CompositionPlan {
        max_kernel_instrs: 1,
        total_constraints: 1,
        max_evaluation_log_size: 5,
        components: vec![CompositionComponentPlan {
            component: "test",
            instance: 0,
            trace_locations: Vec::new(),
            preprocessed_column_indices: Vec::new(),
            trace_log_size: 4,
            evaluation_log_size: 5,
            n_constraints: 1,
            random_coefficient_offset: 0,
            denominator_inverses: Vec::new(),
            base_param_values: Vec::new(),
            ext_param_values: vec![SecureField::zero()],
            ext_param_sources: vec![source],
            kernels: Vec::new(),
        }],
        wave_kernels: Vec::new(),
    }
    .key()
}

#[test]
fn scaled_alpha_scale_is_part_of_protocol_key() {
    assert_ne!(
        key_with_ext_source(CompositionExtParamSource::LookupAlphaPowerScaled {
            power: 2,
            scale: BaseField::from_u32_unchecked(32767),
        }),
        key_with_ext_source(CompositionExtParamSource::LookupAlphaPowerScaled {
            power: 2,
            scale: BaseField::from_u32_unchecked(32768),
        })
    );
}

#[test]
fn base_parameter_values_are_runtime_bindings_but_slot_count_is_topology() {
    let mut plan = CompositionPlan {
        max_kernel_instrs: 1,
        total_constraints: 1,
        max_evaluation_log_size: 5,
        components: vec![CompositionComponentPlan {
            component: "test",
            instance: 0,
            trace_locations: Vec::new(),
            preprocessed_column_indices: Vec::new(),
            trace_log_size: 4,
            evaluation_log_size: 5,
            n_constraints: 1,
            random_coefficient_offset: 0,
            denominator_inverses: Vec::new(),
            base_param_values: vec![BaseField::from_u32_unchecked(7)],
            ext_param_values: Vec::new(),
            ext_param_sources: Vec::new(),
            kernels: Vec::new(),
        }],
        wave_kernels: Vec::new(),
    };
    let original = plan.key();
    plan.components[0].base_param_values[0] = BaseField::from_u32_unchecked(11);
    assert_eq!(
        plan.key(),
        original,
        "statement values are rebound at setup"
    );
    plan.components[0]
        .base_param_values
        .push(BaseField::from_u32_unchecked(13));
    assert_ne!(plan.key(), original, "slot topology changes the graph ABI");

    let bindings = CompositionProofBindings::from_plan(&plan);
    assert_eq!(bindings.component_count(), 1);
    assert_eq!(bindings.base_param_word_count(), 2);
    assert_eq!(
        bindings.component(0),
        Some((
            "test",
            0,
            [
                BaseField::from_u32_unchecked(11),
                BaseField::from_u32_unchecked(13),
            ]
            .as_slice(),
        ))
    );
    assert_eq!(bindings.component(1), None);
}
