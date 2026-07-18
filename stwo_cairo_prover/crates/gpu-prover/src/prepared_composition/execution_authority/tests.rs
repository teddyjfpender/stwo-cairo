use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use super::*;
use crate::shape_executable::ShapeExecutable;

fn sn2() -> Arc<ShapeExecutable> {
    static EXECUTABLE: OnceLock<Arc<ShapeExecutable>> = OnceLock::new();
    Arc::clone(EXECUTABLE.get_or_init(crate::program_image::generated_sn2_replacement))
}

fn authority() -> (Arc<ShapeExecutable>, CompositionExecutionAuthority) {
    let executable = sn2();
    let authority = CompositionExecutionAuthority::compile(executable.arena()).unwrap();
    (executable, authority)
}

fn argument_names(operation: &CompositionOperation) -> Vec<&str> {
    operation
        .invocation
        .arguments
        .iter()
        .map(|argument| argument.name.as_ref())
        .collect()
}

#[test]
fn generated_sn2_seals_exact_wrapper_child_wave_and_part_counts() {
    let (executable, authority) = authority();
    let operations = authority.operations();
    assert_eq!(operations.len(), 31);
    // 1 materialize + 1 powers + 14 waves + 13 lifts + 3 split A +
    // 2 split B = 34. The former 33 count omitted one real child launch.
    assert_eq!(
        operations
            .iter()
            .map(|operation| operation.children.len())
            .sum::<usize>(),
        34
    );
    assert_eq!(authority.waves().len(), 14);
    assert_eq!(
        executable
            .arena()
            .composition()
            .requirements
            .waves
            .iter()
            .map(|wave| wave.parts.len())
            .sum::<usize>(),
        123
    );
    assert_eq!(
        operations
            .iter()
            .filter(|operation| matches!(operation.kind, CompositionOperationKind::Wave { .. }))
            .count(),
        14
    );
    assert_eq!(
        operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation.kind,
                    CompositionOperationKind::LiftAccumulate { .. }
                )
            })
            .count(),
        13
    );
    assert!(matches!(
        operations[0].kind,
        CompositionOperationKind::MaterializeExtParams { count: 4_782, .. }
    ));
    assert_eq!(
        operations[1].kind,
        CompositionOperationKind::GenerateDescendingPowers { count: 1_053 }
    );
    assert_eq!(
        operations
            .iter()
            .map(|operation| operation.abi)
            .collect::<Vec<_>>(),
        std::iter::once(CompositionAbi::MaterializeExtParamsV1)
            .chain(std::iter::once(CompositionAbi::GenerateDescendingPowersV1))
            .chain(std::iter::repeat_n(CompositionAbi::WaveV2, 14))
            .chain(std::iter::repeat_n(CompositionAbi::LiftAccumulateV1, 13))
            .chain([
                CompositionAbi::SplitInverseFusedFirstForwardV1,
                CompositionAbi::SplitForwardAfterFirstIntervalV1,
            ])
            .collect::<Vec<_>>()
    );
    assert_eq!(
        argument_names(&operations[0]),
        [
            "destinations",
            "source_kinds",
            "source_indices",
            "scales",
            "count",
            "z",
            "alpha_powers",
            "alpha_power_count",
            "claimed_sums",
            "claimed_sum_count",
            "stream",
        ]
    );
    assert_eq!(
        argument_names(&operations[1]),
        ["random_coefficient", "powers", "count", "stream"]
    );
    authority.validate_against(executable.arena()).unwrap();
}

#[test]
fn waves_preserve_all_part_ranges_and_exact_aot_order() {
    let (executable, authority) = authority();
    let requirements = &executable.arena().composition().requirements;
    let program =
        CompositionWaveProgram::from_plan(&executable.arena().composition().plan).unwrap();
    let wave_operations = authority
        .operations()
        .iter()
        .filter(|operation| matches!(operation.kind, CompositionOperationKind::Wave { .. }))
        .collect::<Vec<_>>();
    for (wave_index, ((operation, requirement), canonical)) in wave_operations
        .iter()
        .zip(&requirements.waves)
        .zip(program.waves())
        .enumerate()
    {
        assert_eq!(operation.children.len(), 1);
        assert_eq!(
            operation.children[0].symbol.as_ref(),
            executable.arena().composition().plan.wave_kernels[wave_index].kernel_name
        );
        assert_eq!(
            operation.children[0].parameters[0..2],
            [
                ("wave_index", wave_index as u32),
                ("part_count", requirement.parts.len() as u32),
            ]
        );
        let actual_ranges = operation.children[0]
            .effect
            .accesses
            .iter()
            .filter_map(|access| {
                (access.source == Some(CompositionValueRole::RandomCoefficientPowers))
                    .then_some(access.elements)
            })
            .collect::<Vec<_>>();
        let expected_ranges = canonical
            .part_ordinals
            .iter()
            .map(|&ordinal| {
                let part = &program.parts()[ordinal];
                ElementRange {
                    start: part.coefficient_start * 4,
                    end: part.coefficient_end * 4,
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(actual_ranges, expected_ranges);
        assert_eq!(operation.invocation.arguments.len(), 10);
    }
}

#[test]
fn pointer_storage_is_physical_relocation_metadata_not_a_semantic_value() {
    let (executable, authority) = authority();
    let plan = executable.arena();
    let requirements = &plan.composition().requirements;
    let program = CompositionWaveProgram::from_plan(&plan.composition().plan).unwrap();

    let mut expected = BTreeSet::from([
        CompositionRelocationRole::DynamicDestinations,
        CompositionRelocationRole::ClaimedSumPointers,
        CompositionRelocationRole::SplitSourcePointers,
        CompositionRelocationRole::SplitRetainedPointers,
    ]);
    expected.extend(program.parts().iter().map(|part| {
        CompositionRelocationRole::EvaluationPointers {
            component: part.component_index as u32,
        }
    }));
    assert_eq!(
        authority
            .relocations()
            .iter()
            .map(|layout| layout.role)
            .collect::<BTreeSet<_>>(),
        expected
    );
    for relocation in authority.relocations() {
        let binding = plan.binding(relocation.logical).unwrap();
        assert_ne!(relocation.word_len, 0);
        assert_ne!(relocation.alignment_words, 0);
        assert_eq!(relocation.first_word % relocation.alignment_words, 0);
        assert!(relocation.first_word + relocation.word_len <= binding.len_words);
        assert!(!authority.layouts().iter().any(|semantic| {
            semantic.logical == relocation.logical
                && semantic.first_word == relocation.first_word
                && semantic.word_len == relocation.word_len
        }));
    }

    let materialize = &authority.operations()[0];
    let pointer_len = |argument: usize| match &materialize.invocation.arguments[argument].value {
        CompositionInvocationValue::PointerTable { pointee_accesses } => {
            assert!(pointee_accesses.iter().all(Option::is_some));
            pointee_accesses.len()
        }
        other => panic!("expected relocation pointer graph, got {other:?}"),
    };
    assert_eq!(pointer_len(0), requirements.dynamic_ext_param_count);
    assert_eq!(pointer_len(8), requirements.claimed_sum_count);

    for (operation, wave) in authority.operations()[2..16]
        .iter()
        .zip(&requirements.waves)
    {
        assert_eq!(
            operation.invocation.embedded_pointer_tables.len(),
            wave.parts.len()
        );
        for (table, part) in operation
            .invocation
            .embedded_pointer_tables
            .iter()
            .zip(&wave.parts)
        {
            assert_eq!(
                table.pointee_accesses.len(),
                requirements.components[part.component].sources.len()
            );
            assert!(table.pointee_accesses.iter().all(Option::is_some));
        }
    }
}

#[test]
fn pointer_graph_identity_binds_order_nulls_and_leaf_identity_without_table_storage() {
    let (_, authority) = authority();
    let exact = &authority.operations()[0].invocation;
    assert_eq!(
        exact.identity,
        super::encoding::invocation_identity(
            CompositionAbi::MaterializeExtParamsV1,
            &exact.arguments,
            &exact.embedded_pointer_tables,
        )
        .unwrap()
    );

    let changed_identity = |arguments: Vec<CompositionInvocationArgument>| {
        super::encoding::invocation_identity(
            CompositionAbi::MaterializeExtParamsV1,
            &arguments,
            &[],
        )
        .unwrap()
    };
    let mut reordered = exact.arguments.clone();
    let CompositionInvocationValue::PointerTable { pointee_accesses } = &mut reordered[0].value
    else {
        panic!("destinations must be a pointer graph");
    };
    pointee_accesses.swap(0, 1);
    assert_ne!(exact.identity, changed_identity(reordered));

    let mut nulled = exact.arguments.clone();
    let CompositionInvocationValue::PointerTable { pointee_accesses } = &mut nulled[0].value else {
        panic!("destinations must be a pointer graph");
    };
    pointee_accesses[0] = None;
    assert_ne!(exact.identity, changed_identity(nulled));

    let mut changed_leaf = exact.arguments.clone();
    let CompositionInvocationValue::PointerTable { pointee_accesses } = &mut changed_leaf[0].value
    else {
        panic!("destinations must be a pointer graph");
    };
    pointee_accesses[0] = pointee_accesses[1];
    assert_ne!(exact.identity, changed_identity(changed_leaf));

    let wave = &authority.operations()[2].invocation;
    let mut reordered_tables = wave.embedded_pointer_tables.clone();
    reordered_tables.swap(0, 1);
    assert_ne!(
        wave.identity,
        super::encoding::invocation_identity(
            CompositionAbi::WaveV2,
            &wave.arguments,
            &reordered_tables,
        )
        .unwrap()
    );
    let mut nulled_embedded = wave.embedded_pointer_tables.clone();
    nulled_embedded[0].pointee_accesses[0] = None;
    assert_ne!(
        wave.identity,
        super::encoding::invocation_identity(
            CompositionAbi::WaveV2,
            &wave.arguments,
            &nulled_embedded,
        )
        .unwrap()
    );
}

#[test]
fn direct_split_seals_two_outer_abis_five_children_and_final_outputs() {
    let (_, authority) = authority();
    let inverse = &authority.operations()[29];
    let forward = &authority.operations()[30];
    assert_eq!(inverse.abi, CompositionAbi::SplitInverseFusedFirstForwardV1);
    assert_eq!(
        forward.abi,
        CompositionAbi::SplitForwardAfterFirstIntervalV1
    );
    assert_eq!(inverse.invocation.arguments.len(), 9);
    assert_eq!(forward.invocation.arguments.len(), 7);
    assert_eq!(
        argument_names(inverse),
        [
            "source_values",
            "retained_outputs",
            "log_n",
            "inverse_twiddles",
            "inverse_twiddle_words",
            "forward_twiddles",
            "forward_twiddle_words",
            "eval_domain_size",
            "stream",
        ]
    );
    assert_eq!(
        argument_names(forward),
        [
            "values",
            "log_n",
            "num_poly",
            "forward_twiddles",
            "forward_twiddle_words",
            "eval_domain_size",
            "stream",
        ]
    );
    assert_eq!(
        inverse
            .children
            .iter()
            .map(|child| child.symbol.as_ref())
            .collect::<Vec<_>>(),
        [
            "b2n_init_block_warp_batch<2>",
            "b2n_noinit_block_batch<4,false>",
            "composition_split_boundary_batch<3,true>",
        ]
    );
    assert_eq!(
        forward
            .children
            .iter()
            .map(|child| child.symbol.as_ref())
            .collect::<Vec<_>>(),
        [
            "n2b_nofinal_block_batch<4,4>",
            "n2b_final_block_warp_batch<2,true>",
        ]
    );
    assert_eq!(inverse.children[0].launch.grid, [16_384, 1, 4]);
    assert_eq!(inverse.children[1].launch.grid, [32, 64, 4]);
    assert_eq!(inverse.children[2].launch.grid, [8_192, 1, 4]);
    assert_eq!(forward.children[0].launch.grid, [32, 64, 8]);
    assert_eq!(forward.children[1].launch.grid, [16_384, 1, 8]);
    assert_eq!(
        *authority.outputs(),
        std::array::from_fn(|column| CompositionValueRole::SplitRetained {
            canonical_column: column as u8,
            generation: 3,
        })
    );
}

#[test]
fn structural_tampering_and_reordering_fail_closed() {
    let (executable, exact) = authority();
    let plan = executable.arena();
    let mut cases = Vec::new();

    let mut reordered_wrappers = exact.clone();
    reordered_wrappers.operations.swap(0, 1);
    cases.push(reordered_wrappers);

    let mut reordered_waves = exact.clone();
    reordered_waves.waves.swap(0, 1);
    cases.push(reordered_waves);

    let mut reordered_lifts = exact.clone();
    reordered_lifts.operations.swap(16, 17);
    cases.push(reordered_lifts);

    let mut nested_symbol = exact.clone();
    nested_symbol.operations[29].children[0].symbol = "alternate_kernel".into();
    cases.push(nested_symbol);

    let mut alternate_output = exact.clone();
    alternate_output.outputs.swap(0, 1);
    cases.push(alternate_output);

    let mut source_drift = exact.clone();
    source_drift.source_identity[0] ^= 1;
    cases.push(source_drift);

    let mut effect_drift = exact.clone();
    effect_drift.operations[2].children[0]
        .effect
        .accesses
        .swap(0, 1);
    cases.push(effect_drift);

    let mut relocation_drift = exact.clone();
    relocation_drift.relocations.swap(0, 1);
    cases.push(relocation_drift);

    let mut pointer_graph_drift = exact.clone();
    pointer_graph_drift.operations[2]
        .invocation
        .embedded_pointer_tables
        .swap(0, 1);
    cases.push(pointer_graph_drift);

    for drifted in cases {
        assert_eq!(
            drifted.validate_against(plan),
            Err(CompositionAuthorityError::InvalidIdentity)
        );
    }
}

#[test]
fn legacy_mode_and_invalid_target_fail_before_execution_authority() {
    let legacy = crate::program_image::generated_sn2();
    assert!(matches!(
        CompositionExecutionAuthority::compile(legacy.arena()),
        Err(CompositionAuthorityError::UnsupportedLaunchMode(_))
    ));

    let (executable, authority) = authority();
    assert_eq!(
        authority.bind_linked(executable.arena(), 9),
        Err(CompositionAuthorityError::InvalidTargetSm(9))
    );
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        assert_eq!(authority.bind_linked(executable.arena(), 90).unwrap(), None);
    }
}
