use stwo_backend_cuda::{
    WitnessInputCompactAbiAccess, WitnessInputCompactAbiArgumentKind, WitnessInputSeedAbiAccess,
    WitnessInputSeedAbiArgumentKind,
};

use super::*;

#[test]
fn missing_compact_sources_roll_back_earlier_seed_allocations() {
    let fixture = fixture();
    let mut values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    let unchanged = values.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut values).is_err());
    assert_eq!(values, unchanged);
}

#[test]
fn planned_seed_and_compact_drift_fail_transactionally() {
    let fixture = fixture();
    let seed = fixture
        .executable
        .arena()
        .witness()
        .components
        .iter()
        .find(|component| component.input_seed.is_some())
        .unwrap()
        .clone();
    let compact = fixture
        .executable
        .arena()
        .witness()
        .components
        .iter()
        .find(|component| {
            component
                .input_compact
                .as_ref()
                .is_some_and(|compact| compact.sources.len() > 1)
        })
        .unwrap()
        .clone();

    let mut seed_mutations: Vec<Box<dyn Fn(&mut crate::arena_plan::PlannedWitnessComponent)>> =
        Vec::new();
    seed_mutations.push(Box::new(|component| {
        component
            .input_seed
            .as_mut()
            .unwrap()
            .requirements
            .scalar_words += 1;
    }));
    seed_mutations.push(Box::new(|component| {
        component
            .input_seed
            .as_mut()
            .unwrap()
            .slots
            .consumer_input_columns
            .swap(0, 1);
    }));
    seed_mutations.push(Box::new(|component| {
        let seed = component.input_seed.as_mut().unwrap();
        seed.slots.output_pointers = seed.slots.scalar_values;
    }));
    for mutate in seed_mutations {
        let mut changed = seed.clone();
        mutate(&mut changed);
        let mut values = fixture.initial_values.clone();
        assert!(lower_component(fixture.executable.arena(), &changed, &mut values).is_err());
        assert_eq!(values, fixture.initial_values);
    }

    let mut compact_mutations: Vec<Box<dyn Fn(&mut crate::arena_plan::PlannedWitnessComponent)>> =
        Vec::new();
    compact_mutations.push(Box::new(|component| {
        component.input_compact.as_mut().unwrap().sources.swap(0, 1);
    }));
    compact_mutations.push(Box::new(|component| {
        component
            .input_compact
            .as_mut()
            .unwrap()
            .requirements
            .descriptor_words += 1;
    }));
    compact_mutations.push(Box::new(|component| {
        component
            .input_compact
            .as_mut()
            .unwrap()
            .slots
            .consumer_input_columns
            .swap(0, 1);
    }));
    compact_mutations.push(Box::new(|component| {
        let compact = component.input_compact.as_mut().unwrap();
        core::mem::swap(
            &mut compact.slots.sort_keys_a,
            &mut compact.slots.sort_keys_b,
        );
    }));
    compact_mutations.push(Box::new(|component| {
        let compact = component.input_compact.as_mut().unwrap();
        compact.slots.sort_temp = compact.slots.scan_temp;
    }));
    for mutate in compact_mutations {
        let mut changed = compact.clone();
        mutate(&mut changed);
        let mut values = fixture.initial_values.clone();
        assert!(lower_component(fixture.executable.arena(), &changed, &mut values).is_err());
        assert_eq!(values, fixture.initial_values);
    }
}

#[test]
fn every_seed_and_compact_abi_field_is_authoritative() {
    let fixture = fixture();
    for setup in &fixture.lowered {
        match setup {
            LoweredWitnessInputSetup::Seed(seed) => {
                let authoritative = seed.contract.abi().arguments();
                for index in 0..authoritative.len() {
                    let mut changed = authoritative.to_vec();
                    changed[index].ordinal ^= 0x80;
                    assert!(seed_invocation_using_abi_for_test(seed, &changed).is_err());
                    let mut changed = authoritative.to_vec();
                    changed[index].name = "wrong_role";
                    assert!(seed_invocation_using_abi_for_test(seed, &changed).is_err());
                    let mut changed = authoritative.to_vec();
                    changed[index].kind =
                        if changed[index].kind == WitnessInputSeedAbiArgumentKind::CudaStream {
                            WitnessInputSeedAbiArgumentKind::U32
                        } else {
                            WitnessInputSeedAbiArgumentKind::CudaStream
                        };
                    assert!(seed_invocation_using_abi_for_test(seed, &changed).is_err());
                    let mut changed = authoritative.to_vec();
                    changed[index].access = if changed[index].access
                        == WitnessInputSeedAbiAccess::OrderedExecutionStream
                    {
                        WitnessInputSeedAbiAccess::ScalarWordCount
                    } else {
                        WitnessInputSeedAbiAccess::OrderedExecutionStream
                    };
                    assert!(seed_invocation_using_abi_for_test(seed, &changed).is_err());
                }
            }
            LoweredWitnessInputSetup::Compact(compact) => {
                let authoritative = compact.contract.abi().arguments();
                for index in 0..authoritative.len() {
                    let mut changed = authoritative.to_vec();
                    changed[index].ordinal ^= 0x80;
                    assert!(projection::compact_invocation_using_abi_for_test(
                        compact, 4, 8, &changed
                    )
                    .is_err());
                    let mut changed = authoritative.to_vec();
                    changed[index].name = "wrong_role";
                    assert!(projection::compact_invocation_using_abi_for_test(
                        compact, 4, 8, &changed
                    )
                    .is_err());
                    let mut changed = authoritative.to_vec();
                    changed[index].kind =
                        if changed[index].kind == WitnessInputCompactAbiArgumentKind::CudaStream {
                            WitnessInputCompactAbiArgumentKind::U32
                        } else {
                            WitnessInputCompactAbiArgumentKind::CudaStream
                        };
                    assert!(projection::compact_invocation_using_abi_for_test(
                        compact, 4, 8, &changed
                    )
                    .is_err());
                    let mut changed = authoritative.to_vec();
                    changed[index].access = if changed[index].access
                        == WitnessInputCompactAbiAccess::OrderedExecutionStream
                    {
                        WitnessInputCompactAbiAccess::EdgeCount
                    } else {
                        WitnessInputCompactAbiAccess::OrderedExecutionStream
                    };
                    assert!(projection::compact_invocation_using_abi_for_test(
                        compact, 4, 8, &changed
                    )
                    .is_err());
                }
            }
        }
    }
}

#[test]
fn retained_stage_mutations_fail_reconstruction_validation() {
    let fixture = fixture();
    let mut mutations = Vec::new();

    let mut changed = fixture.lowered.clone();
    let LoweredWitnessInputSetup::Seed(seed) = &mut changed[0] else {
        panic!("first generated setup changed kind")
    };
    seed.scalar_source.arena.len_words -= 1;
    mutations.push(changed);

    let seed_index = fixture
        .lowered
        .iter()
        .position(
            |setup| matches!(setup, LoweredWitnessInputSetup::Seed(seed) if seed.outputs.len() > 1),
        )
        .unwrap();
    let mut changed = fixture.lowered.clone();
    let LoweredWitnessInputSetup::Seed(seed) = &mut changed[seed_index] else {
        unreachable!()
    };
    seed.outputs.swap(0, 1);
    mutations.push(changed);

    let compact_index = fixture
        .lowered
        .iter()
        .position(|setup| matches!(setup, LoweredWitnessInputSetup::Compact(_)))
        .unwrap();
    let mut changed = fixture.lowered.clone();
    let LoweredWitnessInputSetup::Compact(compact) = &mut changed[compact_index] else {
        unreachable!()
    };
    compact.descriptor_value = compact.outputs[0].version;
    mutations.push(changed);

    let mut changed = fixture.lowered.clone();
    let LoweredWitnessInputSetup::Compact(compact) = &mut changed[compact_index] else {
        unreachable!()
    };
    compact.scratch.swap(0, 1);
    mutations.push(changed);

    let mut changed = fixture.lowered.clone();
    match &mut changed[compact_index] {
        LoweredWitnessInputSetup::Seed(seed) => seed.position.level += 1,
        LoweredWitnessInputSetup::Compact(compact) => compact.position.level += 1,
    }
    mutations.push(changed);

    for changed in mutations {
        assert!(validate(fixture.executable.arena(), &fixture.values, &changed).is_err());
    }
}
