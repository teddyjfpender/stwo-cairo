use stwo_backend_cuda::{WitnessCasmInputAbiAccess, WitnessCasmInputAbiArgumentKind};

use super::*;
use crate::compiled_proof::EffectContract;

#[test]
fn missing_writer_sources_and_partial_staging_lineage_roll_back() {
    let fixture = fixture();
    let mut empty =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    let unchanged = empty.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut empty).is_err());
    assert_eq!(empty, unchanged);

    let staging = fixture.lowered[0].staging.value;
    let mut partial = fixture.initial_values.clone();
    partial.extend_ordered([staging]).unwrap();
    partial.transition(staging).unwrap();
    let unchanged = partial.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut partial).is_err());
    assert_eq!(partial, unchanged);

    let mut extra = fixture.values.clone();
    extra.transition(staging).unwrap();
    let unchanged = extra.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut extra).is_err());
    assert_eq!(extra, unchanged);

    let output = fixture.lowered[0].outputs[0].value;
    let mut transitioned_output = fixture.initial_values.clone();
    transitioned_output.transition(output).unwrap();
    let unchanged = transitioned_output.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut transitioned_output).is_err());
    assert_eq!(transitioned_output, unchanged);
}

#[test]
fn planned_requirements_slots_and_output_order_fail_transactionally() {
    let fixture = fixture();
    let component = fixture
        .executable
        .arena()
        .witness()
        .components
        .iter()
        .find(|component| component.input_casm.is_some())
        .unwrap()
        .clone();
    let mut mutations: Vec<Box<dyn Fn(&mut crate::arena_plan::PlannedWitnessComponent)>> =
        Vec::new();
    mutations.push(Box::new(|component| {
        component
            .input_casm
            .as_mut()
            .unwrap()
            .requirements
            .staging_words += 1;
    }));
    mutations.push(Box::new(|component| {
        component
            .input_casm
            .as_mut()
            .unwrap()
            .requirements
            .consumer_input_column_words[0] += 1;
    }));
    mutations.push(Box::new(|component| {
        component
            .input_casm
            .as_mut()
            .unwrap()
            .slots
            .consumer_input_columns
            .swap(0, 1);
    }));
    mutations.push(Box::new(|component| {
        let casm = component.input_casm.as_mut().unwrap();
        casm.slots
            .consumer_input_columns
            .push(casm.slots.consumer_input_columns[0]);
    }));
    mutations.push(Box::new(|component| {
        let casm = component.input_casm.as_mut().unwrap();
        casm.slots.staging = casm.slots.consumer_input_columns[0];
    }));
    mutations.push(Box::new(|component| {
        component
            .input_casm
            .as_mut()
            .unwrap()
            .requirements
            .include_iota = true;
    }));
    for mutate in mutations {
        let mut changed = component.clone();
        mutate(&mut changed);
        let mut values = fixture.initial_values.clone();
        let unchanged = values.clone();
        assert!(
            lower_component_for_test(fixture.executable.arena(), &changed, &mut values).is_err()
        );
        assert_eq!(values, unchanged);
    }
}

#[test]
fn every_casm_abi_field_is_authoritative() {
    let lane = &fixture().lowered[0];
    let authoritative = lane.contract.abi().arguments();
    assert!(
        invocation_using_abi_for_test(lane, &authoritative[..authoritative.len() - 1]).is_err()
    );
    for index in 0..authoritative.len() {
        let mut changed = authoritative.to_vec();
        changed[index].ordinal ^= 0x80;
        assert!(invocation_using_abi_for_test(lane, &changed).is_err());

        let mut changed = authoritative.to_vec();
        changed[index].name = "wrong_role";
        assert!(invocation_using_abi_for_test(lane, &changed).is_err());

        let mut changed = authoritative.to_vec();
        changed[index].kind = if changed[index].kind == WitnessCasmInputAbiArgumentKind::CudaStream
        {
            WitnessCasmInputAbiArgumentKind::U32
        } else {
            WitnessCasmInputAbiArgumentKind::CudaStream
        };
        assert!(invocation_using_abi_for_test(lane, &changed).is_err());

        let mut changed = authoritative.to_vec();
        changed[index].access =
            if changed[index].access == WitnessCasmInputAbiAccess::OrderedExecutionStream {
                WitnessCasmInputAbiAccess::RealRowCount
            } else {
                WitnessCasmInputAbiAccess::OrderedExecutionStream
            };
        assert!(invocation_using_abi_for_test(lane, &changed).is_err());
    }
}

#[test]
fn retained_fragment_mutations_fail_reconstruction() {
    let fixture = fixture();
    let mut mutations = Vec::new();

    let mut changed = fixture.lowered.clone();
    changed[0].position.level += 1;
    mutations.push(changed);

    let mut changed = fixture.lowered.clone();
    changed[0].staging.version = changed[1].staging.version;
    mutations.push(changed);

    let mut changed = fixture.lowered.clone();
    changed[1].staging.previous = None;
    mutations.push(changed);

    let mut changed = fixture.lowered.clone();
    changed[0].outputs.swap(0, 1);
    mutations.push(changed);

    let mut changed = fixture.lowered.clone();
    changed[0].outputs[0].writer_use = WitnessCasmWriterUse::InactiveMechanical;
    mutations.push(changed);

    let mut changed = fixture.lowered.clone();
    changed[0].invocation.arguments.swap(0, 1);
    mutations.push(changed);

    let mut changed = fixture.lowered.clone();
    let mut accesses = changed[0].effect.accesses().to_vec();
    accesses[0].source_mut().unwrap().value.elements.end -= 1;
    changed[0].effect = EffectContract::new(accesses, Vec::new()).unwrap();
    mutations.push(changed);

    for changed in mutations {
        assert!(validate(fixture.executable.arena(), &fixture.values, &changed).is_err());
    }
}

#[test]
fn projection_semantics_reject_mutated_invocations_and_effects() {
    let lane = &fixture().lowered[0];
    projection::validate_lowered(lane).unwrap();

    let mut changed = lane.clone();
    changed.invocation.arguments.swap(0, 1);
    assert!(projection::validate_lowered(&changed).is_err());

    let mut changed = lane.clone();
    let mut accesses = changed.effect.accesses().to_vec();
    accesses[0].source_mut().unwrap().value.elements.end -= 1;
    changed.effect = EffectContract::new(accesses, Vec::new()).unwrap();
    assert!(projection::validate_lowered(&changed).is_err());
}
