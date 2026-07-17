use super::*;
use crate::compiled_proof::{EffectContract, ValueVersion};

#[test]
fn lowering_is_idempotent_and_partial_or_transitioned_state_rolls_back() {
    let fixture = fixture();
    let mut repeated = fixture.values.clone();
    assert_eq!(
        lower_stage(fixture.executable.arena(), &mut repeated).unwrap(),
        fixture.lowered
    );
    assert_eq!(repeated, fixture.values);

    let catalogs = fixture
        .lowered
        .destinations
        .iter()
        .map(|destination| destination.value)
        .collect::<Vec<_>>();
    let mut partial =
        adapter::SemanticValueMap::allocate_ordered(catalogs.iter().take(1).copied()).unwrap();
    let before = partial.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut partial).is_err());
    assert_eq!(partial, before);

    let mut transitioned = fixture.values.clone();
    transitioned
        .transition(fixture.lowered.destinations[0].value)
        .unwrap();
    let before = transitioned.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut transitioned).is_err());
    assert_eq!(transitioned, before);
}

#[test]
fn exact_validation_rejects_order_and_relocation_drift() {
    let fixture = fixture();

    let mut changed = fixture.lowered.clone();
    changed.destinations.swap(0, 1);
    assert!(validate(fixture.executable.arena(), &fixture.values, &changed).is_err());

    let mut changed = fixture.lowered.clone();
    changed.relocations.destination_lengths = changed.relocations.destination_pointers;
    assert!(validate(fixture.executable.arena(), &fixture.values, &changed).is_err());
}

#[test]
fn abi_mutations_and_retained_invocation_or_effect_drift_are_rejected() {
    let fixture = fixture();
    let mut abi = fixture.lowered.contract.abi().arguments().to_vec();
    abi[0].name = "wrong_destinations";
    assert_eq!(
        semantic::invocation_for_test(&fixture.lowered, &abi),
        Err(InvocationShapeError::InvalidStructuredAbi)
    );

    let mut changed = fixture.lowered.clone();
    changed.invocation.arguments[2].value = AotArgumentValue::U32(21);
    assert!(projection::validate_lowered(&changed).is_err());

    let mut changed = fixture.lowered.clone();
    let mut accesses = changed.effect.accesses().to_vec();
    let EffectAccess::Write { destination } = &mut accesses[0] else {
        panic!("clear destination must be a write");
    };
    destination.value.version = ValueVersion(u32::MAX);
    changed.effect = EffectContract::new(accesses, Vec::new()).unwrap();
    assert!(projection::validate_lowered(&changed).is_err());
}

#[test]
fn projection_rejects_destination_geometry_drift() {
    let fixture = fixture();

    let mut changed = fixture.lowered.clone();
    changed.destinations[0].ordinal = 1;
    assert!(projection::validate_lowered(&changed).is_err());

    let mut changed = fixture.lowered.clone();
    changed.destinations[0].elements.end -= 1;
    assert!(projection::validate_lowered(&changed).is_err());

    let mut changed = fixture.lowered.clone();
    changed.relocations.destination_lengths.len_words += 1;
    assert!(projection::validate_lowered(&changed).is_err());

    let mut changed = fixture.lowered.clone();
    let requirements =
        stwo_backend_cuda::witness_feed_clear_workspace_requirements(&[1, 2]).unwrap();
    changed.contract = WitnessFeedClearContract::compile(&requirements).unwrap();
    assert!(projection::validate_lowered(&changed).is_err());
}
