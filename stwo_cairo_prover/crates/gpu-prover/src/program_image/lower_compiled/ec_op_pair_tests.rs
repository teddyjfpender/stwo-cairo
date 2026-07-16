use super::*;
use crate::compiled_proof::{
    EffectAccess, EffectBindingId, EffectContract, InPlaceDiscipline, ValueVersion,
};
use crate::resident_runtime::producer_schedule::BaseProducerSchedule;

fn generated_native_ec_op() -> producer_prefix::LoweredNativeEcOpProducer {
    let executable = tests::generated_sn2();
    let image = ArenaProgramInventory::from_planned_parts(
        executable.topology(),
        executable.transcript(),
        executable.arena(),
    )
    .unwrap();
    let schedule = BaseProducerSchedule::compile(executable.arena()).unwrap();
    let module = [9; 32];
    producer_prefix::map_scheduled_base_producers_with_native_authority(
        &image,
        executable.arena(),
        &schedule,
        |contract| {
            ec_op_execution_authority::NativeEcOpLinkedModuleAuthority::bind_exact(
                contract, module, module, 89,
            )
            .map(Some)
        },
    )
    .unwrap()
    .bound[7]
        .native_ec_op()
        .unwrap()
        .clone()
}

fn write_to_read(effect: &EffectContract, binding: EffectBindingId) -> EffectContract {
    let mut accesses = effect.accesses().to_vec();
    let access = accesses
        .iter_mut()
        .find(|access| {
            access
                .destination()
                .is_some_and(|destination| destination.binding == binding)
        })
        .unwrap();
    let EffectAccess::Write { destination } = access else {
        panic!("selected EC-op output is not an exact write")
    };
    *access = EffectAccess::Read {
        source: *destination,
    };
    EffectContract::new(accesses, effect.module_globals().to_vec()).unwrap()
}

fn assert_recomputed_identity_changes(
    native: &producer_prefix::LoweredNativeEcOpProducer,
    changed: &ec_op_prefix::StaticEcOpInvocation,
) {
    let changed_effect = ec_op_prefix::exact_effect(changed).unwrap();
    let rebound = native
        .execution
        .linked
        .clone()
        .bind_lowered(&native.contract.authority, changed, &changed_effect)
        .unwrap();
    assert_ne!(
        rebound.invocation_identity,
        native.execution.invocation_identity
    );
    assert_ne!(rebound.identity, native.execution.identity);
    assert!(rebound
        .validate_exact(
            &native.execution.linked,
            &native.contract.authority,
            &native.contract.invocation,
            &native.contract.effect,
        )
        .is_err());
}

#[test]
fn paired_execution_authority_seals_lowered_invocation_and_effect() {
    let native = generated_native_ec_op();
    let invocation = &native.contract.invocation;
    let effect = &native.contract.effect;
    let linked = &native.execution.linked;

    // A coordinated inner/outer recomputation for another linked module and
    // SM remains invalid against the independently trusted linked authority.
    let forged = ec_op_execution_authority::NativeEcOpLinkedModuleAuthority::bind_exact(
        &native.contract.authority,
        [8; 32],
        [8; 32],
        90,
    )
    .unwrap()
    .bind_lowered(&native.contract.authority, invocation, effect)
    .unwrap();
    assert!(forged
        .validate_exact(linked, &native.contract.authority, invocation, effect)
        .is_err());

    // Lookup, trace and partial outputs are exact writes, not coarse ranges.
    for binding in [
        invocation.lookup_words.binding.unwrap(),
        invocation.trace_columns[0].binding.unwrap(),
        invocation.partial_input_columns[0].binding.unwrap(),
    ] {
        assert!(linked
            .clone()
            .bind_lowered(
                &native.contract.authority,
                invocation,
                &write_to_read(effect, binding),
            )
            .is_err());
    }

    // Equal-shape table reordering remains executable-looking but changes the
    // invocation and paired identities, then fails validation against source.
    let mut changed = invocation.clone();
    assert_eq!(
        changed.execution_tables[1].value.value_words.len(),
        changed.execution_tables[2].value.value_words.len()
    );
    let first = changed.execution_tables[1].value.clone();
    changed.execution_tables[1].value = changed.execution_tables[2].value.clone();
    changed.execution_tables[2].value = first;
    assert_recomputed_identity_changes(&native, &changed);

    // Catalog values and descriptor relocation are not in EffectContract, so
    // the invocation identity must independently seal them.
    let mut changed = invocation.clone();
    changed.lookup_words.value.value.0 += 1;
    assert_recomputed_identity_changes(&native, &changed);
    let mut changed = invocation.clone();
    changed.execution_table_pointers.value.0 += 1;
    assert_recomputed_identity_changes(&native, &changed);

    // Semantic versions and valid atomic discipline drift can be re-encoded;
    // both must produce a different paired receipt.
    let mut changed = invocation.clone();
    changed.lookup_words.version =
        Some(ValueVersion(changed.lookup_words.version.unwrap().0 + 100));
    assert_recomputed_identity_changes(&native, &changed);
    let mut changed = invocation.clone();
    changed.multiplicities[0].source = ValueVersion(changed.multiplicities[0].source.0 + 100);
    assert_recomputed_identity_changes(&native, &changed);
    let mut changed = invocation.clone();
    changed.multiplicities[0].destination =
        ValueVersion(changed.multiplicities[0].destination.0 + 100);
    assert_recomputed_identity_changes(&native, &changed);
    let mut changed = invocation.clone();
    changed.multiplicities[0].alias.discipline = InPlaceDiscipline::BlockBarrierPhases;
    assert_recomputed_identity_changes(&native, &changed);

    // Contract geometry, range, binding, vector and alias-ID drift reject
    // before any recomputed outer authority can be admitted.
    let mut rejected = invocation.clone();
    rejected.lookup_words.value.value_words.end -= 1;
    assert!(linked
        .clone()
        .bind_lowered(
            &native.contract.authority,
            &rejected,
            &ec_op_prefix::exact_effect(&rejected).unwrap(),
        )
        .is_err());
    let mut rejected = invocation.clone();
    rejected.lookup_words.binding = None;
    assert!(ec_op_prefix::exact_effect(&rejected).is_err());
    let mut rejected = invocation.clone();
    rejected.execution_tables.pop();
    assert!(linked
        .clone()
        .bind_lowered(&native.contract.authority, &rejected, effect)
        .is_err());
    let mut rejected = invocation.clone();
    rejected.row_count += 1;
    assert!(linked
        .clone()
        .bind_lowered(&native.contract.authority, &rejected, effect)
        .is_err());
    let mut rejected = invocation.clone();
    rejected.partial_row_count += 1;
    assert!(linked
        .clone()
        .bind_lowered(&native.contract.authority, &rejected, effect)
        .is_err());
    let mut rejected = invocation.clone();
    rejected.execution_table_pointers.value_words.end -= 1;
    assert!(linked
        .clone()
        .bind_lowered(&native.contract.authority, &rejected, effect)
        .is_err());
    let mut rejected = invocation.clone();
    rejected.multiplicities[0].alias.id.0 += 1;
    assert!(ec_op_prefix::exact_effect(&rejected).is_err());
}
