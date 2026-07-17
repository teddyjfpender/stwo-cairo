use super::*;
use crate::compiled_proof::{EffectContract, StaticCudaWrapperId, ValueVersion};

const MODULE: [u8; 32] = [9; 32];
const TARGET_SM: u32 = 89;

fn lowered_ec_op() -> ec_op_prefix::LoweredNativeEcOpContract {
    let executable = tests::generated_sn2_replacement();
    executable
        .replacement_base_producers()
        .unwrap()
        .producers
        .iter()
        .find_map(|producer| match producer {
            producer_prefix::SemanticBaseProducer::NativeEcOp { contract, .. } => {
                Some(contract.clone())
            }
            _ => None,
        })
        .expect("SN2 replacement fixture must contain native EC-op")
}

fn linked_ec_op(
    lowered: &ec_op_prefix::LoweredNativeEcOpContract,
) -> ec_op_execution_authority::NativeEcOpLinkedModuleAuthority {
    ec_op_execution_authority::NativeEcOpLinkedModuleAuthority::bind_exact(
        &lowered.authority,
        MODULE,
        MODULE,
        TARGET_SM,
    )
    .unwrap()
}

#[test]
fn ec_op_projection_copies_every_exact_linked_receipt() {
    let lowered = lowered_ec_op();
    let linked = linked_ec_op(&lowered);
    let projected =
        static_wrapper_projection::ec_op(StaticCudaWrapperId(1), &linked, &lowered).unwrap();

    assert!(projected.has_valid_identity().unwrap());
    assert_eq!(projected.static_module_build_identity(), &MODULE);
    assert_eq!(projected.consumer_target_sm(), TARGET_SM);
    assert_eq!(projected.wrapper_symbol(), linked.entry_symbol.as_bytes());
    assert_eq!(projected.semantic_abi_identity(), &linked.abi_identity);
    assert_eq!(
        projected.semantic_effect_identity(),
        &linked.effect_identity
    );
    assert_eq!(
        projected.aggregate_contract_identity(),
        &linked.contract_identity
    );
    assert_eq!(projected.linked_module_identity(), &linked.identity);
    assert_eq!(projected.accepted_effect(), lowered.effect.id());
    let projected_launches = projected.kernel_launches().collect::<Vec<_>>();
    assert_eq!(projected_launches.len(), linked.launches.len());
    for (projected, exact) in projected_launches.into_iter().zip(linked.launches) {
        assert_eq!(projected.symbol(), exact.entry_symbol.as_bytes());
        assert_eq!(projected.launch().grid, exact.launch.grid);
        assert_eq!(projected.launch().block, exact.launch.block);
        assert_eq!(
            projected.launch().dynamic_shared_bytes,
            exact.launch.dynamic_shared_bytes
        );
        assert_eq!(projected.launch().cooperative, exact.launch.cooperative);
        assert_eq!(projected.launch().cluster, None);
    }
}

#[test]
fn ec_op_projection_rejects_linked_and_lowered_drift() {
    let lowered = lowered_ec_op();
    let linked = linked_ec_op(&lowered);
    let mutations: [fn(&mut ec_op_execution_authority::NativeEcOpLinkedModuleAuthority); 13] = [
        |changed| changed.static_module_build_identity[0] ^= 1,
        |changed| changed.expected_static_module_build_identity[0] ^= 1,
        |changed| changed.consumer_target_sm += 1,
        |changed| changed.source_identity[0] ^= 1,
        |changed| changed.abi_identity[0] ^= 1,
        |changed| changed.effect_identity[0] ^= 1,
        |changed| changed.launch_identity[0] ^= 1,
        |changed| changed.contract_identity[0] ^= 1,
        |changed| changed.entry_symbol = "wrong_wrapper",
        |changed| changed.launches[0].entry_symbol = "wrong_kernel",
        |changed| changed.launches.swap(0, 1),
        |changed| changed.launches[0].launch.grid[0] += 1,
        |changed| changed.identity[0] ^= 1,
    ];
    for mutate in mutations {
        let mut changed = linked.clone();
        mutate(&mut changed);
        assert!(
            static_wrapper_projection::ec_op(StaticCudaWrapperId(1), &changed, &lowered).is_err()
        );
    }

    let mut changed = lowered.clone();
    changed.invocation.row_count += 1;
    assert!(static_wrapper_projection::ec_op(StaticCudaWrapperId(1), &linked, &changed).is_err());
    let mut changed = lowered.clone();
    let mut accesses = changed.effect.accesses().to_vec();
    accesses[0].source_mut().unwrap().value.version = ValueVersion(10_000);
    changed.effect = EffectContract::new(accesses, Vec::new()).unwrap();
    assert!(static_wrapper_projection::ec_op(StaticCudaWrapperId(1), &linked, &changed).is_err());
    assert!(static_wrapper_projection::ec_op(StaticCudaWrapperId(0), &linked, &lowered).is_err());
}

#[test]
fn direct_blake_g_projection_copies_every_exact_linked_receipt() {
    let (_, lowered) = blake_g_direct_tests::lowered_direct();
    let linked =
        blake_g_direct_execution_authority::NativeBlakeGDirectLinkedModuleAuthority::bind_exact(
            &lowered.authority,
            MODULE,
            MODULE,
            TARGET_SM,
            TARGET_SM,
        )
        .unwrap();
    let projected =
        static_wrapper_projection::blake_g_direct(StaticCudaWrapperId(2), &linked, &lowered)
            .unwrap();
    let launch = linked.contract.wrapper_launch();

    assert!(projected.has_valid_identity().unwrap());
    assert_eq!(projected.static_module_build_identity(), &MODULE);
    assert_eq!(projected.consumer_target_sm(), TARGET_SM);
    assert_eq!(
        projected.wrapper_symbol(),
        linked.contract.abi().entry_symbol().as_bytes()
    );
    assert_eq!(
        projected.semantic_abi_identity(),
        &linked.contract.abi_identity()
    );
    assert_eq!(
        projected.semantic_effect_identity(),
        &linked.contract.effect_identity()
    );
    assert_eq!(
        projected.aggregate_contract_identity(),
        &linked.contract.identity()
    );
    assert_eq!(projected.linked_module_identity(), &linked.identity);
    assert_eq!(projected.accepted_effect(), lowered.effect.id());
    let projected_launches = projected.kernel_launches().collect::<Vec<_>>();
    assert_eq!(projected_launches.len(), 1);
    assert_eq!(
        projected_launches[0].symbol(),
        launch.audited_internal_kernel_symbol().as_bytes()
    );
    assert_eq!(projected_launches[0].launch().grid, launch.grid);
    assert_eq!(projected_launches[0].launch().block, launch.block);
    assert_eq!(
        projected_launches[0].launch().dynamic_shared_bytes,
        launch.dynamic_shared_bytes
    );
    assert_eq!(
        projected_launches[0].launch().cooperative,
        launch.cooperative
    );
    assert_eq!(projected_launches[0].launch().cluster, None);
}

#[test]
fn direct_blake_g_projection_rejects_linked_and_lowered_drift() {
    let (executable, lowered) = blake_g_direct_tests::lowered_direct();
    let linked =
        blake_g_direct_execution_authority::NativeBlakeGDirectLinkedModuleAuthority::bind_exact(
            &lowered.authority,
            MODULE,
            MODULE,
            TARGET_SM,
            TARGET_SM,
        )
        .unwrap();
    let mutations: [fn(
        &mut blake_g_direct_execution_authority::NativeBlakeGDirectLinkedModuleAuthority,
    ); 4] = [
        |changed| changed.static_module_build_identity[0] ^= 1,
        |changed| changed.expected_static_module_build_identity[0] ^= 1,
        |changed| changed.consumer_target_sm += 1,
        |changed| changed.identity[0] ^= 1,
    ];
    for mutate in mutations {
        let mut changed = linked.clone();
        mutate(&mut changed);
        assert!(static_wrapper_projection::blake_g_direct(
            StaticCudaWrapperId(2),
            &changed,
            &lowered,
        )
        .is_err());
    }

    let component = executable
        .arena()
        .witness()
        .components
        .iter()
        .find(|component| component.component == "blake_g")
        .unwrap();
    assert!(lowered.authority.n_real_rows() > 0);
    let mut changed_linked = linked.clone();
    changed_linked.contract = stwo_backend_cuda::BlakeGDirectCompositeContract::compile(
        &component.program,
        lowered.authority.n_real_rows() - 1,
        lowered.authority.padded_rows(),
    )
    .unwrap();
    assert!(static_wrapper_projection::blake_g_direct(
        StaticCudaWrapperId(2),
        &changed_linked,
        &lowered,
    )
    .is_err());

    let mut changed = lowered.clone();
    changed.invocation.padded_rows += 1;
    assert!(
        static_wrapper_projection::blake_g_direct(StaticCudaWrapperId(2), &linked, &changed)
            .is_err()
    );
    let mut changed = lowered.clone();
    let mut accesses = changed.effect.accesses().to_vec();
    accesses[0].source_mut().unwrap().value.version = ValueVersion(10_000);
    changed.effect = EffectContract::new(accesses, Vec::new()).unwrap();
    assert!(
        static_wrapper_projection::blake_g_direct(StaticCudaWrapperId(2), &linked, &changed)
            .is_err()
    );
    assert!(
        static_wrapper_projection::blake_g_direct(StaticCudaWrapperId(0), &linked, &lowered)
            .is_err()
    );
}
