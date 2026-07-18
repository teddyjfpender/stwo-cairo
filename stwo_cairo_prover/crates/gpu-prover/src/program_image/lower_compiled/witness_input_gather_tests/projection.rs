use super::*;
use crate::compiled_proof::{LaunchGeometry, StaticCudaWrapperId};

#[test]
fn linked_projection_seals_exact_build_abi_invocation_effect_source_and_launch() {
    let fixture = fixture();
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        for lowered in &fixture.fragments {
            assert_eq!(lowered.contract.bind_static_build(89).unwrap(), None);
        }
        return;
    }

    let target = stwo_backend_cuda_kernels::static_cuda_module_target_sms()[0];
    let lowered = &fixture.fragments[0];
    let linked = lowered.contract.bind_static_build(target).unwrap().unwrap();
    let projected = witness_input_gather::project_static_wrapper(
        fixture.executable.arena(),
        &fixture.values,
        StaticCudaWrapperId(1),
        &linked,
        lowered,
    )
    .unwrap();
    let wrapper = &projected.wrapper;
    let launch = lowered.contract.wrapper_launch();
    let installed_launches = wrapper.kernel_launches().collect::<Vec<_>>();

    assert_eq!(
        wrapper.static_module_build_identity(),
        &linked.module_build_identity()
    );
    assert_eq!(wrapper.consumer_target_sm(), target);
    assert_eq!(
        wrapper.wrapper_symbol(),
        lowered.contract.abi().entry_symbol().as_bytes()
    );
    assert_eq!(
        wrapper.semantic_abi_identity(),
        &lowered.contract.abi_identity()
    );
    assert_eq!(
        wrapper.semantic_effect_identity(),
        &lowered.contract.effect_identity()
    );
    assert_eq!(
        wrapper.aggregate_contract_identity(),
        &lowered.contract.identity()
    );
    assert_eq!(wrapper.linked_module_identity(), &linked.identity());
    assert_eq!(
        wrapper.accepted_invocation(),
        lowered.invocation.contract_id().unwrap()
    );
    assert_eq!(wrapper.accepted_effect(), lowered.effect.id());
    assert_ne!(lowered.contract.source_identity(), [0; 32]);
    assert_ne!(linked.static_build_source_identity(), [0; 32]);
    assert_eq!(installed_launches.len(), 1);
    assert_eq!(
        installed_launches[0].symbol(),
        launch.audited_internal_kernel_symbol().as_bytes()
    );
    assert_eq!(
        installed_launches[0].launch(),
        LaunchGeometry {
            grid: launch.grid,
            block: launch.block,
            cluster: None,
            dynamic_shared_bytes: launch.dynamic_shared_bytes,
            cooperative: launch.cooperative,
        }
    );
}

#[test]
fn projection_rejects_fabricated_lineage_and_retained_fragment_drift() {
    let fixture = fixture();
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        return;
    }

    let target = stwo_backend_cuda_kernels::static_cuda_module_target_sms()[0];
    let lowered = &fixture.fragments[0];
    let linked = lowered.contract.bind_static_build(target).unwrap().unwrap();
    let project = |candidate: &witness_input_gather::LoweredWitnessInputGather| {
        witness_input_gather::project_static_wrapper(
            fixture.executable.arena(),
            &fixture.values,
            StaticCudaWrapperId(1),
            &linked,
            candidate,
        )
    };

    assert!(witness_input_gather::project_static_wrapper(
        fixture.executable.arena(),
        &fixture.initial_values,
        StaticCudaWrapperId(1),
        &linked,
        lowered,
    )
    .is_err());
    assert!(witness_input_gather::project_static_wrapper(
        fixture.executable.arena(),
        &fixture.values,
        StaticCudaWrapperId(0),
        &linked,
        lowered,
    )
    .is_err());

    let other = fixture
        .fragments
        .iter()
        .find(|candidate| candidate.contract.identity() != lowered.contract.identity())
        .unwrap();
    let wrong_linked = other.contract.bind_static_build(target).unwrap().unwrap();
    assert!(witness_input_gather::project_static_wrapper(
        fixture.executable.arena(),
        &fixture.values,
        StaticCudaWrapperId(1),
        &wrong_linked,
        lowered,
    )
    .is_err());

    let mut changed = lowered.clone();
    changed.invocation.arguments.swap(0, 1);
    assert!(project(&changed).is_err());

    let mut changed = lowered.clone();
    let mut accesses = changed.effect.accesses().to_vec();
    accesses[0].source_mut().unwrap().value.elements.end -= 1;
    changed.effect = EffectContract::new(accesses, Vec::new()).unwrap();
    assert!(project(&changed).is_err());

    let mut changed = lowered.clone();
    changed.sources[0].arena.len_words -= 1;
    assert!(project(&changed).is_err());

    let mut changed = lowered.clone();
    changed.outputs.swap(0, 1);
    assert!(project(&changed).is_err());
}
