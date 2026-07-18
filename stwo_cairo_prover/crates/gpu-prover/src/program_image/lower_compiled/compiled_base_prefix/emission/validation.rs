use super::causal::statement_host_ingress;
use super::sequence::{causal_events, CausalBaseEvent};
use super::*;

#[allow(clippy::too_many_arguments)]
pub(in crate::program_image::lower_compiled) fn validate_sealed_prefix(
    authority: &BaseProducerAuthority,
    causal_setup: &CausalWitnessSetup,
    next_producer: usize,
    target_sm: u32,
    kernel_by_build_authority: &BTreeMap<[u8; 32], usize>,
    kernels: &[AotKernelAuthority],
    static_wrappers: &[StaticCudaWrapperAuthority],
    module_global_initializers: &[ModuleGlobalInitializer],
    effects: &[EffectContract],
    partitions: &[PartitionAuthority],
    monolithic: &PartitionAuthority,
    operations: &[OpNode],
) -> Result<(), ()> {
    let expected = causal_events(authority, causal_setup, next_producer)?;
    let expected_partitions = (!operations.is_empty())
        .then(|| monolithic.clone())
        .into_iter()
        .collect::<Vec<_>>();
    if operations.len() != expected.len()
        || partitions != expected_partitions
        || kernel_by_build_authority.len() != kernels.len()
        || kernel_by_build_authority
            .values()
            .copied()
            .collect::<BTreeSet<_>>()
            != (0..kernels.len()).collect()
        || kernels
            .iter()
            .enumerate()
            .any(|(index, kernel)| kernel.id().0 as usize != index + 1)
        || static_wrappers.iter().enumerate().any(|(index, wrapper)| {
            wrapper.id().0 as usize != index + 1
                || wrapper.consumer_target_sm() != target_sm
                || !wrapper.has_valid_identity().unwrap_or(false)
        })
        || effects.windows(2).any(|pair| pair[0].id() >= pair[1].id())
    {
        return Err(());
    }

    let mut used_kernels = BTreeMap::<AotKernelId, BTreeSet<_>>::new();
    let mut used_effects = BTreeSet::new();
    let mut used_initializers = BTreeSet::new();
    let mut next_wrapper = 1usize;
    for (index, (operation, event)) in operations.iter().zip(expected).enumerate() {
        if operation.id.0 as usize != index
            || operation.semantic_id.0 as usize != index + 1
            || operation.partition != monolithic.id()
            || operation.stage
                != ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
        {
            return Err(());
        }
        let effect = effects
            .iter()
            .find(|effect| effect.id() == operation.effect)
            .ok_or(())?;
        used_effects.insert(operation.effect);
        match (event, &operation.primitive) {
            (
                CausalBaseEvent::Recorded(recorded),
                ExecutionPrimitive::AotKernel { kernel, launch },
            ) if *launch == recorded.source.launch => {
                if operation.invocation.as_ref() != Some(&recorded.invocation) {
                    return Err(());
                }
                let authority = kernels
                    .iter()
                    .find(|authority| authority.id() == *kernel)
                    .ok_or(())?;
                if effect.accesses() != recorded.effect.accesses()
                    || validate_recorded_module_global_effect(
                        &recorded.source,
                        authority.module(),
                        effect,
                        module_global_initializers,
                    )
                    .is_err()
                {
                    return Err(());
                }
                used_initializers.extend(
                    effect
                        .module_globals()
                        .iter()
                        .map(|global| global.initializer),
                );
                used_kernels.entry(*kernel).or_default().insert((
                    operation.effect,
                    operation.partition,
                    operation
                        .invocation
                        .as_ref()
                        .ok_or(())?
                        .contract_id()
                        .map_err(|_| ())?,
                ));
            }
            (
                CausalBaseEvent::Static(request),
                ExecutionPrimitive::StaticCudaWrapper { wrapper },
            ) if wrapper.0 as usize == next_wrapper => {
                let authority = static_wrappers.get(next_wrapper - 1).ok_or(())?;
                let invocation = operation.invocation.as_ref().ok_or(())?;
                if authority.id() != *wrapper
                    || authority.accepted_effect() != operation.effect
                    || authority.accepted_invocation()
                        != invocation.contract_id().map_err(|_| ())?
                    || invocation != &request.invocation_for_wrapper(authority).map_err(|_| ())?
                    || effect != request.effect().map_err(|_| ())?
                {
                    return Err(());
                }
                next_wrapper += 1;
            }
            (CausalBaseEvent::HostIngress(lowered), primitive) => {
                let (expected_primitive, expected_effect) =
                    statement_host_ingress(lowered).map_err(|_| ())?;
                if operation.invocation.is_some()
                    || primitive != &expected_primitive
                    || effect != &expected_effect
                {
                    return Err(());
                }
            }
            _ => return Err(()),
        }
    }
    let declared_effects = effects
        .iter()
        .map(EffectContract::id)
        .collect::<BTreeSet<_>>();
    let declared_initializers = module_global_initializers
        .iter()
        .enumerate()
        .map(|(index, initializer)| {
            (initializer.id().0 as usize == index
                && initializer.has_valid_identity().unwrap_or(false))
            .then_some(initializer.id())
            .ok_or(())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if next_wrapper != static_wrappers.len() + 1
        || declared_effects.len() != effects.len()
        || declared_effects != used_effects
        || declared_initializers.len() != module_global_initializers.len()
        || declared_initializers != used_initializers
        || used_kernels.len() != kernels.len()
    {
        return Err(());
    }
    for kernel in kernels {
        let accepted = kernel
            .accepted_executions()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if accepted.len() != kernel.accepted_executions().len()
            || used_kernels.remove(&kernel.id()) != Some(accepted)
        {
            return Err(());
        }
    }
    used_kernels.is_empty().then_some(()).ok_or(())
}
