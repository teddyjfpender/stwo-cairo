//! Canonical executable projection of Relation challenge expansion.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    RelationChallengeAbiAccess, RelationChallengeAbiArgumentKind,
    RelationChallengeExpansionAuthority, RelationChallengeInvocationValue,
    RelationChallengeValueRole,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, BoundValueRange, EffectAccess, EffectBindingId,
    ElementRange, LaunchGeometry, StaticCudaExecutionStepIdentity, StaticCudaLaunchIdentity,
    ValueRange,
};

const WRAPPER_SYMBOL: &str = "stwo_relation_expand_challenges_on";

pub(super) fn lower(
    authority: &RelationChallengeExpansionAuthority,
    inventory: &RelationInventory,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredRelationChallenge, InvocationShapeError> {
    authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    let drawn = inventory.drawn();
    let drawn_version = values.version(drawn.catalog)?;
    let alpha = inventory.role(RelationValueRole::AlphaPowers)?;
    let z = inventory.role(RelationValueRole::ChallengeZ)?;
    let alpha_version = values.allocate_output(alpha.catalog)?;
    let z_version = values.allocate_output(z.catalog)?;
    let bindings = [
        challenge_binding(EffectBindingId(0), drawn, drawn_version)?,
        challenge_binding(EffectBindingId(1), alpha, alpha_version)?,
        challenge_binding(EffectBindingId(2), z, z_version)?,
    ];
    let effect = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bindings[0],
            },
            EffectAccess::Write {
                destination: bindings[1],
            },
            EffectAccess::Write {
                destination: bindings[2],
            },
        ],
        Vec::new(),
    )
    .map_err(|_| InvocationShapeError::InvalidRelationBinding)?;
    let invocation = invocation(authority, &bindings)?;
    validate_exact_bindings(&invocation, &effect)?;
    Ok(LoweredRelationChallenge {
        authority: authority.clone(),
        invocation,
        effect,
        drawn,
        drawn_version,
        alpha,
        alpha_version,
        z,
        z_version,
    })
}

pub(super) fn validate(lowered: &LoweredRelationChallenge) -> Result<(), InvocationShapeError> {
    lowered
        .authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    if lowered.drawn.words != lowered.authority.drawn_words()
        || lowered.alpha.words
            != lowered
                .authority
                .alpha_words()
                .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?
        || lowered.z.words != lowered.authority.z_words()
    {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    let bindings = [
        challenge_binding(EffectBindingId(0), lowered.drawn, lowered.drawn_version)?,
        challenge_binding(EffectBindingId(1), lowered.alpha, lowered.alpha_version)?,
        challenge_binding(EffectBindingId(2), lowered.z, lowered.z_version)?,
    ];
    let exact_effect = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bindings[0],
            },
            EffectAccess::Write {
                destination: bindings[1],
            },
            EffectAccess::Write {
                destination: bindings[2],
            },
        ],
        Vec::new(),
    )
    .map_err(|_| InvocationShapeError::InvalidRelationBinding)?;
    if lowered.invocation != invocation(&lowered.authority, &bindings)?
        || lowered.effect != exact_effect
    {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    validate_exact_bindings(&lowered.invocation, &lowered.effect)
}

pub(super) fn resolve_static_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &LoweredRelationChallenge,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    validate(lowered)?;
    let Some(linked) = lowered
        .authority
        .bind_static_build(target_sm)
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?
    else {
        return Ok(None);
    };
    linked
        .validate_for_target(&lowered.authority, target_sm)
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    if linked.contract_identity() != lowered.authority.identity()
        || [
            linked.module_build_identity(),
            linked.static_build_source_identity(),
            linked.sm_identity(),
            linked.identity(),
        ]
        .contains(&[0; 32])
    {
        return Err(InvocationShapeError::InvalidRelationAuthority);
    }
    let child = lowered.authority.child();
    let launch = StaticCudaLaunchIdentity::new(
        child.symbol.as_bytes().to_vec(),
        LaunchGeometry {
            grid: child.grid,
            block: child.block,
            cluster: child.cluster,
            dynamic_shared_bytes: child.dynamic_shared_bytes,
            cooperative: child.cooperative,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    StaticCudaWrapperAuthority::new_with_execution_steps(
        id,
        linked.module_build_identity(),
        target_sm,
        WRAPPER_SYMBOL.as_bytes().to_vec(),
        lowered.authority.abi_identity(),
        lowered.authority.effect_identity(),
        lowered.authority.identity(),
        linked.identity(),
        vec![StaticCudaExecutionStepIdentity::KernelLaunch(launch)],
        lowered
            .invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidRelationBinding)?,
        lowered.effect.id(),
    )
    .map(Some)
    .map_err(|_| InvocationShapeError::InvalidRelationAuthority)
}

fn invocation(
    authority: &RelationChallengeExpansionAuthority,
    bindings: &[BoundValueRange; 3],
) -> Result<AotInvocation, InvocationShapeError> {
    if authority.invocation().len()
        != authority
            .invocation()
            .last()
            .map_or(0, |argument| usize::from(argument.ordinal) + 1)
    {
        return Err(InvocationShapeError::InvalidRelationAuthority);
    }
    let mut arguments = Vec::with_capacity(authority.invocation().len().saturating_sub(1));
    for (descriptor, supplied) in stwo_backend_cuda::RELATION_CHALLENGE_ARGUMENTS
        .iter()
        .zip(authority.invocation())
    {
        if descriptor.ordinal != supplied.ordinal || descriptor.name != supplied.name {
            return Err(InvocationShapeError::InvalidRelationAuthority);
        }
        let value = match (descriptor.kind, descriptor.access, supplied.value) {
            (
                RelationChallengeAbiArgumentKind::DeviceConstPointerU32,
                RelationChallengeAbiAccess::ReadDrawnZAlpha,
                RelationChallengeInvocationValue::Role(RelationChallengeValueRole::DrawnZAlpha),
            ) => AotArgumentValue::DevicePointer(Some(bindings[0].binding)),
            (
                RelationChallengeAbiArgumentKind::DeviceMutPointerU32,
                RelationChallengeAbiAccess::WriteAlphaPowers,
                RelationChallengeInvocationValue::Role(RelationChallengeValueRole::AlphaPowers),
            ) => AotArgumentValue::DevicePointer(Some(bindings[1].binding)),
            (
                RelationChallengeAbiArgumentKind::U32,
                RelationChallengeAbiAccess::AlphaPowerCount,
                RelationChallengeInvocationValue::U32(value),
            ) if value == authority.max_alpha_powers() => AotArgumentValue::U32(value),
            (
                RelationChallengeAbiArgumentKind::DeviceMutPointerU32,
                RelationChallengeAbiAccess::WriteChallengeZ,
                RelationChallengeInvocationValue::Role(RelationChallengeValueRole::ChallengeZ),
            ) => AotArgumentValue::DevicePointer(Some(bindings[2].binding)),
            (
                RelationChallengeAbiArgumentKind::CudaStream,
                RelationChallengeAbiAccess::OrderedExecutionStream,
                RelationChallengeInvocationValue::OrderedStream,
            ) => continue,
            _ => return Err(InvocationShapeError::InvalidRelationAuthority),
        };
        let ordinal =
            u8::try_from(arguments.len()).map_err(|_| InvocationShapeError::SizeOverflow)?;
        arguments.push(AotArgumentBinding { ordinal, value });
    }
    if arguments.len() + 1 != authority.invocation().len() {
        return Err(InvocationShapeError::InvalidRelationAuthority);
    }
    Ok(AotInvocation { arguments })
}

fn challenge_binding(
    binding: EffectBindingId,
    arena: RelationArenaRange,
    version: ValueVersion,
) -> Result<BoundValueRange, InvocationShapeError> {
    let end = arena
        .start_word
        .checked_add(arena.words)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let elements = ElementRange::new(arena.start_word, end)
        .ok_or(InvocationShapeError::InvalidRelationBinding)?;
    Ok(BoundValueRange {
        binding,
        value: ValueRange { version, elements },
    })
}

fn validate_exact_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
) -> Result<(), InvocationShapeError> {
    if invocation.arguments.is_empty()
        || invocation
            .arguments
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| usize::from(argument.ordinal) != ordinal)
    {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|bound| bound.binding)
        .collect::<BTreeSet<_>>();
    let actual = invocation
        .arguments
        .iter()
        .filter_map(|argument| match argument.value {
            AotArgumentValue::DevicePointer(Some(binding)) => Some(binding),
            AotArgumentValue::U32(_) => None,
            _ => Some(EffectBindingId(u32::MAX)),
        })
        .collect::<BTreeSet<_>>();
    (actual == expected)
        .then_some(())
        .ok_or(InvocationShapeError::InvalidRelationBinding)
}
