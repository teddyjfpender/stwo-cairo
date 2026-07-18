//! Ordered semantic lowering for expansion, fused body, and segmented tail.

use stwo_backend_cuda::{
    RelationAccess, RelationAccessKind, RelationChallengeExpansionAuthority,
    RelationExecutionAuthority, RelationValueOwnership, RelationValueRole,
};

use super::*;

#[derive(Clone, Debug)]
struct RoleState {
    role: RelationValueRole,
    ownership: RelationValueOwnership,
    arena: RelationArenaRange,
    first: Option<ValueVersion>,
    current: Option<ValueVersion>,
    transitioned: bool,
}

pub(super) fn lower(
    authority: &RelationExecutionAuthority,
    challenge_authority: &RelationChallengeExpansionAuthority,
    inventory: &RelationInventory,
    values: &mut adapter::SemanticValueMap,
) -> Result<
    (
        LoweredRelationChallenge,
        Vec<LoweredRelationRole>,
        [LoweredRelationWrapper; 2],
    ),
    InvocationShapeError,
> {
    let challenge = challenge_execution::lower(challenge_authority, inventory, values)?;
    let descriptors = values.register_fixed_u32(authority.descriptor_words().to_vec())?;
    let geometry = values.register_fixed_u32(authority.geometry_words().to_vec())?;
    let mut states = initial_states(authority, inventory, &challenge, values)?;
    install_fixed_metadata(&mut states, descriptors, geometry)?;
    let body_accesses =
        lower_wrapper_accesses(&authority.wrappers()[0], inventory, values, &mut states)?;
    let tail_accesses =
        lower_wrapper_accesses(&authority.wrappers()[1], inventory, values, &mut states)?;
    let roles = finish_roles(authority, states)?;
    let body =
        wrapper_execution::lower(authority, &authority.wrappers()[0], body_accesses, &roles)?;
    let tail =
        wrapper_execution::lower(authority, &authority.wrappers()[1], tail_accesses, &roles)?;
    Ok((challenge, roles, [body, tail]))
}

fn install_fixed_metadata(
    states: &mut [RoleState],
    descriptors: ValueVersion,
    geometry: ValueVersion,
) -> Result<(), InvocationShapeError> {
    for (role, version) in [
        (RelationValueRole::Descriptors, descriptors),
        (RelationValueRole::Geometry, geometry),
    ] {
        let index = unique_role_index(states, role)?;
        let state = states
            .get_mut(index)
            .ok_or(InvocationShapeError::InvalidRelationBinding)?;
        if state.ownership != RelationValueOwnership::PreparedMetadata
            || state.first.is_some()
            || state.current.is_some()
        {
            return Err(InvocationShapeError::InvalidRelationAuthority);
        }
        state.first = Some(version);
        state.current = Some(version);
    }
    Ok(())
}

fn initial_states(
    authority: &RelationExecutionAuthority,
    inventory: &RelationInventory,
    challenge: &LoweredRelationChallenge,
    values: &adapter::SemanticValueMap,
) -> Result<Vec<RoleState>, InvocationShapeError> {
    authority
        .values()
        .iter()
        .map(|layout| {
            let arena = inventory.role(layout.role)?;
            let version = match layout.ownership {
                RelationValueOwnership::ExternalSource => Some(values.version(arena.catalog)?),
                RelationValueOwnership::TranscriptChallenge => Some(match layout.role {
                    RelationValueRole::AlphaPowers => challenge.alpha_version,
                    RelationValueRole::ChallengeZ => challenge.z_version,
                    _ => return Err(InvocationShapeError::InvalidRelationAuthority),
                }),
                RelationValueOwnership::PreparedMetadata
                | RelationValueOwnership::ExecutionOutput
                | RelationValueOwnership::ExecutionScratch
                | RelationValueOwnership::ReservedUnused => None,
            };
            Ok(RoleState {
                role: layout.role,
                ownership: layout.ownership,
                arena,
                first: version,
                current: version,
                transitioned: false,
            })
        })
        .collect()
}

fn lower_wrapper_accesses(
    authority: &RelationWrapperExecution,
    inventory: &RelationInventory,
    values: &mut adapter::SemanticValueMap,
    states: &mut [RoleState],
) -> Result<Vec<LoweredRelationAccess>, InvocationShapeError> {
    if authority.accesses
        != authority
            .children
            .iter()
            .flat_map(|child| child.accesses.iter().copied())
            .collect::<Vec<_>>()
    {
        return Err(InvocationShapeError::InvalidRelationAuthority);
    }
    authority
        .accesses
        .iter()
        .enumerate()
        .map(|(index, access)| {
            lower_access(
                u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?,
                *access,
                inventory,
                values,
                states,
            )
        })
        .collect::<Result<Vec<_>, _>>()
}

fn lower_access(
    authority_index: u32,
    access: RelationAccess,
    inventory: &RelationInventory,
    values: &mut adapter::SemanticValueMap,
    states: &mut [RoleState],
) -> Result<LoweredRelationAccess, InvocationShapeError> {
    let role_index = unique_role_index(states, access.role)?;
    let state = &mut states[role_index];
    let whole = inventory.role(access.role)?;
    let end = access
        .start_word
        .checked_add(access.words)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if end > whole.words || access.words == 0 {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    let arena = RelationArenaRange {
        catalog: whole.catalog,
        arena: whole.arena,
        start_word: whole
            .start_word
            .checked_add(access.start_word)
            .ok_or(InvocationShapeError::SizeOverflow)?,
        words: access.words,
    };
    let (source, destination) = match access.kind {
        RelationAccessKind::Read => (read(state)?, None),
        RelationAccessKind::Write => (None, Some(write(state, values)?)),
        RelationAccessKind::ReadWrite => read_write(state, values)?,
    };
    Ok(LoweredRelationAccess {
        authority_index,
        role: access.role,
        kind: access.kind,
        arena,
        source,
        destination,
    })
}

fn read(state: &RoleState) -> Result<Option<ValueVersion>, InvocationShapeError> {
    match state.ownership {
        RelationValueOwnership::PreparedMetadata => Ok(state.current),
        RelationValueOwnership::ExternalSource
        | RelationValueOwnership::TranscriptChallenge
        | RelationValueOwnership::ExecutionOutput
        | RelationValueOwnership::ExecutionScratch => state
            .current
            .map(Some)
            .ok_or(InvocationShapeError::InvalidRelationBinding),
        RelationValueOwnership::ReservedUnused => {
            Err(InvocationShapeError::InvalidRelationAuthority)
        }
    }
}

fn write(
    state: &mut RoleState,
    values: &mut adapter::SemanticValueMap,
) -> Result<ValueVersion, InvocationShapeError> {
    if state.current.is_some() {
        return Err(InvocationShapeError::InvalidRelationAuthority);
    }
    let version = match state.ownership {
        RelationValueOwnership::ExecutionOutput => values.allocate_output(state.arena.catalog)?,
        RelationValueOwnership::ExecutionScratch => values.allocate_ephemeral()?,
        _ => return Err(InvocationShapeError::InvalidRelationAuthority),
    };
    state.first = Some(version);
    state.current = Some(version);
    Ok(version)
}

fn read_write(
    state: &mut RoleState,
    values: &mut adapter::SemanticValueMap,
) -> Result<(Option<ValueVersion>, Option<ValueVersion>), InvocationShapeError> {
    let source = state
        .current
        .ok_or(InvocationShapeError::InvalidRelationBinding)?;
    match state.ownership {
        RelationValueOwnership::ExecutionOutput if !state.transitioned => {
            let (exact_source, destination) = values.transition(state.arena.catalog)?;
            if exact_source != source {
                return Err(InvocationShapeError::InvalidRelationBinding);
            }
            state.current = Some(destination);
            state.transitioned = true;
            Ok((Some(source), Some(destination)))
        }
        RelationValueOwnership::ExecutionOutput | RelationValueOwnership::ExecutionScratch
            if state.transitioned
                || state.ownership == RelationValueOwnership::ExecutionScratch =>
        {
            Ok((Some(source), Some(source)))
        }
        _ => Err(InvocationShapeError::InvalidRelationAuthority),
    }
}

fn unique_role_index(
    states: &[RoleState],
    role: RelationValueRole,
) -> Result<usize, InvocationShapeError> {
    let mut matches = states
        .iter()
        .enumerate()
        .filter_map(|(index, state)| (state.role == role).then_some(index));
    let exact = matches
        .next()
        .ok_or(InvocationShapeError::InvalidRelationBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    Ok(exact)
}

fn finish_roles(
    authority: &RelationExecutionAuthority,
    states: Vec<RoleState>,
) -> Result<Vec<LoweredRelationRole>, InvocationShapeError> {
    if states.len() != authority.values().len() {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    states
        .into_iter()
        .zip(authority.values())
        .map(|(state, layout)| {
            if state.role != layout.role
                || state.ownership != layout.ownership
                || match state.ownership {
                    RelationValueOwnership::ExternalSource
                    | RelationValueOwnership::TranscriptChallenge
                    | RelationValueOwnership::ExecutionOutput
                    | RelationValueOwnership::ExecutionScratch => {
                        state.first.is_none() || state.current.is_none()
                    }
                    RelationValueOwnership::PreparedMetadata => match state.role {
                        RelationValueRole::Descriptors | RelationValueRole::Geometry => {
                            state.first.is_none()
                                || state.current.is_none()
                                || state.first != state.current
                        }
                        _ => state.first.is_some() || state.current.is_some(),
                    },
                    RelationValueOwnership::ReservedUnused => {
                        state.first.is_some() || state.current.is_some()
                    }
                }
            {
                return Err(InvocationShapeError::InvalidRelationBinding);
            }
            Ok(LoweredRelationRole {
                role: state.role,
                ownership: state.ownership,
                arena: state.arena,
                first_version: state.first,
                final_version: state.current,
            })
        })
        .collect()
}
