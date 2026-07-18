//! Transactional generated-Relation projection.
//!
//! The exact sequence is challenge expansion, fused body, segmented tail.
//! Prepared pointer/geometry storage stays address-free metadata rather than a
//! fabricated causal root. Only transcript output and real trace sources may
//! enter as pre-existing semantic values.

use stwo_backend_cuda::{
    RelationAccessKind, RelationChallengeExpansionAuthority, RelationExecutionAuthority,
    RelationValueOwnership, RelationValueRole, RelationWrapperExecution,
};

use super::*;
use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::ValueVersion;

mod inventory;
mod semantic;
#[cfg(test)]
mod tests;

use inventory::{RelationArenaRange, RelationInventory};

const RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.lowered-relation.v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredRelationRole {
    pub(super) role: RelationValueRole,
    pub(super) ownership: RelationValueOwnership,
    pub(super) arena: RelationArenaRange,
    pub(super) first_version: Option<ValueVersion>,
    pub(super) final_version: Option<ValueVersion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredRelationAccess {
    pub(super) authority_index: u32,
    pub(super) role: RelationValueRole,
    pub(super) kind: RelationAccessKind,
    pub(super) arena: RelationArenaRange,
    pub(super) source: Option<ValueVersion>,
    pub(super) destination: Option<ValueVersion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredRelationChallenge {
    authority: RelationChallengeExpansionAuthority,
    drawn: RelationArenaRange,
    drawn_version: ValueVersion,
    alpha: RelationArenaRange,
    alpha_version: ValueVersion,
    z: RelationArenaRange,
    z_version: ValueVersion,
}

impl LoweredRelationChallenge {
    pub(super) const fn authority(&self) -> &RelationChallengeExpansionAuthority {
        &self.authority
    }

    pub(super) const fn drawn_version(&self) -> ValueVersion {
        self.drawn_version
    }

    pub(super) const fn alpha_version(&self) -> ValueVersion {
        self.alpha_version
    }

    pub(super) const fn z_version(&self) -> ValueVersion {
        self.z_version
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredRelationWrapper {
    authority: RelationWrapperExecution,
    accesses: Vec<LoweredRelationAccess>,
}

impl LoweredRelationWrapper {
    pub(super) const fn authority(&self) -> &RelationWrapperExecution {
        &self.authority
    }

    pub(super) fn accesses(&self) -> &[LoweredRelationAccess] {
        &self.accesses
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredRelation {
    authority: RelationExecutionAuthority,
    challenge: LoweredRelationChallenge,
    roles: Vec<LoweredRelationRole>,
    wrappers: [LoweredRelationWrapper; 2],
    digest: [u8; 32],
}

impl LoweredRelation {
    pub(super) const fn authority(&self) -> &RelationExecutionAuthority {
        &self.authority
    }

    pub(super) const fn challenge(&self) -> &LoweredRelationChallenge {
        &self.challenge
    }

    pub(super) fn roles(&self) -> &[LoweredRelationRole] {
        &self.roles
    }

    pub(super) const fn wrappers(&self) -> &[LoweredRelationWrapper; 2] {
        &self.wrappers
    }

    pub(super) const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Lower expansion -> fused body -> segmented tail or publish nothing.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredRelation, InvocationShapeError> {
    let planned = arena.relation();
    let authority = RelationExecutionAuthority::compile(
        planned.execution.kernel_program(),
        &planned.requirements,
        planned.launch_mode,
        arena.protocol_identity().relation_tail_mode,
    )
    .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    let challenge =
        RelationChallengeExpansionAuthority::compile(authority.program().max_alpha_powers)
            .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    challenge
        .validate()
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    let inventory = RelationInventory::compile(arena, &authority, &challenge)?;

    let mut next_values = values.clone();
    let (challenge, roles, wrappers) =
        semantic::lower(&authority, &challenge, &inventory, &mut next_values)?;
    let mut lowered = LoweredRelation {
        authority,
        challenge,
        roles,
        wrappers,
        digest: [0; 32],
    };
    lowered.digest = receipt_digest(&lowered)?;
    validate_receipt(&lowered)?;
    *values = next_values;
    Ok(lowered)
}

/// Rebuild from the pre-Relation allocator and require exact post-state.
pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredRelation,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidRelationBinding)
    }
}

fn validate_receipt(lowered: &LoweredRelation) -> Result<(), InvocationShapeError> {
    lowered
        .authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    lowered
        .challenge
        .authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    if lowered.challenge.authority.max_alpha_powers()
        != lowered.authority.program().max_alpha_powers
        || lowered.roles.len() != lowered.authority.values().len()
        || lowered
            .roles
            .iter()
            .zip(lowered.authority.values())
            .any(|(local, exact)| {
                local.role != exact.role
                    || local.ownership != exact.ownership
                    || local.arena.words != exact.words
            })
        || lowered
            .wrappers
            .iter()
            .zip(lowered.authority.wrappers())
            .any(|(local, exact)| {
                &local.authority != exact || local.accesses.len() != exact.accesses.len()
            })
        || receipt_digest(lowered)? != lowered.digest
    {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    Ok(())
}

fn receipt_digest(lowered: &LoweredRelation) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&lowered.authority.identity());
    hasher.update(&lowered.challenge.authority.identity());
    hash_range(&mut hasher, lowered.challenge.drawn);
    hash_version(&mut hasher, Some(lowered.challenge.drawn_version));
    hash_range(&mut hasher, lowered.challenge.alpha);
    hash_version(&mut hasher, Some(lowered.challenge.alpha_version));
    hash_range(&mut hasher, lowered.challenge.z);
    hash_version(&mut hasher, Some(lowered.challenge.z_version));
    hash_size(&mut hasher, lowered.roles.len())?;
    for role in &lowered.roles {
        hash_role(&mut hasher, role.role);
        hasher.update(&[role.ownership as u8]);
        hash_range(&mut hasher, role.arena);
        hash_version(&mut hasher, role.first_version);
        hash_version(&mut hasher, role.final_version);
    }
    hash_size(&mut hasher, lowered.wrappers.len())?;
    for wrapper in &lowered.wrappers {
        hasher.update(&[wrapper.authority.stage as u8]);
        hash_size(&mut hasher, wrapper.accesses.len())?;
        for access in &wrapper.accesses {
            hasher.update(&access.authority_index.to_le_bytes());
            hash_role(&mut hasher, access.role);
            hasher.update(&[access.kind as u8]);
            hash_range(&mut hasher, access.arena);
            hash_version(&mut hasher, access.source);
            hash_version(&mut hasher, access.destination);
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_range(hasher: &mut blake3::Hasher, range: RelationArenaRange) {
    hasher.update(&range.catalog.0.to_le_bytes());
    hasher.update(&range.arena.logical.0.to_le_bytes());
    hasher.update(&range.arena.physical.0.to_le_bytes());
    hasher.update(&(range.start_word as u64).to_le_bytes());
    hasher.update(&(range.words as u64).to_le_bytes());
}

fn hash_version(hasher: &mut blake3::Hasher, version: Option<ValueVersion>) {
    match version {
        Some(version) => {
            hasher.update(&[1]);
            hasher.update(&version.0.to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hash_size(hasher: &mut blake3::Hasher, value: usize) -> Result<(), InvocationShapeError> {
    hasher.update(
        &u64::try_from(value)
            .map_err(|_| InvocationShapeError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn hash_role(hasher: &mut blake3::Hasher, role: RelationValueRole) {
    inventory::hash_role(hasher, role);
}
