//! Exact semantic projection of the generated direct Base commitment.
//!
//! The upstream program remains the schedule authority. This module only
//! binds its address-free roles to the sealed arena and projects each ordinary
//! wrapper into the canonical `CompiledProof` invocation/effect vocabulary.

use stwo_backend_cuda::{
    BaseCommitAccessKind, BaseCommitOperation, BaseCommitProgramAuthority, BaseCommitValueRole,
};

use super::*;
use crate::arena_plan::{ArenaBinding, CommitmentTreeId, ProofArenaPlan};
use crate::compiled_proof::{
    AotInvocation, EffectBindingId, EffectContract, StaticCudaWrapperAuthority,
    StaticCudaWrapperId, ValueVersion,
};

mod bindings;
mod invocation;
mod semantic;
mod static_execution;
#[cfg(test)]
mod tests;

use bindings::BaseCommitInventory;

const RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.lowered-base-commit.v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredBaseCommitAccess {
    pub(super) authority_index: u32,
    pub(super) kind: BaseCommitAccessKind,
    pub(super) role: BaseCommitValueRole,
    pub(super) arena: ArenaBinding,
    pub(super) binding: EffectBindingId,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredBaseCommitScratch {
    pub(super) binding: EffectBindingId,
    pub(super) version: ValueVersion,
    pub(super) words: usize,
    pub(super) alignment_words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredBaseCommitOperation {
    ordinal: u32,
    authority: BaseCommitOperation,
    accesses: Vec<LoweredBaseCommitAccess>,
    scratch: Option<LoweredBaseCommitScratch>,
    invocation: AotInvocation,
    effect: EffectContract,
}

impl LoweredBaseCommitOperation {
    pub(super) const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub(super) const fn authority(&self) -> &BaseCommitOperation {
        &self.authority
    }

    pub(super) fn accesses(&self) -> &[LoweredBaseCommitAccess] {
        &self.accesses
    }

    pub(super) const fn scratch(&self) -> Option<LoweredBaseCommitScratch> {
        self.scratch
    }

    pub(super) const fn invocation(&self) -> &AotInvocation {
        &self.invocation
    }

    pub(super) const fn effect(&self) -> &EffectContract {
        &self.effect
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredBaseCommit {
    authority: BaseCommitProgramAuthority,
    operations: Vec<LoweredBaseCommitOperation>,
    digest: [u8; 32],
}

impl LoweredBaseCommit {
    pub(super) const fn authority(&self) -> &BaseCommitProgramAuthority {
        &self.authority
    }

    pub(super) fn operations(&self) -> &[LoweredBaseCommitOperation] {
        &self.operations
    }

    pub(super) const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Lower the complete Base commit or leave the caller's semantic map untouched.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredBaseCommit, InvocationShapeError> {
    let planned = arena
        .commitment(CommitmentTreeId::Base)
        .ok_or(InvocationShapeError::InvalidBaseCommitAuthority)?;
    let commit = planned
        .commit_program
        .as_ref()
        .ok_or(InvocationShapeError::InvalidBaseCommitAuthority)?;
    let direct = planned
        .direct_retained_b2n_program
        .as_ref()
        .ok_or(InvocationShapeError::InvalidBaseCommitAuthority)?;
    let authority = BaseCommitProgramAuthority::compile(commit, direct)
        .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?;
    authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?;
    let inventory = BaseCommitInventory::compile(arena, planned, &authority)?;

    let mut next_values = values.clone();
    inventory.install_external_inputs(&mut next_values)?;
    let operations = semantic::lower_operations(arena, &authority, &inventory, &mut next_values)?;
    let digest = receipt_digest(&authority, &operations)?;
    let lowered = LoweredBaseCommit {
        authority,
        operations,
        digest,
    };
    validate_receipt(&lowered)?;
    *values = next_values;
    Ok(lowered)
}

/// Rebuild from the pre-stage allocator. This is the append cursor's equality
/// gate and catches both receipt mutation and allocator drift.
pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredBaseCommit,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidBaseCommitBinding)
    }
}

/// Resolve one already-lowered operation against the embedded static build.
/// `None` is a retryable publication frontier, never a semantic fallback.
pub(super) fn resolve_static_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &LoweredBaseCommit,
    operation_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    validate_receipt(lowered)?;
    let Some(linked) = lowered
        .authority
        .bind_static_build(target_sm)
        .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?
    else {
        return Ok(None);
    };
    static_execution::project_wrapper(id, &linked, lowered, operation_ordinal).map(Some)
}

pub(super) fn validate_receipt(lowered: &LoweredBaseCommit) -> Result<(), InvocationShapeError> {
    lowered
        .authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?;
    if lowered.operations.len() != lowered.authority.operations().len()
        || lowered
            .operations
            .iter()
            .zip(lowered.authority.operations())
            .enumerate()
            .any(|(ordinal, (local, exact))| {
                local.ordinal as usize != ordinal || &local.authority != exact
            })
        || receipt_digest(&lowered.authority, &lowered.operations)? != lowered.digest
    {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn tamper_receipt_digest_for_test(lowered: &mut LoweredBaseCommit) {
    lowered.digest[0] ^= 1;
}

fn receipt_digest(
    authority: &BaseCommitProgramAuthority,
    operations: &[LoweredBaseCommitOperation],
) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&authority.identity());
    hash_size(&mut hasher, operations.len())?;
    for operation in operations {
        hasher.update(&operation.ordinal.to_le_bytes());
        hasher.update(&operation.authority.identity);
        hasher.update(
            operation
                .invocation
                .contract_id()
                .map_err(|_| InvocationShapeError::InvalidBaseCommitBinding)?
                .as_bytes(),
        );
        hasher.update(operation.effect.id().as_bytes());
        hash_size(&mut hasher, operation.accesses.len())?;
        for access in &operation.accesses {
            hasher.update(&access.authority_index.to_le_bytes());
            hasher.update(&[access_kind_tag(access.kind)]);
            hash_role(&mut hasher, access.role);
            hasher.update(&access.arena.logical.0.to_le_bytes());
            hasher.update(&access.arena.physical.0.to_le_bytes());
            hash_size(&mut hasher, access.arena.len_words)?;
            hasher.update(&access.binding.0.to_le_bytes());
            hasher.update(&access.version.0.to_le_bytes());
        }
        match operation.scratch {
            None => {
                hasher.update(&[0]);
            }
            Some(scratch) => {
                hasher.update(&[1]);
                hasher.update(&scratch.binding.0.to_le_bytes());
                hasher.update(&scratch.version.0.to_le_bytes());
                hash_size(&mut hasher, scratch.words)?;
                hash_size(&mut hasher, scratch.alignment_words)?;
            }
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_size(hasher: &mut blake3::Hasher, value: usize) -> Result<(), InvocationShapeError> {
    hasher.update(
        &u64::try_from(value)
            .map_err(|_| InvocationShapeError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn access_kind_tag(kind: BaseCommitAccessKind) -> u8 {
    match kind {
        BaseCommitAccessKind::Read => 1,
        BaseCommitAccessKind::Write => 2,
        BaseCommitAccessKind::ReadWrite => 3,
    }
}

fn hash_role(hasher: &mut blake3::Hasher, role: BaseCommitValueRole) {
    match role {
        BaseCommitValueRole::SourceEvaluation { canonical_column } => {
            hasher.update(&[1]);
            hasher.update(&canonical_column.to_le_bytes());
        }
        BaseCommitValueRole::RetainedStageTwo { canonical_column } => {
            hasher.update(&[2]);
            hasher.update(&canonical_column.to_le_bytes());
        }
        BaseCommitValueRole::RetainedEvaluation { canonical_column } => {
            hasher.update(&[3]);
            hasher.update(&canonical_column.to_le_bytes());
        }
        BaseCommitValueRole::State { version, log_size } => {
            hasher.update(&[4]);
            hasher.update(&version.to_le_bytes());
            hasher.update(&log_size.to_le_bytes());
        }
        BaseCommitValueRole::HashLayer { log_size } => {
            hasher.update(&[5]);
            hasher.update(&log_size.to_le_bytes());
        }
    }
}
