//! Upstream BaseCommit effect and invocation projection.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::{
    BaseCommitAccess, BaseCommitAccessKind, BaseCommitAliasAuthority, BaseCommitAliasDiscipline,
    BaseCommitAliasRequirement, BaseCommitDependencyRange, BaseCommitDependencyRole,
    BaseCommitExecutionBuffer, BaseCommitExecutionStep, BaseCommitOperation,
    BaseCommitPointerTarget, BaseCommitProgramAuthority, BaseCommitValueRole,
};

use super::*;
use crate::compiled_proof::{
    BoundValueRange, EffectAccess, ElementRange, InPlaceAliasAuthority, InPlaceAliasId,
    InPlaceAliasRequirement, InPlaceDiscipline, ValueRange,
};

#[derive(Clone, Debug)]
pub(super) enum LocalPointer {
    Single(EffectBindingId),
    Table(Vec<EffectBindingId>),
    Stream,
}

pub(super) fn lower_operations(
    arena: &ProofArenaPlan,
    authority: &BaseCommitProgramAuthority,
    inventory: &BaseCommitInventory,
    values: &mut adapter::SemanticValueMap,
) -> Result<Vec<LoweredBaseCommitOperation>, InvocationShapeError> {
    let mut roles = initial_roles(authority, inventory, values)?;
    authority
        .operations()
        .iter()
        .enumerate()
        .map(|(ordinal, operation)| {
            lower_operation(
                arena,
                u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
                operation,
                inventory,
                values,
                &mut roles,
            )
        })
        .collect()
}

fn initial_roles(
    authority: &BaseCommitProgramAuthority,
    inventory: &BaseCommitInventory,
    values: &adapter::SemanticValueMap,
) -> Result<BTreeMap<BaseCommitValueRole, ValueVersion>, InvocationShapeError> {
    let mut roles = BTreeMap::new();
    for layout in authority.layouts() {
        if !matches!(layout.role, BaseCommitValueRole::SourceEvaluation { .. }) {
            continue;
        }
        let (catalog, _) = inventory.role(layout.role)?;
        let version = values.version(catalog)?;
        if roles.insert(layout.role, version).is_some() {
            return Err(InvocationShapeError::InvalidBaseCommitBinding);
        }
    }
    Ok(roles)
}

fn lower_operation(
    arena: &ProofArenaPlan,
    ordinal: u32,
    operation: &BaseCommitOperation,
    inventory: &BaseCommitInventory,
    values: &mut adapter::SemanticValueMap,
    roles: &mut BTreeMap<BaseCommitValueRole, ValueVersion>,
) -> Result<LoweredBaseCommitOperation, InvocationShapeError> {
    operation
        .effect
        .validate_for_abi(operation.abi)
        .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?;
    operation
        .invocation
        .validate(operation.abi, &operation.kind, &operation.effect)
        .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?;

    let aliases = alias_maps(&operation.effect.aliases, operation.effect.accesses.len())?;
    let mut next_binding = 0u32;
    let mut local_accesses = Vec::new();
    let mut access_bindings = vec![None; operation.effect.accesses.len()];
    let mut access_receipts = vec![None; operation.effect.accesses.len()];
    for (index, access) in operation.effect.accesses.iter().enumerate() {
        if aliases.destinations.contains(&index) {
            continue;
        }
        if let Some(&(alias_index, alias)) = aliases.sources.get(&index) {
            let destination_index = usize::try_from(alias.destination_access)
                .map_err(|_| InvocationShapeError::SizeOverflow)?;
            let destination = operation
                .effect
                .accesses
                .get(destination_index)
                .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
            let pair = lower_alias(
                alias_index,
                *alias,
                index,
                *access,
                destination_index,
                *destination,
                inventory,
                values,
                roles,
                &mut next_binding,
            )?;
            local_accesses.push(pair.effect);
            access_bindings[index] = Some(pair.source.binding);
            access_bindings[destination_index] = Some(pair.destination.binding);
            access_receipts[index] = Some(pair.source);
            access_receipts[destination_index] = Some(pair.destination);
            continue;
        }
        let receipt = lower_single(index, *access, inventory, values, roles, &mut next_binding)?;
        let bound = bound(receipt.binding, receipt.version, elements(*access)?);
        local_accesses.push(match access.kind {
            BaseCommitAccessKind::Read => EffectAccess::Read { source: bound },
            BaseCommitAccessKind::Write => EffectAccess::Write { destination: bound },
            BaseCommitAccessKind::ReadWrite => {
                return Err(InvocationShapeError::InvalidBaseCommitAuthority)
            }
        });
        access_bindings[index] = Some(receipt.binding);
        access_receipts[index] = Some(receipt);
    }
    if access_bindings.iter().any(Option::is_none) || access_receipts.iter().any(Option::is_none) {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    let access_bindings = access_bindings
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    let accesses = access_receipts
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;

    let (pointers, scratch) = lower_installed(
        arena,
        operation,
        inventory,
        values,
        &access_bindings,
        &mut next_binding,
        &mut local_accesses,
    )?;
    let effect = EffectContract::new(local_accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidBaseCommitBinding)?;
    let invocation = invocation::compile(operation, &pointers, inventory)?;
    invocation::validate_exact_bindings(&invocation, &effect)?;
    Ok(LoweredBaseCommitOperation {
        ordinal,
        authority: operation.clone(),
        accesses,
        scratch,
        invocation,
        effect,
    })
}

struct AliasMaps<'a> {
    sources: BTreeMap<usize, (u32, &'a BaseCommitAliasAuthority)>,
    destinations: BTreeSet<usize>,
}

fn alias_maps(
    aliases: &[BaseCommitAliasAuthority],
    access_count: usize,
) -> Result<AliasMaps<'_>, InvocationShapeError> {
    let mut sources = BTreeMap::new();
    let mut destinations = BTreeSet::new();
    for (alias_index, alias) in aliases.iter().enumerate() {
        let source =
            usize::try_from(alias.source_access).map_err(|_| InvocationShapeError::SizeOverflow)?;
        let destination = usize::try_from(alias.destination_access)
            .map_err(|_| InvocationShapeError::SizeOverflow)?;
        if source >= access_count
            || destination >= access_count
            || source >= destination
            || sources
                .insert(
                    source,
                    (
                        u32::try_from(alias_index)
                            .map_err(|_| InvocationShapeError::SizeOverflow)?,
                        alias,
                    ),
                )
                .is_some()
            || !destinations.insert(destination)
        {
            return Err(InvocationShapeError::InvalidBaseCommitBinding);
        }
    }
    Ok(AliasMaps {
        sources,
        destinations,
    })
}

struct LoweredAlias {
    effect: EffectAccess,
    source: LoweredBaseCommitAccess,
    destination: LoweredBaseCommitAccess,
}

#[allow(clippy::too_many_arguments)]
fn lower_alias(
    alias_index: u32,
    alias: BaseCommitAliasAuthority,
    source_index: usize,
    source: BaseCommitAccess,
    destination_index: usize,
    destination: BaseCommitAccess,
    inventory: &BaseCommitInventory,
    values: &mut adapter::SemanticValueMap,
    roles: &mut BTreeMap<BaseCommitValueRole, ValueVersion>,
    next_binding: &mut u32,
) -> Result<LoweredAlias, InvocationShapeError> {
    if source.kind != BaseCommitAccessKind::Read || destination.kind != BaseCommitAccessKind::Write
    {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    let (source_catalog, source_arena) = inventory.role(source.role)?;
    let (destination_catalog, destination_arena) = inventory.role(destination.role)?;
    let source_version = read_role(source.role, source_catalog, values, roles)?;
    let required = alias.requirement == BaseCommitAliasRequirement::Required;
    if required != (source_catalog == destination_catalog && source_arena == destination_arena)
        || !required && source_arena.physical == destination_arena.physical
    {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    let destination_version = if required {
        let (transition_source, destination_version) = values.transition(destination_catalog)?;
        if transition_source != source_version {
            return Err(InvocationShapeError::InvalidBaseCommitBinding);
        }
        destination_version
    } else {
        values.allocate_output(destination_catalog)?
    };
    insert_role(roles, destination.role, destination_version)?;

    let source_binding = EffectBindingId(*next_binding);
    *next_binding = next_binding
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let destination_binding = if required {
        source_binding
    } else {
        let binding = EffectBindingId(*next_binding);
        *next_binding = next_binding
            .checked_add(1)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        binding
    };
    let local_alias = InPlaceAliasAuthority {
        id: InPlaceAliasId(alias_index),
        requirement: match alias.requirement {
            BaseCommitAliasRequirement::Required => InPlaceAliasRequirement::Required,
            BaseCommitAliasRequirement::Optional => InPlaceAliasRequirement::Permitted,
        },
        discipline: match alias.discipline {
            BaseCommitAliasDiscipline::ExactLowerPrefixReadBeforeWrite => {
                InPlaceDiscipline::ExactLowerPrefixReadBeforeWrite
            }
            BaseCommitAliasDiscipline::ElementWiseReadBeforeWrite => {
                InPlaceDiscipline::ElementWiseReadBeforeWrite
            }
            BaseCommitAliasDiscipline::OrderedCompositeInPlace => {
                InPlaceDiscipline::OrderedCompositeInPlace
            }
        },
    };
    let source_receipt = receipt(
        source_index,
        source,
        source_arena,
        source_binding,
        source_version,
    )?;
    let destination_receipt = receipt(
        destination_index,
        destination,
        destination_arena,
        destination_binding,
        destination_version,
    )?;
    Ok(LoweredAlias {
        effect: EffectAccess::ReadWrite {
            source: bound(source_binding, source_version, elements(source)?),
            destination: bound(
                destination_binding,
                destination_version,
                elements(destination)?,
            ),
            in_place: Some(local_alias),
        },
        source: source_receipt,
        destination: destination_receipt,
    })
}

fn lower_single(
    index: usize,
    access: BaseCommitAccess,
    inventory: &BaseCommitInventory,
    values: &mut adapter::SemanticValueMap,
    roles: &mut BTreeMap<BaseCommitValueRole, ValueVersion>,
    next_binding: &mut u32,
) -> Result<LoweredBaseCommitAccess, InvocationShapeError> {
    let (catalog, arena) = inventory.role(access.role)?;
    let version = match access.kind {
        BaseCommitAccessKind::Read => read_role(access.role, catalog, values, roles)?,
        BaseCommitAccessKind::Write => {
            let version = values.allocate_output(catalog)?;
            insert_role(roles, access.role, version)?;
            version
        }
        BaseCommitAccessKind::ReadWrite => {
            return Err(InvocationShapeError::InvalidBaseCommitAuthority)
        }
    };
    let binding = EffectBindingId(*next_binding);
    *next_binding = next_binding
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    receipt(index, access, arena, binding, version)
}

fn read_role(
    role: BaseCommitValueRole,
    catalog: ArenaCatalogValueId,
    values: &adapter::SemanticValueMap,
    roles: &BTreeMap<BaseCommitValueRole, ValueVersion>,
) -> Result<ValueVersion, InvocationShapeError> {
    let version = *roles
        .get(&role)
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    if values.version(catalog)? != version {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    Ok(version)
}

fn insert_role(
    roles: &mut BTreeMap<BaseCommitValueRole, ValueVersion>,
    role: BaseCommitValueRole,
    version: ValueVersion,
) -> Result<(), InvocationShapeError> {
    roles
        .insert(role, version)
        .is_none()
        .then_some(())
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)
}

fn receipt(
    index: usize,
    access: BaseCommitAccess,
    arena: ArenaBinding,
    binding: EffectBindingId,
    version: ValueVersion,
) -> Result<LoweredBaseCommitAccess, InvocationShapeError> {
    Ok(LoweredBaseCommitAccess {
        authority_index: u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?,
        kind: access.kind,
        role: access.role,
        arena,
        binding,
        version,
    })
}

fn lower_installed(
    arena: &ProofArenaPlan,
    operation: &BaseCommitOperation,
    inventory: &BaseCommitInventory,
    values: &mut adapter::SemanticValueMap,
    access_bindings: &[EffectBindingId],
    next_binding: &mut u32,
    effects: &mut Vec<EffectAccess>,
) -> Result<(Vec<LocalPointer>, Option<LoweredBaseCommitScratch>), InvocationShapeError> {
    let mut pointers = Vec::with_capacity(operation.effect.pointer_bindings.len());
    let mut scratch = None;
    for pointer in &operation.effect.pointer_bindings {
        let local = match &pointer.target {
            BaseCommitPointerTarget::Values { access_indices } => {
                LocalPointer::Single(single_binding(access_indices, access_bindings)?)
            }
            BaseCommitPointerTarget::PointerTable {
                table,
                access_indices,
            } => {
                inventory.validate_pointer_table(arena, table.role, table.range)?;
                LocalPointer::Table(unique_bindings(access_indices, access_bindings)?)
            }
            BaseCommitPointerTarget::Installed { access } => {
                let binding = EffectBindingId(*next_binding);
                *next_binding = next_binding
                    .checked_add(1)
                    .ok_or(InvocationShapeError::SizeOverflow)?;
                match access.role {
                    BaseCommitDependencyRole::InverseTwiddles
                    | BaseCommitDependencyRole::ForwardTwiddles => {
                        if access.kind != BaseCommitAccessKind::Read {
                            return Err(InvocationShapeError::InvalidBaseCommitBinding);
                        }
                        let (catalog, _, elements) =
                            inventory.installed_value(access.role, access.range)?;
                        effects.push(EffectAccess::Read {
                            source: bound(binding, values.version(catalog)?, elements),
                        });
                    }
                    BaseCommitDependencyRole::InPlaceScratch => {
                        let BaseCommitDependencyRange::Whole { words } = access.range else {
                            return Err(InvocationShapeError::InvalidBaseCommitBinding);
                        };
                        if access.kind != BaseCommitAccessKind::ReadWrite || scratch.is_some() {
                            return Err(InvocationShapeError::InvalidBaseCommitBinding);
                        }
                        let version = values.allocate_ephemeral()?;
                        effects.push(EffectAccess::Write {
                            destination: bound(
                                binding,
                                version,
                                ElementRange::new(0, words)
                                    .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?,
                            ),
                        });
                        let output = LoweredBaseCommitScratch {
                            binding,
                            version,
                            words,
                            alignment_words: inventory.state_alignment_words(),
                        };
                        validate_scratch_fill(
                            operation,
                            pointer.argument_ordinal,
                            output,
                            &pointers,
                            effects,
                        )?;
                        scratch = Some(output);
                    }
                    BaseCommitDependencyRole::BatchSourcePointerTable { .. }
                    | BaseCommitDependencyRole::BatchRetainedPointerTable { .. } => {
                        return Err(InvocationShapeError::InvalidBaseCommitBinding)
                    }
                }
                LocalPointer::Single(binding)
            }
            BaseCommitPointerTarget::ExecutionStream => LocalPointer::Stream,
        };
        pointers.push(local);
    }
    Ok((pointers, scratch))
}

fn validate_scratch_fill(
    operation: &BaseCommitOperation,
    scratch_ordinal: u8,
    scratch: LoweredBaseCommitScratch,
    prior_pointers: &[LocalPointer],
    effects: &[EffectAccess],
) -> Result<(), InvocationShapeError> {
    let expected_bytes = u64::try_from(
        scratch
            .words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(InvocationShapeError::SizeOverflow)?,
    )
    .map_err(|_| InvocationShapeError::SizeOverflow)?;
    let Some(BaseCommitExecutionStep::DeviceCopyD2D {
        source:
            BaseCommitExecutionBuffer::WrapperArgument {
                ordinal: source_ordinal,
                byte_offset: 0,
            },
        destination:
            BaseCommitExecutionBuffer::WrapperArgument {
                ordinal: destination_ordinal,
                byte_offset: 0,
            },
        bytes,
    }) = operation.execution.first()
    else {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    };
    if *destination_ordinal != scratch_ordinal
        || *source_ordinal == scratch_ordinal
        || *bytes != expected_bytes
        || operation
            .execution
            .iter()
            .filter(|step| matches!(step, BaseCommitExecutionStep::DeviceCopyD2D { .. }))
            .count()
            != 1
    {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    let mut source_pointers = operation
        .effect
        .pointer_bindings
        .iter()
        .enumerate()
        .filter(|(_, binding)| binding.argument_ordinal == *source_ordinal);
    let (source_index, _) = source_pointers
        .next()
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    if source_pointers.next().is_some() {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    let LocalPointer::Single(source_binding) = prior_pointers
        .get(source_index)
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?
    else {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    };
    let mut readable = effects
        .iter()
        .filter_map(EffectAccess::source)
        .filter(|source| source.binding == *source_binding);
    let source = readable
        .next()
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    if readable.next().is_some() || source.value.elements.len() < scratch.words {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    Ok(())
}

fn single_binding(
    indices: &[u32],
    bindings: &[EffectBindingId],
) -> Result<EffectBindingId, InvocationShapeError> {
    let values = unique_bindings(indices, bindings)?;
    let [binding] = values.as_slice() else {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    };
    Ok(*binding)
}

fn unique_bindings(
    indices: &[u32],
    bindings: &[EffectBindingId],
) -> Result<Vec<EffectBindingId>, InvocationShapeError> {
    let mut unique = Vec::new();
    let mut seen = BTreeSet::new();
    for &index in indices {
        let binding = *bindings
            .get(usize::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?)
            .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
        if unique.last() == Some(&binding) {
            continue;
        }
        if !seen.insert(binding) {
            return Err(InvocationShapeError::InvalidBaseCommitBinding);
        }
        unique.push(binding);
    }
    (!unique.is_empty())
        .then_some(unique)
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)
}

fn elements(access: BaseCommitAccess) -> Result<ElementRange, InvocationShapeError> {
    ElementRange::new(
        access.first_word,
        access
            .first_word
            .checked_add(access.word_len)
            .ok_or(InvocationShapeError::SizeOverflow)?,
    )
    .ok_or(InvocationShapeError::InvalidBaseCommitBinding)
}

fn bound(
    binding: EffectBindingId,
    version: ValueVersion,
    elements: ElementRange,
) -> BoundValueRange {
    BoundValueRange {
        binding,
        value: ValueRange { version, elements },
    }
}
