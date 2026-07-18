//! Receipt and invocation gates for terminal Composition projection.

use super::*;

pub(super) fn required_upstream_catalogs(
    arena: &ProofArenaPlan,
) -> Result<Vec<ArenaCatalogValueId>, InvocationShapeError> {
    let authority = CompositionExecutionAuthority::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    authority
        .validate_against(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    let layouts = exact_layouts(&authority)?;
    [
        CompositionValueRole::InverseTwiddles,
        CompositionValueRole::ForwardTwiddles,
    ]
    .map(|role| {
        layouts
            .get(&role)
            .copied()
            .map(catalog)
            .ok_or(InvocationShapeError::InvalidCompositionBinding)
    })
    .into_iter()
    .collect()
}

pub(super) fn validate_receipt(
    arena: &ProofArenaPlan,
    waves: &LoweredCompositionWaves,
    lowered: &LoweredCompositionTerminal,
) -> Result<(), InvocationShapeError> {
    lowered
        .authority
        .validate_against(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    let first = lowered
        .authority
        .operations()
        .len()
        .checked_sub(lowered.operations.len())
        .ok_or(InvocationShapeError::InvalidCompositionAuthority)?;
    let expected_operations = lift_count(&lowered.authority) + 2;
    let expected_first = 2usize
        .checked_add(waves.waves().len())
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if waves.authority() != &lowered.authority
        || lowered.operations.len() != expected_operations
        || first != expected_first
        || lowered
            .operations
            .iter()
            .zip(&lowered.authority.operations()[first..])
            .enumerate()
            .any(|(index, (operation, exact))| {
                operation.operation_ordinal as usize != first + index
                    || &operation.operation != exact
                    || operation.bindings.len() != operation.effect.accesses().len()
            })
        || lowered
            .outputs
            .iter()
            .zip(lowered.authority.outputs())
            .any(|(output, role)| output.role != *role)
        || receipt_digest(
            &lowered.authority,
            waves,
            &lowered.operations,
            &lowered.outputs,
        )? != lowered.digest
    {
        return Err(InvocationShapeError::InvalidCompositionBinding);
    }
    Ok(())
}

pub(super) fn receipt_digest(
    authority: &CompositionExecutionAuthority,
    waves: &LoweredCompositionWaves,
    operations: &[LoweredCompositionTerminalOperation],
    outputs: &[LoweredCompositionTerminalOutput; RETAINED],
) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&authority.identity());
    hasher.update(&waves.digest());
    hasher.update(&(operations.len() as u64).to_le_bytes());
    for operation in operations {
        hasher.update(&operation.operation_ordinal.to_le_bytes());
        hasher.update(&operation.operation.identity);
        hasher.update(operation.effect.id().as_bytes());
        hasher.update(
            operation
                .invocation
                .contract_id()
                .map_err(|_| InvocationShapeError::InvalidCompositionBinding)?
                .as_bytes(),
        );
        for binding in &operation.bindings {
            hash_optional_role(&mut hasher, binding.source_role)?;
            hash_optional_role(&mut hasher, binding.destination_role)?;
            hasher.update(&binding.binding.0.to_le_bytes());
            hash_optional_role(&mut hasher, Some(binding.arena.role))?;
            hasher.update(&binding.arena.logical.0.to_le_bytes());
            hasher.update(&(binding.arena.first_word as u64).to_le_bytes());
            hasher.update(&(binding.arena.word_len as u64).to_le_bytes());
            hasher.update(&(binding.arena.alignment_words as u64).to_le_bytes());
            hash_optional_version(&mut hasher, binding.source_version);
            hash_optional_version(&mut hasher, binding.destination_version);
            hasher.update(&(binding.role_elements.start as u64).to_le_bytes());
            hasher.update(&(binding.role_elements.end as u64).to_le_bytes());
            hasher.update(&(binding.elements.start as u64).to_le_bytes());
            hasher.update(&(binding.elements.end as u64).to_le_bytes());
            hasher.update(&[match binding.kind {
                CompositionAccessKind::Read => 0,
                CompositionAccessKind::Write => 1,
                CompositionAccessKind::ReadWriteRequired => 2,
            }]);
        }
    }
    for output in outputs {
        hash_optional_role(&mut hasher, Some(output.role))?;
        hash_optional_role(&mut hasher, Some(output.arena.role))?;
        hasher.update(&output.arena.logical.0.to_le_bytes());
        hasher.update(&(output.arena.first_word as u64).to_le_bytes());
        hasher.update(&(output.arena.word_len as u64).to_le_bytes());
        hasher.update(&(output.arena.alignment_words as u64).to_le_bytes());
        hasher.update(&output.version.0.to_le_bytes());
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_optional_role(
    hasher: &mut blake3::Hasher,
    role: Option<CompositionValueRole>,
) -> Result<(), InvocationShapeError> {
    let Some(role) = role else {
        hasher.update(&[0]);
        return Ok(());
    };
    hasher.update(&[1]);
    match role {
        CompositionValueRole::Accumulator {
            log_size,
            coordinate,
            generation,
        } => {
            hasher.update(&[0, coordinate, generation]);
            hasher.update(&log_size.to_le_bytes());
        }
        CompositionValueRole::SplitRetained {
            canonical_column,
            generation,
        } => {
            hasher.update(&[1, canonical_column, generation]);
        }
        CompositionValueRole::ForwardTwiddles => {
            hasher.update(&[2]);
        }
        CompositionValueRole::InverseTwiddles => {
            hasher.update(&[3]);
        }
        _ => return Err(InvocationShapeError::InvalidCompositionBinding),
    }
    Ok(())
}

fn hash_optional_version(hasher: &mut blake3::Hasher, version: Option<ValueVersion>) {
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

pub(super) fn validate_invocation_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
) -> Result<(), InvocationShapeError> {
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for (ordinal, argument) in invocation.arguments.iter().enumerate() {
        if ordinal != argument.ordinal as usize {
            return Err(InvocationShapeError::InvalidCompositionBinding);
        }
        match &argument.value {
            AotArgumentValue::U32(_) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => insert_once(&mut actual, *binding)?,
            AotArgumentValue::DevicePointerTable(entries) => {
                if entries.is_empty() {
                    return Err(InvocationShapeError::InvalidCompositionBinding);
                }
                for binding in entries.iter().flatten() {
                    insert_once(&mut actual, *binding)?;
                }
            }
            AotArgumentValue::DevicePointerRangeSetValue { ranges } => {
                if ranges.is_empty() {
                    return Err(InvocationShapeError::InvalidCompositionBinding);
                }
                for binding in ranges {
                    insert_once(&mut actual, *binding)?;
                }
            }
            _ => return Err(InvocationShapeError::InvalidCompositionBinding),
        }
    }
    (actual == expected)
        .then_some(())
        .ok_or(InvocationShapeError::InvalidCompositionBinding)
}

fn insert_once(
    bindings: &mut BTreeSet<EffectBindingId>,
    binding: EffectBindingId,
) -> Result<(), InvocationShapeError> {
    bindings
        .insert(binding)
        .then_some(())
        .ok_or(InvocationShapeError::InvalidCompositionBinding)
}
