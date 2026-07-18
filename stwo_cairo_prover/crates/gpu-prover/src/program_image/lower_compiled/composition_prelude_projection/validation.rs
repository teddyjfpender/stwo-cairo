use super::*;

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
    let mut insert = |binding| {
        actual
            .insert(binding)
            .then_some(())
            .ok_or(InvocationShapeError::InvalidCompositionPreludeBinding)
    };
    for argument in &invocation.arguments {
        match &argument.value {
            AotArgumentValue::DevicePointer(Some(binding))
            | AotArgumentValue::DeviceFixedU32 { binding, .. } => insert(*binding)?,
            AotArgumentValue::DevicePointerTable(entries) => {
                for &binding in entries.iter().flatten() {
                    insert(binding)?;
                }
            }
            AotArgumentValue::DevicePointerRangeSetValue { ranges } => {
                for &binding in ranges {
                    insert(binding)?;
                }
            }
            AotArgumentValue::U32(_) => {}
            _ => return Err(InvocationShapeError::InvalidCompositionPreludeBinding),
        }
    }
    if actual != expected {
        return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
    }
    Ok(())
}

pub(super) fn offset_range(
    layout: CompositionLayout,
    relative: ElementRange,
) -> Result<ElementRange, InvocationShapeError> {
    Ok(ElementRange {
        start: layout
            .first_word
            .checked_add(relative.start)
            .ok_or(InvocationShapeError::SizeOverflow)?,
        end: layout
            .first_word
            .checked_add(relative.end)
            .ok_or(InvocationShapeError::SizeOverflow)?,
    })
}

pub(super) fn validate_receipt(
    arena: &ProofArenaPlan,
    lowered: &LoweredCompositionPrelude,
) -> Result<(), InvocationShapeError> {
    lowered
        .authority
        .validate_against(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionPreludeAuthority)?;
    let expected = lowered
        .authority
        .operations()
        .get(..2)
        .ok_or(InvocationShapeError::InvalidCompositionPreludeAuthority)?;
    if lowered.operations.len() != 2
        || lowered.operations.iter().zip(expected).enumerate().any(
            |(index, (operation, expected))| {
                operation.operation_ordinal as usize != index
                    || &operation.operation != expected
                    || operation.bindings.len() != operation.effect.accesses().len()
            },
        )
        || receipt_digest(&lowered.authority, &lowered.operations, &lowered.values)?
            != lowered.digest
    {
        return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
    }
    let expected_wave_inputs = lowered
        .authority
        .layouts()
        .iter()
        .filter(|layout| {
            matches!(
                layout.role,
                CompositionValueRole::ExtParam { .. }
                    | CompositionValueRole::RandomCoefficientPowers
            )
        })
        .count();
    let actual_wave_inputs = lowered
        .values
        .keys()
        .filter(|role| {
            matches!(
                role,
                CompositionValueRole::ExtParam { .. }
                    | CompositionValueRole::RandomCoefficientPowers
            )
        })
        .count();
    if actual_wave_inputs != expected_wave_inputs {
        return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
    }
    Ok(())
}

pub(super) fn receipt_digest(
    authority: &CompositionExecutionAuthority,
    operations: &[LoweredCompositionPreludeOperation],
    values: &BTreeMap<CompositionValueRole, LoweredCompositionPreludeValue>,
) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&authority.identity());
    hasher.update(&(operations.len() as u64).to_le_bytes());
    for operation in operations {
        hasher.update(&operation.operation_ordinal.to_le_bytes());
        hasher.update(&operation.operation.identity);
        hasher.update(operation.effect.id().as_bytes());
        hasher.update(
            operation
                .invocation
                .contract_id()
                .map_err(|_| InvocationShapeError::InvalidCompositionPreludeBinding)?
                .as_bytes(),
        );
        for binding in &operation.bindings {
            hash_role(&mut hasher, binding.role);
            hasher.update(&binding.binding.0.to_le_bytes());
            hasher.update(&binding.version.0.to_le_bytes());
            hasher.update(&(binding.role_elements.start as u64).to_le_bytes());
            hasher.update(&(binding.role_elements.end as u64).to_le_bytes());
            hasher.update(&(binding.elements.start as u64).to_le_bytes());
            hasher.update(&(binding.elements.end as u64).to_le_bytes());
        }
    }
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values.values() {
        hash_role(&mut hasher, value.role);
        hasher.update(&value.arena.logical.0.to_le_bytes());
        hasher.update(&(value.arena.first_word as u64).to_le_bytes());
        hasher.update(&(value.arena.word_len as u64).to_le_bytes());
        hasher.update(&value.version.0.to_le_bytes());
        match value.kind {
            LoweredCompositionPreludeValueKind::CatalogInput(catalog) => {
                hasher.update(&[0]);
                hasher.update(&catalog.0.to_le_bytes());
            }
            LoweredCompositionPreludeValueKind::Fixed => {
                hasher.update(&[1]);
            }
            LoweredCompositionPreludeValueKind::DynamicOutput => {
                hasher.update(&[2]);
            }
            LoweredCompositionPreludeValueKind::CatalogOutput(catalog) => {
                hasher.update(&[3]);
                hasher.update(&catalog.0.to_le_bytes());
            }
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_role(hasher: &mut blake3::Hasher, role: CompositionValueRole) {
    match role {
        CompositionValueRole::Descriptor { kind, index } => {
            hasher.update(&[0, descriptor_tag(kind)]);
            hasher.update(&index.to_le_bytes());
        }
        CompositionValueRole::RandomCoefficient => {
            hasher.update(&[1]);
        }
        CompositionValueRole::RandomCoefficientPowers => {
            hasher.update(&[2]);
        }
        CompositionValueRole::RelationZ => {
            hasher.update(&[3]);
        }
        CompositionValueRole::RelationAlphaPowers => {
            hasher.update(&[4]);
        }
        CompositionValueRole::ClaimedSum { component } => {
            hasher.update(&[5]);
            hasher.update(&component.to_le_bytes());
        }
        CompositionValueRole::ExtParam { component, slot } => {
            hasher.update(&[6]);
            hasher.update(&component.to_le_bytes());
            hasher.update(&slot.to_le_bytes());
        }
        CompositionValueRole::DirectEvaluation { plan_column } => {
            hasher.update(&[7]);
            hasher.update(&plan_column.to_le_bytes());
        }
        CompositionValueRole::Accumulator {
            log_size,
            coordinate,
            generation,
        } => {
            hasher.update(&[8, coordinate, generation]);
            hasher.update(&log_size.to_le_bytes());
        }
        CompositionValueRole::SplitRetained {
            canonical_column,
            generation,
        } => {
            hasher.update(&[9, canonical_column, generation]);
        }
        CompositionValueRole::ForwardTwiddles => {
            hasher.update(&[10]);
        }
        CompositionValueRole::InverseTwiddles => {
            hasher.update(&[11]);
        }
    }
}

const fn descriptor_tag(kind: CompositionDescriptorRole) -> u8 {
    match kind {
        CompositionDescriptorRole::DynamicSourceKinds => 0,
        CompositionDescriptorRole::DynamicSourceIndices => 1,
        CompositionDescriptorRole::DynamicScales => 2,
        CompositionDescriptorRole::WaveParts => 3,
        CompositionDescriptorRole::InteractionOffsets => 4,
        CompositionDescriptorRole::DenominatorInverses => 5,
        CompositionDescriptorRole::BaseParams => 6,
    }
}
