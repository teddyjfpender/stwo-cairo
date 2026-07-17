use std::collections::BTreeSet;

use super::*;

pub(super) fn validate(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    validate_fixed_values(input)?;
    validate_module_initializers(input)?;
    validate_partitions(input)
}

fn validate_fixed_values(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    if input
        .fixed_values
        .windows(2)
        .any(|pair| pair[0].constant() >= pair[1].constant())
    {
        return Err(CompiledProofError::NonCanonicalFixedValues);
    }
    let mut constants = input
        .values
        .iter()
        .filter_map(|value| match value.origin {
            ValueOrigin::Constant(constant) => Some((constant, value.version)),
            _ => None,
        })
        .collect::<Vec<_>>();
    constants.sort_unstable_by_key(|(constant, _)| *constant);
    if input
        .fixed_values
        .iter()
        .map(|fixed| (fixed.constant(), fixed.value()))
        .ne(constants)
    {
        return Err(CompiledProofError::NonCanonicalFixedValues);
    }

    for fixed in &input.fixed_values {
        let value = super::value(input, fixed.value())?;
        if value.origin != ValueOrigin::Constant(fixed.constant())
            || value.region != Region::FixedData
        {
            return Err(CompiledProofError::InvalidFixedValue);
        }
        let logical_bytes = value.layout.logical_bytes()?;
        match fixed.initializer() {
            FixedValueInitializer::InlineBytes(bytes) => {
                if bytes.len() != logical_bytes
                    || fixed.content_digest()
                        != &super::super::structural_authority::fixed_content_digest(bytes)
                {
                    return Err(CompiledProofError::InvalidFixedValue);
                }
            }
            FixedValueInitializer::InlineU32(words) => {
                let bytes = words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>();
                if value.layout.element != ElementType::U32
                    || value.alignment < core::mem::align_of::<u32>()
                    || words.len() != value.layout.element_count()?
                    || fixed.content_digest()
                        != &super::super::structural_authority::fixed_content_digest(&bytes)
                {
                    return Err(CompiledProofError::InvalidFixedValue);
                }
            }
            FixedValueInitializer::DeterministicRecipe(recipe) => {
                if recipe.is_empty() || logical_bytes == 0 {
                    return Err(CompiledProofError::InvalidFixedValue);
                }
            }
        }
    }
    Ok(())
}

fn validate_module_initializers(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    let mut symbols = BTreeSet::new();
    for (index, initializer) in input.module_global_initializers.iter().enumerate() {
        let expected = ModuleGlobalInitializerId(
            u32::try_from(index).map_err(|_| CompiledProofError::SizeOverflow)?,
        );
        if initializer.id() != expected
            || !initializer.has_valid_identity()?
            || !symbols.insert((initializer.module(), initializer.symbol()))
        {
            return Err(CompiledProofError::NonCanonicalModuleGlobalInitializers);
        }
        for atom in initializer.atoms() {
            if let ModuleGlobalInitializerAtom::FixedValueAddress {
                value,
                source_byte_offset,
                ..
            } = atom
            {
                let fixed = input
                    .fixed_values
                    .iter()
                    .find(|fixed| fixed.value() == *value)
                    .ok_or(CompiledProofError::InvalidModuleGlobalInitializer)?;
                let desc = super::value(input, fixed.value())?;
                if *source_byte_offset >= desc.layout.logical_bytes()?
                    || *source_byte_offset % desc.layout.element.bytes != 0
                {
                    return Err(CompiledProofError::InvalidModuleGlobalInitializer);
                }
            }
        }
    }

    let declared = input
        .module_global_initializers
        .iter()
        .map(ModuleGlobalInitializer::id)
        .collect::<Vec<_>>();
    let used = input
        .effects
        .iter()
        .flat_map(EffectContract::module_globals)
        .map(|global| global.initializer)
        .collect::<BTreeSet<_>>();
    if declared.iter().copied().ne(used.iter().copied()) {
        return Err(CompiledProofError::NonCanonicalModuleGlobalInitializers);
    }
    Ok(())
}

fn validate_partitions(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    if input
        .partitions
        .windows(2)
        .any(|pair| pair[0].id() >= pair[1].id())
    {
        return Err(CompiledProofError::NonCanonicalPartitionAuthority);
    }
    for partition in &input.partitions {
        if !partition.has_valid_identity()? {
            return Err(CompiledProofError::InvalidPartitionAuthority);
        }
    }
    let declared = input
        .partitions
        .iter()
        .map(PartitionAuthority::id)
        .collect::<Vec<_>>();
    let used = input
        .operations
        .iter()
        .map(|operation| operation.partition)
        .collect::<BTreeSet<_>>();
    if declared.iter().copied().ne(used.iter().copied()) {
        return Err(CompiledProofError::NonCanonicalPartitionAuthority);
    }
    for operation in &input.operations {
        if input
            .partitions
            .binary_search_by_key(&operation.partition, PartitionAuthority::id)
            .is_err()
        {
            return Err(CompiledProofError::UnknownPartitionAuthority {
                operation: operation.id,
            });
        }
    }
    Ok(())
}
