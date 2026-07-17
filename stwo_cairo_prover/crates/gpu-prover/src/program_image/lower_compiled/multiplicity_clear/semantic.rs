//! Exact first-write SSA, effect and ABI projection for the batched clear.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    WitnessFeedClearAbiAccess, WitnessFeedClearAbiArgument, WitnessFeedClearAbiArgumentKind,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, BoundValueRange, EffectAccess, ValueRange,
};

pub(super) fn exact_clear_outputs(
    values: &mut adapter::SemanticValueMap,
    catalogs: impl IntoIterator<Item = ArenaCatalogValueId>,
) -> Result<Vec<ValueVersion>, InvocationShapeError> {
    let catalogs = catalogs.into_iter().collect::<Vec<_>>();
    if catalogs.is_empty()
        || catalogs.iter().copied().collect::<BTreeSet<_>>().len() != catalogs.len()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let existing = catalogs
        .iter()
        .map(|&catalog| values.versions_for(catalog).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let completed = existing.iter().all(|versions| versions.len() == 1);
    let not_started = existing.iter().all(Vec::is_empty);
    if !completed && !not_started {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }

    if not_started {
        values.extend_ordered(catalogs.iter().copied())?;
    }
    let outputs = catalogs
        .iter()
        .map(|&catalog| {
            let versions = values.versions_for(catalog).collect::<Vec<_>>();
            match versions.as_slice() {
                [output] if values.version(catalog) == Ok(*output) => Ok(*output),
                _ => Err(InvocationShapeError::InvalidScheduledProducerBinding),
            }
        })
        .collect::<Result<Vec<ValueVersion>, _>>()?;
    let (catalog_first, transitioned, fixed) = values.allocation_classes();
    if outputs.iter().any(|output| {
        !catalog_first.contains(output) || transitioned.contains(output) || fixed.contains(output)
    }) {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(outputs)
}

pub(super) fn effect(
    contract: &WitnessFeedClearContract,
    destinations: &[MultiplicityClearDestinationBinding],
    lengths_value: ValueVersion,
    lengths_binding: EffectBindingId,
) -> Result<EffectContract, InvocationShapeError> {
    validate_geometry(contract, destinations)?;
    if lengths_binding.0 as usize != destinations.len() {
        return Err(InvocationShapeError::InvalidAdapterEffect);
    }
    let mut accesses = Vec::with_capacity(destinations.len() + 1);
    accesses.extend(destinations.iter().map(|destination| EffectAccess::Write {
        destination: bound(
            destination.binding,
            destination.version,
            destination.elements,
        ),
    }));
    accesses.push(EffectAccess::Read {
        source: bound(
            lengths_binding,
            lengths_value,
            ElementRange::new(0, contract.effect_geometry().destination_lengths.len())
                .ok_or(InvocationShapeError::InvalidStructuredAbi)?,
        ),
    });
    EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidAdapterEffect)
}

pub(super) fn invocation(
    contract: &WitnessFeedClearContract,
    destinations: &[MultiplicityClearDestinationBinding],
    lengths_value: ValueVersion,
    lengths_binding: EffectBindingId,
) -> Result<AotInvocation, InvocationShapeError> {
    invocation_using_abi(
        contract,
        destinations,
        lengths_value,
        lengths_binding,
        contract.abi().arguments(),
    )
}

pub(super) fn invocation_using_abi(
    contract: &WitnessFeedClearContract,
    destinations: &[MultiplicityClearDestinationBinding],
    lengths_value: ValueVersion,
    lengths_binding: EffectBindingId,
    abi: &[WitnessFeedClearAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    validate_geometry(contract, destinations)?;
    if abi != contract.abi().arguments()
        || abi
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let mut arguments = Vec::with_capacity(abi.len().saturating_sub(1));
    for descriptor in abi {
        use WitnessFeedClearAbiAccess as Access;
        use WitnessFeedClearAbiArgumentKind as Kind;
        let value = match (
            descriptor.ordinal,
            descriptor.name,
            descriptor.kind,
            descriptor.access,
        ) {
            (0, "destinations_dev", Kind::DeviceMutPointerTableU32, Access::WriteDestinations) => {
                AotArgumentValue::DevicePointerTable(
                    destinations
                        .iter()
                        .map(|destination| Some(destination.binding))
                        .collect(),
                )
            }
            (1, "lengths_dev", Kind::DeviceConstPointerU32, Access::ReadDestinationLengths) => {
                AotArgumentValue::DeviceFixedU32 {
                    value: lengths_value,
                    binding: lengths_binding,
                }
            }
            (2, "n_destinations", Kind::U32, Access::DestinationCount) => AotArgumentValue::U32(
                u32::try_from(destinations.len())
                    .map_err(|_| InvocationShapeError::SizeOverflow)?,
            ),
            (3, "max_words", Kind::U32, Access::MaximumDestinationWords) => AotArgumentValue::U32(
                u32::try_from(contract.requirements().max_destination_words)
                    .map_err(|_| InvocationShapeError::SizeOverflow)?,
            ),
            (4, "stream", Kind::CudaStream, Access::OrderedExecutionStream) => continue,
            _ => return Err(InvocationShapeError::InvalidStructuredAbi),
        };
        arguments.push(AotArgumentBinding {
            ordinal: descriptor.ordinal,
            value,
        });
    }
    if arguments.len() + 1 != abi.len()
        || arguments
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    Ok(AotInvocation { arguments })
}

pub(super) fn validate_exact_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
    lengths_value: ValueVersion,
    lengths_binding: EffectBindingId,
) -> Result<(), InvocationShapeError> {
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for argument in &invocation.arguments {
        let mut insert = |binding| {
            actual
                .insert(binding)
                .then_some(())
                .ok_or(InvocationShapeError::InvalidAdapterEffect)
        };
        match &argument.value {
            AotArgumentValue::U32(_) => {}
            AotArgumentValue::DevicePointerTable(entries) => {
                if entries.is_empty() || entries.iter().any(Option::is_none) {
                    return Err(InvocationShapeError::InvalidAdapterEffect);
                }
                for &binding in entries.iter().flatten() {
                    insert(binding)?;
                }
            }
            AotArgumentValue::DeviceFixedU32 { value, binding }
                if (*value, *binding) == (lengths_value, lengths_binding) =>
            {
                insert(*binding)?;
            }
            AotArgumentValue::Usize(_)
            | AotArgumentValue::DevicePointer(_)
            | AotArgumentValue::DeviceFixedU32 { .. } => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
        }
    }
    if actual == expected {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidAdapterEffect)
    }
}

fn validate_geometry(
    contract: &WitnessFeedClearContract,
    destinations: &[MultiplicityClearDestinationBinding],
) -> Result<(), InvocationShapeError> {
    let geometry = contract.effect_geometry();
    if destinations.len() != geometry.destinations.len()
        || destinations.len() != geometry.destination_lengths.len()
        || destinations
            .iter()
            .zip(&geometry.destinations)
            .enumerate()
            .any(|(index, (destination, expected))| {
                destination.ordinal as usize != index
                    || destination.binding.0 as usize != index
                    || destination.elements.start != expected.write_start_words
                    || destination.elements.len() != expected.write_len_words
                    || destination.elements.len() != geometry.destination_lengths[index] as usize
            })
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    Ok(())
}

const fn bound(
    binding: EffectBindingId,
    version: ValueVersion,
    elements: ElementRange,
) -> BoundValueRange {
    BoundValueRange {
        binding,
        value: ValueRange { version, elements },
    }
}

#[cfg(test)]
pub(super) fn invocation_for_test(
    lowered: &LoweredMultiplicityClear,
    abi: &[WitnessFeedClearAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    invocation_using_abi(
        &lowered.contract,
        &lowered.destinations,
        lowered.lengths_value,
        lowered.lengths_binding,
        abi,
    )
}
