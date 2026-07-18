//! Exact effect and static-wrapper ABI projection.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    WitnessFeedAbiAccess, WitnessFeedAbiArgument, WitnessFeedAbiArgumentKind,
    WitnessFeedDestinationEffect,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AtomicOperation, BoundValueRange, EffectAccess,
    InPlaceAliasId, InPlaceAliasRequirement, InPlaceDiscipline, ValueRange,
};

pub(super) fn transition_destination(
    exact: bindings::ExactDestination<'_>,
    geometry: &WitnessFeedDestinationEffect,
    ordinal: usize,
    binding: EffectBindingId,
    values: &mut adapter::SemanticValueMap,
) -> Result<MultiplicityFeedTransition, InvocationShapeError> {
    if geometry.destination_ordinal as usize != ordinal
        || geometry.atomic_start_words != 0
        || geometry.atomic_len_words != exact.value.words
        || geometry.may_write_ranges.is_empty()
        || geometry.may_write_ranges.iter().any(|range| {
            range.len_words == 0
                || range
                    .start_words
                    .checked_add(range.len_words)
                    .is_none_or(|end| end > exact.value.words)
        })
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let lineage = values.versions_for(exact.value.id).collect::<Vec<_>>();
    let (catalog_first, _, fixed) = values.allocation_classes();
    if lineage.is_empty()
        || !catalog_first.contains(&lineage[0])
        || fixed.iter().any(|version| lineage.contains(version))
        || values.version(exact.value.id)
            != lineage
                .last()
                .copied()
                .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let (source, destination) = values.transition(exact.value.id)?;
    let elements = ElementRange::new(0, exact.value.words)
        .ok_or(InvocationShapeError::InvalidStructuredAbi)?;
    Ok(MultiplicityFeedTransition {
        ordinal: u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
        name: exact.name,
        arena: ArenaBinding {
            logical: exact.value.logical,
            physical: exact.value.physical,
            len_words: exact.value.words,
        },
        value: exact.value.id,
        elements,
        binding,
        source,
        destination,
        alias: InPlaceAliasAuthority {
            id: InPlaceAliasId(
                u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
            ),
            requirement: InPlaceAliasRequirement::Required,
            discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
        },
    })
}

pub(super) fn effect(
    contract: &WitnessFeedContract,
    source: &MultiplicityFeedValueBinding,
    descriptor_value: ValueVersion,
    descriptor_binding: EffectBindingId,
    luts: &[MultiplicityFeedLutBinding],
    destinations: &[MultiplicityFeedTransition],
) -> Result<EffectContract, InvocationShapeError> {
    validate_parts(contract, source, luts, destinations)?;
    let descriptor_elements = ElementRange::new(0, contract.effect_geometry().descriptor_words)
        .ok_or(InvocationShapeError::InvalidStructuredAbi)?;
    let mut accesses = Vec::with_capacity(2 + luts.len() + destinations.len());
    accesses.push(EffectAccess::Read {
        source: bound(source.binding, source.version, source.elements),
    });
    accesses.push(EffectAccess::Read {
        source: bound(descriptor_binding, descriptor_value, descriptor_elements),
    });
    accesses.extend(luts.iter().map(|lut| EffectAccess::Read {
        source: bound(lut.input.binding, lut.input.version, lut.input.elements),
    }));
    accesses.extend(destinations.iter().map(|destination| EffectAccess::Atomic {
        source: bound(
            destination.binding,
            destination.source,
            destination.elements,
        ),
        destination: bound(
            destination.binding,
            destination.destination,
            destination.elements,
        ),
        operation: AtomicOperation::AddU32,
        in_place: destination.alias,
    }));
    EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidAdapterEffect)
}

pub(super) fn invocation(
    contract: &WitnessFeedContract,
    source: &MultiplicityFeedValueBinding,
    descriptor_value: ValueVersion,
    descriptor_binding: EffectBindingId,
    luts: &[MultiplicityFeedLutBinding],
    destinations: &[MultiplicityFeedTransition],
) -> Result<AotInvocation, InvocationShapeError> {
    invocation_using_abi(
        contract,
        source,
        descriptor_value,
        descriptor_binding,
        luts,
        destinations,
        contract.abi().arguments(),
    )
}

fn invocation_using_abi(
    contract: &WitnessFeedContract,
    source: &MultiplicityFeedValueBinding,
    descriptor_value: ValueVersion,
    descriptor_binding: EffectBindingId,
    luts: &[MultiplicityFeedLutBinding],
    destinations: &[MultiplicityFeedTransition],
    abi: &[WitnessFeedAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    validate_parts(contract, source, luts, destinations)?;
    if abi != contract.abi().arguments()
        || abi
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let geometry = contract.effect_geometry();
    let mut arguments = Vec::with_capacity(abi.len() - 1);
    for descriptor in abi {
        use WitnessFeedAbiAccess as Access;
        use WitnessFeedAbiArgumentKind as Kind;
        let value = match (
            descriptor.ordinal,
            descriptor.name,
            descriptor.kind,
            descriptor.access,
        ) {
            (0, "sub_words_dev", Kind::DeviceConstPointerU32, Access::ReadSource) => {
                AotArgumentValue::DevicePointer(Some(source.binding))
            }
            (1, "column_length", Kind::U32, Access::RowCount) => {
                AotArgumentValue::U32(geometry.row_domain.row_count)
            }
            (2, "descs_dev", Kind::DeviceConstPointerU32, Access::ReadDescriptors) => {
                AotArgumentValue::DeviceFixedU32 {
                    value: descriptor_value,
                    binding: descriptor_binding,
                }
            }
            (3, "n_descs", Kind::U32, Access::DescriptorCount) => AotArgumentValue::U32(
                u32::try_from(contract.requirements().descriptor_count)
                    .map_err(|_| InvocationShapeError::SizeOverflow)?,
            ),
            (4, "luts_dev", Kind::DeviceConstPointerTableU32, Access::ReadLuts) => {
                AotArgumentValue::DevicePointerTable(if luts.is_empty() {
                    vec![None]
                } else {
                    luts.iter().map(|lut| Some(lut.input.binding)).collect()
                })
            }
            (5, "counts_dev", Kind::DeviceMutPointerTableU32, Access::AtomicDestinations) => {
                AotArgumentValue::DevicePointerTable(
                    destinations
                        .iter()
                        .map(|destination| Some(destination.binding))
                        .collect(),
                )
            }
            (6, "stream", Kind::CudaStream, Access::OrderedExecutionStream) => continue,
            _ => return Err(InvocationShapeError::InvalidStructuredAbi),
        };
        arguments.push(AotArgumentBinding {
            ordinal: descriptor.ordinal,
            value,
        });
    }
    if arguments.len() != 6
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
    descriptor_value: ValueVersion,
    descriptor_binding: EffectBindingId,
    zero_luts: bool,
) -> Result<(), InvocationShapeError> {
    if !effect.registered_fixed_source_reads().is_empty() || invocation.arguments.len() != 6 {
        return Err(InvocationShapeError::InvalidAdapterEffect);
    }
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
            AotArgumentValue::U32(_) | AotArgumentValue::Usize(_) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => insert(*binding)?,
            AotArgumentValue::DevicePointer(None) => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
            AotArgumentValue::DevicePointerTable(entries) => {
                if entries.is_empty() {
                    return Err(InvocationShapeError::InvalidAdapterEffect);
                }
                for &binding in entries.iter().flatten() {
                    insert(binding)?;
                }
            }
            AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(_) => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
            AotArgumentValue::DeviceMixedFixedSourcePointerTable(_) => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
            AotArgumentValue::DeviceFixedU32 { value, binding }
                if (*value, *binding) == (descriptor_value, descriptor_binding) =>
            {
                insert(*binding)?;
            }
            AotArgumentValue::DeviceFixedU32 { .. } => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
        }
    }
    let lut_argument = &invocation.arguments[4].value;
    let dummy_is_exact = matches!(
        lut_argument,
        AotArgumentValue::DevicePointerTable(entries)
            if (zero_luts && entries == &[None])
                || (!zero_luts && entries.iter().all(Option::is_some))
    );
    if actual == expected && dummy_is_exact {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidAdapterEffect)
    }
}

pub(super) fn validate_geometry(
    lowered: &LoweredMultiplicityFeed,
) -> Result<(), InvocationShapeError> {
    validate_parts(
        &lowered.contract,
        &lowered.source,
        &lowered.luts,
        &lowered.destinations,
    )?;
    if lowered.descriptor_binding != EffectBindingId(1)
        || lowered.relocations.descriptor_workspace.len_words
            != lowered.contract.requirements().descriptor_words
        || lowered.relocations.lut_pointers.len_words
            != lowered.contract.requirements().lut_pointer_words
        || lowered.relocations.multiplicity_pointers.len_words
            != lowered.contract.requirements().multiplicity_pointer_words
        || (matches!(lowered.owner, MultiplicityFeedOwner::PublicMemorySeed)
            && !lowered.luts.is_empty())
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    Ok(())
}

fn validate_parts(
    contract: &WitnessFeedContract,
    source: &MultiplicityFeedValueBinding,
    luts: &[MultiplicityFeedLutBinding],
    destinations: &[MultiplicityFeedTransition],
) -> Result<(), InvocationShapeError> {
    let geometry = contract.effect_geometry();
    if source.binding != EffectBindingId(0)
        || source.elements.start != geometry.source.read_start_words
        || source.elements.len() != geometry.source.read_len_words
        || luts.len() != geometry.lut_reads.len()
        || destinations.len() != geometry.destinations.len()
        || luts.iter().zip(&geometry.lut_reads).any(|(lut, read)| {
            lut.input.elements.start != read.read_start_words
                || lut.input.elements.len() != read.read_len_words
                || lut.content_identity != read.content_identity
        })
        || destinations
            .iter()
            .zip(&geometry.destinations)
            .enumerate()
            .any(|(ordinal, (destination, expected))| {
                destination.ordinal as usize != ordinal
                    || destination.elements.start != expected.atomic_start_words
                    || destination.elements.len() != expected.atomic_len_words
                    || destination.alias.id.0 as usize != ordinal
                    || destination.alias.requirement != InPlaceAliasRequirement::Required
                    || destination.alias.discipline != InPlaceDiscipline::ElementWiseReadBeforeWrite
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
    lowered: &LoweredMultiplicityFeed,
    abi: &[WitnessFeedAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    invocation_using_abi(
        &lowered.contract,
        &lowered.source,
        lowered.descriptor_value,
        lowered.descriptor_binding,
        &lowered.luts,
        &lowered.destinations,
        abi,
    )
}
