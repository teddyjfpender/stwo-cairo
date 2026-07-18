//! Exact source, geometry, ABI and effect validation.

use std::collections::BTreeSet;

use super::{
    range, InvocationShapeError, LoweredFixedTableMaterialization, LoweredFixedTableSource,
    LoweredFixedTableStage,
};
use crate::compiled_proof::{
    AotArgumentValue, AotInvocation, EffectAccess, EffectContract, FixedSourcePointerEntry,
};

pub(super) fn validate_internal(
    stage: &LoweredFixedTableStage,
) -> Result<(), InvocationShapeError> {
    if stage.tables.is_empty()
        || stage
            .tables
            .windows(2)
            .any(|pair| pair[0].component >= pair[1].component)
    {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }
    for table in &stage.tables {
        validate_table(table)?;
    }
    Ok(())
}

pub(super) fn validate_table(
    table: &LoweredFixedTableMaterialization,
) -> Result<(), InvocationShapeError> {
    table
        .contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
    if !table
        .effect
        .has_valid_identity()
        .map_err(|_| InvocationShapeError::InvalidFixedTableBinding)?
    {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }
    let requirements = table.contract.requirements();
    let rows = requirements.row_count;
    let arguments = &table.invocation.arguments;
    let expected_sources = table
        .sources
        .iter()
        .map(|source| match source {
            LoweredFixedTableSource::Arena { binding, .. } => {
                FixedSourcePointerEntry::EffectBinding(*binding)
            }
            LoweredFixedTableSource::Registered { read, .. } => {
                FixedSourcePointerEntry::Registered(read.clone())
            }
        })
        .collect::<Vec<_>>();
    let expected_source_argument = source_argument(expected_sources)?;
    let expected_registered = table
        .sources
        .iter()
        .filter_map(|source| match source {
            LoweredFixedTableSource::Registered { read, .. } => Some(read.clone()),
            LoweredFixedTableSource::Arena { .. } => None,
        })
        .collect::<Vec<_>>();
    let multiplicity_words = rows
        .checked_mul(requirements.multiplicity_column_count)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let lookup_words = rows
        .checked_mul(requirements.lookup_output_count)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if arguments.len() != 9
        || arguments
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
        || &arguments[0].value != &expected_source_argument
        || !matches!(
            &arguments[1].value,
            AotArgumentValue::DevicePointerTable(entries)
                if entries.len() == requirements.multiplicity_column_count
                    && entries.iter().all(Option::is_some)
        )
        || !matches!(&arguments[2].value, AotArgumentValue::DeviceFixedU32 { .. })
        || !matches!(
            &arguments[3].value,
            AotArgumentValue::DevicePointerTable(entries)
                if entries.len() == requirements.trace_output_count
                    && entries.iter().all(Option::is_some)
        )
        || &arguments[4].value
            != &AotArgumentValue::U32(
                u32::try_from(requirements.trace_output_count)
                    .map_err(|_| InvocationShapeError::SizeOverflow)?,
            )
        || !matches!(&arguments[5].value, AotArgumentValue::DeviceFixedU32 { .. })
        || !matches!(
            &arguments[6].value,
            AotArgumentValue::DevicePointerTable(entries)
                if entries.len() == requirements.lookup_output_count
                    && entries.iter().all(Option::is_some)
        )
        || &arguments[7].value
            != &AotArgumentValue::U32(
                u32::try_from(requirements.lookup_output_count)
                    .map_err(|_| InvocationShapeError::SizeOverflow)?,
            )
        || &arguments[8].value
            != &AotArgumentValue::U32(
                u32::try_from(rows).map_err(|_| InvocationShapeError::SizeOverflow)?,
            )
        || table.sources.len() != table.contract.config().source_column_count
        || table.multiplicity.len_words != multiplicity_words
        || table.trace_outputs.len() != requirements.trace_output_count
        || table
            .trace_outputs
            .iter()
            .any(|output| output.len_words != rows)
        || table.lookup_output.len_words != lookup_words
        || table.workspace.trace_outputs
            != table
                .trace_outputs
                .iter()
                .map(|binding| binding.physical)
                .collect::<Vec<_>>()
        || table.workspace.lookup_output != table.lookup_output.physical
        || table.workspace.source_pointers.is_some() != !table.sources.is_empty()
        || table.effect.registered_fixed_source_reads() != expected_registered
        || !invocation_matches_effect(&table.invocation, &table.effect)?
    {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }
    for source in &table.sources {
        match source {
            LoweredFixedTableSource::Arena {
                arena,
                version,
                elements,
                binding,
                ..
            } => {
                if arena.len_words != rows
                    || *elements != range(0, rows)?
                    || !table.effect.accesses().iter().any(|access| {
                        matches!(
                            access,
                            EffectAccess::Read { source }
                                if source.binding == *binding
                                    && source.value.version == *version
                                    && source.value.elements == *elements
                        )
                    })
                {
                    return Err(InvocationShapeError::InvalidFixedTableBinding);
                }
            }
            LoweredFixedTableSource::Registered { read, .. } => {
                if read.elements() != range(0, rows)? {
                    return Err(InvocationShapeError::InvalidFixedTableBinding);
                }
            }
        }
    }
    Ok(())
}

pub(super) fn source_argument(
    entries: Vec<FixedSourcePointerEntry>,
) -> Result<AotArgumentValue, InvocationShapeError> {
    let ordinary = entries
        .iter()
        .all(|source| matches!(source, FixedSourcePointerEntry::EffectBinding(_)));
    let registered = entries
        .iter()
        .all(|source| matches!(source, FixedSourcePointerEntry::Registered(_)));
    match (ordinary, registered) {
        _ if entries.is_empty() => Ok(AotArgumentValue::DevicePointer(None)),
        (true, false) => Ok(AotArgumentValue::DevicePointerTable(
            entries
                .into_iter()
                .map(|source| match source {
                    FixedSourcePointerEntry::EffectBinding(binding) => Some(binding),
                    FixedSourcePointerEntry::Registered(_) => unreachable!(),
                })
                .collect(),
        )),
        (false, true) => Ok(AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(
            entries
                .into_iter()
                .map(|source| match source {
                    FixedSourcePointerEntry::Registered(read) => read,
                    FixedSourcePointerEntry::EffectBinding(_) => unreachable!(),
                })
                .collect(),
        )),
        (false, false) => Ok(AotArgumentValue::DeviceMixedFixedSourcePointerTable(
            entries,
        )),
        (true, true) => Err(InvocationShapeError::InvalidFixedTableBinding),
    }
}

fn invocation_matches_effect(
    invocation: &AotInvocation,
    effect: &EffectContract,
) -> Result<bool, InvocationShapeError> {
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    let mut registered = Vec::new();
    for argument in &invocation.arguments {
        let mut insert = |binding| actual.insert(binding);
        let unique = match &argument.value {
            AotArgumentValue::U32(_)
            | AotArgumentValue::Usize(_)
            | AotArgumentValue::DevicePointer(None) => true,
            AotArgumentValue::DevicePointer(Some(binding)) => insert(*binding),
            AotArgumentValue::DevicePointerTable(entries) => {
                entries.iter().flatten().copied().all(&mut insert)
            }
            AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(reads) => {
                registered.extend(reads.iter().cloned());
                true
            }
            AotArgumentValue::DeviceMixedFixedSourcePointerTable(entries) => {
                entries.iter().all(|entry| match entry {
                    FixedSourcePointerEntry::EffectBinding(binding) => insert(*binding),
                    FixedSourcePointerEntry::Registered(read) => {
                        registered.push(read.clone());
                        true
                    }
                })
            }
            AotArgumentValue::DeviceFixedU32 { binding, .. } => insert(*binding),
            AotArgumentValue::DevicePointerTableValue(_)
            | AotArgumentValue::DeviceNestedPointerTableValue { .. }
            | AotArgumentValue::DeviceRecordPointerGraphValue { .. }
            | AotArgumentValue::DevicePointerRangeSetValue { .. }
            | AotArgumentValue::HostFixedU32(_) => false,
        };
        if !unique {
            return Ok(false);
        }
    }
    Ok(actual == expected && registered == effect.registered_fixed_source_reads())
}
