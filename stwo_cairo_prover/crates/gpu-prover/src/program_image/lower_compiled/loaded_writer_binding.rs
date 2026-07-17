//! Exact arena-slice admission for one prepared recorded witness writer.

use stwo_backend_cuda::{ArenaSlice, ArenaSlotId, DeviceArena, PreparedWitnessGraph};

use super::{
    BaseProducerCatalog, InvocationShapeError, InvocationTarget, RecordedWitnessInvocationShape,
    SourceArgument,
};
use crate::arena_plan::{BufferPurpose, PlannedWitnessComponent};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlannedSliceBinding {
    slot: ArenaSlotId,
    words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WriterBindingPlan {
    row_count: usize,
    input_columns: Vec<PlannedSliceBinding>,
    output_columns: Vec<PlannedSliceBinding>,
    multiplicity_columns: Vec<PlannedSliceBinding>,
    lookup_words: PlannedSliceBinding,
    sub_words: PlannedSliceBinding,
    descriptors: Vec<PlannedSliceBinding>,
    multiplicity_dummy: Option<PlannedSliceBinding>,
}

impl WriterBindingPlan {
    pub(super) fn from_planned(
        planned: &PlannedWitnessComponent,
    ) -> Result<Self, InvocationShapeError> {
        if planned.blake_g_contract != crate::arena_plan::BlakeGWitnessContract::Recorded {
            return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
        }
        let requirements = &planned.requirements;
        let slots = &planned.slots;
        if requirements.row_count == 0 {
            return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
        }
        let multiplicity_dummy = match (
            slots.multiplicity_dummy,
            requirements.multiplicity_dummy_words,
        ) {
            (Some(slot), Some(words)) => Some(PlannedSliceBinding { slot, words }),
            (None, None) => None,
            _ => return Err(InvocationShapeError::LoadedAotAuthorityMismatch),
        };
        Ok(Self {
            row_count: requirements.row_count,
            input_columns: planned_slices(&slots.input_columns, &requirements.input_column_words)?,
            output_columns: planned_slices(
                &slots.output_columns,
                &requirements.output_column_words,
            )?,
            multiplicity_columns: planned_slices(
                &slots.multiplicity_columns,
                &requirements.multiplicity_column_words,
            )?,
            lookup_words: PlannedSliceBinding {
                slot: slots.lookup_words,
                words: requirements.lookup_words,
            },
            sub_words: PlannedSliceBinding {
                slot: slots.sub_words,
                words: requirements.sub_words,
            },
            descriptors: vec![
                PlannedSliceBinding {
                    slot: slots.input_pointers,
                    words: requirements.input_pointer_words,
                },
                PlannedSliceBinding {
                    slot: slots.execution_table_pointers,
                    words: requirements.execution_table_pointer_words,
                },
                PlannedSliceBinding {
                    slot: slots.execution_table_strides,
                    words: requirements.execution_table_stride_words,
                },
                PlannedSliceBinding {
                    slot: slots.output_pointers,
                    words: requirements.output_pointer_words,
                },
                PlannedSliceBinding {
                    slot: slots.multiplicity_pointers,
                    words: requirements.multiplicity_pointer_words,
                },
            ],
            multiplicity_dummy,
        })
    }

    pub(super) fn from_invocation(
        invocation: &RecordedWitnessInvocationShape,
        catalog: &BaseProducerCatalog,
    ) -> Result<Self, InvocationShapeError> {
        let [inputs, table_pointers, table_strides, outputs, multiplicities, lookup, sub, rows] =
            invocation.source_arguments.as_slice()
        else {
            return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
        };
        let (input_descriptor, input_columns) = pointer_table_plan(inputs, 0, catalog)?;
        let (table_pointer_descriptor, table_targets) =
            pointer_table_plan(table_pointers, 1, catalog)?;
        let (table_stride_descriptor, table_extents) =
            scalar_array_plan(table_strides, 2, catalog)?;
        validate_execution_table_targets(table_pointers, &table_targets, &table_extents, catalog)?;
        let (output_descriptor, output_columns) = pointer_table_plan(outputs, 3, catalog)?;
        let (multiplicity_descriptor, multiplicity_targets) =
            pointer_table_plan(multiplicities, 4, catalog)?;
        let [multiplicity_dummy] = multiplicity_targets.as_slice() else {
            return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
        };
        let lookup_words = direct_pointer_plan(lookup, 5, catalog)?;
        let sub_words = direct_pointer_plan(sub, 6, catalog)?;
        let SourceArgument::U32 { ordinal: 7, value } = rows else {
            return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
        };
        let row_count = usize::try_from(*value)
            .ok()
            .filter(|&rows| rows != 0)
            .ok_or(InvocationShapeError::LoadedAotAuthorityMismatch)?;
        Ok(Self {
            row_count,
            input_columns,
            output_columns,
            multiplicity_columns: Vec::new(),
            lookup_words,
            sub_words,
            descriptors: vec![
                input_descriptor,
                table_pointer_descriptor,
                table_stride_descriptor,
                output_descriptor,
                multiplicity_descriptor,
            ],
            multiplicity_dummy: Some(*multiplicity_dummy),
        })
    }
}

fn planned_slices(
    slots: &[ArenaSlotId],
    words: &[usize],
) -> Result<Vec<PlannedSliceBinding>, InvocationShapeError> {
    if slots.len() != words.len() || words.contains(&0) {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    Ok(slots
        .iter()
        .copied()
        .zip(words.iter().copied())
        .map(|(slot, words)| PlannedSliceBinding { slot, words })
        .collect())
}

fn pointer_table_plan(
    argument: &SourceArgument,
    expected_ordinal: u8,
    catalog: &BaseProducerCatalog,
) -> Result<(PlannedSliceBinding, Vec<PlannedSliceBinding>), InvocationShapeError> {
    let SourceArgument::PointerTable {
        ordinal,
        descriptor,
        entries,
        ..
    } = argument
    else {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    };
    if *ordinal != expected_ordinal || entries.is_empty() {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    let descriptor = catalog_range_plan(descriptor, catalog)?;
    let expected_words = entries
        .len()
        .checked_mul(super::POINTER_WORDS)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if descriptor.words != expected_words {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    let targets = entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let start = index
                .checked_mul(super::POINTER_WORDS)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            let end = start
                .checked_add(super::POINTER_WORDS)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            if entry.entry
                != u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?
                || entry.descriptor_words != (start..end)
            {
                return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
            }
            target_plan(&entry.target, catalog)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((descriptor, targets))
}

fn scalar_array_plan(
    argument: &SourceArgument,
    expected_ordinal: u8,
    catalog: &BaseProducerCatalog,
) -> Result<(PlannedSliceBinding, Vec<usize>), InvocationShapeError> {
    let SourceArgument::ScalarArray {
        ordinal,
        value,
        entries,
    } = argument
    else {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    };
    let value = catalog_range_plan(value, catalog)?;
    if *ordinal != expected_ordinal
        || value.words != entries.len()
        || entries
            .iter()
            .enumerate()
            .any(|(index, entry)| entry.index != index as u32)
    {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    let entries = entries
        .iter()
        .map(|entry| usize::try_from(entry.value).map_err(|_| InvocationShapeError::SizeOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((value, entries))
}

fn validate_execution_table_targets(
    argument: &SourceArgument,
    targets: &[PlannedSliceBinding],
    extents: &[usize],
    catalog: &BaseProducerCatalog,
) -> Result<(), InvocationShapeError> {
    let SourceArgument::PointerTable { entries, .. } = argument else {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    };
    let [address_words, big_words, small_words] = extents else {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    };
    if entries.len() != super::EXECUTION_TABLE_POINTERS || targets.len() != entries.len() {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    for (index, (entry, target)) in entries.iter().zip(targets).enumerate() {
        let value = catalog.value(entry.target.value)?;
        let (purpose, ordinal, words) = if index == 0 {
            (
                BufferPurpose::ExecutionTableRawAddressToId,
                0,
                *address_words,
            )
        } else if index <= super::EXECUTION_TABLE_BIG_LIMBS {
            (BufferPurpose::ExecutionTableBigLimb, index - 1, *big_words)
        } else {
            (
                BufferPurpose::ExecutionTableSmallLimb,
                index - 1 - super::EXECUTION_TABLE_BIG_LIMBS,
                *small_words,
            )
        };
        if value.purpose != purpose
            || value.ordinal
                != u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?
            || target.words != words
        {
            return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
        }
    }
    Ok(())
}

fn direct_pointer_plan(
    argument: &SourceArgument,
    expected_ordinal: u8,
    catalog: &BaseProducerCatalog,
) -> Result<PlannedSliceBinding, InvocationShapeError> {
    let SourceArgument::DirectPointer { ordinal, target } = argument else {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    };
    if *ordinal != expected_ordinal {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    target_plan(target, catalog)
}

fn catalog_range_plan(
    range: &crate::program_image::ArenaCatalogRange,
    catalog: &BaseProducerCatalog,
) -> Result<PlannedSliceBinding, InvocationShapeError> {
    let value = catalog.value(range.value)?;
    if range.value_words.start != 0
        || range.value_words.end == 0
        || range.value_words.end != value.words
    {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    Ok(PlannedSliceBinding {
        slot: value.physical,
        words: range.value_words.end,
    })
}

fn target_plan(
    target: &InvocationTarget,
    catalog: &BaseProducerCatalog,
) -> Result<PlannedSliceBinding, InvocationShapeError> {
    let value = catalog.value(target.value)?;
    // Table columns are padded physical allocations. Their exact logical
    // prefix is sealed independently against the published stride scalars.
    if target.elements.start != 0
        || target.elements.end > value.words
        || (target.elements.end == 0 && target.access != super::InvocationAccess::Inactive)
    {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    Ok(PlannedSliceBinding {
        slot: value.physical,
        words: target.elements.end,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SliceBindingFields {
    pub(super) slot: ArenaSlotId,
    pub(super) offset_words: usize,
    pub(super) words: usize,
    pub(super) pointer_token: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WriterBindingFields {
    pub(super) input_columns: Vec<SliceBindingFields>,
    pub(super) output_columns: Vec<SliceBindingFields>,
    pub(super) multiplicity_columns: Vec<SliceBindingFields>,
    pub(super) lookup_words: SliceBindingFields,
    pub(super) sub_words: SliceBindingFields,
    pub(super) descriptors: Vec<SliceBindingFields>,
    pub(super) multiplicity_dummy: Option<SliceBindingFields>,
}

impl WriterBindingFields {
    pub(super) fn from_plan(
        arena: &DeviceArena,
        plan: &WriterBindingPlan,
    ) -> Result<Self, InvocationShapeError> {
        Ok(Self {
            input_columns: bind_planned_slices(arena, &plan.input_columns)?,
            output_columns: bind_planned_slices(arena, &plan.output_columns)?,
            multiplicity_columns: bind_planned_slices(arena, &plan.multiplicity_columns)?,
            lookup_words: bind_planned_slice(arena, plan.lookup_words)?,
            sub_words: bind_planned_slice(arena, plan.sub_words)?,
            descriptors: bind_planned_slices(arena, &plan.descriptors)?,
            multiplicity_dummy: plan
                .multiplicity_dummy
                .map(|binding| bind_planned_slice(arena, binding))
                .transpose()?,
        })
    }

    pub(super) fn from_writer(
        arena: &DeviceArena,
        writer: &PreparedWitnessGraph<'_>,
    ) -> Result<Self, InvocationShapeError> {
        Ok(Self {
            input_columns: bind_actual_slices(arena, writer.input_columns())?,
            output_columns: bind_actual_slices(arena, writer.output_columns())?,
            multiplicity_columns: bind_actual_slices(arena, writer.multiplicity_columns())?,
            lookup_words: bind_actual_slice(arena, writer.lookup_words())?,
            sub_words: bind_actual_slice(arena, writer.sub_words())?,
            descriptors: bind_actual_slices(arena, &writer.descriptor_slices())?,
            multiplicity_dummy: writer
                .multiplicity_dummy()
                .map(|slice| bind_actual_slice(arena, slice))
                .transpose()?,
        })
    }
}

fn bind_planned_slices(
    arena: &DeviceArena,
    bindings: &[PlannedSliceBinding],
) -> Result<Vec<SliceBindingFields>, InvocationShapeError> {
    bindings
        .iter()
        .copied()
        .map(|binding| bind_planned_slice(arena, binding))
        .collect()
}

fn bind_planned_slice(
    arena: &DeviceArena,
    binding: PlannedSliceBinding,
) -> Result<SliceBindingFields, InvocationShapeError> {
    let slice = arena
        .bind(binding.slot)
        .map_err(|_| InvocationShapeError::LoadedAotAuthorityMismatch)?;
    if slice.len_words() < binding.words {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    let spec = arena
        .layout()
        .slot(binding.slot)
        .ok_or(InvocationShapeError::LoadedAotAuthorityMismatch)?;
    Ok(SliceBindingFields {
        slot: binding.slot,
        offset_words: spec.offset_words,
        words: binding.words,
        pointer_token: pointer_token(slice)?,
    })
}

fn bind_actual_slices(
    arena: &DeviceArena,
    slices: &[ArenaSlice],
) -> Result<Vec<SliceBindingFields>, InvocationShapeError> {
    slices
        .iter()
        .copied()
        .map(|slice| bind_actual_slice(arena, slice))
        .collect()
}

fn bind_actual_slice(
    arena: &DeviceArena,
    slice: ArenaSlice,
) -> Result<SliceBindingFields, InvocationShapeError> {
    if !slice.belongs_to(arena.context()) {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    let spec = arena
        .layout()
        .slot(slice.id())
        .ok_or(InvocationShapeError::LoadedAotAuthorityMismatch)?;
    let expected_pointer = arena.base_ptr().as_ptr().wrapping_add(spec.offset_words);
    if slice.as_u32_ptr() != expected_pointer {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    Ok(SliceBindingFields {
        slot: slice.id(),
        offset_words: spec.offset_words,
        words: slice.len_words(),
        pointer_token: pointer_token(slice)?,
    })
}

fn pointer_token(slice: ArenaSlice) -> Result<u64, InvocationShapeError> {
    u64::try_from(slice.as_u32_ptr() as usize).map_err(|_| InvocationShapeError::SizeOverflow)
}

pub(super) fn validate_writer_binding(
    expected: &WriterBindingFields,
    actual: &WriterBindingFields,
) -> Result<(), InvocationShapeError> {
    if actual != expected {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    Ok(())
}
