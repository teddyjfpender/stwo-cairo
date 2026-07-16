//! Schedule-driven lowering at the witness-to-Base-interpolation boundary.

use std::ops::Range;

use stwo_backend_cuda::{
    InterpolationBatchAuthority, InterpolationEffectAbi, InterpolationLaunchMode,
    InterpolationPrimitiveAbi,
};

use super::*;
use crate::compiled_proof::{
    BoundValueRange, EffectAccess, EffectBindingId, EffectContract, ElementRange,
    InPlaceAliasAuthority, InPlaceAliasId, InPlaceAliasRequirement, InPlaceDiscipline, ValueRange,
};
use crate::resident_runtime::producer_schedule::{
    BaseInterpolationBatch, BaseProducerSchedule, ScheduledGlobalValue, ScheduledValue,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StaticInterpolationColumnBinding {
    pub(super) entry: u32,
    pub(super) input_descriptor_words: Range<usize>,
    pub(super) output_descriptor_words: Range<usize>,
    pub(super) evaluations: ArenaCatalogValueId,
    pub(super) coefficients: ArenaCatalogValueId,
    pub(super) source: EffectBindingId,
    pub(super) destination: EffectBindingId,
    pub(super) exact_in_place: bool,
}

/// Honest receipt for the composite prepared wrapper. It is not mislabeled as
/// one `AotInvocation`: stage-wise mode includes D2D copies, and either raw C
/// entry may submit several kernel launches.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StaticInterpolationInvocation {
    pub(super) input_pointers: ArenaCatalogRange,
    pub(super) output_pointers: ArenaCatalogRange,
    pub(super) columns: Vec<StaticInterpolationColumnBinding>,
    pub(super) inverse_twiddles: EffectBindingId,
    pub(super) log_size: u32,
    pub(super) column_count: u32,
    pub(super) twiddle_words: u32,
    pub(super) evaluation_domain_size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredBaseInterpolationBatch {
    pub(super) batch: u32,
    pub(super) authority: InterpolationBatchAuthority,
    pub(super) invocation: StaticInterpolationInvocation,
    pub(super) effect: EffectContract,
}

pub(super) fn invocation_catalog_order(source: &[SourceArgument]) -> Vec<ArenaCatalogValueId> {
    source
        .iter()
        .flat_map(|argument| match argument {
            SourceArgument::PointerTable { entries, .. } => entries
                .iter()
                .filter(|entry| entry.target.access != InvocationAccess::Inactive)
                .map(|entry| entry.target.value)
                .collect::<Vec<_>>(),
            SourceArgument::DirectPointer { target, .. }
                if target.access != InvocationAccess::Inactive =>
            {
                vec![target.value]
            }
            SourceArgument::ScalarArray { .. }
            | SourceArgument::U32 { .. }
            | SourceArgument::DirectPointer { .. } => Vec::new(),
        })
        .collect()
}

/// Extend semantic-version encounter order from the shared producer schedule,
/// never from arena catalog order. Descriptor values remain in the checked
/// composite invocation receipt; only dereferenced values receive versions.
pub(super) fn append_interpolation_catalog_order(
    image: &ArenaProgramInventory,
    schedule: &BaseProducerSchedule,
    ordered_values: &mut Vec<ArenaCatalogValueId>,
) -> Result<(), InvocationShapeError> {
    let interpolation = schedule
        .interpolation()
        .ok_or(InvocationShapeError::FrontierDidNotAdvance)?;
    ordered_values.push(catalog_global(image, &interpolation.inverse_twiddles)?.id);
    for batch in &interpolation.batches {
        validate_batch_authority(interpolation.mode, batch)?;
        catalog_global(image, &batch.input_pointers)?;
        catalog_global(image, &batch.output_pointers)?;
        for column in &batch.columns {
            ordered_values.push(catalog_scheduled(image, &column.evaluations)?.id);
            ordered_values.push(catalog_scheduled(image, &column.coefficients)?.id);
        }
    }
    Ok(())
}

pub(super) fn lower_base_interpolation(
    image: &ArenaProgramInventory,
    schedule: &BaseProducerSchedule,
    values: &adapter::SemanticValueMap,
) -> Result<Vec<LoweredBaseInterpolationBatch>, InvocationShapeError> {
    let interpolation = schedule
        .interpolation()
        .ok_or(InvocationShapeError::FrontierDidNotAdvance)?;
    let mut lowered = Vec::with_capacity(interpolation.batches.len());
    for (batch_index, batch) in interpolation.batches.iter().enumerate() {
        validate_batch_authority(interpolation.mode, batch)?;
        let input_pointers = whole_catalog_range(catalog_global(image, &batch.input_pointers)?)?;
        let output_pointers = whole_catalog_range(catalog_global(image, &batch.output_pointers)?)?;
        let pointer_words = batch
            .columns
            .len()
            .checked_mul(POINTER_WORDS)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        if input_pointers.value_words != (0..pointer_words)
            || output_pointers.value_words != (0..pointer_words)
        {
            return Err(InvocationShapeError::InvalidBaseInterpolationBinding);
        }

        let mut next_binding = 0u32;
        let mut next_alias = 0u32;
        let mut accesses = Vec::with_capacity(batch.columns.len() + 1);
        let mut columns = Vec::with_capacity(batch.columns.len());
        for (column_index, column) in batch.columns.iter().enumerate() {
            let evaluations = catalog_scheduled(image, &column.evaluations)?;
            let coefficients = catalog_scheduled(image, &column.coefficients)?;
            let source = EffectBindingId(next_binding);
            next_binding = next_binding
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            let destination = EffectBindingId(next_binding);
            next_binding = next_binding
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            let elements = ElementRange::new(0, batch.authority.value_words())
                .ok_or(InvocationShapeError::InvalidBaseInterpolationBinding)?;
            let source_range = BoundValueRange {
                binding: source,
                value: ValueRange {
                    version: values.version(evaluations.id)?,
                    elements,
                },
            };
            let destination_range = BoundValueRange {
                binding: destination,
                value: ValueRange {
                    version: values.version(coefficients.id)?,
                    elements,
                },
            };
            let exact_in_place = column.evaluations.physical == column.coefficients.physical;
            let in_place = if exact_in_place {
                let alias = InPlaceAliasAuthority {
                    id: InPlaceAliasId(next_alias),
                    requirement: InPlaceAliasRequirement::Required,
                    discipline: InPlaceDiscipline::BlockBarrierPhases,
                };
                next_alias = next_alias
                    .checked_add(1)
                    .ok_or(InvocationShapeError::SizeOverflow)?;
                Some(alias)
            } else {
                None
            };
            accesses.push(EffectAccess::ReadWrite {
                source: source_range,
                destination: destination_range,
                in_place,
            });
            let descriptor_start = column_index
                .checked_mul(POINTER_WORDS)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            let descriptor_end = descriptor_start
                .checked_add(POINTER_WORDS)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            columns.push(StaticInterpolationColumnBinding {
                entry: u32::try_from(column_index)
                    .map_err(|_| InvocationShapeError::SizeOverflow)?,
                input_descriptor_words: descriptor_start..descriptor_end,
                output_descriptor_words: descriptor_start..descriptor_end,
                evaluations: evaluations.id,
                coefficients: coefficients.id,
                source,
                destination,
                exact_in_place,
            });
        }

        let twiddles = catalog_global(image, &interpolation.inverse_twiddles)?;
        let evaluation_domain_size = usize::try_from(batch.authority.evaluation_domain_size())
            .map_err(|_| InvocationShapeError::SizeOverflow)?;
        let suffix_start = interpolation
            .inverse_twiddles
            .words
            .checked_sub(evaluation_domain_size)
            .ok_or(InvocationShapeError::InvalidBaseInterpolationBinding)?;
        let inverse_twiddles = EffectBindingId(next_binding);
        accesses.push(EffectAccess::Read {
            source: BoundValueRange {
                binding: inverse_twiddles,
                value: ValueRange {
                    version: values.version(twiddles.id)?,
                    elements: ElementRange::new(suffix_start, interpolation.inverse_twiddles.words)
                        .ok_or(InvocationShapeError::InvalidBaseInterpolationBinding)?,
                },
            },
        });
        let effect = EffectContract::new(accesses, Vec::new())
            .map_err(|_| InvocationShapeError::InvalidAdapterEffect)?;
        lowered.push(LoweredBaseInterpolationBatch {
            batch: u32::try_from(batch_index).map_err(|_| InvocationShapeError::SizeOverflow)?,
            authority: batch.authority.clone(),
            invocation: StaticInterpolationInvocation {
                input_pointers,
                output_pointers,
                columns,
                inverse_twiddles,
                log_size: batch.log_size,
                column_count: u32::try_from(batch.columns.len())
                    .map_err(|_| InvocationShapeError::SizeOverflow)?,
                twiddle_words: u32::try_from(interpolation.inverse_twiddles.words)
                    .map_err(|_| InvocationShapeError::SizeOverflow)?,
                evaluation_domain_size: batch.authority.evaluation_domain_size(),
            },
            effect,
        });
    }
    Ok(lowered)
}

fn validate_batch_authority(
    mode: InterpolationLaunchMode,
    batch: &BaseInterpolationBatch,
) -> Result<(), InvocationShapeError> {
    let expected = InterpolationBatchAuthority::compile(mode, batch.log_size, batch.columns.len())
        .map_err(|_| InvocationShapeError::InvalidBaseInterpolationAuthority)?;
    let expected_abi = match mode {
        InterpolationLaunchMode::StageWiseCopyThenInPlace => {
            InterpolationPrimitiveAbi::StageWiseCopyThenInPlaceV1
        }
        InterpolationLaunchMode::StageFusedOutOfPlace => {
            InterpolationPrimitiveAbi::StageFusedOutOfPlaceV1
        }
    };
    if batch.authority != expected
        || batch.authority.identity() == [0; 32]
        || batch.authority.primitive().identity() == [0; 32]
        || batch.authority.primitive().mode() != mode
        || batch.authority.primitive().abi() != expected_abi
        || batch.authority.primitive().effect()
            != InterpolationEffectAbi::PairedFullRangeB2nWithTwiddleSuffixV1
    {
        return Err(InvocationShapeError::InvalidBaseInterpolationAuthority);
    }
    Ok(())
}

fn catalog_scheduled<'a>(
    image: &'a ArenaProgramInventory,
    scheduled: &ScheduledValue,
) -> Result<&'a ProgramValueDesc, InvocationShapeError> {
    let value = catalog_logical(image, scheduled.logical)?;
    if value.component != Some(scheduled.component)
        || value.part != Some(scheduled.part)
        || value.purpose != scheduled.purpose
        || value.ordinal != scheduled.ordinal
        || value.layout.element_count().ok() != Some(scheduled.words)
    {
        return Err(InvocationShapeError::InvalidBaseInterpolationBinding);
    }
    Ok(value)
}

fn catalog_global<'a>(
    image: &'a ArenaProgramInventory,
    scheduled: &ScheduledGlobalValue,
) -> Result<&'a ProgramValueDesc, InvocationShapeError> {
    let value = catalog_logical(image, scheduled.logical)?;
    if value.component.is_some()
        || value.part.is_some()
        || value.purpose != scheduled.purpose
        || value.ordinal != scheduled.ordinal
        || value.layout.element_count().ok() != Some(scheduled.words)
    {
        return Err(InvocationShapeError::InvalidBaseInterpolationBinding);
    }
    Ok(value)
}

fn catalog_logical(
    image: &ArenaProgramInventory,
    logical: crate::arena_plan::LogicalBufferId,
) -> Result<&ProgramValueDesc, InvocationShapeError> {
    image
        .values
        .get(logical.0 as usize)
        .filter(|value| value.logical == logical)
        .ok_or(InvocationShapeError::InvalidCatalogRange(
            ArenaCatalogValueId(logical.0),
        ))
}
