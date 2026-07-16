//! Schedule-driven semantic bindings at the witness-to-interpolation boundary.

use std::collections::BTreeSet;

use super::*;
use crate::arena_plan::{BufferPurpose, PlannedWitnessComponent};
use crate::compiled_proof::ValueVersion;
use crate::resident_runtime::producer_schedule::BaseProducerSchedule;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BaseInterpolationBindingFrontier {
    pub(super) batch: u32,
    pub(super) column: u32,
    pub(super) evaluations: ArenaCatalogValueId,
    pub(super) coefficients: ArenaCatalogValueId,
    pub(super) evaluation_version: ValueVersion,
    pub(super) coefficient_version: ValueVersion,
    pub(super) missing: [MissingOperationField; 2],
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

pub(super) fn base_interpolation_frontier(
    image: &ArenaProgramInventory,
    schedule: &BaseProducerSchedule,
    planned: &PlannedWitnessComponent,
    produced: &[ArenaCatalogValueId],
    ordered_values: &mut Vec<ArenaCatalogValueId>,
) -> Result<Vec<BaseInterpolationBindingFrontier>, InvocationShapeError> {
    let interpolation = schedule
        .interpolation()
        .ok_or(InvocationShapeError::FrontierDidNotAdvance)?;
    let produced_evaluations = produced
        .iter()
        .copied()
        .filter(|&id| {
            image
                .values
                .get(id.0 as usize)
                .is_some_and(|value| value.purpose == BufferPurpose::BaseTrace)
        })
        .collect::<BTreeSet<_>>();
    let produced_logicals = produced_evaluations
        .iter()
        .map(|&id| {
            image
                .values
                .get(id.0 as usize)
                .map(|value| value.logical)
                .ok_or(InvocationShapeError::InvalidCatalogRange(id))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut frontiers = Vec::new();
    for (batch, scheduled_batch) in interpolation.batches.iter().enumerate() {
        for (column, scheduled) in scheduled_batch.columns.iter().enumerate() {
            if !produced_logicals.contains(&scheduled.evaluations.logical) {
                continue;
            }
            let evaluations = catalog_logical(image, scheduled.evaluations.logical)?;
            let coefficients = catalog_logical(image, scheduled.coefficients.logical)?;
            if !produced_evaluations.contains(&evaluations.id)
                || scheduled.evaluations.component != planned.component
                || scheduled.evaluations.part != planned.part
                || scheduled.evaluations.purpose != BufferPurpose::BaseTrace
                || scheduled.coefficients.component != planned.component
                || scheduled.coefficients.part != planned.part
                || scheduled.coefficients.purpose != BufferPurpose::BaseCoefficients
                || evaluations.layout.element_count().ok() != Some(scheduled.evaluations.words)
                || coefficients.layout.element_count().ok() != Some(scheduled.coefficients.words)
            {
                return Err(InvocationShapeError::FrontierDidNotAdvance);
            }
            ordered_values.push(coefficients.id);
            frontiers.push(BaseInterpolationBindingFrontier {
                batch: u32::try_from(batch).map_err(|_| InvocationShapeError::SizeOverflow)?,
                column: u32::try_from(column).map_err(|_| InvocationShapeError::SizeOverflow)?,
                evaluations: evaluations.id,
                coefficients: coefficients.id,
                evaluation_version: ValueVersion(0),
                coefficient_version: ValueVersion(0),
                missing: [
                    MissingOperationField::PrimitiveAuthority,
                    MissingOperationField::EffectContract,
                ],
            });
        }
    }
    if frontiers.len() != planned.program.n_cols as usize {
        return Err(InvocationShapeError::FrontierDidNotAdvance);
    }
    Ok(frontiers)
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

#[cfg(test)]
pub(super) fn try_lower_base_interpolation(
    frontiers: &[BaseInterpolationBindingFrontier],
) -> Result<(), InvocationShapeError> {
    if frontiers.is_empty()
        || frontiers.iter().any(|frontier| {
            frontier.missing
                != [
                    MissingOperationField::PrimitiveAuthority,
                    MissingOperationField::EffectContract,
                ]
        })
    {
        return Err(InvocationShapeError::FrontierDidNotAdvance);
    }
    Err(InvocationShapeError::MissingBaseInterpolationAuthority)
}
