//! Exact schedule, catalog-role and arena-range binding.

use std::collections::HashSet;

use stwo_backend_cuda::{
    WitnessCasmInputAbi, WitnessCasmInputColumnValue, WitnessCasmInputEffectAbi,
    WitnessCasmInputRowDomain, WITNESS_CASM_STATE_WORDS,
};

use super::*;
use crate::arena_plan::{BufferPurpose, PlannedWitnessCasmInput};
use crate::resident_runtime::producer_schedule::{BaseProducerSchedule, WitnessProducerKind};

#[derive(Clone, Copy)]
pub(super) struct ScheduledWitnessCasmLane<'a> {
    pub(super) position: ProducerSchedulePosition,
    pub(super) component: &'a PlannedWitnessComponent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExactArenaRange {
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExactOutput {
    pub(super) ordinal: u32,
    pub(super) value_kind: WitnessCasmInputColumnValue,
    pub(super) writer_use: WitnessCasmWriterUse,
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
}

pub(super) struct ExactWitnessCasmLane<'a> {
    pub(super) position: ProducerSchedulePosition,
    pub(super) component: &'a PlannedWitnessComponent,
    pub(super) contract: WitnessCasmInputContract,
    pub(super) staging: ExactArenaRange,
    pub(super) outputs: Vec<ExactOutput>,
}

pub(super) fn scheduled_lanes(
    arena: &ProofArenaPlan,
) -> Result<Vec<ScheduledWitnessCasmLane<'_>>, InvocationShapeError> {
    let schedule = BaseProducerSchedule::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidScheduledProducerBinding)?;
    let planned_count = arena
        .witness()
        .components
        .iter()
        .filter(|component| component.input_casm.is_some())
        .count();
    let mut lowered = Vec::with_capacity(planned_count);
    let mut seen = HashSet::new();
    let mut ordinal = 0usize;
    for (level, lanes) in schedule.witness_levels().iter().enumerate() {
        for (lane, producer) in lanes.iter().enumerate() {
            let position = ProducerSchedulePosition {
                level: to_u32(level)?,
                lane: to_u32(lane)?,
                ordinal: to_u32(ordinal)?,
            };
            ordinal = ordinal
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            let Some(part) = producer.part else {
                continue;
            };
            let component = exact_component(arena, producer.component, part)?;
            if component.input_casm.is_none() {
                continue;
            }
            if producer.kind != WitnessProducerKind::Recorded
                || component.input_gather.is_some()
                || component.input_seed.is_some()
                || component.input_compact.is_some()
                || !seen.insert((component.component, component.part))
            {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            lowered.push(ScheduledWitnessCasmLane {
                position,
                component,
            });
        }
    }
    if lowered.len() != planned_count {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(lowered)
}

pub(super) fn bind_lane<'a>(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    scheduled: ScheduledWitnessCasmLane<'a>,
) -> Result<ExactWitnessCasmLane<'a>, InvocationShapeError> {
    let planned = scheduled
        .component
        .input_casm
        .as_ref()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    let contract = WitnessCasmInputContract::compile(&planned.requirements)
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    validate_contract(&contract, planned)?;
    let staging = staging(arena, catalog, planned, &contract)?;
    let mut outputs = outputs(arena, catalog, scheduled.component, planned, &contract)?;
    bind_writer_use(catalog, arena, scheduled.component, &mut outputs)?;
    require_distinct(
        std::iter::once(&staging.arena).chain(outputs.iter().map(|output| &output.arena)),
    )?;
    Ok(ExactWitnessCasmLane {
        position: scheduled.position,
        component: scheduled.component,
        contract,
        staging,
        outputs,
    })
}

fn validate_contract(
    contract: &WitnessCasmInputContract,
    planned: &PlannedWitnessCasmInput,
) -> Result<(), InvocationShapeError> {
    let requirements = contract.requirements();
    let geometry = contract.effect_geometry();
    let launch = contract.launch();
    let source_words = (geometry.source_rows as usize)
        .checked_mul(geometry.state_words_per_row as usize)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let expected_fixed = [
        to_u32(WITNESS_CASM_STATE_WORDS)?,
        to_u32(requirements.n_real_rows)?,
        to_u32(requirements.consumer_rows)?,
        u32::from(requirements.include_iota),
    ];
    if requirements != &planned.requirements
        || contract.abi() != WitnessCasmInputAbi::RowMajorStateScatterV1
        || contract.effect() != WitnessCasmInputEffectAbi::ScatterStateAndMechanicalColumnsV1
        || contract.row_domain() != WitnessCasmInputRowDomain::RealPrefixWithRowZeroPaddingV1
        || contract.abi().arguments().len() != 9
        || contract.fixed_words() != &expected_fixed
        || geometry.source_start_word != 0
        || geometry.source_rows as usize != requirements.n_real_rows
        || geometry.state_words_per_row as usize != WITNESS_CASM_STATE_WORDS
        || source_words != requirements.staging_words
        || geometry.consumer_rows as usize != requirements.consumer_rows
        || geometry.output_columns.len() != requirements.consumer_input_column_words.len()
        || launch.symbol() != "witness_casm_input_scatter_kernel"
        || launch.grid != [to_u32(requirements.consumer_rows)?.div_ceil(256), 1, 1]
        || launch.block != [256, 1, 1]
        || launch.dynamic_shared_bytes != 0
        || launch.cooperative
        || launch.cluster.is_some()
        || [
            contract.static_source_identity(),
            contract.wrapper_source_identity(),
            contract.source_identity(),
            contract.requirements_identity(),
            contract.fixed_identity(),
            contract.abi_identity(),
            contract.effect_identity(),
            contract.launch_identity(),
            contract.identity(),
        ]
        .contains(&[0; 32])
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    Ok(())
}

fn staging(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    planned: &PlannedWitnessCasmInput,
    contract: &WitnessCasmInputContract,
) -> Result<ExactArenaRange, InvocationShapeError> {
    let value = unique_role(catalog, None, None, BufferPurpose::WitnessInput, 0)?;
    if value.physical != planned.slots.staging
        || value.words < contract.requirements().staging_words
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(ExactArenaRange {
        arena: exact_arena(arena, value)?,
        value: value.id,
        elements: exact_range(0, contract.requirements().staging_words, value.words)?,
    })
}

fn outputs(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    planned: &PlannedWitnessCasmInput,
    contract: &WitnessCasmInputContract,
) -> Result<Vec<ExactOutput>, InvocationShapeError> {
    let expected = planned.requirements.consumer_input_column_words.len();
    if planned.slots.consumer_input_columns.len() != expected
        || contract.effect_geometry().output_columns.len() != expected
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    planned
        .slots
        .consumer_input_columns
        .iter()
        .copied()
        .zip(&planned.requirements.consumer_input_column_words)
        .zip(&contract.effect_geometry().output_columns)
        .enumerate()
        .map(|(ordinal, ((physical, &words), effect))| {
            let ordinal_u32 = to_u32(ordinal)?;
            let value = unique_role(
                catalog,
                Some(component.component),
                Some(component.part),
                BufferPurpose::WitnessInput,
                ordinal_u32,
            )?;
            if value.physical != physical
                || value.words != words
                || effect.column_ordinal != ordinal_u32
                || effect.written_words as usize != words
            {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            Ok(ExactOutput {
                ordinal: ordinal_u32,
                value_kind: effect.value,
                writer_use: WitnessCasmWriterUse::Active,
                arena: exact_arena(arena, value)?,
                value: value.id,
                elements: exact_range(0, words, value.words)?,
            })
        })
        .collect()
}

fn bind_writer_use(
    catalog: &BaseProducerCatalog,
    arena: &ProofArenaPlan,
    component: &PlannedWitnessComponent,
    outputs: &mut [ExactOutput],
) -> Result<(), InvocationShapeError> {
    let writer = derive_invocation(catalog, arena, component)?;
    let SourceArgument::PointerTable {
        ordinal, entries, ..
    } = writer
        .source_arguments
        .first()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?
    else {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    };
    if *ordinal != 0 || entries.len() != outputs.len() {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    for (entry, output) in entries.iter().zip(outputs) {
        if entry.entry != output.ordinal
            || entry.target.value != output.value
            || entry.target.elements != (output.elements.start..output.elements.end)
        {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
        let use_kind = match entry.target.access {
            InvocationAccess::Read => WitnessCasmWriterUse::Active,
            InvocationAccess::Inactive
                if matches!(
                    output.value_kind,
                    WitnessCasmInputColumnValue::Enabler | WitnessCasmInputColumnValue::Iota
                ) =>
            {
                WitnessCasmWriterUse::InactiveMechanical
            }
            InvocationAccess::Inactive | InvocationAccess::Write => {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding)
            }
        };
        output.writer_use = use_kind;
    }
    Ok(())
}

fn unique_role<'a>(
    catalog: &'a BaseProducerCatalog,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    purpose: BufferPurpose,
    ordinal: u32,
) -> Result<&'a BaseCatalogValue, InvocationShapeError> {
    let mut matches = catalog.values.iter().filter(|value| {
        value.component == component
            && value.part == part
            && value.purpose == purpose
            && value.ordinal == ordinal
    });
    let value = matches
        .next()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(value)
}

fn exact_arena(
    arena: &ProofArenaPlan,
    value: &BaseCatalogValue,
) -> Result<ArenaBinding, InvocationShapeError> {
    let exact = ArenaBinding {
        logical: value.logical,
        physical: value.physical,
        len_words: value.words,
    };
    if arena.binding(value.logical) == Some(exact) {
        Ok(exact)
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

fn exact_range(
    start: usize,
    len: usize,
    capacity: usize,
) -> Result<ElementRange, InvocationShapeError> {
    let end = start
        .checked_add(len)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if end > capacity {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    ElementRange::new(start, end).ok_or(InvocationShapeError::InvalidStructuredAbi)
}

fn require_distinct<'a>(
    values: impl IntoIterator<Item = &'a ArenaBinding>,
) -> Result<(), InvocationShapeError> {
    let mut logical = BTreeSet::new();
    let mut physical = BTreeSet::new();
    for value in values {
        if !logical.insert(value.logical) || !physical.insert(value.physical) {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
    }
    Ok(())
}

fn exact_component<'a>(
    arena: &'a ProofArenaPlan,
    component: &'static str,
    part: TracePartId,
) -> Result<&'a PlannedWitnessComponent, InvocationShapeError> {
    let mut matches = arena
        .witness()
        .components
        .iter()
        .filter(|planned| planned.component == component && planned.part == part);
    let component = matches
        .next()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(component)
}

#[cfg(test)]
pub(super) fn position_for(
    arena: &ProofArenaPlan,
    component: &PlannedWitnessComponent,
) -> Result<ProducerSchedulePosition, InvocationShapeError> {
    scheduled_lanes(arena)?
        .into_iter()
        .find(|lane| {
            lane.component.component == component.component && lane.component.part == component.part
        })
        .map(|lane| lane.position)
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)
}

fn to_u32(value: usize) -> Result<u32, InvocationShapeError> {
    u32::try_from(value).map_err(|_| InvocationShapeError::SizeOverflow)
}
