//! Catalog roles, arena relocations and schedule-edge geometry.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::{
    WitnessInputCompactAbi, WitnessInputCompactEffectAbi, WitnessInputCompactExecution,
    WitnessInputCompactRowDomain, WitnessInputSeedAbi, WitnessInputSeedEffectAbi,
    WitnessInputSeedRowDomain,
};

use super::*;
use crate::arena_plan::{BufferPurpose, PlannedWitnessInputCompact, PlannedWitnessInputSeed};

mod compact;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExactSemanticArenaBinding {
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExactSeedBindings {
    pub(super) relocations: WitnessInputSeedRelocations,
    pub(super) scalar: ExactSemanticArenaBinding,
    pub(super) outputs: Vec<ExactSemanticArenaBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExactCompactBindings {
    pub(super) relocations: WitnessInputCompactRelocations,
    pub(super) sources: Vec<ExactSemanticArenaBinding>,
    pub(super) outputs: Vec<ExactSemanticArenaBinding>,
    pub(super) scratch: Vec<ExactSemanticArenaBinding>,
}

pub(super) fn validate_seed_contract(
    contract: &WitnessInputSeedContract,
    planned: &PlannedWitnessInputSeed,
) -> Result<(), InvocationShapeError> {
    let requirements = contract.requirements();
    let launch = contract.launch();
    if requirements != &planned.requirements
        || contract.abi() != WitnessInputSeedAbi::ScalarExpansionV1
        || contract.effect() != WitnessInputSeedEffectAbi::RepeatScalarsAndWriteMechanicalTailV1
        || contract.row_domain() != WitnessInputSeedRowDomain::RealPrefixWithPaddedScalarRowsV1
        || contract.abi().arguments().len() != 8
        || contract.effect_geometry().output_columns.len()
            != requirements.consumer_input_column_words.len()
        || launch.symbol() != "witness_input_seed_kernel"
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

pub(super) fn validate_compact_contract(
    contract: &WitnessInputCompactContract,
    planned: &PlannedWitnessInputCompact,
) -> Result<(), InvocationShapeError> {
    let requirements = contract.requirements();
    if requirements != &planned.requirements
        || contract.abi() != WitnessInputCompactAbi::CanonicalTupleCompactionV1
        || contract.effect() != WitnessInputCompactEffectAbi::GatherStableLexicographicRleScatterV1
        || contract.row_domain() != WitnessInputCompactRowDomain::CanonicalUniquePowerOfTwoPaddingV1
        || contract.abi().arguments().len() != 26
        || contract.descriptor_words().len() != requirements.descriptor_words
        || contract.effect_geometry().sources.len() != requirements.edges.len()
        || contract.effect_geometry().outputs.len()
            != requirements.consumer_input_column_words.len()
        || contract.stages().iter().enumerate().any(|(index, stage)| {
            stage.ordinal as usize != index
                || matches!(
                    stage.execution,
                    WitnessInputCompactExecution::Cub {
                        library_managed_launch_geometry: false,
                        ..
                    } | WitnessInputCompactExecution::Cub {
                        ordered_on_wrapper_stream: false,
                        ..
                    }
                )
        })
        || [
            contract.static_source_identity(),
            contract.wrapper_source_identity(),
            contract.source_identity(),
            contract.requirements_identity(),
            contract.descriptor_identity(),
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

pub(super) fn seed(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    planned: &PlannedWitnessInputSeed,
    contract: &WitnessInputSeedContract,
) -> Result<ExactSeedBindings, InvocationShapeError> {
    let requirements = contract.requirements();
    let scalar = exact_role_binding(
        arena,
        catalog,
        component.component,
        component.part,
        BufferPurpose::WitnessInputSeedScalars,
        0,
        planned.slots.scalar_values,
        requirements.scalar_words,
        exact_elements(
            contract.effect_geometry().scalar_source_start_word as usize,
            contract.effect_geometry().scalar_source_words as usize,
            requirements.scalar_words,
        )?,
    )?;
    let output_pointers = exact_role_arena(
        arena,
        catalog,
        component.component,
        component.part,
        BufferPurpose::WitnessInputSeedOutputPointers,
        0,
        planned.slots.output_pointers,
        requirements.output_pointer_words,
    )?;
    let outputs = planned
        .slots
        .consumer_input_columns
        .iter()
        .copied()
        .zip(&requirements.consumer_input_column_words)
        .zip(&contract.effect_geometry().output_columns)
        .enumerate()
        .map(|(ordinal, ((physical, &words), effect))| {
            if effect.column_ordinal as usize != ordinal || effect.written_words as usize != words {
                return Err(InvocationShapeError::InvalidStructuredAbi);
            }
            exact_role_binding(
                arena,
                catalog,
                component.component,
                component.part,
                BufferPurpose::WitnessInput,
                to_u32(ordinal)?,
                physical,
                words,
                exact_elements(0, words, words)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if outputs.len() != requirements.consumer_input_column_words.len()
        || outputs.len() != contract.effect_geometry().output_columns.len()
        || outputs.last().is_none()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    require_distinct(
        std::iter::once(&scalar.arena)
            .chain(outputs.iter().map(|binding| &binding.arena))
            .chain(std::iter::once(&output_pointers)),
    )?;
    Ok(ExactSeedBindings {
        relocations: WitnessInputSeedRelocations { output_pointers },
        scalar,
        outputs,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compact(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    planned: &PlannedWitnessInputCompact,
    contract: &WitnessInputCompactContract,
    position: ProducerSchedulePosition,
    producer_levels: &BTreeMap<&'static str, u32>,
) -> Result<ExactCompactBindings, InvocationShapeError> {
    let requirements = contract.requirements();
    let source_pointers = exact_role_arena(
        arena,
        catalog,
        component.component,
        component.part,
        BufferPurpose::WitnessInputCompactSourcePointers,
        0,
        planned.slots.source_pointers,
        requirements.source_pointer_words,
    )?;
    let descriptor_workspace = exact_role_arena(
        arena,
        catalog,
        component.component,
        component.part,
        BufferPurpose::WitnessInputCompactDescriptors,
        0,
        planned.slots.descriptors,
        requirements.descriptor_words,
    )?;
    let output_pointers = exact_role_arena(
        arena,
        catalog,
        component.component,
        component.part,
        BufferPurpose::WitnessInputCompactOutputPointers,
        0,
        planned.slots.output_pointers,
        requirements.output_pointer_words,
    )?;
    let sources = compact::sources(
        arena,
        catalog,
        component,
        planned,
        contract,
        position,
        producer_levels,
    )?;
    let outputs = compact::outputs(arena, catalog, component, planned, contract)?;
    let scratch = compact::scratch(arena, catalog, component, planned, contract)?;
    require_distinct(
        sources
            .iter()
            .chain(&outputs)
            .chain(&scratch)
            .map(|binding| &binding.arena)
            .chain([&source_pointers, &descriptor_workspace, &output_pointers].into_iter()),
    )?;
    Ok(ExactCompactBindings {
        relocations: WitnessInputCompactRelocations {
            source_pointers,
            descriptor_workspace,
            output_pointers,
        },
        sources,
        outputs,
        scratch,
    })
}

#[allow(clippy::too_many_arguments)]
fn exact_role_binding(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &'static str,
    part: TracePartId,
    purpose: BufferPurpose,
    ordinal: u32,
    physical: stwo_backend_cuda::ArenaSlotId,
    words: usize,
    elements: ElementRange,
) -> Result<ExactSemanticArenaBinding, InvocationShapeError> {
    let arena_binding = exact_role_arena(
        arena, catalog, component, part, purpose, ordinal, physical, words,
    )?;
    Ok(ExactSemanticArenaBinding {
        value: ArenaCatalogValueId(arena_binding.logical.0),
        arena: arena_binding,
        elements,
    })
}

#[allow(clippy::too_many_arguments)]
fn exact_role_arena(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &'static str,
    part: TracePartId,
    purpose: BufferPurpose,
    ordinal: u32,
    physical: stwo_backend_cuda::ArenaSlotId,
    words: usize,
) -> Result<ArenaBinding, InvocationShapeError> {
    let mut matches = catalog.values.iter().filter(|value| {
        value.component == Some(component)
            && value.part == Some(part)
            && value.purpose == purpose
            && value.ordinal == ordinal
    });
    let value = matches
        .next()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    if matches.next().is_some()
        || value.physical != physical
        || value.words != words
        || arena.binding(value.logical)
            != Some(ArenaBinding {
                logical: value.logical,
                physical,
                len_words: words,
            })
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(ArenaBinding {
        logical: value.logical,
        physical,
        len_words: words,
    })
}

fn exact_elements(
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
    bindings: impl IntoIterator<Item = &'a ArenaBinding>,
) -> Result<(), InvocationShapeError> {
    let mut logical = BTreeSet::new();
    let mut physical = BTreeSet::new();
    for binding in bindings {
        if !logical.insert(binding.logical) || !physical.insert(binding.physical) {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
    }
    Ok(())
}

pub(super) fn exact_component<'a>(
    arena: &'a ProofArenaPlan,
    component: &'static str,
    part: TracePartId,
) -> Result<&'a PlannedWitnessComponent, InvocationShapeError> {
    let mut matches = arena
        .witness()
        .components
        .iter()
        .filter(|planned| planned.component == component && planned.part == part);
    let planned = matches
        .next()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(planned)
}

pub(super) fn producer_levels(
    schedule: &BaseProducerSchedule,
) -> Result<BTreeMap<&'static str, u32>, InvocationShapeError> {
    let mut levels = BTreeMap::new();
    for (level, lanes) in schedule.witness_levels().iter().enumerate() {
        let level = to_u32(level)?;
        for producer in lanes {
            if levels.insert(producer.component, level).is_some() {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
        }
    }
    Ok(levels)
}

#[cfg(test)]
pub(super) fn position_for(
    schedule: &BaseProducerSchedule,
    component: &'static str,
    part: TracePartId,
) -> Result<ProducerSchedulePosition, InvocationShapeError> {
    let mut ordinal = 0usize;
    let mut found = None;
    for (level, lanes) in schedule.witness_levels().iter().enumerate() {
        for (lane, producer) in lanes.iter().enumerate() {
            if producer.component == component && producer.part == Some(part) {
                if found.is_some() {
                    return Err(InvocationShapeError::InvalidScheduledProducerBinding);
                }
                found = Some(ProducerSchedulePosition {
                    level: to_u32(level)?,
                    lane: to_u32(lane)?,
                    ordinal: to_u32(ordinal)?,
                });
            }
            ordinal = ordinal
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
        }
    }
    found.ok_or(InvocationShapeError::InvalidScheduledProducerBinding)
}
