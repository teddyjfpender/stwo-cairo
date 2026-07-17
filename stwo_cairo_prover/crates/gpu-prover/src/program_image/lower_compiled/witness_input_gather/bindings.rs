//! Exact arena, schedule-edge, payload-range, and relocation binding.

use super::*;
use crate::arena_plan::{BufferPurpose, PlannedWitnessInputGather};
use crate::compiled_proof::{BoundValueRange, ValueRange};
use crate::schedule::InputEdge;
use crate::schedule_table::CAIRO_SCHEDULE;

pub(super) fn exact_sources(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    planned: &PlannedWitnessInputGather,
    contract: &WitnessInputGatherContract,
    position: ProducerSchedulePosition,
    producer_levels: &BTreeMap<&'static str, u32>,
    values: &adapter::SemanticValueMap,
) -> Result<Vec<WitnessInputGatherSourceBinding>, InvocationShapeError> {
    let effects = &contract.effect_geometry().edges;
    let canonical = canonical_edges(arena, catalog, component.component)?;
    if planned.producers.len() != planned.sources.len()
        || planned.sources.len() != effects.len()
        || effects.len() != planned.requirements.edges.len()
        || canonical.len() != effects.len()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    planned
        .producers
        .iter()
        .zip(&planned.sources)
        .zip(effects)
        .enumerate()
        .map(|(ordinal, ((&producer, &binding), effect))| {
            let (canonical_producer, word_base, words_per_instance, n_instances) =
                canonical[ordinal];
            let edge = planned.requirements.edges[ordinal].edge;
            if producer != canonical_producer
                || edge.word_base != word_base
                || edge.words_per_instance != words_per_instance
                || edge.n_instances != n_instances
                || effect.source_ordinal as usize != ordinal
                || producer_levels
                    .get(producer)
                    .is_none_or(|&level| level >= position.level)
                || arena.binding(binding.logical) != Some(binding)
            {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            let value = catalog.value(ArenaCatalogValueId(binding.logical.0))?;
            if value.logical != binding.logical
                || value.physical != binding.physical
                || value.words != binding.len_words
                || value.component != Some(producer)
                || value.part != Some(TracePartId::Main)
                || value.purpose != BufferPurpose::SubcomponentInputs
                || value.ordinal != 0
            {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            // Prior-level values must already exist; never allocate a source.
            values.version(value.id)?;
            Ok(WitnessInputGatherSourceBinding {
                producer,
                arena: binding,
                pointer_words: pointer_entry_range(
                    ordinal,
                    planned.requirements.source_pointer_words,
                )?,
                value: value.id,
                elements: exact_range(
                    effect.source_start_words,
                    effect.source_len_words,
                    binding.len_words,
                )?,
                binding: EffectBindingId(to_u32(ordinal)?),
            })
        })
        .collect()
}

fn canonical_edges(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    consumer: &'static str,
) -> Result<Vec<(&'static str, usize, usize, usize)>, InvocationShapeError> {
    let node = CAIRO_SCHEDULE
        .nodes
        .iter()
        .find(|node| node.id == consumer)
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    node.inputs
        .iter()
        .filter_map(|edge| {
            let InputEdge::Producer {
                of,
                word_base,
                words_per_instance,
                n_instances,
            } = edge
            else {
                return None;
            };
            catalog
                .values
                .iter()
                .any(|value| {
                    value.component == Some(*of)
                        && value.part == Some(TracePartId::Main)
                        && value.purpose == BufferPurpose::SubcomponentInputs
                        && arena.binding(value.logical).is_some()
                })
                .then_some(Ok((
                    *of,
                    *word_base as usize,
                    *words_per_instance as usize,
                    *n_instances as usize,
                )))
        })
        .collect()
}

pub(super) fn exact_output_catalog<'a>(
    arena: &ProofArenaPlan,
    catalog: &'a BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    planned: &PlannedWitnessInputGather,
    contract: &WitnessInputGatherContract,
) -> Result<Vec<&'a BaseCatalogValue>, InvocationShapeError> {
    let expected_words = &planned.requirements.consumer_input_column_words;
    if planned.slots.consumer_input_columns.len() != expected_words.len()
        || contract.effect_geometry().output_writes.len() != expected_words.len()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    planned
        .slots
        .consumer_input_columns
        .iter()
        .zip(expected_words)
        .enumerate()
        .map(|(ordinal, (&physical, &words))| {
            let value = exact_role_value(
                catalog,
                component.component,
                component.part,
                BufferPurpose::WitnessInput,
                to_u32(ordinal)?,
            )?;
            let binding = exact_arena_binding(arena, value)?;
            if binding.physical != physical || binding.len_words != words {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            Ok(value)
        })
        .collect()
}

pub(super) fn exact_relocations(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    planned: &PlannedWitnessInputGather,
    contract: &WitnessInputGatherContract,
) -> Result<WitnessInputGatherRelocations, InvocationShapeError> {
    let source_pointers = exact_role_binding(
        arena,
        catalog,
        component,
        BufferPurpose::WitnessInputGatherSourcePointers,
        planned.slots.source_pointers,
        planned.requirements.source_pointer_words,
    )?;
    let descriptor_workspace = exact_role_binding(
        arena,
        catalog,
        component,
        BufferPurpose::WitnessInputGatherDescriptors,
        planned.slots.descriptors,
        contract.descriptor_words().len(),
    )?;
    let output_pointers = exact_role_binding(
        arena,
        catalog,
        component,
        BufferPurpose::WitnessInputGatherOutputPointers,
        planned.slots.output_pointers,
        planned.requirements.output_pointer_words,
    )?;
    if [
        source_pointers.logical,
        descriptor_workspace.logical,
        output_pointers.logical,
    ]
    .into_iter()
    .collect::<BTreeSet<_>>()
    .len()
        != 3
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(WitnessInputGatherRelocations {
        source_pointers,
        descriptor_workspace,
        output_pointers,
    })
}

fn exact_role_binding(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    purpose: BufferPurpose,
    physical: stwo_backend_cuda::ArenaSlotId,
    words: usize,
) -> Result<ArenaBinding, InvocationShapeError> {
    let value = exact_role_value(catalog, component.component, component.part, purpose, 0)?;
    let binding = exact_arena_binding(arena, value)?;
    if binding.physical != physical || binding.len_words != words {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(binding)
}

fn exact_role_value<'a>(
    catalog: &'a BaseProducerCatalog,
    component: &'static str,
    part: TracePartId,
    purpose: BufferPurpose,
    ordinal: u32,
) -> Result<&'a BaseCatalogValue, InvocationShapeError> {
    let mut matches = catalog.values.iter().filter(|value| {
        value.component == Some(component)
            && value.part == Some(part)
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

pub(super) fn exact_arena_binding(
    arena: &ProofArenaPlan,
    value: &BaseCatalogValue,
) -> Result<ArenaBinding, InvocationShapeError> {
    let binding = ArenaBinding {
        logical: value.logical,
        physical: value.physical,
        len_words: value.words,
    };
    if arena.binding(value.logical) == Some(binding) {
        Ok(binding)
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
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

pub(super) fn exact_range(
    start: usize,
    len: usize,
    available: usize,
) -> Result<ElementRange, InvocationShapeError> {
    let end = start
        .checked_add(len)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if end > available {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    ElementRange::new(start, end).ok_or(InvocationShapeError::InvalidScheduledProducerBinding)
}

pub(super) fn pointer_entry_range(
    ordinal: usize,
    table_words: usize,
) -> Result<ElementRange, InvocationShapeError> {
    let start = ordinal
        .checked_mul(POINTER_WORDS)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let end = start
        .checked_add(POINTER_WORDS)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if end > table_words {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    ElementRange::new(start, end).ok_or(InvocationShapeError::InvalidScheduledProducerBinding)
}

pub(super) fn bound(
    binding: EffectBindingId,
    version: ValueVersion,
    elements: ElementRange,
) -> BoundValueRange {
    BoundValueRange {
        binding,
        value: ValueRange { version, elements },
    }
}

pub(super) fn to_u32(value: usize) -> Result<u32, InvocationShapeError> {
    u32::try_from(value).map_err(|_| InvocationShapeError::SizeOverflow)
}
