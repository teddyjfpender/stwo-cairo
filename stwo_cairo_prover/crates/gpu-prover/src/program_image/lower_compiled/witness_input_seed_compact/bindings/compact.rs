//! Compact-only producer edges, output roles and transient scratch.

use super::*;
use crate::schedule::InputEdge;
use crate::schedule_table::CAIRO_SCHEDULE;

pub(super) fn sources(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    planned: &PlannedWitnessInputCompact,
    contract: &WitnessInputCompactContract,
    position: ProducerSchedulePosition,
    producer_levels: &BTreeMap<&'static str, u32>,
) -> Result<Vec<ExactSemanticArenaBinding>, InvocationShapeError> {
    let node = CAIRO_SCHEDULE
        .nodes
        .iter()
        .find(|node| node.id == component.component)
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    let canonical = node
        .inputs
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
                        && value.ordinal == 0
                })
                .then_some((*of, *word_base, *words_per_instance, *n_instances))
        })
        .collect::<Vec<_>>();
    let effects = &contract.effect_geometry().sources;
    if canonical.len() != planned.sources.len()
        || planned.sources.len() != contract.requirements().edges.len()
        || planned.sources.len() != effects.len()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    planned
        .sources
        .iter()
        .zip(&contract.requirements().edges)
        .zip(effects)
        .zip(canonical)
        .enumerate()
        .map(
            |(ordinal, (((&binding, edge), effect), (producer, base, words, instances)))| {
                if edge.edge.word_base != base as usize
                    || edge.edge.words_per_instance != words as usize
                    || edge.edge.n_instances != instances as usize
                    || effect.source_ordinal as usize != ordinal
                    || producer_levels
                        .get(producer)
                        .is_none_or(|&level| level >= position.level)
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
                    || arena.binding(value.logical) != Some(binding)
                {
                    return Err(InvocationShapeError::InvalidScheduledProducerBinding);
                }
                let elements =
                    exact_elements(effect.read_start_words, effect.read_len_words, value.words)?;
                if elements.end != edge.required_source_words {
                    return Err(InvocationShapeError::InvalidStructuredAbi);
                }
                Ok(ExactSemanticArenaBinding {
                    arena: binding,
                    value: value.id,
                    elements,
                })
            },
        )
        .collect()
}

pub(super) fn outputs(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    planned: &PlannedWitnessInputCompact,
    contract: &WitnessInputCompactContract,
) -> Result<Vec<ExactSemanticArenaBinding>, InvocationShapeError> {
    let effects = &contract.effect_geometry().outputs;
    if planned.slots.consumer_input_columns.len()
        != contract.requirements().consumer_input_column_words.len()
        || effects.len() != contract.requirements().consumer_input_column_words.len()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    planned
        .slots
        .consumer_input_columns
        .iter()
        .copied()
        .zip(&contract.requirements().consumer_input_column_words)
        .zip(effects)
        .enumerate()
        .map(|(ordinal, ((physical, &words), effect))| {
            if effect.output_ordinal as usize != ordinal {
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
                exact_elements(effect.write_start_words, effect.write_len_words, words)?,
            )
        })
        .collect()
}

pub(super) fn scratch(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    planned: &PlannedWitnessInputCompact,
    contract: &WitnessInputCompactContract,
) -> Result<Vec<ExactSemanticArenaBinding>, InvocationShapeError> {
    let scratch = contract.effect_geometry().scratch;
    let roles = [
        (
            BufferPurpose::WitnessInputCompactTupleScratch,
            0,
            planned.slots.tuple_scratch,
            scratch.tuple_words,
        ),
        (
            BufferPurpose::WitnessInputCompactSortKey,
            0,
            planned.slots.sort_keys_a,
            scratch.sort_key_words_each,
        ),
        (
            BufferPurpose::WitnessInputCompactSortKey,
            1,
            planned.slots.sort_keys_b,
            scratch.sort_key_words_each,
        ),
        (
            BufferPurpose::WitnessInputCompactSortIndex,
            0,
            planned.slots.sort_indices_a,
            scratch.sort_index_words_each,
        ),
        (
            BufferPurpose::WitnessInputCompactSortIndex,
            1,
            planned.slots.sort_indices_b,
            scratch.sort_index_words_each,
        ),
        (
            BufferPurpose::WitnessInputCompactRunHeads,
            0,
            planned.slots.run_heads,
            scratch.run_words_each,
        ),
        (
            BufferPurpose::WitnessInputCompactRunPositions,
            0,
            planned.slots.run_positions,
            scratch.run_words_each,
        ),
        (
            BufferPurpose::WitnessInputCompactUniqueCount,
            0,
            planned.slots.n_unique,
            scratch.unique_count_words,
        ),
        (
            BufferPurpose::WitnessInputCompactSortTemp,
            0,
            planned.slots.sort_temp,
            scratch.sort_temp_capacity_words,
        ),
        (
            BufferPurpose::WitnessInputCompactScanTemp,
            0,
            planned.slots.scan_temp,
            scratch.scan_temp_capacity_words,
        ),
    ];
    roles
        .into_iter()
        .map(|(purpose, ordinal, physical, words)| {
            exact_role_binding(
                arena,
                catalog,
                component.component,
                component.part,
                purpose,
                ordinal,
                physical,
                words,
                exact_elements(0, words, words)?,
            )
        })
        .collect()
}
