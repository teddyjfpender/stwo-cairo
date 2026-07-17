//! Exact multiplicity lineage reconstruction.

use std::collections::{BTreeMap, BTreeSet};

use super::*;

pub(super) fn validate_prefix(
    clear: &LoweredMultiplicityClear,
    seed: Option<&LoweredMultiplicityFeed>,
    transitions_after_step: &[Vec<TransitionReceipt>],
    values: &SemanticValueMap,
) -> Result<(), InvocationShapeError> {
    reconstruct(clear, seed, transitions_after_step, values).map(|_| ())
}

pub(super) fn exact_post_witness_current(
    clear: &LoweredMultiplicityClear,
    seed: Option<&LoweredMultiplicityFeed>,
    transitions_after_step: &[Vec<TransitionReceipt>],
    values: &SemanticValueMap,
) -> Result<Vec<PostWitnessMultiplicity>, InvocationShapeError> {
    reconstruct(clear, seed, transitions_after_step, values)
}

fn reconstruct(
    clear: &LoweredMultiplicityClear,
    seed: Option<&LoweredMultiplicityFeed>,
    transitions_after_step: &[Vec<TransitionReceipt>],
    values: &SemanticValueMap,
) -> Result<Vec<PostWitnessMultiplicity>, InvocationShapeError> {
    let mut lineages = BTreeMap::new();
    let mut names = BTreeSet::new();
    for destination in &clear.destinations {
        if destination.ordinal as usize != lineages.len()
            || !names.insert(destination.name)
            || lineages
                .insert(
                    destination.value,
                    (
                        destination.ordinal,
                        destination.name,
                        destination.elements,
                        vec![destination.version],
                    ),
                )
                .is_some()
        {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
    }
    if lineages.is_empty() {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }

    if let Some(seed) = seed {
        for transition in feed_transitions(seed) {
            append(&mut lineages, transition)?;
        }
    }
    for transitions in transitions_after_step {
        for transition in transitions {
            append(&mut lineages, transition.clone())?;
        }
    }

    let mut result = lineages
        .into_iter()
        .map(|(value, (ordinal, name, _, lineage))| {
            if values.versions_for(value).collect::<Vec<_>>() != lineage {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            Ok(PostWitnessMultiplicity {
                ordinal,
                name,
                value,
                current: *lineage
                    .last()
                    .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?,
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    result.sort_by_key(|entry| entry.ordinal);
    if result
        .iter()
        .enumerate()
        .any(|(ordinal, entry)| entry.ordinal as usize != ordinal)
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(result)
}

fn append(
    lineages: &mut BTreeMap<
        ArenaCatalogValueId,
        (u32, &'static str, ElementRange, Vec<ValueVersion>),
    >,
    transition: TransitionReceipt,
) -> Result<(), InvocationShapeError> {
    if lineages
        .values()
        .any(|(_, _, _, versions)| versions.contains(&transition.destination))
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let (_, _, elements, lineage) = lineages
        .get_mut(&transition.value)
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    if transition.elements != *elements
        || lineage.last() != Some(&transition.source)
        || transition.source == transition.destination
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    lineage.push(transition.destination);
    Ok(())
}
