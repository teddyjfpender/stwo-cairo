use std::collections::BTreeSet;

use super::*;

pub(super) fn validate_lineages(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    let mut previous_ordinal = None;
    let mut claimed_predecessors = BTreeSet::new();
    for operation in &input.operations {
        let ExecutionPrimitive::StatementHostIngress {
            source,
            predecessor,
        } = &operation.primitive
        else {
            continue;
        };
        let invalid = || CompiledProofError::InvalidStatementHostLineage {
            operation: operation.id,
        };
        if previous_ordinal.is_some_and(|ordinal| ordinal >= source.producer_ordinal) {
            return Err(invalid());
        }
        previous_ordinal = Some(source.producer_ordinal);

        let Some(predecessor) = predecessor else {
            continue;
        };
        if !claimed_predecessors.insert(predecessor.version) {
            return Err(invalid());
        }
        let predecessor_value = super::value(input, predecessor.version).map_err(|_| invalid())?;
        let predecessor_full = full_range(predecessor_value).ok_or_else(invalid)?;
        let destination = ingress_destination(input, operation).ok_or_else(invalid)?;
        let destination_value = super::value(input, destination.version).map_err(|_| invalid())?;
        let ValueOrigin::OpOutput(producer) = predecessor_value.origin else {
            return Err(invalid());
        };
        let producer_operation = input
            .operations
            .get(producer.0 as usize)
            .filter(|candidate| candidate.id == producer && producer < operation.id)
            .ok_or_else(invalid)?;
        let ExecutionPrimitive::StatementHostIngress {
            source: producer_source,
            ..
        } = &producer_operation.primitive
        else {
            return Err(invalid());
        };
        if *predecessor != predecessor_full
            || predecessor_value.region != Region::Dynamic
            || predecessor_value.layout != destination_value.layout
            || predecessor_value.alignment != destination_value.alignment
            || ingress_destination(input, producer_operation) != Some(*predecessor)
            || producer_source.producer_ordinal >= source.producer_ordinal
            || has_use_at_or_after(input, predecessor.version, operation.id)
        {
            return Err(invalid());
        }
    }
    Ok(())
}

fn ingress_destination(input: &CompiledProofInput, operation: &OpNode) -> Option<ValueRange> {
    input
        .effects
        .iter()
        .find(|effect| effect.id() == operation.effect)?
        .accesses()
        .iter()
        .find_map(|access| access.destination().map(|range| range.value))
}

fn full_range(value: &ValueDesc) -> Option<ValueRange> {
    Some(ValueRange {
        version: value.version,
        elements: ElementRange::new(0, value.layout.element_count().ok()?)?,
    })
}

fn has_use_at_or_after(input: &CompiledProofInput, version: ValueVersion, operation: OpId) -> bool {
    input
        .operations
        .iter()
        .skip(operation.0 as usize)
        .any(|candidate| {
            input
                .effects
                .iter()
                .find(|effect| effect.id() == candidate.effect)
                .is_some_and(|effect| {
                    effect.accesses().iter().any(|access| {
                        access
                            .source()
                            .is_some_and(|source| source.value.version == version)
                    })
                })
        })
        || input
            .transcript_inputs
            .iter()
            .any(|binding| binding.value == version)
        || input
            .output
            .fragments
            .iter()
            .any(|fragment| fragment.source.version == version)
}
