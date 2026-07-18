//! The sole implicit SSA write: an in-place atomic prefix carries its source
//! suffix into the destination version unchanged.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AtomicCarryForward {
    pub source: ValueRange,
    pub destination: ValueRange,
    pub full_source: ValueRange,
    pub full_destination: ValueRange,
}

/// Recognize the exact RC99-shaped partial transition. This deliberately does
/// not admit partial `ReadWrite`, suffix-only, interior, or non-u32 effects.
pub(crate) fn exact_partial_atomic_carry_forward(
    values: &[ValueDesc],
    access: &EffectAccess,
) -> Option<AtomicCarryForward> {
    let EffectAccess::Atomic {
        source,
        destination,
        operation: AtomicOperation::AddU32,
        in_place,
    } = access
    else {
        return None;
    };
    let source_value = values.get(source.value.version.0 as usize)?;
    let destination_value = values.get(destination.value.version.0 as usize)?;
    let words = source_value.layout.element_count().ok()?;
    let written = source.value.elements;
    if source_value.version != source.value.version
        || destination_value.version != destination.value.version
        || source.binding != destination.binding
        || source_value.layout != destination_value.layout
        || source_value.layout.element != ElementType::U32
        || source.value.elements != destination.value.elements
        || written.start != 0
        || written.end >= words
        || in_place.requirement != InPlaceAliasRequirement::Required
        || in_place.discipline != InPlaceDiscipline::ElementWiseReadBeforeWrite
    {
        return None;
    }
    let full = ElementRange::new(0, words)?;
    Some(AtomicCarryForward {
        source: source.value,
        destination: destination.value,
        full_source: ValueRange {
            version: source.value.version,
            elements: full,
        },
        full_destination: ValueRange {
            version: destination.value.version,
            elements: full,
        },
    })
}

pub(super) fn validate_write_coverage(
    input: &CompiledProofInput,
    writes: &mut [Vec<(OpId, ElementRange)>],
) -> Result<(), CompiledProofError> {
    for value in &input.values {
        let ranges = &mut writes[value.version.0 as usize];
        let ValueOrigin::OpOutput(producer) = value.origin else {
            if !ranges.is_empty() {
                return Err(CompiledProofError::ProducerMismatch {
                    value: value.version,
                });
            }
            continue;
        };
        if ranges.iter().any(|(writer, _)| *writer != producer) {
            return Err(CompiledProofError::ProducerMismatch {
                value: value.version,
            });
        }
        ranges.sort_unstable_by_key(|(_, range)| (range.start, range.end));
        let mut cursor = 0;
        for &(_, range) in ranges.iter() {
            if range.start < cursor {
                return Err(CompiledProofError::OverlappingWrite {
                    value: value.version,
                });
            }
            if range.start != cursor {
                return Err(CompiledProofError::IncompleteWrite {
                    value: value.version,
                });
            }
            cursor = range.end;
        }
        let total = value.layout.element_count()?;
        if cursor != total && !is_exact_carried_prefix(input, value, ranges) {
            return Err(CompiledProofError::IncompleteWrite {
                value: value.version,
            });
        }
    }
    Ok(())
}

fn is_exact_carried_prefix(
    input: &CompiledProofInput,
    value: &ValueDesc,
    writes: &[(OpId, ElementRange)],
) -> bool {
    let [(producer, written)] = writes else {
        return false;
    };
    let Some(operation) = input
        .operations
        .get(producer.0 as usize)
        .filter(|operation| operation.id == *producer)
    else {
        return false;
    };
    let Some(effect) = input
        .effects
        .iter()
        .find(|effect| effect.id() == operation.effect)
    else {
        return false;
    };
    let mut carried = effect.accesses().iter().filter_map(|access| {
        exact_partial_atomic_carry_forward(&input.values, access)
            .filter(|carry| carry.destination.version == value.version)
    });
    carried
        .next()
        .is_some_and(|carry| carry.destination.elements == *written)
        && carried.next().is_none()
}
