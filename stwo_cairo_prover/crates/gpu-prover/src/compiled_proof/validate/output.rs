use std::collections::BTreeMap;

use super::*;

pub(super) fn validate(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    input
        .output
        .layout
        .validate()
        .map_err(|_| CompiledProofError::NonCanonicalProofLayout)?;
    let destinations = layout_ranges(&input.output.layout);
    let mut cursor = 0;
    for range in &destinations {
        if range.start != cursor || range.is_empty() {
            return Err(CompiledProofError::NonCanonicalProofLayout);
        }
        cursor = range.end;
    }
    if cursor != input.output.layout.total_words {
        return Err(CompiledProofError::NonCanonicalProofLayout);
    }
    if input.output.sections.len() != ProofBundleSection::CANONICAL.len() {
        return Err(CompiledProofError::ProofSectionCount {
            expected: ProofBundleSection::CANONICAL.len(),
            actual: input.output.sections.len(),
        });
    }

    let mut source_ranges = BTreeMap::<ValueVersion, Vec<ElementRange>>::new();
    for (index, ((binding, section), destination)) in input
        .output
        .sections
        .iter()
        .zip(ProofBundleSection::CANONICAL)
        .zip(destinations)
        .enumerate()
    {
        if binding.section != section {
            return Err(CompiledProofError::ProofSectionOrder { index });
        }
        let source = super::value(input, binding.value)?;
        let ValueOrigin::OpOutput(_) = source.origin else {
            return Err(CompiledProofError::ProofSectionOrigin { index });
        };
        if source.region != Region::Output {
            return Err(CompiledProofError::ProofSectionOrigin { index });
        }
        super::validate_words(
            BindingKind::ProofOutput,
            index as u32,
            source,
            binding.elements,
            destination.len(),
        )?;
        source_ranges
            .entry(binding.value)
            .or_default()
            .push(binding.elements);
    }
    for value in input
        .values
        .iter()
        .filter(|value| value.region == Region::Output)
    {
        let Some(ranges) = source_ranges.get_mut(&value.version) else {
            return Err(CompiledProofError::InvalidProofAssembly);
        };
        ranges.sort_unstable_by_key(|range| (range.start, range.end));
        let mut cursor = 0;
        for range in ranges {
            if range.start != cursor {
                return Err(CompiledProofError::InvalidProofAssembly);
            }
            cursor = range.end;
        }
        if cursor != value.layout.element_count()?
            || value.layout.element != ElementType::U32
            || value.alignment < core::mem::align_of::<u32>()
        {
            return Err(CompiledProofError::InvalidProofAssembly);
        }
    }
    super::reject_overlaps(
        BindingKind::ProofOutput,
        input
            .output
            .sections
            .iter()
            .map(|binding| (binding.value, binding.elements)),
    )
}

fn layout_ranges(
    layout: &crate::proof_bundle::ResidentProofBundleLayout,
) -> [std::ops::Range<usize>; 8] {
    [
        layout.commitments.clone(),
        layout.interaction_claim.clone(),
        layout.interaction_pow.clone(),
        layout.sampled_values.clone(),
        layout.fri_commitments.clone(),
        layout.final_line_poly.clone(),
        layout.query_pow.clone(),
        layout.decommitment.clone(),
    ]
}
