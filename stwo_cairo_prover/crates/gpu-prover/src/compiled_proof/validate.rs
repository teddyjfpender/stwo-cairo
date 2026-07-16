use std::collections::BTreeSet;

use stwo_backend_cuda::TranscriptOperation;

use super::*;

pub(super) fn validate(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), CompiledProofError> {
    validate_operations(input, transcript)?;
    validate_values(input)?;
    validate_authority(input)?;
    validate_transcript(input, transcript)?;
    validate_output(input)?;
    Ok(())
}

fn validate_operations(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), CompiledProofError> {
    let mut previous_stage = None;
    for (index, operation) in input.operations.iter().enumerate() {
        let expected = OpId(u32::try_from(index).map_err(|_| CompiledProofError::SizeOverflow)?);
        if operation.id != expected {
            return Err(CompiledProofError::NonDenseOperation {
                expected,
                actual: operation.id,
            });
        }
        let stage =
            stage_index(operation.stage, transcript).ok_or(CompiledProofError::UnknownStage {
                operation: operation.id,
            })?;
        if previous_stage.is_some_and(|previous| previous > stage) {
            return Err(CompiledProofError::StageOrder {
                previous: OpId(operation.id.0 - 1),
                current: operation.id,
            });
        }
        previous_stage = Some(stage);

        let inputs = unique_edges(operation.id, &operation.inputs)?;
        let outputs = unique_edges(operation.id, &operation.outputs)?;
        if let Some(value) = inputs.intersection(&outputs).next() {
            return Err(CompiledProofError::InputOutputAlias {
                operation: operation.id,
                value: **value,
            });
        }
        for &value in inputs.union(&outputs) {
            if value.0 as usize >= input.values.len() {
                return Err(CompiledProofError::UnknownValue {
                    operation: operation.id,
                    value: *value,
                });
            }
        }
    }
    Ok(())
}

fn unique_edges<'a>(
    operation: OpId,
    values: &'a [ValueId],
) -> Result<BTreeSet<&'a ValueId>, CompiledProofError> {
    let set = values.iter().collect::<BTreeSet<_>>();
    if set.len() != values.len() {
        let mut seen = BTreeSet::new();
        let value = values
            .iter()
            .copied()
            .find(|value| !seen.insert(*value))
            .ok_or(CompiledProofError::SizeOverflow)?;
        return Err(CompiledProofError::DuplicateOperationEdge { operation, value });
    }
    Ok(set)
}

fn validate_values(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    let mut actual_consumers = vec![Vec::new(); input.values.len()];
    for operation in &input.operations {
        for &value in &operation.inputs {
            actual_consumers[value.0 as usize].push(operation.id);
        }
        for &value in &operation.outputs {
            if input.values[value.0 as usize].origin != ValueOrigin::OpOutput(operation.id) {
                return Err(CompiledProofError::ProducerMismatch { value });
            }
        }
    }

    for (index, value) in input.values.iter().enumerate() {
        let expected = ValueId(u32::try_from(index).map_err(|_| CompiledProofError::SizeOverflow)?);
        if value.id != expected {
            return Err(CompiledProofError::NonDenseValue {
                expected,
                actual: value.id,
            });
        }
        if validate_value_layout(value).is_err()
            || value.alignment == 0
            || !value.alignment.is_power_of_two()
        {
            return Err(CompiledProofError::InvalidValue { value: value.id });
        }
        if value.consumers.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(CompiledProofError::ConsumerOrder { value: value.id });
        }
        for &consumer in &value.consumers {
            if consumer.0 as usize >= input.operations.len() {
                return Err(CompiledProofError::ConsumerMismatch { value: value.id });
            }
            if let ValueOrigin::OpOutput(producer) = value.origin {
                if producer >= consumer {
                    return Err(CompiledProofError::ProducerAfterConsumer {
                        value: value.id,
                        consumer,
                    });
                }
            }
        }
        if value.consumers != actual_consumers[index] {
            return Err(CompiledProofError::ConsumerMismatch { value: value.id });
        }
        if let ValueOrigin::OpOutput(producer) = value.origin {
            let operation = input.operations.get(producer.0 as usize).ok_or(
                CompiledProofError::UnknownProducer {
                    value: value.id,
                    producer,
                },
            )?;
            if operation.id != producer
                || operation
                    .outputs
                    .iter()
                    .filter(|&&output| output == value.id)
                    .count()
                    != 1
            {
                return Err(CompiledProofError::ProducerMismatch { value: value.id });
            }
        }
    }
    Ok(())
}

fn validate_authority(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    let semantic = input
        .operations
        .iter()
        .map(|operation| operation.semantic_id)
        .collect::<BTreeSet<_>>();
    if semantic.len() != input.operations.len() {
        return Err(CompiledProofError::AuthorityMismatch(
            AuthorityKind::SemanticOperation,
        ));
    }
    let kernels = input
        .operations
        .iter()
        .map(|operation| operation.kernel_id)
        .collect::<BTreeSet<_>>();
    let effects = input
        .operations
        .iter()
        .map(|operation| operation.effects)
        .collect::<BTreeSet<_>>();
    exact_authority(
        AuthorityKind::SemanticOperation,
        &input.authority.operations,
        semantic,
        |id| id.0,
    )?;
    exact_authority(
        AuthorityKind::AotKernel,
        &input.authority.kernels,
        kernels,
        |id| id.0,
    )?;
    exact_authority(
        AuthorityKind::EffectContract,
        &input.authority.effects,
        effects,
        |id| u32::from(id.0 != [0; 32]),
    )
}

fn exact_authority<T: Copy + Ord>(
    kind: AuthorityKind,
    declared: &[T],
    used: BTreeSet<T>,
    raw: impl Fn(T) -> u32,
) -> Result<(), CompiledProofError> {
    if declared.is_empty()
        || declared.iter().copied().any(|id| raw(id) == 0)
        || declared.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(CompiledProofError::NonCanonicalAuthority(kind));
    }
    if declared.iter().copied().ne(used) {
        return Err(CompiledProofError::AuthorityMismatch(kind));
    }
    Ok(())
}

fn validate_transcript(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), CompiledProofError> {
    if input.transcript_inputs.len() != transcript.inputs().len() {
        return Err(CompiledProofError::TranscriptInputCount {
            expected: transcript.inputs().len(),
            actual: input.transcript_inputs.len(),
        });
    }
    for (index, (binding, requirement)) in input
        .transcript_inputs
        .iter()
        .zip(transcript.inputs())
        .enumerate()
    {
        let expected_id = requirement.semantic.id()?;
        if binding.id != expected_id {
            return Err(CompiledProofError::TranscriptBindingOrder {
                kind: BindingKind::TranscriptInput,
                index,
            });
        }
        let value = value(input, binding.value)?;
        validate_words(
            BindingKind::TranscriptInput,
            binding.id.0,
            value,
            &binding.value_words,
            requirement.min_words,
        )?;
        if let ValueOrigin::OpOutput(producer) = value.origin {
            let produced = stage_index(input.operations[producer.0 as usize].stage, transcript)
                .ok_or(CompiledProofError::UnknownStage {
                    operation: producer,
                })?;
            let consumed = transcript_input_stage(transcript, binding.id).ok_or(
                CompiledProofError::TranscriptCausality {
                    kind: BindingKind::TranscriptInput,
                    id: binding.id.0,
                },
            )?;
            if produced > consumed {
                return Err(CompiledProofError::TranscriptCausality {
                    kind: BindingKind::TranscriptInput,
                    id: binding.id.0,
                });
            }
        } else if let ValueOrigin::TranscriptOutput(output) = value.origin {
            let drawn = transcript_output_stage(transcript, output).ok_or(
                CompiledProofError::TranscriptCausality {
                    kind: BindingKind::TranscriptInput,
                    id: binding.id.0,
                },
            )?;
            let consumed = transcript_input_stage(transcript, binding.id).ok_or(
                CompiledProofError::TranscriptCausality {
                    kind: BindingKind::TranscriptInput,
                    id: binding.id.0,
                },
            )?;
            if drawn >= consumed {
                return Err(CompiledProofError::TranscriptCausality {
                    kind: BindingKind::TranscriptInput,
                    id: binding.id.0,
                });
            }
        }
    }
    reject_overlaps(
        BindingKind::TranscriptInput,
        input
            .transcript_inputs
            .iter()
            .map(|binding| (binding.value, &binding.value_words)),
    )?;

    if input.transcript_outputs.len() != transcript.outputs().len() {
        return Err(CompiledProofError::TranscriptOutputCount {
            expected: transcript.outputs().len(),
            actual: input.transcript_outputs.len(),
        });
    }
    for (index, (binding, requirement)) in input
        .transcript_outputs
        .iter()
        .zip(transcript.outputs())
        .enumerate()
    {
        let expected_id = requirement.semantic.id()?;
        if binding.id != expected_id {
            return Err(CompiledProofError::TranscriptBindingOrder {
                kind: BindingKind::TranscriptOutput,
                index,
            });
        }
        let value = value(input, binding.value)?;
        validate_words(
            BindingKind::TranscriptOutput,
            binding.id.0,
            value,
            &binding.value_words,
            requirement.min_words,
        )?;
        let value_words = validate_value_layout(value)? / core::mem::size_of::<u32>();
        if binding.value_words != (0..value_words) {
            return Err(CompiledProofError::BindingRange {
                kind: BindingKind::TranscriptOutput,
                id: binding.id.0,
            });
        }
        if value.origin != ValueOrigin::TranscriptOutput(binding.id) {
            return Err(CompiledProofError::TranscriptBindingOrigin { output: binding.id });
        }
        let drawn = transcript_output_stage(transcript, binding.id).ok_or(
            CompiledProofError::TranscriptCausality {
                kind: BindingKind::TranscriptOutput,
                id: binding.id.0,
            },
        )?;
        if value.consumers.iter().any(|consumer| {
            stage_index(input.operations[consumer.0 as usize].stage, transcript)
                .is_none_or(|stage| stage <= drawn)
        }) {
            return Err(CompiledProofError::TranscriptCausality {
                kind: BindingKind::TranscriptOutput,
                id: binding.id.0,
            });
        }
    }
    for value in &input.values {
        if let ValueOrigin::TranscriptOutput(id) = value.origin {
            let bindings = input
                .transcript_outputs
                .iter()
                .filter(|binding| binding.id == id && binding.value == value.id)
                .count();
            if bindings != 1 {
                return Err(CompiledProofError::OrphanTranscriptOutput { value: value.id });
            }
        }
    }
    reject_overlaps(
        BindingKind::TranscriptOutput,
        input
            .transcript_outputs
            .iter()
            .map(|binding| (binding.value, &binding.value_words)),
    )
}

fn validate_output(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    input
        .output
        .layout
        .validate()
        .map_err(|_| CompiledProofError::NonCanonicalProofLayout)?;
    let ranges = layout_ranges(&input.output.layout);
    let mut cursor = 0usize;
    for range in &ranges {
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
    let bundle = input.output.sections[0].value;
    for (index, ((binding, section), destination)) in input
        .output
        .sections
        .iter()
        .zip(ProofBundleSection::CANONICAL)
        .zip(&ranges)
        .enumerate()
    {
        if binding.section != section || binding.value != bundle {
            return Err(CompiledProofError::ProofSectionOrder { index });
        }
        let source = value(input, binding.value)?;
        if !matches!(source.origin, ValueOrigin::OpOutput(_)) {
            return Err(CompiledProofError::ProofSectionOrigin { index });
        }
        validate_words(
            BindingKind::ProofOutput,
            index as u32,
            source,
            &binding.value_words,
            destination.len(),
        )?;
        if binding.value_words != *destination {
            return Err(CompiledProofError::BindingRange {
                kind: BindingKind::ProofOutput,
                id: index as u32,
            });
        }
    }
    let bundle_value = value(input, bundle)?;
    let expected_bytes = input
        .output
        .layout
        .total_words
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(CompiledProofError::SizeOverflow)?;
    let ValueOrigin::OpOutput(assembly) = bundle_value.origin else {
        return Err(CompiledProofError::InvalidProofAssembly);
    };
    let assembly = input
        .operations
        .get(assembly.0 as usize)
        .ok_or(CompiledProofError::InvalidProofAssembly)?;
    if bundle_value.layout.element.bytes != core::mem::size_of::<u32>()
        || bundle_value.alignment < core::mem::align_of::<u32>()
        || validate_value_layout(bundle_value)? != expected_bytes
        || assembly.stage != ProofStage::AfterTranscript
        || assembly.outputs.as_slice() != [bundle]
    {
        return Err(CompiledProofError::InvalidProofAssembly);
    }
    reject_overlaps(
        BindingKind::ProofOutput,
        input
            .output
            .sections
            .iter()
            .map(|binding| (binding.value, &binding.value_words)),
    )
}

fn value(input: &CompiledProofInput, id: ValueId) -> Result<&ValueDesc, CompiledProofError> {
    input
        .values
        .get(id.0 as usize)
        .filter(|value| value.id == id)
        .ok_or(CompiledProofError::InvalidValue { value: id })
}

fn validate_words(
    kind: BindingKind,
    id: u32,
    value: &ValueDesc,
    words: &Range<usize>,
    expected_words: usize,
) -> Result<(), CompiledProofError> {
    let bytes = validate_value_layout(value)?;
    if value.layout.element.bytes != core::mem::size_of::<u32>()
        || value.alignment < core::mem::align_of::<u32>()
        || bytes % core::mem::size_of::<u32>() != 0
        || words.start >= words.end
        || words.end > bytes / core::mem::size_of::<u32>()
        || words.len() != expected_words
    {
        return Err(CompiledProofError::BindingRange { kind, id });
    }
    Ok(())
}

fn validate_value_layout(value: &ValueDesc) -> Result<usize, CompiledProofError> {
    if value.layout.element.bytes == 0 {
        return Err(CompiledProofError::InvalidValue { value: value.id });
    }
    let mut tags = BTreeSet::new();
    let mut expected_stride = value.layout.element.bytes;
    for axis in &value.layout.axes {
        if axis.extent == 0
            || axis.stride_bytes == 0
            || axis.stride_bytes % value.layout.element.bytes != 0
            || axis.stride_bytes != expected_stride
            || !tags.insert(axis.tag)
        {
            return Err(CompiledProofError::InvalidValue { value: value.id });
        }
        expected_stride = expected_stride
            .checked_mul(axis.extent)
            .ok_or(CompiledProofError::SizeOverflow)?;
    }
    let logical_bytes = value
        .layout
        .logical_bytes()
        .map_err(|_| CompiledProofError::InvalidValue { value: value.id })?;
    if expected_stride != logical_bytes {
        return Err(CompiledProofError::InvalidValue { value: value.id });
    }
    Ok(logical_bytes)
}

fn reject_overlaps<'a>(
    kind: BindingKind,
    bindings: impl Iterator<Item = (ValueId, &'a Range<usize>)>,
) -> Result<(), CompiledProofError> {
    let bindings = bindings.collect::<Vec<_>>();
    for (index, (left_value, left)) in bindings.iter().enumerate() {
        for (right_value, right) in &bindings[index + 1..] {
            if left_value == right_value && left.start < right.end && right.start < left.end {
                return Err(CompiledProofError::BindingOverlap {
                    kind,
                    value: *left_value,
                });
            }
        }
    }
    Ok(())
}

fn stage_index(stage: ProofStage, transcript: &CairoBlake2sTranscriptPlan) -> Option<usize> {
    match stage {
        ProofStage::BeforeTranscript(segment) => transcript
            .segments()
            .iter()
            .position(|candidate| candidate.segment == segment),
        ProofStage::AfterTranscript => Some(transcript.segments().len()),
    }
}

fn transcript_input_stage(
    transcript: &CairoBlake2sTranscriptPlan,
    id: TranscriptInputId,
) -> Option<usize> {
    transcript_operation_stage(transcript, |operation| {
        operation_input(*operation) == Some(id)
    })
}

fn transcript_output_stage(
    transcript: &CairoBlake2sTranscriptPlan,
    id: TranscriptOutputId,
) -> Option<usize> {
    transcript_operation_stage(transcript, |operation| {
        operation_output(*operation) == Some(id)
    })
}

fn transcript_operation_stage(
    transcript: &CairoBlake2sTranscriptPlan,
    matches: impl Fn(&TranscriptOperation) -> bool,
) -> Option<usize> {
    let operation = transcript
        .schedule()
        .operations()
        .iter()
        .position(matches)?;
    transcript
        .segments()
        .iter()
        .position(|segment| segment.operation_range.contains(&operation))
}

fn operation_input(operation: TranscriptOperation) -> Option<TranscriptInputId> {
    match operation {
        TranscriptOperation::MixFelts { source, .. }
        | TranscriptOperation::MixU32s { source, .. }
        | TranscriptOperation::MixU64 { source, .. }
        | TranscriptOperation::AbsorbRoot { source, .. }
        | TranscriptOperation::AbsorbPowNonce { source, .. } => Some(source),
        TranscriptOperation::DrawSecureFelt { .. }
        | TranscriptOperation::DrawSecureFelts { .. }
        | TranscriptOperation::DrawU32s { .. }
        | TranscriptOperation::DrawQueries { .. } => None,
    }
}

fn operation_output(operation: TranscriptOperation) -> Option<TranscriptOutputId> {
    match operation {
        TranscriptOperation::DrawSecureFelt { output, .. }
        | TranscriptOperation::DrawSecureFelts { output, .. }
        | TranscriptOperation::DrawU32s { output, .. }
        | TranscriptOperation::DrawQueries { output, .. } => Some(output),
        TranscriptOperation::MixFelts { .. }
        | TranscriptOperation::MixU32s { .. }
        | TranscriptOperation::MixU64 { .. }
        | TranscriptOperation::AbsorbRoot { .. }
        | TranscriptOperation::AbsorbPowNonce { .. } => None,
    }
}

fn layout_ranges(layout: &ResidentProofBundleLayout) -> [Range<usize>; 8] {
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
