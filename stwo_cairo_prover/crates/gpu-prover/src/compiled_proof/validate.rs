use std::collections::BTreeSet;

use stwo_backend_cuda::TranscriptOperation;

use super::*;

mod finalizer;
mod output;
mod primitive;
mod structural_authority;
mod transcript_segments;

pub(super) fn validate(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), CompiledProofError> {
    validate_values(input)?;
    structural_authority::validate(input)?;
    validate_authorities(input)?;
    validate_operations(input, transcript)?;
    validate_transcript(input, transcript)?;
    transcript_segments::validate(input, transcript)?;
    output::validate(input)?;
    finalizer::validate(input, transcript)
}

fn validate_values(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    let mut external_inputs = BTreeSet::new();
    let mut constants = BTreeSet::new();
    let mut transcript_outputs = BTreeSet::new();
    for (index, value) in input.values.iter().enumerate() {
        let expected =
            ValueVersion(u32::try_from(index).map_err(|_| CompiledProofError::SizeOverflow)?);
        if value.version != expected {
            return Err(CompiledProofError::NonDenseValue {
                expected,
                actual: value.version,
            });
        }
        validate_value_layout(value)?;
        if value.alignment == 0
            || !value.alignment.is_power_of_two()
            || !origin_region_compatible(value.origin, value.region)
            || match value.origin {
                ValueOrigin::ExternalInput(id) => !external_inputs.insert(id),
                ValueOrigin::Constant(id) => !constants.insert(id),
                ValueOrigin::TranscriptOutput(id) => !transcript_outputs.insert(id),
                ValueOrigin::OpOutput(_) => false,
            }
        {
            return Err(CompiledProofError::InvalidValue {
                value: value.version,
            });
        }
        if let ValueOrigin::OpOutput(producer) = value.origin {
            if input
                .operations
                .get(producer.0 as usize)
                .is_none_or(|operation| operation.id != producer)
            {
                return Err(CompiledProofError::UnknownProducer {
                    value: value.version,
                    producer,
                });
            }
        }
    }
    Ok(())
}

fn origin_region_compatible(origin: ValueOrigin, region: Region) -> bool {
    match origin {
        ValueOrigin::ExternalInput(_) => region == Region::Input,
        ValueOrigin::Constant(_) => region == Region::FixedData,
        ValueOrigin::TranscriptOutput(_) => region == Region::Dynamic,
        ValueOrigin::OpOutput(_) => matches!(region, Region::Dynamic | Region::Output),
    }
}

fn validate_authorities(input: &CompiledProofInput) -> Result<(), CompiledProofError> {
    let mut semantics = BTreeSet::new();
    for operation in &input.operations {
        if operation.semantic_id.0 == 0 || !semantics.insert(operation.semantic_id) {
            return Err(CompiledProofError::DuplicateSemanticOperation(
                operation.semantic_id,
            ));
        }
    }

    if input
        .effects
        .windows(2)
        .any(|pair| pair[0].id() >= pair[1].id())
    {
        return Err(CompiledProofError::NonCanonicalEffectAuthority);
    }
    for effect in &input.effects {
        if !effect.has_valid_identity()? {
            return Err(CompiledProofError::InvalidEffectContract(effect.id()));
        }
    }

    if input
        .kernels
        .windows(2)
        .any(|pair| pair[0].id() >= pair[1].id())
    {
        return Err(CompiledProofError::NonCanonicalKernelAuthority);
    }
    for kernel in &input.kernels {
        if !kernel.has_valid_identity()? {
            return Err(CompiledProofError::NonCanonicalKernelAuthority);
        }
        let mut used = BTreeSet::new();
        for operation in &input.operations {
            if matches!(
                operation.primitive,
                ExecutionPrimitive::AotKernel { kernel: id, .. } if id == kernel.id()
            ) {
                if kernel
                    .accepted_effects()
                    .binary_search(&operation.effect)
                    .is_err()
                {
                    return Err(CompiledProofError::KernelEffectNotAccepted {
                        operation: operation.id,
                    });
                }
                used.insert(operation.effect);
            }
        }
        if kernel.accepted_effects().iter().copied().ne(used) {
            return Err(CompiledProofError::NonCanonicalKernelEffects(kernel.id()));
        }
    }

    let declared_effects = input
        .effects
        .iter()
        .map(EffectContract::id)
        .collect::<Vec<_>>();
    let used_effects = input
        .operations
        .iter()
        .map(|operation| operation.effect)
        .collect::<BTreeSet<_>>();
    if declared_effects
        .iter()
        .copied()
        .ne(used_effects.iter().copied())
    {
        return Err(CompiledProofError::NonCanonicalEffectAuthority);
    }

    let declared_kernels = input
        .kernels
        .iter()
        .map(AotKernelAuthority::id)
        .collect::<Vec<_>>();
    let used_kernels = input
        .operations
        .iter()
        .filter_map(|operation| match operation.primitive {
            ExecutionPrimitive::AotKernel { kernel, .. } => Some(kernel),
            ExecutionPrimitive::DeviceCopyD2D { .. }
            | ExecutionPrimitive::DeviceMemsetByte { .. } => None,
        })
        .collect::<BTreeSet<_>>();
    if declared_kernels
        .iter()
        .copied()
        .ne(used_kernels.iter().copied())
    {
        return Err(CompiledProofError::NonCanonicalKernelAuthority);
    }
    Ok(())
}

fn validate_operations(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), CompiledProofError> {
    let mut previous_stage = None;
    let mut writes = vec![Vec::<(OpId, ElementRange)>::new(); input.values.len()];
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

        let effect = effect(input, operation.effect).ok_or(CompiledProofError::UnknownEffect {
            operation: operation.id,
        })?;
        primitive::validate(input, operation, effect)?;
        for access in effect.accesses() {
            if let Some(source) = access.source() {
                validate_bound_range(input, operation.id, *source)?;
                validate_source(input, transcript, operation, source.value.version)?;
            }
            if let Some(destination) = access.destination() {
                validate_bound_range(input, operation.id, *destination)?;
                let value = value(input, destination.value.version)?;
                if value.origin != ValueOrigin::OpOutput(operation.id) {
                    return Err(CompiledProofError::ProducerMismatch {
                        value: value.version,
                    });
                }
                writes[value.version.0 as usize].push((operation.id, destination.value.elements));
            }
            if let (Some(source), Some(destination)) = (access.source(), access.destination()) {
                let source_value = value(input, source.value.version)?;
                let destination_value = value(input, destination.value.version)?;
                if source_value.layout != destination_value.layout
                    || source_value.layout.element != destination_value.layout.element
                    || matches!(
                        access,
                        EffectAccess::Atomic {
                            operation: AtomicOperation::AddU32,
                            ..
                        }
                    ) && source_value.layout.element != ElementType::U32
                {
                    return Err(CompiledProofError::InvalidValueTransition);
                }
            }
        }
    }
    validate_write_coverage(input, &mut writes)
}

fn validate_source(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
    consumer: &OpNode,
    version: ValueVersion,
) -> Result<(), CompiledProofError> {
    match value(input, version)?.origin {
        ValueOrigin::ExternalInput(_) | ValueOrigin::Constant(_) => Ok(()),
        ValueOrigin::OpOutput(producer) if producer < consumer.id => Ok(()),
        ValueOrigin::OpOutput(_) => Err(CompiledProofError::ProducerAfterConsumer {
            value: version,
            consumer: consumer.id,
        }),
        ValueOrigin::TranscriptOutput(output) => {
            let drawn = transcript_output_stage(transcript, output).ok_or(
                CompiledProofError::TranscriptCausality {
                    kind: BindingKind::TranscriptOutput,
                    id: output.0,
                },
            )?;
            let consumed = stage_index(consumer.stage, transcript).ok_or(
                CompiledProofError::UnknownStage {
                    operation: consumer.id,
                },
            )?;
            if consumed <= drawn {
                return Err(CompiledProofError::TranscriptCausality {
                    kind: BindingKind::TranscriptOutput,
                    id: output.0,
                });
            }
            Ok(())
        }
    }
}

fn validate_write_coverage(
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
        if cursor != value.layout.element_count()? {
            return Err(CompiledProofError::IncompleteWrite {
                value: value.version,
            });
        }
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
        let source = value(input, binding.value)?;
        validate_words(
            BindingKind::TranscriptInput,
            binding.id.0,
            source,
            binding.elements,
            requirement.min_words,
        )?;
        let consumed = transcript_input_stage(transcript, binding.id).ok_or(
            CompiledProofError::TranscriptCausality {
                kind: BindingKind::TranscriptInput,
                id: binding.id.0,
            },
        )?;
        match source.origin {
            ValueOrigin::OpOutput(producer) => {
                let produced = stage_index(input.operations[producer.0 as usize].stage, transcript)
                    .ok_or(CompiledProofError::UnknownStage {
                        operation: producer,
                    })?;
                if produced > consumed {
                    return Err(CompiledProofError::TranscriptCausality {
                        kind: BindingKind::TranscriptInput,
                        id: binding.id.0,
                    });
                }
            }
            ValueOrigin::TranscriptOutput(output) => {
                if transcript_output_stage(transcript, output).is_none_or(|drawn| drawn >= consumed)
                {
                    return Err(CompiledProofError::TranscriptCausality {
                        kind: BindingKind::TranscriptInput,
                        id: binding.id.0,
                    });
                }
            }
            ValueOrigin::ExternalInput(_) | ValueOrigin::Constant(_) => {}
        }
    }
    reject_overlaps(
        BindingKind::TranscriptInput,
        input
            .transcript_inputs
            .iter()
            .map(|binding| (binding.value, binding.elements)),
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
        let output = value(input, binding.value)?;
        validate_words(
            BindingKind::TranscriptOutput,
            binding.id.0,
            output,
            binding.elements,
            requirement.min_words,
        )?;
        if output.origin != ValueOrigin::TranscriptOutput(binding.id)
            || binding.elements != ElementRange::new(0, output.layout.element_count()?).unwrap()
        {
            return Err(CompiledProofError::TranscriptBindingOrigin { output: binding.id });
        }
        let drawn = transcript_output_stage(transcript, binding.id).ok_or(
            CompiledProofError::TranscriptCausality {
                kind: BindingKind::TranscriptOutput,
                id: binding.id.0,
            },
        )?;
        if consumers(input, output.version).any(|operation| {
            stage_index(operation.stage, transcript).is_none_or(|stage| stage <= drawn)
        }) {
            return Err(CompiledProofError::TranscriptCausality {
                kind: BindingKind::TranscriptOutput,
                id: binding.id.0,
            });
        }
    }
    for value in &input.values {
        if let ValueOrigin::TranscriptOutput(id) = value.origin {
            let count = input
                .transcript_outputs
                .iter()
                .filter(|binding| binding.id == id && binding.value == value.version)
                .count();
            if count != 1 {
                return Err(CompiledProofError::OrphanTranscriptOutput {
                    value: value.version,
                });
            }
        }
    }
    reject_overlaps(
        BindingKind::TranscriptOutput,
        input
            .transcript_outputs
            .iter()
            .map(|binding| (binding.value, binding.elements)),
    )
}

fn consumers<'a>(
    input: &'a CompiledProofInput,
    version: ValueVersion,
) -> impl Iterator<Item = &'a OpNode> {
    input.operations.iter().filter(move |operation| {
        effect(input, operation.effect).is_some_and(|effect| {
            effect.accesses().iter().any(|access| {
                access
                    .source()
                    .is_some_and(|source| source.value.version == version)
            })
        })
    })
}

fn effect(input: &CompiledProofInput, id: EffectContractId) -> Option<&EffectContract> {
    input.effects.iter().find(|effect| effect.id() == id)
}

pub(super) fn value(
    input: &CompiledProofInput,
    version: ValueVersion,
) -> Result<&ValueDesc, CompiledProofError> {
    input
        .values
        .get(version.0 as usize)
        .filter(|value| value.version == version)
        .ok_or(CompiledProofError::InvalidValue { value: version })
}

fn validate_bound_range(
    input: &CompiledProofInput,
    operation: OpId,
    range: BoundValueRange,
) -> Result<(), CompiledProofError> {
    let value = input
        .values
        .get(range.value.version.0 as usize)
        .filter(|value| value.version == range.value.version)
        .ok_or(CompiledProofError::UnknownValue {
            operation,
            value: range.value.version,
        })?;
    if range.value.elements.is_empty() || range.value.elements.end > value.layout.element_count()? {
        return Err(CompiledProofError::InvalidEffectRange);
    }
    Ok(())
}

fn range_bytes(input: &CompiledProofInput, range: ValueRange) -> Result<usize, CompiledProofError> {
    range
        .elements
        .len()
        .checked_mul(value(input, range.version)?.layout.element.bytes)
        .ok_or(CompiledProofError::SizeOverflow)
}

pub(super) fn validate_words(
    kind: BindingKind,
    id: u32,
    value: &ValueDesc,
    elements: ElementRange,
    expected_words: usize,
) -> Result<(), CompiledProofError> {
    if value.layout.element != ElementType::U32
        || value.alignment < core::mem::align_of::<u32>()
        || elements.is_empty()
        || elements.end > value.layout.element_count()?
        || elements.len() != expected_words
    {
        return Err(CompiledProofError::BindingRange { kind, id });
    }
    Ok(())
}

fn validate_value_layout(value: &ValueDesc) -> Result<usize, CompiledProofError> {
    if value.layout.element.bytes == 0 {
        return Err(CompiledProofError::InvalidValue {
            value: value.version,
        });
    }
    let mut tags = BTreeSet::new();
    let mut expected_stride = value.layout.element.bytes;
    for axis in &value.layout.axes {
        if axis.extent == 0 || axis.stride_bytes != expected_stride || !tags.insert(axis.tag) {
            return Err(CompiledProofError::InvalidValue {
                value: value.version,
            });
        }
        expected_stride = expected_stride
            .checked_mul(axis.extent)
            .ok_or(CompiledProofError::SizeOverflow)?;
    }
    if expected_stride != value.layout.logical_bytes()? {
        return Err(CompiledProofError::InvalidValue {
            value: value.version,
        });
    }
    Ok(expected_stride)
}

pub(super) fn reject_overlaps(
    kind: BindingKind,
    bindings: impl Iterator<Item = (ValueVersion, ElementRange)>,
) -> Result<(), CompiledProofError> {
    let bindings = bindings.collect::<Vec<_>>();
    for (index, (left_value, left)) in bindings.iter().enumerate() {
        for (right_value, right) in &bindings[index + 1..] {
            if left_value == right_value && left.overlaps(*right) {
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
